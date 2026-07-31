#!/usr/bin/env python3
"""
run_null_calibration.py

Unattended, resumable driver for the null-calibration replicate loop. It
composes the two existing scripts --

    scripts/simulate_null_persite.py   (batched AliSim simulation)
    scripts/score_null_replicate.py    (ASR + scoring + key-check + .npz)

-- without changing either. Nothing here reimplements simulation or
scoring logic; this file is orchestration only (batching, bounded
concurrency, resumability, cleanup, and a progress log).

Why Python and not a shell script
-----------------------------------
The three requirements that matter most here -- bounded concurrency for the
scoring stage, structured validity checks on .npz files, and per-replicate
failure isolation with a machine-readable progress log -- are all native to
Python (``concurrent.futures``, ``numpy.load`` inside ``try/except``,
``json``) and comparatively fragile in POSIX shell/zsh (background-job
counting, no structured exceptions, and zsh's non-word-splitting behaviour
on unquoted expansions makes array/argument handling error-prone). The
actual heavy lifting is still two subprocess calls per unit of work, exactly
as a shell driver would do it -- this is orchestration, not computation.

Batching model
---------------
AliSim's expensive one-time cost per invocation (the 2M-site reference-
frequency probe plus the per-column --mdef/-q files) is paid once per call
to simulate_null_persite.py, not once per alignment (see that script's
--num-alignments option and this repo's config/snakemake.yaml note on
measured cost). Replicates are therefore grouped into batches of
--batch-size; one simulate_null_persite.py invocation with
``--num-alignments <batch size>`` produces every alignment needed for a
batch in one call. AliSim's own naming for a batch of N is
``<prefix>_1.fa .. <prefix>_N.fa`` (1-indexed, verified empirically against
IQ-TREE 2.3.6), so replicate ``i`` within batch ``[start, end)`` maps to
local position ``i - start + 1``.

Each batch gets its own deterministic seed, ``base_seed + batch_index``,
documented here rather than left as an unexplained per-call random draw:
every replicate produced from a given ``(base_seed, batch_index)`` pair is
therefore reproducible from the CLI arguments alone.

Resumability
-------------
A replicate is considered done if ``<out-dir>/npz/rep_<i>.npz`` exists,
loads without error, and contains every array score_null_replicate.py's
own ``write_npz`` writes (see ``is_valid_npz``). Before touching a batch at
all, every replicate in it is checked; a batch already fully done is
skipped entirely (no simulate call, no scoring). A batch with a mix of
done/not-done replicates is re-simulated (AliSim is deterministic for a
given seed + count, so regenerating already-done alignments is harmless)
but only the not-done replicates are scored; the freshly (re)generated
alignment for an already-done replicate is discarded immediately without
scoring.

Cleanup
-------
score_null_replicate.py deletes its own scratch workdir (containing the
~600 MB .state file) only on its own successful return; if it raises
partway through (a failed ASR call, or the key-check assertion), that
workdir is left behind. Since this driver assigns and owns a distinct
--workdir per replicate, it removes that directory itself after every
subprocess call, success or failure, so no .state file is ever left behind
by an interrupted or failed replicate. Each batch's simulated alignments
are deleted individually once their replicate has been scored (whether
that scoring succeeded or failed), and the batch's scratch directory
(the .mdef.nex/.parts files and any leftover files) is removed once every
replicate in the batch has been attempted.

Failure handling
-----------------
A single replicate's simulate- or score-stage failure is caught, recorded
in the progress log with the stage and a truncated error, and does not
stop the run. The process exits non-zero at the end if any replicate
failed, so an unattended overnight run can be checked with a plain exit
code the next morning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "snakemake.yaml"
SIMULATE_SCRIPT = REPO_ROOT / "scripts" / "simulate_null_persite.py"
SCORE_SCRIPT = REPO_ROOT / "scripts" / "score_null_replicate.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def subprocess_env() -> dict:
    """Environment for the simulate/score subprocess calls.

    score_null_replicate.py internally shells out to ``python -m
    badasp.scoring`` (see its score_replicate()), which resolves the
    ``badasp`` package only if ``src/`` is on PYTHONPATH -- confirmed by
    running that exact import with PYTHONPATH unset (fails) vs
    PYTHONPATH=src (succeeds). Rather than rely on the caller having
    exported that before starting an unattended overnight run, this driver
    sets it explicitly for every subprocess it launches, prepended to
    whatever PYTHONPATH (if any) is already inherited.
    """
    env = os.environ.copy()
    src_dir = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([src_dir, str(REPO_ROOT), existing]) if existing \
        else os.pathsep.join([src_dir, str(REPO_ROOT)])
    return env

# The canonical list of arrays score_null_replicate.py's write_npz() persists.
# Imported (not re-typed) so this driver's idea of "a valid .npz" can never
# drift from what that script actually writes.
from scripts.score_null_replicate import SCORE_COLUMNS  # noqa: E402

_KEY_CHECK_RE = re.compile(
    r"\[keys\]\s+observed=([\d,]+)\s+null=([\d,]+)\s+"
    r"missing_from_null=([\d,]+)\s+\(([\d.]+)%\)\s+null_only_dropped=([\d,]+)"
)
_CLADE_CHECK_RE = re.compile(r"\[keys\]\s+clade sizes identical on all ([\d,]+) shared tests")
_ASR_TIME_RE = re.compile(r"\[asr\]\s+([\d.]+)s")
_SCORE_TIME_RE = re.compile(r"\[score\]\s+([\d.]+)s")


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def parse_key_check(stdout: str) -> dict:
    """Extract the key-check numbers score_null_replicate.py prints to stdout."""
    out: dict = {
        "observed_tests": None,
        "null_tests": None,
        "missing_from_null": None,
        "missing_frac_pct": None,
        "null_only_dropped": None,
        "clade_sizes_identical": None,
        "asr_seconds": None,
        "score_seconds": None,
    }
    m = _KEY_CHECK_RE.search(stdout)
    if m:
        out["observed_tests"] = int(m.group(1).replace(",", ""))
        out["null_tests"] = int(m.group(2).replace(",", ""))
        out["missing_from_null"] = int(m.group(3).replace(",", ""))
        out["missing_frac_pct"] = float(m.group(4))
        out["null_only_dropped"] = int(m.group(5).replace(",", ""))
    cm = _CLADE_CHECK_RE.search(stdout)
    if cm:
        out["clade_sizes_identical"] = int(cm.group(1).replace(",", ""))
    am = _ASR_TIME_RE.search(stdout)
    if am:
        out["asr_seconds"] = float(am.group(1))
    sm = _SCORE_TIME_RE.search(stdout)
    if sm:
        out["score_seconds"] = float(sm.group(1))
    return out


def is_valid_npz(path: Path, expected_len: Optional[int] = None) -> bool:
    """A replicate is "done" iff its .npz exists, loads, and has every
    SCORE_COLUMNS array present with a consistent (and, if known, expected)
    length. Any exception (truncated file, bad zip, missing arrays) counts
    as invalid, so a corrupted .npz is simply redone.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            if any(c not in data.files for c in SCORE_COLUMNS):
                return False
            lengths = {int(data[c].shape[0]) for c in SCORE_COLUMNS}
            if len(lengths) != 1:
                return False
            n = lengths.pop()
            if expected_len is not None and n != expected_len:
                return False
    except Exception:
        return False
    return True


def count_csv_data_rows(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for _ in fh) - 1


def file_info(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def git_sha(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def alisim_output_path(prefix: Path, pos: int, batch_size: int) -> Path:
    """AliSim's own output naming for --num-alignments N: verified empirically
    against IQ-TREE 2.3.6 that N==1 writes ``<prefix>.fa`` (no numeric
    suffix), while N>1 writes ``<prefix>_1.fa .. <prefix>_N.fa`` (plain,
    unpadded 1-based indices). ``pos`` is the alignment's 1-based position
    within its batch.
    """
    if batch_size == 1:
        return Path(f"{prefix}.fa")
    return Path(f"{prefix}_{pos}.fa")


def batch_ranges(num_replicates: int, batch_size: int) -> list:
    ranges = []
    start = 0
    while start < num_replicates:
        end = min(start + batch_size, num_replicates)
        ranges.append((start, end))
        start = end
    return ranges


def append_progress(log_path: Path, record: dict) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(log_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def score_one(
    i: int,
    seed: int,
    batch_idx: int,
    fasta: Path,
    npz_path: Path,
    work: Path,
    args: argparse.Namespace,
) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCORE_SCRIPT),
        "--sim-alignment", str(fasta),
        "--reconciled-tree", str(args.reconciled_tree),
        "--asr-tree", str(args.asr_tree),
        "--observed-scores", str(args.observed_scores),
        "--out-npz", str(npz_path),
        "--model", args.asr_model,
        "--min-clade", str(args.min_clade),
        "--node-naming", args.node_naming,
        "--threads", str(args.threads_per_replicate),
        "--iqtree", args.iqtree_bin,
        "--workdir", str(work),
        "--max-missing-frac", str(args.max_missing_frac),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env())
    elapsed = time.time() - t0

    # This driver owns `work` (a per-replicate --workdir), so it is
    # responsible for removing it whether score_null_replicate.py succeeded
    # or raised partway through -- that script only cleans up on success.
    shutil.rmtree(work, ignore_errors=True)
    # Delete the simulated alignment once this replicate has been attempted,
    # success or failure: it is reproducible from (seed, batch_idx) alone.
    fasta.unlink(missing_ok=True)

    record = {
        "replicate": i,
        "batch": batch_idx,
        "seed": seed,
        "status": "OK" if result.returncode == 0 else "FAILED",
        "wall_clock_s": round(elapsed, 1),
        "stage_failed": None if result.returncode == 0 else "score",
        **parse_key_check(result.stdout),
    }
    if result.returncode != 0:
        tail = (result.stdout[-1500:] + "\n" + result.stderr[-1500:]).replace("\n", " | ")
        record["error"] = tail
    return record


def process_batch(
    batch_idx: int,
    start: int,
    end: int,
    args: argparse.Namespace,
    expected_len: Optional[int],
    width: int,
    npz_dir: Path,
    scratch_dir: Path,
    progress_log: Path,
) -> list:
    indices = list(range(start, end))
    npz_paths = {i: npz_dir / f"rep_{i:0{width}d}.npz" for i in indices}
    needing = [i for i in indices if args.force or not is_valid_npz(npz_paths[i], expected_len)]

    if not needing:
        print(f"[batch {batch_idx}] all {len(indices)} replicate(s) already done, skipping")
        return []

    batch_seed = args.seed + batch_idx
    batch_dir = scratch_dir / f"batch_{batch_idx:0{width}d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    sim_prefix = batch_dir / "sim"

    cmd = [
        sys.executable, str(SIMULATE_SCRIPT),
        "--config", str(args.config),
        "--composition-alignment", str(args.composition_alignment),
        "--sim-tree", str(args.sim_tree),
        "--out-prefix", str(sim_prefix),
        "--shrinkage", str(args.shrinkage),
        "--num-alignments", str(len(indices)),
        "--seed", str(batch_seed),
        "--threads", str(args.sim_threads),
        "--reference-mc-length", str(args.reference_mc_length),
        "--iqtree-bin", args.iqtree_bin,
        "--redo",
    ]
    if args.gap_mask_source is not None:
        cmd += ["--gap-mask-source", str(args.gap_mask_source)]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env())
    sim_elapsed = time.time() - t0

    if result.returncode != 0:
        tail = (result.stdout[-1500:] + "\n" + result.stderr[-1500:]).replace("\n", " | ")
        print(f"[batch {batch_idx}] simulate FAILED after {sim_elapsed:.1f}s", file=sys.stderr)
        for i in needing:
            append_progress(progress_log, {
                "replicate": i, "batch": batch_idx, "seed": batch_seed,
                "status": "FAILED", "wall_clock_s": round(sim_elapsed, 1),
                "stage_failed": "simulate", "error": tail,
            })
        shutil.rmtree(batch_dir, ignore_errors=True)
        return list(needing)

    print(f"[batch {batch_idx}] simulate: {sim_elapsed:.1f}s for {len(indices)} alignment(s)")

    failed: list = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {}
        for i in indices:
            pos = i - start + 1
            fasta = alisim_output_path(sim_prefix, pos, len(indices))
            if i not in needing:
                # Already done; discard the (deterministically identical)
                # regenerated alignment without scoring it again.
                fasta.unlink(missing_ok=True)
                continue
            work = scratch_dir / f"rep_{i:0{width}d}_work"
            fut = pool.submit(score_one, i, batch_seed, batch_idx, fasta, npz_paths[i], work, args)
            futures[fut] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # driver-side bug, not a scorer failure
                record = {
                    "replicate": i, "batch": batch_idx, "seed": batch_seed,
                    "status": "FAILED", "wall_clock_s": None,
                    "stage_failed": "driver", "error": repr(exc),
                }
            append_progress(progress_log, record)
            status = record["status"]
            print(f"[batch {batch_idx}] replicate {i}: {status} "
                  f"({record.get('wall_clock_s')}s)")
            if status != "OK":
                failed.append(i)

    shutil.rmtree(batch_dir, ignore_errors=True)
    return failed


def parse_args(argv=None) -> argparse.Namespace:
    # Pre-parse only --config, with add_help=False, so --help on the real
    # parser (below) can show every option instead of exiting early on a
    # partial parser -- see simulate_null_persite.py's docstring for the bug
    # this pattern avoids.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_config(pre_args.config)
    nc = cfg.get("null_calibration", {})
    paths = cfg.get("paths", {})
    tools = cfg.get("tools", {})

    parser = argparse.ArgumentParser(
        description="Unattended, resumable driver for the null-calibration "
                    "replicate loop (batched simulation + bounded-concurrency "
                    "scoring via the existing simulate_null_persite.py / "
                    "score_null_replicate.py scripts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to config/snakemake.yaml.")
    parser.add_argument("--num-replicates", type=int, default=nc.get("replicates", 1000),
                        help="Total number of null replicates to produce.")
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Replicates simulated per simulate_null_persite.py "
                             "invocation (amortizes its one-time setup cost).")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Bounded concurrency for the scoring stage. Each "
                             "replicate's ASR uses --threads-per-replicate "
                             "threads and ~3.3 GB; size this so "
                             "concurrency * threads-per-replicate stays within "
                             "the machine's cores and concurrency * ~3.3 GB "
                             "stays within available RAM (with headroom).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory: <out-dir>/npz/rep_*.npz "
                             "(consumed by calibrate_switch_threshold.py), "
                             "<out-dir>/logs/progress.jsonl, "
                             "<out-dir>/scratch/ (batch scratch, removed as it goes), "
                             "<out-dir>/run_manifest.json.")
    parser.add_argument("--seed", type=int, default=nc.get("seed"),
                        help="Base seed. Batch b's AliSim seed is "
                             "base_seed + b (documented, not looked up "
                             "per-replicate at random).")
    parser.add_argument("--composition-alignment", type=Path,
                        default=REPO_ROOT / paths.get("trimmed_fasta", "data/interim/IPR019888_trimmed.aln"),
                        help="Real alignment for per-column null composition "
                             "(passed through to simulate_null_persite.py).")
    parser.add_argument("--sim-tree", type=Path,
                        default=REPO_ROOT / nc.get("sim_tree", "data/interim/iqtree_asr/IPR019888.treefile"),
                        help="Tree to simulate along (passed through to "
                             "simulate_null_persite.py).")
    parser.add_argument("--reconciled-tree", type=Path, required=True,
                        help="ASR topology (-te) -- pipeline output, no default "
                             "assumed since it may be regenerated.")
    parser.add_argument("--asr-tree", type=Path, required=True,
                        help="Rooted/mapped scoring tree -- pipeline output, "
                             "no default assumed since it may be regenerated.")
    parser.add_argument("--observed-scores", type=Path, required=True,
                        help="Observed score CSV defining the canonical test "
                             "keys/row order -- pipeline output, no default "
                             "assumed since it may be regenerated.")
    parser.add_argument("--shrinkage", type=float, default=nc.get("shrinkage", 0.15),
                        help="Passed through to simulate_null_persite.py.")
    parser.add_argument("--asr-model", default=nc.get("asr_model", "LG+G"),
                        help="Passed through to score_null_replicate.py's --model; "
                             "must match the real run's ASR model.")
    parser.add_argument("--min-clade", type=int, default=5,
                        help="Passed through to score_null_replicate.py.")
    parser.add_argument("--node-naming", choices=["legacy", "strict"], default="strict",
                        help="Passed through to score_null_replicate.py.")
    parser.add_argument("--threads-per-replicate", type=int, default=2,
                        help="-T threads for each replicate's ASR/scoring subprocess.")
    parser.add_argument("--sim-threads", type=int, default=1,
                        help="-T threads for the batched AliSim call.")
    parser.add_argument("--reference-mc-length", type=int, default=2_000_000,
                        help="Passed through to simulate_null_persite.py's "
                             "--reference-mc-length (lower for small/test runs).")
    parser.add_argument("--iqtree-bin", default=tools.get("iqtree", "iqtree2"),
                        help="IQ-TREE executable.")
    parser.add_argument("--max-missing-frac", type=float, default=0.01,
                        help="Passed through to score_null_replicate.py.")
    parser.add_argument("--gap-mask-source", type=Path, default=None,
                        help="Passed through to simulate_null_persite.py "
                             "(default there: same as --composition-alignment).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing valid .npz files and redo every "
                             "replicate anyway.")
    args = parser.parse_args(argv)
    if args.seed is None:
        parser.error("--seed has no config default (null_calibration.seed missing); pass it explicitly.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    npz_dir = args.out_dir / "npz"
    scratch_dir = args.out_dir / "scratch"
    logs_dir = args.out_dir / "logs"
    for d in (npz_dir, scratch_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    progress_log = logs_dir / "progress.jsonl"

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(REPO_ROOT),
        "num_replicates": args.num_replicates,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "shrinkage": args.shrinkage,
        "asr_model": args.asr_model,
        "min_clade": args.min_clade,
        "node_naming": args.node_naming,
        "inputs": {
            "composition_alignment": file_info(args.composition_alignment),
            "sim_tree": file_info(args.sim_tree),
            "reconciled_tree": file_info(args.reconciled_tree),
            "asr_tree": file_info(args.asr_tree),
            "observed_scores": file_info(args.observed_scores),
        },
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    expected_len = count_csv_data_rows(args.observed_scores)
    width = max(4, len(str(max(args.num_replicates - 1, 0))))
    ranges = batch_ranges(args.num_replicates, args.batch_size)

    all_failed: list = []
    for batch_idx, (start, end) in enumerate(ranges):
        failed = process_batch(
            batch_idx, start, end, args, expected_len, width,
            npz_dir, scratch_dir, progress_log,
        )
        all_failed.extend(failed)

    n_ok = args.num_replicates - len(all_failed)
    print(f"Done: {n_ok}/{args.num_replicates} replicate(s) OK, {len(all_failed)} failed.")
    if all_failed:
        print(f"Failed replicate indices: {sorted(all_failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
