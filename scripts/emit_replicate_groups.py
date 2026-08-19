#!/usr/bin/env python3
"""
emit_replicate_groups.py

Produces the two-column ``rep_file,invocation`` CSV that
``scripts/calibrate_switch_threshold.py --replicate-groups-csv`` expects:
for every ``rep_*.npz`` replicate under a run directory, which
``simulate()`` invocation (AliSim call) actually produced it.

``calibrate_switch_threshold.py`` deliberately refuses to infer this mapping
itself for the "index arithmetic" case (``invocation = rep_index //
alignments_per_invocation``), because that rule is recorded only in an
external sbatch script (``scripts/euler/run_score_null_calibration_array.sh``)
and not in the replicate ``.npz`` files -- inferring it inside the
calibration code would invent provenance rather than recover it. This script
is the explicit, auditable place where that inference happens instead, one
run directory at a time, with the chosen route (and the arithmetic constant,
when used) written into a sidecar file next to the output CSV.

Resolution routes, tried in order, stopping at the first that applies to a
given (sub-)run directory:

1. ``<run-dir>/run_manifest.json`` (written by ``scripts/merge_null_runs.py``)
   with a ``replicates`` list of ``{"merged_index": i, "source": path}``
   entries: group the replicates by the run directory their ``source`` came
   from, and recurse into each such source run directory (trying routes
   1-4 again) to resolve *its* labels. Each recursed label is prefixed with
   its source run directory's own name (e.g. ``euler_run:0``), so replicates
   pooled from different runs can never collide on a bare label even if
   those runs happened to use the same convention.
2. ``<run-dir>/provenance.json`` (written by
   ``scripts/stage_null_replicate_set.py``) with a ``replicates`` list of
   ``{"rep_id": ..., "invocation": ...}`` entries: use the recorded
   invocation directly.
3. ``<run-dir>/logs/progress.jsonl`` (written by
   ``scripts/run_null_calibration.py``), one JSON object per line with
   ``{"replicate": i, "batch": b, ...}`` fields: use ``batch``.
4. Index arithmetic: ``invocation = rep_index // alignments_per_invocation``.
   This is the convention documented at
   ``scripts/euler/run_score_null_calibration_array.sh:78-79``
   (``CHUNK=$(( IDX / 10 ))``), so ``--alignments-per-invocation`` must be
   supplied explicitly by the caller with the value that applies to *this*
   run directory -- it is never assumed or defaulted, since the value lives
   only in that external script and reusing "10" silently here would be
   exactly the kind of inferred provenance this script exists to avoid.

Finally, every label produced for the top-level ``--run-dir`` (whichever
route produced it) is prefixed with ``--invocation-prefix`` (default: the
run directory's own name), so CSVs emitted from separate invocations of
this script against different run directories can be concatenated later
without their invocation labels colliding.

This script only ever reads existing provenance files and replicate
filenames; it never invents a mapping that is not already recorded
somewhere (route 4's arithmetic rule is recorded in the sbatch script cited
above, and the caller supplies its constant explicitly).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def git_sha(root: Path) -> Optional[str]:
    """Same approach as scripts/stage_null_replicate_set.py's git_sha() /
    scripts/calibrate_switch_threshold.py's git_sha()."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _rep_index(path: Path) -> Optional[int]:
    """Extract the integer index from a ``rep_<digits>.npz`` filename, or
    None if the name doesn't follow that convention.
    """
    stem = path.stem
    if not stem.startswith("rep_"):
        return None
    suffix = stem[len("rep_"):]
    return int(suffix) if suffix.isdigit() else None


def _resolve_source_path(raw: str, base: Path) -> Path:
    src_path = Path(raw)
    if not src_path.is_absolute():
        src_path = base / src_path
    return src_path


def resolve_group_labels(
    run_dir: Path, files: List[Path], alignments_per_invocation: Optional[int]
) -> Tuple[List[str], Dict[str, str]]:
    """Resolve a bare (unprefixed) invocation label for each entry of
    `files` (all expected to live directly under `run_dir/npz`), trying
    routes 1-4 in order (see module docstring).

    Returns `(labels, routes_by_source)` where `labels` has one entry per
    `files` entry (in the same order) and `routes_by_source` maps each
    source run directory's name (there may be several, if route 1
    recursed across multiple source runs) to the route string used to
    resolve *that* source's labels.

    Raises SystemExit if no route applies and route 4 cannot be attempted
    because `alignments_per_invocation` is None, or if route 4 is reached
    but some file does not follow the rep_<digits>.npz naming.
    """
    run_dir = Path(run_dir)
    run_name = run_dir.name

    # --- Route 1: merge_null_runs.py's run_manifest.json --------------------
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
        replicates = manifest.get("replicates") if isinstance(manifest, dict) else None
        is_merge_manifest = (
            isinstance(replicates, list) and len(replicates) > 0
            and all(isinstance(e, dict) and "source" in e and "merged_index" in e for e in replicates)
        )
        if is_merge_manifest:
            by_index = {int(e["merged_index"]): e["source"] for e in replicates}
            indices = [_rep_index(f) for f in files]
            if all(i is not None and i in by_index for i in indices):
                labels: List[Optional[str]] = [None] * len(files)
                routes_by_source: Dict[str, str] = {}
                sources_by_dir: Dict[Path, List[Tuple[int, Path]]] = {}
                for pos, i in enumerate(indices):
                    src_path = _resolve_source_path(by_index[i], REPO_ROOT)
                    sources_by_dir.setdefault(src_path.parent, []).append((pos, src_path))

                for src_npz_dir, entries in sources_by_dir.items():
                    positions = [p for p, _ in entries]
                    sub_files = [f for _, f in entries]
                    sub_run_dir = src_npz_dir.parent
                    sub_labels, sub_routes = resolve_group_labels(
                        sub_run_dir, sub_files, alignments_per_invocation
                    )
                    prefix = sub_run_dir.name
                    for pos, lbl in zip(positions, sub_labels):
                        labels[pos] = f"{prefix}:{lbl}"
                    routes_by_source.update(sub_routes)

                return [lbl for lbl in labels], routes_by_source  # type: ignore[misc]

    # --- Route 2: stage_null_replicate_set.py's provenance.json -------------
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        provenance = None
        try:
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError):
            provenance = None
        replicates = provenance.get("replicates") if isinstance(provenance, dict) else None
        is_stage_provenance = (
            isinstance(replicates, list) and len(replicates) > 0
            and all(isinstance(e, dict) and "rep_id" in e and "invocation" in e for e in replicates)
        )
        if is_stage_provenance:
            by_rep_id = {e["rep_id"]: str(e["invocation"]) for e in replicates}
            stems = [f.stem for f in files]
            if all(s in by_rep_id for s in stems):
                route = (
                    f"provenance.json at {provenance_path} "
                    "(stage_null_replicate_set.py's replicates[].invocation)"
                )
                return [by_rep_id[s] for s in stems], {run_name: route}

    # --- Route 3: run_null_calibration.py's logs/progress.jsonl -------------
    progress_path = run_dir / "logs" / "progress.jsonl"
    if progress_path.exists():
        by_replicate: Dict[int, str] = {}
        try:
            for line in progress_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict) and "replicate" in rec and "batch" in rec:
                    by_replicate[int(rec["replicate"])] = str(rec["batch"])
        except (OSError, json.JSONDecodeError):
            by_replicate = {}
        if by_replicate:
            indices = [_rep_index(f) for f in files]
            if all(i is not None and i in by_replicate for i in indices):
                route = (
                    f"logs/progress.jsonl at {progress_path} "
                    "({'replicate','batch'} records)"
                )
                return [by_replicate[i] for i in indices], {run_name: route}

    # --- Route 4: index arithmetic (requires --alignments-per-invocation) ---
    if alignments_per_invocation is None:
        raise SystemExit(
            f"{run_dir}: no run_manifest.json (replicates[].source), no "
            f"provenance.json (replicates[].rep_id/.invocation), and no "
            f"logs/progress.jsonl (replicate/batch records) found. The only "
            "remaining resolution route is index arithmetic "
            "(invocation = rep_index // alignments_per_invocation), which "
            "requires --alignments-per-invocation. That convention is "
            "documented at scripts/euler/run_score_null_calibration_array.sh:78-79 "
            "(`CHUNK=$(( IDX / 10 ))`) -- pass --alignments-per-invocation with "
            "the value that applies to this run directory's simulate "
            "invocations rather than letting this script assume one."
        )
    indices = [_rep_index(f) for f in files]
    if any(i is None for i in indices):
        bad = [f.name for f, i in zip(files, indices) if i is None]
        raise SystemExit(
            f"{run_dir}: cannot apply index arithmetic -- these files do not "
            f"follow the rep_<digits>.npz naming: {bad}"
        )
    route = (
        f"index arithmetic (invocation = rep_index // {alignments_per_invocation}; "
        "see scripts/euler/run_score_null_calibration_array.sh:78-79)"
    )
    labels = [str(i // alignments_per_invocation) for i in indices]  # type: ignore[operator]
    return labels, {run_name: route}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="replicate run directory containing npz/rep_*.npz")
    p.add_argument("--out-csv", type=Path, required=True,
                   help="where to write the rep_file,invocation CSV")
    p.add_argument("--alignments-per-invocation", type=int, default=None,
                   help="number of alignments each simulate invocation produced; "
                        "only used (and required) if index arithmetic (route 4) "
                        "is reached -- see module docstring for where this value "
                        "is documented")
    p.add_argument("--invocation-prefix", type=str, default=None,
                   help="prepended to every emitted invocation label "
                        "(default: --run-dir's own directory name), so labels "
                        "from separate invocations of this script cannot "
                        "collide when their CSVs are combined")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved mapping; write nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    npz_dir = run_dir / "npz"

    if not npz_dir.is_dir():
        raise SystemExit(f"--run-dir {run_dir} has no npz/ subdirectory")

    files = sorted(npz_dir.glob("rep_*.npz"))
    if not files:
        raise SystemExit(f"No rep_*.npz files found under {npz_dir}")

    invocation_prefix = (
        args.invocation_prefix if args.invocation_prefix is not None else run_dir.name
    )

    labels, routes_by_source = resolve_group_labels(
        run_dir, files, args.alignments_per_invocation
    )
    final_labels = [f"{invocation_prefix}:{lbl}" for lbl in labels]

    rows = [
        {"rep_file": f.name, "invocation": lbl}
        for f, lbl in zip(files, final_labels)
    ]
    n_invocations = len(set(final_labels))

    if args.dry_run:
        print(f"[dry-run] would write {len(rows)} row(s), {n_invocations} distinct "
              f"invocation(s), to {args.out_csv}")
        for row in rows:
            print(f"  {row['rep_file']} -> {row['invocation']}")
        print(f"[dry-run] routes used: {routes_by_source}")
        return 0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rep_file", "invocation"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {args.out_csv} ({len(rows)} row(s), {n_invocations} distinct invocation(s))")

    sidecar = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(REPO_ROOT),
        "argv": list(sys.argv),
        "run_dir": str(run_dir),
        "routes_by_source": routes_by_source,
        "n_replicates": len(rows),
        "n_invocations": n_invocations,
    }
    sidecar_path = args.out_csv.with_suffix(args.out_csv.suffix + ".provenance.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(f"[write] {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
