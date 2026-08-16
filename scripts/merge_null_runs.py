#!/usr/bin/env python3
"""
merge_null_runs.py

Combine the .npz replicates of several null-calibration runs into one
directory with contiguous rep_*.npz numbering, so
calibrate_switch_threshold.py (which globs `<run dir>/npz/rep_*.npz`) can
use them together.

When this is valid
------------------
Replicates from different runs are exchangeable draws from the same null
only if they were produced by the same simulator against the same inputs.
This script therefore refuses to merge unless every source agrees on:

  * the number of tests per replicate -- a mismatch means the replicate was
    positionally aligned to a DIFFERENT observed-scores table, and mixing
    them would silently misalign every test;
  * the observed-scores path recorded in each run's run_manifest.json;
  * the shrinkage used to build the per-site frequency vectors.

Distinct seeds across runs are expected and are not an error -- that is
what makes the replicates independent. Identical seeds ARE flagged, since
two runs with the same seed produce the same alignments and would inflate
the apparent replicate count without adding information.

Metadata agreement is necessary but NOT sufficient
--------------------------------------------------
Two local runs (`persite`, 40 replicates, and `persite_batch2`, 21) agreed
on every field above -- same observed table, same shrinkage, same node set,
same clade sizes, byte-comparable per-column frequency vectors -- and still
produced null tails 6x apart: 82 vs 493 split exceedances per replicate at
t = 1.8675, with completely non-overlapping ranges (45-156 vs 343-648). The
difference tracks `RC`, which depends only on the simulated leaf alignment,
so it originates in simulation rather than scoring; the cause is not yet
identified, and `persite` has no run_manifest.json recording how it was
produced. Merging them makes the global threshold rule fail outright.

So this script also compares the runs' actual null tails and refuses to
merge sets that are not plausibly draws from the same null. That check is
what the metadata comparison alone would have missed.

Files are symlinked by default rather than copied: replicates are ~3.3 MB
each and the sources are the real artifacts. Pass --copy for a
self-contained directory (e.g. before rsyncing somewhere else).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# --check-only verdict exit codes. Deliberately NOT 1 or 2: argparse uses 2
# and SystemExit("No rep_*.npz found ...") uses 1, so reusing them let an
# empty run directory be reported by a shell gate as "not comparable" -- a
# real verdict -- when the script had simply errored. That happened.
EXIT_NOT_COMPARABLE = 3
EXIT_UNDETERMINED = 4


def npz_dir_of(run_dir: Path) -> Path:
    """Accept either a run directory or the npz directory itself."""
    candidate = run_dir / "npz"
    return candidate if candidate.exists() else run_dir


def read_manifest(run_dir: Path) -> Optional[dict]:
    for candidate in (run_dir / "run_manifest.json", run_dir.parent / "run_manifest.json"):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def n_tests_of(path: Path) -> int:
    with np.load(path) as data:
        return int(data["badasp_score_left"].shape[0])


def split_statistic(path: Path) -> np.ndarray:
    """NaN-safe fmax(left, right), matching null_model._split_statistic."""
    with np.load(path) as data:
        return np.fmax(
            data["badasp_score_left"].astype(np.float64),
            data["badasp_score_right"].astype(np.float64),
        )


def exceedance_counts(files: List[Path], threshold: float) -> np.ndarray:
    """Per-replicate count of splits at or above `threshold`."""
    return np.array([
        int(np.nansum(split_statistic(f) >= threshold)) for f in files
    ], dtype=float)


# Below this many pooled exceedances the ratio of run means is dominated by
# Poisson noise (and can be infinite when one run happens to have zero), so
# the check reports "undetermined" rather than a verdict it cannot support.
MIN_EXCEEDANCES_TO_JUDGE = 30


def tail_comparability(sources: List[dict], quantile: float, max_ratio: float) -> dict:
    """Compare the runs' own null tails, not just their metadata.

    A shared threshold is taken from the pooled draws, and each run's
    per-replicate exceedance count is measured at it. Exchangeable runs give
    similar counts; the ratio of the largest to the smallest run mean is the
    summary. Ranges that do not even overlap are reported separately, since
    that rules out sampling variation without needing a distributional
    assumption.
    """
    pooled = np.concatenate([
        split_statistic(f)[np.isfinite(split_statistic(f))] for s in sources for f in s["files"][:5]
    ])
    threshold = float(np.quantile(pooled, quantile))
    per_run = []
    for s in sources:
        counts = exceedance_counts(s["files"], threshold)
        per_run.append({
            "run_dir": str(s["run_dir"]),
            "mean_exceedances": round(float(counts.mean()), 2),
            "min": int(counts.min()),
            "max": int(counts.max()),
        })
    total = sum(r["mean_exceedances"] * len(s["files"]) for r, s in zip(per_run, sources))
    result = {
        "threshold": round(threshold, 4),
        "quantile": quantile,
        "per_run": per_run,
        "total_exceedances": round(float(total), 1),
        "max_ratio_allowed": max_ratio,
    }
    if total < MIN_EXCEEDANCES_TO_JUDGE:
        result.update({
            "comparable": None,
            "undetermined_reason": (
                f"only {total:.0f} pooled exceedances at q={quantile} "
                f"(need {MIN_EXCEEDANCES_TO_JUDGE}); too few to distinguish a "
                "real tail difference from Poisson noise."
            ),
        })
        return result

    means = np.array([r["mean_exceedances"] for r in per_run])
    ratio = float(means.max() / max(means.min(), 1e-9))
    disjoint = any(
        a["max"] < b["min"] or b["max"] < a["min"]
        for i, a in enumerate(per_run) for b in per_run[i + 1:]
    )
    result.update({
        "mean_ratio": round(ratio, 2),
        "ranges_disjoint": disjoint,
        "comparable": ratio <= max_ratio and not disjoint,
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge null-calibration replicate directories into one "
                    "with contiguous rep_*.npz numbering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True,
                        help="Two or more run directories (or npz directories) "
                             "to merge, in the order their replicates should be "
                             "numbered.")
    parser.add_argument("--check-only", action="store_true",
                        help="Report the tail-comparability check and exit "
                             "without writing anything. Use this to ask "
                             "whether two replicate sets behave like the same "
                             "null before committing to a merge. Exit status "
                             "carries the verdict: 0 comparable, "
                             f"{EXIT_NOT_COMPARABLE} not comparable, "
                             f"{EXIT_UNDETERMINED} undetermined. Those are "
                             "deliberately not 1 or 2, so an ordinary error "
                             "exit cannot be mistaken for a verdict.")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="Write the comparability report to this path "
                             "(works with --check-only).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Destination run directory; replicates are written "
                             "to <out-dir>/npz/.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy replicates instead of symlinking them.")
    parser.add_argument("--tail-quantile", type=float, default=0.999,
                        help="Pooled quantile defining the shared threshold at "
                             "which each run's null tail is measured.")
    parser.add_argument("--max-tail-ratio", type=float, default=1.5,
                        help="Largest allowed ratio between run mean exceedance "
                             "counts. Two local runs that agreed on all metadata "
                             "still differed by 6.0x.")
    parser.add_argument("--allow-incomparable-tails", action="store_true",
                        help="Merge even when the runs' null tails differ beyond "
                             "--max-tail-ratio. Only use this if the difference "
                             "is understood.")
    parser.add_argument("--allow-manifest-mismatch", action="store_true",
                        help="Proceed even if the runs disagree on the observed "
                             "scores path or shrinkage. Only use this when the "
                             "difference is known to be cosmetic (e.g. an "
                             "absolute vs relative path to the same file).")
    args = parser.parse_args(argv)

    if len(args.run_dir) < 2:
        parser.error("--run-dir needs at least two directories to merge.")
    if args.out_dir is None and not args.check_only:
        parser.error("--out-dir is required unless --check-only is given.")

    sources = []
    for run_dir in args.run_dir:
        npz_dir = npz_dir_of(run_dir)
        files = sorted(npz_dir.glob("rep_*.npz"))
        if not files:
            raise SystemExit(f"No rep_*.npz found in {npz_dir}")
        sources.append({
            "run_dir": run_dir,
            "npz_dir": npz_dir,
            "files": files,
            "manifest": read_manifest(run_dir),
            "n_tests": n_tests_of(files[0]),
        })
        print(f"{run_dir}: {len(files)} replicate(s), n_tests={sources[-1]['n_tests']}")

    # --- Compatibility checks --------------------------------------------
    n_tests = {s["n_tests"] for s in sources}
    if len(n_tests) > 1:
        raise SystemExit(
            f"Replicates disagree on the number of tests ({sorted(n_tests)}). "
            "They were aligned to different observed-scores tables and must "
            "not be merged."
        )

    def manifest_field(source, *path, default=None):
        node = source["manifest"] or {}
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    observed = {manifest_field(s, "inputs", "observed_scores", "path") for s in sources}
    shrinkages = {manifest_field(s, "shrinkage") for s in sources}
    seeds = [manifest_field(s, "seed") for s in sources]

    problems = []
    if len({o for o in observed if o is not None}) > 1:
        problems.append(f"observed-scores paths differ: {sorted(o for o in observed if o)}")
    if len({v for v in shrinkages if v is not None}) > 1:
        problems.append(f"shrinkage differs across runs: {sorted(v for v in shrinkages if v is not None)}")
    if problems:
        message = "; ".join(problems)
        if not args.allow_manifest_mismatch:
            raise SystemExit(
                f"Refusing to merge: {message}. Re-run with "
                "--allow-manifest-mismatch only if this difference is known "
                "to be cosmetic."
            )
        print(f"WARNING: proceeding despite: {message}", file=sys.stderr)

    known_seeds = [s for s in seeds if s is not None]
    if len(set(known_seeds)) < len(known_seeds):
        print(
            f"WARNING: repeated seed(s) among {known_seeds}: runs sharing a seed "
            "produce the same alignments, so merging them inflates the replicate "
            "count without adding independent information.",
            file=sys.stderr,
        )

    # --- Do the runs actually behave like the same null? -------------------
    tails = tail_comparability(sources, args.tail_quantile, args.max_tail_ratio)
    print(f"\nNull tail at pooled q={tails['quantile']} (score >= {tails['threshold']}):")
    for run in tails["per_run"]:
        print(f"  {run['run_dir']}: mean {run['mean_exceedances']} "
              f"exceedances/replicate (range {run['min']}..{run['max']})")
    if tails["comparable"] is None:
        print(f"  tail comparability UNDETERMINED: {tails['undetermined_reason']}")
    else:
        print(f"  ratio of run means: {tails['mean_ratio']}x"
              + ("  RANGES DO NOT OVERLAP" if tails["ranges_disjoint"] else ""))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(tails, indent=2) + "\n")
        print(f"Wrote {args.report_json}")
    if args.check_only:
        # Exit status carries the verdict so a shell gate can branch on it:
        # 0 comparable, 1 not comparable, 2 undetermined.
        return {True: 0,
                False: EXIT_NOT_COMPARABLE,
                None: EXIT_UNDETERMINED}[tails["comparable"]]
    if tails["comparable"] is False:
        message = (
            f"the runs' null tails differ by {tails['mean_ratio']}x"
            + (" and their per-replicate ranges do not even overlap, which rules "
               "out sampling variation" if tails["ranges_disjoint"] else "")
            + f" (allowed: {args.max_tail_ratio}x)"
        )
        if not args.allow_incomparable_tails:
            raise SystemExit(
                f"Refusing to merge: {message}. These are not draws from the "
                "same null; pooling them would corrupt every error-rate "
                "estimate. Identify the cause before merging, or pass "
                "--allow-incomparable-tails if it is understood."
            )
        print(f"WARNING: proceeding despite {message}", file=sys.stderr)

    # --- Write ------------------------------------------------------------
    out_npz = args.out_dir / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)
    for stale in out_npz.glob("rep_*.npz"):
        stale.unlink()

    total = sum(len(s["files"]) for s in sources)
    width = max(4, len(str(total - 1)))
    index = 0
    provenance: List[dict] = []
    for source in sources:
        for src in source["files"]:
            dst = out_npz / f"rep_{index:0{width}d}.npz"
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
            provenance.append({"merged_index": index, "source": str(src)})
            index += 1

    manifest = {
        "merged_from": [
            {
                "run_dir": str(s["run_dir"]),
                "n_replicates": len(s["files"]),
                "seed": manifest_field(s, "seed"),
                "shrinkage": manifest_field(s, "shrinkage"),
                "observed_scores": manifest_field(s, "inputs", "observed_scores", "path"),
            }
            for s in sources
        ],
        "n_replicates": total,
        "n_tests": sources[0]["n_tests"],
        "mode": "copy" if args.copy else "symlink",
        "tail_comparability": tails,
        "generated": datetime.now(timezone.utc).isoformat(),
        "replicates": provenance,
        # Carried forward so calibrate_switch_threshold.py's own
        # observed-table consistency check still has something to check
        # against, rather than falling back to "no manifest found".
        "inputs": {"observed_scores": {"path": next(iter(o for o in observed if o), None)}},
        "shrinkage": next(iter(v for v in shrinkages if v is not None), None),
        "seed": seeds[0] if seeds else None,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nMerged {total} replicate(s) into {out_npz} "
          f"({'copies' if args.copy else 'symlinks'})")
    print(f"Wrote {args.out_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
