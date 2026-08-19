#!/usr/bin/env python3
"""
stage_null_replicate_set.py

Stages a batch of null-replicate .npz files produced under some external
naming scheme (e.g. a treelen-scan batch named ``tl_chunk<C>_sim1.npz``)
into the ``rep_NNNN.npz`` naming that
scripts/calibrate_switch_threshold.py's loader expects (it globs
``rep_*.npz`` in the null directory). Files are only *copied* into the new
names, never moved or symlinked, so the originals survive untouched.

The new ``rep_NNNN`` index is arbitrary (just sort order); the mapping back
to the file's originating invocation is therefore written down explicitly
in ``provenance.json`` rather than left to be inferred from the index.

Seed convention
-----------------
The simulate seed recorded for a given invocation index ``c`` is
``seed_base + c * seed_stride``. This is the same convention documented at
scripts/package_null_calibration_for_euler.py's ``CHUNK_SEED_STRIDE``
constant and used when it builds the per-chunk sbatch seed line (see that
module around lines 231 and 384). This script only *records* the seed
implied by that convention for traceability -- it does not invoke any
simulation itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_null_replicate import SCORE_COLUMNS  # noqa: E402

# Defaults for the seed convention described in the module docstring above.
SEED_BASE_DEFAULT = 20260731
SEED_STRIDE_DEFAULT = 100003


def git_sha(root: Path) -> Optional[str]:
    """Same approach as scripts/calibrate_switch_threshold.py's git_sha()."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate_seed_for_invocation(invocation: str, seed_base: int, seed_stride: int) -> Optional[int]:
    """seed_base + int(invocation) * seed_stride (see module docstring for
    the origin of this convention). Returns None if `invocation` is not a
    plain non-negative integer, since the convention is only defined for
    integer invocation indices.
    """
    if not invocation.isdigit():
        return None
    return seed_base + int(invocation) * seed_stride


def match_files(source_dir: Path, pattern: "re.Pattern[str]") -> List[Dict]:
    """Match every regular file directly inside `source_dir` against
    `pattern` (fullmatch on the filename) and extract its `invocation`
    group. Files that don't match are skipped with a warning listing them.
    """
    matched: List[Dict] = []
    unmatched: List[str] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        m = pattern.fullmatch(path.name)
        if m is None:
            unmatched.append(path.name)
            continue
        matched.append({"path": path, "invocation": m.group("invocation")})
    if unmatched:
        warnings.warn(f"Skipping {len(unmatched)} file(s) not matching --pattern: {unmatched}")
    return matched


def sort_key(record: Dict):
    invocation = record["invocation"]
    primary = int(invocation) if invocation.isdigit() else invocation
    return (primary, record["path"].name)


def validate_npz(path: Path) -> int:
    """Check that `path` contains every SCORE_COLUMNS array at one common
    length; return that length. Raises SystemExit (not a warning) on any
    problem, since a single bad replicate must abort the whole staging run
    rather than being silently skipped or partially staged.
    """
    try:
        with np.load(path) as data:
            missing = [c for c in SCORE_COLUMNS if c not in data.files]
            if missing:
                raise SystemExit(
                    f"{path}: missing array(s) {missing}; refusing to stage this set."
                )
            lengths = {c: int(data[c].shape[0]) for c in SCORE_COLUMNS}
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"{path}: could not be read as a valid .npz ({exc}); refusing to stage this set."
        )
    distinct = set(lengths.values())
    if len(distinct) != 1:
        raise SystemExit(
            f"{path}: SCORE_COLUMNS arrays have inconsistent lengths {lengths}; "
            "refusing to stage this set."
        )
    return distinct.pop()


def build_plan(records: List[Dict], seed_base: int, seed_stride: int) -> List[Dict]:
    plan = []
    for idx, rec in enumerate(records):
        plan.append({
            "rep_id": f"rep_{idx:04d}",
            "source_path": rec["path"],
            "source_file": rec["path"].name,
            "invocation": rec["invocation"],
            "simulate_seed": simulate_seed_for_invocation(rec["invocation"], seed_base, seed_stride),
        })
    return plan


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--source-dir", type=Path, required=True,
                   help="directory holding the input .npz files")
    p.add_argument("--pattern", required=True,
                   help="Python regex with exactly one named group 'invocation', "
                        "fullmatch'd against each file's name; non-matching files "
                        "are skipped with a warning")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="where to write npz/rep_NNNN.npz and provenance.json")
    p.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT,
                   help=f"see module docstring for the seed convention (default {SEED_BASE_DEFAULT})")
    p.add_argument("--seed-stride", type=int, default=SEED_STRIDE_DEFAULT,
                   help=f"see module docstring for the seed convention (default {SEED_STRIDE_DEFAULT})")
    p.add_argument("--force", action="store_true",
                   help="allow staging into a non-empty <out-dir>/npz/")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be written; write nothing")
    args = p.parse_args(argv)

    try:
        pattern = re.compile(args.pattern)
    except re.error as exc:
        p.error(f"--pattern is not a valid regex: {exc}")
        raise  # pragma: no cover - p.error() exits
    if list(pattern.groupindex) != ["invocation"]:
        p.error("--pattern must contain exactly one named group, 'invocation'")
    args.pattern_re = pattern
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    source_dir: Path = args.source_dir
    out_dir: Path = args.out_dir
    npz_dir = out_dir / "npz"

    if not source_dir.is_dir():
        raise SystemExit(f"--source-dir {source_dir} is not a directory")

    records = match_files(source_dir, args.pattern_re)
    if not records:
        raise SystemExit(f"No files in {source_dir} matched --pattern {args.pattern!r}")

    records.sort(key=sort_key)

    lengths = [validate_npz(rec["path"]) for rec in records]
    distinct_lengths = set(lengths)
    if len(distinct_lengths) != 1:
        detail = {rec["path"].name: n for rec, n in zip(records, lengths)}
        raise SystemExit(
            f"Replicates disagree on test count: {detail}; refusing to stage this set."
        )
    n_tests = distinct_lengths.pop()

    plan = build_plan(records, args.seed_base, args.seed_stride)

    if args.dry_run:
        print(f"[dry-run] would stage {len(plan)} replicate(s) ({n_tests} tests each) into {npz_dir}")
        for entry in plan:
            print(f"  {entry['rep_id']} <- {entry['source_file']} "
                  f"(invocation={entry['invocation']}, simulate_seed={entry['simulate_seed']})")
        print(f"[dry-run] would write {out_dir / 'provenance.json'}")
        return 0

    if npz_dir.exists() and any(npz_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"{npz_dir} already exists and is non-empty; pass --force to overwrite."
        )

    npz_dir.mkdir(parents=True, exist_ok=True)

    replicate_records = []
    for entry in plan:
        dest = npz_dir / f"{entry['rep_id']}.npz"
        shutil.copy2(entry["source_path"], dest)
        replicate_records.append({
            "rep_id": entry["rep_id"],
            "source_file": entry["source_file"],
            "source_sha256": sha256_of(entry["source_path"]),
            "invocation": entry["invocation"],
            "simulate_seed": entry["simulate_seed"],
        })
        print(f"[stage] {entry['source_file']} -> {dest.name}")

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(REPO_ROOT),
        "argv": list(sys.argv),
        "source_dir": str(source_dir),
        "pattern": args.pattern,
        "n_replicates": len(replicate_records),
        "n_tests": n_tests,
        "replicates": replicate_records,
    }
    provenance_path = out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[write] {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
