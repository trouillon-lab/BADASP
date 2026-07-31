#!/usr/bin/env python3
"""
measure_c20_cost_scaling.py

Measure how IQ-TREE's LG+C20+F+G mixture-model fit scales in memory and
wall-clock time with taxon count, by fitting it on pruned subsets of the
real alignment/tree (see prune_alignment_subset.py) of increasing size.
Used to size an Euler job for the full 21,218-taxon fit -- not part of
the production pipeline.

Each subset fit also writes a .sitefreq file (via --tree-freq), which
lets the same run double as the empirical test of whether AliSim can
consume that file (see verify_null_simulation.py workflow / accompanying
notes) at negligible extra cost.

For every requested taxon count this script:
  1. prunes the real alignment/tree to N taxa (deterministic, seeded)
  2. runs `iqtree2 -m LG+C20+F+G --tree-freq <tree>` on the subset
  3. parses IQ-TREE's own reported "xxx MB RAM is required" figure and
     measured wall-clock time from the log
  4. appends one row per subset to a CSV

This does not itself decide what to run on Euler -- it only measures.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RAM_RE = re.compile(r"NOTE:\s*([\d.]+)\s*MB RAM(?:\s*\(([\d.]+)\s*GB\))?\s*is required", re.IGNORECASE)


def prune_subset(n_taxa: int, seed: int, alignment: Path, tree: Path, outdir: Path,
                  python_bin: str) -> tuple[Path, Path]:
    prefix = f"subset_n{n_taxa}_seed{seed}"
    fasta_path = outdir / f"{prefix}.fasta"
    tree_path = outdir / f"{prefix}.treefile"
    if fasta_path.exists() and tree_path.exists():
        return fasta_path, tree_path
    cmd = [
        python_bin, str(REPO_ROOT / "scripts" / "prune_alignment_subset.py"),
        "--alignment", str(alignment), "--tree", str(tree),
        "--n-taxa", str(n_taxa), "--seed", str(seed),
        "--outdir", str(outdir), "--prefix", prefix,
    ]
    subprocess.run(cmd, check=True)
    return fasta_path, tree_path


def run_c20_fit(fasta_path: Path, tree_path: Path, outdir: Path, prefix: str,
                 iqtree_bin: str, threads: int, timeout: float | None) -> dict:
    log_path = outdir / f"{prefix}.log"
    cmd = [
        iqtree_bin,
        "-s", str(fasta_path),
        "-t", str(tree_path),
        "--tree-fix",
        "-m", "LG+C20+F+G",
        "--tree-freq", str(tree_path),
        "--prefix", str(outdir / prefix),
        "-T", str(threads),
    ]
    print("Running:")
    print("  " + " ".join(cmd))
    start = time.monotonic()
    with open(log_path, "w") as log_fh:
        log_fh.write("$ " + " ".join(cmd) + "\n\n")
        log_fh.flush()
        try:
            result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                     cwd=REPO_ROOT, timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            result = None
    elapsed = time.monotonic() - start

    log_text = log_path.read_text()
    ram_matches = RAM_RE.findall(log_text)
    ram_mb = float(ram_matches[0][0]) if ram_matches else None

    return {
        "elapsed_sec": elapsed,
        "timed_out": timed_out,
        "returncode": None if result is None else result.returncode,
        "ram_mb_reported": ram_mb,
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure LG+C20+F+G fit cost (RAM, wall-clock) vs taxon count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-taxa", type=int, nargs="+", required=True,
                         help="Taxon counts to test, e.g. 500 2000 5000.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alignment", type=Path,
                         default=REPO_ROOT / "data" / "interim" / "IPR019888_trimmed.aln")
    parser.add_argument("--tree", type=Path,
                         default=REPO_ROOT / "data" / "interim" / "iqtree_asr" / "IPR019888.treefile")
    parser.add_argument("--outdir", type=Path, required=True,
                         help="Scratch directory for subset alignments/trees and IQ-TREE outputs.")
    parser.add_argument("--csv-out", type=Path, required=True,
                         help="CSV file to append/write results to.")
    parser.add_argument("--iqtree-bin", default="iqtree2")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=None,
                         help="Per-subset timeout in seconds.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)

    write_header = not args.csv_out.exists()
    with open(args.csv_out, "a") as csv_fh:
        if write_header:
            csv_fh.write("n_taxa,seed,elapsed_sec,timed_out,returncode,ram_mb_reported,log_path\n")

        for n_taxa in args.n_taxa:
            print(f"\n=== n_taxa={n_taxa} ===")
            fasta_path, tree_path = prune_subset(
                n_taxa, args.seed, args.alignment, args.tree, args.outdir, args.python_bin,
            )
            prefix = f"c20_fit_n{n_taxa}_seed{args.seed}"
            stats = run_c20_fit(fasta_path, tree_path, args.outdir, prefix,
                                 args.iqtree_bin, args.threads, args.timeout)
            print(f"  elapsed={stats['elapsed_sec']:.1f}s  "
                  f"timed_out={stats['timed_out']}  "
                  f"returncode={stats['returncode']}  "
                  f"ram_mb_reported={stats['ram_mb_reported']}")
            csv_fh.write(f"{n_taxa},{args.seed},{stats['elapsed_sec']:.2f},"
                         f"{stats['timed_out']},{stats['returncode']},"
                         f"{stats['ram_mb_reported']},{stats['log_path']}\n")
            csv_fh.flush()

    print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
