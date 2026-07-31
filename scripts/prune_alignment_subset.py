#!/usr/bin/env python3
"""
prune_alignment_subset.py

Create a size-controlled subset of the real alignment + tree, for
feasibility/cost experiments (e.g. measuring how IQ-TREE's C20 mixture
model fit scales with taxon count) without running the full 21,218-taxon
dataset. Not part of the production pipeline -- a throwaway-input
generator for calibration experiments.

Selects N taxa uniformly at random (seeded, reproducible) from the real
alignment's identifier set, prunes the real ML tree down to those taxa
with ete3 (preserving branch lengths), and writes a matching FASTA
subset. Every path, N and seed is a CLI argument -- nothing hardcoded.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ete3 import Tree

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            seqs[name] = "".join(chunks)
    return seqs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune the real alignment/tree to a random subset of N taxa.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alignment", type=Path,
        default=REPO_ROOT / "data" / "interim" / "IPR019888_trimmed.aln",
    )
    parser.add_argument(
        "--tree", type=Path,
        default=REPO_ROOT / "data" / "interim" / "iqtree_asr" / "IPR019888.treefile",
    )
    parser.add_argument("--n-taxa", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prefix", default=None, help="Default: subset_n<N>_seed<seed>")
    args = parser.parse_args()

    prefix = args.prefix or f"subset_n{args.n_taxa}_seed{args.seed}"
    args.outdir.mkdir(parents=True, exist_ok=True)

    seqs = read_fasta(args.alignment)
    all_names = sorted(seqs.keys())
    if args.n_taxa > len(all_names):
        raise SystemExit(f"--n-taxa {args.n_taxa} exceeds available taxa {len(all_names)}")

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(all_names, args.n_taxa))

    tree = Tree(str(args.tree), format=1)
    tree.prune(chosen, preserve_branch_length=True)

    tree_path = args.outdir / f"{prefix}.treefile"
    tree.write(outfile=str(tree_path), format=1)

    fasta_path = args.outdir / f"{prefix}.fasta"
    with open(fasta_path, "w") as fh:
        for name in chosen:
            fh.write(f">{name}\n{seqs[name]}\n")

    print(f"Selected {len(chosen)} taxa (seed={args.seed}) from {len(all_names)} available.")
    print(f"Wrote pruned tree:  {tree_path}")
    print(f"Wrote subset FASTA: {fasta_path}  ({len(chosen)} seqs x {len(next(iter(seqs.values())))} cols)")


if __name__ == "__main__":
    main()
