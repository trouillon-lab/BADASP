#!/usr/bin/env python3
"""
perturb_topology.py

Apply random nearest-neighbour-interchange (NNI) moves to a tree, keeping
its leaf set and branch lengths, and write the result.

Why
---
The simulated null's ancestral reconstruction is over-confident relative to
the real data (see src/badasp/posterior_correction.py): the null's sequences
are generated under nearly the model the reconstruction assumes, so
reconstruction is easy on them, while real sequences violate the model and
make it genuinely uncertain.

The intended remedy was a heterotachous simulator, but IQ-TREE 2.4.0 cannot
combine heterotachy (`+Hn`) with the named frequency mixture (`FMIX`) that
the per-site-frequency null requires -- it crashes, because heterotachy is
itself implemented as a mixture. Dropping `FMIX` is not an option, since a
site-homogeneous null was already shown to be unusable here.

Simulating on a perturbed topology while reconstructing on the original is
the alternative the project plan already names (its "Tier 2" sensitivity
tier): it injects real, quantifiable model violation without needing any
model support. Topological error is also violation the real analysis
genuinely has -- the estimated tree is not the true tree -- so the null
inherits a form of error the observed data also carries.

An NNI move is used rather than random branch-length noise because
reconstruction re-optimises branch lengths anyway, so length perturbation
would largely be absorbed; a changed topology cannot be.

This script only perturbs a tree. Whether the resulting null's `p_AC`
distribution moves closer to the observed one is a measurement, made
elsewhere, and is not claimed here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ete3 import Tree

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_tree(path: Path) -> Tree:
    """Read a Newick tree, tolerating the internal-label conventions
    IQ-TREE and rooting tools produce."""
    last = None
    for fmt in (1, 0, 5):
        try:
            return Tree(str(path), format=fmt)
        except Exception as exc:  # noqa: BLE001 - ete3 raises broadly here
            last = exc
    raise SystemExit(f"Could not parse {path} in Newick format 1, 0 or 5: {last}")


def eligible_nodes(tree: Tree) -> list:
    """Internal nodes whose branch can host an NNI.

    An NNI swaps one child of a node with one child of its parent, so the
    node needs a parent that is itself internal and not the root, and both
    must have at least two children (the tree is multifurcating in places,
    which is fine -- we just need two things to exchange).
    """
    out = []
    for node in tree.traverse():
        if node.is_leaf() or node.is_root():
            continue
        parent = node.up
        if parent is None or parent.is_root():
            continue
        if len(node.children) >= 2 and len(parent.children) >= 2:
            out.append(node)
    return out


def apply_nni_moves(tree: Tree, n_moves: int, rng: np.random.Generator) -> int:
    """Apply up to `n_moves` random NNI moves in place; return how many stuck.

    Each move detaches one child of a node and one child of that node's
    parent and re-attaches them to the other's position. Branch lengths
    travel with the moved subtrees, so the total tree length is unchanged --
    only the topology differs.
    """
    applied = 0
    for _ in range(n_moves):
        candidates = eligible_nodes(tree)
        if not candidates:
            break
        node = candidates[rng.integers(len(candidates))]
        parent = node.up
        siblings = [c for c in parent.children if c is not node]
        if not siblings or not node.children:
            continue
        child = node.children[rng.integers(len(node.children))]
        sibling = siblings[rng.integers(len(siblings))]
        # Detach both, then re-attach each where the other was.
        child.detach()
        sibling.detach()
        node.add_child(sibling)
        parent.add_child(child)
        applied += 1
    return applied


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Apply random NNI moves to a tree, preserving leaves and branch lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in-tree", type=Path, required=True)
    p.add_argument("--out-tree", type=Path, required=True)
    p.add_argument("--n-moves", type=int, required=True,
                   help="Number of NNI moves to attempt. Expressed as a count "
                        "rather than a fraction so the amount of injected "
                        "topological error is explicit and reportable.")
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args(argv)

    tree = load_tree(args.in_tree)
    before_leaves = sorted(leaf.name for leaf in tree)
    before_length = sum(n.dist for n in tree.traverse())
    before_topology = tree.write(format=9)

    applied = apply_nni_moves(tree, args.n_moves, np.random.default_rng(args.seed))

    after_leaves = sorted(leaf.name for leaf in tree)
    after_length = sum(n.dist for n in tree.traverse())

    # The null pipeline matches simulated tips to the real alignment by name
    # and applies its gap mask that way, so losing or renaming a leaf would
    # silently corrupt the replicate rather than fail.
    if before_leaves != after_leaves:
        raise SystemExit(
            f"Leaf set changed: {len(before_leaves)} -> {len(after_leaves)}. Refusing to write."
        )
    if not np.isclose(before_length, after_length, rtol=1e-9):
        raise SystemExit(
            f"Total branch length changed: {before_length:.6f} -> {after_length:.6f}. "
            "An NNI move should relocate subtrees, not rescale them."
        )
    if applied and tree.write(format=9) == before_topology:
        raise SystemExit(
            f"{applied} move(s) reported but the topology is unchanged; the "
            "perturbation did nothing."
        )

    args.out_tree.parent.mkdir(parents=True, exist_ok=True)
    tree.write(outfile=str(args.out_tree), format=1)
    print(f"{len(after_leaves)} leaves preserved, total branch length "
          f"{after_length:.4f} unchanged")
    print(f"applied {applied}/{args.n_moves} NNI moves (seed {args.seed})")
    print(f"Wrote {args.out_tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
