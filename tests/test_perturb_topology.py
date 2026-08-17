"""Tests for scripts/perturb_topology.py.

The perturbed tree feeds a simulation whose tips are matched to the real
alignment by name and gap-masked that way, so a lost or renamed leaf would
corrupt the replicate silently rather than fail. These check the invariants
that protect against that, plus that the perturbation actually perturbs.
"""

from pathlib import Path

import numpy as np
import pytest
from ete3 import Tree

from scripts import perturb_topology as pt

NEWICK = (
    "(((A:0.1,B:0.2)i1:0.3,(C:0.15,D:0.25)i2:0.35)i3:0.4,"
    "((E:0.1,F:0.2)i4:0.3,(G:0.15,H:0.25)i5:0.35)i6:0.4);"
)


def _write(tmp_path, text=NEWICK, name="in.tree"):
    p = tmp_path / name
    p.write_text(text + "\n")
    return p


def _run(tmp_path, n_moves, seed=1, text=NEWICK):
    src = _write(tmp_path, text)
    out = tmp_path / f"out_{n_moves}_{seed}.tree"
    assert pt.main(["--in-tree", str(src), "--out-tree", str(out),
                    "--n-moves", str(n_moves), "--seed", str(seed)]) == 0
    return Tree(str(out), format=1)


def test_leaf_set_is_preserved(tmp_path):
    before = sorted(leaf.name for leaf in pt.load_tree(_write(tmp_path)))
    after = sorted(leaf.name for leaf in _run(tmp_path, 5))
    assert before == after


def test_total_branch_length_is_preserved(tmp_path):
    before = sum(n.dist for n in pt.load_tree(_write(tmp_path)).traverse())
    after = sum(n.dist for n in _run(tmp_path, 5).traverse())
    assert np.isclose(before, after, rtol=1e-9)


def test_topology_actually_changes(tmp_path):
    before = pt.load_tree(_write(tmp_path)).write(format=9)
    assert _run(tmp_path, 5).write(format=9) != before


def test_zero_moves_leaves_the_topology_alone(tmp_path):
    before = pt.load_tree(_write(tmp_path)).write(format=9)
    assert _run(tmp_path, 0).write(format=9) == before


def test_same_seed_gives_the_same_tree(tmp_path):
    a = _run(tmp_path, 4, seed=7).write(format=9)
    b = _run(tmp_path, 4, seed=7).write(format=9)
    assert a == b


def test_different_seeds_give_different_trees(tmp_path):
    a = _run(tmp_path, 6, seed=1).write(format=9)
    b = _run(tmp_path, 6, seed=2).write(format=9)
    assert a != b


def test_more_moves_move_further(tmp_path):
    """A larger move count should not produce a tree closer to the original;
    this is what makes --n-moves a usable dial for how much error to inject."""
    src = pt.load_tree(_write(tmp_path))
    base = {frozenset(leaf.name for leaf in n) for n in src.traverse() if not n.is_leaf()}

    def shared(n_moves):
        t = _run(tmp_path, n_moves, seed=3)
        got = {frozenset(leaf.name for leaf in n) for n in t.traverse() if not n.is_leaf()}
        return len(base & got)

    assert shared(12) <= shared(2)


def test_multifurcating_tree_is_handled(tmp_path):
    """The real ASR tree contains polytomies (559 of them), so a move must
    cope with a node having more than two children."""
    poly = "((A:0.1,B:0.2,C:0.3)i1:0.4,(D:0.1,E:0.2,F:0.3)i2:0.4);"
    tree = _run(tmp_path, 4, text=poly)
    assert sorted(leaf.name for leaf in tree) == list("ABCDEF")


def test_unparseable_tree_is_rejected(tmp_path):
    bad = tmp_path / "bad.tree"
    bad.write_text("this is not newick\n")
    with pytest.raises(SystemExit, match="Could not parse"):
        pt.main(["--in-tree", str(bad), "--out-tree", str(tmp_path / "o.tree"),
                 "--n-moves", "1", "--seed", "1"])
