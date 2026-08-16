"""Tests for the taxon-set guard in scripts/simulate_null_persite.py.

These cover the failure mode that motivated the guard: Euler job 10856979
simulated against a stale 21,641-sequence composition alignment while the
tree had 21,218 leaves, exited 0, was reported COMPLETED by SLURM, and
produced an alignment whose per-column frequencies and gap pattern were both
wrong. Nothing in the pipeline noticed until the downstream key check failed
hours later. Only the guard is exercised here -- AliSim itself is not run.
"""

import re

import pytest

from scripts import simulate_null_persite as snp

# (A,B) and (C,D) are cherries; N1/N2 are internal labels that must NOT be
# mistaken for leaves.
NEWICK = "((A:0.1,B:0.2)N1:0.3,(C:0.1,D:0.2)N2:0.3);"


def _write_tree(tmp_path, text=NEWICK):
    p = tmp_path / "sim.treefile"
    p.write_text(text)
    return p


def test_leaf_names_excludes_internal_labels(tmp_path):
    assert snp.read_newick_leaf_names(_write_tree(tmp_path)) == ["A", "B", "C", "D"]


def test_leaf_names_strips_comments(tmp_path):
    tree = _write_tree(tmp_path, "((A:0.1[&rate=1],B:0.2)N1:0.3,(C:0.1,D:0.2)N2:0.3);")
    assert snp.read_newick_leaf_names(tree) == ["A", "B", "C", "D"]


def test_matching_taxon_sets_pass(tmp_path):
    tree = _write_tree(tmp_path)
    snp.check_taxon_sets(tmp_path / "aln.fa", {"A", "B", "C", "D"}, tree, False)


def test_alignment_superset_is_fatal_by_default(tmp_path):
    """The stale-alignment case: extra sequences skew every column's
    frequency vector even though every tree tip is still present."""
    tree = _write_tree(tmp_path)
    with pytest.raises(SystemExit, match="1 alignment sequences absent from the tree"):
        snp.check_taxon_sets(tmp_path / "aln.fa", {"A", "B", "C", "D", "E"}, tree, False)


def test_alignment_superset_allowed_with_flag(tmp_path):
    tree = _write_tree(tmp_path)
    snp.check_taxon_sets(tmp_path / "aln.fa", {"A", "B", "C", "D", "E"}, tree, True)


def test_tree_only_leaf_is_fatal_even_with_flag(tmp_path):
    """A tip missing from the alignment can never be gap-masked, so it is
    fatal regardless of --allow-taxon-mismatch."""
    tree = _write_tree(tmp_path)
    with pytest.raises(SystemExit, match="1 tree leaves absent from the alignment"):
        snp.check_taxon_sets(tmp_path / "aln.fa", {"A", "B", "C"}, tree, True)


def test_duplicate_tip_names_are_fatal(tmp_path):
    tree = _write_tree(tmp_path, "((A:0.1,B:0.2)N1:0.3,(A:0.1,D:0.2)N2:0.3);")
    with pytest.raises(SystemExit, match="distinct"):
        snp.check_taxon_sets(tmp_path / "aln.fa", {"A", "B", "D"}, tree, False)


@pytest.mark.parametrize("n_skipped", [1, 92])
def test_gap_mask_skip_is_fatal(tmp_path, n_skipped):
    """apply_gap_mask leaves a sequence ungapped when the name matches but the
    length does not; that must abort rather than emit the alignment."""
    sim = tmp_path / "rep_1.fa"
    sim.write_text("".join(f">S{i}\nAAAA\n" for i in range(n_skipped + 1)))
    # S0 gets a correct-length mask; the rest are wrong-length, so skipped.
    mask = {"S0": "A-A-"}
    mask.update({f"S{i}": "AA" for i in range(1, n_skipped + 1)})
    n_masked, skipped, _ = snp.apply_gap_mask(sim, mask)
    assert (n_masked, skipped) == (1, n_skipped)
