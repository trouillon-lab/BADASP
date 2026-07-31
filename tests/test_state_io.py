from pathlib import Path

import numpy as np
import pytest

from src.badasp.state_io import (
    AA_ORDER,
    StateFileError,
    load_alignment_matrix,
    load_state_array,
)


def _write_state_file(path: Path, extra_lines: str = "") -> None:
    header = "Node\tSite\tState\t" + "\t".join(f"p_{aa}" for aa in AA_ORDER) + "\n"
    prob_line = "\t" + "\t".join(["0.0"] * 20)

    def row(node: str, site: int, state: str, probs: str = prob_line) -> str:
        return f"{node}\t{site}\t{state}{probs}\n"

    # Row where p_A = 0.7 and p_R = 0.3, all others 0.0.
    values = {aa: 0.0 for aa in AA_ORDER}
    values["A"] = 0.7
    values["R"] = 0.3
    probs_node1_site1 = "\t" + "\t".join(f"{values[aa]}" for aa in AA_ORDER)

    lines = [header]
    lines.append(row("Node1", 1, "A", probs_node1_site1))
    lines.append(row("Node1", 2, "G"))
    lines.append(row("Node2", 1, "R"))
    if extra_lines:
        lines.append(extra_lines)
    path.write_text("".join(lines), encoding="utf-8")


def test_load_state_array_shapes_and_lookups(tmp_path: Path) -> None:
    state_file = tmp_path / "test.state"
    _write_state_file(state_file)

    arr = load_state_array(state_file, n_sites=2)
    assert arr.n_sites == 2
    assert set(arr.node_index) == {"Node1", "Node2"}
    assert arr.states.shape == (2, 2)
    assert arr.probs.shape == (2, 2, 20)

    assert arr.posterior("Node1", 1, "A") == pytest.approx(0.7)
    assert arr.posterior("Node1", 1, "R") == pytest.approx(0.3)
    assert arr.posterior("Node1", 1, "a") == pytest.approx(0.7)  # case-insensitive


def test_ancestral_sequence_round_trip_and_missing_site(tmp_path: Path) -> None:
    state_file = tmp_path / "test.state"
    _write_state_file(state_file)

    # n_sites=3 so that site 3 (never present in the file) becomes "X".
    arr = load_state_array(state_file, n_sites=3)
    assert arr.ancestral_sequence("Node1") == "AGX"
    assert arr.ancestral_sequence("Node2") == "RXX"
    assert arr.ancestral_sequence("NoSuchNode") is None


def test_posterior_missing_node_raises_keyerror(tmp_path: Path) -> None:
    state_file = tmp_path / "test.state"
    _write_state_file(state_file)

    arr = load_state_array(state_file, n_sites=2)
    with pytest.raises(KeyError):
        arr.posterior("NoSuchNode", 1, "A")


def test_require_nodes_raises_state_file_error(tmp_path: Path) -> None:
    state_file = tmp_path / "test.state"
    _write_state_file(state_file)

    arr = load_state_array(state_file, n_sites=2)
    arr.require_nodes(["Node1", "Node2"])  # should not raise
    with pytest.raises(StateFileError):
        arr.require_nodes(["Node1", "MissingNode"])


def test_load_state_array_wrong_columns_raises(tmp_path: Path) -> None:
    state_file = tmp_path / "bad.state"
    state_file.write_text(
        "Node\tSite\tState\tsomething_else\n"
        "Node1\t1\tA\t0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(StateFileError):
        load_state_array(state_file)


def test_load_state_array_restricted_to_nodes(tmp_path: Path) -> None:
    state_file = tmp_path / "test.state"
    _write_state_file(state_file)

    arr = load_state_array(state_file, nodes=["Node2"], n_sites=2)
    assert set(arr.node_index) == {"Node2"}
    assert arr.ancestral_sequence("Node1") is None


def test_load_alignment_matrix_shape_and_row_indices(tmp_path: Path) -> None:
    aln_file = tmp_path / "aln.fasta"
    aln_file.write_text(
        ">seqA\nMKT\n>seqB\nMK-\n>seqC\nMXT\n",
        encoding="utf-8",
    )
    mat = load_alignment_matrix(aln_file)
    assert mat.n_sites == 3
    assert mat.matrix.shape == (3, 3)
    assert len(mat) == 3

    rows = mat.row_indices(["seqA", "seqC", "unknown_seq"])
    assert rows.tolist() == [mat.seq_index["seqA"], mat.seq_index["seqC"]]


def test_load_alignment_matrix_column_counts(tmp_path: Path) -> None:
    aln_file = tmp_path / "aln.fasta"
    aln_file.write_text(
        ">seqA\nMKT\n>seqB\nMK-\n>seqC\nMXT\n",
        encoding="utf-8",
    )
    mat = load_alignment_matrix(aln_file)
    rows = mat.row_indices(["seqA", "seqB", "seqC"])

    counts_pos0 = mat.column_counts(rows, 0)
    assert counts_pos0[ord("M")] == 3

    counts_pos2 = mat.column_counts(rows, 2)
    assert counts_pos2[ord("T")] == 2
    assert counts_pos2[ord("-")] == 1


def test_load_alignment_matrix_raises_on_ragged_input(tmp_path: Path) -> None:
    aln_file = tmp_path / "ragged.fasta"
    aln_file.write_text(">seqA\nMKT\n>seqB\nMK\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_alignment_matrix(aln_file)


def test_load_alignment_matrix_raises_on_duplicate_identifiers(tmp_path: Path) -> None:
    aln_file = tmp_path / "dup.fasta"
    aln_file.write_text(">seqA\nMKT\n>seqA\nMKT\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_alignment_matrix(aln_file)
