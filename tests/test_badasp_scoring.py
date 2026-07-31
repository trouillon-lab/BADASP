import csv
import random
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from Bio import SeqIO
from ete3 import Tree
from src.badasp.scoring import (
    load_state_file,
    calculate_recent_conservation,
    calculate_recent_conservation_counts,
    calculate_ancestral_conservation,
    extract_posterior_probability,
    score_tree_nodes,
)
from src.badasp.state_io import AA_ORDER, StateFileError

def test_load_state_file(tmp_path: Path) -> None:
    state_file = tmp_path / "test.state"
    state_file.write_text(
        "#Some comment\n"
        "Node\tSite\tState\tA\tR\tN\tD\tC\tQ\tE\tG\tH\tI\tL\tK\tM\tF\tP\tS\tT\tW\tY\tV\n"
        "Node1\t1\tA\t0.9\t0.05\t0.05\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n"
        "Node1\t2\tG\t0.05\t0\t0\t0\t0\t0\t0\t0.9\t0.05\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n",
        encoding="utf-8"
    )
    res = load_state_file(state_file)
    assert "Node1" in res
    df = res["Node1"]
    assert len(df) == 2
    assert df.iloc[0]["Site"] == 1
    assert df.iloc[0]["State"] == "A"
    assert df.iloc[0]["probs"][0] == 0.9

def test_conservation_calculations() -> None:
    # Test calculate_recent_conservation
    seqs = ["MKT", "MKT", "MKT"]
    assert calculate_recent_conservation(seqs, 0) == pytest.approx(0.6, abs=1e-4)  # Perfect M conservation normalized
    
    # Test calculate_ancestral_conservation
    assert calculate_ancestral_conservation("A", "A") == 1.0
    assert calculate_ancestral_conservation("A", "R") == -1.0
    assert calculate_ancestral_conservation("-", "A") == -1.0

def test_extract_posterior_probability() -> None:
    probs = [0.0] * 20
    probs[0] = 0.8  # 'A' probability
    df = pd.DataFrame([{"Site": 1, "State": "A", "probs": probs}])
    state_data = {"Node1": df}
    assert extract_posterior_probability(state_data, "Node1", 1, "A") == 0.8
    assert extract_posterior_probability(state_data, "Node1", 1, "R") == 0.0
    assert extract_posterior_probability(state_data, "Node2", 1, "A") == 0.0

def test_score_tree_nodes_with_samples_file(tmp_path: Path) -> None:
    # 1. Create a mock consensus tree (without NHX tags)
    # Leaves: 3 left, 2 right
    alerax_tree_file = tmp_path / "IPR019888.nwk"
    alerax_tree_file.write_text("(((L1:0.1,L2:0.1):0.1,L3:0.1):0.1,(R1:0.1,R2:0.1):0.1);\n", encoding="utf-8")
    
    # 2. Create the samples newick file under /all/IPR019888_samples.newick
    all_dir = tmp_path / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    samples_file = all_dir / "IPR019888_samples.newick"
    # This tree contains events as node names (e.g. S, D, T)
    samples_file.write_text(
        "(((L1:0.1,L2:0.1)S:0.1,L3:0.1)D:0.1,(R1:0.1,R2:0.1)S:0.1)D:0.1;\n", encoding="utf-8"
    )
    
    # 3. Create ASR tree matching consensus leaves
    asr_tree_file = tmp_path / "asr.treefile"
    asr_tree_file.write_text("(((L1:0.1,L2:0.1)Node2:0.1,L3:0.1)Node1:0.1,(R1:0.1,R2:0.1)Node3:0.1)Node0:0.1;\n", encoding="utf-8")
    
    # 4. Create state file
    state_file = tmp_path / "asr.state"
    # Node0 to Node3 and L1, L2, L3, R1, R2
    prob_line = "\t" + "\t".join(["0.05"] * 20)
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "Node0\t1\tA" + prob_line + "\n"
        "Node1\t1\tA" + prob_line + "\n"
        "Node2\t1\tA" + prob_line + "\n"
        "Node3\t1\tR" + prob_line + "\n"
        "L1\t1\tA" + prob_line + "\n"
        "L2\t1\tA" + prob_line + "\n"
        "L3\t1\tA" + prob_line + "\n"
        "R1\t1\tR" + prob_line + "\n"
        "R2\t1\tR" + prob_line + "\n",
        encoding="utf-8"
    )
    
    # 5. Create alignment
    align_file = tmp_path / "trimmed.aln"
    align_file.write_text(
        ">L1\nMAP\n>L2\nMAP\n>L3\nMAP\n>R1\nRAP\n>R2\nRAP\n",
        encoding="utf-8"
    )
    
    output_csv = tmp_path / "raw_scores.csv"
    
    # Run node scoring with min_clade_size = 1
    score_tree_nodes(
        alerax_tree_path=alerax_tree_file,
        asr_tree_path=asr_tree_file,
        state_path=state_file,
        alignment_path=align_file,
        output_csv=output_csv,
        min_clade_size=1
    )
    
    assert output_csv.exists()
    
    # Read the output CSV
    df = pd.read_csv(output_csv)
    assert len(df) > 0
    # Node0 (the root) should be scored and mapped to Duplication
    event_dict = df.groupby("node_name")["event_type"].first().to_dict()
    assert event_dict.get("Node0") == "Duplication"

def test_score_tree_nodes_with_transfers_and_new_metrics(tmp_path: Path) -> None:
    # 1. Mock consensus tree
    alerax_tree_file = tmp_path / "IPR019888_trans.nwk"
    alerax_tree_file.write_text("(((L1:0.1,L2:0.1):0.1,L3:0.1):0.1,((R1:0.1,R2:0.1):0.1,(R3:0.1,R4:0.1):0.1):0.1);\n", encoding="utf-8")
    
    # 2. Samples tree
    all_dir = tmp_path / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    samples_file = all_dir / "IPR019888_trans_samples.newick"
    samples_file.write_text(
        "(((L1:0.1,L2:0.1)S:0.1,L3:0.1)S:0.1,((R1:0.1,R2:0.1)S:0.1,(R3:0.1,R4:0.1)S:0.1)T:0.1)D:0.1;\n", encoding="utf-8"
    )
    
    # 3. ASR Tree with branch lengths to check distance_from_root
    asr_tree_file = tmp_path / "asr_trans.treefile"
    asr_tree_file.write_text("(((L1:0.05,L2:0.05)Node2:0.1,L3:0.1)Node1:0.15,((R1:0.08,R2:0.08)Node4:0.1,(R3:0.08,R4:0.08)Node5:0.1)Node3:0.25)Node0:0.0;\n", encoding="utf-8")
    
    # 4. State file
    state_file = tmp_path / "asr_trans.state"
    prob_line = "\t" + "\t".join(["0.05"] * 20)
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "Node0\t1\tA" + prob_line + "\n"
        "Node1\t1\tA" + prob_line + "\n"
        "Node2\t1\tA" + prob_line + "\n"
        "Node3\t1\tR" + prob_line + "\n"
        "Node4\t1\tR" + prob_line + "\n"
        "Node5\t1\tR" + prob_line + "\n"
        "L1\t1\tA" + prob_line + "\n"
        "L2\t1\tA" + prob_line + "\n"
        "L3\t1\tA" + prob_line + "\n"
        "R1\t1\tR" + prob_line + "\n"
        "R2\t1\tR" + prob_line + "\n"
        "R3\t1\tR" + prob_line + "\n"
        "R4\t1\tR" + prob_line + "\n",
        encoding="utf-8"
    )
    
    # 5. Alignment
    align_file = tmp_path / "trimmed_trans.aln"
    align_file.write_text(
        ">L1\nMAP\n>L2\nMAP\n>L3\nMAP\n>R1\nRAP\n>R2\nRAP\n>R3\nRAP\n>R4\nRAP\n",
        encoding="utf-8"
    )
    
    output_csv = tmp_path / "raw_scores_trans.csv"
    
    # Run node scoring
    score_tree_nodes(
        alerax_tree_path=alerax_tree_file,
        asr_tree_path=asr_tree_file,
        state_path=state_file,
        alignment_path=align_file,
        output_csv=output_csv,
        min_clade_size=1
    )
    
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    
    # Assert columns exist
    assert "distance_from_root" in df.columns
    assert "clade_size_left" in df.columns
    assert "clade_size_right" in df.columns
    assert "clade_size_total" in df.columns
    
    # Check node events and sizes
    node0_rows = df[df["node_name"] == "Node0"]
    assert not node0_rows.empty
    assert node0_rows.iloc[0]["event_type"] == "Duplication"
    assert node0_rows.iloc[0]["clade_size_left"] == 3
    assert node0_rows.iloc[0]["clade_size_right"] == 4
    assert node0_rows.iloc[0]["clade_size_total"] == 7
    assert pytest.approx(node0_rows.iloc[0]["distance_from_root"]) == 0.0
    
    node3_rows = df[df["node_name"] == "Node3"]
    assert not node3_rows.empty
    assert node3_rows.iloc[0]["event_type"] == "Transfer"
    assert node3_rows.iloc[0]["clade_size_left"] == 2
    assert node3_rows.iloc[0]["clade_size_right"] == 2
    assert node3_rows.iloc[0]["clade_size_total"] == 4
    assert pytest.approx(node3_rows.iloc[0]["distance_from_root"]) == 0.25


def _reference_recent_conservation(sequences, position):
    """The pre-refactor loop implementation, kept here only as a test oracle.

    This is a literal copy of the original ``calculate_recent_conservation``
    body (before it was made to delegate to the vectorized count-based form),
    so the vectorized path can be checked against it directly rather than
    trusted on the strength of the derivation alone.
    """
    from collections import Counter as _Counter

    from src.badasp.scoring import _blosum62_pair_score

    if not sequences:
        return 0.0
    residues = [seq[position] for seq in sequences if len(seq) > position and seq[position] not in {"-", "."}]
    if not residues:
        return 0.0
    if len(residues) == 1:
        return 1.0

    counts = _Counter(residues)
    residue_types = list(counts)
    total_pairs = (len(residues) * (len(residues) - 1)) // 2
    if total_pairs == 0:
        return 0.0

    weighted_pair_sum = 0.0
    for i, aa_i in enumerate(residue_types):
        count_i = counts[aa_i]
        same_pairs = (count_i * (count_i - 1)) // 2
        if same_pairs:
            weighted_pair_sum += same_pairs * _blosum62_pair_score(aa_i, aa_i)
        for aa_j in residue_types[i + 1 :]:
            weighted_pair_sum += (count_i * counts[aa_j]) * _blosum62_pair_score(aa_i, aa_j)

    mean_score = weighted_pair_sum / float(total_pairs)
    normalized_score = (mean_score + 4.0) / 15.0
    return float(np.clip(normalized_score, 0.0, 1.0))


def test_recent_conservation_vectorized_matches_loop() -> None:
    rng = random.Random(20260731)
    alphabet = list(AA_ORDER) + [aa.lower() for aa in AA_ORDER] + ["X", "x", "-", "."]

    test_cases = []
    for _ in range(300):
        n_seqs = rng.randint(0, 10)
        length = 4
        seqs = ["".join(rng.choice(alphabet) for _ in range(length)) for _ in range(n_seqs)]
        pos = rng.randrange(length)
        test_cases.append((seqs, pos))

    # Explicit edge cases called out in the task: empty input, all-gap column,
    # single-residue column, two-residue column, and columns containing X.
    test_cases.append(([], 0))
    test_cases.append((["----", "----", "--.."], 0))  # all-gap column
    test_cases.append((["A---", "----", "----"], 0))  # single residue
    test_cases.append((["AKKT", "RKKT"], 0))  # two distinct residues
    test_cases.append((["XKKT", "XKKT", "aKKT"], 0))  # X + lowercase
    test_cases.append((["A", "R", "N", "D", "a", "r"], 0))  # all-distinct + case folding

    for seqs, pos in test_cases:
        expected = _reference_recent_conservation(seqs, pos)

        actual = calculate_recent_conservation(seqs, pos)
        assert actual == pytest.approx(expected, abs=1e-9), (seqs, pos)

        # Drive the counts-based entry point the way the array backend does:
        # a raw ASCII bincount over the column that still includes gap
        # characters, relying on calculate_recent_conservation_counts to zero
        # them out itself.
        residues_col = [s[pos] for s in seqs if len(s) > pos]
        if residues_col:
            codes = np.frombuffer("".join(residues_col).encode("ascii", "replace"), dtype=np.uint8)
            counts = np.bincount(codes, minlength=256).astype(np.float64)
        else:
            counts = np.zeros(256, dtype=np.float64)
        actual_counts = calculate_recent_conservation_counts(counts)
        assert actual_counts == pytest.approx(expected, abs=1e-9), (seqs, pos)


def test_score_tree_nodes_missing_state_raises(tmp_path: Path) -> None:
    """A node required as a scored parent's child but absent from the .state
    file must raise rather than silently defaulting its posterior to 0.0."""
    alerax_tree_file = tmp_path / "IPR019888.nwk"
    alerax_tree_file.write_text("(((L1:0.1,L2:0.1):0.1,L3:0.1):0.1,(R1:0.1,R2:0.1):0.1);\n", encoding="utf-8")

    all_dir = tmp_path / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    samples_file = all_dir / "IPR019888_samples.newick"
    samples_file.write_text(
        "(((L1:0.1,L2:0.1)S:0.1,L3:0.1)D:0.1,(R1:0.1,R2:0.1)S:0.1)D:0.1;\n", encoding="utf-8"
    )

    asr_tree_file = tmp_path / "asr.treefile"
    asr_tree_file.write_text(
        "(((L1:0.1,L2:0.1)Node2:0.1,L3:0.1)Node1:0.1,(R1:0.1,R2:0.1)Node3:0.1)Node0:0.1;\n", encoding="utf-8"
    )

    state_file = tmp_path / "asr.state"
    prob_line = "\t" + "\t".join(["0.05"] * 20)
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "Node0\t1\tA" + prob_line + "\n"
        "Node1\t1\tA" + prob_line + "\n"
        "Node2\t1\tA" + prob_line + "\n"
        "Node3\t1\tR" + prob_line + "\n"
        "L1\t1\tA" + prob_line + "\n"
        "L2\t1\tA" + prob_line + "\n"
        "L3\t1\tA" + prob_line + "\n"
        # R1's state row is deliberately omitted -- R1 is required as a scored
        # child of Node3, so this must be caught up front, not scored as 0.0.
        "R2\t1\tR" + prob_line + "\n",
        encoding="utf-8"
    )

    align_file = tmp_path / "trimmed.aln"
    align_file.write_text(
        ">L1\nMAP\n>L2\nMAP\n>L3\nMAP\n>R1\nRAP\n>R2\nRAP\n",
        encoding="utf-8"
    )

    output_csv = tmp_path / "raw_scores.csv"

    with pytest.raises(StateFileError, match="R1"):
        score_tree_nodes(
            alerax_tree_path=alerax_tree_file,
            asr_tree_path=asr_tree_file,
            state_path=state_file,
            alignment_path=align_file,
            output_csv=output_csv,
            min_clade_size=1,
        )


def test_node_naming_legacy_reproduces_collision_strict_does_not(tmp_path: Path) -> None:
    """Reproduces the fabricated-name collision bug end to end.

    The ASR tree below has one internal node with no name (as if a rooting
    tool such as MAD had invented it while arbitrarily resolving a
    polytomy): the parent of leaves RD1/RD2. Under 'legacy' node naming, a
    fresh preorder counter assigns it the name "Node3" -- which is *also*
    the real, pre-existing name of an unrelated node (the parent of leaves
    RC1/RC2) elsewhere in the same tree. This is confirmed directly below
    by walking the parsed tree with the exact same preorder/idx logic
    'legacy' uses, rather than assuming it.

    That collision is exercised three ways, matching the three name-keyed
    lookups in ``score_tree_nodes``:
      - event type: the fabricated node's own clade (RD1, RD2) is tagged
        "Speciation" in the AleRax samples tree, but under 'legacy' picks
        up "Duplication" -- the *other* colliding node's event -- because
        both write into the same ``asr_nodes_to_events["Node3"]`` slot.
      - ancestral sequence: node "GPB" scores its left child by name
        ("Node3"); under 'legacy' this silently returns the *real* Node3's
        ancestral residue ("M") for the fabricated node, which has no
        genuine ancestral-state record at all.
      - recent conservation (RC): keyed by (name, position), so it leaks
        the same way -- and here it corrupts the *legitimate* Node3's own
        outer-clade RC value too (borrowed from the unrelated RD1/RD2
        clade), showing the collision harms both sides, not just the
        fabricated node.

    Under 'strict', the fabricated node gets a guaranteed-unique name, so
    none of the above happens: its own event type is computed correctly,
    and any pair that needs its (nonexistent) ancestral sequence is skipped
    via the pre-existing empty-sequence ``continue`` instead of borrowing
    data.
    """
    alerax_tree_file = tmp_path / "alerax.nwk"
    alerax_tree_file.write_text(
        "(((RD1:0.1,RD2:0.1):0.1,GPB_sib:0.1):0.1,((RC1:0.1,RC2:0.1):0.1,GPA_sib:0.1):0.1);\n",
        encoding="utf-8",
    )

    all_dir = tmp_path / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    (all_dir / "alerax_samples.newick").write_text(
        "(((RD1:0.1,RD2:0.1)S:0.1,GPB_sib:0.1)T:0.1,((RC1:0.1,RC2:0.1)D:0.1,GPA_sib:0.1)T:0.1);\n",
        encoding="utf-8",
    )

    asr_tree_file = tmp_path / "asr.treefile"
    asr_newick = (
        "(((RD1:0.1,RD2:0.1):0.1,GPB_sib:0.1)GPB:0.1,"
        "((RC1:0.1,RC2:0.1)Node3:0.1,GPA_sib:0.1)GPA:0.1)Root:0.1;\n"
    )
    asr_tree_file.write_text(asr_newick, encoding="utf-8")

    # Confirm, rather than assume, that 'legacy' naming assigns the blank
    # node (parent of RD1/RD2) exactly the name "Node3" -- the same fresh
    # preorder/idx counter score_tree_nodes uses internally.
    check_tree = Tree(asr_newick, format=1)
    idx = 1
    legacy_name_for_blank = None
    for node in check_tree.traverse("preorder"):
        if not node.is_leaf():
            if not node.name:
                legacy_name_for_blank = f"Node{idx}"
            idx += 1
    assert legacy_name_for_blank == "Node3"

    state_file = tmp_path / "asr.state"
    prob_line = "\t" + "\t".join(["0.05"] * 20)
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "Node3\t1\tM" + prob_line + "\n"
        "GPA_sib\t1\tA" + prob_line + "\n"
        "GPB_sib\t1\tA" + prob_line + "\n"
        "RC1\t1\tM" + prob_line + "\n"
        "RC2\t1\tM" + prob_line + "\n"
        "RD1\t1\tR" + prob_line + "\n"
        "RD2\t1\tD" + prob_line + "\n",
        encoding="utf-8",
    )

    align_file = tmp_path / "trimmed.aln"
    align_file.write_text(
        ">RD1\nR\n>RD2\nD\n>GPB_sib\nA\n>RC1\nM\n>RC2\nM\n>GPA_sib\nA\n",
        encoding="utf-8",
    )

    # --- legacy: reproduces the collision ---
    legacy_csv = tmp_path / "legacy.csv"
    score_tree_nodes(
        alerax_tree_path=alerax_tree_file,
        asr_tree_path=asr_tree_file,
        state_path=state_file,
        alignment_path=align_file,
        output_csv=legacy_csv,
        min_clade_size=1,
        node_naming="legacy",
    )
    legacy_df = pd.read_csv(legacy_csv)

    # Both the fabricated node's own clade and the real Node3's clade are
    # reported under the identical name "Node3" -- the collision itself.
    assert (legacy_df["node_name"] == "Node3").sum() == 2

    fabricated_row = legacy_df[(legacy_df["left_child"] == "RD1") & (legacy_df["right_child"] == "RD2")].iloc[0]
    assert fabricated_row["node_name"] == "Node3"
    # True event (from the samples file) is Speciation; the collision makes
    # it inherit the real Node3's Duplication label instead.
    assert fabricated_row["event_type"] == "Duplication"

    gpb_row = legacy_df[legacy_df["node_name"] == "GPB"].iloc[0]
    assert gpb_row["left_child"] == "Node3"
    # The fabricated node has no ancestral-state record of its own; legacy
    # silently borrows the real Node3's residue ("M") instead of skipping.
    assert gpb_row["aa_left"] == "M"

    gpa_row = legacy_df[legacy_df["node_name"] == "GPA"].iloc[0]
    # Real Node3's own leaves (RC1, RC2) are both "M": the correct RC over
    # that 2-sequence column is 0.6 (see the strict assertion below). Under
    # legacy it is corrupted to 0.1333 -- borrowed from the unrelated
    # RD1/RD2 (R, D) column via the shared (name, position) cache key.
    assert gpa_row["rc_left"] == pytest.approx(0.1333, abs=1e-4)

    # --- strict: no leakage in any direction ---
    strict_csv = tmp_path / "strict.csv"
    score_tree_nodes(
        alerax_tree_path=alerax_tree_file,
        asr_tree_path=asr_tree_file,
        state_path=state_file,
        alignment_path=align_file,
        output_csv=strict_csv,
        min_clade_size=1,
        node_naming="strict",
    )
    strict_df = pd.read_csv(strict_csv)

    # No more name collisions: every node_name is unique.
    assert strict_df["node_name"].duplicated().sum() == 0

    strict_fabricated_row = strict_df[
        (strict_df["left_child"] == "RD1") & (strict_df["right_child"] == "RD2")
    ].iloc[0]
    assert strict_fabricated_row["node_name"] != "Node3"
    # Event type no longer leaks: correctly Speciation.
    assert strict_fabricated_row["event_type"] == "Speciation"

    # The GPB pair needed the fabricated node's (nonexistent) ancestral
    # sequence and is correctly skipped rather than scored on borrowed data.
    assert (strict_df["node_name"] == "GPB").sum() == 0

    strict_gpa_row = strict_df[strict_df["node_name"] == "GPA"].iloc[0]
    # Real Node3's own RC value is no longer corrupted.
    assert strict_gpa_row["rc_left"] == pytest.approx(0.6, abs=1e-4)


def test_node_naming_invalid_value_raises(tmp_path: Path) -> None:
    alerax_tree_file = tmp_path / "IPR019888.nwk"
    alerax_tree_file.write_text("(L1:0.1,L2:0.1);\n", encoding="utf-8")
    asr_tree_file = tmp_path / "asr.treefile"
    asr_tree_file.write_text("(L1:0.1,L2:0.1)Node0:0.1;\n", encoding="utf-8")
    state_file = tmp_path / "asr.state"
    prob_line = "\t" + "\t".join(["0.05"] * 20)
    state_file.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "L1\t1\tA" + prob_line + "\n"
        "L2\t1\tA" + prob_line + "\n",
        encoding="utf-8",
    )
    align_file = tmp_path / "trimmed.aln"
    align_file.write_text(">L1\nA\n>L2\nA\n", encoding="utf-8")

    with pytest.raises(ValueError, match="node_naming"):
        score_tree_nodes(
            alerax_tree_path=alerax_tree_file,
            asr_tree_path=asr_tree_file,
            state_path=state_file,
            alignment_path=align_file,
            output_csv=tmp_path / "out.csv",
            min_clade_size=1,
            node_naming="bogus",
        )
