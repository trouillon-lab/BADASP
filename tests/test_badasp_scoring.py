import csv
from pathlib import Path
import pytest
import pandas as pd
from Bio import SeqIO
from ete3 import Tree
from src.badasp.scoring import (
    load_state_file,
    calculate_recent_conservation,
    calculate_ancestral_conservation,
    extract_posterior_probability,
    score_tree_nodes,
)

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
    assert pytest.approx(calculate_recent_conservation(seqs, 0), 1e-4) == 0.6  # Perfect M conservation normalized
    
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
        "Node\tSite\tState\tA\tR\tN\tD\tC\tQ\tE\tG\tH\tI\tL\tK\tM\tF\tP\tS\tT\tW\tY\tV\n"
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
        "Node\tSite\tState\tA\tR\tN\tD\tC\tQ\tE\tG\tH\tI\tL\tK\tM\tF\tP\tS\tT\tW\tY\tV\n"
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
