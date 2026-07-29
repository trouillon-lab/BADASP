"""Unit tests for comparing initial vs rec_check AleRax runs."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from ete3 import Tree

from scripts.compare_rec_check_runs import (
    load_level1_sample_stats,
    load_level2_consensus_events,
    load_level3_scored_events,
    compare_node_classifications,
    generate_comparison_summary,
)


@pytest.fixture
def dummy_run_dirs():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Setup run A
        run_a = root / "run_a"
        reconc_a = run_a / "reconciliations"
        all_a = reconc_a / "all"
        all_a.mkdir(parents=True)
        
        # Write dummy event count files
        (all_a / "TEST_eventCounts_0.txt").write_text("S:40\nD:10\nT:50\n")
        (all_a / "TEST_eventCounts_1.txt").write_text("S:42\nD:8\nT:50\n")
        
        # Write dummy tree
        tree_a_str = "((seq1:0.1,seq2:0.1)N1:0.2,(seq3:0.1,seq4:0.1)N2:0.2)Root:0.1;"
        tree_a_path = reconc_a / "TEST.nwk"
        tree_a_path.write_text(tree_a_str)
        
        # Write dummy samples tree
        samples_a_str = "((seq1:0.1,seq2:0.1)S:0.2,(seq3:0.1,seq4:0.1)D:0.2)S:0.1;\n((seq1:0.1,seq2:0.1)S:0.2,(seq3:0.1,seq4:0.1)D:0.2)S:0.1;\n"
        (all_a / "TEST_samples.newick").write_text(samples_a_str)
        
        # Write dummy scores CSV
        df_scores_a = pd.DataFrame([
            {"node_name": "N1", "event_type": "Speciation", "site": 1, "score": 1.5},
            {"node_name": "N2", "event_type": "Duplication", "site": 1, "score": 1.2},
        ])
        scores_a_path = run_a / "raw_node_scores.csv"
        df_scores_a.to_csv(scores_a_path, index=False)

        # Setup run B (with seq1, seq2, seq3, seq5 - partial overlap)
        run_b = root / "run_b"
        reconc_b = run_b / "reconciliations"
        all_b = reconc_b / "all"
        all_b.mkdir(parents=True)
        
        (all_b / "TEST_eventCounts_0.txt").write_text("S:35\nD:15\nT:50\n")
        
        tree_b_str = "((seq1:0.1,seq2:0.1)N1:0.2,(seq3:0.1,seq5:0.1)N3:0.2)Root:0.1;"
        tree_b_path = reconc_b / "TEST.nwk"
        tree_b_path.write_text(tree_b_str)
        
        samples_b_str = "((seq1:0.1,seq2:0.1)S:0.2,(seq3:0.1,seq5:0.1)T:0.2)S:0.1;\n"
        (all_b / "TEST_samples.newick").write_text(samples_b_str)
        
        df_scores_b = pd.DataFrame([
            {"node_name": "N1", "event_type": "Speciation", "site": 1, "score": 1.4},
            {"node_name": "N3", "event_type": "Transfer", "site": 1, "score": 1.1},
        ])
        scores_b_path = run_b / "raw_node_scores.csv"
        df_scores_b.to_csv(scores_b_path, index=False)

        yield {
            "family": "TEST",
            "run_a": run_a,
            "reconc_a": reconc_a,
            "scores_a": scores_a_path,
            "run_b": run_b,
            "reconc_b": reconc_b,
            "scores_b": scores_b_path,
        }


def test_load_level1_sample_stats(dummy_run_dirs):
    stats = load_level1_sample_stats(dummy_run_dirs["reconc_a"] / "all", dummy_run_dirs["family"])
    assert stats["num_samples"] == 2
    assert stats["mean_s"] == 41.0
    assert stats["mean_d"] == 9.0
    assert stats["mean_t"] == 50.0
    assert stats["total_classified"] == 100.0


def test_load_level2_consensus_events(dummy_run_dirs):
    reconc_dir = dummy_run_dirs["reconc_a"]
    events, counts = load_level2_consensus_events(reconc_dir / "TEST.nwk", reconc_dir / "all" / "TEST_samples.newick")
    
    sig_12 = ("seq1", "seq2")
    sig_34 = ("seq3", "seq4")
    
    assert sig_12 in events
    assert events[sig_12] == "Speciation"
    assert events[sig_34] == "Duplication"
    assert counts["Speciation"] == 2  # Root + N1
    assert counts["Duplication"] == 1  # N2


def test_load_level3_scored_events(dummy_run_dirs):
    scores_path = dummy_run_dirs["scores_a"]
    counts = load_level3_scored_events(scores_path)
    assert counts["Speciation"] == 1
    assert counts["Duplication"] == 1
    assert counts["Transfer"] == 0


def test_compare_node_classifications(dummy_run_dirs):
    reconc_a = dummy_run_dirs["reconc_a"]
    reconc_b = dummy_run_dirs["reconc_b"]
    
    events_a, _ = load_level2_consensus_events(reconc_a / "TEST.nwk", reconc_a / "all" / "TEST_samples.newick")
    events_b, _ = load_level2_consensus_events(reconc_b / "TEST.nwk", reconc_b / "all" / "TEST_samples.newick")
    
    comp = compare_node_classifications(events_a, events_b)
    
    # Common signature: ("seq1", "seq2")
    assert comp["num_common_nodes"] == 1
    assert comp["num_matching_classifications"] == 1
    assert comp["concordance_rate"] == 100.0
    assert ("Speciation", "Speciation") in comp["confusion_matrix"]
    assert comp["confusion_matrix"][("Speciation", "Speciation")] == 1


def test_generate_comparison_summary(dummy_run_dirs):
    stats_a = load_level1_sample_stats(dummy_run_dirs["reconc_a"] / "all", dummy_run_dirs["family"])
    stats_b = load_level1_sample_stats(dummy_run_dirs["reconc_b"] / "all", dummy_run_dirs["family"])
    
    events_a, counts_a2 = load_level2_consensus_events(dummy_run_dirs["reconc_a"] / "TEST.nwk", dummy_run_dirs["reconc_a"] / "all" / "TEST_samples.newick")
    events_b, counts_b2 = load_level2_consensus_events(dummy_run_dirs["reconc_b"] / "TEST.nwk", dummy_run_dirs["reconc_b"] / "all" / "TEST_samples.newick")
    
    node_comp = compare_node_classifications(events_a, events_b)
    
    report = generate_comparison_summary(
        family="TEST",
        init_stats={"l1": stats_a, "l2": counts_a2, "l3": load_level3_scored_events(dummy_run_dirs["scores_a"])},
        rec_stats={"l1": stats_b, "l2": counts_b2, "l3": load_level3_scored_events(dummy_run_dirs["scores_b"])},
        node_comp=node_comp
    )
    
    assert "# AleRax Run Comparison" in report
    assert "Concordance Rate" in report
