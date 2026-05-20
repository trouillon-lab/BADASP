from pathlib import Path
import tempfile

import pandas as pd
import pytest
import numpy as np

from src.badasp_core import (
    BADASPCore,
    build_duplication_sister_pairs,
    calculate_ancestral_conservation,
    compute_multilevel_badasp_scores,
    identify_sdps,
    load_reconciliation_events,
    load_state_file,
    summarize_duplication_outputs,
)


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def duplication_alignment(temp_data_dir):
    alignment_path = temp_data_dir / "dup_alignment.aln"
    alignment_path.write_text(
        ">A1\nACDE\n"
        ">A2\nACDE\n"
        ">B1\nACDF\n"
        ">B2\nACDF\n",
        encoding="utf-8",
    )
    return alignment_path


@pytest.fixture
def duplication_assignments(temp_data_dir):
    assignments_path = temp_data_dir / "assignments.csv"
    pd.DataFrame({"sequence_id": ["A1", "A2", "B1", "B2"]}).to_csv(assignments_path, index=False)
    return assignments_path


@pytest.fixture
def duplication_tree(temp_data_dir):
    tree_path = temp_data_dir / "dup.treefile"
    tree_path.write_text(
        "((A1:0.1,A2:0.1)Node2:0.1,(B1:0.1,B2:0.1)Node3:0.1)Node1;",
        encoding="utf-8",
    )
    return tree_path


@pytest.fixture
def duplication_ancestral_sequences(temp_data_dir):
    ancestral_path = temp_data_dir / "dup_ancestors.fasta"
    ancestral_path.write_text(
        ">Node2\nACDE\n"
        ">Node3\nACDF\n",
        encoding="utf-8",
    )
    return ancestral_path


@pytest.fixture
def duplication_state_file(temp_data_dir):
    state_path = temp_data_dir / "dup.state"
    state_path.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "Node2\t1\tA\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node2\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node2\t3\tD\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node2\t4\tE\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node3\t1\tA\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node3\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node3\t3\tD\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "Node3\t4\tF\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.93\t0.01\t0.01\t0.01\t0.01\t0.01\n",
        encoding="utf-8",
    )
    return state_path


@pytest.fixture
def duplication_reconciliation_csv(temp_data_dir):
    reconciliation_path = temp_data_dir / "duplication_nodes.csv"
    pd.DataFrame(
        {
            "node_name": ["Node1", "Node2", "Node3"],
            "event_type": ["Duplication", "Speciation", "Speciation"],
        }
    ).to_csv(reconciliation_path, index=False)
    return reconciliation_path


def test_load_state_file_parses_correctly(duplication_state_file):
    state_data = load_state_file(duplication_state_file)
    assert "Node2" in state_data
    assert "Node3" in state_data


def test_load_reconciliation_events_parses_csv(duplication_reconciliation_csv):
    events = load_reconciliation_events(duplication_reconciliation_csv)
    assert events["Node1"] == "Duplication"
    assert events["Node2"] == "Speciation"


def test_calculate_ancestral_conservation_binary():
    assert calculate_ancestral_conservation("A", "A") == 1
    assert calculate_ancestral_conservation("A", "F") == -1


def test_build_duplication_sister_pairs_returns_left_right_children(duplication_tree, duplication_reconciliation_csv):
    from Bio import Phylo

    tree = Phylo.read(duplication_tree, "newick")
    events = load_reconciliation_events(duplication_reconciliation_csv)

    pairs = build_duplication_sister_pairs(tree, events, min_clade_size=1)

    assert pairs == [("Node1", "Node2", "Node3")]


def test_compute_multilevel_badasp_scores_returns_duplication_payload(
    duplication_alignment,
    duplication_assignments,
    duplication_ancestral_sequences,
    duplication_state_file,
    duplication_tree,
    duplication_reconciliation_csv,
):
    results = compute_multilevel_badasp_scores(
        alignment_path=duplication_alignment,
        assignments_path=duplication_assignments,
        ancestral_path=duplication_ancestral_sequences,
        state_path=duplication_state_file,
        tree_path=duplication_tree,
        reconciliation_csv=duplication_reconciliation_csv,
        min_clade_size=1,
    )

    assert {"duplications", "speciations", "combined", "global_layer_summary"} <= set(results.keys())
    payload = results["duplications"]
    assert set(payload.keys()) >= {"pairwise", "scores", "sdps", "threshold", "pairs", "candidate_pairs"}
    assert payload["candidate_pairs"] == 1
    assert len(payload["scores"]) == 4
    assert not payload["pairwise"].empty
    assert "Event_Type" in payload["pairwise"].columns
    assert "layer_index" in payload["pairwise"].columns
    assert "LDO_Label" in payload["pairwise"].columns
    assert "MDO_Label" in payload["pairwise"].columns


def test_identify_sdps_prefers_switch_count():
    scores_df = pd.DataFrame(
        {
            "position": [1, 2, 3, 4],
            "max_score": [0.8, 0.6, 1.2, 1.0],
            "switch_count": [2, 5, 5, 1],
            "global_threshold": [0.75, 0.75, 0.75, 0.75],
            "badasp_score": [0.8, 0.6, 1.2, 1.0],
        }
    )
    sdps, threshold = identify_sdps(scores_df)
    assert threshold == pytest.approx(0.75)
    assert set(sdps["position"]) == {1, 2, 3, 4}


def test_identify_sdps_returns_empty_when_no_switches():
    scores_df = pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.0, 0.0, 0.0],
            "switch_count": [0, 0, 0],
            "global_threshold": [0.0, 0.0, 0.0],
            "badasp_score": [0.0, 0.0, 0.0],
        }
    )
    sdps, threshold = identify_sdps(scores_df)
    assert sdps.empty
    assert threshold == 0.0


def test_summarize_duplication_outputs_matches_pooled_threshold_identity():
    pairwise_df = pd.DataFrame(
        {
            "duplication_node": ["Node1", "Node1", "Node2", "Node2", "Node3"],
            "left_child": ["A", "A", "B", "B", "C"],
            "right_child": ["D", "D", "E", "E", "F"],
            "pair": ["A-D", "A-D", "B-E", "B-E", "C-F"],
            "position": [10, 11, 10, 11, 12],
            "rc": [0.5, 0.5, 0.5, 0.5, 0.5],
            "ac": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "p_ac": [0.5, 0.5, 0.5, 0.5, 0.5],
            "score": [-1.0, 0.1, 0.2, 0.3, 1.5],
        }
    )

    score_df, sdp_df, threshold = summarize_duplication_outputs(pairwise_df=pairwise_df, aln_length=15)

    expected_threshold = float(np.percentile(pairwise_df["score"].to_numpy(dtype=float), 95))
    expected_crossings = pairwise_df[pairwise_df["score"] >= expected_threshold]
    assert threshold == pytest.approx(expected_threshold)
    assert int(expected_crossings.shape[0]) == int(sdp_df["switch_count"].sum())
    assert int((score_df["switch_count"] > 0).sum()) == int(sdp_df.shape[0])


def test_badasp_core_writes_duplication_outputs(
    duplication_alignment,
    duplication_assignments,
    duplication_ancestral_sequences,
    duplication_state_file,
    duplication_tree,
    duplication_reconciliation_csv,
    temp_data_dir,
):
    output_dir = temp_data_dir / "out"

    core = BADASPCore(
        alignment_path=duplication_alignment,
        assignments_path=duplication_assignments,
        ancestral_path=duplication_ancestral_sequences,
        state_path=duplication_state_file,
        tree_path=duplication_tree,
        min_clade_size=1,
        reconciliation_csv=duplication_reconciliation_csv,
    )

    results = core.compute_scores()
    assert {"duplications", "speciations", "combined", "global_layer_summary"} <= set(results.keys())

    sdps, thresholds = core.identify_sdps()
    assert {"duplications", "speciations", "combined"} <= set(sdps.keys())
    assert {"duplications", "speciations", "combined"} <= set(thresholds.keys())

    core.save_results(output_dir)
    assert (output_dir / "badasp_scores_duplications.csv").exists()
    assert (output_dir / "badasp_sdps_duplications.csv").exists()
    assert (output_dir / "raw_pairwise_duplications.csv").exists()


def test_save_results_writes_duplication_pairwise_columns(
    duplication_alignment,
    duplication_assignments,
    duplication_ancestral_sequences,
    duplication_state_file,
    duplication_tree,
    duplication_reconciliation_csv,
    temp_data_dir,
):
    output_dir = temp_data_dir / "out"
    core = BADASPCore(
        alignment_path=duplication_alignment,
        assignments_path=duplication_assignments,
        ancestral_path=duplication_ancestral_sequences,
        state_path=duplication_state_file,
        tree_path=duplication_tree,
        min_clade_size=1,
        reconciliation_csv=duplication_reconciliation_csv,
    )

    core.compute_scores()
    core.save_results(output_dir)

    assert (output_dir / "badasp_scores_combined.csv").exists()
    assert (output_dir / "badasp_sdps_combined.csv").exists()
    assert (output_dir / "raw_pairwise_combined.csv").exists()
    assert (output_dir / "global_layer_summary.csv").exists()
    df = pd.read_csv(output_dir / "raw_pairwise_duplications.csv")
    assert set(df["duplication_node"].unique()) == {"Node1"}
    assert set(df["left_child"].unique()) == {"Node2"}
    assert set(df["right_child"].unique()) == {"Node3"}
    assert set(df.columns) >= {
        "duplication_node",
        "left_child",
        "right_child",
        "pair",
        "position",
        "rc",
        "ac",
        "p_ac",
        "score",
        "Event_Type",
        "LDO_Label",
        "MDO_Label",
        "layer_index",
    }
    assert set(df["Event_Type"].unique()) <= {"Duplication", "Speciation", "Unknown"}


@pytest.fixture
def multitrack_alignment(temp_data_dir):
    alignment_path = temp_data_dir / "multitrack.aln"
    alignment_path.write_text(
        ">A1\nACDE\n"
        ">A2\nACDE\n"
        ">B1\nACDF\n"
        ">B2\nACDF\n"
        ">C1\nACGH\n"
        ">C2\nACGH\n"
        ">D1\nACGI\n"
        ">D2\nACGI\n",
        encoding="utf-8",
    )
    return alignment_path


@pytest.fixture
def multitrack_assignments_dir(temp_data_dir):
    assignments_root = temp_data_dir / "assignments_root"
    layer_dir = assignments_root / "layer_01"
    layer_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sequence_id": ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"],
            "cluster_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "layer_index": [1] * 8,
            "threshold": [1.0] * 8,
        }
    ).to_csv(layer_dir / "tree_cluster_assignments_layer01.csv", index=False)
    pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 4],
            "lca_node": ["N1", "N2", "N4", "N5"],
        }
    ).to_csv(layer_dir / "tree_clusters_layer01.csv", index=False)
    return assignments_root


@pytest.fixture
def multitrack_tree(temp_data_dir):
    tree_path = temp_data_dir / "multitrack.treefile"
    tree_path.write_text(
        "(((A1:0.1,A2:0.1)N1:0.1,(B1:0.1,B2:0.1)N2:0.1)N3:0.2,((C1:0.1,C2:0.1)N4:0.1,(D1:0.1,D2:0.1)N5:0.1)N6:0.2)Root;",
        encoding="utf-8",
    )
    return tree_path


@pytest.fixture
def multitrack_ancestral_sequences(temp_data_dir):
    ancestral_path = temp_data_dir / "multitrack_ancestors.fasta"
    ancestral_path.write_text(
        ">N1\nACDE\n"
        ">N2\nACDF\n"
        ">N3\nACDE\n"
        ">N4\nACGH\n"
        ">N5\nACGI\n"
        ">N6\nACGH\n",
        encoding="utf-8",
    )
    return ancestral_path


@pytest.fixture
def multitrack_state_file(temp_data_dir):
    state_path = temp_data_dir / "multitrack.state"
    rows = [
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V",
    ]
    node_states = {
        "N1": ["A", "C", "D", "E"],
        "N2": ["A", "C", "D", "F"],
        "N3": ["A", "C", "D", "E"],
        "N4": ["A", "C", "G", "H"],
        "N5": ["A", "C", "G", "I"],
        "N6": ["A", "C", "G", "H"],
    }
    for node, states in node_states.items():
        for site, state in enumerate(states, start=1):
            probs = ["0.01"] * 20
            aa_order = "ARNDCQEGHILKMFPSTWYV"
            probs[aa_order.index(state)] = "0.95"
            rows.append("\t".join([node, str(site), state, *probs]))
    state_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return state_path


@pytest.fixture
def multitrack_reconciliation_csv(temp_data_dir):
    reconciliation_path = temp_data_dir / "multitrack_reconciliation.csv"
    pd.DataFrame(
        {
            "node_name": ["N3", "N6"],
            "event_type": ["Duplication", "Speciation"],
        }
    ).to_csv(reconciliation_path, index=False)
    return reconciliation_path


def test_compute_multilevel_badasp_scores_emits_all_tracks(
    multitrack_alignment,
    multitrack_assignments_dir,
    multitrack_ancestral_sequences,
    multitrack_state_file,
    multitrack_tree,
    multitrack_reconciliation_csv,
):
    results = compute_multilevel_badasp_scores(
        alignment_path=multitrack_alignment,
        assignments_path=multitrack_assignments_dir,
        ancestral_path=multitrack_ancestral_sequences,
        state_path=multitrack_state_file,
        tree_path=multitrack_tree,
        reconciliation_csv=multitrack_reconciliation_csv,
        min_clade_size=2,
    )

    assert {"duplications", "speciations", "combined", "global_layer_summary"} <= set(results.keys())
    assert results["duplications"]["pairwise"]["Event_Type"].eq("Duplication").all()
    assert results["speciations"]["pairwise"]["Event_Type"].eq("Speciation").all()
    assert set(results["combined"]["pairwise"]["Event_Type"].unique()) == {"Duplication", "Speciation"}

    summary = results["global_layer_summary"]
    assert {"layer_index", "linkage_threshold", "number_valid_pairs", "total_duplication_sdps", "total_speciation_sdps"} <= set(summary.columns)
    assert int(summary.iloc[0]["total_duplication_sdps"]) > 0
    assert int(summary.iloc[0]["total_speciation_sdps"]) > 0


def test_badasp_core_writes_three_track_outputs_and_global_summary(
    multitrack_alignment,
    multitrack_assignments_dir,
    multitrack_ancestral_sequences,
    multitrack_state_file,
    multitrack_tree,
    multitrack_reconciliation_csv,
    temp_data_dir,
):
    output_dir = temp_data_dir / "out"

    core = BADASPCore(
        alignment_path=multitrack_alignment,
        assignments_path=multitrack_assignments_dir,
        ancestral_path=multitrack_ancestral_sequences,
        state_path=multitrack_state_file,
        tree_path=multitrack_tree,
        min_clade_size=2,
        reconciliation_csv=multitrack_reconciliation_csv,
    )

    core.compute_scores()
    core.save_results(output_dir)

    expected_root_files = [
        "badasp_scores_duplications.csv",
        "badasp_sdps_duplications.csv",
        "raw_pairwise_duplications.csv",
        "badasp_scores_speciations.csv",
        "badasp_sdps_speciations.csv",
        "raw_pairwise_speciations.csv",
        "badasp_scores_combined.csv",
        "badasp_sdps_combined.csv",
        "raw_pairwise_combined.csv",
        "global_layer_summary.csv",
    ]
    for filename in expected_root_files:
        assert (output_dir / filename).exists()

    layer_dir = output_dir / "layer_01"
    assert (layer_dir / "badasp_scores_duplications.csv").exists()
    assert (layer_dir / "badasp_scores_speciations.csv").exists()
    assert (layer_dir / "badasp_scores_combined.csv").exists()

    summary = pd.read_csv(output_dir / "global_layer_summary.csv")
    assert summary.shape[0] == 1
    assert summary.iloc[0]["total_duplication_sdps"] > 0
    assert summary.iloc[0]["total_speciation_sdps"] > 0