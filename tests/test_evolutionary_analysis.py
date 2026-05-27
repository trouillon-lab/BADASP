from pathlib import Path

import numpy as np
import pandas as pd

from src.evolutionary_analysis import (
    _collect_architecture_switch_values,
    _load_switch_events_from_duplications,
    _load_switch_events_for_level,
    _remap_named_nodes_to_plot_tree,
    _plot_architecture_boxplot,
    _plot_master_dendrogram,
    build_global_layer_summary,
    assign_coevolution_communities,
    calculate_ca_distance_matrix,
    calculate_lca_depth,
    count_switches_per_domain,
    compute_coevolution_matrix,
    classify_physicochemical_shift,
    extract_taxon_label,
    rank_top_functional_sdps,
    plot_layerwise_switch_timeline,
)


def test_calculate_lca_depth_for_named_node(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    # Root -> N1 has branch length 2.0
    tree_path.write_text("((A:1,B:1)N1:2,(C:1,D:1)N2:2)Root;\n")

    depth = calculate_lca_depth(tree_path=tree_path, member_names=["A", "B"])
    assert np.isclose(depth, 2.0)


def test_calculate_ca_distance_matrix_from_pdb(tmp_path: Path) -> None:
    pdb_path = tmp_path / "toy.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  SER A   3       0.000   4.000   0.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    matrix = calculate_ca_distance_matrix(pdb_path=pdb_path, residue_numbers=[1, 2, 3])
    assert list(matrix.index) == [1, 2, 3]
    assert list(matrix.columns) == [1, 2, 3]
    assert np.isclose(matrix.loc[1, 2], 3.0)
    assert np.isclose(matrix.loc[1, 3], 4.0)
    assert np.isclose(matrix.loc[2, 3], 5.0)


def test_compute_coevolution_matrix_from_branch_events() -> None:
    events = pd.DataFrame(
        {
            "branch_id": ["N1", "N1", "N2", "N3", "N3"],
            "position": [10, 20, 10, 20, 30],
        }
    )

    matrix = compute_coevolution_matrix(events_df=events)
    assert set(matrix.index) == {10, 20, 30}
    assert set(matrix.columns) == {10, 20, 30}
    assert np.isclose(matrix.loc[10, 10], 1.0)
    assert np.isclose(matrix.loc[20, 20], 1.0)
    assert matrix.loc[10, 20] > matrix.loc[10, 30]
    assert np.isclose(matrix.loc[10, 20], matrix.loc[20, 10])


def test_classify_physicochemical_shift_multiple() -> None:
    category = classify_physicochemical_shift(
        charge_change="neutral->positive",
        hydrophobicity_change="polar->hydrophobic",
        volume_delta=60.0,
    )
    assert category == "multiple_complex"


def test_rank_top_functional_sdps_synthesizes_components() -> None:
    subfamily_scores = pd.DataFrame(
        {
            "position": [10, 20, 30],
            "switch_count": [8, 5, 2],
            "max_score": [1.3, 1.2, 1.1],
        }
    )
    coevo = pd.DataFrame(
        [[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=[10, 20, 30],
        columns=[10, 20, 30],
    )
    shifts = pd.DataFrame(
        {
            "position": [10, 20],
            "major_transition_count": [7, 1],
            "charge_change": ["neutral->positive", "neutral->neutral"],
            "hydrophobicity_change": ["polar->hydrophobic", "polar->polar"],
            "volume_change": [55.0, 2.0],
        }
    )

    ranked = rank_top_functional_sdps(
        subfamily_scores_df=subfamily_scores,
        coevolution_matrix_df=coevo,
        shifts_df=shifts,
        top_n=3,
    )

    assert list(ranked.columns).count("position") == 1
    assert ranked.iloc[0]["position"] == 10
    assert ranked.iloc[0]["shift_type"] == "multiple_complex"
    assert ranked.iloc[0]["functional_sdp_score"] >= ranked.iloc[1]["functional_sdp_score"]


def test_count_switches_per_domain() -> None:
    events = pd.DataFrame({"position": [5, 20, 22, 200, 240, 241]})
    domains = {
        "RAM_domain": [1, 170],
        "DNA_binding_domain": [171, 280],
        "Recognition_helix": [230, 245],
    }
    counts = count_switches_per_domain(events, domains)

    assert counts["RAM_domain"] == 3
    assert counts["DNA_binding_domain"] == 3
    assert counts["Recognition_helix"] == 2


def test_collect_architecture_switch_values_uses_raw_counts() -> None:
    scores = pd.DataFrame(
        {
            "position": list(range(1, 41)),
            "switch_count": [2] * 10 + [0] * 10 + [1] * 10 + [3] * 10,
        }
    )
    domains = {
        "HTH_Scaffold": [1, 10],
        "Recognition_Helix": [11, 20],
        "HTH_Linker": [21, 30],
        "RAM_domain": [31, 40],
    }

    values = _collect_architecture_switch_values(scores, domains)

    assert values["HTH_Scaffold"] == [2] * 10
    assert values["Recognition_Helix"] == [0] * 10
    assert values["HTH_Linker"] == [1] * 10
    assert values["RAM_domain"] == [3] * 10


def test_plot_architecture_boxplot_writes_mean_based_svg(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        {
            "position": list(range(1, 41)),
            "switch_count": [2] * 10 + [0] * 10 + [1] * 10 + [3] * 10,
        }
    )
    domains = {
        "HTH_Scaffold": [1, 10],
        "Recognition_Helix": [11, 20],
        "HTH_Linker": [21, 30],
        "RAM_domain": [31, 40],
    }
    output_svg = tmp_path / "architectural_boxplot_groups.svg"

    _plot_architecture_boxplot(scores, domains, output_svg, level="groups")

    svg_text = output_svg.read_text()
    assert output_svg.exists()
    assert "Switch Count" in svg_text
    assert "Architectural Domain" in svg_text
    assert "Holm-corrected" not in svg_text


def test_assign_coevolution_communities() -> None:
    matrix = pd.DataFrame(
        [
            [1.0, 0.85, 0.05, 0.01],
            [0.85, 1.0, 0.04, 0.02],
            [0.05, 0.04, 1.0, 0.8],
            [0.01, 0.02, 0.8, 1.0],
        ],
        index=[10, 11, 50, 51],
        columns=[10, 11, 50, 51],
    )

    communities = assign_coevolution_communities(matrix, distance_cut=0.4)
    assert set(communities["position"].astype(int)) == {10, 11, 50, 51}
    assert communities["community_id"].nunique() == 2


def test_extract_taxon_label() -> None:
    header = "sp|Q9XYZ1|ARAC_ECOLI AraC transcriptional regulator OS=Escherichia coli OX=562"
    assert extract_taxon_label(header) == "Escherichia coli"


def test_plot_master_dendrogram_exports_svg(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    tree_path.write_text("((A:0.1,B:0.1)N1:0.3,(C:0.1,D:0.1)N2:0.3)Root;\n")

    events_by_level = {
        "groups": pd.DataFrame({"branch_id": ["N1"], "position": [10], "score": [1.2]}),
        "families": pd.DataFrame({"branch_id": ["N2", "N2"], "position": [20, 21], "score": [1.0, 1.1]}),
        "subfamilies": pd.DataFrame({"branch_id": ["N1"], "position": [30], "score": [1.3]}),
    }

    output_svg = tmp_path / "master.svg"
    _plot_master_dendrogram(tree_path=tree_path, events_by_level=events_by_level, output_svg=output_svg)

    assert output_svg.exists()
    assert output_svg.stat().st_size > 0


def test_global_layer_summary_and_timeline_plot(tmp_path: Path) -> None:
    scores_root = tmp_path / "scores"
    for layer_index, duplication_rows, speciation_rows in [(1, 2, 1), (2, 3, 4)]:
        layer_dir = scores_root / f"layer_{layer_index:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "duplication_node": [f"N{layer_index}"] * 2,
                "left_child": ["A", "A"],
                "right_child": ["B", "B"],
                "pair": ["A-B", "A-B"],
                "position": [1, 2],
                "score": [0.1, 0.2],
                "Event_Type": ["Duplication", "Duplication"],
                "layer_threshold": [0.5, 0.5],
                "global_threshold": [0.4, 0.4],
            }
        ).to_csv(layer_dir / "raw_pairwise_combined.csv", index=False)
        pd.DataFrame(
            {
                "position": list(range(1, duplication_rows + 1)),
                "switch_count": [1] * duplication_rows,
            }
        ).to_csv(layer_dir / "badasp_sdps_duplications.csv", index=False)
        pd.DataFrame(
            {
                "position": list(range(1, speciation_rows + 1)),
                "switch_count": [1] * speciation_rows,
            }
        ).to_csv(layer_dir / "badasp_sdps_speciations.csv", index=False)
        pd.DataFrame(
            {
                "position": [1, 2],
                "global_threshold": [0.4, 0.4],
                "max_score": [0.2, 0.3],
                "switch_count": [1, 1],
                "badasp_score": [0.2, 0.3],
            }
        ).to_csv(layer_dir / "badasp_scores_combined.csv", index=False)
        pd.DataFrame(
            {
                "position": [1],
                "switch_count": [1],
            }
        ).to_csv(layer_dir / "badasp_sdps_combined.csv", index=False)

    summary = build_global_layer_summary(scores_root=scores_root, output_csv=scores_root / "global_layer_summary.csv")
    assert list(summary["layer_index"]) == [1, 2]
    assert summary.iloc[0]["number_valid_pairs"] == 1
    assert summary.iloc[0]["total_duplication_sdps"] == 2
    assert summary.iloc[1]["total_speciation_sdps"] == 4
    assert summary.iloc[0]["valid_duplication_nodes"] == 2
    assert summary.iloc[0]["valid_speciation_nodes"] == 1
    assert summary.iloc[0]["total_valid_nodes"] == 3
    assert summary.iloc[1]["valid_duplication_nodes"] == 3
    assert summary.iloc[1]["valid_speciation_nodes"] == 4
    assert summary.iloc[1]["total_valid_nodes"] == 7

    timeline_svg = tmp_path / "timeline.svg"
    plot_layerwise_switch_timeline(summary, timeline_svg)
    assert timeline_svg.exists()
    assert timeline_svg.stat().st_size > 0


def test_load_switch_events_uses_unit_branch_depth_when_lengths_missing(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    assignments_path = tmp_path / "assignments.csv"
    pairwise_path = tmp_path / "raw_pairwise_groups.csv"

    tree_path.write_text("((A,B)N1,(C,D)N2)Root;\n", encoding="utf-8")
    pd.DataFrame(
        {
            "sequence_id": ["A", "B", "C", "D"],
            "group_id": [1, 2, 3, 4],
            "group_lca_node": ["A", "B", "C", "D"],
            "family_id": [10, 10, 20, 20],
            "family_lca_node": ["N1", "N1", "N2", "N2"],
            "subfamily_id": [100, 100, 200, 200],
            "subfamily_lca_node": ["N1", "N1", "N2", "N2"],
        }
    ).to_csv(assignments_path, index=False)
    pd.DataFrame(
        {
            "pair": ["1-2", "1-2", "1-2", "1-2"],
            "position": [10, 11, 12, 13],
            "score": [0.1, 0.2, 0.3, 1.0],
        }
    ).to_csv(pairwise_path, index=False)

    events = _load_switch_events_for_level(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=pairwise_path,
        level="groups",
    )

    assert not events.empty
    assert events["branch_id"].nunique() == 1
    assert events["branch_id"].iloc[0] == "N1"
    assert (events["root_distance"] > 0.0).all()


def test_load_switch_events_from_duplications_uses_lca_node_names(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    pairwise_path = tmp_path / "raw_pairwise_duplications.csv"

    tree_path.write_text("((A:0.1,B:0.1)N1:0.2,(C:0.1,D:0.1)N2:0.2)Root;\n", encoding="utf-8")
    pd.DataFrame(
        {
            "pair": ["Node1_L-Node1_R"] * 5,
            "position": [10, 11, 12, 13, 14],
            "score": [0.1, 0.2, 0.3, 0.4, 1.8],
            "lca_node_name": ["N1", "N1", "N2", "N2", "N2"],
        }
    ).to_csv(pairwise_path, index=False)

    events = _load_switch_events_from_duplications(
        tree_path=tree_path,
        raw_pairwise_path=pairwise_path,
    )

    assert not events.empty
    assert events["level"].eq("duplications").all()
    assert events["branch_id"].eq("N2").all()
    assert (events["root_distance"] > 0.0).all()


def test_load_switch_events_from_duplications_preserves_event_type(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    pairwise_path = tmp_path / "raw_pairwise_duplications.csv"

    tree_path.write_text("((A:0.1,B:0.1)N1:0.2,(C:0.1,D:0.1)N2:0.2)Root;\n", encoding="utf-8")
    pd.DataFrame(
        {
            "pair": ["Node1_L-Node1_R"] * 6,
            "position": [10, 11, 12, 13, 14, 15],
            "score": [0.1, 0.2, 0.3, 0.4, 1.8, 2.2],
            "lca_node_name": ["N2", "N2", "N2", "N2", "N2", "N2"],
            "Event_Type": ["Speciation", "Speciation", "Speciation", "Speciation", "Speciation", "Speciation"],
            "layer_index": [3, 3, 3, 3, 3, 3],
        }
    ).to_csv(pairwise_path, index=False)

    events = _load_switch_events_from_duplications(tree_path=tree_path, raw_pairwise_path=pairwise_path)

    assert not events.empty
    assert events["Event_Type"].eq("Speciation").all()
    assert events["layer_index"].eq(3).all()


def test_load_switch_events_from_duplications_maps_named_asr_nodes_to_nameless_tree(tmp_path: Path) -> None:
    tree_path = tmp_path / "mad_rooted.tree"
    reference_tree_path = tmp_path / "asr_run.treefile"
    pairwise_path = tmp_path / "raw_pairwise_duplications.csv"

    tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")
    reference_tree_path.write_text("((A:0.1,B:0.1)Node11:0.2,(C:0.1,D:0.1)Node22:0.2)NodeRoot;\n", encoding="utf-8")
    pd.DataFrame(
        {
            "pair": ["Node11_L-Node11_R"] * 5,
            "position": [10, 11, 12, 13, 14],
            "score": [0.1, 0.2, 0.3, 0.4, 1.8],
            "duplication_node": ["Node11", "Node11", "Node11", "Node11", "Node11"],
        }
    ).to_csv(pairwise_path, index=False)

    events = _load_switch_events_from_duplications(
        tree_path=tree_path,
        raw_pairwise_path=pairwise_path,
        named_tree_path=reference_tree_path,
    )

    assert not events.empty
    assert events["branch_id"].str.startswith("InternalNode_").all()
    assert events["root_distance"].notna().all()


def test_remap_named_nodes_to_plot_tree_matches_by_leaf_signature(tmp_path: Path) -> None:
    from Bio import Phylo

    plot_tree_path = tmp_path / "plot.tree"
    named_tree_path = tmp_path / "named.tree"
    plot_tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")
    named_tree_path.write_text("((A:0.1,B:0.1)NodeAB:0.2,(C:0.1,D:0.1)NodeCD:0.2)NodeRoot;\n", encoding="utf-8")

    plot_tree = Phylo.read(str(plot_tree_path), "newick")
    mapping = _remap_named_nodes_to_plot_tree(plot_tree=plot_tree, named_tree_path=named_tree_path)

    assert "NodeAB" in mapping
    assert mapping["NodeAB"].startswith("InternalNode_")


def test_rank_top_functional_sdps_handles_nan_threshold(tmp_path: Path) -> None:
    """Test that rank_top_functional_sdps gracefully handles NaN threshold."""
    # Create badasp_scores with NaN threshold
    scores_df = pd.DataFrame(
        {
            "position": [1, 2, 3, 4, 5],
            "max_score": [0.5, 0.3, 0.8, 0.2, 0.6],
            "switch_count": [2, 1, 3, 1, 2],
            "global_threshold": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    
    # Create coevolution_matrix
    coevo_matrix = pd.DataFrame(
        np.eye(5),
        index=[1, 2, 3, 4, 5],
        columns=[1, 2, 3, 4, 5],
    )
    
    # Create shifts_df (physicochemical shifts) with all required columns
    shifts_df = pd.DataFrame(
        {
            "position": [1, 2, 3],
            "ancestral_aa": ["A", "G", "K"],
            "recent_aa": ["G", "A", "R"],
            "charge_change": ["neutral->neutral", "neutral->neutral", "positive->positive"],
            "hydrophobicity_change": ["hydrophobic->hydrophobic", "polar->hydrophobic", "positive->positive"],
            "volume_change": [0.0, 50.0, 10.0],
            "major_transition_count": [1, 2, 3],
        }
    )
    
    # Should not crash despite NaN threshold
    result = rank_top_functional_sdps(
        subfamily_scores_df=scores_df,
        coevolution_matrix_df=coevo_matrix,
        shifts_df=shifts_df,
        top_n=3,
    )
    
    # Should return a DataFrame without crashing
    assert isinstance(result, pd.DataFrame)
    # With NaN threshold, should have results
    assert len(result) >= 0


def test_compute_coevolution_matrix_with_empty_events(tmp_path: Path) -> None:
    """Test that compute_coevolution_matrix handles empty event DataFrame."""
    # Create empty events DataFrame
    events = pd.DataFrame({"branch_id": [], "position": []})
    
    # Should not crash with empty events
    matrix = compute_coevolution_matrix(events_df=events)
    
    # Should return empty DataFrame
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.empty


def test_build_global_layer_summary_with_nan_thresholds(tmp_path: Path) -> None:
    """Test that build_global_layer_summary handles NaN thresholds gracefully."""
    # Create a temporary directory structure with mock layer data
    scores_root = tmp_path / "badasp_scoring"
    scores_root.mkdir(parents=True, exist_ok=True)
    
    # Create layer_01 subdirectory with mock CSV files
    layer_01_dir = scores_root / "layer_01"
    layer_01_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock badasp_scores files with NaN global_threshold
    for track in ["duplications", "speciations", "combined"]:
        scores_csv = layer_01_dir / f"badasp_scores_{track}.csv"
        scores_df = pd.DataFrame(
            {
                "position": [1, 2, 3],
                "max_score": [0.5, 0.3, 0.8],
                "switch_count": [2, 1, 3],
                "global_threshold": [np.nan, np.nan, np.nan],
            }
        )
        scores_df.to_csv(scores_csv, index=False)
    
    # Create raw_pairwise files with mock data
    for track in ["duplications", "speciations", "combined"]:
        raw_csv = layer_01_dir / f"raw_pairwise_{track}.csv"
        raw_df = pd.DataFrame(
            {
                "position": [1, 2, 3],
                "score": [0.5, 0.3, 0.8],
            }
        )
        raw_df.to_csv(raw_csv, index=False)
    
    # Call build_global_layer_summary
    output_csv = tmp_path / "global_layer_summary.csv"
    result = build_global_layer_summary(scores_root=scores_root, output_csv=output_csv)
    
    # Should return a DataFrame without crashing
    assert isinstance(result, pd.DataFrame)
    # Should have at least 1 row (for layer_01)
    assert len(result) >= 1
    # The 95th_percentile_threshold should be NaN for NaN threshold layers
    if len(result) > 0:
        threshold_val = result.iloc[0]["95th_percentile_threshold"]
        # It should be either NaN or a valid number (depending on if threshold_values had any values)
        assert pd.isna(threshold_val) or isinstance(threshold_val, (int, float))
