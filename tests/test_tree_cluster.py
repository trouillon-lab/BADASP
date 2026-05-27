from pathlib import Path
import csv
import shutil

import pandas as pd
from Bio import Phylo
import pytest
from scipy.cluster.hierarchy import fcluster, linkage

from src.tree_cluster import build_parser, choose_distance_threshold, cluster_tree_topologically


def test_choose_distance_threshold_targets_reasonable_clade_range() -> None:
    points = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [10.0], [10.2], [10.3], [15.0], [15.1], [15.2]]
    linkage_matrix = linkage(points, method="average")
    linkage_rows = linkage_matrix.tolist()

    threshold = choose_distance_threshold(
        linkage_rows,
        target_min_clades=3,
        target_max_clades=6,
        min_clade_size=1,
    )
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")
    clade_count = len(set(int(x) for x in cluster_ids))

    assert 3 <= clade_count <= 6


def test_cluster_tree_topologically_writes_multilevel_assignments(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"
    rooted_tree_out = tmp_path / "midpoint_rooted.tree"

    tree_path.write_text(
        "(((A:0.05,B:0.05):0.2,(C:0.05,D:0.05):0.2):0.4,((E:0.05,F:0.05):0.2,(G:0.05,H:0.05):0.2):0.4);\n",
        encoding="utf-8",
    )

    level_counts, level_thresholds = cluster_tree_topologically(
        tree_path=tree_path,
        clusters_output=clusters_csv,
        assignments_output=assignments_csv,
        min_clade_size=1,
        rooted_tree_output=rooted_tree_out,
        rooting_method="midpoint",
        group_target_min_clades=2,
        group_target_max_clades=3,
        family_target_min_clades=3,
        family_target_max_clades=5,
        subfamily_target_min_clades=4,
        subfamily_target_max_clades=8,
    )

    assert clusters_csv.exists()
    assert assignments_csv.exists()
    assert rooted_tree_out.exists()
    parsed_tree = Phylo.read(str(rooted_tree_out), "newick")
    assert parsed_tree.root is not None

    assert set(level_counts) == {"group", "family", "subfamily"}
    assert set(level_thresholds) == {"group", "family", "subfamily"}

    # Coarse-to-fine hierarchy should produce non-decreasing cluster counts.
    assert level_counts["group"] <= level_counts["family"] <= level_counts["subfamily"]

    assignments = pd.read_csv(assignments_csv)
    expected_cols = {
        "sequence_id",
        "group_id",
        "group_lca_node",
        "family_id",
        "family_lca_node",
        "subfamily_id",
        "subfamily_lca_node",
    }
    assert set(assignments.columns) == expected_cols
    assert len(assignments) == 8

    # Every sequence should map through all three hierarchy levels.
    assert assignments["group_id"].notna().all()
    assert assignments["family_id"].notna().all()
    assert assignments["subfamily_id"].notna().all()

    # Nesting rules: one subfamily -> one family, one family -> one group.
    sub_to_family = assignments.groupby("subfamily_id")["family_id"].nunique()
    fam_to_group = assignments.groupby("family_id")["group_id"].nunique()
    assert (sub_to_family == 1).all()
    assert (fam_to_group == 1).all()


def test_cluster_tree_topologically_filters_small_clades(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"

    tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No clades survive"):
        cluster_tree_topologically(
            tree_path=tree_path,
            clusters_output=clusters_csv,
            assignments_output=assignments_csv,
            min_clade_size=5,
            rooting_method="midpoint",
            group_target_min_clades=2,
            group_target_max_clades=2,
            family_target_min_clades=3,
            family_target_max_clades=3,
            subfamily_target_min_clades=4,
            subfamily_target_max_clades=4,
        )


def test_write_hierarchical_tree_artifacts_writes_level_specific_csv_and_tree_files(tmp_path: Path) -> None:
    from src.tree_cluster import write_hierarchical_tree_artifacts

    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"
    rooted_tree = tmp_path / "midpoint_rooted.tree"
    output_dir = tmp_path / "level_artifacts"

    clusters_csv.write_text(
        "level,cluster_id,member_count,lca_node\n"
        "group,1,4,G1\n"
        "family,10,2,F10\n"
        "subfamily,100,1,S100\n",
        encoding="utf-8",
    )
    assignments_csv.write_text(
        "sequence_id,group_id,group_lca_node,family_id,family_lca_node,subfamily_id,subfamily_lca_node\n"
        "A,1,G1,10,F10,100,S100\n",
        encoding="utf-8",
    )
    rooted_tree.write_text("(A:0.1,B:0.1)Root;\n", encoding="utf-8")

    write_hierarchical_tree_artifacts(
        clusters_csv=clusters_csv,
        assignments_csv=assignments_csv,
        rooted_tree_output=rooted_tree,
        output_dir=output_dir,
    )

    expected_files = [
        output_dir / "tree_clusters_groups.csv",
        output_dir / "tree_clusters_families.csv",
        output_dir / "tree_clusters_subfamilies.csv",
        output_dir / "tree_cluster_assignments_groups.csv",
        output_dir / "tree_cluster_assignments_families.csv",
        output_dir / "tree_cluster_assignments_subfamilies.csv",
        output_dir / "midpoint_rooted_groups.tree",
        output_dir / "midpoint_rooted_families.tree",
        output_dir / "midpoint_rooted_subfamilies.tree",
    ]
    for path in expected_files:
        assert path.exists()
        assert path.stat().st_size > 0

    groups_csv = pd.read_csv(output_dir / "tree_clusters_groups.csv")
    assert list(groups_csv["level"]) == ["group"]


def test_cluster_tree_topologically_keeps_group_members_even_if_family_filtered(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"

    tree_path.write_text(
        "(((A:0.01,B:0.01):0.01,C:0.45):0.5,((D:0.01,E:0.01):0.01,F:0.45):0.5);\n",
        encoding="utf-8",
    )

    cluster_tree_topologically(
        tree_path=tree_path,
        clusters_output=clusters_csv,
        assignments_output=assignments_csv,
        min_clade_size=2,
        rooting_method="midpoint",
        group_distance_threshold=1.0,
        family_distance_threshold=0.3,
        subfamily_distance_threshold=0.02,
    )

    assignments = pd.read_csv(assignments_csv)
    by_seq = assignments.set_index("sequence_id")

    # C and F belong to valid groups of size 3 but fall into filtered family/subfamily singletons.
    assert pd.notna(by_seq.loc["C", "group_id"])
    assert pd.notna(by_seq.loc["F", "group_id"])
    assert pd.isna(by_seq.loc["C", "family_id"])
    assert pd.isna(by_seq.loc["F", "family_id"])


def test_build_parser_defaults_rooting_method_to_mad() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.rooting_method == "mad"


def test_cluster_tree_topologically_uses_mad_rooting_when_requested(monkeypatch, tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"
    rooted_tree_out = tmp_path / "mad_rooted.tree"

    tree_path.write_text(
        "(((A:0.05,B:0.05):0.2,(C:0.05,D:0.05):0.2):0.4,((E:0.05,F:0.05):0.2,(G:0.05,H:0.05):0.2):0.4);\n",
        encoding="utf-8",
    )

    calls = []

    def _mock_root_tree(input_tree, output_tree, method):
        calls.append((input_tree, output_tree, method))
        shutil.copyfile(input_tree, output_tree)
        return output_tree

    monkeypatch.setattr("src.tree_cluster.root_tree", _mock_root_tree)

    level_counts, _ = cluster_tree_topologically(
        tree_path=tree_path,
        clusters_output=clusters_csv,
        assignments_output=assignments_csv,
        rooted_tree_output=rooted_tree_out,
        min_clade_size=1,
        rooting_method="mad",
        group_target_min_clades=2,
        group_target_max_clades=3,
        family_target_min_clades=3,
        family_target_max_clades=5,
        subfamily_target_min_clades=4,
        subfamily_target_max_clades=8,
    )

    assert calls
    assert calls[0][2] == "mad"
    assert rooted_tree_out.exists()
    assert level_counts["group"] > 0


def test_write_multithreshold_cluster_artifacts_writes_layered_outputs(tmp_path: Path) -> None:
    from src.tree_cluster import write_multithreshold_cluster_artifacts

    rooted_tree = tmp_path / "rooted.tree"
    rooted_tree.write_text(
        "(((A:0.05,B:0.05):0.2,(C:0.05,D:0.05):0.2):0.4,((E:0.05,F:0.05):0.2,(G:0.05,H:0.05):0.2):0.4);\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "layers"
    layer_records = write_multithreshold_cluster_artifacts(
        rooted_tree_path=rooted_tree,
        output_dir=output_dir,
        layer_count=5,
        min_clade_size=1,
    )

    assert layer_records
    assert len(layer_records) >= 2
    assert (output_dir / "tree_cluster_layers.csv").exists()

    assignments_files = sorted(output_dir.glob("tree_cluster_assignments_layer*.csv"))
    clusters_files = sorted(output_dir.glob("tree_clusters_layer*.csv"))
    assert len(assignments_files) == len(clusters_files) == len(layer_records)

    sample_assignments = pd.read_csv(assignments_files[-1])
    assert set(sample_assignments.columns) == {"sequence_id", "cluster_id", "lca_node", "layer_index", "threshold"}
    assert sample_assignments["cluster_id"].nunique() >= 2


def test_build_parser_exposes_multi_threshold_layer_count() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.multi_threshold_layers == 8