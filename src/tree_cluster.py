import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

try:
    from src.visualization import LEVEL_COLORS, build_terminal_color_map, plot_topological_dendrogram, plot_topological_tree_dendrogram, plot_tree_with_switches
    from src.tree_rooting import root_tree
except ModuleNotFoundError:
    from visualization import LEVEL_COLORS, build_terminal_color_map, plot_topological_dendrogram, plot_topological_tree_dendrogram, plot_tree_with_switches
    from tree_rooting import root_tree


def _compute_subtree_heights(clade: Clade) -> Dict[int, float]:
    heights: Dict[int, float] = {}

    def _walk(node: Clade) -> float:
        if node.is_terminal():
            heights[id(node)] = 0.0
            return 0.0

        child_heights: List[float] = []
        for child in node.clades:
            branch_len = child.branch_length or 0.0
            child_heights.append(branch_len + _walk(child))
        node_height = max(child_heights) if child_heights else 0.0
        heights[id(node)] = node_height
        return node_height

    _walk(clade)
    return heights


def tree_to_linkage(tree: Tree) -> Tuple[List[str], List[List[float]]]:
    leaves = tree.get_terminals()
    leaf_index = {id(leaf): i for i, leaf in enumerate(leaves)}
    labels = [leaf.name or f"leaf_{i}" for i, leaf in enumerate(leaves)]

    if len(leaves) < 2:
        raise ValueError("Need at least 2 leaves to build linkage.")

    heights = _compute_subtree_heights(tree.root)
    cluster_map: Dict[int, Tuple[int, int]] = {id(leaf): (leaf_index[id(leaf)], 1) for leaf in leaves}

    next_id = len(leaves)
    linkage_rows: List[List[float]] = []

    for node in tree.get_nonterminals(order="postorder"):
        child_clusters = [cluster_map[id(child)] for child in node.clades if id(child) in cluster_map]
        if len(child_clusters) < 2:
            continue

        node_distance = max(0.0, 2.0 * heights[id(node)])
        merged_id, merged_size = child_clusters[0]

        for child_id, child_size in child_clusters[1:]:
            total = merged_size + child_size
            linkage_rows.append([float(merged_id), float(child_id), float(node_distance), float(total)])
            merged_id = next_id
            merged_size = total
            next_id += 1

        cluster_map[id(node)] = (merged_id, merged_size)

    expected_rows = len(leaves) - 1
    if len(linkage_rows) != expected_rows:
        raise ValueError(
            f"Invalid linkage conversion. Expected {expected_rows} rows, got {len(linkage_rows)}."
        )

    return labels, linkage_rows


def _resolve_lca_label(tree: Tree, members: List[str]) -> str:
    lca = tree.common_ancestor(members)
    if lca.name:
        return lca.name

    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if node is lca:
            return f"InternalNode_{idx}"
    return "InternalNode_unknown"


def _build_level_assignments(
    labels: Sequence[str],
    linkage_rows: Sequence[Sequence[float]],
    threshold: float,
    min_clade_size: int,
) -> Dict[int, List[str]]:
    cluster_ids = [int(x) for x in fcluster(linkage_rows, t=threshold, criterion="distance")]
    assignments: Dict[int, List[str]] = {}
    for terminal_name, cluster_id in zip(labels, cluster_ids):
        assignments.setdefault(cluster_id, []).append(terminal_name)
    filtered = {cid: members for cid, members in assignments.items() if len(members) >= min_clade_size}
    return filtered


def _level_membership_map(assignments: Dict[int, List[str]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for cluster_id, members in assignments.items():
        for terminal in members:
            mapping[terminal] = cluster_id
    return mapping


def _clade_count_at_threshold(linkage_rows: Sequence[Sequence[float]], threshold: float) -> int:
    cluster_ids = fcluster(linkage_rows, t=threshold, criterion="distance")
    return len(set(int(x) for x in cluster_ids))


def _surviving_clade_count_at_threshold(
    linkage_rows: Sequence[Sequence[float]],
    threshold: float,
    min_clade_size: int,
) -> int:
    cluster_ids = [int(x) for x in fcluster(linkage_rows, t=threshold, criterion="distance")]
    sizes: Dict[int, int] = {}
    for cid in cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    return sum(1 for size in sizes.values() if size >= min_clade_size)


def _pick_even_threshold_layers(
    linkage_rows: Sequence[Sequence[float]],
    layer_count: int,
) -> List[float]:
    z = np.asarray(linkage_rows, dtype=float)
    if z.size == 0:
        return []

    unique_distances = sorted(set(float(x) for x in z[:, 2]), reverse=True)
    if not unique_distances:
        return []

    if layer_count <= 1:
        return [unique_distances[0]]

    if len(unique_distances) <= layer_count:
        return unique_distances

    raw_indices = np.linspace(0, len(unique_distances) - 1, num=layer_count)
    selected_indices = sorted(set(int(round(i)) for i in raw_indices))
    return [unique_distances[i] for i in selected_indices]


def _materialize_multithreshold_layer(
    tree: Tree,
    labels: Sequence[str],
    linkage_rows: Sequence[Sequence[float]],
    threshold: float,
    layer_index: int,
    min_clade_size: int,
    output_dir: Path,
) -> Optional[Dict[str, float]]:
    assignments = _build_level_assignments(labels, linkage_rows, threshold, min_clade_size)
    if len(assignments) < 2:
        return None

    membership = _level_membership_map(assignments)
    lca_map = {cluster_id: _resolve_lca_label(tree, members) for cluster_id, members in assignments.items()}
    assignment_path = output_dir / f"tree_cluster_assignments_layer{layer_index:02d}.csv"
    cluster_path = output_dir / f"tree_clusters_layer{layer_index:02d}.csv"

    assignment_rows = []
    for sequence_id in sorted(labels):
        cluster_id = membership.get(sequence_id)
        if cluster_id is None:
            continue
        assignment_rows.append(
            {
                "sequence_id": sequence_id,
                "cluster_id": int(cluster_id),
                "lca_node": lca_map[int(cluster_id)],
                "layer_index": int(layer_index),
                "threshold": float(threshold),
            }
        )
    pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)

    cluster_rows = [
        {
            "layer_index": int(layer_index),
            "threshold": float(threshold),
            "cluster_id": int(cluster_id),
            "member_count": int(len(members)),
            "lca_node": lca_map[int(cluster_id)],
        }
        for cluster_id, members in sorted(assignments.items(), key=lambda item: int(item[0]))
    ]
    pd.DataFrame(cluster_rows).to_csv(cluster_path, index=False)

    return {
        "layer_index": int(layer_index),
        "threshold": float(threshold),
        "cluster_count": int(len(assignments)),
        "assignments_csv": str(assignment_path),
        "clusters_csv": str(cluster_path),
    }


def write_multithreshold_cluster_artifacts(
    rooted_tree_path: Path,
    output_dir: Path,
    layer_count: int = 8,
    min_clade_size: int = 5,
    max_layer_cluster_count: int = 400,
    workers: int = 1,
) -> List[Dict[str, float]]:
    tree = Phylo.read(str(rooted_tree_path), "newick")
    labels, linkage_rows = tree_to_linkage(tree)

    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _pick_even_threshold_layers(linkage_rows, max(2, int(layer_count)))

    candidate_layers: List[Tuple[int, float]] = []
    previous_membership: Optional[Dict[str, int]] = None
    layer_index = 0
    for threshold in thresholds:
        assignments = _build_level_assignments(labels, linkage_rows, threshold, min_clade_size)
        if len(assignments) < 2:
            continue
        if len(assignments) > int(max_layer_cluster_count):
            continue

        membership = _level_membership_map(assignments)
        if previous_membership is not None and membership == previous_membership:
            continue

        layer_index += 1
        previous_membership = membership
        candidate_layers.append((layer_index, float(threshold)))

    if not candidate_layers:
        pd.DataFrame([]).to_csv(output_dir / "tree_cluster_layers.csv", index=False)
        return []

    layer_records: List[Dict[str, float]] = []
    if int(workers) > 1 and len(candidate_layers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = [
                executor.submit(
                    _materialize_multithreshold_layer,
                    tree,
                    labels,
                    linkage_rows,
                    threshold,
                    layer_idx,
                    min_clade_size,
                    output_dir,
                )
                for layer_idx, threshold in candidate_layers
            ]
            for future in futures:
                record = future.result()
                if record is not None:
                    layer_records.append(record)
    else:
        for layer_idx, threshold in candidate_layers:
            record = _materialize_multithreshold_layer(
                tree,
                labels,
                linkage_rows,
                threshold,
                layer_idx,
                min_clade_size,
                output_dir,
            )
            if record is not None:
                layer_records.append(record)

    layer_records = sorted(layer_records, key=lambda item: int(item["layer_index"]))
    pd.DataFrame(layer_records).to_csv(output_dir / "tree_cluster_layers.csv", index=False)
    return layer_records


def fast_multi_layer_clustering(
    tree_path: Path,
    layer_count: int,
    min_clade_size: int,
    output_dir: Path,
) -> List[Dict[str, float]]:
    tree = Phylo.read(str(tree_path), "newick")
    
    # Ensure all internal nodes have names
    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"

    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)

    max_depth = max(depths.values())
    thresholds = np.linspace(0, max_depth, num=int(layer_count) + 2)[1:-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_records = []
    layer_index = 0
    previous_assignments = None

    for threshold in thresholds:
        clades_at_threshold = []

        def traverse(node: Clade, parent_depth: float):
            d = depths[node]
            if parent_depth <= threshold < d:
                clades_at_threshold.append(node)
                return
            if node.is_terminal() and d <= threshold:
                clades_at_threshold.append(node)
                return
            for child in node.clades:
                traverse(child, d)

        traverse(tree.root, -1.0)

        assignments = {}
        for clade in clades_at_threshold:
            members = [leaf.name for leaf in clade.get_terminals() if leaf.name]
            if len(members) >= min_clade_size:
                assignments[id(clade)] = (clade, members)

        if len(assignments) < 2:
            continue

        sorted_clades = sorted(assignments.values(), key=lambda x: (-len(x[1]), x[0].name or ""))
        
        cluster_rows = []
        assignment_rows = []
        
        current_cluster_id = 1
        for clade, members in sorted_clades:
            lca_name = str(clade.name)
            cluster_rows.append({
                "layer_index": layer_index + 1,
                "threshold": float(threshold),
                "cluster_id": current_cluster_id,
                "member_count": len(members),
                "lca_node": lca_name,
            })
            for member in members:
                assignment_rows.append({
                    "sequence_id": member,
                    "cluster_id": current_cluster_id,
                    "lca_node": lca_name,
                    "layer_index": layer_index + 1,
                    "threshold": float(threshold),
                })
            current_cluster_id += 1

        if not cluster_rows:
            continue

        member_map = {row["sequence_id"]: row["cluster_id"] for row in assignment_rows}
        if previous_assignments is not None and member_map == previous_assignments:
            continue

        layer_index += 1
        previous_assignments = member_map

        assignment_path = output_dir / f"tree_cluster_assignments_layer{layer_index:02d}.csv"
        cluster_path = output_dir / f"tree_clusters_layer{layer_index:02d}.csv"

        pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)
        pd.DataFrame(cluster_rows).to_csv(cluster_path, index=False)

        layer_records.append({
            "layer_index": layer_index,
            "threshold": float(threshold),
            "cluster_count": len(cluster_rows),
            "assignments_csv": str(assignment_path),
            "clusters_csv": str(cluster_path),
        })

    pd.DataFrame(layer_records).to_csv(output_dir / "tree_cluster_layers.csv", index=False)
    return layer_records


def choose_distance_threshold(
    linkage_rows: Sequence[Sequence[float]],
    target_min_clades: int = 20,
    target_max_clades: int = 80,
    min_clade_size: int = 1,
    iterations: int = 30,
) -> float:
    if target_min_clades <= 0 or target_max_clades < target_min_clades:
        raise ValueError("Invalid target clade range.")

    z = np.asarray(linkage_rows, dtype=float)
    min_dist = float(np.min(z[:, 2]))
    max_dist = float(np.max(z[:, 2]))

    low = min_dist
    high = max_dist
    midpoint = (target_min_clades + target_max_clades) / 2.0

    sampled: List[Tuple[float, int]] = []

    low_count = _surviving_clade_count_at_threshold(linkage_rows, low, min_clade_size)
    high_count = _surviving_clade_count_at_threshold(linkage_rows, high, min_clade_size)
    sampled.append((low, low_count))
    sampled.append((high, high_count))

    if target_min_clades <= low_count <= target_max_clades:
        return low
    if target_min_clades <= high_count <= target_max_clades:
        return high

    for _ in range(iterations):
        mid = (low + high) / 2.0
        mid_count = _surviving_clade_count_at_threshold(linkage_rows, mid, min_clade_size)
        sampled.append((mid, mid_count))

        if target_min_clades <= mid_count <= target_max_clades:
            return mid

        if mid_count > target_max_clades:
            low = mid
        else:
            high = mid

    best_threshold, _ = min(sampled, key=lambda x: abs(x[1] - midpoint))
    return best_threshold


def cluster_tree_topologically(
    tree_path: Path,
    clusters_output: Path,
    assignments_output: Path,
    rooted_tree_output: Optional[Path] = None,
    rooting_method: str = "mad",
    group_distance_threshold: Optional[float] = None,
    family_distance_threshold: Optional[float] = None,
    subfamily_distance_threshold: Optional[float] = None,
    group_target_min_clades: int = 5,
    group_target_max_clades: int = 10,
    family_target_min_clades: int = 30,
    family_target_max_clades: int = 40,
    subfamily_target_min_clades: int = 100,
    subfamily_target_max_clades: int = 150,
    min_clade_size: int = 5,
    dendrogram_output: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    rooted_tree_path = rooted_tree_output
    if rooted_tree_path is None:
        rooted_tree_path = assignments_output.parent / f"{tree_path.stem}_{rooting_method}_rooted.tree"

    rooted_tree_path = root_tree(
        input_tree=tree_path,
        output_tree=rooted_tree_path,
        method=rooting_method,
    )

    tree = Phylo.read(str(rooted_tree_path), "newick")

    labels, linkage_rows = tree_to_linkage(tree)

    if group_distance_threshold is None:
        group_distance_threshold = choose_distance_threshold(
            linkage_rows=linkage_rows,
            target_min_clades=group_target_min_clades,
            target_max_clades=group_target_max_clades,
            min_clade_size=min_clade_size,
        )

    if family_distance_threshold is None:
        family_distance_threshold = choose_distance_threshold(
            linkage_rows=linkage_rows,
            target_min_clades=family_target_min_clades,
            target_max_clades=family_target_max_clades,
            min_clade_size=min_clade_size,
        )

    if subfamily_distance_threshold is None:
        subfamily_distance_threshold = choose_distance_threshold(
            linkage_rows=linkage_rows,
            target_min_clades=subfamily_target_min_clades,
            target_max_clades=subfamily_target_max_clades,
            min_clade_size=min_clade_size,
        )

    # Ensure deep-to-shallow ordering by distance threshold.
    ordered_thresholds = sorted(
        [group_distance_threshold, family_distance_threshold, subfamily_distance_threshold], reverse=True
    )
    group_distance_threshold, family_distance_threshold, subfamily_distance_threshold = ordered_thresholds

    group_assignments = _build_level_assignments(labels, linkage_rows, group_distance_threshold, min_clade_size)
    family_assignments = _build_level_assignments(labels, linkage_rows, family_distance_threshold, min_clade_size)
    subfamily_assignments = _build_level_assignments(labels, linkage_rows, subfamily_distance_threshold, min_clade_size)

    if not group_assignments or not family_assignments or not subfamily_assignments:
        raise ValueError("No clades survive the minimum size threshold.")

    group_map = _level_membership_map(group_assignments)
    family_map = _level_membership_map(family_assignments)
    subfamily_map = _level_membership_map(subfamily_assignments)

    # Keep sequences independently per level. A sequence may be valid for groups
    # while filtered at family/subfamily due level-specific min-clade-size rules.
    sequence_ids = sorted(set(labels))
    if not sequence_ids:
        raise ValueError("No sequences found in clustered tree labels.")

    group_lca = {cid: _resolve_lca_label(tree, members) for cid, members in group_assignments.items()}
    family_lca = {cid: _resolve_lca_label(tree, members) for cid, members in family_assignments.items()}
    subfamily_lca = {cid: _resolve_lca_label(tree, members) for cid, members in subfamily_assignments.items()}

    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with assignments_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence_id",
                "group_id",
                "group_lca_node",
                "family_id",
                "family_lca_node",
                "subfamily_id",
                "subfamily_lca_node",
            ]
        )
        for sequence_id in sequence_ids:
            gid = group_map.get(sequence_id)
            fid = family_map.get(sequence_id)
            sid = subfamily_map.get(sequence_id)
            writer.writerow(
                [
                    sequence_id,
                    gid if gid is not None else "",
                    group_lca[gid] if gid is not None else "",
                    fid if fid is not None else "",
                    family_lca[fid] if fid is not None else "",
                    sid if sid is not None else "",
                    subfamily_lca[sid] if sid is not None else "",
                ]
            )

    clusters_output.parent.mkdir(parents=True, exist_ok=True)
    with clusters_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["level", "cluster_id", "member_count", "lca_node"])
        for level_name, assignments, lca_map in (
            ("group", group_assignments, group_lca),
            ("family", family_assignments, family_lca),
            ("subfamily", subfamily_assignments, subfamily_lca),
        ):
            for cluster_id in sorted(assignments):
                writer.writerow([level_name, cluster_id, len(assignments[cluster_id]), lca_map[cluster_id]])

    if dendrogram_output is not None:
        plot_topological_tree_dendrogram(rooted_tree_path, dendrogram_output)

    level_counts = {
        "group": len(group_assignments),
        "family": len(family_assignments),
        "subfamily": len(subfamily_assignments),
    }
    level_thresholds = {
        "group": float(group_distance_threshold),
        "family": float(family_distance_threshold),
        "subfamily": float(subfamily_distance_threshold),
    }
    return level_counts, level_thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topological clustering of a phylogenetic tree.")
    parser.add_argument("--tree", default="data/interim/IPR019888.tree")
    parser.add_argument("--clusters-output", default="results/topological_clustering/tree_clusters.csv")
    parser.add_argument("--assignments-output", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--rooted-tree-output", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--rooting-method", choices=["midpoint", "mad"], default="mad")
    parser.add_argument("--group-distance-threshold", type=float, default=None)
    parser.add_argument("--family-distance-threshold", type=float, default=None)
    parser.add_argument("--subfamily-distance-threshold", type=float, default=None)
    parser.add_argument("--group-target-min-clades", type=int, default=5)
    parser.add_argument("--group-target-max-clades", type=int, default=10)
    parser.add_argument("--family-target-min-clades", type=int, default=30)
    parser.add_argument("--family-target-max-clades", type=int, default=40)
    parser.add_argument("--subfamily-target-min-clades", type=int, default=100)
    parser.add_argument("--subfamily-target-max-clades", type=int, default=150)
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--dendrogram-output", default="results/topological_clustering/tree_dendrogram.svg")
    parser.add_argument("--multi-layer", type=int, default=0, help="If >0, bypass SciPy and run fast O(N) multi-layer clustering with this many layers.")
    parser.add_argument("--max-layer-cluster-count", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def write_hierarchical_tree_artifacts(
    clusters_csv: Path,
    assignments_csv: Path,
    rooted_tree_output: Path,
    output_dir: Path,
) -> None:
    clusters_df = pd.read_csv(clusters_csv)
    assignments_df = pd.read_csv(assignments_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    level_specs = {
        "groups": "group",
        "families": "family",
        "subfamilies": "subfamily",
    }

    for plural_level, singular_level in level_specs.items():
        level_clusters = clusters_df[clusters_df["level"] == singular_level].copy()
        level_clusters.to_csv(output_dir / f"tree_clusters_{plural_level}.csv", index=False)

        level_assignments = assignments_df[
            ["sequence_id", f"{singular_level}_id", f"{singular_level}_lca_node"]
        ].copy()
        level_assignments.columns = ["sequence_id", "cluster_id", "lca_node"]
        level_assignments.to_csv(output_dir / f"tree_cluster_assignments_{plural_level}.csv", index=False)

        shutil.copyfile(rooted_tree_output, output_dir / f"midpoint_rooted_{plural_level}.tree")


def write_hierarchical_dendrograms(
    tree_path: Path,
    assignments_csv: Path,
    thresholds: Dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for plural_level, singular_level in (("groups", "group"), ("families", "family"), ("subfamilies", "subfamily")):
        terminal_colors = build_terminal_color_map(assignments_csv, f"{singular_level}_id")
        plot_topological_tree_dendrogram(
            tree_path,
            output_dir / f"tree_dendrogram_{plural_level}.svg",
            title=f"Topological Clustering Dendrogram ({plural_level.capitalize()})",
            line_color="#B0B0B0",
            terminal_colors=terminal_colors,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.multi_layer > 0:
        rooted_tree_path = Path(args.rooted_tree_output)
        if not rooted_tree_path.exists():
            rooted_tree_path = root_tree(
                input_tree=Path(args.tree),
                output_tree=rooted_tree_path,
                method=args.rooting_method,
            )
        layer_records = fast_multi_layer_clustering(
            tree_path=rooted_tree_path,
            layer_count=args.multi_layer,
            min_clade_size=args.min_clade_size,
            output_dir=Path(args.assignments_output).parent,
        )
        print(f"Saved fast O(N) multi-threshold layers: {len(layer_records)} (tree_cluster_assignments_layerXX.csv / tree_clusters_layerXX.csv)")
        return

    level_counts, level_thresholds = cluster_tree_topologically(
        tree_path=Path(args.tree),
        clusters_output=Path(args.clusters_output),
        assignments_output=Path(args.assignments_output),
        rooted_tree_output=Path(args.rooted_tree_output),
        rooting_method=args.rooting_method,
        group_distance_threshold=args.group_distance_threshold,
        family_distance_threshold=args.family_distance_threshold,
        subfamily_distance_threshold=args.subfamily_distance_threshold,
        group_target_min_clades=args.group_target_min_clades,
        group_target_max_clades=args.group_target_max_clades,
        family_target_min_clades=args.family_target_min_clades,
        family_target_max_clades=args.family_target_max_clades,
        subfamily_target_min_clades=args.subfamily_target_min_clades,
        subfamily_target_max_clades=args.subfamily_target_max_clades,
        min_clade_size=args.min_clade_size,
        dendrogram_output=Path(args.dendrogram_output),
    )

    write_hierarchical_dendrograms(
        tree_path=Path(args.rooted_tree_output),
        assignments_csv=Path(args.assignments_output),
        thresholds=level_thresholds,
        output_dir=Path(args.assignments_output).parent,
    )
    write_hierarchical_tree_artifacts(
        clusters_csv=Path(args.clusters_output),
        assignments_csv=Path(args.assignments_output),
        rooted_tree_output=Path(args.rooted_tree_output),
        output_dir=Path(args.assignments_output).parent,
    )
    print(f"Groups generated: {level_counts['group']}")
    print(f"Families generated: {level_counts['family']}")
    print(f"Subfamilies generated: {level_counts['subfamily']}")
    print(
        "Thresholds used: "
        f"group={level_thresholds['group']:.6f}, "
        f"family={level_thresholds['family']:.6f}, "
        f"subfamily={level_thresholds['subfamily']:.6f}"
    )
    print("Saved tree artifacts: tree_dendrogram_groups.svg, tree_dendrogram_families.svg, tree_dendrogram_subfamilies.svg")
    print("Saved tree CSV/tree artifacts for groups, families, and subfamilies")


if __name__ == "__main__":
    main()
