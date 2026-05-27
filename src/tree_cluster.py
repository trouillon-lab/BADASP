import matplotlib
matplotlib.use('Agg')
import argparse
import csv
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Global profiling state
_PROFILE_DATA = {
    "resolve_lca_calls": 0,
    "resolve_lca_total_time": 0.0,
    "build_level_assignments_calls": 0,
    "build_level_assignments_total_time": 0.0,
}
# LCA cache to avoid repeated common_ancestor work
_LCA_CACHE: Dict[frozenset, str] = {}

try:
    from src.visualization import (
        LEVEL_COLORS,
        build_terminal_cluster_map,
        build_terminal_color_map,
        plot_topological_dendrogram,
        plot_topological_tree_dendrogram,
        plot_tree_with_switches,
    )
    from src.tree_rooting import root_tree
except ModuleNotFoundError:
    from visualization import (
        LEVEL_COLORS,
        build_terminal_cluster_map,
        build_terminal_color_map,
        plot_topological_dendrogram,
        plot_topological_tree_dendrogram,
        plot_tree_with_switches,
    )
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


def _resolve_lca_label(
    tree: Tree,
    members: List[str],
    name_to_clade: Optional[Dict[str, Clade]] = None,
    internal_index: Optional[Dict[int, int]] = None,
) -> str:
    """Resolve LCA label using precomputed maps when available and cache results.

    members: list of terminal names
    name_to_clade: optional mapping name->Clade to allow passing Clade objects to common_ancestor
    internal_index: optional mapping id(node)->index to avoid scanning nonterminals
    """
    key = frozenset(members)
    if key in _LCA_CACHE:
        return _LCA_CACHE[key]

    try:
        # If provided, translate member names to clade objects to avoid repeated name lookups
        if name_to_clade is not None:
            clades = [name_to_clade[m] for m in members if m in name_to_clade]
            lca = tree.common_ancestor(clades)
        else:
            lca = tree.common_ancestor(members)

        if lca.name:
            _LCA_CACHE[key] = lca.name
            return lca.name

        if internal_index is not None:
            idx = internal_index.get(id(lca))
            if idx is not None:
                label = f"InternalNode_{idx}"
                _LCA_CACHE[key] = label
                return label

        # Fallback: attempt to locate preorder index once
        for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
            if node is lca:
                label = f"InternalNode_{idx}"
                _LCA_CACHE[key] = label
                return label

        label = "InternalNode_unknown"
        _LCA_CACHE[key] = label
        return label
    except (AttributeError, ValueError):
        label = f"InternalNode_fallback_{hash(tuple(sorted(members))) % 10000}"
        _LCA_CACHE[key] = label
        return label


def _build_level_assignments(
    labels: Sequence[str],
    linkage_rows: Sequence[Sequence[float]],
    threshold: float,
    min_clade_size: int,
) -> Dict[int, List[str]]:
    global _PROFILE_DATA
    start = time.time()
    _PROFILE_DATA["build_level_assignments_calls"] += 1
    
    cluster_ids = [int(x) for x in fcluster(linkage_rows, t=threshold, criterion="distance")]
    assignments: Dict[int, List[str]] = {}
    for terminal_name, cluster_id in zip(labels, cluster_ids):
        assignments.setdefault(cluster_id, []).append(terminal_name)
    filtered = {cid: members for cid, members in assignments.items() if len(members) >= min_clade_size}
    
    elapsed = time.time() - start
    _PROFILE_DATA["build_level_assignments_total_time"] += elapsed
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


def _generate_oversampled_thresholds(
    linkage_rows: Sequence[Sequence[float]],
    oversample: int = 100,
) -> List[float]:
    """Generate an oversampled set of candidate thresholds across the linkage distance range.

    Uses the cophenetic/linkage distances (third column of linkage_rows) which
    correspond to dendrogram heights. This matches visual dendrogram slicing.
    """
    z = np.asarray(linkage_rows, dtype=float)
    if z.size == 0:
        return []

    # Use unique distances if there are plenty, otherwise linspace across min..max
    unique_distances = sorted(set(float(x) for x in z[:, 2]), reverse=True)
    if len(unique_distances) >= oversample:
        # sample approximately evenly from unique distances
        raw_indices = np.linspace(0, len(unique_distances) - 1, num=oversample)
        selected_indices = sorted(set(int(round(i)) for i in raw_indices))
        return [unique_distances[i] for i in selected_indices]

    # Fallback: produce oversample points between max and min distance
    min_d = float(np.min(z[:, 2]))
    max_d = float(np.max(z[:, 2]))
    if oversample <= 1 or max_d == min_d:
        return [max_d]
    raw = np.linspace(max_d, min_d, num=oversample)
    # snap to nearest unique distance when possible to avoid below-resolution picks
    snapped = []
    for r in raw:
        # choose nearest unique distance
        if unique_distances:
            nearest = min(unique_distances, key=lambda x: abs(x - r))
            snapped.append(float(nearest))
        else:
            snapped.append(float(r))
    # deduplicate while preserving order
    out: List[float] = []
    seen = set()
    for v in snapped:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _materialize_multithreshold_layer(
    tree: Tree,
    labels: Sequence[str],
    linkage_rows: Sequence[Sequence[float]],
    threshold: float,
    layer_index: int,
    min_clade_size: int,
    output_dir: Path,
    name_to_clade: Optional[Dict[str, Clade]] = None,
    internal_index: Optional[Dict[int, int]] = None,
) -> Optional[Dict[str, float]]:
    print(f"[PROFILE] _materialize_multithreshold_layer(layer_index={layer_index}, threshold={threshold:.6f}) - START")
    start_overall = time.time()
    
    start = time.time()
    assignments = _build_level_assignments(labels, linkage_rows, threshold, min_clade_size)
    print(f"  [PROFILE] _build_level_assignments took {time.time() - start:.3f}s, {len(assignments)} clusters")
    
    if len(assignments) < 2:
        print(f"[PROFILE] _materialize_multithreshold_layer(layer_index={layer_index}) - SKIP (< 2 clusters)")
        return None

    start = time.time()
    membership = _level_membership_map(assignments)
    print(f"  [PROFILE] _level_membership_map took {time.time() - start:.3f}s")
    
    start = time.time()
    lca_map = {
        cluster_id: _resolve_lca_label(tree, members, name_to_clade=name_to_clade, internal_index=internal_index)
        for cluster_id, members in assignments.items()
    }
    lca_time = time.time() - start
    print(f"  [PROFILE] LCA resolution took {lca_time:.3f}s for {len(assignments)} clusters")
    
    layer_dir = output_dir / f"layer_{layer_index:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = layer_dir / f"tree_cluster_assignments_layer{layer_index:02d}.csv"
    cluster_path = layer_dir / f"tree_clusters_layer{layer_index:02d}.csv"

    start = time.time()
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
    print(f"  [PROFILE] Assignment CSV write took {time.time() - start:.3f}s")
    

    start = time.time()
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
    print(f"  [PROFILE] Cluster CSV write took {time.time() - start:.3f}s")
    

    total_elapsed = time.time() - start_overall
    print(f"[PROFILE] _materialize_multithreshold_layer(layer_index={layer_index}) - END ({total_elapsed:.3f}s total)")
    
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
    print(f"[PROFILE] write_multithreshold_cluster_artifacts - START (layer_count={layer_count}, workers={workers})")
    start_overall = time.time()
    
    start = time.time()
    tree = Phylo.read(str(rooted_tree_path), "newick")
    labels, linkage_rows = tree_to_linkage(tree)
    print(f"[PROFILE] tree_to_linkage took {time.time() - start:.3f}s, {len(labels)} labels")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simplified linear threshold selection: cut the linkage distance range into evenly spaced thresholds
    start = time.time()
    z = np.asarray(linkage_rows, dtype=float)
    if z.size == 0:
        pd.DataFrame([]).to_csv(output_dir / "tree_cluster_layers.csv", index=False)
        return []

    max_d = float(np.max(z[:, 2]))
    min_d = float(np.min(z[:, 2]))
    num_layers = int(layer_count) if int(layer_count) > 0 else 1
    thresholds = list(np.linspace(max_d, min_d, num=num_layers))
    print(f"[PROFILE] Linear thresholds generated in {time.time() - start:.3f}s, count={len(thresholds)}")

    # Build final candidate_layers with sequential layer ids (deep -> shallow order)
    candidate_layers: List[Tuple[int, float]] = [(i + 1, float(thresholds[i])) for i in range(len(thresholds))]

    if not candidate_layers:
        pd.DataFrame([]).to_csv(output_dir / "tree_cluster_layers.csv", index=False)
        return []

    # Precompute terminal name -> Clade mapping and internal node index to speed up LCA resolution
    name_to_clade: Dict[str, Clade] = {str(t.name): t for t in tree.get_terminals()}
    internal_index: Dict[int, int] = {id(node): idx for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1)}

    # Implement roll-down inheritance: process layers deep->shallow sequentially
    layer_records: List[Dict[str, float]] = []
    print(f"[PROFILE] Processing {len(candidate_layers)} layers sequentially with roll-down inheritance")
    start = time.time()

    # Precompute linkage matrix once for dendrogram plotting
    z = np.asarray(linkage_rows, dtype=float)

    # Track current assignment per sequence (None => unassigned)
    current_assignment: Dict[str, Optional[int]] = {str(lbl): None for lbl in labels}

    # Iterate layers in the order candidate_layers already provides (deep -> shallow)
    for idx, (layer_idx, threshold) in enumerate(candidate_layers, start=1):
        print(f"[PROFILE] Starting roll-down layer {layer_idx}/{len(candidate_layers)} (threshold={threshold:.6f})")

        # Compute raw assignments at this threshold
        assignments = _build_level_assignments(labels, linkage_rows, float(threshold), min_clade_size)
        membership_new = _level_membership_map(assignments)

        # Determine which clusters are valid at this threshold (>= min_clade_size)
        valid_clusters = {int(cid) for cid, members in assignments.items() if len(members) >= int(min_clade_size)}

        # Apply inheritance: update current_assignment only when new cluster is valid
        for seq in sorted(labels):
            new_cid = membership_new.get(seq)
            if new_cid is not None and int(new_cid) in valid_clusters:
                current_assignment[seq] = int(new_cid)
            else:
                # keep previous assignment (may be None)
                pass

        # Build final cluster membership mapping after inheritance
        cluster_members_final: Dict[int, List[str]] = {}
        for seq, cid in current_assignment.items():
            if cid is None:
                continue
            cluster_members_final.setdefault(int(cid), []).append(seq)

        # Resolve LCAs for final clusters using the actual member lists
        start_lca = time.time()
        lca_map: Dict[int, str] = {
            int(cluster_id): _resolve_lca_label(tree, members, name_to_clade=name_to_clade, internal_index=internal_index)
            for cluster_id, members in cluster_members_final.items()
        }
        lca_elapsed = time.time() - start_lca
        print(f"  [PROFILE] LCA resolution (final clusters) took {lca_elapsed:.3f}s for {len(lca_map)} clusters")

        # Materialize CSV artifacts for this layer
        layer_dir = output_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        assignment_path = layer_dir / f"tree_cluster_assignments_layer{layer_idx:02d}.csv"
        cluster_path = layer_dir / f"tree_clusters_layer{layer_idx:02d}.csv"

        start_write = time.time()
        assignment_rows = []
        for sequence_id in sorted(labels):
            cid = current_assignment.get(sequence_id)
            assignment_rows.append(
                {
                    "sequence_id": sequence_id,
                    "cluster_id": int(cid) if cid is not None else "",
                    "lca_node": lca_map[int(cid)] if (cid is not None and int(cid) in lca_map) else "",
                    "layer_index": int(layer_idx),
                    "threshold": float(threshold),
                }
            )
        pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)
        print(f"  [PROFILE] Assignment CSV write took {time.time() - start_write:.3f}s")

        # Also copy to output root for backward compatibility
        try:
            shutil.copyfile(str(assignment_path), str(output_dir / assignment_path.name))
        except Exception:
            pass

        start_write = time.time()
        cluster_rows = []
        if cluster_members_final:
            cluster_rows = [
                {
                    "layer_index": int(layer_idx),
                    "threshold": float(threshold),
                    "cluster_id": int(cluster_id),
                    "member_count": int(len(members)),
                    "lca_node": lca_map[int(cluster_id)],
                }
                for cluster_id, members in sorted(cluster_members_final.items(), key=lambda item: int(item[0]))
            ]
        pd.DataFrame(cluster_rows).to_csv(cluster_path, index=False)
        print(f"  [PROFILE] Cluster CSV write took {time.time() - start_write:.3f}s")

        # Also copy clusters to output root for backward compatibility
        try:
            shutil.copyfile(str(cluster_path), str(output_dir / cluster_path.name))
        except Exception:
            pass

        

        # Generate SciPy linkage-based dendrogram for this layer (no labels for large trees)
        try:
            import matplotlib.pyplot as _plt
            from scipy.cluster.hierarchy import dendrogram as _dendrogram

            fig, ax = _plt.subplots(figsize=(12, 3))
            # Plot dendrogram without labels to avoid enormous text rendering
            _dendrogram(z, no_labels=True, color_threshold=float(threshold))
            ax = _plt.gca()
            ax.axhline(y=float(threshold), color="red", linestyle="--")
            linkage_svg = layer_dir / f"linkage_dendrogram_layer{layer_idx:02d}.svg"
            fig.savefig(str(linkage_svg), bbox_inches="tight")
            _plt.close(fig)
        except Exception as e:
            print(f"  [WARN] linkage dendrogram generation failed for layer {layer_idx}: {e}")

        layer_records.append(
            {
                "layer_index": int(layer_idx),
                "threshold": float(threshold),
                "cluster_count": int(len(cluster_members_final)),
                "assignments_csv": str(assignment_path),
                "clusters_csv": str(cluster_path),
            }
        )

        print(f"[PROFILE] Layer {layer_idx} done ({time.time() - start:.3f}s elapsed so far)")

    layer_records = sorted(layer_records, key=lambda item: int(item["layer_index"]))
    pd.DataFrame(layer_records).to_csv(output_dir / "tree_cluster_layers.csv", index=False)
    # Generate clades vs threshold summary plot (deep -> shallow across X)
    try:
        import matplotlib.pyplot as _plt

        thresholds_plot = [float(r["threshold"]) for r in layer_records]
        clade_counts = [int(r["cluster_count"]) for r in layer_records]

        fig, ax = _plt.subplots(figsize=(8, 4))
        ax.plot(thresholds_plot, clade_counts, marker="o", linestyle="-", color="#2C7FB8")
        ax.set_xlabel("Linkage Threshold (deep -> shallow)")
        ax.set_ylabel("Number of Valid Clades")
        ax.set_title("Clades vs Threshold (per selected layers)")
        ax.grid(alpha=0.3)
        summary_path = output_dir / "clades_vs_threshold.svg"
        fig.savefig(str(summary_path), bbox_inches="tight")
        _plt.close(fig)
    except Exception:
        pass

    total_elapsed = time.time() - start_overall
    print(f"[PROFILE] write_multithreshold_cluster_artifacts - END ({total_elapsed:.3f}s total)")
    print(f"[PROFILE] SUMMARY: _resolve_lca_label() called {_PROFILE_DATA['resolve_lca_calls']} times, total={_PROFILE_DATA['resolve_lca_total_time']:.3f}s, avg={_PROFILE_DATA['resolve_lca_total_time']/_PROFILE_DATA['resolve_lca_calls']:.6f}s/call" if _PROFILE_DATA['resolve_lca_calls'] > 0 else "")
    print(f"[PROFILE] SUMMARY: _build_level_assignments() called {_PROFILE_DATA['build_level_assignments_calls']} times, total={_PROFILE_DATA['build_level_assignments_total_time']:.3f}s, avg={_PROFILE_DATA['build_level_assignments_total_time']/_PROFILE_DATA['build_level_assignments_calls']:.6f}s/call" if _PROFILE_DATA['build_level_assignments_calls'] > 0 else "")
    
    return layer_records


def choose_distance_threshold(
    linkage_rows: Sequence[Sequence[float]],
    target_min_clades: int = 20,
    target_max_clades: int = 80,
    min_clade_size: int = 1,
    iterations: int = 30,
) -> float:
    print(f"[PROFILE] choose_distance_threshold(target={target_min_clades}-{target_max_clades}, min_size={min_clade_size}, iterations={iterations}) - START")
    start_overall = time.time()
    
    if target_min_clades <= 0 or target_max_clades < target_min_clades:
        raise ValueError("Invalid target clade range.")

    z = np.asarray(linkage_rows, dtype=float)
    min_dist = float(np.min(z[:, 2]))
    max_dist = float(np.max(z[:, 2]))

    low = min_dist
    high = max_dist
    midpoint = (target_min_clades + target_max_clades) / 2.0

    sampled: List[Tuple[float, int]] = []

    start = time.time()
    low_count = _surviving_clade_count_at_threshold(linkage_rows, low, min_clade_size)
    print(f"  [PROFILE] Initial low_count took {time.time() - start:.3f}s")
    
    start = time.time()
    high_count = _surviving_clade_count_at_threshold(linkage_rows, high, min_clade_size)
    print(f"  [PROFILE] Initial high_count took {time.time() - start:.3f}s")
    
    sampled.append((low, low_count))
    sampled.append((high, high_count))

    if target_min_clades <= low_count <= target_max_clades:
        print(f"[PROFILE] choose_distance_threshold - EARLY EXIT at low (found={low_count})")
        return low
    if target_min_clades <= high_count <= target_max_clades:
        print(f"[PROFILE] choose_distance_threshold - EARLY EXIT at high (found={high_count})")
        return high

    for iteration in range(iterations):
        mid = (low + high) / 2.0
        start = time.time()
        mid_count = _surviving_clade_count_at_threshold(linkage_rows, mid, min_clade_size)
        iter_time = time.time() - start
        print(f"  [PROFILE] Binary search iteration {iteration+1}/{iterations}: mid_count={mid_count}, time={iter_time:.3f}s")
        
        sampled.append((mid, mid_count))

        if target_min_clades <= mid_count <= target_max_clades:
            total = time.time() - start_overall
            print(f"[PROFILE] choose_distance_threshold - CONVERGED at iteration {iteration+1}/{iterations} (found={mid_count}, total={total:.3f}s)")
            return mid

        if mid_count > target_max_clades:
            low = mid
        else:
            high = mid

    best_threshold, _ = min(sampled, key=lambda x: abs(x[1] - midpoint))
    total = time.time() - start_overall
    print(f"[PROFILE] choose_distance_threshold - NO CONVERGENCE after {iterations} iterations (best={_}, total={total:.3f}s)")
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
    skip_dendrograms: bool = False,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    print(f"[PROFILE] cluster_tree_topologically() - START")
    start_overall = time.time()
    
    rooted_tree_path = rooted_tree_output
    if rooted_tree_path is None:
        rooted_tree_path = assignments_output.parent / f"{tree_path.stem}_{rooting_method}_rooted.tree"

    start = time.time()
    print(f"[PROFILE] root_tree() - START")
    rooted_tree_path = root_tree(
        input_tree=tree_path,
        output_tree=rooted_tree_path,
        method=rooting_method,
    )
    print(f"[PROFILE] root_tree() took {time.time() - start:.3f}s, output={rooted_tree_path}")

    start = time.time()
    tree = Phylo.read(str(rooted_tree_path), "newick")
    print(f"[PROFILE] Phylo.read() took {time.time() - start:.3f}s")

    start = time.time()
    labels, linkage_rows = tree_to_linkage(tree)
    print(f"[PROFILE] tree_to_linkage() took {time.time() - start:.3f}s, {len(labels)} labels")

    if group_distance_threshold is None:
        start = time.time()
        group_distance_threshold = choose_distance_threshold(
            linkage_rows=linkage_rows,
            target_min_clades=group_target_min_clades,
            target_max_clades=group_target_max_clades,
            min_clade_size=min_clade_size,
        )
        print(f"[PROFILE] choose_distance_threshold(group) took {time.time() - start:.3f}s")

    if family_distance_threshold is None:
        start = time.time()
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

    # Only plot hierarchical dendrograms if not skipped
    if dendrogram_output is not None and not skip_dendrograms:
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
    parser.add_argument("--multi-threshold-layers", type=int, default=8)
    # Alias for compatibility with older CLI usage
    parser.add_argument("--multi-layer", type=int, dest="multi_threshold_layers", help="Alias for --multi-threshold-layers")
    parser.add_argument("--max-layer-cluster-count", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-layer-plots", action="store_true", default=False, help="Skip generating per-layer dendrogram SVGs")
    parser.add_argument("--skip-hierarchical-plots", action="store_true", default=False, help="Skip generating hierarchical (group/family/subfamily) dendrogram SVGs")
    return parser

def write_hierarchical_tree_artifacts(
    clusters_csv: Path,
    assignments_csv: Path,
    rooted_tree_output: Path,
    output_dir: Path,
) -> None:
    """Writes level-specific hierarchical CSV and tree files for backward compatibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and split clusters
    if clusters_csv.exists():
        df_clusters = pd.read_csv(clusters_csv)
        for lvl, suffix in [("group", "groups"), ("family", "families"), ("subfamily", "subfamilies")]:
            sub = df_clusters[df_clusters["level"] == lvl]
            sub.to_csv(output_dir / f"tree_clusters_{suffix}.csv", index=False)
            
    # Load and split assignments
    if assignments_csv.exists():
        df_assign = pd.read_csv(assignments_csv)
        for lvl, suffix in [("group", "groups"), ("family", "families"), ("subfamily", "subfamilies")]:
            cols = ["sequence_id", f"{lvl}_id", f"{lvl}_lca_node"]
            # Map back to group_id/group_lca_node but rename to cluster_id/lca_node
            sub = df_assign[cols].rename(columns={
                f"{lvl}_id": "cluster_id",
                f"{lvl}_lca_node": "lca_node"
            })
            sub.to_csv(output_dir / f"tree_cluster_assignments_{suffix}.csv", index=False)
            
    # Copy trees
    if rooted_tree_output.exists():
        for suffix in ["groups", "families", "subfamilies"]:
            shutil.copyfile(str(rooted_tree_output), str(output_dir / f"midpoint_rooted_{suffix}.tree"))


def _plot_layer_dendrogram_task(tree_path_str: str, layer_assignments_csv_str: str, output_svg_str: str, layer_idx: str):
    """Worker task to plot a single layer dendrogram. Ensures matplotlib memory is flushed after plotting."""
    try:
        from pathlib import Path as _Path
        try:
            from src.visualization import build_terminal_color_map as _build_terminal_color_map, plot_topological_tree_dendrogram as _plot_topological_tree_dendrogram
        except Exception:
            from visualization import build_terminal_color_map as _build_terminal_color_map, plot_topological_tree_dendrogram as _plot_topological_tree_dendrogram

        tree_path = _Path(tree_path_str)
        layer_assignments_csv = _Path(layer_assignments_csv_str)
        output_svg = _Path(output_svg_str)

        terminal_colors = _build_terminal_color_map(layer_assignments_csv, 'cluster_id')
        try:
            from src.visualization import build_terminal_cluster_map as _build_terminal_cluster_map
        except Exception:
            from visualization import build_terminal_cluster_map as _build_terminal_cluster_map
        terminal_clusters = _build_terminal_cluster_map(layer_assignments_csv, 'cluster_id')
        _plot_topological_tree_dendrogram(
            tree_path,
            output_svg,
            title=f"Topological Clustering Dendrogram (Layer {layer_idx})",
            line_color="#B0B0B0",
            terminal_colors=terminal_colors,
            terminal_clusters=terminal_clusters,
        )
        
        # CRITICAL: flush matplotlib figures from memory immediately after writing
        try:
            import matplotlib.pyplot as _plt
            _plt.close('all')
        except Exception:
            pass
        
        return (True, layer_idx, "ok")
    except Exception as e:
        return (False, layer_idx, str(e))


def write_layer_dendrograms(
    tree_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> None:
    """Generate dendrograms for all multi-threshold layers using threading with progress bars."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('layer_')],
        key=lambda d: int(d.name.split('_')[1])
    )

    tasks = []
    skipped_existing = 0
    for layer_dir in layer_dirs:
        layer_idx = layer_dir.name.replace('layer_', '')
        layer_assignments_csv = layer_dir / f"tree_cluster_assignments_layer{layer_idx}.csv"
        if not layer_assignments_csv.exists():
            continue
        output_svg = layer_dir / f"tree_dendrogram_layer{layer_idx}.svg"
        if output_svg.exists() and output_svg.stat().st_size > 0:
            skipped_existing += 1
            continue
        tasks.append((str(tree_path), str(layer_assignments_csv), str(output_svg), layer_idx))

    if not tasks:
        if skipped_existing > 0:
            print(f"✓ Skipped {skipped_existing} existing non-empty layer dendrogram SVG(s)")
        return

    generated_count = 0
    # Use ThreadPoolExecutor with progress bar for plotting
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(tasks)))) as executor:
        futures = {executor.submit(_plot_layer_dendrogram_task, t[0], t[1], t[2], t[3]): t[3] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Plotting layer dendrograms"):
            try:
                success, layer_idx, msg = future.result()
                if success:
                    generated_count += 1
            except Exception:
                pass

    if generated_count > 0:
        print(f"✓ Generated {generated_count}/{len(tasks)} per-layer dendrograms in layer_XX/ subdirectories")
    if skipped_existing > 0:
        print(f"✓ Skipped {skipped_existing} existing non-empty layer dendrogram SVG(s)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(f"[PROFILE] main() START - multi-threshold clustering only")
    start = time.time()
    layer_records = write_multithreshold_cluster_artifacts(
        rooted_tree_path=Path(args.rooted_tree_output),
        output_dir=Path(args.assignments_output).parent,
        layer_count=int(args.multi_threshold_layers),
        min_clade_size=int(args.min_clade_size),
        max_layer_cluster_count=int(args.max_layer_cluster_count),
        workers=int(args.workers),
    )
    print(f"[PROFILE] write_multithreshold_cluster_artifacts() took {time.time() - start:.3f}s")

    # Optionally generate per-layer dendrograms (fast if skipped)
    if not args.skip_layer_plots:
        write_layer_dendrograms(tree_path=Path(args.rooted_tree_output), output_dir=Path(args.assignments_output).parent, workers=args.workers)
    
    print(f"[PROFILE] write_multithreshold_cluster_artifacts() START")
    start = time.time()
    layer_records = write_multithreshold_cluster_artifacts(
        rooted_tree_path=Path(args.rooted_tree_output),
        output_dir=Path(args.assignments_output).parent,
        layer_count=args.multi_threshold_layers,
        min_clade_size=args.min_clade_size,
        max_layer_cluster_count=args.max_layer_cluster_count,
        workers=args.workers,
    )
    print(f"[PROFILE] write_multithreshold_cluster_artifacts() took {time.time() - start:.3f}s")
    
    if not args.skip_layer_plots:
        print(f"[PROFILE] write_layer_dendrograms() START")
        start = time.time()
        write_layer_dendrograms(
            tree_path=Path(args.rooted_tree_output),
            output_dir=Path(args.assignments_output).parent,
            workers=args.workers,
        )
        print(f"[PROFILE] write_layer_dendrograms() took {time.time() - start:.3f}s")
    
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
    print(f"Saved multi-threshold layers: {len(layer_records)} (tree_cluster_assignments_layerXX.csv / tree_clusters_layerXX.csv)")


if __name__ == "__main__":
    main()
