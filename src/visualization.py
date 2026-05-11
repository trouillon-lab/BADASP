import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade, Tree
from scipy.cluster.hierarchy import fcluster


LEVEL_COLORS = {
    "groups": "#1F77B4",
    "families": "#D95F02",
    "subfamilies": "#2CA02C",
}


def build_terminal_color_map(assignments_path: Path, cluster_column: str) -> Dict[str, str]:
    assignments = pd.read_csv(assignments_path)
    if cluster_column not in assignments.columns:
        raise ValueError(f"Missing required column in {assignments_path}: {cluster_column}")

    palette = [mcolors.to_hex(color) for color in (list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors))]
    color_map: Dict[str, str] = {}
    unique_clusters = list(dict.fromkeys(assignments[cluster_column].dropna().tolist()))

    for index, cluster_id in enumerate(unique_clusters):
        color = palette[index % len(palette)]
        members = assignments[assignments[cluster_column] == cluster_id]["sequence_id"].astype(str)
        for sequence_id in members:
            color_map[str(sequence_id)] = color

    return color_map


def _subtree_terminal_color(
    clade: Clade,
    terminal_colors: Optional[Dict[str, str]],
    cache: Dict[int, Optional[str]],
) -> Optional[str]:
    if terminal_colors is None:
        return None
    key = id(clade)
    if key in cache:
        return cache[key]

    if clade.is_terminal():
        color = terminal_colors.get(str(clade.name))
        cache[key] = color
        return color

    child_colors = [_subtree_terminal_color(child, terminal_colors, cache) for child in clade.clades]
    non_null_colors = [color for color in child_colors if color is not None]
    # Keep a subtree colored when all mapped descendants agree; unmapped leaves do not force gray.
    color = non_null_colors[0] if non_null_colors and len(set(non_null_colors)) == 1 else None
    cache[key] = color
    return color


def default_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/sequence_filtering/raw_length_dist.svg"),
        Path("results/alignment_qc/msa_gap_profile.svg"),
        Path("results/topological_clustering/tree_dendrogram.svg"),
    )


def default_hierarchical_badasp_plot_paths() -> Tuple[Path, Path]:
    return (
        Path("results/badasp_scoring/hierarchical_distributions.svg"),
        Path("results/badasp_scoring/hierarchical_switch_counts.svg"),
    )


def default_individual_badasp_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/badasp_scoring/badasp_score_distribution_groups.svg"),
        Path("results/badasp_scoring/badasp_score_distribution_families.svg"),
        Path("results/badasp_scoring/badasp_score_distribution_subfamilies.svg"),
    )


def default_tree_switch_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/badasp_scoring/tree_switches_groups.svg"),
        Path("results/badasp_scoring/tree_switches_families.svg"),
        Path("results/badasp_scoring/tree_switches_subfamilies.svg"),
    )


def default_duplication_badasp_plot_paths() -> Tuple[Path, Path, Path, Path]:
    return (
        Path("results/badasp_scoring/badasp_score_distribution_duplications.svg"),
        Path("results/badasp_scoring/switch_counts_duplications.svg"),
        Path("results/badasp_scoring/tree_switches_duplications.svg"),
        Path("results/badasp_scoring/dendrogram_switches_duplications.svg"),
    )


def _read_fasta_lengths(fasta_path: Path) -> List[int]:
    return [len(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def plot_sequence_length_distribution(fasta_path: Path, output_svg: Path) -> None:
    lengths = _read_fasta_lengths(fasta_path)
    if not lengths:
        raise ValueError(f"No sequences found in {fasta_path}")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=80, kde=True, color="#2E86AB")
    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length (AA)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def compute_gap_percentages(msa_path: Path) -> List[float]:
    sequences = [str(record.seq) for record in SeqIO.parse(str(msa_path), "fasta")]
    if not sequences:
        raise ValueError(f"No aligned sequences found in {msa_path}")

    aln_len = len(sequences[0])
    gap_percentages: List[float] = []
    for i in range(aln_len):
        gap_count = sum(1 for seq in sequences if seq[i] == "-")
        gap_percentages.append((gap_count / len(sequences)) * 100.0)
    return gap_percentages


def plot_gap_percentage_per_column(msa_path: Path, output_svg: Path) -> None:
    gap_percentages = compute_gap_percentages(msa_path)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(gap_percentages) + 1), gap_percentages, color="#F18F01", linewidth=1.0)
    plt.title("MSA Gap Percentage per Column")
    plt.xlabel("Alignment Column")
    plt.ylabel("Gap Percentage (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_topological_dendrogram(
    linkage_matrix: Sequence[Sequence[float]],
    output_svg: Path,
    max_leaves: int = 200,
    color_threshold: float = 0.0,
) -> None:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(linkage_matrix, dtype=float)
    if z.shape[0] < 1:
        raise ValueError("Need at least 2 leaves to draw dendrogram.")

    n_leaves = z.shape[0] + 1
    leaves_order = list(range(n_leaves))
    x_by_node, y_by_node, descendants = _compute_dendrogram_node_coords(z, leaves_order)

    width = 12 if n_leaves <= max_leaves else min(40, 12 + (n_leaves / max_leaves) * 2.0)
    fig, ax = plt.subplots(figsize=(width, 6))

    palette = [mcolors.to_hex(c) for c in (list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors))]
    if color_threshold > 0.0:
        leaf_cluster_ids = [int(x) for x in fcluster(z, t=float(color_threshold), criterion="distance")]
        unique_clusters = sorted(set(leaf_cluster_ids))
        cluster_color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
    else:
        leaf_cluster_ids = [0 for _ in range(n_leaves)]
        cluster_color_map = {0: "#666666"}

    for merge_idx, row in enumerate(z):
        left = int(row[0])
        right = int(row[1])
        node_id = n_leaves + merge_idx
        node_height = float(row[2])

        leaf_ids = descendants[node_id]
        leaf_clusters = {leaf_cluster_ids[leaf_id] for leaf_id in leaf_ids}
        if color_threshold > 0.0 and node_height <= float(color_threshold) and len(leaf_clusters) == 1:
            color = cluster_color_map[next(iter(leaf_clusters))]
        else:
            color = "#666666"

        ax.plot([x_by_node[left], x_by_node[left]], [y_by_node[left], node_height], color=color, linewidth=1.0)
        ax.plot([x_by_node[right], x_by_node[right]], [y_by_node[right], node_height], color=color, linewidth=1.0)
        ax.plot([x_by_node[left], x_by_node[right]], [node_height, node_height], color=color, linewidth=1.0)

    ax.set_title("Topological Clustering Dendrogram")
    ax.set_xlabel("Collapsed Leaf Groups")
    ax.set_ylabel("Cophenetic Distance")
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def _load_score_table(score_path: Path) -> pd.DataFrame:
    df = pd.read_csv(score_path)
    required_columns = {"position", "switch_count", "global_threshold", "badasp_score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {score_path}: {sorted(missing)}")
    return df


def _load_raw_switch_table(raw_pairwise_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_pairwise_path)
    required_columns = {"position", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {raw_pairwise_path}: {sorted(missing)}")
    return df


def _load_pairwise_table(pairwise_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pairwise_path)
    required_columns = {"pair", "position", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {pairwise_path}: {sorted(missing)}")
    return df


def _compute_95th_threshold(scores: np.ndarray) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return 0.0
    return float(np.percentile(clean, 95))


def plot_badasp_score_distribution(
    raw_pairwise_path: Path,
    output_svg: Path,
    title: str,
    color: str,
) -> None:
    df = _load_pairwise_table(raw_pairwise_path)
    scores = df["score"].astype(float).to_numpy()
    threshold = _compute_95th_threshold(scores)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=40, stat="count", color=color, alpha=0.35)
    plt.axvline(threshold, color=color, linestyle="--", linewidth=2.0, label=f"95th percentile = {threshold:.6f}")
    plt.title(title)
    plt.xlabel("Raw BADASP Score")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_hierarchical_badasp_distributions(
    group_pairwise: Path,
    family_pairwise: Path,
    subfamily_pairwise: Path,
    output_svg: Path,
) -> None:
    score_tables = {
        "Groups": _load_pairwise_table(group_pairwise),
        "Families": _load_pairwise_table(family_pairwise),
        "Subfamilies": _load_pairwise_table(subfamily_pairwise),
    }
    colors = {
        "Groups": "#1F77B4",
        "Families": "#D95F02",
        "Subfamilies": "#2CA02C",
    }

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))

    thresholds = {
        label: _compute_95th_threshold(df["score"].astype(float).to_numpy())
        for label, df in score_tables.items()
    }

    for label, df in score_tables.items():
        scores = df["score"].astype(float).to_numpy()
        threshold = thresholds[label]
        sns.histplot(
            scores,
            bins=40,
            stat="count",
            element="step",
            fill=False,
            common_bins=True,
            color=colors[label],
            label=label,
            linewidth=2.0,
        )
        plt.axvline(threshold, color=colors[label], linestyle="--", linewidth=1.5, alpha=0.8)

    threshold_legend = [
        Line2D([0], [0], color=colors[label], linestyle="--", linewidth=1.5, label=f"{label} 95th pct.")
        for label in score_tables
    ]
    density_legend = [
        Line2D([0], [0], color=colors[label], linewidth=2.0, label=label)
        for label in score_tables
    ]
    plt.legend(handles=density_legend + threshold_legend, loc="best", frameon=False, ncol=2)
    plt.title("Hierarchical BADASP Score Distributions")
    plt.xlabel("Raw BADASP Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_hierarchical_switch_counts(
    group_scores: Path,
    family_scores: Path,
    subfamily_scores: Path,
    output_svg: Path,
) -> None:
    score_tables = [
        ("Groups", _load_score_table(group_scores), "#1F77B4"),
        ("Families", _load_score_table(family_scores), "#D95F02"),
        ("Subfamilies", _load_score_table(subfamily_scores), "#2CA02C"),
    ]

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (label, df, color) in zip(axes, score_tables):
        positions = df["position"].astype(int).to_numpy()
        switch_counts = df["switch_count"].astype(int).to_numpy()
        ax.bar(positions, switch_counts, color=color, width=1.0, alpha=0.9)
        ax.set_ylabel("Switches")
        ax.set_title(label)
        ax.set_xlim(1, int(positions.max()))
        ax.set_ylim(0, max(1, int(switch_counts.max())) + 1)

    axes[-1].set_xlabel("Alignment Position")
    fig.suptitle("Hierarchical BADASP Switch Counts Across the Alignment", y=0.995)
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def plot_duplication_badasp_distribution(raw_pairwise_path: Path, output_svg: Path) -> None:
    df = _load_pairwise_table(raw_pairwise_path)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    if "Event_Type" not in df.columns or df["Event_Type"].dropna().nunique() <= 1:
        plot_badasp_score_distribution(
            raw_pairwise_path=raw_pairwise_path,
            output_svg=output_svg,
            title="Dual-Track BADASP Score Distribution",
            color="#B24A2A",
        )
        return

    event_palette = {
        "Duplication": "#B24A2A",
        "Speciation": "#2A6FB2",
        "Unknown": "#7A7A7A",
    }
    scores = pd.to_numeric(df["score"], errors="coerce")
    threshold = _compute_95th_threshold(scores.to_numpy(dtype=float))

    plt.figure(figsize=(10, 6))
    for event_type, group in df.groupby("Event_Type"):
        event_scores = pd.to_numeric(group["score"], errors="coerce")
        event_scores = event_scores[np.isfinite(event_scores)]
        if event_scores.empty:
            continue
        sns.histplot(
            event_scores,
            bins=40,
            stat="count",
            element="step",
            fill=False,
            color=event_palette.get(str(event_type), "#7A7A7A"),
            linewidth=1.8,
            label=str(event_type),
        )

    plt.axvline(threshold, color="#333333", linestyle="--", linewidth=1.6, label=f"95th percentile = {threshold:.6f}")
    plt.title("Dual-Track BADASP Score Distribution")
    plt.xlabel("Raw BADASP Score")
    plt.ylabel("Count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_duplication_switch_counts(raw_pairwise_path: Path, output_svg: Path, percentile: float = 95.0) -> None:
    raw = _load_raw_switch_table(raw_pairwise_path)
    scores = pd.to_numeric(raw["score"], errors="coerce")
    scores = scores[np.isfinite(scores)]
    threshold = float(np.percentile(scores.to_numpy(dtype=float), float(percentile))) if not scores.empty else 0.0
    switched = raw[pd.to_numeric(raw["score"], errors="coerce") >= threshold].copy()
    if switched.empty:
        positions = np.array([], dtype=int)
        switch_counts = np.array([], dtype=int)
        top_col = 0
        top_count = 0
        grouped = pd.DataFrame(columns=["position", "Event_Type", "switch_count"])
    else:
        switch_df = switched.groupby("position", as_index=False).size().rename(columns={"size": "switch_count"})
        switch_df = switch_df.sort_values(["position"]).copy()
        positions = switch_df["position"].astype(int).to_numpy()
        switch_counts = switch_df["switch_count"].astype(int).to_numpy()
        top_row = switch_df.sort_values(["switch_count", "position"], ascending=[False, True]).head(1)
        top_col = int(top_row.iloc[0]["position"]) if not top_row.empty else 0
        top_count = int(top_row.iloc[0]["switch_count"]) if not top_row.empty else 0
        if "Event_Type" in switched.columns:
            grouped = (
                switched.groupby(["position", "Event_Type"], as_index=False)
                .size()
                .rename(columns={"size": "switch_count"})
            )
        else:
            grouped = pd.DataFrame(columns=["position", "Event_Type", "switch_count"])

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    if len(positions):
        if not grouped.empty and grouped["Event_Type"].nunique() > 1:
            palette = {"Duplication": "#B24A2A", "Speciation": "#2A6FB2", "Unknown": "#7A7A7A"}
            pivoted = grouped.pivot(index="position", columns="Event_Type", values="switch_count").fillna(0.0)
            bottom = np.zeros(len(pivoted.index), dtype=float)
            for event_type in sorted(pivoted.columns):
                values = pivoted[event_type].to_numpy(dtype=float)
                ax.bar(
                    pivoted.index.to_numpy(dtype=int),
                    values,
                    bottom=bottom,
                    width=1.0,
                    alpha=0.9,
                    color=palette.get(str(event_type), "#7A7A7A"),
                    label=str(event_type),
                )
                bottom += values
            ax.legend(frameon=False)
        else:
            ax.bar(positions, switch_counts, color="#B24A2A", width=1.0, alpha=0.9)
    ax.set_xlabel("Alignment Column Index")
    ax.set_ylabel("Switches")
    ax.set_title("Dual-Track BADASP Switch Counts")
    if len(positions):
        ax.set_xlim(1, int(positions.max()))
        ax.set_ylim(0, max(1, int(switch_counts.max())) + 1)
    ax.text(
        0.99,
        0.95,
        f"{percentile:g}th pct = {threshold:.6f}; top switch: alignment col {top_col} (count={top_count})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#6B1F10",
    )
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def plot_individual_hierarchical_badasp_distributions(
    group_pairwise: Path,
    family_pairwise: Path,
    subfamily_pairwise: Path,
    output_group_svg: Path,
    output_family_svg: Path,
    output_subfamily_svg: Path,
) -> None:
    plot_badasp_score_distribution(
        raw_pairwise_path=group_pairwise,
        output_svg=output_group_svg,
        title="Groups BADASP Score Distribution",
        color="#1F77B4",
    )
    plot_badasp_score_distribution(
        raw_pairwise_path=family_pairwise,
        output_svg=output_family_svg,
        title="Families BADASP Score Distribution",
        color="#D95F02",
    )
    plot_badasp_score_distribution(
        raw_pairwise_path=subfamily_pairwise,
        output_svg=output_subfamily_svg,
        title="Subfamilies BADASP Score Distribution",
        color="#2CA02C",
    )


def _ensure_tree_node_names(tree: Tree) -> None:
    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"


def _leaf_signature(node: Clade) -> Tuple[str, ...]:
    return tuple(sorted(str(terminal.name) for terminal in node.get_terminals() if terminal.name))


def _remap_named_nodes_to_plot_tree(
    plot_tree: Tree,
    named_tree_path: Optional[Path],
) -> Dict[str, str]:
    _ensure_tree_node_names(plot_tree)
    if named_tree_path is None or not named_tree_path.exists():
        return {}

    named_tree = Phylo.read(str(named_tree_path), "newick")
    _ensure_tree_node_names(named_tree)

    named_signatures: Dict[Tuple[str, ...], str] = {}
    for node in named_tree.get_nonterminals(order="level"):
        if not node.name:
            continue
        signature = _leaf_signature(node)
        if signature:
            named_signatures[signature] = str(node.name)

    plot_signature_to_name = {
        _leaf_signature(node): str(node.name)
        for node in plot_tree.get_nonterminals(order="level")
        if node.name
    }

    remap: Dict[str, str] = {}
    for signature, source_name in named_signatures.items():
        mapped_name = plot_signature_to_name.get(signature)
        if mapped_name:
            remap[source_name] = mapped_name
    return remap


def _build_y_positions(tree: Tree) -> Dict[Clade, float]:
    terminals = tree.get_terminals()
    y_positions: Dict[Clade, float] = {leaf: float(i) for i, leaf in enumerate(reversed(terminals), start=1)}

    def _assign(node: Clade) -> float:
        if node in y_positions:
            return y_positions[node]
        child_ys = [_assign(child) for child in node.clades]
        y_positions[node] = (min(child_ys) + max(child_ys)) / 2.0
        return y_positions[node]

    _assign(tree.root)
    return y_positions


def _draw_rotated_tree_axes(
    ax,
    tree: Tree,
    line_color: str = "#666666",
    line_width: float = 0.7,
    terminal_colors: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[Clade, float], Dict[Clade, float]]:
    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    x_positions = _build_y_positions(tree)
    subtree_colors: Dict[int, Optional[str]] = {}

    def _draw(node: Clade) -> None:
        x = x_positions[node]
        y = depths[node]
        for child in node.clades:
            child_x = x_positions[child]
            child_y = depths[child]
            child_color = _subtree_terminal_color(child, terminal_colors, subtree_colors)
            branch_color = child_color or line_color
            ax.plot([x, child_x], [y, y], color=branch_color, linewidth=line_width)
            ax.plot([child_x, child_x], [y, child_y], color=branch_color, linewidth=line_width)
            _draw(child)

    _draw(tree.root)

    return x_positions, depths


def plot_topological_tree_dendrogram(
    tree_path: Path,
    output_svg: Path,
    title: str = "Topological Clustering Dendrogram",
    line_color: str = "#B0B0B0",
    terminal_colors: Optional[Dict[str, str]] = None,
) -> None:
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_tree_node_names(tree)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    _draw_rotated_tree_axes(ax, tree, line_color=line_color, line_width=0.8, terminal_colors=terminal_colors)
    ax.set_title(title)
    ax.set_xlabel("Taxa / internal nodes")
    ax.set_ylabel("Branch length from root")
    ax.set_xticks([])
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def build_switch_node_map(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_path: Path,
    level: str,
) -> Dict[str, int]:
    level_map = {
        "groups": "group",
        "families": "family",
        "subfamilies": "subfamily",
    }
    if level not in level_map:
        raise ValueError(f"Unsupported level: {level}")

    singular = level_map[level]
    id_col = f"{singular}_id"

    tree = Phylo.read(str(tree_path), "newick")
    _ensure_tree_node_names(tree)

    assignments = pd.read_csv(assignments_path)
    members_by_cluster = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()

    raw_pairwise = _load_pairwise_table(raw_pairwise_path)
    if raw_pairwise.empty:
        return {}

    threshold = float(np.percentile(raw_pairwise["score"].astype(float), 95))
    switched = raw_pairwise[raw_pairwise["score"] > threshold]
    pair_switch_counts = switched.groupby("pair").size().to_dict()

    node_switch_counts: Dict[str, int] = defaultdict(int)
    for pair, switch_count in pair_switch_counts.items():
        try:
            left_str, right_str = str(pair).split("-")
            left_id = int(left_str)
            right_id = int(right_str)
        except ValueError:
            continue
        if left_id not in members_by_cluster or right_id not in members_by_cluster:
            continue
        pair_members = list(members_by_cluster[left_id]) + list(members_by_cluster[right_id])
        lca = tree.common_ancestor(pair_members)
        if not lca.name:
            continue
        node_switch_counts[lca.name] += int(switch_count)

    return dict(node_switch_counts)


def build_duplication_switch_node_map(raw_pairwise_path: Path) -> Dict[str, int]:
    raw_pairwise = _load_pairwise_table(raw_pairwise_path)
    if raw_pairwise.empty:
        return {}

    node_column = None
    for candidate in ("lca_node_name", "lca_node_id", "duplication_node"):
        if candidate in raw_pairwise.columns:
            node_column = candidate
            break
    if node_column is None:
        raise ValueError(
            "Duplication pairwise table requires one of: lca_node_name, lca_node_id, duplication_node"
        )

    threshold = float(np.percentile(raw_pairwise["score"].astype(float), 95))
    switched = raw_pairwise[raw_pairwise["score"] > threshold].copy()
    if switched.empty:
        return {}

    switched[node_column] = switched[node_column].astype(str)
    return switched.groupby(node_column).size().astype(int).to_dict()


def generate_duplication_tree_switch_plot(
    rooted_tree_path: Path,
    raw_pairwise_duplications: Path,
    output_svg: Path,
    reference_asr_tree_path: Optional[Path] = Path("data/interim/asr_run.treefile"),
) -> None:
    node_switch_map = build_duplication_switch_node_map(raw_pairwise_duplications)

    plot_tree = Phylo.read(str(rooted_tree_path), "newick")
    _ensure_tree_node_names(plot_tree)
    asr_to_plot_name = _remap_named_nodes_to_plot_tree(plot_tree, named_tree_path=reference_asr_tree_path)
    if asr_to_plot_name:
        remapped_counts: Dict[str, int] = defaultdict(int)
        for node_name, count in node_switch_map.items():
            remapped_counts[asr_to_plot_name.get(str(node_name), str(node_name))] += int(count)
        node_switch_map = dict(remapped_counts)

    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=node_switch_map,
        output_svg=output_svg,
        title="Switch Events on Tree: Duplication-Directed BADASP",
        line_color="#B0B0B0",
    )


def plot_tree_with_switches(
    tree_path: Path,
    node_switch_counts: Dict[str, int],
    output_svg: Path,
    title: str,
    line_color: str = "#B0B0B0",
    terminal_colors: Optional[Dict[str, str]] = None,
) -> None:
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_tree_node_names(tree)

    fig, ax = plt.subplots(figsize=(14, 10))
    x_positions, depths = _draw_rotated_tree_axes(
        ax,
        tree,
        line_color=line_color,
        line_width=0.7,
        terminal_colors=terminal_colors,
    )

    switched_nodes = [(node, count) for node, count in node_switch_counts.items() if count > 0]
    if switched_nodes:
        node_lookup = {clade.name: clade for clade in tree.find_clades() if clade.name}
        counts = np.array([count for _, count in switched_nodes], dtype=float)
        max_count = float(counts.max()) if len(counts) else 1.0

        xs = []
        ys = []
        sizes = []
        colors = []
        for node_name, count in switched_nodes:
            if node_name not in node_lookup:
                continue
            clade = node_lookup[node_name]
            xs.append(x_positions[clade])
            ys.append(depths[clade])
            sizes.append(30.0 + (220.0 * (count / max_count)))
            colors.append(count)

        scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="OrRd", alpha=0.9, edgecolor="#222222", linewidth=0.3)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Switch count")

    ax.set_title(title)
    ax.set_xlabel("Taxa / internal nodes")
    ax.set_ylabel("Branch length from root")
    ax.set_xticks([])
    ax.invert_yaxis()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def generate_tree_switch_plots(
    rooted_tree_path: Path,
    assignments_path: Path,
    output_groups_svg: Path,
    output_families_svg: Path,
    output_subfamilies_svg: Path,
    raw_pairwise_groups: Path,
    raw_pairwise_families: Path,
    raw_pairwise_subfamilies: Path,
) -> None:
    groups_map = build_switch_node_map(
        tree_path=rooted_tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_groups,
        level="groups",
    )
    families_map = build_switch_node_map(
        tree_path=rooted_tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_families,
        level="families",
    )
    subfamilies_map = build_switch_node_map(
        tree_path=rooted_tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_subfamilies,
        level="subfamilies",
    )

    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=groups_map,
        output_svg=output_groups_svg,
        title="Switch Events on Tree: Groups",
        line_color="#B0B0B0",
    )
    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=families_map,
        output_svg=output_families_svg,
        title="Switch Events on Tree: Families",
        line_color="#B0B0B0",
    )
    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=subfamilies_map,
        output_svg=output_subfamilies_svg,
        title="Switch Events on Tree: Subfamilies",
        line_color="#B0B0B0",
    )


def _compute_dendrogram_node_coords(
    linkage_matrix: np.ndarray,
    leaves_order: List[int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, frozenset]]:
    leaf_x = {leaf_idx: 5.0 + (10.0 * rank) for rank, leaf_idx in enumerate(leaves_order)}
    x_by_node: Dict[int, float] = {}
    y_by_node: Dict[int, float] = {}
    descendants: Dict[int, frozenset] = {}

    n_leaves = linkage_matrix.shape[0] + 1
    for i in range(n_leaves):
        x_by_node[i] = leaf_x[i]
        y_by_node[i] = 0.0
        descendants[i] = frozenset({i})

    for i, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node_id = n_leaves + i
        x_by_node[node_id] = (x_by_node[left] + x_by_node[right]) / 2.0
        y_by_node[node_id] = float(row[2])
        descendants[node_id] = descendants[left] | descendants[right]

    return x_by_node, y_by_node, descendants


def plot_dendrogram_with_switches(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_path: Path,
    level: str,
    output_svg: Path,
    title: str,
    color_threshold: float,
    line_color: str = "#B0B0B0",
    terminal_colors: Optional[Dict[str, str]] = None,
    min_clade_size: int = 5,
) -> None:
    level_map = {
        "groups": "group",
        "families": "family",
        "subfamilies": "subfamily",
    }
    if level not in level_map:
        raise ValueError(f"Unsupported level: {level}")
    singular = level_map[level]
    id_col = f"{singular}_id"

    tree = Phylo.read(str(tree_path), "newick")
    _ensure_tree_node_names(tree)

    assignments = pd.read_csv(assignments_path)
    cluster_members = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()

    raw_pairwise = _load_pairwise_table(raw_pairwise_path)
    switch_threshold = float(np.percentile(raw_pairwise["score"].astype(float), 95)) if not raw_pairwise.empty else 0.0
    switched = raw_pairwise[raw_pairwise["score"] > switch_threshold]
    pair_switch_counts = switched.groupby("pair").size().to_dict()
    pairs_with_switches = [(pair, int(count)) for pair, count in pair_switch_counts.items() if int(count) > 0]

    node_switch_counts: Dict[int, int] = defaultdict(int)
    for pair, count in pairs_with_switches:
        try:
            left_str, right_str = str(pair).split("-")
            left_id = int(left_str)
            right_id = int(right_str)
        except ValueError:
            continue
        if left_id not in cluster_members or right_id not in cluster_members:
            continue

        pair_members = cluster_members[left_id] + cluster_members[right_id]
        lca = tree.common_ancestor(pair_members)
        if lca.name:
            node_switch_counts[lca.name] += int(count)

    node_lookup = {clade.name: clade for clade in tree.find_clades() if clade.name}

    print(f"Total pairs with switches > 0: {len(pairs_with_switches)}")
    print(f"Total scatter points successfully mapped to coordinates: {len([n for n in node_switch_counts if n in node_lookup])}")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    x_positions, depths = _draw_rotated_tree_axes(
        ax,
        tree,
        line_color=line_color,
        line_width=0.8,
        terminal_colors=terminal_colors,
    )

    mapped_coordinates = []
    for node_name, count in node_switch_counts.items():
        if count <= 0 or node_name not in node_lookup:
            continue
        clade = node_lookup[node_name]
        if len(clade.get_terminals()) < min_clade_size:
            continue
        mapped_coordinates.append((float(x_positions[clade]), float(depths[clade]), int(count)))

    if mapped_coordinates:
        switch_values = np.array([entry[2] for entry in mapped_coordinates], dtype=float)
        max_val = float(switch_values.max()) if len(switch_values) else 1.0
        xs = [entry[0] for entry in mapped_coordinates]
        ys = [entry[1] for entry in mapped_coordinates]
        sizes = [40.0 + (260.0 * (val / max_val)) for val in switch_values]

        scatter = ax.scatter(
            xs,
            ys,
            c=switch_values,
            s=sizes,
            cmap="OrRd",
            edgecolor="#222222",
            linewidth=0.4,
            alpha=0.9,
            zorder=5,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label("Switch count")

    ax.set_title(title)
    ax.set_xlabel("Taxa / internal nodes")
    ax.set_ylabel("Branch length from root")
    ax.set_xticks([])
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def generate_dendrogram_switch_plots(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_groups: Path,
    raw_pairwise_families: Path,
    raw_pairwise_subfamilies: Path,
    output_groups_svg: Path,
    output_families_svg: Path,
    output_subfamilies_svg: Path,
    group_threshold: float,
    family_threshold: float,
    subfamily_threshold: float,
    min_clade_size: int = 5,
) -> None:
    groups_terminal_colors = build_terminal_color_map(assignments_path, "group_id")
    families_terminal_colors = build_terminal_color_map(assignments_path, "family_id")
    subfamilies_terminal_colors = build_terminal_color_map(assignments_path, "subfamily_id")

    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_groups,
        level="groups",
        output_svg=output_groups_svg,
        title="Groups Dendrogram with Switch Events",
        color_threshold=group_threshold,
        line_color="#B0B0B0",
        terminal_colors=groups_terminal_colors,
        min_clade_size=min_clade_size,
    )
    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_families,
        level="families",
        output_svg=output_families_svg,
        title="Families Dendrogram with Switch Events",
        color_threshold=family_threshold,
        line_color="#B0B0B0",
        terminal_colors=families_terminal_colors,
        min_clade_size=min_clade_size,
    )
    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_subfamilies,
        level="subfamilies",
        output_svg=output_subfamilies_svg,
        title="Subfamilies Dendrogram with Switch Events",
        color_threshold=subfamily_threshold,
        line_color="#B0B0B0",
        terminal_colors=subfamilies_terminal_colors,
        min_clade_size=min_clade_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and duplication-directed BADASP visualizations.")
    default_length_out, default_gap_out, _ = default_plot_paths()
    default_dup_dist_out, default_dup_switch_out, default_dup_tree_out, default_dup_dendrogram_out = default_duplication_badasp_plot_paths()
    parser.add_argument("--fasta", default=None, help="Input FASTA for length distribution plot.")
    parser.add_argument("--length-output", default=str(default_length_out))
    parser.add_argument("--msa", default=None, help="Input MSA FASTA for gap-per-column plot.")
    parser.add_argument("--gap-output", default=str(default_gap_out))
    parser.add_argument("--duplication-pairwise", default="results/badasp_scoring/raw_pairwise.csv")
    parser.add_argument("--rooted-tree", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--duplication-distribution-output", default=str(default_dup_dist_out))
    parser.add_argument("--duplication-switch-output", default=str(default_dup_switch_out))
    parser.add_argument("--tree-switch-duplications-output", default=str(default_dup_tree_out))
    parser.add_argument("--dendrogram-switch-duplications-output", default=str(default_dup_dendrogram_out))
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    if args.fasta and not args.plots_only:
        plot_sequence_length_distribution(Path(args.fasta), Path(args.length_output))
        print(f"Saved length distribution: {args.length_output}")

    if args.msa and not args.plots_only:
        plot_gap_percentage_per_column(Path(args.msa), Path(args.gap_output))
        print(f"Saved gap profile: {args.gap_output}")

    pairwise_path = Path(args.duplication_pairwise)
    rooted_tree_path = Path(args.rooted_tree)

    if pairwise_path.exists():
        plot_duplication_badasp_distribution(
            raw_pairwise_path=pairwise_path,
            output_svg=Path(args.duplication_distribution_output),
        )
        print(f"Saved duplication score distribution: {args.duplication_distribution_output}")

    if pairwise_path.exists():
        plot_duplication_switch_counts(
            raw_pairwise_path=pairwise_path,
            output_svg=Path(args.duplication_switch_output),
        )
        print(f"Saved duplication switch counts: {args.duplication_switch_output}")

    if rooted_tree_path.exists() and pairwise_path.exists():
        generate_duplication_tree_switch_plot(
            rooted_tree_path=rooted_tree_path,
            raw_pairwise_duplications=pairwise_path,
            output_svg=Path(args.tree_switch_duplications_output),
        )
        print(f"Saved tree switches plot (duplications): {args.tree_switch_duplications_output}")

        generate_duplication_tree_switch_plot(
            rooted_tree_path=rooted_tree_path,
            raw_pairwise_duplications=pairwise_path,
            output_svg=Path(args.dendrogram_switch_duplications_output),
        )
        print(f"Saved dendrogram switches plot (duplications): {args.dendrogram_switch_duplications_output}")


if __name__ == "__main__":
    main()
