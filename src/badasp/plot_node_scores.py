"""Plot BADASP node scores: vertical tree dendrogram, distributions, and relationship plots.

Replaces horizontal tree rendering with a Bio.Phylo + matplotlib vertical
rectangular landscape dendrogram (root at the top, leaves spreading horizontally).
Score relationship scatter plots are saved as rasterised PNG at 300 DPI to handle
560k+ data points, and correlation stats are output to a report file.
Adds a tree scoring diagnostics map showing scored vs. skipped nodes and reasons.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports of the src package
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as stats
from Bio import Phylo


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVENT_COLORS = {
    "Duplication": "#d62728",
    "Speciation": "#1f77b4",
    "Transfer": "#2ca02c",
}
GRAY = "#a0a0a0"


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
def plot_score_distributions(scores_df: pd.DataFrame, out_dir: Path, alignment_path: Path) -> None:
    """Plot statistical distributions of the BADASP scores."""
    from Bio import AlignIO
    
    # 1. Compute MSA column occupancies
    print(f"Loading alignment from {alignment_path} to calculate column occupancies...")
    alignment = AlignIO.read(alignment_path, "fasta")
    num_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()

    if num_seqs == 0:
        print("Warning: alignment is empty; skipping occupancy filtering.")
        return

    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)

    # 2. Filter scores by occupancy >= 80%
    scores_filtered = scores_df.copy()
    scores_filtered["occupancy"] = scores_filtered["position"].map(occupancies)
    scores_filtered = scores_filtered[scores_filtered["occupancy"] >= 0.8].copy()
    
    # Compute max score per node-position comparison
    scores_filtered["max_score"] = scores_filtered[["badasp_score_left", "badasp_score_right"]].max(axis=1)

    # 3. Calculate category-specific 95th percentile of max site scores (similar to event decoupling)
    thresholds = {}
    for ev in ["Duplication", "Speciation", "Transfer"]:
        ev_scores = scores_filtered[scores_filtered["event_type"] == ev]["max_score"].dropna()
        if not ev_scores.empty:
            thresholds[ev] = float(np.percentile(ev_scores, 95))

    # 4. Melt directional scores for violin plot
    scores_melted = pd.melt(
        scores_filtered,
        id_vars=["node_name", "event_type", "position"],
        value_vars=["badasp_score_left", "badasp_score_right"],
        var_name="direction",
        value_name="badasp_score",
    )

    # Violin plot by event type (horizontal orientation)
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=scores_melted, y="event_type", x="badasp_score",
        inner="quartile", palette=EVENT_COLORS,
    )
    # Add vertical dashed lines for category-specific 95th percentile thresholds
    for ev, thresh in thresholds.items():
        color = EVENT_COLORS.get(ev, "black")
        plt.axvline(
            thresh, color=color, linestyle="--", linewidth=1.5,
            label=f"{ev} 95th% ({thresh:.3f})"
        )
    plt.title("Distribution of Directional BADASP Scores (Occupancy >= 80%)")
    plt.xlabel("BADASP Score")
    plt.ylabel("Event Type")
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(out_dir / "score_distribution_by_event.svg", format="svg")
    plt.close()

    # Max score per site histogram (filtered)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=scores_filtered, x="max_score", hue="event_type",
        element="step", stat="density", common_norm=False,
        palette=EVENT_COLORS,
    )
    plt.title("Distribution of Maximum BADASP Scores per Site (Occupancy >= 80%)")
    plt.xlabel("Max BADASP Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / "max_score_distribution.svg", format="svg")
    plt.close()


# ---------------------------------------------------------------------------
# Relationship scatter plots & statistics calculation
# ---------------------------------------------------------------------------
def plot_score_relationships(scores_df: pd.DataFrame, out_dir: Path) -> None:
    """All individual comparison scores vs root distance and log-scaled clade size.

    Outputs:
      - score_relationships.png (rasterised at 300 DPI to handle 560k+ points).
      - score_relationships_stats.txt (text table of correlation values).
    """
    if "distance_from_root" not in scores_df.columns or "clade_size_left" not in scores_df.columns:
        print("Required metrics columns missing from input CSV. Skipping relationship plots.")
        return

    # Melt left/right directions
    df_left = scores_df[
        ["node_name", "event_type", "distance_from_root", "clade_size_left", "badasp_score_left"]
    ].copy()
    df_left.columns = ["node_name", "event_type", "distance_from_root", "clade_size", "badasp_score"]

    df_right = scores_df[
        ["node_name", "event_type", "distance_from_root", "clade_size_right", "badasp_score_right"]
    ].copy()
    df_right.columns = ["node_name", "event_type", "distance_from_root", "clade_size", "badasp_score"]

    all_scores_df = pd.concat([df_left, df_right], ignore_index=True)
    all_scores_df = all_scores_df.dropna(subset=["badasp_score", "distance_from_root", "clade_size"])

    if all_scores_df.empty:
        print("No scored comparisons found. Skipping relationship plots.")
        return

    # Correlation stats
    if len(all_scores_df) > 1:
        pearson_r_dist, pearson_p_dist = stats.pearsonr(
            all_scores_df["distance_from_root"], all_scores_df["badasp_score"],
        )
        spearman_r_dist, spearman_p_dist = stats.spearmanr(
            all_scores_df["distance_from_root"], all_scores_df["badasp_score"],
        )
        log_clade_sizes = np.log10(all_scores_df["clade_size"].clip(lower=1))
        pearson_r_size, pearson_p_size = stats.pearsonr(
            log_clade_sizes, all_scores_df["badasp_score"],
        )
        spearman_r_size, spearman_p_size = stats.spearmanr(
            all_scores_df["clade_size"], all_scores_df["badasp_score"],
        )
    else:
        pearson_r_dist = spearman_r_dist = 0.0
        pearson_p_dist = spearman_p_dist = 1.0
        pearson_r_size = spearman_r_size = 0.0
        pearson_p_size = spearman_p_size = 1.0

    # Save stats report
    report = (
        "================================================================================\n"
        "BADASP Node Score Relationship Statistics\n"
        "================================================================================\n"
        f"Total comparisons evaluated: {len(all_scores_df)}\n\n"
        "1. Correlation: BADASP Score vs. Distance from Root\n"
        "--------------------------------------------------------------------------------\n"
        f"Pearson correlation coefficient:  {pearson_r_dist:.6f} (p-value: {pearson_p_dist:.2e})\n"
        f"Spearman rank correlation (rho):  {spearman_r_dist:.6f} (p-value: {spearman_p_dist:.2e})\n\n"
        "2. Correlation: BADASP Score vs. Clade Size\n"
        "--------------------------------------------------------------------------------\n"
        f"Pearson corr. (Score vs log10):   {pearson_r_size:.6f} (p-value: {pearson_p_size:.2e})\n"
        f"Spearman rank corr. (raw size):   {spearman_r_size:.6f} (p-value: {spearman_p_size:.2e})\n"
        "================================================================================\n"
    )

    print(report)
    out_file = out_dir / "score_relationships_stats.txt"
    with open(out_file, "w") as f:
        f.write(report)
    print(f"Saved relationship statistics to {out_file}")

    # Plot the scatter plots
    palette = {k: v for k, v in EVENT_COLORS.items()}
    palette["Unresolved"] = "#7f7f7f"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: score vs distance from root
    sns.scatterplot(
        data=all_scores_df, x="distance_from_root", y="badasp_score",
        hue="event_type", palette=palette, alpha=0.4, s=8,
        ax=axes[0], rasterized=True,
    )
    if len(all_scores_df) > 1:
        sns.regplot(
            data=all_scores_df, x="distance_from_root", y="badasp_score",
            scatter=False, color="black", ax=axes[0],
        )
    axes[0].set_title("BADASP Score vs. Distance from Root (All Scores)")
    axes[0].set_xlabel("Distance from Root (branch length)")
    axes[0].set_ylabel("BADASP Score")
    stats_text_dist = (
        f"Pearson r = {pearson_r_dist:.3f} (p = {pearson_p_dist:.1e})\n"
        f"Spearman \u03c1 = {spearman_r_dist:.3f} (p = {spearman_p_dist:.1e})"
    )
    axes[0].text(
        0.05, 0.95, stats_text_dist,
        transform=axes[0].transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel 2: score vs clade size (log)
    sns.scatterplot(
        data=all_scores_df, x="clade_size", y="badasp_score",
        hue="event_type", palette=palette, alpha=0.4, s=8,
        ax=axes[1], rasterized=True,
    )
    axes[1].set_xscale("log")
    if len(all_scores_df) > 1:
        sns.regplot(
            data=all_scores_df, x="clade_size", y="badasp_score",
            scatter=False, color="black", logx=True, ax=axes[1],
        )
    axes[1].set_title("BADASP Score vs. Clade Size (All Scores, Log scale)")
    axes[1].set_xlabel("Number of Leaves in Clade (Log scale)")
    axes[1].set_ylabel("BADASP Score")
    stats_text_size = (
        f"Pearson r (log10) = {pearson_r_size:.3f} (p = {pearson_p_size:.1e})\n"
        f"Spearman \u03c1 = {spearman_r_size:.3f} (p = {spearman_p_size:.1e})"
    )
    axes[1].text(
        0.05, 0.95, stats_text_size,
        transform=axes[1].transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_dir / "score_relationships.png", format="png", dpi=300)
    plt.close()
    print(f"Saved relationship plot to {out_dir / 'score_relationships.png'}")


# ---------------------------------------------------------------------------
# Vertical rectangular tree dendrogram (Bio.Phylo + matplotlib)
# ---------------------------------------------------------------------------
def _build_y_positions(tree) -> Dict:
    """Assign evenly-spaced positions to all leaf nodes, then internal nodes
    get the midpoint of their children. Returns {clade: pos}."""
    terminals = tree.get_terminals(order="preorder")
    positions: Dict = {}
    for idx, leaf in enumerate(terminals):
        positions[leaf] = idx

    def _assign_internal(clade):
        if clade.is_terminal():
            return positions[clade]
        child_pos = [_assign_internal(c) for c in clade.clades]
        pos = (min(child_pos) + max(child_pos)) / 2.0
        positions[clade] = pos
        return pos

    _assign_internal(tree.root)
    return positions


def plot_tree_mapping(tree_path: Path, scores_df: pd.DataFrame, out_dir: Path) -> None:
    """Draw a wide vertical rectangular dendrogram with neutral branches and
    node scatter markers colored by event type matching the legacy Bio.Phylo style.
    
    Root is at the TOP, leaves grow downward. Canvas size matches the legacy
    (90, 24) layout to prevent vertical compression and ensure readability.
    """
    _parent_cache.clear()
    tree = Phylo.read(str(tree_path), "newick")

    # --- build per-node score / event lookups ---
    scores_df = scores_df.copy()
    scores_df["max_badasp_score"] = scores_df[["badasp_score_left", "badasp_score_right"]].max(axis=1)
    node_max_scores = scores_df.groupby("node_name")["max_badasp_score"].max().to_dict()
    node_events = scores_df.groupby("node_name")["event_type"].first().to_dict()

    global_max = max(node_max_scores.values()) if node_max_scores else 1.0

    # --- coordinate helpers ---
    x_positions = _build_y_positions(tree)  # topological spread
    n_leaves = len(tree.get_terminals())

    # Figure dimensions matching the legacy (90, 24) layout.
    # This provides a 3.75:1 aspect ratio which displays beautifully on screen
    # without flattening the tree, while maintaining huge resolution for horizontal zoom.
    fig_width = 90
    fig_height = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Helper function for branch length depth
    def _y_depth(clade) -> float:
        return tree.distance(clade)

    # --- draw rectangular branches in neutral gray ---
    for clade in tree.find_clades(order="preorder"):
        x = x_positions[clade]
        y = _y_depth(clade)

        # Draw horizontal connector at parent depth connecting all its children
        if not clade.is_terminal():
            child_xs = [x_positions[c] for c in clade.clades]
            x_min, x_max = min(child_xs), max(child_xs)
            ax.plot(
                [x_min, x_max], [y, y],
                color="#B0B0B0",
                linewidth=0.7,
                solid_capstyle="round",
                zorder=1
            )

        # Draw vertical branch from parent depth to child depth at child's X position
        parent = _find_parent(tree, clade)
        if parent is not None:
            y_parent = _y_depth(parent)
            ax.plot(
                [x, x], [y_parent, y],
                color="#B0B0B0",
                linewidth=0.7,
                solid_capstyle="round",
                zorder=1
            )

    # --- scored node markers grouped by event type ---
    event_markers = {
        "Duplication": {"xs": [], "ys": [], "sizes": []},
        "Speciation": {"xs": [], "ys": [], "sizes": []},
        "Transfer": {"xs": [], "ys": [], "sizes": []},
    }

    for clade in tree.find_clades(order="preorder"):
        name = clade.name or ""
        if name in node_max_scores:
            score = node_max_scores[name]
            event = node_events.get(name, "Speciation")
            if event not in event_markers:
                event_markers[event] = {"xs": [], "ys": [], "sizes": []}
            
            x = x_positions[clade]
            y = _y_depth(clade)
            
            # Sizing scale: significantly reduced to prevent obscuring details and overlapping
            s = 15.0 + 120.0 * (score / global_max)
            
            event_markers[event]["xs"].append(x)
            event_markers[event]["ys"].append(y)
            event_markers[event]["sizes"].append(s)

    for event, data in event_markers.items():
        if data["xs"]:
            ax.scatter(
                data["xs"], data["ys"],
                s=data["sizes"],
                color=EVENT_COLORS.get(event, GRAY),
                edgecolor="black",
                linewidths=0.4,
                alpha=0.8,
                zorder=10,
            )

    # --- legend ---
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Duplication (Red)",
               markerfacecolor=EVENT_COLORS["Duplication"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Speciation (Blue)",
               markerfacecolor=EVENT_COLORS["Speciation"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Transfer (Green)",
               markerfacecolor=EVENT_COLORS["Transfer"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], color="#B0B0B0", linewidth=1.5, label="Branches"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, frameon=True, facecolor="white", edgecolor="none")

    ax.set_ylabel("Branch length distance from root (depth)", fontsize=16)
    ax.set_xlabel("Taxa / internal nodes (topological spread)", fontsize=16)
    ax.set_title(
        "BADASP Node Score Mapping (Red=Dup, Blue=Spec, Green=Trans)",
        fontsize=22, pad=20,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    
    # Calculate max depth and add padding above the root (Y=0) to prevent clipping
    max_depth = max(_y_depth(c) for c in tree.find_clades())
    padding = max(0.02, max_depth * 0.02)
    # Invert Y-axis manually with padding (root at top with padding, leaves at bottom with padding)
    ax.set_ylim(max_depth + padding, -padding)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    out_svg = out_dir / "tree_score_mapping.svg"
    out_png = out_dir / "tree_score_mapping.png"
    
    # Save high-resolution outputs
    fig.savefig(str(out_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tree SVG: {out_svg}")
    print(f"  Tree PNG: {out_png}")


# ---------------------------------------------------------------------------
# Node Scoring Diagnostics Map (All internal nodes Scored vs Skipped Reasons)
# ---------------------------------------------------------------------------
def plot_scoring_diagnostics(tree_path: Path, scores_df: pd.DataFrame, out_dir: Path) -> None:
    """Draw a vertical landscape dendrogram mapping all internal nodes, colored
    by whether they were scored or why they were skipped.
    
    Outputs:
      - tree_scoring_diagnostics.svg
      - tree_scoring_diagnostics.png (300 DPI)
    """
    from ete3 import Tree as EteTree
    
    stem = tree_path.stem
    family_name = stem.removesuffix("_rooted")
    root_dir = Path.cwd()
    alerax_tree_path = root_dir / "results" / "reconciliation" / "alerax" / family_name / "reconciliations" / f"{family_name}.nwk"
    state_path = tree_path.parent / f"{family_name}.state"

    if not alerax_tree_path.exists() or not state_path.exists():
        print(f"Required reconciled tree ({alerax_tree_path}) or state file ({state_path}) missing. Skipping diagnostics plot.")
        return

    print("Generating scoring diagnostics tree plot...")
    _parent_cache.clear()
    
    # Load state file and trees
    from src.badasp.scoring import load_state_file, _reconstruct_ancestral_sequence_from_state
    state_data = load_state_file(state_path)
    alerax_tree = EteTree(str(alerax_tree_path), format=1)
    tree = Phylo.read(str(tree_path), "newick")

    # Replicate event mapping
    alerax_events = {}
    samples_path = alerax_tree_path.parent / "all" / f"{family_name}_samples.newick"
    
    if samples_path.exists():
        clade_event_counts = {}
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample_tree = EteTree(line, format=1)
                    for node in sample_tree.traverse():
                        if not node.is_leaf() and node.name in {"S", "D", "T"}:
                            sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
                            if sig not in clade_event_counts:
                                clade_event_counts[sig] = Counter()
                            clade_event_counts[sig][node.name] += 1
                except Exception:
                    continue
        
        for node in alerax_tree.traverse():
            if not node.is_leaf():
                sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
                if sig in clade_event_counts:
                    majority_ev, _ = clade_event_counts[sig].most_common(1)[0]
                    if majority_ev == "D":
                        alerax_events[sig] = "Duplication"
                    elif majority_ev == "S":
                        alerax_events[sig] = "Speciation"
                    elif majority_ev == "T":
                        alerax_events[sig] = "Transfer"
                    else:
                        alerax_events[sig] = "Unresolved"
                else:
                    alerax_events[sig] = "Unresolved"
    else:
        for node in alerax_tree.traverse():
            if not node.is_leaf():
                sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
                ev = getattr(node, "Ev", "Unresolved")
                if ev == "D":
                    alerax_events[sig] = "Duplication"
                elif ev == "S":
                    alerax_events[sig] = "Speciation"
                elif ev == "T":
                    alerax_events[sig] = "Transfer"
                else:
                    alerax_events[sig] = "Unresolved"

    # Map ASR Nodes
    asr_tree = EteTree(str(tree_path), format=1)
    asr_nodes_to_events = {}
    leaf_sig_to_event = {}  # bio.Phylo nodes may be unnamed; key by leaf-sig instead
    idx = 1
    for node in asr_tree.traverse("preorder"):
        if not node.is_leaf():
            if not node.name:
                node.name = f"Node{idx}"
            idx += 1
            sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
            ev = alerax_events.get(sig, "Unresolved")
            asr_nodes_to_events[node.name] = ev
            leaf_sig_to_event[sig] = ev

    ancestral_seqs = {}
    for node_name in asr_nodes_to_events:
        seq = _reconstruct_ancestral_sequence_from_state(state_data, node_name)
        if seq:
            ancestral_seqs[node_name] = seq

    # Classify each node in Phylo tree
    node_categories = {}
    min_clade_size = 5
    
    for clade in tree.find_clades(order="preorder"):
        if clade.is_terminal():
            continue
            
        # 1. Polytomy check
        if len(clade.clades) != 2:
            node_categories[clade] = "Polytomy"
            continue
            
        left, right = clade.clades
        left_name = left.name
        right_name = right.name
        
        # 2. Unnamed children
        if not left_name or not right_name:
            node_categories[clade] = "Unnamed Children"
            continue
            
        # 3. Event type check (Bio.Phylo may leave internal nodes unnamed; look up by leaf sig)
        clade_sig = tuple(sorted(t.name for t in clade.get_terminals() if t.name))
        event_type = leaf_sig_to_event.get(clade_sig, "Unresolved")
        if event_type not in {"Speciation", "Duplication", "Transfer"}:
            node_categories[clade] = "Unresolved/Ignored Event"
            continue

        left_leaves = len(left.get_terminals())
        right_leaves = len(right.get_terminals())
        
        # 4. Clade size filter
        if left_leaves < min_clade_size or right_leaves < min_clade_size:
            node_categories[clade] = "Clade Size Filter (< 5 leaves)"
            continue

        # 5. Missing ASR sequence
        left_sequence = ancestral_seqs.get(left_name, "")
        right_sequence = ancestral_seqs.get(right_name, "")
        if not left_sequence or not right_sequence:
            node_categories[clade] = "Missing ASR Sequence"
            continue
            
        # If passed all, it is scored!
        node_categories[clade] = "Scored"

    # --- coordinates ---
    x_positions = _build_y_positions(tree)

    fig_width = 90
    fig_height = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def _y_depth(clade) -> float:
        return tree.distance(clade)

    # --- draw rectangular branches in neutral gray ---
    for clade in tree.find_clades(order="preorder"):
        x = x_positions[clade]
        y = _y_depth(clade)

        if not clade.is_terminal():
            child_xs = [x_positions[c] for c in clade.clades]
            x_min, x_max = min(child_xs), max(child_xs)
            ax.plot(
                [x_min, x_max], [y, y],
                color="#B0B0B0",
                linewidth=0.7,
                solid_capstyle="round",
                zorder=1
            )

        parent = _find_parent(tree, clade)
        if parent is not None:
            y_parent = _y_depth(parent)
            ax.plot(
                [x, x], [y_parent, y],
                color="#B0B0B0",
                linewidth=0.7,
                solid_capstyle="round",
                zorder=1
            )

    # --- node markers grouped by diagnostics category ---
    diag_colors = {
        "Scored": "#2ca02c",                         # Green
        "Clade Size Filter (< 5 leaves)": "#d62728", # Red
        "Polytomy": "#ff7f0e",                       # Orange
        "Unresolved/Ignored Event": "#9467bd",        # Purple
        "Unnamed Children": "#8c564b",               # Brown
        "Missing ASR Sequence": "#e377c2",           # Pink
    }
    
    diag_markers = {k: {"xs": [], "ys": [], "sizes": []} for k in diag_colors}

    for clade, category in node_categories.items():
        x = x_positions[clade]
        y = _y_depth(clade)
        
        # Select marker size
        if category == "Scored":
            s = 35.0
        elif category == "Clade Size Filter (< 5 leaves)":
            s = 8.0  # very small to avoid cluttering the 18k dots
        else:
            s = 15.0
            
        diag_markers[category]["xs"].append(x)
        diag_markers[category]["ys"].append(y)
        diag_markers[category]["sizes"].append(s)

    for category, data in diag_markers.items():
        if data["xs"]:
            is_clade_filter = (category == "Clade Size Filter (< 5 leaves)")
            ax.scatter(
                data["xs"], data["ys"],
                s=data["sizes"],
                color=diag_colors[category],
                edgecolor="black" if not is_clade_filter else "none",
                linewidths=0.2,
                alpha=0.5 if is_clade_filter else 0.85,
                zorder=10 if category == "Scored" else 5,
            )

    # --- legend ---
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Scored (Green)",
               markerfacecolor=diag_colors["Scored"], markersize=10, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Clade Size Filter (< 5 leaves) (Red)",
               markerfacecolor=diag_colors["Clade Size Filter (< 5 leaves)"], markersize=6, markeredgecolor="none"),
        Line2D([0], [0], marker="o", color="w", label="Polytomy (Orange)",
               markerfacecolor=diag_colors["Polytomy"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Unresolved/Ignored Event (Purple)",
               markerfacecolor=diag_colors["Unresolved/Ignored Event"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], color="#D0D0D0", linewidth=1.5, label="Branches"),
    ]
    
    # Add other active categories to legend if they occurred
    for cat in ["Unnamed Children", "Missing ASR Sequence"]:
        if diag_markers[cat]["xs"]:
            legend_elements.insert(-1, Line2D([0], [0], marker="o", color="w", label=f"{cat}",
                   markerfacecolor=diag_colors[cat], markersize=8, markeredgecolor="black", markeredgewidth=0.4))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, frameon=True, facecolor="white", edgecolor="none")

    ax.set_ylabel("Branch length distance from root (depth)", fontsize=16)
    ax.set_xlabel("Taxa / internal nodes (topological spread)", fontsize=16)
    ax.set_title(
        "BADASP Node Scoring Diagnostics Map (Scored vs. Skipped Reasons)",
        fontsize=22, pad=20,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    
    # Y-axis padding
    max_depth = max(_y_depth(c) for c in tree.find_clades())
    padding = max(0.02, max_depth * 0.02)
    ax.set_ylim(max_depth + padding, -padding)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    out_svg = out_dir / "tree_scoring_diagnostics.svg"
    out_png = out_dir / "tree_scoring_diagnostics.png"
    
    fig.savefig(str(out_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostics Tree SVG: {out_svg}")
    print(f"  Diagnostics Tree PNG: {out_png}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_parent_cache: Dict = {}


def _find_parent(tree, child_clade):
    """Return the parent clade of *child_clade*, or None for root."""
    if not _parent_cache:
        # Build the cache once
        for clade in tree.find_clades(order="preorder"):
            for c in clade.clades:
                _parent_cache[id(c)] = clade
    return _parent_cache.get(id(child_clade))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot BADASP Node Scores")
    parser.add_argument("--scores", type=Path, required=True, help="Path to raw_node_scores.csv")
    parser.add_argument("--tree", type=Path, required=True, help="Path to ASR treefile")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"), help="Path to trimmed alignment FASTA")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)

    print("Generating statistical plots...")
    plot_score_distributions(scores, args.outdir, args.alignment)

    print("Generating relationship plots and statistics...")
    plot_score_relationships(scores, args.outdir)

    print("Generating tree mapping plot...")
    plot_tree_mapping(args.tree, scores, args.outdir)
    
    # Generate tree scoring diagnostics map (Scored vs. Skipped reasons)
    plot_scoring_diagnostics(args.tree, scores, args.outdir)
    
    print(f"Plots saved to {args.outdir}")
