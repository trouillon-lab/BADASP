#!/usr/bin/env python3
"""
plot_full_classification_tree.py

Generates a vertical rectangular dendrogram of the gene tree where every internal
node is colored by its evolutionary classification (Speciation, Duplication,
Transfer, or Unresolved) from the AleRax reconciliation.
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Bio import Phylo
from ete3 import Tree as EteTree

# Define visual style palettes matching the pipeline
EVENT_COLORS = {
    "Duplication": "#d62728", # Red
    "Speciation": "#1f77b4",  # Blue
    "Transfer": "#2ca02c",    # Green
    "Unresolved": "#a0a0a0",  # Gray
}

_parent_cache: Dict = {}

def _find_parent(tree, child_clade):
    """Return the parent clade of *child_clade*, or None for root."""
    if not _parent_cache:
        for clade in tree.find_clades(order="preorder"):
            for c in clade.clades:
                _parent_cache[id(c)] = clade
    return _parent_cache.get(id(child_clade))

def _build_y_positions(tree) -> Dict:
    """Assign positions to leaves and midpoints to internal nodes."""
    terminals = tree.get_terminals(order="preorder")
    positions = {}
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

def main():
    parser = argparse.ArgumentParser(description="Plot Full Classification Tree Dendrogram")
    parser.add_argument(
        "--tree", type=Path, default=Path("data/interim/iqtree_asr/IPR019888.treefile"),
        help="Path to ASR treefile."
    )
    parser.add_argument(
        "--alerax-tree", type=Path, default=Path("results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk"),
        help="Path to AleRax reconciled tree."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("results/badasp_scoring/plots"),
        help="Output directory for the plots."
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading trees...")
    tree = Phylo.read(str(args.tree), "newick")
    alerax_tree = EteTree(str(args.alerax_tree), format=1)

    # 1. Map AleRax majority event types using leaf signatures
    print("Mapping AleRax events...")
    alerax_events = {}
    family_name = args.alerax_tree.stem
    samples_path = args.alerax_tree.parent / "all" / f"{family_name}_samples.newick"
    
    if samples_path.exists():
        print(f"Parsing samples from {samples_path}...")
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

    # 2. Map ASR Nodes to classifications
    asr_tree = EteTree(str(args.tree), format=1)
    asr_nodes_to_events = {}
    idx = 1
    for node in asr_tree.traverse("preorder"):
        if not node.is_leaf():
            if not node.name:
                node.name = f"Node{idx}"
            idx += 1
            sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
            asr_nodes_to_events[node.name] = alerax_events.get(sig, "Unresolved")

    # 3. Build coordinates
    _parent_cache.clear()
    x_positions = _build_y_positions(tree)

    # Use the same wide landscape layout as tree_score_mapping.png
    fig_width = 90
    fig_height = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def _y_depth(clade) -> float:
        return tree.distance(clade)

    # 4. Draw rectangular branches
    print("Drawing branches...")
    for clade in tree.find_clades(order="preorder"):
        x = x_positions[clade]
        y = _y_depth(clade)

        if not clade.is_terminal():
            child_xs = [x_positions[c] for c in clade.clades]
            x_min, x_max = min(child_xs), max(child_xs)
            ax.plot(
                [x_min, x_max], [y, y],
                color="#D0D0D0",
                linewidth=0.5,
                solid_capstyle="round",
                zorder=1
            )

        parent = _find_parent(tree, clade)
        if parent is not None:
            y_parent = _y_depth(parent)
            ax.plot(
                [x, x], [y_parent, y],
                color="#D0D0D0",
                linewidth=0.5,
                solid_capstyle="round",
                zorder=1
            )

    # 5. Group node markers by event classification
    print("Plotting node markers...")
    markers = {k: {"xs": [], "ys": [], "sizes": []} for k in EVENT_COLORS}

    for clade in tree.find_clades(order="preorder"):
        if clade.is_terminal():
            continue
        
        event = asr_nodes_to_events.get(clade.name, "Unresolved")
        x = x_positions[clade]
        y = _y_depth(clade)
        
        # Omit unresolved nodes or make them extremely small to avoid clutter
        if event == "Unresolved":
            s = 3.0
            alpha = 0.2
        else:
            s = 15.0
            alpha = 0.8
            
        markers[event]["xs"].append(x)
        markers[event]["ys"].append(y)
        markers[event]["sizes"].append(s)

    # Plot each event type
    for event, data in markers.items():
        if data["xs"]:
            is_unresolved = (event == "Unresolved")
            ax.scatter(
                data["xs"], data["ys"],
                s=data["sizes"],
                color=EVENT_COLORS[event],
                edgecolor="black" if not is_unresolved else "none",
                linewidths=0.3,
                alpha=0.3 if is_unresolved else 0.8,
                zorder=5 if is_unresolved else 10,
            )

    # 6. Add legend and labels
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Duplication (Red)",
               markerfacecolor=EVENT_COLORS["Duplication"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Speciation (Blue)",
               markerfacecolor=EVENT_COLORS["Speciation"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Transfer (Green)",
               markerfacecolor=EVENT_COLORS["Transfer"], markersize=8, markeredgecolor="black", markeredgewidth=0.4),
        Line2D([0], [0], marker="o", color="w", label="Unresolved / Other (Gray)",
               markerfacecolor=EVENT_COLORS["Unresolved"], markersize=5, markeredgecolor="none"),
        Line2D([0], [0], color="#D0D0D0", linewidth=1.5, label="Branches"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, frameon=True, facecolor="white", edgecolor="none")

    ax.set_ylabel("Branch length distance from root (depth)", fontsize=16)
    ax.set_xlabel("Taxa / internal nodes (topological spread)", fontsize=16)
    ax.set_title(
        "BADASP Gene Tree Full Reconciliation Classification (Red=Dup, Blue=Spec, Green=Trans)",
        fontsize=22, pad=20,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    
    # Y-axis padding and inversion
    max_depth = max(_y_depth(c) for c in tree.find_clades())
    padding = max(0.02, max_depth * 0.02)
    ax.set_ylim(max_depth + padding, -padding)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    out_svg = args.outdir / "tree_full_classification.svg"
    out_png = args.outdir / "tree_full_classification.png"
    
    fig.savefig(str(out_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Success! Saved full classification tree to:")
    print(f"  SVG: {out_svg}")
    print(f"  PNG: {out_png}")

if __name__ == "__main__":
    main()
