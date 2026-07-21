#!/usr/bin/env python
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from Bio import Phylo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def build_x_positions(tree) -> dict:
    """Assign positions to leaves and compute internal node midpoints."""
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

def find_parent(tree, child_clade, cache) -> getattr:
    if not cache:
        for clade in tree.find_clades(order="preorder"):
            for c in clade.clades:
                cache[id(c)] = clade
    return cache.get(id(child_clade))

def draw_tree_on_axis(ax, tree_path: Path, title: str) -> float:
    print(f"Reading tree from {tree_path}...")
    tree = Phylo.read(str(tree_path), "newick")
    x_positions = build_x_positions(tree)
    parent_cache = {}

    # Pre-build parent cache
    find_parent(tree, tree.root, parent_cache)

    # Compute depths in O(N) by traversing preorder and accumulating distances
    print("Computing branch depths...")
    depths = {tree.root: 0.0}
    for clade in tree.find_clades(order="preorder"):
        if clade == tree.root:
            continue
        parent = parent_cache.get(id(clade))
        depths[clade] = depths[parent] + (clade.branch_length or 0.0)

    segments = []
    max_depth = 0.0
    n_leaves = len(tree.get_terminals())

    # Build segments list for LineCollection
    print("Building line segments...")
    for clade in tree.find_clades(order="preorder"):
        x = x_positions[clade]
        y = depths[clade]
        max_depth = max(max_depth, y)

        # Draw horizontal connector at parent depth connecting all children
        if not clade.is_terminal():
            child_xs = [x_positions[c] for c in clade.clades]
            x_min, x_max = min(child_xs), max(child_xs)
            segments.append([(x_min, y), (x_max, y)])

        # Draw vertical branch from parent depth to child depth
        parent = parent_cache.get(id(clade))
        if parent is not None:
            y_parent = depths[parent]
            segments.append([(x, y_parent), (x, y)])

    print("Adding LineCollection to axis...")
    lc = LineCollection(segments, colors="#2C3E50", linewidths=0.4, zorder=1)
    ax.add_collection(lc)

    ax.set_title(title, fontsize=24, pad=15, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    
    # Invert Y-axis manually (root at top, leaves at bottom)
    padding = max(0.02, max_depth * 0.02)
    ax.set_ylim(max_depth + padding, -padding)
    ax.set_xlim(-50, n_leaves + 50)
    return max_depth

def main():
    tree_orig = ROOT / "data/interim/iqtree_asr/IPR019888.treefile"
    tree_rooted = ROOT / "data/interim/iqtree_asr/IPR019888.treefile.rooted"
    out_dir = ROOT / "results/badasp_scoring/plots"
    out_png = out_dir / "rooting_comparison.png"

    if not tree_orig.exists():
        print(f"Original tree {tree_orig} does not exist!")
        sys.exit(1)
    if not tree_rooted.exists() or tree_rooted.stat().st_size == 0:
        print(f"Rooted tree {tree_rooted} does not exist or is empty!")
        sys.exit(1)

    print("Generating rooting comparison plot...")
    fig, axes = plt.subplots(2, 1, figsize=(60, 32), sharex=True)
    fig.patch.set_facecolor("white")

    draw_tree_on_axis(axes[0], tree_orig, "Original ASR Tree (Reconciled Topology, IQ-TREE Branch Lengths, Rooted Arbitrarily by IQ-TREE)")
    draw_tree_on_axis(axes[1], tree_rooted, "MAD-Rooted ASR Tree (Reconciled Topology, IQ-TREE Branch Lengths, Rooted by MAD)")

    # Set labels
    axes[0].set_ylabel("Branch length depth from root", fontsize=18)
    axes[1].set_ylabel("Branch length depth from root", fontsize=18)
    axes[1].set_xlabel("Leaves spread (topological sequence)", fontsize=18)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving comparison plot to {out_png}...")
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
