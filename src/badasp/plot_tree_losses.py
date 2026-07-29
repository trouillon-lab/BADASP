"""Plot gene loss events (Speciation Losses and Transfer Losses) on the gene tree dendrogram.

Maps branch-specific loss events from AleRax sample reconciliations (recPhyloXML) 
onto the branches of the vertical rectangular gene tree dendrogram.
Saves the branch-wise loss counts to a CSV file and generates high-quality SVG/PNG plots.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports of the src package
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Bio import Phylo


# ---------------------------------------------------------------------------
# Constants & Colors matching established palette
# ---------------------------------------------------------------------------
COLORS = {
    "SL": "#85c1e9",  # Speciation Losses - Light blue
    "TL": "#82e0aa",  # Transfer Losses - Light green
}
GRAY = "#B0B0B0"


# ---------------------------------------------------------------------------
# Leaf signature mapping
# ---------------------------------------------------------------------------
def build_leaf_sig_map(tree) -> Tuple[Dict[Tuple[str, ...], any], Dict[any, Tuple[str, ...]]]:
    """Precompute leaf signatures for all nodes in the tree to allow fast mapping.
    
    Returns:
        leaf_sig_to_node: Dict mapping sorted leaf tuple signature to tree clade object.
        node_to_leaf_sig: Dict mapping tree clade object to sorted leaf tuple signature.
    """
    leaf_sig_to_node = {}
    node_to_leaf_sig = {}

    def _traverse(clade):
        if clade.is_terminal():
            sig = (clade.name,)
            leaf_sig_to_node[sig] = clade
            node_to_leaf_sig[clade] = sig
            return [clade.name]
        
        leaves = []
        for child in clade.clades:
            leaves.extend(_traverse(child))
        
        sig = tuple(sorted(leaves))
        leaf_sig_to_node[sig] = clade
        node_to_leaf_sig[clade] = sig
        return leaves

    _traverse(tree.root)
    return leaf_sig_to_node, node_to_leaf_sig


# ---------------------------------------------------------------------------
# XML Parsing and Loss Extraction
# ---------------------------------------------------------------------------
def get_surviving_leaves(clade_elem, ns) -> List[str]:
    """Recursively extract all surviving leaf names under an XML clade."""
    leaves = []
    name_elem = clade_elem.find('ns:name', ns)
    if name_elem is not None and name_elem.text == 'loss':
        return leaves
    
    children = clade_elem.findall('ns:clade', ns)
    if not children:
        if name_elem is not None and name_elem.text is not None and name_elem.text != 'NULL':
            leaves.append(name_elem.text)
    else:
        for child in children:
            leaves.extend(get_surviving_leaves(child, ns))
    return leaves


def parse_sample_losses(xml_path: Path, leaf_sig_to_node: Dict, ns: Dict) -> Tuple[List[Tuple[Tuple[str, ...], str]], int]:
    """Parse a single recPhyloXML sample file and extract all loss events.
    
    Returns:
        A list of tuples: (sibling_leaf_signature, loss_type)
        where loss_type is 'SL' or 'TL'.
        Also returns the count of unmapped losses.
    """
    try:
        tree_xml = ET.parse(xml_path)
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None, 0
        
    root_xml = tree_xml.getroot()
    rec_gene_tree = root_xml.find('.//ns:recGeneTree', ns)
    if rec_gene_tree is None:
        return [], 0

    # Build parent map
    parent_map = {c: p for p in rec_gene_tree.iter() for c in p}
    
    sample_events = []
    unmapped = 0

    for clade in rec_gene_tree.findall('.//ns:clade', ns):
        name_elem = clade.find('ns:name', ns)
        if name_elem is not None and name_elem.text == 'loss':
            parent = parent_map.get(clade)
            if parent is not None:
                # Sibling of the loss representing the surviving lineage
                siblings = [c for c in parent.findall('ns:clade', ns) if c != clade]
                if not siblings:
                    continue
                sibling = siblings[0]
                
                # Get surviving leaves of sibling
                surv_leaves = get_surviving_leaves(sibling, ns)
                if not surv_leaves:
                    continue
                
                sig = tuple(sorted(surv_leaves))
                
                # Determine loss type (Speciation Loss vs Transfer Loss)
                parent_events_rec = parent.find('ns:eventsRec', ns)
                is_tl = False
                if parent_events_rec is not None:
                    p_evs = {child.tag.split('}')[-1] for child in parent_events_rec}
                    if 'transferBack' in p_evs or 'branchingOut' in p_evs:
                        is_tl = True
                
                loss_type = 'TL' if is_tl else 'SL'
                
                # Verify that the sibling signature exists in our master tree
                if sig in leaf_sig_to_node:
                    sample_events.append((sig, loss_type))
                else:
                    unmapped += 1
                    
    return sample_events, unmapped


# ---------------------------------------------------------------------------
# Coordinate System matching plot_node_scores.py
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


_parent_cache: Dict = {}

def _find_parent(tree, child_clade):
    """Return the parent clade of *child_clade*, or None for root."""
    if not _parent_cache:
        for clade in tree.find_clades(order="preorder"):
            for c in clade.clades:
                _parent_cache[id(c)] = clade
    return _parent_cache.get(id(child_clade))


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Reconciled Gene Losses on Dendrogram")
    parser.add_argument("--tree", type=Path, required=True, help="Path to ASR master treefile")
    parser.add_argument("--xml-dir", type=Path, required=True, help="Path to directory containing recPhyloXML samples")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--min-freq", type=float, default=0.05, help="Minimum sample frequency to plot loss marker (default 0.05)")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    analyses_dir = args.outdir.parent / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Master Tree
    print("Loading master tree...")
    tree = Phylo.read(str(args.tree), "newick")
    print(f"Master tree loaded with {len(tree.get_terminals())} leaves.")

    # 2. Precompute Leaf Signature Maps
    print("Precomputing leaf signatures...")
    leaf_sig_to_node, node_to_leaf_sig = build_leaf_sig_map(tree)
    print(f"Leaf signatures built for {len(leaf_sig_to_node)} nodes.")

    # 3. Find XML files
    xml_files = sorted(list(args.xml_dir.glob("*_sample_*.xml")))
    n_samples = len(xml_files)
    if n_samples == 0:
        print(f"Error: No recPhyloXML sample files found in {args.xml_dir}")
        sys.exit(1)
    print(f"Found {n_samples} recPhyloXML sample files.")

    # 4. Parse XML files and aggregate counts
    print("Parsing sample files and mapping losses...")
    ns = {'ns': 'http://www.recg.org'}

    # Track counts of SL and TL per branch (identified by sibling signature in master tree)
    branch_sl_counts = {}
    branch_tl_counts = {}
    total_unmapped = 0
    total_losses_parsed = 0
    n_parsed = 0  # only count successfully parsed files for frequency denominator

    for idx, xml_path in enumerate(xml_files):
        if (idx + 1) % 10 == 0 or (idx + 1) == n_samples:
            print(f"  Parsed {idx + 1}/{n_samples} samples...")

        events, unmapped = parse_sample_losses(xml_path, leaf_sig_to_node, ns)
        if events is None:
            continue  # XML parse failure; skip but don't count toward denominator
        n_parsed += 1
        total_unmapped += unmapped
        total_losses_parsed += len(events) + unmapped

        for sig, loss_type in events:
            if loss_type == 'SL':
                branch_sl_counts[sig] = branch_sl_counts.get(sig, 0) + 1
            else:
                branch_tl_counts[sig] = branch_tl_counts.get(sig, 0) + 1

    print("Mapping and aggregation completed!")
    print(f"Total losses parsed: {total_losses_parsed}")
    print(f"Successfully mapped SL: {sum(branch_sl_counts.values())}")
    print(f"Successfully mapped TL: {sum(branch_tl_counts.values())}")
    print(f"Unmapped losses (topological mismatch): {total_unmapped}")

    # 5. Assign names to internal nodes if unnamed (matching diagnostics map)
    # This ensures every node in the master tree has a name for CSV reporting
    idx = 1
    for node in tree.find_clades(order="preorder"):
        if not node.is_terminal():
            if not node.name:
                node.name = f"Node{idx}"
            idx += 1

    # 6. Save Branch-wise Loss Counts to CSV
    print("Saving loss counts to CSV...")
    csv_rows = []
    
    # We want to export a clean report of every node's incoming branch event rates
    for node in tree.find_clades(order="preorder"):
        if node == tree.root:
            continue  # Root has no incoming branch
            
        sig = node_to_leaf_sig.get(node)
        sl_c = branch_sl_counts.get(sig, 0)
        tl_c = branch_tl_counts.get(sig, 0)
        
        # Only write rows that have at least one loss to keep the file reasonably sized
        if sl_c > 0 or tl_c > 0:
            csv_rows.append({
                "node_name": node.name or "Unnamed",
                "is_leaf": 1 if node.is_terminal() else 0,
                "speciation_losses": sl_c,
                "transfer_losses": tl_c,
                "total_losses": sl_c + tl_c,
                "speciation_loss_frequency": sl_c / n_parsed if n_parsed > 0 else 0.0,
                "transfer_loss_frequency": tl_c / n_parsed if n_parsed > 0 else 0.0,
                "total_loss_frequency": (sl_c + tl_c) / n_parsed if n_parsed > 0 else 0.0
            })
            
    df_losses = pd.DataFrame(csv_rows)
    if not df_losses.empty:
        df_losses = df_losses.sort_values(by="total_losses", ascending=False)
    
    csv_out = analyses_dir / "branch_loss_counts.csv"
    df_losses.to_csv(csv_out, index=False)
    print(f"Saved branch-wise loss counts to {csv_out}")

    # 7. Generate vertical rectangular dendrogram
    print("Generating tree loss mapping plot...")
    x_positions = _build_y_positions(tree)
    
    fig_width = 90
    fig_height = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def _y_depth(clade) -> float:
        return tree.distance(clade)

    # Draw rectangular branches in neutral gray
    for clade in tree.find_clades(order="preorder"):
        x = x_positions[clade]
        y = _y_depth(clade)

        # Draw horizontal connector
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

        # Draw vertical branch
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

    # Plot loss markers on the branches
    min_count = args.min_freq * n_samples
    
    sl_plot_data = {"xs": [], "ys": [], "sizes": [], "alphas": []}
    tl_plot_data = {"xs": [], "ys": [], "sizes": [], "alphas": []}
    
    for clade in tree.find_clades(order="preorder"):
        parent = _find_parent(tree, clade)
        if parent is None:
            continue
            
        sig = node_to_leaf_sig.get(clade)
        sl_count = branch_sl_counts.get(sig, 0)
        tl_count = branch_tl_counts.get(sig, 0)
        
        y = _y_depth(clade)
        y_parent = _y_depth(parent)
        y_mid = (y_parent + y) / 2.0
        x = x_positions[clade]
        
        # We only plot losses above the minimum count threshold
        has_sl = sl_count >= min_count
        has_tl = tl_count >= min_count
        
        if has_sl and has_tl:
            # Sibling has both SL and TL: offset them slightly in X to avoid perfect overlap
            # Offset is 0.25 (since adjacent leaves are at distance 1.0)
            sl_plot_data["xs"].append(x - 0.25)
            sl_plot_data["ys"].append(y_mid)
            sl_plot_data["sizes"].append(15.0 + 120.0 * (sl_count / n_samples))
            sl_plot_data["alphas"].append(min(0.9, 0.4 + 0.5 * (sl_count / n_samples)))
            
            tl_plot_data["xs"].append(x + 0.25)
            tl_plot_data["ys"].append(y_mid)
            tl_plot_data["sizes"].append(15.0 + 120.0 * (tl_count / n_samples))
            tl_plot_data["alphas"].append(min(0.9, 0.4 + 0.5 * (tl_count / n_samples)))
            
        elif has_sl:
            sl_plot_data["xs"].append(x)
            sl_plot_data["ys"].append(y_mid)
            sl_plot_data["sizes"].append(15.0 + 120.0 * (sl_count / n_samples))
            sl_plot_data["alphas"].append(min(0.9, 0.4 + 0.5 * (sl_count / n_samples)))
            
        elif has_tl:
            tl_plot_data["xs"].append(x)
            tl_plot_data["ys"].append(y_mid)
            tl_plot_data["sizes"].append(15.0 + 120.0 * (tl_count / n_samples))
            tl_plot_data["alphas"].append(min(0.9, 0.4 + 0.5 * (tl_count / n_samples)))

    # Render SL markers (triangles pointing down 'v')
    if sl_plot_data["xs"]:
        for x, y, s, a in zip(sl_plot_data["xs"], sl_plot_data["ys"], sl_plot_data["sizes"], sl_plot_data["alphas"]):
            ax.scatter(
                [x], [y],
                s=s,
                marker="v",
                color=COLORS["SL"],
                edgecolor="black",
                linewidths=0.3,
                alpha=a,
                zorder=10
            )

    # Render TL markers (triangles pointing down 'v')
    if tl_plot_data["xs"]:
        for x, y, s, a in zip(tl_plot_data["xs"], tl_plot_data["ys"], tl_plot_data["sizes"], tl_plot_data["alphas"]):
            ax.scatter(
                [x], [y],
                s=s,
                marker="v",
                color=COLORS["TL"],
                edgecolor="black",
                linewidths=0.3,
                alpha=a,
                zorder=10
            )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="v", color="w", label="Speciation Loss (Light Blue)",
               markerfacecolor=COLORS["SL"], markersize=10, markeredgecolor="black", markeredgewidth=0.3),
        Line2D([0], [0], marker="v", color="w", label="Transfer Loss (Light Green)",
               markerfacecolor=COLORS["TL"], markersize=10, markeredgecolor="black", markeredgewidth=0.3),
        Line2D([0], [0], color="#B0B0B0", linewidth=1.5, label="Branches"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, frameon=True, facecolor="white", edgecolor="none")

    ax.set_ylabel("Branch length distance from root (depth)", fontsize=16)
    ax.set_xlabel("Taxa / internal nodes (topological spread)", fontsize=16)
    ax.set_title(
        f"Reconciled Gene Loss Events Mapped on Phylogeny (SL=Blue, TL=Green, Min Freq={args.min_freq:.2%})",
        fontsize=22, pad=20,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])

    # Y-axis inversion and padding
    max_depth = max(_y_depth(c) for c in tree.find_clades())
    padding = max(0.02, max_depth * 0.02)
    ax.set_ylim(max_depth + padding, -padding)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    out_svg = args.outdir / "tree_losses_mapping.svg"
    out_png = args.outdir / "tree_losses_mapping.png"

    fig.savefig(str(out_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated loss mapping plots:")
    print(f"  Losses Tree SVG: {out_svg}")
    print(f"  Losses Tree PNG: {out_png}")
