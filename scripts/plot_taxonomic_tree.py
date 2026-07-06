#!/usr/bin/env python3
"""
plot_taxonomic_tree.py

Generates two vertical rectangular dendrograms of the gene tree:
1. Leaves colored by taxonomic Domain (superkingdom: Bacteria, Archaea, Eukaryota).
2. Leaves colored by taxonomic Phylum (Actinomycetota, Bacillota, Pseudomonadota, etc.).
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from Bio import Phylo

try:
    from ete3 import NCBITaxa
except ImportError:
    NCBITaxa = None

# ---------------------------------------------------------------------------
# Taxonomy Resolver with expanded caching (domain + phylum)
# ---------------------------------------------------------------------------
class TaxonomyResolver:
    def __init__(self, cache_path: str = "ncbi_taxonomy_map.csv"):
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            try:
                df = pd.read_csv(self.cache_path, dtype=str).set_index("taxid")
                self.cache = df.to_dict(orient="index")
                print(f"Loaded {len(self.cache)} cached taxonomic records from {self.cache_path}")
            except Exception as e:
                print(f"Warning: Failed to load cache from {self.cache_path}: {e}")

    def resolve_taxids(self, taxids: List[str]) -> None:
        """Resolve missing TaxIDs against NCBI to get superkingdom and phylum."""
        missing_taxids = [
            t for t in taxids 
            if t not in self.cache 
            or self.cache[t].get("superkingdom", "Unknown") == "Unknown" 
            or self.cache[t].get("phylum", "Unknown") == "Unknown"
        ]
        
        if missing_taxids:
            print(f"Resolving {len(missing_taxids)} TaxIDs against NCBI database...")
            if NCBITaxa is None:
                print("Warning: ete3.NCBITaxa is not installed. Defaulting missing taxids to 'Unknown'.")
                for taxid_str in missing_taxids:
                    if taxid_str not in self.cache:
                        self.cache[taxid_str] = {}
                    self.cache[taxid_str]["superkingdom"] = "Unknown"
                    self.cache[taxid_str]["phylum"] = "Unknown"
            else:
                try:
                    ncbi = NCBITaxa()
                    for taxid_str in missing_taxids:
                        try:
                            taxid = int(taxid_str)
                            lineage = ncbi.get_lineage(taxid)
                            names = ncbi.get_taxid_translator(lineage)
                            ranks = ncbi.get_rank(lineage)
                            
                            superkingdom = "Unknown"
                            phylum = "Unknown"
                            for tid, rank in ranks.items():
                                if rank in {"superkingdom", "domain"}:
                                    superkingdom = names[tid]
                                elif rank == "phylum":
                                    phylum = names[tid]
                                    
                            if taxid_str not in self.cache:
                                self.cache[taxid_str] = {}
                            self.cache[taxid_str]["superkingdom"] = superkingdom
                            self.cache[taxid_str]["phylum"] = phylum
                        except Exception:
                            if taxid_str not in self.cache:
                                self.cache[taxid_str] = {}
                            self.cache[taxid_str]["superkingdom"] = "Unknown"
                            self.cache[taxid_str]["phylum"] = "Unknown"
                except Exception as e:
                    print(f"Warning: Failed to query ete3.NCBITaxa: {e}. Defaulting to 'Unknown'.")
                    for taxid_str in missing_taxids:
                        if taxid_str not in self.cache:
                            self.cache[taxid_str] = {}
                        self.cache[taxid_str]["superkingdom"] = "Unknown"
                        self.cache[taxid_str]["phylum"] = "Unknown"
                    
            # Save cache file
            try:
                df_cache = pd.DataFrame.from_dict(self.cache, orient="index")
                df_cache.index.name = "taxid"
                df_cache.reset_index().to_csv(self.cache_path, index=False)
                print(f"Updated taxonomy cache saved to {self.cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save taxonomy cache: {e}")

    def get_domain(self, taxid: str) -> str:
        return self.cache.get(taxid, {}).get("superkingdom", "Unknown")

    def get_phylum(self, taxid: str) -> str:
        return self.cache.get(taxid, {}).get("phylum", "Unknown")

# ---------------------------------------------------------------------------
# Coordinate Helpers
# ---------------------------------------------------------------------------
_parent_cache: Dict = {}

def _find_parent(tree, child_clade):
    if not _parent_cache:
        for clade in tree.find_clades(order="preorder"):
            for c in clade.clades:
                _parent_cache[id(c)] = clade
    return _parent_cache.get(id(child_clade))

def _build_y_positions(tree) -> Dict:
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

def parse_treerecs_mapping(mapping_file: Path) -> Dict[str, str]:
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping

# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------
def plot_taxonomic_dendrogram(
    tree,
    x_positions,
    leaf_colors: List[str],
    legend_elements: List,
    title: str,
    out_svg: Path,
    out_png: Path
) -> None:
    _parent_cache.clear()
    fig_width = 90
    fig_height = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def _y_depth(clade) -> float:
        return tree.distance(clade)

    # 1. Draw rectangular branches in neutral gray
    print("Drawing branches...")
    for clade in tree.find_clades(order="preorder"):
        x = x_positions[clade]
        y = _y_depth(clade)

        if not clade.is_terminal():
            child_xs = [x_positions[c] for c in clade.clades]
            x_min, x_max = min(child_xs), max(child_xs)
            ax.plot(
                [x_min, x_max], [y, y],
                color="#D8D8D8",
                linewidth=0.5,
                solid_capstyle="round",
                zorder=1
            )

        parent = _find_parent(tree, clade)
        if parent is not None:
            y_parent = _y_depth(parent)
            ax.plot(
                [x, x], [y_parent, y],
                color="#D8D8D8",
                linewidth=0.5,
                solid_capstyle="round",
                zorder=1
            )

    # 2. Plot leaf color dots at the tips
    print("Plotting leaf markers...")
    leaf_xs = []
    leaf_ys = []
    for leaf in tree.get_terminals(order="preorder"):
        leaf_xs.append(x_positions[leaf])
        leaf_ys.append(_y_depth(leaf))

    ax.scatter(
        leaf_xs, leaf_ys,
        s=12.0,
        c=leaf_colors,
        edgecolor="none",
        alpha=0.9,
        zorder=10
    )

    # 3. Legend and labels
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, frameon=True, facecolor="white", edgecolor="none")

    ax.set_ylabel("Branch length distance from root (depth)", fontsize=16)
    ax.set_xlabel("Taxa (topological spread)", fontsize=16)
    ax.set_title(title, fontsize=22, pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    
    max_depth = max(_y_depth(c) for c in tree.find_clades())
    padding = max(0.02, max_depth * 0.02)
    ax.set_ylim(max_depth + padding, -padding)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    
    fig.savefig(str(out_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(out_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Plot Taxonomic Leaf-Colored Dendrograms")
    parser.add_argument(
        "--tree", type=Path, default=Path("data/interim/iqtree_asr/IPR019888.treefile"),
        help="Path to ASR treefile."
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("data/interim/alerax/IPR019888.treerecs_mapping.link"),
        help="Path to mapping file."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("results/badasp_scoring/plots"),
        help="Output directory for the plots."
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading tree and mapping...")
    tree = Phylo.read(str(args.tree), "newick")
    gene_to_taxid = parse_treerecs_mapping(args.mapping)

    # 1. Resolve taxonomy for all leaf taxids
    leaf_taxids = list(set(gene_to_taxid.get(leaf.name, "Unknown") for leaf in tree.get_terminals()))
    leaf_taxids = [t for t in leaf_taxids if t != "Unknown"]
    
    resolver = TaxonomyResolver(cache_path="ncbi_taxonomy_map.csv")
    resolver.resolve_taxids(leaf_taxids)

    # 2. Map leaves to Domain and Phylum
    leaves = tree.get_terminals(order="preorder")
    leaf_domains = []
    leaf_phyla = []
    for leaf in leaves:
        taxid = gene_to_taxid.get(leaf.name, "Unknown")
        domain = resolver.get_domain(taxid)
        phylum = resolver.get_phylum(taxid)
        leaf_domains.append(domain)
        leaf_phyla.append(phylum)

    x_positions = _build_y_positions(tree)

    # ---------------------------------------------------------------------------
    # Tree 1: Colored by Domain (Superkingdom)
    # ---------------------------------------------------------------------------
    print("\nProcessing Domain (Superkingdom) Tree...")
    domain_colors_map = {
        "Bacteria": "#1f77b4",   # Blue
        "Archaea": "#ff7f0e",    # Orange
        "Eukaryota": "#2ca02c",  # Green
        "Unknown": "#a0a0a0"     # Gray
    }
    
    # Map any other domain (e.g. Viruses) to Unknown
    leaf_colors_domain = [domain_colors_map.get(d, "#a0a0a0") for d in leaf_domains]
    
    # Count occurrences
    domain_counts = Counter(leaf_domains)
    legend_elements_domain = [
        Line2D([0], [0], marker="o", color="w", label=f"{dom} ({domain_counts[dom]:,} leaves)",
               markerfacecolor=color, markersize=10)
        for dom, color in domain_colors_map.items() if domain_counts[dom] > 0 or dom == "Unknown"
    ]
    
    plot_taxonomic_dendrogram(
        tree, x_positions, leaf_colors_domain, legend_elements_domain,
        title="BADASP Gene Tree Leaves Colored by Taxonomic Domain (Superkingdom)",
        out_svg=args.outdir / "tree_taxonomy_domain.svg",
        out_png=args.outdir / "tree_taxonomy_domain.png"
    )

    # ---------------------------------------------------------------------------
    # Tree 2: Colored by Phylum
    # ---------------------------------------------------------------------------
    print("\nProcessing Phylum Tree...")
    phylum_counts = Counter(leaf_phyla)
    
    # Select top phyla to color, group others into "Other"
    top_phyla = [p for p, _ in phylum_counts.most_common(8) if p != "Unknown"]
    
    # Visually distinct palette for top phyla
    phylum_colors_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#17becf"
    ]
    
    phylum_colors_map = {phylum: phylum_colors_palette[i] for i, phylum in enumerate(top_phyla)}
    phylum_colors_map["Other"] = "#bcbd22"
    phylum_colors_map["Unknown"] = "#a0a0a0"

    leaf_colors_phylum = []
    for p in leaf_phyla:
        if p in phylum_colors_map:
            leaf_colors_phylum.append(phylum_colors_map[p])
        elif p == "Unknown":
            leaf_colors_phylum.append(phylum_colors_map["Unknown"])
        else:
            leaf_colors_phylum.append(phylum_colors_map["Other"])

    # Count "Other" leaves
    other_count = sum(phylum_counts[p] for p in phylum_counts if p not in top_phyla and p != "Unknown")
    
    legend_elements_phylum = []
    for phylum, color in phylum_colors_map.items():
        if phylum == "Other":
            legend_elements_phylum.append(
                Line2D([0], [0], marker="o", color="w", label=f"Other Phyla ({other_count:,} leaves)",
                       markerfacecolor=color, markersize=10)
            )
        elif phylum == "Unknown":
            legend_elements_phylum.append(
                Line2D([0], [0], marker="o", color="w", label=f"Unknown ({phylum_counts['Unknown']:,} leaves)",
                       markerfacecolor=color, markersize=10)
            )
        else:
            legend_elements_phylum.append(
                Line2D([0], [0], marker="o", color="w", label=f"{phylum} ({phylum_counts[phylum]:,} leaves)",
                       markerfacecolor=color, markersize=10)
            )

    plot_taxonomic_dendrogram(
        tree, x_positions, leaf_colors_phylum, legend_elements_phylum,
        title="BADASP Gene Tree Leaves Colored by Taxonomic Phylum (Top 8 + Other)",
        out_svg=args.outdir / "tree_taxonomy_phylum.svg",
        out_png=args.outdir / "tree_taxonomy_phylum.png"
    )

if __name__ == "__main__":
    main()
