#!/usr/bin/env python3
"""
visualize_alerax_macro.py

A standalone, pipeline-ready Python script to parse, aggregate, and visualize 
the output results of an AleRax DTL reconciliation run at the taxonomic macro-level.
"""

import os
import sys
import glob
import re
import argparse
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from Bio import Phylo

try:
    from ete3 import NCBITaxa
except ImportError:
    NCBITaxa = None

# Define visual style palettes
COLORS = {
    "S": "#3498db",   # Speciations - Sleek blue
    "SL": "#85c1e9",  # Speciation Losses - Light blue
    "D": "#e74c3c",   # Duplications - Coral red
    "DL": "#f1948a",  # Duplication Losses - Light coral
    "T": "#2ecc71",   # Transfers - Emerald green
    "TL": "#82e0aa",  # Transfer Losses - Light emerald green
    "L": "#95a5a6"    # Losses - Gray
}

def find_file(directory: str, pattern: str) -> str:
    """Helper function to recursively find a file matching a pattern."""
    for root, _, files in os.walk(directory):
        for f in files:
            if glob.fnmatch.fnmatch(f, pattern):
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not find any file matching pattern '{pattern}' in {directory}")

def parse_species_tree(input_dir: str) -> List[str]:
    """Parses species TaxIDs (leaf names) from the starting species tree."""
    try:
        tree_file = find_file(input_dir, "*species_tree.nwk")
    except FileNotFoundError:
        try:
            tree_file = find_file(input_dir, "*.newick")
        except FileNotFoundError:
            # Fallback to general interim directory
            fallbacks = ["data/interim", "data/sandbox_alerax", "results/reconciliation/alerax"]
            tree_file = None
            for fb in fallbacks:
                if os.path.exists(fb):
                    try:
                        tree_file = find_file(fb, "*species_tree.nwk")
                        break
                    except FileNotFoundError:
                        try:
                            tree_file = find_file(fb, "*.newick")
                            break
                        except FileNotFoundError:
                            continue
            if not tree_file:
                raise FileNotFoundError(f"Could not find species tree in {input_dir} or fallbacks.")
        
    tree = Phylo.read(tree_file, "newick")
    return [leaf.name for leaf in tree.get_terminals() if leaf.name]

def parse_treerecs_mapping(input_dir: str) -> Dict[str, str]:
    """Parses the gene-to-species mapping file (*treerecs_mapping.link)."""
    try:
        mapping_file = find_file(input_dir, "*treerecs_mapping.link")
    except FileNotFoundError:
        fallbacks = ["data/interim", "data/sandbox_alerax", "results/reconciliation/alerax"]
        mapping_file = None
        for fb in fallbacks:
            if os.path.exists(fb):
                try:
                    mapping_file = find_file(fb, "*treerecs_mapping.link")
                    break
                except FileNotFoundError:
                    continue
        if not mapping_file:
            raise FileNotFoundError(f"Could not find treerecs_mapping.link in {input_dir} or fallbacks.")
            
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                gene_id, taxid = parts[0], parts[1]
                mapping[gene_id] = taxid
    return mapping

def parse_event_counts_with_errors(input_dir: str) -> pd.DataFrame:
    """Parses tree-wide DTL event counts across all 100 sample files."""
    try:
        search_path = os.path.join(input_dir, "reconciliations", "all", "*_eventCounts_*.txt")
        files = glob.glob(search_path)
        if not files:
            # Fallback search
            all_file = find_file(input_dir, "*_eventCounts_0.txt")
            search_path = os.path.join(os.path.dirname(all_file), "*_eventCounts_*.txt")
            files = glob.glob(search_path)
    except FileNotFoundError:
        # Fallback directory
        fallbacks = ["data/interim", "results/reconciliation/alerax"]
        files = []
        for fb in fallbacks:
            if os.path.exists(fb):
                try:
                    all_file = find_file(fb, "*_eventCounts_0.txt")
                    search_path = os.path.join(os.path.dirname(all_file), "*_eventCounts_*.txt")
                    files = glob.glob(search_path)
                    if files:
                        break
                except FileNotFoundError:
                    continue
                    
    if not files:
        raise FileNotFoundError(f"No eventCounts files found under {input_dir} or fallbacks.")
        
    records = []
    for f in files:
        record = {}
        with open(f, "r") as file:
            for line in file:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                record[key] = int(val)
        records.append(record)
        
    df = pd.DataFrame(records)
    for event in ["S", "SL", "D", "DL", "T", "TL", "L"]:
        if event not in df.columns:
            df[event] = 0
            
    summary_df = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std()
    })
    return summary_df

def parse_model_parameters(input_dir: str) -> Dict[str, float]:
    """Parses optimized global D, L, T rate parameters."""
    try:
        param_file = find_file(input_dir, "model_parameters.txt")
    except FileNotFoundError:
        fallbacks = ["data/interim", "results/reconciliation/alerax"]
        param_file = None
        for fb in fallbacks:
            if os.path.exists(fb):
                try:
                    param_file = find_file(fb, "model_parameters.txt")
                    break
                except FileNotFoundError:
                    continue
        if not param_file:
            raise FileNotFoundError(f"Could not find model_parameters.txt in {input_dir} or fallbacks.")
            
    with open(param_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if len(lines) < 2:
        raise ValueError(f"Parameters file {param_file} contains insufficient data.")
        
    headers = lines[0].split()
    first_data_line = lines[1].split()
    
    rates = {}
    for h, v in zip(headers[1:], first_data_line[1:]):
        rates[h] = float(v)
    return rates

class TaxonomyResolver:
    """Resolves and caches NCBI taxonomic classification (Phylum/Class)."""
    def __init__(self, cache_path: str = "ncbi_taxonomy_map.csv"):
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, dtype=str).set_index("taxid")
                self.cache = df.to_dict(orient="index")
                print(f"Loaded {len(self.cache)} cached taxonomic records from {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to load cache from {cache_path}: {e}")

    def resolve_taxids(self, taxids: List[str]) -> pd.DataFrame:
        """Resolves TaxIDs using ete3.NCBITaxa and updates the local CSV cache."""
        missing_taxids = [t for t in taxids if t not in self.cache]
        
        if missing_taxids:
            print(f"Resolving {len(missing_taxids)} TaxIDs against NCBI database...")
            if NCBITaxa is None:
                print("Warning: ete3.NCBITaxa is not installed. Defaulting missing taxids to 'Unknown'.")
                for taxid_str in missing_taxids:
                    self.cache[taxid_str] = {"phylum": "Unknown", "class": "Unknown"}
            else:
                try:
                    ncbi = NCBITaxa()
                    for taxid_str in missing_taxids:
                        try:
                            taxid = int(taxid_str)
                            lineage = ncbi.get_lineage(taxid)
                            names = ncbi.get_taxid_translator(lineage)
                            ranks = ncbi.get_rank(lineage)
                            
                            phylum = "Unknown"
                            klass = "Unknown"
                            for tid, rank in ranks.items():
                                if rank == "phylum":
                                    phylum = names[tid]
                                elif rank == "class":
                                    klass = names[tid]
                                    
                            self.cache[taxid_str] = {"phylum": phylum, "class": klass}
                        except Exception:
                            self.cache[taxid_str] = {"phylum": "Unknown", "class": "Unknown"}
                except Exception as e:
                    print(f"Warning: Failed to query ete3.NCBITaxa: {e}. Defaulting to 'Unknown'.")
                    for taxid_str in missing_taxids:
                        self.cache[taxid_str] = {"phylum": "Unknown", "class": "Unknown"}
                    
            # Save cache file
            try:
                parent_dir = os.path.dirname(self.cache_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                df_cache = pd.DataFrame.from_dict(self.cache, orient="index")
                df_cache.index.name = "taxid"
                df_cache.reset_index().to_csv(self.cache_path, index=False)
                print(f"Cached taxonomy mapping saved to {self.cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save taxonomy cache: {e}")
                
        resolved = {t: self.cache[t] for t in taxids if t in self.cache}
        df_resolved = pd.DataFrame.from_dict(resolved, orient="index")
        df_resolved.index.name = "taxid"
        return df_resolved

def build_node_to_phylum_map(input_dir: str, resolver: TaxonomyResolver) -> Dict[str, str]:
    """
    Builds a mapping from species tree leaf and internal nodes to resolved Phyla.
    Traverses the species tree bottom-up to map ancestral nodes to their dominant descendant phylum.
    """
    try:
        tree_file = find_file(input_dir, "*species_tree.nwk")
    except FileNotFoundError:
        try:
            tree_file = find_file(input_dir, "*.newick")
        except FileNotFoundError:
            fallbacks = ["data/interim", "data/sandbox_alerax", "results/reconciliation/alerax"]
            tree_file = None
            for fb in fallbacks:
                if os.path.exists(fb):
                    try:
                        tree_file = find_file(fb, "*species_tree.nwk")
                        break
                    except FileNotFoundError:
                        try:
                            tree_file = find_file(fb, "*.newick")
                            break
                        except FileNotFoundError:
                            continue
            if not tree_file:
                raise FileNotFoundError("Could not find species tree file.")
        
    tree = Phylo.read(tree_file, "newick")
    
    # 1. Resolve all leaves
    leaf_taxids = [leaf.name for leaf in tree.get_terminals() if leaf.name]
    df_tax = resolver.resolve_taxids(leaf_taxids)
    leaf_to_phylum = df_tax["phylum"].to_dict()
    
    node_to_leaves = {}
    
    def assign_leaves(clade):
        if clade.is_terminal():
            node_to_leaves[clade.name] = [clade.name]
            return [clade.name]
            
        all_leaves = []
        for child in clade.clades:
            all_leaves.extend(assign_leaves(child))
            
        if clade.name:
            node_to_leaves[clade.name] = all_leaves
        return all_leaves
        
    assign_leaves(tree.root)
    
    # Map every clade to its dominant phylum
    node_to_phylum = {}
    for node, leaves in node_to_leaves.items():
        phyla = [leaf_to_phylum.get(leaf, "Unknown") for leaf in leaves]
        phyla = [p for p in phyla if p != "Unknown"]
        if phyla:
            dominant_phylum = max(set(phyla), key=phyla.count)
            node_to_phylum[node] = dominant_phylum
        else:
            node_to_phylum[node] = "Unknown"
            
    # Include all leaf taxids explicitly
    for leaf in leaf_taxids:
        node_to_phylum[leaf] = leaf_to_phylum.get(leaf, "Unknown")
        
    return node_to_phylum

def parse_transfers_with_taxonomy(input_dir: str, resolver: TaxonomyResolver) -> pd.DataFrame:
    """Parses transfer events and maps donor and recipient nodes to their respective Phyla."""
    try:
        summary_path = os.path.join(input_dir, "reconciliations", "summaries", "*_meanTransfers.txt")
        files = glob.glob(summary_path)
        if not files:
            summary_file = find_file(input_dir, "*_meanTransfers.txt")
        else:
            summary_file = files[0]
    except FileNotFoundError:
        fallbacks = ["data/interim", "results/reconciliation/alerax"]
        summary_file = None
        for fb in fallbacks:
            if os.path.exists(fb):
                try:
                    summary_file = find_file(fb, "*_meanTransfers.txt")
                    break
                except FileNotFoundError:
                    continue
        if not summary_file:
            raise FileNotFoundError("Could not find mean transfers file.")
        
    df = pd.read_csv(summary_file, sep=r"\s+", names=["donor", "recipient", "frequency"], dtype={"donor": str, "recipient": str, "frequency": float})
    
    node_to_phylum = build_node_to_phylum_map(input_dir, resolver)
    
    df["donor_phylum"] = df["donor"].map(node_to_phylum).fillna("Unknown")
    df["recipient_phylum"] = df["recipient"].map(node_to_phylum).fillna("Unknown")
    
    return df[["donor_phylum", "recipient_phylum", "frequency"]]

def plot_event_and_rates_2panel(df_events: pd.DataFrame, rates: Dict[str, float], out_path: str):
    """Generates a 2-panel figure showing the global event DTL profile and parameter rates side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    
    # Panel 1: Horizontal stacked bar of events
    events = ["S", "SL", "D", "T", "TL"]
    event_names = {
        "S": "Speciations",
        "SL": "Speciation Losses",
        "D": "Duplications",
        "T": "Transfers",
        "TL": "Transfer Losses"
    }
    
    left = 0
    y_pos = [0]
    
    for event in events:
        mean_val = df_events.loc[event, "mean"]
        std_val = df_events.loc[event, "std"]
        ax1.barh(
            y_pos, mean_val, left=left, xerr=std_val,
            color=COLORS[event], edgecolor="black", height=0.5,
            label=f"{event_names[event]} ({mean_val:.0f})",
            error_kw=dict(ecolor="black", lw=1.5, capsize=4)
        )
        left += mean_val
        
    ax1.set_yticks([])
    ax1.set_xlabel("Reconciliation Event Totals", fontsize=11)
    ax1.set_title("DTL Global Event Profile (100 Samples)", fontsize=12, weight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    
    # Panel 2: Rates comparison
    sorted_rates = dict(sorted(rates.items(), key=lambda item: item[1], reverse=True))
    labels = list(sorted_rates.keys())
    values = list(sorted_rates.values())
    
    rate_names = {
        "D": "Duplication (D)",
        "T": "Transfer (T)",
        "L": "Loss (L)"
    }
    x_labels = [rate_names.get(h, h) for h in labels]
    bar_colors = [COLORS.get(h, "#95a5a6") for h in labels]
    
    bars = ax2.bar(x_labels, values, color=bar_colors, edgecolor="black", linewidth=1.2, width=0.4)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, weight="bold"
        )
        
    ax2.set_title("AleRax Optimized DTL Rates", fontsize=12, weight="bold")
    ax2.set_ylabel("Rate Parameter Value", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    caption = "Note: Dataset pre-filtered at CD-HIT 80% (reflects deep-time macroevolutionary rates)."
    fig.text(0.5, 0.02, caption, ha="center", fontsize=9, style="italic", color="gray")
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

def plot_taxonomic_distribution(mapping: Dict[str, str], resolver: TaxonomyResolver, out_path: str):
    """Plots a bar chart showing the family size distribution across resolved Phyla."""
    unique_taxids = list(set(mapping.values()))
    df_tax = resolver.resolve_taxids(unique_taxids)
    
    gene_phyla = []
    for gene_id, taxid in mapping.items():
        phylum = df_tax.loc[taxid, "phylum"] if taxid in df_tax.index else "Unknown"
        gene_phyla.append(phylum)
        
    df_counts = pd.Series(gene_phyla).value_counts().reset_index()
    df_counts.columns = ["Phylum", "Gene Count"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=df_counts,
        x="Gene Count",
        y="Phylum",
        hue="Phylum",
        palette="viridis",
        edgecolor="black",
        ax=ax,
        legend=False
    )
    
    ax.set_title("Taxonomic Distribution of AsnC/Lrp Transcription Factors", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Gene Count in IPR019888", fontsize=11)
    ax.set_ylabel("Phylum", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

def plot_macro_hgt_network(df_transfers: pd.DataFrame, mapping: Dict[str, str], resolver: TaxonomyResolver, out_path: str, top_n: int = 15):
    """Plots a directed graph representing macro-HGT highway pathways between phyla."""
    df_grouped = df_transfers.groupby(["donor_phylum", "recipient_phylum"])["frequency"].sum().reset_index()
    
    df_grouped = df_grouped[
        (df_grouped["donor_phylum"] != "Unknown") & 
        (df_grouped["recipient_phylum"] != "Unknown") & 
        (df_grouped["donor_phylum"] != df_grouped["recipient_phylum"])
    ]
    
    top_df = df_grouped.nlargest(top_n, "frequency")
    
    unique_taxids = list(set(mapping.values()))
    df_tax = resolver.resolve_taxids(unique_taxids)
    phyla_counts = {}
    for gene_id, taxid in mapping.items():
        phylum = df_tax.loc[taxid, "phylum"] if taxid in df_tax.index else "Unknown"
        phyla_counts[phylum] = phyla_counts.get(phylum, 0) + 1
        
    G = nx.DiGraph()
    for _, row in top_df.iterrows():
        G.add_edge(row["donor_phylum"], row["recipient_phylum"], weight=row["frequency"])
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    
    pos = nx.circular_layout(G)
    
    node_sizes = [300 + 4.0 * phyla_counts.get(node, 100) for node in G.nodes()]
    
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1.0
    edge_widths = [1 + 6 * (w / max_weight) for w in weights]
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="skyblue",
        edgecolors="navy",
        linewidths=1.5,
        alpha=0.9,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color="steelblue",
        alpha=0.6,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15",
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        font_size=9,
        font_weight="bold",
        font_family="sans-serif",
        ax=ax
    )
    
    ax.set_title(f"Macro-HGT Highways between Phyla (Top {top_n} Inter-Phylum Transfers)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize macroevolutionary taxonomic profiles of AleRax DTL outputs."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to AleRax output directory containing the species tree and DTL reconciliations."
    )
    parser.add_argument(
        "--taxonomy_map", default="ncbi_taxonomy_map.csv",
        help="Path to look for or save the cached CSV taxonomy mapping file."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Path to save generated plots."
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "svg", "pdf"],
        help="Output image format (default: png)."
    )
    parser.add_argument(
        "--top_hgt", type=int, default=15,
        help="Number of top phylum transfer highways to plot (default: 15)."
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Parsing species leaves and gene mapping under: {args.input_dir}")
    
    try:
        species_taxids = parse_species_tree(args.input_dir)
        gene_mapping = parse_treerecs_mapping(args.input_dir)
        rates = parse_model_parameters(args.input_dir)
        df_events = parse_event_counts_with_errors(args.input_dir)
    except Exception as e:
        print(f"Error parsing initial files: {e}", file=sys.stderr)
        sys.exit(1)
        
    resolver = TaxonomyResolver(cache_path=args.taxonomy_map)
    resolver.resolve_taxids(species_taxids)
    
    print("Generating macroevolutionary visualizations...")
    
    plot_event_and_rates_2panel(
        df_events, rates,
        os.path.join(args.output_dir, f"global_event_profile_and_rates.{args.format}")
    )
    
    plot_taxonomic_distribution(
        gene_mapping, resolver,
        os.path.join(args.output_dir, f"taxonomic_distribution.{args.format}")
    )
    
    try:
        df_transfers = parse_transfers_with_taxonomy(args.input_dir, resolver)
        plot_macro_hgt_network(
            df_transfers, gene_mapping, resolver,
            os.path.join(args.output_dir, f"macro_hgt_highway_network.{args.format}"),
            top_n=args.top_hgt
        )
    except Exception as e:
        print(f"Warning: Failed to construct macro HGT network: {e}", file=sys.stderr)
        
    print(f"Success! Macro visualizations generated at {args.output_dir}")

if __name__ == "__main__":
    main()
