#!/usr/bin/env python3
"""
visualize_alerax.py

A standalone, highly modular Python script to visualize the output results of an 
AleRax DTL (Duplication, Transfer, Loss) reconciliation run. 

Generates 4 publication-quality visualizations:
1. Global Event Profile (Donut chart of mean event counts)
2. Reconciliation Variance (Violin/boxplot across sampled reconciliations)
3. HGT Highway Network (Directed network graph of top transfers)
4. DTL Parameter Rate Comparison (Bar chart of optimized global rates)
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Define a visually pleasing, publication-quality color palette
COLORS = {
    "S": "#3498db",   # Speciations - Sleek blue
    "SL": "#85c1e9",  # Speciation Losses - Light blue
    "D": "#e74c3c",   # Duplications - Coral red
    "DL": "#f1948a",  # Duplication Losses - Light coral
    "T": "#2ecc71",   # Transfers - Emerald green
    "TL": "#82e0aa",  # Transfer Losses - Light emerald green
    "L": "#95a5a6"    # Losses - Gray
}

def parse_event_counts(input_dir: str) -> pd.DataFrame:
    """
    Parses tree-wide DTL event counts from individual reconciliation sample files.
    AleRax writes these under reconciliations/all/*_eventCounts_*.txt
    
    Args:
        input_dir: Path to the AleRax output directory.
        
    Returns:
        A Pandas DataFrame containing event counts for each sample/iteration.
    """
    search_path = os.path.join(input_dir, "reconciliations", "all", "*_eventCounts_*.txt")
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No eventCounts files found in {search_path}")
        
    records = []
    for f in sorted(files):
        sample_id = int(os.path.basename(f).split("_")[-1].replace(".txt", ""))
        record = {"sample_id": sample_id}
        with open(f, "r") as file:
            for line in file:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                record[key] = int(val)
        records.append(record)
        
    df = pd.DataFrame(records).set_index("sample_id").sort_index()
    # Ensure all expected event keys are present (fill with 0 if absent)
    for event in ["S", "SL", "D", "DL", "T", "TL", "L", "Leaf"]:
        if event not in df.columns:
            df[event] = 0
    return df[["S", "SL", "D", "DL", "T", "TL", "L", "Leaf"]]

def parse_model_parameters(input_dir: str) -> Dict[str, float]:
    """
    Parses optimized global D, L, T rate parameters.
    AleRax writes these under model_parameters/model_parameters.txt
    
    Args:
        input_dir: Path to the AleRax output directory.
        
    Returns:
        A dictionary mapping rate types (D, L, T) to their optimized float values.
    """
    param_file = os.path.join(input_dir, "model_parameters", "model_parameters.txt")
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Model parameters file not found at {param_file}")
        
    with open(param_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if len(lines) < 2:
        raise ValueError(f"Parameters file {param_file} contains insufficient data.")
        
    headers = lines[0].split()
    first_data_line = lines[1].split()
    
    # Map headers to values
    rates = {}
    for h, v in zip(headers[1:], first_data_line[1:]):  # Skip node index
        rates[h] = float(v)
        
    return rates

def parse_transfers(input_dir: str) -> pd.DataFrame:
    """
    Parses mean HGT transfer frequencies from the summary file.
    AleRax writes this to reconciliations/summaries/*_meanTransfers.txt
    
    Args:
        input_dir: Path to the AleRax output directory.
        
    Returns:
        A DataFrame representing HGT transfer pathways (donor, recipient, frequency).
    """
    summary_path = os.path.join(input_dir, "reconciliations", "summaries", "*_meanTransfers.txt")
    files = glob.glob(summary_path)
    if not files:
        raise FileNotFoundError(f"No meanTransfers files found in {summary_path}")
        
    transfers_file = files[0]
    df = pd.read_csv(
        transfers_file, 
        sep=r"\s+", 
        names=["donor", "recipient", "frequency"],
        dtype={"donor": str, "recipient": str, "frequency": float}
    )
    return df

def plot_global_profile(df: pd.DataFrame, out_path: str):
    """Generates a donut chart displaying mean event proportions."""
    # Exclude Leaf and unused categories (L, DL are usually 0 in DTL models)
    core_events = ["S", "SL", "D", "T", "TL"]
    means = df[core_events].mean()
    means = means[means > 0]  # Only plot categories with non-zero counts
    
    labels = [f"{event}\n({means[event]:.0f})" for event in means.index]
    colors = [COLORS[event] for event in means.index]
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(
        means,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops=dict(color="black", fontsize=10, weight="bold"),
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    
    # Stylize percentage labels
    for autotext in autotexts:
        autotext.set_color("dimgray")
        autotext.set_fontsize(9)
        
    ax.set_title("AleRax Reconciliation Global Event Profile", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_variance(df: pd.DataFrame, out_path: str):
    """Generates a violin plot with overlay jitter points of DTL event distributions."""
    core_events = ["S", "SL", "D", "T", "TL"]
    melted_df = df[core_events].melt(var_name="Event", value_name="Count")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Base violin plot
    sns.violinplot(
        data=melted_df,
        x="Event",
        y="Count",
        hue="Event",
        palette=COLORS,
        inner=None,
        ax=ax,
        density_norm="width",
        alpha=0.6,
        legend=False
    )
    
    # Overlay boxplot for summary stats
    sns.boxplot(
        data=melted_df,
        x="Event",
        y="Count",
        width=0.15,
        color="dimgray",
        boxprops=dict(facecolor="white", edgecolor="black", alpha=0.9),
        ax=ax,
        showfliers=False
    )
    
    # Jittered strip plot for sample representation
    sns.stripplot(
        data=melted_df,
        x="Event",
        y="Count",
        hue="Event",
        palette=COLORS,
        size=4,
        jitter=0.2,
        alpha=0.4,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        legend=False
    )
    
    ax.set_title("Reconciliation Event Variance Across Sampled Reconciliations", fontsize=13, weight="bold", pad=15)
    ax.set_xlabel("Reconciliation Event Type", fontsize=11)
    ax.set_ylabel("Count per Sample", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_hgt_network(df: pd.DataFrame, out_path: str, top_n: int = 25):
    """Generates a directed network graph illustrating top transfer pathways."""
    # Filter top transfers
    top_df = df.nlargest(top_n, "frequency")
    
    # Build NetworkX graph
    G = nx.DiGraph()
    for _, row in top_df.iterrows():
        G.add_edge(row["donor"], row["recipient"], weight=row["frequency"])
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Clean up axes
    ax.axis("off")
    
    # Position nodes using circular or spring layout
    pos = nx.spring_layout(G, k=1.2, seed=42)
    
    # Degree for sizing nodes
    in_degrees = dict(G.in_degree(weight="weight"))
    out_degrees = dict(G.out_degree(weight="weight"))
    node_sizes = [500 + 400 * (in_degrees.get(n, 0) + out_degrees.get(n, 0)) for n in G.nodes()]
    
    # Edge widths/colors based on transfer frequency
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1.0
    edge_widths = [1 + 5 * (w / max_weight) for w in weights]
    
    # Draw graph elements
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="#2ecc71",
        edgecolors="darkgreen",
        linewidths=1.5,
        alpha=0.9,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color="mediumseagreen",
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
    
    ax.set_title(f"HGT Highway Network (Top {top_n} Mean Transfers)", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_rates(rates: Dict[str, float], out_path: str):
    """Generates a publication-quality bar chart comparing optimized DTL rates."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sorted_rates = dict(sorted(rates.items(), key=lambda item: item[1], reverse=True))
    labels = list(sorted_rates.keys())
    values = list(sorted_rates.values())
    
    # Clean map rates to names
    rate_names = {
        "D": "Duplication (D)",
        "T": "Transfer (T)",
        "L": "Loss (L)"
    }
    x_labels = [rate_names.get(h, h) for h in labels]
    bar_colors = [COLORS.get(h, "#95a5a6") for h in labels]
    
    bars = ax.bar(x_labels, values, color=bar_colors, edgecolor="black", linewidth=1.2, width=0.5)
    
    # Value annotations on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, weight="bold"
        )
        
    ax.set_title("AleRax Optimized DTL Rate Parameters", fontsize=13, weight="bold", pad=15)
    ax.set_ylabel("Rate Parameter Value", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize DTL outputs and model parameters from an AleRax run."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to AleRax output directory containing model_parameters and reconciliations."
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
        "--top_hgt", type=int, default=25,
        help="Number of top transfers to include in network graph (default: 25)."
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Parsing AleRax results from: {args.input_dir}")
    
    try:
        # Load datasets
        df_events = parse_event_counts(args.input_dir)
        rates = parse_model_parameters(args.input_dir)
        df_transfers = parse_transfers(args.input_dir)
    except Exception as e:
        print(f"Error parsing input data: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Generating visualizations...")
    
    # 1. Global event proportions
    plot_global_profile(
        df_events,
        os.path.join(args.output_dir, f"global_event_profile.{args.format}")
    )
    
    # 2. Reconciliation variance
    plot_variance(
        df_events,
        os.path.join(args.output_dir, f"reconciliation_variance.{args.format}")
    )
    
    # 3. Directed Highway Network Graph
    plot_hgt_network(
        df_transfers,
        os.path.join(args.output_dir, f"hgt_highway_network.{args.format}"),
        top_n=args.top_hgt
    )
    
    # 4. Parameter Rates Comparison
    plot_rates(
        rates,
        os.path.join(args.output_dir, f"dtl_parameter_rates.{args.format}")
    )
    
    print(f"Success! Visualizations written to {args.output_dir}")

if __name__ == "__main__":
    main()
