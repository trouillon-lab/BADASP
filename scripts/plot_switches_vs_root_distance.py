#!/usr/bin/env python
"""Analyze and plot number of switches vs distance from root for each node.

Outputs:
- results/badasp_scoring/clade_size_adjusted/min_clade_5/occupancy_80/switches_vs_root_distance.svg
- results/badasp_scoring/clade_size_adjusted/min_clade_5/occupancy_80/switches_vs_root_distance.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from Bio import AlignIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
EVENT_COLORS = {
    "Duplication": "#c0392b",
    "Speciation": "#2980b9",
    "Transfer": "#27ae60",
    "Overall": "#2c3e50"
}

def main():
    scores_path = Path("results/badasp_scoring/raw_node_scores.csv")
    alignment_path = Path("data/interim/IPR019888_trimmed.aln")
    min_clade_size = 5
    min_occupancy = 0.8
    percentile = 99.9

    out_dir = Path("results/badasp_scoring/clade_size_adjusted/min_clade_5/occupancy_80")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists():
        print(f"Error: Raw scores file not found at {scores_path}")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_csv(scores_path)
    df = df[(df["clade_size_left"] >= min_clade_size) & (df["clade_size_right"] >= min_clade_size)]

    # Load alignment to calculate occupancies
    alignment = AlignIO.read(alignment_path, "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {"-", "."})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)

    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= min_occupancy].copy()

    # Melt left and right
    df_left = df_filtered[["node_name", "event_type", "position", "clade_size_left", "badasp_score_left"]].rename(
        columns={"badasp_score_left": "score", "clade_size_left": "clade_size"}
    )
    df_right = df_filtered[["node_name", "event_type", "position", "clade_size_right", "badasp_score_right"]].rename(
        columns={"badasp_score_right": "score", "clade_size_right": "clade_size"}
    )
    melted_df = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["score", "clade_size"])

    # Bin using quantile (deciles)
    melted_df["clade_bin"] = pd.qcut(melted_df["clade_size"], q=10, duplicates="drop")
    bin_categories = sorted(melted_df["clade_bin"].cat.categories)

    # Map bins to df_filtered
    def _map_to_bin(val):
        for interval in bin_categories:
            if val in interval:
                return interval
        return bin_categories[-1] if val > bin_categories[-1].right else bin_categories[0]

    df_filtered["bin_left"] = df_filtered["clade_size_left"].apply(_map_to_bin)
    df_filtered["bin_right"] = df_filtered["clade_size_right"].apply(_map_to_bin)

    # Calculate thresholds for 99.9th percentile (overall event-agnostic)
    thresholds = {}
    for bin_interval in bin_categories:
        bin_df = melted_df[melted_df["clade_bin"] == bin_interval]
        scores = bin_df["score"].dropna()
        thresholds[("overall", bin_interval)] = float(np.percentile(scores, percentile)) if len(scores) > 0 else np.nan

    # Identify switches for overall
    is_switch_list = []
    for _, row in df_filtered.iterrows():
        bin_l = row["bin_left"]
        bin_r = row["bin_right"]
        thresh_l = thresholds.get(("overall", bin_l), np.inf)
        thresh_r = thresholds.get(("overall", bin_r), np.inf)
        score_l = row["badasp_score_left"]
        score_r = row["badasp_score_right"]
        switch_l = (not np.isnan(score_l)) and (score_l >= thresh_l)
        switch_r = (not np.isnan(score_r)) and (score_r >= thresh_r)
        is_switch_list.append(switch_l or switch_r)
    df_filtered["is_switch_overall"] = is_switch_list

    # Group by node_name to aggregate switches on each node
    # Since distance_from_root is identical for a given node_name, we take first
    node_groups = df_filtered.groupby("node_name")
    node_data = []
    for node_name, group in node_groups:
        total_sw = group["is_switch_overall"].sum()
        event_type = group["event_type"].iloc[0]
        dist = group["distance_from_root"].iloc[0]
        
        node_data.append({
            "node_name": node_name,
            "event_type": event_type,
            "distance_from_root": dist,
            "switches": total_sw
        })
    df_nodes = pd.DataFrame(node_data)

    print("Generating switches vs distance from root plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PANEL A: Scatter Plot of Switches vs Distance from Root ---
    ax = axes[0]
    
    # Separate nodes with zero switches vs nodes with > 0 switches
    df_active = df_nodes[df_nodes["switches"] > 0]
    df_inactive = df_nodes[df_nodes["switches"] == 0]
    
    # Plot inactive nodes with small transparent grey dots to show background density
    ax.scatter(
        df_inactive["distance_from_root"],
        df_inactive["switches"],
        color="#bdc3c7",
        alpha=0.2,
        s=15,
        label="No switches (background)"
    )
    
    # Plot active nodes with color coded by event type
    for event, color in [("Duplication", EVENT_COLORS["Duplication"]), 
                          ("Speciation", EVENT_COLORS["Speciation"]), 
                          ("Transfer", EVENT_COLORS["Transfer"])]:
        ev_df = df_active[df_active["event_type"] == event]
        if not ev_df.empty:
            ax.scatter(
                ev_df["distance_from_root"],
                ev_df["switches"],
                color=color,
                alpha=0.85,
                s=40,
                edgecolor="black",
                linewidths=0.5,
                label=f"{event} ({len(ev_df)} nodes)"
            )
            
    # Add text labels for the top 5 nodes with most switches
    top_nodes = df_active.sort_values(by="switches", ascending=False).head(5)
    for _, row in top_nodes.iterrows():
        ax.text(
            row["distance_from_root"],
            row["switches"] + 0.3,
            f"{row['node_name']}\n({int(row['switches'])})",
            ha="center", va="bottom",
            fontsize=8, color="#2c3e50", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec="#2c3e50", lw=0.5, alpha=0.8)
        )

    ax.set_title("A. Node Switch Counts vs. Distance from Root", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Phylogenetic Distance from Root (subs/site)", fontsize=11)
    ax.set_ylabel("Number of Switches on Node", fontsize=11)
    ax.legend(frameon=True, loc="upper right")

    # --- PANEL B: Kernel Density Estimate of Switch Depth Distribution ---
    ax = axes[1]
    
    # We want the density of individual switch events along the root-distance axis.
    # To do this, we replicate the distance_from_root for each switch.
    switch_depths = []
    for _, row in df_active.iterrows():
        switch_depths.extend([row["distance_from_root"]] * int(row["switches"]))
        
    df_sw_depths = pd.DataFrame({"distance_from_root": switch_depths})
    
    # Replicate event-specific switch depths
    for event in ["Duplication", "Speciation", "Transfer"]:
        ev_df = df_active[df_active["event_type"] == event]
        ev_depths = []
        for _, row in ev_df.iterrows():
            ev_depths.extend([row["distance_from_root"]] * int(row["switches"]))
        
        if ev_depths:
            sns.kdeplot(
                data=ev_depths,
                color=EVENT_COLORS[event],
                fill=True,
                alpha=0.2,
                linewidth=1.5,
                label=f"{event} switches (N={len(ev_depths)})",
                ax=ax
            )
            
    # Overall KDE
    if switch_depths:
        sns.kdeplot(
            data=switch_depths,
            color=EVENT_COLORS["Overall"],
            linewidth=2.0,
            linestyle="--",
            label=f"Overall switches (N={len(switch_depths)})",
            ax=ax
        )
        
    ax.set_title("B. Density Distribution of Switches along Root Distance", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Phylogenetic Distance from Root (subs/site)", fontsize=11)
    ax.set_ylabel("Density of Switch Events", fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    out_svg = out_dir / "switches_vs_root_distance.svg"
    out_png = out_dir / "switches_vs_root_distance.png"
    
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved plots to:\n  SVG: {out_svg}\n  PNG: {out_png}")

if __name__ == "__main__":
    main()
