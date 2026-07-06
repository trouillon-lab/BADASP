#!/usr/bin/env python
"""Analyze and plot number of switches vs minimum leaf size in clades compared.

Outputs:
- results/badasp_scoring/clade_size_adjusted/min_clade_5/occupancy_80/switches_vs_clade_size.svg
- results/badasp_scoring/clade_size_adjusted/min_clade_5/occupancy_80/switches_vs_clade_size.png
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

    # Calculate min clade size for each comparison
    df_filtered["min_clade_size"] = df_filtered[["clade_size_left", "clade_size_right"]].min(axis=1)

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

    # Calculate thresholds for 99.9th percentile
    thresholds = {}
    # Event-specific thresholds
    for event in ["Duplication", "Speciation", "Transfer"]:
        event_df = melted_df[melted_df["event_type"] == event]
        for bin_interval in bin_categories:
            bin_df = event_df[event_df["clade_bin"] == bin_interval]
            scores = bin_df["score"].dropna()
            thresholds[(event, bin_interval)] = float(np.percentile(scores, percentile)) if len(scores) > 0 else np.nan

    # Overall (event-agnostic) thresholds
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

    # Identify switches for event-specific
    is_switch_event_specific = []
    for _, row in df_filtered.iterrows():
        event = row["event_type"]
        bin_l = row["bin_left"]
        bin_r = row["bin_right"]
        thresh_l = thresholds.get((event, bin_l), np.inf)
        thresh_r = thresholds.get((event, bin_r), np.inf)
        score_l = row["badasp_score_left"]
        score_r = row["badasp_score_right"]
        switch_l = (not np.isnan(score_l)) and (score_l >= thresh_l)
        switch_r = (not np.isnan(score_r)) and (score_r >= thresh_r)
        is_switch_event_specific.append(switch_l or switch_r)
    df_filtered["is_switch_event_specific"] = is_switch_event_specific

    # Calculate min clade size for comparisons
    # Map min clade size of comparison to bins
    df_filtered["min_clade_bin"] = df_filtered["min_clade_size"].apply(_map_to_bin)

    print("Generating switches vs clade size plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PANEL A: Switch counts by clade size decile bins (Event-agnostic / Overall) ---
    ax = axes[0]
    bin_switches = []
    for cat in bin_categories:
        bin_df = df_filtered[df_filtered["min_clade_bin"] == cat]
        sw_total = bin_df["is_switch_overall"].sum()
        sw_dup = bin_df[bin_df["event_type"] == "Duplication"]["is_switch_overall"].sum()
        sw_spec = bin_df[bin_df["event_type"] == "Speciation"]["is_switch_overall"].sum()
        sw_trans = bin_df[bin_df["event_type"] == "Transfer"]["is_switch_overall"].sum()
        
        bin_switches.append({
            "Bin": f"{int(cat.left)}-{int(cat.right)}" if not np.isinf(cat.right) else f">{int(cat.left)}",
            "Duplication": sw_dup,
            "Speciation": sw_spec,
            "Transfer": sw_trans,
            "Overall": sw_total
        })
    df_bin_sw = pd.DataFrame(bin_switches)

    # Melt for plotting
    df_bin_melt = df_bin_sw.melt(id_vars="Bin", value_vars=["Duplication", "Speciation", "Transfer"], 
                                 var_name="Event Type", value_name="Switches")
    
    sns.barplot(
        data=df_bin_melt,
        x="Bin",
        y="Switches",
        hue="Event Type",
        palette=[EVENT_COLORS["Duplication"], EVENT_COLORS["Speciation"], EVENT_COLORS["Transfer"]],
        ax=ax
    )
    ax.set_title("A. 99.9th% Overall Switches Binned by Min Clade Size", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Minimum Clade Size of Sister Pairs (Decile Bins)", fontsize=11)
    ax.set_ylabel("Number of Switches", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Event Type", frameon=True)

    # Add text labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="semibold"
            )

    # --- PANEL B: Cumulative switch count remaining as minimum leaf size filter increases ---
    ax = axes[1]
    
    # We will test M thresholds from 5 to 500
    m_thresholds = np.unique(np.logspace(np.log10(5), np.log10(1000), num=100).astype(int))
    
    overall_surviving = []
    dup_surviving = []
    spec_surviving = []
    trans_surviving = []
    
    for M in m_thresholds:
        surv_df = df_filtered[df_filtered["min_clade_size"] >= M]
        overall_surviving.append(surv_df["is_switch_overall"].sum())
        dup_surviving.append(surv_df[surv_df["event_type"] == "Duplication"]["is_switch_overall"].sum())
        spec_surviving.append(surv_df[surv_df["event_type"] == "Speciation"]["is_switch_overall"].sum())
        trans_surviving.append(surv_df[surv_df["event_type"] == "Transfer"]["is_switch_overall"].sum())
        
    ax.plot(m_thresholds, overall_surviving, color=EVENT_COLORS["Overall"], label="Overall (Event-Agnostic)", linewidth=2.5, linestyle="--")
    ax.plot(m_thresholds, dup_surviving, color=EVENT_COLORS["Duplication"], label="Duplication", linewidth=2.0)
    ax.plot(m_thresholds, spec_surviving, color=EVENT_COLORS["Speciation"], label="Speciation", linewidth=2.0)
    ax.plot(m_thresholds, trans_surviving, color=EVENT_COLORS["Transfer"], label="Transfer", linewidth=2.0)
    
    ax.set_xscale("log")
    ax.set_title("B. Cumulative Switch Survival Curve vs. Min Clade Size", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Minimum Clade Size Filter (log scale)", fontsize=11)
    ax.set_ylabel("Number of Surviving Switches", fontsize=11)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=True)

    # Adjust layout and save
    plt.tight_layout()
    out_svg = out_dir / "switches_vs_clade_size.svg"
    out_png = out_dir / "switches_vs_clade_size.png"
    
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved plots to:\n  SVG: {out_svg}\n  PNG: {out_png}")

if __name__ == "__main__":
    main()
