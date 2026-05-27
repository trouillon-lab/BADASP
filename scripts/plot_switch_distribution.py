#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    scores_root = Path("results/badasp_scoring")
    output_dir = Path("results/evolutionary_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_files = sorted(scores_root.glob("layer_*/badasp_scores_combined.csv"))
    
    all_data = []
    for f in layer_files:
        try:
            layer_idx = int(f.parent.name.split("_")[-1])
        except (ValueError, IndexError):
            continue

        if layer_idx == 1:
            continue

        df = pd.read_csv(f)
        if df.empty or "position" not in df.columns or "switch_count" not in df.columns:
            continue

        df_filtered = df[["position", "switch_count"]].copy()
        df_filtered["layer"] = layer_idx
        all_data.append(df_filtered)

    if not all_data:
        print("No switch count data found!")
        sys.exit(1)

    df_all = pd.concat(all_data, ignore_index=True)
    
    # Filter for active positions (switch_count > 0) to highlight the signal distribution
    df_active = df_all[df_all["switch_count"] > 0].copy()

    # Set premium Seaborn style
    sns.set_theme(style="ticks")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Overall distribution histogram of active switch counts
    sns.histplot(
        data=df_active,
        x="switch_count",
        kde=True,
        ax=ax1,
        color="#2B5C8F",
        edgecolor="w",
        linewidth=0.8,
        bins=25
    )
    ax1.set_title("Overall Distribution of Active Switch Counts (Counts > 0)", pad=15, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Raw Switch Count per Residue", fontweight="bold")
    ax1.set_ylabel("Frequency", fontweight="bold")
    ax1.grid(color="#EEEEEE", linestyle="--", linewidth=0.5)

    # Panel B: Chronological distribution of switch counts across Layers
    sns.boxplot(
        data=df_active,
        x="layer",
        y="switch_count",
        ax=ax2,
        palette="crest",
        fliersize=2,
        linewidth=1.2
    )
    ax2.set_title("Active Switch Count Distribution Across Clade Layers (2-20)", pad=15, fontsize=12, fontweight="bold")
    ax2.set_xlabel("Evolutionary Clade Layer Index", fontweight="bold")
    ax2.set_ylabel("Raw Switch Count", fontweight="bold")
    ax2.grid(color="#EEEEEE", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    output_path = output_dir / "switch_counts_distribution.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"Saved distribution plot to: {output_path}")

    # Generate summary stats
    print("\nSummary Statistics of Active Switch Counts (overall):")
    print(df_active["switch_count"].describe())

    print("\nTotal active switches per layer:")
    layer_sums = df_all.groupby("layer")["switch_count"].sum()
    print(layer_sums)

if __name__ == "__main__":
    main()
