"""Analyze and plot BADASP score distributions across 10 sequence bins.

Compares unfiltered score distributions against occupancy-filtered score distributions
to visualize the score inflation in the N- and C-terminal tails.
Saves plots and stats to results/badasp_scoring/threshold_comparison/occupancy_XX/.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from Bio import AlignIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_msa_occupancies(alignment_path: Path) -> dict:
    """Calculate the occupancy fraction for each column in the MSA (1-indexed)."""
    print(f"Loading alignment from {alignment_path}...")
    alignment = AlignIO.read(alignment_path, "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occ = 1.0 - (gaps / num_seqs)
        occupancies[col + 1] = occ  # 1-indexed position
        
    return occupancies


def main() -> None:
    parser = argparse.ArgumentParser(description="BADASP Score Distribution across Sequence Bins")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8, help="Occupancy threshold used for filtering (0.0 to 1.0)")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for results")
    args = parser.parse_args()

    occ_pct = int(args.min_occupancy * 100)
    
    # 1. Load Data
    if not args.scores.exists():
        print(f"Error: Raw scores file not found at {args.scores}")
        sys.exit(1)
        
    print("Loading raw scores...")
    df = pd.read_csv(args.scores)
    df["max_score"] = df[["badasp_score_left", "badasp_score_right"]].max(axis=1)
    
    # Calculate occupancies
    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    
    # Setup Output Directory
    if args.outdir is not None:
        out_dir = args.outdir
    else:
        out_dir = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define 10 Sequence Bins
    bin_edges = np.linspace(1, 170, 11, dtype=int)
    bin_labels = []
    
    # Add bin column
    df["sequence_bin"] = None
    for i in range(10):
        start = bin_edges[i]
        end = bin_edges[i+1] - 1
        label = f"Bin {i+1}\n({start}-{end})"
        bin_labels.append(label)
        mask = (df["position"] >= start) & (df["position"] <= end)
        df.loc[mask, "sequence_bin"] = label
        
    # Make bin column a categorical to preserve order in plots
    df["sequence_bin"] = pd.Categorical(df["sequence_bin"], categories=bin_labels, ordered=True)

    # 3. Compile Statistics per Bin (Unfiltered vs Filtered)
    stats_rows = []
    
    # Unfiltered stats
    for label in bin_labels:
        b_df = df[df["sequence_bin"] == label]
        scores = b_df["max_score"].dropna()
        stats_rows.append({
            "bin": label.replace("\n", " "),
            "filter_status": "Unfiltered",
            "total_comparisons": len(b_df),
            "mean_score": scores.mean(),
            "median_score": scores.median(),
            "std_score": scores.std(),
            "p95_score": np.percentile(scores, 95) if len(scores) > 0 else np.nan,
            "p97_score": np.percentile(scores, 97) if len(scores) > 0 else np.nan,
            "p99_score": np.percentile(scores, 99) if len(scores) > 0 else np.nan,
        })
        
    # Filtered stats
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()
    for label in bin_labels:
        b_df = df_filtered[df_filtered["sequence_bin"] == label]
        scores = b_df["max_score"].dropna()
        stats_rows.append({
            "bin": label.replace("\n", " "),
            "filter_status": f"Filtered (>= {args.min_occupancy:.0%})",
            "total_comparisons": len(b_df),
            "mean_score": scores.mean() if len(scores) > 0 else np.nan,
            "median_score": scores.median() if len(scores) > 0 else np.nan,
            "std_score": scores.std() if len(scores) > 0 else np.nan,
            "p95_score": np.percentile(scores, 95) if len(scores) > 0 else np.nan,
            "p97_score": np.percentile(scores, 97) if len(scores) > 0 else np.nan,
            "p99_score": np.percentile(scores, 99) if len(scores) > 0 else np.nan,
        })
        
    df_stats = pd.DataFrame(stats_rows)
    stats_csv = out_dir / "sequence_bins_stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"Saved sequence bin statistics to {stats_csv}")

    # 4. Generate 2-Panel Box Plot
    print("Generating box plots...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharey=True)
    
    # Panel A: Unfiltered Box Plot
    sns.boxplot(
        data=df,
        x="sequence_bin",
        y="max_score",
        palette="Blues",
        ax=axes[0],
        fliersize=0.5,
        width=0.6
    )
    medians_unfiltered = df.groupby("sequence_bin")["max_score"].median()
    axes[0].plot(range(10), medians_unfiltered.values, color="#e74c3c", linewidth=2.0, linestyle="--", marker="o", label="Median Score")
    
    axes[0].set_title("A. Score Distribution across 10 Sequence Bins (Unfiltered - Tail Inflation Visible)", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("", fontsize=12)
    axes[0].set_ylabel("BADASP Score", fontsize=12)
    axes[0].legend(loc="upper right")
    
    # Panel B: Filtered Box Plot
    sns.boxplot(
        data=df_filtered,
        x="sequence_bin",
        y="max_score",
        palette="Greens",
        ax=axes[1],
        fliersize=0.5,
        width=0.6
    )
    if len(df_filtered) > 0:
        medians_filtered = df_filtered.groupby("sequence_bin")["max_score"].median()
        x_indices = [i for i, label in enumerate(bin_labels) if label in medians_filtered.index]
        axes[1].plot(x_indices, medians_filtered.values, color="#e74c3c", linewidth=2.0, linestyle="--", marker="o", label="Median Score")
        
    for i, label in enumerate(bin_labels):
        bin_data = df_filtered[df_filtered["sequence_bin"] == label]
        if len(bin_data) == 0:
            axes[1].text(
                i, 0.0, 
                "FILTERED OUT\n(100% Low Occupancy)", 
                ha="center", va="center", 
                fontsize=10, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )
            
    axes[1].set_title(f"B. Score Distribution across 10 Sequence Bins (Filtered - Occupancy >= {args.min_occupancy:.0%})", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xlabel("Sequence Coordinate Bins", fontsize=12)
    axes[1].set_ylabel("BADASP Score", fontsize=12)
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    
    plot_svg = out_dir / "sequence_bins_score_distribution.svg"
    plot_png = out_dir / "sequence_bins_score_distribution.png"
    
    fig.savefig(str(plot_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 5. Generate 2-Panel Violin Plot
    print("Generating violin plots...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharey=True)
    
    # Panel A: Unfiltered Violin Plot
    sns.violinplot(
        data=df,
        x="sequence_bin",
        y="max_score",
        palette="Blues",
        ax=axes[0],
        inner="quartile",
        linewidth=1.2
    )
    axes[0].plot(range(10), medians_unfiltered.values, color="#e74c3c", linewidth=2.0, linestyle="--", marker="o", label="Median Score")
    
    axes[0].set_title("A. Score Density across 10 Sequence Bins (Unfiltered - Tail Inflation Visible)", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("", fontsize=12)
    axes[0].set_ylabel("BADASP Score", fontsize=12)
    axes[0].legend(loc="upper right")
    
    # Panel B: Filtered Violin Plot
    sns.violinplot(
        data=df_filtered,
        x="sequence_bin",
        y="max_score",
        palette="Greens",
        ax=axes[1],
        inner="quartile",
        linewidth=1.2
    )
    if len(df_filtered) > 0:
        axes[1].plot(x_indices, medians_filtered.values, color="#e74c3c", linewidth=2.0, linestyle="--", marker="o", label="Median Score")
        
    for i, label in enumerate(bin_labels):
        bin_data = df_filtered[df_filtered["sequence_bin"] == label]
        if len(bin_data) == 0:
            axes[1].text(
                i, 0.0, 
                "FILTERED OUT\n(100% Low Occupancy)", 
                ha="center", va="center", 
                fontsize=10, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )
            
    axes[1].set_title(f"B. Score Density across 10 Sequence Bins (Filtered - Occupancy >= {args.min_occupancy:.0%})", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xlabel("Sequence Coordinate Bins", fontsize=12)
    axes[1].set_ylabel("BADASP Score", fontsize=12)
    axes[1].legend(loc="upper right")
    
    plt.tight_layout()
    
    violin_svg = out_dir / "sequence_bins_violin_distribution.svg"
    violin_png = out_dir / "sequence_bins_violin_distribution.png"
    
    fig.savefig(str(violin_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(violin_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Generated plots successfully:")
    print(f"  Box Plots: {plot_svg} / {plot_png}")
    print(f"  Violin Plots: {violin_svg} / {violin_png}")
    print("\nSequence bins score distribution analysis completed successfully.")


if __name__ == "__main__":
    main()
