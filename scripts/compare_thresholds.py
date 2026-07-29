"""Compare BADASP evolutionary switch thresholds with MSA column occupancy filtering.

Analyzes the number of switches, their distribution along the protein sequence, 
and their distribution in evolutionary time (distance from root).
Filters out positions with occupancy in the MSA below a specified threshold.
Saves tables and high-resolution plots to results/badasp_scoring/threshold_comparison/occupancy_XX/.
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
    parser = argparse.ArgumentParser(description="BADASP Threshold Comparison with MSA Occupancy Filtering")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8, help="Filter out positions with occupancy below this value (0.0 to 1.0)")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for results")
    args = parser.parse_args()

    # 1. Load Raw Scores
    if not args.scores.exists():
        print(f"Error: Raw scores file not found at {args.scores}")
        sys.exit(1)
        
    print("Loading raw scores...")
    df = pd.read_csv(args.scores)
    
    # Calculate the maximum score per node-split at each position
    df["max_score"] = df[["badasp_score_left", "badasp_score_right"]].max(axis=1)
    
    # 2. Calculate Occupancy and Filter Positions
    occupancies = calculate_msa_occupancies(args.alignment)
    
    # Map occupancy to the dataframe
    df["occupancy"] = df["position"].map(occupancies)
    
    # Identify filtered positions
    filtered_positions = [pos for pos, occ in occupancies.items() if occ < args.min_occupancy]
    print(f"\nMSA Occupancy Filtering (Cutoff: {args.min_occupancy:.2%}):")
    print(f"  Total positions in MSA: {len(occupancies)}")
    print(f"  Filtered out {len(filtered_positions)} positions with occupancy < {args.min_occupancy:.2%}:")
    if filtered_positions:
        print(f"    Positions: {sorted(filtered_positions)}")
    else:
        print("    None")
        
    # Apply filter to DataFrame
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()
    print(f"  Remaining positions to evaluate: {df_filtered['position'].nunique()}")
    print(f"  Remaining scoring rows: {len(df_filtered)} (down from {len(df)})")
    
    # Setup output directories based on occupancy threshold
    occ_pct = int(args.min_occupancy * 100)
    if args.outdir is not None:
        out_dir = args.outdir
    else:
        out_dir = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Calculate Thresholds and Filter Switches
    percentiles = [95, 97, 99]
    hard_thresholds = [1.7, 1.8, 1.9]
    
    configs = []
    print("\nCalculating thresholds and filtering switches on filtered sites...")
    
    # Calculate percentiles on filtered dataset
    for p in percentiles:
        thresh = float(np.percentile(df_filtered["max_score"].dropna(), p))
        configs.append({
            "label": f"{p}th%",
            "value": thresh,
            "col_name": f"switches_p{p}",
            "type": "Percentile"
        })
        
    # Add hard thresholds
    for val in hard_thresholds:
        configs.append({
            "label": f"Hard {val}",
            "value": val,
            "col_name": f"switches_h{val}",
            "type": "Hard Threshold"
        })
        
    switches_dfs = {}
    for cfg in configs:
        thresh_val = cfg["value"]
        label = cfg["label"]
        sdf = df_filtered[df_filtered["max_score"] >= thresh_val].copy()
        switches_dfs[label] = sdf
        print(f"  {label} threshold: {thresh_val:.6f} ({len(sdf)} switches)")

    # 4. Compile Summary Statistics
    stats_rows = []
    for cfg in configs:
        label = cfg["label"]
        sdf = switches_dfs[label]
        unique_pos = sdf["position"].nunique()
        mean_dist = sdf["distance_from_root"].mean()
        median_dist = sdf["distance_from_root"].median()
        
        stats_rows.append({
            "threshold_label": label,
            "threshold_value": cfg["value"],
            "threshold_type": cfg["type"],
            "total_switches": len(sdf),
            "unique_positions": unique_pos,
            "mean_distance_from_root": mean_dist,
            "median_distance_from_root": median_dist
        })
        
    df_stats = pd.DataFrame(stats_rows)
    stats_csv = out_dir / "threshold_comparison_stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"\nSaved summary statistics to {stats_csv}")

    # 5. Compile Positional Switch Counts
    # Create a dense matrix of position (1-169) vs threshold switch counts
    all_positions = pd.DataFrame({"position": range(1, 170)})
    
    positional_dfs = [all_positions]
    for cfg in configs:
        label = cfg["label"]
        col_name = cfg["col_name"]
        pos_counts = (
            switches_dfs[label].groupby("position")
            .size()
            .reset_index(name=col_name)
        )
        positional_dfs.append(pos_counts)
        
    # Merge all and fill NaNs with 0
    df_positional = positional_dfs[0]
    for pos_df in positional_dfs[1:]:
        df_positional = df_positional.merge(pos_df, on="position", how="left")
    df_positional = df_positional.fillna(0).astype(int)
    
    positional_csv = out_dir / "positional_switches_comparison.csv"
    df_positional.to_csv(positional_csv, index=False)
    print(f"Saved positional switch comparison to {positional_csv}")

    # 6. Generate Multi-Panel Comparison Plot
    print("\nGenerating comparison plots...")
    sns.set_theme(style="whitegrid")
    
    # Setup 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    
    # Panel A: Total Switches & Unique Positions (Bar Plot)
    melted_stats = pd.melt(
        df_stats,
        id_vars=["threshold_label", "threshold_type"],
        value_vars=["total_switches", "unique_positions"],
        var_name="metric",
        value_name="count"
    )
    melted_stats["metric"] = melted_stats["metric"].replace({
        "total_switches": "Total Switch Events",
        "unique_positions": "Unique Positions (Sites)"
    })
    
    sns.barplot(
        data=melted_stats, x="threshold_label", y="count", hue="metric",
        palette=["#3498db", "#e74c3c"], ax=axes[0]
    )
    axes[0].set_title(f"A. Switch Events and Sites (Occupancy >= {args.min_occupancy:.0%})", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xlabel("Threshold Cutoff", fontsize=12)
    axes[0].legend(title="Metric", frameon=True, fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)
    
    # Add values on top of bars
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(
                f"{int(height)}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="semibold"
            )

    # Panel B: Positional Distribution Along the Protein Sequence (Line Plot)
    colors_list = sns.color_palette("muted", len(configs))
    colors = {cfg["label"]: colors_list[i] for i, cfg in enumerate(configs)}
    
    for cfg in configs:
        label = cfg["label"]
        col_name = cfg["col_name"]
        axes[1].plot(
            df_positional["position"],
            df_positional[col_name],
            label=f"{label} (Thresh: {cfg['value']:.3f})",
            color=colors[label],
            linewidth=1.3,
            alpha=0.8
        )
    axes[1].set_title("B. Switch Distribution Along Sequence (Filtered)", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xlabel("Protein Alignment Position", fontsize=12)
    axes[1].set_ylabel("Switch Count", fontsize=12)
    axes[1].set_xlim(1, 169)
    axes[1].legend(title="Thresholds", frameon=True, fontsize=9, loc="upper right")
    
    # Highlight the RAM domain (residues 68-152)
    axes[1].axvspan(68, 152, color="gray", alpha=0.1, label="RAM Domain")

    # Panel C: Evolutionary Time Distribution
    for cfg in configs:
        label = cfg["label"]
        sdf = switches_dfs[label]
        if len(sdf) > 0:
            sns.kdeplot(
                data=sdf,
                x="distance_from_root",
                label=f"{label} ({len(sdf)} switches)",
                color=colors[label],
                fill=True,
                alpha=0.08,
                linewidth=1.8,
                ax=axes[2]
            )
    axes[2].set_title("C. Switch Density Over Evolutionary Time (Filtered)", fontsize=14, fontweight="bold", pad=15)
    axes[2].set_xlabel("Evolutionary Distance from Root (branch length)", fontsize=12)
    axes[2].set_ylabel("Density of Divergence Events", fontsize=12)
    axes[2].legend(title="Thresholds", frameon=True, fontsize=9)

    plt.tight_layout()
    
    plot_svg = out_dir / "switch_threshold_comparison.svg"
    plot_png = out_dir / "switch_threshold_comparison.png"
    
    fig.savefig(str(plot_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Generated comparison plots:")
    print(f"  SVG: {plot_svg}")
    print(f"  PNG: {plot_png}")
    print(f"\nThreshold comparison analysis (Occupancy >= {args.min_occupancy:.0%}) completed successfully.")


if __name__ == "__main__":
    main()
