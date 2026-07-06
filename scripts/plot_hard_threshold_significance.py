"""Generate a detailed positional switch plot for Hard 1.7 threshold with Binomial significance lines.

Does not use a rolling window, showing the exact raw switch count per position.
Computes and overlays uncorrected (nominal 5%) and Bonferroni-corrected significance thresholds.
Saves outputs to results/badasp_scoring/threshold_comparison/occupancy_XX/.
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
import scipy.stats as stats
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


def find_contiguous_regions(positions: list) -> list:
    """Find contiguous ranges of integers in a list. Returns list of (start, end) tuples."""
    if not positions:
        return []
    
    sorted_pos = sorted(positions)
    regions = []
    start = sorted_pos[0]
    prev = sorted_pos[0]
    
    for pos in sorted_pos[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            regions.append((start, prev))
            start = pos
            prev = pos
    regions.append((start, prev))
    return regions


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed Hard 1.7 Significance Plotting with MSA Occupancy Filtering")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--positional", type=Path, default=None, help="Path to positional comparison CSV")
    parser.add_argument("--min-occupancy", type=float, default=0.8, help="Occupancy threshold used for filtering (0.0 to 1.0)")
    args = parser.parse_args()

    occ_pct = int(args.min_occupancy * 100)
    
    # 1. Resolve Input Paths
    if not args.scores.exists():
        print(f"Error: Raw scores file not found at {args.scores}")
        sys.exit(1)
        
    if args.positional is None:
        pos_csv = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}/positional_switches_comparison.csv")
    else:
        pos_csv = args.positional
        
    if not pos_csv.exists():
        print(f"Error: Positional comparison file not found at {pos_csv}")
        sys.exit(1)
        
    # Setup Output Directory
    out_dir = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Data
    print("Loading raw scores and calculating occupancies...")
    df_raw = pd.read_csv(args.scores)
    df_raw["max_score"] = df_raw[["badasp_score_left", "badasp_score_right"]].max(axis=1)
    
    occupancies = calculate_msa_occupancies(args.alignment)
    df_raw["occupancy"] = df_raw["position"].map(occupancies)
    
    filtered_positions = [pos for pos, occ in occupancies.items() if occ < args.min_occupancy]
    filtered_regions = find_contiguous_regions(filtered_positions)
    
    # Filter raw scores
    df_filtered = df_raw[df_raw["occupancy"] >= args.min_occupancy].copy()
    
    # 3. Calculate Statistical Significance Parameters
    # Total comparisons and unique active sites
    total_comparisons = len(df_filtered)
    num_active_positions = df_filtered["position"].nunique()
    
    # M = comparisons per position (scored internal node splits)
    M = total_comparisons // num_active_positions
    
    # Total switches at Hard 1.7
    total_switches = len(df_filtered[df_filtered["max_score"] >= 1.7])
    
    # Background probability of a switch occurring at any site/comparison
    p = total_switches / total_comparisons
    expected_switches = M * p
    
    # Find critical values programmatically
    # Critical value k is the smallest integer where P(X >= k) < alpha
    alpha_nominal = 0.05
    alpha_bonf = 0.05 / num_active_positions
    
    k_nominal = 0
    for k in range(1, M + 1):
        if stats.binom.sf(k - 1, M, p) < alpha_nominal:
            k_nominal = k
            break
            
    k_bonf = 0
    for k in range(1, M + 1):
        if stats.binom.sf(k - 1, M, p) < alpha_bonf:
            k_bonf = k
            break
            
    print(f"\nStatistical Significance Calculation (Hard 1.7):")
    print(f"  Scored Clade Splits (M): {M}")
    print(f"  Active Sites evaluated: {num_active_positions}")
    print(f"  Total switches at Hard 1.7: {total_switches}")
    print(f"  Background switch probability (p): {p:.6f} ({p:.4%})")
    print(f"  Expected switches per site (M * p): {expected_switches:.4f}")
    print(f"  Nominal 5% Significance Threshold: {k_nominal} (P(X >= {k_nominal}) = {stats.binom.sf(k_nominal-1, M, p):.6f})")
    print(f"  Bonferroni-Corrected 5% Threshold: {k_bonf} (P(X >= {k_bonf}) = {stats.binom.sf(k_bonf-1, M, p):.6f})")

    # 4. Load Positional Switch Counts
    print(f"Loading positional switch counts from {pos_csv}...")
    df_pos = pd.read_csv(pos_csv)
    
    # 5. Plot detailed significance figure
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Plot raw switch counts as a bar plot (needle/discrete representation)
    ax.bar(
        df_pos["position"],
        df_pos["switches_h1.7"],
        color="#2980b9",
        width=0.8,
        edgecolor="none",
        alpha=0.85,
        label="Raw Switch Count (Hard 1.7)",
        zorder=3
    )
    
    # Plot significance threshold lines
    ax.axhline(
        k_nominal,
        color="#27ae60",
        linestyle="--",
        linewidth=2.0,
        label=f"Nominal 5% Significance (k >= {k_nominal})",
        zorder=4
    )
    ax.axhline(
        k_bonf,
        color="#c0392b",
        linestyle="--",
        linewidth=2.0,
        label=f"Bonferroni-Corrected 5% Significance (k >= {k_bonf})",
        zorder=4
    )
    
    # Draw vertical shading for domains
    domains = {
        "N-terminus": (1, 5),
        "HTH Scaffold": (6, 34),
        "Recognition Helix": (35, 50),
        "HTH Linker": (51, 67),
        "RAM Domain": (68, 152),
        "C-terminus": (153, 169)
    }
    
    domain_colors = ["#ecf0f1", "#d5dbdb", "#a2d9ce", "#abebc6", "#fadbd8", "#fdebd0"]
    for i, (name, (start, end)) in enumerate(domains.items()):
        ax.axvspan(start - 0.5, end + 0.5, color=domain_colors[i % len(domain_colors)], alpha=0.15, zorder=1)
        midpoint = (start + end) / 2
        ax.text(
            midpoint, ax.get_ylim()[1] * 0.95, 
            name, 
            ha="center", va="top", 
            fontsize=10, fontweight="bold", color="#2c3e50",
            zorder=2
        )
        
    # Draw vertical hatched spans over filtered-out regions
    first_patch = True
    for start, end in filtered_regions:
        ax.axvspan(
            start - 0.5, end + 0.5, 
            hatch="///", 
            facecolor="none", 
            edgecolor="#7f8c8d", 
            linewidth=0.8,
            alpha=0.3,
            label="Filtered (Low Occupancy < 80%)" if first_patch else "",
            zorder=2
        )
        first_patch = False
        midpoint = (start + end) / 2
        ax.text(
            midpoint, ax.get_ylim()[0] + 0.5, 
            "Filtered", 
            ha="center", va="bottom", 
            fontsize=8, color="#7f8c8d", fontweight="bold", rotation=90,
            zorder=2
        )
        
    # Annotate specific residues exceeding the Bonferroni threshold
    signif_positions = df_pos[df_pos["switches_h1.7"] >= k_bonf]
    for _, row in signif_positions.iterrows():
        pos = row["position"]
        val = row["switches_h1.7"]
        ax.text(
            pos, val + 0.3, 
            f"Col {pos}\n({val})", 
            ha="center", va="bottom", 
            fontsize=8, color="#7f0000", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fdf2e9", ec="#f5b041", lw=0.5),
            zorder=5
        )
        
    # Add annotation box with statistical parameters
    stats_text = (
        f"Binomial Null Model Parameters:\n"
        f"• Clade comparisons per site (M): {M}\n"
        f"• Total active sites: {num_active_positions}\n"
        f"• Background switch rate (p): {p:.4%}\n"
        f"• Expected random switches/site: {expected_switches:.3f}\n"
        f"• nominal α = 0.05 cutoff: {k_nominal} switches\n"
        f"• Bonferroni α = 0.05/{num_active_positions} cutoff: {k_bonf} switches"
    )
    ax.text(
        0.02, 0.82, 
        stats_text, 
        transform=ax.transAxes, 
        fontsize=10, 
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdfefe", ec="#bdc3c7", lw=1.0, alpha=0.9),
        zorder=5
    )
    
    # Styling
    ax.set_title(f"Positional Switch Distribution with Binomial Significance Thresholds (Hard 1.7, Occupancy >= {args.min_occupancy:.0%})", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Protein Alignment Position", fontsize=12)
    ax.set_ylabel("Switch Count (Divergence Events)", fontsize=12)
    ax.set_xlim(0.5, 169.5)
    ax.set_ylim(0, max(df_pos["switches_h1.7"]) * 1.15)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    
    plt.tight_layout()
    
    plot_svg = out_dir / "hard_threshold_1.7_significance.svg"
    plot_png = out_dir / "hard_threshold_1.7_significance.png"
    
    fig.savefig(str(plot_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nGenerated significance plots successfully:")
    print(f"  SVG: {plot_svg}")
    print(f"  PNG: {plot_png}")
    print("Hard 1.7 significance plot analysis completed successfully.")


if __name__ == "__main__":
    main()
