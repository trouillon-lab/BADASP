"""Generate detailed plots comparing Hard thresholds (1.7, 1.8, 1.9) and Percentile thresholds (95th, 97th, 99th) with MSA occupancy filtering.

Plots for both modes:
1. Positional switch counts with centered moving averages (window=5) and shaded low-occupancy regions.
2. Switch density (switches/site) per functional/architectural domain with active site counts.
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


def process_mode(
    mode: str,
    df: pd.DataFrame,
    filtered_positions: list,
    filtered_regions: list,
    out_dir: Path,
    min_occupancy: float,
    threshold_vals: dict
) -> None:
    """Process and generate plots/CSVs for either 'hard' or 'percentile' mode."""
    print(f"\nProcessing detailed plots for {mode} thresholds...")
    
    # Configure parameters based on mode
    if mode == "hard":
        thresholds = [1.7, 1.8, 1.9]
        col_prefix = "switches_h"
        color_palette = ["#2980b9", "#d35400", "#c0392b"]
        raw_colors = ["#3498db", "#f39c12", "#e74c3c"]
        title_label = "Hard Thresholds"
        file_prefix = "hard_thresholds"
        
        # Build labels and columns
        cfg_list = []
        for val in thresholds:
            label = f"Hard {val}"
            cfg_list.append({
                "val": val,
                "label": label,
                "col": f"{col_prefix}{val}",
                "legend_label": f"{label} (Thresh: {val:.1f})"
            })
    else:  # percentile
        thresholds = [95, 97, 99]
        col_prefix = "switches_p"
        color_palette = ["#2980b9", "#d35400", "#c0392b"]
        raw_colors = ["#3498db", "#f39c12", "#e74c3c"]
        title_label = "Percentile Thresholds"
        file_prefix = "percentile_thresholds"
        
        # Build labels and columns
        cfg_list = []
        for val in thresholds:
            label = f"{val}th%"
            thresh_val = threshold_vals.get(label, None)
            legend_label = f"{label} (Thresh: {thresh_val:.3f})" if thresh_val is not None else label
            cfg_list.append({
                "val": val,
                "label": label,
                "col": f"{col_prefix}{val}",
                "legend_label": legend_label
            })

    # Compute Moving Averages (centered window of 5)
    window_size = 5
    df_mode = df.copy()
    for cfg in cfg_list:
        df_mode[f"smooth_{cfg['col']}"] = df_mode[cfg['col']].rolling(window=window_size, center=True, min_periods=1).mean()
        
    # Define Architectural Domains (residue coordinates, 1-indexed)
    domains = {
        "N-terminus": (1, 5),
        "HTH Scaffold": (6, 34),
        "Recognition Helix": (35, 50),
        "HTH Linker": (51, 67),
        "RAM Domain": (68, 152),
        "C-terminus": (153, 169)
    }
    
    # Tally Switches and Active Sites by Domain
    domain_data = []
    for name, (start, end) in domains.items():
        domain_positions = list(range(start, end + 1))
        active_positions = [p for p in domain_positions if p not in filtered_positions]
        active_count = len(active_positions)
        total_count = len(domain_positions)
        
        mask = (df_mode["position"] >= start) & (df_mode["position"] <= end)
        
        row_dict = {
            "Domain": name,
            "Residues": f"{start}-{end}",
            "Length": total_count,
            "Active Sites": active_count
        }
        
        for cfg in cfg_list:
            sum_switches = df_mode.loc[mask, cfg['col']].sum()
            row_dict[f"{cfg['label']} Switches"] = sum_switches
            row_dict[f"{cfg['label']} Density"] = sum_switches / total_count if total_count > 0 else 0.0
            
        domain_data.append(row_dict)
        
    df_domains = pd.DataFrame(domain_data)
    df_domains.to_csv(out_dir / f"{file_prefix}_domain_counts.csv", index=False)
    print(f"  Saved domain counts and densities to {out_dir / f'{file_prefix}_domain_counts.csv'}")
    
    # Generate 2-Panel Figure
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    
    # Panel A: Positional Plot with Overlaid Lollipops (with slight offsets for distinct rendering)
    offsets = [-0.15, 0.0, 0.15]
    for i, cfg in enumerate(cfg_list):
        offset = offsets[i]
        axes[0].vlines(df_mode["position"] + offset, 0, df_mode[cfg['col']], color=color_palette[i], alpha=0.5, linewidth=1.0)
        axes[0].scatter(df_mode["position"] + offset, df_mode[cfg['col']], color=color_palette[i], s=20, label=cfg['legend_label'], zorder=3)
        
    # Annotate top 5 peaks for the primary/lowest threshold to display key positions
    prim_cfg = cfg_list[0]
    top_peaks = df_mode[df_mode[prim_cfg['col']] > 0].sort_values(prim_cfg['col'], ascending=False).head(5)
    for _, row in top_peaks.iterrows():
        pos = int(row["position"])
        val = int(row[prim_cfg['col']])
        axes[0].text(
            pos, val + 0.3,
            f"Col {pos}\n({val})",
            ha="center", va="bottom",
            fontsize=8, color="#2c3e50", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec=color_palette[0], lw=0.5, alpha=0.85),
            zorder=5
        )
        
    # Draw vertical shading for domains
    domain_colors = ["#ecf0f1", "#d5dbdb", "#a2d9ce", "#abebc6", "#fadbd8", "#fdebd0"]
    for i, (name, (start, end)) in enumerate(domains.items()):
        axes[0].axvspan(start, end, color=domain_colors[i % len(domain_colors)], alpha=0.2)
        midpoint = (start + end) / 2
        axes[0].text(midpoint, axes[0].get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
        
    # Draw vertical hatched spans over filtered-out regions
    first_patch = True
    for start, end in filtered_regions:
        axes[0].axvspan(
            start - 0.5, end + 0.5, 
            hatch="///", 
            facecolor="none", 
            edgecolor="#7f8c8d", 
            linewidth=0.8,
            alpha=0.4,
            label="Filtered (Low Occupancy < 80%)" if first_patch else ""
        )
        first_patch = False
        midpoint = (start + end) / 2
        axes[0].text(
            midpoint, axes[0].get_ylim()[0] + 0.5, 
            "Filtered", 
            ha="center", va="bottom", 
            fontsize=8, color="#7f8c8d", fontweight="bold", rotation=90
        )
        
    axes[0].set_title(f"A. Positional Switch Lollipop Distribution (Occupancy >= {min_occupancy:.0%})", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("Protein Alignment Position", fontsize=12)
    axes[0].set_ylabel("Switch Count (Divergence Events)", fontsize=12)
    axes[0].set_xlim(0.5, 169.5)
    axes[0].set_ylim(0, max(max(df_mode[cfg['col']].max() for cfg in cfg_list), 1) * 1.25)
    axes[0].legend(frameon=True, fontsize=10, loc="upper right")
    
    # Panel B: Domain Bar Plot (Switch Densities)
    # Melt domain data for plotting
    value_vars = [f"{cfg['label']} Density" for cfg in cfg_list]
    melted_domains = pd.melt(
        df_domains,
        id_vars=["Domain", "Residues", "Length", "Active Sites"],
        value_vars=value_vars,
        var_name="Threshold",
        value_name="Density"
    )
    
    # Clean up labels for the legend
    replace_dict = {f"{cfg['label']} Density": cfg['label'] for cfg in cfg_list}
    melted_domains["Threshold"] = melted_domains["Threshold"].replace(replace_dict)
    
    # Create clean x-axis labels
    x_labels = []
    for _, row in df_domains.iterrows():
        x_labels.append(f"{row['Domain']}\n({row['Active Sites']}/{row['Length']} sites active)")
        
    sns.barplot(
        data=melted_domains,
        x="Domain",
        y="Density",
        hue="Threshold",
        palette=color_palette,
        ax=axes[1]
    )
    
    axes[1].set_xticks(range(len(df_domains)))
    axes[1].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    axes[1].set_title(f"B. Switch Density across Domains (Switches / Residue, Occupancy >= {min_occupancy:.0%})", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_xlabel("Domain (Active / Total nominal residues)", fontsize=12)
    axes[1].set_ylabel("Switch Density (Events / Site)", fontsize=12)
    axes[1].legend(title="Threshold Cutoff", frameon=True, fontsize=10)
    
    # Add values on top of bars
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(
                f"{height:.2f}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
            
    # Add a visual "Filtered Out" indicator text over 100% filtered domains
    for i, (_, row) in enumerate(df_domains.iterrows()):
        if row["Active Sites"] == 0:
            axes[1].text(
                i, 0.05, 
                "FILTERED\n(Low Occupancy)", 
                ha="center", va="bottom", 
                fontsize=9, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )
            
    plt.tight_layout()
    
    plot_svg = out_dir / f"{file_prefix}_details.svg"
    plot_png = out_dir / f"{file_prefix}_details.png"
    
    fig.savefig(str(plot_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Generated detailed comparison plots:")
    print(f"    SVG: {plot_svg}")
    print(f"    PNG: {plot_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed Hard and Percentile Thresholds Plotting with MSA Occupancy Filtering")
    parser.add_argument("--min-occupancy", type=float, default=0.8, help="Occupancy threshold used for filtering (0.0 to 1.0)")
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--positional", type=Path, default=None, help="Path to positional comparison CSV (defaults to auto-resolved path)")
    args = parser.parse_args()

    occ_pct = int(args.min_occupancy * 100)
    
    # 1. Resolve Input Path
    if args.positional is None:
        pos_csv = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}/positional_switches_comparison.csv")
    else:
        pos_csv = args.positional
        
    if not pos_csv.exists():
        print(f"Error: Positional comparison file not found at {pos_csv}")
        print("Please run scripts/compare_thresholds.py first to generate this file.")
        sys.exit(1)
        
    print(f"Loading positional switch data from {pos_csv}...")
    df = pd.read_csv(pos_csv)
    
    # 2. Calculate Occupancy and Contiguous Filtered Regions
    occupancies = calculate_msa_occupancies(args.alignment)
    filtered_positions = [pos for pos, occ in occupancies.items() if occ < args.min_occupancy]
    filtered_regions = find_contiguous_regions(filtered_positions)
    
    # Setup Output Directory
    out_dir = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Load Threshold Values from Stats CSV if available
    stats_csv = out_dir / "threshold_comparison_stats.csv"
    threshold_vals = {}
    if stats_csv.exists():
        print(f"Loading threshold values from {stats_csv}...")
        stats_df = pd.read_csv(stats_csv)
        for _, row in stats_df.iterrows():
            threshold_vals[row["threshold_label"]] = row["threshold_value"]
            
    # 4. Process both Hard and Percentile modes
    process_mode("hard", df, filtered_positions, filtered_regions, out_dir, args.min_occupancy, threshold_vals)
    process_mode("percentile", df, filtered_positions, filtered_regions, out_dir, args.min_occupancy, threshold_vals)
    
    print("\nDetailed hard and percentile thresholds analysis completed successfully.")


if __name__ == "__main__":
    main()
