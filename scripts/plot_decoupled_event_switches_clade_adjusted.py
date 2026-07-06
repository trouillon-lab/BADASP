"""Analyze and plot BADASP evolutionary switches adjusted for clade size.

Groups clade sizes into bins, computes adaptive percentile thresholds 
(stricter for small clades), and identifies switches using left/right clade sizes.
Generates:
- Threshold vs. Clade Size relationship plot.
- Comparative 4-panel switch plots for event-specific and overall thresholding.
- Event-agnostic overall switch plot.
- Grouped bar plot comparing domain switch proportions across multiple percentiles.
Saves outputs to results/badasp_scoring/clade_size_adjusted/occupancy_XX/.
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


# Architectural Domains
DOMAINS = {
    "N-terminus": (1, 5),
    "HTH Scaffold": (6, 34),
    "Recognition Helix": (35, 50),
    "HTH Linker": (51, 67),
    "RAM Domain": (68, 152),
    "C-terminus": (153, 169)
}

EVENT_COLORS = {
    "Duplication": "#c0392b",
    "Speciation": "#2980b9",
    "Transfer": "#27ae60",
    "Overall": "#2c3e50"
}


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


def bin_clade_sizes(series: pd.Series, method: str, num_bins: int = 10) -> tuple:
    """Group clade sizes into categorical bins.
    
    Returns:
        Categorical Series of bins and sorted bin categories.
    """
    if method == "quantile":
        bins = pd.qcut(series, q=num_bins, duplicates="drop")
        categories = sorted(bins.cat.categories)
        return bins, categories
    
    elif method == "log-spaced":
        custom_edges = [5, 8, 13, 22, 38, 66, 115, 200, 350, 600, 1200, np.inf]
        bins = pd.cut(series, bins=custom_edges, include_lowest=True)
        categories = sorted(bins.cat.categories)
        return bins, categories
    
    else:
        raise ValueError(f"Unknown bin method: {method}")


def calculate_bin_thresholds(melted_df: pd.DataFrame, score_col: str, bin_col: str, percentile: float = 95) -> dict:
    """Calculate percentile threshold per bin for each event and overall.
    
    Returns:
        dict: {(event_type, bin_interval): threshold}
    """
    thresholds = {}
    
    # 1. Event-specific thresholds
    for event in ["Duplication", "Speciation", "Transfer"]:
        event_df = melted_df[melted_df["event_type"] == event]
        for bin_interval in melted_df[bin_col].cat.categories:
            bin_df = event_df[event_df[bin_col] == bin_interval]
            scores = bin_df[score_col].dropna()
            if len(scores) > 0:
                thresholds[(event, bin_interval)] = float(np.percentile(scores, percentile))
            else:
                thresholds[(event, bin_interval)] = np.nan
                
    # 2. Overall (event-agnostic) thresholds
    for bin_interval in melted_df[bin_col].cat.categories:
        bin_df = melted_df[melted_df[bin_col] == bin_interval]
        scores = bin_df[score_col].dropna()
        if len(scores) > 0:
            thresholds[("overall", bin_interval)] = float(np.percentile(scores, percentile))
        else:
            thresholds[("overall", bin_interval)] = np.nan
            
    return thresholds


def identify_switches(df: pd.DataFrame, thresholds: dict, event_specific: bool) -> pd.Series:
    """Determine if a comparison has a switch on left or right child branches.
    
    Condition: badasp_score >= bin_threshold
    """
    first_key = list(thresholds.keys())[0]
    intervals = [k[1] for k in thresholds.keys() if k[0] == first_key[0]]
    
    def _map_to_bin(val):
        for interval in intervals:
            if val in interval:
                return interval
        return intervals[-1] if val > intervals[-1].right else intervals[0]
        
    df = df.copy()
    
    if "bin_left" not in df.columns:
        df["bin_left"] = df["clade_size_left"].apply(_map_to_bin)
    if "bin_right" not in df.columns:
        df["bin_right"] = df["clade_size_right"].apply(_map_to_bin)
        
    is_switch = []
    for _, row in df.iterrows():
        event = row["event_type"]
        bin_l = row["bin_left"]
        bin_r = row["bin_right"]
        
        if event_specific:
            thresh_l = thresholds.get((event, bin_l), np.inf)
            thresh_r = thresholds.get((event, bin_r), np.inf)
        else:
            thresh_l = thresholds.get(("overall", bin_l), np.inf)
            thresh_r = thresholds.get(("overall", bin_r), np.inf)
            
        score_l = row["badasp_score_left"]
        score_r = row["badasp_score_right"]
        
        switch_l = (not np.isnan(score_l)) and (score_l >= thresh_l)
        switch_r = (not np.isnan(score_r)) and (score_r >= thresh_r)
        
        is_switch.append(switch_l or switch_r)
        
    return pd.Series(is_switch, index=df.index)


def plot_threshold_relationships(melted_df: pd.DataFrame, thresholds: dict, bin_col: str, percentile: float, out_dir: Path) -> None:
    """Plot the threshold vs. clade size relationship curves."""
    print("Generating threshold vs. clade size relationship plot...")
    plt.figure(figsize=(10, 6))
    
    bin_x_coords = {}
    for bin_interval in melted_df[bin_col].cat.categories:
        bin_vals = melted_df[melted_df[bin_col] == bin_interval]["clade_size"]
        if not bin_vals.empty:
            bin_x_coords[bin_interval] = np.exp(np.log(bin_vals.clip(lower=1)).mean())
        else:
            bin_x_coords[bin_interval] = bin_interval.mid
            
    sorted_bins = sorted(bin_x_coords.keys(), key=lambda x: x.left)
    x_vals = [bin_x_coords[b] for b in sorted_bins]
    
    styles = {
        "Duplication": {"color": EVENT_COLORS["Duplication"], "marker": "o", "linestyle": "-"},
        "Speciation": {"color": EVENT_COLORS["Speciation"], "marker": "s", "linestyle": "-"},
        "Transfer": {"color": EVENT_COLORS["Transfer"], "marker": "^", "linestyle": "-"},
        "overall": {"color": EVENT_COLORS["Overall"], "marker": "x", "linestyle": "--", "label": "Overall (Event-Agnostic)"}
    }
    
    for key, style in styles.items():
        y_vals = [thresholds.get((key, b), np.nan) for b in sorted_bins]
        
        valid_idx = [i for i, y in enumerate(y_vals) if not np.isnan(y)]
        plot_x = [x_vals[i] for i in valid_idx]
        plot_y = [y_vals[i] for i in valid_idx]
        
        label = style.get("label", key)
        plt.plot(
            plot_x, plot_y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=8,
            label=label
        )
        
    plt.xscale("log")
    plt.xlabel("Clade Size (geometric mean of bin, log scale)", fontsize=12)
    plt.ylabel(f"{percentile}th Percentile BADASP Threshold", fontsize=12)
    plt.title(f"BADASP Switch Threshold ({percentile}th%) vs. Clade Size Relationship", fontsize=14, fontweight="bold", pad=15)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(frameon=True, fontsize=10.5, loc="upper right")
    plt.tight_layout()
    
    plt.savefig(out_dir / "clade_adjusted_threshold_relationship.svg", format="svg")
    plt.savefig(out_dir / "clade_adjusted_threshold_relationship.png", format="png", dpi=300)
    plt.close()
    print(f"Saved threshold relationship plot to {out_dir}")


def generate_switch_lollipop_plot(df_filtered: pd.DataFrame, df_pos: pd.DataFrame, df_domains: pd.DataFrame, 
                                  event_specific: bool, percentile: float, filtered_regions: list, 
                                  filtered_positions: list, out_file_base: Path) -> None:
    """Generate 4-panel lollipop figure comparing positional switches and domain densities."""
    print(f"Generating 4-panel figure for {'event_specific' if event_specific else 'overall'} thresholds...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(4, 1, figsize=(18, 24), sharey=False)
    
    event_types = ["Duplication", "Speciation", "Transfer"]
    domain_colors = ["#ecf0f1", "#d5dbdb", "#a2d9ce", "#abebc6", "#fadbd8", "#fdebd0"]
    
    for i, event in enumerate(event_types):
        ax = axes[i]
        col_sw = f"{event.lower()}_switches_adjusted"
        
        color = EVENT_COLORS[event]
        
        ax.vlines(df_pos["position"], 0, df_pos[col_sw], color=color, alpha=0.7, linewidth=1.5)
        ax.scatter(df_pos["position"], df_pos[col_sw], color=color, s=25, zorder=3, label=f"Clade-Adjusted {percentile}th% Switches")
        
        top_peaks = df_pos[df_pos[col_sw] > 0].sort_values(col_sw, ascending=False).head(5)
        for _, row in top_peaks.iterrows():
            pos = int(row["position"])
            val = int(row[col_sw])
            ax.text(
                pos, val + 0.3,
                f"Col {pos}\n({val})",
                ha="center", va="bottom",
                fontsize=8, color="#2c3e50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec=color, lw=0.5, alpha=0.85),
                zorder=5
            )
            
        for d_idx, (name, (start, end)) in enumerate(DOMAINS.items()):
            ax.axvspan(start, end, color=domain_colors[d_idx % len(domain_colors)], alpha=0.15)
            midpoint = (start + end) / 2
            ax.text(midpoint, ax.get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
            
        first_patch = True
        for start, end in filtered_regions:
            ax.axvspan(
                start - 0.5, end + 0.5, 
                hatch="///", facecolor="none", edgecolor="#7f8c8d", linewidth=0.8, alpha=0.4,
                label="Filtered (Low Occupancy < 80%)" if first_patch else ""
            )
            first_patch = False
            midpoint = (start + end) / 2
            ax.text(
                midpoint, ax.get_ylim()[0] + 0.2, 
                "Filtered", 
                ha="center", va="bottom", fontsize=8, color="#7f8c8d", fontweight="bold", rotation=90
            )
            
        strat_lbl = "Event-Specific" if event_specific else "Overall-Agnostic"
        ax.set_title(f"{chr(65+i)}. {event}-Arising Switches ({strat_lbl} Clade-Adjusted {percentile}th% Thresholds)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Protein Alignment Position", fontsize=11)
        ax.set_ylabel("Switch Count", fontsize=11)
        ax.set_xlim(0.5, 169.5)
        ax.set_ylim(0, max(df_pos[col_sw].max(), 1) * 1.25)
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        
    x_labels = []
    for name in DOMAINS.keys():
        row_dup = df_domains[(df_domains["domain"] == name) & (df_domains["event_type"] == "Duplication")].iloc[0]
        x_labels.append(f"{name}\n({row_dup['active_sites']}/{row_dup['total_sites']} sites active)")
        
    sns.barplot(
        data=df_domains,
        x="domain",
        y="density_adjusted",
        hue="event_type",
        palette=[EVENT_COLORS["Duplication"], EVENT_COLORS["Speciation"], EVENT_COLORS["Transfer"]],
        ax=axes[3]
    )
    
    axes[3].set_xticks(range(len(DOMAINS)))
    axes[3].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    strat_lbl = "Event-Specific" if event_specific else "Overall-Agnostic"
    axes[3].set_title(f"D. Switch Density across Domains ({strat_lbl} Clade-Adjusted {percentile}th% thresholds, Occupancy >= 80%)", fontsize=13, fontweight="bold", pad=15)
    axes[3].set_xlabel("Domain (Active / Total nominal residues)", fontsize=11)
    axes[3].set_ylabel("Switch Density (Events / Site)", fontsize=11)
    axes[3].legend(title="Evolutionary Event Type", frameon=True, fontsize=10.5)
    
    for p in axes[3].patches:
        height = p.get_height()
        if height > 0:
            axes[3].annotate(
                f"{height:.2f}",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
            
    for i, name in enumerate(DOMAINS.keys()):
        row_dup = df_domains[(df_domains["domain"] == name) & (df_domains["event_type"] == "Duplication")].iloc[0]
        if row_dup["active_sites"] == 0:
            axes[3].text(
                i, 0.02, 
                "FILTERED", 
                ha="center", va="bottom", 
                fontsize=9, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )
            
    plt.tight_layout()
    
    fig.savefig(f"{out_file_base}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{out_file_base}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lollipop comparison figure to {out_file_base}")


def generate_event_agnostic_lollipop_plot(df_pos: pd.DataFrame, df_domains: pd.DataFrame, 
                                          percentile: float, filtered_regions: list, 
                                          filtered_positions: list, out_file_base: Path) -> None:
    """Generate 2-panel figure showing switches event agnostically (lollipop + domain density)."""
    print(f"Generating event-agnostic 2-panel figure for {percentile}th% thresholds...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharey=False)
    
    domain_colors = ["#ecf0f1", "#d5dbdb", "#a2d9ce", "#abebc6", "#fadbd8", "#fdebd0"]
    
    # Panel A: Lollipop plot of aggregated overall switches (stacked by category contribution)
    ax = axes[0]
    pos = df_pos["position"]
    dup = df_pos["duplication_switches_overall"]
    spec = df_pos["speciation_switches_overall"]
    trans = df_pos["transfer_switches_overall"]
    total = df_pos["overall_switches_adjusted"]

    # Plot stacked segments
    width = 0.4
    ax.bar(pos, dup, width=width, color=EVENT_COLORS["Duplication"], alpha=0.85, label="Duplication Contribution", zorder=3)
    ax.bar(pos, spec, bottom=dup, width=width, color=EVENT_COLORS["Speciation"], alpha=0.85, label="Speciation Contribution", zorder=3)
    ax.bar(pos, trans, bottom=dup+spec, width=width, color=EVENT_COLORS["Transfer"], alpha=0.85, label="Transfer Contribution", zorder=3)
    
    # White dots with dark outline at the top of the stacked stems
    active_mask = total > 0
    ax.scatter(pos[active_mask], total[active_mask], color="#ffffff", edgecolor="#2c3e50", linewidth=1.2, s=30, zorder=4, label=f"Overall {percentile}th% Switches")
    
    top_peaks = df_pos.sort_values("overall_switches_adjusted", ascending=False).head(5)
    for _, row in top_peaks.iterrows():
        pos_val = int(row["position"])
        val = int(row["overall_switches_adjusted"])
        if val > 0:
            ax.text(
                pos_val, val + 0.3,
                f"Col {pos_val}\n({val})",
                ha="center", va="bottom",
                fontsize=8, color="#2c3e50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec="#2c3e50", lw=0.5, alpha=0.85),
                zorder=5
            )
            
    for d_idx, (name, (start, end)) in enumerate(DOMAINS.items()):
        ax.axvspan(start, end, color=domain_colors[d_idx % len(domain_colors)], alpha=0.15)
        midpoint = (start + end) / 2
        ax.text(midpoint, ax.get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
        
    first_patch = True
    for start, end in filtered_regions:
        ax.axvspan(
            start - 0.5, end + 0.5, 
            hatch="///", facecolor="none", edgecolor="#7f8c8d", linewidth=0.8, alpha=0.4,
            label="Filtered (Low Occupancy < 80%)" if first_patch else ""
        )
        first_patch = False
        midpoint = (start + end) / 2
        ax.text(
            midpoint, ax.get_ylim()[0] + 0.2, 
            "Filtered", 
            ha="center", va="bottom", fontsize=8, color="#7f8c8d", fontweight="bold", rotation=90
        )
        
    ax.set_title(f"A. Event-Agnostic Overall Switches (Clade-Adjusted {percentile}th% Thresholds)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Protein Alignment Position", fontsize=11)
    ax.set_ylabel("Switch Count (All events combined)", fontsize=11)
    ax.set_xlim(0.5, 169.5)
    ax.set_ylim(0, max(total.max(), 1) * 1.25)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")
    
    # Panel B: Bar plot of domain switch density for overall switches (stacked by category)
    x_labels = []
    dup_densities = []
    spec_densities = []
    trans_densities = []
    total_densities = []
    
    for name, (start, end) in DOMAINS.items():
        row = df_domains[df_domains["domain"] == name].iloc[0]
        x_labels.append(f"{name}\n({row['active_sites']}/{row['total_sites']} sites active)")
        
        total_sites = end - start + 1
        dup_sw = df_pos.loc[(df_pos["position"] >= start) & (df_pos["position"] <= end), "duplication_switches_overall"].sum()
        spec_sw = df_pos.loc[(df_pos["position"] >= start) & (df_pos["position"] <= end), "speciation_switches_overall"].sum()
        trans_sw = df_pos.loc[(df_pos["position"] >= start) & (df_pos["position"] <= end), "transfer_switches_overall"].sum()
        
        dup_densities.append(dup_sw / total_sites if total_sites > 0 else 0.0)
        spec_densities.append(spec_sw / total_sites if total_sites > 0 else 0.0)
        trans_densities.append(trans_sw / total_sites if total_sites > 0 else 0.0)
        total_densities.append((dup_sw + spec_sw + trans_sw) / total_sites if total_sites > 0 else 0.0)
        
    x_indices = np.arange(len(DOMAINS))
    width = 0.5
    
    axes[1].bar(x_indices, dup_densities, width=width, color=EVENT_COLORS["Duplication"], alpha=0.85, label="Duplication Contribution", zorder=3)
    axes[1].bar(x_indices, spec_densities, bottom=dup_densities, width=width, color=EVENT_COLORS["Speciation"], alpha=0.85, label="Speciation Contribution", zorder=3)
    axes[1].bar(x_indices, trans_densities, bottom=np.array(dup_densities) + np.array(spec_densities), width=width, color=EVENT_COLORS["Transfer"], alpha=0.85, label="Transfer Contribution", zorder=3)
    
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    axes[1].set_title(f"B. Event-Agnostic Switch Density across Domains (Clade-Adjusted {percentile}th% thresholds, Occupancy >= 80%)", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_xlabel("Domain (Active / Total nominal residues)", fontsize=11)
    axes[1].set_ylabel("Switch Density (Events / Site)", fontsize=11)
    axes[1].legend(frameon=True, fontsize=9.5, loc="upper right")
    
    for x_pos, total_density in zip(x_indices, total_densities):
        if total_density > 0:
            axes[1].annotate(
                f"{total_density:.2f}",
                xy=(x_pos, total_density),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
            
    for i, name in enumerate(DOMAINS.keys()):
        row = df_domains[df_domains["domain"] == name].iloc[0]
        if row["active_sites"] == 0:
            axes[1].text(
                i, 0.02, 
                "FILTERED", 
                ha="center", va="bottom", 
                fontsize=9, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )
            
    plt.tight_layout()
    
    fig.savefig(f"{out_file_base}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{out_file_base}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved event-agnostic lollipop comparison figure to {out_file_base}")


def plot_domain_proportions_comparison(props_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot the proportion of switches per domain compared across multiple thresholds."""
    print("Generating domain switch proportions comparison plot...")
    plt.figure(figsize=(12, 7))
    
    sns.barplot(
        data=props_df,
        x="domain",
        y="proportion",
        hue="threshold",
        palette="crest"  # Nice sequential palette
    )
    
    plt.title("Proportion of Event-Agnostic Switches in Each Domain across Thresholds", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Architectural Domain", fontsize=12)
    plt.ylabel("Proportion of Total Switches (Normalized per threshold)", fontsize=12)
    plt.legend(title="Percentile Threshold", frameon=True, fontsize=10.5)
    plt.tight_layout()
    
    plt.savefig(out_dir / "domain_switch_proportions_by_threshold.svg", format="svg")
    plt.savefig(out_dir / "domain_switch_proportions_by_threshold.png", format="png", dpi=300)
    plt.close()
    print(f"Saved domain proportions bar plot to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BADASP Clade-Size-Adjusted Switch Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8, help="Occupancy threshold used for filtering (0.0 to 1.0)")
    parser.add_argument("--bin-method", type=str, default="quantile", choices=["quantile", "log-spaced"], help="Binning strategy")
    parser.add_argument("--num-bins", type=int, default=10, help="Number of bins for quantile strategy")
    parser.add_argument("--percentile", type=float, default=99.0, help="Target percentile threshold for switch definitions")
    parser.add_argument("--min-clade-size", type=int, default=5, help="Minimum clade size filter for comparisons")
    args = parser.parse_args()

    occ_pct = int(args.min_occupancy * 100)
    out_dir = Path(f"results/badasp_scoring/clade_size_adjusted/min_clade_{args.min_clade_size}/occupancy_{occ_pct}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    if not args.scores.exists():
        print(f"Error: Raw scores file not found at {args.scores}")
        sys.exit(1)
        
    print(f"Loading data (filtering clade sizes >= {args.min_clade_size})...")
    df = pd.read_csv(args.scores)
    df = df[(df["clade_size_left"] >= args.min_clade_size) & (df["clade_size_right"] >= args.min_clade_size)]
    
    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    
    filtered_positions = [pos for pos, occ in occupancies.items() if occ < args.min_occupancy]
    filtered_regions = find_contiguous_regions(filtered_positions)
    
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()
    
    # 2. Melt left and right child scores to build the background population
    df_left = df_filtered[
        ["node_name", "event_type", "position", "clade_size_left", "badasp_score_left"]
    ].rename(columns={"badasp_score_left": "score", "clade_size_left": "clade_size"})
    
    df_right = df_filtered[
        ["node_name", "event_type", "position", "clade_size_right", "badasp_score_right"]
    ].rename(columns={"badasp_score_right": "score", "clade_size_right": "clade_size"})
    
    melted_df = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["score", "clade_size"])
    
    # 3. Bin clade sizes
    print(f"Binning clade sizes using method: {args.bin_method} (bins={args.num_bins if args.bin_method == 'quantile' else 'custom log-spaced'})...")
    melted_df["clade_bin"], bin_categories = bin_clade_sizes(melted_df["clade_size"], args.bin_method, args.num_bins)
    
    # Map bins to main df
    first_key = bin_categories[0]
    def _map_to_bin(val):
        for interval in bin_categories:
            if val in interval:
                return interval
        return bin_categories[-1] if val > bin_categories[-1].right else bin_categories[0]
        
    df_filtered["bin_left"] = df_filtered["clade_size_left"].apply(_map_to_bin)
    df_filtered["bin_right"] = df_filtered["clade_size_right"].apply(_map_to_bin)
    
    # 4. Calculate adaptive thresholds for multiple percentiles: 95, 97, 99, 99.9
    percentiles_to_check = [95.0, 97.0, 99.0, 99.9]
    all_thresholds = {}
    for p in percentiles_to_check:
        all_thresholds[p] = calculate_bin_thresholds(melted_df, "score", "clade_bin", percentile=p)
        
    # Plot relationship curve for the target percentile
    target_thresholds = all_thresholds[args.percentile]
    plot_threshold_relationships(melted_df, target_thresholds, "clade_bin", args.percentile, out_dir)
    
    # 5. Evaluate target switches and export tables for Event-Specific and Overall
    stats_rows = []
    df_pos = pd.DataFrame({"position": range(1, 170)})
    domain_rows = []
    
    for strategy_name, is_event_specific in [("event_specific", True), ("overall", False)]:
        # Determine switches on parent dataframe
        df_filtered[f"is_switch_{strategy_name}"] = identify_switches(
            df_filtered, target_thresholds, event_specific=is_event_specific
        )
        
        # Tally switches
        for event in ["Duplication", "Speciation", "Transfer"]:
            df_event = df_filtered[df_filtered["event_type"] == event]
            total_comp = len(df_event)
            
            if total_comp == 0:
                continue
                
            sw_count = df_event[f"is_switch_{strategy_name}"].sum()
            
            stats_rows.append({
                "strategy": strategy_name,
                "event_type": event,
                "total_comparisons": total_comp,
                "clade_adjusted_switches": sw_count,
                "clade_adjusted_proportion": sw_count / total_comp if total_comp > 0 else 0.0
            })
            
            print(f"Percentile {args.percentile} | Strategy: {strategy_name} | {event}: {sw_count} switches / {total_comp} comparisons")
            
            # Save positional counts
            sdf = df_event[df_event[f"is_switch_{strategy_name}"]]
            pos_counts = sdf.groupby("position").size().reset_index(name=f"{event.lower()}_switches_adjusted")
            
            df_pos_temp = pd.DataFrame({"position": range(1, 170)})
            df_pos_temp = df_pos_temp.merge(pos_counts, on="position", how="left").fillna(0)
            df_pos[f"{event.lower()}_switches_{strategy_name}"] = df_pos_temp[f"{event.lower()}_switches_adjusted"].astype(int)
            
            # Domain Switch Densities
            for name, (start, end) in DOMAINS.items():
                domain_positions = list(range(start, end + 1))
                active_positions = [p for p in domain_positions if p not in filtered_positions]
                active_count = len(active_positions)
                total_count = len(domain_positions)
                
                mask = (df_pos_temp["position"] >= start) & (df_pos_temp["position"] <= end)
                sum_sw = df_pos_temp.loc[mask, f"{event.lower()}_switches_adjusted"].sum()
                
                domain_rows.append({
                    "strategy": strategy_name,
                    "event_type": event,
                    "domain": name,
                    "residues": f"{start}-{end}",
                    "total_sites": total_count,
                    "active_sites": active_count,
                    "switches_adjusted": sum_sw,
                    "density_adjusted": sum_sw / total_count if total_count > 0 else 0.0
                })
                
        # Generate 4-panel switch plots for this strategy
        df_pos_for_plot = df_pos[["position"]].copy()
        for event in ["Duplication", "Speciation", "Transfer"]:
            df_pos_for_plot[f"{event.lower()}_switches_adjusted"] = df_pos[f"{event.lower()}_switches_{strategy_name}"]
            
        df_domains_for_plot = pd.DataFrame([r for r in domain_rows if r["strategy"] == strategy_name])
        
        generate_switch_lollipop_plot(
            df_filtered=df_filtered,
            df_pos=df_pos_for_plot,
            df_domains=df_domains_for_plot,
            event_specific=is_event_specific,
            percentile=args.percentile,
            filtered_regions=filtered_regions,
            filtered_positions=filtered_positions,
            out_file_base=out_dir / f"decoupled_event_switches_clade_adjusted_{strategy_name}"
        )
        
    # 6. Generate Event-Agnostic overall switch plot
    # Compute total switch count per position across all event types combined under overall (event-agnostic) thresholds
    df_filtered["is_switch_overall_agnostic"] = identify_switches(
        df_filtered, target_thresholds, event_specific=False
    )
    
    overall_pos_counts = df_filtered[df_filtered["is_switch_overall_agnostic"]].groupby("position").size().reset_index(name="overall_switches")
    df_pos_overall = pd.DataFrame({"position": range(1, 170)}).merge(overall_pos_counts, on="position", how="left").fillna(0)
    df_pos_overall["overall_switches"] = df_pos_overall["overall_switches"].astype(int)
    
    # Save overall agnostic count into df_pos as a column
    df_pos["overall_switches_adjusted"] = df_pos_overall["overall_switches"]
    
    # Domain Switch Densities for Overall switches
    overall_domain_rows = []
    for name, (start, end) in DOMAINS.items():
        domain_positions = list(range(start, end + 1))
        active_positions = [p for p in domain_positions if p not in filtered_positions]
        active_count = len(active_positions)
        total_count = len(domain_positions)
        
        mask = (df_pos_overall["position"] >= start) & (df_pos_overall["position"] <= end)
        sum_sw = df_pos_overall.loc[mask, "overall_switches"].sum()
        
        overall_domain_rows.append({
            "domain": name,
            "residues": f"{start}-{end}",
            "total_sites": total_count,
            "active_sites": active_count,
            "switches_adjusted": sum_sw,
            "density_adjusted": sum_sw / total_count if total_count > 0 else 0.0
        })
    df_domains_overall = pd.DataFrame(overall_domain_rows)
    
    generate_event_agnostic_lollipop_plot(
        df_pos=df_pos,
        df_domains=df_domains_overall,
        percentile=args.percentile,
        filtered_regions=filtered_regions,
        filtered_positions=filtered_positions,
        out_file_base=out_dir / "event_agnostic_switches_clade_adjusted"
    )
    
    # 7. Generate multi-percentile domain proportion comparison (Fisher-style normalization)
    # Compare 95th, 97th, 99th, 99.9th percentiles
    proportion_rows = []
    for p in percentiles_to_check:
        p_thresholds = all_thresholds[p]
        df_p_switches = identify_switches(df_filtered, p_thresholds, event_specific=False)
        
        # Aggregate count per position
        p_pos_counts = df_filtered[df_p_switches].groupby("position").size().reset_index(name="switches")
        p_pos_all = pd.DataFrame({"position": range(1, 170)}).merge(p_pos_counts, on="position", how="left").fillna(0)
        
        total_switches_at_p = p_pos_all["switches"].sum()
        print(f"Total overall switches at {p}th%: {total_switches_at_p}")
        
        for name, (start, end) in DOMAINS.items():
            mask = (p_pos_all["position"] >= start) & (p_pos_all["position"] <= end)
            domain_sw_count = p_pos_all.loc[mask, "switches"].sum()
            
            proportion = domain_sw_count / total_switches_at_p if total_switches_at_p > 0 else 0.0
            proportion_rows.append({
                "threshold": f"{p}th%",
                "domain": name,
                "switch_count": domain_sw_count,
                "proportion": proportion
            })
            
    df_proportions = pd.DataFrame(proportion_rows)
    plot_domain_proportions_comparison(df_proportions, out_dir)
    
    # 8. Save data tables
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(out_dir / "event_decoupled_stats_clade_adjusted.csv", index=False)
    df_pos.to_csv(out_dir / "event_positional_switches_clade_adjusted.csv", index=False)
    
    # Combine regular and overall domain densities to save
    df_domains = pd.DataFrame(domain_rows)
    df_domains_overall["strategy"] = "overall_agnostic"
    df_domains_overall["event_type"] = "Overall"
    df_domains_all = pd.concat([df_domains, df_domains_overall], ignore_index=True)
    df_domains_all.to_csv(out_dir / "event_domain_densities_clade_adjusted.csv", index=False)
    
    print("\nClade size-adjusted switch analysis completed successfully.")


if __name__ == "__main__":
    main()
