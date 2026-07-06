#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from Bio import AlignIO

# Define domains and colors to match project styling
EVENT_COLORS = {
    "Duplication": "#D62728",
    "Speciation": "#1F77B4",
    "Transfer": "#2CA02C",
    "Overall": "#7F7F7F"
}


def calculate_msa_occupancies(alignment_path: Path) -> dict:
    alignment = AlignIO.read(alignment_path, "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)
    return occupancies


def compute_flat_vs_adjusted(scores_path: Path, alignment_path: Path, min_clade: int, min_occ: float) -> pd.DataFrame:
    df = pd.read_csv(scores_path)
    df = df[(df["clade_size_left"] >= min_clade) & (df["clade_size_right"] >= min_clade)]
    
    occupancies = calculate_msa_occupancies(alignment_path)
    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= min_occ].copy()

    # Melt to get background
    df_left = df_filtered[["badasp_score_left", "event_type"]].rename(columns={"badasp_score_left": "score"})
    df_right = df_filtered[["badasp_score_right", "event_type"]].rename(columns={"badasp_score_right": "score"})
    melted = pd.concat([df_left, df_right], ignore_index=True).dropna()

    # Overall flat 99th% threshold
    flat_overall_thresh = np.percentile(melted["score"], 99.0)
    # Event-specific flat 99th% thresholds
    flat_event_thresholds = {}
    for event in ["Duplication", "Speciation", "Transfer"]:
        event_scores = melted[melted["event_type"] == event]["score"]
        flat_event_thresholds[event] = np.percentile(event_scores, 99.0)

    # Count flat overall switches
    df_filtered["is_switch_flat_overall"] = (df_filtered["badasp_score_left"] >= flat_overall_thresh) | (df_filtered["badasp_score_right"] >= flat_overall_thresh)
    # Count flat specific switches
    df_filtered["is_switch_flat_specific"] = False
    for event in ["Duplication", "Speciation", "Transfer"]:
        thresh = flat_event_thresholds[event]
        mask = df_filtered["event_type"] == event
        df_filtered.loc[mask, "is_switch_flat_specific"] = (df_filtered.loc[mask, "badasp_score_left"] >= thresh) | (df_filtered.loc[mask, "badasp_score_right"] >= thresh)

    # Compile table
    rows = []
    for event in ["Duplication", "Speciation", "Transfer"]:
        total_comp = len(df_filtered[df_filtered["event_type"] == event])
        flat_overall = df_filtered[df_filtered["event_type"] == event]["is_switch_flat_overall"].sum()
        flat_spec = df_filtered[df_filtered["event_type"] == event]["is_switch_flat_specific"].sum()
        rows.append({
            "min_clade": min_clade,
            "event_type": event,
            "total_comparisons": total_comp,
            "flat_overall_switches": flat_overall,
            "flat_overall_proportion": flat_overall / total_comp,
            "flat_event_specific_switches": flat_spec,
            "flat_event_specific_proportion": flat_spec / total_comp
        })
    return pd.DataFrame(rows)


def plot_custom_upset(membership_dict: dict, set_names: list, out_file_base: Path) -> None:
    """Generate a custom publication-quality UpSet plot."""
    sns.set_theme(style="white")
    
    # Sort intersections by size descending
    sorted_intersections = sorted(membership_dict.items(), key=lambda item: len(item[1]), reverse=True)
    
    labels = [item[0] for item in sorted_intersections]
    sizes = [len(item[1]) for item in sorted_intersections]
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], width_ratios=[1, 4], hspace=0.15, wspace=0.1)
    
    # 1. Top-right: Bar plot of intersection sizes
    ax_bar = plt.subplot(gs[0, 1])
    bars = ax_bar.bar(range(len(sizes)), sizes, color="#34495e", width=0.6, edgecolor="#2c3e50")
    ax_bar.set_ylabel("Intersection Size\n(Number of Residues)", fontsize=11, fontweight="bold")
    ax_bar.set_title("UpSet Plot of Switched residues (99th% Overall-Agnostic, min_clade=10)", fontsize=13, fontweight="bold", pad=15)
    ax_bar.set_xticks([])
    ax_bar.set_xlim(-0.5, len(sizes) - 0.5)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_bar.annotate(f"{height}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
            
    # 2. Bottom-right: Matrix grid
    ax_matrix = plt.subplot(gs[1, 1], sharex=ax_bar)
    ax_matrix.set_yticks(range(len(set_names)))
    ax_matrix.set_yticklabels(set_names, fontsize=10, fontweight="bold")
    ax_matrix.set_xlim(-0.5, len(sizes) - 0.5)
    ax_matrix.set_ylim(-0.5, len(set_names) - 0.5)
    ax_matrix.grid(False)
    
    # Plot empty dots grid
    for x in range(len(sizes)):
        for y in range(len(set_names)):
            ax_matrix.plot(x, y, 'o', color='#eaeded', markersize=12)
            
    # Plot active dots and lines
    for x, (label, positions) in enumerate(sorted_intersections):
        active_indices = []
        for set_idx, name in enumerate(set_names):
            if name in label:
                active_indices.append(set_idx)
                
        if len(active_indices) > 1:
            ax_matrix.plot([x, x], [min(active_indices), max(active_indices)], color="#2c3e50", lw=2.5, zorder=1)
            
        for y in active_indices:
            color = EVENT_COLORS.get(set_names[y], "#2c3e50")
            ax_matrix.plot(x, y, 'o', color=color, markersize=14, zorder=2)
            
    ax_matrix.set_xticks(range(len(sizes)))
    ax_matrix.set_xticklabels([f"Int {i+1}" for i in range(len(sizes))], fontsize=9, fontweight="bold")
    ax_matrix.set_xlabel("Intersection Group", fontsize=11, fontweight="bold", labelpad=10)
    
    # 3. Bottom-left: Horizontal bar plot of total set sizes
    ax_totals = plt.subplot(gs[1, 0])
    set_totals = []
    for name in set_names:
        total = sum(len(pos) for label, pos in sorted_intersections if name in label)
        set_totals.append(total)
        
    y_pos = range(len(set_names))
    colors = [EVENT_COLORS.get(name, "#2c3e50") for name in set_names]
    ax_totals.barh(y_pos, set_totals, color=colors, height=0.5, edgecolor="#2c3e50", align='center')
    ax_totals.set_yticks(y_pos)
    ax_totals.set_yticklabels(set_names, fontsize=10, fontweight="bold")
    ax_totals.invert_xaxis()  # Reverse the bar direction
    ax_totals.set_xlabel("Total Residues with switches", fontsize=10, fontweight="bold")
    ax_totals.set_ylim(-0.5, len(set_names) - 0.5)
    
    for i, v in enumerate(set_totals):
        ax_totals.text(v + 3, i, str(v), ha="right", va="center", fontsize=9, fontweight="bold", color="white")
        
    plt.tight_layout()
    plt.savefig(f"{out_file_base}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{out_file_base}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved UpSet plot to {out_file_base}")


def main() -> None:
    scores_path = Path("results/badasp_scoring/raw_node_scores.csv")
    alignment_path = Path("data/interim/IPR019888_trimmed.aln")
    positional_csv_path = Path("results/badasp_scoring/clade_size_adjusted/min_clade_10/occupancy_80/event_positional_switches_clade_adjusted.csv")
    out_dir = Path("results/badasp_scoring/clade_size_adjusted/min_clade_10/occupancy_80")
    
    # 1. Compute flat vs adjusted comparison for min_clade=10 and min_clade=5
    print("Computing comparisons between flat and clade-adjusted thresholds...")
    df_10 = compute_flat_vs_adjusted(scores_path, alignment_path, min_clade=10, min_occ=0.8)
    df_5 = compute_flat_vs_adjusted(scores_path, alignment_path, min_clade=5, min_occ=0.8)
    
    # Combine and save comparison table
    df_comp_all = pd.concat([df_5, df_10], ignore_index=True)
    df_comp_all.to_csv(out_dir / "flat_vs_adjusted_comparison.csv", index=False)
    print(f"Saved comparison table to {out_dir / 'flat_vs_adjusted_comparison.csv'}")

    # Print comparison
    print("\n=== FLAT VS. CLADE-ADJUSTED 99th% SWITCH COUNT COMPARISON ===")
    print(df_comp_all.to_string(index=False))
    
    # 2. Build UpSet plot set membership
    print("\nBuilding set membership for UpSet plot...")
    df_pos = pd.read_csv(positional_csv_path)
    
    # Binary set mapping for each alignment position
    S_dup = set(df_pos[df_pos["duplication_switches_overall"] > 0]["position"])
    S_spec = set(df_pos[df_pos["speciation_switches_overall"] > 0]["position"])
    S_trans = set(df_pos[df_pos["transfer_switches_overall"] > 0]["position"])
    
    # Intersections
    all_positions = set(range(1, 170))
    
    membership_dict = {
        "Duplication": S_dup - (S_spec | S_trans),
        "Speciation": S_spec - (S_dup | S_trans),
        "Transfer": S_trans - (S_dup | S_spec),
        "Duplication & Speciation": (S_dup & S_spec) - S_trans,
        "Duplication & Transfer": (S_dup & S_trans) - S_spec,
        "Speciation & Transfer": (S_spec & S_trans) - S_dup,
        "Duplication & Speciation & Transfer": S_dup & S_spec & S_trans
    }
    
    # Print UpSet stats
    print("\n=== UPSET INTERSECTION SIZES (Number of residues with switches > 0) ===")
    for k, v in sorted(membership_dict.items(), key=lambda item: len(item[1]), reverse=True):
        print(f"{k}: {len(v)} residues")
        
    set_names = ["Duplication", "Speciation", "Transfer"]
    plot_custom_upset(membership_dict, set_names, out_dir / "upset_plot_clade_10")


if __name__ == "__main__":
    main()
