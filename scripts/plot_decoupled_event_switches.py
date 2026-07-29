"""Analyze and plot BADASP evolutionary switches decoupled by event type (Duplication, Speciation, Transfer).

Calculates event-specific percentile thresholds and hard threshold switch counts.
Generates a comparative 4-panel figure:
- Panel A: Duplication positional switches.
- Panel B: Speciation positional switches.
- Panel C: Transfer positional switches.
- Panel D: Domain switch density comparison bar plot.
Saves outputs to results/badasp_scoring/event_decoupling/occupancy_XX/.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="BADASP Decoupled Event-Type Switch Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8, help="Occupancy threshold used for filtering (0.0 to 1.0)")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory for results")
    args = parser.parse_args()

    occ_pct = int(args.min_occupancy * 100)
    
    # Setup Output Directory
    if args.outdir is not None:
        out_dir = args.outdir
    else:
        out_dir = Path(f"results/badasp_scoring/event_decoupling/occupancy_{occ_pct}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    if not args.scores.exists():
        print(f"Error: Raw scores file not found at {args.scores}")
        sys.exit(1)
        
    print("Loading raw scores and calculating occupancies...")
    df = pd.read_csv(args.scores)
    df["max_score"] = df[["badasp_score_left", "badasp_score_right"]].max(axis=1)
    
    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    
    filtered_positions = [pos for pos, occ in occupancies.items() if occ < args.min_occupancy]
    filtered_regions = find_contiguous_regions(filtered_positions)
    
    # Filter raw scores
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()
    num_active_positions = df_filtered["position"].nunique()
    
    # 2. Process each Event Type separately
    event_types = ["Duplication", "Speciation", "Transfer"]
    stats_rows = []
    
    # Position matrix to store all raw switch tallies (1 to 169)
    df_pos = pd.DataFrame({"position": range(1, 170)})
    
    # Architectural Domains
    domains = {
        "N-terminus": (1, 5),
        "HTH Scaffold": (6, 34),
        "Recognition Helix": (35, 50),
        "HTH Linker": (51, 67),
        "RAM Domain": (68, 152),
        "C-terminus": (153, 169)
    }
    
    domain_rows = []
    
    for event_type in event_types:
        df_event = df_filtered[df_filtered["event_type"] == event_type].copy()
        total_comp = len(df_event)
        
        if total_comp == 0:
            print(f"Warning: No scored records found for event type: {event_type}")
            for label in ["p95", "p97", "p99", "h17", "h18", "h19"]:
                df_pos[f"{event_type.lower()}_switches_{label}"] = 0
            continue
            
        # Calculate event-specific percentiles
        p95 = float(np.percentile(df_event["max_score"].dropna(), 95)) if len(df_event) > 0 else np.nan
        p97 = float(np.percentile(df_event["max_score"].dropna(), 97)) if len(df_event) > 0 else np.nan
        p99 = float(np.percentile(df_event["max_score"].dropna(), 99)) if len(df_event) > 0 else np.nan
        
        # Count switches
        sw_p95 = len(df_event[df_event["max_score"] >= p95])
        sw_p97 = len(df_event[df_event["max_score"] >= p97])
        sw_p99 = len(df_event[df_event["max_score"] >= p99])
        sw_h17 = len(df_event[df_event["max_score"] >= 1.7])
        sw_h18 = len(df_event[df_event["max_score"] >= 1.8])
        sw_h19 = len(df_event[df_event["max_score"] >= 1.9])
        
        stats_rows.append({
            "event_type": event_type,
            "total_comparisons": total_comp,
            "p95_threshold": p95,
            "p95_switches": sw_p95,
            "p97_threshold": p97,
            "p97_switches": sw_p97,
            "p99_threshold": p99,
            "p99_switches": sw_p99,
            "hard1.7_switches": sw_h17,
            "hard1.8_switches": sw_h18,
            "hard1.9_switches": sw_h19
        })
        
        print(f"\n{event_type} Switch Stats:")
        print(f"  Total comparisons: {total_comp}")
        print(f"  97th% threshold: {p97:.6f} ({sw_p97} switches)")
        print(f"  Hard 1.7 switches: {sw_h17}")
        
        # Positional Switch Tallies
        # Create columns in positional dataframe
        for label, val in [("p95", p95), ("p97", p97), ("p99", p99), ("h17", 1.7), ("h18", 1.8), ("h19", 1.9)]:
            sdf = df_event[df_event["max_score"] >= val]
            pos_counts = sdf.groupby("position").size().reset_index(name=f"{event_type.lower()}_switches_{label}")
            df_pos = df_pos.merge(pos_counts, on="position", how="left").fillna(0)
            df_pos[f"{event_type.lower()}_switches_{label}"] = df_pos[f"{event_type.lower()}_switches_{label}"].astype(int)
            
        # Domain switch densities
        for name, (start, end) in domains.items():
            domain_positions = list(range(start, end + 1))
            active_positions = [p for p in domain_positions if p not in filtered_positions]
            active_count = len(active_positions)
            total_count = len(domain_positions)
            
            mask = (df_pos["position"] >= start) & (df_pos["position"] <= end)
            
            sum_p95 = df_pos.loc[mask, f"{event_type.lower()}_switches_p95"].sum()
            sum_p97 = df_pos.loc[mask, f"{event_type.lower()}_switches_p97"].sum()
            sum_h17 = df_pos.loc[mask, f"{event_type.lower()}_switches_h17"].sum()
            sum_h19 = df_pos.loc[mask, f"{event_type.lower()}_switches_h19"].sum()
            
            domain_rows.append({
                "event_type": event_type,
                "domain": name,
                "residues": f"{start}-{end}",
                "total_sites": total_count,
                "active_sites": active_count,
                "p95_switches": sum_p95,
                "p95_density": sum_p95 / total_count if total_count > 0 else 0.0,
                "p97_switches": sum_p97,
                "p97_density": sum_p97 / total_count if total_count > 0 else 0.0,
                "hard1.7_switches": sum_h17,
                "hard1.7_density": sum_h17 / total_count if total_count > 0 else 0.0,
                "hard1.9_switches": sum_h19,
                "hard1.9_density": sum_h19 / total_count if total_count > 0 else 0.0
            })
            
    df_stats = pd.DataFrame(stats_rows)
    stats_csv = out_dir / "event_decoupled_stats.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"\nSaved event decoupled statistics to {stats_csv}")
    
    df_pos.to_csv(out_dir / "event_positional_switches.csv", index=False)
    print(f"Saved event positional switches to {out_dir / 'event_positional_switches.csv'}")
    
    df_domains = pd.DataFrame(domain_rows)
    df_domains.to_csv(out_dir / "event_domain_densities.csv", index=False)
    print(f"Saved event domain densities to {out_dir / 'event_domain_densities.csv'}")
    
    # 3. Generate 4-Panel Comparative Plot
    print("\nGenerating 4-panel comparative figure...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(4, 1, figsize=(18, 24), sharey=False)
    
    # Styling configurations for panels A, B, C
    def _get_stat(event_type, col):
        mask = df_stats["event_type"] == event_type
        return float(df_stats.loc[mask, col].values[0]) if mask.any() else 0.0

    panel_cfgs = {
        "Duplication": {
            "title": "A. Duplication-Arising Switches (Specific 97th% = {thresh:.3f})",
            "ax": axes[0],
            "color": "#c0392b",
            "raw_color": "#e74c3c",
            "col_p97": "duplication_switches_p97",
            "col_h17": "duplication_switches_h17",
            "thresh_p97": _get_stat("Duplication", "p97_threshold"),
        },
        "Speciation": {
            "title": "B. Speciation-Arising Switches (Specific 97th% = {thresh:.3f})",
            "ax": axes[1],
            "color": "#2980b9",
            "raw_color": "#3498db",
            "col_p97": "speciation_switches_p97",
            "col_h17": "speciation_switches_h17",
            "thresh_p97": _get_stat("Speciation", "p97_threshold"),
        },
        "Transfer": {
            "title": "C. Transfer-Arising Switches (Specific 97th% = {thresh:.3f})",
            "ax": axes[2],
            "color": "#27ae60",
            "raw_color": "#2ecc71",
            "col_p97": "transfer_switches_p97",
            "col_h17": "transfer_switches_h17",
            "thresh_p97": _get_stat("Transfer", "p97_threshold"),
        },
    }

    # Common vertical shading colors for domains
    domain_colors = ["#ecf0f1", "#d5dbdb", "#a2d9ce", "#abebc6", "#fadbd8", "#fdebd0"]
    
    for ev_type, cfg in panel_cfgs.items():
        ax = cfg["ax"]
        col_p97 = cfg["col_p97"]
        col_h17 = cfg["col_h17"]
        
        # Plot Hard 1.7 baseline as grey dashed lollipops
        ax.vlines(df_pos["position"], 0, df_pos[col_h17], color="#7f8c8d", alpha=0.3, linewidth=1.0, linestyle=":")
        ax.scatter(df_pos["position"], df_pos[col_h17], color="none", edgecolor="#7f8c8d", s=15, alpha=0.5, label="Hard 1.7 Threshold")
        
        # Plot Event-Specific 97th% as solid colored lollipops
        ax.vlines(df_pos["position"], 0, df_pos[col_p97], color=cfg["color"], alpha=0.7, linewidth=1.5)
        ax.scatter(df_pos["position"], df_pos[col_p97], color=cfg["color"], s=25, zorder=3, label=f"Event-Specific 97th% (Thresh: {cfg['thresh_p97']:.3f})")

        # Annotate top 5 peak positions with switches in this panel (specific 97th% first)
        top_peaks = df_pos[df_pos[col_p97] > 0].sort_values(col_p97, ascending=False).head(5)
        for _, row in top_peaks.iterrows():
            pos = int(row["position"])
            val = int(row[col_p97])
            ax.text(
                pos, val + 0.3,
                f"Col {pos}\n({val})",
                ha="center", va="bottom",
                fontsize=8, color="#2c3e50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec=cfg["color"], lw=0.5, alpha=0.85),
                zorder=5
            )
        
        # Shade domains
        for i, (name, (start, end)) in enumerate(domains.items()):
            ax.axvspan(start, end, color=domain_colors[i % len(domain_colors)], alpha=0.15)
            midpoint = (start + end) / 2
            ax.text(midpoint, ax.get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
            
        # Shade filtered regions
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
            
        ax.set_title(cfg["title"].format(thresh=cfg["thresh_p97"]), fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Protein Alignment Position", fontsize=11)
        ax.set_ylabel("Switch Count", fontsize=11)
        ax.set_xlim(0.5, 169.5)
        ax.set_ylim(0, max(df_pos[col_p97].max(), df_pos[col_h17].max(), 1) * 1.25)
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        
    # Panel D: Domain Switch Density Comparison Bar Plot (using specific 97th% thresholds)
    # Prepare labels and data
    x_labels = []
    for name in domains.keys():
        dup_rows = df_domains[(df_domains["domain"] == name) & (df_domains["event_type"] == "Duplication")]
        if dup_rows.empty:
            x_labels.append(f"{name}\n(N/A)")
        else:
            row_dup = dup_rows.iloc[0]
            x_labels.append(f"{name}\n({row_dup['active_sites']}/{row_dup['total_sites']} sites active)")
        
    sns.barplot(
        data=df_domains,
        x="domain",
        y="p97_density",
        hue="event_type",
        palette=["#e74c3c", "#3498db", "#2ecc71"], # Red, Blue, Green
        ax=axes[3]
    )
    
    axes[3].set_xticks(range(len(domains)))
    axes[3].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    axes[3].set_title("D. Decoupled Switch Density across Domains (Specific 97th% Cutoffs, Occupancy >= 80%)", fontsize=13, fontweight="bold", pad=15)
    axes[3].set_xlabel("Domain (Active / Total nominal residues)", fontsize=11)
    axes[3].set_ylabel("Switch Density (Events / Site)", fontsize=11)
    axes[3].legend(title="Evolutionary Event Type", frameon=True, fontsize=10.5)
    
    # Add values on top of bars
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
            
    # Add Filtered Out indicator
    for i, name in enumerate(domains.keys()):
        dup_rows = df_domains[(df_domains["domain"] == name) & (df_domains["event_type"] == "Duplication")]
        if not dup_rows.empty and dup_rows.iloc[0]["active_sites"] == 0:
            axes[3].text(
                i, 0.02,
                "FILTERED",
                ha="center", va="bottom",
                fontsize=9, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )

    plt.tight_layout()

    plot_svg = out_dir / "decoupled_event_switches_comparison.svg"
    plot_png = out_dir / "decoupled_event_switches_comparison.png"
    
    fig.savefig(str(plot_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nGenerated decoupled comparative plots successfully:")
    print(f"  SVG: {plot_svg}")
    print(f"  PNG: {plot_png}")

    # 4. Generate 4-Panel Comparative Plot (Hard 1.7 Only)
    print("\nGenerating 4-panel comparative figure (Hard 1.7 only)...")
    fig, axes = plt.subplots(4, 1, figsize=(18, 24), sharey=False)
    
    panel_cfgs_h17 = {
        "Duplication": {
            "title": "A. Duplication-Arising Switches (Hard 1.7 Threshold)",
            "ax": axes[0],
            "color": "#c0392b",   # Dark Red
            "raw_color": "#e74c3c", # Light Red
            "col_h17": "duplication_switches_h17"
        },
        "Speciation": {
            "title": "B. Speciation-Arising Switches (Hard 1.7 Threshold)",
            "ax": axes[1],
            "color": "#2980b9",   # Dark Blue
            "raw_color": "#3498db", # Light Blue
            "col_h17": "speciation_switches_h17"
        },
        "Transfer": {
            "title": "C. Transfer-Arising Switches (Hard 1.7 Threshold)",
            "ax": axes[2],
            "color": "#27ae60",   # Dark Green
            "raw_color": "#2ecc71", # Light Green
            "col_h17": "transfer_switches_h17"
        }
    }
    
    for ev_type, cfg in panel_cfgs_h17.items():
        ax = cfg["ax"]
        col_h17 = cfg["col_h17"]
        
        # Plot Hard 1.7 as solid colored lollipops
        ax.vlines(df_pos["position"], 0, df_pos[col_h17], color=cfg["color"], alpha=0.7, linewidth=1.5)
        ax.scatter(df_pos["position"], df_pos[col_h17], color=cfg["color"], s=25, zorder=3, label="Hard 1.7 Threshold")

        # Annotate top 5 peak positions with switches in this panel
        top_peaks = df_pos[df_pos[col_h17] > 0].sort_values(col_h17, ascending=False).head(5)
        for _, row in top_peaks.iterrows():
            pos = int(row["position"])
            val = int(row[col_h17])
            ax.text(
                pos, val + 0.3,
                f"Col {pos}\n({val})",
                ha="center", va="bottom",
                fontsize=8, color="#2c3e50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec=cfg["color"], lw=0.5, alpha=0.85),
                zorder=5
            )
        
        # Shade domains
        for i, (name, (start, end)) in enumerate(domains.items()):
            ax.axvspan(start, end, color=domain_colors[i % len(domain_colors)], alpha=0.15)
            midpoint = (start + end) / 2
            ax.text(midpoint, ax.get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
            
        # Shade filtered regions
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
            
        ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Protein Alignment Position", fontsize=11)
        ax.set_ylabel("Switch Count", fontsize=11)
        ax.set_xlim(0.5, 169.5)
        ax.set_ylim(0, max(df_pos[col_h17].max(), 1) * 1.25)
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        
    # Panel D: Domain Switch Density Comparison Bar Plot (using Hard 1.7 threshold only)
    sns.barplot(
        data=df_domains,
        x="domain",
        y="hard1.7_density",
        hue="event_type",
        palette=["#e74c3c", "#3498db", "#2ecc71"], # Red, Blue, Green
        ax=axes[3]
    )
    
    axes[3].set_xticks(range(len(domains)))
    axes[3].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    axes[3].set_title("D. Decoupled Switch Density across Domains (Hard 1.7 Cutoff, Occupancy >= 80%)", fontsize=13, fontweight="bold", pad=15)
    axes[3].set_xlabel("Domain (Active / Total nominal residues)", fontsize=11)
    axes[3].set_ylabel("Switch Density (Events / Site)", fontsize=11)
    axes[3].legend(title="Evolutionary Event Type", frameon=True, fontsize=10.5)
    
    # Add values on top of bars
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
            
    # Add Filtered Out indicator
    for i, name in enumerate(domains.keys()):
        dup_rows = df_domains[(df_domains["domain"] == name) & (df_domains["event_type"] == "Duplication")]
        if not dup_rows.empty and dup_rows.iloc[0]["active_sites"] == 0:
            axes[3].text(
                i, 0.02,
                "FILTERED",
                ha="center", va="bottom",
                fontsize=9, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )

    plt.tight_layout()

    plot_h17_svg = out_dir / "decoupled_event_switches_hard1.7_comparison.svg"
    plot_h17_png = out_dir / "decoupled_event_switches_hard1.7_comparison.png"
    
    fig.savefig(str(plot_h17_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_h17_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nGenerated decoupled comparative plots (Hard 1.7 only) successfully:")
    print(f"  SVG: {plot_h17_svg}")
    print(f"  PNG: {plot_h17_png}")

    # 5. Generate 4-Panel Comparative Plot (Hard 1.9 Only)
    print("\nGenerating 4-panel comparative figure (Hard 1.9 only)...")
    fig, axes = plt.subplots(4, 1, figsize=(18, 24), sharey=False)
    
    panel_cfgs_h19 = {
        "Duplication": {
            "title": "A. Duplication-Arising Switches (Hard 1.9 Threshold)",
            "ax": axes[0],
            "color": "#c0392b",   # Dark Red
            "raw_color": "#e74c3c", # Light Red
            "col_h19": "duplication_switches_h19"
        },
        "Speciation": {
            "title": "B. Speciation-Arising Switches (Hard 1.9 Threshold)",
            "ax": axes[1],
            "color": "#2980b9",   # Dark Blue
            "raw_color": "#3498db", # Light Blue
            "col_h19": "speciation_switches_h19"
        },
        "Transfer": {
            "title": "C. Transfer-Arising Switches (Hard 1.9 Threshold)",
            "ax": axes[2],
            "color": "#27ae60",   # Dark Green
            "raw_color": "#2ecc71", # Light Green
            "col_h19": "transfer_switches_h19"
        }
    }
    
    for ev_type, cfg in panel_cfgs_h19.items():
        ax = cfg["ax"]
        col_h19 = cfg["col_h19"]
        
        # Plot Hard 1.9 as solid colored lollipops
        ax.vlines(df_pos["position"], 0, df_pos[col_h19], color=cfg["color"], alpha=0.7, linewidth=1.5)
        ax.scatter(df_pos["position"], df_pos[col_h19], color=cfg["color"], s=25, zorder=3, label="Hard 1.9 Threshold")

        # Annotate top 5 peak positions with switches in this panel
        top_peaks = df_pos[df_pos[col_h19] > 0].sort_values(col_h19, ascending=False).head(5)
        for _, row in top_peaks.iterrows():
            pos = int(row["position"])
            val = int(row[col_h19])
            ax.text(
                pos, val + 0.3,
                f"Col {pos}\n({val})",
                ha="center", va="bottom",
                fontsize=8, color="#2c3e50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec=cfg["color"], lw=0.5, alpha=0.85),
                zorder=5
            )
        
        # Shade domains
        for i, (name, (start, end)) in enumerate(domains.items()):
            ax.axvspan(start, end, color=domain_colors[i % len(domain_colors)], alpha=0.15)
            midpoint = (start + end) / 2
            ax.text(midpoint, ax.get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
            
        # Shade filtered regions
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
            
        ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Protein Alignment Position", fontsize=11)
        ax.set_ylabel("Switch Count", fontsize=11)
        ax.set_xlim(0.5, 169.5)
        ax.set_ylim(0, max(df_pos[col_h19].max(), 1) * 1.25)
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        
    # Panel D: Domain Switch Density Comparison Bar Plot (using Hard 1.9 threshold only)
    sns.barplot(
        data=df_domains,
        x="domain",
        y="hard1.9_density",
        hue="event_type",
        palette=["#e74c3c", "#3498db", "#2ecc71"], # Red, Blue, Green
        ax=axes[3]
    )
    
    axes[3].set_xticks(range(len(domains)))
    axes[3].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    axes[3].set_title("D. Decoupled Switch Density across Domains (Hard 1.9 Cutoff, Occupancy >= 80%)", fontsize=13, fontweight="bold", pad=15)
    axes[3].set_xlabel("Domain (Active / Total nominal residues)", fontsize=11)
    axes[3].set_ylabel("Switch Density (Events / Site)", fontsize=11)
    axes[3].legend(title="Evolutionary Event Type", frameon=True, fontsize=10.5)
    
    # Add values on top of bars
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
            
    # Add Filtered Out indicator
    for i, name in enumerate(domains.keys()):
        dup_rows = df_domains[(df_domains["domain"] == name) & (df_domains["event_type"] == "Duplication")]
        if not dup_rows.empty and dup_rows.iloc[0]["active_sites"] == 0:
            axes[3].text(
                i, 0.02,
                "FILTERED",
                ha="center", va="bottom",
                fontsize=9, color="#7f8c8d", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#bdc3c7", lw=1)
            )

    plt.tight_layout()

    plot_h19_svg = out_dir / "decoupled_event_switches_hard1.9_comparison.svg"
    plot_h19_png = out_dir / "decoupled_event_switches_hard1.9_comparison.png"
    
    fig.savefig(str(plot_h19_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_h19_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nGenerated decoupled comparative plots (Hard 1.9 only) successfully:")
    print(f"  SVG: {plot_h19_svg}")
    print(f"  PNG: {plot_h19_png}")

    # 6. Generate 4-Panel Comparative Plot (Dynamic 95th Percentile Only)
    print("\nGenerating 4-panel comparative figure (Dynamic 95th percentile only)...")
    fig, axes = plt.subplots(4, 1, figsize=(18, 24), sharey=False)
    
    panel_cfgs_p95 = {
        "Duplication": {
            "title": "A. Duplication-Arising Switches (Dynamic 95th% Threshold = {thresh:.3f})",
            "ax": axes[0],
            "color": "#c0392b",   # Dark Red
            "raw_color": "#e74c3c", # Light Red
            "col_p95": "duplication_switches_p95",
            "thresh_p95": df_stats.loc[df_stats["event_type"] == "Duplication", "p95_threshold"].values[0] if len(df_stats) > 0 else 0.0
        },
        "Speciation": {
            "title": "B. Speciation-Arising Switches (Dynamic 95th% Threshold = {thresh:.3f})",
            "ax": axes[1],
            "color": "#2980b9",   # Dark Blue
            "raw_color": "#3498db", # Light Blue
            "col_p95": "speciation_switches_p95",
            "thresh_p95": df_stats.loc[df_stats["event_type"] == "Speciation", "p95_threshold"].values[0] if len(df_stats) > 0 else 0.0
        },
        "Transfer": {
            "title": "C. Transfer-Arising Switches (Dynamic 95th% Threshold = {thresh:.3f})",
            "ax": axes[2],
            "color": "#27ae60",   # Dark Green
            "raw_color": "#2ecc71", # Light Green
            "col_p95": "transfer_switches_p95",
            "thresh_p95": df_stats.loc[df_stats["event_type"] == "Transfer", "p95_threshold"].values[0] if len(df_stats) > 0 else 0.0
        }
    }
    
    for ev_type, cfg in panel_cfgs_p95.items():
        ax = cfg["ax"]
        col_p95 = cfg["col_p95"]
        
        # Plot Dynamic 95th% as solid colored lollipops
        ax.vlines(df_pos["position"], 0, df_pos[col_p95], color=cfg["color"], alpha=0.7, linewidth=1.5)
        ax.scatter(df_pos["position"], df_pos[col_p95], color=cfg["color"], s=25, zorder=3, label=f"Dynamic 95th% Threshold (Thresh: {cfg['thresh_p95']:.3f})")

        # Annotate top 5 peak positions with switches in this panel
        top_peaks = df_pos[df_pos[col_p95] > 0].sort_values(col_p95, ascending=False).head(5)
        for _, row in top_peaks.iterrows():
            pos = int(row["position"])
            val = int(row[col_p95])
            ax.text(
                pos, val + 0.3,
                f"Col {pos}\n({val})",
                ha="center", va="bottom",
                fontsize=8, color="#2c3e50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="#fdfefe", ec=cfg["color"], lw=0.5, alpha=0.85),
                zorder=5
            )
        
        # Shade domains
        for i, (name, (start, end)) in enumerate(domains.items()):
            ax.axvspan(start, end, color=domain_colors[i % len(domain_colors)], alpha=0.15)
            midpoint = (start + end) / 2
            ax.text(midpoint, ax.get_ylim()[1] * 0.93, name, ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")
            
        # Shade filtered regions
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
            
        ax.set_title(cfg["title"].format(thresh=cfg["thresh_p95"]), fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Protein Alignment Position", fontsize=11)
        ax.set_ylabel("Switch Count", fontsize=11)
        ax.set_xlim(0.5, 169.5)
        ax.set_ylim(0, max(df_pos[col_p95].max(), 1) * 1.25)
        ax.legend(frameon=True, fontsize=9.5, loc="upper right")
        
    # Panel D: Domain Switch Density Comparison Bar Plot (using Dynamic 95th% threshold)
    sns.barplot(
        data=df_domains,
        x="domain",
        y="p95_density",
        hue="event_type",
        palette=["#e74c3c", "#3498db", "#2ecc71"], # Red, Blue, Green
        ax=axes[3]
    )
    
    axes[3].set_xticks(range(len(domains)))
    axes[3].set_xticklabels(x_labels, fontsize=10, fontweight="semibold")
    axes[3].set_title("D. Decoupled Switch Density across Domains (Dynamic 95th% Cutoff, Occupancy >= 80%)", fontsize=13, fontweight="bold", pad=15)
    axes[3].set_xlabel("Domain (Active / Total nominal residues)", fontsize=11)
    axes[3].set_ylabel("Switch Density (Events / Site)", fontsize=11)
    axes[3].legend(title="Evolutionary Event Type", frameon=True, fontsize=10.5)
    
    # Add values on top of bars
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
            
    # Add Filtered Out indicator
    for i, name in enumerate(domains.keys()):
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
    
    plot_p95_svg = out_dir / "decoupled_event_switches_dynamic_p95_comparison.svg"
    plot_p95_png = out_dir / "decoupled_event_switches_dynamic_p95_comparison.png"
    
    fig.savefig(str(plot_p95_svg), format="svg", bbox_inches="tight")
    fig.savefig(str(plot_p95_png), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nGenerated decoupled comparative plots (Dynamic 95th% only) successfully:")
    print(f"  SVG: {plot_p95_svg}")
    print(f"  PNG: {plot_p95_png}")
    print("\nDecoupled event switch analysis completed successfully.")


if __name__ == "__main__":
    main()
