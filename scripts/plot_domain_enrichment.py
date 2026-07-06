"""Plot domain enrichment of switches for each evolutionary event type.

Calculates enrichment of switches within each architectural domain over the
baseline switch rate of the full, non-filtered sequence (169 positions) for
both percentile (95th%, 97th%, 99th%) and hard (1.7, 1.8, 1.9) thresholds.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def main() -> None:
    parser = argparse.ArgumentParser(description="BADASP Switch Domain Enrichment Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    args = parser.parse_args()

    # Create output directory
    out_dir = Path("results/badasp_scoring/domain_enrichment")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.scores.exists():
        print(f"Error: Scores file not found at {args.scores}")
        sys.exit(1)

    print("Loading raw scores and mapping alignment occupancies...")
    df = pd.read_csv(args.scores)
    
    # Calculate column occupancies
    from Bio import AlignIO
    alignment = AlignIO.read("data/interim/IPR019888_trimmed.aln", "fasta")
    num_seqs = len(alignment)
    aln_len = alignment.get_alignment_length()
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)

    df["occupancy"] = df["position"].map(occupancies)
    # Apply >=80% occupancy filter so low-occupancy residues are filtered out (switches = 0)
    df_filtered = df[df["occupancy"] >= 0.8].copy()
    df_filtered["max_score"] = df_filtered[["badasp_score_left", "badasp_score_right"]].max(axis=1)

    # Define domains (nominal coordinates on full 1-169 sequence)
    domains = {
        "HTH Scaffold": (6, 34),
        "Recognition Helix": (35, 50),
        "HTH Linker": (51, 67),
        "RAM Domain": (68, 152),
    }

    # Identify active positions overall (occupancy >= 80% and in core domains)
    active_positions_overall = []
    for name, (start, end) in domains.items():
        domain_positions = range(start, end + 1)
        active_positions_overall.extend([p for p in domain_positions if occupancies[p] >= 0.8])
    total_len = len(active_positions_overall)  # should be 139

    # Filter raw scores to include ONLY these active positions in core domains
    df_filtered = df_filtered[df_filtered["position"].isin(active_positions_overall)].copy()

    # Threshold configurations
    percentiles = [95, 97, 99]
    hard_thresholds = [1.7, 1.8, 1.9]

    # Compute percentiles on the pooled filtered max scores (all event types combined)
    all_scores = df_filtered["max_score"].dropna().values
    p_vals = {p: float(np.percentile(all_scores, p)) for p in percentiles}

    # Count pooled switches at each position (1 to 169) for each threshold
    df_pos = pd.DataFrame({"position": range(1, 170)})
    
    # Percentiles
    for p, val in p_vals.items():
        # Filter scores exceeding threshold
        sdf = df_filtered[df_filtered["max_score"] >= val]
        # Count switches per position
        pos_counts = sdf.groupby("position").size().reset_index(name=f"p{p}")
        df_pos = df_pos.merge(pos_counts, on="position", how="left").fillna(0)
        df_pos[f"p{p}"] = df_pos[f"p{p}"].astype(int)

    # Hard thresholds
    for h in hard_thresholds:
        sdf = df_filtered[df_filtered["max_score"] >= h]
        pos_counts = sdf.groupby("position").size().reset_index(name=f"h{h}")
        df_pos = df_pos.merge(pos_counts, on="position", how="left").fillna(0)
        df_pos[f"h{h}"] = df_pos[f"h{h}"].astype(int)

    # Store enrichment records
    enrichment_records = []

    # Calculate enrichment for each threshold and domain (using active positions only)
    for name, (start, end) in domains.items():
        domain_positions = list(range(start, end + 1))
        active_positions = [p for p in domain_positions if p in active_positions_overall]
        dom_len = len(active_positions)

        # Disregard domains that have no active residues (like N-terminus)
        if dom_len == 0:
            continue

        # Percentiles
        for p in percentiles:
            sum_sw = df_pos.loc[df_pos["position"].isin(active_positions), f"p{p}"].sum()
            tot_sw = df_pos.loc[df_pos["position"].isin(active_positions_overall), f"p{p}"].sum()
            
            dom_rate = sum_sw / dom_len
            base_rate = tot_sw / total_len
            enrichment = dom_rate / base_rate if base_rate > 0 else 0.0

            enrichment_records.append({
                "domain": name,
                "active_sites": dom_len,
                "threshold_type": "Percentile",
                "threshold_label": f"{p}th%",
                "threshold_val": p_vals[p],
                "domain_switches": sum_sw,
                "total_switches": tot_sw,
                "enrichment": enrichment
            })

        # Hard Thresholds
        for h in hard_thresholds:
            sum_sw = df_pos.loc[df_pos["position"].isin(active_positions), f"h{h}"].sum()
            tot_sw = df_pos.loc[df_pos["position"].isin(active_positions_overall), f"h{h}"].sum()

            dom_rate = sum_sw / dom_len
            base_rate = tot_sw / total_len
            enrichment = dom_rate / base_rate if base_rate > 0 else 0.0

            enrichment_records.append({
                "domain": name,
                "active_sites": dom_len,
                "threshold_type": "Hard Threshold",
                "threshold_label": f"Hard {h}",
                "threshold_val": h,
                "domain_switches": sum_sw,
                "total_switches": tot_sw,
                "enrichment": enrichment
            })

    df_enrich = pd.DataFrame(enrichment_records)
    csv_out = out_dir / "domain_enrichment_stats.csv"
    df_enrich.to_csv(csv_out, index=False)
    print(f"Saved statistics table to {csv_out}")

    # Set up plotting style
    sns.set_theme(style="whitegrid")
    
    # Unified comparison plot: all bars in one file
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # Custom color palette for the 6 thresholds
    threshold_palette = {
        "95th%": "#85c1e9",
        "97th%": "#3498db",
        "99th%": "#1f618d",
        "Hard 1.7": "#f1948a",
        "Hard 1.8": "#e74c3c",
        "Hard 1.9": "#922b21"
    }

    # Format the domain labels to show active sites
    df_enrich["domain_label"] = df_enrich.apply(
        lambda r: f"{r['domain']}\n({r['active_sites']} active sites)", axis=1
    )

    sns.barplot(
        data=df_enrich,
        x="domain_label",
        y="enrichment",
        hue="threshold_label",
        palette=threshold_palette,
        ax=ax
    )
    
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Baseline (1.0x)")
    ax.set_title("Pooled Switch Domain Enrichment Comparison (Occupancy >= 80% Active Residues Only)", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Fold Enrichment (Domain Switch Density / Baseline Switch Density)", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Domain (Excluding 100% Filtered-Out Domains)", fontsize=11, fontweight="semibold")
    ax.legend(title="Threshold Label", loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    # Annotate bars
    for patch in ax.patches:
        val = patch.get_height()
        if val > 0:
            ax.annotate(
                f"{val:.2f}x",
                xy=(patch.get_x() + patch.get_width() / 2, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2c3e50"
            )

    plt.tight_layout()
    out_svg = out_dir / "domain_enrichment_comparison.svg"
    out_png = out_dir / "domain_enrichment_comparison.png"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_svg}")

if __name__ == "__main__":
    main()
