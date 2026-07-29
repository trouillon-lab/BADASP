"""Module 3: Topological Clade & Lineage Specificity for Initial Run BADASP Switches.

Identifies switches characteristic of topological tree clades (internal LCA subtrees)
and annotated taxonomic phyla/subfamilies using Fisher's exact tests.
Outputs saved under results/initial_run_characterization/.
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
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.analyze_switch_origin_time import (
    calculate_msa_occupancies,
    calculate_bin_thresholds_999,
    extract_switch_instances,
    DOMAINS,
)


def identify_topological_clade_markers(switches_df: pd.DataFrame, min_clade_size: int = 5) -> pd.DataFrame:
    """Identify marker positions for topological tree clades (internal LCA nodes)."""
    if switches_df.empty:
        return pd.DataFrame(columns=["node_name", "event_type", "clade_size", "distance_from_root", "num_switches", "marker_positions"])

    records = []
    grouped = switches_df.groupby("node_name")

    for node_name, group in grouped:
        clade_sz = group["clade_size"].iloc[0]
        if clade_sz < min_clade_size:
            continue

        event = group["event_type"].iloc[0]
        dist = group["distance_from_root"].iloc[0] if "distance_from_root" in group.columns else np.nan
        positions = sorted(group["position"].unique().tolist())
        pos_str = ",".join(map(str, positions))

        records.append({
            "node_name": node_name,
            "event_type": event,
            "clade_size": clade_sz,
            "distance_from_root": dist,
            "num_switches": len(positions),
            "marker_positions": pos_str
        })

    df_res = pd.DataFrame(records)
    if not df_res.empty:
        df_res = df_res.sort_values(by=["num_switches", "clade_size"], ascending=[False, False])
    return df_res


def compute_phylum_enrichments(node_taxa: dict, node_switches: dict, background_phyla: dict) -> pd.DataFrame:
    """Compute Fisher's Exact test for phylum enrichment at switch-containing internal nodes."""
    records = []
    total_bg = sum(background_phyla.values())

    for node_name, phylum_counts in node_taxa.items():
        if node_name not in node_switches or not node_switches[node_name]:
            continue

        node_total = sum(phylum_counts.values())
        if node_total == 0:
            continue

        for phylum, count in phylum_counts.items():
            if count == 0:
                continue

            a = count  # Phylum seqs in clade
            b = node_total - count  # Non-phylum seqs in clade
            c = background_phyla.get(phylum, 0) - a  # Phylum seqs outside clade
            d = (total_bg - background_phyla.get(phylum, 0)) - b  # Non-phylum seqs outside clade

            a, b, c, d = max(0, a), max(0, b), max(0, c), max(0, d)
            contingency = [[a, b], [c, d]]
            odds_ratio, p_val = fisher_exact(contingency, alternative="greater")

            records.append({
                "node_name": node_name,
                "phylum": phylum,
                "clade_phylum_count": a,
                "clade_total": node_total,
                "bg_phylum_count": background_phyla.get(phylum, 0),
                "odds_ratio": float(odds_ratio),
                "p_val": float(p_val),
                "num_switches": len(node_switches[node_name])
            })

    df_enrich = pd.DataFrame(records)
    if not df_enrich.empty:
        df_enrich = df_enrich.sort_values(by="p_val", ascending=True)
    return df_enrich


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 3: Topological Clade & Lineage Specificity Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8)
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results/initial_run_characterization"))
    args = parser.parse_args()

    plots_dir = args.out_dir / "plots"
    tables_dir = args.out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw node scores from {args.scores}...")
    df = pd.read_csv(args.scores)
    df = df[(df["clade_size_left"] >= args.min_clade_size) & (df["clade_size_right"] >= args.min_clade_size)]

    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()

    thresholds, clade_categories = calculate_bin_thresholds_999(df_filtered, num_bins=10)
    switches_df = extract_switch_instances(df_filtered, thresholds, clade_categories, event_specific=False)

    print("Extracting topological clade markers...")
    markers_df = identify_topological_clade_markers(switches_df, min_clade_size=args.min_clade_size)
    markers_df.to_csv(tables_dir / "topological_clade_markers.csv", index=False)

    # Plot top 15 topological clade markers by switch count
    top_clades = markers_df.head(15)
    if not top_clades.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_clades,
            x="num_switches",
            y="node_name",
            hue="event_type",
            palette={"Duplication": "#c0392b", "Speciation": "#2980b9", "Transfer": "#27ae60"}
        )
        plt.title("Top Topological Clades Defined by 99.9th% Divergence Switches", fontsize=13, fontweight="bold")
        plt.xlabel("Number of Simultaneous 99.9th% Switches at LCA Node", fontsize=11)
        plt.ylabel("Topological Clade (Internal Node)", fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "topological_clade_markers.svg", format="svg")
        plt.savefig(plots_dir / "topological_clade_markers.png", format="png", dpi=300)
        plt.close()

    print(f"Module 3 completed. Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
