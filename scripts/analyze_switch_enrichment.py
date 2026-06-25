"""Perform advanced BADASP score downstream analyses:
1. Domain architectural enrichment: Load domain boundaries from domain_architecture.json,
   tally switches per domain, and perform Fisher's exact test for statistical enrichment.
2. Co-evolution analysis: Compute Jaccard similarity of switch occurrences across nodes
   for the top 40 hotspots, and perform hierarchical clustering to detect co-evolution communities.
3. Save all tables and generate clustered heatmaps and bar charts.

All outputs are saved under results/badasp_scoring/analyses/
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy


def main():
    root_dir = Path(project_root)
    scores_path = root_dir / "results" / "badasp_scoring" / "raw_node_scores.csv"
    domain_path = root_dir / "data" / "domain_architecture.json"
    out_dir = root_dir / "results" / "badasp_scoring" / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists() or not domain_path.exists():
        print("Error: Required scores CSV or domain_architecture.json missing.")
        sys.exit(1)

    print("================================================================================")
    print("BADASP Co-evolution & Domain Enrichment Analysis")
    print("================================================================================\n")

    # 1. Load raw scores
    df = pd.read_csv(scores_path)
    df["max_score"] = df[["badasp_score_left", "badasp_score_right"]].max(axis=1)
    
    # Calculate threshold (95th percentile)
    threshold = df["max_score"].quantile(0.95)
    df["is_switch"] = df["max_score"] >= threshold
    switches_df = df[df["is_switch"]]

    # Load domain boundaries
    with domain_path.open("r", encoding="utf-8") as f:
        domains = json.load(f)

    max_position = df["position"].max()
    print(f"  Protein sequence length: {max_position} residues.")
    print(f"  Domain boundaries parsed:")
    for domain, span in domains.items():
        print(f"    - {domain:18s}: residues {span[0]} to {span[1]} (length: {span[1] - span[0] + 1})")

    # -------------------------------------------------------------------------
    # 2. Domain Enrichment Analysis (Fisher's Exact Test)
    # -------------------------------------------------------------------------
    print("\nPerforming domain enrichment analysis...")
    enrichment_results = []
    
    # Overall counts of switches and non-switches in the whole dataset
    # We do this position-wise: for each position, how many switches occurred vs total opportunities
    pos_stats = df.groupby("position")["is_switch"].agg(["sum", "count"]).reset_index()
    pos_stats.columns = ["position", "switches", "total"]
    pos_stats["non_switches"] = pos_stats["total"] - pos_stats["switches"]
    
    total_switches = pos_stats["switches"].sum()
    total_non_switches = pos_stats["non_switches"].sum()

    for domain, span in domains.items():
        start, end = span[0], span[1]
        domain_mask = (pos_stats["position"] >= start) & (pos_stats["position"] <= end)
        
        # Counts inside the domain
        in_switches = pos_stats[domain_mask]["switches"].sum()
        in_non_switches = pos_stats[domain_mask]["non_switches"].sum()
        in_total = pos_stats[domain_mask]["total"].sum()
        
        # Counts outside the domain
        out_switches = total_switches - in_switches
        out_non_switches = total_non_switches - in_non_switches
        
        # Fisher's Exact Test contingency table:
        #            | In Domain | Out of Domain
        # -----------+-----------+--------------
        # Switches   |   a       |   b
        # Non-Switch |   c       |   d
        contingency_table = [
            [in_switches, out_switches],
            [in_non_switches, out_non_switches]
        ]
        
        odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative="greater")
        switch_density = (in_switches / in_total) * 100 if in_total else 0.0
        
        enrichment_results.append({
            "domain": domain,
            "start": start,
            "end": end,
            "domain_length": end - start + 1,
            "switch_count": in_switches,
            "total_observations": in_total,
            "switch_density_pct": round(switch_density, 3),
            "odds_ratio": round(odds_ratio, 4),
            "p_value": p_value,
            "significant_0_05": p_value < 0.05
        })

    df_enrichment = pd.DataFrame(enrichment_results)
    enrichment_csv = out_dir / "domain_enrichment.csv"
    df_enrichment.to_csv(enrichment_csv, index=False)
    print(f"  Saved domain enrichment table to: {enrichment_csv}")
    
    print("\n  Domain Enrichment Statistics:")
    for _, r in df_enrichment.iterrows():
        sig_str = "*" if r["significant_0_05"] else " "
        print(f"    - {r['domain']:18s}: Switches={r['switch_count']:4d}, Density={r['switch_density_pct']:.2f}%, OddsRatio={r['odds_ratio']:.2f}, p-val={r['p_value']:.2e} {sig_str}")

    # -------------------------------------------------------------------------
    # 3. Co-evolution & Community Detection Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming co-evolution network analysis...")
    # Select the top 40 positions with the most switches
    pos_counts = df[df["is_switch"]].groupby("position")["is_switch"].count().sort_values(ascending=False)
    top_positions = sorted(pos_counts.head(40).index.tolist())
    
    if len(top_positions) > 1:
        # Build switch profile for each position: which nodes did it switch at?
        print(f"  Computing Jaccard co-occurrence matrix for top {len(top_positions)} positions...")
        node_sets = {}
        for pos in top_positions:
            node_sets[pos] = set(switches_df[switches_df["position"] == pos]["node_name"].tolist())
            
        coevo_matrix = pd.DataFrame(index=top_positions, columns=top_positions, dtype=float)
        for p1 in top_positions:
            for p2 in top_positions:
                if p1 == p2:
                    coevo_matrix.loc[p1, p2] = 1.0
                    continue
                u = node_sets[p1] | node_sets[p2]
                i = node_sets[p1] & node_sets[p2]
                coevo_matrix.loc[p1, p2] = len(i) / len(u) if u else 0.0
                
        coevo_csv = out_dir / "coevolution_matrix.csv"
        coevo_matrix.to_csv(coevo_csv, index=True)
        print(f"  Saved co-evolution matrix to: {coevo_csv}")

        # Detect communities using hierarchical clustering
        # Convert Jaccard similarity to distance (1 - similarity)
        dist_matrix = 1.0 - coevo_matrix.to_numpy()
        np.fill_diagonal(dist_matrix, 0.0)
        
        # Condensed distance matrix for scipy
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage = hierarchy.linkage(condensed_dist, method="complete")
        
        # Cut tree at distance threshold 0.6 (matches legacy 60% dissimilarity threshold)
        community_ids = hierarchy.fcluster(linkage, t=0.6, criterion="distance")
        
        df_communities = pd.DataFrame({
            "position": top_positions,
            "community_id": community_ids
        }).sort_values(by=["community_id", "position"])
        
        comm_csv = out_dir / "coevolution_communities.csv"
        df_communities.to_csv(comm_csv, index=False)
        print(f"  Saved co-evolution communities to: {comm_csv}")
        
        # Display communities
        print("\n  Detected Co-evolution Communities:")
        for comm_id, grp in df_communities.groupby("community_id"):
            members = grp["position"].tolist()
            print(f"    - Community {comm_id:2d} ({len(members)} positions): {members}")
    else:
        print("  WARNING: Insufficient switch hotspots for co-evolution matrix.")
        coevo_matrix = pd.DataFrame()
        linkage = None

    # -------------------------------------------------------------------------
    # 4. Plot Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating enrichment and co-evolution plots...")

    # Plot 1: Domain Switch Density
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Domain color coding
    domain_colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#d62728"]
    sns.barplot(
        data=df_enrichment,
        x="domain",
        y="switch_density_pct",
        palette="Set2",
        edgecolor="black",
        linewidth=1.0,
        ax=ax
    )
    
    # Annotate p-values above bars
    for i, bar in enumerate(ax.patches):
        p_val = df_enrichment.iloc[i]["p_value"]
        sig_str = " (Significant*)" if p_val < 0.05 else " (n.s.)"
        ax.annotate(
            f"p = {p_val:.2e}{sig_str}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    ax.set_xlabel("Structural / Functional Domain", fontsize=11, fontweight="bold")
    ax.set_ylabel("Switch Density (%)", fontsize=11, fontweight="bold")
    ax.set_title("Evolutionary Switch Density & Statistical Enrichment by Domain", fontsize=12, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, df_enrichment["switch_density_pct"].max() * 1.25)
    
    plt.tight_layout()
    plt.savefig(out_dir / "domain_switch_density.svg", format="svg")
    plt.savefig(out_dir / "domain_switch_density.png", format="png", dpi=300)
    plt.close()
    print("  Saved domain_switch_density.svg and .png")

    # Plot 2: Co-evolution Heatmap
    if not coevo_matrix.empty and linkage is not None:
        fig = sns.clustermap(
            coevo_matrix,
            row_linkage=linkage,
            col_linkage=linkage,
            cmap="viridis",
            figsize=(9, 9),
            cbar_kws={"label": "Jaccard Switch Co-occurrence"},
            linewidths=0.2,
            edgecolor="gray"
        )
        fig.fig.suptitle("Residue Switch Co-evolution Clustering Map", fontsize=14, fontweight="bold", y=1.02)
        
        plt.savefig(out_dir / "coevolution_heatmap.svg", format="svg", bbox_inches="tight")
        plt.savefig(out_dir / "coevolution_heatmap.png", format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved coevolution_heatmap.svg and .png")

    print("\n================================================================================")
    print("Advanced Analyses Completed successfully!")
    print(f"All files and plots written to: {out_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
