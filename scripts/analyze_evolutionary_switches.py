"""Analyze evolutionary switches (95th percentile BADASP scoring sites):
1. Load raw node-wise scores.
2. Calculate the global 95th percentile threshold.
3. Identify switches (exceeding threshold).
4. Tally switches by protein sequence position (overall and by event type).
5. Tally switches by tree node (overall and by event type).
6. Generate beautiful, publication-quality visualizations.

All outputs are saved under results/badasp_scoring/analyses/
"""

import sys
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


def main():
    root_dir = Path(project_root)
    scores_path = root_dir / "results" / "badasp_scoring" / "raw_node_scores.csv"
    out_dir = root_dir / "results" / "badasp_scoring" / "analyses"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists():
        print(f"Error: Scores file {scores_path} not found.")
        sys.exit(1)

    print("================================================================================")
    print("BADASP Evolutionary Switch Analysis")
    print("================================================================================\n")

    # 1. Load raw scores
    print("Loading raw node-wise scores...")
    df = pd.read_csv(scores_path)
    print(f"  Loaded {len(df)} position-wise records across {df['node_name'].nunique()} unique nodes.")

    # 2. Compute max score per row
    df["max_score"] = df[["badasp_score_left", "badasp_score_right"]].max(axis=1)

    # 3. Calculate 95th percentile threshold
    threshold = df["max_score"].quantile(0.95)
    print(f"  Global 95th percentile BADASP threshold: {threshold:.6f}")

    # 4. Identify switches
    df["is_switch"] = df["max_score"] >= threshold
    n_switches = df["is_switch"].sum()
    print(f"  Total evolutionary switch events identified: {n_switches} ({n_switches / len(df) * 100:.2f}% of all positions)")

    # -------------------------------------------------------------------------
    # 5. Tally by Protein Sequence Position
    # -------------------------------------------------------------------------
    print("\nTallying switches by protein sequence position...")
    # Get overall counts per position
    pos_overall = df.groupby("position")["is_switch"].sum().reset_index()
    pos_overall.columns = ["position", "total_switches"]

    # Get counts per position split by event type
    pos_events = df.groupby(["position", "event_type"])["is_switch"].sum().unstack(fill_value=0).reset_index()
    
    # Merge them
    pos_tally = pd.merge(pos_overall, pos_events, on="position")
    
    # Ensure all positions from 1 to max_position are represented
    max_pos = df["position"].max()
    all_positions = pd.DataFrame({"position": range(1, max_pos + 1)})
    pos_tally = pd.merge(all_positions, pos_tally, on="position", how="left").fillna(0)
    
    # Convert counts to integers
    for col in pos_tally.columns:
        pos_tally[col] = pos_tally[col].astype(int)

    pos_csv_path = out_dir / "sequence_switch_tallies.csv"
    pos_tally.to_csv(pos_csv_path, index=False)
    print(f"  Saved sequence-wise switch tallies to: {pos_csv_path}")

    # Display top 10 switch hotspot positions
    top_hotspots = pos_tally.sort_values(by="total_switches", ascending=False).head(10)
    print("\n  Top 10 Switch Hotspot Positions in Sequence:")
    for _, r in top_hotspots.iterrows():
        print(f"    Position {r['position']:3d}: Total={r['total_switches']:2d} (S={r['Speciation']:2d}, D={r['Duplication']:2d}, T={r['Transfer']:2d})")

    # -------------------------------------------------------------------------
    # 6. Tally by Tree Node
    # -------------------------------------------------------------------------
    print("\nTallying switches by tree node...")
    # Count switches per unique node
    node_switches = df.groupby("node_name")["is_switch"].sum().reset_index()
    node_switches.columns = ["node_name", "switch_count"]

    # Extract node metadata (take first occurrence)
    node_meta = df.groupby("node_name")[
        ["event_type", "distance_from_root", "clade_size_left", "clade_size_right", "clade_size_total"]
    ].first().reset_index()

    # Merge
    node_tally = pd.merge(node_meta, node_switches, on="node_name")
    node_tally = node_tally.sort_values(by="switch_count", ascending=False)

    node_csv_path = out_dir / "node_switch_counts.csv"
    node_tally.to_csv(node_csv_path, index=False)
    print(f"  Saved node-wise switch counts to: {node_csv_path}")

    # Display top 10 nodes with the most switches
    top_nodes = node_tally.head(10)
    print("\n  Top 10 Nodes with Most Switch Events:")
    for _, r in top_nodes.iterrows():
        print(f"    Node {r['node_name']:10s} ({r['event_type']:11s}): Switches={int(r['switch_count']):2d}, Depth={r['distance_from_root']:.4f}, CladeSize={int(r['clade_size_total']):3d}")

    # -------------------------------------------------------------------------
    # 7. Generate Visualizations
    # -------------------------------------------------------------------------
    print("\nGenerating publication-quality visualizations...")
    colors = {
        "Speciation": "#1f77b4",  # Blue
        "Duplication": "#d62728", # Red
        "Transfer": "#2ca02c"     # Green
    }

    # Plot 1: Sequence-wise Switch Profile (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    p = pos_tally["position"]
    s_vals = pos_tally.get("Speciation", pd.Series(0, index=p.index))
    d_vals = pos_tally.get("Duplication", pd.Series(0, index=p.index))
    t_vals = pos_tally.get("Transfer", pd.Series(0, index=p.index))

    # Stacked bars
    ax.bar(p, d_vals, label="Duplication", color=colors["Duplication"], edgecolor="none")
    ax.bar(p, s_vals, bottom=d_vals, label="Speciation", color=colors["Speciation"], edgecolor="none")
    ax.bar(p, t_vals, bottom=d_vals + s_vals, label="Transfer", color=colors["Transfer"], edgecolor="none")

    ax.set_xlabel("Alignment Position (Residue Index)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Switch Events", fontsize=12, fontweight="bold")
    ax.set_title("BADASP Evolutionary Switch Profile Along Protein Sequence", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max_pos + 1)
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "sequence_switch_profile.svg", format="svg")
    plt.savefig(out_dir / "sequence_switch_profile.png", format="png", dpi=300)
    plt.close()
    print("  Saved sequence_switch_profile.svg and .png")

    # Plot 2: Donut Chart of Switch Proportions by Event Type
    switch_by_event = df[df["is_switch"]]["event_type"].value_counts()
    labels = [f"{ev}\n({switch_by_event[ev]} switches)" for ev in switch_by_event.index]
    wedge_colors = [colors.get(ev, "#95a5a6") for ev in switch_by_event.index]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        switch_by_event,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=wedge_colors,
        textprops=dict(color="black", fontsize=10, weight="bold"),
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    ax.set_title("Distribution of Switch Events by Evolutionary Classification", fontsize=12, fontweight="bold", pad=15)
    
    plt.tight_layout()
    plt.savefig(out_dir / "event_switch_distribution.svg", format="svg")
    plt.savefig(out_dir / "event_switch_distribution.png", format="png", dpi=300)
    plt.close()
    print("  Saved event_switch_distribution.svg and .png")

    # Plot 3: Node Switch Scatter (Switches vs. Tree Depth & Clade Size)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot nodes with >= 1 switch
    nodes_with_switches = node_tally[node_tally["switch_count"] >= 1]
    
    if not nodes_with_switches.empty:
        sns.scatterplot(
            data=nodes_with_switches,
            x="distance_from_root",
            y="switch_count",
            hue="event_type",
            size="clade_size_total",
            sizes=(30, 300),
            palette=colors,
            alpha=0.75,
            edgecolor="black",
            linewidths=0.5,
            ax=ax
        )
        ax.set_xlabel("Evolutionary Depth (Distance from Root)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Switches per Node", fontsize=12, fontweight="bold")
        ax.set_title("Evolutionary Switches per Node vs. Phylogenetic Depth", fontsize=13, fontweight="bold", pad=15)
        ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="none", loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No nodes with >= 1 switch found.", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "node_switch_scatter.svg", format="svg")
    plt.savefig(out_dir / "node_switch_scatter.png", format="png", dpi=300)
    plt.close()
    print("  Saved node_switch_scatter.svg and .png")

    # Plot 4: Switch Depth Density (Where in the tree are switches located?)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    switches_df = df[df["is_switch"]]
    
    if not switches_df.empty:
        sns.kdeplot(
            data=switches_df,
            x="distance_from_root",
            hue="event_type",
            fill=True,
            common_norm=False,
            palette=colors,
            alpha=0.4,
            linewidth=1.5,
            ax=ax
        )
        ax.set_xlabel("Evolutionary Depth (Distance from Root)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density of Switch Events", fontsize=12, fontweight="bold")
        ax.set_title("Phylogenetic Depth Distribution of Evolutionary Switches", fontsize=13, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No switches found to plot density.", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "switch_depth_density.svg", format="svg")
    plt.savefig(out_dir / "switch_depth_density.png", format="png", dpi=300)
    plt.close()
    print("  Saved switch_depth_density.svg and .png")

    print("\n================================================================================")
    print("Analysis Completed Successfully!")
    print(f"All files and plots written to: {out_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
