"""Module 1: Data-Driven Time & Mechanism of Origin Analysis for Initial Run BADASP Switches.

Analyzes switch rates, domain densities, and positional patterns across data-driven
temporal quantiles of tree depth (distance_from_root). All outputs saved under
results/initial_run_characterization/.
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
    """Calculate MSA column occupancy fraction (1-indexed)."""
    alignment = AlignIO.read(alignment_path, "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)
    return occupancies


def bin_root_distances_quantile(df: pd.DataFrame, num_quantiles: int = 3) -> tuple:
    """Bin distance_from_root into equal-sample quantiles in a purely data-driven way.

    Returns:
        tuple: (DataFrame with 'time_bin' column, list of bin label categories)
    """
    df = df.copy()
    full_labels = ["Recent", "Mid", "Deep/Ancient"] if num_quantiles == 3 else [f"Q{i+1}" for i in range(num_quantiles)]

    # Probe actual bin count: duplicates="drop" may silently reduce it below num_quantiles
    probe = pd.qcut(df["distance_from_root"], q=num_quantiles, duplicates="drop")
    n_actual = probe.cat.categories.size
    labels = full_labels[:n_actual]

    qcut_series, _ = pd.qcut(df["distance_from_root"], q=num_quantiles, labels=labels, retbins=True, duplicates="drop")
    df["time_bin"] = qcut_series
    return df, list(qcut_series.cat.categories)


def calculate_bin_thresholds_999(df_filtered: pd.DataFrame, num_bins: int = 10) -> dict:
    """Calculate 99.9th percentile thresholds on left/right melted scores with clade size decile binning."""
    df_left = df_filtered[
        ["node_name", "event_type", "position", "clade_size_left", "badasp_score_left"]
    ].rename(columns={"badasp_score_left": "score", "clade_size_left": "clade_size"})
    
    df_right = df_filtered[
        ["node_name", "event_type", "position", "clade_size_right", "badasp_score_right"]
    ].rename(columns={"badasp_score_right": "score", "clade_size_right": "clade_size"})
    
    melted_df = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["score", "clade_size"])
    melted_df["clade_bin"] = pd.qcut(melted_df["clade_size"], q=num_bins, duplicates="drop")
    
    thresholds = {}
    for event in ["Duplication", "Speciation", "Transfer"]:
        event_df = melted_df[melted_df["event_type"] == event]
        for bin_interval in melted_df["clade_bin"].cat.categories:
            bin_df = event_df[event_df["clade_bin"] == bin_interval]
            scores = bin_df["score"].dropna()
            thresholds[(event, bin_interval)] = float(np.percentile(scores, 99.9)) if len(scores) > 0 else np.nan

    for bin_interval in melted_df["clade_bin"].cat.categories:
        bin_df = melted_df[melted_df["clade_bin"] == bin_interval]
        scores = bin_df["score"].dropna()
        thresholds[("overall", bin_interval)] = float(np.percentile(scores, 99.9)) if len(scores) > 0 else np.nan

    categories = sorted(melted_df["clade_bin"].cat.categories)
    return thresholds, categories


def extract_switch_instances(df_filtered: pd.DataFrame, thresholds: dict, categories: list, event_specific: bool = False) -> pd.DataFrame:
    """Extract individual switch instances for left and right clade comparisons."""
    def _map_to_bin(val):
        for interval in categories:
            if val in interval:
                return interval
        return categories[-1] if val > categories[-1].right else categories[0]

    df = df_filtered.copy()
    df["bin_left"] = df["clade_size_left"].apply(_map_to_bin)
    df["bin_right"] = df["clade_size_right"].apply(_map_to_bin)

    switch_records = []
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

        if not np.isnan(score_l) and score_l >= thresh_l:
            rec = row.to_dict()
            rec["branch"] = "left"
            rec["badasp_score"] = score_l
            rec["clade_size"] = row["clade_size_left"]
            rec["aa_target"] = row["aa_left"]
            switch_records.append(rec)

        if not np.isnan(score_r) and score_r >= thresh_r:
            rec = row.to_dict()
            rec["branch"] = "right"
            rec["badasp_score"] = score_r
            rec["clade_size"] = row["clade_size_right"]
            rec["aa_target"] = row["aa_right"]
            switch_records.append(rec)

    return pd.DataFrame(switch_records)


def compute_temporal_switch_rates(switches_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Compute switch count, node count, and switch rate per temporal bin and event type."""
    records = []
    
    time_bins = nodes_df["time_bin"].unique()
    event_types = ["Duplication", "Speciation", "Transfer", "Overall"]

    for tb in time_bins:
        tb_nodes = nodes_df[nodes_df["time_bin"] == tb]
        tb_switches = switches_df[switches_df["time_bin"] == tb] if not switches_df.empty else pd.DataFrame()

        for event in event_types:
            if event == "Overall":
                n_nodes = len(tb_nodes["node_name"].unique())
                n_switches = len(tb_switches) if not tb_switches.empty else 0
            else:
                n_nodes = len(tb_nodes[tb_nodes["event_type"] == event]["node_name"].unique())
                n_switches = len(tb_switches[tb_switches["event_type"] == event]) if not tb_switches.empty else 0

            rate = n_switches / n_nodes if n_nodes > 0 else 0.0
            records.append({
                "time_bin": tb,
                "event_type": event,
                "node_count": n_nodes,
                "switch_count": n_switches,
                "switch_rate": rate
            })

    return pd.DataFrame(records)


def compute_domain_temporal_densities(switches_df: pd.DataFrame, domains: dict = DOMAINS) -> pd.DataFrame:
    """Compute switch density per domain across temporal bins."""
    records = []
    if switches_df.empty:
        return pd.DataFrame(columns=["time_bin", "domain", "residues", "switch_count", "switch_density"])

    for tb in switches_df["time_bin"].unique():
        tb_switches = switches_df[switches_df["time_bin"] == tb]

        for d_name, (start, end) in domains.items():
            total_sites = end - start + 1
            n_sw = len(tb_switches[(tb_switches["position"] >= start) & (tb_switches["position"] <= end)])
            density = n_sw / total_sites if total_sites > 0 else 0.0

            records.append({
                "time_bin": tb,
                "domain": d_name,
                "residues": f"{start}-{end}",
                "switch_count": n_sw,
                "switch_density": density
            })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: Time & Mechanism Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8)
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--num-quantiles", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("results/initial_run_characterization"))
    args = parser.parse_args()

    plots_dir = args.out_dir / "plots"
    tables_dir = args.out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw node scores from {args.scores}...")
    df = pd.read_csv(args.scores)
    df = df[(df["clade_size_left"] >= args.min_clade_size) & (df["clade_size_right"] >= args.min_clade_size)]

    # Filter occupancy >= 80%
    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()

    # Get distinct internal nodes for distance binning
    node_df = df_filtered[["node_name", "event_type", "distance_from_root", "clade_size_total"]].drop_duplicates("node_name")
    node_binned, time_categories = bin_root_distances_quantile(node_df, num_quantiles=args.num_quantiles)
    
    # Calculate 99.9th percentile thresholds with decile clade size binning
    thresholds, clade_categories = calculate_bin_thresholds_999(df_filtered, num_bins=10)

    # Extract switch instances
    switches_df = extract_switch_instances(df_filtered, thresholds, clade_categories, event_specific=False)
    if not switches_df.empty:
        switches_df = switches_df.merge(node_binned[["node_name", "time_bin"]], on="node_name", how="left")

    # 1. Compute Switch Rates
    rates_df = compute_temporal_switch_rates(switches_df, node_binned)
    rates_df.to_csv(tables_dir / "temporal_switch_rates.csv", index=False)

    # 2. Compute Domain Densities
    domain_densities_df = compute_domain_temporal_densities(switches_df)
    domain_densities_df.to_csv(tables_dir / "temporal_domain_densities.csv", index=False)

    # Plot 1: Temporal Switch Rates Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=rates_df,
        x="time_bin",
        y="switch_rate",
        hue="event_type",
        palette=[EVENT_COLORS["Duplication"], EVENT_COLORS["Speciation"], EVENT_COLORS["Transfer"], EVENT_COLORS["Overall"]]
    )
    plt.title("BADASP Switch Rate across Data-Driven Evolutionary Time Bins (99.9th%)", fontsize=13, fontweight="bold")
    plt.xlabel("Evolutionary Depth (Quantiles of Root Distance)", fontsize=11)
    plt.ylabel("Switch Rate (Switches / Scored Node)", fontsize=11)
    plt.legend(title="Event Type", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "temporal_switch_rates.svg", format="svg")
    plt.savefig(plots_dir / "temporal_switch_rates.png", format="png", dpi=300)
    plt.close()

    # Plot 2: Domain Densities across Time
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=domain_densities_df,
        x="domain",
        y="switch_density",
        hue="time_bin",
        palette="viridis"
    )
    plt.title("Domain Switch Density across Evolutionary Time Quantiles", fontsize=13, fontweight="bold")
    plt.xlabel("Architectural Domain", fontsize=11)
    plt.ylabel("Switch Density (Switches / Residue)", fontsize=11)
    plt.legend(title="Time Quantile", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "temporal_domain_densities.svg", format="svg")
    plt.savefig(plots_dir / "temporal_domain_densities.png", format="png", dpi=300)
    plt.close()

    print(f"Module 1 completed. Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
