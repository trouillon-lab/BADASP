"""Module 5: Bradley & Beltrao (2019) Kinase Replication & Structural Validation for Initial Run BADASP Switches.

Performs 3D spatial clustering (KS test), functional domain enrichment (Fisher's exact test),
event-type physicochemical differentials, and ChimeraX structural scripting.
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
from scipy.stats import ks_2samp, fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.pdb_mapper import PDBMapper
from scripts.analyze_switch_origin_time import (
    calculate_msa_occupancies,
    calculate_bin_thresholds_999,
    extract_switch_instances,
    DOMAINS,
)
from scripts.analyze_coswitching_networks import calculate_3d_distances
from scripts.analyze_physicochemical_properties import annotate_physicochemical_shifts


def compute_3d_ks_clustering(dist_matrix: np.ndarray, positions: list, sdp_positions: list) -> dict:
    """Perform 2-sample Kolmogorov-Smirnov test comparing pairwise 3D C-alpha distances of SDP sites vs background."""
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    sdp_set = set(sdp_positions)

    sdp_dists = []
    bg_dists = []

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            p1, p2 = positions[i], positions[j]
            d = dist_matrix[i, j]

            if not np.isnan(d) and d > 0:
                if p1 in sdp_set and p2 in sdp_set:
                    sdp_dists.append(d)
                else:
                    bg_dists.append(d)

    if len(sdp_dists) >= 2 and len(bg_dists) >= 2:
        ks_stat, p_val = ks_2samp(sdp_dists, bg_dists)
        return {
            "ks_stat": float(ks_stat),
            "p_val": float(p_val),
            "sdp_pair_count": len(sdp_dists),
            "bg_pair_count": len(bg_dists),
            "sdp_mean_dist": float(np.mean(sdp_dists)),
            "bg_mean_dist": float(np.mean(bg_dists))
        }
    return {
        "ks_stat": np.nan, "p_val": np.nan,
        "sdp_pair_count": len(sdp_dists), "bg_pair_count": len(bg_dists),
        "sdp_mean_dist": np.nan, "bg_mean_dist": np.nan
    }


def compute_domain_fisher_enrichments(sdp_positions: list, active_positions: list, domains: dict = DOMAINS) -> pd.DataFrame:
    """Compute 2x2 contingency Fisher's Exact tests for SDP enrichment per domain."""
    sdp_set = set(sdp_positions)
    records = []

    for d_name, (start, end) in domains.items():
        domain_active = [p for p in active_positions if start <= p <= end]
        non_domain_active = [p for p in active_positions if not (start <= p <= end)]

        a = sum(1 for p in domain_active if p in sdp_set)  # SDPs in domain
        b = len(domain_active) - a                         # Non-SDPs in domain
        c = sum(1 for p in non_domain_active if p in sdp_set)  # SDPs outside domain
        d = len(non_domain_active) - c                     # Non-SDPs outside domain

        contingency = [[a, b], [c, d]]
        odds_ratio, p_val = fisher_exact(contingency, alternative="greater")

        records.append({
            "domain": d_name,
            "residues": f"{start}-{end}",
            "sdps_in_domain": a,
            "non_sdps_in_domain": b,
            "sdps_outside": c,
            "non_sdps_outside": d,
            "odds_ratio": float(odds_ratio),
            "p_val": float(p_val)
        })

    return pd.DataFrame(records)


def compare_event_physicochemical_differentials(annotated_switches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean Grantham distance, net charge shift, hydropathy delta, and volume shift per event type."""
    if annotated_switches_df.empty or "event_type" not in annotated_switches_df.columns:
        return pd.DataFrame()

    agg_dict = {}
    if "position" in annotated_switches_df.columns:
        agg_dict["position"] = "count"
    if "grantham_distance" in annotated_switches_df.columns:
        agg_dict["grantham_distance"] = "mean"
    if "charge_shift" in annotated_switches_df.columns:
        agg_dict["charge_shift"] = "mean"
    if "hydropathy_shift" in annotated_switches_df.columns:
        agg_dict["hydropathy_shift"] = "mean"
    if "volume_shift" in annotated_switches_df.columns:
        agg_dict["volume_shift"] = "mean"

    if not agg_dict:
        return pd.DataFrame()

    summary = annotated_switches_df.groupby("event_type").agg(agg_dict).reset_index()
    if "grantham_distance" in summary.columns:
        summary = summary.rename(columns={"grantham_distance": "mean_grantham"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5: Bradley & Beltrao (2019) Kinase Replication")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--pdb", type=Path, default=Path("data/raw/2cg4_dna_aligned.pdb"))
    parser.add_argument("--min-occupancy", type=float, default=0.8)
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results/initial_run_characterization"))
    args = parser.parse_args()

    plots_dir = args.out_dir / "plots"
    tables_dir = args.out_dir / "tables"
    cxc_dir = args.out_dir / "cxc"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cxc_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw node scores from {args.scores}...")
    df = pd.read_csv(args.scores)
    df = df[(df["clade_size_left"] >= args.min_clade_size) & (df["clade_size_right"] >= args.min_clade_size)]

    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()

    active_positions = sorted([pos for pos, occ in occupancies.items() if occ >= args.min_occupancy])

    thresholds, clade_categories = calculate_bin_thresholds_999(df_filtered, num_bins=10)
    switches_df = extract_switch_instances(df_filtered, thresholds, clade_categories, event_specific=False)
    annotated_df = annotate_physicochemical_shifts(switches_df)

    sdp_positions = sorted(switches_df["position"].unique().tolist())
    print(f"Identified {len(sdp_positions)} unique SDP positions at 99.9th% threshold across active positions.")

    # 1. 3D KS Spatial Clustering Test
    if args.pdb.exists():
        print(f"Evaluating 3D spatial clustering on {args.pdb}...")
        dist_matrix = calculate_3d_distances(args.pdb, active_positions)
        ks_res = compute_3d_ks_clustering(dist_matrix, active_positions, sdp_positions)
        print(f"3D Spatial Clustering KS-test: KS = {ks_res['ks_stat']:.4f}, p = {ks_res['p_val']:.4e}")
        print(f"SDP mean C-alpha distance: {ks_res['sdp_mean_dist']:.2f}Å vs Background: {ks_res['bg_mean_dist']:.2f}Å")

        with open(tables_dir / "bradley_beltrao_spatial_ks_test.txt", "w") as f:
            for k, v in ks_res.items():
                f.write(f"{k}: {v}\n")

    # 2. Domain Fisher's Exact Test
    print("Computing Fisher's exact domain enrichments...")
    fisher_df = compute_domain_fisher_enrichments(sdp_positions, active_positions)
    fisher_df.to_csv(tables_dir / "bradley_beltrao_domain_fisher_enrichment.csv", index=False)

    # 3. Event Physicochemical Differentials
    diff_df = compare_event_physicochemical_differentials(annotated_df)
    diff_df.to_csv(tables_dir / "event_physicochemical_differentials.csv", index=False)

    # 4. Generate ChimeraX Script
    if args.pdb.exists():
        print("Generating ChimeraX structural script for 99.9th% switches...")
        mapper = PDBMapper(pdb_id="2cg4_dna_aligned", pdb_file=str(args.pdb))
        mapper.map_alignment_to_structure(args.alignment)

        # Temp CSV of SDP counts
        sdp_counts = switches_df.groupby("position").size().reset_index(name="switch_count")
        temp_csv = cxc_dir / "temp_sdp_counts.csv"
        sdp_counts.to_csv(temp_csv, index=False)

        output_cxc = cxc_dir / "highlight_initial_run_switches_99.9.cxc"
        mapper.generate_single_chimerax_script(
            alignment_path=args.alignment,
            sdp_csv=temp_csv,
            output_cxc=output_cxc,
            level_label="Initial Run BADASP 99.9th% Switches"
        )
        if temp_csv.exists():
            temp_csv.unlink()
        print(f"Saved ChimeraX script to {output_cxc}")

    # Plot 1: Event-type Grantham Distance Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=diff_df,
        x="event_type",
        y="mean_grantham",
        palette={"Duplication": "#c0392b", "Speciation": "#2980b9", "Transfer": "#27ae60"}
    )
    plt.title("Mean Grantham Chemical Severity by Event Type (99.9th%)", fontsize=13, fontweight="bold")
    plt.xlabel("Evolutionary Event Type", fontsize=11)
    plt.ylabel("Mean Grantham Distance", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "event_grantham_comparison.svg", format="svg")
    plt.savefig(plots_dir / "event_grantham_comparison.png", format="png", dpi=300)
    plt.close()

    print(f"Module 5 completed. Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
