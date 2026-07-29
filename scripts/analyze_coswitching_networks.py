"""Module 2: Co-Switching & Positional Correlation Networks for Initial Run BADASP Switches.

Builds binary co-occurrence matrices, computes pairwise association metrics (Jaccard, Pearson),
evaluates 3D C-alpha physical proximity of co-switching pairs, and generates network figures.
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
from Bio import AlignIO, PDB
from scipy.stats import ks_2samp
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


def build_cooccurrence_matrix(switches_df: pd.DataFrame, active_positions: list) -> tuple:
    """Build binary matrix M where M[node, pos] = 1 if a switch occurred at node for pos.
    
    Returns:
        tuple: (binary matrix np.ndarray, list of positions, list of node names)
    """
    if switches_df.empty:
        return np.zeros((1, len(active_positions))), active_positions, ["none"]

    nodes = sorted(switches_df["node_name"].unique())
    positions = sorted(active_positions)

    matrix = np.zeros((len(nodes), len(positions)), dtype=int)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    for _, row in switches_df.iterrows():
        n = row["node_name"]
        p = row["position"]
        if n in node_to_idx and p in pos_to_idx:
            matrix[node_to_idx[n], pos_to_idx[p]] = 1

    return matrix, positions, nodes


def compute_pairwise_associations(matrix: np.ndarray, positions: list) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity and Pearson correlation across position pairs."""
    num_pos = len(positions)
    records = []

    for i in range(num_pos):
        vec_i = matrix[:, i]
        pos_i = positions[i]
        sum_i = vec_i.sum()

        for j in range(i + 1, num_pos):
            vec_j = matrix[:, j]
            pos_j = positions[j]
            sum_j = vec_j.sum()

            intersection = np.logical_and(vec_i, vec_j).sum()
            union = np.logical_or(vec_i, vec_j).sum()

            jaccard = intersection / union if union > 0 else 0.0

            # Pearson correlation
            if sum_i > 0 and sum_j > 0:
                corr = np.corrcoef(vec_i, vec_j)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0

            records.append({
                "pos1": pos_i,
                "pos2": pos_j,
                "cooccurrence_count": int(intersection),
                "jaccard": jaccard,
                "pearson": corr
            })

    return pd.DataFrame(records)


def calculate_3d_distances(pdb_path: Path, positions: list, alignment_path: Path = None) -> np.ndarray:
    """Extract C-alpha 3D coordinates from PDB file and build distance matrix."""
    parser = PDB.MMCIFParser(QUIET=True) if str(pdb_path).endswith(".cif") else PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue) and "CA" in residue:
                    res_num = residue.get_id()[1]
                    ca_coords[res_num] = residue["CA"].get_coord()
            break  # first chain
        break  # first model

    n = len(positions)
    dist_matrix = np.full((n, n), np.nan)

    for i, p1 in enumerate(positions):
        for j, p2 in enumerate(positions):
            if i == j:
                dist_matrix[i, j] = 0.0
            elif p1 in ca_coords and p2 in ca_coords:
                dist_matrix[i, j] = np.linalg.norm(ca_coords[p1] - ca_coords[p2])

    return dist_matrix


def evaluate_spatial_coswitching_clustering(assoc_df: pd.DataFrame, dist_matrix: np.ndarray, positions: list, jaccard_threshold: float = 0.05) -> dict:
    """Perform 2-sample Kolmogorov-Smirnov test comparing 3D distances of co-switching vs non-co-switching pairs."""
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    coswitching_dists = []
    background_dists = []

    for _, row in assoc_df.iterrows():
        p1 = int(row["pos1"])
        p2 = int(row["pos2"])

        if p1 in pos_to_idx and p2 in pos_to_idx:
            idx1 = pos_to_idx[p1]
            idx2 = pos_to_idx[p2]
            d = dist_matrix[idx1, idx2]

            if not np.isnan(d):
                if row["jaccard"] >= jaccard_threshold:
                    coswitching_dists.append(d)
                else:
                    background_dists.append(d)

    if len(coswitching_dists) >= 2 and len(background_dists) >= 2:
        ks_stat, p_val = ks_2samp(coswitching_dists, background_dists)
        res = {
            "ks_stat": float(ks_stat),
            "p_val": float(p_val),
            "n_coswitching_pairs": len(coswitching_dists),
            "n_background_pairs": len(background_dists),
            "coswitching_mean_dist": float(np.mean(coswitching_dists)),
            "background_mean_dist": float(np.mean(background_dists)),
        }
    else:
        res = {
            "ks_stat": np.nan,
            "p_val": np.nan,
            "n_coswitching_pairs": len(coswitching_dists),
            "n_background_pairs": len(background_dists),
            "coswitching_mean_dist": np.nan,
            "background_mean_dist": np.nan,
        }

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 2: Co-Switching Networks Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--pdb", type=Path, default=Path("data/raw/2cg4_dna_aligned.pdb"))
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

    active_positions = sorted([pos for pos, occ in occupancies.items() if occ >= args.min_occupancy])

    thresholds, clade_categories = calculate_bin_thresholds_999(df_filtered, num_bins=10)
    switches_df = extract_switch_instances(df_filtered, thresholds, clade_categories, event_specific=False)

    print("Building co-occurrence matrix...")
    matrix, positions, nodes = build_cooccurrence_matrix(switches_df, active_positions)

    print("Computing pairwise association metrics...")
    assoc_df = compute_pairwise_associations(matrix, positions)
    assoc_df.to_csv(tables_dir / "coswitching_pairs.csv", index=False)

    # Co-occurrence Jaccard heatmap matrix assembly
    n_pos = len(positions)
    jaccard_matrix = np.zeros((n_pos, n_pos))
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    for _, row in assoc_df.iterrows():
        i = pos_to_idx[int(row["pos1"])]
        j = pos_to_idx[int(row["pos2"])]
        jaccard_matrix[i, j] = row["jaccard"]
        jaccard_matrix[j, i] = row["jaccard"]

    # Save heatmap plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        jaccard_matrix,
        xticklabels=positions,
        yticklabels=positions,
        cmap="mako",
        cbar_kws={"label": "Jaccard Co-Switching Similarity"}
    )
    plt.title("Residue Co-Switching Similarity Matrix (99.9th% Switches)", fontsize=13, fontweight="bold")
    plt.xlabel("MSA Column Position", fontsize=11)
    plt.ylabel("MSA Column Position", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "coswitching_heatmap.svg", format="svg")
    plt.savefig(plots_dir / "coswitching_heatmap.png", format="png", dpi=300)
    plt.close()

    # 3D spatial contact validation
    if args.pdb.exists():
        print(f"Calculating 3D distances from {args.pdb}...")
        dist_matrix = calculate_3d_distances(args.pdb, positions)
        spatial_res = evaluate_spatial_coswitching_clustering(assoc_df, dist_matrix, positions, jaccard_threshold=0.05)
        
        print(f"Spatial clustering test (Jaccard >= 0.05): KS stat = {spatial_res['ks_stat']:.4f}, p = {spatial_res['p_val']:.4e}")
        print(f"Co-switching mean distance: {spatial_res['coswitching_mean_dist']:.2f}Å vs Background: {spatial_res['background_mean_dist']:.2f}Å")
        
        with open(tables_dir / "coswitching_spatial_test_stats.txt", "w") as f:
            for k, v in spatial_res.items():
                f.write(f"{k}: {v}\n")

    print(f"Module 2 completed. Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
