#!/usr/bin/env python3
"""Residue switch trajectory evolutionary clustering and 3D structural mapping."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.PDB import MMCIFParser, PDBParser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from src.pdb_mapper import PDBMapper

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
# Premium color palette for publication-quality visual aesthetics
PALETTE = {
    0: "#BAB0AC",  # Background / Muted Light Gray
    1: "#E15759",  # Active Cluster 1: Vibrant Salmon/Red
    2: "#4E79A7",  # Active Cluster 2: Elegant Steel Blue
    3: "#76B7B2",  # Active Cluster 3: Premium Soft Teal
    4: "#59A14F",  # Active Cluster 4: Emerald Green
    5: "#F28E2B",  # Active Cluster 5: Warm Orange
    6: "#B07AA1",  # Active Cluster 6: Soft Purple
    7: "#FF9DA7",  # Active Cluster 7: Pale Pink
    8: "#9C755F",  # Active Cluster 8: Light Brown
    9: "#90A4AE",  # Active Cluster 9: Blue Gray
    10: "#D32F2F", # Active Cluster 10: Dark Red
    11: "#1976D2", # Active Cluster 11: Dark Blue
    12: "#388E3C", # Active Cluster 12: Dark Green
    13: "#FBC02D", # Active Cluster 13: Vibrant Yellow
    14: "#8E24AA", # Active Cluster 14: Dark Purple
    15: "#00838F", # Active Cluster 15: Deep Soft Blue-Teal
}

def _normalize_residue_number(value: object) -> Optional[int]:
    """Normalize mapper outputs to a single integer residue number."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer, float, np.floating, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, tuple) and len(value) >= 2:
        return _normalize_residue_number(value[1])
    if isinstance(value, list):
        for candidate in value:
            normalized = _normalize_residue_number(candidate)
            if normalized is not None:
                return normalized
        return None
    return None


def calculate_ca_distance_matrix(pdb_path: Path, residue_numbers: Sequence[int]) -> pd.DataFrame:
    """Calculate Euclidean distance matrix between C-alpha coordinates in a PDB or MMCIF model."""
    pdb_path = Path(pdb_path)
    if pdb_path.suffix.lower() == ".cif":
        structure = MMCIFParser(QUIET=True).get_structure("structure", str(pdb_path))
    else:
        structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))

    ca_coords: Dict[int, np.ndarray] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                try:
                    resseq = int(residue.id[1])
                except (ValueError, IndexError):
                    continue
                if resseq not in residue_numbers:
                    continue
                if "CA" not in residue:
                    continue
                ca_coords[resseq] = residue["CA"].coord
            if ca_coords:
                break
        if ca_coords:
            break

    ordered = [int(r) for r in residue_numbers if int(r) in ca_coords]
    matrix = pd.DataFrame(index=ordered, columns=ordered, dtype=float)
    for left in ordered:
        for right in ordered:
            matrix.loc[left, right] = float(np.linalg.norm(ca_coords[left] - ca_coords[right]))
    return matrix


def load_all_layer_trajectories(
    scores_root: Path,
    track: str,
    metric: str,
    normalize_by_layer: bool = True,
    normalize_method: str = "sum"
) -> pd.DataFrame:
    """Load trajectories across all layers for a given track and metric.
    
    Excludes layer_01 to prevent stale ghost file impacts.
    """
    scores_root = Path(scores_root)
    layer_files = sorted(scores_root.glob("layer_*/badasp_scores_*.csv"))
    target_suffix = f"badasp_scores_{track}.csv"
    layer_files = [f for f in layer_files if f.name == target_suffix]

    layer_data: Dict[int, Dict[int, float]] = {}
    for f in layer_files:
        try:
            layer_idx = int(f.parent.name.split("_")[-1])
        except (ValueError, IndexError):
            continue

        # Skip Layer 1 because it represents 1 cluster (conceptually invalid for pairwise comparisons)
        if layer_idx == 1:
            continue

        df = pd.read_csv(f)
        if df.empty:
            continue

        if "position" not in df.columns or metric not in df.columns:
            continue

        series = df.set_index("position")[metric].astype(float)
        
        # Determine effective method based on flags
        eff_method = "none" if not normalize_by_layer else normalize_method
        
        # Apply layer-wise scaling only if NOT using a global method
        if eff_method not in ["global_max", "log1p"]:
            if eff_method == "sum":
                col_sum = float(series.sum())
                if col_sum > 0:
                    series = series / col_sum
            elif eff_method == "max":
                col_max = float(series.max())
                if col_max > 0:
                    series = series / col_max
            elif eff_method == "mean":
                col_mean = float(series.mean())
                if col_mean > 0:
                    series = series / col_mean
            elif eff_method == "zscore":
                col_mean = float(series.mean())
                col_std = float(series.std())
                if col_std == 0:
                    col_std = 1.0
                series = (series - col_mean) / col_std

        layer_data[layer_idx] = series.to_dict()

    if not layer_data:
        raise ValueError(f"No trajectory data found for track '{track}' and metric '{metric}' under {scores_root}")

    all_positions = sorted(list(set(
        pos for layer_dict in layer_data.values() for pos in layer_dict
    )))

    matrix_data = []
    for pos in all_positions:
        row = {"position": pos}
        for layer_idx, pos_dict in sorted(layer_data.items()):
            row[f"Layer_{layer_idx:02d}"] = float(pos_dict.get(pos, 0.0))
        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data).set_index("position")

    # Filter out dead layers (columns containing absolutely zero switches)
    active_cols = [col for col in df_matrix.columns if (df_matrix[col] > 0).any()]
    dropped_cols = [col for col in df_matrix.columns if col not in active_cols]
    if dropped_cols:
        logger.info(f"Filtered out dead layers (all zero switches): {dropped_cols}")
    df_matrix = df_matrix[active_cols]

    # Filter out dead residues (rows containing absolutely zero switches across surviving layers)
    active_rows = df_matrix.index[(df_matrix > 0).any(axis=1)]
    dropped_rows_count = len(df_matrix) - len(active_rows)
    if dropped_rows_count > 0:
        logger.info(f"Filtered out {dropped_rows_count} dead residues (all zero switches across surviving layers)")
    df_matrix = df_matrix.loc[active_rows]

    # Apply global normalization/transformations if specified
    if normalize_by_layer:
        if normalize_method == "global_max":
            global_max = float(df_matrix.max().max())
            if global_max > 0:
                logger.info(f"Applying global max normalization with scaling factor: {global_max}")
                df_matrix = df_matrix / global_max
        elif normalize_method == "log1p":
            logger.info("Applying log1p transformation: np.log1p(matrix)")
            df_matrix = np.log1p(df_matrix)

    return df_matrix


def plot_clustering_parameter_sweep(sweep_df: pd.DataFrame, output_path: Path) -> None:
    """Plot Calinski-Harabasz and Silhouette scores across K=2-20."""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = "#4E79A7"
    ax1.set_xlabel("Number of Clusters (K)", fontweight="bold")
    ax1.set_ylabel("Calinski-Harabasz Score", color=color, fontweight="bold")
    line1 = ax1.plot(sweep_df["K"], sweep_df["Calinski_Harabasz"], color=color, marker="o", linewidth=2, label="Calinski-Harabasz")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color = "#E15759"
    ax2.set_ylabel("Silhouette Score", color=color, fontweight="bold")
    line2 = ax2.plot(sweep_df["K"], sweep_df["Silhouette"], color=color, marker="s", linewidth=2, linestyle="--", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    plt.title("Trajectory Clustering Parameter Sweep (K = 2 to 20)", fontweight="bold", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="svg")
    plt.close()


def plot_trajectory_dendrogram(link: np.ndarray, output_path: Path) -> None:
    """Plot the hierarchical clustering dendrogram of switch trajectories."""
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    dendrogram(
        link,
        no_labels=True,  # Too many leaves
        color_threshold=link[-3, 2],  # Color the branches for K=4
        above_threshold_color="#BAB0AC"
    )
    plt.title("Hierarchical Clustering Dendrogram of Active Switch Trajectories (K = 4)", fontweight="bold", fontsize=12, pad=15)
    plt.xlabel("Residue Positions", fontweight="bold")
    plt.ylabel("Ward Linkage Distance", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, format="svg")
    plt.close()


def cluster_trajectories(
    active_df: pd.DataFrame,
    k_active: int,
    optimize_k: bool,
    standardize: bool,
    output_dir: Optional[Path] = None,
    track: str = ""
) -> Tuple[np.ndarray, int, float]:
    X = active_df.values
    if standardize:
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        X_proc = (X - means) / stds
    else:
        X_proc = X

    link = linkage(X_proc, method="ward", metric="euclidean")

    # ----------------------------------------------------
    # Diagnostic Loop: Evaluate K from 3 up to 20
    # ----------------------------------------------------
    max_k_diagnostic = min(20, len(active_df) - 1)
    sweep_rows = []
    if max_k_diagnostic >= 3:
        logger.info("=== Diagnostic Clustering Evaluation (K = 3 to 20) ===")
        for k in range(3, max_k_diagnostic + 1):
            try:
                lbls = fcluster(link, k, criterion="maxclust")
                ch_score = calinski_harabasz_score(X_proc, lbls)
                sil_score = silhouette_score(X_proc, lbls)
                logger.info(f"  K = {k:2d} | Calinski-Harabasz = {ch_score:10.4f} | Silhouette = {sil_score:7.4f}")
                sweep_rows.append({
                    "K": k,
                    "Calinski_Harabasz": ch_score,
                    "Silhouette": sil_score
                })
            except Exception as e:
                logger.warning(f"  K = {k:2d} failed to evaluate: {e}")
        logger.info("=====================================================")
        
        # Save sweep results to CSV and SVG plot if output_dir is provided
        if output_dir is not None:
            try:
                sweep_df = pd.DataFrame(sweep_rows)
                suffix = f"_{track}" if track else ""
                sweep_csv = output_dir / f"clustering_parameter_sweep{suffix}.csv"
                sweep_csv.parent.mkdir(parents=True, exist_ok=True)
                sweep_df.to_csv(sweep_csv, index=False)
                logger.info(f"Saved clustering parameter sweep results to: {sweep_csv}")
                
                sweep_svg = output_dir / f"clustering_parameter_sweep{suffix}.svg"
                plot_clustering_parameter_sweep(sweep_df, sweep_svg)
                logger.info(f"Saved clustering parameter sweep plot to: {sweep_svg}")

                dendrogram_svg = output_dir / f"switch_trajectory_dendrogram{suffix}.svg"
                plot_trajectory_dendrogram(link, dendrogram_svg)
                logger.info(f"Saved switch trajectory dendrogram to: {dendrogram_svg}")
            except Exception as e:
                logger.warning(f"Failed to write parameter sweep files: {e}")

    # Dynamically select optimal K via Silhouette + Calinski-Harabasz optimization if optimize_k is enabled
    if optimize_k:
        if sweep_rows:
            # Consider K in the range [3, 15] as requested
            valid_sweep = [r for r in sweep_rows if 3 <= r["K"] <= 15]
            if not valid_sweep:
                valid_sweep = sweep_rows
            
            max_sil = max(r["Silhouette"] for r in valid_sweep)
            # Find all candidates with Silhouette score within 0.01 of the maximum
            candidates = [r for r in valid_sweep if (max_sil - r["Silhouette"]) <= 0.01]
            # Select the one with the highest Calinski-Harabasz score among these candidates
            best_row = max(candidates, key=lambda r: r["Calinski_Harabasz"])
            best_k = best_row["K"]
            logger.info(f"Dynamically selected optimal K={best_k} (Silhouette = {best_row['Silhouette']:.4f}, CH = {best_row['Calinski_Harabasz']:.4f})")
        else:
            best_k = min(4, len(active_df) - 1)
            if best_k < 3:
                best_k = 3
            logger.info(f"No sweep results found. Falling back to default K={best_k}.")
            
        labels = fcluster(link, best_k, criterion="maxclust")
        best_ch = float(calinski_harabasz_score(X_proc, labels)) if len(active_df) > 2 else 0.0
    else:
        best_k = k_active
        if len(active_df) >= best_k:
            labels = fcluster(link, best_k, criterion="maxclust")
        else:
            labels = np.ones(len(active_df), dtype=int)
            best_k = 1
        if len(active_df) > 2:
            best_ch = float(calinski_harabasz_score(X_proc, labels))
        else:
            best_ch = 0.0

    return labels, best_k, best_ch


def run_domain_enrichment(
    cluster_series: pd.Series,
    domain_arch: Dict[str, Sequence[int]],
    total_positions: Sequence[int]
) -> pd.DataFrame:
    """Run Fisher's exact test to check each cluster for enrichment in structural domains."""
    rows = []
    all_pos_set = set(total_positions)

    for cluster_id, group in cluster_series.groupby(cluster_series):
        cluster_set = set(group.index.tolist())
        cluster_size = len(cluster_set)

        for domain, (start, end) in domain_arch.items():
            domain_set = set(range(start, end + 1)) & all_pos_set
            domain_size = len(domain_set)

            overlap = len(cluster_set & domain_set)

            in_c_in_d = overlap
            not_c_in_d = domain_size - overlap
            in_c_not_d = cluster_size - overlap
            not_c_not_d = len(all_pos_set) - domain_size - in_c_not_d

            table = [
                [in_c_in_d, not_c_in_d],
                [in_c_not_d, not_c_not_d]
            ]

            _, p_value = fisher_exact(table, alternative="greater")

            rows.append({
                "cluster": cluster_id,
                "domain": domain,
                "overlap": overlap,
                "cluster_size": cluster_size,
                "domain_size": domain_size,
                "p_value": float(p_value)
            })

    return pd.DataFrame(rows)


def run_spatial_permutation_test(
    cluster_series: pd.Series,
    pdb_path: Path,
    mapping: Dict[int, int],
    n_permutations: int = 2000
) -> pd.DataFrame:
    """Verify if cluster residues are spatially grouped in 3D using a C-alpha distance permutation test."""
    pdb_path = Path(pdb_path)
    
    # 1. Map positions in clusters to their integer residue numbers
    mapped_clusters: Dict[int, List[int]] = {}
    all_mapped_res: List[int] = []
    pos_to_res: Dict[int, int] = {}

    for pos, cid in cluster_series.items():
        if pos not in mapping:
            continue
        resnum = _normalize_residue_number(mapping[pos])
        if resnum is None:
            continue
        pos_to_res[pos] = resnum
        all_mapped_res.append(resnum)
        mapped_clusters.setdefault(cid, []).append(resnum)

    if not all_mapped_res:
        logger.warning("No clustered residues could be mapped onto structure.")
        return pd.DataFrame()

    # 2. Get C-alpha coordinates
    dist_matrix = calculate_ca_distance_matrix(pdb_path, all_mapped_res)
    
    rows = []
    # 3. Permute for each cluster
    for cid, res_list in sorted(mapped_clusters.items()):
        # Filter list to only residues that are successfully in the distance matrix
        res_list = [r for r in res_list if r in dist_matrix.index]
        m = len(res_list)
        if m < 2:
            rows.append({
                "cluster": cid,
                "cluster_size": m,
                "obs_mean_distance": np.nan,
                "perm_mean_distance": np.nan,
                "z_score": np.nan,
                "p_value": np.nan
            })
            continue

        # Observed mean distance in the cluster
        obs_sub = dist_matrix.loc[res_list, res_list]
        obs_arr = obs_sub.to_numpy()
        obs_tri = obs_arr[np.triu_indices(obs_arr.shape[0], k=1)]
        obs_mean = float(np.mean(obs_tri))

        # Permutations
        perm_means = []
        all_indices = dist_matrix.index.tolist()
        for _ in range(n_permutations):
            perm_subset = np.random.choice(all_indices, size=m, replace=False)
            perm_sub = dist_matrix.loc[perm_subset, perm_subset]
            perm_arr = perm_sub.to_numpy()
            perm_tri = perm_arr[np.triu_indices(perm_arr.shape[0], k=1)]
            perm_means.append(np.mean(perm_tri))

        perm_means = np.array(perm_means)
        mu_perm = np.mean(perm_means)
        sigma_perm = np.std(perm_means)
        if sigma_perm == 0:
            sigma_perm = 1.0

        z_score = (obs_mean - mu_perm) / sigma_perm
        # Spatial cohesion: we are interested if obs_mean is smaller than null
        p_val = float((np.sum(perm_means <= obs_mean) + 1) / (n_permutations + 1))

        rows.append({
            "cluster": cid,
            "cluster_size": m,
            "obs_mean_distance": obs_mean,
            "perm_mean_distance": mu_perm,
            "z_score": z_score,
            "p_value": p_val
        })

    return pd.DataFrame(rows)


def generate_cxc_script(
    cluster_series: pd.Series,
    mapping: Dict[int, int],
    pdb_path: Path,
    output_cxc: Path
) -> None:
    """Generate a clean, high-resolution ChimeraX script to highlight trajectory clusters."""
    output_cxc.parent.mkdir(parents=True, exist_ok=True)

    chain_ids = sorted({chain_id for values in mapping.values() for chain_id, _ in (values if isinstance(values, list) else [values]) if chain_id})
    chain_selector = "/" + ",".join(chain_ids) if chain_ids else ""

    full_assignments = pd.Series(0, index=pd.Index(sorted(mapping.keys()), dtype=int))
    if not cluster_series.empty:
        full_assignments.loc[cluster_series.index.intersection(full_assignments.index)] = cluster_series.loc[
            cluster_series.index.intersection(full_assignments.index)
        ].astype(int)
    
    lines = [
        "del all",
        f"open {pdb_path.absolute()}",
        "view",
        "color protein gainsboro",
        "color nucleic lightsteelblue",
    ]

    for cid, group in full_assignments.groupby(full_assignments):
        color = PALETTE.get(cid, "#777777")
        resnums = []
        for pos in group.index.tolist():
            if pos in mapping:
                resnum = _normalize_residue_number(mapping[pos])
                if resnum is not None:
                    resnums.append(str(resnum))
        if resnums:
            res_str = ",".join(resnums)
            if chain_selector:
                lines.append(f"color {chain_selector}:{res_str} {color}")
            else:
                lines.append(f"color :{res_str} {color}")

    with output_cxc.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutionary Switch Trajectory Clustering")
    parser.add_argument("--scores-root", default="results/badasp_scoring", help="Directory of scoring layers")
    parser.add_argument("--track", default="combined", choices=["duplications", "speciations", "combined"])
    parser.add_argument("--metric", default="switch_count", choices=["badasp_score", "switch_count", "is_sdp"])
    parser.add_argument("--k-active", type=int, default=4, help="Default number of active clusters")
    parser.add_argument("--optimize-k", action="store_true", help="Automatically find the best active cluster count K")
    parser.add_argument("--standardize", action="store_true", default=True, help="Standardize trajectories row-wise to cluster by shape instead of magnitude (defaults to True)")
    parser.add_argument("--no-standardize", action="store_false", dest="standardize", help="Disable row-wise standardization of trajectories")
    parser.add_argument("--pdb", default="data/raw/AF_with_loop.cif", help="Reference structural model")
    parser.add_argument("--pdb-id", default="AF_with_loop", help="Structural identifier")
    parser.add_argument("--msa", default="data/interim/IPR019888_trimmed.aln", help="Trimmed input alignment")
    parser.add_argument("--domain-architecture", default="data/domain_architecture.json", help="JSON containing domain coordinates")
    parser.add_argument("--output-dir", default="results/evolutionary_analysis", help="Output directory")
    parser.add_argument("--no-normalize-by-layer", action="store_false", dest="normalize_by_layer", help="Disable normalisation of switch counts by the total number of switches in each layer")
    parser.add_argument("--normalize-method", default="sum", choices=["sum", "max", "mean", "zscore", "global_max", "log1p", "none"], help="Method to normalize/scale switch counts in each layer")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.track in ["duplications", "speciations"]:
        output_dir = output_dir / args.track
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trajectories from {args.scores_root} for track '{args.track}' using '{args.metric}'...")
    df_matrix = load_all_layer_trajectories(
        args.scores_root, 
        args.track, 
        args.metric, 
        normalize_by_layer=args.normalize_by_layer,
        normalize_method=args.normalize_method
    )
    
    # 1. Background separation
    sums = df_matrix.sum(axis=1)
    is_background = sums <= 1e-5
    
    active_df = df_matrix[~is_background]
    bg_df = df_matrix[is_background]
    
    logger.info(f"Total positions loaded: {len(df_matrix)}. Active residues: {len(active_df)}, Background residues: {len(bg_df)}.")

    # 2. Clustering
    if not active_df.empty:
        active_labels, best_k, best_sil = cluster_trajectories(
            active_df=active_df,
            k_active=args.k_active,
            optimize_k=args.optimize_k,
            standardize=args.standardize,
            output_dir=output_dir,
            track=args.track
        )
        logger.info(f"Clustering completed with {best_k} active clusters. Calinski-Harabasz score: {best_sil:.4f}")
    else:
        active_labels = np.array([])
        best_k = 0
        best_sil = 0.0
        logger.warning("No active switching residues found to cluster.")

    # 3. Combine assignments (0 represents background, 1..K represent active clusters)
    cluster_assignments = pd.Series(0, index=df_matrix.index)
    if not active_df.empty:
        cluster_assignments.loc[active_df.index] = active_labels

    # Save details
    summary_df = df_matrix.copy()
    summary_df["cluster_id"] = cluster_assignments
    summary_csv = output_dir / f"switch_trajectories_{args.track}.csv"
    summary_df.to_csv(summary_csv)
    logger.info(f"Saved switch trajectories to: {summary_csv}")

    # 4. Domain enrichment
    if Path(args.domain_architecture).exists():
        with open(args.domain_architecture, "r") as f:
            domain_arch = json.load(f)
        enrich_df = run_domain_enrichment(cluster_assignments, domain_arch, df_matrix.index.tolist())
        enrich_csv = output_dir / f"switch_trajectory_domain_enrichment_{args.track}.csv"
        enrich_df.to_csv(enrich_csv, index=False)
        logger.info(f"Saved domain enrichment stats to: {enrich_csv}")
    else:
        logger.warning(f"Domain architecture file not found: {args.domain_architecture}")

    # 5. PDB Mapping & Spatial Permutation Test
    mapper = PDBMapper(pdb_id=args.pdb_id, pdb_file=args.pdb)
    try:
        mapping = mapper.map_alignment_to_structure(Path(args.msa))
        spatial_df = run_spatial_permutation_test(cluster_assignments, Path(args.pdb), mapping)
        if not spatial_df.empty:
            spatial_csv = output_dir / f"switch_trajectory_spatial_cohesion_{args.track}.csv"
            spatial_df.to_csv(spatial_csv, index=False)
            logger.info(f"Saved 3D spatial permutation test results to: {spatial_csv}")
            
            print(f"\n=== 3D Spatial Permutation Test (Cohesion) [{args.track.upper()}] ===")
            for _, row in spatial_df.iterrows():
                cid = int(row["cluster"])
                cname = "Background" if cid == 0 else f"Cluster {cid}"
                print(f"{cname} (n={int(row['cluster_size'])}):")
                print(f"  Observed Mean Distance : {row['obs_mean_distance']:.2f} A")
                print(f"  Expected Mean Distance : {row['perm_mean_distance']:.2f} A")
                print(f"  Z-Score                : {row['z_score']:.2f}")
                print(f"  Empirical p-value      : {row['p_value']:.4f}")
            print("==============================================\n")
            
        cxc_path = Path("results/structural_mapping") / f"highlight_trajectory_clusters_{output_dir.name}_{args.track}.cxc"
        generate_cxc_script(cluster_assignments, mapping, Path(args.pdb), cxc_path)
        logger.info(f"Saved ChimeraX structural rendering script to: {cxc_path}")
        
        # Also save to the standard default path highlight_trajectory_clusters.cxc
        default_cxc_path = Path("results/structural_mapping") / f"highlight_trajectory_clusters_{args.track}.cxc"
        generate_cxc_script(cluster_assignments, mapping, Path(args.pdb), default_cxc_path)
        logger.info(f"Saved default ChimeraX structural rendering script to: {default_cxc_path}")
    except Exception as e:
        logger.error(f"Failed during structure mapping/distance calculation: {e}", exc_info=True)

    # 6. Beautiful Premium Plots (SVGs)
    sns.set_theme(style="white", palette="muted")
    
    # Determine custom colorbar label based on normalization method
    cbar_label = f"Intensity ({args.metric})"
    if args.normalize_method == "log1p":
        cbar_label = "Log1p(Switch Count)"
    elif args.normalize_method == "global_max":
        cbar_label = "Global Relative Intensity"
    elif args.normalize_method == "zscore":
        cbar_label = "Z-Score Standardized Intensity"
    elif args.normalize_method == "mean":
        cbar_label = "Mean Normalized Intensity"
    elif args.normalize_method == "max":
        cbar_label = "Max Normalized Intensity"
    elif args.normalize_method == "sum":
        cbar_label = "Sum Normalized Intensity"
    # 6a. Heatmap (Seaborn Clustermap for publication-grade layout with row Z-score standardisation)
    if not active_df.empty:
        # Standardize if args.standardize is set, to match clustering linkage
        if args.standardize:
            X = active_df.values
            means = X.mean(axis=1, keepdims=True)
            stds = X.std(axis=1, keepdims=True)
            stds[stds == 0] = 1.0
            X_proc = (X - means) / stds
        else:
            X_proc = active_df.values
            
        link = linkage(X_proc, method="ward", metric="euclidean")
        
        # Build a row Z-scored standardized DataFrame for heatmap display
        # This is the most common approach and ensures visual colors align perfectly with tree leaf clusters!
        active_df_std = pd.DataFrame(X_proc, index=active_df.index, columns=active_df.columns)
        
        # Color coding active labels to their PALETTE colors
        row_colors = pd.Series(active_labels, index=active_df.index).map(PALETTE).fillna("#777777")
        
        # Plot the publication-ready clustermap using our precomputed linkage!
        # row_cluster=True with row_linkage=link ensures it automatically clusters rows and sorts them perfectly.
        # col_cluster=False preserves chronological left-to-right order on the X-axis!
        g = sns.clustermap(
            active_df_std,
            row_linkage=link,
            row_cluster=True,
            col_cluster=False,
            row_colors=row_colors,
            cmap="vlag",  # Beautiful standard diverging colormap
            center=0,     # Center colormap at Z-score = 0
            cbar_kws={"label": "Row Z-Score (Relative Intensity)"},
            linewidths=0.1,
            linecolor="#eeeeee",
            yticklabels=True,
            figsize=(10, 8)
        )
        
        # Format axes ticklabels to be absolutely professional
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        
        # Add beautiful title
        g.fig.suptitle(f"Active Switch Trajectory Clustered Heatmap ({args.track.capitalize()})", y=1.02, fontsize=14, fontweight="bold")
        
        heatmap_path = output_dir / f"switch_trajectory_heatmap_{args.track}.svg"
        g.savefig(heatmap_path, format="svg", bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Clustered Heatmap plot to: {heatmap_path}")

    # 6b. Profile Line Plot
    # Z-score standardize df_matrix row-wise to align trajectory shapes perfectly on the same scale!
    X_mat = df_matrix.values
    means_mat = X_mat.mean(axis=1, keepdims=True)
    stds_mat = X_mat.std(axis=1, keepdims=True)
    stds_mat[stds_mat == 0] = 1.0
    df_matrix_std = pd.DataFrame(
        (X_mat - means_mat) / stds_mat,
        index=df_matrix.index,
        columns=df_matrix.columns
    )

    plt.figure(figsize=(10, 5))
    layers = [col for col in df_matrix_std.columns]
    layer_ticks = [int(col.split("_")[-1]) for col in layers]

    for cid, group in cluster_assignments.groupby(cluster_assignments):
        cname = "Background / Muted" if cid == 0 else f"Active Cluster {cid}"
        color = PALETTE.get(cid, "#777777")
        c_matrix = df_matrix_std.loc[group.index]
        
        # Plot very fine, translucent individual residue trajectories
        for _, row in c_matrix.iterrows():
            plt.plot(layer_ticks, row.values, color=color, alpha=0.12, linewidth=0.5, zorder=1)
            
        # Plot bold, thicker cluster mean trajectory
        mean_profile = c_matrix.mean(axis=0).values
        plt.plot(layer_ticks, mean_profile, label=f"{cname} (n={len(group)})", color=color, linewidth=3.0, zorder=2)

    plt.xlabel("Evolutionary Timeline Layer Index", fontweight="bold")
    plt.ylabel("Row Z-Score (Relative Intensity)", fontweight="bold")
    plt.title(f"Evolutionary Switch Trajectory Profiles ({args.track.capitalize()})", pad=15, fontsize=14, fontweight="bold")
    plt.xticks(layer_ticks)
    plt.grid(color="#EEEEEE", linestyle="--", linewidth=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    profile_path = output_dir / f"switch_trajectory_profiles_{args.track}.svg"
    plt.savefig(profile_path, format="svg")
    plt.close()
    logger.info(f"Saved Trajectory Profile plot to: {profile_path}")
if __name__ == "__main__":
    main()
