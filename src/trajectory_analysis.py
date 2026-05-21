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
from sklearn.metrics import silhouette_score

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
    normalize_by_layer: bool = True
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
        if normalize_by_layer:
            col_sum = float(series.sum())
            if col_sum > 0:
                series = series / col_sum

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
    return df_matrix


def cluster_trajectories(
    active_df: pd.DataFrame,
    k_active: int,
    optimize_k: bool,
    standardize: bool
) -> Tuple[np.ndarray, int, float]:
    """Cluster active trajectories using Ward linkage hierarchical clustering."""
    X = active_df.values
    if standardize:
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        X_proc = (X - means) / stds
    else:
        X_proc = X

    link = linkage(X_proc, method="ward", metric="euclidean")

    if optimize_k and len(active_df) > 2:
        best_k = k_active
        best_silhouette = -1.0
        max_search = min(6, len(active_df) - 1)
        if max_search >= 2:
            for k in range(2, max_search + 1):
                labels = fcluster(link, k, criterion="maxclust")
                score = silhouette_score(X_proc, labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_k = k
            labels = fcluster(link, best_k, criterion="maxclust")
        else:
            labels = fcluster(link, 1, criterion="maxclust")
            best_k = 1
            best_silhouette = 0.0
    else:
        best_k = k_active
        if len(active_df) >= best_k:
            labels = fcluster(link, best_k, criterion="maxclust")
        else:
            labels = np.ones(len(active_df), dtype=int)
            best_k = 1
        if len(active_df) > 2:
            best_silhouette = float(silhouette_score(X_proc, labels))
        else:
            best_silhouette = 0.0

    return labels, best_k, best_silhouette


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
    
    lines = [
        "del all",
        f"open {pdb_path.absolute()}",
        "view",
        "color protein gainsboro",
        "color nucleic lightsteelblue",
    ]

    for cid, group in cluster_series.groupby(cluster_series):
        # Background cluster is kept gray
        if cid == 0:
            continue
            
        color = PALETTE.get(cid, "#999999")
        resnums = []
        for pos in group.index.tolist():
            if pos in mapping:
                resnum = _normalize_residue_number(mapping[pos])
                if resnum is not None:
                    resnums.append(str(resnum))
                    
        if resnums:
            res_str = ",".join(resnums)
            lines.append(f"color /A,B:{res_str} {color}")

    with output_cxc.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutionary Switch Trajectory Clustering")
    parser.add_argument("--scores-root", default="results/badasp_scoring", help="Directory of scoring layers")
    parser.add_argument("--track", default="combined", choices=["duplications", "speciations", "combined"])
    parser.add_argument("--metric", default="switch_count", choices=["badasp_score", "switch_count", "is_sdp"])
    parser.add_argument("--k-active", type=int, default=3, help="Default number of active clusters")
    parser.add_argument("--optimize-k", action="store_true", help="Automatically find the best active cluster count K")
    parser.add_argument("--standardize", action="store_false", help="Standardize trajectories to cluster based on shape instead of magnitude")
    parser.add_argument("--pdb", default="data/raw/AF_with_loop.cif", help="Reference structural model")
    parser.add_argument("--pdb-id", default="AF_with_loop", help="Structural identifier")
    parser.add_argument("--msa", default="data/interim/IPR019888_trimmed.aln", help="Trimmed input alignment")
    parser.add_argument("--domain-architecture", default="data/domain_architecture.json", help="JSON containing domain coordinates")
    parser.add_argument("--output-dir", default="results/evolutionary_analysis", help="Output directory")
    parser.add_argument("--no-normalize-by-layer", action="store_false", dest="normalize_by_layer", help="Disable normalisation of switch counts by the total number of switches in each layer")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading trajectories from {args.scores_root} for track '{args.track}' using '{args.metric}'...")
    df_matrix = load_all_layer_trajectories(args.scores_root, args.track, args.metric, normalize_by_layer=args.normalize_by_layer)
    
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
            standardize=args.standardize
        )
        logger.info(f"Clustering completed with {best_k} active clusters. Silhouette score: {best_sil:.4f}")
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
    summary_csv = output_dir / "switch_trajectories.csv"
    summary_df.to_csv(summary_csv)
    logger.info(f"Saved switch trajectories to: {summary_csv}")

    # 4. Domain enrichment
    if Path(args.domain_architecture).exists():
        with open(args.domain_architecture, "r") as f:
            domain_arch = json.load(f)
        enrich_df = run_domain_enrichment(cluster_assignments, domain_arch, df_matrix.index.tolist())
        enrich_csv = output_dir / "switch_trajectory_domain_enrichment.csv"
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
            spatial_csv = output_dir / "switch_trajectory_spatial_cohesion.csv"
            spatial_df.to_csv(spatial_csv, index=False)
            logger.info(f"Saved 3D spatial permutation test results to: {spatial_csv}")
            
            print("\n=== 3D Spatial Permutation Test (Cohesion) ===")
            for _, row in spatial_df.iterrows():
                cid = int(row["cluster"])
                cname = "Background" if cid == 0 else f"Cluster {cid}"
                print(f"{cname} (n={int(row['cluster_size'])}):")
                print(f"  Observed Mean Distance : {row['obs_mean_distance']:.2f} A")
                print(f"  Expected Mean Distance : {row['perm_mean_distance']:.2f} A")
                print(f"  Z-Score                : {row['z_score']:.2f}")
                print(f"  Empirical p-value      : {row['p_value']:.4f}")
            print("==============================================\n")
            
        cxc_path = Path("results/structural_mapping/highlight_trajectory_clusters.cxc")
        generate_cxc_script(cluster_assignments, mapping, Path(args.pdb), cxc_path)
        logger.info(f"Saved ChimeraX structural rendering script to: {cxc_path}")
    except Exception as e:
        logger.error(f"Failed during structure mapping/distance calculation: {e}", exc_info=True)

    # 6. Beautiful Premium Plots (SVGs)
    sns.set_theme(style="white", palette="muted")
    
    # 6a. Heatmap
    if not active_df.empty:
        plt.figure(figsize=(10, 8))
        # Order rows by hierarchical clustering for visual clarity
        active_sorted = active_df.copy()
        active_sorted["cluster_id"] = active_labels
        active_sorted = active_sorted.sort_values("cluster_id")
        
        # Premium Heatmap with distinct active cluster markers
        g = sns.heatmap(
            active_sorted.drop(columns="cluster_id"),
            cmap="crest",
            cbar_kws={"label": f"Intensity ({args.metric})"},
            linewidths=0.1,
            linecolor="#eeeeee"
        )
        plt.title(f"Active Switch Trajectory Clustered Heatmap ({args.track.capitalize()})", pad=15)
        plt.xlabel("Evolutionary Layer")
        plt.ylabel("Residue Position")
        plt.tight_layout()
        heatmap_path = output_dir / "switch_trajectory_heatmap.svg"
        plt.savefig(heatmap_path, format="svg")
        plt.close()
        logger.info(f"Saved Clustered Heatmap plot to: {heatmap_path}")

    # 6b. Profile Line Plot
    plt.figure(figsize=(10, 5))
    layers = [col for col in df_matrix.columns]
    layer_ticks = [int(col.split("_")[-1]) for col in layers]

    for cid, group in cluster_assignments.groupby(cluster_assignments):
        cname = "Background / Muted" if cid == 0 else f"Active Cluster {cid}"
        color = PALETTE.get(cid, "#777777")
        c_matrix = df_matrix.loc[group.index]
        
        mean_profile = c_matrix.mean(axis=0).values
        std_err = c_matrix.sem(axis=0).values
        
        plt.plot(layer_ticks, mean_profile, label=f"{cname} (n={len(group)})", color=color, linewidth=2.5)
        plt.fill_between(
            layer_ticks,
            mean_profile - std_err,
            mean_profile + std_err,
            color=color,
            alpha=0.15
        )

    plt.xlabel("Evolutionary Timeline Layer Index", fontweight="bold")
    plt.ylabel(f"Mean {args.metric.replace('_', ' ').capitalize()}", fontweight="bold")
    plt.title(f"Evolutionary Switch Trajectory Profiles ({args.track.capitalize()})", pad=15, fontsize=14, fontweight="bold")
    plt.xticks(layer_ticks)
    plt.grid(color="#EEEEEE", linestyle="--", linewidth=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    profile_path = output_dir / "switch_trajectory_profiles.svg"
    plt.savefig(profile_path, format="svg")
    plt.close()
    logger.info(f"Saved Trajectory Profile plot to: {profile_path}")


if __name__ == "__main__":
    main()
