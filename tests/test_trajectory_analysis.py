import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from src.trajectory_analysis import (
    cluster_trajectories,
    run_domain_enrichment,
    load_all_layer_trajectories,
    run_spatial_permutation_test
)


def test_cluster_trajectories():
    """Verify that hierarchical clustering groups active trajectories correctly and isolates background."""
    # Create distinct trajectory matrices:
    # 2 residues that spike early, 2 residues that spike late
    data = [
        [5.0, 5.0, 0.0, 0.0, 0.0],
        [4.8, 5.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.9, 5.1],
        [0.0, 0.0, 0.0, 5.1, 4.9]
    ]
    df = pd.DataFrame(data, index=[1, 2, 3, 4], columns=["Layer_02", "Layer_03", "Layer_04", "Layer_05", "Layer_06"])

    labels, best_k, silhouette = cluster_trajectories(df, k_active=2, optimize_k=False, standardize=False)
    
    assert best_k == 2
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_run_domain_enrichment():
    """Verify that Fisher's domain enrichment correctly identifies statistically enriched clusters."""
    # Active Cluster 1 is highly enriched in Domain_A
    cluster_assignments = pd.Series([1, 1, 1, 2, 2, 0, 0, 0, 0, 0], index=range(1, 11))
    
    domain_arch = {
        "Domain_A": [1, 3],  # positions 1-3
        "Domain_B": [8, 10]  # positions 8-10
    }
    
    enrich_df = run_domain_enrichment(cluster_assignments, domain_arch, range(1, 11))
    
    assert not enrich_df.empty
    
    # Check that Cluster 1's overlap with Domain_A is exactly 3
    sub = enrich_df[(enrich_df["cluster"] == 1) & (enrich_df["domain"] == "Domain_A")]
    assert len(sub) == 1
    assert sub.iloc[0]["overlap"] == 3
    assert sub.iloc[0]["p_value"] < 0.1  # Significant enrichment


def test_load_all_layer_trajectories(tmp_path):
    """Test loading and combining of trajectories across mock layer directories."""
    # Setup mock folder layout
    scoring_root = tmp_path / "badasp_scoring"
    scoring_root.mkdir()
    
    for l_idx in [1, 2, 3]:
        layer_dir = scoring_root / f"layer_{l_idx:02d}"
        layer_dir.mkdir()
        
        # Write mock score files
        csv_path = layer_dir / "badasp_scores_duplications.csv"
        df = pd.DataFrame({
            "position": [1, 2, 3],
            "badasp_score": [float(l_idx * 1.5), float(l_idx * 2.0), 0.0]
        })
        df.to_csv(csv_path, index=False)

    df_matrix_raw = load_all_layer_trajectories(scoring_root, "duplications", "badasp_score", normalize_by_layer=False)
    
    # Verify that Layer 1 is correctly ignored to prevent stale ghost file impacts
    assert "Layer_01" not in df_matrix_raw.columns
    assert "Layer_02" in df_matrix_raw.columns
    assert "Layer_03" in df_matrix_raw.columns
    
    # Verify position index and raw values
    assert df_matrix_raw.loc[1, "Layer_02"] == 3.0
    assert df_matrix_raw.loc[2, "Layer_03"] == 6.0

    # Verify normalized loader behavior
    df_matrix_norm = load_all_layer_trajectories(scoring_root, "duplications", "badasp_score", normalize_by_layer=True)
    # Total sum for Layer_02 is 3.0 (pos 1) + 4.0 (pos 2) + 0.0 (pos 3) = 7.0
    # Expected value at pos 1: 3.0 / 7.0 = 0.42857142857142855
    assert abs(df_matrix_norm.loc[1, "Layer_02"] - (3.0 / 7.0)) < 1e-6
