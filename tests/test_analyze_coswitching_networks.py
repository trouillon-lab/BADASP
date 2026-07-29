"""Unit tests for scripts/analyze_coswitching_networks.py."""

import pytest
import pandas as pd
import numpy as np

from scripts.analyze_coswitching_networks import (
    build_cooccurrence_matrix,
    compute_pairwise_associations,
    calculate_3d_distances,
    evaluate_spatial_coswitching_clustering,
)


@pytest.fixture
def dummy_switches_df():
    """Generate dummy switch occurrences dataframe."""
    np.random.seed(42)
    nodes = [f"Node_{i}" for i in range(20)]
    records = []
    
    # Simulate correlated positions (e.g. pos 35 and 38 switch together on nodes 0..5)
    for n in nodes[:6]:
        records.append({"node_name": n, "position": 35})
        records.append({"node_name": n, "position": 38})
        
    # Uncorrelated switches
    for i in range(15):
        records.append({
            "node_name": np.random.choice(nodes),
            "position": np.random.choice([10, 20, 30, 40, 50])
        })
        
    return pd.DataFrame(records)


def test_build_cooccurrence_matrix(dummy_switches_df):
    matrix, positions, nodes = build_cooccurrence_matrix(dummy_switches_df, active_positions=[10, 20, 30, 35, 38, 40, 50])
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[1] == 7
    # Check that node 0 has 1 at positions 35 and 38
    pos_idx_35 = positions.index(35)
    pos_idx_38 = positions.index(38)
    assert matrix[:, pos_idx_35].sum() >= 6
    assert matrix[:, pos_idx_38].sum() >= 6


def test_compute_pairwise_associations(dummy_switches_df):
    matrix, positions, nodes = build_cooccurrence_matrix(dummy_switches_df, active_positions=[10, 20, 30, 35, 38, 40, 50])
    assoc_df = compute_pairwise_associations(matrix, positions)
    
    assert isinstance(assoc_df, pd.DataFrame)
    assert "pos1" in assoc_df.columns
    assert "pos2" in assoc_df.columns
    assert "jaccard" in assoc_df.columns
    assert "pearson" in assoc_df.columns
    
    # 35 and 38 should have high Jaccard similarity
    pair_row = assoc_df[((assoc_df["pos1"] == 35) & (assoc_df["pos2"] == 38)) | ((assoc_df["pos1"] == 38) & (assoc_df["pos2"] == 35))]
    assert not pair_row.empty
    assert pair_row.iloc[0]["jaccard"] > 0.5


def test_spatial_coswitching_clustering_func():
    np.random.seed(42)
    # Mock 3D distance matrix for 5 positions
    positions = [10, 20, 30, 35, 38]
    n = len(positions)
    dist_matrix = np.random.uniform(5.0, 35.0, size=(n, n))
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    dist_matrix[3, 4] = 4.5  # 35 and 38 are close (4.5Å)
    dist_matrix[4, 3] = 4.5
    
    assoc_df = pd.DataFrame([
        {"pos1": 35, "pos2": 38, "jaccard": 0.8},
        {"pos1": 10, "pos2": 20, "jaccard": 0.1},
        {"pos1": 10, "pos2": 30, "jaccard": 0.05},
        {"pos1": 20, "pos2": 30, "jaccard": 0.02},
    ])
    
    res = evaluate_spatial_coswitching_clustering(assoc_df, dist_matrix, positions, jaccard_threshold=0.5)
    assert "ks_stat" in res
    assert "p_val" in res
    assert "coswitching_mean_dist" in res

