"""Unit tests for scripts/replicate_bradley_beltrao_analyses.py."""

import pytest
import pandas as pd
import numpy as np

from scripts.replicate_bradley_beltrao_analyses import (
    compute_3d_ks_clustering,
    compute_domain_fisher_enrichments,
    compare_event_physicochemical_differentials,
)


def test_compute_3d_ks_clustering():
    # Mock distance matrix for 5 sites
    positions = [10, 20, 35, 38, 50]
    n = len(positions)
    dist_matrix = np.full((n, n), 20.0)
    np.fill_diagonal(dist_matrix, 0.0)
    # 35, 38, 50 are close to each other (4.0Å)
    dist_matrix[2, 3] = dist_matrix[3, 2] = 4.0
    dist_matrix[2, 4] = dist_matrix[4, 2] = 5.0
    dist_matrix[3, 4] = dist_matrix[4, 3] = 4.5

    sdp_positions = [35, 38, 50]
    res = compute_3d_ks_clustering(dist_matrix, positions, sdp_positions)

    assert "ks_stat" in res
    assert "p_val" in res
    assert "sdp_mean_dist" in res
    assert res["sdp_mean_dist"] < res["bg_mean_dist"]


def test_compute_domain_fisher_enrichments():
    # 147 active sites total, 16 in Rec Helix, 81 in RAM domain
    active_positions = list(range(1, 148))
    # 10 SDP positions, 5 of which are in Rec Helix (35-50)
    sdp_positions = [35, 36, 37, 38, 39, 70, 75, 80, 85, 90]

    domains = {
        "Recognition Helix": (35, 50),
        "RAM Domain": (68, 152)
    }

    df_enrich = compute_domain_fisher_enrichments(sdp_positions, active_positions, domains)

    assert isinstance(df_enrich, pd.DataFrame)
    assert "domain" in df_enrich.columns
    assert "odds_ratio" in df_enrich.columns
    assert "p_val" in df_enrich.columns
    assert len(df_enrich) == 2


def test_compare_event_physicochemical_differentials():
    df_switches = pd.DataFrame([
        {"event_type": "Duplication", "grantham_distance": 120.0, "charge_shift": 1.0},
        {"event_type": "Duplication", "grantham_distance": 90.0, "charge_shift": -1.0},
        {"event_type": "Speciation", "grantham_distance": 40.0, "charge_shift": 0.0},
        {"event_type": "Speciation", "grantham_distance": 35.0, "charge_shift": 0.0},
        {"event_type": "Transfer", "grantham_distance": 60.0, "charge_shift": 1.0},
    ])

    diff_df = compare_event_physicochemical_differentials(df_switches)

    assert isinstance(diff_df, pd.DataFrame)
    assert "event_type" in diff_df.columns
    assert "mean_grantham" in diff_df.columns
    assert diff_df[diff_df["event_type"] == "Duplication"]["mean_grantham"].iloc[0] > diff_df[diff_df["event_type"] == "Speciation"]["mean_grantham"].iloc[0]
