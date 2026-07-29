"""Unit tests for scripts/analyze_switch_origin_time.py."""

import pytest
import pandas as pd
import numpy as np

from scripts.analyze_switch_origin_time import (
    bin_root_distances_quantile,
    compute_temporal_switch_rates,
    compute_domain_temporal_densities,
)


@pytest.fixture
def dummy_nodes_df():
    """Generate dummy scored nodes dataframe with distance_from_root and event_type."""
    np.random.seed(42)
    n = 100
    distances = np.random.uniform(0.5, 4.5, size=n)
    events = np.random.choice(["Duplication", "Speciation", "Transfer"], size=n)
    
    nodes_df = pd.DataFrame({
        "node_name": [f"Node_{i}" for i in range(n)],
        "event_type": events,
        "distance_from_root": distances,
        "clade_size_total": np.random.randint(10, 500, size=n)
    })
    return nodes_df


@pytest.fixture
def dummy_switches_df(dummy_nodes_df):
    """Generate dummy switches dataframe linked to dummy_nodes_df."""
    np.random.seed(42)
    # Pick 30 switch instances
    sample_nodes = dummy_nodes_df.sample(30, replace=True)
    positions = np.random.randint(1, 169, size=30)
    
    switches_df = sample_nodes.copy()
    switches_df["position"] = positions
    switches_df["badasp_score"] = np.random.uniform(1.6, 2.5, size=30)
    return switches_df


def test_bin_root_distances_quantile(dummy_nodes_df):
    binned_df, categories = bin_root_distances_quantile(dummy_nodes_df, num_quantiles=3)
    
    assert "time_bin" in binned_df.columns
    assert len(categories) == 3
    # Check that equal or roughly equal samples are in each quantile bin
    counts = binned_df["time_bin"].value_counts()
    assert len(counts) == 3
    assert abs(counts.iloc[0] - counts.iloc[1]) <= 2


def test_compute_temporal_switch_rates(dummy_nodes_df, dummy_switches_df):
    nodes_binned, categories = bin_root_distances_quantile(dummy_nodes_df, num_quantiles=3)
    
    # Add time_bin to switches_df by merging on node_name
    switches_binned = dummy_switches_df.merge(
        nodes_binned[["node_name", "time_bin"]], on="node_name", how="inner"
    )
    
    rates_df = compute_temporal_switch_rates(switches_binned, nodes_binned)
    
    assert isinstance(rates_df, pd.DataFrame)
    assert "time_bin" in rates_df.columns
    assert "event_type" in rates_df.columns
    assert "switch_count" in rates_df.columns
    assert "node_count" in rates_df.columns
    assert "switch_rate" in rates_df.columns
    assert len(rates_df) > 0


def test_compute_domain_temporal_densities(dummy_switches_df, dummy_nodes_df):
    nodes_binned, categories = bin_root_distances_quantile(dummy_nodes_df, num_quantiles=3)
    switches_binned = dummy_switches_df.merge(
        nodes_binned[["node_name", "time_bin"]], on="node_name", how="inner"
    )
    
    domains = {
        "HTH Scaffold": (6, 34),
        "Recognition Helix": (35, 50),
        "RAM Domain": (68, 152)
    }
    
    densities_df = compute_domain_temporal_densities(switches_binned, domains=domains)
    
    assert isinstance(densities_df, pd.DataFrame)
    assert "time_bin" in densities_df.columns
    assert "domain" in densities_df.columns
    assert "switch_density" in densities_df.columns
