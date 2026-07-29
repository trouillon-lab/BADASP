"""Unit tests for scripts/analyze_taxonomic_switches.py."""

import pytest
import pandas as pd
import numpy as np

from scripts.analyze_taxonomic_switches import (
    identify_topological_clade_markers,
    compute_phylum_enrichments,
)


@pytest.fixture
def dummy_switches_df():
    """Generate dummy switch instances with node names, positions, and clade sizes."""
    records = []

    # Node_10 defines a major clade of size 150 with switches at pos 35 and 38
    for pos in [35, 38]:
        records.append({
            "node_name": "Node_10",
            "position": pos,
            "event_type": "Duplication",
            "clade_size": 150,
            "badasp_score": 2.1,
            "distance_from_root": 1.2
        })

    # Node_20 defines a clade of size 80 with switch at pos 95
    records.append({
        "node_name": "Node_20",
        "position": 95,
        "event_type": "Speciation",
        "clade_size": 80,
        "badasp_score": 1.8,
        "distance_from_root": 2.5
    })

    return pd.DataFrame(records)


def test_identify_topological_clade_markers(dummy_switches_df):
    markers_df = identify_topological_clade_markers(dummy_switches_df, min_clade_size=50)

    assert isinstance(markers_df, pd.DataFrame)
    assert "node_name" in markers_df.columns
    assert "marker_positions" in markers_df.columns
    assert "num_switches" in markers_df.columns

    # Node_10 should have 2 marker positions (35, 38)
    n10_row = markers_df[markers_df["node_name"] == "Node_10"]
    assert not n10_row.empty
    assert n10_row.iloc[0]["num_switches"] == 2


def test_compute_phylum_enrichments():
    # Mock node phylum composition
    node_taxa = {
        "Node_10": {"Bacillota": 120, "Pseudomonadota": 30},
        "Node_20": {"Bacillota": 10, "Actinomycetota": 70}
    }
    node_switches = {
        "Node_10": [35, 38],
        "Node_20": [95]
    }
    background_phyla = {"Bacillota": 5000, "Pseudomonadota": 4000, "Actinomycetota": 3000}

    enrich_df = compute_phylum_enrichments(node_taxa, node_switches, background_phyla)

    assert isinstance(enrich_df, pd.DataFrame)
    assert "node_name" in enrich_df.columns
    assert "phylum" in enrich_df.columns
    assert "odds_ratio" in enrich_df.columns
    assert "p_val" in enrich_df.columns
