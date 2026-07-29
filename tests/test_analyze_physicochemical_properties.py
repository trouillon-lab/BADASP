"""Unit tests for scripts/analyze_physicochemical_properties.py."""

import pytest
import pandas as pd
import numpy as np

from scripts.analyze_physicochemical_properties import (
    get_grantham_distance,
    compute_charge_shift,
    compute_hydropathy_shift,
    compute_volume_shift,
    annotate_physicochemical_shifts,
)


def test_grantham_distance():
    # Known Grantham distances: Ala -> Ala = 0, Ala -> Cys = 195, Lys -> Arg = 26
    assert get_grantham_distance('A', 'A') == 0
    assert get_grantham_distance('A', 'C') == 195
    assert get_grantham_distance('K', 'R') == 26
    assert get_grantham_distance('X', 'A') == 0.0  # unknown fallback


def test_charge_shift():
    # K (+1) to D (-1) -> delta = -2
    assert compute_charge_shift('K', 'D') == -2
    # E (-1) to R (+1) -> delta = +2
    assert compute_charge_shift('E', 'R') == 2
    # A (0) to K (+1) -> delta = +1
    assert compute_charge_shift('A', 'K') == 1
    # A (0) to L (0) -> delta = 0
    assert compute_charge_shift('A', 'L') == 0


def test_hydropathy_shift():
    # I (4.5) to K (-3.9) -> delta = -8.4
    delta_h = compute_hydropathy_shift('I', 'K')
    assert pytest.approx(delta_h, 0.1) == -8.4


def test_annotate_physicochemical_shifts():
    df = pd.DataFrame([
        {"position": 35, "aa_left": "K", "aa_target": "D", "badasp_score": 2.1},
        {"position": 77, "aa_left": "I", "aa_target": "V", "badasp_score": 1.9},
    ])
    
    annotated = annotate_physicochemical_shifts(df)
    
    assert "grantham_distance" in annotated.columns
    assert "charge_shift" in annotated.columns
    assert "hydropathy_shift" in annotated.columns
    assert "volume_shift" in annotated.columns
    
    assert annotated.iloc[0]["charge_shift"] == -2
    assert annotated.iloc[1]["charge_shift"] == 0
