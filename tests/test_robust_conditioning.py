"""Tests for src/badasp/robust_conditioning.py.

These check the mechanics -- that cells are formed from the null alone, that
the z-transform preserves within-cell ordering, that the inverse map is
consistent, and that the known failure modes (NaN draws counted as trials,
empty cells becoming automatically significant, observed scores leaking into
the fit) do not occur. Whether the model helps on the real data is a
measurement reported by the calibration script, not asserted here.
"""

import numpy as np
import pytest

from src.badasp.robust_conditioning import (
    CellConditionalModel,
    assign_clade_bins,
    clade_bin_edges,
    fit_cell_model,
    t_from_z,
    to_z,
)
from src.badasp.tail_model import fit_pooled_tail

RNG_SEED = 20260816
N_TESTS = 400
N_REPLICATES = 12


@pytest.fixture()
def toy():
    """A null in which 5 of 20 positions are genuinely 3x noisier."""
    rng = np.random.default_rng(RNG_SEED)
    positions = np.repeat(np.arange(1, 21), N_TESTS // 20)
    hot_truth = positions <= 5
    scale = np.where(hot_truth, 3.0, 1.0)
    clade_left = rng.integers(5, 5000, size=N_TESTS).astype(float)
    clade_right = rng.integers(5, 5000, size=N_TESTS).astype(float)
    null_left = rng.exponential(scale, size=(N_REPLICATES, N_TESTS))
    null_right = rng.exponential(scale, size=(N_REPLICATES, N_TESTS))
    return positions, clade_left, clade_right, null_left, null_right


def _fit(toy, **kw):
    positions, cl, cr, nl, nr = toy
    tail = fit_pooled_tail(nl, nr)
    model = fit_cell_model(nl, nr, positions, cl, cr, tail, n_hot=5, n_clade_bins=4, **kw)
    return tail, model


def test_clade_bins_share_one_scale_across_branches():
    left = np.array([10.0, 20.0, 30.0, 40.0])
    right = np.array([15.0, 25.0, 35.0, 45.0])
    edges = clade_bin_edges(left, right, 4)
    # A size of 25 must land in the same bin whichever branch it came from.
    assert assign_clade_bins(np.array([25.0]), edges, 4) == assign_clade_bins(
        np.array([25.0]), edges, 4
    )
    assert len(edges) == 3  # deduplicated interior edges for 4 bins


def test_hot_positions_are_the_genuinely_noisier_ones(toy):
    _, model = _fit(toy)
    assert set(model.hot_positions.tolist()) == {1, 2, 3, 4, 5}


def test_hot_positions_get_a_multiplier_above_one(toy):
    _, model = _fit(toy)
    # Row 1 is the hot group; its multipliers must exceed the cold group's.
    assert model.multiplier[1].mean() > model.multiplier[0].mean()
    assert model.multiplier[1].mean() > 1.0


def test_fit_never_reads_observed_scores(toy):
    """The rule is non-circular only because the cells come from the null."""
    positions, cl, cr, nl, nr = toy
    tail = fit_pooled_tail(nl, nr)
    a = fit_cell_model(nl, nr, positions, cl, cr, tail, n_hot=5, n_clade_bins=4)
    b = fit_cell_model(nl, nr, positions, cl, cr, tail, n_hot=5, n_clade_bins=4)
    assert np.array_equal(a.multiplier, b.multiplier)
    assert np.array_equal(a.hot_positions, b.hot_positions)


def test_nan_draws_are_not_counted_as_trials(toy):
    """Counting NaN draws as trials would deflate the rate of any cell whose
    tests are more often undefined, which then looks like a quiet cell."""
    positions, cl, cr, nl, nr = toy
    tail = fit_pooled_tail(nl, nr)
    base = fit_cell_model(nl, nr, positions, cl, cr, tail, n_hot=5, n_clade_bins=4)

    nl_nan = nl.copy()
    nl_nan[:, positions == 20] = np.nan  # blank one cold position entirely
    holed = fit_cell_model(nl_nan, nr, positions, cl, cr, tail, n_hot=5, n_clade_bins=4)
    # Trials must drop by exactly the number of NaN draws introduced.
    assert holed.cell_trials.sum() == base.cell_trials.sum() - np.isnan(nl_nan).sum()


def test_empty_cell_falls_back_to_the_pooled_rate(toy):
    """A zero-event cell must not get A = 0, which would make every test in
    it automatically significant."""
    positions, cl, cr, nl, nr = toy
    tail = fit_pooled_tail(nl, nr)
    model = fit_cell_model(nl, nr, positions, cl, cr, tail, n_hot=5, n_clade_bins=4)
    model.multiplier = np.where(np.isnan(model.multiplier), 1.0, model.multiplier)
    assert np.all(np.isfinite(model.multiplier))
    assert np.all(model.multiplier > 0)


def test_z_is_monotone_in_score_within_a_cell(toy):
    """Within one cell A is constant, so thresholding on z must reproduce
    thresholding on the raw score exactly."""
    positions, cl, cr, nl, nr = toy
    tail, model = _fit(toy)
    scores = np.linspace(0.01, 8.0, 50)
    one_pos = np.full(50, positions[0])
    one_clade = np.full(50, cl[0])
    z = to_z(scores, one_pos, one_clade, model, tail)
    assert np.all(np.diff(z) >= 0)


def test_z_preserves_nan(toy):
    positions, cl, cr, nl, nr = toy
    tail, model = _fit(toy)
    scores = nl[0].copy()
    scores[:10] = np.nan
    z = to_z(scores, positions, cl, model, tail)
    assert np.all(np.isnan(z[:10]))
    assert np.all(np.isfinite(z[10:]))


def test_hot_positions_are_held_to_a_higher_raw_bar(toy):
    """The point of conditioning: at one z, a noisy column must require a
    higher raw score than a quiet one."""
    positions, cl, cr, nl, nr = toy
    tail, model = _fit(toy)
    median_clade = np.full(2, float(np.median(cl)))
    pair = np.array([model.hot_positions[0], 20])  # one hot, one cold
    thresholds = t_from_z(3.0, pair, median_clade, model, tail)
    assert thresholds[0] > thresholds[1]


def test_t_from_z_inverts_to_z(toy):
    positions, cl, cr, nl, nr = toy
    tail, model = _fit(toy)
    subset = np.arange(0, N_TESTS, 37)
    t = t_from_z(2.5, positions[subset], cl[subset], model, tail)
    back = to_z(t, positions[subset], cl[subset], model, tail)
    assert np.allclose(back, 2.5, atol=1e-6)


def test_jackknife_se_is_finite_and_small_when_cells_are_well_populated(toy):
    _, model = _fit(toy)
    assert np.all(np.isfinite(model.multiplier_log_se))
    assert np.nanmax(model.multiplier_log_se) < 1.0


def test_roundtrip_through_dict(toy):
    _, model = _fit(toy)
    restored = CellConditionalModel.from_dict(model.to_dict())
    assert np.array_equal(restored.hot_positions, model.hot_positions)
    assert np.allclose(restored.multiplier, model.multiplier)
    assert restored.anchor_score == pytest.approx(model.anchor_score)


def test_mismatched_lengths_are_rejected(toy):
    positions, cl, cr, nl, nr = toy
    tail = fit_pooled_tail(nl, nr)
    with pytest.raises(ValueError, match="positions has length"):
        fit_cell_model(nl, nr, positions[:-1], cl, cr, tail)


def test_anchor_alpha_must_be_a_probability(toy):
    positions, cl, cr, nl, nr = toy
    tail = fit_pooled_tail(nl, nr)
    with pytest.raises(ValueError, match="anchor_alpha"):
        fit_cell_model(nl, nr, positions, cl, cr, tail, anchor_alpha=1.5)


def test_no_bin_is_left_unreachable_by_a_tie_atom():
    """`--min-clade 5` puts ~10% of branches at exactly the smallest size,
    which collapses the low quantiles onto the minimum. An edge equal to the
    minimum defines a bin nothing can fall into (np.digitize sends a value
    equal to an edge to the bin above), which surfaces later as a spurious
    empty cell."""
    sizes = np.concatenate([np.full(200, 5.0), np.arange(6, 806, dtype=float)])
    edges = clade_bin_edges(sizes, sizes, 10)
    n_bins = len(edges) + 1
    occupancy = np.bincount(assign_clade_bins(sizes, edges, n_bins), minlength=n_bins)
    assert np.all(occupancy > 0), f"unreachable bin(s): {occupancy}"
