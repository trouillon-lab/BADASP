"""Tests for src/badasp/posterior_correction.py.

These exercise the mechanics on synthetic data with a known discrepancy:
that the map recovers the target distribution, that it handles the atom at
p_AC = 1.0 that defeats deterministic quantile mapping, that it is
reproducible, and that it never reads the bins it was told to hold out.
Whether the correction helps on the real null is measured by the
calibration script, not asserted here.
"""

import numpy as np
import pytest

from src.badasp.posterior_correction import (
    PosteriorCorrection,
    assign_bins,
    corrected_null_score,
    fit_posterior_correction,
    rc_bin_edges,
)

SEED = 20260816
N = 40000


def _make(rng, n, excess, rc_dependent, atom=0.15):
    """p_AC rising with RC, plus an atom at exactly 1.0.

    `excess` is how over-confident this sample is. The continuous part is
    kept clear of the 1.0 ceiling on purpose: clipping there would make the
    REALIZED gap shrink with RC even when `excess` is constant, so the
    fixture would silently stop testing the constant-gap case the method
    assumes. `rc_dependent` deliberately reintroduces RC dependence for the
    test that documents what happens when the assumption fails.
    """
    rc = rng.uniform(0.1, 0.9, n)
    weight = (1.0 - rc) if rc_dependent else 1.0
    base = 0.30 + 0.45 * rc + 0.05 * rng.beta(2, 2, n)     # in [0.30, 0.76]
    conf = np.clip(base + excess * weight, 0.0, 0.95)
    conf[rng.random(n) < (atom + excess)] = 1.0
    return rc, conf


@pytest.fixture()
def synthetic():
    """The case C′ targets: the null is over-confident by an amount that is
    CONSTANT in RC, matching what was measured inside the AC = -1 stratum
    (gap 0.020, 0.044, 0.038, 0.034, 0.031 across signal-free deciles).
    Constancy is what makes the pooled map valid for the unfitted,
    conserved bins."""
    rng = np.random.default_rng(SEED)
    obs_rc, obs_p = _make(rng, N, 0.0, rc_dependent=False)
    null_rc, null_p = _make(rng, N, 0.12, rc_dependent=False)
    return obs_rc, obs_p, null_rc, null_p


def _residual_by_bin(correction, null_rc, null_p, obs_rc, obs_p, seed):
    """Mean corrected-null minus observed p_AC, per RC bin."""
    out = correction.apply(null_rc, null_p, seed=seed)
    nb = assign_bins(null_rc, correction.rc_edges)
    ob = assign_bins(obs_rc, correction.rc_edges)
    fitted = set(int(b) for b in correction.fitted_bins)
    res = {}
    for b in range(correction.n_bins):
        mn, mo = nb == b, ob == b
        if mn.sum() < 10 or mo.sum() < 10:
            continue
        res[b] = (float(out[mn].mean() - obs_p[mo].mean()), b in fitted)
    return res


def test_rc_bin_edges_drop_unreachable_bins():
    rc = np.concatenate([np.full(500, 0.2), np.linspace(0.2, 1.0, 500)])
    edges = rc_bin_edges(rc, 10)
    occupancy = np.bincount(assign_bins(rc, edges), minlength=len(edges) + 1)
    assert np.all(occupancy > 0)


def test_correction_moves_null_toward_observed(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    before = abs(null_p.mean() - obs_p.mean())
    after = abs(c.apply(null_rc, null_p, seed=SEED).mean() - obs_p.mean())
    assert after < before / 4, f"gap {before:.4f} -> {after:.4f}"


def test_fitted_bins_are_matched_almost_exactly(synthetic):
    """Within a fitted bin the map is a direct quantile match, so there is
    no extrapolation and almost nothing should remain."""
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    res = _residual_by_bin(c, null_rc, null_p, obs_rc, obs_p, SEED)
    fitted = [abs(r) for r, was_fitted in res.values() if was_fitted]
    assert max(fitted) < 0.01, fitted


def test_constant_gap_extrapolates_to_unfitted_bins(synthetic):
    """The load-bearing assumption: when the gap really is constant in RC,
    borrowing the quantile-wise shift must also correct the bins that were
    held back."""
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    res = _residual_by_bin(c, null_rc, null_p, obs_rc, obs_p, SEED)
    held = [abs(r) for r, was_fitted in res.values() if not was_fitted]
    assert held, "no bins were held back"
    assert max(held) < 0.03, held


def test_rc_dependent_gap_breaks_the_extrapolation():
    """The limitation, made visible rather than left to be discovered: if
    the gap is NOT constant in RC, borrowing the shift OVER-corrects the
    held-back conserved bins -- by more than the original gap, so the
    correction is worse than doing nothing there. This is why the constancy
    of the AC = -1 gap has to be checked on the real data, not assumed."""
    rng = np.random.default_rng(SEED + 1)
    obs_rc, obs_p = _make(rng, N, 0.0, rc_dependent=True)
    null_rc, null_p = _make(rng, N, 0.12, rc_dependent=True)
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    res = _residual_by_bin(c, null_rc, null_p, obs_rc, obs_p, SEED)
    fitted = [abs(r) for r, f in res.values() if f]
    held = [abs(r) for r, f in res.values() if not f]
    assert max(fitted) < 0.01
    assert max(held) > 3 * max(fitted), (fitted, held)


def test_atom_is_split_not_collapsed(synthetic):
    """p_AC has a large atom at exactly 1.0. A deterministic quantile map
    sends every 1.0 to a single value; the corrected atom must instead match
    the observed atom's size."""
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    out = c.apply(null_rc, null_p, seed=SEED)
    atom_in = float((null_p == 1.0).mean())
    atom_out = float((out == 1.0).mean())
    atom_target = float((obs_p == 1.0).mean())
    assert atom_out < atom_in, "the atom was not shrunk at all"
    assert abs(atom_out - atom_target) < 0.05, (atom_out, atom_target)
    # The mapped values must not all land on one point.
    assert len(np.unique(out[null_p == 1.0])) > 100


def test_apply_is_reproducible_and_seed_dependent(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    a = c.apply(null_rc, null_p, seed=1)
    b = c.apply(null_rc, null_p, seed=1)
    d = c.apply(null_rc, null_p, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, d)


def test_only_signal_free_bins_are_fitted_by_default(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p,
                                 signal_free_rc_quantile=0.5)
    assert len(c.fitted_bins) < c.n_bins, "every bin was fitted; nothing held back"
    assert c.fitted_bins.max() < c.n_bins - 1


def test_held_out_bins_do_not_influence_the_fit(synthetic):
    """Corrupting a held-out bin's observed values must not change the map --
    otherwise the 'signal-free fit' claim is false."""
    obs_rc, obs_p, null_rc, null_p = synthetic
    fit_bins = [0, 1, 2]
    base = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p, fit_bins=fit_bins)
    edges = rc_bin_edges(obs_rc, 10)
    corrupt = obs_p.copy()
    corrupt[assign_bins(obs_rc, edges) >= 5] = 0.0
    other = fit_posterior_correction(null_rc, null_p, obs_rc, corrupt, fit_bins=fit_bins)
    assert np.allclose(base.target_quantiles, other.target_quantiles)
    assert np.allclose(base.pooled_target, other.pooled_target)


def test_unfitted_bins_use_the_pooled_map(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p, fit_bins=[0, 1])
    high = c.n_bins - 1
    assert high not in set(c.fitted_bins.tolist())
    rc = np.full(500, 0.95)
    p = np.linspace(0.5, 1.0, 500)
    assert np.all(np.isfinite(c.apply(rc, p, seed=SEED)))


def test_corrected_score_leaves_ac_plus1_untouched(synthetic):
    """AC=+1 scores are RC - p_AC <= 1, far below any threshold, so they are
    deliberately left alone rather than given randomization noise."""
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    ac = np.where(np.arange(len(null_rc)) % 2 == 0, 1.0, -1.0)
    got = corrected_null_score(null_rc, ac, null_p, c, seed=SEED)
    plus = ac == 1.0
    assert np.allclose(got[plus], null_rc[plus] - null_p[plus])
    assert not np.allclose(got[~plus], null_rc[~plus] + null_p[~plus])


def test_shape_mismatch_is_rejected(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    with pytest.raises(ValueError, match="same tests"):
        c.apply(null_rc[:-1], null_p, seed=SEED)


def test_roundtrip_through_dict(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    c = fit_posterior_correction(null_rc, null_p, obs_rc, obs_p)
    r = PosteriorCorrection.from_dict(c.to_dict())
    assert np.array_equal(r.fitted_bins, c.fitted_bins)
    assert np.allclose(
        r.apply(null_rc, null_p, seed=SEED), c.apply(null_rc, null_p, seed=SEED)
    )


def test_refuses_when_no_bin_has_enough_data(synthetic):
    obs_rc, obs_p, null_rc, null_p = synthetic
    with pytest.raises(ValueError, match="cannot fit a correction"):
        fit_posterior_correction(null_rc[:10], null_p[:10], obs_rc[:10], obs_p[:10],
                                 min_per_bin=200)
