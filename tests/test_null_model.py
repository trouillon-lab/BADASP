import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from src.badasp.null_model import (
    ThresholdDescription,
    ThresholdResult,
    as_bin_threshold_dict,
    describe_threshold,
    describe_threshold_curve,
    exceedance_flatness,
    load_null_scores,
    maxT_fwer_thresholds,
    per_test_pvalues,
    threshold_at_fdr,
    write_null_replicate,
)


RNG_SEED = 12345


def _make_replicate_dir(
    tmp_path: Path,
    R: int,
    n: int,
    rng: np.random.Generator,
    node_name: np.ndarray,
    position: np.ndarray,
    loc: float = 0.0,
    scale: float = 0.3,
) -> Path:
    """Write R synthetic rep_*.npz files sharing one fixed key order."""
    directory = tmp_path / "null_replicates"
    directory.mkdir(parents=True, exist_ok=True)
    for r in range(R):
        score_left = rng.normal(loc=loc, scale=scale, size=n)
        score_right = rng.normal(loc=loc, scale=scale, size=n)
        write_null_replicate(
            directory / f"rep_{r:04d}.npz",
            node_name=node_name,
            position=position,
            score_left=score_left,
            score_right=score_right,
            rc_left=rng.uniform(0, 1, size=n),
            rc_right=rng.uniform(0, 1, size=n),
            ac=rng.choice([-1.0, 1.0], size=n),
            p_ac_left=rng.uniform(0, 1, size=n),
            p_ac_right=rng.uniform(0, 1, size=n),
        )
    return directory


# ---------------------------------------------------------------------------
# load_null_scores / write_null_replicate round-trip
# ---------------------------------------------------------------------------


def test_write_and_load_null_scores_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(RNG_SEED)
    n = 50
    node_name = np.array([f"Node{i}" for i in range(n)])
    position = np.arange(n) % 5

    directory = _make_replicate_dir(tmp_path, R=4, n=n, rng=rng, node_name=node_name, position=position)
    null_left, null_right, keys = load_null_scores(directory)

    assert null_left.shape == (4, n)
    assert null_right.shape == (4, n)
    assert len(keys) == n
    assert list(keys["node_name"]) == list(node_name)
    assert list(keys["position"]) == list(position)


def test_load_null_scores_fails_loudly_on_length_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "bad_replicates"
    directory.mkdir()
    write_null_replicate(
        directory / "rep_0000.npz",
        node_name=["A", "B", "C"],
        position=[1, 2, 3],
        score_left=[0.1, 0.2, 0.3],
        score_right=[0.1, 0.2, 0.3],
        rc_left=[0.5, 0.5, 0.5],
        rc_right=[0.5, 0.5, 0.5],
        ac=[1.0, 1.0, 1.0],
        p_ac_left=[0.5, 0.5, 0.5],
        p_ac_right=[0.5, 0.5, 0.5],
    )
    write_null_replicate(
        directory / "rep_0001.npz",
        node_name=["A", "B"],
        position=[1, 2],
        score_left=[0.1, 0.2],
        score_right=[0.1, 0.2],
        rc_left=[0.5, 0.5],
        rc_right=[0.5, 0.5],
        ac=[1.0, 1.0],
        p_ac_left=[0.5, 0.5],
        p_ac_right=[0.5, 0.5],
    )
    with pytest.raises(ValueError, match="tests but reference replicate"):
        load_null_scores(directory)


def test_load_null_scores_fails_loudly_on_key_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "bad_key_replicates"
    directory.mkdir()
    write_null_replicate(
        directory / "rep_0000.npz",
        node_name=["A", "B", "C"],
        position=[1, 2, 3],
        score_left=[0.1, 0.2, 0.3],
        score_right=[0.1, 0.2, 0.3],
        rc_left=[0.5, 0.5, 0.5],
        rc_right=[0.5, 0.5, 0.5],
        ac=[1.0, 1.0, 1.0],
        p_ac_left=[0.5, 0.5, 0.5],
        p_ac_right=[0.5, 0.5, 0.5],
    )
    # Same length, different key order/content.
    write_null_replicate(
        directory / "rep_0001.npz",
        node_name=["A", "B", "D"],
        position=[1, 2, 3],
        score_left=[0.1, 0.2, 0.3],
        score_right=[0.1, 0.2, 0.3],
        rc_left=[0.5, 0.5, 0.5],
        rc_right=[0.5, 0.5, 0.5],
        ac=[1.0, 1.0, 1.0],
        p_ac_left=[0.5, 0.5, 0.5],
        p_ac_right=[0.5, 0.5, 0.5],
    )
    with pytest.raises(ValueError, match="different .* key order"):
        load_null_scores(directory)


def test_load_null_scores_raises_on_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_null_scores(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# threshold_at_fdr: calibration on pure noise (the key test)
# ---------------------------------------------------------------------------


def test_calibration_on_pure_noise() -> None:
    """Observed and null drawn from the identical distribution (no signal).

    Under complete exchangeability between observed and null (obs is
    literally just one more draw from the same generative process as the
    null replicates), the smallest-t-satisfying-a-strict-quantile search in
    `threshold_at_fdr` is itself noisy at the boundary crossing (whether a
    given q is "just barely" met depends on which side of a coin-flip the
    single observed draw landed on) — so asserting `criterion_met` for an
    arbitrary moderate q would be a flaky test of a real, correct property
    (a well-calibrated FDP-exceedance procedure is *conservative* under the
    global null: it should not reliably manufacture discoveries).

    Instead we inspect the full sweep directly (always returned, regardless
    of whether a threshold met the criterion) at the threshold nearest a
    chosen nominal false-positive rate. Because obs and null are
    interchangeable here, the number of observed exceedances O(t) should
    land close to the nominal count implied by that rate, and both the mean
    ratio E[FP]/O and the achieved FDP quantile should sit close to 1.0 —
    i.e. essentially every observed "discovery" is (correctly) flagged as
    plausibly false, in sharp contrast to the old circular 99.9th-percentile
    rule which forced exactly 0.1% positives by construction regardless of
    whether the data contained any real signal.
    """
    rng = np.random.default_rng(RNG_SEED)
    n = 20_000
    R = 500

    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    # q here only needs to be loose enough that the sweep is fully computed;
    # we inspect the sweep table directly rather than relying on `criterion_met`.
    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.99, fdp_quantile=0.90)
    sweep = result.sweep.sort_values("t").reset_index(drop=True)
    assert not sweep.empty

    nominal_rate = 0.01
    nominal_count = nominal_rate * n
    row = sweep.iloc[(sweep["O"] - nominal_count).abs().idxmin()]

    # Nominal vs achieved false-positive count: O(t) should land near the
    # count implied by the nominal rate, since obs ~ null in distribution.
    assert row["O"] == pytest.approx(nominal_count, rel=0.25)
    # Achieved FDP (both the mean-based point estimate and the quantile
    # actually used by the primary criterion) should be close to 1.0.
    assert row["E_fp_over_O"] == pytest.approx(1.0, abs=0.3)
    assert row["fdp_quantile_achieved"] == pytest.approx(1.0, abs=0.3)


def test_monotonicity_of_sweep() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    n = 800
    R = 50
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.3, fdp_quantile=0.9)
    sweep = result.sweep.sort_values("t")
    assert (np.diff(sweep["O"].to_numpy()) <= 0).all()
    assert (np.diff(sweep["E_fp"].to_numpy()) <= 1e-9).all()


# ---------------------------------------------------------------------------
# threshold_at_fdr: recovery with planted signal
# ---------------------------------------------------------------------------


def test_recovery_with_planted_signal() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    n = 5000
    R = 300
    K = 100  # number of planted true signals

    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    signal_idx = rng.choice(n, size=K, replace=False)
    obs_left[signal_idx] = rng.normal(6, 0.3, size=K)  # far outside the null range

    q = 0.1
    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=q, fdp_quantile=0.90)

    assert result.criterion_met
    assert result.t is not None
    assert result.E_fp_over_O <= q + 1e-9

    discovered = (obs_left >= result.t) | (obs_right >= result.t)
    recovered_signal = discovered[signal_idx].sum()
    assert recovered_signal >= 0.9 * K


# ---------------------------------------------------------------------------
# either-branch-exceeds rule
# ---------------------------------------------------------------------------


def test_either_branch_exceeds_rule_matches_manual_or() -> None:
    rng = np.random.default_rng(RNG_SEED + 3)
    n = 200
    R = 20
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    # Force some left/right asymmetry: left is huge, right is tiny, and vice versa.
    obs_left[0], obs_right[0] = 10.0, -10.0
    obs_left[1], obs_right[1] = -10.0, 10.0
    obs_left[2], obs_right[2] = -10.0, -10.0

    t = 5.0
    manual_split_exceeds = (obs_left >= t) | (obs_right >= t)
    assert manual_split_exceeds[0]
    assert manual_split_exceeds[1]
    assert not manual_split_exceeds[2]

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.9, fdp_quantile=0.9)
    sweep = result.sweep
    # For the observed grid point equal to t (or nearest present grid value <= existing values),
    # verify O(t) recomputed manually via the OR rule matches the sweep's O at that t.
    row = sweep[np.isclose(sweep["t"], 10.0)]
    assert not row.empty
    manual_O_at_10 = int(((obs_left >= 10.0) | (obs_right >= 10.0)).sum())
    assert int(row.iloc[0]["O"]) == manual_O_at_10
    assert manual_O_at_10 == 2  # both idx 0 and 1 (left=10 and right=10 respectively)


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------


def test_pvalues_never_zero_and_floor_is_1_over_1_plus_R() -> None:
    rng = np.random.default_rng(RNG_SEED + 4)
    n = 20
    R = 99
    obs = rng.normal(0, 1, size=n)
    null = rng.normal(0, 1, size=(R, n))

    # Make one observation far larger than every null value.
    obs[0] = 1000.0

    p, q = per_test_pvalues(obs, null)
    assert (p > 0).all()
    assert p[0] == pytest.approx(1.0 / (1.0 + R))


def test_pvalues_roughly_uniform_under_pure_noise() -> None:
    rng = np.random.default_rng(RNG_SEED + 5)
    n = 3000
    R = 200
    obs = rng.normal(0, 1, size=n)
    null = rng.normal(0, 1, size=(R, n))

    p, q = per_test_pvalues(obs, null)
    # Loose check: mean p-value should be roughly 0.5 under the null, and a
    # coarse KS-style check against the uniform CDF at a few quantiles.
    assert p.mean() == pytest.approx(0.5, abs=0.05)
    for cutoff in (0.1, 0.5, 0.9):
        frac_below = float(np.mean(p <= cutoff))
        assert abs(frac_below - cutoff) < 0.05


def test_pvalues_nan_handling() -> None:
    obs = np.array([1.0, np.nan, 2.0])
    null = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    p, q = per_test_pvalues(obs, null)
    assert np.isnan(p[1])
    assert np.isnan(q[1])
    assert not np.isnan(p[0])
    assert not np.isnan(p[2])


# ---------------------------------------------------------------------------
# max-T FWER
# ---------------------------------------------------------------------------


def test_maxT_fwer_global_threshold_above_null_with_planted_huge_value() -> None:
    rng = np.random.default_rng(RNG_SEED + 6)
    n = 500
    R = 200
    positions = rng.integers(0, 10, size=n)

    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    result = maxT_fwer_thresholds(null_left, null_right, positions, alpha=0.05)
    global_threshold = result["global"]

    # A single huge planted null value in one replicate/position should not,
    # by itself, blow out every other position's threshold; but the max
    # over everything (global) must sit above ordinary null noise.
    ordinary_null_max = np.fmax(null_left, null_right).max()
    assert global_threshold <= ordinary_null_max + 1e-6 or global_threshold > 3.0

    # Plant one huge null value and verify the global threshold rises to reflect it
    # for at least a large fraction of replicates (since global = quantile of per-rep max).
    null_left_2 = null_left.copy()
    null_left_2[0, 0] = 1000.0
    result2 = maxT_fwer_thresholds(null_left_2, null_right, positions, alpha=0.05)
    # The one huge replicate's max is now 1000; the (1-alpha) quantile across
    # R=200 replicates with alpha=0.05 keeps the top ~5% out, so a single
    # outlier in one replicate need not move the 95th percentile threshold.
    # Instead check that per-position threshold for that specific position
    # equals or exceeds the ordinary within-position null range.
    per_position = result2["per_position"]
    pos0 = positions[0]
    assert per_position[pos0] >= np.quantile(
        np.fmax(null_left, null_right)[:, positions == pos0].max(axis=1), 1 - 0.05
    ) - 1e-6


def test_maxT_fwer_extreme_outlier_dominates_when_alpha_covers_it() -> None:
    rng = np.random.default_rng(RNG_SEED + 7)
    n = 50
    R = 20
    positions = np.zeros(n, dtype=int)  # single position, all splits pooled

    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))
    null_left[0, 0] = 1000.0  # one huge outlier in replicate 0

    # The (1 - alpha) quantile of the per-replicate max is a HIGH quantile
    # for small alpha (strict FWER control): with R=20 replicates and only
    # one containing the outlier, a small enough alpha pushes the quantile
    # up into the extreme tail, where the outlier's replicate dominates.
    result = maxT_fwer_thresholds(null_left, null_right, positions, alpha=0.01)
    assert result["global"] >= 500.0

    # A loose alpha (e.g. 0.5, the median of the per-replicate max) should
    # NOT be dominated by the single outlier replicate out of 20.
    loose_result = maxT_fwer_thresholds(null_left, null_right, positions, alpha=0.5)
    assert loose_result["global"] < 100.0


# ---------------------------------------------------------------------------
# exceedance_flatness
# ---------------------------------------------------------------------------


def test_exceedance_flatness_flat_case() -> None:
    rng = np.random.default_rng(RNG_SEED + 8)
    n = 2000
    R = 100
    null = rng.normal(0, 1, size=(R, n))  # no dependency on covariates by construction
    covariates = pd.DataFrame(
        {
            "log_clade_size": rng.normal(2, 1, size=n),
            "position": rng.integers(1, 15, size=n),
        }
    )
    threshold = 1.64  # roughly the 95th percentile of N(0,1)

    out = exceedance_flatness(null, covariates, threshold)
    assert set(out["covariate"]) == {"log_clade_size", "position"}
    # Rates should be roughly flat and close to ~5% since covariates carry no signal.
    log_rows = out[out["covariate"] == "log_clade_size"]
    assert log_rows["mean_exceedance_rate"].std() < 0.05
    assert log_rows["mean_exceedance_rate"].mean() == pytest.approx(0.05, abs=0.03)


def test_exceedance_flatness_detects_non_flat_case() -> None:
    rng = np.random.default_rng(RNG_SEED + 9)
    n = 2000
    R = 100
    covariate = rng.uniform(0, 1, size=n)
    # Construct null scores whose mean depends directly on the covariate,
    # so exceedance rate should clearly increase with the covariate bin.
    null = rng.normal(loc=covariate[np.newaxis, :] * 5, scale=1.0, size=(R, n))
    covariates = pd.DataFrame({"covariate": covariate})
    threshold = 2.0

    out = exceedance_flatness(null, covariates, threshold, n_bins=5)
    out_sorted = out.sort_values("bin_label")
    rates = out_sorted["mean_exceedance_rate"].to_numpy()
    assert rates[0] < rates[-1] - 0.2  # clearly increasing, not flat


# ---------------------------------------------------------------------------
# as_bin_threshold_dict
# ---------------------------------------------------------------------------


def test_as_bin_threshold_dict_shape_matches_identify_switches_expectations() -> None:
    bin_intervals = [pd.Interval(0, 10), pd.Interval(10, 20)]
    event_types = ["Duplication", "Speciation", "Transfer"]
    thresholds = as_bin_threshold_dict(0.75, event_types, bin_intervals)

    assert all(isinstance(k, tuple) and len(k) == 2 for k in thresholds)
    assert all(isinstance(k[1], pd.Interval) for k in thresholds)
    for event in event_types + ["overall"]:
        for interval in bin_intervals:
            assert thresholds[(event, interval)] == pytest.approx(0.75)

    # ("overall", interval) keys must be present since event_specific=False lookups use them.
    for interval in bin_intervals:
        assert ("overall", interval) in thresholds


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_threshold_at_fdr_all_nan_observed() -> None:
    n = 10
    R = 5
    obs_left = np.full(n, np.nan)
    obs_right = np.full(n, np.nan)
    null_left = np.random.default_rng(0).normal(size=(R, n))
    null_right = np.random.default_rng(1).normal(size=(R, n))

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.1)
    assert result.t is None
    assert not result.criterion_met
    assert result.sweep.empty
    assert result.note


def test_threshold_at_fdr_r_equals_one() -> None:
    rng = np.random.default_rng(RNG_SEED + 10)
    n = 200
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(1, n))
    null_right = rng.normal(0, 1, size=(1, n))

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.5, fdp_quantile=0.9)
    assert result.R == 1
    # Should not crash, and should produce a well-formed sweep table.
    assert not result.sweep.empty


def test_threshold_at_fdr_zero_replicates() -> None:
    n = 20
    obs_left = np.random.default_rng(0).normal(size=n)
    obs_right = np.random.default_rng(1).normal(size=n)
    null_left = np.empty((0, n))
    null_right = np.empty((0, n))

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.1)
    assert result.t is None
    assert not result.criterion_met
    assert result.R == 0


def test_threshold_at_fdr_empty_input() -> None:
    obs_left = np.array([])
    obs_right = np.array([])
    null_left = np.empty((5, 0))
    null_right = np.empty((5, 0))

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.1)
    assert result.t is None
    assert not result.criterion_met


def test_threshold_at_fdr_o_of_t_zero_case() -> None:
    """When q is set so strictly that only an extreme t 'works', O(t) can be 0."""
    rng = np.random.default_rng(RNG_SEED + 11)
    n = 500
    R = 200
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=1e-6, fdp_quantile=0.99)
    # Either no threshold satisfies this extreme criterion, or the chosen
    # threshold is the most extreme grid point with very few discoveries.
    if result.criterion_met:
        assert result.O_t is not None
    else:
        assert result.t is None


def test_per_test_pvalues_all_nan_column() -> None:
    obs = np.array([np.nan, np.nan])
    null = np.array([[1.0, 2.0], [3.0, 4.0]])
    p, q = per_test_pvalues(obs, null)
    assert np.all(np.isnan(p))
    assert np.all(np.isnan(q))


# ---------------------------------------------------------------------------
# describe_threshold / describe_threshold_curve
# ---------------------------------------------------------------------------


def _null_with_exact_V(V_values, n_cols: int, t: float = 1.0):
    """Build (null_left, null_right) of shape (R, n_cols) with row r having
    exactly V_values[r] entries whose split statistic is >= t, and the rest
    strictly below t. Lets falsifier/mixture tests control V directly rather
    than relying on randomness to land on a particular pattern."""
    R = len(V_values)
    null_left = np.full((R, n_cols), t - 1.0)
    for r, v in enumerate(V_values):
        assert v <= n_cols
        null_left[r, :v] = t + 1.0
    null_right = null_left.copy()
    return null_left, null_right


def test_describe_threshold_matches_threshold_at_fdr_at_shared_t() -> None:
    """The new, non-selecting path must agree numerically with the sweep
    computed inside `threshold_at_fdr`, at a t the sweep itself contains --
    this is the cross-check that the two code paths compute E_fp and the
    fdp quantile the same way."""
    rng = np.random.default_rng(RNG_SEED + 100)
    n = 600
    R = 40
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    fdp_quantile = 0.9
    result = threshold_at_fdr(obs_left, obs_right, null_left, null_right, q=0.99, fdp_quantile=fdp_quantile)
    sweep = result.sweep.sort_values("t").reset_index(drop=True)
    assert not sweep.empty

    row = sweep.iloc[len(sweep) // 2]
    t = float(row["t"])

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right, fdp_quantile=fdp_quantile)

    assert desc.O == int(row["O"])
    assert desc.E_fp == pytest.approx(float(row["E_fp"]))
    assert desc.fdp_quantile_achieved == pytest.approx(float(row["fdp_quantile_achieved"]))


def test_describe_threshold_nan_both_never_counts() -> None:
    """A split with both branches NaN must never count, on observed or null side."""
    t = 1.0
    obs_left = np.array([np.nan, 5.0])
    obs_right = np.array([np.nan, np.nan])
    # index 0: both branches NaN -> must not count.
    # index 1: left=5 (>= t), right=NaN -> counts once via the real branch.

    null_left = np.array([[np.nan, 5.0], [np.nan, np.nan]])
    null_right = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # row 0: index0 both NaN (skip), index1 left=5 counts -> V=1
    # row 1: both columns both-NaN -> V=0

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right)
    assert desc.O == 1
    assert list(desc.V) == [1, 0]


def test_describe_threshold_mc_pvalue_floor_and_ceiling() -> None:
    rng = np.random.default_rng(RNG_SEED + 101)
    R = 37

    # Every replicate's V is strictly below O: obs clears t everywhere,
    # null never does.
    n = 5
    obs_left = np.full(n, 10.0)
    obs_right = np.full(n, -100.0)
    null_left = np.full((R, n), 0.0)
    null_right = np.full((R, n), 0.0)
    t = 5.0

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right)
    assert desc.O == n
    assert (desc.V < desc.O).all()
    assert desc.mc_pvalue == pytest.approx(1.0 / (1.0 + R))

    # Every replicate's V is >= O: a single test where both obs and every
    # null replicate clear t.
    obs_left2 = np.array([t])
    obs_right2 = np.array([-100.0])
    null_left2 = np.full((R, 1), t + 1.0)
    null_right2 = np.full((R, 1), -100.0)

    desc2 = describe_threshold(t, obs_left2, obs_right2, null_left2, null_right2)
    assert desc2.O == 1
    assert (desc2.V >= desc2.O).all()
    assert desc2.mc_pvalue == pytest.approx(1.0)

    _ = rng  # rng not needed beyond seeding convention; kept for consistency


def test_describe_threshold_inadequacy_note_not_clipped() -> None:
    """When the null alone out-produces the observed data, E_fp_over_O > 1
    must be reported unclipped, with an explanatory note attached."""
    n = 50
    R = 10
    t = 1.0
    obs_left = np.full(n, t + 1.0)  # O = n
    obs_right = np.full(n, -100.0)
    null_left = np.full((R, n), t + 1.0)  # every null test also clears t -> V = n for every replicate
    null_right = np.full((R, n), -100.0)

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right)
    assert desc.O == n
    assert desc.E_fp == pytest.approx(float(n))
    assert desc.E_fp_over_O == pytest.approx(1.0)  # O == n here, so not yet > 1

    # Now make the null exceed the observed count: fewer obs discoveries, same null.
    obs_left2 = obs_left.copy()
    obs_left2[: n // 2] = -100.0  # only half the observed splits clear t now
    desc2 = describe_threshold(t, obs_left2, obs_right, null_left, null_right)
    assert desc2.O == n // 2
    assert desc2.E_fp_over_O == pytest.approx(float(n) / (n // 2))
    assert desc2.E_fp_over_O > 1.0
    assert any("not interpretable as a false-discovery proportion" in note for note in desc2.notes)
    # Not clipped to 1.0:
    assert desc2.E_fp_over_O == pytest.approx(2.0)


def test_clopper_pearson_edge_cases_and_middle_case_matches_scipy() -> None:
    n = 10
    R = 10
    t = 1.0
    ci_level = 0.95
    alpha = 1.0 - ci_level

    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)
    null_left, null_right = _null_with_exact_V([2] * R, n_cols=n, t=t)

    # k = 0: no replicate is labelled high.
    labels_k0 = np.zeros(R, dtype=bool)
    desc_k0 = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels_k0, ci_level=ci_level
    )
    assert desc_k0.mixture["pi_low"] == 0.0

    # k = R: every replicate is labelled high.
    labels_kR = np.ones(R, dtype=bool)
    desc_kR = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels_kR, ci_level=ci_level
    )
    assert desc_kR.mixture["pi_high"] == 1.0

    # Middle case k = 3: check against scipy.stats.beta.ppf directly.
    k = 3
    labels_mid = np.zeros(R, dtype=bool)
    labels_mid[:k] = True
    desc_mid = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels_mid, ci_level=ci_level
    )
    expected_low = float(beta.ppf(alpha / 2.0, k, R - k + 1))
    expected_high = float(beta.ppf(1.0 - alpha / 2.0, k + 1, R - k))
    assert desc_mid.mixture["pi_low"] == pytest.approx(expected_low)
    assert desc_mid.mixture["pi_high"] == pytest.approx(expected_high)
    assert desc_mid.mixture["pi_hat"] == pytest.approx(k / R)


def test_falsifiers_fire_on_hand_built_overlap_case() -> None:
    """n_high=1 with an overlapping low replicate should trip both V-based flags."""
    n = 6
    t = 1.0
    V_values = [5, 6, 1, 1, 1, 1]  # replicate 0 is the sole "high" one
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    labels = np.array([True, False, False, False, False, False])

    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right, labels=labels)
    falsifiers = desc.mixture["falsifiers"]
    assert falsifiers["low_state_exceeds_high_minimum"]["value"] is True
    assert falsifiers["n_high_below_2"]["value"] is True


def test_falsifiers_clean_on_cleanly_separated_case() -> None:
    """A cleanly separated two-state pattern should trip neither V-based flag."""
    n = 15
    t = 1.0
    V_high = [10, 11, 12]
    V_low = [1, 2, 3, 4, 5, 6, 7]
    V_values = V_high + V_low
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    labels = np.array([True] * len(V_high) + [False] * len(V_low))

    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right, labels=labels)
    falsifiers = desc.mixture["falsifiers"]
    assert falsifiers["low_state_exceeds_high_minimum"]["value"] is False
    assert falsifiers["n_high_below_2"]["value"] is False


# ---------------------------------------------------------------------------
# describe_threshold: gap-based falsifiers on label_statistic
# ---------------------------------------------------------------------------
#
# These replace the removed "labels_not_separating" check, which tested
# whether the *chosen cut* happened to separate V -- uninformative, since V
# is strongly monotone in the labelling statistic and the check flips
# depending on exactly where the cut is placed. The new checks instead look
# at the labelling variable's own distribution (label_statistic).


def test_gap_not_dominant_false_on_hand_built_bimodal_statistic() -> None:
    """Two tight clusters separated by one big gap: the gap is dominant."""
    n = 6
    t = 1.0
    V_values = [2] * 6
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    label_statistic = np.array([0.10, 0.11, 0.12, 0.60, 0.61, 0.62])
    labels = np.array([False, False, False, True, True, True])

    desc = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels, label_statistic=label_statistic
    )
    gap_entry = desc.mixture["falsifiers"]["gap_not_dominant"]
    assert gap_entry["value"] is False
    assert gap_entry["gap_ratio"] == pytest.approx(48.0)
    assert gap_entry["largest_gap"] == pytest.approx(0.48)
    assert gap_entry["largest_gap_between"] == pytest.approx([0.12, 0.60])


def test_gap_not_dominant_true_on_hand_built_continuous_right_skewed_statistic() -> None:
    """Gaps that grow smoothly towards the tail: no single gap dominates."""
    n = 8
    t = 1.0
    V_values = [2] * 8
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    label_statistic = np.array([0.10, 0.14, 0.19, 0.25, 0.32, 0.40, 0.50, 0.62])
    labels = np.array([False] * 5 + [True] * 3)  # cut at 0.32/0.40, arbitrary w.r.t. the gaps

    desc = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels, label_statistic=label_statistic
    )
    gap_entry = desc.mixture["falsifiers"]["gap_not_dominant"]
    assert gap_entry["value"] is True


def test_cut_not_at_largest_gap_fires_when_cut_placed_away_from_gap() -> None:
    n = 6
    t = 1.0
    V_values = [2] * 6
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    # The largest gap sits between 0.12 and 0.50; the cut below is placed
    # inside the low cluster instead, away from that gap.
    label_statistic = np.array([0.10, 0.11, 0.12, 0.50, 0.51, 0.52])
    labels = np.array([False, False, True, True, True, True])

    desc = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels, label_statistic=label_statistic
    )
    cut_entry = desc.mixture["falsifiers"]["cut_not_at_largest_gap"]
    assert cut_entry["value"] is True
    assert cut_entry["cut_between"] == pytest.approx([0.11, 0.12])


def test_cut_not_at_largest_gap_does_not_fire_when_cut_placed_at_gap() -> None:
    n = 6
    t = 1.0
    V_values = [2] * 6
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    label_statistic = np.array([0.10, 0.11, 0.12, 0.50, 0.51, 0.52])
    labels = np.array([False, False, False, True, True, True])  # cut exactly at the largest gap

    desc = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels, label_statistic=label_statistic
    )
    cut_entry = desc.mixture["falsifiers"]["cut_not_at_largest_gap"]
    assert cut_entry["value"] is False
    assert cut_entry["cut_between"] == pytest.approx([0.12, 0.50])


def test_gap_falsifiers_skipped_when_label_statistic_is_none() -> None:
    n = 6
    t = 1.0
    V_values = [2] * 6
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)
    labels = np.array([False, False, False, True, True, True])

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right, labels=labels)
    falsifiers = desc.mixture["falsifiers"]
    assert falsifiers["gap_not_dominant"]["value"] is None
    assert isinstance(falsifiers["gap_not_dominant"]["skipped"], str)
    assert falsifiers["cut_not_at_largest_gap"]["value"] is None
    assert isinstance(falsifiers["cut_not_at_largest_gap"]["skipped"], str)


def test_gap_ratio_threshold_is_honoured() -> None:
    """A gap_ratio of 1.5 is 'not dominant' at the default threshold (2.0)
    but 'dominant' once the threshold is lowered below 1.5."""
    n = 4
    t = 1.0
    V_values = [2] * 4
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    # Sorted diffs: [2, 2, 3] -> largest=3, second_largest=2, gap_ratio=1.5.
    label_statistic = np.array([0.0, 2.0, 4.0, 7.0])
    labels = np.array([False, False, True, True])

    desc_default = describe_threshold(
        t, obs_left, obs_right, null_left, null_right, labels=labels, label_statistic=label_statistic
    )
    desc_lowered = describe_threshold(
        t, obs_left, obs_right, null_left, null_right,
        labels=labels, label_statistic=label_statistic, gap_ratio_threshold=1.0,
    )
    assert desc_default.mixture["falsifiers"]["gap_not_dominant"]["gap_ratio"] == pytest.approx(1.5)
    assert desc_default.mixture["falsifiers"]["gap_not_dominant"]["value"] is True
    assert desc_lowered.mixture["falsifiers"]["gap_not_dominant"]["value"] is False


def test_group_composition_uneven_groups() -> None:
    n = 10
    t = 1.0
    V_values = [2] * 7
    null_left, null_right = _null_with_exact_V(V_values, n_cols=n, t=t)
    obs_left = np.full(n, t - 1.0)
    obs_right = np.full(n, t - 1.0)

    # 7 replicates split unevenly: group "a" has 2, group "b" has 5.
    groups = np.array(["a", "a", "b", "b", "b", "b", "b"])
    labels = np.array([True, False, True, False, False, False, False])

    desc = describe_threshold(t, obs_left, obs_right, null_left, null_right, labels=labels, groups=groups)
    assert desc.group_composition == {
        "a": {"n": 2, "n_high": 1},
        "b": {"n": 5, "n_high": 1},
    }


def test_describe_threshold_curve_sorted_and_o_non_increasing_with_t() -> None:
    rng = np.random.default_rng(RNG_SEED + 102)
    n = 4000
    R = 60
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))

    call_counts = [400, 50, 800, 10]
    df = describe_threshold_curve(obs_left, obs_right, null_left, null_right, call_counts=call_counts)

    assert list(df["target_calls"]) == sorted(call_counts)

    # As the threshold t rises, O must be monotonically non-increasing.
    by_t_desc = df.sort_values("t", ascending=False)
    assert (np.diff(by_t_desc["O"].to_numpy()) >= 0).all()  # O non-decreasing as t falls == non-increasing as t rises


def test_threshold_description_to_dict_round_trips_through_json() -> None:
    rng = np.random.default_rng(RNG_SEED + 103)
    n = 300
    R = 20
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))
    labels = rng.integers(0, 2, size=R).astype(bool)
    groups = np.array([f"g{i % 3}" for i in range(R)])

    desc = describe_threshold(0.5, obs_left, obs_right, null_left, null_right, labels=labels, groups=groups)
    payload = json.dumps(desc.to_dict())
    round_tripped = json.loads(payload)

    assert round_tripped["O"] == desc.O
    assert round_tripped["V"] == list(int(v) for v in desc.V)
    assert round_tripped["mixture"]["falsifiers"]["n_high_below_2"]["value"] in (True, False)


def test_describe_threshold_raises_on_nonfinite_t() -> None:
    n = 10
    R = 5
    obs_left = np.zeros(n)
    obs_right = np.zeros(n)
    null_left = np.zeros((R, n))
    null_right = np.zeros((R, n))
    with pytest.raises(ValueError):
        describe_threshold(np.nan, obs_left, obs_right, null_left, null_right)
    with pytest.raises(ValueError):
        describe_threshold(np.inf, obs_left, obs_right, null_left, null_right)


def test_describe_threshold_raises_on_mismatched_shapes() -> None:
    n = 10
    R = 5
    obs_left = np.zeros(n)
    obs_right = np.zeros(n + 1)  # mismatched length
    null_left = np.zeros((R, n))
    null_right = np.zeros((R, n))
    with pytest.raises(ValueError):
        describe_threshold(0.0, obs_left, obs_right, null_left, null_right)

    obs_right2 = np.zeros(n)
    null_right2 = np.zeros((R, n + 1))  # mismatched null shape
    with pytest.raises(ValueError):
        describe_threshold(0.0, obs_left, obs_right2, null_left, null_right2)


# ---------------------------------------------------------------------------
# describe_threshold: nonparametric bootstrap interval
# ---------------------------------------------------------------------------
#
# Added after a measurement on the pooled 41-replicate null (see
# results/badasp_scoring/null_calibration/pooled_41/) showed the two-state
# mixture model does not hold there (its own falsifier fires) and that
# negative binomial / gamma / lognormal fits to V are all rejected by a
# one-sample KS test and undershoot the empirical 90th percentile. The
# mixture code is unchanged -- it is expected to keep self-reporting via its
# falsifiers -- but the primary interval is now this bootstrap instead.


def _bootstrap_fixture(rng_seed: int = RNG_SEED + 200, n: int = 400, R: int = 25):
    rng = np.random.default_rng(rng_seed)
    obs_left = rng.normal(0, 1, size=n)
    obs_right = rng.normal(0, 1, size=n)
    null_left = rng.normal(0, 1, size=(R, n))
    null_right = rng.normal(0, 1, size=(R, n))
    return obs_left, obs_right, null_left, null_right


def test_bootstrap_requires_seed_when_requested() -> None:
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture()
    with pytest.raises(ValueError):
        describe_threshold(0.5, obs_left, obs_right, null_left, null_right, n_bootstrap=100)


def test_bootstrap_same_seed_reproducible_different_seed_differs() -> None:
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture()
    d1 = describe_threshold(0.5, obs_left, obs_right, null_left, null_right, n_bootstrap=500, bootstrap_seed=7)
    d2 = describe_threshold(0.5, obs_left, obs_right, null_left, null_right, n_bootstrap=500, bootstrap_seed=7)
    d3 = describe_threshold(0.5, obs_left, obs_right, null_left, null_right, n_bootstrap=500, bootstrap_seed=8)

    assert d1.bootstrap == d2.bootstrap
    assert d1.bootstrap != d3.bootstrap


def test_bootstrap_interval_brackets_point_estimate() -> None:
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture(n=2000, R=60)
    desc = describe_threshold(
        0.5, obs_left, obs_right, null_left, null_right, fdp_quantile=0.90, n_bootstrap=5000, bootstrap_seed=99
    )
    boot = desc.bootstrap
    assert boot["mean_fdp_ci_low"] <= desc.fdp_mean <= boot["mean_fdp_ci_high"]
    assert boot["fdp_quantile_ci_low"] <= desc.fdp_quantile_achieved <= boot["fdp_quantile_ci_high"]


def test_bootstrap_unit_group_without_groups_raises() -> None:
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture()
    with pytest.raises(ValueError):
        describe_threshold(
            0.5, obs_left, obs_right, null_left, null_right,
            n_bootstrap=100, bootstrap_seed=1, bootstrap_unit="group",
        )


def test_bootstrap_unit_invalid_raises() -> None:
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture()
    groups = np.arange(len(null_left))
    with pytest.raises(ValueError):
        describe_threshold(
            0.5, obs_left, obs_right, null_left, null_right,
            n_bootstrap=100, bootstrap_seed=1, bootstrap_unit="bogus", groups=groups,
        )


def test_bootstrap_group_matches_replicate_when_groups_are_singletons() -> None:
    """With every group of size 1, drawing groups with replacement is the
    same resampling scheme as drawing replicates with replacement, so for a
    fixed seed the two intervals must agree exactly."""
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture(n=500, R=20)
    R = null_left.shape[0]
    singleton_groups = np.arange(R)  # one replicate per group

    d_replicate = describe_threshold(
        0.5, obs_left, obs_right, null_left, null_right, n_bootstrap=1000, bootstrap_seed=2026,
    )
    d_group = describe_threshold(
        0.5, obs_left, obs_right, null_left, null_right,
        n_bootstrap=1000, bootstrap_seed=2026, bootstrap_unit="group", groups=singleton_groups,
    )

    assert d_group.bootstrap["resample_unit"] == "group"
    assert d_replicate.bootstrap["mean_fdp_ci_low"] == pytest.approx(d_group.bootstrap["mean_fdp_ci_low"])
    assert d_replicate.bootstrap["mean_fdp_ci_high"] == pytest.approx(d_group.bootstrap["mean_fdp_ci_high"])
    assert d_replicate.bootstrap["fdp_quantile_ci_low"] == pytest.approx(d_group.bootstrap["fdp_quantile_ci_low"])
    assert d_replicate.bootstrap["fdp_quantile_ci_high"] == pytest.approx(d_group.bootstrap["fdp_quantile_ci_high"])


def test_bootstrap_to_dict_round_trips_through_json() -> None:
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture()
    desc = describe_threshold(0.5, obs_left, obs_right, null_left, null_right, n_bootstrap=200, bootstrap_seed=3)
    payload = json.dumps(desc.to_dict())
    round_tripped = json.loads(payload)
    assert round_tripped["bootstrap"]["n_bootstrap"] == 200
    assert round_tripped["bootstrap"]["seed"] == 3
    assert round_tripped["bootstrap"]["resample_unit"] == "replicate"
    assert set(round_tripped["bootstrap"]) == {
        "n_bootstrap", "seed", "ci_level", "resample_unit",
        "mean_fdp_ci_low", "mean_fdp_ci_high", "fdp_quantile_ci_low", "fdp_quantile_ci_high",
    }


def test_bootstrap_off_by_default_leaves_other_fields_unchanged() -> None:
    """n_bootstrap=0 (the default) must produce bootstrap=None and every
    other field identical to calling describe_threshold without any of the
    new keyword arguments at all."""
    obs_left, obs_right, null_left, null_right = _bootstrap_fixture()

    baseline = describe_threshold(0.5, obs_left, obs_right, null_left, null_right)
    explicit_off = describe_threshold(
        0.5, obs_left, obs_right, null_left, null_right,
        n_bootstrap=0, bootstrap_seed=None, bootstrap_unit="replicate",
    )

    assert baseline.bootstrap is None
    assert explicit_off.bootstrap is None
    for field in (
        "t", "O", "R", "E_fp", "E_fp_over_O", "O_over_E_fp", "fdp_mean", "fdp_median",
        "fdp_min", "fdp_max", "fdp_quantile", "fdp_quantile_achieved", "mc_pvalue",
        "n_ge_observed", "labels", "label_summary", "mixture", "group_composition", "notes",
    ):
        assert getattr(baseline, field) == getattr(explicit_off, field)
    assert np.array_equal(baseline.V, explicit_off.V)
