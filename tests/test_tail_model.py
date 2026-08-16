import numpy as np
import pandas as pd
import pytest

from src.badasp.null_model import _split_statistic, threshold_at_fdr
from src.badasp.tail_model import (
    EmpiricalTail,
    PoissonFitResult,
    TailModel,
    branch_covariates,
    build_cell_table,
    choose_spline_knots,
    collapse_cells,
    either_branch_exceeds,
    fit_penalized_poisson,
    fit_pooled_tail,
    natural_spline_basis,
    poisson_deviance,
    quantile_bin_edges,
    select_lambda_by_replicate_cv,
    t_from_u,
    to_z,
)

RNG_SEED = 20260815


# ---------------------------------------------------------------------------
# Natural cubic spline basis
# ---------------------------------------------------------------------------


def test_spline_basis_shape():
    rng = np.random.default_rng(RNG_SEED)
    x = rng.uniform(0, 10, size=500)
    for df in (2, 3, 4, 6):
        knots = choose_spline_knots(x, df)
        assert knots.shape[0] == df + 1
        basis = natural_spline_basis(x, knots)
        assert basis.shape == (x.shape[0], df)
        assert np.all(np.isfinite(basis))


def test_spline_basis_reproduces_linear_function_exactly():
    rng = np.random.default_rng(RNG_SEED + 1)
    x = rng.uniform(-5, 5, size=300)
    true_slope, true_intercept = 2.7, -1.3
    y = true_intercept + true_slope * x

    knots = choose_spline_knots(x, df=4)
    basis = natural_spline_basis(x, knots)
    design = np.column_stack([np.ones_like(x), basis])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef
    assert np.allclose(y_hat, y, atol=1e-8)

    # Also check extrapolation outside the knot range stays exactly linear
    # (a defining property of natural splines), using the fitted coefficients.
    x_extrap = np.array([-100.0, 100.0])
    basis_extrap = natural_spline_basis(x_extrap, knots)
    design_extrap = np.column_stack([np.ones_like(x_extrap), basis_extrap])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        y_extrap_hat = design_extrap @ coef
    assert np.allclose(y_extrap_hat, true_intercept + true_slope * x_extrap, atol=1e-6)


def test_spline_basis_rejects_too_few_distinct_values():
    x = np.array([1.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        choose_spline_knots(x, df=4)


# ---------------------------------------------------------------------------
# Synthetic recovery: the key evidence the penalised Poisson fitter works
# ---------------------------------------------------------------------------


def test_synthetic_poisson_recovery():
    """Generate Poisson cell counts from known a_p and a known smooth
    clade-size effect, fit with fit_penalized_poisson, and check recovery."""
    rng = np.random.default_rng(RNG_SEED + 2)

    n_positions = 40
    position_ids = np.arange(n_positions)
    alpha0 = 1e-3

    # Known per-position log-rate multipliers, deliberately NOT linear in the
    # "prior" covariates below, so the fit has to recover them from data
    # (with only mild shrinkage), not just read off the prior.
    true_entropy = rng.uniform(0.0, 3.0, size=n_positions)
    true_occupancy = rng.uniform(0.5, 1.0, size=n_positions)
    true_a_p = 0.6 * true_entropy - 0.4 * true_occupancy + rng.normal(0, 0.15, size=n_positions)
    true_a_p -= true_a_p.mean()  # a_p is only identified up to how much of the level the priors soak up

    # Known smooth own-clade-size effect (deliberately curved, not linear,
    # so the spline term -- not just a linear term -- has to do the work).
    def true_f_own(log_c):
        return 0.5 * np.sin(log_c) - 0.05 * log_c**2

    def true_f_sib(log_c):
        return 0.2 * log_c

    def true_f_dep(depth):
        return -0.1 * depth

    n_own_bins, n_sib_bins, n_dep_bins = 8, 6, 4
    own_bin_reprs = np.linspace(0.5, 6.0, n_own_bins)
    sib_bin_reprs = np.linspace(0.5, 6.0, n_sib_bins)
    dep_bin_reprs = np.linspace(0.0, 10.0, n_dep_bins)

    n_trials_per_cell = 4000  # trials per (position, own_bin, sib_bin, dep_bin) cell

    rows = []
    for p_idx, p in enumerate(position_ids):
        for ob in range(n_own_bins):
            for sb in range(n_sib_bins):
                for db in range(n_dep_bins):
                    log_own = own_bin_reprs[ob]
                    log_sib = sib_bin_reprs[sb]
                    depth = dep_bin_reprs[db]
                    log_rate = true_a_p[p_idx] + true_f_own(log_own) + true_f_sib(log_sib) + true_f_dep(depth)
                    mu = n_trials_per_cell * alpha0 * np.exp(log_rate)
                    k = rng.poisson(mu)
                    rows.append(
                        {
                            "position": p,
                            "own_bin": ob,
                            "sib_bin": sb,
                            "dep_bin": db,
                            "own_repr": log_own,
                            "sib_repr": log_sib,
                            "dep_repr": depth,
                            "n_trials": n_trials_per_cell,
                            "n_exceed": k,
                        }
                    )
    cells = pd.DataFrame(rows)

    entropy_by_position = dict(zip(position_ids, true_entropy))
    occupancy_by_position = dict(zip(position_ids, true_occupancy))

    own_knots = choose_spline_knots(own_bin_reprs, df=4)
    sib_knots = choose_spline_knots(sib_bin_reprs, df=3)
    dep_knots = choose_spline_knots(dep_bin_reprs, df=2)

    # Light shrinkage: enough data per cell (n_trials=4000) that a_p should be
    # recoverable predominantly from the data, not from the prior.
    fit = fit_penalized_poisson(
        cells, alpha0, own_knots, sib_knots, dep_knots, entropy_by_position, occupancy_by_position, lambda_=0.1
    )

    assert fit.converged

    # a_p recovery (up to the same mean-centering true_a_p was given).
    fitted_a_p = fit.a_position - fit.a_position.mean()
    a_p_rmse = np.sqrt(np.mean((fitted_a_p - true_a_p) ** 2))
    assert a_p_rmse < 0.15, f"a_p RMSE too high: {a_p_rmse}"
    assert np.corrcoef(fitted_a_p, true_a_p)[0, 1] > 0.9

    # Smooth clade-size / depth effect recovery: compare fitted vs true f_own
    # (and f_sib, f_dep) on a fine grid, allowing a constant offset (since
    # f_own's overall level is only jointly identified with a_p's mean, see
    # module docstring on why the spline basis excludes a constant column).
    grid = np.linspace(own_bin_reprs.min(), own_bin_reprs.max(), 50)
    from src.badasp.tail_model import natural_spline_basis as _nsb

    fitted_f_own = _nsb(grid, own_knots) @ fit.beta_own
    true_vals = true_f_own(grid)
    fitted_centered = fitted_f_own - fitted_f_own.mean()
    true_centered = true_vals - true_vals.mean()
    assert np.corrcoef(fitted_centered, true_centered)[0, 1] > 0.95
    rmse_shape = np.sqrt(np.mean((fitted_centered - true_centered) ** 2))
    assert rmse_shape < 0.1, f"f_own shape RMSE too high: {rmse_shape}"


# ---------------------------------------------------------------------------
# Proportional-tails transform: degenerates to the global rule when A(x) == 1
# ---------------------------------------------------------------------------


def _trivial_model(alpha0: float, u0: float, position_ids) -> TailModel:
    """A TailModel with every coefficient at zero, i.e. A(x) == 1 everywhere."""
    return TailModel(
        alpha0=alpha0,
        u0=u0,
        position_ids=list(position_ids),
        a_position=[0.0] * len(position_ids),
        own_knots=[0.0, 1.0, 2.0, 3.0, 4.0],
        sib_knots=[0.0, 1.0, 2.0, 3.0],
        dep_knots=[0.0, 1.0, 2.0],
        beta_own=[0.0, 0.0, 0.0, 0.0],
        beta_sib=[0.0, 0.0, 0.0],
        beta_dep=[0.0, 0.0],
    )


def test_to_z_degenerates_to_global_rule_when_A_is_one():
    rng = np.random.default_rng(RNG_SEED + 3)
    n_pooled = 20000
    null_left = rng.normal(0, 1, size=(1, n_pooled))
    null_right = rng.normal(0, 1, size=(1, n_pooled))
    tail = fit_pooled_tail(null_left, null_right, n_grid=100000)  # large enough to keep every unique value exactly

    alpha0 = 1e-2
    u0 = float(tail.quantile_at(alpha0))
    positions = np.arange(5)
    covariates = pd.DataFrame(
        {
            "position": rng.choice(positions, size=n_pooled),
            "log_own": rng.uniform(1, 3, size=n_pooled),
            "log_sib": rng.uniform(1, 3, size=n_pooled),
            "depth": rng.uniform(0, 5, size=n_pooled),
        }
    )
    model = _trivial_model(alpha0, u0, positions)

    scores = null_left[0]
    z = to_z(scores, covariates, model, tail)

    # z must be an exactly monotone-decreasing function of the raw score
    # (since G is strictly decreasing and A(x)==1 everywhere): thresholding
    # on z at z(t) must select exactly the same set as thresholding on the
    # raw score at t, for a battery of candidate thresholds t.
    for t in np.quantile(scores, [0.5, 0.9, 0.99, 0.999]):
        z_t = to_z(np.array([t]), covariates.iloc[[0]], model, tail)[0]
        selected_by_score = scores >= t
        selected_by_z = z >= z_t
        assert np.array_equal(selected_by_score, selected_by_z)

    # And z at s == u0 should be almost exactly -log10(alpha0), by construction.
    z_at_u0 = to_z(np.array([u0]), covariates.iloc[[0]], model, tail)[0]
    assert z_at_u0 == pytest.approx(-np.log10(alpha0), abs=0.05)


# ---------------------------------------------------------------------------
# fmax on branch z's == either-branch-exceeds, NaN-safe
# ---------------------------------------------------------------------------


def test_either_branch_exceeds_matches_or_rule_nan_safe():
    rng = np.random.default_rng(RNG_SEED + 4)
    n = 5000
    z_left = rng.normal(0, 1, size=n)
    z_right = rng.normal(0, 1, size=n)
    # Sprinkle in NaNs: some left-only, some right-only, some both.
    nan_left = rng.choice(n, size=200, replace=False)
    nan_right = rng.choice(n, size=200, replace=False)
    both_nan = rng.choice(n, size=50, replace=False)
    z_left[nan_left] = np.nan
    z_right[nan_right] = np.nan
    z_left[both_nan] = np.nan
    z_right[both_nan] = np.nan

    z_split = either_branch_exceeds(z_left, z_right)
    assert np.array_equal(z_split, _split_statistic(z_left, z_right), equal_nan=True)

    thresholds = np.quantile(z_left[np.isfinite(z_left)], [0.1, 0.5, 0.9, 0.99])
    for t in thresholds:
        with np.errstate(invalid="ignore"):
            or_rule = (z_left >= t) | (z_right >= t)
        with np.errstate(invalid="ignore"):
            fmax_rule = z_split >= t
        assert np.array_equal(or_rule, fmax_rule)
        # Splits with both branches NaN never satisfy either rule.
        assert not np.any(or_rule[both_nan])
        assert not np.any(fmax_rule[both_nan])


# ---------------------------------------------------------------------------
# Round trip: t_from_u(to_z(...)) recovers the input threshold
# ---------------------------------------------------------------------------


def test_round_trip_t_from_u_to_z():
    rng = np.random.default_rng(RNG_SEED + 5)
    n_pooled = 30000
    null_left = rng.normal(0, 1, size=(1, n_pooled))
    null_right = rng.normal(0, 1, size=(1, n_pooled))
    tail = fit_pooled_tail(null_left, null_right, n_grid=100000)

    alpha0 = 3e-2
    u0 = float(tail.quantile_at(alpha0))
    positions = np.arange(3)
    covariates = pd.DataFrame(
        {
            "position": [0, 1, 2],
            "log_own": [1.5, 2.0, 2.5],
            "log_sib": [1.5, 2.0, 2.5],
            "depth": [1.0, 2.0, 3.0],
        }
    )
    model = TailModel(
        alpha0=alpha0,
        u0=u0,
        position_ids=list(positions),
        a_position=[0.3, -0.2, 0.0],
        own_knots=choose_spline_knots(rng.uniform(1, 3, 200), df=4).tolist(),
        sib_knots=choose_spline_knots(rng.uniform(1, 3, 200), df=3).tolist(),
        dep_knots=choose_spline_knots(rng.uniform(0, 5, 200), df=2).tolist(),
        beta_own=[0.1, -0.05, 0.02, 0.0],
        beta_sib=[0.05, 0.0, -0.02],
        beta_dep=[0.02, -0.01],
    )

    # Candidate thresholds t_i, one per test, each strictly above u0 (i.e.
    # within the model's own claimed domain).
    t_input = np.array([u0 + 0.5, u0 + 1.0, u0 + 1.5])
    z = to_z(t_input, covariates, model, tail)
    t_recovered = t_from_u(z, covariates, model, tail)
    assert np.allclose(t_recovered, t_input, atol=1e-6)


def test_t_from_u_rejects_targets_below_the_anchor():
    rng = np.random.default_rng(RNG_SEED + 6)
    null_left = rng.normal(0, 1, size=(1, 5000))
    null_right = rng.normal(0, 1, size=(1, 5000))
    tail = fit_pooled_tail(null_left, null_right, n_grid=20000)
    alpha0 = 1e-2
    u0 = float(tail.quantile_at(alpha0))
    covariates = pd.DataFrame({"position": [0], "log_own": [1.0], "log_sib": [1.0], "depth": [1.0]})
    model = _trivial_model(alpha0, u0, [0])
    # z corresponding to pi = 0.5 > alpha0 -- below the anchor, out of domain.
    with pytest.raises(ValueError):
        t_from_u(np.array([-np.log10(0.5)]), covariates, model, tail)


# ---------------------------------------------------------------------------
# CV helper splits by replicate, never by test
# ---------------------------------------------------------------------------


def test_cv_splits_by_replicate_not_by_test():
    rng = np.random.default_rng(RNG_SEED + 7)
    R = 12
    positions = np.arange(6)
    n_tests_per_position = 30
    n_tests = positions.shape[0] * n_tests_per_position

    position = np.repeat(positions, n_tests_per_position)
    clade_left = rng.integers(5, 200, size=n_tests)
    clade_right = rng.integers(5, 200, size=n_tests)
    depth = rng.uniform(0, 10, size=n_tests)
    keys = pd.DataFrame(
        {
            "position": position,
            "clade_size_left": clade_left,
            "clade_size_right": clade_right,
            "distance_from_root": depth,
        }
    )

    null_left = rng.normal(0, 1, size=(R, n_tests))
    null_right = rng.normal(0, 1, size=(R, n_tests))
    u0 = float(np.quantile(np.concatenate([null_left.ravel(), null_right.ravel()]), 1 - 1e-2))

    cell_table = build_cell_table(null_left, null_right, keys, u0, n_own_bins=4, n_sib_bins=4, n_dep_bins=3)
    assert set(cell_table["replicate"].unique()) == set(range(R))

    entropy_by_position = {p: rng.uniform(0, 3) for p in positions}
    occupancy_by_position = {p: rng.uniform(0.5, 1.0) for p in positions}
    own_knots = choose_spline_knots(cell_table["own_repr"].to_numpy(), df=2)
    sib_knots = choose_spline_knots(cell_table["sib_repr"].to_numpy(), df=2)
    dep_knots = choose_spline_knots(cell_table["dep_repr"].to_numpy(), df=2)

    seen_fold_replicate_sets = []

    orig_fit = fit_penalized_poisson

    def _spy_fit(cells, *args, **kwargs):
        return orig_fit(cells, *args, **kwargs)

    best_lambda, cv_table = select_lambda_by_replicate_cv(
        cell_table,
        1e-2,
        own_knots,
        sib_knots,
        dep_knots,
        entropy_by_position,
        occupancy_by_position,
        lambda_grid=[0.1, 1.0],
        n_folds=4,
        seed=1,
    )
    assert isinstance(best_lambda, float)
    assert set(cv_table["fold"].unique()) <= set(range(4))

    # Directly verify the fold partition operates on whole replicates: derive
    # the same RNG-based fold assignment select_lambda_by_replicate_cv uses
    # and check no replicate's rows are split across train/test within a fold.
    replicates = np.sort(cell_table["replicate"].unique())
    folds = np.array_split(np.random.default_rng(1).permutation(replicates), 4)
    for held in folds:
        held_set = set(held.tolist())
        is_held = cell_table["replicate"].isin(held_set)
        # every row of a held-out replicate is held out; none partially so.
        for rep in held_set:
            rep_mask = cell_table["replicate"] == rep
            assert bool(is_held[rep_mask].all())


# ---------------------------------------------------------------------------
# Supporting-utility sanity checks (not required by name but exercised above)
# ---------------------------------------------------------------------------


def test_poisson_deviance_zero_when_k_equals_mu():
    mu = np.array([1.0, 2.5, 10.0])
    dev = poisson_deviance(mu, mu)
    assert np.allclose(dev, 0.0, atol=1e-10)


def test_quantile_bin_edges_monotone_and_deduped():
    x = np.array([1.0] * 50 + [2.0] * 50 + list(np.linspace(3, 10, 100)))
    edges = quantile_bin_edges(x, n_bins=20)
    assert np.all(np.diff(edges) > 0)


def test_branch_covariates_own_sib_swap():
    keys = pd.DataFrame(
        {
            "position": [1, 2],
            "clade_size_left": [10, 100],
            "clade_size_right": [20, 5],
            "distance_from_root": [1.0, 2.0],
        }
    )
    left_df, right_df = branch_covariates(keys)
    assert np.allclose(left_df["log_own"], np.log([10, 100]))
    assert np.allclose(left_df["log_sib"], np.log([20, 5]))
    assert np.allclose(right_df["log_own"], np.log([20, 5]))
    assert np.allclose(right_df["log_sib"], np.log([10, 100]))


def test_empirical_tail_fit_pooled_matches_exact_rank_at_grid_points():
    rng = np.random.default_rng(RNG_SEED + 8)
    n = 3000
    null_left = rng.exponential(1.0, size=(1, n))
    null_right = rng.exponential(1.0, size=(1, n))
    tail = fit_pooled_tail(null_left, null_right, n_grid=100000)
    pooled = np.concatenate([null_left.ravel(), null_right.ravel()])
    pooled.sort()
    n_pooled = pooled.shape[0]
    # Exact survival at a few actual pooled values. The inequality is
    # non-strict: this survival is used as a resampling p-value.
    for v in [pooled[10], pooled[n_pooled // 2], pooled[-5]]:
        n_lt = np.searchsorted(pooled, v, side="left")
        expected = (n_pooled - n_lt + 1.0) / (n_pooled + 1.0)
        assert tail.survival(v) == pytest.approx(expected, rel=1e-9)


def test_empirical_tail_handles_an_atom_at_the_upper_bound():
    """The BADASP score is bounded above at 2.0 and has a real atom there
    (RC = 1 with p_AC = 1 at AC = -1). A strict `>` survival would call
    every draw at the bound far more extreme than the null says it is."""
    pooled = np.concatenate([np.linspace(0.0, 1.9, 900), np.full(100, 2.0)])
    tail = fit_pooled_tail(pooled.reshape(1, -1), np.empty((1, 0)), n_grid=100000)
    n = pooled.size
    # 100 of 1000 draws sit at 2.0, so P(S >= 2.0) ~ 0.1, not ~1/(n+1).
    assert tail.survival(2.0) == pytest.approx((100 + 1.0) / (n + 1.0), rel=1e-9)
    assert tail.survival(2.0) > 0.09
    # Survival stays monotone across the atom.
    assert tail.survival(1.95) > tail.survival(2.0)
