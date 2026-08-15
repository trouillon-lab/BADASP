"""Covariate-conditional tail model for the BADASP simulation-based null.

Background
----------
`null_model.threshold_at_fdr` picks a single GLOBAL split-level threshold `t`
by comparing observed exceedances against simulated-null exceedances. That
is well-calibrated only if the null exceedance rate at `t` is roughly
constant across tests; `null_model.exceedance_flatness` is the diagnostic
for that, and on this dataset it is not flat -- position alone explains a
large, non-monotone share of null exceedances (see the module-level
docstring notes in the caller; not re-derived here).

This module fits a proportional-tails model instead of a single global rate:

    P(S > s | x) = A(x) * G(s)      for s >= u0

`G` is the pooled empirical null tail (pooled across ~21.7M null branch
draws, so its shape is treated as known essentially exactly). `A(x)` is a
single scalar rate multiplier per test, built from the position, own- and
sibling-clade size, and depth of that test, fit once by a penalised Poisson
regression on aggregated (replicate, position, clade-size bin, depth bin)
cell counts. This converts "estimate a ~3e-4 quantile from ~20 null
exceedances at one position" into "estimate one smooth multiplier from many
more exceedances pooled across an anchor probability `alpha0` that is
larger (hence better estimated) than the final calling threshold."

Integration contract (see the "Fit at branch level, then combine" note in
the task this module was built against)
----------------------------------------------------------------------------
`null_model.threshold_at_fdr` and `null_model._split_statistic` are used
UNCHANGED and UNMODIFIED. This module never touches `null_model.py`; it only
transforms branch-level scores into a branch-level `z` before those two
functions ever see the data:

    z_branch = -log10(pi_branch),  pi_branch = P(S_null > s_branch | x_branch)
    z_split  = null_model._split_statistic(z_left, z_right)  # fmax, verbatim

Because `z_branch` is a strictly monotone *decreasing* transform of
`s_branch` (see `EmpiricalTail.survival`), `z_left >= z_t` for some
threshold `z_t` is exactly equivalent to `s_left <= t` for the corresponding
`t`, so the either-branch-exceeds rule and everything `threshold_at_fdr`
computes from it (O(t), E[FP], the FDP-quantile sweep) survive verbatim when
called on `z_left, z_right` in place of the raw scores. `t_from_u` is the
inverse map back from a chosen scalar `u*` on the z-scale to a per-test
threshold `t_i` on the original score scale.

Honesty note (per repository convention: no claim of correctness before a
test proves it)
----------------------------------------------------------------------------
This is a first implementation. The synthetic-recovery test in
`tests/test_tail_model.py` is the evidence that the penalised Poisson fitter
recovers known effects; whether the proportional-tails assumption itself
holds on the *real* null (as opposed to being a reasonable, checkable
hypothesis) is reported separately, not asserted here or in any docstring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from .null_model import _split_statistic
from .state_io import AlignmentMatrix, load_alignment_matrix

# ---------------------------------------------------------------------------
# Repository-relative paths (never hardcode data paths outside this section)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "snakemake.yaml"

# The gap symbols recognised by the existing scoring code
# (`calculate_recent_conservation`, src/badasp/scoring.py:209); reused here so
# "occupancy" means the same thing everywhere in the repo.
GAP_CHARS: Tuple[str, ...] = ("-", ".")

# The 20 standard amino-acid one-letter codes, as ASCII byte values, used to
# restrict the Shannon-entropy calculation to real residues (ambiguity codes
# such as X/B/Z/U/O and lowercase soft-masked letters are excluded from the
# entropy distribution, though they still count as "occupied" for occupancy).
_STANDARD_AA_CODES = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)


def default_alignment_path(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """The alignment `score_tree_nodes` was run on (`paths.trimmed_fasta`).

    Every analysis script in this repo that needs the real alignment
    defaults its `--alignment` argument to this same file (see e.g.
    `scripts/compare_thresholds.py`); reading it from the shared config file
    here keeps this module from pasting that path as a second copy.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    return REPO_ROOT / cfg["paths"]["trimmed_fasta"]


# ---------------------------------------------------------------------------
# Pooled empirical null tail G(s) = P(S_null > s)
# ---------------------------------------------------------------------------


@dataclass
class EmpiricalTail:
    """Pooled empirical survival function of the null, `G(s) = P(S_null > s)`.

    Represented as a set of (score, survival) anchor points with `survival`
    computed by the add-one rule `(n_exceeding + 1) / (n_pooled + 1)` (the
    standard continuity correction for resampling p-values; it keeps
    `survival` strictly positive everywhere, including above the largest
    pooled draw, so `-log10(survival)` never overflows). Between anchor
    points, `survival(s)` interpolates *linearly in log(survival)* against
    `s` -- a monotone interpolant is all `to_z`'s "thresholding on z
    reproduces thresholding on s" property needs (see the module docstring),
    it is not a claim that the true empirical step function is smooth
    in between.

    When `n_grid` in `fit_pooled_tail` is large enough to hold every
    distinct pooled value (true for anything test-scale), no compression
    happens at all and every anchor point is an exact, individually
    computed (score, exact empirical survival) pair -- `survival()` queried
    at any of the original pooled scores then returns the exact value, not
    an approximation. For the real ~21.7M-draw null this is genuinely a
    lossy compression to `n_grid` representative points; that trade (a few
    thousand points instead of tens of millions, at the cost of
    interpolation error away from the kept points) is what makes the model
    JSON-serialisable and cheap to re-evaluate.
    """

    score_grid: np.ndarray  # ascending, unique
    survival_grid: np.ndarray  # same length, strictly descending
    n_pooled: int

    def __post_init__(self) -> None:
        self.score_grid = np.asarray(self.score_grid, dtype=np.float64)
        self.survival_grid = np.asarray(self.survival_grid, dtype=np.float64)
        if self.score_grid.shape != self.survival_grid.shape:
            raise ValueError("score_grid and survival_grid must have the same shape.")
        if self.score_grid.shape[0] < 2:
            raise ValueError("EmpiricalTail needs at least 2 grid points.")
        if np.any(np.diff(self.score_grid) <= 0):
            raise ValueError("score_grid must be strictly increasing.")
        if np.any(np.diff(self.survival_grid) >= 0):
            raise ValueError("survival_grid must be strictly decreasing.")

    def survival(self, s) -> np.ndarray:
        """`P(S_null > s)`, log-linearly interpolated between grid anchors.

        Queries below/above the grid's range are held flat at the nearest
        anchor's survival (equivalent to `numpy.interp`'s default flat
        extrapolation), never extrapolated to survival > 1 or <= 0.
        """
        s = np.asarray(s, dtype=np.float64)
        log_surv_grid = np.log(self.survival_grid)
        log_surv = np.interp(s, self.score_grid, log_surv_grid)
        return np.exp(log_surv)

    def quantile_at(self, alpha) -> np.ndarray:
        """Inverse of `survival`: the score `s` with `survival(s) ~= alpha`.

        `alpha` may be a scalar or array; values are clipped to the grid's
        own achievable survival range first (this tail is only ever meant
        to be queried inside that range -- see `t_from_u`'s domain check).
        """
        alpha = np.asarray(alpha, dtype=np.float64)
        lo, hi = self.survival_grid[-1], self.survival_grid[0]
        alpha_clipped = np.clip(alpha, lo, hi)
        log_alpha = np.log(alpha_clipped)
        # survival_grid is descending -> reverse both arrays so xp is ascending for np.interp.
        xp = np.log(self.survival_grid[::-1])
        fp = self.score_grid[::-1]
        return np.interp(log_alpha, xp, fp)

    def to_dict(self) -> dict:
        return {
            "score_grid": self.score_grid.tolist(),
            "survival_grid": self.survival_grid.tolist(),
            "n_pooled": int(self.n_pooled),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmpiricalTail":
        return cls(
            score_grid=np.asarray(d["score_grid"], dtype=np.float64),
            survival_grid=np.asarray(d["survival_grid"], dtype=np.float64),
            n_pooled=int(d["n_pooled"]),
        )


def fit_pooled_tail(null_left: np.ndarray, null_right: np.ndarray, n_grid: int = 4000) -> EmpiricalTail:
    """Pool null branch scores (both sides, all replicates) into one `EmpiricalTail`.

    `null_left`/`null_right` are the branch-level null score arrays (e.g.
    `badasp_score_left`/`badasp_score_right` stacked over replicates, shape
    `(R, n_tests)`, or any array-like -- flattened and NaN-dropped before
    pooling, matching the module-wide NaN policy in `null_model.py`).

    Compression to `n_grid` points is exact whenever the pooled data has at
    most `n_grid` distinct values (every value kept, no interpolation error
    at any of them); above that it keeps `n_grid` points chosen at
    log-spaced target survival probabilities (dense resolution near the
    tail, coarse in the bulk we do not care about), each recomputed as an
    *exact* (score, survival) pair from the full pooled data rather than
    just carrying the (possibly off-grid) target probability forward.
    """
    pooled = np.concatenate(
        [np.asarray(null_left, dtype=np.float64).ravel(), np.asarray(null_right, dtype=np.float64).ravel()]
    )
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError("No finite null branch scores to pool.")
    pooled.sort()
    n = pooled.size

    def _exact_survival(values: np.ndarray) -> np.ndarray:
        # count of pooled draws strictly greater than `values`, add-one corrected.
        n_leq = np.searchsorted(pooled, values, side="right")
        return (n - n_leq + 1.0) / (n + 1.0)

    unique_vals = np.unique(pooled)
    if unique_vals.shape[0] <= n_grid:
        score_grid = unique_vals
    else:
        min_prob = 1.0 / (n + 1.0)
        target_probs = np.geomspace(min_prob, 1.0, n_grid)
        ranks_from_top = np.clip(np.round(target_probs * (n + 1.0)), 1, n).astype(np.int64)
        idx = np.clip(n - ranks_from_top, 0, n - 1)
        score_grid = np.unique(pooled[idx])

    survival_grid = _exact_survival(score_grid)
    return EmpiricalTail(score_grid=score_grid, survival_grid=survival_grid, n_pooled=n)


# ---------------------------------------------------------------------------
# Branch-level covariates
# ---------------------------------------------------------------------------


def branch_covariates(keys: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split split-level keys into left/right branch-level covariate frames.

    `keys` must carry `position`, `clade_size_left`, `clade_size_right`,
    `distance_from_root` (exactly the columns `null_model.load_observed_scores`
    returns alongside the score arrays, and the columns of
    `raw_node_scores.csv`, which the `null_calibration/persite` npz files are
    positionally aligned to with no key arrays of their own).

    Returns `(left_df, right_df)`, each with columns
    `["position", "log_own", "log_sib", "depth"]`, positionally aligned to
    the branch score arrays of that side (e.g. `badasp_score_left` /
    `badasp_score_right`, or `null_left` / `null_right`).

    `own`/`sib` are `clade_size_left`/`clade_size_right` from that branch's
    own point of view (left branch: own=left, sib=right; right branch:
    own=right, sib=left) -- per the task spec, own and sibling clade size
    are used separately and `clade_size_total`/`min` are never used.
    `event_type` is never read here or anywhere downstream in this module.
    """
    depth = keys["distance_from_root"].to_numpy(dtype=np.float64)
    position = keys["position"].to_numpy()
    log_cl = np.log(keys["clade_size_left"].to_numpy(dtype=np.float64))
    log_cr = np.log(keys["clade_size_right"].to_numpy(dtype=np.float64))
    left_df = pd.DataFrame({"position": position, "log_own": log_cl, "log_sib": log_cr, "depth": depth})
    right_df = pd.DataFrame({"position": position, "log_own": log_cr, "log_sib": log_cl, "depth": depth})
    return left_df, right_df


def compute_column_stats(alignment_matrix: AlignmentMatrix, csv_positions: Sequence[int]) -> pd.DataFrame:
    """Per-position Shannon entropy (nats) and occupancy for each `position` id.

    `position` in `raw_node_scores.csv` is 1-based (`pos + 1` in
    `src/badasp/scoring.py:483,513`, where `pos` is the 0-based column index
    into the alignment); this function applies that same `-1` offset before
    indexing `alignment_matrix`.

    Entropy is computed over the 20 standard amino acids only (gaps and
    ambiguity codes excluded from the frequency distribution); occupancy is
    the fraction of non-gap sequences at that column, using the same gap
    symbol set as `calculate_recent_conservation` (`src/badasp/scoring.py:209`).
    These are used only as the *prior mean* for the per-position rate
    effect `a_p` in `fit_penalized_poisson`, not as free covariates on their
    own -- per the task spec they explain only a fraction of the position
    variation and are not meant to replace the fitted `a_p`.
    """
    positions = np.asarray(sorted({int(p) for p in csv_positions}))
    n_total = alignment_matrix.matrix.shape[0]
    gap_codes = {ord(c) for c in GAP_CHARS}
    rows = []
    for p in positions:
        col_idx = int(p) - 1
        counts = np.bincount(alignment_matrix.matrix[:, col_idx], minlength=256)
        n_gap = int(sum(counts[c] for c in gap_codes))
        aa_counts = counts[_STANDARD_AA_CODES]
        n_aa = int(aa_counts.sum())
        if n_aa > 0:
            freqs = aa_counts / n_aa
            freqs = freqs[freqs > 0]
            entropy = float(-np.sum(freqs * np.log(freqs)))
        else:
            entropy = 0.0
        occupancy = float((n_total - n_gap) / n_total) if n_total > 0 else 0.0
        rows.append({"position": int(p), "entropy": entropy, "occupancy": occupancy})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Natural cubic spline basis (implemented directly in numpy; no patsy/statsmodels)
# ---------------------------------------------------------------------------


def choose_spline_knots(x: np.ndarray, df: int, boundary_quantiles: Tuple[float, float] = (0.0, 100.0)) -> np.ndarray:
    """Knot placement matching R's `splines::ns(x, df=df)` convention.

    Boundary knots at `boundary_quantiles` of `x` (defaults to the data
    extremes) and `df - 1` interior knots at evenly spaced quantiles in
    between, for `df + 1` knots total -- which `natural_spline_basis` turns
    into exactly `df` non-constant basis columns.
    """
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}.")
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite values to place spline knots from.")
    lo_q, hi_q = boundary_quantiles
    quantile_points = np.linspace(lo_q, hi_q, df + 1) / 100.0
    knots = np.unique(np.quantile(x, quantile_points))
    if knots.shape[0] < df + 1:
        raise ValueError(
            f"Only {knots.shape[0]} distinct knot quantiles available for df={df} "
            f"(need {df + 1}); covariate has too few distinct values for this many degrees of freedom."
        )
    return knots


def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis, `len(knots) - 1` non-constant columns.

    Construction follows Hastie, Tibshirani & Friedman, "The Elements of
    Statistical Learning" (2nd ed.), eq. (5.4)-(5.5): with knots
    `xi_1 < ... < xi_K`,

        d_k(X) = [(X - xi_k)_+^3 - (X - xi_K)_+^3] / (xi_K - xi_k),  k = 1, ..., K-1
        N_1(X) = 1,  N_2(X) = X,  N_{k+2}(X) = d_k(X) - d_{K-1}(X),  k = 1, ..., K-2

    This function returns `[N_2, ..., N_K]` -- the constant column `N_1` is
    deliberately dropped (matching R's `splines::ns()`, which never
    includes an intercept column), because in this module the per-position
    effect `a_p` already plays the role of an intercept; keeping a second,
    redundant constant in the spline basis would make the two only
    jointly, not separately, identifiable.

    Values of `x` outside `[knots[0], knots[-1]]` extrapolate *linearly*
    (a defining property of natural splines, not an approximation specific
    to this implementation).
    """
    x = np.asarray(x, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    K = knots.shape[0]
    if K < 2:
        raise ValueError("natural_spline_basis needs at least 2 knots.")
    if np.any(np.diff(knots) <= 0):
        raise ValueError("knots must be strictly increasing.")

    def _pos_cube(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0, v**3, 0.0)

    n_d = K - 1  # d_1 .. d_{K-1}, 0-indexed as d_vals[:, 0..K-2]
    last_knot = knots[-1]
    d_vals = np.empty((x.shape[0], n_d), dtype=np.float64)
    for k in range(n_d):
        denom = last_knot - knots[k]
        d_vals[:, k] = (_pos_cube(x - knots[k]) - _pos_cube(x - last_knot)) / denom

    n_cols = K - 1  # N_2 .. N_K
    basis = np.empty((x.shape[0], n_cols), dtype=np.float64)
    basis[:, 0] = x  # N_2
    d_last = d_vals[:, n_d - 1]  # d_{K-1}
    for k in range(n_cols - 1):  # produces N_3 .. N_K, i.e. K-2 columns, using d_1 .. d_{K-2}
        basis[:, k + 1] = d_vals[:, k] - d_last
    return basis


# ---------------------------------------------------------------------------
# Aggregating branch-level exceedances into regression cells
# ---------------------------------------------------------------------------


def quantile_bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges for `x`, deduplicated (fewer bins if `x` is coarse).

    Same binning convention `null_model.exceedance_flatness` uses
    (`pd.qcut(..., duplicates="drop")`), reimplemented here as edges so the
    same edges can be shared between the own-clade and sibling-clade axes.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.shape[0] < 2:
        raise ValueError("Not enough distinct values to form even one bin.")
    return edges


def _bin_with_repr(x: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bin index (0-based, clipped into range) and the bin's mean value.

    The mean is computed once, globally, over all of `x` that falls in each
    bin -- used as the single representative covariate value the spline
    basis is evaluated at for every cell in that bin (see module docstring:
    cells are `position x own-bin x sib-bin x depth-bin`).
    """
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.shape[0] - 2)
    n_bins = edges.shape[0] - 1
    sums = np.bincount(idx, weights=x, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return idx, means


def build_cell_table(
    null_left: np.ndarray,
    null_right: np.ndarray,
    keys: pd.DataFrame,
    u0: float,
    n_own_bins: int = 10,
    n_sib_bins: int = 10,
    n_dep_bins: int = 5,
) -> pd.DataFrame:
    """Aggregate branch-level null exceedances (>= `u0`) into regression cells.

    One row per (replicate, position, own-clade bin, sibling-clade bin,
    depth bin) combination that actually occurs, with columns `replicate`,
    `position`, `own_repr`, `sib_repr`, `dep_repr` (the bin's mean covariate
    value -- see `_bin_with_repr`), `n_trials` (finite branch scores
    assigned to that cell) and `n_exceed` (how many of those are `>= u0`).
    Left and right branches contribute to the same table (both are draws
    from the same conditional model, distinguished only by their own/sib
    covariate values, not by which physical side they came from).

    `n_own_bins`/`n_sib_bins`/`n_dep_bins` trade cell resolution against
    per-cell sample size; the defaults are a documented starting point (not
    a validated choice) balancing having enough distinct covariate values
    per position against not aggregating so coarsely that the smooth terms
    have nothing left to fit.

    NaN branch scores (gap positions) are dropped from both `n_trials` and
    `n_exceed`, matching the NaN policy in `null_model.py`: a NaN branch
    never contributes information and is not a "failed trial" either.
    """
    null_left = np.asarray(null_left, dtype=np.float64)
    null_right = np.asarray(null_right, dtype=np.float64)
    R, n_tests = null_left.shape
    if null_right.shape != (R, n_tests):
        raise ValueError("null_left and null_right must share shape (R, n_tests).")
    if len(keys) != n_tests:
        raise ValueError(f"keys has {len(keys)} rows but null arrays have {n_tests} tests.")

    left_df, right_df = branch_covariates(keys)

    own_edges = quantile_bin_edges(
        np.concatenate([left_df["log_own"].to_numpy(), right_df["log_own"].to_numpy()]), n_own_bins
    )
    sib_edges = quantile_bin_edges(
        np.concatenate([left_df["log_sib"].to_numpy(), right_df["log_sib"].to_numpy()]), n_sib_bins
    )
    dep_edges = quantile_bin_edges(left_df["depth"].to_numpy(), n_dep_bins)

    own_bin_L, own_repr_by_bin = _bin_with_repr(left_df["log_own"].to_numpy(), own_edges)
    own_bin_R, _ = _bin_with_repr(right_df["log_own"].to_numpy(), own_edges)
    sib_bin_L, sib_repr_by_bin = _bin_with_repr(left_df["log_sib"].to_numpy(), sib_edges)
    sib_bin_R, _ = _bin_with_repr(right_df["log_sib"].to_numpy(), sib_edges)
    dep_bin, dep_repr_by_bin = _bin_with_repr(left_df["depth"].to_numpy(), dep_edges)  # depth shared by both branches

    position = left_df["position"].to_numpy()

    frames = []
    for side_scores, own_bin, sib_bin in ((null_left, own_bin_L, sib_bin_L), (null_right, own_bin_R, sib_bin_R)):
        finite = np.isfinite(side_scores)  # (R, n_tests)
        exceed = finite & (side_scores >= u0)
        for r in range(R):
            df = pd.DataFrame(
                {
                    "position": position,
                    "own_bin": own_bin,
                    "sib_bin": sib_bin,
                    "dep_bin": dep_bin,
                    "trial": finite[r].astype(np.int64),
                    "exceed": exceed[r].astype(np.int64),
                }
            )
            grouped = df.groupby(["position", "own_bin", "sib_bin", "dep_bin"], as_index=False, observed=True).agg(
                n_trials=("trial", "sum"), n_exceed=("exceed", "sum")
            )
            grouped = grouped[grouped["n_trials"] > 0]
            grouped["replicate"] = r
            frames.append(grouped)

    cells = pd.concat(frames, ignore_index=True)
    # Collapse the two branches' contributions to identical (replicate, position, bins) cells.
    cells = cells.groupby(["replicate", "position", "own_bin", "sib_bin", "dep_bin"], as_index=False).agg(
        n_trials=("n_trials", "sum"), n_exceed=("n_exceed", "sum")
    )
    cells["own_repr"] = own_repr_by_bin[cells["own_bin"].to_numpy()]
    cells["sib_repr"] = sib_repr_by_bin[cells["sib_bin"].to_numpy()]
    cells["dep_repr"] = dep_repr_by_bin[cells["dep_bin"].to_numpy()]
    return cells[["replicate", "position", "own_bin", "sib_bin", "dep_bin", "own_repr", "sib_repr", "dep_repr", "n_trials", "n_exceed"]]


def collapse_cells(cell_table: pd.DataFrame) -> pd.DataFrame:
    """Sum `n_trials`/`n_exceed` over replicates, keeping bins/position separate.

    Used to build the training (or held-out) aggregate for a given subset
    of replicates during cross-validation, and to build the final
    all-replicates aggregate for the production fit.
    """
    return cell_table.groupby(["position", "own_bin", "sib_bin", "dep_bin"], as_index=False).agg(
        own_repr=("own_repr", "first"),
        sib_repr=("sib_repr", "first"),
        dep_repr=("dep_repr", "first"),
        n_trials=("n_trials", "sum"),
        n_exceed=("n_exceed", "sum"),
    )


# ---------------------------------------------------------------------------
# Penalised Poisson regression
# ---------------------------------------------------------------------------


@dataclass
class PoissonFitResult:
    """One penalised-Poisson fit: per-position effects, spline coefficients,
    and the (unpenalised) linear prior `a_p ~ c0 + c1*entropy_p + c2*occupancy_p`
    the position effects are shrunk toward."""

    position_ids: np.ndarray  # (P,), sorted, the positions this fit covers
    a_position: np.ndarray  # (P,)
    beta_own: np.ndarray
    beta_sib: np.ndarray
    beta_dep: np.ndarray
    c: np.ndarray  # [c0, c1, c2]
    converged: bool
    n_iter: int
    message: str
    objective: float


def _unpack_theta(theta: np.ndarray, P: int, d1: int, d2: int, d3: int):
    i = 0
    a_p = theta[i : i + P]
    i += P
    beta_own = theta[i : i + d1]
    i += d1
    beta_sib = theta[i : i + d2]
    i += d2
    beta_dep = theta[i : i + d3]
    i += d3
    c = theta[i : i + 3]
    return a_p, beta_own, beta_sib, beta_dep, c


def _neg_log_posterior_and_grad(
    theta: np.ndarray,
    position_idx: np.ndarray,
    Bown: np.ndarray,
    Bsib: np.ndarray,
    Bdep: np.ndarray,
    offset: np.ndarray,
    n_exceed: np.ndarray,
    entropy_p: np.ndarray,
    occupancy_p: np.ndarray,
    lambda_: float,
    P: int,
) -> Tuple[float, np.ndarray]:
    d1, d2, d3 = Bown.shape[1], Bsib.shape[1], Bdep.shape[1]
    a_p, beta_own, beta_sib, beta_dep, c = _unpack_theta(theta, P, d1, d2, d3)

    # np.errstate here suppresses a spurious "divide by zero"/"overflow" RuntimeWarning that
    # macOS Accelerate BLAS raises on some matmuls against a near-zero coefficient vector
    # (e.g. theta initialised at x0=0); verified benign -- the returned values are correct,
    # this is a BLAS-backend quirk, not a real numerical issue in this computation.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        eta = offset + a_p[position_idx] + Bown @ beta_own + Bsib @ beta_sib + Bdep @ beta_dep
    eta = np.clip(eta, -50.0, 50.0)  # numerical safety valve; well outside any plausible fitted range
    mu = np.exp(eta)

    # Reduced Poisson NLL: sum(mu - K*eta); drops the K*log(K) - K term, which is
    # constant in theta and does not affect the argmin or the gradient.
    nll = float(np.sum(mu - n_exceed * eta))
    resid = mu - n_exceed

    grad_a = np.bincount(position_idx, weights=resid, minlength=P)
    grad_beta_own = Bown.T @ resid
    grad_beta_sib = Bsib.T @ resid
    grad_beta_dep = Bdep.T @ resid

    Z = np.column_stack([np.ones(P), entropy_p, occupancy_p])
    r = a_p - Z @ c
    penalty = lambda_ * float(np.sum(r**2))
    grad_a = grad_a + 2.0 * lambda_ * r
    grad_c = -2.0 * lambda_ * (Z.T @ r)

    total = nll + penalty
    grad = np.concatenate([grad_a, grad_beta_own, grad_beta_sib, grad_beta_dep, grad_c])
    return total, grad


def fit_penalized_poisson(
    cells: pd.DataFrame,
    alpha0: float,
    own_knots: np.ndarray,
    sib_knots: np.ndarray,
    dep_knots: np.ndarray,
    entropy_by_position: Dict[int, float],
    occupancy_by_position: Dict[int, float],
    lambda_: float,
    x0: Optional[np.ndarray] = None,
) -> PoissonFitResult:
    """Penalised Poisson fit of

        log E[n_exceed] = log(alpha0) + log(n_trials)
                           + a_position + f_own(log_own) + f_sib(log_sib) + f_dep(depth)

    penalised by `lambda_ * sum_p (a_p - c0 - c1*entropy_p - c2*occupancy_p)^2`
    (c0, c1, c2 unpenalised). `cells` must have one row per (position,
    own/sib/dep bin) combination already collapsed across whatever set of
    replicates is being fit on (see `collapse_cells`), with columns
    `position`, `own_repr`, `sib_repr`, `dep_repr`, `n_trials`, `n_exceed`.

    `log(n_trials)` is the exposure offset the aggregation makes necessary:
    a cell with more (branch, replicate) draws trivially has proportionally
    more expected exceedances, and that must be accounted for before
    `a_position`/`f_own`/`f_sib`/`f_dep` can be interpreted as *rate*
    effects. `entropy_by_position`/`occupancy_by_position` must cover every
    position id present in `cells`.

    Optimised with `scipy.optimize.minimize(..., method="L-BFGS-B")` using
    an analytic gradient (see `_neg_log_posterior_and_grad`); `x0` defaults
    to all zeros (`A(x) = 1` everywhere, `a_p` at the unshrunk prior's own
    zero point), a principled rather than arbitrary starting point for a
    log-link GLM with no prior fit to warm-start from.
    """
    position_ids = np.sort(cells["position"].unique())
    P = position_ids.shape[0]
    pos_to_idx = {p: i for i, p in enumerate(position_ids)}
    position_idx = cells["position"].map(pos_to_idx).to_numpy()

    entropy_p = np.array([entropy_by_position[p] for p in position_ids], dtype=np.float64)
    occupancy_p = np.array([occupancy_by_position[p] for p in position_ids], dtype=np.float64)

    Bown = natural_spline_basis(cells["own_repr"].to_numpy(dtype=np.float64), own_knots)
    Bsib = natural_spline_basis(cells["sib_repr"].to_numpy(dtype=np.float64), sib_knots)
    Bdep = natural_spline_basis(cells["dep_repr"].to_numpy(dtype=np.float64), dep_knots)
    offset = np.log(alpha0) + np.log(cells["n_trials"].to_numpy(dtype=np.float64))
    n_exceed = cells["n_exceed"].to_numpy(dtype=np.float64)

    d1, d2, d3 = Bown.shape[1], Bsib.shape[1], Bdep.shape[1]
    n_params = P + d1 + d2 + d3 + 3
    if x0 is None:
        x0 = np.zeros(n_params)
    elif x0.shape[0] != n_params:
        raise ValueError(f"x0 has {x0.shape[0]} entries, expected {n_params}.")

    result = minimize(
        _neg_log_posterior_and_grad,
        x0,
        args=(position_idx, Bown, Bsib, Bdep, offset, n_exceed, entropy_p, occupancy_p, lambda_, P),
        jac=True,
        method="L-BFGS-B",
    )
    a_p, beta_own, beta_sib, beta_dep, c = _unpack_theta(result.x, P, d1, d2, d3)
    return PoissonFitResult(
        position_ids=position_ids,
        a_position=a_p,
        beta_own=beta_own,
        beta_sib=beta_sib,
        beta_dep=beta_dep,
        c=c,
        converged=bool(result.success),
        n_iter=int(result.nit),
        message=str(result.message),
        objective=float(result.fun),
    )


def predict_log_mu(
    fit: PoissonFitResult,
    cells: pd.DataFrame,
    alpha0: float,
    own_knots: np.ndarray,
    sib_knots: np.ndarray,
    dep_knots: np.ndarray,
) -> np.ndarray:
    """`log E[n_exceed]` for `cells` under a fitted `PoissonFitResult`.

    `cells` need not be the same rows the model was fit on (used for
    out-of-fold prediction in `select_lambda_by_replicate_cv`); every
    `position` in `cells` must be one `fit.position_ids` covers.
    """
    pos_to_a = dict(zip(fit.position_ids.tolist(), fit.a_position.tolist()))
    missing = set(cells["position"].unique()) - set(pos_to_a)
    if missing:
        raise KeyError(f"positions {sorted(missing)} were not present in the fitted model.")
    a = cells["position"].map(pos_to_a).to_numpy(dtype=np.float64)
    Bown = natural_spline_basis(cells["own_repr"].to_numpy(dtype=np.float64), own_knots)
    Bsib = natural_spline_basis(cells["sib_repr"].to_numpy(dtype=np.float64), sib_knots)
    Bdep = natural_spline_basis(cells["dep_repr"].to_numpy(dtype=np.float64), dep_knots)
    offset = np.log(alpha0) + np.log(cells["n_trials"].to_numpy(dtype=np.float64))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # see note in _neg_log_posterior_and_grad
        return offset + a + Bown @ fit.beta_own + Bsib @ fit.beta_sib + Bdep @ fit.beta_dep


def poisson_deviance(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-observation Poisson deviance, `2*(k*log(k/mu) - (k - mu))`.

    Uses the standard convention `k*log(k/mu) = 0` when `k == 0`.
    """
    k = np.asarray(k, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(k > 0, k * np.log(k / mu), 0.0)
    return 2.0 * (term - (k - mu))


# ---------------------------------------------------------------------------
# Replicate-fold cross-validation for the penalty strength lambda
# ---------------------------------------------------------------------------


def select_lambda_by_replicate_cv(
    cell_table: pd.DataFrame,
    alpha0: float,
    own_knots: np.ndarray,
    sib_knots: np.ndarray,
    dep_knots: np.ndarray,
    entropy_by_position: Dict[int, float],
    occupancy_by_position: Dict[int, float],
    lambda_grid: Sequence[float],
    n_folds: int = 5,
    seed: int = 0,
) -> Tuple[float, pd.DataFrame]:
    """Choose `lambda` by out-of-fold Poisson deviance, folding over REPLICATES.

    `cell_table` is the per-replicate cell table from `build_cell_table`
    (one row per (replicate, position, bins) combination). Replicates --
    never individual cells/tests -- are randomly partitioned into
    `n_folds` groups; for each candidate `lambda_` and each fold, the model
    is fit on the other folds' replicates (`collapse_cells`-aggregated) and
    evaluated by total Poisson deviance on the held-out fold's replicates
    (aggregated the same way). This is the only correct level to fold at:
    every replicate shares the same fixed set of (position, bin) cells, so
    folding by test/cell instead would leak the very quantity (the
    replicate-to-replicate noise in a cell's rate) cross-validation is
    supposed to be measuring.

    Returns `(best_lambda, cv_table)` where `cv_table` has one row per
    `(lambda, fold)` with the held-out deviance and cell count, for
    inspection.
    """
    replicates = np.sort(cell_table["replicate"].unique())
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(replicates), n_folds)

    rows = []
    for lam in lambda_grid:
        for k, held in enumerate(folds):
            held_set = set(held.tolist())
            is_held = cell_table["replicate"].isin(held_set)
            train_cells = collapse_cells(cell_table[~is_held])
            test_cells = collapse_cells(cell_table[is_held])
            if train_cells.empty or test_cells.empty:
                continue
            fit = fit_penalized_poisson(
                train_cells, alpha0, own_knots, sib_knots, dep_knots, entropy_by_position, occupancy_by_position, lam
            )
            # Positions absent from the training fold cannot be scored out-of-fold honestly.
            scoreable = test_cells["position"].isin(set(fit.position_ids.tolist()))
            test_scoreable = test_cells[scoreable]
            if test_scoreable.empty:
                continue
            log_mu = predict_log_mu(fit, test_scoreable, alpha0, own_knots, sib_knots, dep_knots)
            mu = np.exp(np.clip(log_mu, -50.0, 50.0))
            dev = poisson_deviance(test_scoreable["n_exceed"].to_numpy(dtype=np.float64), mu)
            rows.append({"lambda": lam, "fold": k, "held_out_deviance": float(dev.sum()), "n_cells": int(len(test_scoreable))})

    cv_table = pd.DataFrame(rows)
    if cv_table.empty:
        raise ValueError("Cross-validation produced no scoreable folds; check lambda_grid/n_folds/data size.")
    summary = cv_table.groupby("lambda", as_index=False)["held_out_deviance"].sum()
    best_lambda = float(summary.loc[summary["held_out_deviance"].idxmin(), "lambda"])
    return best_lambda, cv_table


# ---------------------------------------------------------------------------
# TailModel: the fitted, JSON-serialisable state
# ---------------------------------------------------------------------------


@dataclass
class TailModel:
    """Fitted covariate-conditional tail model state.

    Deliberately does NOT embed the pooled empirical tail `G` (`EmpiricalTail`):
    that object is cheap to rebuild from the null replicate arrays whenever
    needed (`fit_pooled_tail`) and this dataclass's job is just to hold the
    small, genuinely-fitted state (coefficients, knots, alpha0, covariate
    names, u0), matching the required serialisation contract. Every
    function that needs `A(x)*G(s)` (`to_z`, `t_from_u`) takes an
    `EmpiricalTail` as an explicit separate argument.
    """

    alpha0: float
    u0: float
    covariate_names: List[str] = field(default_factory=lambda: ["position", "log_own", "log_sib", "depth"])
    position_ids: List[int] = field(default_factory=list)
    a_position: List[float] = field(default_factory=list)
    own_knots: List[float] = field(default_factory=list)
    sib_knots: List[float] = field(default_factory=list)
    dep_knots: List[float] = field(default_factory=list)
    beta_own: List[float] = field(default_factory=list)
    beta_sib: List[float] = field(default_factory=list)
    beta_dep: List[float] = field(default_factory=list)
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    lambda_: float = 0.0
    n_replicates: int = 0
    fit_converged: bool = False
    fit_message: str = ""

    def predict_log_A(self, covariates: pd.DataFrame) -> np.ndarray:
        """`log A(x)` for each row of `covariates` (columns: `covariate_names`)."""
        a_lookup = dict(zip(self.position_ids, self.a_position))
        pos = covariates["position"].to_numpy()
        missing = sorted(set(np.unique(pos).tolist()) - set(a_lookup))
        if missing:
            raise KeyError(f"positions {missing} were not covered by this fitted TailModel.")
        a = np.array([a_lookup[p] for p in pos], dtype=np.float64)
        Bown = natural_spline_basis(covariates["log_own"].to_numpy(dtype=np.float64), np.asarray(self.own_knots))
        Bsib = natural_spline_basis(covariates["log_sib"].to_numpy(dtype=np.float64), np.asarray(self.sib_knots))
        Bdep = natural_spline_basis(covariates["depth"].to_numpy(dtype=np.float64), np.asarray(self.dep_knots))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # see note in _neg_log_posterior_and_grad
            return (
                a
                + Bown @ np.asarray(self.beta_own, dtype=np.float64)
                + Bsib @ np.asarray(self.beta_sib, dtype=np.float64)
                + Bdep @ np.asarray(self.beta_dep, dtype=np.float64)
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TailModel":
        return cls(**d)

    def to_json(self, path: Optional[Path] = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_json(cls, source) -> "TailModel":
        """`source` is either a JSON string or a path to a JSON file."""
        if isinstance(source, (str, Path)) and Path(source).exists():
            text = Path(source).read_text()
        else:
            text = source
        return cls.from_dict(json.loads(text))


def fit_tail_model(
    null_left: np.ndarray,
    null_right: np.ndarray,
    keys: pd.DataFrame,
    alignment_matrix: AlignmentMatrix,
    alpha0: float = 1e-3,
    own_df: int = 4,
    sib_df: int = 3,
    dep_df: int = 2,
    n_own_bins: int = 10,
    n_sib_bins: int = 10,
    n_dep_bins: int = 5,
    lambda_grid: Optional[Sequence[float]] = None,
    n_folds: int = 5,
    seed: int = 0,
) -> Tuple[TailModel, EmpiricalTail, pd.DataFrame]:
    """End-to-end fit: pooled tail, per-position entropy/occupancy prior,
    cell aggregation, replicate-fold CV over `lambda_grid`, and a final fit
    on all replicates at the selected `lambda`.

    `alpha0` is the anchor exceedance probability defining `u0` (the
    pooled-tail quantile the proportional-tails model is anchored at); the
    caller is expected to probe a handful of `alpha0` values for stability,
    per the task this module was built against -- this function does not
    do that probing itself, it fits one model for one `alpha0`.

    Returns `(model, tail, cv_table)`. `tail` (the `EmpiricalTail`) is
    returned alongside `model` because `to_z`/`t_from_u` need both and
    `model` alone does not embed it (see `TailModel`'s docstring).
    """
    tail = fit_pooled_tail(null_left, null_right)
    u0 = float(tail.quantile_at(alpha0))

    cell_table = build_cell_table(null_left, null_right, keys, u0, n_own_bins, n_sib_bins, n_dep_bins)

    left_df, right_df = branch_covariates(keys)
    own_knots = choose_spline_knots(pd.concat([left_df["log_own"], right_df["log_own"]]).to_numpy(), own_df)
    sib_knots = choose_spline_knots(pd.concat([left_df["log_sib"], right_df["log_sib"]]).to_numpy(), sib_df)
    dep_knots = choose_spline_knots(left_df["depth"].to_numpy(), dep_df)

    col_stats = compute_column_stats(alignment_matrix, keys["position"].unique())
    entropy_by_position = dict(zip(col_stats["position"], col_stats["entropy"]))
    occupancy_by_position = dict(zip(col_stats["position"], col_stats["occupancy"]))

    if lambda_grid is None:
        lambda_grid = np.geomspace(1e-3, 1e3, 13)

    best_lambda, cv_table = select_lambda_by_replicate_cv(
        cell_table,
        alpha0,
        own_knots,
        sib_knots,
        dep_knots,
        entropy_by_position,
        occupancy_by_position,
        lambda_grid,
        n_folds=n_folds,
        seed=seed,
    )

    final_cells = collapse_cells(cell_table)
    fit = fit_penalized_poisson(
        final_cells, alpha0, own_knots, sib_knots, dep_knots, entropy_by_position, occupancy_by_position, best_lambda
    )

    model = TailModel(
        alpha0=alpha0,
        u0=u0,
        position_ids=fit.position_ids.tolist(),
        a_position=fit.a_position.tolist(),
        own_knots=own_knots.tolist(),
        sib_knots=sib_knots.tolist(),
        dep_knots=dep_knots.tolist(),
        beta_own=fit.beta_own.tolist(),
        beta_sib=fit.beta_sib.tolist(),
        beta_dep=fit.beta_dep.tolist(),
        c0=float(fit.c[0]),
        c1=float(fit.c[1]),
        c2=float(fit.c[2]),
        lambda_=best_lambda,
        n_replicates=int(cell_table["replicate"].nunique()),
        fit_converged=fit.converged,
        fit_message=fit.message,
    )
    return model, tail, cv_table


# ---------------------------------------------------------------------------
# The z-scale transform and its inverse
# ---------------------------------------------------------------------------


def to_z(scores: np.ndarray, covariates: pd.DataFrame, model: TailModel, tail: EmpiricalTail) -> np.ndarray:
    """Branch-level `z = -log10(pi)`, `pi = P(S_null > s | x)` under the fitted model.

    For `s >= model.u0`, `pi = A(x) * G(s)` (the fitted proportional-tails
    model). For `s < model.u0`, `pi = G(s)` directly (`A(x) == 1`): the
    model is anchored at, and only claims validity above, `u0`, so it is
    not extrapolated into the bulk. When `A(x) == 1` for every test (e.g. a
    model fit with `lambda_` large enough to shrink every `a_p` and spline
    coefficient to ~0), this reduces to `-log10(G(s))` everywhere -- a
    single strictly monotone transform of `s`, so thresholding on `z`
    reproduces thresholding on `s` exactly (see module docstring).

    `scores` may be any shape broadcastable against `covariates`'s row
    count in its trailing axis (e.g. `(n_tests,)` for one observed/null
    replicate, or `(R, n_tests)` for a stack of null replicates); NaN
    scores propagate to NaN `z`, matching `null_model.py`'s NaN policy so
    `null_model._split_statistic`'s `fmax` behaves identically on `z` as on
    the raw scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    log_A = model.predict_log_A(covariates)
    A = np.exp(log_A)
    G = tail.survival(scores)
    pi = np.where(scores >= model.u0, A * G, G)
    pi = np.clip(pi, np.finfo(np.float64).tiny, 1.0)
    with np.errstate(invalid="ignore"):
        z = -np.log10(pi)
    return np.where(np.isnan(scores), np.nan, z)


def t_from_u(u, covariates: pd.DataFrame, model: TailModel, tail: EmpiricalTail) -> np.ndarray:
    """Inverse of `to_z`'s tail branch: the per-test threshold `t_i` on the
    original score scale that a chosen scalar `u*` on the z-scale maps to
    for each row of `covariates`.

    Only defined within the tail model's own domain of validity (`s >= u0`,
    equivalently `pi <= alpha0`): raises `ValueError` if `u` implies
    `pi > alpha0` (i.e. asks this model to say something about the bulk it
    was never fit to describe). Per-test, `pi_target / A(x)` is clipped to
    at most `alpha0` before inversion, so a test with a suppressed rate
    (`A(x) < 1`) that would otherwise need `pi_target / A(x) > alpha0` is
    given `u0` itself rather than an out-of-domain extrapolation below it.
    """
    log_A = model.predict_log_A(covariates)
    A = np.exp(log_A)
    u_arr = np.asarray(u, dtype=np.float64)
    pi_target = 10.0 ** (-u_arr)
    if np.any(pi_target > model.alpha0 * (1.0 + 1e-9)):
        raise ValueError(
            f"t_from_u called with a target below the model's own anchor (pi_target > alpha0={model.alpha0}); "
            "the proportional-tails model is not fit to describe the bulk below u0."
        )
    pi_for_G = np.minimum(pi_target / A, model.alpha0)
    return tail.quantile_at(pi_for_G)


def either_branch_exceeds(z_left: np.ndarray, z_right: np.ndarray) -> np.ndarray:
    """`z_split = fmax(z_left, z_right)`, delegating verbatim to
    `null_model._split_statistic` -- see that function's docstring for why
    `fmax` on z is exactly the either-branch-exceeds rule, NaN-safe, for
    every threshold. Exposed here under a name specific to this module so
    callers of `tail_model` do not need to import a private `null_model` symbol.
    """
    return _split_statistic(z_left, z_right)
