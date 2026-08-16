"""Coarse cell-conditional threshold for the BADASP simulation-based null.

Why this exists alongside `tail_model`
--------------------------------------
`tail_model` fits a per-position rate multiplier by penalised Poisson
regression. On synthetic data it recovers known parameters, but on the real
data at R=40 it over-reaches: each position carries only ~20 null exceedance
events, the fitted multiplier spans ~3.5e4x where the measured heterogeneity
(347x across position, 6.1x across clade size) supports ~2e3x, and the
anchor-stability gate fails (moving `alpha0` from 1e-3 to 3e-4 moves the
multiplier ratio to 2.4e6x). It needs roughly 150 events per position, i.e.
~300 replicates.

This module is the fallback the plan names for that case: coarsen the
conditioning until every cell is well populated, and then use plain
empirical rates with no optimiser, no spline basis and nothing to overfit.
Cells are `n_hot` "hot" positions vs the rest, crossed with clade-size
deciles -- 20 cells at the defaults, each carrying ~164 null exceedances at
R=40.

    P(S > s | cell) = A(cell) * G(s)        for s >= the anchor u0

`G` is the pooled empirical null tail (`tail_model.EmpiricalTail`, shared
with the finer model rather than reimplemented) and `A(cell)` is the cell's
empirical exceedance rate over the pooled rate.

What conditioning on position does and does not do
--------------------------------------------------
Conditioning on the *observed* count per position would force every column
to yield the same number of switches -- that is the circular percentile rule
being retired. Conditioning on the *simulated null* rate per position only
normalises each column's noise floor: a column may still show far more
discoveries than its neighbours, it just has to beat its own noise to do so.

`event_type` is deliberately absent from the cell definition. It is
post-treatment relative to the scientific question, so conditioning on it
would remove real D/S/T signal along with the 1.2x null fluctuation. It is
used only as a negative control.

Integration contract
--------------------
Scores are transformed at BRANCH level to `z = -log10(A * G(s))` and the
left/right pair is combined with `fmax`, which is exactly what
`null_model._split_statistic` does. So `null_model.threshold_at_fdr` is
called completely unchanged on the z arrays, and "either branch exceeds"
survives verbatim.

Verification status
-------------------
The cross-validated numbers this module was promoted from are recorded in
`scripts/calibrate_switch_threshold.py`'s output, not asserted here. Nothing
in this file claims the model is correct for a dataset it has not been run
against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .tail_model import EmpiricalTail

# Defaults chosen so every cell stays well populated at R=40 (~164 null
# exceedances per cell). `N_HOT_DEFAULT` captures ~65% of the null
# exceedance mass in ~25 positions; `ANCHOR_ALPHA_DEFAULT` is a moderate
# anchor with most exceedances above it, so the extrapolation down to the
# ~3e-4 operating point is short.
N_HOT_DEFAULT = 25
N_CLADE_BINS_DEFAULT = 10
ANCHOR_ALPHA_DEFAULT = 3e-3


def clade_bin_edges(
    clade_left: np.ndarray, clade_right: np.ndarray, n_bins: int
) -> np.ndarray:
    """Interior log-clade-size bin edges, from the union of both branches.

    Both branches of a split are binned on one shared scale; binning them
    separately would make a left and a right branch of identical size land
    in different bins.
    """
    both = np.concatenate([np.asarray(clade_left, float), np.asarray(clade_right, float)])
    both = np.log(np.clip(both, 1.0, None))
    quantiles = np.quantile(both, np.linspace(0.0, 1.0, n_bins + 1))
    # Interior edges only -- np.digitize handles the open ends itself.
    edges = np.unique(quantiles[1:-1])
    # Drop edges at or below the data minimum. `--min-clade 5` puts a large
    # atom at the smallest clade size (10.6% of branches sit at exactly
    # log 5 on this dataset), so the low quantiles collapse onto it. Since
    # np.digitize places a value equal to an edge in the bin ABOVE it,
    # such an edge defines a bin that no test can ever fall into -- which
    # then shows up downstream as a spurious "empty cell". Requesting 10
    # bins on this data yields 9 real ones; the caller reads the achieved
    # count back off the returned edges rather than assuming it got what
    # it asked for.
    return edges[edges > both.min()]


def assign_clade_bins(clade_sizes: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    log_sizes = np.log(np.clip(np.asarray(clade_sizes, float), 1.0, None))
    return np.clip(np.digitize(log_sizes, edges), 0, n_bins - 1)


@dataclass
class CellConditionalModel:
    """Empirical exceedance-rate multiplier per (position group, clade bin).

    `multiplier` is `A(cell)`: the cell's null exceedance rate above
    `anchor_score` divided by the pooled rate. A cell with no null
    exceedances at all falls back to `A = 1.0` (the pooled rate) rather than
    to 0, which would make every test in it automatically significant. That
    fallback is conservative -- it holds such a cell to the pooled bar
    rather than a lower one -- and `n_empty_cells` records how often it
    fired.
    """

    hot_positions: np.ndarray       # actual position labels, not indices
    clade_edges: np.ndarray         # interior log-clade-size edges
    n_clade_bins: int
    multiplier: np.ndarray          # [2, n_clade_bins]; row 0 = cold, row 1 = hot
    multiplier_log_se: np.ndarray   # [2, n_clade_bins] jackknife SE of log A
    cell_events: np.ndarray         # [2, n_clade_bins] null exceedance counts
    cell_trials: np.ndarray         # [2, n_clade_bins] finite null draws
    anchor_alpha: float
    anchor_score: float
    pooled_rate: float
    n_empty_cells: int

    def multipliers_for(self, positions: np.ndarray, clade_sizes: np.ndarray) -> np.ndarray:
        """`A` for each test, from its position group and clade-size bin."""
        group = np.isin(np.asarray(positions), self.hot_positions).astype(int)
        bins = assign_clade_bins(clade_sizes, self.clade_edges, self.n_clade_bins)
        return self.multiplier[group, bins]

    def to_dict(self) -> dict:
        return {
            "hot_positions": np.asarray(self.hot_positions).tolist(),
            "clade_edges": np.asarray(self.clade_edges).tolist(),
            "n_clade_bins": int(self.n_clade_bins),
            "multiplier": np.asarray(self.multiplier).tolist(),
            "multiplier_log_se": np.asarray(self.multiplier_log_se).tolist(),
            "cell_events": np.asarray(self.cell_events).tolist(),
            "cell_trials": np.asarray(self.cell_trials).tolist(),
            "anchor_alpha": float(self.anchor_alpha),
            "anchor_score": float(self.anchor_score),
            "pooled_rate": float(self.pooled_rate),
            "n_empty_cells": int(self.n_empty_cells),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CellConditionalModel":
        return cls(
            hot_positions=np.asarray(d["hot_positions"]),
            clade_edges=np.asarray(d["clade_edges"], dtype=np.float64),
            n_clade_bins=int(d["n_clade_bins"]),
            multiplier=np.asarray(d["multiplier"], dtype=np.float64),
            multiplier_log_se=np.asarray(d["multiplier_log_se"], dtype=np.float64),
            cell_events=np.asarray(d["cell_events"], dtype=np.float64),
            cell_trials=np.asarray(d["cell_trials"], dtype=np.float64),
            anchor_alpha=float(d["anchor_alpha"]),
            anchor_score=float(d["anchor_score"]),
            pooled_rate=float(d["pooled_rate"]),
            n_empty_cells=int(d["n_empty_cells"]),
        )


def _cell_counts(
    null_left: np.ndarray,
    null_right: np.ndarray,
    group: np.ndarray,
    bin_left: np.ndarray,
    bin_right: np.ndarray,
    anchor: float,
    n_clade_bins: int,
    replicate_subset: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Null exceedance counts and finite-draw trial counts per cell.

    Trials count only FINITE null draws. Counting NaN draws as trials would
    deflate the rate of any cell whose tests are more often undefined, which
    would then be mistaken for a quiet cell.
    """
    events = np.zeros((2, n_clade_bins))
    trials = np.zeros((2, n_clade_bins))
    for values, bins in ((null_left, bin_left), (null_right, bin_right)):
        if replicate_subset is not None:
            values = values[np.asarray(replicate_subset)]
        finite = np.isfinite(values)
        exceed = finite & (values >= anchor)
        # Collapse over replicates first: per-test counts, then bin.
        per_test_exceed = exceed.sum(axis=0)
        per_test_finite = finite.sum(axis=0)
        cell = group * n_clade_bins + bins
        events += np.bincount(
            cell, weights=per_test_exceed, minlength=2 * n_clade_bins
        ).reshape(2, n_clade_bins)
        trials += np.bincount(
            cell, weights=per_test_finite, minlength=2 * n_clade_bins
        ).reshape(2, n_clade_bins)
    return events, trials


def _rates_to_multiplier(events: np.ndarray, trials: np.ndarray) -> Tuple[np.ndarray, float, int]:
    pooled = events.sum() / trials.sum() if trials.sum() > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(trials > 0, events / trials, np.nan)
        multiplier = rate / pooled
    empty = ~(np.isfinite(multiplier) & (multiplier > 0))
    multiplier = np.where(empty, 1.0, multiplier)
    return multiplier, float(pooled), int(empty.sum())


def fit_cell_model(
    null_left: np.ndarray,
    null_right: np.ndarray,
    positions: np.ndarray,
    clade_left: np.ndarray,
    clade_right: np.ndarray,
    tail: EmpiricalTail,
    *,
    n_hot: int = N_HOT_DEFAULT,
    n_clade_bins: int = N_CLADE_BINS_DEFAULT,
    anchor_alpha: float = ANCHOR_ALPHA_DEFAULT,
) -> CellConditionalModel:
    """Fit the cell multipliers from null replicates only.

    `null_left`/`null_right` have shape `(R, n_tests)`; `positions`,
    `clade_left`, `clade_right` are per-test and share the key order of the
    observed table. Nothing from the observed scores enters this fit -- that
    is what keeps the rule non-circular.

    `tail` supplies the anchor score via `tail.quantile_at(anchor_alpha)`,
    so the anchor is defined on the same pooled tail the z-transform later
    divides by.
    """
    null_left = np.asarray(null_left, dtype=np.float64)
    null_right = np.asarray(null_right, dtype=np.float64)
    positions = np.asarray(positions)
    if null_left.ndim != 2 or null_right.ndim != 2:
        raise ValueError("null_left and null_right must be 2-D (R, n_tests).")
    if null_left.shape != null_right.shape:
        raise ValueError("null_left and null_right must have the same shape.")
    n_replicates, n_tests = null_left.shape
    for name, arr in (("positions", positions), ("clade_left", clade_left),
                      ("clade_right", clade_right)):
        if len(arr) != n_tests:
            raise ValueError(
                f"{name} has length {len(arr)} but the null arrays have "
                f"{n_tests} tests."
            )
    if not 0 < anchor_alpha < 1:
        raise ValueError("anchor_alpha must be in (0, 1).")

    anchor = float(np.asarray(tail.quantile_at(anchor_alpha)))

    edges = clade_bin_edges(clade_left, clade_right, n_clade_bins)
    # Duplicate quantile edges are dropped, so the achieved bin count can be
    # lower than requested; everything downstream uses the achieved one.
    n_clade_bins = len(edges) + 1
    bin_left = assign_clade_bins(clade_left, edges, n_clade_bins)
    bin_right = assign_clade_bins(clade_right, edges, n_clade_bins)

    # Hot positions are chosen from these null replicates alone. Under
    # cross-validation this must be re-derived per fold, or the held-out
    # replicates leak into the model through the choice of which columns
    # are "hot".
    unique_positions = np.unique(positions)
    position_index = np.searchsorted(unique_positions, positions)
    exceed_per_test = (
        (np.isfinite(null_left) & (null_left >= anchor)).sum(axis=0)
        + (np.isfinite(null_right) & (null_right >= anchor)).sum(axis=0)
    )
    per_position = np.bincount(
        position_index, weights=exceed_per_test, minlength=len(unique_positions)
    )
    n_hot = min(n_hot, len(unique_positions))
    hot_positions = unique_positions[np.argsort(-per_position, kind="stable")[:n_hot]]
    group = np.isin(positions, hot_positions).astype(int)

    events, trials = _cell_counts(
        null_left, null_right, group, bin_left, bin_right, anchor, n_clade_bins
    )
    multiplier, pooled, n_empty = _rates_to_multiplier(events, trials)

    # Delete-one-replicate jackknife on log A. The independent unit is the
    # replicate (all ~1607 splits at one position in one replicate share a
    # simulated column and one ASR reconstruction), so a Poisson SE on the
    # raw event count would badly understate the uncertainty.
    log_se = np.full_like(multiplier, np.nan, dtype=np.float64)
    if n_replicates > 1:
        loo = []
        for r in range(n_replicates):
            keep = np.setdiff1d(np.arange(n_replicates), [r])
            e_r, t_r = _cell_counts(
                null_left, null_right, group, bin_left, bin_right, anchor,
                n_clade_bins, replicate_subset=keep,
            )
            m_r, _, _ = _rates_to_multiplier(e_r, t_r)
            loo.append(np.log(m_r))
        loo = np.stack(loo)
        mean = loo.mean(axis=0)
        log_se = np.sqrt((n_replicates - 1) / n_replicates * ((loo - mean) ** 2).sum(axis=0))

    return CellConditionalModel(
        hot_positions=hot_positions,
        clade_edges=edges,
        n_clade_bins=n_clade_bins,
        multiplier=multiplier,
        multiplier_log_se=log_se,
        cell_events=events,
        cell_trials=trials,
        anchor_alpha=anchor_alpha,
        anchor_score=anchor,
        pooled_rate=pooled,
        n_empty_cells=n_empty,
    )


def to_z(
    scores: np.ndarray,
    positions: np.ndarray,
    clade_sizes: np.ndarray,
    model: CellConditionalModel,
    tail: EmpiricalTail,
) -> np.ndarray:
    """`z = -log10(A(cell) * G(s))`, the cell-standardised branch statistic.

    Higher z means more extreme relative to that test's own null. Within a
    single cell `A` is constant and `G` is strictly decreasing, so
    thresholding on z reproduces thresholding on the raw score exactly --
    the conditioning only changes how cells compare to each other.

    NaN scores stay NaN, so `null_model`'s "NaN-both splits never count"
    rule is preserved.
    """
    scores = np.asarray(scores, dtype=np.float64)
    multipliers = model.multipliers_for(positions, clade_sizes)
    if multipliers.shape[-1] != scores.shape[-1]:
        raise ValueError(
            f"scores last axis is {scores.shape[-1]} but positions/clade_sizes "
            f"give {multipliers.shape[-1]} tests."
        )
    survival = tail.survival(scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.clip(multipliers * survival, 1e-300, None))


def t_from_z(
    z: float,
    positions: np.ndarray,
    clade_sizes: np.ndarray,
    model: CellConditionalModel,
    tail: EmpiricalTail,
) -> np.ndarray:
    """Invert `to_z`: the raw score each test must reach to hit threshold `z`.

    This is what makes a conditioned call set readable -- a reader can see
    which positions were held to a higher bar, in the units the scores are
    reported in.
    """
    multipliers = model.multipliers_for(positions, clade_sizes)
    target_survival = np.power(10.0, -float(z)) / multipliers
    return tail.quantile_at(target_survival)
