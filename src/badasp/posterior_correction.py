"""Correct the simulated null's ASR posterior (`p_AC`) to match the observed data.

The problem this addresses
--------------------------
The BADASP score is `RC - AC * p_AC`. The simulated null reproduces `RC`
faithfully but not `p_AC`: because the null's sequences are generated under
(nearly) the model the ancestral reconstruction assumes, the reconstruction
is more confident on simulated data than on real data. Measured on the
`AC = +1` stratum, substituting the observed `p_AC` into the null score
moves the observed/null ratio from 3.21/4.19 to 0.99/0.99, while
substituting the observed `RC` moves it not at all -- so `p_AC` carries the
entire discrepancy.

Because `AC` flips the sign of `p_AC`, one defect produces two opposite
symptoms: the null runs too cold where ancestors agree and too hot where
they differ. The second is what matters, since that is where switches live.

Why the correction is fitted where it is
----------------------------------------
`AC` is decided by the same posterior that `p_AC` reports -- a reconstruction
is likelier to call "ancestors differ" when it is confident -- so
stratifying on `AC` selects on `p_AC` and the two strata are NOT
exchangeable. Measured: a correction fitted on `AC = +1` mispredicts the
`AC = -1` discrepancy by ~38% even in bins where neither stratum can
contain a switch.

So the correction is fitted INSIDE `AC = -1`, on its low-`RC` rows. Those
are signal-free by BADASP's own definition: a switch requires the clade to
be internally conserved, so a diverse clade cannot be one. Within
`AC = -1` the discrepancy is close to constant in `RC` (mean gap 0.020,
0.044, 0.038, 0.034, 0.031 across the signal-free deciles), which is what
makes extrapolation to the conserved bins a mild step rather than a leap --
and it is checkable by holding out signal-free bins.

Only `AC = -1` needs correcting in practice: at `AC = +1` the score is
`RC - p_AC <= 1`, well below any calling threshold, so those rows never
enter the discovery set.

Method
------
Randomized quantile mapping per `RC` bin. `p_AC` has a large atom at
exactly 1.0, and a deterministic map cannot split an atom -- it would send
every null `p_AC = 1.0` to a single value, when the correct behaviour is to
spread them so the corrected atom matches the observed one. Randomizing
within each value's CDF jump (the standard distributional transform) does
that, and reduces to ordinary quantile mapping wherever the distribution is
continuous. It needs a seed to stay reproducible.

Verification status
-------------------
Nothing here asserts the correction works. `fit_posterior_correction`
records which bins it was fitted on so a caller can hold bins out, and the
score-level controls are computed by the calibration script, not claimed
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# Quantile-grid resolution used to represent each bin's distribution. Large
# enough that an atom shows up as a long plateau, which is what lets the
# randomized map recover the size of its CDF jump.
N_QUANTILE_GRID_DEFAULT = 4001

# Fraction of the RC range treated as signal-free when no explicit mask is
# given. A switch requires an internally conserved clade, so the least
# conserved rows cannot be switches; the bottom half is a deliberately
# conservative default, and `fitted_bins` records what was actually used.
SIGNAL_FREE_RC_QUANTILE_DEFAULT = 0.5

N_RC_BINS_DEFAULT = 10


def rc_bin_edges(rc: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior RC bin edges at equally spaced quantiles of `rc`.

    Duplicate edges are dropped (RC has a large atom at 1.0, which would
    otherwise define bins nothing can fall into), so the achieved bin count
    can be lower than requested -- read it back off the returned array.
    """
    rc = np.asarray(rc, dtype=np.float64)
    rc = rc[np.isfinite(rc)]
    edges = np.unique(np.quantile(rc, np.linspace(0.0, 1.0, n_bins + 1))[1:-1])
    return edges[edges > rc.min()]


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(np.asarray(values, dtype=np.float64), edges), 0, len(edges))


@dataclass
class PosteriorCorrection:
    """A per-`RC`-bin randomized quantile map from null `p_AC` to observed `p_AC`.

    `source_quantiles[b]` and `target_quantiles[b]` are the null and observed
    `p_AC` quantile functions for bin `b`, on the shared probability grid
    `quantile_grid`. Bins listed in `fitted_bins` were estimated from data;
    every other bin reuses `pooled_source`/`pooled_target`, the maps built
    from all fitted bins together.
    """

    rc_edges: np.ndarray
    quantile_grid: np.ndarray
    source_quantiles: np.ndarray      # [n_bins, n_grid]
    target_quantiles: np.ndarray      # [n_bins, n_grid]
    pooled_source: np.ndarray         # [n_grid] mean over fitted bins, reporting only
    pooled_target: np.ndarray         # [n_grid] mean over fitted bins, reporting only
    fitted_bins: np.ndarray           # bin indices estimated from data
    n_fitted_source: np.ndarray       # draws behind each fitted bin
    n_fitted_target: np.ndarray

    @property
    def n_bins(self) -> int:
        return len(self.rc_edges) + 1

    def apply(self, rc: np.ndarray, p_ac: np.ndarray, seed: int) -> np.ndarray:
        """Map null `p_AC` values onto the observed scale, given their `RC`.

        `seed` is required rather than optional: the map is randomized
        within each value's CDF jump, so an unseeded call would silently
        make the calibration irreproducible.
        """
        rc = np.asarray(rc, dtype=np.float64)
        p_ac = np.asarray(p_ac, dtype=np.float64)
        if rc.shape != p_ac.shape:
            raise ValueError(
                f"rc has shape {rc.shape} but p_ac has shape {p_ac.shape}; "
                "they must describe the same tests."
            )
        rng = np.random.default_rng(seed)
        bins = assign_bins(rc, self.rc_edges)
        fitted = set(int(b) for b in self.fitted_bins)
        out = np.array(p_ac, dtype=np.float64, copy=True)

        for b in range(self.n_bins):
            mask = (bins == b) & np.isfinite(p_ac)
            if not mask.any():
                continue
            # Every bin has its own source/target now: fitted bins from
            # data, the rest reconstructed as source + the borrowed shift.
            src, tgt = self.source_quantiles[b], self.target_quantiles[b]
            out[mask] = _randomized_map(p_ac[mask], src, tgt, self.quantile_grid, rng)
        return out

    def mean_shift_by_bin(self) -> np.ndarray:
        """Mean (null - observed) `p_AC` per fitted bin: the gap being removed."""
        return np.array([
            float(self.source_quantiles[b].mean() - self.target_quantiles[b].mean())
            for b in self.fitted_bins
        ])

    def to_dict(self) -> dict:
        return {
            "rc_edges": self.rc_edges.tolist(),
            "quantile_grid": self.quantile_grid.tolist(),
            "source_quantiles": self.source_quantiles.tolist(),
            "target_quantiles": self.target_quantiles.tolist(),
            "pooled_source": self.pooled_source.tolist(),
            "pooled_target": self.pooled_target.tolist(),
            "fitted_bins": self.fitted_bins.tolist(),
            "n_fitted_source": self.n_fitted_source.tolist(),
            "n_fitted_target": self.n_fitted_target.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PosteriorCorrection":
        arr = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(
            rc_edges=arr("rc_edges"),
            quantile_grid=arr("quantile_grid"),
            source_quantiles=arr("source_quantiles"),
            target_quantiles=arr("target_quantiles"),
            pooled_source=arr("pooled_source"),
            pooled_target=arr("pooled_target"),
            fitted_bins=np.asarray(d["fitted_bins"], dtype=int),
            n_fitted_source=np.asarray(d["n_fitted_source"], dtype=int),
            n_fitted_target=np.asarray(d["n_fitted_target"], dtype=int),
        )


def _randomized_map(
    values: np.ndarray,
    source_q: np.ndarray,
    target_q: np.ndarray,
    grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Distributional transform: value -> uniform draw inside its CDF jump ->
    inverse target CDF.

    An atom appears as a plateau in `source_q`; searching the plateau from
    the left and the right recovers the size of that value's CDF jump, and
    drawing uniformly inside it is what lets a single input value map onto a
    range of outputs. Where the distribution is continuous the jump has zero
    width and this reduces to ordinary quantile mapping.
    """
    lo_idx = np.searchsorted(source_q, values, side="left")
    hi_idx = np.searchsorted(source_q, values, side="right")
    last = len(grid) - 1
    lo = grid[np.clip(lo_idx, 0, last)]
    hi = grid[np.clip(hi_idx - 1, 0, last)]
    hi = np.maximum(hi, lo)
    u = lo + rng.random(values.shape) * (hi - lo)
    return np.interp(u, grid, target_q)


def fit_posterior_correction(
    null_rc: np.ndarray,
    null_p_ac: np.ndarray,
    observed_rc: np.ndarray,
    observed_p_ac: np.ndarray,
    *,
    n_rc_bins: int = N_RC_BINS_DEFAULT,
    signal_free_rc_quantile: float = SIGNAL_FREE_RC_QUANTILE_DEFAULT,
    fit_bins: Optional[Sequence[int]] = None,
    n_quantile_grid: int = N_QUANTILE_GRID_DEFAULT,
    min_per_bin: int = 200,
) -> PosteriorCorrection:
    """Fit the `p_AC` correction on signal-free rows only.

    Every array must already be restricted to the stratum being corrected
    (in practice `AC = -1`) and flattened to branch level. `null_*` may
    pool replicates.

    Bins are chosen from the OBSERVED `RC` distribution so the same binning
    applies to both sides. By default the bins below
    `signal_free_rc_quantile` of observed `RC` are treated as signal-free
    and used for fitting; pass `fit_bins` explicitly to hold bins out for
    validation. Bins that are not fitted receive the pooled map.
    """
    null_rc = np.asarray(null_rc, dtype=np.float64).ravel()
    null_p_ac = np.asarray(null_p_ac, dtype=np.float64).ravel()
    observed_rc = np.asarray(observed_rc, dtype=np.float64).ravel()
    observed_p_ac = np.asarray(observed_p_ac, dtype=np.float64).ravel()
    if null_rc.shape != null_p_ac.shape:
        raise ValueError("null_rc and null_p_ac must have the same shape.")
    if observed_rc.shape != observed_p_ac.shape:
        raise ValueError("observed_rc and observed_p_ac must have the same shape.")
    if not 0.0 < signal_free_rc_quantile <= 1.0:
        raise ValueError("signal_free_rc_quantile must be in (0, 1].")

    edges = rc_bin_edges(observed_rc, n_rc_bins)
    n_bins = len(edges) + 1
    grid = np.linspace(0.0, 1.0, n_quantile_grid)

    obs_bins = assign_bins(observed_rc, edges)
    null_bins = assign_bins(null_rc, edges)

    if fit_bins is None:
        cutoff = np.quantile(observed_rc[np.isfinite(observed_rc)], signal_free_rc_quantile)
        candidate = [b for b in range(n_bins) if b <= assign_bins(np.array([cutoff]), edges)[0]]
    else:
        candidate = [int(b) for b in fit_bins]

    # The NULL side is available in every bin -- the null contains no
    # switches anywhere -- so each bin gets its own source quantiles. Only
    # the OBSERVED side has to be restricted to signal-free bins.
    source_q = np.zeros((n_bins, n_quantile_grid))
    for b in range(n_bins):
        sm = (null_bins == b) & np.isfinite(null_p_ac)
        if sm.sum():
            source_q[b] = np.quantile(null_p_ac[sm], grid)

    target_q = np.zeros((n_bins, n_quantile_grid))
    fitted, n_src, n_tgt = [], [], []
    for b in candidate:
        sm = (null_bins == b) & np.isfinite(null_p_ac)
        tm = (obs_bins == b) & np.isfinite(observed_p_ac)
        if sm.sum() < min_per_bin or tm.sum() < min_per_bin:
            continue
        target_q[b] = np.quantile(observed_p_ac[tm], grid)
        fitted.append(b); n_src.append(int(sm.sum())); n_tgt.append(int(tm.sum()))

    if not fitted:
        raise ValueError(
            f"No RC bin had at least {min_per_bin} draws on both sides; "
            "cannot fit a correction."
        )

    # Extrapolate the correction MAGNITUDE, not the map. A quantile map
    # transports one distribution onto another, so reusing a low-RC map in a
    # conserved bin would mis-rank every value -- `p_AC`'s distribution
    # shifts with RC, and a value at its own bin's median can sit at the
    # pooled distribution's 95th percentile. Instead each bin keeps its own
    # source quantiles and borrows only the quantile-wise shift
    # `target - source` averaged over the fitted bins, which is exactly what
    # "the gap is constant in RC" means operationally.
    delta = np.mean([target_q[b] - source_q[b] for b in fitted], axis=0)
    for b in range(n_bins):
        if b not in fitted:
            target_q[b] = np.maximum.accumulate(np.clip(source_q[b] + delta, 0.0, 1.0))

    pooled_source = np.mean([source_q[b] for b in fitted], axis=0)
    pooled_target = np.mean([target_q[b] for b in fitted], axis=0)

    return PosteriorCorrection(
        rc_edges=edges,
        quantile_grid=grid,
        source_quantiles=source_q,
        target_quantiles=target_q,
        pooled_source=pooled_source,
        pooled_target=pooled_target,
        fitted_bins=np.asarray(fitted, dtype=int),
        n_fitted_source=np.asarray(n_src, dtype=int),
        n_fitted_target=np.asarray(n_tgt, dtype=int),
    )


def corrected_null_score(
    rc: np.ndarray, ac: np.ndarray, p_ac: np.ndarray,
    correction: PosteriorCorrection, seed: int,
) -> np.ndarray:
    """`RC - AC * corrected(p_AC)`, correcting only the `AC = -1` rows.

    `AC = +1` rows are left untouched: their score is `RC - p_AC <= 1`, far
    below any calling threshold, so they never enter the discovery set and
    correcting them would only add randomization noise.
    """
    rc = np.asarray(rc, dtype=np.float64)
    ac = np.asarray(ac, dtype=np.float64)
    p_ac = np.asarray(p_ac, dtype=np.float64)
    out = np.array(p_ac, dtype=np.float64, copy=True)
    target = ac == -1.0
    if target.any():
        out[target] = correction.apply(rc[target], p_ac[target], seed=seed)
    return rc - ac * out
