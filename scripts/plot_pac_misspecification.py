#!/usr/bin/env python3
"""
plot_pac_misspecification.py

Three figures describing why the simulated null's ancestral-state posterior
(`p_AC`) is mis-specified, what that does to the switch calibration, and how
much of it the correction in src/badasp/posterior_correction.py removes.

Intended for discussion, not as a claim that any of it is resolved -- the
third panel shows the part that is not.

  1. p_ac_distribution.png   the defect itself: the null's `p_AC = 1.0` atom
                             is ~3x oversized, and how that varies with RC
  2. pac_sign_flip.png       why one defect produces two opposite symptoms,
                             since `AC` flips the sign of `p_AC`
  3. pac_correction.png      held-out calibration before and after the
                             correction, across tail probabilities
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.badasp.posterior_correction import (  # noqa: E402
    assign_bins,
    fit_posterior_correction,
    rc_bin_edges,
)

DPI = 150
OBS_COLOUR = "#2c3e50"
NULL_COLOUR = "#c0392b"
FIXED_COLOUR = "#1e8449"


def load(null_run_dir: Path, observed_scores: Path):
    obs = pd.read_csv(observed_scores)
    files = sorted((null_run_dir / "npz").glob("rep_*.npz")) or sorted(
        null_run_dir.glob("rep_*.npz")
    )
    if not files:
        raise SystemExit(f"No rep_*.npz found under {null_run_dir}")
    n_ac = np.stack([np.load(f)["ac"] for f in files]).astype(float)
    cols = {}
    for name in ("rc_left", "rc_right", "p_ac_left", "p_ac_right",
                 "badasp_score_left", "badasp_score_right"):
        cols[name] = np.stack([np.load(f)[name] for f in files]).astype(float)
    return obs, cols, n_ac, len(files)


def branch_arrays(obs, cols, n_ac, want_ac):
    """Observed and null (rc, p_ac) for one AC stratum, both branches pooled."""
    o_ac = obs["ac"].to_numpy(float)
    mo, mn = o_ac == want_ac, n_ac == want_ac
    o_rc = np.concatenate([obs[f"rc_{s}"].to_numpy(float)[mo] for s in ("left", "right")])
    o_p = np.concatenate([obs[f"p_ac_{s}"].to_numpy(float)[mo] for s in ("left", "right")])
    n_rc = np.concatenate([cols[f"rc_{s}"][mn] for s in ("left", "right")])
    n_p = np.concatenate([cols[f"p_ac_{s}"][mn] for s in ("left", "right")])
    return o_rc, o_p, n_rc, n_p


def fig_distribution(obs, cols, n_ac, out_path):
    o_rc, o_p, n_rc, n_p = branch_arrays(obs, cols, n_ac, -1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    # Full 0-1 range, not a zoom on the tail: a third of the observed values
    # lie below 0.90, and cropping there makes p_AC look far more
    # concentrated near 1 than it is. Values at exactly 1.0 are drawn as a
    # separate pair of bars, because a point mass has no density.
    # Cumulative rather than a histogram: it reads off directly how much mass
    # sits below any value, and the point mass at exactly 1.0 shows up as the
    # vertical jump at the right edge instead of being invisible.
    o_atom, n_atom = float((o_p == 1.0).mean()), float((n_p == 1.0).mean())
    grid = np.linspace(0.0, 1.0, 501)
    for values, colour, label in ((o_p, OBS_COLOUR, "observed"),
                                  (n_p, NULL_COLOUR, "simulated null")):
        v = np.sort(values[np.isfinite(values)])
        # P(p_AC < x), so the jump to 1.0 at the right edge IS the point mass.
        ax.step(grid, np.searchsorted(v, grid, side="left") / len(v),
                where="post", lw=2, color=colour, label=label)
    for frac, colour in ((1 - o_atom, OBS_COLOUR), (1 - n_atom, NULL_COLOUR)):
        ax.plot([1.0, 1.0], [frac, 1.0], lw=3, color=colour, alpha=0.9)
    ax.axvline(0.90, ls=":", color="grey", lw=1)
    ax.text(0.87, 0.50,
            f"below 0.90:\nobserved {np.mean(o_p < 0.90):.0%}\nnull {np.mean(n_p < 0.90):.0%}",
            fontsize=7, ha="right", va="center")
    ax.annotate(
        f"vertical jump at exactly 1.0\n= the 'atom' (a point mass):\n"
        f"observed {o_atom:.1%}, null {n_atom:.1%}  ({n_atom / o_atom:.1f}x)",
        xy=(1.0, 1 - n_atom * 0.6), xytext=(0.44, 0.13), fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=1, color="black"))
    ax.set_xlim(0.0, 1.06); ax.set_ylim(0, 1.04)
    ax.set_xlabel("posterior probability of the ancestral call, $p_{AC}$")
    ax.set_ylabel("cumulative fraction of branches")
    ax.set_title("Where ancestors differ (AC = $-$1)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    edges = rc_bin_edges(o_rc, 10)
    ob, nb = assign_bins(o_rc, edges), assign_bins(n_rc, edges)
    xs, ratio, rc_mid = [], [], []
    for b in range(len(edges) + 1):
        m_o, m_n = ob == b, nb == b
        if m_o.sum() < 200 or m_n.sum() < 200:
            continue
        a = float((o_p[m_o] == 1.0).mean())
        c = float((n_p[m_n] == 1.0).mean())
        if a > 0:
            xs.append(b); ratio.append(c / a); rc_mid.append(float(np.median(o_rc[m_o])))
    ax.plot(rc_mid, ratio, "o-", color=NULL_COLOUR, lw=2)
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.text(rc_mid[0], 1.06, "no mis-specification", fontsize=7, color="grey")
    ax.set_xlabel("within-clade conservation, RC (bin median)")
    ax.set_ylabel("null / observed atom size")
    ax.set_title("The defect is not constant: the null's over-confidence\n"
                 "shrinks as clades get more conserved", fontsize=9)
    ax.set_ylim(0, max(ratio) * 1.15)

    fig.suptitle("The simulated null's ancestral reconstruction is over-confident",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return {"observed_atom": o_atom, "null_atom": n_atom,
            "ratio_by_rc": list(zip(rc_mid, ratio))}


def _label_alpha_axis(ax, alphas):
    """Label the tail-probability axis as 'top X% of null scores'.

    `alpha` is the fraction of simulated-null scores above the threshold, so
    alpha = 0.01 means 'the bar that only the top 1% of null comparisons
    clear'. Spelling that out avoids the reader having to translate a bare
    probability, and 'tail' into plain language.
    """
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(alphas)
    ax.set_xticklabels([("top %g%%" % (a * 100)) for a in alphas], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("how high a score we demand\n(kept: the top % of simulated-null comparisons)")


def _oe(o_stat, n_stat, alpha, n_rep):
    o_stat = o_stat[np.isfinite(o_stat)]
    n_stat = n_stat[np.isfinite(n_stat)]
    t = np.quantile(n_stat, 1 - alpha)
    e = (n_stat >= t).sum() / n_rep
    o = int((o_stat >= t).sum())
    return (o / e) if e > 0 else np.nan


def fig_sign_flip(obs, cols, n_ac, n_rep, out_path):
    alphas = np.array([3e-2, 1e-2, 3e-3, 1e-3, 3e-4])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    results = {}
    for want, label, style in ((1.0, "AC = +1  (cannot be a switch)", "s--"),
                               (-1.0, "AC = $-$1  (where switches are)", "o-")):
        o_rc, o_p, n_rc, n_p = branch_arrays(obs, cols, n_ac, want)
        o_stat, n_stat = o_rc - want * o_p, n_rc - want * n_p
        ys = [_oe(o_stat, n_stat, a, n_rep) for a in alphas]
        results[label] = ys
        ax.plot(alphas, ys, style, lw=2, label=label)
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_yscale("log")
    _label_alpha_axis(ax, alphas)
    ax.set_ylabel("real switches found / number the null\nproduces by chance  (1.0 = calibrated)")
    ax.set_title("One defect, two opposite symptoms\n"
                 "$p_{AC}$ enters the score as $RC - AC \\cdot p_{AC}$, so its sign flips",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.text(0.97, 0.95, "above 1: null too COLD --\nmanufactures apparent signal\nwhere none can exist",
            transform=ax.transAxes, fontsize=7, ha="right", va="top", color="#1f5fa8")
    ax.text(0.97, 0.06, "below 1: null too HOT --\nburies real switches",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom", color="#d2691e")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return {k: [float(v) for v in vs] for k, vs in results.items()}


def fig_correction(obs, cols, n_ac, n_rep, out_path, fit_bins=(0, 1, 2), held=(3, 4)):
    """Held-out calibration: the correction never sees bins `held`."""
    o_rc, o_p, n_rc, n_p = branch_arrays(obs, cols, n_ac, -1.0)
    edges = rc_bin_edges(o_rc, 10)
    ob, nb = assign_bins(o_rc, edges), assign_bins(n_rc, edges)
    correction = fit_posterior_correction(n_rc, n_p, o_rc, o_p, fit_bins=list(fit_bins))
    fixed_p = correction.apply(n_rc, n_p, seed=20260816)

    m_o, m_n = np.isin(ob, held), np.isin(nb, held)
    o_stat = o_rc[m_o] + o_p[m_o]
    alphas = np.array([3e-2, 1e-2, 3e-3, 1e-3])
    series = {
        "simulated null, uncorrected": (n_rc[m_n] + n_p[m_n], NULL_COLOUR, "o-"),
        "after correcting $p_{AC}$": (n_rc[m_n] + fixed_p[m_n], FIXED_COLOUR, "s-"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    out = {}
    for label, (n_stat, colour, style) in series.items():
        ys = [_oe(o_stat, n_stat, a, n_rep) for a in alphas]
        out[label] = [float(v) for v in ys]
        ax.plot(alphas, ys, style, lw=2, color=colour, label=label)
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    _label_alpha_axis(ax, alphas)
    ax.set_ylabel("real switches found / number the null\nproduces by chance  (1.0 = calibrated)")
    ax.set_title(
        f"Correction validated on held-out signal-free data\n"
        f"(fitted on RC bins {list(fit_bins)}, evaluated on bins {list(held)} only)",
        fontsize=9, pad=10)
    ax.set_ylim(-0.05, 1.55)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    # Both curves sit BELOW 1.0, i.e. the null produces more high scores than
    # reality does. That is the null running hot, which hides real switches --
    # the conservative direction. Spelled out because "below 1" is easy to
    # misread as the harmless case.
    ax.text(0.02, 0.94, "1.0 = null behaves like the real data.\n"
                        "Below 1.0 the null runs HOT: it invents high scores,\n"
                        "so real switches get buried. Errs on the safe side.",
            transform=ax.transAxes, fontsize=7, color="#7b241c", va="top")
    ax.text(0.36, 0.60, "calibrated across this range", transform=ax.transAxes,
            fontsize=7, color=FIXED_COLOUR, ha="center")
    ax.annotate("still short here: the null is\nstill too hot at the very top,\n"
                "so this end stays conservative\n(unresolved)",
                xy=(1.05e-3, 0.44), xytext=(8e-3, 0.10), fontsize=7, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--null-run-dir", type=Path, required=True)
    p.add_argument("--observed-scores", type=Path,
                   default=REPO_ROOT / "results/badasp_scoring/raw_node_scores.csv")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results/badasp_scoring/null_calibration/figures")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    obs, cols, n_ac, n_rep = load(args.null_run_dir, args.observed_scores)
    print(f"{n_rep} null replicates, {len(obs):,} observed tests")

    for name, fn in (
        ("p_ac_distribution.png", lambda pth: fig_distribution(obs, cols, n_ac, pth)),
        ("pac_sign_flip.png", lambda pth: fig_sign_flip(obs, cols, n_ac, n_rep, pth)),
        ("pac_correction.png", lambda pth: fig_correction(obs, cols, n_ac, n_rep, pth)),
    ):
        path = args.out_dir / name
        summary = fn(path)
        print(f"Wrote {path}")
        print(f"   {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
