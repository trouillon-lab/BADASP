#!/usr/bin/env python3
"""
calibrate_switch_threshold.py

Aggregates the null-calibration replicate .npz files produced by
run_null_calibration.py / score_null_replicate.py against the observed
score table, and emits the calibrated switch threshold plus the mandatory
diagnostics.

This script is aggregation and reporting only. Every piece of statistical
machinery -- the FDP-exceedance threshold search, Westfall-Young max-T FWER
thresholds, matched empirical p-values/BH q-values, and the exceedance-
flatness diagnostic -- comes from src/badasp/null_model.py and is not
reimplemented here.

Why the .npz format doesn't go through null_model.load_null_scores
--------------------------------------------------------------------
null_model.load_null_scores() reads a *different* on-disk shape (arrays
named ``score_left``/``score_right`` plus ``node_name``/``position`` key
arrays) than what score_null_replicate.py actually writes (its own
``write_npz``: ``SCORE_COLUMNS`` = rc_left, rc_right, ac, p_ac_left,
p_ac_right, badasp_score_left, badasp_score_right, with NO key arrays --
each replicate is pre-aligned, once, to the observed table's own row order
at score time). Since score_null_replicate.py cannot be modified, this
script reads its actual format directly (a small amount of I/O glue, not
"maths") and hands the resulting stacked arrays to null_model's functions.

Observed-table / null alignment safety
-----------------------------------------
Every null replicate was aligned to a *specific* observed-scores CSV's row
order at the moment score_null_replicate.py wrote it (see its write_npz
docstring). If that CSV is later regenerated with a different row set (as
is happening concurrently elsewhere in this repo -- a polytomy fix changes
the node set), reusing the replicates against the new file would silently
misalign every test. This script therefore defaults --observed-scores to
the exact path recorded in the calibration run's own run_manifest.json (see
run_null_calibration.py) and refuses to proceed if the file at that path no
longer matches the size recorded at run time, rather than silently
assuming the current contents are still the ones the null was built from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "snakemake.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "badasp_scoring" / "null_calibration"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_null_replicate import SCORE_COLUMNS  # noqa: E402
from src.badasp.null_model import (  # noqa: E402
    ThresholdResult,
    as_bin_threshold_dict,
    exceedance_flatness,
    load_observed_scores,
    maxT_fwer_thresholds,
    per_test_pvalues,
    threshold_at_fdr,
)
# `_split_statistic` is the shared, NaN-safe fmax(left, right) device the
# rest of null_model.py uses to turn a pair of branch scores into one
# per-split statistic; reused here (not re-derived) for the same reason
# every other call site in null_model.py uses it: `NaN >= t` is always
# False, so a split with one real and one NaN branch is still tested on its
# real branch, and a split with both NaN never contributes to any count.
from src.badasp.null_model import _split_statistic  # noqa: E402

from scripts.plot_decoupled_event_switches_clade_adjusted import (  # noqa: E402
    bin_clade_sizes,
    calculate_bin_thresholds,
    identify_switches,
)


# ---------------------------------------------------------------------------
# Small shared helpers (mirrors run_null_calibration.py's; kept local so this
# script stays independently runnable without importing that one).
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def file_info(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def git_sha(root: Path) -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Interval):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Loading null replicates (score_null_replicate.py's actual .npz format)
# ---------------------------------------------------------------------------


def load_null_replicates(null_dir: Path, n_tests: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Path]]:
    """Stack every rep_*.npz in `null_dir` into (R, n_tests) arrays.

    Each file is score_null_replicate.py's own write_npz() output: the
    SCORE_COLUMNS arrays only, pre-aligned to the observed table's row
    order (see module docstring). A file whose length does not match
    `n_tests` is rejected loudly rather than silently reshaped/dropped,
    since a length mismatch means it was aligned to a *different* observed
    table than the one this run is using.
    """
    files = sorted(Path(null_dir).glob("rep_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rep_*.npz files found in {null_dir}")

    left_chunks, right_chunks, ac_chunks = [], [], []
    used: List[Path] = []
    for f in files:
        try:
            with np.load(f) as data:
                missing = [c for c in SCORE_COLUMNS if c not in data.files]
                if missing:
                    warnings.warn(f"Skipping {f}: missing arrays {missing}")
                    continue
                n = data["badasp_score_left"].shape[0]
                if n != n_tests:
                    raise ValueError(
                        f"{f} has {n} tests but the observed table has {n_tests}; "
                        "this replicate was aligned to a different observed-scores "
                        "CSV and cannot be mixed into this calibration."
                    )
                left_chunks.append(data["badasp_score_left"].astype(np.float64))
                right_chunks.append(data["badasp_score_right"].astype(np.float64))
                ac_chunks.append(data["ac"].astype(np.float64))
                used.append(f)
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and "aligned to a different" in str(exc):
                raise
            warnings.warn(f"Skipping unreadable replicate {f}: {exc}")

    if not left_chunks:
        raise FileNotFoundError(f"No valid rep_*.npz replicates could be loaded from {null_dir}")

    return np.stack(left_chunks), np.stack(right_chunks), np.stack(ac_chunks), used


def check_observed_scores_consistency(observed_scores: Path, manifest: Optional[dict]) -> Optional[str]:
    """Guard against aggregating null replicates against a *different*
    observed-scores file than the one they were aligned to (see module
    docstring). Returns a warning string if the check could not be
    performed (no manifest / no size recorded), or raises SystemExit if a
    mismatch is detected. Returns None if the check passed.
    """
    if manifest is None:
        return ("no run_manifest.json found next to the null replicates; "
                "cannot verify --observed-scores is the same file the null "
                "was aligned to.")
    recorded = (manifest.get("inputs") or {}).get("observed_scores")
    if not recorded or not recorded.get("exists"):
        return "run_manifest.json has no recorded observed-scores file info to check against."
    current = file_info(observed_scores)
    if not current["exists"]:
        raise SystemExit(f"--observed-scores {observed_scores} does not exist.")
    if current["size_bytes"] != recorded["size_bytes"]:
        raise SystemExit(
            f"--observed-scores {observed_scores} is {current['size_bytes']} bytes, but the "
            f"null-calibration run recorded {recorded['size_bytes']} bytes for "
            f"{recorded['path']} (mtime {recorded['mtime']}). Every null replicate was "
            "aligned to that file's row order at score time; if the observed score table "
            "has since been regenerated (e.g. a node-set change), this null is no longer "
            "valid for it and the replicate loop must be rerun against the new table."
        )
    return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def self_calibration_diagnostic(
    null_left: np.ndarray,
    null_right: np.ndarray,
    t: Optional[float],
    holdout_fraction: float,
    seed: int,
) -> dict:
    """Null-on-null self-calibration: hold out a random ~holdout_fraction of
    replicates, build the null from the rest, treat each held-out replicate
    as if it were "observed" (it has no real signal by construction) and
    check that applying the already-chosen threshold t yields a realized
    FDP-hat around 1 -- i.e. that essentially every "discovery" among the
    held-out replicate's own splits is, correctly, a false one.
    """
    R = null_left.shape[0]
    if t is None:
        return {"skipped": "no threshold available (criterion_met=False); "
                            "self-calibration requires a chosen t."}
    if R < 2:
        return {"skipped": f"only {R} replicate(s) available; need at least 2 "
                            "(one to hold out, one to build the null from)."}

    rng = np.random.default_rng(seed)
    n_heldout = max(1, round(R * holdout_fraction))
    n_heldout = min(n_heldout, R - 1)  # always leave >=1 replicate to build the null from
    heldout_idx = rng.choice(R, size=n_heldout, replace=False)
    build_mask = np.ones(R, dtype=bool)
    build_mask[heldout_idx] = False

    build_stat = _split_statistic(null_left[build_mask], null_right[build_mask])
    per_replicate = []
    for h in heldout_idx:
        heldout_stat = _split_statistic(null_left[h], null_right[h])
        valid = np.isfinite(heldout_stat)
        O_t = int(np.sum(heldout_stat[valid] >= t))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            V_per_build_rep = np.nansum(build_stat >= t, axis=1)
        E_fp = float(np.mean(V_per_build_rep))
        fdp_hat = E_fp / max(O_t, 1)
        per_replicate.append({
            "held_out_replicate_index": int(h),
            "O_t": O_t,
            "E_fp_from_build_set": round(E_fp, 3),
            "fdp_hat": round(fdp_hat, 3),
        })

    fdp_hats = [r["fdp_hat"] for r in per_replicate]
    O_ts = [r["O_t"] for r in per_replicate]
    return {
        "n_build_replicates": int(build_mask.sum()),
        "n_heldout_replicates": int(n_heldout),
        "threshold_used": t,
        "per_heldout_replicate": per_replicate,
        "mean_O_t": round(float(np.mean(O_ts)), 3),
        "mean_fdp_hat": round(float(np.mean(fdp_hats)), 3),
        "median_fdp_hat": round(float(np.median(fdp_hats)), 3),
        "interpretation": "fdp_hat should be close to 1: a held-out null "
                          "replicate has no real signal, so essentially every "
                          "one of its own discoveries at t is, correctly, a "
                          "false one.",
    }


def load_site_rates(path: Optional[Path]) -> Optional[pd.Series]:
    """Load a per-position site-rate covariate (e.g. IQ-TREE's --rate .rate
    file, or any whitespace/comma-separated file with a Site/Position column
    and a Rate column). Returns a pd.Series indexed by 1-based position, or
    None if no file was given -- callers must handle None by skipping the
    site-rate diagnostic honestly rather than fabricating a value.
    """
    if path is None:
        return None
    df = pd.read_csv(path, sep=None, engine="python", comment="#")
    cols = {c.lower(): c for c in df.columns}
    site_col = cols.get("site") or cols.get("position")
    rate_col = cols.get("rate")
    if site_col is None or rate_col is None:
        raise ValueError(
            f"--site-rate-file {path} must have a Site/Position column and a "
            f"Rate column; found columns {list(df.columns)}"
        )
    return pd.Series(df[rate_col].to_numpy(dtype=float), index=df[site_col].to_numpy(dtype=int))


def compute_flatness_diagnostics(
    null_stat: np.ndarray,
    keys: pd.DataFrame,
    t: Optional[float],
    site_rates: Optional[pd.Series],
    n_bins_decile: int,
    n_bins_quartile: int,
) -> Tuple[pd.DataFrame, dict]:
    """Realized null exceedance rate at t, across every covariate the plan
    calls for: clade-size decile, clade-depth quartile, site-rate quartile
    (if available), position (exact, one bin per position) and event type.
    """
    if t is None:
        return (
            pd.DataFrame(columns=["covariate", "bin_label", "mean_exceedance_rate", "n_tests", "threshold"]),
            {"skipped": "no threshold available (criterion_met=False)"},
        )

    frames = []

    clade_size = pd.to_numeric(keys["clade_size_total"], errors="coerce").clip(lower=1)
    frames.append(exceedance_flatness(
        null_stat, pd.DataFrame({"clade_size_total_decile": np.log(clade_size)}),
        threshold=t, n_bins=n_bins_decile,
    ))

    frames.append(exceedance_flatness(
        null_stat, pd.DataFrame({"distance_from_root_quartile": keys["distance_from_root"]}),
        threshold=t, n_bins=n_bins_quartile,
    ))

    site_rate_status = "computed"
    if site_rates is not None:
        mapped = keys["position"].map(site_rates)
        if mapped.isna().all():
            site_rate_status = "skipped: --site-rate-file had no entries matching any observed position"
        else:
            frames.append(exceedance_flatness(
                null_stat, pd.DataFrame({"site_rate_quartile": mapped}),
                threshold=t, n_bins=n_bins_quartile,
            ))
    else:
        site_rate_status = "skipped: no --site-rate-file given"

    n_positions = keys["position"].nunique()
    frames.append(exceedance_flatness(
        null_stat, pd.DataFrame({"position": keys["position"].astype(str)}),
        threshold=t, max_categorical=n_positions + 1,
    ))

    frames.append(exceedance_flatness(
        null_stat, pd.DataFrame({"event_type": keys["event_type"]}),
        threshold=t,
    ))

    combined = pd.concat(frames, ignore_index=True)
    return combined, {"site_rate_diagnostic": site_rate_status}


def ac_plus1_control(
    obs_stat: np.ndarray,
    null_stat: np.ndarray,
    ac: np.ndarray,
    t: Optional[float],
    max_null_sample: int = 200_000,
) -> dict:
    """AC=+1 rows can contain no genuine switch (AC=+1 means the ancestral
    reconstruction did not even change state, so there is nothing for a
    "switch" to be about); any gap between the observed and null score
    distribution restricted to these rows is therefore measured
    misspecification, not signal.
    """
    mask = ac == 1
    n_rows = int(mask.sum())
    if n_rows == 0:
        return {"skipped": "no AC=+1 rows in the observed table."}

    obs_sub = obs_stat[mask]
    obs_finite = obs_sub[np.isfinite(obs_sub)]
    null_sub = null_stat[:, mask]
    null_finite = null_sub[np.isfinite(null_sub)]

    rng = np.random.default_rng(0)
    if null_finite.size > max_null_sample:
        null_finite = rng.choice(null_finite, size=max_null_sample, replace=False)

    result = {
        "n_rows": n_rows,
        "observed": {
            "mean": float(np.mean(obs_finite)) if obs_finite.size else None,
            "std": float(np.std(obs_finite)) if obs_finite.size else None,
            "quantiles": {q: float(np.quantile(obs_finite, q)) for q in (0.5, 0.9, 0.99)} if obs_finite.size else None,
        },
        "null_pooled": {
            "mean": float(np.mean(null_finite)) if null_finite.size else None,
            "std": float(np.std(null_finite)) if null_finite.size else None,
            "quantiles": {q: float(np.quantile(null_finite, q)) for q in (0.5, 0.9, 0.99)} if null_finite.size else None,
        },
    }
    if obs_finite.size and null_finite.size:
        ks = scipy_stats.ks_2samp(obs_finite, null_finite)
        result["ks_test"] = {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)}
    if t is not None and obs_finite.size:
        result["observed_exceedance_rate_at_t"] = float(np.mean(obs_finite >= t))
        result["null_exceedance_rate_at_t"] = float(np.mean(null_finite >= t)) if null_finite.size else None
    return result


def p_ac_minus1_comparison(ac: np.ndarray, null_ac: np.ndarray) -> dict:
    """Observed vs null P(AC = -1), the fraction of tests where the
    ancestral reconstruction reports a state change at all (a prerequisite
    for a "switch" in either direction). A large gap here indicates the
    null's ASR reconstructs changes at a different rate than the real data,
    independent of any true switching signal.
    """
    p_obs = float(np.mean(ac == -1))
    per_rep = np.mean(null_ac == -1, axis=1)
    return {
        "p_observed": p_obs,
        "p_null_mean": float(np.mean(per_rep)),
        "p_null_std": float(np.std(per_rep)),
        "p_null_pooled": float(np.mean(null_ac == -1)),
        "difference_observed_minus_null_mean": p_obs - float(np.mean(per_rep)),
    }


def head_to_head_vs_percentile_rule(
    keys: pd.DataFrame,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    t: Optional[float],
    percentile: float,
    n_bins: int,
) -> dict:
    """Compares the calibrated global threshold t against this repo's
    existing per-(event, clade-size-decile) percentile rule (reusing
    scripts/plot_decoupled_event_switches_clade_adjusted.py's own
    bin_clade_sizes / calculate_bin_thresholds / identify_switches, not a
    reimplementation of it), on the SAME observed score table.

    Scope note: unlike that script's own CLI, this comparison does not
    reapply its occupancy filter (which needs the real alignment file and
    is orthogonal to null calibration) and does not re-filter by clade size
    (the observed table was already produced with a --min-clade cutoff at
    scoring time). It is a like-for-like comparison of the two *switch-
    calling rules* on the same test set, not a reproduction of that
    script's own figures.
    """
    if t is None:
        return {"skipped": "no threshold available (criterion_met=False); "
                            "cannot compare a rule against a threshold that "
                            "does not exist."}

    df = keys.copy()
    df["badasp_score_left"] = obs_left
    df["badasp_score_right"] = obs_right

    df_left = df[["event_type", "clade_size_left", "badasp_score_left"]].rename(
        columns={"clade_size_left": "clade_size", "badasp_score_left": "score"}
    )
    df_right = df[["event_type", "clade_size_right", "badasp_score_right"]].rename(
        columns={"clade_size_right": "clade_size", "badasp_score_right": "score"}
    )
    melted = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["score", "clade_size"])
    melted["clade_bin"], bin_categories = bin_clade_sizes(melted["clade_size"], "quantile", n_bins)

    def _map_to_bin(val):
        for interval in bin_categories:
            if val in interval:
                return interval
        return bin_categories[-1] if val > bin_categories[-1].right else bin_categories[0]

    df["bin_left"] = df["clade_size_left"].apply(_map_to_bin)
    df["bin_right"] = df["clade_size_right"].apply(_map_to_bin)

    percentile_thresholds = calculate_bin_thresholds(melted, "score", "clade_bin", percentile=percentile)
    new_thresholds = as_bin_threshold_dict(t, df["event_type"].unique(), bin_categories)

    is_switch_percentile = identify_switches(df, percentile_thresholds, event_specific=False)
    is_switch_new = identify_switches(df, new_thresholds, event_specific=False)

    confusion = pd.crosstab(
        pd.Series(is_switch_percentile, name=f"percentile_{percentile}_rule"),
        pd.Series(is_switch_new, name="calibrated_global_threshold"),
    )
    return {
        "percentile_rule": {"percentile": percentile, "num_bins": n_bins,
                             "n_switches": int(is_switch_percentile.sum())},
        "calibrated_rule": {"threshold": t, "n_switches": int(is_switch_new.sum())},
        "n_comparisons": int(len(df)),
        "confusion_matrix": confusion.to_dict(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_threshold_sweep(sweep: pd.DataFrame, chosen_t: Optional[float], out_path: Path) -> None:
    if sweep.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(sweep["t"], sweep["O"], label="O(t) observed discoveries", color="#2c3e50")
    axes[0].plot(sweep["t"], sweep["E_fp"], label="E[FP](t)", color="#c0392b")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[1].plot(sweep["t"], sweep["fdp_quantile_achieved"], color="#8e44ad")
    axes[1].set_ylabel("achieved FDP quantile")
    axes[1].set_xlabel("candidate threshold t")
    if chosen_t is not None:
        for ax in axes:
            ax.axvline(chosen_t, color="black", linestyle="--", linewidth=1, label="chosen t")
    fig.suptitle("Threshold sweep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_flatness(flatness_df: pd.DataFrame, out_path: Path) -> None:
    if flatness_df.empty:
        return
    covariates = flatness_df["covariate"].unique()
    fig, axes = plt.subplots(len(covariates), 1, figsize=(8, 2.6 * len(covariates)))
    if len(covariates) == 1:
        axes = [axes]
    for ax, cov in zip(axes, covariates):
        sub = flatness_df[flatness_df["covariate"] == cov]
        labels = sub["bin_label"].astype(str)
        ax.bar(range(len(sub)), sub["mean_exceedance_rate"], color="#2980b9")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(cov, fontsize=9)
        ax.set_ylabel("null exceedance rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ac_plus1_control(obs_stat: np.ndarray, null_stat: np.ndarray, ac: np.ndarray, out_path: Path) -> None:
    mask = ac == 1
    if not mask.any():
        return
    obs_sub = obs_stat[mask]
    obs_sub = obs_sub[np.isfinite(obs_sub)]
    null_sub = null_stat[:, mask]
    null_sub = null_sub[np.isfinite(null_sub)]
    if obs_sub.size == 0 or null_sub.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_sub, bins=60, density=True, alpha=0.5, label="null (AC=+1 rows)", color="#7f8c8d")
    ax.hist(obs_sub, bins=60, density=True, alpha=0.5, label="observed (AC=+1 rows)", color="#e67e22")
    ax.set_xlabel("split statistic")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("AC=+1 control: observed vs null score distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_config(pre_args.config)
    nc = cfg.get("null_calibration", {})

    parser = argparse.ArgumentParser(
        description="Aggregate null-calibration replicates into the calibrated "
                    "switch threshold, its diagnostics, and a per-test table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--null-run-dir", type=Path, required=True,
                        help="run_null_calibration.py's --out-dir (expects "
                             "<dir>/npz/rep_*.npz and, if present, "
                             "<dir>/run_manifest.json).")
    parser.add_argument("--observed-scores", type=Path, default=None,
                        help="Observed score CSV. Defaults to the path recorded "
                             "in <null-run-dir>/run_manifest.json; if given "
                             "explicitly, still checked against that manifest "
                             "(see module docstring on alignment safety).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-fdr", type=float, default=nc.get("target_fdr", 0.05))
    parser.add_argument("--fwer-alpha", type=float, default=nc.get("fwer_alpha", 0.05))
    parser.add_argument("--fdp-quantile", type=float, default=nc.get("fdp_quantile", 0.90))
    parser.add_argument("--holdout-fraction", type=float, default=0.1,
                        help="Fraction of replicates held out for the "
                             "null-on-null self-calibration diagnostic.")
    parser.add_argument("--holdout-seed", type=int, default=None,
                        help="Seed for the self-calibration holdout split "
                             "(default: the run's own seed from run_manifest.json).")
    parser.add_argument("--site-rate-file", type=Path, default=None,
                        help="Optional per-position site-rate file (e.g. IQ-TREE's "
                             "--rate output) for the site-rate-quartile exceedance "
                             "flatness diagnostic. Skipped honestly (not fabricated) "
                             "if omitted.")
    parser.add_argument("--percentile-comparison", type=float, default=99.9,
                        help="Percentile for the head-to-head comparison against "
                             "the existing per-(event, clade-size-decile) rule.")
    parser.add_argument("--num-bins-decile", type=int, default=10)
    parser.add_argument("--num-bins-quartile", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    npz_dir = args.null_run_dir / "npz"
    if not npz_dir.exists():
        npz_dir = args.null_run_dir  # allow pointing directly at the npz dir

    manifest_path = args.null_run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    observed_scores = args.observed_scores
    if observed_scores is None:
        if manifest is None:
            raise SystemExit(
                f"--observed-scores not given and no run_manifest.json found at "
                f"{manifest_path}; cannot determine which observed table the "
                "null replicates were aligned to."
            )
        observed_scores = Path(manifest["inputs"]["observed_scores"]["path"])

    consistency_note = check_observed_scores_consistency(observed_scores, manifest)
    if consistency_note:
        print(f"WARNING: {consistency_note}", file=sys.stderr)

    print(f"Loading observed scores from {observed_scores}...")
    obs_left, obs_right, keys = load_observed_scores(observed_scores)
    n_tests = len(obs_left)
    print(f"  {n_tests:,} tests")

    print(f"Loading null replicates from {npz_dir}...")
    null_left, null_right, null_ac, used_files = load_null_replicates(npz_dir, n_tests)
    R = null_left.shape[0]
    print(f"  {R} replicate(s) loaded")

    seed = args.holdout_seed
    if seed is None:
        seed = (manifest or {}).get("seed", 0)

    flavour = (manifest or {}).get("flavour") or load_config(args.config).get("null_calibration", {}).get("flavour")
    shrinkage = (manifest or {}).get("shrinkage")

    # --- Primary calibration ---------------------------------------------
    result: ThresholdResult = threshold_at_fdr(
        obs_left, obs_right, null_left, null_right, q=args.target_fdr, fdp_quantile=args.fdp_quantile
    )
    fwer = maxT_fwer_thresholds(null_left, null_right, keys["position"].to_numpy(), alpha=args.fwer_alpha)

    thresholds_json = {
        "flavour": flavour,
        "R": R,
        "seed": seed,
        "shrinkage": shrinkage,
        "target_fdr": args.target_fdr,
        "fwer_alpha": args.fwer_alpha,
        "fdp_quantile": args.fdp_quantile,
        "t": result.t,
        "O_t": result.O_t,
        "E_fp": result.E_fp,
        "E_fp_over_O": result.E_fp_over_O,
        "fdp_quantile_achieved": result.fdp_quantile_achieved,
        "criterion_met": result.criterion_met,
        "zero_discoveries": result.zero_discoveries,
        "note": result.note,
        "fwer_per_position_threshold": {str(k): v for k, v in fwer["per_position"].items()},
        "fwer_global_threshold": fwer["global"],
        "git_sha": git_sha(REPO_ROOT),
        "generated": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "observed_scores": file_info(observed_scores),
            "null_run_dir": str(args.null_run_dir),
            "null_replicate_files": [str(f) for f in used_files],
            "run_manifest": file_info(manifest_path) if manifest_path.exists() else None,
            "consistency_check_note": consistency_note,
        },
    }
    (args.out_dir / "thresholds.json").write_text(
        json.dumps(thresholds_json, indent=2, default=_json_default) + "\n"
    )
    print(f"Wrote {args.out_dir / 'thresholds.json'} (t={result.t}, criterion_met={result.criterion_met})")

    if not result.sweep.empty:
        result.sweep.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
        plot_threshold_sweep(result.sweep, result.t, plots_dir / "threshold_sweep.png")

    # --- Per-test table -----------------------------------------------------
    obs_stat = _split_statistic(obs_left, obs_right)
    null_stat = _split_statistic(null_left, null_right)
    p_values, q_values = per_test_pvalues(obs_stat, null_stat)
    per_test = keys.copy()
    per_test["badasp_score_left"] = obs_left
    per_test["badasp_score_right"] = obs_right
    per_test["split_statistic"] = obs_stat
    per_test["p_value"] = p_values
    per_test["q_value"] = q_values
    per_test.to_csv(args.out_dir / "per_test_pvalues.csv", index=False)
    print(f"Wrote {args.out_dir / 'per_test_pvalues.csv'} ({len(per_test):,} rows)")

    # --- Diagnostics ----------------------------------------------------------
    diagnostics: Dict[str, object] = {}

    diagnostics["self_calibration"] = self_calibration_diagnostic(
        null_left, null_right, result.t, args.holdout_fraction, seed
    )

    site_rates = load_site_rates(args.site_rate_file)
    flatness_df, flatness_note = compute_flatness_diagnostics(
        null_stat, keys, result.t, site_rates, args.num_bins_decile, args.num_bins_quartile
    )
    if not flatness_df.empty:
        flatness_df.to_csv(args.out_dir / "exceedance_flatness.csv", index=False)
        plot_flatness(flatness_df, plots_dir / "exceedance_flatness.png")
    diagnostics["exceedance_flatness"] = {**flatness_note, "n_rows": int(len(flatness_df))}

    diagnostics["ac_plus1_control"] = ac_plus1_control(obs_stat, null_stat, keys["ac"].to_numpy(), result.t)
    if result.t is not None:
        plot_ac_plus1_control(obs_stat, null_stat, keys["ac"].to_numpy(), plots_dir / "ac_plus1_control.png")

    diagnostics["p_ac_minus1"] = p_ac_minus1_comparison(keys["ac"].to_numpy(), null_ac)

    diagnostics["head_to_head_vs_percentile_rule"] = head_to_head_vs_percentile_rule(
        keys, obs_left, obs_right, result.t, args.percentile_comparison, args.num_bins_decile
    )

    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=_json_default) + "\n"
    )
    print(f"Wrote {args.out_dir / 'diagnostics.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
