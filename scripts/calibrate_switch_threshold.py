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
from typing import Dict, List, Optional, Sequence, Tuple

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

# Pre-registered reporting-curve anchor call counts for --error-profile-calls's
# default. These are used only when --operating-point is given and
# --error-profile-calls is not; they were picked in advance as reporting
# points of interest, not chosen after looking at what any particular null
# run produces.
DEFAULT_ERROR_PROFILE_CALLS = (51, 81, 240, 500, 795, 1500, 2023)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_null_replicate import SCORE_COLUMNS  # noqa: E402
from src.badasp.null_model import (  # noqa: E402
    ThresholdResult,
    as_bin_threshold_dict,
    describe_threshold,
    describe_threshold_curve,
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
from src.badasp.robust_conditioning import (  # noqa: E402
    ANCHOR_ALPHA_DEFAULT,
    N_CLADE_BINS_DEFAULT,
    N_HOT_DEFAULT,
    CellConditionalModel,
    fit_cell_model,
    t_from_z,
    to_z,
)
from src.badasp.tail_model import EmpiricalTail, fit_pooled_tail  # noqa: E402

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


def _rep_index(path: Path) -> Optional[int]:
    """Extract the integer index from a ``rep_<digits>.npz`` filename, or
    None if the name doesn't follow that convention. String parsing only
    (no regex import) since this repo's rep_NNNN naming is fixed and simple.
    """
    stem = path.stem
    if not stem.startswith("rep_"):
        return None
    suffix = stem[len("rep_"):]
    return int(suffix) if suffix.isdigit() else None


def infer_replicate_groups(used_files: List[Path], null_dir: Path) -> Tuple[Optional[np.ndarray], str]:
    """Resolve each loaded replicate's simulate-invocation label, i.e. which
    `simulate()` call produced it, WITHOUT ever guessing from the replicate's
    position in `used_files`.

    `null_dir` is the directory `used_files` were loaded from (the ``npz``
    directory `load_null_replicates` was pointed at); its parent is treated
    as "the run directory" for locating provenance files, mirroring how
    `load_null_replicates`/`main` already resolve `run_manifest.json` next to
    the ``npz`` directory.

    Resolution order, first route that produces a label for every file in
    `used_files` wins:

    1. ``<null_dir>/../run_manifest.json`` written by `merge_null_runs.py`:
       has a ``replicates`` list of ``{"merged_index": i, "source": path}``
       dicts. Each `used_files` entry is matched to its `source` path by its
       ``rep_NNNN`` index, the sources are grouped by the directory they came
       from, and THIS SAME FUNCTION is called again on each such directory
       (recursively) with that directory's own subset of source files, so a
       merged set inherits whatever real per-invocation labels its
       constituent runs have. If a constituent run itself has no resolvable
       labels, every replicate merged from it is labelled with that run's own
       directory name -- still real, recorded provenance (the `source` path
       already on file), not an invented index-based split.
    2. ``<null_dir>/../provenance.json`` written by
       `stage_null_replicate_set.py`: a ``replicates`` list of
       ``{"rep_id": "rep_0004", "invocation": "9", ...}`` dicts, matched to
       `used_files` by `rep_id` (== filename stem).
    3. ``<null_dir>/../logs/progress.jsonl``: one JSON object per line with
       ``{"replicate": i, "batch": b, ...}`` fields, matched to `used_files`
       by the `rep_NNNN` index.
    4. Otherwise `(None, "<reason>")`. Deliberately NOT attempted: any
       arithmetic on the replicate index (e.g. a `chunk = index // 10` rule)
       -- that convention lives only inside an external sbatch script (see
       `euler_run`'s generation) and is not recorded anywhere this function
       can read, so inferring it here would fabricate provenance rather than
       recover it.

    Returns
    -------
    (groups, route) where `groups` is an object-dtype array of length
    `len(used_files)` (or None if unresolved) and `route` is a human-readable
    string naming which route was used (or, if unresolved, why).
    """
    null_dir = Path(null_dir)
    run_dir = null_dir.parent

    # --- Route 1: merge_null_runs.py's run_manifest.json --------------------
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = None
        replicates = manifest.get("replicates") if isinstance(manifest, dict) else None
        is_merge_manifest = (
            isinstance(replicates, list) and len(replicates) > 0
            and all(isinstance(e, dict) and "source" in e and "merged_index" in e for e in replicates)
        )
        if is_merge_manifest:
            by_index = {int(e["merged_index"]): e["source"] for e in replicates}
            indices = [_rep_index(f) for f in used_files]
            if all(i is not None and i in by_index for i in indices):
                labels = np.empty(len(used_files), dtype=object)
                sources_by_dir: Dict[Path, List[Tuple[int, Path]]] = {}
                for pos, i in enumerate(indices):
                    src_path = Path(by_index[i])
                    if not src_path.is_absolute():
                        src_path = REPO_ROOT / src_path
                    sources_by_dir.setdefault(src_path.parent, []).append((pos, src_path))

                for src_npz_dir, entries in sources_by_dir.items():
                    positions = [p for p, _ in entries]
                    sub_files = [f for _, f in entries]
                    sub_labels, _sub_route = infer_replicate_groups(sub_files, src_npz_dir)
                    run_name = src_npz_dir.parent.name
                    if sub_labels is None:
                        for pos in positions:
                            labels[pos] = run_name
                    else:
                        for pos, lbl in zip(positions, sub_labels):
                            labels[pos] = f"{run_name}:{lbl}"
                return labels, (
                    f"run_manifest.json at {manifest_path} (merge_null_runs.py's "
                    "replicates[].source), resolved per originating source "
                    "directory, recursing into each source's own provenance "
                    "where available"
                )

    # --- Route 2: stage_null_replicate_set.py's provenance.json -------------
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        provenance = None
        try:
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError):
            provenance = None
        replicates = provenance.get("replicates") if isinstance(provenance, dict) else None
        is_stage_provenance = (
            isinstance(replicates, list) and len(replicates) > 0
            and all(isinstance(e, dict) and "rep_id" in e and "invocation" in e for e in replicates)
        )
        if is_stage_provenance:
            by_rep_id = {e["rep_id"]: e["invocation"] for e in replicates}
            stems = [f.stem for f in used_files]
            if all(s in by_rep_id for s in stems):
                return (
                    np.array([by_rep_id[s] for s in stems], dtype=object),
                    f"provenance.json at {provenance_path} "
                    "(stage_null_replicate_set.py's replicates[].invocation)",
                )

    # --- Route 3: logs/progress.jsonl ---------------------------------------
    progress_path = run_dir / "logs" / "progress.jsonl"
    if progress_path.exists():
        by_replicate: Dict[int, object] = {}
        try:
            for line in progress_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict) and "replicate" in rec and "batch" in rec:
                    by_replicate[int(rec["replicate"])] = rec["batch"]
        except (OSError, json.JSONDecodeError):
            by_replicate = {}
        if by_replicate:
            indices = [_rep_index(f) for f in used_files]
            if all(i is not None and i in by_replicate for i in indices):
                return (
                    np.array([by_replicate[i] for i in indices], dtype=object),
                    f"logs/progress.jsonl at {progress_path} "
                    "({'replicate','batch'} records)",
                )

    return None, (
        f"no usable run_manifest.json (replicates[].source) at {manifest_path}, "
        f"no provenance.json (replicates[].rep_id/.invocation) at "
        f"{provenance_path}, and no logs/progress.jsonl (replicate/batch "
        f"records) at {progress_path}; replicate provenance cannot be "
        "inferred (arithmetic on the replicate index is deliberately not "
        "attempted -- see infer_replicate_groups docstring)."
    )


def load_replicate_groups_csv(path: Path, used_files: List[Path]) -> Tuple[np.ndarray, str]:
    """Explicit override for `infer_replicate_groups`: a two-column CSV with
    headers `rep_file,invocation`, matched to `used_files` by filename (not
    by row order), so the CSV need not list files in the same order they
    were loaded. Every loaded replicate must have a matching row.
    """
    table = pd.read_csv(path)
    if not {"rep_file", "invocation"}.issubset(table.columns):
        raise SystemExit(
            f"--replicate-groups-csv {path} must have columns 'rep_file' and "
            f"'invocation'; found {list(table.columns)}."
        )
    by_name = {Path(str(row["rep_file"])).name: row["invocation"] for _, row in table.iterrows()}
    missing = [f.name for f in used_files if f.name not in by_name]
    if missing:
        raise SystemExit(
            f"--replicate-groups-csv {path} has no row for {len(missing)} loaded "
            f"replicate(s) (e.g. {missing[:5]}); every loaded replicate must be "
            "covered by the CSV."
        )
    groups = np.array([by_name[f.name] for f in used_files], dtype=object)
    return groups, f"--replicate-groups-csv {path} (explicit override)"


def parse_operating_point_spec(raw: str) -> Tuple[str, float]:
    """Parse the --operating-point flag: 'calls:<int>' or 't:<float>'.

    Never selects anything itself -- this only validates and unpacks the
    string the caller already chose; resolving 'calls:<int>' into an actual
    threshold value happens in `main`, where the observed statistic is
    available.
    """
    if ":" not in raw:
        raise SystemExit(
            f"--operating-point {raw!r} not understood; accepted forms are "
            "'calls:<int>' (e.g. 'calls:795') or 't:<float>' (e.g. 't:1.5')."
        )
    kind, _, value_str = raw.partition(":")
    if kind == "calls":
        try:
            return "calls", float(int(value_str))
        except ValueError:
            raise SystemExit(
                f"--operating-point {raw!r} not understood; 'calls:<int>' "
                "needs an integer call count. Accepted forms are "
                "'calls:<int>' (e.g. 'calls:795') or 't:<float>' (e.g. 't:1.5')."
            )
    if kind == "t":
        try:
            return "t", float(value_str)
        except ValueError:
            raise SystemExit(
                f"--operating-point {raw!r} not understood; 't:<float>' needs "
                "a numeric threshold. Accepted forms are 'calls:<int>' (e.g. "
                "'calls:795') or 't:<float>' (e.g. 't:1.5')."
            )
    raise SystemExit(
        f"--operating-point {raw!r} not understood; accepted forms are "
        "'calls:<int>' (e.g. 'calls:795') or 't:<float>' (e.g. 't:1.5')."
    )


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


# ---------------------------------------------------------------------------
# Cell-conditional thresholds
# ---------------------------------------------------------------------------


def fit_conditioning(
    null_left: np.ndarray,
    null_right: np.ndarray,
    keys: pd.DataFrame,
    n_hot: int,
    n_clade_bins: int,
    anchor_alpha: float,
    replicates: Optional[np.ndarray] = None,
) -> Tuple[EmpiricalTail, CellConditionalModel]:
    """Fit the pooled tail and the cell multipliers on null replicates only.

    `replicates` restricts the fit to a subset (used by the CV diagnostic);
    both the tail AND the cell model must be re-fit per fold, since the
    choice of which positions are "hot" is itself estimated from the null.
    """
    sub = slice(None) if replicates is None else np.asarray(replicates)
    nl, nr = null_left[sub], null_right[sub]
    tail = fit_pooled_tail(nl, nr)
    model = fit_cell_model(
        nl, nr,
        keys["position"].to_numpy(),
        keys["clade_size_left"].to_numpy(float),
        keys["clade_size_right"].to_numpy(float),
        tail,
        n_hot=n_hot,
        n_clade_bins=n_clade_bins,
        anchor_alpha=anchor_alpha,
    )
    return tail, model


def apply_conditioning(
    values_left: np.ndarray,
    values_right: np.ndarray,
    keys: pd.DataFrame,
    tail: EmpiricalTail,
    model: CellConditionalModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """Branch-level scores -> branch-level z. Works for 1-D observed arrays
    and 2-D (R, n_tests) null arrays alike."""
    position = keys["position"].to_numpy()
    return (
        to_z(values_left, position, keys["clade_size_left"].to_numpy(float), model, tail),
        to_z(values_right, position, keys["clade_size_right"].to_numpy(float), model, tail),
    )


def conditioned_cv_diagnostic(
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    null_left: np.ndarray,
    null_right: np.ndarray,
    keys: pd.DataFrame,
    q: float,
    fdp_quantile: float,
    n_hot: int,
    n_clade_bins: int,
    anchor_alpha: float,
    n_folds: int,
) -> dict:
    """Rotating replicate-level cross-validation of the conditioned rule.

    Folds are at the REPLICATE level because that is the unit of dependence
    -- every split at one position in one replicate shares a simulated
    column and one ASR reconstruction.

    Per fold, the tail, the cell model AND the threshold are all derived
    from the in-fold replicates alone, and the held-out replicates are used
    only to evaluate. Selecting the threshold on all replicates and only
    then holding some out (which `self_calibration_diagnostic` does) is
    tolerable for a one-parameter global rule but not for a model that
    estimates which positions are hot.

    `fdp_hat` here is the held-out expected false-discovery proportion, so
    it should come in at or below `q`; it is NOT the null-on-null
    `self_calibration_diagnostic` statistic, which targets 1.
    """
    R = null_left.shape[0]
    if R < 2 * n_folds:
        return {"skipped": f"{R} replicates is too few for {n_folds}-fold "
                            f"replicate-level CV (need >= {2 * n_folds})."}

    position = keys["position"].to_numpy()
    unique_positions = np.unique(position)
    position_index = np.searchsorted(unique_positions, position)

    folds = np.array_split(np.arange(R), n_folds)
    per_fold = []
    for k, held in enumerate(folds):
        in_fold = np.setdiff1d(np.arange(R), held)
        tail, model = fit_conditioning(
            null_left, null_right, keys, n_hot, n_clade_bins, anchor_alpha, in_fold
        )
        z_obs_l, z_obs_r = apply_conditioning(obs_left, obs_right, keys, tail, model)
        z_in_l, z_in_r = apply_conditioning(
            null_left[in_fold], null_right[in_fold], keys, tail, model
        )
        res = threshold_at_fdr(z_obs_l, z_obs_r, z_in_l, z_in_r, q=q, fdp_quantile=fdp_quantile)
        if not res.criterion_met:
            per_fold.append({"fold": k, "criterion_met": False})
            continue

        z_out_l, z_out_r = apply_conditioning(
            null_left[held], null_right[held], keys, tail, model
        )
        held_stat = _split_statistic(z_out_l, z_out_r)
        exceed = held_stat >= res.t
        V = exceed.sum(axis=1)
        O_t = int(np.sum(_split_statistic(z_obs_l, z_obs_r) >= res.t))
        per_position = np.bincount(
            position_index, weights=exceed.sum(axis=0), minlength=len(unique_positions)
        )
        total = per_position.sum()
        per_fold.append({
            "fold": k,
            "criterion_met": True,
            "n_in_fold": int(len(in_fold)),
            "n_held_out": int(len(held)),
            "t": round(float(res.t), 4),
            "O_t": O_t,
            "heldout_E_fp": round(float(V.mean()), 2),
            "heldout_fdp_hat": round(float(V.mean() / max(O_t, 1)), 4),
            "heldout_top10_position_share": round(
                float(np.sort(per_position)[::-1][:10].sum() / max(total, 1)), 4
            ),
        })

    ok = [f for f in per_fold if f.get("criterion_met")]
    summary = {
        "n_folds": n_folds,
        "target_q": q,
        "per_fold": per_fold,
        "n_folds_criterion_met": len(ok),
    }
    if ok:
        summary["mean_heldout_fdp_hat"] = round(
            float(np.mean([f["heldout_fdp_hat"] for f in ok])), 4
        )
        summary["mean_heldout_top10_position_share"] = round(
            float(np.mean([f["heldout_top10_position_share"] for f in ok])), 4
        )
    summary["interpretation"] = (
        "heldout_fdp_hat is the realised false-discovery proportion on "
        "replicates excluded from BOTH the fit and the threshold choice; it "
        "should be <= target_q. heldout_top10_position_share is the share of "
        "held-out null exceedances falling in the 10 noisiest positions "
        "(5.9% would be flat)."
    )
    return summary


def conditioned_bin_threshold_dict(
    keys: pd.DataFrame,
    z_threshold: float,
    tail: EmpiricalTail,
    model: CellConditionalModel,
    event_types: Sequence[str],
    num_bins: int,
) -> Tuple[Dict, dict]:
    """Project the conditioned rule onto the legacy
    ``{(event_type, clade_bin): raw threshold}`` dict.

    This is LOSSY and is provided only so the four downstream analysis
    scripts keep running unmodified: the real rule also varies by position,
    which this dict cannot express. `per_test_calls.csv` is the source of
    truth. The projection reports, for each clade-size bin, the raw-score
    threshold a position in the COLD group would have to clear -- the cold
    group because it holds the large majority of positions, so this is the
    threshold most tests in that bin actually face. Tests at hot positions
    face a strictly higher bar than this dict shows, which makes the
    projection ANTI-conservative for exactly those columns.
    """
    sizes = pd.Series(np.concatenate([
        keys["clade_size_left"].to_numpy(float), keys["clade_size_right"].to_numpy(float)
    ])).dropna()
    # Same construction the downstream consumers use
    # (calculate_bin_thresholds_999: qcut over the melted per-branch sizes).
    bin_labels = pd.qcut(sizes, q=num_bins, duplicates="drop")
    categories = bin_labels.cat.categories

    cold_positions = np.setdiff1d(np.unique(keys["position"].to_numpy()), model.hot_positions)
    if cold_positions.size == 0:  # degenerate: every position was called hot
        cold_positions = np.unique(keys["position"].to_numpy())
    representative_position = np.full(len(categories), cold_positions[0])

    # Median observed size within each bin, so the representative clade size
    # is one that actually occurs rather than an interval midpoint.
    representative_size = sizes.groupby(bin_labels, observed=False).median().to_numpy()

    raw = t_from_z(z_threshold, representative_position, representative_size, model, tail)
    groups = list(dict.fromkeys(list(event_types) + ["overall"]))
    mapping = {
        (group, interval): float(value)
        for group in groups
        for interval, value in zip(categories, raw)
    }
    provenance = {
        "lossy_projection": True,
        "source_of_truth": "per_test_calls.csv",
        "representative_position_group": "cold",
        "per_bin_threshold": {
            str(interval): round(float(value), 4)
            for interval, value in zip(categories, raw)
        },
    }
    return mapping, provenance


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
    null_ac: Optional[np.ndarray] = None,
) -> dict:
    """AC=+1 rows can contain no genuine switch (AC=+1 means the ancestral
    reconstruction did not even change state, so there is nothing for a
    "switch" to be about); any gap between the observed and null score
    distribution restricted to these rows is therefore measured
    misspecification, not signal.

    Each dataset must be restricted by *its own* AC. AC is an outcome of the
    reconstruction, not a fixed property of a comparison, so a test with AC=+1
    in the observed data often has AC=-1 in a given replicate. Masking the null
    with the observed AC therefore mixes in null rows whose ancestral states did
    change, and those score ``RC + p_AC`` rather than ``RC - p_AC``. That is
    detectable: it puts null "AC=+1" mass above 1.0, which is unreachable when
    AC=+1 (RC <= 1 and p_AC >= 0). ``null_ac`` is the (R, n_tests) matrix of the
    replicates' own AC values; without it this comparison is not meaningful.
    """
    mask = ac == 1
    n_rows = int(mask.sum())
    if n_rows == 0:
        return {"skipped": "no AC=+1 rows in the observed table."}

    obs_sub = obs_stat[mask]
    obs_finite = obs_sub[np.isfinite(obs_sub)]
    if null_ac is None:
        return {
            "skipped": "null AC matrix not supplied; refusing to mask the null "
                       "with the observed AC, which would compare different "
                       "score definitions.",
            "n_rows": n_rows,
        }
    null_mask = (null_ac == 1) & np.isfinite(null_stat)
    null_finite = null_stat[null_mask]

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
    calibrated_calls: Optional[np.ndarray] = None,
) -> dict:
    """Compares the calibrated rule against this repo's
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
    is_switch_percentile = identify_switches(df, percentile_thresholds, event_specific=False)

    if calibrated_calls is not None:
        # Under conditioning `t` lives on the z scale, so it cannot be fed
        # to identify_switches, which compares RAW scores. The call set is
        # passed in already evaluated instead.
        is_switch_new = np.asarray(calibrated_calls, dtype=bool)
        calibrated_desc = {"threshold_z": t, "n_switches": int(is_switch_new.sum()),
                           "scale": "z = -log10(A * G(s))"}
    else:
        new_thresholds = as_bin_threshold_dict(t, df["event_type"].unique(), bin_categories)
        is_switch_new = identify_switches(df, new_thresholds, event_specific=False)
        calibrated_desc = {"threshold": t, "n_switches": int(is_switch_new.sum()),
                           "scale": "raw BADASP score"}

    confusion = pd.crosstab(
        pd.Series(is_switch_percentile, name=f"percentile_{percentile}_rule"),
        pd.Series(is_switch_new, name="calibrated_rule"),
    )
    return {
        "percentile_rule": {"percentile": percentile, "num_bins": n_bins,
                             "n_switches": int(is_switch_percentile.sum())},
        "calibrated_rule": calibrated_desc,
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
    parser.add_argument(
        "--conditioning", choices=["none", "cell"], default="cell",
        help="'cell' standardises each branch score against its own "
             "(position group x clade-size bin) null rate before "
             "thresholding; 'none' reproduces the single global threshold. "
             "Both are always reported; this selects which one is primary.",
    )
    parser.add_argument("--n-hot", type=int, default=N_HOT_DEFAULT,
                        help="Positions assigned to the noisy group.")
    parser.add_argument("--n-clade-bins", type=int, default=N_CLADE_BINS_DEFAULT,
                        help="Clade-size bins per position group. Bins whose "
                             "quantile edges collapse onto the minimum clade "
                             "size are dropped, so the achieved count can be "
                             "lower than this.")
    parser.add_argument("--anchor-alpha", type=float, default=ANCHOR_ALPHA_DEFAULT,
                        help="Pooled tail probability defining the anchor u0 "
                             "at which cell rates are estimated.")
    parser.add_argument("--cv-folds", type=int, default=4,
                        help="Replicate-level folds for the conditioned rule's "
                             "cross-validation.")
    parser.add_argument(
        "--operating-point", type=str, default=None,
        help="Describe (not select) the error rate of an externally chosen "
             "threshold, reported alongside -- never instead of -- the "
             "FDP-exceedance criterion above. Accepted forms: 'calls:<int>' "
             "(t is set to the quantile of the primary observed split "
             "statistic giving that many calls) or 't:<float>' (t given "
             "directly, on whichever scale --conditioning produces: raw score "
             "for 'none', z for 'cell'). Off by default; when given, adds an "
             "'error_profile' block to thresholds.json and an "
             "error_profile_curve.csv, and unblocks the diagnostics that "
             "otherwise skip when the FDP criterion finds no threshold.",
    )
    parser.add_argument(
        "--error-profile-calls", type=str, default=None,
        help="Comma-separated call counts for the describe_threshold_curve "
             "report written when --operating-point is given. These are "
             "reporting anchors declared in advance of inspecting this run's "
             "results, not call counts chosen after looking at them. Default "
             f"when --operating-point is given: {','.join(str(c) for c in DEFAULT_ERROR_PROFILE_CALLS)}.",
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=20000,
        help="Nonparametric bootstrap resamples for the --operating-point "
             "error-rate description (null_model.describe_threshold's "
             "n_bootstrap). Used only when --operating-point is given.",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=20260819,
        help="Seed for the --operating-point bootstrap, so the reported "
             "interval is reproducible from this recorded value rather than "
             "unlogged global RNG state. Used only when --operating-point is "
             "given.",
    )
    parser.add_argument(
        "--elevated-pac-threshold", type=float, default=0.17,
        help="P(AC=-1) value above which a null replicate is labelled "
             "'elevated' for describe_threshold's two-state-mixture report "
             "(used only when --operating-point is given). This cut is "
             "purely descriptive, not a validated boundary: its own "
             "falsifier checks (null_model.describe_threshold's "
             "mixture.falsifiers, computed on the per-replicate P(AC=-1) "
             "values themselves, not on this cut's effect on V) are "
             "reported right next to it -- those checks are what tell the "
             "reader whether a two-state description of the null "
             "replicates is actually supported by the data at hand, and no "
             "particular outcome should be assumed without reading them.",
    )
    parser.add_argument(
        "--replicate-groups-csv", type=Path, default=None,
        help="Optional two-column CSV (headers 'rep_file,invocation') giving "
             "each loaded replicate's simulate-invocation label explicitly, "
             "overriding automatic resolution via infer_replicate_groups. "
             "Used only when --operating-point is given.",
    )
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

    # --- Conditioning (fitted on the null only) ----------------------------
    # Both rules are always computed so the comparison is always available;
    # --conditioning only decides which one is primary.
    tail = model = None
    stat_left, stat_right = obs_left, obs_right
    null_stat_left, null_stat_right = null_left, null_right
    conditioning_info: Dict[str, object] = {"mode": args.conditioning}

    if args.conditioning == "cell":
        print(f"Fitting cell conditioning (n_hot={args.n_hot}, "
              f"n_clade_bins={args.n_clade_bins}, anchor_alpha={args.anchor_alpha})...")
        tail, model = fit_conditioning(
            null_left, null_right, keys, args.n_hot, args.n_clade_bins, args.anchor_alpha
        )
        stat_left, stat_right = apply_conditioning(obs_left, obs_right, keys, tail, model)
        null_stat_left, null_stat_right = apply_conditioning(
            null_left, null_right, keys, tail, model
        )
        conditioning_info.update({
            "anchor_alpha": model.anchor_alpha,
            "anchor_score": round(float(model.anchor_score), 4),
            "n_hot_positions": int(len(model.hot_positions)),
            "hot_positions": model.hot_positions.tolist(),
            "n_clade_bins_achieved": int(model.n_clade_bins),
            "multiplier_min": round(float(model.multiplier.min()), 4),
            "multiplier_max": round(float(model.multiplier.max()), 4),
            "min_cell_events": int(model.cell_events.min()),
            "median_cell_events": float(np.median(model.cell_events)),
            "n_empty_cells": int(model.n_empty_cells),
            "max_multiplier_log_se": round(float(np.nanmax(model.multiplier_log_se)), 4),
            "units": "thresholds below are on the z scale, -log10(A * G(s)), "
                     "not raw BADASP scores; per_test_calls.csv carries each "
                     "test's own raw-score threshold.",
        })
        print(f"  anchor u0={model.anchor_score:.4f}  "
              f"A in [{model.multiplier.min():.2f}, {model.multiplier.max():.2f}]  "
              f"min cell events={int(model.cell_events.min())}")

    # --- Primary calibration ---------------------------------------------
    result: ThresholdResult = threshold_at_fdr(
        stat_left, stat_right, null_stat_left, null_stat_right,
        q=args.target_fdr, fdp_quantile=args.fdp_quantile,
    )
    fwer = maxT_fwer_thresholds(
        null_stat_left, null_stat_right, keys["position"].to_numpy(), alpha=args.fwer_alpha
    )

    # The unconditioned rule, always reported as the comparison baseline.
    global_result = (
        result if args.conditioning == "none"
        else threshold_at_fdr(
            obs_left, obs_right, null_left, null_right,
            q=args.target_fdr, fdp_quantile=args.fdp_quantile,
        )
    )

    # obs_stat/null_stat are the primary (possibly cell-conditioned) split
    # statistics; moved up from the per-test-table section below because the
    # optional --operating-point block (next) needs them too.
    obs_stat = _split_statistic(stat_left, stat_right)
    null_stat = _split_statistic(null_stat_left, null_stat_right)

    # --- Optional: describe (not select) the error rate of an externally
    # chosen operating point ------------------------------------------------
    # Off by default. When given, this NEVER touches result.t, criterion_met,
    # zero_discoveries or note -- it only adds an additional, clearly-labelled
    # description of a threshold the caller picked by their own rule (e.g.
    # "call the top N splits"), so a number is reported even when the
    # FDP-exceedance criterion above found nothing.
    error_profile: Optional[dict] = None
    diag_t: Optional[float] = result.t
    diag_threshold_source: Optional[str] = None
    op_t: Optional[float] = None

    if args.operating_point is not None:
        op_kind, op_value = parse_operating_point_spec(args.operating_point)
        finite_primary = obs_stat[np.isfinite(obs_stat)]
        if op_kind == "calls":
            if finite_primary.size == 0:
                raise SystemExit(
                    "--operating-point calls:N requires at least one finite "
                    "observed split statistic; none are finite here."
                )
            n_calls = int(op_value)
            frac = n_calls / finite_primary.size
            op_t = float(np.quantile(finite_primary, 1.0 - frac))
        else:  # "t"
            op_t = float(op_value)
        op_O = int(np.sum(finite_primary >= op_t))

        if result.t is None:
            diag_t = op_t
            diag_threshold_source = (
                "this threshold came from the --operating-point rule "
                f"({args.operating_point!r}), NOT the FDP-exceedance "
                "criterion above, which found no threshold meeting its "
                "target."
            )

        if args.replicate_groups_csv is not None:
            groups_arr, groups_route = load_replicate_groups_csv(args.replicate_groups_csv, used_files)
        else:
            groups_arr, groups_route = infer_replicate_groups(used_files, npz_dir)

        per_rep_p_ac_minus1 = np.mean(null_ac == -1, axis=1)
        elevated_labels = per_rep_p_ac_minus1 > args.elevated_pac_threshold

        if args.error_profile_calls is not None:
            curve_calls = [int(tok) for tok in args.error_profile_calls.split(",") if tok.strip()]
        else:
            curve_calls = list(DEFAULT_ERROR_PROFILE_CALLS)

        def _describe_at(
            t_value: float, ol: np.ndarray, orr: np.ndarray, nl: np.ndarray, nr: np.ndarray,
            groups_for_desc: Optional[np.ndarray],
        ) -> dict:
            desc = describe_threshold(
                t_value, ol, orr, nl, nr,
                labels=elevated_labels, label_statistic=per_rep_p_ac_minus1, groups=groups_for_desc,
                fdp_quantile=args.fdp_quantile,
                n_bootstrap=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_unit="replicate",
            ).to_dict()
            desc["n_bootstrap"] = int(args.bootstrap_resamples)
            desc["bootstrap_seed"] = int(args.bootstrap_seed)
            desc["bootstrap_unit"] = "replicate"
            return desc

        def _describe_cluster_at(
            t_value: float, ol: np.ndarray, orr: np.ndarray, nl: np.ndarray, nr: np.ndarray,
            groups_for_desc: Optional[np.ndarray], reason_if_none: str,
        ) -> dict:
            if groups_for_desc is None:
                return {"skipped": reason_if_none}
            desc = describe_threshold(
                t_value, ol, orr, nl, nr,
                labels=elevated_labels, label_statistic=per_rep_p_ac_minus1, groups=groups_for_desc,
                fdp_quantile=args.fdp_quantile,
                n_bootstrap=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_unit="group",
            ).to_dict()
            desc["n_bootstrap"] = int(args.bootstrap_resamples)
            desc["bootstrap_seed"] = int(args.bootstrap_seed)
            desc["bootstrap_unit"] = "group"
            return desc

        selection_rule = (
            f"t was fixed by the caller's --operating-point rule "
            f"({args.operating_point!r}) and was NOT chosen by optimising any "
            "null-derived quantity (e.g. minimising the achieved FDP or "
            "maximising O over a grid, as threshold_at_fdr does) -- this is "
            "why the reported error rate is not selection-biased."
        )

        error_profile = {
            "operating_point": {"raw": args.operating_point, "t": op_t, "O": op_O},
            "selection_rule": selection_rule,
            "description": _describe_at(op_t, stat_left, stat_right, null_stat_left, null_stat_right, groups_arr),
            "description_cluster_bootstrap": _describe_cluster_at(
                op_t, stat_left, stat_right, null_stat_left, null_stat_right, groups_arr, groups_route
            ),
            "provenance": {
                "replicate_group_resolution": groups_route,
                "elevated_pac_threshold": float(args.elevated_pac_threshold),
            },
        }

        curve_df = describe_threshold_curve(
            stat_left, stat_right, null_stat_left, null_stat_right,
            call_counts=curve_calls,
            labels=elevated_labels, label_statistic=per_rep_p_ac_minus1,
            groups=groups_arr, fdp_quantile=args.fdp_quantile,
        )
        curve_df.to_csv(args.out_dir / "error_profile_curve.csv", index=False)
        print(f"Wrote {args.out_dir / 'error_profile_curve.csv'} "
              f"({len(curve_df)} reporting anchor(s))")

        if args.conditioning == "cell":
            # Mirror global_rule_baseline: the same error_profile, computed on
            # the UNCONDITIONED statistic, for comparison. When the operating
            # point was given as an absolute 't:<float>', that literal number
            # is reused here even though it was chosen on the z scale, purely
            # so the two blocks stay structurally comparable -- see the "note"
            # field below, which flags this rather than pretending the two t
            # values are on the same scale.
            obs_stat_raw = _split_statistic(obs_left, obs_right)
            finite_raw = obs_stat_raw[np.isfinite(obs_stat_raw)]
            if op_kind == "calls" and finite_raw.size > 0:
                op_t_raw = float(np.quantile(finite_raw, 1.0 - (int(op_value) / finite_raw.size)))
                raw_note = None
            else:
                op_t_raw = op_t
                raw_note = (
                    None if op_kind == "calls" else
                    "t was given directly via 't:<float>' on the primary "
                    "(z) scale; the same numeric value is reused here against "
                    "the raw score, which is comparable only in the sense "
                    "that both are literally the same number -- it is NOT a "
                    "rescaled equivalent threshold."
                )
            op_O_raw = int(np.sum(finite_raw >= op_t_raw)) if finite_raw.size else 0
            baseline_profile = {
                "operating_point": {"raw": args.operating_point, "t": op_t_raw, "O": op_O_raw},
                "selection_rule": selection_rule,
                "description": _describe_at(op_t_raw, obs_left, obs_right, null_left, null_right, groups_arr),
                "description_cluster_bootstrap": _describe_cluster_at(
                    op_t_raw, obs_left, obs_right, null_left, null_right, groups_arr, groups_route
                ),
            }
            if raw_note is not None:
                baseline_profile["note"] = raw_note
            error_profile["global_rule_baseline"] = baseline_profile

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
        "conditioning": conditioning_info,
        "global_rule_baseline": {
            "t": global_result.t,
            "O_t": global_result.O_t,
            "E_fp": global_result.E_fp,
            "E_fp_over_O": global_result.E_fp_over_O,
            "fdp_quantile_achieved": global_result.fdp_quantile_achieved,
            "criterion_met": global_result.criterion_met,
            "note": "single global threshold on the raw score, reported for "
                    "comparison regardless of --conditioning.",
        },
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
    if error_profile is not None:
        thresholds_json["error_profile"] = error_profile
    (args.out_dir / "thresholds.json").write_text(
        json.dumps(thresholds_json, indent=2, default=_json_default) + "\n"
    )
    print(f"Wrote {args.out_dir / 'thresholds.json'} (t={result.t}, criterion_met={result.criterion_met})")

    if not result.sweep.empty:
        result.sweep.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
        plot_threshold_sweep(result.sweep, result.t, plots_dir / "threshold_sweep.png")

    # --- Per-test call table (the source of truth) --------------------------
    # Position-conditional thresholds do not fit the legacy
    # {(event_type, clade_bin): t} dict, so this table -- not that dict --
    # is what a reader should cite. It ships each test's own raw-score
    # threshold so it is visible which positions were held to a higher bar.
    # (obs_stat/null_stat were computed earlier, before thresholds.json, so
    # the optional --operating-point block above could use them too.)
    p_values, q_values = per_test_pvalues(obs_stat, null_stat)
    per_test = keys.copy()
    per_test["badasp_score_left"] = obs_left
    per_test["badasp_score_right"] = obs_right
    per_test["split_statistic"] = obs_stat
    per_test["p_value"] = p_values
    per_test["q_value"] = q_values
    if model is not None:
        per_test["z_left"] = stat_left
        per_test["z_right"] = stat_right
        per_test["cell_multiplier_left"] = model.multipliers_for(
            keys["position"].to_numpy(), keys["clade_size_left"].to_numpy(float)
        )
        per_test["cell_multiplier_right"] = model.multipliers_for(
            keys["position"].to_numpy(), keys["clade_size_right"].to_numpy(float)
        )
        per_test["position_group"] = np.where(
            np.isin(keys["position"].to_numpy(), model.hot_positions), "hot", "cold"
        )
        if result.t is not None:
            per_test["threshold_raw_left"] = t_from_z(
                result.t, keys["position"].to_numpy(),
                keys["clade_size_left"].to_numpy(float), model, tail,
            )
            per_test["threshold_raw_right"] = t_from_z(
                result.t, keys["position"].to_numpy(),
                keys["clade_size_right"].to_numpy(float), model, tail,
            )
    if diag_t is not None:
        per_test["called"] = obs_stat >= diag_t
    per_test.to_csv(args.out_dir / "per_test_calls.csv", index=False)
    print(f"Wrote {args.out_dir / 'per_test_calls.csv'} ({len(per_test):,} rows"
          + (f", {int(per_test['called'].sum()):,} called)" if diag_t is not None else ")"))

    # --- Diagnostics ----------------------------------------------------------
    diagnostics: Dict[str, object] = {}

    diagnostics["self_calibration"] = self_calibration_diagnostic(
        null_stat_left, null_stat_right, diag_t, args.holdout_fraction, seed
    )
    if diag_threshold_source is not None and "skipped" not in diagnostics["self_calibration"]:
        diagnostics["self_calibration"]["threshold_source"] = diag_threshold_source

    if args.conditioning == "cell":
        diagnostics["conditioned_cross_validation"] = conditioned_cv_diagnostic(
            obs_left, obs_right, null_left, null_right, keys,
            q=args.target_fdr, fdp_quantile=args.fdp_quantile,
            n_hot=args.n_hot, n_clade_bins=args.n_clade_bins,
            anchor_alpha=args.anchor_alpha, n_folds=args.cv_folds,
        )

    site_rates = load_site_rates(args.site_rate_file)
    flatness_df, flatness_note = compute_flatness_diagnostics(
        null_stat, keys, diag_t, site_rates, args.num_bins_decile, args.num_bins_quartile
    )
    if not flatness_df.empty:
        flatness_df.to_csv(args.out_dir / "exceedance_flatness.csv", index=False)
        plot_flatness(flatness_df, plots_dir / "exceedance_flatness.png")
    if diag_threshold_source is not None and "skipped" not in flatness_note:
        flatness_note["threshold_source"] = diag_threshold_source
    diagnostics["exceedance_flatness"] = {**flatness_note, "n_rows": int(len(flatness_df))}

    diagnostics["ac_plus1_control"] = ac_plus1_control(
        obs_stat, null_stat, keys["ac"].to_numpy(), diag_t, null_ac=null_ac
    )
    if diag_threshold_source is not None and "skipped" not in diagnostics["ac_plus1_control"]:
        diagnostics["ac_plus1_control"]["threshold_source"] = diag_threshold_source
    if diag_t is not None:
        plot_ac_plus1_control(obs_stat, null_stat, keys["ac"].to_numpy(), plots_dir / "ac_plus1_control.png")

    diagnostics["p_ac_minus1"] = p_ac_minus1_comparison(keys["ac"].to_numpy(), null_ac)

    diagnostics["head_to_head_vs_percentile_rule"] = head_to_head_vs_percentile_rule(
        keys, obs_left, obs_right, diag_t, args.percentile_comparison,
        args.num_bins_decile,
        calibrated_calls=(obs_stat >= diag_t) if (model is not None and diag_t is not None) else None,
    )
    if diag_threshold_source is not None and "skipped" not in diagnostics["head_to_head_vs_percentile_rule"]:
        diagnostics["head_to_head_vs_percentile_rule"]["threshold_source"] = diag_threshold_source

    # --- Legacy bin-threshold dict (compatibility shim) ---------------------
    # The four downstream analysis scripts consume
    # {(event_type, clade_bin): raw threshold}. Keep emitting it so they run
    # unmodified, but it cannot express the position component -- see
    # conditioned_bin_threshold_dict's docstring.
    if model is not None and result.t is not None:
        _, bin_provenance = conditioned_bin_threshold_dict(
            keys, result.t, tail, model,
            keys["event_type"].unique(), args.num_bins_decile,
        )
        (args.out_dir / "legacy_bin_thresholds.json").write_text(
            json.dumps(bin_provenance, indent=2, default=_json_default) + "\n"
        )
        print(f"Wrote {args.out_dir / 'legacy_bin_thresholds.json'} "
              "(lossy projection; per_test_calls.csv is the source of truth)")

    (args.out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=_json_default) + "\n"
    )
    print(f"Wrote {args.out_dir / 'diagnostics.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
