"""Tests for scripts/calibrate_switch_threshold.py.

Uses small synthetic observed/null tables (score_null_replicate.py's actual
.npz shape: SCORE_COLUMNS arrays, no key arrays, pre-aligned by row order)
rather than real ASR output, so these run in well under a second and don't
need IQ-TREE. A handful of tests get a large planted score so a threshold is
reliably found, exercising the "success" diagnostics path; a separate test
covers the pure-noise "no threshold found" path.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import calibrate_switch_threshold as cst

RNG_SEED = 4242
EVENT_TYPES = ["Duplication", "Speciation", "Transfer"]


def _write_observed_scores(path: Path, n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame({
        "node_name": [f"Node{i}" for i in range(n)],
        "event_type": rng.choice(EVENT_TYPES, size=n),
        "left_child": [f"L{i}" for i in range(n)],
        "right_child": [f"R{i}" for i in range(n)],
        "position": rng.integers(1, 31, size=n),
        "aa_left": "A",
        "aa_right": "G",
        "rc_left": rng.uniform(0, 1, size=n),
        "rc_right": rng.uniform(0, 1, size=n),
        "ac": rng.choice([-1.0, 1.0], size=n, p=[0.3, 0.7]),
        "p_ac_left": rng.uniform(0, 1, size=n),
        "p_ac_right": rng.uniform(0, 1, size=n),
        "badasp_score_left": rng.normal(0, 0.3, size=n),
        "badasp_score_right": rng.normal(0, 0.3, size=n),
        "distance_from_root": rng.uniform(0.1, 20, size=n),
        "clade_size_left": rng.integers(5, 500, size=n),
        "clade_size_right": rng.integers(5, 500, size=n),
    })
    df["clade_size_total"] = df["clade_size_left"] + df["clade_size_right"]
    df.to_csv(path, index=False)
    return df


def _write_null_replicates(directory: Path, n: int, R: int, rng: np.random.Generator) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for r in range(R):
        payload = {
            "rc_left": rng.uniform(0, 1, size=n).astype(np.float32),
            "rc_right": rng.uniform(0, 1, size=n).astype(np.float32),
            "ac": rng.choice([-1.0, 1.0], size=n, p=[0.14, 0.86]).astype(np.float32),
            "p_ac_left": rng.uniform(0, 1, size=n).astype(np.float32),
            "p_ac_right": rng.uniform(0, 1, size=n).astype(np.float32),
            "badasp_score_left": rng.normal(0, 0.3, size=n).astype(np.float32),
            "badasp_score_right": rng.normal(0, 0.3, size=n).astype(np.float32),
        }
        np.savez_compressed(directory / f"rep_{r:04d}.npz", **payload)


def _build_fixture(tmp_path, n=600, R=30, plant=True, write_manifest=True):
    rng = np.random.default_rng(RNG_SEED)
    obs_path = tmp_path / "observed_scores.csv"
    df = _write_observed_scores(obs_path, n, rng)

    run_dir = tmp_path / "null_run"
    npz_dir = run_dir / "npz"
    _write_null_replicates(npz_dir, n, R, rng)

    if plant:
        # Plant a handful of large observed scores so a threshold is
        # reliably found (mirrors test_null_model.py's own
        # test_recovery_with_planted_signal approach).
        df.loc[df.index[-5:], "badasp_score_left"] = 10.0
        df.to_csv(obs_path, index=False)

    if write_manifest:
        manifest = {
            "created": "2026-01-01T00:00:00+00:00",
            "git_sha": "deadbeef",
            "seed": 777,
            "shrinkage": 0.15,
            "inputs": {"observed_scores": cst.file_info(obs_path)},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))

    return obs_path, run_dir


# ---------------------------------------------------------------------------
# Loading / consistency checks
# ---------------------------------------------------------------------------


def test_load_null_replicates_stacks_all_files(tmp_path):
    _, run_dir = _build_fixture(tmp_path, n=50, R=4)
    left, right, ac, used = cst.load_null_replicates(run_dir / "npz", n_tests=50)
    assert left.shape == (4, 50)
    assert right.shape == (4, 50)
    assert ac.shape == (4, 50)
    assert len(used) == 4


def test_load_null_replicates_rejects_length_mismatch(tmp_path):
    _, run_dir = _build_fixture(tmp_path, n=50, R=2)
    with pytest.raises(ValueError, match="aligned to a different"):
        cst.load_null_replicates(run_dir / "npz", n_tests=999)
    # a genuine per-file mismatch (not "all files wrong") raises ValueError
    npz_dir = run_dir / "npz"
    rng = np.random.default_rng(0)
    bad_payload = {c: rng.normal(size=10).astype(np.float32) for c in
                   ("rc_left", "rc_right", "ac", "p_ac_left", "p_ac_right",
                    "badasp_score_left", "badasp_score_right")}
    np.savez_compressed(npz_dir / "rep_9999.npz", **bad_payload)
    with pytest.raises(ValueError, match="aligned to a different"):
        cst.load_null_replicates(npz_dir, n_tests=50)


def test_check_observed_scores_consistency_passes_when_matching(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    note = cst.check_observed_scores_consistency(obs_path, manifest)
    assert note is None


def test_check_observed_scores_consistency_fails_on_size_mismatch(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    obs_path.write_text(obs_path.read_text() + "extra,columns,appended\n1,2,3\n")
    with pytest.raises(SystemExit, match="no longer valid"):
        cst.check_observed_scores_consistency(obs_path, manifest)


def test_check_observed_scores_consistency_none_manifest_returns_warning():
    note = cst.check_observed_scores_consistency(Path("/dev/null"), None)
    assert note is not None and "no run_manifest.json" in note


# ---------------------------------------------------------------------------
# End-to-end main(), planted-signal case (threshold IS found)
# ---------------------------------------------------------------------------


def test_main_end_to_end_with_planted_signal(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=30, plant=True)
    out_dir = tmp_path / "calib_out"
    rc = cst.main([
        "--null-run-dir", str(run_dir),
        "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir),
        "--target-fdr", "0.2",
        "--fdp-quantile", "0.8",
    ])
    assert rc == 0

    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert thresholds["R"] == 30
    assert thresholds["seed"] == 777
    assert thresholds["shrinkage"] == 0.15
    assert thresholds["criterion_met"] is True
    assert thresholds["t"] is not None
    assert thresholds["O_t"] >= 1
    assert "git_sha" in thresholds
    assert thresholds["inputs"]["observed_scores"]["exists"] is True
    assert len(thresholds["fwer_per_position_threshold"]) > 0

    per_test = pd.read_csv(out_dir / "per_test_pvalues.csv")
    assert len(per_test) == 600
    assert {"p_value", "q_value", "split_statistic"}.issubset(per_test.columns)
    # 1/(1+R) floor, R=30 -- tiny tolerance for to_csv's default float precision
    assert (per_test["p_value"] >= 1.0 / 31 - 1e-9).all()

    diagnostics = json.loads((out_dir / "diagnostics.json").read_text())
    assert "skipped" not in diagnostics["self_calibration"]
    assert diagnostics["self_calibration"]["threshold_used"] == thresholds["t"]
    assert 0 <= diagnostics["self_calibration"]["mean_fdp_hat"]

    assert (out_dir / "exceedance_flatness.csv").exists()
    flat = pd.read_csv(out_dir / "exceedance_flatness.csv")
    assert set(flat["covariate"].unique()) >= {
        "clade_size_total_decile", "distance_from_root_quartile", "position", "event_type"
    }
    assert diagnostics["exceedance_flatness"]["site_rate_diagnostic"] == "skipped: no --site-rate-file given"

    ac_ctrl = diagnostics["ac_plus1_control"]
    assert ac_ctrl["n_rows"] > 0
    assert "ks_test" in ac_ctrl

    p_minus1 = diagnostics["p_ac_minus1"]
    assert 0 <= p_minus1["p_observed"] <= 1
    assert 0 <= p_minus1["p_null_mean"] <= 1

    h2h = diagnostics["head_to_head_vs_percentile_rule"]
    assert "confusion_matrix" in h2h
    assert h2h["calibrated_rule"]["threshold"] == thresholds["t"]

    assert (out_dir / "plots" / "threshold_sweep.png").exists()
    assert (out_dir / "plots" / "exceedance_flatness.png").exists()
    assert (out_dir / "plots" / "ac_plus1_control.png").exists()


def test_main_pure_noise_reports_criterion_not_met_honestly(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    out_dir = tmp_path / "calib_out_noise"
    rc = cst.main([
        "--null-run-dir", str(run_dir),
        "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir),
        "--target-fdr", "0.01",
        "--fdp-quantile", "0.95",
    ])
    assert rc == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert thresholds["criterion_met"] is False
    assert thresholds["t"] is None
    assert thresholds["note"]  # a non-empty explanation, not silently blank

    diagnostics = json.loads((out_dir / "diagnostics.json").read_text())
    # nothing that needs a threshold is silently computed anyway
    assert "skipped" in diagnostics["self_calibration"]
    assert "skipped" in diagnostics["exceedance_flatness"]
    assert "skipped" in diagnostics["head_to_head_vs_percentile_rule"]
    # AC=+1 control and P(AC=-1) don't need a threshold, so they still run
    assert "skipped" not in diagnostics["ac_plus1_control"]
    assert "skipped" not in diagnostics["p_ac_minus1"]


def test_main_requires_observed_scores_or_manifest(tmp_path):
    _, run_dir = _build_fixture(tmp_path, write_manifest=False)
    out_dir = tmp_path / "calib_out"
    with pytest.raises(SystemExit, match="run_manifest.json"):
        cst.main(["--null-run-dir", str(run_dir), "--out-dir", str(out_dir)])


# ---------------------------------------------------------------------------
# Optional site-rate covariate
# ---------------------------------------------------------------------------


def test_load_site_rates_none_when_not_given():
    assert cst.load_site_rates(None) is None


def test_load_site_rates_parses_site_and_rate_columns(tmp_path):
    path = tmp_path / "sites.rate"
    path.write_text("Site\tCategory\tRate\n1\t1\t0.5\n2\t2\t1.5\n3\t1\t0.9\n")
    rates = cst.load_site_rates(path)
    assert rates.loc[2] == pytest.approx(1.5)


def test_load_site_rates_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="Site/Position"):
        cst.load_site_rates(path)


def test_site_rate_diagnostic_included_when_file_given(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=30, plant=True)
    rate_path = tmp_path / "sites.rate"
    rates_df = pd.DataFrame({"Site": range(1, 31), "Rate": np.linspace(0.1, 3.0, 30)})
    rates_df.to_csv(rate_path, sep="\t", index=False)

    out_dir = tmp_path / "calib_out_rate"
    rc = cst.main([
        "--null-run-dir", str(run_dir),
        "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir),
        "--target-fdr", "0.2",
        "--fdp-quantile", "0.8",
        "--site-rate-file", str(rate_path),
    ])
    assert rc == 0
    diagnostics = json.loads((out_dir / "diagnostics.json").read_text())
    assert diagnostics["exceedance_flatness"]["site_rate_diagnostic"] == "computed"
    flat = pd.read_csv(out_dir / "exceedance_flatness.csv")
    assert "site_rate_quartile" in set(flat["covariate"].unique())
