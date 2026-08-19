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

    per_test = pd.read_csv(out_dir / "per_test_calls.csv")
    assert len(per_test) == 600
    assert {"p_value", "q_value", "split_statistic", "called"}.issubset(per_test.columns)
    # Conditioning is the default, so each test carries its own raw-score bar.
    assert {"z_left", "z_right", "position_group",
            "threshold_raw_left", "threshold_raw_right"}.issubset(per_test.columns)
    assert per_test["called"].sum() == thresholds["O_t"]
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
    assert h2h["calibrated_rule"]["threshold_z"] == thresholds["t"]

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


# ---------------------------------------------------------------------------
# Cell conditioning
# ---------------------------------------------------------------------------


def test_conditioning_none_reproduces_the_global_rule(tmp_path):
    """--conditioning none must leave the primary threshold on the raw score
    scale and identical to the reported baseline."""
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=30, plant=True)
    out_dir = tmp_path / "calib_none"
    assert cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.2", "--fdp-quantile", "0.8",
        "--conditioning", "none",
    ]) == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert thresholds["conditioning"]["mode"] == "none"
    assert thresholds["t"] == thresholds["global_rule_baseline"]["t"]
    assert thresholds["O_t"] == thresholds["global_rule_baseline"]["O_t"]
    per_test = pd.read_csv(out_dir / "per_test_calls.csv")
    assert "z_left" not in per_test.columns


def test_conditioning_reports_the_global_rule_as_a_baseline(tmp_path):
    """The comparison must always be available, so a conditioned run still
    reports what the single global threshold would have given."""
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=30, plant=True)
    out_dir = tmp_path / "calib_cell"
    assert cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.2", "--fdp-quantile", "0.8",
    ]) == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert thresholds["conditioning"]["mode"] == "cell"
    baseline = thresholds["global_rule_baseline"]
    assert baseline["criterion_met"] is True
    # The conditioned threshold is on the z scale, the baseline on the raw
    # score scale; they must not be conflated.
    assert thresholds["t"] != baseline["t"]
    assert "z scale" in thresholds["conditioning"]["units"]


def test_conditioned_cross_validation_holds_out_replicates(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=30, plant=True)
    out_dir = tmp_path / "calib_cv"
    assert cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.2", "--fdp-quantile", "0.8",
        "--cv-folds", "3",
    ]) == 0
    cv = json.loads((out_dir / "diagnostics.json").read_text())["conditioned_cross_validation"]
    assert "skipped" not in cv
    assert cv["n_folds"] == 3
    assert len(cv["per_fold"]) == 3
    for fold in cv["per_fold"]:
        if fold.get("criterion_met"):
            # Every replicate is used exactly once as held-out.
            assert fold["n_in_fold"] + fold["n_held_out"] == 30


def test_conditioned_cv_skips_when_replicates_are_too_few(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=4, plant=True)
    out_dir = tmp_path / "calib_cv_few"
    cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.2", "--fdp-quantile", "0.8",
        "--cv-folds", "4",
    ])
    cv = json.loads((out_dir / "diagnostics.json").read_text())["conditioned_cross_validation"]
    assert "skipped" in cv


def test_legacy_bin_dict_is_emitted_and_flagged_lossy(tmp_path):
    """Downstream consumers still need the {(event, bin): t} shape, but it
    cannot express the position component and must say so."""
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=30, plant=True)
    out_dir = tmp_path / "calib_bins"
    assert cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.2", "--fdp-quantile", "0.8",
    ]) == 0
    bins = json.loads((out_dir / "legacy_bin_thresholds.json").read_text())
    assert bins["lossy_projection"] is True
    assert bins["source_of_truth"] == "per_test_calls.csv"
    assert len(bins["per_bin_threshold"]) > 1


# ---------------------------------------------------------------------------
# --operating-point / error_profile (describe-only threshold reporting)
# ---------------------------------------------------------------------------


def test_error_profile_absent_without_operating_point(tmp_path):
    """The additive invariant: with no --operating-point, nothing new appears."""
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    out_dir = tmp_path / "calib_no_op"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
    ])
    assert rc == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert "error_profile" not in thresholds
    assert not (out_dir / "error_profile_curve.csv").exists()


def test_operating_point_calls_reports_error_profile_on_pure_noise(tmp_path):
    """--operating-point works even when the FDP criterion finds nothing:
    error_profile is populated while t/criterion_met stay honest (None/False).
    Uses a larger n so the default --error-profile-calls anchors (up to 2023)
    are within the number of finite observed split statistics available.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=2200, R=15, plant=False)
    out_dir = tmp_path / "calib_op_noise"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--bootstrap-resamples", "50", "--bootstrap-seed", "1",
    ])
    assert rc == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert thresholds["criterion_met"] is False
    assert thresholds["t"] is None
    assert thresholds["note"]

    ep = thresholds["error_profile"]
    assert ep["operating_point"]["raw"] == "calls:20"
    assert ep["operating_point"]["t"] is not None
    assert ep["description"]["O"] == ep["operating_point"]["O"]
    assert ep["description"]["n_bootstrap"] == 50
    assert ep["description"]["bootstrap_unit"] == "replicate"
    assert "NOT chosen by optimising" in ep["selection_rule"]
    assert "replicate_group_resolution" in ep["provenance"]

    assert (out_dir / "error_profile_curve.csv").exists()
    curve = pd.read_csv(out_dir / "error_profile_curve.csv")
    assert list(curve["target_calls"]) == sorted(cst.DEFAULT_ERROR_PROFILE_CALLS)


def test_operating_point_custom_error_profile_calls(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    out_dir = tmp_path / "calib_op_custom_curve"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    curve = pd.read_csv(out_dir / "error_profile_curve.csv")
    assert list(curve["target_calls"]) == [5, 20, 50]


def test_gated_diagnostics_unblocked_with_operating_point(tmp_path):
    """self_calibration, exceedance_flatness, ac_plus1_control and
    head_to_head_vs_percentile_rule normally short-circuit with 'skipped'
    when the FDP criterion finds no t; --operating-point unblocks them
    without touching thresholds.json's own criterion fields.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    out_dir = tmp_path / "calib_gated"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0

    diagnostics = json.loads((out_dir / "diagnostics.json").read_text())
    for key in ("self_calibration", "exceedance_flatness", "head_to_head_vs_percentile_rule"):
        d = diagnostics[key]
        assert "skipped" not in d
        assert "threshold_source" in d
        assert "operating-point" in d["threshold_source"]
    assert "threshold_source" in diagnostics["ac_plus1_control"]
    assert (out_dir / "plots" / "ac_plus1_control.png").exists()

    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert thresholds["criterion_met"] is False
    assert thresholds["t"] is None
    assert thresholds["zero_discoveries"] is False
    assert thresholds["note"]


def test_per_test_calls_called_column_only_with_operating_point(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)

    out_dir_no_flag = tmp_path / "calib_called_no_flag"
    cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir_no_flag), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
    ])
    per_test_no_flag = pd.read_csv(out_dir_no_flag / "per_test_calls.csv")
    assert "called" not in per_test_no_flag.columns

    out_dir = tmp_path / "calib_called"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    per_test = pd.read_csv(out_dir / "per_test_calls.csv")
    assert "called" in per_test.columns
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    assert int(per_test["called"].sum()) == thresholds["error_profile"]["operating_point"]["O"]


def test_operating_point_malformed_exits_naming_accepted_forms(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    out_dir = tmp_path / "calib_bad_op"
    with pytest.raises(SystemExit, match="calls:<int>"):
        cst.main([
            "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
            "--out-dir", str(out_dir), "--operating-point", "bogus",
        ])


def _assert_json_primitive(value, path="$"):
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_json_primitive(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_json_primitive(v, f"{path}[{i}]")
    else:
        assert isinstance(value, (str, int, float, bool)) or value is None, (
            f"non-JSON-primitive type {type(value)} at {path}: {value!r}"
        )


def test_thresholds_json_fully_serializable_with_operating_point(tmp_path):
    """Guards against numpy types leaking into thresholds.json -- including
    via a --replicate-groups-csv column that pandas infers as int64 because
    every invocation label happens to look numeric.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=6, plant=False)
    npz_dir = run_dir / "npz"
    files = sorted(npz_dir.glob("rep_*.npz"))
    csv_path = tmp_path / "groups.csv"
    rows = ["rep_file,invocation"] + [f"{f.name},{100 + i}" for i, f in enumerate(files)]
    csv_path.write_text("\n".join(rows) + "\n")

    out_dir = tmp_path / "calib_serializable"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--error-profile-calls", "5,20,50",
        "--replicate-groups-csv", str(csv_path),
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    parsed = json.loads((out_dir / "thresholds.json").read_text())
    _assert_json_primitive(parsed)


# ---------------------------------------------------------------------------
# infer_replicate_groups: the four provenance-resolution routes
# ---------------------------------------------------------------------------


def _touch_npz_files(npz_dir: Path, n: int) -> list:
    npz_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        f = npz_dir / f"rep_{i:04d}.npz"
        f.write_bytes(b"")
        files.append(f)
    return files


def test_infer_replicate_groups_route2_provenance_json(tmp_path):
    run_dir = tmp_path / "staged"
    npz_dir = run_dir / "npz"
    files = _touch_npz_files(npz_dir, 3)
    provenance = {
        "replicates": [
            {"rep_id": "rep_0000", "invocation": "5"},
            {"rep_id": "rep_0001", "invocation": "6"},
            {"rep_id": "rep_0002", "invocation": "7"},
        ]
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance))

    groups, route = cst.infer_replicate_groups(files, npz_dir)
    assert groups is not None
    assert list(groups) == ["5", "6", "7"]
    assert "provenance.json" in route


def test_infer_replicate_groups_route3_progress_jsonl(tmp_path):
    run_dir = tmp_path / "batch"
    npz_dir = run_dir / "npz"
    files = _touch_npz_files(npz_dir, 3)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    records = [{"replicate": i, "batch": i // 2} for i in range(3)]
    (logs_dir / "progress.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    groups, route = cst.infer_replicate_groups(files, npz_dir)
    assert groups is not None
    assert list(groups) == [0, 0, 1]
    assert "progress.jsonl" in route


def test_infer_replicate_groups_route4_none_with_reason(tmp_path):
    run_dir = tmp_path / "bare"
    npz_dir = run_dir / "npz"
    files = _touch_npz_files(npz_dir, 2)

    groups, reason = cst.infer_replicate_groups(files, npz_dir)
    assert groups is None
    assert "index" in reason  # explains the arithmetic-on-index fallback is refused


def test_infer_replicate_groups_route1_merge_manifest_recurses(tmp_path):
    """A merged run_manifest.json (merge_null_runs.py's shape) resolves by
    recursing into each source directory's own provenance; a source with no
    resolvable provenance of its own falls back to its run-directory name
    (real recorded provenance, not index arithmetic) -- mirroring pooled_41's
    actual euler_run (bare) + treelen_scan (has provenance.json) composition.
    """
    sub_a = tmp_path / "subA"
    sub_a_npz = sub_a / "npz"
    sub_a_files = _touch_npz_files(sub_a_npz, 2)
    (sub_a / "provenance.json").write_text(json.dumps({
        "replicates": [
            {"rep_id": "rep_0000", "invocation": "100"},
            {"rep_id": "rep_0001", "invocation": "101"},
        ]
    }))

    sub_b = tmp_path / "subB"
    sub_b_npz = sub_b / "npz"
    sub_b_files = _touch_npz_files(sub_b_npz, 1)  # bare: no provenance anywhere

    merged = tmp_path / "merged"
    merged_npz = merged / "npz"
    merged_npz.mkdir(parents=True)
    all_sources = sub_a_files + sub_b_files
    merged_files = []
    replicates_manifest = []
    for idx, src in enumerate(all_sources):
        dst = merged_npz / f"rep_{idx:04d}.npz"
        dst.write_bytes(b"")
        merged_files.append(dst)
        replicates_manifest.append({"merged_index": idx, "source": str(src)})
    (merged / "run_manifest.json").write_text(json.dumps({"replicates": replicates_manifest}))

    groups, route = cst.infer_replicate_groups(merged_files, merged_npz)
    assert groups is not None
    assert "run_manifest.json" in route
    assert list(groups) == ["subA:100", "subA:101", "subB"]


def test_replicate_groups_csv_overrides_automatic_resolution(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=6, plant=False)
    npz_dir = run_dir / "npz"
    files = sorted(npz_dir.glob("rep_*.npz"))
    csv_path = tmp_path / "groups.csv"
    rows = ["rep_file,invocation"] + [f"{f.name},grpX" for f in files]
    csv_path.write_text("\n".join(rows) + "\n")

    out_dir = tmp_path / "calib_groups_csv"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--error-profile-calls", "5,20,50",
        "--replicate-groups-csv", str(csv_path),
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    prov = thresholds["error_profile"]["provenance"]["replicate_group_resolution"]
    assert "--replicate-groups-csv" in prov
    assert "explicit override" in prov
    gc = thresholds["error_profile"]["description"]["group_composition"]
    assert gc is not None
    assert set(gc.keys()) == {"grpX"}
    assert gc["grpX"]["n"] == 6


# ---------------------------------------------------------------------------
# --min-clade-filter and --operating-point pct:<float>
# ---------------------------------------------------------------------------
#
# The fixture's clade_size_left/clade_size_right are drawn (via _build_fixture
# -> _write_observed_scores) from rng.integers(5, 500, ...) with a fixed seed,
# so for a given n the number of rows passing a given --min-clade-filter is a
# deterministic (if not hand-derivable) property of that seed. The values
# used below (n=600, filter=250 -> 153 rows in stratum; filter=500 -> 0 rows)
# were found by running the same draw once and reading off the counts, not
# hand-picked to make a test pass after the fact.


def test_min_clade_filter_reduces_test_count_and_records_stratum(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=False)
    df = pd.read_csv(obs_path)
    n_tests_total = len(df)
    expected_in_stratum = int((np.minimum(df["clade_size_left"], df["clade_size_right"]) >= 250).sum())
    assert 100 <= expected_in_stratum < n_tests_total  # the filter must actually reduce the count

    out_dir = tmp_path / "calib_stratum"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--conditioning", "none",
        "--min-clade-filter", "250", "--operating-point", "calls:20",
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    stratum = thresholds["error_profile"]["stratum"]
    assert stratum["min_clade_filter"] == 250
    assert stratum["n_tests_total"] == n_tests_total
    assert stratum["n_tests_in_stratum"] == expected_in_stratum
    assert stratum["n_scorable_in_stratum"] == expected_in_stratum  # no NaNs in this fixture
    assert stratum["fraction_retained"] == pytest.approx(expected_in_stratum / n_tests_total)

    op = thresholds["error_profile"]["operating_point"]
    assert op["form"] == "calls"
    assert op["scope"] == "stratum"


def test_operating_point_pct_and_calls_agree_within_stratum(tmp_path):
    """A 'calls:<int>' operating point and the 'pct:<float>' operating point
    giving the exact same within-stratum fraction must resolve to the same
    t -- both go through the identical
    ``np.quantile(finite_stratum, 1 - fraction)`` call on the identical
    filtered array, so this is an equality, not merely an approximation.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=False)
    df = pd.read_csv(obs_path)
    n_in_stratum = int((np.minimum(df["clade_size_left"], df["clade_size_right"]) >= 250).sum())
    n_calls = 15
    pct = n_calls / n_in_stratum  # exact same float division `main` performs for calls:<int>

    common_args = [
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--target-fdr", "0.01", "--fdp-quantile", "0.95", "--conditioning", "none",
        "--min-clade-filter", "250", "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ]

    out_calls = tmp_path / "calib_calls"
    cst.main(common_args + ["--out-dir", str(out_calls), "--operating-point", f"calls:{n_calls}"])
    out_pct = tmp_path / "calib_pct"
    cst.main(common_args + ["--out-dir", str(out_pct), "--operating-point", f"pct:{pct!r}"])

    t_calls = json.loads((out_calls / "thresholds.json").read_text())["error_profile"]["operating_point"]["t"]
    t_pct = json.loads((out_pct / "thresholds.json").read_text())["error_profile"]["operating_point"]["t"]
    assert t_calls == pytest.approx(t_pct, rel=1e-12)


def test_min_clade_filter_called_column_respects_stratum(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=True)
    out_dir = tmp_path / "calib_called_stratum"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.20", "--fdp-quantile", "0.95",
        "--conditioning", "none",
        "--min-clade-filter", "250", "--operating-point", "pct:0.1",
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    per_test = pd.read_csv(out_dir / "per_test_calls.csv")
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    O = thresholds["error_profile"]["operating_point"]["O"]

    assert "in_stratum" in per_test.columns
    assert not per_test.loc[~per_test["in_stratum"], "called"].any()
    assert int(per_test["called"].sum()) == O


def test_min_clade_filter_too_few_scorable_tests_exits(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=False)
    out_dir = tmp_path / "calib_too_few"
    with pytest.raises(SystemExit, match=r"0 scorable"):
        cst.main([
            "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
            "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
            "--conditioning", "none",
            "--min-clade-filter", "500", "--operating-point", "calls:20",
        ])


def test_operating_point_pct_out_of_range_rejected(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    out_dir = tmp_path / "calib_bad_pct"
    for bad in ("pct:0", "pct:1", "pct:1.5", "pct:-0.1"):
        with pytest.raises(SystemExit, match=r"between 0 and 1"):
            cst.main([
                "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
                "--out-dir", str(out_dir), "--operating-point", bad,
            ])


def test_min_clade_filter_invariant_unfiltered_thresholds_unchanged(tmp_path):
    """The central invariant of --min-clade-filter: it must not change
    thresholds['t'], criterion_met, or threshold_sweep.csv relative to a run
    without it -- those are computed on the unfiltered data exactly as
    before; only the error_profile block, error_profile_curve.csv, and the
    per_test_calls.csv 'called'/'in_stratum' columns are allowed to differ.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=True)

    out_unfiltered = tmp_path / "calib_unfiltered"
    cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_unfiltered), "--target-fdr", "0.20", "--fdp-quantile", "0.95",
        "--conditioning", "none",
    ])
    out_filtered = tmp_path / "calib_filtered"
    cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_filtered), "--target-fdr", "0.20", "--fdp-quantile", "0.95",
        "--conditioning", "none",
        "--min-clade-filter", "250", "--operating-point", "pct:0.1",
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])

    t_unfiltered = json.loads((out_unfiltered / "thresholds.json").read_text())
    t_filtered = json.loads((out_filtered / "thresholds.json").read_text())
    assert t_filtered["t"] == t_unfiltered["t"]
    assert t_filtered["criterion_met"] == t_unfiltered["criterion_met"]

    sweep_unfiltered = pd.read_csv(out_unfiltered / "threshold_sweep.csv")
    sweep_filtered = pd.read_csv(out_filtered / "threshold_sweep.csv")
    pd.testing.assert_frame_equal(sweep_unfiltered, sweep_filtered)


# ---------------------------------------------------------------------------
# --comparison-null-run-dir: the two-null error bracket
# ---------------------------------------------------------------------------


def _build_comparison_null_dir(tmp_path: Path, name: str, n: int, R: int, seed: int) -> Path:
    """A second, independently-built null run directory in the same
    <dir>/npz/rep_*.npz layout `load_null_replicates` (and therefore
    --comparison-null-run-dir) expects, reusing `_write_null_replicates` so
    it is the identical on-disk shape as the primary fixture's own null.
    """
    rng = np.random.default_rng(seed)
    comp_run_dir = tmp_path / name
    _write_null_replicates(comp_run_dir / "npz", n, R, rng)
    return comp_run_dir


def test_comparison_null_without_operating_point_exits(tmp_path):
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    comp_run_dir = _build_comparison_null_dir(tmp_path, "comp_null_no_op", n=300, R=5, seed=101)
    out_dir = tmp_path / "calib_comp_no_op"
    with pytest.raises(SystemExit, match="--operating-point"):
        cst.main([
            "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
            "--out-dir", str(out_dir),
            "--comparison-null-run-dir", str(comp_run_dir),
        ])


def test_comparison_null_reports_bracket_at_common_threshold(tmp_path):
    """error_profile['comparison'] exists, is evaluated at the SAME t the
    primary run's own error_profile['description'] resolved (never a
    threshold separately derived from the comparison null), and its bracket
    correctly brackets [min, max] of the two nulls' fdp_mean with consistent
    naming.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=True)
    comp_run_dir = _build_comparison_null_dir(tmp_path, "comp_null", n=600, R=7, seed=202)
    out_dir = tmp_path / "calib_comp"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.20", "--fdp-quantile", "0.95",
        "--conditioning", "none",
        "--operating-point", "pct:0.02",
        "--comparison-null-run-dir", str(comp_run_dir),
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    thresholds = json.loads((out_dir / "thresholds.json").read_text())
    ep = thresholds["error_profile"]
    comparison = ep["comparison"]

    assert comparison["null_run_dir"] == str(comp_run_dir)
    assert comparison["R"] == 7
    assert len(comparison["null_replicate_files"]) == 7
    # The essential requirement: both nulls evaluated at ONE common t.
    assert comparison["description"]["t"] == ep["description"]["t"]

    bracket = comparison["bracket"]
    assert bracket["fdp_low"] <= bracket["fdp_high"]
    assert bracket["bracket_width"] == pytest.approx(bracket["fdp_high"] - bracket["fdp_low"])

    primary_mean = ep["description"]["fdp_mean"]
    comp_mean = comparison["description"]["fdp_mean"]
    if primary_mean <= comp_mean:
        assert bracket["which_null_is_low"] == "primary"
        assert bracket["which_null_is_high"] == "comparison"
        assert bracket["fdp_low"] == pytest.approx(primary_mean)
        assert bracket["fdp_high"] == pytest.approx(comp_mean)
    else:
        assert bracket["which_null_is_low"] == "comparison"
        assert bracket["which_null_is_high"] == "primary"
        assert bracket["fdp_low"] == pytest.approx(comp_mean)
        assert bracket["fdp_high"] == pytest.approx(primary_mean)

    caveat = comparison["caveat"].lower()
    assert "opposite" in caveat
    assert "o/e" in caveat or "observed/expected" in caveat

    curve = pd.read_csv(out_dir / "error_profile_curve.csv")
    assert "comparison_fdp_mean" in curve.columns
    assert curve["comparison_fdp_mean"].notna().all()


def test_comparison_null_mismatched_test_count_rejected(tmp_path):
    """A comparison null aligned to a different-sized observed table is
    rejected by load_null_replicates' own length-mismatch guard -- no
    second loader was written for this flag.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    comp_run_dir = _build_comparison_null_dir(tmp_path, "comp_null_bad_n", n=250, R=5, seed=303)
    out_dir = tmp_path / "calib_comp_bad_n"
    with pytest.raises(ValueError, match="aligned to a different"):
        cst.main([
            "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
            "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
            "--operating-point", "calls:20",
            "--comparison-null-run-dir", str(comp_run_dir),
            "--bootstrap-resamples", "50",
        ])


def test_comparison_null_does_not_alter_primary_computation(tmp_path):
    """Section-1 invariant: thresholds['t'], criterion_met, zero_discoveries,
    note, the FWER thresholds, threshold_sweep.csv, global_rule_baseline, and
    error_profile['description'] must be byte-identical with vs without
    --comparison-null-run-dir -- the flag only ever ADDS
    error_profile['comparison'] and error_profile_curve.csv's
    'comparison_fdp_mean' column.
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=600, R=15, plant=True)
    comp_run_dir = _build_comparison_null_dir(tmp_path, "comp_null_inv", n=600, R=6, seed=404)

    common_args = [
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--target-fdr", "0.20", "--fdp-quantile", "0.95", "--conditioning", "none",
        "--operating-point", "pct:0.02", "--bootstrap-resamples", "50",
        "--error-profile-calls", "5,20,50",
    ]
    out_without = tmp_path / "calib_without_comp"
    rc_without = cst.main(common_args + ["--out-dir", str(out_without)])
    out_with = tmp_path / "calib_with_comp"
    rc_with = cst.main(
        common_args + ["--out-dir", str(out_with), "--comparison-null-run-dir", str(comp_run_dir)]
    )
    assert rc_without == 0
    assert rc_with == 0

    t_without = json.loads((out_without / "thresholds.json").read_text())
    t_with = json.loads((out_with / "thresholds.json").read_text())

    for key in (
        "t", "O_t", "E_fp", "E_fp_over_O", "fdp_quantile_achieved",
        "criterion_met", "zero_discoveries", "note",
        "fwer_per_position_threshold", "fwer_global_threshold",
        "global_rule_baseline",
    ):
        assert t_with[key] == t_without[key], f"{key} differs with vs without --comparison-null-run-dir"

    assert "comparison" not in t_without["error_profile"]
    assert "comparison" in t_with["error_profile"]
    ep_without = {k: v for k, v in t_without["error_profile"].items() if k != "comparison"}
    ep_with = {k: v for k, v in t_with["error_profile"].items() if k != "comparison"}
    assert ep_without == ep_with

    sweep_without = pd.read_csv(out_without / "threshold_sweep.csv")
    sweep_with = pd.read_csv(out_with / "threshold_sweep.csv")
    pd.testing.assert_frame_equal(sweep_without, sweep_with)

    curve_without = pd.read_csv(out_without / "error_profile_curve.csv")
    curve_with = pd.read_csv(out_with / "error_profile_curve.csv")
    assert "comparison_fdp_mean" not in curve_without.columns
    assert "comparison_fdp_mean" in curve_with.columns
    pd.testing.assert_frame_equal(
        curve_without, curve_with.drop(columns=["comparison_fdp_mean"])
    )


def test_thresholds_json_serializable_with_comparison_null(tmp_path):
    """Guards against numpy types leaking into thresholds.json via the new
    error_profile['comparison'] block (mirrors
    test_thresholds_json_fully_serializable_with_operating_point).
    """
    obs_path, run_dir = _build_fixture(tmp_path, n=300, R=15, plant=False)
    comp_run_dir = _build_comparison_null_dir(tmp_path, "comp_null_json", n=300, R=5, seed=505)
    out_dir = tmp_path / "calib_comp_json"
    rc = cst.main([
        "--null-run-dir", str(run_dir), "--observed-scores", str(obs_path),
        "--out-dir", str(out_dir), "--target-fdr", "0.01", "--fdp-quantile", "0.95",
        "--operating-point", "calls:20",
        "--comparison-null-run-dir", str(comp_run_dir),
        "--error-profile-calls", "5,20,50",
        "--bootstrap-resamples", "50",
    ])
    assert rc == 0
    parsed = json.loads((out_dir / "thresholds.json").read_text())
    _assert_json_primitive(parsed)
    assert "comparison" in parsed["error_profile"]
