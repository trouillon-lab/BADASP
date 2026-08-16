"""Tests for scripts/merge_null_runs.py.

The failure this guards against is silent: replicates carry no key arrays,
only row-aligned score vectors, so merging two runs that were aligned to
different observed tables misaligns every test without any error.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import merge_null_runs as mnr
from scripts.score_null_replicate import SCORE_COLUMNS


def _make_run(root: Path, name: str, n_reps: int, n_tests: int,
              seed=1, shrinkage=0.15, observed="results/obs.csv") -> Path:
    run_dir = root / name
    npz_dir = run_dir / "npz"
    npz_dir.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for i in range(n_reps):
        np.savez(
            npz_dir / f"rep_{i:04d}.npz",
            **{c: rng.normal(size=n_tests).astype(np.float32) for c in SCORE_COLUMNS},
        )
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "seed": seed,
        "shrinkage": shrinkage,
        "inputs": {"observed_scores": {"path": observed}},
    }))
    return run_dir


def test_merge_produces_contiguous_numbering(tmp_path):
    a = _make_run(tmp_path, "a", 3, 50, seed=1)
    b = _make_run(tmp_path, "b", 2, 50, seed=2)
    out = tmp_path / "merged"
    assert mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(out)]) == 0

    files = sorted((out / "npz").glob("rep_*.npz"))
    assert [f.name for f in files] == [f"rep_{i:04d}.npz" for i in range(5)]
    # Every merged replicate must still load with the expected shape.
    for f in files:
        with np.load(f) as data:
            assert data["badasp_score_left"].shape == (50,)


def test_merge_refuses_mismatched_test_counts(tmp_path):
    a = _make_run(tmp_path, "a", 2, 50)
    b = _make_run(tmp_path, "b", 2, 51)
    with pytest.raises(SystemExit, match="disagree on the number of tests"):
        mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(tmp_path / "m")])


def test_merge_refuses_different_observed_tables(tmp_path):
    a = _make_run(tmp_path, "a", 2, 50, observed="results/obs_old.csv")
    b = _make_run(tmp_path, "b", 2, 50, observed="results/obs_new.csv")
    with pytest.raises(SystemExit, match="observed-scores paths differ"):
        mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(tmp_path / "m")])


def test_merge_refuses_different_shrinkage(tmp_path):
    a = _make_run(tmp_path, "a", 2, 50, shrinkage=0.15)
    b = _make_run(tmp_path, "b", 2, 50, shrinkage=0.30)
    with pytest.raises(SystemExit, match="shrinkage differs"):
        mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(tmp_path / "m")])


def test_mismatch_can_be_overridden_explicitly(tmp_path):
    a = _make_run(tmp_path, "a", 2, 50, observed="/abs/obs.csv")
    b = _make_run(tmp_path, "b", 2, 50, observed="obs.csv")
    out = tmp_path / "merged"
    assert mnr.main([
        "--run-dir", str(a), str(b), "--out-dir", str(out), "--allow-manifest-mismatch",
    ]) == 0
    assert len(list((out / "npz").glob("rep_*.npz"))) == 4


def test_repeated_seed_warns_but_proceeds(tmp_path, capsys):
    """Same seed means the same simulated alignments, so the extra
    replicates add no independent information."""
    a = _make_run(tmp_path, "a", 2, 50, seed=7)
    b = _make_run(tmp_path, "b", 2, 50, seed=7)
    assert mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(tmp_path / "m")]) == 0
    assert "repeated seed" in capsys.readouterr().err


def test_manifest_records_provenance_of_every_replicate(tmp_path):
    a = _make_run(tmp_path, "a", 3, 50, seed=1)
    b = _make_run(tmp_path, "b", 2, 50, seed=2)
    out = tmp_path / "merged"
    mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(out)])
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["n_replicates"] == 5
    assert manifest["n_tests"] == 50
    assert [m["n_replicates"] for m in manifest["merged_from"]] == [3, 2]
    assert len(manifest["replicates"]) == 5
    # Provenance must point back at the original files, not the merged names.
    assert all("/a/npz/" in r["source"] or "/b/npz/" in r["source"]
               for r in manifest["replicates"])
    # Carried forward so the calibration script's consistency check still works.
    assert manifest["inputs"]["observed_scores"]["path"] == "results/obs.csv"


def test_merge_accepts_an_npz_dir_directly(tmp_path):
    a = _make_run(tmp_path, "a", 2, 50, seed=1)
    b = _make_run(tmp_path, "b", 2, 50, seed=2)
    out = tmp_path / "merged"
    assert mnr.main([
        "--run-dir", str(a / "npz"), str(b / "npz"), "--out-dir", str(out),
    ]) == 0
    assert len(list((out / "npz").glob("rep_*.npz"))) == 4


def test_rerunning_a_merge_does_not_accumulate_stale_replicates(tmp_path):
    a = _make_run(tmp_path, "a", 3, 50, seed=1)
    b = _make_run(tmp_path, "b", 2, 50, seed=2)
    out = tmp_path / "merged"
    mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(out)])
    mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(out)])
    assert len(list((out / "npz").glob("rep_*.npz"))) == 5


def test_repeating_the_run_dir_flag_is_rejected_not_silently_dropped(tmp_path):
    """argparse's nargs='+' overwrites on a repeated flag rather than
    appending, so `--run-dir A --run-dir B` would quietly merge only B."""
    a = _make_run(tmp_path, "a", 2, 50, seed=1)
    b = _make_run(tmp_path, "b", 2, 50, seed=2)
    with pytest.raises(SystemExit):
        mnr.main(["--run-dir", str(a), "--run-dir", str(b), "--out-dir", str(tmp_path / "m")])


def _make_run_with_tail(root: Path, name: str, n_reps: int, n_tests: int, scale: float) -> Path:
    """A run whose null tail is controlled by `scale`, so two runs can be
    made genuinely non-exchangeable on purpose."""
    run_dir = root / name
    npz_dir = run_dir / "npz"
    npz_dir.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for i in range(n_reps):
        arrays = {c: rng.normal(size=n_tests).astype(np.float32) for c in SCORE_COLUMNS}
        arrays["badasp_score_left"] = rng.exponential(scale, size=n_tests).astype(np.float32)
        arrays["badasp_score_right"] = rng.exponential(scale, size=n_tests).astype(np.float32)
        np.savez(npz_dir / f"rep_{i:04d}.npz", **arrays)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "seed": abs(hash(name)) % 1000,
        "shrinkage": 0.15,
        "inputs": {"observed_scores": {"path": "results/obs.csv"}},
    }))
    return run_dir


def test_merge_refuses_runs_whose_null_tails_disagree(tmp_path):
    """The real incident: two runs agreed on every metadata field and still
    had null tails 6x apart, which would corrupt every error-rate estimate."""
    a = _make_run_with_tail(tmp_path, "cold", 6, 4000, scale=1.0)
    b = _make_run_with_tail(tmp_path, "hot", 6, 4000, scale=2.0)
    with pytest.raises(SystemExit, match="null tails differ"):
        mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(tmp_path / "m")])


def test_incomparable_tails_can_be_overridden_explicitly(tmp_path):
    a = _make_run_with_tail(tmp_path, "cold2", 6, 4000, scale=1.0)
    b = _make_run_with_tail(tmp_path, "hot2", 6, 4000, scale=2.0)
    out = tmp_path / "merged"
    assert mnr.main([
        "--run-dir", str(a), str(b), "--out-dir", str(out), "--allow-incomparable-tails",
    ]) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    # The override must still record that the tails disagreed.
    assert manifest["tail_comparability"]["comparable"] is False


def test_comparable_runs_merge_and_record_the_check(tmp_path):
    a = _make_run_with_tail(tmp_path, "same_a", 6, 4000, scale=1.0)
    b = _make_run_with_tail(tmp_path, "same_b", 6, 4000, scale=1.0)
    out = tmp_path / "merged"
    assert mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(out)]) == 0
    tails = json.loads((out / "run_manifest.json").read_text())["tail_comparability"]
    assert tails["comparable"] is True
    assert tails["mean_ratio"] < 1.5


def test_tail_check_reports_undetermined_rather_than_guessing(tmp_path):
    """With too few exceedances the ratio is Poisson noise (and can be
    infinite when one run has zero), so no verdict may be claimed."""
    a = _make_run(tmp_path, "tiny_a", 2, 50, seed=1)
    b = _make_run(tmp_path, "tiny_b", 2, 50, seed=2)
    out = tmp_path / "merged"
    assert mnr.main(["--run-dir", str(a), str(b), "--out-dir", str(out)]) == 0
    tails = json.loads((out / "run_manifest.json").read_text())["tail_comparability"]
    assert tails["comparable"] is None
    assert "too few" in tails["undetermined_reason"]
