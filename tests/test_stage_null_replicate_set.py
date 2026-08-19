"""Tests for scripts/stage_null_replicate_set.py.

Uses tiny synthetic .npz files (a handful of rows, all seven SCORE_COLUMNS
arrays) rather than real ASR/scoring output, mirroring the fixture style in
tests/test_calibrate_switch_threshold.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import stage_null_replicate_set as srs
from scripts.score_null_replicate import SCORE_COLUMNS

N_ROWS = 6


def _write_npz(path: Path, n: int = N_ROWS, rng=None) -> None:
    rng = rng or np.random.default_rng(0)
    payload = {c: rng.normal(size=n).astype(np.float32) for c in SCORE_COLUMNS}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _write_source_batch(source_dir: Path, invocations, n: int = N_ROWS) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for c in invocations:
        _write_npz(source_dir / f"tl_chunk{c}_sim1.npz", n=n, rng=rng)


PATTERN = r"tl_chunk(?P<invocation>\d+)_sim1\.npz"


def test_index_assignment_order(tmp_path):
    source_dir = tmp_path / "source"
    # Deliberately out of order and non-zero-padded, to check numeric sort.
    _write_source_batch(source_dir, [10, 5, 9, 11])
    out_dir = tmp_path / "out"

    rc = srs.main([
        "--source-dir", str(source_dir),
        "--pattern", PATTERN,
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    npz_dir = out_dir / "npz"
    staged = sorted(p.name for p in npz_dir.glob("rep_*.npz"))
    assert staged == ["rep_0000.npz", "rep_0001.npz", "rep_0002.npz", "rep_0003.npz"]

    provenance = json.loads((out_dir / "provenance.json").read_text())
    ordered_invocations = [r["invocation"] for r in provenance["replicates"]]
    assert ordered_invocations == ["5", "9", "10", "11"]
    ordered_rep_ids = [r["rep_id"] for r in provenance["replicates"]]
    assert ordered_rep_ids == ["rep_0000", "rep_0001", "rep_0002", "rep_0003"]


def test_provenance_matches_staged_files(tmp_path):
    source_dir = tmp_path / "source"
    _write_source_batch(source_dir, [5, 6])
    out_dir = tmp_path / "out"

    rc = srs.main([
        "--source-dir", str(source_dir),
        "--pattern", PATTERN,
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["n_replicates"] == 2
    assert provenance["n_tests"] == N_ROWS
    assert provenance["source_dir"] == str(source_dir)
    assert provenance["pattern"] == PATTERN
    assert "created" in provenance
    assert "git_sha" in provenance
    assert "argv" in provenance

    for entry in provenance["replicates"]:
        staged_path = out_dir / "npz" / f"{entry['rep_id']}.npz"
        assert staged_path.exists()
        source_path = source_dir / entry["source_file"]
        assert srs.sha256_of(source_path) == entry["source_sha256"]
        # staged file is a copy, so it should hash identically too
        assert srs.sha256_of(staged_path) == entry["source_sha256"]


def test_missing_array_causes_hard_refusal_and_stages_nothing(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_npz(source_dir / "tl_chunk5_sim1.npz")

    # A second file missing one required array.
    bad_payload = {c: np.zeros(N_ROWS, dtype=np.float32) for c in SCORE_COLUMNS if c != "ac"}
    np.savez_compressed(source_dir / "tl_chunk6_sim1.npz", **bad_payload)

    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="missing array"):
        srs.main([
            "--source-dir", str(source_dir),
            "--pattern", PATTERN,
            "--out-dir", str(out_dir),
        ])

    assert not (out_dir / "npz").exists()
    assert not (out_dir / "provenance.json").exists()


def test_length_mismatch_across_files_is_refused(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_npz(source_dir / "tl_chunk5_sim1.npz", n=N_ROWS)
    _write_npz(source_dir / "tl_chunk6_sim1.npz", n=N_ROWS + 1)

    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match="disagree on test count"):
        srs.main([
            "--source-dir", str(source_dir),
            "--pattern", PATTERN,
            "--out-dir", str(out_dir),
        ])
    assert not (out_dir / "npz").exists()


def test_non_matching_filenames_are_skipped_and_reported(tmp_path):
    source_dir = tmp_path / "source"
    _write_source_batch(source_dir, [5, 6])
    _write_npz(source_dir / "unrelated_file.npz")

    out_dir = tmp_path / "out"
    with pytest.warns(UserWarning, match="unrelated_file.npz"):
        rc = srs.main([
            "--source-dir", str(source_dir),
            "--pattern", PATTERN,
            "--out-dir", str(out_dir),
        ])
    assert rc == 0
    staged = sorted(p.name for p in (out_dir / "npz").glob("rep_*.npz"))
    assert len(staged) == 2


def test_force_semantics(tmp_path):
    source_dir = tmp_path / "source"
    _write_source_batch(source_dir, [5, 6])
    out_dir = tmp_path / "out"

    rc = srs.main([
        "--source-dir", str(source_dir),
        "--pattern", PATTERN,
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    # Without --force, re-running against the now-non-empty npz/ refuses.
    with pytest.raises(SystemExit, match="non-empty"):
        srs.main([
            "--source-dir", str(source_dir),
            "--pattern", PATTERN,
            "--out-dir", str(out_dir),
        ])

    # With --force, it proceeds.
    rc = srs.main([
        "--source-dir", str(source_dir),
        "--pattern", PATTERN,
        "--out-dir", str(out_dir),
        "--force",
    ])
    assert rc == 0
    staged = sorted(p.name for p in (out_dir / "npz").glob("rep_*.npz"))
    assert len(staged) == 2


def test_dry_run_writes_nothing(tmp_path, capsys):
    source_dir = tmp_path / "source"
    _write_source_batch(source_dir, [5, 6])
    out_dir = tmp_path / "out"

    rc = srs.main([
        "--source-dir", str(source_dir),
        "--pattern", PATTERN,
        "--out-dir", str(out_dir),
        "--dry-run",
    ])
    assert rc == 0
    assert not out_dir.exists()
    captured = capsys.readouterr()
    assert "dry-run" in captured.out


def test_seed_formula_for_known_invocation():
    seed_base = 20260731
    seed_stride = 100003
    # invocation "7" -> seed_base + 7 * seed_stride, per the convention
    # documented at scripts/package_null_calibration_for_euler.py.
    expected = seed_base + 7 * seed_stride
    assert srs.simulate_seed_for_invocation("7", seed_base, seed_stride) == expected
    assert srs.simulate_seed_for_invocation("not_a_number", seed_base, seed_stride) is None


def test_pattern_without_invocation_group_rejected(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit):
        srs.main([
            "--source-dir", str(source_dir),
            "--pattern", r"tl_chunk(?P<other>\d+)_sim1\.npz",
            "--out-dir", str(out_dir),
        ])
