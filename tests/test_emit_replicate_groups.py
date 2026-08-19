"""Tests for scripts/emit_replicate_groups.py.

Uses tiny synthetic .npz files (this script never reads their array
contents -- only filenames and sibling provenance files matter -- so the
fixture files just need to exist) laid out under tmp_path to exercise each
of the four resolution routes described in the module docstring, plus the
recursive run_manifest.json route, the missing-argument failure of route 4,
--dry-run, and the CSV/sidecar output shape.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import emit_replicate_groups as erg


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dummy=np.zeros(1, dtype=np.float32))


def _make_run_dir(root: Path, name: str, n_replicates: int) -> Path:
    run_dir = root / name
    for i in range(n_replicates):
        _write_npz(run_dir / "npz" / f"rep_{i:04d}.npz")
    return run_dir


def _read_csv(path: Path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _read_sidecar(out_csv: Path) -> dict:
    sidecar_path = out_csv.with_suffix(out_csv.suffix + ".provenance.json")
    return json.loads(sidecar_path.read_text())


# ---------------------------------------------------------------------------
# Route 2: stage_null_replicate_set.py's provenance.json
# ---------------------------------------------------------------------------


def test_route2_provenance_json(tmp_path):
    run_dir = _make_run_dir(tmp_path, "staged_run", 2)
    provenance = {
        "replicates": [
            {"rep_id": "rep_0000", "invocation": "5"},
            {"rep_id": "rep_0001", "invocation": "6"},
        ]
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance))
    out_csv = tmp_path / "out" / "groups.csv"

    rc = erg.main(["--run-dir", str(run_dir), "--out-csv", str(out_csv)])
    assert rc == 0

    rows = _read_csv(out_csv)
    by_file = {r["rep_file"]: r["invocation"] for r in rows}
    assert by_file["rep_0000.npz"] == "staged_run:5"
    assert by_file["rep_0001.npz"] == "staged_run:6"

    sidecar = _read_sidecar(out_csv)
    assert sidecar["n_replicates"] == 2
    assert sidecar["n_invocations"] == 2
    assert "provenance.json" in sidecar["routes_by_source"]["staged_run"]


# ---------------------------------------------------------------------------
# Route 3: run_null_calibration.py's logs/progress.jsonl
# ---------------------------------------------------------------------------


def test_route3_progress_jsonl(tmp_path):
    run_dir = _make_run_dir(tmp_path, "progress_run", 3)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    lines = [
        json.dumps({"replicate": 0, "batch": 2}),
        json.dumps({"replicate": 1, "batch": 2}),
        json.dumps({"replicate": 2, "batch": 3}),
    ]
    (logs_dir / "progress.jsonl").write_text("\n".join(lines) + "\n")
    out_csv = tmp_path / "out" / "groups.csv"

    rc = erg.main(["--run-dir", str(run_dir), "--out-csv", str(out_csv)])
    assert rc == 0

    rows = _read_csv(out_csv)
    by_file = {r["rep_file"]: r["invocation"] for r in rows}
    assert by_file["rep_0000.npz"] == "progress_run:2"
    assert by_file["rep_0001.npz"] == "progress_run:2"
    assert by_file["rep_0002.npz"] == "progress_run:3"

    sidecar = _read_sidecar(out_csv)
    assert "progress.jsonl" in sidecar["routes_by_source"]["progress_run"]


# ---------------------------------------------------------------------------
# Route 4: index arithmetic
# ---------------------------------------------------------------------------


def test_route4_index_arithmetic(tmp_path):
    run_dir = _make_run_dir(tmp_path, "raw_run", 5)  # rep_0000..rep_0004
    out_csv = tmp_path / "out" / "groups.csv"

    rc = erg.main([
        "--run-dir", str(run_dir),
        "--out-csv", str(out_csv),
        "--alignments-per-invocation", "2",
    ])
    assert rc == 0

    rows = _read_csv(out_csv)
    by_file = {r["rep_file"]: r["invocation"] for r in rows}
    # index // 2: 0,1 -> 0 ; 2,3 -> 1 ; 4 -> 2
    assert by_file["rep_0000.npz"] == "raw_run:0"
    assert by_file["rep_0001.npz"] == "raw_run:0"
    assert by_file["rep_0002.npz"] == "raw_run:1"
    assert by_file["rep_0003.npz"] == "raw_run:1"
    assert by_file["rep_0004.npz"] == "raw_run:2"

    sidecar = _read_sidecar(out_csv)
    assert sidecar["n_invocations"] == 3
    assert "index arithmetic" in sidecar["routes_by_source"]["raw_run"]


def test_route4_missing_alignments_per_invocation_exits_cleanly(tmp_path):
    run_dir = _make_run_dir(tmp_path, "raw_run", 3)
    out_csv = tmp_path / "out" / "groups.csv"

    with pytest.raises(SystemExit) as excinfo:
        erg.main(["--run-dir", str(run_dir), "--out-csv", str(out_csv)])

    message = str(excinfo.value)
    assert "--alignments-per-invocation" in message
    assert "run_score_null_calibration_array.sh" in message
    assert not out_csv.exists()


# ---------------------------------------------------------------------------
# Route 1: merge_null_runs.py's run_manifest.json, recursive
# ---------------------------------------------------------------------------


def test_route1_recursive_manifest_produces_prefixed_labels(tmp_path):
    # source_a resolves via route 4 (arithmetic); source_b resolves via
    # route 2 (provenance.json). Both are pooled into `merged_run`.
    source_a = _make_run_dir(tmp_path, "source_a", 4)  # rep_0000..rep_0003
    source_b = _make_run_dir(tmp_path, "source_b", 2)  # rep_0000..rep_0001
    (source_b / "provenance.json").write_text(json.dumps({
        "replicates": [
            {"rep_id": "rep_0000", "invocation": "7"},
            {"rep_id": "rep_0001", "invocation": "8"},
        ]
    }))

    merged_run = _make_run_dir(tmp_path, "merged_run", 6)  # rep_0000..rep_0005
    manifest_replicates = [
        {"merged_index": 0, "source": str(source_a / "npz" / "rep_0000.npz")},
        {"merged_index": 1, "source": str(source_a / "npz" / "rep_0001.npz")},
        {"merged_index": 2, "source": str(source_a / "npz" / "rep_0002.npz")},
        {"merged_index": 3, "source": str(source_a / "npz" / "rep_0003.npz")},
        {"merged_index": 4, "source": str(source_b / "npz" / "rep_0000.npz")},
        {"merged_index": 5, "source": str(source_b / "npz" / "rep_0001.npz")},
    ]
    (merged_run / "run_manifest.json").write_text(json.dumps({"replicates": manifest_replicates}))

    out_csv = tmp_path / "out" / "groups.csv"
    rc = erg.main([
        "--run-dir", str(merged_run),
        "--out-csv", str(out_csv),
        "--alignments-per-invocation", "2",
    ])
    assert rc == 0

    rows = _read_csv(out_csv)
    assert len(rows) == 6
    by_file = {r["rep_file"]: r["invocation"] for r in rows}

    # source_a: index // 2 -> 0,0,1,1 (2 distinct), each prefixed with
    # source_a's own name, then the top-level merged_run prefix.
    assert by_file["rep_0000.npz"] == "merged_run:source_a:0"
    assert by_file["rep_0001.npz"] == "merged_run:source_a:0"
    assert by_file["rep_0002.npz"] == "merged_run:source_a:1"
    assert by_file["rep_0003.npz"] == "merged_run:source_a:1"
    # source_b: recorded invocations 7, 8.
    assert by_file["rep_0004.npz"] == "merged_run:source_b:7"
    assert by_file["rep_0005.npz"] == "merged_run:source_b:8"

    distinct = set(by_file.values())
    assert len(distinct) == 4  # 2 from source_a + 2 from source_b, none colliding

    sidecar = _read_sidecar(out_csv)
    assert sidecar["n_replicates"] == 6
    assert sidecar["n_invocations"] == 4
    assert "index arithmetic" in sidecar["routes_by_source"]["source_a"]
    assert "provenance.json" in sidecar["routes_by_source"]["source_b"]


def test_route1_pooled_like_group_counts(tmp_path):
    """Mirrors the shape of the real pooled_41 set this script was written
    for: one source resolved by arithmetic into several unevenly-sized
    invocations, another source with its own recorded invocations, pooled
    by a run_manifest.json. Checks the group-count arithmetic specifically
    (sizes 9,10,10,2 -> 4 groups from one source; 10 recorded invocations
    from the other -> 14 groups total), on a smaller scale (31 -> 13
    replicates, alignments_per_invocation=3 instead of 10) so the test
    stays fast while covering the same "uneven last chunk" edge case.
    """
    # 13 replicates missing index 1 (mirrors euler_run's missing rep_0001),
    # alignments_per_invocation=3 -> chunks of size 2,3,3,3,2 -> 5 groups.
    indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    source_a = tmp_path / "source_a"
    for i in indices:
        _write_npz(source_a / "npz" / f"rep_{i:04d}.npz")

    source_b = _make_run_dir(tmp_path, "source_b", 3)
    (source_b / "provenance.json").write_text(json.dumps({
        "replicates": [
            {"rep_id": "rep_0000", "invocation": "20"},
            {"rep_id": "rep_0001", "invocation": "21"},
            {"rep_id": "rep_0002", "invocation": "22"},
        ]
    }))

    merged_run = tmp_path / "merged_run"
    manifest_replicates = []
    merged_idx = 0
    for i in indices:
        _write_npz(merged_run / "npz" / f"rep_{merged_idx:04d}.npz")
        manifest_replicates.append({
            "merged_index": merged_idx,
            "source": str(source_a / "npz" / f"rep_{i:04d}.npz"),
        })
        merged_idx += 1
    for i in range(3):
        _write_npz(merged_run / "npz" / f"rep_{merged_idx:04d}.npz")
        manifest_replicates.append({
            "merged_index": merged_idx,
            "source": str(source_b / "npz" / f"rep_{i:04d}.npz"),
        })
        merged_idx += 1
    (merged_run / "run_manifest.json").write_text(json.dumps({"replicates": manifest_replicates}))

    out_csv = tmp_path / "out" / "groups.csv"
    rc = erg.main([
        "--run-dir", str(merged_run),
        "--out-csv", str(out_csv),
        "--alignments-per-invocation", "3",
    ])
    assert rc == 0

    rows = _read_csv(out_csv)
    assert len(rows) == 15  # 12 + 3

    sidecar = _read_sidecar(out_csv)
    assert sidecar["n_replicates"] == 15
    # source_a's 12 replicates (indices 0,2..12) split by //3 into chunks
    # {0,1,2,3,4} = 5 groups; source_b contributes 3 recorded invocations.
    assert sidecar["n_invocations"] == 8


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path, capsys):
    run_dir = _make_run_dir(tmp_path, "raw_run", 3)
    out_csv = tmp_path / "out" / "groups.csv"

    rc = erg.main([
        "--run-dir", str(run_dir),
        "--out-csv", str(out_csv),
        "--alignments-per-invocation", "2",
        "--dry-run",
    ])
    assert rc == 0

    assert not out_csv.exists()
    assert not out_csv.parent.exists() or not any(out_csv.parent.iterdir())
    sidecar_path = out_csv.with_suffix(out_csv.suffix + ".provenance.json")
    assert not sidecar_path.exists()

    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert "rep_0000.npz" in captured.out


# ---------------------------------------------------------------------------
# CSV header and basename form
# ---------------------------------------------------------------------------


def test_csv_header_and_basename_form(tmp_path):
    run_dir = _make_run_dir(tmp_path, "raw_run", 2)
    out_csv = tmp_path / "out" / "groups.csv"

    rc = erg.main([
        "--run-dir", str(run_dir),
        "--out-csv", str(out_csv),
        "--alignments-per-invocation", "1",
    ])
    assert rc == 0

    with open(out_csv, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == ["rep_file", "invocation"]
        data_rows = list(reader)

    rep_files = [row[0] for row in data_rows]
    # basenames only, never a path (npz/ dir stripped, no directory separators)
    for name in rep_files:
        assert name == Path(name).name
        assert "/" not in name
    assert set(rep_files) == {"rep_0000.npz", "rep_0001.npz"}


# ---------------------------------------------------------------------------
# --invocation-prefix override
# ---------------------------------------------------------------------------


def test_invocation_prefix_override(tmp_path):
    run_dir = _make_run_dir(tmp_path, "raw_run", 2)
    out_csv = tmp_path / "out" / "groups.csv"

    rc = erg.main([
        "--run-dir", str(run_dir),
        "--out-csv", str(out_csv),
        "--alignments-per-invocation", "1",
        "--invocation-prefix", "custom_prefix",
    ])
    assert rc == 0

    rows = _read_csv(out_csv)
    for row in rows:
        assert row["invocation"].startswith("custom_prefix:")
