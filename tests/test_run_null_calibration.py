"""Tests for scripts/run_null_calibration.py.

The simulate/score subprocess stages are stubbed with small fake scripts
(fixtures below) so these tests exercise the *driver's* batching,
resumability, cleanup and failure-handling logic without needing IQ-TREE or
any real alignment/tree data. End-to-end validation against the real
scripts, on a small pruned subset, was done manually (see the task report);
that is not repeated here since it takes tens of seconds per replicate.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from scripts import run_null_calibration as rnc
from scripts.score_null_replicate import SCORE_COLUMNS

FAKE_SIMULATE = textwrap.dedent(
    """
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--num-alignments", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args, _ = p.parse_known_args()

    n = args.num_alignments
    for k in range(1, n + 1):
        out = Path(f"{args.out_prefix}.fa") if n == 1 else Path(f"{args.out_prefix}_{k}.fa")
        out.write_text(f">taxon_a\\nAAAA\\n>taxon_b\\nAAAA\\n")
    print(f"[fake-simulate] wrote {n} alignment(s), seed={args.seed}")
    """
)

# The fake score script:
#  * derives the expected test count from --observed-scores (a tiny 2-row CSV
#    the tests write),
#  * writes a valid SCORE_COLUMNS .npz,
#  * prints the same "[keys] ..." / "[asr] ..." / "[score] ..." lines the
#    real score_null_replicate.py prints, so parse_key_check() is exercised
#    against realistic text,
#  * fails (nonzero exit) if its --workdir's replicate index (parsed from the
#    driver's own "rep_<i>_work" naming) is listed in $FAKE_FAIL_REPLICATES.
FAKE_SCORE = textwrap.dedent(
    """
    import argparse
    import os
    import re
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--sim-alignment", type=Path, required=True)
    p.add_argument("--observed-scores", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    args, _ = p.parse_known_args()

    m = re.search(r"rep_(\\d+)_work", args.workdir.name)
    rep_id = int(m.group(1)) if m else None
    fail_set = {int(x) for x in os.environ.get("FAKE_FAIL_REPLICATES", "").split(",") if x}
    if rep_id in fail_set:
        print("[fake-score] forced failure", file=sys.stderr)
        sys.exit(7)

    n = sum(1 for _ in open(args.observed_scores)) - 1
    print(f"[asr] 0.1s -> fake.state (1 MB)")
    payload = {c: np.zeros(n, dtype=np.float32) for c in
               ("rc_left", "rc_right", "ac", "p_ac_left", "p_ac_right",
                "badasp_score_left", "badasp_score_right")}
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **payload)
    print(f"[score] 0.1s -> fake_scores.csv")
    print(f"[keys] observed={n:,} null={n:,} missing_from_null=0 (0.0000%) null_only_dropped=0")
    print(f"[keys] clade sizes identical on all {n:,} shared tests")
    """
)


@pytest.fixture()
def fake_scripts(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_scripts")
    sim = d / "fake_simulate.py"
    score = d / "fake_score.py"
    sim.write_text(FAKE_SIMULATE)
    score.write_text(FAKE_SCORE)
    return sim, score


@pytest.fixture()
def observed_scores_csv(tmp_path):
    path = tmp_path / "observed_scores.csv"
    path.write_text(
        "node_name,left_child,right_child,position,badasp_score_left,badasp_score_right\n"
        "N1,N2,N3,1,0.1,0.2\n"
        "N1,N2,N3,2,0.3,0.4\n"
        "N4,N5,N6,1,0.5,0.6\n"
    )
    return path


def _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch, **overrides):
    sim, score = fake_scripts
    monkeypatch.setattr(rnc, "SIMULATE_SCRIPT", sim)
    monkeypatch.setattr(rnc, "SCORE_SCRIPT", score)

    align = tmp_path / "align.fasta"
    align.write_text(">a\nAAAA\n")
    tree = tmp_path / "tree.nwk"
    tree.write_text("(a:0.1,b:0.1);\n")

    out_dir = overrides.pop("out_dir", tmp_path / "out")
    argv = [
        "--num-replicates", str(overrides.pop("num_replicates", 3)),
        "--batch-size", str(overrides.pop("batch_size", 2)),
        "--concurrency", str(overrides.pop("concurrency", 2)),
        "--out-dir", str(out_dir),
        "--seed", str(overrides.pop("seed", 42)),
        "--composition-alignment", str(align),
        "--sim-tree", str(tree),
        "--reconciled-tree", str(tree),
        "--asr-tree", str(tree),
        "--observed-scores", str(observed_scores_csv),
        "--reference-mc-length", "10",
    ]
    for k, v in overrides.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    rc = rnc.main(argv)
    return rc, out_dir


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_alisim_output_path_no_suffix_when_batch_size_one(tmp_path):
    prefix = tmp_path / "sim"
    assert rnc.alisim_output_path(prefix, 1, 1) == Path(f"{prefix}.fa")


def test_alisim_output_path_numbered_when_batch_size_gt_one():
    prefix = Path("/x/sim")
    assert rnc.alisim_output_path(prefix, 1, 3) == Path("/x/sim_1.fa")
    assert rnc.alisim_output_path(prefix, 3, 3) == Path("/x/sim_3.fa")


def test_batch_ranges_partitions_evenly_and_with_remainder():
    assert rnc.batch_ranges(6, 2) == [(0, 2), (2, 4), (4, 6)]
    assert rnc.batch_ranges(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert rnc.batch_ranges(0, 2) == []


def test_parse_key_check_extracts_all_fields():
    stdout = (
        "[asr] 313.2s -> asr.state (600 MB)\n"
        "[score] 133.4s -> scores.csv\n"
        "[keys] observed=21,691 null=21,650 missing_from_null=41 (0.1890%) null_only_dropped=12\n"
        "[keys] clade sizes identical on all 21,650 shared tests\n"
    )
    parsed = rnc.parse_key_check(stdout)
    assert parsed["observed_tests"] == 21691
    assert parsed["null_tests"] == 21650
    assert parsed["missing_from_null"] == 41
    assert parsed["missing_frac_pct"] == pytest.approx(0.1890)
    assert parsed["null_only_dropped"] == 12
    assert parsed["clade_sizes_identical"] == 21650
    assert parsed["asr_seconds"] == pytest.approx(313.2)
    assert parsed["score_seconds"] == pytest.approx(133.4)


def test_parse_key_check_missing_lines_are_none():
    parsed = rnc.parse_key_check("nothing relevant here\n")
    assert all(v is None for v in parsed.values())


def test_is_valid_npz_missing_file(tmp_path):
    assert rnc.is_valid_npz(tmp_path / "does_not_exist.npz") is False


def test_is_valid_npz_corrupted_file(tmp_path):
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not a real npz file")
    assert rnc.is_valid_npz(path) is False


def test_is_valid_npz_valid_file(tmp_path):
    path = tmp_path / "good.npz"
    payload = {c: np.zeros(5, dtype=np.float32) for c in SCORE_COLUMNS}
    np.savez_compressed(path, **payload)
    assert rnc.is_valid_npz(path) is True
    assert rnc.is_valid_npz(path, expected_len=5) is True
    assert rnc.is_valid_npz(path, expected_len=6) is False


def test_is_valid_npz_missing_required_array(tmp_path):
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, rc_left=np.zeros(5, dtype=np.float32))
    assert rnc.is_valid_npz(path) is False


def test_count_csv_data_rows(observed_scores_csv):
    assert rnc.count_csv_data_rows(observed_scores_csv) == 3


def test_file_info_reports_size_and_mtime(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    info = rnc.file_info(f)
    assert info["exists"] is True
    assert info["size_bytes"] == 5
    assert "mtime" in info
    assert rnc.file_info(tmp_path / "missing.txt")["exists"] is False
    assert rnc.file_info(None) is None


# ---------------------------------------------------------------------------
# End-to-end driver behaviour, against the fake simulate/score scripts
# ---------------------------------------------------------------------------


def test_fresh_run_completes_all_replicates(tmp_path, fake_scripts, observed_scores_csv, monkeypatch):
    rc, out_dir = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                        num_replicates=3, batch_size=2)
    assert rc == 0
    npz_files = sorted((out_dir / "npz").glob("rep_*.npz"))
    assert len(npz_files) == 3
    for f in npz_files:
        assert rnc.is_valid_npz(f, expected_len=3)
    # scratch is cleaned up after a successful run
    assert list((out_dir / "scratch").iterdir()) == []
    # manifest records the inputs actually used
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["num_replicates"] == 3
    assert manifest["inputs"]["observed_scores"]["exists"] is True


def test_resume_skips_completed_replicates(tmp_path, fake_scripts, observed_scores_csv, monkeypatch):
    rc, out_dir = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                        num_replicates=3, batch_size=2)
    assert rc == 0
    mtimes_before = {f.name: f.stat().st_mtime_ns for f in (out_dir / "npz").glob("rep_*.npz")}

    rc2, out_dir2 = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                         num_replicates=3, batch_size=2, out_dir=out_dir)
    assert rc2 == 0
    mtimes_after = {f.name: f.stat().st_mtime_ns for f in (out_dir / "npz").glob("rep_*.npz")}
    # Every .npz is untouched by the resumed run (skipped, not rewritten).
    assert mtimes_before == mtimes_after

    progress_lines = (out_dir / "logs" / "progress.jsonl").read_text().strip().splitlines()
    assert len(progress_lines) == 3  # only the first run's completions were logged


def test_corrupted_npz_is_redone_others_left_alone(tmp_path, fake_scripts, observed_scores_csv, monkeypatch):
    rc, out_dir = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                        num_replicates=3, batch_size=2)
    assert rc == 0

    target = out_dir / "npz" / "rep_0001.npz"
    other = out_dir / "npz" / "rep_0000.npz"
    mtime_other_before = other.stat().st_mtime_ns
    target.write_bytes(b"corrupted, not a valid npz")

    rc2, _ = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                  num_replicates=3, batch_size=2, out_dir=out_dir)
    assert rc2 == 0
    assert rnc.is_valid_npz(target, expected_len=3)
    assert other.stat().st_mtime_ns == mtime_other_before  # untouched

    progress_lines = (out_dir / "logs" / "progress.jsonl").read_text().strip().splitlines()
    # 3 from the fresh run + 1 for the single redone replicate
    assert len(progress_lines) == 4
    last = json.loads(progress_lines[-1])
    assert last["replicate"] == 1
    assert last["status"] == "OK"


def test_replicate_failure_is_recorded_and_run_continues(
    tmp_path, fake_scripts, observed_scores_csv, monkeypatch
):
    monkeypatch.setenv("FAKE_FAIL_REPLICATES", "1")
    rc, out_dir = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                        num_replicates=3, batch_size=3, concurrency=1)
    assert rc == 1  # non-zero because a replicate failed

    npz_files = sorted((out_dir / "npz").glob("rep_*.npz"))
    assert len(npz_files) == 2  # replicates 0 and 2 still completed
    assert not (out_dir / "npz" / "rep_0001.npz").exists()

    records = [json.loads(l) for l in (out_dir / "logs" / "progress.jsonl").read_text().splitlines()]
    by_rep = {r["replicate"]: r for r in records}
    assert by_rep[0]["status"] == "OK"
    assert by_rep[1]["status"] == "FAILED"
    assert by_rep[1]["stage_failed"] == "score"
    assert by_rep[2]["status"] == "OK"
    # the failed replicate's scratch workdir must not survive the run
    assert not any(p.name.startswith("rep_0001") for p in (out_dir / "scratch").iterdir())


def test_whole_batch_simulate_failure_marks_every_replicate_in_batch(
    tmp_path, observed_scores_csv, monkeypatch
):
    failing_sim = tmp_path / "failing_simulate.py"
    failing_sim.write_text("import sys\nsys.exit(3)\n")
    score = tmp_path / "unused_score.py"
    score.write_text("import sys\nsys.exit(0)\n")
    rc, out_dir = _run(tmp_path, (failing_sim, score), observed_scores_csv, monkeypatch,
                        num_replicates=2, batch_size=2)
    assert rc == 1
    records = [json.loads(l) for l in (out_dir / "logs" / "progress.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(r["status"] == "FAILED" and r["stage_failed"] == "simulate" for r in records)
    assert list((out_dir / "npz").glob("rep_*.npz")) == []


def test_force_flag_redoes_already_valid_replicates(tmp_path, fake_scripts, observed_scores_csv, monkeypatch):
    rc, out_dir = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                        num_replicates=2, batch_size=2)
    assert rc == 0
    mtimes_before = {f.name: f.stat().st_mtime_ns for f in (out_dir / "npz").glob("rep_*.npz")}

    rc2, _ = _run(tmp_path, fake_scripts, observed_scores_csv, monkeypatch,
                  num_replicates=2, batch_size=2, out_dir=out_dir, force=True)
    assert rc2 == 0
    mtimes_after = {f.name: f.stat().st_mtime_ns for f in (out_dir / "npz").glob("rep_*.npz")}
    assert mtimes_before != mtimes_after


# ---------------------------------------------------------------------------
# simulate_null_persite.py --help regression (the documented known bug fix)
# ---------------------------------------------------------------------------


def test_simulate_null_persite_help_lists_all_options():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "simulate_null_persite.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    for expected in ("--num-alignments", "--seed", "--shrinkage", "--reference-model",
                     "--gap-mask-source", "--workdir", "--redo"):
        assert expected in result.stdout, f"{expected} missing from --help output"
