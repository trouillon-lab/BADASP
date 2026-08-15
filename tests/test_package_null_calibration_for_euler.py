import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts.package_null_calibration_for_euler import (
    ARRAY_THROTTLE_DEFAULT,
    SCORE_ASR_MEASURED_MEM_GB,
    SCORE_LOADED_MINUTES_HIGH,
    SCORE_MEM_HEADROOM_GB,
    SCORE_THREADS,
    SCORE_WALLTIME_SAFETY_FACTOR,
    SIMULATE_MEASURED_MEM_GB,
    SIMULATE_MEASURED_SECONDS,
    SIMULATE_MEM_HEADROOM_GB,
    SIMULATE_WALLTIME_SAFETY_FACTOR,
    _fmt_hms,
    package_null_calibration_for_euler,
)


def _make_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "snakemake.yaml"
    config_path.write_text(yaml.safe_dump({
        "paths": {"trimmed_fasta": "data/interim/IPR019888_trimmed.aln"},
        "null_calibration": {
            "replicates": 20,
            "seed": 123,
            "shrinkage": 0.2,
            "asr_model": "LG+G",
            "sim_tree": "data/interim/iqtree_asr/IPR019888.treefile",
        },
    }))
    return config_path


def _bash_syntax_ok(path: Path) -> bool:
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    return result.returncode == 0


@pytest.fixture()
def package(tmp_path):
    config_path = _make_config(tmp_path)
    tar_path = package_null_calibration_for_euler(
        root=tmp_path,
        config_path=config_path,
        num_replicates=20,
        array_throttle=5,
        project_root_remote="/cluster/project/beltrao/lucla/repos/badasp",
    )
    package_dir = tmp_path / "results" / "badasp_scoring" / "null_calibration" / "euler_package"
    return tar_path, package_dir


def test_package_creates_expected_files(package):
    tar_path, package_dir = package
    assert tar_path.exists() and tar_path.stat().st_size > 0
    for name in (
        "run_simulate_null_calibration.sh",
        "run_score_null_calibration_array.sh",
        "submit_null_calibration.sh",
        "INSTRUCTIONS.txt",
    ):
        assert (package_dir / name).exists(), f"{name} missing"


def test_generated_scripts_are_valid_bash(package):
    _, package_dir = package
    for name in (
        "run_simulate_null_calibration.sh",
        "run_score_null_calibration_array.sh",
        "submit_null_calibration.sh",
    ):
        assert _bash_syntax_ok(package_dir / name), f"{name} failed bash -n"


def test_tarball_bundles_only_generated_scripts_not_data(package):
    tar_path, _ = package
    with tarfile.open(tar_path, "r:gz") as tar:
        names = sorted(Path(n).name for n in tar.getnames())
    assert names == sorted([
        "run_simulate_null_calibration.sh",
        "run_score_null_calibration_array.sh",
        "submit_null_calibration.sh",
        "INSTRUCTIONS.txt",
    ])


def test_euler_conventions_present_in_both_sbatch_scripts(package):
    _, package_dir = package
    for name in ("run_simulate_null_calibration.sh", "run_score_null_calibration_array.sh"):
        content = (package_dir / name).read_text()
        assert "module load stack/2025-06 gcc/12.2.0" in content
        assert "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in content
        assert "--mem-per-cpu=" in content
        assert "#SBATCH --mem=" not in content
        assert "%j" in content or "%A_%a" in content
        assert "/cluster/project/beltrao/lucla/repos/badasp" in content


def test_array_spec_uses_num_replicates_and_throttle(package):
    _, package_dir = package
    content = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "#SBATCH --array=0-19%5" in content  # num_replicates=20, throttle=5


def test_simulate_job_is_not_an_array(package):
    _, package_dir = package
    content = (package_dir / "run_simulate_null_calibration.sh").read_text()
    assert "#SBATCH --array" not in content
    assert "--num-alignments 20" in content


def test_resource_block_matches_measured_constants(package):
    _, package_dir = package
    simulate = (package_dir / "run_simulate_null_calibration.sh").read_text()
    expected_sim_time = _fmt_hms(SIMULATE_MEASURED_SECONDS * SIMULATE_WALLTIME_SAFETY_FACTOR)
    expected_sim_mem = f"{SIMULATE_MEASURED_MEM_GB + SIMULATE_MEM_HEADROOM_GB:.1f}G"
    assert f"#SBATCH --time={expected_sim_time}" in simulate
    assert f"#SBATCH --mem-per-cpu={expected_sim_mem}" in simulate

    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    expected_score_time = _fmt_hms(SCORE_LOADED_MINUTES_HIGH * 60 * SCORE_WALLTIME_SAFETY_FACTOR)
    expected_score_mem_mb = round((SCORE_ASR_MEASURED_MEM_GB + SCORE_MEM_HEADROOM_GB) * 1024 / SCORE_THREADS)
    assert f"#SBATCH --time={expected_score_time}" in score
    assert f"#SBATCH --mem-per-cpu={expected_score_mem_mb}M" in score
    assert f"--cpus-per-task={SCORE_THREADS}" in score


def test_score_walltime_derived_from_loaded_not_standalone_measurement(package):
    """The scoring array's walltime must come from the contention-realistic
    40-46 min/replicate figure, not the superseded standalone-measurement
    estimate (446 s x 1.6 = 11:54) that this script previously used and that
    did not survive contact with load."""
    _, package_dir = package
    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "#SBATCH --time=00:11:54" not in score
    expected_score_time = _fmt_hms(SCORE_LOADED_MINUTES_HIGH * 60 * SCORE_WALLTIME_SAFETY_FACTOR)
    assert f"#SBATCH --time={expected_score_time}" in score
    # Sanity: the corrected walltime must be substantially larger than the
    # old, wrong estimate -- guards against silently reverting the fix.
    old_wrong_seconds = (313 + 133) * 1.6
    new_seconds = SCORE_LOADED_MINUTES_HIGH * 60 * SCORE_WALLTIME_SAFETY_FACTOR
    assert new_seconds > old_wrong_seconds * 2


def test_array_throttle_default_is_lowered_for_bandwidth_bound_workload():
    """The default array throttle must reflect that this workload is
    memory-bandwidth-bound (packing many tasks per node does not scale
    throughput), not the old core-bound-assuming default of 50."""
    assert ARRAY_THROTTLE_DEFAULT < 50


def test_score_array_documents_bandwidth_bound_node_packing_tradeoff(package):
    _, package_dir = package
    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "memory-bandwidth" in score
    assert "--exclusive" in score
    assert "--ntasks-per-node" in score
    assert "NodeList" in score


def test_pilot_and_instructions_flag_different_hardware(package):
    _, package_dir = package
    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "DIFFERENT HARDWARE" in score
    assert "Apple Silicon" in score

    instructions = (package_dir / "INSTRUCTIONS.txt").read_text()
    assert "DIFFERENT" in instructions
    assert "Apple Silicon" in instructions
    assert "NodeList" in instructions


def test_score_array_reuses_is_valid_npz_not_reimplemented(package):
    _, package_dir = package
    content = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "from scripts.run_null_calibration import is_valid_npz" in content
    assert "is_valid_npz(Path(" in content


def test_pilot_instruction_present_in_score_array_and_instructions(package):
    _, package_dir = package
    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "2-task pilot" in score
    assert "seff" in score
    assert "sacct" in score

    instructions = (package_dir / "INSTRUCTIONS.txt").read_text()
    assert "2-task" in instructions
    assert "seff" in instructions


def test_score_script_propagates_model_min_clade_node_naming(tmp_path):
    config_path = _make_config(tmp_path)
    package_null_calibration_for_euler(
        root=tmp_path, config_path=config_path, num_replicates=10,
        asr_model="LG+G4", min_clade=8, node_naming="legacy",
    )
    content = (tmp_path / "results/badasp_scoring/null_calibration/euler_package/"
               "run_score_null_calibration_array.sh").read_text()
    assert "--model LG+G4" in content
    assert "--min-clade 8" in content
    assert "--node-naming legacy" in content


def test_submit_wrapper_chains_jobs_with_dependency(package):
    _, package_dir = package
    content = (package_dir / "submit_null_calibration.sh").read_text()
    assert "sbatch --parsable run_simulate_null_calibration.sh" in content
    assert "--dependency=afterok:$SIM_JOBID" in content


def test_custom_project_root_remote_used_everywhere(tmp_path):
    config_path = _make_config(tmp_path)
    custom_root = "/cluster/scratch/someone/badasp"
    package_null_calibration_for_euler(
        root=tmp_path, config_path=config_path, num_replicates=5,
        project_root_remote=custom_root,
    )
    package_dir = tmp_path / "results/badasp_scoring/null_calibration/euler_package"
    for name in ("run_simulate_null_calibration.sh", "run_score_null_calibration_array.sh"):
        content = (package_dir / name).read_text()
        assert custom_root in content
        assert "/cluster/project/beltrao/lucla/repos/badasp" not in content


def test_rejects_invalid_num_replicates_and_throttle(tmp_path):
    config_path = _make_config(tmp_path)
    with pytest.raises(ValueError):
        package_null_calibration_for_euler(root=tmp_path, config_path=config_path, num_replicates=0)
    with pytest.raises(ValueError):
        package_null_calibration_for_euler(root=tmp_path, config_path=config_path,
                                           num_replicates=5, array_throttle=0)


def test_fmt_hms():
    assert _fmt_hms(0) == "00:00:00"
    assert _fmt_hms(65) == "00:01:05"
    assert _fmt_hms(3661) == "01:01:01"
