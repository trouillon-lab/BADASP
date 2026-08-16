import re
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts.package_null_calibration_for_euler import (
    ARRAY_THROTTLE_DEFAULT,
    CHUNK_SEED_STRIDE,
    SCORE_EULER_ASR_SECONDS,
    SCORE_EULER_MEM_GB,
    SCORE_EULER_SCORING_SECONDS,
    SCORE_LAPTOP_LOADED_MINUTES_HIGH,
    SCORE_MEM_HEADROOM_GB,
    SCORE_THREADS,
    SCORE_WALLTIME_SAFETY_FACTOR,
    SIMULATE_EULER_MEM_GB,
    SIMULATE_EULER_PROBE_SECONDS,
    SIMULATE_EULER_SECONDS_PER_ALIGNMENT,
    SIMULATE_MEM_HEADROOM_GB,
    SIMULATE_THREADS,
    SIMULATE_WALLTIME_SAFETY_FACTOR,
    _fmt_hms,
    package_null_calibration_for_euler,
)

# num_replicates=20 with this chunk size gives 4 simulate chunks, so the
# tests exercise a genuine multi-chunk array rather than a degenerate one.
CHUNK_SIZE = 5


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
        simulate_chunk_size=CHUNK_SIZE,
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
        "verify_euler_inputs.sh",
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
        "verify_euler_inputs.sh",
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


def test_simulate_job_is_a_chunked_array(package):
    """One long simulate job was a single point of failure and, at ~897 s per
    alignment on Euler, would exceed the walltime limit outright for a few
    hundred replicates."""
    _, package_dir = package
    content = (package_dir / "run_simulate_null_calibration.sh").read_text()
    n_chunks = -(-20 // CHUNK_SIZE)
    assert f"#SBATCH --array=0-{n_chunks - 1}%5" in content
    assert f"--num-alignments {CHUNK_SIZE}" in content
    # The whole-run count must NOT appear as a per-task alignment count.
    assert "--num-alignments 20" not in content


def test_simulate_chunks_get_distinct_seeds(package):
    """Identical seeds across chunks would produce identical alignments and
    silently collapse the effective replicate count."""
    _, package_dir = package
    content = (package_dir / "run_simulate_null_calibration.sh").read_text()
    assert f"SEED=$(( 123 + CHUNK * {CHUNK_SEED_STRIDE} ))" in content
    assert "--seed $SEED" in content


def test_score_array_maps_index_to_chunk_and_position(package):
    _, package_dir = package
    content = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert f"CHUNK=$(( IDX / {CHUNK_SIZE} ))" in content
    assert f"CHUNK_POS=$(( IDX % {CHUNK_SIZE} + 1 ))" in content
    assert 'SIM_FA="$RUN_DIR/sim/chunk_${CHUNK}/sim_${CHUNK_POS}.fa"' in content
    # Missing alignment must fail loudly rather than be passed to IQ-TREE.
    assert 'if [ ! -f "$SIM_FA" ]; then' in content


def test_memory_specs_are_integers_slurm_accepts(package):
    """SLURM rejects a decimal --mem-per-cpu such as '5.0G'; the previous
    simulate script emitted exactly that."""
    _, package_dir = package
    for name in ("run_simulate_null_calibration.sh", "run_score_null_calibration_array.sh"):
        for line in (package_dir / name).read_text().splitlines():
            if line.startswith("#SBATCH --mem-per-cpu="):
                value = line.split("=", 1)[1]
                assert re.fullmatch(r"\d+[KMGT]", value), f"{name}: bad mem spec {value!r}"


def test_simulate_requests_the_cores_its_walltime_assumes(package):
    """Pilot 10854059 timed out with zero alignments because it requested 1
    CPU while its walltime came from a 4-thread measurement."""
    _, package_dir = package
    content = (package_dir / "run_simulate_null_calibration.sh").read_text()
    assert f"#SBATCH --cpus-per-task={SIMULATE_THREADS}" in content
    assert SIMULATE_THREADS > 1
    assert "--threads $SLURM_CPUS_PER_TASK" in content


def test_resource_block_matches_measured_constants(package):
    _, package_dir = package
    simulate = (package_dir / "run_simulate_null_calibration.sh").read_text()
    expected_sim_time = _fmt_hms(
        (SIMULATE_EULER_SECONDS_PER_ALIGNMENT * CHUNK_SIZE + SIMULATE_EULER_PROBE_SECONDS)
        * SIMULATE_WALLTIME_SAFETY_FACTOR
    )
    expected_sim_mem_mb = round(
        (SIMULATE_EULER_MEM_GB + SIMULATE_MEM_HEADROOM_GB) * 1024 / SIMULATE_THREADS
    )
    assert f"#SBATCH --time={expected_sim_time}" in simulate
    assert f"#SBATCH --mem-per-cpu={expected_sim_mem_mb}M" in simulate

    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    expected_score_time = _fmt_hms(
        (SCORE_EULER_ASR_SECONDS + SCORE_EULER_SCORING_SECONDS) * SCORE_WALLTIME_SAFETY_FACTOR
    )
    expected_score_mem_mb = round((SCORE_EULER_MEM_GB + SCORE_MEM_HEADROOM_GB) * 1024 / SCORE_THREADS)
    assert f"#SBATCH --time={expected_score_time}" in score
    assert f"#SBATCH --mem-per-cpu={expected_score_mem_mb}M" in score
    assert f"--cpus-per-task={SCORE_THREADS}" in score


def test_simulate_walltime_scales_with_chunk_size(package, tmp_path):
    """The old bug: walltime was derived from a single alignment and never
    multiplied by how many the job actually produces."""
    config_path = _make_config(tmp_path / "alt")
    package_null_calibration_for_euler(
        root=tmp_path / "alt",
        config_path=config_path,
        num_replicates=20,
        array_throttle=5,
        simulate_chunk_size=CHUNK_SIZE * 2,
        project_root_remote="/cluster/project/beltrao/lucla/repos/badasp",
    )
    big = (tmp_path / "alt" / "results" / "badasp_scoring" / "null_calibration"
           / "euler_package" / "run_simulate_null_calibration.sh").read_text()
    _, package_dir = package
    small = (package_dir / "run_simulate_null_calibration.sh").read_text()

    def walltime_seconds(text):
        spec = next(l for l in text.splitlines() if l.startswith("#SBATCH --time="))
        h, m, s = (int(x) for x in spec.split("=", 1)[1].split(":"))
        return h * 3600 + m * 60 + s

    # Doubling the alignments per chunk must roughly double the walltime; the
    # one-time reference probe keeps it just under exactly 2x.
    ratio = walltime_seconds(big) / walltime_seconds(small)
    assert 1.9 < ratio <= 2.0


def test_score_walltime_derived_from_euler_native_measurement(package):
    """The walltime must come from the Euler-native per-replicate figure, not
    from either superseded laptop estimate: the standalone one (446 s x 1.6 =
    11:54) or the loaded-laptop one (46 min), both of which under-sized it."""
    _, package_dir = package
    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "#SBATCH --time=00:11:54" not in score
    expected_score_time = _fmt_hms(
        (SCORE_EULER_ASR_SECONDS + SCORE_EULER_SCORING_SECONDS) * SCORE_WALLTIME_SAFETY_FACTOR
    )
    assert f"#SBATCH --time={expected_score_time}" in score
    # Guards against reverting to either laptop-derived figure.
    new_seconds = (SCORE_EULER_ASR_SECONDS + SCORE_EULER_SCORING_SECONDS) * SCORE_WALLTIME_SAFETY_FACTOR
    assert new_seconds > (313 + 133) * 1.6 * 2
    assert new_seconds > SCORE_LAPTOP_LOADED_MINUTES_HIGH * 60 * SCORE_WALLTIME_SAFETY_FACTOR * 2


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


def test_scripts_state_which_euler_job_measured_them(package):
    """The sizing is now Euler-native, so the scripts must cite the jobs it
    came from rather than carrying the old cross-hardware caveat."""
    _, package_dir = package
    score = (package_dir / "run_score_null_calibration_array.sh").read_text()
    assert "EULER-NATIVE" in score
    assert "10861338" in score
    assert "NodeList" in score

    simulate = (package_dir / "run_simulate_null_calibration.sh").read_text()
    assert "10856979" in simulate
    assert "10854059" in simulate  # why cpus-per-task is not 1

    instructions = (package_dir / "INSTRUCTIONS.txt").read_text()
    assert "EULER-NATIVE" in instructions
    assert "NodeList" in instructions


def test_unmeasured_simulate_memory_scaling_is_flagged(package):
    """MaxRSS was measured at 4 alignments per invocation; a 25-alignment
    chunk is an extrapolation and must say so rather than read as measured."""
    _, package_dir = package
    simulate = (package_dir / "run_simulate_null_calibration.sh").read_text()
    assert "UNMEASURED" in simulate
    assert "MaxRSS" in simulate
    instructions = (package_dir / "INSTRUCTIONS.txt").read_text()
    assert "unmeasured" in instructions.lower()


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


def test_samples_newick_is_in_the_rsync_list(package):
    """The AleRax samples file is easy to miss because no script names it on
    the command line -- score_tree_nodes derives it from --reconciled-tree's
    directory. A stale copy raises no error: it just relabels nodes, and any
    node not labelled Speciation/Duplication/Transfer is dropped from
    scoring. On Euler that silently cut scored node pairs from 1,607 to 946."""
    _, package_dir = package
    instructions = (package_dir / "INSTRUCTIONS.txt").read_text()
    assert "reconciliations/all/IPR019888_samples.newick" in instructions


def test_input_verification_script_covers_every_input(package):
    _, package_dir = package
    script = package_dir / "verify_euler_inputs.sh"
    assert script.exists()
    content = script.read_text()
    for required in (
        "data/interim/IPR019888_trimmed.aln",
        "data/interim/iqtree_asr/IPR019888.treefile",
        "data/interim/iqtree_asr/IPR019888_rooted.tree",
        "reconciliations/IPR019888.nwk",
        "reconciliations/all/IPR019888_samples.newick",
        "results/badasp_scoring/raw_node_scores.csv",
    ):
        assert required in content, f"{required} not checksummed"
    # Must work on both machines: md5sum on Linux, md5 on macOS.
    assert "md5sum" in content and "md5 -q" in content
    assert _bash_syntax_ok(script)


def test_input_verification_script_reports_real_checksums(package, tmp_path):
    """Guards the regression where the script cd'd somewhere without the data
    and reported every input MISSING."""
    _, package_dir = package
    root = tmp_path / "fakeroot"
    (root / "data" / "interim").mkdir(parents=True)
    target = root / "data" / "interim" / "IPR019888_trimmed.aln"
    target.write_text(">a\nMKV\n")
    result = subprocess.run(
        ["bash", str(package_dir / "verify_euler_inputs.sh"), str(root)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    line = next(l for l in result.stdout.splitlines() if "IPR019888_trimmed.aln" in l)
    assert "MISSING" not in line, line
    assert str(target.stat().st_size) in line


def test_input_verification_script_rejects_a_bad_root(package):
    _, package_dir = package
    result = subprocess.run(
        ["bash", str(package_dir / "verify_euler_inputs.sh"), "/definitely/not/here"],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "No such project root" in result.stderr


def test_verification_script_is_bundled(package):
    tar_path, _ = package
    with tarfile.open(tar_path, "r:gz") as tar:
        assert any(Path(n).name == "verify_euler_inputs.sh" for n in tar.getnames())
