#!/usr/bin/env python3
"""
package_null_calibration_for_euler.py

Generates the Euler-side SLURM scripts for the null-calibration replicate
loop and bundles them into a tar.gz, mirroring
scripts/package_rec_check_for_euler.py's convention (a package_*() function
building a small package directory, then a tar.gz of it).

Two jobs, matching how the work actually splits
--------------------------------------------------
* One (non-array) job runs simulate_null_persite.py once with
  ``--num-alignments <num_replicates>``, producing every replicate's
  simulated alignment in a single batched AliSim invocation (see that
  script's own docstring on why batching amortizes its one-time setup cost;
  this is the same reason run_null_calibration.py batches locally).
* A SLURM array job, one task per replicate, runs score_null_replicate.py
  directly on that replicate's simulated alignment. This does NOT go
  through run_null_calibration.py's own batching/ThreadPoolExecutor
  concurrency control -- that mechanism exists for bounding concurrency on
  a single machine (see its --concurrency option); on a cluster, SLURM's
  own array scheduler already provides the concurrency across many nodes,
  so nesting one inside the other would be redundant. Each array task
  still skips a replicate whose .npz already exists and is valid, reusing
  run_null_calibration.is_valid_npz (not reimplementing it), so a partially
  completed array can be resubmitted for just the failed/missing indices.

Data vs. code transfer
------------------------
Per this project's established Euler workflow (code via GitHub push + then
`git pull` on Euler; data files not tracked by git via `rsync` -- see this
repo's own Euler-workflow notes), this script does NOT bundle the
alignment, trees, or observed-score CSV into the tarball: they are
gitignored data files (results/, data/interim/**) and must be rsynced to
the cluster directly, like every other data transfer in this repo.
scripts/run_null_calibration.py, simulate_null_persite.py and
score_null_replicate.py reach Euler via `git pull` since they are tracked
in git already. This script only generates the two sbatch scripts, a
submission wrapper, and an instructions file.

Resource sizing (see MEASURED_* constants below)
---------------------------------------------------
Sized from the measured, task-provided per-invocation figures, not padding:
  * simulate_null_persite.py, full scale: 197 s wall-clock, 4.5 GB peak RAM
    for one invocation (the one-time 2M-site reference-frequency probe
    dominates cost, which is why batching many replicates into one
    --num-alignments call is cheap regardless of how many are requested).
    Its dependence on --num-alignments count specifically has not been
    measured, so the simulate job's walltime below is a flagged, generous
    starting estimate, not a tight measured bound.
  * score_null_replicate.py, one replicate: ASR 313 s at ~3.3 GB peak, then
    scoring 133 s at under 1 GB. The two phases run sequentially within one
    task, so a task's peak memory need is the larger of the two (~3.3 GB),
    not their sum.
Both remain ESTIMATES until run on Euler itself: a 2-task pilot of the
scoring array is required before submitting the full array (see that
script's own leading comment), checked with `seff <jobid>` or
`sacct --format=MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS`, and the resource
block re-derived from that pilot rather than trusted as final.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "snakemake.yaml"
DEFAULT_PROJECT_ROOT_REMOTE = "/cluster/project/beltrao/lucla/repos/badasp"

# Measured, not guessed -- see the module docstring for where each figure
# comes from. Kept as named constants so every number in the generated
# sbatch scripts traces back to one of these, rather than being retyped.
SIMULATE_MEASURED_SECONDS = 197
SIMULATE_MEASURED_MEM_GB = 4.5
SCORE_ASR_MEASURED_SECONDS = 313
SCORE_ASR_MEASURED_MEM_GB = 3.3
SCORE_SCORING_MEASURED_SECONDS = 133
SCORE_SCORING_MEASURED_MEM_GB = 1.0  # "under 1 GB"
SCORE_THREADS = 2  # matches the -T 2 the ASR measurement above was taken at,
                    # and score_null_replicate.py's own --threads default.

# Small, explicitly-flagged safety margins -- not "just in case" padding:
# the simulate job's walltime margin is large because its scaling with
# --num-alignments hasn't been measured (see docstring); the scoring
# array's margins are modest because that phase's own peak has been
# measured directly and only needs headroom for run-to-run variance.
SIMULATE_WALLTIME_SAFETY_FACTOR = 4.0
SIMULATE_MEM_HEADROOM_GB = 0.5
SCORE_WALLTIME_SAFETY_FACTOR = 1.6
SCORE_MEM_HEADROOM_GB = 0.3


def _fmt_hms(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _npz_width(num_replicates: int) -> int:
    return max(4, len(str(max(num_replicates - 1, 0))))


def _load_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def package_null_calibration_for_euler(
    root: Optional[Path] = None,
    config_path: Optional[Path] = None,
    num_replicates: Optional[int] = None,
    seed: Optional[int] = None,
    shrinkage: Optional[float] = None,
    asr_model: Optional[str] = None,
    min_clade: int = 5,
    node_naming: str = "strict",
    array_throttle: int = 50,
    project_root_remote: str = DEFAULT_PROJECT_ROOT_REMOTE,
) -> Path:
    """Generate the Euler package (sbatch scripts + instructions + tarball).

    Returns the path to the created tar.gz bundle.
    """
    if root is None:
        root = REPO_ROOT
    cfg = _load_config(config_path or (root / "config" / "snakemake.yaml"))
    nc = cfg.get("null_calibration", {})
    paths = cfg.get("paths", {})

    num_replicates = num_replicates if num_replicates is not None else nc.get("replicates", 1000)
    seed = seed if seed is not None else nc.get("seed")
    shrinkage = shrinkage if shrinkage is not None else nc.get("shrinkage", 0.15)
    asr_model = asr_model or nc.get("asr_model", "LG+G")
    if array_throttle < 1:
        raise ValueError("array_throttle must be >= 1")
    if num_replicates < 1:
        raise ValueError("num_replicates must be >= 1")

    width = _npz_width(num_replicates)

    # Remote paths, derived from the same relative paths this repo already
    # uses locally (config/snakemake.yaml + the task's stated ASR-topology
    # path), rooted at the Euler project directory -- never a separately
    # hand-maintained set of remote paths.
    remote_alignment = f"{project_root_remote}/{paths.get('trimmed_fasta', 'data/interim/IPR019888_trimmed.aln')}"
    remote_sim_tree = f"{project_root_remote}/{nc.get('sim_tree', 'data/interim/iqtree_asr/IPR019888.treefile')}"
    remote_reconciled_tree = (
        f"{project_root_remote}/results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk"
    )
    remote_asr_tree = f"{project_root_remote}/data/interim/iqtree_asr/IPR019888_rooted.tree"
    remote_observed_scores = f"{project_root_remote}/results/badasp_scoring/raw_node_scores.csv"
    remote_run_dir = f"{project_root_remote}/results/badasp_scoring/null_calibration/run"
    remote_logs_dir = f"{project_root_remote}/results/badasp_scoring/null_calibration/logs"

    package_dir = root / "results" / "badasp_scoring" / "null_calibration" / "euler_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    # --- Simulate job (single job, not an array) --------------------------
    simulate_walltime = _fmt_hms(SIMULATE_MEASURED_SECONDS * SIMULATE_WALLTIME_SAFETY_FACTOR)
    simulate_mem_gb = SIMULATE_MEASURED_MEM_GB + SIMULATE_MEM_HEADROOM_GB
    simulate_script = package_dir / "run_simulate_null_calibration.sh"
    simulate_script.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=null_calib_simulate\n"
        f"#SBATCH --output={remote_logs_dir}/simulate_%j.out\n"
        f"#SBATCH --error={remote_logs_dir}/simulate_%j.err\n"
        f"#SBATCH --time={simulate_walltime}\n"
        "#SBATCH --cpus-per-task=1\n"
        f"#SBATCH --mem-per-cpu={simulate_mem_gb:.1f}G\n"
        "\n"
        "# One batched AliSim invocation producing every replicate's simulated\n"
        f"# alignment ({num_replicates} of them). Sized from the measured full-scale\n"
        f"# figures ({SIMULATE_MEASURED_SECONDS} s, {SIMULATE_MEASURED_MEM_GB} GB for one\n"
        "# invocation; see package_null_calibration_for_euler.py's module docstring).\n"
        "# The walltime carries a larger safety factor than the scoring array below\n"
        "# because this invocation's scaling with --num-alignments has not itself been\n"
        "# measured -- re-size after actually observing this job (seff/sacct), not\n"
        "# from this default.\n"
        "\n"
        "module load stack/2025-06 gcc/12.2.0\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "\n"
        f"PROJECT_ROOT={project_root_remote}\n"
        'export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"\n'
        f"RUN_DIR={remote_run_dir}\n"
        'mkdir -p "$RUN_DIR/sim" "' + remote_logs_dir + '"\n'
        "\n"
        'echo "Starting null-calibration simulate job on $(hostname) at $(date)"\n'
        "\n"
        '$PROJECT_ROOT/venv/bin/python "$PROJECT_ROOT/scripts/simulate_null_persite.py" \\\n'
        f'  --composition-alignment "{remote_alignment}" \\\n'
        f'  --sim-tree "{remote_sim_tree}" \\\n'
        '  --out-prefix "$RUN_DIR/sim/sim" \\\n'
        f"  --shrinkage {shrinkage} \\\n"
        f"  --num-alignments {num_replicates} \\\n"
        f"  --seed {seed} \\\n"
        "  --threads $SLURM_CPUS_PER_TASK \\\n"
        "  --redo\n"
        "\n"
        'echo "Simulate job finished at $(date)"\n',
        encoding="utf-8",
    )
    simulate_script.chmod(0o755)

    # --- Scoring job (array, one task per replicate) -----------------------
    score_walltime = _fmt_hms(
        (SCORE_ASR_MEASURED_SECONDS + SCORE_SCORING_MEASURED_SECONDS) * SCORE_WALLTIME_SAFETY_FACTOR
    )
    score_mem_gb = SCORE_ASR_MEASURED_MEM_GB + SCORE_MEM_HEADROOM_GB
    score_mem_per_cpu_mb = round(score_mem_gb * 1024 / SCORE_THREADS)
    array_spec = f"0-{num_replicates - 1}%{array_throttle}"
    score_script = package_dir / "run_score_null_calibration_array.sh"
    score_script.write_text(
        "#!/bin/bash\n"
        "#SBATCH --job-name=null_calib_score\n"
        f"#SBATCH --array={array_spec}\n"
        f"#SBATCH --output={remote_logs_dir}/score_%A_%a.out\n"
        f"#SBATCH --error={remote_logs_dir}/score_%A_%a.err\n"
        f"#SBATCH --time={score_walltime}\n"
        f"#SBATCH --cpus-per-task={SCORE_THREADS}\n"
        f"#SBATCH --mem-per-cpu={score_mem_per_cpu_mb}M\n"
        "\n"
        "# IMPORTANT: before submitting the full array above, submit a 2-task pilot\n"
        "# first (e.g. `sbatch --array=0-1 run_score_null_calibration_array.sh`) and\n"
        "# check its actual usage with `seff <jobid>` or\n"
        "# `sacct --format=MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS -j <jobid>` before\n"
        "# submitting the rest. Re-derive --time/--mem-per-cpu above from that pilot's\n"
        "# real numbers -- the values here are sized from measurements taken outside\n"
        "# Euler (see package_null_calibration_for_euler.py's module docstring) and\n"
        "# should not be trusted as final for this cluster's hardware.\n"
        "#\n"
        f"# Per-task sizing: {SCORE_THREADS} CPUs (matches score_null_replicate.py's own\n"
        f"# -T {SCORE_THREADS}); mem-per-cpu = ({SCORE_ASR_MEASURED_MEM_GB} GB measured ASR peak +\n"
        f"# {SCORE_MEM_HEADROOM_GB} GB headroom) / {SCORE_THREADS} CPUs. The ASR phase's peak, not the sum of\n"
        "# ASR + scoring, is used because the two phases run sequentially within one\n"
        "# task (scoring runs after ASR has already released its memory).\n"
        "\n"
        "module load stack/2025-06 gcc/12.2.0\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "\n"
        f"PROJECT_ROOT={project_root_remote}\n"
        'export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"\n'
        f"RUN_DIR={remote_run_dir}\n"
        'mkdir -p "$RUN_DIR/npz" "$RUN_DIR/scratch" "' + remote_logs_dir + '"\n'
        "\n"
        "IDX=$SLURM_ARRAY_TASK_ID\n"
        "POS=$((IDX + 1))\n"
        f'REP_ID=$(printf "%0{width}d" "$IDX")\n'
        'NPZ="$RUN_DIR/npz/rep_${REP_ID}.npz"\n'
        "\n"
        "# Resumable: reuse run_null_calibration.py's own .npz validity check (not\n"
        "# reimplemented here) so a partially completed array can be resubmitted for\n"
        "# just the missing/invalid indices without redoing finished replicates.\n"
        'if [ -f "$NPZ" ]; then\n'
        '  if $PROJECT_ROOT/venv/bin/python -c "\n'
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, '$PROJECT_ROOT')\n"
        "from scripts.run_null_calibration import is_valid_npz\n"
        "sys.exit(0 if is_valid_npz(Path('$NPZ')) else 1)\n"
        '"; then\n'
        '    echo "Replicate $IDX already done ($NPZ); skipping."\n'
        "    exit 0\n"
        "  fi\n"
        "fi\n"
        "\n"
        'SIM_FA="$RUN_DIR/sim/sim_${POS}.fa"\n'
        'WORK="$RUN_DIR/scratch/rep_${REP_ID}_work"\n'
        "\n"
        'echo "Starting replicate $IDX (alignment $SIM_FA) on $(hostname) at $(date)"\n'
        "\n"
        '$PROJECT_ROOT/venv/bin/python "$PROJECT_ROOT/scripts/score_null_replicate.py" \\\n'
        '  --sim-alignment "$SIM_FA" \\\n'
        f'  --reconciled-tree "{remote_reconciled_tree}" \\\n'
        f'  --asr-tree "{remote_asr_tree}" \\\n'
        f'  --observed-scores "{remote_observed_scores}" \\\n'
        '  --out-npz "$NPZ" \\\n'
        f"  --model {asr_model} \\\n"
        f"  --min-clade {min_clade} \\\n"
        f"  --node-naming {node_naming} \\\n"
        "  --threads $SLURM_CPUS_PER_TASK \\\n"
        '  --workdir "$WORK"\n'
        "STATUS=$?\n"
        "\n"
        "# Delete the simulated alignment once attempted (success or failure), same\n"
        "# policy as run_null_calibration.py's local driver: it is reproducible from\n"
        "# the simulate job's (seed, num_replicates) alone.\n"
        'rm -f "$SIM_FA"\n'
        "\n"
        'echo "Replicate $IDX finished at $(date) with status $STATUS"\n'
        "exit $STATUS\n",
        encoding="utf-8",
    )
    score_script.chmod(0o755)

    # --- Submission wrapper (chains the two jobs via --dependency) --------
    submit_script = package_dir / "submit_null_calibration.sh"
    submit_script.write_text(
        "#!/bin/bash\n"
        "# Submits the simulate job, then the scoring array with a dependency on it.\n"
        "# Run this from the Euler project root after `git pull`.\n"
        "set -euo pipefail\n"
        "\n"
        "SIM_JOBID=$(sbatch --parsable run_simulate_null_calibration.sh)\n"
        'echo "Submitted simulate job: $SIM_JOBID"\n'
        "\n"
        "SCORE_JOBID=$(sbatch --parsable --dependency=afterok:$SIM_JOBID run_score_null_calibration_array.sh)\n"
        'echo "Submitted scoring array job: $SCORE_JOBID (depends on $SIM_JOBID)"\n'
        'echo "Remember: run a 2-task pilot of the scoring array first (see its own leading comment) before trusting the full array at scale."\n',
        encoding="utf-8",
    )
    submit_script.chmod(0o755)

    instructions_path = package_dir / "INSTRUCTIONS.txt"
    instructions_path.write_text(
        "Null-calibration Euler package\n"
        "==============================\n"
        "\n"
        "1. Push code changes to GitHub, then on Euler:\n"
        f"     cd {project_root_remote} && git pull origin main\n"
        "\n"
        "2. rsync the data inputs this run needs (these are gitignored, not code):\n"
        f"     rsync -avz --relative {paths.get('trimmed_fasta', 'data/interim/IPR019888_trimmed.aln')} "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        f"     rsync -avz --relative {nc.get('sim_tree', 'data/interim/iqtree_asr/IPR019888.treefile')} "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        "     rsync -avz --relative data/interim/iqtree_asr/IPR019888_rooted.tree "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        "     rsync -avz --relative results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        "     rsync -avz --relative results/badasp_scoring/raw_node_scores.csv "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        "\n"
        "3. Extract this tarball's contents into the Euler project root and run:\n"
        "     bash submit_null_calibration.sh\n"
        "\n"
        "4. IMPORTANT -- before trusting the full scoring array, submit a 2-task\n"
        "   pilot first and check `seff <jobid>` / `sacct --format=MaxRSS,TotalCPU,\n"
        "   Elapsed,ReqMem,ReqCPUS`; re-size --time/--mem-per-cpu in\n"
        "   run_score_null_calibration_array.sh from that pilot's real numbers.\n"
        "\n"
        "5. Once the array completes, aggregate on Euler or after rsyncing "
        f"{remote_run_dir}/npz/ back:\n"
        "     python scripts/calibrate_switch_threshold.py --null-run-dir <run dir> "
        "--observed-scores <observed CSV>\n",
        encoding="utf-8",
    )

    tar_path = root / "results" / "badasp_scoring" / "null_calibration" / "null_calibration_euler_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in (simulate_script, score_script, submit_script, instructions_path):
            tar.add(f, arcname=f"null_calibration_euler/{f.name}")

    print(f"Wrote {simulate_script}")
    print(f"Wrote {score_script}")
    print(f"Wrote {submit_script}")
    print(f"Wrote {instructions_path}")
    print(f"Created tarball {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")
    return tar_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Euler sbatch scripts + tarball for the "
                    "null-calibration replicate loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--num-replicates", type=int, default=None,
                        help="Default: config null_calibration.replicates.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Default: config null_calibration.seed.")
    parser.add_argument("--shrinkage", type=float, default=None,
                        help="Default: config null_calibration.shrinkage.")
    parser.add_argument("--asr-model", default=None,
                        help="Default: config null_calibration.asr_model.")
    parser.add_argument("--min-clade", type=int, default=5)
    parser.add_argument("--node-naming", choices=["legacy", "strict"], default="strict")
    parser.add_argument("--array-throttle", type=int, default=50,
                        help="Max concurrent array tasks (SLURM %% syntax). This "
                             "is a cluster-citizenship/fair-share choice, not a "
                             "measured value -- adjust to your project's Euler "
                             "allocation.")
    parser.add_argument("--project-root-remote", default=DEFAULT_PROJECT_ROOT_REMOTE)
    args = parser.parse_args()
    package_null_calibration_for_euler(
        config_path=args.config,
        num_replicates=args.num_replicates,
        seed=args.seed,
        shrinkage=args.shrinkage,
        asr_model=args.asr_model,
        min_clade=args.min_clade,
        node_naming=args.node_naming,
        array_throttle=args.array_throttle,
        project_root_remote=args.project_root_remote,
    )


if __name__ == "__main__":
    main()
