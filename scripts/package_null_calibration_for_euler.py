#!/usr/bin/env python3
"""
package_null_calibration_for_euler.py

Generates the Euler-side SLURM scripts for the null-calibration replicate
loop and bundles them into a tar.gz, mirroring
scripts/package_rec_check_for_euler.py's convention (a package_*() function
building a small package directory, then a tar.gz of it).

Two jobs, matching how the work actually splits
--------------------------------------------------
* A SLURM array job, one task per CHUNK of alignments, runs
  simulate_null_persite.py with ``--num-alignments <chunk size>`` and a
  chunk-specific seed. Batching within a chunk amortizes the one-time
  reference-frequency probe (see that script's own docstring), but a single
  non-array job for the whole run is not viable on Euler: at ~897 s per
  alignment, 300 alignments is ~75 h -- over the walltime limit, entirely
  serial, and a single point of failure where one timeout loses everything.
  Each chunk re-pays the ~5 s probe, which is a negligible price for the
  parallelism and the smaller blast radius.
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

Resource sizing (see the EULER_* constants below)
---------------------------------------------------
Every figure is now measured ON EULER. The laptop numbers this script used
previously under-sized wall-clock by roughly 4x and are retained only as
provenance constants:
  * simulate_null_persite.py (job 10856979, COMPLETED): 4 alignments at 4
    threads in 3587 s of AliSim => ~897 s per alignment; MaxRSS 5.64 GB.
    The one-time reference-frequency probe is ~5 s. Memory scaling with
    --num-alignments is still UNMEASURED (that MaxRSS is for 4 alignments
    per invocation), which is a reason to keep chunks modest and to check
    the first chunk's MaxRSS as it lands.
  * score_null_replicate.py (job 10861338): ASR 8152.3 s + scoring 243.8 s
    = 8396.1 s per replicate at 2 threads; MaxRSS 3.42 GB. That job exited
    non-zero, but only at the post-scoring key check and only because it
    was deliberately pointed at a quarantined alignment with the wrong
    taxa; both compute stages ran to completion, so the timings and MaxRSS
    are valid. The two phases run sequentially within one task, so peak
    memory is the larger of the two, not their sum.
  * The Euler/laptop ratio is 4.0x for scoring and 3.7x for simulation --
    consistent, and an ordinary per-core speed difference rather than a
    misconfiguration. An earlier note in this file described the scoring
    slowdown as ~23x and "unexplained"; that came from comparing against a
    313 s standalone ASR figure which the laptop does not reproduce (the
    same work takes ~34 min there at concurrency 1). The 23x was an
    artifact of the baseline, not a real effect.
  * This workload is memory-bandwidth-bound, not core-bound: one IQ-TREE
    ASR process holds ~116% CPU whether run alone or alongside a second
    one, and two concurrent ASRs match one in throughput. That is why the
    scoring array requests 2 CPUs and throttles concurrency rather than
    packing tasks densely. AliSim, by contrast, does parallelise (~399%
    CPU sustained), which is why the simulate job requests 4.
  * Consequence for array sizing: because the bottleneck is shared memory
    bandwidth rather than core count, packing many scoring tasks onto one
    node will NOT deliver throughput proportional to how many are packed --
    it will just make every task on that node slower and raise the risk of
    a walltime kill (which wastes the whole task, not just the excess). See
    the array-throttle discussion next to ARRAY_THROTTLE_DEFAULT below for
    how this script responds to that (short version: a lower default
    concurrency cap, not `--exclusive`, and not a guessed
    `--ntasks-per-node`, since neither of the latter two is a value this
    repo can derive without Euler's own node topology).
  * Replicate count: a separate measurement established that positions are
    statistically independent within a replicate
    (Var_r(V_r)/sum_p Var_r(K_rp) = 1.013, mean pairwise correlation
    +0.0021). That means the noisiest downstream quantity -- a
    90th-percentile FDP estimated from only 40 replicates -- can be
    tightened by block-bootstrapping positions rather than by buying more
    replicates outright. More replicates still help (they improve
    per-position tail estimation, currently ~20 null events per position),
    but the array does not need to be enormous: a few hundred replicates,
    not the low thousands, is the right order of magnitude to submit here.
Two things remain unverified and must be checked as the first tasks land
rather than after the run completes:
  * simulate memory at the configured chunk size (see above -- AliSim's
    scaling with --num-alignments has not been measured);
  * scoring behaviour when several tasks share one node's memory bus. The
    per-replicate figure above is from a single task on a single node, so a
    2-task pilot of the scoring array is still required, checked with
    `seff <jobid>` or
    `sacct --format=MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS,NodeList`.
Neither of these is a claim about correctness of the shipped numbers; they
are the two places where the shipped numbers could still be wrong.
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
# --- Euler-native measurements ----------------------------------------
# Every figure below was measured ON EULER, not extrapolated from the
# laptop. The laptop numbers that used to size these scripts are retained
# further down purely as provenance; they were wrong by ~4x and are no
# longer used arithmetically.
#
# Simulation. Two Euler measurements, and they disagree:
#   * job 10856979 (COMPLETED, 4 alignments at 4 threads): AliSim 3587 s
#     total => ~897 s per alignment; MaxRSS 5.64 GB.
#   * job 10868280 (TIMEOUT at 39 min, 2 alignments at 4 threads, node
#     eu-a2p-411): AliSim setup + the FIRST alignment alone took ~1350 s
#     (job start 10:02:45, sim_1 written 10:25); the second was still
#     running when the walltime killed it. MaxRSS 5.62 GB.
# The 897 s average is therefore not safe to size from -- it either
# amortised AliSim's one-time model/tree setup over 4 alignments or ran on
# a faster node. Until a clean per-alignment cadence is measured (read the
# sim_*.fa mtimes of the first chunk to complete), size from the 1350 s
# figure and apply it to EVERY alignment in a chunk. That deliberately
# over-counts setup once per alignment, and is the conservative direction:
# a walltime kill wastes the whole chunk.
SIMULATE_EULER_SECONDS_PER_ALIGNMENT = 1350
SIMULATE_EULER_PROBE_SECONDS = 5
SIMULATE_EULER_MEM_GB = 5.64
SIMULATE_THREADS = 4  # AliSim parallelises well (~399% CPU sustained). The
                      # earlier pilot 10854059 TIMED OUT producing zero
                      # alignments precisely because this was 1 while the
                      # walltime came from a 4-thread measurement.

# Scoring, job 10861338 (2 threads, Euler-native):
# [asr] 8152.3 s + [score] 243.8 s = 8396.1 s per replicate; MaxRSS 3.42 GB.
# That job FAILED, but only at the post-scoring key check, and only because
# it was deliberately pointed at a quarantined alignment whose taxa were
# wrong. Both stages ran to completion first, so the timings and MaxRSS are
# valid measurements of the real work.
SCORE_EULER_ASR_SECONDS = 8152.3
SCORE_EULER_SCORING_SECONDS = 243.8
SCORE_EULER_MEM_GB = 3.42
SCORE_THREADS = 2  # matches the -T 2 the measurement above was taken at,
                    # and score_null_replicate.py's own --threads default.

# --- Superseded laptop measurements (provenance only) -----------------
# Kept so the ~4x Euler/laptop ratio stays legible and so nobody
# re-derives sizing from them by accident. NOT used in any arithmetic.
# The 313 s standalone ASR figure in particular is not reproducible: the
# same work takes ~34 min on the laptop at concurrency 1 (measured from
# the batch2 replicate cadence), which is what makes Euler 4.0x slower
# rather than the 23x that comparing against 313 s implied.
SIMULATE_LAPTOP_MEASURED_SECONDS = 197
SCORE_LAPTOP_ASR_STANDALONE_SECONDS = 313
SCORE_LAPTOP_LOADED_MINUTES_HIGH = 46

# Small, explicitly-flagged safety margins -- not "just in case" padding.
# Both walltime factors are modest now that the base figures are
# Euler-native rather than cross-hardware extrapolations; they cover
# ordinary run-to-run and node-to-node variance only. The simulate memory
# headroom is the larger of the two because SIMULATE_EULER_MEM_GB was
# measured at 4 alignments per invocation and AliSim's memory scaling with
# --num-alignments has NOT been measured -- which is the main reason to
# keep chunks modest rather than simulating everything in one task.

# Raised from 1.3 after observing the limit: of 30 chunks at 04:52:30, task
# 22 timed out at 04:53:24 while the other 29 completed. Same node-to-node
# variability the scoring array shows. 1.6 x (10 x 1350 s + probe) = 6:00:08.
# A timed-out chunk is recoverable (the staging rename means it promotes no
# alignment at all, so nothing invalid is left behind and it can simply be
# resubmitted), which is why this factor stays below the scoring array's.
SIMULATE_WALLTIME_SAFETY_FACTOR = 1.6
SIMULATE_MEM_HEADROOM_GB = 1.0

# Sized for CO-LOCATED tasks, which is the case that actually fails.
#
# Measured on job 10920406, and the pattern is exact: every task that ran
# alone on a node COMPLETED (2:06:38 on eu-a2p-417, 1:56:53 on 344, 1:50:12
# on 485), and every task that shared a node with a sibling TIMED OUT
# (tasks 2+3 on eu-a2p-409, tasks 5+6 on eu-a2p-430, all four still
# unfinished at 4:26). Two ASR processes on one node more than double each
# other's wall-clock -- the workload is memory-bandwidth-bound, and a node's
# memory bus is shared even though its cores are not.
#
# An earlier revision of this comment claimed the bandwidth effect was a
# laptop artifact that "does not transfer to a cluster" because SLURM
# spreads tasks across nodes. That was wrong: SLURM spreads them, but it
# also PACKS two onto one node whenever that fits, which is exactly the
# harmful case.
#
# --exclusive would prevent it outright but reserves ~128 cores to run a
# 2-thread task: 25-60x the allocation of simply tolerating the contention.
# So the request stays at 2 CPUs and the walltime covers the slow case
# instead: 2.7 x 8,396 s = 6:17:49, i.e. ~3x the solo time and comfortably
# past the 4:26 at which co-located tasks were still running.
SCORE_WALLTIME_SAFETY_FACTOR = 2.7
SCORE_MEM_HEADROOM_GB = 0.6

# Alignments produced per simulate array task. At the conservative 1350 s
# each this is ~4.9 h per chunk including the safety factor -- well inside
# Euler's limits, more chunks running in parallel, and a timeout costs 10
# alignments rather than the whole run. Lowered from 25 after job 10868280
# showed the per-alignment cost was ~1.5x what the 4-alignment average
# implied: with an uncertain rate, smaller chunks are the cheaper mistake.
SIMULATE_CHUNK_SIZE_DEFAULT = 10

# Chunks must not share a random stream. Consecutive seeds would very
# likely be fine, but a large prime stride costs nothing and removes the
# question entirely.
CHUNK_SEED_STRIDE = 100003

# Default SLURM array throttle (the `%N` in `--array=0-K%N`), i.e. the max
# number of scoring tasks allowed to run concurrently.
#
# This was previously 10 (and submissions used 5) on the grounds that the
# workload is memory-bandwidth-bound, so packing tasks would not scale
# throughput. That measurement was taken on a LAPTOP -- one machine, one
# memory bus, where two concurrent ASRs really do match one in throughput.
# It does not transfer to a cluster: SLURM spreads array tasks across
# independent nodes (observed: eu-a2p-279, 344, 409, 417, 430 for five
# tasks), and separate nodes have separate memory buses, so the contention
# the low cap was protecting against mostly does not arise.
#
# Co-location still happens and may still cost something -- two tasks on
# eu-a2p-409 were slower than one alone on eu-a2p-279 (2:40+ vs 1:41) --
# but node-to-node speed varies by 2x on identical work anyway, so that
# comparison cannot separate the two effects, and throttling the whole
# array to guard against it wastes most of the cluster. Raise this further
# to suit the project's fair-share allocation; it is a citizenship choice,
# not a throughput optimum.
ARRAY_THROTTLE_DEFAULT = 20


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
    array_throttle: int = ARRAY_THROTTLE_DEFAULT,
    simulate_chunk_size: int = SIMULATE_CHUNK_SIZE_DEFAULT,
    run_subdir: str = "run",
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
    if simulate_chunk_size < 1:
        raise ValueError("simulate_chunk_size must be >= 1")

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
    remote_run_dir = f"{project_root_remote}/results/badasp_scoring/null_calibration/{run_subdir}"
    remote_logs_dir = f"{project_root_remote}/results/badasp_scoring/null_calibration/logs"

    # A non-default run_subdir (e.g. a pilot) gets its own package directory
    # and job names so it can coexist with the real run rather than
    # overwriting its scripts or being indistinguishable in squeue.
    suffix = "" if run_subdir == "run" else f"_{run_subdir}"
    package_dir = (root / "results" / "badasp_scoring" / "null_calibration"
                   / f"euler_package{suffix}")
    package_dir.mkdir(parents=True, exist_ok=True)

    # --- Simulate job (array, one task per chunk of alignments) -----------
    # Chunked rather than one long job for three reasons: 300 alignments in a
    # single task is ~75 h (over Euler's limit), it wastes the parallelism
    # AliSim cannot provide within one invocation, and a timeout would lose
    # every alignment rather than one chunk's worth.
    n_chunks = -(-num_replicates // simulate_chunk_size)  # ceil
    simulate_walltime = _fmt_hms(
        (SIMULATE_EULER_SECONDS_PER_ALIGNMENT * simulate_chunk_size
         + SIMULATE_EULER_PROBE_SECONDS) * SIMULATE_WALLTIME_SAFETY_FACTOR
    )
    # SLURM rejects a decimal --mem-per-cpu ("5.0G"); it must be an integer
    # plus a unit, so this is always emitted in whole MB.
    simulate_mem_per_cpu_mb = round(
        (SIMULATE_EULER_MEM_GB + SIMULATE_MEM_HEADROOM_GB) * 1024 / SIMULATE_THREADS
    )
    simulate_array_spec = f"0-{n_chunks - 1}%{array_throttle}"
    simulate_script = package_dir / "run_simulate_null_calibration.sh"
    simulate_script.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name=null_calib_simulate{suffix}\n"
        f"#SBATCH --array={simulate_array_spec}\n"
        f"#SBATCH --output={remote_logs_dir}/simulate_%A_%a.out\n"
        f"#SBATCH --error={remote_logs_dir}/simulate_%A_%a.err\n"
        f"#SBATCH --time={simulate_walltime}\n"
        f"#SBATCH --cpus-per-task={SIMULATE_THREADS}\n"
        f"#SBATCH --mem-per-cpu={simulate_mem_per_cpu_mb}M\n"
        "\n"
        f"# {n_chunks} array tasks x {simulate_chunk_size} alignments = "
        f"{n_chunks * simulate_chunk_size} (>= {num_replicates} requested).\n"
        f"# Sized from Euler job 10856979: {SIMULATE_EULER_SECONDS_PER_ALIGNMENT} s per\n"
        f"# alignment at {SIMULATE_THREADS} threads, MaxRSS {SIMULATE_EULER_MEM_GB} GB.\n"
        "#\n"
        f"# --cpus-per-task is {SIMULATE_THREADS}, NOT 1: AliSim sustains ~399% CPU, and\n"
        "# the walltime above is a 4-thread measurement. Pilot 10854059 requested 1 CPU\n"
        "# against a 4-thread-derived walltime and TIMED OUT with zero alignments.\n"
        "#\n"
        "# CAVEAT: MaxRSS was measured at 4 alignments per invocation. AliSim's memory\n"
        f"# scaling with --num-alignments is UNMEASURED, so a {simulate_chunk_size}-alignment\n"
        "# chunk may need more. Check `sacct -j <id> --format=MaxRSS` on the first chunk\n"
        "# to land and re-size before trusting the rest.\n"
        "\n"
        "module load stack/2025-06 gcc/12.2.0\n"
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
        "\n"
        f"PROJECT_ROOT={project_root_remote}\n"
        'export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"\n'
        f"RUN_DIR={remote_run_dir}\n"
        "CHUNK=$SLURM_ARRAY_TASK_ID\n"
        # Each chunk writes into its own directory so the per-chunk AliSim
        # outputs (always numbered sim_1..sim_N) cannot collide.
        'CHUNK_DIR="$RUN_DIR/sim/chunk_${CHUNK}"\n'
        'mkdir -p "$CHUNK_DIR" "' + remote_logs_dir + '"\n'
        "\n"
        # Distinct stream per chunk -- see CHUNK_SEED_STRIDE.
        f"SEED=$(( {seed} + CHUNK * {CHUNK_SEED_STRIDE} ))\n"
        "\n"
        'echo "Starting simulate chunk $CHUNK (seed $SEED) on $(hostname) at $(date)"\n'
        "\n"
        '$PROJECT_ROOT/venv/bin/python "$PROJECT_ROOT/scripts/simulate_null_persite.py" \\\n'
        f'  --composition-alignment "{remote_alignment}" \\\n'
        f'  --sim-tree "{remote_sim_tree}" \\\n'
        '  --out-prefix "$CHUNK_DIR/sim" \\\n'
        f"  --shrinkage {shrinkage} \\\n"
        f"  --num-alignments {simulate_chunk_size} \\\n"
        '  --seed $SEED \\\n'
        "  --threads $SLURM_CPUS_PER_TASK \\\n"
        "  --redo\n"
        "\n"
        'echo "Simulate chunk $CHUNK finished at $(date)"\n',
        encoding="utf-8",
    )
    simulate_script.chmod(0o755)

    # --- Scoring job (array, one task per replicate) -----------------------
    # Walltime is sized from the Euler-native per-replicate measurement
    # (job 10861338), superseding both the standalone laptop sum and the
    # loaded-laptop figure that replaced it.
    score_walltime = _fmt_hms(
        (SCORE_EULER_ASR_SECONDS + SCORE_EULER_SCORING_SECONDS) * SCORE_WALLTIME_SAFETY_FACTOR
    )
    score_mem_gb = SCORE_EULER_MEM_GB + SCORE_MEM_HEADROOM_GB
    score_mem_per_cpu_mb = round(score_mem_gb * 1024 / SCORE_THREADS)
    array_spec = f"0-{num_replicates - 1}%{array_throttle}"
    score_script = package_dir / "run_score_null_calibration_array.sh"
    score_script.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name=null_calib_score{suffix}\n"
        f"#SBATCH --array={array_spec}\n"
        f"#SBATCH --output={remote_logs_dir}/score_%A_%a.out\n"
        f"#SBATCH --error={remote_logs_dir}/score_%A_%a.err\n"
        f"#SBATCH --time={score_walltime}\n"
        f"#SBATCH --cpus-per-task={SCORE_THREADS}\n"
        f"#SBATCH --mem-per-cpu={score_mem_per_cpu_mb}M\n"
        "\n"
        "# These numbers are EULER-NATIVE, measured on a Euler compute node by job\n"
        f"# 10861338: [asr] {SCORE_EULER_ASR_SECONDS} s + [score] {SCORE_EULER_SCORING_SECONDS} s\n"
        f"# = {SCORE_EULER_ASR_SECONDS + SCORE_EULER_SCORING_SECONDS:.0f} s per replicate at "
        f"{SCORE_THREADS} threads, MaxRSS {SCORE_EULER_MEM_GB} GB. They supersede the\n"
        "# earlier laptop extrapolations, which under-estimated wall-clock by ~4x.\n"
        "# For context, Euler is ~4.0x slower per replicate than the laptop at the same\n"
        "# thread count -- the same ratio AliSim shows (3.7x), i.e. an ordinary\n"
        "# per-core speed difference and not a misconfiguration.\n"
        "#\n"
        "# A 2-task pilot is still worth running before the full array, because these\n"
        "# figures come from a SINGLE task on a SINGLE node and say nothing about what\n"
        "# happens when several tasks share one node's memory bus -- and this is a\n"
        "# memory-bandwidth-bound workload, so that is the contention that matters:\n"
        "#   sbatch --array=0-1 run_score_null_calibration_array.sh\n"
        "#   seff <jobid>\n"
        "#   sacct --format=MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS,NodeList -j <jobid>\n"
        "# (NodeList shows whether the two tasks co-located, which is the scenario the\n"
        "# throttle below protects against; if they did and each ran much slower than\n"
        f"# {SCORE_EULER_ASR_SECONDS + SCORE_EULER_SCORING_SECONDS:.0f} s, lower the throttle rather than raising the walltime.)\n"
        "#\n"
        f"# Per-task sizing: {SCORE_THREADS} CPUs (matches score_null_replicate.py's own\n"
        f"# -T {SCORE_THREADS}); mem-per-cpu = ({SCORE_EULER_MEM_GB} GB measured peak +\n"
        f"# {SCORE_MEM_HEADROOM_GB} GB headroom) / {SCORE_THREADS} CPUs. Walltime is the measured\n"
        f"# per-replicate total x {SCORE_WALLTIME_SAFETY_FACTOR} for run-to-run and node-to-node variance.\n"
        "#\n"
        "# Node-packing note: the array throttle below (the `%N` suffix) caps how many\n"
        "# scoring tasks run concurrently CLUSTER-WIDE; it does not by itself stop\n"
        "# several of them from landing on, and jointly saturating, the SAME node\n"
        "# (SLURM's own bin-packing decides that, and this repo has no way to measure\n"
        "# Euler's node topology to control it more precisely without the pilot\n"
        "# above). Deliberately not using `--exclusive` here: it would guarantee no\n"
        "# same-node contention but reserves an entire node for a 2-thread task,\n"
        "# which is a large, unjustified resource request for the throughput gained.\n"
        "# `--ntasks-per-node` is also not the right lever: each array index is an\n"
        "# independent single-task job, and that flag governs placement within one\n"
        "# multi-task job, not cross-job packing across an array. If the pilot's\n"
        "# NodeList shows harmful same-node stacking, escalate by lowering the\n"
        "# throttle further (or, as a last resort, switching to `--exclusive`) rather\n"
        "# than assuming more parallelism will help -- it will not, for a\n"
        "# bandwidth-bound workload like this one.\n"
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
        # The simulate array writes chunk C's alignments to sim/chunk_C/ as
        # sim_1..sim_<chunk_size>, so a global replicate index maps to a
        # (chunk, position-within-chunk) pair.
        f"CHUNK=$(( IDX / {simulate_chunk_size} ))\n"
        f"CHUNK_POS=$(( IDX % {simulate_chunk_size} + 1 ))\n"
        'SIM_FA="$RUN_DIR/sim/chunk_${CHUNK}/sim_${CHUNK_POS}.fa"\n'
        'WORK="$RUN_DIR/scratch/rep_${REP_ID}_work"\n'
        "\n"
        'if [ ! -f "$SIM_FA" ]; then\n'
        '  echo "Missing simulated alignment $SIM_FA -- did simulate chunk $CHUNK '
        'succeed? Check its log before resubmitting this index." >&2\n'
        "  exit 1\n"
        "fi\n"
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
        f'echo "Submitted simulate array job: $SIM_JOBID ({n_chunks} chunks x '
        f'{simulate_chunk_size} alignments)"\n'
        "\n"
        # afterok on an array job id waits for EVERY task in the array to
        # succeed, which is what we want: a scoring index whose chunk failed
        # has no alignment to read.
        "SCORE_JOBID=$(sbatch --parsable --dependency=afterok:$SIM_JOBID run_score_null_calibration_array.sh)\n"
        'echo "Submitted scoring array job: $SCORE_JOBID (depends on all of $SIM_JOBID)"\n'
        'echo "Remember: check the first simulate chunk\'s MaxRSS (its memory scaling with --num-alignments is unmeasured) and run a 2-task pilot of the scoring array before trusting either at scale."\n',
        encoding="utf-8",
    )
    submit_script.chmod(0o755)

    # --- Input checksum script -------------------------------------------
    # Every input the scoring array reads, in one place, so drift between the
    # laptop and Euler is a 5-second check rather than a 1.2-hour failure.
    verify_script = package_dir / "verify_euler_inputs.sh"
    remote_samples = (
        f"{project_root_remote}/results/reconciliation/alerax/IPR019888/"
        "reconciliations/all/IPR019888_samples.newick"
    )
    verify_script.write_text(
        "#!/bin/bash\n"
        "# Print md5 + size for every input the null-calibration jobs read.\n"
        "# Run on BOTH machines and diff the output. An input that differs\n"
        "# silently changes results rather than raising an error -- the\n"
        "# AleRax samples file in particular decides which nodes are scored\n"
        "# at all, and a stale copy cut the scored node pairs to 59% without\n"
        "# any error until the post-ASR key check.\n"
        "#\n"
        "# Usage: verify_euler_inputs.sh [project root]\n"
        f"# Defaults to {project_root_remote} (Euler); pass your local repo\n"
        "# root when running it on the laptop. md5sum and md5 are both handled.\n"
        "set -u\n"
        f'ROOT="${{1:-{project_root_remote}}}"\n'
        'if [ ! -d "$ROOT" ]; then\n'
        '  echo "No such project root: $ROOT" >&2\n'
        '  echo "Usage: $0 [project root]" >&2\n'
        "  exit 2\n"
        "fi\n"
        'cd "$ROOT" || exit 2\n'
        'echo "# project root: $ROOT"\n'
        "for f in \\\n"
        f"  {paths.get('trimmed_fasta', 'data/interim/IPR019888_trimmed.aln')} \\\n"
        f"  {nc.get('sim_tree', 'data/interim/iqtree_asr/IPR019888.treefile')} \\\n"
        "  data/interim/iqtree_asr/IPR019888_rooted.tree \\\n"
        "  results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk \\\n"
        "  results/reconciliation/alerax/IPR019888/reconciliations/all/IPR019888_samples.newick \\\n"
        "  results/badasp_scoring/raw_node_scores.csv \\\n"
        "; do\n"
        '  if [ -f "$f" ]; then\n'
        '    printf "%-78s %s %s\\n" "$f" "$( (md5sum "$f" 2>/dev/null || md5 -q "$f") | cut -c1-32 )" "$(wc -c <"$f" | tr -d " ")"\n'
        "  else\n"
        '    printf "%-78s %s\\n" "$f" "MISSING"\n'
        "  fi\n"
        "done\n",
        encoding="utf-8",
    )
    verify_script.chmod(0o755)

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
        # This one is easy to miss because no script names it on the command
        # line -- score_tree_nodes derives it from --reconciled-tree's own
        # directory. Omitting it does NOT cause an error: a stale copy just
        # relabels nodes, and every node whose label is not Speciation/
        # Duplication/Transfer is silently dropped from scoring
        # (src/badasp/scoring.py:412). On Euler a stale copy cut the scored
        # node pairs from 1,607 to 946 (59%), which only surfaced ~1.2 h
        # later at the post-ASR key check.
        "     rsync -avz --relative results/reconciliation/alerax/IPR019888/reconciliations/all/IPR019888_samples.newick "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        "     rsync -avz --relative results/badasp_scoring/raw_node_scores.csv "
        f"lucla@euler.ethz.ch:{project_root_remote}/\n"
        "\n"
        "3. Confirm both machines see identical inputs BEFORE submitting:\n"
        "     bash verify_euler_inputs.sh          # on Euler\n"
        "     bash verify_euler_inputs.sh          # and locally, then diff\n"
        "   A differing input does not raise an error -- it silently changes\n"
        "   which nodes are scored. Do not skip this.\n"
        "\n"
        "4. Extract this tarball's contents into the Euler project root and run:\n"
        "     bash submit_null_calibration.sh\n"
        "\n"
        "5. Resource sizing. Both sbatch scripts now carry EULER-NATIVE numbers\n"
        "   (simulate: job 10856979; score: job 10861338), not the earlier laptop\n"
        "   extrapolations, which were wrong by ~4x. Two things are still unmeasured\n"
        "   and should be checked as the first tasks land rather than after the run:\n"
        f"     a. Simulate memory at {simulate_chunk_size} alignments per chunk. MaxRSS was measured\n"
        "        at 4 alignments per invocation; AliSim's scaling with --num-alignments\n"
        "        is unknown. Check the first chunk to finish:\n"
        "          sacct -j <simjobid> --format=JobID%18,State,Elapsed,MaxRSS,ReqMem\n"
        "     b. Scoring under same-node contention. The measurement is from a single\n"
        "        task on a single node. Submit a 2-task pilot first and check\n"
        "        `seff <jobid>` / `sacct --format=MaxRSS,TotalCPU,Elapsed,ReqMem,\n"
        "        ReqCPUS,NodeList` -- NodeList shows whether the two tasks co-located,\n"
        "        which is what the array throttle protects against for this\n"
        "        memory-bandwidth-bound workload. If co-located tasks run much slower,\n"
        "        lower the throttle rather than raising the walltime.\n"
        "\n"
        "6. Once the array completes, aggregate on Euler or after rsyncing "
        f"{remote_run_dir}/npz/ back:\n"
        "     python scripts/calibrate_switch_threshold.py --null-run-dir <run dir> "
        "--observed-scores <observed CSV>\n",
        encoding="utf-8",
    )

    tar_path = (root / "results" / "badasp_scoring" / "null_calibration"
                / f"null_calibration_euler_bundle{suffix}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in (simulate_script, score_script, submit_script, verify_script, instructions_path):
            tar.add(f, arcname=f"null_calibration_euler/{f.name}")

    print(f"Wrote {simulate_script}")
    print(f"Wrote {score_script}")
    print(f"Wrote {submit_script}")
    print(f"Wrote {verify_script}")
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
                        help="Default: config null_calibration.replicates. Positions "
                             "within a replicate were measured to be statistically "
                             "independent (Var_r(V_r)/sum_p Var_r(K_rp) = 1.013), so "
                             "the noisiest downstream quantity is better tightened by "
                             "block-bootstrapping positions than by adding replicates -- "
                             "a few hundred is the right order of magnitude to submit "
                             "here, not the config default's low thousands; pass this "
                             "flag explicitly to override it.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Default: config null_calibration.seed.")
    parser.add_argument("--shrinkage", type=float, default=None,
                        help="Default: config null_calibration.shrinkage.")
    parser.add_argument("--asr-model", default=None,
                        help="Default: config null_calibration.asr_model.")
    parser.add_argument("--min-clade", type=int, default=5)
    parser.add_argument("--node-naming", choices=["legacy", "strict"], default="strict")
    parser.add_argument("--array-throttle", type=int, default=ARRAY_THROTTLE_DEFAULT,
                        help="Max concurrent array tasks cluster-wide (SLURM %% "
                             "syntax). This workload is memory-bandwidth-bound, so "
                             "high concurrency does not scale throughput and raises "
                             "the odds of harmful same-node stacking -- see "
                             "ARRAY_THROTTLE_DEFAULT's comment in this module for "
                             "the full reasoning. Not a measured value -- adjust to "
                             "your project's Euler allocation and to what the "
                             "pilot's `sacct ... NodeList` shows.")
    parser.add_argument("--simulate-chunk-size", type=int, default=SIMULATE_CHUNK_SIZE_DEFAULT,
                        help="Alignments produced per simulate array task. Trades "
                             "per-chunk walltime (~897 s per alignment on Euler) "
                             "against the number of alignments a single timeout "
                             "would lose. See SIMULATE_CHUNK_SIZE_DEFAULT.")
    parser.add_argument("--run-subdir", default="run",
                        help="Directory under results/badasp_scoring/null_calibration/ "
                             "that the jobs read alignments from and write .npz to. "
                             "Use a distinct name for a pilot so it cannot be "
                             "confused with, or skipped as already-done by, the real run.")
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
        simulate_chunk_size=args.simulate_chunk_size,
        run_subdir=args.run_subdir,
        project_root_remote=args.project_root_remote,
    )


if __name__ == "__main__":
    main()
