#!/bin/bash
#SBATCH --job-name=null_calib_score
#SBATCH --array=0-299%20
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/logs/score_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/logs/score_%A_%a.err
#SBATCH --time=11:39:40
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2058M

# These numbers are EULER-NATIVE, measured on a Euler compute node by job
# 10861338: [asr] 8152.3 s + [score] 243.8 s
# = 8396 s per replicate at 2 threads, MaxRSS 3.42 GB. They supersede the
# earlier laptop extrapolations, which under-estimated wall-clock by ~4x.
# For context, Euler is ~4.0x slower per replicate than the laptop at the same
# thread count -- the same ratio AliSim shows (3.7x), i.e. an ordinary
# per-core speed difference and not a misconfiguration.
#
# A 2-task pilot is still worth running before the full array, because these
# figures come from a SINGLE task on a SINGLE node and say nothing about what
# happens when several tasks share one node's memory bus -- and this is a
# memory-bandwidth-bound workload, so that is the contention that matters:
#   sbatch --array=0-1 run_score_null_calibration_array.sh
#   seff <jobid>
#   sacct --format=MaxRSS,TotalCPU,Elapsed,ReqMem,ReqCPUS,NodeList -j <jobid>
# (NodeList shows whether the two tasks co-located, which is the scenario the
# throttle below protects against; if they did and each ran much slower than
# 8396 s, lower the throttle rather than raising the walltime.)
#
# Per-task sizing: 2 CPUs (matches score_null_replicate.py's own
# -T 2); mem-per-cpu = (3.42 GB measured peak +
# 0.6 GB headroom) / 2 CPUs. Walltime is the measured
# per-replicate total x 5.0 for run-to-run and node-to-node variance.
#
# Node-packing note: the array throttle below (the `%N` suffix) caps how many
# scoring tasks run concurrently CLUSTER-WIDE; it does not by itself stop
# several of them from landing on, and jointly saturating, the SAME node
# (SLURM's own bin-packing decides that, and this repo has no way to measure
# Euler's node topology to control it more precisely without the pilot
# above). Deliberately not using `--exclusive` here: it would guarantee no
# same-node contention but reserves an entire node for a 2-thread task,
# which is a large, unjustified resource request for the throughput gained.
# `--ntasks-per-node` is also not the right lever: each array index is an
# independent single-task job, and that flag governs placement within one
# multi-task job, not cross-job packing across an array. If the pilot's
# NodeList shows harmful same-node stacking, escalate by lowering the
# throttle further (or, as a last resort, switching to `--exclusive`) rather
# than assuming more parallelism will help -- it will not, for a
# bandwidth-bound workload like this one.

module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"
RUN_DIR=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/run
mkdir -p "$RUN_DIR/npz" "$RUN_DIR/scratch" "/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/logs"

IDX=$SLURM_ARRAY_TASK_ID
REP_ID=$(printf "%04d" "$IDX")
NPZ="$RUN_DIR/npz/rep_${REP_ID}.npz"

# Resumable: reuse run_null_calibration.py's own .npz validity check (not
# reimplemented here) so a partially completed array can be resubmitted for
# just the missing/invalid indices without redoing finished replicates.
if [ -f "$NPZ" ]; then
  if $PROJECT_ROOT/venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, '$PROJECT_ROOT')
from scripts.run_null_calibration import is_valid_npz
sys.exit(0 if is_valid_npz(Path('$NPZ')) else 1)
"; then
    echo "Replicate $IDX already done ($NPZ); skipping."
    exit 0
  fi
fi

CHUNK=$(( IDX / 10 ))
CHUNK_POS=$(( IDX % 10 + 1 ))
SIM_FA="$RUN_DIR/sim/chunk_${CHUNK}/sim_${CHUNK_POS}.fa"
WORK="$RUN_DIR/scratch/rep_${REP_ID}_work"

if [ ! -f "$SIM_FA" ]; then
  echo "Missing simulated alignment $SIM_FA -- did simulate chunk $CHUNK succeed? Check its log before resubmitting this index." >&2
  exit 1
fi

echo "Starting replicate $IDX (alignment $SIM_FA) on $(hostname) at $(date)"

$PROJECT_ROOT/venv/bin/python "$PROJECT_ROOT/scripts/score_null_replicate.py" \
  --sim-alignment "$SIM_FA" \
  --reconciled-tree "/cluster/project/beltrao/lucla/repos/badasp/results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk" \
  --asr-tree "/cluster/project/beltrao/lucla/repos/badasp/data/interim/iqtree_asr/IPR019888_rooted.tree" \
  --observed-scores "/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/raw_node_scores.csv" \
  --out-npz "$NPZ" \
  --model LG+G \
  --min-clade 5 \
  --node-naming strict \
  --threads $SLURM_CPUS_PER_TASK \
  --workdir "$WORK"
STATUS=$?

# Delete the simulated alignment once attempted (success or failure), same
# policy as run_null_calibration.py's local driver: it is reproducible from
# the simulate job's (seed, num_replicates) alone.
rm -f "$SIM_FA"

echo "Replicate $IDX finished at $(date) with status $STATUS"
exit $STATUS
