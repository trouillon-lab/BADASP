#!/bin/bash
#SBATCH --job-name=null_calib_simulate
#SBATCH --array=0-29%10
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/logs/simulate_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/logs/simulate_%A_%a.err
#SBATCH --time=06:00:08
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1700M

# 30 array tasks x 10 alignments = 300 (>= 300 requested).
# Sized from Euler job 10856979: 1350 s per
# alignment at 4 threads, MaxRSS 5.64 GB.
#
# --cpus-per-task is 4, NOT 1: AliSim sustains ~399% CPU, and
# the walltime above is a 4-thread measurement. Pilot 10854059 requested 1 CPU
# against a 4-thread-derived walltime and TIMED OUT with zero alignments.
#
# CAVEAT: MaxRSS was measured at 4 alignments per invocation. AliSim's memory
# scaling with --num-alignments is UNMEASURED, so a 10-alignment
# chunk may need more. Check `sacct -j <id> --format=MaxRSS` on the first chunk
# to land and re-size before trusting the rest.

module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"
RUN_DIR=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/run
CHUNK=$SLURM_ARRAY_TASK_ID
CHUNK_DIR="$RUN_DIR/sim/chunk_${CHUNK}"
mkdir -p "$CHUNK_DIR" "/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/logs"

SEED=$(( 20260731 + CHUNK * 100003 ))

echo "Starting simulate chunk $CHUNK (seed $SEED) on $(hostname) at $(date)"

$PROJECT_ROOT/venv/bin/python "$PROJECT_ROOT/scripts/simulate_null_persite.py" \
  --composition-alignment "/cluster/project/beltrao/lucla/repos/badasp/data/interim/IPR019888_trimmed.aln" \
  --sim-tree "/cluster/project/beltrao/lucla/repos/badasp/data/interim/iqtree_asr/IPR019888.treefile" \
  --out-prefix "$CHUNK_DIR/sim" \
  --shrinkage 0.15 \
  --num-alignments 10 \
  --seed $SEED \
  --threads $SLURM_CPUS_PER_TASK \
  --redo

echo "Simulate chunk $CHUNK finished at $(date)"
