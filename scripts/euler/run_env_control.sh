#!/bin/bash
#SBATCH --job-name=null_calib_envctl
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/env_control/envctl_%j.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/env_control/envctl_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2300M
#
# Cross-environment control. Scores ONE alignment that has already been scored
# locally under IQ-TREE 2.3.6/macOS ARM, here under Euler's 2.4.0/Linux x86,
# with alignment, trees, observed table, model, min-clade, node-naming and
# thread count all held fixed. Isolates the ASR environment.
#
# Local arm (measured 2026-08-18): BEST SCORE FOUND -3877830.299,
# Total tree length 5740.694, P(AC=-1) 0.2684, atom 0.3994, exceedances 1314.
#
# Sizing: mem-per-cpu from the observed 3.87 GiB peak MaxRSS across COMPLETED
# scoring tasks plus the repo's 0.6 GiB headroom constant, over 2 CPUs.
# Walltime 08:00:00 = 1.4x the 05:41:59 observed maximum; the array script's
# 11:39:40 was sized for 10-20 self-co-located tasks, which a single task
# does not have.
#
# Deliberately does NOT delete the input alignment (the array wrapper does);
# this alignment is one of the ~268 unused ones and must survive for reuse.

module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:$PYTHONPATH"
OUT=$ROOT/results/badasp_scoring/null_calibration/env_control
mkdir -p "$OUT"

SIM_FA=$ROOT/results/badasp_scoring/null_calibration/run/sim/chunk_4/sim_1.fa
echo "Scoring $SIM_FA on $(hostname) at $(date)"
echo "alignment md5: $(md5sum $SIM_FA | cut -d\  -f1)  (expect d0d3b76ba5f668d2cbfaa379413cf23b)"
iqtree2 --version | head -1

$ROOT/venv/bin/python $ROOT/scripts/score_null_replicate.py \
  --sim-alignment "$SIM_FA" \
  --reconciled-tree "$ROOT/results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk" \
  --asr-tree "$ROOT/data/interim/iqtree_asr/IPR019888_rooted.tree" \
  --observed-scores "$ROOT/results/badasp_scoring/raw_node_scores.csv" \
  --out-npz "$OUT/env_control_euler.npz" \
  --model LG+G \
  --min-clade 5 \
  --node-naming strict \
  --threads $SLURM_CPUS_PER_TASK \
  --workdir "$OUT/work_euler" \
  --keep-intermediates
STATUS=$?
echo "exit=$STATUS at $(date)"
grep -E "BEST SCORE FOUND|Total tree length" "$OUT/work_euler/asr.log" 2>/dev/null
exit $STATUS
