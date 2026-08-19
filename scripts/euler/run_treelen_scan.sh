#!/bin/bash
#SBATCH --job-name=null_calib_treelen
#SBATCH --array=0-9%5
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/treelen_scan/tl_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/treelen_scan/tl_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2300M
#
# Purpose: turn a 2-point hunch into a measured relationship. For 10 previously
# unscored simulated alignments, record the ASR inferred total tree length and
# log-likelihood alongside P(AC=-1) and the p_AC atom, to test whether inferred
# tree length predicts the "ancestors differ" rate, and to estimate the
# between-alignment variance of the null properly.
#
# One alignment per distinct simulate seed (sim_1 of chunks 5..14) so the draws
# are not correlated through a shared simulate invocation.
#
# Sizing, measured not padded:
#   mem-per-cpu 2300M  -- control job 11092279 peaked at 3.49 GiB of the 4.49 GiB
#                         this grants over 2 CPUs.
#   time 08:00:00      -- that control ran 01:59:20 uncontended; the maximum
#                         observed among COMPLETED co-located scoring tasks was
#                         05:41:59. 08:00 is 1.4x that observed maximum, and well
#                         below the 11:39:40 the old array used.
#   array throttle %5  -- 11 tasks previously stacked on eu-a2p-279 and ran
#                         2.4-2.9x slower than solo. Throttling to 5 halves the
#                         stacking exposure for a 10-task array; the cost is
#                         wall-clock only, which is not the binding constraint.
#
# Does NOT delete the input alignment: these 10 are part of the ~268 unused pool
# and must survive for reuse. Keeps asr.log and asr.treefile, deletes asr.state
# (~600 MB each) and the checkpoint after reading the numbers out.

module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:$PYTHONPATH"
OUT=$ROOT/results/badasp_scoring/null_calibration/treelen_scan
mkdir -p "$OUT"

CHUNK=$(( SLURM_ARRAY_TASK_ID + 5 ))
SIM_FA=$ROOT/results/badasp_scoring/null_calibration/run/sim/chunk_${CHUNK}/sim_1.fa
WORK=$OUT/work_c${CHUNK}
NPZ=$OUT/tl_chunk${CHUNK}_sim1.npz

if [ ! -f "$SIM_FA" ]; then echo "missing $SIM_FA" >&2; exit 1; fi
echo "chunk $CHUNK  alignment $SIM_FA  on $(hostname) at $(date)"

$ROOT/venv/bin/python $ROOT/scripts/score_null_replicate.py \
  --sim-alignment "$SIM_FA" \
  --reconciled-tree "$ROOT/results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk" \
  --asr-tree "$ROOT/data/interim/iqtree_asr/IPR019888_rooted.tree" \
  --observed-scores "$ROOT/results/badasp_scoring/raw_node_scores.csv" \
  --out-npz "$NPZ" \
  --model LG+G --min-clade 5 --node-naming strict \
  --threads $SLURM_CPUS_PER_TASK \
  --workdir "$WORK" --keep-intermediates
STATUS=$?

LOGL=$(grep "BEST SCORE FOUND" "$WORK/asr.log" 2>/dev/null | awk "{print \$NF}")
TLEN=$(grep "Total tree length" "$WORK/asr.log" 2>/dev/null | awk "{print \$NF}")
echo "RESULT chunk=$CHUNK status=$STATUS logL=$LOGL treelen=$TLEN npz=$NPZ"
echo "$CHUNK,$STATUS,$LOGL,$TLEN" >> "$OUT/treelen_summary.csv"

# keep the small artefacts, drop the 600 MB state file and checkpoint
rm -f "$WORK/asr.state" "$WORK/asr.ckp.gz" "$WORK/null_scores.csv"
exit $STATUS
