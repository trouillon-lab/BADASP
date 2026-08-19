#!/bin/bash
#SBATCH --job-name=null_calib_invscan
#SBATCH --array=0-14%4
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/invocation_scan/inv_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/invocation_scan/inv_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2300M
#
# Purpose: extend the one-alignment-per-simulate-invocation sample from 10 to 25
# by scoring sim_1.fa of chunks 15..29, which are so far untouched.
#
# What it is for: the between-dataset spread of the null's false-positive count
# is dominated by a two-state pattern -- most simulated datasets sit at
# P(AC=-1) ~ 0.14 ("cold"), a minority at ~0.23 ("hot"), and the hot ones
# produce 2-3x more false switch calls. Every interval-valued error statement
# depends on the proportion of hot datasets, which is currently estimated from
# 5 events out of 31. These 15 draws each come from a distinct simulate seed,
# so they add 15 independent observations of that proportion.
#
# Sizing, measured not padded, unchanged from run_treelen_scan.sh:
#   mem-per-cpu 2300M  -- control job 11092279 peaked at 3.49 GiB of the 4.49 GiB
#                         this grants over 2 CPUs.
#   time 08:00:00      -- that control ran 01:59:20 uncontended; the largest
#                         observed co-located scoring task was 06:45:37.
#   array throttle %4  -- job 11179304 is running at %4 while this is queued, so
#                         %4 here caps combined concurrency at 8 tasks. Raise only
#                         after a task actually hits a limit.
#
# Does NOT delete the input alignment: these are part of the unused pool and must
# survive for reuse (the generated array wrapper's `rm -f "$SIM_FA"` is exactly
# what this script avoids). Keeps asr.log/.iqtree/.treefile so gamma alpha, total
# tree length and logL can be read out; deletes asr.state (~600 MB each).
#
# Each task writes its OWN one-line summary file rather than appending to a shared
# CSV, so 4 concurrent tasks cannot interleave. Concatenate at collection time.

module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:$PYTHONPATH"
OUT=$ROOT/results/badasp_scoring/null_calibration/invocation_scan
mkdir -p "$OUT"

FIRST_CHUNK=15
CHUNK=$(( SLURM_ARRAY_TASK_ID + FIRST_CHUNK ))
SIM_FA=$ROOT/results/badasp_scoring/null_calibration/run/sim/chunk_${CHUNK}/sim_1.fa
WORK=$OUT/work_c${CHUNK}
NPZ=$OUT/inv_chunk${CHUNK}_sim1.npz

if [ ! -f "$SIM_FA" ]; then echo "missing $SIM_FA" >&2; exit 1; fi
if [ -s "$NPZ" ]; then echo "chunk $CHUNK already scored ($NPZ); skipping."; exit 0; fi
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

LOGL=$(grep "BEST SCORE FOUND" "$WORK/asr.log" 2>/dev/null | awk '{print $NF}')
TLEN=$(grep "Total tree length" "$WORK/asr.log" 2>/dev/null | awk '{print $NF}')
ALPHA=$(grep -i "Gamma shape alpha" "$WORK/asr.iqtree" 2>/dev/null | head -1 | awk '{print $NF}')
SEED=$(grep -iE "^Seed:|random seed number" "$WORK/asr.log" 2>/dev/null | head -1 | tr -dc '0-9')

echo "RESULT chunk=$CHUNK status=$STATUS logL=$LOGL treelen=$TLEN alpha=$ALPHA seed=$SEED npz=$NPZ"
printf 'chunk,sim,status,logL,treelen,alpha,iqtree_seed\n%s,1,%s,%s,%s,%s,%s\n' \
  "$CHUNK" "$STATUS" "$LOGL" "$TLEN" "$ALPHA" "$SEED" > "$OUT/summary_c${CHUNK}.csv"

rm -f "$WORK/asr.state" "$WORK/asr.ckp.gz" "$WORK/null_scores.csv"
exit $STATUS
