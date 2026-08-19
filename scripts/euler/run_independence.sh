#!/bin/bash
#SBATCH --job-name=null_calib_indep
#SBATCH --array=0-7%4
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/indep_test/ind_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/indep_test/ind_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2300M
#
# Independence-structure test. Question: is the "elevated" state a property of
# the individual simulated alignment, or of the simulate invocation that made it?
#
# Design: chunk_5/sim_1 measured ELEVATED (P(AC=-1)=0.2092); chunk_6/sim_1
# measured NORMAL (0.1336). Score 4 more alignments from each of those two
# invocations. If the state is per-invocation, chunk_5s siblings should be
# mostly elevated and chunk_6s mostly normal. If per-alignment, both should be
# mixed at the background rate (~20-30%).
#
# Motivation: the existing 31 replicates give ICC=0.185 across only 4
# invocations (F=2.76), implying effective n~14 of 31 -- but that ICC is itself
# imprecise at k=4. This test targets the question directly.
#
# Sizing: mem-per-cpu 2300M (control job peaked 3.49 GiB of 4.49 GiB granted).
# time 08:00:00 retained -- it accommodated the largest run observed so far
# (06:45:37, job 11145668_6); the limit has not been hit, so it is not raised.
# Throttle %4 (down from 5): the 06:45:37 case was two of our tasks sharing
# eu-a2p-281, so less self-stacking is the cheap mitigation.
#
# Does NOT delete input alignments. Drops asr.state after reading the numbers.

module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:$PYTHONPATH"
OUT=$ROOT/results/badasp_scoring/null_calibration/indep_test
mkdir -p "$OUT"

# tasks 0-3 -> chunk_5 sim_2..sim_5 ; tasks 4-7 -> chunk_6 sim_2..sim_5
if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
  CHUNK=5; SIM=$(( SLURM_ARRAY_TASK_ID + 2 ))
else
  CHUNK=6; SIM=$(( SLURM_ARRAY_TASK_ID - 4 + 2 ))
fi
SIM_FA=$ROOT/results/badasp_scoring/null_calibration/run/sim/chunk_${CHUNK}/sim_${SIM}.fa
WORK=$OUT/work_c${CHUNK}_s${SIM}
NPZ=$OUT/ind_chunk${CHUNK}_sim${SIM}.npz

if [ ! -f "$SIM_FA" ]; then echo "missing $SIM_FA" >&2; exit 1; fi
echo "chunk $CHUNK sim $SIM  on $(hostname) at $(date)"

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
echo "RESULT chunk=$CHUNK sim=$SIM status=$STATUS logL=$LOGL treelen=$TLEN"
echo "$CHUNK,$SIM,$STATUS,$LOGL,$TLEN" >> "$OUT/indep_summary.csv"
rm -f "$WORK/asr.state" "$WORK/asr.ckp.gz" "$WORK/null_scores.csv"
exit $STATUS
