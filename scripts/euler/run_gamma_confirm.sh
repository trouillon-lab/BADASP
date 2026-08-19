#!/bin/bash
#SBATCH --job-name=null_gamma_confirm
#SBATCH --array=0-3%2
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/gamma_confirm/gc_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/gamma_confirm/gc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1700M
#
# Purpose: settle whether adding the observed among-site rate heterogeneity to the
# simulation really makes the null MORE over-confident, or whether the 3-replicate
# pilot just drew three hot datasets.
#
# The pilot (jobs 11188620/11188621) fixed the rate mis-specification as intended --
# fitted gamma alpha moved from 8.74-12.09 to 1.14-1.98 against the real alignment's
# 1.7211 -- but the p_AC==1 atom ROSE from 2.42x to 3.92x the observed value and the
# expected false-call count at 795 switch calls went from ~509 to ~748. All 3 pilot
# datasets came from ONE simulate invocation, so treatment and invocation are
# confounded there. Under the existing 41-replicate distribution the probability that
# 3 draws all land at or above the pilot's minimum is ~0.011 on the atom and on the
# false-call count -- suggestive, not settled.
#
# This job draws 4 replicates from 4 DISTINCT simulate invocations, one alignment
# each, so the invocation cannot explain the effect. Simulate and score run in the
# same task because both are short under this treatment.
#
# Sizing, from the pilot's own measured usage, not padded:
#   cpus 4 / mem-per-cpu 1700M -- simulate peaked at 5.45 GiB of the 6.64 GiB this
#                                 grants; scoring peaked at 3.49 GiB.
#   time 02:00:00              -- pilot simulate ran 00:31:04 and each scoring task
#                                 ~00:36, so ~67 min combined; this is 1.7x that.
#                                 Raise only if a task actually hits it.
#   throttle %2                -- jobs 11179304 and 11185529 are already running at
#                                 %4 each, so %2 here caps total concurrency at 10.
#
# Seeds keep the project's convention seed = 20260731 + c * 100003, using reserved
# high indices 101-104 so they cannot collide with production chunks 0-29 or with the
# pilot's index 100.

set -uo pipefail
module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
OUT=$ROOT/results/badasp_scoring/null_calibration/gamma_confirm
mkdir -p "$OUT"

# Read the real alignment's fitted rate-heterogeneity parameter from the
# reconstruction report rather than pasting the number in.
ASR_REPORT=$ROOT/data/interim/iqtree_asr/IPR019888.iqtree
GAMMA_ALPHA=$(grep -im1 "Gamma shape alpha" "$ASR_REPORT" | awk '{print $NF}')
if [ -z "$GAMMA_ALPHA" ]; then
  echo "could not read 'Gamma shape alpha' from $ASR_REPORT" >&2; exit 1
fi

RESERVED_INDEX=$(( 101 + SLURM_ARRAY_TASK_ID ))
SEED=$(( 20260731 + RESERVED_INDEX * 100003 ))
DIR=$OUT/inv_${RESERVED_INDEX}
mkdir -p "$DIR"
SIM_FA=$DIR/sim_1.fa
NPZ=$OUT/gamma_inv${RESERVED_INDEX}.npz
WORK=$DIR/work

echo "invocation index $RESERVED_INDEX  seed $SEED  gamma alpha $GAMMA_ALPHA  on $(hostname) at $(date)"

if [ -s "$NPZ" ]; then echo "already scored ($NPZ); skipping."; exit 0; fi

if [ ! -s "$SIM_FA" ]; then
  $ROOT/venv/bin/python "$ROOT/scripts/simulate_null_persite.py" \
    --composition-alignment "$ROOT/data/interim/IPR019888_trimmed.aln" \
    --sim-tree "$ROOT/data/interim/iqtree_asr/IPR019888.treefile" \
    --out-prefix "$DIR/sim" \
    --shrinkage 0.15 \
    --num-alignments 1 \
    --gamma-alpha "$GAMMA_ALPHA" \
    --seed $SEED \
    --threads $SLURM_CPUS_PER_TASK \
    --redo
  SIMSTATUS=$?
  if [ $SIMSTATUS -ne 0 ] || [ ! -s "$SIM_FA" ]; then
    echo "simulate failed (status $SIMSTATUS) or produced no $SIM_FA" >&2; exit 1
  fi
fi

$ROOT/venv/bin/python $ROOT/scripts/score_null_replicate.py \
  --sim-alignment "$SIM_FA" \
  --reconciled-tree "$ROOT/results/reconciliation/alerax/IPR019888/reconciliations/IPR019888.nwk" \
  --asr-tree "$ROOT/data/interim/iqtree_asr/IPR019888_rooted.tree" \
  --observed-scores "$ROOT/results/badasp_scoring/raw_node_scores.csv" \
  --out-npz "$NPZ" \
  --model LG+G --min-clade 5 --node-naming strict \
  --threads 2 \
  --workdir "$WORK" --keep-intermediates
STATUS=$?

LOGL=$(grep "BEST SCORE FOUND" "$WORK/asr.log" 2>/dev/null | awk '{print $NF}')
TLEN=$(grep -im1 "Total tree length" "$WORK/asr.iqtree" 2>/dev/null | awk '{print $NF}')
ALPHA=$(grep -im1 "Gamma shape alpha" "$WORK/asr.iqtree" 2>/dev/null | awk '{print $NF}')
echo "RESULT invocation=$RESERVED_INDEX seed=$SEED status=$STATUS alpha=$ALPHA logL=$LOGL treelen=$TLEN npz=$NPZ"
printf 'invocation,seed,status,alpha,logL,treelen\n%s,%s,%s,%s,%s,%s\n' \
  "$RESERVED_INDEX" "$SEED" "$STATUS" "$ALPHA" "$LOGL" "$TLEN" > "$OUT/summary_inv${RESERVED_INDEX}.csv"

rm -f "$WORK/asr.state" "$WORK/asr.ckp.gz" "$WORK/null_scores.csv"
exit $STATUS
