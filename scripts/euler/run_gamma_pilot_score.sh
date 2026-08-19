#!/bin/bash
#SBATCH --job-name=null_gamma_score
#SBATCH --array=1-3
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/gamma_pilot/gscore_%A_%a.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/gamma_pilot/gscore_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2300M
#
# Scores the 3 gamma-heterogeneity pilot alignments produced by
# run_gamma_pilot_sim.sh, keeping the workdir so the fitted gamma shape alpha,
# total tree length and log-likelihood can be read back out. The alpha is the
# primary readout: prediction P-D is that it lands near the real alignment's
# 1.7211 instead of the 8.74-12.09 the existing null replicates produce.
#
# Sizing identical to run_invocation_scan.sh, which is measured: control job
# 11092279 ran 01:59:20 with MaxRSS 3.49 GiB; largest observed co-located scoring
# task 06:45:37. Array is only 3 tasks so no throttle is needed.
#
# Does not delete the input alignment.

set -uo pipefail
module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:$PYTHONPATH"
OUT=$ROOT/results/badasp_scoring/null_calibration/gamma_pilot

IDX=$SLURM_ARRAY_TASK_ID
SIM_FA=$OUT/sim_${IDX}.fa
WORK=$OUT/work_s${IDX}
NPZ=$OUT/gamma_sim${IDX}.npz

if [ ! -f "$SIM_FA" ]; then echo "missing $SIM_FA" >&2; ls -la "$OUT" >&2; exit 1; fi
if [ -s "$NPZ" ]; then echo "already scored ($NPZ); skipping."; exit 0; fi
echo "scoring $SIM_FA on $(hostname) at $(date)"

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
TLEN=$(grep -im1 "Total tree length" "$WORK/asr.iqtree" 2>/dev/null | awk '{print $NF}')
ALPHA=$(grep -im1 "Gamma shape alpha" "$WORK/asr.iqtree" 2>/dev/null | awk '{print $NF}')
echo "RESULT sim=$IDX status=$STATUS alpha=$ALPHA logL=$LOGL treelen=$TLEN npz=$NPZ"
printf 'sim,status,alpha,logL,treelen\n%s,%s,%s,%s,%s\n' "$IDX" "$STATUS" "$ALPHA" "$LOGL" "$TLEN" \
  > "$OUT/summary_s${IDX}.csv"

rm -f "$WORK/asr.state" "$WORK/asr.ckp.gz" "$WORK/null_scores.csv"
exit $STATUS
