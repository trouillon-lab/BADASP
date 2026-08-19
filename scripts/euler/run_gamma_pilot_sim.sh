#!/bin/bash
#SBATCH --job-name=null_gamma_sim
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/gamma_pilot/gsim_%j.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/results/badasp_scoring/null_calibration/gamma_pilot/gsim_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1700M
#
# Purpose: test whether the null's missing among-site rate heterogeneity is the
# cause of its over-confident ancestral reconstruction.
#
# Measured problem this addresses: the real alignment's reconstruction fits a
# gamma shape alpha of 1.7211 (strong column-to-column rate variation), while
# reconstructions on the existing simulated null alignments fit 8.74-12.09
# (essentially no variation), because the production simulate run never passed
# --gamma-alpha and so added no rate heterogeneity at all. Across 10 scored
# alignments, the fitted alpha separates the noisy from the quiet datasets with no
# overlap (Spearman +0.78 vs P(AC=-1), p=0.0075) while no alignment-composition
# statistic separates them at all.
#
# This job changes exactly one thing: it passes --gamma-alpha, set to the value
# read out of the real alignment's own .iqtree report at run time so the number is
# never hardcoded.
#
# Sizing, measured not padded: the production simulate array (job 10880947) ran
# 01:42-04:19 for 10 alignments per task with max MaxRSS 5.45 GiB of the 6.64 GiB
# that 4 CPUs x 1700M grants. This task simulates 3 alignments, strictly less work,
# so the same CPU/memory request stands and 04:00:00 is below the 06:00:08 that
# array used. Raise only if this actually times out.
#
# Seed: keeps the project's documented convention seed = 20260731 + c * 100003
# with a reserved high index c = 100, so it cannot collide with production chunks
# 0-29.

set -uo pipefail
module load stack/2025-06 gcc/12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ROOT=/cluster/project/beltrao/lucla/repos/badasp
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
OUT=$ROOT/results/badasp_scoring/null_calibration/gamma_pilot
mkdir -p "$OUT"

# The observed alignment's fitted rate-heterogeneity parameter, read from the
# reconstruction report rather than pasted in, so the source is explicit.
ASR_REPORT=$ROOT/data/interim/iqtree_asr/IPR019888.iqtree
GAMMA_ALPHA=$(grep -im1 "Gamma shape alpha" "$ASR_REPORT" | awk '{print $NF}')
if [ -z "$GAMMA_ALPHA" ]; then
  echo "could not read 'Gamma shape alpha' from $ASR_REPORT" >&2; exit 1
fi
echo "gamma alpha read from $ASR_REPORT: $GAMMA_ALPHA"

RESERVED_INDEX=100
SEED=$(( 20260731 + RESERVED_INDEX * 100003 ))
N_ALIGNMENTS=3
echo "simulating $N_ALIGNMENTS alignments, seed $SEED, gamma alpha $GAMMA_ALPHA, on $(hostname) at $(date)"

$ROOT/venv/bin/python "$ROOT/scripts/simulate_null_persite.py" \
  --composition-alignment "$ROOT/data/interim/IPR019888_trimmed.aln" \
  --sim-tree "$ROOT/data/interim/iqtree_asr/IPR019888.treefile" \
  --out-prefix "$OUT/sim" \
  --shrinkage 0.15 \
  --num-alignments $N_ALIGNMENTS \
  --gamma-alpha "$GAMMA_ALPHA" \
  --seed $SEED \
  --threads $SLURM_CPUS_PER_TASK \
  --redo

echo "GAMMA_ALPHA_USED=$GAMMA_ALPHA SEED=$SEED" > "$OUT/sim_provenance.txt"
ls -la "$OUT"
