#!/bin/bash
#SBATCH -J alerax_reccheck_td50
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --time=72:00:00
#SBATCH -o alerax_reccheck_td50_%j.out
#SBATCH -e alerax_reccheck_td50_%j.err

# AleRax rec_check run with T:D = 50:1 fixed rates.
#
# The rec_check uses all paralogs within species covering all CD-HIT
# clusters (19,524 sequences). Rates from unconstrained run:
#   D = 0.005500  (= T / 50)
#   T = 0.275005  (converged)
#   L = 0.876323  (converged)

set -euo pipefail

module load stack/2025-06 gcc/12.2.0 cmake/3.30.5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=/cluster/project/beltrao/lucla/repos/badasp

ALERAX=${REPO}/tools/AleRax/build/bin/alerax
FAMILIES=${REPO}/data/alerax_td50/IPR019888_reccheck_td50.families.txt
SPECIES_TREE=${REPO}/rec_check_euler/IPR019888_species_tree.nwk
PARAM_FILE=${REPO}/data/alerax_td50/model_params_reccheck_td50.txt
OUTPUT_PREFIX=${REPO}/results/dtl_sensitivity/alerax_reccheck_td50/IPR019888

mkdir -p $(dirname ${OUTPUT_PREFIX})

echo "Starting AleRax rec_check T:D=50 job on $(hostname) at $(date)"
echo "Rates: D=0.005500, T=0.275005, L=0.876323 (fixed)"

${ALERAX} \
    -f ${FAMILIES} \
    -s ${SPECIES_TREE} \
    --rec-model UndatedDTL \
    --model-parametrization ${PARAM_FILE} \
    --fix-rates \
    -p ${OUTPUT_PREFIX} \
    --prune-species-tree

echo "AleRax finished at $(date)"
