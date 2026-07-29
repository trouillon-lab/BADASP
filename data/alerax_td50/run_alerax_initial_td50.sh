#!/bin/bash
#SBATCH -J alerax_initial_td50
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=96:00:00
#SBATCH -o alerax_initial_td50_%j.out
#SBATCH -e alerax_initial_td50_%j.err

# AleRax initial run with T:D = 50:1 fixed rates.
#
# Motivation: Tria & Martin (2021) show empirical T:D ~50-100:1 for
# prokaryote signal transduction genes. Bremer et al. (2022) show that
# AleRax's default 1:1 prior systematically inflates duplications.
# We fix D = T_converged / 50 while keeping converged T and L.
#
# Rates (from unconstrained run):
#   D = 0.010595  (= T / 50)
#   T = 0.529761  (converged)
#   L = 0.815743  (converged)

set -euo pipefail

module load stack/2025-06 gcc/12.2.0 cmake/3.30.5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

REPO=/cluster/project/beltrao/lucla/repos/badasp

ALERAX=${REPO}/tools/AleRax/build/bin/alerax
FAMILIES=${REPO}/data/alerax_td50/IPR019888_initial_td50.families.txt
SPECIES_TREE=${REPO}/data/interim/alerax/IPR019888_species_tree.nwk
PARAM_FILE=${REPO}/data/alerax_td50/model_params_initial_td50.txt
OUTPUT_PREFIX=${REPO}/results/dtl_sensitivity/alerax_initial_td50/IPR019888

mkdir -p $(dirname ${OUTPUT_PREFIX})

echo "Starting AleRax initial-run T:D=50 job on $(hostname) at $(date)"
echo "Rates: D=0.010595, T=0.529761, L=0.815743 (fixed)"

${ALERAX} \
    -f ${FAMILIES} \
    -s ${SPECIES_TREE} \
    --rec-model UndatedDTL \
    --model-parametrization ${PARAM_FILE} \
    --fix-rates \
    -p ${OUTPUT_PREFIX} \
    --memory-savings \
    --prune-species-tree

echo "AleRax finished at $(date)"
