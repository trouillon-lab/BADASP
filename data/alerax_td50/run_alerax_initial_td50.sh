#!/bin/bash
#SBATCH -J alerax_initial_td50
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=500G
#SBATCH --time=96:00:00
#SBATCH -o alerax_initial_td50_%j.out
#SBATCH -e alerax_initial_td50_%j.err

# AleRax initial run with T:D = 50:1 fixed rates.
#
# Motivation: Tria & Martin (2021) show empirical T:D ~50-100:1 for
# prokaryote signal transduction genes. Bremer et al. (2022) show that
# the AleRax default 1:1 prior systematically inflates duplications and
# biases root inference. We fix D = T_converged / 50 while keeping the
# converged T and L rates from the unconstrained run.
#
# Rates (from unconstrained run):
#   D = 0.010595  (= T / 50)
#   T = 0.529761  (converged)
#   L = 0.815743  (converged)

set -euo pipefail

REPO=/cluster/project/beltrao/lucla/repos/badasp

ALERAX=${REPO}/tools/AleRax/build/bin/alerax
FAMILIES=${REPO}/data/alerax_td50/IPR019888_initial_td50.families.txt
SPECIES_TREE=${REPO}/data/interim/alerax/IPR019888_species_tree.nwk
PARAM_FILE=${REPO}/data/alerax_td50/model_params_initial_td50.txt
OUTPUT_PREFIX=${REPO}/results/dtl_sensitivity/alerax_initial_td50/IPR019888

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
