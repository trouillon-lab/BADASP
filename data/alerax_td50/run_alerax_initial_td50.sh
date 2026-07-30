#!/bin/bash
#SBATCH --job-name=alerax_initial_td50
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/data/alerax_td50/alerax_initial_td50.%j.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/data/alerax_td50/alerax_initial_td50.%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256G

# AleRax initial run with T:D=50:1 fixed rates.
# Rates from unconstrained run (alerax_clean): T=0.529761, L=0.815743, D=0.0565401
# D_fixed = T/50 = 0.010595; T and L kept at converged values.

module load stack/2025-06 gcc/12.2.0 cmake/3.30.5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BINARY=/cluster/project/beltrao/lucla/repos/badasp/tools/AleRax/build/bin/alerax
FAMILIES=/cluster/project/beltrao/lucla/repos/badasp/data/alerax_td50/IPR019888_initial_td50.families.txt
SPECIES_TREE=/cluster/project/beltrao/lucla/repos/badasp/data/interim/alerax_clean/IPR019888_species_tree_clean.nwk
OUTPUT_DIR=/cluster/project/beltrao/lucla/repos/badasp/results/dtl_sensitivity/alerax_initial_td50/IPR019888

mkdir -p $OUTPUT_DIR

echo "Starting AleRax initial T:D=50 job on $(hostname) at $(date)"

$BINARY \
    -f $FAMILIES \
    -s $SPECIES_TREE \
    --rec-model UndatedDTL \
    --d 0.010595 \
    --l 0.815743 \
    --t 0.529761 \
    --fix-rates \
    -p $OUTPUT_DIR \
    --memory-savings \
    --prune-species-tree

echo "AleRax finished at $(date)"
