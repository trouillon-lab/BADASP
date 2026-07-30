#!/bin/bash
#SBATCH --job-name=alerax_reccheck_td50
#SBATCH --output=/cluster/project/beltrao/lucla/repos/badasp/data/alerax_td50/alerax_reccheck_td50.%j.out
#SBATCH --error=/cluster/project/beltrao/lucla/repos/badasp/data/alerax_td50/alerax_reccheck_td50.%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G

# AleRax rec_check run with T:D=50:1 fixed rates.
# Rates from unconstrained rec_check run: T=0.275005, L=0.876323, D=0.354916
# D_fixed = T/50 = 0.005500; T and L kept at converged values.

module load stack/2025-06 gcc/12.2.0 cmake/3.30.5
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BINARY=/cluster/project/beltrao/lucla/repos/badasp/tools/AleRax/build/bin/alerax
FAMILIES=/cluster/project/beltrao/lucla/repos/badasp/data/alerax_td50/IPR019888_reccheck_td50.families.txt
SPECIES_TREE=/cluster/project/beltrao/lucla/repos/badasp/rec_check_euler/IPR019888_species_tree.nwk
OUTPUT_DIR=/cluster/project/beltrao/lucla/repos/badasp/results/dtl_sensitivity/alerax_reccheck_td50/IPR019888

mkdir -p $OUTPUT_DIR

echo "Starting AleRax rec_check T:D=50 job on $(hostname) at $(date)"

$BINARY \
    -f $FAMILIES \
    -s $SPECIES_TREE \
    --rec-model UndatedDTL \
    --d 0.005500 \
    --l 0.876323 \
    --t 0.275005 \
    --fix-rates \
    -p $OUTPUT_DIR \
    --prune-species-tree

echo "AleRax finished at $(date)"
