#!/bin/bash
# Submits the simulate job, then the scoring array with a dependency on it.
# Run this from the Euler project root after `git pull`.
set -euo pipefail

SIM_JOBID=$(sbatch --parsable run_simulate_null_calibration.sh)
echo "Submitted simulate array job: $SIM_JOBID (30 chunks x 10 alignments)"

SCORE_JOBID=$(sbatch --parsable --dependency=afterok:$SIM_JOBID run_score_null_calibration_array.sh)
echo "Submitted scoring array job: $SCORE_JOBID (depends on all of $SIM_JOBID)"
echo "Remember: check the first simulate chunk's MaxRSS (its memory scaling with --num-alignments is unmeasured) and run a 2-task pilot of the scoring array before trusting either at scale."
