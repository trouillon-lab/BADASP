# Euler job scripts

SLURM scripts for the null-calibration work on ETH Euler
(`/cluster/project/beltrao/lucla/repos/badasp`). Every one of these was
previously untracked and lived only on the cluster, so the runs behind the
project's numbers were not reproducible from this repo. They are committed here
as **the record of what actually ran**.

## Generated vs hand-written

`run_simulate_null_calibration.sh`, `run_score_null_calibration_array.sh`,
`submit_null_calibration.sh` and `verify_euler_inputs.sh` are emitted by
`scripts/package_null_calibration_for_euler.py` (see `:346`, `:413`, `:534`,
`:558`). The copies here were taken from Euler, not regenerated, because the
cluster copies are what produced the existing replicate data.

**One known divergence between the cluster copy and what the generator emits
today:** `run_simulate_null_calibration.sh` on Euler carries
`#SBATCH --array=0-29%10`, whereas the generator writes `%20`. The throttle was
edited on the cluster after generation. The copy here is the cluster one.

The remaining scripts are hand-written one-offs, each written for a single
experiment and not produced by any generator:

| script | what it was for |
| :-- | :-- |
| `verify_euler_inputs.sh` | md5 + size of the six shared inputs; run on both machines and diff before submitting |
| `run_env_control.sh` | scored one fixed alignment on Euler to test for an ASR environment effect (result: none — bit-identical to the local run) |
| `run_treelen_scan.sh` | scored `sim_1.fa` of chunks 5–14, one alignment per simulate invocation, recording inferred tree length and logL |
| `run_independence.sh` | scores `chunk_5/sim_2..5` and `chunk_6/sim_2..5` to test whether the elevated-noise state is a per-invocation or per-alignment property |
| `run_invocation_scan.sh` | extends the one-per-invocation sample to chunks 15–29 |
| `run_gamma_pilot_sim.sh`, `run_gamma_pilot_score.sh` | simulate and score 3 alignments with the observed among-site rate heterogeneity applied |
| `run_gamma_confirm.sh` | repeats the rate-heterogeneity treatment across 4 distinct simulate invocations |

## Conventions these scripts follow

- **Simulate seed for invocation `c` is `20260731 + c * 100003`**
  (`run_simulate_null_calibration.sh:33`). Reserved high indices 100–104 are used
  by the rate-heterogeneity experiments so they cannot collide with production
  chunks 0–29.
- **Replicate index → alignment** for the production array is
  `chunk = IDX / 10`, `position = IDX % 10 + 1`
  (`run_score_null_calibration_array.sh:78-79`). This mapping is recorded nowhere
  in the replicate `.npz` files, which is why `calibrate_switch_threshold.py`
  refuses to infer it and takes it via `--replicate-groups-csv` instead.
- **The production array deletes its input alignment** (`rm -f "$SIM_FA"`,
  `run_score_null_calibration_array.sh:106`). The one-off scripts deliberately do
  not, so alignments from the unused pool survive for reuse.
- Resource requests are sized from observed usage and documented in each script's
  header comment. Raise a limit only after a job actually hits it.
