#!/usr/bin/env python3
"""Score one simulated null replicate end to end.

Pipeline per replicate, mirroring the observed-data path exactly:

  1. IQ-TREE ancestral state reconstruction on the simulated alignment, using
     the *same* command as the real run (fixed reconciled topology via ``-te``,
     same substitution model), so the null inherits the same reconstruction
     error and the same model misspecification as the observed scores.
  2. BADASP scoring with the same ``score_tree_nodes`` code path and the same
     ``--min-clade`` / ``--node-naming`` settings.
  3. Assert the resulting (node, position) key set is identical to the observed
     score table's. The null is only usable as a *matched* null if test i in a
     replicate refers to the same comparison as test i in the observed data.
  4. Persist a compact float array of the null scores and delete the large
     intermediates (an IQ-TREE ``.state`` for this dataset is ~600 MB, which is
     not storable across many replicates).

Only the leaf alignment from the simulator is used. Any true internal-node
sequences a simulator may emit are deliberately ignored: the observed score
depends on *estimated* ancestral states and their posterior probabilities, so a
null built from true ancestral states would contain no reconstruction error and
would not be comparable.

Nothing here is verified to reproduce any particular published result; the key
identity check in step 3 is an assertion about this run, not a claim about the
correctness of the underlying scores.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Columns persisted per replicate. These are the quantities the calibration
# needs; the full CSV is not kept because it does not fit for many replicates.
SCORE_COLUMNS = [
    "rc_left",
    "rc_right",
    "ac",
    "p_ac_left",
    "p_ac_right",
    "badasp_score_left",
    "badasp_score_right",
]
# A comparison is identified by its parent, its two children and the alignment
# position. ``node_name`` alone is not sufficient: the rooted tree produced by
# scripts/root_and_map_tree.py contains one duplicated pre-existing name
# (Node3106 is assigned to two distinct nodes by the leaf-set signature
# transfer, which its complement-matching step makes possible), so
# (node_name, position) has 338 duplicate rows while including the children is
# unique. Verified unique in both the observed and a simulated score table.
KEY_COLUMNS = ["node_name", "left_child", "right_child", "position"]


def run_asr(
    alignment: Path,
    reconciled_tree: Path,
    model: str,
    prefix: Path,
    threads: str,
    iqtree_binary: str,
    redo: bool = True,
) -> Path:
    """Run IQ-TREE ASR on ``alignment`` with the topology fixed to ``reconciled_tree``.

    Returns the path to the resulting ``.state`` file. The command is kept
    deliberately identical in shape to the pipeline's own ASR rule.
    """
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        iqtree_binary,
        "-s", str(alignment),
        "-m", model,
        "-te", str(reconciled_tree),
        "-asr",
        "--prefix", str(prefix),
        "-T", threads,
    ]
    if redo:
        cmd.append("-redo")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    state = prefix.with_suffix(prefix.suffix + ".state") if prefix.suffix else Path(f"{prefix}.state")
    if not state.exists():
        raise FileNotFoundError(f"IQ-TREE produced no .state file at {state}")
    return state


def score_replicate(
    state: Path,
    alerax_tree: Path,
    asr_tree: Path,
    alignment: Path,
    output_csv: Path,
    min_clade: int,
    node_naming: str,
) -> None:
    """Invoke the scoring module as a subprocess so the CLI path is the tested one."""
    cmd = [
        sys.executable, "-m", "badasp.scoring",
        "--alerax-tree", str(alerax_tree),
        "--asr-tree", str(asr_tree),
        "--state", str(state),
        "--alignment", str(alignment),
        "--output", str(output_csv),
        "--min-clade", str(min_clade),
        "--node-naming", node_naming,
    ]
    subprocess.run(cmd, check=True)


def check_keys(
    null_csv: Path, observed_csv: Path, max_missing_frac: float = 0.01
) -> pd.DataFrame:
    """Check the null key set against the observed one; return the null frame.

    The observed table defines the canonical set of tests, so:

    * **null-only keys are dropped.** These are comparisons the observed run
      skipped but the simulated run did not. They arise legitimately: IQ-TREE's
      ``--asr-min`` defaults to the equilibrium frequency and emits a ``-``
      ancestral state when no residue clears it, and whether that happens
      depends on the posterior shape, which differs between real and simulated
      data even when the gap pattern is copied exactly. The scorer then skips
      the position. There is nothing to compare these against, so they are not
      part of the test set.
    * **observed-only keys are missing null draws.** A few are tolerable and are
      carried as NaN (the calibration's exceedance rule is NaN-safe). Many
      indicate the clade structure is not being held fixed, which would
      invalidate the matched null, so this raises above ``max_missing_frac``.

    Clade sizes are required to agree exactly on the shared keys.
    """
    null = pd.read_csv(null_csv)
    obs = pd.read_csv(observed_csv)

    null_keys = set(map(tuple, null[KEY_COLUMNS].to_numpy()))
    obs_keys = set(map(tuple, obs[KEY_COLUMNS].to_numpy()))
    missing = obs_keys - null_keys
    extra = null_keys - obs_keys
    missing_frac = len(missing) / max(len(obs_keys), 1)

    print(
        f"[keys] observed={len(obs_keys):,} null={len(null_keys):,} "
        f"missing_from_null={len(missing):,} ({missing_frac:.4%}) "
        f"null_only_dropped={len(extra):,}"
    )
    if missing_frac > max_missing_frac:
        raise AssertionError(
            f"{len(missing):,} observed keys ({missing_frac:.2%}) have no null "
            f"draw, above the {max_missing_frac:.2%} tolerance; examples: "
            f"{sorted(missing)[:5]}"
        )

    merged = obs.merge(
        null, on=KEY_COLUMNS, suffixes=("_obs", "_null"), validate="one_to_one"
    )
    for col in ("clade_size_left", "clade_size_right"):
        a, b = merged[f"{col}_obs"], merged[f"{col}_null"]
        if not (a == b).all():
            n = int((a != b).sum())
            raise AssertionError(
                f"{col} differs on {n} rows between observed and null; the "
                "clade structure is not being held fixed"
            )
    print(f"[keys] clade sizes identical on all {len(merged):,} shared tests")
    return null


def write_npz(null: pd.DataFrame, observed_csv: Path, out_npz: Path) -> None:
    """Persist null scores in the observed table's row order, as float32.

    Row order is taken from the observed table so every replicate shares one
    canonical ordering and the calibration can stack them without re-keying.
    """
    obs = pd.read_csv(observed_csv, usecols=KEY_COLUMNS)
    aligned = obs.merge(null, on=KEY_COLUMNS, how="left", validate="one_to_one")
    payload = {c: aligned[c].to_numpy(dtype=np.float32) for c in SCORE_COLUMNS}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sim-alignment", type=Path, required=True,
                   help="simulated leaf alignment for this replicate")
    p.add_argument("--reconciled-tree", type=Path, required=True,
                   help="topology passed to IQ-TREE -te; must match the real run")
    p.add_argument("--asr-tree", type=Path, required=True,
                   help="rooted/mapped ASR tree used by the scorer")
    p.add_argument("--observed-scores", type=Path, required=True,
                   help="observed score CSV defining the canonical key set and row order")
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--model", default="LG+G",
                   help="substitution model for ASR; must match the real run")
    p.add_argument("--min-clade", type=int, default=5)
    p.add_argument("--node-naming", choices=["legacy", "strict"], default="strict",
                   help="'strict' is the default here: under 'legacy', fabricated "
                        "node names collide and the null/observed key correspondence "
                        "the matched null depends on is not well defined")
    p.add_argument("--threads", default="2")
    p.add_argument("--iqtree", default="iqtree2")
    p.add_argument("--workdir", type=Path, default=None,
                   help="scratch directory for IQ-TREE output (default: alongside --out-npz)")
    p.add_argument("--max-missing-frac", type=float, default=0.01,
                   help="fail if more than this fraction of observed tests have "
                        "no null draw in this replicate (default 0.01)")
    p.add_argument("--keep-intermediates", action="store_true",
                   help="do not delete the .state file and IQ-TREE scratch output")
    args = p.parse_args()

    work = args.workdir or (args.out_npz.parent / f"{args.out_npz.stem}_work")
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "asr"

    t0 = time.time()
    state = run_asr(
        alignment=args.sim_alignment,
        reconciled_tree=args.reconciled_tree,
        model=args.model,
        prefix=prefix,
        threads=args.threads,
        iqtree_binary=args.iqtree,
    )
    t_asr = time.time() - t0
    print(f"[asr] {t_asr:.1f}s -> {state} ({state.stat().st_size / 1e6:.0f} MB)")

    scored = work / "null_scores.csv"
    t0 = time.time()
    score_replicate(
        state=state,
        alerax_tree=args.reconciled_tree,
        asr_tree=args.asr_tree,
        alignment=args.sim_alignment,
        output_csv=scored,
        min_clade=args.min_clade,
        node_naming=args.node_naming,
    )
    t_score = time.time() - t0
    print(f"[score] {t_score:.1f}s -> {scored}")

    null = check_keys(scored, args.observed_scores, args.max_missing_frac)

    write_npz(null, args.observed_scores, args.out_npz)
    print(f"[write] {args.out_npz} ({args.out_npz.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_intermediates:
        shutil.rmtree(work, ignore_errors=True)
        print(f"[clean] removed {work}")


if __name__ == "__main__":
    main()
