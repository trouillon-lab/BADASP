#!/usr/bin/env python3
"""
simulate_null_alignments.py

Wrap IQ-TREE's AliSim to generate replicate null alignments for BADASP
switch-detection threshold calibration.

Two null flavours are supported (selected via --flavour, or by overriding
--sim-model / --site-freq / --site-rate directly):

  A (negative control): a single, fully-specified site-homogeneous model
    (e.g. LG+G4{alpha}). The real alignment is only used to copy the gap
    pattern (-s), not to fit any parameters, so this is expected to be
    fast. Per-column composition is NOT expected to match the real
    alignment -- that mismatch is the point of this control.

  B (primary null): a site-heterogeneous mixture model (e.g. LG+C20+F+G)
    fitted on the real alignment, combined with --site-freq MEAN and
    --site-rate MEAN so each simulated column inherits the posterior-mean
    amino-acid profile and rate of the corresponding real column. Fitting
    the mixture model on the real alignment is the dominant cost.

Every path, model string, replicate count and seed is a CLI argument.
Defaults are sourced from config/snakemake.yaml's `null_calibration` and
`paths` blocks where a corresponding key exists -- nothing is hardcoded
in this script itself.

This script only constructs and runs the AliSim command; it does not
interpret or verify the output. Use verify_null_simulation.py for that.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "snakemake.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.iqtree_bin,
        "--alisim", str(args.out_prefix),
        "-t", str(args.sim_tree),
        "-m", args.sim_model,
        "-s", str(args.alignment),
        "--num-alignments", str(args.num_alignments),
        "-af", "fasta",
        "--seed", str(args.seed),
        "-T", str(args.threads),
        # Providing both -s and -t alongside --alisim otherwise makes
        # IQ-TREE additionally attempt a real ML tree search on top of
        # the simulation (and that search errors out on the ML tree's
        # unrooted trifurcating root). --tree-fix disables the search
        # so -s/-t are used only for gap-copying / model reference.
        "--tree-fix",
        # Without --prefix, IQ-TREE derives an output prefix from -s and
        # writes .iqtree/.log/.ckp.gz/.treefile side files next to the
        # real input alignment. Pin the prefix to our own output
        # location so the real data directory is never touched.
        "--prefix", str(args.out_prefix),
    ]
    if args.site_freq is not None:
        cmd += ["--site-freq", args.site_freq]
    if args.site_rate is not None:
        cmd += ["--site-rate", args.site_rate]
    if args.branch_scale is not None:
        cmd += ["--branch-scale", str(args.branch_scale)]
    if args.write_all:
        cmd.append("--write-all")
    if args.no_copy_gaps:
        cmd.append("--no-copy-gaps")
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IQ-TREE AliSim to generate null-model replicate alignments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config/snakemake.yaml (source of defaults below).",
    )
    # Parse just --config first so remaining defaults can be sourced from it.
    known_args, _ = parser.parse_known_args()
    cfg = load_config(known_args.config)
    nc = cfg.get("null_calibration", {})
    paths = cfg.get("paths", {})
    tools = cfg.get("tools", {})

    parser.add_argument(
        "--flavour",
        choices=["A", "B"],
        default=nc.get("flavour", "A"),
        help="Null flavour label (A = site-homogeneous negative control, "
             "B = site-heterogeneous mixture primary null). Only used for "
             "naming/logging -- the actual model comes from --sim-model.",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=REPO_ROOT / paths.get("trimmed_fasta", "data/interim/IPR019888_trimmed.aln"),
        help="Real alignment (-s): supplies the gap pattern (and, for "
             "flavour B, the data AliSim fits the mixture model on).",
    )
    parser.add_argument(
        "--sim-tree",
        type=Path,
        default=REPO_ROOT / nc.get("sim_tree", "data/interim/iqtree_asr/IPR019888.treefile"),
        help="Tree with branch lengths to simulate along (-t). Must carry "
             "real ML branch lengths, not a placeholder-length topology.",
    )
    parser.add_argument(
        "--sim-model",
        default=nc.get("sim_model", "LG+G4{1.7211}"),
        help="AliSim -m model string, e.g. 'LG+G4{1.7211}' (flavour A) or "
             "'LG+C20+F+G' (flavour B).",
    )
    parser.add_argument(
        "--num-alignments",
        type=int,
        default=nc.get("replicates", 2),
        help="Number of replicate alignments to simulate (--num-alignments).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=nc.get("seed", 1),
        help="AliSim random seed (--seed). AliSim defaults to the CPU "
             "clock if unset, which is not reproducible, so this is "
             "always passed explicitly.",
    )
    parser.add_argument(
        "--site-freq",
        choices=["MEAN", "SAMPLING", "MODEL"],
        default=None,
        help="--site-freq option for mixture models (flavour B). Leave "
             "unset for flavour A.",
    )
    parser.add_argument(
        "--site-rate",
        choices=["MEAN", "SAMPLING", "MODEL"],
        default=None,
        help="--site-rate option to mimic per-site rate heterogeneity "
             "from the input alignment (flavour B). Leave unset for "
             "flavour A.",
    )
    parser.add_argument(
        "--branch-scale",
        type=float,
        default=None,
        help="Optional --branch-scale factor applied to all branch lengths.",
    )
    parser.add_argument(
        "--write-all",
        action="store_true",
        help="Pass --write-all to also output simulated internal (ancestral) sequences.",
    )
    parser.add_argument(
        "--no-copy-gaps",
        action="store_true",
        help="Pass --no-copy-gaps to disable copying the real alignment's gap pattern.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="-T thread count for IQ-TREE/AliSim.",
    )
    parser.add_argument(
        "--iqtree-bin",
        default=tools.get("iqtree", "iqtree2"),
        help="IQ-TREE executable.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "badasp_scoring" / "null_calibration",
        help="Directory to write simulated alignments and logs under "
             "(a flavour-specific subdirectory is created).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Prefix for output files within the flavour subdirectory "
             "(default: 'flavour_<A|B>_seed<seed>').",
    )
    parser.add_argument(
        "--extra-args",
        default=None,
        help="Extra raw arguments appended verbatim to the AliSim command "
             "(shell-split), for options not exposed above.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds for the AliSim subprocess. "
             "If exceeded, the process is killed and a TimeoutExpired is raised.",
    )
    args = parser.parse_args()

    flavour_dir = args.outdir / f"flavour_{args.flavour}"
    flavour_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"flavour_{args.flavour}_seed{args.seed}"
    args.out_prefix = flavour_dir / run_name
    log_path = flavour_dir / f"{run_name}.alisim.log"

    cmd = build_command(args)
    print("Running AliSim command:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"Logging stdout/stderr to: {log_path}")

    start = time.monotonic()
    with open(log_path, "w") as log_fh:
        log_fh.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        log_fh.flush()
        try:
            result = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            print(f"AliSim TIMED OUT after {elapsed:.1f}s (timeout={args.timeout}s). "
                  f"See partial log at {log_path}")
            raise
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print(f"AliSim FAILED (exit {result.returncode}) after {elapsed:.1f}s. "
              f"See log: {log_path}", file=sys.stderr)
        sys.exit(result.returncode)

    out_files = sorted(flavour_dir.glob(f"{run_name}*.fa")) + \
        sorted(flavour_dir.glob(f"{run_name}*.fasta"))
    total_bytes = sum(f.stat().st_size for f in out_files)
    print(f"AliSim completed in {elapsed:.2f}s ({elapsed / 60:.2f} min).")
    print(f"Output files ({len(out_files)}):")
    for f in out_files:
        print(f"  {f}  ({f.stat().st_size / 1e6:.2f} MB)")
    print(f"Total output size: {total_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
