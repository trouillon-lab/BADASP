#!/usr/bin/env python3
"""
simulate_null_persite.py

Simulate a null-model amino-acid alignment with IQ-TREE's AliSim in which
EVERY alignment column gets its own amino-acid equilibrium-frequency
vector, at non-mixture (cheap) cost -- i.e. without fitting a C20-style
frequency mixture model.

Why this exists
----------------
The natural way to hand AliSim per-site frequencies would be a RAxML-style
partition file with 169 single-site partitions, each `LG+F{f1,...,f20}`.
That fails: the commas inside `+F{...}` collide with the partition file's
own comma delimiter ("ERROR: Close bracket not found in +F{0.078333").

The fix used here is a NEXUS model-definition file (`--mdef FILE`, "Name
of a NEXUS model file to define new models"). It lets you declare named
frequency vectors and named models *outside* the partition file, so the
partition file only ever contains bare model names with no commas:

    #nexus
    begin models;
      frequency sf1 = 0.0700 0.0500 ... ;   [ 20 numbers, order see AA_ORDER ]
      model m1 = LG+FMIX{sf1};
      ...
    end;

and a partition file of lines `m1, p1 = 1-1`.

This exact form was verified empirically against IQ-TREE 2.3.6
(iqtree2 --version): `LG+F<name>` (bare suffix, no braces) is rejected
at runtime ("Unknown state frequency type +F<name>") even though IQ-TREE's
own model report sometimes *displays* fitted per-class models that way --
that display form is not valid re-parseable input syntax. The form that
does work is a single-component frequency mixture, `LG+FMIX{name}`
(no weight needed for one component). A sanity check (a single frequency
vector concentrated at ~95% on one residue, attached to a 200-taxon real
tree) recovered ~95.5% of that residue at the tip, confirming the named
vector is used as the process's stationary distribution and not silently
ignored or reinterpreted.

`-q` (edge-linked equal partition model) is used so every partition
shares the same branch lengths -- we do not want per-partition branch
length rescaling (`-Q`/`-p`) for a null model whose entire premise is
"the tree and its branch lengths are held fixed."

Gaps are NOT copied by passing -s to AliSim (see build_alisim_command's
docstring for why: with this partitioned model, -s silently triggers a
real per-partition ML rate fit against the real data, even under
--tree-fix, which is both slow and not what "simulate along a fixed
tree" is supposed to mean). Instead, the real alignment's gap pattern is
applied afterwards in Python (apply_gap_mask()), which is behaviourally
equivalent to AliSim's own gap-copying without the fitting side effect.

No hardcoded frequencies
-------------------------
Per-column target frequencies are computed from an input alignment
(--composition-alignment), then shrunk toward a reference model's
equilibrium frequencies (--reference-model, default LG, matching the
substitution matrix already fitted to this data -- see
config/snakemake.yaml's null_calibration.sim_model) by a documented,
tunable --shrinkage weight (default 0.15). The reference equilibrium
vector itself is never pasted into this file as a numeric constant --
it is estimated at runtime by asking IQ-TREE to simulate a very long
single-branch (near-zero length) alignment under --reference-model and
reading back the realized composition (see estimate_reference_freqs()).
That is the tool's own model, read back from the tool, not a value we
looked up and transcribed.

Caveat (documented, not resolved): per-column empirical composition
includes whatever real evolutionary signal is in the alignment,
including any genuine switch-like shift between clades. Shrinking
toward the reference model's global equilibrium bounds how far a
single site's profile can be pulled by that signal, but does not
remove it -- so this null is somewhat anti-conservative at
signal-bearing sites. Larger --shrinkage trades fidelity to the real
per-site profile for more conservatism.

Per-site rates are secondary in this script: if --gamma-alpha is given,
every per-site model gets `+G4{alpha}` appended (a fixed, not estimated,
shape parameter), which gives each site its own drawn discrete-gamma rate
category -- a cheap per-site rate attachment, not a fit to real per-site
rates. If omitted, no rate heterogeneity is added at all.

This script only builds the mdef/partition files and invokes AliSim.
Use verify_null_simulation.py to check the output's composition fidelity
against a real alignment (Spearman of per-column Shannon entropy and
top-residue frequency); this script does not claim any fidelity target
has been met.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "snakemake.yaml"

# Canonical amino-acid ordering used throughout. Confirmed empirically to
# match IQ-TREE's own internal ordering: estimate_reference_freqs() run
# under -m LG reproduces the well-known Le & Gascuel (2008) LG frequencies
# in exactly this order (A ~0.079, R ~0.056, ..., V ~0.069).
AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
GAP_CHARS = set("-.")


def load_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            seqs[name] = "".join(chunks)
    return seqs


def compute_column_counts(seqs: dict[str, str]) -> list[Counter]:
    """Per-column counts over AA_ORDER only; gaps and any other character
    (X, B, Z, U, O, ...) are excluded from the denominator entirely."""
    ncol = len(next(iter(seqs.values())))
    counts = [Counter() for _ in range(ncol)]
    for seq in seqs.values():
        for j, ch in enumerate(seq):
            if ch in AA_ORDER:
                counts[j][ch] += 1
    return counts


def estimate_reference_freqs(
    iqtree_bin: str,
    model: str,
    length: int,
    seed: int,
    workdir: Path,
    redo: bool,
) -> list[float]:
    """Estimate a substitution model's equilibrium amino-acid frequencies
    by asking IQ-TREE's AliSim to simulate a very long alignment along a
    near-zero-length two-taxon tree under `model` and reading back the
    realized composition. With branch length ~0, the two tips are
    (numerically) draws from the model's stationary distribution itself,
    so this recovers the model's built-in frequency vector to floating
    precision without ever hardcoding it in this script.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    tree_path = workdir / f"refmodel_{model.replace('+', '_')}.tree"
    tree_path.write_text("(A:0.0000001,B:0.0000001);\n")
    out_prefix = workdir / f"refmodel_{model.replace('+', '_')}_len{length}_seed{seed}"

    cmd = [
        iqtree_bin,
        "--alisim", str(out_prefix),
        "-t", str(tree_path),
        "-m", model,
        "--length", str(length),
        "--seed", str(seed),
        "-af", "fasta",
    ]
    if redo:
        cmd.append("-redo")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Reference-frequency estimation failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )

    fa_path = out_prefix.with_suffix(".fa")
    if not fa_path.exists():
        candidates = sorted(workdir.glob(out_prefix.name + "*.fa"))
        if not candidates:
            raise RuntimeError(f"No AliSim output found for reference estimate at {out_prefix}*")
        fa_path = candidates[0]

    seqs = read_fasta(fa_path)
    counts = Counter()
    for seq in seqs.values():
        counts.update(ch for ch in seq if ch in AA_ORDER)
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError(f"Reference-frequency estimate produced no residues from {fa_path}")
    return [counts.get(aa, 0) / total for aa in AA_ORDER]


def shrink_column_freqs(
    counts: Counter,
    reference: list[float],
    shrinkage: float,
    min_nongap: int,
) -> list[float]:
    total = sum(counts.values())
    if total < min_nongap:
        # Too few (or zero) real residues in this column to estimate a
        # per-column composition at all -- fall back entirely to the
        # reference model's equilibrium frequencies.
        raw = list(reference)
    else:
        raw = [counts.get(aa, 0) / total for aa in AA_ORDER]
    shrunk = [(1.0 - shrinkage) * r + shrinkage * f for r, f in zip(raw, reference)]
    s = sum(shrunk)
    shrunk = [v / s for v in shrunk]
    if min(shrunk) <= 0.0:
        raise RuntimeError(
            "Shrunk frequency vector has a non-positive entry; this should "
            "not happen for shrinkage > 0 with a reference model that has "
            "no zero frequencies. Got: " + repr(shrunk)
        )
    return shrunk


def write_mdef(
    freq_vectors: list[list[float]],
    base_model: str,
    gamma_alpha: float | None,
    path: Path,
) -> list[str]:
    model_names = [f"m{j + 1}" for j in range(len(freq_vectors))]
    gamma_suffix = f"+G4{{{gamma_alpha}}}" if gamma_alpha is not None else ""
    with open(path, "w") as fh:
        fh.write("#nexus\n\nbegin models;\n")
        for j, freqs in enumerate(freq_vectors):
            values = " ".join(f"{v:.8f}" for v in freqs)
            fh.write(f"  frequency sf{j + 1} = {values} ;\n")
        for j, name in enumerate(model_names):
            fh.write(f"  model {name} = {base_model}+FMIX{{sf{j + 1}}}{gamma_suffix};\n")
        fh.write("end;\n")
    return model_names


def write_partition(model_names: list[str], path: Path) -> None:
    with open(path, "w") as fh:
        for j, name in enumerate(model_names):
            site = j + 1
            fh.write(f"{name}, p{site} = {site}-{site}\n")


def build_alisim_command(args: argparse.Namespace, mdef_path: Path, part_path: Path) -> list[str]:
    # Deliberately never pass -s (the real alignment) to AliSim here. With a
    # *partitioned* (--mdef/-q) model, doing so does not just copy the gap
    # pattern: IQ-TREE additionally estimates each partition's free relative-
    # rate ("Speed") parameter by maximum likelihood against the real data in
    # -s, even with --tree-fix set (--tree-fix only suppresses topology/
    # branch-length search, not this per-partition rate fit). Confirmed
    # empirically -- with -s present the log shows "Estimate model
    # parameters" / "Current log-likelihood at step N" convergence output
    # even though every per-site frequency vector is already fully fixed,
    # and wall-clock exploded with taxon count (200 taxa: ~10s; 2000 taxa:
    # still running and climbing past 4GB RAM after 7 minutes, killed). It
    # also reproducibly triggers a segfault inside IQ-TREE's own partitioned-
    # tree-linking code (PhyloSuperTree::linkTree/mapTrees) on the full
    # 21,218-taxon tree, for both -q and -Q -- i.e. -s here routes through
    # IQ-TREE's real ML-analysis code path, not the lightweight alisim-only
    # path. Dropping -s avoids all of this; the gap mask is instead applied
    # ourselves afterwards in Python (see apply_gap_mask()), which is exactly
    # equivalent to "copy the gap pattern" without the fitting side effect.
    cmd = [
        args.iqtree_bin,
        "--alisim", str(args.out_prefix),
        "-t", str(args.sim_tree),
        "--mdef", str(mdef_path),
        "-q", str(part_path),
        "--seed", str(args.seed),
        "-af", "fasta",
        "-n", "0",  # required: the tree is multifurcating; IQ-TREE otherwise
                    # errors with "Tree search does not work with initial
                    # multifurcating tree."
        "--num-alignments", str(args.num_alignments),
        "-T", str(args.threads),
    ]
    if args.redo:
        cmd.append("-redo")
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def apply_gap_mask(sim_fasta_path: Path, mask_seqs: dict[str, str]) -> tuple[int, int, int]:
    """Overwrite sim_fasta_path in place, forcing '-' at every position that
    is a gap in mask_seqs for the matching sequence name. Equivalent to
    AliSim's own --no-copy-gaps=false behaviour, applied in Python so the
    real alignment never has to be passed to AliSim as -s (see
    build_alisim_command). Returns (n_sequences_masked, n_sequences_skipped,
    n_gap_cells_written).
    """
    sim_seqs = read_fasta(sim_fasta_path)
    order = list(sim_seqs.keys())
    n_masked = 0
    n_skipped = 0
    n_gap_cells = 0
    out_seqs: dict[str, str] = {}
    for name in order:
        sseq = sim_seqs[name]
        rseq = mask_seqs.get(name)
        if rseq is None or len(rseq) != len(sseq):
            out_seqs[name] = sseq
            n_skipped += 1
            continue
        chars = list(sseq)
        for j, ch in enumerate(rseq):
            if ch in GAP_CHARS:
                chars[j] = "-"
                n_gap_cells += 1
        out_seqs[name] = "".join(chars)
        n_masked += 1
    with open(sim_fasta_path, "w") as fh:
        for name in order:
            fh.write(f">{name}\n{out_seqs[name]}\n")
    return n_masked, n_skipped, n_gap_cells


def read_newick_leaf_names(path: Path) -> list[str]:
    """Return the leaf labels of a Newick tree, without a tree library.

    A leaf label is a token appearing immediately after '(' or ',';
    internal labels follow ')' and are therefore not matched. Square-bracket
    comments are stripped first. Checked against ete3's own parser on
    data/interim/iqtree_asr/IPR019888.treefile: both yield the same set of
    21,218 names.
    """
    text = re.sub(r"\[[^\]]*\]", "", path.read_text())
    return [m.group(1) for m in re.finditer(r"[(,]\s*([^(),:;\s]+)", text)]


def check_taxon_sets(
    composition_alignment: Path,
    aln_names: set[str],
    sim_tree: Path,
    allow_mismatch: bool,
) -> None:
    """Fail loudly when the composition alignment and the simulation tree
    describe different taxon sets.

    This exists because the failure it catches is silent: a stale composition
    alignment still produces per-column frequency vectors, AliSim still exits
    0, and the SLURM job still reports COMPLETED -- but the frequencies were
    computed over the wrong set of sequences and the gap mask cannot be
    applied to the tips it failed to name-match. Euler job 10856979 ran to
    completion exactly this way (21,641-sequence alignment against a
    21,218-leaf tree) and its output had to be quarantined. An existence
    check on the two paths does not catch it.
    """
    leaf_names = read_newick_leaf_names(sim_tree)
    leaf_set = set(leaf_names)
    if len(leaf_set) != len(leaf_names):
        raise SystemExit(
            f"Simulation tree {sim_tree} has {len(leaf_names)} leaf labels but only "
            f"{len(leaf_set)} distinct ones; duplicate tip names make the gap mask "
            f"ambiguous."
        )

    missing = leaf_set - aln_names          # tip that cannot be gap-masked at all
    extra = aln_names - leaf_set            # sequence that skews the column frequencies

    print(f"Taxon-set check: {len(leaf_set)} tree leaves vs {len(aln_names)} "
          f"alignment sequences ({len(missing)} tree-only, {len(extra)} alignment-only)")
    if not missing and not extra:
        return

    detail = (
        f"Taxon-set mismatch between --composition-alignment {composition_alignment} "
        f"({len(aln_names)} sequences) and --sim-tree {sim_tree} "
        f"({len(leaf_set)} leaves): {len(missing)} tree leaves absent from the "
        f"alignment, {len(extra)} alignment sequences absent from the tree.\n"
        f"  example tree-only leaves: {sorted(missing)[:5]}\n"
        f"  example alignment-only sequences: {sorted(extra)[:5]}"
    )
    if missing or not allow_mismatch:
        # Tree-only leaves are always fatal: those tips get no gap mask.
        # Alignment-only sequences are fatal by default because they bias
        # every column's frequency vector, but --allow-taxon-mismatch permits
        # them for deliberate pruned-subtree test runs.
        raise SystemExit(
            detail + "\nRefusing to simulate. Pass --allow-taxon-mismatch only for a "
            "deliberate pruned-subset run in which the alignment is a superset of "
            "the tree's tips."
        )
    print("WARNING: proceeding under --allow-taxon-mismatch.\n" + detail, file=sys.stderr)


def main() -> None:
    # Pre-parse only --config, on a separate add_help=False parser, so that
    # --help exits early here (showing just --config) instead of on the real
    # parser below, which has every other option. The real parser (with
    # add_help defaulted to True) is what --help/-h ends up hitting.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    known_args, _ = pre_parser.parse_known_args()
    cfg = load_config(known_args.config)
    nc = cfg.get("null_calibration", {})
    paths = cfg.get("paths", {})
    tools = cfg.get("tools", {})

    parser = argparse.ArgumentParser(
        description="Simulate a per-site (per-column) frequency null alignment "
                    "with IQ-TREE AliSim via a NEXUS --mdef model file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config/snakemake.yaml (source of a few defaults below).",
    )
    parser.add_argument(
        "--composition-alignment",
        type=Path,
        default=REPO_ROOT / paths.get("trimmed_fasta", "data/interim/IPR019888_trimmed.aln"),
        help="Real alignment used to compute each column's target amino-acid "
             "frequency vector. Independent of --sim-tree's taxon set: this "
             "is a per-column property of the family, not of any one subset "
             "of taxa.",
    )
    parser.add_argument(
        "--sim-tree",
        type=Path,
        default=REPO_ROOT / nc.get("sim_tree", "data/interim/iqtree_asr/IPR019888.treefile"),
        help="Tree (with branch lengths) to simulate along (-t). May be a "
             "pruned subset tree for a smaller test run.",
    )
    parser.add_argument(
        "--gap-mask-source",
        type=Path,
        default=None,
        help="Alignment supplying the gap pattern to overlay onto the "
             "simulated output, matched by sequence name (default: same as "
             "--composition-alignment, which contains every taxon's real gap "
             "pattern). Applied in Python after simulation -- NOT passed to "
             "AliSim as -s, since that would also trigger a real per-"
             "partition ML rate fit against the real data (see "
             "build_alisim_command). Mutually exclusive with --no-gap-mask.",
    )
    parser.add_argument(
        "--no-gap-mask",
        action="store_true",
        help="Skip gap masking; leave the simulated alignment fully ungapped "
             "(mutually exclusive with --gap-mask-source).",
    )
    parser.add_argument(
        "--allow-taxon-mismatch",
        action="store_true",
        help="Permit --composition-alignment to contain sequences absent from "
             "--sim-tree (intended only for deliberate pruned-subtree runs). "
             "Tree leaves absent from the alignment remain fatal regardless.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="AliSim output prefix (also used as the base name for the "
             "generated --mdef/-q files, written alongside it with a "
             "'.mdef.nex' / '.parts' suffix).",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.15,
        help="Weight (0-1) given to --reference-model's equilibrium "
             "frequencies when building each column's target vector: "
             "target = (1 - shrinkage) * column_empirical + shrinkage * "
             "reference_equilibrium. Guards against zero frequencies from "
             "small per-column counts. NOTE: column composition includes any "
             "real switch signal at that site; shrinkage bounds but does not "
             "remove that anti-conservative effect.",
    )
    parser.add_argument(
        "--min-nongap-count",
        type=int,
        default=1,
        help="Minimum number of non-gap residues required in a composition-"
             "alignment column before using its empirical frequencies at all; "
             "columns with fewer fall back entirely to --reference-model's "
             "equilibrium frequencies.",
    )
    parser.add_argument(
        "--reference-model",
        default=nc.get("sim_model", "LG+G4{1.7211}").split("+")[0],
        help="Base substitution model: (a) supplies exchangeabilities for "
             "every per-site LG+FMIX{...} model, and (b) is the shrinkage "
             "target equilibrium distribution (estimated at runtime, see "
             "--reference-mc-length; never hardcoded in this script).",
    )
    parser.add_argument(
        "--reference-mc-length",
        type=int,
        default=2_000_000,
        help="Number of sites simulated under --reference-model along a "
             "near-zero-branch two-taxon tree to Monte-Carlo estimate its "
             "equilibrium frequency vector.",
    )
    parser.add_argument(
        "--gamma-alpha",
        type=float,
        default=None,
        help="If given, append +G4{alpha} to every per-site model, giving "
             "each site its own drawn discrete-gamma rate class (a fixed, "
             "not estimated, shape parameter -- cheap but not a fit to real "
             "per-site rates). Omit for no rate heterogeneity.",
    )
    parser.add_argument(
        "--num-alignments",
        type=int,
        default=nc.get("replicates", 1),
        help="Number of replicate alignments to simulate (--num-alignments).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="AliSim random seed (--seed), also used to seed the reference-"
             "frequency Monte Carlo estimate unless --reference-mc-seed is given.",
    )
    parser.add_argument(
        "--reference-mc-seed",
        type=int,
        default=None,
        help="Seed for the reference-frequency Monte Carlo estimate "
             "(default: same as --seed).",
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
        "--workdir",
        type=Path,
        default=None,
        help="Directory to write the generated --mdef/-q files and the "
             "reference-frequency probe run into (default: --out-prefix's "
             "parent directory).",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Pass -redo to AliSim (and to the reference-frequency probe run).",
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
        help="Optional wall-clock timeout in seconds for the AliSim subprocess.",
    )
    args = parser.parse_args()

    if args.no_gap_mask and args.gap_mask_source is not None:
        parser.error("--no-gap-mask and --gap-mask-source are mutually exclusive.")
    if not (0.0 < args.shrinkage <= 1.0):
        parser.error("--shrinkage must be in (0, 1] (0 disables the zero-frequency guard).")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.workdir or args.out_prefix.parent
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading composition alignment: {args.composition_alignment}")
    seqs = read_fasta(args.composition_alignment)
    ncol = len(next(iter(seqs.values())))
    print(f"  {len(seqs)} sequences x {ncol} columns")

    check_taxon_sets(
        args.composition_alignment, set(seqs), args.sim_tree, args.allow_taxon_mismatch,
    )

    counts = compute_column_counts(seqs)

    mc_seed = args.reference_mc_seed if args.reference_mc_seed is not None else args.seed
    print(f"Estimating {args.reference_model} equilibrium frequencies via "
          f"{args.reference_mc_length}-site AliSim Monte Carlo (seed={mc_seed})...")
    reference_freqs = estimate_reference_freqs(
        args.iqtree_bin, args.reference_model, args.reference_mc_length,
        mc_seed, workdir / "reference_freq_probe", args.redo,
    )
    print("  reference frequencies (" + ",".join(AA_ORDER) + "):")
    print("  " + " ".join(f"{v:.4f}" for v in reference_freqs))

    n_fallback = sum(1 for c in counts if sum(c.values()) < args.min_nongap_count)
    if n_fallback:
        print(f"  {n_fallback}/{ncol} columns fall back fully to the reference "
              f"frequencies (fewer than {args.min_nongap_count} non-gap residues).")

    freq_vectors = [
        shrink_column_freqs(c, reference_freqs, args.shrinkage, args.min_nongap_count)
        for c in counts
    ]

    mdef_path = args.out_prefix.with_suffix(".mdef.nex")
    part_path = args.out_prefix.with_suffix(".parts")
    model_names = write_mdef(freq_vectors, args.reference_model, args.gamma_alpha, mdef_path)
    write_partition(model_names, part_path)
    print(f"Wrote {mdef_path} ({len(freq_vectors)} named frequency vectors + models)")
    print(f"Wrote {part_path} ({len(model_names)} single-site partitions)")

    cmd = build_alisim_command(args, mdef_path, part_path)
    log_path = args.out_prefix.with_suffix(".alisim.log")
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

    out_files = sorted(args.out_prefix.parent.glob(args.out_prefix.name + "*.fa")) + \
        sorted(args.out_prefix.parent.glob(args.out_prefix.name + "*.fasta"))
    print(f"AliSim completed in {elapsed:.2f}s ({elapsed / 60:.2f} min).")

    if not args.no_gap_mask:
        mask_source_path = args.gap_mask_source or args.composition_alignment
        mask_seqs = seqs if mask_source_path == args.composition_alignment else read_fasta(mask_source_path)
        print(f"Applying gap mask from {mask_source_path} to {len(out_files)} output file(s)...")
        for f in out_files:
            n_masked, n_skipped, n_gap_cells = apply_gap_mask(f, mask_seqs)
            print(f"  {f.name}: masked {n_masked} sequences ({n_gap_cells} gap cells written), "
                  f"{n_skipped} sequences had no name/length match and were left ungapped")
            if n_skipped:
                # Backstop for check_taxon_sets(): that compares names, this also
                # catches a length mismatch (right taxa, wrong number of columns).
                # An ungapped tip is not a valid null draw -- it changes clade
                # occupancy and therefore which nodes qualify for scoring.
                raise SystemExit(
                    f"{f.name}: {n_skipped} simulated sequences had no name/length "
                    f"match in the gap-mask source {mask_source_path} and were left "
                    f"ungapped. Refusing to emit an alignment whose gap pattern does "
                    f"not match the real one."
                )

    total_bytes = sum(f.stat().st_size for f in out_files)
    print(f"Output files ({len(out_files)}):")
    for f in out_files:
        print(f"  {f}  ({f.stat().st_size / 1e6:.2f} MB)")
    print(f"Total output size: {total_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
