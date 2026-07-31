#!/usr/bin/env python3
"""
verify_null_simulation.py

Verification gate for AliSim-simulated null alignments (see
simulate_null_alignments.py). Compares one simulated replicate against
the real alignment on explicit, reportable criteria and emits a
PASS/FAIL verdict per criterion plus a small summary figure.

Criteria:
  1. Sequence-name identity (same identifier set as the real alignment).
  2. Gap-mask identity, position by position.
  3. Per-column occupancy identity (follows from 2, checked independently
     since it is the quantity the pipeline filters on at >= occupancy
     threshold).
  4. Per-column composition preservation: Spearman correlation (real vs
     simulated) of per-column Shannon entropy and per-column frequency
     of the most common residue.
  5. Per-column substitution-rate correlation (optional, requires a
     precomputed per-site rate file from IQ-TREE's -wsr/--rate output;
     skipped if not supplied).

This script does not claim a simulation is valid -- it only reports the
measured criteria. Whether those measurements meet an acceptance bar
(e.g. Spearman >= 0.9) is a judgement made by whoever reads the report,
not asserted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
GAP_CHARS = set("-.")


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


def to_matrix(seqs: dict[str, str], order: list[str]) -> np.ndarray:
    ncol = len(next(iter(seqs.values())))
    mat = np.empty((len(order), ncol), dtype="<U1")
    for i, name in enumerate(order):
        mat[i, :] = list(seqs[name])
    return mat


def gap_mask(mat: np.ndarray) -> np.ndarray:
    return np.isin(mat, list(GAP_CHARS))


def column_occupancy(mask: np.ndarray) -> np.ndarray:
    # mask True = gap; occupancy = fraction non-gap
    return 1.0 - mask.mean(axis=0)


def shannon_entropy(col: np.ndarray, is_gap: np.ndarray) -> float:
    residues = col[~is_gap]
    if residues.size == 0:
        return 0.0
    _, counts = np.unique(residues, return_counts=True)
    freqs = counts / counts.sum()
    return float(-np.sum(freqs * np.log2(freqs)))


def top_residue_freq(col: np.ndarray, is_gap: np.ndarray) -> float:
    residues = col[~is_gap]
    if residues.size == 0:
        return 0.0
    _, counts = np.unique(residues, return_counts=True)
    return float(counts.max() / counts.sum())


def per_column_stats(mat: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ncol = mat.shape[1]
    entropy = np.empty(ncol)
    top_freq = np.empty(ncol)
    for j in range(ncol):
        entropy[j] = shannon_entropy(mat[:, j], mask[:, j])
        top_freq[j] = top_residue_freq(mat[:, j], mask[:, j])
    return entropy, top_freq


def read_site_rates(path: Path) -> np.ndarray:
    """Parse an IQ-TREE .rate file (columns: Site, Rate, ...)."""
    rates = []
    with open(path) as fh:
        header = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if header is None:
                header = parts
                continue
            rates.append(float(parts[1]))
    return np.array(rates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an AliSim-simulated null alignment against the real alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--real-alignment",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "IPR019888_trimmed.aln",
        help="Real (observed) alignment, FASTA.",
    )
    parser.add_argument(
        "--sim-alignment",
        type=Path,
        required=True,
        help="Simulated replicate alignment to verify, FASTA.",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.8,
        help="Occupancy threshold used by the pipeline (badasp_min_occupancy).",
    )
    parser.add_argument(
        "--real-site-rates",
        type=Path,
        default=None,
        help="Optional IQ-TREE .rate file (per-site rates) for the real "
             "alignment/tree, for criterion 5. Skipped if not given.",
    )
    parser.add_argument(
        "--sim-site-rates",
        type=Path,
        default=None,
        help="Optional matching per-site rate file for the simulated "
             "replicate, for criterion 5.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label for this verification run (default: sim alignment stem).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "badasp_scoring" / "null_calibration" / "verification",
        help="Directory to write the JSON report and summary PNG.",
    )
    args = parser.parse_args()

    label = args.label or args.sim_alignment.stem
    args.outdir.mkdir(parents=True, exist_ok=True)

    real_seqs = read_fasta(args.real_alignment)
    sim_seqs = read_fasta(args.sim_alignment)

    report: dict = {"label": label, "real_alignment": str(args.real_alignment),
                     "sim_alignment": str(args.sim_alignment)}

    # --- Criterion 1: sequence-name identity ---
    real_names = set(real_seqs.keys())
    sim_names = set(sim_seqs.keys())
    missing_in_sim = sorted(real_names - sim_names)
    extra_in_sim = sorted(sim_names - real_names)
    names_pass = (len(missing_in_sim) == 0) and (len(extra_in_sim) == 0)
    report["criterion_1_names"] = {
        "pass": names_pass,
        "n_real": len(real_names),
        "n_sim": len(sim_names),
        "n_missing_in_sim": len(missing_in_sim),
        "n_extra_in_sim": len(extra_in_sim),
        "missing_in_sim_sample": missing_in_sim[:10],
        "extra_in_sim_sample": extra_in_sim[:10],
    }
    print(f"[1] Sequence names: {'PASS' if names_pass else 'FAIL'} "
          f"(real={len(real_names)}, sim={len(sim_names)}, "
          f"missing={len(missing_in_sim)}, extra={len(extra_in_sim)})")

    if not names_pass:
        print("FATAL: name sets differ; cannot align matrices column-by-column. "
              "Reporting names-only result and stopping.")
        with open(args.outdir / f"{label}_verification.json", "w") as fh:
            json.dump(report, fh, indent=2)
        return

    order = sorted(real_names)
    real_mat = to_matrix(real_seqs, order)
    sim_mat = to_matrix(sim_seqs, order)

    if real_mat.shape != sim_mat.shape:
        report["criterion_2_gap_mask"] = {
            "pass": False,
            "error": f"shape mismatch real={real_mat.shape} sim={sim_mat.shape}",
        }
        print(f"[2] Gap mask: FAIL (shape mismatch real={real_mat.shape} sim={sim_mat.shape})")
        with open(args.outdir / f"{label}_verification.json", "w") as fh:
            json.dump(report, fh, indent=2)
        return

    real_mask = gap_mask(real_mat)
    sim_mask = gap_mask(sim_mat)

    # --- Criterion 2: gap-mask identity ---
    mismatch = real_mask != sim_mask
    n_mismatch = int(mismatch.sum())
    gap_pass = n_mismatch == 0
    mismatch_rows, mismatch_cols = np.where(mismatch)
    sample_locs = [
        {"seq": order[r], "col": int(c)}
        for r, c in list(zip(mismatch_rows, mismatch_cols))[:10]
    ]
    report["criterion_2_gap_mask"] = {
        "pass": gap_pass,
        "n_cells": int(mismatch.size),
        "n_mismatch": n_mismatch,
        "mismatch_fraction": n_mismatch / mismatch.size,
        "sample_mismatch_locations": sample_locs,
    }
    print(f"[2] Gap mask identical: {'PASS' if gap_pass else 'FAIL'} "
          f"(mismatches={n_mismatch}/{mismatch.size})")

    # --- Criterion 3: per-column occupancy identity ---
    real_occ = column_occupancy(real_mask)
    sim_occ = column_occupancy(sim_mask)
    occ_diff = np.abs(real_occ - sim_occ)
    occ_pass = bool(np.all(occ_diff < 1e-9))
    report["criterion_3_occupancy"] = {
        "pass": occ_pass,
        "max_abs_diff": float(occ_diff.max()),
        "n_columns_differing": int(np.sum(occ_diff >= 1e-9)),
        "n_columns_ge_threshold_real": int(np.sum(real_occ >= args.occupancy_threshold)),
        "n_columns_ge_threshold_sim": int(np.sum(sim_occ >= args.occupancy_threshold)),
    }
    print(f"[3] Per-column occupancy identical: {'PASS' if occ_pass else 'FAIL'} "
          f"(max abs diff={occ_diff.max():.2e}, "
          f">= {args.occupancy_threshold} occupancy: real={int(np.sum(real_occ >= args.occupancy_threshold))}, "
          f"sim={int(np.sum(sim_occ >= args.occupancy_threshold))})")

    # --- Criterion 4: per-column composition (entropy & top-residue freq) ---
    real_entropy, real_top = per_column_stats(real_mat, real_mask)
    sim_entropy, sim_top = per_column_stats(sim_mat, sim_mask)
    entropy_rho, entropy_p = spearmanr(real_entropy, sim_entropy)
    top_rho, top_p = spearmanr(real_top, sim_top)
    report["criterion_4_composition"] = {
        "entropy_spearman_rho": float(entropy_rho),
        "entropy_spearman_p": float(entropy_p),
        "top_residue_freq_spearman_rho": float(top_rho),
        "top_residue_freq_spearman_p": float(top_p),
        "n_columns": int(real_mat.shape[1]),
        "acceptance_threshold_note": "Plan's acceptance criterion for flavour B is "
                                      "Spearman >= 0.9. Flavour A (site-homogeneous "
                                      "negative control) is expected to score poorly here.",
    }
    print(f"[4] Composition Spearman rho: entropy={entropy_rho:.4f} (p={entropy_p:.2e}), "
          f"top-residue-freq={top_rho:.4f} (p={top_p:.2e})")

    # --- Criterion 5: per-column rate correlation (optional) ---
    if args.real_site_rates and args.sim_site_rates:
        real_rates = read_site_rates(args.real_site_rates)
        sim_rates = read_site_rates(args.sim_site_rates)
        if len(real_rates) == len(sim_rates) == real_mat.shape[1]:
            rate_rho, rate_p = spearmanr(real_rates, sim_rates)
            report["criterion_5_rate"] = {
                "spearman_rho": float(rate_rho),
                "spearman_p": float(rate_p),
                "status": "computed",
            }
            print(f"[5] Per-column rate Spearman rho: {rate_rho:.4f} (p={rate_p:.2e})")
        else:
            report["criterion_5_rate"] = {
                "status": "skipped",
                "reason": f"length mismatch real={len(real_rates)} sim={len(sim_rates)} ncol={real_mat.shape[1]}",
            }
            print("[5] Per-column rate: SKIPPED (length mismatch)")
    else:
        report["criterion_5_rate"] = {
            "status": "deferred",
            "reason": "no --real-site-rates/--sim-site-rates supplied",
        }
        print("[5] Per-column rate: DEFERRED (no rate files supplied)")

    # --- Write JSON + CSV + PNG ---
    json_path = args.outdir / f"{label}_verification.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {json_path}")

    csv_path = args.outdir / f"{label}_per_column_metrics.csv"
    with open(csv_path, "w") as fh:
        fh.write("column,real_occupancy,sim_occupancy,real_entropy,sim_entropy,real_top_freq,sim_top_freq\n")
        for j in range(real_mat.shape[1]):
            fh.write(f"{j},{real_occ[j]:.6f},{sim_occ[j]:.6f},"
                     f"{real_entropy[j]:.6f},{sim_entropy[j]:.6f},"
                     f"{real_top[j]:.6f},{sim_top[j]:.6f}\n")
    print(f"Wrote {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(real_occ, sim_occ, s=8, alpha=0.6)
    lims = [0, 1]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_xlabel("Real occupancy")
    axes[0].set_ylabel("Simulated occupancy")
    axes[0].set_title(f"Occupancy (max |diff|={occ_diff.max():.2e})")

    axes[1].scatter(real_entropy, sim_entropy, s=8, alpha=0.6)
    axes[1].set_xlabel("Real Shannon entropy")
    axes[1].set_ylabel("Simulated Shannon entropy")
    axes[1].set_title(f"Entropy (Spearman rho={entropy_rho:.3f})")

    axes[2].scatter(real_top, sim_top, s=8, alpha=0.6)
    axes[2].set_xlabel("Real top-residue frequency")
    axes[2].set_ylabel("Simulated top-residue frequency")
    axes[2].set_title(f"Top-residue freq (Spearman rho={top_rho:.3f})")

    fig.suptitle(f"Null simulation verification: {label}")
    fig.tight_layout()
    png_path = args.outdir / f"{label}_verification.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
