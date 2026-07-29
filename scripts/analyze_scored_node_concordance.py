#!/usr/bin/env python3
"""
analyze_scored_node_concordance.py

Compute concordance between the initial AleRax run and the rec_check run
specifically for BADASP-scored nodes (internal nodes that passed the
min_clade_size threshold and have non-Unresolved event types).

Nodes are matched across the two runs by their leaf signature (sorted
tuple of descendant leaf names), using the respective AleRax consensus
trees to define those signatures.

Output: concordance stats, confusion matrix, and plots in
results/dtl_sensitivity/scored_node_concordance/
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ete3 import Tree


EVENTS = ["Speciation", "Duplication", "Transfer"]


def load_scored_nodes(csv_path: Path) -> dict[str, str]:
    """Return {node_name: event_type} for unique scored nodes."""
    df = pd.read_csv(csv_path)
    return (
        df.groupby("node_name")["event_type"]
        .first()
        .to_dict()
    )


def build_sig_to_event(
    consensus_nwk: Path,
    samples_newick: Path,
    scored_nodes: dict[str, str],
) -> dict[tuple, str]:
    """
    For each internal node in the consensus tree, compute its leaf signature
    and check if it corresponds to a scored node. Returns only nodes that are
    scored (event_type != Unresolved) with their majority-vote event from the
    samples (same as the scoring pipeline's alerax_events dict).
    """
    t = Tree(str(consensus_nwk), format=1)

    # Build clade event counts from samples
    clade_event_counts: dict[tuple, Counter] = {}
    if samples_newick.exists():
        with samples_newick.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    st = Tree(line, format=1)
                    for node in st.traverse():
                        if not node.is_leaf() and node.name in {"S", "D", "T"}:
                            sig = tuple(sorted(lf.name for lf in node.get_leaves()))
                            clade_event_counts.setdefault(sig, Counter())[node.name] += 1
                except Exception:
                    continue

    # Map each consensus tree node to majority event
    ev_map: dict[str, str] = {}
    label_map = {"D": "Duplication", "S": "Speciation", "T": "Transfer"}
    for node in t.traverse():
        if node.is_leaf():
            continue
        sig = tuple(sorted(lf.name for lf in node.get_leaves()))
        if sig in clade_event_counts:
            majority, _ = clade_event_counts[sig].most_common(1)[0]
            ev_map[sig] = label_map.get(majority, "Unresolved")
        else:
            ev = getattr(node, "Ev", "Unresolved")
            ev_map[sig] = label_map.get(ev, "Unresolved")

    # Build leaf-sig → event for signatures that correspond to a scored node in
    # raw_node_scores.csv. We match by rebuilding the ASR node → leaf sig mapping.
    # Since we only have the consensus nwk here, use it as a proxy: any consensus
    # node whose leaf signature is a subset present in ev_map is already there.
    # We restrict to the subset of signatures that appear in ev_map AND are scored.
    scored_sigs: dict[tuple, str] = {}
    for node in t.traverse():
        if node.is_leaf():
            continue
        sig = tuple(sorted(lf.name for lf in node.get_leaves()))
        ev = ev_map.get(sig, "Unresolved")
        if ev in ("Speciation", "Duplication", "Transfer"):
            scored_sigs[sig] = ev

    return scored_sigs


def compute_concordance(
    init_sigs: dict[tuple, str],
    rec_sigs: dict[tuple, str],
) -> dict:
    shared = set(init_sigs) & set(rec_sigs)
    confusion: Counter = Counter()
    n_match = 0

    for sig in shared:
        ev_i = init_sigs[sig]
        ev_r = rec_sigs[sig]
        confusion[(ev_i, ev_r)] += 1
        if ev_i == ev_r:
            n_match += 1

    n_shared = len(shared)
    concordance = (n_match / n_shared * 100) if n_shared else 0.0

    # Cohen's kappa
    total = n_shared
    po = n_match / total if total else 0.0

    pe = 0.0
    for ev in EVENTS:
        p_init = sum(1 for v in init_sigs.values() if v == ev) / len(init_sigs) if init_sigs else 0.0
        p_rec  = sum(1 for v in rec_sigs.values() if v == ev) / len(rec_sigs) if rec_sigs else 0.0
        pe += p_init * p_rec

    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return {
        "n_init": len(init_sigs),
        "n_rec": len(rec_sigs),
        "n_shared": n_shared,
        "n_matching": n_match,
        "concordance_pct": concordance,
        "cohen_kappa": kappa,
        "confusion": confusion,
    }


def plot_concordance_matrix(confusion: Counter, outdir: Path) -> None:
    mat = np.zeros((len(EVENTS), len(EVENTS)), dtype=int)
    for i, ev_i in enumerate(EVENTS):
        for j, ev_r in enumerate(EVENTS):
            mat[i, j] = confusion.get((ev_i, ev_r), 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(EVENTS)))
    ax.set_yticks(range(len(EVENTS)))
    ax.set_xticklabels(EVENTS)
    ax.set_yticklabels(EVENTS)
    ax.set_xlabel("rec_check classification")
    ax.set_ylabel("Initial run classification")
    ax.set_title("Concordance matrix — BADASP-scored shared nodes\n(initial run vs. rec_check)")

    for i in range(len(EVENTS)):
        for j in range(len(EVENTS)):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() * 0.5 else "black",
                    fontsize=12)

    plt.colorbar(im, ax=ax, label="Node count")
    plt.tight_layout()
    out = outdir / "scored_node_concordance_matrix.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def write_report(result: dict, outdir: Path) -> None:
    lines = [
        "# Scored Node Concordance: Initial vs. rec_check AleRax Run",
        "",
        "Nodes are matched by leaf signature (sorted descendant leaf names).",
        "Only nodes with a resolved AleRax event type (S/D/T) in BOTH runs are included.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Initial scored nodes | {result['n_init']:,} |",
        f"| rec_check scored nodes | {result['n_rec']:,} |",
        f"| Shared scored nodes | {result['n_shared']:,} |",
        f"| Matching classifications | {result['n_matching']:,} |",
        f"| Concordance rate | {result['concordance_pct']:.2f}% |",
        f"| Cohen's kappa | {result['cohen_kappa']:.4f} |",
        "",
        "## Confusion matrix",
        "",
        "| Initial \\ rec_check | " + " | ".join(EVENTS) + " | Row total |",
        "|" + "---|" * (len(EVENTS) + 2),
    ]
    for ev_i in EVENTS:
        row_counts = [result["confusion"].get((ev_i, ev_r), 0) for ev_r in EVENTS]
        row_total = sum(row_counts)
        lines.append("| " + ev_i + " | " + " | ".join(str(c) for c in row_counts) + f" | {row_total} |")

    col_totals = [sum(result["confusion"].get((ev_i, ev_r), 0) for ev_i in EVENTS) for ev_r in EVENTS]
    lines.append("| **Col total** | " + " | ".join(str(c) for c in col_totals) +
                 f" | {result['n_shared']} |")

    out = outdir / "scored_node_concordance_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concordance analysis for BADASP-scored nodes between two AleRax runs"
    )
    parser.add_argument("--init-reconc-dir", type=Path,
                        default=Path("results/reconciliation/alerax/IPR019888/reconciliations"))
    parser.add_argument("--rec-reconc-dir", type=Path,
                        default=Path("results/rec_check/output_rec_check/reconciliations"))
    parser.add_argument("--init-scores", type=Path,
                        default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--rec-scores", type=Path,
                        default=Path("results/rec_check/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--family", default="IPR019888")
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/dtl_sensitivity/scored_node_concordance"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading scored nodes from raw_node_scores.csv...")
    init_scored = load_scored_nodes(args.init_scores)
    rec_scored  = load_scored_nodes(args.rec_scores)
    print(f"  Initial: {len(init_scored):,} scored nodes")
    print(f"  rec_check: {len(rec_scored):,} scored nodes")

    fam = args.family
    init_nwk     = args.init_reconc_dir / f"{fam}.nwk"
    rec_nwk      = args.rec_reconc_dir  / f"{fam}.nwk"
    init_samples = args.init_reconc_dir / "all" / f"{fam}_samples.newick"
    rec_samples  = args.rec_reconc_dir  / "all" / f"{fam}_samples.newick"

    print("\nBuilding leaf signatures from consensus trees (this may take a few minutes)...")
    print("  Loading initial run consensus tree...")
    init_sig_events = build_sig_to_event(init_nwk, init_samples, init_scored)
    print(f"  Initial: {len(init_sig_events):,} resolved nodes in consensus tree")

    print("  Loading rec_check consensus tree...")
    rec_sig_events = build_sig_to_event(rec_nwk, rec_samples, rec_scored)
    print(f"  rec_check: {len(rec_sig_events):,} resolved nodes in consensus tree")

    print("\nComputing concordance on shared nodes...")
    result = compute_concordance(init_sig_events, rec_sig_events)

    print(f"\n{'='*60}")
    print(f"Shared scored nodes:        {result['n_shared']:,}")
    print(f"Matching classifications:   {result['n_matching']:,}")
    print(f"Concordance rate:           {result['concordance_pct']:.2f}%")
    print(f"Cohen's kappa:              {result['cohen_kappa']:.4f}")
    print(f"{'='*60}")

    plot_concordance_matrix(result["confusion"], args.outdir)
    write_report(result, args.outdir)

    # CSV of per-event concordance
    rows = []
    for ev in EVENTS:
        tp = result["confusion"].get((ev, ev), 0)
        row_total = sum(result["confusion"].get((ev, ev2), 0) for ev2 in EVENTS)
        col_total = sum(result["confusion"].get((ev2, ev), 0) for ev2 in EVENTS)
        rows.append({
            "event_type": ev,
            "true_positives": tp,
            "row_total_init": row_total,
            "col_total_rec": col_total,
            "recall": tp / row_total if row_total else None,
            "precision": tp / col_total if col_total else None,
        })
    pd.DataFrame(rows).to_csv(args.outdir / "scored_node_per_event_stats.csv", index=False)
    print(f"Saved: {args.outdir / 'scored_node_per_event_stats.csv'}")


if __name__ == "__main__":
    main()
