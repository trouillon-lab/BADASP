#!/usr/bin/env python3
"""
analyze_reccheck_dup_depth.py

Compare the depth and clade-size distributions of Duplication-classified
nodes between the initial AleRax run and the rec_check run.

The rec_check includes all paralogs within selected species (CD-HIT was
not applied), so it sees many within-species duplications that are
collapsed in the initial run. This script tests whether those "extra"
rec_check duplications are shallow (near tips, high distance-from-root)
vs. deep (ancestral, low distance-from-root).

Output: plots and summary stats in results/dtl_sensitivity/reccheck_dup_depth/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_unique_nodes(csv_path: Path) -> pd.DataFrame:
    """Return one row per (node_name, event_type) with depth and clade_size."""
    df = pd.read_csv(csv_path)
    return (
        df.groupby("node_name")
        .agg(
            event_type=("event_type", "first"),
            distance_from_root=("distance_from_root", "first"),
            clade_size_total=("clade_size_total", "first"),
        )
        .reset_index()
    )


def describe_event(df: pd.DataFrame, event: str) -> dict:
    sub = df[df["event_type"] == event]
    return {
        "n": len(sub),
        "depth_median": sub["distance_from_root"].median(),
        "depth_mean": sub["distance_from_root"].mean(),
        "depth_q25": sub["distance_from_root"].quantile(0.25),
        "depth_q75": sub["distance_from_root"].quantile(0.75),
        "clade_median": sub["clade_size_total"].median(),
        "clade_q25": sub["clade_size_total"].quantile(0.25),
        "clade_q75": sub["clade_size_total"].quantile(0.75),
    }


def plot_distributions(
    init_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    outdir: Path,
) -> None:
    dup_init = init_df[init_df["event_type"] == "Duplication"]
    dup_rec  = rec_df[rec_df["event_type"] == "Duplication"]

    all_events = ["Speciation", "Duplication", "Transfer"]
    colors = {"Speciation": "#4e79a7", "Duplication": "#f28e2b", "Transfer": "#59a14f"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Duplication node distributions: initial run vs. rec_check", fontsize=14)

    # --- Panel A: distance from root (depth) ---
    ax = axes[0]
    data_a = [dup_init["distance_from_root"].dropna(), dup_rec["distance_from_root"].dropna()]
    labels_a = [
        f"Initial\n(n={len(dup_init):,})",
        f"rec_check\n(n={len(dup_rec):,})",
    ]
    vp = ax.violinplot(data_a, positions=[1, 2], showmedians=True, showextrema=False)
    for patch, col in zip(vp["bodies"], ["#4e79a7", "#f28e2b"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels_a)
    ax.set_ylabel("Distance from root")
    ax.set_title("Depth of Duplication nodes\n(higher = more recent / near tips)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(
        dup_init["distance_from_root"].dropna(),
        dup_rec["distance_from_root"].dropna(),
        alternative="two-sided",
    )
    ptext = f"p = {pval:.2e}" if pval >= 1e-4 else f"p < 10⁻⁴"
    ax.text(0.5, 0.98, f"Mann-Whitney U, {ptext}", transform=ax.transAxes,
            ha="center", va="top", fontsize=9)

    # --- Panel B: clade size ---
    ax = axes[1]
    data_b = [dup_init["clade_size_total"].dropna(), dup_rec["clade_size_total"].dropna()]
    vp2 = ax.violinplot(data_b, positions=[1, 2], showmedians=True, showextrema=False)
    for patch, col in zip(vp2["bodies"], ["#4e79a7", "#f28e2b"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    vp2["cmedians"].set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels_a)
    ax.set_ylabel("Total clade size (leaf count)")
    ax.set_title("Clade size of Duplication nodes\n(smaller = more localized)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stat2, pval2 = stats.mannwhitneyu(
        dup_init["clade_size_total"].dropna(),
        dup_rec["clade_size_total"].dropna(),
        alternative="two-sided",
    )
    ptext2 = f"p = {pval2:.2e}" if pval2 >= 1e-4 else f"p < 10⁻⁴"
    ax.text(0.5, 0.98, f"Mann-Whitney U, {ptext2}", transform=ax.transAxes,
            ha="center", va="top", fontsize=9)

    plt.tight_layout()
    out_path = outdir / "dup_depth_distributions.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_event_counts(
    init_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    outdir: Path,
) -> None:
    events = ["Speciation", "Duplication", "Transfer"]
    colors = {"Speciation": "#4e79a7", "Duplication": "#f28e2b", "Transfer": "#59a14f"}

    init_counts = {e: (init_df["event_type"] == e).sum() for e in events}
    rec_counts  = {e: (rec_df["event_type"] == e).sum() for e in events}

    x = np.arange(len(events))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, [init_counts[e] for e in events], width,
                   label="Initial run", color="#4e79a7", alpha=0.85)
    bars2 = ax.bar(x + width/2, [rec_counts[e] for e in events], width,
                   label="rec_check run", color="#f28e2b", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylabel("Unique scored nodes")
    ax.set_title("Scored node event classification counts\nper AleRax run")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = outdir / "event_counts_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_depth_scatter(
    init_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Scatter of depth vs. clade_size for D nodes, both runs overlaid."""
    dup_init = init_df[init_df["event_type"] == "Duplication"]
    dup_rec  = rec_df[rec_df["event_type"] == "Duplication"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        dup_init["clade_size_total"], dup_init["distance_from_root"],
        s=20, alpha=0.5, color="#4e79a7", label=f"Initial (n={len(dup_init):,})", edgecolor="none"
    )
    ax.scatter(
        dup_rec["clade_size_total"], dup_rec["distance_from_root"],
        s=20, alpha=0.5, color="#f28e2b", label=f"rec_check (n={len(dup_rec):,})", edgecolor="none"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Total clade size (log scale)")
    ax.set_ylabel("Distance from root")
    ax.set_title("Duplication nodes: depth vs. clade size\n(intended to reveal whether extra rec_check D-nodes are shallow/small)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = outdir / "dup_depth_vs_cladesize.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_stats_table(
    init_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    outdir: Path,
) -> None:
    rows = []
    for event in ["Speciation", "Duplication", "Transfer"]:
        for label, df in [("initial", init_df), ("rec_check", rec_df)]:
            d = describe_event(df, event)
            rows.append({
                "run": label,
                "event_type": event,
                "n_nodes": d["n"],
                "depth_median": round(d["depth_median"], 4) if d["n"] else None,
                "depth_q25": round(d["depth_q25"], 4) if d["n"] else None,
                "depth_q75": round(d["depth_q75"], 4) if d["n"] else None,
                "clade_size_median": round(d["clade_median"], 1) if d["n"] else None,
                "clade_size_q25": round(d["clade_q25"], 1) if d["n"] else None,
                "clade_size_q75": round(d["clade_q75"], 1) if d["n"] else None,
            })

    out_csv = outdir / "dup_depth_summary_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare duplication depth between AleRax runs")
    parser.add_argument("--init-scores", type=Path,
                        default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--rec-scores", type=Path,
                        default=Path("results/rec_check/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("results/dtl_sensitivity/reccheck_dup_depth"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading scored nodes...")
    init_df = load_unique_nodes(args.init_scores)
    rec_df  = load_unique_nodes(args.rec_scores)
    print(f"  Initial run: {len(init_df):,} unique nodes")
    print(f"  rec_check:   {len(rec_df):,} unique nodes")

    print("\nGenerating plots...")
    plot_event_counts(init_df, rec_df, args.outdir)
    plot_distributions(init_df, rec_df, args.outdir)
    plot_depth_scatter(init_df, rec_df, args.outdir)
    write_stats_table(init_df, rec_df, args.outdir)

    # Quick console summary
    n_d_init = (init_df["event_type"] == "Duplication").sum()
    n_d_rec  = (rec_df["event_type"] == "Duplication").sum()
    med_depth_init = init_df.loc[init_df["event_type"] == "Duplication", "distance_from_root"].median()
    med_depth_rec  = rec_df.loc[rec_df["event_type"] == "Duplication", "distance_from_root"].median()
    print(f"\nDuplication node counts: initial={n_d_init}, rec_check={n_d_rec} (+{n_d_rec - n_d_init})")
    print(f"Median depth of D nodes: initial={med_depth_init:.4f}, rec_check={med_depth_rec:.4f}")
    direction = "shallower (closer to tips)" if med_depth_rec > med_depth_init else "deeper (more ancestral)"
    print(f"  → rec_check D nodes tend to be {direction}")


if __name__ == "__main__":
    main()
