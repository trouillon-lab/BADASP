"""
Compare unconstrained vs T:D=50:1 fixed AleRax reconciliation results
across both the initial and rec_check gene tree sets.

Empirical benchmark: Tria & Martin (2021) estimate T:D >= 50:1 for prokaryotes.
"""

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent

RUNS = {
    ("Initial", "Unconstrained"): ROOT / "results/reconciliation/alerax/IPR019888/reconciliations/totalSpeciesEventCounts.txt",
    ("Initial", "T:D=50:1"):      ROOT / "results/dtl_sensitivity/alerax_initial_td50/IPR019888/reconciliations/totalSpeciesEventCounts.txt",
    ("rec_check", "Unconstrained"): ROOT / "results/rec_check/output_rec_check/reconciliations/totalSpeciesEventCounts.txt",
    ("rec_check", "T:D=50:1"):    ROOT / "results/dtl_sensitivity/alerax_reccheck_td50/IPR019888/reconciliations/totalSpeciesEventCounts.txt",
}

def load_events(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            rows.append({
                "species": row["species_label"].strip(),
                "S": float(row["speciations"]),
                "D": float(row["duplications"]),
                "T": float(row["transfers"]),
                "L": float(row["losses"]),
            })
    return rows

def total(rows, key):
    return sum(r[key] for r in rows)

# ── load all four datasets ────────────────────────────────────────────────────
data = {k: load_events(v) for k, v in RUNS.items()}

# ── compute summary table ─────────────────────────────────────────────────────
summary = {}
for key, rows in data.items():
    S = total(rows, "S")
    D = total(rows, "D")
    T = total(rows, "T")
    L = total(rows, "L")
    td = T / D if D > 0 else float("inf")
    summary[key] = dict(S=S, D=D, T=T, L=L, TD=td, n=len(rows))

print(f"{'Run':<30} {'Species':>8} {'S':>8} {'D':>8} {'T':>8} {'L':>8} {'T:D':>8}")
print("-" * 82)
for (dataset, model), v in summary.items():
    label = f"{dataset} ({model})"
    print(f"{label:<30} {v['n']:>8} {v['S']:>8,.0f} {v['D']:>8,.0f} "
          f"{v['T']:>8,.0f} {v['L']:>8,.0f} {v['TD']:>7.1f}:1")

# ── Figure 1: event counts bar chart ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("AleRax event counts: unconstrained vs fixed T:D=50:1", fontsize=13, y=1.01)

colors_unc = {"D": "#E07B54", "T": "#5B8DB8", "L": "#7DBB7D"}
colors_td5 = {"D": "#C04020", "T": "#2B5D88", "L": "#4D8B4D"}

for ax, dataset, title in [
    (axes[0], "Initial",    "Initial gene tree (21,218 seqs, 8,775 species)"),
    (axes[1], "rec_check",  "rec_check gene tree (19,528 seqs, 1,986 species)"),
]:
    unc = summary[(dataset, "Unconstrained")]
    td5 = summary[(dataset, "T:D=50:1")]

    x = np.arange(3)
    w = 0.35
    events = ["D", "T", "L"]
    labels = ["Duplications", "Transfers", "Losses"]

    bars_unc = ax.bar(x - w/2, [unc[e] for e in events], w,
                      color=[colors_unc[e] for e in events], label="Unconstrained",
                      edgecolor="white", linewidth=0.5)
    bars_td5 = ax.bar(x + w/2, [td5[e] for e in events], w,
                      color=[colors_td5[e] for e in events], label="T:D=50:1",
                      edgecolor="white", linewidth=0.5, alpha=0.85)

    # annotate % change on top of TD50 bars
    for bar_u, bar_t, e in zip(bars_unc, bars_td5, events):
        pct = 100 * (bar_t.get_height() - bar_u.get_height()) / bar_u.get_height()
        ax.text(bar_t.get_x() + bar_t.get_width()/2,
                bar_t.get_height() + max(unc[e], td5[e]) * 0.015,
                f"{pct:+.0f}%", ha="center", va="bottom", fontsize=8.5,
                color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total events (averaged over 100 samples)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)

    # add T:D annotation
    td_unc = unc["T"] / unc["D"] if unc["D"] else float("inf")
    td_td5 = td5["T"] / td5["D"] if td5["D"] else float("inf")
    ax.text(0.97, 0.97,
            f"Observed T:D\nUnconstrained: {td_unc:.1f}:1\nT:D=50:1 run: {td_td5:.1f}:1",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cccccc"))

    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = ROOT / "results/dtl_sensitivity/event_counts_comparison.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out1}")

# ── Figure 2: per-species T:D ratio distributions ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Per-species T:D ratio distribution", fontsize=13, y=1.01)

# Tria & Martin 2021 empirical estimate (lower bound)
EMPIRICAL_TD = 50.0

for ax, dataset, title in [
    (axes[0], "Initial",    "Initial gene tree"),
    (axes[1], "rec_check",  "rec_check gene tree"),
]:
    unc_rows = data[(dataset, "Unconstrained")]
    td5_rows = data[(dataset, "T:D=50:1")]

    # per-species T:D (only species with D > 0)
    def td_ratios(rows):
        return [r["T"] / r["D"] for r in rows if r["D"] > 0]

    td_unc = td_ratios(unc_rows)
    td_td5 = td_ratios(td5_rows)

    bins = np.logspace(-2, 3, 40)
    ax.hist(td_unc, bins=bins, alpha=0.65, color="#5B8DB8",
            label=f"Unconstrained  (n={len(td_unc)}, median={np.median(td_unc):.1f}:1)")
    ax.hist(td_td5, bins=bins, alpha=0.65, color="#C04020",
            label=f"T:D=50:1  (n={len(td_td5)}, median={np.median(td_td5):.1f}:1)")

    ax.axvline(EMPIRICAL_TD, color="#2a2a2a", lw=1.5, ls="--",
               label=f"Tria & Martin (2021) ≥{EMPIRICAL_TD:.0f}:1")
    ax.set_xscale("log")
    ax.set_xlabel("T:D ratio (log scale, per species)")
    ax.set_ylabel("Number of species")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = ROOT / "results/dtl_sensitivity/td_ratio_distributions.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Figure 3: fraction of species at or above empirical threshold ─────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

thresholds = np.logspace(-1, 3, 200)
plot_configs = [
    ("Initial", "Unconstrained", "#5B8DB8", "-"),
    ("Initial", "T:D=50:1",      "#C04020", "--"),
    ("rec_check", "Unconstrained", "#7DBB7D", "-"),
    ("rec_check", "T:D=50:1",      "#2D6E2D", "--"),
]

for dataset, model, color, ls in plot_configs:
    rows = data[(dataset, model)]
    ratios = [r["T"] / r["D"] for r in rows if r["D"] > 0]
    ratios = np.array(ratios)
    frac = [(ratios >= t).mean() for t in thresholds]
    ax.plot(thresholds, frac, color=color, ls=ls,
            label=f"{dataset} {model}")

ax.axvline(EMPIRICAL_TD, color="#2a2a2a", lw=1.5, ls=":",
           label=f"Empirical lower bound (≥50:1, Tria & Martin 2021)")
ax.set_xscale("log")
ax.set_xlabel("Minimum T:D threshold")
ax.set_ylabel("Fraction of species with T:D ≥ threshold")
ax.set_title("Fraction of species meeting empirical T:D ≥ threshold\n(as a check of model realism)", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out3 = ROOT / "results/dtl_sensitivity/td_threshold_curve.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── print fraction at empirical threshold ─────────────────────────────────────
print(f"\nFraction of species with observed T:D >= {EMPIRICAL_TD:.0f}:1:")
for (dataset, model), rows in data.items():
    ratios = [r["T"] / r["D"] for r in rows if r["D"] > 0]
    frac = np.mean(np.array(ratios) >= EMPIRICAL_TD)
    print(f"  {dataset} ({model}): {frac:.1%}  (n={len(ratios)} species with D>0)")
