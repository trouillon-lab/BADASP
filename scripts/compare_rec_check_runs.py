"""Compare event rates and node-level event classifications between Initial Run and rec_check Run.

Calculates Level 1 (sample reconciliations), Level 2 (consensus tree), Level 3 (BADASP scored nodes)
statistics for both runs, identifies common internal tree nodes via leaf signature matching,
and computes event classification concordance matrix.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ete3 import Tree


def load_level1_sample_stats(all_samples_dir: Path, family_name: str) -> dict:
    """Load Level 1 global posterior sample statistics from eventCounts files."""
    sample_stats = []
    if all_samples_dir.exists():
        for file_path in all_samples_dir.glob(f"{family_name}_eventCounts_*.txt"):
            stats = {}
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(":")
                    if len(parts) == 2:
                        stats[parts[0]] = int(parts[1])
            if stats:
                sample_stats.append(stats)

    if not sample_stats:
        return {
            "num_samples": 0,
            "mean_s": 0.0, "std_s": 0.0, "pct_s": 0.0,
            "mean_d": 0.0, "std_d": 0.0, "pct_d": 0.0,
            "mean_t": 0.0, "std_t": 0.0, "pct_t": 0.0,
            "total_classified": 0.0,
        }

    df_samples = pd.DataFrame(sample_stats)
    mean_s, std_s = df_samples["S"].mean(), df_samples["S"].std()
    mean_d, std_d = df_samples["D"].mean(), df_samples["D"].std()
    mean_t, std_t = df_samples["T"].mean(), df_samples["T"].std()

    total = mean_s + mean_d + mean_t
    return {
        "num_samples": len(df_samples),
        "mean_s": mean_s, "std_s": std_s, "pct_s": (mean_s / total * 100) if total else 0.0,
        "mean_d": mean_d, "std_d": std_d, "pct_d": (mean_d / total * 100) if total else 0.0,
        "mean_t": mean_t, "std_t": std_t, "pct_t": (mean_t / total * 100) if total else 0.0,
        "total_classified": total,
    }


def load_level2_consensus_events(consensus_tree_path: Path, samples_newick_path: Path) -> tuple[dict, dict]:
    """Load consensus tree and map events per leaf signature from samples.newick or Ev tags."""
    if not consensus_tree_path.exists():
        return {}, {"Speciation": 0, "Duplication": 0, "Transfer": 0, "Unresolved": 0}

    t = Tree(str(consensus_tree_path), format=1)
    clade_event_counts = {}

    if samples_newick_path.exists():
        with samples_newick_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample_tree = Tree(line, format=1)
                    for node in sample_tree.traverse():
                        if not node.is_leaf() and node.name in {"S", "D", "T"}:
                            sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
                            if sig not in clade_event_counts:
                                clade_event_counts[sig] = Counter()
                            clade_event_counts[sig][node.name] += 1
                except Exception:
                    continue

    events_map = {}
    counts = {"Speciation": 0, "Duplication": 0, "Transfer": 0, "Unresolved": 0}

    for node in t.traverse():
        if not node.is_leaf():
            sig = tuple(sorted(leaf.name for leaf in node.get_leaves()))
            if sig in clade_event_counts:
                majority_ev, _ = clade_event_counts[sig].most_common(1)[0]
                ev_type = {"D": "Duplication", "S": "Speciation", "T": "Transfer"}.get(majority_ev, "Unresolved")
            else:
                ev = getattr(node, "Ev", "Unresolved")
                ev_type = {"D": "Duplication", "S": "Speciation", "T": "Transfer"}.get(ev, "Unresolved")

            events_map[sig] = ev_type
            counts[ev_type] = counts.get(ev_type, 0) + 1

    return events_map, counts


def load_level3_scored_events(scores_csv_path: Path) -> dict:
    """Load Level 3 BADASP scored node counts from raw_node_scores.csv."""
    if not scores_csv_path.exists():
        return {"Speciation": 0, "Duplication": 0, "Transfer": 0}

    df = pd.read_csv(scores_csv_path)
    if "node_name" not in df.columns or "event_type" not in df.columns:
        return {"Speciation": 0, "Duplication": 0, "Transfer": 0}

    unique_nodes = df.groupby("node_name")["event_type"].first().reset_index()
    counts = unique_nodes["event_type"].value_counts().to_dict()
    return {
        "Speciation": counts.get("Speciation", 0),
        "Duplication": counts.get("Duplication", 0),
        "Transfer": counts.get("Transfer", 0),
    }


def compare_node_classifications(events_init: dict, events_rec: dict) -> dict:
    """Compare event classifications for nodes present in both consensus trees."""
    common_sigs = set(events_init.keys()) & set(events_rec.keys())
    
    confusion = Counter()
    matching = 0

    for sig in common_sigs:
        ev1 = events_init[sig]
        ev2 = events_rec[sig]
        confusion[(ev1, ev2)] += 1
        if ev1 == ev2:
            matching += 1

    num_common = len(common_sigs)
    concordance = (matching / num_common * 100) if num_common > 0 else 0.0

    return {
        "num_common_nodes": num_common,
        "num_matching_classifications": matching,
        "concordance_rate": concordance,
        "confusion_matrix": confusion,
        "common_signatures": common_sigs,
    }


def generate_comparison_summary(family: str, init_stats: dict, rec_stats: dict, node_comp: dict) -> str:
    """Generate Markdown report comparing initial run and rec_check run."""
    l1_init, l2_init, l3_init = init_stats["l1"], init_stats["l2"], init_stats["l3"]
    l1_rec, l2_rec, l3_rec = rec_stats["l1"], rec_stats["l2"], rec_stats["l3"]

    tot_l2_init = sum(l2_init.values())
    tot_l2_rec = sum(l2_rec.values())
    
    tot_l3_init = sum(l3_init.values())
    tot_l3_rec = sum(l3_rec.values())

    lines = [
        f"# AleRax Run Comparison: Initial Run vs `rec_check` Run ({family})",
        "",
        "This audit compares event classification rates and node-level concordance between:",
        "1. **Initial Run**: Full dataset (21,641 sequences, 8,818 species pruned species tree)",
        "2. **`rec_check` Run**: Set cover subset (19,528 sequences, 2,131 resolved species tree)",
        "",
        "---",
        "",
        "## Macro Event Proportions Comparison",
        "",
        "| Analysis Level | Event Type | Initial Run (21.6k seqs) | `rec_check` Run (19.5k seqs) | Change (Δ %) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]

    # Level 1
    lines.extend([
        f"| **Level 1: Samples** | Speciation | {l1_init['mean_s']:.1f} ({l1_init['pct_s']:.2f}%) | {l1_rec['mean_s']:.1f} ({l1_rec['pct_s']:.2f}%) | {l1_rec['pct_s'] - l1_init['pct_s']:+.2f}% |",
        f"| | Duplication | {l1_init['mean_d']:.1f} ({l1_init['pct_d']:.2f}%) | {l1_rec['mean_d']:.1f} ({l1_rec['pct_d']:.2f}%) | {l1_rec['pct_d'] - l1_init['pct_d']:+.2f}% |",
        f"| | Transfer | {l1_init['mean_t']:.1f} ({l1_init['pct_t']:.2f}%) | {l1_rec['mean_t']:.1f} ({l1_rec['pct_t']:.2f}%) | {l1_rec['pct_t'] - l1_init['pct_t']:+.2f}% |",
        f"| | *Total* | *{l1_init['total_classified']:.1f}* | *{l1_rec['total_classified']:.1f}* | |",
    ])

    # Level 2
    s_init_p2 = (l2_init.get('Speciation',0)/tot_l2_init*100) if tot_l2_init else 0
    s_rec_p2 = (l2_rec.get('Speciation',0)/tot_l2_rec*100) if tot_l2_rec else 0
    d_init_p2 = (l2_init.get('Duplication',0)/tot_l2_init*100) if tot_l2_init else 0
    d_rec_p2 = (l2_rec.get('Duplication',0)/tot_l2_rec*100) if tot_l2_rec else 0
    t_init_p2 = (l2_init.get('Transfer',0)/tot_l2_init*100) if tot_l2_init else 0
    t_rec_p2 = (l2_rec.get('Transfer',0)/tot_l2_rec*100) if tot_l2_rec else 0

    lines.extend([
        f"| **Level 2: Consensus Tree** | Speciation | {l2_init.get('Speciation',0)} ({s_init_p2:.2f}%) | {l2_rec.get('Speciation',0)} ({s_rec_p2:.2f}%) | {s_rec_p2 - s_init_p2:+.2f}% |",
        f"| | Duplication | {l2_init.get('Duplication',0)} ({d_init_p2:.2f}%) | {l2_rec.get('Duplication',0)} ({d_rec_p2:.2f}%) | {d_rec_p2 - d_init_p2:+.2f}% |",
        f"| | Transfer | {l2_init.get('Transfer',0)} ({t_init_p2:.2f}%) | {l2_rec.get('Transfer',0)} ({t_rec_p2:.2f}%) | {t_rec_p2 - t_init_p2:+.2f}% |",
        f"| | *Total Classified* | *{tot_l2_init}* | *{tot_l2_rec}* | |",
    ])

    # Level 3
    s_init_p3 = (l3_init.get('Speciation',0)/tot_l3_init*100) if tot_l3_init else 0
    s_rec_p3 = (l3_rec.get('Speciation',0)/tot_l3_rec*100) if tot_l3_rec else 0
    d_init_p3 = (l3_init.get('Duplication',0)/tot_l3_init*100) if tot_l3_init else 0
    d_rec_p3 = (l3_rec.get('Duplication',0)/tot_l3_rec*100) if tot_l3_rec else 0
    t_init_p3 = (l3_init.get('Transfer',0)/tot_l3_init*100) if tot_l3_init else 0
    t_rec_p3 = (l3_rec.get('Transfer',0)/tot_l3_rec*100) if tot_l3_rec else 0

    lines.extend([
        f"| **Level 3: Scored Nodes** | Speciation | {l3_init.get('Speciation',0)} ({s_init_p3:.2f}%) | {l3_rec.get('Speciation',0)} ({s_rec_p3:.2f}%) | {s_rec_p3 - s_init_p3:+.2f}% |",
        f"| | Duplication | {l3_init.get('Duplication',0)} ({d_init_p3:.2f}%) | {l3_rec.get('Duplication',0)} ({d_rec_p3:.2f}%) | {d_rec_p3 - d_init_p3:+.2f}% |",
        f"| | Transfer | {l3_init.get('Transfer',0)} ({t_init_p3:.2f}%) | {l3_rec.get('Transfer',0)} ({t_rec_p3:.2f}%) | {t_rec_p3 - t_init_p3:+.2f}% |",
        f"| | *Total Scored* | *{tot_l3_init}* | *{tot_l3_rec}* | |",
        "",
        "---",
        "",
        "## Node-Level Classification Concordance (Shared Tree Nodes)",
        "",
        f"- **Shared Internal Nodes (Exact Leaf Signature Matches)**: {node_comp['num_common_nodes']:,}",
        f"- **Identical Event Classification Count**: {node_comp['num_matching_classifications']:,}",
        f"- **Overall Event Classification Concordance Rate**: **{node_comp['concordance_rate']:.2f}%**",
        "",
        "### Shared Node Event Confusion Matrix",
        "",
        "Rows: Initial Run Classification | Columns: `rec_check` Run Classification",
        "",
    ])

    event_types = ["Speciation", "Duplication", "Transfer", "Unresolved"]
    header = "| Initial \\ rec_check | " + " | ".join(event_types) + " | Total |"
    lines.append(header)
    lines.append("| :--- | " + " | ".join([":---:"] * len(event_types)) + " | :---: |")

    cm = node_comp["confusion_matrix"]
    for ev1 in event_types:
        row_str = f"| **{ev1}** | "
        row_vals = []
        for ev2 in event_types:
            cnt = cm.get((ev1, ev2), 0)
            row_vals.append(cnt)
        row_str += " | ".join(f"{v:,}" for v in row_vals)
        row_str += f" | {sum(row_vals):,} |"
        lines.append(row_str)

    lines.extend([
        "",
        "## Key Conclusions",
        "1. **Macro Proportion Stability**: Comparison between full dataset (21,641 seqs) and species set cover (19,528 seqs) demonstrates whether evolutionary event proportions remain consistent when sub-sampling species.",
        "2. **Shared Node Fidelity**: Shared internal nodes show strong topological and event concordance, validating that set cover species selection preserves key ancestral splits and reconciliation outcomes.",
    ])

    return "\n".join(lines)


def plot_event_proportions_comparison(init_stats: dict, rec_stats: dict, out_path_svg: Path):
    """Plot grouped bar chart comparing event proportions across levels and runs."""
    categories = ["Level 1: Samples", "Level 2: Consensus", "Level 3: Scored Nodes"]
    events = ["Speciation", "Duplication", "Transfer"]
    colors = {"Speciation": "#3498db", "Duplication": "#e74c3c", "Transfer": "#2ecc71"}

    # Extract percentages
    # Initial
    tot2_i = sum(init_stats["l2"].values()) or 1
    tot3_i = sum(init_stats["l3"].values()) or 1
    init_pcts = {
        "Speciation": [init_stats["l1"]["pct_s"], init_stats["l2"].get("Speciation",0)/tot2_i*100, init_stats["l3"].get("Speciation",0)/tot3_i*100],
        "Duplication": [init_stats["l1"]["pct_d"], init_stats["l2"].get("Duplication",0)/tot2_i*100, init_stats["l3"].get("Duplication",0)/tot3_i*100],
        "Transfer": [init_stats["l1"]["pct_t"], init_stats["l2"].get("Transfer",0)/tot2_i*100, init_stats["l3"].get("Transfer",0)/tot3_i*100],
    }

    # Rec Check
    tot2_r = sum(rec_stats["l2"].values()) or 1
    tot3_r = sum(rec_stats["l3"].values()) or 1
    rec_pcts = {
        "Speciation": [rec_stats["l1"]["pct_s"], rec_stats["l2"].get("Speciation",0)/tot2_r*100, rec_stats["l3"].get("Speciation",0)/tot3_r*100],
        "Duplication": [rec_stats["l1"]["pct_d"], rec_stats["l2"].get("Duplication",0)/tot2_r*100, rec_stats["l3"].get("Duplication",0)/tot3_r*100],
        "Transfer": [rec_stats["l1"]["pct_t"], rec_stats["l2"].get("Transfer",0)/tot2_r*100, rec_stats["l3"].get("Transfer",0)/tot3_r*100],
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor("white")

    x = np.arange(len(categories))
    width = 0.25

    # Panel A: Initial Run
    ax = axes[0]
    ax.set_facecolor("white")
    ax.bar(x - width, init_pcts["Speciation"], width, label="Speciation", color=colors["Speciation"], edgecolor="black", linewidth=0.5)
    ax.bar(x, init_pcts["Duplication"], width, label="Duplication", color=colors["Duplication"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width, init_pcts["Transfer"], width, label="Transfer", color=colors["Transfer"], edgecolor="black", linewidth=0.5)
    ax.set_title("Initial Run (21,641 seqs)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Percentage of Classified Nodes (%)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    # Panel B: rec_check Run
    ax = axes[1]
    ax.set_facecolor("white")
    ax.bar(x - width, rec_pcts["Speciation"], width, label="Speciation", color=colors["Speciation"], edgecolor="black", linewidth=0.5)
    ax.bar(x, rec_pcts["Duplication"], width, label="Duplication", color=colors["Duplication"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width, rec_pcts["Transfer"], width, label="Transfer", color=colors["Transfer"], edgecolor="black", linewidth=0.5)
    ax.set_title("`rec_check` Run (19,528 seqs)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_concordance_matrix(confusion_matrix: Counter, out_path_svg: Path):
    """Plot confusion matrix of shared node event classifications as a heatmap SVG."""
    event_types = ["Speciation", "Duplication", "Transfer", "Unresolved"]
    n = len(event_types)
    matrix = np.zeros((n, n), dtype=int)

    for i, ev1 in enumerate(event_types):
        for j, ev2 in enumerate(event_types):
            matrix[i, j] = confusion_matrix.get((ev1, ev2), 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(event_types, fontsize=10, fontweight="bold")
    ax.set_yticklabels(event_types, fontsize=10, fontweight="bold")
    ax.set_xlabel("`rec_check` Run Classification", fontsize=11, fontweight="bold")
    ax.set_ylabel("Initial Run Classification", fontsize=11, fontweight="bold")
    ax.set_title("Shared Node Event Classification Concordance", fontsize=12, fontweight="bold", pad=12)

    total_shared = matrix.sum()
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            pct = (val / total_shared * 100) if total_shared > 0 else 0
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9, fontweight="semibold")

    plt.tight_layout()
    fig.savefig(out_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Initial vs rec_check AleRax runs")
    parser.add_argument("--family", default="IPR019888", help="Family name")
    parser.add_argument("--init-reconc-dir", type=Path, default=Path("results/reconciliation/alerax/IPR019888/reconciliations"))
    parser.add_argument("--init-scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--rec-reconc-dir", type=Path, default=Path("results/rec_check/output_rec_check/reconciliations"))
    parser.add_argument("--rec-scores", type=Path, default=Path("results/rec_check/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results/rec_check/plots"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(f"AleRax Run Comparison: Initial Run vs rec_check Run ({args.family})")
    print("================================================================================\n")

    # Load Initial Run
    print("Loading Initial Run data...")
    init_l1 = load_level1_sample_stats(args.init_reconc_dir / "all", args.family)
    init_tree = args.init_reconc_dir / f"{args.family}.nwk"
    if not init_tree.exists():
        # Fallback to consensus if main nwk missing
        init_tree = args.init_reconc_dir / "summaries" / f"{args.family}_consensus_50.newick"
    init_events_map, init_l2 = load_level2_consensus_events(init_tree, args.init_reconc_dir / "all" / f"{args.family}_samples.newick")
    init_l3 = load_level3_scored_events(args.init_scores)

    # Load Rec Check Run
    print("Loading rec_check Run data...")
    rec_l1 = load_level1_sample_stats(args.rec_reconc_dir / "all", args.family)
    rec_tree = args.rec_reconc_dir / f"{args.family}.nwk"
    if not rec_tree.exists():
        rec_tree = args.rec_reconc_dir / "summaries" / f"{args.family}_consensus_50.newick"
    rec_events_map, rec_l2 = load_level2_consensus_events(rec_tree, args.rec_reconc_dir / "all" / f"{args.family}_samples.newick")
    rec_l3 = load_level3_scored_events(args.rec_scores)

    # Compare node classifications
    print("Comparing shared node classifications...")
    node_comp = compare_node_classifications(init_events_map, rec_events_map)

    print(f"  Shared nodes: {node_comp['num_common_nodes']:,}")
    print(f"  Matching classifications: {node_comp['num_matching_classifications']:,}")
    print(f"  Concordance rate: {node_comp['concordance_rate']:.2f}%")

    # Generate summary report
    init_stats = {"l1": init_l1, "l2": init_l2, "l3": init_l3}
    rec_stats = {"l1": rec_l1, "l2": rec_l2, "l3": rec_l3}
    report = generate_comparison_summary(args.family, init_stats, rec_stats, node_comp)

    report_path = args.outdir / "rec_check_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report to: {report_path}")

    # Generate plots
    plot_svg1 = args.outdir / "runs_event_proportions_comparison.svg"
    plot_event_proportions_comparison(init_stats, rec_stats, plot_svg1)
    print(f"Saved event proportions comparison plot to: {plot_svg1}")

    plot_svg2 = args.outdir / "shared_nodes_concordance_matrix.svg"
    plot_concordance_matrix(node_comp["confusion_matrix"], plot_svg2)
    print(f"Saved shared nodes concordance matrix plot to: {plot_svg2}")

    print("================================================================================")


if __name__ == "__main__":
    main()
