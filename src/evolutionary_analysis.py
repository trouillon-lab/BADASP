from __future__ import annotations
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import re
import subprocess
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Phylo, SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBParser
from Bio.Phylo.BaseTree import Clade, Tree
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from src.pdb_mapper import PDBMapper


LEVELS = ["groups", "families", "subfamilies"]
DUPLICATION_LEVEL = "duplications"
LEVEL_MAP = {
    "groups": "group",
    "families": "family",
    "subfamilies": "subfamily",
}


def _ensure_node_names(tree) -> None:
    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"


def _leaf_signature(node: Clade) -> Tuple[str, ...]:
    return tuple(sorted(str(terminal.name) for terminal in node.get_terminals() if terminal.name))


def _build_signature_to_named_node(tree: Tree) -> Dict[Tuple[str, ...], str]:
    signatures: Dict[Tuple[str, ...], str] = {}
    for node in tree.get_nonterminals(order="level"):
        if not node.name:
            continue
        signature = _leaf_signature(node)
        if signature:
            signatures[signature] = str(node.name)
    return signatures


def _remap_named_nodes_to_plot_tree(
    plot_tree: Tree,
    named_tree_path: Optional[Path],
) -> Dict[str, str]:
    _ensure_node_names(plot_tree)
    if named_tree_path is None or not named_tree_path.exists():
        return {}

    named_tree = Phylo.read(str(named_tree_path), "newick")
    _ensure_node_names(named_tree)
    named_signatures = _build_signature_to_named_node(named_tree)

    plot_signature_to_name = {
        _leaf_signature(node): str(node.name)
        for node in plot_tree.get_nonterminals(order="level")
        if node.name
    }

    remap: Dict[str, str] = {}
    for signature, source_name in named_signatures.items():
        mapped_name = plot_signature_to_name.get(signature)
        if mapped_name:
            remap[source_name] = mapped_name
    return remap


def calculate_lca_depth(tree_path: Path, member_names: Sequence[str]) -> float:
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)
    lca = tree.common_ancestor(list(member_names))
    return float(tree.distance(tree.root, lca))


def calculate_ca_distance_matrix(pdb_path: Path, residue_numbers: Sequence[int]) -> pd.DataFrame:
    structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))

    ca_coords: Dict[int, np.ndarray] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                resseq = int(residue.id[1])
                if resseq not in residue_numbers:
                    continue
                if "CA" not in residue:
                    continue
                ca_coords[resseq] = residue["CA"].coord
            if ca_coords:
                break
        if ca_coords:
            break

    ordered = [int(r) for r in residue_numbers if int(r) in ca_coords]
    matrix = pd.DataFrame(index=ordered, columns=ordered, dtype=float)
    for left in ordered:
        for right in ordered:
            matrix.loc[left, right] = float(np.linalg.norm(ca_coords[left] - ca_coords[right]))
    return matrix


def _normalize_residue_number(value: object) -> Optional[int]:
    """Normalize mapper outputs to a single integer residue number.

    Mapper values may be ints, strings, tuples like (chain, resnum), or lists
    of such tuples when multiple chains align to one MSA column.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer, float, np.floating, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, tuple) and len(value) >= 2:
        return _normalize_residue_number(value[1])
    if isinstance(value, list):
        for candidate in value:
            normalized = _normalize_residue_number(candidate)
            if normalized is not None:
                return normalized
        return None
    return None


def compute_coevolution_matrix(events_df: pd.DataFrame) -> pd.DataFrame:
    required = {"branch_id", "position"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    branch_sets: Dict[int, set] = {}
    for position, group in events_df.groupby("position"):
        branch_sets[int(position)] = set(group["branch_id"].astype(str))

    positions = sorted(branch_sets)
    matrix = pd.DataFrame(index=positions, columns=positions, dtype=float)
    for left in positions:
        for right in positions:
            if left == right:
                matrix.loc[left, right] = 1.0
                continue
            union = branch_sets[left] | branch_sets[right]
            intersection = branch_sets[left] & branch_sets[right]
            matrix.loc[left, right] = float(len(intersection) / len(union)) if union else 0.0
    return matrix


def classify_physicochemical_shift(
    charge_change: str,
    hydrophobicity_change: str,
    volume_delta: float,
    volume_threshold: float = 45.0,
) -> str:
    charge_parts = str(charge_change).split("->")
    hydro_parts = str(hydrophobicity_change).split("->")
    charge_shift = len(charge_parts) == 2 and charge_parts[0] != charge_parts[1]
    hydro_shift = len(hydro_parts) == 2 and hydro_parts[0] != hydro_parts[1]
    size_shift = bool(pd.notna(volume_delta)) and abs(float(volume_delta)) >= float(volume_threshold)

    shift_count = int(charge_shift) + int(hydro_shift) + int(size_shift)
    if shift_count >= 2:
        return "multiple_complex"
    if charge_shift:
        return "charge_shift"
    if hydro_shift:
        return "hydrophobicity_shift"
    if size_shift:
        return "size_shift"
    return "none"


def rank_top_functional_sdps(
    subfamily_scores_df: pd.DataFrame,
    coevolution_matrix_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    if subfamily_scores_df.empty:
        return pd.DataFrame(
            columns=[
                "position",
                "switch_count",
                "max_score",
                "mean_coevolution",
                "major_transition_count",
                "shift_type",
                "functional_sdp_score",
            ]
        )

    scores = subfamily_scores_df[["position", "switch_count", "max_score"]].copy()
    scores["position"] = scores["position"].astype(int)

    coevo_strength: Dict[int, float] = {}
    if not coevolution_matrix_df.empty:
        for pos in coevolution_matrix_df.index:
            row = coevolution_matrix_df.loc[pos].drop(labels=[pos], errors="ignore")
            coevo_strength[int(pos)] = float(row.mean()) if not row.empty else 0.0
    scores["mean_coevolution"] = scores["position"].map(coevo_strength).fillna(0.0)

    shifts = shifts_df.copy()
    if shifts.empty:
        shifts = pd.DataFrame(columns=["position", "major_transition_count", "charge_change", "hydrophobicity_change", "volume_change"])
    shifts["position"] = shifts["position"].astype(int)
    shifts["shift_type"] = shifts.apply(
        lambda r: classify_physicochemical_shift(r["charge_change"], r["hydrophobicity_change"], r["volume_change"]),
        axis=1,
    )
    shifts["shift_strength"] = (
        shifts["major_transition_count"].fillna(0).astype(float)
        + shifts["volume_change"].abs().fillna(0).astype(float) / 25.0
        + shifts["shift_type"].isin(["multiple_complex", "charge_shift", "hydrophobicity_shift", "size_shift"]).astype(float)
    )

    merged = scores.merge(
        shifts[["position", "major_transition_count", "shift_type", "shift_strength"]],
        on="position",
        how="left",
    )
    merged["major_transition_count"] = merged["major_transition_count"].fillna(0.0)
    merged["shift_type"] = merged["shift_type"].fillna("none")
    merged["shift_strength"] = merged["shift_strength"].fillna(0.0)

    def _minmax(series: pd.Series) -> pd.Series:
        min_val = float(series.min())
        max_val = float(series.max())
        if max_val - min_val < 1e-12:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - min_val) / (max_val - min_val)

    merged["switch_norm"] = _minmax(merged["switch_count"].astype(float))
    merged["coevo_norm"] = _minmax(merged["mean_coevolution"].astype(float))
    merged["shift_norm"] = _minmax(merged["shift_strength"].astype(float))
    merged["functional_sdp_score"] = 0.45 * merged["switch_norm"] + 0.35 * merged["coevo_norm"] + 0.20 * merged["shift_norm"]

    cols = [
        "position",
        "switch_count",
        "max_score",
        "mean_coevolution",
        "major_transition_count",
        "shift_type",
        "functional_sdp_score",
    ]
    ranked = merged.sort_values(["functional_sdp_score", "switch_count", "max_score"], ascending=[False, False, False])[cols]
    return ranked.head(top_n).reset_index(drop=True)


def count_switches_per_domain(events_df: pd.DataFrame, domain_arch: Dict[str, Sequence[int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {domain: 0 for domain in domain_arch}
    if events_df.empty:
        return counts

    positions = events_df["position"].astype(int).tolist()
    for pos in positions:
        for domain, span in domain_arch.items():
            start, end = int(span[0]), int(span[1])
            if start <= pos <= end:
                counts[domain] += 1
    return counts


def assign_coevolution_communities(matrix: pd.DataFrame, distance_cut: float = 0.6) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame(columns=["position", "community_id"])

    positions = [int(pos) for pos in matrix.index.tolist()]
    if len(positions) == 1:
        return pd.DataFrame({"position": positions, "community_id": [1]})

    dist_matrix = 1.0 - matrix.to_numpy(dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    labels = hierarchy.fcluster(linkage, t=float(distance_cut), criterion="distance")
    return pd.DataFrame({"position": positions, "community_id": labels.astype(int)})


def extract_taxon_label(header: str) -> str:
    os_match = re.search(r"OS=([^=]+?)(?:\sOX=|\sGN=|\sPE=|\sSV=|$)", header)
    if os_match:
        return os_match.group(1).strip()

    bracket_match = re.search(r"\[([^\]]+)\]", header)
    if bracket_match:
        return bracket_match.group(1).strip()

    tokens = str(header).split("|")
    if len(tokens) >= 3 and tokens[2].strip():
        return tokens[2].strip().split()[0]

    words = str(header).split()
    if words:
        return words[0].strip()
    return "Unknown"


def _plot_clustered_heatmap(
    matrix: pd.DataFrame,
    output_svg: Path,
    title: str,
    cmap: str,
    cbar_label: Optional[str] = None,
    is_distance_matrix: bool = False,
) -> None:
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    if len(matrix.index) > 1:
        if is_distance_matrix:
            condensed = squareform(values, checks=False)
        else:
            condensed = pdist(values, metric="euclidean")
        linkage = hierarchy.linkage(condensed, method="average")
    else:
        linkage = None

    cluster = sns.clustermap(
        matrix,
        cmap=cmap,
        linewidths=0.0,
        figsize=(10, 9),
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage,
        col_linkage=linkage,
        cbar_kws={"label": cbar_label} if cbar_label else None,
    )
    cluster.fig.suptitle(title, y=1.02)
    cluster.fig.savefig(output_svg, format="svg")
    plt.close(cluster.fig)


def _plot_switch_timeline(events_df: pd.DataFrame, output_svg: Path) -> None:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    palette = {
        "groups": "#1F77B4",
        "families": "#D95F02",
        "subfamilies": "#2CA02C",
        DUPLICATION_LEVEL: "#B24A2A",
    }

    plt.figure(figsize=(11, 6))
    if "Event_Type" in events_df.columns and events_df["Event_Type"].dropna().nunique() > 1:
        sns.histplot(
            data=events_df,
            x="root_distance",
            hue="Event_Type",
            bins=40,
            stat="count",
            element="step",
            fill=False,
            common_bins=True,
            palette={"Duplication": "#B24A2A", "Speciation": "#2A6FB2", "Unknown": "#7A7A7A"},
        )
    elif "level" in events_df.columns and events_df["level"].nunique() > 1:
        sns.histplot(
            data=events_df,
            x="root_distance",
            hue="level",
            bins=40,
            stat="count",
            element="step",
            fill=False,
            common_bins=True,
            palette=palette,
        )
    else:
        sns.histplot(
            data=events_df,
            x="root_distance",
            bins=40,
            stat="count",
            color=palette[DUPLICATION_LEVEL],
        )
    plt.xlabel("Distance from Root")
    plt.ylabel("Switch Frequency")
    plt.title("Evolutionary Timeline of BADASP Switch Events")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def _layer_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"layer_(\d+)", path.name)
    layer_index = int(match.group(1)) if match else 10**9
    return layer_index, path.name


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def build_global_layer_summary(scores_root: Path, output_csv: Optional[Path] = None) -> pd.DataFrame:
    scores_root = Path(scores_root)
    output_csv = Path(output_csv) if output_csv is not None else scores_root / "global_layer_summary.csv"
    source_summary_path = scores_root / "badasp_layer_summary.csv"
    layer_dirs = sorted(scores_root.glob("layer_*"), key=_layer_sort_key)

    if not layer_dirs and source_summary_path.exists() and source_summary_path.stat().st_size > 0:
        summary_df = pd.read_csv(source_summary_path)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        return summary_df

    rows: List[dict] = []
    for layer_dir in layer_dirs:
        combined_pairwise = _read_csv_if_exists(layer_dir / "raw_pairwise_combined.csv")
        raw_dup = _read_csv_if_exists(layer_dir / "raw_pairwise_duplications.csv")
        raw_spec = _read_csv_if_exists(layer_dir / "raw_pairwise_speciations.csv")
        dup_node_source = raw_dup if not raw_dup.empty else _read_csv_if_exists(layer_dir / "badasp_sdps_duplications.csv")
        spec_node_source = raw_spec if not raw_spec.empty else _read_csv_if_exists(layer_dir / "badasp_sdps_speciations.csv")

        def _count_valid_nodes(df: pd.DataFrame) -> int:
            if df.empty:
                return 0
            for column_name in ("duplication_node", "source_parent_node", "lca_node_name", "parent_node", "pair"):
                if column_name in df.columns:
                    values = df[column_name].dropna().astype(str)
                    if not values.empty:
                        return int(values.nunique())
            return int(len(df))

        valid_duplication_nodes = _count_valid_nodes(dup_node_source)
        valid_speciation_nodes = _count_valid_nodes(spec_node_source)

        layer_index = int(re.search(r"layer_(\d+)", layer_dir.name).group(1)) if re.search(r"layer_(\d+)", layer_dir.name) else 0
        linkage_threshold = float("nan")
        if not combined_pairwise.empty and "layer_threshold" in combined_pairwise.columns:
            thresholds = pd.to_numeric(combined_pairwise["layer_threshold"], errors="coerce").dropna()
            if not thresholds.empty:
                linkage_threshold = float(thresholds.iloc[0])

        valid_pairs = 0
        if not combined_pairwise.empty:
            pair_cols = [col for col in ["duplication_node", "left_child", "right_child"] if col in combined_pairwise.columns]
            if len(pair_cols) == 3:
                valid_pairs = int(combined_pairwise[pair_cols].drop_duplicates().shape[0])
            elif "pair" in combined_pairwise.columns:
                valid_pairs = int(combined_pairwise[["pair"]].drop_duplicates().shape[0])

        # Comparisons (valid sister clade pairs)
        dup_comps = raw_dup["pair"].nunique() if not raw_dup.empty and "pair" in raw_dup.columns else 0
        spec_comps = raw_spec["pair"].nunique() if not raw_spec.empty and "pair" in raw_spec.columns else 0
        comb_comps = combined_pairwise["pair"].nunique() if not combined_pairwise.empty and "pair" in combined_pairwise.columns else 0

        # Switch events (rows >= 95th percentile score)
        dup_switches = 0
        if not raw_dup.empty and "score" in raw_dup.columns:
            scores = pd.to_numeric(raw_dup["score"], errors="coerce").dropna()
            if not scores.empty:
                thresh = np.percentile(scores, 95)
                dup_switches = int((scores >= thresh).sum())
        else:
            dup_sdps = _read_csv_if_exists(layer_dir / "badasp_sdps_duplications.csv")
            dup_switches = len(dup_sdps)

        spec_switches = 0
        if not raw_spec.empty and "score" in raw_spec.columns:
            scores = pd.to_numeric(raw_spec["score"], errors="coerce").dropna()
            if not scores.empty:
                thresh = np.percentile(scores, 95)
                spec_switches = int((scores >= thresh).sum())
        else:
            spec_sdps = _read_csv_if_exists(layer_dir / "badasp_sdps_speciations.csv")
            spec_switches = len(spec_sdps)

        comb_switches = 0
        if not combined_pairwise.empty and "score" in combined_pairwise.columns:
            scores = pd.to_numeric(combined_pairwise["score"], errors="coerce").dropna()
            if not scores.empty:
                thresh = np.percentile(scores, 95)
                comb_switches = int((scores >= thresh).sum())
        else:
            comb_sdps = _read_csv_if_exists(layer_dir / "badasp_sdps_combined.csv")
            comb_switches = len(comb_sdps)

        percentile_threshold = float("nan")
        if not combined_pairwise.empty and "score" in combined_pairwise.columns:
            scores = pd.to_numeric(combined_pairwise["score"], errors="coerce").dropna()
            if not scores.empty:
                percentile_threshold = float(np.percentile(scores, 95))

        rows.append(
            {
                "layer_index": layer_index,
                "linkage_threshold": linkage_threshold,
                "number_valid_pairs": comb_comps if comb_comps > 0 else valid_pairs,
                "number_duplication_pairs": dup_comps,
                "number_speciation_pairs": spec_comps,
                "valid_duplication_nodes": valid_duplication_nodes,
                "valid_speciation_nodes": valid_speciation_nodes,
                "total_valid_nodes": int(valid_duplication_nodes + valid_speciation_nodes),
                "95th_percentile_threshold": percentile_threshold,
                "total_duplication_sdps": dup_switches,
                "total_speciation_sdps": spec_switches,
                "total_combined_sdps": comb_switches,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("layer_index").reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    if source_summary_path != output_csv:
        summary_df.to_csv(source_summary_path, index=False)
    return summary_df


def plot_layerwise_switch_timeline(summary_df: pd.DataFrame, output_svg: Path) -> None:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return

    timeline = summary_df.sort_values("layer_index").copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = timeline["layer_index"].astype(int)

    duplication_nodes = (
        timeline["valid_duplication_nodes"].astype(float)
        if "valid_duplication_nodes" in timeline.columns
        else timeline["number_duplication_pairs"].astype(float)
        if "number_duplication_pairs" in timeline.columns
        else pd.Series(0.0, index=timeline.index)
    )
    speciation_nodes = (
        timeline["valid_speciation_nodes"].astype(float)
        if "valid_speciation_nodes" in timeline.columns
        else timeline["number_speciation_pairs"].astype(float)
        if "number_speciation_pairs" in timeline.columns
        else pd.Series(0.0, index=timeline.index)
    )
    total_nodes = (
        timeline["number_valid_pairs"].astype(float)
        if "number_valid_pairs" in timeline.columns
        else timeline["candidate_pairs"].astype(float)
        if "candidate_pairs" in timeline.columns
        else duplication_nodes.add(speciation_nodes, fill_value=0.0)
    )

    duplication_switches = timeline.get("total_duplication_sdps", pd.Series(0.0, index=timeline.index)).astype(float)
    speciation_switches = timeline.get("total_speciation_sdps", pd.Series(0.0, index=timeline.index)).astype(float)

    duplication_rate = np.divide(
        duplication_switches.to_numpy(dtype=float),
        total_nodes.to_numpy(dtype=float),
        out=np.zeros(len(timeline), dtype=float),
        where=total_nodes.to_numpy(dtype=float) > 0,
    )
    speciation_rate = np.divide(
        speciation_switches.to_numpy(dtype=float),
        total_nodes.to_numpy(dtype=float),
        out=np.zeros(len(timeline), dtype=float),
        where=total_nodes.to_numpy(dtype=float) > 0,
    )

    ax.plot(x, duplication_rate, marker="o", linewidth=2.0, color="#B24A2A", label="Duplications")
    ax.plot(x, speciation_rate, marker="o", linewidth=2.0, color="#2A6FB2", label="Speciations")

    ax2 = ax.twinx()
    ax2.plot(x, total_nodes.to_numpy(dtype=float), marker="o", linewidth=2.0, color="#666666", linestyle="--", label="Evaluated Nodes")

    ax.set_xlabel("Layer (ancient -> recent)")
    ax.set_ylabel("Switches / Evaluated Nodes")
    ax2.set_ylabel("Evaluated Nodes")
    ax.set_xticks(x.tolist())
    ax.set_title("BADASP Switch Event Timeline")

    if "linkage_threshold" in timeline.columns and timeline["linkage_threshold"].notna().any():
        threshold_labels = [f"{float(val):.2f}" if pd.notna(val) else "" for val in timeline["linkage_threshold"]]
        top_axis = ax.twiny()
        top_axis.set_xlim(ax.get_xlim())
        top_axis.set_xticks(x.tolist())
        top_axis.set_xticklabels(threshold_labels, rotation=45, ha="left")
        top_axis.set_xlabel("Linkage Threshold")

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def _plot_master_dendrogram(
    tree_path: Path,
    events_by_level: Dict[str, pd.DataFrame],
    output_svg: Path,
) -> None:
    """Draw one tree-depth dendrogram with switch events from all hierarchy levels overlaid at LCA nodes."""
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)

    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)

    terminals = tree.get_terminals()
    x_pos: Dict[object, float] = {term: float(i) for i, term in enumerate(terminals, start=1)}

    def _assign_x(clade) -> float:
        if clade in x_pos:
            return x_pos[clade]
        child_xs = [_assign_x(child) for child in clade.clades]
        x_pos[clade] = (min(child_xs) + max(child_xs)) / 2.0 if child_xs else 0.0
        return x_pos[clade]

    _assign_x(tree.root)

    fig, ax = plt.subplots(figsize=(12, 10))

    def _draw(node) -> None:
        x = float(x_pos[node])
        y = float(depths[node])
        for child in node.clades:
            child_x = float(x_pos[child])
            child_y = float(depths[child])
            ax.plot([x, child_x], [y, y], color="#B0B0B0", linewidth=0.8, zorder=1)
            ax.plot([child_x, child_x], [y, child_y], color="#B0B0B0", linewidth=0.8, zorder=1)
            _draw(child)

    _draw(tree.root)

    level_colors = {
        "groups": "#1F77B4",
        "families": "#D95F02",
        "subfamilies": "#2CA02C",
        DUPLICATION_LEVEL: "#B24A2A",
    }
    level_labels = {
        "groups": "Groups",
        "families": "Families",
        "subfamilies": "Subfamilies",
        DUPLICATION_LEVEL: "Duplications",
    }

    node_by_name = {str(clade.name): clade for clade in tree.find_clades() if clade.name}
    all_switch_counts: List[float] = []
    grouped_by_level: Dict[str, pd.DataFrame] = {}

    for level, events_df in events_by_level.items():
        if events_df.empty:
            continue
        if "switch_count" in events_df.columns:
            grouped = (
                events_df.groupby("branch_id")
                .agg(switch_count=("switch_count", "sum"))
                .reset_index()
            )
        else:
            grouped = (
                events_df.groupby("branch_id")
                .agg(switch_count=("position", "size"), mean_score=("score", "mean"))
                .reset_index()
            )
        grouped_by_level[level] = grouped
        all_switch_counts.extend(grouped["switch_count"].astype(float).tolist())

    max_switch = max(all_switch_counts) if all_switch_counts else 1.0

    # Draw denser points first, then broader categories on top.
    preferred_order = ["subfamilies", "families", "groups", DUPLICATION_LEVEL]
    level_order = [level for level in preferred_order if level in grouped_by_level]
    level_order.extend([level for level in grouped_by_level if level not in level_order])
    style_map = {
        "subfamilies": {"marker": "o", "alpha": 0.75, "filled": True, "zorder": 3, "base": 28.0, "scale": 180.0},
        "families": {"marker": "o", "alpha": 0.95, "filled": False, "zorder": 4, "base": 40.0, "scale": 220.0},
        "groups": {"marker": "s", "alpha": 1.0, "filled": False, "zorder": 5, "base": 52.0, "scale": 260.0},
        DUPLICATION_LEVEL: {"marker": "o", "alpha": 0.9, "filled": True, "zorder": 4, "base": 34.0, "scale": 220.0},
    }

    for level in level_order:
        grouped = grouped_by_level.get(level)
        if grouped is None or grouped.empty:
            continue

        style = style_map.get(
            level,
            {"marker": "o", "alpha": 0.9, "filled": True, "zorder": 4, "base": 34.0, "scale": 220.0},
        )
        x_vals: List[float] = []
        y_vals: List[float] = []
        sizes: List[float] = []

        for _, row in grouped.iterrows():
            branch_id = str(row["branch_id"])
            if branch_id not in node_by_name:
                continue
            clade = node_by_name[branch_id]
            x_vals.append(float(x_pos[clade]))
            y_vals.append(float(depths[clade]))
            sizes.append(float(style["base"]) + float(style["scale"]) * (float(row["switch_count"]) / float(max_switch)))

        if not x_vals:
            continue

        ax.scatter(
            x_vals,
            y_vals,
            s=sizes,
            c=level_colors.get(level, "#444444") if style["filled"] else "none",
            alpha=float(style["alpha"]),
            linewidths=1.1,
            edgecolors=level_colors.get(level, "#444444"),
            marker=str(style["marker"]),
            label=level_labels.get(level, level),
            zorder=int(style["zorder"]),
        )

    ax.set_xlabel("Taxa / internal nodes")
    ax.set_ylabel("Branch length from root")
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Master Dendrogram with BADASP Switch Events")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(title="Event Source", loc="upper left")
    fig.tight_layout()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def _plot_linkage_master_dendrogram(
    tree_path: Path,
    events_by_level: Dict[str, pd.DataFrame],
    output_svg: Path,
) -> None:
    """Draw one linkage cophenetic-distance master dendrogram with switch events from all hierarchy levels overlaid at LCA nodes."""
    from scipy.cluster.hierarchy import dendrogram
    from src.tree_cluster import tree_to_linkage
    from src.visualization import _compute_dendrogram_node_coords

    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)

    labels, linkage_rows = tree_to_linkage(tree)
    z = np.asarray(linkage_rows, dtype=float)
    n_leaves = len(labels)
    label_to_index = {name: i for i, name in enumerate(labels)}

    dend_dict = dendrogram(z, no_plot=True)
    leaves_order = dend_dict["leaves"]

    x_by_node, y_by_node, descendants = _compute_dendrogram_node_coords(z, leaves_order)

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, row in enumerate(z):
        left = int(row[0])
        right = int(row[1])
        p = n_leaves + i
        ax.plot([x_by_node[left], x_by_node[right]], [y_by_node[p], y_by_node[p]], color="#B0B0B0", linewidth=0.8, zorder=1)
        ax.plot([x_by_node[left], x_by_node[left]], [y_by_node[left], y_by_node[p]], color="#B0B0B0", linewidth=0.8, zorder=1)
        ax.plot([x_by_node[right], x_by_node[right]], [y_by_node[right], y_by_node[p]], color="#B0B0B0", linewidth=0.8, zorder=1)

    level_colors = {
        "groups": "#1F77B4",
        "families": "#D95F02",
        "subfamilies": "#2CA02C",
        DUPLICATION_LEVEL: "#B24A2A",
    }
    level_labels = {
        "groups": "Groups",
        "families": "Families",
        "subfamilies": "Subfamilies",
        DUPLICATION_LEVEL: "Duplications",
    }

    node_by_name = {str(clade.name): clade for clade in tree.find_clades() if clade.name}
    all_switch_counts: List[float] = []
    grouped_by_level: Dict[str, pd.DataFrame] = {}

    for level, events_df in events_by_level.items():
        if events_df.empty:
            continue
        if "switch_count" in events_df.columns:
            grouped = (
                events_df.groupby("branch_id")
                .agg(switch_count=("switch_count", "sum"))
                .reset_index()
            )
        else:
            grouped = (
                events_df.groupby("branch_id")
                .agg(switch_count=("position", "size"), mean_score=("score", "mean"))
                .reset_index()
            )
        grouped_by_level[level] = grouped
        all_switch_counts.extend(grouped["switch_count"].astype(float).tolist())

    max_switch = max(all_switch_counts) if all_switch_counts else 1.0

    gene_to_linkage_node = {}
    for node_name, clade in node_by_name.items():
        terminals = clade.get_terminals()
        leaf_indices = frozenset(label_to_index[t.name] for t in terminals if t.name in label_to_index)
        gene_to_linkage_node[node_name] = leaf_indices

    descendants_to_linkage_id = {desc: nid for nid, desc in descendants.items()}

    preferred_order = ["subfamilies", "families", "groups", DUPLICATION_LEVEL]
    level_order = [level for level in preferred_order if level in grouped_by_level]
    level_order.extend([level for level in grouped_by_level if level not in level_order])
    style_map = {
        "subfamilies": {"marker": "o", "alpha": 0.75, "filled": True, "zorder": 3, "base": 28.0, "scale": 180.0},
        "families": {"marker": "o", "alpha": 0.95, "filled": False, "zorder": 4, "base": 40.0, "scale": 220.0},
        "groups": {"marker": "s", "alpha": 1.0, "filled": False, "zorder": 5, "base": 52.0, "scale": 260.0},
        DUPLICATION_LEVEL: {"marker": "o", "alpha": 0.9, "filled": True, "zorder": 4, "base": 34.0, "scale": 220.0},
    }

    for level in level_order:
        grouped = grouped_by_level.get(level)
        if grouped is None or grouped.empty:
            continue

        style = style_map.get(
            level,
            {"marker": "o", "alpha": 0.9, "filled": True, "zorder": 4, "base": 34.0, "scale": 220.0},
        )
        x_vals: List[float] = []
        y_vals: List[float] = []
        sizes: List[float] = []

        for _, row in grouped.iterrows():
            branch_id = str(row["branch_id"])
            if branch_id not in node_by_name:
                continue
            leaf_indices = gene_to_linkage_node.get(branch_id)
            if not leaf_indices:
                continue
            linkage_id = descendants_to_linkage_id.get(leaf_indices)
            if linkage_id is None:
                continue
            x_vals.append(float(x_by_node[linkage_id]))
            y_vals.append(float(y_by_node[linkage_id]))
            sizes.append(float(style["base"]) + float(style["scale"]) * (float(row["switch_count"]) / float(max_switch)))

        if not x_vals:
            continue

        ax.scatter(
            x_vals,
            y_vals,
            s=sizes,
            c=level_colors.get(level, "#444444") if style["filled"] else "none",
            alpha=float(style["alpha"]),
            linewidths=1.1,
            edgecolors=level_colors.get(level, "#444444"),
            marker=str(style["marker"]),
            label=level_labels.get(level, level),
            zorder=int(style["zorder"]),
        )

    ax.set_xlabel("Taxa / internal nodes")
    ax.set_ylabel("Hierarchical Linkage Distance")
    ax.set_xticks([])
    ax.set_title("Master Linkage Dendrogram with BADASP Switch Events")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(title="Event Source", loc="upper right")
    fig.tight_layout()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def _domain_residue_width(domain_arch: Dict[str, Sequence[int]], domain: str) -> int:
    start, end = [int(x) for x in domain_arch[domain]]
    return max(1, (end - start) + 1)


def _collect_architecture_switch_values(level_scores: pd.DataFrame, domain_arch: Dict[str, Sequence[int]]) -> Dict[str, List[int]]:
    values: Dict[str, List[int]] = {domain: [] for domain in domain_arch}
    if level_scores.empty:
        return values

    score_table = level_scores[["position", "switch_count"]].copy()
    score_table["position"] = score_table["position"].astype(int)
    score_table["switch_count"] = score_table["switch_count"].fillna(0).astype(int)

    for domain, span in domain_arch.items():
        start, end = int(span[0]), int(span[1])
        domain_values = score_table.loc[
            (score_table["position"] >= start) & (score_table["position"] <= end),
            "switch_count",
        ]
        values[domain] = [int(value) for value in domain_values.tolist()]
    return values


def _compute_architecture_enrichment(level_scores: pd.DataFrame, domain_arch: Dict[str, Sequence[int]]) -> pd.DataFrame:
    domain_values = _collect_architecture_switch_values(level_scores, domain_arch)
    rows: List[dict] = []
    for domain, values in domain_values.items():
        count = sum(values)
        width = _domain_residue_width(domain_arch, domain)
        density = float(count) / float(width) if width > 0 else 0.0
        rows.append({
            "domain": domain,
            "switch_count": count,
            "domain_width": width,
            "switch_density": density
        })
    return pd.DataFrame(rows)


def _plot_architecture_boxplot(
    level_scores: pd.DataFrame,
    domain_arch: Dict[str, Sequence[int]],
    output_svg: Path,
    level: str,
) -> None:
    domain_values = _collect_architecture_switch_values(level_scores, domain_arch)

    domains = list(domain_arch.keys())
    data = [domain_values[domain] for domain in domains]
    color = "#4C78A8"

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    box = ax.boxplot(
        data,
        tick_labels=[f"{domain}\n(n={len(values)})" for domain, values in zip(domains, data)],
        vert=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        medianprops={"linewidth": 0.0, "color": "none"},
        meanprops={"color": "#111111", "linewidth": 1.8},
        boxprops={"linewidth": 1.1, "edgecolor": "#444444"},
        whiskerprops={"linewidth": 1.0, "color": "#444444"},
        capprops={"linewidth": 1.0, "color": "#444444"},
    )

    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.set_xlabel("Switch Count")
    ax.set_ylabel("Architectural Domain")
    ax.set_title(f"Architectural Switch Count Distribution ({level.capitalize()})")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.86, bottom=0.20)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def _plot_architecture_distribution(
    counts: Dict[str, int],
    domain_arch: Dict[str, Sequence[int]],
    output_svg: Path,
    level: str,
    normalize: bool = False,
) -> None:
    rows: List[dict] = []
    for domain, count in counts.items():
        width = _domain_residue_width(domain_arch, domain) if domain in domain_arch else 1
        density = float(count) / float(width)
        rows.append({"domain": domain, "switch_count": int(count), "switch_density": density, "domain_width": int(width)})

    df = pd.DataFrame(rows)
    y_col = "switch_density" if normalize else "switch_count"
    y_label = "Switches per residue" if normalize else "Switch Count"
    suffix = " (Normalized)" if normalize else ""

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="domain", y=y_col, color="#4C78A8")
    plt.ylabel(y_label)
    plt.xlabel("Domain")
    plt.title(f"Architectural Switch Distribution ({level.capitalize()}){suffix}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def _top_correlated_pairs(matrix: pd.DataFrame, top_n: int = 5) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    for i, left in enumerate(matrix.index.tolist()):
        for right in matrix.columns.tolist()[i + 1 :]:
            pairs.append((int(left), int(right), float(matrix.loc[left, right])))
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]


def _mean_upper_triangle(matrix: pd.DataFrame) -> float:
    if matrix.empty:
        return float("nan")
    arr = matrix.to_numpy(dtype=float)
    tri = arr[np.triu_indices(arr.shape[0], k=1)]
    if tri.size == 0:
        return float("nan")
    return float(np.nanmean(tri))


def _load_switch_events_for_level(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_path: Path,
    level: str,
) -> pd.DataFrame:
    if level == DUPLICATION_LEVEL:
        return _load_switch_events_from_duplications(
            tree_path=tree_path,
            raw_pairwise_path=raw_pairwise_path,
        )

    singular = LEVEL_MAP[level]
    id_col = f"{singular}_id"

    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)
    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)

    assignments = pd.read_csv(assignments_path)
    cluster_members = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()

    raw = pd.read_csv(raw_pairwise_path)
    threshold = float(np.percentile(raw["score"].astype(float), 95)) if not raw.empty else 0.0
    switched = raw[raw["score"] > threshold].copy()

    rows: List[dict] = []
    for _, row in switched.iterrows():
        try:
            left_str, right_str = str(row["pair"]).split("-")
            left_id = int(left_str)
            right_id = int(right_str)
        except ValueError:
            continue

        if left_id not in cluster_members or right_id not in cluster_members:
            continue

        members = list(cluster_members[left_id]) + list(cluster_members[right_id])
        lca = tree.common_ancestor(members)
        depth = float(depths.get(lca, 0.0))
        rows.append(
            {
                "level": level,
                "pair": str(row["pair"]),
                "position": int(row["position"]),
                "score": float(row["score"]),
                "branch_id": str(lca.name),
                "root_distance": depth,
            }
        )

    return pd.DataFrame(rows)


def _load_switch_events_from_duplications(
    tree_path: Path,
    raw_pairwise_path: Path,
    named_tree_path: Optional[Path] = Path("data/interim/asr_run.treefile"),
) -> pd.DataFrame:
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)
    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)

    raw = pd.read_csv(raw_pairwise_path)
    required = {"pair", "position", "score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in {raw_pairwise_path}: {sorted(missing)}")

    if raw.empty:
        return pd.DataFrame(columns=["level", "pair", "position", "score", "branch_id", "root_distance"])

    threshold = float(np.percentile(raw["score"].astype(float), 95))
    switched = raw[raw["score"] > threshold].copy()

    node_column = None
    for candidate in ("lca_node_name", "lca_node_id", "duplication_node"):
        if candidate in switched.columns:
            node_column = candidate
            break
    if node_column is None:
        raise ValueError(
            f"Missing required duplication LCA column in {raw_pairwise_path}: one of ['lca_node_name', 'lca_node_id', 'duplication_node']"
        )

    asr_to_plot_name = _remap_named_nodes_to_plot_tree(tree, named_tree_path=named_tree_path)
    named_depths: Dict[str, float] = {}
    if named_tree_path is not None and named_tree_path.exists():
        named_tree = Phylo.read(str(named_tree_path), "newick")
        _ensure_node_names(named_tree)
        source_depths = named_tree.depths()
        if not max(source_depths.values()):
            source_depths = named_tree.depths(unit_branch_lengths=True)
        named_depths = {
            str(clade.name): float(source_depths.get(clade, np.nan))
            for clade in named_tree.find_clades()
            if clade.name
        }

    node_by_name = {str(clade.name): clade for clade in tree.find_clades() if clade.name}
    rows: List[dict] = []
    for _, row in switched.iterrows():
        source_branch_id = str(row[node_column])
        branch_id = asr_to_plot_name.get(source_branch_id, source_branch_id)
        clade = node_by_name.get(branch_id)
        if clade is not None:
            root_distance = float(depths.get(clade, np.nan))
        else:
            root_distance = float(named_depths.get(source_branch_id, np.nan))
        event_value = row["Event_Type"] if "Event_Type" in row else row.get("event_type", "Unknown")
        layer_index = int(row["layer_index"]) if "layer_index" in row and pd.notna(row["layer_index"]) else 1
        rows.append(
            {
                "level": DUPLICATION_LEVEL,
                "pair": str(row["pair"]),
                "position": int(row["position"]),
                "score": float(row["score"]),
                "source_branch_id": source_branch_id,
                "branch_id": branch_id,
                "root_distance": root_distance,
                "Event_Type": str(event_value),
                "layer_index": int(layer_index),
            }
        )

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    if events["root_distance"].isna().any():
        depth_fallback = tree.depths(unit_branch_lengths=True)
        events["root_distance"] = events["branch_id"].map(
            lambda x: float(depth_fallback.get(node_by_name.get(str(x)), np.nan))
        )
    return events


def _consensus_sequence(msa_path: Path) -> str:
    records = list(SeqIO.parse(str(msa_path), "fasta"))
    aln_len = len(str(records[0].seq))
    consensus_chars: List[str] = []
    for i in range(aln_len):
        residues = [str(rec.seq)[i] for rec in records if str(rec.seq)[i] != "-"]
        consensus_chars.append(Counter(residues).most_common(1)[0][0] if residues else "X")
    return "".join(consensus_chars)


def _safe_window(sequence: str, center_1based: int, width: int = 15) -> str:
    idx = max(0, center_1based - 1)
    half = width // 2
    start = max(0, idx - half)
    end = min(len(sequence), idx + half + 1)
    window = sequence[start:end]
    if len(window) < 7:
        return sequence[max(0, idx - 3): min(len(sequence), idx + 4)]
    return window





@lru_cache(maxsize=2048)
def _protparam(sequence: str) -> dict:
    """Compute GRAVY and isoelectric point using local ProteinAnalysis."""
    cleaned = "".join([aa for aa in sequence.upper() if aa.isalpha()])
    if not cleaned:
        return {}
    analysis = ProteinAnalysis(cleaned)
    return {
        "gravy": float(analysis.gravy()),
        "isoelectric_point": float(analysis.isoelectric_point()),
    }


def _charge_class(residue: str) -> str:
    positive = {"K", "R", "H"}
    negative = {"D", "E"}
    if residue in positive:
        return "positive"
    if residue in negative:
        return "negative"
    return "neutral"


def _hydrophobic_class(residue: str) -> str:
    hydrophobic = {"A", "V", "I", "L", "M", "F", "W", "Y", "P"}
    return "hydrophobic" if residue in hydrophobic else "polar"


def _volume(residue: str) -> float:
    volumes = {
        "A": 88.6,
        "R": 173.4,
        "N": 114.1,
        "D": 111.1,
        "C": 108.5,
        "Q": 143.8,
        "E": 138.4,
        "G": 60.1,
        "H": 153.2,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "M": 162.9,
        "F": 189.9,
        "P": 112.7,
        "S": 89.0,
        "T": 116.1,
        "W": 227.8,
        "Y": 193.6,
        "V": 140.0,
    }
    return float(volumes.get(residue, np.nan))


def _compute_level_physicochemical_shifts(
    level: str,
    level_scores: pd.DataFrame,
    assignments: pd.DataFrame,
    lca_to_asr: Dict[str, str],
    ancestral_records: Dict[str, str],
    alignment_records: Dict[str, str],
    consensus: str,
    tu_tools: List[str],
) -> pd.DataFrame:
    if level not in LEVEL_MAP:
        return pd.DataFrame()

    singular = LEVEL_MAP[level]
    id_col = f"{singular}_id"
    lca_col = f"{singular}_lca_node"

    grouped = assignments.groupby(id_col)
    top_positions = (
        level_scores[level_scores["switch_count"] > 0]
        .sort_values(["switch_count", "max_score"], ascending=[False, False])["position"]
        .astype(int)
        .head(10)
        .tolist()
    )

    rows: List[dict] = []
    for pos in top_positions:
        transitions = Counter()
        for _, cluster_df in grouped:
            lca_node = str(cluster_df[lca_col].iloc[0])
            lca_node = lca_to_asr.get(lca_node, lca_node)
            if lca_node not in ancestral_records:
                continue
            anc_seq = ancestral_records[lca_node]
            if pos - 1 >= len(anc_seq):
                continue
            anc_aa = anc_seq[pos - 1]
            recent_residues: List[str] = []
            for seq_id in cluster_df["sequence_id"].tolist():
                seq = alignment_records.get(seq_id)
                if seq is None or pos - 1 >= len(seq):
                    continue
                aa = seq[pos - 1]
                if aa != "-":
                    recent_residues.append(aa)
            if not recent_residues:
                continue
            recent_aa = Counter(recent_residues).most_common(1)[0][0]
            if anc_aa == "-" or anc_aa == recent_aa:
                continue
            transitions[(anc_aa, recent_aa)] += 1

        if not transitions:
            continue

        (anc_aa, recent_aa), n_switches = transitions.most_common(1)[0]
        window = _safe_window(consensus, pos, width=15)
        center = min(len(window) - 1, len(window) // 2)
        anc_window = window[:center] + anc_aa + window[center + 1 :]
        recent_window = window[:center] + recent_aa + window[center + 1 :]
        anc_props = _protparam(anc_window)
        recent_props = _protparam(recent_window)

        rows.append(
            {
                "level": level,
                "position": pos,
                "ancestral_aa": anc_aa,
                "recent_aa": recent_aa,
                "major_transition_count": int(n_switches),
                "charge_change": f"{_charge_class(anc_aa)}->{_charge_class(recent_aa)}",
                "hydrophobicity_change": f"{_hydrophobic_class(anc_aa)}->{_hydrophobic_class(recent_aa)}",
                "volume_change": float(_volume(recent_aa) - _volume(anc_aa)),
                "tu_tool_used": "ProtParam_calculate",
                "tu_related_tools": ";".join(tu_tools),
                "tu_gravy_ancestral": anc_props.get("gravy", np.nan),
                "tu_gravy_recent": recent_props.get("gravy", np.nan),
                "tu_delta_gravy": float(recent_props.get("gravy", 0.0) - anc_props.get("gravy", 0.0)),
                "tu_pI_ancestral": anc_props.get("isoelectric_point", np.nan),
                "tu_pI_recent": recent_props.get("isoelectric_point", np.nan),
                "tu_delta_pI": float(recent_props.get("isoelectric_point", 0.0) - anc_props.get("isoelectric_point", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def _collect_taxonomic_distribution(
    level: str,
    assignments: pd.DataFrame,
    descriptions: Dict[str, str],
    top_positions: Sequence[int],
    alignment_records: Dict[str, str],
) -> pd.DataFrame:
    if level not in LEVEL_MAP:
        return pd.DataFrame()

    rows: List[dict] = []
    singular = LEVEL_MAP[level]
    id_col = f"{singular}_id"

    seq_taxon_rows: List[Tuple[str, str]] = []
    for seq_id in assignments["sequence_id"].tolist():
        desc = descriptions.get(seq_id, seq_id)
        seq_taxon_rows.append((seq_id, extract_taxon_label(desc)))
    seq_taxon = pd.DataFrame(seq_taxon_rows, columns=["sequence_id", "taxon"]) if seq_taxon_rows else pd.DataFrame(columns=["sequence_id", "taxon"])
    merged = assignments[["sequence_id", id_col]].merge(seq_taxon, on="sequence_id", how="left")

    taxon_totals = merged.groupby("taxon").size().sort_values(ascending=False)
    major_taxa = set(taxon_totals.head(25).index.tolist())

    for pos in [int(p) for p in top_positions]:
        for taxon in major_taxa:
            seq_ids = merged[merged["taxon"] == taxon]["sequence_id"].tolist()
            if not seq_ids:
                continue
            residues: List[str] = []
            for seq_id in seq_ids:
                seq = alignment_records.get(seq_id)
                if seq is None or pos - 1 >= len(seq):
                    continue
                aa = seq[pos - 1]
                if aa != "-":
                    residues.append(aa)
            total = len(seq_ids)
            present = len(residues)
            dominant = Counter(residues).most_common(1)[0][0] if residues else "-"
            rows.append(
                {
                    "level": level,
                    "position": pos,
                    "taxon": taxon,
                    "present_count": present,
                    "total_sequences": total,
                    "presence_fraction": float(present / total) if total else 0.0,
                    "dominant_residue": dominant,
                }
            )

    return pd.DataFrame(rows)


def run_phase7_analyses(
    tree_path: Path,
    raw_pairwise_duplications: Path,
    duplication_scores_path: Path,
    msa_path: Path,
    ancestral_fasta_path: Path,
    asr_mapping_path: Path,
    domain_architecture_path: Path,
    pdb_path: Path,
    output_dir: Path,
    assignments_path: Optional[Path] = None,
    reference_asr_tree_path: Optional[Path] = Path("data/interim/asr_run.treefile"),
    pdb_id: str = "2cg4",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_levels = [DUPLICATION_LEVEL]
    assignments = pd.read_csv(assignments_path) if assignments_path and assignments_path.exists() else pd.DataFrame()
    alignment_parsed = list(SeqIO.parse(str(msa_path), "fasta"))
    alignment_records = {rec.id: str(rec.seq) for rec in alignment_parsed}
    descriptions = {rec.id: rec.description for rec in alignment_parsed}
    ancestral_records = {rec.id: str(rec.seq) for rec in SeqIO.parse(str(ancestral_fasta_path), "fasta")}
    if asr_mapping_path.exists():
        asr_map_df = pd.read_csv(asr_mapping_path)
        lca_to_asr = {
            str(row["lca_node"]): str(row["lca_node_asr"])
            for _, row in asr_map_df.iterrows()
            if pd.notna(row.get("lca_node")) and pd.notna(row.get("lca_node_asr"))
        }
    else:
        lca_to_asr = {}
    with domain_architecture_path.open("r") as handle:
        domain_architecture = json.load(handle)

    consensus = _consensus_sequence(msa_path)
    mapper = PDBMapper(pdb_id=pdb_id, pdb_file=str(pdb_path))
    mapping = mapper.map_alignment_to_structure(msa_path)

    events_by_level: Dict[str, pd.DataFrame] = {
        DUPLICATION_LEVEL: _load_switch_events_from_duplications(
            tree_path=tree_path,
            raw_pairwise_path=raw_pairwise_duplications,
            named_tree_path=reference_asr_tree_path,
        )
    }
    score_by_level: Dict[str, pd.DataFrame] = {
        DUPLICATION_LEVEL: pd.read_csv(duplication_scores_path),
    }

    timeline_svg = output_dir / "switch_timeline.svg"
    all_events = pd.concat([events_by_level[level] for level in analysis_levels], ignore_index=True)
    _plot_switch_timeline(all_events, timeline_svg)
    master_dendrogram_svg = output_dir / "master_dendrogram_switches.svg"
    _plot_master_dendrogram(tree_path=tree_path, events_by_level=events_by_level, output_svg=master_dendrogram_svg)
    master_linkage_dendrogram_svg = output_dir / "master_linkage_dendrogram_switches.svg"
    _plot_linkage_master_dendrogram(tree_path=tree_path, events_by_level=events_by_level, output_svg=master_linkage_dendrogram_svg)

    community_rows: List[pd.DataFrame] = []
    tax_rows: List[pd.DataFrame] = []
    all_shifts: List[pd.DataFrame] = []
    outputs: Dict[str, Path] = {
        "switch_timeline_svg": timeline_svg,
        "master_dendrogram_switches_svg": master_dendrogram_svg,
        "master_linkage_dendrogram_switches_svg": master_linkage_dendrogram_svg,
    }

    for level in analysis_levels:
        level_scores = score_by_level[level]
        level_events = events_by_level[level]

        top_positions = (
            level_scores[level_scores["switch_count"] > 0]
            .sort_values(["switch_count", "max_score"], ascending=[False, False])["position"]
            .astype(int)
            .head(15)
            .tolist()
        )

        residue_numbers = [
            resnum
            for pos in top_positions
            if pos in mapping
            for resnum in [_normalize_residue_number(mapping[pos])]
            if resnum is not None
        ]
        distance_matrix = calculate_ca_distance_matrix(pdb_path, residue_numbers)
        distance_csv = output_dir / f"distance_matrix_{level}.csv"
        distance_matrix.to_csv(distance_csv, index=True)
        outputs[f"distance_matrix_csv_{level}"] = distance_csv

        heatmap_svg = output_dir / f"sdp_distance_heatmap_{level}.svg"
        _plot_clustered_heatmap(
            matrix=distance_matrix,
            output_svg=heatmap_svg,
            title=f"Top SDP C-alpha Distance Heatmap ({level.capitalize()})",
            cmap="mako_r",
            cbar_label="Distance (Å)",
            is_distance_matrix=True,
        )
        outputs[f"sdp_distance_heatmap_svg_{level}"] = heatmap_svg

        top_events = level_events[level_events["position"].isin(top_positions[:40])]
        if top_events.empty:
            coevo_matrix = pd.DataFrame(index=top_positions[:1], columns=top_positions[:1], data=1.0) if top_positions else pd.DataFrame()
        else:
            coevo_matrix = compute_coevolution_matrix(top_events[["branch_id", "position"]])

        coevo_csv = output_dir / f"coevolution_matrix_{level}.csv"
        coevo_matrix.to_csv(coevo_csv, index=True)
        outputs[f"coevolution_matrix_csv_{level}"] = coevo_csv

        coevo_svg = output_dir / f"coevolution_matrix_{level}.svg"
        _plot_clustered_heatmap(
            matrix=coevo_matrix,
            output_svg=coevo_svg,
            title=f"Co-evolution Matrix ({level.capitalize()})",
            cmap="viridis",
        )
        outputs[f"coevolution_matrix_svg_{level}"] = coevo_svg

        top_pairs = _top_correlated_pairs(coevo_matrix, top_n=5)
        mean_top_distance = _mean_upper_triangle(distance_matrix)
        print(f"Phase 7b terminal statistics ({level})")
        print(f"- Mean distance between top SDPs: {mean_top_distance:.3f} A")
        print("- Top 5 most highly correlated residue pairs:")
        for left, right, value in top_pairs:
            print(f"  ({left}, {right}) -> {value:.4f}")

        shifts_df = _compute_level_physicochemical_shifts(
            level=level,
            level_scores=level_scores,
            assignments=assignments,
            lca_to_asr=lca_to_asr,
            ancestral_records=ancestral_records,
            alignment_records=alignment_records,
            consensus=consensus,
            tu_tools=[],
        )
        shifts_csv = output_dir / f"physicochemical_shifts_{level}.csv"
        shifts_df.to_csv(shifts_csv, index=False)
        outputs[f"physicochemical_shifts_csv_{level}"] = shifts_csv
        all_shifts.append(shifts_df)

        functional_df = rank_top_functional_sdps(
            subfamily_scores_df=level_scores,
            coevolution_matrix_df=coevo_matrix,
            shifts_df=shifts_df,
            top_n=25,
        )
        functional_csv = output_dir / f"top_functional_sdps_{level}.csv"
        functional_df.to_csv(functional_csv, index=False)
        outputs[f"top_functional_sdps_csv_{level}"] = functional_csv

        domain_counts = count_switches_per_domain(level_events, domain_architecture)
        arch_svg = output_dir / f"architectural_distribution_{level}.svg"
        _plot_architecture_distribution(domain_counts, domain_architecture, arch_svg, level, normalize=False)
        outputs[f"architectural_distribution_svg_{level}"] = arch_svg
        arch_norm_svg = output_dir / f"architectural_distribution_{level}_normalized.svg"
        _plot_architecture_distribution(domain_counts, domain_architecture, arch_norm_svg, level, normalize=True)
        outputs[f"architectural_distribution_svg_normalized_{level}"] = arch_norm_svg

        arch_box_svg = output_dir / f"architectural_boxplot_{level}.svg"
        _plot_architecture_boxplot(level_scores, domain_architecture, arch_box_svg, level)
        outputs[f"architectural_boxplot_svg_{level}"] = arch_box_svg

        enrichment_df = _compute_architecture_enrichment(level_scores, domain_architecture)
        enrichment_csv = output_dir / f"architectural_enrichment_{level}.csv"
        enrichment_df.to_csv(enrichment_csv, index=False)
        outputs[f"architectural_enrichment_csv_{level}"] = enrichment_csv

        communities = assign_coevolution_communities(coevo_matrix, distance_cut=0.6)
        communities["level"] = level
        communities["community_size"] = communities.groupby("community_id")["position"].transform("count")
        community_rows.append(communities)

        if not assignments.empty:
            tax_level = _collect_taxonomic_distribution(
                level=level,
                assignments=assignments,
                descriptions=descriptions,
                top_positions=functional_df["position"].astype(int).head(15).tolist() if not functional_df.empty else [],
                alignment_records=alignment_records,
            )
            tax_rows.append(tax_level)

    communities_df = pd.concat(community_rows, ignore_index=True) if community_rows else pd.DataFrame(columns=["position", "community_id", "level", "community_size"])
    communities_csv = output_dir / "coevolution_communities.csv"
    communities_df.to_csv(communities_csv, index=False)
    outputs["coevolution_communities_csv"] = communities_csv

    tax_df = pd.concat(tax_rows, ignore_index=True) if tax_rows else pd.DataFrame()
    tax_csv = output_dir / "taxonomic_sdp_distribution.csv"
    tax_df.to_csv(tax_csv, index=False)
    outputs["taxonomic_sdp_distribution_csv"] = tax_csv

    shifts_all_df = pd.concat(all_shifts, ignore_index=True) if all_shifts else pd.DataFrame()
    shifts_all_csv = output_dir / "physicochemical_shifts.csv"
    shifts_all_df.to_csv(shifts_all_csv, index=False)
    outputs["physicochemical_shifts_csv"] = shifts_all_csv

    families_subfamilies = shifts_all_df.copy() if not shifts_all_df.empty else pd.DataFrame()
    map_input_csv = output_dir / "physicochemical_shifts_for_mapping.csv"
    families_subfamilies.to_csv(map_input_csv, index=False)

    if not families_subfamilies.empty:
        physio_cxc = Path("results/structural_mapping/highlight_physicochemistry.cxc")
        mapper.generate_physicochemical_chimerax_script(
            alignment_path=msa_path,
            physicochemical_csv=map_input_csv,
            output_cxc=physio_cxc,
        )
        outputs["physicochemical_mapping_cxc"] = physio_cxc

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7 evolutionary and physicochemical analysis")
    parser.add_argument("--tree", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--raw-pairwise-duplications", default="results/badasp_scoring/raw_pairwise_duplications.csv")
    parser.add_argument("--duplication-scores", default="results/badasp_scoring/badasp_scores_duplications.csv")
    parser.add_argument("--msa", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--ancestral", default="data/interim/ancestral_sequences.fasta")
    parser.add_argument("--asr-map", default="results/topological_clustering/tree_clusters_asr_mapped.csv")
    parser.add_argument("--reference-asr-tree", default="data/interim/asr_run.treefile")
    parser.add_argument("--domain-architecture", default="data/domain_architecture.json")
    parser.add_argument("--pdb", default="data/raw/2cg4.pdb")
    parser.add_argument("--pdb-id", default="2cg4")
    parser.add_argument("--output-dir", default="results/evolutionary_analysis")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    scores_root = Path(args.duplication_scores).parent
    layer_score_files = sorted(scores_root.glob("layer_*/badasp_scores_duplications.csv"), key=_layer_sort_key)
    if layer_score_files:
        summary_csv = Path(args.output_dir) / "global_layer_summary.csv"
        summary_df = build_global_layer_summary(scores_root=scores_root, output_csv=summary_csv)
        timeline_svg = Path(args.output_dir) / "switch_timeline.svg"
        plot_layerwise_switch_timeline(summary_df=summary_df, output_svg=timeline_svg)
        print(f"Saved global layer summary: {summary_csv}")
        print(f"Saved layer timeline plot: {timeline_svg}")
        for score_file in layer_score_files:
            layer_dir = score_file.parent
            pairwise_file = layer_dir / "raw_pairwise_duplications.csv"
            if not pairwise_file.exists():
                continue
            layer_out = Path(args.output_dir) / layer_dir.name
            outputs = run_phase7_analyses(
                tree_path=Path(args.tree),
                raw_pairwise_duplications=pairwise_file,
                duplication_scores_path=score_file,
                msa_path=Path(args.msa),
                ancestral_fasta_path=Path(args.ancestral),
                asr_mapping_path=Path(args.asr_map),
                domain_architecture_path=Path(args.domain_architecture),
                pdb_path=Path(args.pdb),
                output_dir=layer_out,
                assignments_path=Path(args.assignments) if args.assignments else None,
                reference_asr_tree_path=Path(args.reference_asr_tree) if args.reference_asr_tree else None,
                pdb_id=str(args.pdb_id),
            )
            for label, path in outputs.items():
                print(f"Saved {label}: {path}")

        # Run time-calibrated chronological timeline analysis
        try:
            from src.chronological_timeline import run_chronology_pipeline, load_calibrations
            from src.visualization import plot_chronological_dendrogram
            from Bio import Phylo
            from ete3 import NCBITaxa
            import json
            import numpy as np
            import pandas as pd

            chrono_svg = Path(args.output_dir) / "chronological_switch_timeline.svg"
            asr_tree_path = Path(args.reference_asr_tree) if args.reference_asr_tree else Path("data/interim/asr_run.treefile")
            
            node_ages = run_chronology_pipeline(
                tree_path=asr_tree_path,
                fasta_path=Path(args.msa).parent / "IPR019888_length_filtered.fasta",
                calibration_config=Path("data/time_calibration.json"),
                duplications_path=scores_root / "raw_pairwise_duplications.csv",
                speciations_path=scores_root / "raw_pairwise_speciations.csv",
                output_svg=chrono_svg
            )
            print(f"Saved time-calibrated chronological timeline: {chrono_svg}")

            # Plot chronological dendrogram with collapsed clades
            print("Generating publication-ready time-calibrated chronological dendrogram...")
            
            # Load and resolve calibration anchors
            calibrations = load_calibrations(Path("data/time_calibration.json"))
            ncbi = NCBITaxa()
            anchor_points = {}
            for taxid, age in calibrations.items():
                try:
                    name = ncbi.get_taxid_translator([taxid]).get(taxid, str(taxid))
                    anchor_points[f"{name} ({int(age)} Mya)"] = age
                except Exception:
                    anchor_points[f"TaxID {taxid} ({int(age)} Mya)"] = age

            # Aggregate duplications and speciations switch counts (at 95th percentile)
            node_switches = {}
            def aggregate_switches(path: Path, node_col_candidates: tuple):
                if not path.exists():
                    return
                df = pd.read_csv(path)
                if "score" not in df.columns:
                    return
                scores = pd.to_numeric(df["score"], errors="coerce").dropna()
                if scores.empty:
                    return
                threshold = np.percentile(scores, 95)
                switched_df = df[df["score"] >= threshold]
                
                node_col = None
                for c in node_col_candidates:
                    if c in switched_df.columns:
                        node_col = c
                        break
                if node_col:
                    for node in switched_df[node_col].astype(str):
                        if node and node != "nan":
                            node_switches[node] = node_switches.get(node, 0) + 1

            aggregate_switches(scores_root / "raw_pairwise_duplications.csv", ("duplication_node", "lca_node_name"))
            aggregate_switches(scores_root / "raw_pairwise_speciations.csv", ("duplication_node", "lca_node_name"))

            # Read full ASR tree
            tree = Phylo.read(str(asr_tree_path), "newick")
            dendrogram_path = Path(args.output_dir) / "chronological_dendrogram_switches.svg"
            
            plot_chronological_dendrogram(
                tree=tree,
                node_ages=node_ages,
                node_switches=node_switches,
                anchor_points=anchor_points,
                output_path=dendrogram_path
            )
            print(f"Saved time-calibrated chronological dendrogram: {dendrogram_path}")

        except Exception as e:
            print(f"Warning: Failed to run chronological timeline/dendrogram pipeline: {e}")
            import traceback
            traceback.print_exc()
        return

    outputs = run_phase7_analyses(
        tree_path=Path(args.tree),
        raw_pairwise_duplications=Path(args.raw_pairwise_duplications),
        duplication_scores_path=Path(args.duplication_scores),
        msa_path=Path(args.msa),
        ancestral_fasta_path=Path(args.ancestral),
        asr_mapping_path=Path(args.asr_map),
        domain_architecture_path=Path(args.domain_architecture),
        pdb_path=Path(args.pdb),
        output_dir=Path(args.output_dir),
        assignments_path=Path(args.assignments) if args.assignments else None,
        reference_asr_tree_path=Path(args.reference_asr_tree) if args.reference_asr_tree else None,
        pdb_id=str(args.pdb_id),
    )
    for label, path in outputs.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
