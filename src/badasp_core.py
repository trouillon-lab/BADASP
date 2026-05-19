"""Phase 5 duplication-directed BADASP scoring for high-confidence duplication nodes."""

from __future__ import annotations

import argparse
import csv
import logging
import warnings
from collections import defaultdict
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade, Tree
from tqdm import tqdm

logger = logging.getLogger(__name__)

SKIPPED_AUDIT_LOG = Path("results/badasp_scoring/skipped_audit.log")

TRACK_EVENT_TYPES = {
    "duplications": "Duplication",
    "speciations": "Speciation",
    "combined": None,
}

TRACK_OUTPUT_SUFFIXES = {
    "duplications": "duplications",
    "speciations": "speciations",
    "combined": "combined",
}


@lru_cache(maxsize=1)
def _get_blosum62_matrix() -> Dict[Tuple[str, str], int]:
    return {
        ('A', 'A'): 4,   ('A', 'R'): -1,  ('A', 'N'): -2,  ('A', 'D'): -2,  ('A', 'C'): 0,
        ('A', 'Q'): -1,  ('A', 'E'): -1,  ('A', 'G'): 0,   ('A', 'H'): -2,  ('A', 'I'): -1,
        ('A', 'L'): -1,  ('A', 'K'): -1,  ('A', 'M'): -1,  ('A', 'F'): -2,  ('A', 'P'): -1,
        ('A', 'S'): 1,   ('A', 'T'): 0,   ('A', 'W'): -3,  ('A', 'Y'): -2,  ('A', 'V'): 0,
        ('R', 'A'): -1,  ('R', 'R'): 5,   ('R', 'N'): 0,   ('R', 'D'): -2,  ('R', 'C'): -3,
        ('R', 'Q'): 1,   ('R', 'E'): 0,   ('R', 'G'): -2,  ('R', 'H'): 0,   ('R', 'I'): -3,
        ('R', 'L'): -2,  ('R', 'K'): 2,   ('R', 'M'): -1,  ('R', 'F'): -3,  ('R', 'P'): -2,
        ('R', 'S'): -1,  ('R', 'T'): -1,  ('R', 'W'): -3,  ('R', 'Y'): -2,  ('R', 'V'): -3,
        ('N', 'A'): -2,  ('N', 'R'): 0,   ('N', 'N'): 6,   ('N', 'D'): 1,   ('N', 'C'): -3,
        ('N', 'Q'): 0,   ('N', 'E'): 0,   ('N', 'G'): 0,   ('N', 'H'): 1,   ('N', 'I'): -3,
        ('N', 'L'): -4,  ('N', 'K'): 0,   ('N', 'M'): -2,  ('N', 'F'): -3,  ('N', 'P'): -2,
        ('N', 'S'): 1,   ('N', 'T'): 0,   ('N', 'W'): -4,  ('N', 'Y'): -2,  ('N', 'V'): -3,
        ('D', 'A'): -2,  ('D', 'R'): -2,  ('D', 'N'): 1,   ('D', 'D'): 6,   ('D', 'C'): -3,
        ('D', 'Q'): 0,   ('D', 'E'): 2,   ('D', 'G'): -1,  ('D', 'H'): -1,  ('D', 'I'): -3,
        ('D', 'L'): -4,  ('D', 'K'): -1,  ('D', 'M'): -3,  ('D', 'F'): -3,  ('D', 'P'): -1,
        ('D', 'S'): 0,   ('D', 'T'): -1,  ('D', 'W'): -4,  ('D', 'Y'): -3,  ('D', 'V'): -3,
        ('C', 'A'): 0,   ('C', 'R'): -3,  ('C', 'N'): -3,  ('C', 'D'): -3,  ('C', 'C'): 9,
        ('C', 'Q'): -3,  ('C', 'E'): -4,  ('C', 'G'): -3,  ('C', 'H'): -3,  ('C', 'I'): -1,
        ('C', 'L'): -1,  ('C', 'K'): -3,  ('C', 'M'): -1,  ('C', 'F'): -2,  ('C', 'P'): -3,
        ('C', 'S'): -1,  ('C', 'T'): -1,  ('C', 'W'): -2,  ('C', 'Y'): -2,  ('C', 'V'): -1,
        ('Q', 'A'): -1,  ('Q', 'R'): 1,   ('Q', 'N'): 0,   ('Q', 'D'): 0,   ('Q', 'C'): -3,
        ('Q', 'Q'): 5,   ('Q', 'E'): 2,   ('Q', 'G'): -2,  ('Q', 'H'): 0,   ('Q', 'I'): -3,
        ('Q', 'L'): -2,  ('Q', 'K'): 1,   ('Q', 'M'): 0,   ('Q', 'F'): -3,  ('Q', 'P'): -1,
        ('Q', 'S'): 0,   ('Q', 'T'): -1,  ('Q', 'W'): -2,  ('Q', 'Y'): -1,  ('Q', 'V'): -2,
        ('E', 'A'): -1,  ('E', 'R'): 0,   ('E', 'N'): 0,   ('E', 'D'): 2,   ('E', 'C'): -4,
        ('E', 'Q'): 2,   ('E', 'E'): 5,   ('E', 'G'): -2,  ('E', 'H'): 0,   ('E', 'I'): -3,
        ('E', 'L'): -3,  ('E', 'K'): 1,   ('E', 'M'): -2,  ('E', 'F'): -3,  ('E', 'P'): -1,
        ('E', 'S'): 0,   ('E', 'T'): -1,  ('E', 'W'): -3,  ('E', 'Y'): -2,  ('E', 'V'): -2,
        ('G', 'A'): 0,   ('G', 'R'): -2,  ('G', 'N'): 0,   ('G', 'D'): -1,  ('G', 'C'): -3,
        ('G', 'Q'): -2,  ('G', 'E'): -2,  ('G', 'G'): 6,   ('G', 'H'): -2,  ('G', 'I'): -4,
        ('G', 'L'): -4,  ('G', 'K'): -2,  ('G', 'M'): -3,  ('G', 'F'): -3,  ('G', 'P'): -2,
        ('G', 'S'): 0,   ('G', 'T'): -2,  ('G', 'W'): -2,  ('G', 'Y'): -3,  ('G', 'V'): -3,
        ('H', 'A'): -2,  ('H', 'R'): 0,   ('H', 'N'): 1,   ('H', 'D'): -1,  ('H', 'C'): -3,
        ('H', 'Q'): 0,   ('H', 'E'): 0,   ('H', 'G'): -2,  ('H', 'H'): 8,   ('H', 'I'): -3,
        ('H', 'L'): -3,  ('H', 'K'): -1,  ('H', 'M'): -2,  ('H', 'F'): -1,  ('H', 'P'): -2,
        ('H', 'S'): -1,  ('H', 'T'): -2,  ('H', 'W'): -2,  ('H', 'Y'): 2,   ('H', 'V'): -3,
        ('I', 'A'): -1,  ('I', 'R'): -3,  ('I', 'N'): -3,  ('I', 'D'): -3,  ('I', 'C'): -1,
        ('I', 'Q'): -3,  ('I', 'E'): -3,  ('I', 'G'): -4,  ('I', 'H'): -3,  ('I', 'I'): 4,
        ('I', 'L'): 2,   ('I', 'K'): -3,  ('I', 'M'): 1,   ('I', 'F'): 0,   ('I', 'P'): -3,
        ('I', 'S'): -2,  ('I', 'T'): -1,  ('I', 'W'): -3,  ('I', 'Y'): -1,  ('I', 'V'): 3,
        ('L', 'A'): -1,  ('L', 'R'): -2,  ('L', 'N'): -4,  ('L', 'D'): -4,  ('L', 'C'): -1,
        ('L', 'Q'): -2,  ('L', 'E'): -3,  ('L', 'G'): -4,  ('L', 'H'): -3,  ('L', 'I'): 2,
        ('L', 'L'): 4,   ('L', 'K'): -2,  ('L', 'M'): 2,   ('L', 'F'): 0,   ('L', 'P'): -3,
        ('L', 'S'): -2,  ('L', 'T'): -1,  ('L', 'W'): -2,  ('L', 'Y'): -1,  ('L', 'V'): 1,
        ('K', 'A'): -1,  ('K', 'R'): 2,   ('K', 'N'): 0,   ('K', 'D'): -1,  ('K', 'C'): -3,
        ('K', 'Q'): 1,   ('K', 'E'): 1,   ('K', 'G'): -2,  ('K', 'H'): -1,  ('K', 'I'): -3,
        ('K', 'L'): -2,  ('K', 'K'): 5,   ('K', 'M'): -1,  ('K', 'F'): -3,  ('K', 'P'): -1,
        ('K', 'S'): 0,   ('K', 'T'): -1,  ('K', 'W'): -3,  ('K', 'Y'): -2,  ('K', 'V'): -2,
        ('M', 'A'): -1,  ('M', 'R'): -1,  ('M', 'N'): -2,  ('M', 'D'): -3,  ('M', 'C'): -1,
        ('M', 'Q'): 0,   ('M', 'E'): -2,  ('M', 'G'): -3,  ('M', 'H'): -2,  ('M', 'I'): 1,
        ('M', 'L'): 2,   ('M', 'K'): -1,  ('M', 'M'): 5,   ('M', 'F'): 0,   ('M', 'P'): -2,
        ('M', 'S'): -1,  ('M', 'T'): -1,  ('M', 'W'): -1,  ('M', 'Y'): -1,  ('M', 'V'): 1,
        ('F', 'A'): -2,  ('F', 'R'): -3,  ('F', 'N'): -3,  ('F', 'D'): -3,  ('F', 'C'): -2,
        ('F', 'Q'): -3,  ('F', 'E'): -3,  ('F', 'G'): -3,  ('F', 'H'): -1,  ('F', 'I'): 0,
        ('F', 'L'): 0,   ('F', 'K'): -3,  ('F', 'M'): 0,   ('F', 'F'): 6,   ('F', 'P'): -4,
        ('F', 'S'): -2,  ('F', 'T'): -2,  ('F', 'W'): 1,   ('F', 'Y'): 3,   ('F', 'V'): -1,
        ('P', 'A'): -1,  ('P', 'R'): -2,  ('P', 'N'): -2,  ('P', 'D'): -1,  ('P', 'C'): -3,
        ('P', 'Q'): -1,  ('P', 'E'): -1,  ('P', 'G'): -2,  ('P', 'H'): -2,  ('P', 'I'): -3,
        ('P', 'L'): -3,  ('P', 'K'): -1,  ('P', 'M'): -2,  ('P', 'F'): -4,  ('P', 'P'): 7,
        ('P', 'S'): -1,  ('P', 'T'): -1,  ('P', 'W'): -4,  ('P', 'Y'): -3,  ('P', 'V'): -2,
        ('S', 'A'): 1,   ('S', 'R'): -1,  ('S', 'N'): 1,   ('S', 'D'): 0,   ('S', 'C'): -1,
        ('S', 'Q'): 0,   ('S', 'E'): 0,   ('S', 'G'): 0,   ('S', 'H'): -1,  ('S', 'I'): -2,
        ('S', 'L'): -2,  ('S', 'K'): 0,   ('S', 'M'): -1,  ('S', 'F'): -2,  ('S', 'P'): -1,
        ('S', 'S'): 4,   ('S', 'T'): 1,   ('S', 'W'): -3,  ('S', 'Y'): -2,  ('S', 'V'): -2,
        ('T', 'A'): 0,   ('T', 'R'): -1,  ('T', 'N'): 0,   ('T', 'D'): -1,  ('T', 'C'): -1,
        ('T', 'Q'): -1,  ('T', 'E'): -1,  ('T', 'G'): -2,  ('T', 'H'): -2,  ('T', 'I'): -1,
        ('T', 'L'): -1,  ('T', 'K'): -1,  ('T', 'M'): -1,  ('T', 'F'): -2,  ('T', 'P'): -1,
        ('T', 'S'): 1,   ('T', 'T'): 5,   ('T', 'W'): -2,  ('T', 'Y'): -2,  ('T', 'V'): 0,
        ('W', 'A'): -3,  ('W', 'R'): -3,  ('W', 'N'): -4,  ('W', 'D'): -4,  ('W', 'C'): -2,
        ('W', 'Q'): -2,  ('W', 'E'): -3,  ('W', 'G'): -2,  ('W', 'H'): -2,  ('W', 'I'): -3,
        ('W', 'L'): -2,  ('W', 'K'): -3,  ('W', 'M'): -1,  ('W', 'F'): 1,   ('W', 'P'): -4,
        ('W', 'S'): -3,  ('W', 'T'): -2,  ('W', 'W'): 11,  ('W', 'Y'): 2,   ('W', 'V'): -3,
        ('Y', 'A'): -2,  ('Y', 'R'): -2,  ('Y', 'N'): -2,  ('Y', 'D'): -3,  ('Y', 'C'): -2,
        ('Y', 'Q'): -1,  ('Y', 'E'): -2,  ('Y', 'G'): -3,  ('Y', 'H'): 2,   ('Y', 'I'): -1,
        ('Y', 'L'): -1,  ('Y', 'K'): -2,  ('Y', 'M'): -1,  ('Y', 'F'): 3,   ('Y', 'P'): -3,
        ('Y', 'S'): -2,  ('Y', 'T'): -2,  ('Y', 'W'): 2,   ('Y', 'Y'): 7,   ('Y', 'V'): -1,
        ('V', 'A'): 0,   ('V', 'R'): -3,  ('V', 'N'): -3,  ('V', 'D'): -3,  ('V', 'C'): -1,
        ('V', 'Q'): -2,  ('V', 'E'): -2,  ('V', 'G'): -3,  ('V', 'H'): -3,  ('V', 'I'): 3,
        ('V', 'L'): 1,   ('V', 'K'): -2,  ('V', 'M'): 1,   ('V', 'F'): -1,  ('V', 'P'): -2,
        ('V', 'S'): -2,  ('V', 'T'): 0,   ('V', 'W'): -3,  ('V', 'Y'): -1,  ('V', 'V'): 4,
    }


def _blosum62_pair_score(aa1: str, aa2: str) -> float:
    matrix = _get_blosum62_matrix()
    aa1 = aa1.upper()
    aa2 = aa2.upper()
    if (aa1, aa2) in matrix:
        return float(matrix[(aa1, aa2)])
    if (aa2, aa1) in matrix:
        return float(matrix[(aa2, aa1)])
    return 0.0


def load_state_file(state_path: Path) -> Dict[str, pd.DataFrame]:
    state_path = Path(state_path)
    state_data: Dict[str, List[Dict[str, object]]] = {}

    with state_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    data_lines = []
    for line in lines:
        if line.strip().startswith("#"):
            continue
        if line.strip().startswith("Node") and "Site" in line and "State" in line:
            continue
        data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data lines found in state file {state_path}")

    for line in data_lines:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        try:
            node = parts[0]
            site = int(parts[1])
            state = parts[2]
            probs = [float(p) for p in parts[3:]]
        except (ValueError, IndexError):
            continue
        state_data.setdefault(node, []).append({"Site": site, "State": state, "probs": probs, "raw": parts[3:]})

    return {node: pd.DataFrame(rows) for node, rows in state_data.items()}


def load_reconciliation_events(reconciliation_csv: Optional[Path]) -> Dict[str, str]:
    if reconciliation_csv is None:
        return {}

    reconciliation_csv = Path(reconciliation_csv)
    if not reconciliation_csv.exists():
        raise FileNotFoundError(f"Reconciliation CSV not found: {reconciliation_csv}")

    reconciliation_df = pd.read_csv(reconciliation_csv)
    required_columns = {"node_name", "event_type"}
    missing = required_columns - set(reconciliation_df.columns)
    if missing:
        raise ValueError(f"Reconciliation CSV is missing required columns: {sorted(missing)}")

    events: Dict[str, str] = {}
    for _, row in reconciliation_df.iterrows():
        node_name = row["node_name"]
        event_type = row["event_type"]
        if pd.isna(node_name) or pd.isna(event_type):
            continue
        node_label = str(node_name).strip()
        event_label = str(event_type).strip()
        if node_label:
            events[node_label] = event_label
    return events


def _leaf_signature(node) -> Tuple[str, ...]:
    return tuple(sorted(terminal.name for terminal in node.get_terminals() if terminal.name))


def _build_named_node_signatures(tree: Tree) -> Dict[str, Tuple[str, ...]]:
    signatures: Dict[str, Tuple[str, ...]] = {}
    for node in tree.get_nonterminals(order="level"):
        if not node.name:
            continue
        signature = _leaf_signature(node)
        if signature:
            signatures[str(node.name)] = signature
    return signatures


def _build_node_lookup(tree: Tree) -> Dict[str, Clade]:
    node_lookup: Dict[str, Clade] = {}
    for node in tree.get_nonterminals(order="level"):
        if node.name:
            node_lookup[str(node.name)] = node
    for node in tree.get_terminals():
        if node.name:
            node_lookup[str(node.name)] = node
    return node_lookup


def _build_signature_to_name_lookup(tree: Tree) -> Dict[Tuple[str, ...], str]:
    return {signature: node_name for node_name, signature in _build_named_node_signatures(tree).items()}


def _load_layer_cluster_lcas(layer_path: Path) -> Dict[int, str]:
    cluster_path = layer_path.with_name(layer_path.name.replace("tree_cluster_assignments", "tree_clusters"))
    if not cluster_path.exists():
        return {}

    cluster_df = pd.read_csv(cluster_path)
    required_columns = {"cluster_id", "lca_node"}
    if not required_columns.issubset(cluster_df.columns):
        return {}

    cluster_df = cluster_df.loc[cluster_df["cluster_id"].notna() & cluster_df["lca_node"].notna(), ["cluster_id", "lca_node"]].copy()
    cluster_df["cluster_id"] = pd.to_numeric(cluster_df["cluster_id"], errors="coerce")
    cluster_df = cluster_df.loc[cluster_df["cluster_id"].notna()].copy()
    if cluster_df.empty:
        return {}

    cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(int)
    cluster_df["lca_node"] = cluster_df["lca_node"].astype(str)
    return dict(zip(cluster_df["cluster_id"].tolist(), cluster_df["lca_node"].tolist()))


def _remap_reconciliation_events_to_asr_nodes(
    reconciliation_events: Dict[str, str],
    topology_tree: Tree,
    asr_tree: Tree,
) -> Dict[str, str]:
    topology_signatures = _build_named_node_signatures(topology_tree)
    asr_signature_to_name = {
        signature: node_name for node_name, signature in _build_named_node_signatures(asr_tree).items()
    }

    remapped: Dict[str, str] = {}
    unmapped_topology_nodes: List[str] = []

    for topology_node_name, event_type in reconciliation_events.items():
        topology_signature = topology_signatures.get(topology_node_name)
        if topology_signature is None:
            unmapped_topology_nodes.append(topology_node_name)
            continue

        asr_node_name = asr_signature_to_name.get(topology_signature)
        if asr_node_name is None:
            unmapped_topology_nodes.append(topology_node_name)
            continue

        previous = remapped.get(asr_node_name)
        if previous is None:
            remapped[asr_node_name] = event_type
            continue

        if previous != event_type:
            remapped[asr_node_name] = "Speciation" if "Speciation" in {previous, event_type} else event_type

    if unmapped_topology_nodes:
        warnings.warn(
            f"Unmapped reconciliation nodes: {len(unmapped_topology_nodes)}. "
            f"Examples: {unmapped_topology_nodes[:5]}",
            UserWarning,
            stacklevel=2,
        )

    return remapped


def _filter_pairs_by_reconciliation(
    pairs: List[Tuple[int, int]],
    level_lcas: Dict[int, str],
    reconciliation_events: Dict[str, str],
) -> Tuple[List[Tuple[int, int]], int, int]:
    if not reconciliation_events:
        return pairs, 0, 0

    lca_values = set(level_lcas.values())
    recon_keys = set(reconciliation_events.keys())
    matching_nodes = lca_values & recon_keys
    if not matching_nodes:
        raise ValueError(
            f"No reconciliation node names matched ASR LCA nodes. ASR sample: {sorted(lca_values)[:3]}, "
            f"Reconciliation sample: {sorted(recon_keys)[:3]}"
        )
        return pairs, 0, 0

    kept_pairs: List[Tuple[int, int]] = []
    skipped_pairs = 0
    skipped_speciation_pairs = 0
    for cluster_a, cluster_b in pairs:
        lca_a = level_lcas[cluster_a]
        lca_b = level_lcas[cluster_b]
        event_a = reconciliation_events.get(lca_a, "Duplication")
        event_b = reconciliation_events.get(lca_b, "Duplication")
        if event_a != "Duplication" or event_b != "Duplication":
            skipped_pairs += 1
            if event_a == "Speciation" or event_b == "Speciation":
                skipped_speciation_pairs += 1
            continue
        kept_pairs.append((cluster_a, cluster_b))
    return kept_pairs, skipped_pairs, skipped_speciation_pairs


def _validate_lca_coverage(level: str, level_lcas: Dict[int, str], ancestral_seqs: Dict[str, str], min_coverage: float = 0.95) -> None:
    if not level_lcas:
        raise ValueError(f"No LCA nodes were resolved for {level}.")

    missing = [node_id for node_id in level_lcas.values() if node_id not in ancestral_seqs]
    coverage = 1.0 - (len(missing) / float(len(level_lcas)))
    if missing:
        message = (
            f"{level.capitalize()} ASR coverage is {coverage:.2%} ({len(missing)}/{len(level_lcas)} LCA nodes missing from ancestral FASTA)."
        )
        if coverage < min_coverage:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)


def parse_clade_assignments(assignments_path: Path) -> pd.DataFrame:
    return pd.read_csv(assignments_path)


def calculate_recent_conservation(sequences: List[str], position: int) -> float:
    if not sequences:
        return 0.0
    residues = [seq[position] for seq in sequences if len(seq) > position and seq[position] not in {"-", "."}]
    if not residues:
        return 0.0
    if len(residues) == 1:
        return 1.0

    counts = Counter(residues)
    residue_types = list(counts)
    total_pairs = (len(residues) * (len(residues) - 1)) // 2
    if total_pairs == 0:
        return 0.0

    weighted_pair_sum = 0.0
    for i, aa_i in enumerate(residue_types):
        count_i = counts[aa_i]

        # Same-residue pairs contribute nC2 pairs at the diagonal substitution score.
        same_pairs = (count_i * (count_i - 1)) // 2
        if same_pairs:
            weighted_pair_sum += same_pairs * _blosum62_pair_score(aa_i, aa_i)

        for aa_j in residue_types[i + 1 :]:
            weighted_pair_sum += (count_i * counts[aa_j]) * _blosum62_pair_score(aa_i, aa_j)

    mean_score = weighted_pair_sum / float(total_pairs)
    normalized_score = (mean_score + 4.0) / 15.0
    return float(np.clip(normalized_score, 0.0, 1.0))


def calculate_ancestral_conservation(aa1: str, aa2: str) -> float:
    return 1.0 if aa1.upper() == aa2.upper() and aa1.upper() not in {"-", "."} else -1.0


def extract_posterior_probability(state_data: Dict[str, pd.DataFrame], node: str, site: int, aa: str) -> float:
    if node not in state_data:
        return 0.0
    node_df = state_data[node]
    site_rows = node_df[node_df["Site"] == site]
    if site_rows.empty:
        return 0.0
    row = site_rows.iloc[0]
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aa_upper = aa.upper()
    if aa_upper not in aa_order:
        return 0.0
    aa_idx = aa_order.index(aa_upper)
    if "probs" in row and aa_idx < len(row["probs"]):
        return float(row["probs"][aa_idx])
    return 0.0


def _reconstruct_ancestral_sequence_from_state(state_data: Dict[str, pd.DataFrame], node: str) -> Optional[str]:
    if node not in state_data:
        return None

    node_df = state_data[node]
    if node_df.empty or "Site" not in node_df.columns or "State" not in node_df.columns:
        return None

    sequence_parts: List[str] = []
    for _, row in node_df.sort_values("Site").iterrows():
        state = str(row["State"]).strip()
        if not state:
            sequence_parts.append("X")
            continue
        sequence_parts.append(state[0])
    return "".join(sequence_parts)


def _level_columns(assignments: pd.DataFrame, level: str) -> Tuple[str, str]:
    id_col = f"{level}_id"
    lca_col = f"{level}_lca_node"
    if id_col not in assignments.columns or lca_col not in assignments.columns:
        raise KeyError(f"Missing {id_col} or {lca_col} in assignments")
    return id_col, lca_col


def _resolve_hierarchical_lca_nodes(assignments: pd.DataFrame, tree: Tree) -> Dict[str, str]:
    lca_columns = [col for col in assignments.columns if col.endswith("_lca_node")]
    if not lca_columns:
        return {}

    terminal_names = {terminal.name for terminal in tree.get_terminals() if terminal.name}
    node_lookup = _build_node_lookup(tree)
    signature_to_name = _build_signature_to_name_lookup(tree)
    node_members: Dict[str, set] = defaultdict(set)
    for _, row in assignments.iterrows():
        sequence_id = str(row["sequence_id"])
        for col in lca_columns:
            raw_label = row[col]
            if pd.isna(raw_label):
                continue
            label = str(raw_label).strip()
            if label:
                node_members[label].add(sequence_id)

    resolved: Dict[str, str] = {}
    # Ensure internal nodes in provided tree have stable names for lookup.
    _ensure_node_names(tree, prefix="ASRNode")

    for label, members in node_members.items():
        present_members = [member for member in members if member in terminal_names]
        if not present_members:
            raise ValueError(f"No members for {label} were found in the ASR tree.")

        signature = tuple(sorted(present_members))
        resolved_name = signature_to_name.get(signature)
        if resolved_name is None:
            resolved_name = str(node_lookup[label].name) if label in node_lookup and node_lookup[label].name else f"ASRNode_unknown_{label}"
        resolved[label] = resolved_name
    return resolved


def _nearest_sister_pairs_for_level(
    assignments: pd.DataFrame,
    level: str,
    tree: Tree,
    parent_col: Optional[str] = None,
) -> List[Tuple[int, int]]:
    id_col, _ = _level_columns(assignments, level)
    _, lca_col = _level_columns(assignments, level)
    members_by_cluster: Dict[int, List[str]] = {
        int(cluster_id): [str(sequence_id) for sequence_id in group["sequence_id"].tolist()]
        for cluster_id, group in assignments.groupby(id_col)
    }

    cluster_to_parent: Dict[int, Optional[int]] = {}
    if parent_col is None:
        for cluster_id in members_by_cluster:
            cluster_to_parent[cluster_id] = None
    else:
        parent_lookup = assignments[[id_col, parent_col]].drop_duplicates().set_index(id_col)[parent_col].to_dict()
        for cluster_id in members_by_cluster:
            cluster_to_parent[cluster_id] = int(parent_lookup[cluster_id])

    pairs = set()
    cluster_ids = sorted(members_by_cluster.keys())

    node_lookup = _build_node_lookup(tree)
    cluster_nodes: Dict[int, Clade] = {}
    for cluster_id in tqdm(cluster_ids, desc=f"Resolving {level} LCAs", unit="cluster"):
        lca_name = None
        if lca_col in assignments.columns:
            cluster_rows = assignments.loc[assignments[id_col] == cluster_id, lca_col].dropna()
            if not cluster_rows.empty:
                lca_name = str(cluster_rows.iloc[0])
        if lca_name is None:
            continue
        node = node_lookup.get(lca_name)
        if node is None:
            continue
        cluster_nodes[cluster_id] = node

    for cluster_id in tqdm(cluster_ids, desc=f"Pairing {level}s", unit="cluster"):
        parent_value = cluster_to_parent[cluster_id]
        candidates = [
            other_id
            for other_id in cluster_ids
            if other_id != cluster_id and cluster_to_parent[other_id] == parent_value
        ]
        if not candidates:
            continue

        cluster_node = cluster_nodes[cluster_id]
        nearest_id = None
        nearest_distance = float("inf")
        for other_id in candidates:
            other_node = cluster_nodes[other_id]
            distance = float(tree.distance(cluster_node, other_node))
            if distance < nearest_distance or (
                distance == nearest_distance and (nearest_id is None or other_id < nearest_id)
            ):
                nearest_distance = distance
                nearest_id = other_id

        if nearest_id is not None:
            pairs.add(tuple(sorted((cluster_id, nearest_id))))

    return sorted(pairs)


def build_duplication_sister_pairs(
    tree: Tree,
    reconciliation_events: Dict[str, str],
    min_clade_size: int = 5,
) -> List[Tuple[str, str, str]]:
    node_lookup = {node.name: node for node in tree.get_nonterminals() if node.name}
    duplication_pairs: List[Tuple[str, str, str]] = []

    for duplication_node in tqdm(
        sorted(node_name for node_name, event_type in reconciliation_events.items() if event_type == "Duplication"),
        desc="Resolving duplication nodes",
        unit="node",
    ):
        node = node_lookup.get(duplication_node)
        if node is None or len(node.clades) != 2:
            continue

        left, right = node.clades
        left_name = getattr(left, "name", None)
        right_name = getattr(right, "name", None)
        if not left_name or not right_name:
            continue

        if len(left.get_terminals()) < min_clade_size or len(right.get_terminals()) < min_clade_size:
            continue

        duplication_pairs.append((str(duplication_node), str(left_name), str(right_name)))

    return duplication_pairs


def _ensure_node_names(tree: Tree, prefix: str = "InternalNode") -> None:
    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"{prefix}_{idx}"


def _discover_layer_assignment_files(assignments_path: Path) -> List[Path]:
    assignments_path = Path(assignments_path)
    # Prefer explicit per-layer directories created by tree_cluster's multi-threshold run
    layer_files = sorted(assignments_path.glob("layer_*/tree_cluster_assignments_layer*.csv"))
    if not layer_files:
        layer_files = sorted(assignments_path.parent.glob("layer_*/tree_cluster_assignments_layer*.csv"))
    if layer_files:
        return layer_files
    return [assignments_path]


def _load_layer_assignments(layer_path: Path) -> Tuple[pd.DataFrame, int, float]:
    df = pd.read_csv(layer_path)
    if {"sequence_id", "cluster_id"}.issubset(df.columns):
        layer_df = df[["sequence_id", "cluster_id"]].copy()
        layer_df["cluster_id"] = pd.to_numeric(layer_df["cluster_id"], errors="coerce")
        layer_df = layer_df.loc[layer_df["cluster_id"].notna()].copy()
        layer_df["cluster_id"] = layer_df["cluster_id"].astype(int)
        layer_df["sequence_id"] = layer_df["sequence_id"].astype(str)
        layer_index = int(df["layer_index"].iloc[0]) if "layer_index" in df.columns and not df.empty else 1
        threshold = float(df["threshold"].iloc[0]) if "threshold" in df.columns and not df.empty else float("nan")
        return layer_df, layer_index, threshold

    if "sequence_id" not in df.columns:
        raise ValueError(f"Missing sequence_id column in assignments file: {layer_path}")

    for id_col in ("subfamily_id", "family_id", "group_id"):
        if id_col in df.columns:
            temp = df[["sequence_id", id_col]].copy()
            temp["cluster_id"] = pd.to_numeric(temp[id_col], errors="coerce")
            temp = temp.loc[temp["cluster_id"].notna(), ["sequence_id", "cluster_id"]].copy()
            if temp.empty:
                continue
            temp["cluster_id"] = temp["cluster_id"].astype(int)
            temp["sequence_id"] = temp["sequence_id"].astype(str)
            return temp, 1, float("nan")

    raise ValueError(f"No usable cluster_id columns found in assignments file: {layer_path}")


def _normalize_event_type(event_value: Optional[str]) -> str:
    label = str(event_value).strip().lower() if event_value is not None else ""
    if label == "duplication":
        return "Duplication"
    if label == "speciation":
        return "Speciation"
    return "Unknown"


def _track_pairwise_subset(pairwise_df: pd.DataFrame, track_name: str) -> pd.DataFrame:
    if track_name == "combined":
        return pairwise_df.copy()
    event_type = TRACK_EVENT_TYPES.get(track_name)
    if event_type is None or pairwise_df.empty or "Event_Type" not in pairwise_df.columns:
        empty_subset = pairwise_df.iloc[0:0].copy()
        return cast(pd.DataFrame, empty_subset if track_name != "combined" else pairwise_df.copy())
    mask = pairwise_df["Event_Type"].astype(str) == str(event_type)
    subset = cast(pd.DataFrame, pairwise_df.loc[mask].copy())
    return subset


def _empty_pairwise_like(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    columns = list(pairwise_df.columns)
    return pd.DataFrame(columns=columns)


def _unique_pair_list(pairwise_df: pd.DataFrame) -> List[Tuple[str, str]]:
    if pairwise_df.empty or not {"left_child", "right_child"}.issubset(pairwise_df.columns):
        return []
    return [
        (str(row["left_child"]), str(row["right_child"]))
        for _, row in pairwise_df[["left_child", "right_child"]].drop_duplicates().iterrows()
    ]


def _build_layer_sister_pairs(
    layer_assignments: pd.DataFrame,
    layer_cluster_lcas: Dict[int, str],
    topology_tree: Tree,
    asr_tree: Tree,
    reconciliation_events: Dict[str, str],
    min_clade_size: int,
    topology_node_lookup: Dict[str, Clade],
    asr_node_lookup: Dict[str, Clade],
    asr_signature_to_name: Dict[Tuple[str, ...], str],
) -> List[Dict[str, object]]:
    topology_leaves = {terminal.name for terminal in topology_tree.get_terminals() if terminal.name}
    asr_leaves = {terminal.name for terminal in asr_tree.get_terminals() if terminal.name}

    grouped = layer_assignments.groupby("cluster_id")
    members_by_cluster: Dict[int, List[str]] = {}
    for cluster_id, cluster_df in grouped:
        members = [str(member) for member in cluster_df["sequence_id"].tolist()]
        present = [m for m in members if m in topology_leaves and m in asr_leaves]
        if len(present) >= min_clade_size:
            members_by_cluster[int(cluster_id)] = sorted(set(present))

    if len(members_by_cluster) < 2:
        return []

    cluster_topology_lca: Dict[int, Clade] = {}
    cluster_asr_lca_name: Dict[int, str] = {}
    for cluster_id, members in members_by_cluster.items():
        cluster_lca_name = layer_cluster_lcas.get(cluster_id)
        if not cluster_lca_name:
            continue
        topo_lca = topology_node_lookup.get(cluster_lca_name)
        if topo_lca is None:
            continue

        member_signature = tuple(sorted(member for member in members if member in asr_leaves))
        asr_lca_name = asr_signature_to_name.get(member_signature)
        if asr_lca_name is None or asr_lca_name not in asr_node_lookup:
            continue
        cluster_topology_lca[cluster_id] = topo_lca
        cluster_asr_lca_name[cluster_id] = asr_lca_name

    if len(cluster_topology_lca) < 2:
        return []

    asr_sig_to_name = {signature: node_name for node_name, signature in _build_named_node_signatures(asr_tree).items()}

    leaf_to_cluster: Dict[str, int] = {}
    for cluster_id, members in members_by_cluster.items():
        for member in members:
            leaf_to_cluster[member] = cluster_id

    seen_pairs = set()
    sister_rows: List[Dict[str, object]] = []
    for parent in topology_tree.get_nonterminals(order="postorder"):
        if len(parent.clades) != 2:
            continue

        child_cluster_ids: List[int] = []
        valid_parent = True
        for child in parent.clades:
            child_member_ids = {
                leaf_to_cluster[leaf.name]
                for leaf in child.get_terminals()
                if leaf.name in leaf_to_cluster
            }
            if len(child_member_ids) != 1:
                valid_parent = False
                break
            child_cluster_ids.append(next(iter(child_member_ids)))

        if not valid_parent or child_cluster_ids[0] == child_cluster_ids[1]:
            continue

        left_cluster, right_cluster = child_cluster_ids
        if left_cluster not in cluster_topology_lca or right_cluster not in cluster_topology_lca:
            continue

        pair_key = tuple(sorted((left_cluster, right_cluster)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        left_distance = float(topology_tree.distance(parent, cluster_topology_lca[left_cluster]))
        right_distance = float(topology_tree.distance(parent, cluster_topology_lca[right_cluster]))

        if left_distance <= right_distance:
            ldo_cluster, mdo_cluster = left_cluster, right_cluster
        else:
            ldo_cluster, mdo_cluster = right_cluster, left_cluster

        parent_name = str(parent.name) if parent.name else ""
        parent_signature = _leaf_signature(parent)
        source_parent_name = asr_sig_to_name.get(parent_signature, parent_name)
        
        event_type_candidate = reconciliation_events.get(source_parent_name)
        if event_type_candidate is None:
            event_type_candidate = reconciliation_events.get(parent_name)
        if event_type_candidate is None and parent_signature in asr_sig_to_name:
            event_type_candidate = reconciliation_events.get(asr_sig_to_name[parent_signature])
        
        event_type = _normalize_event_type(event_type_candidate)

        sister_rows.append(
            {
                "parent_node": parent_name,
                "source_parent_node": source_parent_name,
                "event_type": event_type,
                "left_cluster": int(left_cluster),
                "right_cluster": int(right_cluster),
                "left_lca": cluster_asr_lca_name[left_cluster],
                "right_lca": cluster_asr_lca_name[right_cluster],
                "left_branch_length": float(left_distance),
                "right_branch_length": float(right_distance),
                "ldo_cluster": int(ldo_cluster),
                "mdo_cluster": int(mdo_cluster),
                "ldo_lca": cluster_asr_lca_name[ldo_cluster],
                "mdo_lca": cluster_asr_lca_name[mdo_cluster],
                "left_members": members_by_cluster[left_cluster],
                "right_members": members_by_cluster[right_cluster],
            }
        )

    return sister_rows


def compute_multilevel_badasp_scores(
    alignment_path: Path,
    assignments_path: Path,
    ancestral_path: Path,
    state_path: Path,
    tree_path: Path,
    min_clade_size: int = 5,
    reconciliation_csv: Optional[Path] = None,
) -> Dict[str, object]:
    alignment = AlignIO.read(alignment_path, "fasta")
    ancestral_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(ancestral_path, "fasta")}
    state_data = load_state_file(state_path)
    topology_tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(topology_tree)
    topology_node_lookup = _build_node_lookup(topology_tree)
    reconciliation_events = load_reconciliation_events(reconciliation_csv)

    asr_tree = topology_tree
    asr_tree_path = Path(state_path).with_suffix(".treefile")
    if asr_tree_path.exists():
        asr_tree = Phylo.read(str(asr_tree_path), "newick")
    _ensure_node_names(asr_tree, prefix="ASRNode")
    asr_node_lookup = _build_node_lookup(asr_tree)
    asr_signature_to_name = _build_signature_to_name_lookup(asr_tree)

    if reconciliation_events:
        remapped_reconciliation_events = _remap_reconciliation_events_to_asr_nodes(
            reconciliation_events,
            topology_tree,
            asr_tree,
        )
        if remapped_reconciliation_events:
            reconciliation_events = {**reconciliation_events, **remapped_reconciliation_events}

    layer_files = _discover_layer_assignment_files(assignments_path)
    seq_dict = {record.id: str(record.seq) for record in alignment}
    aln_length = alignment.get_alignment_length()

    pairwise_rows: List[Dict[str, Any]] = []
    rc_cache: Dict[Tuple[str, int], float] = {}
    posterior_cache: Dict[Tuple[str, int, str], float] = {}
    reconstructed_sequences: Dict[str, str] = {}
    skipped_missing_nodes = 0
    layer_summaries: List[Dict[str, Any]] = []
    per_layer_track_results: Dict[str, Dict[int, Dict[str, Any]]] = {
        track_name: {} for track_name in TRACK_EVENT_TYPES
    }

    def _ancestral_sequence(node_name: str) -> Optional[str]:
        if node_name in ancestral_seqs:
            return ancestral_seqs[node_name]
        if node_name not in reconstructed_sequences:
            reconstructed_sequences[node_name] = _reconstruct_ancestral_sequence_from_state(state_data, node_name) or ""
        return reconstructed_sequences[node_name] or None

    def _rc(node_name: str, pos: int, sequences: List[str]) -> float:
        key = (node_name, pos)
        if key not in rc_cache:
            rc_cache[key] = calculate_recent_conservation(sequences, pos)
        return rc_cache[key]

    def _posterior(node: str, site: int, aa: str) -> float:
        key = (node, site, aa)
        if key not in posterior_cache:
            posterior_cache[key] = extract_posterior_probability(state_data, node, site, aa)
        return posterior_cache[key]

    fallback_done = False
    for layer_file in tqdm(layer_files, desc="BADASP layers", unit="layer"):
        try:
            layer_df, layer_index, layer_threshold = _load_layer_assignments(layer_file)
            layer_cluster_lcas = _load_layer_cluster_lcas(layer_file)
            sister_pairs = _build_layer_sister_pairs(
                layer_assignments=layer_df,
                layer_cluster_lcas=layer_cluster_lcas,
                topology_tree=topology_tree,
                asr_tree=asr_tree,
                reconciliation_events=reconciliation_events,
                min_clade_size=min_clade_size,
                topology_node_lookup=topology_node_lookup,
                asr_node_lookup=asr_node_lookup,
                asr_signature_to_name=asr_signature_to_name,
            )
        except ValueError as err:
            if "No usable cluster_id columns" not in str(err) or fallback_done:
                raise

            node_lookup = {node.name: node for node in asr_tree.get_nonterminals() if node.name}
            sister_pairs = []
            for duplication_node, left_name, right_name in build_duplication_sister_pairs(
                asr_tree,
                reconciliation_events,
                min_clade_size=min_clade_size,
            ):
                parent = node_lookup.get(duplication_node)
                left_node = node_lookup.get(left_name)
                right_node = node_lookup.get(right_name)
                if parent is None or left_node is None or right_node is None:
                    continue
                left_members = [leaf.name for leaf in left_node.get_terminals() if leaf.name]
                right_members = [leaf.name for leaf in right_node.get_terminals() if leaf.name]
                left_distance = float(asr_tree.distance(parent, left_node))
                right_distance = float(asr_tree.distance(parent, right_node))
                if left_distance <= right_distance:
                    ldo_cluster, mdo_cluster = 1, 2
                    ldo_lca, mdo_lca = left_name, right_name
                else:
                    ldo_cluster, mdo_cluster = 2, 1
                    ldo_lca, mdo_lca = right_name, left_name

                sister_pairs.append(
                    {
                        "parent_node": str(duplication_node),
                        "source_parent_node": str(duplication_node),
                        "event_type": _normalize_event_type(reconciliation_events.get(duplication_node)),
                        "left_cluster": 1,
                        "right_cluster": 2,
                        "left_lca": str(left_name),
                        "right_lca": str(right_name),
                        "left_branch_length": left_distance,
                        "right_branch_length": right_distance,
                        "ldo_cluster": int(ldo_cluster),
                        "mdo_cluster": int(mdo_cluster),
                        "ldo_lca": str(ldo_lca),
                        "mdo_lca": str(mdo_lca),
                        "left_members": left_members,
                        "right_members": right_members,
                    }
                )

            layer_index = 1
            layer_threshold = float("nan")
            fallback_done = True

        if not sister_pairs:
            continue

        print(f"[BADASP] Layer {layer_index}: input threshold={layer_threshold:.6f}, pair_candidates={len(sister_pairs)}")
        scored_pairs = 0
        layer_pairwise_rows: List[Dict[str, Any]] = []
        for pair_meta in tqdm(sister_pairs, desc=f"Scoring layer {layer_index}", unit="pair"):
            pair_meta_typed = cast(Dict[str, Any], pair_meta)
            left_name = str(pair_meta_typed["left_lca"])
            right_name = str(pair_meta_typed["right_lca"])

            left_sequence = _ancestral_sequence(left_name)
            right_sequence = _ancestral_sequence(right_name)
            if not left_sequence or not right_sequence:
                skipped_missing_nodes += 1
                continue

            left_members = cast(List[str], pair_meta_typed["left_members"])
            right_members = cast(List[str], pair_meta_typed["right_members"])
            left_sequences = [seq_dict[sid] for sid in left_members if sid in seq_dict]
            right_sequences = [seq_dict[sid] for sid in right_members if sid in seq_dict]
            if not left_sequences or not right_sequences:
                skipped_missing_nodes += 1
                continue

            scored_pairs += 1
            for pos in range(aln_length):
                if pos >= len(left_sequence) or pos >= len(right_sequence):
                    continue

                aa_left = left_sequence[pos]
                aa_right = right_sequence[pos]
                if aa_left in {"-", "."} or aa_right in {"-", "."}:
                    continue

                rc_left = _rc(left_name, pos, left_sequences)
                rc_right = _rc(right_name, pos, right_sequences)
                rc = (rc_left + rc_right) / 2.0
                ac = calculate_ancestral_conservation(aa_left, aa_right)
                p_left = _posterior(left_name, pos + 1, aa_left)
                p_right = _posterior(right_name, pos + 1, aa_right)
                p_ac = (p_left + p_right) / 2.0
                score = rc - (ac * p_ac)

                row = {
                    "layer_index": int(layer_index),
                    "layer_threshold": float(layer_threshold),
                    "duplication_node": str(pair_meta_typed["source_parent_node"]),
                    "lca_node_name": str(pair_meta_typed["parent_node"]),
                    "left_child": left_name,
                    "right_child": right_name,
                    "pair": f"{left_name}-{right_name}",
                    "position": int(pos + 1),
                    "rc": float(rc),
                    "ac": float(ac),
                    "p_ac": float(p_ac),
                    "score": float(score),
                    "Event_Type": str(pair_meta_typed["event_type"]),
                    "branch_len_left": float(pair_meta_typed["left_branch_length"]),
                    "branch_len_right": float(pair_meta_typed["right_branch_length"]),
                    "LDO_Cluster_ID": int(pair_meta_typed["ldo_cluster"]),
                    "MDO_Cluster_ID": int(pair_meta_typed["mdo_cluster"]),
                    "LDO_Node": str(pair_meta_typed["ldo_lca"]),
                    "MDO_Node": str(pair_meta_typed["mdo_lca"]),
                    "LDO_Label": str(pair_meta_typed["ldo_lca"]),
                    "MDO_Label": str(pair_meta_typed["mdo_lca"]),
                }
                pairwise_rows.append(row)
                layer_pairwise_rows.append(row)

        layer_pairwise_df = pd.DataFrame(layer_pairwise_rows)
        layer_track_results: Dict[str, Dict[str, Any]] = {}
        for track_name in TRACK_EVENT_TYPES:
            track_pairwise_df = _track_pairwise_subset(layer_pairwise_df, track_name)
            track_score_df, track_sdp_df, track_threshold = summarize_duplication_outputs(
                pairwise_df=track_pairwise_df,
                aln_length=aln_length,
            )
            layer_track_results[track_name] = {
                "scores": track_score_df,
                "sdps": track_sdp_df,
                "threshold": float(track_threshold),
                "pairwise": track_pairwise_df,
            }
            per_layer_track_results[track_name][int(layer_index)] = {
                "scores": track_score_df,
                "sdps": track_sdp_df,
                "threshold": float(track_threshold),
                "pairwise": track_pairwise_df.copy(),
                "candidate_pairs": int(len(sister_pairs)),
                "scored_pairs": int(scored_pairs),
                "linkage_threshold": float(layer_threshold),
                "layer_threshold": float(layer_threshold),
            }

        duplication_threshold = float(cast(float, layer_track_results["duplications"]["threshold"]))
        print(f"[BADASP] Layer {layer_index}: 95th percentile threshold={duplication_threshold:.6f}")
        layer_summaries.append(
            {
                "layer_index": int(layer_index),
                "linkage_threshold": float(layer_threshold),
                "layer_threshold": float(layer_threshold),
                "number_valid_pairs": int(scored_pairs),
                "candidate_pairs": int(len(sister_pairs)),
                "duplication_95th_percentile_threshold": float(cast(float, layer_track_results["duplications"]["threshold"])),
                "speciation_95th_percentile_threshold": float(cast(float, layer_track_results["speciations"]["threshold"])),
                "combined_95th_percentile_threshold": float(cast(float, layer_track_results["combined"]["threshold"])),
                "total_duplication_sdps": int(len(cast(pd.DataFrame, layer_track_results["duplications"]["sdps"]))),
                "total_speciation_sdps": int(len(cast(pd.DataFrame, layer_track_results["speciations"]["sdps"]))),
                "total_combined_sdps": int(len(cast(pd.DataFrame, layer_track_results["combined"]["sdps"]))),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        pairwise_df = pd.DataFrame(
            columns=[
                "layer_index",
                "layer_threshold",
                "duplication_node",
                "lca_node_name",
                "left_child",
                "right_child",
                "pair",
                "position",
                "rc",
                "ac",
                "p_ac",
                "score",
                "Event_Type",
                "branch_len_left",
                "branch_len_right",
                "LDO_Cluster_ID",
                "MDO_Cluster_ID",
                "LDO_Node",
                "MDO_Node",
                "LDO_Label",
                "MDO_Label",
            ]
        )

    global_layer_summary = pd.DataFrame(layer_summaries)
    track_results: Dict[str, Dict[str, Any]] = {}
    for track_name in TRACK_EVENT_TYPES:
        track_pairwise_df = _track_pairwise_subset(pairwise_df, track_name)
        track_score_df, track_sdp_df, track_threshold = summarize_duplication_outputs(
            pairwise_df=track_pairwise_df,
            aln_length=aln_length,
            percentile=95.0,
        )
        track_results[track_name] = {
            "pairwise": track_pairwise_df,
            "scores": track_score_df,
            "sdps": track_sdp_df,
            "threshold": float(track_threshold),
            "pairs": _unique_pair_list(track_pairwise_df),
            "filtered_pairs": int(skipped_missing_nodes),
            "filtered_speciation_pairs": int((pairwise_df["Event_Type"] == "Speciation").sum()) if track_name == "duplications" and "Event_Type" in pairwise_df.columns else 0,
            "candidate_pairs": int(sum(int(item["candidate_pairs"]) for item in layer_summaries)),
            "layer_scores": per_layer_track_results[track_name],
            "layer_summary": global_layer_summary,
        }

    return {**track_results, "global_layer_summary": global_layer_summary}


def summarize_duplication_outputs(
    pairwise_df: pd.DataFrame,
    aln_length: int,
    percentile: Optional[float] = 95.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    if pairwise_df.empty:
        threshold = float("nan") if percentile is None else 0.0
        score_df = pd.DataFrame(
            {
                "position": np.arange(1, aln_length + 1, dtype=int),
                "max_score": np.zeros(aln_length, dtype=float),
                "switch_count": np.zeros(aln_length, dtype=int),
                "global_threshold": np.full(aln_length, threshold, dtype=float),
                "badasp_score": np.zeros(aln_length, dtype=float),
            }
        )
        sdp_df = score_df.iloc[0:0].copy()
        return score_df, sdp_df, threshold

    valid_scores = pd.Series(pd.to_numeric(pairwise_df["score"], errors="coerce"), dtype=float)
    valid_scores = valid_scores[np.isfinite(valid_scores.to_numpy(dtype=float))]
    if percentile is None or valid_scores.empty:
        threshold = float("nan")
    else:
        threshold = float(np.percentile(valid_scores.to_numpy(dtype=float), float(percentile)))

    position_summary = (
        pairwise_df.groupby("position", as_index=False)
        .agg(max_score=("score", "max"))
        .astype({"position": int, "max_score": float})
    )

    switched_events = pairwise_df[pairwise_df["score"] >= threshold].copy() if np.isfinite(threshold) else pairwise_df.iloc[0:0].copy()
    switch_counts = switched_events.groupby("position").size().reset_index(name="switch_count")
    switch_counts = switch_counts.astype({"position": int, "switch_count": int})

    all_positions = pd.DataFrame({"position": np.arange(1, aln_length + 1, dtype=int)})
    score_df = all_positions.merge(position_summary, on="position", how="left")
    score_df = score_df.merge(switch_counts, on="position", how="left")
    score_df["max_score"] = score_df["max_score"].fillna(0.0).astype(float)
    score_df["switch_count"] = score_df["switch_count"].fillna(0).astype(int)
    score_df["global_threshold"] = float(threshold) if np.isfinite(threshold) else np.nan
    score_df["badasp_score"] = score_df["max_score"].astype(float)

    sdp_df = cast(pd.DataFrame, switched_events.groupby("position").size().reset_index(name="switch_count"))
    sdp_df = sdp_df.merge(position_summary, on="position", how="left")
    sdp_df["global_threshold"] = float(threshold) if np.isfinite(threshold) else np.nan
    sdp_df["badasp_score"] = sdp_df["max_score"].astype(float)
    sdp_df = cast(
        pd.DataFrame,
        sdp_df.sort_values(by=["switch_count", "max_score", "position"], ascending=[False, False, True]).reset_index(drop=True),
    )
    return score_df, sdp_df, threshold


def identify_sdps(scores_df: pd.DataFrame, percentile: float = 95.0) -> Tuple[pd.DataFrame, float]:
    if "switch_count" in scores_df.columns:
        if scores_df.empty:
            return cast(pd.DataFrame, scores_df.iloc[0:0].copy()), 0.0

        if "global_threshold" in scores_df.columns:
            threshold = float(scores_df["global_threshold"].iloc[0])
        else:
            threshold = float(np.percentile(scores_df["badasp_score"], percentile))

        sdps = scores_df[scores_df["switch_count"] > 0].copy()
        if sdps.empty:
            return cast(pd.DataFrame, scores_df.iloc[0:0].copy()), threshold
        sdps = cast(
            pd.DataFrame,
            sdps.sort_values(by=["switch_count", "max_score", "position"], ascending=[False, False, True]).reset_index(drop=True),
        )
        return sdps, threshold

    threshold = float(np.percentile(scores_df["badasp_score"], percentile))
    sdps = scores_df[scores_df["badasp_score"] >= threshold].copy()
    sdps = cast(pd.DataFrame, sdps.sort_values(by="badasp_score", ascending=False).reset_index(drop=True))
    return sdps, threshold


class BADASPCore:
    def __init__(
        self,
        alignment_path: Path,
        assignments_path: Path,
        ancestral_path: Path,
        state_path: Path,
        tree_path: Path,
        min_clade_size: int = 5,
        reconciliation_csv: Optional[Path] = None,
    ):
        self.alignment_path = Path(alignment_path)
        self.assignments_path = Path(assignments_path)
        self.ancestral_path = Path(ancestral_path)
        self.state_path = Path(state_path)
        self.tree_path = Path(tree_path)
        self.min_clade_size = min_clade_size
        self.reconciliation_csv = Path(reconciliation_csv) if reconciliation_csv is not None else None
        self.results: Optional[Dict[str, object]] = None
        self.sdps: Optional[Dict[str, pd.DataFrame]] = None
        self.thresholds: Optional[Dict[str, float]] = None

    def compute_scores(self) -> Dict[str, object]:
        self.results = compute_multilevel_badasp_scores(
            alignment_path=self.alignment_path,
            assignments_path=self.assignments_path,
            ancestral_path=self.ancestral_path,
            state_path=self.state_path,
            tree_path=self.tree_path,
            min_clade_size=self.min_clade_size,
            reconciliation_csv=self.reconciliation_csv,
        )
        logger.info("BADASP multilevel scores computed")
        return self.results

    def identify_sdps(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
        if self.results is None:
            self.compute_scores()
        assert self.results is not None
        self.sdps = {
            level: payload["sdps"]
            for level, payload in self.results.items()
            if isinstance(payload, dict) and "sdps" in payload
        }
        self.thresholds = {
            level: payload["threshold"]
            for level, payload in self.results.items()
            if isinstance(payload, dict) and "threshold" in payload
        }
        return self.sdps, self.thresholds

    def save_results(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.results is None:
            raise ValueError("Run compute_scores() first")
        track_payloads: Dict[str, Dict[str, Any]] = {
            level: payload
            for level, payload in self.results.items()
            if isinstance(payload, dict) and {"pairwise", "scores", "sdps"}.issubset(payload)
        }
        if not track_payloads:
            raise KeyError("Missing track payloads in BADASP results")

        for level, payload in sorted(track_payloads.items()):
            suffix = TRACK_OUTPUT_SUFFIXES.get(level, level)
            scores_df = cast(pd.DataFrame, payload["scores"])
            sdps_df = cast(pd.DataFrame, payload["sdps"])
            pairwise_df = cast(pd.DataFrame, payload["pairwise"])
            scores_df.to_csv(output_dir / f"badasp_scores_{suffix}.csv", index=False)
            sdps_df.to_csv(output_dir / f"badasp_sdps_{suffix}.csv", index=False)
            pairwise_df.to_csv(output_dir / f"raw_pairwise_{suffix}.csv", index=False)

            layer_scores = cast(Dict[int, Dict[str, Any]], payload.get("layer_scores", {}))
            for layer_index, layer_payload in sorted(layer_scores.items(), key=lambda item: int(item[0])):
                layer_dir = output_dir / f"layer_{int(layer_index):02d}"
                layer_dir.mkdir(parents=True, exist_ok=True)

                pairwise_df = cast(Optional[pd.DataFrame], layer_payload.get("pairwise"))
                scores_df = cast(Optional[pd.DataFrame], layer_payload.get("scores"))
                sdps_df = cast(Optional[pd.DataFrame], layer_payload.get("sdps"))
                if pairwise_df is not None:
                    pairwise_df.to_csv(layer_dir / f"raw_pairwise_{suffix}.csv", index=False)
                if scores_df is not None:
                    scores_df.to_csv(layer_dir / f"badasp_scores_{suffix}.csv", index=False)
                if sdps_df is not None:
                    sdps_df.to_csv(layer_dir / f"badasp_sdps_{suffix}.csv", index=False)

        global_layer_summary = self.results.get("global_layer_summary")
        if isinstance(global_layer_summary, pd.DataFrame) and not global_layer_summary.empty:
            global_layer_summary.to_csv(output_dir / "global_layer_summary.csv", index=False)
            global_layer_summary.to_csv(output_dir / "badasp_layer_summary.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 multi-level BADASP scoring")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--ancestral", default="data/interim/ancestral_sequences.fasta")
    parser.add_argument("--state", default="data/interim/asr_run.state")
    parser.add_argument("--tree", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--output-dir", default="results/badasp_scoring")
    parser.add_argument("--reconciliation-csv", default="results/reconciliation/duplication_nodes.csv")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    core = BADASPCore(
        alignment_path=Path(args.alignment),
        assignments_path=Path(args.assignments),
        ancestral_path=Path(args.ancestral),
        state_path=Path(args.state),
        tree_path=Path(args.tree),
        min_clade_size=args.min_clade_size,
        reconciliation_csv=Path(args.reconciliation_csv) if args.reconciliation_csv else None,
    )
    core.compute_scores()
    core.identify_sdps()
    core.save_results(Path(args.output_dir))

    if core.results is not None:
        payload = cast(Dict[str, Any], core.results.get("duplications", {}))
        kept_pairs = len(cast(List[Tuple[str, str]], payload.get("pairs", [])))
        filtered_pairs = int(cast(int, payload.get("filtered_pairs", 0)))
        threshold = float(cast(float, payload.get("threshold", 0.0)))
        print(f"Scored {kept_pairs} duplication pairs and filtered {filtered_pairs} candidates.")
        print(f"Global SDP threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
