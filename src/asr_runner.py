import argparse
import csv
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def run_iqtree_asr(
    alignment_path: Path,
    tree_path: Path,
    output_prefix: Path,
    iqtree_binary: str = "iqtree2",
    model: str = "LG+G",
    threads: str = "AUTO",
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        iqtree_binary,
        "-s",
        str(alignment_path),
        "-te",
        str(tree_path),
        "-m",
        model,
        "-asr",
        "-T",
        str(threads),
        "--prefix",
        str(output_prefix),
    ]
    subprocess.run(cmd, check=True)


def wait_for_file(file_path: Path, timeout_seconds: int = 300, poll_interval: float = 1.0) -> Path:
    deadline = time.time() + timeout_seconds
    while time.time() <= deadline:
        if file_path.exists():
            return file_path
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for file: {file_path}")


def parse_iqtree_state_sequences(state_file: Path) -> Dict[str, str]:
    if not state_file.exists():
        raise FileNotFoundError(f"IQ-TREE state file not found: {state_file}")

    per_node_sites: Dict[str, Dict[int, str]] = defaultdict(dict)
    with state_file.open("r", encoding="utf-8") as handle:
        header_found = False
        for raw in tqdm(handle, desc="Parsing IQ-TREE .state", unit="line"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not header_found:
                if len(parts) < 3 or parts[0] != "Node" or parts[1] != "Site":
                    raise ValueError("Unexpected IQ-TREE .state format header.")
                header_found = True
                continue

            if len(parts) < 3:
                continue
            node, site_str, state = parts[0], parts[1], parts[2]
            if state == "-":
                state = "X"
            per_node_sites[node][int(site_str)] = state

    if not header_found:
        raise ValueError("Unexpected IQ-TREE .state format header.")

    node_sequences: Dict[str, str] = {}
    for node, site_map in per_node_sites.items():
        max_site = max(site_map) if site_map else 0
        seq_chars = [site_map.get(i, "X") for i in range(1, max_site + 1)]
        node_sequences[node] = "".join(seq_chars)

    return node_sequences


def _read_clade_members(assignments_csv: Path) -> Dict[int, List[str]]:
    clades: Dict[int, List[str]] = defaultdict(list)
    with assignments_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clade_id = int(row["clade_id"])
            clades[clade_id].append(row["terminal_name"])
    return clades


def _read_hierarchical_lca_members(assignments_csv: Path) -> Dict[str, List[str]]:
    members_by_node: Dict[str, List[str]] = defaultdict(list)
    with assignments_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return members_by_node

        lca_columns = [col for col in reader.fieldnames if col.endswith("_lca_node")]
        if not lca_columns:
            return members_by_node

        for row in reader:
            sequence_id = (row.get("sequence_id") or "").strip()
            if not sequence_id:
                continue
            for col in lca_columns:
                value = (row.get(col) or "").strip()
                if value and sequence_id not in members_by_node[value]:
                    members_by_node[value].append(sequence_id)
    return members_by_node


def _load_asr_node_mapping(mapping_csv: Path) -> Dict[str, str]:
    if not mapping_csv.exists():
        return {}

    mapping: Dict[str, str] = {}
    with mapping_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return mapping
        required = {"lca_node", "lca_node_asr"}
        if not required.issubset(reader.fieldnames):
            return mapping
        for row in reader:
            lca_node = (row.get("lca_node") or "").strip()
            asr_node = (row.get("lca_node_asr") or "").strip()
            if lca_node and asr_node:
                mapping[lca_node] = asr_node
    return mapping


def _resolve_cluster_token(token: str, candidate_ids: List[str]) -> str:
    exact_matches = [candidate for candidate in candidate_ids if candidate == token]
    if exact_matches:
        return exact_matches[0]

    prefix_matches = [candidate for candidate in candidate_ids if candidate.startswith(token)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ValueError(f"Ambiguous CD-HIT token {token!r}: {prefix_matches[:5]}")
    return token


def _load_cdhit_representative_mapping(cluster_map_csv: Path, clustered_fasta: Optional[Path] = None) -> Dict[str, str]:
    if not cluster_map_csv.exists():
        return {}

    representative_ids: List[str] = []
    if clustered_fasta is not None and clustered_fasta.exists():
        representative_ids = [record.id for record in SeqIO.parse(str(clustered_fasta), "fasta")]

    mapping: Dict[str, str] = {}
    current_representative: Optional[str] = None
    with cluster_map_csv.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                current_representative = None
                continue
            if ">" not in line:
                continue

            member_token = line.split(">", 1)[1].split()[0].strip().rstrip(".")
            if line.endswith("*"):
                current_representative = _resolve_cluster_token(member_token, representative_ids)
                mapping[member_token] = current_representative
                mapping[current_representative] = current_representative
                continue
            if current_representative is not None:
                mapping[member_token] = current_representative
    return mapping


def _resolve_representative_tip(member_id: str, representative_map: Dict[str, str]) -> str:
    if member_id in representative_map:
        return representative_map[member_id]

    for prefix_end in range(len(member_id) - 1, 0, -1):
        prefix = member_id[:prefix_end]
        if prefix in representative_map:
            return representative_map[prefix]

    return member_id


def extract_lca_ancestral_sequences(
    tree_path: Path,
    assignments_csv: Path,
    node_sequences: Dict[str, str],
    output_fasta: Path,
    asr_mapping_csv: Optional[Path] = None,
    cluster_mapping_csv: Optional[Path] = None,
    min_clade_size: int = 5,
) -> int:
    hierarchical_nodes = _read_hierarchical_lca_members(assignments_csv)
    asr_node_map = _load_asr_node_mapping(asr_mapping_csv) if asr_mapping_csv is not None else {}
    clustered_fasta = cluster_mapping_csv.with_suffix("") if cluster_mapping_csv is not None else None
    representative_map = (
        _load_cdhit_representative_mapping(cluster_mapping_csv, clustered_fasta=clustered_fasta)
        if cluster_mapping_csv is not None
        else {}
    )
    tree = Phylo.read(str(tree_path), "newick")
    tree_tip_names = {terminal.name for terminal in tree.get_terminals() if terminal.name}

    records: List[SeqRecord] = []

    if hierarchical_nodes:
        seen_nodes = set()
        for node_id in tqdm(sorted(hierarchical_nodes), desc="Extracting hierarchical LCAs", unit="node"):
            members = hierarchical_nodes[node_id]
            resolved_node_id = asr_node_map.get(node_id, node_id)
            if resolved_node_id == node_id:
                tree_members = sorted(
                    {
                        resolved_member
                        for resolved_member in (
                            _resolve_representative_tip(member, representative_map) for member in members
                        )
                        if resolved_member in tree_tip_names
                    }
                )
                if not tree_members:
                    raise KeyError(f"No representative members available for LCA node {node_id}.")
                try:
                    lca_node = tree.common_ancestor(tree_members)
                except Exception as exc:  # pragma: no cover - defensive wrap for tree lookup failures
                    raise KeyError(
                        f"No ancestral sequence found for LCA node {node_id} using members {tree_members[:5]}..."
                    ) from exc
                resolved_node_id = lca_node.name
                if not resolved_node_id:
                    raise ValueError(f"LCA node for {node_id} has no node name in ASR tree.")
            if resolved_node_id in seen_nodes:
                continue
            if resolved_node_id not in node_sequences:
                raise KeyError(f"No ancestral sequence found for LCA node {resolved_node_id}.")
            seen_nodes.add(resolved_node_id)
            records.append(
                SeqRecord(
                    Seq(node_sequences[resolved_node_id]),
                    id=resolved_node_id,
                    description="source=hierarchy_mapping",
                )
            )
    else:
        # Backward-compatible fallback for legacy single-level assignments.
        tree = Phylo.read(str(tree_path), "newick")
        clades = _read_clade_members(assignments_csv)
        seen_nodes = set()

        for clade_id, members in tqdm(sorted(clades.items()), desc="Extracting legacy clade LCAs", unit="clade"):
            if len(members) < min_clade_size:
                continue

            lca_node = tree.common_ancestor(members)
            if not lca_node.name:
                raise ValueError(f"LCA node for clade {clade_id} has no node name in ASR tree.")

            node_id = lca_node.name
            if node_id in seen_nodes:
                continue
            if node_id not in node_sequences:
                raise KeyError(f"No ancestral sequence found for LCA node {node_id}.")

            seen_nodes.add(node_id)
            records.append(
                SeqRecord(
                    Seq(node_sequences[node_id]),
                    id=node_id,
                    description=f"clade_id={clade_id};member_count={len(members)}",
                )
            )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(output_fasta), "fasta")
    return len(records)


def _discover_layer_assignment_files(assignments_path: Path) -> List[Path]:
    p = Path(assignments_path)
    if p.is_dir():
        layer_files = sorted(p.glob("layer_*/tree_cluster_assignments_layer*.csv"))
        if not layer_files:
            layer_files = sorted(p.glob("tree_cluster_assignments_layer*.csv"))
        return layer_files

    parent = p.parent
    layer_files = sorted(parent.glob("layer_*/tree_cluster_assignments_layer*.csv"))
    if layer_files:
        return layer_files
    # fallback to the single file
    return [p]


def extract_lca_ancestral_sequences_from_layers(
    tree_path: Path,
    assignments_dir: Path,
    node_sequences: Dict[str, str],
    output_fasta: Path,
    asr_mapping_csv: Optional[Path] = None,
    cluster_mapping_csv: Optional[Path] = None,
    min_clade_size: int = 5,
) -> int:
    """Scan all layer assignment files and extract unique LCA ancestral sequences.

    assignments_dir: directory containing `layer_XX/tree_cluster_assignments_layerXX.csv` files
    """
    asr_node_map = _load_asr_node_mapping(asr_mapping_csv) if asr_mapping_csv is not None else {}
    clustered_fasta = cluster_mapping_csv.with_suffix("") if cluster_mapping_csv is not None else None
    representative_map = (
        _load_cdhit_representative_mapping(cluster_mapping_csv, clustered_fasta=clustered_fasta)
        if cluster_mapping_csv is not None
        else {}
    )

    tree = Phylo.read(str(tree_path), "newick")
    tree_tip_names = {terminal.name for terminal in tree.get_terminals() if terminal.name}

    # collect members per reported LCA label across all layer files
    members_by_node: Dict[str, List[str]] = defaultdict(list)
    layer_files = _discover_layer_assignment_files(assignments_dir)
    for f in layer_files:
        try:
            with f.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    seq = (row.get("sequence_id") or row.get("terminal_name") or "").strip()
                    lca = (row.get("lca_node") or row.get("lca") or "").strip()
                    if seq and lca and seq not in members_by_node[lca]:
                        members_by_node[lca].append(seq)
        except Exception:
            continue

    records: List[SeqRecord] = []
    seen = set()
    for node_id in tqdm(sorted(members_by_node), desc="Extracting LCAs from layers", unit="node"):
        members = members_by_node[node_id]
        resolved_node_id = asr_node_map.get(node_id, node_id)
        if resolved_node_id == node_id:
            tree_members = sorted(
                {
                    resolved_member
                    for resolved_member in (
                        _resolve_representative_tip(member, representative_map) for member in members
                    )
                    if resolved_member in tree_tip_names
                }
            )
            if not tree_members:
                # skip nodes without representatives in tree
                continue
            try:
                lca_node = tree.common_ancestor(tree_members)
            except Exception:
                continue
            resolved_node_id = lca_node.name or resolved_node_id

        if resolved_node_id in seen:
            continue
        if resolved_node_id not in node_sequences:
            # skip missing ASR nodes
            continue
        seen.add(resolved_node_id)
        records.append(SeqRecord(Seq(node_sequences[resolved_node_id]), id=resolved_node_id, description="source=layer_mapping"))

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(output_fasta), "fasta")
    return len(records)


def run_asr_pipeline(
    alignment_path: Path,
    tree_path: Path,
    assignments_csv: Path,
    output_fasta: Path,
    iqtree_binary: str = "iqtree2",
    output_prefix: Path = Path("data/interim/asr_run"),
    asr_mapping_csv: Optional[Path] = None,
    cluster_mapping_csv: Optional[Path] = None,
    min_clade_size: int = 5,
    reuse_existing: bool = False,
) -> int:
    state_file = output_prefix.with_suffix(".state")
    asr_tree = output_prefix.with_suffix(".treefile")

    if not reuse_existing or not state_file.exists() or not asr_tree.exists():
        run_iqtree_asr(
            alignment_path=alignment_path,
            tree_path=tree_path,
            output_prefix=output_prefix,
            iqtree_binary=iqtree_binary,
        )

    wait_for_file(state_file)

    node_sequences = parse_iqtree_state_sequences(state_file)

    assignments_path = Path(assignments_csv)
    if assignments_path.is_dir():
        return extract_lca_ancestral_sequences_from_layers(
            tree_path=asr_tree,
            assignments_dir=assignments_path,
            node_sequences=node_sequences,
            output_fasta=output_fasta,
            asr_mapping_csv=asr_mapping_csv,
            cluster_mapping_csv=cluster_mapping_csv,
            min_clade_size=min_clade_size,
        )

    return extract_lca_ancestral_sequences(
        tree_path=asr_tree,
        assignments_csv=Path(assignments_csv),
        node_sequences=node_sequences,
        output_fasta=output_fasta,
        asr_mapping_csv=asr_mapping_csv,
        cluster_mapping_csv=cluster_mapping_csv,
        min_clade_size=min_clade_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IQ-TREE ASR and extract LCA ancestral sequences.")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--tree", default="data/interim/IPR019888.tree")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--output", default="data/interim/ancestral_sequences.fasta")
    parser.add_argument("--iqtree-binary", default="iqtree2")
    parser.add_argument("--prefix", default="data/interim/asr_run")
    parser.add_argument("--asr-map", default="results/topological_clustering/tree_clusters_asr_mapped.csv")
    parser.add_argument("--cluster-map", default="data/interim/IPR019888_clustered.fasta.clstr")
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    written = run_asr_pipeline(
        alignment_path=Path(args.alignment),
        tree_path=Path(args.tree),
        assignments_csv=Path(args.assignments),
        output_fasta=Path(args.output),
        iqtree_binary=args.iqtree_binary,
        output_prefix=Path(args.prefix),
        asr_mapping_csv=Path(args.asr_map) if args.asr_map else None,
        cluster_mapping_csv=Path(args.cluster_map) if args.cluster_map else None,
        min_clade_size=args.min_clade_size,
        reuse_existing=args.reuse_existing,
    )
    print(f"LCA ancestral sequences written: {written}")


if __name__ == "__main__":
    main()
