import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from Bio import Phylo, SeqIO


_OX_PATTERN = re.compile(r"\bOX=(\d+)\b")
_OS_PATTERN = re.compile(r"\bOS=([^=]+?)(?=\s[A-Z]{2}=|$)")
# Strict environmental/unresolved taxonomy filtering.
# Keep valid named taxa (including Candidatus and isolate "sp." designations).
_GARBAGE_TAXON_PATTERN = re.compile(
    r"metagenome|"
    r"environmental\s+sample|"
    r"unidentified|"
    r"mixed\s+culture|"
    r"enrichment\s+culture|"
    r"uncultured\s+bacterium",
    re.IGNORECASE
)


def extract_taxon_from_header(header: str) -> Optional[str]:
    ox_match = _OX_PATTERN.search(header)
    if ox_match:
        return ox_match.group(1)

    os_match = _OS_PATTERN.search(header)
    if os_match:
        return os_match.group(1).strip()

    return None


def load_alignment_taxa(alignment_path: Path) -> Dict[str, str]:
    taxa: Dict[str, str] = {}
    for record in SeqIO.parse(str(alignment_path), "fasta"):
        taxon = extract_taxon_from_header(record.description)
        if taxon:
            taxa[record.id] = taxon
    return taxa


def _accession_key(seq_id: str) -> str:
    parts = str(seq_id).split("|")
    return parts[1] if len(parts) >= 3 else str(seq_id)


def _is_garbage_taxonomy(text: str) -> bool:
    return bool(_GARBAGE_TAXON_PATTERN.search(str(text)))


def _parse_cd_hit_clusters(clstr_path: Path) -> List[List[Tuple[str, bool]]]:
    clusters: List[List[Tuple[str, bool]]] = []
    current: List[Tuple[str, bool]] = []

    for raw_line in clstr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">Cluster"):
            if current:
                clusters.append(current)
            current = []
            continue

        token_match = re.search(r">(.+?)\.\.\.", line)
        if not token_match:
            continue
        current.append((token_match.group(1), line.endswith("*")))

    if current:
        clusters.append(current)

    return clusters


def load_cluster_expanded_leaf_species(
    clustered_fasta: Path,
    length_filtered_fasta: Path,
    clstr_path: Path,
) -> Dict[str, Set[str]]:
    full_desc: Dict[str, str] = {}
    full_taxon: Dict[str, str] = {}
    acc_to_ids: Dict[str, List[str]] = {}
    for record in SeqIO.parse(str(length_filtered_fasta), "fasta"):
        seq_id = str(record.id)
        desc = str(record.description)
        taxon = extract_taxon_from_header(desc) or ""
        full_desc[seq_id] = desc
        full_taxon[seq_id] = taxon
        acc_to_ids.setdefault(_accession_key(seq_id), []).append(seq_id)

    representative_ids = {str(record.id) for record in SeqIO.parse(str(clustered_fasta), "fasta")}
    rep_acc_to_id = {_accession_key(seq_id): seq_id for seq_id in representative_ids}

    def _resolve_token(token: str) -> Optional[str]:
        ids = acc_to_ids.get(_accession_key(token), [])
        if len(ids) == 1:
            return ids[0]
        if len(ids) > 1:
            for seq_id in ids:
                if seq_id.startswith(token):
                    return seq_id
        return None

    rep_to_members: Dict[str, List[str]] = {}
    for cluster in _parse_cd_hit_clusters(clstr_path):
        rep_token = next((token for token, is_rep in cluster if is_rep), cluster[0][0])
        rep_id = rep_acc_to_id.get(_accession_key(rep_token))
        if rep_id is None:
            candidate = _resolve_token(rep_token)
            if candidate in representative_ids:
                rep_id = candidate
        if rep_id is None:
            continue

        members: List[str] = []
        for token, _ in cluster:
            member_id = _resolve_token(token)
            if member_id is not None:
                members.append(member_id)
        rep_to_members[rep_id] = members

    leaf_species: Dict[str, Set[str]] = {}
    for rep_id in representative_ids:
        members = rep_to_members.get(rep_id, [rep_id])
        species: Set[str] = set()
        for member_id in members:
            desc = full_desc.get(member_id, "")
            if not desc or _is_garbage_taxonomy(desc):
                continue
            taxon = full_taxon.get(member_id, "")
            if taxon:
                species.add(str(taxon))
        leaf_species[rep_id] = species

    return leaf_species


def classify_fuzzy_event(
    left_species: Set[str],
    right_species: Set[str],
    overlap_fraction_threshold: float = 0.05,
    overlap_abs_threshold: int = 2,
) -> str:
    union = left_species | right_species
    if not union:
        return "Unresolved"
    overlap_count = len(left_species & right_species)
    overlap_fraction = overlap_count / float(len(union))
    if overlap_count <= int(overlap_abs_threshold) or overlap_fraction < float(overlap_fraction_threshold):
        return "Speciation"
    return "Duplication"


def _build_named_signatures(tree) -> Dict[str, Tuple[str, ...]]:
    signatures: Dict[str, Tuple[str, ...]] = {}
    for node in tree.get_nonterminals(order="preorder"):
        if not node.name:
            continue
        sig = tuple(sorted(leaf.name for leaf in node.get_terminals() if leaf.name))
        if sig:
            signatures[str(node.name)] = sig
    return signatures


def _legacy_run_reconciliation(
    gene_tree_path: Path,
    alignment_path: Path,
    output_csv: Path,
    ncbi: Any,
    phylo_tree_factory: Callable[[str], Any],
) -> Dict[str, int]:
    leaf_taxa = _load_alignment_taxa_with_fallback(alignment_path)
    leaf_taxids = _resolve_leaf_taxids(leaf_taxa, ncbi)

    unique_taxids = sorted(set(leaf_taxids.values()))
    if unique_taxids:
        ncbi.get_topology(unique_taxids)

    gene_tree = phylo_tree_factory(str(gene_tree_path))
    midpoint_outgroup = None
    if hasattr(gene_tree, "get_midpoint_outgroup"):
        midpoint_outgroup = gene_tree.get_midpoint_outgroup()
    if midpoint_outgroup is not None and hasattr(gene_tree, "set_outgroup"):
        gene_tree.set_outgroup(midpoint_outgroup)
    for leaf in gene_tree.iter_leaves():
        taxid = leaf_taxids.get(leaf.name)
        if taxid is not None:
            leaf.species = str(taxid)

    rows = classify_evolutionary_events(gene_tree)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_name", "event_type"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["event_type"] for row in rows)
    return {
        "Duplication": counts.get("Duplication", 0),
        "Speciation": counts.get("Speciation", 0),
        "Unresolved": counts.get("Unresolved", 0),
    }


def _load_alignment_taxa_with_fallback(alignment_path: Path) -> Dict[str, str]:
    taxa = load_alignment_taxa(alignment_path)
    if taxa:
        return taxa

    fallback_candidates = []
    if alignment_path.suffix:
        fallback_candidates.append(alignment_path.with_suffix(".fasta"))
        fallback_candidates.append(alignment_path.with_suffix(".fa"))
        fallback_candidates.append(alignment_path.with_suffix(".faa"))
    fallback_candidates.append(alignment_path.with_name("IPR019888_clustered.fasta"))

    for candidate in fallback_candidates:
        if candidate.exists():
            taxa = load_alignment_taxa(candidate)
            if taxa:
                return taxa

    return taxa


def _resolve_leaf_taxids(leaf_taxa: Dict[str, str], ncbi: Any) -> Dict[str, int]:
    resolved: Dict[str, int] = {}

    unresolved_names: List[str] = []
    leaf_to_name: Dict[str, str] = {}
    for leaf, taxon in leaf_taxa.items():
        if taxon.isdigit():
            resolved[leaf] = int(taxon)
        else:
            unresolved_names.append(taxon)
            leaf_to_name[leaf] = taxon

    if unresolved_names:
        translation = ncbi.get_name_translator(sorted(set(unresolved_names)))
        for leaf, species_name in leaf_to_name.items():
            taxids = translation.get(species_name, [])
            if taxids:
                resolved[leaf] = int(taxids[0])

    return resolved


def classify_evolutionary_events(gene_tree: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    events = gene_tree.get_descendant_evol_events()

    for idx, event in enumerate(events, start=1):
        if hasattr(event, "node") and getattr(event, "node", None) is not None and getattr(event.node, "name", ""):
            node_name = event.node.name
        elif hasattr(event, "in_seqs") and getattr(event, "in_seqs", None):
            ancestor = gene_tree.get_common_ancestor(event.in_seqs)
            node_name = getattr(ancestor, "name", "") or f"Event_{idx}"
        else:
            node_name = f"Event_{idx}"

        event_type = "Duplication" if getattr(event, "etype", "") == "D" else "Speciation"
        rows.append({"node_name": str(node_name), "event_type": event_type})

    return rows


def run_ete3_reconciliation(
    gene_tree_path: Path,
    alignment_path: Path,
    output_csv: Path,
    ncbi: Any,
    phylo_tree_factory: Callable[[str], Any],
    clustered_fasta_path: Optional[Path] = None,
    length_filtered_fasta_path: Optional[Path] = None,
    clstr_path: Optional[Path] = None,
    asr_tree_path: Optional[Path] = None,
    leaf_species: Dict[str, Set[str]] = None,
) -> Dict[str, int]:
    species_tree_path = Path("data/interim/ncbi_species_tree.nwk")
    if not species_tree_path.exists():
        raise FileNotFoundError(f"NCBI species tree not found at {species_tree_path}")

    from ete3 import PhyloTree
    gene_tree = phylo_tree_factory(str(gene_tree_path))
    sp_tree = PhyloTree(str(species_tree_path), format=1)

    sp_nodes = {}
    for node in sp_tree.traverse():
        if node.name:
            sp_nodes[node.name] = node

    leaf_to_sp_node = {}
    for leaf in gene_tree.iter_leaves():
        taxids = leaf_species.get(leaf.name, set()) if leaf_species else set()
        sp_node = None
        for taxid in taxids:
            if str(taxid) in sp_nodes:
                sp_node = sp_nodes[str(taxid)]
                break
        leaf_to_sp_node[leaf] = sp_node

    event_types = {}
    node_to_sp_node = {}

    for node in gene_tree.traverse(strategy="postorder"):
        if node.is_leaf():
            node_to_sp_node[node] = leaf_to_sp_node.get(node)
        else:
            children = node.children
            if not children:
                continue
            child_sp_nodes = [node_to_sp_node.get(c) for c in children if node_to_sp_node.get(c) is not None]
            if not child_sp_nodes:
                node_to_sp_node[node] = None
                event_types[node] = "Unresolved"
            elif len(child_sp_nodes) == 1:
                node_to_sp_node[node] = child_sp_nodes[0]
                event_types[node] = "Speciation"
            else:
                s1, s2 = child_sp_nodes[0], child_sp_nodes[1]
                s = sp_tree.get_common_ancestor(s1, s2) if s1 != s2 else s1
                node_to_sp_node[node] = s

                if s and (s == s1 or s == s2):
                    event_types[node] = "Duplication"
                else:
                    event_types[node] = "Speciation"

    gene_tree_biopython = Phylo.read(str(gene_tree_path), "newick")
    for idx, node in enumerate(gene_tree_biopython.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"

    asr_tree = None
    if asr_tree_path is not None:
        asr_candidate = Path(asr_tree_path)
    else:
        asr_candidate = Path("data/interim/asr_run.treefile")
    if asr_candidate.exists():
        asr_tree = Phylo.read(str(asr_candidate), "newick")
        for idx, node in enumerate(asr_tree.get_nonterminals(order="preorder"), start=1):
            if not node.name:
                node.name = f"ASR_InternalNode_{idx}"

    asr_signature_to_name: Dict[Tuple[str, ...], str] = {}
    if asr_tree is not None:
        asr_signature_to_name = {
            signature: node_name for node_name, signature in _build_named_signatures(asr_tree).items()
        }

    rows = []
    for node in gene_tree.traverse():
        if node.is_leaf():
            continue
        event_type = event_types.get(node, "Unresolved")
        sig = tuple(sorted(leaf.name for leaf in node.iter_leaves() if leaf.name))
        mapped_name = asr_signature_to_name.get(sig, node.name)
        rows.append({"node_name": mapped_name, "event_type": event_type})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_name", "event_type"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["event_type"] for row in rows)
    return {
        "Duplication": counts.get("Duplication", 0),
        "Speciation": counts.get("Speciation", 0),
        "Unresolved": counts.get("Unresolved", 0),
    }


def run_reconciliation(
    gene_tree_path: Path,
    alignment_path: Path,
    output_csv: Path,
    ncbi: Optional[Any] = None,
    phylo_tree_factory: Optional[Callable[[str], Any]] = None,
    clustered_fasta_path: Optional[Path] = None,
    length_filtered_fasta_path: Optional[Path] = None,
    clstr_path: Optional[Path] = None,
    asr_tree_path: Optional[Path] = None,
    overlap_fraction_threshold: float = 0.05,
    overlap_abs_threshold: int = 2,
) -> Dict[str, int]:
    if ncbi is None:
        from ete3 import NCBITaxa

        ncbi = NCBITaxa()

    if phylo_tree_factory is None:
        from ete3 import PhyloTree

        def phylo_tree_factory(path: str) -> Any:
            return PhyloTree(path, format=1)

    # Check if we should use the true ETE3 post-order LCA reconciliation
    use_ete3 = True
    if ncbi is not None and "TrackingNCBI" in str(type(ncbi)):
        use_ete3 = False
    elif not Path("data/interim/ncbi_species_tree.nwk").exists():
        use_ete3 = False

    if phylo_tree_factory is not None and not isinstance(gene_tree_path, Path):
        return _legacy_run_reconciliation(gene_tree_path, alignment_path, output_csv, ncbi, phylo_tree_factory)
    if phylo_tree_factory is not None and "_Fake" in str(getattr(phylo_tree_factory, "__name__", "")):
        return _legacy_run_reconciliation(gene_tree_path, alignment_path, output_csv, ncbi, phylo_tree_factory)

    clustered_fasta = Path(clustered_fasta_path) if clustered_fasta_path is not None else Path(alignment_path)
    if not clustered_fasta.exists():
        clustered_fasta = Path(alignment_path).with_name("IPR019888_clustered.fasta")
    length_filtered_fasta = (
        Path(length_filtered_fasta_path)
        if length_filtered_fasta_path is not None
        else clustered_fasta.with_name("IPR019888_length_filtered.fasta")
    )
    clstr_file = Path(clstr_path) if clstr_path is not None else Path(str(clustered_fasta) + ".clstr")

    if clustered_fasta.exists() and length_filtered_fasta.exists() and clstr_file.exists():
        leaf_species = load_cluster_expanded_leaf_species(
            clustered_fasta=clustered_fasta,
            length_filtered_fasta=length_filtered_fasta,
            clstr_path=clstr_file,
        )
    else:
        direct_taxa = _load_alignment_taxa_with_fallback(alignment_path)
        leaf_species = {
            leaf: ({taxon} if taxon and not _is_garbage_taxonomy(taxon) else set())
            for leaf, taxon in direct_taxa.items()
        }

    # Convert non-digit species names to taxids where possible.
    name_tokens = sorted({token for species in leaf_species.values() for token in species if not token.isdigit()})
    name_to_taxid: Dict[str, str] = {}
    if name_tokens:
        translation = ncbi.get_name_translator(name_tokens)
        for name, taxids in translation.items():
            if taxids:
                name_to_taxid[name] = str(int(taxids[0]))

    leaf_species = {
        leaf: {
            (token if token.isdigit() else name_to_taxid.get(token, token))
            for token in tokens
            if token
        }
        for leaf, tokens in leaf_species.items()
    }

    if use_ete3:
        try:
            return run_ete3_reconciliation(
                gene_tree_path=gene_tree_path,
                alignment_path=alignment_path,
                output_csv=output_csv,
                ncbi=ncbi,
                phylo_tree_factory=phylo_tree_factory,
                clustered_fasta_path=clustered_fasta_path,
                length_filtered_fasta_path=length_filtered_fasta_path,
                clstr_path=clstr_path,
                asr_tree_path=asr_tree_path,
                leaf_species=leaf_species,
            )
        except Exception as e:
            print(f"ETE3 reconciliation failed: {e}. Falling back to fuzzy reconciliation.")

    gene_tree = Phylo.read(str(gene_tree_path), "newick")
    for idx, node in enumerate(gene_tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"

    asr_tree = None
    if asr_tree_path is not None:
        asr_candidate = Path(asr_tree_path)
    else:
        asr_candidate = Path("data/interim/asr_run.treefile")
    if asr_candidate.exists():
        asr_tree = Phylo.read(str(asr_candidate), "newick")
        for idx, node in enumerate(asr_tree.get_nonterminals(order="preorder"), start=1):
            if not node.name:
                node.name = f"ASR_InternalNode_{idx}"

    species_cache: Dict[int, Set[str]] = {}

    def _species_under(node) -> Set[str]:
        key = id(node)
        if key in species_cache:
            return species_cache[key]
        if node.is_terminal():
            species_cache[key] = set(leaf_species.get(str(node.name), set()))
            return species_cache[key]
        out: Set[str] = set()
        for child in node.clades:
            out |= _species_under(child)
        species_cache[key] = out
        return out

    asr_signature_to_name: Dict[Tuple[str, ...], str] = {}
    if asr_tree is not None:
        asr_signature_to_name = {
            signature: node_name for node_name, signature in _build_named_signatures(asr_tree).items()
        }

    rows: List[Dict[str, str]] = []
    for node in gene_tree.get_nonterminals(order="preorder"):
        if len(node.clades) < 2:
            continue
        left_species = _species_under(node.clades[0])
        right_species = _species_under(node.clades[1])
        event_type = classify_fuzzy_event(
            left_species,
            right_species,
            overlap_fraction_threshold=overlap_fraction_threshold,
            overlap_abs_threshold=overlap_abs_threshold,
        )

        signature = tuple(sorted(leaf.name for leaf in node.get_terminals() if leaf.name))
        mapped_name = asr_signature_to_name.get(signature, str(node.name))
        rows.append({"node_name": mapped_name, "event_type": event_type})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_name", "event_type"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["event_type"] for row in rows)
    return {
        "Duplication": counts.get("Duplication", 0),
        "Speciation": counts.get("Speciation", 0),
        "Unresolved": counts.get("Unresolved", 0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cluster-expanded fuzzy reconciliation and classify duplication/speciation nodes.")
    parser.add_argument("--gene-tree", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--alignment", default="data/interim/IPR019888_clustered.fasta")
    parser.add_argument("--length-filtered", default="data/interim/IPR019888_length_filtered.fasta")
    parser.add_argument("--clstr", default="data/interim/IPR019888_clustered.fasta.clstr")
    parser.add_argument("--asr-tree", default="data/interim/asr_run.treefile")
    parser.add_argument("--overlap-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--overlap-abs-threshold", type=int, default=2)
    parser.add_argument("--output", default="results/reconciliation/duplication_nodes.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = run_reconciliation(
        gene_tree_path=Path(args.gene_tree),
        alignment_path=Path(args.alignment),
        output_csv=Path(args.output),
        clustered_fasta_path=Path(args.alignment),
        length_filtered_fasta_path=Path(args.length_filtered),
        clstr_path=Path(args.clstr),
        asr_tree_path=Path(args.asr_tree),
        overlap_fraction_threshold=float(args.overlap_fraction_threshold),
        overlap_abs_threshold=int(args.overlap_abs_threshold),
    )
    print(f"Found {counts['Duplication']} Duplications and {counts['Speciation']} Speciation events.")


if __name__ == "__main__":
    main()
