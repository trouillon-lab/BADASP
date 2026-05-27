from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

from Bio import SeqIO
from ete3 import NCBITaxa, Tree


_OX_PATTERN = re.compile(r"\bOX=(\d+)\b")
_OS_PATTERN = re.compile(r"\bOS=([^=]+?)(?=\s[A-Z]{2}=|$)")


def extract_taxid_token(header: str) -> Optional[str]:
    match = _OX_PATTERN.search(header)
    if match:
        return match.group(1)

    match = _OS_PATTERN.search(header)
    if match:
        return match.group(1).strip()

    return None


def _accession_key(seq_id: str) -> str:
    parts = str(seq_id).split("|")
    return parts[1] if len(parts) >= 3 else str(seq_id)


def _resolve_species_token(token: str, ncbi: Optional[NCBITaxa] = None) -> str:
    if token.isdigit():
        return token

    if ncbi is None:
        return token

    candidates = [token, token.replace("_", " ")]
    for candidate in candidates:
        translation = ncbi.get_name_translator([candidate])
        taxids = translation.get(candidate)
        if taxids:
            return str(int(taxids[0]))

    return token


def build_alerax_mapping_file(
    clustered_fasta: Path,
    raw_fasta: Path,
    output_path: Path,
    ncbi: Optional[NCBITaxa] = None,
) -> Path:
    """Write an AleRax-compatible gene-to-species mapping file.

    The file format matches the `treerecs_mapping.link` convention used by the
    AleRax repository: one tab-separated gene/species pair per line.
    """

    raw_taxa: dict[str, str] = {}
    for record in SeqIO.parse(str(raw_fasta), "fasta"):
        taxon_token = extract_taxid_token(record.description)
        if taxon_token is None:
            continue
        resolved_token = _resolve_species_token(taxon_token, ncbi=ncbi)
        raw_taxa[str(record.id)] = resolved_token
        raw_taxa[_accession_key(str(record.id))] = resolved_token

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in SeqIO.parse(str(clustered_fasta), "fasta"):
        gene_id = str(record.id)
        species = raw_taxa.get(gene_id) or raw_taxa.get(_accession_key(gene_id))
        if species is None:
            continue
        lines.append(f"{gene_id}\t{species}")

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def build_alerax_family_file(
    bootstrapped_gene_trees: Path,
    mapping_file: Path,
    output_path: Path,
    family_name: str = "IPR019888",
) -> Path:
    """Write the minimal family file consumed by AleRax."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "[FAMILIES]",
                f"- {family_name}",
                f"gene_tree = {bootstrapped_gene_trees.resolve()}",
                f"mapping = {mapping_file.resolve()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def normalize_species_tree(
    source_tree: Path,
    raw_fasta: Path,
    output_path: Path,
    ncbi: Optional[NCBITaxa] = None,
) -> Path:
    """Normalize a species tree to the observed taxids in the raw FASTA.

    The provided NCBI tree may use scientific names or contain leaves not present
    in the sequence set. This helper prefers the provided tree when possible and
    falls back to an NCBI topology built from the observed TaxIDs.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed_taxids = {
        int(match.group(1))
        for record in SeqIO.parse(str(raw_fasta), "fasta")
        for match in [_OX_PATTERN.search(record.description)]
        if match is not None
    }

    tree: Optional[Tree] = None
    if source_tree.exists() and source_tree.stat().st_size > 0:
        try:
            tree = Tree(str(source_tree), format=1)
        except Exception:
            tree = None

    if tree is not None:
        surviving_names: list[str] = []
        for leaf in list(tree.iter_leaves()):
            resolved = _resolve_species_token(str(leaf.name), ncbi=ncbi)
            if resolved.isdigit() and int(resolved) in observed_taxids:
                leaf.name = resolved
                surviving_names.append(resolved)

        if len(surviving_names) >= 2:
            tree.prune(surviving_names, preserve_branch_length=True)
            tree.write(format=1, outfile=str(output_path))
            return output_path

    if ncbi is None:
        ncbi = NCBITaxa()

    if not observed_taxids:
        raise ValueError(f"No TaxIDs found in {raw_fasta}")

    topology = ncbi.get_topology(sorted(observed_taxids))
    for node in topology.traverse():
        if node.name:
            node.name = str(node.name)
    topology.write(format=1, outfile=str(output_path))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build AleRax-compatible inputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapping_parser = subparsers.add_parser("mapping", help="Write a gene-to-species mapping file.")
    mapping_parser.add_argument("--clustered-fasta", required=True, type=Path)
    mapping_parser.add_argument("--raw-fasta", required=True, type=Path)
    mapping_parser.add_argument("--output", required=True, type=Path)

    species_parser = subparsers.add_parser("species-tree", help="Normalize the species tree for AleRax.")
    species_parser.add_argument("--source-tree", required=True, type=Path)
    species_parser.add_argument("--raw-fasta", required=True, type=Path)
    species_parser.add_argument("--output", required=True, type=Path)

    families_parser = subparsers.add_parser("families", help="Write the AleRax family file.")
    families_parser.add_argument("--boot-trees", required=True, type=Path)
    families_parser.add_argument("--mapping", required=True, type=Path)
    families_parser.add_argument("--output", required=True, type=Path)
    families_parser.add_argument("--family-name", default="IPR019888")

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "mapping":
        build_alerax_mapping_file(
            clustered_fasta=args.clustered_fasta,
            raw_fasta=args.raw_fasta,
            output_path=args.output,
        )
        return 0

    if args.command == "species-tree":
        normalize_species_tree(
            source_tree=args.source_tree,
            raw_fasta=args.raw_fasta,
            output_path=args.output,
        )
        return 0

    if args.command == "families":
        build_alerax_family_file(
            bootstrapped_gene_trees=args.boot_trees,
            mapping_file=args.mapping,
            output_path=args.output,
            family_name=args.family_name,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())