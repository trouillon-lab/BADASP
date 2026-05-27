from pathlib import Path
import csv

from ete3 import PhyloTree

from src.reconciliation import (
    _is_garbage_taxonomy,
    classify_fuzzy_event,
    load_alignment_taxa,
    load_cluster_expanded_leaf_species,
    run_reconciliation,
)


class _TrackingNCBI:
    def __init__(self):
        self.requested_taxids = []
        self.name_map = {
            "Human sapiens": [9606],
            "Pan troglodytes": [9598],
            "Mus musculus": [10090],
        }

    def get_name_translator(self, names):
        return {name: self.name_map[name] for name in names if name in self.name_map}

    def get_topology(self, taxids):
        self.requested_taxids = list(taxids)
        return "species_tree"


def test_tiny_tree_reconciliation_counts_are_exact(tmp_path: Path) -> None:
    gene_tree = tmp_path / "tiny_gene.tree"
    alignment = tmp_path / "tiny_alignment.fasta"
    output_csv = tmp_path / "duplication_nodes.csv"

    gene_tree.write_text("((Human_GeneA,Human_GeneB),(Chimp_GeneA,Mouse_GeneA));\n", encoding="utf-8")
    alignment.write_text(
        ">Human_GeneA OS=Human sapiens OX=9606\nAAAA\n"
        ">Human_GeneB OS=Human sapiens OX=9606\nAAAA\n"
        ">Chimp_GeneA OS=Pan troglodytes OX=9598\nAAAA\n"
        ">Mouse_GeneA OS=Mus musculus OX=10090\nAAAA\n",
        encoding="utf-8",
    )

    species_tree = "((Human sapiens,Pan troglodytes),Mus musculus);"
    assert species_tree

    ncbi = _TrackingNCBI()
    counts = run_reconciliation(
        gene_tree_path=gene_tree,
        alignment_path=alignment,
        output_csv=output_csv,
        ncbi=ncbi,
        phylo_tree_factory=lambda path: PhyloTree(path, format=1),
    )

    assert counts == {"Duplication": 0, "Speciation": 3, "Unresolved": 0}
    assert output_csv.exists()

    with output_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert sorted(row["event_type"] for row in rows) == ["Speciation", "Speciation", "Speciation"]


def test_reconciliation_falls_back_to_clustered_fasta_when_trimmed_alignment_has_no_headers(tmp_path: Path) -> None:
    gene_tree = tmp_path / "tiny_gene.tree"
    trimmed_alignment = tmp_path / "IPR019888_trimmed.aln"
    clustered_fasta = tmp_path / "IPR019888_clustered.fasta"
    output_csv = tmp_path / "duplication_nodes.csv"

    gene_tree.write_text("((Human_GeneA,Human_GeneB),(Chimp_GeneA,Mouse_GeneA));\n", encoding="utf-8")
    trimmed_alignment.write_text(
        ">Human_GeneA\nAAAA\n"
        ">Human_GeneB\nAAAA\n"
        ">Chimp_GeneA\nAAAA\n"
        ">Mouse_GeneA\nAAAA\n",
        encoding="utf-8",
    )
    clustered_fasta.write_text(
        ">Human_GeneA OS=Human sapiens OX=9606\nAAAA\n"
        ">Human_GeneB OS=Human sapiens OX=9606\nAAAA\n"
        ">Chimp_GeneA OS=Pan troglodytes OX=9598\nAAAA\n"
        ">Mouse_GeneA OS=Mus musculus OX=10090\nAAAA\n",
        encoding="utf-8",
    )

    taxa = load_alignment_taxa(clustered_fasta)
    assert len(taxa) == 4

    ncbi = _TrackingNCBI()
    counts = run_reconciliation(
        gene_tree_path=gene_tree,
        alignment_path=trimmed_alignment,
        output_csv=output_csv,
        ncbi=ncbi,
        phylo_tree_factory=lambda path: PhyloTree(path, format=1),
    )

    assert counts == {"Duplication": 0, "Speciation": 3, "Unresolved": 0}
    assert output_csv.exists()


def test_cluster_expansion_recovers_real_species_from_metagenome_representative(tmp_path: Path) -> None:
    clustered_fasta = tmp_path / "IPR019888_clustered.fasta"
    length_filtered_fasta = tmp_path / "IPR019888_length_filtered.fasta"
    clstr = tmp_path / "IPR019888_clustered.fasta.clstr"

    clustered_fasta.write_text(
        ">tr|REP123|REP123_META OS=marine metagenome OX=412755\nAAAA\n",
        encoding="utf-8",
    )
    length_filtered_fasta.write_text(
        ">tr|REP123|REP123_META OS=marine metagenome OX=412755\nAAAA\n"
        ">tr|REAL01|REAL01_BACT OS=Escherichia coli OX=562\nAAAA\n",
        encoding="utf-8",
    )
    clstr.write_text(
        ">Cluster 0\n"
        "0 4aa, >tr|REP123|REP123_ME... *\n"
        "1 4aa, >tr|REAL01|REAL01_BA... at 100.00%\n",
        encoding="utf-8",
    )

    species = load_cluster_expanded_leaf_species(
        clustered_fasta=clustered_fasta,
        length_filtered_fasta=length_filtered_fasta,
        clstr_path=clstr,
    )

    assert "tr|REP123|REP123_META" in species
    assert species["tr|REP123|REP123_META"] == {"562"}


def test_classify_fuzzy_event_thresholds() -> None:
    assert classify_fuzzy_event({"1", "2", "3"}, {"4", "5"}) == "Speciation"
    assert classify_fuzzy_event({"1", "2", "3"}, {"2", "3", "4"}) == "Speciation"
    assert classify_fuzzy_event({str(i) for i in range(100)}, {str(i) for i in range(50, 150)}) == "Duplication"
    assert classify_fuzzy_event(set(), set()) == "Unresolved"


def test_strict_garbage_taxonomy_terms_only() -> None:
    # Strict bucket terms should be filtered.
    assert _is_garbage_taxonomy("marine metagenome")
    assert _is_garbage_taxonomy("environmental sample")
    assert _is_garbage_taxonomy("mixed culture")
    assert _is_garbage_taxonomy("enrichment culture")
    assert _is_garbage_taxonomy("uncultured bacterium")
    assert _is_garbage_taxonomy("unidentified")


def test_valid_named_taxa_are_not_filtered() -> None:
    # Valid named taxa and isolate labels should remain usable for reconciliation.
    assert not _is_garbage_taxonomy("Candidatus Erwinia haradaeae")
    assert not _is_garbage_taxonomy("Acerihabitans sp. KWT182")
    assert not _is_garbage_taxonomy("uncultured Poseidoniia archaeon")
    assert not _is_garbage_taxonomy("Alphaproteobacteria bacterium")
