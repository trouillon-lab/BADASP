from pathlib import Path

import csv

from src.reconciliation import (
    classify_evolutionary_events,
    extract_taxon_from_header,
    load_alignment_taxa,
    run_reconciliation,
)


def test_extract_taxon_from_header_prefers_ox_taxid() -> None:
    header = "sp|P12345|ABC_XYZ Some protein OS=Escherichia coli OX=562 GN=abc"
    assert extract_taxon_from_header(header) == "562"


def test_extract_taxon_from_header_falls_back_to_os_species_name() -> None:
    header = "tr|Q9TEST|Q9TEST_ORG hypothetical protein OS=Bacillus subtilis GN=xyz"
    assert extract_taxon_from_header(header) == "Bacillus subtilis"


def test_load_alignment_taxa_maps_sequence_ids(tmp_path: Path) -> None:
    aln = tmp_path / "tiny.aln"
    aln.write_text(
        ">seqA OS=Escherichia coli OX=562\nAAAA\n"
        ">seqB OS=Bacillus subtilis\nAAAT\n",
        encoding="utf-8",
    )

    taxa = load_alignment_taxa(aln)

    assert taxa["seqA"] == "562"
    assert taxa["seqB"] == "Bacillus subtilis"


class _FakeLeaf:
    def __init__(self, name: str):
        self.name = name
        self.species = None


class _FakeAncestor:
    def __init__(self, name: str):
        self.name = name


class _FakeEvent:
    def __init__(self, etype: str, in_seqs):
        self.etype = etype
        self.in_seqs = in_seqs


class _FakeGeneTree:
    def __init__(self, newick_path: str):
        self.newick_path = newick_path
        self._leaves = [_FakeLeaf("seqA"), _FakeLeaf("seqB")]

    def iter_leaves(self):
        return iter(self._leaves)

    def get_descendant_evol_events(self):
        return [_FakeEvent("D", ["seqA"]), _FakeEvent("S", ["seqA", "seqB"])]

    def get_common_ancestor(self, names):
        if names == ["seqA"]:
            return _FakeAncestor("NodeDup")
        return _FakeAncestor("NodeSpec")


class _FakeNCBI:
    def __init__(self):
        self.requested_names = []
        self.requested_taxids = []

    def get_name_translator(self, names):
        self.requested_names.extend(names)
        out = {}
        if "Bacillus subtilis" in names:
            out["Bacillus subtilis"] = [1423]
        return out

    def get_topology(self, taxids):
        self.requested_taxids = list(taxids)
        return "species_tree"


def test_classify_evolutionary_events_returns_expected_labels() -> None:
    tree = _FakeGeneTree("dummy")
    rows = classify_evolutionary_events(tree)

    assert rows == [
        {"node_name": "NodeDup", "event_type": "Duplication"},
        {"node_name": "NodeSpec", "event_type": "Speciation"},
    ]


def test_run_reconciliation_writes_csv_and_summary_counts(tmp_path: Path) -> None:
    gene_tree = tmp_path / "mad_rooted.tree"
    aln = tmp_path / "tiny.aln"
    out_csv = tmp_path / "duplication_nodes.csv"

    gene_tree.write_text("(seqA:0.1,seqB:0.2)Root;\n", encoding="utf-8")
    aln.write_text(
        ">seqA OS=Escherichia coli OX=562\nAAAA\n"
        ">seqB OS=Bacillus subtilis\nAAAT\n",
        encoding="utf-8",
    )

    fake_ncbi = _FakeNCBI()

    counts = run_reconciliation(
        gene_tree_path=gene_tree,
        alignment_path=aln,
        output_csv=out_csv,
        ncbi=fake_ncbi,
        phylo_tree_factory=_FakeGeneTree,
    )

    assert counts == {"Duplication": 1, "Speciation": 1, "Unresolved": 0}
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"node_name": "NodeDup", "event_type": "Duplication"},
        {"node_name": "NodeSpec", "event_type": "Speciation"},
    ]

    # One taxid was direct (562), one resolved via species-name lookup (1423).
    assert set(fake_ncbi.requested_taxids) == {562, 1423}
