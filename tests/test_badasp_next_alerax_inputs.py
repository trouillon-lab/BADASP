from pathlib import Path

from src.badasp_next.alerax_inputs import (
    build_alerax_family_file,
    build_alerax_mapping_file,
    normalize_species_tree,
)


def _write_fasta(path: Path) -> None:
    path.write_text(
        ">sp|A0A000|SEQ_A protein A OS=Species alpha OX=1111\n"
        "MKT\n"
        ">sp|A0A001|SEQ_B protein B OS=Species beta OX=2222\n"
        "MKR\n",
        encoding="utf-8",
    )


def test_build_alerax_mapping_and_family_file(tmp_path: Path) -> None:
    raw_fasta = tmp_path / "raw.fasta"
    clustered_fasta = tmp_path / "clustered.fasta"
    mapping_file = tmp_path / "treerecs_mapping.link"
    family_file = tmp_path / "families.txt"
    boot_trees = tmp_path / "IPR019888.boottrees"

    _write_fasta(raw_fasta)
    _write_fasta(clustered_fasta)
    boot_trees.write_text("(A,B);\n", encoding="utf-8")

    build_alerax_mapping_file(clustered_fasta, raw_fasta, mapping_file)
    build_alerax_family_file(boot_trees, mapping_file, family_file, family_name="IPR019888")

    assert mapping_file.read_text(encoding="utf-8").splitlines() == [
        "sp|A0A000|SEQ_A\t1111",
        "sp|A0A001|SEQ_B\t2222",
    ]

    family_text = family_file.read_text(encoding="utf-8")
    assert family_text.startswith("[FAMILIES]\n- IPR019888\n")
    assert "gene_tree = " in family_text
    assert "mapping = " in family_text


def test_normalize_species_tree_prunes_unobserved_taxids(tmp_path: Path) -> None:
    raw_fasta = tmp_path / "raw.fasta"
    source_tree = tmp_path / "source.nwk"
    normalized_tree = tmp_path / "normalized.nwk"

    _write_fasta(raw_fasta)
    source_tree.write_text("(1111:0.1,2222:0.2,3333:0.3);\n", encoding="utf-8")

    normalize_species_tree(source_tree, raw_fasta, normalized_tree, ncbi=None)

    normalized_text = normalized_tree.read_text(encoding="utf-8")
    assert "1111" in normalized_text
    assert "2222" in normalized_text
    assert "3333" not in normalized_text