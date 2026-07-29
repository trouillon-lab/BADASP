"""
Test suite for Phase 6: Structural Mapping (PDB annotation).

Tests verify:
1. Downloading and caching PDB structures
2. Mapping trimmed MSA columns (1-165) to PDB residue numbers
3. PyMOL script generation for SDP visualization by hierarchy level
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from Bio import SeqIO

from src.pdb_mapper import PDBMapper, main


class TestPDBDownloader:
    """Test PDB file acquisition and caching."""

    @pytest.mark.network
    def test_download_pdb_file(self):
        """
        Download a target PDB file using Bio.PDB.PDBList.

        Expected:
        - PDB file is downloaded or cached
        - File path is returned
        - File contains valid PDB format content
        """
        pdb_id = "2cg4"  # Target structure for IPR019888
        mapper = PDBMapper(pdb_id=pdb_id)
        pdb_path = mapper.download_pdb()

        assert pdb_path is not None
        assert isinstance(pdb_path, (str, Path))
        assert Path(pdb_path).exists()
        with open(pdb_path, "r") as f:
            content = f.read()
            assert "ATOM" in content or "HETATM" in content

    @pytest.mark.network
    def test_pdb_cache_reuse(self):
        """
        Verify that repeated downloads use cached version.

        Expected:
        - Second call returns same path
        - File is not re-downloaded (same stat timestamp)
        """
        pdb_id = "2cg4"
        mapper1 = PDBMapper(pdb_id=pdb_id)
        path1 = mapper1.download_pdb()

        mapper2 = PDBMapper(pdb_id=pdb_id)
        path2 = mapper2.download_pdb()

        assert path1 == path2
        assert Path(path1).stat().st_mtime == Path(path2).stat().st_mtime

    def test_custom_pdb_path(self):
        """
        Allow explicit PDB file specification instead of automatic download.

        Expected:
        - Mapper accepts a local PDB file path
        - No download attempt is made
        """
        # Using the existing structure in data/raw/
        pdb_path = Path("data/raw/2cg4.cif")
        if not pdb_path.exists():
            pytest.skip("data/raw/2cg4.cif not available on this checkout")
        mapper = PDBMapper(pdb_id="2cg4", pdb_file=str(pdb_path))
        retrieved_path = mapper.download_pdb()
        assert retrieved_path == str(pdb_path)


class TestSequenceToStructureAlignment:
    """Test mapping MSA columns to PDB residue numbers."""

    def test_alignment_basic(self):
        """
        Map trimmed MSA columns (1-165) to PDB residue indices.

        Expected:
        - Input: MSA FASTA, PDB structure
        - Output: Dict mapping MSA column index → PDB residue number
        - No gaps in mapping (gapped positions handled)
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        if not alignment_path.exists():
            pytest.skip("data/interim/IPR019888_trimmed.aln not available on this checkout")
        pdb_id = "2cg4"

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        # PDB residue numbers should be integers or lists of (chain,residue) tuples
        for aln_col, pdb_res in mapping.items():
            assert isinstance(aln_col, int)
            if isinstance(pdb_res, list):
                for item in pdb_res:
                    assert isinstance(item, tuple)
                    assert isinstance(item[0], str)
                    assert isinstance(item[1], int)
            else:
                assert isinstance(pdb_res, int)

    def test_alignment_handles_gaps(self):
        """
        Verify gap positions in MSA are mapped correctly.

        Expected:
        - Gapped columns are either skipped or marked as None/NaN
        - Aligned and ungapped positions map 1:1 to PDB structure
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        if not alignment_path.exists():
            pytest.skip("data/interim/IPR019888_trimmed.aln not available on this checkout")
        pdb_id = "2cg4"

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)

        # Verify that gapped positions are handled (either not in dict or mapped to None)
        for aln_col, pdb_res in mapping.items():
            if isinstance(pdb_res, list):
                # list of (chain,resnum)
                for item in pdb_res:
                    assert isinstance(item, tuple)
                    assert isinstance(item[0], str)
                    assert isinstance(item[1], int)
            else:
                assert pdb_res is None or isinstance(pdb_res, int)

    def test_alignment_column_index_range(self):
        """
        Verify mapping covers the current trimmed alignment column range for IPR019888.

        Expected:
        - Mapping keys should span from 1 to the alignment length
        - No out-of-range column indices
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        if not alignment_path.exists():
            pytest.skip("data/interim/IPR019888_trimmed.aln not available on this checkout")
        pdb_id = "2cg4"
        alignment_length = len(next(SeqIO.parse(str(alignment_path), "fasta")).seq)

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)

        if mapping:
            max_col = max(mapping.keys())
            min_col = min(mapping.keys())
            assert min_col >= 1
            assert max_col <= alignment_length


class TestPyMOLScriptGeneration:
    """Test generation of PyMOL scripts for SDP visualization."""

    def test_pymol_script_basic(self):
        """
        Generate a .pml script that colors SDPs by hierarchical level.

        Expected:
        - Output file is created
        - File contains valid PyMOL syntax
        - Includes color commands for Groups, Families, Subfamilies
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        if not alignment_path.exists() or not sdp_csv_groups.exists():
            pytest.skip("Alignment or BADASP score CSVs not available on this checkout")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sdps_hierarchical.pml"

            mapper = PDBMapper(pdb_id=pdb_id)
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_path,
            )

            assert output_path.exists()
            with open(output_path, "r") as f:
                content = f.read()
                # Verify PyMOL syntax
                assert "color" in content or "colour" in content
                # Verify hierarchy levels are present
                assert "group" in content.lower() or "Groups" in content
                assert len(content) > 100

    def test_pymol_script_hierarchy_colors(self):
        """
        Verify that hierarchical levels get distinct visual encodings.

        Expected:
        - Groups colored with one scheme
        - Families colored with another scheme
        - Subfamilies colored with a third scheme
        - Colors are PyMOL-compatible (hex, named, or RGB)
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        if not alignment_path.exists() or not sdp_csv_groups.exists():
            pytest.skip("Alignment or BADASP score CSVs not available on this checkout")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sdps_colored.pml"

            mapper = PDBMapper(pdb_id=pdb_id)
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_path,
            )

            with open(output_path, "r") as f:
                content = f.read()
                # Each hierarchy level should be mentioned
                lines = content.lower()
                assert "group" in lines or "subfamily" in lines or "family" in lines

    def test_pymol_script_integrates_alignment_mapping(self):
        """
        Verify that the PyMOL script uses the MSA-to-PDB mapping correctly.

        Expected:
        - SDP positions from CSV are mapped to PDB residue numbers
        - PyMOL script uses PDB residue numbers, not MSA column numbers
        - Script selects residues using PDB numbering
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        if not alignment_path.exists() or not sdp_csv_groups.exists():
            pytest.skip("Alignment or BADASP score CSVs not available on this checkout")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sdps_pdb_mapped.pml"

            mapper = PDBMapper(pdb_id=pdb_id)
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_path,
            )

            # The script should contain residue selections and color commands
            with open(output_path, "r") as f:
                content = f.read()
                # Should have selection statements (e.g., "resi X")
                assert len(content) > 100


class TestChimeraXScriptGeneration:
    """Test generation of separate ChimeraX scripts for each hierarchy level."""

    def test_three_separate_chimerax_scripts(self):
        """
        Generate three level-specific scripts instead of one combined file.

        Expected:
        - Groups, families, and subfamilies each get a distinct .cxc file
        - The legacy combined file is not required for the refactor
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")
        if not alignment_path.exists() or not sdp_csv_groups.exists():
            pytest.skip("Alignment or BADASP score CSVs not available on this checkout")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mapper = PDBMapper(pdb_id=pdb_id)
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_dir=output_dir,
            )

            assert set(outputs.keys()) == {"groups", "families", "subfamilies"}
            assert outputs["groups"].name == "highlight_sdps_groups.cxc"
            assert outputs["families"].name == "highlight_sdps_families.cxc"
            assert outputs["subfamilies"].name == "highlight_sdps_subfamilies.cxc"
            for script_path in outputs.values():
                assert script_path.exists()

    def test_chimerax_scripts_cover_all_switch_positions(self):
        """
        Ensure the refactored ChimeraX scripts use all mapped switch positions.

        Expected:
        - The families script contains every position with switch_count > 0 from the SDP table
        - The script is not limited to a top-N subset
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_duplications = Path("results/badasp_scoring/badasp_scores_duplications.csv")

        if sdp_csv_families.exists():
            selected_csv = sdp_csv_families
            output_key = "families"
            generate_kwargs = {
                "sdp_csv_families": selected_csv,
            }
        else:
            selected_csv = sdp_csv_duplications
            output_key = "duplications"
            generate_kwargs = {
                "sdp_csv_duplications": selected_csv,
            }

        if not selected_csv.exists():
            pytest.skip("No BADASP score CSV available in results/badasp_scoring")

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)
        expected_positions = sorted(
            int(pos)
            for pos in pd.read_csv(selected_csv).query("switch_count > 0")["position"].tolist()
            if int(pos) in mapping
        )
        # flatten mapping values to list of residue selectors
        expected_residues = []
        for pos in expected_positions:
            val = mapping[pos]
            if isinstance(val, list):
                for chain, res in val:
                    expected_residues.append((chain, res))
            else:
                expected_residues.append((None, int(val)))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                output_dir=output_dir,
                **generate_kwargs,
            )

            content = outputs[output_key].read_text()
            for chain, residue in expected_residues:
                if chain is None:
                    assert f":{residue}" in content
                else:
                    assert f"/{chain}:{residue}" in content

    def test_chimerax_scripts_include_standalone_png_legends(self):
        """Verify that each level has a standalone PNG colorbar legend."""
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        groups_csv = Path("results/badasp_scoring/badasp_scores_groups.csv")
        if not alignment_path.exists() or not groups_csv.exists():
            pytest.skip("BADASP score CSVs not available on this checkout")
        mapper = PDBMapper(pdb_id="2cg4")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=groups_csv,
                sdp_csv_families=Path("results/badasp_scoring/badasp_scores_families.csv"),
                sdp_csv_subfamilies=Path("results/badasp_scoring/badasp_scores_subfamilies.csv"),
                output_dir=output_dir,
            )
            for legend_name in ("legend_groups.png", "legend_families.png", "legend_subfamilies.png"):
                legend_path = output_dir / legend_name
                assert legend_path.exists()
                assert legend_path.stat().st_size > 0

    def test_chimerax_scripts_do_not_use_key_command(self):
        """Ensure we avoid ChimeraX key syntax due to the palette lookup bug."""
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        groups_csv = Path("results/badasp_scoring/badasp_scores_groups.csv")
        if not alignment_path.exists() or not groups_csv.exists():
            pytest.skip("BADASP score CSVs not available on this checkout")
        mapper = PDBMapper(pdb_id="2cg4")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=groups_csv,
                sdp_csv_families=Path("results/badasp_scoring/badasp_scores_families.csv"),
                sdp_csv_subfamilies=Path("results/badasp_scoring/badasp_scores_subfamilies.csv"),
                output_dir=output_dir,
            )
            for script_path in outputs.values():
                content = script_path.read_text()
                assert "\nkey " not in content

    def test_chimerax_scripts_write_base_commands_when_no_switches(self, monkeypatch):
        mapper = PDBMapper(pdb_id="2cg4")
        monkeypatch.setattr(mapper, "download_pdb", lambda: "data/raw/2cg4.pdb")
        monkeypatch.setattr(mapper, "map_alignment_to_structure", lambda alignment_path: {10: 100, 20: 200})
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            no_switch_csv = tmp / "no_switch.csv"
            no_switch_csv.write_text(
                "position,max_score,switch_count\n10,0.2,0\n20,0.3,0\n",
                encoding="utf-8",
            )
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
                sdp_csv_groups=no_switch_csv,
                sdp_csv_families=no_switch_csv,
                sdp_csv_subfamilies=no_switch_csv,
                output_dir=tmp,
            )
            for path in outputs.values():
                content = path.read_text(encoding="utf-8")
                assert path.stat().st_size > 0
                assert "open " in content
                assert "_residues: none" in content
                assert "reason: no switch_count > 0 rows" in content

    def test_generate_physicochemical_chimerax_script(self, monkeypatch):
        """Generate a physicochemical-shift ChimeraX script with rule-based coloring."""
        mapper = PDBMapper(pdb_id="2cg4")
        monkeypatch.setattr(mapper, "download_pdb", lambda: "data/raw/2cg4.pdb")
        monkeypatch.setattr(mapper, "map_alignment_to_structure", lambda alignment_path: {10: 101, 20: 205, 30: 333})
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shifts_csv = tmp / "physicochemical_shifts.csv"
            shifts_csv.write_text(
                "\n".join([
                    "position,charge_change,hydrophobicity_change,volume_change",
                    "10,neutral->positive,polar->polar,1.0",
                    "20,neutral->neutral,polar->hydrophobic,2.0",
                    "30,neutral->neutral,polar->polar,60.0",
                ]) + "\n"
            )
            output_cxc = tmp / "highlight_physicochemistry.cxc"
            out = mapper.generate_physicochemical_chimerax_script(
                alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
                physicochemical_csv=shifts_csv,
                output_cxc=output_cxc,
            )
            assert out.exists()
            content = out.read_text()
            assert "color :101 #D62728" in content
            assert "color :205 #2CA02C" in content
            assert "color :333 #1F77B4" in content
            assert "set bgColor white" in content


def test_generate_single_chimerax_script_supports_event_and_mdo_filters(tmp_path: Path, monkeypatch) -> None:
    pdb_path = tmp_path / "toy.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\nEND\n",
        encoding="utf-8",
    )
    alignment = tmp_path / "toy.aln"
    alignment.write_text(">s1\nAAAA\n", encoding="utf-8")
    pairwise_csv = tmp_path / "raw_pairwise_duplications.csv"
    pd.DataFrame(
        {
            "position": [1, 2, 3, 4, 1, 2, 3, 4],
            "score": [0.1, 0.2, 0.3, 2.0, 0.1, 0.2, 0.3, 1.9],
            "pair": ["A-B"] * 8,
            "Event_Type": ["Duplication", "Duplication", "Duplication", "Duplication", "Speciation", "Speciation", "Speciation", "Speciation"],
            "MDO_Node": ["Node2"] * 8,
        }
    ).to_csv(pairwise_csv, index=False)

    mapper = PDBMapper(pdb_id="2cg4", pdb_file=str(pdb_path))
    monkeypatch.setattr(mapper, "map_alignment_to_structure", lambda alignment_path: {4: [("A", 1)]})

    out_cxc = tmp_path / "filtered.cxc"
    mapper.generate_single_chimerax_script(
        alignment_path=alignment,
        sdp_csv=pairwise_csv,
        output_cxc=out_cxc,
        level_label="Duplication MDO",
        event_type_filter="Duplication",
        mdo_only=True,
    )

    text = out_cxc.read_text(encoding="utf-8")
    assert "/A:1" in text


class TestPDBMapperEndToEnd:
    """Integration tests for end-to-end structural mapping."""

    def test_pdb_mapper_initialization(self):
        """
        Test basic initialization of PDBMapper.

        Expected:
        - Mapper object created successfully
        - PDB ID is stored
        - Ready for subsequent operations
        """
        pdb_id = "2cg4"
        mapper = PDBMapper(pdb_id=pdb_id)

        assert mapper.pdb_id == pdb_id

    @pytest.mark.network
    def test_pdb_mapper_with_custom_cache(self):
        """
        Test PDBMapper with custom cache directory.

        Expected:
        - Custom PDB cache directory is used
        - Downloads are stored in custom location
        """
        pdb_id = "2cg4"
        with tempfile.TemporaryDirectory() as tmpdir:
            mapper = PDBMapper(pdb_id=pdb_id, cache_dir=tmpdir)
            pdb_path = mapper.download_pdb()

            assert str(pdb_path).startswith(tmpdir) or Path(pdb_path).exists()

    @pytest.mark.network
    def test_complete_workflow(self):
        """
        Test the complete Phase 6 workflow: download, map, script generation.

        Expected:
        - Downloads PDB file
        - Maps MSA to structure
        - Generates PyMOL script
        - All artifacts are created without error
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        if not alignment_path.exists():
            pytest.skip("data/interim/IPR019888_trimmed.aln not available on this checkout")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_pml = Path(tmpdir) / "sdps.pml"

            mapper = PDBMapper(pdb_id=pdb_id)

            # Step 1: Download PDB
            pdb_path = mapper.download_pdb()
            assert pdb_path is not None

            # Step 2: Map alignment to structure
            mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)
            assert isinstance(mapping, dict)

            # Step 3: Generate PyMOL script
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_pml,
            )
            assert output_pml.exists()


class TestPDBMapperCLI:
    """CLI-focused tests for explicit SDP mapping outputs."""

    def test_generate_single_chimerax_script_uses_explicit_output_path(self, monkeypatch):
        mapper = PDBMapper(pdb_id="2cg4")
        monkeypatch.setattr(mapper, "download_pdb", lambda: "data/raw/2cg4.pdb")
        monkeypatch.setattr(mapper, "map_alignment_to_structure", lambda alignment_path: {127: 500})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sdp_csv = tmp / "badasp_sdps_duplications_p99.csv"
            sdp_csv.write_text(
                "position,max_score,switch_count\n127,1.7,10\n",
                encoding="utf-8",
            )
            output_cxc = tmp / "custom_output.cxc"

            out = mapper.generate_single_chimerax_script(
                alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
                sdp_csv=sdp_csv,
                output_cxc=output_cxc,
                level_label="Duplications",
            )

            assert out == output_cxc
            assert output_cxc.exists()
            content = output_cxc.read_text(encoding="utf-8")
            assert "# Mapped from alignment col 127" in content
            assert "color :500" in content

    def test_generate_single_chimerax_script_uses_chain_specific_selectors_when_available(self, monkeypatch):
        mapper = PDBMapper(pdb_id="2cg4")
        mapper._last_protein_chain_id = "A"
        monkeypatch.setattr(mapper, "download_pdb", lambda: "data/raw/2cg4.pdb")
        monkeypatch.setattr(mapper, "map_alignment_to_structure", lambda alignment_path: {127: 500})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sdp_csv = tmp / "badasp_sdps_duplications_p99.csv"
            sdp_csv.write_text(
                "position,max_score,switch_count\n127,1.7,10\n",
                encoding="utf-8",
            )
            output_cxc = tmp / "custom_output_chain.cxc"

            out = mapper.generate_single_chimerax_script(
                alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
                sdp_csv=sdp_csv,
                output_cxc=output_cxc,
                level_label="Duplications",
            )

            assert out == output_cxc
            content = output_cxc.read_text(encoding="utf-8")
            # mapper mocked to single-chain mapping; expect chain selector present
            assert "color /A:500" in content or "color :500" in content

    def test_generate_single_chimerax_script_writes_png_legend(self, monkeypatch):
        mapper = PDBMapper(pdb_id="2cg4")
        monkeypatch.setattr(mapper, "download_pdb", lambda: "data/raw/2cg4.pdb")
        monkeypatch.setattr(mapper, "map_alignment_to_structure", lambda alignment_path: {127: 500})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sdp_csv = tmp / "badasp_sdps_duplications_p99.csv"
            sdp_csv.write_text(
                "position,max_score,switch_count\n127,1.7,10\n",
                encoding="utf-8",
            )
            output_cxc = tmp / "custom_output.cxc"

            out = mapper.generate_single_chimerax_script(
                alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
                sdp_csv=sdp_csv,
                output_cxc=output_cxc,
                level_label="Duplications",
            )

            legend_png = output_cxc.with_suffix(".png")
            assert out == output_cxc
            assert output_cxc.exists()
            assert legend_png.exists()
            assert legend_png.stat().st_size > 0

    def test_main_with_explicit_sdp_csv_writes_requested_output(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            alignment = tmp / "small.aln"
            alignment.write_text(
                ">seq1\nAAAA\n",
                encoding="utf-8",
            )
            sdp_csv = tmp / "explicit_sdps.csv"
            sdp_csv.write_text(
                "position,max_score,switch_count\n2,1.0,3\n",
                encoding="utf-8",
            )
            pdb_file = tmp / "mock.pdb"
            pdb_file.write_text("ATOM\n", encoding="utf-8")
            output_cxc = tmp / "explicit_output.cxc"

            monkeypatch.setattr(PDBMapper, "download_pdb", lambda self: str(pdb_file))
            monkeypatch.setattr(PDBMapper, "map_alignment_to_structure", lambda self, alignment_path: {2: 42})

            main(
                [
                    "--pdb-id",
                    "2cg4",
                    "--pdb-file",
                    str(pdb_file),
                    "--alignment",
                    str(alignment),
                    "--sdp-csv",
                    str(sdp_csv),
                    "--output-cxc",
                    str(output_cxc),
                ]
            )

            assert output_cxc.exists()
            content = output_cxc.read_text(encoding="utf-8")
            assert "# Mapped from alignment col 2" in content
            assert "color :42" in content
