from pathlib import Path

from Bio import Phylo
import pytest

from src.tree_rooting import root_tree


def test_root_tree_midpoint_writes_rooted_tree(tmp_path: Path) -> None:
    input_tree = tmp_path / "input.tree"
    output_tree = tmp_path / "rooted.tree"
    input_tree.write_text("((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.4);\n", encoding="utf-8")

    rooted_path = root_tree(input_tree=input_tree, output_tree=output_tree, method="midpoint")

    assert rooted_path == output_tree
    assert output_tree.exists()
    parsed = Phylo.read(str(output_tree), "newick")
    assert parsed.root is not None


def test_root_tree_mad_invokes_external_tool_and_uses_generated_output(monkeypatch, tmp_path: Path) -> None:
    input_tree = tmp_path / "input.tree"
    output_tree = tmp_path / "mad_rooted.tree"
    mad_generated = tmp_path / "input.tree.rooted"
    input_tree.write_text("((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.4);\n", encoding="utf-8")

    calls = []

    def _mock_run(cmd, check, capture_output=False, text=False, **kwargs):
        calls.append(cmd)
        assert check is True
        mad_generated.write_text("[&R]((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.4);\n", encoding="utf-8")
        return None

    mad_exec = tmp_path / "mad.py"
    mad_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr("subprocess.run", _mock_run)
    monkeypatch.setattr("src.tree_rooting.CANONICAL_MAD_EXECUTABLE", mad_exec)

    rooted_path = root_tree(
        input_tree=input_tree,
        output_tree=output_tree,
        method="mad",
    )

    assert calls
    assert calls[0][0] == str(mad_exec)
    assert calls[0][1] == str(input_tree)
    assert rooted_path == output_tree
    assert output_tree.exists()


def test_root_tree_mad_raises_if_no_rooted_output_found(monkeypatch, tmp_path: Path) -> None:
    input_tree = tmp_path / "input.tree"
    output_tree = tmp_path / "mad_rooted.tree"
    input_tree.write_text("((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.4);\n", encoding="utf-8")

    mad_exec = tmp_path / "mad.py"
    mad_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setattr("src.tree_rooting.CANONICAL_MAD_EXECUTABLE", mad_exec)

    def _mock_run(cmd, check, capture_output=False, text=False, **kwargs):
        assert check is True
        return None

    monkeypatch.setattr("subprocess.run", _mock_run)

    with pytest.raises(FileNotFoundError, match="MAD rooted tree output not found"):
        root_tree(
            input_tree=input_tree,
            output_tree=output_tree,
            method="mad",
        )


def test_root_tree_mad_raises_on_negative_branch_lengths_and_empty_output(
    monkeypatch, tmp_path: Path
) -> None:
    input_tree = tmp_path / "input.tree"
    output_tree = tmp_path / "mad_rooted.tree"
    empty_rooted = tmp_path / "input.tree.rooted"
    input_tree.write_text("((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.4);\n", encoding="utf-8")

    mad_exec = tmp_path / "mad.py"
    mad_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setattr("src.tree_rooting.CANONICAL_MAD_EXECUTABLE", mad_exec)

    class _MockCompletedProcess:
        stdout = (
            "MAD phylogenetic rooting\n"
            "<<< Error analyzing file 'input.tree':\n"
            "<<<         Cowardly refusing to root trees with negative branch lengths.\n"
        )
        stderr = ""

    def _mock_run(cmd, check, capture_output=False, text=False, **kwargs):
        assert check is True
        empty_rooted.write_text("", encoding="utf-8")
        return _MockCompletedProcess()

    monkeypatch.setattr("subprocess.run", _mock_run)

    with pytest.raises(RuntimeError, match="Cowardly refusing"):
        root_tree(
            input_tree=input_tree,
            output_tree=output_tree,
            method="mad",
        )


def test_root_tree_mad_falls_back_to_midpoint_when_executable_missing(monkeypatch, tmp_path: Path) -> None:
    input_tree = tmp_path / "input.tree"
    output_tree = tmp_path / "fallback_rooted.tree"
    input_tree.write_text("((A:0.1,B:0.2):0.3,(C:0.2,D:0.1):0.4);\n", encoding="utf-8")

    monkeypatch.setattr("src.tree_rooting.CANONICAL_MAD_EXECUTABLE", tmp_path / "missing_mad.py")

    with pytest.warns(UserWarning, match="falling back to midpoint rooting"):
        rooted_path = root_tree(
            input_tree=input_tree,
            output_tree=output_tree,
            method="mad",
        )

    assert rooted_path == output_tree
    assert output_tree.exists()
    parsed = Phylo.read(str(output_tree), "newick")
    assert parsed.root is not None