import sys
import tarfile
from pathlib import Path
import pytest


def _make_alerax_fixtures(root: Path):
    rec_check_dir = root / "data/rec_check/alerax"
    rec_check_dir.mkdir(parents=True, exist_ok=True)
    (rec_check_dir / "IPR019888_distribution.nwk").write_text("(A:0.1,B:0.1);\n", encoding="utf-8")
    (rec_check_dir / "IPR019888.treerecs_mapping.link").write_text("A\t1\nB\t2\n", encoding="utf-8")
    (rec_check_dir / "IPR019888_species_tree.nwk").write_text("(1:0.1,2:0.1);\n", encoding="utf-8")


def test_package_rec_check_creates_valid_bundle(tmp_path: Path):
    """package_rec_check() creates a tar.gz bundle with expected members."""
    _make_alerax_fixtures(tmp_path)

    # Import the script's main function, passing tmp_path as the root
    from scripts.package_rec_check_for_euler import package_rec_check
    tar_path = package_rec_check(root=tmp_path)

    assert tar_path.exists(), "tar.gz bundle was not created"
    assert tar_path.stat().st_size > 0

    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        assert "rec_check_euler/IPR019888_distribution.nwk" in names
        assert "rec_check_euler/IPR019888.treerecs_mapping.link" in names
        assert "rec_check_euler/IPR019888_species_tree.nwk" in names
        assert "rec_check_euler/IPR019888.families.txt" in names
        assert "rec_check_euler/run_alerax_rec_check.sh" in names


def test_package_rec_check_families_txt_content(tmp_path: Path):
    """families.txt inside the bundle has correct AleRax format."""
    _make_alerax_fixtures(tmp_path)

    from scripts.package_rec_check_for_euler import package_rec_check
    tar_path = package_rec_check(root=tmp_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.extractfile("rec_check_euler/IPR019888.families.txt")
        content = member.read().decode("utf-8")

    assert "[FAMILIES]" in content
    assert "gene_tree = IPR019888_distribution.nwk" in content
    assert "mapping = IPR019888.treerecs_mapping.link" in content


def test_package_rec_check_sbatch_script_content(tmp_path: Path):
    """The sbatch run script inside the bundle contains the alerax invocation."""
    _make_alerax_fixtures(tmp_path)

    from scripts.package_rec_check_for_euler import package_rec_check
    tar_path = package_rec_check(root=tmp_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.extractfile("rec_check_euler/run_alerax_rec_check.sh")
        content = member.read().decode("utf-8")

    assert "#!/bin/bash" in content
    assert "#SBATCH" in content
    assert "alerax" in content
    assert "IPR019888.families.txt" in content


def test_package_rec_check_euler_bundle_live():
    """Integration smoke-test: runs against real data/rec_check/alerax if present."""
    real_alerax = Path("data/rec_check/alerax")
    if not real_alerax.exists():
        pytest.skip("data/rec_check/alerax not available on this checkout")

    from scripts.package_rec_check_for_euler import package_rec_check
    tar_path = package_rec_check()

    assert tar_path.exists()
    assert tar_path.stat().st_size > 0
