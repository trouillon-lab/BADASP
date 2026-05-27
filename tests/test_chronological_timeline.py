import pytest
from pathlib import Path
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade
from src.chronological_timeline import load_calibrations, ChronoCalibrator, parse_accession_taxids

def test_load_calibrations(tmp_path):
    config = tmp_path / "calib.json"
    config.write_text('{"131567": 3800.0, "2": 3200.0}', encoding="utf-8")
    calibs = load_calibrations(config)
    assert calibs[131567] == 3800.0
    assert calibs[2] == 3200.0

def test_parse_accession_taxids(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">sp|O59188|REG6_PYRHO lrpA OS=Pyrococcus OX=70601 GN=lrp\nMSGLK\n", encoding="utf-8")
    acc_to_taxid, taxid_to_desc = parse_accession_taxids(fasta)
    assert acc_to_taxid.get("O59188") == 70601
    assert 70601 in taxid_to_desc

def test_monotonicity_interpolation(tmp_path):
    # Construct a simple synthetic tree: (A:1.0, B:2.0)C:1.0;
    tree_file = tmp_path / "test.tree"
    tree_file.write_text("((A:1.0, B:2.0)C:1.0)D:1.0;\n", encoding="utf-8")
    
    acc_to_taxid = {"A": 70601, "B": 83332}
    taxid_to_desc = {70601: "OS=Pyrococcus OX=70601", 83332: "OS=Mycobacterium OX=83332"}
    
    calib_json = tmp_path / "calib.json"
    calib_json.write_text('{"131567": 3800.0}', encoding="utf-8")
    
    calibrator = ChronoCalibrator(
        tree_path=tree_file,
        acc_to_taxid=acc_to_taxid,
        taxid_to_desc=taxid_to_desc,
        calibration_config=calib_json
    )
    
    # Manually assign mock split nodes
    split_nodes = {}
    node_ages = calibrator.calibrate_chronogram(split_nodes)
    
    # Check that root age matches default 3800 Mya, tips are 0 Mya, and parent > child + epsilon
    tree = Phylo.read(str(tree_file), "newick")
    clades_by_name = {c.name: c for c in tree.find_clades() if c.name}
    
    assert node_ages[tree.root.name] == 3800.0
    assert node_ages["A"] == 0.0
    assert node_ages["B"] == 0.0
    
    # Check parent > child + epsilon monotonicity
    # D is parent of C, C is parent of A and B
    assert node_ages[tree.root.name] > node_ages["C"] + 0.1
    assert node_ages["C"] > node_ages["A"] + 0.1
    assert node_ages["C"] > node_ages["B"] + 0.1
