import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch
from ete3 import Tree as EteTree

from scripts.plot_full_classification_tree import main

@pytest.fixture
def mock_trees_dir():
    """Create a temporary directory with mock ASR and AleRax trees."""
    temp_dir = tempfile.mkdtemp()
    
    # 1. Create a mock ASR tree
    # Node1 is root, Node2 and Node3 are children
    asr_tree = EteTree("((SeqA:0.1,SeqB:0.1)Node2:0.2,(SeqC:0.15,SeqD:0.15)Node3:0.25)Node1:0.0;", format=1)
    asr_tree_path = Path(temp_dir) / "mock_asr.treefile"
    asr_tree.write(outfile=str(asr_tree_path), format=1)
    
    # 2. Create a mock AleRax tree with Ev annotation on nodes
    # Node1 is root (S), Node2 is D, Node3 is T
    alerax_tree = EteTree("((SeqA:0.1,SeqB:0.1)Node2:0.2,(SeqC:0.15,SeqD:0.15)Node3:0.25)Node1:0.0;", format=1)
    for node in alerax_tree.traverse():
        if node.name == "Node2":
            node.add_features(Ev="D")
        elif node.name == "Node3":
            node.add_features(Ev="T")
        elif node.name == "Node1":
            node.add_features(Ev="S")
            
    alerax_tree_path = Path(temp_dir) / "mock_alerax.nwk"
    alerax_tree.write(outfile=str(alerax_tree_path), format=1)
    
    yield Path(temp_dir), asr_tree_path, alerax_tree_path
    
    shutil.rmtree(temp_dir)

def test_plot_full_classification_tree_main(mock_trees_dir):
    temp_path, asr_tree_path, alerax_tree_path = mock_trees_dir
    
    outdir = temp_path / "plots"
    
    test_args = [
        "plot_full_classification_tree.py",
        "--tree", str(asr_tree_path),
        "--alerax-tree", str(alerax_tree_path),
        "--outdir", str(outdir)
    ]
    
    with patch("sys.argv", test_args):
        main()
        
    # Check that outputs were generated
    assert (outdir / "tree_full_classification.svg").exists()
    assert (outdir / "tree_full_classification.png").exists()
