import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch
from ete3 import Tree as EteTree

from scripts.plot_taxonomic_tree import main

@pytest.fixture
def mock_taxonomy_data():
    """Create a temporary directory with mock tree and mapping."""
    temp_dir = tempfile.mkdtemp()
    
    # 1. Create a mock ASR tree
    asr_tree = EteTree("((SeqA:0.1,SeqB:0.1)Node2:0.2,(SeqC:0.15,SeqD:0.15)Node3:0.25)Node1:0.0;", format=1)
    asr_tree_path = Path(temp_dir) / "mock_asr.treefile"
    asr_tree.write(outfile=str(asr_tree_path), format=1)
    
    # 2. Create a mock mapping file
    mapping_path = Path(temp_dir) / "mock_mapping.link"
    with open(mapping_path, "w") as f:
        # Map SeqA to E. coli (562), SeqB to B. subtilis (1390), SeqC and SeqD to Archaea/Other
        f.write("SeqA 562\n")
        f.write("SeqB 1390\n")
        f.write("SeqC 2157\n")
        f.write("SeqD 2157\n")
        
    yield Path(temp_dir), asr_tree_path, mapping_path
    
    shutil.rmtree(temp_dir)

def test_plot_taxonomic_tree_main(mock_taxonomy_data):
    temp_path, asr_tree_path, mapping_path = mock_taxonomy_data
    
    outdir = temp_path / "plots"
    
    test_args = [
        "plot_taxonomic_tree.py",
        "--tree", str(asr_tree_path),
        "--mapping", str(mapping_path),
        "--outdir", str(outdir)
    ]
    
    # Patch NCBITaxa to avoid external API calls during testing
    with patch("scripts.plot_taxonomic_tree.NCBITaxa") as mock_ncbi_cls:
        # Set up mock taxonomy resolver cache instead of querying database
        with patch("scripts.plot_taxonomic_tree.TaxonomyResolver") as mock_resolver_cls:
            mock_resolver = mock_resolver_cls.return_value
            mock_resolver.cache = {
                "562": {"superkingdom": "Bacteria", "phylum": "Pseudomonadota"},
                "1390": {"superkingdom": "Bacteria", "phylum": "Bacillota"},
                "2157": {"superkingdom": "Archaea", "phylum": "Methanobacteriota"}
            }
            def get_domain(taxid):
                return mock_resolver.cache.get(taxid, {}).get("superkingdom", "Unknown")
            def get_phylum(taxid):
                return mock_resolver.cache.get(taxid, {}).get("phylum", "Unknown")
            mock_resolver.get_domain.side_effect = get_domain
            mock_resolver.get_phylum.side_effect = get_phylum
            
            with patch("sys.argv", test_args):
                main()
        
    # Check that outputs were generated
    assert (outdir / "tree_taxonomy_domain.svg").exists()
    assert (outdir / "tree_taxonomy_domain.png").exists()
    assert (outdir / "tree_taxonomy_phylum.svg").exists()
    assert (outdir / "tree_taxonomy_phylum.png").exists()
