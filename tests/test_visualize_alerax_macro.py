import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import shutil
from unittest.mock import MagicMock, patch

# Note: scripts/visualize_alerax_macro.py does not exist yet. We import the expected functions/classes.
from scripts.visualize_alerax_macro import (
    TaxonomyResolver,
    parse_species_tree,
    parse_treerecs_mapping,
    parse_event_counts_with_errors,
    parse_model_parameters,
    parse_transfers_with_taxonomy,
    main
)

@pytest.fixture
def mock_taxonomy_db():
    """Mock ete3.NCBITaxa instance for unit testing without local downloads."""
    mock_ncbi = MagicMock()
    
    # Mock get_lineage, get_taxid_translator, and get_rank
    # Let's say:
    # 70601 (species) -> lineage: [1, 2, 70601]
    # 1 -> cellular organisms
    # 2 -> Bacteria (Phylum: Proteobacteria, Class: Alphaproteobacteria)
    def mock_get_lineage(taxid):
        return [1, 2, int(taxid)]
        
    def mock_get_rank(lineage):
        return {1: "superkingdom", 2: "phylum", lineage[-1]: "class"}
        
    def mock_get_taxid_translator(lineage):
        return {
            1: "Cellular organisms",
            2: "Proteobacteria",
            lineage[-1]: f"Class_{lineage[-1]}"
        }
        
    mock_ncbi.get_lineage.side_effect = mock_get_lineage
    mock_ncbi.get_rank.side_effect = mock_get_rank
    mock_ncbi.get_taxid_translator.side_effect = mock_get_taxid_translator
    
    return mock_ncbi

@pytest.fixture
def mock_alerax_macro_dir():
    """Create temporary directory structure mimicking AleRax outputs for macro visualizations."""
    temp_dir = tempfile.mkdtemp()
    
    # Create reconciliations/all
    all_dir = os.path.join(temp_dir, "reconciliations", "all")
    os.makedirs(all_dir, exist_ok=True)
    
    # Write 3 mock sample event count files
    for i in range(3):
        content = f"S:{9000 + i}\nSL:{19000 - i}\nD:{1300 + i}\nDL:0\nT:{10000 + i*10}\nTL:{7000 - i}\nL:0\nLeaf:21641\n"
        with open(os.path.join(all_dir, f"IPR019888_eventCounts_{i}.txt"), "w") as f:
            f.write(content)
            
    # Create model_parameters
    param_dir = os.path.join(temp_dir, "model_parameters")
    os.makedirs(param_dir, exist_ok=True)
    param_content = "node D L T\n2565781 0.0664735 0.823578 0.533322\n3101447 0.0664735 0.823578 0.533322\n"
    with open(os.path.join(param_dir, "model_parameters.txt"), "w") as f:
        f.write(param_content)
        
    # Create species_trees
    species_dir = os.path.join(temp_dir, "species_trees")
    os.makedirs(species_dir, exist_ok=True)
    # 3 species TaxIDs: 70601, 186497, 208964
    species_content = "((70601:0.1,186497:0.15)Node1:0.2,208964:0.3)Root;\n"
    with open(os.path.join(species_dir, "starting_species_tree.newick"), "w") as f:
        f.write(species_content)
        
    # Create reconciliations/summaries
    summary_dir = os.path.join(temp_dir, "reconciliations", "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    transfer_content = "70601 186497 2.81\n186497 208964 1.64\n208964 70601 1.5\n"
    with open(os.path.join(summary_dir, "IPR019888_meanTransfers.txt"), "w") as f:
        f.write(transfer_content)
        
    # Create treerecs mapping file
    # We put it in data/interim/alerax/IPR019888.treerecs_mapping.link inside temp_dir
    map_dir = os.path.join(temp_dir, "treerecs_mapping")
    os.makedirs(map_dir, exist_ok=True)
    map_content = "gene1\t70601\ngene2\t186497\ngene3\t208964\ngene4\t70601\n"
    with open(os.path.join(map_dir, "treerecs_mapping.link"), "w") as f:
        f.write(map_content)
        
    yield temp_dir
    
    shutil.rmtree(temp_dir)

def test_parse_species_tree(mock_alerax_macro_dir):
    taxids = parse_species_tree(mock_alerax_macro_dir)
    assert set(taxids) == {"70601", "186497", "208964"}

def test_parse_treerecs_mapping(mock_alerax_macro_dir):
    mapping = parse_treerecs_mapping(mock_alerax_macro_dir)
    assert len(mapping) == 4
    assert mapping["gene1"] == "70601"
    assert mapping["gene3"] == "208964"

def test_taxonomy_resolver(mock_alerax_macro_dir, mock_taxonomy_db):
    # Mock NCBITaxa download/init
    with patch("scripts.visualize_alerax_macro.NCBITaxa", return_value=mock_taxonomy_db):
        cache_path = os.path.join(mock_alerax_macro_dir, "ncbi_taxonomy_map.csv")
        resolver = TaxonomyResolver(cache_path=cache_path)
        
        taxids = ["70601", "186497"]
        mapping_df = resolver.resolve_taxids(taxids)
        
        assert isinstance(mapping_df, pd.DataFrame)
        assert len(mapping_df) == 2
        assert "phylum" in mapping_df.columns
        assert "class" in mapping_df.columns
        assert mapping_df.loc["70601", "phylum"] == "Proteobacteria"
        
        # Verify it cached to file
        assert os.path.exists(cache_path)
        
        # Verify reloading reads from cache instead of mock_ncbi (which we can patch to fail)
        with patch("scripts.visualize_alerax_macro.NCBITaxa", side_effect=Exception("DB Error")):
            resolver_cached = TaxonomyResolver(cache_path=cache_path)
            cached_df = resolver_cached.resolve_taxids(["70601"])
            assert cached_df.loc["70601", "phylum"] == "Proteobacteria"

def test_parse_event_counts_with_errors(mock_alerax_macro_dir):
    summary_df = parse_event_counts_with_errors(mock_alerax_macro_dir)
    assert isinstance(summary_df, pd.DataFrame)
    assert set(summary_df.columns) == {"mean", "std"}
    assert summary_df.loc["S", "mean"] == pytest.approx(9001.0)
    assert summary_df.loc["S", "std"] > 0
    assert summary_df.loc["D", "mean"] == pytest.approx(1301.0)

def test_parse_transfers_with_taxonomy(mock_alerax_macro_dir, mock_taxonomy_db):
    with patch("scripts.visualize_alerax_macro.NCBITaxa", return_value=mock_taxonomy_db):
        cache_path = os.path.join(mock_alerax_macro_dir, "ncbi_taxonomy_map.csv")
        resolver = TaxonomyResolver(cache_path=cache_path)
        taxids = ["70601", "186497", "208964"]
        resolver.resolve_taxids(taxids)
        
        transfers_df = parse_transfers_with_taxonomy(mock_alerax_macro_dir, resolver)
        assert isinstance(transfers_df, pd.DataFrame)
        assert len(transfers_df) == 3
        assert set(transfers_df.columns) == {"donor_phylum", "recipient_phylum", "frequency"}
        # Both mapped to Proteobacteria phylum
        assert transfers_df.loc[0, "donor_phylum"] == "Proteobacteria"
        assert transfers_df.loc[0, "recipient_phylum"] == "Proteobacteria"

def test_main_macro_execution(mock_alerax_macro_dir, mock_taxonomy_db):
    output_dir = os.path.join(mock_alerax_macro_dir, "plots")
    cache_path = os.path.join(mock_alerax_macro_dir, "ncbi_taxonomy_map.csv")
    
    test_args = [
        "visualize_alerax_macro.py",
        "--input_dir", mock_alerax_macro_dir,
        "--taxonomy_map", cache_path,
        "--output_dir", output_dir,
        "--format", "svg",
        "--top_hgt", "10"
    ]
    
    with patch("scripts.visualize_alerax_macro.NCBITaxa", return_value=mock_taxonomy_db):
        with patch("sys.argv", test_args):
            main()
            
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "global_event_profile_and_rates.svg"))
    assert os.path.exists(os.path.join(output_dir, "taxonomic_distribution.svg"))
    assert os.path.exists(os.path.join(output_dir, "macro_hgt_highway_network.svg"))
