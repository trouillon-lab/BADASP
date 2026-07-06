import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# We will import these from our script
from scripts.plot_decoupled_event_switches_clade_adjusted import (
    main,
    bin_clade_sizes,
    calculate_bin_thresholds,
    identify_switches
)

@pytest.fixture
def mock_data_dir():
    """Create temporary directory with mock alignment and raw scores."""
    temp_dir = tempfile.mkdtemp()
    
    # 1. Create a mock alignment with 169 positions and 4 sequences
    alignment_path = Path(temp_dir) / "mock_trimmed.aln"
    
    seqs = []
    for seq_idx in range(4):
        seq_chars = []
        for pos in range(1, 170):
            if pos <= 9:
                char = "-" if seq_idx in {0, 1} else "A"
            elif pos >= 161:
                char = "-" if seq_idx in {0, 1, 2} else "T"
            else:
                char = "L"
            seq_chars.append(char)
        
        seqs.append(
            SeqRecord(
                Seq("".join(seq_chars)),
                id=f"Seq_{seq_idx}",
                description=""
            )
        )
    SeqIO.write(seqs, alignment_path, "fasta")
    
    # 2. Create a mock raw scores CSV with event_type and clade size columns
    scores_path = Path(temp_dir) / "mock_raw_scores.csv"
    rows = []
    np.random.seed(42)
    
    events = ["Duplication", "Speciation", "Transfer"]
    
    # Generate comparisons
    for event_type in events:
        for pos in range(1, 170):
            for comp in range(5):
                # Ensure some variation in clade sizes (from 5 to 500)
                clade_size_left = int(np.random.choice([5, 10, 20, 50, 100, 200, 500]))
                clade_size_right = int(np.random.choice([5, 10, 20, 50, 100, 200, 500]))
                
                # Make small clades have higher variance/scores
                var_left = 1.0 / np.log(clade_size_left)
                var_right = 1.0 / np.log(clade_size_right)
                
                score_left = np.random.normal(0.5, var_left)
                score_right = np.random.normal(0.4, var_right)
                
                rows.append({
                    "node_name": f"Node_{event_type}_{pos}_{comp}",
                    "event_type": event_type,
                    "position": pos,
                    "badasp_score_left": score_left,
                    "badasp_score_right": score_right,
                    "distance_from_root": 3.0,
                    "clade_size_left": clade_size_left,
                    "clade_size_right": clade_size_right,
                })
            
    df = pd.DataFrame(rows)
    df.to_csv(scores_path, index=False)
    
    yield Path(temp_dir), alignment_path, scores_path
    
    shutil.rmtree(temp_dir)


def test_bin_clade_sizes():
    """Test that clade sizes are correctly binned under both methods."""
    clade_sizes = pd.Series([5, 8, 12, 20, 50, 100, 500, 1000])
    
    # Quantile binning (deciles)
    bins_q, categories_q = bin_clade_sizes(clade_sizes, method="quantile", num_bins=4)
    assert len(bins_q) == len(clade_sizes)
    # Check that categories are ordered
    assert bins_q.dtype.name == "category"
    
    # Log-spaced binning
    bins_l, categories_l = bin_clade_sizes(clade_sizes, method="log-spaced")
    assert len(bins_l) == len(clade_sizes)
    assert bins_l.dtype.name == "category"


def test_calculate_bin_thresholds():
    """Test calculation of 95th percentile thresholds per bin/event."""
    df = pd.DataFrame({
        "event_type": ["Duplication"] * 10 + ["Speciation"] * 10,
        "score": list(range(10)) + list(range(10, 20)),
        "bin": ["Bin1"] * 5 + ["Bin2"] * 5 + ["Bin1"] * 5 + ["Bin2"] * 5
    })
    # Set custom categories to ensure order works
    df["bin"] = pd.Categorical(df["bin"], categories=["Bin1", "Bin2"], ordered=True)
    
    thresholds = calculate_bin_thresholds(df, "score", "bin", percentile=95)
    
    # Check keys
    assert ("Duplication", "Bin1") in thresholds
    assert ("Speciation", "Bin2") in thresholds
    assert ("overall", "Bin1") in thresholds
    
    # Duplication Bin1 scores: 0,1,2,3,4. 95th% of 5 values is 3.8
    assert thresholds[("Duplication", "Bin1")] == pytest.approx(3.8)
    # Speciation Bin2 scores: 15,16,17,18,19. 95th% is 18.8
    assert thresholds[("Speciation", "Bin2")] == pytest.approx(18.8)


def test_identify_switches():
    """Test switch identification logic based on adaptive thresholds."""
    df = pd.DataFrame({
        "event_type": ["Duplication", "Duplication", "Speciation"],
        "badasp_score_left": [1.5, 1.2, 0.9],
        "badasp_score_right": [0.8, 1.6, 1.1],
        "bin_left": ["Bin1", "Bin2", "Bin1"],
        "bin_right": ["Bin1", "Bin1", "Bin2"]
    })
    
    thresholds = {
        ("Duplication", "Bin1"): 1.4,
        ("Duplication", "Bin2"): 1.7,
        ("Speciation", "Bin1"): 1.0,
        ("Speciation", "Bin2"): 1.2,
    }
    
    switches = identify_switches(df, thresholds, event_specific=True)
    # Row 0: left_score=1.5 >= left_thresh=1.4 -> True
    # Row 1: left_score=1.2 < left_thresh=1.7 and right_score=1.6 >= right_thresh=1.4 -> True
    # Row 2: left_score=0.9 < left_thresh=1.0 and right_score=1.1 < right_thresh=1.2 -> False
    assert list(switches) == [True, True, False]
def test_plot_decoupled_event_switches_clade_adjusted_main(mock_data_dir):
    """Test main execution of clade-adjusted plotting script with --min-clade-size parameter."""
    temp_path, alignment_path, scores_path = mock_data_dir
    
    min_occ = 0.4
    occ_pct = int(min_occ * 100)
    min_clade = 6
    
    # Patch command line args
    test_args = [
        "plot_decoupled_event_switches_clade_adjusted.py",
        "--scores", str(scores_path),
        "--alignment", str(alignment_path),
        "--min-occupancy", str(min_occ),
        "--bin-method", "quantile",
        "--num-bins", "4",
        "--percentile", "99",
        "--min-clade-size", str(min_clade)
    ]
    
    with patch("sys.argv", test_args):
        # Redirect outputs to temp_path
        with patch("scripts.plot_decoupled_event_switches_clade_adjusted.Path") as mock_path_cls:
            def side_effect(arg):
                if str(arg).startswith("results/badasp_scoring"):
                    rel_path = Path(arg).relative_to("results/badasp_scoring")
                    return temp_path / rel_path
                return Path(arg)
                
            mock_path_cls.side_effect = side_effect
            main()
            
    # Check that outputs were generated in the redirected directory
    redirected_out_dir = temp_path / "clade_size_adjusted" / f"min_clade_{min_clade}" / f"occupancy_{occ_pct}"
    stats_csv = redirected_out_dir / "event_decoupled_stats_clade_adjusted.csv"
    pos_csv = redirected_out_dir / "event_positional_switches_clade_adjusted.csv"
    domain_csv = redirected_out_dir / "event_domain_densities_clade_adjusted.csv"
    
    plot_rel = redirected_out_dir / "clade_adjusted_threshold_relationship.svg"
    plot_spec_svg = redirected_out_dir / "decoupled_event_switches_clade_adjusted_event_specific.svg"
    plot_over_svg = redirected_out_dir / "decoupled_event_switches_clade_adjusted_overall.svg"
    
    # New plots
    plot_agnostic = redirected_out_dir / "event_agnostic_switches_clade_adjusted.svg"
    plot_props = redirected_out_dir / "domain_switch_proportions_by_threshold.svg"
    
    assert stats_csv.exists()
    assert pos_csv.exists()
    assert domain_csv.exists()
    assert plot_rel.exists()
    assert plot_spec_svg.exists()
    assert plot_over_svg.exists()
    assert plot_agnostic.exists()
    assert plot_props.exists()
