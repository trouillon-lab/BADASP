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
from scripts.plot_decoupled_event_switches import (
    main
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
    
    # 2. Create a mock raw scores CSV with event_type column
    scores_path = Path(temp_dir) / "mock_raw_scores.csv"
    rows = []
    np.random.seed(42)
    
    # Generate comparisons for three event types
    # Duplication, Speciation, Transfer
    events = ["Duplication", "Speciation", "Transfer"]
    
    for event_idx, event_type in enumerate(events):
        # 5 comparisons per event type per position
        for pos in range(1, 170):
            for comp in range(5):
                mean_shift = 0.4 if (pos == 50 and event_type == "Duplication") else 0.0
                score_left = np.random.normal(0.0 + mean_shift, 0.5)
                score_right = np.random.normal(-0.2 + mean_shift, 0.4)
                rows.append({
                    "node_name": f"Node_{event_type}_{pos}_{comp}",
                    "event_type": event_type,
                    "position": pos,
                    "badasp_score_left": score_left,
                    "badasp_score_right": score_right,
                    "distance_from_root": 3.0
                })
            
    df = pd.DataFrame(rows)
    df.to_csv(scores_path, index=False)
    
    yield Path(temp_dir), alignment_path, scores_path
    
    shutil.rmtree(temp_dir)


def test_plot_decoupled_event_switches_main(mock_data_dir):
    temp_path, alignment_path, scores_path = mock_data_dir
    
    min_occ = 0.4
    occ_pct = int(min_occ * 100)
    
    # Patch command line args
    test_args = [
        "plot_decoupled_event_switches.py",
        "--scores", str(scores_path),
        "--alignment", str(alignment_path),
        "--min-occupancy", str(min_occ)
    ]
    
    with patch("sys.argv", test_args):
        with patch("scripts.plot_decoupled_event_switches.Path") as mock_path_cls:
            def side_effect(arg):
                if str(arg).startswith("results/badasp_scoring"):
                    rel_path = Path(arg).relative_to("results/badasp_scoring")
                    return temp_path / rel_path
                return Path(arg)
                
            mock_path_cls.side_effect = side_effect
            main()
            
    # Check that outputs were generated in the redirected directory
    redirected_out_dir = temp_path / "event_decoupling" / f"occupancy_{occ_pct}"
    stats_csv = redirected_out_dir / "event_decoupled_stats.csv"
    pos_csv = redirected_out_dir / "event_positional_switches.csv"
    domain_csv = redirected_out_dir / "event_domain_densities.csv"
    plot_svg = redirected_out_dir / "decoupled_event_switches_comparison.svg"
    plot_png = redirected_out_dir / "decoupled_event_switches_comparison.png"
    plot_h17_svg = redirected_out_dir / "decoupled_event_switches_hard1.7_comparison.svg"
    plot_h17_png = redirected_out_dir / "decoupled_event_switches_hard1.7_comparison.png"
    plot_h19_svg = redirected_out_dir / "decoupled_event_switches_hard1.9_comparison.svg"
    plot_h19_png = redirected_out_dir / "decoupled_event_switches_hard1.9_comparison.png"
    
    assert stats_csv.exists()
    assert pos_csv.exists()
    assert domain_csv.exists()
    assert plot_svg.exists()
    assert plot_png.exists()
    assert plot_h17_svg.exists()
    assert plot_h17_png.exists()
    assert plot_h19_svg.exists()
    assert plot_h19_png.exists()
    
    # Read stats CSV and verify structure
    df_stats = pd.read_csv(stats_csv)
    assert len(df_stats) == 3  # Duplication, Speciation, Transfer
    assert set(df_stats["event_type"]) == {"Duplication", "Speciation", "Transfer"}
    assert "total_comparisons" in df_stats.columns
    assert "p97_threshold" in df_stats.columns
    assert "p97_switches" in df_stats.columns
    assert "hard1.7_switches" in df_stats.columns
