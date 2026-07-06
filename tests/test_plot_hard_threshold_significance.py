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
from scripts.plot_hard_threshold_significance import (
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
    
    # 2. Create a mock raw scores CSV
    scores_path = Path(temp_dir) / "mock_raw_scores.csv"
    rows = []
    np.random.seed(42)
    
    # Generate 10 comparisons per position to have enough statistics
    for pos in range(1, 170):
        for comp in range(10):
            score_left = np.random.normal(0.5 if pos == 50 else 0.0, 0.5)
            score_right = np.random.normal(0.4 if pos == 50 else -0.2, 0.4)
            rows.append({
                "node": f"Node_{pos}_{comp}",
                "position": pos,
                "badasp_score_left": score_left,
                "badasp_score_right": score_right,
                "distance_from_root": 3.0
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(scores_path, index=False)
    
    # 3. Create a mock positional switches CSV
    # Need column switches_h1.7
    pos_csv_path = Path(temp_dir) / "positional_switches_comparison.csv"
    pos_rows = []
    for pos in range(1, 170):
        # Position 50 has 8 switches, others have 0 or 1
        switches = 8 if pos == 50 else (1 if pos % 10 == 0 else 0)
        pos_rows.append({
            "position": pos,
            "switches_h1.7": switches
        })
    pd.DataFrame(pos_rows).to_csv(pos_csv_path, index=False)
    
    yield Path(temp_dir), alignment_path, scores_path, pos_csv_path
    
    shutil.rmtree(temp_dir)


def test_plot_hard_threshold_significance_main(mock_data_dir):
    temp_path, alignment_path, scores_path, pos_csv_path = mock_data_dir
    
    min_occ = 0.4
    occ_pct = int(min_occ * 100)
    
    # Patch command line args
    test_args = [
        "plot_hard_threshold_significance.py",
        "--scores", str(scores_path),
        "--alignment", str(alignment_path),
        "--positional", str(pos_csv_path),
        "--min-occupancy", str(min_occ)
    ]
    
    mock_out_dir = temp_path / f"occupancy_{occ_pct}"
    
    with patch("sys.argv", test_args):
        with patch("scripts.plot_hard_threshold_significance.Path") as mock_path_cls:
            def side_effect(arg):
                if str(arg).startswith("results/badasp_scoring"):
                    rel_path = Path(arg).relative_to("results/badasp_scoring")
                    return temp_path / rel_path
                return Path(arg)
                
            mock_path_cls.side_effect = side_effect
            main()
            
    # Check that outputs were generated in the redirected directory
    redirected_out_dir = temp_path / "threshold_comparison" / f"occupancy_{occ_pct}"
    plot_svg = redirected_out_dir / "hard_threshold_1.7_significance.svg"
    plot_png = redirected_out_dir / "hard_threshold_1.7_significance.png"
    
    assert plot_svg.exists()
    assert plot_png.exists()
