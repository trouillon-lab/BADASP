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
from scripts.plot_sequence_bins_distribution import (
    calculate_msa_occupancies,
    main
)

@pytest.fixture
def mock_data_dir():
    """Create temporary directory with mock alignment and raw scores."""
    temp_dir = tempfile.mkdtemp()
    
    # 1. Create a mock alignment with 169 positions and 4 sequences
    # Sequence length must be exactly 169 to match 10-bin boundaries
    alignment_path = Path(temp_dir) / "mock_trimmed.aln"
    
    # Make some positions have gaps to test occupancy
    # Pos 1-9: 50% occupancy (2 gaps out of 4)
    # Pos 10-160: 100% occupancy (0 gaps)
    # Pos 161-169: 25% occupancy (3 gaps out of 4)
    seqs = []
    for seq_idx in range(4):
        seq_chars = []
        for pos in range(1, 170):
            if pos <= 9:
                # 50% gaps overall
                char = "-" if seq_idx in {0, 1} else "A"
            elif pos >= 161:
                # 75% gaps overall
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
    # Need scores for all 169 positions
    scores_path = Path(temp_dir) / "mock_raw_scores.csv"
    rows = []
    np.random.seed(42)
    
    # Generate 5 comparisons per position to have enough statistics
    for pos in range(1, 170):
        for comp in range(5):
            # Normal distribution of scores, but let's inflate tail scores slightly
            # to mimic the biological effect we observed
            mean_shift = 0.5 if (pos <= 9 or pos >= 161) else 0.0
            score_left = np.random.normal(0.0 + mean_shift, 0.5)
            score_right = np.random.normal(-0.2 + mean_shift, 0.4)
            rows.append({
                "node": f"Node_{pos}_{comp}",
                "position": pos,
                "badasp_score_left": score_left,
                "badasp_score_right": score_right
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(scores_path, index=False)
    
    yield Path(temp_dir), alignment_path, scores_path
    
    shutil.rmtree(temp_dir)


def test_calculate_msa_occupancies(mock_data_dir):
    _, alignment_path, _ = mock_data_dir
    occupancies = calculate_msa_occupancies(alignment_path)
    
    assert len(occupancies) == 169
    # Positions 1-9 should have 0.5 occupancy
    for pos in range(1, 10):
        assert pytest.approx(occupancies[pos], 1e-6) == 0.5
        
    # Positions 10-160 should have 1.0 occupancy
    for pos in range(10, 161):
        assert pytest.approx(occupancies[pos], 1e-6) == 1.0
        
    # Positions 161-169 should have 0.25 occupancy
    for pos in range(161, 170):
        assert pytest.approx(occupancies[pos], 1e-6) == 0.25


def test_plot_sequence_bins_distribution_main(mock_data_dir):
    temp_path, alignment_path, scores_path = mock_data_dir
    
    # We will run main with an occupancy filter of 0.4.
    # This should keep positions 1-9 (occupancy 0.5) but filter out 161-169 (occupancy 0.25).
    min_occ = 0.4
    occ_pct = int(min_occ * 100)
    
    # Patch command line args
    test_args = [
        "plot_sequence_bins_distribution.py",
        "--scores", str(scores_path),
        "--alignment", str(alignment_path),
        "--min-occupancy", str(min_occ)
    ]
    
    # Mock output folder would be inside results/badasp_scoring/threshold_comparison/occupancy_40.
    # To keep workspace clean, let's patch results folder to be in temp_dir.
    # We can patch Path.mkdir or change out_dir directly by patching in the script.
    # Actually, in the script:
    # out_dir = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}")
    # Let's patch out_dir directly by patching Path inside the script.
    
    mock_out_dir = temp_path / f"occupancy_{occ_pct}"
    
    with patch("sys.argv", test_args):
        # Patch the target directory path logic inside plot_sequence_bins_distribution.main
        with patch("scripts.plot_sequence_bins_distribution.Path") as mock_path_cls:
            # We want Path to behave normally, except when it is creating out_dir
            # We can mock out_dir construction:
            # Let's inspect scripts/plot_sequence_bins_distribution.py:
            # line 66: out_dir = Path(f"results/badasp_scoring/threshold_comparison/occupancy_{occ_pct}")
            # We can instead just mock the write operations or let it write to a temp dir.
            # Actually, to make it extremely simple, we can patch the Path constructor
            # to redirect results/badasp_scoring/threshold_comparison/occupancy_40 to mock_out_dir.
            def side_effect(arg):
                if str(arg).startswith("results/badasp_scoring"):
                    # Redirect to temp path
                    rel_path = Path(arg).relative_to("results/badasp_scoring")
                    return temp_path / rel_path
                return Path(arg)
                
            mock_path_cls.side_effect = side_effect
            main()
            
    # Check that outputs were generated in the redirected directory
    redirected_out_dir = temp_path / "threshold_comparison" / f"occupancy_{occ_pct}"
    stats_csv = redirected_out_dir / "sequence_bins_stats.csv"
    plot_svg = redirected_out_dir / "sequence_bins_score_distribution.svg"
    plot_png = redirected_out_dir / "sequence_bins_score_distribution.png"
    violin_svg = redirected_out_dir / "sequence_bins_violin_distribution.svg"
    violin_png = redirected_out_dir / "sequence_bins_violin_distribution.png"
    
    assert stats_csv.exists()
    assert plot_svg.exists()
    assert plot_png.exists()
    assert violin_svg.exists()
    assert violin_png.exists()
    
    # Read stats CSV and verify structure
    df_stats = pd.read_csv(stats_csv)
    assert len(df_stats) == 20  # 10 bins * 2 (Unfiltered vs Filtered)
    assert set(df_stats["filter_status"].unique()) == {"Unfiltered", f"Filtered (>= {min_occ:.0%})"}
    
    # Bin 10 (positions 153-169) contains positions 161-169 with 0.25 occupancy.
    # Since min-occupancy is 0.4, these positions should be filtered out.
    # Only positions 153-160 (8 positions out of 17) remain in Bin 10 for the filtered set.
    # So the total_comparisons in Bin 10 for filtered should be 8 * 5 = 40.
    # For unfiltered, Bin 10 has 17 positions * 5 comparisons = 85.
    
    bin_10_unfiltered = df_stats[(df_stats["bin"] == "Bin 10 (153-169)") & (df_stats["filter_status"] == "Unfiltered")]
    bin_10_filtered = df_stats[(df_stats["bin"] == "Bin 10 (153-169)") & (df_stats["filter_status"].str.startswith("Filtered"))]
    
    assert bin_10_unfiltered.iloc[0]["total_comparisons"] == 85
    assert bin_10_filtered.iloc[0]["total_comparisons"] == 40
    
    # Bin 1 (positions 1-16) contains positions 1-9 (occupancy 0.5) and 10-16 (occupancy 1.0).
    # Since min-occupancy is 0.4, all positions in Bin 1 are retained (occupancy >= 0.5 >= 0.4).
    # So total_comparisons in Bin 1 should be identical between filtered and unfiltered (16 * 5 = 80).
    bin_1_unfiltered = df_stats[(df_stats["bin"] == "Bin 1 (1-16)") & (df_stats["filter_status"] == "Unfiltered")]
    bin_1_filtered = df_stats[(df_stats["bin"] == "Bin 1 (1-16)") & (df_stats["filter_status"].str.startswith("Filtered"))]
    
    assert bin_1_unfiltered.iloc[0]["total_comparisons"] == 80
    assert bin_1_filtered.iloc[0]["total_comparisons"] == 80
