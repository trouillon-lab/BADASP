"""Unit tests for rec_check score and switch analysis pipeline execution."""

import pytest
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO

from scripts.compare_thresholds import calculate_msa_occupancies


from Bio.Align import MultipleSeqAlignment

@pytest.fixture
def dummy_alignment():
    with TemporaryDirectory() as tmpdir:
        aln_path = Path(tmpdir) / "dummy.aln"
        records = [
            SeqRecord(Seq("ACGT--ACGT"), id="seq1"),
            SeqRecord(Seq("ACGT--ACGT"), id="seq2"),
            SeqRecord(Seq("ACGTACACGT"), id="seq3"),
        ]
        aln = MultipleSeqAlignment(records)
        AlignIO.write(aln, str(aln_path), "fasta")
        yield aln_path


def test_calculate_msa_occupancies(dummy_alignment):
    occs = calculate_msa_occupancies(dummy_alignment)
    assert len(occs) == 10
    # Positions 5 and 6 have 2 gaps out of 3 seqs -> occupancy = 1/3 ~ 0.333
    assert pytest.approx(occs[5], 0.01) == 0.333
    # Position 1 has 0 gaps -> occupancy = 1.0
    assert pytest.approx(occs[1], 0.01) == 1.0


def test_rec_check_scores_file_validity():
    scores_path = Path("results/rec_check/badasp_scoring/raw_node_scores.csv")
    if not scores_path.exists():
        pytest.skip("rec_check scores not available on this checkout")
    
    df = pd.read_csv(scores_path)
    assert "node_name" in df.columns
    assert "position" in df.columns
    assert "badasp_score_left" in df.columns
    assert "badasp_score_right" in df.columns
    assert "event_type" in df.columns
    assert len(df) > 0, "Scores CSV must not be empty"
