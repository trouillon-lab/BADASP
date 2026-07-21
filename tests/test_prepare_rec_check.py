import sys
from pathlib import Path
from Bio import SeqIO
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_rec_check import write_filtered_fasta

def test_write_filtered_fasta(tmp_path):
    # Create a dummy raw FASTA file
    raw_fasta = tmp_path / "raw.fasta"
    output_fasta = tmp_path / "output.fasta"
    
    records = [
        ">tr|A0A062VCL5|A0A062VCL5_9ACTN TF IPR019888 OS=Actinomadura namibiensis OX=100053 GN=asnC PE=3 SV=1\nMSERTTAIQVEGRD",
        ">tr|B0A062VCL5|B0A062VCL5_9ACTN TF IPR019888 OS=Streptomyces coelicolor OX=100291 GN=asnC PE=3 SV=1\nMSERTTAIQVEGRD",
        ">tr|C0A062VCL5|C0A062VCL5_9ACTN TF IPR019888 OS=Escherichia coli OX=562 GN=asnC PE=3 SV=1\nMSERTTAIQVEGRD",
        ">tr|D0A062VCL5|D0A062VCL5_9ACTN TF IPR019888 OS=uncultured bacterium OX=199 GN=asnC PE=3 SV=1\nMSERTTAIQVEGRD",
    ]
    raw_fasta.write_text("\n".join(records) + "\n", encoding="utf-8")
    
    # Select only OX=100053, OX=562, and OX=199 (but OX=199 is environmental and should be skipped)
    selected_species = {"100053", "562", "199"}
    
    count = write_filtered_fasta(raw_fasta, selected_species, output_fasta)
    assert count == 2
    
    # Verify written file contents
    written_records = list(SeqIO.parse(str(output_fasta), "fasta"))
    assert len(written_records) == 2
    assert "A0A062VCL5" in written_records[0].id
    assert "C0A062VCL5" in written_records[1].id
    assert "B0A062VCL5" not in [r.id for r in written_records]
    assert "D0A062VCL5" not in [r.id for r in written_records]
