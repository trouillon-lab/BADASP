#!/usr/bin/env python
"""Run CD-HIT at 0.55, solve species set cover, and write the subset FASTA.
"""

import sys
import subprocess
from pathlib import Path
import re
from Bio import SeqIO
from ete3 import NCBITaxa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.badasp.alerax_inputs import extract_taxid_token
from scripts.exploratory_set_cover import parse_fasta_headers, parse_clstr_file, solve_set_cover

def is_environmental(description: str) -> bool:
    """Detect environmental/unresolved sequence description."""
    if not description:
        return False
    pattern = re.compile(
        r"metagenome|"
        r"environmental\s+sample|"
        r"unidentified|"
        r"mixed\s+culture|"
        r"enrichment\s+culture|"
        r"uncultured",
        re.IGNORECASE
    )
    return bool(pattern.search(description))

def parse_fasta_headers_filtered(fasta_path: Path) -> dict:
    """Map sequence accession to taxid, filtering out environmental ones."""
    print(f"Parsing FASTA from {fasta_path} (filtering environmental)...")
    acc_to_species = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if is_environmental(record.description):
            continue
            
        parts = record.id.split("|")
        accession = parts[1] if len(parts) >= 3 else record.id
        
        species = extract_taxid_token(record.description)
        if species:
            acc_to_species[accession] = species
            acc_to_species[record.id] = species
            
    print(f"  Parsed {len(acc_to_species)} sequence mappings.")
    return acc_to_species

def run_cdhit_055(raw_fasta: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fasta = out_dir / "cdhit_T0.55.fasta"
    clstr_file = out_fasta.with_suffix(".fasta.clstr")
    
    # Check if clstr file already exists so we can reuse it to save time
    if clstr_file.exists():
        print(f"CD-HIT .clstr file already exists at {clstr_file}, reusing it.")
        return clstr_file
        
    cmd = [
        "cd-hit",
        "-i", str(raw_fasta),
        "-o", str(out_fasta),
        "-c", "0.55",
        "-n", "3",
        "-d", "0",
        "-T", "4",
        "-M", "4000"
    ]
    print("Running CD-HIT at 0.55 on raw FASTA...")
    subprocess.run(cmd, check=True)
    return clstr_file

def write_filtered_fasta(raw_fasta: Path, selected_species: set, output_fasta: Path) -> int:
    print(f"Filtering FASTA and writing to {output_fasta}...")
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    
    written_count = 0
    with open(output_fasta, "w", encoding="utf-8") as out_f:
        for record in SeqIO.parse(str(raw_fasta), "fasta"):
            if is_environmental(record.description):
                continue
            species = extract_taxid_token(record.description)
            if species and species in selected_species:
                SeqIO.write(record, out_f, "fasta")
                written_count += 1
                
    print(f"Successfully wrote {written_count} sequences to {output_fasta}.")
    return written_count

def main():
    raw_fasta = ROOT / "data/raw/IPR019888.fasta"
    temp_dir = ROOT / "data/rec_check/temp"
    output_fasta = ROOT / "data/rec_check/IPR019888_rec_check_raw.fasta"
    report_path = ROOT / "results/rec_check/species_set_cover_report.md"
    
    # 1. Run CD-HIT 0.55
    clstr_file = run_cdhit_055(raw_fasta, temp_dir)
    
    # 2. Parse headers and clstr
    acc_to_species = parse_fasta_headers_filtered(raw_fasta)
    clusters = parse_clstr_file(clstr_file, acc_to_species)
    
    # 3. Solve set cover
    selected_species, n_essential, n_greedy = solve_set_cover(clusters)
    selected_species_set = set(selected_species)
    
    # 4. Write filtered FASTA
    written_count = write_filtered_fasta(raw_fasta, selected_species_set, output_fasta)
    
    # 5. Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ncbi = NCBITaxa()
    species_details = []
    for sp in selected_species:
        if sp.isdigit():
            try:
                name_dict = ncbi.get_taxid_translator([int(sp)])
                name = name_dict.get(int(sp), "Unknown")
                species_details.append(f"{sp} ({name})")
            except Exception:
                species_details.append(f"{sp} (TaxID)")
        else:
            species_details.append(sp)
            
    report_content = [
        "# Species Reconciliation Check: Minimal Species Set Cover of CD-HIT 0.55 Clusters",
        "",
        f"- **Total CD-HIT Clusters**: {len(clusters):,}",
        f"- **Unique Species Present in Dataset**: {len(set(acc_to_species.values())):,}",
        f"- **Pre-selected Species (Essential/Dominating)**: {n_essential:,}",
        f"- **Additional Species Selected (Greedy)**: {n_greedy:,}",
        f"- **Minimum Species Required (Total Cover)**: **{len(selected_species):,}**",
        f"- **Total Sequences Extracted (All TFs of Selected Species)**: **{written_count:,}**",
        "",
        "## Selected Minimal Species Set",
        ""
    ]
    for idx, sp in enumerate(sorted(species_details)):
        report_content.append(f"{idx+1}. {sp}")
        
    report_path.write_text("\n".join(report_content) + "\n", encoding="utf-8")
    print(f"Saved report to: {report_path}")

if __name__ == "__main__":
    main()
