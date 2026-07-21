#!/usr/bin/env python
"""Solve the Hitting Set (Set Cover) problem on CD-HIT clusters to find 
the minimum number of species needed to cover all clusters.
"""

from pathlib import Path
import re
import sys
from Bio import SeqIO

# Import extract_taxid_token from alerax_inputs if possible
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.badasp.alerax_inputs import extract_taxid_token

def parse_fasta_headers(fasta_path: Path) -> dict:
    """Map sequence accession to taxid/species name."""
    print(f"Parsing FASTA from {fasta_path}...")
    acc_to_species = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        accession = parts[1] if len(parts) >= 3 else record.id
        
        species = extract_taxid_token(record.description)
        if species:
            acc_to_species[accession] = species
            acc_to_species[record.id] = species
            
    print(f"  Parsed {len(acc_to_species)} sequence mappings.")
    return acc_to_species

def parse_clstr_file(clstr_path: Path, acc_to_species: dict) -> list:
    """Parse CD-HIT clstr file and group by cluster."""
    print(f"Parsing clstr file from {clstr_path}...")
    clusters = []
    current_cluster = []
    
    with open(clstr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            else:
                # Format: 0	200aa, >tr|A0A062VCL5|A0A06... *
                match = re.search(r">(\S+)", line)
                if match:
                    seq_name = match.group(1)
                    if seq_name.endswith("..."):
                        seq_name = seq_name[:-3]
                    
                    parts = seq_name.split("|")
                    accession = parts[1] if len(parts) >= 3 else seq_name
                    
                    species = acc_to_species.get(accession) or acc_to_species.get(seq_name)
                    if species:
                        current_cluster.append(species)
                    else:
                        current_cluster.append("unknown")
                            
    if current_cluster:
        clusters.append(current_cluster)
        
    print(f"  Parsed {len(clusters)} clusters.")
    return clusters

def solve_set_cover(clusters: list) -> tuple:
    """Solve set cover using reduction rules and greedy coverage on the remaining subproblem."""
    cluster_to_species = [set(sp for sp in c if sp != "unknown") for c in clusters]
    # Filter out empty clusters (only environmental/unknown sequences) to prevent empty sets
    # from dominating and incorrectly deleting resolved clusters
    cluster_to_species = [s for s in cluster_to_species if s]
    num_clusters = len(cluster_to_species)
    print(f"Total clusters: {num_clusters}")
    
    # Active structures for reduction
    active_clusters = {i: s.copy() for i, s in enumerate(cluster_to_species)}
    species_to_clusters = {}
    for c_idx, sp_set in active_clusters.items():
        for sp in sp_set:
            species_to_clusters.setdefault(sp, set()).add(c_idx)
            
    selected = set()
    changed = True
    
    print("Running set cover reduction rules...")
    while changed:
        changed = False
        
        # 1. Essential species (only choice for some cluster)
        essential = set()
        for c_idx, sp_set in list(active_clusters.items()):
            if len(sp_set) == 1:
                essential.add(list(sp_set)[0])
                
        if essential:
            print(f"  Selecting {len(essential)} essential species...")
            for sp in essential:
                selected.add(sp)
                covered = species_to_clusters.get(sp, set())
                for c_idx in list(covered):
                    if c_idx in active_clusters:
                        for other_sp in active_clusters[c_idx]:
                            if other_sp != sp:
                                species_to_clusters[other_sp].discard(c_idx)
                        del active_clusters[c_idx]
                if sp in species_to_clusters:
                    del species_to_clusters[sp]
            changed = True
            continue
            
        # 2. Dominated species (if s1 is a subset of s2, remove s1)
        dominated_sp = set()
        sp_list = list(species_to_clusters.keys())
        for i in range(len(sp_list)):
            s1 = sp_list[i]
            cov1 = species_to_clusters[s1]
            for j in range(len(sp_list)):
                if i == j:
                    continue
                s2 = sp_list[j]
                cov2 = species_to_clusters[s2]
                if cov1.issubset(cov2) and len(cov1) < len(cov2):
                    dominated_sp.add(s1)
                    break
                    
        if dominated_sp:
            print(f"  Removing {len(dominated_sp)} dominated species...")
            for sp in dominated_sp:
                del species_to_clusters[sp]
                for c_idx in list(active_clusters.keys()):
                    active_clusters[c_idx].discard(sp)
            changed = True
            continue
            
        # 3. Dominated clusters (if c1 is subset of c2, covering c1 covers c2, we prune c2)
        dominated_cl = set()
        cl_list = list(active_clusters.keys())
        for i in range(len(cl_list)):
            c1 = cl_list[i]
            sp1 = active_clusters[c1]
            for j in range(len(cl_list)):
                if i == j:
                    continue
                c2 = cl_list[j]
                sp2 = active_clusters[c2]
                if sp1.issubset(sp2) and len(sp1) < len(sp2):
                    dominated_cl.add(c2)
                    
        if dominated_cl:
            print(f"  Removing {len(dominated_cl)} dominated clusters...")
            for c_idx in dominated_cl:
                for sp in active_clusters[c_idx]:
                    species_to_clusters[sp].discard(c_idx)
                del active_clusters[c_idx]
            changed = True
            continue

    print(f"Reductions completed. Pre-selected essential/dominating species: {len(selected)}")
    print(f"Remaining subproblem size: {len(active_clusters)} clusters, {len(species_to_clusters)} candidate species.")

    # 4. Greedy cover solver on remaining subproblem
    uncovered = set(active_clusters.keys())
    greedy_additional = []
    temp_species_to_clusters = {sp: s.copy() for sp, s in species_to_clusters.items()}
    
    while uncovered:
        best_sp = None
        best_cover = set()
        for sp, cov in temp_species_to_clusters.items():
            intersection = cov.intersection(uncovered)
            if len(intersection) > len(best_cover):
                best_sp = sp
                best_cover = intersection
                
        if not best_sp or not best_cover:
            break
            
        greedy_additional.append(best_sp)
        uncovered.difference_update(best_cover)
        
    print(f"Greedy subproblem solver selected {len(greedy_additional)} additional species.")
    total_selected = list(selected) + greedy_additional
    return total_selected, len(selected), len(greedy_additional)

def main():
    raw_fasta = Path("data/raw/IPR019888.fasta")
    clstr_file = Path("data/interim/IPR019888_clustered.fasta.clstr")
    
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "species_set_cover_report.md"
    
    acc_to_species = parse_fasta_headers(raw_fasta)
    clusters = parse_clstr_file(clstr_file, acc_to_species)
    
    selected_species, n_essential, n_greedy = solve_set_cover(clusters)
    
    # Translate species taxids using NCBITaxa
    from ete3 import NCBITaxa
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
        "# Exploratory Analysis: Minimal Species Set Cover of CD-HIT Clusters",
        "",
        "This report identifies the minimum number of species required to cover all CD-HIT sequence clusters (i.e. to have at least one sequence from that species in every cluster).",
        "",
        "## Summary Metrics",
        "",
        f"- **Total CD-HIT Clusters**: {len(clusters):,}",
        f"- **Unique Species Present in Dataset**: {len(set(acc_to_species.values())):,}",
        f"- **Pre-selected Species (Essential/Dominating)**: {n_essential:,}",
        f"- **Additional Species Selected (Greedy)**: {n_greedy:,}",
        f"- **Minimum Species Required (Total Cover)**: **{len(selected_species):,}**",
        "",
        "## Selected Minimal Species Set",
        "",
        "The following species cover all sequence clusters at minimum:"
    ]
    
    for idx, sp in enumerate(sorted(species_details)):
        report_content.append(f"{idx+1}. {sp}")
        
    report_content_str = "\n".join(report_content) + "\n"
    report_path.write_text(report_content_str, encoding="utf-8")
    
    print(f"\nSaved report to: {report_path}")
    print(f"Total species needed: {len(selected_species)}")

if __name__ == "__main__":
    main()
