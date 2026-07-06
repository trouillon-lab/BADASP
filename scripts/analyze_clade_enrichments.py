#!/usr/bin/env python3
import sys
import re
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom
from ete3 import Tree
from Bio import SeqIO


def parse_fasta_headers(fasta_path: Path) -> pd.DataFrame:
    """Parse UniProt headers from fasta file to extract metadata."""
    print(f"Parsing FASTA headers from {fasta_path}...")
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
        parts = header.split('|')
        if len(parts) < 3:
            continue
        db = parts[0].lstrip('>')
        acc = parts[1]
        rest = parts[2]
        entry = rest.split()[0]
        
        # Extract OS, OX, GN
        os_match = re.search(r'OS=(.*?)(?: OX=| GN=| PE=| SV=|$)', rest)
        ox_match = re.search(r'OX=(\d+)', rest)
        gn_match = re.search(r'GN=(.*?)(?: PE=| SV=|$)', rest)
        
        # Protein name is between entry name and OS=
        prot_match = re.search(rf'{entry}\s+(.*?)\s+OS=', rest)
        prot_name = prot_match.group(1) if prot_match else 'Unknown'
        
        os_str = os_match.group(1) if os_match else 'Unknown'
        ox_val = int(ox_match.group(1)) if ox_match else None
        gn_str = gn_match.group(1).lower() if gn_match else 'unknown'
        
        # Categorize functional family
        func_cat = "Other / Uncharacterized"
        if "siroheme decarboxylase" in prot_name.lower() or "nird" in gn_str or "nirg" in gn_str or "ahba" in gn_str or "ahbb" in gn_str:
            func_cat = "Siroheme Decarboxylase (Enzyme)"
        elif any(k in prot_name.lower() for k in ["regulator", "regulatory", "lrp", "asnc", "decr", "hth", "winged", "repressor", "activator"]):
            func_cat = "Lrp/AsnC Transcription Factor (Regulator)"
            
        records.append({
            "leaf_name": header.split()[0].lstrip('>'),
            "accession": acc,
            "entry_name": entry,
            "organism": os_str,
            "taxid": ox_val,
            "gene": gn_str,
            "protein_name": prot_name,
            "functional_category": func_cat
        })
    return pd.DataFrame(records)


def bin_clade_sizes(series: pd.Series, num_bins: int = 10) -> tuple:
    bins = pd.qcut(series, q=num_bins, duplicates="drop")
    categories = sorted(bins.cat.categories)
    return bins, categories


def calculate_bin_thresholds(melted_df: pd.DataFrame, score_col: str, bin_col: str, percentile: float = 99.9) -> dict:
    thresholds = {}
    # Overall/event-agnostic thresholds per bin
    for bin_interval in melted_df[bin_col].cat.categories:
        bin_df = melted_df[melted_df[bin_col] == bin_interval]
        if not bin_df.empty:
            thresholds[("overall", bin_interval)] = np.nanpercentile(bin_df[score_col], percentile)
        else:
            thresholds[("overall", bin_interval)] = np.inf
    return thresholds


def run_hypergeometric_enrichment(bg_df: pd.DataFrame, target_leaves: set, attribute_col: str) -> pd.DataFrame:
    """Run hypergeometric test for enrichment of an attribute in the target clades."""
    bg_total = len(bg_df)
    target_total = len(bg_df[bg_df["leaf_name"].isin(target_leaves)])
    
    # Calculate counts for each attribute value
    bg_counts = bg_df[attribute_col].value_counts()
    target_counts = bg_df[bg_df["leaf_name"].isin(target_leaves)][attribute_col].value_counts()
    
    results = []
    for val, bg_k in bg_counts.items():
        target_k = target_counts.get(val, 0)
        
        # Hypergeometric test survival function: P(X >= target_k)
        # sf(x, M, n, N) = P(X > x) for hypergeom with parameters:
        # M: total population size (bg_total)
        # n: total number of successes in population (bg_k)
        # N: sample size (target_total)
        p_val = hypergeom.sf(target_k - 1, bg_total, bg_k, target_total) if target_k > 0 else 1.0
        
        expected = (bg_k / bg_total) * target_total
        enrichment_fold = target_k / expected if expected > 0 else 0.0
        
        results.append({
            "attribute_value": val,
            "observed_in_target": target_k,
            "expected_in_target": expected,
            "fold_enrichment": enrichment_fold,
            "observed_in_bg": bg_k,
            "proportion_in_target": target_k / target_total if target_total > 0 else 0.0,
            "proportion_in_bg": bg_k / bg_total,
            "p_value": p_val
        })
        
    res_df = pd.DataFrame(results)
    
    # Benjamini-Hochberg FDR correction
    if not res_df.empty:
        res_df = res_df.sort_values(by="p_value")
        m = len(res_df)
        res_df["rank"] = range(1, m + 1)
        res_df["fdr_corrected_p_value"] = res_df["p_value"] * m / res_df["rank"]
        # Clip to max of 1.0
        res_df["fdr_corrected_p_value"] = res_df["fdr_corrected_p_value"].clip(upper=1.0)
        # Force monotonic p-values for FDR
        res_df["fdr_corrected_p_value"] = res_df["fdr_corrected_p_value"].iloc[::-1].cummin().iloc[::-1]
        res_df = res_df.drop(columns=["rank"])
        
    return res_df.sort_values(by="p_value")


def main() -> None:
    fasta_path = Path("data/interim/IPR019888_clustered.fasta")
    tax_path = Path("data/ncbi_taxonomy_map.csv")
    tree_path = Path("data/interim/asr_run.treefile")
    scores_path = Path("results/badasp_scoring/raw_node_scores.csv")
    
    out_dir = Path("results/clade_enrichment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse protein metadata
    metadata_df = parse_fasta_headers(fasta_path)
    
    # 2. Merge taxonomy map
    print("Loading taxonomy mapping...")
    tax_df = pd.read_csv(tax_path)
    metadata_df = metadata_df.merge(tax_df, on="taxid", how="left")
    metadata_df["phylum"] = metadata_df["phylum"].fillna("Unknown")
    metadata_df["class"] = metadata_df["class"].fillna("Unknown")
    
    # 3. Load tree and map clades
    print(f"Loading tree from {tree_path}...")
    tree = Tree(str(tree_path), format=1)
    
    print("Pre-mapping leaf names to descendants for tree nodes...")
    node_leaves = {}
    for node in tree.traverse():
        if not node.is_leaf():
            node_leaves[node.name] = set(leaf.name for leaf in node.get_leaves())

    # 4. Load raw node scores and identify 99.9th% switches
    print("Loading raw node scores...")
    df = pd.read_csv(scores_path)
    
    # Filter clade sizes and occupancy >= 0.8
    df = df[(df["clade_size_left"] >= 5) & (df["clade_size_right"] >= 5)]
    from Bio import AlignIO
    alignment = AlignIO.read("data/interim/IPR019888_trimmed.aln", "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)
    
    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= 0.8].copy()
    
    # Melt scores to construct background population for thresholding
    df_left = df_filtered[["node_name", "event_type", "position", "clade_size_left", "badasp_score_left"]].rename(
        columns={"badasp_score_left": "score", "clade_size_left": "clade_size"}
    )
    df_right = df_filtered[["node_name", "event_type", "position", "clade_size_right", "badasp_score_right"]].rename(
        columns={"badasp_score_right": "score", "clade_size_right": "clade_size"}
    )
    melted_df = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["score", "clade_size"])
    
    # Bin clade sizes
    melted_df["clade_bin"], bin_categories = bin_clade_sizes(melted_df["clade_size"], num_bins=10)
    
    # Calculate thresholds for 99.9th percentile
    target_percentile = 99.9
    thresholds = calculate_bin_thresholds(melted_df, "score", "clade_bin", percentile=target_percentile)
    
    # Map bins to parent df
    def _map_to_bin(val):
        for interval in bin_categories:
            if val in interval:
                return interval
        return bin_categories[-1] if val > bin_categories[-1].right else bin_categories[0]
        
    df_filtered["bin_left"] = df_filtered["clade_size_left"].apply(_map_to_bin)
    df_filtered["bin_right"] = df_filtered["clade_size_right"].apply(_map_to_bin)
    
    # Identify switches
    is_switch = []
    for _, row in df_filtered.iterrows():
        bin_l = row["bin_left"]
        bin_r = row["bin_right"]
        thresh_l = thresholds.get(("overall", bin_l), np.inf)
        thresh_r = thresholds.get(("overall", bin_r), np.inf)
        
        switch_l = (not np.isnan(row["badasp_score_left"])) and (row["badasp_score_left"] >= thresh_l)
        switch_r = (not np.isnan(row["badasp_score_right"])) and (row["badasp_score_right"] >= thresh_r)
        is_switch.append(switch_l or switch_r)
        
    df_filtered["is_switch"] = is_switch
    switch_nodes = df_filtered[df_filtered["is_switch"]]["node_name"].unique()
    print(f"Identified {len(switch_nodes)} unique internal nodes (comparisons) with at least one 99.9th% switch.")

    # 5. Extract leaves descending from these switch nodes
    target_leaves = set()
    for node_name in switch_nodes:
        # Get leaves descending from this node (clade comparison members)
        if node_name in node_leaves:
            target_leaves.update(node_leaves[node_name])
            
    print(f"Total unique target leaves (proteins in active divergence clades): {len(target_leaves)} / {len(metadata_df)}")

    # 6. Run enrichment for Phylum, Class, and Functional Category
    print("\n--- Running Enrichment Analysis ---")
    phylum_enrich = run_hypergeometric_enrichment(metadata_df, target_leaves, "phylum")
    class_enrich = run_hypergeometric_enrichment(metadata_df, target_leaves, "class")
    func_enrich = run_hypergeometric_enrichment(metadata_df, target_leaves, "functional_category")
    
    # Save tables
    phylum_enrich.to_csv(out_dir / "phylum_enrichment.csv", index=False)
    class_enrich.to_csv(out_dir / "class_enrichment.csv", index=False)
    func_enrich.to_csv(out_dir / "functional_category_enrichment.csv", index=False)
    print(f"Saved enrichment CSVs to {out_dir}")
    
    # Plot results
    # 1. Phylum Enrichment Plot
    plt.figure(figsize=(10, 6))
    sig_phylum = phylum_enrich[phylum_enrich["fdr_corrected_p_value"] < 0.05].copy()
    if not sig_phylum.empty:
        # Plot -log10 p-value
        sig_phylum["-log10_fdr_p"] = -np.log10(sig_phylum["fdr_corrected_p_value"] + 1e-100)
        sns.barplot(
            data=sig_phylum.sort_values(by="-log10_fdr_p", ascending=False).head(15),
            x="-log10_fdr_p",
            y="attribute_value",
            palette="Reds_r"
        )
        plt.axvline(-np.log10(0.05), color="red", linestyle="--", label="FDR p = 0.05")
        plt.title("Phylum Enrichment in Active Divergence Clades (99.9th% switches)", fontsize=12, fontweight="bold")
        plt.xlabel("-log10(FDR-corrected p-value)", fontsize=10)
        plt.ylabel("Phylum", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "phylum_enrichment.png", dpi=300)
        plt.savefig(out_dir / "phylum_enrichment.svg", format="svg")
        plt.close()
        
    # 2. Functional Category Enrichment Plot
    plt.figure(figsize=(8, 5))
    func_enrich["-log10_fdr_p"] = -np.log10(func_enrich["fdr_corrected_p_value"] + 1e-100)
    sns.barplot(
        data=func_enrich.sort_values(by="-log10_fdr_p", ascending=False),
        x="-log10_fdr_p",
        y="attribute_value",
        palette="Blues_r"
    )
    plt.axvline(-np.log10(0.05), color="red", linestyle="--", label="FDR p = 0.05")
    plt.title("Functional Family Enrichment in Active Divergence Clades (99.9th%)", fontsize=11, fontweight="bold")
    plt.xlabel("-log10(FDR-corrected p-value)", fontsize=10)
    plt.ylabel("Functional Family", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "functional_category_enrichment.png", dpi=300)
    plt.savefig(out_dir / "functional_category_enrichment.svg", format="svg")
    plt.close()

    # Generate Summary Report
    print("\n--- Summary of Functional Family Enrichment ---")
    print(func_enrich[["attribute_value", "observed_in_target", "observed_in_bg", "fold_enrichment", "fdr_corrected_p_value"]].to_string(index=False))
    
    print("\n--- Top Significant Phyla (FDR p < 0.05) ---")
    print(phylum_enrich[phylum_enrich["fdr_corrected_p_value"] < 0.05][["attribute_value", "observed_in_target", "fold_enrichment", "fdr_corrected_p_value"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
