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
from ete3 import Tree
from Bio import SeqIO


def main() -> None:
    fasta_path = Path("data/interim/IPR019888_clustered.fasta")
    tax_path = Path("data/ncbi_taxonomy_map.csv")
    tree_path = Path("data/interim/asr_run.treefile")
    scores_path = Path("results/badasp_scoring/raw_node_scores.csv")
    
    out_dir = Path("results/deep_clade_investigation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load tree
    print("Loading tree...")
    tree = Tree(str(tree_path), format=1)
    
    # 2. Load scores and identify nodes with root distance > 7
    print("Loading raw node scores...")
    scores_df = pd.read_csv(scores_path)
    
    # Filter clade sizes and occupancy >= 0.8
    scores_df = scores_df[(scores_df["clade_size_left"] >= 5) & (scores_df["clade_size_right"] >= 5)]
    from Bio import AlignIO
    alignment = AlignIO.read("data/interim/IPR019888_trimmed.aln", "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)
    
    scores_df["occupancy"] = scores_df["position"].map(occupancies)
    scores_df = scores_df[scores_df["occupancy"] >= 0.8].copy()
    
    nodes_gt7 = scores_df[scores_df["distance_from_root"] > 7]["node_name"].unique()
    print(f"Found {len(nodes_gt7)} internal nodes with distance from root > 7.")
    
    # 3. Find LCA of all leaves descending from these nodes
    print("Finding LCA of the deep root-distance clades...")
    all_leaves_gt7 = set()
    for node_name in nodes_gt7:
        node = tree.search_nodes(name=node_name)[0]
        all_leaves_gt7.update(leaf.name for leaf in node.get_leaves())
        
    lca_node = tree.get_common_ancestor(list(all_leaves_gt7))
    lca_name = lca_node.name
    print(f"LCA of deep clades is node: {lca_name}")
    print(f"LCA node cumulative root distance: {tree.get_distance(tree.get_tree_root(), lca_node):.4f}")
    
    # Get all leaves under this LCA
    lca_leaves = set(leaf.name for leaf in lca_node.get_leaves())
    print(f"Total leaves under LCA node {lca_name}: {len(lca_leaves)}")
    
    # 4. Parse FASTA to extract metadata for LCA leaves
    print("Parsing FASTA headers for deep clade leaves...")
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
        name = header.split()[0].lstrip('>')
        if name in lca_leaves:
            parts = header.split('|')
            acc = parts[1]
            rest = parts[2]
            entry = rest.split()[0]
            
            os_match = re.search(r'OS=(.*?)(?: OX=| GN=| PE=| SV=|$)', rest)
            ox_match = re.search(r'OX=(\d+)', rest)
            gn_match = re.search(r'GN=(.*?)(?: PE=| SV=|$)', rest)
            prot_match = re.search(rf'{entry}\s+(.*?)\s+OS=', rest)
            
            os_str = os_match.group(1) if os_match else 'Unknown'
            ox_val = int(ox_match.group(1)) if ox_match else None
            gn_str = gn_match.group(1).lower() if gn_match else 'unknown'
            prot_name = prot_match.group(1) if prot_match else 'Unknown'
            
            genus = os_str.split()[0] if os_str != 'Unknown' else 'Unknown'
            
            records.append({
                "leaf_name": name,
                "accession": acc,
                "entry_name": entry,
                "organism": os_str,
                "genus": genus,
                "taxid": ox_val,
                "gene": gn_str,
                "protein_name": prot_name
            })
            
    deep_metadata_df = pd.DataFrame(records)
    
    # Load and merge taxonomy mapping
    tax_df = pd.read_csv(tax_path)
    deep_metadata_df = deep_metadata_df.merge(tax_df, on="taxid", how="left")
    deep_metadata_df["phylum"] = deep_metadata_df["phylum"].fillna("Unknown")
    deep_metadata_df["class"] = deep_metadata_df["class"].fillna("Unknown")
    
    # Save metadata table
    deep_metadata_df.to_csv(out_dir / "deep_clade_proteins.csv", index=False)
    print(f"Saved metadata of {len(deep_metadata_df)} deep clade proteins to CSV.")
    
    # 5. Extract BADASP score stats and switch events inside this clade
    # Find all internal nodes (comparisons) that are descendants of LCA node
    lca_descendant_nodes = set(node.name for node in lca_node.traverse() if not node.is_leaf() and node.name)
    deep_scores_df = scores_df[scores_df["node_name"].isin(lca_descendant_nodes)].copy()
    
    # Calculate thresholds for 99.9th% (quantile binning with 10 bins)
    # Using overall melted scores to get the correct threshold for each bin
    df_left = scores_df[["node_name", "event_type", "position", "clade_size_left", "badasp_score_left"]].rename(
        columns={"badasp_score_left": "score", "clade_size_left": "clade_size"}
    )
    df_right = scores_df[["node_name", "event_type", "position", "clade_size_right", "badasp_score_right"]].rename(
        columns={"badasp_score_right": "score", "clade_size_right": "clade_size"}
    )
    melted_df = pd.concat([df_left, df_right], ignore_index=True).dropna(subset=["score", "clade_size"])
    bins = pd.qcut(melted_df["clade_size"], q=10, duplicates="drop")
    bin_categories = sorted(bins.cat.categories)
    
    thresholds = {}
    melted_df["clade_bin"] = pd.cut(melted_df["clade_size"], bins=pd.IntervalIndex(bin_categories))
    for bin_interval in bin_categories:
        bin_df = melted_df[melted_df["clade_bin"] == bin_interval]
        if not bin_df.empty:
            thresholds[bin_interval] = np.nanpercentile(bin_df["score"], 99.9)
        else:
            thresholds[bin_interval] = np.inf
            
    # Function to map clade size to threshold
    def get_thresh(clade_size):
        for interval in bin_categories:
            if clade_size in interval:
                return thresholds.get(interval, np.inf)
        return thresholds[bin_categories[-1]] if clade_size > bin_categories[-1].right else thresholds[bin_categories[0]]

    # Identify switches inside the deep clade
    deep_switches = []
    for _, row in deep_scores_df.iterrows():
        thresh_l = get_thresh(row["clade_size_left"])
        thresh_r = get_thresh(row["clade_size_right"])
        
        switch_l = (not np.isnan(row["badasp_score_left"])) and (row["badasp_score_left"] >= thresh_l)
        switch_r = (not np.isnan(row["badasp_score_right"])) and (row["badasp_score_right"] >= thresh_r)
        
        if switch_l or switch_r:
            deep_switches.append({
                "node_name": row["node_name"],
                "event_type": row["event_type"],
                "position": row["position"],
                "badasp_score_left": row["badasp_score_left"],
                "badasp_score_right": row["badasp_score_right"],
                "clade_size_left": row["clade_size_left"],
                "clade_size_right": row["clade_size_right"],
                "distance_from_root": row["distance_from_root"]
            })
            
    deep_switches_df = pd.DataFrame(deep_switches)
    deep_switches_df.to_csv(out_dir / "deep_clade_switches.csv", index=False)
    print(f"Identified {len(deep_switches_df)} specific switch positions under Node {lca_name}.")
    
    # 6. Plotting
    # Plot 1: Taxonomy summary under LCA (Genus distribution)
    plt.figure(figsize=(10, 6))
    top_genera = deep_metadata_df["genus"].value_counts().head(15)
    sns.barplot(x=top_genera.values, y=top_genera.index, palette="viridis")
    plt.title(f"Genus Distribution in Deep Clade (under Node {lca_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Sequences", fontsize=10)
    plt.ylabel("Genus", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "genus_distribution.png", dpi=300)
    plt.savefig(out_dir / "genus_distribution.svg", format="svg")
    plt.close()
    
    # Plot 2: BADASP scores distribution inside deep clade vs all other clades
    other_scores_df = scores_df[~scores_df["node_name"].isin(lca_descendant_nodes)].copy()
    
    # Take max score per comparison-position to plot
    deep_max = np.maximum(deep_scores_df["badasp_score_left"], deep_scores_df["badasp_score_right"])
    other_max = np.maximum(other_scores_df["badasp_score_left"], other_scores_df["badasp_score_right"])
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(other_max, label="All Other Clades", color="gray", fill=True, alpha=0.3)
    sns.kdeplot(deep_max, label="Deep Clade (Pseudomonadota)", color="crimson", fill=True, alpha=0.5)
    plt.axvline(np.nanpercentile(melted_df["score"], 99.9), color="red", linestyle="--", label="Overall 99.9th% Threshold")
    plt.title(f"BADASP Score Distribution: Deep Clade vs Rest of Tree", fontsize=11, fontweight="bold")
    plt.xlabel("Max BADASP Score", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "badasp_score_comparison.png", dpi=300)
    plt.savefig(out_dir / "badasp_score_comparison.svg", format="svg")
    plt.close()
    
    # Print key statistics
    print("\n--- Deep Clade Statistics ---")
    print(f"Phylum breakdown:\n{deep_metadata_df['phylum'].value_counts()}")
    print(f"Class breakdown:\n{deep_metadata_df['class'].value_counts()}")
    print(f"Top 5 Genera:\n{deep_metadata_df['genus'].value_counts().head(5)}")
    print(f"Top 5 Protein Names:\n{deep_metadata_df['protein_name'].value_counts().head(5)}")
    print(f"Number of switches: {len(deep_switches_df)} events across {deep_switches_df['position'].nunique()} positions.")
    
    if not deep_switches_df.empty:
        print("\nTop 10 highest-scoring switch positions in deep clade:")
        print(deep_switches_df.sort_values(by="badasp_score_left", ascending=False).head(10)[["node_name", "position", "badasp_score_left", "badasp_score_right"]])


if __name__ == "__main__":
    main()
