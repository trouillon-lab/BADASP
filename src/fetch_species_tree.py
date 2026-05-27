import argparse
import re
from pathlib import Path
from ete3 import NCBITaxa, Tree
from Bio import SeqIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import Phylo

_OX_PATTERN = re.compile(r"\bOX=(\d+)\b")

def extract_taxids_from_fasta(fasta_path: Path) -> set:
    taxids = set()
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        ox_match = _OX_PATTERN.search(record.description)
        if ox_match:
            taxids.add(int(ox_match.group(1)))
    return taxids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=Path("data/interim/IPR019888_length_filtered.fasta"))
    parser.add_argument("--out-nwk", type=Path, default=Path("data/interim/ncbi_species_tree.nwk"))
    parser.add_argument("--out-png", type=Path, default=Path("results/evolutionary_analysis/ncbi_species_tree.png"))
    parser.add_argument("--collapse-rank", type=str, default="phylum", help="Rank to collapse at (e.g. phylum, class)")
    args = parser.parse_args()

    print(f"Extracting TaxIDs from {args.fasta}...")
    taxids = extract_taxids_from_fasta(args.fasta)
    print(f"Found {len(taxids)} unique TaxIDs.")

    print("Fetching topology from NCBI...")
    ncbi = NCBITaxa()
    # get_topology returns an ETE3 tree where node names are TaxIDs
    tree = ncbi.get_topology(taxids)

    print("Annotating names and ranks...")
    for node in tree.traverse():
        taxid = int(node.name)
        rank_dict = ncbi.get_rank([taxid])
        name_dict = ncbi.get_taxid_translator([taxid])
        node.add_features(
            rank=rank_dict.get(taxid, "no rank"),
            sci_name=name_dict.get(taxid, f"TaxID_{taxid}")
        )
        # We rename the node to its scientific name for readability, 
        # but store taxid as an attribute just in case.
        node.taxid = taxid

    # Save full uncollapsed tree
    args.out_nwk.parent.mkdir(parents=True, exist_ok=True)
    tree.write(format=1, outfile=str(args.out_nwk))
    print(f"Saved full uncollapsed tree to {args.out_nwk}")

    # Collapse for visualization
    print(f"Collapsing tree at rank: {args.collapse_rank}...")
    for node in list(tree.traverse("postorder")):
        if hasattr(node, "rank") and node.rank == args.collapse_rank:
            # Delete children to collapse
            for child in list(node.children):
                child.detach()

    # Save collapsed tree
    collapsed_nwk = args.out_nwk.with_suffix(".collapsed.nwk")
    for node in tree.traverse():
        node.name = node.sci_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace(";", "")
    tree.write(format=1, outfile=str(collapsed_nwk))
    print(f"Saved collapsed tree to {collapsed_nwk}")

    # Visualize with Bio.Phylo
    print(f"Rendering visualization to {args.out_png}...")
    phylo_tree = Phylo.read(str(collapsed_nwk), "newick")
    fig = plt.figure(figsize=(16, max(8, len(phylo_tree.get_terminals()) * 0.2)))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(phylo_tree, axes=ax, do_show=False)
    ax.set_title(f"NCBI Species Tree (Collapsed at {args.collapse_rank.capitalize()})")
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {args.out_png}")

if __name__ == "__main__":
    main()
