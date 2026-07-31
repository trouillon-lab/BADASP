#!/usr/bin/env python
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from ete3 import Tree

def main():
    parser = argparse.ArgumentParser(description="Root tree using MAD and transfer node names from unrooted tree.")
    parser.add_argument("--unrooted-tree", type=Path, required=True, help="Original ASR tree with node names")
    parser.add_argument("--output-tree", type=Path, required=True, help="Output rooted tree with transferred node names")
    args = parser.parse_args()

    # MAD outputs to <input_tree>.rooted
    rooted_tmp = args.unrooted_tree.with_name(f"{args.unrooted_tree.name}.rooted")
    if not rooted_tmp.exists():
        rooted_tmp = args.unrooted_tree.with_suffix(f"{args.unrooted_tree.suffix}.rooted")

    # 1. Run MAD rooting on unrooted tree (if not already done)
    if rooted_tmp.exists() and rooted_tmp.stat().st_size > 0:
        print(f"Rooted tree cache found at {rooted_tmp}. Skipping MAD execution.")
    else:
        mad_script = Path("venv/bin/mad.py")
        if not mad_script.exists():
            print(f"MAD script not found: {mad_script}")
            sys.exit(1)

        print(f"Running MAD rooting on {args.unrooted_tree}...")
        # -p: keep polytomies flat instead of arbitrarily resolving them into
        #     zero-length branches (these fabricated nodes have no ancestral-state
        #     record, since they don't exist in the ASR treefile).
        # -t: retain tiny (<1e-6) branch lengths instead of contracting them to 0.0,
        #     which would otherwise recreate the same problem.
        subprocess.run([sys.executable, str(mad_script), "-p", "-t", str(args.unrooted_tree)], check=True)

        if not rooted_tmp.exists():
            rooted_tmp = args.unrooted_tree.with_name(f"{args.unrooted_tree.name}.rooted")
        if not rooted_tmp.exists():
            rooted_tmp = args.unrooted_tree.with_suffix(f"{args.unrooted_tree.suffix}.rooted")
        if not rooted_tmp.exists():
            print("Error: MAD output file not found.")
            sys.exit(1)

    # 2. Load trees
    print("Loading trees to map node names...")
    t_orig = Tree(str(args.unrooted_tree), format=1)
    t_root = Tree(str(rooted_tmp), format=1)

    # Index original node names by leaf set
    orig_node_by_sig = {}
    all_leaves = set(t_orig.get_leaf_names())
    for node in t_orig.traverse():
        if not node.is_leaf() and node.name:
            sig = frozenset(node.get_leaf_names())
            orig_node_by_sig[sig] = node.name

    # Map names to rooted tree nodes
    mapped = 0
    unmapped = 0
    for node in t_root.traverse():
        if not node.is_leaf():
            sig = frozenset(node.get_leaf_names())
            name = orig_node_by_sig.get(sig)
            if not name:
                comp_sig = frozenset(all_leaves - sig)
                name = orig_node_by_sig.get(comp_sig)
            if name:
                node.name = name
                mapped += 1
            else:
                node.name = ""  # Clear any default or trash labels
                unmapped += 1

    print(f"Successfully mapped {mapped} node names. {unmapped} nodes unmapped (new root splits).")

    # 3. Write final output tree
    args.output_tree.parent.mkdir(parents=True, exist_ok=True)
    t_root.write(outfile=str(args.output_tree), format=1)
    print(f"Saved rooted tree to {args.output_tree}")

    # Keep the rooted cache file for reference, but copy it if needed.
    # Note: We do NOT delete rooted_tmp here because it is a valuable cache.

if __name__ == "__main__":
    main()
