#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO


def calculate_msa_occupancies(alignment_path: Path) -> dict:
    alignment = AlignIO.read(alignment_path, "fasta")
    aln_len = alignment.get_alignment_length()
    num_seqs = len(alignment)
    occupancies = {}
    for col in range(aln_len):
        chars = [alignment[seq_idx][col] for seq_idx in range(num_seqs)]
        gaps = sum(1 for c in chars if c in {'-', '.'})
        occupancies[col + 1] = 1.0 - (gaps / num_seqs)
    return occupancies


def main() -> None:
    scores_path = Path("results/badasp_scoring/raw_node_scores.csv")
    alignment_path = Path("data/interim/IPR019888_trimmed.aln")
    out_dir = Path("results/badasp_scoring/clade_size_adjusted/min_clade_5/occupancy_80")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw node scores...")
    df = pd.read_csv(scores_path)
    
    # Apply occupancy filter (min_clade=5 is default in the file)
    df = df[(df["clade_size_left"] >= 5) & (df["clade_size_right"] >= 5)]
    
    print("Filtering alignment occupancy >= 0.8...")
    occupancies = calculate_msa_occupancies(alignment_path)
    df["occupancy"] = df["position"].map(occupancies)
    df = df[df["occupancy"] >= 0.8].copy()

    # We construct a matrix of branch scores.
    # Each row is a branch (left or right child for each node comparison).
    print("Reshaping scores to branch-level long format...")
    left_df = df[["node_name", "event_type", "clade_size_left", "position", "badasp_score_left"]].copy()
    left_df["branch_id"] = left_df["node_name"] + "_left"
    left_df = left_df.rename(columns={"clade_size_left": "clade_size", "badasp_score_left": "score"})
    
    right_df = df[["node_name", "event_type", "clade_size_right", "position", "badasp_score_right"]].copy()
    right_df["branch_id"] = right_df["node_name"] + "_right"
    right_df = right_df.rename(columns={"clade_size_right": "clade_size", "badasp_score_right": "score"})
    
    branch_long = pd.concat([left_df, right_df], ignore_index=True).dropna()

    print("Pivoting to wide matrix (branches x positions)...")
    matrix_df = branch_long.pivot(index=["branch_id", "clade_size", "event_type"], columns="position", values="score")
    
    # Extract metadata and core score matrix
    branch_metadata = matrix_df.index.to_frame(index=False)
    scores_matrix = matrix_df.values  # Shape: (N_branches, N_positions)
    positions = matrix_df.columns.values
    N_branches, N_positions = scores_matrix.shape
    print(f"Matrix shape: {N_branches} branches x {N_positions} positions")

    # Calculate observed mean scores per position (ignoring NaNs)
    observed_means = np.nanmean(scores_matrix, axis=0)

    # Run permutation test directly on BADASP scores
    n_permutations = 500
    print(f"\n--- Running permutation test directly on BADASP scores ({n_permutations} iterations) ---")
    
    perm_max_means = []
    perm_all_means = []

    np.random.seed(42)  # For reproducibility

    for p in range(n_permutations):
        # Shuffle columns independently for each row
        shuffle_idx = np.random.rand(N_branches, N_positions).argsort(axis=1)
        shuffled_scores = np.take_along_axis(scores_matrix, shuffle_idx, axis=1)
        
        # Calculate mean score per position (ignoring NaNs)
        shuffled_means = np.nanmean(shuffled_scores, axis=0)
        
        perm_max_means.append(np.nanmax(shuffled_means))
        # Keep non-nan elements for overall distribution
        perm_all_means.extend(shuffled_means[~np.isnan(shuffled_means)])

        if (p + 1) % 100 == 0:
            print(f"  Completed {p + 1} / {n_permutations} permutations")

    # Compute significance thresholds on mean scores (ignoring NaNs)
    fwer_95 = np.nanpercentile(perm_max_means, 95)
    fwer_99 = np.nanpercentile(perm_max_means, 99)
    print(f"FWER 95% threshold of mean score (empirical p < 0.05): {fwer_95:.5f}")
    print(f"FWER 99% threshold of mean score (empirical p < 0.01): {fwer_99:.5f}")

    # Calculate empirical p-values for observed positions
    pos_summary = []
    for i, pos in enumerate(positions):
        obs = observed_means[i]
        p_uncorrected = (np.array(perm_all_means) >= obs).mean()
        p_fwer = (np.array(perm_max_means) >= obs).mean()
        pos_summary.append({
            "position": pos,
            "observed_mean_score": obs,
            "p_uncorrected": p_uncorrected,
            "p_fwer": p_fwer
        })
        
    summary_df = pd.DataFrame(pos_summary)
    summary_df.to_csv(out_dir / "permutation_test_raw_score_stats.csv", index=False)
    print(f"Saved stats table to {out_dir / 'permutation_test_raw_score_stats.csv'}")

    # Plot distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(perm_all_means, bins=30, color="#7f8c8d", alpha=0.6, label="Null (shuffled)", kde=True)
    sns.histplot(perm_max_means, bins=25, color="#e74c3c", alpha=0.3, label="Null FWER Max", kde=False)
    sns.histplot(observed_means, bins=25, color="#2980b9", alpha=0.8, label="Observed Means", kde=False)
    
    plt.axvline(fwer_95, color="#c0392b", linestyle="--", linewidth=2, label=f"FWER 95% threshold ({fwer_95:.5f})")
    plt.title("Permutation Distribution vs. Observed Mean BADASP Scores (min_clade=5)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean BADASP Score at Position", fontsize=10, fontweight="bold")
    plt.ylabel("Frequency", fontsize=10, fontweight="bold")
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "permutation_test_raw_score_distributions.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "permutation_test_raw_score_distributions.svg", format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved distributions plot to {out_dir / 'permutation_test_raw_score_distributions.png'}")

    # Output top significant positions
    print("\n=== Top Significant Positions by Mean BADASP Score (empirical p_fwer < 0.05) ===")
    sig_df = summary_df[summary_df["p_fwer"] < 0.05]
    print(sig_df.sort_values(by="observed_mean_score", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
