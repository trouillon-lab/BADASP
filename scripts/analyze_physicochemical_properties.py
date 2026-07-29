"""Module 4: Physicochemical Property Shift Analysis for Initial Run BADASP Switches.

Calculates Grantham distances, net charge shifts, hydropathy deltas, and molecular volume shifts
for 99.9th percentile switches, and contrasts domain-specific physicochemical profiles.
Outputs saved under results/initial_run_characterization/.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.analyze_switch_origin_time import (
    calculate_msa_occupancies,
    calculate_bin_thresholds_999,
    extract_switch_instances,
    DOMAINS,
)


# Kyte-Doolittle Hydropathy Scale
KYTE_DOOLITTLE = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}

# Net Charge at physiological pH
NET_CHARGE = {
    'K': 1.0, 'R': 1.0, 'H': 1.0,
    'D': -1.0, 'E': -1.0,
    'A': 0.0, 'C': 0.0, 'F': 0.0, 'G': 0.0, 'I': 0.0,
    'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0
}

# Residue Molecular Volume (Å³)
MOLECULAR_VOLUME = {
    'G': 60.1, 'A': 88.6, 'S': 89.0, 'C': 108.5, 'D': 111.1,
    'P': 112.7, 'N': 114.1, 'T': 116.1, 'E': 138.4, 'V': 140.0,
    'Q': 143.8, 'H': 153.2, 'M': 162.9, 'L': 166.7, 'I': 166.7,
    'K': 168.6, 'R': 173.4, 'F': 189.9, 'Y': 193.6, 'W': 227.8
}

# Grantham Distance Matrix (Grantham 1974)
GRANTHAM_MATRIX = {
    ('A','A'):0, ('A','C'):195, ('A','D'):126, ('A','E'):107, ('A','F'):113, ('A','G'):60, ('A','H'):86, ('A','I'):94, ('A','K'):106, ('A','L'):96, ('A','M'):84, ('A','N'):111, ('A','P'):27, ('A','Q'):91, ('A','R'):112, ('A','S'):58, ('A','T'):58, ('A','V'):64, ('A','W'):148, ('A','Y'):112,
    ('C','C'):0, ('C','D'):154, ('C','E'):170, ('C','F'):205, ('C','G'):159, ('C','H'):202, ('C','I'):198, ('C','K'):202, ('C','L'):198, ('C','M'):196, ('C','N'):139, ('C','P'):169, ('C','Q'):154, ('C','R'):180, ('C','S'):112, ('C','T'):119, ('C','V'):192, ('C','W'):215, ('C','Y'):194,
    ('D','D'):0, ('D','E'):45, ('D','F'):177, ('D','G'):42, ('D','H'):81, ('D','I'):168, ('D','K'):101, ('D','L'):172, ('D','M'):160, ('D','N'):23, ('D','P'):108, ('D','Q'):61, ('D','R'):96, ('D','S'):65, ('D','T'):85, ('D','V'):152, ('D','W'):181, ('D','Y'):160,
    ('E','E'):0, ('E','F'):140, ('E','G'):87, ('E','H'):40, ('E','I'):134, ('E','K'):56, ('E','L'):138, ('E','M'):126, ('E','N'):42, ('E','P'):93, ('E','Q'):29, ('E','R'):54, ('E','S'):80, ('E','T'):65, ('E','V'):121, ('E','W'):152, ('E','Y'):122,
    ('F','F'):0, ('F','G'):153, ('F','H'):100, ('F','I'):21, ('F','K'):102, ('F','L'):22, ('F','M'):28, ('F','N'):158, ('F','P'):134, ('F','Q'):116, ('F','R'):104, ('F','S'):155, ('F','T'):103, ('F','V'):50, ('F','W'):40, ('F','Y'):22,
    ('G','G'):0, ('G','H'):89, ('G','I'):135, ('G','K'):127, ('G','L'):138, ('G','M'):127, ('G','N'):80, ('G','P'):42, ('G','Q'):87, ('G','R'):125, ('G','S'):56, ('G','T'):59, ('G','V'):109, ('G','W'):184, ('G','Y'):147,
    ('H','H'):0, ('H','I'):94, ('H','K'):32, ('H','L'):99, ('H','M'):87, ('H','N'):68, ('H','P'):77, ('H','Q'):24, ('H','R'):29, ('H','S'):78, ('H','T'):47, ('H','V'):84, ('H','W'):115, ('H','Y'):83,
    ('I','I'):0, ('I','K'):102, ('I','L'):5, ('I','M'):10, ('I','N'):149, ('I','P'):95, ('I','Q'):109, ('I','R'):97, ('I','S'):142, ('I','T'):89, ('I','V'):29, ('I','W'):61, ('I','Y'):33,
    ('K','K'):0, ('K','L'):107, ('K','M'):95, ('K','N'):94, ('K','P'):103, ('K','Q'):53, ('K','R'):26, ('K','S'):121, ('K','T'):78, ('K','V'):97, ('K','W'):110, ('K','Y'):85,
    ('L','L'):0, ('L','M'):15, ('L','N'):153, ('L','P'):98, ('L','Q'):113, ('L','R'):102, ('L','S'):145, ('L','T'):92, ('L','V'):32, ('L','W'):61, ('L','Y'):36,
    ('M','M'):0, ('M','N'):142, ('M','P'):87, ('M','Q'):101, ('M','R'):91, ('M','S'):135, ('M','T'):81, ('M','V'):21, ('M','W'):67, ('M','Y'):36,
    ('N','N'):0, ('N','P'):91, ('N','Q'):46, ('N','R'):86, ('N','S'):46, ('N','T'):65, ('N','V'):133, ('N','W'):174, ('N','Y'):143,
    ('P','P'):0, ('P','Q'):87, ('P','R'):103, ('P','S'):74, ('P','T'):38, ('P','V'):68, ('P','W'):147, ('P','Y'):110,
    ('Q','Q'):0, ('Q','R'):43, ('Q','S'):68, ('Q','T'):42, ('Q','V'):96, ('Q','W'):130, ('Q','Y'):99,
    ('R','R'):0, ('R','S'):110, ('R','T'):71, ('R','V'):96, ('R','W'):101, ('R','Y'):77,
    ('S','S'):0, ('S','T'):58, ('S','V'):124, ('S','W'):177, ('S','Y'):144,
    ('T','T'):0, ('T','V'):69, ('T','W'):128, ('T','Y'):92,
    ('V','V'):0, ('V','W'):88, ('V','Y'):55,
    ('W','W'):0, ('W','Y'):37,
    ('Y','Y'):0
}


def get_grantham_distance(aa1: str, aa2: str) -> float:
    """Retrieve Grantham distance between two amino acids."""
    if aa1 == aa2:
        return 0.0
    pair = (aa1.upper(), aa2.upper())
    rev_pair = (aa2.upper(), aa1.upper())
    if pair in GRANTHAM_MATRIX:
        return float(GRANTHAM_MATRIX[pair])
    elif rev_pair in GRANTHAM_MATRIX:
        return float(GRANTHAM_MATRIX[rev_pair])
    return 0.0


def compute_charge_shift(aa_ancestral: str, aa_target: str) -> float:
    """Compute net charge shift delta (target - ancestral)."""
    q_anc = NET_CHARGE.get(aa_ancestral.upper(), 0.0)
    q_tgt = NET_CHARGE.get(aa_target.upper(), 0.0)
    return q_tgt - q_anc


def compute_hydropathy_shift(aa_ancestral: str, aa_target: str) -> float:
    """Compute Kyte-Doolittle hydropathy shift delta (target - ancestral)."""
    h_anc = KYTE_DOOLITTLE.get(aa_ancestral.upper(), 0.0)
    h_tgt = KYTE_DOOLITTLE.get(aa_target.upper(), 0.0)
    return h_tgt - h_anc


def compute_volume_shift(aa_ancestral: str, aa_target: str) -> float:
    """Compute molecular volume shift delta in Å³ (target - ancestral)."""
    v_anc = MOLECULAR_VOLUME.get(aa_ancestral.upper(), 0.0)
    v_tgt = MOLECULAR_VOLUME.get(aa_target.upper(), 0.0)
    return v_tgt - v_anc


def annotate_physicochemical_shifts(switches_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate switches dataframe with physicochemical property shifts."""
    df = switches_df.copy()
    
    granthams = []
    charges = []
    hydropathies = []
    volumes = []
    domain_names = []

    for _, row in df.iterrows():
        if pd.isna(row["aa_left"]) or pd.isna(row["aa_target"]):
            granthams.append(float("nan"))
            charges.append(float("nan"))
            hydropathies.append(float("nan"))
            volumes.append(float("nan"))
            domain_names.append("Unassigned")
            continue
        anc = str(row["aa_left"])
        tgt = str(row["aa_target"])
        pos = int(row["position"])

        g = get_grantham_distance(anc, tgt)
        q = compute_charge_shift(anc, tgt)
        h = compute_hydropathy_shift(anc, tgt)
        v = compute_volume_shift(anc, tgt)

        d_name = "Unassigned"
        for d, (start, end) in DOMAINS.items():
            if start <= pos <= end:
                d_name = d
                break

        granthams.append(g)
        charges.append(q)
        hydropathies.append(h)
        volumes.append(v)
        domain_names.append(d_name)

    df["grantham_distance"] = granthams
    df["charge_shift"] = charges
    df["hydropathy_shift"] = hydropathies
    df["volume_shift"] = volumes
    df["domain"] = domain_names

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 4: Physicochemical Property Analysis")
    parser.add_argument("--scores", type=Path, default=Path("results/badasp_scoring/raw_node_scores.csv"))
    parser.add_argument("--alignment", type=Path, default=Path("data/interim/IPR019888_trimmed.aln"))
    parser.add_argument("--min-occupancy", type=float, default=0.8)
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("results/initial_run_characterization"))
    args = parser.parse_args()

    plots_dir = args.out_dir / "plots"
    tables_dir = args.out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw node scores from {args.scores}...")
    df = pd.read_csv(args.scores)
    df = df[(df["clade_size_left"] >= args.min_clade_size) & (df["clade_size_right"] >= args.min_clade_size)]

    occupancies = calculate_msa_occupancies(args.alignment)
    df["occupancy"] = df["position"].map(occupancies)
    df_filtered = df[df["occupancy"] >= args.min_occupancy].copy()

    thresholds, clade_categories = calculate_bin_thresholds_999(df_filtered, num_bins=10)
    switches_df = extract_switch_instances(df_filtered, thresholds, clade_categories, event_specific=False)

    print("Annotating physicochemical property shifts...")
    annotated_df = annotate_physicochemical_shifts(switches_df)
    annotated_df.to_csv(tables_dir / "physicochemical_switch_properties.csv", index=False)

    # Summarize per domain
    domain_summary = annotated_df.groupby("domain")[
        ["grantham_distance", "charge_shift", "hydropathy_shift", "volume_shift"]
    ].agg(["mean", "std", "median"]).reset_index()
    domain_summary.to_csv(tables_dir / "domain_physicochemical_summary.csv", index=False)

    # Plot 1: Grantham Distance Violin Plot per Domain
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=annotated_df,
        x="domain",
        y="grantham_distance",
        palette="magma",
        inner="quartile"
    )
    plt.title("Grantham Chemical Distance of 99.9th% Switches across Architectural Domains", fontsize=13, fontweight="bold")
    plt.xlabel("Architectural Domain", fontsize=11)
    plt.ylabel("Grantham Distance (Chemical Severity)", fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "physicochemical_grantham_domain_violin.svg", format="svg")
    plt.savefig(plots_dir / "physicochemical_grantham_domain_violin.png", format="png", dpi=300)
    plt.close()

    # Plot 2: Charge vs Hydropathy Shift Scatter by Domain
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=annotated_df,
        x="hydropathy_shift",
        y="charge_shift",
        hue="domain",
        style="domain",
        s=80,
        alpha=0.8,
        palette="Set2"
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Charge Shift vs Hydropathy Shift of 99.9th% Divergence Switches", fontsize=13, fontweight="bold")
    plt.xlabel("Hydropathy Shift (Δ Kyte-Doolittle)", fontsize=11)
    plt.ylabel("Net Charge Shift (Δ Net Charge)", fontsize=11)
    plt.legend(title="Domain", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "physicochemical_charge_hydropathy_scatter.svg", format="svg")
    plt.savefig(plots_dir / "physicochemical_charge_hydropathy_scatter.png", format="png", dpi=300)
    plt.close()

    print(f"Module 4 completed. Outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
