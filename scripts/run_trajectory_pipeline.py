#!/usr/bin/env python
import os
import subprocess
from pathlib import Path
import shutil

def main():
    print("=== STARTING PHASE 7 AUTOMATION PIPELINE ===")
    
    # 1. Create subdirectories
    evo_dir = Path("results/evolutionary_analysis")
    dup_dir = evo_dir / "duplications"
    spec_dir = evo_dir / "speciations"
    timeline_dir = evo_dir / "timeline"
    
    for d in [dup_dir, spec_dir, timeline_dir]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created/verified subdirectory: {d}")

    # 2. Run trajectory clustering for duplications (with --optimize-k)
    print("\n>>> Running Trajectory Clustering for DUPLICATIONS...")
    cmd_dup = [
        "venv/bin/python", "src/trajectory_analysis.py",
        "--track", "duplications",
        "--optimize-k"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(cmd_dup, env=env, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR running duplications trajectory clustering: {res.stderr}")
        return

    # 3. Run trajectory clustering for speciations (with --optimize-k)
    print("\n>>> Running Trajectory Clustering for SPECIATIONS...")
    cmd_spec = [
        "venv/bin/python", "src/trajectory_analysis.py",
        "--track", "speciations",
        "--optimize-k"
    ]
    res = subprocess.run(cmd_spec, env=env, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR running speciations trajectory clustering: {res.stderr}")
        return

    # 4. Run evolutionary_analysis.py (to build layer summary, layer timeline, absolute time timeline/dendrogram)
    print("\n>>> Running Evolutionary and Chronological Analysis...")
    cmd_evo = [
        "venv/bin/python", "src/evolutionary_analysis.py",
        "--tree", "results/topological_clustering/mad_rooted.tree",
        "--output-dir", "results/evolutionary_analysis"
    ]
    res = subprocess.run(cmd_evo, env=env, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR running evolutionary analysis: {res.stderr}")
        return

    # 5. Run scripts/regenerate_stale_plots.py (to build relative time timeline in timeline/)
    print("\n>>> Running Stale Plot Regeneration...")
    cmd_regen = [
        "venv/bin/python", "scripts/regenerate_stale_plots.py"
    ]
    res = subprocess.run(cmd_regen, env=env, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"ERROR running stale plot regeneration: {res.stderr}")
        return

    # 6. Purge stale legacy/orphaned files from root results/evolutionary_analysis/
    print("\n>>> Purging stale/orphaned legacy files from root results/evolutionary_analysis/...")
    legacy_files = [
        "switch_trajectory_heatmap_combined.svg",
        "switch_trajectory_heatmap_duplications.svg",
        "switch_trajectory_heatmap_speciations.svg",
        "switch_trajectory_profiles_combined.svg",
        "switch_trajectory_profiles_duplications.svg",
        "switch_trajectory_profiles_speciations.svg",
        "switch_trajectory_dendrogram_combined.svg",
        "switch_trajectory_dendrogram_duplications.svg",
        "switch_trajectory_dendrogram_speciations.svg",
        "switch_trajectory_domain_enrichment_combined.csv",
        "switch_trajectory_domain_enrichment_duplications.csv",
        "switch_trajectory_domain_enrichment_speciations.csv",
        "switch_trajectory_spatial_cohesion_combined.csv",
        "switch_trajectory_spatial_cohesion_duplications.csv",
        "switch_trajectory_spatial_cohesion_speciations.csv",
        "switch_trajectories_combined.csv",
        "switch_trajectories_duplications.csv",
        "switch_trajectories_speciations.csv",
        "clustering_parameter_sweep_combined.csv",
        "clustering_parameter_sweep_combined.svg",
        "clustering_parameter_sweep_duplications.csv",
        "clustering_parameter_sweep_duplications.svg",
        "clustering_parameter_sweep_speciations.csv",
        "clustering_parameter_sweep_speciations.svg",
        "chronological_switch_timeline.svg",
        "chronological_dendrogram_switches.svg",
    ]
    
    purged_count = 0
    for f_name in legacy_files:
        f_path = evo_dir / f_name
        if f_path.exists():
            f_path.unlink()
            purged_count += 1
            print(f"  Purged legacy file: {f_path}")
            
    print(f"Purged {purged_count} legacy files.")
    print("\n=== PHASE 7 AUTOMATION PIPELINE EXECUTED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
