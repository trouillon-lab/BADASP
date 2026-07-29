#!/usr/bin/env python
import tarfile
from pathlib import Path


def package_rec_check(root: Path = None) -> Path:
    """Package rec_check data for Euler cluster submission.

    Returns the path to the created tar.gz bundle.
    """
    if root is None:
        root = Path(__file__).resolve().parents[1]

    rec_check_alerax = root / "data/rec_check/alerax"
    package_dir = root / "data/rec_check/euler_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a relative-path families.txt for remote Euler run
    remote_families = package_dir / "IPR019888.families.txt"
    remote_families.write_text(
        "[FAMILIES]\n"
        "- IPR019888\n"
        "gene_tree = IPR019888_distribution.nwk\n"
        "mapping = IPR019888.treerecs_mapping.link\n",
        encoding="utf-8"
    )

    # 2. Source files from AleRax rec_check directory
    dist_tree = rec_check_alerax / "IPR019888_distribution.nwk"
    mapping_link = rec_check_alerax / "IPR019888.treerecs_mapping.link"
    species_tree = rec_check_alerax / "IPR019888_species_tree.nwk"

    # 3. Create sbatch run script for Euler
    sbatch_script = package_dir / "run_alerax_rec_check.sh"
    sbatch_script.write_text(
        "#!/bin/bash\n"
        "#SBATCH -J rec_check_alerax\n"
        "#SBATCH -n 1\n"
        "#SBATCH -c 4\n"
        "#SBATCH --mem-per-cpu=16G\n"
        "#SBATCH --time=120:00:00\n"
        "#SBATCH -o rec_check_alerax_%j.out\n"
        "#SBATCH -e rec_check_alerax_%j.err\n\n"
        "# Load modules if needed on Euler\n"
        "# module load gcc/11.4.0 cmake\n\n"
        'echo "Starting AleRax rec_check job on $(hostname) at $(date)..."\n\n'
        "# Run AleRax\n"
        "alerax -f IPR019888.families.txt \\\n"
        "       -s IPR019888_species_tree.nwk \\\n"
        "       -p output_rec_check \\\n"
        "       --prune-species-tree\n\n"
        'echo "AleRax finished at $(date)!"\n',
        encoding="utf-8"
    )
    sbatch_script.chmod(0o755)

    # 4. Build tar.gz bundle
    tar_path = root / "data/rec_check/rec_check_euler_bundle.tar.gz"
    print(f"Creating tarball bundle at {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(dist_tree, arcname="rec_check_euler/IPR019888_distribution.nwk")
        tar.add(mapping_link, arcname="rec_check_euler/IPR019888.treerecs_mapping.link")
        tar.add(species_tree, arcname="rec_check_euler/IPR019888_species_tree.nwk")
        tar.add(remote_families, arcname="rec_check_euler/IPR019888.families.txt")
        tar.add(sbatch_script, arcname="rec_check_euler/run_alerax_rec_check.sh")

    print(f"Done! Created tarball size: {tar_path.stat().st_size / (1024*1024):.2f} MB")
    return tar_path


if __name__ == "__main__":
    package_rec_check()
