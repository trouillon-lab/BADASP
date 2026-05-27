#!/usr/bin/env python3
"""Generate a high-resolution ChimeraX script to color structure using the moving average switch counts."""

import json
import logging
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from src.pdb_mapper import PDBMapper

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def _normalize_residue_number(value: object) -> Optional[int]:
    """Normalize mapper outputs to a single integer residue number."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer, float, np.floating, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, tuple) and len(value) >= 2:
        return _normalize_residue_number(value[1])
    if isinstance(value, list):
        for candidate in value:
            normalized = _normalize_residue_number(candidate)
            if normalized is not None:
                return normalized
        return None
    return None

def hex_interpolate(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    c1 = np.array([int(color1[i:i+2], 16) for i in (1, 3, 5)])
    c2 = np.array([int(color2[i:i+2], 16) for i in (1, 3, 5)])
    interp = (1 - t) * c1 + t * c2
    return "#" + "".join(f"{int(round(x)):02x}" for x in interp)

def get_gradient_color(val: float, max_val: float) -> str:
    """Map value to gainsboro -> orange -> crimson gradient."""
    if max_val <= 0:
        return "#dcdcdc"
    t = min(1.0, max(0.0, val / max_val))
    if t == 0:
        return "#dcdcdc"  # Gainsboro
    elif t <= 0.5:
        # Interpolate Gainsboro (#DCDCDC) to Orange (#F28E2B)
        return hex_interpolate("#dcdcdc", "#f28e2b", t * 2)
    else:
        # Interpolate Orange (#F28E2B) to Crimson (#D62728)
        return hex_interpolate("#f28e2b", "#d62728", (t - 0.5) * 2)

def generate_cxc_for_track(track_name: str, csv_filename: str, output_filename: str) -> None:
    csv_path = Path("results/evolutionary_analysis") / csv_filename
    msa_path = Path("data/interim/IPR019888_trimmed.aln")
    pdb_path = Path("data/raw/AF_with_loop.cif")
    output_cxc = Path("results/structural_mapping") / output_filename

    if not csv_path.exists():
        logger.error(f"Enrichment CSV not found: {csv_path}")
        return
    if not msa_path.exists():
        logger.error(f"MSA not found: {msa_path}")
        return
    if not pdb_path.exists():
        logger.error(f"PDB/CIF not found: {pdb_path}")
        return

    logger.info(f"Mapping alignment positions to 3D structure residues for track '{track_name}'...")
    mapper = PDBMapper(pdb_id="AF_with_loop", pdb_file=pdb_path)
    mapping = mapper.map_alignment_to_structure(msa_path)

    df = pd.read_csv(csv_path)
    if "position" not in df.columns or "moving_average_switch_count" not in df.columns:
        logger.error(f"CSV for '{track_name}' must contain 'position' and 'moving_average_switch_count' columns.")
        return

    max_ma = float(df["moving_average_switch_count"].max())
    logger.info(f"[{track_name}] Max moving average value: {max_ma}")

    lines = [
        "del all",
        f"open {pdb_path.absolute()}",
        "view",
        "color protein gainsboro",
        "color nucleic lightsteelblue",
    ]

    colored_count = 0
    # Group residues by their computed hex color to minimize ChimeraX command lines
    color_groups: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        pos = int(row["position"])
        ma_val = float(row["moving_average_switch_count"])
        
        if pos not in mapping:
            continue
            
        resnum = _normalize_residue_number(mapping[pos])
        if resnum is None:
            continue
            
        color = get_gradient_color(ma_val, max_ma)
        # Skip if color is Gainsboro (default background color)
        if color == "#dcdcdc":
            continue
            
        color_groups.setdefault(color, []).append(str(resnum))
        colored_count += 1

    for color, resnums in sorted(color_groups.items()):
        res_str = ",".join(resnums)
        lines.append(f"color /A,B:{res_str} {color}")

    output_cxc.parent.mkdir(parents=True, exist_ok=True)
    with output_cxc.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Successfully generated moving average CXC coloring script for '{track_name}' with {colored_count} colored residues: {output_cxc}")


def main() -> None:
    # 1. Duplications
    generate_cxc_for_track("duplications", "global_architectural_enrichment_duplications.csv", "highlight_moving_average_duplications.cxc")
    # 2. Speciations
    generate_cxc_for_track("speciations", "global_architectural_enrichment_speciations.csv", "highlight_moving_average_speciations.cxc")
    # 3. Combined
    generate_cxc_for_track("combined", "global_architectural_enrichment.csv", "highlight_moving_average_combined.cxc")
    # 4. Legacy compatibility
    generate_cxc_for_track("combined", "global_architectural_enrichment.csv", "highlight_moving_average.cxc")


if __name__ == "__main__":
    main()
