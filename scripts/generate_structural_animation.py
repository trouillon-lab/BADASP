#!/usr/bin/env python3
"""
Generate ChimeraX master animation scripts (.cxc) compiling layers 1-20.
"""

import sys
from pathlib import Path

def generate_animation_cxc(track: str, base_dir: Path, wait_frames: int = 25) -> Path:
    base_dir = Path(base_dir)
    output_path = base_dir / f"animate_{track}.cxc"
    
    # Locate all layer .cxc files sorted sequentially
    layer_dirs = sorted(base_dir.glob("layer_*"))
    if not layer_dirs:
        print(f"Error: No layer directories found under {base_dir}", file=sys.stderr)
        sys.exit(1)
        
    cxc_files = []
    for ldir in layer_dirs:
        layer_num = ldir.name.split("_")[-1]
        cxc_file = ldir / f"layer_{layer_num}_{track}.cxc"
        if cxc_file.exists():
            cxc_files.append((layer_num, cxc_file))
            
    if not cxc_files:
        print(f"Error: No .cxc files found for track '{track}'", file=sys.stderr)
        sys.exit(1)
        
    pdb_path = Path(__file__).resolve().parent.parent / "data/raw/AF_with_loop.cif"
    pdb_line = f"open {pdb_path}"
    
    lines = [
        f"# Master ChimeraX Animation for {track.capitalize()} layers 1-20",
        "del all",  # Close any previously loaded models/objects
        pdb_line,
        "delete /C:1-8,33-40 /D:1-8,33-40",  # Shorten DNA by 20% on both ends
        "set bgColor white",
        "lighting soft",
        "lighting shadows false",
        "lighting depthCue false",
        "graphics silhouettes true color black width 4",
        "material dull",
        "show cartoon",
        "hide atoms",
        "color protein gainsboro",
        "color nucleic lightsteelblue",  # Highlight the DNA molecule beautifully
        "view",  # Center the structure and the DNA complex
        "",
        "# Start recording movie at high resolution with supersampling",
        "movie record size 1920,1080 supersample 3",
        ""
    ]
    
    for idx, (layer_num, cxc_file) in enumerate(cxc_files):
        lines.append(f"# --- LAYER {layer_num} ---")
        lines.append("color protein gainsboro")  # reset background color
        
        # Extract only the color commands from this layer file
        color_commands = []
        for line in cxc_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("color ") and not line.endswith("gainsboro"):
                color_commands.append(line)
        
        if color_commands:
            lines.extend(color_commands)
        else:
            lines.append("# No mapped switched residues in this layer")
            
        # Hold frame
        lines.append(f"wait {wait_frames}")
        lines.append("")
        
    # Hold longer on the last frame
    lines.append(f"wait {wait_frames * 2}")
    lines.append("")
    lines.append("# Stop and encode movie to GIF")
    lines.append("movie stop")
    
    gif_filename = base_dir / f"animate_{track}.gif"
    lines.append(f"movie encode {gif_filename.resolve()} format gif")
    
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Generated {output_path}")
    return output_path

if __name__ == "__main__":
    base_results_dir = Path("results/structural_mapping")
    tracks = ["duplications", "speciations", "combined"]
    
    for trk in tracks:
        generate_animation_cxc(trk, base_results_dir)
