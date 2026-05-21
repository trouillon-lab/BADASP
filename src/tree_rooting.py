import shutil
import subprocess
import warnings
from pathlib import Path

from Bio import Phylo


CANONICAL_MAD_EXECUTABLE = Path("venv/bin/mad.py")


def root_tree(
    input_tree: Path,
    output_tree: Path,
    method: str = "mad",
) -> Path:
    output_tree.parent.mkdir(parents=True, exist_ok=True)

    # RESUME LOGIC: Skip if output already exists and is non-empty
    if output_tree.exists() and output_tree.stat().st_size > 0:
        return output_tree

    if method == "midpoint":
        tree = Phylo.read(str(input_tree), "newick")
        tree.root_at_midpoint()
        Phylo.write(tree, str(output_tree), "newick")
        return output_tree

    if method != "mad":
        raise ValueError(f"Unsupported rooting method: {method}. Expected one of: mad, midpoint")

    mad_script = CANONICAL_MAD_EXECUTABLE
    if not mad_script.exists():
        warnings.warn(
            f"MAD executable '{mad_script}' not found; falling back to midpoint rooting.",
            UserWarning,
            stacklevel=2,
        )
        tree = Phylo.read(str(input_tree), "newick")
        tree.root_at_midpoint()
        Phylo.write(tree, str(output_tree), "newick")
        return output_tree

    result = subprocess.run(
        [str(mad_script), str(input_tree)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    rooted_candidates = [
        input_tree.parent / f"{input_tree.name}.rooted",
        input_tree.with_suffix(f"{input_tree.suffix}.rooted"),
        input_tree.with_name(f"{input_tree.stem}.rooted"),
    ]
    for candidate in rooted_candidates:
        if candidate.exists():
            shutil.copyfile(candidate, output_tree)
            return output_tree

    stdout = "" if result is None else (result.stdout or "")
    if stdout and "(" in stdout:
        output_tree.write_text(stdout.strip() + "\n", encoding="utf-8")
        return output_tree

    raise FileNotFoundError(
        "MAD rooted tree output not found. Expected a '.rooted' file or Newick on stdout."
    )
