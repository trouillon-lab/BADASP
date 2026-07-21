import sys
from pathlib import Path
import subprocess
from ete3 import Tree

def test_root_and_map_tree(tmp_path: Path) -> None:
    unrooted_tree = tmp_path / "test_unrooted.tree"
    output_tree = tmp_path / "test_rooted.tree"

    # Write a simple unrooted tree with named internal nodes
    # Leaves: SeqA, SeqB, SeqC
    unrooted_tree.write_text("((SeqA:0.1,SeqB:0.2)Node1:0.3,SeqC:0.4);\n", encoding="utf-8")

    # Run root_and_map_tree.py script
    script_path = Path(__file__).resolve().parents[1] / "scripts/root_and_map_tree.py"
    assert script_path.exists(), f"Script not found at {script_path}"

    cmd = [
        sys.executable,
        str(script_path),
        "--unrooted-tree", str(unrooted_tree),
        "--output-tree", str(output_tree)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)

    assert output_tree.exists(), "Rooted tree output file was not created"
    
    # Load output tree and verify that Node1 and Node0 are mapped correctly!
    t = Tree(str(output_tree), format=1)
    
    # Get internal node names
    internal_names = [node.name for node in t.traverse() if not node.is_leaf()]
    print("Internal node names in rooted tree:", internal_names)
    
    # Verify that Node1 exists and has mapped correctly
    assert "Node1" in internal_names, "Node1 was not mapped to the rooted tree"
