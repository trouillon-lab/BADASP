import tempfile
from pathlib import Path
from scripts.generate_structural_animation import generate_animation_cxc

def test_generate_animation_cxc_creates_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create mock layer directories and cxc files
        layer_01 = tmp_path / "layer_01"
        layer_01.mkdir()
        cxc_file_1 = layer_01 / "layer_01_duplications.cxc"
        cxc_file_1.write_text(
            "open /path/to/2cg4.pdb\n"
            "color /A:18 #BA7373\n"
            "color /A:21 #7A0000\n",
            encoding="utf-8"
        )
        
        layer_02 = tmp_path / "layer_02"
        layer_02.mkdir()
        cxc_file_2 = layer_02 / "layer_02_duplications.cxc"
        cxc_file_2.write_text(
            "open /path/to/2cg4.pdb\n"
            "color /A:30 #BA7373\n",
            encoding="utf-8"
        )
        
        # Run compiler
        output_path = generate_animation_cxc("duplications", tmp_path, wait_frames=10)
        
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        
        assert "Master ChimeraX Animation for Duplications" in content
        assert "AF_with_loop.cif" in content
        assert "del all" in content
        assert "view" in content
        assert "color nucleic lightsteelblue" in content
        assert "movie record size 1920,1080 supersample 3" in content
        assert "# --- LAYER 01 ---" in content
        assert "color protein gainsboro" in content
        assert "color /A:18 #BA7373" in content
        assert "color /A:21 #7A0000" in content
        assert "wait 10" in content
        assert "# --- LAYER 02 ---" in content
        assert "color /A:30 #BA7373" in content
        assert "movie stop" in content
        assert "movie encode" in content
        assert "animate_duplications.gif" in content
