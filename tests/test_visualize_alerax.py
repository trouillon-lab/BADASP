import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import shutil
from unittest.mock import patch
from scripts.visualize_alerax import (
    parse_event_counts,
    parse_model_parameters,
    parse_transfers,
    main
)

@pytest.fixture
def mock_alerax_dir():
    """Create a temporary directory structure mimicking AleRax outputs."""
    temp_dir = tempfile.mkdtemp()
    
    # Create reconciliations/all
    all_dir = os.path.join(temp_dir, "reconciliations", "all")
    os.makedirs(all_dir, exist_ok=True)
    
    # Write 3 mock sample event count files
    for i in range(3):
        content = f"S:{9000 + i}\nSL:{19000 - i}\nD:{1300 + i}\nDL:0\nT:{10000 + i*10}\nTL:{7000 - i}\nL:0\nLeaf:21641\n"
        with open(os.path.join(all_dir, f"IPR019888_eventCounts_{i}.txt"), "w") as f:
            f.write(content)
            
    # Create model_parameters
    param_dir = os.path.join(temp_dir, "model_parameters")
    os.makedirs(param_dir, exist_ok=True)
    param_content = "node D L T\n2565781 0.0664735 0.823578 0.533322\n3101447 0.0664735 0.823578 0.533322\n"
    with open(os.path.join(param_dir, "model_parameters.txt"), "w") as f:
        f.write(param_content)
        
    # Create reconciliations/summaries
    summary_dir = os.path.join(temp_dir, "reconciliations", "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    # Using IPR019888_meanTransfers.txt (which is what AleRax produces in summaries)
    transfer_content = "2986804 1579316 2.81\n2043170 652676 1.64\n3402707 Node_1 1.5\n"
    with open(os.path.join(summary_dir, "IPR019888_meanTransfers.txt"), "w") as f:
        f.write(transfer_content)
        
    yield temp_dir
    
    shutil.rmtree(temp_dir)

def test_parse_event_counts(mock_alerax_dir):
    df = parse_event_counts(mock_alerax_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {"S", "SL", "D", "DL", "T", "TL", "L", "Leaf"}
    assert df.loc[0, "S"] == 9000
    assert df.loc[1, "D"] == 1301
    assert df.loc[2, "T"] == 10020

def test_parse_model_parameters(mock_alerax_dir):
    rates = parse_model_parameters(mock_alerax_dir)
    assert isinstance(rates, dict)
    assert set(rates.keys()) == {"D", "L", "T"}
    assert pytest.approx(rates["D"], 1e-6) == 0.0664735
    assert pytest.approx(rates["L"], 1e-6) == 0.823578
    assert pytest.approx(rates["T"], 1e-6) == 0.533322

def test_parse_transfers(mock_alerax_dir):
    df = parse_transfers(mock_alerax_dir)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {"donor", "recipient", "frequency"}
    assert df.loc[0, "donor"] == "2986804"
    assert df.loc[0, "recipient"] == "1579316"
    assert pytest.approx(df.loc[0, "frequency"], 1e-2) == 2.81

def test_main_execution(mock_alerax_dir):
    output_dir = os.path.join(mock_alerax_dir, "plots")
    
    # Call main with arguments
    test_args = [
        "visualize_alerax.py",
        "--input_dir", mock_alerax_dir,
        "--output_dir", output_dir,
        "--format", "svg"
    ]
    
    with patch("sys.argv", test_args):
        main()
        
    # Check if files were generated
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "global_event_profile.svg"))
    assert os.path.exists(os.path.join(output_dir, "reconciliation_variance.svg"))
    assert os.path.exists(os.path.join(output_dir, "hgt_highway_network.svg"))
    assert os.path.exists(os.path.join(output_dir, "dtl_parameter_rates.svg"))
