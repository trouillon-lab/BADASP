import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that require live network access (e.g. RCSB PDB downloads).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: mark test as requiring live network access (skipped by default; use --run-network to enable).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="requires live network — pass --run-network to enable")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
