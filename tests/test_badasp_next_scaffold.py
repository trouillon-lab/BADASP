from pathlib import Path

from src.badasp_next import PipelineConfig, default_pipeline_config
from src.badasp_next.pipeline import build_stage_manifest


def test_default_pipeline_config_isolated_results_root() -> None:
    config = default_pipeline_config(Path("/tmp/badasp"))

    assert isinstance(config, PipelineConfig)
    assert config.results_dir == Path("/tmp/badasp/results/badasp_next")


def test_stage_manifest_points_into_new_namespace() -> None:
    config = default_pipeline_config(Path("/tmp/badasp"))
    manifest = build_stage_manifest(config)

    assert [stage["name"] for stage in manifest] == ["ingest", "prepare", "analyze", "publish"]
    assert all("results/badasp_next" in stage["results_dir"] for stage in manifest)