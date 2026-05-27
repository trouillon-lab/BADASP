from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the replacement pipeline."""

    project_root: Path
    data_dir: Path
    results_dir: Path


def default_pipeline_config(project_root: Path | None = None) -> PipelineConfig:
    """Build a default configuration anchored at the repository root."""

    root = project_root or Path(__file__).resolve().parents[2]
    return PipelineConfig(
        project_root=root,
        data_dir=root / "data",
        results_dir=root / "results" / "badasp_next",
    )