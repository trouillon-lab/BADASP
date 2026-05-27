from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig


@dataclass(frozen=True)
class PipelineStage:
    """A named stage in the replacement pipeline."""

    name: str
    description: str


DEFAULT_STAGE_ORDER: Sequence[PipelineStage] = (
    PipelineStage(name="ingest", description="Load and validate source inputs."),
    PipelineStage(name="prepare", description="Normalize inputs for downstream analysis."),
    PipelineStage(name="analyze", description="Run the main analytical workflow."),
    PipelineStage(name="publish", description="Write results to the new output tree."),
)


def build_stage_manifest(config: PipelineConfig) -> list[dict[str, str]]:
    """Return a serializable manifest for the replacement pipeline."""

    return [
        {
            "name": stage.name,
            "description": stage.description,
            "results_dir": str(config.results_dir / stage.name),
        }
        for stage in DEFAULT_STAGE_ORDER
    ]


def ensure_results_root(config: PipelineConfig) -> Path:
    """Return the isolated results root for the new pipeline."""

    return config.results_dir