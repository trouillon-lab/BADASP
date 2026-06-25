"""New BADASP pipeline namespace.

This package is the starting point for the replacement pipeline and stays
separate from the legacy modules in `src/`.
"""

from .config import PipelineConfig, default_pipeline_config

__all__ = ["PipelineConfig", "default_pipeline_config"]