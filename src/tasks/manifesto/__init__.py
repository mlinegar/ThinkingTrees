"""
Manifesto RILE Scoring Task.

This module provides the RILE (Left-Right) political positioning task implementation.
RILE scores range from -100 (far left) to +100 (far right).

Usage:
    from src.tasks.manifesto import ManifestoTask, RILE_SCALE

    task = ManifestoTask()
    rubric = task.create_rubric()
    metric = task.create_metric()
"""

from .task import ManifestoTask, RILE_SCALE, ManifestoDomain
from .constants import (
    RILE_RANGE,
    RILE_MIN,
    RILE_MAX,
)
from .rubrics import (
    RILE_PRESERVATION_RUBRIC,
    RILE_TASK_CONTEXT,
)
from .oracle import (
    RILEPositionOracle,
    compute_rile_score,
)

__all__ = [
    # Main task
    "ManifestoTask",
    "RILE_SCALE",

    # Constants
    "RILE_RANGE",
    "RILE_MIN",
    "RILE_MAX",

    # Rubrics
    "RILE_PRESERVATION_RUBRIC",
    "RILE_TASK_CONTEXT",

    # Oracle
    "RILEPositionOracle",
    "compute_rile_score",

    # Backward compatibility
    "ManifestoDomain",
]
