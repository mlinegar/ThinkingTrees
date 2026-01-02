"""
DEPRECATED: Import from src.tasks.manifesto instead.

This module exists only for backwards compatibility and will be removed.
All manifesto-related code has been consolidated into src/tasks/manifesto/.

Migration:
    # Old (deprecated)
    from src.manifesto import ManifestoTask, create_rile_scorer

    # New (canonical)
    from src.tasks.manifesto import ManifestoTask, create_rile_scorer
"""

import warnings

warnings.warn(
    "Importing from src.manifesto is deprecated. "
    "Use src.tasks.manifesto instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from canonical location for backwards compatibility
from src.tasks.manifesto import (
    # Task
    ManifestoTask,
    RILE_SCALE,
    # Constants
    RILE_MIN,
    RILE_MAX,
    RILE_RANGE,
    # Rubrics
    RILE_PRESERVATION_RUBRIC,
    RILE_TASK_CONTEXT,
    # Oracle
    create_rile_scorer,
    # Data loading
    ManifestoDataset,
    ManifestoDataLoader,
    ManifestoSample,
    create_pilot_dataset,
    # Summarizers
    LeafSummarizer,
    MergeSummarizer,
    # Pipeline
    ManifestoPipeline,
    ManifestoPipelineWithStrategy,
    create_training_examples,
    rile_metric,
)

__all__ = [
    "ManifestoTask",
    "RILE_SCALE",
    "RILE_MIN",
    "RILE_MAX",
    "RILE_RANGE",
    "RILE_PRESERVATION_RUBRIC",
    "RILE_TASK_CONTEXT",
    "create_rile_scorer",
    "ManifestoDataset",
    "ManifestoDataLoader",
    "ManifestoSample",
    "create_pilot_dataset",
    "LeafSummarizer",
    "MergeSummarizer",
    "ManifestoPipeline",
    "ManifestoPipelineWithStrategy",
    "create_training_examples",
    "rile_metric",
]
