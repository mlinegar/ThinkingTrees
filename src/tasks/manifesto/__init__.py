"""
Manifesto RILE Scoring Task.

This is the canonical location for all manifesto-related code.
RILE scores range from -100 (far left) to +100 (far right).

Usage:
    from src.tasks.manifesto import ManifestoTask, RILE_SCALE

    task = ManifestoTask()
    rubric = task.create_rubric()
    metric = task.create_metric()

Data loading:
    from src.tasks.manifesto import ManifestoDataset, ManifestoDataLoader

    loader = ManifestoDataLoader()
    samples = loader.get_temporal_split()

Pipeline components:
    from src.tasks.manifesto import ManifestoPipelineWithStrategy

    pipeline = ManifestoPipelineWithStrategy(judge=judge)
    result = pipeline(text="...")
"""

from .task import ManifestoTask, RILE_SCALE
from .constants import (
    RILE_RANGE,
    RILE_MIN,
    RILE_MAX,
)
from .rubrics import (
    RILE_PRESERVATION_RUBRIC,
    RILE_TASK_CONTEXT,
)

# Oracle (moved from src/manifesto/position_oracle.py)
from .oracle import create_rile_scorer

# Data loading (moved from src/manifesto/data_loader.py)
from .data_loader import (
    ManifestoDataset,
    ManifestoDataLoader,
    ManifestoSample,
    create_pilot_dataset,
)

# Summarizer (moved from src/manifesto/dspy_summarizer.py)
from .summarizer import (
    LeafSummarizer,
    MergeSummarizer,
)

# Pipeline
from .pipeline import (
    # Signatures
    RILESummarize,
    RILEMerge,
    RILEScoreSignature,
    # Modules
    ManifestoSummarizer,
    ManifestoMerger,
    ManifestoScorer,
    StrategyCompatibleSummarizer,
    StrategyCompatibleMerger,
    # Pipelines
    ManifestoPipeline,
    ManifestoPipelineWithStrategy,
    # Training helpers
    create_training_examples,
    rile_metric,
    is_placeholder,
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
    "create_rile_scorer",

    # Data loading
    "ManifestoDataset",
    "ManifestoDataLoader",
    "ManifestoSample",
    "create_pilot_dataset",

    # Summarizers
    "LeafSummarizer",
    "MergeSummarizer",

    # Pipeline signatures
    "RILESummarize",
    "RILEMerge",
    "RILEScoreSignature",

    # Pipeline modules
    "ManifestoSummarizer",
    "ManifestoMerger",
    "ManifestoScorer",
    "StrategyCompatibleSummarizer",
    "StrategyCompatibleMerger",

    # Full pipelines
    "ManifestoPipeline",
    "ManifestoPipelineWithStrategy",

    # Training helpers
    "create_training_examples",
    "rile_metric",
    "is_placeholder",
]
