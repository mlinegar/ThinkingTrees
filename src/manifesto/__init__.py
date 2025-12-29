"""
Document processing integration for OPS evaluation.

This module provides tools for loading Manifesto Project data,
creating train/test splits, and evaluating RILE score prediction.

New in Phase 3/4: Training framework integration for oracle approximation.
"""

from .constants import RILE_MIN, RILE_MAX, RILE_RANGE
from .data_loader import DocSample, DocDataset, ManifestoSample, ManifestoDataset, create_pilot_dataset
from .rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT
from .ops_pipeline import ManifestoOPSPipeline, SimplePipeline, PipelineConfig, DocResult, ManifestoResult
from .evaluation import ManifestoEvaluator, EvaluationMetrics, save_results, load_results
from .position_oracle import (
    RILESimilarityScorer,  # Alias for create_rile_scorer
    create_rile_scorer,    # Factory for RILE similarity scorer
)

# Training framework integration
from .training_integration import (
    ManifestoTrainingSource,
    TrainableManifestoPipeline,
    create_rile_training_pipeline,
    create_rile_training_collector,
    create_rile_training_example,
)

__all__ = [
    # RILE constants
    "RILE_MIN",
    "RILE_MAX",
    "RILE_RANGE",
    # Data loading (new names)
    "DocSample",
    "DocDataset",
    # Data loading (backwards compat)
    "ManifestoSample",
    "ManifestoDataset",
    "create_pilot_dataset",
    # Rubrics
    "RILE_PRESERVATION_RUBRIC",
    "RILE_TASK_CONTEXT",
    # Pipeline
    "ManifestoOPSPipeline",
    "SimplePipeline",
    "PipelineConfig",
    "DocResult",  # New name
    "ManifestoResult",  # Backwards compat
    # Evaluation
    "ManifestoEvaluator",
    "EvaluationMetrics",
    "save_results",
    "load_results",
    # Position oracles
    "RILESimilarityScorer",
    "create_rile_scorer",
    # Training integration
    "ManifestoTrainingSource",
    "TrainableManifestoPipeline",
    "create_rile_training_pipeline",
    "create_rile_training_collector",
    "create_rile_training_example",
]
