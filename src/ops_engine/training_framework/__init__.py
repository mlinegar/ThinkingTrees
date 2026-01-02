"""
Oracle Approximation Training Framework for OPS.

A reusable package for training learned oracle functions that approximate
human judgment in Oracle-Preserving Summarization systems.

Key Components:
- UnifiedTrainingCollector: Aggregates training data from multiple sources
- OracleInferRetrieveRank: Main inference module using Infer-Retrieve-Rank
- Training data sources: NodeLevelHumanSource, FullDocumentLabelSource

Example Usage:
    from ops_engine.training_framework import (
        UnifiedTrainingCollector,
        NodeLevelHumanSource,
        FullDocumentLabelSource,
    )

    # Collect from multiple sources
    collector = UnifiedTrainingCollector()
    collector.add_source(NodeLevelHumanSource(review_queue))
    collector.add_source(FullDocumentLabelSource(threshold=10.0))

    # Get training data
    trainset = collector.get_dspy_trainset(max_examples=50)

    # Statistics
    print(collector.get_statistics())
"""

from .core import (
    ViolationType,
    TrainingExampleLabel,
    UnifiedTrainingExample,
    TrainingDataSource,
    OracleReviewResult,
    # Prediction and verification
    Prediction,
    LawCheckResult,
)

from .config import (
    OracleIRRConfig,
    OptimizationConfig,
    TrainingDataConfig,
    OnlineLearningConfig,
    FrameworkConfig,
)

from .data_sources import (
    NodeLevelHumanSource,
    FullDocumentLabelSource,
    OracleAutoReviewSource,
    UnifiedTrainingCollector,
)

# Retriever for example-based few-shot learning
from .inference import (
    Retriever,
)

from .verification import (
    OracleNodeVerifier,
    NodeVerificationResult,
    TreeVerifier,
)

from .metrics import (
    calibration_error,
    law_compliance_rate,
    overall_compliance_rate,
    # Preferred API - scale-aware metric factories
    create_cached_oracle_metric,
    create_oracle_metric,
    create_merge_metric,
    get_cache_stats,
    metric,
    oracle_score_prediction,
    # Score-centric metric converters
    oracle_as_metric,
    oracle_as_metric_with_feedback,
    # Summarization-specific
    summarization,
)

# Phase 3: Optimization
# High-level optimizer wrapper (used by online_learning.py)
from .optimization import (
    OracleOptimizer,
    OptimizationResult,
    create_optimizer,
)

# Optimizer Registry (preferred API for new code)
from .optimizers import (
    OptimizerRegistry,
    get_optimizer,
    auto_select_optimizer,
    list_optimizers,
    BaseOptimizer,
    AbstractOptimizer,
    # Specific optimizers
    GEPAOptimizer,
    BootstrapOptimizer,
    BootstrapRandomSearchOptimizer,
    MIPROOptimizer,
    # Parallel optimization
    ParallelModuleOptimizer,
    ModuleOptimizationConfig,
    create_parallel_optimizer,
)


# Phase 5: Task Plugins
from .tasks import (
    TaskPlugin,
    AbstractTask,
    TaskRegistry,
    get_task,
    list_tasks,
    OutputType,
    ScaleDefinition,
    LabelDefinition,
    TaskConfig,
    UnifiedTrainingSource,
    UnifiedResult,
    DEFAULT_UNIFIED_RUBRIC,
)


# Preference learning - base infrastructure
from .base_preference import (
    BasePreferenceCollector,
    CandidateInfo,
    PreferenceResult,
    CollectionStatistics,
)
from .preference_engine import (
    PreferenceEngine,
    PreferenceEngineConfig,
    PreferenceDerivationStrategy,
    DEFAULT_GENRM_ENGINE,
    DEFAULT_ORACLE_ENGINE,
)
from .preference import (
    PreferencePair,
    PairwiseJudge,
    PreferenceCollector,
    PreferenceDataset,
    GenerationConfig,
)
from .genrm_preference import (
    GenRMJudge,
    GenRMResult,
    GenRMPreferenceCollector,
)
from .ops_comparison_module import (
    OPSLawComparison,
    OPSComparisonModule,
)
# Labeled tree data structures (formerly "ground truth")
from .labeled_tree import (
    LabeledNode,
    LabeledTree,
    LabeledDataset,
)

__all__ = [
    # Core types
    'ViolationType',
    'TrainingExampleLabel',
    'UnifiedTrainingExample',
    'TrainingDataSource',
    'OracleReviewResult',

    # Prediction and verification
    'Prediction',
    'LawCheckResult',

    # Configuration
    'OracleIRRConfig',
    'OptimizationConfig',
    'TrainingDataConfig',
    'OnlineLearningConfig',
    'FrameworkConfig',

    # Data sources
    'NodeLevelHumanSource',
    'FullDocumentLabelSource',
    'OracleAutoReviewSource',
    'UnifiedTrainingCollector',

    # Retriever
    'Retriever',

    # Verification
    'OracleNodeVerifier',
    'NodeVerificationResult',
    'TreeVerifier',

    # Metrics
    'calibration_error',
    'law_compliance_rate',
    'overall_compliance_rate',
    'create_cached_oracle_metric',
    'get_cache_stats',
    # Score-centric metric converters (preferred)
    'oracle_as_metric',
    'oracle_as_metric_with_feedback',

    # Optimization (Phase 3 - legacy)
    'OracleOptimizer',
    'OptimizationResult',
    'create_optimizer',

    # Optimizer Registry (Phase 3b)
    'OptimizerRegistry',
    'get_optimizer',
    'auto_select_optimizer',
    'list_optimizers',
    'BaseOptimizer',
    'AbstractOptimizer',
    'GEPAOptimizer',
    'BootstrapOptimizer',
    'BootstrapRandomSearchOptimizer',
    'MIPROOptimizer',
    'ParallelModuleOptimizer',
    'ModuleOptimizationConfig',
    'create_parallel_optimizer',

    # Task Plugins (Phase 5)
    'TaskPlugin',
    'AbstractTask',
    'TaskRegistry',
    'get_task',
    'list_tasks',
    'OutputType',
    'ScaleDefinition',
    'LabelDefinition',
    'TaskConfig',
    'UnifiedTrainingSource',
    'UnifiedResult',
    'DEFAULT_UNIFIED_RUBRIC',

    # Preference learning
    'PreferencePair',
    'PairwiseJudge',
    'PreferenceCollector',
    'PreferenceDataset',
    'GenerationConfig',
    'CollectionStatistics',
    'GenRMJudge',
    'GenRMPreferenceCollector',
    'OPSLawComparison',
    'OPSComparisonModule',
    # Labeled tree data structures
    'LabeledNode',
    'LabeledTree',
    'LabeledDataset',
]

# Note: train_oracle() convenience function has been removed.
# Classification-based training has been deprecated in favor of continuous score prediction.
# Task-specific training lives under src/tasks (e.g., src/tasks/manifesto).
