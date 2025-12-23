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
    # Label space abstraction
    LabelSpace,
    CategoricalLabelSpace,
    OrdinalLabelSpace,
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

# Phase 2 imports
from .inference import (
    Retriever,
    OracleClassifySignature,
    OracleViolationSignature,
    OracleClassifier,
    ViolationClassifier,
    RILEClassifier,
)

from .verification import (
    OracleNodeVerifier,
    NodeVerificationResult,
    TreeVerifier,
)

from .metrics import (
    classification_accuracy,
    distance_weighted_accuracy,
    mean_absolute_error,
    within_threshold_accuracy,
    calibration_error,
    law_compliance_rate,
    overall_compliance_rate,
    create_classification_metric,
    create_violation_metric,
    evaluate_classifier,
    EvaluationResult,
    # Score-centric metric converters (new, preferred)
    oracle_as_metric,
    oracle_as_metric_with_feedback,
)

# Phase 3: Optimization (legacy)
from .optimization import (
    OracleOptimizer,
    OptimizationResult,
    create_optimizer,
)

# Phase 3b: New Optimizer Registry (from optimizers submodule)
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

# Phase 4: Online Learning
from .online_learning import (
    OnlineLearningManager,
    OracleReviewResult,
    create_online_manager,
)

# Phase 5: Domain Plugins
from .domains import (
    DomainPlugin,
    AbstractDomain,
    DomainRegistry,
    get_domain,
    list_domains,
    ManifestoDomain,
)

# Phase 6: Bootstrapping
from .bootstrapping import (
    HierarchicalBootstrapper,
    BootstrappedExample,
    ProcessedDocument,
    create_bootstrapped_trainset,
)

# Preference learning
from .preference import (
    PreferencePair,
    PairwiseJudge,
    PreferenceCollector,
    PreferenceDataset,
    GenerationConfig,
)
from .genrm_preference import (
    GenRMJudge,
    GenRMPreferenceCollector,
)
from .oracle_preference import (
    OraclePreferenceCollector,
    OraclePreferenceConfig,
)
from .ops_comparison_module import (
    OPSLawComparison,
    OPSComparisonModule,
)
from .oracle_ground_truth import (
    ChunkGroundTruth,
    ManifestoGroundTruthTree,
    GroundTruthDataset,
)

__all__ = [
    # Core types
    'ViolationType',
    'TrainingExampleLabel',
    'UnifiedTrainingExample',
    'TrainingDataSource',
    'OracleReviewResult',

    # Label space abstraction
    'LabelSpace',
    'CategoricalLabelSpace',
    'OrdinalLabelSpace',

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

    # Inference (Phase 2)
    'Retriever',
    'OracleClassifySignature',
    'OracleViolationSignature',
    'OracleClassifier',
    'ViolationClassifier',
    'RILEClassifier',

    # Verification (Phase 2)
    'OracleNodeVerifier',
    'NodeVerificationResult',
    'TreeVerifier',

    # Metrics (Phase 2)
    'classification_accuracy',
    'distance_weighted_accuracy',
    'mean_absolute_error',
    'within_threshold_accuracy',
    'calibration_error',
    'law_compliance_rate',
    'overall_compliance_rate',
    'create_classification_metric',
    'create_violation_metric',
    'evaluate_classifier',
    'EvaluationResult',
    # Score-centric metric converters (new, preferred)
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

    # Domain Plugins (Phase 5)
    'DomainPlugin',
    'AbstractDomain',
    'DomainRegistry',
    'get_domain',
    'list_domains',
    'ManifestoDomain',

    # Bootstrapping (Phase 6)
    'HierarchicalBootstrapper',
    'BootstrappedExample',
    'ProcessedDocument',
    'create_bootstrapped_trainset',

    # Preference learning
    'PreferencePair',
    'PairwiseJudge',
    'PreferenceCollector',
    'PreferenceDataset',
    'GenerationConfig',
    'GenRMJudge',
    'GenRMPreferenceCollector',
    'OraclePreferenceCollector',
    'OraclePreferenceConfig',
    'OPSLawComparison',
    'OPSComparisonModule',
    'ChunkGroundTruth',
    'ManifestoGroundTruthTree',
    'GroundTruthDataset',

    # Online Learning (Phase 4)
    'OnlineLearningManager',
    'OracleReviewResult',
    'create_online_manager',
    'train_oracle',
]


# =============================================================================
# Convenience Function: train_oracle()
# =============================================================================

def train_oracle(
    collector: UnifiedTrainingCollector,
    label_space: LabelSpace = None,
    config: FrameworkConfig = None,
    checkpoint_path: str = None,
):
    """
    High-level convenience function to train an oracle classifier.

    This is the main entry point for training oracle approximation models.
    It handles:
    - Creating the classifier with appropriate label space
    - Setting up retrieval augmentation
    - Running optimization with DSPy
    - Evaluating the trained model
    - Saving checkpoints

    Args:
        collector: UnifiedTrainingCollector with training data from multiple sources
        label_space: Label space for classification (default: CategoricalLabelSpace for violations)
        config: Framework configuration (uses defaults if None)
        checkpoint_path: Optional path to load existing checkpoint

    Returns:
        Tuple of (trained_classifier, evaluation_result)

    Example:
        from ops_engine.training_framework import (
            train_oracle,
            UnifiedTrainingCollector,
            NodeLevelHumanSource,
            FullDocumentLabelSource,
        )

        # Collect from multiple sources
        collector = UnifiedTrainingCollector()
        collector.add_source(NodeLevelHumanSource(review_queue))
        collector.add_source(FullDocumentLabelSource(rile_results))

        # Train
        classifier, results = train_oracle(collector)

        # Use
        prediction = classifier(original, summary, rubric)
        print(f"Label: {prediction.label}, Confidence: {prediction.confidence}")
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Use defaults
    config = config or FrameworkConfig()
    label_space = label_space or CategoricalLabelSpace.from_violation_types()

    # Create retriever
    retriever = Retriever(model_name=config.oracle_irr.retriever_model_name)

    # Create classifier
    classifier = OracleClassifier(label_space, retriever, config.oracle_irr)

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                classifier.load(str(checkpoint_path))
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

    # Get training data
    trainset = collector.get_dspy_trainset(
        max_examples=config.optimization.max_examples,
        balanced=True,
    )

    if len(trainset) < config.optimization.min_training_examples:
        logger.warning(
            f"Only {len(trainset)} examples available, "
            f"need {config.optimization.min_training_examples} for training"
        )
        return classifier, None

    logger.info(f"Training with {len(trainset)} examples")

    # Create metric
    metric = create_classification_metric(label_space, weighted=label_space.is_ordinal)

    # Optimize
    optimizer = OracleOptimizer(config.optimization)
    classifier = optimizer.optimize(classifier, trainset, metric=metric)

    # Evaluate
    predictions = []
    ground_truth = []
    for ex in trainset[:50]:  # Evaluate on subset for speed
        try:
            pred = classifier(
                original_content=ex.original_content,
                summary=ex.summary,
                rubric=ex.rubric,
            )
            predictions.append(pred)
            ground_truth.append(ex.label)
        except Exception as e:
            logger.debug(f"Evaluation error: {e}")

    result = None
    if predictions:
        result = evaluate_classifier(
            predictions,
            ground_truth,
            label_space,
        )
        logger.info(f"Training complete - Accuracy: {result.accuracy:.3f}")

    # Save checkpoint
    if config.optimization.save_checkpoints:
        optimizer.save_checkpoint(classifier)

    return classifier, result
