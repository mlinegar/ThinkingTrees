"""
Training Framework Integration for Manifesto RILE Scoring.

This module bridges the oracle approximation training framework with the
Manifesto Project RILE scoring pipeline, enabling:

1. Training data collection from:
   - Full document RILE labels (ground truth from Manifesto Project)
   - OPS audit failures (from tree verification)
   - Human reviews (from ReviewQueue)

2. Oracle classifier for RILE prediction at tree nodes

3. OPS law verification with RILE-appropriate aggregation

Example:
    from src.manifesto.training_integration import (
        create_rile_training_pipeline,
        ManifestoTrainingSource,
    )

    # Create pipeline with training support
    pipeline = create_rile_training_pipeline(config)

    # Process manifestos
    for sample in dataset:
        result = pipeline.process_manifesto(sample)

    # Get training data and train oracle
    collector = pipeline.get_training_collector()
    classifier, metrics = train_rile_oracle(collector)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import dspy

from src.ops_engine.training_framework import (
    # Core types
    UnifiedTrainingExample,
    TrainingExampleLabel,
    ViolationType,
    LabelSpace,
    OrdinalLabelSpace,
    Prediction,
    LawCheckResult,
    # Data collection
    TrainingDataSource,
    UnifiedTrainingCollector,
    FullDocumentLabelSource,
    # Inference & verification
    OracleClassifier,
    Retriever,
    OracleNodeVerifier,
    TreeVerifier,
    # Optimization
    OracleOptimizer,
    train_oracle,
    # Metrics
    evaluate_classifier,
    create_classification_metric,
    EvaluationResult,
    # Config
    FrameworkConfig,
    OracleIRRConfig,
    OptimizationConfig,
)

from src.ops_engine.scoring import normalize_error_to_score

from .constants import RILE_MIN, RILE_MAX, RILE_RANGE
from .data_loader import ManifestoSample, ManifestoDataset
from .ops_pipeline import ManifestoResult, PipelineConfig
from .rubrics import RILE_TASK_CONTEXT
from .signatures import RILEScorer

logger = logging.getLogger(__name__)


# =============================================================================
# RILE Label Space
# =============================================================================

def create_rile_label_space(bin_size: float = 10.0) -> OrdinalLabelSpace:
    """
    Create RILE label space for oracle classification.

    Uses the OrdinalLabelSpace from training_framework with RILE-specific
    descriptions for retrieval augmentation.

    Args:
        bin_size: Size of each bin (default 10 = 21 bins from -100 to +100)

    Returns:
        Configured OrdinalLabelSpace for RILE
    """
    return OrdinalLabelSpace.for_rile(bin_size=bin_size)


# =============================================================================
# Manifesto Training Data Source
# =============================================================================

@dataclass
class RILEPredictionResult:
    """Result of a RILE prediction from manifesto processing."""
    manifesto_id: str
    ground_truth_rile: float
    predicted_rile: Optional[float]
    summary: str
    original_text: str


class ManifestoTrainingSource(TrainingDataSource):
    """
    Training data source from Manifesto Project results.

    Extracts training examples from processed manifestos:
    - Full document labels: Ground truth RILE → discretized label
    - Error analysis: High-error predictions become training examples
    - Tree nodes: Each summarization step can provide training data
    """

    def __init__(
        self,
        label_space: OrdinalLabelSpace,
        error_threshold_high: float = 20.0,  # RILE points
        error_threshold_low: float = 10.0,
        rubric: str = "",
    ):
        """
        Initialize the training source.

        Args:
            label_space: RILE label space for discretization
            error_threshold_high: Above this error → positive example (violation)
            error_threshold_low: Below this error → negative example (good)
            rubric: Task rubric for examples
        """
        self.label_space = label_space
        self.error_threshold_high = error_threshold_high
        self.error_threshold_low = error_threshold_low
        self.rubric = rubric or RILE_TASK_CONTEXT

        self._results: List[ManifestoResult] = []
        self._processed_count = 0

    def add_result(self, result: ManifestoResult) -> None:
        """Add a processed manifesto result."""
        self._results.append(result)

    def add_results(self, results: List[ManifestoResult]) -> None:
        """Add multiple processed manifesto results."""
        self._results.extend(results)

    @property
    def source_name(self) -> str:
        return "manifesto_rile"

    @property
    def source_confidence(self) -> float:
        """Ground truth labels are highly reliable."""
        return 0.95

    def get_examples(self) -> List[UnifiedTrainingExample]:
        """
        Extract training examples from manifesto results.

        Strategy:
        - High prediction error → POSITIVE (violation - info lost)
        - Low prediction error → NEGATIVE (good summary)
        - Mid-range errors → skip (ambiguous)
        """
        examples = []

        for result in self._results:
            if result.error is not None or result.predicted_rile is None:
                continue

            error = abs(result.predicted_rile - result.ground_truth_rile)

            # Determine label based on prediction error
            if error >= self.error_threshold_high:
                # High error = info was lost
                label = TrainingExampleLabel.POSITIVE
                violation_type = ViolationType.SUFFICIENCY
                confidence = min(0.95, 0.7 + error / 100)  # Higher error = higher confidence
            elif error <= self.error_threshold_low:
                # Low error = summary preserved info
                label = TrainingExampleLabel.NEGATIVE
                violation_type = ViolationType.NONE
                confidence = min(0.95, 0.7 + (self.error_threshold_low - error) / 20)
            else:
                # Mid-range error = ambiguous, skip
                continue

            # Get the discretized RILE label
            rile_label = self.label_space.nearest_label(result.ground_truth_rile)

            example = UnifiedTrainingExample(
                example_id=f"manifesto_{result.manifesto_id}",
                source_type=self.source_name,
                original_content=f"[{result.party_name} - {result.country} {result.year}]",  # Metadata as proxy
                summary=result.final_summary,
                rubric=self.rubric,
                context={
                    'manifesto_id': result.manifesto_id,
                    'country': result.country,
                    'year': result.year,
                    'party': result.party_name,
                    'ground_truth_rile': result.ground_truth_rile,
                    'predicted_rile': result.predicted_rile,
                    'prediction_error': error,
                    'discretized_label': rile_label,
                },
                label=label,
                violation_type=violation_type,
                human_reasoning=(
                    f"Ground truth RILE: {result.ground_truth_rile:.1f}, "
                    f"Predicted: {result.predicted_rile:.1f}, "
                    f"Error: {error:.1f} points"
                ),
                confidence=confidence,
            )
            examples.append(example)

        self._processed_count = len(examples)
        return examples

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the training source."""
        errors = [
            abs(r.predicted_rile - r.ground_truth_rile)
            for r in self._results
            if r.predicted_rile is not None and r.error is None
        ]

        return {
            'total_results': len(self._results),
            'processed_examples': self._processed_count,
            'high_error_count': sum(1 for e in errors if e >= self.error_threshold_high),
            'low_error_count': sum(1 for e in errors if e <= self.error_threshold_low),
            'mean_error': sum(errors) / len(errors) if errors else 0,
            'error_threshold_high': self.error_threshold_high,
            'error_threshold_low': self.error_threshold_low,
        }


# =============================================================================
# RILE Oracle Classifier
# =============================================================================

class RILEOracleClassifier(OracleClassifier):
    """
    Specialized oracle classifier for RILE score prediction.

    Uses the ordinal label space with RILE-specific features:
    - Distance-weighted accuracy (close guesses get partial credit)
    - Political context retrieval
    - Left/right indicator analysis
    """

    def __init__(
        self,
        bin_size: float = 10.0,
        retriever: Optional[Retriever] = None,
        config: Optional[OracleIRRConfig] = None,
    ):
        """
        Initialize RILE classifier.

        Args:
            bin_size: RILE bin size (10 = 21 bins)
            retriever: Optional retriever for augmentation
            config: Configuration
        """
        label_space = create_rile_label_space(bin_size)
        super().__init__(label_space, retriever, config)
        self.bin_size = bin_size

    def predict_rile(
        self,
        text: str,
        rubric: str = RILE_TASK_CONTEXT,
    ) -> Tuple[float, float, str]:
        """
        Predict RILE score for text.

        Args:
            text: Political text to score
            rubric: Task context

        Returns:
            Tuple of (predicted_rile, confidence, reasoning)
        """
        # For direct RILE prediction, we classify the text
        # using the same text as both "original" and "summary"
        pred = self(
            original_content=text,
            summary=text,
            rubric=rubric,
        )

        try:
            rile_value = float(pred.label)
        except ValueError:
            rile_value = 0.0

        return rile_value, pred.confidence, pred.reasoning


# =============================================================================
# RILE Node Verifier
# =============================================================================

class RILENodeVerifier(OracleNodeVerifier):
    """
    OPS law verifier specialized for RILE preservation.

    Checks that summarization preserves political position information
    at each node in the OPS tree.
    """

    def __init__(
        self,
        classifier: RILEOracleClassifier,
        rile_threshold: float = 10.0,
    ):
        """
        Initialize RILE verifier.

        Args:
            classifier: RILE oracle classifier
            rile_threshold: Maximum acceptable RILE drift (in points)
        """
        super().__init__(classifier, tolerance=rile_threshold / RILE_RANGE)  # Normalized
        self.rile_threshold = rile_threshold

    def _labels_equivalent(self, label_a: str, label_b: str) -> bool:
        """Check if two RILE labels are within threshold."""
        try:
            diff = abs(float(label_a) - float(label_b))
            return diff <= self.rile_threshold
        except ValueError:
            return label_a == label_b


# =============================================================================
# Training Pipeline
# =============================================================================

def create_rile_training_collector(
    results: List[ManifestoResult],
    bin_size: float = 10.0,
    error_threshold_high: float = 20.0,
    error_threshold_low: float = 10.0,
) -> UnifiedTrainingCollector:
    """
    Create a training data collector from manifesto results.

    Args:
        results: Processed manifesto results
        bin_size: RILE discretization bin size
        error_threshold_high: Above this = violation
        error_threshold_low: Below this = good

    Returns:
        Configured UnifiedTrainingCollector
    """
    label_space = create_rile_label_space(bin_size)

    source = ManifestoTrainingSource(
        label_space=label_space,
        error_threshold_high=error_threshold_high,
        error_threshold_low=error_threshold_low,
    )
    source.add_results(results)

    collector = UnifiedTrainingCollector()
    collector.add_source(source)

    return collector


def train_rile_oracle(
    collector: UnifiedTrainingCollector,
    bin_size: float = 10.0,
    config: Optional[FrameworkConfig] = None,
) -> Tuple[RILEOracleClassifier, Optional[EvaluationResult]]:
    """
    Train a RILE oracle classifier from collected training data.

    Args:
        collector: Training data collector
        bin_size: RILE bin size
        config: Framework configuration

    Returns:
        Tuple of (trained_classifier, evaluation_result)
    """
    config = config or FrameworkConfig()
    label_space = create_rile_label_space(bin_size)

    # Create retriever
    retriever = Retriever(model_name=config.oracle_irr.retriever_model_name)

    # Create classifier
    classifier = RILEOracleClassifier(
        bin_size=bin_size,
        retriever=retriever,
        config=config.oracle_irr,
    )

    # Get training data
    trainset = collector.get_dspy_trainset(
        max_examples=config.optimization.max_examples,
        balanced=True,
    )

    if len(trainset) < config.optimization.min_training_examples:
        logger.warning(
            f"Only {len(trainset)} training examples, "
            f"need {config.optimization.min_training_examples}"
        )
        return classifier, None

    logger.info(f"Training RILE oracle with {len(trainset)} examples")

    # Create distance-weighted metric for ordinal space
    metric = create_classification_metric(label_space, weighted=True)

    # Optimize
    optimizer = OracleOptimizer(config.optimization)
    classifier = optimizer.optimize(classifier, trainset, metric=metric)

    # Evaluate
    predictions = []
    ground_truth = []
    for ex in trainset[:50]:
        try:
            pred = classifier(
                original_content=ex.original_content,
                summary=ex.summary,
                rubric=ex.rubric,
            )
            predictions.append(pred)
            # Use discretized label from context if available
            label = ex.context.get('discretized_label', ex.label)
            ground_truth.append(str(label))
        except Exception as e:
            logger.debug(f"Evaluation error: {e}")

    result = None
    if predictions:
        result = evaluate_classifier(predictions, ground_truth, label_space)
        logger.info(
            f"Training complete - Accuracy: {result.accuracy:.3f}, "
            f"Weighted: {result.weighted_accuracy:.3f}"
        )

    return classifier, result


# =============================================================================
# Integration with ManifestoOPSPipeline
# =============================================================================

class TrainableManifestoPipeline:
    """
    Extended ManifestoOPSPipeline with training framework integration.

    Collects training data during processing and supports oracle optimization.
    """

    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        training_config: Optional[FrameworkConfig] = None,
        bin_size: float = 10.0,
    ):
        """
        Initialize trainable pipeline.

        Args:
            pipeline_config: Manifesto pipeline configuration
            training_config: Training framework configuration
            bin_size: RILE discretization bin size
        """
        from .ops_pipeline import ManifestoOPSPipeline

        self.pipeline_config = pipeline_config or PipelineConfig()
        self.training_config = training_config or FrameworkConfig()
        self.bin_size = bin_size

        # Core pipeline
        self._pipeline = ManifestoOPSPipeline(self.pipeline_config)

        # Training support
        self.label_space = create_rile_label_space(bin_size)
        self.training_source = ManifestoTrainingSource(
            label_space=self.label_space,
            error_threshold_high=self.training_config.training_data.error_threshold_high,
            error_threshold_low=self.training_config.training_data.error_threshold_low,
        )

        # Results tracking
        self._results: List[ManifestoResult] = []

    def process_manifesto(
        self,
        sample: ManifestoSample,
        run_baseline: Optional[bool] = None,
    ) -> ManifestoResult:
        """
        Process manifesto and collect training data.

        Args:
            sample: Manifesto sample
            run_baseline: Whether to run baseline scoring

        Returns:
            ManifestoResult
        """
        result = self._pipeline.process_manifesto(sample, run_baseline)
        self._results.append(result)
        self.training_source.add_result(result)
        return result

    def process_batch(
        self,
        samples: List[ManifestoSample],
        run_baseline: Optional[bool] = None,
    ) -> List[ManifestoResult]:
        """Process multiple manifestos."""
        results = []
        for sample in samples:
            result = self.process_manifesto(sample, run_baseline)
            results.append(result)
        return results

    def get_training_collector(self) -> UnifiedTrainingCollector:
        """Get training data collector with all collected examples."""
        collector = UnifiedTrainingCollector()
        collector.add_source(self.training_source)
        return collector

    def get_results(self) -> List[ManifestoResult]:
        """Get all processed results."""
        return self._results

    def get_review_queue(self):
        """Get the review queue from underlying pipeline."""
        return self._pipeline.get_review_queue()

    def train_oracle(self) -> Tuple[RILEOracleClassifier, Optional[EvaluationResult]]:
        """
        Train RILE oracle from collected data.

        Returns:
            Tuple of (classifier, evaluation_result)
        """
        collector = self.get_training_collector()
        return train_rile_oracle(
            collector,
            bin_size=self.bin_size,
            config=self.training_config,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline and training statistics."""
        return {
            'processed_manifestos': len(self._results),
            'training_source': self.training_source.get_statistics(),
            'errors': sum(1 for r in self._results if r.error is not None),
            'successful': sum(1 for r in self._results if r.error is None),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_rile_training_pipeline(
    pipeline_config: Optional[PipelineConfig] = None,
    training_config: Optional[FrameworkConfig] = None,
    bin_size: float = 10.0,
) -> TrainableManifestoPipeline:
    """
    Create a trainable Manifesto OPS pipeline.

    Args:
        pipeline_config: Pipeline configuration
        training_config: Training framework configuration
        bin_size: RILE bin size

    Returns:
        TrainableManifestoPipeline instance
    """
    return TrainableManifestoPipeline(
        pipeline_config=pipeline_config,
        training_config=training_config,
        bin_size=bin_size,
    )


def quick_train_from_results(
    results: List[ManifestoResult],
    bin_size: float = 10.0,
) -> Tuple[RILEOracleClassifier, Optional[EvaluationResult]]:
    """
    Quick training from existing manifesto results.

    Args:
        results: Processed manifesto results
        bin_size: RILE bin size

    Returns:
        Tuple of (classifier, evaluation_result)
    """
    collector = create_rile_training_collector(results, bin_size)
    return train_rile_oracle(collector, bin_size)


# =============================================================================
# Summarization Training Data (for Two-Step Iterative Optimization)
# =============================================================================

@dataclass
class SummarizationTrainingExample:
    """
    Training example for optimizing summarization prompts.

    Used in the two-step iterative optimization:
    1. Train oracle classifier on current summaries
    2. Optimize summarizers using oracle + human feedback as metric

    The example contains the original text chunk and ground truth RILE
    score from the full document. During optimization, the summarizer
    generates a summary which is then evaluated by the oracle.
    """

    example_id: str
    original_text: str
    rubric: str
    ground_truth_rile: float
    human_score: Optional[float] = None  # From review queue if available
    human_feedback: Optional[str] = None
    manifesto_id: Optional[str] = None
    chunk_idx: Optional[int] = None

    def to_dspy_example(self) -> dspy.Example:
        """
        Convert to DSPy Example for training.

        The example uses original_text and rubric as inputs,
        with ground_truth_rile and human_score as context for the metric.
        """
        return dspy.Example(
            original_text=self.original_text,
            content=self.original_text,  # Alias for compatibility
            rubric=self.rubric,
            ground_truth_rile=self.ground_truth_rile,
            human_score=self.human_score,
            manifesto_id=self.manifesto_id,
        ).with_inputs('original_text', 'rubric', 'content')


@dataclass
class MergeTrainingExample:
    """
    Training example for optimizing merge summarization prompts.

    Used to train the merger module that combines two summaries.
    """

    example_id: str
    left_summary: str
    right_summary: str
    rubric: str
    ground_truth_rile: float
    human_score: Optional[float] = None
    manifesto_id: Optional[str] = None
    level: int = 0  # Tree level where merge occurred

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example for training."""
        return dspy.Example(
            left_summary=self.left_summary,
            right_summary=self.right_summary,
            rubric=self.rubric,
            ground_truth_rile=self.ground_truth_rile,
            human_score=self.human_score,
        ).with_inputs('left_summary', 'right_summary', 'rubric')


def collect_summarization_training_data(
    results: List[ManifestoResult],
    rubric: str = RILE_TASK_CONTEXT,
    review_queue: Optional[Any] = None,
) -> List[SummarizationTrainingExample]:
    """
    Collect training data for summarization optimization from manifesto results.

    Extracts chunks and their document-level RILE scores as training examples.
    Optionally incorporates human feedback from review queue.

    Args:
        results: Processed manifesto results (with chunks populated)
        rubric: The rubric to use for summarization
        review_queue: Optional ReviewQueue for human feedback

    Returns:
        List of SummarizationTrainingExample
    """
    examples = []

    for result in results:
        if result.error is not None:
            continue

        # Get chunks from result
        chunks = getattr(result, 'chunks', [])
        if not chunks:
            # If no chunks stored, create a single example from the text
            # This is a fallback for results processed without DSPy
            continue

        for i, chunk in enumerate(chunks):
            # Check for human feedback
            human_score = None
            human_feedback = None
            if review_queue:
                try:
                    review = review_queue.get_review(
                        result.manifesto_id,
                        chunk_idx=i
                    )
                    if review:
                        human_score = getattr(review, 'score', None)
                        human_feedback = getattr(review, 'feedback', None)
                except Exception:
                    pass

            example = SummarizationTrainingExample(
                example_id=f"{result.manifesto_id}_chunk_{i}",
                original_text=chunk,
                rubric=rubric,
                ground_truth_rile=result.ground_truth_rile,
                human_score=human_score,
                human_feedback=human_feedback,
                manifesto_id=result.manifesto_id,
                chunk_idx=i,
            )
            examples.append(example)

    logger.info(f"Collected {len(examples)} summarization training examples")
    return examples


def collect_merge_training_data(
    results: List[ManifestoResult],
    rubric: str = RILE_TASK_CONTEXT,
) -> List[MergeTrainingExample]:
    """
    Collect training data for merge optimization from manifesto results.

    Creates merge examples from pairs of leaf summaries.
    For each document with 2+ chunks, creates merge examples.

    Args:
        results: Processed manifesto results (with leaf_summaries populated)
        rubric: The rubric to use for merging

    Returns:
        List of MergeTrainingExample
    """
    examples = []

    for result in results:
        if result.error is not None:
            continue

        # Get leaf summaries from result
        leaf_summaries = getattr(result, 'leaf_summaries', [])
        if len(leaf_summaries) < 2:
            # Need at least 2 summaries to merge
            continue

        # Create merge examples from adjacent pairs
        for i in range(0, len(leaf_summaries) - 1, 2):
            left_summary = leaf_summaries[i]
            right_summary = leaf_summaries[i + 1] if i + 1 < len(leaf_summaries) else ""

            if not right_summary:
                continue

            example = MergeTrainingExample(
                example_id=f"{result.manifesto_id}_merge_{i}",
                left_summary=left_summary,
                right_summary=right_summary,
                rubric=rubric,
                ground_truth_rile=result.ground_truth_rile,
                manifesto_id=result.manifesto_id,
                level=1,  # First merge level
            )
            examples.append(example)

    logger.info(f"Collected {len(examples)} merge training examples")
    return examples


def create_oracle_trainset(
    results: List[ManifestoResult],
    bin_size: float = 10.0,
) -> List[dspy.Example]:
    """
    Create DSPy trainset for oracle classifier from manifesto results.

    Args:
        results: Processed manifesto results
        bin_size: RILE bin size for label discretization

    Returns:
        List of DSPy Example objects for oracle training
    """
    label_space = create_rile_label_space(bin_size)
    examples = []

    for result in results:
        if result.error is not None or not result.final_summary:
            continue

        # Discretize the RILE score to a label
        rile_label = label_space.nearest_label(result.ground_truth_rile)

        example = dspy.Example(
            original_content=f"[{result.party_name} - {result.country} {result.year}]",
            summary=result.final_summary,
            rubric=RILE_TASK_CONTEXT,
            label=rile_label,
            ground_truth_rile=result.ground_truth_rile,
        ).with_inputs('original_content', 'summary', 'rubric')

        examples.append(example)

    logger.info(f"Created {len(examples)} oracle training examples")
    return examples


# =============================================================================
# Training Example Creation
# =============================================================================

def create_rile_training_example(
    result: ManifestoResult,
    include_metadata: bool = True,
) -> dspy.Example:
    """
    Create DSPy training example from manifesto result.

    This is a simplified way to create training examples with all
    the fields needed for RILE metrics.

    Args:
        result: Processed ManifestoResult
        include_metadata: Whether to include manifesto_id, country, year

    Returns:
        dspy.Example with fields for RILE metrics

    Example:
        result = pipeline.process_manifesto(sample)
        example = create_rile_training_example(result)
        # Use with create_metric() from ops_engine.training_framework.metrics
    """
    fields = {
        # Text fields (for similarity/prediction scoring)
        'original_text': result.final_summary,
        'original_content': result.final_summary,
        'summary': result.final_summary,
        'rubric': RILE_TASK_CONTEXT,

        # RILE scores
        'ground_truth_rile': result.ground_truth_rile,
        'predicted_rile': result.predicted_rile,

        # Direction (derived from RILE)
        'direction': _rile_to_direction(result.ground_truth_rile),
    }

    if include_metadata:
        fields['manifesto_id'] = result.manifesto_id
        fields['country'] = result.country
        fields['year'] = result.year
        fields['party_name'] = result.party_name

    return dspy.Example(**fields).with_inputs('original_text', 'rubric')


def _rile_to_direction(rile: float) -> str:
    """Convert RILE score to direction label."""
    if rile is None:
        return "center"
    if rile < -20:
        return "left"
    elif rile > 20:
        return "right"
    else:
        return "center"
