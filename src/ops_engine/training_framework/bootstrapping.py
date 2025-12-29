"""
Hierarchical bootstrapping for generating training examples.

This module provides utilities for generating summary-level training examples
from document-level labels. It uses a combined approach:
1. Oracle-based labeling: Use a score predictor to label intermediate summaries
2. Document-level error: Use prediction error to identify good/bad summaries
3. Weighted combination: Combine both signals with configurable weights
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy

from .core import (
    TrainingExampleLabel,
    UnifiedTrainingExample,
    ViolationType,
)

logger = logging.getLogger(__name__)


@dataclass
class BootstrappedExample:
    """A training example generated through bootstrapping."""

    # Source information
    document_id: str
    node_id: str
    node_type: str  # 'leaf', 'merge'

    # Content
    original_content: str
    summary: str
    rubric: str

    # Labels
    label: TrainingExampleLabel
    confidence: float

    # Diagnostic info
    oracle_score: Optional[float] = None
    document_error: Optional[float] = None
    oracle_weight: float = 0.0
    error_weight: float = 0.0

    # Context
    expected_score: Optional[float] = None
    parent_context: Optional[str] = None

    def to_unified_example(self) -> UnifiedTrainingExample:
        """Convert to UnifiedTrainingExample format."""
        return UnifiedTrainingExample(
            example_id=f"{self.document_id}:{self.node_id}",
            source_type="bootstrapped",
            original_content=self.original_content,
            summary=self.summary,
            rubric=self.rubric,
            context={
                'document_id': self.document_id,
                'node_id': self.node_id,
                'node_type': self.node_type,
                'oracle_score': self.oracle_score,
                'document_error': self.document_error,
                'expected_score': self.expected_score,
                'parent_context': self.parent_context,
            },
            label=self.label,
            violation_type=ViolationType.SUFFICIENCY if self.label == TrainingExampleLabel.POSITIVE else ViolationType.NONE,
            confidence=self.confidence,
        )

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format."""
        is_violation = self.label == TrainingExampleLabel.POSITIVE

        return dspy.Example(
            original_content=self.original_content,
            summary=self.summary,
            rubric=self.rubric,
            is_violation=is_violation,
            confidence=self.confidence,
            node_type=self.node_type,
        ).with_inputs("original_content", "summary", "rubric")


@dataclass
class ProcessedDocument:
    """
    A document processed through the OPS pipeline.

    Contains the tree structure and node summaries for bootstrapping.
    """

    document_id: str
    ground_truth_label: float  # Document-level ground truth (e.g., RILE score)
    predicted_label: float     # Model's prediction
    prediction_error: float    # abs(predicted - ground_truth)
    rubric: str

    # Tree structure
    tree_nodes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_good_document(self) -> bool:
        """Whether this document was well-summarized (low error)."""
        return self.prediction_error < 10.0

    @property
    def is_bad_document(self) -> bool:
        """Whether this document lost information (high error)."""
        return self.prediction_error >= 20.0


class HierarchicalBootstrapper:
    """
    Generate summary-level training examples from document-level labels.

    This bootstrapper uses a combined approach:
    1. **Oracle labeling**: Run the oracle on intermediate summaries to check
       if they preserve the document-level signal.
    2. **Error-based labeling**: Documents with low prediction error have
       "good" summaries, documents with high error have "bad" summaries.
    3. **Weighted combination**: Final label confidence is weighted by both
       oracle assessment and document-level error.

    Example:
        bootstrapper = HierarchicalBootstrapper(
            oracle_weight=0.6,
            error_weight=0.4,
            error_threshold_high=20.0,
            error_threshold_low=10.0,
        )

        examples = bootstrapper.bootstrap_examples(
            documents=processed_documents,
            oracle=oracle_classifier,
        )
    """

    def __init__(
        self,
        oracle_weight: float = 0.6,
        error_weight: float = 0.4,
        error_threshold_high: float = 20.0,
        error_threshold_low: float = 10.0,
        min_confidence: float = 0.5,
        max_examples_per_document: int = 10,
    ):
        """
        Initialize bootstrapper.

        Args:
            oracle_weight: Weight for oracle-based labeling (0-1)
            error_weight: Weight for error-based labeling (0-1)
            error_threshold_high: Error above which is considered bad
            error_threshold_low: Error below which is considered good
            min_confidence: Minimum confidence to include example
            max_examples_per_document: Max examples to extract per document
        """
        self.oracle_weight = oracle_weight
        self.error_weight = error_weight
        self.error_threshold_high = error_threshold_high
        self.error_threshold_low = error_threshold_low
        self.min_confidence = min_confidence
        self.max_examples_per_document = max_examples_per_document

        # Normalize weights
        total = self.oracle_weight + self.error_weight
        self.oracle_weight = self.oracle_weight / total
        self.error_weight = self.error_weight / total

    def bootstrap_examples(
        self,
        documents: List[ProcessedDocument],
        oracle: Optional[Any] = None,
    ) -> List[BootstrappedExample]:
        """
        Bootstrap training examples from processed documents.

        Args:
            documents: List of processed documents with tree structures
            oracle: Optional score predictor for labeling (any callable)

        Returns:
            List of bootstrapped examples
        """
        all_examples = []

        for doc in documents:
            try:
                examples = self._process_document(doc, oracle)
                all_examples.extend(examples[:self.max_examples_per_document])
            except Exception as e:
                logger.warning(f"Failed to process document {doc.document_id}: {e}")
                continue

        logger.info(
            f"Bootstrapped {len(all_examples)} examples from {len(documents)} documents"
        )

        return all_examples

    def _process_document(
        self,
        doc: ProcessedDocument,
        oracle: Optional[Any],
    ) -> List[BootstrappedExample]:
        """Process a single document to extract training examples."""
        examples = []

        # Compute error-based signal
        error_signal = self._compute_error_signal(doc.prediction_error)

        for node in doc.tree_nodes:
            # Skip if no summary
            if 'summary' not in node:
                continue

            # Get oracle signal if available
            oracle_signal = None
            oracle_score = None
            if oracle is not None:
                try:
                    oracle_signal, oracle_score = self._compute_oracle_signal(
                        node, doc, oracle
                    )
                except Exception as e:
                    logger.debug(f"Oracle signal failed for node {node.get('id', 'unknown')}: {e}")

            # Combine signals
            label, confidence = self._combine_signals(
                oracle_signal=oracle_signal,
                error_signal=error_signal,
                doc_error=doc.prediction_error,
            )

            # Skip low confidence
            if confidence < self.min_confidence:
                continue

            # Create example
            example = BootstrappedExample(
                document_id=doc.document_id,
                node_id=node.get('id', 'unknown'),
                node_type=node.get('type', 'unknown'),
                original_content=node.get('original_content', node.get('content', '')),
                summary=node['summary'],
                rubric=doc.rubric,
                label=label,
                confidence=confidence,
                oracle_score=oracle_score,
                document_error=doc.prediction_error,
                oracle_weight=self.oracle_weight,
                error_weight=self.error_weight,
                expected_score=doc.ground_truth_label,
            )
            examples.append(example)

        return examples

    def _compute_error_signal(self, error: float) -> Optional[Tuple[TrainingExampleLabel, float]]:
        """
        Compute label signal from document error.

        Returns:
            Tuple of (label, confidence) or None if ambiguous
        """
        if error >= self.error_threshold_high:
            # High error = bad summary = POSITIVE (violation)
            # Confidence increases with error
            confidence = min(0.95, 0.6 + (error - self.error_threshold_high) / 50)
            return (TrainingExampleLabel.POSITIVE, confidence)

        elif error <= self.error_threshold_low:
            # Low error = good summary = NEGATIVE (no violation)
            # Confidence increases as error approaches 0
            confidence = min(0.95, 0.6 + (self.error_threshold_low - error) / 20)
            return (TrainingExampleLabel.NEGATIVE, confidence)

        else:
            # Ambiguous - return None
            return None

    def _compute_oracle_signal(
        self,
        node: Dict[str, Any],
        doc: ProcessedDocument,
        oracle: Any,
    ) -> Tuple[Optional[Tuple[TrainingExampleLabel, float]], Optional[float]]:
        """
        Compute label signal from oracle assessment.

        Checks if the summary preserves the document's ideological positioning.

        Returns:
            Tuple of ((label, confidence), oracle_score) or (None, score)
        """
        summary = node['summary']
        original = node.get('original_content', node.get('content', ''))

        # Run oracle on summary
        try:
            result = oracle(
                original_content=original,
                summary=summary,
                rubric=doc.rubric,
            )

            # Get prediction and compare to ground truth
            predicted_score = getattr(result, 'predicted_label', None)
            if predicted_score is None:
                return None, None

            # Compare to document ground truth
            expected = doc.ground_truth_label
            deviation = abs(predicted_score - expected)

            # High deviation = summary doesn't preserve signal
            if deviation >= 20:
                label = TrainingExampleLabel.POSITIVE
                confidence = min(0.9, 0.5 + deviation / 50)
            elif deviation <= 10:
                label = TrainingExampleLabel.NEGATIVE
                confidence = min(0.9, 0.5 + (10 - deviation) / 20)
            else:
                # Ambiguous
                return None, predicted_score

            return (label, confidence), predicted_score

        except Exception as e:
            logger.debug(f"Oracle failed: {e}")
            return None, None

    def _combine_signals(
        self,
        oracle_signal: Optional[Tuple[TrainingExampleLabel, float]],
        error_signal: Optional[Tuple[TrainingExampleLabel, float]],
        doc_error: float,
    ) -> Tuple[TrainingExampleLabel, float]:
        """
        Combine oracle and error signals into final label.

        Args:
            oracle_signal: (label, confidence) from oracle or None
            error_signal: (label, confidence) from error or None
            doc_error: Raw document error

        Returns:
            Tuple of (final_label, final_confidence)
        """
        signals = []

        if oracle_signal is not None:
            signals.append((oracle_signal[0], oracle_signal[1], self.oracle_weight))

        if error_signal is not None:
            signals.append((error_signal[0], error_signal[1], self.error_weight))

        if not signals:
            # No signals - use error heuristic
            if doc_error >= 15:
                return TrainingExampleLabel.POSITIVE, 0.5
            else:
                return TrainingExampleLabel.NEGATIVE, 0.5

        # Weighted vote
        positive_score = 0.0
        negative_score = 0.0
        total_weight = 0.0

        for label, confidence, weight in signals:
            weighted = confidence * weight
            if label == TrainingExampleLabel.POSITIVE:
                positive_score += weighted
            else:
                negative_score += weighted
            total_weight += weight

        # Normalize and decide
        if total_weight > 0:
            positive_score /= total_weight
            negative_score /= total_weight

        if positive_score > negative_score:
            return TrainingExampleLabel.POSITIVE, positive_score
        else:
            return TrainingExampleLabel.NEGATIVE, negative_score


def create_bootstrapped_trainset(
    bootstrapped_examples: List[BootstrappedExample],
    max_examples: int = 100,
    balance: bool = True,
) -> List[dspy.Example]:
    """
    Convert bootstrapped examples to DSPy trainset.

    Args:
        bootstrapped_examples: List of BootstrappedExample
        max_examples: Maximum examples to return
        balance: Whether to balance positive/negative

    Returns:
        List of DSPy Examples
    """
    if balance:
        positives = [e for e in bootstrapped_examples if e.label == TrainingExampleLabel.POSITIVE]
        negatives = [e for e in bootstrapped_examples if e.label == TrainingExampleLabel.NEGATIVE]

        # Sort by confidence
        positives.sort(key=lambda x: x.confidence, reverse=True)
        negatives.sort(key=lambda x: x.confidence, reverse=True)

        # Interleave
        per_class = max_examples // 2
        selected = positives[:per_class] + negatives[:per_class]
    else:
        # Sort by confidence and take top
        bootstrapped_examples.sort(key=lambda x: x.confidence, reverse=True)
        selected = bootstrapped_examples[:max_examples]

    return [e.to_dspy_example() for e in selected]
