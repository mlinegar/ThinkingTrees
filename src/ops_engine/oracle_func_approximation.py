"""
Oracle Function Approximation - Learned oracle for reviewing flagged nodes.

This module implements a learned DSPy function that approximates the oracle function
for reviewing flagged audit nodes. It is trained on positive (true violations)
and negative (false positives) examples from historical human reviews.

Key components:
- OracleFuncTrainingExample: Training data format with labels
- OracleFuncTrainingCollector: Collects and manages positive/negative examples
- LearnedOracleFunc: DSPy module trained via BootstrapFewShot
- OracleFuncReviewEngine: Orchestrates automated review of flagged nodes
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable

import dspy

from src.core.signatures import OracleFuncApproximation, OracleFuncReviewer
from src.ops_engine.auditor import (
    ReviewQueue, FlaggedItem, ReviewPriority, AuditCheckResult
)

logger = logging.getLogger(__name__)


class ExampleLabel(Enum):
    """Label for training examples."""
    POSITIVE = "positive"  # True violation - confirmed by human review
    NEGATIVE = "negative"  # False positive - flagged but approved by human


@dataclass
class OracleFuncTrainingExample:
    """
    A single training example for the oracle function approximation model.

    Positive examples: Items flagged by audit that were confirmed as true
                      violations by human review (review_result=False)
    Negative examples: Items flagged by audit that were approved by human
                      review as false positives (review_result=True)
    """
    # Input fields
    original_content: str
    summary: str
    rubric: str
    check_type: str
    approx_discrepancy: float

    # Label
    label: ExampleLabel

    # For positive examples: the corrected summary from human review
    corrected_summary: Optional[str] = None

    # Metadata
    item_id: Optional[str] = None
    node_id: Optional[str] = None
    tree_id: Optional[str] = None
    reviewed_at: Optional[str] = None
    human_reasoning: Optional[str] = None

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format for training."""
        is_violation = self.label == ExampleLabel.POSITIVE
        return dspy.Example(
            original_content=self.original_content,
            summary=self.summary,
            rubric=self.rubric,
            check_type=self.check_type,
            approx_discrepancy=self.approx_discrepancy,
            is_true_violation=is_violation,
            confidence=1.0 if is_violation else 0.9,  # Human-labeled examples have high confidence
            corrected_summary=self.corrected_summary or "",
            reasoning=self.human_reasoning or ""
        ).with_inputs("original_content", "summary", "rubric", "check_type", "approx_discrepancy")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'original_content': self.original_content,
            'summary': self.summary,
            'rubric': self.rubric,
            'check_type': self.check_type,
            'approx_discrepancy': self.approx_discrepancy,
            'label': self.label.value,
            'corrected_summary': self.corrected_summary,
            'item_id': self.item_id,
            'node_id': self.node_id,
            'tree_id': self.tree_id,
            'reviewed_at': self.reviewed_at,
            'human_reasoning': self.human_reasoning
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OracleFuncTrainingExample':
        """Deserialize from dictionary."""
        data = dict(data)
        data['label'] = ExampleLabel(data.get('label', 'positive'))
        return cls(**data)

    @classmethod
    def from_flagged_item(
        cls,
        item: FlaggedItem,
        include_auto_reviewed: bool = False
    ) -> Optional['OracleFuncTrainingExample']:
        """
        Create training example from a reviewed FlaggedItem.

        Returns None if the item hasn't been reviewed or is auto-reviewed
        (unless include_auto_reviewed=True).

        Args:
            item: The flagged item to convert
            include_auto_reviewed: If False (default), skip items with
                                   review_source="oracle_func_auto" to prevent
                                   feedback collapse from training on model outputs
        """
        if not item.reviewed:
            return None

        # Skip auto-reviewed items by default to prevent feedback collapse
        if not include_auto_reviewed and getattr(item, 'review_source', 'human') == 'oracle_func_auto':
            return None

        # Determine label based on review result
        # review_result=True means approved (false positive) -> NEGATIVE
        # review_result=False means needs fix (true violation) -> POSITIVE
        label = ExampleLabel.NEGATIVE if item.review_result else ExampleLabel.POSITIVE

        return cls(
            original_content=item.input_a,
            summary=item.input_b,
            rubric=item.rubric,
            check_type=item.check_type,
            approx_discrepancy=item.approx_discrepancy,
            label=label,
            corrected_summary=item.corrected_summary,
            item_id=item.item_id,
            node_id=item.node_id,
            tree_id=item.tree_id,
            reviewed_at=item.reviewed_at,
            human_reasoning=item.review_reasoning
        )


class OracleFuncTrainingCollector:
    """
    Collects and manages training data for the oracle function approximation model.

    Maintains separate lists of positive (true violations) and negative
    (false positives) examples, with methods to balance the dataset and
    export for training.
    """

    def __init__(self):
        self.positive_examples: List[OracleFuncTrainingExample] = []
        self.negative_examples: List[OracleFuncTrainingExample] = []

    def add_example(self, example: OracleFuncTrainingExample) -> None:
        """Add a training example to the appropriate list."""
        if example.label == ExampleLabel.POSITIVE:
            self.positive_examples.append(example)
        else:
            self.negative_examples.append(example)

    def add_positive(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        check_type: str,
        approx_discrepancy: float,
        corrected_summary: str,
        reasoning: Optional[str] = None
    ) -> OracleFuncTrainingExample:
        """Add a positive example (true violation)."""
        example = OracleFuncTrainingExample(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            check_type=check_type,
            approx_discrepancy=approx_discrepancy,
            label=ExampleLabel.POSITIVE,
            corrected_summary=corrected_summary,
            human_reasoning=reasoning,
            reviewed_at=datetime.now().isoformat()
        )
        self.positive_examples.append(example)
        return example

    def add_negative(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        check_type: str,
        approx_discrepancy: float,
        reasoning: Optional[str] = None
    ) -> OracleFuncTrainingExample:
        """Add a negative example (false positive - actually approved)."""
        example = OracleFuncTrainingExample(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            check_type=check_type,
            approx_discrepancy=approx_discrepancy,
            label=ExampleLabel.NEGATIVE,
            human_reasoning=reasoning,
            reviewed_at=datetime.now().isoformat()
        )
        self.negative_examples.append(example)
        return example

    def extract_from_review_queue(self, queue: ReviewQueue) -> Tuple[int, int]:
        """
        Extract training examples from reviewed items in a ReviewQueue.

        Returns:
            Tuple of (positive_count, negative_count) added
        """
        pos_added = 0
        neg_added = 0

        for item in queue.items:
            example = OracleFuncTrainingExample.from_flagged_item(item)
            if example is not None:
                self.add_example(example)
                if example.label == ExampleLabel.POSITIVE:
                    pos_added += 1
                else:
                    neg_added += 1

        return pos_added, neg_added

    def get_balanced_examples(
        self,
        max_examples: Optional[int] = None,
        balance_ratio: float = 1.0
    ) -> List[OracleFuncTrainingExample]:
        """
        Get a balanced dataset of positive and negative examples.

        Args:
            max_examples: Maximum total examples to return
            balance_ratio: Ratio of negative to positive (1.0 = equal)

        Returns:
            List of balanced training examples
        """
        n_positive = len(self.positive_examples)
        n_negative = len(self.negative_examples)

        if n_positive == 0:
            logger.warning("No positive examples available")
            return list(self.negative_examples)
        if n_negative == 0:
            logger.warning("No negative examples available")
            return list(self.positive_examples)

        # Calculate balanced counts
        target_negative = int(n_positive * balance_ratio)
        actual_negative = min(target_negative, n_negative)
        actual_positive = min(n_positive, int(actual_negative / balance_ratio))

        if max_examples:
            total = actual_positive + actual_negative
            if total > max_examples:
                scale = max_examples / total
                actual_positive = int(actual_positive * scale)
                actual_negative = int(actual_negative * scale)

        # Sample from each list
        import random
        sampled_positive = random.sample(self.positive_examples, min(actual_positive, n_positive))
        sampled_negative = random.sample(self.negative_examples, min(actual_negative, n_negative))

        result = sampled_positive + sampled_negative
        random.shuffle(result)
        return result

    def get_all_examples(self) -> List[OracleFuncTrainingExample]:
        """Get all examples without balancing."""
        return self.positive_examples + self.negative_examples

    def get_dspy_examples(
        self,
        balanced: bool = True,
        max_examples: Optional[int] = None
    ) -> List[dspy.Example]:
        """Get examples in DSPy format."""
        if balanced:
            examples = self.get_balanced_examples(max_examples)
        else:
            examples = self.get_all_examples()
            if max_examples:
                examples = examples[:max_examples]
        return [e.to_dspy_example() for e in examples]

    def save(self, filepath: Path) -> None:
        """Save training data to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'saved_at': datetime.now().isoformat(),
            'positive_count': len(self.positive_examples),
            'negative_count': len(self.negative_examples),
            'positive_examples': [e.to_dict() for e in self.positive_examples],
            'negative_examples': [e.to_dict() for e in self.negative_examples]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self)} training examples to {filepath}")

    def load(self, filepath: Path) -> Tuple[int, int]:
        """
        Load training data from JSON file.

        Returns:
            Tuple of (positive_count, negative_count) loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Training data file not found: {filepath}")
            return 0, 0

        with open(filepath) as f:
            data = json.load(f)

        pos_loaded = 0
        neg_loaded = 0

        for item in data.get('positive_examples', []):
            self.positive_examples.append(OracleFuncTrainingExample.from_dict(item))
            pos_loaded += 1

        for item in data.get('negative_examples', []):
            self.negative_examples.append(OracleFuncTrainingExample.from_dict(item))
            neg_loaded += 1

        logger.info(f"Loaded {pos_loaded} positive, {neg_loaded} negative examples from {filepath}")
        return pos_loaded, neg_loaded

    def get_statistics(self) -> Dict[str, Any]:
        """Get training data statistics."""
        all_examples = self.get_all_examples()
        check_types = {}
        for e in all_examples:
            check_types[e.check_type] = check_types.get(e.check_type, 0) + 1

        return {
            'total_examples': len(self),
            'positive_examples': len(self.positive_examples),
            'negative_examples': len(self.negative_examples),
            'balance_ratio': (
                len(self.negative_examples) / len(self.positive_examples)
                if self.positive_examples else 0.0
            ),
            'by_check_type': check_types,
            'avg_discrepancy_positive': (
                sum(e.approx_discrepancy for e in self.positive_examples) / len(self.positive_examples)
                if self.positive_examples else 0.0
            ),
            'avg_discrepancy_negative': (
                sum(e.approx_discrepancy for e in self.negative_examples) / len(self.negative_examples)
                if self.negative_examples else 0.0
            )
        }

    def __len__(self) -> int:
        return len(self.positive_examples) + len(self.negative_examples)


class OracleFuncMetric:
    """
    Metric function for evaluating oracle function approximation predictions.

    Evaluates both the classification accuracy (is_true_violation) and
    optionally the quality of corrected summaries.
    """

    def __init__(
        self,
        classification_weight: float = 0.7,
        correction_weight: float = 0.3,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            classification_weight: Weight for classification accuracy
            correction_weight: Weight for correction quality (positive examples only)
            confidence_threshold: Minimum confidence to consider a prediction valid
        """
        self.classification_weight = classification_weight
        self.correction_weight = correction_weight
        self.confidence_threshold = confidence_threshold
        self.call_count = 0

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """Evaluate a prediction against the example."""
        self.call_count += 1

        # Extract values
        expected_violation = getattr(example, 'is_true_violation', False)
        predicted_violation = getattr(prediction, 'is_true_violation', False)
        confidence = getattr(prediction, 'confidence', 0.5)

        # Classification score
        classification_correct = (expected_violation == predicted_violation)
        classification_score = 1.0 if classification_correct else 0.0

        # Apply confidence penalty
        if confidence < self.confidence_threshold:
            classification_score *= 0.5

        # Correction score (only for positive examples with corrections)
        correction_score = 0.0
        if expected_violation and hasattr(example, 'corrected_summary') and example.corrected_summary:
            predicted_correction = getattr(prediction, 'corrected_summary', '')
            if predicted_correction:
                # Simple word overlap for correction quality
                expected_words = set(example.corrected_summary.lower().split())
                predicted_words = set(predicted_correction.lower().split())
                if expected_words:
                    overlap = len(expected_words & predicted_words) / len(expected_words)
                    correction_score = overlap

        # Weighted combination
        if expected_violation:
            total = (
                self.classification_weight * classification_score +
                self.correction_weight * correction_score
            )
        else:
            total = classification_score

        return total


@dataclass
class OracleFuncConfig:
    """Configuration for the oracle function approximation system."""
    # Training configuration
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 8
    balance_ratio: float = 1.0
    min_training_examples: int = 4

    # Review configuration
    confidence_threshold: float = 0.6
    auto_approve_threshold: float = 0.8
    auto_reject_threshold: float = 0.8

    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("data/oracle_func_checkpoints"))
    training_data_path: Path = field(default_factory=lambda: Path("data/oracle_func_training.json"))


@dataclass
class OracleFuncReviewResult:
    """Result of reviewing a single flagged item with learned oracle function."""
    item_id: str
    is_true_violation: bool
    confidence: float
    corrected_summary: Optional[str]
    reasoning: str
    auto_decided: bool  # Whether confidence was high enough for auto-decision


class LearnedOracleFunc:
    """
    Learned oracle using DSPy BootstrapFewShot for reviewing flagged nodes.

    This class manages the training and application of the oracle function
    approximation model, including:
    - Collecting training data from human reviews
    - Training via DSPy's BootstrapFewShot
    - Reviewing new flagged items
    - Saving/loading trained models
    """

    def __init__(
        self,
        config: Optional[OracleFuncConfig] = None,
        metric: Optional[OracleFuncMetric] = None
    ):
        self.config = config or OracleFuncConfig()
        self.metric = metric or OracleFuncMetric()
        self.training_collector = OracleFuncTrainingCollector()
        self._compiled_module: Optional[dspy.Module] = None
        self._base_module = OracleFuncReviewer()
        self._is_trained = False

    def add_training_example(self, example: OracleFuncTrainingExample) -> None:
        """Add a training example."""
        self.training_collector.add_example(example)

    def add_positive_example(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        check_type: str,
        approx_discrepancy: float,
        corrected_summary: str,
        reasoning: Optional[str] = None
    ) -> OracleFuncTrainingExample:
        """Add a positive (true violation) example."""
        return self.training_collector.add_positive(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            check_type=check_type,
            approx_discrepancy=approx_discrepancy,
            corrected_summary=corrected_summary,
            reasoning=reasoning
        )

    def add_negative_example(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        check_type: str,
        approx_discrepancy: float,
        reasoning: Optional[str] = None
    ) -> OracleFuncTrainingExample:
        """Add a negative (false positive) example."""
        return self.training_collector.add_negative(
            original_content=original_content,
            summary=summary,
            rubric=rubric,
            check_type=check_type,
            approx_discrepancy=approx_discrepancy,
            reasoning=reasoning
        )

    def extract_training_from_queue(self, queue: ReviewQueue) -> Tuple[int, int]:
        """Extract training examples from a review queue."""
        return self.training_collector.extract_from_review_queue(queue)

    def train(
        self,
        teacher_lm: Optional[Any] = None,
        force_retrain: bool = False
    ) -> bool:
        """
        Train the oracle function approximation model using collected examples.

        Args:
            teacher_lm: Optional teacher LM for bootstrap
            force_retrain: Force retraining even if already trained

        Returns:
            True if training succeeded
        """
        if self._is_trained and not force_retrain:
            logger.info("Model already trained. Use force_retrain=True to retrain.")
            return True

        examples = self.training_collector.get_dspy_examples(balanced=True)

        if len(examples) < self.config.min_training_examples:
            logger.warning(
                f"Not enough training examples: {len(examples)} < {self.config.min_training_examples}"
            )
            return False

        try:
            # Configure bootstrap optimizer
            teleprompter = dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                max_rounds=1
            )

            if teacher_lm:
                teleprompter.teacher_lm = teacher_lm

            # Run optimization
            self._compiled_module = teleprompter.compile(
                self._base_module,
                trainset=examples
            )

            self._is_trained = True
            logger.info(f"Successfully trained oracle function approximation with {len(examples)} examples")

            # Save checkpoint
            self._save_checkpoint()

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def review_item(self, item: FlaggedItem) -> OracleFuncReviewResult:
        """
        Review a single flagged item using the learned oracle function.

        Args:
            item: The flagged item to review

        Returns:
            OracleFuncReviewResult with the review decision
        """
        module = self._compiled_module if self._is_trained else self._base_module

        try:
            result = module(
                original_content=item.input_a,
                summary=item.input_b,
                rubric=item.rubric,
                check_type=item.check_type,
                approx_discrepancy=item.approx_discrepancy
            )

            is_violation = result.get('is_true_violation', False)
            confidence = result.get('confidence', 0.5)

            # Determine if we can auto-decide
            auto_decided = False
            if is_violation and confidence >= self.config.auto_reject_threshold:
                auto_decided = True
            elif not is_violation and confidence >= self.config.auto_approve_threshold:
                auto_decided = True

            return OracleFuncReviewResult(
                item_id=item.item_id,
                is_true_violation=is_violation,
                confidence=confidence,
                corrected_summary=result.get('corrected_summary'),
                reasoning=result.get('reasoning', ''),
                auto_decided=auto_decided
            )

        except Exception as e:
            logger.error(f"Review failed for item {item.item_id}: {e}")
            return OracleFuncReviewResult(
                item_id=item.item_id,
                is_true_violation=True,  # Conservative: assume violation on error
                confidence=0.0,
                corrected_summary=None,
                reasoning=f"Review error: {e}",
                auto_decided=False
            )

    def review_queue(
        self,
        queue: ReviewQueue,
        auto_apply: bool = False,
        priority_min: ReviewPriority = ReviewPriority.LOW
    ) -> List[OracleFuncReviewResult]:
        """
        Review all unreviewed items in a queue.

        Args:
            queue: The review queue to process
            auto_apply: If True, automatically apply high-confidence decisions
            priority_min: Minimum priority level to review

        Returns:
            List of review results
        """
        results = []
        batch = queue.get_batch(
            limit=1000,  # Process all
            priority_min=priority_min,
            unreviewed_only=True
        )

        for item in batch:
            result = self.review_item(item)
            results.append(result)

            if auto_apply and result.auto_decided:
                # Apply the decision to the item
                item.reviewed = True
                item.review_result = not result.is_true_violation
                item.review_reasoning = f"[Auto-reviewed by oracle func] {result.reasoning}"
                item.reviewed_at = datetime.now().isoformat()
                item.review_source = "oracle_func_auto"  # Mark as auto-reviewed for training filtering

                if result.is_true_violation and result.corrected_summary:
                    item.corrected_summary = result.corrected_summary

                queue.update_item(item)
                logger.info(f"Auto-applied review for {item.item_id}: violation={result.is_true_violation}")

        return results

    def _save_checkpoint(self) -> Optional[Path]:
        """Save trained model checkpoint."""
        if not self._is_trained or not self._compiled_module:
            return None

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(datetime.now().timestamp())
        checkpoint_path = self.config.checkpoint_dir / f"oracle_func_model_{timestamp}.json"

        try:
            self._compiled_module.save(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            # Fallback: save training data
            self.training_collector.save(
                self.config.checkpoint_dir / f"oracle_func_training_{timestamp}.json"
            )
            return None

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load a trained model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            self._compiled_module = OracleFuncReviewer()
            self._compiled_module.load(str(checkpoint_path))
            self._is_trained = True
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save_training_data(self, filepath: Optional[Path] = None) -> None:
        """Save training data to file."""
        path = filepath or self.config.training_data_path
        self.training_collector.save(path)

    def load_training_data(self, filepath: Optional[Path] = None) -> Tuple[int, int]:
        """Load training data from file."""
        path = filepath or self.config.training_data_path
        return self.training_collector.load(path)

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learned oracle."""
        return {
            'is_trained': self._is_trained,
            'metric_calls': self.metric.call_count,
            'training_data': self.training_collector.get_statistics()
        }


class OracleFuncReviewEngine:
    """
    High-level engine for automated review of flagged nodes using learned oracle function.

    Integrates the LearnedOracleFunc with the auditor's ReviewQueue to provide
    a complete workflow for:
    1. Collecting training data from human reviews
    2. Training the oracle function approximation
    3. Auto-reviewing new flagged items
    4. Generating training examples from auto-reviews for continuous learning
    """

    def __init__(
        self,
        config: Optional[OracleFuncConfig] = None,
        review_queue: Optional[ReviewQueue] = None
    ):
        self.config = config or OracleFuncConfig()
        self.oracle = LearnedOracleFunc(config=self.config)
        self.review_queue = review_queue

    def initialize_from_queue(self, queue: Optional[ReviewQueue] = None) -> Tuple[int, int]:
        """
        Initialize training data from an existing review queue.

        Returns:
            Tuple of (positive_count, negative_count) extracted
        """
        queue = queue or self.review_queue
        if queue is None:
            logger.warning("No review queue provided")
            return 0, 0

        return self.oracle.extract_training_from_queue(queue)

    def train(self, force_retrain: bool = False) -> bool:
        """Train or retrain the oracle function approximation model."""
        return self.oracle.train(force_retrain=force_retrain)

    def review_flagged_nodes(
        self,
        queue: Optional[ReviewQueue] = None,
        auto_apply: bool = True,
        priority_min: ReviewPriority = ReviewPriority.MEDIUM
    ) -> List[OracleFuncReviewResult]:
        """
        Review flagged nodes in the queue using the learned oracle function.

        Args:
            queue: Review queue (uses self.review_queue if not provided)
            auto_apply: Apply high-confidence decisions automatically
            priority_min: Minimum priority to review

        Returns:
            List of review results
        """
        queue = queue or self.review_queue
        if queue is None:
            raise ValueError("No review queue provided")

        if not self.oracle.is_trained:
            logger.warning("Oracle not trained. Results may be lower quality.")

        return self.oracle.review_queue(
            queue=queue,
            auto_apply=auto_apply,
            priority_min=priority_min
        )

    def get_items_for_human_review(
        self,
        queue: Optional[ReviewQueue] = None,
        limit: int = 10
    ) -> List[Tuple[FlaggedItem, OracleFuncReviewResult]]:
        """
        Get items that need human review (low confidence from oracle function).

        Returns items with their oracle function predictions so humans can verify.
        """
        queue = queue or self.review_queue
        if queue is None:
            return []

        # Get unreviewed items
        batch = queue.get_batch(limit=limit * 2, unreviewed_only=True)

        # Review with oracle function but don't auto-apply
        results = []
        for item in batch:
            result = self.oracle.review_item(item)
            if not result.auto_decided:
                results.append((item, result))
            if len(results) >= limit:
                break

        return results

    def continuous_learn(
        self,
        queue: Optional[ReviewQueue] = None,
        retrain_threshold: int = 10
    ) -> bool:
        """
        Extract new training examples and optionally retrain.

        Args:
            queue: Review queue with newly reviewed items
            retrain_threshold: Minimum new examples before retraining

        Returns:
            True if retraining occurred
        """
        queue = queue or self.review_queue
        if queue is None:
            return False

        pos, neg = self.oracle.extract_training_from_queue(queue)
        new_examples = pos + neg

        if new_examples >= retrain_threshold:
            logger.info(f"Retraining with {new_examples} new examples")
            return self.oracle.train(force_retrain=True)

        return False

    def save_state(self) -> None:
        """Save all state (training data and model checkpoint)."""
        self.oracle.save_training_data()
        if self.oracle.is_trained:
            self.oracle._save_checkpoint()

    def load_state(self) -> bool:
        """Load training data and latest checkpoint."""
        self.oracle.load_training_data()

        # Try to load latest checkpoint
        if self.config.checkpoint_dir.exists():
            checkpoints = sorted(self.config.checkpoint_dir.glob("oracle_func_model_*.json"))
            if checkpoints:
                return self.oracle.load_checkpoint(checkpoints[-1])

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self.oracle.get_statistics()
        if self.review_queue:
            stats['queue_statistics'] = self.review_queue.get_statistics()
        return stats


# Convenience functions

def create_oracle_func_reviewer(
    config: Optional[OracleFuncConfig] = None,
    review_queue: Optional[ReviewQueue] = None
) -> OracleFuncReviewEngine:
    """Create a configured OracleFuncReviewEngine."""
    return OracleFuncReviewEngine(config=config, review_queue=review_queue)


def train_oracle_func_from_reviews(
    queue: ReviewQueue,
    config: Optional[OracleFuncConfig] = None
) -> LearnedOracleFunc:
    """
    Convenience function to train oracle function approximation from a review queue.

    Args:
        queue: Review queue with human-reviewed items
        config: Optional configuration

    Returns:
        Trained LearnedOracleFunc
    """
    oracle = LearnedOracleFunc(config=config)
    pos, neg = oracle.extract_training_from_queue(queue)
    logger.info(f"Extracted {pos} positive, {neg} negative examples from queue")

    if oracle.train():
        return oracle
    else:
        logger.warning("Training failed, returning untrained oracle")
        return oracle
