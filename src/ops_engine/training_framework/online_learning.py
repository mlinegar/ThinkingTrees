"""
Online Learning Manager for Oracle Approximation.

This module implements continuous learning with human-in-the-loop support,
adapting from oracle_func_approximation.py patterns to prevent feedback collapse.

Key features:
- Automatic training data collection from reviews
- Threshold-based retraining triggers
- Feedback collapse prevention (filters oracle-auto-reviewed items)
- High-confidence auto-review with human escalation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple

import dspy

from .core import (
    UnifiedTrainingExample,
    TrainingExampleLabel,
    ViolationType,
    Prediction,
)
from .config import OptimizationConfig, OnlineLearningConfig
from .data_sources import UnifiedTrainingCollector, NodeLevelHumanSource
from .optimization import OracleOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Review Result
# =============================================================================

@dataclass
class OracleReviewResult:
    """Result of an oracle review on an item."""

    item_id: str
    is_violation: bool
    violation_type: ViolationType
    confidence: float
    reasoning: str
    corrected_summary: Optional[str] = None
    review_source: str = "oracle_classifier_auto"  # Distinct from human

    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'is_violation': self.is_violation,
            'violation_type': self.violation_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'corrected_summary': self.corrected_summary,
            'review_source': self.review_source,
        }


# =============================================================================
# Online Learning Manager
# =============================================================================

class OnlineLearningManager:
    """
    Manages continuous learning cycle for oracle approximation.

    Workflow:
    1. Audit pipeline flags items for review
    2. Human or oracle reviews items
    3. Extract training data from reviews (with feedback collapse prevention)
    4. Retrain when threshold met
    5. Auto-review new items with improved oracle
    """

    def __init__(
        self,
        classifier: dspy.Module,
        config: Optional[OnlineLearningConfig] = None,
        review_queue: Optional[Any] = None,  # ReviewQueue from auditor
    ):
        """
        Initialize online learning manager.

        Args:
            classifier: The score predictor module to manage
            config: Online learning configuration
            review_queue: Optional ReviewQueue for human feedback
        """
        self.classifier = classifier
        self.config = config or OnlineLearningConfig()
        self.review_queue = review_queue

        # State tracking
        self.iteration_count = 0
        self.examples_since_last_train = 0
        self.total_examples_collected = 0
        self.auto_reviews_applied = 0
        self.human_reviews_collected = 0

        # Training data collection
        self.training_collector = UnifiedTrainingCollector()

        # Optimization
        self.optimizer = OracleOptimizer(self.config.optimization)

        # Review history
        self.review_history: List[OracleReviewResult] = []

    # =========================================================================
    # Training Data Collection
    # =========================================================================

    def process_new_reviews(self) -> int:
        """
        Extract training data from newly reviewed items.

        Implements feedback collapse prevention by filtering oracle-auto-reviewed
        items unless explicitly configured to include them.

        Returns:
            Number of new examples extracted
        """
        if self.review_queue is None:
            logger.warning("No review queue configured")
            return 0

        new_examples = self._extract_from_queue()
        added_count = len(new_examples)

        self.examples_since_last_train += added_count
        self.total_examples_collected += added_count

        logger.info(
            f"Extracted {added_count} new training examples "
            f"({self.examples_since_last_train} since last train)"
        )

        return added_count

    def _extract_from_queue(self) -> List[UnifiedTrainingExample]:
        """
        Extract training data from review queue with feedback collapse prevention.

        Key principle: By default, we EXCLUDE items that were auto-reviewed by
        the oracle classifier to prevent the model from training on its own
        predictions, which leads to feedback collapse.
        """
        examples = []

        # Get reviewed items from queue
        reviewed_items = getattr(self.review_queue, 'get_reviewed_items', None)
        if reviewed_items is None:
            # Try alternative access pattern
            reviewed_items = [
                item for item in getattr(self.review_queue, 'items', [])
                if getattr(item, 'reviewed', False)
            ]
        else:
            reviewed_items = reviewed_items()

        for item in reviewed_items:
            # CRITICAL: Feedback collapse prevention
            review_source = getattr(item, 'review_source', 'human')

            # Skip oracle-auto-reviewed items unless explicitly configured
            if not self.config.include_auto_reviewed:
                if review_source in ('oracle_classifier_auto', 'oracle_func_auto'):
                    logger.debug(f"Skipping auto-reviewed item {item.id} to prevent feedback collapse")
                    continue

            # Convert to training example
            example = self._flagged_item_to_example(item)
            if example is not None:
                # Apply confidence discount for auto-reviewed items
                if 'auto' in review_source:
                    example.confidence *= self.config.auto_reviewed_discount

                examples.append(example)
                self.training_collector.add_example(example)

                if review_source == 'human':
                    self.human_reviews_collected += 1

        return examples

    def _flagged_item_to_example(self, item: Any) -> Optional[UnifiedTrainingExample]:
        """Convert a flagged item to training example."""
        try:
            # Determine label based on review result
            review_result = getattr(item, 'review_result', None)
            is_violation = review_result in ('rejected', 'violation', 'flagged')

            # Determine violation type
            violation_type = ViolationType.NONE
            if is_violation:
                violation_type_str = getattr(item, 'violation_type', 'sufficiency')
                try:
                    violation_type = ViolationType(violation_type_str.lower())
                except ValueError:
                    violation_type = ViolationType.SUFFICIENCY

            return UnifiedTrainingExample(
                example_id=f"online_{item.id if hasattr(item, 'id') else id(item)}",
                source_type="online_review",
                original_content=getattr(item, 'input_a', '') or getattr(item, 'original_content', ''),
                summary=getattr(item, 'input_b', '') or getattr(item, 'summary', ''),
                rubric=getattr(item, 'rubric', ''),
                context={
                    'check_type': getattr(item, 'check_type', 'unknown'),
                    'node_id': getattr(item, 'node_id', None),
                    'tree_id': getattr(item, 'tree_id', None),
                },
                label=TrainingExampleLabel.POSITIVE if is_violation else TrainingExampleLabel.NEGATIVE,
                violation_type=violation_type,
                corrected_summary=getattr(item, 'corrected_summary', None),
                human_reasoning=getattr(item, 'human_reasoning', None),
                confidence=1.0 if getattr(item, 'review_source', 'human') == 'human' else 0.7,
            )
        except Exception as e:
            logger.warning(f"Could not convert item to training example: {e}")
            return None

    def add_human_feedback(
        self,
        item_id: str,
        is_violation: bool,
        violation_type: Optional[ViolationType] = None,
        corrected_summary: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> UnifiedTrainingExample:
        """
        Record human feedback directly (without review queue).

        Args:
            item_id: Identifier for the item being reviewed
            is_violation: Whether this is a true violation
            violation_type: Type of violation if is_violation
            corrected_summary: Human-provided correction
            reasoning: Human explanation

        Returns:
            Created training example
        """
        example = UnifiedTrainingExample(
            example_id=f"human_{item_id}_{int(datetime.now().timestamp())}",
            source_type="human_direct",
            original_content="",  # Would need to be filled by caller
            summary="",
            rubric="",
            context={'item_id': item_id},
            label=TrainingExampleLabel.POSITIVE if is_violation else TrainingExampleLabel.NEGATIVE,
            violation_type=violation_type or ViolationType.NONE,
            corrected_summary=corrected_summary,
            human_reasoning=reasoning,
            confidence=1.0,  # Full confidence for human feedback
        )

        self.training_collector.add_example(example)
        self.examples_since_last_train += 1
        self.human_reviews_collected += 1

        return example

    # =========================================================================
    # Retraining
    # =========================================================================

    def should_retrain(self) -> bool:
        """Check if retraining threshold has been met."""
        return (
            self.examples_since_last_train >= self.config.retrain_threshold
            and self.training_collector.total_examples >= self.config.min_examples_for_retrain
        )

    def retrain(self, force: bool = False) -> Optional[OptimizationResult]:
        """
        Retrain classifier with accumulated examples.

        Args:
            force: Force retraining even if threshold not met

        Returns:
            OptimizationResult or None if retrain skipped
        """
        if not force and not self.should_retrain():
            logger.info("Retrain threshold not met, skipping")
            return None

        # Get balanced training data
        trainset = self.training_collector.get_dspy_trainset(
            max_examples=self.config.max_examples_per_retrain,
            balanced=self.config.balance_positive_negative,
        )

        if len(trainset) < self.config.min_examples_for_retrain:
            logger.warning(
                f"Only {len(trainset)} examples available, "
                f"need {self.config.min_examples_for_retrain}"
            )
            return None

        logger.info(f"Retraining with {len(trainset)} examples")

        # Optimize
        try:
            self.classifier = self.optimizer.optimize(self.classifier, trainset)
            result = self.optimizer.optimization_history[-1]

            # Update state
            self.iteration_count += 1
            self.examples_since_last_train = 0

            logger.info(
                f"Retrain complete (iteration {self.iteration_count}): "
                f"metric {result.metric_before:.3f} → {result.metric_after:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return None

    # =========================================================================
    # Auto-Review
    # =========================================================================

    def auto_review_item(
        self,
        original_content: str,
        summary: str,
        rubric: str,
        item_id: Optional[str] = None,
    ) -> OracleReviewResult:
        """
        Review a single item using the classifier.

        Args:
            original_content: Original content before summarization
            summary: The summary to evaluate
            rubric: Information preservation criteria
            item_id: Optional item identifier

        Returns:
            OracleReviewResult with prediction and confidence
        """
        item_id = item_id or f"item_{int(datetime.now().timestamp())}"

        try:
            prediction = self.classifier(
                original_content=original_content,
                summary=summary,
                rubric=rubric,
            )

            # Interpret prediction
            label = prediction.label
            is_violation = label.lower() != 'none'

            try:
                violation_type = ViolationType(label.lower()) if is_violation else ViolationType.NONE
            except ValueError:
                violation_type = ViolationType.SUFFICIENCY if is_violation else ViolationType.NONE

            result = OracleReviewResult(
                item_id=item_id,
                is_violation=is_violation,
                violation_type=violation_type,
                confidence=prediction.confidence,
                reasoning=prediction.reasoning,
                review_source="oracle_classifier_auto",
            )

        except Exception as e:
            logger.error(f"Auto-review failed for {item_id}: {e}")
            result = OracleReviewResult(
                item_id=item_id,
                is_violation=False,
                violation_type=ViolationType.NONE,
                confidence=0.0,
                reasoning=f"Review failed: {e}",
                review_source="oracle_classifier_auto",
            )

        self.review_history.append(result)
        return result

    def auto_review_pending(
        self,
        auto_apply: bool = False,
        max_items: Optional[int] = None,
    ) -> List[OracleReviewResult]:
        """
        Review pending items in the queue with classifier.

        Args:
            auto_apply: Whether to auto-apply high-confidence decisions
            max_items: Maximum items to review

        Returns:
            List of review results
        """
        if self.review_queue is None:
            logger.warning("No review queue configured")
            return []

        results = []
        pending_items = list(self.review_queue.get_batch(max_items or 100))

        for item in pending_items:
            # Get content from item
            original_content = getattr(item, 'input_a', '') or getattr(item, 'original_content', '')
            summary = getattr(item, 'input_b', '') or getattr(item, 'summary', '')
            rubric = getattr(item, 'rubric', '')
            item_id = getattr(item, 'id', str(id(item)))

            result = self.auto_review_item(original_content, summary, rubric, item_id)
            results.append(result)

            # Auto-apply if confidence is high enough
            if auto_apply and result.confidence >= self.config.auto_apply_threshold:
                self._apply_review(item, result)
                self.auto_reviews_applied += 1

        return results

    def _apply_review(self, item: Any, result: OracleReviewResult) -> None:
        """Apply an auto-review result to an item."""
        try:
            item.reviewed = True
            item.review_result = "rejected" if result.is_violation else "approved"
            item.review_source = result.review_source  # Mark as auto-reviewed
            item.auto_confidence = result.confidence

            if result.corrected_summary:
                item.corrected_summary = result.corrected_summary

            logger.debug(f"Applied auto-review to {result.item_id}: {item.review_result}")

        except Exception as e:
            logger.warning(f"Could not apply review to item: {e}")

    def get_items_for_human_review(
        self,
        max_items: int = 10,
    ) -> List[Tuple[Any, float]]:
        """
        Get items that need human review due to low confidence.

        Returns items where oracle confidence is below threshold,
        prioritized by uncertainty (confidence closest to 0.5).

        Args:
            max_items: Maximum items to return

        Returns:
            List of (item, confidence) tuples
        """
        if self.review_queue is None:
            return []

        candidates = []

        for item in self.review_queue.get_batch(max_items * 3):
            # Skip already reviewed items
            if getattr(item, 'reviewed', False):
                continue

            # Get oracle confidence for this item
            result = self.auto_review_item(
                original_content=getattr(item, 'input_a', ''),
                summary=getattr(item, 'input_b', ''),
                rubric=getattr(item, 'rubric', ''),
                item_id=getattr(item, 'id', None),
            )

            # Below threshold = needs human review
            if result.confidence < self.config.human_review_threshold:
                # Uncertainty score: how close to 0.5 (maximum uncertainty)
                uncertainty = 1.0 - abs(0.5 - result.confidence) * 2
                candidates.append((item, result.confidence, uncertainty))

        # Sort by uncertainty (most uncertain first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return [(item, conf) for item, conf, _ in candidates[:max_items]]

    # =========================================================================
    # State Management
    # =========================================================================

    def save_state(self, filepath: Optional[Path] = None) -> Path:
        """Save manager state to file."""
        import json

        filepath = filepath or self.config.state_file
        if filepath is None:
            filepath = Path("data/online_learning_state.json")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'timestamp': datetime.now().isoformat(),
            'iteration_count': self.iteration_count,
            'examples_since_last_train': self.examples_since_last_train,
            'total_examples_collected': self.total_examples_collected,
            'auto_reviews_applied': self.auto_reviews_applied,
            'human_reviews_collected': self.human_reviews_collected,
            'config': self.config.to_dict(),
            'collector_stats': self.training_collector.get_statistics(),
            'optimizer_stats': self.optimizer.get_stats(),
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved online learning state to {filepath}")
        return filepath

    def load_state(self, filepath: Path) -> bool:
        """Load manager state from file."""
        import json

        filepath = Path(filepath)
        if not filepath.exists():
            return False

        try:
            with open(filepath) as f:
                state = json.load(f)

            self.iteration_count = state.get('iteration_count', 0)
            self.examples_since_last_train = state.get('examples_since_last_train', 0)
            self.total_examples_collected = state.get('total_examples_collected', 0)
            self.auto_reviews_applied = state.get('auto_reviews_applied', 0)
            self.human_reviews_collected = state.get('human_reviews_collected', 0)

            if 'config' in state:
                self.config = OnlineLearningConfig.from_dict(state['config'])

            logger.info(f"Loaded online learning state from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'iteration_count': self.iteration_count,
            'examples_since_last_train': self.examples_since_last_train,
            'total_examples_collected': self.total_examples_collected,
            'auto_reviews_applied': self.auto_reviews_applied,
            'human_reviews_collected': self.human_reviews_collected,
            'ready_for_retrain': self.should_retrain(),
            'collector_stats': self.training_collector.get_statistics(),
            'optimizer_stats': self.optimizer.get_stats(),
            'review_history_length': len(self.review_history),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_online_manager(
    classifier: dspy.Module,
    review_queue: Optional[Any] = None,
    config: Optional[OnlineLearningConfig] = None,
) -> OnlineLearningManager:
    """
    Create an online learning manager.

    Args:
        classifier: The score predictor module to manage
        review_queue: Optional ReviewQueue for feedback
        config: Configuration

    Returns:
        Configured OnlineLearningManager
    """
    return OnlineLearningManager(
        classifier=classifier,
        config=config,
        review_queue=review_queue,
    )
