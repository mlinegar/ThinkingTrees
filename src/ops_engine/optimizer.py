"""
Optimizer module for OPS (Oracle-Preserving Summarization).

This module implements bootstrap learning from audit failures:
- Collects training examples from human-reviewed audit failures
- Uses DSPy's BootstrapFewShot to learn from failures
- Compiles improved summarizers from edge cases
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict, Tuple
from pathlib import Path
import dspy

from src.core.data_models import OPSTree, OPSNode, AuditStatus
from src.core.signatures import RecursiveSummary, Summarizer
from src.ops_engine.auditor import (
    ReviewQueue, FlaggedItem, ReviewPriority,
    AuditReport, OPSAuditor
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for optimization."""
    content: str  # The input content to summarize
    rubric: str  # The information preservation criteria
    summary: str  # The correct/improved summary
    source: str = "human"  # Source of the correction (human, oracle, etc.)
    node_id: Optional[str] = None
    tree_id: Optional[str] = None
    discrepancy_score: float = 0.0

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format."""
        return dspy.Example(
            content=self.content,
            rubric=self.rubric,
            summary=self.summary
        ).with_inputs("content", "rubric")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'content': self.content,
            'rubric': self.rubric,
            'summary': self.summary,
            'source': self.source,
            'node_id': self.node_id,
            'tree_id': self.tree_id,
            'discrepancy_score': self.discrepancy_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingExample':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 8
    num_candidate_programs: int = 10
    metric_threshold: float = 0.8
    save_checkpoints: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("data/checkpoints"))


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    success: bool
    examples_used: int
    improvement_score: float
    iterations: int
    message: str = ""
    checkpoint_path: Optional[Path] = None


class TrainingDataCollector:
    """
    Collects and manages training data from audit failures and reviews.
    """

    def __init__(self):
        self.examples: List[TrainingExample] = []
        self._by_tree: Dict[str, List[TrainingExample]] = {}

    def add_example(self, example: TrainingExample) -> None:
        """Add a training example."""
        self.examples.append(example)
        if example.tree_id:
            if example.tree_id not in self._by_tree:
                self._by_tree[example.tree_id] = []
            self._by_tree[example.tree_id].append(example)

    def add_from_review(
        self,
        flagged_item: FlaggedItem,
        corrected_summary: str,
        original_content: Optional[str] = None
    ) -> TrainingExample:
        """
        Create training example from a reviewed flagged item.

        Args:
            flagged_item: The flagged item that was reviewed
            corrected_summary: The human-corrected summary
            original_content: The original content (uses input_a if not provided)

        Returns:
            The created TrainingExample
        """
        example = TrainingExample(
            content=original_content or flagged_item.input_a,
            rubric=flagged_item.rubric,
            summary=corrected_summary,
            source="human_review",
            node_id=flagged_item.node_id,
            tree_id=flagged_item.tree_id,
            discrepancy_score=flagged_item.approx_discrepancy
        )
        self.add_example(example)
        return example

    def add_from_review_queue(self, queue: ReviewQueue) -> int:
        """
        Extract training examples from reviewed items in queue.

        Returns:
            Number of examples added
        """
        added = 0
        for item in queue.items:
            if item.reviewed and item.corrected_summary:
                self.add_from_review(item, item.corrected_summary)
                added += 1
        return added

    def get_examples(
        self,
        min_discrepancy: float = 0.0,
        source_filter: Optional[str] = None
    ) -> List[TrainingExample]:
        """Get filtered examples."""
        result = self.examples

        if min_discrepancy > 0:
            result = [e for e in result if e.discrepancy_score >= min_discrepancy]

        if source_filter:
            result = [e for e in result if e.source == source_filter]

        return result

    def get_dspy_examples(self) -> List[dspy.Example]:
        """Get all examples in DSPy format."""
        return [e.to_dspy_example() for e in self.examples]

    def save(self, filepath: Path) -> None:
        """Save training data to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self.examples]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path) -> int:
        """
        Load training data from JSON.

        Returns:
            Number of examples loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return 0

        with open(filepath) as f:
            data = json.load(f)

        for item in data:
            self.add_example(TrainingExample.from_dict(item))

        return len(data)

    def __len__(self) -> int:
        return len(self.examples)


class SummaryMetric:
    """
    Metric for evaluating summary quality.

    Wraps an oracle judge to provide DSPy-compatible metric function.
    """

    def __init__(self, oracle_fn: Optional[Callable] = None, threshold: float = 0.8):
        """
        Args:
            oracle_fn: Optional oracle function for evaluation.
                      Should take (original, summary, rubric) -> score
            threshold: Minimum score to consider a summary passing
        """
        self.oracle_fn = oracle_fn
        self.threshold = threshold
        self.call_count = 0

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """
        Evaluate prediction against example.

        Returns score from 0 to 1.
        """
        self.call_count += 1

        # Get predicted and expected summaries
        predicted = prediction.summary if hasattr(prediction, 'summary') else str(prediction)
        expected = example.summary

        if self.oracle_fn:
            # Use oracle for evaluation
            score = self.oracle_fn(
                example.content,
                predicted,
                example.rubric
            )
            return float(score >= self.threshold)

        # Simple string similarity fallback
        return self._simple_similarity(expected, predicted)

    def _simple_similarity(self, expected: str, predicted: str) -> float:
        """Simple word overlap similarity."""
        expected_words = set(expected.lower().split())
        predicted_words = set(predicted.lower().split())

        if not expected_words:
            return 0.0

        overlap = len(expected_words & predicted_words)
        return overlap / len(expected_words)


class OPSOptimizer:
    """
    Optimizer for OPS summarization using DSPy bootstrap learning.

    This optimizer:
    1. Collects training examples from human-reviewed audit failures
    2. Uses DSPy's BootstrapFewShot to learn from these examples
    3. Produces improved summarizer modules
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        metric: Optional[SummaryMetric] = None
    ):
        self.config = config or OptimizationConfig()
        self.metric = metric or SummaryMetric()
        self.training_data = TrainingDataCollector()
        self._compiled_module: Optional[dspy.Module] = None
        self._optimization_history: List[OptimizationResult] = []

    def add_training_example(self, example: TrainingExample) -> None:
        """Add a training example."""
        self.training_data.add_example(example)

    def add_from_audit_failure(
        self,
        node: OPSNode,
        tree_id: str,
        rubric: str,
        corrected_summary: str,
        original_content: Optional[str] = None
    ) -> TrainingExample:
        """
        Add training example from a failed audit with correction.

        Args:
            node: The node that failed audit
            tree_id: ID of the tree containing the node
            rubric: The rubric used for summarization
            corrected_summary: The human-corrected summary
            original_content: Original content (inferred from node if not provided)

        Returns:
            Created TrainingExample
        """
        # Determine original content
        if original_content is None:
            if node.is_leaf:
                original_content = node.raw_text_span or node.summary
            else:
                # For internal nodes, use concatenated child summaries
                child_summaries = []
                for child in node.children:
                    child_summaries.append(child.summary)
                original_content = "\n\n".join(child_summaries)

        example = TrainingExample(
            content=original_content,
            rubric=rubric,
            summary=corrected_summary,
            source="audit_failure",
            node_id=node.id,
            tree_id=tree_id,
            discrepancy_score=node.discrepancy_score
        )
        self.training_data.add_example(example)
        return example

    def extract_from_review_queue(self, queue: ReviewQueue) -> int:
        """
        Extract training examples from reviewed items.

        Returns number of examples extracted.
        """
        return self.training_data.add_from_review_queue(queue)

    def optimize(
        self,
        base_module: Optional[dspy.Module] = None,
        teacher_lm: Optional[Any] = None
    ) -> OptimizationResult:
        """
        Run optimization using collected training data.

        Args:
            base_module: Module to optimize (creates new Summarizer if None)
            teacher_lm: Optional teacher LM for bootstrap

        Returns:
            OptimizationResult with details
        """
        examples = self.training_data.get_dspy_examples()

        if len(examples) < 2:
            return OptimizationResult(
                success=False,
                examples_used=len(examples),
                improvement_score=0.0,
                iterations=0,
                message="Not enough training examples (need at least 2)"
            )

        # Create base module if not provided
        if base_module is None:
            base_module = Summarizer()

        try:
            # Configure bootstrap optimizer
            teleprompter = dspy.BootstrapFewShot(
                metric=self.metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                max_rounds=1  # Single round for efficiency
            )

            # Set teacher if provided
            if teacher_lm:
                teleprompter.teacher_lm = teacher_lm

            # Run optimization
            compiled = teleprompter.compile(
                base_module,
                trainset=examples
            )

            self._compiled_module = compiled

            # Save checkpoint if configured
            checkpoint_path = None
            if self.config.save_checkpoints:
                checkpoint_path = self._save_checkpoint(compiled)

            result = OptimizationResult(
                success=True,
                examples_used=len(examples),
                improvement_score=0.0,  # Would need eval to compute
                iterations=1,
                message="Optimization completed successfully",
                checkpoint_path=checkpoint_path
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            result = OptimizationResult(
                success=False,
                examples_used=len(examples),
                improvement_score=0.0,
                iterations=0,
                message=f"Optimization failed: {str(e)}"
            )

        self._optimization_history.append(result)
        return result

    def get_optimized_summarizer(self) -> Optional[Callable[[str, str], str]]:
        """
        Get the optimized summarizer function.

        Returns:
            Callable that takes (content, rubric) and returns summary,
            or None if no optimization has been run
        """
        if self._compiled_module is None:
            return None

        def summarizer(content: str, rubric: str) -> str:
            return self._compiled_module(content=content, rubric=rubric)

        return summarizer

    def _save_checkpoint(self, module: dspy.Module) -> Path:
        """Save optimization checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save module state
        import time
        timestamp = int(time.time())
        checkpoint_path = self.config.checkpoint_dir / f"summarizer_{timestamp}.json"

        # DSPy modules can be saved via their state
        try:
            module.save(str(checkpoint_path))
        except Exception:
            # Fallback: save training data used
            self.training_data.save(
                self.config.checkpoint_dir / f"training_data_{timestamp}.json"
            )

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Load optimized module from checkpoint.

        Returns True if successful.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return False

        try:
            self._compiled_module = Summarizer()
            self._compiled_module.load(str(checkpoint_path))
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save_training_data(self, filepath: Path) -> None:
        """Save training data to file."""
        self.training_data.save(filepath)

    def load_training_data(self, filepath: Path) -> int:
        """Load training data from file. Returns count loaded."""
        return self.training_data.load(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'training_examples': len(self.training_data),
            'optimizations_run': len(self._optimization_history),
            'has_compiled_module': self._compiled_module is not None,
            'metric_calls': self.metric.call_count,
            'history': [
                {
                    'success': r.success,
                    'examples_used': r.examples_used,
                    'improvement': r.improvement_score
                }
                for r in self._optimization_history
            ]
        }


def create_optimizer(
    config: Optional[OptimizationConfig] = None,
    oracle_fn: Optional[Callable] = None
) -> OPSOptimizer:
    """
    Convenience function to create an optimizer.

    Args:
        config: Optimization configuration
        oracle_fn: Optional oracle function for metric evaluation

    Returns:
        Configured OPSOptimizer
    """
    metric = SummaryMetric(oracle_fn=oracle_fn) if oracle_fn else None
    return OPSOptimizer(config=config, metric=metric)


def optimize_from_reviews(
    queue: ReviewQueue,
    oracle_fn: Optional[Callable] = None,
    config: Optional[OptimizationConfig] = None
) -> Tuple[OPSOptimizer, OptimizationResult]:
    """
    Convenience function to optimize from a review queue.

    Args:
        queue: Review queue with reviewed items
        oracle_fn: Optional oracle function for metric
        config: Optimization configuration

    Returns:
        Tuple of (optimizer, result)
    """
    optimizer = create_optimizer(config=config, oracle_fn=oracle_fn)
    optimizer.extract_from_review_queue(queue)
    result = optimizer.optimize()
    return optimizer, result
