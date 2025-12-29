"""
Judge Optimization Module - Tournament of Tournaments.

This module provides functionality for optimizing GenRM judge prompts using DSPy.
It implements the "tournament of tournaments" concept where we optimize the judge
itself to improve comparison accuracy.

The optimized judge can then be used in TournamentStrategy for better
preference collection and summary selection.

Usage:
    from src.training.judge_optimization import (
        JudgeOptimizer,
        create_judge_trainset,
        derive_ground_truth_preference,
    )

    # Create optimizer
    optimizer = JudgeOptimizer(budget='medium', num_threads=4)

    # Optimize judge from preference pairs
    optimized_judge = optimizer.optimize(preference_pairs)

    # Use in TournamentStrategy
    strategy = TournamentStrategy(base=..., judge=optimized_judge)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import dspy

from src.ops_engine.training_framework.genrm_dspy import GenRMComparisonModule
from src.ops_engine.training_framework.preference import PreferencePair

PreferenceLabeler = Callable[[PreferencePair, float], Optional[str]]

logger = logging.getLogger(__name__)


# =============================================================================
# Training Data Preparation
# =============================================================================

def derive_ground_truth_preference(
    pair: PreferencePair,
    tie_margin: float = 0.5,
    preference_labeler: Optional[PreferenceLabeler] = None,
) -> Optional[str]:
    """
    Derive ground truth preference from oracle scores.

    By default:
    - If oracle_error_a/b are present, lower error is better.
    - Otherwise, falls back to score_estimate_a/b where higher is better.

    Args:
        pair: PreferencePair with optional score estimates
        tie_margin: Score difference below this is considered a tie
        preference_labeler: Optional override for custom metrics

    Returns:
        'A', 'B', 'tie', or None if no ground truth available
    """
    if preference_labeler is not None:
        return preference_labeler(pair, tie_margin)

    if pair.oracle_error_a is not None and pair.oracle_error_b is not None:
        diff = pair.oracle_error_a - pair.oracle_error_b
        if abs(diff) < tie_margin:
            return 'tie'
        return 'A' if diff < 0 else 'B'  # Lower error is better

    if pair.score_estimate_a is None or pair.score_estimate_b is None:
        return None

    diff = pair.score_estimate_a - pair.score_estimate_b

    if abs(diff) < tie_margin:
        return 'tie'
    return 'A' if diff > 0 else 'B'  # Higher score is better


def make_preference_labeler(
    metric_name: str,
    prefer_lower: bool = False,
) -> PreferenceLabeler:
    """
    Create a preference labeler from a metric stored on PreferencePair.

    Args:
        metric_name: Base metric name (e.g., "oracle_error", "score_estimate")
        prefer_lower: If True, lower metric value is better
    """
    metric_a_field = f"{metric_name}_a"
    metric_b_field = f"{metric_name}_b"

    def _label(pair: PreferencePair, tie_margin: float = 0.5) -> Optional[str]:
        value_a = getattr(pair, metric_a_field, None)
        value_b = getattr(pair, metric_b_field, None)
        if value_a is None or value_b is None:
            return None

        diff = value_a - value_b
        if abs(diff) < tie_margin:
            return 'tie'

        if prefer_lower:
            return 'A' if diff < 0 else 'B'
        return 'A' if diff > 0 else 'B'

    return _label


def create_judge_trainset(
    pairs: List[PreferencePair],
    tie_margin: float = 0.5,
    use_oracle_as_ground_truth: bool = True,
    max_original_text_chars: int = 4000,
    preference_labeler: Optional[PreferenceLabeler] = None,
) -> List[dspy.Example]:
    """
    Create DSPy training examples for judge optimization.

    Args:
        pairs: List of PreferencePair objects
        tie_margin: Score difference below this is considered a tie
        use_oracle_as_ground_truth: If True, derive ground truth from oracle scores.
                                   If False, use the existing 'preferred' field.
        max_original_text_chars: Truncate original text to this length
        preference_labeler: Optional override for custom preference labeling

    Returns:
        List of dspy.Example with judge training data
    """
    examples = []

    for pair in pairs:
        if use_oracle_as_ground_truth:
            ground_truth = derive_ground_truth_preference(
                pair,
                tie_margin=tie_margin,
                preference_labeler=preference_labeler,
            )
            if ground_truth is None:
                # Skip pairs without oracle scores
                continue
        else:
            ground_truth = pair.preferred

        example = dspy.Example(
            context=pair.rubric,
            original_text=pair.original_text[:max_original_text_chars],
            summary_a=pair.summary_a,
            summary_b=pair.summary_b,
            law_type=pair.law_type,
            ground_truth_preference=ground_truth,
        ).with_inputs('context', 'original_text', 'summary_a', 'summary_b', 'law_type')

        examples.append(example)

    logger.info(f"Created {len(examples)} training examples for judge optimization")
    return examples


# =============================================================================
# Metrics
# =============================================================================

def judge_accuracy_metric(example, prediction, trace=None) -> float:
    """
    Metric for judge accuracy: does the judge predict the correct preference?

    Returns 1.0 for correct, 0.0 for incorrect, 0.5 for tie mismatches.
    """
    try:
        predicted = getattr(prediction, 'preference', None)
        ground_truth = example.ground_truth_preference

        if predicted == ground_truth:
            return 1.0
        elif predicted == 'tie' or ground_truth == 'tie':
            # Partial credit for tie-related mismatches
            return 0.5
        else:
            return 0.0
    except (AttributeError, TypeError):
        return 0.0


def judge_accuracy_with_confidence(example, prediction, trace=None) -> float:
    """
    Metric that weights accuracy by confidence.

    Rewards confident correct predictions, penalizes confident wrong predictions.
    """
    try:
        predicted = getattr(prediction, 'preference', None)
        ground_truth = example.ground_truth_preference

        # Get confidence from ranking_score if available
        try:
            ranking_score = float(prediction.ranking_score)
            # Convert ranking_score (1-6) to confidence (0-1)
            # 1 or 6 = high confidence, 3.5 = low confidence
            confidence = abs(ranking_score - 3.5) / 2.5
        except (ValueError, TypeError, AttributeError):
            confidence = 0.5

        if predicted == ground_truth:
            return 0.5 + 0.5 * confidence  # 0.5 to 1.0
        elif predicted == 'tie' or ground_truth == 'tie':
            return 0.5  # Neutral for ties
        else:
            return 0.5 - 0.5 * confidence  # 0.0 to 0.5

    except (AttributeError, TypeError):
        return 0.0


# =============================================================================
# Judge Optimizer
# =============================================================================

@dataclass
class JudgeOptimizationConfig:
    """Configuration for judge optimization."""
    budget: str = 'light'  # 'light', 'medium', 'heavy', 'superheavy'
    num_threads: int = 4
    tie_margin: float = 0.05  # In metric units (normalized errors => 0-1)
    test_split: float = 0.2
    use_confidence_metric: bool = False
    checkpoint_dir: Optional[Path] = None
    preference_labeler: Optional[PreferenceLabeler] = None


class JudgeOptimizer:
    """
    Optimizer for GenRM judge prompts.

    Implements tournament of tournaments by training GenRMComparisonModule
    with optimizable DSPy prompts.
    """

    def __init__(
        self,
        config: Optional[JudgeOptimizationConfig] = None,
        budget: str = 'light',
        num_threads: int = 4,
    ):
        """
        Initialize judge optimizer.

        Args:
            config: Full configuration (if provided, overrides other args)
            budget: GEPA budget ('light', 'medium', 'heavy', 'superheavy')
            num_threads: Number of parallel evaluation threads
        """
        if config is not None:
            self.config = config
        else:
            self.config = JudgeOptimizationConfig(
                budget=budget,
                num_threads=num_threads,
            )

    def optimize(
        self,
        pairs: List[PreferencePair],
        use_oracle_as_ground_truth: bool = True,
        initial_judge: Optional[GenRMComparisonModule] = None,
    ) -> Tuple[GenRMComparisonModule, dict]:
        """
        Optimize GenRMComparisonModule using GEPA.

        Args:
            pairs: List of PreferencePair for training
            use_oracle_as_ground_truth: Derive ground truth from oracle scores
            initial_judge: Optional starting judge (used for baseline + warm start)

        Returns:
            Tuple of (optimized_judge, evaluation_results)
        """
        # Create training examples
        all_examples = create_judge_trainset(
            pairs,
            tie_margin=self.config.tie_margin,
            use_oracle_as_ground_truth=use_oracle_as_ground_truth,
            preference_labeler=self.config.preference_labeler,
        )

        judge_module = initial_judge
        if judge_module is None:
            judge_module = GenRMComparisonModule(use_dspy_predictor=True)
        elif not (
            getattr(judge_module, "use_dspy_predictor", False)
            or getattr(judge_module, "use_dspy_prompt", False)
        ):
            logger.warning("Initial judge does not support DSPy prompts; starting from a fresh DSPy judge")
            judge_module = GenRMComparisonModule(use_dspy_predictor=True)

        if len(all_examples) < 10:
            logger.warning(f"Only {len(all_examples)} examples, returning unoptimized judge")
            return judge_module, {'error': 'insufficient_data'}

        # Split train/test
        import random
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - self.config.test_split))
        trainset = all_examples[:split_idx]
        testset = all_examples[split_idx:]

        logger.info(f"Judge optimization: {len(trainset)} train, {len(testset)} test examples")

        # Select metric
        metric_fn = (
            judge_accuracy_with_confidence if self.config.use_confidence_metric
            else judge_accuracy_metric
        )

        # Evaluate baseline
        baseline_results = self._evaluate(judge_module, testset)
        logger.info(f"Baseline accuracy: {baseline_results['accuracy']:.3f}")

        # Create optimizer
        optimizer = dspy.GEPA(
            metric=metric_fn,
            auto=self.config.budget,
            num_threads=self.config.num_threads,
        )

        # Run optimization
        logger.info(f"Starting GEPA optimization (budget={self.config.budget})...")
        optimized_judge = optimizer.compile(
            judge_module,
            trainset=trainset,
        )

        # Evaluate optimized
        optimized_results = self._evaluate(optimized_judge, testset)
        logger.info(f"Optimized accuracy: {optimized_results['accuracy']:.3f}")
        logger.info(f"Improvement: {optimized_results['accuracy'] - baseline_results['accuracy']:+.3f}")

        results = {
            'baseline': baseline_results,
            'optimized': optimized_results,
            'improvement': optimized_results['accuracy'] - baseline_results['accuracy'],
            'train_size': len(trainset),
            'test_size': len(testset),
            'budget': self.config.budget,
        }

        return optimized_judge, results

    def _evaluate(
        self,
        judge: GenRMComparisonModule,
        testset: List[dspy.Example],
    ) -> dict:
        """Evaluate judge accuracy on test set."""
        correct = 0
        total = 0

        for example in testset:
            try:
                result = judge.forward(
                    context=example.context,
                    original_text=example.original_text,
                    summary_a=example.summary_a,
                    summary_b=example.summary_b,
                    law_type=example.law_type,
                )

                predicted = getattr(result, 'preference', 'tie')
                ground_truth = example.ground_truth_preference

                total += 1
                if predicted == ground_truth:
                    correct += 1

            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }

    def save(self, judge: GenRMComparisonModule, path: Path) -> None:
        """Save optimized judge to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        judge.save(str(path))
        logger.info(f"Saved optimized judge to {path}")

    def load(
        self,
        path: Path,
        use_dspy_prompt: bool = True,
        prompt_lm: Optional[dspy.LM] = None,
    ) -> GenRMComparisonModule:
        """Load optimized judge from file."""
        judge = GenRMComparisonModule(use_dspy_prompt=use_dspy_prompt, prompt_lm=prompt_lm)
        try:
            judge.load(str(path))
        except Exception as e:
            if use_dspy_prompt:
                logger.warning(f"Prompt-tuned judge load failed ({e}); retrying as DSPy predictor")
                judge = GenRMComparisonModule(use_dspy_predictor=True)
                judge.load(str(path))
            else:
                raise
        logger.info(f"Loaded optimized judge from {path}")
        return judge


# =============================================================================
# Convenience Functions
# =============================================================================

def optimize_judge_from_preferences(
    preferences: List[PreferencePair],
    budget: str = 'light',
    num_threads: int = 4,
    output_path: Optional[Path] = None,
) -> Tuple[GenRMComparisonModule, dict]:
    """
    Convenience function to optimize judge from preference pairs.

    Args:
        preferences: List of PreferencePair
        budget: GEPA budget
        num_threads: Parallel threads
        output_path: Optional path to save optimized judge

    Returns:
        Tuple of (optimized_judge, evaluation_results)
    """
    optimizer = JudgeOptimizer(budget=budget, num_threads=num_threads)
    judge, results = optimizer.optimize(preferences)

    if output_path:
        optimizer.save(judge, output_path)

    return judge, results


def load_optimized_judge(
    path: Path,
    use_dspy_prompt: bool = True,
    prompt_lm: Optional[dspy.LM] = None,
) -> GenRMComparisonModule:
    """Load optimized judge from file."""
    optimizer = JudgeOptimizer()
    return optimizer.load(path, use_dspy_prompt=use_dspy_prompt, prompt_lm=prompt_lm)
