"""
Tournament of Tournaments - Iterative Judge Optimization.

This module implements the full iterative tournament of tournaments loop for
improving the GenRM judge's ability to predict which summaries lead to better
downstream oracle performance.

The loop:
1. Build trees with current judge → collect preferences as free byproduct
2. Enrich preferences with oracle scores (ground truth from downstream task)
3. Optimize judge to predict oracle preferences
4. Evaluate improvement and check convergence
5. Repeat until convergence

The key insight is that the training signal comes from **downstream oracle
performance**, not from the judge's own scores. This avoids circular logic.

Usage:
    from src.training.tournament_loop import (
        TournamentOfTournamentsTrainer,
        ToTConfig,
    )

    # Create trainer
    trainer = TournamentOfTournamentsTrainer(
        summarizer=summarizer,
        oracle_predict=task.create_oracle_scorer(),
        initial_judge=judge,
        config=ToTConfig(max_iterations=5),
        output_dir=output_dir,
    )

    # Run training loop
    result = trainer.train(samples, rubric)
    print(f"Final accuracy: {result.final_judge_accuracy:.3f}")
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ops_engine.training_framework.preference import PreferencePair
    from src.ops_engine.training_framework.genrm_preference import GenRMJudge
    from src.ops_engine.training_framework.genrm_dspy import GenRMComparisonModule

from src.training.utils import normalize_error_to_01

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ToTConfig:
    """Configuration for Tournament of Tournaments."""

    # Iteration limits
    max_iterations: int = 5
    min_iterations: int = 1

    # Convergence criteria
    convergence_threshold: float = 0.01  # Stop if improvement < this
    convergence_patience: int = 2        # Stop after N iterations without improvement

    # Tree building
    # Number of candidate summaries generated per tournament round.
    # 4 provides a good diversity-cost tradeoff:
    # - Higher (8+): More diversity but 4x more LLM calls per merge
    # - Lower (2): Faster but less exploration of summary space
    # - 4: 4 candidates → 6 pairwise comparisons, covers space well
    k_candidates: int = 4
    n_samples_per_iteration: int = 50    # Samples to process per iteration
    candidate_temperature: float = 0.9   # Temperature for candidate generation

    # Judge optimization
    judge_budget: str = 'medium'         # 'light', 'medium', 'heavy', 'superheavy'
    num_threads: int = 4                 # Parallel evaluation threads
    judge_test_split: float = 0.2        # Holdout split for judge evaluation

    # Oracle comparison (normalized units when errors are normalized)
    tie_margin: float = 0.05             # Error difference below this = tie
    normalize_errors: bool = True        # Normalize errors to 0-1
    scale_range: Optional[float] = None  # Range for normalization (auto-detected or 1.0)
                                         # For RILE (-100 to +100), use scale_range=200
                                         # For 0-1 scales, use scale_range=1.0 (default)

    # Sampling
    shuffle_samples_each_iteration: bool = True
    random_seed: int = 42

    # Preference labeling (optional override)
    preference_labeler: Optional[Callable[['PreferencePair', float], Optional[str]]] = None

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1        # Save every N iterations


@dataclass
class ToTIterationResult:
    """Result from one iteration of the training loop."""

    iteration: int
    n_trees_built: int
    n_preferences_collected: int
    n_preferences_with_oracle: int
    judge_accuracy_before: float
    judge_accuracy_after: float
    improvement: float
    duration_seconds: float


@dataclass
class ToTResult:
    """Complete training result from the tournament of tournaments loop."""

    converged: bool
    convergence_reason: str  # 'patience', 'threshold', 'max_iterations'
    final_iteration: int
    iterations: List[ToTIterationResult] = field(default_factory=list)
    final_judge_accuracy: float = 0.0
    improvement_history: List[float] = field(default_factory=list)
    optimized_judge_path: Optional[Path] = None


# =============================================================================
# Tournament of Tournaments Trainer
# =============================================================================

class TournamentOfTournamentsTrainer:
    """
    Full iterative tournament of tournaments loop.

    Each iteration:
    1. Build trees with current judge → collect preferences
    2. Enrich preferences with oracle scores
    3. Optimize judge to predict oracle preferences
    4. Evaluate improvement
    5. Check convergence

    The training signal comes from downstream oracle performance,
    not from the judge's own scores. This is key to avoiding circular logic.
    """

    def __init__(
        self,
        summarizer: Callable[[str, str], str],
        oracle_predict: Callable[[str], float],
        initial_judge: 'GenRMJudge',
        config: ToTConfig,
        output_dir: Path,
        prompt_lm: Optional[Any] = None,
    ):
        """
        Initialize the trainer.

        Args:
            summarizer: Function(content, rubric) -> summary
            oracle_predict: Function(text) -> score (e.g., RILE predictor)
            initial_judge: GenRMJudge instance for initial tournament selection
            config: Training configuration
            output_dir: Directory for outputs and checkpoints
        """
        self.summarizer = summarizer
        self.oracle_predict = oracle_predict
        self.judge = initial_judge
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_lm = prompt_lm

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Track current DSPy-wrapped judge (for optimization)
        self._current_dspy_judge: Optional['GenRMComparisonModule'] = None

    def train(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
    ) -> ToTResult:
        """
        Run full tournament of tournaments loop.

        Args:
            samples: List of dicts with 'text', 'doc_id', 'reference_score'
            rubric: Information preservation criteria

        Returns:
            ToTResult with training statistics and final judge path
        """
        from src.ops_engine.training_framework.genrm_dspy import GenRMComparisonModule

        # Initialize prompt-tuned GenRM judge for optimization + tournament selection
        self._current_dspy_judge = GenRMComparisonModule(
            genrm_judge=self.judge,
            use_dspy_prompt=True,
            prompt_lm=self.prompt_lm,
        )

        iterations = []
        best_accuracy = 0.0
        patience_counter = 0
        convergence_reason = 'max_iterations'
        final_optimized_judge = None

        for iteration in range(1, self.config.max_iterations + 1):
            iteration_start = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"TOURNAMENT OF TOURNAMENTS - Iteration {iteration}")
            logger.info(f"{'='*60}")

            # Step 1: Build trees with current judge, collect preferences
            preferences = self._build_trees_and_collect_preferences(
                samples, rubric, iteration
            )

            if not preferences:
                logger.warning(f"Iteration {iteration}: No preferences collected, skipping")
                continue

            # Step 2: Enrich with oracle scores
            enriched = self._enrich_with_oracle(preferences, samples)

            if len(enriched) < 10:
                logger.warning(f"Iteration {iteration}: Only {len(enriched)} enriched examples, may be insufficient")

            # Step 3: Optimize judge and evaluate on holdout
            optimized_judge, opt_results = self._optimize_judge(enriched, iteration)
            final_optimized_judge = optimized_judge

            if opt_results and 'baseline' in opt_results and 'optimized' in opt_results:
                accuracy_before = opt_results['baseline'].get('accuracy', 0.0)
                accuracy_after = opt_results['optimized'].get('accuracy', 0.0)
            else:
                accuracy_before = self._evaluate_judge(enriched)
                accuracy_after = self._evaluate_judge(enriched, optimized_judge)

            logger.info(f"  Judge accuracy before: {accuracy_before:.3f}")
            logger.info(f"  Judge accuracy after: {accuracy_after:.3f}")

            improvement = accuracy_after - accuracy_before
            logger.info(f"  Improvement: {improvement:+.3f}")

            iteration_duration = time.time() - iteration_start

            # Record iteration result
            result = ToTIterationResult(
                iteration=iteration,
                n_trees_built=len(samples[:self.config.n_samples_per_iteration]),
                n_preferences_collected=len(preferences),
                n_preferences_with_oracle=len(enriched),
                judge_accuracy_before=accuracy_before,
                judge_accuracy_after=accuracy_after,
                improvement=improvement,
                duration_seconds=iteration_duration,
            )
            iterations.append(result)

            # Step 4: Update internal state for next iteration
            self._current_dspy_judge = optimized_judge

            # Step 5: Check convergence
            if accuracy_after > best_accuracy + self.config.convergence_threshold:
                best_accuracy = accuracy_after
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"  No improvement (patience: {patience_counter}/{self.config.convergence_patience})")

            if patience_counter >= self.config.convergence_patience:
                convergence_reason = 'patience'
                logger.info(f"Converged after {iteration} iterations (patience exhausted)")
                break

            if iteration >= self.config.min_iterations and improvement < self.config.convergence_threshold:
                convergence_reason = 'threshold'
                logger.info(f"Converged after {iteration} iterations (minimal improvement)")
                break

            # Save checkpoint
            if self.config.save_checkpoints and iteration % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(optimized_judge, iteration)

        # Save final judge
        judge_path = self.output_dir / 'optimized_judge' / 'judge_final.json'
        if final_optimized_judge is not None:
            self._save_judge(final_optimized_judge, judge_path)
        else:
            judge_path = None

        return ToTResult(
            converged=convergence_reason in ('patience', 'threshold'),
            convergence_reason=convergence_reason,
            final_iteration=iteration if iterations else 0,
            iterations=iterations,
            final_judge_accuracy=accuracy_after if iterations else 0.0,
            improvement_history=[it.improvement for it in iterations],
            optimized_judge_path=judge_path,
        )

    def _build_trees_and_collect_preferences(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
        iteration: int,
    ) -> List['PreferencePair']:
        """
        Build trees using current judge, collect preferences as byproduct.

        Preferences are "free" - we get them from the tournament selection
        process without any extra GenRM calls.
        """
        from src.ops_engine.builder import TreeBuilder, BuildConfig
        from src.core.strategy import CallableStrategy, TournamentStrategy, TournamentConfig

        base_strategy = CallableStrategy(self.summarizer)
        judge = self._current_dspy_judge or self.judge
        strategy = TournamentStrategy(
            base=base_strategy,
            judge=judge,
            config=TournamentConfig(
                k=self.config.k_candidates,
                temperature=self.config.candidate_temperature,
            ),
        )
        builder = TreeBuilder(strategy=strategy, config=BuildConfig())

        all_preferences = []
        samples_to_process = list(samples)
        if self.config.shuffle_samples_each_iteration:
            rng = random.Random(self.config.random_seed + iteration)
            rng.shuffle(samples_to_process)
        samples_to_process = samples_to_process[:self.config.n_samples_per_iteration]

        logger.info(f"  Building {len(samples_to_process)} trees...")

        for idx, sample in enumerate(samples_to_process):
            try:
                text = sample.get('text', '')
                if not text:
                    continue

                result = builder.build_sync(text, rubric)

                # Tag preferences with document ID
                doc_id = sample.get('doc_id', f"doc_{idx}")
                for pref in result.preferences:
                    pref.source_example_id = doc_id
                    pref.reference_score = sample.get('reference_score')

                all_preferences.extend(result.preferences)

            except Exception as e:
                logger.warning(f"  Tree building failed for sample {idx}: {e}")

            # Reset for next tree
            builder.reset()

        logger.info(f"  Collected {len(all_preferences)} preferences from {len(samples_to_process)} trees")
        return all_preferences

    def _enrich_with_oracle(
        self,
        preferences: List['PreferencePair'],
        samples: List[Dict[str, Any]],
    ) -> List['PreferencePair']:
        """
        Add oracle scores to preferences.

        For each preference pair, we:
        1. Score each summary with the oracle (e.g., RILE predictor)
        2. Look up the ground truth score for this document
        3. Compute oracle error for each summary
        4. Update the preference with this data

        The oracle error tells us which summary is objectively better
        for the downstream task.
        """
        # Create lookup for ground truth scores (if available)
        gt_lookup = {
            s.get('doc_id', f"doc_{i}"): s.get('reference_score')
            for i, s in enumerate(samples)
        }

        enriched = []
        for pref in preferences:
            try:
                # Score each summary with oracle
                score_a = self.oracle_predict(pref.summary_a)
                score_b = self.oracle_predict(pref.summary_b)

                # Get ground truth for this document (if available)
                gt = gt_lookup.get(pref.source_example_id)

                # Compute errors (lower is better) when GT exists
                raw_error_a = abs(score_a - gt) if gt is not None else None
                raw_error_b = abs(score_b - gt) if gt is not None else None

                error_a = raw_error_a
                error_b = raw_error_b
                if self.config.normalize_errors and gt is not None:
                    scale_range = self.config.scale_range
                    if scale_range is None:
                        logger.warning(
                            "No scale_range specified for error normalization. "
                            "Using default 1.0 (DSPy convention). "
                            "For RILE (-100 to +100), use scale_range=200."
                        )
                        scale_range = 1.0
                    error_a = normalize_error_to_01(raw_error_a, scale_range)
                    error_b = normalize_error_to_01(raw_error_b, scale_range)

                # Update preference with oracle data
                pref.score_estimate_a = score_a
                pref.score_estimate_b = score_b
                pref.oracle_error_a = error_a
                pref.oracle_error_b = error_b
                if gt is not None:
                    pref.reference_score = gt

                enriched.append(pref)

            except Exception as e:
                logger.debug(f"Oracle enrichment failed for preference: {e}")

        logger.info(f"  Enriched {len(enriched)}/{len(preferences)} preferences with oracle scores")
        return enriched

    def _optimize_judge(
        self,
        preferences: List['PreferencePair'],
        iteration: int,
    ) -> tuple['GenRMComparisonModule', dict]:
        """
        Train judge to predict oracle preferences using GEPA.

        The ground truth is derived from oracle errors:
        - Lower oracle error = better summary
        - We train the judge to predict which summary has lower oracle error
        """
        from src.training.judge_optimization import JudgeOptimizer, JudgeOptimizationConfig

        config = JudgeOptimizationConfig(
            budget=self.config.judge_budget,
            num_threads=self.config.num_threads,
            tie_margin=self.config.tie_margin,
            test_split=self.config.judge_test_split,
            checkpoint_dir=self.checkpoint_dir,
            preference_labeler=self.config.preference_labeler,
        )

        optimizer = JudgeOptimizer(config=config)
        if self.prompt_lm is not None:
            import dspy
            with dspy.context(lm=self.prompt_lm):
                optimized, results = optimizer.optimize(
                    preferences,
                    initial_judge=self._current_dspy_judge,
                )
        else:
            optimized, results = optimizer.optimize(
                preferences,
                initial_judge=self._current_dspy_judge,
            )

        logger.info(f"  Optimization results: {results}")

        return optimized, results

    def _evaluate_judge(
        self,
        preferences: List['PreferencePair'],
        judge: Optional['GenRMComparisonModule'] = None,
    ) -> float:
        """
        Evaluate judge accuracy on oracle-labeled preferences.

        Returns accuracy: proportion of correct preference predictions.
        """
        from src.training.judge_optimization import derive_ground_truth_preference

        judge = judge or self._current_dspy_judge
        if judge is None:
            return 0.0

        correct = 0
        total = 0

        for pref in preferences:
            try:
                # Get oracle ground truth
                gt = derive_ground_truth_preference(
                    pref,
                    tie_margin=self.config.tie_margin,
                    preference_labeler=self.config.preference_labeler,
                )
                if gt is None:
                    continue

                # Get judge prediction
                result = judge.forward(
                    context=pref.rubric,
                    original_text=pref.original_text,
                    summary_a=pref.summary_a,
                    summary_b=pref.summary_b,
                    law_type=pref.law_type,
                )

                predicted = getattr(result, 'preference', 'tie')
                total += 1
                if predicted == gt:
                    correct += 1

            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                total += 1  # Count as error (incorrect)

        return correct / max(1, total)

    def _save_checkpoint(
        self,
        judge: 'GenRMComparisonModule',
        iteration: int,
    ) -> None:
        """Save iteration checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'judge_iter_{iteration}.json'
        self._save_judge(judge, checkpoint_path)
        logger.info(f"  Saved checkpoint: {checkpoint_path}")

    def _save_judge(
        self,
        judge: 'GenRMComparisonModule',
        path: Path,
    ) -> None:
        """Save judge to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            judge.save(str(path))
        except Exception as e:
            logger.warning(f"Failed to save judge to {path}: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def run_tournament_of_tournaments(
    summarizer: Callable[[str, str], str],
    oracle_predict: Callable[[str], float],
    initial_judge: 'GenRMJudge',
    samples: List[Dict[str, Any]],
    rubric: str,
    output_dir: Path,
    max_iterations: int = 5,
    k_candidates: int = 4,
    judge_budget: str = 'medium',
    prompt_lm: Optional[Any] = None,
) -> ToTResult:
    """
    Convenience function to run tournament of tournaments.

    Args:
        summarizer: Function(content, rubric) -> summary
        oracle_predict: Function(text) -> score
        initial_judge: GenRMJudge instance
        samples: List of {text, doc_id, reference_score}
        rubric: Information preservation criteria
        output_dir: Output directory
        max_iterations: Maximum iterations
        k_candidates: Candidates per tournament
        judge_budget: GEPA budget

    Returns:
        ToTResult with training statistics
    """
    config = ToTConfig(
        max_iterations=max_iterations,
        k_candidates=k_candidates,
        judge_budget=judge_budget,
    )

    trainer = TournamentOfTournamentsTrainer(
        summarizer=summarizer,
        oracle_predict=oracle_predict,
        initial_judge=initial_judge,
        config=config,
        output_dir=output_dir,
        prompt_lm=prompt_lm,
    )

    return trainer.train(samples, rubric)


# Re-export from judge_optimization for backward compatibility
from src.training.judge_optimization import load_optimized_judge
