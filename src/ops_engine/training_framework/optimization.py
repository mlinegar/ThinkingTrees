"""
DSPy Optimization for Oracle Approximation.

This module implements staged optimization for OracleClassifier with
swappable optimizer backends.

Key features:
- Swappable optimizers: GEPA (default), MIPROv2, BootstrapFewShot
- Single-stage optimization with configurable budget
- Multi-stage optimization with module freezing
- Checkpointing with full metadata
- Stage-specific metric support
- Feedback-rich metrics for GEPA reflection

Note: For the new registry-based optimizer system, see the optimizers/
submodule. This file is maintained for backward compatibility.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple

import dspy

from .core import LabelSpace, Prediction, UnifiedTrainingExample
from .config import OptimizationConfig
from .metrics import (
    create_classification_metric,
    create_violation_metric,
    evaluate_classifier,
    EvaluationResult,
)

# Note: The new registry-based optimizer system is in the optimizers/ submodule.
# To use it, import directly:
#   from ops_engine.training_framework.optimizers import get_optimizer, OptimizerRegistry
# Or via the training_framework package:
#   from ops_engine.training_framework import get_optimizer, OptimizerRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Result Tracking
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of a single optimization run or stage."""

    iteration: int
    stage: str  # "full", "classify", "retrieve", etc.
    timestamp: str

    # Metrics
    metric_before: float
    metric_after: float
    improvement: float

    # Data info
    examples_used: int
    trainset_size: int
    valset_size: int

    # Checkpoint
    checkpoint_path: Optional[Path] = None

    # Additional metadata
    config_snapshot: Optional[Dict] = None
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_message is None

    def to_dict(self) -> Dict:
        return {
            'iteration': self.iteration,
            'stage': self.stage,
            'timestamp': self.timestamp,
            'metric_before': self.metric_before,
            'metric_after': self.metric_after,
            'improvement': self.improvement,
            'examples_used': self.examples_used,
            'trainset_size': self.trainset_size,
            'valset_size': self.valset_size,
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'config_snapshot': self.config_snapshot,
            'error_message': self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationResult':
        data = dict(data)
        if data.get('checkpoint_path'):
            data['checkpoint_path'] = Path(data['checkpoint_path'])
        return cls(**data)


# =============================================================================
# Oracle Optimizer
# =============================================================================

class OracleOptimizer:
    """
    Stage-by-stage optimization for OracleClassifier.

    Supports two optimization modes:
    1. Single-stage: Optimize entire classifier at once
    2. Staged (left-to-right): Optimize retrieval, then classification

    Uses the _compiled flag pattern from xmc.dspy to freeze modules
    during staged optimization.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.optimization_history: List[OptimizationResult] = []
        self._iteration_count = 0

        # Ensure checkpoint directory exists
        if self.config.save_checkpoints:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def optimize(
        self,
        classifier: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        metric: Optional[Callable] = None,
        teacher: Optional[dspy.Module] = None,
    ) -> dspy.Module:
        """
        Single-stage optimization using BootstrapFewShot.

        Args:
            classifier: The OracleClassifier module to optimize
            trainset: Training examples
            valset: Optional validation examples (uses trainset if None)
            metric: DSPy metric function (auto-creates if None)
            teacher: Optional teacher module for bootstrap

        Returns:
            Optimized classifier module
        """
        self._iteration_count += 1
        timestamp = datetime.now().isoformat()

        if len(trainset) < self.config.min_training_examples:
            logger.warning(
                f"Only {len(trainset)} examples, need {self.config.min_training_examples}"
            )

        valset = valset or trainset

        # Get label space from classifier if available
        label_space = getattr(classifier, 'label_space', None)

        # Create default metric if not provided
        # Use feedback-rich metrics for GEPA optimizer
        use_feedback = self.config.optimizer_type == "gepa"
        if metric is None:
            if label_space:
                metric = create_classification_metric(
                    label_space,
                    weighted=label_space.is_ordinal,
                    with_feedback=use_feedback,
                )
            else:
                metric = create_violation_metric()

        # Evaluate before optimization
        metric_before = self._evaluate_metric(classifier, valset, metric)

        try:
            # Create compiler
            compiler = self._create_compiler(metric)

            # Compile based on optimizer type
            if self.config.optimizer_type == "gepa":
                # GEPA uses student/trainset/valset signature
                compiled = compiler.compile(
                    student=classifier,
                    trainset=trainset,
                    valset=valset,
                )
            elif self.config.optimizer_type == "mipro":
                # MIPROv2 uses student/trainset signature
                compiled = compiler.compile(
                    student=classifier,
                    trainset=trainset,
                )
            else:
                # Bootstrap uses classifier/teacher/trainset signature
                if teacher is None:
                    teacher = classifier.deepcopy() if hasattr(classifier, 'deepcopy') else None
                compiled = compiler.compile(
                    classifier,
                    teacher=teacher,
                    trainset=trainset,
                )

            # Evaluate after optimization
            metric_after = self._evaluate_metric(compiled, valset, metric)

            # Save checkpoint
            checkpoint_path = None
            if self.config.save_checkpoints:
                checkpoint_path = self.save_checkpoint(compiled)

            # Record result
            result = OptimizationResult(
                iteration=self._iteration_count,
                stage="full",
                timestamp=timestamp,
                metric_before=metric_before,
                metric_after=metric_after,
                improvement=metric_after - metric_before,
                examples_used=len(trainset),
                trainset_size=len(trainset),
                valset_size=len(valset),
                checkpoint_path=checkpoint_path,
                config_snapshot=self.config.to_dict(),
            )
            self.optimization_history.append(result)

            logger.info(
                f"Optimization complete: {metric_before:.3f} → {metric_after:.3f} "
                f"(+{result.improvement:.3f})"
            )

            return compiled

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            result = OptimizationResult(
                iteration=self._iteration_count,
                stage="full",
                timestamp=timestamp,
                metric_before=metric_before,
                metric_after=metric_before,
                improvement=0.0,
                examples_used=len(trainset),
                trainset_size=len(trainset),
                valset_size=len(valset),
                error_message=str(e),
            )
            self.optimization_history.append(result)
            raise

    def optimize_staged(
        self,
        classifier: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        stages: Optional[List[str]] = None,
        stage_metrics: Optional[Dict[str, Callable]] = None,
    ) -> dspy.Module:
        """
        Left-to-right staged optimization with module freezing.

        Optimizes stages sequentially, freezing each stage after optimization
        to prevent interference with subsequent stages.

        Args:
            classifier: The OracleClassifier module to optimize
            trainset: Training examples
            valset: Optional validation examples
            stages: List of stage names to optimize (default: ["retrieve", "classify"])
            stage_metrics: Dict mapping stage name to metric function

        Returns:
            Optimized classifier module
        """
        valset = valset or trainset
        stages = stages or self._infer_stages(classifier)
        stage_metrics = stage_metrics or {}

        label_space = getattr(classifier, 'label_space', None)

        # Default metrics for each stage
        default_metric = create_classification_metric(
            label_space, weighted=label_space.is_ordinal
        ) if label_space else create_violation_metric()

        logger.info(f"Starting staged optimization: {stages}")

        for stage_idx, stage in enumerate(stages):
            self._iteration_count += 1
            timestamp = datetime.now().isoformat()

            logger.info(f"Optimizing stage {stage_idx + 1}/{len(stages)}: {stage}")

            # Get stage-specific metric or use default
            metric = stage_metrics.get(stage, default_metric)

            # Freeze all other stages
            self._freeze_other_stages(classifier, stage, stages)

            # Evaluate before
            metric_before = self._evaluate_metric(classifier, valset, metric)

            try:
                # Create stage-specific compiler
                compiler = self._create_compiler(metric)

                # Compile this stage
                classifier = compiler.compile(
                    classifier,
                    trainset=trainset,
                )

                # Mark this stage as compiled
                self._mark_stage_compiled(classifier, stage)

                # Evaluate after
                metric_after = self._evaluate_metric(classifier, valset, metric)

                # Save checkpoint
                checkpoint_path = None
                if self.config.save_checkpoints:
                    checkpoint_path = self.save_checkpoint(
                        classifier,
                        suffix=f"_stage_{stage}"
                    )

                # Record result
                result = OptimizationResult(
                    iteration=self._iteration_count,
                    stage=stage,
                    timestamp=timestamp,
                    metric_before=metric_before,
                    metric_after=metric_after,
                    improvement=metric_after - metric_before,
                    examples_used=len(trainset),
                    trainset_size=len(trainset),
                    valset_size=len(valset),
                    checkpoint_path=checkpoint_path,
                    config_snapshot=self.config.to_dict(),
                )
                self.optimization_history.append(result)

                logger.info(
                    f"Stage '{stage}' complete: {metric_before:.3f} → {metric_after:.3f}"
                )

            except Exception as e:
                logger.error(f"Stage '{stage}' optimization failed: {e}")
                result = OptimizationResult(
                    iteration=self._iteration_count,
                    stage=stage,
                    timestamp=timestamp,
                    metric_before=metric_before,
                    metric_after=metric_before,
                    improvement=0.0,
                    examples_used=len(trainset),
                    trainset_size=len(trainset),
                    valset_size=len(valset),
                    error_message=str(e),
                )
                self.optimization_history.append(result)
                # Continue with next stage rather than failing completely
                continue

        return classifier

    def _create_compiler(
        self,
        metric: Callable,
    ) -> Any:
        """Create DSPy compiler based on config.optimizer_type."""
        optimizer_type = self.config.optimizer_type

        if optimizer_type == "gepa":
            try:
                # Wrap metric to extract score from dict (GEPA expects float returns)
                # GEPA requires 5 arguments: (gold, pred, trace, pred_name, pred_trace)
                def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                    result = metric(gold, pred, trace, pred_name, pred_trace)
                    if isinstance(result, dict):
                        return result.get('score', 0.0)
                    return result

                # Build GEPA kwargs - handle 'superheavy' and explicit max_metric_calls
                gepa_kwargs = {
                    "metric": gepa_metric,
                    "reflection_lm": dspy.settings.lm,
                    "use_merge": self.config.enable_merge,
                    "max_merge_invocations": self.config.max_merge_invocations,
                    "num_threads": self.config.num_threads,
                    "track_stats": self.config.track_stats,
                }
                if self.config.log_dir:
                    gepa_kwargs["log_dir"] = str(self.config.log_dir)

                # Handle budget: explicit max_metric_calls overrides auto
                if self.config.max_metric_calls:
                    logger.info(f"Creating GEPA optimizer (max_metric_calls={self.config.max_metric_calls})")
                    gepa_kwargs["max_metric_calls"] = self.config.max_metric_calls
                elif self.config.gepa_auto == "superheavy":
                    # 'superheavy' uses max_metric_calls instead of auto
                    logger.info("Creating GEPA optimizer (budget=superheavy, max_metric_calls=5000)")
                    gepa_kwargs["max_metric_calls"] = 5000
                else:
                    # Standard budgets: light, medium, heavy
                    logger.info(f"Creating GEPA optimizer (budget={self.config.gepa_auto})")
                    gepa_kwargs["auto"] = self.config.gepa_auto

                return dspy.GEPA(**gepa_kwargs)
            except Exception as e:
                logger.warning(f"GEPA creation failed: {e}, falling back to bootstrap")
                optimizer_type = "bootstrap"

        if optimizer_type == "mipro":
            try:
                # Wrap metric to extract score from dict (MIPROv2 expects float returns)
                def mipro_metric(gold, pred, trace=None):
                    result = metric(gold, pred, trace)
                    if isinstance(result, dict):
                        return result.get('score', 0.0)
                    return result

                logger.info(f"Creating MIPROv2 optimizer (budget={self.config.gepa_auto})")
                return dspy.MIPROv2(
                    metric=mipro_metric,
                    auto=self.config.gepa_auto,
                    num_threads=self.config.num_threads,
                )
            except Exception as e:
                logger.warning(f"MIPROv2 creation failed: {e}, falling back to bootstrap")
                optimizer_type = "bootstrap"

        # Default: bootstrap
        logger.info("Creating BootstrapFewShot optimizer")

        # Wrap metric to extract score from dict (Bootstrap can't handle dict returns)
        def bootstrap_metric(gold, pred, trace=None):
            result = metric(gold, pred, trace)
            if isinstance(result, dict):
                return result.get('score', 0.0)
            return result

        try:
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
            return BootstrapFewShotWithRandomSearch(
                metric=bootstrap_metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                max_rounds=self.config.max_rounds,
                num_candidate_programs=self.config.num_candidate_programs,
                num_threads=self.config.num_threads,
            )
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Falling back to basic BootstrapFewShot: {e}")
            return dspy.BootstrapFewShot(
                metric=bootstrap_metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                max_rounds=self.config.max_rounds,
            )

    def _evaluate_metric(
        self,
        classifier: dspy.Module,
        examples: List[dspy.Example],
        metric: Callable,
    ) -> float:
        """Evaluate classifier on examples using metric."""
        if not examples:
            return 0.0

        scores = []
        for example in examples[:min(20, len(examples))]:  # Limit for speed
            try:
                # Call classifier
                prediction = classifier(
                    original_content=example.original_content,
                    summary=example.summary,
                    rubric=example.rubric,
                )
                result = metric(example, prediction)
                # Handle feedback-rich metrics that return dict
                if isinstance(result, dict):
                    score = result.get('score', 0.0)
                else:
                    score = result
                scores.append(score)
            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _infer_stages(self, classifier: dspy.Module) -> List[str]:
        """Infer optimization stages from classifier structure."""
        stages = []

        # Check for retriever
        if hasattr(classifier, 'retriever') and classifier.retriever is not None:
            stages.append("retrieve")

        # Classification is always a stage
        stages.append("classify")

        return stages

    def _freeze_other_stages(
        self,
        classifier: dspy.Module,
        current_stage: str,
        all_stages: List[str],
    ) -> None:
        """Freeze all stages except the current one."""
        # Map stage names to module attributes
        stage_modules = {
            "retrieve": ["retriever"],
            "classify": ["classify", "detect_violation"],
        }

        for stage in all_stages:
            modules = stage_modules.get(stage, [stage])
            should_freeze = (stage != current_stage)

            for module_name in modules:
                module = getattr(classifier, module_name, None)
                if module is not None:
                    # Freeze by setting _compiled flag on inner CoT modules
                    if hasattr(module, 'cot'):
                        module.cot._compiled = should_freeze
                    elif hasattr(module, '_compiled'):
                        module._compiled = should_freeze

    def _mark_stage_compiled(self, classifier: dspy.Module, stage: str) -> None:
        """Mark a stage as compiled after optimization."""
        stage_modules = {
            "retrieve": ["retriever"],
            "classify": ["classify", "detect_violation"],
        }

        modules = stage_modules.get(stage, [stage])
        for module_name in modules:
            module = getattr(classifier, module_name, None)
            if module is not None:
                if hasattr(module, 'cot'):
                    module.cot._compiled = True
                elif hasattr(module, '_compiled'):
                    module._compiled = True

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(
        self,
        classifier: dspy.Module,
        suffix: str = "",
    ) -> Path:
        """
        Save classifier checkpoint with full metadata.

        Args:
            classifier: Classifier to save
            suffix: Optional suffix for checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        timestamp = int(time.time())
        filename = f"oracle_classifier_{timestamp}{suffix}.json"
        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'iteration': self._iteration_count,
            'config': self.config.to_dict(),
            'history': [r.to_dict() for r in self.optimization_history],
        }

        # Save classifier state if possible
        try:
            if hasattr(classifier, 'dump_state'):
                checkpoint_data['classifier_state'] = classifier.dump_state()

            # Also try DSPy's save method
            state_path = checkpoint_path.with_suffix('.state.json')
            if hasattr(classifier, 'save'):
                classifier.save(str(state_path))
                checkpoint_data['state_path'] = str(state_path)
        except Exception as e:
            logger.warning(f"Could not save full classifier state: {e}")

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        classifier: Optional[dspy.Module] = None,
    ) -> Tuple[Optional[dspy.Module], Dict]:
        """
        Load checkpoint and optionally restore classifier state.

        Args:
            checkpoint_path: Path to checkpoint file
            classifier: Optional classifier to restore state into

        Returns:
            Tuple of (classifier or None, checkpoint_metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        # Restore history
        if 'history' in checkpoint_data:
            self.optimization_history = [
                OptimizationResult.from_dict(r)
                for r in checkpoint_data['history']
            ]

        self._iteration_count = checkpoint_data.get('iteration', 0)

        # Try to restore classifier state
        if classifier is not None:
            try:
                if 'classifier_state' in checkpoint_data:
                    classifier.load_state(checkpoint_data['classifier_state'])
                elif 'state_path' in checkpoint_data:
                    classifier.load(checkpoint_data['state_path'])
            except Exception as e:
                logger.warning(f"Could not restore classifier state: {e}")

        return classifier, checkpoint_data

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if not self.config.checkpoint_dir.exists():
            return None

        checkpoints = list(self.config.checkpoint_dir.glob("oracle_classifier_*.json"))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'iterations': self._iteration_count,
            'history_length': len(self.optimization_history),
            'total_improvement': sum(
                r.improvement for r in self.optimization_history if r.success
            ),
            'successful_optimizations': sum(
                1 for r in self.optimization_history if r.success
            ),
            'failed_optimizations': sum(
                1 for r in self.optimization_history if not r.success
            ),
            'stages_optimized': list(set(
                r.stage for r in self.optimization_history
            )),
            'latest_metric': (
                self.optimization_history[-1].metric_after
                if self.optimization_history else None
            ),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_optimizer(config: Optional[OptimizationConfig] = None) -> OracleOptimizer:
    """Create an OracleOptimizer with optional config."""
    return OracleOptimizer(config)


