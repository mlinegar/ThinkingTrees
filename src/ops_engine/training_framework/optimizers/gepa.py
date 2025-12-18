"""
GEPA (Gradient-free Efficient Prompt Adaptation) optimizer wrapper.

GEPA is the default optimizer for complex optimization tasks. It uses
reflection and merge capabilities for sophisticated prompt optimization.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import dspy

from .base import AbstractOptimizer, OptimizationResult
from .registry import register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer("gepa")
class GEPAOptimizer(AbstractOptimizer):
    """
    GEPA optimizer wrapper.

    GEPA (Gradient-free Efficient Prompt Adaptation) is a sophisticated
    optimizer that uses reflection and instruction merging to optimize
    DSPy modules.

    Key features:
    - Reflection-based optimization with teacher LM
    - Instruction merging for combining good prompts
    - Budget control via auto modes or max_metric_calls
    - Track statistics for optimization analysis
    """

    @property
    def name(self) -> str:
        return "gepa"

    @property
    def supports_parallel(self) -> bool:
        # GEPA uses num_threads internally for parallel evaluation
        return True

    def compile(
        self,
        student: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        metric: Optional[Callable] = None,
        teacher: Optional[dspy.Module] = None,
        **kwargs,
    ) -> dspy.Module:
        """
        Compile using GEPA optimizer.

        Args:
            student: Module to optimize
            trainset: Training examples
            valset: Validation examples (GEPA uses these for validation)
            metric: DSPy metric function
            teacher: Not used by GEPA (uses reflection LM instead)
            **kwargs: Additional arguments (e.g., override config)

        Returns:
            Optimized module
        """
        valset = valset or trainset
        self._log_compile_start(len(trainset), len(valset))

        # Wrap metric for GEPA's 5-argument signature
        wrapped_metric = self._wrap_metric_gepa(metric) if metric else None

        # Build GEPA kwargs
        gepa_kwargs = self._build_gepa_kwargs(wrapped_metric)

        # Override with any kwargs passed in
        gepa_kwargs.update(kwargs.get('gepa_kwargs', {}))

        try:
            optimizer = dspy.GEPA(**gepa_kwargs)
            compiled = optimizer.compile(
                student=student,
                trainset=trainset,
                valset=valset,
            )
            return compiled

        except Exception as e:
            logger.error(f"GEPA compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """
        Estimate metric calls based on GEPA budget setting.

        Args:
            dataset_size: Number of training examples

        Returns:
            Estimated metric call count
        """
        if self.config is None:
            return 1000  # Default estimate

        # Check for explicit max_metric_calls
        max_calls = getattr(self.config, 'max_metric_calls', None)
        if max_calls:
            return max_calls

        # Estimate based on budget mode
        budget = getattr(self.config, 'gepa_auto', 'medium')
        budget_estimates = {
            'light': 300,
            'medium': 1000,
            'heavy': 3000,
            'superheavy': 5000,
        }
        return budget_estimates.get(budget, 1000)

    def _wrap_metric_gepa(self, metric: Callable) -> Callable:
        """
        Wrap metric for GEPA's 5-argument signature.

        GEPA expects: (gold, pred, trace, pred_name, pred_trace) -> float

        Args:
            metric: Original metric function

        Returns:
            GEPA-compatible metric function
        """
        def wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
            result = metric(gold, pred, trace, pred_name, pred_trace)
            if isinstance(result, dict):
                return result.get('score', 0.0)
            return result

        return wrapped

    def _build_gepa_kwargs(self, metric: Optional[Callable]) -> Dict[str, Any]:
        """Build kwargs for GEPA constructor."""
        kwargs: Dict[str, Any] = {
            'metric': metric,
            'reflection_lm': dspy.settings.lm,  # Use current LM for reflection
        }

        if self.config is None:
            # Default configuration
            kwargs['auto'] = 'medium'
            kwargs['num_threads'] = 64
            return kwargs

        # Configure from config
        kwargs['use_merge'] = getattr(self.config, 'enable_merge', True)
        kwargs['max_merge_invocations'] = getattr(self.config, 'max_merge_invocations', 5)
        kwargs['num_threads'] = getattr(self.config, 'num_threads', 128)
        kwargs['track_stats'] = getattr(self.config, 'track_stats', True)

        # Log directory
        log_dir = getattr(self.config, 'log_dir', None)
        if log_dir:
            kwargs['log_dir'] = str(log_dir)

        # Budget control
        max_metric_calls = getattr(self.config, 'max_metric_calls', None)
        gepa_auto = getattr(self.config, 'gepa_auto', 'heavy')

        if max_metric_calls:
            logger.info(f"GEPA: Using explicit max_metric_calls={max_metric_calls}")
            kwargs['max_metric_calls'] = max_metric_calls
        elif gepa_auto == 'superheavy':
            # 'superheavy' uses max_metric_calls instead of auto
            logger.info("GEPA: Using superheavy budget (max_metric_calls=5000)")
            kwargs['max_metric_calls'] = 5000
        else:
            logger.info(f"GEPA: Using auto='{gepa_auto}' budget")
            kwargs['auto'] = gepa_auto

        return kwargs
