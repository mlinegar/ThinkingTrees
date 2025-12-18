"""
MIPROv2 optimizer wrapper.

MIPRO (Multi-stage Instruction Proposal and Optimization) is a sophisticated
optimizer for large datasets that combines instruction optimization with
few-shot demonstration selection.
"""

import logging
from typing import Callable, List, Optional

import dspy

from .base import AbstractOptimizer
from .registry import register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer("mipro")
class MIPROOptimizer(AbstractOptimizer):
    """
    MIPROv2 optimizer wrapper.

    MIPRO (Multi-stage Instruction Proposal and Optimization) uses a
    multi-stage approach to optimize both instructions and demonstrations.

    Best for:
    - Large datasets (200+ examples)
    - When instruction optimization is important
    - Production-quality optimization

    Budget modes:
    - light: Quick optimization (~300 metric calls)
    - medium: Balanced (~1000 metric calls)
    - heavy: Thorough (~3000 metric calls)
    """

    @property
    def name(self) -> str:
        return "mipro"

    @property
    def supports_parallel(self) -> bool:
        # MIPROv2 uses num_threads for parallel evaluation
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
        Compile using MIPROv2.

        Args:
            student: Module to optimize
            trainset: Training examples
            valset: Not directly used by MIPROv2
            metric: DSPy metric function
            teacher: Not used by MIPROv2
            **kwargs: Additional arguments

        Returns:
            Optimized module
        """
        self._log_compile_start(len(trainset), len(valset or trainset))

        # Wrap metric to extract score from dict returns
        wrapped_metric = self.wrap_metric(metric) if metric else None

        # Build optimizer kwargs
        opt_kwargs = self._build_kwargs()

        try:
            optimizer = dspy.MIPROv2(
                metric=wrapped_metric,
                **opt_kwargs,
            )

            # MIPROv2 uses student/trainset signature
            compiled = optimizer.compile(
                student=student,
                trainset=trainset,
            )
            return compiled

        except Exception as e:
            logger.error(f"MIPROv2 compilation failed: {e}")
            raise

    def estimate_budget(self, dataset_size: int) -> int:
        """Estimate metric calls based on budget mode."""
        if self.config is None:
            return 1000  # Default medium

        budget = getattr(self.config, 'mipro_auto', 'medium')
        budget_estimates = {
            'light': 300,
            'medium': 1000,
            'heavy': 3000,
        }
        return budget_estimates.get(budget, 1000)

    def _build_kwargs(self) -> dict:
        """Build kwargs for MIPROv2 constructor."""
        if self.config is None:
            return {
                'auto': 'medium',
                'num_threads': 64,
            }

        budget = getattr(self.config, 'mipro_auto', 'medium')
        num_threads = getattr(self.config, 'num_threads', 64)

        return {
            'auto': budget,
            'num_threads': num_threads,
        }
