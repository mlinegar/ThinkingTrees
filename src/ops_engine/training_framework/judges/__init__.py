"""
Judge Registry for Pairwise Comparison.

Provides a registry pattern for pairwise comparison judges, similar to OptimizerRegistry.
Judges compare two summaries and determine which better preserves task-relevant information.

Usage:
    from src.ops_engine.training_framework.judges import get_judge, JudgeConfig

    # Get a judge by name
    config = JudgeConfig(type="genrm", base_url="http://localhost:8001/v1")
    judge = get_judge("genrm", config)

    # Compare summaries
    result = judge.compare(
        context="Preserve political position information",
        original_text="...",
        summary_a="...",
        summary_b="...",
    )
    print(result.preferred)  # "A", "B", or "tie"

Available judges:
    - dspy: DSPy-based judge using PairwiseComparison signature (optimizable)
    - genrm: NVIDIA Qwen3-Nemotron GenRM model
    - oracle: Oracle-based judge using scoring function
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

from .base import (
    BaseJudge,
    CompilableJudge,
    AsyncJudge,
    JudgeConfig,
    JudgeResult,
    JudgeError,
    is_judge_error,
    judge_result_from_dict,
)

logger = logging.getLogger(__name__)

# Registry of judge implementations
_JUDGE_REGISTRY: Dict[str, Callable[..., BaseJudge]] = {}


def register_judge(name: str):
    """
    Decorator to register a judge factory.

    Args:
        name: Name to register the judge under

    Returns:
        Decorator function
    """
    def decorator(factory_fn: Callable[..., BaseJudge]):
        _JUDGE_REGISTRY[name.lower()] = factory_fn
        return factory_fn
    return decorator


def get_judge(
    name: str,
    config: Optional[JudgeConfig] = None,
    **kwargs,
) -> BaseJudge:
    """
    Get a judge by name from the registry.

    Args:
        name: Judge name ("dspy", "genrm", "oracle")
        config: Judge configuration
        **kwargs: Additional arguments passed to factory

    Returns:
        Configured judge instance

    Raises:
        ValueError: If judge name is not registered
    """
    name_lower = name.lower()

    if name_lower not in _JUDGE_REGISTRY:
        available = list(_JUDGE_REGISTRY.keys())
        raise ValueError(
            f"Unknown judge type: '{name}'. Available: {available}"
        )

    factory = _JUDGE_REGISTRY[name_lower]
    return factory(config=config, **kwargs)


def list_judges() -> list:
    """Return list of registered judge names."""
    return list(_JUDGE_REGISTRY.keys())


# =============================================================================
# Register built-in judges
# =============================================================================

@register_judge("dspy")
def _create_dspy_judge(
    config: Optional[JudgeConfig] = None,
    **kwargs,
) -> BaseJudge:
    """Create DSPy-based judge."""
    from .dspy import DSPyJudge
    return DSPyJudge(config=config)


@register_judge("genrm")
def _create_genrm_judge(
    config: Optional[JudgeConfig] = None,
    **kwargs,
) -> BaseJudge:
    """Create GenRM-based judge."""
    from .genrm import GenRMJudgeWrapper
    return GenRMJudgeWrapper(config=config)


@register_judge("oracle")
def _create_oracle_judge(
    config: Optional[JudgeConfig] = None,
    oracle_fn: Optional[Callable[[str], float]] = None,
    tie_margin: float = 0.05,
    **kwargs,
) -> BaseJudge:
    """
    Create oracle-based judge.

    Args:
        config: Judge configuration
        oracle_fn: Scoring function (required)
        tie_margin: Normalized error margin for ties

    Raises:
        ValueError: If oracle_fn not provided
    """
    from .oracle import OracleJudge

    # Get oracle_fn from config or kwarg
    fn = oracle_fn
    if fn is None and config is not None:
        fn = config.oracle_fn

    if fn is None:
        raise ValueError(
            "oracle_fn is required for oracle judge. "
            "Pass it as kwarg or set config.oracle_fn"
        )

    if config is None:
        config = JudgeConfig(type="oracle", tie_margin=tie_margin)

    return OracleJudge(oracle_fn=fn, config=config)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Registry functions
    "get_judge",
    "list_judges",
    "register_judge",
    # Base types
    "BaseJudge",
    "CompilableJudge",
    "AsyncJudge",
    "JudgeConfig",
    "JudgeResult",
    "JudgeError",
    # Utilities
    "is_judge_error",
    "judge_result_from_dict",
]
