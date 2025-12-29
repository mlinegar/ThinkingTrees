"""Unified oracle calling utilities.

This module provides a consistent interface for calling oracles across the codebase,
handling both new-style ScoringOracle and legacy callable interfaces.
"""
from typing import Callable, Tuple, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ops_engine.scoring import ScoringOracle, OracleScore


def is_scoring_oracle(oracle_fn) -> bool:
    """
    Check if oracle implements ScoringOracle protocol.

    ScoringOracle has a `score(input_a, input_b, rubric)` method that returns OracleScore.
    Legacy oracles are callables returning (is_congruent, discrepancy, reasoning).

    Args:
        oracle_fn: Either a ScoringOracle instance or a legacy callable

    Returns:
        True if oracle_fn implements ScoringOracle protocol
    """
    return hasattr(oracle_fn, 'score') and callable(getattr(oracle_fn, 'score', None))


def call_oracle(
    oracle_fn,
    input_a: str,
    input_b: str,
    rubric: str,
    threshold: float = 0.1,
) -> Tuple[bool, float, str]:
    """
    Unified oracle calling with auto-detection of interface type.

    This function handles both:
    - ScoringOracle: Has `.score()` method returning OracleScore
    - Legacy callable: Returns (is_congruent, discrepancy, reasoning) directly

    Args:
        oracle_fn: Either ScoringOracle or legacy callable
        input_a: First input to compare
        input_b: Second input to compare
        rubric: The evaluation rubric/context
        threshold: Discrepancy threshold for congruence determination

    Returns:
        Tuple of (is_congruent, discrepancy, reasoning)
    """
    from src.ops_engine.scoring import OracleScore

    if is_scoring_oracle(oracle_fn):
        # New-style ScoringOracle
        result: OracleScore = oracle_fn.score(input_a, input_b, rubric)
        discrepancy = result.to_discrepancy()
        is_congruent = discrepancy <= threshold
        return is_congruent, discrepancy, result.reasoning
    else:
        # Legacy callable oracle
        return oracle_fn(input_a, input_b, rubric)


def adapt_legacy_oracle(
    oracle_fn: Callable[[str, str, str], Tuple[bool, float, str]],
) -> "ScoringOracle":
    """
    Wrap legacy oracle callable in ScoringOracle interface.

    Use this to convert old-style oracle functions to the new ScoringOracle
    protocol for consistent handling.

    Args:
        oracle_fn: Legacy callable with signature (input_a, input_b, rubric) -> (is_congruent, discrepancy, reasoning)

    Returns:
        A ScoringOracle-compatible wrapper
    """
    from src.ops_engine.scoring import OracleScore

    class LegacyOracleAdapter:
        """Adapter wrapping legacy oracle in ScoringOracle interface."""

        def __init__(self, fn: Callable[[str, str, str], Tuple[bool, float, str]]):
            self._fn = fn

        def score(self, input_a: str, input_b: str, rubric: str) -> OracleScore:
            """Call legacy oracle and wrap result in OracleScore."""
            is_congruent, discrepancy, reasoning = self._fn(input_a, input_b, rubric)
            return OracleScore.from_discrepancy(discrepancy, reasoning)

    return LegacyOracleAdapter(oracle_fn)


def get_oracle_score(
    oracle_fn,
    input_a: str,
    input_b: str,
    rubric: str,
) -> "OracleScore":
    """
    Get OracleScore from any oracle type.

    Unlike call_oracle which returns the tuple, this always returns OracleScore
    for use cases that need the full score object.

    Args:
        oracle_fn: Either ScoringOracle or legacy callable
        input_a: First input to compare
        input_b: Second input to compare
        rubric: The evaluation rubric/context

    Returns:
        OracleScore object
    """
    from src.ops_engine.scoring import OracleScore

    if is_scoring_oracle(oracle_fn):
        return oracle_fn.score(input_a, input_b, rubric)
    else:
        # Legacy callable - wrap in OracleScore
        is_congruent, discrepancy, reasoning = oracle_fn(input_a, input_b, rubric)
        return OracleScore.from_discrepancy(discrepancy, reasoning)
