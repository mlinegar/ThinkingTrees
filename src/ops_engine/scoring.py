"""
Score-centric oracle types for the OPS engine.

This module provides the core types for score-based oracles that align with
DSPy's metric optimization paradigm. The key insight is:
- Score is the PRIMARY output (0.0-1.0, where 1.0 = good/similar)
- Classification is DERIVED (threshold the score when needed)

Usage:
    from ops_engine.scoring import OracleScore, ScoringOracle, oracle_as_metric

    class MyScorer:
        def score(self, input_a: str, input_b: str, rubric: str) -> OracleScore:
            similarity = compute_similarity(input_a, input_b)
            return OracleScore(score=similarity, reasoning="...")

    # Use directly as DSPy metric
    metric = oracle_as_metric(MyScorer())
    optimizer.compile(student, trainset, metric=metric)
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable


# =============================================================================
# Score Normalization Utility
# =============================================================================

def normalize_error_to_score(error: float, max_error: float) -> float:
    """
    Convert an error value to a normalized score where 1.0 = best, 0.0 = worst.

    This is the standard pattern for converting distance/error to similarity/score:
    - score = 1.0 when error = 0
    - score = 0.0 when error >= max_error
    - Linear interpolation between

    Args:
        error: The absolute error/distance (will use abs(error) internally)
        max_error: The maximum error value (defines the scale)

    Returns:
        Normalized score in [0.0, 1.0] where higher is better

    Raises:
        ValueError: If max_error <= 0

    Examples:
        # RILE similarity using RILE_RANGE constant (200-point scale, -100 to +100)
        from src.manifesto.constants import RILE_RANGE  # = 200.0
        score = normalize_error_to_score(abs(rile_a - rile_b), RILE_RANGE)
        # rile_a=-50, rile_b=+50 → error=100 → score=0.5

        # Configurable scale (e.g., more lenient 100-point scale)
        score = normalize_error_to_score(abs(pred - gt), max_error=100.0)
        # pred=45, gt=50 → error=5 → score=0.95

        # Generic distance
        score = normalize_error_to_score(distance, max_distance)
    """
    if max_error <= 0:
        raise ValueError(f"max_error must be positive, got {max_error}")
    return max(0.0, 1.0 - abs(error) / max_error)


# =============================================================================
# Bounded Scale (Generic)
# =============================================================================

@dataclass(frozen=True)
class BoundedScale:
    """
    A bounded linear scale for score normalization.

    Represents any continuous range with defined bounds.
    Handles the math of converting distances to normalized scores.

    This is the generic foundation for domain-specific scales like RILE.
    Domain modules should create their own BoundedScale instances with
    appropriate bounds (e.g., RILE_SCALE = BoundedScale(-100.0, 100.0)).

    Examples:
        # Political positioning (-100 to +100)
        political = BoundedScale(-100.0, 100.0)
        score = political.values_to_score(pred, gt)  # 0.0-1.0

        # Percentage scale (0 to 100)
        pct = BoundedScale(0.0, 100.0)

        # Sentiment (-1 to +1)
        sentiment = BoundedScale(-1.0, 1.0)

        # Temperature (Fahrenheit, water phases)
        temp = BoundedScale(32.0, 212.0)
    """
    min_value: float
    max_value: float

    def __post_init__(self):
        """Validate that min < max."""
        if self.min_value >= self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be less than "
                f"max_value ({self.max_value})"
            )

    @property
    def range(self) -> float:
        """Total range of the scale."""
        return self.max_value - self.min_value

    def distance_to_score(self, distance: float) -> float:
        """
        Convert distance to normalized score.

        Args:
            distance: Absolute distance between two values

        Returns:
            Score in [0.0, 1.0] where 1.0 = no distance
        """
        return normalize_error_to_score(distance, max_error=self.range)

    def values_to_score(self, value_a: float, value_b: float) -> float:
        """
        Compute similarity score between two values on this scale.

        Args:
            value_a: First value
            value_b: Second value

        Returns:
            Score in [0.0, 1.0] where 1.0 = identical
        """
        return self.distance_to_score(abs(value_a - value_b))

    def clamp(self, value: float) -> float:
        """Clamp value to scale bounds."""
        return max(self.min_value, min(self.max_value, value))

    def normalize(self, value: float) -> float:
        """
        Normalize value to [0.0, 1.0] range.

        Maps min_value -> 0.0, max_value -> 1.0.
        """
        return (value - self.min_value) / self.range

    def denormalize(self, normalized: float) -> float:
        """
        Convert normalized [0.0, 1.0] back to scale value.

        Maps 0.0 -> min_value, 1.0 -> max_value.
        """
        return self.min_value + normalized * self.range


# Pre-defined scales for common use cases
UNIT_SCALE = BoundedScale(0.0, 1.0)       # 0-1 probabilities/scores
PERCENT_SCALE = BoundedScale(0.0, 100.0)  # Percentages
SYMMETRIC_SCALE = BoundedScale(-1.0, 1.0)  # Sentiment, correlation


# =============================================================================
# Oracle Score
# =============================================================================

@dataclass(frozen=True)
class OracleScore:
    """
    Primary output of a ScoringOracle.

    Convention: score uses SIMILARITY (1.0 = good, 0.0 = bad)
    This aligns with DSPy metrics and most ML conventions.

    Attributes:
        score: Primary output, 0.0-1.0 where higher = better match
        reasoning: Human-readable explanation of the score
        metadata: Optional domain-specific details (e.g., {'rile_a': 45, 'rile_b': 52})
    """

    score: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        """Validate and clamp score to [0.0, 1.0] range."""
        if not 0.0 <= self.score <= 1.0:
            clamped = max(0.0, min(1.0, self.score))
            object.__setattr__(self, 'score', clamped)

    def passes_threshold(self, threshold: float = 0.9) -> bool:
        """
        Derive classification from score.

        Use this when you need a binary decision (e.g., audit flagging).

        Args:
            threshold: Minimum score to pass (default 0.9 = 90% similar)

        Returns:
            True if score >= threshold
        """
        return self.score >= threshold

    def to_discrepancy(self) -> float:
        """
        Convert to old discrepancy convention for backward compatibility.

        Old convention: 0.0 = good, 1.0 = bad
        New convention: 1.0 = good, 0.0 = bad

        Returns:
            Discrepancy score (1.0 - similarity)
        """
        return 1.0 - self.score

    def as_metric(self) -> float:
        """
        Return score directly usable as DSPy metric.

        Since OracleScore.score is already 0.0-1.0 with 1.0 = good,
        it can be used directly as a DSPy metric return value.
        """
        return self.score

    @classmethod
    def from_discrepancy(
        cls,
        discrepancy: float,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'OracleScore':
        """
        Create from old discrepancy convention.

        Use this to migrate code that computes discrepancy (0.0 = good).

        Args:
            discrepancy: Old-style score where 0.0 = good, 1.0 = bad
            reasoning: Explanation of the score
            metadata: Optional domain-specific details

        Returns:
            OracleScore with inverted score (similarity convention)
        """
        return cls(
            score=1.0 - discrepancy,
            reasoning=reasoning,
            metadata=metadata,
        )

    def to_legacy_tuple(self, threshold: float = 0.9) -> Tuple[bool, float, str]:
        """
        Convert to old OracleJudge return format.

        Args:
            threshold: Classification threshold

        Returns:
            (is_congruent, discrepancy, reasoning) tuple
        """
        return (
            self.passes_threshold(threshold),
            self.to_discrepancy(),
            self.reasoning,
        )

    @classmethod
    def from_error(
        cls,
        error: float,
        max_error: float,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'OracleScore':
        """
        Create OracleScore from error value using standard normalization.

        This is a convenience constructor for the common pattern of converting
        an error/distance to a normalized score.

        Args:
            error: The absolute error/distance value (will be abs()'d)
            max_error: The maximum error value (defines the scale)
            reasoning: Explanation of the score
            metadata: Optional domain-specific details

        Returns:
            OracleScore with normalized score (1.0 = no error, 0.0 = max error)

        Example:
            # From RILE prediction error (using RILE_RANGE = 200.0 for full scale)
            from src.manifesto.constants import RILE_RANGE
            score = OracleScore.from_error(
                error=abs(predicted - ground_truth),
                max_error=RILE_RANGE,  # 200.0, or use a smaller value for more lenient scoring
                reasoning=f"RILE error: {abs(predicted - ground_truth):.0f}"
            )
        """
        score = normalize_error_to_score(error, max_error)
        return cls(score=score, reasoning=reasoning, metadata=metadata)


@runtime_checkable
class ScoringOracle(Protocol):
    """
    Protocol for score-centric oracles.

    Unlike the legacy OracleJudge which returns classification first,
    ScoringOracle returns a continuous score as the primary output.

    Implementations should return OracleScore with:
    - score: 0.0-1.0, where 1.0 means perfect match/similarity
    - reasoning: Human-readable explanation
    - metadata: Optional domain-specific details

    Example:
        class RILEScorer:
            def score(self, input_a: str, input_b: str, rubric: str) -> OracleScore:
                from src.manifesto.constants import RILE_RANGE
                rile_a = self._compute_rile(input_a)
                rile_b = self._compute_rile(input_b)
                similarity = normalize_error_to_score(abs(rile_a - rile_b), RILE_RANGE)
                return OracleScore(
                    score=similarity,
                    reasoning=f"RILE: {rile_a} vs {rile_b}",
                    metadata={'rile_a': rile_a, 'rile_b': rile_b}
                )
    """

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        """
        Score similarity between two inputs according to rubric.

        Args:
            input_a: First input (typically original/source text)
            input_b: Second input (typically summary/target text)
            rubric: Criteria/context for comparison

        Returns:
            OracleScore with score (1.0 = perfect match, 0.0 = no match)
        """
        ...


# =============================================================================
# Backward Compatibility Adapters
# =============================================================================

class LegacyOracleAdapter:
    """
    Adapts a new ScoringOracle to the legacy OracleJudge interface.

    Use this when you have a new-style scorer but need to pass it to
    code that expects the old (bool, float, str) return format.

    Example:
        scorer = RILEScorer()
        legacy_oracle = LegacyOracleAdapter(scorer, threshold=0.95)
        auditor = OPSAuditor(oracle=legacy_oracle)  # Old code works
    """

    def __init__(self, scoring_oracle: ScoringOracle, threshold: float = 0.9):
        """
        Initialize adapter.

        Args:
            scoring_oracle: New-style scorer implementing ScoringOracle
            threshold: Classification threshold for passes_threshold()
        """
        self._oracle = scoring_oracle
        self._threshold = threshold
        self._warned = False

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> Tuple[bool, float, str]:
        """
        Legacy interface: returns (is_congruent, discrepancy, reasoning).

        Emits deprecation warning on first use.
        """
        if not self._warned:
            warnings.warn(
                "OracleJudge tuple interface is deprecated. "
                "Use ScoringOracle.score() returning OracleScore instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned = True

        result = self._oracle.score(input_a, input_b, rubric)
        return result.to_legacy_tuple(self._threshold)


def as_scoring_oracle(legacy_oracle: Callable) -> ScoringOracle:
    """
    Wrap a legacy OracleJudge to behave as a ScoringOracle.

    Use this when you have old-style oracle code that returns
    (bool, float, str) tuples and want to use it with new-style APIs.

    Args:
        legacy_oracle: Callable with signature (input_a, input_b, rubric) -> (bool, float, str)

    Returns:
        Object implementing ScoringOracle protocol

    Example:
        old_oracle = SimpleOracleJudge()
        new_oracle = as_scoring_oracle(old_oracle)
        result = new_oracle.score(text_a, text_b, rubric)
        print(result.score)  # 0.85
    """

    class WrappedLegacyOracle:
        """Wrapper that adapts legacy oracle to ScoringOracle."""

        def score(self, input_a: str, input_b: str, rubric: str) -> OracleScore:
            is_congruent, discrepancy, reasoning = legacy_oracle(input_a, input_b, rubric)
            return OracleScore.from_discrepancy(discrepancy, reasoning)

    return WrappedLegacyOracle()


# =============================================================================
# Metric Integration
# =============================================================================

def oracle_as_metric(
    oracle: ScoringOracle,
    original_field: str = 'original',
    summary_field: str = 'summary',
    rubric_field: str = 'rubric',
) -> Callable:
    """
    Convert a ScoringOracle to a DSPy-compatible metric function.

    The oracle's score IS the metric - no complex wrapping needed.
    This is the primary way to use oracles for DSPy optimization.

    Args:
        oracle: ScoringOracle implementation
        original_field: Attribute name for original text on gold example
        summary_field: Attribute name for summary on prediction
        rubric_field: Attribute name for rubric on gold example

    Returns:
        DSPy metric function: (gold, pred, trace?) -> float

    Example:
        scorer = RILEScorer()
        metric = oracle_as_metric(scorer)

        # Use in optimization
        optimizer = dspy.BootstrapFewShot(metric=metric)
        compiled = optimizer.compile(student, trainset)
    """

    def metric(gold, pred, trace=None) -> float:
        # Extract texts from example/prediction objects
        original = getattr(gold, original_field, '') or getattr(gold, 'text', '') or str(gold)
        summary = getattr(pred, summary_field, '') or str(pred)
        rubric = getattr(gold, rubric_field, '')

        # Score and return directly
        result = oracle.score(original, summary, rubric)
        return result.as_metric()

    return metric


def oracle_as_metric_with_feedback(
    oracle: ScoringOracle,
    original_field: str = 'original',
    summary_field: str = 'summary',
    rubric_field: str = 'rubric',
) -> Callable:
    """
    Convert a ScoringOracle to a GEPA-compatible metric with feedback.

    GEPA can use feedback strings for reflection-based optimization.

    Args:
        oracle: ScoringOracle implementation
        original_field: Attribute name for original text on gold example
        summary_field: Attribute name for summary on prediction
        rubric_field: Attribute name for rubric on gold example

    Returns:
        GEPA metric function: (gold, pred, trace?, pred_name?, pred_trace?) -> dict
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dict:
        original = getattr(gold, original_field, '') or getattr(gold, 'text', '') or str(gold)
        summary = getattr(pred, summary_field, '') or str(pred)
        rubric = getattr(gold, rubric_field, '')

        result = oracle.score(original, summary, rubric)

        return {
            'score': result.score,
            'feedback': result.reasoning,
        }

    return metric
