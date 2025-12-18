"""
Position Oracle for RILE-based auditing.

This module provides an oracle that uses RILE (left-right) scoring
to verify that political position information is preserved through
summarization layers. It integrates with the OPS auditor.

New API (ScoringOracle):
    scorer = RILESimilarityScorer()
    result = scorer.score(original, summary, rubric)
    print(result.score)  # 0.95 (similarity, 1.0 = identical RILE)

Legacy API (OracleJudge):
    oracle = RILEPositionOracle()
    is_congruent, discrepancy, reasoning = oracle(original, summary, rubric)
"""

from typing import Tuple, Optional
import warnings
import dspy

from .constants import RILE_RANGE
from .rubrics import RILE_TASK_CONTEXT
from .signatures import RILEScorer
from src.ops_engine.scoring import OracleScore, normalize_error_to_score


# =============================================================================
# Score-Centric Oracles (New, Preferred API)
# =============================================================================

class RILESimilarityScorer:
    """
    Score-centric RILE oracle implementing ScoringOracle protocol.

    This is the preferred API for new code. Returns OracleScore with
    similarity (1.0 = identical RILE scores) as primary output.

    Example:
        scorer = RILESimilarityScorer()
        result = scorer.score(original_text, summary_text, rubric)
        print(result.score)  # 0.95
        print(result.passes_threshold(0.9))  # True
        print(result.metadata)  # {'rile_original': 45, 'rile_summary': 52, ...}

    The score is calculated as: 1.0 - (|rile_a - rile_b| / RILE_RANGE)
    Since RILE ranges from -100 to +100 (200 points), this gives:
    - Score 1.0 = identical RILE scores
    - Score 0.95 = 10 point difference
    - Score 0.0 = maximum difference (200 points)
    """

    def __init__(
        self,
        task_context: Optional[str] = None,
    ):
        """
        Initialize the RILE similarity scorer.

        Args:
            task_context: The RILE scoring task explanation (uses default if None)
        """
        self.task_context = task_context or RILE_TASK_CONTEXT
        self._rile_scorer = RILEScorer()

    def score(
        self,
        input_a: str,
        input_b: str,
        rubric: str,
    ) -> OracleScore:
        """
        Score RILE similarity between two texts.

        Args:
            input_a: First text (typically original/detailed)
            input_b: Second text (typically summary)
            rubric: Comparison criteria (can provide additional context)

        Returns:
            OracleScore with similarity (1.0 = identical RILE scores)
        """
        try:
            # Score both texts
            result_a = self._rile_scorer(text=input_a, task_context=self.task_context)
            result_b = self._rile_scorer(text=input_b, task_context=self.task_context)

            rile_a = float(result_a['rile_score'])
            rile_b = float(result_b['rile_score'])

            # Calculate similarity (1.0 = identical, 0.0 = max difference)
            diff = abs(rile_a - rile_b)
            similarity = normalize_error_to_score(diff, max_error=RILE_RANGE)

            # Build reasoning
            reasoning = f"RILE: {rile_a:.1f} vs {rile_b:.1f}, diff={diff:.1f}"

            return OracleScore(
                score=similarity,
                reasoning=reasoning,
                metadata={
                    'rile_original': rile_a,
                    'rile_summary': rile_b,
                    'difference': diff,
                    'left_indicators_original': result_a.get('left_indicators', ''),
                    'right_indicators_original': result_a.get('right_indicators', ''),
                    'left_indicators_summary': result_b.get('left_indicators', ''),
                    'right_indicators_summary': result_b.get('right_indicators', ''),
                },
            )

        except Exception as e:
            # Return low score on error
            return OracleScore(
                score=0.0,
                reasoning=f"Oracle error: {str(e)}",
            )


# =============================================================================
# Legacy Oracles (Deprecated, for backward compatibility)
# =============================================================================

class RILEPositionOracle:
    """
    DEPRECATED: Use RILESimilarityScorer instead.

    Oracle for auditing RILE position preservation.
    Returns legacy (bool, float, str) tuple format.
    """

    def __init__(
        self,
        threshold: float = 10.0,
        task_context: Optional[str] = None
    ):
        """
        Initialize the RILE position oracle.

        Args:
            threshold: Maximum acceptable RILE score difference (default 10 points)
            task_context: The RILE scoring task explanation (uses default if None)
        """
        self.threshold = threshold
        self.task_context = task_context or RILE_TASK_CONTEXT
        self._scorer = RILESimilarityScorer(task_context=self.task_context)
        self._warned = False

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str
    ) -> Tuple[bool, float, str]:
        """
        Compare two texts for RILE equivalence.

        DEPRECATED: Use RILESimilarityScorer.score() instead.

        Returns:
            Tuple of (is_congruent, discrepancy_score, reasoning)
        """
        if not self._warned:
            warnings.warn(
                "RILEPositionOracle is deprecated. Use RILESimilarityScorer instead, "
                "which returns OracleScore with score (1.0 = good) as primary output.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned = True

        # Use new scorer internally
        result = self._scorer.score(input_a, input_b, rubric)

        # Convert score back to discrepancy for legacy format
        discrepancy = result.to_discrepancy()

        # Use threshold to determine congruence
        # threshold is in RILE points (e.g., 10), convert to similarity threshold
        # 10 point diff = 10/200 = 0.05 discrepancy = 0.95 similarity
        similarity_threshold = normalize_error_to_score(self.threshold, max_error=RILE_RANGE)
        is_congruent = result.score >= similarity_threshold

        # Build reasoning with threshold info
        reasoning = f"{result.reasoning}, threshold: {self.threshold} points"

        return is_congruent, discrepancy, reasoning


class SimpleRILEOracle:
    """
    Simplified RILE oracle for testing without LLM.

    Uses keyword matching to estimate left/right emphasis.
    Useful for testing the pipeline before using real LLM calls.
    """

    # Keywords associated with left positions
    LEFT_KEYWORDS = {
        'welfare', 'equality', 'nationalization', 'regulation',
        'peace', 'internationalism', 'labor', 'workers', 'union',
        'education', 'healthcare', 'environment', 'redistribution',
        'public', 'social', 'collective', 'planning', 'intervention',
        'progressive', 'anti-military', 'cooperation', 'solidarity'
    }

    # Keywords associated with right positions
    RIGHT_KEYWORDS = {
        'free market', 'enterprise', 'deregulation', 'privatization',
        'military', 'defense', 'tradition', 'national pride', 'patriot',
        'law and order', 'family values', 'property', 'individual',
        'business', 'tax cut', 'incentive', 'competition', 'freedom',
        'conservative', 'heritage', 'sovereignty', 'security'
    }

    def __init__(self, threshold: float = 10.0):
        """
        Initialize simple RILE oracle.

        Args:
            threshold: Maximum acceptable RILE score difference
        """
        self.threshold = threshold

    def _estimate_rile(self, text: str) -> float:
        """
        Estimate RILE score based on keyword frequency.

        Returns score from -100 to +100.
        """
        text_lower = text.lower()
        words = text_lower.split()

        left_count = sum(1 for word in words if any(kw in word for kw in self.LEFT_KEYWORDS))
        right_count = sum(1 for word in words if any(kw in word for kw in self.RIGHT_KEYWORDS))

        total = left_count + right_count
        if total == 0:
            return 0.0

        # Score from -100 (all left) to +100 (all right)
        left_share = left_count / total
        right_share = right_count / total

        return (right_share - left_share) * 100

    def __call__(
        self,
        input_a: str,
        input_b: str,
        rubric: str
    ) -> Tuple[bool, float, str]:
        """
        Compare two texts for RILE equivalence using keyword matching.

        Args:
            input_a: First text
            input_b: Second text
            rubric: Information preservation rubric (ignored in simple oracle)

        Returns:
            Tuple of (is_congruent, discrepancy_score, reasoning)
        """
        score_a = self._estimate_rile(input_a)
        score_b = self._estimate_rile(input_b)

        diff = abs(score_a - score_b)
        discrepancy = min(diff / RILE_RANGE, 1.0)
        is_congruent = diff <= self.threshold

        reasoning = (
            f"Keyword RILE: {score_a:.1f} vs {score_b:.1f}, "
            f"Diff: {diff:.1f}, Threshold: {self.threshold}"
        )

        return is_congruent, discrepancy, reasoning


def create_position_oracle(
    use_llm: bool = True,
    threshold: float = 10.0,
    task_context: Optional[str] = None
) -> 'RILEPositionOracle':
    """
    Factory function to create a position oracle.

    Args:
        use_llm: If True, uses LLM-based oracle; otherwise uses simple keyword oracle
        threshold: RILE score difference threshold
        task_context: Custom task context for RILE scoring

    Returns:
        Configured oracle instance
    """
    if use_llm:
        return RILEPositionOracle(
            threshold=threshold,
            task_context=task_context
        )
    else:
        return SimpleRILEOracle(threshold=threshold)
