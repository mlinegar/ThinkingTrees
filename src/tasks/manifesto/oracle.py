"""
Position Oracle for RILE-based auditing.

This module provides an oracle that uses RILE (left-right) scoring
to verify that political position information is preserved through
summarization layers. It integrates with the OPS auditor.

Usage:
    scorer = create_rile_scorer()
    result = scorer.score(original, summary, rubric)
    print(result.score)  # 0.95 (similarity, 1.0 = identical RILE)
"""

from typing import Optional

from .constants import RILE_MIN, RILE_MAX
from .rubrics import RILE_TASK_CONTEXT
from .signatures import RILEScorer
from src.ops_engine.scoring import SimilarityScorer, BoundedScale


# RILE scale: -100 (left) to +100 (right)
RILE_SCALE = BoundedScale(RILE_MIN, RILE_MAX)


def create_rile_scorer(task_context: Optional[str] = None) -> SimilarityScorer:
    """
    Create a RILE similarity scorer.

    Returns a generic SimilarityScorer configured for RILE political positioning.
    The score is: 1.0 - abs(rile_a - rile_b) / 200

    Args:
        task_context: Optional context for RILE scoring (uses default if None)

    Returns:
        SimilarityScorer configured for RILE scale

    Example:
        scorer = create_rile_scorer()
        result = scorer.score(original_text, summary_text, "")
        print(result.score)  # 0.95 (10-point difference)
    """
    context = task_context or RILE_TASK_CONTEXT
    rile_dspy = RILEScorer()

    def extract_rile(text: str) -> float:
        """Extract RILE score from text using LLM."""
        result = rile_dspy(text=text, task_context=context)
        return float(result['rile_score'])

    return SimilarityScorer(
        value_extractor=extract_rile,
        scale=RILE_SCALE,
        name="RILE",
    )
