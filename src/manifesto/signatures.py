"""
DSPy signatures for Manifesto Project RILE scoring.

This module defines signatures for:
- RILEScore: Predict RILE score from political text
- RILEComparison: Compare RILE scores between original and summary
"""

import dspy
from typing import Optional


class RILEScore(dspy.Signature):
    """
    Score political text on the left-right (RILE) scale.

    The RILE scale ranges from -100 (far left) to +100 (far right).
    The score is based on emphasis on left vs right policy positions.
    """
    task_context: str = dspy.InputField(
        desc="Explanation of the RILE scoring task and left/right indicators"
    )
    text: str = dspy.InputField(
        desc="Political manifesto text or summary to score"
    )
    rile_score: float = dspy.OutputField(
        desc="RILE score from -100 (far left) to +100 (far right)"
    )
    left_indicators: str = dspy.OutputField(
        desc="Key left-leaning positions identified in the text"
    )
    right_indicators: str = dspy.OutputField(
        desc="Key right-leaning positions identified in the text"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of how the score was determined"
    )


class PairwiseSummaryComparison(dspy.Signature):
    """
    Compare two summaries and select the one that better preserves information.

    Used by large oracle models to generate preference data for training.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria (e.g., RILE political positioning)"
    )
    original_text: str = dspy.InputField(
        desc="Original source text being summarized"
    )
    summary_a: str = dspy.InputField(
        desc="First candidate summary"
    )
    summary_b: str = dspy.InputField(
        desc="Second candidate summary"
    )
    ground_truth_score: float = dspy.InputField(
        desc="Ground truth score for the original text (e.g., RILE score)"
    )

    preferred: str = dspy.OutputField(
        desc="Which summary is better: 'A', 'B', or 'tie'"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of why this summary better preserves the information"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the preference judgment (0.0 to 1.0)"
    )
    score_estimate_a: float = dspy.OutputField(
        desc="Estimated score for summary A"
    )
    score_estimate_b: float = dspy.OutputField(
        desc="Estimated score for summary B"
    )


class RILEComparison(dspy.Signature):
    """
    Compare RILE scores between two versions of text.

    Used for auditing whether summarization preserves position information.
    """
    task_context: str = dspy.InputField(
        desc="Explanation of the RILE scoring task"
    )
    original_text: str = dspy.InputField(
        desc="Original text (more detailed)"
    )
    summary_text: str = dspy.InputField(
        desc="Summarized text"
    )
    original_rile: float = dspy.OutputField(
        desc="RILE score for original text"
    )
    summary_rile: float = dspy.OutputField(
        desc="RILE score for summary text"
    )
    score_difference: float = dspy.OutputField(
        desc="Absolute difference between scores"
    )
    is_preserved: bool = dspy.OutputField(
        desc="Whether position information is adequately preserved (diff < 10 points)"
    )
    drift_explanation: str = dspy.OutputField(
        desc="Explanation of any position drift between original and summary"
    )


# Module implementations

class RILEScorer(dspy.Module):
    """DSPy module for RILE scoring."""

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(RILEScore)

    def forward(self, text: str, task_context: str) -> dict:
        """
        Score text on the RILE scale.

        Args:
            text: Political text to score
            task_context: Explanation of the scoring task

        Returns:
            Dictionary with score and analysis
        """
        result = self.score(task_context=task_context, text=text)
        return {
            'rile_score': result.rile_score,
            'left_indicators': result.left_indicators,
            'right_indicators': result.right_indicators,
            'reasoning': result.reasoning,
        }


class RILEComparator(dspy.Module):
    """DSPy module for comparing RILE scores between texts."""

    def __init__(self, threshold: float = 10.0):
        """
        Initialize comparator.

        Args:
            threshold: Maximum acceptable score difference for preservation
        """
        super().__init__()
        self.compare = dspy.ChainOfThought(RILEComparison)
        self.threshold = threshold

    def forward(self, original_text: str, summary_text: str, task_context: str) -> dict:
        """
        Compare RILE positions between original and summary.

        Args:
            original_text: Original text
            summary_text: Summary text
            task_context: Explanation of the scoring task

        Returns:
            Dictionary with comparison results
        """
        result = self.compare(
            task_context=task_context,
            original_text=original_text,
            summary_text=summary_text
        )
        return {
            'original_rile': result.original_rile,
            'summary_rile': result.summary_rile,
            'score_difference': result.score_difference,
            'is_preserved': result.is_preserved,
            'drift_explanation': result.drift_explanation,
        }


