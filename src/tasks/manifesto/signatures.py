"""
DSPy signatures for Manifesto Project RILE scoring.

This module provides domain-specific signatures that extend the generic
MetricScore pattern from src.core.signatures for political text scoring.

Signatures:
- RILEScore: Score political text on the left-right RILE scale
- SimpleScore: Simplified scorer for model reliability
- PairwiseSummaryComparison: Compare summaries for preference generation
- RILEComparison: Audit whether summarization preserves political position

See src.core.signatures.MetricScore for the generic scoring pattern.
"""

import logging
import dspy
from typing import Optional

from src.core.output_parser import get_field, NormalizedOutputAccessor

logger = logging.getLogger(__name__)


class RILEScore(dspy.Signature):
    """
    Score text on the RILE (Right-Left) political scale.

    Domain-specific extension of MetricScore for political manifesto scoring.
    Scale: -100 (far left) to +100 (far right).
    """
    task_context: str = dspy.InputField(
        desc="Explanation of the scoring task and dimension indicators"
    )
    text: str = dspy.InputField(
        desc="Text to score"
    )
    rile_score: float = dspy.OutputField(
        desc="Score on the specified scale. Output a single number."
    )
    left_indicators: str = dspy.OutputField(
        desc="Key indicators for the lower end of the scale"
    )
    right_indicators: str = dspy.OutputField(
        desc="Key indicators for the higher end of the scale"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of how the score was determined"
    )


class SimpleScore(dspy.Signature):
    """
    Score text on a bounded numeric scale with minimal output fields.

    A simpler signature with fewer output fields to reduce JSON parsing failures.
    Use this when model reliability is an issue.
    """
    task_context: str = dspy.InputField(
        desc="Scoring task description and criteria"
    )
    text: str = dspy.InputField(
        desc="Text to score"
    )
    score: float = dspy.OutputField(
        desc="Score on the specified scale. Output a single number."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the score"
    )


class PairwiseSummaryComparison(dspy.Signature):
    """
    Compare two summaries and select the one that better preserves information.

    Used by oracle models to generate preference data for training.
    See src.core.signatures.PairwiseComparison for the generic version.
    """
    rubric: str = dspy.InputField(
        desc="Information preservation criteria"
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
        desc="Ground truth score for the original text"
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
        desc="Estimated score for summary A. Output a single number."
    )
    score_estimate_b: float = dspy.OutputField(
        desc="Estimated score for summary B. Output a single number."
    )


class RILEComparison(dspy.Signature):
    """
    Compare scores between original and summarized text.

    Used for auditing whether summarization preserves target information.
    """
    task_context: str = dspy.InputField(
        desc="Explanation of the scoring task"
    )
    original_text: str = dspy.InputField(
        desc="Original text (more detailed)"
    )
    summary_text: str = dspy.InputField(
        desc="Summarized text"
    )
    original_rile: float = dspy.OutputField(
        desc="Score for original text. Output a single number."
    )
    summary_rile: float = dspy.OutputField(
        desc="Score for summary text. Output a single number."
    )
    score_difference: float = dspy.OutputField(
        desc="Absolute difference between scores. Output a single number."
    )
    is_preserved: bool = dspy.OutputField(
        desc="Whether information is adequately preserved"
    )
    drift_explanation: str = dspy.OutputField(
        desc="Explanation of any drift between original and summary"
    )


# Module implementations

class RILEScorer(dspy.Module):
    """DSPy module for RILE scoring."""

    def __init__(self):
        super().__init__()
        self.score = dspy.ChainOfThought(RILEScore)

    def forward(
        self,
        text: str = None,
        task_context: str = None,
        # Training example format (alternative signature)
        summary: str = None,
        rubric: str = None,
        original_content: str = None,  # Accepted but not used for pure scoring
    ) -> dict:
        """
        Score text on the RILE scale.

        Accepts either:
        - text + task_context (original format)
        - summary + rubric + original_content (training example format)

        Args:
            text: Political text to score
            task_context: Explanation of the scoring task
            summary: Alternative name for text (from training examples)
            rubric: Alternative name for task_context (from training examples)
            original_content: Ignored, accepted for compatibility

        Returns:
            Dictionary with score and analysis
        """
        # Support both calling conventions
        actual_text = text if text is not None else summary
        actual_context = task_context if task_context is not None else rubric

        if actual_text is None:
            raise ValueError("Either 'text' or 'summary' must be provided")
        if actual_context is None:
            raise ValueError("Either 'task_context' or 'rubric' must be provided")

        result = self.score(task_context=actual_context, text=actual_text)

        # Use normalized accessor to handle key casing variations
        # (e.g., LLM may output 'RILE_score' or 'riLE_score' instead of 'rile_score')
        accessor = NormalizedOutputAccessor(result)

        return {
            'rile_score': accessor.get('rile_score', 0.0),
            'left_indicators': accessor.get('left_indicators', ''),
            'right_indicators': accessor.get('right_indicators', ''),
            'reasoning': accessor.get('reasoning', ''),
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

        # Use normalized accessor to handle key casing variations
        accessor = NormalizedOutputAccessor(result)

        return {
            'original_rile': accessor.get('original_rile', 0.0),
            'summary_rile': accessor.get('summary_rile', 0.0),
            'score_difference': accessor.get('score_difference', 0.0),
            'is_preserved': accessor.get('is_preserved', True),
            'drift_explanation': accessor.get('drift_explanation', ''),
        }


