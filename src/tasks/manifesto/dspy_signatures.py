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
import re
import dspy
from typing import Any, Optional

from src.core.output_parser import NormalizedOutputAccessor
from src.core.prompting import parse_numeric_score
from .constants import RILE_MIN, RILE_MAX

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
    score: float = dspy.OutputField(
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

    A compact signature with a single output field to reduce format drift
    and truncation during optimization/evaluation loops.
    """
    task_context: str = dspy.InputField(
        desc="Scoring task description and criteria"
    )
    text: str = dspy.InputField(
        desc="Text to score"
    )
    score: float = dspy.OutputField(
        desc=(
            "Numeric score on the exact scale defined in task_context. "
            "Output a single number only (format examples, do not copy: -12, 0, 37.5; "
            "no markdown/backticks/code fences, no extra text); "
            "do not invent an alternate scale; "
            "do not output multiple numbers, ranges, or lists."
        )
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
    reference_score: float = dspy.InputField(
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


def _coerce_rile_score(raw_value: Any, raw_result: Any) -> Optional[float]:
    if raw_value is not None:
        parsed = parse_numeric_score(
            str(raw_value),
            min_value=RILE_MIN,
            max_value=RILE_MAX,
            allow_llm_fallback=False,
        )
        if parsed is not None:
            return parsed

    # Fallback: parse the last in-range number from the full (possibly messy)
    # result representation. This recovers from format drift where the adapter
    # returns a partial field or different casing/structure.
    if raw_result is not None:
        parsed = parse_numeric_score(
            str(raw_result),
            min_value=RILE_MIN,
            max_value=RILE_MAX,
            allow_llm_fallback=False,
        )
        if parsed is not None:
            return parsed

    return None


# Module implementations

class RILEScorer(dspy.Module):
    """DSPy module for RILE scoring."""

    def __init__(self, use_cot: bool = False):
        super().__init__()
        # Default to the compact signature to keep scorer outputs short and
        # stable during GEPA optimization/evaluation loops.
        # Also cap completion tokens so the scorer cannot ramble; we only need
        # a single numeric value.
        scorer_max_tokens = 32
        scorer_temperature = 0.0
        score_signature = SimpleScore
        if use_cot:
            self.score = dspy.ChainOfThought(
                score_signature,
                max_tokens=scorer_max_tokens,
                temperature=scorer_temperature,
            )
        else:
            self.score = dspy.Predict(
                score_signature,
                max_tokens=scorer_max_tokens,
                temperature=scorer_temperature,
            )

    def forward(
        self,
        text: str = None,
        task_context: str = None,
        # Training example format (alternative signature)
        summary: str = None,
        rubric: str = None,
        original_content: str = None,  # Accepted but not used for pure scoring
        dspy_config: Optional[dict[str, Any]] = None,
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

        request_config: Optional[dict[str, Any]] = None
        if isinstance(dspy_config, dict) and dspy_config:
            request_config = dict(dspy_config)

        def _extract_lm_response_from_exception(exc: Exception) -> Optional[str]:
            message = str(exc or "")
            if not message:
                return None
            match = re.search(
                r"LM Response:\s*(.*?)(?:\n\nExpected to find|\Z)",
                message,
                flags=re.DOTALL,
            )
            if not match:
                return None
            candidate = match.group(1).strip()
            return candidate or None

        def _score_once(task_ctx: str) -> Optional[float]:
            try:
                if request_config is None:
                    result = self.score(task_context=task_ctx, text=actual_text)
                else:
                    result = self.score(
                        task_context=task_ctx,
                        text=actual_text,
                        config=request_config,
                    )
            except Exception as exc:
                lm_response = _extract_lm_response_from_exception(exc)
                if lm_response:
                    parsed = parse_numeric_score(
                        lm_response,
                        min_value=RILE_MIN,
                        max_value=RILE_MAX,
                        allow_llm_fallback=False,
                    )
                    if parsed is not None:
                        return parsed
                logger.warning("RILEScorer prediction failed; defaulting to neutral. Error: %s", exc)
                return None

            accessor = NormalizedOutputAccessor(result)
            return _coerce_rile_score(accessor.get("score", None), result)

        raw_score = _score_once(actual_context)

        if raw_score is None:
            retry_context = (
                f"{actual_context}\n\n"
                "IMPORTANT: Output ONLY the numeric score as plain text. "
                "No words, labels, units, punctuation (other than a leading '-' and optional decimal point)."
            )
            raw_score = _score_once(retry_context)

        if raw_score is None:
            logger.warning("RILEScorer could not parse score after retry; defaulting to neutral score 0.0")
            raw_score = 0.0
        normalized = (raw_score - RILE_MIN) / (RILE_MAX - RILE_MIN)
        normalized = max(0.0, min(1.0, normalized))

        return {'score': normalized}


class RILEComparator(dspy.Module):
    """DSPy module for comparing RILE scores between texts."""

    def __init__(self, threshold: float = 10.0, use_cot: bool = False):
        """
        Initialize comparator.

        Args:
            threshold: Maximum acceptable score difference for preservation
        """
        super().__init__()
        if use_cot:
            self.compare = dspy.ChainOfThought(RILEComparison)
        else:
            self.compare = dspy.Predict(RILEComparison)
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

        raw_original = _coerce_rile_score(accessor.get('original_rile', None), result)
        raw_summary = _coerce_rile_score(accessor.get('summary_rile', None), result)
        raw_original = 0.0 if raw_original is None else raw_original
        raw_summary = 0.0 if raw_summary is None else raw_summary

        norm_original = (raw_original - RILE_MIN) / (RILE_MAX - RILE_MIN)
        norm_summary = (raw_summary - RILE_MIN) / (RILE_MAX - RILE_MIN)
        norm_original = max(0.0, min(1.0, norm_original))
        norm_summary = max(0.0, min(1.0, norm_summary))

        return {
            'original_rile': norm_original,
            'summary_rile': norm_summary,
            'score_difference': abs(norm_original - norm_summary),
            'is_preserved': accessor.get('is_preserved', True),
            'drift_explanation': accessor.get('drift_explanation', ''),
        }
