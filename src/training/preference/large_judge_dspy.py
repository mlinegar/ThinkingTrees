"""DSPy pairwise judge module backed by the currently configured large LLM."""

from __future__ import annotations

from typing import Optional

import dspy


def _normalize_preference(raw_preference: str) -> str:
    rendered = str(raw_preference or "").strip().lower()
    if not rendered:
        return "tie"
    if "tie" in rendered or "equal" in rendered or "neither" in rendered or "both" in rendered:
        return "tie"
    if rendered == "a" or rendered.startswith("a ") or "summary a" in rendered or "response a" in rendered:
        return "A"
    if rendered == "b" or rendered.startswith("b ") or "summary b" in rendered or "response b" in rendered:
        return "B"
    if rendered in {"1", "response 1", "summary 1"}:
        return "A"
    if rendered in {"2", "response 2", "summary 2"}:
        return "B"
    if "a" in rendered and "b" not in rendered:
        return "A"
    if "b" in rendered and "a" not in rendered:
        return "B"
    return "tie"


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class LargeJudgeComparisonSignature(dspy.Signature):
    """Pairwise comparison signature for large-model judging."""

    context: str = dspy.InputField(desc="What information must be preserved.")
    original_text: str = dspy.InputField(desc="Original source text.")
    summary_a: str = dspy.InputField(desc="Candidate summary A.")
    summary_b: str = dspy.InputField(desc="Candidate summary B.")
    law_type: str = dspy.InputField(desc="Local law type: sufficiency, idempotence, merge.")

    preference: str = dspy.OutputField(desc="Preferred summary: A, B, or tie.")
    reasoning: str = dspy.OutputField(desc="Short comparison rationale.")
    score_a: str = dspy.OutputField(desc="Helpfulness score for summary A (1-5).")
    score_b: str = dspy.OutputField(desc="Helpfulness score for summary B (1-5).")
    confidence: str = dspy.OutputField(desc="Confidence in [0,1].")


class LargeJudgeComparisonModule(dspy.Module):
    """Tournament-compatible pairwise judge using the active DSPy LM."""

    def __init__(self, *, use_cot: bool = True):
        super().__init__()
        self.use_dspy_predictor = True
        self.use_dspy_prompt = False
        self.judge_backend = "large_qwen"
        self.compare = (
            dspy.ChainOfThought(LargeJudgeComparisonSignature)
            if bool(use_cot)
            else dspy.Predict(LargeJudgeComparisonSignature)
        )

    def forward(
        self,
        *,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
    ) -> dspy.Prediction:
        result = self.compare(
            context=str(context or ""),
            original_text=str(original_text or ""),
            summary_a=str(summary_a or ""),
            summary_b=str(summary_b or ""),
            law_type=str(law_type or "sufficiency"),
        )

        preference = _normalize_preference(getattr(result, "preference", "tie"))
        score_a = _safe_float(getattr(result, "score_a", 3.0), 3.0)
        score_b = _safe_float(getattr(result, "score_b", 3.0), 3.0)
        score_a = min(5.0, max(1.0, score_a))
        score_b = min(5.0, max(1.0, score_b))
        parsed_confidence = _safe_float(getattr(result, "confidence", 0.5), 0.5)
        score_diff = abs(score_a - score_b)
        derived_confidence = min(1.0, 0.5 + score_diff * 0.125)
        if preference == "tie":
            confidence = 0.5
        else:
            confidence = max(0.5, min(1.0, max(parsed_confidence, derived_confidence)))

        if preference == "A":
            ranking_score = 1
        elif preference == "B":
            ranking_score = 6
        else:
            ranking_score = 3

        return dspy.Prediction(
            preference=preference,
            reasoning=str(getattr(result, "reasoning", "") or ""),
            confidence=confidence,
            score_a=str(score_a),
            score_b=str(score_b),
            helpfulness_a=score_a,
            helpfulness_b=score_b,
            ranking_score=ranking_score,
        )


__all__ = [
    "LargeJudgeComparisonModule",
]
