from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from .ids import stable_id
from .serialization import as_compact_str, to_jsonable

MIN_PROPENSITY = 1e-8
DEFAULT_PROPENSITY = 1.0
MAX_PROPENSITY = 1.0


def _normalize_propensity(value: Optional[float], name: str) -> float:
    if value is None:
        return DEFAULT_PROPENSITY
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > MAX_PROPENSITY:
        raise ValueError(f"{name} must be finite and in (0, {MAX_PROPENSITY}], got {value!r}")
    return parsed


@dataclass(frozen=True)
class IPWMetadata:
    """Design-based sampling metadata for unbiased/consistent risk estimation."""

    doc_propensity: float = DEFAULT_PROPENSITY
    node_propensity: float = DEFAULT_PROPENSITY
    label_propensity: float = DEFAULT_PROPENSITY
    joint_propensity: Optional[float] = None
    sampling_scheme: Optional[str] = None
    node_type: Optional[str] = None

    def effective_joint_propensity(self, *, min_propensity: float = MIN_PROPENSITY) -> float:
        doc_p = _normalize_propensity(self.doc_propensity, "doc_propensity")
        node_p = _normalize_propensity(self.node_propensity, "node_propensity")
        label_p = _normalize_propensity(self.label_propensity, "label_propensity")
        if self.joint_propensity is None:
            joint = doc_p * node_p * label_p
        else:
            joint = _normalize_propensity(self.joint_propensity, "joint_propensity")
        return max(float(min_propensity), float(joint))

    def ipw_weight(
        self,
        *,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> float:
        weight = 1.0 / self.effective_joint_propensity(min_propensity=min_propensity)
        if max_weight is not None:
            weight = min(weight, float(max_weight))
        return float(weight)


@dataclass
class PairwisePreference:
    """Backend-agnostic pairwise preference record (convertible to training formats)."""

    example_id: str
    candidate_a: Any
    candidate_b: Any
    preferred: Literal["A", "B", "tie"]
    confidence: float

    # Optional context/diagnostics
    input: Any = ""
    rubric: str = ""
    reference: Optional[float] = None
    score_a: Optional[float] = None
    score_b: Optional[float] = None
    reasoning: str = ""
    ipw: IPWMetadata = field(default_factory=IPWMetadata)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def pair_id(self, *, n_chars: int = 16) -> str:
        payload = {
            "example_id": self.example_id,
            "rubric": self.rubric,
            "input": to_jsonable(self.input),
            "candidate_a": to_jsonable(self.candidate_a),
            "candidate_b": to_jsonable(self.candidate_b),
            "preferred": self.preferred,
        }
        return stable_id(payload, n_chars=n_chars)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id(),
            "example_id": self.example_id,
            "input": to_jsonable(self.input),
            "rubric": self.rubric,
            "candidate_a": to_jsonable(self.candidate_a),
            "candidate_b": to_jsonable(self.candidate_b),
            "preferred": self.preferred,
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
            "reference": self.reference,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "doc_propensity": self.ipw.doc_propensity,
            "node_propensity": self.ipw.node_propensity,
            "label_propensity": self.ipw.label_propensity,
            "joint_propensity": self.ipw.joint_propensity,
            "sampling_scheme": self.ipw.sampling_scheme,
            "node_type": self.ipw.node_type,
            "sample_weight": self.ipw.ipw_weight(),
            "timestamp": self.timestamp,
        }

    def to_training_preference_pair(self) -> Any:
        """Convert to the repo's canonical PreferencePair type (lazy import)."""
        from src.training.preference.types import PreferencePair

        reference_score = float(self.reference) if self.reference is not None else 0.0
        return PreferencePair(
            pair_id=self.pair_id(),
            source_example_id=str(self.example_id),
            original_text=as_compact_str(self.input),
            rubric=str(self.rubric or ""),
            reference_score=reference_score,
            summary_a=as_compact_str(self.candidate_a),
            summary_b=as_compact_str(self.candidate_b),
            preferred=self.preferred,
            reasoning=str(self.reasoning or ""),
            confidence=float(self.confidence),
            doc_propensity=float(self.ipw.doc_propensity),
            node_propensity=float(self.ipw.node_propensity),
            label_propensity=float(self.ipw.label_propensity),
            joint_propensity=self.ipw.joint_propensity,
            sampling_scheme=self.ipw.sampling_scheme,
            node_type=self.ipw.node_type,
            score_estimate_a=self.score_a,
            score_estimate_b=self.score_b,
        )
