"""
Generalized preference types for ThinkingTrees.

This module provides type-agnostic preference collection -- supporting pairwise
preferences, scalar ratings, written critiques, and arbitrary combinations.
All preference types carry IPW propensity annotations from the audit sampling
design, enabling unbiased downstream estimation and training.

The key abstraction is the PreferenceRequest/PreferenceResponse pair:
- PreferenceRequest declares what preference data is wanted (via PreferenceDimension)
- PreferenceResponse carries whatever the collector provides

Responses are always convertible to:
- SupervisionDataset / ResponseJudgment / ComparativeJudgment
- BinaryComparison (backward-compatible binary projection)
- DSPy metric dict {'score': float, 'feedback': str} (optimizer-compatible)
- FlaggedItem update fields (human review bridge)

Usage:
    from src.preference_collection import PreferenceRequest, PreferenceResponse, PreferenceDimension

    # Create a pairwise request
    request = PreferenceRequest(
        request_id="req_1",
        text_a="Summary A...",
        text_b="Summary B...",
        original_text="Original document...",
        rubric="Preserve political positions",
    )

    # Or a scalar rating request
    request = PreferenceRequest(
        request_id="req_2",
        text_a="Summary to rate...",
        original_text="Original document...",
        rubric="Rate faithfulness 1-5",
        dimensions=[PreferenceDimension(kind="scalar", name="faithfulness", scale=(1.0, 5.0))],
    )

    # Responses are flexible
    response = PreferenceResponse(
        request_id="req_1",
        preferred="A",
        scores={"faithfulness": 4.2},
        critique="Summary A better preserves the key arguments.",
        confidence=0.85,
        source="llm_judge",
    )

    # Convert to DSPy metric
    metric = response.to_dspy_metric()
    # {'score': 4.2, 'feedback': 'Summary A better preserves the key arguments.'}

    # Convert to a binary comparison
    pair = response.to_binary_comparison(request)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.supervision_metadata import judgment_supervision_metadata
from src.training.supervision import (
    BinaryComparison,
    BinaryProjectionDataset,
    ResponseJudgment,
    SupervisionDataset,
)

logger = logging.getLogger(__name__)

DEFAULT_PROPENSITY = 1.0
MIN_PROPENSITY = 1e-8

try:
    from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord
except Exception:  # pragma: no cover - treepo is an editable dependency in normal use.
    Candidate = None  # type: ignore[assignment]
    PreferenceDataset = None  # type: ignore[assignment]
    PreferenceRecord = None  # type: ignore[assignment]


# =============================================================================
# PreferenceDimension
# =============================================================================

@dataclass
class PreferenceDimension:
    """A single dimension of preference data being requested.

    The system is agnostic about types -- these are the built-in kinds,
    but 'custom' is always available for arbitrary structured preference data.

    Built-in kinds:
        pairwise: Compare text_a vs text_b, return preferred A/B/tie
        scalar: Rate on a numeric scale, return score
        critique: Provide written critique, return critique text
        custom: Arbitrary structured preference data via extra dict
    """
    kind: str  # "pairwise", "scalar", "critique", "custom"
    name: Optional[str] = None  # e.g., "helpfulness", "faithfulness"
    scale: Optional[Tuple[float, float]] = None  # (min, max) for scalar
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind}
        if self.name is not None:
            d["name"] = self.name
        if self.scale is not None:
            d["scale"] = list(self.scale)
        if self.options:
            d["options"] = self.options
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceDimension":
        d = dict(data)
        if "scale" in d and d["scale"] is not None:
            d["scale"] = tuple(d["scale"])
        return cls(**d)


# =============================================================================
# PreferenceRequest
# =============================================================================

@dataclass
class PreferenceRequest:
    """Context for requesting preference data on a tree node or comparison.

    Agnostic about what kind of preference data is requested. The ``dimensions``
    field declares what the requester wants; the collector provides what
    it can. If ``dimensions`` is empty, the request auto-infers from content:
    pairwise if ``text_b`` is set, scalar otherwise.

    IPW propensity fields are propagated from the audit sampling design
    (``AuditReport.inclusion_probability_map``) so that downstream
    estimators and training can apply inverse-probability weighting.
    """
    # Identity
    request_id: str

    # Content to evaluate
    text_a: str = ""
    text_b: Optional[str] = None
    original_text: str = ""
    rubric: str = ""
    reference_score: Optional[float] = None

    # What preference data is requested
    dimensions: List[PreferenceDimension] = field(default_factory=list)

    # Tree/audit context
    node_id: Optional[str] = None
    tree_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    law_type: str = "sufficiency"

    # IPW propensity (propagated from audit)
    sampling: SamplingMetadata = field(
        default_factory=lambda: SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
    )

    # Metadata
    priority: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if not self.dimensions:
            if self.text_b is not None:
                self.dimensions = [PreferenceDimension(kind="pairwise")]
            else:
                self.dimensions = [PreferenceDimension(kind="scalar")]
        if not isinstance(self.sampling, SamplingMetadata):
            self.sampling = SamplingMetadata.from_dict(self.sampling)
        if self.sampling.unit_kind is None:
            self.sampling = self.sampling.with_updates(unit_kind=ObservationUnitKind.PAIR)

    @property
    def is_pairwise(self) -> bool:
        return any(d.kind == "pairwise" for d in self.dimensions)

    @property
    def joint_propensity(self) -> float:
        return self.sampling.effective_joint_propensity(min_propensity=0.0)

    @classmethod
    def from_flagged_item(cls, item: Any) -> "PreferenceRequest":
        """Create a PreferenceRequest from an existing FlaggedItem.

        Bridges the audit review queue to the generalized preference system.
        """
        return cls(
            request_id=f"flag_{item.item_id}",
            text_a=item.input_a,
            text_b=item.input_b if item.input_b else None,
            rubric=item.rubric,
            node_id=item.node_id,
            tree_id=item.tree_id,
            law_type=item.check_type,
            priority=getattr(item, "priority", type("", (), {"value": 0})).value
            if hasattr(getattr(item, "priority", None), "value")
            else 0,
            dimensions=[
                PreferenceDimension(kind="pairwise")
                if item.input_b
                else PreferenceDimension(kind="scalar"),
                PreferenceDimension(kind="critique"),
            ],
            context={
                "approx_discrepancy": item.approx_discrepancy,
                "approx_reasoning": item.approx_reasoning,
                "node_level": getattr(item, "node_level", 0),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "reference_score": self.reference_score,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "node_id": self.node_id,
            "tree_id": self.tree_id,
            "source_doc_id": self.source_doc_id,
            "law_type": self.law_type,
            "sampling": self.sampling.to_dict(),
            "priority": self.priority,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceRequest":
        d = dict(data)
        if "dimensions" in d:
            d["dimensions"] = [PreferenceDimension.from_dict(dim) for dim in d["dimensions"]]
        if "sampling" not in d:
            d["sampling"] = {
                "document_propensity": d.pop("doc_propensity", DEFAULT_PROPENSITY),
                "unit_propensity": d.pop("node_propensity", DEFAULT_PROPENSITY),
                "label_propensity": d.pop("label_propensity", DEFAULT_PROPENSITY),
                "sampling_scheme": d.pop("sampling_scheme", None),
                "unit_kind": "pair",
            }
        return cls(**d)


# =============================================================================
# PreferenceResponse
# =============================================================================

@dataclass
class PreferenceResponse:
    """Multi-dimensional preference response.

    Can carry any combination of:
    - Pairwise preference (A/B/tie)
    - Scalar rating(s) keyed by dimension name
    - Written critique / natural language critique
    - Arbitrary structured data in ``extra``

    Always convertible to:
    - DSPy metric dict via ``to_dspy_metric()``
    - ResponseJudgment / ComparativeJudgment via supervision helpers
    - BinaryComparison via ``to_binary_comparison(request)``
    - PreferencePair via ``to_preference_pair(request)`` for compatibility
    - FlaggedItem update via ``to_flagged_item_update()``
    """
    # Identity (matches request)
    request_id: str

    # Pairwise preference
    preferred: Optional[Literal["A", "B", "tie"]] = None

    # Scalar ratings (dimension_name -> value)
    scores: Dict[str, float] = field(default_factory=dict)

    # Written critique
    critique: str = ""

    # Reasoning / confidence
    reasoning: str = ""
    confidence: float = 0.5
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None

    # Arbitrary structured data
    extra: Dict[str, Any] = field(default_factory=dict)

    # Source metadata
    source: str = "unknown"
    judge_model: str = ""
    timestamp: Optional[str] = None
    raw_result: Optional[Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    @classmethod
    def from_human_preference(
        cls,
        *,
        request_id: str,
        preferred: Optional[Literal["A", "B", "tie"]] = None,
        scores: Optional[Dict[str, float]] = None,
        critique: str = "",
        reasoning: str = "",
        confidence: float = 1.0,
        score_estimate_a: Optional[float] = None,
        score_estimate_b: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        judge_model: str = "",
    ) -> "PreferenceResponse":
        """Create a canonical human-sourced response for programmatic or API use."""
        return cls(
            request_id=request_id,
            preferred=preferred,
            scores=dict(scores or {}),
            critique=critique,
            reasoning=reasoning,
            confidence=confidence,
            score_estimate_a=score_estimate_a,
            score_estimate_b=score_estimate_b,
            extra=dict(extra or {}),
            source="human",
            judge_model=judge_model,
        )

    @classmethod
    def from_human_pairwise_preference(
        cls,
        *,
        request_id: str,
        preferred: Literal["A", "B", "tie"],
        reasoning: str = "",
        critique: str = "",
        confidence: float = 1.0,
        score_estimate_a: Optional[float] = None,
        score_estimate_b: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "PreferenceResponse":
        """Create a human pairwise response that can project to binary/comparative supervision."""
        return cls.from_human_preference(
            request_id=request_id,
            preferred=preferred,
            reasoning=reasoning,
            critique=critique,
            confidence=confidence,
            score_estimate_a=score_estimate_a,
            score_estimate_b=score_estimate_b,
            extra=extra,
        )

    @classmethod
    def from_human_scalar_preference(
        cls,
        *,
        request_id: str,
        score: float,
        dimension_name: str = "score",
        reasoning: str = "",
        critique: str = "",
        confidence: float = 1.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "PreferenceResponse":
        """Create a human scalar response that can project to response supervision."""
        return cls.from_human_preference(
            request_id=request_id,
            scores={str(dimension_name): float(score)},
            reasoning=reasoning,
            critique=critique,
            confidence=confidence,
            extra=extra,
        )

    def _combined_reasoning(self) -> str:
        reasoning = str(self.reasoning or "").strip()
        critique = str(self.critique or "").strip()
        parts: List[str] = []
        if reasoning:
            parts.append(reasoning)
        if critique and (not reasoning or critique not in reasoning):
            parts.append(critique)
        return "\n".join(parts).strip()

    # --- DSPy compatibility ---

    def to_dspy_metric(self) -> Dict[str, Any]:
        """Convert to DSPy metric format: {'score': float, 'feedback': str}.

        Score derivation priority:
        1. ``scores["score"]`` if present
        2. Mean of all entries in ``scores`` if non-empty
        3. Preference + confidence mapping if ``preferred`` is set
        4. 0.5 (neutral) as fallback
        """
        if "score" in self.scores:
            score = self.scores["score"]
        elif self.scores:
            score = sum(self.scores.values()) / len(self.scores)
        elif self.preferred is not None:
            if self.preferred == "A":
                score = 0.5 + self.confidence * 0.5
            elif self.preferred == "B":
                score = 0.5 - self.confidence * 0.5
            else:
                score = 0.5
        else:
            score = 0.5

        feedback = self._combined_reasoning()
        return {"score": score, "feedback": feedback}

    def to_response_judgment(
        self,
        request: "PreferenceRequest",
        *,
        response_id: Optional[str] = None,
        response_text: Optional[str] = None,
        score_value: Optional[float] = None,
    ) -> ResponseJudgment:
        """Convert scalar preference into a canonical response judgment."""
        scalar_dimension = next(
            (dimension for dimension in request.dimensions if dimension.kind == "scalar"),
            None,
        )
        signal_name = (
            (scalar_dimension.name if scalar_dimension and scalar_dimension.name else None)
            or self.extra.get("response_signal_name")
            or next(iter(self.scores.keys()), None)
            or "response_score"
        )
        signal_min: Optional[float] = None
        signal_max: Optional[float] = None
        if scalar_dimension is not None and scalar_dimension.scale is not None:
            signal_min, signal_max = scalar_dimension.scale
        else:
            signal_min = self.extra.get("response_signal_min")
            signal_max = self.extra.get("response_signal_max")

        if score_value is None:
            if scalar_dimension and scalar_dimension.name and scalar_dimension.name in self.scores:
                score_value = self.scores[scalar_dimension.name]
            elif self.scores:
                score_value = next(iter(self.scores.values()))
            elif self.score_estimate_a is not None:
                score_value = self.score_estimate_a
            else:
                metric = self.to_dspy_metric()
                score_value = metric["score"]

        resolved_response_id = response_id or ("A" if request.text_b is None else "A")
        resolved_response_text = response_text or request.text_a
        return ResponseJudgment(
            judgment_id=f"{self.request_id}:{resolved_response_id}",
            source_example_id=request.node_id or request.request_id,
            original_text=request.original_text,
            rubric=request.rubric,
            response=resolved_response_text,
            response_id=resolved_response_id,
            reference_score=request.reference_score or 0.0,
            law_type=request.law_type,
            source_doc_id=request.source_doc_id,
            sampling=request.sampling,
            supervision_metadata=judgment_supervision_metadata(
                application_name="preference_collection",
                law_type=request.law_type,
                response_signal_name=signal_name,
                response_signal_min=signal_min,
                response_signal_max=signal_max,
                metadata={
                    "request_id": request.request_id,
                    "preference_source": self.source,
                },
            ),
            response_signal_value=score_value,
            judge_model=self.judge_model,
            timestamp=self.timestamp,
            truth_label_source=self.source,
            metadata={
                "reasoning": self.reasoning,
                "critique": self.critique,
                "source": self.source,
                **dict(self.extra),
            },
        )

    def to_comparative_judgment(
        self,
        request: "PreferenceRequest",
        pair_id: Optional[str] = None,
    ) -> Any:
        """Convert pairwise preference into a canonical comparative judgment."""
        if request.text_b is None:
            raise ValueError("Comparative judgments require pairwise preference with text_b.")
        return self.to_binary_comparison(request, pair_id=pair_id).to_comparative_judgment()

    # --- Binary comparison compatibility ---

    def to_binary_comparison(
        self,
        request: "PreferenceRequest",
        pair_id: Optional[str] = None,
    ) -> BinaryComparison:
        """Convert pairwise human/LLM/oracle preference data into a canonical binary comparison."""
        combined_reasoning = self._combined_reasoning()
        supervision = judgment_supervision_metadata(
            application_name="preference_collection",
            supervision_channel_name="judgment_supervision",
            supervision_signal_name="judgment",
            preference_family="pairwise",
            law_type=request.law_type,
            comparison_signal_name=self.extra.get("comparison_signal_name"),
            comparison_signal_min=self.extra.get("comparison_signal_min"),
            comparison_signal_max=self.extra.get("comparison_signal_max"),
            response_signal_name=self.extra.get("response_signal_name"),
            response_signal_min=self.extra.get("response_signal_min"),
            response_signal_max=self.extra.get("response_signal_max"),
            metadata={
                "request_id": request.request_id,
                "preference_source": self.source,
            },
        )

        return BinaryComparison(
            pair_id=pair_id or self.request_id,
            source_example_id=request.node_id or request.request_id,
            original_text=request.original_text,
            rubric=request.rubric,
            reference_score=request.reference_score or 0.0,
            summary_a=request.text_a,
            summary_b=request.text_b or "",
            preferred=self.preferred or "tie",
            reasoning=combined_reasoning,
            confidence=self.confidence,
            score_estimate_a=self.score_estimate_a,
            score_estimate_b=self.score_estimate_b,
            comparison_signal_value=self.extra.get("comparison_signal_value"),
            judge_model=self.judge_model,
            sampling=request.sampling,
            preference_supervision=supervision,
            source_doc_id=request.source_doc_id,
            truth_label_source=self.source,
            source_observation_ids=[request.request_id],
            law_type=request.law_type,
        )

    def to_preference_pair(
        self,
        request: "PreferenceRequest",
        pair_id: Optional[str] = None,
    ) -> Any:
        """Backward-compatible alias for ``to_binary_comparison``."""
        return self.to_binary_comparison(request, pair_id=pair_id)

    def to_preference_record(
        self,
        request: "PreferenceRequest",
        *,
        record_id: Optional[str] = None,
    ) -> Any:
        """Convert this response into the unified treepo unit/candidate record."""
        return preference_record_from_response(request, self, record_id=record_id)

    # --- FlaggedItem compatibility ---

    def to_flagged_item_update(self) -> Dict[str, Any]:
        """Return fields suitable for updating a FlaggedItem after review."""
        # Derive approval from preference or score
        if self.preferred is not None:
            approved = self.preferred != "B"
        elif self.scores:
            mean_score = sum(self.scores.values()) / len(self.scores)
            approved = mean_score >= 0.5
        else:
            approved = True

        review_reasoning = self._combined_reasoning()

        return {
            "reviewed": True,
            "review_result": approved,
            "review_reasoning": review_reasoning,
            "corrected_summary": self.extra.get("corrected_summary"),
            "reviewed_at": self.timestamp,
            "review_source": self.source,
        }

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "request_id": self.request_id,
            "preferred": self.preferred,
            "scores": self.scores,
            "critique": self.critique,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "score_estimate_a": self.score_estimate_a,
            "score_estimate_b": self.score_estimate_b,
            "extra": self.extra,
            "source": self.source,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
        }
        # Skip raw_result in serialization (may not be JSON-safe)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceResponse":
        d = dict(data)
        d.pop("raw_result", None)
        return cls(**d)


def preference_record_from_response(
    request: PreferenceRequest,
    response: PreferenceResponse,
    *,
    record_id: Optional[str] = None,
) -> Any:
    """Project a completed request/response pair into treepo ``PreferenceRecord``."""
    if Candidate is None or PreferenceRecord is None:
        raise RuntimeError("treepo is required for preference record export")
    unit_id = str(request.node_id or request.request_id)
    target = _preference_target(request)
    metadata = {
        "request_id": request.request_id,
        "law_type": request.law_type,
        "source": response.source,
        "judge_model": response.judge_model,
        "confidence": float(response.confidence),
        "reasoning": response.reasoning,
        "critique": response.critique,
        **dict(response.extra or {}),
    }
    context = {
        "original_text": request.original_text,
        "rubric": request.rubric,
        "reference_score": request.reference_score,
        "request_context": dict(request.context or {}),
    }
    if request.text_b is not None:
        tied = response.preferred == "tie"
        candidates = (
            Candidate(
                id="A",
                value=request.text_a,
                score=response.score_estimate_a,
                rank=1 if tied else None,
                preferred=response.preferred == "A",
                metadata={"side": "A"},
            ),
            Candidate(
                id="B",
                value=request.text_b,
                score=response.score_estimate_b,
                rank=1 if tied else None,
                preferred=response.preferred == "B",
                metadata={"side": "B"},
            ),
        )
    else:
        score_value = _scalar_score(response)
        candidates = (
            Candidate(
                id="A",
                value=request.text_a,
                score=score_value,
                metadata={"side": "A", "scores": dict(response.scores or {})},
            ),
        )
    return PreferenceRecord(
        record_id=str(record_id or request.request_id),
        unit_id=unit_id,
        unit_type=_preference_unit_type(request),
        target=target,
        context=context,
        candidates=candidates,
        weight=1.0,
        propensity=max(MIN_PROPENSITY, request.joint_propensity),
        metadata=metadata,
        tree_id=request.tree_id,
        doc_id=request.source_doc_id,
        node_id=request.node_id,
    )


def preference_dataset_from_responses(
    items: Sequence[Tuple[PreferenceRequest, PreferenceResponse]],
) -> Any:
    """Build the unified treepo ``PreferenceDataset`` from completed responses."""
    if PreferenceDataset is None:
        raise RuntimeError("treepo is required for preference dataset export")
    return PreferenceDataset.from_records(
        [
            preference_record_from_response(request, response)
            for request, response in items
        ]
    )


def supervision_dataset_from_responses(
    items: Sequence[Tuple[PreferenceRequest, PreferenceResponse]],
    *,
    include_pairwise_response_scores: bool = True,
) -> SupervisionDataset:
    """Convert completed responses into the existing supervision surface."""
    dataset = SupervisionDataset()
    for request, response in items:
        if request.is_pairwise and response.preferred is not None:
            dataset.add_comparative_judgment(response.to_comparative_judgment(request))
            if include_pairwise_response_scores:
                if response.score_estimate_a is not None:
                    dataset.add_response_judgment(
                        response.to_response_judgment(
                            request,
                            response_id="A",
                            response_text=request.text_a,
                            score_value=response.score_estimate_a,
                        )
                    )
                if request.text_b is not None and response.score_estimate_b is not None:
                    dataset.add_response_judgment(
                        response.to_response_judgment(
                            request,
                            response_id="B",
                            response_text=request.text_b,
                            score_value=response.score_estimate_b,
                        )
                    )
        elif not request.is_pairwise:
            dataset.add_response_judgment(response.to_response_judgment(request))
    return dataset


def binary_projection_dataset_from_responses(
    items: Sequence[Tuple[PreferenceRequest, PreferenceResponse]],
    *,
    projection: str = "adjacent",
) -> BinaryProjectionDataset:
    """Convert completed responses into the canonical binary optimizer projection."""
    return supervision_dataset_from_responses(items).project_binary(projection=projection)


def preference_propensity_diagnostics(
    items: Sequence[Tuple[PreferenceRequest, PreferenceResponse]],
) -> Dict[str, Any]:
    """Compute IPW diagnostics from request propensities."""
    if not items:
        return {
            "n_items": 0,
            "effective_sample_size": 0.0,
            "effective_sample_ratio": 0.0,
        }
    weights = [1.0 / max(MIN_PROPENSITY, request.joint_propensity) for request, _ in items]
    n = len(weights)
    sum_w = sum(weights)
    sum_w_sq = sum(w * w for w in weights)
    neff = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0
    return {
        "n_items": n,
        "effective_sample_size": neff,
        "effective_sample_ratio": neff / n if n > 0 else 0.0,
        "mean_weight": sum_w / n,
        "min_weight": min(weights),
        "max_weight": max(weights),
    }


def _preference_target(request: PreferenceRequest) -> Literal["f", "g", "both"]:
    target = str((request.context or {}).get("target") or "").strip()
    if target in {"f", "g", "both"}:
        return target  # type: ignore[return-value]
    if request.node_id:
        return "g"
    return "f"


def _preference_unit_type(request: PreferenceRequest) -> str:
    if request.text_b is not None:
        return "pair"
    if request.node_id:
        return "node"
    return "response"


def _scalar_score(response: PreferenceResponse) -> Optional[float]:
    if "score" in response.scores:
        return float(response.scores["score"])
    if response.scores:
        return float(next(iter(response.scores.values())))
    if response.score_estimate_a is not None:
        return float(response.score_estimate_a)
    return None
