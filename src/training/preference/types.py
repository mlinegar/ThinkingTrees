"""
Shared preference data types and protocols for preference learning.

This module contains data classes shared between preference collectors,
separated to avoid circular imports between preference.py and base_preference.py.

Preference Learning Framework
-----------------------------
The framework is designed to be method-agnostic, supporting modern preference
learning methods beyond the original DPO (Direct Preference Optimization):

- **DPO** (Direct Preference Optimization): Pairwise preferences with sigmoid loss
- **GRPO** (Group Relative Policy Optimization): Group-wise rankings (DeepSeek style)
- **PPO** (Proximal Policy Optimization): Reward-based training
- **RLHF** (Reinforcement Learning from Human Feedback): Reward models

The key abstraction is PreferencePair, which captures pairwise judgments
that can be converted to various training formats via `to_preference_format()`.

Theoretical Foundation
----------------------
The preference learning guarantees are proven in the Lean formalization:
- PreferenceLearning.lean: Abstract preference learning framework
- DPO.lean: DPO as a concrete instance of the framework

When the local laws (L1, L2, L3) hold, preference learning on summarized
data is equivalent to preference learning on original data. This applies
to ANY preference learning method that satisfies oracle-measurability.

Available Derivers
------------------
- JudgeDeriver: Uses LLM judge (DSPy PairwiseJudge) for comparison
- GenRMDeriver: Uses NVIDIA GenRM model for comparison
- OracleDeriver: Uses oracle scores to derive preferences

Usage:
    from src.training.preference.types import (
        get_deriver,
        JudgeDeriver,
        GenRMDeriver,
        OracleDeriver,
    )

    # Get a deriver by name
    deriver = get_deriver("genrm", judge=my_genrm_judge)

    # Derive preference
    result = deriver.derive(
        summary_a="...",
        summary_b="...",
        context="Preserve political position...",
        original_text="...",
    )

    # Export to various preference learning formats
    dataset = PreferenceDataset(pairs)
    dpo_data = dataset.to_preference_format(method="dpo")
    grpo_data = dataset.to_preference_format(method="grpo")
"""

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Type, runtime_checkable

import dspy

from src.core.prompting import default_summarize_prompt
from src.core.provenance import normalize_truth_label_source
from src.stats.sampling import (
    largest_remainder_allocation as _largest_remainder_allocation,
    pps_inclusion_probabilities as _pps_inclusion_probabilities,
    systematic_pps_sample_indices as _systematic_pps_sample_indices,
)

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_PROPENSITY = 1.0
MIN_PROPENSITY = 1e-8
MAX_PROPENSITY = 1.0


PromptBuilder = Callable[[str, str], Any]


def render_prompt(
    text: str,
    rubric: str,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """Render a prompt string using a prompt builder or the default template."""
    builder = prompt_builder or default_summarize_prompt
    prompt = builder(text, rubric)
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(prompt)


def compute_propensity_diagnostics(
    pairs: List["PreferencePair"],
    *,
    include_ties: bool = True,
    min_propensity: float = MIN_PROPENSITY,
    max_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute propensity/IPW diagnostics for a set of preference pairs.

    Used for reporting effective sample size and weight concentration in
    `final_stats.json` for judge/generator training subsets.
    """
    if include_ties:
        used_pairs = list(pairs)
    else:
        used_pairs = [pair for pair in pairs if pair.preferred != "tie"]

    n_total = len(pairs)
    n_used = len(used_pairs)

    if n_used == 0:
        return {
            "n_pairs_total": n_total,
            "n_pairs_used": 0,
            "n_ties_excluded": n_total,
            "include_ties": include_ties,
            "effective_sample_size": 0.0,
            "effective_sample_ratio": 0.0,
            "mean_joint_propensity": 0.0,
            "min_joint_propensity": 0.0,
            "max_joint_propensity": 0.0,
            "mean_sample_weight": 0.0,
            "min_sample_weight": 0.0,
            "max_sample_weight": 0.0,
            "sum_sample_weight": 0.0,
            "max_weight_clip": max_weight,
        }

    propensities = [
        pair.effective_joint_propensity(min_propensity=min_propensity)
        for pair in used_pairs
    ]
    weights = [
        pair.ipw_weight(min_propensity=min_propensity, max_weight=max_weight)
        for pair in used_pairs
    ]

    sum_w = sum(weights)
    sum_w_sq = sum(weight * weight for weight in weights)
    neff = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0
    neff_ratio = (neff / n_used) if n_used > 0 else 0.0

    return {
        "n_pairs_total": n_total,
        "n_pairs_used": n_used,
        "n_ties_excluded": n_total - n_used,
        "include_ties": include_ties,
        "effective_sample_size": neff,
        "effective_sample_ratio": neff_ratio,
        "mean_joint_propensity": sum(propensities) / n_used,
        "min_joint_propensity": min(propensities),
        "max_joint_propensity": max(propensities),
        "mean_sample_weight": sum_w / n_used,
        "min_sample_weight": min(weights),
        "max_sample_weight": max(weights),
        "sum_sample_weight": sum_w,
        "max_weight_clip": max_weight,
    }


# =============================================================================
# PreferenceDeriver Protocol
# =============================================================================

@dataclass
class PreferenceDerivationResult:
    """Result from preference derivation."""
    preferred: Literal["A", "B", "tie"]
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None
    raw_result: Optional[Any] = None


@runtime_checkable
class PreferenceDeriver(Protocol):
    """
    Protocol for preference derivation strategies.

    Derivers compare two summaries and determine which better preserves
    task-relevant information. Different implementations use different
    comparison mechanisms (LLM judge, GenRM, oracle scores).
    """

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """
        Derive preference between two summaries.

        Args:
            summary_a: First candidate summary
            summary_b: Second candidate summary
            context: Description of what information to preserve (rubric)
            original_text: Original text being summarized
            reference_score: Ground truth score for original text (if available)
            law_type: OPS law type ("sufficiency", "idempotence", "merge")
            **kwargs: Additional arguments for specific derivers

        Returns:
            PreferenceDerivationResult with preference, confidence, and reasoning
        """
        ...


# =============================================================================
# Deriver Registry
# =============================================================================

_DERIVER_REGISTRY: Dict[str, Type["PreferenceDeriver"]] = {}


def register_deriver(name: str):
    """Decorator to register a deriver class."""
    def decorator(cls: Type[PreferenceDeriver]):
        _DERIVER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_deriver(name: str, **kwargs) -> PreferenceDeriver:
    """
    Get a preference deriver by name.

    Args:
        name: Deriver name ("judge", "genrm", "oracle")
        **kwargs: Arguments passed to deriver constructor

    Returns:
        Configured deriver instance

    Raises:
        ValueError: If deriver name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _DERIVER_REGISTRY:
        available = list(_DERIVER_REGISTRY.keys())
        raise ValueError(f"Unknown deriver: '{name}'. Available: {available}")

    return _DERIVER_REGISTRY[name_lower](**kwargs)


def list_derivers() -> List[str]:
    """Return list of registered deriver names."""
    return list(_DERIVER_REGISTRY.keys())


# =============================================================================
# Deriver Implementations
# =============================================================================

@register_deriver("judge")
class JudgeDeriver:
    """
    Preference deriver using LLM judge (DSPy PairwiseJudge).

    Uses chain-of-thought reasoning to determine which summary
    better preserves the target information.
    """

    def __init__(self, judge: Optional[Any] = None, use_cot: bool = True):
        """
        Initialize the judge deriver.

        Args:
            judge: Optional pre-initialized PairwiseJudge. If None, creates one.
            use_cot: Whether to use chain-of-thought reasoning
        """
        self.judge = judge
        self.use_cot = use_cot

    def _ensure_judge(self):
        """Lazily create judge if not provided."""
        if self.judge is None:
            from src.training.preference.collector import PairwiseJudge
            self.judge = PairwiseJudge(use_cot=self.use_cot)
        return self.judge

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """Derive preference using LLM judge."""
        judge = self._ensure_judge()

        result = judge.forward(
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            rubric=context,
            reference_score=reference_score or 0.0,
        )

        return PreferenceDerivationResult(
            preferred=result.get("preferred", "tie"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
            score_estimate_a=result.get("score_estimate_a"),
            score_estimate_b=result.get("score_estimate_b"),
            raw_result=result,
        )


@register_deriver("genrm")
class GenRMDeriver:
    """
    Preference deriver using NVIDIA GenRM model.

    Uses the special response_1/response_2 format for comparison
    with ranking scores (1-6) and helpfulness scores (1-5).
    """

    def __init__(self, judge: Any):
        """
        Initialize the GenRM deriver.

        Args:
            judge: GenRMJudge instance
        """
        self.judge = judge

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """Derive preference using GenRM judge."""
        from src.training.preference.genrm import is_genrm_error

        result = self.judge.compare(
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
        )

        if is_genrm_error(result):
            return PreferenceDerivationResult(
                preferred="tie",
                confidence=0.0,
                reasoning=f"Error: {result.error_message}",
                raw_result=result,
            )

        # Map ranking score (1-6) to confidence (0-1)
        ranking_confidence = {
            1: 0.95, 2: 0.75, 3: 0.55,
            4: 0.55, 5: 0.75, 6: 0.95,
        }
        confidence = ranking_confidence.get(result.ranking_score, 0.5)

        return PreferenceDerivationResult(
            preferred=result.preferred,
            confidence=confidence,
            reasoning=result.reasoning,
            score_estimate_a=result.helpfulness_a,
            score_estimate_b=result.helpfulness_b,
            raw_result=result,
        )


@register_deriver("oracle")
class OracleDeriver:
    """
    Preference deriver using oracle scoring function.

    Compares summaries by computing oracle scores for each and
    determining which has lower error relative to ground truth.
    """

    def __init__(
        self,
        oracle_predict: Callable[[str], float],
        tie_margin: float = 0.05,
        scale_range: Optional[float] = None,
    ):
        """
        Initialize the oracle deriver.

        Args:
            oracle_predict: Function that scores text
            tie_margin: Normalized error margin for ties (default 5%)
            scale_range: Range of the scale for normalization
        """
        self.oracle_predict = oracle_predict
        self.tie_margin = tie_margin
        self.scale_range = scale_range

    def derive(
        self,
        summary_a: str,
        summary_b: str,
        context: str,
        original_text: str,
        reference_score: Optional[float] = None,
        law_type: str = "sufficiency",
        **kwargs,
    ) -> PreferenceDerivationResult:
        """Derive preference using oracle scores."""
        # Get ground truth if not provided
        if reference_score is None:
            reference_score = self.oracle_predict(original_text)

        # Score both summaries
        score_a = self.oracle_predict(summary_a)
        score_b = self.oracle_predict(summary_b)

        # Compute errors
        error_a = abs(score_a - reference_score)
        error_b = abs(score_b - reference_score)

        # Normalize errors if scale_range provided
        if self.scale_range is not None and self.scale_range > 0:
            norm_error_a = error_a / self.scale_range
            norm_error_b = error_b / self.scale_range
        else:
            norm_error_a = error_a
            norm_error_b = error_b

        # Determine preference
        error_diff = norm_error_a - norm_error_b

        if abs(error_diff) <= self.tie_margin:
            preferred = "tie"
            confidence = 0.5
            reasoning = f"Tie: errors within margin. A={norm_error_a:.3f}, B={norm_error_b:.3f}"
        elif error_diff > 0:
            preferred = "B"
            confidence = min(0.95, 0.5 + abs(error_diff) * 2)
            reasoning = f"B has lower error ({norm_error_b:.3f} vs {norm_error_a:.3f})"
        else:
            preferred = "A"
            confidence = min(0.95, 0.5 + abs(error_diff) * 2)
            reasoning = f"A has lower error ({norm_error_a:.3f} vs {norm_error_b:.3f})"

        return PreferenceDerivationResult(
            preferred=preferred,
            confidence=confidence,
            reasoning=reasoning,
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            raw_result={
                "error_a": error_a,
                "error_b": error_b,
                "reference_score": reference_score,
            },
        )


@dataclass
class PreferencePair:
    """
    A single pairwise preference judgment.

    Represents the output of comparing two candidate summaries
    and determining which better preserves the target information.
    """
    # Identifiers
    pair_id: str
    source_example_id: str

    # Input context
    original_text: str
    rubric: str
    reference_score: float

    # Candidate summaries
    summary_a: str
    summary_b: str

    # Judgment
    preferred: Literal["A", "B", "tie"]
    reasoning: str
    confidence: float

    # Fields with defaults (must come after required fields)
    law_type: str = "sufficiency"
    source_doc_id: Optional[str] = None
    three_layer_roles: Dict[str, str] = field(default_factory=dict)
    truth_label_source: str = "unknown"
    oracle_view: Optional[str] = None
    oracle_proxy_source: Optional[str] = None

    # TreePO/IPW metadata (optional, used for weighted estimation/training)
    doc_propensity: float = DEFAULT_GLOBAL_PROPENSITY
    node_propensity: float = DEFAULT_GLOBAL_PROPENSITY
    label_propensity: float = DEFAULT_GLOBAL_PROPENSITY
    joint_propensity: Optional[float] = None
    sampling_scheme: Optional[str] = None
    node_type: Optional[str] = None

    # Optional audit alignment metadata (Phase 1.5 TreePO audit)
    audit_tree_id: Optional[str] = None
    audit_passed: Optional[bool] = None
    audit_violation_rate: Optional[float] = None
    audit_union_bound: Optional[float] = None
    audit_violation_ci_low: Optional[float] = None
    audit_violation_ci_high: Optional[float] = None

    # Score estimates from judge
    score_estimate_a: Optional[float] = None
    score_estimate_b: Optional[float] = None
    oracle_error_a: Optional[float] = None
    oracle_error_b: Optional[float] = None

    # Metadata
    judge_model: str = ""
    timestamp: Optional[str] = None
    generation_config_a: Optional[Dict[str, Any]] = None
    generation_config_b: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        self.truth_label_source = normalize_truth_label_source(self.truth_label_source)
        if self.source_doc_id is not None:
            self.source_doc_id = str(self.source_doc_id)
        if self.oracle_view is not None:
            self.oracle_view = str(self.oracle_view)
        if self.oracle_proxy_source is not None:
            self.oracle_proxy_source = str(self.oracle_proxy_source)
        if self.three_layer_roles is None:
            self.three_layer_roles = {}
        elif not isinstance(self.three_layer_roles, dict):
            self.three_layer_roles = dict(self.three_layer_roles)

        self.doc_propensity = self._normalize_propensity_component(
            self.doc_propensity, "doc_propensity"
        )
        self.node_propensity = self._normalize_propensity_component(
            self.node_propensity, "node_propensity"
        )
        self.label_propensity = self._normalize_propensity_component(
            self.label_propensity, "label_propensity"
        )

        if self.joint_propensity is None:
            self.joint_propensity = (
                self.doc_propensity * self.node_propensity * self.label_propensity
            )
        else:
            self.joint_propensity = self._normalize_propensity_component(
                self.joint_propensity, "joint_propensity"
            )

    @staticmethod
    def _normalize_propensity_component(value: Any, name: str) -> float:
        """Normalize propensity value; missing values default to uniform."""
        if value is None:
            return DEFAULT_GLOBAL_PROPENSITY
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric or None, got {value!r}")
        if not math.isfinite(parsed) or parsed <= 0 or parsed > MAX_PROPENSITY:
            raise ValueError(
                f"{name} must be finite and in (0, {MAX_PROPENSITY}], got {parsed!r}"
            )
        return parsed

    def effective_joint_propensity(self, min_propensity: float = MIN_PROPENSITY) -> float:
        """Joint propensity with global-uniform fallback and numerical floor."""
        joint = self.joint_propensity
        if joint is None:
            joint = self.doc_propensity * self.node_propensity * self.label_propensity
        return max(min_propensity, float(joint))

    def ipw_weight(
        self,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> float:
        """Inverse-propensity weight for this preference pair."""
        weight = 1.0 / self.effective_joint_propensity(min_propensity=min_propensity)
        if max_weight is not None:
            weight = min(weight, max_weight)
        return weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_id": self.pair_id,
            "source_example_id": self.source_example_id,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "source_doc_id": self.source_doc_id,
            "three_layer_roles": self.three_layer_roles,
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "preferred": self.preferred,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "doc_propensity": self.doc_propensity,
            "node_propensity": self.node_propensity,
            "label_propensity": self.label_propensity,
            "joint_propensity": self.joint_propensity,
            "sampling_scheme": self.sampling_scheme,
            "node_type": self.node_type,
            "audit_tree_id": self.audit_tree_id,
            "audit_passed": self.audit_passed,
            "audit_violation_rate": self.audit_violation_rate,
            "audit_union_bound": self.audit_union_bound,
            "audit_violation_ci_low": self.audit_violation_ci_low,
            "audit_violation_ci_high": self.audit_violation_ci_high,
            "score_estimate_a": self.score_estimate_a,
            "score_estimate_b": self.score_estimate_b,
            "oracle_error_a": self.oracle_error_a,
            "oracle_error_b": self.oracle_error_b,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
            "generation_config_a": self.generation_config_a,
            "generation_config_b": self.generation_config_b,
            "sample_weight": self.ipw_weight(),
        }

    def treepo_metadata(self) -> Dict[str, Any]:
        """Return compact TreePO metadata for downstream export formats."""
        metadata = {
            "doc_propensity": self.doc_propensity,
            "node_propensity": self.node_propensity,
            "label_propensity": self.label_propensity,
            "joint_propensity": self.effective_joint_propensity(),
            "sample_weight": self.ipw_weight(),
            "sampling_scheme": self.sampling_scheme,
            "node_type": self.node_type,
            "audit_tree_id": self.audit_tree_id,
            "audit_passed": self.audit_passed,
            "audit_violation_rate": self.audit_violation_rate,
            "audit_union_bound": self.audit_union_bound,
            "audit_violation_ci_low": self.audit_violation_ci_low,
            "audit_violation_ci_high": self.audit_violation_ci_high,
            "truth_label_source": self.truth_label_source,
            "oracle_view": self.oracle_view,
            "oracle_proxy_source": self.oracle_proxy_source,
            "source_doc_id": self.source_doc_id,
        }
        return {key: value for key, value in metadata.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreferencePair':
        """Create from dictionary."""
        payload = dict(data)
        # Derived/runtime field; recomputed from propensities.
        payload.pop("sample_weight", None)
        return cls(**payload)

    def get_winner(self) -> Optional[str]:
        """Return the winning summary, or None for ties."""
        if self.preferred == "A":
            return self.summary_a
        elif self.preferred == "B":
            return self.summary_b
        return None

    def get_loser(self) -> Optional[str]:
        """Return the losing summary, or None for ties."""
        if self.preferred == "A":
            return self.summary_b
        elif self.preferred == "B":
            return self.summary_a
        return None


@dataclass
class GenerationConfig:
    """Configuration for generating candidate summaries."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    prompt_variant: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "prompt_variant": self.prompt_variant,
        }


class PreferenceDataset:
    """
    Dataset of preference pairs for training.

    Supports saving/loading, filtering, and conversion to training formats.
    """

    def __init__(self, pairs: Optional[List[PreferencePair]] = None):
        """
        Initialize the dataset.

        Args:
            pairs: Initial list of preference pairs
        """
        self.pairs = pairs or []

    def add_pair(self, pair: PreferencePair):
        """Add a preference pair to the dataset."""
        self.pairs.append(pair)

    def add_pairs(self, pairs: List[PreferencePair]):
        """Add multiple preference pairs."""
        self.pairs.extend(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]

    def get_sample_weights(
        self,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> List[float]:
        """Return IPW sample weights for all pairs."""
        return [
            pair.ipw_weight(min_propensity=min_propensity, max_weight=max_weight)
            for pair in self.pairs
        ]

    def propensity_diagnostics(
        self,
        *,
        include_ties: bool = True,
        min_propensity: float = MIN_PROPENSITY,
        max_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Dataset-level wrapper around `compute_propensity_diagnostics`."""
        return compute_propensity_diagnostics(
            self.pairs,
            include_ties=include_ties,
            min_propensity=min_propensity,
            max_weight=max_weight,
        )

    def resample_by_propensity(
        self,
        target_size: Optional[int] = None,
        seed: int = 42,
        max_weight: Optional[float] = None,
        min_propensity: float = MIN_PROPENSITY,
        strategy: Literal["multinomial", "pps_systematic", "stratified_multinomial"] = "pps_systematic",
        stratify_by: Optional[str] = None,
    ) -> 'PreferenceDataset':
        """
        Backward-compatible wrapper for propensity-based sampling.

        Historically this performed multinomial resampling with replacement.
        New strategies are available for efficiency and variance control.
        """
        return self.sample_by_propensity(
            target_size=target_size,
            seed=seed,
            max_weight=max_weight,
            min_propensity=min_propensity,
            strategy=strategy,
            stratify_by=stratify_by,
        )

    def sample_by_propensity(
        self,
        target_size: Optional[int] = None,
        seed: int = 42,
        max_weight: Optional[float] = None,
        min_propensity: float = MIN_PROPENSITY,
        strategy: Literal["multinomial", "pps_systematic", "stratified_multinomial"] = "pps_systematic",
        stratify_by: Optional[str] = None,
    ) -> 'PreferenceDataset':
        """
        Sample pairs according to propensity-derived weights.

        Strategies:
        - `multinomial`: with-replacement weighted sampling.
        - `pps_systematic`: fixed-size PPS without replacement when possible.
        - `stratified_multinomial`: weighted multinomial sampling within strata.
        """
        if not self.pairs:
            return PreferenceDataset([])

        size = int(target_size or len(self.pairs))
        if size <= 0:
            return PreferenceDataset([])

        rng = random.Random(seed)
        weights = self.get_sample_weights(
            min_propensity=min_propensity,
            max_weight=max_weight,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            return PreferenceDataset(self.pairs.copy())

        if strategy == "multinomial":
            sampled_pairs = rng.choices(self.pairs, weights=weights, k=size)
            return PreferenceDataset(sampled_pairs)

        if strategy == "pps_systematic":
            n = len(self.pairs)
            if size >= n:
                full = self.pairs.copy()
                extra = rng.choices(self.pairs, weights=weights, k=size - n)
                return PreferenceDataset(full + extra)

            inclusion_probs = _pps_inclusion_probabilities(weights, size)
            sampled_indices = _systematic_pps_sample_indices(inclusion_probs, size, rng)
            sampled_pairs = [self.pairs[index] for index in sampled_indices]
            return PreferenceDataset(sampled_pairs)

        if strategy == "stratified_multinomial":
            strata_key = stratify_by or "law_type"
            grouped_indices: Dict[str, List[int]] = defaultdict(list)
            for index, pair in enumerate(self.pairs):
                value = getattr(pair, strata_key, None)
                grouped_indices[str(value)].append(index)

            keys = list(grouped_indices.keys())
            if not keys:
                sampled_pairs = rng.choices(self.pairs, weights=weights, k=size)
                return PreferenceDataset(sampled_pairs)

            group_weight_sums = [
                sum(weights[index] for index in grouped_indices[key])
                for key in keys
            ]
            total_group_weight = sum(group_weight_sums)
            if total_group_weight <= 0:
                sampled_pairs = rng.choices(self.pairs, k=size)
                return PreferenceDataset(sampled_pairs)

            quotas = [
                size * (group_weight / total_group_weight)
                for group_weight in group_weight_sums
            ]
            allocation = _largest_remainder_allocation(size, quotas)
            sampled_pairs: List[PreferencePair] = []
            for key, group_size in zip(keys, allocation):
                if group_size <= 0:
                    continue
                group_indices = grouped_indices[key]
                group_pairs = [self.pairs[index] for index in group_indices]
                group_weights = [weights[index] for index in group_indices]
                sampled_pairs.extend(rng.choices(group_pairs, weights=group_weights, k=group_size))
            return PreferenceDataset(sampled_pairs)

        raise ValueError(
            f"Unknown sampling strategy: {strategy!r}. "
            "Expected one of {'multinomial', 'pps_systematic', 'stratified_multinomial'}."
        )

    def filter_by_confidence(self, min_confidence: float) -> 'PreferenceDataset':
        """Return new dataset with pairs above confidence threshold."""
        filtered = [p for p in self.pairs if p.confidence >= min_confidence]
        return PreferenceDataset(filtered)

    def filter_non_ties(self) -> 'PreferenceDataset':
        """Return new dataset excluding ties."""
        filtered = [p for p in self.pairs if p.preferred != "tie"]
        return PreferenceDataset(filtered)

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
    ) -> Tuple['PreferenceDataset', 'PreferenceDataset']:
        """
        Split into train and validation sets.

        Args:
            train_ratio: Fraction for training set
            shuffle: Whether to shuffle before splitting

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        pairs = self.pairs.copy()
        if shuffle:
            random.shuffle(pairs)

        split_idx = int(len(pairs) * train_ratio)
        return (
            PreferenceDataset(pairs[:split_idx]),
            PreferenceDataset(pairs[split_idx:]),
        )

    def to_dspy_examples(self) -> List[dspy.Example]:
        """
        Convert to DSPy examples for training.

        Returns:
            List of DSPy examples with inputs and preferred output
        """
        examples = []
        for pair in self.pairs:
            if pair.preferred == "tie":
                continue

            example = dspy.Example(
                law_type=pair.law_type,
                rubric=pair.rubric,
                original_text=pair.original_text,
                summary_a=pair.summary_a,
                summary_b=pair.summary_b,
                reference_score=pair.reference_score,
                preferred=pair.preferred,
                reasoning=pair.reasoning,
                confidence=pair.confidence,
                sample_weight=pair.ipw_weight(),
                joint_propensity=pair.effective_joint_propensity(),
            ).with_inputs(
                "law_type", "rubric", "original_text", "summary_a", "summary_b", "reference_score"
            )
            examples.append(example)

        return examples

    def to_preference_format(
        self,
        method: Literal["dpo", "grpo", "rlhf", "general"] = "general",
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to preference learning format for various methods.

        This is the unified export method for preference learning data.
        The format depends on the downstream training method.

        Theoretical Foundation
        ----------------------
        From PreferenceLearning.lean: Any oracle-measurable preference learning
        method achieves equivalent results on oracle-preserving summaries.
        DPO, GRPO, PPO, etc. all satisfy this property when properly configured.

        Args:
            method: Target training method format
                - "dpo": Returns prompt/chosen/rejected for DPO-style training
                - "grpo": Returns group ranking format (placeholder for future)
                - "rlhf": Returns prompt/response/score for reward model training
                - "general": Returns full context with all fields
            law_type: Filter by OPS law type (sufficiency, merge, idempotence)
            prompt_builder: Optional prompt builder for generating prompts

        Returns:
            List of preference examples in the requested format
        """
        if method == "dpo":
            return self._to_dpo_format(law_type, prompt_builder)
        elif method == "grpo":
            return self._to_grpo_format(law_type, prompt_builder)
        elif method == "rlhf":
            return self._to_rlhf_format(law_type, prompt_builder)
        else:
            return self._to_general_format(law_type)

    def _to_dpo_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to DPO (Direct Preference Optimization) format.

        DPO format uses pairwise preferences with prompt/chosen/rejected structure.
        This is the concrete instantiation of the abstract preference learning
        framework from PreferenceLearning.lean.

        Returns:
            List of dicts with prompt, chosen, rejected
        """
        dpo_data = []
        for pair in self.pairs:
            if pair.preferred == "tie":
                continue
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            if pair.preferred == "A":
                chosen = pair.summary_a
                rejected = pair.summary_b
            else:
                chosen = pair.summary_b
                rejected = pair.summary_a

            metadata = {
                "pair_id": pair.pair_id,
                "confidence": pair.confidence,
                "reference_score": pair.reference_score,
                "law_type": pair.law_type,
                "truth_label_source": pair.truth_label_source,
                "oracle_view": pair.oracle_view,
                "oracle_proxy_source": pair.oracle_proxy_source,
                "source_doc_id": pair.source_doc_id,
                "three_layer_roles": pair.three_layer_roles,
            }
            treepo_meta = pair.treepo_metadata()
            if treepo_meta:
                metadata["treepo"] = treepo_meta

            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "sample_weight": pair.ipw_weight(),
                "metadata": metadata,
            })

        return dpo_data

    def _to_grpo_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to GRPO (Group Relative Policy Optimization) format.

        GRPO uses group-wise comparisons rather than strict pairwise preferences.
        This format is compatible with DeepSeek-style preference learning.

        Note: GRPO typically works with groups of K responses. For pairwise data,
        we structure it as a 2-element group with relative ranking.

        Returns:
            List of dicts with prompt, responses (ranked list), and ranking info
        """
        grpo_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            # For GRPO, we provide ranked responses rather than chosen/rejected
            if pair.preferred == "A":
                ranked_responses = [pair.summary_a, pair.summary_b]
                ranks = [1, 2]
            elif pair.preferred == "B":
                ranked_responses = [pair.summary_b, pair.summary_a]
                ranks = [1, 2]
            else:  # tie
                ranked_responses = [pair.summary_a, pair.summary_b]
                ranks = [1, 1]  # Equal rank for ties

            grpo_data.append({
                "prompt": prompt,
                "responses": ranked_responses,
                "ranks": ranks,
                "confidence": pair.confidence,
                "sample_weight": pair.ipw_weight(),
                "metadata": {
                    "pair_id": pair.pair_id,
                    "reference_score": pair.reference_score,
                    "law_type": pair.law_type,
                    "original_preferred": pair.preferred,
                    "truth_label_source": pair.truth_label_source,
                    "oracle_view": pair.oracle_view,
                    "oracle_proxy_source": pair.oracle_proxy_source,
                    "source_doc_id": pair.source_doc_id,
                    "three_layer_roles": pair.three_layer_roles,
                    "treepo": pair.treepo_metadata(),
                },
            })

        return grpo_data

    def _to_rlhf_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to RLHF (Reinforcement Learning from Human Feedback) format.

        RLHF format provides responses with scalar scores for reward model training.
        Confidence is converted to a relative score differential.

        Returns:
            List of dicts with prompt, response, score
        """
        rlhf_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            # Generate score based on preference and confidence
            if pair.preferred == "A":
                score_a = 0.5 + pair.confidence * 0.5
                score_b = 0.5 - pair.confidence * 0.5
            elif pair.preferred == "B":
                score_a = 0.5 - pair.confidence * 0.5
                score_b = 0.5 + pair.confidence * 0.5
            else:  # tie
                score_a = 0.5
                score_b = 0.5

            rlhf_data.extend([
                {
                    "prompt": prompt,
                    "response": pair.summary_a,
                    "score": score_a,
                    "sample_weight": pair.ipw_weight(),
                    "metadata": {
                        "pair_id": pair.pair_id,
                        "response_id": "A",
                        "reference_score": pair.reference_score,
                        "law_type": pair.law_type,
                        "truth_label_source": pair.truth_label_source,
                        "oracle_view": pair.oracle_view,
                        "oracle_proxy_source": pair.oracle_proxy_source,
                        "source_doc_id": pair.source_doc_id,
                        "three_layer_roles": pair.three_layer_roles,
                        "treepo": pair.treepo_metadata(),
                    },
                },
                {
                    "prompt": prompt,
                    "response": pair.summary_b,
                    "score": score_b,
                    "sample_weight": pair.ipw_weight(),
                    "metadata": {
                        "pair_id": pair.pair_id,
                        "response_id": "B",
                        "reference_score": pair.reference_score,
                        "law_type": pair.law_type,
                        "truth_label_source": pair.truth_label_source,
                        "oracle_view": pair.oracle_view,
                        "oracle_proxy_source": pair.oracle_proxy_source,
                        "source_doc_id": pair.source_doc_id,
                        "three_layer_roles": pair.three_layer_roles,
                        "treepo": pair.treepo_metadata(),
                    },
                },
            ])

        return rlhf_data

    def _to_general_format(self, law_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert to general preference format with all fields.

        This format preserves all information and can be adapted to any
        preference learning method downstream.

        Returns:
            List of dicts with full preference pair information
        """
        general_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            general_data.append({
                "pair_id": pair.pair_id,
                "rubric": pair.rubric,
                "original_text": pair.original_text,
                "summary_a": pair.summary_a,
                "summary_b": pair.summary_b,
                "preferred": pair.preferred,
                "confidence": pair.confidence,
                "reasoning": pair.reasoning,
                "reference_score": pair.reference_score,
                "law_type": pair.law_type,
                "truth_label_source": pair.truth_label_source,
                "oracle_view": pair.oracle_view,
                "oracle_proxy_source": pair.oracle_proxy_source,
                "source_doc_id": pair.source_doc_id,
                "three_layer_roles": pair.three_layer_roles,
                "sample_weight": pair.ipw_weight(),
                "treepo": pair.treepo_metadata(),
            })

        return general_data

    def to_reward_model_format(
        self,
        law_type: Optional[str] = None,
        include_oracle_scores: bool = True,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to reward model training format.

        This format is optimized for training reward models that approximate
        the oracle/judge. Each response gets a scalar score derived from
        the pairwise comparisons.

        The score computation uses:
        - Oracle estimate scores if available (from GenRM or oracle scorer)
        - Fallback to preference + confidence-based scoring

        Args:
            law_type: Optional filter for specific law type
            include_oracle_scores: Include raw oracle score estimates if available
            prompt_builder: Optional prompt builder for generating prompts

        Returns:
            List of dicts with prompt, response, score, and optional oracle_estimate
        """
        rm_data = []
        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, prompt_builder)

            # Use oracle estimate scores if available, else derive from preference
            if include_oracle_scores and pair.score_estimate_a is not None:
                score_a = pair.score_estimate_a
                score_b = pair.score_estimate_b if pair.score_estimate_b is not None else 0.5
            else:
                # Derive scores from preference and confidence
                if pair.preferred == "A":
                    score_a = 0.5 + pair.confidence * 0.5
                    score_b = 0.5 - pair.confidence * 0.5
                elif pair.preferred == "B":
                    score_a = 0.5 - pair.confidence * 0.5
                    score_b = 0.5 + pair.confidence * 0.5
                else:  # tie
                    score_a = 0.5
                    score_b = 0.5

            base_metadata = {
                "pair_id": pair.pair_id,
                "reference_score": pair.reference_score,
                "law_type": pair.law_type,
                "confidence": pair.confidence,
                "preferred": pair.preferred,
                "truth_label_source": pair.truth_label_source,
                "oracle_view": pair.oracle_view,
                "oracle_proxy_source": pair.oracle_proxy_source,
                "source_doc_id": pair.source_doc_id,
                "three_layer_roles": pair.three_layer_roles,
                "treepo": pair.treepo_metadata(),
            }

            rm_data.append({
                "prompt": prompt,
                "response": pair.summary_a,
                "score": score_a,
                "sample_weight": pair.ipw_weight(),
                "oracle_estimate": pair.score_estimate_a,
                "oracle_error": pair.oracle_error_a,
                "metadata": {**base_metadata, "response_id": "A"},
            })
            rm_data.append({
                "prompt": prompt,
                "response": pair.summary_b,
                "score": score_b,
                "sample_weight": pair.ipw_weight(),
                "oracle_estimate": pair.score_estimate_b,
                "oracle_error": pair.oracle_error_b,
                "metadata": {**base_metadata, "response_id": "B"},
            })

        return rm_data

    def to_grouped_grpo_format(
        self,
        law_type: Optional[str] = None,
        min_group_size: int = 2,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to grouped GRPO format for k-wise rankings.

        Groups multiple responses for the same input (original_text + rubric)
        and provides rankings across all responses. This supports the
        Plackett-Luce style k-wise GRPO objective.

        Args:
            law_type: Optional filter for specific law type
            min_group_size: Minimum group size to include (default: 2)
            prompt_builder: Optional prompt builder for generating prompts

        Returns:
            List of dicts with prompt, responses (k items), and rankings
        """
        from collections import defaultdict

        # Group pairs by (original_text, rubric) to collect all responses
        groups: Dict[tuple, Dict] = defaultdict(lambda: {
            "responses": {},  # response_text -> best_score
            "pairs": [],
        })

        for pair in self.pairs:
            if law_type is not None and pair.law_type != law_type:
                continue

            key = (pair.original_text[:500], pair.rubric[:200])  # Truncate for grouping

            # Track all unique responses with their best estimated score
            for resp_text, score_est in [
                (pair.summary_a, pair.score_estimate_a),
                (pair.summary_b, pair.score_estimate_b),
            ]:
                existing = groups[key]["responses"].get(resp_text)
                if score_est is not None:
                    if existing is None or score_est > existing:
                        groups[key]["responses"][resp_text] = score_est
                elif existing is None:
                    # No score estimate, use preference-based heuristic
                    if pair.preferred == "A" and resp_text == pair.summary_a:
                        groups[key]["responses"][resp_text] = 0.5 + pair.confidence * 0.5
                    elif pair.preferred == "B" and resp_text == pair.summary_b:
                        groups[key]["responses"][resp_text] = 0.5 + pair.confidence * 0.5
                    else:
                        groups[key]["responses"][resp_text] = 0.5 - pair.confidence * 0.5

            groups[key]["pairs"].append(pair)
            groups[key]["rubric"] = pair.rubric
            groups[key]["original_text"] = pair.original_text
            groups[key]["reference_score"] = pair.reference_score

        # Convert groups to GRPO format
        grpo_data = []
        for key, group_data in groups.items():
            responses_with_scores = list(group_data["responses"].items())

            if len(responses_with_scores) < min_group_size:
                continue

            # Sort by score (highest first) to determine ranks
            sorted_responses = sorted(responses_with_scores, key=lambda x: x[1] or 0, reverse=True)
            responses = [r[0] for r in sorted_responses]
            scores = [r[1] or 0.5 for r in sorted_responses]

            # Compute ranks (1 = best, handle ties)
            ranks = []
            current_rank = 1
            for i, score in enumerate(scores):
                if i > 0 and score < scores[i - 1]:
                    current_rank = i + 1
                ranks.append(current_rank)

            prompt = render_prompt(group_data["original_text"], group_data["rubric"], prompt_builder)

            grpo_data.append({
                "prompt": prompt,
                "responses": responses,
                "ranks": ranks,
                "scores": scores,
                "k": len(responses),
                "metadata": {
                    "reference_score": group_data["reference_score"],
                    "law_type": group_data["pairs"][0].law_type if group_data["pairs"] else None,
                    "num_source_pairs": len(group_data["pairs"]),
                },
            })

        return grpo_data

    def to_dpo_format(
        self,
        law_type: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert to DPO (Direct Preference Optimization) format.

        .. deprecated::
            Use `to_preference_format(method='dpo')` instead for consistency
            with the generalized preference learning framework.

        Returns:
            List of dicts with prompt, chosen, rejected
        """
        import warnings
        warnings.warn(
            "to_dpo_format() is deprecated. Use to_preference_format(method='dpo') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._to_dpo_format(law_type, prompt_builder)

    def save(self, path: Path):
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "num_pairs": len(self.pairs),
            "pairs": [p.to_dict() for p in self.pairs],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.pairs)} preference pairs to {path}")

    @classmethod
    def load(cls, path: Path) -> 'PreferenceDataset':
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)

        pairs = [PreferencePair.from_dict(p) for p in data["pairs"]]
        logger.info(f"Loaded {len(pairs)} preference pairs from {path}")

        return cls(pairs)

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics about the dataset."""
        non_ties = [p for p in self.pairs if p.preferred != "tie"]
        with_propensity = self.pairs
        with_audit_context = [
            p for p in self.pairs
            if getattr(p, "audit_tree_id", None) is not None
        ]
        propensity_stats = self.propensity_diagnostics(include_ties=True)

        return {
            "total_pairs": len(self.pairs),
            "non_tie_pairs": len(non_ties),
            "tie_pairs": len(self.pairs) - len(non_ties),
            "prefer_a": sum(1 for p in self.pairs if p.preferred == "A"),
            "prefer_b": sum(1 for p in self.pairs if p.preferred == "B"),
            "avg_confidence": (
                sum(p.confidence for p in self.pairs) / len(self.pairs)
                if self.pairs else 0
            ),
            "high_confidence_pairs": sum(1 for p in self.pairs if p.confidence >= 0.8),
            "pairs_with_propensity": len(with_propensity),
            "pairs_with_audit_context": len(with_audit_context),
            "mean_joint_propensity": propensity_stats["mean_joint_propensity"],
            "mean_sample_weight": propensity_stats["mean_sample_weight"],
            "max_sample_weight": propensity_stats["max_sample_weight"],
            "effective_sample_size": propensity_stats["effective_sample_size"],
            "effective_sample_ratio": propensity_stats["effective_sample_ratio"],
        }
