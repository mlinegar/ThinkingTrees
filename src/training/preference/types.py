"""
Shared preference data types and protocols.

This module contains data classes shared between preference collectors,
separated to avoid circular imports between preference.py and base_preference.py.

Also provides the PreferenceDeriver protocol for pluggable preference derivation.

Available derivers:
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
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Type, runtime_checkable

import dspy

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair_id": self.pair_id,
            "source_example_id": self.source_example_id,
            "original_text": self.original_text,
            "rubric": self.rubric,
            "reference_score": self.reference_score,
            "law_type": self.law_type,
            "summary_a": self.summary_a,
            "summary_b": self.summary_b,
            "preferred": self.preferred,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "score_estimate_a": self.score_estimate_a,
            "score_estimate_b": self.score_estimate_b,
            "oracle_error_a": self.oracle_error_a,
            "oracle_error_b": self.oracle_error_b,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp,
            "generation_config_a": self.generation_config_a,
            "generation_config_b": self.generation_config_b,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreferencePair':
        """Create from dictionary."""
        return cls(**data)

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
            ).with_inputs(
                "law_type", "rubric", "original_text", "summary_a", "summary_b", "reference_score"
            )
            examples.append(example)

        return examples

    def to_dpo_format(self, law_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert to DPO (Direct Preference Optimization) format.

        Returns:
            List of dicts with prompt, chosen, rejected
        """
        dpo_data = []
        for pair in self.pairs:
            if pair.preferred == "tie":
                continue
            if law_type is not None and pair.law_type != law_type:
                continue

            prompt = f"""Summarize the following text while preserving: {pair.rubric}

Text: {pair.original_text}

Summary:"""

            if pair.preferred == "A":
                chosen = pair.summary_a
                rejected = pair.summary_b
            else:
                chosen = pair.summary_b
                rejected = pair.summary_a

            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "pair_id": pair.pair_id,
                    "confidence": pair.confidence,
                    "reference_score": pair.reference_score,
                    "law_type": pair.law_type,
                },
            })

        return dpo_data

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
        }
