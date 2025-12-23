"""
Preference Learning Infrastructure for Oracle-Preserving Summarization.

This module provides infrastructure for training oracle/judge models using
pairwise preference data. The workflow is:
1. Generate multiple candidate summaries from a smaller model (e.g., Nemotron-nano)
2. Use a large oracle model (e.g., Nemotron-253B) to compare and rank outputs
3. Build preference pairs for training
4. Train a smaller judge model to mimic the large oracle's preferences
5. Use the judge to guide distillation training of the summarizer

Key components:
- PreferencePair: A single pairwise preference judgment
- PairwiseJudge: DSPy module for comparing summaries
- PreferenceCollector: Generates diverse outputs and collects preferences
- PreferenceDataset: Manages preference pairs for training
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Tuple

import dspy

from src.manifesto.signatures import PairwiseSummaryComparison

logger = logging.getLogger(__name__)


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
    ground_truth_score: float

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
            "ground_truth_score": self.ground_truth_score,
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


class PairwiseJudge(dspy.Module):
    """
    DSPy module for comparing two summaries using a large oracle model.

    Uses chain-of-thought reasoning to determine which summary
    better preserves the target information.
    """

    def __init__(self, use_cot: bool = True):
        """
        Initialize the judge module.

        Args:
            use_cot: Whether to use chain-of-thought reasoning
        """
        super().__init__()
        if use_cot:
            self.compare = dspy.ChainOfThought(PairwiseSummaryComparison)
        else:
            self.compare = dspy.Predict(PairwiseSummaryComparison)

    def forward(
        self,
        original_text: str,
        summary_a: str,
        summary_b: str,
        rubric: str,
        ground_truth_score: float,
    ) -> Dict[str, Any]:
        """
        Compare two summaries and determine which is better.

        Args:
            original_text: Original source text
            summary_a: First candidate summary
            summary_b: Second candidate summary
            rubric: Information preservation criteria
            ground_truth_score: Ground truth score for original text

        Returns:
            Dictionary with preference judgment and reasoning
        """
        result = self.compare(
            rubric=rubric,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            ground_truth_score=ground_truth_score,
        )

        # Normalize preferred to uppercase
        preferred = str(result.preferred).upper().strip()
        if preferred not in ["A", "B", "TIE"]:
            # Try to extract from reasoning
            if "A" in preferred and "B" not in preferred:
                preferred = "A"
            elif "B" in preferred and "A" not in preferred:
                preferred = "B"
            else:
                preferred = "tie"
        elif preferred == "TIE":
            preferred = "tie"

        # Parse confidence
        try:
            confidence = float(result.confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        # Parse score estimates
        score_a = None
        raw_score_a = getattr(result, "score_estimate_a", None)
        if raw_score_a is not None:
            try:
                score_a = float(raw_score_a)
            except (ValueError, TypeError):
                score_a = None

        score_b = None
        raw_score_b = getattr(result, "score_estimate_b", None)
        if raw_score_b is not None:
            try:
                score_b = float(raw_score_b)
            except (ValueError, TypeError):
                score_b = None

        return {
            "preferred": preferred,
            "reasoning": str(result.reasoning),
            "confidence": confidence,
            "score_estimate_a": score_a,
            "score_estimate_b": score_b,
        }


@dataclass
class GenerationConfig:
    """Configuration for generating candidate summaries."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2000
    prompt_variant: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "prompt_variant": self.prompt_variant,
        }


class PreferenceCollector:
    """
    Collects preference pairs by generating diverse outputs and comparing them.

    Workflow:
    1. For each input example, generate k candidate summaries
    2. Create all pairwise comparisons (k choose 2)
    3. Use the judge model to compare each pair
    4. Store the preference pairs
    """

    def __init__(
        self,
        summarizer: dspy.Module,
        judge: PairwiseJudge,
        k_candidates: int = 4,
        generation_configs: Optional[List[GenerationConfig]] = None,
    ):
        """
        Initialize the collector.

        Args:
            summarizer: DSPy module for generating summaries
            judge: PairwiseJudge for comparing summaries
            k_candidates: Number of candidate summaries to generate per input
            generation_configs: Configurations for generating diverse outputs
        """
        self.summarizer = summarizer
        self.judge = judge
        self.k_candidates = k_candidates

        # Default configs with varying temperatures
        if generation_configs is None:
            self.generation_configs = [
                GenerationConfig(temperature=0.3, prompt_variant="concise"),
                GenerationConfig(temperature=0.5, prompt_variant="default"),
                GenerationConfig(temperature=0.7, prompt_variant="detailed"),
                GenerationConfig(temperature=0.9, prompt_variant="creative"),
            ]
        else:
            self.generation_configs = generation_configs

        self.pairs: List[PreferencePair] = []
        self._pair_counter = 0

    def generate_candidates(
        self,
        content: str,
        rubric: str,
    ) -> List[Tuple[str, GenerationConfig]]:
        """
        Generate k candidate summaries for the input.

        Args:
            content: Input content to summarize
            rubric: Information preservation rubric

        Returns:
            List of (summary, config) tuples
        """
        candidates = []

        lm = getattr(dspy, "settings", None)
        lm = getattr(lm, "lm", None)

        for config in self.generation_configs[:self.k_candidates]:
            prev_temp = None
            prev_top_p = None
            prev_max_tokens = None
            if lm is not None and hasattr(lm, "kwargs"):
                prev_temp = lm.kwargs.get("temperature")
                prev_top_p = lm.kwargs.get("top_p")
                prev_max_tokens = lm.kwargs.get("max_tokens")
                lm.kwargs["temperature"] = config.temperature
                lm.kwargs["top_p"] = config.top_p
                lm.kwargs["max_tokens"] = config.max_tokens

            try:
                result = self.summarizer(content=content, rubric=rubric)
                summary = getattr(result, 'summary', str(result))
                candidates.append((summary, config))
            except Exception as e:
                logger.warning(f"Failed to generate candidate: {e}")
            finally:
                if lm is not None and hasattr(lm, "kwargs"):
                    if prev_temp is None:
                        lm.kwargs.pop("temperature", None)
                    else:
                        lm.kwargs["temperature"] = prev_temp
                    if prev_top_p is None:
                        lm.kwargs.pop("top_p", None)
                    else:
                        lm.kwargs["top_p"] = prev_top_p
                    if prev_max_tokens is None:
                        lm.kwargs.pop("max_tokens", None)
                    else:
                        lm.kwargs["max_tokens"] = prev_max_tokens

        return candidates

    def collect_pairs_for_example(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: float,
        law_type: str = "sufficiency",
        judge_model: str = "",
    ) -> List[PreferencePair]:
        """
        Generate candidates and collect all pairwise preferences for one example.

        Args:
            example_id: Unique identifier for this example
            original_text: Original source text
            rubric: Information preservation rubric
            ground_truth_score: Ground truth score for original
            judge_model: Name of the judge model being used

        Returns:
            List of preference pairs
        """
        # Generate candidates
        candidates = self.generate_candidates(original_text, rubric)

        if len(candidates) < 2:
            logger.warning(f"Only generated {len(candidates)} candidates for {example_id}")
            return []

        pairs = []

        # Create all pairwise comparisons
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                summary_a, config_a = candidates[i]
                summary_b, config_b = candidates[j]

                # Randomly swap to avoid position bias
                if random.random() < 0.5:
                    summary_a, summary_b = summary_b, summary_a
                    config_a, config_b = config_b, config_a

                try:
                    # Get judge's preference
                    result = self.judge(
                        original_text=original_text,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        rubric=rubric,
                        ground_truth_score=ground_truth_score,
                    )

                    self._pair_counter += 1
                    pair = PreferencePair(
                        pair_id=f"pair_{self._pair_counter:06d}",
                        source_example_id=example_id,
                        original_text=original_text,
                        rubric=rubric,
                        ground_truth_score=ground_truth_score,
                        law_type=law_type,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        preferred=result["preferred"],
                        reasoning=result["reasoning"],
                        confidence=result["confidence"],
                        score_estimate_a=result.get("score_estimate_a"),
                        score_estimate_b=result.get("score_estimate_b"),
                        judge_model=judge_model,
                        generation_config_a=config_a.to_dict(),
                        generation_config_b=config_b.to_dict(),
                    )
                    pairs.append(pair)
                    self.pairs.append(pair)

                except Exception as e:
                    logger.error(f"Failed to judge pair: {e}")
                    continue

        return pairs

    def get_all_pairs(self) -> List[PreferencePair]:
        """Return all collected preference pairs."""
        return self.pairs

    def get_non_tie_pairs(self) -> List[PreferencePair]:
        """Return only pairs with clear preferences (no ties)."""
        return [p for p in self.pairs if p.preferred != "tie"]

    def get_high_confidence_pairs(self, threshold: float = 0.7) -> List[PreferencePair]:
        """Return pairs with confidence above threshold."""
        return [p for p in self.pairs if p.confidence >= threshold]


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
                ground_truth_score=pair.ground_truth_score,
                preferred=pair.preferred,
                reasoning=pair.reasoning,
                confidence=pair.confidence,
            ).with_inputs(
                "law_type", "rubric", "original_text", "summary_a", "summary_b", "ground_truth_score"
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
                    "ground_truth_score": pair.ground_truth_score,
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
