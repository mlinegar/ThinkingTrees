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

import logging
from typing import List, Optional, Dict, Any, Callable, Literal

import dspy

# Import generic signature from core; task-specific versions live under src/tasks
from src.config.constants import DIVERSE_TEMPERATURES
from src.core.signatures import PairwiseComparison
from src.core.output_parser import NormalizedOutputAccessor

# Import shared data types (separated to avoid circular imports)
from .preference_types import PreferencePair, GenerationConfig, PreferenceDataset

# Import base class for OPS law support (idempotence, merge)
from .base_preference import (
    BasePreferenceCollector,
    CandidateInfo,
    PreferenceResult,
    CollectionStatistics,
)

logger = logging.getLogger(__name__)


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
            self.compare = dspy.ChainOfThought(PairwiseComparison)
        else:
            self.compare = dspy.Predict(PairwiseComparison)

    def forward(
        self,
        original_text: str,
        summary_a: str,
        summary_b: str,
        rubric: str,
        reference_score: float,
    ) -> Dict[str, Any]:
        """
        Compare two summaries and determine which is better.

        Args:
            original_text: Original source text
            summary_a: First candidate summary
            summary_b: Second candidate summary
            rubric: Information preservation criteria
            reference_score: Ground truth score for original text

        Returns:
            Dictionary with preference judgment and reasoning
        """
        result = self.compare(
            rubric=rubric,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            reference_score=reference_score,
        )

        # Use normalized accessor to handle key casing variations
        accessor = NormalizedOutputAccessor(result)

        # Normalize preferred to uppercase
        preferred = str(accessor.get('preferred', 'tie')).upper().strip()
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

        # Parse confidence (using normalized accessor)
        try:
            confidence = float(accessor.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        # Parse score estimates (using normalized accessor for case-insensitive field access)
        score_a = None
        raw_score_a = accessor.get('score_estimate_a')
        if raw_score_a is not None:
            try:
                score_a = float(raw_score_a)
            except (ValueError, TypeError):
                score_a = None

        score_b = None
        raw_score_b = accessor.get('score_estimate_b')
        if raw_score_b is not None:
            try:
                score_b = float(raw_score_b)
            except (ValueError, TypeError):
                score_b = None

        return {
            "preferred": preferred,
            "reasoning": str(accessor.get('reasoning', '')),
            "confidence": confidence,
            "score_estimate_a": score_a,
            "score_estimate_b": score_b,
        }


class PreferenceCollector(BasePreferenceCollector[GenerationConfig]):
    """
    Unified preference collector with strategy-based preference derivation.

    Inherits from BasePreferenceCollector to support all three OPS laws:
    - Sufficiency: original text → summary preserves information
    - Idempotence: summarize(summary) ≈ summary
    - Merge: summarize(A + B) preserves information from both

    Supports three strategies for deriving preferences:
    - "judge": Uses PairwiseJudge for LLM-based comparison (default)
    - "genrm": Uses GenRMJudge for NVIDIA's Nemotron GenRM comparison
    - "oracle": Uses oracle predictions to compute error-based preferences

    Workflow:
    1. For each input example, generate k candidate summaries
    2. Create all pairwise comparisons (k choose 2)
    3. Use the configured strategy to compare each pair
    4. Store the preference pairs

    Example:
        # Judge-based (default, backward compatible)
        collector = PreferenceCollector(summarizer, judge=my_judge)

        # GenRM-based
        from src.ops_engine.training_framework.genrm_preference import GenRMJudge
        genrm = GenRMJudge(base_url="http://localhost:8000")
        collector = PreferenceCollector(summarizer, strategy="genrm", genrm_judge=genrm)

        # Oracle-based
        collector = PreferenceCollector(
            summarizer,
            strategy="oracle",
            oracle_predict=lambda text: my_oracle(text),
        )

        # Collect pairs for different OPS laws
        sufficiency_pairs = collector.collect_pairs_for_example(
            example_id="doc1", original_text=text, rubric=rubric, law_type="sufficiency"
        )
        idempotence_pairs = collector.collect_pairs_for_example(
            example_id="doc1", original_text=text, rubric=rubric, law_type="idempotence"
        )
        merge_pairs = collector.collect_pairs_for_example(
            example_id="doc1", original_text=text, rubric=rubric, law_type="merge"
        )
    """

    def __init__(
        self,
        summarizer: dspy.Module,
        judge: Optional[PairwiseJudge] = None,
        k: int = 4,
        generation_configs: Optional[List[GenerationConfig]] = None,
        # Strategy support
        strategy: Literal["judge", "genrm", "oracle"] = "judge",
        genrm_judge: Optional[Any] = None,  # GenRMJudge
        oracle_predict: Optional[Any] = None,  # Callable[[str], float]
        tie_margin: float = 5.0,  # For oracle strategy
    ):
        """
        Initialize the collector.

        Args:
            summarizer: DSPy module for generating summaries
            judge: PairwiseJudge for comparing summaries (required for strategy="judge")
            k: Number of candidate summaries to generate per input
            generation_configs: Configurations for generating diverse outputs
            strategy: Preference derivation strategy ("judge", "genrm", "oracle")
            genrm_judge: GenRMJudge instance (required for strategy="genrm")
            oracle_predict: Function (text) -> float (required for strategy="oracle")
            tie_margin: Error margin for ties in oracle strategy
        """
        # Build generation configs first (needed for super().__init__)
        if generation_configs is None:
            prompt_variants = ["concise", "default", "detailed", "creative"]
            generation_configs = [
                GenerationConfig(temperature=temp, prompt_variant=variant)
                for temp, variant in zip(DIVERSE_TEMPERATURES, prompt_variants)
            ]

        # Initialize base class (provides OPS law support)
        super().__init__(
            summarizer=summarizer,
            k_candidates=k,
            generation_configs=generation_configs,
        )

        self.k = k
        self.strategy = strategy
        self.tie_margin = tie_margin

        # Strategy-specific components
        self.judge = judge
        self.genrm_judge = genrm_judge
        self.oracle_predict = oracle_predict

        # Validate strategy configuration
        if strategy == "judge" and judge is None:
            raise ValueError("judge is required when strategy='judge'")
        if strategy == "genrm" and genrm_judge is None:
            raise ValueError("genrm_judge is required when strategy='genrm'")
        if strategy == "oracle" and oracle_predict is None:
            raise ValueError("oracle_predict is required when strategy='oracle'")

        self._oracle_cache: Dict[str, float] = {}  # For oracle strategy

    def _create_candidate_metadata(
        self,
        gen_config: GenerationConfig,
        index: int,
    ) -> GenerationConfig:
        """Create metadata for a candidate (returns the GenerationConfig itself)."""
        return gen_config

    def _get_generation_config_dict(
        self,
        metadata: GenerationConfig,
    ) -> Dict[str, Any]:
        """Convert GenerationConfig metadata to dictionary."""
        return metadata.to_dict()

    def _get_pair_id_prefix(self) -> str:
        """Return strategy-specific pair ID prefix."""
        return {"judge": "pair", "genrm": "genrm", "oracle": "oracle"}.get(self.strategy, "pref")

    def _get_judge_model_name(self) -> str:
        """Return name of the judge model."""
        if self.strategy == "genrm" and self.genrm_judge is not None:
            return getattr(self.genrm_judge, "model_name", "genrm")
        return ""

    def _get_oracle_score(self, text: str) -> float:
        """Get oracle score with caching (for oracle strategy)."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self._oracle_cache:
            self._oracle_cache[text_hash] = self.oracle_predict(text)
        return self._oracle_cache[text_hash]

    def _derive_preference(
        self,
        candidate_a: CandidateInfo[GenerationConfig],
        candidate_b: CandidateInfo[GenerationConfig],
        context: Dict[str, Any],
    ) -> PreferenceResult:
        """
        Derive preference using the configured strategy.

        Implements the abstract method from BasePreferenceCollector.

        Args:
            candidate_a: First candidate with summary and metadata
            candidate_b: Second candidate with summary and metadata
            context: Dictionary with original_text, rubric, reference_score, law_type

        Returns:
            PreferenceResult with preference, confidence, and reasoning
        """
        summary_a = candidate_a.summary
        summary_b = candidate_b.summary
        original_text = context.get("original_text", "")
        rubric = context.get("rubric", "")
        reference_score = context.get("reference_score", 0.0)
        law_type = context.get("law_type", "sufficiency")

        if self.strategy == "judge":
            # PairwiseJudge-based comparison
            result = self.judge(
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                rubric=rubric,
                reference_score=reference_score,
            )
            return PreferenceResult(
                preferred=result["preferred"],
                reasoning=result["reasoning"],
                confidence=result["confidence"],
                score_estimate_a=result.get("score_estimate_a"),
                score_estimate_b=result.get("score_estimate_b"),
            )

        elif self.strategy == "genrm":
            # GenRM-based comparison
            result = self.genrm_judge.compare(
                context=rubric,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
                law_type=law_type,
            )
            # Handle error results
            if hasattr(result, 'is_error') and result.is_error():
                return PreferenceResult(
                    preferred="tie",
                    reasoning=f"GenRM error: {result.error_message}",
                    confidence=0.0,
                )
            return PreferenceResult(
                preferred=result.preferred,
                reasoning=result.reasoning,
                confidence=result.confidence,
                score_estimate_a=result.helpfulness_a,
                score_estimate_b=result.helpfulness_b,
            )

        elif self.strategy == "oracle":
            # Oracle-based comparison using error difference
            score_a = self._get_oracle_score(summary_a)
            score_b = self._get_oracle_score(summary_b)

            error_a = abs(score_a - reference_score)
            error_b = abs(score_b - reference_score)
            error_diff = error_a - error_b

            # Lower error is better, so positive diff means A is worse
            if error_diff > self.tie_margin:
                preferred = "B"
                confidence = min(0.5 + abs(error_diff) / 50, 0.95)
            elif error_diff < -self.tie_margin:
                preferred = "A"
                confidence = min(0.5 + abs(error_diff) / 50, 0.95)
            else:
                preferred = "tie"
                confidence = 0.5

            return PreferenceResult(
                preferred=preferred,
                reasoning=f"Oracle errors: A={error_a:.2f}, B={error_b:.2f}",
                confidence=confidence,
                score_estimate_a=score_a,
                score_estimate_b=score_b,
                oracle_error_a=error_a,
                oracle_error_b=error_b,
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    # Backward compatibility: collect_pairs delegates to collect_pairs_for_example
    def collect_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        reference_score: float,
        law_type: str = "sufficiency",
        judge_model: str = "",
    ) -> List[PreferencePair]:
        """
        Generate candidates and collect all pairwise preferences for one example.

        This is a backward-compatible wrapper around collect_pairs_for_example.

        Args:
            example_id: Unique identifier for this example
            original_text: Original source text
            rubric: Information preservation rubric
            reference_score: Ground truth score for original
            law_type: OPS law type (sufficiency, idempotence, merge)
            judge_model: Name of the judge model being used (ignored, uses strategy)

        Returns:
            List of preference pairs
        """
        return self.collect_pairs_for_example(
            example_id=example_id,
            original_text=original_text,
            rubric=rubric,
            reference_score=reference_score,
            law_type=law_type,
        )

    @property
    def pairs(self) -> List[PreferencePair]:
        """Return all collected preference pairs (property for backward compatibility)."""
        # Use parent class's pairs list
        return super().get_all_pairs()

    def get_non_tie_pairs(self) -> List[PreferencePair]:
        """Return only pairs with clear preferences (no ties)."""
        return [p for p in self.pairs if p.preferred != "tie"]

    def get_high_confidence_pairs(self, threshold: float = 0.7) -> List[PreferencePair]:
        """Return pairs with confidence above threshold."""
        return [p for p in self.pairs if p.confidence >= threshold]

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive collection statistics."""
        return self.stats.to_dict()
