"""
Oracle-labeled preference collection for OPS summarization.

This module generates candidate summaries, scores them with an oracle
(e.g., RILE scorer), and converts oracle errors into pairwise preferences.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .base_preference import BasePreferenceCollector, CandidateInfo, PreferenceResult
from .preference import GenerationConfig, PreferenceDataset
from .preference_engine import (
    DEFAULT_ORACLE_ENGINE,
    PreferenceDerivationStrategy,
    PreferenceEngine,
    PreferenceEngineConfig,
)

logger = logging.getLogger(__name__)


class OracleScoringError(Exception):
    """Raised when oracle scoring fails.

    This exception is raised instead of silently falling back to ground_truth,
    which would create artificial ties and corrupt training data.
    """
    pass


OraclePredictor = Callable[[str], float]


@dataclass
class OraclePreferenceConfig:
    """Configuration for oracle-labeled preference collection."""

    tie_margin: float = 5.0
    confidence_floor: float = 0.5


@dataclass
class OracleCandidateMetadata:
    """Metadata for an oracle-scored candidate."""

    generation_config: GenerationConfig
    oracle_score: Optional[float] = None
    oracle_error: Optional[float] = None


class OraclePreferenceCollector(BasePreferenceCollector[OracleCandidateMetadata]):
    """
    Collect preference pairs using a numeric oracle (e.g., RILE score).

    Workflow:
    1. Generate k candidate summaries
    2. Compute oracle value for each candidate (and original)
    3. Convert oracle errors into pairwise preferences with a tie margin

    This class extends BasePreferenceCollector to use oracle-based scoring
    for preference derivation instead of judge-based comparison.
    """

    def __init__(
        self,
        summarizer,
        oracle_predict: OraclePredictor,
        k_candidates: int = 4,
        generation_configs: Optional[List[GenerationConfig]] = None,
        config: Optional[OraclePreferenceConfig] = None,
        oracle_name: str = "",
    ):
        """
        Initialize the oracle preference collector.

        Args:
            summarizer: DSPy module or callable for generating summaries
            oracle_predict: Function to predict oracle score from text
            k_candidates: Number of candidate summaries per input
            generation_configs: Configurations for diverse generation
            config: Oracle preference configuration (tie margin, etc.)
            oracle_name: Name of oracle for metadata
        """
        super().__init__(
            summarizer=summarizer,
            k_candidates=k_candidates,
            generation_configs=generation_configs,
        )

        self.oracle_predict = oracle_predict
        self.config = config or OraclePreferenceConfig()
        self.oracle_name = oracle_name

        # Create preference engine with oracle config
        self.preference_engine = PreferenceEngine(
            PreferenceEngineConfig(
                strategy=PreferenceDerivationStrategy.ERROR_DIFFERENCE,
                tie_margin=self.config.tie_margin,
                confidence_floor=self.config.confidence_floor,
            )
        )

        # Cache for oracle scores during collection
        self._candidate_scores: Dict[int, OracleCandidateMetadata] = {}

        # Text-based cache for oracle predictions (avoids redundant calls)
        self._oracle_cache: Dict[str, float] = {}

    def _get_oracle_score(self, text: str) -> float:
        """Get oracle score with text-based caching."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self._oracle_cache:
            self._oracle_cache[text_hash] = self.oracle_predict(text)
        return self._oracle_cache[text_hash]

    def clear_oracle_cache(self) -> None:
        """Clear the oracle prediction cache."""
        self._oracle_cache.clear()

    def _create_candidate_metadata(
        self,
        gen_config: GenerationConfig,
        index: int,
    ) -> OracleCandidateMetadata:
        """Create metadata with generation config (oracle scores added later)."""
        return OracleCandidateMetadata(generation_config=gen_config)

    def _get_generation_config_dict(
        self,
        metadata: OracleCandidateMetadata,
    ) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return metadata.generation_config.to_dict()

    def _get_pair_id_prefix(self) -> str:
        """Return oracle-specific pair ID prefix."""
        return "oracle"

    def _get_judge_model_name(self) -> str:
        """Return oracle name."""
        return self.oracle_name

    def _derive_preference(
        self,
        candidate_a: CandidateInfo[OracleCandidateMetadata],
        candidate_b: CandidateInfo[OracleCandidateMetadata],
        context: Dict[str, Any],
    ) -> PreferenceResult:
        """
        Derive preference from oracle error comparison.

        Args:
            candidate_a: First candidate
            candidate_b: Second candidate
            context: Contains reference_score, law_type, etc.

        Returns:
            PreferenceResult with preference based on oracle errors
        """
        ground_truth = context.get("reference_score", 0.0)
        law_type = context.get("law_type", "sufficiency")

        # Compute oracle scores and errors for each candidate (with caching)
        # NOTE: We raise OracleScoringError instead of falling back to ground_truth.
        # Falling back would create artificial ties (error=0) and corrupt training data.
        try:
            score_a = self._get_oracle_score(candidate_a.summary)
        except Exception as e:
            raise OracleScoringError(f"Oracle scoring failed for candidate A: {e}") from e

        try:
            score_b = self._get_oracle_score(candidate_b.summary)
        except Exception as e:
            raise OracleScoringError(f"Oracle scoring failed for candidate B: {e}") from e

        # For idempotence, compute idempotence error (drift under re-summarization)
        if law_type == "idempotence":
            resummary_a = context.get("resummary_a", "")
            resummary_b = context.get("resummary_b", "")

            if resummary_a and resummary_b:
                try:
                    score_resummary_a = self._get_oracle_score(resummary_a)
                    error_a = abs(score_a - score_resummary_a)
                except Exception as e:
                    raise OracleScoringError(f"Oracle scoring failed for resummary A: {e}") from e

                try:
                    score_resummary_b = self._get_oracle_score(resummary_b)
                    error_b = abs(score_b - score_resummary_b)
                except Exception as e:
                    raise OracleScoringError(f"Oracle scoring failed for resummary B: {e}") from e
            else:
                error_a = abs(score_a - ground_truth)
                error_b = abs(score_b - ground_truth)
        else:
            # Sufficiency and merge: error is deviation from ground truth
            error_a = abs(score_a - ground_truth)
            error_b = abs(score_b - ground_truth)

        # Use preference engine to derive preference
        preferred, confidence = self.preference_engine.derive_preference(
            error_a=error_a,
            error_b=error_b,
        )

        return PreferenceResult(
            preferred=preferred,
            confidence=confidence,
            reasoning=f"Oracle errors: A={error_a:.2f}, B={error_b:.2f}",
            score_estimate_a=score_a,
            score_estimate_b=score_b,
            oracle_error_a=error_a,
            oracle_error_b=error_b,
        )
