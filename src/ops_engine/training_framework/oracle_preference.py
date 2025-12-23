"""
Oracle-labeled preference collection for OPS summarization.

This module generates candidate summaries, scores them with an oracle
(e.g., RILE scorer), and converts oracle errors into pairwise preferences.
"""

import logging
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import dspy

from .preference import GenerationConfig, PreferencePair, PreferenceDataset

logger = logging.getLogger(__name__)


OraclePredictor = Callable[[str], float]


@dataclass
class OraclePreferenceConfig:
    """Configuration for oracle-labeled preference collection."""
    tie_margin: float = 5.0
    confidence_floor: float = 0.5


class OraclePreferenceCollector:
    """
    Collect preference pairs using a numeric oracle (e.g., RILE score).

    Workflow:
    1. Generate k candidate summaries
    2. Compute oracle value for each candidate (and original)
    3. Convert oracle errors into pairwise preferences with a tie margin
    """

    def __init__(
        self,
        summarizer,
        oracle_predict: OraclePredictor,
        k_candidates: int = 4,
        generation_configs: Optional[List[GenerationConfig]] = None,
        config: Optional[OraclePreferenceConfig] = None,
    ):
        self.summarizer = summarizer
        self.oracle_predict = oracle_predict
        self.k_candidates = k_candidates
        self.generation_configs = generation_configs or [
            GenerationConfig(temperature=0.3, prompt_variant="concise"),
            GenerationConfig(temperature=0.5, prompt_variant="default"),
            GenerationConfig(temperature=0.7, prompt_variant="detailed"),
            GenerationConfig(temperature=0.9, prompt_variant="creative"),
        ]
        self.config = config or OraclePreferenceConfig()

        self.pairs: List[PreferencePair] = []
        self._pair_counter = 0

    def generate_candidates(
        self,
        content: str,
        rubric: str,
    ) -> List[Tuple[str, GenerationConfig]]:
        """Generate k candidate summaries for the input."""
        candidates: List[Tuple[str, GenerationConfig]] = []

        for gen_config in self.generation_configs[:self.k_candidates]:
            lm = getattr(dspy, "settings", None)
            lm = getattr(lm, "lm", None)
            prev_temp = None
            prev_top_p = None
            prev_max_tokens = None
            if lm is not None and hasattr(lm, "kwargs"):
                prev_temp = lm.kwargs.get("temperature")
                prev_top_p = lm.kwargs.get("top_p")
                prev_max_tokens = lm.kwargs.get("max_tokens")
                lm.kwargs["temperature"] = gen_config.temperature
                lm.kwargs["top_p"] = gen_config.top_p
                lm.kwargs["max_tokens"] = gen_config.max_tokens

            try:
                result = self.summarizer(content=content, rubric=rubric)
                summary = getattr(result, "summary", str(result))
                candidates.append((summary, gen_config))
            except Exception as exc:
                logger.warning(f"Failed to generate candidate: {exc}")
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

    def _preference_from_errors(self, error_a: float, error_b: float) -> str:
        diff = error_a - error_b
        if abs(diff) < self.config.tie_margin:
            return "tie"
        return "A" if diff < 0 else "B"

    def _confidence_from_errors(self, error_a: float, error_b: float) -> float:
        diff = abs(error_a - error_b)
        if diff < self.config.tie_margin:
            return self.config.confidence_floor
        scaled = (diff - self.config.tie_margin) / max(1e-6, 2 * self.config.tie_margin)
        return min(1.0, max(self.config.confidence_floor, self.config.confidence_floor + scaled))

    def collect_pairs_for_example(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: Optional[float] = None,
        law_type: str = "sufficiency",
        oracle_name: str = "",
    ) -> List[PreferencePair]:
        """
        Generate candidates and collect oracle-labeled preference pairs.

        Args:
            example_id: Unique identifier for this example
            original_text: Original source text
            rubric: Information preservation rubric
            ground_truth_score: Optional oracle value for original
            law_type: OPS law type (sufficiency, idempotence, merge)
            oracle_name: Oracle identifier for metadata

        Returns:
            List of preference pairs
        """
        if law_type == "idempotence":
            return self._collect_idempotence_pairs(
                example_id, original_text, rubric, ground_truth_score, oracle_name
            )
        elif law_type == "merge":
            return self._collect_merge_pairs(
                example_id, original_text, rubric, ground_truth_score, oracle_name
            )
        else:  # sufficiency
            return self._collect_sufficiency_pairs(
                example_id, original_text, rubric, ground_truth_score, oracle_name
            )

    def _collect_sufficiency_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: Optional[float],
        oracle_name: str,
    ) -> List[PreferencePair]:
        """Collect preference pairs for sufficiency law."""
        candidates = self.generate_candidates(original_text, rubric)
        if len(candidates) < 2:
            logger.warning(f"Only generated {len(candidates)} candidates for {example_id}")
            return []

        if ground_truth_score is None:
            ground_truth_score = self.oracle_predict(original_text)

        candidate_scores: List[Tuple[str, GenerationConfig, float, float]] = []
        for summary, gen_config in candidates:
            try:
                score = self.oracle_predict(summary)
                error = abs(score - ground_truth_score)
                candidate_scores.append((summary, gen_config, score, error))
            except Exception as exc:
                logger.warning(f"Oracle scoring failed: {exc}")

        pairs: List[PreferencePair] = []
        for i in range(len(candidate_scores)):
            for j in range(i + 1, len(candidate_scores)):
                summary_a, config_a, score_a, error_a = candidate_scores[i]
                summary_b, config_b, score_b, error_b = candidate_scores[j]

                # Store original indices for metadata
                idx_a, idx_b = i, j

                swapped = random.random() < 0.5
                if swapped:
                    summary_a, summary_b = summary_b, summary_a
                    config_a, config_b = config_b, config_a
                    score_a, score_b = score_b, score_a
                    error_a, error_b = error_b, error_a
                    idx_a, idx_b = idx_b, idx_a

                preferred = self._preference_from_errors(error_a, error_b)
                confidence = self._confidence_from_errors(error_a, error_b)

                if swapped and preferred != "tie":
                    preferred = "B" if preferred == "A" else "A"

                self._pair_counter += 1
                pair = PreferencePair(
                    pair_id=f"oracle_{self._pair_counter:06d}",
                    source_example_id=example_id,
                    original_text=original_text,
                    rubric=rubric,
                    ground_truth_score=float(ground_truth_score),
                    law_type="sufficiency",
                    summary_a=summary_a,
                    summary_b=summary_b,
                    preferred=preferred,
                    reasoning=f"Oracle errors: A={error_a:.2f}, B={error_b:.2f} (candidates {idx_a},{idx_b})",
                    confidence=confidence,
                    score_estimate_a=score_a,
                    score_estimate_b=score_b,
                    oracle_error_a=error_a,
                    oracle_error_b=error_b,
                    judge_model=oracle_name,
                    generation_config_a=config_a.to_dict(),
                    generation_config_b=config_b.to_dict(),
                )
                pairs.append(pair)
                self.pairs.append(pair)

        return pairs

    def _collect_idempotence_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: Optional[float],
        oracle_name: str,
    ) -> List[PreferencePair]:
        """
        Collect preference pairs for idempotence law.

        Idempotence: summarize(summarize(x)) ≈ summarize(x)

        Generates k candidate summaries, then re-summarizes each one.
        Compares re-summaries with original summaries using oracle error.
        """
        candidates = self.generate_candidates(original_text, rubric)
        if len(candidates) < 2:
            logger.warning(f"Only generated {len(candidates)} candidates for {example_id}")
            return []

        if ground_truth_score is None:
            ground_truth_score = self.oracle_predict(original_text)

        # Generate re-summaries and compute errors
        candidate_scores: List[Tuple[str, str, GenerationConfig, float, float]] = []
        for summary, gen_config in candidates:
            try:
                # Re-summarize the summary
                result = self.summarizer(content=summary, rubric=rubric)
                re_summary = getattr(result, "summary", str(result))

                # Score both summary and re-summary
                score_summary = self.oracle_predict(summary)
                score_re_summary = self.oracle_predict(re_summary)

                # Idempotence error: how much does re-summarizing change the oracle value?
                idempotence_error = abs(score_summary - score_re_summary)

                candidate_scores.append((summary, re_summary, gen_config, score_summary, idempotence_error))
            except Exception as exc:
                logger.warning(f"Idempotence scoring failed: {exc}")

        pairs: List[PreferencePair] = []
        for i in range(len(candidate_scores)):
            for j in range(i + 1, len(candidate_scores)):
                summary_a, re_summary_a, config_a, score_a, error_a = candidate_scores[i]
                summary_b, re_summary_b, config_b, score_b, error_b = candidate_scores[j]

                # Store original indices
                idx_a, idx_b = i, j

                swapped = random.random() < 0.5
                if swapped:
                    summary_a, summary_b = summary_b, summary_a
                    re_summary_a, re_summary_b = re_summary_b, re_summary_a
                    config_a, config_b = config_b, config_a
                    score_a, score_b = score_b, score_a
                    error_a, error_b = error_b, error_a
                    idx_a, idx_b = idx_b, idx_a

                preferred = self._preference_from_errors(error_a, error_b)
                confidence = self._confidence_from_errors(error_a, error_b)

                if swapped and preferred != "tie":
                    preferred = "B" if preferred == "A" else "A"

                self._pair_counter += 1
                pair = PreferencePair(
                    pair_id=f"oracle_idem_{self._pair_counter:06d}",
                    source_example_id=example_id,
                    original_text=original_text,
                    rubric=rubric,
                    ground_truth_score=float(ground_truth_score),
                    law_type="idempotence",
                    summary_a=summary_a,
                    summary_b=summary_b,
                    preferred=preferred,
                    reasoning=f"Idempotence errors: A={error_a:.2f}, B={error_b:.2f} (candidates {idx_a},{idx_b})",
                    confidence=confidence,
                    score_estimate_a=score_a,
                    score_estimate_b=score_b,
                    oracle_error_a=error_a,
                    oracle_error_b=error_b,
                    judge_model=oracle_name,
                    generation_config_a=config_a.to_dict(),
                    generation_config_b=config_b.to_dict(),
                )
                pairs.append(pair)
                self.pairs.append(pair)

        return pairs

    def _collect_merge_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: Optional[float],
        oracle_name: str,
    ) -> List[PreferencePair]:
        """
        Collect preference pairs for merge law.

        Merge: summarize(summary_a + summary_b) should preserve both

        Splits the original text in half, summarizes each half,
        then merges summaries and re-summarizes. Compares merged summaries
        using oracle error against the full original text.
        """
        if ground_truth_score is None:
            ground_truth_score = self.oracle_predict(original_text)

        # Split original text approximately in half
        words = original_text.split()
        mid = len(words) // 2
        text_a = " ".join(words[:mid])
        text_b = " ".join(words[mid:])

        # Generate k/2 summaries for each half
        k_per_half = max(2, self.k_candidates // 2)

        candidates_a = self.generate_candidates(text_a, rubric)[:k_per_half]
        candidates_b = self.generate_candidates(text_b, rubric)[:k_per_half]

        if len(candidates_a) < 1 or len(candidates_b) < 1:
            logger.warning(f"Insufficient candidates for merge law on {example_id}")
            return []

        # Create merged summaries
        candidate_scores: List[Tuple[str, str, str, GenerationConfig, GenerationConfig, float, float]] = []
        for summary_a, config_a in candidates_a:
            for summary_b, config_b in candidates_b:
                try:
                    # Merge and re-summarize
                    merged_text = f"{summary_a}\n\n{summary_b}"
                    result = self.summarizer(content=merged_text, rubric=rubric)
                    merged_summary = getattr(result, "summary", str(result))

                    # Score the merged summary
                    score_merged = self.oracle_predict(merged_summary)

                    # Merge error: how well does merged summary preserve the original?
                    merge_error = abs(score_merged - ground_truth_score)

                    candidate_scores.append((
                        summary_a, summary_b, merged_summary,
                        config_a, config_b, score_merged, merge_error
                    ))
                except Exception as exc:
                    logger.warning(f"Merge scoring failed: {exc}")

        pairs: List[PreferencePair] = []
        for i in range(len(candidate_scores)):
            for j in range(i + 1, len(candidate_scores)):
                _, _, merged_a, config_a_1, config_a_2, score_a, error_a = candidate_scores[i]
                _, _, merged_b, config_b_1, config_b_2, score_b, error_b = candidate_scores[j]

                # Store original indices
                idx_a, idx_b = i, j

                swapped = random.random() < 0.5
                if swapped:
                    merged_a, merged_b = merged_b, merged_a
                    config_a_1, config_b_1 = config_b_1, config_a_1
                    config_a_2, config_b_2 = config_b_2, config_a_2
                    score_a, score_b = score_b, score_a
                    error_a, error_b = error_b, error_a
                    idx_a, idx_b = idx_b, idx_a

                preferred = self._preference_from_errors(error_a, error_b)
                confidence = self._confidence_from_errors(error_a, error_b)

                if swapped and preferred != "tie":
                    preferred = "B" if preferred == "A" else "A"

                self._pair_counter += 1
                pair = PreferencePair(
                    pair_id=f"oracle_merge_{self._pair_counter:06d}",
                    source_example_id=example_id,
                    original_text=original_text,
                    rubric=rubric,
                    ground_truth_score=float(ground_truth_score),
                    law_type="merge",
                    summary_a=merged_a,
                    summary_b=merged_b,
                    preferred=preferred,
                    reasoning=f"Merge errors: A={error_a:.2f}, B={error_b:.2f} (candidates {idx_a},{idx_b})",
                    confidence=confidence,
                    score_estimate_a=score_a,
                    score_estimate_b=score_b,
                    oracle_error_a=error_a,
                    oracle_error_b=error_b,
                    judge_model=oracle_name,
                    generation_config_a=config_a_1.to_dict(),
                    generation_config_b=config_b_1.to_dict(),
                )
                pairs.append(pair)
                self.pairs.append(pair)

        return pairs

    def get_all_pairs(self) -> List[PreferencePair]:
        """Return all collected pairs."""
        return self.pairs

    def get_dataset(self) -> PreferenceDataset:
        """Return all pairs as a PreferenceDataset."""
        return PreferenceDataset(self.pairs)

    def get_statistics(self) -> Dict[str, float]:
        """Return basic collection statistics."""
        if not self.pairs:
            return {"total_pairs": 0}

        prefer_a = sum(1 for p in self.pairs if p.preferred == "A")
        prefer_b = sum(1 for p in self.pairs if p.preferred == "B")
        ties = sum(1 for p in self.pairs if p.preferred == "tie")

        return {
            "total_pairs": len(self.pairs),
            "prefer_a": prefer_a,
            "prefer_b": prefer_b,
            "ties": ties,
            "avg_confidence": sum(p.confidence for p in self.pairs) / len(self.pairs),
            "position_balance": abs(prefer_a - prefer_b) / max(1, prefer_a + prefer_b),
        }
