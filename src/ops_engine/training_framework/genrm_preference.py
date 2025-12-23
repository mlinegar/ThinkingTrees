"""
GenRM-based Preference Collection for OPS Summarization.

Uses NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM model to compare summaries.
The GenRM model uses a special format with response_1 and response_2 roles
and produces helpfulness scores (1-5) and ranking scores (1-6).

Ranking score interpretation:
    1 = Response 1 is much better than Response 2
    2 = Response 1 is better than Response 2
    3 = Response 1 is slightly better than Response 2
    4 = Response 2 is slightly better than Response 1
    5 = Response 2 is better than Response 1
    6 = Response 2 is much better than Response 1

Usage:
    from src.ops_engine.training_framework.genrm_preference import (
        GenRMJudge,
        GenRMPreferenceCollector,
    )

    # Create judge connected to GenRM server
    judge = GenRMJudge(base_url="http://localhost:8001/v1")

    # Compare two summaries
    result = judge.compare(
        context="Summarize this political text preserving its left-right stance",
        original_text="...",
        summary_a="...",
        summary_b="...",
    )
    # result.preferred = "A" or "B" or "tie"
    # result.ranking_score = 1-6
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Literal

import requests
import dspy

from .preference import PreferencePair, PreferenceDataset

logger = logging.getLogger(__name__)


@dataclass
class GenRMResult:
    """Result from GenRM comparison."""
    preferred: Literal["A", "B", "tie"]
    ranking_score: int  # 1-6
    helpfulness_a: float  # 1-5
    helpfulness_b: float  # 1-5
    reasoning: str
    confidence: float
    raw_response: str = ""


class GenRMJudge:
    """
    Judge using NVIDIA's Qwen3-Nemotron-235B-A22B-GenRM.

    Uses the special response_1/response_2 format for comparison.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ):
        """
        Initialize the GenRM judge.

        Args:
            base_url: vLLM server base URL
            model_name: Model name for API requests
            temperature: Generation temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens for response
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def _build_messages(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages in GenRM format with response_1 and response_2.

        The format requires:
        - Conversation history with user/assistant roles
        - Last turn must be a user message
        - Two candidate responses as response_1 and response_2
        """
        law_instructions = {
            "sufficiency": (
                "Compare which summary better preserves the oracle-relevant "
                "information from the original text."
            ),
            "idempotence": (
                "Compare which summary is more stable under re-summarization. "
                "Use the provided resummaries to judge drift."
            ),
            "merge": (
                "Compare which merged summary better preserves the information "
                "from its child summaries."
            ),
        }
        instruction = law_instructions.get(law_type, law_instructions["sufficiency"])

        original_section = ""
        if original_text.strip():
            original_section = f"\n\nOriginal Text:\n{original_text}"

        extra_section = ""
        if extra_context:
            extra_section = f"\n\nAdditional Context:\n{extra_context}"

        # Create the comparison task as user message
        user_message = (
            "Please compare the following two candidate summaries.\n"
            f"OPS law: {law_type}\n"
            f"{instruction}\n\n"
            f"Context (what to preserve): {context}"
            f"{original_section}"
            f"{extra_section}\n\n"
            "Evaluate the candidates below on:\n"
            "1. Preservation of oracle-relevant information\n"
            "2. Accuracy and faithfulness\n"
            "3. Completeness vs. conciseness tradeoff"
        )

        return [
            {"role": "user", "content": user_message},
            {"role": "response_1", "content": summary_a},
            {"role": "response_2", "content": summary_b},
        ]

    def compare(
        self,
        context: str,
        original_text: str,
        summary_a: str,
        summary_b: str,
        law_type: str = "sufficiency",
        extra_context: Optional[str] = None,
    ) -> GenRMResult:
        """
        Compare two summaries using GenRM.

        Args:
            context: Description of what information to preserve
            original_text: Original text being summarized
            summary_a: First candidate summary
            summary_b: Second candidate summary

        Returns:
            GenRMResult with preference and scores
        """
        messages = self._build_messages(
            context=context,
            original_text=original_text,
            summary_a=summary_a,
            summary_b=summary_b,
            law_type=law_type,
            extra_context=extra_context,
        )

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_tokens,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return self._parse_genrm_response(content)

        except Exception as e:
            logger.error(f"GenRM request failed: {e}")
            # Return tie with low confidence on error
            return GenRMResult(
                preferred="tie",
                ranking_score=3,
                helpfulness_a=3.0,
                helpfulness_b=3.0,
                reasoning=f"Error during comparison: {e}",
                confidence=0.0,
                raw_response="",
            )

    def _parse_genrm_response(self, content: str) -> GenRMResult:
        """
        Parse GenRM response to extract scores and preference.

        GenRM typically produces structured output with:
        - Helpfulness scores for each response (1-5)
        - Ranking score (1-6) indicating preference strength
        """
        # Try to extract helpfulness scores
        helpfulness_a = 3.0
        helpfulness_b = 3.0
        ranking_score = 3

        # Look for patterns like "Helpfulness: X/5" or "Score: X"
        help_pattern = r"(?:helpfulness|score|rating)[^\d]*(\d(?:\.\d)?)"

        # Extract numbers from response
        numbers = re.findall(r"\b([1-5](?:\.[0-9])?)\b", content)
        if len(numbers) >= 2:
            try:
                helpfulness_a = float(numbers[0])
                helpfulness_b = float(numbers[1])
            except ValueError:
                pass

        # Look for ranking score (1-6)
        ranking_pattern = r"(?:ranking|overall|preference)[^\d]*([1-6])"
        ranking_match = re.search(ranking_pattern, content, re.IGNORECASE)
        if ranking_match:
            ranking_score = int(ranking_match.group(1))
        else:
            # Infer from helpfulness
            if helpfulness_a > helpfulness_b + 0.5:
                ranking_score = 2  # A is better
            elif helpfulness_b > helpfulness_a + 0.5:
                ranking_score = 5  # B is better
            else:
                ranking_score = 3  # Roughly equal

        # Determine preference from ranking score
        if ranking_score <= 2:
            preferred = "A"
            confidence = (3 - ranking_score) * 0.3 + 0.4  # 0.7-1.0
        elif ranking_score >= 5:
            preferred = "B"
            confidence = (ranking_score - 4) * 0.3 + 0.4  # 0.7-1.0
        elif ranking_score == 3:
            preferred = "tie"
            confidence = 0.5
        else:
            preferred = "B"
            confidence = 0.55

        return GenRMResult(
            preferred=preferred,
            ranking_score=ranking_score,
            helpfulness_a=helpfulness_a,
            helpfulness_b=helpfulness_b,
            reasoning=content[:500],  # First 500 chars as reasoning
            confidence=confidence,
            raw_response=content,
        )


class GenRMPreferenceCollector:
    """
    Collects preference pairs using GenRM for comparison.

    Generates multiple candidate summaries and uses GenRM to compare them,
    building a preference dataset for training.
    """

    def __init__(
        self,
        summarizer,  # DSPy module or callable that produces summaries
        judge: GenRMJudge,
        k_candidates: int = 4,
        temperatures: Optional[List[float]] = None,
    ):
        """
        Initialize the collector.

        Args:
            summarizer: Module for generating summaries
            judge: GenRMJudge for comparing summaries
            k_candidates: Number of candidate summaries per input
            temperatures: List of temperatures for diverse generation
        """
        self.summarizer = summarizer
        self.judge = judge
        self.k_candidates = k_candidates
        self.temperatures = temperatures or [0.3, 0.5, 0.7, 0.9]

        self.pairs: List[PreferencePair] = []
        self._pair_counter = 0

    def generate_candidates(
        self,
        content: str,
        rubric: str,
    ) -> List[str]:
        """Generate k candidate summaries with varying temperatures."""
        candidates = []

        for temp in self.temperatures[:self.k_candidates]:
            lm = getattr(dspy, "settings", None)
            lm = getattr(lm, "lm", None)
            prev_temp = None
            if lm is not None and hasattr(lm, "kwargs"):
                prev_temp = lm.kwargs.get("temperature")
                lm.kwargs["temperature"] = temp

            try:
                result = self.summarizer(content=content, rubric=rubric)
                summary = getattr(result, 'summary', str(result))
                candidates.append(summary)
            except Exception as e:
                logger.warning(f"Failed to generate candidate: {e}")
            finally:
                if lm is not None and hasattr(lm, "kwargs"):
                    if prev_temp is None:
                        lm.kwargs.pop("temperature", None)
                    else:
                        lm.kwargs["temperature"] = prev_temp

        return candidates

    def collect_pairs_for_example(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: float = 0.0,
        law_type: str = "sufficiency",
    ) -> List[PreferencePair]:
        """
        Generate candidates and collect all pairwise preferences.

        Args:
            example_id: Unique identifier
            original_text: Original text to summarize
            rubric: What information to preserve
            ground_truth_score: Optional ground truth score

        Returns:
            List of preference pairs
        """
        if law_type == "idempotence":
            return self._collect_idempotence_pairs(
                example_id, original_text, rubric, ground_truth_score
            )
        if law_type == "merge":
            return self._collect_merge_pairs(
                example_id, original_text, rubric, ground_truth_score
            )
        return self._collect_sufficiency_pairs(
            example_id, original_text, rubric, ground_truth_score
        )

    def _collect_sufficiency_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: float,
    ) -> List[PreferencePair]:
        """Collect preference pairs for sufficiency."""
        candidates = self.generate_candidates(original_text, rubric)

        if len(candidates) < 2:
            logger.warning(f"Only {len(candidates)} candidates for {example_id}")
            return []

        pairs: List[PreferencePair] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                summary_a = candidates[i]
                summary_b = candidates[j]
                idx_a, idx_b = i, j

                swapped = random.random() < 0.5
                if swapped:
                    summary_a, summary_b = summary_b, summary_a
                    idx_a, idx_b = idx_b, idx_a

                try:
                    result = self.judge.compare(
                        context=rubric,
                        original_text=original_text,
                        summary_a=summary_a,
                        summary_b=summary_b,
                        law_type="sufficiency",
                    )

                    preferred = result.preferred

                    self._pair_counter += 1
                    pair = PreferencePair(
                        pair_id=f"genrm_{self._pair_counter:06d}",
                        source_example_id=example_id,
                        original_text=original_text,
                        rubric=rubric,
                        ground_truth_score=ground_truth_score,
                        law_type="sufficiency",
                        summary_a=summary_a,
                        summary_b=summary_b,
                        preferred=preferred,
                        reasoning=f"{result.reasoning} (candidates {idx_a},{idx_b})",
                        confidence=result.confidence,
                        score_estimate_a=result.helpfulness_a,
                        score_estimate_b=result.helpfulness_b,
                        judge_model="qwen3-nemotron-genrm",
                        generation_config_a={"temperature": self.temperatures[idx_a]},
                        generation_config_b={"temperature": self.temperatures[idx_b]},
                    )
                    pairs.append(pair)
                    self.pairs.append(pair)

                except Exception as e:
                    logger.error(f"Failed to judge pair: {e}")

        return pairs

    def _collect_idempotence_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: float,
    ) -> List[PreferencePair]:
        """Collect preference pairs for idempotence."""
        candidates = self.generate_candidates(original_text, rubric)

        if len(candidates) < 2:
            logger.warning(f"Only {len(candidates)} candidates for {example_id}")
            return []

        resummaries: List[str] = []
        for summary in candidates:
            try:
                result = self.summarizer(content=summary, rubric=rubric)
                resummary = getattr(result, "summary", str(result))
                resummaries.append(resummary)
            except Exception as e:
                logger.warning(f"Failed to generate resummary: {e}")
                resummaries.append("")

        pairs: List[PreferencePair] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                summary_a = candidates[i]
                summary_b = candidates[j]
                resummary_a = resummaries[i]
                resummary_b = resummaries[j]
                idx_a, idx_b = i, j

                swapped = random.random() < 0.5
                if swapped:
                    summary_a, summary_b = summary_b, summary_a
                    resummary_a, resummary_b = resummary_b, resummary_a
                    idx_a, idx_b = idx_b, idx_a

                extra_context = (
                    "Idempotence check (re-summaries):\n"
                    f"Candidate A resummary:\n{resummary_a}\n\n"
                    f"Candidate B resummary:\n{resummary_b}"
                )

                try:
                    result = self.judge.compare(
                        context=rubric,
                        original_text="",
                        summary_a=summary_a,
                        summary_b=summary_b,
                        law_type="idempotence",
                        extra_context=extra_context,
                    )

                    preferred = result.preferred

                    self._pair_counter += 1
                    pair = PreferencePair(
                        pair_id=f"genrm_idem_{self._pair_counter:06d}",
                        source_example_id=example_id,
                        original_text=original_text,
                        rubric=rubric,
                        ground_truth_score=ground_truth_score,
                        law_type="idempotence",
                        summary_a=summary_a,
                        summary_b=summary_b,
                        preferred=preferred,
                        reasoning=f"{result.reasoning} (candidates {idx_a},{idx_b})",
                        confidence=result.confidence,
                        score_estimate_a=result.helpfulness_a,
                        score_estimate_b=result.helpfulness_b,
                        judge_model="qwen3-nemotron-genrm",
                        generation_config_a={"temperature": self.temperatures[idx_a]},
                        generation_config_b={"temperature": self.temperatures[idx_b]},
                    )
                    pairs.append(pair)
                    self.pairs.append(pair)

                except Exception as e:
                    logger.error(f"Failed to judge pair: {e}")

        return pairs

    def _collect_merge_pairs(
        self,
        example_id: str,
        original_text: str,
        rubric: str,
        ground_truth_score: float,
    ) -> List[PreferencePair]:
        """Collect preference pairs for merge consistency."""
        words = original_text.split()
        if not words:
            logger.warning(f"No text for merge pairing on {example_id}")
            return []

        mid = len(words) // 2
        text_a = " ".join(words[:mid])
        text_b = " ".join(words[mid:])

        k_per_half = max(2, self.k_candidates // 2)
        candidates_a = self.generate_candidates(text_a, rubric)[:k_per_half]
        candidates_b = self.generate_candidates(text_b, rubric)[:k_per_half]

        if not candidates_a or not candidates_b:
            logger.warning(f"Insufficient candidates for merge on {example_id}")
            return []

        merged_candidates: List[Tuple[str, str, str, int, int]] = []
        for idx_a, summary_a in enumerate(candidates_a):
            for idx_b, summary_b in enumerate(candidates_b):
                merged_text = f"{summary_a}\n\n{summary_b}"
                try:
                    result = self.summarizer(content=merged_text, rubric=rubric)
                    merged_summary = getattr(result, "summary", str(result))
                    merged_candidates.append((summary_a, summary_b, merged_summary, idx_a, idx_b))
                except Exception as e:
                    logger.warning(f"Failed to generate merged summary: {e}")

        if len(merged_candidates) < 2:
            logger.warning(f"Only {len(merged_candidates)} merged candidates for {example_id}")
            return []

        pairs: List[PreferencePair] = []
        for i in range(len(merged_candidates)):
            for j in range(i + 1, len(merged_candidates)):
                left_a, right_a, merged_a, idx_a_left, idx_a_right = merged_candidates[i]
                left_b, right_b, merged_b, idx_b_left, idx_b_right = merged_candidates[j]
                idx_a, idx_b = i, j

                swapped = random.random() < 0.5
                if swapped:
                    left_a, left_b = left_b, left_a
                    right_a, right_b = right_b, right_a
                    merged_a, merged_b = merged_b, merged_a
                    idx_a_left, idx_b_left = idx_b_left, idx_a_left
                    idx_a_right, idx_b_right = idx_b_right, idx_a_right
                    idx_a, idx_b = idx_b, idx_a

                extra_context = (
                    "Merge check (child summaries):\n"
                    f"Candidate A left summary:\n{left_a}\n"
                    f"Candidate A right summary:\n{right_a}\n\n"
                    f"Candidate B left summary:\n{left_b}\n"
                    f"Candidate B right summary:\n{right_b}"
                )

                try:
                    result = self.judge.compare(
                        context=rubric,
                        original_text="",
                        summary_a=merged_a,
                        summary_b=merged_b,
                        law_type="merge",
                        extra_context=extra_context,
                    )

                    preferred = result.preferred

                    self._pair_counter += 1
                    pair = PreferencePair(
                        pair_id=f"genrm_merge_{self._pair_counter:06d}",
                        source_example_id=example_id,
                        original_text=original_text,
                        rubric=rubric,
                        ground_truth_score=ground_truth_score,
                        law_type="merge",
                        summary_a=merged_a,
                        summary_b=merged_b,
                        preferred=preferred,
                        reasoning=f"{result.reasoning} (candidates {idx_a},{idx_b})",
                        confidence=result.confidence,
                        score_estimate_a=result.helpfulness_a,
                        score_estimate_b=result.helpfulness_b,
                        judge_model="qwen3-nemotron-genrm",
                        generation_config_a={
                            "temperature_left": self.temperatures[idx_a_left],
                            "temperature_right": self.temperatures[idx_a_right],
                        },
                        generation_config_b={
                            "temperature_left": self.temperatures[idx_b_left],
                            "temperature_right": self.temperatures[idx_b_right],
                        },
                    )
                    pairs.append(pair)
                    self.pairs.append(pair)

                except Exception as e:
                    logger.error(f"Failed to judge pair: {e}")

        return pairs

    def get_all_pairs(self) -> List[PreferencePair]:
        """Return all collected pairs."""
        return self.pairs

    def get_dataset(self) -> PreferenceDataset:
        """Return pairs as a PreferenceDataset."""
        return PreferenceDataset(self.pairs)

    def get_statistics(self) -> Dict[str, Any]:
        """Return collection statistics."""
        if not self.pairs:
            return {"total": 0}

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


def create_genrm_comparison_prompt(
    rubric: str,
    original_text: str,
    summary_a: str,
    summary_b: str,
    law_type: str = "sufficiency",
    extra_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Create a GenRM-format prompt for summary comparison.

    This is a convenience function for direct API usage.

    Returns:
        List of messages in GenRM format
    """
    original_section = ""
    if original_text.strip():
        original_section = f"\n\nOriginal text:\n{original_text}"

    extra_section = ""
    if extra_context:
        extra_section = f"\n\nAdditional context:\n{extra_context}"

    return [
        {
            "role": "user",
            "content": (
                "Compare these two candidate summaries.\n"
                f"OPS law: {law_type}\n"
                f"Preservation criteria: {rubric}"
                f"{original_section}"
                f"{extra_section}\n\n"
                "Evaluate which candidate better preserves the information specified "
                "in the criteria."
            ),
        },
        {"role": "response_1", "content": summary_a},
        {"role": "response_2", "content": summary_b},
    ]
