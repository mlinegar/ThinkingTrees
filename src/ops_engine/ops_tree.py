"""
OPS tree building helpers.

Provides chunking and tournament selection utilities for TreeBuilder:
- chunk_binary: Split text into 2 chunks (for mini-tree construction)
- candidates: Generate k summary candidates
- tournament: Select best candidate via pairwise comparison
"""

import logging
from typing import List, Tuple, TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from src.ops_engine.training_framework.preference import PreferencePair
    from src.ops_engine.training_framework.genrm_preference import GenRMJudge

logger = logging.getLogger(__name__)


def chunk_binary(text: str, max_chars: int = 8000) -> List[str]:
    """
    Split text into exactly 2 chunks for mini-tree construction.

    Splits at the midpoint, preferring sentence boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (for truncation)

    Returns:
        List of exactly 2 chunks
    """
    if not text or not text.strip():
        return ["", ""]

    text = text.strip()

    # Truncate if needed
    if len(text) > max_chars * 2:
        text = text[:max_chars * 2]

    # Find midpoint
    midpoint = len(text) // 2

    # Look for sentence boundary near midpoint (within 20% of doc length)
    search_range = len(text) // 5
    best_split = midpoint

    # Search for sentence endings near midpoint
    for offset in range(0, search_range):
        # Check forward
        pos = midpoint + offset
        if pos < len(text) and text[pos] in '.!?\n':
            best_split = pos + 1
            break
        # Check backward
        pos = midpoint - offset
        if pos > 0 and text[pos] in '.!?\n':
            best_split = pos + 1
            break

    left = text[:best_split].strip()
    right = text[best_split:].strip()

    # Ensure we have two non-empty chunks
    if not left:
        left = right[:len(right)//2]
        right = right[len(right)//2:]
    if not right:
        right = left[len(left)//2:]
        left = left[:len(left)//2]

    return [left, right]


def candidates(
    text: str,
    summarizer: dspy.Module,
    rubric: str,
    k: int = 4,
    temperatures: List[float] = None,
) -> List[str]:
    """
    Generate K summary candidates with varying temperatures.

    Args:
        text: Text to summarize
        summarizer: DSPy summarizer module
        rubric: What information to preserve
        k: Number of candidates
        temperatures: Temperature values (defaults to [0.3, 0.5, 0.7, 0.9])

    Returns:
        List of candidate summaries
    """
    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 0.9]

    summaries = []

    for temp in temperatures[:k]:
        try:
            current_lm = dspy.settings.lm
            with dspy.context(lm=current_lm.copy(temperature=temp)):
                result = summarizer(content=text[:8000], rubric=rubric)
                summary = getattr(result, 'summary', str(result))
                if summary and len(summary.strip()) > 10:
                    summaries.append(summary)
        except Exception as e:
            logger.debug(f"Candidate generation failed at temp={temp}: {e}")

    return summaries


def tournament(
    candidate_summaries: List[str],
    judge: 'GenRMJudge',
    original_text: str,
    rubric: str,
    segment_id: str = None,
    law_type: str = "sufficiency",
) -> Tuple[str, List['PreferencePair']]:
    """
    Use GenRM pairwise comparisons to find the best candidate.

    Returns both the winner AND the preference pairs collected during tournament.
    Preferences are FREE as a byproduct of selection.

    Args:
        candidate_summaries: List of candidate summaries
        judge: GenRMJudge instance
        original_text: Original text being summarized
        rubric: What information to preserve
        segment_id: Identifier for this segment
        law_type: OPS law type (sufficiency, merge, idempotence)

    Returns:
        Tuple of (best_candidate, list of PreferencePair)
    """
    from src.ops_engine.training_framework.preference import PreferencePair

    if len(candidate_summaries) == 0:
        raise ValueError("No candidates provided")
    if len(candidate_summaries) == 1:
        return candidate_summaries[0], []

    # Tournament: compare pairs, keep winners, collect preferences
    remaining = candidate_summaries.copy()
    preferences = []
    round_num = 0

    while len(remaining) > 1:
        next_round = []
        for match_num, i in enumerate(range(0, len(remaining), 2)):
            if i + 1 < len(remaining):
                result = judge.compare(
                    context=rubric,
                    original_text=original_text,
                    summary_a=remaining[i],
                    summary_b=remaining[i + 1],
                    law_type=law_type,
                )

                # Capture preference pair (FREE - no extra GenRM call!)
                pair = PreferencePair(
                    pair_id=f"tournament_{segment_id}_r{round_num}_m{match_num}",
                    source_example_id=segment_id or "unknown",
                    original_text=original_text[:4000],  # Truncate for storage
                    rubric=rubric,
                    ground_truth_score=None,  # Filled by tree validation
                    summary_a=remaining[i],
                    summary_b=remaining[i + 1],
                    preferred=result.preferred,
                    reasoning=getattr(result, 'reasoning', "") or "",
                    confidence=getattr(result, 'confidence', 0.5) or 0.5,
                    law_type=law_type,
                    score_estimate_a=getattr(result, 'helpfulness_a', None),
                    score_estimate_b=getattr(result, 'helpfulness_b', None),
                    judge_model="qwen3-nemotron-genrm",
                )
                preferences.append(pair)

                winner = remaining[i] if result.preferred == "A" else remaining[i + 1]
                next_round.append(winner)
            else:
                # Odd one out advances
                next_round.append(remaining[i])
        remaining = next_round
        round_num += 1

    return remaining[0], preferences
