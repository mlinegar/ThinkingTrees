"""
Core prompt builder helpers.

These are the default prompt builders used by strategies when no task-specific
prompts are provided. They're intentionally simple and task-agnostic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import re


@dataclass
class PromptBuilders:
    """Container for task-specific prompt builders."""
    summarize: Callable[[str, str], List[Dict[str, str]]]
    merge: Callable[[str, str, str], List[Dict[str, str]]]
    score: Optional[Callable[[str, str], List[Dict[str, str]]]] = None
    audit: Optional[Callable[[str, str, str], List[Dict[str, str]]]] = None


def default_summarize_prompt(text: str, rubric: str) -> List[Dict[str, str]]:
    """Default summarization prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are a text summarizer. Preserve all information "
                "relevant to the preservation criteria. Be concise but complete."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following text, preserving information "
                f"relevant to: {rubric}\n\n"
                f"TEXT:\n{text}\n\n"
                f"SUMMARY:"
            ),
        },
    ]


def default_merge_prompt(left: str, right: str, rubric: str) -> List[Dict[str, str]]:
    """Default merge prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are a text summarizer. Combine the following two "
                "summaries into one coherent summary, preserving all "
                "relevant information."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Combine these two summaries, preserving information relevant to: {rubric}\n\n"
                f"SUMMARY 1:\n{left}\n\n"
                f"SUMMARY 2:\n{right}\n\n"
                f"COMBINED SUMMARY:"
            ),
        },
    ]


def parse_numeric_score(response: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Optional[float]:
    """Parse a numeric score from text."""
    if not response:
        return None

    match = re.search(r"(-?\d+\.?\d*)", response)
    if not match:
        return None

    try:
        value = float(match.group(1))
    except ValueError:
        return None

    if min_value is not None and value < min_value:
        return None
    if max_value is not None and value > max_value:
        return None
    return value
