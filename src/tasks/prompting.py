"""
Prompt builder helpers for task plugins.

This module re-exports from src.core.prompting for backwards compatibility.
The canonical location is now src/core/prompting.py.
"""

from src.core.prompting import (
    PromptBuilders,
    default_summarize_prompt,
    default_merge_prompt,
    parse_numeric_score,
)

__all__ = [
    "PromptBuilders",
    "default_summarize_prompt",
    "default_merge_prompt",
    "parse_numeric_score",
]
