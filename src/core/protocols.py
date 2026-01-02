"""Shared protocol definitions and utility functions for the codebase."""

from typing import Protocol


class Summarizer(Protocol):
    """Protocol for synchronous summarization functions.

    Used by:
    - builder.py: build() helper function with SyncSummarizerAdapter
    - auditor.py: idempotence and substitution checks
    """

    def __call__(self, text: str, rubric: str) -> str:
        """
        Summarize text according to rubric.

        Args:
            text: Input text to summarize
            rubric: Information preservation criteria

        Returns:
            Summary string
        """
        ...


def format_merge_input(*summaries: str) -> str:
    """
    Format merge input by concatenating summaries.

    This is the canonical way to format input for merge operations across the codebase.
    Used in:
    - Tree building (merging nodes)
    - Auditing (merge consistency checks)
    - Preference collection (merge candidate generation)

    Args:
        *summaries: Two or more summaries to concatenate

    Returns:
        Summaries joined with double newlines
    """
    return "\n\n".join(summaries)
