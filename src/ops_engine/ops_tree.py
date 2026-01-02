"""
OPS tree building helpers.

Provides chunking utilities for TreeBuilder:
- chunk_binary: Split text into 2 chunks (for mini-tree construction)

Note: Tournament selection is now handled by TournamentStrategy in src/core/strategy.py
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def chunk_binary(text: str, max_chars: int = 8000) -> List[str]:
    """
    Split text into exactly 2 chunks for mini-tree construction.

    Splits at the midpoint, preferring sentence boundaries.

    Args:
        text: Text to split
        max_chars: Maximum characters per chunk (hint only; no truncation)

    Returns:
        List of exactly 2 chunks
    """
    if not text or not text.strip():
        return ["", ""]

    text = text.strip()

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
