"""
Document chunking for OPS tree construction.

Thin wrapper around langextract's chunking with tiktoken token counting.
"""

from dataclasses import dataclass, field
from typing import List, Iterator
from pathlib import Path

from langextract.chunking import ChunkIterator
from langextract.core.tokenizer import RegexTokenizer


@dataclass
class TextChunk:
    """
    A chunk of text from a document.

    Attributes:
        text: The chunk content
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        chunk_index: Index of this chunk in the sequence
        token_count: Number of tokens (if computed)
        metadata: Additional information about the chunk
    """
    text: str
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk."""
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TextChunk({self.chunk_index}, tokens={self.token_count}, chars={self.char_count})"


class Chunker:
    """
    Chunks documents using langextract's sentence-aware chunking with tiktoken.

    Example:
        >>> chunker = Chunker(max_tokens=2000)
        >>> chunks = chunker.chunk("Long document...")
        >>> for chunk in chunks:
        ...     print(f"Tokens: {chunk.token_count}")
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        model: str = "gpt-4",
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            model: Model name for token counting (e.g., "gpt-4", "qwen3")
        """
        self.max_tokens = max_tokens
        self.model = model
        self._token_counter = None
        self._tokenizer = RegexTokenizer()

    def _get_token_counter(self):
        """Lazy load token counter."""
        if self._token_counter is None:
            from src.preprocessing.tokenizer import TokenCounter
            self._token_counter = TokenCounter(model=self.model)
        return self._token_counter

    def chunk(self, text: str) -> List[TextChunk]:
        """
        Chunk text using langextract's sentence-aware chunking.

        Args:
            text: Text to chunk

        Returns:
            List of TextChunk objects with token counts
        """
        if not text or not text.strip():
            return []

        counter = self._get_token_counter()

        # Estimate max_chars from max_tokens
        max_chars = counter.estimate_chars_from_tokens(self.max_tokens)

        chunk_iter = ChunkIterator(
            text=text,
            max_char_buffer=max_chars,
            tokenizer_impl=self._tokenizer
        )

        chunks = []
        for i, le_chunk in enumerate(chunk_iter):
            chunk_text = le_chunk.chunk_text
            char_interval = le_chunk.char_interval
            token_count = counter.count(chunk_text)

            chunks.append(TextChunk(
                text=chunk_text,
                start_char=char_interval.start_pos,
                end_char=char_interval.end_pos,
                chunk_index=i,
                token_count=token_count,
            ))

        return chunks

    def chunk_file(self, filepath: Path) -> List[TextChunk]:
        """
        Load and chunk a text file.

        Args:
            filepath: Path to the text file

        Returns:
            List of TextChunk objects
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = filepath.read_text(encoding='utf-8')
        chunks = self.chunk(text)

        for chunk in chunks:
            chunk.metadata['source_file'] = str(filepath)

        return chunks

    def iter_chunks(self, text: str) -> Iterator[TextChunk]:
        """Iterate over chunks."""
        for chunk in self.chunk(text):
            yield chunk


def chunk_text(
    text: str,
    max_tokens: int = 2000,
    model: str = "gpt-4"
) -> List[TextChunk]:
    """
    Convenience function for chunking text.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        model: Model name for token counting

    Returns:
        List of TextChunk objects
    """
    chunker = Chunker(max_tokens=max_tokens, model=model)
    return chunker.chunk(text)


def chunk_for_ops(
    text: str,
    max_chars: int = 2000,
    strategy: str = "sentence"
) -> List[TextChunk]:
    """
    Chunk text for OPS tree construction.

    Simple character-based chunking for OPS, using sentence boundaries.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        strategy: Chunking strategy ("sentence" or "paragraph")

    Returns:
        List of TextChunk objects
    """
    if not text or not text.strip():
        return []

    # Simple sentence-based chunking
    # Split on sentence boundaries and group up to max_chars
    import re

    if strategy == "paragraph":
        # Split on double newlines
        segments = re.split(r'\n\n+', text)
    else:
        # Split on sentence endings
        segments = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0
    current_start = 0

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        segment_length = len(segment)

        if current_length + segment_length + 1 > max_chars and current_chunk:
            # Finalize current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                chunk_index=len(chunks),
            ))
            current_start = current_start + len(chunk_text) + 1
            current_chunk = []
            current_length = 0

        current_chunk.append(segment)
        current_length += segment_length + 1

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(TextChunk(
            text=chunk_text,
            start_char=current_start,
            end_char=current_start + len(chunk_text),
            chunk_index=len(chunks),
        ))

    return chunks


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
DocumentChunker = Chunker
ParagraphChunker = Chunker  # Legacy name, Chunker handles paragraphs


