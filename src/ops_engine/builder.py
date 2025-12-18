"""
OPS Tree Builder - Constructs summarization trees from document chunks.

The builder creates trees bottom-up, starting from leaf nodes (text chunks)
and recursively summarizing pairs of nodes until a single root remains.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol, Any
from pathlib import Path
import logging

from src.core.data_models import (
    OPSNode, OPSTree, create_leaf_node, create_internal_node
)
from src.preprocessing.chunker import TextChunk, DocumentChunker, chunk_for_ops


logger = logging.getLogger(__name__)


class Summarizer(Protocol):
    """Protocol for summarization functions."""

    def __call__(self, content: str, rubric: str) -> str:
        """
        Summarize content according to rubric.

        Args:
            content: Text to summarize
            rubric: Information preservation criteria

        Returns:
            Summary text
        """
        ...


class IdentitySummarizer:
    """
    Summarizer that returns content unchanged.
    Useful for testing tree structure without LLM calls.
    """

    def __call__(self, content: str, rubric: str) -> str:
        return content


class TruncatingSummarizer:
    """
    Simple summarizer that truncates to a max length.
    Useful for testing without LLM.
    """

    def __init__(self, max_length: int = 500, suffix: str = "..."):
        self.max_length = max_length
        self.suffix = suffix

    def __call__(self, content: str, rubric: str) -> str:
        if len(content) <= self.max_length:
            return content
        return content[:self.max_length - len(self.suffix)] + self.suffix


class ConcatenatingSummarizer:
    """
    Summarizer that concatenates with a separator.
    Useful for testing to see the full tree content.
    """

    def __init__(self, prefix: str = "[Summary] "):
        self.prefix = prefix

    def __call__(self, content: str, rubric: str) -> str:
        # Add a prefix to show summarization happened
        lines = content.strip().split('\n')
        # Take first and last line to show range
        if len(lines) > 2:
            return f"{self.prefix}From: {lines[0][:50]}... To: {lines[-1][:50]}..."
        return f"{self.prefix}{content[:100]}..."


@dataclass
class BuildConfig:
    """Configuration for tree building."""

    # Chunking settings
    max_chunk_chars: int = 2000
    min_chunk_chars: int = 100
    chunk_strategy: str = "sentence"  # "sentence" or "paragraph"

    # Tree settings
    merge_strategy: str = "binary"  # "binary" for 2-way merge

    # Debug settings
    verbose: bool = False


@dataclass
class BuildResult:
    """Result of tree building operation."""
    tree: OPSTree
    chunks_created: int
    nodes_created: int
    levels_created: int
    errors: List[str] = field(default_factory=list)


class OPSTreeBuilder:
    """
    Builds OPS trees from documents.

    The builder:
    1. Chunks the document into leaf-sized pieces
    2. Creates leaf nodes from chunks
    3. Recursively merges nodes bottom-up
    4. Returns a complete tree with root summary

    Example:
        >>> builder = OPSTreeBuilder(summarizer=my_llm_summarizer)
        >>> tree = builder.build_from_text(document_text, rubric="Preserve key facts")
        >>> print(tree.final_summary)
    """

    def __init__(
        self,
        summarizer: Summarizer,
        config: Optional[BuildConfig] = None
    ):
        """
        Initialize the builder.

        Args:
            summarizer: Function to summarize text
            config: Build configuration
        """
        self.summarizer = summarizer
        self.config = config or BuildConfig()
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0
        }

    def build_from_text(self, text: str, rubric: str = "") -> BuildResult:
        """
        Build a tree from raw text.

        Args:
            text: Document text to process
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        if not text or not text.strip():
            raise ValueError("Cannot build tree from empty text")

        # Chunk the text
        chunks = chunk_for_ops(
            text,
            max_chars=self.config.max_chunk_chars,
            strategy=self.config.chunk_strategy
        )

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        return self.build_from_chunks(chunks, rubric)

    def build_from_chunks(
        self,
        chunks: List[TextChunk],
        rubric: str = ""
    ) -> BuildResult:
        """
        Build a tree from pre-chunked text.

        Args:
            chunks: List of TextChunk objects
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        if not chunks:
            raise ValueError("Cannot build tree from empty chunks list")

        errors = []

        # Create leaf nodes
        leaves = []
        for i, chunk in enumerate(chunks):
            try:
                leaf = create_leaf_node(chunk.text, node_id=f"leaf_{i}")
                leaf.audit_result.trace = {'chunk_index': i}
                leaves.append(leaf)
            except Exception as e:
                errors.append(f"Failed to create leaf {i}: {e}")

        if not leaves:
            raise ValueError("No leaf nodes created")

        # Build tree bottom-up
        tree = self._build_tree_from_leaves(leaves, rubric)

        # Calculate statistics
        total_nodes = tree.node_count

        return BuildResult(
            tree=tree,
            chunks_created=len(chunks),
            nodes_created=total_nodes,
            levels_created=tree.height + 1,
            errors=errors
        )

    def build_from_file(self, filepath: Path, rubric: str = "") -> BuildResult:
        """
        Build a tree from a file.

        Args:
            filepath: Path to text file
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = filepath.read_text(encoding='utf-8')
        result = self.build_from_text(text, rubric)

        # Add file metadata
        result.tree.metadata['source_file'] = str(filepath)

        return result

    def _build_tree_from_leaves(
        self,
        leaves: List[OPSNode],
        rubric: str
    ) -> OPSTree:
        """
        Build tree by recursively merging leaves.

        Args:
            leaves: List of leaf nodes
            rubric: Information preservation criteria

        Returns:
            Complete OPSTree
        """
        if len(leaves) == 1:
            # Single leaf is the root
            return OPSTree(root=leaves[0], rubric=rubric)

        current_level = list(leaves)
        level_num = 0

        if self.config.verbose:
            logger.info(f"Starting build with {len(leaves)} leaves")

        while len(current_level) > 1:
            level_num += 1

            if self.config.verbose:
                logger.info(f"Building level {level_num} from {len(current_level)} nodes")

            # Collect pairs for this level
            pairs = []
            odd_node = None
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    pairs.append((current_level[i], current_level[i + 1]))
                else:
                    odd_node = current_level[i]

            # Process pairs in parallel using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            next_level = []
            if pairs:
                # Send all requests in parallel - vLLM handles batching internally
                with ThreadPoolExecutor(max_workers=len(pairs)) as executor:
                    # Submit all merge tasks
                    future_to_idx = {
                        executor.submit(self._merge_nodes, left, right, rubric, level_num): idx
                        for idx, (left, right) in enumerate(pairs)
                    }

                    # Collect results in order
                    results = [None] * len(pairs)
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            results[idx] = future.result()
                        except Exception as e:
                            logger.error(f"Merge failed for pair {idx}: {e}")
                            # Create a fallback node with concatenated content
                            left, right = pairs[idx]
                            fallback_content = f"{left.content}\n\n{right.content}"[:5000]
                            results[idx] = create_internal_node(
                                summary=fallback_content,
                                children=[left, right],
                                level=level_num
                            )

                    next_level = [r for r in results if r is not None]

            # Add odd node if present
            if odd_node is not None:
                next_level.append(odd_node)

            current_level = next_level

        root = current_level[0]

        # Ensure root level is correct
        if not root.is_leaf and root.level == 0:
            root.level = level_num

        return OPSTree(root=root, rubric=rubric)

    def _merge_nodes(
        self,
        left: OPSNode,
        right: OPSNode,
        rubric: str,
        level: int
    ) -> OPSNode:
        """
        Merge two nodes into a parent node.

        Args:
            left: Left child
            right: Right child
            rubric: Summarization rubric
            level: Level for the new node

        Returns:
            Parent node with summary
        """
        # Combine child summaries
        combined = f"{left.summary}\n\n{right.summary}"

        # Summarize
        self._build_stats['summarizer_calls'] += 1
        self._build_stats['total_input_chars'] += len(combined)

        summary = self.summarizer(combined, rubric)

        self._build_stats['total_output_chars'] += len(summary)

        # Create parent
        parent = create_internal_node(
            left=left,
            right=right,
            summary=summary,
            node_id=f"node_L{level}_{self._build_stats['summarizer_calls']}"
        )

        return parent

    def get_stats(self) -> dict:
        """Get build statistics."""
        return dict(self._build_stats)

    def reset_stats(self) -> None:
        """Reset build statistics."""
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0
        }


def build_ops_tree(
    text: str,
    rubric: str = "",
    summarizer: Optional[Summarizer] = None,
    max_chunk_chars: int = 2000
) -> OPSTree:
    """
    Convenience function to build an OPS tree.

    Args:
        text: Document text
        rubric: Information preservation criteria
        summarizer: Summarization function (defaults to identity)
        max_chunk_chars: Maximum chunk size

    Returns:
        OPSTree
    """
    if summarizer is None:
        summarizer = IdentitySummarizer()

    config = BuildConfig(max_chunk_chars=max_chunk_chars)
    builder = OPSTreeBuilder(summarizer=summarizer, config=config)
    result = builder.build_from_text(text, rubric)

    return result.tree


def build_test_tree(num_leaves: int = 4) -> OPSTree:
    """
    Build a simple test tree with predictable structure.

    Args:
        num_leaves: Number of leaf nodes

    Returns:
        OPSTree with numbered nodes
    """
    # Create simple numbered content
    chunks = [TextChunk(text=f"Chunk {i} content.", start_char=i*20, end_char=(i+1)*20, chunk_index=i)
              for i in range(num_leaves)]

    summarizer = ConcatenatingSummarizer()
    config = BuildConfig(verbose=False)
    builder = OPSTreeBuilder(summarizer=summarizer, config=config)

    result = builder.build_from_chunks(chunks, rubric="Test rubric")
    return result.tree
