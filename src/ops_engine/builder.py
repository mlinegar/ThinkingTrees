"""
OPS Tree Builder - Constructs summarization trees from document chunks.

The builder creates trees bottom-up, starting from leaf nodes (text chunks)
and recursively summarizing pairs of nodes until a single root remains.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol, Any, Dict, TYPE_CHECKING
from pathlib import Path
import logging
import asyncio
import os

if TYPE_CHECKING:
    from src.ops_engine.training_framework.preference import PreferencePair
    from src.ops_engine.training_framework.genrm_preference import GenRMJudge
    from src.core.strategy import SummarizationStrategy

from src.core.data_models import (
    Node, Tree, leaf, node
)
from src.preprocessing.chunker import TextChunk, DocumentChunker, chunk_for_ops as chunk
from src.ops_engine.ops_tree import candidates, tournament


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

    # Training mode settings (used when judge is provided)
    k: int = 4  # candidates per node
    temperatures: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])

    # Concurrency settings
    max_workers: Optional[int] = None  # Cap merge workers (default: CPU-based)

    # Debug settings
    verbose: bool = False


@dataclass
class BuildResult:
    """Result of tree building operation."""
    tree: Tree
    chunks_created: int
    nodes_created: int
    levels_created: int
    errors: List[str] = field(default_factory=list)
    preferences: List['PreferencePair'] = field(default_factory=list)


class TreeBuilder:
    """
    Builds OPS trees from documents.

    The builder:
    1. Chunks the document into leaf-sized pieces
    2. Creates leaf nodes from chunks
    3. Recursively merges nodes bottom-up
    4. Returns a complete tree with root summary

    Example:
        >>> builder = TreeBuilder(summarizer=my_llm_summarizer)
        >>> tree = builder.build_from_text(document_text, rubric="Preserve key facts")
        >>> print(tree.final_summary)
    """

    def __init__(
        self,
        summarizer: Summarizer,
        judge: Optional['GenRMJudge'] = None,
        config: Optional[BuildConfig] = None
    ):
        """
        Initialize the builder.

        Args:
            summarizer: Function to summarize text
            judge: Optional GenRM judge (enables tournament mode)
            config: Build configuration
        """
        self.summarizer = summarizer
        self.judge = judge
        self.config = config or BuildConfig()
        self._preferences: List['PreferencePair'] = []
        self._errors: List[str] = []  # Track errors during build
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0
        }

    def _summarize(self, content: str, rubric: str, node_id: str, law_type: str = "sufficiency") -> str:
        """Summarize content. Uses tournament selection if judge is available."""
        if self.judge is None:
            return self.summarizer(content, rubric)

        # Generate k candidates
        cands = candidates(content, self.summarizer, rubric, k=self.config.k, temperatures=self.config.temperatures)

        if len(cands) < 2:
            return cands[0] if cands else self.summarizer(content, rubric)

        # Tournament selection
        winner, prefs = tournament(cands, self.judge, content, rubric, segment_id=node_id, law_type=law_type)
        self._preferences.extend(prefs)
        return winner

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
        chunks = chunk(
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

        # Reset errors for this build
        self._errors = []

        # Create leaf nodes
        leaves = []
        for i, chunk in enumerate(chunks):
            try:
                leaf_node = leaf(chunk.text, node_id=f"leaf_{i}")
                leaf_node.audit_result.trace = {'chunk_index': i}
                leaves.append(leaf_node)
            except Exception as e:
                self._errors.append(f"Failed to create leaf {i}: {e}")

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
            errors=self._errors.copy(),
            preferences=self._preferences.copy(),
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
        leaves: List[Node],
        rubric: str
    ) -> Tree:
        """
        Build tree by recursively merging leaves.

        Args:
            leaves: List of leaf nodes
            rubric: Information preservation criteria

        Returns:
            Complete Tree
        """
        if len(leaves) == 1:
            # Single leaf is the root
            return Tree(root=leaves[0], rubric=rubric)

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
                max_workers = self.config.max_workers or min(
                    len(pairs),
                    max(4, (os.cpu_count() or 4)),
                )
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                            # Log and track the error
                            error_msg = f"Merge failed at level {level_num} pair {idx}: {e}"
                            logger.warning(error_msg)
                            self._errors.append(error_msg)
                            # Create a fallback node with concatenated content
                            left, right = pairs[idx]
                            fallback_content = f"{left.summary}\n\n{right.summary}"[:5000]
                            fallback_node = node(
                                left=left,
                                right=right,
                                summary=fallback_content,
                                node_id=f"fallback_L{level_num}_{idx}"
                            )
                            # Mark node as fallback in metadata for downstream visibility
                            fallback_node.audit_result.trace = {
                                'is_fallback': True,
                                'error': str(e),
                            }
                            results[idx] = fallback_node

                    next_level = [r for r in results if r is not None]

            # Add odd node if present
            if odd_node is not None:
                next_level.append(odd_node)

            current_level = next_level

        root = current_level[0]

        # Ensure root level is correct
        if not root.is_leaf and root.level == 0:
            root.level = level_num

        return Tree(root=root, rubric=rubric)

    def _merge_nodes(
        self,
        left: Node,
        right: Node,
        rubric: str,
        level: int
    ) -> Node:
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

        self._build_stats['summarizer_calls'] += 1
        self._build_stats['total_input_chars'] += len(combined)

        node_id = f"L{level}_{self._build_stats['summarizer_calls']}"
        summary = self._summarize(combined, rubric, node_id, law_type="merge")

        self._build_stats['total_output_chars'] += len(summary)

        # Create parent
        parent = node(
            left=left,
            right=right,
            summary=summary,
            node_id=node_id
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

    def reset_preferences(self) -> None:
        """Reset collected preferences for next tree."""
        self._preferences = []

    def reset(self) -> None:
        """Reset all state for reuse."""
        self.reset_stats()
        self.reset_preferences()
        self._errors = []


# =============================================================================
# Async Tree Builder (uses Strategy interface)
# =============================================================================

class AsyncTreeBuilder:
    """
    Async tree builder using the Strategy interface.

    This builder works with any SummarizationStrategy (DSPy or batched),
    enabling the same tree-building logic to work with different backends.

    Example:
        # With DSPy strategy
        strategy = DSPyStrategy(LeafSummarizer(), MergeSummarizer())
        builder = AsyncTreeBuilder(strategy)
        result = await builder.build_from_text(text, rubric)

        # With batched strategy
        async with AsyncBatchLLMClient(url) as client:
            strategy = BatchedStrategy(client)
            builder = AsyncTreeBuilder(strategy)
            result = await builder.build_from_text(text, rubric)
    """

    def __init__(
        self,
        strategy: "SummarizationStrategy",
        config: Optional[BuildConfig] = None
    ):
        """
        Initialize the async builder.

        Args:
            strategy: SummarizationStrategy for summarize/merge operations
            config: Build configuration
        """
        self.strategy = strategy
        self.config = config or BuildConfig()
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0
        }

    async def build_from_text(self, text: str, rubric: str = "") -> BuildResult:
        """
        Build a tree from raw text asynchronously.

        Args:
            text: Document text to process
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        if not text or not text.strip():
            raise ValueError("Cannot build tree from empty text")

        # Chunk the text
        chunks = chunk(
            text,
            max_chars=self.config.max_chunk_chars,
            strategy=self.config.chunk_strategy
        )

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        return await self.build_from_chunks(chunks, rubric)

    async def build_from_chunks(
        self,
        chunks: List[TextChunk],
        rubric: str = ""
    ) -> BuildResult:
        """
        Build a tree from pre-chunked text asynchronously.

        Args:
            chunks: List of TextChunk objects
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        if not chunks:
            raise ValueError("Cannot build tree from empty chunks list")

        errors = []

        # Create leaf nodes with summaries (all in parallel)
        leaves = []
        leaf_tasks = []

        for i, chunk_obj in enumerate(chunks):
            async def summarize_leaf(idx: int, text: str):
                try:
                    summary = await self.strategy.summarize(text, rubric)
                    self._build_stats['summarizer_calls'] += 1
                    self._build_stats['total_input_chars'] += len(text)
                    self._build_stats['total_output_chars'] += len(summary)
                    return idx, leaf(text, summary=summary, node_id=f"leaf_{idx}")
                except Exception as e:
                    errors.append(f"Failed to summarize leaf {idx}: {e}")
                    return idx, leaf(text, node_id=f"leaf_{idx}")

            leaf_tasks.append(summarize_leaf(i, chunk_obj.text))

        # Await all leaf summarizations in parallel
        results = await asyncio.gather(*leaf_tasks, return_exceptions=True)

        # Sort by index and extract nodes
        valid_results = []
        for item in results:
            if isinstance(item, Exception):
                continue
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            idx, node_obj = item
            valid_results.append((idx, node_obj))
        valid_results.sort(key=lambda x: x[0])
        leaves = [n for _, n in valid_results]

        if not leaves:
            raise ValueError("No leaf nodes created")

        # Build tree bottom-up
        tree = await self._build_tree_from_leaves(leaves, rubric)

        return BuildResult(
            tree=tree,
            chunks_created=len(chunks),
            nodes_created=tree.node_count,
            levels_created=tree.height + 1,
            errors=errors,
            preferences=[],  # No tournament in async builder yet
        )

    async def _build_tree_from_leaves(
        self,
        leaves: List[Node],
        rubric: str
    ) -> Tree:
        """
        Build tree by recursively merging leaves asynchronously.

        Args:
            leaves: List of leaf nodes
            rubric: Information preservation criteria

        Returns:
            Complete Tree
        """
        if len(leaves) == 1:
            return Tree(root=leaves[0], rubric=rubric)

        current_level = list(leaves)
        level_num = 0

        if self.config.verbose:
            logger.info(f"Starting async build with {len(leaves)} leaves")

        while len(current_level) > 1:
            level_num += 1

            if self.config.verbose:
                logger.info(f"Building level {level_num} from {len(current_level)} nodes")

            # Collect pairs for this level
            pairs = []
            odd_node = None
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    pairs.append((i // 2, current_level[i], current_level[i + 1]))
                else:
                    odd_node = current_level[i]

            # Create merge tasks for all pairs
            async def merge_pair(idx: int, left: Node, right: Node, level: int):
                try:
                    return idx, await self._merge_nodes(left, right, rubric, level)
                except Exception as e:
                    logger.error(f"Merge failed for pair {idx}: {e}")
                    fallback_content = f"{left.summary}\n\n{right.summary}"[:5000]
                    return idx, node(
                        left=left,
                        right=right,
                        summary=fallback_content,
                        node_id=f"fallback_L{level}_{idx}"
                    )

            merge_tasks = [merge_pair(idx, left, right, level_num)
                          for idx, left, right in pairs]

            # Await all merges in parallel
            results = await asyncio.gather(*merge_tasks)

            # Sort by index and build next level
            results = sorted(results, key=lambda x: x[0])
            next_level = [n for _, n in results]

            if odd_node is not None:
                next_level.append(odd_node)

            current_level = next_level

        root = current_level[0]

        if not root.is_leaf and root.level == 0:
            root.level = level_num

        return Tree(root=root, rubric=rubric)

    async def _merge_nodes(
        self,
        left: Node,
        right: Node,
        rubric: str,
        level: int
    ) -> Node:
        """
        Merge two nodes into a parent node asynchronously.

        Args:
            left: Left child
            right: Right child
            rubric: Summarization rubric
            level: Level for the new node

        Returns:
            Parent node with summary
        """
        self._build_stats['summarizer_calls'] += 1
        self._build_stats['total_input_chars'] += len(left.summary) + len(right.summary)

        node_id = f"L{level}_{self._build_stats['summarizer_calls']}"
        summary = await self.strategy.merge(left.summary, right.summary, rubric)

        self._build_stats['total_output_chars'] += len(summary)

        return node(
            left=left,
            right=right,
            summary=summary,
            node_id=node_id
        )

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


async def async_build(
    text: str,
    rubric: str,
    strategy: "SummarizationStrategy",
    max_chars: int = 2000,
) -> Tree:
    """
    Build an OPS tree asynchronously using a strategy.

    Args:
        text: Document text
        rubric: Information preservation criteria
        strategy: SummarizationStrategy to use
        max_chars: Maximum chunk size

    Returns:
        Tree
    """
    config = BuildConfig(max_chunk_chars=max_chars)
    builder = AsyncTreeBuilder(strategy=strategy, config=config)
    result = await builder.build_from_text(text, rubric)
    return result.tree


def build(
    text: str,
    rubric: str = "",
    summarizer: Optional[Summarizer] = None,
    max_chars: int = 2000
) -> Tree:
    """
    Build an OPS tree from text.

    Args:
        text: Document text
        rubric: Information preservation criteria
        summarizer: Summarization function (defaults to identity)
        max_chars: Maximum chunk size

    Returns:
        Tree
    """
    if summarizer is None:
        summarizer = IdentitySummarizer()

    config = BuildConfig(max_chunk_chars=max_chars)
    builder = TreeBuilder(summarizer=summarizer, config=config)
    result = builder.build_from_text(text, rubric)

    return result.tree


def build_test_tree(num_leaves: int = 4) -> Tree:
    """
    Build a simple test tree with predictable structure.

    Args:
        num_leaves: Number of leaf nodes

    Returns:
        Tree with numbered nodes
    """
    # Create simple numbered content
    chunks = [TextChunk(text=f"Chunk {i} content.", start_char=i*20, end_char=(i+1)*20, chunk_index=i)
              for i in range(num_leaves)]

    summarizer = ConcatenatingSummarizer()
    config = BuildConfig(verbose=False)
    builder = TreeBuilder(summarizer=summarizer, config=config)

    result = builder.build_from_chunks(chunks, rubric="Test rubric")
    return result.tree
