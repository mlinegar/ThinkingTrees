"""
OPS Tree Builder - Constructs summarization trees from document chunks.

The builder creates trees bottom-up, starting from leaf nodes (text chunks)
and recursively summarizing pairs of nodes until a single root remains.

This module provides a unified TreeBuilder that works with any SummarizationStrategy:
- DSPyStrategy for optimization/training
- BatchedStrategy for high-throughput inference
- TournamentStrategy for learning with preference collection

The builder is async-first with a sync wrapper for compatibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, TYPE_CHECKING
from pathlib import Path
import logging
import asyncio
import os

if TYPE_CHECKING:
    from src.training.preference.types import PreferencePair

from src.core.data_models import (
    Node, Tree, leaf, node
)
from src.preprocessing.chunker import TextChunk, DocumentChunker, chunk_for_ops as chunk
from src.core.strategy import SummarizationStrategy, TournamentStrategy
from src.core.protocols import format_merge_input, Summarizer
from src.core.async_utils import gather_with_cleanup


logger = logging.getLogger(__name__)


# =============================================================================
# Chunking Helpers
# =============================================================================

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


# =============================================================================
# Test/Mock Summarizers (for testing without LLM)
# =============================================================================

class IdentitySummarizer:
    """
    Summarizer that returns content unchanged.
    Useful for testing tree structure without LLM calls.
    """

    def __call__(self, content: str, rubric: str) -> str:
        return content


class ConcatenatingSummarizer:
    """
    Summarizer that concatenates with a separator.
    Useful for testing to see the full tree content.
    """

    def __init__(self, prefix: str = "[Summary] "):
        self.prefix = prefix

    def __call__(self, content: str, rubric: str) -> str:
        # Add a prefix to show summarization happened
        return f"{self.prefix}{content}"


class TruncatingSummarizer:
    """
    Summarizer that truncates content to a max length.
    Useful for testing with predictable output sizes.
    """

    def __init__(self, max_length: int = 100):
        self.max_length = max_length

    def __call__(self, content: str, rubric: str) -> str:
        if len(content) <= self.max_length:
            return content
        return content[:self.max_length - 3] + "..."


# =============================================================================
# Configuration and Results
# =============================================================================

@dataclass
class BuildConfig:
    """Configuration for tree building."""

    # Chunking settings
    max_chunk_chars: int = 2000
    min_chunk_chars: int = 100
    chunk_strategy: str = "sentence"  # "sentence" or "paragraph"

    # Tree settings
    merge_strategy: str = "binary"  # "binary" for 2-way merge

    # Tournament settings (used by TournamentStrategy)
    k: int = 4  # Number of candidates for tournament selection

    # Execution mode
    pipelined: bool = True  # Use pipelined execution (submit merges as dependencies complete)

    # Concurrency and cleanup
    max_concurrent_requests: int = 200
    task_cancel_timeout: float = 30.0
    document_retry_delay: float = 1.0

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


# =============================================================================
# Unified Tree Builder (async-first)
# =============================================================================

class TreeBuilder:
    """
    Unified async-first tree builder using SummarizationStrategy.

    This builder works with any strategy:
    - DSPyStrategy: For DSPy-based optimization/training
    - BatchedStrategy: For high-throughput batched inference
    - TournamentStrategy: Wraps any strategy with tournament selection

    The builder is async-first with a sync wrapper for compatibility.

    Example:
        # Async usage with batched strategy
        strategy = BatchedStrategy(client)
        builder = TreeBuilder(strategy)
        result = await builder.build(text, rubric)

        # Sync wrapper
        result = builder.build_sync(text, rubric)

        # With tournament selection (for learning)
        tournament = TournamentStrategy(base=strategy, judge=judge)
        builder = TreeBuilder(tournament)
        result = await builder.build(text, rubric)
        preferences = tournament.get_preferences()  # Free byproduct!
    """

    def __init__(
        self,
        strategy: SummarizationStrategy,
        config: Optional[BuildConfig] = None
    ):
        """
        Initialize the unified builder.

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

    async def build(self, text: str, rubric: str = "") -> BuildResult:
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
        leaves = await self._build_leaves(chunks, rubric, errors)

        if not leaves:
            raise ValueError("No leaf nodes created")

        # Build tree bottom-up (pipelined or level-wise based on config)
        if self.config.pipelined:
            tree = await self._build_tree_pipelined(leaves, rubric, errors)
        else:
            tree = await self._build_tree_from_leaves(leaves, rubric, errors)

        # Collect preferences if strategy supports it (e.g., TournamentStrategy)
        preferences = []
        if hasattr(self.strategy, 'get_preferences'):
            preferences = self.strategy.get_preferences()

        return BuildResult(
            tree=tree,
            chunks_created=len(chunks),
            nodes_created=tree.node_count,
            levels_created=tree.height + 1,
            errors=errors,
            preferences=preferences,
        )

    async def build_from_file(self, filepath: Path, rubric: str = "") -> BuildResult:
        """
        Build a tree from a file asynchronously.

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
        result = await self.build(text, rubric)

        # Add file metadata
        result.tree.metadata['source_file'] = str(filepath)

        return result

    def build_sync(self, text: str, rubric: str = "") -> BuildResult:
        """
        Synchronous wrapper for build().

        Args:
            text: Document text to process
            rubric: Information preservation criteria

        Returns:
            BuildResult containing the tree and statistics
        """
        return asyncio.run(self.build(text, rubric))

    async def _build_leaves(
        self,
        chunks: List[TextChunk],
        rubric: str,
        errors: List[str],
    ) -> List[Node]:
        """Build leaf nodes with summaries in parallel."""

        async def summarize_leaf(idx: int, text: str) -> tuple[int, Node]:
            try:
                summary = await self.strategy.summarize(text, rubric)
                self._build_stats['summarizer_calls'] += 1
                self._build_stats['total_input_chars'] += len(text)
                self._build_stats['total_output_chars'] += len(summary)
                return idx, leaf(text, summary=summary, node_id=f"leaf_{idx}")
            except Exception as e:
                errors.append(f"Failed to summarize leaf {idx}: {e}")
                # Return leaf without summary as fallback
                return idx, leaf(text, node_id=f"leaf_{idx}")

        # Create all leaf tasks
        leaf_tasks = [
            summarize_leaf(i, chunk_obj.text)
            for i, chunk_obj in enumerate(chunks)
        ]

        # Await all leaf summarizations in parallel (with cleanup on cancellation)
        results = await gather_with_cleanup(leaf_tasks, return_exceptions=True)

        # Sort by index and extract nodes
        valid_results = []
        for item in results:
            if isinstance(item, Exception):
                errors.append(f"Leaf task failed: {item}")
                continue
            if isinstance(item, tuple) and len(item) == 2:
                valid_results.append(item)

        valid_results.sort(key=lambda x: x[0])
        return [n for _, n in valid_results]

    async def _build_tree_from_leaves(
        self,
        leaves: List[Node],
        rubric: str,
        errors: List[str],
    ) -> Tree:
        """Build tree by recursively merging leaves asynchronously."""
        if len(leaves) == 1:
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
                    pairs.append((i // 2, current_level[i], current_level[i + 1]))
                else:
                    odd_node = current_level[i]

            # Create merge tasks for all pairs
            async def merge_pair(idx: int, left: Node, right: Node, level: int) -> tuple[int, Node]:
                try:
                    return idx, await self._merge_nodes(left, right, rubric, level)
                except Exception as e:
                    # Re-raise instead of silent fallback to truncated concatenation
                    # Truncated text as summary corrupts data quality silently
                    error_msg = f"Merge failed at level {level} pair {idx}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    raise

            merge_tasks = [
                merge_pair(idx, left, right, level_num)
                for idx, left, right in pairs
            ]

            # Await all merges in parallel (with cleanup on cancellation)
            results = await gather_with_cleanup(merge_tasks, return_exceptions=False)

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
        """Merge two nodes into a parent node asynchronously."""
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

    async def _build_tree_pipelined(
        self,
        leaves: List[Node],
        rubric: str,
        errors: List[str],
    ) -> Tree:
        """
        Build tree with pipelined execution - submit merges as soon as children ready.

        Unlike _build_tree_from_leaves which waits for each level to complete,
        this method submits merges as soon as their dependencies are satisfied.
        This reduces latency by allowing work to overlap.

        Pattern: Uses asyncio.wait(FIRST_COMPLETED) to process completions
        and submit newly-ready work immediately.
        """
        if len(leaves) == 1:
            return Tree(root=leaves[0], rubric=rubric)

        # Build merge dependency graph upfront
        @dataclass
        class MergeTask:
            id: int
            level: int
            left_idx: int   # Index into leaves (level 0) or merge ID (higher levels)
            right_idx: int
            left_is_merge: bool = False  # True if left_idx refers to a merge result
            right_is_merge: bool = False

        # Pre-compute all merges needed
        merges: Dict[int, MergeTask] = {}
        merge_id = 0

        # Level 0: pair up leaves
        current_refs: List[tuple[int, bool]] = [(i, False) for i in range(len(leaves))]  # (idx, is_merge)
        level = 0

        while len(current_refs) > 1:
            level += 1
            next_refs = []

            for i in range(0, len(current_refs), 2):
                if i + 1 < len(current_refs):
                    left_idx, left_is_merge = current_refs[i]
                    right_idx, right_is_merge = current_refs[i + 1]

                    merges[merge_id] = MergeTask(
                        id=merge_id,
                        level=level,
                        left_idx=left_idx,
                        right_idx=right_idx,
                        left_is_merge=left_is_merge,
                        right_is_merge=right_is_merge,
                    )
                    next_refs.append((merge_id, True))
                    merge_id += 1
                else:
                    # Odd node carries forward
                    next_refs.append(current_refs[i])

            current_refs = next_refs

        # Execute with pipelining
        completed: Dict[int, Node] = {}  # merge_id -> resulting Node
        pending: Dict[int, asyncio.Task] = {}  # merge_id -> task

        async def execute_merge(m: MergeTask) -> tuple[int, Node]:
            """Execute a single merge and return (merge_id, result_node)."""
            # Get left node
            if m.left_is_merge:
                left_node = completed[m.left_idx]
            else:
                left_node = leaves[m.left_idx]

            # Get right node
            if m.right_is_merge:
                right_node = completed[m.right_idx]
            else:
                right_node = leaves[m.right_idx]

            try:
                merged = await self._merge_nodes(left_node, right_node, rubric, m.level)
                return m.id, merged
            except Exception as e:
                error_msg = f"Pipelined merge {m.id} failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                raise

        def is_ready(m: MergeTask) -> bool:
            """Check if a merge's dependencies are satisfied."""
            if m.left_is_merge and m.left_idx not in completed:
                return False
            if m.right_is_merge and m.right_idx not in completed:
                return False
            return True

        # Submit all initially ready merges (level 1 - leaf pairs)
        for m in merges.values():
            if is_ready(m) and m.id not in pending:
                pending[m.id] = asyncio.create_task(execute_merge(m))

        if self.config.verbose:
            logger.info(f"Pipelined build: {len(merges)} total merges, "
                       f"{len(pending)} initially ready")

        try:
            while pending:
                # Wait for any task to complete
                done, _ = await asyncio.wait(
                    pending.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    merge_id_result, result_node = await task
                    completed[merge_id_result] = result_node
                    del pending[merge_id_result]

                    if self.config.verbose:
                        logger.debug(f"Merge {merge_id_result} complete, "
                                    f"{len(completed)}/{len(merges)} done")

                    # Check for newly ready merges
                    for m in merges.values():
                        if is_ready(m) and m.id not in pending and m.id not in completed:
                            pending[m.id] = asyncio.create_task(execute_merge(m))

        finally:
            # Cancel remaining tasks on exception
            if pending:
                logger.debug(f"Cleaning up {len(pending)} pending pipelined merge tasks...")
                for task in pending.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending.values(), return_exceptions=True)

        # Get final result - the last merge completed
        if completed:
            final_id = max(completed.keys())
            root = completed[final_id]
        else:
            # Shouldn't happen, but handle gracefully
            root = leaves[0]

        return Tree(root=root, rubric=rubric)

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

    def reset(self) -> None:
        """Reset all state for reuse."""
        self.reset_stats()
        # Reset tournament preferences if strategy supports it
        if hasattr(self.strategy, 'reset_preferences'):
            self.strategy.reset_preferences()


# =============================================================================
# Helper Functions
# =============================================================================

async def async_build(
    text: str,
    rubric: str,
    strategy: SummarizationStrategy,
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
    builder = TreeBuilder(strategy=strategy, config=config)
    result = await builder.build(text, rubric)
    return result.tree


def build(
    text: str,
    rubric: str = "",
    summarizer: Optional[Summarizer] = None,
    max_chars: int = 2000
) -> Tree:
    """
    Build an OPS tree from text synchronously.

    For new code, prefer using TreeBuilder with build_sync().

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

    # Create a simple async wrapper for the sync summarizer
    class SyncSummarizerAdapter:
        def __init__(self, sync_fn: Summarizer):
            self._fn = sync_fn

        async def summarize(self, content: str, rubric: str) -> str:
            return await asyncio.to_thread(self._fn, content, rubric)

        async def merge(self, left: str, right: str, rubric: str) -> str:
            combined = format_merge_input(left, right)
            return await asyncio.to_thread(self._fn, combined, rubric)

    adapter = SyncSummarizerAdapter(summarizer)
    config = BuildConfig(max_chunk_chars=max_chars)
    builder = TreeBuilder(strategy=adapter, config=config)

    return builder.build_sync(text, rubric).tree


def build_test_tree(num_leaves: int = 4) -> Tree:
    """
    Build a simple test tree with predictable structure.

    Args:
        num_leaves: Number of leaf nodes

    Returns:
        Tree with numbered nodes
    """
    # Create simple numbered content
    chunks = [
        TextChunk(
            text=f"Chunk {i} content.",
            start_char=i*20,
            end_char=(i+1)*20,
            chunk_index=i
        )
        for i in range(num_leaves)
    ]

    summarizer = ConcatenatingSummarizer()

    # Use the sync adapter pattern
    class TestAdapter:
        async def summarize(self, content: str, rubric: str) -> str:
            return summarizer(content, rubric)

        async def merge(self, left: str, right: str, rubric: str) -> str:
            combined = format_merge_input(left, right)
            return summarizer(combined, rubric)

    config = BuildConfig(verbose=False)
    builder = TreeBuilder(strategy=TestAdapter(), config=config)

    async def _build():
        return await builder.build_from_chunks(chunks, rubric="Test rubric")

    result = asyncio.run(_build())
    return result.tree

# Backwards compatibility alias
AsyncTreeBuilder = TreeBuilder
