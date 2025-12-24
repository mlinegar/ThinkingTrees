"""
OPS Tree Builder - Constructs summarization trees from document chunks.

The builder creates trees bottom-up, starting from leaf nodes (text chunks)
and recursively summarizing pairs of nodes until a single root remains.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Protocol, Any, Dict
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


class PreferenceScorer(Protocol):
    """Protocol for ranking candidate summaries."""

    def score(self, candidates: List[str], rubric: str = "", context: Optional[str] = None) -> List[float]:
        """
        Score candidate summaries.

        Args:
            candidates: Summaries to score
            rubric: Information preservation criteria
            context: Optional source text used to generate candidates

        Returns:
            A list of scores aligned with the candidates list.
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
    teacher_guided: bool = False
    student_only: bool = True

    # Debug settings
    verbose: bool = False


@dataclass
class BuildResult:
    """Result of tree building operation."""
    tree: Optional[OPSTree]
    chunks_created: int
    nodes_created: int
    levels_created: int
    errors: List["BuildError"] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Return True if the build encountered fatal errors."""
        return self.tree is None


@dataclass
class BuildError:
    """Structured error information collected during tree building."""

    stage: str
    message: str
    exception_type: Optional[str] = None
    details: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class MergeError(Exception):
    """Raised when a merge operation fails before fallback handling."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


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
        config: Optional[BuildConfig] = None,
        preference_scorer: Optional[PreferenceScorer] = None,
    ):
        """
        Initialize the builder.

        Args:
            summarizer: Function to summarize text
            config: Build configuration
            preference_scorer: Optional scorer used to select among candidate summaries
        """
        self.summarizer = summarizer
        self.config = config or BuildConfig()
        self.preference_scorer = preference_scorer
        self._build_stats = {
            'summarizer_calls': 0,
            'total_input_chars': 0,
            'total_output_chars': 0,
            'preference_calls': 0,
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

        errors: List[BuildError] = []

        # Create leaf nodes
        leaves = []
        for i, chunk in enumerate(chunks):
            try:
                leaf = create_leaf_node(chunk.text, node_id=f"leaf_{i}")
                leaf.audit_result.trace = {'chunk_index': i}
                leaves.append(leaf)
            except Exception as e:
                logger.exception("Failed to create leaf %s", i)
                errors.append(
                    BuildError(
                        stage="leaf_creation",
                        message="Failed to create leaf",
                        exception_type=type(e).__name__,
                        details=str(e),
                        context={"chunk_index": i}
                    )
                )

        if not leaves:
            raise ValueError("No leaf nodes created")

        # Build tree bottom-up
        try:
            tree = self._build_tree_from_leaves(leaves, rubric, errors)
        except Exception as e:
            logger.exception("Failed to build tree from leaves")
            errors.append(
                BuildError(
                    stage="tree_build",
                    message="Unhandled error during tree construction",
                    exception_type=type(e).__name__,
                    details=str(e),
                    context={"leaves": len(leaves)}
                )
            )
            tree = None

        # Calculate statistics
        total_nodes = tree.node_count if tree else len(leaves)
        levels_created = tree.height + 1 if tree else 0

        return BuildResult(
            tree=tree,
            chunks_created=len(chunks),
            nodes_created=total_nodes,
            levels_created=levels_created,
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
        if result.tree is not None:
            result.tree.metadata['source_file'] = str(filepath)
        else:
            logger.warning("Tree build failed for %s; returning partial result", filepath)

        return result

    def _build_tree_from_leaves(
        self,
        leaves: List[OPSNode],
        rubric: str,
        errors: Optional[List[BuildError]] = None
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
                            left, right = pairs[idx]
                            logger.exception(
                                "Merge failed for pair %s at level %s", idx, level_num
                            )
                            error_stage = "summarizer" if isinstance(e, MergeError) else "merge"
                            error_entry = BuildError(
                                stage=error_stage,
                                message="Failed to merge node pair",
                                exception_type=type(e).__name__,
                                details=str(e),
                                context={
                                    "pair_index": idx,
                                    "level": level_num,
                                    "left_id": left.id,
                                    "right_id": right.id,
                                },
                            )
                            if isinstance(e, MergeError):
                                error_entry.context.update(getattr(e, "context", {}))
                            if errors is not None:
                                errors.append(error_entry)

                            fallback_content = self._fallback_summary(left, right)
                            results[idx] = create_internal_node(
                                left=left,
                                right=right,
                                summary=fallback_content,
                                node_id=f"fallback_L{level_num}_{idx}",
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

    def _generate_candidate_summaries(self, combined: str, rubric: str) -> List[str]:
        """Generate one or more candidate summaries from combined text."""
        result = self.summarizer(combined, rubric)

        if isinstance(result, list):
            return [str(candidate) for candidate in result if candidate]

        return [str(result)]

    def _select_summary(
        self,
        candidates: List[str],
        rubric: str,
        context: str,
    ) -> str:
        """Select a summary using the preference scorer when configured."""
        if not candidates:
            raise ValueError("No candidate summaries provided for selection")

        if (
            self.preference_scorer is not None
            and self.config.teacher_guided
            and not self.config.student_only
        ):
            try:
                scores = self.preference_scorer.score(
                    candidates=candidates,
                    rubric=rubric,
                    context=context,
                )
            except TypeError:
                # Backward compatibility for positional-only scorers
                scores = self.preference_scorer.score(candidates, rubric, context)

            if scores:
                self._build_stats['preference_calls'] += 1
                best_idx = max(range(len(candidates)), key=lambda i: scores[i])
                return candidates[best_idx]

        return candidates[0]
    @staticmethod
    def _node_text_for_fallback(node: OPSNode) -> str:
        """Return the best available text for a node."""
        if node.summary:
            return node.summary
        if node.raw_text_span:
            return node.raw_text_span
        return ""

    def _fallback_summary(self, left: OPSNode, right: OPSNode) -> str:
        """Construct a safe fallback summary from two child nodes."""
        left_text = self._node_text_for_fallback(left)
        right_text = self._node_text_for_fallback(right)
        combined = f"{left_text}\n\n{right_text}".strip()
        # Avoid overly long summaries if raw text is large
        return combined[:5000] if combined else "[Fallback merge produced empty summary]"

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

        candidates = self._generate_candidate_summaries(combined, rubric)
        summary = self._select_summary(candidates, rubric, combined)
        try:
            summary = self.summarizer(combined, rubric)
        except Exception as e:
            logger.exception(
                "Summarizer failed while merging nodes %s and %s at level %s",
                left.id,
                right.id,
                level,
            )
            raise MergeError(
                "Summarizer failed during merge",
                context={
                    "level": level,
                    "left_id": left.id,
                    "right_id": right.id,
                },
            ) from e

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
            'total_output_chars': 0,
            'preference_calls': 0,
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
