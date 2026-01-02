"""
Batch Tree Orchestrator - Multi-document level-wise tree building.

This module provides BatchTreeOrchestrator for processing multiple documents
with optimal batching. Unlike per-document processing, this orchestrator:

1. Pre-chunks ALL documents
2. Submits ALL leaf summaries together (one big batch)
3. Awaits ALL responses
4. For each tree level, submits ALL merges across ALL documents
5. Continues level-by-level until all trees are complete

This ensures the underlying LLM server sees maximum batch sizes at each level.

Usage:
    strategy = BatchedStrategy(client)
    orchestrator = BatchTreeOrchestrator(strategy)
    results = await orchestrator.process_documents(docs, rubric)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ops_engine.training_framework.preference import PreferencePair

from src.core.data_models import Node, Tree, leaf, node
from src.preprocessing.chunker import TextChunk, chunk_for_ops as chunk
from src.core.strategy import SummarizationStrategy, TournamentStrategy, tournament_doc_id
from src.ops_engine.builder import BuildConfig, BuildResult
from src.core.protocols import format_merge_input


logger = logging.getLogger(__name__)


@dataclass
class DocumentState:
    """Tracks tree-building state for a single document during orchestration."""
    doc_id: str
    sample: Any  # Original document/sample object
    chunks: List[TextChunk] = field(default_factory=list)
    current_level: List[Node] = field(default_factory=list)
    level_num: int = 0
    error: Optional[str] = None


class BatchTreeOrchestrator:
    """
    Orchestrates tree building across multiple documents with level-wise batching.

    This orchestrator maximizes throughput by pooling LLM requests across all
    documents at each tree level. This is more efficient than building one
    tree at a time because:

    1. Leaf summaries for ALL documents are batched together
    2. Level-N merges for ALL documents are batched together
    3. The LLM server sees larger batches = better GPU utilization

    Example:
        # Simple inference
        strategy = BatchedStrategy(client)
        orchestrator = BatchTreeOrchestrator(strategy)
        results = await orchestrator.process_documents(docs, rubric)

        # With tournament selection (learning mode)
        tournament = TournamentStrategy(base=strategy, judge=judge)
        orchestrator = BatchTreeOrchestrator(tournament)
        results = await orchestrator.process_documents(docs, rubric)
        # Get preferences from the tournament strategy
        preferences = tournament.get_preferences()
    """

    def __init__(
        self,
        strategy: SummarizationStrategy,
        config: Optional[BuildConfig] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            strategy: SummarizationStrategy for summarize/merge operations
            config: Build configuration (chunking, etc.)
        """
        self.strategy = strategy
        self.config = config or BuildConfig()
        self._build_stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_merges': 0,
            'total_levels': 0,
        }

    async def process_documents(
        self,
        documents: List[Any],
        rubric: str,
        get_text_fn: Optional[Callable[[Any], str]] = None,
        get_id_fn: Optional[Callable[[Any], str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[BuildResult]:
        """
        Process multiple documents with level-wise batching.

        Args:
            documents: List of documents to process
            rubric: Information preservation criteria
            get_text_fn: Function to extract text from document (default: str(doc))
            get_id_fn: Function to extract ID from document (default: index-based)
            progress_callback: Optional callback(phase, completed, total)

        Returns:
            List of BuildResult, one per document
        """
        # Default extractors
        if get_text_fn is None:
            get_text_fn = lambda doc: str(doc) if isinstance(doc, str) else getattr(doc, 'text', str(doc))
        if get_id_fn is None:
            get_id_fn = lambda doc: str(hash(doc))

        # Phase 1: Chunk all documents
        logger.info(f"Phase 1: Chunking {len(documents)} documents...")
        states = await self._chunk_all_documents(
            documents, get_text_fn, get_id_fn, progress_callback
        )

        # Phase 2: Build all leaf nodes (one big batch)
        logger.info("Phase 2: Building leaf summaries...")
        await self._build_all_leaves(states, rubric, progress_callback)

        # Phase 3: Build trees level-by-level
        logger.info("Phase 3: Building trees level-by-level...")
        await self._build_trees_levelwise(states, rubric, progress_callback)

        # Phase 4: Convert to BuildResults
        results = self._create_results(states, rubric)

        self._build_stats['documents_processed'] = len(documents)
        logger.info(f"Batch processing complete: {len(results)} trees built")

        return results

    async def _chunk_all_documents(
        self,
        documents: List[Any],
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable],
    ) -> List[DocumentState]:
        """Chunk all documents upfront."""
        states = []
        total_chunks = 0

        for i, doc in enumerate(documents):
            doc_id = get_id_fn(doc)
            try:
                text = get_text_fn(doc)
                if not text or len(text.strip()) == 0:
                    logger.warning(f"Document {doc_id} has no text, skipping")
                    states.append(DocumentState(
                        doc_id=doc_id,
                        sample=doc,
                        error="No text content",
                    ))
                    continue

                chunks = chunk(
                    text,
                    max_chars=self.config.max_chunk_chars,
                    strategy=self.config.chunk_strategy,
                )

                if not chunks:
                    logger.warning(f"Document {doc_id} produced no chunks, skipping")
                    states.append(DocumentState(
                        doc_id=doc_id,
                        sample=doc,
                        error="Chunking failed",
                    ))
                    continue

                states.append(DocumentState(
                    doc_id=doc_id,
                    sample=doc,
                    chunks=chunks,
                ))
                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Failed to chunk document {doc_id}: {e}")
                states.append(DocumentState(
                    doc_id=doc_id,
                    sample=doc,
                    error=str(e),
                ))

        self._build_stats['total_chunks'] = total_chunks
        logger.info(f"  Chunked {len(documents)} documents into {total_chunks} total chunks")

        if progress_callback:
            progress_callback("chunk", len(documents), len(documents))

        return states

    async def _build_all_leaves(
        self,
        states: List[DocumentState],
        rubric: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """Build leaf nodes for all documents in one big batch."""
        # Collect all leaf tasks: (state_idx, chunk_idx, chunk_text, doc_id)
        leaf_tasks_info = []
        for state_idx, state in enumerate(states):
            if state.error:
                continue
            for chunk_idx, chunk_obj in enumerate(state.chunks):
                leaf_tasks_info.append((state_idx, chunk_idx, chunk_obj.text, state.doc_id))

        if not leaf_tasks_info:
            return

        # Create coroutines for all leaf summarizations
        async def summarize_leaf(state_idx: int, chunk_idx: int, text: str, doc_id: str):
            token = tournament_doc_id.set(str(doc_id))
            try:
                summary = await self.strategy.summarize(text, rubric)
                return state_idx, chunk_idx, leaf(
                    text, summary=summary, node_id=f"d{state_idx}_leaf_{chunk_idx}"
                )
            except Exception as e:
                # Re-raise instead of silent fallback to truncated text
                # Truncated text as summary corrupts data quality silently
                logger.error(f"Leaf summarization failed for doc {state_idx} chunk {chunk_idx}: {e}")
                raise
            finally:
                tournament_doc_id.reset(token)

        # Launch all in parallel
        tasks = [
            summarize_leaf(state_idx, chunk_idx, text, doc_id)
            for state_idx, chunk_idx, text, doc_id in leaf_tasks_info
        ]

        logger.info(f"  Submitting {len(tasks)} leaf summarization tasks...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by document
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Leaf task failed with exception: {result}")
                continue

            state_idx, chunk_idx, leaf_node = result
            state = states[state_idx]

            # Ensure current_level has space
            while len(state.current_level) <= chunk_idx:
                state.current_level.append(None)
            state.current_level[chunk_idx] = leaf_node

        # Fill any gaps (shouldn't happen but be safe)
        for state in states:
            state.current_level = [n for n in state.current_level if n is not None]

        completed = len([r for r in results if not isinstance(r, Exception)])
        logger.info(f"  Completed {completed} leaf summaries")

        if progress_callback:
            progress_callback("leaf", completed, len(tasks))

    async def _build_trees_levelwise(
        self,
        states: List[DocumentState],
        rubric: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """Build trees level-by-level across all documents."""
        level_num = 0
        max_levels = 0

        # Continue until all documents have a single root
        while True:
            # Find documents that need merging (more than 1 node)
            docs_needing_merge = [
                (idx, state)
                for idx, state in enumerate(states)
                if state.error is None and len(state.current_level) > 1
            ]

            if not docs_needing_merge:
                break

            level_num += 1
            max_levels = max(max_levels, level_num)

            logger.info(f"  Level {level_num}: Merging for {len(docs_needing_merge)} documents...")

            # Collect all merge tasks for this level
            merge_tasks_info = []  # (state_idx, pair_idx, left, right, doc_id)

            for state_idx, state in docs_needing_merge:
                state.level_num = level_num

                # Pair up nodes
                for i in range(0, len(state.current_level) - 1, 2):
                    left = state.current_level[i]
                    right = state.current_level[i + 1]
                    pair_idx = i // 2
                    merge_tasks_info.append((state_idx, pair_idx, left, right, state.doc_id))

            if not merge_tasks_info:
                break

            # Create merge coroutines
            async def merge_pair(
                state_idx: int,
                pair_idx: int,
                left: Node,
                right: Node,
                level: int,
                doc_id: str,
            ):
                token = tournament_doc_id.set(str(doc_id))
                try:
                    summary = await self.strategy.merge(left.summary, right.summary, rubric)
                    return state_idx, pair_idx, node(
                        left=left,
                        right=right,
                        summary=summary,
                        node_id=f"d{state_idx}_L{level}_{pair_idx}"
                    )
                except Exception as e:
                    # Re-raise instead of silent fallback to truncated concatenation
                    # Truncated text as summary corrupts data quality silently
                    logger.error(f"Merge failed for doc {state_idx} pair {pair_idx}: {e}")
                    raise
                finally:
                    tournament_doc_id.reset(token)

            # Launch all merges for this level
            tasks = [
                merge_pair(state_idx, pair_idx, left, right, level_num, doc_id)
                for state_idx, pair_idx, left, right, doc_id in merge_tasks_info
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Organize results by document
            next_levels: Dict[int, List[tuple[int, Node]]] = {
                state_idx: [] for state_idx, _ in docs_needing_merge
            }

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Merge task failed: {result}")
                    continue

                state_idx, pair_idx, merged_node = result
                next_levels[state_idx].append((pair_idx, merged_node))

            self._build_stats['total_merges'] += len(merge_tasks_info)

            # Build next level for each document
            for state_idx, state in docs_needing_merge:
                # Get merged nodes sorted by pair index
                merged = sorted(next_levels.get(state_idx, []), key=lambda x: x[0])
                new_level = [n for _, n in merged]

                # Handle odd node (carry forward)
                if len(state.current_level) % 2 == 1:
                    new_level.append(state.current_level[-1])

                state.current_level = new_level

            completed = len([r for r in results if not isinstance(r, Exception)])
            logger.info(f"    Completed {completed} merges")

            if progress_callback:
                progress_callback(f"merge_L{level_num}", completed, len(tasks))

        self._build_stats['total_levels'] = max_levels

    def _create_results(
        self,
        states: List[DocumentState],
        rubric: str,
    ) -> List[BuildResult]:
        """Convert document states to BuildResult objects."""
        results = []

        # Collect preferences if strategy supports it
        preferences = []
        if hasattr(self.strategy, 'get_preferences'):
            preferences = self.strategy.get_preferences()

        for state in states:
            if state.error:
                # Create an empty result for failed documents
                results.append(BuildResult(
                    tree=Tree(root=leaf("", node_id="error"), rubric=rubric),
                    chunks_created=0,
                    nodes_created=0,
                    levels_created=0,
                    errors=[state.error],
                    preferences=[],
                ))
                continue

            if not state.current_level:
                results.append(BuildResult(
                    tree=Tree(root=leaf("", node_id="empty"), rubric=rubric),
                    chunks_created=len(state.chunks),
                    nodes_created=0,
                    levels_created=0,
                    errors=["No nodes created"],
                    preferences=[],
                ))
                continue

            # Get root node
            root = state.current_level[0]

            # Create tree
            tree = Tree(root=root, rubric=rubric)
            tree.metadata['doc_id'] = state.doc_id

            # Filter preferences for this document (if any)
            doc_id = str(state.doc_id)
            doc_preferences = [
                p for p in preferences
                if getattr(p, "source_example_id", "") == doc_id
                or getattr(p, "source_example_id", "").startswith(f"{doc_id}:")
            ] if preferences else []

            results.append(BuildResult(
                tree=tree,
                chunks_created=len(state.chunks),
                nodes_created=tree.node_count,
                levels_created=tree.height + 1,
                errors=[],
                preferences=doc_preferences,
            ))

        return results

    def get_stats(self) -> dict:
        """Get orchestration statistics."""
        return dict(self._build_stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._build_stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_merges': 0,
            'total_levels': 0,
        }

    def reset(self) -> None:
        """Reset all state for reuse."""
        self.reset_stats()
        # Reset tournament preferences if strategy supports it
        if hasattr(self.strategy, 'reset_preferences'):
            self.strategy.reset_preferences()


# =============================================================================
# Convenience Functions
# =============================================================================

async def batch_build_trees(
    documents: List[Any],
    strategy: SummarizationStrategy,
    rubric: str,
    get_text_fn: Optional[Callable[[Any], str]] = None,
    get_id_fn: Optional[Callable[[Any], str]] = None,
    max_chunk_chars: int = 2000,
) -> List[BuildResult]:
    """
    Build trees for multiple documents with optimal batching.

    Convenience function that creates an orchestrator and processes documents.

    Args:
        documents: List of documents
        strategy: SummarizationStrategy to use
        rubric: Information preservation criteria
        get_text_fn: Function to extract text from document
        get_id_fn: Function to extract ID from document
        max_chunk_chars: Maximum chunk size

    Returns:
        List of BuildResult, one per document
    """
    config = BuildConfig(max_chunk_chars=max_chunk_chars)
    orchestrator = BatchTreeOrchestrator(strategy=strategy, config=config)
    return await orchestrator.process_documents(
        documents=documents,
        rubric=rubric,
        get_text_fn=get_text_fn,
        get_id_fn=get_id_fn,
    )
