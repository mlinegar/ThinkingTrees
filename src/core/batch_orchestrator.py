"""
Batch Tree Orchestrator - Global pipelined tree building across documents.

This module provides BatchTreeOrchestrator for processing multiple documents
with optimal batching. Unlike per-document processing, this orchestrator:

1. Pre-chunks ALL documents
2. Submits ALL leaf summaries together (one big batch)
3. Schedules merges globally as dependencies become ready
4. Continues until all trees are complete

This keeps the underlying LLM server fed with ready work across docs and levels.

Usage:
    strategy = BatchedStrategy(client)
    orchestrator = BatchTreeOrchestrator(strategy)
    results = await orchestrator.process_documents(docs, rubric)
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, TYPE_CHECKING, Deque, Tuple

if TYPE_CHECKING:
    from src.training.preference import PreferencePair

from src.core.data_models import Node, Tree, leaf, node
from src.preprocessing.chunker import TextChunk, chunk_for_ops as chunk
from src.preprocessing.visual_feedback import extract_content_weights_from_chunks
from src.core.strategy import SummarizationStrategy, TournamentStrategy, tournament_doc_id
from src.tree.builder import BuildConfig, BuildResult
from src.core.protocols import format_merge_input
from src.core.async_utils import gather_with_cleanup, cancel_tasks


logger = logging.getLogger(__name__)


@dataclass
class PlanMergeTask:
    """Merge task metadata for pre-sketching and scheduling."""
    doc_idx: int
    id: int
    level: int
    left_idx: int
    right_idx: int
    left_is_merge: bool = False
    right_is_merge: bool = False


@dataclass
class DocPlan:
    """Precomputed plan for cascading tree construction."""
    merges: Dict[int, PlanMergeTask]
    remaining_deps: Dict[int, int]
    dependents_by_leaf: Dict[int, List[int]]
    dependents_by_merge: Dict[int, List[int]]
    final_ref: Tuple[int, bool]
    max_level: int
    plan_summary: Dict[str, Any]


@dataclass
class DocumentState:
    """Tracks tree-building state for a single document during orchestration."""
    doc_id: str
    sample: Any  # Original document/sample object
    chunks: List[TextChunk] = field(default_factory=list)
    current_level: List[Node] = field(default_factory=list)
    level_num: int = 0
    error: Optional[str] = None
    leaf_failures: int = 0
    merge_failures: int = 0
    plan: Optional[DocPlan] = None


class BatchTreeOrchestrator:
    """
    Orchestrates tree building across multiple documents with global pipelined batching.

    This orchestrator maximizes throughput by pooling LLM requests across all
    documents and scheduling merges as soon as dependencies are ready:

    1. Leaf summaries for ALL documents are batched together
    2. Merges across ALL documents are submitted as they become ready
    3. The LLM server sees a continuous stream of ready work

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
            'leaf_failures': 0,
            'merge_failures': 0,
            'documents_with_failures': 0,
        }

    async def process_documents(
        self,
        documents: List[Any],
        rubric: str,
        get_text_fn: Optional[Callable[[Any], str]] = None,
        get_id_fn: Optional[Callable[[Any], str]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        max_retries: int = 0,
    ) -> List[BuildResult]:
        """
        Process multiple documents with global pipelined batching.

        If config.pipelined is False, falls back to level-wise batching.

        Args:
            documents: List of documents to process
            rubric: Information preservation criteria
            get_text_fn: Function to extract text from document (default: str(doc))
            get_id_fn: Function to extract ID from document (default: index-based)
            progress_callback: Optional callback(phase, completed, total)
            max_retries: Number of retry attempts for failed documents (default: 0)

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

        # Phase 2: Build trees (cascading or level-wise based on config)
        if self.config.pipelined:
            logger.info("Phase 2: Building trees with cascading execution...")
            await self._build_trees_cascading(states, rubric, progress_callback)
        else:
            logger.info("Phase 2: Building leaf summaries...")
            await self._build_all_leaves(states, rubric, progress_callback)
            logger.info("Phase 3: Building trees level-by-level...")
            await self._build_trees_levelwise(states, rubric, progress_callback)

        # Phase 4: Convert to BuildResults
        results = self._create_results(states, rubric)

        # Phase 5: Retry failed documents
        if max_retries > 0:
            results = await self._retry_failed_documents(
                results, documents, rubric, get_text_fn, get_id_fn,
                progress_callback, max_retries
            )

        # Log summary of any remaining failures
        self._log_failures(results, documents, get_id_fn)

        self._build_stats['documents_with_failures'] = sum(
            1 for state in states
            if state.error or state.leaf_failures > 0 or state.merge_failures > 0
        )
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
            state_idx = len(states)
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

                plan = self._create_doc_plan(state_idx, doc_id, chunks)
                states.append(DocumentState(
                    doc_id=doc_id,
                    sample=doc,
                    chunks=chunks,
                    plan=plan,
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

    def _create_doc_plan(
        self,
        state_idx: int,
        doc_id: str,
        chunks: List[TextChunk],
    ) -> DocPlan:
        """Precompute merge plan and a lightweight summary for visualization/history."""
        leaf_ids = [f"d{state_idx}_leaf_{i}" for i in range(len(chunks))]
        leaf_nodes = [
            {
                "id": leaf_ids[i],
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "char_count": len(chunk.text),
                "token_count": chunk.token_count,
            }
            for i, chunk in enumerate(chunks)
        ]

        levels: List[List[str]] = [leaf_ids.copy()] if leaf_ids else []
        edges: List[Dict[str, str]] = []
        merge_node_ids: Dict[int, str] = {}

        merges: Dict[int, PlanMergeTask] = {}
        remaining_deps: Dict[int, int] = {}
        dependents_by_leaf: Dict[int, List[int]] = {}
        dependents_by_merge: Dict[int, List[int]] = {}

        merge_id = 0
        level = 0
        current_refs: List[Tuple[int, bool]] = [(i, False) for i in range(len(chunks))]

        while len(current_refs) > 1:
            level += 1
            if len(levels) <= level:
                levels.append([])

            next_refs: List[Tuple[int, bool]] = []
            for i in range(0, len(current_refs), 2):
                if i + 1 < len(current_refs):
                    left_idx, left_is_merge = current_refs[i]
                    right_idx, right_is_merge = current_refs[i + 1]

                    merge_task = PlanMergeTask(
                        doc_idx=state_idx,
                        id=merge_id,
                        level=level,
                        left_idx=left_idx,
                        right_idx=right_idx,
                        left_is_merge=left_is_merge,
                        right_is_merge=right_is_merge,
                    )
                    merges[merge_id] = merge_task

                    merge_node_id = f"d{state_idx}_L{level}_{merge_id}"
                    merge_node_ids[merge_id] = merge_node_id
                    levels[level].append(merge_node_id)

                    edges.append({
                        "parent": merge_node_id,
                        "left": merge_node_ids[left_idx] if left_is_merge else leaf_ids[left_idx],
                        "right": merge_node_ids[right_idx] if right_is_merge else leaf_ids[right_idx],
                    })

                    remaining_deps[merge_id] = 0
                    if left_is_merge:
                        dependents_by_merge.setdefault(left_idx, []).append(merge_id)
                    else:
                        dependents_by_leaf.setdefault(left_idx, []).append(merge_id)
                    remaining_deps[merge_id] += 1

                    if right_is_merge:
                        dependents_by_merge.setdefault(right_idx, []).append(merge_id)
                    else:
                        dependents_by_leaf.setdefault(right_idx, []).append(merge_id)
                    remaining_deps[merge_id] += 1

                    next_refs.append((merge_id, True))
                    merge_id += 1
                else:
                    next_refs.append(current_refs[i])

            current_refs = next_refs

        final_ref = current_refs[0] if current_refs else (0, False)
        root_id = None
        if current_refs:
            final_idx, final_is_merge = current_refs[0]
            root_id = merge_node_ids[final_idx] if final_is_merge else leaf_ids[final_idx]

        plan_summary = {
            "doc_id": str(doc_id),
            "doc_index": state_idx,
            "leaf_count": len(leaf_ids),
            "merge_count": len(merges),
            "levels": levels,
            "edges": edges,
            "root_id": root_id,
            "leaf_nodes": leaf_nodes,
        }

        return DocPlan(
            merges=merges,
            remaining_deps=remaining_deps,
            dependents_by_leaf=dependents_by_leaf,
            dependents_by_merge=dependents_by_merge,
            final_ref=final_ref,
            max_level=level,
            plan_summary=plan_summary,
        )

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
                ), None
            except Exception as e:
                # Log and mark failure instead of silent fallback to truncated text
                # Truncated text as summary corrupts data quality silently
                logger.error(f"Leaf summarization failed for doc {state_idx} chunk {chunk_idx}: {e}")
                return state_idx, chunk_idx, None, str(e)
            finally:
                tournament_doc_id.reset(token)

        # Launch all in parallel
        tasks = [
            summarize_leaf(state_idx, chunk_idx, text, doc_id)
            for state_idx, chunk_idx, text, doc_id in leaf_tasks_info
        ]

        logger.info(f"  Submitting {len(tasks)} leaf summarization tasks...")

        results = await gather_with_cleanup(tasks, return_exceptions=True)

        # Organize results by document
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Leaf task failed with exception: {result}")
                continue

            state_idx, chunk_idx, leaf_node, error = result
            state = states[state_idx]
            if error:
                state.leaf_failures += 1
                self._build_stats['leaf_failures'] += 1
                continue

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

    async def _build_trees_cascading(
        self,
        states: List[DocumentState],
        rubric: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """
        Build trees with cascading execution across leaves and merges.

        This submits leaf summaries and merges as soon as their inputs are ready,
        allowing per-document tree construction to cascade without global barriers.
        """
        docs_to_build = [
            (idx, state)
            for idx, state in enumerate(states)
            if state.error is None and state.chunks
        ]

        if not docs_to_build:
            return

        plans: Dict[int, DocPlan] = {}
        completed_leaves: Dict[int, Dict[int, Node]] = {}
        completed_merges: Dict[int, Dict[int, Node]] = {}

        total_leaves = 0
        total_merges = 0
        max_levels = 0

        leaf_queue: Deque[Tuple[int, int, str, str]] = deque()
        ready_merges: Deque[Tuple[int, int]] = deque()
        failed_docs: set[int] = set()

        # Build dependency graphs and enqueue leaves
        for state_idx, state in docs_to_build:
            plan = state.plan or self._create_doc_plan(state_idx, state.doc_id, state.chunks)
            state.plan = plan
            plans[state_idx] = plan

            completed_leaves[state_idx] = {}
            completed_merges[state_idx] = {}

            for leaf_idx, chunk_obj in enumerate(state.chunks):
                leaf_queue.append((state_idx, leaf_idx, chunk_obj.text, state.doc_id))

            total_leaves += len(state.chunks)
            total_merges += len(plan.merges)
            max_levels = max(max_levels, plan.max_level)

        self._build_stats['total_merges'] += total_merges
        self._build_stats['total_levels'] = max(self._build_stats['total_levels'], max_levels)

        max_inflight = max(1, self.config.max_concurrent_requests)
        logger.info(
            "  Cascading build: leaves=%d merges=%d max_inflight=%d",
            total_leaves,
            total_merges,
            max_inflight,
        )

        async def summarize_leaf(
            doc_idx: int,
            leaf_idx: int,
            text: str,
            doc_id: str,
        ) -> tuple[int, int, Node, Optional[str]]:
            token = tournament_doc_id.set(str(doc_id))
            try:
                summary = await self.strategy.summarize(text, rubric)
                node_id = f"d{doc_idx}_leaf_{leaf_idx}"
                return doc_idx, leaf_idx, leaf(text, summary=summary, node_id=node_id), None
            except Exception as e:
                logger.error(f"Leaf summarization failed for doc {doc_idx} chunk {leaf_idx}: {e}")
                node_id = f"d{doc_idx}_leaf_{leaf_idx}"
                return doc_idx, leaf_idx, leaf(text, summary="", node_id=node_id), str(e)
            finally:
                tournament_doc_id.reset(token)

        async def execute_merge(doc_idx: int, merge_id: int) -> tuple[int, int, Node]:
            plan = plans[doc_idx]
            merge_task = plan.merges[merge_id]

            left = completed_merges[doc_idx][merge_task.left_idx] if merge_task.left_is_merge else completed_leaves[doc_idx][merge_task.left_idx]
            right = completed_merges[doc_idx][merge_task.right_idx] if merge_task.right_is_merge else completed_leaves[doc_idx][merge_task.right_idx]

            token = tournament_doc_id.set(str(states[doc_idx].doc_id))
            try:
                summary = await self.strategy.merge(left.summary, right.summary, rubric)
                return doc_idx, merge_id, node(
                    left=left,
                    right=right,
                    summary=summary,
                    node_id=f"d{doc_idx}_L{merge_task.level}_{merge_id}"
                )
            finally:
                tournament_doc_id.reset(token)

        pending: Dict[asyncio.Task, Tuple[str, int, int]] = {}
        completed_leaves_count = 0
        completed_merges_count = 0
        prefer_merge = True
        progress_started = time.monotonic()
        last_progress_log = progress_started
        last_progress_completed = 0

        def _maybe_log_progress(force: bool = False) -> None:
            nonlocal last_progress_log, last_progress_completed
            completed_total = completed_leaves_count + completed_merges_count
            now = time.monotonic()
            should_log = force
            if not should_log:
                if (now - last_progress_log) >= 30.0:
                    should_log = True
                elif (completed_total - last_progress_completed) >= 250:
                    should_log = True
            if not should_log:
                return

            elapsed = max(1e-6, now - progress_started)
            rate = completed_total / elapsed
            total_tasks = total_leaves + total_merges
            stats = None
            try:
                stats = getattr(getattr(self.strategy, "client", None), "stats", None)
            except Exception:
                stats = None
            stats_str = f" stats={stats}" if stats is not None else ""

            logger.info(
                "  Cascading progress: leaves=%d/%d merges=%d/%d done=%d/%d pending=%d leaf_q=%d merge_q=%d rate=%.2f items/s%s",
                completed_leaves_count,
                total_leaves,
                completed_merges_count,
                total_merges,
                completed_total,
                total_tasks,
                len(pending),
                len(leaf_queue),
                len(ready_merges),
                rate,
                stats_str,
            )
            last_progress_log = now
            last_progress_completed = completed_total

        def pump_ready_queue() -> None:
            nonlocal prefer_merge
            while len(pending) < max_inflight and (ready_merges or leaf_queue):
                choose_merge = False
                if ready_merges and leaf_queue:
                    choose_merge = prefer_merge
                elif ready_merges:
                    choose_merge = True

                if choose_merge and ready_merges:
                    doc_idx, merge_id = ready_merges.popleft()
                    if doc_idx in failed_docs:
                        continue
                    task = asyncio.create_task(execute_merge(doc_idx, merge_id))
                    pending[task] = ("merge", doc_idx, merge_id)
                    prefer_merge = False
                    continue

                if leaf_queue:
                    doc_idx, leaf_idx, text, doc_id = leaf_queue.popleft()
                    if doc_idx in failed_docs:
                        continue
                    task = asyncio.create_task(summarize_leaf(doc_idx, leaf_idx, text, doc_id))
                    pending[task] = ("leaf", doc_idx, leaf_idx)
                    prefer_merge = True

        pump_ready_queue()

        try:
            while pending:
                _maybe_log_progress()
                done, _ = await asyncio.wait(
                    pending.keys(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    kind, doc_idx, item_id = pending.pop(task)
                    if task.cancelled():
                        continue

                    if kind == "leaf":
                        try:
                            doc_idx, leaf_idx, leaf_node, error = await task
                        except Exception as e:
                            logger.error(f"Leaf task failed for doc {doc_idx} chunk {item_id}: {e}")
                            states[doc_idx].leaf_failures += 1
                            self._build_stats['leaf_failures'] += 1
                            continue

                        completed_leaves[doc_idx][leaf_idx] = leaf_node
                        completed_leaves_count += 1
                        if error:
                            states[doc_idx].leaf_failures += 1
                            self._build_stats['leaf_failures'] += 1

                        for dependent_id in plans[doc_idx].dependents_by_leaf.get(leaf_idx, []):
                            plans[doc_idx].remaining_deps[dependent_id] -= 1
                            if plans[doc_idx].remaining_deps[dependent_id] == 0:
                                ready_merges.append((doc_idx, dependent_id))

                        if progress_callback:
                            progress_callback("leaf", completed_leaves_count, total_leaves)

                    else:
                        try:
                            doc_idx, merge_id, merged_node = await task
                        except Exception as e:
                            logger.error(f"Cascading merge failed for doc {doc_idx} task {item_id}: {e}")
                            states[doc_idx].merge_failures += 1
                            self._build_stats['merge_failures'] += 1
                            failed_docs.add(doc_idx)

                            # Cancel any in-flight work for this document
                            tasks_to_cancel = [
                                t for t, (_, d_idx, _) in pending.items() if d_idx == doc_idx
                            ]
                            for t in tasks_to_cancel:
                                t.cancel()
                                pending.pop(t, None)
                            if tasks_to_cancel:
                                await cancel_tasks(tasks_to_cancel, timeout=self.config.task_cancel_timeout)

                            if leaf_queue:
                                leaf_queue = deque([item for item in leaf_queue if item[0] != doc_idx])
                            if ready_merges:
                                ready_merges = deque([item for item in ready_merges if item[0] != doc_idx])
                            continue

                        completed_merges[doc_idx][merge_id] = merged_node
                        completed_merges_count += 1

                        for dependent_id in plans[doc_idx].dependents_by_merge.get(merge_id, []):
                            plans[doc_idx].remaining_deps[dependent_id] -= 1
                            if plans[doc_idx].remaining_deps[dependent_id] == 0:
                                ready_merges.append((doc_idx, dependent_id))

                        if progress_callback:
                            progress_callback("merge", completed_merges_count, total_merges)

                pump_ready_queue()
        finally:
            if pending:
                await cancel_tasks(pending.keys(), timeout=self.config.task_cancel_timeout)

        # Finalize roots per document
        for state_idx, state in docs_to_build:
            plan = plans.get(state_idx)
            if plan is None:
                continue

            root_node: Optional[Node] = None
            if state_idx in failed_docs:
                if completed_merges[state_idx]:
                    root_node = completed_merges[state_idx][max(completed_merges[state_idx].keys())]
                elif completed_leaves[state_idx]:
                    root_node = completed_leaves[state_idx].get(0)
                    if root_node is None:
                        root_node = next(iter(completed_leaves[state_idx].values()), None)
            else:
                final_idx, final_is_merge = plan.final_ref
                if final_is_merge:
                    root_node = completed_merges[state_idx].get(final_idx)
                else:
                    root_node = completed_leaves[state_idx].get(final_idx)

            if root_node is None and completed_leaves[state_idx]:
                root_node = completed_leaves[state_idx].get(0)
                if root_node is None:
                    root_node = next(iter(completed_leaves[state_idx].values()), None)
                states[state_idx].merge_failures += 1
                self._build_stats['merge_failures'] += 1

            if root_node is not None:
                states[state_idx].current_level = [root_node]

        completed_docs = len(docs_to_build) - len(failed_docs)
        _maybe_log_progress(force=True)
        logger.info(
            f"  Cascading build complete: {completed_docs}/{len(docs_to_build)} documents"
        )

        if progress_callback:
            progress_callback("pipelined_merge", completed_docs, len(docs_to_build))

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
                    ), None
                except Exception as e:
                    # Log and mark failure instead of silent fallback to truncated concatenation
                    # Truncated text as summary corrupts data quality silently
                    logger.error(f"Merge failed for doc {state_idx} pair {pair_idx}: {e}")
                    return state_idx, pair_idx, None, str(e)
                finally:
                    tournament_doc_id.reset(token)

            # Launch all merges for this level
            tasks = [
                merge_pair(state_idx, pair_idx, left, right, level_num, doc_id)
                for state_idx, pair_idx, left, right, doc_id in merge_tasks_info
            ]

            results = await gather_with_cleanup(tasks, return_exceptions=True)

            # Organize results by document
            next_levels: Dict[int, List[tuple[int, Node]]] = {
                state_idx: [] for state_idx, _ in docs_needing_merge
            }

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Merge task failed: {result}")
                    continue

                state_idx, pair_idx, merged_node, error = result
                if error:
                    states[state_idx].merge_failures += 1
                    self._build_stats['merge_failures'] += 1
                    continue

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

    async def _build_trees_pipelined(
        self,
        states: List[DocumentState],
        rubric: str,
        progress_callback: Optional[Callable],
    ) -> None:
        """
        Build trees with global pipelined execution across all documents.

        Unlike level-wise building which synchronizes all documents at each level,
        this method submits merges as soon as their dependencies are satisfied,
        across ALL documents, to maximize throughput.
        """
        # Filter documents that need tree building
        docs_to_build = [
            (idx, state)
            for idx, state in enumerate(states)
            if state.error is None and len(state.current_level) > 1
        ]

        if not docs_to_build:
            return

        logger.info(f"  Global pipelined tree building for {len(docs_to_build)} documents...")

        @dataclass
        class MergeTask:
            doc_idx: int
            id: int
            level: int
            left_idx: int
            right_idx: int
            left_is_merge: bool = False
            right_is_merge: bool = False

        @dataclass
        class DocPlan:
            leaves: List[Node]
            merges: Dict[int, MergeTask]
            remaining_deps: Dict[int, int]
            dependents: Dict[int, List[int]]
            final_ref: Tuple[int, bool]
            max_level: int

        plans: Dict[int, DocPlan] = {}
        completed: Dict[int, Dict[int, Node]] = {}

        total_merges = 0
        max_levels = 0

        # Build dependency graphs per document
        for state_idx, state in docs_to_build:
            leaves = state.current_level
            merges: Dict[int, MergeTask] = {}
            remaining_deps: Dict[int, int] = {}
            dependents: Dict[int, List[int]] = {}

            merge_id = 0
            level = 0
            current_refs: List[Tuple[int, bool]] = [(i, False) for i in range(len(leaves))]

            while len(current_refs) > 1:
                level += 1
                next_refs: List[Tuple[int, bool]] = []
                for i in range(0, len(current_refs), 2):
                    if i + 1 < len(current_refs):
                        left_idx, left_is_merge = current_refs[i]
                        right_idx, right_is_merge = current_refs[i + 1]

                        merges[merge_id] = MergeTask(
                            doc_idx=state_idx,
                            id=merge_id,
                            level=level,
                            left_idx=left_idx,
                            right_idx=right_idx,
                            left_is_merge=left_is_merge,
                            right_is_merge=right_is_merge,
                        )

                        deps = []
                        if left_is_merge:
                            deps.append(left_idx)
                        if right_is_merge:
                            deps.append(right_idx)

                        remaining_deps[merge_id] = len(deps)
                        for dep in deps:
                            dependents.setdefault(dep, []).append(merge_id)

                        next_refs.append((merge_id, True))
                        merge_id += 1
                    else:
                        next_refs.append(current_refs[i])

                current_refs = next_refs

            final_ref = current_refs[0] if current_refs else (0, False)
            plans[state_idx] = DocPlan(
                leaves=leaves,
                merges=merges,
                remaining_deps=remaining_deps,
                dependents=dependents,
                final_ref=final_ref,
                max_level=level,
            )
            completed[state_idx] = {}

            total_merges += len(merges)
            max_levels = max(max_levels, level)

        self._build_stats['total_merges'] += total_merges
        self._build_stats['total_levels'] = max(self._build_stats['total_levels'], max_levels)

        # Scheduler state
        ready: Deque[Tuple[int, int]] = deque()
        pending: Dict[asyncio.Task, Tuple[int, int]] = {}
        failed_docs: set[int] = set()

        for state_idx, plan in plans.items():
            for merge_id, deps_remaining in plan.remaining_deps.items():
                if deps_remaining == 0:
                    ready.append((state_idx, merge_id))

        max_inflight = max(1, self.config.max_concurrent_requests)

        async def execute_merge(state_idx: int, merge_id: int) -> tuple[int, int, Node]:
            plan = plans[state_idx]
            merge_task = plan.merges[merge_id]

            left = completed[state_idx][merge_task.left_idx] if merge_task.left_is_merge else plan.leaves[merge_task.left_idx]
            right = completed[state_idx][merge_task.right_idx] if merge_task.right_is_merge else plan.leaves[merge_task.right_idx]

            token = tournament_doc_id.set(str(states[state_idx].doc_id))
            try:
                summary = await self.strategy.merge(left.summary, right.summary, rubric)
                return state_idx, merge_id, node(
                    left=left,
                    right=right,
                    summary=summary,
                    node_id=f"d{state_idx}_L{merge_task.level}_{merge_id}"
                )
            finally:
                tournament_doc_id.reset(token)

        def pump_ready_queue() -> None:
            while ready and len(pending) < max_inflight:
                state_idx, merge_id = ready.popleft()
                if state_idx in failed_docs:
                    continue
                task = asyncio.create_task(execute_merge(state_idx, merge_id))
                pending[task] = (state_idx, merge_id)

        pump_ready_queue()

        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending.keys(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    state_idx, merge_id = pending.pop(task)
                    if task.cancelled():
                        continue

                    try:
                        state_idx, merge_id, merged_node = await task
                    except Exception as e:
                        logger.error(f"Pipelined merge failed for doc {state_idx} task {merge_id}: {e}")
                        states[state_idx].merge_failures += 1
                        self._build_stats['merge_failures'] += 1
                        failed_docs.add(state_idx)

                        # Cancel any in-flight merges for this document
                        tasks_to_cancel = [
                            pending_task
                            for pending_task, (doc_idx, _) in list(pending.items())
                            if doc_idx == state_idx
                        ]
                        for pending_task in tasks_to_cancel:
                            pending_task.cancel()
                            del pending[pending_task]
                        if tasks_to_cancel:
                            await cancel_tasks(tasks_to_cancel, timeout=self.config.task_cancel_timeout)
                        continue

                    completed[state_idx][merge_id] = merged_node

                    for dependent_id in plans[state_idx].dependents.get(merge_id, []):
                        plans[state_idx].remaining_deps[dependent_id] -= 1
                        if plans[state_idx].remaining_deps[dependent_id] == 0:
                            ready.append((state_idx, dependent_id))

                pump_ready_queue()
        finally:
            if pending:
                await cancel_tasks(pending.keys(), timeout=self.config.task_cancel_timeout)

        # Finalize roots per document
        for state_idx, state in docs_to_build:
            plan = plans.get(state_idx)
            if plan is None:
                continue

            root_node: Optional[Node] = None
            if state_idx in failed_docs:
                if completed[state_idx]:
                    root_node = completed[state_idx][max(completed[state_idx].keys())]
                elif plan.leaves:
                    root_node = plan.leaves[0]
            else:
                final_idx, final_is_merge = plan.final_ref
                if final_is_merge:
                    root_node = completed[state_idx].get(final_idx)
                else:
                    root_node = plan.leaves[final_idx] if plan.leaves else None

            if root_node is None and plan.leaves:
                root_node = plan.leaves[0]
                states[state_idx].merge_failures += 1
                self._build_stats['merge_failures'] += 1

            if root_node is not None:
                states[state_idx].current_level = [root_node]

        completed_count = len(docs_to_build) - len(failed_docs)
        logger.info(
            f"  Global pipelined build complete: {completed_count}/{len(docs_to_build)} documents"
        )

        if progress_callback:
            progress_callback("pipelined_merge", completed_count, len(docs_to_build))

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
            if state.plan and state.plan.plan_summary:
                tree.metadata['tree_plan'] = state.plan.plan_summary
            if state.leaf_failures or state.merge_failures:
                tree.metadata['leaf_failures'] = state.leaf_failures
                tree.metadata['merge_failures'] = state.merge_failures

            # Filter preferences for this document (if any)
            doc_id = str(state.doc_id)
            doc_preferences = [
                p for p in preferences
                if getattr(p, "source_example_id", "") == doc_id
                or getattr(p, "source_example_id", "").startswith(f"{doc_id}:")
            ] if preferences else []

            # Extract per-leaf info scores for content-weighted audit sampling.
            content_weights = extract_content_weights_from_chunks(state.chunks)

            results.append(BuildResult(
                tree=tree,
                chunks_created=len(state.chunks),
                nodes_created=tree.node_count,
                levels_created=tree.height + 1,
                errors=[],
                preferences=doc_preferences,
                content_weights=content_weights,
            ))

        return results

    async def _retry_failed_documents(
        self,
        results: List[BuildResult],
        documents: List[Any],
        rubric: str,
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable],
        max_retries: int,
    ) -> List[BuildResult]:
        """
        Retry processing for failed documents.

        Args:
            results: Current results list (will be modified in place for successes)
            documents: Original documents
            rubric: Information preservation criteria
            get_text_fn: Function to extract text from document
            get_id_fn: Function to extract ID from document
            progress_callback: Optional progress callback
            max_retries: Number of retry attempts

        Returns:
            Updated results list with successful retries replaced
        """
        for attempt in range(1, max_retries + 1):
            # Find failed document indices
            failed_indices = [i for i, r in enumerate(results) if r.errors]

            if not failed_indices:
                logger.info("All documents processed successfully, no retries needed")
                break

            logger.info(f"Retry attempt {attempt}/{max_retries}: {len(failed_indices)} failed documents")

            # Brief delay before retry (from config)
            await asyncio.sleep(self.config.document_retry_delay)

            # Collect failed documents
            retry_docs = [documents[i] for i in failed_indices]

            # Re-chunk failed documents
            retry_states = await self._chunk_all_documents(
                retry_docs, get_text_fn, get_id_fn, None
            )

            # Build trees for retry batch
            if self.config.pipelined:
                await self._build_trees_cascading(retry_states, rubric, None)
            else:
                await self._build_all_leaves(retry_states, rubric, None)
                await self._build_trees_levelwise(retry_states, rubric, None)

            # Convert to results
            retry_results = self._create_results(retry_states, rubric)

            # Replace successful retries in original results
            successes = 0
            for orig_idx, retry_result in zip(failed_indices, retry_results):
                if not retry_result.errors:
                    results[orig_idx] = retry_result
                    successes += 1

            logger.info(f"  Retry attempt {attempt}: {successes}/{len(failed_indices)} recovered")

            if progress_callback:
                progress_callback(f"retry_{attempt}", successes, len(failed_indices))

        return results

    def _log_failures(
        self,
        results: List[BuildResult],
        documents: List[Any],
        get_id_fn: Callable[[Any], str],
    ) -> None:
        """
        Log summary of failed documents.

        Args:
            results: Processing results
            documents: Original documents
            get_id_fn: Function to extract ID from document
        """
        failed = [(i, r) for i, r in enumerate(results) if r.errors]

        if not failed:
            return

        logger.warning(f"\n{'='*50}")
        logger.warning(f"FAILED DOCUMENTS: {len(failed)}/{len(results)}")

        for idx, result in failed:
            doc_id = get_id_fn(documents[idx])
            # Get text length safely
            try:
                text_len = len(str(documents[idx]))
            except Exception:
                text_len = -1

            error = result.errors[0] if result.errors else "Unknown"
            logger.warning(f"  [{idx}] {doc_id}: {error} (len={text_len})")

        logger.warning(f"{'='*50}\n")

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
            'leaf_failures': 0,
            'merge_failures': 0,
            'documents_with_failures': 0,
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
