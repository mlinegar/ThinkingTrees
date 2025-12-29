"""
Batched Document Pipeline for High-Throughput Processing.

This module implements truly parallel processing of documents:
- Multiple documents processed concurrently
- All LLM requests pooled and batched
- Optimal vLLM GPU utilization

Example throughput comparison:
- Sequential: 1 doc/min × 100 docs = 100 minutes
- Batched (50 concurrent): ~5 minutes for same 100 docs

Usage:
    from src.pipelines.batched import BatchedDocPipeline

    pipeline = BatchedDocPipeline(config)
    results = await pipeline.process_batch_async(samples)

    # Or from sync code:
    results = pipeline.process_batch(samples)
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING

# Optional dspy import for module injection
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

if TYPE_CHECKING:
    from src.manifesto.dspy_summarizer import LeafSummarizer, MergeSummarizer

from src.core.batch_processor import (
    AsyncBatchLLMClient,
    MultiServerBatchClient,
    BatchOrchestrator,
    BatchRequest,
    LevelWiseBatchProcessor,
    DocumentTreeState,
)
from src.core.strategy import (
    SummarizationStrategy,
    DSPyStrategy,
    BatchedStrategy,
)
from src.ops_engine.builder import AsyncTreeBuilder, BuildConfig
from src.core.progress import (
    PipelineProgress,
    display_batch_summary,
)
from src.config.concurrency import ConcurrencyConfig, get_concurrency_config
from src.config import get_task_model_url, get_genrm_url
from src.core.documents import DocumentSample, DocumentResult
from src.tasks.prompting import PromptBuilders, default_merge_prompt, default_summarize_prompt
from src.preprocessing.chunker import chunk_for_ops

logger = logging.getLogger(__name__)

# Backwards compatibility aliases
DocSample = DocumentSample
DocResult = DocumentResult
ManifestoSample = DocumentSample
ManifestoResult = DocumentResult


def _extract_reference_score(sample: Any) -> Optional[float]:
    reference = getattr(sample, "reference_score", None)
    if reference is None:
        reference = getattr(sample, "score", None)
    return reference


def _extract_metadata(sample: Any) -> Dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        metadata = {}

    for key, attr in (
        ("party_name", "party_name"),
        ("country_name", "country_name"),
        ("year", "year"),
        ("party_id", "party_id"),
        ("party_abbrev", "party_abbrev"),
        ("country_code", "country_code"),
    ):
        if key not in metadata and hasattr(sample, attr):
            metadata[key] = getattr(sample, attr)

    return metadata


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BatchedPipelineConfig:
    """Configuration for batched pipeline."""

    # vLLM server settings - can be single URL or list of URLs for load balancing
    # Defaults are loaded from config/settings.yaml or environment variables
    task_model_url: str = dataclass_field(default_factory=get_task_model_url)
    task_model_urls: Optional[List[str]] = None  # Multiple servers for load balancing
    auditor_model_url: str = dataclass_field(default_factory=get_genrm_url)  # Deprecated (GenRM server)

    # Batching settings
    max_concurrent_requests: int = 200    # Max concurrent HTTP requests
    batch_size: int = 50                  # Requests per batch (independent from max_concurrent)
    max_concurrent_documents: int = 30    # Max documents in parallel (increased from 20)
    batch_timeout: float = 0.02           # Max wait to fill batch (20ms, was 50ms)

    # Tree building
    # Increased from 2000 to 4000 to reduce chunk count and tree depth
    max_chunk_chars: int = 4000
    max_tokens_summary: int = 500
    max_tokens_score: int = 200

    # Concurrency configuration (prevents thread explosion)
    concurrency: ConcurrencyConfig = dataclass_field(default_factory=get_concurrency_config)

    # Auditing (deprecated - retained for compatibility)
    audit_budget: int = 10
    rile_threshold: float = 10.0

    # Processing options
    run_baseline: bool = True
    run_audit: bool = True

    # Progress reporting
    show_progress: bool = True

    # Task configuration (to be supplied by task plugins)
    rubric: str = ""
    task_context: str = ""
    prompt_builders: Optional[PromptBuilders] = None
    score_parser: Optional[Callable[[str], Optional[float]]] = None

    def __post_init__(self):
        if self.prompt_builders is None:
            try:
                from src.config.settings import load_settings, get_default_task, get_task_config
                from src.tasks import get_task

                settings = load_settings()
                task_name = get_default_task(settings)
                task = get_task(task_name, **get_task_config(task_name, settings))

                if not self.rubric:
                    self.rubric = task.create_rubric()
                if not self.task_context:
                    self.task_context = task.get_task_context()

                self.prompt_builders = task.create_prompt_builders()
                if self.score_parser is None:
                    self.score_parser = task.parse_score
            except Exception:
                self.prompt_builders = PromptBuilders(
                    summarize=default_summarize_prompt,
                    merge=default_merge_prompt,
                    score=None,
                    audit=None,
                )

    # DSPy module support (for training/optimization mode)
    # When set, uses DSPy modules instead of raw prompts
    use_dspy_modules: bool = False

    # Level-wise batching (recommended for maximum throughput)
    # When True, processes all documents level-by-level for better batching
    use_levelwise_batching: bool = True


# =============================================================================
# DSPy-based Tree Building (for training/optimization)
# =============================================================================

def build_tree_with_dspy(
    text: str,
    doc_id: str,
    leaf_summarizer: "LeafSummarizer",
    merge_summarizer: "MergeSummarizer",
    config: BatchedPipelineConfig,
) -> Dict[str, Any]:
    """
    Build OPS tree for a document using DSPy modules.

    This function is used during DSPy optimization to allow the summarization
    prompts to be learned. Unlike build_tree_for_document, this runs
    synchronously and uses DSPy modules directly.

    Args:
        text: Full document text
        doc_id: Document identifier
        leaf_summarizer: DSPy module for leaf summarization
        merge_summarizer: DSPy module for merge summarization
        config: Pipeline config

    Returns:
        Tree structure with root summary and intermediate chunks
    """
    if not DSPY_AVAILABLE:
        raise RuntimeError("DSPy not available but DSPy modules were requested")

    # Chunk the text using langextract-based chunker
    text_chunks = chunk_for_ops(text, max_chars=config.max_chunk_chars)
    chunks = [c.text for c in text_chunks]
    if len(chunks) > 5:
        logger.info(f"[{doc_id}] Processing {len(chunks)} chunks...")

    if len(chunks) == 0:
        return {"root": {"summary": ""}, "height": 0, "leaf_count": 0, "chunks": []}

    if len(chunks) == 1:
        # Single chunk - just summarize
        summary = leaf_summarizer(content=chunks[0], rubric=config.rubric)
        return {
            "root": {"summary": summary, "content": chunks[0]},
            "height": 1,
            "leaf_count": 1,
            "chunks": chunks,
            "leaf_summaries": [summary],
        }

    # Multi-chunk: build tree level by level
    # Level 0: Summarize all chunks (concurrent for better GPU utilization)
    # IMPORTANT: Use ConcurrencyConfig to prevent thread explosion when called
    # from parent ThreadPoolExecutor (e.g., process_batch_with_dspy)
    current_level = []
    leaf_summaries = []

    def summarize_chunk(args):
        i, chunk = args
        summary = leaf_summarizer(content=chunk, rubric=config.rubric)
        return i, chunk, summary

    # Use config's concurrency settings instead of hardcoded values
    # Timeout scales with chunk count to prevent failures on large documents
    max_chunk_workers = config.concurrency.get_chunk_workers(len(chunks))
    leaf_timeout = config.concurrency.get_leaf_timeout(len(chunks))
    with ThreadPoolExecutor(max_workers=max_chunk_workers) as executor:
        futures = [executor.submit(summarize_chunk, (i, chunk)) for i, chunk in enumerate(chunks)]
        results = []
        for future in as_completed(futures, timeout=leaf_timeout):
            results.append(future.result())

    # Sort by index to maintain order
    results.sort(key=lambda x: x[0])
    for i, chunk, summary in results:
        leaf_summaries.append(summary)
        current_level.append({
            "id": f"{doc_id}_leaf_{i}",
            "content": chunk,
            "summary": summary,
            "level": 0,
        })

    # Build up the tree by merging pairs (concurrent within each level)
    level_num = 0
    while len(current_level) > 1:
        level_num += 1

        # Collect pairs for this level
        pairs = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                pairs.append((i // 2, current_level[i], current_level[i + 1]))
            else:
                pairs.append((i // 2, current_level[i], None))  # Odd node

        def merge_pair(args):
            idx, left, right = args
            if right is None:
                return idx, left  # Odd node carries forward
            left_text = left.get("summary") or left.get("content", "")
            right_text = right.get("summary") or right.get("content", "")
            merged_summary = merge_summarizer(
                left_summary=left_text,
                right_summary=right_text,
                rubric=config.rubric
            )
            return idx, {
                "id": f"{doc_id}_merge_{level_num}_{idx}",
                "summary": merged_summary,
                "level": level_num,
                "children": [left, right],
            }

        # Use config's concurrency settings instead of hardcoded values
        # Timeout scales with number of pairs to prevent failures
        max_merge_workers = config.concurrency.get_merge_workers(len(pairs))
        merge_timeout = config.concurrency.get_merge_timeout(len(pairs))
        with ThreadPoolExecutor(max_workers=max_merge_workers) as executor:
            futures = [executor.submit(merge_pair, pair) for pair in pairs]
            merge_results = []
            for future in as_completed(futures, timeout=merge_timeout):
                merge_results.append(future.result())

        # Sort by index to maintain order
        merge_results.sort(key=lambda x: x[0])
        current_level = [node for _, node in merge_results]

    root = current_level[0] if current_level else {"summary": ""}
    return {
        "root": root,
        "height": level_num,
        "leaf_count": len(chunks),
        "chunks": chunks,
        "leaf_summaries": leaf_summaries,
    }


# =============================================================================
# Tree Building (Batched)
# =============================================================================

async def build_tree_for_document(
    text: str,
    doc_id: str,
    client: AsyncBatchLLMClient,
    config: BatchedPipelineConfig,
) -> Dict[str, Any]:
    """
    Build OPS tree for a single document using batched requests.

    Args:
        text: Full document text
        doc_id: Document identifier
        client: Batch LLM client
        config: Pipeline config

    Returns:
        Tree structure with root summary
    """
    # Chunk the text using langextract-based chunker
    text_chunks = chunk_for_ops(text, max_chars=config.max_chunk_chars)
    chunks = [c.text for c in text_chunks]
    logger.debug(f"[{doc_id}] Chunked into {len(chunks)} chunks")

    if len(chunks) == 0:
        return {"root": {"summary": ""}, "height": 0, "leaf_count": 0}

    if len(chunks) == 1:
        # Single chunk - just summarize
        request = BatchRequest(
            request_id=f"{doc_id}_single",
            messages=config.prompt_builders.summarize(chunks[0], config.rubric),
            max_tokens=config.max_tokens_summary,
            document_id=doc_id,
            request_type="summarize",
        )
        await client.submit(request)
        response = await client.await_response(request.request_id)
        return {
            "root": {"summary": response.content, "content": chunks[0]},
            "height": 1,
            "leaf_count": 1,
        }

    # Multi-chunk: build tree level by level
    # Level 0: Summarize all chunks
    current_level = []
    leaf_requests = []

    for i, chunk in enumerate(chunks):
        request = BatchRequest(
            request_id=f"{doc_id}_leaf_{i}",
            messages=config.prompt_builders.summarize(chunk, config.rubric),
            max_tokens=config.max_tokens_summary,
            document_id=doc_id,
            request_type="summarize",
        )
        leaf_requests.append((request, chunk))

    # Submit all leaf summarizations
    for request, _ in leaf_requests:
        await client.submit(request)

    # Await all leaf responses
    for i, (request, chunk) in enumerate(leaf_requests):
        response = await client.await_response(request.request_id)
        current_level.append({
            "id": request.request_id,
            "content": chunk,
            "summary": response.content if not response.error else chunk[:500],
            "level": 0,
        })

    # Build up the tree
    level_num = 0
    while len(current_level) > 1:
        level_num += 1
        next_level = []
        merge_requests = []

        # Pair up nodes
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                left = current_level[i]
                right = current_level[i + 1]

                left_text = left.get("summary") or left.get("content", "")
                right_text = right.get("summary") or right.get("content", "")

                request = BatchRequest(
                    request_id=f"{doc_id}_merge_{level_num}_{len(merge_requests)}",
                    messages=config.prompt_builders.merge(left_text, right_text, config.rubric),
                    max_tokens=config.max_tokens_summary,
                    document_id=doc_id,
                    request_type="merge",
                )
                merge_requests.append((request, left, right))
            else:
                # Odd node carries forward
                next_level.append(current_level[i])

        # Submit all merges for this level
        for request, _, _ in merge_requests:
            await client.submit(request)

        # Await all merges
        for j, (request, left, right) in enumerate(merge_requests):
            response = await client.await_response(request.request_id)
            next_level.append({
                "id": request.request_id,
                "summary": response.content if not response.error else "",
                "level": level_num,
                "children": [left, right],
            })

        current_level = next_level

    root = current_level[0] if current_level else {"summary": ""}
    return {
        "root": root,
        "height": level_num,
        "leaf_count": len(chunks),
    }


# =============================================================================
# RILE Scoring (Batched)
# =============================================================================

# =============================================================================
# Batched Pipeline
# =============================================================================

class BatchedDocPipeline:
    """
    High-throughput batched pipeline for document processing.

    Processes multiple documents concurrently, pooling all LLM requests
    for optimal GPU utilization.

    For DSPy optimization, use with leaf_summarizer and merge_summarizer:
        from src.manifesto.dspy_summarizer import LeafSummarizer, MergeSummarizer

        pipeline = BatchedDocPipeline(
            config=config,
            leaf_summarizer=LeafSummarizer(),
            merge_summarizer=MergeSummarizer(),
        )
        # Use process_with_dspy for training
        result = pipeline.process_with_dspy(sample)
    """

    def __init__(
        self,
        config: Optional[BatchedPipelineConfig] = None,
        leaf_summarizer: Optional["LeafSummarizer"] = None,
        merge_summarizer: Optional["MergeSummarizer"] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            leaf_summarizer: Optional DSPy module for leaf summarization (training mode)
            merge_summarizer: Optional DSPy module for merge summarization (training mode)
        """
        self.config = config or BatchedPipelineConfig()
        self._results: List[DocResult] = []

        # DSPy modules for training/optimization mode
        self.leaf_summarizer = leaf_summarizer
        self.merge_summarizer = merge_summarizer

    def process_with_dspy(
        self,
        sample: DocSample,
    ) -> DocResult:
        """
        Process a single document using DSPy modules.

        This method is used during DSPy optimization to allow the summarization
        prompts to be learned. Unlike async methods, this runs synchronously.

        Args:
            sample: Document sample to process

        Returns:
            DocResult with tree info and summaries
        """
        if self.leaf_summarizer is None or self.merge_summarizer is None:
            raise ValueError(
                "DSPy processing requires leaf_summarizer and merge_summarizer. "
                "Initialize pipeline with these modules for training mode."
            )

        doc_id = sample.doc_id
        start_time = time.time()
        logger.debug(f"Starting {doc_id} ({len(sample.text)} chars)")

        result = DocResult(
            doc_id=doc_id,
            reference_score=_extract_reference_score(sample),
            original_length=len(sample.text),
            metadata=_extract_metadata(sample),
        )

        try:
            # Build tree using DSPy modules
            tree = build_tree_with_dspy(
                text=sample.text,
                doc_id=doc_id,
                leaf_summarizer=self.leaf_summarizer,
                merge_summarizer=self.merge_summarizer,
                config=self.config,
            )

            result.tree_height = tree["height"]
            result.tree_leaves = tree["leaf_count"]
            result.final_summary = tree["root"].get("summary", "")
            result.summary_length = len(result.final_summary)
            result.compression_ratio = (
                result.original_length / max(result.summary_length, 1)
            )

            # Store additional tree info for training
            result.chunks = tree.get("chunks", [])
            result.leaf_summaries = tree.get("leaf_summaries", [])

        except Exception as e:
            logger.error(f"Error processing {doc_id} with DSPy: {e}")
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def process_batch_with_dspy(
        self,
        samples: List[DocSample],
        show_progress: bool = True,
    ) -> List[DocResult]:
        """
        Process multiple documents using DSPy modules (concurrent).

        Uses ThreadPoolExecutor to process multiple documents in parallel,
        matching the concurrency of the async batch processing.

        Args:
            samples: List of document samples
            show_progress: Whether to show progress

        Returns:
            List of DocResult
        """
        max_workers = self.config.max_concurrent_documents
        results = [None] * len(samples)
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.process_with_dspy, sample): i
                for i, sample in enumerate(samples)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                sample = samples[idx]
                try:
                    result = future.result(timeout=600)  # 10 min timeout per doc
                    results[idx] = result
                    self._results.append(result)
                    completed += 1
                    if show_progress:
                        logger.info(f"Completed {completed}/{len(samples)}: {sample.doc_id}")
                except TimeoutError:
                    failed += 1
                    logger.error(f"Timeout processing {sample.doc_id} (failed {failed}/{len(samples)})")
                    # Create error result
                    results[idx] = DocResult(
                        doc_id=sample.doc_id,
                        error="Timeout after 600s"
                    )
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing {sample.doc_id}: {e} (failed {failed}/{len(samples)})")
                    results[idx] = DocResult(
                        doc_id=sample.doc_id,
                        error=str(e)
                    )

        return results

    async def process_with_strategy(
        self,
        sample: DocSample,
        strategy: SummarizationStrategy,
    ) -> DocResult:
        """
        Process a single document using a SummarizationStrategy.

        This method uses the unified AsyncTreeBuilder with any strategy
        (DSPy or batched). Use this for new code that wants to leverage
        the Strategy pattern.

        Args:
            sample: Document sample to process
            strategy: SummarizationStrategy to use for summarization

        Returns:
            DocResult with tree info and summaries
        """
        doc_id = sample.doc_id
        start_time = time.time()

        result = DocResult(
            doc_id=doc_id,
            reference_score=_extract_reference_score(sample),
            original_length=len(sample.text),
            metadata=_extract_metadata(sample),
        )

        try:
            # Build tree using Strategy-based AsyncTreeBuilder
            build_config = BuildConfig(
                max_chunk_chars=self.config.max_chunk_chars,
            )
            builder = AsyncTreeBuilder(strategy=strategy, config=build_config)
            build_result = await builder.build_from_text(sample.text, self.config.rubric)

            result.tree_height = build_result.tree.height
            result.tree_leaves = build_result.tree.leaf_count
            result.final_summary = build_result.tree.final_summary
            result.summary_length = len(result.final_summary)
            result.compression_ratio = (
                result.original_length / max(result.summary_length, 1)
            )

        except Exception as e:
            logger.error(f"Error processing {doc_id} with strategy: {e}")
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    async def process_batch_with_strategy(
        self,
        samples: List[DocSample],
        strategy: SummarizationStrategy,
        show_progress: bool = True,
    ) -> List[DocResult]:
        """
        Process multiple documents using a SummarizationStrategy.

        This method processes documents concurrently using asyncio.gather
        with the unified AsyncTreeBuilder and Strategy pattern.

        Args:
            samples: List of document samples
            strategy: SummarizationStrategy to use
            show_progress: Whether to show progress

        Returns:
            List of DocResult
        """
        logger.info(f"Processing {len(samples)} documents with strategy")

        async def process_one(idx: int, sample: DocSample):
            result = await self.process_with_strategy(sample, strategy)
            if show_progress:
                logger.info(f"Completed {idx + 1}/{len(samples)}: {sample.doc_id}")
            return idx, result

        tasks = [process_one(i, sample) for i, sample in enumerate(samples)]

        # Process with limited concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)

        async def limited_task(task):
            async with semaphore:
                return await task

        results_with_idx = await asyncio.gather(
            *[limited_task(task) for task in tasks],
            return_exceptions=True
        )

        # Sort by index and extract results
        results = [None] * len(samples)
        for item in results_with_idx:
            if isinstance(item, tuple):
                idx, result = item
                results[idx] = result
                self._results.append(result)
            elif isinstance(item, Exception):
                logger.error(f"Task failed: {item}")

        return [r for r in results if r is not None]

    async def process_single_async(
        self,
        sample: DocSample,
        client: AsyncBatchLLMClient,
    ) -> DocResult:
        """
        Process a single document (used internally by batch processor).

        Args:
            sample: Document sample
            client: Batch LLM client

        Returns:
            DocResult
        """
        doc_id = sample.doc_id
        start_time = time.time()

        result = DocResult(
            doc_id=doc_id,
            reference_score=_extract_reference_score(sample),
            original_length=len(sample.text),
            metadata=_extract_metadata(sample),
        )

        try:
            # 1. Build tree
            tree = await build_tree_for_document(
                sample.text, doc_id, client, self.config
            )

            result.tree_height = tree["height"]
            result.tree_leaves = tree["leaf_count"]
            result.final_summary = tree["root"].get("summary", "")
            result.summary_length = len(result.final_summary)
            result.compression_ratio = (
                result.original_length / max(result.summary_length, 1)
            )

            # 2. Score summary if scoring is configured
            if self.config.prompt_builders.score and self.config.score_parser:
                score_request = BatchRequest(
                    request_id=f"{doc_id}_score",
                    messages=self.config.prompt_builders.score(
                        result.final_summary,
                        self.config.task_context,
                    ),
                    max_tokens=self.config.max_tokens_score,
                    document_id=doc_id,
                    request_type="score",
                )
                await client.submit(score_request)
                score_response = await client.await_response(score_request.request_id)

                result.estimated_score = self.config.score_parser(score_response.content)
                result.reasoning = score_response.content or ""

                # 3. Baseline score (optional)
                if self.config.run_baseline:
                    baseline_text = sample.text[:50000]
                    baseline_request = BatchRequest(
                        request_id=f"{doc_id}_baseline",
                        messages=self.config.prompt_builders.score(
                            baseline_text,
                            self.config.task_context,
                        ),
                        max_tokens=self.config.max_tokens_score,
                        document_id=doc_id,
                        request_type="baseline",
                    )
                    await client.submit(baseline_request)
                    baseline_response = await client.await_response(baseline_request.request_id)
                    result.baseline_score = self.config.score_parser(baseline_response.content)

        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            result.error = str(e)

        return result

    async def process_batch_async(
        self,
        samples: List[DocSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[DocResult]:
        """
        Process multiple documents with full batching.

        By default uses level-wise batching for maximum throughput.
        Set config.use_levelwise_batching=False for per-document processing.

        Args:
            samples: List of document samples
            progress_callback: Optional progress callback(completed, total)
            show_progress: Override config.show_progress setting

        Returns:
            List of DocResults
        """
        # Use level-wise batching by default for better throughput
        if self.config.use_levelwise_batching:
            # Convert simple callback to phase-aware callback
            phase_callback = None
            if progress_callback:
                def phase_callback(phase: str, completed: int, total: int):
                    # For compatibility, only call on document-level phases
                    if phase == "chunk":
                        progress_callback(completed, total)

            return await self.process_batch_levelwise_async(
                samples, phase_callback, show_progress
            )

        # Legacy per-document processing
        logger.info(f"Starting batched processing of {len(samples)} documents (per-document mode)")
        start_time = time.time()

        # Determine whether to show progress
        use_progress = show_progress if show_progress is not None else self.config.show_progress

        # Use multi-server client if multiple URLs provided
        server_urls = self.config.task_model_urls or [self.config.task_model_url]

        if len(server_urls) > 1:
            logger.info(f"Using {len(server_urls)} servers for load balancing")
            client = MultiServerBatchClient(
                servers=server_urls,
                max_concurrent_per_server=self.config.max_concurrent_requests,
                batch_size=self.config.batch_size,
                batch_timeout=self.config.batch_timeout,
            )
        else:
            client = AsyncBatchLLMClient(
                base_url=server_urls[0],
                max_concurrent=self.config.max_concurrent_requests,
                batch_size=self.config.batch_size,
                batch_timeout=self.config.batch_timeout,
            )

        async with client:
            orchestrator = BatchOrchestrator(
                client=client,
                max_concurrent_documents=self.config.max_concurrent_documents,
            )

            # Set up progress tracking
            with PipelineProgress(disable=not use_progress) as progress:
                # Start document processing phase
                progress.start_phase(
                    "documents",
                    total=len(samples),
                    description="Processing documents"
                )

                # Create callback that updates rich progress
                def combined_callback(completed: int, total: int):
                    # Update rich progress bar with live token stats
                    progress.update(
                        "documents",
                        advance=1,
                        stats=client.stats,
                    )
                    # Also call user's callback if provided
                    if progress_callback:
                        progress_callback(completed, total)

                results = await orchestrator.process_documents(
                    documents=samples,
                    process_fn=lambda sample, c: self.process_single_async(sample, c),
                    progress_callback=combined_callback,
                )

                # Mark phase complete
                progress.complete_phase("documents")

            # Filter out None results (failures)
            results = [r for r in results if r is not None]

            elapsed = time.time() - start_time
            logger.info(
                f"Batched processing complete: {len(results)}/{len(samples)} succeeded "
                f"in {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)"
            )

            # Display summary table
            if use_progress:
                display_batch_summary(client.stats, title="Batch Processing Summary")
            else:
                logger.info(f"LLM stats: {client.stats}")

            self._results.extend(results)
            return results

    def process_batch(
        self,
        samples: List[DocSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[DocResult]:
        """
        Sync wrapper for batch processing.

        Args:
            samples: List of manifesto samples
            progress_callback: Optional progress callback
            show_progress: Override config.show_progress setting

        Returns:
            List of DocResults
        """
        return asyncio.run(self.process_batch_async(samples, progress_callback, show_progress))

    def get_results(self) -> List[DocResult]:
        """Get all processed results."""
        return self._results

    async def process_batch_levelwise_async(
        self,
        samples: List[DocSample],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[DocResult]:
        """
        Process documents using level-wise batching for maximum throughput.

        This method processes ALL documents level-by-level:
        1. Chunks ALL documents
        2. Submits ALL leaf summaries together
        3. Awaits ALL responses
        4. Submits ALL level-1 merges together
        5. Continues until all trees are complete
        6. Scores ALL documents together

        This ensures vLLM sees the largest possible batches at each tree level.

        Args:
            samples: List of document samples
            progress_callback: Optional callback(phase, completed, total)
            show_progress: Override config.show_progress setting

        Returns:
            List of DocResults
        """
        logger.info(f"Starting LEVEL-WISE batched processing of {len(samples)} documents")
        start_time = time.time()

        use_progress = show_progress if show_progress is not None else self.config.show_progress
        server_urls = self.config.task_model_urls or [self.config.task_model_url]

        if len(server_urls) > 1:
            logger.info(f"Using {len(server_urls)} servers for load balancing")
            client = MultiServerBatchClient(
                servers=server_urls,
                max_concurrent_per_server=self.config.max_concurrent_requests,
                batch_size=self.config.batch_size,
                batch_timeout=self.config.batch_timeout,
            )
        else:
            client = AsyncBatchLLMClient(
                base_url=server_urls[0],
                max_concurrent=self.config.max_concurrent_requests,
                batch_size=self.config.batch_size,
                batch_timeout=self.config.batch_timeout,
            )

        results = []

        async with client:
            # Phase 1: Build all trees level-wise
            level_processor = LevelWiseBatchProcessor(
                client=client,
                rubric=self.config.rubric,
                max_chunk_chars=self.config.max_chunk_chars,
                max_tokens_summary=self.config.max_tokens_summary,
                summarize_prompt_fn=self.config.prompt_builders.summarize,
                merge_prompt_fn=self.config.prompt_builders.merge,
            )

            tree_states = await level_processor.process_all_documents(
                documents=samples,
                get_text_fn=lambda s: s.text,
                get_id_fn=lambda s: getattr(s, "doc_id", getattr(s, "manifesto_id", "")),
                progress_callback=progress_callback,
            )

            scores = {}
            baselines = {}

            if self.config.prompt_builders.score and self.config.score_parser:
                # Phase 2: Score ALL documents' summaries together
                logger.info(f"Phase 4: Scoring {len(tree_states)} documents...")
                score_requests = []  # [(state_idx, request)]

                for state_idx, state in enumerate(tree_states):
                    if state.error or not state.root_summary:
                        continue

                    request = BatchRequest(
                        request_id=f"{state.doc_id}_score",
                        messages=self.config.prompt_builders.score(
                            state.root_summary,
                            self.config.task_context,
                        ),
                        max_tokens=self.config.max_tokens_score,
                        document_id=state.doc_id,
                        request_type="score",
                    )
                    score_requests.append((state_idx, request))
                    await client.submit(request)

                logger.info(f"  Submitted {len(score_requests)} score requests...")

                # Await all scores
                for state_idx, request in score_requests:
                    response = await client.await_response(request.request_id)
                    scores[state_idx] = self.config.score_parser(response.content)

                # Phase 3: Baseline scores (optional)
                if self.config.run_baseline:
                    logger.info(f"Phase 5: Computing baseline scores...")
                    baseline_requests = []

                    for state_idx, state in enumerate(tree_states):
                        if state.error:
                            continue

                        sample = state.sample
                        baseline_text = sample.text[:50000]

                        request = BatchRequest(
                            request_id=f"{state.doc_id}_baseline",
                            messages=self.config.prompt_builders.score(
                                baseline_text,
                                self.config.task_context,
                            ),
                            max_tokens=self.config.max_tokens_score,
                            document_id=state.doc_id,
                            request_type="baseline",
                        )
                        baseline_requests.append((state_idx, request))
                        await client.submit(request)

                    logger.info(f"  Submitted {len(baseline_requests)} baseline requests...")

                    for state_idx, request in baseline_requests:
                        response = await client.await_response(request.request_id)
                        baselines[state_idx] = self.config.score_parser(response.content)

            # Convert tree states to DocResults
            for state_idx, state in enumerate(tree_states):
                sample = state.sample

                # Extract leaf summaries from level_history if available
                leaf_summaries = []
                if state.level_history and len(state.level_history) > 0:
                    # Level 0 contains the leaf nodes with their summaries
                    leaf_summaries = [
                        node.get('summary', '')
                        for node in state.level_history[0]
                    ]

                result = DocResult(
                    doc_id=state.doc_id,
                    reference_score=_extract_reference_score(sample),
                    original_length=len(sample.text),
                    tree_height=state.tree_height,
                    tree_leaves=state.leaf_count,
                    final_summary=state.root_summary,
                    summary_length=len(state.root_summary),
                    compression_ratio=len(sample.text) / max(len(state.root_summary), 1),
                    estimated_score=scores.get(state_idx),
                    baseline_score=baselines.get(state_idx),
                    error=state.error,
                    chunks=state.chunks,
                    leaf_summaries=leaf_summaries,
                    level_history=state.level_history,
                    metadata=_extract_metadata(sample),
                )
                results.append(result)

            elapsed = time.time() - start_time
            logger.info(
                f"Level-wise processing complete: {len(results)}/{len(samples)} in "
                f"{elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)"
            )

            if use_progress:
                display_batch_summary(client.stats, title="Level-Wise Batch Summary")
            else:
                logger.info(f"LLM stats: {client.stats}")

            self._results.extend(results)
            return results


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_documents_batched(
    samples: List[DocSample],
    config: Optional[BatchedPipelineConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[DocResult]:
    """
    High-level async function to process documents with batching.

    Args:
        samples: Document samples to process
        config: Pipeline configuration
        progress_callback: Progress callback

    Returns:
        List of DocResults
    """
    pipeline = BatchedDocPipeline(config)
    return await pipeline.process_batch_async(samples, progress_callback)


def run_batched_experiment(
    samples: List[DocSample],
    config: Optional[BatchedPipelineConfig] = None,
) -> List[DocResult]:
    """
    Sync convenience function for running batched experiments.

    Args:
        samples: Document samples
        config: Configuration

    Returns:
        Results
    """
    pipeline = BatchedDocPipeline(config)
    return pipeline.process_batch(samples)


# Backwards compatibility alias
BatchedManifestoPipeline = BatchedDocPipeline
process_manifestos_batched = process_documents_batched
