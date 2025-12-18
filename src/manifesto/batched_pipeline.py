"""
Batched Manifesto Pipeline for High-Throughput Processing.

This module implements truly parallel processing of manifestos:
- Multiple documents processed concurrently
- All LLM requests pooled and batched
- Optimal vLLM GPU utilization

Example throughput comparison:
- Sequential: 1 doc/min × 100 docs = 100 minutes
- Batched (50 concurrent): ~5 minutes for same 100 docs

Usage:
    from src.manifesto.batched_pipeline import BatchedManifestoPipeline

    pipeline = BatchedManifestoPipeline(config)
    results = await pipeline.process_batch_async(samples)

    # Or from sync code:
    results = pipeline.process_batch(samples)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING
from pathlib import Path

# Optional dspy import for module injection
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

if TYPE_CHECKING:
    from .dspy_summarizer import LeafSummarizer, MergeSummarizer

from src.core.batch_processor import (
    AsyncBatchLLMClient,
    MultiServerBatchClient,
    BatchOrchestrator,
    BatchRequest,
    BatchResponse,
    BatchStats,
    process_samples_batched,
    LevelWiseBatchProcessor,
    DocumentTreeState,
)
from src.core.progress import (
    PipelineProgress,
    display_batch_summary,
    create_progress_callback,
)

from .data_loader import ManifestoSample
from .ops_pipeline import ManifestoResult, PipelineConfig
from .rubrics import RILE_PRESERVATION_RUBRIC, RILE_TASK_CONTEXT

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BatchedPipelineConfig:
    """Configuration for batched pipeline."""

    # vLLM server settings - can be single URL or list of URLs for load balancing
    task_model_url: str = "http://localhost:8000/v1"
    task_model_urls: Optional[List[str]] = None  # Multiple servers for load balancing
    auditor_model_url: str = "http://localhost:8001/v1"  # Can be same as task

    # Batching settings
    max_concurrent_requests: int = 100    # Max concurrent HTTP requests AND batch size
    max_concurrent_documents: int = 50    # Max documents in parallel
    batch_timeout: float = 0.05           # Max wait to fill batch (50ms)

    # Tree building
    max_chunk_chars: int = 2000
    max_tokens_summary: int = 500
    max_tokens_score: int = 200

    # Auditing
    audit_budget: int = 10
    rile_threshold: float = 10.0

    # Processing options
    run_baseline: bool = True
    run_audit: bool = True

    # Progress reporting
    show_progress: bool = True

    # Rubrics
    rubric: str = RILE_PRESERVATION_RUBRIC
    task_context: str = RILE_TASK_CONTEXT

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

    # Chunk the text
    chunks = chunk_text(text, config.max_chunk_chars)
    logger.debug(f"[{doc_id}] DSPy mode: Chunked into {len(chunks)} chunks")

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
    # Level 0: Summarize all chunks
    current_level = []
    leaf_summaries = []

    for i, chunk in enumerate(chunks):
        summary = leaf_summarizer(content=chunk, rubric=config.rubric)
        leaf_summaries.append(summary)
        current_level.append({
            "id": f"{doc_id}_leaf_{i}",
            "content": chunk,
            "summary": summary,
            "level": 0,
        })

    # Build up the tree by merging pairs
    level_num = 0
    while len(current_level) > 1:
        level_num += 1
        next_level = []

        # Pair up nodes
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                left = current_level[i]
                right = current_level[i + 1]

                left_text = left.get("summary") or left.get("content", "")
                right_text = right.get("summary") or right.get("content", "")

                # Use DSPy merge module
                merged_summary = merge_summarizer(
                    left_summary=left_text,
                    right_summary=right_text,
                    rubric=config.rubric
                )

                next_level.append({
                    "id": f"{doc_id}_merge_{level_num}_{len(next_level)}",
                    "summary": merged_summary,
                    "level": level_num,
                    "children": [left, right],
                })
            else:
                # Odd node carries forward
                next_level.append(current_level[i])

        current_level = next_level

    root = current_level[0] if current_level else {"summary": ""}
    return {
        "root": root,
        "height": level_num,
        "leaf_count": len(chunks),
        "chunks": chunks,
        "leaf_summaries": leaf_summaries,
    }


# =============================================================================
# Prompt Builders
# =============================================================================

def build_summarize_prompt(text: str, rubric: str) -> List[Dict[str, str]]:
    """Build summarization prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are a political text summarizer. Preserve all information "
                "relevant to left-right political positioning. Be concise but complete."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following political text, preserving information "
                f"relevant to: {rubric}\n\n"
                f"TEXT:\n{text}\n\n"
                f"SUMMARY:"
            )
        }
    ]


def build_merge_prompt(left: str, right: str, rubric: str) -> List[Dict[str, str]]:
    """Build merge/combine prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are a political text summarizer. Combine the following two "
                "summaries into one coherent summary, preserving all politically "
                "relevant information."
            )
        },
        {
            "role": "user",
            "content": (
                f"Combine these two summaries, preserving information relevant to: {rubric}\n\n"
                f"SUMMARY 1:\n{left}\n\n"
                f"SUMMARY 2:\n{right}\n\n"
                f"COMBINED SUMMARY:"
            )
        }
    ]


def build_rile_score_prompt(text: str, task_context: str) -> List[Dict[str, str]]:
    """Build RILE scoring prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are a political scientist expert in analyzing party manifestos. "
                "Score texts on the RILE (Right-Left) scale from -100 to +100."
            )
        },
        {
            "role": "user",
            "content": (
                f"{task_context}\n\n"
                f"TEXT TO SCORE:\n{text}\n\n"
                f"Provide your RILE score as a single number between -100 and +100. "
                f"Format: RILE_SCORE: <number>\n"
                f"Then briefly explain your reasoning."
            )
        }
    ]


def build_audit_prompt(original: str, summary: str, rubric: str) -> List[Dict[str, str]]:
    """Build audit/oracle prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are auditing whether a summary preserves politically relevant "
                "information from the original text."
            )
        },
        {
            "role": "user",
            "content": (
                f"Does the summary preserve the political position information "
                f"from the original?\n\n"
                f"Criteria: {rubric}\n\n"
                f"ORIGINAL:\n{original[:2000]}...\n\n"
                f"SUMMARY:\n{summary}\n\n"
                f"Answer PASS if information is preserved, FAIL if not. "
                f"Format: VERDICT: PASS/FAIL\nREASON: <explanation>"
            )
        }
    ]


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
    # Chunk the text
    chunks = chunk_text(text, config.max_chunk_chars)
    logger.debug(f"[{doc_id}] Chunked into {len(chunks)} chunks")

    if len(chunks) == 0:
        return {"root": {"summary": ""}, "height": 0, "leaf_count": 0}

    if len(chunks) == 1:
        # Single chunk - just summarize
        request = BatchRequest(
            request_id=f"{doc_id}_single",
            messages=build_summarize_prompt(chunks[0], config.rubric),
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
            messages=build_summarize_prompt(chunk, config.rubric),
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
                    messages=build_merge_prompt(left_text, right_text, config.rubric),
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


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into chunks."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Handle very long paragraphs
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            # Force split
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i:i+max_chars])

    return final_chunks


# =============================================================================
# RILE Scoring (Batched)
# =============================================================================

def parse_rile_score(response: str) -> Optional[float]:
    """Extract RILE score from response."""
    import re

    # Try to find "RILE_SCORE: X" pattern
    match = re.search(r'RILE_SCORE:\s*(-?\d+\.?\d*)', response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try to find any number in reasonable range
    numbers = re.findall(r'-?\d+\.?\d*', response)
    for num_str in numbers:
        try:
            num = float(num_str)
            if -100 <= num <= 100:
                return num
        except ValueError:
            continue

    return None


# =============================================================================
# Batched Pipeline
# =============================================================================

class BatchedManifestoPipeline:
    """
    High-throughput batched pipeline for manifesto processing.

    Processes multiple documents concurrently, pooling all LLM requests
    for optimal GPU utilization.

    For DSPy optimization, use with leaf_summarizer and merge_summarizer:
        from src.manifesto.dspy_summarizer import LeafSummarizer, MergeSummarizer

        pipeline = BatchedManifestoPipeline(
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
        self._results: List[ManifestoResult] = []

        # DSPy modules for training/optimization mode
        self.leaf_summarizer = leaf_summarizer
        self.merge_summarizer = merge_summarizer

    def process_with_dspy(
        self,
        sample: ManifestoSample,
    ) -> ManifestoResult:
        """
        Process a single manifesto using DSPy modules.

        This method is used during DSPy optimization to allow the summarization
        prompts to be learned. Unlike async methods, this runs synchronously.

        Args:
            sample: Manifesto sample to process

        Returns:
            ManifestoResult with tree info and summaries
        """
        if self.leaf_summarizer is None or self.merge_summarizer is None:
            raise ValueError(
                "DSPy processing requires leaf_summarizer and merge_summarizer. "
                "Initialize pipeline with these modules for training mode."
            )

        doc_id = sample.manifesto_id
        start_time = time.time()

        result = ManifestoResult(
            manifesto_id=doc_id,
            party_name=sample.party_name,
            country=sample.country_name,
            year=sample.year,
            ground_truth_rile=sample.rile,
            original_length=len(sample.text),
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
        samples: List[ManifestoSample],
        show_progress: bool = True,
    ) -> List[ManifestoResult]:
        """
        Process multiple manifestos using DSPy modules (synchronous).

        Args:
            samples: List of manifesto samples
            show_progress: Whether to show progress

        Returns:
            List of ManifestoResult
        """
        results = []
        for i, sample in enumerate(samples):
            if show_progress:
                logger.info(f"Processing {i+1}/{len(samples)}: {sample.manifesto_id}")
            result = self.process_with_dspy(sample)
            results.append(result)
            self._results.append(result)
        return results

    async def process_single_async(
        self,
        sample: ManifestoSample,
        client: AsyncBatchLLMClient,
    ) -> ManifestoResult:
        """
        Process a single manifesto (used internally by batch processor).

        Args:
            sample: Manifesto sample
            client: Batch LLM client

        Returns:
            ManifestoResult
        """
        doc_id = sample.manifesto_id
        start_time = time.time()

        result = ManifestoResult(
            manifesto_id=doc_id,
            party_name=sample.party_name,
            country=sample.country_name,
            year=sample.year,
            ground_truth_rile=sample.rile,
            original_length=len(sample.text),
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

            # 2. Score RILE from summary
            score_request = BatchRequest(
                request_id=f"{doc_id}_score",
                messages=build_rile_score_prompt(
                    result.final_summary,
                    self.config.task_context
                ),
                max_tokens=self.config.max_tokens_score,
                document_id=doc_id,
                request_type="score",
            )
            await client.submit(score_request)
            score_response = await client.await_response(score_request.request_id)

            result.predicted_rile = parse_rile_score(score_response.content)
            result.reasoning = score_response.content

            # 3. Baseline score (optional)
            if self.config.run_baseline:
                # Use truncated full text
                baseline_text = sample.text[:50000]
                baseline_request = BatchRequest(
                    request_id=f"{doc_id}_baseline",
                    messages=build_rile_score_prompt(
                        baseline_text,
                        self.config.task_context
                    ),
                    max_tokens=self.config.max_tokens_score,
                    document_id=doc_id,
                    request_type="baseline",
                )
                await client.submit(baseline_request)
                baseline_response = await client.await_response(baseline_request.request_id)
                result.baseline_rile = parse_rile_score(baseline_response.content)

        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            result.error = str(e)

        return result

    async def process_batch_async(
        self,
        samples: List[ManifestoSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[ManifestoResult]:
        """
        Process multiple manifestos with full batching.

        By default uses level-wise batching for maximum throughput.
        Set config.use_levelwise_batching=False for per-document processing.

        Args:
            samples: List of manifesto samples
            progress_callback: Optional progress callback(completed, total)
            show_progress: Override config.show_progress setting

        Returns:
            List of ManifestoResults
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
        logger.info(f"Starting batched processing of {len(samples)} manifestos (per-document mode)")
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
                batch_size=self.config.max_concurrent_requests,
                batch_timeout=self.config.batch_timeout,
            )
        else:
            client = AsyncBatchLLMClient(
                base_url=server_urls[0],
                max_concurrent=self.config.max_concurrent_requests,
                batch_size=self.config.max_concurrent_requests,
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
        samples: List[ManifestoSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[ManifestoResult]:
        """
        Sync wrapper for batch processing.

        Args:
            samples: List of manifesto samples
            progress_callback: Optional progress callback
            show_progress: Override config.show_progress setting

        Returns:
            List of ManifestoResults
        """
        return asyncio.run(self.process_batch_async(samples, progress_callback, show_progress))

    def get_results(self) -> List[ManifestoResult]:
        """Get all processed results."""
        return self._results

    async def process_batch_levelwise_async(
        self,
        samples: List[ManifestoSample],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        show_progress: Optional[bool] = None,
    ) -> List[ManifestoResult]:
        """
        Process manifestos using level-wise batching for maximum throughput.

        This method processes ALL documents level-by-level:
        1. Chunks ALL documents
        2. Submits ALL leaf summaries together
        3. Awaits ALL responses
        4. Submits ALL level-1 merges together
        5. Continues until all trees are complete
        6. Scores ALL documents together

        This ensures vLLM sees the largest possible batches at each tree level.

        Args:
            samples: List of manifesto samples
            progress_callback: Optional callback(phase, completed, total)
            show_progress: Override config.show_progress setting

        Returns:
            List of ManifestoResults
        """
        logger.info(f"Starting LEVEL-WISE batched processing of {len(samples)} manifestos")
        start_time = time.time()

        use_progress = show_progress if show_progress is not None else self.config.show_progress
        server_urls = self.config.task_model_urls or [self.config.task_model_url]

        if len(server_urls) > 1:
            logger.info(f"Using {len(server_urls)} servers for load balancing")
            client = MultiServerBatchClient(
                servers=server_urls,
                max_concurrent_per_server=self.config.max_concurrent_requests,
                batch_size=self.config.max_concurrent_requests,
                batch_timeout=self.config.batch_timeout,
            )
        else:
            client = AsyncBatchLLMClient(
                base_url=server_urls[0],
                max_concurrent=self.config.max_concurrent_requests,
                batch_size=self.config.max_concurrent_requests,
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
                summarize_prompt_fn=build_summarize_prompt,
                merge_prompt_fn=build_merge_prompt,
            )

            tree_states = await level_processor.process_all_documents(
                documents=samples,
                get_text_fn=lambda s: s.text,
                get_id_fn=lambda s: s.manifesto_id,
                progress_callback=progress_callback,
            )

            # Phase 2: Score ALL documents' summaries together
            logger.info(f"Phase 4: Scoring {len(tree_states)} documents...")
            score_requests = []  # [(state_idx, request)]

            for state_idx, state in enumerate(tree_states):
                if state.error or not state.root_summary:
                    continue

                request = BatchRequest(
                    request_id=f"{state.doc_id}_score",
                    messages=build_rile_score_prompt(state.root_summary, self.config.task_context),
                    max_tokens=self.config.max_tokens_score,
                    document_id=state.doc_id,
                    request_type="score",
                )
                score_requests.append((state_idx, request))
                await client.submit(request)

            logger.info(f"  Submitted {len(score_requests)} score requests...")

            # Await all scores
            scores = {}  # state_idx -> rile_score
            for state_idx, request in score_requests:
                response = await client.await_response(request.request_id)
                scores[state_idx] = parse_rile_score(response.content)

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
                        messages=build_rile_score_prompt(baseline_text, self.config.task_context),
                        max_tokens=self.config.max_tokens_score,
                        document_id=state.doc_id,
                        request_type="baseline",
                    )
                    baseline_requests.append((state_idx, request))
                    await client.submit(request)

                logger.info(f"  Submitted {len(baseline_requests)} baseline requests...")

                baselines = {}  # state_idx -> baseline_rile
                for state_idx, request in baseline_requests:
                    response = await client.await_response(request.request_id)
                    baselines[state_idx] = parse_rile_score(response.content)
            else:
                baselines = {}

            # Convert tree states to ManifestoResults
            for state_idx, state in enumerate(tree_states):
                sample = state.sample

                result = ManifestoResult(
                    manifesto_id=state.doc_id,
                    party_name=sample.party_name,
                    country=sample.country_name,
                    year=sample.year,
                    ground_truth_rile=sample.rile,
                    original_length=len(sample.text),
                    tree_height=state.tree_height,
                    tree_leaves=state.leaf_count,
                    final_summary=state.root_summary,
                    summary_length=len(state.root_summary),
                    compression_ratio=len(sample.text) / max(len(state.root_summary), 1),
                    predicted_rile=scores.get(state_idx),
                    baseline_rile=baselines.get(state_idx),
                    error=state.error,
                    chunks=state.chunks,
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

async def process_manifestos_batched(
    samples: List[ManifestoSample],
    config: Optional[BatchedPipelineConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[ManifestoResult]:
    """
    High-level async function to process manifestos with batching.

    Args:
        samples: Manifesto samples to process
        config: Pipeline configuration
        progress_callback: Progress callback

    Returns:
        List of ManifestoResults
    """
    pipeline = BatchedManifestoPipeline(config)
    return await pipeline.process_batch_async(samples, progress_callback)


def run_batched_experiment(
    samples: List[ManifestoSample],
    config: Optional[BatchedPipelineConfig] = None,
) -> List[ManifestoResult]:
    """
    Sync convenience function for running batched experiments.

    Args:
        samples: Manifesto samples
        config: Configuration

    Returns:
        Results
    """
    pipeline = BatchedManifestoPipeline(config)
    return pipeline.process_batch(samples)
