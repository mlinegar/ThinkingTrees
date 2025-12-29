"""
Batched Request Processing for vLLM.

This module implements high-throughput batched processing that:
1. Pools requests from multiple documents/trees
2. Sends concurrent batches to vLLM (leveraging its internal batching)
3. Routes responses back to waiting coroutines

The key insight: while we can't parallelize tree levels (children before parents),
we CAN parallelize across multiple documents AND pool requests from the same
level across many trees.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    BatchOrchestrator                         │
    │  Manages N concurrent documents, pools their requests        │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │    Request Pool     │
                   │  (async queue)      │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Batch Workers     │
                   │  (N concurrent)     │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   vLLM Server       │
                   └─────────────────────┘
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Awaitable, Tuple, Union
from collections import defaultdict
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from src.preprocessing.chunker import chunk_for_ops

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Types
# =============================================================================

@dataclass
class BatchRequest:
    """A single LLM request in the batch pool."""
    request_id: str
    messages: List[Dict[str, str]]
    max_tokens: int = 8192
    temperature: float = 0.7

    # Tracking
    document_id: Optional[str] = None
    request_type: str = "summarize"  # summarize, audit, score
    priority: int = 0  # Higher = more urgent

    # Response handling
    future: Optional[asyncio.Future] = None
    submitted_at: Optional[float] = None


@dataclass
class BatchResponse:
    """Response from vLLM."""
    request_id: str
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    batches_sent: int = 0
    wall_clock_start: float = 0.0
    wall_clock_end: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return self.total_latency_ms / self.completed_requests

    @property
    def requests_per_second(self) -> float:
        if self.total_latency_ms == 0:
            return 0.0
        return self.completed_requests / (self.total_latency_ms / 1000)

    @property
    def wall_clock_seconds(self) -> float:
        """Wall clock time in seconds. Uses current time if not yet stopped."""
        if self.wall_clock_start == 0:
            return 0.0
        # If not stopped yet, use current time for live updates
        end_time = self.wall_clock_end if self.wall_clock_end > 0 else time.time()
        return end_time - self.wall_clock_start

    @property
    def tokens_per_second(self) -> float:
        """Total tokens per second (wall clock time)."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_tokens / self.wall_clock_seconds

    @property
    def read_tokens_per_second(self) -> float:
        """Prompt/input tokens per second."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.prompt_tokens / self.wall_clock_seconds

    @property
    def write_tokens_per_second(self) -> float:
        """Completion/output tokens per second."""
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.wall_clock_seconds

    def __str__(self) -> str:
        return (
            f"BatchStats(reqs={self.completed_requests}/{self.total_requests}, "
            f"tokens={self.total_tokens:,}, "
            f"tok/s={self.tokens_per_second:.0f} "
            f"[r:{self.read_tokens_per_second:.0f}, w:{self.write_tokens_per_second:.0f}])"
        )


# =============================================================================
# Async Batch Client
# =============================================================================

class AsyncBatchLLMClient:
    """
    Async client for batched LLM requests.

    Pools requests and sends them concurrently to vLLM, which handles
    internal batching for optimal GPU utilization.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        max_concurrent: int = 200,  # Max concurrent requests to vLLM
        batch_size: int = 50,       # Requests per batch
        batch_timeout: float = 0.1,  # Max wait to fill batch (seconds)
        model: str = None,  # Auto-detect from server if None
        request_timeout: float = 300.0,  # Per-request timeout (5 minutes)
    ):
        """
        Initialize async batch client.

        Args:
            base_url: vLLM server URL
            max_concurrent: Maximum concurrent HTTP requests
            batch_size: Target batch size before sending
            batch_timeout: Max time to wait for batch to fill
            model: Model name for vLLM (auto-detected if None)
            request_timeout: Per-request HTTP timeout in seconds
        """
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._model = model  # Will be set during start() if None
        self.request_timeout = request_timeout

        # Request pool
        self._request_queue: asyncio.Queue[BatchRequest] = None
        self._pending_futures: Dict[str, asyncio.Future] = {}

        # Concurrency control
        self._semaphore: asyncio.Semaphore = None
        self._session: aiohttp.ClientSession = None

        # Statistics
        self.stats = BatchStats()

        # State
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def model(self) -> str:
        """Get model name (auto-detected if not set)."""
        return self._model or "unknown"

    async def _detect_model(self) -> str:
        """Auto-detect model name from vLLM server."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("data"):
                            model_id = data["data"][0]["id"]
                            logger.info(f"Auto-detected model: {model_id}")
                            return model_id
        except Exception as e:
            logger.warning(f"Failed to auto-detect model: {e}")
        return "default"

    async def start(self):
        """Start the batch processor."""
        if self._running:
            return

        # Auto-detect model if not specified
        if self._model is None:
            self._model = await self._detect_model()

        self._request_queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        # Set connector limit to match max_concurrent (default aiohttp limit is 100)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        # Set timeout for all requests
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self._running = True

        # Track start time
        self.stats.wall_clock_start = time.time()

        # Start batch worker
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.debug(f"Batch client started (max_concurrent={self.max_concurrent}, model={self._model})")

    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        self.stats.wall_clock_end = time.time()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
        logger.info(f"Batch client stopped. Stats: {self.stats}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def submit(self, request: BatchRequest) -> str:
        """
        Submit a request to the pool.

        Returns immediately with a request_id. Use await_response() to get result.
        """
        if not self._running:
            raise RuntimeError("Batch client not started")

        # Create future for response
        request.future = asyncio.get_event_loop().create_future()
        request.submitted_at = time.time()
        self._pending_futures[request.request_id] = request.future

        # Add to queue
        await self._request_queue.put(request)
        self.stats.total_requests += 1

        return request.request_id

    async def await_response(
        self,
        request_id: str,
        timeout: float = 600.0,  # 10 minutes default (increased for large queues)
    ) -> BatchResponse:
        """
        Wait for a submitted request to complete.

        Args:
            request_id: The request ID to wait for
            timeout: Maximum wait time in seconds (default 10 minutes)

        Returns:
            BatchResponse with the result
        """
        if request_id not in self._pending_futures:
            raise KeyError(f"Unknown request_id: {request_id}")

        future = self._pending_futures[request_id]

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            del self._pending_futures[request_id]
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out after {timeout:.0f}s "
                        f"({len(self._pending_futures)} still pending)")
            del self._pending_futures[request_id]
            return BatchResponse(
                request_id=request_id,
                content="",
                error=f"Timeout after {timeout}s"
            )

    async def call(self, request: BatchRequest) -> BatchResponse:
        """Submit and await in one call (convenience method)."""
        await self.submit(request)
        return await self.await_response(request.request_id)

    async def _batch_worker(self):
        """Background worker that collects and sends batches."""
        while self._running:
            try:
                batch = []
                deadline = time.time() + self.batch_timeout

                # Collect requests until batch_size or timeout
                while len(batch) < self.batch_size:
                    timeout = max(0.001, deadline - time.time())
                    try:
                        request = await asyncio.wait_for(
                            self._request_queue.get(),
                            timeout=timeout
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    # Send batch concurrently
                    await self._send_batch(batch)
                    self.stats.batches_sent += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                await asyncio.sleep(0.1)

    async def _send_batch(self, batch: List[BatchRequest]):
        """Send a batch of requests concurrently."""
        tasks = [self._send_single(req) for req in batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_single(self, request: BatchRequest):
        """Send a single request with semaphore control."""
        async with self._semaphore:
            start_time = time.time()
            try:
                payload = {
                    "model": self._model,
                    "messages": request.messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }

                async with self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer EMPTY"}
                ) as resp:
                    data = await resp.json()

                    latency = (time.time() - start_time) * 1000

                    if resp.status == 200:
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})

                        response = BatchResponse(
                            request_id=request.request_id,
                            content=content,
                            usage=usage,
                            latency_ms=latency,
                        )
                        self.stats.completed_requests += 1
                        self.stats.total_latency_ms += latency
                        self.stats.total_tokens += usage.get("total_tokens", 0)
                        self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
                        self.stats.completion_tokens += usage.get("completion_tokens", 0)
                    else:
                        response = BatchResponse(
                            request_id=request.request_id,
                            content="",
                            error=f"HTTP {resp.status}: {data}",
                            latency_ms=latency,
                        )
                        self.stats.failed_requests += 1

                    # Resolve the future
                    if request.future and not request.future.done():
                        request.future.set_result(response)

            except aiohttp.ClientError as e:
                # Connection errors, timeouts, etc.
                error_msg = f"{type(e).__name__}: {str(e) or 'Connection failed'}"
                logger.error(f"Request {request.request_id} failed: {error_msg}")
                self.stats.failed_requests += 1
                if request.future and not request.future.done():
                    request.future.set_result(BatchResponse(
                        request_id=request.request_id,
                        content="",
                        error=error_msg,
                    ))
            except asyncio.TimeoutError:
                error_msg = "Request timed out"
                logger.error(f"Request {request.request_id} failed: {error_msg}")
                self.stats.failed_requests += 1
                if request.future and not request.future.done():
                    request.future.set_result(BatchResponse(
                        request_id=request.request_id,
                        content="",
                        error=error_msg,
                    ))
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                logger.error(f"Request {request.request_id} failed: {error_msg}")
                self.stats.failed_requests += 1
                if request.future and not request.future.done():
                    request.future.set_result(BatchResponse(
                        request_id=request.request_id,
                        content="",
                        error=error_msg,
                    ))


# =============================================================================
# Multi-Server Load Balancer
# =============================================================================

class MultiServerBatchClient:
    """
    Load balances requests across multiple vLLM servers.

    Uses round-robin scheduling to distribute requests evenly.
    Aggregates stats from all underlying clients.
    """

    def __init__(
        self,
        servers: List[str],  # List of base URLs, e.g., ["http://localhost:8000/v1", "http://localhost:8002/v1"]
        max_concurrent_per_server: int = 200,
        batch_size: int = 50,
        batch_timeout: float = 0.1,
    ):
        """
        Initialize multi-server client.

        Args:
            servers: List of vLLM server URLs
            max_concurrent_per_server: Max concurrent requests per server
            batch_size: Requests per batch
            batch_timeout: Max wait to fill batch
        """
        self.servers = servers
        self.clients: List[AsyncBatchLLMClient] = []
        self._counter = 0  # Round-robin counter
        self._lock = asyncio.Lock() if asyncio else None
        self._request_client_map: Dict[str, AsyncBatchLLMClient] = {}  # request_id -> client (O(1) lookup)

        # Create a client for each server
        for server_url in servers:
            client = AsyncBatchLLMClient(
                base_url=server_url,
                max_concurrent=max_concurrent_per_server,
                batch_size=batch_size,
                batch_timeout=batch_timeout,
            )
            self.clients.append(client)

    @property
    def stats(self) -> BatchStats:
        """Aggregate stats from all clients."""
        combined = BatchStats()
        for client in self.clients:
            combined.total_requests += client.stats.total_requests
            combined.completed_requests += client.stats.completed_requests
            combined.failed_requests += client.stats.failed_requests
            combined.total_tokens += client.stats.total_tokens
            combined.prompt_tokens += client.stats.prompt_tokens
            combined.completion_tokens += client.stats.completion_tokens
            combined.total_latency_ms += client.stats.total_latency_ms
            combined.batches_sent += client.stats.batches_sent
        # Use wall clock from first client
        if self.clients:
            combined.wall_clock_start = self.clients[0].stats.wall_clock_start
            combined.wall_clock_end = self.clients[0].stats.wall_clock_end
        return combined

    async def start(self):
        """Start all underlying clients."""
        self._lock = asyncio.Lock()
        await asyncio.gather(*[c.start() for c in self.clients])
        models = [c.model for c in self.clients]
        logger.info(f"Multi-server client started with {len(self.clients)} servers: {models}")

    async def stop(self):
        """Stop all underlying clients."""
        await asyncio.gather(*[c.stop() for c in self.clients])
        logger.info(f"Multi-server client stopped. Combined stats: {self.stats}")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def _get_next_client(self) -> AsyncBatchLLMClient:
        """Get next client using round-robin."""
        client = self.clients[self._counter % len(self.clients)]
        self._counter += 1
        return client

    async def submit(self, request: BatchRequest) -> str:
        """Submit request to next available server (round-robin)."""
        client = self._get_next_client()
        request_id = await client.submit(request)
        # Store mapping for O(1) lookup in await_response
        self._request_client_map[request_id] = client
        return request_id

    async def await_response(self, request_id: str) -> BatchResponse:
        """Wait for response using O(1) client lookup."""
        # Direct lookup using stored mapping
        client = self._request_client_map.get(request_id)
        if client is None:
            raise KeyError(f"Unknown request_id: {request_id}")

        response = await client.await_response(request_id)
        # Clean up mapping after response received
        del self._request_client_map[request_id]
        return response

    async def call(self, request: BatchRequest) -> BatchResponse:
        """Submit and await in one call (no mapping needed, direct to client)."""
        client = self._get_next_client()
        # call() handles submit+await internally on the same client
        return await client.call(request)


# =============================================================================
# Multi-Document Batch Orchestrator
# =============================================================================

class BatchOrchestrator:
    """
    Orchestrates batched processing across multiple documents.

    Key strategy:
    - Process documents in waves
    - At each tree level, collect ALL requests across ALL documents
    - Send them as one big batch to vLLM
    - This maximizes GPU utilization

    Example with 100 documents, 10 chunks each:
    - Level 0 (leaves): 1000 summarization requests batched together
    - Level 1: ~500 merge requests batched
    - Level 2: ~250 merge requests batched
    - etc.
    """

    def __init__(
        self,
        client: Union[AsyncBatchLLMClient, MultiServerBatchClient],
        max_concurrent_documents: int = 50,
    ):
        """
        Initialize orchestrator.

        Args:
            client: Async batch LLM client (single or multi-server)
            max_concurrent_documents: Max documents to process simultaneously
        """
        self.client = client
        self.max_concurrent_documents = max_concurrent_documents

        # Statistics
        self.documents_processed = 0
        self.total_requests = 0

    async def process_documents(
        self,
        documents: List[Any],
        process_fn: Callable[[Any, AsyncBatchLLMClient], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        Process multiple documents with batched LLM calls.

        Args:
            documents: List of documents to process
            process_fn: Async function(doc, client) -> result
            progress_callback: Optional callback(completed, total)

        Returns:
            List of results in same order as input
        """
        results = [None] * len(documents)
        completed = 0
        total_waves = (len(documents) + self.max_concurrent_documents - 1) // self.max_concurrent_documents
        show_per_doc = len(documents) <= 20  # Show per-doc status for small batches

        # Process in waves of max_concurrent_documents
        for wave_start in range(0, len(documents), self.max_concurrent_documents):
            wave_end = min(wave_start + self.max_concurrent_documents, len(documents))
            wave_docs = documents[wave_start:wave_end]
            wave_indices = list(range(wave_start, wave_end))
            wave_num = wave_start // self.max_concurrent_documents + 1

            logger.info(f"Wave {wave_num}/{total_waves}: Processing {len(wave_docs)} documents...")
            wave_start_time = time.time()

            # Process wave concurrently
            tasks = [
                process_fn(doc, self.client)
                for doc in wave_docs
            ]
            wave_results = await asyncio.gather(*tasks, return_exceptions=True)

            wave_elapsed = time.time() - wave_start_time
            wave_failures = sum(1 for r in wave_results if isinstance(r, Exception))

            # Store results and optionally show per-doc status
            for idx, result in zip(wave_indices, wave_results):
                if isinstance(result, Exception):
                    logger.error(f"  ✗ Doc {idx}: {str(result)[:60]}")
                    results[idx] = None
                else:
                    results[idx] = result
                    # Show per-doc status for small batches (domain-agnostic)
                    if show_per_doc:
                        # Use generic doc_id or fall back to manifesto_id for backwards compat
                        doc_id = getattr(result, 'doc_id', None) or getattr(result, 'manifesto_id', None)
                        if doc_id:
                            # Get predicted/truth scores (try generic then domain-specific)
                            pred = getattr(result, 'predicted_score', None) or getattr(result, 'predicted_rile', None)
                            truth = getattr(result, 'ground_truth_score', None) or getattr(result, 'ground_truth_rile', None)
                            leaves = getattr(result, 'tree_leaves', 0)
                            pred_str = f"{pred:.1f}" if pred is not None else "?"
                            truth_str = f"{truth:.1f}" if truth is not None else "?"
                            logger.info(f"  ✓ {doc_id}: pred={pred_str}, truth={truth_str}, leaves={leaves}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(documents))

            # Wave summary
            logger.info(f"Wave {wave_num}/{total_waves}: Done in {wave_elapsed:.1f}s "
                       f"({len(wave_docs) - wave_failures}/{len(wave_docs)} succeeded)")

        self.documents_processed += len(documents)
        return results


# =============================================================================
# Level-wise Batch Tree Builder
# =============================================================================

async def build_tree_batched(
    chunks: List[str],
    rubric: str,
    client: AsyncBatchLLMClient,
    document_id: str,
    summarize_prompt_fn: Callable[[str, str], List[Dict[str, str]]],
    merge_prompt_fn: Callable[[str, str, str], List[Dict[str, str]]],
) -> Dict[str, Any]:
    """
    Build an OPS tree using batched requests.

    Each tree level's requests are batched together.
    When processing multiple documents, their requests at the same level
    will be pooled by the AsyncBatchLLMClient.

    Args:
        chunks: List of text chunks (leaves)
        rubric: Summarization rubric
        client: Batch LLM client
        document_id: Identifier for this document
        summarize_prompt_fn: Function(text, rubric) -> messages
        merge_prompt_fn: Function(left, right, rubric) -> messages

    Returns:
        Tree structure with summaries
    """
    import uuid

    # Initialize leaves
    current_level = []
    for i, chunk in enumerate(chunks):
        node = {
            "id": f"{document_id}_leaf_{i}",
            "content": chunk,
            "summary": None,
            "level": 0,
            "children": [],
        }
        current_level.append(node)

    level_num = 0

    # Build tree level by level
    while len(current_level) > 1:
        level_num += 1
        next_level = []

        # Pair up nodes
        pairs = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                pairs.append((current_level[i], current_level[i + 1]))
            else:
                # Odd node - carry forward
                next_level.append(current_level[i])

        # Create batch requests for all merges at this level
        merge_requests = []
        for left, right in pairs:
            left_text = left.get("summary") or left["content"]
            right_text = right.get("summary") or right["content"]

            messages = merge_prompt_fn(left_text, right_text, rubric)

            request = BatchRequest(
                request_id=f"{document_id}_merge_{level_num}_{len(merge_requests)}",
                messages=messages,
                document_id=document_id,
                request_type="merge",
            )
            merge_requests.append((request, left, right))

        # Submit all requests (they'll be batched with requests from other docs)
        for request, _, _ in merge_requests:
            await client.submit(request)

        # Await all responses
        for request, left, right in merge_requests:
            response = await client.await_response(request.request_id)

            # Create parent node
            parent = {
                "id": f"{document_id}_node_{level_num}_{len(next_level)}",
                "content": None,
                "summary": response.content if not response.error else f"[Error: {response.error}]",
                "level": level_num,
                "children": [left, right],
            }
            next_level.append(parent)

        current_level = next_level

    # Return root
    root = current_level[0] if current_level else None
    return {
        "root": root,
        "height": level_num,
        "leaf_count": len(chunks),
    }


# =============================================================================
# Batch Audit Checks
# =============================================================================

async def audit_nodes_batched(
    nodes: List[Dict[str, Any]],
    oracle_prompt_fn: Callable[[str, str, str], List[Dict[str, str]]],
    client: AsyncBatchLLMClient,
    rubric: str,
    document_id: str,
) -> List[Dict[str, Any]]:
    """
    Audit multiple nodes with batched oracle calls.

    Args:
        nodes: Nodes to audit
        oracle_prompt_fn: Function(original, summary, rubric) -> messages
        client: Batch LLM client
        rubric: Audit rubric
        document_id: Document identifier

    Returns:
        List of audit results
    """
    # Create requests for all nodes
    requests = []
    for i, node in enumerate(nodes):
        original = node.get("content") or ""
        summary = node.get("summary") or ""

        messages = oracle_prompt_fn(original, summary, rubric)

        request = BatchRequest(
            request_id=f"{document_id}_audit_{i}",
            messages=messages,
            document_id=document_id,
            request_type="audit",
        )
        requests.append((request, node))

    # Submit all
    for request, _ in requests:
        await client.submit(request)

    # Await all
    results = []
    for request, node in requests:
        response = await client.await_response(request.request_id)
        results.append({
            "node_id": node["id"],
            "passed": "pass" in response.content.lower() if response.content else False,
            "response": response.content,
            "error": response.error,
        })

    return results


# =============================================================================
# Level-Wise Multi-Document Batch Processor
# =============================================================================

@dataclass
class DocumentTreeState:
    """Tracks tree-building state for a single document."""
    doc_id: str
    sample: Any
    chunks: List[str] = field(default_factory=list)
    current_level: List[Dict[str, Any]] = field(default_factory=list)
    level_num: int = 0
    error: Optional[str] = None

    # Level history for merge law support
    # Each entry is a list of nodes at that level (level 0 = leaves, level N = root)
    level_history: List[List[Dict[str, Any]]] = field(default_factory=list)

    # Results
    root_summary: str = ""
    tree_height: int = 0
    leaf_count: int = 0


class LevelWiseBatchProcessor:
    """
    Processes multiple document trees level-by-level for maximum batching.

    Unlike per-document processing, this:
    1. Pre-chunks ALL documents
    2. Submits ALL leaf summaries together
    3. Awaits ALL responses
    4. Submits ALL level-1 merges together
    5. Continues level by level

    This ensures vLLM sees the largest possible batches at each level.
    """

    def __init__(
        self,
        client: Union[AsyncBatchLLMClient, MultiServerBatchClient],
        rubric: str,
        max_chunk_chars: int = 2000,
        max_tokens_summary: int = 500,
        summarize_prompt_fn: Optional[Callable] = None,
        merge_prompt_fn: Optional[Callable] = None,
    ):
        self.client = client
        self.rubric = rubric
        self.max_chunk_chars = max_chunk_chars
        self.max_tokens_summary = max_tokens_summary

        # Use provided prompt functions or defaults
        self.summarize_prompt_fn = summarize_prompt_fn
        self.merge_prompt_fn = merge_prompt_fn

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using langextract-based chunker."""
        text_chunks = chunk_for_ops(text, max_chars=self.max_chunk_chars)
        return [c.text for c in text_chunks]

    async def process_all_documents(
        self,
        documents: List[Any],
        get_text_fn: Callable[[Any], str],
        get_id_fn: Callable[[Any], str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[DocumentTreeState]:
        """
        Process all documents using level-wise batching.

        Args:
            documents: List of documents
            get_text_fn: Function to extract text from document
            get_id_fn: Function to extract ID from document
            progress_callback: Optional callback(phase, completed, total)

        Returns:
            List of DocumentTreeState with results
        """
        # Phase 1: Pre-chunk all documents
        logger.info(f"Phase 1: Chunking {len(documents)} documents...")
        states = []
        total_chunks = 0

        for doc in documents:
            doc_id = get_id_fn(doc)
            try:
                text = get_text_fn(doc)
                if not text or len(text.strip()) == 0:
                    logger.warning(f"Document {doc_id} has no text content, skipping")
                    state = DocumentTreeState(
                        doc_id=doc_id,
                        sample=doc,
                        chunks=[],
                        leaf_count=0,
                        error="No text content",
                    )
                    states.append(state)
                    continue

                chunks = self._chunk_text(text)

                if not chunks or len(chunks) == 0:
                    logger.warning(f"Document {doc_id} produced no chunks, skipping")
                    state = DocumentTreeState(
                        doc_id=doc_id,
                        sample=doc,
                        chunks=[],
                        leaf_count=0,
                        error="Chunking failed",
                    )
                    states.append(state)
                    continue

                state = DocumentTreeState(
                    doc_id=doc_id,
                    sample=doc,
                    chunks=chunks,
                    leaf_count=len(chunks),
                )
                states.append(state)
                total_chunks += len(chunks)
            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {e}")
                state = DocumentTreeState(
                    doc_id=doc_id,
                    sample=doc,
                    chunks=[],
                    leaf_count=0,
                    error=str(e),
                )
                states.append(state)

        logger.info(f"  Total chunks across all docs: {total_chunks}")
        if progress_callback:
            progress_callback("chunk", len(documents), len(documents))

        # Phase 2: Submit ALL leaf summaries
        logger.info(f"Phase 2: Submitting {total_chunks} leaf summaries...")
        leaf_requests = []  # [(state_idx, chunk_idx, request)]

        for state_idx, state in enumerate(states):
            if len(state.chunks) == 0:
                state.error = "No chunks"
                continue

            for chunk_idx, chunk in enumerate(state.chunks):
                request = BatchRequest(
                    request_id=f"{state.doc_id}_leaf_{chunk_idx}",
                    messages=self.summarize_prompt_fn(chunk, self.rubric),
                    max_tokens=self.max_tokens_summary,
                    document_id=state.doc_id,
                    request_type="summarize",
                )
                leaf_requests.append((state_idx, chunk_idx, request))
                await self.client.submit(request)

        logger.info(f"  Submitted {len(leaf_requests)} requests, awaiting responses...")

        # Await ALL leaf responses
        completed = 0
        for state_idx, chunk_idx, request in leaf_requests:
            state = states[state_idx]
            response = await self.client.await_response(request.request_id)

            # Initialize current_level if needed
            while len(state.current_level) <= chunk_idx:
                state.current_level.append(None)

            state.current_level[chunk_idx] = {
                "id": request.request_id,
                "content": state.chunks[chunk_idx],
                "summary": response.content if not response.error else state.chunks[chunk_idx][:500],
                "level": 0,
            }

            completed += 1
            if progress_callback and completed % 100 == 0:
                progress_callback("leaf", completed, len(leaf_requests))

        logger.info(f"  Completed {completed} leaf summaries")
        if progress_callback:
            progress_callback("leaf", completed, len(leaf_requests))

        # Save leaf level (level 0) to history for merge law support
        for state in states:
            if state.current_level and state.error is None:
                state.level_history.append([node.copy() for node in state.current_level])

        # Phase 3: Build trees level by level
        level_num = 0
        docs_needing_merge = [s for s in states if len(s.current_level) > 1 and s.error is None]

        while docs_needing_merge:
            level_num += 1
            logger.info(f"Phase 3.{level_num}: Merging level {level_num} for {len(docs_needing_merge)} documents...")

            # Collect ALL merge requests for this level
            merge_requests = []  # [(state_idx, pair_idx, request, left, right)]

            for state in docs_needing_merge:
                state.level_num = level_num
                pairs = []

                # Pair up nodes
                for i in range(0, len(state.current_level), 2):
                    if i + 1 < len(state.current_level):
                        left = state.current_level[i]
                        right = state.current_level[i + 1]
                        pairs.append((left, right))

                # Create merge requests for this doc
                for pair_idx, (left, right) in enumerate(pairs):
                    left_text = left.get("summary") or left.get("content", "")
                    right_text = right.get("summary") or right.get("content", "")

                    request = BatchRequest(
                        request_id=f"{state.doc_id}_merge_{level_num}_{pair_idx}",
                        messages=self.merge_prompt_fn(left_text, right_text, self.rubric),
                        max_tokens=self.max_tokens_summary,
                        document_id=state.doc_id,
                        request_type="merge",
                    )
                    merge_requests.append((states.index(state), pair_idx, request, left, right))

            if not merge_requests:
                break

            # Submit ALL merge requests for this level
            for _, _, request, _, _ in merge_requests:
                await self.client.submit(request)

            logger.info(f"  Submitted {len(merge_requests)} merge requests, awaiting...")

            # Await ALL merge responses and build next level
            next_levels = {states.index(s): [] for s in docs_needing_merge}

            completed = 0
            for state_idx, pair_idx, request, left, right in merge_requests:
                response = await self.client.await_response(request.request_id)

                next_levels[state_idx].append({
                    "id": request.request_id,
                    "summary": response.content if not response.error else "",
                    "level": level_num,
                    "children": [left, right],
                })

                completed += 1

            # Handle odd nodes (carry forward)
            for state in docs_needing_merge:
                state_idx = states.index(state)
                if len(state.current_level) % 2 == 1:
                    next_levels[state_idx].append(state.current_level[-1])

                state.current_level = next_levels[state_idx]
                state.tree_height = level_num

                # Save this level to history for merge law support
                if state.current_level:
                    state.level_history.append([node.copy() for node in state.current_level])

            logger.info(f"  Completed {completed} merges")
            if progress_callback:
                progress_callback(f"merge_L{level_num}", completed, len(merge_requests))

            # Update which docs need more merging
            docs_needing_merge = [s for s in states if len(s.current_level) > 1 and s.error is None]

        # Extract final summaries
        for state in states:
            if state.current_level and state.error is None:
                root = state.current_level[0]
                state.root_summary = root.get("summary") or root.get("content", "")

        logger.info(f"Tree building complete: {len(states)} documents processed")
        return states


# =============================================================================
# Convenience Functions
# =============================================================================

def run_batched(coro):
    """Run an async coroutine from sync code."""
    return asyncio.get_event_loop().run_until_complete(coro)


async def process_samples_batched(
    samples: List[Any],
    process_fn: Callable[[Any, AsyncBatchLLMClient], Awaitable[Any]],
    base_url: str = "http://localhost:8000/v1",
    max_concurrent: int = 200,
    max_concurrent_documents: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """
    High-level function to process samples with batching.

    Args:
        samples: List of samples to process
        process_fn: Async function(sample, client) -> result
        base_url: vLLM server URL
        max_concurrent: Max concurrent LLM requests
        max_concurrent_documents: Max concurrent documents
        progress_callback: Progress callback(completed, total)

    Returns:
        List of results
    """
    async with AsyncBatchLLMClient(
        base_url=base_url,
        max_concurrent=max_concurrent,
    ) as client:
        orchestrator = BatchOrchestrator(
            client=client,
            max_concurrent_documents=max_concurrent_documents,
        )

        results = await orchestrator.process_documents(
            documents=samples,
            process_fn=process_fn,
            progress_callback=progress_callback,
        )

        logger.info(f"Batch processing complete: {client.stats}")
        return results
