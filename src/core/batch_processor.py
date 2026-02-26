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
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Awaitable, Tuple, Union, Set
import aiohttp

from src.preprocessing.chunker import chunk_for_ops
from src.core.data_models import Node
from src.config.constants import LOG_TRUNCATE_LENGTH
from src.core.async_utils import cancel_tasks

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
    cache_key: Optional[str] = None


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
    cache_hits: int = 0
    cache_misses: int = 0
    cache_writes: int = 0
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
        api_key: str = "EMPTY",  # vLLM/SGLang use "EMPTY"; set real key for OpenAI
        recover_base_url_callback: Optional[Callable[[str], bool]] = None,
        recovery_cooldown_seconds: float = 120.0,
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
            api_key: API key for Authorization header (default "EMPTY" for local servers)
            recover_base_url_callback: Optional callback(base_url)->bool to auto-recover
                failed servers (e.g. orchestrator restart/wake).
            recovery_cooldown_seconds: Cooldown between recovery attempts per base_url.
        """
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._model = model  # Will be set during start() if None
        self.request_timeout = request_timeout
        self.api_key = api_key
        self.recover_base_url_callback = recover_base_url_callback
        self.recovery_cooldown_seconds = max(0.0, float(recovery_cooldown_seconds))
        self._last_recovery_attempt: float = 0.0
        self._recovery_lock: Optional[asyncio.Lock] = None

        # Optional disk-backed response cache (opt-in via env vars).
        # This is a pragmatic analogue of "persistent KV" for repeated
        # document reruns: skip identical LLM calls entirely.
        #
        # Env vars:
        #   TT_RESPONSE_CACHE_DIR=/path/to/cache
        #   TT_RESPONSE_CACHE_MODE=off|read|write|readwrite   (default: off)
        #   TT_RESPONSE_CACHE_REQUEST_TYPES=summarize,merge   (optional filter)
        self._response_cache = None
        self._response_cache_mode = "off"
        self._response_cache_request_types: Optional[Set[str]] = None
        cache_dir = str(os.getenv("TT_RESPONSE_CACHE_DIR", "") or "").strip()
        if cache_dir:
            mode = str(os.getenv("TT_RESPONSE_CACHE_MODE", "") or "").strip().lower()
            if mode in {"off", "read", "write", "readwrite"}:
                self._response_cache_mode = mode
            elif mode:
                self._response_cache_mode = "readwrite"
            else:
                self._response_cache_mode = "readwrite"
            try:
                from pathlib import Path

                from src.core.response_cache import FileResponseCache

                self._response_cache = FileResponseCache(Path(cache_dir))
            except Exception:
                self._response_cache = None
                self._response_cache_mode = "off"

        raw_types = str(os.getenv("TT_RESPONSE_CACHE_REQUEST_TYPES", "") or "").strip()
        if raw_types:
            selected = {part.strip() for part in raw_types.split(",") if part.strip()}
            self._response_cache_request_types = selected or None

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
        self._active_batch_tasks: Set[asyncio.Task] = set()
        self._max_inflight_batches = max(1, math.ceil(self.max_concurrent / max(1, self.batch_size)))

    @property
    def model(self) -> str:
        """Get model name (auto-detected if not set)."""
        return self._model or "unknown"

    def _response_cache_allows(self, request: BatchRequest) -> bool:
        if self._response_cache is None or self._response_cache_mode == "off":
            return False
        if self._response_cache_request_types is None:
            return True
        return str(request.request_type) in self._response_cache_request_types

    async def _detect_model(self) -> str:
        """Auto-detect model name from vLLM server."""
        from src.core.model_detection import detect_model_async
        return await detect_model_async(self.base_url, fallback="default")

    def _handle_request_error(
        self,
        request: BatchRequest,
        error_msg: str,
    ) -> None:
        """Handle request errors consistently.

        Args:
            request: The failed request
            error_msg: Error message to include in the response
        """
        logger.error(f"Request {request.request_id} failed: {error_msg}")
        self.stats.failed_requests += 1
        if request.future and not request.future.done():
            request.future.set_result(BatchResponse(
                request_id=request.request_id,
                content="",
                error=error_msg,
            ))

    async def start(self):
        """Start the batch processor."""
        if self._running:
            return

        # Auto-detect model if not specified
        if self._model is None:
            self._model = await self._detect_model()

        self._request_queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._recovery_lock = asyncio.Lock()
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

        # Clean up any pending futures to prevent memory leaks
        # This handles cases where submit() was called but await_response() was not
        if self._pending_futures:
            orphaned = len(self._pending_futures)
            for request_id, future in list(self._pending_futures.items()):
                if not future.done():
                    future.set_result(BatchResponse(
                        request_id=request_id,
                        content="",
                        error="Batch client stopped",
                    ))
            self._pending_futures.clear()
            if orphaned > 0:
                logger.debug(f"Cleaned up {orphaned} orphaned futures on stop")

        if self._active_batch_tasks:
            await cancel_tasks(self._active_batch_tasks)
            self._active_batch_tasks.clear()
        if self._request_queue:
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
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
        request.future = asyncio.get_running_loop().create_future()
        request.submitted_at = time.time()
        self._pending_futures[request.request_id] = request.future
        self.stats.total_requests += 1

        # Optional disk cache short-circuit.
        if self._response_cache_allows(request):
            try:
                from src.core.response_cache import make_chat_cache_key

                request.cache_key = make_chat_cache_key(
                    model=self.model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    extra={"request_type": request.request_type},
                )
            except Exception:
                request.cache_key = None

            if request.cache_key and self._response_cache_mode in {"read", "readwrite"}:
                cached = self._response_cache.get(request.cache_key)
                if cached is not None:
                    self.stats.cache_hits += 1
                    response = BatchResponse(
                        request_id=request.request_id,
                        content=cached.content,
                        usage=cached.usage,
                        error=None,
                        latency_ms=0.0,
                    )
                    self.stats.completed_requests += 1
                    self.stats.total_tokens += int(cached.usage.get("total_tokens", 0) or 0)
                    self.stats.prompt_tokens += int(cached.usage.get("prompt_tokens", 0) or 0)
                    self.stats.completion_tokens += int(cached.usage.get("completion_tokens", 0) or 0)
                    if request.future and not request.future.done():
                        request.future.set_result(response)
                    return request.request_id
                self.stats.cache_misses += 1

        # Add to queue
        await self._request_queue.put(request)

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
                    # Throttle number of in-flight batches to avoid task buildup
                    if len(self._active_batch_tasks) >= self._max_inflight_batches:
                        done, _ = await asyncio.wait(
                            self._active_batch_tasks,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            if task.exception():
                                logger.debug(f"Batch task error: {task.exception()}")

                    task = asyncio.create_task(self._send_batch(batch))
                    self._active_batch_tasks.add(task)
                    task.add_done_callback(self._active_batch_tasks.discard)
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
            payload = {
                "model": self._model,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            max_attempts = 2  # Initial try + one retry after recovery.
            for attempt in range(max_attempts):
                try:
                    async with self._session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    ) as resp:
                        latency = (time.time() - start_time) * 1000
                        if resp.status == 200:
                            data = await resp.json()
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
                            if (
                                self._response_cache is not None
                                and self._response_cache_mode in {"write", "readwrite"}
                                and self._response_cache_allows(request)
                            ):
                                try:
                                    from src.core.response_cache import CachedChatResponse, FileResponseCache

                                    cache_key = request.cache_key
                                    if not cache_key:
                                        from src.core.response_cache import make_chat_cache_key

                                        cache_key = make_chat_cache_key(
                                            model=self.model,
                                            messages=request.messages,
                                            max_tokens=request.max_tokens,
                                            temperature=request.temperature,
                                            extra={"request_type": request.request_type},
                                        )
                                    if cache_key:
                                        self._response_cache.set(
                                            cache_key,
                                            CachedChatResponse(
                                                content=content,
                                                usage={str(k): int(v) for k, v in dict(usage or {}).items()},
                                                model=str(data.get("model") or self.model),
                                                created_at=FileResponseCache.now_iso(),
                                            ),
                                        )
                                        self.stats.cache_writes += 1
                                except Exception:
                                    pass
                            if request.future and not request.future.done():
                                request.future.set_result(response)
                            return

                        # Non-200 response.
                        body_text = await resp.text()
                        error_msg = f"HTTP {resp.status}: {body_text[:LOG_TRUNCATE_LENGTH]}"
                        recoverable_status = resp.status in {408, 429, 500, 502, 503, 504}
                        if attempt < (max_attempts - 1) and recoverable_status:
                            recovered = await self._maybe_recover_server(
                                reason=f"http_{resp.status}"
                            )
                            if recovered:
                                logger.warning(
                                    "Recovered %s after %s; retrying request %s",
                                    self.base_url,
                                    error_msg.split(":", 1)[0],
                                    request.request_id,
                                )
                                continue

                        response = BatchResponse(
                            request_id=request.request_id,
                            content="",
                            error=error_msg,
                            latency_ms=latency,
                        )
                        self.stats.failed_requests += 1
                        if request.future and not request.future.done():
                            request.future.set_result(response)
                        return

                except aiohttp.ClientError as e:
                    if attempt < (max_attempts - 1):
                        recovered = await self._maybe_recover_server(reason=type(e).__name__)
                        if recovered:
                            logger.warning(
                                "Recovered %s after %s; retrying request %s",
                                self.base_url,
                                type(e).__name__,
                                request.request_id,
                            )
                            continue
                    self._handle_request_error(
                        request,
                        f"{type(e).__name__}: {str(e) or 'Connection failed'}",
                    )
                    return
                except asyncio.TimeoutError:
                    if attempt < (max_attempts - 1):
                        recovered = await self._maybe_recover_server(reason="timeout")
                        if recovered:
                            logger.warning(
                                "Recovered %s after timeout; retrying request %s",
                                self.base_url,
                                request.request_id,
                            )
                            continue
                    self._handle_request_error(request, "Request timed out")
                    return
                except Exception as e:
                    self._handle_request_error(
                        request,
                        f"{type(e).__name__}: {str(e) or 'Unknown error'}",
                    )
                    return

    async def _maybe_recover_server(self, *, reason: str) -> bool:
        """Run recovery callback at most once per cooldown window."""
        if self.recover_base_url_callback is None:
            return False

        now = time.monotonic()
        if self._recovery_lock is None:
            self._recovery_lock = asyncio.Lock()
        async with self._recovery_lock:
            if (now - self._last_recovery_attempt) < self.recovery_cooldown_seconds:
                return False
            self._last_recovery_attempt = now

        try:
            logger.warning(
                "Attempting batch-client server recovery for %s (%s)",
                self.base_url,
                reason,
            )
            recovered = await asyncio.to_thread(self.recover_base_url_callback, self.base_url)
        except Exception as exc:
            logger.warning("Batch-client server recovery callback failed for %s: %s", self.base_url, exc)
            return False

        if recovered:
            logger.info("Batch-client server recovery succeeded for %s", self.base_url)
            return True
        logger.warning("Batch-client server recovery reported failure for %s", self.base_url)
        return False


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
        api_key: str = "EMPTY",
        recover_base_url_callback: Optional[Callable[[str], bool]] = None,
        recovery_cooldown_seconds: float = 120.0,
    ):
        """
        Initialize multi-server client.

        Args:
            servers: List of vLLM server URLs
            max_concurrent_per_server: Max concurrent requests per server
            batch_size: Requests per batch
            batch_timeout: Max wait to fill batch
            api_key: API key for Authorization header
        """
        self.servers = servers
        self.clients: List[AsyncBatchLLMClient] = []
        self._counter = 0  # Round-robin counter
        self._lock: Optional[asyncio.Lock] = None  # Created in start()
        self._request_client_map: Dict[str, AsyncBatchLLMClient] = {}  # request_id -> client (O(1) lookup)

        # Create a client for each server
        for server_url in servers:
            client = AsyncBatchLLMClient(
                base_url=server_url,
                max_concurrent=max_concurrent_per_server,
                batch_size=batch_size,
                batch_timeout=batch_timeout,
                api_key=api_key,
                recover_base_url_callback=recover_base_url_callback,
                recovery_cooldown_seconds=recovery_cooldown_seconds,
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
                    logger.error(f"  ✗ Doc {idx}: {str(result)[:LOG_TRUNCATE_LENGTH]}")
                    results[idx] = None
                else:
                    results[idx] = result
                    # Show per-doc status for small batches (domain-agnostic)
                    if show_per_doc:
                        # Use generic doc_id
                        doc_id = getattr(result, 'doc_id', None)
                        if doc_id:
                            # Get predicted/truth scores using canonical field names
                            pred = getattr(result, 'estimated_score', None)
                            truth = getattr(result, 'reference_score', None)
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
# Convenience Functions
# =============================================================================

def run_batched(coro):
    """Run an async coroutine from sync code.

    Note: Prefer using asyncio.run() directly in new code.
    This function exists for backwards compatibility.
    """
    return asyncio.run(coro)


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
