"""
Batched GenRM Client for High-Throughput Preference Collection.

This module provides an async batch client for GenRM requests, optimized for
the 235B GenRM model with semaphore-controlled concurrent requests.

Design: Uses simple semaphore-based concurrency control that works with
asyncio.run() patterns (no background worker needed).

Usage:
    from src.training.preference.genrm_batch import AsyncBatchGenRMClient

    client = AsyncBatchGenRMClient(base_url="http://localhost:8001/v1")
    result = await client.call(GenRMComparisonRequest(...))
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import aiohttp

from src.config.concurrency import get_concurrency_config
from .genrm import GenRMResult, GenRMErrorResult, GenRMComparisonResult

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Types
# =============================================================================

@dataclass
class GenRMComparisonRequest:
    """A single GenRM comparison request."""
    request_id: str
    context: str
    original_text: str
    summary_a: str
    summary_b: str
    law_type: str = "sufficiency"
    extra_context: Optional[str] = None


@dataclass
class GenRMBatchStats:
    """Statistics for GenRM batch processing."""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    wall_clock_start: float = 0.0
    wall_clock_end: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return self.total_latency_ms / self.completed_requests

    @property
    def wall_clock_seconds(self) -> float:
        if self.wall_clock_start == 0:
            return 0.0
        end_time = self.wall_clock_end if self.wall_clock_end > 0 else time.time()
        return end_time - self.wall_clock_start

    @property
    def tokens_per_second(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_tokens / self.wall_clock_seconds

    @property
    def requests_per_second(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.completed_requests / self.wall_clock_seconds

    def __str__(self) -> str:
        return (
            f"GenRMBatchStats(reqs={self.completed_requests}/{self.total_requests}, "
            f"tokens={self.total_tokens:,}, "
            f"tok/s={self.tokens_per_second:.0f}, "
            f"req/s={self.requests_per_second:.2f})"
        )


# =============================================================================
# Async Batch GenRM Client
# =============================================================================

class AsyncBatchGenRMClient:
    """
    Async client for batched GenRM comparison requests.

    Uses semaphore-controlled concurrency to send requests to the GenRM server.
    The vLLM server handles internal batching for optimal GPU utilization.

    Optimized for the 235B GenRM model with:
    - Lower concurrency (50 vs 200) to avoid OOM
    - Longer timeouts (600s) for generation

    This client auto-initializes per async context (works with asyncio.run()).
    """

    # Class-level cache for model names per server URL
    _model_cache: Dict[str, str] = {}

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        max_concurrent: int = 50,
        model: Optional[str] = None,
        request_timeout: float = 600.0,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 16384,
    ):
        """
        Initialize async batch GenRM client.

        Args:
            base_url: GenRM server URL (default port 8001)
            max_concurrent: Maximum concurrent HTTP requests
            model: Model name for vLLM (auto-detected if None)
            request_timeout: Per-request HTTP timeout in seconds
            temperature: Generation temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens for response
        """
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self._model = model
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Session management (shared when using context manager)
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._pending_futures: Dict[str, asyncio.Future] = {}

        # Task tracking for proper cleanup (prevents orphaned tasks)
        self._active_tasks: Set[asyncio.Task] = set()

        # Cumulative statistics
        self.stats = GenRMBatchStats()

    @property
    def model(self) -> str:
        """Get model name (auto-detected if not set)."""
        return self._model or "unknown"

    async def __aenter__(self) -> "AsyncBatchGenRMClient":
        """Async context manager entry - creates shared session for efficient reuse."""
        self._connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=aiohttp.ClientTimeout(total=self.request_timeout)
        )
        await self._ensure_model_detected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit - cleans up all resources."""
        await self.close()
        return False

    async def _ensure_model_detected(self):
        """Ensure model name is detected."""
        if self._model is None:
            self._model = await self._detect_model()
        if self.stats.wall_clock_start == 0:
            self.stats.wall_clock_start = time.time()

    async def _detect_model(self) -> str:
        """Auto-detect model name from GenRM server."""
        # Check class-level cache first
        if self.base_url in AsyncBatchGenRMClient._model_cache:
            return AsyncBatchGenRMClient._model_cache[self.base_url]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    if data.get("data") and len(data["data"]) > 0:
                        model_id = data["data"][0]["id"]
                        AsyncBatchGenRMClient._model_cache[self.base_url] = model_id
                        logger.info(f"Auto-detected GenRM model: {model_id}")
                        return model_id
        except Exception as e:
            logger.warning(f"Failed to auto-detect GenRM model: {e}")

        return "unknown"

    async def call(self, request: GenRMComparisonRequest) -> GenRMComparisonResult:
        """
        Send a single comparison request.

        Uses shared session if available (context manager), otherwise creates
        a fresh session per call for compatibility with asyncio.run() patterns.
        """
        # Auto-detect model if not specified (uses cached value)
        if self._model is None:
            self._model = await self._detect_model()

        # Use shared session if available (context manager pattern)
        if self._session and not self._session.closed:
            return await self._send_single_with_session(request, self._session)

        # Fallback: Create a fresh session for this call (properly closed at end)
        # This is inefficient - recommend using context manager for production
        logger.warning("GenRM client used without context manager - creating per-call session (inefficient)")
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            return await self._send_single_with_session(request, session)

    async def submit(self, request: GenRMComparisonRequest) -> str:
        """
        Submit a comparison request and store its future.

        Use await_response() to get the result later.
        Returns the request_id.

        Tasks are tracked in _active_tasks for proper cleanup on close().
        """
        await self._ensure_model_detected()

        # Create future and start the request
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_futures[request.request_id] = future

        # Start the request in background - TRACK the task for cleanup
        task = asyncio.create_task(self._send_and_resolve(request, future))
        self._active_tasks.add(task)
        # Auto-remove from tracking set when task completes
        task.add_done_callback(self._active_tasks.discard)
        self.stats.total_requests += 1

        return request.request_id

    async def await_response(
        self,
        request_id: str,
        timeout: float = 600.0,
    ) -> GenRMComparisonResult:
        """Wait for a submitted request to complete."""
        if request_id not in self._pending_futures:
            raise KeyError(f"Unknown request_id: {request_id}")

        future = self._pending_futures[request_id]

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            del self._pending_futures[request_id]
            return response
        except asyncio.TimeoutError:
            logger.error(f"GenRM request {request_id} timed out after {timeout:.0f}s")
            del self._pending_futures[request_id]
            return GenRMErrorResult(
                error_type="timeout",
                error_message=f"Timeout after {timeout}s",
            )

    async def _send_and_resolve(
        self,
        request: GenRMComparisonRequest,
        future: asyncio.Future,
    ):
        """Send request and resolve its future.

        Uses shared session if available, otherwise creates a fresh session.
        """
        try:
            # Use shared session if available (context manager pattern)
            if self._session and not self._session.closed:
                result = await self._send_single_with_session(request, self._session)
            else:
                # Fallback: fresh session per request
                connector = aiohttp.TCPConnector(limit=self.max_concurrent)
                timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    result = await self._send_single_with_session(request, session)

            if not future.done():
                future.set_result(result)
        except asyncio.CancelledError:
            # Task was cancelled - cancel the future so callers fail fast
            # instead of waiting for their full timeout
            if not future.done():
                future.cancel()
            raise
        except Exception as e:
            if not future.done():
                future.set_result(GenRMErrorResult(
                    error_type="network",
                    error_message=str(e),
                ))

    def _build_messages(self, request: GenRMComparisonRequest) -> List[Dict[str, str]]:
        """Build GenRM-format messages for a comparison request."""
        law_instructions = {
            "sufficiency": (
                "Compare which summary better preserves the oracle-relevant "
                "information from the original text."
            ),
            "idempotence": (
                "Compare which summary is more stable under re-summarization."
            ),
            "merge": (
                "Compare which merged summary better preserves the information "
                "from its child summaries."
            ),
        }
        instruction = law_instructions.get(request.law_type, law_instructions["sufficiency"])

        original_section = ""
        if request.original_text.strip():
            original_section = f"\n\nOriginal Text:\n{request.original_text}"

        extra_section = ""
        if request.extra_context:
            extra_section = f"\n\nAdditional Context:\n{request.extra_context}"

        user_message = (
            "Please compare the following two candidate summaries.\n"
            f"OPS law: {request.law_type}\n"
            f"{instruction}\n\n"
            f"Context (what to preserve): {request.context}"
            f"{original_section}"
            f"{extra_section}\n\n"
            "Evaluate the candidates below on:\n"
            "1. Preservation of oracle-relevant information\n"
            "2. Accuracy and faithfulness\n"
            "3. Completeness vs. conciseness tradeoff"
        )

        return [
            {"role": "user", "content": user_message},
            {"role": "response_1", "content": request.summary_a},
            {"role": "response_2", "content": request.summary_b},
        ]

    def _parse_genrm_response(self, content: str) -> GenRMResult:
        """Parse GenRM response to extract scores and preference."""
        helpfulness_a = 3.0
        helpfulness_b = 3.0
        ranking_score = 3
        reasoning = ""

        # Try to parse JSON output format first
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{[^{}]*"score_1"[^{}]*\})', content, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                helpfulness_a = float(result.get('score_1', 3))
                helpfulness_b = float(result.get('score_2', 3))
                ranking_score = int(result.get('ranking', 3))
                reasoning = result.get('response_1_analysis', '') + '\n' + result.get('response_2_analysis', '')
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug(f"JSON parsing failed, falling back to regex: {e}")

        # Fallback: regex-based extraction
        if helpfulness_a == 3.0 and helpfulness_b == 3.0 and ranking_score == 3:
            numbers = re.findall(r"\b([1-5](?:\.[0-9])?)\b", content)
            if len(numbers) >= 2:
                try:
                    helpfulness_a = float(numbers[0])
                    helpfulness_b = float(numbers[1])
                except ValueError:
                    pass

            ranking_pattern = r"(?:ranking|overall|preference)[^\d]*([1-6])"
            ranking_match = re.search(ranking_pattern, content, re.IGNORECASE)
            if ranking_match:
                ranking_score = int(ranking_match.group(1))
            else:
                if helpfulness_a > helpfulness_b + 0.5:
                    ranking_score = 2
                elif helpfulness_b > helpfulness_a + 0.5:
                    ranking_score = 5
                else:
                    ranking_score = 3

        # Determine preference from ranking score
        if ranking_score <= 2:
            preferred = "A"
            confidence = (3 - ranking_score) * 0.3 + 0.4
        elif ranking_score >= 5:
            preferred = "B"
            confidence = (ranking_score - 4) * 0.3 + 0.4
        else:
            preferred = "tie"
            confidence = 0.5

        return GenRMResult(
            preferred=preferred,
            ranking_score=ranking_score,
            helpfulness_a=helpfulness_a,
            helpfulness_b=helpfulness_b,
            reasoning=reasoning if reasoning else content,
            confidence=confidence,
            raw_response=content,
        )

    async def _send_single_with_session(
        self,
        request: GenRMComparisonRequest,
        session: aiohttp.ClientSession,
    ) -> GenRMComparisonResult:
        """Send a single request using the provided session."""
        start_time = time.time()
        try:
            messages = self._build_messages(request)
            payload = {
                "model": self._model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY"}
            ) as resp:
                data = await resp.json()
                latency = (time.time() - start_time) * 1000

                if resp.status == 200:
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})

                    result = self._parse_genrm_response(content)

                    self.stats.completed_requests += 1
                    self.stats.total_latency_ms += latency
                    self.stats.total_tokens += usage.get("total_tokens", 0)
                    self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.stats.completion_tokens += usage.get("completion_tokens", 0)

                    return result
                else:
                    self.stats.failed_requests += 1
                    return GenRMErrorResult(
                        error_type="server_error",
                        error_message=f"HTTP {resp.status}: {data}",
                    )

        except asyncio.TimeoutError:
            self.stats.failed_requests += 1
            logger.error(f"GenRM request {request.request_id} timed out")
            return GenRMErrorResult(
                error_type="timeout",
                error_message=f"Request timeout after {self.request_timeout}s",
            )
        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"GenRM request {request.request_id} failed: {e}")
            return GenRMErrorResult(
                error_type="network",
                error_message=str(e),
            )

    async def close(self, timeout: float = None):
        """Close the client, cancelling all pending tasks and closing the session.

        Args:
            timeout: Maximum time to wait for tasks to cancel.
                     Defaults to config.task_cancel_timeout (30s).
        """
        config = get_concurrency_config()
        timeout = timeout if timeout is not None else config.task_cancel_timeout
        self.stats.wall_clock_end = time.time()

        # Force-cancel all active tasks first
        if self._active_tasks:
            num_tasks = len(self._active_tasks)
            logger.debug(f"Force-cancelling {num_tasks} active tasks...")

            for task in self._active_tasks:
                if not task.done():
                    task.cancel()

            # Wait for cancellation to complete (with timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=timeout
                )
                logger.debug(f"Successfully cancelled {num_tasks} tasks")
            except asyncio.TimeoutError:
                # Tasks didn't respond to cancellation - log and continue cleanup
                remaining = sum(1 for t in self._active_tasks if not t.done())
                logger.warning(
                    f"Timeout ({timeout}s) waiting for task cancellation. "
                    f"{remaining}/{num_tasks} tasks may still be running."
                )
                # Force clear anyway - we've done our best
                # The tasks will be garbage collected eventually

        # Clear tracking collections
        self._active_tasks.clear()
        self._pending_futures.clear()

        # Close the session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        # Close the connector
        if self._connector and not self._connector.closed:
            await self._connector.close()
            self._connector = None

        logger.info(f"GenRM batch client closed. Stats: {self.stats}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_genrm_batch_client(
    base_url: str = "http://localhost:8001/v1",
    max_concurrent: int = 50,
    batch_size: int = 10,  # Unused but kept for API compatibility
    batch_timeout: float = 0.2,  # Unused but kept for API compatibility
    temperature: float = 0.6,
    max_tokens: int = 16384,
    **kwargs,
) -> AsyncBatchGenRMClient:
    """
    Create an AsyncBatchGenRMClient with recommended settings for the 235B model.

    Args:
        base_url: GenRM server URL
        max_concurrent: Maximum concurrent requests (default 50 for large model)
        batch_size: (unused, kept for API compatibility)
        batch_timeout: (unused, kept for API compatibility)
        temperature: Generation temperature
        max_tokens: Maximum response tokens

    Returns:
        Configured AsyncBatchGenRMClient
    """
    return AsyncBatchGenRMClient(
        base_url=base_url,
        max_concurrent=max_concurrent,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
