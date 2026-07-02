"""
Human preference collector.

Enqueues PreferenceRequests into a PreferenceStore for human review via the API.
Bridges to the existing ReviewQueue/FlaggedItem infrastructure.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from src.preference_collection.collector import register_collector
from src.preference_collection.types import PreferenceRequest, PreferenceResponse

logger = logging.getLogger(__name__)


@register_collector("human")
class HumanCollector:
    """Collector that queues requests for human review.

    Has two modes:
    - **Non-blocking** (default): enqueues and returns a pending response immediately.
      The actual response arrives later via the API.
    - **Blocking**: waits (polls) until a human submits a response via the store.

    Usage:
        from src.preference_collection.store import PreferenceStore

        store = PreferenceStore()
        collector = HumanCollector(store=store, blocking=False)

        # Non-blocking: enqueue and get a placeholder
        response = collector.collect(request)
        assert response.source == "human_pending"

        # Later, human submits via API -> store.submit(request_id, response)
        # Retrieve completed:
        dataset = store.to_preference_dataset()
    """

    def __init__(
        self,
        store: Optional[Any] = None,
        blocking: bool = False,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ):
        """
        Args:
            store: PreferenceStore instance. If None, creates one.
            blocking: If True, collect() blocks until a response is submitted.
            poll_interval: Seconds between polls in blocking mode.
            timeout: Maximum seconds to wait in blocking mode.
        """
        if store is None:
            from src.preference_collection.store import PreferenceStore
            store = PreferenceStore()
        self.store = store
        self.blocking = blocking
        self.poll_interval = poll_interval
        self.timeout = timeout

    def collect(
        self,
        request: PreferenceRequest,
        **kwargs: Any,
    ) -> PreferenceResponse:
        """Enqueue request for human review.

        In non-blocking mode: returns a pending placeholder immediately.
        In blocking mode: polls until a response is submitted or timeout.
        """
        self.store.enqueue(request)

        if not self.blocking:
            return PreferenceResponse(
                request_id=request.request_id,
                reasoning="Awaiting human review",
                source="human_pending",
                extra={"status": "pending"},
            )

        # Blocking: poll for response
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            completed = self.store.get_completed(limit=10000)
            for req, resp in completed:
                if req.request_id == request.request_id:
                    return resp
            time.sleep(self.poll_interval)

        # Timeout
        return PreferenceResponse(
            request_id=request.request_id,
            reasoning=f"Human review timed out after {self.timeout}s",
            source="human_timeout",
            extra={"status": "timeout"},
        )

    async def collect_async(
        self,
        request: PreferenceRequest,
        **kwargs: Any,
    ) -> PreferenceResponse:
        """Async version: enqueue and optionally wait."""
        self.store.enqueue(request)

        if not self.blocking:
            return PreferenceResponse(
                request_id=request.request_id,
                reasoning="Awaiting human review",
                source="human_pending",
                extra={"status": "pending"},
            )

        # Async polling
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            completed = self.store.get_completed(limit=10000)
            for req, resp in completed:
                if req.request_id == request.request_id:
                    return resp
            await asyncio.sleep(self.poll_interval)

        return PreferenceResponse(
            request_id=request.request_id,
            reasoning=f"Human review timed out after {self.timeout}s",
            source="human_timeout",
            extra={"status": "timeout"},
        )
