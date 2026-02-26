"""
FeedbackStore -- manages pending requests and completed responses.

Bridges the feedback API to the existing ReviewQueue infrastructure.
Provides the state backend for the FastAPI server and the HumanCollector.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.feedback.types import FeedbackDataset, FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)


class FeedbackStore:
    """In-memory + file-backed store for feedback requests and responses.

    Thread-safe via a lock for concurrent API access.

    Usage:
        store = FeedbackStore()

        # Enqueue a request
        store.enqueue(request)

        # List pending
        pending = store.get_pending(limit=10)

        # Submit a response
        store.submit("req_1", response)

        # Export completed
        dataset = store.to_feedback_dataset()
    """

    def __init__(
        self,
        review_queue: Optional[Any] = None,
        max_pending: int = 10000,
    ):
        """
        Args:
            review_queue: Optional ReviewQueue instance for bridging audit items.
            max_pending: Maximum pending requests before evicting lowest-priority.
        """
        self.review_queue = review_queue
        self.max_pending = max_pending

        self._lock = threading.Lock()
        self._pending: Dict[str, FeedbackRequest] = {}
        self._completed: Dict[str, Tuple[FeedbackRequest, FeedbackResponse]] = {}

    # --- Enqueue / Submit ---

    def enqueue(self, request: FeedbackRequest) -> str:
        """Add a feedback request to the pending queue.

        Returns:
            The request_id.
        """
        with self._lock:
            if len(self._pending) >= self.max_pending:
                self._evict_lowest_priority()
            self._pending[request.request_id] = request

            # Also add to ReviewQueue if it's a flagged-item-style request
            if self.review_queue is not None and request.context.get("approx_discrepancy") is not None:
                self._bridge_to_review_queue(request)

        logger.debug("Enqueued feedback request: %s", request.request_id)
        return request.request_id

    def submit(self, request_id: str, response: FeedbackResponse) -> bool:
        """Submit a response for a pending request.

        Returns:
            True if the request was found and response recorded.
        """
        with self._lock:
            request = self._pending.pop(request_id, None)
            if request is None:
                logger.warning("No pending request with id: %s", request_id)
                return False
            self._completed[request_id] = (request, response)

            # Bridge back to ReviewQueue
            if self.review_queue is not None:
                self._bridge_response_to_review_queue(request, response)

        logger.debug("Submitted feedback response for: %s", request_id)
        return True

    # --- Query ---

    def get_pending(
        self,
        limit: int = 10,
        min_priority: int = 0,
    ) -> List[FeedbackRequest]:
        """Get pending requests, sorted by priority (highest first).

        Args:
            limit: Maximum number of requests to return.
            min_priority: Minimum priority level.
        """
        with self._lock:
            items = [
                req for req in self._pending.values()
                if req.priority >= min_priority
            ]
        items.sort(key=lambda r: -r.priority)
        return items[:limit]

    def get_request(self, request_id: str) -> Optional[FeedbackRequest]:
        """Get a specific pending request by ID."""
        with self._lock:
            return self._pending.get(request_id)

    def get_completed(
        self,
        limit: int = 100,
    ) -> List[Tuple[FeedbackRequest, FeedbackResponse]]:
        """Get completed request/response pairs."""
        with self._lock:
            items = list(self._completed.values())
        return items[:limit]

    # --- Export ---

    def to_feedback_dataset(self) -> FeedbackDataset:
        """Export all completed items as a FeedbackDataset."""
        with self._lock:
            items = list(self._completed.values())
        return FeedbackDataset(items)

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            n_pending = len(self._pending)
            n_completed = len(self._completed)
            sources: Dict[str, int] = {}
            for _, resp in self._completed.values():
                sources[resp.source] = sources.get(resp.source, 0) + 1
        return {
            "pending": n_pending,
            "completed": n_completed,
            "total": n_pending + n_completed,
            "sources": sources,
        }

    # --- Persistence ---

    def save(self, path: Path) -> None:
        """Save store state to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "saved_at": datetime.now().isoformat(),
                "pending": {
                    rid: req.to_dict() for rid, req in self._pending.items()
                },
                "completed": {
                    rid: {
                        "request": req.to_dict(),
                        "response": resp.to_dict(),
                    }
                    for rid, (req, resp) in self._completed.items()
                },
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved feedback store to %s", path)

    def load(self, path: Path) -> None:
        """Load store state from JSON file (merges with existing)."""
        with open(path) as f:
            data = json.load(f)
        with self._lock:
            for rid, req_data in data.get("pending", {}).items():
                if rid not in self._pending:
                    self._pending[rid] = FeedbackRequest.from_dict(req_data)
            for rid, item_data in data.get("completed", {}).items():
                if rid not in self._completed:
                    req = FeedbackRequest.from_dict(item_data["request"])
                    resp = FeedbackResponse.from_dict(item_data["response"])
                    self._completed[rid] = (req, resp)
        logger.info("Loaded feedback store from %s", path)

    # --- ReviewQueue bridge ---

    def import_from_review_queue(self) -> int:
        """Import unreviewed FlaggedItems from the ReviewQueue as pending requests.

        Returns:
            Number of items imported.
        """
        if self.review_queue is None:
            return 0
        count = 0
        for item in self.review_queue.get_unreviewed_items():
            request = FeedbackRequest.from_flagged_item(item)
            if request.request_id not in self._pending:
                self.enqueue(request)
                count += 1
        logger.info("Imported %d items from ReviewQueue", count)
        return count

    def _bridge_to_review_queue(self, request: FeedbackRequest) -> None:
        """Add request to ReviewQueue as a FlaggedItem (for legacy compat)."""
        # Only bridge if the request came from an audit flagged item
        pass  # ReviewQueue.add() requires Node/AuditCheckResult objects;
        # full bridging is handled via import_from_review_queue()

    def _bridge_response_to_review_queue(
        self,
        request: FeedbackRequest,
        response: FeedbackResponse,
    ) -> None:
        """Update ReviewQueue item with the feedback response."""
        if self.review_queue is None:
            return
        # Extract the original FlaggedItem ID
        item_id = request.request_id
        if item_id.startswith("flag_"):
            item_id = item_id[len("flag_"):]
        item = self.review_queue.get_by_id(item_id)
        if item is None:
            return
        update = response.to_flagged_item_update()
        item.reviewed = update["reviewed"]
        item.review_result = update["review_result"]
        item.review_reasoning = update["review_reasoning"]
        item.corrected_summary = update.get("corrected_summary")
        item.reviewed_at = update["reviewed_at"]
        item.review_source = update["review_source"]
        self.review_queue.update_item(item)

    # --- Internal ---

    def _evict_lowest_priority(self) -> None:
        """Remove lowest-priority pending request to make room."""
        if not self._pending:
            return
        lowest = min(self._pending.values(), key=lambda r: r.priority)
        del self._pending[lowest.request_id]

    def clear(self) -> None:
        """Clear all pending and completed items."""
        with self._lock:
            self._pending.clear()
            self._completed.clear()
