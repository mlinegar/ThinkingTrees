"""
FastAPI server for the ThinkingTrees preference collection system.

Serves PreferenceRequests to human or programmatic reviewers and accepts
PreferenceResponses. Bridges to PreferenceStore for state management.

Endpoints:
    GET  /preferences/pending          -- pending requests (paginated, priority-sorted)
    GET  /preferences/{request_id}     -- single request with context
    POST /preferences/{request_id}     -- submit preference response
    POST /preferences/batch            -- submit multiple responses
    GET  /preferences/export/supervision       -- completed items as canonical supervision
    GET  /preferences/export/binary_projection -- completed items as binary optimizer view
    GET  /preferences/stats            -- queue statistics
    GET  /health                    -- health check

Usage:
    # Start the server
    python -m src.preference_collection.server --port 8100

    # Or programmatically
    from src.preference_collection.server import create_app, get_store
    app = create_app(store=my_store)
    uvicorn.run(app, port=8100)

    # Submit preference via curl
    curl -X POST http://localhost:8100/preferences/req_1 \\
        -H "Content-Type: application/json" \\
        -d '{"preferred": "A", "confidence": 0.9, "reasoning": "Better preserves key points"}'
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Store singleton (set via create_app or module-level)
_store = None


def get_store():
    """Get or create the global PreferenceStore."""
    global _store
    if _store is None:
        from src.preference_collection.store import PreferenceStore
        _store = PreferenceStore()
    return _store


def set_store(store: Any) -> None:
    """Set the global PreferenceStore (for dependency injection)."""
    global _store
    _store = store


def create_app(store: Optional[Any] = None):
    """Create the FastAPI application.

    Args:
        store: Optional PreferenceStore. If None, creates a global one.

    Returns:
        FastAPI app instance.

    Raises:
        ImportError: If fastapi is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required for the preference server. "
            "Install with: pip install fastapi uvicorn"
        )

    if store is not None:
        set_store(store)

    app = FastAPI(
        title="ThinkingTrees Preference API",
        description="Collect human or programmatic preference on tree node summaries.",
        version="0.1.0",
    )

    # --- Pydantic models for request/response validation ---

    class DimensionSchema(BaseModel):
        kind: str
        name: Optional[str] = None
        scale: Optional[List[float]] = None
        options: Dict[str, Any] = Field(default_factory=dict)

    class PreferenceRequestSchema(BaseModel):
        request_id: str
        text_a: str = ""
        text_b: Optional[str] = None
        original_text: str = ""
        rubric: str = ""
        reference_score: Optional[float] = None
        dimensions: List[DimensionSchema] = Field(default_factory=list)
        node_id: Optional[str] = None
        tree_id: Optional[str] = None
        source_doc_id: Optional[str] = None
        law_type: str = "sufficiency"
        doc_propensity: float = 1.0
        node_propensity: float = 1.0
        label_propensity: float = 1.0
        sampling_scheme: Optional[str] = None
        priority: int = 0
        context: Dict[str, Any] = Field(default_factory=dict)
        created_at: Optional[str] = None

    class PreferenceResponseSchema(BaseModel):
        preferred: Optional[str] = None
        scores: Dict[str, float] = Field(default_factory=dict)
        critique: str = ""
        reasoning: str = ""
        confidence: float = 0.5
        score_estimate_a: Optional[float] = None
        score_estimate_b: Optional[float] = None
        extra: Dict[str, Any] = Field(default_factory=dict)
        source: str = "human"
        judge_model: str = ""

    class BatchResponseItem(BaseModel):
        request_id: str
        response: PreferenceResponseSchema

    class StatsResponse(BaseModel):
        pending: int
        completed: int
        total: int
        sources: Dict[str, int]

    # --- Endpoints ---

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/preferences/pending", response_model=List[PreferenceRequestSchema])
    async def get_pending(limit: int = 10, min_priority: int = 0):
        """Get pending preference requests, sorted by priority."""
        store = get_store()
        pending = store.get_pending(limit=limit, min_priority=min_priority)
        return [PreferenceRequestSchema(**req.to_dict()) for req in pending]

    @app.get("/preferences/{request_id}")
    async def get_request(request_id: str):
        """Get a specific pending preference request."""
        store = get_store()
        req = store.get_request(request_id)
        if req is None:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        return PreferenceRequestSchema(**req.to_dict())

    @app.post("/preferences/{request_id}")
    async def submit_response(request_id: str, response: PreferenceResponseSchema):
        """Submit a preference response for a pending request."""
        from src.preference_collection.types import PreferenceResponse as PR

        store = get_store()
        preference_response = PR(
            request_id=request_id,
            preferred=response.preferred,
            scores=response.scores,
            critique=response.critique,
            reasoning=response.reasoning,
            confidence=response.confidence,
            score_estimate_a=response.score_estimate_a,
            score_estimate_b=response.score_estimate_b,
            extra=response.extra,
            source=response.source,
            judge_model=response.judge_model,
        )
        success = store.submit(request_id, preference_response)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No pending request with id: {request_id}"
            )
        return {"status": "ok", "request_id": request_id}

    @app.post("/preferences/batch")
    async def submit_batch(items: List[BatchResponseItem]):
        """Submit multiple preference responses at once."""
        from src.preference_collection.types import PreferenceResponse as PR

        store = get_store()
        results = []
        for item in items:
            preference_response = PR(
                request_id=item.request_id,
                preferred=item.response.preferred,
                scores=item.response.scores,
                critique=item.response.critique,
                reasoning=item.response.reasoning,
                confidence=item.response.confidence,
                score_estimate_a=item.response.score_estimate_a,
                score_estimate_b=item.response.score_estimate_b,
                extra=item.response.extra,
                source=item.response.source,
                judge_model=item.response.judge_model,
            )
            success = store.submit(item.request_id, preference_response)
            results.append({
                "request_id": item.request_id,
                "status": "ok" if success else "not_found",
            })
        return {"results": results}

    @app.get("/preferences/stats", response_model=StatsResponse)
    async def get_stats():
        """Get preference queue statistics."""
        store = get_store()
        return StatsResponse(**store.get_statistics())

    @app.get("/preferences/export/supervision")
    async def export_supervision():
        """Export completed preference as canonical supervision records."""
        store = get_store()
        dataset = store.to_supervision_dataset()
        return {
            "summary": dataset.summary(),
            "response_judgments": [
                judgment.to_dict() for judgment in dataset.response_judgments
            ],
            "comparative_judgments": [
                record.to_dict() for record in dataset.comparative_judgments
            ],
        }

    @app.get("/preferences/export/binary_projection")
    async def export_binary_projection(projection: str = "adjacent"):
        """Export completed preference as the binary optimizer projection."""
        store = get_store()
        dataset = store.to_binary_projection_dataset(projection=projection)
        return dataset.to_dict()

    return app


# --- CLI entry point ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ThinkingTrees Preference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8100, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install uvicorn")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
