"""Factory for the shared OpenAI-compatible batch client stack.

The repo has several thin adapters over the same batching substrate. Keep the
single-vs-multi endpoint choice here so DSPy, inference engines, and pipelines
do not drift in routing, timeout, or recovery behavior.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from src.core.batch_processor import (
    AsyncBatchLLMClient,
    MultiServerBatchClient,
    parse_routing_policy,
)
from src.core.batch_transport import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
    normalize_base_urls,
)


def build_batch_client(
    *,
    server_urls: Optional[Sequence[Any]] = None,
    base_url: Optional[Any] = None,
    api_base: Optional[Any] = None,
    api_bases: Optional[Sequence[Any]] = None,
    model: Optional[str] = None,
    api_key: str = "EMPTY",
    max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_timeout: float = DEFAULT_BATCH_TIMEOUT_SECONDS,
    request_timeout: float = DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    routing_policy: str = DEFAULT_BATCH_ROUTING_POLICY,
    recover_base_url_callback: Any = None,
    recovery_cooldown_seconds: Optional[float] = None,
    metrics_collector: Any = None,
    call_sink: Any = None,
) -> AsyncBatchLLMClient | MultiServerBatchClient:
    """Build the canonical batch client for one or more chat endpoints."""

    urls = normalize_base_urls(
        api_base=api_base if api_base is not None else base_url,
        api_bases=api_bases if api_bases is not None else server_urls,
    )
    if not urls:
        raise ValueError("build_batch_client requires at least one endpoint URL.")

    resolved_model = None if model in {None, "", "default"} else str(model)
    cooldown_seconds = 120.0 if recovery_cooldown_seconds is None else float(recovery_cooldown_seconds)
    if len(urls) > 1:
        return MultiServerBatchClient(
            servers=urls,
            max_concurrent_per_server=max(1, int(max_concurrent)),
            batch_size=max(1, int(batch_size)),
            batch_timeout=max(0.0, float(batch_timeout)),
            request_timeout=max(1.0, float(request_timeout)),
            model=resolved_model,
            api_key=str(api_key or "EMPTY"),
            recover_base_url_callback=recover_base_url_callback,
            recovery_cooldown_seconds=cooldown_seconds,
            routing_policy=parse_routing_policy(routing_policy).value,
            metrics_collector=metrics_collector,
            call_sink=call_sink,
        )

    return AsyncBatchLLMClient(
        base_url=urls[0],
        max_concurrent=max(1, int(max_concurrent)),
        batch_size=max(1, int(batch_size)),
        batch_timeout=max(0.0, float(batch_timeout)),
        model=resolved_model,
        request_timeout=max(1.0, float(request_timeout)),
        api_key=str(api_key or "EMPTY"),
        recover_base_url_callback=recover_base_url_callback,
        recovery_cooldown_seconds=cooldown_seconds,
        call_sink=call_sink,
    )


__all__ = ["build_batch_client"]
