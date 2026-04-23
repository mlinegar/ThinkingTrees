"""
DSPy configuration with XMLAdapter for robust output parsing.

This module provides centralized DSPy configuration that uses XMLAdapter
instead of the default ChatAdapter. XMLAdapter uses <field_name>value</field_name>
format which is more robust for parsing than the [[ ## field_name ## ]] format.

Also provides a unified LM factory for creating vLLM-backed DSPy language models.
"""

import atexit
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Any, Tuple, Sequence

import dspy
from dspy.adapters import XMLAdapter

from src.core.model_detection import detect_model_from_port, get_context_window_from_port
from src.config.context_window import ContextWindowManager, DEFAULT_CONTEXT_WINDOW

logger = logging.getLogger(__name__)


_xml_adapter: Optional[XMLAdapter] = None
_dspy_cache_runtime_signature: Optional[Tuple[bool, bool, str, int, int]] = None
_dspy_cache_cleanup_registered = False


class LoadBalancedLM(dspy.LM):
    """A DSPy LM that round-robins requests across multiple `api_base` URLs."""

    def __init__(
        self,
        *args: Any,
        api_bases: Sequence[str],
        cooldown_seconds: float = 30.0,
        **kwargs: Any,
    ) -> None:
        api_bases = [str(base).rstrip("/") for base in api_bases if str(base).strip()]
        if not api_bases:
            raise ValueError("LoadBalancedLM requires at least one api_base URL")

        deduped: list[str] = []
        seen = set()
        for base in api_bases:
            if base in seen:
                continue
            seen.add(base)
            deduped.append(base)

        self._api_bases = deduped
        self._rr_lock = threading.Lock()
        self._rr_next = 0
        self._unhealthy_until: dict[str, float] = {}
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))

        super().__init__(*args, api_base=self._api_bases[0], **kwargs)

    def _is_unhealthy(self, api_base: str, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        return self._unhealthy_until.get(api_base, 0.0) > now

    def _mark_unhealthy(self, api_base: str) -> None:
        if self._cooldown_seconds <= 0:
            return
        self._unhealthy_until[api_base] = time.monotonic() + self._cooldown_seconds

    def _mark_healthy(self, api_base: str) -> None:
        self._unhealthy_until.pop(api_base, None)

    def _pick_api_base(self) -> str:
        with self._rr_lock:
            n = len(self._api_bases)
            start = self._rr_next
            now = time.monotonic()
            for offset in range(n):
                idx = (start + offset) % n
                base = self._api_bases[idx]
                if not self._is_unhealthy(base, now=now):
                    self._rr_next = (idx + 1) % n
                    return base
            # Everything is currently marked unhealthy; fall back to round-robin anyway.
            base = self._api_bases[start]
            self._rr_next = (start + 1) % n
            return base

    def _ordered_api_bases(self) -> list[str]:
        first = self._pick_api_base()
        now = time.monotonic()
        with self._rr_lock:
            healthy = [
                base
                for base in self._api_bases
                if base != first and not self._is_unhealthy(base, now=now)
            ]
            unhealthy = [
                base
                for base in self._api_bases
                if base != first and self._is_unhealthy(base, now=now)
            ]
        return [first, *healthy, *unhealthy]

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "connection error" in message
            or "connection refused" in message
            or "connecterror" in message
            or "apiconnectionerror" in message
            or "failed to establish a new connection" in message
            or "temporary failure in name resolution" in message
            or "timed out" in message
            or "timeout" in message
        )

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        call_kwargs = dict(kwargs)
        bases = self._ordered_api_bases()
        for idx, api_base in enumerate(bases):
            call_kwargs["api_base"] = api_base
            try:
                response = super().forward(prompt=prompt, messages=messages, **call_kwargs)
                with self._rr_lock:
                    self._mark_healthy(api_base)
                return response
            except Exception as exc:
                if not self._is_connection_error(exc) or idx == len(bases) - 1:
                    raise
                with self._rr_lock:
                    self._mark_unhealthy(api_base)
                logger.warning(
                    "LM connection error via %s; retrying on alternate server (%d/%d): %s",
                    api_base,
                    idx + 1,
                    len(bases),
                    exc,
                )
        raise RuntimeError("Exhausted all api_base retry options")

    async def aforward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        call_kwargs = dict(kwargs)
        bases = self._ordered_api_bases()
        for idx, api_base in enumerate(bases):
            call_kwargs["api_base"] = api_base
            try:
                response = await super().aforward(prompt=prompt, messages=messages, **call_kwargs)
                with self._rr_lock:
                    self._mark_healthy(api_base)
                return response
            except Exception as exc:
                if not self._is_connection_error(exc) or idx == len(bases) - 1:
                    raise
                with self._rr_lock:
                    self._mark_unhealthy(api_base)
                logger.warning(
                    "Async LM connection error via %s; retrying on alternate server (%d/%d): %s",
                    api_base,
                    idx + 1,
                    len(bases),
                    exc,
                )
        raise RuntimeError("Exhausted all api_base retry options")


def get_xml_adapter() -> XMLAdapter:
    """
    Get or create a singleton XMLAdapter instance.

    Returns:
        XMLAdapter instance for use with dspy.configure()
    """
    global _xml_adapter
    if _xml_adapter is None:
        _xml_adapter = XMLAdapter()
    return _xml_adapter


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _close_cache_instance(cache_obj: Any) -> None:
    if cache_obj is None:
        return
    try:
        disk_cache = getattr(cache_obj, "disk_cache", None)
        close_fn = getattr(disk_cache, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception as exc:
        logger.debug("DSPy cache close failed: %s", exc)


def close_dspy_cache(*, reset_memory_cache: bool = False) -> None:
    """
    Close any open DSPy cache handles.

    This is intended for end-of-run cleanup in long-lived training processes.
    A later `configure_dspy(...)` call will recreate the cache if needed.
    """
    global _dspy_cache_runtime_signature

    cache_obj = getattr(dspy, "cache", None)
    if cache_obj is None:
        _dspy_cache_runtime_signature = None
        return

    if reset_memory_cache:
        try:
            reset_fn = getattr(cache_obj, "reset_memory_cache", None)
            if callable(reset_fn):
                reset_fn()
        except Exception as exc:
            logger.debug("DSPy memory cache reset failed: %s", exc)

    _close_cache_instance(cache_obj)
    _dspy_cache_runtime_signature = None


def configure_dspy_cache_from_env(*, force: bool = False) -> tuple[bool, bool]:
    """
    Reconfigure DSPy's global cache from environment flags.

    Supported env vars:
    - `TT_DSPY_ENABLE_DISK_CACHE`  (default: enabled)
    - `TT_DSPY_ENABLE_MEMORY_CACHE` (default: enabled)
    - `DSPY_CACHEDIR`
    - `DSPY_CACHE_LIMIT`
    - `TT_DSPY_MEMORY_CACHE_MAX_ENTRIES`
    """
    global _dspy_cache_runtime_signature
    global _dspy_cache_cleanup_registered

    enable_disk_cache = _env_flag("TT_DSPY_ENABLE_DISK_CACHE", True)
    enable_memory_cache = _env_flag("TT_DSPY_ENABLE_MEMORY_CACHE", True)
    disk_cache_dir = str(
        os.getenv("DSPY_CACHEDIR")
        or (Path.home() / ".dspy_cache")
    )
    try:
        disk_size_limit_bytes = int(float(os.getenv("DSPY_CACHE_LIMIT", "30000000000")))
    except (TypeError, ValueError):
        disk_size_limit_bytes = int(3e10)
    try:
        memory_max_entries = int(float(os.getenv("TT_DSPY_MEMORY_CACHE_MAX_ENTRIES", "1000000")))
    except (TypeError, ValueError):
        memory_max_entries = 1000000

    signature = (
        bool(enable_disk_cache),
        bool(enable_memory_cache),
        str(disk_cache_dir),
        int(disk_size_limit_bytes),
        int(memory_max_entries),
    )
    if not force and signature == _dspy_cache_runtime_signature:
        return enable_disk_cache, enable_memory_cache

    old_cache = getattr(dspy, "cache", None)
    dspy.configure_cache(
        enable_disk_cache=enable_disk_cache,
        enable_memory_cache=enable_memory_cache,
        disk_cache_dir=disk_cache_dir,
        disk_size_limit_bytes=disk_size_limit_bytes,
        memory_max_entries=memory_max_entries,
    )
    new_cache = getattr(dspy, "cache", None)
    if old_cache is not None and old_cache is not new_cache:
        _close_cache_instance(old_cache)

    _dspy_cache_runtime_signature = signature
    if not _dspy_cache_cleanup_registered:
        atexit.register(close_dspy_cache)
        _dspy_cache_cleanup_registered = True

    logger.info(
        "DSPy cache config: disk=%s memory=%s dir=%s",
        str(enable_disk_cache).lower(),
        str(enable_memory_cache).lower(),
        disk_cache_dir,
    )
    return enable_disk_cache, enable_memory_cache


def configure_dspy(
    lm: dspy.LM,
    adapter: Optional[Any] = None,
    **kwargs
) -> None:
    """
    Configure DSPy with XMLAdapter by default.

    This is a drop-in replacement for dspy.configure() that uses XMLAdapter
    for more robust output parsing.

    Args:
        lm: The DSPy language model to use
        adapter: Optional custom adapter (defaults to XMLAdapter)
        **kwargs: Additional arguments passed to dspy.configure()
            (e.g., async_max_workers)

    Example:
        from src.config.dspy_config import configure_dspy

        lm = dspy.LM("openai/model", api_base="...", api_key="...")
        configure_dspy(lm=lm)
    """
    configure_dspy_cache_from_env()

    if adapter is None:
        adapter = get_xml_adapter()

    # Merge optional DSPy runtime overrides from settings.yaml.
    try:
        from src.config.settings import load_settings

        settings = load_settings()
        dspy_overrides = settings.get("dspy", {}) if isinstance(settings, dict) else {}
    except Exception:
        dspy_overrides = {}

    for key, value in dspy_overrides.items():
        if key not in kwargs:
            kwargs[key] = value

    dspy.configure(lm=lm, adapter=adapter, **kwargs)


def create_vllm_lm(
    port: int,
    model: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
    cache: bool = True,
    **kwargs,
) -> dspy.LM:
    """
    Create a DSPy LM configured for a local vLLM server.

    This factory provides a consistent way to create DSPy language models
    for vLLM backends, with automatic model detection and context-aware
    max_tokens calculation.

    Args:
        port: vLLM server port (e.g., 8000)
        model: Model name. If None, auto-detects from server.
        temperature: Sampling temperature (default: 0.5)
        max_tokens: Maximum tokens to generate. If None, calculated from
                   context window using ContextWindowManager (recommended).
        cache: Enable DSPy caching (default: True)
        **kwargs: Additional arguments passed to dspy.LM()

    Returns:
        Configured dspy.LM instance

    Example:
        from src.config.dspy_config import create_vllm_lm, configure_dspy

        # Auto-detect model and calculate safe max_tokens
        lm = create_vllm_lm(port=8000)
        configure_dspy(lm=lm)

        # Explicit model
        lm = create_vllm_lm(port=8000, model="qwen-30b-thinking")
    """
    if model is None:
        model = detect_model_from_port(port=port)

    # If max_tokens not specified, calculate from context window
    if max_tokens is None:
        context_window = get_context_window_from_port(port=port)
        manager = ContextWindowManager(context_window=context_window)
        max_tokens = manager.max_output_tokens

    return dspy.LM(
        model=f"openai/{model}",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        **kwargs,
    )


def create_vllm_lm_multi(
    ports: Sequence[int],
    model: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
    cache: bool = True,
    **kwargs,
) -> dspy.LM:
    """
    Create a DSPy LM load-balanced across multiple local vLLM servers.

    Args:
        ports: vLLM server ports (e.g., [8000, 8002])
        model: Model name. If None, auto-detects from the first port.
        temperature: Sampling temperature (default: 0.5)
        max_tokens: Maximum tokens to generate. If None, computed from the first port.
        cache: Enable DSPy caching (default: True)
        **kwargs: Additional arguments passed to dspy.LM()

    Returns:
        A dspy.LM instance that round-robins requests across the provided ports.
    """
    deduped_ports: list[int] = []
    seen = set()
    for port in ports:
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            continue
        if port_int in seen:
            continue
        seen.add(port_int)
        deduped_ports.append(port_int)

    if not deduped_ports:
        raise ValueError("create_vllm_lm_multi requires at least one valid port")

    primary_port = deduped_ports[0]
    if model is None:
        model = detect_model_from_port(port=primary_port)

    if max_tokens is None:
        context_window = get_context_window_from_port(port=primary_port)
        manager = ContextWindowManager(context_window=context_window)
        max_tokens = manager.max_output_tokens

    api_bases = [f"http://localhost:{p}/v1" for p in deduped_ports]
    # When load-balancing across multiple local endpoints, prefer quick failover
    # to the next base instead of retrying a dead port multiple times.
    kwargs.setdefault("num_retries", 0)
    return LoadBalancedLM(
        model=f"openai/{model}",
        api_bases=api_bases,
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        **kwargs,
    )


def create_vllm_lm_with_manager(
    port: int,
    model: Optional[str] = None,
    temperature: float = 0.5,
    cache: bool = True,
    task: str = "default",
    **kwargs,
) -> Tuple[dspy.LM, ContextWindowManager]:
    """
    Create a DSPy LM and its associated ContextWindowManager.

    This is the recommended way to create an LM when you need to make
    context-aware decisions about max_tokens for individual requests.

    Args:
        port: vLLM server port (e.g., 8000)
        model: Model name. If None, auto-detects from server.
        temperature: Sampling temperature (default: 0.5)
        cache: Enable DSPy caching (default: True)
        task: Task type for allocation ("default", "summarizer", "scorer")
        **kwargs: Additional arguments passed to dspy.LM()

    Returns:
        Tuple of (dspy.LM, ContextWindowManager)

    Example:
        from src.config.dspy_config import create_vllm_lm_with_manager

        lm, manager = create_vllm_lm_with_manager(port=8000, task="scorer")

        # Use manager to get safe max_tokens for a specific input
        input_tokens = count_tokens(my_prompt)
        safe_max_tokens = manager.get_safe_max_tokens(input_tokens)
    """
    from src.config.context_window import create_manager_for_task

    if model is None:
        model = detect_model_from_port(port=port)

    # Get context window and create task-appropriate manager
    context_window = get_context_window_from_port(port=port)
    manager = create_manager_for_task(context_window=context_window, task=task)

    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=manager.max_output_tokens,
        cache=cache,
        **kwargs,
    )

    return lm, manager
