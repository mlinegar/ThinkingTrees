"""
DSPy configuration with XMLAdapter for robust output parsing.

This module provides centralized DSPy configuration that uses XMLAdapter
instead of the default ChatAdapter. XMLAdapter uses <field_name>value</field_name>
format which is more robust for parsing than the [[ ## field_name ## ]] format.

Also provides a unified LM factory for creating vLLM-backed DSPy language models.
"""

import dspy
from dspy.adapters import XMLAdapter
from typing import Optional, Any, Tuple

from src.core.model_detection import detect_model_from_port, get_context_window_from_port
from src.config.context_window import ContextWindowManager, DEFAULT_CONTEXT_WINDOW


_xml_adapter: Optional[XMLAdapter] = None


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
