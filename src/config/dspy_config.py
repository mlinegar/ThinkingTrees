"""
DSPy configuration with XMLAdapter for robust output parsing.

This module provides centralized DSPy configuration that uses XMLAdapter
instead of the default ChatAdapter. XMLAdapter uses <field_name>value</field_name>
format which is more robust for parsing than the [[ ## field_name ## ]] format.

Also provides a unified LM factory for creating vLLM-backed DSPy language models.
"""

import dspy
from dspy.adapters import XMLAdapter
from typing import Optional, Any

from src.core.model_detection import detect_model_from_port


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

    dspy.configure(lm=lm, adapter=adapter, **kwargs)


def create_vllm_lm(
    port: int,
    model: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 8192,
    cache: bool = True,
    **kwargs,
) -> dspy.LM:
    """
    Create a DSPy LM configured for a local vLLM server.

    This factory provides a consistent way to create DSPy language models
    for vLLM backends, with automatic model detection.

    Args:
        port: vLLM server port (e.g., 8000)
        model: Model name. If None, auto-detects from server.
        temperature: Sampling temperature (default: 0.5)
        max_tokens: Maximum tokens to generate (default: 8192)
        cache: Enable DSPy caching (default: True)
        **kwargs: Additional arguments passed to dspy.LM()

    Returns:
        Configured dspy.LM instance

    Example:
        from src.config.dspy_config import create_vllm_lm, configure_dspy

        # Auto-detect model
        lm = create_vllm_lm(port=8000)
        configure_dspy(lm=lm)

        # Explicit model
        lm = create_vllm_lm(port=8000, model="qwen-30b-thinking")
    """
    if model is None:
        model = detect_model_from_port(port=port)

    return dspy.LM(
        model=f"openai/{model}",
        api_base=f"http://localhost:{port}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        **kwargs,
    )
