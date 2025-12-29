"""
DSPy configuration with XMLAdapter for robust output parsing.

This module provides centralized DSPy configuration that uses XMLAdapter
instead of the default ChatAdapter. XMLAdapter uses <field_name>value</field_name>
format which is more robust for parsing than the [[ ## field_name ## ]] format.
"""

import dspy
from dspy.adapters import XMLAdapter
from typing import Optional, Any


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
