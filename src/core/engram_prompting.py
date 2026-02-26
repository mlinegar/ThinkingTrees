"""
Prompt-builder wrappers for Engram-style static memory injection.

These utilities are task-agnostic: they wrap an existing prompt builder and
append a compact "STATIC MEMORY" block derived from the input text.

This mirrors Engram's core idea: offload stereotyped/local pattern handling to
deterministic lookup so model compute can focus on reasoning.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from src.core.engram_memory import (
    EngramMemoryConfig,
    extract_engram_memory_items,
    format_engram_memory_block,
)


def _inject_memory_into_messages(
    messages: List[Dict[str, str]],
    memory_block: str,
) -> List[Dict[str, str]]:
    if not memory_block or not messages:
        return messages

    # Copy to avoid mutating task-provided builders.
    out: List[Dict[str, str]] = [dict(m) for m in messages]

    for msg in out:
        if msg.get("role") == "system" and msg.get("content"):
            msg["content"] = (
                msg["content"].rstrip()
                + "\n- Preserve any STATIC MEMORY items exactly if they appear.\n"
                + "- Do not output the STATIC MEMORY list.\n"
            )
            break

    for msg in out:
        if msg.get("role") == "user":
            msg["content"] = (msg.get("content") or "").rstrip() + "\n\n" + memory_block
            break

    return out


def wrap_summarize_prompt_with_engram_memory(
    prompt_fn: Callable[[str, str], List[Dict[str, str]]],
    config: EngramMemoryConfig,
) -> Callable[[str, str], List[Dict[str, str]]]:
    """Wrap a (text, rubric)->messages prompt builder with Engram static memory."""

    def wrapped(text: str, rubric: str) -> List[Dict[str, str]]:
        base = prompt_fn(text, rubric)
        items = extract_engram_memory_items(text, config)
        block = format_engram_memory_block(items)
        return _inject_memory_into_messages(base, block)

    return wrapped


def wrap_merge_prompt_with_engram_memory(
    prompt_fn: Callable[[str, str, str], List[Dict[str, str]]],
    config: EngramMemoryConfig,
) -> Callable[[str, str, str], List[Dict[str, str]]]:
    """Wrap a (left, right, rubric)->messages prompt builder with Engram static memory."""

    def wrapped(left: str, right: str, rubric: str) -> List[Dict[str, str]]:
        base = prompt_fn(left, right, rubric)
        joined = f"{left}\n\n{right}"
        items = extract_engram_memory_items(joined, config)
        block = format_engram_memory_block(items)
        return _inject_memory_into_messages(base, block)

    return wrapped

