from __future__ import annotations

from src.core.engram_memory import EngramMemoryConfig
from src.core.engram_prompting import wrap_summarize_prompt_with_engram_memory
from src.core.prompting import default_summarize_prompt
from src.core.semantic_prompting import (
    SEMANTIC_PROMPT_VERSION,
    clear_semantic_memory_registry,
    register_semantic_memory_for_doc,
    semantic_document_memory,
    wrap_summarize_prompt_with_semantic_memory,
)
from src.core.strategy import tournament_doc_id


def _system_message(messages):
    return next((m.get("content", "") for m in messages if m.get("role") == "system"), "")


def test_semantic_prompt_wrapper_injects_block_and_version():
    wrapped = wrap_summarize_prompt_with_semantic_memory(default_summarize_prompt)
    payload = {
        "neighbors": [
            {
                "doc_id": "11320_199809",
                "scope": "same_party_country",
                "year": 1998,
                "similarity": 0.91,
                "score": 0.95,
                "rile": -12.0,
                "delta_rile": 0.08,
                "snippets": [{"similarity": 0.9, "text": "public welfare and labor protections"}],
            }
        ]
    }
    token = semantic_document_memory.set(payload)
    try:
        messages = wrapped("Texto de prueba", "Preserve position-relevant content.")
    finally:
        semantic_document_memory.reset(token)

    assert any(m.get("role") == "user" and "SEMANTIC MEMORY" in (m.get("content") or "") for m in messages)
    assert SEMANTIC_PROMPT_VERSION in _system_message(messages)


def test_semantic_and_engram_wrappers_compose():
    base = default_summarize_prompt
    with_semantic = wrap_summarize_prompt_with_semantic_memory(base)
    with_both = wrap_summarize_prompt_with_engram_memory(
        with_semantic,
        EngramMemoryConfig(enabled=True, max_items=8, max_chars=200),
    )
    payload = {"neighbors": [{"doc_id": "d1", "scope": "global", "similarity": 0.8, "score": 0.8}]}
    token = semantic_document_memory.set(payload)
    try:
        messages = with_both("Alice visited Berlin on 2026-01-01", "Preserve entities and dates.")
    finally:
        semantic_document_memory.reset(token)

    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    system = _system_message(messages)
    assert "SEMANTIC MEMORY" in user
    assert "STATIC MEMORY" in user
    assert "Preserve any STATIC MEMORY items" in system


def test_registry_lookup_via_tournament_doc_id():
    wrapped = wrap_summarize_prompt_with_semantic_memory(default_summarize_prompt)
    clear_semantic_memory_registry()
    register_semantic_memory_for_doc(
        "doc_123",
        {"neighbors": [{"doc_id": "d2", "scope": "global", "similarity": 0.7, "score": 0.7}]},
    )
    token = tournament_doc_id.set("doc_123")
    try:
        messages = wrapped("hello", "rubric")
    finally:
        tournament_doc_id.reset(token)
        clear_semantic_memory_registry()
    assert any(m.get("role") == "user" and "SEMANTIC MEMORY" in (m.get("content") or "") for m in messages)
