from src.core.conditional_memory import ConditionalMemory
from src.core.engram_memory import EngramMemoryConfig, extract_engram_memory_items
from src.core.engram_prompting import wrap_summarize_prompt_with_engram_memory
from src.core.prompting import default_summarize_prompt


def test_extract_engram_memory_items_finds_entities_and_ids():
    text = (
        "Diana, Princess of Wales (1961-1997) visited https://example.com.\n"
        "Ticket: 550e8400-e29b-41d4-a716-446655440000\n"
        "Build ID: ABC_DEF_12345\n"
        "Ref: 123456\n"
    )
    cfg = EngramMemoryConfig(enabled=True, max_items=64, max_chars=10_000)
    items = extract_engram_memory_items(text, cfg)

    assert "Princess of Wales" in items
    assert "https://example.com" in items
    assert "550e8400-e29b-41d4-a716-446655440000" in items
    assert "ABC_DEF_12345" in items
    assert "123456" in items


def test_extract_engram_memory_items_respects_limits():
    text = " ".join([f"https://example.com/{i}" for i in range(100)])
    cfg = EngramMemoryConfig(enabled=True, max_items=3, max_chars=60)
    items = extract_engram_memory_items(text, cfg)
    assert len(items) <= 3
    assert sum(len(item) for item in items) <= 60


def test_extract_engram_memory_items_caches_in_conditional_memory(tmp_path):
    text = (
        "Diana, Princess of Wales (1961-1997) visited https://example.com.\n"
        "Ticket: 550e8400-e29b-41d4-a716-446655440000\n"
        "Build ID: ABC_DEF_12345\n"
        "Ref: 123456\n"
    )
    cfg = EngramMemoryConfig(enabled=True, max_items=16, max_chars=500)
    db_path = tmp_path / "memory.db"

    writer = ConditionalMemory(sqlite_path=db_path, l1_capacity=16, mode="readwrite")
    try:
        items_pass1 = extract_engram_memory_items(text, cfg, memory=writer)
        assert items_pass1

        before = writer.report()
        items_pass2 = extract_engram_memory_items(text, cfg, memory=writer)
        after = writer.report()
        assert items_pass2 == items_pass1
        assert (after["l1_hits"] + after["l2_hits"]) > (before["l1_hits"] + before["l2_hits"])
    finally:
        writer.close()

    reader = ConditionalMemory(sqlite_path=db_path, l1_capacity=16, mode="read")
    try:
        items_cross_run = extract_engram_memory_items(text, cfg, memory=reader)
        assert items_cross_run == items_pass1
        assert reader.report()["l2_hits"] >= 1
    finally:
        reader.close()


def test_wrap_summarize_prompt_injects_static_memory_block():
    text = "Only Alexander the Great could tame the horse Bucephalus."
    cfg = EngramMemoryConfig(enabled=True, max_items=16, max_chars=500)
    wrapped = wrap_summarize_prompt_with_engram_memory(default_summarize_prompt, cfg)
    messages = wrapped(text, "Preserve all named entities.")

    assert any(m.get("role") == "user" and "STATIC MEMORY" in (m.get("content") or "") for m in messages)
    assert any(
        m.get("role") == "system" and "Preserve any STATIC MEMORY items" in (m.get("content") or "")
        for m in messages
    )
