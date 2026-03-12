from src.core.engram_memory import EngramMemoryConfig
from src.core.engram_prompting import (
    ENGRAM_PROMPT_VERSION,
    wrap_summarize_prompt_with_engram_memory,
)
from src.core.prompting import default_summarize_prompt


def _system_message(messages):
    return next((m.get("content", "") for m in messages if m.get("role") == "system"), "")


def test_engram_prompt_prefix_has_stable_version_tag():
    cfg = EngramMemoryConfig(enabled=True, max_items=8, max_chars=400)
    wrapped = wrap_summarize_prompt_with_engram_memory(default_summarize_prompt, cfg)
    rubric = "Preserve key entities and dates exactly."

    msg_a = wrapped("Alice visited Paris on 2026-01-01", rubric)
    msg_b = wrapped("Bob visited London on 2026-02-02", rubric)

    system_a = _system_message(msg_a)
    system_b = _system_message(msg_b)

    assert ENGRAM_PROMPT_VERSION in system_a
    assert system_a == system_b
