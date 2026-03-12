"""Tests for prompting helpers."""

from src.core.prompting import clean_summary_text, parse_numeric_score, sanitize_instruction_text


def test_parse_numeric_score_accepts_clean_numeric():
    assert parse_numeric_score("42", min_value=-100, max_value=100) == 42.0
    assert parse_numeric_score(" -7.5 ", min_value=-100, max_value=100) == -7.5


def test_parse_numeric_score_prefers_numeric_line():
    response = "Reasoning:\n- mixed cues\n\n12.5\n"
    assert parse_numeric_score(response, min_value=-100, max_value=100) == 12.5


def test_parse_numeric_score_avoids_instruction_range_tokens():
    response = (
        "Score range: -100 to +100.\n"
        "Output only one number.\n"
        "Final score: 14"
    )
    assert parse_numeric_score(response, min_value=-100, max_value=100) == 14.0


def test_parse_numeric_score_reads_json_score_field():
    response = '{"score": -33.5, "reasoning": "market-friendly but welfare-supportive"}'
    assert parse_numeric_score(response, min_value=-100, max_value=100) == -33.5


def test_parse_numeric_score_rejects_out_of_range_only_values():
    response = "The score might be 140."
    assert parse_numeric_score(response, min_value=-100, max_value=100, allow_llm_fallback=False) is None


def test_parse_numeric_score_returns_none_for_non_numeric_without_fallback():
    response = "No score present here."
    assert parse_numeric_score(response, min_value=0, max_value=1, allow_llm_fallback=False) is None


def test_parse_numeric_score_ignores_range_echo_without_prediction():
    response = "Score range: -100 to +100. Output only one value."
    assert parse_numeric_score(response, min_value=-100, max_value=100, allow_llm_fallback=False) is None


def test_parse_numeric_score_calls_llm_fallback(monkeypatch):
    calls = {"count": 0}

    def fake_fallback(response: str, *, min_value, max_value):
        calls["count"] += 1
        assert response == "unparseable"
        assert min_value == 0.0
        assert max_value == 1.0
        return 0.25

    monkeypatch.setattr("src.core.prompting._extract_with_llm_fallback", fake_fallback)

    assert parse_numeric_score("unparseable", min_value=0.0, max_value=1.0, allow_llm_fallback=True) == 0.25
    assert calls["count"] == 1


def test_clean_summary_text_strips_think_block():
    response = "<think>reasoning</think>\nActual summary."
    assert clean_summary_text(response) == "Actual summary."


def test_clean_summary_text_strips_stray_close_tag_prefix():
    response = "We need to summarize...\n</think>\nActual summary."
    assert clean_summary_text(response) == "Actual summary."


def test_clean_summary_text_strips_outer_code_fence():
    response = "```\nActual summary.\n```"
    assert clean_summary_text(response) == "Actual summary."


def test_sanitize_instruction_text_strips_instruction_writing_meta():
    raw = (
        "You are to output a single numeric score in [-100, +100].\n"
        "Wrap in triple backticks.\n"
        "Thus final answer: a code block with the instruction.\n"
        "</think>\n"
        "```\n"
        "You must output exactly one numeric score in the range [-100, +100].\n"
        "Do not include extra text.\n"
        "```\n"
    )
    cleaned = sanitize_instruction_text(raw)
    assert "wrap in triple backticks" not in cleaned.lower()
    assert "thus final answer" not in cleaned.lower()
    assert "```" not in cleaned
    assert "</think>" not in cleaned.lower()
    assert "You must output exactly one numeric score" in cleaned
