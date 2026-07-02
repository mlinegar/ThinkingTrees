from __future__ import annotations

from types import SimpleNamespace

from src.runtime.answering import extract_choice_logprobs, parse_multi_choice_text


def test_extract_choice_logprobs_from_openai_style_payload() -> None:
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            token="B",
                            logprob=-0.2,
                            top_logprobs=[
                                SimpleNamespace(token="A", logprob=-1.2),
                                SimpleNamespace(token="B", logprob=-0.2),
                                SimpleNamespace(token=" C", logprob=-0.7),
                            ],
                        )
                    ]
                )
            )
        ]
    )

    scores = extract_choice_logprobs(raw)

    assert scores["B"] == -0.2
    assert scores["A"] == -1.2
    assert scores["C"] == -0.7


def test_parse_multi_choice_prefers_answer_pattern_then_trailing_letter() -> None:
    assert parse_multi_choice_text("The answer is C because the evidence says so.") == "C"
    assert parse_multi_choice_text("Options A, B, and D are distractors. C") == "C"

