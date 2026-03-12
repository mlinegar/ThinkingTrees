"""Tests for feedback <-> preference conversion utilities."""

from src.feedback.types import FeedbackRequest, FeedbackResponse


def test_feedback_response_to_preference_pair_uses_critique_when_reasoning_empty():
    request = FeedbackRequest(
        request_id="r1",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
    )
    response = FeedbackResponse(
        request_id="r1",
        preferred="A",
        critique="Human critique text",
        reasoning="",
        source="human",
    )

    pair = response.to_preference_pair(request)
    assert pair.preferred == "A"
    assert pair.reasoning == "Human critique text"


def test_feedback_response_to_preference_pair_merges_reasoning_and_critique():
    request = FeedbackRequest(
        request_id="r2",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
    )
    response = FeedbackResponse(
        request_id="r2",
        preferred="B",
        critique="Extra critique",
        reasoning="Core reasoning",
        source="human",
    )

    pair = response.to_preference_pair(request)
    assert pair.preferred == "B"
    assert "Core reasoning" in pair.reasoning
    assert "Extra critique" in pair.reasoning

