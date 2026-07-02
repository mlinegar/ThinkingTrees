"""Tests for request/response projection into preference data."""

from src.preference_collection.types import (
    PreferenceRequest,
    PreferenceResponse,
    preference_dataset_from_responses,
    supervision_dataset_from_responses,
)


def test_preference_response_to_preference_pair_uses_critique_when_reasoning_empty():
    request = PreferenceRequest(
        request_id="r1",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
    )
    response = PreferenceResponse(
        request_id="r1",
        preferred="A",
        critique="Human critique text",
        reasoning="",
        source="human",
    )

    pair = response.to_preference_pair(request)
    assert pair.preferred == "A"
    assert pair.reasoning == "Human critique text"
    binary = response.to_binary_comparison(request)
    assert binary.truth_label_source == "human"
    assert binary.source_observation_ids == ["r1"]


def test_preference_response_to_preference_pair_merges_reasoning_and_critique():
    request = PreferenceRequest(
        request_id="r2",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
    )
    response = PreferenceResponse(
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


def test_preference_response_to_preference_pair_preserves_comparative_signal() -> None:
    request = PreferenceRequest(
        request_id="r3",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
        law_type="sufficiency",
    )
    response = PreferenceResponse(
        request_id="r3",
        preferred="A",
        reasoning="GenRM-style preference",
        score_estimate_a=5.0,
        score_estimate_b=2.0,
        extra={
            "comparison_signal_name": "genrm_ranking_score",
            "comparison_signal_value": 1.0,
            "comparison_signal_min": 1.0,
            "comparison_signal_max": 6.0,
            "response_signal_name": "genrm_helpfulness",
            "response_signal_min": 1.0,
            "response_signal_max": 5.0,
        },
    )

    pair = response.to_preference_pair(request)
    assert pair.comparison_signal_value == 1.0
    assert pair.preference_supervision.comparison_signal_name == "genrm_ranking_score"
    assert pair.preference_supervision.response_signal_name == "genrm_helpfulness"


def test_preference_response_to_response_judgment_uses_scalar_dimension() -> None:
    request = PreferenceRequest(
        request_id="scalar_req",
        text_a="Summary A",
        original_text="Original",
        rubric="Rubric",
        law_type="sufficiency",
        dimensions=[],
    )
    response = PreferenceResponse(
        request_id="scalar_req",
        scores={"faithfulness": 4.5},
        critique="Strong coverage",
        source="human",
    )

    judgment = response.to_response_judgment(request)
    assert judgment.response == "Summary A"
    assert judgment.response_signal_value == 4.5
    assert judgment.supervision_metadata.response_signal_name == "faithfulness"


def test_responses_to_preference_and_supervision_datasets_support_mixed_items() -> None:
    pair_request = PreferenceRequest(
        request_id="pair_req",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
        law_type="sufficiency",
    )
    pair_response = PreferenceResponse(
        request_id="pair_req",
        preferred="A",
        score_estimate_a=0.9,
        score_estimate_b=0.3,
        source="llm_judge",
    )
    scalar_request = PreferenceRequest(
        request_id="scalar_req",
        text_a="Only response",
        original_text="Original 2",
        rubric="Rubric 2",
        law_type="merge",
        dimensions=[],
    )
    scalar_response = PreferenceResponse(
        request_id="scalar_req",
        scores={"score": 0.8},
        source="human",
    )

    items = [(pair_request, pair_response), (scalar_request, scalar_response)]
    preferences = preference_dataset_from_responses(items)
    supervision = supervision_dataset_from_responses(items)

    assert preferences.summary()["n_units"] == 2
    assert preferences.summary()["n_candidates"] == 3
    assert len(preferences.to_records("dpo")) == 1
    assert len(supervision.comparative_judgments) == 1
    assert len(supervision.response_judgments) == 3
    projected = supervision.project_binary(projection="adjacent")
    assert len(projected.pairs) == 2


def test_preference_response_from_human_pairwise_preference_projects_cleanly() -> None:
    request = PreferenceRequest(
        request_id="human_pair_req",
        text_a="Candidate A",
        text_b="Candidate B",
        original_text="Original",
        rubric="Rubric",
        law_type="sufficiency",
    )
    response = PreferenceResponse.from_human_pairwise_preference(
        request_id="human_pair_req",
        preferred="B",
        reasoning="B keeps the key claim",
        confidence=1.0,
        score_estimate_a=0.2,
        score_estimate_b=0.9,
        extra={"comparison_signal_name": "human_vote_margin"},
    )

    binary = response.to_binary_comparison(request)
    assert binary.preferred == "B"
    assert binary.preference_supervision.application_name == "preference_collection"
    assert binary.preference_supervision.preference_family == "pairwise"
    assert binary.truth_label_source == "human"
