from src.feedback.store import FeedbackStore
from src.feedback.types import FeedbackRequest


def test_feedback_store_human_helpers_export_supervision_and_binary_projection() -> None:
    store = FeedbackStore()

    pair_request = FeedbackRequest(
        request_id="pair_req",
        text_a="A",
        text_b="B",
        original_text="Original",
        rubric="Rubric",
        law_type="sufficiency",
    )
    scalar_request = FeedbackRequest(
        request_id="scalar_req",
        text_a="Only response",
        original_text="Original scalar",
        rubric="Rate quality",
        law_type="merge",
        dimensions=[],
    )

    store.enqueue(pair_request)
    store.enqueue(scalar_request)

    assert store.submit_human_pairwise_feedback(
        "pair_req",
        preferred="A",
        reasoning="A is better",
        confidence=1.0,
        score_estimate_a=0.8,
        score_estimate_b=0.3,
    )
    assert store.submit_human_scalar_feedback(
        "scalar_req",
        score=0.75,
        dimension_name="quality",
        reasoning="Looks good",
    )

    supervision = store.to_supervision_dataset()
    assert len(supervision.comparative_judgments) == 1
    assert len(supervision.response_judgments) == 3

    binary_projection = store.to_binary_projection_dataset(projection="adjacent")
    assert len(binary_projection.comparisons) == 2
    assert all(pair.truth_label_source == "human" for pair in binary_projection.comparisons)
