from __future__ import annotations

from types import SimpleNamespace

from src.tasks.manifesto.lawstress_generator import LawStressRecord, normalize_rile
from src.tasks.manifesto.lawstress_proxy import (
    build_proxy_training_examples,
    evaluate_embedding_proxy,
)
from src.training.embedding_proxy import LabeledEmbeddingExample


def _make_record(example_id: str) -> LawStressRecord:
    return LawStressRecord(
        example_id=example_id,
        split="train",
        bin_name="center",
        law_target="c1_sufficiency",
        family="polarity_cancellation",
        difficulty="control",
        anchor_source="synthetic",
        text="DOC TEXT",
        segment_a="SEG A",
        segment_b="SEG B",
        policy_atoms=[],
        target_raw=10.0,
        y_raw=10.0,
        y_norm=normalize_rile(10.0),
        yA_raw=8.0,
        yB_raw=12.0,
        y_merge_expected_raw=10.0,
        teacher_score_doc=9.0,
        teacher_score_segment_a=7.0,
        teacher_score_segment_b=11.0,
        naive_summary="NAIVE",
        naive_score_raw=5.0,
        naive_drift_norm=0.02,
        reference_summary="REF",
        attempts_used=1,
    )


def test_build_proxy_training_examples_includes_multiple_text_types() -> None:
    record = _make_record("lawstress_train_0001")
    examples = build_proxy_training_examples([record])

    assert len(examples) == 5
    doc_ids = {ex.doc_id for ex in examples}
    assert "lawstress_train_0001:doc" in doc_ids
    assert "lawstress_train_0001:segment_a" in doc_ids
    assert "lawstress_train_0001:segment_b" in doc_ids
    assert "lawstress_train_0001:naive_summary" in doc_ids
    assert "lawstress_train_0001:reference_summary" in doc_ids


def test_evaluate_embedding_proxy_groups_by_text_type() -> None:
    # Stub embedding client: embed as [score] so model can return it.
    class StubEmbeddingClient:
        def embed_texts(self, texts):
            return [[float(text)] for text in texts]

    class StubModel:
        def predict_from_embedding(self, embedding):
            return float(embedding[0])

    examples = [
        LabeledEmbeddingExample(doc_id="ex1:doc", text="0.2", target_score=0.2),
        LabeledEmbeddingExample(doc_id="ex2:doc", text="0.4", target_score=0.4),
        LabeledEmbeddingExample(doc_id="ex3:naive_summary", text="0.1", target_score=0.1),
    ]

    metrics = evaluate_embedding_proxy(
        StubModel(),
        embedding_client=StubEmbeddingClient(),
        eval_examples=examples,
    )

    assert metrics["overall"]["n"] == 3
    assert metrics["by_type"]["doc"]["n"] == 2
    assert metrics["by_type"]["naive_summary"]["n"] == 1
    assert metrics["overall"]["mae"] == 0.0

