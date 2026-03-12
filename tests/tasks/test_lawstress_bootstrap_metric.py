from __future__ import annotations

from types import SimpleNamespace

from src.tasks.manifesto.lawstress_bootstrap_metric import (
    LawStressBootstrapObjectiveConfig,
    create_lawstress_bootstrap_metric,
)
from src.tasks.manifesto.lawstress_eval import LawStressEvalConfig


class _StubEmbeddingClient:
    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def embed_texts(self, texts):
        # Encode desired score in the embedding itself.
        return [[float(self._mapping.get(text, 0.5))] for text in texts]


class _StubProxyModel:
    def predict_from_embedding(self, embedding):
        return float(embedding[0])


def test_metric_c2_penalizes_strict_same_side_fail() -> None:
    mapping = {
        "s1": 0.6,
        "s2": 0.4,  # flips side around neutral=0.5 relative to s1
    }
    metric = create_lawstress_bootstrap_metric(
        proxy_model=_StubProxyModel(),
        embedding_client=_StubEmbeddingClient(mapping),
        config=LawStressEvalConfig(c2_threshold_norm=0.06),
    )

    gold = SimpleNamespace(
        law_target="c2_idempotence",
        text="TEXT",
        y_doc_norm=0.6,
        segment_a="A",
        segment_b="B",
        teacher_score_segment_a_raw=10.0,
        teacher_score_segment_b_raw=10.0,
    )
    pred = SimpleNamespace(summary1="s1", summary2="s2")

    out = metric(gold, pred, None, None, None)
    assert 0.0 <= out["score"] <= 1.0
    assert out["score"] == 0.0


def test_metric_c1_rewards_small_error() -> None:
    mapping = {"good": 0.55, "bad": 0.10}
    metric = create_lawstress_bootstrap_metric(
        proxy_model=_StubProxyModel(),
        embedding_client=_StubEmbeddingClient(mapping),
        config=LawStressEvalConfig(c1_threshold_norm=0.10),
    )

    gold = SimpleNamespace(
        law_target="c1_sufficiency",
        text="TEXT" * 100,
        y_doc_norm=0.50,
        segment_a="A",
        segment_b="B",
        teacher_score_segment_a_raw=0.0,
        teacher_score_segment_b_raw=0.0,
    )

    out_good = metric(gold, SimpleNamespace(summary1="good"), None, None, None)
    out_bad = metric(gold, SimpleNamespace(summary1="bad"), None, None, None)

    assert out_good["score"] > out_bad["score"]


def test_metric_c3_min_objective_targets_weakest_component() -> None:
    segment_a = "A " * 250
    segment_b = "B " * 250
    summary_a = ("alpha " * 24).strip()
    summary_b = ("beta " * 24).strip()
    merged = ("merge " * 24).strip()
    joint = ("joint " * 24).strip()

    mapping = {
        summary_a: 0.90,  # poor segment-A sufficiency vs target 0.5
        summary_b: 0.50,  # good segment-B sufficiency
        merged: 0.50,  # good C3 merge expectation
        joint: 0.50,  # good C3 substitution
    }
    gold = SimpleNamespace(
        law_target="c3_merge",
        text=(segment_a + "\n" + segment_b),
        y_doc_norm=0.50,
        segment_a=segment_a,
        segment_b=segment_b,
        teacher_score_segment_a_raw=0.0,
        teacher_score_segment_b_raw=0.0,
    )
    pred = SimpleNamespace(
        summary_a=summary_a,
        summary_b=summary_b,
        merged_summary=merged,
        joint_segments_summary=joint,
    )

    weighted_metric = create_lawstress_bootstrap_metric(
        proxy_model=_StubProxyModel(),
        embedding_client=_StubEmbeddingClient(mapping),
        config=LawStressEvalConfig(c1_threshold_norm=0.10, c3_threshold_norm=0.08),
        objective=LawStressBootstrapObjectiveConfig(aggregate_mode="weighted_mean"),
    )
    min_metric = create_lawstress_bootstrap_metric(
        proxy_model=_StubProxyModel(),
        embedding_client=_StubEmbeddingClient(mapping),
        config=LawStressEvalConfig(c1_threshold_norm=0.10, c3_threshold_norm=0.08),
        objective=LawStressBootstrapObjectiveConfig(aggregate_mode="min"),
    )

    out_weighted = weighted_metric(gold, pred, None, None, None)
    out_min = min_metric(gold, pred, None, None, None)

    assert out_weighted["score"] > out_min["score"]
    assert out_weighted["details"]["objective_mode"] == "weighted_mean"
    assert out_min["details"]["objective_mode"] == "min"


def test_metric_default_objective_is_bottleneck_min() -> None:
    segment_a = "A " * 250
    segment_b = "B " * 250
    summary_a = ("alpha " * 24).strip()
    summary_b = ("beta " * 24).strip()
    merged = ("merge " * 24).strip()
    joint = ("joint " * 24).strip()

    mapping = {
        summary_a: 0.90,
        summary_b: 0.50,
        merged: 0.50,
        joint: 0.50,
    }
    gold = SimpleNamespace(
        law_target="c3_merge",
        text=(segment_a + "\n" + segment_b),
        y_doc_norm=0.50,
        segment_a=segment_a,
        segment_b=segment_b,
        teacher_score_segment_a_raw=0.0,
        teacher_score_segment_b_raw=0.0,
    )
    pred = SimpleNamespace(
        summary_a=summary_a,
        summary_b=summary_b,
        merged_summary=merged,
        joint_segments_summary=joint,
    )

    metric = create_lawstress_bootstrap_metric(
        proxy_model=_StubProxyModel(),
        embedding_client=_StubEmbeddingClient(mapping),
        config=LawStressEvalConfig(c1_threshold_norm=0.10, c3_threshold_norm=0.08),
    )
    out = metric(gold, pred, None, None, None)
    assert out["details"]["objective_mode"] == "min"
