from __future__ import annotations

import numpy as np

from src.ctreepo.opt import collect_pairwise_preferences, to_training_preference_dataset
from src.ctreepo.sim.core.segment_lda_ops_weight_recovery import (
    _oracle_from_prefix,
    _prefix_counts,
    _span_features_from_prefix,
    generate_segment_lda_docs,
    sample_topic_distributions,
)


class _LDASketchCandidateGenerator:
    def __init__(self, *, feat_true: np.ndarray, feat_bad: np.ndarray):
        self.feat_true = np.asarray(feat_true, dtype=np.float64)
        self.feat_bad = np.asarray(feat_bad, dtype=np.float64)

    def generate(self, _doc, *, n: int, seed: int | None = None):
        del _doc, n, seed
        return (self.feat_true, self.feat_bad)


def test_opt_layer_handles_segment_lda_oracle_setting() -> None:
    # Tiny Segment-LDA doc (fast).
    topics, _meta = sample_topic_distributions(
        vocab_size=24,
        n_topics=3,
        topic_concentration=0.3,
        emission_mode="disjoint",
        anchor_words_per_topic=0,
        anchor_multiplier=1.0,
        seed=0,
    )
    docs, _stats = generate_segment_lda_docs(
        1,
        topics=topics,
        min_tokens=24,
        max_tokens=24,
        min_segments=1,
        max_segments=3,
        min_seg_len=4,
        max_seg_len=12,
        leaf_tokens=4,
        align_segments_to_leaves=True,
        doc_topic_concentration=0.4,
        topic_process="bag_of_words",
        boundary_profile="uniform",
        boundary_profile_strength=0.0,
        boundary_profile_seed=0,
        seed=1,
    )
    assert len(docs) == 1
    doc = docs[0]

    # Oracle: a simple, non-permutation-invariant linear functional of unigram counts.
    theta = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    w_big = np.zeros((3 * 3,), dtype=np.float64)

    topic_prefix, bigram_prefix = _prefix_counts(doc.topics, n_topics=3)
    span = (0, len(doc.topics))
    feat_true, _first, _last = _span_features_from_prefix(
        topic_prefix,
        bigram_prefix,
        doc.topics,
        span,
        n_topics=3,
    )
    y_true = _oracle_from_prefix(theta, w_big, topic_prefix, bigram_prefix, doc.topics, span)
    # Degenerate draw guard: ensure y_true differs from the corrupted-topics baseline.
    if y_true == 0.0:
        topics_mut = list(doc.topics)
        topics_mut[0] = 1
        doc = type(doc)(tokens=doc.tokens, topics=tuple(topics_mut))
        topic_prefix, bigram_prefix = _prefix_counts(doc.topics, n_topics=3)
        feat_true, _first, _last = _span_features_from_prefix(
            topic_prefix,
            bigram_prefix,
            doc.topics,
            span,
            n_topics=3,
        )
        y_true = _oracle_from_prefix(theta, w_big, topic_prefix, bigram_prefix, doc.topics, span)

    # Misspecified sketch: compute features under a corrupted topic sequence.
    topics_bad = tuple(0 for _ in doc.topics)
    topic_prefix_bad, bigram_prefix_bad = _prefix_counts(topics_bad, n_topics=3)
    feat_bad, _b_first, _b_last = _span_features_from_prefix(
        topic_prefix_bad,
        bigram_prefix_bad,
        topics_bad,
        span,
        n_topics=3,
    )

    beta = np.concatenate([theta, w_big], axis=0).astype(np.float64, copy=False)

    def utility(_doc, feat: np.ndarray) -> float:
        y_hat = float(np.dot(beta, np.asarray(feat, dtype=np.float64)))
        return -abs(y_hat - float(y_true))

    records = collect_pairwise_preferences(
        [doc],
        candidate_generator=_LDASketchCandidateGenerator(feat_true=feat_true, feat_bad=feat_bad),
        utility_fn=utility,
        rubric="segment-lda-ops",
        seed=0,
    )
    assert records[0].preferred == "A"

    dataset = to_training_preference_dataset(records)
    pair = dataset[0]
    assert pair.preferred == "A"
    # The opt layer should serialize numpy candidates as JSON arrays (not "array([...])").
    assert pair.summary_a.strip().startswith("[")
    assert pair.summary_b.strip().startswith("[")
