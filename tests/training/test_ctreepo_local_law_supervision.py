from __future__ import annotations

import math

import pytest

from src.training.ctreepo_trainer import CTreePOTrainer, CTreePOTrainingConfig
from src.tree.ctreepo_model import CTreePOConfig


def test_prepare_trees_from_precomputed_labels_nodes_with_oracle_scores():
    trainer = CTreePOTrainer(
        CTreePOTrainingConfig(
            model=CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
            n_epochs=1,
            batch_size=1,
            n_audit=4,
            leaf_audit_weight=0.1,
            audit_weight=0.2,
            device="cpu",
        ),
        node_oracle_predictor=lambda text: float(len(text)),
        node_oracle_source_kind="oracle_callback",
        node_oracle_source_spec="tests.training.fake_oracle:score_span",
    )

    embeddings_by_doc = {
        "doc1": (
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [(0, 3), (3, 7)],
            "abcdefg",
            12.0,
        ),
    }
    built = trainer.prepare_trees_from_precomputed(embeddings_by_doc, split="train")

    assert built == 1
    nodes, _rile, _doc_id = trainer.train_trees[0]
    assert len(nodes) == 3
    for node in nodes:
        assert node.oracle_scores["rile"] == float(len(node.text_span))


def test_evaluate_reports_local_law_counts_and_rates_when_labels_exist():
    trainer = CTreePOTrainer(
        CTreePOTrainingConfig(
            model=CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
            n_epochs=1,
            batch_size=1,
            n_audit=4,
            leaf_audit_weight=0.1,
            audit_weight=0.2,
            local_law_violation_threshold=5.0,
            device="cpu",
        ),
        node_oracle_predictor=lambda text: float(len(text)),
        node_oracle_source_kind="oracle_callback",
        node_oracle_source_spec="tests.training.fake_oracle:score_span",
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [(0, 3), (3, 7)],
                "abcdefg",
                12.0,
            ),
        },
        split="train",
    )

    metrics = trainer.evaluate(trainer.train_trees, epoch=0)

    assert metrics.leaf_oracle_count == 2
    assert metrics.merge_oracle_count == 1
    assert metrics.node_oracle_label_rate == 1.0
    assert math.isfinite(metrics.node_oracle_mae)
    assert math.isfinite(metrics.leaf_oracle_mae)
    assert math.isfinite(metrics.merge_oracle_mae)
    assert 0.0 <= metrics.leaf_violation_rate <= 1.0
    assert 0.0 <= metrics.merge_violation_rate <= 1.0


def test_train_requires_local_law_supervision_when_requested():
    trainer = CTreePOTrainer(
        CTreePOTrainingConfig(
            model=CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
            n_epochs=1,
            batch_size=1,
            leaf_audit_weight=0.1,
            audit_weight=0.2,
            require_local_law_supervision=True,
            device="cpu",
        ),
        node_oracle_predictor=None,
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [(0, 3), (3, 7)],
                "abcdefg",
                12.0,
            ),
        },
        split="train",
    )

    with pytest.raises(ValueError, match="Local-law supervision was required but inactive"):
        trainer.train()


def test_train_allows_required_local_law_supervision_when_labels_exist():
    trainer = CTreePOTrainer(
        CTreePOTrainingConfig(
            model=CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
            n_epochs=1,
            batch_size=1,
            eval_every=1,
            n_audit=4,
            leaf_audit_weight=0.1,
            audit_weight=0.2,
            require_local_law_supervision=True,
            device="cpu",
        ),
        node_oracle_predictor=lambda text: float(len(text)),
        node_oracle_source_kind="oracle_callback",
        node_oracle_source_spec="tests.training.fake_oracle:score_span",
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [(0, 3), (3, 7)],
                "abcdefg",
                12.0,
            ),
        },
        split="train",
    )

    result = trainer.train()

    assert result.epochs_completed == 1
    assert result.local_law_summary["require_local_law_supervision"] is True
    assert result.local_law_summary["node_label_source_kind"] == "oracle_callback"
    assert result.local_law_summary["node_label_source_spec"] == "tests.training.fake_oracle:score_span"
    problem = dict(result.compositional_learning_problem)
    assert problem["name"] == "ctreepo_local_law_training"
    assert problem["uses_full_document_labels"] is True
    assert problem["uses_sampled_substructure_labels"] is True
    channels = list(problem["supervision_channels"])
    assert channels[0]["kind"] == "full_document"
    assert channels[0]["active"] is True
    assert channels[0]["label_source"] == "dataset"
    assert channels[1]["kind"] == "sampled_substructure"
    assert channels[1]["active"] is True
    assert channels[1]["label_source"] == "oracle"
    assert channels[1]["delivery_mode"] == "online_oracle_query"
    assert channels[1]["requires_propensity_logging"] is False
    assert channels[1]["query_policy"]["selection_strategy"] == "all_observed_tree_nodes"
    assert channels[1]["query_policy"]["logs_realized_propensities"] is False
