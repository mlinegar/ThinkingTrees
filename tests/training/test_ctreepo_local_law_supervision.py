from __future__ import annotations

import math

import pytest
import torch

from src.training.config_sections import RuntimeConfig, TrainConfig, ValidationConfig
from src.training.ctreepo_trainer import (
    CTreePOTrainer,
    CTreePOTrainingConfig,
    LocalLawSupervisionConfig,
    TreeOperatorObjectiveConfig,
)
from src.tree.ctreepo_model import CTreePOConfig


def _training_config(
    *,
    model: CTreePOConfig | None = None,
    epochs: int = 1,
    batch_size: int = 1,
    eval_every: int = 1,
    n_audit: int = 5,
    leaf_audit_weight: float = 0.0,
    merge_audit_weight: float = 0.5,
    local_law_violation_threshold: float = 10.0,
    require_local_law_supervision: bool = False,
    device: str = "cpu",
) -> CTreePOTrainingConfig:
    return CTreePOTrainingConfig(
        model=model or CTreePOConfig(embedding_dim=4, sketch_dim=8, hidden_dim=16),
        train=TrainConfig(epochs=epochs, batch_size=batch_size),
        validation=ValidationConfig(eval_every=eval_every),
        runtime=RuntimeConfig(device=device),
        objective=TreeOperatorObjectiveConfig(
            leaf_audit_weight=leaf_audit_weight,
            merge_audit_weight=merge_audit_weight,
            local_law_violation_threshold=local_law_violation_threshold,
        ),
        supervision=LocalLawSupervisionConfig(
            n_audit=n_audit,
            require_local_law_supervision=require_local_law_supervision,
        ),
    )


def test_prepare_trees_from_precomputed_labels_nodes_with_oracle_scores():
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=1,
            n_audit=4,
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
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


def test_prepare_trees_from_precomputed_canonicalizes_leaf_embeddings_and_runtime_cache():
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=1,
            device="cpu",
        ),
        node_oracle_predictor=lambda text: float(len(text)),
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
                [(0, 2), (2, 5), (5, 8)],
                "abcdefgh",
                12.0,
            ),
        },
        split="train",
    )

    nodes, _rile, _doc_id = trainer.train_trees[0]
    leaf_embeddings = [node.embedding for node in nodes if node.is_leaf]

    assert leaf_embeddings
    assert all(isinstance(embedding, torch.Tensor) for embedding in leaf_embeddings)
    assert all(embedding.dtype == torch.float32 for embedding in leaf_embeddings)
    assert all(embedding.device.type == "cpu" for embedding in leaf_embeddings)
    assert trainer._packed_tree_for_nodes(nodes) is trainer._packed_tree_for_nodes(nodes)
    assert trainer._split_runtime_metadata["train"]["runtime_data_mode"] == "staged"


def test_evaluate_reports_local_law_counts_and_rates_when_labels_exist():
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=1,
            n_audit=4,
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
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


def test_train_step_and_evaluate_record_packed_runtime_metadata() -> None:
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=2,
            n_audit=4,
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
            device="cpu",
        ),
        node_oracle_predictor=lambda text: float(len(text)),
        node_oracle_source_kind="oracle_callback",
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [(0, 4), (4, 8)],
                "abcdefgh",
                10.0,
            ),
            "doc2": (
                [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]],
                [(0, 4), (4, 8)],
                "qrstuvwx",
                12.0,
            ),
        },
        split="train",
    )
    optimizer = trainer._make_optimizer()
    trainer.train_step(trainer.train_trees, optimizer)
    metrics = trainer.evaluate(trainer.train_trees, epoch=0)

    train_runtime = dict(trainer._last_train_step_stats["runtime"])
    eval_runtime = dict(trainer._last_eval_runtime_stats)

    assert metrics.n_docs == 2
    for runtime in (train_runtime, eval_runtime):
        assert runtime["runtime_data_mode"] == "staged"
        assert runtime["packed_executor_mode"] == "fixed_fused"
        assert runtime["host_to_device_bytes"] == 0
        assert runtime["host_to_device_events"] == 0
        assert runtime["resident_store_hits"] == 0
        assert runtime["resident_store_misses"] == 2
        assert "materialized_node_sketch_count" in runtime


def test_packed_batch_cache_hits_on_repeated_batches() -> None:
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=2,
            device="cpu",
        ),
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [(0, 4), (4, 8)],
                "abcdefgh",
                10.0,
            ),
            "doc2": (
                [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]],
                [(0, 4), (4, 8)],
                "qrstuvwx",
                12.0,
            ),
        },
        split="train",
    )

    first = trainer._forward_packed_batch(trainer.train_trees, materialize_nodes=False)
    second = trainer._forward_packed_batch(trainer.train_trees, materialize_nodes=False)

    assert first.runtime_stats["packed_batch_cache_hit"] is False
    assert second.runtime_stats["packed_batch_cache_hit"] is True
    assert first.runtime_stats["packed_bucket_store_hit"] is True
    assert second.runtime_stats["packed_bucket_store_hit"] is True
    assert first.runtime_stats["packed_bucket_store_mode"] == "staged_rows"
    assert second.runtime_stats["packed_bucket_store_mode"] == "staged_rows"
    assert len(trainer._packed_batch_cache) == 1


def test_fixed_shape_bucket_store_serves_different_subsets() -> None:
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=2,
            device="cpu",
        ),
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [(0, 4), (4, 8)],
                "abcdefgh",
                10.0,
            ),
            "doc2": (
                [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]],
                [(0, 4), (4, 8)],
                "qrstuvwx",
                12.0,
            ),
            "doc3": (
                [[0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0]],
                [(0, 4), (4, 8)],
                "ijklmnop",
                14.0,
            ),
        },
        split="train",
    )

    first_subset = trainer._forward_packed_batch(trainer.train_trees[:2], materialize_nodes=False)
    second_subset = trainer._forward_packed_batch(trainer.train_trees[1:], materialize_nodes=False)

    assert trainer._split_runtime_metadata["train"]["fixed_shape_bucket_store_count"] == 1
    assert trainer._split_runtime_metadata["train"]["fixed_shape_dense_bucket_store_count"] == 0
    assert first_subset.runtime_stats["packed_bucket_store_hit"] is True
    assert second_subset.runtime_stats["packed_bucket_store_hit"] is True
    assert first_subset.runtime_stats["packed_bucket_store_mode"] == "staged_rows"
    assert second_subset.runtime_stats["packed_bucket_store_mode"] == "staged_rows"
    assert first_subset.runtime_stats["packed_batch_cache_hit"] is False
    assert second_subset.runtime_stats["packed_batch_cache_hit"] is False


def test_train_requires_local_law_supervision_when_requested():
    trainer = CTreePOTrainer(
        _training_config(
            batch_size=1,
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
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
        _training_config(
            batch_size=1,
            eval_every=1,
            n_audit=4,
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
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
