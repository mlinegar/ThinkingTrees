from __future__ import annotations

import random

import pytest
import torch

from src.training.config_sections import RunConfig, RuntimeConfig, TrainConfig, ValidationConfig
from src.training.ctreepo_trainer import (
    CTreePOTrainer,
    CTreePOTrainingConfig,
    LocalLawSupervisionConfig,
    TreeOperatorObjectiveConfig,
)
from src.training.tree_model_v2_trainer import (
    RealDocumentTaskAdapter,
    ScalarTarget,
    TreeNodeRef,
    TreeSupervisionBatch,
    TreeModelV2Trainer,
    TreeModelV2ObjectiveConfig,
    TreeModelV2ScoreTargetConfig,
    TreeModelV2TrainingConfig,
)
from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel
from src.tree.embedding_tree import build_embedding_tree, forward_ctreepo_batch
from src.tree.packed_execution import (
    PackedForwardResult,
    build_packed_embedding_tree,
    build_packed_tree_batch,
    forward_packed_tree_batch,
)


def test_ctreepo_training_config_defaults_to_v2_surface() -> None:
    config = CTreePOTrainingConfig()

    assert config.model.tree_model_version == "v2"


def _make_labeled_tree(text: str, rile: float, *, embedding_dim: int = 4):
    windows = [(0, len(text) // 2), (len(text) // 2, len(text))]
    embeddings = [[0.1] * embedding_dim, [0.4] * embedding_dim]
    nodes = build_embedding_tree(text, embeddings, windows)
    for node in nodes:
        node.oracle_scores["rile"] = float(len(node.text_span))
    return nodes, float(rile), f"doc_{abs(hash(text)) % 1000}"


def _v2_config() -> TreeModelV2TrainingConfig:
    return TreeModelV2TrainingConfig(
        score_targets=TreeModelV2ScoreTargetConfig(target_min=-100.0, target_max=100.0),
        objective=TreeModelV2ObjectiveConfig(
            root_weight=1.0,
            leaf_scalar_weight=0.2,
            internal_scalar_weight=0.5,
            fiber_same_weight=0.1,
            fiber_different_weight=0.1,
        ),
    )


def _ctreepo_config(
    *,
    model: CTreePOConfig,
    epochs: int = 1,
    batch_size: int = 1,
    eval_every: int = 1,
    seed: int = 42,
    n_audit: int = 5,
    leaf_audit_weight: float = 0.0,
    merge_audit_weight: float = 0.5,
    contrastive_weight: float = 0.1,
    device: str = "cpu",
) -> CTreePOTrainingConfig:
    return CTreePOTrainingConfig(
        model=model,
        run=RunConfig(seed=seed),
        train=TrainConfig(epochs=epochs, batch_size=batch_size),
        validation=ValidationConfig(eval_every=eval_every),
        runtime=RuntimeConfig(device=device),
        objective=TreeOperatorObjectiveConfig(
            leaf_audit_weight=leaf_audit_weight,
            merge_audit_weight=merge_audit_weight,
            contrastive_weight=contrastive_weight,
        ),
        supervision=LocalLawSupervisionConfig(n_audit=n_audit),
    )


class _FixedScalarAdapter:
    name = "fixed_scalar"
    head_name = "count"
    supervision_mode = "sparse_local_law"

    def build_supervision_batch(self, batch_items):
        return TreeSupervisionBatch(
            mode="sparse_local_law",
            node_scalar_targets=(
                ScalarTarget(
                    node_ref=TreeNodeRef(0, 0),
                    value=0.6,
                    head="count",
                    normalized=True,
                    kind="leaf",
                    proxy_value=0.2,
                    oracle_value=0.6,
                    observed=True,
                    propensity=0.5,
                    local_law_adjustment=True,
                ),
            ),
            adapter_name=self.name,
        )

    def compute_auxiliary_losses(self, **kwargs):
        return {}


class _FixedScalarModel:
    has_phi = False

    def predict_normalized_batch(self, states, head="count"):
        return states[:, 0]


def test_tree_model_v2_corrected_local_law_target_uses_live_prediction_loss() -> None:
    state = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    trainer = TreeModelV2Trainer(
        model=_FixedScalarModel(),
        adapter=_FixedScalarAdapter(),
        forward_batch=lambda current_model, items: None,
        state_getter=lambda item, node_index: state,
        config=TreeModelV2TrainingConfig(
            score_targets=TreeModelV2ScoreTargetConfig(target_min=0.0, target_max=1.0),
            objective=TreeModelV2ObjectiveConfig(leaf_scalar_weight=1.0),
        ),
        device=torch.device("cpu"),
    )

    prepared = trainer.prepare_batch([object()])
    raw_loss, count = trainer._compiled_scalar_targets_loss(
        prepared,
        prepared.leaf_scalar_group,
    )

    assert count == 1
    # proxy=(0-.2)^2=.04, oracle=(0-.6)^2=.36, corrected=.04+(.36-.04)/.5=.68
    assert raw_loss.item() == pytest.approx(0.68)
    assert raw_loss.requires_grad is True


def test_real_document_tree_model_v2_trainer_builds_sparse_batch_and_loss() -> None:
    model = CTreePOModel(
        CTreePOConfig(
            embedding_dim=4,
            sketch_dim=8,
            hidden_dim=16,
            tree_model_version="v2",
        )
    )
    batch = [
        _make_labeled_tree("abcdefgh", 12.0),
        _make_labeled_tree("qrstuvwx", 42.0),
    ]
    trainer = TreeModelV2Trainer(
        model=model,
        adapter=RealDocumentTaskAdapter(
            max_leaf_targets_per_doc=2,
            max_internal_targets_per_doc=1,
            enable_fiber_constraints=True,
            fiber_same_threshold=5.0,
            fiber_diff_threshold=20.0,
        ),
        forward_batch=lambda current_model, items: forward_ctreepo_batch(
            current_model,
            [nodes for nodes, _rile, _doc_id in items],
        ),
        state_getter=lambda item, node_index: item[0][node_index].sketch,
        config=_v2_config(),
        device=torch.device("cpu"),
    )

    prepared = trainer.prepare_batch(batch)
    loss_sum, n_terms, stats = trainer.compute_supervision_loss(prepared)

    assert prepared.supervision_batch.mode == "sparse_local_law"
    assert len(prepared.supervision_batch.root_scalar_targets) == 2
    assert len(prepared.supervision_batch.node_scalar_targets) == 6
    assert len(prepared.supervision_batch.fiber_pair_targets) == 1
    assert stats["supervision_mode"] == "sparse_local_law"
    assert stats["adapter_name"] == "real_document_rile"
    assert stats["root_scalar_target_count"] == 2
    assert stats["node_scalar_target_count"] == 6
    assert stats["expanded_fiber_pair_count"] == 1
    assert stats["uses_fiber_supervision"] is True
    assert n_terms == 2
    assert stats["loss_term_count"] == 2
    assert float(loss_sum.item()) >= 0.0


def test_tree_model_v2_trainer_accepts_packed_forward_results() -> None:
    model = CTreePOModel(
        CTreePOConfig(
            embedding_dim=4,
            sketch_dim=8,
            hidden_dim=16,
            tree_model_version="v2",
        )
    )
    batch = [
        _make_labeled_tree("abcdefgh", 12.0),
        _make_labeled_tree("qrstuvwx", 42.0),
    ]

    def _forward_packed(current_model, items):
        packed_trees = [build_packed_embedding_tree(nodes) for nodes, _rile, _doc_id in items]
        packed_batch = build_packed_tree_batch(packed_trees, device=torch.device("cpu"))
        return forward_packed_tree_batch(current_model, packed_batch, materialize_nodes=False)

    trainer = TreeModelV2Trainer(
        model=model,
        adapter=RealDocumentTaskAdapter(
            max_leaf_targets_per_doc=2,
            max_internal_targets_per_doc=1,
            rng=random.Random(0),
            enable_fiber_constraints=True,
            fiber_same_threshold=5.0,
            fiber_diff_threshold=20.0,
        ),
        forward_batch=_forward_packed,
        state_getter=lambda item, node_index: item[0][node_index].sketch,
        config=_v2_config(),
        device=torch.device("cpu"),
    )

    prepared = trainer.prepare_batch(batch)
    loss_sum, n_terms, stats = trainer.compute_supervision_loss(prepared)

    assert isinstance(prepared.forward_result, PackedForwardResult)
    assert prepared.state_batch is not None
    assert prepared.ref_index
    assert stats["expanded_fiber_pair_count"] == 1
    assert n_terms == 2
    assert stats["loss_term_count"] == 2
    assert float(loss_sum.item()) >= 0.0


def test_tree_model_v2_local_supervision_does_not_dilute_root_normalization() -> None:
    model = CTreePOModel(
        CTreePOConfig(
            embedding_dim=4,
            sketch_dim=8,
            hidden_dim=16,
            tree_model_version="v2",
        )
    )
    batch = [
        _make_labeled_tree("abcdefgh", 12.0),
        _make_labeled_tree("qrstuvwx", 42.0),
    ]
    shared_kwargs = dict(
        model=model,
        forward_batch=lambda current_model, items: forward_ctreepo_batch(
            current_model,
            [nodes for nodes, _rile, _doc_id in items],
        ),
        state_getter=lambda item, node_index: item[0][node_index].sketch,
        config=_v2_config(),
        device=torch.device("cpu"),
    )
    root_only_trainer = TreeModelV2Trainer(
        adapter=RealDocumentTaskAdapter(
            max_leaf_targets_per_doc=0,
            max_internal_targets_per_doc=0,
            enable_fiber_constraints=False,
        ),
        **shared_kwargs,
    )
    local_trainer = TreeModelV2Trainer(
        adapter=RealDocumentTaskAdapter(
            max_leaf_targets_per_doc=2,
            max_internal_targets_per_doc=1,
            enable_fiber_constraints=True,
            fiber_same_threshold=5.0,
            fiber_diff_threshold=20.0,
        ),
        **shared_kwargs,
    )

    prepared_root_only = root_only_trainer.prepare_batch(batch)
    prepared_local = local_trainer.prepare_batch(batch)
    root_only_loss, root_only_terms, root_only_stats = root_only_trainer.compute_supervision_loss(
        prepared_root_only
    )
    local_loss, local_terms, local_stats = local_trainer.compute_supervision_loss(
        prepared_local
    )

    assert root_only_terms == 2
    assert local_terms == 2
    assert root_only_stats["root_scalar_target_count"] == 2
    assert local_stats["root_scalar_target_count"] == 2
    assert local_stats["node_scalar_target_count"] > root_only_stats["node_scalar_target_count"]
    assert local_stats["expanded_fiber_pair_count"] > root_only_stats["expanded_fiber_pair_count"]
    assert local_stats["root_scalar_loss"] == pytest.approx(root_only_stats["root_scalar_loss"])
    assert float(local_loss.item()) > float(root_only_loss.item())


def test_ctreepo_trainer_v2_path_uses_shared_fiber_constraints_and_backpropagates() -> None:
    trainer = CTreePOTrainer(
        _ctreepo_config(
            model=CTreePOConfig(
                embedding_dim=4,
                sketch_dim=8,
                hidden_dim=16,
                tree_model_version="v2",
            ),
            batch_size=2,
            n_audit=2,
            leaf_audit_weight=0.1,
            merge_audit_weight=0.2,
            contrastive_weight=0.1,
            device="cpu",
        ),
        node_oracle_predictor=lambda text: float(len(text)),
        node_oracle_source_kind="oracle_callback",
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                [(0, 4), (4, 8)],
                "abcdefgh",
                10.0,
            ),
            "doc2": (
                [[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]],
                [(0, 4), (4, 8)],
                "qrstuvwx",
                12.0,
            ),
            "doc3": (
                [[0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3]],
                [(0, 4), (4, 8)],
                "ijklmnop",
                70.0,
            ),
        },
        split="train",
    )
    optimizer = trainer._make_optimizer()
    loss = trainer.train_step(trainer.train_trees, optimizer)

    assert trainer.model.has_phi is True
    assert trainer._use_shared_fiber_constraints is True
    assert trainer._last_shared_supervision_stats["supervision_mode"] == "sparse_local_law"
    assert trainer._last_shared_supervision_stats["uses_fiber_supervision"] is True
    assert trainer._last_shared_supervision_stats["same_fiber_pair_count"] >= 1
    assert trainer._last_shared_supervision_stats["different_fiber_pair_count"] >= 1
    assert trainer._last_wrapper_regularization_stats["legacy_sketch_contrastive_active"] is False
    leaf_grads = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in trainer.model.leaf_projector.parameters()
    )
    merge_grads = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in trainer.model.merge_module.parameters()
    )
    phi_grads = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in trainer.model.phi_projector.parameters()
    )
    assert leaf_grads
    assert merge_grads
    assert phi_grads
    assert loss >= 0.0


def test_ctreepo_trainer_legacy_path_keeps_sketch_contrastive_only_for_legacy() -> None:
    trainer = CTreePOTrainer(
        _ctreepo_config(
            model=CTreePOConfig(
                embedding_dim=4,
                sketch_dim=8,
                hidden_dim=16,
                tree_model_version="legacy",
            ),
            batch_size=2,
            contrastive_weight=0.1,
            device="cpu",
        ),
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                [(0, 4), (4, 8)],
                "abcdefgh",
                10.0,
            ),
            "doc2": (
                [[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]],
                [(0, 4), (4, 8)],
                "qrstuvwx",
                12.0,
            ),
        },
        split="train",
    )
    optimizer = trainer._make_optimizer()
    loss = trainer.train_step(trainer.train_trees, optimizer)

    assert trainer.model.has_phi is False
    assert trainer._use_shared_fiber_constraints is False
    assert trainer._last_shared_supervision_stats["uses_fiber_supervision"] is False
    assert trainer._last_wrapper_regularization_stats["legacy_sketch_contrastive_active"] is True
    assert trainer._last_wrapper_regularization_stats["legacy_sketch_contrastive_term_count"] == 1


def test_ctreepo_trainer_records_reproducibility_in_training_result(tmp_path) -> None:
    trainer = CTreePOTrainer(
        _ctreepo_config(
            model=CTreePOConfig(
                embedding_dim=4,
                sketch_dim=8,
                hidden_dim=16,
                tree_model_version="v2",
            ),
            eval_every=1,
            batch_size=1,
            seed=19,
            device="cpu",
        ),
    )

    trainer.prepare_trees_from_precomputed(
        {
            "doc1": (
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                [(0, 4), (4, 8)],
                "abcdefgh",
                10.0,
            ),
            "doc2": (
                [[0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2]],
                [(0, 4), (4, 8)],
                "qrstuvwx",
                12.0,
            ),
        },
        split="train",
    )

    result = trainer.train(output_dir=tmp_path)

    assert result.reproducibility["seed"] == 19
    assert result.reproducibility["torch_seed_applied"] is True
    assert (tmp_path / "training_result.json").exists()
