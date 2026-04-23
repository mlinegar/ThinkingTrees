from __future__ import annotations

import pytest
import torch

from src.ctreepo.sim.core.markov_changepoint_ops_count import ChangepointMarkovDoc
from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    FNOCountSketch,
    HAS_NEURAL_OPERATOR,
    _precompute_balanced_doc_state_views,
    _prepare_fno_count_docs,
)
from src.training.tree_model_v2_trainer import (
    MarkovTaskAdapter,
    TreeModelV2Trainer,
    TreeModelV2ObjectiveConfig,
    TreeModelV2ScoreTargetConfig,
    TreeModelV2TrainingConfig,
    build_markov_dense_supervision_batch,
)


def test_markov_task_adapter_builds_dense_supervision_batch() -> None:
    docs = [
        ChangepointMarkovDoc(
            tokens=(0, 1, 1, 2, 2, 3, 3, 3),
            token_regimes=(0, 0, 1, 1, 2, 2, 2, 3),
            transition_regimes=(0, 0, 1, 1, 2, 2, 2, 3),
            true_boundaries=(2, 4, 7),
        ),
        ChangepointMarkovDoc(
            tokens=(1, 1, 2, 2, 2, 4, 4, 5),
            token_regimes=(1, 1, 1, 2, 2, 2, 3, 3),
            transition_regimes=(1, 1, 1, 2, 2, 2, 3, 3),
            true_boundaries=(3, 6),
        ),
    ]
    fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=2)
    batch = build_markov_dense_supervision_batch(
        fno_docs,
        theorem_feature_adapter_name="markov_score_endpoints",
        target_scale=8.0,
    )

    assert batch.mode == "dense_local_law"
    assert len(batch.root_scalar_targets) == 2
    assert len(batch.node_scalar_targets) == sum(
        len(doc.leaf_token_ids) + len(doc.merge_counts_balanced)
        for doc in fno_docs
    )
    assert len(batch.fiber_pair_targets) > 0
    aux_names = {target.name for target in batch.auxiliary_targets}
    assert {"first_regime", "last_regime", "join_bit"} <= aux_names


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_markov_tree_model_v2_trainer_backpropagates_through_shared_dense_batch() -> None:
    class _DocWithView:
        def __init__(self, doc):
            self.doc = doc
            self.view = None

        def __getattr__(self, name):
            return getattr(self.doc, name)

    docs = [
        ChangepointMarkovDoc(
            tokens=(0, 1, 1, 2, 2, 3, 3, 3),
            token_regimes=(0, 0, 1, 1, 2, 2, 2, 3),
            transition_regimes=(0, 0, 1, 1, 2, 2, 2, 3),
            true_boundaries=(2, 4, 7),
        ),
        ChangepointMarkovDoc(
            tokens=(1, 1, 2, 2, 2, 4, 4, 5),
            token_regimes=(1, 1, 1, 2, 2, 2, 3, 3),
            transition_regimes=(1, 1, 1, 2, 2, 2, 3, 3),
            true_boundaries=(3, 6),
        ),
    ]
    fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=2)
    batch_items = [_DocWithView(doc) for doc in fno_docs]
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=2,
        state_dim=8,
        hidden_dim=16,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        theorem_surface_mode="factorized_score_fiber",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
        theorem_score_dim=1,
        theorem_fiber_dim=15,
        tree_model_version="v2",
    )
    tree_model = model.as_tree_model_v2()
    trainer = TreeModelV2Trainer(
        model=tree_model,
        adapter=MarkovTaskAdapter(
            theorem_feature_adapter_name="markov_score_endpoints",
            target_scale=8.0,
        ),
        forward_batch=lambda _current_model, items: [
            setattr(batch_item, "view", view)
            for batch_item, view in zip(
                items,
                _precompute_balanced_doc_state_views(
                    model,
                    [item.doc for item in items],
                    device=torch.device("cpu"),
                    collect_merge_states=True,
                ),
            )
        ],
        state_getter=lambda item, node_index: (
            item.view.state_batch[node_index]
            if node_index < int(item.view.state_batch.shape[0])
            else item.view.merge_states[node_index - int(item.view.state_batch.shape[0])]
        ),
        config=TreeModelV2TrainingConfig(
            score_targets=TreeModelV2ScoreTargetConfig(target_min=0.0, target_max=8.0),
            objective=TreeModelV2ObjectiveConfig(
                root_weight=1.0,
                leaf_scalar_weight=0.2,
                internal_scalar_weight=0.2,
                fiber_same_weight=0.1,
                fiber_different_weight=0.1,
            ),
        ),
        device=torch.device("cpu"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    prepared = trainer.prepare_batch(batch_items)
    loss_sum, n_terms, stats = trainer.compute_supervision_loss(prepared)
    loss = loss_sum / float(max(1, n_terms))
    loss.backward()

    encoder_grads = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for module in (model.token_embedding, model.fno_encoder, model.leaf_proj)
        for param in module.parameters()
    )
    merge_grads = bool(model.merger is not None) and any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.merger.parameters()
    )
    phi_grads = bool(model.phi_projector is not None) and any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in model.phi_projector.parameters()
    )

    assert stats["supervision_mode"] == "dense_local_law"
    assert stats["adapter_name"] == "markov_count_tree"
    assert stats["root_scalar_target_count"] == 2
    assert stats["node_scalar_target_count"] == sum(
        len(doc.leaf_token_ids) + len(doc.merge_counts_balanced)
        for doc in fno_docs
    )
    assert stats["uses_fiber_supervision"] is True
    assert stats["expanded_fiber_pair_count"] > 0
    assert n_terms > 0
    assert float(loss.item()) >= 0.0
    assert encoder_grads
    assert merge_grads
    assert phi_grads
