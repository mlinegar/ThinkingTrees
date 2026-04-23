from __future__ import annotations

import torch

from src.tree.core_model import (
    EmbeddingProjectorBackend,
    ScoreFiberConfig,
    TokenSequenceEncoderBackend,
    TreeStateCore,
    TreeStateCoreConfig,
)
from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel
from src.tree.tree_model_v2 import TreeModelV2


def test_token_sequence_encoder_backend_wraps_callable_batch_encoder() -> None:
    def _encode_tokens_batch(token_ids, *, device):
        lengths = torch.tensor(
            [len(tokens) for tokens in token_ids],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)
        return lengths.repeat(1, 4)

    backend = TokenSequenceEncoderBackend(
        state_dim=4,
        encode_tokens_batch=_encode_tokens_batch,
    )
    encoded = backend.encode(token_ids=((1, 2, 3), (4,)), device=torch.device("cpu"))

    assert encoded.shape == (2, 4)
    assert backend.leaf_state_dim == 4
    assert torch.allclose(encoded[0], torch.tensor([3.0, 3.0, 3.0, 3.0]))


def test_tree_model_v2_composes_embedding_backend_and_shared_core() -> None:
    encoder = EmbeddingProjectorBackend(embedding_dim=6, state_dim=8, hidden_dim=12)
    core = TreeStateCore(
        TreeStateCoreConfig(
            state_dim=8,
            merge_type="gated",
            phi_config=ScoreFiberConfig(phi_dim=10, score_dim=1, fiber_dim=9),
            head_names=("rile",),
        )
    )
    model = TreeModelV2(
        encoder_backend=encoder,
        state_core=core,
        default_head="rile",
    )

    leaves = model.encode_leaf_batch(torch.randn(3, 6))
    merged = model.merge_batch(leaves[:2], leaves[1:])
    pred = model.predict_batch(merged)
    phi = model.phi_batch(merged)

    assert leaves.shape == (3, 8)
    assert merged.shape == (2, 8)
    assert pred.shape == (2, 1)
    assert phi is not None
    assert phi.shape == (2, 10)


def test_ctreepo_model_v2_auto_enables_score_fiber_surface() -> None:
    model = CTreePOModel(
        CTreePOConfig(
            embedding_dim=6,
            sketch_dim=8,
            hidden_dim=12,
            phi_dim=0,
            tree_model_version="v2",
        )
    )
    tree_model = model.as_tree_model_v2()
    leaves = tree_model.encode_leaf_batch(torch.randn(2, 6))
    phi = tree_model.phi_batch(leaves)

    assert model.uses_tree_model_v2 is True
    assert model.has_phi is True
    assert phi is not None
    assert phi.shape[0] == 2
    assert phi.shape[1] >= 16
