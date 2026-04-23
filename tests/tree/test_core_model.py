"""Tests for the shared tree-neural core model."""

import torch
import pytest

from src.tree.core_model import (
    EmbeddingProjectorBackend,
    PhiProjector,
    ScalarReadoutHead,
    ScoreFiberConfig,
    TreeNeuralCore,
    TreeNeuralCoreConfig,
)


# ---------------------------------------------------------------------------
# EncoderBackend
# ---------------------------------------------------------------------------


class TestEmbeddingProjectorBackend:
    def test_output_shape(self):
        enc = EmbeddingProjectorBackend(embedding_dim=64, state_dim=16, hidden_dim=32)
        out = enc.encode(embeddings=torch.randn(5, 64))
        assert out.shape == (5, 16)

    def test_state_dim_property(self):
        enc = EmbeddingProjectorBackend(embedding_dim=64, state_dim=16)
        assert enc.state_dim == 16

    def test_gradient_flow(self):
        enc = EmbeddingProjectorBackend(embedding_dim=64, state_dim=16, hidden_dim=32)
        emb = torch.randn(3, 64, requires_grad=True)
        out = enc.encode(embeddings=emb)
        out.sum().backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0

    def test_rejects_missing_embeddings(self):
        enc = EmbeddingProjectorBackend(embedding_dim=64, state_dim=16)
        with pytest.raises(ValueError, match="requires embeddings"):
            enc.encode(token_ids=[[1, 2, 3]])

    def test_forward_matches_encode(self):
        enc = EmbeddingProjectorBackend(embedding_dim=64, state_dim=16, hidden_dim=32)
        emb = torch.randn(3, 64)
        assert torch.allclose(enc.forward(emb), enc.encode(embeddings=emb))


# ---------------------------------------------------------------------------
# ScoreFiberConfig
# ---------------------------------------------------------------------------


class TestScoreFiberConfig:
    def test_valid_config(self):
        cfg = ScoreFiberConfig(phi_dim=48, score_dim=1, fiber_dim=47, aux_dim=0)
        assert cfg.phi_dim == 48

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError, match="!="):
            ScoreFiberConfig(phi_dim=48, score_dim=1, fiber_dim=10, aux_dim=0)

    def test_with_aux(self):
        cfg = ScoreFiberConfig(phi_dim=48, score_dim=1, fiber_dim=31, aux_dim=16)
        assert cfg.score_dim + cfg.fiber_dim + cfg.aux_dim == cfg.phi_dim


# ---------------------------------------------------------------------------
# PhiProjector
# ---------------------------------------------------------------------------


class TestPhiProjector:
    def setup_method(self):
        self.cfg = ScoreFiberConfig(phi_dim=16, score_dim=1, fiber_dim=12, aux_dim=3, hidden_dim=32)
        self.proj = PhiProjector(state_dim=8, config=self.cfg)

    def test_output_shape(self):
        phi = self.proj(torch.randn(5, 8))
        assert phi.shape == (5, 16)

    def test_score_slice_shape(self):
        phi = self.proj(torch.randn(5, 8))
        assert self.proj.score(phi).shape == (5, 1)

    def test_fiber_slice_shape(self):
        phi = self.proj(torch.randn(5, 8))
        assert self.proj.fiber(phi).shape == (5, 12)

    def test_aux_slice_shape(self):
        phi = self.proj(torch.randn(5, 8))
        assert self.proj.aux(phi).shape == (5, 3)

    def test_fiber_aux_shape(self):
        phi = self.proj(torch.randn(5, 8))
        assert self.proj.fiber_aux(phi).shape == (5, 15)  # 12 + 3

    def test_slices_cover_full_phi(self):
        phi = self.proj(torch.randn(3, 8))
        reconstructed = torch.cat(
            [self.proj.score(phi), self.proj.fiber(phi), self.proj.aux(phi)],
            dim=-1,
        )
        assert torch.allclose(phi, reconstructed)

    def test_gradient_flows_through_all_slices(self):
        state = torch.randn(3, 8, requires_grad=True)
        phi = self.proj(state)

        # Score slice gradient
        self.proj.score(phi).sum().backward(retain_graph=True)
        assert state.grad is not None
        score_grad = state.grad.clone()
        state.grad = None

        # Fiber slice gradient
        self.proj.fiber(phi).sum().backward(retain_graph=True)
        assert state.grad is not None
        fiber_grad = state.grad.clone()
        state.grad = None

        # Both should have non-zero gradients
        assert score_grad.abs().sum() > 0
        assert fiber_grad.abs().sum() > 0

    def test_no_aux_fiber_aux_equals_fiber(self):
        cfg = ScoreFiberConfig(phi_dim=16, score_dim=1, fiber_dim=15, aux_dim=0, hidden_dim=32)
        proj = PhiProjector(state_dim=8, config=cfg)
        phi = proj(torch.randn(3, 8))
        assert torch.allclose(proj.fiber_aux(phi), proj.fiber(phi))


# ---------------------------------------------------------------------------
# ScalarReadoutHead
# ---------------------------------------------------------------------------


class TestScalarReadoutHead:
    def test_output_range(self):
        head = ScalarReadoutHead(input_dim=8, target_min=-100.0, target_max=100.0)
        out = head(torch.randn(100, 8))
        assert out.min() >= -100.0
        assert out.max() <= 100.0

    def test_normalized_range(self):
        head = ScalarReadoutHead(input_dim=8)
        out = head.forward_normalized(torch.randn(100, 8))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batched_shape(self):
        head = ScalarReadoutHead(input_dim=8)
        assert head(torch.randn(5, 8)).shape == (5, 1)


# ---------------------------------------------------------------------------
# TreeNeuralCore
# ---------------------------------------------------------------------------


class TestTreeNeuralCore:
    def test_basic_creation(self):
        cfg = TreeNeuralCoreConfig(state_dim=16, merge_type="gated")
        core = TreeNeuralCore(cfg)
        assert core.phi_projector is None  # No phi by default

    def test_with_phi(self):
        phi_cfg = ScoreFiberConfig(phi_dim=16, score_dim=1, fiber_dim=15)
        cfg = TreeNeuralCoreConfig(state_dim=32, phi_config=phi_cfg)
        core = TreeNeuralCore(cfg)
        assert core.phi_projector is not None

    def test_merge_output_shape(self):
        cfg = TreeNeuralCoreConfig(state_dim=16, merge_type="gated")
        core = TreeNeuralCore(cfg)
        left = torch.randn(3, 16)
        right = torch.randn(3, 16)
        merged = core.merge(left, right)
        assert merged.shape == (3, 16)

    def test_merge_batch(self):
        cfg = TreeNeuralCoreConfig(state_dim=16, merge_type="gated")
        core = TreeNeuralCore(cfg)
        left = torch.randn(5, 16)
        right = torch.randn(5, 16)
        merged = core.merge_batch(left, right)
        assert merged.shape == (5, 16)

    def test_predict(self):
        cfg = TreeNeuralCoreConfig(state_dim=16, head_names=("rile",))
        core = TreeNeuralCore(cfg)
        pred = core.predict(torch.randn(3, 16), "rile")
        assert pred.shape == (3, 1)

    def test_predict_normalized(self):
        cfg = TreeNeuralCoreConfig(state_dim=16)
        core = TreeNeuralCore(cfg)
        pred = core.predict_normalized(torch.randn(3, 16))
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_predict_batch(self):
        cfg = TreeNeuralCoreConfig(state_dim=16)
        core = TreeNeuralCore(cfg)
        pred = core.predict_batch(torch.randn(10, 16))
        assert pred.shape == (10, 1)

    def test_predict_confidence_range(self):
        cfg = TreeNeuralCoreConfig(state_dim=16)
        core = TreeNeuralCore(cfg)
        conf = core.predict_confidence(torch.randn(10, 16))
        assert conf.min() >= -1.0  # theoretical range
        assert conf.max() <= 1.0

    def test_phi_returns_none_without_config(self):
        cfg = TreeNeuralCoreConfig(state_dim=16)
        core = TreeNeuralCore(cfg)
        assert core.phi(torch.randn(3, 16)) is None

    def test_phi_returns_tensor_with_config(self):
        phi_cfg = ScoreFiberConfig(phi_dim=8, score_dim=1, fiber_dim=7, hidden_dim=16)
        cfg = TreeNeuralCoreConfig(state_dim=16, phi_config=phi_cfg)
        core = TreeNeuralCore(cfg)
        phi = core.phi(torch.randn(3, 16))
        assert phi is not None
        assert phi.shape == (3, 8)

    def test_phi_score_and_fiber(self):
        phi_cfg = ScoreFiberConfig(phi_dim=8, score_dim=1, fiber_dim=7, hidden_dim=16)
        cfg = TreeNeuralCoreConfig(state_dim=16, phi_config=phi_cfg)
        core = TreeNeuralCore(cfg)
        state = torch.randn(3, 16)
        score = core.phi_score(state)
        fiber = core.phi_fiber(state)
        assert score.shape == (3, 1)
        assert fiber.shape == (3, 7)

    def test_all_merge_types(self):
        for merge_type in ("gated", "mlp", "avg", "residual_gated", "bilinear"):
            cfg = TreeNeuralCoreConfig(state_dim=16, merge_type=merge_type)
            core = TreeNeuralCore(cfg)
            merged = core.merge(torch.randn(2, 16), torch.randn(2, 16))
            assert merged.shape == (2, 16), f"Failed for merge_type={merge_type}"

    def test_full_gradient_flow(self):
        """Verify gradients flow from loss through predict, merge, and phi."""
        phi_cfg = ScoreFiberConfig(phi_dim=8, score_dim=1, fiber_dim=7, hidden_dim=16)
        cfg = TreeNeuralCoreConfig(state_dim=16, phi_config=phi_cfg)
        core = TreeNeuralCore(cfg)

        # Simulate a mini tree: 2 leaves → 1 root
        left = torch.randn(1, 16, requires_grad=True)
        right = torch.randn(1, 16, requires_grad=True)

        root = core.merge(left, right)
        pred = core.predict_normalized(root, "rile")

        # Also compute phi on the root for fiber loss
        phi = core.phi(root)
        fiber = core.phi_projector.fiber(phi)

        loss = pred.sum() + fiber.sum()
        loss.backward()

        # Gradients should reach leaves
        assert left.grad is not None and left.grad.abs().sum() > 0
        assert right.grad is not None and right.grad.abs().sum() > 0

        # And model parameters
        grad_exists = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in core.parameters()
        )
        assert grad_exists

    def test_encoder_plus_core_end_to_end(self):
        """Full pipeline: encoder → core merge → predict."""
        enc = EmbeddingProjectorBackend(embedding_dim=64, state_dim=16, hidden_dim=32)
        phi_cfg = ScoreFiberConfig(phi_dim=8, score_dim=1, fiber_dim=7, hidden_dim=16)
        cfg = TreeNeuralCoreConfig(state_dim=16, phi_config=phi_cfg)
        core = TreeNeuralCore(cfg)

        # Encode 4 leaves
        embs = torch.randn(4, 64)
        states = enc.encode(embeddings=embs)
        assert states.shape == (4, 16)

        # Merge pairs: (0,1) and (2,3)
        left = states[:2]
        right = states[2:]
        merged = core.merge_batch(left, right)
        assert merged.shape == (2, 16)

        # Merge to root
        root = core.merge(merged[0:1], merged[1:2])
        assert root.shape == (1, 16)

        # Predict
        pred = core.predict(root, "rile")
        assert pred.shape == (1, 1)

        # Phi
        phi = core.phi(root)
        assert phi.shape == (1, 8)

    def test_gradient_through_encoder_and_core(self):
        """Verify gradients flow from core loss back through encoder."""
        enc = EmbeddingProjectorBackend(embedding_dim=32, state_dim=8, hidden_dim=16)
        cfg = TreeNeuralCoreConfig(state_dim=8, merge_type="gated")
        core = TreeNeuralCore(cfg)

        embs = torch.randn(2, 32, requires_grad=True)
        states = enc.encode(embeddings=embs)
        root = core.merge(states[0:1], states[1:2])
        loss = core.predict_normalized(root).sum()
        loss.backward()

        assert embs.grad is not None and embs.grad.abs().sum() > 0
        enc_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
        core_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in core.parameters())
        assert enc_grads
        assert core_grads
