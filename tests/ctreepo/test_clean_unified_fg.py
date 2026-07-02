"""Tests for the clean f/g composition model in clean_unified_fg.py."""

from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.ctreepo.sim.core.clean_unified_fg import (
    CleanLeafEncoder,
    CleanLeafTokenEmbedding,
    CleanMergeG,
    CleanScorerF,
    CleanScorerFNO,
    CleanUnifiedFG,
    CleanUnifiedG,
    CleanUnifiedNO,
    ExactZeroMarkovFG,
    TreeForwardOutputNO,
    leaf_mse_loss,
    LearnedLocalLawMarkovFG,
    LearnedMarkovSketchLeafEncoder,
    MarkovSketchLeafEncoder,
    MarkovSketchMergeG,
    merge_mse_loss,
    root_mse_loss,
)
from src.ctreepo.sim.core.markov_local_laws import markov_approx_local_laws_bundle
from scripts.probe_clean_unified_no import (
    _balanced_merge_state_triples,
    _eval_markov_node_witness_head,
    _markov_node_witness_targets_for_leaves,
)


def _build_model(fno_width: int = 64, **kwargs) -> CleanUnifiedFG:
    return CleanUnifiedFG(
        vocab_size=16,
        target_scale=10.0,
        fno_width=fno_width,
        fno_n_modes=4,
        fno_n_layers=2,
        **kwargs,
    )


class TestCleanLeafEncoder:
    def test_output_shape_equals_fno_width(self) -> None:
        torch.manual_seed(0)
        enc = CleanLeafEncoder(
            vocab_size=16, fno_width=32, fno_n_modes=4, fno_n_layers=2,
        )
        toks = torch.randint(0, 16, (4, 32))
        state = enc(toks)
        # state_dim is fno_width by construction (no pool-to-state layer)
        assert state.shape == (4, 32)
        assert enc.state_dim == 32

    def test_pooling_mode_validation(self) -> None:
        with pytest.raises(ValueError, match="pooling_mode"):
            CleanLeafEncoder(
                vocab_size=16, fno_width=64, fno_n_modes=4, fno_n_layers=2,
                pooling_mode="max",
            )

    def test_padding_aware_default_mask(self) -> None:
        torch.manual_seed(0)
        enc = CleanLeafEncoder(
            vocab_size=16, fno_width=64, fno_n_modes=4, fno_n_layers=2,
            pooling_mode="sum",
        )
        toks = torch.randint(0, 16, (2, 16))
        toks[1, 12:] = enc.pad_id
        state_default = enc(toks)
        explicit_mask = (toks != enc.pad_id).to(torch.float32)
        state_explicit = enc(toks, token_mask=explicit_mask)
        assert torch.allclose(state_default, state_explicit)

    def test_only_three_modules_present(self) -> None:
        # The leaf encoder should literally be: token_embedding + FNO + (no pool layer).
        # Pooling is done inline via apply_fno_token_encoder. So only two
        # learnable submodules: the embedding and the FNO.
        enc = CleanLeafEncoder(
            vocab_size=16, fno_width=32, fno_n_modes=4, fno_n_layers=2,
        )
        learnable_children = [
            (name, mod) for name, mod in enc.named_children()
            if any(p.requires_grad for p in mod.parameters(recurse=True))
        ]
        names = sorted(name for name, _ in learnable_children)
        assert names == ["fno", "token_embedding"], (
            f"expected only token_embedding and fno; got {names}"
        )


class TestCleanMergeG:
    def test_output_shape(self) -> None:
        torch.manual_seed(0)
        g = CleanMergeG(state_dim=32)
        left = torch.randn(5, 32)
        right = torch.randn(5, 32)
        merged = g(left, right)
        assert merged.shape == (5, 32)

    def test_g_is_not_symmetric_by_default(self) -> None:
        # An honest merge: order matters (left/right are distinct).
        torch.manual_seed(0)
        g = CleanMergeG(state_dim=16)
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        m_ab = g(a, b)
        m_ba = g(b, a)
        assert not torch.allclose(m_ab, m_ba, atol=1e-6)

    def test_g_is_a_single_linear(self) -> None:
        # g is literally one nn.Linear; not an MLP.
        g = CleanMergeG(state_dim=32)
        children = list(g.children())
        assert len(children) == 1, f"g should have exactly one child module; got {len(children)}"
        assert isinstance(children[0], nn.Linear)
        assert children[0].in_features == 64  # 2 * state_dim
        assert children[0].out_features == 32


class TestCleanScorerF:
    def test_scalar_output(self) -> None:
        torch.manual_seed(0)
        f = CleanScorerF(state_dim=32)
        states = torch.randn(7, 32)
        counts = f(states)
        assert counts.shape == (7,)

    def test_f_applies_to_leaf_or_merge_state_indistinguishably(self) -> None:
        torch.manual_seed(0)
        f = CleanScorerF(state_dim=32)
        state = torch.randn(1, 32)
        out1 = f(state)
        out2 = f(state)
        assert torch.equal(out1, out2)

    def test_f_is_a_single_linear(self) -> None:
        # f is literally one nn.Linear; not an MLP.
        f = CleanScorerF(state_dim=32)
        children = list(f.children())
        assert len(children) == 1
        assert isinstance(children[0], nn.Linear)
        assert children[0].in_features == 32
        assert children[0].out_features == 1


class TestCleanUnifiedFGTreeComposition:
    def test_n_merges_equals_n_leaves_minus_one(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        for n_leaves in [1, 2, 3, 4, 5, 8, 16, 17]:
            toks = torch.randint(0, 16, (n_leaves, 16))
            out = m(toks)
            assert len(out.merge_states) == n_leaves - 1, (
                f"n_leaves={n_leaves}: expected {n_leaves - 1} merges, got {len(out.merge_states)}"
            )
            assert len(out.leaf_states) == n_leaves
            assert out.leaf_counts_norm.shape == (n_leaves,)
            assert out.merge_counts_norm.shape == (n_leaves - 1,)

    def test_single_leaf_doc_root_is_leaf(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (1, 16))
        out = m(toks)
        assert torch.equal(out.root_state, out.leaf_states[0])
        assert out.root_count_norm == out.leaf_counts_norm[0]

    def test_multi_leaf_root_is_last_merge(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        assert torch.equal(out.root_state, out.merge_states[-1])
        assert out.root_count_norm == out.merge_counts_norm[-1]

    def test_predict_count_unnormalizes(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        unnorm = m.predict_count(out.root_state)
        norm = out.root_count_norm
        assert torch.allclose(unnorm, norm * m.target_scale)

    def test_three_named_submodules_present(self) -> None:
        m = _build_model()
        assert isinstance(m.leaf_encoder, CleanLeafEncoder)
        assert isinstance(m.g, CleanMergeG)
        assert isinstance(m.f, CleanScorerF)

    def test_f_at_root_matches_root_count(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        # Apply f directly to the root state -- should match the recorded
        # root_count_norm (since forward_doc applied f to every node).
        f_at_root = m.f(out.root_state.unsqueeze(0)).squeeze(0)
        assert torch.allclose(f_at_root, out.root_count_norm)

    def test_f_at_leaf_matches_leaf_counts(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        # Apply f directly to a leaf state -- should match the recorded
        # leaf prediction (f operates on state, doesn't care where it came from).
        leaf_state = out.leaf_states[2].unsqueeze(0)
        f_at_leaf = m.f(leaf_state).squeeze(0)
        assert torch.allclose(f_at_leaf, out.leaf_counts_norm[2])

    def test_g_then_f_works(self) -> None:
        # The whole point: f composes with g. f(g(a, b)) is well-defined.
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (2, 16))
        out = m(toks)
        # Manually apply g then f on the two leaf states
        manual_merge = m.g(
            out.leaf_states[0].unsqueeze(0),
            out.leaf_states[1].unsqueeze(0),
        )
        manual_f = m.f(manual_merge).squeeze(0)
        # Should match the recorded root prediction
        assert torch.allclose(manual_f, out.root_count_norm)

    def test_backprop_reaches_all_parameters(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        loss = out.root_count_norm ** 2 + out.leaf_counts_norm.sum() + out.merge_counts_norm.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"


class TestLossHelpers:
    def test_root_mse_zero_at_perfect_prediction(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        # Use the model's own prediction as the "true" count
        true_root = out.root_count_norm.detach() * m.target_scale
        loss = root_mse_loss(out, root_count=true_root, target_scale=m.target_scale)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_leaf_mse_observed_mask_filters_correctly(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        # All-observed should equal mean; mask of (1, 0, 0, 0) should equal first leaf's loss
        true_counts = torch.zeros(4)
        loss_all = leaf_mse_loss(out, leaf_counts=true_counts, target_scale=m.target_scale)
        loss_first = leaf_mse_loss(
            out, leaf_counts=true_counts,
            leaf_observed=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            target_scale=m.target_scale,
        )
        # First-only should equal sq_err on leaf 0 alone
        expected_first = (out.leaf_counts_norm[0] - 0.0) ** 2
        assert torch.allclose(loss_first, expected_first, atol=1e-6)
        # Mean over 4 should differ from first-only
        assert not torch.allclose(loss_all, loss_first)

    def test_merge_mse_handles_zero_merges(self) -> None:
        torch.manual_seed(0)
        m = _build_model()
        toks = torch.randint(0, 16, (1, 16))
        out = m(toks)  # zero merges
        loss = merge_mse_loss(
            out, merge_counts=torch.empty(0), target_scale=m.target_scale
        )
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Exact-zero Markov control: canonical (count, first, last) sketch.
# ---------------------------------------------------------------------------


def _expected_transition_count(tokens: torch.Tensor, block_by_token: list[int]) -> int:
    blocks = [int(block_by_token[int(tok)]) for tok in tokens.flatten().tolist()]
    return sum(1 for left, right in zip(blocks[:-1], blocks[1:]) if left != right)


class TestExactZeroMarkovFG:
    def test_leaf_encoder_recovers_count_first_last_for_length_ladder(self) -> None:
        block_by_token = [0, 0, 1, 1]
        encoder = MarkovSketchLeafEncoder(
            block_by_token=block_by_token,
            target_scale=64.0,
            vocab_size=4,
            n_regimes=2,
        )
        for length in [1, 2, 4, 8, 16, 32, 64]:
            tokens = torch.tensor(
                [[0 if idx % 2 == 0 else 2 for idx in range(length)]],
                dtype=torch.long,
            )
            state = encoder(tokens)
            assert state.shape == (1, 5)
            assert (state[0, 0] * 64.0).item() == pytest.approx(length - 1)
            assert torch.equal(state[0, 1:3], torch.tensor([1.0, 0.0]))
            expected_last = torch.tensor([1.0, 0.0]) if length % 2 == 1 else torch.tensor([0.0, 1.0])
            assert torch.equal(state[0, 3:5], expected_last)

    def test_forward_doc_exact_root_across_balanced_tree(self) -> None:
        block_by_token = [0, 0, 1, 1]
        model = ExactZeroMarkovFG(
            block_by_token=block_by_token,
            target_scale=64.0,
            vocab_size=4,
            n_regimes=2,
            learnable_join=True,
        )
        leaf_tokens = torch.tensor(
            [
                [0, 0, 2, 2],
                [2, 0, 0, 0],
                [1, 1, 3, 3],
            ],
            dtype=torch.long,
        )
        expected = _expected_transition_count(leaf_tokens, block_by_token)
        out = model(leaf_tokens)
        assert len(out.leaf_states) == 3
        assert len(out.merge_states) == 2
        assert (out.root_count_norm * 64.0).item() == pytest.approx(expected)
        assert model.predict_count(out.root_state).item() == pytest.approx(expected)
        assert model.g.join_table.requires_grad

    def test_learnable_join_initializes_to_exact_merge(self) -> None:
        block_by_token = [0, 0, 1, 1]
        encoder = MarkovSketchLeafEncoder(
            block_by_token=block_by_token,
            target_scale=64.0,
            vocab_size=4,
            n_regimes=2,
        )
        learned_merge = MarkovSketchMergeG(
            n_regimes=2,
            target_scale=64.0,
            learnable_join=True,
        )
        exact_merge = MarkovSketchMergeG(
            n_regimes=2,
            target_scale=64.0,
            learnable_join=False,
        )
        tokens = torch.tensor(
            [
                [0, 2, 2, 2],
                [2, 2, 0, 0],
            ],
            dtype=torch.long,
        )
        states = encoder(tokens)
        learned = learned_merge(states[:1], states[1:])
        exact = exact_merge(states[:1], states[1:])
        assert torch.allclose(learned, exact)
        assert learned_merge.join_table.requires_grad
        assert not exact_merge.join_table.requires_grad


class TestLearnedLocalLawMarkovFG:
    def test_learned_leaf_projection_initializes_to_exact_state(self) -> None:
        block_by_token = [0, 0, 1, 1]
        encoder = LearnedMarkovSketchLeafEncoder(
            block_by_token=block_by_token,
            target_scale=64.0,
            vocab_size=4,
            n_regimes=2,
        )
        tokens = torch.tensor([[0, 2, 2, 1], [1, 1, 3, 0]], dtype=torch.long)
        learned = encoder(tokens)
        exact = encoder.exact_state(tokens)
        assert torch.allclose(learned, exact)
        assert encoder.projection.weight.requires_grad
        assert encoder.projection.bias.requires_grad

    def test_learned_fg_exact_initialized_rollout_has_zero_law_residuals(self) -> None:
        block_by_token = [0, 0, 1, 1]
        learned = LearnedLocalLawMarkovFG(
            block_by_token=block_by_token,
            target_scale=64.0,
            vocab_size=4,
            n_regimes=2,
        )
        exact = ExactZeroMarkovFG(
            block_by_token=block_by_token,
            target_scale=64.0,
            vocab_size=4,
            n_regimes=2,
            learnable_join=False,
        )
        leaf_tokens = torch.tensor(
            [
                [0, 0, 2, 2],
                [2, 0, 0, 0],
                [1, 1, 3, 3],
                [3, 3, 1, 0],
            ],
            dtype=torch.long,
        )
        learned_out = learned(leaf_tokens)
        exact_out = exact(leaf_tokens)
        assert torch.allclose(learned_out.root_state, exact_out.root_state)
        assert learned.predict_count(learned_out.root_state).item() == pytest.approx(
            exact.predict_count(exact_out.root_state).item()
        )
        assert learned.g.join_table.requires_grad

        leaf_pred = torch.stack(learned_out.leaf_states).detach().numpy()
        leaf_target = torch.stack(exact_out.leaf_states).detach().numpy()
        merge_pred = torch.stack(learned_out.merge_states).detach().numpy()
        merge_target = torch.stack(exact_out.merge_states).detach().numpy()
        bundle = markov_approx_local_laws_bundle(
            leaf_pred=leaf_pred,
            leaf_target=leaf_target,
            merge_pred=merge_pred,
            merge_target=merge_target,
            idempotence_pred=leaf_pred,
            idempotence_target=leaf_target,
        )
        assert bundle.eps_leaf == pytest.approx(0.0)
        assert bundle.eps_merge == pytest.approx(0.0)
        assert bundle.eps_idemp == pytest.approx(0.0)

    def test_learned_fg_rollout_ladder_matches_exact_initialization(self) -> None:
        block_by_token = [0, 0, 1, 1]
        model = LearnedLocalLawMarkovFG(
            block_by_token=block_by_token,
            target_scale=128.0,
            vocab_size=4,
            n_regimes=2,
        )
        for length in [1, 2, 4, 8, 16, 32, 64]:
            doc = torch.tensor(
                [[0 if idx % 2 == 0 else 2 for idx in range(length)]],
                dtype=torch.long,
            )
            out = model(doc)
            assert np.isfinite(out.root_count_norm.detach().numpy()).all()
            assert (out.root_count_norm * 128.0).item() == pytest.approx(length - 1)


# ---------------------------------------------------------------------------
# Operator-variant tests: state-as-function with FNO-based g and f.
# ---------------------------------------------------------------------------


def _build_no_model(channels: int = 16, **kwargs) -> CleanUnifiedNO:
    return CleanUnifiedNO(
        vocab_size=16,
        target_scale=10.0,
        channels=channels,
        g_n_modes=4,
        g_n_layers=2,
        scorer_n_modes=4,
        scorer_n_layers=2,
        **kwargs,
    )


class TestCleanLeafTokenEmbedding:
    def test_emits_function_shape(self) -> None:
        emb = CleanLeafTokenEmbedding(vocab_size=16, channels=16)
        toks = torch.randint(0, 16, (3, 32))
        out = emb(toks)
        assert out.shape == (3, 16, 32)


class TestCleanUnifiedG:
    def test_leaf_call_signature_matches_f_style_single_input(self) -> None:
        # g(content) means g(content, null), giving it the same direct-call
        # feel as f(state).
        torch.manual_seed(0)
        g = CleanUnifiedG(channels=16, n_modes=4, n_layers=2)
        content = torch.randn(2, 16, 64)
        out = g(content)
        assert out.shape == (2, 16, 64)

    def test_pair_input_signature_kept_for_compatibility(self) -> None:
        # A pre-built pair tensor is still accepted, but the reference lane
        # should usually call g(content) or g(left, right).
        torch.manual_seed(0)
        g = CleanUnifiedG(channels=16, n_modes=4, n_layers=2)
        x = torch.randn(2, 32, 64)  # (B, 2C, L)
        out = g(x)
        assert out.shape == (2, 16, 64)

    def test_merge_uses_same_g_call(self) -> None:
        # g.merge(left, right) is only a compatibility alias for g(left, right).
        torch.manual_seed(0)
        g = CleanUnifiedG(channels=16, n_modes=4, n_layers=2)
        left = torch.randn(1, 16, 32)
        right = torch.randn(1, 16, 32)
        out_via_merge = g.merge(left, right)
        out_via_direct_call = g(left, right)
        out_via_pair_input = g(torch.cat([left, right], dim=-2))
        assert torch.allclose(out_via_merge, out_via_direct_call)
        assert torch.allclose(out_via_merge, out_via_pair_input)

    def test_encode_leaf_uses_same_g_call(self) -> None:
        # g.encode_leaf(emb) is only a compatibility alias for g(emb).
        torch.manual_seed(0)
        g = CleanUnifiedG(channels=16, n_modes=4, n_layers=2)
        embedded = torch.randn(1, 16, 32)
        out_via_encode = g.encode_leaf(embedded)
        out_via_direct_call = g(embedded)
        zero_half = torch.zeros_like(embedded)
        out_via_pair_input = g(torch.cat([embedded, zero_half], dim=-2))
        assert torch.allclose(out_via_encode, out_via_direct_call)
        assert torch.allclose(out_via_encode, out_via_pair_input)

    def test_g_is_a_single_fno(self) -> None:
        g = CleanUnifiedG(channels=16, n_modes=4, n_layers=2)
        names = sorted(
            name for name, mod in g.named_children()
            if any(p.requires_grad for p in mod.parameters(recurse=True))
        )
        assert names == ["fno"]


class TestCleanScorerFNO:
    def test_scalar_output(self) -> None:
        torch.manual_seed(0)
        f = CleanScorerFNO(channels=16, n_modes=4, n_layers=2)
        states = torch.randn(5, 16, 32)
        out = f(states)
        assert out.shape == (5,)

    def test_f_is_fno_plus_linear_only(self) -> None:
        f = CleanScorerFNO(channels=16, n_modes=4, n_layers=2)
        names = sorted(
            name for name, mod in f.named_children()
            if any(p.requires_grad for p in mod.parameters(recurse=True))
        )
        assert names == ["fno", "linear"]


class TestCleanUnifiedNOSharedG:
    def test_three_named_submodules_present(self) -> None:
        m = _build_no_model()
        assert isinstance(m.token_embedding, CleanLeafTokenEmbedding)
        assert isinstance(m.g, CleanUnifiedG)
        assert isinstance(m.f, CleanScorerFNO)
        assert m.minimal_unified_gf_contract == (
            "z_x=g(embed(x),null); z_y=g(embed(y),null); "
            "z_xy=g(z_x,z_y); score=f(z_xy)"
        )

    def test_no_separate_learned_leaf_fno_beyond_token_embedding(self) -> None:
        m = _build_no_model()
        learned_children = [
            name for name, mod in m.named_children()
            if any(p.requires_grad for p in mod.parameters(recurse=True))
        ]
        assert learned_children == ["token_embedding", "g", "f"]
        assert not hasattr(m, "leaf_encoder")
        assert not hasattr(m, "fno_encoder")
        assert not hasattr(m, "leaf_proj")

    def test_g_at_leaves_is_same_instance_as_g_at_merges(self) -> None:
        # The whole point: leaves and merges go through the SAME g module.
        m = _build_no_model()
        # Only one g instance in the model
        gs = [mod for _, mod in m.named_modules() if isinstance(mod, CleanUnifiedG)]
        assert len(gs) == 1
        # And its parameters appear exactly once in the model's param list
        g_params = set(id(p) for p in m.g.parameters())
        all_params = set(id(p) for p in m.parameters())
        assert g_params.issubset(all_params)
        # Sanity: no duplicate g (which would inflate param count)
        n_g_params_via_module = sum(p.numel() for p in m.g.parameters())
        # Verify g.fno is referenced exactly once
        fno_modules = [mod for _, mod in m.named_modules() if mod is m.g.fno]
        assert len(fno_modules) == 1, "g.fno should appear once in the module tree"

    def test_forward_doc_cannot_encode_leaves_without_g_forward(self, monkeypatch) -> None:
        torch.manual_seed(0)
        m = _build_no_model()

        def fail_forward(
            _left_or_pair: torch.Tensor,
            _right_state: torch.Tensor | None = None,
        ) -> torch.Tensor:
            raise RuntimeError("leaf path must call g.forward")

        monkeypatch.setattr(m.g, "forward", fail_forward)
        toks = torch.randint(0, 16, (1, 16))
        with pytest.raises(RuntimeError, match="g.forward"):
            m(toks)

    def test_leaf_state_changes_when_shared_g_leaf_call_changes(self, monkeypatch) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (1, 16))
        baseline = m(toks).root_state.detach().clone()
        original_forward = m.g.forward

        def shifted_forward(
            left_or_pair: torch.Tensor,
            right_state: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = original_forward(left_or_pair, right_state)
            if right_state is None and int(left_or_pair.shape[-2]) == m.channels:
                return out + 0.125
            return out

        monkeypatch.setattr(m.g, "forward", shifted_forward)
        shifted = m(toks).root_state.detach()
        assert not torch.allclose(shifted, baseline)

    def test_merge_state_changes_when_same_shared_g_merge_call_changes(self, monkeypatch) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (2, 16))
        baseline = m(toks).root_state.detach().clone()
        original_forward = m.g.forward

        def shifted_forward(
            left_or_pair: torch.Tensor,
            right_state: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = original_forward(left_or_pair, right_state)
            if right_state is not None:
                return out - 0.125
            return out

        monkeypatch.setattr(m.g, "forward", shifted_forward)
        shifted = m(toks).root_state.detach()
        assert not torch.allclose(shifted, baseline)

    def test_n_merges_equals_n_leaves_minus_one(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        for n_leaves in [1, 2, 3, 4, 5, 8]:
            toks = torch.randint(0, 16, (n_leaves, 16))
            out = m(toks)
            assert len(out.merge_states) == n_leaves - 1
            assert len(out.leaf_states) == n_leaves
            assert out.leaf_states[0].shape == (m.channels, 16)
            assert out.root_state.shape == (m.channels, 16)

    def test_manual_replay_of_leaf_path_matches(self) -> None:
        # Manually do (embed -> g.encode_leaf) and verify it matches the leaf
        # state that forward_doc produced.
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        embedded = m.token_embedding(toks)
        manual_leaves = m.g(embedded)
        for i in range(4):
            assert torch.allclose(manual_leaves[i], out.leaf_states[i])

    def test_manual_replay_of_merge_path_matches(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (2, 16))
        out = m(toks)
        # 2-leaf doc has exactly 1 merge (the root)
        manual_root = m.g(
            out.leaf_states[0].unsqueeze(0),
            out.leaf_states[1].unsqueeze(0),
        ).squeeze(0)
        assert torch.allclose(manual_root, out.root_state)

    def test_two_leaf_formula_replay_matches_f_of_g_gx_gy(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (2, 16))
        out = m(toks)
        embedded = m.token_embedding(toks)
        zx, zy = m.g(embedded)
        zxy = m.g(zx.unsqueeze(0), zy.unsqueeze(0))
        score = m.f(zxy).squeeze(0)
        assert torch.allclose(zxy.squeeze(0), out.root_state)
        assert torch.allclose(score, out.root_count_norm)

    def test_multi_leaf_balanced_replay_matches_all_merge_states(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (5, 16))
        out = m(toks)
        cur = list(out.leaf_states)
        replayed_merges = []
        while len(cur) > 1:
            nxt = []
            pair_count = len(cur) // 2
            if pair_count:
                left = torch.stack(cur[: 2 * pair_count : 2], dim=0)
                right = torch.stack(cur[1 : 2 * pair_count : 2], dim=0)
                merged = m.g(left, right)
                replayed_merges.extend([merged[i] for i in range(int(merged.shape[0]))])
                nxt.extend([merged[i] for i in range(int(merged.shape[0]))])
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt

        assert len(replayed_merges) == len(out.merge_states)
        for expected, actual in zip(replayed_merges, out.merge_states):
            assert torch.allclose(expected, actual)
        assert torch.allclose(cur[0], out.root_state)

    def test_reported_node_scores_are_from_f(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        leaf_scores = m.f(torch.stack(out.leaf_states, dim=0))
        merge_scores = m.f(torch.stack(out.merge_states, dim=0))
        assert torch.allclose(leaf_scores, out.leaf_counts_norm)
        assert torch.allclose(merge_scores, out.merge_counts_norm)
        assert torch.allclose(m.f(out.root_state.unsqueeze(0)).squeeze(0), out.root_count_norm)

    def test_backprop_reaches_all_parameters(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        loss = out.root_count_norm ** 2 + out.leaf_counts_norm.sum() + out.merge_counts_norm.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"no grad for {name}"

    def test_loss_helpers_work_on_no_output(self) -> None:
        torch.manual_seed(0)
        m = _build_no_model()
        toks = torch.randint(0, 16, (4, 16))
        out = m(toks)
        true_root = torch.tensor(3.0)
        true_leaves = torch.zeros(4)
        true_merges = torch.zeros(3)
        loss = (
            root_mse_loss(out, root_count=true_root, target_scale=m.target_scale)
            + leaf_mse_loss(out, leaf_counts=true_leaves, target_scale=m.target_scale)
            + merge_mse_loss(out, merge_counts=true_merges, target_scale=m.target_scale)
        )
        assert loss.requires_grad
        assert torch.isfinite(loss)


class TestCleanUnifiedNOMarkovLawHelpers:
    def test_node_witness_targets_match_balanced_forward_order(self) -> None:
        block_by_token = [0, 0, 1, 1]
        target_scale = 64.0
        leaves = [
            [0, 0, 2, 2],
            [2, 0, 0, 0],
            [1, 1, 3, 3],
        ]
        exact = ExactZeroMarkovFG(
            block_by_token=block_by_token,
            target_scale=target_scale,
            vocab_size=4,
            n_regimes=2,
            learnable_join=False,
        )
        out = exact(torch.tensor(leaves, dtype=torch.long))
        targets = _markov_node_witness_targets_for_leaves(
            leaves,
            block_by_token=block_by_token,
            target_scale=target_scale,
            device=torch.device("cpu"),
        )

        leaf_target = targets["leaf"]
        merge_target = targets["merge"]
        leaf_sketch = torch.cat(
            [
                leaf_target["count_norm"].unsqueeze(-1),
                torch.nn.functional.one_hot(leaf_target["first"], num_classes=2).float(),
                torch.nn.functional.one_hot(leaf_target["last"], num_classes=2).float(),
            ],
            dim=-1,
        )
        merge_sketch = torch.cat(
            [
                merge_target["count_norm"].unsqueeze(-1),
                torch.nn.functional.one_hot(merge_target["first"], num_classes=2).float(),
                torch.nn.functional.one_hot(merge_target["last"], num_classes=2).float(),
            ],
            dim=-1,
        )

        assert torch.allclose(leaf_sketch, torch.stack(out.leaf_states))
        assert torch.allclose(merge_sketch, torch.stack(out.merge_states))

    def test_balanced_merge_state_triples_match_forward_doc_order(self) -> None:
        leaf_states = [
            torch.full((2, 1), float(i), dtype=torch.float32)
            for i in range(5)
        ]
        merge_states = [
            torch.full((2, 1), 10.0, dtype=torch.float32),
            torch.full((2, 1), 11.0, dtype=torch.float32),
            torch.full((2, 1), 12.0, dtype=torch.float32),
            torch.full((2, 1), 13.0, dtype=torch.float32),
        ]
        output = TreeForwardOutputNO(
            leaf_states=leaf_states,
            merge_states=merge_states,
            leaf_counts_norm=torch.zeros(5),
            merge_counts_norm=torch.zeros(4),
            root_state=merge_states[-1],
            root_count_norm=torch.tensor(0.0),
        )
        triples = _balanced_merge_state_triples(output)

        assert len(triples) == 4
        assert triples[0][0] is leaf_states[0]
        assert triples[0][1] is leaf_states[1]
        assert triples[0][2] is merge_states[0]
        assert triples[1][0] is leaf_states[2]
        assert triples[1][1] is leaf_states[3]
        assert triples[1][2] is merge_states[1]
        assert triples[2][0] is merge_states[0]
        assert triples[2][1] is merge_states[1]
        assert triples[2][2] is merge_states[2]
        assert triples[3][0] is merge_states[2]
        assert triples[3][1] is leaf_states[4]
        assert triples[3][2] is merge_states[3]

    def test_multi_leaf_decoded_diagnostics_expose_leaf_merge_root_metrics(self) -> None:
        block_by_token = [0, 0, 1, 1]
        target_scale = 64.0
        leaves = [
            [0, 0, 2, 2],
            [2, 0, 0, 0],
            [1, 1, 3, 3],
        ]
        exact = ExactZeroMarkovFG(
            block_by_token=block_by_token,
            target_scale=target_scale,
            vocab_size=4,
            n_regimes=2,
            learnable_join=False,
        )

        class FakeDoc:
            leaf_token_ids = leaves
            leaf_counts = [0.0, 0.0, 0.0]
            merge_counts_balanced = [0.0, 0.0]
            root_count = 0.0

        class FakeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target_scale = target_scale

            def forward(self, leaf_tokens: torch.Tensor) -> TreeForwardOutputNO:
                exact_out = exact(leaf_tokens)

                def lift(state: torch.Tensor) -> torch.Tensor:
                    return state.unsqueeze(-1)

                return TreeForwardOutputNO(
                    leaf_states=[lift(state) for state in exact_out.leaf_states],
                    merge_states=[lift(state) for state in exact_out.merge_states],
                    leaf_counts_norm=exact_out.leaf_counts_norm,
                    merge_counts_norm=exact_out.merge_counts_norm,
                    root_state=lift(exact_out.root_state),
                    root_count_norm=exact_out.root_count_norm,
                )

        class FakeHead(nn.Module):
            def forward(
                self,
                state: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                count = state[:, 0, 0]
                first_logits = state[:, 1:3, 0] * 20.0
                last_logits = state[:, 3:5, 0] * 20.0
                return count, first_logits, last_logits

        metrics = _eval_markov_node_witness_head(
            model=FakeModel(),
            witness_head=FakeHead(),
            docs=[FakeDoc()],
            block_by_token=block_by_token,
            target_scale=target_scale,
            batch_size=1,
            device=torch.device("cpu"),
        )

        assert metrics["status"] == "ok"
        for key in ["leaf", "merge", "root"]:
            assert dict(metrics[key])["status"] == "ok"
            assert dict(metrics[key])["theta_first_regime_accuracy"] == pytest.approx(1.0)
            assert dict(metrics[key])["theta_last_regime_accuracy"] == pytest.approx(1.0)
            assert dict(metrics[key])["count_diagnostics"]["root_mae"] == pytest.approx(0.0)
