"""Tests for the optional JAX/sbijax contextual-sufficiency lane."""

from __future__ import annotations

import tomllib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.probe_clean_unified_no import _contextual_sufficiency_batch_losses
from src.ctreepo.sim.core.clean_unified_fg import CleanUnifiedNO
from src.ctreepo.sim.core.fno_doc_baselines import HAS_NEURAL_OPERATOR
from src.ctreepo.sim.core.contextual_sbijax import (
    ContextualSBIJAXConfig,
    HLL_REGISTER_SKETCH_LAW_SET_ID,
    HLLUnionContext,
    HLLUnionContextProblem,
    build_contextual_query_dataset,
    build_contextual_response_dataset,
    contextual_sbijax_available,
    contextual_sbijax_provenance,
    exact_count_for_tokens,
    exact_root_witness_diagnostics,
    fit_contextual_sbijax,
    fit_contextual_sbijax_exact_zero_markov,
    fit_contextual_sbijax_identity_theta,
    fit_contextual_sbijax_learned_local_laws,
    fit_contextual_sbijax_nass_nle,
    fit_contextual_sbijax_npe_direct,
    fit_contextual_sbijax_package_direct,
    fit_contextual_sbijax_posterior_direct,
    fit_contextual_sbijax_theta_supervised,
    hll_register_sketch_targets_for_dataset,
    hybrid_summary_diagnostics,
    load_markov_contextual_splits,
    load_markov_contextual_splits_from_bundle,
    markov_exact_sketch_oracle_diagnostics,
    markov_exact_sketch_targets_for_dataset,
    MarkovTwoSidedContextProblem,
    make_synthetic_markov_docs,
    _make_jax_fno_summary_net,
    _require_contextual_sbi,
    pad_fragment,
    palette_block_map,
    with_package_theta_target,
)
from src.ctreepo.sim.core.markov_hazard_panels import build_markov_hazard_panel_data_bundle
from src.ctreepo.sim.core.markov_local_laws import MARKOV_COUNT_SKETCH_LAW_SET_ID


def test_contextual_sbi_extra_pins_sbijax() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    deps = pyproject["project"]["optional-dependencies"]["contextual_sbi"]
    assert "sbijax==0.3.6" in deps
    assert "fastprogress>=1.0.0,<1.1.0" in deps
    assert "starlette>=0.40.0,<0.51.0" in deps


def test_contextual_response_dataset_shapes_and_context_reuse() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=4,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=1,
    )
    train = build_contextual_response_dataset(
        docs,
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=2,
        fragment_len=6,
        response_signature_contexts=3,
        seed=2,
    )
    val = build_contextual_response_dataset(
        docs,
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=3,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    assert train.item_tokens.shape == (8, 6)
    assert train.span_tokens.shape == (8, 6)
    assert train.response_signatures.shape == (8, 3)
    assert train.context_left_tokens.shape == (3, 6)
    assert train.metadata["problem_id"] == "markov_changepoint_count"
    assert train.metadata["context_kind"] == "markov_two_sided"
    assert train.metadata["block_by_token"] == block_by_token
    assert train.metadata["n_regimes"] == 2
    assert val.response_signatures.shape == (4, 3)
    assert val.context_left_raw == train.context_left_raw
    assert val.context_right_raw == train.context_right_raw


class _ToyOffsetContextProblem:
    problem_id = "toy_offset_parity"
    context_kind = "toy_offset"
    vocab_size = 10
    target_scale = 1.0

    def sample_item_tokens(
        self,
        source: Sequence[int],
        *,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[int]:
        del rng
        return list(source[: int(item_len)])

    def sample_contexts(
        self,
        sources: Sequence[Sequence[int]],
        *,
        n_contexts: int,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[int]:
        del sources, item_len, rng
        return tuple(range(int(n_contexts)))

    def evaluate_query(self, context: int, item_tokens: Sequence[int]) -> float:
        return float((sum(int(tok) for tok in item_tokens) + int(context)) % 2)

    def context_payload(self, context: int) -> Mapping[str, Any]:
        return {"kind": self.context_kind, "offset": int(context)}

    def context_tensors(
        self,
        contexts: Sequence[int],
        *,
        item_len: int,
        pad_id: int,
    ) -> Mapping[str, np.ndarray]:
        del item_len, pad_id
        return {"offset": np.asarray([[int(ctx)] for ctx in contexts], dtype=np.int32)}


class _ToyVectorContextProblem(_ToyOffsetContextProblem):
    problem_id = "toy_vector_response"
    context_kind = "toy_vector_offset"

    def evaluate_query(self, context: int, item_tokens: Sequence[int]) -> np.ndarray:
        parity = (sum(int(tok) for tok in item_tokens) + int(context)) % 2
        return np.asarray([float(context), float(parity)], dtype=np.float32)


def test_generic_contextual_query_dataset_has_no_left_right_assumption() -> None:
    dataset = build_contextual_query_dataset(
        [[1, 2, 3], [4, 5, 6]],
        problem=_ToyOffsetContextProblem(),
        samples_per_source=1,
        item_len=2,
        n_contexts=3,
        seed=123,
    )
    assert dataset.item_tokens.shape == (2, 2)
    assert dataset.response_signatures.shape == (2, 3)
    assert dataset.context_payloads == (
        {"kind": "toy_offset", "offset": 0},
        {"kind": "toy_offset", "offset": 1},
        {"kind": "toy_offset", "offset": 2},
    )
    assert dataset.context_tensors["offset"].shape == (3, 1)
    assert dataset.context_left_tokens.shape == (0, 0)
    assert dataset.metadata["problem_id"] == "toy_offset_parity"
    assert dataset.metadata["context_kind"] == "toy_offset"
    assert dataset.metadata["response_target_shape"] == []
    assert dataset.metadata["response_signature_dim"] == 3


def test_generic_contextual_query_dataset_supports_vector_responses() -> None:
    dataset = build_contextual_query_dataset(
        [[1, 2, 3], [4, 5, 6]],
        problem=_ToyVectorContextProblem(),
        samples_per_source=1,
        item_len=2,
        n_contexts=2,
        seed=123,
    )
    assert dataset.item_tokens.shape == (2, 2)
    assert dataset.response_signatures.shape == (2, 2, 2)
    assert dataset.context_left_tokens.shape == (0, 0)
    assert dataset.metadata["problem_id"] == "toy_vector_response"
    assert dataset.metadata["context_kind"] == "toy_vector_offset"
    assert dataset.metadata["response_target_shape"] == [2]
    assert dataset.metadata["response_signature_dim"] == 4


def test_hll_union_context_problem_targets_and_responses() -> None:
    from src.tree.hll import HLLConfig, HyperLogLogSketch

    problem = HLLUnionContextProblem(
        vocab_size=16,
        target_scale=32.0,
        precision=4,
        hash_bits=64,
    )
    contexts = (
        HLLUnionContext(tokens=(1, 2, 3, 3)),
        HLLUnionContext(tokens=(8, 9)),
    )
    dataset = build_contextual_query_dataset(
        [[1, 4, 5, 6], [7, 8, 9, 10]],
        problem=problem,
        samples_per_source=1,
        item_len=4,
        n_contexts=2,
        seed=123,
        contexts=contexts,
    )
    targets = hll_register_sketch_targets_for_dataset(dataset, precision=4)
    expected = (
        HyperLogLogSketch.from_tokens(HLLConfig(precision=4), [1, 2, 3, 3, 1, 4, 5, 6]).estimate()
        / 32.0
    )
    assert dataset.metadata["problem_id"] == "hll_cardinality"
    assert dataset.metadata["context_kind"] == "hll_union"
    assert dataset.metadata["hll_register_count"] == 16
    assert dataset.response_signatures.shape == (2, 2)
    assert dataset.response_signatures[0, 0] == pytest.approx(expected)
    assert targets.shape == (2, 16)
    assert np.all(targets >= 0.0)
    assert np.all(targets <= 1.0)


def test_markov_two_sided_problem_reproduces_response_signature_values() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    problem = MarkovTwoSidedContextProblem(
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
    )
    contexts = problem.sample_contexts(
        [[0, 1, 4, 5], [2, 3, 6, 7]],
        n_contexts=2,
        item_len=3,
        rng=np.random.default_rng(1),
    )
    dataset = build_contextual_query_dataset(
        [[1, 4, 5, 6]],
        problem=problem,
        samples_per_source=1,
        item_len=3,
        n_contexts=2,
        seed=2,
        contexts=contexts,
    )
    assert dataset.item_tokens.shape == (1, 3)
    assert dataset.context_left_tokens.shape == (2, 3)
    item = [int(tok) for tok in dataset.item_tokens[0] if int(tok) != dataset.pad_id]
    expected = [
        exact_count_for_tokens(
            list(ctx.left_tokens) + item + list(ctx.right_tokens),
            block_by_token=block_by_token,
        )
        / 32.0
        for ctx in contexts
    ]
    assert np.allclose(dataset.response_signatures, np.asarray([expected], dtype=np.float32))
    assert np.array_equal(
        dataset.context_left_tokens[0],
        np.asarray(
            pad_fragment(contexts[0].left_tokens, fragment_len=3, pad_id=8),
            dtype=np.int32,
        ),
    )


def test_markov_exact_sketch_targets_encode_count_first_last() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    problem = MarkovTwoSidedContextProblem(
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
    )
    dataset = build_contextual_query_dataset(
        [[1, 4, 5, 6]],
        problem=problem,
        samples_per_source=1,
        item_len=4,
        n_contexts=2,
        seed=2,
    )
    targets = markov_exact_sketch_targets_for_dataset(
        dataset,
        block_by_token=block_by_token,
        target_scale=32.0,
        n_regimes=2,
    )
    assert targets.shape == (1, 5)
    assert np.isclose(targets[0, 0], 1.0 / 32.0)
    assert np.array_equal(targets[0, 1:3], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.array_equal(targets[0, 3:5], np.asarray([0.0, 1.0], dtype=np.float32))


def test_markov_exact_sketch_oracle_reconstructs_contextual_responses() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    problem = MarkovTwoSidedContextProblem(
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
    )
    dataset = build_contextual_query_dataset(
        [[1, 4, 5, 6], [6, 5, 2, 1]],
        problem=problem,
        samples_per_source=1,
        item_len=4,
        n_contexts=3,
        seed=2,
    )
    diagnostics = markov_exact_sketch_oracle_diagnostics(
        dataset,
        block_by_token=block_by_token,
        target_scale=32.0,
        n_regimes=2,
    )
    assert diagnostics["state_dim"] == 5
    assert diagnostics["contextual_mae"] <= 1e-7
    assert diagnostics["contextual_mse"] <= 1e-12


def test_hybrid_summary_diagnostics_reports_component_and_product_collisions() -> None:
    response_signatures = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    base_states = np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32)
    neural_states = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float32)

    diagnostics = hybrid_summary_diagnostics(
        base_states=base_states,
        neural_states=neural_states,
        response_signatures=response_signatures,
        state_eps=1e-8,
        response_eps=1e-8,
    )

    assert diagnostics["diagnostic"] == "hybrid_summary_finite_response_collision"
    assert diagnostics["n"] == 3
    assert diagnostics["base_bad_collision_pair_count"] == 1
    assert diagnostics["neural_bad_collision_pair_count"] == 1
    assert diagnostics["hybrid_bad_collision_pair_count"] == 0
    assert diagnostics["hybrid_state_dim"] == 2


def test_markov_contextual_splits_build_and_reuse_context_bank() -> None:
    splits = load_markov_contextual_splits(
        benchmark="recoverable_v5_t2048",
        doc_tokens=24,
        train_docs=4,
        val_docs=2,
        test_docs=2,
        leaf_tokens=24,
        expected_boundaries=2.0,
        seed=7,
    )
    assert splits.metadata["data_source"] == "markov"
    assert splits.metadata["markov_loader"] == "direct_ops_count"
    assert splits.metadata["train_docs"] == 4
    assert splits.metadata["val_docs"] == 2
    assert splits.metadata["test_docs"] == 2
    assert len(splits.train_docs[0]) == 24
    assert len(splits.test_root_counts) == 2

    train = build_contextual_response_dataset(
        splits.train_docs,
        block_by_token=splits.block_by_token,
        vocab_size=16,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=8,
    )
    val = build_contextual_response_dataset(
        splits.val_docs,
        block_by_token=splits.block_by_token,
        vocab_size=16,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=9,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    witness = exact_root_witness_diagnostics(
        splits.test_docs,
        block_by_token=splits.block_by_token,
        root_counts=splits.test_root_counts,
    )
    assert train.span_tokens.shape == (4, 6)
    assert val.response_signatures.shape == (2, 3)
    assert val.context_left_raw == train.context_left_raw
    assert val.context_right_raw == train.context_right_raw
    assert witness["root_mae"] == 0.0
    assert witness["max_abs_error"] == 0.0


def test_markov_contextual_splits_load_saved_hazard_panel_bundle(tmp_path: Path) -> None:
    bundle = build_markov_hazard_panel_data_bundle(
        "paper_hazard_panel_v1_t128",
        train_docs=8,
        val_docs=4,
        test_docs=4,
        seed=11,
    )
    bundle_path = tmp_path / "panel_bundle.json"
    bundle.save(bundle_path)

    splits = load_markov_contextual_splits_from_bundle(
        bundle_path,
        train_docs=4,
        val_docs=2,
        test_docs=2,
    )
    assert splits.metadata["markov_loader"] == "saved_ops_count_bundle"
    assert splits.metadata["hazard_panel_id"] == "paper_hazard_panel_v1_t128"
    assert splits.metadata["train_docs"] == 4
    assert splits.metadata["vocab_size"] == 48
    assert splits.metadata["n_regimes"] == 12
    assert len(splits.metadata["condition_ids"]["train"]) == 4
    assert set(splits.metadata["condition_counts"]["train"].values()) == {1}
    assert len(splits.train_docs) == 4
    assert len(splits.train_docs[0]) == 128

    witness = exact_root_witness_diagnostics(
        splits.test_docs,
        block_by_token=splits.block_by_token,
        root_counts=splits.test_root_counts,
    )
    assert witness["root_mae"] == 0.0
    assert witness["max_abs_error"] == 0.0

    train = build_contextual_response_dataset(
        splits.train_docs,
        block_by_token=splits.block_by_token,
        vocab_size=int(splits.metadata["vocab_size"]),
        target_scale=24.0,
        samples_per_doc=1,
        fragment_len=8,
        response_signature_contexts=3,
        seed=12,
    )
    diagnostics = markov_exact_sketch_oracle_diagnostics(
        train,
        block_by_token=splits.block_by_token,
        target_scale=24.0,
        n_regimes=int(splits.metadata["n_regimes"]),
    )
    assert diagnostics["contextual_mae"] <= 1e-7
    assert diagnostics["state_dim"] == 25


def test_contextual_sbijax_provenance_surface_is_stable() -> None:
    provenance = contextual_sbijax_provenance(
        method="nasss",
        response_signature_contexts=4,
        response_signature_slices=2,
    )
    assert provenance["backend_package"] == "sbijax"
    assert provenance["method"] == "nasss"
    assert provenance["trainer"] == "repo"
    assert provenance["response_signature_contexts"] == 4
    assert provenance["response_signature_slices"] == 2
    assert "installed" in provenance


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="requires neuraloperator")
def test_pytorch_contextual_loss_supports_signature_only_generic_problem() -> None:
    torch.manual_seed(0)
    model = CleanUnifiedNO(
        vocab_size=10,
        target_scale=1.0,
        channels=4,
        g_n_modes=2,
        g_n_layers=1,
        scorer_n_modes=2,
        scorer_n_layers=1,
    )
    response_regressor = nn.Linear(4, 2)
    contextual_loss, dependence_loss, n_queries = _contextual_sufficiency_batch_losses(
        model=model,
        flat_train_docs=[[1, 2, 3, 4], [4, 5, 6, 7]],
        batch_indices=[0, 1],
        block_by_token=[0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        target_scale=1.0,
        samples_per_doc=1,
        fragment_len=4,
        rng=np.random.default_rng(1),
        device=torch.device("cpu"),
        response_regressor=response_regressor,
        response_signature_contexts=3,
        response_signature_slices=2,
        dependence_objective="regression",
        contextual_problem=_ToyOffsetContextProblem(),
    )
    assert n_queries == 0
    assert torch.isfinite(contextual_loss)
    assert torch.isfinite(dependence_loss)
    assert dependence_loss.item() >= 0.0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="requires neuraloperator")
def test_pytorch_contextual_loss_uses_markov_enacted_context_executor() -> None:
    torch.manual_seed(0)
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    model = CleanUnifiedNO(
        vocab_size=8,
        target_scale=32.0,
        channels=4,
        g_n_modes=2,
        g_n_layers=1,
        scorer_n_modes=2,
        scorer_n_layers=1,
    )
    contextual_loss, dependence_loss, n_queries = _contextual_sufficiency_batch_losses(
        model=model,
        flat_train_docs=[[1, 2, 4, 5], [2, 3, 6, 7]],
        batch_indices=[0, 1],
        block_by_token=block_by_token,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=4,
        rng=np.random.default_rng(2),
        device=torch.device("cpu"),
        response_signature_contexts=2,
        dependence_objective="none",
        contextual_problem=MarkovTwoSidedContextProblem(
            block_by_token=block_by_token,
            vocab_size=8,
            target_scale=32.0,
        ),
    )
    assert n_queries == 4
    assert torch.isfinite(contextual_loss)
    assert torch.isfinite(dependence_loss)
    assert dependence_loss.item() == 0.0


def _tiny_markov_exact_sbijax_datasets(seed: int = 60):
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=int(seed),
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=int(seed) + 1,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=int(seed) + 2,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    return train, val


@pytest.mark.skipif(
    not contextual_sbijax_available(),
    reason='requires optional dependency: pip install -e ".[contextual_sbi]"',
)
def test_tiny_contextual_sbijax_fit_runs() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=10,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=11,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=12,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    result = fit_contextual_sbijax(
        train,
        val,
        config=ContextualSBIJAXConfig(
            method="nasss",
            vocab_size=8,
            embedding_dim=8,
            state_dim=4,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            n_iter=2,
            batch_size=4,
            seed=13,
        ),
    )
    assert len(result.history) == 2
    assert result.provenance["backend_package"] == "sbijax"
    assert result.val_diagnostics["n"] == 2
    assert "contextual_mae" in result.val_diagnostics


@pytest.mark.skipif(
    not contextual_sbijax_available(),
    reason='requires optional dependency: pip install -e ".[contextual_sbi]"',
)
@pytest.mark.parametrize("method", ["nasss", "nass"])
def test_tiny_contextual_sbijax_package_direct_runs(method: str) -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=20,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=21,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=22,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    result = fit_contextual_sbijax_package_direct(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="package",
            method=method,
            summary_activation="tanh",
            vocab_size=8,
            embedding_dim=8,
            state_dim=4,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            n_iter=2,
            batch_size=4,
            seed=23,
        ),
    )
    assert len(result.history) == 2
    assert result.provenance["backend_package"] == "sbijax"
    assert result.provenance["trainer"] == "package"
    assert result.provenance["sbijax_class"] == ("NASSS" if method == "nasss" else "NASS")
    assert result.provenance["input_encoding"] == "normalized_token_ids"
    assert result.provenance["summary_activation"] == "tanh"
    assert result.provenance["downstream_readout"] == "haiku_mlp_mse"
    assert result.train_diagnostics["state_dim"] == 4
    assert result.val_diagnostics["n"] == 2
    assert "contextual_mae" in result.val_diagnostics
    assert "pred_truth_corr" in result.val_diagnostics
    assert "collision_rate" in result.val_diagnostics
    assert "pred_std" in result.val_diagnostics
    for row in result.history:
        assert np.isfinite(row["train_package_loss"])
        assert np.isfinite(row["train_readout_mse"])
        assert np.isfinite(row["val_readout_mse"])


@pytest.mark.skipif(
    not contextual_sbijax_available(),
    reason='requires optional dependency: pip install -e ".[contextual_sbi]"',
)
def test_tiny_contextual_sbijax_package_direct_markov_exact_theta_runs() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=30,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=31,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=32,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    result = fit_contextual_sbijax_package_direct(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="package",
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="regime_one_hot",
            vocab_size=8,
            embedding_dim=8,
            state_dim=4,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            n_iter=2,
            batch_size=4,
            seed=33,
        ),
    )
    assert len(result.history) == 2
    assert result.provenance["package_theta"] == "markov_exact_sketch"
    assert result.provenance["package_theta_dim"] == 5
    assert result.provenance["input_encoding"] == "regime_one_hot"
    assert result.val_diagnostics["n"] == 2
    assert "contextual_mae" in result.val_diagnostics


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
@pytest.mark.parametrize("trainer", ["npe", "nass_nle"])
def test_tiny_contextual_sbijax_package_inference_trainers_run(trainer: str) -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=37,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=38,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=39,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    config = ContextualSBIJAXConfig(
        trainer=trainer,
        method="nass",
        package_theta="markov_exact_sketch",
        input_encoding="markov_exact_sketch",
        vocab_size=8,
        embedding_dim=8,
        state_dim=5,
        hidden_dim=8,
        response_signature_contexts=3,
        response_signature_slices=2,
        n_iter=2,
        batch_size=4,
        posterior_samples=2,
        density_components=2,
        seed=40,
    )
    result = (
        fit_contextual_sbijax_npe_direct(train, val, config=config)
        if trainer == "npe"
        else fit_contextual_sbijax_nass_nle(train, val, config=config)
    )
    assert result.provenance["trainer"] == trainer
    assert result.provenance["density_estimator"] == "sbijax.nn.make_mdn"
    assert len(result.history) >= 1
    assert result.val_diagnostics["n"] == 2
    assert np.isfinite(result.val_diagnostics["contextual_mae"])
    if trainer == "npe":
        assert result.provenance["sbijax_class"] == "NPE"
        assert "val_npe_loss" in result.history[-1]
    else:
        assert result.provenance["likelihood_estimator"] == "sbijax.NLE"
        assert "val_nle_loss" in result.history[-1]


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
@pytest.mark.parametrize(
    ("posterior_estimator", "density_family"),
    [("npe", "mdn"), ("fmpe", "cnf"), ("cmpe", "cm")],
)
def test_tiny_contextual_sbijax_posterior_exact_markov_runs(
    posterior_estimator: str,
    density_family: str,
) -> None:
    train, val = _tiny_markov_exact_sbijax_datasets(seed=61)
    result = fit_contextual_sbijax_posterior_direct(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="posterior",
            posterior_estimator=posterior_estimator,
            density_family=density_family,
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="markov_exact_sketch",
            vocab_size=8,
            embedding_dim=8,
            state_dim=5,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            n_iter=2,
            batch_size=4,
            posterior_samples=2,
            posterior_eval_samples=2,
            posterior_eval_batch_size=2,
            density_components=2,
            seed=62,
        ),
    )
    expected_class = {"npe": "NPE", "fmpe": "FMPE", "cmpe": "CMPE"}[posterior_estimator]
    assert result.provenance["trainer"] == "posterior"
    assert result.provenance["sbijax_class"] == expected_class
    assert result.provenance["posterior_estimator"] == posterior_estimator
    assert result.provenance["density_family"] == density_family
    assert result.provenance["downstream_readout"] == "deterministic_markov_exact_sketch"
    assert len(result.history) >= 1
    assert f"val_{posterior_estimator}_loss" in result.history[-1]
    assert np.isfinite(result.history[-1]["val_posterior_loss"])
    assert np.isfinite(result.val_diagnostics["theta_mse"])
    assert np.isfinite(result.val_diagnostics["theta_count_raw_mae"])
    assert np.isfinite(result.val_diagnostics["contextual_raw_mae"])


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_contextual_sbijax_posterior_rejects_unsupported_pair() -> None:
    train, val = _tiny_markov_exact_sbijax_datasets(seed=71)
    with pytest.raises(ValueError, match="unsupported posterior estimator/density pair"):
        fit_contextual_sbijax_posterior_direct(
            train,
            val,
            config=ContextualSBIJAXConfig(
                trainer="posterior",
                posterior_estimator="fmpe",
                density_family="mdn",
                method="nass",
                package_theta="markov_exact_sketch",
                input_encoding="markov_exact_sketch",
                vocab_size=8,
                response_signature_contexts=3,
                response_signature_slices=2,
                n_iter=1,
                batch_size=4,
                posterior_samples=2,
                seed=72,
            ),
        )


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_theta_supervised_runs() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=41,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=42,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=43,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    result = fit_contextual_sbijax_theta_supervised(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="theta_supervised",
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="regime_one_hot",
            vocab_size=8,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            n_iter=2,
            batch_size=4,
            seed=44,
        ),
    )
    assert len(result.history) == 2
    assert result.provenance["trainer"] == "theta_supervised"
    assert result.provenance["package_theta_dim"] == 5
    assert result.val_diagnostics["n"] == 2


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_identity_theta_is_exact() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=51,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=52,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=53,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    result = fit_contextual_sbijax_identity_theta(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="identity_theta",
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="markov_exact_sketch",
            vocab_size=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            seed=54,
        ),
    )
    assert result.provenance["trainer"] == "identity_theta"
    assert result.provenance["package_theta_dim"] == 5
    assert result.history[0]["train_theta_mse"] == pytest.approx(0.0)
    assert result.history[0]["val_theta_mse"] == pytest.approx(0.0)
    assert result.train_diagnostics["contextual_mae"] < 1e-7
    assert result.val_diagnostics["contextual_mae"] < 1e-7


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_exact_zero_markov_is_exact() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=61,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=62,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=63,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    config = ContextualSBIJAXConfig(
        trainer="exact_zero_markov",
        method="nass",
        package_theta="markov_exact_sketch",
        input_encoding="regime_one_hot",
        vocab_size=8,
        response_signature_contexts=3,
        response_signature_slices=2,
        seed=64,
    )
    result = fit_contextual_sbijax_exact_zero_markov(train, val, config=config)
    assert result.provenance["trainer"] == "exact_zero_markov"
    assert result.provenance["summary_network"] == "deterministic_structural_markov_sketch"
    assert result.provenance["decoder_kind"] == "exact"
    assert result.provenance["exact_zero_claim"] is True
    assert result.provenance["effective_input_encoding"] == "markov_exact_sketch"
    assert result.history[0]["train_theta_mse"] == pytest.approx(0.0)
    assert result.history[0]["val_theta_mse"] == pytest.approx(0.0)
    assert result.train_diagnostics["contextual_mae"] < 1e-7
    assert result.val_diagnostics["contextual_mae"] < 1e-7
    assert result.train_diagnostics["theta_mae"] < 1e-7
    assert result.val_diagnostics["theta_mae"] < 1e-7
    assert result.val_diagnostics["theta_count_raw_mae"] < 1e-7
    assert result.val_diagnostics["theta_first_regime_accuracy"] == pytest.approx(1.0)
    assert result.val_diagnostics["theta_last_regime_accuracy"] == pytest.approx(1.0)

    dispatched = fit_contextual_sbijax(train, val, config=config)
    assert dispatched.provenance["trainer"] == "exact_zero_markov"
    assert dispatched.val_diagnostics["contextual_mae"] < 1e-7


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_learned_local_laws_dense_exact_is_exact() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=71,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=72,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=73,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    config = ContextualSBIJAXConfig(
        trainer="learned_local_laws",
        method="nass",
        package_theta="markov_exact_sketch",
        input_encoding="markov_exact_sketch",
        vocab_size=8,
        response_signature_contexts=3,
        response_signature_slices=2,
        local_law_supervision_mode="dense_exact",
        n_iter=0,
        seed=74,
    )
    result = fit_contextual_sbijax_learned_local_laws(train, val, config=config)
    assert result.provenance["trainer"] == "learned_local_laws"
    assert result.provenance["decoder_kind"] == "exact"
    assert result.provenance["baseline_role"] == "local_law_learned"
    assert result.provenance["law_set_id"] == MARKOV_COUNT_SKETCH_LAW_SET_ID
    assert result.train_diagnostics["contextual_mae"] < 1e-7
    assert result.val_diagnostics["contextual_mae"] < 1e-7
    assert result.train_diagnostics["theta_mae"] < 1e-7
    assert result.val_diagnostics["theta_mae"] < 1e-7
    assert result.val_diagnostics["eps_leaf"] < 1e-7
    assert result.val_diagnostics["eps_merge"] < 1e-7
    assert result.val_diagnostics["eps_idemp"] < 1e-7

    dispatched = fit_contextual_sbijax(train, val, config=config)
    assert dispatched.provenance["trainer"] == "learned_local_laws"
    assert dispatched.val_diagnostics["contextual_mae"] < 1e-7


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_learned_local_laws_hll_runs() -> None:
    problem = HLLUnionContextProblem(
        vocab_size=16,
        target_scale=16.0,
        precision=4,
        hash_bits=64,
    )
    docs = [
        [0, 1, 2, 3, 4, 5],
        [2, 3, 5, 7, 11, 13],
        [1, 1, 2, 2, 3, 3],
        [8, 9, 10, 11, 12, 13],
        [0, 2, 4, 6, 8, 10],
        [5, 6, 7, 8, 9, 10],
    ]
    train = build_contextual_query_dataset(
        docs[:4],
        problem=problem,
        samples_per_source=1,
        item_len=5,
        n_contexts=2,
        seed=91,
    )
    contexts = tuple(
        HLLUnionContext(tokens=tuple(int(tok) for tok in payload["tokens"]))
        for payload in train.context_payloads
    )
    val = build_contextual_query_dataset(
        docs[4:],
        problem=problem,
        samples_per_source=1,
        item_len=5,
        n_contexts=2,
        seed=92,
        contexts=contexts,
    )
    train = with_package_theta_target(
        train,
        name="hll_register_sketch",
        targets=hll_register_sketch_targets_for_dataset(train, precision=4),
    )
    val = with_package_theta_target(
        val,
        name="hll_register_sketch",
        targets=hll_register_sketch_targets_for_dataset(val, precision=4),
    )
    result = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="learned_local_laws",
            method="nass",
            package_theta="hll_register_sketch",
            input_encoding="one_hot_token_ids",
            vocab_size=16,
            hidden_dim=8,
            response_signature_contexts=2,
            response_signature_slices=2,
            local_law_supervision_mode="dense_exact",
            local_law_hll_estimate_weight=0.25,
            local_law_explicit_state_decoder=True,
            local_law_summary_dim=12,
            law_architecture="learned_merge",
            n_iter=1,
            batch_size=4,
            seed=93,
        ),
    )
    assert result.provenance["trainer"] == "learned_local_laws"
    assert result.provenance["law_set_id"] == HLL_REGISTER_SKETCH_LAW_SET_ID
    assert result.provenance["package_theta"] == "hll_register_sketch"
    assert result.provenance["merge_network"] == "learned_asymmetric_mlp"
    assert result.provenance["paper_notation_factorization"] == (
        "explicit_g_summary_then_f_state_decoder"
    )
    assert result.provenance["local_law_explicit_state_decoder"] is True
    assert result.provenance["g_summary_dim"] == 12
    assert result.provenance["local_law_hll_estimate_weight"] == 0.25
    assert result.params["explicit_state_decoder"] is True
    assert result.params["summary_dim_effective"] == 12
    assert result.params["law_package"] == "hll_register_sketch"
    assert result.val_diagnostics["law_set_id"] == HLL_REGISTER_SKETCH_LAW_SET_ID
    assert "hll_register_mae" in result.val_diagnostics
    assert np.isfinite(result.history[0]["train_loss"])
    assert np.isfinite(result.history[0]["train_hll_estimate_mse"])
    assert np.isfinite(result.val_diagnostics["hll_estimate_raw_mae"])


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_learned_local_laws_package_aux_runs() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=6,
        doc_tokens=24,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=75,
    )
    train = build_contextual_response_dataset(
        docs[:4],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=76,
    )
    val = build_contextual_response_dataset(
        docs[4:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=3,
        seed=77,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    result = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="learned_local_laws",
            method="nasss",
            package_theta="markov_exact_sketch",
            input_encoding="regime_one_hot",
            vocab_size=8,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            local_law_supervision_mode="dense_exact",
            local_law_package_weight=0.5,
            n_iter=1,
            batch_size=4,
            seed=78,
        ),
    )
    assert result.provenance["trainer"] == "learned_local_laws"
    assert result.provenance["local_law_package_aux_active"] is True
    assert result.provenance["local_law_package_weight"] == pytest.approx(0.5)
    assert result.provenance["local_law_package_objective"] == "nasss"
    assert result.provenance["summary_network"] == "haiku_mlp_local_law_theta_plus_package_aux"
    assert np.isfinite(result.history[0]["train_package_loss"])
    assert np.isfinite(result.history[0]["val_package_loss"])


def _tiny_regime_one_hot_local_law_datasets(
    *,
    n_docs: int = 10,
    doc_tokens: int = 32,
    fragment_len: int = 8,
    seed: int = 90,
) -> tuple[Any, Any]:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=n_docs,
        doc_tokens=doc_tokens,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=3.0,
        seed=seed,
    )
    split = max(2, int(n_docs) // 2)
    train = build_contextual_response_dataset(
        docs[:split],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=fragment_len,
        response_signature_contexts=3,
        seed=seed + 1,
    )
    val = build_contextual_response_dataset(
        docs[split:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=fragment_len,
        response_signature_contexts=3,
        seed=seed + 2,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    return train, val


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_regime_transition_sum_summary_family_runs_and_reports_shape() -> None:
    train, val = _tiny_regime_one_hot_local_law_datasets()
    result = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="learned_local_laws",
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="regime_one_hot",
            local_law_summary_family="regime_transition_sum",
            vocab_size=8,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            local_law_supervision_mode="dense_exact",
            n_iter=0,
            batch_size=4,
            seed=94,
        ),
    )
    assert result.provenance["local_law_summary_family"] == "regime_transition_sum"
    assert result.provenance["summary_network"] == "haiku_regime_transition_sum_local_law_theta"
    assert result.params["summary_kind"] == "regime_transition_sum"
    assert np.isfinite(result.val_diagnostics["theta_count_raw_mae"])
    assert np.isfinite(result.val_diagnostics["theta_mae"])
    assert 0.0 <= result.val_diagnostics["theta_first_regime_accuracy"] <= 1.0
    assert 0.0 <= result.val_diagnostics["theta_last_regime_accuracy"] <= 1.0


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_regime_transition_sum_summary_family_rejects_non_regime_input() -> None:
    train, val = _tiny_regime_one_hot_local_law_datasets()
    with pytest.raises(ValueError, match="requires input_encoding='regime_one_hot'"):
        fit_contextual_sbijax_learned_local_laws(
            train,
            val,
            config=ContextualSBIJAXConfig(
                trainer="learned_local_laws",
                method="nass",
                package_theta="markov_exact_sketch",
                input_encoding="one_hot_token_ids",
                local_law_summary_family="regime_transition_sum",
                vocab_size=8,
                hidden_dim=8,
                response_signature_contexts=3,
                response_signature_slices=2,
                local_law_supervision_mode="dense_exact",
                n_iter=0,
                batch_size=4,
                seed=95,
            ),
        )


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_jax_fno_summary_family_runs_without_norax_dependency() -> None:
    train, val = _tiny_regime_one_hot_local_law_datasets()
    norax_loaded_before = "norax" in sys.modules
    result = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="learned_local_laws",
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="regime_one_hot",
            local_law_summary_family="jax_fno",
            local_law_summary_fno_n_modes=99,
            local_law_summary_fno_n_layers=1,
            local_law_summary_fno_pooling_mode="sum",
            vocab_size=8,
            hidden_dim=8,
            response_signature_contexts=3,
            response_signature_slices=2,
            local_law_supervision_mode="dense_exact",
            n_iter=0,
            batch_size=4,
            seed=194,
        ),
    )
    assert result.provenance["local_law_summary_family"] == "jax_fno"
    assert result.provenance["local_law_summary_family_canonical"] == "jax_fno"
    assert result.provenance["summary_network"] == "internal_jax_fno_local_law_theta"
    assert result.params["summary_kind"] == "jax_fno"
    assert result.provenance["local_law_summary_fno_effective_n_modes"] <= (
        int(train.item_tokens.shape[1]) // 2 + 1
    )
    assert np.isfinite(result.val_diagnostics["theta_mae"])
    if not norax_loaded_before:
        assert "norax" not in sys.modules


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_jax_fno_summary_net_jit_vmap_shape_dtype() -> None:
    deps = _require_contextual_sbi()
    jax = deps.jax
    jnp = deps.jnp
    net = _make_jax_fno_summary_net(
        deps,
        fragment_len=4,
        input_width=3,
        fno_width=5,
        n_modes=99,
        n_layers=1,
        pooling_mode="sum",
        output_dim=5,
    )
    features = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 12) / 10.0
    params = net.init(deps.jr.PRNGKey(197), features)
    jit_out = jax.jit(net.apply)(params, features)
    assert tuple(jit_out.shape) == (2, 5)
    assert jit_out.dtype == jnp.float32

    def apply_one(row):
        return net.apply(params, row[None, :])[0]

    vmap_out = jax.vmap(apply_one)(features)
    np.testing.assert_allclose(
        np.asarray(jit_out),
        np.asarray(vmap_out),
        rtol=2e-3,
        atol=5e-4,
    )


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_jax_fno_summary_family_deterministic_initialization() -> None:
    train, val = _tiny_regime_one_hot_local_law_datasets()
    config = ContextualSBIJAXConfig(
        trainer="learned_local_laws",
        method="nass",
        package_theta="markov_exact_sketch",
        input_encoding="regime_one_hot",
        local_law_summary_family="jax_fno",
        local_law_summary_fno_n_modes=2,
        local_law_summary_fno_n_layers=1,
        vocab_size=8,
        hidden_dim=8,
        response_signature_contexts=3,
        response_signature_slices=2,
        local_law_supervision_mode="dense_exact",
        n_iter=0,
        batch_size=4,
        seed=195,
    )
    first = fit_contextual_sbijax_learned_local_laws(train, val, config=config)
    second = fit_contextual_sbijax_learned_local_laws(train, val, config=config)
    assert second.val_diagnostics["theta_mae"] == pytest.approx(first.val_diagnostics["theta_mae"])
    assert second.val_diagnostics["theta_count_raw_mae"] == pytest.approx(
        first.val_diagnostics["theta_count_raw_mae"]
    )


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_jax_fno_summary_family_rejects_exact_sketch_input() -> None:
    train, val = _tiny_markov_exact_sbijax_datasets(seed=196)
    with pytest.raises(ValueError, match="jax_fno.*markov_exact_sketch"):
        fit_contextual_sbijax_learned_local_laws(
            train,
            val,
            config=ContextualSBIJAXConfig(
                trainer="learned_local_laws",
                method="nass",
                package_theta="markov_exact_sketch",
                input_encoding="markov_exact_sketch",
                local_law_summary_family="jax_fno",
                vocab_size=8,
                hidden_dim=8,
                response_signature_contexts=3,
                response_signature_slices=2,
                local_law_supervision_mode="dense_exact",
                n_iter=0,
                batch_size=4,
                seed=196,
            ),
        )


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_regime_transition_sum_improves_count_over_untrained() -> None:
    train, val = _tiny_regime_one_hot_local_law_datasets(n_docs=14, seed=96)
    base_config = dict(
        trainer="learned_local_laws",
        method="nass",
        package_theta="markov_exact_sketch",
        input_encoding="regime_one_hot",
        local_law_summary_family="regime_transition_sum",
        vocab_size=8,
        hidden_dim=16,
        response_signature_contexts=3,
        response_signature_slices=2,
        local_law_supervision_mode="dense_exact",
        batch_size=7,
        seed=99,
    )
    untrained = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(**base_config, n_iter=0),
    )
    trained = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(**base_config, n_iter=40, learning_rate=0.01),
    )
    assert (
        trained.val_diagnostics["theta_count_raw_mae"]
        < untrained.val_diagnostics["theta_count_raw_mae"]
    )
    assert trained.val_diagnostics["theta_first_regime_accuracy"] >= 0.75
    assert trained.val_diagnostics["theta_last_regime_accuracy"] >= 0.75


@pytest.mark.skipif(
    not contextual_sbijax_available(), reason="optional contextual_sbi deps missing"
)
def test_tiny_contextual_sbijax_learned_local_laws_sparse_ipw_metadata() -> None:
    block_by_token = palette_block_map(vocab_size=8, n_regimes=2)
    docs = make_synthetic_markov_docs(
        n_docs=4,
        doc_tokens=20,
        vocab_size=8,
        n_regimes=2,
        expected_boundaries=2.0,
        seed=81,
    )
    train = build_contextual_response_dataset(
        docs[:3],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=2,
        seed=82,
    )
    val = build_contextual_response_dataset(
        docs[3:],
        block_by_token=block_by_token,
        vocab_size=8,
        target_scale=32.0,
        samples_per_doc=1,
        fragment_len=6,
        response_signature_contexts=2,
        seed=83,
        context_left_tokens=train.context_left_raw,
        context_right_tokens=train.context_right_raw,
    )
    train = with_package_theta_target(
        train,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            train,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    val = with_package_theta_target(
        val,
        name="markov_exact_sketch",
        targets=markov_exact_sketch_targets_for_dataset(
            val,
            block_by_token=block_by_token,
            target_scale=32.0,
            n_regimes=2,
        ),
    )
    result = fit_contextual_sbijax_learned_local_laws(
        train,
        val,
        config=ContextualSBIJAXConfig(
            trainer="learned_local_laws",
            method="nass",
            package_theta="markov_exact_sketch",
            input_encoding="markov_exact_sketch",
            vocab_size=8,
            response_signature_contexts=2,
            response_signature_slices=2,
            local_law_supervision_mode="sparse_ipw",
            local_law_leaf_rate=0.5,
            local_law_merge_rate=1.0,
            local_law_idempotence_rate=0.5,
            n_iter=0,
            seed=84,
        ),
    )
    metadata = result.train_diagnostics["local_law_observation_metadata"]
    assert metadata["supervision_mode"] == "sparse_ipw"
    assert metadata["row_counts"]["leaf_preservation"] == train.item_tokens.shape[0]
    assert metadata["row_counts"]["merge_preservation"] >= 0
    assert metadata["propensity_means"]["merge_preservation"] == pytest.approx(1.0)
    assert np.isfinite(result.history[0]["train_loss"])
