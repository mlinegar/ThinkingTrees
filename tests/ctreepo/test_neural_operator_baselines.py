"""
Tests for the neural operator comparison baselines.

Covers:
- Smoke tests: each model can forward-pass a tiny batch
- Overfit test: MLP bigram can overfit 10 examples
- Integration: _fit_*_baseline returns valid SketchMetrics
- Guard: FNO raises ImportError if neuraloperator not installed
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import inspect
import json
from typing import Mapping, Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.ctreepo.sim.core.fno_doc_baselines as fdb
import src.ctreepo.sim.core.markov_neural_operator_baselines as nob
from src.core.autotune_probe_cache import AUTOTUNE_PROBE_CACHE_VERSION, ProbeCacheStore
from src.core.unified_runtime import (
    GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
    GpuRuntimeConfig,
    GpuRuntimeTelemetry,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    ChangepointMarkovDoc,
    OPSCountConfig,
    run_markov_changepoint_ops_count_experiment,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (
    CNN1DCountPredictor,
    DecodedMarkovSketch,
    DeepONetCountPredictor,
    FNOCountSketch,
    HAS_NEURAL_OPERATOR,
    MLPBigramCountPredictor,
    _deterministic_sample_indices,
    PrototypeClassifier,
    _bigram_features_from_tokens,
    _fit_cnn1d_baseline,
    _fit_deeponet_baseline,
    _fit_fno_baseline_with_predictions,
    _fit_mlp_bigram_baseline,
    _fno_summary_replay_tensors,
    _fno_single_lambda_objective_loss,
    _prepare_fno_count_docs,
    _theorem_count_threshold_pos_weights_from_docs,
    train_fno_tree,
    train_fno_tree_local_law,
)
from src.ctreepo.sim.core.theorem_feature_route import (
    build_theorem_feature_pair_sets,
    load_theorem_feature_stage1_artifact,
    register_theorem_feature_adapter,
    resolve_theorem_feature_adapter,
    theorem_feature_pair_metrics_from_scores,
    write_theorem_feature_stage1_artifact,
)
from treepo.training.local_law import (
    local_law_objective_from_losses,
    local_law_objective_target_mse,
)


def _make_tiny_docs(n: int = 10, seq_len: int = 32, vocab_size: int = 8) -> list:
    """Create tiny synthetic docs for testing."""
    rng = np.random.default_rng(42)
    docs = []
    for _ in range(n):
        n_regimes = 4
        tokens = rng.integers(0, vocab_size, size=seq_len).tolist()
        regimes = rng.integers(0, n_regimes, size=seq_len).tolist()
        boundaries = sorted(rng.choice(range(1, seq_len), size=min(3, seq_len - 1), replace=False).tolist())
        docs.append(ChangepointMarkovDoc(
            tokens=tuple(tokens),
            token_regimes=tuple(regimes),
            transition_regimes=tuple(regimes),
            true_boundaries=tuple(boundaries),
        ))
    return docs


def _make_mixed_length_docs(
    *,
    counts_by_seq_len: Sequence[tuple[int, int]],
    vocab_size: int = 8,
) -> list[ChangepointMarkovDoc]:
    docs: list[ChangepointMarkovDoc] = []
    for seq_len, n_docs in counts_by_seq_len:
        docs.extend(
            _make_tiny_docs(
                n=int(n_docs),
                seq_len=int(seq_len),
                vocab_size=int(vocab_size),
            )
        )
    return docs


def _assert_exact_sketch_metric_mappings_close(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    default_abs_tol: float = 1e-6,
) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        abs_tol = 1e-3 if str(key).startswith("phi_pair_") else float(default_abs_tol)
        if isinstance(expected_value, Mapping):
            assert isinstance(actual_value, Mapping), key
            assert set(actual_value.keys()) == set(expected_value.keys()), key
            for nested_key, nested_expected in expected_value.items():
                nested_actual = actual_value[nested_key]
                if np.isnan(float(nested_expected)):
                    assert np.isnan(float(nested_actual)), f"{key}.{nested_key}"
                else:
                    assert float(nested_actual) == pytest.approx(
                        float(nested_expected),
                        abs=abs_tol,
                    ), f"{key}.{nested_key}"
            continue
        if np.isnan(float(expected_value)):
            assert np.isnan(float(actual_value)), key
        else:
            assert float(actual_value) == pytest.approx(
                float(expected_value),
                abs=abs_tol,
            ), key


def _make_tiny_fno_doc(tokens: Sequence[int], *, root_count: float) -> nob._FNOCountDoc:
    token_ids = tuple(int(t) for t in tokens)
    return nob._FNOCountDoc(
        n_tokens=len(token_ids),
        leaf_token_ids=(token_ids,),
        leaf_counts=(float(root_count),),
        leaf_first_regimes=(0,),
        leaf_last_regimes=(0,),
        leaf_token_lengths=(len(token_ids),),
        merge_counts_balanced=tuple(),
        merge_sizes_balanced=tuple(),
        merge_token_lengths=tuple(),
        root_count=float(root_count),
    )


def test_local_law_objective_corrected_mode_matches_dr_formula():
    out = {
        "all_node_preds": torch.tensor([0.0, 1.0]),
        "all_node_proxy_targets": torch.tensor([0.0, 0.0]),
        "all_node_oracle_targets": torch.tensor([2.0, 1.0]),
        "all_node_observed": torch.tensor([1.0, 0.0]),
        "all_node_propensities": torch.tensor([0.5, 1.0]),
        "all_node_depths": torch.tensor([0.0, 1.0]),
    }
    loss = local_law_objective_target_mse(
        predictions=out["all_node_preds"],
        proxy_targets=out["all_node_proxy_targets"],
        oracle_targets=out["all_node_oracle_targets"],
        observed=out["all_node_observed"],
        propensity=out["all_node_propensities"],
        depths=out["all_node_depths"],
        gamma_depth=0.5,
        objective_mode="corrected_local_law",
    )
    assert float(loss.detach().cpu()) == pytest.approx((8.0 + 0.5) / 1.5)


def test_local_law_objective_exact_proxy_endpoint():
    out = {
        "all_node_preds": torch.tensor([0.0, 1.0, 2.0]),
        "all_node_proxy_targets": torch.tensor([1.0, 1.0, 3.0]),
        "all_node_oracle_targets": torch.tensor([1.0, 1.0, 3.0]),
        "all_node_observed": torch.tensor([0.0, 1.0, 1.0]),
        "all_node_propensities": torch.tensor([0.0, 0.25, 1.0]),
        "all_node_depths": torch.tensor([0.0, 0.0, 0.0]),
    }
    loss = local_law_objective_target_mse(
        predictions=out["all_node_preds"],
        proxy_targets=out["all_node_proxy_targets"],
        oracle_targets=out["all_node_oracle_targets"],
        observed=out["all_node_observed"],
        propensity=out["all_node_propensities"],
        depths=out["all_node_depths"],
        objective_mode="corrected_local_law",
    )
    assert float(loss.detach().cpu()) == pytest.approx((1.0 + 0.0 + 1.0) / 3.0)


def test_local_law_objective_sampled_ipw_mode_uses_observed_subset():
    out = {
        "all_node_preds": torch.tensor([0.0, 100.0]),
        "all_node_proxy_targets": torch.tensor([0.0, 100.0]),
        "all_node_oracle_targets": torch.tensor([2.0, 100.0]),
        "all_node_observed": torch.tensor([1.0, 0.0]),
        "all_node_propensities": torch.tensor([0.1, 1.0]),
        "all_node_depths": torch.tensor([0.0, 0.0]),
    }
    loss = local_law_objective_target_mse(
        predictions=out["all_node_preds"],
        proxy_targets=out["all_node_proxy_targets"],
        oracle_targets=out["all_node_oracle_targets"],
        observed=out["all_node_observed"],
        propensity=out["all_node_propensities"],
        depths=out["all_node_depths"],
        objective_mode="sampled_ipw",
    )
    assert float(loss.detach().cpu()) == pytest.approx(4.0)


def test_local_law_objective_from_losses_matches_target_adapter():
    predictions = torch.tensor([0.0, 1.0])
    proxy_targets = torch.tensor([0.0, 0.0])
    oracle_targets = torch.tensor([2.0, 1.0])
    observed = torch.tensor([1.0, 0.0])
    propensity = torch.tensor([0.5, 1.0])
    depths = torch.tensor([0.0, 1.0])

    direct = local_law_objective_from_losses(
        proxy_loss=(predictions - proxy_targets) ** 2,
        oracle_loss=(predictions - oracle_targets) ** 2,
        observed=observed,
        propensity=propensity,
        depths=depths,
        gamma_depth=0.5,
        objective_mode="corrected_local_law",
    )
    adapter = local_law_objective_target_mse(
        predictions=predictions,
        proxy_targets=proxy_targets,
        oracle_targets=oracle_targets,
        observed=observed,
        propensity=propensity,
        depths=depths,
        gamma_depth=0.5,
        objective_mode="corrected_local_law",
    )

    assert float(direct.detach().cpu()) == pytest.approx(float(adapter.detach().cpu()))


def test_fno_single_lambda_objective_uses_convex_root_local_shares():
    root_loss = torch.tensor(10.0)
    local_law_loss = torch.tensor(2.0)

    combined = _fno_single_lambda_objective_loss(
        root_loss=root_loss,
        local_law_loss=local_law_loss,
        root_objective_share=0.4,
        local_law_objective_share=0.6,
    )
    root_only = _fno_single_lambda_objective_loss(
        root_loss=root_loss,
        local_law_loss=local_law_loss,
        root_objective_share=1.0,
        local_law_objective_share=0.0,
    )
    local_only = _fno_single_lambda_objective_loss(
        root_loss=root_loss,
        local_law_loss=local_law_loss,
        root_objective_share=0.0,
        local_law_objective_share=1.0,
    )

    assert float(combined.detach().cpu()) == pytest.approx(0.4 * 10.0 + 0.6 * 2.0)
    assert float(root_only.detach().cpu()) == pytest.approx(10.0)
    assert float(local_only.detach().cpu()) == pytest.approx(2.0)


def test_fno_single_lambda_objective_rejects_nonconvex_shares():
    with pytest.raises(ValueError, match="convex root/local-law objective"):
        _fno_single_lambda_objective_loss(
            root_loss=torch.tensor(1.0),
            local_law_loss=torch.tensor(1.0),
            root_objective_share=1.0,
            local_law_objective_share=1.0,
        )


def test_fno_tree_local_law_training_requires_resolved_objective_shares():
    signature = inspect.signature(train_fno_tree_local_law)

    assert "root_loss_weight" not in signature.parameters
    assert "local_law_weight" not in signature.parameters
    assert signature.parameters["root_objective_share"].default is inspect.Parameter.empty
    assert signature.parameters["local_law_objective_share"].default is inspect.Parameter.empty


def test_theorem_feature_pair_builder_respects_thresholds():
    class _ThresholdAdapter:
        name = "threshold_scalar_band_test"
        has_canonical_decode = False

        @staticmethod
        def oracle_label(*, count, first=None, last=None, metadata=None):
            return float(count)

        @staticmethod
        def same_pair(left, right, *, same_threshold=None, diff_threshold=None):
            assert same_threshold is not None
            return abs(float(left) - float(right)) <= float(same_threshold)

        @staticmethod
        def different_pair(left, right, *, same_threshold=None, diff_threshold=None):
            assert diff_threshold is not None
            return abs(float(left) - float(right)) >= float(diff_threshold)

        @staticmethod
        def diagnostic_key(label):
            return float(label)

        @staticmethod
        def task_readout_target(label):
            return float(label)

        @staticmethod
        def decode_from_phi(phi):
            return None

    register_theorem_feature_adapter(
        "threshold_scalar_band_test",
        lambda: _ThresholdAdapter(),
        overwrite=True,
    )
    pair_sets = build_theorem_feature_pair_sets(
        [0.0, 0.2, 0.55, 0.9],
        adapter=resolve_theorem_feature_adapter("threshold_scalar_band_test"),
        same_threshold=0.25,
        diff_threshold=0.5,
    )

    assert pair_sets.same_pairs == ((0, 1),)
    assert pair_sets.different_pairs == ((0, 2), (0, 3), (1, 3))


def test_deterministic_sample_indices_are_nested_across_rates() -> None:
    low = _deterministic_sample_indices(n_items=22, rate=2.0 / 22.0, seed=102)
    mid = _deterministic_sample_indices(n_items=22, rate=6.0 / 22.0, seed=102)
    high = _deterministic_sample_indices(n_items=22, rate=10.0 / 22.0, seed=102)

    assert low is not None
    assert mid is not None
    assert high is not None
    assert set(low).issubset(mid)
    assert set(mid).issubset(high)


def test_theorem_feature_pair_metrics_auc_matches_dense_reference():
    same_scores = [0.2, 0.6, 0.6, 0.9]
    different_scores = [0.1, 0.6, 0.8]
    comparisons = (
        np.asarray(same_scores, dtype=np.float64)[:, None]
        - np.asarray(different_scores, dtype=np.float64)[None, :]
    )
    expected_auc = float(
        np.mean((comparisons > 0.0).astype(np.float64))
        + 0.5 * np.mean((comparisons == 0.0).astype(np.float64))
    )

    metrics = theorem_feature_pair_metrics_from_scores(
        same_scores=same_scores,
        different_scores=different_scores,
    )

    assert metrics.phi_pair_auc == pytest.approx(expected_auc)


def test_teacher_first_pair_metrics_cap_sampled_nodes() -> None:
    rng = np.random.default_rng(0)
    n_nodes = 3000
    metrics = nob._teacher_first_pair_metrics_from_node_view(
        stage1_node_scores=rng.normal(size=n_nodes).tolist(),
        final_phi_embeddings=tuple(
            rng.normal(size=8).astype(np.float64) for _ in range(n_nodes)
        ),
        same_threshold=None,
        diff_threshold=None,
        max_nodes=256,
    )

    assert metrics["stage2_fiber_pair_sampled_node_count"] == pytest.approx(256.0)
    assert metrics["stage2_fiber_pair_total_node_count"] == pytest.approx(float(n_nodes))
    assert metrics["stage2_fiber_pair_sampled_pair_count"] == pytest.approx(
        float(256 * 255 // 2)
    )
    assert np.isfinite(float(metrics["stage2_fiber_pair_auc"]))


def test_theorem_feature_stage1_artifact_roundtrip(tmp_path) -> None:
    artifact_dir = tmp_path / "stage1_artifact"
    expected = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    artifact = write_theorem_feature_stage1_artifact(
        artifact_dir,
        model_state={"weight": expected.clone()},
        metadata={
            "selection_metric_name": "val_leaf_codec_direct",
            "selection_metric_value": 0.125,
            "best_epoch": 3,
            "epochs_completed": 4,
            "training_schedule": "two_stage",
            "artifact_source": "trained",
        },
    )
    loaded_artifact, loaded_state = load_theorem_feature_stage1_artifact(artifact_dir)

    assert artifact.selection_metric_name == "val_leaf_codec_direct"
    assert loaded_artifact.best_epoch == 3
    assert torch.equal(loaded_state["weight"], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tree_gpu_batch_store_uses_dense_resident_fixed_shape_buckets() -> None:
    docs = _make_tiny_docs(n=4, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=4)
    runtime_config = GpuRuntimeConfig(
        data_mode="resident",
        preload_splits=("train",),
        preload_targets=False,
    )

    class _PadOnlyModel:
        pad_id = 8

    store, telemetry = nob._build_tree_gpu_batch_store(
        docs=fno_docs,
        model=_PadOnlyModel(),
        device=torch.device("cuda"),
        split_name="train",
        runtime_config=runtime_config,
    )

    assert store is not None
    build_stats = telemetry.as_dict()
    assert int(build_stats["fixed_shape_bucket_store_count"]) == len(store.buckets)
    assert int(build_stats["fixed_shape_dense_bucket_store_count"]) == len(store.buckets)
    assert float(build_stats["fixed_shape_dense_bucket_store_bytes"]) > 0.0
    bucket = next(iter(store.buckets.values()))
    assert bucket["metadata"]["bucket_store_mode"] == "dense_resident"
    assert bucket["metadata"]["resident_layout_mode"] == "dense_fixed_shape"
    assert int(bucket["metadata"]["resident_bucket_bytes"]) > 0

    view_runtime = GpuRuntimeTelemetry(data_mode="resident")
    items = (
        nob._TreeWorkItem(
            doc_index=0,
            doc=fno_docs[0],
            work_kind="train",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
            root_only_supervision=True,
            doc_sequence_supervision=False,
        ),
        nob._TreeWorkItem(
            doc_index=2,
            doc=fno_docs[2],
            work_kind="train",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
            root_only_supervision=True,
            doc_sequence_supervision=False,
        ),
    )
    view = nob._tree_store_view_for_items(
        store,
        items,
        model=_PadOnlyModel(),
        runtime_telemetry=view_runtime,
    )

    assert view is not None
    assert tuple(view.doc_indices) == (0, 2)
    assert view.metadata["bucket_store_mode"] == "dense_resident"
    assert view.metadata["resident_layout_mode"] == "dense_fixed_shape"
    assert int(view.metadata["resident_bucket_bytes"]) > 0
    assert int(view_runtime.as_dict()["resident_store_hits"]) == 1
    assert float(view_runtime.as_dict()["fixed_shape_dense_bucket_store_hits"]) == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tree_gpu_batch_store_auto_queue_dense_view_exposes_masks_and_zero_h2d() -> None:
    docs = _make_mixed_length_docs(
        counts_by_seq_len=((32, 1), (40, 1), (48, 1)),
    )
    fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=8)
    runtime_config = GpuRuntimeConfig(
        data_mode="resident",
        bucket_mode=GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
        preload_splits=("train",),
    )

    class _PadOnlyModel:
        pad_id = 8

    store, telemetry = nob._build_tree_gpu_batch_store(
        docs=fno_docs,
        model=_PadOnlyModel(),
        device=torch.device("cuda"),
        split_name="train",
        runtime_config=runtime_config,
        structural_pad_limit=0.5,
        auto_queue_min_docs=0,
    )

    assert store is not None
    assert int(telemetry.as_dict()["auto_queue_family_count"]) == 1
    assert len(store.buckets) == 1
    bucket = next(iter(store.buckets.values()))
    assert bucket["metadata"]["auto_queue_enabled"] is True
    assert int(bucket["metadata"]["auto_queue_target_n_leaves"]) == 6
    assert "leaf_valid_mask" in bucket["tensors"]
    assert "merge_valid_mask" in bucket["tensors"]
    assert "node_valid_mask" in bucket["tensors"]

    view_runtime = GpuRuntimeTelemetry(data_mode="resident")
    items = tuple(
        nob._tree_work_item_from_doc(
            doc,
            doc_index=idx,
            work_kind="train",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
            root_only_supervision=True,
            doc_sequence_supervision=False,
        )
        for idx, doc in enumerate(fno_docs[:2])
    )
    view = nob._tree_store_view_for_items(
        store,
        items,
        model=_PadOnlyModel(),
        runtime_telemetry=view_runtime,
    )

    assert view is not None
    assert view.metadata["bucket_store_mode"] == "dense_resident"
    assert view.metadata["auto_queue_enabled"] is True
    assert int(view.metadata["auto_queue_target_n_leaves"]) == 6
    assert tuple(view.tensors["leaf_valid_mask"].shape) == (2, 6)
    assert tuple(view.tensors["merge_valid_mask"].shape)[0] == 2
    assert tuple(view.tensors["node_valid_mask"].shape)[0] == 2
    payload = view_runtime.as_dict()
    assert int(payload["resident_store_hits"]) == 1
    assert int(payload["host_to_device_bytes"]) == 0
    assert float(payload["auto_queue_fused_batches"]) == pytest.approx(1.0)


def test_markov_theorem_feature_adapter_preserves_exact_tuple_semantics():
    adapter = resolve_theorem_feature_adapter("markov_count_sketch")
    same_left = adapter.oracle_label(count=2.0, first=1, last=3)
    same_right = adapter.oracle_label(count=2.0, first=1, last=3)
    diff_label = adapter.oracle_label(count=2.0, first=1, last=2)

    assert adapter.same_pair(same_left, same_right) is True
    assert adapter.different_pair(same_left, same_right) is False
    assert adapter.same_pair(same_left, diff_label) is False
    assert adapter.different_pair(same_left, diff_label) is True
    assert adapter.diagnostic_key(same_left) == (2, 1, 3)
    assert adapter.task_readout_target(same_left) == pytest.approx(2.0)


def test_fast_markov_pair_masks_match_generic_pair_builder():
    adapter = resolve_theorem_feature_adapter("markov_count_sketch")
    labels = [
        adapter.oracle_label(count=2.0, first=1, last=3),
        adapter.oracle_label(count=2.0, first=1, last=3),
        adapter.oracle_label(count=2.0, first=2, last=3),
        adapter.oracle_label(count=4.0, first=2, last=1),
    ]
    pair_sets = build_theorem_feature_pair_sets(labels, adapter=adapter)

    class _FastModel:
        theorem_feature_adapter_name = "markov_count_sketch"
        theorem_feature_adapter = adapter
        oracle_metric = None

    count_values = torch.tensor([2.0, 2.0, 2.0, 4.0], dtype=torch.float32)
    count_keys = torch.round(count_values).to(dtype=torch.long)
    first_targets = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    last_targets = torch.tensor([3, 3, 3, 1], dtype=torch.long)
    same_mask, different_mask = nob._fast_markov_pair_masks_from_tensors(
        _FastModel(),
        count_keys=count_keys,
        first_targets=first_targets,
        last_targets=last_targets,
    )

    same_pairs = tuple(
        (int(left), int(right))
        for left, right in torch.nonzero(torch.triu(same_mask, diagonal=1), as_tuple=False).tolist()
    )
    different_pairs = tuple(
        (int(left), int(right))
        for left, right in torch.nonzero(torch.triu(different_mask, diagonal=1), as_tuple=False).tolist()
    )

    assert same_pairs == pair_sets.same_pairs
    assert different_pairs == pair_sets.different_pairs


def test_fast_markov_mask_contrastive_matches_generic_loss():
    adapter = resolve_theorem_feature_adapter("markov_count_sketch")

    class _FastModel:
        theorem_feature_adapter_name = "markov_count_sketch"
        theorem_feature_adapter = adapter
        oracle_metric = None

    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = [
        adapter.oracle_label(count=2.0, first=1, last=3),
        adapter.oracle_label(count=2.0, first=1, last=3),
        adapter.oracle_label(count=2.0, first=2, last=3),
        adapter.oracle_label(count=4.0, first=2, last=1),
    ]
    pair_sets = build_theorem_feature_pair_sets(labels, adapter=adapter)
    generic_loss = nob._pairwise_theorem_feature_contrastive_loss(
        embeddings,
        same_pairs=pair_sets.same_pairs,
        different_pairs=pair_sets.different_pairs,
    )

    count_values = torch.tensor([2.0, 2.0, 2.0, 4.0], dtype=torch.float32)
    count_keys = torch.round(count_values).to(dtype=torch.long)
    first_targets = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    last_targets = torch.tensor([3, 3, 3, 1], dtype=torch.long)
    same_mask, different_mask = nob._fast_markov_pair_masks_from_tensors(
        _FastModel(),
        count_keys=count_keys,
        first_targets=first_targets,
        last_targets=last_targets,
    )
    fast_loss = nob._pairwise_theorem_feature_contrastive_loss_from_masks(
        embeddings,
        same_mask=same_mask,
        different_mask=different_mask,
    )

    assert float(fast_loss.detach().cpu()) == pytest.approx(float(generic_loss.detach().cpu()))


def test_fixed_fused_leaf_root_c2_uses_fast_deferred_markov_path(monkeypatch):
    raw_docs = _make_tiny_docs(n=2, seq_len=32, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=32.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="factorized_score_fiber",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=8,
        theorem_feature_hidden_dim=16,
        theorem_score_dim=1,
        theorem_fiber_dim=7,
        tree_model_version="v2",
    ).to(device=torch.device("cpu"))

    def _unexpected_summary_spec(*args, **kwargs):
        raise AssertionError("leaf/merge summary-spec loss should be skipped when c1=c3=0")

    monkeypatch.setattr(nob, "_summary_spec_supervision_terms_batched", _unexpected_summary_spec)

    items = []
    work_lookup = {}
    for idx, doc in enumerate(fno_docs):
        items.append(
            nob._tree_work_item_from_doc(
                doc,
                doc_index=idx,
                work_kind="root_only",
                collect_leaf=True,
                collect_c2=True,
                collect_c3=False,
                root_only_supervision=True,
                doc_sequence_supervision=False,
                leaf_audit_indices=None,
                c3_audit_indices=set(),
                document_mode="root_only",
            )
        )
        work_lookup[idx] = {
            "doc_index": idx,
            "doc": doc,
            "root_only_supervision": True,
            "doc_sequence_supervision": False,
            "doc_sequence_loss": torch.zeros((), dtype=torch.float32),
            "collect_leaf": True,
            "collect_c2": True,
            "collect_c3": False,
            "leaf_audit_indices": None,
            "c3_audit_indices": set(),
        }

    packed = nob._pack_tree_work_items(
        items,
        max_docs=len(items),
        max_total_leaf_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        bucket_docs_cap_by_n_leaves=None,
    )

    fused = nob._fixed_fused_training_batch_forward(
        model,
        packed[0],
        work_lookup=work_lookup,
        device=torch.device("cpu"),
        root_weight=1.0,
        c1_weight=0.0,
        c2_weight=1.0,
        c3_weight=0.0,
        phi_compose_weight=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="count_only",
        defer_contrastive=True,
    )

    deferred_batch = fused.get("deferred_phi_feature_batch")
    deferred_fast_keys = fused.get("deferred_phi_fast_keys")
    expected_nodes = sum(int(len(doc.leaf_token_ids)) + 1 for doc in fno_docs)

    assert isinstance(deferred_batch, torch.Tensor)
    assert int(deferred_batch.shape[0]) == int(expected_nodes)
    assert isinstance(deferred_fast_keys, dict)
    assert int(deferred_fast_keys["count_keys"].numel()) == int(expected_nodes)
    assert int(fused["component_counts"]["c2_count_loss"]) == int(len(fno_docs))


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_theorem_feature_parent_helpers_match_state_helpers() -> None:
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=32.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="factorized_score_fiber",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=8,
        theorem_feature_hidden_dim=16,
        theorem_score_dim=1,
        theorem_fiber_dim=7,
        tree_model_version="v2",
    ).to(device=torch.device("cpu"))

    left_state = torch.randn((3, 16), dtype=torch.float32)
    right_state = torch.randn((3, 16), dtype=torch.float32)
    left_feature = model.theorem_feature_from_state(left_state)
    right_feature = model.theorem_feature_from_state(right_state)

    expected_score = model.predict_score_parent_from_children(left_state, right_state)
    actual_score = model.predict_score_parent_from_theorem_features(
        left_feature,
        right_feature,
    )
    assert torch.allclose(actual_score, expected_score, atol=1e-6, rtol=1e-6)

    expected_phi = model.predict_phi_parent_from_children(left_state, right_state)
    actual_phi = model.predict_phi_parent_from_theorem_features(
        left_feature,
        right_feature,
    )
    assert torch.allclose(actual_phi, expected_phi, atol=1e-6, rtol=1e-6)

    expected_leaf_phi = model.predict_phi_from_state(left_state)
    actual_leaf_phi = model.predict_phi_from_theorem_feature(left_feature)
    assert torch.allclose(actual_leaf_phi, expected_leaf_phi, atol=1e-6, rtol=1e-6)


def test_scorefiber_length_bucket_adapter_uses_leaf_span_count_when_available():
    adapter = resolve_theorem_feature_adapter("scorefiber_length_bucket")
    small = adapter.oracle_label(count=2.0, metadata={"leaf_span_count": 2})
    medium = adapter.oracle_label(count=2.0, metadata={"leaf_span_count": 4})
    large = adapter.oracle_label(count=2.0, metadata={"leaf_span_count": 6})

    assert adapter.diagnostic_key(small) == 0
    assert adapter.diagnostic_key(medium) == 1
    assert adapter.diagnostic_key(large) == 2
    assert adapter.task_readout_target(large) == pytest.approx(2.0)


def test_tree_work_item_bucket_and_packing_handle_mixed_work_kinds():
    raw_docs = _make_tiny_docs(n=2, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=4)
    aggregate_doc = nob._aggregate_fno_doc_from_leaf_range(
        fno_docs[0],
        start_leaf_idx=0,
        end_leaf_idx=2,
    )

    full_item = nob._tree_work_item_from_doc(
        fno_docs[0],
        doc_index=0,
        work_kind="full_tree",
        collect_leaf=True,
        collect_c2=True,
        collect_c3=True,
    )
    root_item = nob._tree_work_item_from_doc(
        fno_docs[1],
        doc_index=1,
        work_kind="root_only",
    )
    aggregate_item = nob._tree_work_item_from_doc(
        aggregate_doc,
        doc_index=2,
        work_kind="aggregate",
        collect_leaf=True,
    )

    assert nob._tree_work_item_bucket_key(full_item).work_kind == "full_tree"
    assert nob._tree_work_item_bucket_key(root_item).work_kind == "root_only"
    assert nob._tree_work_item_bucket_key(aggregate_item).work_kind == "aggregate"

    packed = nob._pack_tree_work_items(
        (full_item, root_item, aggregate_item),
        max_docs=2,
        max_total_leaf_tokens=256,
        max_total_nodes=64,
        max_total_merge_ops=64,
    )

    assert sum(len(batch.items) for batch in packed) == 3
    assert {item.work_kind for batch in packed for item in batch.items} == {
        "aggregate",
        "full_tree",
        "root_only",
    }
    assert all(batch.padded_leaf_tokens >= batch.actual_leaf_tokens for batch in packed)


def test_tree_work_item_packing_unified_v2_batches_same_shape_docs_together():
    raw_docs = _make_tiny_docs(n=2, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=4)
    items = tuple(
        nob._tree_work_item_from_doc(
            doc,
            doc_index=idx,
            work_kind="full_tree",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
        )
        for idx, doc in enumerate(fno_docs)
    )

    packed = nob._pack_tree_work_items(
        items,
        max_docs=2,
        max_total_leaf_tokens=512,
        max_total_nodes=128,
        max_total_merge_ops=128,
        runtime_mode="unified_v2",
    )

    assert len(packed) == 1
    assert len(packed[0].items) == 2
    assert packed[0].actual_leaf_tokens > 0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_batched_root_predictions_records_batching_metrics():
    raw_docs = _make_tiny_docs(n=3, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
    )
    metrics = nob._BatchingMetricsAccumulator()

    preds, truths = nob._batched_root_predictions(
        model,
        fno_docs,
        device=torch.device("cpu"),
        max_docs=2,
        token_budget=16,
        node_budget=8,
        bucket_docs_cap_by_n_leaves={2: 1},
        batching_metrics=metrics,
    )

    payload = metrics.as_dict(device=torch.device("cpu"))

    assert preds.shape == truths.shape == (3,)
    assert payload["mean_docs_per_batch"] == pytest.approx(1.0)
    assert payload["mean_leaf_tokens_per_batch"] > 0.0
    assert payload["mean_nodes_per_batch"] > 0.0
    assert payload["bucket_utilization_rate"] > 0.0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_batched_root_predictions_fixed_fused_matches_structure_bucket():
    raw_docs = _make_tiny_docs(n=4, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )

    bucket_preds, bucket_truths = nob._batched_root_predictions(
        model,
        fno_docs,
        device=torch.device("cpu"),
        pack_mode="structure_bucket",
        max_docs=2,
        token_budget=64,
        node_budget=16,
    )
    fused_preds, fused_truths = nob._batched_root_predictions(
        model,
        fno_docs,
        device=torch.device("cpu"),
        pack_mode="fixed_fused",
        max_docs=2,
        token_budget=64,
        node_budget=16,
    )

    assert fused_truths.tolist() == pytest.approx(bucket_truths.tolist())
    assert fused_preds.tolist() == pytest.approx(bucket_preds.tolist(), abs=1e-6)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_pack_tree_work_items_auto_queue_merges_compatible_mixed_leaf_counts() -> None:
    raw_docs = _make_mixed_length_docs(
        counts_by_seq_len=((32, 2), (40, 2), (48, 2)),
    )
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    items = tuple(
        nob._tree_work_item_from_doc(
            doc,
            doc_index=idx,
            work_kind="full_tree",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
        )
        for idx, doc in enumerate(fno_docs)
    )

    exact_batches = nob._pack_tree_work_items(
        items,
        max_docs=4,
        max_total_leaf_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        bucket_mode="exact_then_bucketed",
    )
    auto_batches = nob._pack_tree_work_items(
        items,
        max_docs=4,
        max_total_leaf_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        bucket_mode=GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
        structural_pad_limit=0.5,
        auto_queue_min_docs=0,
        auto_queue_target_by_n_leaves={4: 6, 5: 6, 6: 6},
        tail_repack_fill_ratio=0.5,
        tail_repack_min_docs=0,
    )

    assert [len(batch.items) for batch in exact_batches] == [2, 2, 2]
    assert [len(batch.items) for batch in auto_batches] == [4, 2]
    assert all(bool(batch.bucket_key.auto_queue_enabled) for batch in auto_batches)
    assert all(int(batch.bucket_key.n_leaves) == 6 for batch in auto_batches)
    assert all(int(batch.padded_leaf_slots) >= int(batch.actual_leaf_slots) for batch in auto_batches)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_repack_small_tail_tree_batches_preserves_auto_queue_fixed_fused_compatibility() -> None:
    raw_docs = _make_mixed_length_docs(
        counts_by_seq_len=((32, 2), (40, 2)),
    )
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    items = tuple(
        nob._tree_work_item_from_doc(
            doc,
            doc_index=idx,
            work_kind="full_tree",
            collect_leaf=True,
            collect_c2=True,
            collect_c3=True,
        )
        for idx, doc in enumerate(fno_docs)
    )
    bucket_key = nob._BatchBucketKey(
        n_leaves=6,
        work_kind="full_tree",
        collect_leaf=True,
        collect_c2=True,
        collect_c3=True,
        max_leaf_tokens_band=8,
        max_merge_tokens_band=8,
        irregular_leaf_layout=False,
        auto_queue_enabled=True,
    )
    packed_batches = [
        nob._PackedTreeBatch(
            bucket_key=bucket_key,
            items=tuple(items[:2]),
            actual_leaf_tokens=sum(int(item.total_leaf_tokens) for item in items[:2]),
            padded_leaf_tokens=2 * 6 * 8,
            total_nodes=2 * ((2 * 6) - 1),
            total_merge_ops=2 * (6 - 1),
            actual_leaf_slots=sum(int(len(item.doc.leaf_token_ids)) for item in items[:2]),
            padded_leaf_slots=2 * 6,
        ),
        nob._PackedTreeBatch(
            bucket_key=bucket_key,
            items=tuple(items[2:]),
            actual_leaf_tokens=sum(int(item.total_leaf_tokens) for item in items[2:]),
            padded_leaf_tokens=2 * 6 * 8,
            total_nodes=2 * ((2 * 6) - 1),
            total_merge_ops=2 * (6 - 1),
            actual_leaf_slots=sum(int(len(item.doc.leaf_token_ids)) for item in items[2:]),
            padded_leaf_slots=2 * 6,
        ),
    ]

    repacked = nob._repack_small_tail_tree_batches(
        packed_batches,
        max_docs=4,
        max_total_leaf_tokens=4 * 6 * 8,
        max_total_nodes=4 * ((2 * 6) - 1),
        max_total_merge_ops=4 * (6 - 1),
        bucket_docs_cap_by_n_leaves={6: 4},
        tail_repack_fill_ratio=0.75,
        tail_repack_min_docs=0,
    )

    assert [len(batch.items) for batch in repacked] == [4]
    assert all(bool(batch.bucket_key.auto_queue_enabled) for batch in repacked)
    assert all(int(batch.bucket_key.n_leaves) == 6 for batch in repacked)
    assert all(int(batch.padded_leaf_slots) == 24 for batch in repacked)

    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )
    assert all(nob._supports_fixed_fused_batch(model, batch.items) for batch in repacked)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_batched_root_predictions_fixed_fused_auto_queue_matches_structure_bucket_mixed_leaf_counts():
    raw_docs = _make_mixed_length_docs(
        counts_by_seq_len=((32, 2), (40, 2), (48, 2)),
    )
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )

    bucket_preds, bucket_truths = nob._batched_root_predictions(
        model,
        fno_docs,
        device=torch.device("cpu"),
        pack_mode="structure_bucket",
        max_docs=3,
        token_budget=256,
        node_budget=64,
    )
    fused_preds, fused_truths = nob._batched_root_predictions(
        model,
        fno_docs,
        device=torch.device("cpu"),
        pack_mode="fixed_fused",
        runtime_bucket_mode=GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
        max_docs=3,
        token_budget=256,
        node_budget=64,
        structural_pad_limit=0.5,
        auto_queue_min_docs=0,
        auto_queue_min_fill_ratio=0.5,
        auto_queue_target_by_n_leaves={4: 6, 5: 6, 6: 6},
    )

    assert fused_truths.tolist() == pytest.approx(bucket_truths.tolist())
    assert fused_preds.tolist() == pytest.approx(bucket_preds.tolist(), abs=1e-6)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_autotune_tree_batch_budgets_fixed_fused_keys_caps_by_auto_queue_target() -> None:
    raw_docs = _make_mixed_length_docs(
        counts_by_seq_len=((32, 2), (40, 2), (48, 2)),
    )
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
    )

    budgets = nob._autotune_tree_batch_budgets(
        model,
        fno_docs,
        device=torch.device("cpu"),
        legacy_batch_size=64,
        pack_mode="fixed_fused",
        bucket_mode=GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
        structural_pad_limit=0.5,
        auto_queue_min_docs=0,
    )

    assert {n_leaves for n_leaves, _cap in budgets.train_bucket_max_docs_by_n_leaves} == {6}
    assert {n_leaves for n_leaves, _cap in budgets.eval_bucket_max_docs_by_n_leaves} == {6}


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_autotune_tree_batch_budgets_returns_nonzero_caps():
    raw_docs = list(_make_tiny_docs(n=1, seq_len=16, vocab_size=8))
    raw_docs.extend(_make_tiny_docs(n=1, seq_len=32, vocab_size=8))
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
    )

    budgets = nob._autotune_tree_batch_budgets(
        model,
        fno_docs,
        device=torch.device("cpu"),
        legacy_batch_size=64,
    )

    assert budgets.train_leaf_token_budget > 0
    assert budgets.train_node_budget > 0
    assert budgets.eval_leaf_token_budget > 0
    assert budgets.eval_node_budget > 0
    assert budgets.eval_workers_per_mig == 1
    assert budgets.probe_diagnostics.as_dict()["profile_version"] == AUTOTUNE_PROBE_CACHE_VERSION
    assert {n_leaves for n_leaves, _cap in budgets.train_bucket_max_docs_by_n_leaves} == {
        2,
        4,
    }


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_train_single_stage_fixed_fused_uses_fused_helper(monkeypatch):
    raw_docs = _make_tiny_docs(n=2, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )

    calls = {"fused": 0}

    def _fake_fused(*args, **kwargs):
        calls["fused"] += 1
        zero = torch.zeros((), dtype=torch.float32)
        return {
            "batch_loss": zero,
            "component_sums": {
                "root_count_loss": zero,
                "leaf_count_loss": zero,
                "leaf_first_loss": zero,
                "leaf_last_loss": zero,
                "merge_count_loss": zero,
                "merge_first_loss": zero,
                "merge_last_loss": zero,
                "c2_count_loss": zero,
                "c2_first_loss": zero,
                "c2_last_loss": zero,
                "c2_join_loss": zero,
                "c2_on_range_reencode_loss": zero,
                "phi_compose_loss": zero,
                "phi_contrastive_loss": zero,
            },
            "component_counts": {
                "root_count_loss": 0,
                "leaf_count_loss": 0,
                "leaf_first_loss": 0,
                "leaf_last_loss": 0,
                "merge_count_loss": 0,
                "merge_first_loss": 0,
                "merge_last_loss": 0,
                "c2_count_loss": 0,
                "c2_first_loss": 0,
                "c2_last_loss": 0,
                "c2_join_loss": 0,
                "c2_on_range_reencode_loss": 0,
                "phi_compose_loss": 0,
                "phi_contrastive_loss": 0,
            },
            "deferred_phi_features": [],
            "deferred_phi_labels": [],
            "deferred_oracle_vecs": [],
        }

    monkeypatch.setattr(nob, "_fixed_fused_training_batch_forward", _fake_fused)
    monkeypatch.setattr(
        model,
        "forward_doc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("forward_doc should not run")),
    )
    monkeypatch.setattr(
        nob,
        "_eval_fno_exact_sketch_direct_metrics",
        lambda *args, **kwargs: nob._empty_exact_sketch_direct_metrics(),
    )
    monkeypatch.setattr(
        nob,
        "_batched_root_predictions",
        lambda *args, **kwargs: (
            np.zeros((len(kwargs.get("docs", args[1])),), dtype=np.float64),
            np.zeros((len(kwargs.get("docs", args[1])),), dtype=np.float64),
        ),
    )

    result = nob._train_fno_tree_single_stage(
        model=model,
        train_docs=fno_docs,
        val_docs=tuple(),
        device=torch.device("cpu"),
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        c1_weight=1.0,
        c2_weight=1.0,
        c3_weight=1.0,
        root_weight=1.0,
        tree_batch_pack_mode="fixed_fused",
        tree_batch_autotune=False,
        tree_batch_token_budget=64,
        tree_batch_node_budget=16,
    )

    assert calls["fused"] >= 1
    assert result["autotune_probe_profile"]["probe_run_count"] == 0
    assert result["timing_breakdown"]["autotune_total_s"] == pytest.approx(0.0)
    assert result["batching_metrics"]["autotune_probe_runs"] == 0
    assert result["autotuned_batch_budgets"]["probe_cache_hits"] == 0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_train_single_stage_autotune_keeps_configured_train_batch_size(monkeypatch):
    raw_docs = _make_tiny_docs(n=4, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )

    pack_calls = []
    original_pack = nob._pack_tree_work_items

    def _record_pack(items, **kwargs):
        pack_calls.append(
            {
                "n_items": len(items),
                "max_docs": int(kwargs["max_docs"]),
                "bucket_docs_cap_by_n_leaves": dict(
                    kwargs.get("bucket_docs_cap_by_n_leaves") or {}
                ),
            }
        )
        return original_pack(items, **kwargs)

    def _fake_fused(*args, **kwargs):
        zero = torch.zeros((), dtype=torch.float32)
        return {
            "batch_loss": zero,
            "component_sums": {
                "root_count_loss": zero,
                "leaf_count_loss": zero,
                "leaf_first_loss": zero,
                "leaf_last_loss": zero,
                "merge_count_loss": zero,
                "merge_first_loss": zero,
                "merge_last_loss": zero,
                "c2_count_loss": zero,
                "c2_first_loss": zero,
                "c2_last_loss": zero,
                "c2_join_loss": zero,
                "c2_on_range_reencode_loss": zero,
                "phi_compose_loss": zero,
                "phi_contrastive_loss": zero,
            },
            "component_counts": {
                "root_count_loss": 0,
                "leaf_count_loss": 0,
                "leaf_first_loss": 0,
                "leaf_last_loss": 0,
                "merge_count_loss": 0,
                "merge_first_loss": 0,
                "merge_last_loss": 0,
                "c2_count_loss": 0,
                "c2_first_loss": 0,
                "c2_last_loss": 0,
                "c2_join_loss": 0,
                "c2_on_range_reencode_loss": 0,
                "phi_compose_loss": 0,
                "phi_contrastive_loss": 0,
            },
            "deferred_phi_features": [],
            "deferred_phi_labels": [],
            "deferred_oracle_vecs": [],
        }

    monkeypatch.setattr(nob, "_pack_tree_work_items", _record_pack)
    monkeypatch.setattr(nob, "_fixed_fused_training_batch_forward", _fake_fused)
    monkeypatch.setattr(
        nob,
        "_autotune_tree_batch_budgets",
        lambda *args, **kwargs: nob._AutotunedTreeBatchBudgets(
            train_leaf_token_budget=4096,
            train_node_budget=1024,
            eval_leaf_token_budget=4096,
            eval_node_budget=1024,
            eval_workers_per_mig=1,
            train_bucket_max_docs_by_n_leaves=((2, 99),),
            eval_bucket_max_docs_by_n_leaves=((2, 123),),
        ),
    )
    monkeypatch.setattr(
        nob,
        "_eval_fno_exact_sketch_direct_metrics",
        lambda *args, **kwargs: nob._empty_exact_sketch_direct_metrics(),
    )
    monkeypatch.setattr(
        nob,
        "_batched_root_predictions",
        lambda *args, **kwargs: (
            np.zeros((len(kwargs.get("docs", args[1])),), dtype=np.float64),
            np.zeros((len(kwargs.get("docs", args[1])),), dtype=np.float64),
        ),
    )

    result = nob._train_fno_tree_single_stage(
        model=model,
        train_docs=fno_docs,
        val_docs=tuple(),
        device=torch.device("cpu"),
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        c1_weight=1.0,
        c2_weight=1.0,
        c3_weight=1.0,
        root_weight=1.0,
        tree_batch_pack_mode="fixed_fused",
        tree_batch_autotune=True,
    )

    assert pack_calls
    assert pack_calls[0]["max_docs"] == 2
    assert pack_calls[0]["bucket_docs_cap_by_n_leaves"] == {2: 2}
    assert result["autotuned_batch_budgets"]["effective_train_max_docs"] == 2
    assert result["autotuned_batch_budgets"]["train_bucket_max_docs_by_n_leaves"] == {
        "2": 2
    }
    assert result["autotuned_batch_budgets"]["train_bucket_max_docs_by_n_leaves_raw"] == {
        "2": 99
    }


def test_masked_doc_hajek_means_respects_inverse_propensity_weights() -> None:
    values = torch.tensor(
        [
            [1.0, 4.0, 9.0],
            [2.0, 6.0, 8.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
        ]
    )
    propensities = torch.tensor(
        [
            [0.25, 0.50, 1.0],
            [0.50, 1.00, 0.25],
        ],
        dtype=torch.float32,
    )

    means, active = nob._masked_doc_hajek_means(values, mask, propensities)

    assert active.tolist() == [True, True]
    assert means[0].item() == pytest.approx((4.0 * 1.0 + 2.0 * 4.0) / (4.0 + 2.0))
    assert means[1].item() == pytest.approx((2.0 * 2.0 + 4.0 * 8.0) / (2.0 + 4.0))


def test_masked_doc_local_means_supports_subset_mean_and_fixed_k_hajek() -> None:
    values = torch.tensor(
        [
            [1.0, 4.0, 9.0],
            [2.0, 6.0, 8.0],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
        ]
    )
    propensities = torch.tensor(
        [
            [0.25, 0.50, 1.0],
            [0.50, 1.00, 0.25],
        ],
        dtype=torch.float32,
    )

    subset_means, subset_active, subset_numerators, subset_denominators = (
        nob._masked_doc_local_means(
            values,
            mask,
            propensities,
            weighting_mode="subset_mean",
        )
    )
    hajek_means, hajek_active, hajek_numerators, hajek_denominators = (
        nob._masked_doc_local_means(
            values,
            mask,
            propensities,
            weighting_mode="fixed_k_hajek",
        )
    )
    reference_hajek_means, reference_hajek_active = nob._masked_doc_hajek_means(
        values,
        mask,
        propensities,
    )

    assert subset_active.tolist() == [True, True]
    assert subset_means[0].item() == pytest.approx((1.0 + 4.0) / 2.0)
    assert subset_means[1].item() == pytest.approx((2.0 + 8.0) / 2.0)
    assert subset_numerators.tolist() == pytest.approx([5.0, 10.0])
    assert subset_denominators.tolist() == pytest.approx([2.0, 2.0])

    assert hajek_active.tolist() == [True, True]
    assert reference_hajek_active.tolist() == [True, True]
    assert hajek_means.tolist() == pytest.approx(reference_hajek_means.tolist())
    assert hajek_numerators.tolist() == pytest.approx([12.0, 36.0])
    assert hajek_denominators.tolist() == pytest.approx([6.0, 6.0])


def test_bounded_endpoint_surprise_loss_supports_single_item_logits() -> None:
    logits = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32)
    loss = nob._bounded_endpoint_surprise_loss(logits, torch.tensor(1))

    probs = torch.softmax(logits, dim=-1)
    expected = 1.0 - float(probs[1].item())

    assert loss.ndim == 0
    assert float(loss.item()) == pytest.approx(expected)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_bounded_full_sketch_terms_are_unit_bounded() -> None:
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=16,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )
    state_batch = torch.zeros((3, 16), dtype=torch.float32)
    truth_counts = torch.tensor([0.0, 2.0, 8.0], dtype=torch.float32)
    truth_first = torch.tensor([0, 1, 2], dtype=torch.long)
    truth_last = torch.tensor([1, 2, 3], dtype=torch.long)

    terms = nob._local_supervision_terms_batched(
        model,
        state_batch,
        truth_counts=truth_counts,
        truth_first=truth_first,
        truth_last=truth_last,
        supervision_kind="bounded_full_sketch",
    )

    for key in ("count_loss", "first_loss", "last_loss", "total_loss"):
        values = terms[key]
        assert torch.all(values >= 0.0)
        assert torch.all(values <= 1.0 + 1e-6)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_summary_spec_full_sketch_reuses_count_hidden_once(monkeypatch) -> None:
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=16,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )
    state_batch = torch.zeros((5, 16), dtype=torch.float32)
    truth_counts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    truth_first = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    truth_last = torch.tensor([1, 2, 3, 0, 1], dtype=torch.long)

    hidden_calls = 0
    first_calls = 0
    last_calls = 0

    original_hidden = model._count_hidden_from_state
    original_first = model._first_surface_from_state
    original_last = model._last_surface_from_state

    def _count_hidden(state):
        nonlocal hidden_calls
        hidden_calls += 1
        return original_hidden(state)

    def _first_surface(state):
        nonlocal first_calls
        first_calls += 1
        return original_first(state)

    def _last_surface(state):
        nonlocal last_calls
        last_calls += 1
        return original_last(state)

    monkeypatch.setattr(model, "_count_hidden_from_state", _count_hidden)
    monkeypatch.setattr(model, "_first_surface_from_state", _first_surface)
    monkeypatch.setattr(model, "_last_surface_from_state", _last_surface)

    terms = nob._local_supervision_terms_batched(
        model,
        state_batch,
        truth_counts=truth_counts,
        truth_first=truth_first,
        truth_last=truth_last,
        supervision_kind="full_sketch",
    )

    assert hidden_calls <= 1
    assert first_calls <= 1
    assert last_calls <= 1
    assert tuple(terms["total_loss"].shape) == (5,)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_summary_spec_full_sketch_reuses_theorem_feature_once(monkeypatch) -> None:
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=16,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )
    state_batch = torch.zeros((5, 16), dtype=torch.float32)
    truth_counts = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    truth_first = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    truth_last = torch.tensor([1, 2, 3, 0, 1], dtype=torch.long)

    theorem_calls = 0
    original_theorem = model.theorem_feature_from_state

    def _theorem_feature(state):
        nonlocal theorem_calls
        theorem_calls += 1
        return original_theorem(state)

    monkeypatch.setattr(model, "theorem_feature_from_state", _theorem_feature)

    terms = nob._local_supervision_terms_batched(
        model,
        state_batch,
        truth_counts=truth_counts,
        truth_first=truth_first,
        truth_last=truth_last,
        supervision_kind="full_sketch",
    )

    assert theorem_calls == 1
    assert tuple(terms["total_loss"].shape) == (5,)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_probe_tree_docs_cap_uses_persisted_cache_on_second_run(monkeypatch, tmp_path):
    raw_docs = _make_tiny_docs(n=1, seq_len=32, vocab_size=8)
    doc = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)[0]
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )

    class _Props:
        total_memory = 24 * 1024 ** 3
        name = "NVIDIA A100-SXM4-40GB MIG 1g.24gb"
        major = 8
        minor = 0

    reserved_bytes = iter(
        [
            1 * 1024 ** 3,
            2 * 1024 ** 3,
            4 * 1024 ** 3,
            14 * 1024 ** 3,
        ]
    )

    allocated_bytes = iter(
        [
            int(0.8 * 1024 ** 3),
            int(1.6 * 1024 ** 3),
            int(3.2 * 1024 ** 3),
            int(12.0 * 1024 ** 3),
        ]
    )

    def _fake_levels(*args, **kwargs):
        batch_docs = list(args[1])
        batch_size = len(batch_docs)
        return nob._PrecomputedBatchTreeLevels(
            leaf_states=torch.zeros((batch_size, len(doc.leaf_token_ids), model.state_dim)),
            merge_levels=tuple(),
            root_states=torch.zeros((batch_size, model.state_dim)),
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: _Props())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device=None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: next(reserved_bytes))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: next(allocated_bytes))
    monkeypatch.setattr(nob, "_precompute_balanced_doc_state_levels", _fake_levels)
    monkeypatch.setattr(nob, "_autocast_context", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(
        model,
        "predict_canonical_count_from_state",
        lambda root_states: torch.zeros((int(root_states.shape[0]),), dtype=torch.float32),
    )

    cache_store = ProbeCacheStore(root_dir=tmp_path / "probe_cache")
    kwargs = dict(
        model=model,
        doc=doc,
        device=torch.device("cuda"),
        training=False,
        max_candidate_docs=8,
        pack_mode="fixed_fused",
        heuristic_docs_cap=4,
        probe_cache=cache_store,
        model_signature=nob._tree_batch_probe_model_signature(model),
        device_class_signature=nob._tree_batch_probe_device_signature(torch.device("cuda")),
        topology_signature=nob._tree_batch_probe_topology_signature(doc),
    )

    first = nob._probe_tree_docs_cap_for_representative(**kwargs)
    second = nob._probe_tree_docs_cap_for_representative(**kwargs)

    assert first.cache_hit is False
    assert first.selected_docs_cap == 4
    assert first.candidate_evaluations == 4
    assert len(first.run_profile.candidate_profiles) == 4
    assert first.run_profile.candidate_profiles[-1].stop_reason == "target_fraction_exceeded"
    assert second.cache_hit is True
    assert second.selected_docs_cap == 4
    assert second.candidate_evaluations == 0
    assert second.run_profile.stop_reason.startswith("cache_hit:")
    assert len(list((tmp_path / "probe_cache").glob("*.json"))) == 1


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_train_single_stage_bounds_exact_eval_docs_to_train_batch_size(monkeypatch):
    raw_docs = _make_tiny_docs(n=4, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="shared_feature",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_feature_dim=16,
        theorem_feature_hidden_dim=32,
    )

    monkeypatch.setattr(
        nob,
        "_autotune_tree_batch_budgets",
        lambda *args, **kwargs: nob._AutotunedTreeBatchBudgets(
            train_leaf_token_budget=0,
            train_node_budget=0,
            eval_leaf_token_budget=0,
            eval_node_budget=0,
            eval_workers_per_mig=1,
            train_bucket_max_docs_by_n_leaves=((2, 99),),
            eval_bucket_max_docs_by_n_leaves=((2, 99),),
        ),
    )
    monkeypatch.setattr(
        nob,
        "_batched_root_predictions",
        lambda *args, **kwargs: (
            np.zeros((len(kwargs.get("docs", args[1])),), dtype=np.float64),
            np.zeros((len(kwargs.get("docs", args[1])),), dtype=np.float64),
        ),
    )
    captured: dict[str, int] = {}

    def _fake_exact_metric(
        _model,
        _docs,
        *,
        device,
        max_docs=0,
        **kwargs,
    ):
        del device, kwargs
        captured["max_docs"] = int(max_docs)
        return nob._empty_exact_sketch_direct_metrics()

    monkeypatch.setattr(nob, "_eval_fno_exact_sketch_direct_metrics", _fake_exact_metric)

    out = nob._train_fno_tree_single_stage(
        model=model,
        train_docs=fno_docs,
        val_docs=fno_docs[:2],
        device=torch.device("cpu"),
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        c1_weight=1.0,
        c2_weight=1.0,
        c3_weight=1.0,
        root_weight=1.0,
        checkpoint_metric="val_exact_sketch_direct",
        tree_batch_pack_mode="fixed_fused",
        tree_batch_autotune=True,
    )

    assert captured["max_docs"] == 2
    assert out["autotuned_batch_budgets"]["effective_exact_eval_max_docs"] == 2
    assert out["autotuned_batch_budgets"]["eval_bucket_max_docs_by_n_leaves"] == {
        "2": 2
    }


class TestModelSmoke:
    """Each model can forward-pass a tiny batch."""

    def test_mlp_bigram_forward(self):
        model = MLPBigramCountPredictor(input_dim=100, hidden_dim=32, n_count_classes=5)
        x = torch.randn(4, 100)
        out = model(x)
        assert out.shape == (4, 5)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_c2_score_drift_can_ignore_latent_replay_changes(self, monkeypatch):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=1,
            hidden_dim=4,
            target_scale=10.0,
            n_regimes=1,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )
        offset = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
        monkeypatch.setattr(model, "decode_summary", lambda state: state.clone())
        monkeypatch.setattr(model, "encode_summary", lambda summary: summary + offset)
        monkeypatch.setattr(model, "predict_norm_from_state", lambda state: state[..., 0])
        state = torch.tensor([0.25, 0.1, -0.1], dtype=torch.float32)

        base_norm, replay_norm, base_state, replay_state = _fno_summary_replay_tensors(
            model, state
        )

        assert float(F.mse_loss(replay_norm, base_norm).detach().cpu()) == pytest.approx(0.0)
        assert float(F.mse_loss(replay_state, base_state).detach().cpu()) > 0.0

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_c2_score_drift_is_positive_when_replay_changes_score(self, monkeypatch):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=1,
            hidden_dim=4,
            target_scale=10.0,
            n_regimes=1,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )
        score_shift = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32)
        monkeypatch.setattr(model, "decode_summary", lambda state: state.clone())
        monkeypatch.setattr(model, "encode_summary", lambda summary: summary + score_shift)
        monkeypatch.setattr(model, "predict_norm_from_state", lambda state: state[..., 0])
        state = torch.tensor([0.25, 0.1, -0.1], dtype=torch.float32)

        base_norm, replay_norm, base_state, replay_state = _fno_summary_replay_tensors(
            model, state
        )

        assert float(F.mse_loss(replay_norm, base_norm).detach().cpu()) > 0.0
        assert float(F.mse_loss(replay_state, base_state).detach().cpu()) > 0.0

    def test_cnn1d_forward(self):
        model = CNN1DCountPredictor(vocab_size=8, embed_dim=16, n_filters=32, n_count_classes=5)
        tokens = torch.randint(0, 8, (4, 16))
        mask = torch.ones(4, 16)
        out = model(tokens, token_mask=mask)
        assert out.shape == (4, 5)

    def test_deeponet_forward(self):
        model = DeepONetCountPredictor(
            vocab_size=8, embed_dim=16, hidden_dim=64, max_len=16, n_count_classes=5
        )
        tokens = torch.randint(0, 8, (4, 16))
        mask = torch.ones(4, 16)
        out = model(tokens, token_mask=mask)
        assert out.shape == (4, 5)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_forward(self):
        from src.ctreepo.sim.core.markov_neural_operator_baselines import FNOCountPredictor
        model = FNOCountPredictor(
            vocab_size=8, embed_dim=16, n_modes=4, width=16, n_layers=2, n_count_classes=5
        )
        tokens = torch.randint(0, 8, (4, 16))
        mask = torch.ones(4, 16)
        out = model(tokens, token_mask=mask)
        assert out.shape == (4, 5)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_count_ce_root_supervision_produces_discrete_counts(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=4,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=2,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            root_supervision_kind="count_ce",
            root_count_class_values=(0, 2, 5),
        )
        assert model.root_count_classifier is not None
        with torch.no_grad():
            model.root_count_classifier[0].weight.zero_()
            model.root_count_classifier[0].bias.zero_()
            model.root_count_classifier[2].weight.zero_()
            model.root_count_classifier[2].bias.copy_(
                torch.tensor([0.0, 1.5, -0.5], dtype=torch.float32)
            )
        state = torch.zeros(model.summary_dim, dtype=torch.float32)
        logits = model.predict_root_count_logits_from_state(state)
        pred = model.predict_canonical_count_from_state(state)

        assert logits.shape == (3,)
        assert pred.item() == pytest.approx(2.0)
        assert pred.item() in {0.0, 2.0, 5.0}

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_theorem_feature_batched_count_ce_uses_raw_count_targets(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=4,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=2,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            root_supervision_kind="count_ce",
            root_count_class_values=(0, 1, 2, 3, 4, 5),
        )
        with torch.no_grad():
            model.root_count_classifier[0].weight.zero_()
            model.root_count_classifier[0].bias.zero_()
            model.root_count_classifier[2].weight.zero_()
            model.root_count_classifier[2].bias.copy_(
                torch.tensor([0.0, 0.3, 1.1, -0.4, 0.2, 1.7], dtype=torch.float32)
            )
        state_batch = torch.zeros((2, model.summary_dim), dtype=torch.float32)
        truth_targets = torch.tensor([5.0, 2.0], dtype=torch.float32)

        out = nob._theorem_feature_task_supervision_terms_batched(
            model,
            state_batch,
            truth_targets=truth_targets,
        )

        logits = model.predict_root_count_logits_from_state(state_batch)
        expected = F.cross_entropy(
            logits,
            torch.tensor([5, 2], dtype=torch.long),
            reduction="none",
        )
        assert torch.allclose(out["task_loss"], expected)
        assert torch.allclose(out["total_loss"], expected)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_decoded_markov_sketch_surface_uses_latent_state(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=6,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            root_supervision_kind="mse",
            aligned_sketch_surface="decoded_markov_sketch",
        )
        state = model.encode_leaf_tokens(
            [0, 1, 2, 3, 4, 5, 6, 7],
            device=torch.device("cpu"),
        )
        summary = model.decode_summary(state)
        replay = model.encode_summary(summary)
        canonical = model.predict_canonical_count_from_state(state)
        decoded = model.predict_count_from_state(state)

        assert state.shape == (model.state_dim,)
        assert model.summary_dim == 1 + 2 * model.n_regimes
        assert summary.shape == (model.summary_dim,)
        assert replay.shape == (model.state_dim,)
        assert torch.allclose(canonical, decoded)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_slot_summary_spec_requires_divisible_state_dim(self):
        with pytest.raises(ValueError, match="divisible by slot_count"):
            FNOCountSketch(
                vocab_size=8,
                leaf_tokens=8,
                state_dim=10,
                hidden_dim=16,
                target_scale=10.0,
                n_regimes=3,
                fno_width=8,
                fno_n_modes=4,
                fno_n_layers=1,
                summary_spec_name="markov_count_sketch",
                slot_count=4,
            )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_slot_summary_spec_uses_latent_only_merger_input(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
        )
        assert model.merger is None
        assert model.count_slot_merger is not None
        assert model.count_slot_merger[0].in_features == 4 * model.slot_dim
        assert model.residual_slot_merger is not None
        assert model.residual_slot_merger[0].in_features == 2 * model.residual_dim
        state = model.encode_leaf_tokens(
            [0, 1, 2, 3, 4, 5, 6, 7],
            device=torch.device("cpu"),
        )
        summary = model.decode_summary(state)
        replay = model.encode_summary(summary)

        assert state.shape == (model.state_dim,)
        assert model.summary_dim == 1 + 2 * model.n_regimes
        assert summary.shape == (model.summary_dim,)
        assert replay.shape == (model.state_dim,)
        assert model.codec_contract is not None
        assert torch.allclose(
            model.predict_canonical_count_from_state(state),
            model.predict_task_count_from_state(state),
        )
        assert not torch.allclose(
            model.predict_canonical_count_from_state(state),
            model.predict_count_from_state(state),
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_slot_summary_spec_theorem_primary_uses_theorem_root_path(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            summary_spec_root_mode="theorem_primary",
        )
        state = model.encode_leaf_tokens(
            [0, 1, 2, 3, 4, 5, 6, 7],
            device=torch.device("cpu"),
        )

        assert model.uses_theorem_primary_root_mode() is True
        assert torch.allclose(
            model.predict_canonical_count_from_state(state),
            model.predict_count_from_state(state),
        )
        assert not torch.allclose(
            model.predict_canonical_count_from_state(state),
            model.predict_task_count_from_state(state),
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_factored_theorem_readout_uses_theorem_feature_only(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
        )
        assert model.theorem_feature_readout is not None
        with torch.no_grad():
            model.theorem_feature_readout[0].weight.fill_(1.0)
            model.theorem_feature_readout[0].bias.zero_()
            model.theorem_feature_readout[2].weight.fill_(1.0)
            model.theorem_feature_readout[2].bias.zero_()
        count_slot = torch.tensor([0.2, -0.3], dtype=torch.float32)
        first_slot = torch.tensor([0.4, 0.5], dtype=torch.float32)
        last_slot = torch.tensor([-0.1, 0.7], dtype=torch.float32)
        residual_a = torch.tensor([0.0, 0.0], dtype=torch.float32)
        residual_b = torch.tensor([9.0, -9.0], dtype=torch.float32)
        state_a = model._pack_summary_spec_state(
            count_slot, first_slot, last_slot, residual_a
        )
        state_b = model._pack_summary_spec_state(
            count_slot, first_slot, last_slot, residual_b
        )

        task_a = model.predict_task_count_from_state(state_a)
        task_b = model.predict_task_count_from_state(state_b)

        assert model.uses_factored_theorem_readout_root_mode() is True
        assert torch.allclose(task_a, task_b)
        assert torch.allclose(
            model.predict_canonical_count_from_state(state_a),
            model.predict_task_count_from_state(state_a),
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_factored_theorem_readout_rejects_learned_projection(self):
        with pytest.raises(ValueError, match="does not allow"):
            FNOCountSketch(
                vocab_size=8,
                leaf_tokens=8,
                state_dim=8,
                hidden_dim=8,
                target_scale=10.0,
                n_regimes=3,
                fno_width=8,
                fno_n_modes=4,
                fno_n_layers=1,
                summary_spec_name="markov_count_sketch",
                slot_count=4,
                task_head_mode="theorem_feature_scalar",
                summary_spec_root_mode="factored_theorem_readout",
                theorem_surface_mode="learned_projection",
            )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_shared_feature_factored_readout_supports_non_markov_adapter_without_summary_spec(
        self,
    ):
        class _ToyBandAdapter:
            name = "toy_scalar_band_test"
            has_canonical_decode = False

            @staticmethod
            def oracle_label(*, count, first=None, last=None, metadata=None):
                return "low" if float(count) < 1.0 else "high"

            @staticmethod
            def same_pair(left, right, *, same_threshold=None, diff_threshold=None):
                return left == right

            @staticmethod
            def different_pair(left, right, *, same_threshold=None, diff_threshold=None):
                return left != right

            @staticmethod
            def diagnostic_key(label):
                return str(label)

            @staticmethod
            def task_readout_target(label):
                return 0.0 if label == "low" else 1.0

            @staticmethod
            def decode_from_phi(phi):
                return None

        register_theorem_feature_adapter(
            "toy_scalar_band_test",
            lambda: _ToyBandAdapter(),
            overwrite=True,
        )
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=4,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=2,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature",
            theorem_feature_adapter="toy_scalar_band_test",
            theorem_feature_dim=2,
            theorem_feature_hidden_dim=4,
        )
        phi_proj = nn.Linear(model.summary_dim, 2, bias=False)
        with torch.no_grad():
            phi_proj.weight.zero_()
            phi_proj.weight[0, 0] = 1.0
            phi_proj.weight[1, 1] = 1.0
        model.phi_projector = phi_proj
        assert model.theorem_feature_readout is not None
        with torch.no_grad():
            model.theorem_feature_readout[0].weight.fill_(1.0)
            model.theorem_feature_readout[0].bias.zero_()
            model.theorem_feature_readout[2].weight.fill_(1.0)
            model.theorem_feature_readout[2].bias.zero_()
        state_a = torch.tensor([0.2, -0.3, 0.0, 0.0, 3.0, -3.0, 1.0, -1.0])
        state_b = torch.tensor([0.2, -0.3, 9.0, -9.0, 0.0, 0.0, 5.0, -5.0])

        phi_a = model.theorem_feature_from_state(state_a)
        phi_b = model.theorem_feature_from_state(state_b)
        task_a = model.predict_task_count_from_state(state_a)
        task_b = model.predict_task_count_from_state(state_b)

        assert model.use_summary_spec is False
        assert model.uses_factored_theorem_readout_root_mode() is True
        assert torch.allclose(phi_a, phi_b)
        assert torch.allclose(task_a, task_b)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_shared_bottleneck_factored_readout_uses_phi_only(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_bottleneck",
            theorem_feature_dim=2,
            theorem_feature_hidden_dim=4,
        )
        phi_proj = nn.Linear(8, 2, bias=False)
        with torch.no_grad():
            phi_proj.weight.zero_()
            phi_proj.weight[0, 0] = 1.0
            phi_proj.weight[1, 1] = 1.0
        model.phi_projector = phi_proj
        assert model.theorem_feature_readout is not None
        with torch.no_grad():
            model.theorem_feature_readout[0].weight.fill_(1.0)
            model.theorem_feature_readout[0].bias.zero_()
            model.theorem_feature_readout[2].weight.fill_(1.0)
            model.theorem_feature_readout[2].bias.zero_()
        model.summary_decode_trunk = nn.Identity()
        model.summary_count_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.summary_count_head.weight.fill_(1.0)
        first_head = nn.Linear(2, 3, bias=False)
        last_head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            first_head.weight.zero_()
            last_head.weight.zero_()
        model.first_endpoint_proj = first_head
        model.last_endpoint_proj = last_head
        state_a = torch.tensor([0.2, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_b = torch.tensor([0.2, -0.3, 9.0, -9.0, 5.0, -5.0, 1.0, -1.0])

        phi_a = model.theorem_feature_from_state(state_a)
        phi_b = model.theorem_feature_from_state(state_b)
        task_a = model.predict_task_count_from_state(state_a)
        task_b = model.predict_task_count_from_state(state_b)
        theorem_a = model.predict_count_from_state(state_a)
        theorem_b = model.predict_count_from_state(state_b)

        assert torch.allclose(phi_a, phi_b)
        assert torch.allclose(task_a, task_b)
        assert torch.allclose(theorem_a, theorem_b)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_shared_bottleneck_phi_merge_predictor_shape(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            theorem_surface_mode="shared_bottleneck",
            theorem_feature_dim=3,
            theorem_feature_hidden_dim=6,
        )
        left = torch.randn(8)
        right = torch.randn(8)
        pred = model.predict_phi_parent_from_children(left, right)
        assert pred.shape == (3,)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_encode_leaf_tokens_batch_matches_single_encoding(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )
        leaves = ((0, 1, 2), (3, 4), (5,))

        batch_states = model.encode_leaf_tokens_batch(leaves, device=torch.device("cpu"))
        single_states = torch.stack(
            [model.encode_leaf_tokens(leaf, device=torch.device("cpu")) for leaf in leaves],
            dim=0,
        )

        assert torch.allclose(batch_states, single_states, atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_merge_states_balanced_batched_matches_list_path(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
        )
        leaves = ((0, 1, 2), (3, 4), (5, 6), (7,))
        state_batch = model.encode_leaf_tokens_batch(leaves, device=torch.device("cpu"))
        state_list = [state_batch[idx] for idx in range(int(state_batch.shape[0]))]

        root_from_batch, merges_from_batch = model._merge_states(
            state_batch,
            schedule="balanced",
            collect_merge_states=True,
        )
        root_from_list, merges_from_list = model._merge_states(
            state_list,
            schedule="balanced",
            collect_merge_states=True,
        )

        assert torch.allclose(root_from_batch, root_from_list, atol=1e-6, rtol=1e-6)
        assert len(merges_from_batch) == len(merges_from_list)
        for left, right in zip(merges_from_batch, merges_from_list):
            assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_shared_feature_factored_readout_uses_phi_only(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature",
            theorem_feature_dim=2,
            theorem_feature_hidden_dim=4,
        )
        phi_proj = nn.Linear(8, 2, bias=False)
        with torch.no_grad():
            phi_proj.weight.zero_()
            phi_proj.weight[0, 0] = 1.0
            phi_proj.weight[1, 1] = 1.0
        model.phi_projector = phi_proj
        assert model.theorem_feature_readout is not None
        with torch.no_grad():
            model.theorem_feature_readout[0].weight.fill_(1.0)
            model.theorem_feature_readout[0].bias.zero_()
            model.theorem_feature_readout[2].weight.fill_(1.0)
            model.theorem_feature_readout[2].bias.zero_()
        model.summary_decode_trunk = nn.Identity()
        model.summary_count_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.summary_count_head.weight.fill_(1.0)
        first_head = nn.Linear(2, 3, bias=False)
        last_head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            first_head.weight.zero_()
            last_head.weight.zero_()
        model.first_endpoint_proj = first_head
        model.last_endpoint_proj = last_head
        state_a = torch.tensor([0.2, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_b = torch.tensor([0.2, -0.3, 9.0, -9.0, 5.0, -5.0, 1.0, -1.0])

        phi_a = model.theorem_feature_from_state(state_a)
        phi_b = model.theorem_feature_from_state(state_b)
        task_a = model.predict_task_count_from_state(state_a)
        task_b = model.predict_task_count_from_state(state_b)
        theorem_a = model.predict_count_from_state(state_a)
        theorem_b = model.predict_count_from_state(state_b)

        assert torch.allclose(phi_a, phi_b)
        assert torch.allclose(task_a, task_b)
        assert torch.allclose(theorem_a, theorem_b)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_shared_feature_adapters_factored_readout_uses_phi_only(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature_adapters",
            theorem_feature_dim=4,
            theorem_feature_hidden_dim=4,
        )
        phi_proj = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            phi_proj.weight.zero_()
            phi_proj.weight[0, 0] = 1.0
            phi_proj.weight[1, 1] = 1.0
            phi_proj.weight[2, 2] = 1.0
            phi_proj.weight[3, 3] = 1.0
        model.phi_projector = phi_proj
        adapter = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            adapter.weight.zero_()
            adapter.weight[0, 0] = 1.0
            adapter.weight[1, 1] = 1.0
        model.count_phi_adapter = adapter
        model.first_phi_adapter = adapter
        model.last_phi_adapter = adapter
        model.root_phi_adapter = adapter
        model.join_phi_adapter = adapter
        assert model.theorem_feature_readout is not None
        with torch.no_grad():
            model.theorem_feature_readout[0].weight.fill_(1.0)
            model.theorem_feature_readout[0].bias.zero_()
            model.theorem_feature_readout[2].weight.fill_(1.0)
            model.theorem_feature_readout[2].bias.zero_()
        model.summary_decode_trunk = nn.Identity()
        model.summary_count_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.summary_count_head.weight.fill_(1.0)
        first_head = nn.Linear(2, 3, bias=False)
        last_head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            first_head.weight.zero_()
            last_head.weight.zero_()
        model.first_endpoint_proj = first_head
        model.last_endpoint_proj = last_head
        state_a = torch.tensor([0.2, -0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0])
        state_b = torch.tensor([0.2, -0.3, 0.4, 0.5, 9.0, -9.0, 5.0, -5.0])

        phi_a = model.theorem_feature_from_state(state_a)
        phi_b = model.theorem_feature_from_state(state_b)
        task_a = model.predict_task_count_from_state(state_a)
        task_b = model.predict_task_count_from_state(state_b)
        theorem_a = model.predict_count_from_state(state_a)
        theorem_b = model.predict_count_from_state(state_b)

        assert torch.allclose(phi_a, phi_b)
        assert torch.allclose(task_a, task_b)
        assert torch.allclose(theorem_a, theorem_b)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_factorized_score_fiber_task_head_uses_score_slice_only(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="factorized_score_fiber",
            theorem_feature_adapter="markov_score_endpoints",
            theorem_feature_dim=16,
            theorem_feature_hidden_dim=16,
            theorem_score_dim=1,
            theorem_fiber_dim=15,
            theorem_aux_dim=0,
        )
        phi_proj = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            phi_proj.weight.copy_(torch.eye(16))
        model.phi_projector = phi_proj
        model.summary_decode_trunk = nn.Identity()

        state_a = torch.tensor([0.2] + [0.0] * 15, dtype=torch.float32)
        state_b = torch.tensor([0.2] + [float(i) for i in range(1, 16)], dtype=torch.float32)

        task_a = model.predict_task_count_from_state(state_a)
        task_b = model.predict_task_count_from_state(state_b)
        count_hidden_a = model._count_hidden_from_state(state_a)
        count_hidden_b = model._count_hidden_from_state(state_b)

        assert model.theorem_feature_readout is None
        assert torch.allclose(task_a, task_b)
        assert not torch.allclose(count_hidden_a, count_hidden_b)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_factorized_score_fiber_phi_projection_ignores_score_slice(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            theorem_surface_mode="factorized_score_fiber",
            theorem_feature_adapter="markov_score_endpoints",
            theorem_feature_dim=16,
            theorem_feature_hidden_dim=16,
            theorem_score_dim=1,
            theorem_fiber_dim=15,
            theorem_aux_dim=0,
        )
        phi_proj = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            phi_proj.weight.copy_(torch.eye(16))
        model.phi_projector = phi_proj

        state_a = torch.tensor([0.1] + [float(i) for i in range(1, 16)], dtype=torch.float32)
        state_b = torch.tensor([0.9] + [float(i) for i in range(1, 16)], dtype=torch.float32)

        phi_a = model.predict_phi_from_state(state_a)
        phi_b = model.predict_phi_from_state(state_b)

        assert phi_a.shape == (15,)
        assert torch.allclose(phi_a, phi_b)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_factorized_score_fiber_score_merge_is_gated_affine(self):
        class _FixedScoreMerge(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "params",
                    torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shape = tuple(x.shape[:-1]) + (3,)
                return self.params.to(device=x.device, dtype=x.dtype).expand(shape)

        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=4,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=2,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            theorem_surface_mode="factorized_score_fiber",
            theorem_feature_adapter="markov_score_endpoints",
            theorem_feature_dim=4,
            theorem_feature_hidden_dim=8,
            theorem_score_dim=1,
            theorem_fiber_dim=3,
            theorem_aux_dim=0,
        )
        phi_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            phi_proj.weight.copy_(torch.eye(4))
        model.phi_projector = phi_proj
        model.score_merge_predictor = _FixedScoreMerge()

        left_state = torch.tensor([1.5, 0.1, 0.2, 0.3], dtype=torch.float32)
        right_state = torch.tensor([2.0, -0.1, -0.2, -0.3], dtype=torch.float32)

        pred = model.predict_score_parent_from_children(left_state, right_state)
        gate = float(F.softplus(torch.tensor(0.0)))
        expected = gate * 1.5 + gate * 2.0 + 0.5

        assert pred.shape == (1,)
        assert float(pred.detach().cpu().squeeze(0)) == pytest.approx(expected)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_shared_feature_local_supervision_bypasses_summary_spec_terms(
        self,
        monkeypatch,
    ):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=2,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature",
            theorem_feature_dim=3,
            theorem_feature_hidden_dim=4,
        )

        def _fail_summary_spec(*args, **kwargs):
            raise AssertionError("shared-feature local supervision should not use summary-spec replay")

        monkeypatch.setattr(nob, "_summary_spec_supervision_terms", _fail_summary_spec)

        out = model.forward_doc(
            leaf_token_ids=((0,), (1,)),
            leaf_counts=(0.0, 0.0),
            merge_counts_balanced=(1.0,),
            schedule="balanced",
            collect_leaf=True,
            collect_c3=True,
            collect_c2=True,
            device=torch.device("cpu"),
            leaf_first_regimes=(0, 1),
            leaf_last_regimes=(0, 1),
            internal_supervision_kind="full_sketch",
            leaf_exact_supervision=True,
            leaf_supervision_kind="full_sketch",
        )

        assert torch.isfinite(out["leaf_loss"])
        assert torch.isfinite(out["c3_loss"])
        assert torch.isfinite(out["c2_loss"])
        assert float(out["leaf_count"]) == pytest.approx(2.0)
        assert float(out["c3_count"]) == pytest.approx(1.0)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_tree_fno_forward_doc_precomputed_states_matches_direct(self):
        raw_docs = _make_tiny_docs(n=2, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="factorized_score_fiber",
            theorem_feature_adapter="markov_score_endpoints",
            theorem_feature_dim=16,
            theorem_feature_hidden_dim=32,
            theorem_score_dim=1,
            theorem_fiber_dim=15,
            theorem_aux_dim=0,
            score_merge_mode="gated_affine",
        )

        views = nob._precompute_balanced_doc_state_views(
            model,
            fno_docs,
            device=torch.device("cpu"),
            collect_merge_states=True,
        )
        assert len(views) == len(fno_docs)

        for doc, view in zip(fno_docs, views):
            direct = model.forward_doc(
                doc.leaf_token_ids,
                doc.leaf_counts,
                doc.merge_counts_balanced,
                doc.merge_token_lengths,
                schedule="balanced",
                collect_leaf=True,
                collect_c3=True,
                collect_c2=True,
                device=torch.device("cpu"),
                leaf_first_regimes=doc.leaf_first_regimes,
                leaf_last_regimes=doc.leaf_last_regimes,
                internal_supervision_kind="full_sketch",
                leaf_exact_supervision=True,
                leaf_supervision_kind="full_sketch",
            )
            precomputed = model.forward_doc(
                doc.leaf_token_ids,
                doc.leaf_counts,
                doc.merge_counts_balanced,
                doc.merge_token_lengths,
                schedule="balanced",
                collect_leaf=True,
                collect_c3=True,
                collect_c2=True,
                device=torch.device("cpu"),
                leaf_first_regimes=doc.leaf_first_regimes,
                leaf_last_regimes=doc.leaf_last_regimes,
                internal_supervision_kind="full_sketch",
                leaf_exact_supervision=True,
                leaf_supervision_kind="full_sketch",
                precomputed_state_batch=view.state_batch,
                precomputed_root_state=view.root_state,
                precomputed_merge_states=view.merge_states,
            )

            assert torch.allclose(
                direct["root_state"],
                precomputed["root_state"],
                atol=1e-6,
                rtol=1e-6,
            )
            for key in (
                "pred_norm",
                "pred_count",
                "pred_task_count",
                "pred_root_count_canonical",
                "leaf_loss",
                "c2_loss",
                "c3_loss",
                "phi_compose_loss",
                "phi_contrastive_loss",
            ):
                assert torch.allclose(
                    direct[key],
                    precomputed[key],
                    atol=1e-6,
                    rtol=1e-6,
                )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_summary_spec_explicit_theorem_subspace_leaf_encoding_uses_boundary_features(
        self, monkeypatch
    ):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=4,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=2,
            fno_width=2,
            fno_n_modes=2,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            theorem_count_dim=2,
            theorem_first_dim=2,
            theorem_last_dim=2,
        )
        x = torch.tensor(
            [[[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]]],
            dtype=torch.float32,
        )
        pooled = torch.tensor([[25.0, 26.0]], dtype=torch.float32)
        monkeypatch.setattr(
            model,
            "_encode_token_batch",
            lambda tokens, token_mask: (x, pooled),
        )
        model.summary_count_leaf_proj = torch.nn.Identity()
        model.summary_first_leaf_proj = torch.nn.Identity()
        model.summary_last_leaf_proj = torch.nn.Identity()
        model.summary_residual_leaf_proj = torch.nn.Identity()

        state = model.encode_leaf_tokens([0, 1, 2], device=torch.device("cpu"))

        assert torch.allclose(model._count_slot(state), pooled.squeeze(0))
        assert torch.allclose(model._first_slot(state), torch.tensor([10.0, 11.0]))
        assert torch.allclose(model._last_slot(state), torch.tensor([30.0, 31.0]))
        assert torch.allclose(model._residual_slots_flat(state), pooled.squeeze(0))

    def test_prototype_classifier_picks_matching_prototype(self):
        head = PrototypeClassifier(input_dim=2, n_classes=2)
        with torch.no_grad():
            head.prototypes.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            )
            head.log_temperature.fill_(np.log(1e-3))
        logits = head(torch.tensor([1.0, 0.0], dtype=torch.float32))
        assert int(torch.argmax(logits).item()) == 0

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_summary_spec_support_classifier_decodes_count_support_exactly(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=4,
            state_dim=8,
            hidden_dim=2,
            target_scale=1.0,
            n_regimes=2,
            fno_width=2,
            fno_n_modes=2,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            theorem_count_head_mode="support_classifier",
            theorem_count_dim=2,
            theorem_first_dim=2,
            theorem_last_dim=2,
        )
        model.summary_decode_trunk = torch.nn.Identity()
        assert model.summary_count_classifier is not None
        with torch.no_grad():
            model.summary_count_classifier.prototypes.copy_(
                torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
            )
            model.summary_count_classifier.log_temperature.fill_(np.log(1e-3))
        zero_state = model._pack_summary_spec_state(
            torch.tensor([1.0, 0.0]),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )
        one_state = model._pack_summary_spec_state(
            torch.tensor([-1.0, 0.0]),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

        assert float(model.predict_count_from_state(zero_state).detach().cpu()) == pytest.approx(
            0.0,
            abs=1e-6,
        )
        assert float(model.predict_count_from_state(one_state).detach().cpu()) == pytest.approx(
            1.0,
            abs=1e-6,
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_summary_spec_hybrid_ordinal_decodes_ordered_count_without_using_aux_head(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=4,
            state_dim=8,
            hidden_dim=2,
            target_scale=2.0,
            n_regimes=2,
            fno_width=2,
            fno_n_modes=2,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            theorem_count_head_mode="hybrid_ordinal",
            theorem_count_dim=2,
            theorem_first_dim=2,
            theorem_last_dim=2,
            summary_spec_root_mode="theorem_primary",
        )
        model.summary_decode_trunk = torch.nn.Identity()
        assert model.summary_count_ordinal_head is not None
        assert model.summary_count_scalar_aux_head is not None
        with torch.no_grad():
            model.summary_count_ordinal_head.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
            )
            model.summary_count_ordinal_head.bias.zero_()
            model.summary_count_scalar_aux_head.weight.zero_()
            model.summary_count_scalar_aux_head.bias.fill_(10.0)

        zero_state = model._pack_summary_spec_state(
            torch.tensor([-10.0, -10.0]),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )
        one_state = model._pack_summary_spec_state(
            torch.tensor([10.0, -10.0]),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )
        two_state = model._pack_summary_spec_state(
            torch.tensor([10.0, 10.0]),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
        )

        assert float(model.predict_count_from_state(zero_state).detach().cpu()) == pytest.approx(
            0.0,
            abs=1e-4,
        )
        assert float(model.predict_count_from_state(one_state).detach().cpu()) == pytest.approx(
            1.0,
            abs=1e-4,
        )
        assert float(model.predict_count_from_state(two_state).detach().cpu()) == pytest.approx(
            2.0,
            abs=1e-4,
        )
        assert float(
            model.predict_count_scalar_aux_from_state(one_state).detach().cpu()
        ) == pytest.approx(2.0, abs=1e-4)
        assert float(
            model.predict_canonical_count_from_state(one_state).detach().cpu()
        ) == pytest.approx(1.0, abs=1e-4)

    def test_theorem_count_threshold_pos_weights_balance_leaf_merge_and_root_targets(self):
        doc = nob._FNOCountDoc(
            n_tokens=2,
            leaf_token_ids=((0,), (1,)),
            leaf_counts=(0.0, 1.0),
            leaf_first_regimes=(0, 0),
            leaf_last_regimes=(0, 0),
            leaf_token_lengths=(1, 1),
            merge_counts_balanced=(1.0,),
            merge_sizes_balanced=(2,),
            merge_token_lengths=(2,),
            root_count=2.0,
        )

        weights = _theorem_count_threshold_pos_weights_from_docs((doc,), max_count=2)

        assert weights.tolist() == pytest.approx([1.0 / 3.0, 3.0], abs=1e-6)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_markov_codec_contract_matches_exact_join_and_compose(self):
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=10.0,
            n_regimes=3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
        )
        assert model.codec_contract is not None
        model.encode_summary = lambda summary: summary  # type: ignore[method-assign]
        model.decode_markov_codec = lambda state: DecodedMarkovSketch(  # type: ignore[method-assign]
            count=torch.round(state[..., 0] * float(model.target_scale)),
            first=torch.argmax(state[..., 1 : 1 + model.n_regimes], dim=-1),
            last=torch.argmax(state[..., 1 + model.n_regimes :], dim=-1),
        )
        left = DecodedMarkovSketch(
            count=torch.tensor(2.0),
            first=torch.tensor(1),
            last=torch.tensor(1),
        )
        right = DecodedMarkovSketch(
            count=torch.tensor(3.0),
            first=torch.tensor(2),
            last=torch.tensor(0),
        )

        join = model.codec_contract.join(left.last, right.first)
        parent = model.codec_contract.compose(left, right)
        replay = model.codec_contract.reencode(parent)
        replay_decoded = model.codec_contract.decode(replay)

        assert float(join.detach().cpu()) == pytest.approx(1.0)
        assert float(parent.count.detach().cpu()) == pytest.approx(6.0)
        assert int(parent.first.detach().cpu()) == 1
        assert int(parent.last.detach().cpu()) == 0
        assert int(replay_decoded.first.detach().cpu()) == 1
        assert int(replay_decoded.last.detach().cpu()) == 0

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_baseline_exposes_root_summary_probe_audit(self):
        docs = _make_tiny_docs(n=12, seq_len=16, vocab_size=8)
        config = OPSCountConfig(
            n_epochs=1,
            state_dim=16,
            hidden_dim=32,
            batch_size=4,
            lr=1e-3,
            vocab_size=8,
            n_regimes=4,
        )
        fit = _fit_fno_baseline_with_predictions(
            config=config,
            seeds={"effective_model_seed": 0},
            device=torch.device("cpu"),
            train_docs=docs[:8],
            val_docs=docs[8:10],
            test_docs=docs[10:12],
        )
        audit = dict(fit.get("root_summary_probe_audit") or {})

        assert set(audit) == {"train", "val", "test"}
        assert "count_mae" in audit["test"]
        assert "first_accuracy" in audit["test"]
        assert "last_accuracy" in audit["test"]
        assert "exact_summary_match_rate" in audit["test"]

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_probe_encoding_respects_batch_size(self, monkeypatch):
        docs = _make_tiny_docs(n=16, seq_len=16, vocab_size=8)
        config = OPSCountConfig(
            n_epochs=1,
            state_dim=16,
            hidden_dim=32,
            batch_size=2,
            lr=1e-3,
            vocab_size=8,
            n_regimes=4,
        )
        original_encode = fdb.FNOCountPredictor.encode_representation

        def _guard_encode(self, tokens, *, token_mask):
            assert int(tokens.shape[0]) <= int(config.batch_size)
            return original_encode(self, tokens, token_mask=token_mask)

        monkeypatch.setattr(
            fdb.FNOCountPredictor,
            "encode_representation",
            _guard_encode,
        )

        fit = _fit_fno_baseline_with_predictions(
            config=config,
            seeds={"effective_model_seed": 0},
            device=torch.device("cpu"),
            train_docs=docs[:8],
            val_docs=docs[8:12],
            test_docs=docs[12:16],
            train_eval_docs=docs[:12],
        )

        assert dict(fit.get("root_summary_probe_audit") or {})

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_baseline_uses_shared_flat_trainer_and_preserves_train_eval_docs(self):
        docs = _make_tiny_docs(n=12, seq_len=16, vocab_size=8)
        config = OPSCountConfig(
            n_epochs=1,
            state_dim=16,
            hidden_dim=32,
            batch_size=4,
            lr=1e-3,
            vocab_size=8,
            n_regimes=4,
            gpu_runtime_data_mode="cpu_debug",
        )
        fit = _fit_fno_baseline_with_predictions(
            config=config,
            seeds={"effective_model_seed": 0},
            device=torch.device("cpu"),
            train_docs=docs[:4],
            val_docs=docs[8:10],
            test_docs=docs[10:12],
            train_eval_docs=docs[:6],
        )

        assert int(fit.get("train_docs_used", 0)) == 4
        assert np.asarray(fit.get("train_preds")).shape[0] == 6
        assert np.asarray(fit.get("train_truths")).shape[0] == 6
        actual_config = dict(fit.get("baseline_fno_actual_config") or {})
        assert actual_config.get("training_path") == "shared_flat_baseline_trainer"
        assert dict(actual_config.get("runtime_config") or {}).get("data_mode") == "cpu_debug"
        runtime_efficiency = dict(fit.get("runtime_efficiency") or {})
        assert runtime_efficiency.get("runtime_data_mode") == "cpu_debug"
        assert dict(runtime_efficiency.get("cpu_fallback_reason_counts") or {})

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_baseline_rejects_out_of_range_token_ids(self):
        docs = list(_make_tiny_docs(n=12, seq_len=16, vocab_size=8))
        bad_doc = replace(docs[0], tokens=tuple([99] + list(docs[0].tokens[1:])))
        docs[0] = bad_doc
        config = OPSCountConfig(
            n_epochs=1,
            state_dim=16,
            hidden_dim=32,
            batch_size=4,
            lr=1e-3,
            vocab_size=8,
            n_regimes=4,
        )

        with pytest.raises(ValueError, match="outside \\[0, 8\\)"):
            _fit_fno_baseline_with_predictions(
                config=config,
                seeds={"effective_model_seed": 0},
                device=torch.device("cpu"),
                train_docs=tuple(docs[:8]),
                val_docs=tuple(docs[8:10]),
                test_docs=tuple(docs[10:12]),
            )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_summary_spec_leaf_supervision_kind_controls_endpoint_losses(self):
        docs = _make_tiny_docs(n=1, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=16.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
        )

        count_only = model.forward_doc(
            fno_docs[0].leaf_token_ids,
            fno_docs[0].leaf_counts,
            fno_docs[0].merge_counts_balanced,
            schedule="balanced",
            collect_leaf=True,
            collect_c3=False,
            collect_c2=False,
            device=torch.device("cpu"),
            leaf_first_regimes=fno_docs[0].leaf_first_regimes,
            leaf_last_regimes=fno_docs[0].leaf_last_regimes,
            leaf_supervision_kind="count_only",
        )
        full_sketch = model.forward_doc(
            fno_docs[0].leaf_token_ids,
            fno_docs[0].leaf_counts,
            fno_docs[0].merge_counts_balanced,
            schedule="balanced",
            collect_leaf=True,
            collect_c3=False,
            collect_c2=False,
            device=torch.device("cpu"),
            leaf_first_regimes=fno_docs[0].leaf_first_regimes,
            leaf_last_regimes=fno_docs[0].leaf_last_regimes,
            leaf_supervision_kind="full_sketch",
        )

        count_only_components = dict(count_only.get("loss_components") or {})
        full_sketch_components = dict(full_sketch.get("loss_components") or {})
        assert float(count_only_components["leaf_first_loss"].detach().cpu()) == pytest.approx(0.0)
        assert float(count_only_components["leaf_last_loss"].detach().cpu()) == pytest.approx(0.0)
        assert float(full_sketch_components["leaf_first_loss"].detach().cpu()) >= 0.0
        assert float(full_sketch_components["leaf_last_loss"].detach().cpu()) >= 0.0

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_exact_sketch_direct_metrics_uses_adapter_task_target_for_shared_feature(
        self,
        monkeypatch,
    ):
        class _ConstantTaskAdapter:
            name = "constant_task_target_test"
            has_canonical_decode = False

            @staticmethod
            def oracle_label(*, count, first=None, last=None, metadata=None):
                return (float(count), int(first or 0), int(last or 0))

            @staticmethod
            def same_pair(left, right, *, same_threshold=None, diff_threshold=None):
                return left == right

            @staticmethod
            def different_pair(left, right, *, same_threshold=None, diff_threshold=None):
                return left != right

            @staticmethod
            def diagnostic_key(label):
                return label

            @staticmethod
            def task_readout_target(label):
                return 3.5

            @staticmethod
            def decode_from_phi(phi):
                return None

        register_theorem_feature_adapter(
            "constant_task_target_test",
            lambda: _ConstantTaskAdapter(),
            overwrite=True,
        )
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=8,
            target_scale=10.0,
            n_regimes=2,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature",
            theorem_feature_adapter="constant_task_target_test",
            theorem_feature_dim=2,
            theorem_feature_hidden_dim=4,
        )
        root_state = torch.zeros(model.summary_dim, dtype=torch.float32)
        monkeypatch.setattr(
            model,
            "encode_leaf_tokens",
            lambda token_ids, device=None: root_state.clone(),
        )
        monkeypatch.setattr(
            model,
            "_merge_states",
            lambda states, schedule, collect_merge_states=False: (
                states[0],
                [] if collect_merge_states else states[0],
            ),
        )
        monkeypatch.setattr(
            model,
            "predict_task_count_from_state",
            lambda state: torch.tensor(3.5, dtype=state.dtype, device=state.device),
        )
        monkeypatch.setattr(
            model,
            "predict_count_from_state",
            lambda state: torch.tensor(0.0, dtype=state.dtype, device=state.device),
        )
        monkeypatch.setattr(model, "predict_phi_from_state", lambda state: state[:2])

        doc = nob._FNOCountDoc(
            n_tokens=1,
            leaf_token_ids=((0,),),
            leaf_counts=(0.0,),
            leaf_first_regimes=(0,),
            leaf_last_regimes=(0,),
            leaf_token_lengths=(1,),
            merge_counts_balanced=tuple(),
            merge_sizes_balanced=tuple(),
            merge_token_lengths=tuple(),
            root_count=0.0,
        )

        metrics = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            [doc],
            device=torch.device("cpu"),
        )

        assert metrics["task_root_mae"] == pytest.approx(0.0)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_model_batched_matches_legacy(self):
        raw_docs = _make_tiny_docs(n=3, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )

        legacy = nob._eval_fno_model_legacy(
            model,
            fno_docs,
            device=torch.device("cpu"),
            tau=0.5,
        )
        batched = nob._eval_fno_model(
            model,
            fno_docs,
            device=torch.device("cpu"),
            tau=0.5,
        )

        assert batched.root_mae == pytest.approx(legacy.root_mae, abs=1e-6)
        assert batched.root_median_abs_error == pytest.approx(
            legacy.root_median_abs_error,
            abs=1e-6,
        )
        assert batched.root_p95_abs_error == pytest.approx(
            legacy.root_p95_abs_error,
            abs=1e-6,
        )
        assert batched.schedule_spread_mean == pytest.approx(
            legacy.schedule_spread_mean,
            abs=1e-6,
        )
        assert batched.schedule_spread_p95 == pytest.approx(
            legacy.schedule_spread_p95,
            abs=1e-6,
        )
        assert batched.leaf_mae == pytest.approx(legacy.leaf_mae, abs=1e-6)
        assert batched.leaf_violation_rate == pytest.approx(
            legacy.leaf_violation_rate,
            abs=1e-6,
        )
        assert batched.c2_idempotence_mae == pytest.approx(
            legacy.c2_idempotence_mae,
            abs=1e-6,
        )
        assert batched.c2_r2_mae == pytest.approx(legacy.c2_r2_mae, abs=1e-6)
        assert batched.c2_r4_mae == pytest.approx(legacy.c2_r4_mae, abs=1e-6)
        assert batched.merge_mae == pytest.approx(legacy.merge_mae, abs=1e-6)
        assert batched.merge_violation_rate == pytest.approx(
            legacy.merge_violation_rate,
            abs=1e-6,
        )
        assert batched.c2_state_replay_mse == pytest.approx(
            legacy.c2_state_replay_mse,
            abs=1e-6,
        )
        assert batched.n_docs == legacy.n_docs

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_root_only_metrics_matches_full_root_mae(self):
        raw_docs = _make_tiny_docs(n=3, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )

        full_metrics = nob._eval_fno_model(
            model,
            fno_docs,
            device=torch.device("cpu"),
            tau=0.5,
        )
        root_only_metrics = nob._eval_fno_root_only_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
        )

        assert root_only_metrics.root_mae == pytest.approx(full_metrics.root_mae, abs=1e-6)
        assert root_only_metrics.root_median_abs_error == pytest.approx(
            full_metrics.root_median_abs_error,
            abs=1e-6,
        )
        assert root_only_metrics.root_p95_abs_error == pytest.approx(
            full_metrics.root_p95_abs_error,
            abs=1e-6,
        )
        assert root_only_metrics.n_docs == full_metrics.n_docs

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_exact_sketch_direct_metrics_streaming_matches_legacy(self):
        torch.manual_seed(0)
        np.random.seed(0)
        raw_docs = _make_tiny_docs(n=2, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature_adapters",
            theorem_feature_dim=8,
            theorem_feature_hidden_dim=16,
        )

        legacy = nob._eval_fno_exact_sketch_direct_metrics_legacy(
            model,
            fno_docs,
            device=torch.device("cpu"),
            phi_pair_calibration_max_nodes=None,
        )
        streaming = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
            phi_pair_calibration_max_nodes=None,
        )

        _assert_exact_sketch_metric_mappings_close(
            streaming,
            legacy,
            default_abs_tol=1e-6,
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_exact_sketch_direct_metrics_emits_bounded_batch_memory_probes(self):
        torch.manual_seed(0)
        np.random.seed(0)
        raw_docs = _make_tiny_docs(n=3, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature_adapters",
            theorem_feature_dim=8,
            theorem_feature_hidden_dim=16,
        )
        events: list[tuple[str, dict[str, object]]] = []

        metrics = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
            pack_mode="fixed_fused",
            max_docs=2,
            phi_pair_calibration_max_nodes=None,
            memory_probe=lambda event, payload: events.append((str(event), dict(payload))),
        )

        pre_batches = [payload for event, payload in events if event == "pre_exact_eval_batch"]
        post_batches = [payload for event, payload in events if event == "post_exact_eval_batch"]
        trim_batches = [payload for event, payload in events if event == "post_exact_eval_batch_trim"]

        assert len(pre_batches) == 2
        assert len(post_batches) == 2
        assert len(trim_batches) == 2
        assert all(int(payload["batch_docs"]) <= 2 for payload in pre_batches)
        assert all(int(payload["max_docs"]) == 2 for payload in pre_batches)
        assert metrics["n_docs"] == pytest.approx(3.0)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_exact_sketch_direct_metrics_fixed_fused_matches_structure_bucket(self):
        torch.manual_seed(0)
        np.random.seed(0)
        raw_docs = _make_tiny_docs(n=3, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature_adapters",
            theorem_feature_dim=8,
            theorem_feature_hidden_dim=16,
        )

        structure_bucket = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
            pack_mode="structure_bucket",
            phi_pair_calibration_max_nodes=None,
        )
        fixed_fused = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
            pack_mode="fixed_fused",
            phi_pair_calibration_max_nodes=None,
        )

        _assert_exact_sketch_metric_mappings_close(
            fixed_fused,
            structure_bucket,
            default_abs_tol=1e-5,
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_eval_fno_exact_sketch_direct_metrics_fixed_fused_auto_queue_matches_structure_bucket_mixed_leaf_counts(
        self,
    ):
        torch.manual_seed(0)
        np.random.seed(0)
        raw_docs = _make_mixed_length_docs(
            counts_by_seq_len=((32, 2), (40, 2), (48, 2)),
        )
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            task_head_mode="theorem_feature_scalar",
            summary_spec_root_mode="factored_theorem_readout",
            theorem_surface_mode="shared_feature_adapters",
            theorem_feature_dim=8,
            theorem_feature_hidden_dim=16,
        )

        structure_bucket = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
            pack_mode="structure_bucket",
            phi_pair_calibration_max_nodes=None,
        )
        fixed_fused = nob._eval_fno_exact_sketch_direct_metrics(
            model,
            fno_docs,
            device=torch.device("cpu"),
            pack_mode="fixed_fused",
            runtime_bucket_mode=GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
            structural_pad_limit=0.5,
            auto_queue_min_docs=0,
            auto_queue_min_fill_ratio=0.5,
            auto_queue_target_by_n_leaves={4: 6, 5: 6, 6: 6},
            phi_pair_calibration_max_nodes=None,
        )

        _assert_exact_sketch_metric_mappings_close(
            fixed_fused,
            structure_bucket,
            default_abs_tol=1e-5,
        )

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_theorem_primary_root_loss_uses_summary_spec_count_supervision(self, monkeypatch):
        calls = []

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                self.use_summary_spec = True
                self.use_decoded_markov_sketch = False
                self.use_markov_summary_spec = True
                self.root_supervision_kind = "mse"
                self.target_scale = 1.0

            @staticmethod
            def uses_theorem_primary_root_mode() -> bool:
                return True

            def predict_canonical_count_from_state(self, state):
                return state + 1.0

            def forward_doc(self, *args, **kwargs):
                zero = torch.zeros((), dtype=self.weight.dtype)
                return {
                    "root_state": self.weight,
                    "pred_norm": self.weight,
                    "leaf_loss": zero,
                    "leaf_count": 0.0,
                    "c3_loss": zero,
                    "c3_count": 0.0,
                    "c2_loss": zero,
                    "c2_state_replay_mse": zero,
                    "c2_count": 0.0,
                    "loss_components": {
                        "leaf_count_loss": zero,
                        "leaf_first_loss": zero,
                        "leaf_last_loss": zero,
                        "merge_count_loss": zero,
                        "merge_first_loss": zero,
                        "merge_last_loss": zero,
                        "c2_count_loss": zero,
                        "c2_first_loss": zero,
                        "c2_last_loss": zero,
                        "c2_join_loss": zero,
                        "c2_on_range_reencode_loss": zero,
                    },
                }

        doc = nob._FNOCountDoc(
            n_tokens=1,
            leaf_token_ids=((0,),),
            leaf_counts=(0.0,),
            leaf_first_regimes=(0,),
            leaf_last_regimes=(0,),
            leaf_token_lengths=(1,),
            merge_counts_balanced=tuple(),
            merge_sizes_balanced=tuple(),
            merge_token_lengths=tuple(),
            root_count=1.0,
        )

        def _fake_summary_spec_supervision_terms(model, state, *, truth_count, **kwargs):
            calls.append(float(truth_count))
            loss = state.sum() + 1.0
            return {
                "count_loss": loss,
                "first_loss": torch.zeros_like(loss),
                "last_loss": torch.zeros_like(loss),
                "total_loss": loss,
            }

        monkeypatch.setattr(
            nob,
            "_summary_spec_supervision_terms",
            _fake_summary_spec_supervision_terms,
        )
        monkeypatch.setattr(
            nob,
            "_eval_fno_model",
            lambda model, docs, *, device, tau: nob._zero_sketch_metrics(n_docs=len(docs)),
        )
        monkeypatch.setattr(
            nob,
            "_eval_fno_exact_sketch_direct_metrics",
            lambda model, docs, *, device: {
                "root_direct_count_mae": 0.0,
                "task_root_mae": 0.0,
                "task_root_mae_ablation": 0.0,
                "leaf_direct_count_mae": 0.0,
                "leaf_direct_exact_match": 1.0,
                "merge_direct_exact_match": 1.0,
                "merge_join_bit_accuracy": 1.0,
                "c2_on_range_exact_match": 1.0,
                "val_leaf_codec_direct": 0.0,
                "val_theorem_bootstrap_direct": 0.0,
                "val_exact_sketch_direct": 0.0,
                "val_task_root_exact_sketch_direct": 0.0,
                "n_docs": float(len(docs)),
                "n_leaf_nodes": 0.0,
                "n_merge_nodes": 0.0,
            },
        )

        nob._train_fno_tree_single_stage(
            model=DummyModel(),
            train_docs=(doc,),
            val_docs=tuple(),
            device=torch.device("cpu"),
            n_epochs=1,
            batch_size=1,
            lr=1e-3,
            c1_weight=0.0,
            c2_weight=0.0,
            c3_weight=0.0,
            root_weight=1.0,
            checkpoint_metric="val_exact_sketch_direct",
        )

        assert calls == [1.0]

    def test_train_fno_tree_two_stage_schedule_uses_aligned_stage_metrics(self, monkeypatch):
        calls = []

        def _fake_single_stage(**kwargs):
            stage_index = len(calls) + 1
            calls.append(
                {
                    "n_epochs": int(kwargs["n_epochs"]),
                    "root_weight": float(kwargs["root_weight"]),
                    "c3_weight": float(kwargs["c3_weight"]),
                    "leaf_label_rate": float(kwargs["leaf_label_rate"]),
                    "leaf_supervision_kind": str(kwargs["leaf_supervision_kind"]),
                    "internal_supervision_kind": str(kwargs["internal_supervision_kind"]),
                    "tree_local_weighting_mode": str(
                        kwargs["tree_local_weighting_mode"]
                    ),
                    "checkpoint_metric": str(kwargs["checkpoint_metric"]),
                    "eval_mode": str(kwargs["eval_mode"]),
                    "screen_doc_limit": int(kwargs["screen_doc_limit"]),
                    "final_exact_doc_limit_override": int(
                        kwargs["final_exact_doc_limit_override"]
                    ),
                    "batch_pack_mode": str(kwargs["tree_batch_pack_mode"]),
                    "batch_token_budget": int(kwargs["tree_batch_token_budget"]),
                    "batch_node_budget": int(kwargs["tree_batch_node_budget"]),
                    "batch_autotune": bool(kwargs["tree_batch_autotune"]),
                    "eval_workers_per_mig": int(kwargs["tree_eval_workers_per_mig"]),
                }
            )
            fit_diag = nob.TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=int(kwargs["n_epochs"]),
                selection_metric_curve=(0.1,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name=str(kwargs["checkpoint_metric"]),
                selection_metric_value=0.1,
                best_epoch=0,
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": fit_diag,
                "best_epoch": 0,
                "best_val_mae": 0.1,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": str(kwargs["checkpoint_metric"]),
                "selection_metric_curve": (0.1,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
                "timing_breakdown": {
                    "autotune_heuristic_s": 0.1 * stage_index,
                    "autotune_train_probe_s": 0.2 * stage_index,
                    "autotune_eval_probe_s": 0.3 * stage_index,
                    "autotune_cache_lookup_s": 0.01 * stage_index,
                    "autotune_cache_write_s": 0.02 * stage_index,
                    "autotune_total_s": 0.63 * stage_index,
                },
                "batching_metrics": {
                    "autotune_heuristic_time_s": 0.1 * stage_index,
                    "autotune_train_probe_time_s": 0.2 * stage_index,
                    "autotune_eval_probe_time_s": 0.3 * stage_index,
                    "autotune_cache_lookup_time_s": 0.01 * stage_index,
                    "autotune_cache_write_time_s": 0.02 * stage_index,
                    "autotune_cache_hits": stage_index,
                    "autotune_cache_misses": 2 * stage_index,
                    "autotune_cache_writes": stage_index,
                    "autotune_probe_runs": 3 * stage_index,
                    "autotune_probe_candidate_evals": 4 * stage_index,
                },
                "autotuned_batch_budgets": {
                    "probe_cache_version": AUTOTUNE_PROBE_CACHE_VERSION,
                    "probe_cache_hits": stage_index,
                    "probe_cache_misses": 2 * stage_index,
                    "probe_cache_writes": stage_index,
                    "probe_run_count": 3 * stage_index,
                    "probe_candidate_count": 4 * stage_index,
                },
                "autotune_probe_profile": {
                    "profile_version": AUTOTUNE_PROBE_CACHE_VERSION,
                    "heuristic_time_s": 0.1 * stage_index,
                    "train_probe_time_s": 0.2 * stage_index,
                    "eval_probe_time_s": 0.3 * stage_index,
                    "cache_lookup_time_s": 0.01 * stage_index,
                    "cache_write_time_s": 0.02 * stage_index,
                    "cache_hits": stage_index,
                    "cache_misses": 2 * stage_index,
                    "cache_writes": stage_index,
                    "probe_run_count": 3 * stage_index,
                    "probe_candidate_count": 4 * stage_index,
                    "runs": [{"stage": stage_index}],
                },
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel:
            use_summary_spec = True
            use_markov_summary_spec = True

            @staticmethod
            def uses_theorem_primary_root_mode() -> bool:
                return True

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return True

        model = DummyModel()
        out = train_fno_tree(
            model=model,
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=32,
            batch_size=4,
            lr=1e-3,
            c1_weight=0.5,
            c2_weight=0.5,
            c3_weight=0.5,
            root_weight=1.0,
            leaf_label_rate=0.25,
            internal_supervision_kind="count_only",
            internal_label_rate=0.25,
            leaf_supervision_kind="count_only",
            tree_local_weighting_mode="subset_mean",
            checkpoint_metric="val_exact_sketch_direct",
            tree_training_schedule="two_stage",
            tree_stage1_epochs=12,
            tree_stage2_epochs=20,
            tree_stage1_checkpoint_metric="val_theorem_bootstrap_direct",
            tree_stage1_eval_mode="end_only",
            tree_stage1_screen_doc_limit=32,
            tree_stage1_final_exact_doc_limit=7,
            tree_stage1_root_weight=0.4,
            tree_summary_spec_root_mode="theorem_primary",
            tree_batch_pack_mode="structure_bucket",
            tree_batch_token_budget=512,
            tree_batch_node_budget=64,
            tree_batch_autotune=False,
            tree_eval_workers_per_mig=2,
        )

        assert len(calls) == 2
        assert calls[0]["root_weight"] == pytest.approx(0.4)
        assert calls[0]["c3_weight"] == pytest.approx(1.0)
        assert calls[0]["leaf_label_rate"] == pytest.approx(1.0)
        assert calls[0]["leaf_supervision_kind"] == "full_sketch"
        assert calls[0]["internal_supervision_kind"] == "full_sketch"
        assert calls[0]["tree_local_weighting_mode"] == "subset_mean"
        assert calls[0]["eval_mode"] == "end_only"
        assert calls[0]["screen_doc_limit"] == 32
        assert calls[0]["final_exact_doc_limit_override"] == 7
        assert calls[0]["checkpoint_metric"] == "val_theorem_bootstrap_direct"
        assert calls[0]["batch_pack_mode"] == "structure_bucket"
        assert calls[0]["batch_token_budget"] == 512
        assert calls[0]["batch_node_budget"] == 64
        assert calls[0]["batch_autotune"] is False
        assert calls[0]["eval_workers_per_mig"] == 2
        assert calls[1]["leaf_label_rate"] == pytest.approx(0.25)
        assert calls[1]["leaf_supervision_kind"] == "count_only"
        assert calls[1]["tree_local_weighting_mode"] == "subset_mean"
        assert out["timing_breakdown"]["autotune_total_s"] == pytest.approx(0.63 + 1.26)
        assert out["batching_metrics"]["autotune_cache_hits"] == 3
        assert out["batching_metrics"]["autotune_probe_candidate_evals"] == 12
        assert out["autotune_probe_profile"]["probe_run_count"] == 9
        assert out["autotune_probe_profile"]["probe_candidate_count"] == 12
        assert len(out["autotune_probe_profile"]["runs"]) == 2
        assert out["autotuned_batch_budgets"]["probe_cache_hits"] == 3
        assert calls[1]["internal_supervision_kind"] == "count_only"
        assert calls[1]["eval_mode"] == "per_epoch"
        assert calls[1]["screen_doc_limit"] == 0
        assert calls[1]["final_exact_doc_limit_override"] == 0
        assert calls[1]["checkpoint_metric"] == "val_exact_sketch_direct"
        assert calls[1]["batch_pack_mode"] == "structure_bucket"
        assert calls[1]["batch_token_budget"] == 512
        assert calls[1]["batch_node_budget"] == 64
        assert calls[1]["batch_autotune"] is False
        assert calls[1]["eval_workers_per_mig"] == 2
        assert out["training_schedule"] == "two_stage"

    def test_train_fno_tree_two_stage_schedule_respected_without_summary_spec(
        self, monkeypatch
    ):
        calls = []

        def _fake_single_stage(**kwargs):
            calls.append(
                {
                    "n_epochs": int(kwargs["n_epochs"]),
                    "progress_stage_name": str(kwargs["progress_stage_name"]),
                }
            )
            fit_diag = nob.TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=int(kwargs["n_epochs"]),
                selection_metric_curve=(0.1,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name="val_root_mae",
                selection_metric_value=0.1,
                best_epoch=0,
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": fit_diag,
                "best_epoch": 0,
                "best_val_mae": 0.1,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": "val_root_mae",
                "selection_metric_curve": (0.1,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel:
            use_summary_spec = False
            use_markov_summary_spec = False

            @staticmethod
            def uses_theorem_primary_root_mode() -> bool:
                return False

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return False

        out = train_fno_tree(
            model=DummyModel(),
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=32,
            batch_size=4,
            lr=1e-3,
            tree_training_schedule="two_stage",
            tree_stage1_epochs=12,
            tree_stage2_epochs=20,
        )

        assert len(calls) == 2
        assert calls[0]["progress_stage_name"] == "stage1"
        assert calls[0]["n_epochs"] == 12
        assert calls[1]["progress_stage_name"] == "stage2"
        assert calls[1]["n_epochs"] == 20
        assert out["training_schedule"] == "two_stage"

    def test_train_fno_tree_allows_task_root_exact_sketch_checkpoint_metric(
        self, monkeypatch
    ):
        def _fake_single_stage(**kwargs):
            fit_diag = nob.TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=int(kwargs["n_epochs"]),
                selection_metric_curve=(0.2,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name=str(kwargs["checkpoint_metric"]),
                selection_metric_value=0.2,
                best_epoch=0,
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": fit_diag,
                "best_epoch": 0,
                "best_val_mae": 0.2,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": str(kwargs["checkpoint_metric"]),
                "selection_metric_curve": (0.2,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel:
            use_summary_spec = True
            use_markov_summary_spec = True

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return True

        out = train_fno_tree(
            model=DummyModel(),
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=4,
            batch_size=2,
            lr=1e-3,
            checkpoint_metric="val_task_root_exact_sketch_direct",
            tree_training_schedule="single_stage",
        )

        assert out["selection_metric_name"] == "val_task_root_exact_sketch_direct"

    def test_train_fno_tree_can_resume_stage2_from_saved_stage1_artifact(
        self,
        monkeypatch,
        tmp_path,
    ):
        calls = []

        def _fake_single_stage(**kwargs):
            calls.append(kwargs)
            fit_diag = nob.TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=int(kwargs["n_epochs"]),
                selection_metric_curve=(0.1,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name=str(kwargs["checkpoint_metric"]),
                selection_metric_value=0.1,
                best_epoch=0,
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": fit_diag,
                "best_epoch": 0,
                "best_val_mae": 0.1,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": str(kwargs["checkpoint_metric"]),
                "selection_metric_curve": (0.1,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
                "best_model_state": {"weight": torch.tensor([1.0])},
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                self.use_summary_spec = True
                self.use_markov_summary_spec = False

            @staticmethod
            def uses_hybrid_ordinal_count_head() -> bool:
                return False

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return False

        artifact_dir = tmp_path / "saved_stage1"
        stage1_model = DummyModel()
        write_theorem_feature_stage1_artifact(
            artifact_dir,
            model_state=stage1_model.state_dict(),
            metadata={
                "selection_metric_name": "val_leaf_codec_direct",
                "selection_metric_value": 0.25,
                "best_epoch": 2,
                "epochs_completed": 3,
                "training_schedule": "two_stage",
                "artifact_source": "trained",
            },
        )

        out = train_fno_tree(
            model=DummyModel(),
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=2,
            batch_size=2,
            lr=1e-3,
            tree_training_schedule="two_stage",
            tree_stage1_epochs=0,
            tree_stage2_epochs=2,
            tree_stage1_artifact_dir=str(artifact_dir),
        )

        assert len(calls) == 1
        assert int(calls[0]["n_epochs"]) == 2
        assert out["training_schedule"] == "two_stage"
        assert out["stage1_artifact"]["artifact_source"] == "loaded"

    def test_train_fno_tree_auto_reuses_saved_stage1_artifact_when_enabled(
        self,
        monkeypatch,
        tmp_path,
    ):
        calls = []

        def _fake_single_stage(**kwargs):
            calls.append(dict(kwargs))
            fit_diag = nob.TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=int(kwargs["n_epochs"]),
                selection_metric_curve=(0.1,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name=str(kwargs["checkpoint_metric"]),
                selection_metric_value=0.1,
                best_epoch=0,
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": fit_diag,
                "best_epoch": 0,
                "best_val_mae": 0.1,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": str(kwargs["checkpoint_metric"]),
                "selection_metric_curve": (0.1,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
                "best_model_state": {"weight": torch.tensor([1.0])},
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                self.use_summary_spec = True
                self.use_markov_summary_spec = False

            @staticmethod
            def uses_hybrid_ordinal_count_head() -> bool:
                return False

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return False

        artifact_dir = tmp_path / "saved_stage1_auto"
        write_theorem_feature_stage1_artifact(
            artifact_dir,
            model_state=DummyModel().state_dict(),
            metadata={
                "selection_metric_name": "val_leaf_codec_direct",
                "selection_metric_value": 0.25,
                "best_epoch": 2,
                "epochs_completed": 3,
                "training_schedule": "two_stage",
                "artifact_source": "trained",
            },
        )

        out = train_fno_tree(
            model=DummyModel(),
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=4,
            batch_size=2,
            lr=1e-3,
            tree_training_schedule="two_stage",
            tree_stage1_epochs=2,
            tree_stage2_epochs=4,
            tree_stage1_artifact_dir=str(artifact_dir),
            tree_stage1_resume_if_available=True,
        )

        assert len(calls) == 1
        assert int(calls[0]["n_epochs"]) == 4
        assert out["stage1_artifact"]["artifact_source"] == "loaded"

    def test_train_fno_tree_stage1_artifact_metadata_keeps_best_exact_metrics(
        self,
        monkeypatch,
        tmp_path,
    ):
        calls = []

        def _fake_single_stage(**kwargs):
            calls.append(dict(kwargs))
            fit_diag = nob.TrainFitDiagnostics(
                train_loss_final=0.0,
                train_loss_curve=(0.0,),
                epochs_completed=int(kwargs["n_epochs"]),
                selection_metric_curve=(0.1,),
                selection_mode="min",
                selection_split="val",
                selection_metric_name=str(kwargs["checkpoint_metric"]),
                selection_metric_value=0.1,
                best_epoch=0,
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": fit_diag,
                "best_epoch": 0,
                "best_val_mae": 0.1,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": str(kwargs["checkpoint_metric"]),
                "selection_metric_curve": (0.1,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
                "best_model_state": {"weight": torch.tensor([1.0])},
                "best_exact_metrics": {"val_theorem_bootstrap_direct": 0.25},
                "best_exact_metrics_split": "val",
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                self.use_summary_spec = True
                self.use_markov_summary_spec = False

            @staticmethod
            def uses_hybrid_ordinal_count_head() -> bool:
                return False

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return False

        artifact_dir = tmp_path / "stage1_artifact"
        out = train_fno_tree(
            model=DummyModel(),
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=2,
            batch_size=2,
            lr=1e-3,
            tree_training_schedule="two_stage",
            tree_stage1_epochs=1,
            tree_stage2_epochs=1,
            tree_stage1_checkpoint_metric="val_root_mae",
            tree_stage1_artifact_dir=str(artifact_dir),
        )

        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        assert len(calls) == 2
        assert metadata["best_exact_metrics_split"] == "val"
        assert metadata["best_exact_metrics"]["val_theorem_bootstrap_direct"] == pytest.approx(0.25)
        assert out["stage1_artifact"]["artifact_source"] == "trained"

    def test_train_fno_tree_auto_aligns_stage1_checkpoint_metric_when_root_is_off(
        self, monkeypatch
    ):
        calls = []

        def _fake_single_stage(**kwargs):
            calls.append(
                {
                    "checkpoint_metric": kwargs["checkpoint_metric"],
                    "root_weight": kwargs["root_weight"],
                    "n_epochs": kwargs["n_epochs"],
                }
            )
            return {
                "train": {"root_mae": 0.0, "exact_match": 1.0},
                "val": {"root_mae": 0.0, "exact_match": 1.0},
                "fit_diag": nob.TrainFitDiagnostics(
                    train_loss_final=0.0,
                    train_loss_curve=(0.0,),
                    epochs_completed=int(kwargs["n_epochs"]),
                    selection_metric_curve=(0.0,),
                    selection_mode="min",
                    selection_split="val",
                    selection_metric_name=str(kwargs["checkpoint_metric"]),
                    selection_metric_value=0.0,
                    best_epoch=0,
                    train_exact_match_rate=1.0,
                    val_exact_match_rate=1.0,
                    test_exact_match_rate=1.0,
                ),
                "best_epoch": 0,
                "best_val_mae": 0.0,
                "selection_mode": "min",
                "selection_split": "val",
                "selection_metric_name": str(kwargs["checkpoint_metric"]),
                "selection_metric_curve": (0.0,),
                "loss_curve": (0.0,),
                "epochs_completed": int(kwargs["n_epochs"]),
                "training_component_loss_curves": {},
                "training_component_loss_finals": {},
                "best_model_state": {},
                "best_exact_metrics": {},
                "best_exact_metrics_split": "val",
                "elapsed_s_train_loop": 0.0,
                "elapsed_s_screen_eval": 0.0,
                "elapsed_s_exact_metric_eval": 0.0,
                "elapsed_s_split_eval": 0.0,
                "elapsed_s_state_clone": 0.0,
                "timing_breakdown": {},
                "batching_metrics": {},
                "autotuned_batch_budgets": {},
                "autotune_probe_profile": {},
            }

        monkeypatch.setattr(nob, "_train_fno_tree_single_stage", _fake_single_stage)

        class DummyModel:
            use_summary_spec = True
            use_markov_summary_spec = False
            use_decoded_markov_sketch = False

            @staticmethod
            def uses_hybrid_ordinal_count_head() -> bool:
                return False

            @staticmethod
            def uses_theory_aligned_root_surface() -> bool:
                return False

        train_fno_tree(
            model=DummyModel(),
            train_docs=(),
            val_docs=(),
            device=torch.device("cpu"),
            n_epochs=4,
            batch_size=2,
            lr=1e-3,
            tree_training_schedule="two_stage",
            tree_stage1_epochs=2,
            tree_stage2_epochs=2,
            tree_stage1_checkpoint_metric="val_root_mae",
            tree_stage1_root_weight=0.0,
            checkpoint_metric="val_root_mae",
        )

        assert len(calls) == 2
        assert calls[0]["root_weight"] == pytest.approx(0.0)
        assert calls[0]["checkpoint_metric"] == "val_exact_sketch_direct"
        assert calls[1]["checkpoint_metric"] == "val_root_mae"

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_train_fno_tree_emits_epoch_progress_snapshots(self):
        raw_docs = _make_tiny_docs(n=6, seq_len=16, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )
        progress_events = []

        out = train_fno_tree(
            model=model,
            train_docs=fno_docs[:4],
            val_docs=fno_docs[4:6],
            device=torch.device("cpu"),
            n_epochs=2,
            batch_size=2,
            lr=1e-3,
            tree_training_schedule="single_stage",
            tree_batch_autotune=False,
            runtime_config=GpuRuntimeConfig(data_mode="cpu_debug"),
            progress_callback=lambda payload: progress_events.append(dict(payload)),
        )

        assert [str(event["stage"]) for event in progress_events] == [
            "single_stage_train",
            "single_stage_train",
            "single_stage_train",
            "single_stage_final_eval",
        ]
        assert [int(event["epoch_completed"]) for event in progress_events] == [0, 1, 2, 2]
        assert all(int(event["epochs_total"]) == 2 for event in progress_events)
        assert [int(event["stage_epoch_completed"]) for event in progress_events] == [0, 1, 2, 2]
        assert all(int(event["stage_epochs_total"]) == 2 for event in progress_events)
        assert all(str(event["state"]) == "running" for event in progress_events)
        assert str(progress_events[-1]["selection_metric_name"]) == str(out["selection_metric_name"])

    def test_tree_document_loss_batch_scale_is_coverage_invariant_in_supervised_mode(
        self,
    ):
        assert nob._tree_document_loss_batch_scale(
            normalization_mode="supervised_docs",
            batch_docs=10,
            supervised_docs=1,
        ) == pytest.approx(10.0)
        assert nob._tree_document_loss_batch_scale(
            normalization_mode="supervised_docs",
            batch_docs=10,
            supervised_docs=10,
        ) == pytest.approx(1.0)
        assert nob._tree_document_loss_batch_scale(
            normalization_mode="batch_docs",
            batch_docs=10,
            supervised_docs=1,
        ) == pytest.approx(1.0)

    def test_tree_document_loss_auto_mode_resolves_from_explicit_doc_supervision(
        self,
    ):
        assert (
            nob._effective_tree_document_loss_normalization_mode(
                "auto",
                explicit_doc_modes={0: "root_only", 1: "doc_sequence"},
            )
            == "supervised_docs"
        )
        assert (
            nob._effective_tree_document_loss_normalization_mode(
                "auto",
                explicit_doc_modes=None,
            )
            == "batch_docs"
        )

    def test_train_fno_tree_single_stage_reports_document_normalization_metadata(
        self,
        monkeypatch,
    ):
        def _fake_eval_root_only_metrics(*args, **kwargs):
            return nob._zero_sketch_metrics(n_docs=0)

        def _fake_eval_exact(*args, **kwargs):
            docs = list(args[1]) if len(args) > 1 else list(kwargs.get("docs", ()))
            return {
                "root_direct_count_mae": 0.0,
                "task_root_mae": 0.0,
                "task_root_mae_ablation": 0.0,
                "leaf_direct_count_mae": 0.0,
                "leaf_direct_exact_match": 1.0,
                "merge_direct_exact_match": 1.0,
                "merge_join_bit_accuracy": 1.0,
                "c2_on_range_exact_match": 1.0,
                "val_leaf_codec_direct": 0.0,
                "val_theorem_bootstrap_direct": 0.0,
                "val_exact_sketch_direct": 0.0,
                "val_task_root_exact_sketch_direct": 0.0,
                "n_docs": float(len(docs)),
                "n_leaf_nodes": 0.0,
                "n_merge_nodes": 0.0,
            }

        monkeypatch.setattr(nob, "_eval_fno_root_only_metrics", _fake_eval_root_only_metrics)
        monkeypatch.setattr(
            nob,
            "_eval_fno_exact_sketch_direct_metrics",
            _fake_eval_exact,
        )

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                self.target_scale = 8.0
                self.root_supervision_kind = "mse"
                self.use_summary_spec = False
                self.use_markov_summary_spec = False
                self.use_shared_theorem_surface = False
                self.use_decoded_markov_sketch = False
                self.pad_id = 0

            def predict_canonical_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
                return state.reshape(())

            def forward_doc(self, *args, **kwargs):
                pred_norm = self.weight.reshape(())
                zero = pred_norm * 0.0
                return {
                    "pred_norm": pred_norm,
                    "root_state": pred_norm.reshape(1),
                    "leaf_loss": zero,
                    "c2_loss": zero,
                    "c3_loss": zero,
                    "leaf_count": 0.0,
                    "c2_count": 0.0,
                    "c3_count": 0.0,
                    "loss_components": {},
                }

        docs = (
            _make_tiny_fno_doc([0, 1, 2, 3], root_count=1.0),
            _make_tiny_fno_doc([0, 1, 2, 3], root_count=2.0),
        )
        out = nob._train_fno_tree_single_stage(
            model=DummyModel(),
            train_docs=docs,
            val_docs=tuple(),
            device=torch.device("cpu"),
            n_epochs=1,
            batch_size=2,
            lr=1e-3,
            document_supervision_mode_by_doc={0: "root_only"},
            tree_document_loss_normalization_mode="auto",
            tree_batch_autotune=False,
            runtime_config=nob.GpuRuntimeConfig(data_mode="cpu_debug"),
        )

        assert out["tree_document_loss_normalization_mode"] == "auto"
        assert out["effective_tree_document_loss_normalization_mode"] == "supervised_docs"
        assert out["document_supervision_docs_total"] == 1
        assert out["root_supervision_docs_total"] == 1
        assert out["doc_sequence_supervision_docs_total"] == 0
        assert out["document_supervision_coverage_rate"] == pytest.approx(0.5)
        assert out["document_loss_mean_batch_scale"] == pytest.approx(2.0)
        runtime = dict(out["runtime_efficiency"])
        assert runtime["effective_tree_document_loss_normalization_mode"] == "supervised_docs"
        assert runtime["document_loss_mean_batch_scale"] == pytest.approx(2.0)


def test_manifest_supervision_disables_fallback_local_sampling(monkeypatch) -> None:
    recorded: list[dict[str, object]] = []

    monkeypatch.setattr(
        nob,
        "_eval_fno_exact_sketch_direct_metrics",
        lambda *args, **kwargs: {
            "val_root_mae": 0.0,
            "val_leaf_codec_direct": 0.0,
            "val_theorem_bootstrap_direct": 0.0,
            "val_exact_sketch_direct": 0.0,
            "val_task_root_exact_sketch_direct": 0.0,
            "n_docs": 0.0,
            "n_leaf_nodes": 0.0,
            "n_merge_nodes": 0.0,
        },
    )

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            self.target_scale = 8.0
            self.root_supervision_kind = "mse"
            self.use_summary_spec = True
            self.use_markov_summary_spec = True
            self.use_shared_theorem_surface = False
            self.use_decoded_markov_sketch = False
            self.pad_id = 0

        @staticmethod
        def uses_theorem_primary_root_mode() -> bool:
            return False

        def predict_canonical_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
            return state.reshape(())

        def forward_doc(self, *args, **kwargs):
            recorded.append(
                {
                    "collect_leaf": bool(kwargs.get("collect_leaf", False)),
                    "collect_c3": bool(kwargs.get("collect_c3", False)),
                    "leaf_audit_indices": kwargs.get("leaf_audit_indices"),
                    "c3_audit_indices": kwargs.get("c3_audit_indices"),
                }
            )
            pred_norm = self.weight.reshape(())
            zero = pred_norm * 0.0
            return {
                "pred_norm": pred_norm,
                "root_state": self.weight.reshape(1),
                "leaf_loss": zero,
                "c2_loss": zero,
                "c3_loss": zero,
                "leaf_count": 0.0,
                "c2_count": 0.0,
                "c3_count": 0.0,
                "loss_components": {},
            }

    docs = (_make_tiny_fno_doc([0, 1, 2, 3], root_count=1.0),)
    out = nob._train_fno_tree_single_stage(
        model=DummyModel(),
        train_docs=docs,
        val_docs=tuple(),
        device=torch.device("cpu"),
        n_epochs=1,
        batch_size=1,
        lr=1e-3,
        c1_weight=1.0,
        c3_weight=1.0,
        leaf_label_rate=1.0,
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        document_supervision_mode_by_doc={0: "root_only"},
        leaf_audit_indices_by_doc={},
        c3_audit_indices_by_doc={},
        tree_supervision_source="manifest",
        tree_batch_autotune=False,
        runtime_config=nob.GpuRuntimeConfig(data_mode="cpu_debug"),
    )

    train_records = [
        row for row in recorded if row["leaf_audit_indices"] is not None
    ]
    assert train_records
    assert all(row["collect_leaf"] is False for row in train_records)
    assert all(row["collect_c3"] is False for row in train_records)
    assert all(row["leaf_audit_indices"] == set() for row in train_records)
    assert all(row["c3_audit_indices"] == set() for row in train_records)
    assert out["tree_supervision_source"] == "manifest"


def test_span_mass_ipw_sum_matches_hand_computation() -> None:
    values = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    propensities = torch.tensor(
        [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        dtype=torch.float32,
    )
    node_scales = torch.tensor(
        [[0.25, 0.0, 0.25], [0.0, 0.5, 0.0]],
        dtype=torch.float32,
    )

    means, active, numerators, denominators = nob._masked_doc_local_means(
        values,
        mask,
        propensities,
        weighting_mode="span_mass_ipw_sum",
        node_scales=node_scales,
    )

    assert active.tolist() == [True, True]
    assert means.tolist() == pytest.approx([2.5, 10.0])
    assert numerators.tolist() == pytest.approx([2.5, 10.0])
    assert denominators.tolist() == pytest.approx([1.0, 2.0])


def test_c2_pair_inclusion_probabilities_cover_all_pair_classes() -> None:
    assert nob._c2_pair_inclusion_propensity(
        kind_left=nob._C2_NODE_KIND_LEAF,
        kind_right=nob._C2_NODE_KIND_LEAF,
        leaf_population_size=4,
        leaf_sample_size=2,
        merge_population_size=3,
        merge_sample_size=1,
    ) == pytest.approx(1.0 / 6.0)
    assert nob._c2_pair_inclusion_propensity(
        kind_left=nob._C2_NODE_KIND_MERGE,
        kind_right=nob._C2_NODE_KIND_MERGE,
        leaf_population_size=4,
        leaf_sample_size=2,
        merge_population_size=3,
        merge_sample_size=2,
    ) == pytest.approx(1.0 / 3.0)
    assert nob._c2_pair_inclusion_propensity(
        kind_left=nob._C2_NODE_KIND_LEAF,
        kind_right=nob._C2_NODE_KIND_MERGE,
        leaf_population_size=4,
        leaf_sample_size=2,
        merge_population_size=3,
        merge_sample_size=1,
    ) == pytest.approx((2.0 / 4.0) * (1.0 / 3.0))
    assert nob._c2_pair_inclusion_propensity(
        kind_left=nob._C2_NODE_KIND_ROOT,
        kind_right=nob._C2_NODE_KIND_LEAF,
        leaf_population_size=4,
        leaf_sample_size=2,
        merge_population_size=3,
        merge_sample_size=1,
    ) == pytest.approx(0.5)
    assert nob._c2_pair_inclusion_propensity(
        kind_left=nob._C2_NODE_KIND_ROOT,
        kind_right=nob._C2_NODE_KIND_MERGE,
        leaf_population_size=4,
        leaf_sample_size=2,
        merge_population_size=3,
        merge_sample_size=1,
    ) == pytest.approx(1.0 / 3.0)


def test_weighted_c2_pairwise_helpers_match_hand_computation() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.9, -0.1],
        ],
        dtype=torch.float32,
    )
    same_pairs = ((0, 1),)
    different_pairs = ((0, 2), (1, 2))
    same_weights = (2.0,)
    different_weights = (1.0, 3.0)
    pair_weights = torch.zeros((3, 3), dtype=torch.float32)
    pair_weights[0, 1] = pair_weights[1, 0] = same_weights[0]
    pair_weights[0, 2] = pair_weights[2, 0] = different_weights[0]
    pair_weights[1, 2] = pair_weights[2, 1] = different_weights[1]
    same_mask = torch.zeros((3, 3), dtype=torch.bool)
    same_mask[0, 1] = same_mask[1, 0] = True
    different_mask = torch.zeros((3, 3), dtype=torch.bool)
    different_mask[0, 2] = different_mask[2, 0] = True
    different_mask[1, 2] = different_mask[2, 1] = True

    normalized = F.normalize(embeddings, dim=-1)
    same_sim = float((normalized[0] * normalized[1]).sum().item())
    diff_sim_02 = float((normalized[0] * normalized[2]).sum().item())
    diff_sim_12 = float((normalized[1] * normalized[2]).sum().item())
    expected_same = 1.0 - same_sim
    expected_diff = (
        different_weights[0] * max(0.0, diff_sim_02 - 0.5)
        + different_weights[1] * max(0.0, diff_sim_12 - 0.5)
    ) / sum(different_weights)
    expected_loss = 0.5 * (expected_same + expected_diff)

    direct = nob._pairwise_theorem_feature_contrastive_loss(
        embeddings,
        same_pairs=same_pairs,
        different_pairs=different_pairs,
        same_pair_weights=same_weights,
        different_pair_weights=different_weights,
    )
    masked = nob._pairwise_theorem_feature_contrastive_loss_from_masks(
        embeddings,
        same_mask=same_mask,
        different_mask=different_mask,
        pair_weights=pair_weights,
    )
    batched = nob._batched_pairwise_theorem_feature_contrastive_loss_from_masks(
        embeddings.unsqueeze(0),
        same_mask=same_mask.unsqueeze(0),
        different_mask=different_mask.unsqueeze(0),
        pair_weights=pair_weights.unsqueeze(0),
    )

    assert float(direct.detach().cpu()) == pytest.approx(expected_loss)
    assert float(masked.detach().cpu()) == pytest.approx(expected_loss)
    assert float(batched.squeeze(0).detach().cpu()) == pytest.approx(expected_loss)


def test_authoritative_c2_gamma_zero_reduces_to_root_only() -> None:
    pair_weights = nob._c2_pair_weight_matrix(
        node_scales=torch.tensor([0.0, 1.0], dtype=torch.float32),
        node_kind_codes=torch.tensor(
            [nob._C2_NODE_KIND_LEAF, nob._C2_NODE_KIND_ROOT],
            dtype=torch.long,
        ),
        valid_mask=torch.tensor([True, True]),
        leaf_population_size=4,
        leaf_sample_size=1,
        merge_population_size=0,
        merge_sample_size=0,
    )
    same_mask = torch.tensor([[False, True], [True, False]])
    different_mask = torch.zeros_like(same_mask)
    loss = nob._pairwise_theorem_feature_contrastive_loss_from_masks(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        same_mask=same_mask,
        different_mask=different_mask,
        pair_weights=pair_weights,
    )

    assert pair_weights[0, 1].item() == pytest.approx(0.0)
    assert float(loss.detach().cpu()) == pytest.approx(0.0)


def test_c2_pair_weight_matrix_matches_reference_loop() -> None:
    node_scales = torch.tensor([0.25, 0.5, 1.0, 0.75, 0.0], dtype=torch.float32)
    node_kind_codes = torch.tensor(
        [
            nob._C2_NODE_KIND_LEAF,
            nob._C2_NODE_KIND_MERGE,
            nob._C2_NODE_KIND_ROOT,
            nob._C2_NODE_KIND_LEAF,
            nob._C2_NODE_KIND_MERGE,
        ],
        dtype=torch.long,
    )
    valid_mask = torch.tensor([True, True, True, False, True], dtype=torch.bool)
    kwargs = dict(
        leaf_population_size=8,
        leaf_sample_size=3,
        merge_population_size=5,
        merge_sample_size=2,
    )

    actual = nob._c2_pair_weight_matrix(
        node_scales=node_scales,
        node_kind_codes=node_kind_codes,
        valid_mask=valid_mask,
        **kwargs,
    )

    expected = torch.zeros_like(actual)
    for left_idx in range(int(node_scales.shape[0])):
        if not bool(valid_mask[left_idx].item()):
            continue
        left_scale = float(node_scales[left_idx].item())
        if left_scale <= 0.0:
            continue
        left_kind = int(node_kind_codes[left_idx].item())
        for right_idx in range(left_idx + 1, int(node_scales.shape[0])):
            if not bool(valid_mask[right_idx].item()):
                continue
            right_scale = float(node_scales[right_idx].item())
            if right_scale <= 0.0:
                continue
            propensity = nob._c2_pair_inclusion_propensity(
                kind_left=left_kind,
                kind_right=int(node_kind_codes[right_idx].item()),
                **kwargs,
            )
            if propensity <= 0.0:
                continue
            weight = (left_scale * right_scale) ** 0.5 / propensity
            expected[left_idx, right_idx] = float(weight)
            expected[right_idx, left_idx] = float(weight)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual, actual.transpose(0, 1), atol=0.0, rtol=0.0)
    assert torch.allclose(torch.diag(actual), torch.zeros((actual.shape[0],), dtype=actual.dtype))


def test_batched_pairwise_contrastive_loss_can_return_diagnostics() -> None:
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        ],
        dtype=torch.float32,
    )
    same_mask = torch.zeros((2, 3, 3), dtype=torch.bool)
    same_mask[0, 0, 1] = same_mask[0, 1, 0] = True
    same_mask[1, 0, 2] = same_mask[1, 2, 0] = True
    different_mask = torch.zeros_like(same_mask)
    different_mask[0, 0, 2] = different_mask[0, 2, 0] = True
    different_mask[0, 1, 2] = different_mask[0, 2, 1] = True
    different_mask[1, 0, 1] = different_mask[1, 1, 0] = True
    different_mask[1, 1, 2] = different_mask[1, 2, 1] = True
    pair_weights = torch.ones((2, 3, 3), dtype=torch.float32)

    out = nob._batched_pairwise_theorem_feature_contrastive_loss_from_masks(
        embeddings,
        same_mask=same_mask,
        different_mask=different_mask,
        pair_weights=pair_weights,
        return_diagnostics=True,
    )

    assert isinstance(out, dict)
    assert tuple(out["loss"].shape) == (2,)
    assert torch.allclose(out["same_pair_count"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(out["different_pair_count"], torch.tensor([2.0, 2.0]))
    assert torch.all(out["pair_weight_ess"] > 0.0)
    assert torch.all(out["pair_weight_max"] == 1.0)


def test_manifest_span_mass_training_reports_authoritative_gamma_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            self.target_scale = 8.0
            self.root_supervision_kind = "mse"
            self.use_summary_spec = True
            self.use_markov_summary_spec = True
            self.use_shared_theorem_surface = False
            self.use_decoded_markov_sketch = False
            self.pad_id = 0

        @staticmethod
        def uses_theorem_primary_root_mode() -> bool:
            return False

        def predict_canonical_count_from_state(self, state: torch.Tensor) -> torch.Tensor:
            return state.reshape(())

        def forward_doc(self, *args, **kwargs):
            pred_norm = self.weight.reshape(())
            zero = pred_norm * 0.0
            return {
                "pred_norm": pred_norm,
                "root_state": self.weight.reshape(1),
                "leaf_loss": zero,
                "c2_loss": zero,
                "c3_loss": zero,
                "leaf_count": 0.0,
                "c2_count": 0.0,
                "c3_count": 0.0,
                "loss_components": {},
            }

    def _stub_precomputed_views(*args, **kwargs):
        docs_arg = list(kwargs.get("docs") or args[1] or ())
        return [
            nob._PrecomputedDocStateView(
                state_batch=torch.zeros((1, 1), dtype=torch.float32),
                root_state=torch.zeros((1,), dtype=torch.float32),
                merge_states=tuple(),
            )
            for _ in docs_arg
        ]

    monkeypatch.setattr(
        nob,
        "_precompute_balanced_doc_state_views",
        _stub_precomputed_views,
    )

    docs = (_make_tiny_fno_doc([0, 1, 2, 3], root_count=1.0),)
    out = nob._train_fno_tree_single_stage(
        model=DummyModel(),
        train_docs=docs,
        val_docs=tuple(),
        device=torch.device("cpu"),
        n_epochs=1,
        batch_size=1,
        lr=1e-3,
        c1_weight=1.0,
        c3_weight=1.0,
        leaf_label_rate=1.0,
        internal_supervision_kind="count_only",
        internal_label_rate=1.0,
        tree_supervision_source="manifest",
        tree_local_weighting_mode="span_mass_ipw_sum",
        depth_discount_gamma=1.0,
        exact_metric_evaluator=lambda model, eval_docs, **kwargs: {},
        tree_batch_autotune=False,
        runtime_config=nob.GpuRuntimeConfig(data_mode="cpu_debug"),
    )

    assert out["tree_supervision_source"] == "manifest"
    assert out["local_estimand_mode"] == "span_mass_ipw_sum"
    assert out["depth_discount_gamma"] == pytest.approx(1.0)
    assert out["c2_pair_weighting_mode"] == "pair_ipw_geomean"


def test_low_level_ops_runner_rejects_parity_only_supervision_modes() -> None:
    with pytest.raises(ValueError, match="tree_supervision_source='rate'"):
        run_markov_changepoint_ops_count_experiment(
            OPSCountConfig(
                train_docs=1,
                val_docs=0,
                test_docs=0,
                tree_supervision_source="manifest",
            )
        )
    with pytest.raises(ValueError, match="tree_local_weighting_mode='fixed_k_hajek'"):
        run_markov_changepoint_ops_count_experiment(
            OPSCountConfig(
                train_docs=1,
                val_docs=0,
                test_docs=0,
                tree_local_weighting_mode="span_mass_ipw_sum",
            )
        )


class TestBigramFeatures:
    """Bigram feature extraction matches expected dimensions."""

    def test_feature_dim(self):
        V = 8
        tokens = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        mask = np.ones((1, 8), dtype=np.float32)
        feat = _bigram_features_from_tokens(tokens, mask, vocab_size=V)
        # 2*V + V + V^2 + 1 = 16 + 8 + 64 + 1 = 89
        assert feat.shape == (1, 2 * V + V + V * V + 1)

    def test_padding_ignored(self):
        V = 4
        tokens = np.array([[0, 1, 2, 0, 0]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 0, 0]], dtype=np.float32)
        feat = _bigram_features_from_tokens(tokens, mask, vocab_size=V)
        # length feature should be 3.0 (only 3 valid tokens)
        assert feat[0, -1] == 3.0


class TestOverfit:
    """MLP bigram can overfit a tiny dataset to near-zero loss."""

    def test_mlp_bigram_overfit(self):
        docs = _make_tiny_docs(n=10, seq_len=32, vocab_size=8)
        config = OPSCountConfig(
            n_epochs=100,
            state_dim=32,
            hidden_dim=128,
            batch_size=10,
            lr=1e-3,
            vocab_size=8,
        )
        seeds = {"effective_model_seed": 42}
        train_m, val_m, test_m, fit = _fit_mlp_bigram_baseline(
            config=config, seeds=seeds, device=torch.device("cpu"),
            train_docs=docs, val_docs=[], test_docs=docs,
        )
        # With 100 epochs on 10 examples, training loss should be very low
        assert fit.train_loss_final < 1.0, f"Expected overfit, got loss={fit.train_loss_final}"


class TestIntegration:
    """_fit_*_baseline returns valid SketchMetrics."""

    def _run_baseline(self, fit_fn):
        docs = _make_tiny_docs(n=20, seq_len=32, vocab_size=8)
        config = OPSCountConfig(
            n_epochs=3,
            state_dim=16,
            hidden_dim=32,
            batch_size=8,
            lr=1e-3,
            vocab_size=8,
        )
        seeds = {"effective_model_seed": 42}
        train_m, val_m, test_m, fit = fit_fn(
            config=config, seeds=seeds, device=torch.device("cpu"),
            train_docs=docs[:15], val_docs=docs[15:], test_docs=docs[:10],
        )
        assert test_m.root_mae >= 0.0
        assert test_m.n_docs == 10
        assert fit.epochs_completed == 3
        return test_m, fit

    def test_mlp_bigram_integration(self):
        self._run_baseline(_fit_mlp_bigram_baseline)

    def test_cnn1d_integration(self):
        self._run_baseline(_fit_cnn1d_baseline)

    def test_deeponet_integration(self):
        self._run_baseline(_fit_deeponet_baseline)

    @pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
    def test_fno_integration(self):
        from src.ctreepo.sim.core.markov_neural_operator_baselines import _fit_fno_baseline
        self._run_baseline(_fit_fno_baseline)


class TestEmptyDocs:
    """Baselines handle empty doc lists gracefully."""

    def test_mlp_bigram_empty(self):
        config = OPSCountConfig(n_epochs=1, vocab_size=8)
        seeds = {"effective_model_seed": 42}
        train_m, val_m, test_m, fit = _fit_mlp_bigram_baseline(
            config=config, seeds=seeds, device=torch.device("cpu"),
            train_docs=[], val_docs=[], test_docs=[],
        )
        assert test_m.root_mae == 0.0
        assert fit.epochs_completed == 0


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
class TestFNOModelFamily:
    """End-to-end smoke test for model_family='fno' in the main experiment."""

    def test_fno_model_family_smoke(self):
        config = OPSCountConfig(
            model_family="fno",
            n_regimes=2,
            vocab_size=8,
            min_tokens=32,
            max_tokens=32,
            min_segments=2,
            max_segments=4,
            fixed_leaf_tokens=8,
            train_docs=6,
            val_docs=2,
            test_docs=4,
            state_dim=8,
            hidden_dim=16,
            n_epochs=2,
            batch_size=4,
            lr=1e-3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            use_cuda=False,
            seed=42,
        )
        result = run_markov_changepoint_ops_count_experiment(config)
        metrics = result.metrics
        # Verify learned model metrics are populated.
        assert "learned" in metrics
        learned = metrics["learned"]
        assert learned["root_mae"] >= 0.0
        assert learned["leaf_mae"] >= 0.0
        assert learned["merge_mae"] >= 0.0
        assert learned["n_docs"] == 4

    def test_fno_model_family_rejects_law_specific_local_law_bundle(self):
        config = OPSCountConfig(
            model_family="fno",
            n_regimes=2,
            vocab_size=8,
            min_tokens=32,
            max_tokens=32,
            min_segments=2,
            max_segments=4,
            fixed_leaf_tokens=8,
            train_docs=6,
            val_docs=2,
            test_docs=4,
            state_dim=8,
            hidden_dim=16,
            n_epochs=2,
            batch_size=4,
            lr=1e-3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            law_package="c2_only",
            local_law_weight=0.5,
            use_cuda=False,
            seed=42,
        )
        with pytest.raises(ValueError, match="bundled corrected_local_law loss"):
            run_markov_changepoint_ops_count_experiment(config)

    def test_fno_model_family_v2_unified_runtime_smoke(self):
        config = OPSCountConfig(
            model_family="fno",
            n_regimes=2,
            vocab_size=8,
            min_tokens=32,
            max_tokens=32,
            min_segments=2,
            max_segments=4,
            fixed_leaf_tokens=8,
            train_docs=6,
            val_docs=2,
            test_docs=4,
            state_dim=8,
            hidden_dim=16,
            n_epochs=1,
            batch_size=4,
            lr=1e-3,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            use_cuda=False,
            seed=42,
            tree_model_version="v2",
            tree_batch_runtime_mode="unified_v2",
            tree_training_schedule="single_stage",
            tree_task_head_mode="theorem_feature_scalar",
            tree_theorem_surface_mode="factorized_score_fiber",
            tree_summary_spec_root_mode="factored_theorem_readout",
            tree_theorem_feature_dim=16,
            tree_theorem_feature_hidden_dim=32,
            tree_theorem_score_dim=1,
            tree_theorem_fiber_dim=15,
            local_law_weight=0.5,
        )
        result = run_markov_changepoint_ops_count_experiment(config)
        learned = result.metrics["learned"]

        assert result.config["tree_model_version"] == "v2"
        assert result.config["tree_batch_runtime_mode"] == "unified_v2"
        assert result.config["tree_theorem_surface_mode"] == "factorized_score_fiber"
        assert result.config["tree_task_head_mode"] == "theorem_feature_scalar"
        assert result.objective["local_law_c1_weight"] == pytest.approx(1.0 / 6.0)
        assert result.objective["local_law_c2_weight"] == pytest.approx(1.0 / 6.0)
        assert result.objective["local_law_c3_weight"] == pytest.approx(1.0 / 6.0)
        assert learned["root_mae"] >= 0.0
        assert learned["c2_idempotence_mae"] >= 0.0
        assert learned["n_docs"] == 4

    def test_forward_doc_unified_keeps_document_target_separate_from_node_sampling(self):
        docs = _make_tiny_docs(n=1, seq_len=32, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=32.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )
        out = model.forward_doc_unified(
            fno_docs[0].leaf_token_ids,
            fno_docs[0].leaf_counts,
            fno_docs[0].merge_counts_balanced,
            fno_docs[0].root_count,
            doc_id="doc_0",
            schedule="balanced",
            device=torch.device("cpu"),
            sampled_leaf_indices=set(),
            sampled_internal_indices=set(),
            leaf_propensity=0.0,
            internal_propensity=0.0,
            collect_full_trace=True,  # this test reads node_records / state_tree
        )
        node_records = list(out["node_records"])
        root_records = [record for record in node_records if bool(record.is_root)]
        assert len(root_records) == 1
        assert root_records[0].sampled is False
        assert root_records[0].propensity == pytest.approx(0.0)
        assert out["n_sampled_nodes"] == 0
        assert out["document_record"].target == pytest.approx(fno_docs[0].root_count / 32.0)
        assert int(out["all_node_preds"].shape[0]) == len(node_records)
        assert int(out["all_node_proxy_targets"].shape[0]) == len(node_records)
        assert int(out["all_node_oracle_targets"].shape[0]) == len(node_records)
        assert float(out["all_node_observed"].sum().detach().cpu()) == pytest.approx(0.0)
        assert torch.allclose(
            out["all_node_proxy_targets"],
            out["all_node_oracle_targets"],
        )
        trace = out["state_tree"]
        assert trace.root.metadata["doc_id"] == "doc_0"
        assert trace.node_count == len(node_records)
        assert trace.root.metadata["state_kind"] == "markov_fno_state"
        assert trace.root.metadata["observed"] is False

    def test_forward_doc_unified_collect_full_trace_false_skips_telemetry(self):
        """Default collect_full_trace=False skips per-node telemetry sync.

        The training/eval hot path takes this branch. GPU-tensor outputs
        (document_pred_norm, all_node_preds, etc.) must still be correct;
        only the per-node FullTreeNodeRecord / StateTree side-channel is
        skipped to avoid the GPU->CPU sync per node.
        """
        docs = _make_tiny_docs(n=1, seq_len=32, vocab_size=8)
        fno_docs = _prepare_fno_count_docs(docs, leaf_tokens=8)
        model = FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=8,
            hidden_dim=16,
            target_scale=32.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
        )
        kwargs = dict(
            leaf_token_ids=fno_docs[0].leaf_token_ids,
            leaf_counts=fno_docs[0].leaf_counts,
            merge_counts_balanced=fno_docs[0].merge_counts_balanced,
            root_count=fno_docs[0].root_count,
            doc_id="doc_0",
            schedule="balanced",
            device=torch.device("cpu"),
            sampled_leaf_indices=None,
            sampled_internal_indices=None,
            leaf_propensity=1.0,
            internal_propensity=1.0,
        )
        out_full = model.forward_doc_unified(**kwargs, collect_full_trace=True)
        out_fast = model.forward_doc_unified(**kwargs)  # default = False
        # Telemetry side-channel skipped under default.
        assert out_fast["node_records"] == ()
        assert out_fast["document_record"] is None
        assert out_fast["state_tree"] is None
        # GPU-tensor outputs match the telemetry path exactly.
        assert torch.allclose(out_fast["document_pred_norm"], out_full["document_pred_norm"])
        assert torch.allclose(out_fast["all_node_preds"], out_full["all_node_preds"])
        assert torch.allclose(out_fast["all_node_proxy_targets"], out_full["all_node_proxy_targets"])
        assert torch.allclose(out_fast["all_node_oracle_targets"], out_full["all_node_oracle_targets"])
        assert torch.allclose(out_fast["root_pred_count"], out_full["root_pred_count"])


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_unified_g_carrier_projection_reencodes_exact_summary_without_wide_padding():
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="carrier_projection",
        tree_model_version="unified_g",
    )

    assert model.unified_g_summary_dim > model.summary_dim
    summary = torch.tensor(
        [0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=torch.float32,
    )
    decoded = model.decode_summary(model.encode_summary(summary))

    assert torch.allclose(decoded, summary, atol=1e-6)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_carrier_projection_rejects_noncanonical_direct_slot_dims():
    with pytest.raises(ValueError, match="direct Markov sketch slots"):
        FNOCountSketch(
            vocab_size=8,
            leaf_tokens=8,
            state_dim=16,
            hidden_dim=32,
            target_scale=8.0,
            n_regimes=4,
            fno_width=8,
            fno_n_modes=4,
            fno_n_layers=1,
            summary_spec_name="markov_count_sketch",
            slot_count=4,
            theorem_surface_mode="carrier_projection",
            theorem_count_dim=8,
            theorem_first_dim=8,
            theorem_last_dim=8,
        )


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_carrier_projection_dense_leaf_batch_matches_flat_leaf_path():
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=4,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="carrier_projection",
    )
    model.eval()
    tokens = torch.tensor(
        [
            [[1, 2, 3, 4], [2, 3, 8, 8]],
            [[3, 4, 5, 8], [1, 1, 1, 1]],
        ],
        dtype=torch.long,
    )
    mask = tokens.ne(8).to(dtype=torch.float32)

    dense_states = model.encode_leaf_tokens_batch(
        tokens,
        token_mask=mask,
        device=torch.device("cpu"),
    ).reshape(int(tokens.shape[0]), int(tokens.shape[1]), -1)
    flat_states = model.encode_leaf_tokens_batch(
        tokens.reshape(-1, int(tokens.shape[-1])),
        token_mask=mask.reshape(-1, int(mask.shape[-1])),
        device=torch.device("cpu"),
    ).reshape(int(tokens.shape[0]), int(tokens.shape[1]), -1)

    assert torch.allclose(dense_states, flat_states, atol=1e-6)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_carrier_projection_merge_canonicalizes_endpoint_logits_before_merging():
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=16,
        hidden_dim=32,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="carrier_projection",
    )
    left = model.encode_summary(
        torch.tensor(
            [0.25, 0.1, 4.0, 0.2, -1.0, -0.5, 0.2, 3.0, 0.1],
            dtype=torch.float32,
        )
    )
    right = model.encode_summary(
        torch.tensor(
            [0.125, -0.1, 0.3, 2.5, 0.0, 0.4, -0.2, 0.1, 1.5],
            dtype=torch.float32,
        )
    )
    merged = model._merge_summary_spec_states(left, right)
    decoded = model.decode_summary(merged)

    assert torch.argmax(decoded[1:5]).item() == 1
    assert torch.argmax(decoded[5:9]).item() == 3
    assert torch.allclose(decoded[1:5], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(decoded[5:9], torch.tensor([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_carrier_projection_runtime_count_discretization_rounds_merge_slot():
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=4,
        state_dim=16,
        hidden_dim=8,
        target_scale=10.0,
        n_regimes=2,
        fno_width=4,
        fno_n_modes=2,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_surface_mode="carrier_projection",
        runtime_count_discretization="st_round",
    )
    assert model.count_slot_merger is not None
    with torch.no_grad():
        for param in model.count_slot_merger.parameters():
            param.zero_()
        final_linear = model.count_slot_merger[-1]
        final_linear.bias.fill_(0.26)

    left = model.encode_summary(
        torch.tensor([[0.1, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    )
    right = model.encode_summary(
        torch.tensor([[0.1, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    )
    merged = model._merge_state_pairs(left, right)

    assert float(model._count_slot(merged).item()) == pytest.approx(0.3)
    assert float(model.predict_count_from_state(merged).item()) == pytest.approx(3.0)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_opaque_carrier_exact_sketch_uses_explicit_carrier_merge_and_sketch_readout():
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=128,
        hidden_dim=64,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode="opaque_carrier_exact_sketch",
        theorem_feature_dim=128,
        theorem_feature_hidden_dim=256,
        score_merge_mode="exact_projected_sketch",
        merge_hidden_dim=256,
    )

    assert model.carrier_state_dim == 128
    assert model.state_dim == 128 + 1 + 2 * 4
    assert model.summary_state_merger is None
    assert model.carrier_state_merger is not None
    assert int(model.carrier_state_merger[0].in_features) == 256
    assert int(model.carrier_state_merger[-1].out_features) == 128

    summary = torch.tensor(
        [0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=torch.float32,
    )
    state = model.encode_summary(summary)
    decoded = model.decode_summary(state)
    assert torch.allclose(decoded, summary, atol=1e-6)

    mutated = state.clone()
    mutated[..., model._residual_slice] = mutated[..., model._residual_slice] + 5.0
    assert torch.allclose(
        model.predict_task_count_from_state(mutated),
        model.predict_task_count_from_state(state),
        atol=1e-6,
    )

    left_summary = torch.tensor(
        [0.125, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    right_summary = torch.tensor(
        [0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=torch.float32,
    )
    merged_state = model._exact_projected_merge_state(
        model.encode_summary(left_summary),
        model.encode_summary(right_summary),
    )
    merged_decoded = model.decode_summary(merged_state)
    expected_count = (1.0 + 2.0 + 1.0) / 8.0
    assert float(merged_decoded[0]) == pytest.approx(expected_count, abs=1e-6)
    assert torch.argmax(merged_decoded[1:5]).item() == 0
    assert torch.argmax(merged_decoded[5:9]).item() == 3


def test_opaque_carrier_exact_sketch_root_matches_exact_remerge_without_clamp():
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=64,
        hidden_dim=64,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode="opaque_carrier_exact_sketch",
        theorem_feature_dim=64,
        theorem_feature_hidden_dim=128,
        score_merge_mode="exact_projected_sketch",
        merge_hidden_dim=128,
    )

    leaf_summaries = [
        torch.tensor([0.6, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.6, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.6, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    ]
    leaf_states = [model.encode_summary(summary) for summary in leaf_summaries]
    root_state, _merge_states = model._merge_states(
        leaf_states,
        schedule="balanced",
        collect_merge_states=True,
    )
    root_direct = float(model.predict_count_from_state(root_state).detach().cpu().item())
    root_exact = float(
        nob._exact_projected_root_count_from_states(
            model,
            leaf_states,
            schedule="balanced",
        )
    )

    assert root_exact > float(model.target_scale)
    assert root_direct == pytest.approx(root_exact, abs=1e-6)


def test_exact_sketch_selection_uses_certified_root_for_exact_projected_merge():
    class _DummyModel:
        def __init__(self, use_exact_projected_sketch_merge: bool) -> None:
            self.use_exact_projected_sketch_merge = use_exact_projected_sketch_merge

    exact_merge_model = _DummyModel(True)
    learned_merge_model = _DummyModel(False)

    assert nob._exact_sketch_selection_root_mae(
        exact_merge_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=1.75,
    ) == pytest.approx(1.75)
    assert nob._exact_sketch_selection_root_mae(
        learned_merge_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=1.75,
    ) == pytest.approx(0.25)
    assert nob._exact_sketch_selection_root_mae(
        exact_merge_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=float("nan"),
    ) == pytest.approx(0.25)


def test_exact_sketch_selection_split_penalty_only_applies_to_exact_merge_lane():
    class _DummyModel:
        def __init__(self, use_exact_projected_sketch_merge: bool) -> None:
            self.use_exact_projected_sketch_merge = use_exact_projected_sketch_merge

    exact_merge_model = _DummyModel(True)
    learned_merge_model = _DummyModel(False)

    assert nob._exact_sketch_selection_split_penalty(
        exact_merge_model,
        root_mae_oracle_counts_predicted_endpoints=3.0,
        root_mae_predicted_counts_oracle_endpoints=1.0,
    ) == pytest.approx(2.0)
    assert nob._exact_sketch_selection_split_penalty(
        learned_merge_model,
        root_mae_oracle_counts_predicted_endpoints=3.0,
        root_mae_predicted_counts_oracle_endpoints=1.0,
    ) == pytest.approx(0.0)
    assert nob._exact_sketch_selection_split_penalty(
        exact_merge_model,
        root_mae_oracle_counts_predicted_endpoints=float("nan"),
        root_mae_predicted_counts_oracle_endpoints=1.0,
    ) == pytest.approx(0.0)


def test_exact_sketch_selection_exact_merge_fallback_detects_serialized_modes():
    class _DummyModel:
        def __init__(
            self,
            *,
            score_merge_mode: str = "",
            theorem_surface_mode: str = "",
        ) -> None:
            self.score_merge_mode = score_merge_mode
            self.theorem_surface_mode = theorem_surface_mode

    score_mode_model = _DummyModel(score_merge_mode="exact_projected_sketch")
    surface_mode_model = _DummyModel(
        theorem_surface_mode="opaque_carrier_exact_sketch"
    )

    assert nob._exact_sketch_selection_root_mae(
        score_mode_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=1.75,
    ) == pytest.approx(1.75)
    assert nob._exact_sketch_selection_split_penalty(
        score_mode_model,
        root_mae_oracle_counts_predicted_endpoints=3.0,
        root_mae_predicted_counts_oracle_endpoints=1.0,
    ) == pytest.approx(2.0)
    assert nob._exact_sketch_selection_root_mae(
        surface_mode_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=1.75,
    ) == pytest.approx(1.75)

    stale_tree_prefixed_model = _DummyModel()
    stale_tree_prefixed_model.tree_score_merge_mode = "exact_projected_sketch"

    assert nob._exact_sketch_selection_root_mae(
        stale_tree_prefixed_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=1.75,
    ) == pytest.approx(0.25)
    assert nob._exact_sketch_selection_split_penalty(
        stale_tree_prefixed_model,
        root_mae_oracle_counts_predicted_endpoints=3.0,
        root_mae_predicted_counts_oracle_endpoints=1.0,
    ) == pytest.approx(0.0)

    runtime_model = _DummyModel()
    runtime_model.runtime_merge_kind = "exact_projected_sketch"
    assert nob._exact_sketch_selection_root_mae(
        runtime_model,
        root_direct_count_mae=0.25,
        exact_projected_root_mae=1.75,
    ) == pytest.approx(1.75)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_unified_g_exact_projected_label_still_uses_learned_runtime_merge():
    torch.manual_seed(0)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=4,
        state_dim=32,
        hidden_dim=64,
        target_scale=8.0,
        n_regimes=3,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode="factorized_score_fiber",
        theorem_feature_dim=24,
        theorem_feature_hidden_dim=48,
        theorem_score_dim=1,
        theorem_fiber_dim=23,
        theorem_aux_dim=0,
        score_merge_mode="exact_projected_sketch",
        tree_model_version="unified_g",
    )

    left = torch.randn(2, model.state_dim)
    right = torch.randn(2, model.state_dim)
    learned = model._merge_state_pairs(left, right)
    exact = model._exact_projected_merge_state(left, right)

    assert model.uses_unified_g_learned_merge is True
    assert model.exact_projected_merge_is_runtime_merge is False
    assert model.runtime_merge_kind == "learned_unified_g"
    assert nob._selection_uses_exact_projected_merge(model) is False
    assert learned.shape == exact.shape
    assert not torch.allclose(learned.detach(), exact.detach())

    serialized = type("SerializedUnifiedG", (), {})()
    serialized.tree_model_version = "unified_g"
    serialized.use_exact_projected_sketch_merge = True
    assert nob._selection_uses_exact_projected_merge(serialized) is False


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_unified_g_c3_local_law_backprops_through_learned_merge_projector():
    torch.manual_seed(1)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=4,
        state_dim=32,
        hidden_dim=64,
        target_scale=8.0,
        n_regimes=3,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode="factorized_score_fiber",
        theorem_feature_dim=24,
        theorem_feature_hidden_dim=48,
        theorem_score_dim=1,
        theorem_fiber_dim=23,
        theorem_aux_dim=0,
        score_merge_mode="exact_projected_sketch",
        tree_model_version="unified_g",
    )

    left = torch.randn(3, model.state_dim, requires_grad=True)
    right = torch.randn(3, model.state_dim, requires_grad=True)
    parent = model._merge_state_pairs(left, right)
    terms = nob._summary_spec_merge_consistency_terms(
        model,
        left,
        right,
        parent,
        truth_join_bit=1,
    )

    assert terms["total_loss"].requires_grad
    model.zero_grad(set_to_none=True)
    terms["total_loss"].backward()
    grad_norm = 0.0
    assert model.unified_g_merge_summary_proj is not None
    for param in model.unified_g_merge_summary_proj.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().abs().sum().cpu())
    assert grad_norm > 0.0


def test_exact_markov_root_error_decomposition_separates_count_and_endpoint_failures():
    count_failure = nob._exact_markov_root_error_decomposition(
        truth_root_count=1.0,
        predicted_counts=[1.0, 0.0],
        predicted_first=[0, 1],
        predicted_last=[0, 1],
        truth_counts=[0.0, 0.0],
        truth_first=[0, 1],
        truth_last=[0, 1],
        schedule="balanced",
    )
    assert count_failure["root_mae_predicted_counts_oracle_endpoints"] == pytest.approx(1.0)
    assert count_failure["root_mae_oracle_counts_predicted_endpoints"] == pytest.approx(0.0)

    endpoint_failure = nob._exact_markov_root_error_decomposition(
        truth_root_count=1.0,
        predicted_counts=[0.0, 0.0],
        predicted_first=[0, 0],
        predicted_last=[0, 0],
        truth_counts=[0.0, 0.0],
        truth_first=[0, 1],
        truth_last=[0, 1],
        schedule="balanced",
    )
    assert endpoint_failure["root_mae_predicted_counts_oracle_endpoints"] == pytest.approx(0.0)
    assert endpoint_failure["root_mae_oracle_counts_predicted_endpoints"] == pytest.approx(1.0)


@pytest.mark.skipif(not HAS_NEURAL_OPERATOR, reason="neuraloperator not installed")
def test_opaque_carrier_exact_sketch_direct_metrics_are_finite():
    torch.manual_seed(0)
    np.random.seed(0)
    raw_docs = _make_tiny_docs(n=2, seq_len=16, vocab_size=8)
    fno_docs = _prepare_fno_count_docs(raw_docs, leaf_tokens=8)
    model = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=64,
        hidden_dim=64,
        target_scale=8.0,
        n_regimes=4,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode="opaque_carrier_exact_sketch",
        theorem_feature_dim=64,
        theorem_feature_hidden_dim=128,
        score_merge_mode="exact_projected_sketch",
        merge_hidden_dim=256,
    )

    metrics = nob._eval_fno_exact_sketch_direct_metrics(
        model,
        fno_docs,
        device=torch.device("cpu"),
        phi_pair_calibration_max_nodes=None,
    )

    for key in (
        "exact_projected_root_mae",
        "certified_projected_root_mae",
        "root_mae_predicted_counts_predicted_endpoints",
        "root_mae_oracle_counts_predicted_endpoints",
        "root_mae_predicted_counts_oracle_endpoints",
        "learned_merger_gap",
        "c2_on_range_exact_match",
        "leaf_first_accuracy",
        "leaf_last_accuracy",
        "merge_first_accuracy",
        "merge_last_accuracy",
    ):
        assert np.isfinite(float(metrics[key])), key
    assert isinstance(metrics["leaf_count_off_by_k_histogram"], dict)
    assert isinstance(metrics["merge_exact_summary_match_rate_by_depth"], dict)
