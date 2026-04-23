from __future__ import annotations

from types import SimpleNamespace

import torch

from src.core.unified_runtime import (
    BatchPlanCache,
    GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
    GpuBatchStore,
    GpuBatchStoreKey,
    GpuRuntimeConfig,
    WorkItem,
    build_balanced_topology_plan,
    build_leaf_count_auto_queue_targets,
    build_unified_topology_plan,
    gpu_runtime_config_from_mapping,
    plan_work_batches,
)


def test_build_balanced_topology_plan_tracks_levels_and_final_root() -> None:
    plan = build_balanced_topology_plan(
        doc_index=3,
        doc_id="doc-3",
        leaf_metadata=[
            {"char_count": 120, "token_count": 30},
            {"char_count": 80, "token_count": 20},
            {"char_count": 60, "token_count": 15},
            {"char_count": 40, "token_count": 10},
        ],
    )

    assert plan.doc_id == "doc-3"
    assert plan.leaf_count == 4
    assert plan.internal_count == 3
    assert plan.final_ref.is_internal is True
    assert plan.max_level == 2
    assert plan.plan_summary["topology_kind"] == "balanced_binary"
    assert plan.internal_nodes[0].metadata["estimated_input_tokens"] == 50
    assert sorted(plan.dependents_by_leaf.keys()) == [0, 1, 2, 3]


def test_build_unified_topology_plan_preserves_embedding_tree_shape() -> None:
    nodes = [
        SimpleNamespace(is_leaf=True, level=0, text_span="alpha", char_start=0, char_end=5),
        SimpleNamespace(is_leaf=True, level=0, text_span="beta", char_start=6, char_end=10),
        SimpleNamespace(is_leaf=False, level=1, children=(0, 1), char_start=0, char_end=10),
    ]

    plan = build_unified_topology_plan(
        doc_index=0,
        doc_id="u0",
        nodes=nodes,
    )

    assert plan.leaf_count == 2
    assert plan.internal_count == 1
    assert plan.final_ref.is_internal is True
    assert plan.plan_summary["embedding_tree_node_count"] == 3
    assert plan.internal_nodes[0].metadata["embedding_tree_index"] == 2
    assert plan.internal_nodes[0].left.index == 0
    assert plan.internal_nodes[0].right.index == 1


def test_plan_work_batches_respects_shape_and_doc_budget() -> None:
    items = [
        WorkItem(
            item_id="a",
            backend_family="llm_text",
            op_kind="summarize",
            topology_signature="shape-a",
            doc_id="doc-a",
            estimated_tokens=64,
            estimated_nodes=1,
            padding_length=64,
        ),
        WorkItem(
            item_id="b",
            backend_family="llm_text",
            op_kind="summarize",
            topology_signature="shape-a",
            doc_id="doc-b",
            estimated_tokens=60,
            estimated_nodes=1,
            padding_length=64,
        ),
        WorkItem(
            item_id="c",
            backend_family="llm_text",
            op_kind="summarize",
            topology_signature="shape-b",
            doc_id="doc-c",
            estimated_tokens=32,
            estimated_nodes=1,
            padding_length=32,
        ),
    ]

    batches = plan_work_batches(
        items,
        max_docs=2,
        max_total_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
    )

    assert len(batches) == 2
    assert [item.item_id for item in batches[0].items] == ["a", "b"]
    assert [item.item_id for item in batches[1].items] == ["c"]


def test_plan_work_batches_reuses_cached_group_sizes() -> None:
    cache = BatchPlanCache()
    items = [
        WorkItem(
            item_id=f"leaf-{idx}",
            backend_family="llm_text",
            op_kind="summarize",
            topology_signature="shape-a",
            doc_id=f"doc-{idx}",
            estimated_tokens=32,
            estimated_nodes=1,
            padding_length=32,
        )
        for idx in range(4)
    ]

    first = plan_work_batches(
        items,
        max_docs=2,
        max_total_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        plan_cache=cache,
    )
    second = plan_work_batches(
        items,
        max_docs=2,
        max_total_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        plan_cache=cache,
    )

    assert [len(batch.items) for batch in first] == [2, 2]
    assert [len(batch.items) for batch in second] == [2, 2]
    assert cache.misses >= 1
    assert cache.hits >= 1


def test_build_leaf_count_auto_queue_targets_merges_only_compatible_families() -> None:
    targets = build_leaf_count_auto_queue_targets(
        {
            128: 6,
            256: 6,
            512: 3,
            625: 2,
        },
        structural_pad_limit=0.5,
        min_docs=8,
    )

    assert targets == {
        128: 128,
        256: 256,
        512: 625,
        625: 625,
    }


def test_plan_work_batches_leaf_count_auto_queue_uses_family_docs_cap() -> None:
    items = [
        WorkItem(
            item_id=f"leaf-{leaf_count}",
            backend_family="neural_tree",
            op_kind="full_tree",
            topology_signature=f"shape-{leaf_count}",
            doc_id=f"doc-{leaf_count}",
            estimated_tokens=int(leaf_count * 8),
            estimated_nodes=int((2 * 6) - 1),
            estimated_merge_ops=5,
            padding_length=8,
            padding_multiple=6,
            metadata={
                "leaf_count": int(leaf_count),
                "auto_queue_group_key": "tree|train",
                "auto_queue_target_leaf_count": 6,
                "auto_queue_docs_cap_key": "leaf_count_auto_queue:tree|train:n6",
            },
        )
        for leaf_count in (4, 5, 6)
    ]

    batches = plan_work_batches(
        items,
        max_docs=10,
        max_total_tokens=0,
        max_total_nodes=0,
        max_total_merge_ops=0,
        docs_cap_by_signature={"leaf_count_auto_queue:tree|train:n6": 2},
        bucket_mode=GPU_RUNTIME_BUCKET_MODE_LEAF_COUNT_AUTO_QUEUE,
        structural_pad_limit=0.5,
        auto_queue_min_docs=0,
    )

    assert [len(batch.items) for batch in batches] == [2, 1]
    assert {
        item.item_id
        for batch in batches
        for item in batch.items
    } == {"leaf-4", "leaf-5", "leaf-6"}


def test_gpu_runtime_config_defaults_follow_device_type() -> None:
    cuda_cfg = gpu_runtime_config_from_mapping({}, device_type="cuda")
    cpu_cfg = gpu_runtime_config_from_mapping({}, device_type="cpu")

    assert cuda_cfg.data_mode == "resident"
    assert cpu_cfg.data_mode == "cpu_debug"
    assert cpu_cfg.bucket_mode == "exact_then_bucketed"
    assert cpu_cfg.preload_splits == ("train", "val", "test")


def test_gpu_batch_store_returns_tensor_native_view_for_exact_bucket() -> None:
    cfg = GpuRuntimeConfig()
    store = GpuBatchStore(
        backend_family="neural_tree",
        split_name="train",
        config=cfg,
        device="cpu",
    )
    key = GpuBatchStoreKey(
        backend_family="neural_tree",
        topology_signature="shape-a",
        leaf_count_band=4,
        max_leaf_tokens_band=8,
        work_kind="train",
        supervision_mask="root_leaf",
        exact_layout_signature="shape-a",
    )
    store.add_bucket(
        key=key,
        doc_indices=[0, 1],
        tensors={
            "leaf_tokens": torch.arange(16, dtype=torch.long).reshape(2, 2, 4),
            "leaf_mask": torch.ones((2, 2, 4), dtype=torch.float32),
        },
        metadata={"bucket_name": "shape-a"},
    )

    view = store.view_for_doc_indices([1, 0], pad_values={"leaf_tokens": 0.0})

    assert view is not None
    assert tuple(view.doc_indices) == (1, 0)
    assert tuple(view.tensors["leaf_tokens"].shape) == (2, 2, 4)
    assert int(store.telemetry.resident_store_hits) == 1
    assert int(store.telemetry.resident_store_misses) == 0


def test_gpu_batch_store_view_promotes_shared_dense_bucket_metadata() -> None:
    cfg = GpuRuntimeConfig()
    store = GpuBatchStore(
        backend_family="neural_tree",
        split_name="train",
        config=cfg,
        device="cpu",
    )
    key = GpuBatchStoreKey(
        backend_family="neural_tree",
        topology_signature="shape-a",
        leaf_count_band=4,
        max_leaf_tokens_band=8,
        exact_layout_signature="shape-a",
    )
    store.add_bucket(
        key=key,
        doc_indices=[0, 1],
        tensors={
            "leaf_tokens": torch.arange(16, dtype=torch.long).reshape(2, 2, 4),
        },
        metadata={
            "bucket_store_mode": "dense_resident",
            "resident_layout_mode": "dense_fixed_shape",
            "resident_bucket_bytes": 128,
        },
    )

    view = store.view_for_doc_indices([1, 0], pad_values={"leaf_tokens": 0.0})

    assert view is not None
    assert view.metadata["bucket_store_mode"] == "dense_resident"
    assert view.metadata["resident_layout_mode"] == "dense_fixed_shape"
    assert int(view.metadata["resident_bucket_bytes"]) == 128


def test_gpu_batch_store_rejects_incompatible_bucket_mix() -> None:
    cfg = GpuRuntimeConfig()
    store = GpuBatchStore(
        backend_family="neural_tree",
        split_name="train",
        config=cfg,
        device="cpu",
    )
    key_a = GpuBatchStoreKey(
        backend_family="neural_tree",
        topology_signature="shape-a",
        leaf_count_band=4,
        max_leaf_tokens_band=8,
        exact_layout_signature="shape-a",
    )
    key_b = GpuBatchStoreKey(
        backend_family="neural_tree",
        topology_signature="shape-b",
        leaf_count_band=8,
        max_leaf_tokens_band=16,
        exact_layout_signature="shape-b",
    )
    store.add_bucket(
        key=key_a,
        doc_indices=[0],
        tensors={"leaf_tokens": torch.ones((1, 2, 4), dtype=torch.long)},
    )
    store.add_bucket(
        key=key_b,
        doc_indices=[1],
        tensors={"leaf_tokens": torch.ones((1, 2, 4), dtype=torch.long)},
    )

    view = store.view_for_doc_indices([0, 1], pad_values={"leaf_tokens": 0.0})

    assert view is None
    assert int(store.telemetry.resident_store_hits) == 0
    assert int(store.telemetry.resident_store_misses) == 1
    assert store.telemetry.cpu_fallback_reason_counts["incompatible_bucket_mix"] == 1
