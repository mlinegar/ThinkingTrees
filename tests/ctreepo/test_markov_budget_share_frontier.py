from __future__ import annotations

import pytest

from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    _doc_leaf_and_internal_spans,
    build_budgeted_train_supervision_manifest,
    build_markov_changepoint_ops_count_data_bundle,
)


def _tiny_budget_docs():
    cfg = OPSCountConfig(
        train_docs=4,
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        use_cuda=False,
        seed=0,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    return bundle.train_docs


def _manifest_local_call_counts(manifest) -> tuple[int, int]:
    leaf_calls = sum(len(tuple(plan.leaf_indices)) for plan in manifest.doc_plans)
    internal_calls = sum(len(tuple(plan.internal_indices)) for plan in manifest.doc_plans)
    return int(leaf_calls), int(internal_calls)


def test_budget_manifest_accounts_full_doc_and_local_calls_exactly() -> None:
    docs = _tiny_budget_docs()
    cfg = OPSCountConfig(
        train_docs=len(docs),
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        budget_total_calls=8,
        full_doc_budget_share=0.25,
        doc_consumption_mode="root_only",
        local_split_mode="balanced",
        local_allocation_policy="breadth_first",
        use_cuda=False,
    )

    manifest = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=cfg,
        baseline_family="tree_neural",
        seed=0,
    )

    assert manifest is not None
    assert manifest.budget_total_calls == 8
    assert manifest.full_doc_calls_total == 2
    assert manifest.local_calls_total == 6
    assert manifest.budget_total_calls_used == 8


def test_budget_manifest_rejects_non_tree_reference_when_share_is_not_one() -> None:
    docs = _tiny_budget_docs()
    cfg = OPSCountConfig(
        train_docs=len(docs),
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        budget_total_calls=4,
        full_doc_budget_share=0.5,
        doc_consumption_mode="full_doc_only",
        use_cuda=False,
    )

    with pytest.raises(ValueError, match="requires full_doc_budget_share=1.0"):
        build_budgeted_train_supervision_manifest(
            docs=docs,
            config=cfg,
            baseline_family="official_fno",
            seed=0,
        )


def test_budget_manifest_effective_mass_examples_match_leaf_internal_and_document_units() -> None:
    leaf_spans, internal_spans = _doc_leaf_and_internal_spans(
        n_tokens=64,
        leaf_tokens=16,
    )
    leaf_lengths = {int(end - start) for start, end in leaf_spans}
    internal_lengths = {int(end - start) for start, end in internal_spans}

    assert 16 in leaf_lengths
    assert 32 in internal_lengths
    assert 64 in internal_lengths
    assert 16 / 64 == pytest.approx(0.25)
    assert 32 / 64 == pytest.approx(0.5)
    assert 64 / 64 == pytest.approx(1.0)


def test_document_budget_counts_once_for_root_only_and_doc_sequence() -> None:
    docs = _tiny_budget_docs()
    base = dict(
        train_docs=len(docs),
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        budget_total_calls=4,
        full_doc_budget_share=1.0,
        local_split_mode="balanced",
        local_allocation_policy="breadth_first",
        use_cuda=False,
    )

    root_manifest = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=OPSCountConfig(**{**base, "doc_consumption_mode": "root_only"}),
        baseline_family="tree_neural",
        seed=0,
    )
    docseq_manifest = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=OPSCountConfig(**{**base, "doc_consumption_mode": "doc_sequence"}),
        baseline_family="tree_neural",
        seed=0,
    )

    assert root_manifest is not None and docseq_manifest is not None
    assert root_manifest.full_doc_calls_total == 4
    assert docseq_manifest.full_doc_calls_total == 4
    assert root_manifest.local_calls_total == 0
    assert docseq_manifest.local_calls_total == 0
    assert root_manifest.budget_total_calls_used == docseq_manifest.budget_total_calls_used


def test_budget_manifest_realizes_balanced_and_heavy_local_splits() -> None:
    docs = _tiny_budget_docs()
    common = dict(
        train_docs=len(docs),
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        budget_total_calls=8,
        full_doc_budget_share=0.0,
        doc_consumption_mode="root_only",
        local_allocation_policy="breadth_first",
        use_cuda=False,
    )

    balanced = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=OPSCountConfig(**{**common, "local_split_mode": "balanced"}),
        baseline_family="tree_neural",
        seed=0,
    )
    leaf_heavy = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=OPSCountConfig(**{**common, "local_split_mode": "leaf_heavy"}),
        baseline_family="tree_neural",
        seed=0,
    )
    internal_heavy = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=OPSCountConfig(**{**common, "local_split_mode": "internal_heavy"}),
        baseline_family="tree_neural",
        seed=0,
    )

    assert balanced is not None and leaf_heavy is not None and internal_heavy is not None
    assert _manifest_local_call_counts(balanced) == (4, 4)
    assert _manifest_local_call_counts(leaf_heavy) == (6, 2)
    assert _manifest_local_call_counts(internal_heavy) == (2, 6)


def test_budget_manifest_samples_local_units_seeded_without_replacement() -> None:
    docs = _tiny_budget_docs()
    cfg = OPSCountConfig(
        train_docs=len(docs),
        val_docs=0,
        test_docs=0,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=16,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        budget_total_calls=8,
        full_doc_budget_share=0.0,
        doc_consumption_mode="root_only",
        local_split_mode="balanced",
        local_allocation_policy="breadth_first",
        use_cuda=False,
    )

    seed0 = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=cfg,
        baseline_family="tree_neural",
        seed=0,
    )
    seed1 = build_budgeted_train_supervision_manifest(
        docs=docs,
        config=cfg,
        baseline_family="tree_neural",
        seed=1,
    )

    assert seed0 is not None and seed1 is not None
    assert seed0.sampling_scheme == "seeded_random_without_replacement"
    assert seed1.sampling_scheme == "seeded_random_without_replacement"
    assert seed0.doc_plans != seed1.doc_plans
    for manifest in (seed0, seed1):
        for plan in manifest.doc_plans:
            assert len(plan.leaf_indices) == len(set(plan.leaf_indices))
            assert len(plan.internal_indices) == len(set(plan.internal_indices))
            assert all(0 <= int(index) < 4 for index in plan.leaf_indices)
            assert all(0 <= int(index) < 3 for index in plan.internal_indices)
            if plan.leaf_indices:
                assert plan.leaf_propensity == pytest.approx(0.25)
            if plan.internal_indices:
                assert plan.internal_propensity == pytest.approx(1.0 / 3.0)
