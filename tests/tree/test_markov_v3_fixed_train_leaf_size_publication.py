from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_markov_v3_fixed_train_leaf_size_publication import (
    _empirical_bayes_baseline_from_row,
    _preferred_available_scope_key,
    _scope_presentation,
    _scope_presentation_key,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    ChangepointMarkovDoc,
    MarkovOPSDataBundle,
)


def test_preferred_available_scope_key_prefers_sticky_recoverable_family() -> None:
    recovery = {
        "scopes": {
            "recoverable_v4_t128": {"scope_label": "recoverable_v4_t128"},
            "recoverable_v5_t128": {"scope_label": "recoverable_v5_t128"},
        }
    }

    assert (
        _preferred_available_scope_key(
            recovery,
            ("recoverable_v5_t128", "recoverable_v4_t128"),
        )
        == "recoverable_v5_t128"
    )


def test_scope_presentation_key_falls_back_truthfully_for_legacy_structural_scope() -> None:
    legacy_recovery = {
        "scopes": {
            "r12_seg10to12": {"scope_label": "structural_core_v1_t128::r12_seg10to12"},
        }
    }
    sticky_recovery = {
        "scopes": {
            "r12_seg10to12": {"scope_label": "structural_core_v2_t128::r12_seg10to12"},
        }
    }

    assert (
        _scope_presentation_key(legacy_recovery, primary_scope_key="r12_seg10to12")
        == "r12_seg10to12"
    )
    assert (
        _scope_presentation_key(sticky_recovery, primary_scope_key="r12_seg10to12")
        == "r12_p079"
    )


def test_scope_presentation_key_prefers_new_structural_scope_and_falls_back_to_alias() -> None:
    sticky_recovery = {
        "scopes": {
            "r12_p079": {"scope_label": "structural_core_v2_t128::r12_p079"},
        }
    }
    alias_only_recovery = {
        "scopes": {
            "r12_seg10to12": {"scope_label": "structural_core_v2_t128::r12_seg10to12"},
        }
    }

    assert _scope_presentation_key(sticky_recovery, primary_scope_key="r12_p079") == "r12_p079"
    assert (
        _scope_presentation_key(alias_only_recovery, primary_scope_key="r12_p079")
        == "r12_p079"
    )


def test_scope_presentation_uses_current_sticky_benchmark_facts() -> None:
    recoverable = {
        "scopes": {
            "recoverable_v5_t128": {"scope_label": "recoverable_v5_t128"},
        }
    }
    structural = {
        "scopes": {
            "r12_p079": {"scope_label": "structural_core_v2_t128::r12_p079"},
        }
    }

    recoverable_presentation = _scope_presentation(
        recoverable,
        primary_scope_key="recoverable_v5_t128",
    )
    structural_presentation = _scope_presentation(
        structural,
        primary_scope_key="r12_p079",
    )

    assert "0.03937" in recoverable_presentation["subtitle"]
    assert "about 5 expected regime changes" in recoverable_presentation["subtitle"]
    assert "0.07874" in structural_presentation["subtitle"]
    assert "about 10 expected regime changes" in structural_presentation["subtitle"]


def test_empirical_bayes_baseline_collapses_under_disjoint_palette_sticky_benchmark(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "bundle.json"
    summary_path = tmp_path / "summary.json"
    train_doc = ChangepointMarkovDoc(
        tokens=(0, 0, 4, 4, 8, 8),
        token_regimes=(0, 0, 1, 1, 2, 2),
        transition_regimes=(0, 1, 1, 2, 2),
        true_boundaries=(1, 3),
    )
    test_doc = ChangepointMarkovDoc(
        tokens=(0, 0, 4, 8, 8),
        token_regimes=(0, 0, 1, 2, 2),
        transition_regimes=(0, 1, 2, 2),
        true_boundaries=(1, 2),
    )
    MarkovOPSDataBundle(
        train_docs=(train_doc,),
        val_docs=(),
        test_docs=(test_doc,),
        train_corpus_signature="train",
        val_corpus_signature="val",
        test_corpus_signature="test",
    ).save(bundle_path)
    summary_path.write_text(
        json.dumps(
            {
                "config": {
                    "train_docs": 1,
                    "seed": 0,
                    "budget_total_calls_per_doc": 1.0,
                    "full_doc_budget_share": 1.0,
                    "doc_consumption_mode": "root_only",
                    "local_split_mode": "balanced",
                    "local_allocation_policy": "breadth_first",
                    "fixed_leaf_tokens": 128,
                    "leaf_label_rate": 0.0,
                    "internal_supervision_kind": "none",
                    "internal_label_rate": 0.0,
                },
                "benchmark_spec": {
                    "name": "recoverable_v5_t128",
                },
                "bundle_manifest": {
                    "canonical_bundle_path": str(bundle_path),
                },
            }
        ),
        encoding="utf-8",
    )

    baseline = _empirical_bayes_baseline_from_row(
        {
            "source_summary_json": str(summary_path),
            "scope_key": "recoverable_v5_t128",
            "baseline_family": "tree_neural",
            "train_doc_count": 1,
        }
    )

    assert baseline["posterior_collapse_via_disjoint_palettes"] is True
    assert baseline["reviewed_root_docs"] == 1
    assert baseline["test_root_mae"] == 0.0
