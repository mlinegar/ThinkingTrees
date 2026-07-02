from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_expert_anchor_dspy_objective.py"
SPEC = importlib.util.spec_from_file_location("audit_expert_anchor_dspy_objective", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def _summary() -> dict:
    return {
        "role": "g",
        "count": 210,
        "tree_count": 105,
        "total_weight": 105.0,
        "observed_target_count": 105,
        "by_anchor_text_source": {"stored_summary": 105},
        "by_law_role": {
            "full_doc_g_anchor": {"count": 105, "weight": 78.75},
            "leaf_g": {"count": 105, "weight": 26.25},
        },
        "by_target_source": {
            "expert:expert_score_1_7": 105,
            "teacher_node_score": 105,
        },
        "objective": {
            "root_label_sources": ["stored_summary"],
            "root_label_target": "expert",
            "root_share": 0.75,
            "local_law_weight": 0.25,
            "local_law_component_weights": {"teacher_node": 0.25},
            "node_weight_normalization": "per_tree",
            "target_min": 1.0,
            "target_max": 7.0,
            "scorer_output_min": 1.0,
            "scorer_output_max": 7.0,
        },
    }


def test_audit_accepts_expert_anchor_smoke_summary_shape() -> None:
    report = audit_module.audit_summary(_summary())

    assert report["status"] == "ok"
    assert report["anchor_count"] == 105
    assert report["anchor_weight"] == pytest.approx(78.75)
    assert report["teacher_weight"] == pytest.approx(26.25)
    assert report["expected_teacher_weight"] == pytest.approx(26.25)


def test_audit_accepts_root_only_endpoint() -> None:
    payload = _summary()
    payload.update(
        {
            "count": 105,
            "total_weight": 105.0,
            "observed_target_count": 105,
            "by_law_role": {
                "full_doc_g_anchor": {"count": 105, "weight": 105.0},
            },
            "by_target_source": {
                "expert:expert_score_1_7": 105,
            },
        }
    )
    payload["objective"]["root_share"] = 1.0
    payload["objective"]["local_law_weight"] = 0.0
    payload["objective"]["local_law_component_weights"] = {"teacher_node": 0.0}

    report = audit_module.audit_summary(payload, local_law_weight=0.0)

    assert report["anchor_weight"] == pytest.approx(105.0)
    assert report["teacher_count"] == 0
    assert report["teacher_weight"] == pytest.approx(0.0)


def test_audit_accepts_teacher_only_endpoint_with_tree_count() -> None:
    payload = _summary()
    payload.update(
        {
            "count": 105,
            "total_weight": 105.0,
            "observed_target_count": 0,
            "by_anchor_text_source": {},
            "by_law_role": {
                "leaf_g": {"count": 105, "weight": 105.0},
            },
            "by_target_source": {
                "teacher_node_score": 105,
            },
        }
    )
    payload["objective"]["root_share"] = 0.0
    payload["objective"]["local_law_weight"] = 1.0
    payload["objective"]["local_law_component_weights"] = {"teacher_node": 1.0}

    report = audit_module.audit_summary(payload, local_law_weight=1.0)

    assert report["anchor_count"] == 0
    assert report["teacher_weight"] == pytest.approx(105.0)
    assert report["expected_teacher_weight"] == pytest.approx(105.0)


def test_audit_rejects_teacher_weight_not_normalized_per_tree() -> None:
    payload = _summary()
    payload["by_law_role"]["leaf_g"]["weight"] = 105.0
    payload["total_weight"] = 210.0

    with pytest.raises(audit_module.SummaryAuditError, match="teacher-node/local-law"):
        audit_module.audit_summary(payload)


def test_audit_accepts_both_root_label_sources_without_splitting_root_share() -> None:
    payload = _summary()
    payload["count"] = 315
    payload["observed_target_count"] = 210
    payload["total_weight"] = 183.75
    payload["by_anchor_text_source"] = {"stored_summary": 105, "raw_document": 105}
    payload["by_law_role"]["full_doc_g_anchor"] = {"count": 210, "weight": 157.5}
    payload["by_target_source"]["expert:expert_score_1_7"] = 210
    payload["objective"]["root_label_sources"] = ["stored_summary", "raw_document"]

    report = audit_module.audit_summary(
        payload,
        root_label_sources=("stored_summary", "raw_document"),
    )

    assert report["anchor_count"] == 210
    assert report["anchor_weight"] == pytest.approx(157.5)
    assert report["tree_count"] == 105


def test_audit_rejects_teacher_as_full_doc_target() -> None:
    payload = _summary()
    payload["objective"]["root_label_target"] = "teacher"

    with pytest.raises(audit_module.SummaryAuditError, match="root_label_target"):
        audit_module.audit_summary(payload)
