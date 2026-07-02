from __future__ import annotations

from pathlib import Path

import pytest

from src.ctreepo.sim.core.full_doc_config_codec import (
    FULL_DOC_CONFIG_ALIAS_PAIRS,
    canonicalize_full_doc_config_mapping,
    migrate_legacy_public_run_axis_config,
    public_run_axis_from_config_like,
    runtime_config_overrides_from_config_like,
    serialize_full_doc_runtime_config,
    serialize_tree_run_config,
)


def test_runtime_config_overrides_rejects_public_tree_objective_aliases() -> None:
    with pytest.raises(ValueError, match="tree_local_law_weight"):
        runtime_config_overrides_from_config_like(
            {
                "tree_local_law_weight": 0.8,
                "tree_c1_relative_weight": 0.0,
                "tree_c2_relative_weight": 1.0,
                "tree_c3_relative_weight": 0.0,
                "tree_document_loss_normalization_mode": "supervised_docs",
                "tree_supervision_source": "manifest",
            }
        )


def test_runtime_config_overrides_uses_canonical_objective_names() -> None:
    payload = runtime_config_overrides_from_config_like(
        {
            "local_law_weight": 0.8,
            "c1_relative_weight": 0.0,
            "c2_relative_weight": 1.0,
            "c3_relative_weight": 0.0,
            "tree_document_loss_normalization_mode": "supervised_docs",
            "tree_supervision_source": "manifest",
        }
    )

    assert payload["local_law_weight"] == 0.8
    assert payload["c1_relative_weight"] == 0.0
    assert payload["c2_relative_weight"] == 1.0
    assert payload["c3_relative_weight"] == 0.0
    assert payload["tree_document_loss_normalization_mode"] == "supervised_docs"
    assert payload["tree_supervision_source"] == "manifest"
    assert "tree_local_law_weight" not in payload
    assert "tree_task_objective_weight" not in payload


def test_serialize_full_doc_runtime_config_is_json_safe_and_merges_metadata(
    tmp_path: Path,
) -> None:
    payload = serialize_full_doc_runtime_config(
        {
            "local_law_weight": 0.8,
            "prepared_data_root": tmp_path,
            "grid_cell_ids": ("r4_seg4to6", "r12_seg10to12"),
        },
        metadata={"comparison_mode": "comparable"},
    )

    assert payload["local_law_weight"] == 0.8
    assert payload["prepared_data_root"] == str(tmp_path)
    assert payload["grid_cell_ids"] == ["r4_seg4to6", "r12_seg10to12"]
    assert payload["comparison_mode"] == "comparable"
    assert "tree_local_law_weight" not in payload


def test_serialize_tree_run_config_emits_canonical_objective_keys(
    tmp_path: Path,
) -> None:
    payload = serialize_tree_run_config(
        {
            "method_id": "official_fno_sumlen",
            "problem_id": "markov_ops_count",
            "local_law_weight": 0.8,
            "artifact_dir": tmp_path,
        }
    )

    assert payload["method_id"] == "official_fno_sumlen"
    assert payload["local_law_weight"] == 0.8
    assert payload["artifact_dir"] == str(tmp_path)
    assert "tree_local_law_weight" not in payload
    assert "root_share" not in payload


def test_serialize_tree_run_config_allows_explicit_root_share_without_lambda(
    tmp_path: Path,
) -> None:
    payload = serialize_tree_run_config(
        {
            "method_id": "official_fno_sumlen",
            "problem_id": "markov_ops_count",
            "root_share": 0.2,
            "artifact_dir": tmp_path,
        }
    )

    assert payload["root_share"] == 0.2
    assert "tree_local_law_weight" not in payload
    assert "tree_task_objective_weight" not in payload


def test_lambda_and_explicit_root_weight_hybrid_raises() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        canonicalize_full_doc_config_mapping(
            {"local_law_weight": 0.8, "root_share": 0.2}
        )


def test_alias_conflict_raises_when_both_differ() -> None:
    with pytest.raises(ValueError, match="Config alias conflict"):
        canonicalize_full_doc_config_mapping(
            {"tree_c1_relative_weight": 0.5, "c1_relative_weight": 0.3}
        )


def test_alias_no_conflict_when_both_match() -> None:
    result = canonicalize_full_doc_config_mapping(
        {"tree_c1_relative_weight": 0.5, "c1_relative_weight": 0.5}
    )
    assert result["c1_relative_weight"] == 0.5
    assert "tree_c1_relative_weight" not in result


def test_alias_sync_when_one_side_empty() -> None:
    result = canonicalize_full_doc_config_mapping(
        {"tree_c2_relative_weight": 1.0}
    )
    assert result["c2_relative_weight"] == 1.0
    assert "tree_c2_relative_weight" not in result


def test_alias_conflict_all_pairs() -> None:
    for tree_key, runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
        if tree_key in {"tree_local_law_weight", "tree_task_objective_weight"}:
            continue
        with pytest.raises(ValueError, match="Config alias conflict"):
            canonicalize_full_doc_config_mapping(
                {tree_key: 0.1, runtime_key: 0.9}
            )


def test_legacy_objective_config_fields_reject() -> None:
    for key in (
        "task_objective_weight",
        "tree_local_law_weight",
        "tree_task_objective_weight",
    ):
        with pytest.raises(ValueError, match=key):
            canonicalize_full_doc_config_mapping({key: 0.5})


def test_public_run_axis_rejects_legacy_config_fields() -> None:
    with pytest.raises(ValueError, match="baseline_family"):
        canonicalize_full_doc_config_mapping({"baseline_family": "tree_neural"})
    with pytest.raises(ValueError, match="tree_neural_c2"):
        public_run_axis_from_config_like(
            {
                "problem_id": "markov_ops_count",
                "method_id": "tree_neural_c2",
            }
        )


def test_public_run_axis_migration_is_explicit() -> None:
    payload = migrate_legacy_public_run_axis_config(
        {"baseline_family": "tree_neural_c2", "law_package": "tree_c2_only"}
    )

    assert payload["method_id"] == "tree_neural"
    assert payload["law_set_id"] == "on_range_idempotence_only"
