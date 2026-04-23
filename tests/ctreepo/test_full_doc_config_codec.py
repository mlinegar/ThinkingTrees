from __future__ import annotations

from pathlib import Path

import pytest

from src.ctreepo.sim.core.full_doc_config_codec import (
    FULL_DOC_CONFIG_ALIAS_PAIRS,
    canonicalize_full_doc_config_mapping,
    runtime_config_overrides_from_config_like,
    serialize_full_doc_runtime_config,
    serialize_tree_run_config,
)


def test_runtime_config_overrides_normalize_tree_aliases() -> None:
    payload = runtime_config_overrides_from_config_like(
        {
            "tree_local_law_weight": 0.8,
            "tree_task_objective_weight": 0.2,
            "tree_c1_relative_weight": 0.0,
            "tree_c2_relative_weight": 1.0,
            "tree_c3_relative_weight": 0.0,
            "tree_document_loss_normalization_mode": "supervised_docs",
            "tree_supervision_source": "manifest",
        }
    )

    assert payload["local_law_weight"] == 0.8
    assert payload["task_objective_weight"] == 0.2
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
            "tree_local_law_weight": 0.8,
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


def test_serialize_tree_run_config_keeps_tree_keys_and_drops_runtime_aliases(
    tmp_path: Path,
) -> None:
    payload = serialize_tree_run_config(
        {
            "baseline_family": "official_fno_sumlen",
            "local_law_weight": 0.8,
            "task_objective_weight": 0.2,
            "artifact_dir": tmp_path,
        }
    )

    assert payload["baseline_family"] == "official_fno_sumlen"
    assert payload["tree_local_law_weight"] == 0.8
    assert payload["tree_task_objective_weight"] == 0.2
    assert payload["artifact_dir"] == str(tmp_path)
    assert "local_law_weight" not in payload
    assert "task_objective_weight" not in payload


def test_alias_conflict_raises_when_both_differ() -> None:
    with pytest.raises(ValueError, match="Config alias conflict"):
        canonicalize_full_doc_config_mapping(
            {"tree_c1_relative_weight": 0.5, "c1_relative_weight": 0.3}
        )


def test_alias_no_conflict_when_both_match() -> None:
    result = canonicalize_full_doc_config_mapping(
        {"tree_c1_relative_weight": 0.5, "c1_relative_weight": 0.5}
    )
    assert result["tree_c1_relative_weight"] == 0.5
    assert result["c1_relative_weight"] == 0.5


def test_alias_sync_when_one_side_empty() -> None:
    result = canonicalize_full_doc_config_mapping(
        {"tree_c2_relative_weight": 1.0}
    )
    assert result["c2_relative_weight"] == 1.0
    assert result["tree_c2_relative_weight"] == 1.0


def test_alias_conflict_all_pairs() -> None:
    for tree_key, runtime_key in FULL_DOC_CONFIG_ALIAS_PAIRS:
        with pytest.raises(ValueError, match="Config alias conflict"):
            canonicalize_full_doc_config_mapping(
                {tree_key: 0.1, runtime_key: 0.9}
            )
