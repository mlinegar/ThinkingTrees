from __future__ import annotations

from dataclasses import asdict
import math

from src.ctreepo.sim.core.run_config import (
    JobSpec,
    RunConfigSpec,
    config_mapping_for_run_config,
    run_config_from_mapping,
    with_run_intent_overrides,
)
from src.ctreepo.sim.core.tree_neural_facade import (
    config_mapping_for_run_config as tree_neural_config_mapping_for_run_config,
    run_config_from_mapping as tree_neural_run_config_from_mapping,
)


def _base_config(**overrides) -> RunConfigSpec:
    defaults = dict(
        label="test",
        state_dim=128,
        hidden_dim=512,
        n_epochs=32,
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        baseline_family="tree_neural",
    )
    defaults.update(overrides)
    return RunConfigSpec(**defaults)


def test_run_config_spec_has_topology_field() -> None:
    cfg = _base_config(topology="full_doc")
    assert cfg.topology == "full_doc"


def test_run_config_spec_default_topology_is_empty() -> None:
    cfg = _base_config()
    assert cfg.topology == ""


def test_run_config_spec_invalid_topology_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="topology"):
        _base_config(topology="hybrid")


def test_run_config_from_mapping_preserves_topology() -> None:
    cfg = run_config_from_mapping({
        "baseline_family": "official_fno",
        "topology": "full_doc",
        "fixed_leaf_tokens": 128,
    })
    assert cfg.topology == "full_doc"
    assert cfg.fixed_leaf_tokens == 128


def test_with_run_intent_overrides_recomputes_package_semantics() -> None:
    cfg = _base_config(leaf_label_rate=1.0, budget_total_calls_per_doc=0.0)
    updated = with_run_intent_overrides(cfg, budget_total_calls_per_doc=10.0)
    assert updated.budget_total_calls_per_doc == 10.0
    assert updated.package_semantics == "superset"


def test_config_mapping_strips_tree_prefixes() -> None:
    cfg = _base_config(tree_c1_relative_weight=0.5)
    mapping = config_mapping_for_run_config(cfg)
    assert "c1_relative_weight" in mapping
    assert "tree_c1_relative_weight" not in mapping


def test_job_spec_auto_fills_baseline_family() -> None:
    cfg = _base_config(baseline_family="")
    job = JobSpec(
        family="tree_neural",
        train_doc_count=50,
        benchmark="v4",
        hardness_grid="",
        grid_cell_ids=(),
        seeds=(42,),
        config=cfg,
    )
    assert job.config.baseline_family == "tree_neural"


def test_job_spec_rejects_family_mismatch() -> None:
    import pytest
    cfg = _base_config(baseline_family="official_fno")
    with pytest.raises(ValueError, match="family/config mismatch"):
        JobSpec(
            family="tree_neural",
            train_doc_count=50,
            benchmark="v4",
            hardness_grid="",
            grid_cell_ids=(),
            seeds=(42,),
            config=cfg,
        )


def test_tree_neural_facade_run_config_from_mapping_matches_canonical_mapper() -> None:
    mapping = {
        "label": " parity/full_doc ",
        "baseline_family": "official_fno",
        "topology": "full_doc",
        "fixed_leaf_tokens": 128,
        "comparison_mode": "comparable",
        "local_law_weight": 0.25,
        "c1_relative_weight": 0.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 0.0,
        "tree_document_loss_normalization_mode": "supervised_docs",
        "tree_supervision_source": "manifest",
        "gpu_runtime_preload_splits": "train,val",
        "budget_total_calls_per_doc": 1.0,
        "depth_discount_gamma": 0.9,
    }

    shared = run_config_from_mapping(mapping)
    facade_cfg = tree_neural_run_config_from_mapping(mapping)

    shared_dict = asdict(shared)
    facade_dict = asdict(facade_cfg)

    assert shared_dict.keys() == facade_dict.keys()
    for key, shared_value in shared_dict.items():
        mig_value = facade_dict[key]
        if (
            isinstance(shared_value, float)
            and isinstance(mig_value, float)
            and math.isnan(shared_value)
            and math.isnan(mig_value)
        ):
            continue
        assert shared_value == mig_value, key


def test_tree_neural_facade_config_mapping_matches_canonical_config_mapping() -> None:
    cfg = run_config_from_mapping(
        {
            "label": "tree_v3",
            "baseline_family": "tree_neural",
            "topology": "tree",
            "fixed_leaf_tokens": 16,
            "tree_c1_relative_weight": 0.5,
            "tree_c2_relative_weight": 1.0,
            "tree_c3_relative_weight": 0.25,
            "budget_total_calls_per_doc": 2.0,
        }
    )

    assert config_mapping_for_run_config(cfg) == tree_neural_config_mapping_for_run_config(
        RunConfigSpec(**asdict(cfg))
    )


def test_tree_neural_facade_accepts_legacy_objective_field_names() -> None:
    local_cfg = tree_neural_run_config_from_mapping(
        {
            "label": "legacy_local",
            "baseline_family": "tree_neural",
            "tree_local_law_weight": 0.25,
        }
    )
    root_cfg = tree_neural_run_config_from_mapping(
        {
            "label": "legacy_root",
            "baseline_family": "tree_neural",
            "tree_task_objective_weight": 0.75,
        }
    )

    assert local_cfg.tree_local_law_weight == 0.25
    assert root_cfg.tree_task_objective_weight == 0.75
