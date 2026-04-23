from __future__ import annotations

import pytest

from src.ctreepo.sim.core.run_intent import materialize_tree_run_intent


def _minimal_config(**overrides):
    base = {
        "baseline_family": "tree_neural",
        "depth_discount_gamma": 1.0,
        "c1_relative_weight": 1.0,
        "c2_relative_weight": 1.0,
        "c3_relative_weight": 1.0,
        "schedule_consistency_weight": 0.0,
        "local_law_weight": 0.5,
        "task_objective_weight": 0.5,
    }
    base.update(overrides)
    return base


def test_valid_config_passes() -> None:
    intent = materialize_tree_run_intent(_minimal_config())
    assert intent["baseline_family"] == "tree_neural"
    assert intent["depth_discount_gamma"] == 1.0


def test_gamma_boundary_values_accepted() -> None:
    materialize_tree_run_intent(_minimal_config(depth_discount_gamma=0.0))
    materialize_tree_run_intent(_minimal_config(depth_discount_gamma=1.0))


def test_gamma_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="depth_discount_gamma"):
        materialize_tree_run_intent(_minimal_config(depth_discount_gamma=1.5))
    with pytest.raises(ValueError, match="depth_discount_gamma"):
        materialize_tree_run_intent(_minimal_config(depth_discount_gamma=-0.1))


def test_negative_relative_weight_raises() -> None:
    for field in ("c1_relative_weight", "c2_relative_weight", "c3_relative_weight"):
        with pytest.raises(ValueError, match=field):
            materialize_tree_run_intent(_minimal_config(**{field: -0.5}))


def test_negative_schedule_consistency_weight_raises() -> None:
    with pytest.raises(ValueError, match="schedule_consistency_weight"):
        materialize_tree_run_intent(
            _minimal_config(schedule_consistency_weight=-1.0)
        )


def test_negative_optional_weight_raises() -> None:
    with pytest.raises(ValueError, match="local_law_weight"):
        materialize_tree_run_intent(_minimal_config(local_law_weight=-0.1))
    with pytest.raises(ValueError, match="task_objective_weight"):
        materialize_tree_run_intent(_minimal_config(task_objective_weight=-0.1))


def test_none_optional_weight_accepted() -> None:
    intent = materialize_tree_run_intent(
        _minimal_config(local_law_weight=None, task_objective_weight=None)
    )
    assert intent["local_law_weight"] is None
    assert intent["task_objective_weight"] is None


def test_topology_tree_accepted() -> None:
    intent = materialize_tree_run_intent(_minimal_config(topology="tree"))
    assert intent["topology"] == "tree"


def test_topology_full_doc_accepted() -> None:
    intent = materialize_tree_run_intent(_minimal_config(topology="full_doc"))
    assert intent["topology"] == "full_doc"


def test_topology_empty_accepted() -> None:
    intent = materialize_tree_run_intent(_minimal_config(topology=""))
    assert intent["topology"] == ""


def test_topology_invalid_raises() -> None:
    with pytest.raises(ValueError, match="topology"):
        materialize_tree_run_intent(_minimal_config(topology="hybrid"))


def test_topology_included_in_intent_fields() -> None:
    from src.ctreepo.sim.core.run_intent import RUN_INTENT_FIELDS
    assert "topology" in RUN_INTENT_FIELDS


def test_topology_affects_intent_hash() -> None:
    from src.ctreepo.sim.core.run_intent import intent_hash
    intent_tree = materialize_tree_run_intent(_minimal_config(topology="tree"))
    intent_full = materialize_tree_run_intent(_minimal_config(topology="full_doc"))
    assert intent_hash(intent_tree) != intent_hash(intent_full)
