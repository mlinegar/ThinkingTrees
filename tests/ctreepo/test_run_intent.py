from __future__ import annotations

import pytest

from src.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
)
from src.ctreepo.sim.core.run_intent import materialize_tree_run_intent


def _minimal_config(**overrides):
    base = {
        "method_id": "tree_neural",
        "law_set_id": "all",
        "depth_discount_gamma": 1.0,
        "local_law_component_weights": {
            LAW_ID_LEAF_PRESERVATION: 1.0,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 1.0,
            LAW_ID_MERGE_PRESERVATION: 1.0,
        },
        "schedule_consistency_weight": 0.0,
        "local_law_weight": 0.5,
        "root_share": None,
    }
    base.update(overrides)
    return base


def test_valid_config_passes() -> None:
    intent = materialize_tree_run_intent(_minimal_config())
    assert intent["method_id"] == "tree_neural"
    assert intent["law_set_id"] == "all"
    assert "baseline_family" not in intent
    assert intent["depth_discount_gamma"] == 1.0


def test_legacy_private_family_label_can_materialize_intent() -> None:
    intent = materialize_tree_run_intent(
        _minimal_config(
            method_id="",
            baseline_family="tree_neural",
            law_package="all",
        )
    )
    assert intent["method_id"] == "tree_neural"
    assert intent["law_set_id"] == "all"
    assert "baseline_family" not in intent


def test_gamma_boundary_values_accepted() -> None:
    materialize_tree_run_intent(_minimal_config(depth_discount_gamma=0.0))
    materialize_tree_run_intent(_minimal_config(depth_discount_gamma=1.0))


def test_gamma_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="depth_discount_gamma"):
        materialize_tree_run_intent(_minimal_config(depth_discount_gamma=1.5))
    with pytest.raises(ValueError, match="depth_discount_gamma"):
        materialize_tree_run_intent(_minimal_config(depth_discount_gamma=-0.1))


def test_negative_relative_weight_raises() -> None:
    with pytest.raises(ValueError, match="local_law_component_weights"):
        materialize_tree_run_intent(
            _minimal_config(
                local_law_component_weights={
                    LAW_ID_LEAF_PRESERVATION: -0.5,
                    LAW_ID_ON_RANGE_IDEMPOTENCE: 1.0,
                    LAW_ID_MERGE_PRESERVATION: 1.0,
                }
            )
        )


def test_negative_schedule_consistency_weight_raises() -> None:
    with pytest.raises(ValueError, match="schedule_consistency_weight"):
        materialize_tree_run_intent(
            _minimal_config(schedule_consistency_weight=-1.0)
        )


def test_negative_optional_weight_raises() -> None:
    with pytest.raises(ValueError, match="local_law_weight"):
        materialize_tree_run_intent(_minimal_config(local_law_weight=-0.1))
    with pytest.raises(ValueError, match="root_share"):
        materialize_tree_run_intent(_minimal_config(local_law_weight=None, root_share=-0.1))


def test_none_optional_weight_accepted() -> None:
    intent = materialize_tree_run_intent(
        _minimal_config(local_law_weight=None, root_share=None)
    )
    assert intent["local_law_weight"] is None
    assert intent["root_share"] is None


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
