from __future__ import annotations

import pytest

from src.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
)
from src.ctreepo.sim.composite_objective import (
    CompositeObjectiveSpec,
    evaluate_composite_objective,
    evaluate_composite_objective_from_metrics,
    scalarize_objective_estimates,
)


def test_composite_objective_evaluation_combines_normalized_task_and_law_terms() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="root_loss",
        root_share=0.5,
        local_law_component_weights={
            LAW_ID_LEAF_PRESERVATION: 0.125,
            LAW_ID_MERGE_PRESERVATION: 0.375,
        },
        auxiliary_diagnostic_weights={"schedule_consistency": 0.25},
        weighting_scheme="explicit_task_plus_local_law",
        root_share_source="explicit",
    )
    evaluation = evaluate_composite_objective(
        spec,
        task_value=1.25,
        local_law_values={
            LAW_ID_LEAF_PRESERVATION: 2.0,
            LAW_ID_MERGE_PRESERVATION: 3.0,
        },
        proxy_values={"schedule_consistency": 4.0},
    )

    assert evaluation.total == pytest.approx(2.0)
    assert evaluation.task_term == pytest.approx(0.625)
    assert evaluation.local_law_terms[LAW_ID_LEAF_PRESERVATION] == pytest.approx(0.25)
    assert evaluation.local_law_terms[LAW_ID_MERGE_PRESERVATION] == pytest.approx(1.125)
    assert evaluation.proxy_terms["schedule_consistency"] == pytest.approx(1.0)
    assert evaluation.root_share == pytest.approx(0.5)
    assert evaluation.local_law_weight == pytest.approx(0.5)

    flat = evaluation.to_flat_dict(prefix="configured_objective")
    assert flat["configured_objective"] == pytest.approx(2.0)
    assert flat["configured_objective_task_term"] == pytest.approx(0.625)
    assert flat[f"configured_objective_{LAW_ID_LEAF_PRESERVATION}_term"] == pytest.approx(0.25)
    assert flat[f"configured_objective_{LAW_ID_MERGE_PRESERVATION}_term"] == pytest.approx(1.125)
    assert flat["configured_objective_schedule_consistency_term"] == pytest.approx(1.0)


def test_composite_objective_spec_reports_total_non_proxy_weight() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="root_loss",
        root_share=0.5,
        local_law_component_weights={
            LAW_ID_LEAF_PRESERVATION: 0.1,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 0.2,
            LAW_ID_MERGE_PRESERVATION: 0.2,
        },
    )
    payload = spec.to_dict()

    assert payload["root_share"] == pytest.approx(0.5)
    assert payload["total_weight_without_proxy"] == pytest.approx(1.0)
    assert payload["objective_spec"]["schema_version"] == "treepo.objective.v1"
    assert "oracle_gap" not in payload["objective_spec"]["terms"]
    assert payload["objective_spec"]["terms"]["root"]["weight"] == pytest.approx(0.5)
    assert "task_weight" not in payload
    assert "local_law_weights" not in payload
    assert "proxy_weights" not in payload


def test_composite_objective_can_be_recomputed_from_metric_dict() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="mean_aux_oracle_target_abs_error",
        root_share=2.0 / 4.25,
        local_law_component_weights={
            LAW_ID_LEAF_PRESERVATION: 0.5 / 4.25,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 0.25 / 4.25,
            LAW_ID_MERGE_PRESERVATION: 1.5 / 4.25,
        },
        weighting_scheme="explicit_task_plus_local_law",
        root_share_source="explicit",
        metadata={
            "root_metric_name": "mean_aux_oracle_target_abs_error",
            "local_law_metric_names": {
                LAW_ID_LEAF_PRESERVATION: "mean_c1",
                LAW_ID_ON_RANGE_IDEMPOTENCE: "mean_c2_proxy",
                LAW_ID_MERGE_PRESERVATION: "mean_c3",
            },
        },
    )

    evaluation = evaluate_composite_objective_from_metrics(
        spec,
        metrics={
            "mean_aux_oracle_target_abs_error": 1.25,
            "mean_c1": 2.0,
            "mean_c2_proxy": 4.0,
            "mean_c3": 3.0,
            "configured_objective": -999.0,
        },
    )

    assert evaluation.total == pytest.approx(2.1176470588235294)
    assert evaluation.task_term == pytest.approx(0.5882352941176471)
    assert evaluation.local_law_terms[LAW_ID_LEAF_PRESERVATION] == pytest.approx(0.23529411764705882)
    assert evaluation.local_law_terms[LAW_ID_ON_RANGE_IDEMPOTENCE] == pytest.approx(0.23529411764705882)
    assert evaluation.local_law_terms[LAW_ID_MERGE_PRESERVATION] == pytest.approx(1.0588235294117647)


def test_scalarized_objective_estimates_expose_estimator_aware_selection_aliases() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="root_loss",
        root_share=1.0 / 3.0,
        local_law_component_weights={
            LAW_ID_LEAF_PRESERVATION: 1.0 / 6.0,
            LAW_ID_MERGE_PRESERVATION: 0.5,
        },
        auxiliary_diagnostic_weights={"schedule_consistency": 0.1},
    )

    payload = scalarize_objective_estimates(
        spec,
        task_estimates={"exact": 2.0, "ht": 2.0, "hajek": 2.0},
        local_law_estimates={
            LAW_ID_LEAF_PRESERVATION: {"exact": 1.0, "ht": 1.2, "hajek": 1.1},
            LAW_ID_MERGE_PRESERVATION: {"exact": 3.0, "ht": 2.8, "hajek": 2.9},
        },
        proxy_estimates={"schedule_consistency": {"exact": 4.0, "ht": 4.0, "hajek": 4.0}},
        selection_preference="hajek",
    )

    assert payload["selection_metric_name"] == "configured_objective_hajek"
    assert payload["selection_estimator"] == "hajek"
    assert payload["configured_objective"] == pytest.approx(2.3333333333333335)
    assert payload["configured_objective_hajek"] == pytest.approx(2.3)
    assert payload["configured_objective_ht"] == pytest.approx(2.2666666666666666)
    assert payload["local_law_weight"] == pytest.approx(2.0 / 3.0)
