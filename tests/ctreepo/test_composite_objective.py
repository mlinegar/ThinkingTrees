from __future__ import annotations

import pytest

from src.ctreepo.sim.composite_objective import (
    CompositeObjectiveSpec,
    evaluate_composite_objective,
    evaluate_composite_objective_from_metrics,
    scalarize_objective_estimates,
)


def test_composite_objective_evaluation_combines_task_law_and_proxy_terms() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        task_name="task_objective",
        task_weight=2.0,
        local_law_weights={"c1": 0.5, "c3": 1.5},
        proxy_weights={"schedule_consistency": 0.25},
        weighting_scheme="explicit_task_plus_local_law",
        task_weight_source="explicit",
    )
    evaluation = evaluate_composite_objective(
        spec,
        task_value=1.25,
        local_law_values={"c1": 2.0, "c3": 3.0},
        proxy_values={"schedule_consistency": 4.0},
    )

    assert evaluation.total == pytest.approx(9.0)
    assert evaluation.task_term == pytest.approx(2.5)
    assert evaluation.local_law_terms["c1"] == pytest.approx(1.0)
    assert evaluation.local_law_terms["c3"] == pytest.approx(4.5)
    assert evaluation.proxy_terms["schedule_consistency"] == pytest.approx(1.0)

    flat = evaluation.to_flat_dict(prefix="configured_objective")
    assert flat["configured_objective"] == pytest.approx(9.0)
    assert flat["configured_objective_task_term"] == pytest.approx(2.5)
    assert flat["configured_objective_c1_term"] == pytest.approx(1.0)
    assert flat["configured_objective_c3_term"] == pytest.approx(4.5)
    assert flat["configured_objective_schedule_consistency_term"] == pytest.approx(1.0)


def test_composite_objective_spec_reports_total_non_proxy_weight() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        task_name="task_objective",
        task_weight=1.25,
        local_law_weights={"c1": 0.1, "c2": 0.2, "c3": 0.3},
    )
    payload = spec.to_dict()

    assert payload["task_weight"] == pytest.approx(1.25)
    assert payload["total_weight_without_proxy"] == pytest.approx(1.85)


def test_composite_objective_can_be_recomputed_from_metric_dict() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        task_name="task_objective",
        task_weight=2.0,
        local_law_weights={"c1": 0.5, "c2_proxy": 0.25, "c3": 1.5},
        weighting_scheme="explicit_task_plus_local_law",
        task_weight_source="explicit",
        metadata={
            "task_metric_name": "mean_aux_oracle_target_abs_error",
            "local_law_metric_names": {
                "c1": "mean_c1",
                "c2_proxy": "mean_c2_proxy",
                "c3": "mean_c3",
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

    assert evaluation.total == pytest.approx(9.0)
    assert evaluation.task_term == pytest.approx(2.5)
    assert evaluation.local_law_terms["c1"] == pytest.approx(1.0)
    assert evaluation.local_law_terms["c2_proxy"] == pytest.approx(1.0)
    assert evaluation.local_law_terms["c3"] == pytest.approx(4.5)


def test_scalarized_objective_estimates_expose_estimator_aware_selection_aliases() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        task_name="task_objective",
        task_weight=0.5,
        local_law_weights={"c1": 0.25, "c3": 0.75},
        proxy_weights={"schedule_consistency": 0.1},
    )

    payload = scalarize_objective_estimates(
        spec,
        task_estimates={"exact": 2.0, "ht": 2.0, "hajek": 2.0},
        local_law_estimates={
            "c1": {"exact": 1.0, "ht": 1.2, "hajek": 1.1},
            "c3": {"exact": 3.0, "ht": 2.8, "hajek": 2.9},
        },
        proxy_estimates={"schedule_consistency": {"exact": 4.0, "ht": 4.0, "hajek": 4.0}},
        selection_preference="hajek",
    )

    assert payload["selection_metric_name"] == "configured_objective_hajek"
    assert payload["selection_estimator"] == "hajek"
    assert payload["configured_objective"] == pytest.approx(3.9)
    assert payload["configured_objective_hajek"] == pytest.approx(3.85)
    assert payload["configured_objective_ht"] == pytest.approx(3.8)
