from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.ctreepo.contracts import (
    LAW_ID_LEAF_PRESERVATION,
    LAW_ID_MERGE_PRESERVATION,
    LAW_ID_ON_RANGE_IDEMPOTENCE,
    LAW_SET_ALL,
    LAW_SET_LEAF_AND_MERGE_PRESERVATION,
    LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
    ObjectiveSpec,
    OracleObservationDesignSpec,
    ProblemAdapterSpec,
    RunAxisSpec,
    assert_public_contract_clean,
    migrate_legacy_run_axis_mapping,
    oracle_observation_design_metadata,
    resolve_law_set,
)
from src.ctreepo.runtime import load_program, method_descriptors
from src.ctreepo.sim.composite_objective import (
    CompositeObjectiveSpec,
    evaluate_composite_objective,
    resolve_root_local_objective_weights,
    scalarize_objective_estimates,
)
from src.ctreepo.sim.core.leaf_local_mixture_utility import (
    LeafLocalMixtureUtilityConfig,
    _augment_summary_metrics_with_objective_estimators,
    _local_law_objective_spec,
    _objective_metrics_from_summary,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    _resolve_local_law_weights,
)


def test_shared_resolver_lambda_mode_all_laws_splits_lambda_equally() -> None:
    resolved = resolve_root_local_objective_weights(
        local_law_weight=0.6,
        active_laws=(
            LAW_ID_LEAF_PRESERVATION,
            LAW_ID_ON_RANGE_IDEMPOTENCE,
            LAW_ID_MERGE_PRESERVATION,
        ),
    )

    assert resolved.input_mode == "lambda"
    assert resolved.root_share == pytest.approx(0.4)
    assert resolved.as_metadata()["local_law_weight"] == pytest.approx(0.6)
    assert resolved.local_law_shares == pytest.approx(
        {
            LAW_ID_LEAF_PRESERVATION: 0.2,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 0.2,
            LAW_ID_MERGE_PRESERVATION: 0.2,
        }
    )


def test_shared_resolver_lambda_mode_c2_only_puts_all_local_mass_on_c2() -> None:
    resolved = resolve_root_local_objective_weights(
        local_law_weight=0.5,
        active_laws=(LAW_ID_ON_RANGE_IDEMPOTENCE,),
    )

    assert resolved.root_share == pytest.approx(0.5)
    assert resolved.as_metadata()["local_law_weight"] == pytest.approx(0.5)
    assert resolved.local_law_shares == pytest.approx({LAW_ID_ON_RANGE_IDEMPOTENCE: 0.5})


def test_shared_resolver_explicit_mode_normalizes_and_reports_implied_lambda() -> None:
    resolved = resolve_root_local_objective_weights(
        local_law_weight=None,
        active_laws=(
            LAW_ID_LEAF_PRESERVATION,
            LAW_ID_ON_RANGE_IDEMPOTENCE,
            LAW_ID_MERGE_PRESERVATION,
        ),
        explicit_root_weight=2.0,
        explicit_law_weights={
            LAW_ID_LEAF_PRESERVATION: 1.0,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 1.0,
            LAW_ID_MERGE_PRESERVATION: 0.0,
        },
    )

    assert resolved.input_mode == "explicit_weights"
    assert resolved.root_share == pytest.approx(0.5)
    assert resolved.as_metadata()["local_law_weight"] == pytest.approx(0.5)
    assert resolved.local_law_shares == pytest.approx(
        {
            LAW_ID_LEAF_PRESERVATION: 0.25,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 0.25,
            LAW_ID_MERGE_PRESERVATION: 0.0,
        }
    )


def test_shared_resolver_rejects_hybrid_lambda_and_explicit_weights() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_root_local_objective_weights(
            local_law_weight=0.5,
            active_laws=(LAW_ID_LEAF_PRESERVATION,),
            explicit_root_weight=1.0,
        )


def test_markov_lambda_mode_all_laws_splits_lambda_equally() -> None:
    resolved = _resolve_local_law_weights(OPSCountConfig(local_law_weight=0.6))

    assert resolved["parameterization"] == "lambda"
    assert resolved["root_share"] == pytest.approx(0.4)
    assert resolved["local_law_weight"] == pytest.approx(0.6)
    assert resolved["local_law_c1_weight"] == pytest.approx(0.2)
    assert resolved["local_law_c2_weight"] == pytest.approx(0.2)
    assert resolved["local_law_c3_weight"] == pytest.approx(0.2)


def test_markov_lambda_mode_c2_only_puts_all_local_mass_on_c2() -> None:
    resolved = _resolve_local_law_weights(
        OPSCountConfig(law_package="c2_only", local_law_weight=0.5)
    )

    assert resolved["parameterization"] == "law_set_lambda"
    assert resolved["root_share"] == pytest.approx(0.5)
    assert resolved["local_law_weight"] == pytest.approx(0.5)
    assert resolved["local_law_c1_weight"] == pytest.approx(0.0)
    assert resolved["local_law_c2_weight"] == pytest.approx(0.5)
    assert resolved["local_law_c3_weight"] == pytest.approx(0.0)


def test_markov_explicit_weight_mode_normalizes_and_reports_implied_lambda() -> None:
    resolved = _resolve_local_law_weights(
        OPSCountConfig(root_weight=2.0, leaf_weight=1.0, c2_weight=1.0, c3_weight=0.0)
    )

    assert resolved["parameterization"] == "explicit_normalized_weights"
    assert resolved["root_share"] == pytest.approx(0.5)
    assert resolved["local_law_weight"] == pytest.approx(0.5)
    assert resolved["local_law_c1_weight"] == pytest.approx(0.25)
    assert resolved["local_law_c2_weight"] == pytest.approx(0.25)
    assert resolved["local_law_c3_weight"] == pytest.approx(0.0)


def test_markov_hybrid_lambda_and_explicit_weights_raises() -> None:
    with pytest.raises(ValueError, match="local_law_weight cannot be combined"):
        _resolve_local_law_weights(OPSCountConfig(local_law_weight=0.5, leaf_weight=0.1))


def test_markov_zero_local_law_weight_is_root_only() -> None:
    resolved = _resolve_local_law_weights(OPSCountConfig(local_law_weight=0.0))

    assert resolved["root_share"] == pytest.approx(1.0)
    assert resolved["local_law_weight"] == pytest.approx(0.0)
    assert resolved["local_law_c1_weight"] == pytest.approx(0.0)
    assert resolved["local_law_c2_weight"] == pytest.approx(0.0)
    assert resolved["local_law_c3_weight"] == pytest.approx(0.0)


def test_composite_objective_is_strict_normalized_root_local_sum() -> None:
    spec = CompositeObjectiveSpec(
        name="configured_objective",
        selection_metric_name="configured_objective",
        root_metric_name="root_loss",
        root_share=0.5,
        local_law_component_weights={
            LAW_ID_LEAF_PRESERVATION: 0.25,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 0.25,
        },
        auxiliary_diagnostic_weights={"schedule_consistency": 100.0},
    )

    evaluation = evaluate_composite_objective(
        spec,
        task_value=10.0,
        local_law_values={
            LAW_ID_LEAF_PRESERVATION: 2.0,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 6.0,
        },
        proxy_values={"schedule_consistency": 7.0},
    )

    assert evaluation.root_share == pytest.approx(0.5)
    assert evaluation.local_law_weight == pytest.approx(0.5)
    assert evaluation.total == pytest.approx(7.0)
    assert evaluation.proxy_terms["schedule_consistency"] == pytest.approx(700.0)


def test_scalarized_objective_payload_has_no_trust_gate_fields() -> None:
    spec = CompositeObjectiveSpec(
        name="cfg",
        selection_metric_name="cfg",
        root_metric_name="root_loss",
        root_share=0.5,
        local_law_component_weights={LAW_ID_LEAF_PRESERVATION: 0.5},
    )

    payload = scalarize_objective_estimates(
        spec,
        task_estimates={"exact": 4.0, "hajek": 4.0},
        local_law_estimates={LAW_ID_LEAF_PRESERVATION: {"exact": 0.0, "hajek": 0.0}},
        selection_preference="hajek",
    )

    assert payload["cfg"] == pytest.approx(2.0)
    assert payload["cfg_hajek"] == pytest.approx(2.0)
    assert payload["root_share"] == pytest.approx(0.5)
    assert payload["local_law_weight"] == pytest.approx(0.5)
    assert "bias_gap" not in payload


def test_lda_lambda_mode_uses_equal_active_law_shares() -> None:
    spec = _local_law_objective_spec(
        LeafLocalMixtureUtilityConfig(local_law_weight=0.6, law_package="all_laws")
    )

    assert spec.normalized_task_share() == pytest.approx(0.4)
    assert spec.to_dict()["local_law_weight"] == pytest.approx(0.6)
    assert spec.normalized_local_law_weights() == pytest.approx(
        {
            LAW_ID_LEAF_PRESERVATION: 0.2,
            LAW_ID_ON_RANGE_IDEMPOTENCE: 0.2,
            LAW_ID_MERGE_PRESERVATION: 0.2,
        }
    )


def test_lda_explicit_mode_normalizes_root_and_law_weights() -> None:
    spec = _local_law_objective_spec(
        LeafLocalMixtureUtilityConfig(
            law_task_objective_weight=2.0,
            law_c1_weight=1.0,
            law_c2_proxy_weight=1.0,
            law_c3_weight=0.0,
        )
    )

    assert spec.weighting_scheme == "explicit_weights"
    assert spec.normalized_task_share() == pytest.approx(0.5)
    assert spec.to_dict()["local_law_weight"] == pytest.approx(0.5)
    assert spec.normalized_local_law_weights()[LAW_ID_LEAF_PRESERVATION] == pytest.approx(0.25)
    assert spec.normalized_local_law_weights()[LAW_ID_ON_RANGE_IDEMPOTENCE] == pytest.approx(0.25)
    assert spec.normalized_local_law_weights()[LAW_ID_MERGE_PRESERVATION] == pytest.approx(0.0)


def test_lda_hybrid_lambda_and_explicit_weights_raises() -> None:
    config = LeafLocalMixtureUtilityConfig(
        local_law_weight=0.5,
        law_task_objective_weight=0.5,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _local_law_objective_spec(config)


def test_lda_configured_objective_estimator_payload_and_selection_metric_agree() -> None:
    config = LeafLocalMixtureUtilityConfig(local_law_weight=0.6, law_package="all_laws")
    spec = _local_law_objective_spec(config)
    metrics = {
        "mean_aux_oracle_target_abs_error": 10.0,
        "mean_c1": 1.0,
        "mean_c2_proxy": 3.0,
        "mean_c3": 5.0,
    }

    augmented = _augment_summary_metrics_with_objective_estimators(
        metrics,
        objective_spec=spec,
    )
    objective_metrics = _objective_metrics_from_summary(
        augmented,
        objective_spec=spec,
        config=config,
    )

    expected = 0.4 * 10.0 + 0.2 * 1.0 + 0.2 * 3.0 + 0.2 * 5.0
    estimator_payload = dict(augmented["objective_estimator_payload"])
    assert augmented["configured_objective"] == pytest.approx(expected)
    assert augmented["selection_metric_value"] == pytest.approx(expected)
    assert objective_metrics["full_objective_value"] == pytest.approx(expected)
    assert estimator_payload["configured_objective"] == pytest.approx(expected)
    assert estimator_payload["root_share"] == pytest.approx(0.4)
    assert estimator_payload["local_law_weight"] == pytest.approx(0.6)
    assert not any("bias_gap" in str(key) for key in estimator_payload)


def test_public_law_registry_resolves_all_registered_laws() -> None:
    adapter = ProblemAdapterSpec(problem_id="unit_test")

    assert resolve_law_set(LAW_SET_ALL, registered_law_ids=adapter.registered_law_ids()) == (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_MERGE_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
    )
    payload = adapter.to_dict()
    assert [law["law_id"] for law in payload["laws"]] == [
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_MERGE_PRESERVATION,
        LAW_ID_ON_RANGE_IDEMPOTENCE,
    ]
    assert payload["laws"][0]["paper_label"] == "C1"


def test_run_axis_spec_serializes_canonical_public_fields_only() -> None:
    payload = RunAxisSpec(
        problem_id="markov_ops_count",
        method_id="tree_neural",
        law_set_id=LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY,
        root_share=0.5,
        local_law_weight=0.5,
        local_law_component_weights={LAW_ID_ON_RANGE_IDEMPOTENCE: 0.5},
    ).to_dict()

    assert payload["method_id"] == "tree_neural"
    assert payload["law_set_id"] == LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY
    assert payload["local_law_component_weights"] == {LAW_ID_ON_RANGE_IDEMPOTENCE: 0.5}
    for stale in ("baseline_family", "law_package", "tree_families", "fno_families"):
        assert stale not in payload


def test_run_axis_public_parser_rejects_legacy_fields_and_law_encoded_method() -> None:
    with pytest.raises(ValueError, match="baseline_family"):
        RunAxisSpec.from_mapping(
            {
                "problem_id": "markov_ops_count",
                "baseline_family": "tree_neural",
                "method_id": "tree_neural",
            }
        )
    with pytest.raises(ValueError, match="encodes a law set"):
        RunAxisSpec.from_mapping(
            {
                "problem_id": "markov_ops_count",
                "method_id": "tree_neural_c2",
            }
        )


def test_legacy_run_axis_migration_is_explicit() -> None:
    payload = migrate_legacy_run_axis_mapping(
        {
            "baseline_family": "tree_neural_c2",
            "law_package": "tree_c2_only",
            "local_law_weight": 0.5,
        }
    )

    assert payload["method_id"] == "tree_neural"
    assert payload["law_set_id"] == LAW_SET_ON_RANGE_IDEMPOTENCE_ONLY
    assert payload["local_law_weight"] == pytest.approx(0.5)


def test_public_contract_guard_rejects_stale_keys_recursively() -> None:
    with pytest.raises(ValueError, match="bias_gap"):
        assert_public_contract_clean(
            {"nested": {"objective": {"bias_gap": 0.4}}},
            surface="unit test",
        )
    with pytest.raises(ValueError, match="tree_neural_c2"):
        assert_public_contract_clean(
            {
                "method_runs": [
                    {
                        "problem_id": "markov_ops_count",
                        "method_id": "tree_neural_c2",
                        "law_set_id": LAW_SET_ALL,
                    }
                ]
            },
            surface="unit test",
        )
    with pytest.raises(ValueError, match="RunAxisSpec records"):
        assert_public_contract_clean(
            {"method_runs": ["tree_neural:all"]},
            surface="unit test",
        )
    with pytest.raises(ValueError, match="oracle_observation_mode"):
        assert_public_contract_clean(
            {"oracle_observation_mode": "root_only"},
            surface="unit test",
        )
    with pytest.raises(ValueError, match="sampled_node_rate"):
        assert_public_contract_clean(
            {"sampled_node_rate": 0.5},
            surface="unit test",
        )


def test_oracle_observation_design_is_canonical_public_shape() -> None:
    sampled = oracle_observation_design_metadata(
        "sampled_nodes",
        design_parameters={"sampled_node_rate": 0.5},
    )
    fixed = oracle_observation_design_metadata(
        "budgeted_mass",
        design_parameters={
            "root_label_share": 0.25,
            "mass_target_per_doc": 1.0,
            "local_label_pool": "nonroot",
            "local_label_allocation": "span_mass",
        },
    )

    assert sampled["design_id"] == "sampled_nodes"
    assert sampled["design_parameters"]["sampled_node_rate"] == pytest.approx(0.5)
    assert fixed["design_id"] == "budgeted_mass"
    assert_public_contract_clean(
        {
            "oracle_observation_design": sampled,
            "rows": [{"observed": True, "propensity": 1.0}],
        },
        surface="unit test",
    )
    with pytest.raises(ValueError, match="sampled_node_rate"):
        OracleObservationDesignSpec(
            design_id="dense_oracle",
            design_parameters={"sampled_node_rate": 0.5},
        )


def test_leaf_and_merge_law_set_is_public_and_generic() -> None:
    law_ids = ProblemAdapterSpec(problem_id="unit").active_law_ids(
        LAW_SET_LEAF_AND_MERGE_PRESERVATION
    )
    assert law_ids == (
        LAW_ID_LEAF_PRESERVATION,
        LAW_ID_MERGE_PRESERVATION,
    )


def test_checked_in_markov_configs_use_canonical_run_axis_keys() -> None:
    legacy_keys = {
        "baseline_family",
        "tree_families",
        "fno_families",
        "full_doc_anchor_families",
        "law_package",
        "law_package_names",
        "supervision_recovery_tree_family",
        "oracle_budget_tree_families",
        "oracle_budget_reference_families",
    }
    offenders: list[str] = []

    def walk(path: str, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                dotted = f"{path}.{key}" if path else str(key)
                if str(key) in legacy_keys:
                    offenders.append(dotted)
                walk(dotted, item)
        elif isinstance(value, list):
            if path.endswith("method_runs") or path.endswith("reference_method_runs"):
                for idx, item in enumerate(value):
                    if not isinstance(item, dict):
                        offenders.append(f"{path}[{idx}]")
            for idx, item in enumerate(value):
                walk(f"{path}[{idx}]", item)

    for config_path in sorted(Path("config/markov").glob("*.toml")):
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert_public_contract_clean(payload, surface=str(config_path))
        walk(str(config_path), payload)

    assert offenders == []


def test_objective_spec_rejects_legacy_public_fields_and_law_aliases() -> None:
    with pytest.raises(ValueError, match="legacy public objective fields"):
        ObjectiveSpec.from_mapping({"task_weight": 1.0, "root_share": 1.0})

    with pytest.raises(ValueError, match="legacy local-law id"):
        ObjectiveSpec.from_mapping(
            {
                "root_share": 0.5,
                "local_law_component_weights": {"c1": 0.5},
            }
        )


def test_runtime_methods_are_registered_by_method_id() -> None:
    descriptors = method_descriptors()

    assert descriptors["hll"]["method_id"] == "hll"
    assert descriptors["learned_sketch"]["method_family"] == "learned_sketch"


def test_runtime_program_spec_rejects_family_fallback_and_hll_alias_backend() -> None:
    with pytest.raises(ValueError, match="method_id"):
        load_program({"space_kind": "set", "family": "hll"})
    with pytest.raises(ValueError, match="unsupported HLL backend"):
        load_program(
            {
                "space_kind": "set",
                "method_id": "hll",
                "backend_config": {"backend": "hll_native"},
            }
        )


def test_generic_dspy_family_has_no_manifesto_imports() -> None:
    from pathlib import Path

    source = Path("src/ctreepo/dspy_family.py").read_text(encoding="utf-8")

    assert "src.tasks.manifesto" not in source
