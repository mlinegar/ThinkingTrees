"""Tests for theorem-backing assumption presets."""

import torch

from src.core.ops_checks import EvidenceStatus
from src.tree.compositional_operator import (
    FunctionalCompositionalOperator,
    make_text_compositional_operator,
)
from src.tree.theorem_backing import (
    TheoremBackingRoute,
    broadest_approximate_theorem_assumptions,
    broadest_exact_theorem_assumptions,
    exact_oracle_measurement_assumption,
    global_preservation_exact_assumptions,
    high_probability_oracle_measurement_assumption,
    llm_audited_approximate_assumptions,
    llm_direct_exact_assumptions,
    neural_codec_approximate_assumptions,
    neural_codec_exact_assumptions,
    uniform_oracle_measurement_assumption,
)


def test_broadest_exact_theorem_assumptions_expose_local_law_bundle_contract():
    spec = broadest_exact_theorem_assumptions()

    report = spec.capability_report("demo")

    assert spec.route == TheoremBackingRoute.DIRECT_LOCAL_LAWS
    assert spec.theorem_bundle_kind == "LocalLawsBundle"
    assert spec.lean_bridge == "ExactTheoremBacked.ofLocalLaws"
    assert [obligation.symbol for obligation in spec.obligations] == ["L1", "L2", "L3"]
    assert report.supports_theorem_backed_l3 is True
    assert report.evidence_status == EvidenceStatus.THEOREM_BACKED


def test_llm_direct_exact_assumptions_can_back_text_operator():
    spec = llm_direct_exact_assumptions()
    operator = make_text_compositional_operator(
        lambda text, rubric: f"{rubric}:{text}".upper(),
        name="llm_text",
        assumptions=spec.operator_assumptions,
    )

    report = operator.capability_report()

    assert spec.route == TheoremBackingRoute.DIRECT_LOCAL_LAWS
    assert report.supports_theorem_backed_l3 is True
    assert report.evidence_status == EvidenceStatus.THEOREM_BACKED


def test_neural_codec_exact_assumptions_use_sketch_bridge():
    spec = neural_codec_exact_assumptions()
    operator = FunctionalCompositionalOperator[str, torch.Tensor](
        name="codec",
        encode_fn=lambda span: torch.full((2,), float(len(span))),
        merge_fn=lambda left, right: left + right,
        decode_fn=lambda state: str(int(state.reshape(-1)[0].item())),
        combine_fn=lambda left, right: f"{left}|{right}",
        assumptions=spec.operator_assumptions,
    )

    report = operator.capability_report()

    assert spec.route == TheoremBackingRoute.SKETCH_CODEC_LOCAL_LAWS
    assert spec.lean_bridge == "local_laws_bundle_of_sketch"
    assert [obligation.symbol for obligation in spec.obligations] == [
        "SketchLeafPreserving",
        "SketchMergeCompatible",
        "SketchSummaryCompatible",
    ]
    assert report.supports_theorem_backed_l3 is True
    assert report.latent_mergeability_enforced is True


def test_broadest_approximate_theorem_assumptions_expose_budget_bundle():
    spec = broadest_approximate_theorem_assumptions(
        eps_leaf=0.1,
        eps_merge=0.2,
        eps_idemp=0.3,
    )

    report = spec.capability_report("demo")

    assert spec.route == TheoremBackingRoute.DIRECT_APPROX_LOCAL_LAWS
    assert spec.theorem_bundle_kind == "ApproxLocalLawsBundle"
    assert report.evidence_status == EvidenceStatus.APPROX_AUDITED
    assert report.approx_local_laws is not None
    assert report.approx_local_laws.eps_leaf == 0.1
    assert report.approx_local_laws.eps_merge == 0.2
    assert report.approx_local_laws.eps_idemp == 0.3


def test_llm_audited_approximate_assumptions_use_audited_route():
    spec = llm_audited_approximate_assumptions(
        eps_leaf=0.05,
        eps_merge=0.15,
        eps_idemp=0.25,
    )

    report = spec.capability_report("audited_text")

    assert spec.route == TheoremBackingRoute.AUDITED_APPROX_UPPER_BOUNDS
    assert spec.lean_bridge == "approx_bundle_of_audited_upper_bounds"
    assert [obligation.symbol for obligation in spec.obligations] == [
        "leaf_cert",
        "merge_cert",
        "idemp_cert",
    ]
    assert report.approx_local_laws is not None
    assert report.approx_local_laws.eps_idemp == 0.25


def test_neural_codec_approximate_assumptions_use_sketch_budget_bridge():
    spec = neural_codec_approximate_assumptions(
        eps_leaf=0.01,
        eps_merge=0.02,
        eps_idemp=0.03,
    )

    report = spec.capability_report("codec")

    assert spec.route == TheoremBackingRoute.SKETCH_CODEC_APPROX_LOCAL_LAWS
    assert spec.lean_bridge == "approx_bundle_of_sketch"
    assert report.evidence_status == EvidenceStatus.APPROX_AUDITED
    assert report.latent_mergeability_enforced is True
    assert report.approx_local_laws is not None
    assert report.approx_local_laws.eps_merge == 0.02


def test_global_preservation_route_is_stronger_exact_alternative():
    spec = global_preservation_exact_assumptions()

    report = spec.capability_report("global_demo")

    assert spec.route == TheoremBackingRoute.GLOBAL_PRESERVATION
    assert spec.lean_bridge == "GlobalPreservation.toLocalLawsBundle"
    assert [obligation.symbol for obligation in spec.obligations] == [
        "A1_global",
        "A2_global",
        "A3_global",
    ]
    assert report.supports_theorem_backed_l3 is True


def test_theorem_assumption_spec_can_attach_exact_oracle_measurement():
    spec = broadest_exact_theorem_assumptions().with_oracle_measurement(
        exact_oracle_measurement_assumption()
    )

    report = spec.capability_report("oracle_exact_demo")

    assert report.supports_oracle_measurement_bridge is True
    assert report.oracle_measurement is not None
    assert report.oracle_measurement.exact_oracle is True
    assert report.oracle_measurement.uniform_error_bound == 0.0


def test_theorem_assumption_spec_can_attach_high_probability_oracle_measurement():
    spec = neural_codec_approximate_assumptions(
        eps_leaf=0.01,
        eps_merge=0.02,
        eps_idemp=0.03,
    ).with_oracle_measurement(
        high_probability_oracle_measurement_assumption(
            0.15,
            0.05,
            pointwise_error_bound_available=True,
        )
    )

    report = spec.capability_report("codec_with_hp_oracle")

    assert report.supports_oracle_measurement_bridge is True
    assert report.oracle_measurement is not None
    assert report.oracle_measurement.high_probability_error_bound == 0.15
    assert report.oracle_measurement.failure_probability == 0.05
    assert report.oracle_measurement.pointwise_error_bound_available is True


def test_uniform_oracle_measurement_helper_marks_zero_error_as_exact():
    envelope = uniform_oracle_measurement_assumption(0.0)

    assert envelope.exact_oracle is True
    assert envelope.uniform_error_bound == 0.0
