from src.core.ops_checks import (
    ApproxLocalLawsBundle,
    EvidenceStatus,
    LEAN_LOCAL_LAW_MAP,
    LawKind,
    OracleMeasurementEnvelope,
)


def test_lean_local_law_mapping_matches_formalization():
    assert LEAN_LOCAL_LAW_MAP == {
        "C1": "L1",
        "C2": "L3",
        "C3": "L2",
    }
    assert LawKind.L1_LEAF.paper_condition == "C1"
    assert LawKind.L2_MERGE.paper_condition == "C3"
    assert LawKind.L3_IDEMPOTENCE.paper_condition == "C2"


def test_approx_local_laws_bundle_serializes_expected_fields():
    bundle = ApproxLocalLawsBundle(
        eps_leaf=0.1,
        eps_merge=0.2,
        eps_idemp=0.3,
        evidence_status=EvidenceStatus.APPROX_AUDITED,
        notes="test",
    )
    assert bundle.to_dict() == {
        "eps_leaf": 0.1,
        "eps_merge": 0.2,
        "eps_idemp": 0.3,
        "evidence_status": "approx_audited",
        "notes": "test",
    }


def test_oracle_measurement_envelope_serializes_expected_fields():
    envelope = OracleMeasurementEnvelope(
        exact_oracle=False,
        uniform_error_bound=0.4,
        high_probability_error_bound=0.25,
        failure_probability=0.05,
        pointwise_error_bound_available=True,
        notes="oracle audit",
    )
    assert envelope.to_dict() == {
        "exact_oracle": False,
        "uniform_error_bound": 0.4,
        "high_probability_error_bound": 0.25,
        "failure_probability": 0.05,
        "pointwise_error_bound_available": True,
        "notes": "oracle audit",
    }
