"""Tests for the general compositional-operator abstraction."""

import torch

from src.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind
from src.training.embedding_sketch import (
    EmbeddingSketchConfig,
    MergeableEmbeddingSketch,
    SketchState,
)
from src.tree.compositional_operator import (
    FunctionalCompositionalOperator,
    OperatorAssumptionBundle,
    attach_compositional_operator,
    make_text_compositional_operator,
)
from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel
from src.tree.neural_operator import CTreePOOperatorAdapter, MergeableSketchOperatorAdapter


def _theorem_backed_assumptions() -> OperatorAssumptionBundle:
    theorem_law = lambda kind: LawCapabilityReport(
        law_kind=kind,
        available=True,
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        objective_enforced=True,
        exact=True,
    )
    return OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        tree_nesting_supported=True,
        exact_reduction_supported=True,
        leaf_law=theorem_law(LawKind.L1_LEAF),
        merge_law=theorem_law(LawKind.L2_MERGE),
        idempotence_law=theorem_law(LawKind.L3_IDEMPOTENCE),
        notes=("Supplied theorem codec/certificate.",),
    )


def test_assumption_bundle_preserves_explicit_law_overrides():
    assumptions = OperatorAssumptionBundle(
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
    )
    operator = FunctionalCompositionalOperator[str, str](
        name="canonical_text",
        encode_fn=lambda span: span.upper(),
        merge_fn=lambda left, right: f"{left}|{right}",
        decode_fn=lambda sketch: sketch,
        combine_fn=lambda left, right: f"{left}|{right}",
        assumptions=assumptions,
        idempotence_law=LawCapabilityReport(
            law_kind=LawKind.L3_IDEMPOTENCE,
            available=True,
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            objective_enforced=True,
            exact=True,
        ),
    )

    report = operator.capability_report()

    assert report.idempotence_law.objective_enforced is True
    assert report.idempotence_law.exact is True
    assert report.supports_theorem_backed_l3 is True


def test_ctreepo_can_be_paired_with_supplied_theorem_operator():
    model = CTreePOModel(CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8))
    predictor = CTreePOOperatorAdapter(model=model)
    operator = FunctionalCompositionalOperator[str, torch.Tensor](
        name="ctreepo_codec",
        encode_fn=lambda span: torch.full((4,), float(len(span))),
        merge_fn=model.merge,
        decode_fn=lambda state: str(int(state.reshape(-1)[0].item())),
        combine_fn=lambda left, right: f"{left}|{right}",
        assumptions=_theorem_backed_assumptions(),
    )

    paired = attach_compositional_operator(predictor, operator, name="ctreepo_paired")
    prediction = paired.predict_from_span("abcd")

    assert paired.capability_report().supports_theorem_backed_l3 is True
    assert paired.prediction_evidence_status == EvidenceStatus.PROXY_ONLY
    assert prediction.evidence_status == EvidenceStatus.PROXY_ONLY
    assert paired.resummarize("xy") == "2"


def test_mergeable_sketch_can_be_paired_with_supplied_theorem_operator():
    cfg = EmbeddingSketchConfig(
        embedding_dim=8,
        state_dim=4,
        phi_hidden_dim=8,
        readout_hidden_dim=8,
        include_meta=False,
        include_retrieval_features=False,
        include_delta_head=False,
    )
    model = MergeableEmbeddingSketch(cfg)
    predictor = MergeableSketchOperatorAdapter(model=model)
    operator = FunctionalCompositionalOperator[str, SketchState](
        name="mergeable_codec",
        encode_fn=lambda span: SketchState(
            sum_phi=torch.full((1, cfg.state_dim), float(len(span))),
            count=torch.tensor([float(max(len(span), 1))]),
        ),
        merge_fn=lambda left, right: left.merge(right),
        decode_fn=lambda state: f"count={int(state.count.reshape(-1)[0].item())}",
        combine_fn=lambda left, right: f"{left}|{right}",
        assumptions=_theorem_backed_assumptions(),
    )

    paired = predictor.with_compositional_operator(operator, name="mergeable_paired")
    prediction = paired.predict_from_span("abcd")

    assert paired.capability_report().supports_theorem_backed_l3 is True
    assert paired.prediction_evidence_status == EvidenceStatus.PROXY_ONLY
    assert prediction.evidence_status == EvidenceStatus.PROXY_ONLY
    assert paired.resummarize("abc").startswith("count=")


def test_make_text_compositional_operator_keeps_summary_alias_behavior():
    operator = make_text_compositional_operator(
        lambda text, rubric: f"{rubric}:{text}".upper(),
        name="text_codec",
    )

    assert operator.resummarize("abc", rubric="r") == "R:ABC"
    assert operator.capability_report().supports_resummary_idempotence is True
