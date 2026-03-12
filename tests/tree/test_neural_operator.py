"""Unit tests for shared neural-operator adapters."""

import torch

from src.core.ops_checks import EvidenceStatus
from src.training.embedding_sketch import EmbeddingSketchConfig, MergeableEmbeddingSketch
from src.tree.ctreepo_model import CTreePOConfig, CTreePOModel
from src.tree.neural_operator import (
    CTreePOOperatorAdapter,
    FunctionalSketchLawOperator,
    MergeableSketchOperatorAdapter,
    SummaryAutoencoderOperatorAdapter,
    make_deterministic_summary_operator,
)
from src.core.ops_checks import LawCapabilityReport, LawKind


def test_ctreepo_operator_adapter_prediction_shape_and_bounds():
    model = CTreePOModel(CTreePOConfig(embedding_dim=8, sketch_dim=4, hidden_dim=8))
    adapter = CTreePOOperatorAdapter(model=model, head="rile")
    assert adapter.evidence_status == EvidenceStatus.PROXY_ONLY
    report = adapter.capability_report()
    assert report.tree_nesting_supported is True
    assert report.theorem_domain_decode_available is False
    assert report.supports_resummary_idempotence is False

    leaf_state = adapter.encode_leaf(torch.randn(8))
    pred = adapter.predict_from_state(leaf_state)

    assert pred.lower <= pred.mean <= pred.upper
    assert pred.std >= 0.0
    assert pred.confidence is not None
    assert 0.0 <= float(pred.confidence) <= 1.0
    assert pred.evidence_status == EvidenceStatus.PROXY_ONLY
    assert pred.aux.get("head") == "rile"


def test_mergeable_sketch_operator_adapter_prediction_shape_and_bounds():
    cfg = EmbeddingSketchConfig(
        embedding_dim=8,
        state_dim=4,
        phi_hidden_dim=8,
        readout_hidden_dim=8,
        include_meta=False,
        include_retrieval_features=False,
        include_delta_head=True,
    )
    model = MergeableEmbeddingSketch(cfg)
    adapter = MergeableSketchOperatorAdapter(model=model)
    assert adapter.evidence_status == EvidenceStatus.PROXY_ONLY
    report = adapter.capability_report()
    assert report.latent_mergeability_enforced is True
    assert report.theorem_domain_reencode_available is False
    assert report.supports_theorem_backed_l3 is False

    windows = torch.randn(1, 3, 8)
    counts = torch.tensor([3], dtype=torch.int64)
    state = adapter.encode_windows(windows, counts=counts)
    pred = adapter.predict_from_state(state)

    assert pred.lower <= pred.mean <= pred.upper
    assert pred.std >= 0.0
    assert pred.normalized_mean is not None
    assert 0.0 <= float(pred.normalized_mean) <= 1.0
    assert pred.confidence is not None
    assert 0.0 <= float(pred.confidence) <= 1.0
    assert pred.evidence_status == EvidenceStatus.PROXY_ONLY


def test_functional_sketch_operator_reports_theorem_backed_idempotence():
    operator = FunctionalSketchLawOperator[str, str](
        name="upper_canonical",
        encode_fn=lambda span: span.upper(),
        merge_fn=lambda left, right: f"{left}|{right}",
        decode_fn=lambda sketch: sketch,
        combine_fn=lambda left, right: f"{left}|{right}",
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        latent_mergeability_enforced=False,
        tree_nesting_supported=True,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        exact_reduction_supported=True,
        leaf_law=LawCapabilityReport(
            law_kind=LawKind.L1_LEAF,
            available=True,
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            objective_enforced=True,
            exact=True,
        ),
        merge_law=LawCapabilityReport(
            law_kind=LawKind.L2_MERGE,
            available=True,
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            objective_enforced=True,
            exact=True,
        ),
        idempotence_law=LawCapabilityReport(
            law_kind=LawKind.L3_IDEMPOTENCE,
            available=True,
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            objective_enforced=True,
            exact=True,
        ),
    )

    assert operator.resummarize("abc") == "ABC"
    report = operator.capability_report()
    assert report.supports_resummary_idempotence is True
    assert report.supports_theorem_backed_l3 is True


def test_make_deterministic_summary_operator_exposes_string_theorem_domain():
    operator = make_deterministic_summary_operator(
        lambda text, rubric: f"{rubric}:{text}".upper(),
        name="demo_summary",
    )

    assert operator.combine("left", "right") == "PART 1:\nleft\n\nPART 2:\nright"
    assert operator.resummarize("abc", rubric="r") == "R:ABC"
    report = operator.capability_report()
    assert report.theorem_domain_decode_available is True
    assert report.theorem_domain_reencode_available is True
    assert report.idempotence_law.available is True
    assert report.supports_theorem_backed_l3 is False


class _ToySummaryAutoencoder:
    def encode_summary(self, summary: str, **kwargs):
        del kwargs
        return summary.upper()

    def decode_summary(self, sketch: str, **kwargs):
        del kwargs
        return sketch

    def merge(self, left: str, right: str, **kwargs):
        del kwargs
        return f"{left}|{right}"


def test_summary_autoencoder_adapter_exposes_theorem_domain_roundtrip():
    adapter = SummaryAutoencoderOperatorAdapter[str, str](
        model=_ToySummaryAutoencoder(),
        name="toy_autoencoder",
        evidence_status=EvidenceStatus.APPROX_AUDITED,
    )

    encoded = adapter.encode("abc")
    merged = adapter.merge(encoded, adapter.encode("def"))
    assert encoded == "ABC"
    assert adapter.decode(merged) == "ABC|DEF"
    assert adapter.resummarize("ghi") == "GHI"
    report = adapter.capability_report()
    assert report.supports_resummary_idempotence is True
    assert report.evidence_status == EvidenceStatus.APPROX_AUDITED
