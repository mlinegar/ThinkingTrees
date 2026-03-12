import pytest

from src.core.data_models import leaf, node
from src.core.ops_checks import EvidenceStatus, LawCapabilityReport, LawKind
from src.training.core import Prediction
from src.tree.neural_operator import FunctionalSketchLawOperator
from src.tree.verification import OracleNodeVerifier, TreeVerifier


class LengthPredictor:
    def __call__(self, original_content: str, summary: str, rubric: str) -> Prediction:
        del original_content, rubric
        return Prediction(label=str(len(summary)), confidence=1.0, reasoning="len")


def test_merge_consistency_uses_ops_span_for_internal_nodes():
    left = leaf("left", summary="L")
    right = leaf("right", summary="R")
    root = node(left, right, summary="bad")
    root.ops_span = "expected internal span"

    verifier = OracleNodeVerifier(LengthPredictor(), tolerance=0.0)
    result = verifier.check_merge_consistency(
        original_content=root.ops_span,
        merged_summary=root.summary,
        rubric="",
        node_id=root.id,
    )

    assert result.law == "merge_consistency"
    assert not result.passed
    assert float(result.expected_label) == pytest.approx(float(len(root.ops_span)))
    assert result.original_prediction is not None
    assert result.original_prediction.label == str(len(root.ops_span))


def test_tree_verifier_prefers_ops_span_over_child_aggregation():
    left = leaf("aa", summary="x")
    right = leaf("bbb", summary="y")
    root = node(left, right, summary="zzz")
    root.ops_span = "much longer theorem span"

    verifier = TreeVerifier(LengthPredictor(), tolerance=0.0)
    results = verifier.verify_tree(root, rubric="")
    root_result = results[root.id]

    assert "merge_consistency" in root_result.law_results
    merge_result = root_result.law_results["merge_consistency"]
    assert not merge_result.passed
    assert float(merge_result.expected_label) == pytest.approx(float(len(root.ops_span)))


def test_idempotence_check_can_use_theorem_operator_resummary():
    operator = FunctionalSketchLawOperator[str, str](
        name="upper_canonical",
        encode_fn=lambda span: span.upper(),
        merge_fn=lambda left, right: f"{left}|{right}",
        decode_fn=lambda sketch: sketch,
        combine_fn=lambda left, right: f"{left}|{right}",
        evidence_status=EvidenceStatus.THEOREM_BACKED,
        theorem_domain_decode_available=True,
        theorem_domain_reencode_available=True,
        idempotence_law=LawCapabilityReport(
            law_kind=LawKind.L3_IDEMPOTENCE,
            available=True,
            evidence_status=EvidenceStatus.THEOREM_BACKED,
            exact=True,
        ),
    )

    verifier = OracleNodeVerifier(
        LengthPredictor(),
        tolerance=0.0,
        theorem_operator=operator,
    )
    result = verifier.check_idempotence(summary="abc", rubric="")

    assert result.skipped is False
    assert result.passed is True
