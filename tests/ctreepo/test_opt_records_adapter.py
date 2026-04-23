from __future__ import annotations

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.ctreepo.opt.records import PairwisePreference
from src.ctreepo.opt.training_adapter import to_training_preference_dataset


def test_pairwise_preference_to_training_pair_roundtrip() -> None:
    sampling = SamplingMetadata(
        document_propensity=0.5,
        unit_propensity=0.25,
        label_propensity=1.0,
        unit_kind=ObservationUnitKind.PAIR,
    )
    record = PairwisePreference(
        example_id="ex1",
        input="doc",
        rubric="rubric",
        candidate_a="A",
        candidate_b="B",
        preferred="A",
        confidence=0.9,
        reasoning="A closer to oracle",
        reference=3.0,
        score_a=2.9,
        score_b=2.0,
        sampling=sampling,
    )

    pair = record.to_training_preference_pair()
    assert pair.source_example_id == "ex1"
    assert pair.original_text == "doc"
    assert pair.rubric == "rubric"
    assert pair.summary_a == "A"
    assert pair.summary_b == "B"
    assert abs(pair.ipw_weight() - 8.0) < 1e-12
    assert pair.preference_supervision.application_name == "ctreepo_opt_record"

    dataset = to_training_preference_dataset([record])
    assert len(dataset) == 1
