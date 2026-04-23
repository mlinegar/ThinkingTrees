from __future__ import annotations

import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.supervision_metadata import judgment_supervision_metadata
from src.training.collect_preferences import save_results
from src.training.preference.optimizer_adapters import (
    build_dense_scalar_training_records,
    build_scalar_reward_training_records,
)
from src.training.preference.types import PreferencePair
from src.training.supervision import (
    OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION,
    REPRESENTATION_SIMPLEX_VECTOR,
    ResponseJudgment,
    SupervisionDataset,
    TARGET_SIMPLEX_VECTOR,
    coerce_supervision_dataset,
    supervision_training_contract,
)
from src.training.train_model import create_parser


def _make_response_judgment(
    judgment_id: str,
    *,
    example_id: str = "doc1",
    response_id: str = "A",
    response: str = "summary a",
    score: float = 0.8,
    law_type: str = "sufficiency",
    candidate_features: tuple[float, ...] | None = None,
) -> ResponseJudgment:
    return ResponseJudgment(
        judgment_id=judgment_id,
        source_example_id=example_id,
        original_text=f"original {example_id}",
        rubric="rubric",
        response=response,
        response_id=response_id,
        reference_score=0.5,
        law_type=law_type,
        sampling=SamplingMetadata(
            joint_propensity=0.25,
            unit_kind=ObservationUnitKind.PAIR,
        ),
        supervision_metadata=judgment_supervision_metadata(
            application_name="test_supervision",
            law_type=law_type,
            response_signal_name="judge_score",
            response_signal_min=0.0,
            response_signal_max=1.0,
        ),
        response_signal_value=score,
        candidate_features=list(candidate_features) if candidate_features is not None else None,
        judge_model="judge",
    )


def test_response_judgment_roundtrip(tmp_path) -> None:
    dataset = SupervisionDataset(
        response_judgments=[_make_response_judgment("j1")],
    )
    path = tmp_path / "supervision.json"
    dataset.save(path)

    loaded = SupervisionDataset.load(path)
    assert len(loaded.response_judgments) == 1
    judgment = loaded.response_judgments[0]
    assert judgment.judgment_id == "j1"
    assert judgment.response_signal_value == pytest.approx(0.8)
    assert judgment.supervision_metadata.response_signal_name == "judge_score"


def test_response_judgment_roundtrip_preserves_dense_candidate_features(tmp_path) -> None:
    dataset = SupervisionDataset(
        response_judgments=[
            _make_response_judgment(
                "j_dense",
                candidate_features=(0.1, 0.2, 0.3),
            )
        ],
    )
    path = tmp_path / "supervision_dense.json"
    dataset.save(path)

    loaded = SupervisionDataset.load(path)
    judgment = loaded.response_judgments[0]
    assert judgment.candidate_features == pytest.approx([0.1, 0.2, 0.3])


def test_supervision_dataset_groups_scalar_judgments_deterministically() -> None:
    dataset = SupervisionDataset(
        response_judgments=[
            _make_response_judgment("j1", response_id="A", response="summary a", score=0.9),
            _make_response_judgment("j2", response_id="B", response="summary b", score=0.6),
            _make_response_judgment("j3", response_id="C", response="summary c", score=0.2),
        ]
    )

    comparative_dataset = dataset.to_comparative_dataset()
    assert len(comparative_dataset.records) == 1
    record = comparative_dataset.records[0]
    assert [candidate.response for candidate in record.candidates] == [
        "summary a",
        "summary b",
        "summary c",
    ]
    assert [candidate.rank for candidate in record.candidates] == [1, 2, 3]
    assert record.aggregate_sample_weight == pytest.approx(12.0)


def test_supervision_dataset_project_binary_modes() -> None:
    dataset = SupervisionDataset(
        response_judgments=[
            _make_response_judgment("j1", response_id="A", response="summary a", score=0.9),
            _make_response_judgment("j2", response_id="B", response="summary b", score=0.6),
            _make_response_judgment("j3", response_id="C", response="summary c", score=0.2),
        ]
    )

    adjacent = dataset.project_binary(projection="adjacent")
    runner_up = dataset.project_binary(projection="winner_vs_runner_up")

    assert len(adjacent.pairs) == 2
    assert len(runner_up.pairs) == 1
    assert adjacent.pairs[0].summary_a == "summary a"
    assert adjacent.pairs[0].summary_b == "summary b"
    assert runner_up.pairs[0].summary_b == "summary b"


def test_supervision_dataset_builds_scalar_reward_rows_without_multiplying_group_mass() -> None:
    dataset = SupervisionDataset(
        response_judgments=[
            _make_response_judgment("j1", response_id="A", response="summary a", score=0.9),
            _make_response_judgment("j2", response_id="B", response="summary b", score=0.6),
            _make_response_judgment("j3", response_id="C", response="summary c", score=0.2),
        ]
    )

    rows = build_scalar_reward_training_records(dataset)
    assert len(rows) == 3
    assert [row["response"] for row in rows] == ["summary a", "summary b", "summary c"]
    assert sum(float(row["sample_weight"]) for row in rows) == pytest.approx(12.0)


def test_supervision_dataset_builds_dense_scalar_rows_from_featured_judgments() -> None:
    dataset = SupervisionDataset(
        response_judgments=[
            _make_response_judgment(
                "j1",
                response_id="A",
                score=1.0,
                candidate_features=(1.0, 0.0),
            ),
            _make_response_judgment(
                "j2",
                response_id="B",
                score=2.0,
                candidate_features=(0.0, 1.0),
            ),
        ]
    )

    rows = build_dense_scalar_training_records(dataset)
    assert len(rows) == 2
    assert rows[0]["features"] == pytest.approx([1.0, 0.0])
    assert rows[1]["features"] == pytest.approx([0.0, 1.0])
    assert [row["score"] for row in rows] == pytest.approx([1.0, 2.0])


def test_supervision_dataset_builds_dense_vector_rows_from_featured_judgments() -> None:
    dataset = SupervisionDataset(
        response_judgments=[
            _make_response_judgment(
                "jv1",
                response_id="A",
                score=0.0,
                candidate_features=(1.0, 0.0),
            ),
            _make_response_judgment(
                "jv2",
                response_id="B",
                score=0.0,
                candidate_features=(0.0, 1.0),
            ),
        ]
    )
    dataset.response_judgments[0].response_signal_vector = [1.0, 0.0]
    dataset.response_judgments[1].response_signal_vector = [0.0, 1.0]

    rows = dataset.to_dense_vector_training_records()
    assert len(rows) == 2
    assert rows[0]["features"] == pytest.approx([1.0, 0.0])
    assert rows[0]["target"] == pytest.approx([1.0, 0.0])
    assert rows[1]["target"] == pytest.approx([0.0, 1.0])


def test_supervision_training_contract_supports_prefixed_vector_calibration() -> None:
    contract = supervision_training_contract(
        prefix="calibration",
        representation_kind=REPRESENTATION_SIMPLEX_VECTOR,
        target_kind=TARGET_SIMPLEX_VECTOR,
        optimizer_family=OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION,
        optimizer_backend="closed_form_affine_ridge",
        n_train_rows=7,
    )
    assert contract["calibration_representation_kind"] == REPRESENTATION_SIMPLEX_VECTOR
    assert contract["calibration_target_kind"] == TARGET_SIMPLEX_VECTOR
    assert contract["calibration_optimizer_family"] == OPTIMIZER_FAMILY_AFFINE_VECTOR_CALIBRATION
    assert contract["calibration_supervision_mode"] == "dense_affine_simplex_calibration"
    assert contract["calibration_supervision_rows"] == 7


def test_coerce_supervision_dataset_accepts_preference_pairs() -> None:
    pair = PreferencePair(
        pair_id="p1",
        source_example_id="doc_pref",
        original_text="original pref",
        rubric="rubric",
        reference_score=0.4,
        summary_a="summary a",
        summary_b="summary b",
        preferred="A",
        reasoning="a beats b",
        confidence=0.9,
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
    )

    dataset = coerce_supervision_dataset([pair])
    assert len(dataset.comparative_judgments) == 1
    projected = dataset.project_binary(projection="adjacent")
    assert len(projected.pairs) == 1
    assert projected.pairs[0].pair_id.startswith("p1")


def test_collect_preferences_save_results_writes_primary_supervision_artifact(tmp_path) -> None:
    pair = PreferencePair(
        pair_id="p_save",
        source_example_id="doc_save",
        original_text="original save",
        rubric="rubric",
        reference_score=0.5,
        summary_a="summary a",
        summary_b="summary b",
        preferred="A",
        reasoning="reason",
        confidence=0.8,
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
    )

    class DummyCollector:
        def get_supervision_dataset(self):
            return SupervisionDataset(
                comparative_judgments=[pair.to_comparative_judgment()]
            )

        def get_dataset(self):
            from src.training.preference.types import PreferenceDataset

            return PreferenceDataset([pair])

        def get_comparative_dataset(self):
            from src.training.preference.types import ComparativeDataset

            return ComparativeDataset([])

        def get_statistics(self):
            return {"pairs": {"collected": 1}, "preferences": {"prefer_a": 1}}

    class _JudgeType:
        value = "dspy"

    class _Judge:
        judge_type = _JudgeType()

    class DummyConfig:
        output_dir = tmp_path
        output_prefix = "demo"
        task_name = "task"
        judge = _Judge()
        save_dpo_format = False
        law_type = "sufficiency"

        @staticmethod
        def to_dict():
            return {"task_name": "task"}

    output_path = save_results(DummyCollector(), DummyConfig())
    assert output_path.name.startswith("demo_supervision_")
    loaded = SupervisionDataset.load(output_path)
    assert len(loaded.comparative_judgments) == 1


def test_train_model_parser_accepts_supervision_data_flag() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "--type",
            "ops-comparison",
            "--supervision-data",
            "supervision.json",
        ]
    )
    assert args.supervision_data.name == "supervision.json"
