from __future__ import annotations

import importlib
from pathlib import Path
import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.supervision_metadata import judgment_supervision_metadata
from src.training.audit_driven_training import AuditDrivenConfig, AuditDrivenTrainer
from src.training.supervision import (
    BinaryComparison,
    SupervisionDataset,
    save_supervision_artifact_bundle,
)
from src.training.unified_trainer import UnifiedTrainer, UnifiedTrainerConfig


def _pair(pair_id: str = "pair_public") -> BinaryComparison:
    return BinaryComparison(
        pair_id=pair_id,
        source_example_id="doc_public",
        original_text="original text",
        rubric="rubric",
        reference_score=0.5,
        summary_a="summary a",
        summary_b="summary b",
        preferred="A",
        confidence=0.9,
        reasoning="a wins",
        law_type="sufficiency",
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
        preference_supervision=judgment_supervision_metadata(
            application_name="public_surface_test",
            law_type="sufficiency",
        ),
    )


def test_training_preference_package_import_fails() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("src.training.preference").PreferencePair


def test_save_supervision_artifact_bundle_writes_primary_and_binary_views(
    tmp_path: Path,
) -> None:
    dataset = SupervisionDataset(
        comparative_judgments=[_pair().to_comparative_judgment()]
    )

    bundle = save_supervision_artifact_bundle(
        dataset,
        supervision_path=tmp_path / "supervision.json",
        binary_projection_path=tmp_path / "preferences.json",
        comparative_path=tmp_path / "comparative.json",
    )

    assert bundle.supervision_path.exists()
    assert bundle.binary_projection_path is not None and bundle.binary_projection_path.exists()
    assert bundle.comparative_path is not None and bundle.comparative_path.exists()
    restored = SupervisionDataset.load(bundle.supervision_path)
    assert len(restored.comparative_judgments) == 1
    assert len(restored.project_binary(projection="adjacent").comparisons) == 1


def test_audit_driven_trainer_collects_supervision_dataset(tmp_path: Path) -> None:
    counter = {"n": 0}

    def summarizer(text: str, rubric: str) -> str:
        counter["n"] += 1
        return f"candidate_{counter['n']}: {text[:8]}"

    class Judge:
        def compare(self, *, context: str, original_text: str, summary_a: str, summary_b: str):
            class Result:
                preferred = "A"
                confidence = 0.8
                reasoning = "A is better"
                helpfulness_a = 0.7
                helpfulness_b = 0.4

            return Result()

    trainer = AuditDrivenTrainer(
        model_name="dummy/model",
        judge=Judge(),
        summarizer=summarizer,
        output_dir=tmp_path / "audit_driven",
        config=AuditDrivenConfig(k_candidates=3),
    )

    supervision = trainer.collect_preferences_for_violations(
        {
            "sufficiency": [{"input_text": "document text", "rubric": "rubric"}],
            "idempotence": [],
            "merge": [],
        }
    )

    assert isinstance(supervision, SupervisionDataset)
    assert len(supervision.comparative_judgments) >= 1
    assert len(supervision.project_binary(projection="adjacent").comparisons) >= 1


def test_unified_trainer_saves_primary_supervision_artifact(tmp_path: Path) -> None:
    class DummyGeneratorTrainer:
        method_name = "dummy"

        def train(self, preferences, model_name, output_dir, **kwargs):
            return str(output_dir)

    class DummyJudge:
        def compare(self, **kwargs):
            class Result:
                preferred = "A"
                confidence = 0.8
                reasoning = "A"

            return Result()

    trainer = UnifiedTrainer(
        generator_trainer=DummyGeneratorTrainer(),
        genrm_judge=DummyJudge(),
        oracle_predict=lambda text: 0.5,
        config=UnifiedTrainerConfig(max_iterations=1),
        output_dir=tmp_path,
        summarizer=lambda text, rubric: f"{rubric}: {text[:12]}",
    )
    trainer.all_supervision_dataset.add_comparative_judgment(_pair("pair_saved").to_comparative_judgment())

    supervision_path, preferences_path = trainer._save_all_supervision()

    assert supervision_path == tmp_path / "all_supervision.json"
    assert preferences_path == tmp_path / "all_binary_projection.json"
    assert supervision_path.exists()
    assert preferences_path.exists()
