from __future__ import annotations

import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.supervision_metadata import judgment_supervision_metadata
from src.training.config_sections import OptimizerConfig, RunConfig, RuntimeConfig, TrainConfig
from src.training.supervision import (
    DenseScalarModelConfig,
    DenseScalarTrainingConfig,
    ResponseJudgment,
    SupervisionDataset,
    fit_dense_scalar_regressor,
    predict_dense_scalar_regressor,
)


def _judgment(idx: int, *, x0: float, x1: float, y: float) -> ResponseJudgment:
    return ResponseJudgment(
        judgment_id=f"j{idx}",
        source_example_id=f"doc{idx}",
        original_text=f"doc {idx}",
        rubric="predict scalar target",
        response="dense_candidate",
        response_id=f"cand{idx}",
        reference_score=float(y),
        law_type="document_level_target",
        truth_label_source="oracle",
        sampling=SamplingMetadata(
            joint_propensity=1.0,
            unit_kind=ObservationUnitKind.DOCUMENT,
        ),
        supervision_metadata=judgment_supervision_metadata(
            application_name="dense_scalar_test",
            law_type="document_level_target",
            supervision_channel_name="full_document_supervision",
            supervision_signal_name="document_level_target",
            response_signal_name="target_value",
        ),
        response_signal_value=float(y),
        candidate_features=[float(x0), float(x1)],
    )


def test_dense_scalar_regressor_fits_simple_linear_supervision() -> None:
    train = SupervisionDataset(
        response_judgments=[
            _judgment(0, x0=0.0, x1=0.0, y=0.0),
            _judgment(1, x0=1.0, x1=0.0, y=2.0),
            _judgment(2, x0=0.0, x1=1.0, y=3.0),
            _judgment(3, x0=1.0, x1=1.0, y=5.0),
        ]
    )
    val = SupervisionDataset(
        response_judgments=[
            _judgment(10, x0=2.0, x1=1.0, y=7.0),
            _judgment(11, x0=1.0, x1=2.0, y=8.0),
        ]
    )

    model, fit = fit_dense_scalar_regressor(
        train,
        val_supervision=val,
        config=DenseScalarTrainingConfig(
            model=DenseScalarModelConfig(hidden_dims=tuple()),
            optimizer=OptimizerConfig(learning_rate=5e-2, weight_decay=0.0),
            train=TrainConfig(batch_size=2, epochs=250),
            runtime=RuntimeConfig(device="cpu"),
            run=RunConfig(seed=7),
        ),
    )
    pred = predict_dense_scalar_regressor(model, supervision=val, device="cpu")

    assert fit.selection_mode == "best_val_dense_scalar_loss"
    assert fit.input_dim == 2
    assert len(pred) == 2
    assert pred == pytest.approx([7.0, 8.0], abs=0.2)
