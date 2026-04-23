#!/usr/bin/env python3
"""CPU-only walkthrough: weighted SGD matches duplicated-row SGD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.training.config_sections import OptimizerConfig, RunConfig, RuntimeConfig, TrainConfig
from src.training.supervision import (
    DenseScalarModelConfig,
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseScalarTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    fit_dense_scalar_regressor,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_regressor,
    predict_dense_scalar_ridge_regressor,
)


BASE_ROWS = [
    (-1.0, -1.0, 1),
    (0.0, 1.0, 4),
    (2.0, 5.0, 2),
]


def _build_weighted_supervision():
    rows = []
    for idx, (x_value, y_value, sample_weight) in enumerate(BASE_ROWS):
        rows.append(
            DenseSupervisionExample(
                example_id=f"weighted_{idx}",
                source_doc_id=f"weighted_{idx}",
                features=[float(x_value)],
                scalar_target=float(y_value),
                original_text=f"weighted row {idx}",
                response="weighted_candidate",
                response_id=f"weighted_{idx}",
                unit_kind=ObservationUnitKind.DOCUMENT,
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0 / float(sample_weight),
                    label_propensity=1.0,
                    sampling_scheme="integer_weight_example",
                    policy_name="fixed_integer_weights",
                    unit_kind=ObservationUnitKind.DOCUMENT,
                    supports_ipw_estimation=True,
                ),
            )
        )
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="tutorial_weighted_sgd_equivalence",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )


def _build_duplicated_supervision():
    rows = []
    row_index = 0
    for x_value, y_value, sample_weight in BASE_ROWS:
        for _ in range(sample_weight):
            rows.append(
                DenseSupervisionExample(
                    example_id=f"duplicate_{row_index}",
                    source_doc_id=f"duplicate_{row_index}",
                    features=[float(x_value)],
                    scalar_target=float(y_value),
                    original_text=f"duplicated row {row_index}",
                    response="duplicated_candidate",
                    response_id=f"duplicate_{row_index}",
                )
            )
            row_index += 1
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="tutorial_weighted_sgd_equivalence",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )


def run_example() -> dict[str, object]:
    weighted_supervision = _build_weighted_supervision()
    duplicated_supervision = _build_duplicated_supervision()
    eval_x = np.asarray([[-1.5], [0.5], [1.5]], dtype=np.float64)

    config = DenseScalarTrainingConfig(
        model=DenseScalarModelConfig(hidden_dims=tuple()),
        optimizer=OptimizerConfig(learning_rate=5e-2, weight_decay=0.0),
        train=TrainConfig(batch_size=32, epochs=600),
        runtime=RuntimeConfig(device="cpu"),
        run=RunConfig(seed=13),
    )
    weighted_model, weighted_result = fit_dense_scalar_regressor(
        weighted_supervision,
        config=config,
    )
    duplicated_model, duplicated_result = fit_dense_scalar_regressor(
        duplicated_supervision,
        config=config,
    )
    ridge_model, _ = fit_dense_scalar_ridge_regressor(
        weighted_supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
    )

    weighted_predictions = predict_dense_scalar_regressor(
        weighted_model,
        rows=[{"features": row.tolist(), "score": 0.0, "sample_weight": 1.0} for row in eval_x],
        device="cpu",
    )
    duplicated_predictions = predict_dense_scalar_regressor(
        duplicated_model,
        rows=[{"features": row.tolist(), "score": 0.0, "sample_weight": 1.0} for row in eval_x],
        device="cpu",
    )
    ridge_predictions = predict_dense_scalar_ridge_regressor(ridge_model, eval_x)
    max_abs_diff = float(np.max(np.abs(weighted_predictions - duplicated_predictions)))

    return {
        "example": "weighted_sgd_equivalence",
        "base_rows": [
            {"x": x_value, "y": y_value, "sample_weight": sample_weight}
            for x_value, y_value, sample_weight in BASE_ROWS
        ],
        "eval_x": [float(row[0]) for row in eval_x.tolist()],
        "weighted_sgd": {
            "epochs": weighted_result.epochs_completed,
            "final_loss": round(float(weighted_result.train_loss_final), 6),
            "predictions": [round(float(value), 4) for value in weighted_predictions.tolist()],
        },
        "duplicated_sgd": {
            "epochs": duplicated_result.epochs_completed,
            "final_loss": round(float(duplicated_result.train_loss_final), 6),
            "predictions": [round(float(value), 4) for value in duplicated_predictions.tolist()],
        },
        "weighted_ridge_reference": {
            "predictions": [round(float(value), 4) for value in ridge_predictions.tolist()],
        },
        "max_abs_prediction_diff_weighted_vs_duplicated": round(max_abs_diff, 6),
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
