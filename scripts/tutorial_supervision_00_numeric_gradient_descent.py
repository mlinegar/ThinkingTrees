#!/usr/bin/env python3
"""CPU-only walkthrough: one tiny numeric problem trained a few different ways."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

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


def _dataset():
    train_rows = [
        DenseSupervisionExample(example_id=f"x_{idx}", features=[x], scalar_target=3.0 * x + 1.0)
        for idx, x in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0])
    ]
    eval_x = np.asarray([[-1.5], [0.5], [1.5]], dtype=np.float64)
    supervision = build_dense_full_document_supervision_dataset(
        train_rows,
        application_name="tutorial_numeric_gradient_descent",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )
    return supervision, eval_x


def run_example() -> dict[str, object]:
    supervision, eval_x = _dataset()

    ridge_model, ridge_result = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
    )
    ridge_pred = predict_dense_scalar_ridge_regressor(ridge_model, eval_x)

    linear_model, linear_result = fit_dense_scalar_regressor(
        supervision,
        config=DenseScalarTrainingConfig(
            model=DenseScalarModelConfig(hidden_dims=tuple()),
            optimizer=OptimizerConfig(learning_rate=5e-2),
            train=TrainConfig(batch_size=5, epochs=300),
            runtime=RuntimeConfig(device="cpu"),
            run=RunConfig(seed=7),
        ),
    )
    linear_pred = predict_dense_scalar_regressor(
        linear_model,
        rows=[{"features": row.tolist(), "score": 0.0, "sample_weight": 1.0} for row in eval_x],
        device="cpu",
    )

    mlp_model, mlp_result = fit_dense_scalar_regressor(
        supervision,
        config=DenseScalarTrainingConfig(
            model=DenseScalarModelConfig(hidden_dims=(4,)),
            optimizer=OptimizerConfig(learning_rate=5e-2),
            train=TrainConfig(batch_size=5, epochs=300),
            runtime=RuntimeConfig(device="cpu"),
            run=RunConfig(seed=7),
        ),
    )
    mlp_pred = predict_dense_scalar_regressor(
        mlp_model,
        rows=[{"features": row.tolist(), "score": 0.0, "sample_weight": 1.0} for row in eval_x],
        device="cpu",
    )

    return {
        "example": "numeric_gradient_descent_methods",
        "target_rule": "y = 3x + 1",
        "eval_x": [float(row[0]) for row in eval_x.tolist()],
        "methods": {
            "closed_form_ridge": {
                "predictions": [round(float(v), 4) for v in ridge_pred.tolist()],
                "weights": [round(float(v), 4) for v in ridge_model.weights.tolist()],
                "bias": round(float(ridge_model.bias), 4),
                "n_train_rows": ridge_result.n_train_rows,
            },
            "sgd_linear": {
                "predictions": [round(float(v), 4) for v in linear_pred.tolist()],
                "final_loss": round(float(linear_result.train_loss_final), 6),
                "epochs": linear_result.epochs_completed,
            },
            "sgd_mlp": {
                "predictions": [round(float(v), 4) for v in mlp_pred.tolist()],
                "final_loss": round(float(mlp_result.train_loss_final), 6),
                "epochs": mlp_result.epochs_completed,
            },
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
