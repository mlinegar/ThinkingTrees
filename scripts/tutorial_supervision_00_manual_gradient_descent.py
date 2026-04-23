#!/usr/bin/env python3
"""CPU-only walkthrough: hand-written gradient descent, then the shared surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.config_sections import OptimizerConfig, RunConfig, RuntimeConfig, TrainConfig
from src.training.supervision import (
    DenseScalarModelConfig,
    DenseScalarTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    fit_dense_scalar_regressor,
    predict_dense_scalar_regressor,
)


def _tiny_regression_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    y_train = 3.0 * x_train + 1.0
    x_eval = np.asarray([-1.5, 0.5, 1.5], dtype=np.float64)
    return x_train, y_train, x_eval


def _run_manual_gradient_descent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    learning_rate: float = 0.05,
    num_steps: int = 300,
) -> dict[str, object]:
    weight = 0.0
    bias = 0.0
    first_loss = None

    for step in range(num_steps):
        predictions = weight * x_train + bias
        errors = predictions - y_train
        loss = float(np.mean(errors ** 2))
        if step == 0:
            first_loss = loss

        grad_weight = float(2.0 * np.mean(errors * x_train))
        grad_bias = float(2.0 * np.mean(errors))

        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

    final_predictions = weight * x_eval + bias
    final_train_loss = float(np.mean(((weight * x_train + bias) - y_train) ** 2))
    return {
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "initial_loss": round(float(first_loss), 6),
        "final_loss": round(final_train_loss, 6),
        "weight": round(float(weight), 4),
        "bias": round(float(bias), 4),
        "predictions": [round(float(value), 4) for value in final_predictions.tolist()],
        "update_rule": "w <- w - lr * d/dw MSE, b <- b - lr * d/db MSE",
    }


def _run_shared_supervision_sgd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> dict[str, object]:
    rows = [
        DenseSupervisionExample(
            example_id=f"x_{idx}",
            features=[float(x_value)],
            scalar_target=float(y_value),
        )
        for idx, (x_value, y_value) in enumerate(zip(x_train.tolist(), y_train.tolist(), strict=True))
    ]
    supervision = build_dense_full_document_supervision_dataset(
        rows,
        application_name="tutorial_manual_gradient_descent",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )
    model, result = fit_dense_scalar_regressor(
        supervision,
        config=DenseScalarTrainingConfig(
            model=DenseScalarModelConfig(hidden_dims=tuple()),
            optimizer=OptimizerConfig(learning_rate=0.05),
            train=TrainConfig(batch_size=len(rows), epochs=300),
            runtime=RuntimeConfig(device="cpu"),
            run=RunConfig(seed=7),
        ),
    )
    predictions = predict_dense_scalar_regressor(
        model,
        rows=[
            {"features": [float(x_value)], "score": 0.0, "sample_weight": 1.0}
            for x_value in x_eval.tolist()
        ],
        device="cpu",
    )
    return {
        "training_surface": "supervision_dataset",
        "optimizer_family": "dense_scalar_sgd",
        "epochs": result.epochs_completed,
        "final_loss": round(float(result.train_loss_final), 6),
        "predictions": [round(float(value), 4) for value in predictions.tolist()],
    }


def run_example() -> dict[str, object]:
    x_train, y_train, x_eval = _tiny_regression_problem()
    return {
        "example": "manual_gradient_descent_bridge",
        "target_rule": "y = 3x + 1",
        "train_x": [float(value) for value in x_train.tolist()],
        "train_y": [float(value) for value in y_train.tolist()],
        "eval_x": [float(value) for value in x_eval.tolist()],
        "manual_gradient_descent": _run_manual_gradient_descent(x_train, y_train, x_eval),
        "shared_supervision_sgd": _run_shared_supervision_sgd(x_train, y_train, x_eval),
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
