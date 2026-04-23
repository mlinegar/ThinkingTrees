#!/usr/bin/env python3
"""CPU-only walkthrough: fit a tiny ridge regressor on SupervisionDataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_ridge_regressor,
)


def run_example() -> dict[str, object]:
    train_rows = [
        DenseSupervisionExample(example_id="doc_0", features=[0.0, 0.0], scalar_target=0.2),
        DenseSupervisionExample(example_id="doc_1", features=[1.0, 0.0], scalar_target=1.7),
        DenseSupervisionExample(example_id="doc_2", features=[0.0, 1.0], scalar_target=-0.3),
        DenseSupervisionExample(example_id="doc_3", features=[1.0, 1.0], scalar_target=1.2),
        DenseSupervisionExample(example_id="doc_4", features=[2.0, 0.0], scalar_target=3.2),
        DenseSupervisionExample(example_id="doc_5", features=[0.0, 2.0], scalar_target=-0.8),
    ]
    supervision = build_dense_full_document_supervision_dataset(
        train_rows,
        application_name="tutorial_dense_scalar_regression",
        supervision_signal_name="document_level_target",
        response_signal_name="score",
        law_type="document_level_target",
        split="train",
    )
    model, training_result = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
    )
    eval_features = np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
    predictions = predict_dense_scalar_ridge_regressor(model, eval_features)
    return {
        "example": "dense_scalar_regression",
        "training_surface": "supervision_dataset",
        "optimizer_family": "closed_form_scalar_ridge",
        "n_train_rows": training_result.n_train_rows,
        "input_dim": training_result.input_dim,
        "learned_weights": [round(float(value), 4) for value in model.weights.tolist()],
        "learned_bias": round(float(model.bias), 4),
        "eval_predictions": [round(float(value), 4) for value in predictions.tolist()],
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
