#!/usr/bin/env python3
"""CPU-only walkthrough: a tiny Markov-style learner on the shared supervision surface."""

from __future__ import annotations

import json
import random
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


def _generate_sequence(
    rng: random.Random,
    *,
    length: int,
    p00: float,
    p11: float,
) -> list[int]:
    state = rng.choice([0, 1])
    sequence = [state]
    for _ in range(length - 1):
        if state == 0:
            state = 0 if rng.random() < p00 else 1
        else:
            state = 1 if rng.random() < p11 else 0
        sequence.append(state)
    return sequence


def _transition_features(sequence: list[int]) -> list[float]:
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for left, right in zip(sequence[:-1], sequence[1:]):
        counts[(left, right)] += 1
    total = max(1, len(sequence) - 1)
    return [
        counts[(0, 0)] / total,
        counts[(0, 1)] / total,
        counts[(1, 0)] / total,
        counts[(1, 1)] / total,
    ]


def _make_rows(rng: random.Random, n_rows: int) -> list[DenseSupervisionExample]:
    rows: list[DenseSupervisionExample] = []
    for idx in range(n_rows):
        p00 = rng.uniform(0.55, 0.95)
        p11 = rng.uniform(0.20, 0.90)
        sequence = _generate_sequence(rng, length=48, p00=p00, p11=p11)
        rows.append(
            DenseSupervisionExample(
                example_id=f"markov_doc_{idx}",
                features=_transition_features(sequence),
                scalar_target=sum(sequence) / len(sequence),
                original_text=" ".join(str(value) for value in sequence),
                rubric="Predict the fraction of state-1 tokens from transition features.",
                response="markov_dense_features",
                source_doc_id=f"markov_doc_{idx}",
                truth_label_source="oracle",
            )
        )
    return rows


def run_example() -> dict[str, object]:
    train_rng = random.Random(7)
    eval_rng = random.Random(19)
    train_rows = _make_rows(train_rng, 64)
    eval_rows = _make_rows(eval_rng, 8)

    supervision = build_dense_full_document_supervision_dataset(
        train_rows,
        application_name="tutorial_markov_style_regression",
        supervision_signal_name="document_level_target",
        response_signal_name="fraction_state_1",
        law_type="document_level_target",
        split="train",
        response_signal_min=0.0,
        response_signal_max=1.0,
    )
    model, training_result = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-3)),
    )

    eval_features = np.asarray([row.features for row in eval_rows], dtype=np.float64)
    eval_targets = np.asarray([row.scalar_target for row in eval_rows], dtype=np.float64)
    eval_predictions = predict_dense_scalar_ridge_regressor(model, eval_features)
    mae = float(np.mean(np.abs(eval_predictions - eval_targets)))
    return {
        "example": "markov_style_document_regression",
        "training_surface": "supervision_dataset",
        "optimizer_family": "closed_form_scalar_ridge",
        "n_train_rows": training_result.n_train_rows,
        "input_dim": training_result.input_dim,
        "eval_mae": round(mae, 4),
        "sample_predictions": [
            {
                "target": round(float(target), 4),
                "prediction": round(float(prediction), 4),
            }
            for target, prediction in zip(eval_targets[:3], eval_predictions[:3])
        ],
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
