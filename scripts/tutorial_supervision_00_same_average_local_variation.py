#!/usr/bin/env python3
"""CPU-only walkthrough: local variation with the same document-level average."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logged_supervision import ObservationUnitKind
from src.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    build_dense_sampled_substructure_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_ridge_regressor,
)


LOCAL_OFFSETS = (-1.5, -0.5, 0.5, 1.5)


def _base_target(x_value: float) -> float:
    return 3.0 * x_value + 1.0


def _build_datasets():
    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    full_document_rows: list[DenseSupervisionExample] = []
    local_rows: list[DenseSupervisionExample] = []
    per_document = []

    for idx, x_value in enumerate(x_values):
        doc_id = f"doc_{idx}"
        doc_target = _base_target(x_value)
        local_targets = [doc_target + offset for offset in LOCAL_OFFSETS]

        full_document_rows.append(
            DenseSupervisionExample(
                example_id=doc_id,
                source_doc_id=doc_id,
                features=[x_value],
                scalar_target=doc_target,
                original_text=f"toy document at x={x_value}",
                response="document_score",
                response_id=f"{doc_id}:document_score",
            )
        )

        for local_idx, local_target in enumerate(local_targets):
            local_rows.append(
                DenseSupervisionExample(
                    example_id=f"{doc_id}:leaf_{local_idx}",
                    source_doc_id=doc_id,
                    features=[x_value],
                    scalar_target=local_target,
                    original_text=f"toy leaf {local_idx} at x={x_value}",
                    response="leaf_score",
                    response_id=f"{doc_id}:leaf_{local_idx}",
                    unit_kind=ObservationUnitKind.LEAF,
                    metadata={"leaf_index": local_idx},
                )
            )

        per_document.append(
            {
                "document_id": doc_id,
                "x": x_value,
                "document_target": doc_target,
                "local_targets": [round(float(value), 4) for value in local_targets],
                "local_average": round(float(np.mean(local_targets)), 4),
            }
        )

    full_document_supervision = build_dense_full_document_supervision_dataset(
        full_document_rows,
        application_name="tutorial_same_average_local_variation",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )
    sampled_substructure_supervision = build_dense_sampled_substructure_supervision_dataset(
        local_rows,
        application_name="tutorial_same_average_local_variation",
        supervision_signal_name="substructure_level_target",
        response_signal_name="y",
        law_type="substructure_level_target",
        split="train",
    )
    return per_document, full_document_supervision, sampled_substructure_supervision


def run_example() -> dict[str, object]:
    per_document, full_document_supervision, sampled_substructure_supervision = _build_datasets()
    eval_x = np.asarray([[-1.5], [0.5], [1.5]], dtype=np.float64)

    full_document_model, full_document_result = fit_dense_scalar_ridge_regressor(
        full_document_supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
    )
    substructure_model, substructure_result = fit_dense_scalar_ridge_regressor(
        sampled_substructure_supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
    )

    full_document_predictions = predict_dense_scalar_ridge_regressor(
        full_document_model,
        eval_x,
    )
    substructure_predictions = predict_dense_scalar_ridge_regressor(
        substructure_model,
        eval_x,
    )

    return {
        "example": "same_average_local_variation",
        "target_rule": "document target = 3x + 1",
        "local_offsets": list(LOCAL_OFFSETS),
        "per_document": per_document,
        "full_document_supervision": {
            "n_rows": full_document_result.n_train_rows,
            "predictions": [round(float(value), 4) for value in full_document_predictions.tolist()],
            "weights": [round(float(value), 4) for value in full_document_model.weights.tolist()],
            "bias": round(float(full_document_model.bias), 4),
        },
        "sampled_substructure_supervision": {
            "n_rows": substructure_result.n_train_rows,
            "predictions": [round(float(value), 4) for value in substructure_predictions.tolist()],
            "weights": [round(float(value), 4) for value in substructure_model.weights.tolist()],
            "bias": round(float(substructure_model.bias), 4),
        },
        "eval_x": [float(row[0]) for row in eval_x.tolist()],
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
