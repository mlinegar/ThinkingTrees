#!/usr/bin/env python3
"""CPU-only walkthrough: sampled local labels with logged propensities and IPW."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.training.supervision import (
    DenseScalarRidgeModelConfig,
    DenseScalarRidgeTrainingConfig,
    DenseSupervisionExample,
    build_dense_full_document_supervision_dataset,
    build_dense_sampled_substructure_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_ridge_regressor,
)


FULL_LOCAL_OFFSETS = (-1.5, -0.5, 0.5, 1.5)
OBSERVED_OFFSETS_AND_PROPENSITIES = (
    (-0.5, 0.25),
    (1.5, 0.75),
)


def _base_target(x_value: float) -> float:
    return 3.0 * x_value + 1.0


def _build_datasets():
    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    full_document_rows: list[DenseSupervisionExample] = []
    sampled_rows: list[DenseSupervisionExample] = []
    per_document = []

    for idx, x_value in enumerate(x_values):
        doc_id = f"doc_{idx}"
        doc_target = _base_target(x_value)
        full_local_targets = [doc_target + offset for offset in FULL_LOCAL_OFFSETS]

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

        observed_targets = []
        observed_weights = []
        observed_units = []
        for local_idx, (offset, propensity) in enumerate(OBSERVED_OFFSETS_AND_PROPENSITIES):
            local_target = doc_target + offset
            sample_weight = 1.0 / propensity
            unit_id = f"{doc_id}:leaf_{local_idx}"
            sampled_rows.append(
                DenseSupervisionExample(
                    example_id=unit_id,
                    source_doc_id=doc_id,
                    features=[x_value],
                    scalar_target=local_target,
                    original_text=f"observed local piece {local_idx} at x={x_value}",
                    response="leaf_score",
                    response_id=unit_id,
                    unit_kind=ObservationUnitKind.LEAF,
                    sampling=SamplingMetadata(
                        document_propensity=1.0,
                        unit_propensity=propensity,
                        label_propensity=1.0,
                        sampling_scheme="biased_sampled_substructure_supervision",
                        policy_name="favor_high_local_offsets",
                        unit_kind=ObservationUnitKind.LEAF,
                        supports_ipw_estimation=True,
                        metadata={"local_offset": offset},
                    ),
                    metadata={"local_offset": offset},
                )
            )
            observed_targets.append(local_target)
            observed_weights.append(sample_weight)
            observed_units.append(
                {
                    "unit_id": unit_id,
                    "local_offset": offset,
                    "observed_target": round(float(local_target), 4),
                    "logged_propensity": propensity,
                    "ipw_weight": round(float(sample_weight), 4),
                }
            )

        naive_mean = float(np.mean(observed_targets))
        normalized_ipw_mean = float(
            np.average(
                np.asarray(observed_targets, dtype=np.float64),
                weights=np.asarray(observed_weights, dtype=np.float64),
            )
        )
        per_document.append(
            {
                "document_id": doc_id,
                "x": x_value,
                "document_target": round(float(doc_target), 4),
                "full_local_targets": [round(float(value), 4) for value in full_local_targets],
                "observed_units": observed_units,
                "naive_sample_mean": round(naive_mean, 4),
                "normalized_ipw_mean": round(normalized_ipw_mean, 4),
            }
        )

    full_document_supervision = build_dense_full_document_supervision_dataset(
        full_document_rows,
        application_name="tutorial_sampled_local_ipw",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )
    sampled_substructure_supervision = build_dense_sampled_substructure_supervision_dataset(
        sampled_rows,
        application_name="tutorial_sampled_local_ipw",
        supervision_signal_name="substructure_level_target",
        response_signal_name="y",
        law_type="substructure_level_target",
        split="train",
    )
    return per_document, full_document_supervision, sampled_substructure_supervision


def _fit_predictions(supervision, eval_x: np.ndarray, *, sample_weights=None):
    model, result = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
        sample_weights=sample_weights,
    )
    predictions = predict_dense_scalar_ridge_regressor(model, eval_x)
    return {
        "n_rows": result.n_train_rows,
        "weights": [round(float(value), 4) for value in model.weights.tolist()],
        "bias": round(float(model.bias), 4),
        "predictions": [round(float(value), 4) for value in predictions.tolist()],
    }


def run_example() -> dict[str, object]:
    per_document, full_document_supervision, sampled_substructure_supervision = _build_datasets()
    eval_x = np.asarray([[-1.5], [0.5], [1.5]], dtype=np.float64)
    n_sampled_rows = len(sampled_substructure_supervision.response_judgments)

    full_document_fit = _fit_predictions(full_document_supervision, eval_x)
    naive_sampled_fit = _fit_predictions(
        sampled_substructure_supervision,
        eval_x,
        sample_weights=np.ones(n_sampled_rows, dtype=np.float64),
    )
    ipw_sampled_fit = _fit_predictions(sampled_substructure_supervision, eval_x)

    return {
        "example": "sampled_local_ipw",
        "target_rule": "document target = 3x + 1",
        "full_local_offsets": list(FULL_LOCAL_OFFSETS),
        "observed_offsets_and_propensities": [
            {"local_offset": offset, "logged_propensity": propensity}
            for offset, propensity in OBSERVED_OFFSETS_AND_PROPENSITIES
        ],
        "per_document": per_document,
        "eval_x": [float(row[0]) for row in eval_x.tolist()],
        "full_document_reference_fit": full_document_fit,
        "naive_sampled_fit": naive_sampled_fit,
        "ipw_sampled_fit": ipw_sampled_fit,
        "ipw_formula": (
            "normalized_ipw_mean = sum(observed_target / propensity) / "
            "sum(1 / propensity)"
        ),
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
