#!/usr/bin/env python3
"""CPU-only simulation: naive vs IPW regression with biased local labels."""

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


LOCAL_OFFSETS = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
SAMPLING_PROBABILITIES = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
TRAIN_X = np.linspace(-2.5, 2.5, 21, dtype=np.float64)
EVAL_X = np.asarray([[-1.5], [-0.5], [0.5], [1.5]], dtype=np.float64)
N_TRIALS = 300
TRUE_DOCUMENT_TARGET = 1.0


def _target_rule(x_value: float) -> float:
    del x_value
    return TRUE_DOCUMENT_TARGET


def _build_full_document_supervision():
    rows = [
        DenseSupervisionExample(
            example_id=f"doc_{idx}",
            source_doc_id=f"doc_{idx}",
            features=[float(x_value)],
            scalar_target=_target_rule(float(x_value)),
            original_text=f"toy document at x={float(x_value):.2f}",
            response="document_score",
            response_id=f"doc_{idx}:document_score",
        )
        for idx, x_value in enumerate(TRAIN_X.tolist())
    ]
    return build_dense_full_document_supervision_dataset(
        rows,
        application_name="tutorial_ipw_regression_simulation",
        supervision_signal_name="document_level_target",
        response_signal_name="y",
        law_type="document_level_target",
        split="train",
    )


def _build_sampled_supervision(rng: np.random.Generator):
    rows: list[DenseSupervisionExample] = []
    sampled_preview = []
    for idx, x_value in enumerate(TRAIN_X.tolist()):
        sampled_index = int(
            rng.choice(
                len(LOCAL_OFFSETS),
                p=SAMPLING_PROBABILITIES,
            )
        )
        offset = float(LOCAL_OFFSETS[sampled_index])
        propensity = float(SAMPLING_PROBABILITIES[sampled_index])
        sampled_target = _target_rule(float(x_value)) + offset
        row = DenseSupervisionExample(
            example_id=f"doc_{idx}:leaf",
            source_doc_id=f"doc_{idx}",
            features=[float(x_value)],
            scalar_target=float(sampled_target),
            original_text=f"sampled local label at x={float(x_value):.2f}",
            response="leaf_score",
            response_id=f"doc_{idx}:leaf",
            unit_kind=ObservationUnitKind.LEAF,
            sampling=SamplingMetadata(
                document_propensity=1.0,
                unit_propensity=propensity,
                label_propensity=1.0,
                sampling_scheme="categorical_local_sampling",
                policy_name="favor_high_local_offsets",
                unit_kind=ObservationUnitKind.LEAF,
                supports_ipw_estimation=True,
                metadata={"local_offset": offset},
            ),
            metadata={"local_offset": offset},
        )
        rows.append(row)
        if idx < 5:
            sampled_preview.append(
                {
                    "document_id": f"doc_{idx}",
                    "x": round(float(x_value), 4),
                    "sampled_offset": round(offset, 4),
                    "logged_propensity": round(propensity, 4),
                    "sampled_target": round(float(sampled_target), 4),
                }
            )
    supervision = build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="tutorial_ipw_regression_simulation",
        supervision_signal_name="substructure_level_target",
        response_signal_name="y",
        law_type="substructure_level_target",
        split="train",
    )
    return supervision, sampled_preview


def _fit_summary(supervision, *, sample_weights=None):
    model, _ = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
        sample_weights=sample_weights,
    )
    predictions = predict_dense_scalar_ridge_regressor(model, EVAL_X)
    eval_targets = np.asarray([_target_rule(float(row[0])) for row in EVAL_X.tolist()], dtype=np.float64)
    return {
        "slope": float(model.weights[0]),
        "bias": float(model.bias),
        "eval_mae": float(np.mean(np.abs(predictions - eval_targets))),
        "prediction_mean": float(np.mean(predictions)),
    }


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(11)
    full_document_supervision = _build_full_document_supervision()
    reference_fit = _fit_summary(full_document_supervision)

    naive_slopes = []
    naive_biases = []
    naive_maes = []
    ipw_slopes = []
    ipw_biases = []
    ipw_maes = []
    sampled_preview = []

    for trial in range(N_TRIALS):
        sampled_supervision, preview = _build_sampled_supervision(rng)
        naive_fit = _fit_summary(
            sampled_supervision,
            sample_weights=np.ones(len(sampled_supervision.response_judgments), dtype=np.float64),
        )
        ipw_fit = _fit_summary(sampled_supervision)
        if trial == 0:
            sampled_preview = preview
        naive_slopes.append(naive_fit["slope"])
        naive_biases.append(naive_fit["bias"])
        naive_maes.append(naive_fit["eval_mae"])
        ipw_slopes.append(ipw_fit["slope"])
        ipw_biases.append(ipw_fit["bias"])
        ipw_maes.append(ipw_fit["eval_mae"])

    return {
        "example": "ipw_regression_simulation",
        "target_rule": "y = 1.0",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "local_offsets": [float(value) for value in LOCAL_OFFSETS.tolist()],
        "sampling_probabilities": [float(value) for value in SAMPLING_PROBABILITIES.tolist()],
        "sampling_rule": "one biased local label per document per trial",
        "n_trials": N_TRIALS,
        "train_documents": int(TRAIN_X.shape[0]),
        "sample_trial_preview": sampled_preview,
        "reference_full_document_fit": {
            "slope": round(reference_fit["slope"], 4),
            "bias": round(reference_fit["bias"], 4),
            "eval_mae": round(reference_fit["eval_mae"], 4),
        },
        "naive_summary": {
            "mean_slope": round(float(np.mean(naive_slopes)), 4),
            "mean_bias": round(float(np.mean(naive_biases)), 4),
            "mean_eval_mae": round(float(np.mean(naive_maes)), 4),
            "std_eval_mae": round(float(np.std(naive_maes)), 4),
        },
        "ipw_summary": {
            "mean_slope": round(float(np.mean(ipw_slopes)), 4),
            "mean_bias": round(float(np.mean(ipw_biases)), 4),
            "mean_eval_mae": round(float(np.mean(ipw_maes)), 4),
            "std_eval_mae": round(float(np.std(ipw_maes)), 4),
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
