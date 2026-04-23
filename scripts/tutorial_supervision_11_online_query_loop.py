#!/usr/bin/env python3
"""CPU-only walkthrough: online oracle queries and offline logged-data reuse."""

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
    build_dense_sampled_substructure_supervision_dataset,
    fit_dense_scalar_ridge_regressor,
    predict_dense_scalar_ridge_regressor,
)


TRUE_DOCUMENT_TARGET = 1.0
LOCAL_OFFSETS = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
LOCAL_TARGETS = TRUE_DOCUMENT_TARGET + LOCAL_OFFSETS
SAMPLING_PROBABILITIES = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
N_QUERIES = 40


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(307)
    logs = []
    sum_naive = 0.0
    sum_ht = 0.0
    sum_y_over_p = 0.0
    sum_one_over_p = 0.0
    naive_trace = []
    ht_trace = []
    snipw_trace = []
    rows: list[DenseSupervisionExample] = []

    for step in range(N_QUERIES):
        sampled_index = int(rng.choice(len(LOCAL_TARGETS), p=SAMPLING_PROBABILITIES))
        observed_target = float(LOCAL_TARGETS[sampled_index])
        propensity = float(SAMPLING_PROBABILITIES[sampled_index])
        unit_weight = 1.0 / propensity

        sum_naive += observed_target
        sum_ht += observed_target / (len(LOCAL_TARGETS) * propensity)
        sum_y_over_p += observed_target / propensity
        sum_one_over_p += 1.0 / propensity

        naive_estimate = sum_naive / (step + 1)
        ht_estimate = sum_ht / (step + 1)
        snipw_estimate = sum_y_over_p / max(sum_one_over_p, 1e-12)
        naive_trace.append(float(naive_estimate))
        ht_trace.append(float(ht_estimate))
        snipw_trace.append(float(snipw_estimate))

        logs.append(
            {
                "query_index": step,
                "observed_target": round(observed_target, 4),
                "local_offset": round(float(LOCAL_OFFSETS[sampled_index]), 4),
                "logged_propensity": round(propensity, 4),
                "ipw_weight": round(unit_weight, 4),
                "naive_running_mean": round(float(naive_estimate), 4),
                "horvitz_thompson_running_mean": round(float(ht_estimate), 4),
                "self_normalized_ipw_running_mean": round(float(snipw_estimate), 4),
            }
        )
        rows.append(
            DenseSupervisionExample(
                example_id=f"query_{step}",
                source_doc_id="online_constant_target",
                features=[float(step) / max(1, N_QUERIES - 1)],
                scalar_target=observed_target,
                original_text=f"online query {step}",
                response="queried_local_label",
                response_id=f"query_{step}",
                unit_kind=ObservationUnitKind.LEAF,
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=propensity,
                    label_propensity=1.0,
                    sampling_scheme="online_query_loop",
                    policy_name="fixed_biased_query_policy",
                    unit_kind=ObservationUnitKind.LEAF,
                    supports_ipw_estimation=True,
                    metadata={"local_offset": float(LOCAL_OFFSETS[sampled_index])},
                ),
                metadata={"query_index": step},
            )
        )

    supervision = build_dense_sampled_substructure_supervision_dataset(
        rows,
        application_name="tutorial_online_query_loop",
        supervision_signal_name="substructure_level_target",
        response_signal_name="queried_value",
        law_type="substructure_level_target",
        split="train",
    )
    eval_x = np.asarray([[0.5]], dtype=np.float64)
    naive_model, _ = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
        sample_weights=np.ones(len(rows), dtype=np.float64),
    )
    ipw_model, _ = fit_dense_scalar_ridge_regressor(
        supervision,
        config=DenseScalarRidgeTrainingConfig(model=DenseScalarRidgeModelConfig(ridge_alpha=1e-8)),
    )
    offline_naive_prediction = float(predict_dense_scalar_ridge_regressor(naive_model, eval_x)[0])
    offline_ipw_prediction = float(predict_dense_scalar_ridge_regressor(ipw_model, eval_x)[0])

    return {
        "example": "online_query_loop",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "sampling_probabilities": [float(value) for value in SAMPLING_PROBABILITIES.tolist()],
        "local_targets": [float(value) for value in LOCAL_TARGETS.tolist()],
        "n_queries": N_QUERIES,
        "first_queries": logs[:8],
        "final_online_estimates": {
            "naive_running_mean": round(float(naive_trace[-1]), 4),
            "horvitz_thompson_running_mean": round(float(ht_trace[-1]), 4),
            "self_normalized_ipw_running_mean": round(float(snipw_trace[-1]), 4),
        },
        "offline_logged_supervision_fit": {
            "naive_prediction_at_midpoint": round(offline_naive_prediction, 4),
            "ipw_prediction_at_midpoint": round(offline_ipw_prediction, 4),
            "n_logged_judgments": len(supervision.response_judgments),
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
