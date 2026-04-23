#!/usr/bin/env python3
"""CPU-only simulation: naive vs IPW for one biased local-label sample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRUE_DOCUMENT_TARGET = 1.0
LOCAL_OFFSETS = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
SAMPLING_PROBABILITIES = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
N_TRIALS = 2000


def _rmse(values: np.ndarray, target: float) -> float:
    return float(np.sqrt(np.mean((values - float(target)) ** 2)))


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(7)
    local_targets = TRUE_DOCUMENT_TARGET + LOCAL_OFFSETS
    n_units = int(local_targets.shape[0])

    selected_indices = rng.choice(
        n_units,
        size=N_TRIALS,
        replace=True,
        p=SAMPLING_PROBABILITIES,
    )
    observed_targets = local_targets[selected_indices]
    observed_propensities = SAMPLING_PROBABILITIES[selected_indices]

    naive_estimates = np.asarray(observed_targets, dtype=np.float64)
    horvitz_thompson_estimates = np.asarray(
        observed_targets / (n_units * observed_propensities),
        dtype=np.float64,
    )

    return {
        "example": "ipw_mean_simulation_easy",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "local_offsets": [float(value) for value in LOCAL_OFFSETS.tolist()],
        "local_targets": [float(value) for value in local_targets.tolist()],
        "sampling_probabilities": [float(value) for value in SAMPLING_PROBABILITIES.tolist()],
        "sampling_rule": "exactly one local unit is sampled per trial",
        "ipw_formula": "horvitz_thompson_mean = observed_target / (num_units * propensity)",
        "n_trials": N_TRIALS,
        "summary": {
            "naive_mean": round(float(np.mean(naive_estimates)), 4),
            "naive_bias": round(float(np.mean(naive_estimates) - TRUE_DOCUMENT_TARGET), 4),
            "naive_rmse": round(_rmse(naive_estimates, TRUE_DOCUMENT_TARGET), 4),
            "horvitz_thompson_mean": round(float(np.mean(horvitz_thompson_estimates)), 4),
            "horvitz_thompson_bias": round(
                float(np.mean(horvitz_thompson_estimates) - TRUE_DOCUMENT_TARGET),
                4,
            ),
            "horvitz_thompson_rmse": round(
                _rmse(horvitz_thompson_estimates, TRUE_DOCUMENT_TARGET),
                4,
            ),
        },
        "first_trials": [
            {
                "trial": int(index),
                "observed_target": round(float(observed_targets[index]), 4),
                "logged_propensity": round(float(observed_propensities[index]), 4),
                "naive_estimate": round(float(naive_estimates[index]), 4),
                "horvitz_thompson_estimate": round(
                    float(horvitz_thompson_estimates[index]),
                    4,
                ),
            }
            for index in range(8)
        ],
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
