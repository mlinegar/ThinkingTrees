#!/usr/bin/env python3
"""CPU-only simulation: support failure makes the true target unrecoverable."""

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
LOCAL_TARGETS = TRUE_DOCUMENT_TARGET + LOCAL_OFFSETS
SAMPLING_PROBABILITIES = np.asarray([0.0, 0.2, 0.3, 0.5], dtype=np.float64)
N_TRIALS = 4000
DRAWS_PER_TRIAL = 4


def _summary(values: np.ndarray, target: float) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 4),
        "bias": round(float(np.mean(values) - target), 4),
        "std": round(float(np.std(values)), 4),
        "rmse": round(float(np.sqrt(np.mean((values - target) ** 2))), 4),
    }


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(113)
    selected = rng.choice(
        len(LOCAL_TARGETS),
        size=(N_TRIALS, DRAWS_PER_TRIAL),
        replace=True,
        p=SAMPLING_PROBABILITIES,
    )
    observed_targets = LOCAL_TARGETS[selected]
    observed_propensities = SAMPLING_PROBABILITIES[selected]
    inverse_propensities = 1.0 / observed_propensities

    naive = np.mean(observed_targets, axis=1)
    horvitz_thompson = np.mean(
        observed_targets / (len(LOCAL_TARGETS) * observed_propensities),
        axis=1,
    )
    self_normalized_ipw = np.sum(observed_targets * inverse_propensities, axis=1) / np.sum(
        inverse_propensities,
        axis=1,
    )

    observable_targets = LOCAL_TARGETS[SAMPLING_PROBABILITIES > 0]
    observable_support_mean = float(np.mean(observable_targets))
    inaccessible_target = float(LOCAL_TARGETS[SAMPLING_PROBABILITIES == 0][0])

    return {
        "example": "support_failure",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "local_targets": [float(value) for value in LOCAL_TARGETS.tolist()],
        "sampling_probabilities": [float(value) for value in SAMPLING_PROBABILITIES.tolist()],
        "draws_per_trial": DRAWS_PER_TRIAL,
        "n_trials": N_TRIALS,
        "observable_support_mean": round(observable_support_mean, 4),
        "zero_support_unit_target": round(inaccessible_target, 4),
        "summary_against_true_target": {
            "naive": _summary(naive, TRUE_DOCUMENT_TARGET),
            "horvitz_thompson": _summary(horvitz_thompson, TRUE_DOCUMENT_TARGET),
            "self_normalized_ipw": _summary(self_normalized_ipw, TRUE_DOCUMENT_TARGET),
        },
        "summary_against_observable_support_mean": {
            "naive": _summary(naive, observable_support_mean),
            "horvitz_thompson": _summary(horvitz_thompson, observable_support_mean),
            "self_normalized_ipw": _summary(self_normalized_ipw, observable_support_mean),
        },
        "first_trial": {
            "observed_targets": [round(float(value), 4) for value in observed_targets[0].tolist()],
            "logged_propensities": [
                round(float(value), 4) for value in observed_propensities[0].tolist()
            ],
            "naive_estimate": round(float(naive[0]), 4),
            "horvitz_thompson_estimate": round(float(horvitz_thompson[0]), 4),
            "self_normalized_ipw_estimate": round(float(self_normalized_ipw[0]), 4),
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
