#!/usr/bin/env python3
"""CPU-only simulation: ESS and clipping under severe propensity skew."""

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
SAMPLING_PROBABILITIES = np.asarray([0.01, 0.04, 0.15, 0.80], dtype=np.float64)
DRAWS_PER_TRIAL = 4
N_TRIALS = 5000
CLIP_MAX_WEIGHT = 5.0


def _summary(values: np.ndarray, target: float) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 4),
        "bias": round(float(np.mean(values) - target), 4),
        "std": round(float(np.std(values)), 4),
        "rmse": round(float(np.sqrt(np.mean((values - target) ** 2))), 4),
    }


def _effective_sample_size(weights: np.ndarray) -> np.ndarray:
    numerator = np.sum(weights, axis=1) ** 2
    denominator = np.sum(weights ** 2, axis=1)
    return np.asarray(numerator / np.maximum(denominator, 1e-12), dtype=np.float64)


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(211)
    selected = rng.choice(
        len(LOCAL_TARGETS),
        size=(N_TRIALS, DRAWS_PER_TRIAL),
        replace=True,
        p=SAMPLING_PROBABILITIES,
    )
    observed_targets = LOCAL_TARGETS[selected]
    observed_propensities = SAMPLING_PROBABILITIES[selected]
    inverse_propensities = 1.0 / observed_propensities
    clipped_inverse_propensities = np.minimum(inverse_propensities, CLIP_MAX_WEIGHT)

    naive = np.mean(observed_targets, axis=1)
    horvitz_thompson = np.mean(
        observed_targets / (len(LOCAL_TARGETS) * observed_propensities),
        axis=1,
    )
    self_normalized_ipw = np.sum(observed_targets * inverse_propensities, axis=1) / np.sum(
        inverse_propensities,
        axis=1,
    )
    clipped_self_normalized_ipw = np.sum(
        observed_targets * clipped_inverse_propensities,
        axis=1,
    ) / np.sum(clipped_inverse_propensities, axis=1)

    raw_ess = _effective_sample_size(inverse_propensities)
    clipped_ess = _effective_sample_size(clipped_inverse_propensities)

    return {
        "example": "effective_sample_size_clipping",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "local_targets": [float(value) for value in LOCAL_TARGETS.tolist()],
        "sampling_probabilities": [float(value) for value in SAMPLING_PROBABILITIES.tolist()],
        "draws_per_trial": DRAWS_PER_TRIAL,
        "n_trials": N_TRIALS,
        "clip_max_weight": CLIP_MAX_WEIGHT,
        "ess_summary": {
            "raw_mean_ess": round(float(np.mean(raw_ess)), 4),
            "raw_min_ess": round(float(np.min(raw_ess)), 4),
            "clipped_mean_ess": round(float(np.mean(clipped_ess)), 4),
            "clipped_min_ess": round(float(np.min(clipped_ess)), 4),
        },
        "estimators": {
            "naive": _summary(naive, TRUE_DOCUMENT_TARGET),
            "horvitz_thompson": _summary(horvitz_thompson, TRUE_DOCUMENT_TARGET),
            "self_normalized_ipw": _summary(self_normalized_ipw, TRUE_DOCUMENT_TARGET),
            "clipped_self_normalized_ipw": _summary(
                clipped_self_normalized_ipw,
                TRUE_DOCUMENT_TARGET,
            ),
        },
        "first_trial": {
            "observed_targets": [round(float(value), 4) for value in observed_targets[0].tolist()],
            "logged_propensities": [
                round(float(value), 4) for value in observed_propensities[0].tolist()
            ],
            "inverse_propensities": [
                round(float(value), 4) for value in inverse_propensities[0].tolist()
            ],
            "clipped_inverse_propensities": [
                round(float(value), 4)
                for value in clipped_inverse_propensities[0].tolist()
            ],
            "raw_ess": round(float(raw_ess[0]), 4),
            "clipped_ess": round(float(clipped_ess[0]), 4),
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
