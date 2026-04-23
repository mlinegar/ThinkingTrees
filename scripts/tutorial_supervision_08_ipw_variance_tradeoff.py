#!/usr/bin/env python3
"""CPU-only simulation: bias-variance tradeoff for IPW under increasing skew."""

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
N_UNITS = int(LOCAL_TARGETS.shape[0])
N_TRIALS = 6000
CLIPPED_MAX_WEIGHT = 5.0

REGIMES = {
    "uniform": np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
    "mild_skew": np.asarray([0.15, 0.20, 0.30, 0.35], dtype=np.float64),
    "strong_skew": np.asarray([0.05, 0.10, 0.25, 0.60], dtype=np.float64),
    "extreme_skew": np.asarray([0.01, 0.04, 0.15, 0.80], dtype=np.float64),
}

DRAW_COUNTS = (1, 4)


def _summarize(estimates: np.ndarray) -> dict[str, float]:
    bias = float(np.mean(estimates) - TRUE_DOCUMENT_TARGET)
    std = float(np.std(estimates))
    rmse = float(np.sqrt(np.mean((estimates - TRUE_DOCUMENT_TARGET) ** 2)))
    return {
        "mean": round(float(np.mean(estimates)), 4),
        "bias": round(bias, 4),
        "std": round(std, 4),
        "rmse": round(rmse, 4),
    }


def _simulate_regime(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    *,
    draws_per_trial: int,
) -> dict[str, object]:
    selected = rng.choice(
        N_UNITS,
        size=(N_TRIALS, draws_per_trial),
        replace=True,
        p=probabilities,
    )
    observed_targets = LOCAL_TARGETS[selected]
    observed_propensities = probabilities[selected]
    inverse_propensities = 1.0 / observed_propensities

    naive = np.mean(observed_targets, axis=1)
    horvitz_thompson = np.mean(
        observed_targets / (N_UNITS * observed_propensities),
        axis=1,
    )
    self_normalized = np.sum(observed_targets * inverse_propensities, axis=1) / np.sum(
        inverse_propensities,
        axis=1,
    )
    clipped_weights = np.minimum(inverse_propensities, CLIPPED_MAX_WEIGHT)
    clipped_self_normalized = np.sum(observed_targets * clipped_weights, axis=1) / np.sum(
        clipped_weights,
        axis=1,
    )

    first_trial = {
        "observed_targets": [round(float(value), 4) for value in observed_targets[0].tolist()],
        "logged_propensities": [round(float(value), 4) for value in observed_propensities[0].tolist()],
        "inverse_propensities": [round(float(value), 4) for value in inverse_propensities[0].tolist()],
        "naive_estimate": round(float(naive[0]), 4),
        "horvitz_thompson_estimate": round(float(horvitz_thompson[0]), 4),
        "self_normalized_ipw_estimate": round(float(self_normalized[0]), 4),
        "clipped_self_normalized_ipw_estimate": round(float(clipped_self_normalized[0]), 4),
    }
    return {
        "draws_per_trial": draws_per_trial,
        "first_trial": first_trial,
        "naive": _summarize(naive),
        "horvitz_thompson": _summarize(horvitz_thompson),
        "self_normalized_ipw": _summarize(self_normalized),
        "clipped_self_normalized_ipw": _summarize(clipped_self_normalized),
    }


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(101)
    results = {}
    for regime_name, probabilities in REGIMES.items():
        regime_results = {}
        for draws_per_trial in DRAW_COUNTS:
            regime_results[f"draws_{draws_per_trial}"] = _simulate_regime(
                rng,
                probabilities,
                draws_per_trial=draws_per_trial,
            )
        results[regime_name] = {
            "sampling_probabilities": [round(float(value), 4) for value in probabilities.tolist()],
            "summaries": regime_results,
        }

    return {
        "example": "ipw_variance_tradeoff",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "local_offsets": [float(value) for value in LOCAL_OFFSETS.tolist()],
        "local_targets": [float(value) for value in LOCAL_TARGETS.tolist()],
        "n_trials": N_TRIALS,
        "draw_counts": list(DRAW_COUNTS),
        "clipped_max_weight": CLIPPED_MAX_WEIGHT,
        "estimators": {
            "naive": "plain sample mean of observed targets",
            "horvitz_thompson": "mean(y / (N * propensity))",
            "self_normalized_ipw": "sum(y / propensity) / sum(1 / propensity)",
            "clipped_self_normalized_ipw": (
                "sum(y * min(1/propensity, c)) / sum(min(1/propensity, c))"
            ),
        },
        "regimes": results,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
