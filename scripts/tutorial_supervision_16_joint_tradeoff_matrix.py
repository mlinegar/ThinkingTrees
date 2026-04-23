#!/usr/bin/env python3
"""CPU-only simulation: support, skew, and noise jointly."""

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
N_TRIALS = 4000
DRAWS_PER_TRIAL = 4

REGIMES = {
    "clean_uniform": {
        "probabilities": np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
        "noise_std": 0.0,
    },
    "full_support_high_skew": {
        "probabilities": np.asarray([0.05, 0.10, 0.25, 0.60], dtype=np.float64),
        "noise_std": 0.0,
    },
    "full_support_high_skew_high_noise": {
        "probabilities": np.asarray([0.05, 0.10, 0.25, 0.60], dtype=np.float64),
        "noise_std": 0.5,
    },
    "support_failure_high_skew_high_noise": {
        "probabilities": np.asarray([0.00, 0.20, 0.30, 0.50], dtype=np.float64),
        "noise_std": 0.5,
    },
}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 4),
        "bias": round(float(np.mean(values) - TRUE_DOCUMENT_TARGET), 4),
        "std": round(float(np.std(values)), 4),
        "rmse": round(float(np.sqrt(np.mean((values - TRUE_DOCUMENT_TARGET) ** 2))), 4),
    }


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(809)
    results = {}
    for regime_name, cfg in REGIMES.items():
        probabilities = np.asarray(cfg["probabilities"], dtype=np.float64)
        selected = rng.choice(
            len(LOCAL_TARGETS),
            size=(N_TRIALS, DRAWS_PER_TRIAL),
            replace=True,
            p=probabilities,
        )
        base_targets = LOCAL_TARGETS[selected]
        noisy_targets = base_targets + rng.normal(
            0.0,
            float(cfg["noise_std"]),
            size=base_targets.shape,
        )
        observed_propensities = probabilities[selected]
        inverse_propensities = np.where(
            observed_propensities > 0,
            1.0 / observed_propensities,
            0.0,
        )

        naive = np.mean(noisy_targets, axis=1)
        horvitz_thompson = np.mean(
            noisy_targets / np.maximum(len(LOCAL_TARGETS) * observed_propensities, 1e-12),
            axis=1,
        )
        self_normalized_ipw = np.sum(noisy_targets * inverse_propensities, axis=1) / np.maximum(
            np.sum(inverse_propensities, axis=1),
            1e-12,
        )
        results[regime_name] = {
            "sampling_probabilities": [round(float(value), 4) for value in probabilities.tolist()],
            "noise_std": float(cfg["noise_std"]),
            "naive": _summary(naive),
            "horvitz_thompson": _summary(horvitz_thompson),
            "self_normalized_ipw": _summary(self_normalized_ipw),
        }

    return {
        "example": "joint_tradeoff_matrix",
        "true_document_target": TRUE_DOCUMENT_TARGET,
        "local_targets": [float(value) for value in LOCAL_TARGETS.tolist()],
        "draws_per_trial": DRAWS_PER_TRIAL,
        "n_trials": N_TRIALS,
        "regimes": results,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
