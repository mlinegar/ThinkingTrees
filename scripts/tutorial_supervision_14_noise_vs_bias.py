#!/usr/bin/env python3
"""CPU-only simulation: separate label noise from systematic bias."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRUE_TARGET = 1.0
N_OBS_PER_TRIAL = 20
N_TRIALS = 3000

REGIMES = {
    "unbiased_noiseless": {"bias_shift": 0.0, "noise_std": 0.0},
    "unbiased_noisy": {"bias_shift": 0.0, "noise_std": 0.5},
    "biased_noiseless": {"bias_shift": 0.4, "noise_std": 0.0},
    "biased_noisy": {"bias_shift": 0.4, "noise_std": 0.5},
}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 4),
        "bias": round(float(np.mean(values) - TRUE_TARGET), 4),
        "std": round(float(np.std(values)), 4),
        "rmse": round(float(np.sqrt(np.mean((values - TRUE_TARGET) ** 2))), 4),
    }


def run_example() -> dict[str, object]:
    rng = np.random.default_rng(419)
    results = {}
    previews = {}
    for regime_name, cfg in REGIMES.items():
        labels = (
            TRUE_TARGET
            + float(cfg["bias_shift"])
            + rng.normal(0.0, float(cfg["noise_std"]), size=(N_TRIALS, N_OBS_PER_TRIAL))
        )
        trial_means = np.mean(labels, axis=1)
        results[regime_name] = _summary(trial_means)
        previews[regime_name] = [round(float(value), 4) for value in labels[0, :6].tolist()]

    return {
        "example": "noise_vs_bias",
        "true_target": TRUE_TARGET,
        "n_obs_per_trial": N_OBS_PER_TRIAL,
        "n_trials": N_TRIALS,
        "regimes": results,
        "first_trial_samples": previews,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
