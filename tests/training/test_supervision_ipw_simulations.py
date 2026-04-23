from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_script(script_name: str) -> dict[str, object]:
    script_path = REPO_ROOT / "scripts" / script_name
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_easy_ipw_mean_simulation_reduces_bias() -> None:
    payload = _run_script("tutorial_supervision_05_ipw_mean_simulation.py")
    summary = payload["summary"]
    assert abs(float(summary["horvitz_thompson_bias"])) < abs(float(summary["naive_bias"]))
    assert float(summary["horvitz_thompson_rmse"]) < float(summary["naive_rmse"])


def test_ipw_regression_simulation_improves_intercept_and_mae() -> None:
    payload = _run_script("tutorial_supervision_06_ipw_regression_simulation.py")
    reference_bias = float(payload["true_document_target"])
    naive_bias = float(payload["naive_summary"]["mean_bias"])
    ipw_bias = float(payload["ipw_summary"]["mean_bias"])
    assert abs(ipw_bias - reference_bias) < abs(naive_bias - reference_bias)
    assert float(payload["ipw_summary"]["mean_eval_mae"]) < float(
        payload["naive_summary"]["mean_eval_mae"]
    )


def test_ipw_markov_simulation_improves_calibration_and_mae() -> None:
    payload = _run_script("tutorial_supervision_07_ipw_markov_simulation.py")
    target_mean = float(payload["true_document_target"])
    naive_mean = float(payload["naive_summary"]["mean_eval_prediction"])
    ipw_mean = float(payload["ipw_summary"]["mean_eval_prediction"])
    assert abs(ipw_mean - target_mean) < abs(naive_mean - target_mean)
    assert float(payload["ipw_summary"]["mean_eval_mae"]) < float(
        payload["naive_summary"]["mean_eval_mae"]
    )


def test_ipw_variance_tradeoff_shows_bias_variance_structure() -> None:
    payload = _run_script("tutorial_supervision_08_ipw_variance_tradeoff.py")
    extreme_one_draw = payload["regimes"]["extreme_skew"]["summaries"]["draws_1"]
    assert abs(float(extreme_one_draw["horvitz_thompson"]["bias"])) < abs(
        float(extreme_one_draw["naive"]["bias"])
    )
    assert float(extreme_one_draw["horvitz_thompson"]["std"]) > float(
        extreme_one_draw["naive"]["std"]
    )
    assert float(extreme_one_draw["self_normalized_ipw"]["mean"]) == float(
        extreme_one_draw["naive"]["mean"]
    )

    strong_four_draws = payload["regimes"]["strong_skew"]["summaries"]["draws_4"]
    assert abs(float(strong_four_draws["self_normalized_ipw"]["bias"])) < abs(
        float(strong_four_draws["naive"]["bias"])
    )
    assert float(strong_four_draws["self_normalized_ipw"]["std"]) > float(
        strong_four_draws["naive"]["std"]
    )
    assert float(strong_four_draws["clipped_self_normalized_ipw"]["std"]) < float(
        strong_four_draws["self_normalized_ipw"]["std"]
    )
