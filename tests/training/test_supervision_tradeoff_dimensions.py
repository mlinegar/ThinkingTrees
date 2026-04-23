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


def test_support_failure_stays_unrecoverable() -> None:
    payload = _run_script("tutorial_supervision_09_support_failure.py")
    true_target = float(payload["true_document_target"])
    naive_mean = float(payload["summary_against_true_target"]["naive"]["mean"])
    ht_mean = float(payload["summary_against_true_target"]["horvitz_thompson"]["mean"])
    snipw_mean = float(payload["summary_against_true_target"]["self_normalized_ipw"]["mean"])
    assert abs(ht_mean - true_target) < abs(naive_mean - true_target)
    assert abs(snipw_mean - true_target) > 0.1
    assert abs(ht_mean - true_target) > 0.05


def test_ess_clipping_increases_effective_sample_size() -> None:
    payload = _run_script("tutorial_supervision_10_effective_sample_size_clipping.py")
    ess = payload["ess_summary"]
    estimators = payload["estimators"]
    assert float(ess["clipped_mean_ess"]) > float(ess["raw_mean_ess"])
    assert abs(float(estimators["horvitz_thompson"]["bias"])) < abs(
        float(estimators["naive"]["bias"])
    )


def test_online_query_loop_and_offline_reuse_improve_bias() -> None:
    payload = _run_script("tutorial_supervision_11_online_query_loop.py")
    true_target = float(payload["true_document_target"])
    online = payload["final_online_estimates"]
    offline = payload["offline_logged_supervision_fit"]
    assert abs(float(online["self_normalized_ipw_running_mean"]) - true_target) < abs(
        float(online["naive_running_mean"]) - true_target
    )
    assert abs(float(offline["ipw_prediction_at_midpoint"]) - true_target) < abs(
        float(offline["naive_prediction_at_midpoint"]) - true_target
    )


def test_weighted_sgd_matches_duplicated_rows() -> None:
    payload = _run_script("tutorial_supervision_12_weighted_sgd_equivalence.py")
    assert float(payload["max_abs_prediction_diff_weighted_vs_duplicated"]) < 0.02


def test_scalar_comparative_binary_bridge_has_expected_views() -> None:
    payload = _run_script("tutorial_supervision_13_scalar_comparative_binary_bridge.py")
    assert int(payload["supervision_summary"]["total_response_judgments"]) == 3
    assert int(payload["comparative_view"]["n_records"]) == 1
    assert int(payload["binary_adjacent_view"]["n_pairs"]) == 2
    assert int(payload["binary_winner_vs_runner_up_view"]["n_pairs"]) == 1


def test_noise_vs_bias_separates_regimes() -> None:
    payload = _run_script("tutorial_supervision_14_noise_vs_bias.py")
    regimes = payload["regimes"]
    assert abs(float(regimes["unbiased_noiseless"]["bias"])) < 0.02
    assert float(regimes["unbiased_noisy"]["std"]) > float(regimes["unbiased_noiseless"]["std"])
    assert abs(float(regimes["biased_noiseless"]["bias"])) > 0.2


def test_markov_support_diagnostic_shows_ess_drop() -> None:
    payload = _run_script("tutorial_supervision_15_markov_support_diagnostic.py")
    concentrations = payload["concentrations"]
    low = concentrations["0.0"]
    high = concentrations["16.0"]
    true_target = float(payload["true_document_target"])
    assert float(high["mean_ess"]) < float(low["mean_ess"])
    assert abs(float(high["ipw_mean_eval_prediction"]) - true_target) < abs(
        float(high["naive_mean_eval_prediction"]) - true_target
    )


def test_joint_tradeoff_matrix_shows_interactions() -> None:
    payload = _run_script("tutorial_supervision_16_joint_tradeoff_matrix.py")
    regimes = payload["regimes"]
    full_support = regimes["full_support_high_skew_high_noise"]
    support_failure = regimes["support_failure_high_skew_high_noise"]
    assert abs(float(full_support["horvitz_thompson"]["bias"])) < abs(
        float(full_support["naive"]["bias"])
    )
    assert abs(float(support_failure["horvitz_thompson"]["bias"])) > 0.05
