from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script_name", "expected_example"),
    [
        ("tutorial_supervision_00_manual_gradient_descent.py", "manual_gradient_descent_bridge"),
        ("tutorial_supervision_00_same_average_local_variation.py", "same_average_local_variation"),
        ("tutorial_supervision_00_sampled_local_ipw.py", "sampled_local_ipw"),
        ("tutorial_supervision_00_numeric_gradient_descent.py", "numeric_gradient_descent_methods"),
        ("tutorial_supervision_01_dense_scalar_regression.py", "dense_scalar_regression"),
        ("tutorial_supervision_02_grouped_comparative.py", "grouped_comparative_from_scalar_scores"),
        ("tutorial_supervision_03_human_preference.py", "human_preference_store_export"),
        ("tutorial_supervision_04_markov_style.py", "markov_style_document_regression"),
        ("tutorial_supervision_05_ipw_mean_simulation.py", "ipw_mean_simulation_easy"),
        ("tutorial_supervision_06_ipw_regression_simulation.py", "ipw_regression_simulation"),
        ("tutorial_supervision_07_ipw_markov_simulation.py", "ipw_markov_simulation"),
        ("tutorial_supervision_08_ipw_variance_tradeoff.py", "ipw_variance_tradeoff"),
        ("tutorial_supervision_09_support_failure.py", "support_failure"),
        (
            "tutorial_supervision_10_effective_sample_size_clipping.py",
            "effective_sample_size_clipping",
        ),
        ("tutorial_supervision_11_online_query_loop.py", "online_query_loop"),
        ("tutorial_supervision_12_weighted_sgd_equivalence.py", "weighted_sgd_equivalence"),
        (
            "tutorial_supervision_13_scalar_comparative_binary_bridge.py",
            "scalar_comparative_binary_bridge",
        ),
        ("tutorial_supervision_14_noise_vs_bias.py", "noise_vs_bias"),
        (
            "tutorial_supervision_15_markov_support_diagnostic.py",
            "markov_support_diagnostic",
        ),
        ("tutorial_supervision_16_joint_tradeoff_matrix.py", "joint_tradeoff_matrix"),
    ],
)
def test_supervision_tutorial_script_runs(script_name: str, expected_example: str) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["example"] == expected_example
