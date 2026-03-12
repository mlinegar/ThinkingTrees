from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.perf_harness import (
    RegressionRule,
    evaluate_expectation,
    evaluate_regressions,
    extract_metrics,
    has_regression_error,
    load_manifest,
    select_scenarios,
)


def test_manifest_load_and_profile_select(tmp_path: Path) -> None:
    manifest_path = tmp_path / "perf.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "profiles:",
                "  ci:",
                "    include_layers: [micro, meso]",
                "scenarios:",
                "  - id: s1",
                "    layer: micro",
                "    command: echo micro",
                "  - id: s2",
                "    layer: macro",
                "    command: echo macro",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    selected = select_scenarios(manifest, "ci")
    assert [s.scenario_id for s in selected] == ["s1"]


def test_extract_metrics_handles_nested_list_paths(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"aggregate": {"docs_per_second": 1.25}},
                    {"aggregate": {"docs_per_second": 2.5}},
                ],
                "summary": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    observed = extract_metrics(
        metrics_path,
        {
            "cold": "runs.0.aggregate.docs_per_second",
            "warm": "runs.1.aggregate.docs_per_second",
            "missing": "runs.9.aggregate.docs_per_second",
        },
    )
    assert observed["cold"] == 1.25
    assert observed["warm"] == 2.5
    assert observed["missing"] is None


def test_regression_eval_and_error_detection() -> None:
    metrics = {"primary_mean": 72.0, "docs_per_second": 0.0001}
    rules = [
        RegressionRule(metric="primary_mean", op=">=", threshold=70.0, severity="error"),
        RegressionRule(metric="docs_per_second", op=">=", threshold=0.001, severity="warn"),
    ]

    outcomes = evaluate_regressions(metrics=metrics, rules=rules)
    assert outcomes[0]["passed"] is True
    assert outcomes[1]["passed"] is False
    assert has_regression_error(outcomes) is False


def test_manifest_matrix_expansion_and_formatting(tmp_path: Path) -> None:
    manifest_path = tmp_path / "perf_matrix.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "profiles:",
                "  quick:",
                "    include_layers: [meso]",
                "scenarios:",
                "  - id: temporal_{semantic_tag}_{weight_tag}",
                "    layer: meso",
                "    matrix:",
                "      semantic:",
                "        - semantic_tag: sem_on",
                "          semantic_flag: --semantic-memory-features",
                "        - semantic_tag: sem_off",
                "          semantic_flag: --no-semantic-memory-features",
                "      weight:",
                "        - weight_tag: learned",
                "          weight_flag: --learn-loss-weights",
                "        - weight_tag: fixed",
                "          weight_flag: --no-learn-loss-weights",
                "    command: echo {scenario_id} {semantic_flag} {weight_flag}",
                "    metrics_file: outputs/{scenario_id}/metrics.json",
            ]
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    selected = select_scenarios(manifest, "quick")
    ids = [s.scenario_id for s in selected]
    assert ids == [
        "temporal_sem_on_learned",
        "temporal_sem_on_fixed",
        "temporal_sem_off_learned",
        "temporal_sem_off_fixed",
    ]
    assert selected[0].command.endswith("--semantic-memory-features --learn-loss-weights")
    assert selected[-1].command.endswith("--no-semantic-memory-features --no-learn-loss-weights")
    assert selected[0].metrics_file == "outputs/temporal_sem_on_learned/metrics.json"


def test_manifest_expected_failure_modes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "perf_expected.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "profiles:",
                "  nightly:",
                "    include_layers: [meso]",
                "scenarios:",
                "  - id: neg_ctrl",
                "    layer: meso",
                "    command: exit 1",
                "    expected:",
                "      outcome: fail",
                "      failure_modes: [command]",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    selected = select_scenarios(manifest, "nightly")
    assert len(selected) == 1
    assert selected[0].expected_outcome == "fail"
    assert selected[0].expected_failure_modes == ["command"]


def test_expectation_eval_for_expected_failure() -> None:
    reg_fail = evaluate_expectation(
        expected_outcome="fail",
        expected_failure_modes=["regression"],
        command_ok=True,
        regression_ok=False,
    )
    assert reg_fail["actual_outcome"] == "fail"
    assert reg_fail["failure_modes"] == ["regression"]
    assert reg_fail["expectation_met"] is True

    unexpected_pass = evaluate_expectation(
        expected_outcome="fail",
        expected_failure_modes=["command"],
        command_ok=True,
        regression_ok=True,
    )
    assert unexpected_pass["actual_outcome"] == "pass"
    assert unexpected_pass["expectation_met"] is False


def test_manifest_expected_bool_shorthand(tmp_path: Path) -> None:
    manifest_path = tmp_path / "perf_expected_bool.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "scenarios:",
                "  - id: should_fail",
                "    layer: micro",
                "    command: exit 1",
                "    expected: true",
            ]
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    assert len(manifest.scenarios) == 1
    scenario = manifest.scenarios[0]
    assert scenario.expected_outcome == "fail"
    assert scenario.expected_failure_modes == ["any"]
