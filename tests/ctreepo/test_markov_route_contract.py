"""Tests for self-contained Markov route output normalization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.markov_route_contract import (
    COMMON_METRIC_FIELDS,
    normalize_jax_summary,
    normalize_pytorch_summary,
    write_route_outputs,
)
from scripts.run_markov_jax_route import JaxRouteCell, _objective_args


def test_common_metric_schema_contains_required_columns() -> None:
    for field in [
        "root_count_mae",
        "theta_mae",
        "theta_first_regime_accuracy",
        "theta_last_regime_accuracy",
        "eps_leaf",
        "eps_merge",
        "eps_idemp",
        "contextual_mae",
        "pred_truth_corr",
        "pred_std",
    ]:
        assert field in COMMON_METRIC_FIELDS


def test_jax_route_objective_args_keep_witness_and_laws_distinct() -> None:
    base = dict(
        input_encoding="regime_one_hot",
        leaf_tokens=4,
        hidden_dim=8,
        n_iter=1,
        learning_rate=3e-4,
        seed=0,
    )
    witness = _objective_args(JaxRouteCell(objective="jax_fno_node_witness", **base))
    assert "--local-law-summary-family" in witness
    assert witness[witness.index("--local-law-summary-family") + 1] == "jax_fno"
    assert witness[witness.index("--local-law-merge-weight") + 1] == "0.0"
    assert witness[witness.index("--local-law-idempotence-weight") + 1] == "0.0"
    assert witness[witness.index("--local-law-contextual-weight") + 1] == "0.0"

    laws = _objective_args(JaxRouteCell(objective="jax_fno_local_laws", **base))
    assert laws[laws.index("--law-architecture") + 1] == "learned_merge"
    assert laws[laws.index("--c2-merge-target") + 1] == "self_consistency"
    assert laws[laws.index("--local-law-contextual-weight") + 1] == "0.0"


def test_normalize_jax_summary_maps_markov_diagnostics() -> None:
    row = normalize_jax_summary(
        {
            "input_encoding": "regime_one_hot",
            "args": {"fragment_len": 16, "n_iter": 20, "batch_size": 8, "seed": 3},
            "provenance": {
                "local_law_summary_family": "jax_fno",
                "local_law_summary_fno_n_modes": 8,
            },
            "diagnostics": {
                "test": {
                    "theta_count_raw_mae": 0.12,
                    "theta_mae": 0.01,
                    "theta_first_regime_accuracy": 1.0,
                    "theta_last_regime_accuracy": 0.9,
                    "eps_leaf": 0.02,
                    "eps_merge": 0.03,
                    "eps_idemp": 0.04,
                    "contextual_mae": 0.05,
                    "pred_truth_corr": 0.8,
                    "pred_std": 0.7,
                }
            },
        },
        cell_id="cell",
        objective="jax_fno_local_laws",
        output_root="out",
    )
    assert row["route"] == "jax"
    assert row["root_count_mae"] == 0.12
    assert row["theta_first_regime_accuracy"] == 1.0
    assert row["eps_merge"] == 0.03
    assert row["summary_family"] == "jax_fno"


def test_normalize_pytorch_summary_maps_witness_and_law_diagnostics() -> None:
    row = normalize_pytorch_summary(
        {
            "args": {
                "leaf_tokens": 16,
                "doc_tokens": 128,
                "epochs": 1,
                "batch_size": 4,
                "channels": 32,
                "g_n_modes": 8,
                "seed": 0,
            },
            "test_root_mae": 1.2,
            "learned_prediction_diagnostics": {
                "test": {"pred_truth_corr": 0.4, "pred_std": 0.3}
            },
            "markov_local_law_fno_diagnostics": {
                "splits": {
                    "test": {
                        "leaf": {
                            "theta_mae": 0.01,
                            "theta_first_regime_accuracy": 0.9,
                            "theta_last_regime_accuracy": 0.8,
                        },
                        "merge": {
                            "theta_mae": 0.02,
                            "theta_first_regime_accuracy": 0.7,
                            "theta_last_regime_accuracy": 0.6,
                        },
                        "root": {
                            "theta_mae": 0.03,
                            "theta_first_regime_accuracy": 0.5,
                            "theta_last_regime_accuracy": 0.4,
                            "count_diagnostics": {"root_mae": 0.25},
                            "eps_idemp_range": 0.04,
                        },
                    }
                }
            },
        },
        cell_id="cell",
        objective="markov_local_laws_fno",
        output_root="out",
    )
    assert row["route"] == "pytorch"
    assert row["root_count_mae"] == 0.25
    assert row["theta_mae"] == 0.03
    assert row["eps_leaf"] == 0.01
    assert row["eps_merge"] == 0.02
    assert row["pred_std"] == 0.3


def test_write_route_outputs_writes_summary_csv_and_report(tmp_path: Path) -> None:
    rows = [
        {
            "cell_id": "cell",
            "status": "completed",
            "route": "jax",
            "objective": "jax_fno_node_witness",
            "root_count_mae": 0.1,
        }
    ]
    write_route_outputs(
        tmp_path,
        rows,
        title="Route Test",
        manifest={"schema_version": "test"},
    )
    assert json.loads((tmp_path / "summary.json").read_text())["n_completed"] == 1
    with (tmp_path / "grid_summary.csv").open() as fh:
        parsed = list(csv.DictReader(fh))
    assert parsed[0]["root_count_mae"] == "0.1"
    assert "# Route Test" in (tmp_path / "grid_report.md").read_text()
