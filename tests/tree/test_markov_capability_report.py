from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_fake_run(
    path: Path,
    *,
    llw: float,
    scw: float,
    data_seed: int,
    model_seed: int,
    val_leaf: float | None,
    val_merge: float | None,
    val_spread: float | None,
    val_root: float | None,
    val_objective: float | None,
    test_leaf: float,
    test_merge: float,
    test_spread: float,
    test_root: float,
    test_objective: float | None = None,
) -> None:
    learned = {
        "leaf_mae": float(test_leaf),
        "merge_mae": float(test_merge),
        "schedule_spread_mean": float(test_spread),
        "root_mae": float(test_root),
        "test_leaf_mae": float(test_leaf),
        "test_merge_mae": float(test_merge),
        "test_schedule_spread_mean": float(test_spread),
        "test_root_mae": float(test_root),
    }
    if val_leaf is not None:
        learned["val_leaf_mae"] = float(val_leaf)
    if val_merge is not None:
        learned["val_merge_mae"] = float(val_merge)
    if val_spread is not None:
        learned["val_schedule_spread_mean"] = float(val_spread)
    if val_root is not None:
        learned["val_root_mae"] = float(val_root)
    if val_objective is not None:
        learned["val_objective_full_labels"] = float(val_objective)
    if test_objective is not None:
        learned["test_objective_full_labels"] = float(test_objective)

    payload = {
        "config": {
            "n_regimes": 4,
            "fixed_leaf_tokens": 16,
            "train_docs": 128,
            "val_docs": 32,
            "test_docs": 64,
            "audit_fraction": 0.1,
            "root_weight": 1.0,
            "state_dim": 64,
            "hidden_dim": 256,
            "n_epochs": 4,
            "feature_mode": "full",
            "max_segments": 2,
            "schedule_consistency_weight": float(scw),
            "effective_data_seed": int(data_seed),
            "effective_model_seed": int(model_seed),
        },
        "objective": {
            "local_law_weight": float(llw),
        },
        "metrics": {
            "learned": learned,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_capability_report_selects_on_validation_not_test(tmp_path: Path) -> None:
    input_root = tmp_path / "runs"
    output_dir = tmp_path / "report"

    for model_seed in (0, 1):
        _write_fake_run(
            input_root / f"llw0_seed{model_seed}" / f"seed_{model_seed}.json",
            llw=0.0,
            scw=0.1,
            data_seed=0,
            model_seed=model_seed,
            val_leaf=0.50,
            val_merge=0.50,
            val_spread=0.20,
            val_root=0.50,
            val_objective=1.10,
            test_leaf=0.50,
            test_merge=0.50,
            test_spread=0.20,
            test_root=0.50,
            test_objective=1.05,
        )
        _write_fake_run(
            input_root / f"llw05_seed{model_seed}" / f"seed_{model_seed}.json",
            llw=0.5,
            scw=0.1,
            data_seed=0,
            model_seed=model_seed,
            val_leaf=0.30,
            val_merge=0.35,
            val_spread=0.20,
            val_root=0.50,
            val_objective=0.75,
            test_leaf=0.40,
            test_merge=0.45,
            test_spread=0.16,
            test_root=0.50,
            test_objective=0.72,
        )
        _write_fake_run(
            input_root / f"llw10_seed{model_seed}" / f"seed_{model_seed}.json",
            llw=1.0,
            scw=0.1,
            data_seed=0,
            model_seed=model_seed,
            val_leaf=0.38,
            val_merge=0.38,
            val_spread=0.20,
            val_root=0.52,
            val_objective=0.35,
            test_leaf=0.20,
            test_merge=0.25,
            test_spread=0.08,
            test_root=0.52,
            test_objective=0.33,
        )

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_capability_map.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--suite-type",
            "transition_map_suite",
        ],
        cwd=repo_root,
        env=env,
    )

    summary = json.loads((output_dir / "markov_capability_summary.json").read_text(encoding="utf-8"))
    assert int(summary["selected_run_count"]) == 2
    assert int(summary["aggregated_row_count"]) == 1
    selected_rows = list(summary["selected_rows"])
    assert all(float(row["selected_lambda_local"]) == 1.0 for row in selected_rows)
    assert all(row["selection_metric"] == "val_objective_full_labels" for row in selected_rows)
    aggregated = list(summary["aggregated_rows"])[0]
    assert float(aggregated["selected_lambda_local"]) == 1.0
    assert aggregated["selection_metric"] == "val_objective_full_labels"
    assert float(aggregated["val_selection_metric_value"]) == 0.35
    assert float(aggregated["test_objective_for_report"]) == 0.33
    assert float(aggregated["baseline_test_theorem_score_n"]) == 1.05
    assert aggregated["dominant_capability_status"] == "full_success"
    assert "failure_reason" in aggregated

    csv_text = (output_dir / "markov_capability_selected_rows.csv").read_text(encoding="utf-8")
    assert "selected_lambda_local" in csv_text
    assert "baseline_test_theorem_score_n" in csv_text
    assert "capability_status" in csv_text


def test_capability_report_supports_exploratory_compat_mode(tmp_path: Path) -> None:
    input_root = tmp_path / "runs"
    output_dir = tmp_path / "report"
    _write_fake_run(
        input_root / "seed_0.json",
        llw=0.0,
        scw=0.0,
        data_seed=0,
        model_seed=0,
        val_leaf=None,
        val_merge=None,
        val_spread=None,
        val_root=None,
        val_objective=None,
        test_leaf=0.50,
        test_merge=0.50,
        test_spread=0.20,
        test_root=0.50,
    )

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_capability_map.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--suite-type",
            "transition_map_suite",
            "--compat-exploratory",
        ],
        cwd=repo_root,
        env=env,
    )

    summary = json.loads((output_dir / "markov_capability_summary.json").read_text(encoding="utf-8"))
    assert summary["exploratory_only"] is True
