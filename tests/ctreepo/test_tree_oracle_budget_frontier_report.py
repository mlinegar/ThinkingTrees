from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import _payload_from_saved_runs


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _budget_payload() -> dict:
    return _payload_from_saved_runs(
        runs=[
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.08,
                "test_exact_match_rate": 0.8,
                "test_c2_idempotence_mae": 0.02,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "budget_total_calls": 5120,
                "budget_total_calls_per_doc": 0.5,
                "full_doc_budget_share": 0.5,
                "full_doc_calls_total": 2560,
                "local_calls_total": 2560,
                "doc_consumption_mode": "root_only",
                "local_split_mode": "balanced",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 3840.0,
                "effective_full_doc_mass_per_doc": 0.375,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural",
                "seed": 1,
                "train_doc_count": 10240,
                "test_root_mae": 0.075,
                "test_exact_match_rate": 0.81,
                "test_c2_idempotence_mae": 0.02,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.1,
                "local_law_c2_weight": 0.1,
                "local_law_c3_weight": 0.1,
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "budget_total_calls": 5120,
                "budget_total_calls_per_doc": 0.5,
                "full_doc_budget_share": 0.5,
                "full_doc_calls_total": 2560,
                "local_calls_total": 2560,
                "doc_consumption_mode": "doc_sequence",
                "local_split_mode": "balanced",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 3840.0,
                "effective_full_doc_mass_per_doc": 0.375,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.07,
                "test_exact_match_rate": 0.82,
                "test_c2_idempotence_mae": 0.015,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.3,
                "local_law_c3_weight": 0.0,
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "budget_total_calls": 5120,
                "budget_total_calls_per_doc": 0.5,
                "full_doc_budget_share": 0.0,
                "full_doc_calls_total": 0,
                "local_calls_total": 5120,
                "doc_consumption_mode": "root_only",
                "local_split_mode": "leaf_heavy",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 2560.0,
                "effective_full_doc_mass_per_doc": 0.25,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "tree_neural_c2c3",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.06,
                "test_exact_match_rate": 0.84,
                "test_c2_idempotence_mae": 0.012,
                "parameterization": "formal_local_law_weight",
                "optimization_root_weight": 0.7,
                "local_law_c1_weight": 0.0,
                "local_law_c2_weight": 0.2,
                "local_law_c3_weight": 0.1,
                "objective_weights_active": True,
                "c2_metric_kind": "score_drift",
                "budget_total_calls": 10240,
                "budget_total_calls_per_doc": 1.0,
                "full_doc_budget_share": 1.0,
                "full_doc_calls_total": 10240,
                "local_calls_total": 0,
                "doc_consumption_mode": "root_only",
                "local_split_mode": "balanced",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 10240.0,
                "effective_full_doc_mass_per_doc": 1.0,
            },
            {
                "benchmark": "recoverable_v4",
                "cell_id": "recoverable_v4",
                "baseline_family": "official_fno",
                "seed": 0,
                "train_doc_count": 10240,
                "test_root_mae": 0.05,
                "test_exact_match_rate": 0.9,
                "test_c2_idempotence_mae": 0.0,
                "budget_total_calls": 10240,
                "budget_total_calls_per_doc": 1.0,
                "full_doc_budget_share": 1.0,
                "full_doc_calls_total": 10240,
                "local_calls_total": 0,
                "doc_consumption_mode": "full_doc_only",
                "local_split_mode": "inactive_for_family",
                "local_allocation_policy": "breadth_first",
                "effective_full_doc_mass_total": 10240.0,
                "effective_full_doc_mass_per_doc": 1.0,
            },
        ]
    )


def test_full_doc_anchor_diagnostics_pdf_is_archived() -> None:
    script = Path("/home/mlinegar/ThinkingTrees/scripts/report_full_doc_anchor_diagnostics_pdf.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )
    assert result.returncode == 2
    assert "archived" in result.stderr.lower()


def test_tree_oracle_budget_frontier_report_smoke(tmp_path: Path) -> None:
    payload = _budget_payload()
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_pdf = tmp_path / "tree_oracle_budget_frontier_report.pdf"
    script = Path("/home/mlinegar/ThinkingTrees/scripts/report_tree_oracle_budget_frontier_pdf.py")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-json",
            str(summary_json),
            "--output-pdf",
            str(output_pdf),
        ],
        check=True,
        cwd="/home/mlinegar/ThinkingTrees",
    )

    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 0
