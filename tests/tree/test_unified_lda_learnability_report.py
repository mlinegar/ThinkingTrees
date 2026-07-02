from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _write_lda_run(path: Path, *, tau: float, quadratic_utility_weight: float, utility_error: float, law_score: float) -> None:
    payload = {
        "config": {
            "train_docs": 64,
            "seed": 7,
            "local_mixture_concentration": float(tau),
            "quadratic_utility_weight": float(quadratic_utility_weight),
            "law_leaf_query_rate": 0.1,
            "analysis_partition_mode": "aligned",
        },
        "objective": {
            "weighting_scheme": "linear_plus_quadratic_utility",
            "interprets_lambda_as": "quadratic_utility_weight",
            "selection_metric_name": "",
        },
        "local_law": {
            "policy_metrics": {
                "infer_identity": {
                    "mean_c1": 0.4,
                    "mean_c2_proxy": 0.2,
                    "mean_c3": 0.5,
                    "schedule_spread": 0.0,
                    "mean_aux_oracle_target_abs_error": 0.30,
                },
                "law_calibrated_ipw_stabilized": {
                    "mean_c1": 0.1,
                    "mean_c2_proxy": 0.05,
                    "mean_c3": float(law_score),
                    "schedule_spread": 0.0,
                    "mean_aux_oracle_target_abs_error": float(utility_error),
                },
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_unified_lda_learnability_report_uses_lambda_zero_as_baseline(tmp_path: Path) -> None:
    input_root = tmp_path / "lda_runs"
    output_dir = tmp_path / "report"

    _write_lda_run(input_root / "tau1_q0" / "seed_0.json", tau=1.0, quadratic_utility_weight=0.0, utility_error=0.15, law_score=0.20)
    _write_lda_run(input_root / "tau1_q1p5" / "seed_0.json", tau=1.0, quadratic_utility_weight=1.5, utility_error=0.10, law_score=0.05)
    _write_lda_run(input_root / "tau8_q0" / "seed_0.json", tau=8.0, quadratic_utility_weight=0.0, utility_error=0.12, law_score=0.18)
    _write_lda_run(input_root / "tau8_q1p5" / "seed_0.json", tau=8.0, quadratic_utility_weight=1.5, utility_error=0.09, law_score=0.04)

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_learnability.py",
            "--family",
            "lda",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
    )

    summary = json.loads((output_dir / "learnability_summary.json").read_text(encoding="utf-8"))
    md = (output_dir / "learnability.md").read_text(encoding="utf-8")

    assert summary["baseline_axis_name"] == "quadratic_utility_weight"
    assert summary["baseline_axis_label"] == "quadratic_utility_weight"
    assert summary["baseline_value_source"] == "family_default"
    assert summary["baseline_sweep_value"] == 0.0
    assert summary["best_no_local_law_point"]["quadratic_utility_weight"] == 0.0
    assert summary["matched_sparse_baseline"]["quadratic_utility_weight"] == 0.0
    assert summary["objective_lambda_interpretations"] == ["quadratic_utility_weight"]
    assert any("heldout_gain_core" in fig for fig in summary["figures"])
    assert "Baseline comparison" in md
    assert "quadratic_utility_weight=0" in md
    assert "values above `1` are valid" in md


def test_unified_lda_learnability_report_paper_safe_excludes_dgp_multiplier_roots(tmp_path: Path) -> None:
    input_root = tmp_path / "lda_runs"
    output_dir = tmp_path / "report_paper_safe"

    _write_lda_run(input_root / "tau1_q0" / "seed_0.json", tau=1.0, quadratic_utility_weight=0.0, utility_error=0.15, law_score=0.20)
    _write_lda_run(input_root / "tau8_q1p5" / "seed_0.json", tau=8.0, quadratic_utility_weight=1.5, utility_error=0.09, law_score=0.04)

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_learnability.py",
            "--family",
            "lda",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--paper-safe",
        ],
        cwd=repo_root,
        env=env,
    )

    summary = json.loads((output_dir / "learnability_summary.json").read_text(encoding="utf-8"))
    md = (output_dir / "learnability.md").read_text(encoding="utf-8")

    assert summary["status"] == "excluded"
    assert summary["paper_safe"] is True
    assert summary["rows_loaded_before_filter"] == 2
    assert summary["paper_safe_exclusion_reasons"] == {
        "disallowed_lambda_interpretation:quadratic_utility_weight": 2
    }
    assert "excluded from the paper-facing learnability bundle" in md.lower()
