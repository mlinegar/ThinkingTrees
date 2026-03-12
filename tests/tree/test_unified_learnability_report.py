from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from src.tree.markov_changepoint_ops_count_simulation import (
    OPSCountConfig,
    run_markov_changepoint_ops_count_experiment,
)


def _write_markov_runs(input_root: Path) -> None:
    input_root.mkdir(parents=True, exist_ok=True)

    common = dict(
        n_regimes=3,
        vocab_size=32,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=6,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        law_package="all_laws_plus_sched",
        c2_weight=0.2,
        schedule_consistency_weight=0.1,
        violation_tau=0.0,
        data_seed=3,
        use_cuda=False,
        torch_threads=1,
    )

    for llw in (0.0, 1.0):
        summary = run_markov_changepoint_ops_count_experiment(
            OPSCountConfig(**common, seed=int(llw * 10), model_seed=int(llw * 10), local_law_weight=llw)
        )
        out_dir = input_root / f"llw_{str(llw).replace('.', 'p')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"seed_{int(llw * 10)}.json").write_text(summary.to_json(), encoding="utf-8")


def test_unified_markov_learnability_report_exposes_no_local_law_baseline(tmp_path: Path) -> None:
    input_root = tmp_path / "markov_runs"
    output_dir = tmp_path / "report"
    _write_markov_runs(input_root)

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_learnability.py",
            "--family",
            "markov",
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

    assert summary["baseline_value_source"] == "family_default"
    assert summary["baseline_sweep_value"] == 0.0
    assert summary["best_baseline_point"]["lambda_local"] == 0.0
    assert summary["best_no_local_law_point"]["lambda_local"] == 0.0
    assert "normalized_lambda_tradeoff" in summary["objective_weighting_schemes"]
    assert "Best baseline point" in md
    assert "Baseline comparison" in md


def test_unified_markov_learnability_report_accepts_base_override(tmp_path: Path) -> None:
    input_root = tmp_path / "markov_runs"
    output_dir = tmp_path / "report_override"
    _write_markov_runs(input_root)

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_learnability.py",
            "--family",
            "markov",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--base-field",
            "lambda_local",
            "--base-value",
            "1.0",
        ],
        cwd=repo_root,
        env=env,
    )

    summary = json.loads((output_dir / "learnability_summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_axis_name"] == "lambda_local"
    assert summary["baseline_value_source"] == "cli"
    assert summary["baseline_sweep_value"] == 1.0
    assert summary["best_no_local_law_point"]["lambda_local"] == 1.0


def test_unified_markov_learnability_report_paper_safe_excludes_partial_weight_rows(tmp_path: Path) -> None:
    input_root = tmp_path / "markov_runs"
    output_dir = tmp_path / "report_paper_safe"
    _write_markov_runs(input_root)

    broken_path = input_root / "llw_1p0" / "seed_10.json"
    payload = json.loads(broken_path.read_text(encoding="utf-8"))
    payload["objective"].pop("local_law_c2_weight", None)
    broken_path.write_text(json.dumps(payload), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_learnability.py",
            "--family",
            "markov",
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
    assert summary["paper_safe"] is True
    assert summary["rows_loaded_before_filter"] == 2
    assert summary["run_count"] == 1
    assert summary["paper_safe_exclusion_reasons"] == {
        "missing_explicit_local_law_weight:objective_local_law_c2_weight": 1
    }
