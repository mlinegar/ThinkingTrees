from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.report_markov_local_law_learnability import _best_row, _row_selection_objective
from src.tree.markov_changepoint_ops_count_simulation import (
    OPSCountConfig,
    run_markov_changepoint_ops_count_experiment,
)


def test_markov_local_law_report_smoke(tmp_path: Path) -> None:
    input_root = tmp_path / "markov_runs"
    output_dir = tmp_path / "report"
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
        violation_tau=0.0,
        data_seed=3,
        use_cuda=False,
        torch_threads=1,
    )

    for llw in (0.0, 0.5):
        for model_seed in (0, 1):
            summary = run_markov_changepoint_ops_count_experiment(
                OPSCountConfig(**common, seed=model_seed, model_seed=model_seed, local_law_weight=llw)
            )
            out_dir = input_root / f"llw_{str(llw).replace('.', 'p')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"seed_{model_seed}.json"
            out_path.write_text(summary.to_json(), encoding="utf-8")

    legacy_path = input_root / "llw_0p0" / "seed_0.json"
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    learned = legacy_payload["metrics"]["learned"]
    learned.pop("test_objective_full_labels", None)
    learned.pop("test_objective_root_term", None)
    learned.pop("test_objective_leaf_term", None)
    learned.pop("test_objective_merge_term", None)
    learned.pop("test_objective_schedule_consistency_term", None)
    legacy_path.write_text(json.dumps(legacy_payload, indent=2, sort_keys=True), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_local_law_learnability.py",
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
    )

    summary_path = output_dir / "markov_local_law_learnability_summary.json"
    md_path = output_dir / "markov_local_law_learnability.md"
    rows_path = output_dir / "markov_local_law_learnability_rows.json"
    pdf_path = output_dir / "markov_local_law_learnability_report.pdf"
    assert summary_path.exists()
    assert md_path.exists()
    assert rows_path.exists()
    assert pdf_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert int(summary["run_count"]) == 4
    assert len(list(summary["aggregated_rows"])) >= 2
    assert len(list(summary["figures"])) >= 2
    assert summary["pdf"] == str(pdf_path)
    assert int(summary["exact_test_objective_row_count"]) == 3
    assert int(summary["proxy_test_objective_row_count"]) == 1
    assert summary["selection_metric_name"] == "heldout_objective_for_report"
    assert "best_by_objective" in summary
    assert "best_by_theorem_score" in summary
    assert "recommended_sparse_objective_point" in summary
    assert "recommended_sparse_theorem_point" in summary
    assert "metric_definitions" in summary
    assert "figure_titles" in summary


def test_markov_local_law_report_prefers_weighted_objective_for_selection() -> None:
    rows = [
        {
            "train_docs": 128,
            "audit_fraction": 0.1,
            "local_law_weight": 0.5,
            "schedule_consistency_weight": 0.2,
            "heldout_objective_for_report": 0.30,
            "test_objective_full_labels": 0.30,
            "test_unweighted_objective_full_labels": 0.95,
            "theorem_score": 0.80,
        },
        {
            "train_docs": 128,
            "audit_fraction": 0.1,
            "local_law_weight": 1.0,
            "schedule_consistency_weight": 0.2,
            "heldout_objective_for_report": 0.18,
            "test_objective_full_labels": 0.18,
            "test_unweighted_objective_full_labels": 1.20,
            "theorem_score": 0.90,
        },
    ]

    assert _row_selection_objective(rows[1]) == 0.18
    selected = _best_row(
        rows,
        train_docs=128,
        audit_fraction=0.1,
        schedule_consistency_weight=0.2,
    )
    assert selected is rows[1]
