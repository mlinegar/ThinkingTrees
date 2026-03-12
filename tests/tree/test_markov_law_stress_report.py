from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_fake_learned(
    path: Path,
    *,
    law_package: str,
    data_seed: int,
    model_seed: int,
    audit_fraction: float,
    val_c1: float,
    val_c2: float,
    val_c3: float,
    val_root: float,
    test_c1: float,
    test_c2: float,
    test_c3: float,
    test_root: float,
    test_spread: float,
) -> None:
    payload = {
        "config": {
            "n_regimes": 4,
            "fixed_leaf_tokens": 16,
            "train_docs": 128,
            "val_docs": 32,
            "test_docs": 64,
            "audit_fraction": float(audit_fraction),
            "root_weight": 1.0,
            "state_dim": 64,
            "hidden_dim": 256,
            "n_epochs": 4,
            "feature_mode": "full",
            "max_segments": 5,
            "law_package": str(law_package),
            "effective_data_seed": int(data_seed),
            "effective_model_seed": int(model_seed),
        },
        "objective": {"law_package": str(law_package)},
        "metrics": {
            "learned": {
                "val_c1_leaf_mae_n": float(val_c1),
                "val_c2_idempotence_mae_n": float(val_c2),
                "val_c3_merge_mae_n": float(val_c3),
                "val_root_mae_n": float(val_root),
                "val_schedule_spread_mean_n": 0.05,
                "val_theorem_bundle_score_n": float(val_c1 + val_c2 + val_c3),
                "test_c1_leaf_mae_n": float(test_c1),
                "test_c2_idempotence_mae_n": float(test_c2),
                "test_c3_merge_mae_n": float(test_c3),
                "test_root_mae_n": float(test_root),
                "test_schedule_spread_mean_n": float(test_spread),
                "test_theorem_bundle_score_n": float(test_c1 + test_c2 + test_c3),
                "test_c2_r1_mae_n": float(test_c2),
                "test_c2_r2_mae_n": float(test_c2 + 0.01),
                "test_c2_r4_mae_n": float(test_c2 + 0.02),
                "test_resummary_root_drift_r1_n": 0.01,
                "test_resummary_root_drift_r2_n": 0.02,
                "test_resummary_root_drift_r4_n": 0.03,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_fake_exact(path: Path, *, family: str) -> None:
    payload = {
        "config": {
            "n_regimes": 4,
            "fixed_leaf_tokens": 16,
            "train_docs": 128,
            "val_docs": 32,
            "test_docs": 64,
            "audit_fraction": 1.0,
            "root_weight": 1.0,
            "state_dim": 64,
            "hidden_dim": 256,
            "n_epochs": 1,
            "feature_mode": "full",
            "max_segments": 5,
            "exact_family": str(family),
            "effective_data_seed": 0,
            "effective_model_seed": 0,
        },
        "metrics": {
            "stress_family": {
                "stress_family_name": str(family),
                "test_c1_leaf_mae_n": 0.0 if family == "exact" else 0.2,
                "test_c2_idempotence_mae_n": 0.0 if family != "flip_R2" else 0.3,
                "test_c3_merge_mae_n": 0.0 if family != "count_only" else 0.25,
                "test_root_mae_n": 0.0 if family in {"exact", "leaf_bucket"} else 0.25,
                "test_schedule_spread_mean_n": 0.0,
                "test_theorem_bundle_score_n": 0.1,
                "test_c2_r1_mae_n": 0.0,
                "test_c2_r2_mae_n": 0.3 if family == "flip_R2" else 0.0,
                "test_c2_r4_mae_n": 0.3 if family == "flip_R2" else 0.0,
                "test_resummary_root_drift_r1_n": 0.1 if family == "flip_R2" else 0.0,
                "test_resummary_root_drift_r2_n": 0.3 if family == "flip_R2" else 0.0,
                "test_resummary_root_drift_r4_n": 0.3 if family == "flip_R2" else 0.0,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_markov_law_stress_report_emits_pass_rates_and_exact_family_page(tmp_path: Path) -> None:
    input_root = tmp_path / "runs"
    output_dir = tmp_path / "report"
    for model_seed in (0, 1):
        _write_fake_learned(
            input_root / f"root_only_seed{model_seed}" / f"seed_{model_seed}.json",
            law_package="root_only",
            data_seed=0,
            model_seed=model_seed,
            audit_fraction=0.1,
            val_c1=0.40,
            val_c2=0.30,
            val_c3=0.40,
            val_root=0.20,
            test_c1=0.40,
            test_c2=0.30,
            test_c3=0.40,
            test_root=0.20,
            test_spread=0.10,
        )
        _write_fake_learned(
            input_root / f"all_laws_seed{model_seed}" / f"seed_{model_seed}.json",
            law_package="all_laws_plus_sched",
            data_seed=0,
            model_seed=model_seed,
            audit_fraction=0.1,
            val_c1=0.20,
            val_c2=0.10,
            val_c3=0.20,
            val_root=0.20,
            test_c1=0.20,
            test_c2=0.10,
            test_c3=0.20,
            test_root=0.19,
            test_spread=0.05,
        )
    _write_fake_exact(input_root / "exact" / "seed_0.json", family="exact")
    _write_fake_exact(input_root / "flip" / "seed_1.json", family="flip_R2")

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/report_markov_law_stress.py",
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

    summary = json.loads((output_dir / "markov_law_stress_summary.json").read_text(encoding="utf-8"))
    assert int(summary["aggregated_row_count"]) >= 1
    assert "unified_core" in summary
    rows = list(summary["aggregated_rows"])
    assert "c1_pass_rate" in rows[0]
    assert "bundle_full_success_rate" in rows[0]
    assert "root_ratio" in rows[0]
    assert any("exact_family_counterexamples.png" in fig for fig in summary["figures"])
    assert (output_dir / "markov_law_stress_report.pdf").exists()
