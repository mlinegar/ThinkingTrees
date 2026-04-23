from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from src.training.run_pipeline import (
    resolve_embedding_proxy_config,
    resolve_generator_training_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_args() -> Namespace:
    return Namespace(
        port=8000,
        adaptive_embedding_proxy=None,
        adaptive_embedding_api_base=None,
        adaptive_embedding_model=None,
        adaptive_embedding_models_by_adapter=None,
        adaptive_embedding_batch_size=None,
        adaptive_embedding_timeout_sec=None,
        adaptive_embedding_min_samples=None,
        adaptive_embedding_head_method=None,
        adaptive_embedding_target_field=None,
        adaptive_embedding_target_transform=None,
        adaptive_embedding_ridge_lambda=None,
        adaptive_embedding_head_epochs=None,
        adaptive_embedding_head_lr=None,
        adaptive_embedding_head_weight_decay=None,
        adaptive_embedding_full_finetune=None,
        adaptive_embedding_finetune_command=None,
        adaptive_embedding_max_text_chars=None,
        adaptive_embedding_retrain_rounds=None,
        adaptive_embedding_include_val=None,
        adaptive_embedding_truth_sources=None,
        adaptive_embedding_score_key=None,
        embedding_proxy_fail_on_error=None,
        rerun_embedding_proxy_on_resume=None,
        generator_method=None,
        train_generator=None,
        generator_model=None,
        generator_use_lora=None,
        generator_learning_rate=None,
        generator_epochs=None,
        generator_batch_size=None,
        generator_fail_on_error=None,
        rerun_generator_on_resume=None,
        generator_min_preferences=None,
        unified_min_preferences=None,
    )


def test_embedding_proxy_config_resolves_fail_and_rerun_flags() -> None:
    args = _base_args()
    args.embedding_proxy_fail_on_error = True
    args.rerun_embedding_proxy_on_resume = True
    settings = {
        "servers": {"embedding_url": "http://localhost:8003/v1", "embedding_model": "Qwen/Qwen3-Embedding-8B"},
        "chunking": {
            "adaptive": {
                "embedding_proxy": {
                    "enabled": True,
                    "head_method": "ridge",
                    "fail_on_error": False,
                    "rerun_on_resume": False,
                }
            }
        },
    }
    cfg = resolve_embedding_proxy_config(args, settings=settings, adaptive_cfg=None)
    assert cfg.fail_on_error is True
    assert cfg.rerun_on_resume is True


def test_generator_policy_cli_overrides_settings() -> None:
    args = _base_args()
    args.train_generator = True
    args.generator_method = "sft"
    args.generator_use_lora = False
    args.generator_learning_rate = 3e-5
    args.generator_epochs = 7
    args.generator_batch_size = 4
    args.generator_fail_on_error = True
    args.rerun_generator_on_resume = True
    args.generator_min_preferences = 33
    settings = {
        "generator": {
            "enabled": False,
            "method": "dpo",
            "use_lora": True,
            "learning_rate": 1e-5,
            "epochs": 3,
            "batch_size": 2,
            "min_preferences": 50,
            "fail_on_error": False,
            "rerun_on_resume": False,
        }
    }
    policy = resolve_generator_training_policy(args, training_settings=settings)
    assert policy.enabled is True
    assert policy.method == "sft"
    assert policy.use_lora is False
    assert policy.learning_rate == 3e-5
    assert policy.epochs == 7
    assert policy.batch_size == 4
    assert policy.fail_on_error is True
    assert policy.rerun_on_resume is True
    assert policy.min_preferences == 33


def test_wrapper_help_mentions_new_method_flags() -> None:
    script = REPO_ROOT / "scripts" / "run_training_pipeline.sh"
    text = script.read_text(encoding="utf-8")
    assert "ridge|linear_sgd|mil_sgd" in text
    assert "--embedding-proxy-fail-on-error" in text
    assert "--generator-use-lora" in text
    assert "--rerun-generator-on-resume" in text


def test_compare_runner_dry_run_writes_manifest(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "run_method_compare.py"
    output_root = tmp_path / "compare"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest_path = output_root / "method_compare_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "fast-smoke"
    assert len(payload["entries"]) == 4


def test_compare_report_generates_summary_files(tmp_path: Path) -> None:
    output_root = tmp_path / "compare_report"
    run_dir = output_root / "baseline_llm"
    run_dir.mkdir(parents=True, exist_ok=True)
    final_stats = {
        "success": True,
        "train": {"mae": 10.0},
        "test": {"mae": 11.5},
        "method_status": {
            "llm_prompt_optimization": {"completed": True, "skipped": False},
            "embedding_proxy": {"completed": False, "skipped": True},
            "neural_operators": {"completed": False, "skipped": True},
            "generator_finetune": {"completed": False, "skipped": True},
        },
    }
    (run_dir / "final_stats.json").write_text(json.dumps(final_stats), encoding="utf-8")
    canonical_rows = [
        {
            "experiment_id": "exp_llm",
            "phase": "eval",
            "benchmark_ref": {
                "benchmark_id": "treepo_task::manifesto_rile",
                "family": "treepo_task",
                "scope": "manifesto_rile",
                "name": "manifesto_rile",
            },
            "method_ref": {
                "method_id": "llm_prompt_optimization::training_pipeline",
                "family": "llm_prompt_optimization",
                "variant": "training_pipeline",
                "adapter": "treepo_training",
            },
            "split": "test",
            "train_docs": 30,
            "supervision_ref": {
                "topology_scope": "document",
                "unit_selector": "document",
                "supervision_kind": "scalar",
                "label_source": "dataset_labels",
                "labeler_kind": "gold_score",
                "doc_sample_probability": 1.0,
                "coverage_label": "100% labeled docs",
            },
            "metric_name": "mae",
            "metric_value": 11.5,
            "artifact_refs": [],
        }
    ]
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in canonical_rows) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "mode": "fast-smoke",
        "task": "manifesto_rile",
        "dataset": "manifesto",
        "entries": [
            {
                "profile": "baseline_llm",
                "status": "success",
                "exit_code": 0,
                "duration_seconds": 12.3,
                "run_dir": str(run_dir),
            }
        ],
    }
    manifest_path = output_root / "method_compare_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report_script = REPO_ROOT / "scripts" / "report_method_compare.py"
    proc = subprocess.run(
        [sys.executable, str(report_script), "--manifest", str(manifest_path)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary_path = output_root / "comparison_summary.json"
    assert summary_path.exists()
    assert (output_root / "comparison_summary.md").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["canonical_reporting"]["row_count"] == 1
    assert summary["canonical_reporting"]["method_families"] == ["llm_prompt_optimization"]
    assert summary["canonical_reporting"]["main_body_plot_specs"] == []
