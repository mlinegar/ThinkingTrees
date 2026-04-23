from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.training.run_pipeline import parse_args, save_results


def test_save_results_writes_canonical_experiment_outputs(tmp_path: Path) -> None:
    neural_output_dir = tmp_path / "neural_operators"
    neural_output_dir.mkdir(parents=True, exist_ok=True)
    ctreepo_dir = neural_output_dir / "ctreepo"
    ctreepo_dir.mkdir(parents=True, exist_ok=True)
    trained_modules_dir = tmp_path / "trained_modules"
    trained_modules_dir.mkdir(parents=True, exist_ok=True)
    proxy_models_dir = tmp_path / "proxy_models"
    proxy_models_dir.mkdir(parents=True, exist_ok=True)
    generator_dir = tmp_path / "generator"
    generator_dir.mkdir(parents=True, exist_ok=True)
    phase2_runtime_dir = tmp_path / "checkpoints" / "phase2_runtime" / "sig123"
    phase2_runtime_dir.mkdir(parents=True, exist_ok=True)
    phase2_gepa_exports_dir = phase2_runtime_dir / "gepa_exports"
    phase2_gepa_exports_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "reproducibility_manifest.json").write_text("{}", encoding="utf-8")
    (neural_output_dir / "reproducibility_manifest.json").write_text("{}", encoding="utf-8")
    (neural_output_dir / "search_spec.json").write_text("{}", encoding="utf-8")
    (neural_output_dir / "search_results.json").write_text("{}", encoding="utf-8")
    (ctreepo_dir / "best.pt").write_text("stub", encoding="utf-8")
    (ctreepo_dir / "training_result.json").write_text("{}", encoding="utf-8")
    (ctreepo_dir / "reproducibility_manifest.json").write_text("{}", encoding="utf-8")
    (trained_modules_dir / "scorer_final.json").write_text("{}", encoding="utf-8")
    (trained_modules_dir / "leaf_summarizer_final.json").write_text("{}", encoding="utf-8")
    (trained_modules_dir / "merge_summarizer_final.json").write_text("{}", encoding="utf-8")
    (proxy_models_dir / "embedding_proxy.json").write_text("{}", encoding="utf-8")
    (proxy_models_dir / "embedding_finetune.jsonl").write_text("{}\n", encoding="utf-8")
    (generator_dir / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
    (phase2_runtime_dir / "state.json").write_text("{}", encoding="utf-8")
    (phase2_gepa_exports_dir / "scorer_gepa_trajectory_snapshot.json").write_text(
        "{}",
        encoding="utf-8",
    )
    nested_row = {
        "experiment_id": "nested",
        "phase": "eval",
        "benchmark_ref": {
            "benchmark_id": "treepo_task::manifesto_rile",
            "family": "treepo_task",
            "scope": "manifesto_rile",
            "name": "manifesto_rile",
        },
        "method_ref": {
            "method_id": "ctreepo::training_pipeline",
            "family": "ctreepo",
            "variant": "local_law_training",
            "adapter": "treepo_training",
        },
        "split": "validation",
        "metric_name": "root_mae",
        "metric_value": 0.11,
        "supervision_ref": {
            "topology_scope": "tree",
            "unit_selector": "internal",
            "supervision_kind": "scalar",
            "label_source": "label_now",
            "labeler_kind": "oracle_score",
            "coverage_label": "internal_only",
        },
        "control_ref": {
            "control_family": "ctreepo_local_law",
            "law_ids": ["L1", "L2"],
            "applies_to": "leaf+internal",
            "enabled": True,
            "source_kind": "task_oracle",
        },
        "artifact_refs": [
            "ctreepo_training_result_path",
            "summary_json",
            "search_spec_json",
            "search_results_json",
            "reproducibility_manifest_json",
        ],
    }
    (neural_output_dir / "results.jsonl").write_text(json.dumps(nested_row) + "\n", encoding="utf-8")
    (neural_output_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "ctreepo",
                        "artifacts": {
                            "best_model_path": str(ctreepo_dir / "best.pt"),
                            "training_result_path": str(ctreepo_dir / "training_result.json"),
                            "reproducibility_manifest_path": str(ctreepo_dir / "reproducibility_manifest.json"),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    stats = {
        "success": True,
        "task": "manifesto_rile",
        "train": {
            "mae": 0.2,
            "pearson_r": 0.7,
            "n_evaluated": 4,
        },
        "test": {
            "mae": 0.3,
            "spearman_r": 0.6,
            "n_evaluated": 2,
        },
        "method_status": {
            "llm_prompt_optimization": {
                "enabled": True,
                "attempted": True,
                "completed": True,
                "skipped": False,
                "duration_seconds": 12.0,
            },
            "embedding_proxy": {
                "enabled": True,
                "attempted": True,
                "completed": True,
                "skipped": False,
                "duration_seconds": 4.0,
            },
            "generator_finetune": {
                "enabled": True,
                "attempted": True,
                "completed": False,
                "skipped": True,
                "duration_seconds": 0.0,
            },
        },
        "rounds": [{"round": 1, "metric_before": 0.61, "metric_after": 0.74}],
        "optimizer_diagnostics": {
            "runs": [
                {
                    "optimizer_requested": "gepa",
                    "optimizer_used": "gepa",
                    "component": "scorer",
                    "dataset_size": 12,
                    "iteration": 1,
                    "compile_status": "completed",
                    "heldout_gain": 0.13,
                    "train_gain": 0.09,
                    "input_mutation_flags": {},
                }
            ],
            "cell_summaries": [],
            "comparison_control_runs": [],
        },
        "phase2_runtime_signature_id": "sig123",
        "phase2_runtime_resume_dir": str(phase2_runtime_dir),
        "scorer_module_path": str(trained_modules_dir / "scorer_final.json"),
        "leaf_summarizer_module_path": str(trained_modules_dir / "leaf_summarizer_final.json"),
        "merge_summarizer_module_path": str(trained_modules_dir / "merge_summarizer_final.json"),
        "treepo_audit": {
            "aggregate": {
                "n_trees": 3,
                "nodes_audited": 9,
                "nodes_failed": 1,
                "failure_rate": 1.0 / 9.0,
            },
            "pooled_ipw": {
                "violation_rate": 0.15,
            },
        },
        "neural_operator_training": {
            "output_dir": str(neural_output_dir),
            "summary_path": str(neural_output_dir / "summary.json"),
            "ctreepo_local_law": {
                "leaf_audit_weight": 0.4,
                "merge_audit_weight": 0.8,
                "violation_threshold": 7.5,
                "label_source_kind": "task_oracle",
            },
        },
        "adaptive_embedding_proxy_training": {
            "artifact_path": str(proxy_models_dir / "embedding_proxy.json"),
            "full_finetune": {
                "dataset_export": {
                    "path": str(proxy_models_dir / "embedding_finetune.jsonl"),
                    "rows": 1,
                }
            },
        },
        "generator_training": {
            "model_path": str(generator_dir / "adapter_model.safetensors"),
        },
    }
    args = argparse.Namespace(
        task="manifesto_rile",
        enable_treepo_audit=True,
        treepo_audit_idempotence=True,
        treepo_audit_sample_budget=10,
        treepo_audit_sampling_probability=1.0,
        treepo_audit_sampling_strategy="random",
        treepo_audit_discrepancy_threshold=0.1,
        train_samples=12,
    )

    save_results(stats, tmp_path, args=args)

    assert (tmp_path / "final_stats.json").exists()
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "experiment_status.json").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert (tmp_path / "results.jsonl").exists()

    status = json.loads((tmp_path / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    artifacts = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf-8"))
    assert "reproducibility_manifest_json" in artifacts["artifacts"]
    assert "neural_operator_reproducibility_manifest_json" in artifacts["artifacts"]
    assert "neural_operator_search_spec_json" in artifacts["artifacts"]
    assert "neural_operator_search_results_json" in artifacts["artifacts"]
    assert "ctreepo_training_result_path" in artifacts["artifacts"]
    assert "phase2_optimization_trace_spec_json" in artifacts["artifacts"]
    assert "phase2_optimization_trace_results_json" in artifacts["artifacts"]
    assert "phase2_runtime_state_json" in artifacts["artifacts"]
    assert "phase2_scorer_gepa_trajectory_snapshot_json" in artifacts["artifacts"]
    assert "scorer_module_path" in artifacts["artifacts"]
    assert "leaf_summarizer_module_path" in artifacts["artifacts"]
    assert "merge_summarizer_module_path" in artifacts["artifacts"]
    assert "embedding_proxy_training_spec_json" in artifacts["artifacts"]
    assert "embedding_proxy_training_results_json" in artifacts["artifacts"]
    assert "embedding_proxy_artifact_json" in artifacts["artifacts"]
    assert "embedding_proxy_finetune_dataset_jsonl" in artifacts["artifacts"]
    assert "generator_training_spec_json" in artifacts["artifacts"]
    assert "generator_training_results_json" in artifacts["artifacts"]
    assert "generator_model_path" in artifacts["artifacts"]

    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metric_names = {row["metric_name"] for row in rows}
    assert "mae" in metric_names
    assert "failure_rate" in metric_names
    assert "violation_rate" in metric_names
    assert "root_mae" in metric_names
    llm_mae_row = next(row for row in rows if row["metric_name"] == "mae" and row["split"] == "train")
    assert llm_mae_row["train_docs"] == 12
    assert llm_mae_row["supervision_ref"]["coverage_label"] == "100% labeled docs"
    imported_ctreepo_row = next(row for row in rows if row["metric_name"] == "root_mae")
    assert imported_ctreepo_row["train_docs"] == 12
    assert imported_ctreepo_row["metadata"]["imported_via"] == "run_pipeline"
    assert "ctreepo_training_result_path" in imported_ctreepo_row["artifact_refs"]
    assert "neural_operator_summary_json" in imported_ctreepo_row["artifact_refs"]
    assert "neural_operator_search_spec_json" in imported_ctreepo_row["artifact_refs"]
    assert "neural_operator_search_results_json" in imported_ctreepo_row["artifact_refs"]
    assert "neural_operator_reproducibility_manifest_json" in imported_ctreepo_row["artifact_refs"]
    llm_train_mae_row = next(row for row in rows if row["metric_name"] == "mae" and row["split"] == "train")
    assert "phase2_optimization_trace_results_json" in llm_train_mae_row["artifact_refs"]
    assert "scorer_module_path" in llm_train_mae_row["artifact_refs"]
    embedding_status_row = next(
        row for row in rows if row["metric_name"] == "completed" and row["metadata"]["method_key"] == "embedding_proxy"
    )
    assert "embedding_proxy_training_spec_json" in embedding_status_row["artifact_refs"]
    assert "embedding_proxy_training_results_json" in embedding_status_row["artifact_refs"]
    assert "embedding_proxy_artifact_json" in embedding_status_row["artifact_refs"]
    generator_status_row = next(
        row for row in rows if row["metric_name"] == "completed" and row["metadata"]["method_key"] == "generator_finetune"
    )
    assert "generator_training_spec_json" in generator_status_row["artifact_refs"]
    assert "generator_training_results_json" in generator_status_row["artifact_refs"]
    assert "generator_model_path" in generator_status_row["artifact_refs"]


def test_parse_args_accepts_neural_operator_search_specs(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--output-dir",
            "outputs/test_run_pipeline_search",
            "--neural-operators-ctreepo-search-spec",
            "config/ctreepo_search.json",
            "--neural-operators-mergeable-search-spec",
            "config/mergeable_search.json",
        ],
    )

    args = parse_args()

    assert args.neural_operators_ctreepo_search_spec == "config/ctreepo_search.json"
    assert args.neural_operators_mergeable_search_spec == "config/mergeable_search.json"
