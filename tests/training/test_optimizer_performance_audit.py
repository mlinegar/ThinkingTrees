from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import dspy

from src.training.config import OptimizationConfig
from src.training.optimization.bootstrap import (
    BootstrapRandomSearchOptimizer,
    LabeledFewShotOptimizer,
)
from src.training.optimization.mipro import MIPROOptimizer
from src.training.optimization.performance import (
    CLASS_DATA_LIMITED,
    CLASS_FORCED_CONTROL,
    CLASS_IMPLEMENTATION_FALLBACK,
    CLASS_OBJECTIVE_MISMATCH,
    CLASS_RUNTIME_FAILURE,
    CLASS_WORKS,
    summarize_optimizer_runs,
)
from src.training.optimization.registry import OptimizerRegistry


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_auto_select_uses_config_defaults_for_random_search_threshold() -> None:
    assert OptimizerRegistry.auto_select(100, config=None) == "bootstrap_random_search"


def test_summarize_optimizer_runs_classifies_key_cases() -> None:
    works_rows = [
        {
            "optimizer_requested": "gepa",
            "optimizer_used": "gepa",
            "component": "scorer",
            "dataset_regime": "<=bootstrap_threshold(10)",
            "budget_mode": "medium",
            "compile_status": "completed",
            "heldout_gain": 0.4,
            "train_gain": 0.5,
            "input_mutation_flags": {},
        }
        for _ in range(5)
    ]
    fallback_rows = [
        {
            "optimizer_requested": "bootstrap_random_search",
            "optimizer_used": "bootstrap",
            "component": "scorer",
            "dataset_regime": "<=bootstrap_threshold(10)",
            "budget_mode": "medium",
            "compile_status": "fallback",
            "heldout_gain": 0.1,
            "train_gain": 0.2,
            "input_mutation_flags": {},
        }
        for _ in range(2)
    ] + [
        {
            "optimizer_requested": "bootstrap_random_search",
            "optimizer_used": "bootstrap_random_search",
            "component": "scorer",
            "dataset_regime": "<=bootstrap_threshold(10)",
            "budget_mode": "medium",
            "compile_status": "completed",
            "heldout_gain": 0.2,
            "train_gain": 0.2,
            "input_mutation_flags": {},
        }
        for _ in range(3)
    ]
    mismatch_rows = [
        {
            "optimizer_requested": "mipro",
            "optimizer_used": "mipro",
            "component": "leaf_summarizer",
            "dataset_regime": ">mipro_threshold(200)",
            "budget_mode": "medium",
            "compile_status": "completed",
            "heldout_gain": -0.2,
            "train_gain": 0.3,
            "input_mutation_flags": {"train_compaction": {"truncated": True}},
        }
        for _ in range(5)
    ]
    control_rows = [
        {
            "optimizer_requested": "gepa",
            "optimizer_used": "gepa",
            "component": "comparison_module",
            "dataset_regime": "<=bootstrap_threshold(10)",
            "budget_mode": "medium",
            "compile_status": "completed",
            "heldout_gain": 0.0,
            "train_gain": 0.0,
            "input_mutation_flags": {},
            "comparison_control_flag": True,
        }
    ]
    data_limited_rows = [
        {
            "optimizer_requested": "bootstrap",
            "optimizer_used": "bootstrap",
            "component": "merge_summarizer",
            "dataset_regime": "<=bootstrap_threshold(10)",
            "budget_mode": "medium",
            "compile_status": "skipped",
            "heldout_gain": float("nan"),
            "train_gain": float("nan"),
            "input_mutation_flags": {},
        }
        for _ in range(4)
    ] + [
        {
            "optimizer_requested": "bootstrap",
            "optimizer_used": "bootstrap",
            "component": "merge_summarizer",
            "dataset_regime": "<=bootstrap_threshold(10)",
            "budget_mode": "medium",
            "compile_status": "completed",
            "heldout_gain": 0.1,
            "train_gain": 0.1,
            "input_mutation_flags": {},
        }
    ]
    runtime_failure_rows = [
        {
            "optimizer_requested": "gepa",
            "optimizer_used": "gepa",
            "component": "leaf_summarizer",
            "dataset_regime": "(random_search_threshold(120),mipro_threshold(200)]",
            "budget_mode": "medium",
            "compile_status": "failed",
            "heldout_gain": float("nan"),
            "train_gain": float("nan"),
            "input_mutation_flags": {},
        }
        for _ in range(2)
    ] + [
        {
            "optimizer_requested": "gepa",
            "optimizer_used": "gepa",
            "component": "leaf_summarizer",
            "dataset_regime": "(random_search_threshold(120),mipro_threshold(200)]",
            "budget_mode": "medium",
            "compile_status": "completed",
            "heldout_gain": 0.1,
            "train_gain": 0.1,
            "input_mutation_flags": {},
        }
        for _ in range(3)
    ]

    summaries = summarize_optimizer_runs(
        works_rows
        + fallback_rows
        + mismatch_rows
        + control_rows
        + data_limited_rows
        + runtime_failure_rows
    )
    by_key = {
        (row["optimizer_requested"], row["component"]): row["classification"]
        for row in summaries
    }
    assert by_key[("gepa", "scorer")] == CLASS_WORKS
    assert by_key[("bootstrap_random_search", "scorer")] == CLASS_IMPLEMENTATION_FALLBACK
    assert by_key[("mipro", "leaf_summarizer")] == CLASS_OBJECTIVE_MISMATCH
    assert by_key[("gepa", "comparison_module")] == CLASS_FORCED_CONTROL
    assert by_key[("bootstrap", "merge_summarizer")] == CLASS_DATA_LIMITED
    assert by_key[("gepa", "leaf_summarizer")] == CLASS_RUNTIME_FAILURE


def test_labeled_fewshot_unavailable_sets_noop_audit(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dspy.teleprompt":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    opt = LabeledFewShotOptimizer(config=OptimizationConfig())
    student = object()
    compiled = opt.compile(student=student, trainset=[], valset=[], metric=None)
    assert compiled is student
    assert opt.last_compile_audit["compile_status"] == "noop"
    assert opt.last_compile_audit["fallback_reason"] == "teleprompter_unavailable"


def test_bootstrap_random_search_unavailable_sets_fallback_audit(monkeypatch) -> None:
    original_import = builtins.__import__

    class FakeBootstrapFewShot:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, student, teacher=None, trainset=None):
            return {"compiled": "bootstrap", "student": student}

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dspy.teleprompt":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr("src.training.optimization.bootstrap.dspy.BootstrapFewShot", FakeBootstrapFewShot)
    opt = BootstrapRandomSearchOptimizer(config=OptimizationConfig())
    compiled = opt.compile(student={"student": True}, trainset=[], valset=[], metric=None)
    assert compiled["compiled"] == "bootstrap"
    assert opt.last_compile_audit["compile_status"] == "fallback"
    assert opt.last_compile_audit["optimizer_used"] == "bootstrap"


def test_mipro_compaction_sets_input_mutation_flags(monkeypatch) -> None:
    class DummyStudent:
        def forward(self, summary):
            return summary

    class FakeMIPRO:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, student=None, trainset=None, **kwargs):
            return {"compiled_trainset": trainset, "kwargs": kwargs}

    monkeypatch.setattr("src.training.optimization.mipro.dspy.MIPROv2", FakeMIPRO)
    example = dspy.Example(
        original_content="x" * 50,
        summary="short summary",
        rubric="rubric",
    ).with_inputs("original_content", "summary", "rubric")
    config = OptimizationConfig(
        mipro_max_example_chars=10,
        mipro_drop_optional_original_content=True,
    )
    opt = MIPROOptimizer(config=config)
    opt.compile(
        student=DummyStudent(),
        trainset=[example],
        valset=[example],
        metric=None,
    )
    flags = opt.last_compile_audit["input_mutation_flags"]["train_compaction"]
    assert flags["truncated"] is True
    assert flags["dropped_optional_original_content"] is True


def test_optimizer_audit_scripts_generate_manifest_and_summary(tmp_path: Path) -> None:
    run_script = REPO_ROOT / "scripts" / "run_optimizer_performance_audit.py"
    report_script = REPO_ROOT / "scripts" / "report_optimizer_performance_audit.py"
    output_root = tmp_path / "audit"

    proc = subprocess.run(
        [
            sys.executable,
            str(run_script),
            "--dry-run",
            "--output-root",
            str(output_root),
            "--seeds",
            "0",
            "--optimizers",
            "gepa",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest_path = output_root / "optimizer_audit_manifest.json"
    assert manifest_path.exists()

    run_dir = output_root / "gepa__train_10__seed_0"
    run_dir.mkdir(parents=True, exist_ok=True)
    final_stats = {
        "optimizer_diagnostics": {
            "runs": [
                {
                    "optimizer_requested": "gepa",
                    "optimizer_used": "gepa",
                    "component": "scorer",
                    "dataset_regime": "<=bootstrap_threshold(10)",
                    "budget_mode": "medium",
                    "compile_status": "completed",
                    "heldout_gain": 0.2,
                    "train_gain": 0.3,
                    "input_mutation_flags": {},
                }
            ]
        }
    }
    (run_dir / "final_stats.json").write_text(json.dumps(final_stats), encoding="utf-8")
    markov_summary = {
        "witness_gap_table": [
            {
                "baseline_family": "official_fno",
                "train_doc_count": 8,
                "test_root_mae_mean": 0.1,
                "gap_to_ridge_control": 0.09,
                "gap_to_exact_witness": 0.1,
                "cause_code": "optimization_limit",
                "objective_variant": "count_ce_only",
            }
        ]
    }
    markov_summary_path = output_root / "markov_summary.json"
    markov_summary_path.write_text(json.dumps(markov_summary), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(report_script),
            "--manifest",
            str(manifest_path),
            "--markov-summary",
            str(markov_summary_path),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (output_root / "optimizer_performance_summary.json").exists()
    assert (output_root / "optimizer_performance_summary.md").exists()


def test_save_results_writes_standalone_optimizer_manifest(tmp_path: Path) -> None:
    from src.training.run_pipeline import save_results

    stats = {
        "optimizer_diagnostics": {
            "runs": [
                {
                    "optimizer_requested": "gepa",
                    "optimizer_used": "gepa",
                    "component": "scorer",
                    "dataset_regime": "<=bootstrap_threshold(10)",
                    "budget_mode": "medium",
                    "compile_status": "completed",
                    "heldout_gain": 0.1,
                    "train_gain": 0.2,
                    "input_mutation_flags": {},
                }
            ],
            "cell_summaries": [
                {
                    "optimizer_requested": "gepa",
                    "component": "scorer",
                    "dataset_regime": "<=bootstrap_threshold(10)",
                    "budget_mode": "medium",
                    "classification": "works",
                }
            ],
            "comparison_control_runs": [
                {
                    "optimizer_requested": "gepa",
                    "optimizer_used": "gepa",
                    "component": "comparison_module",
                    "comparison_control_flag": True,
                }
            ],
        }
    }

    save_results(stats, tmp_path)

    final_stats_path = tmp_path / "final_stats.json"
    audit_manifest_path = tmp_path / "optimizer_audit_manifest.json"
    assert final_stats_path.exists()
    assert audit_manifest_path.exists()

    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    assert audit_manifest["final_stats_path"] == str(final_stats_path)
    assert len(audit_manifest["runs"]) == 1
    assert len(audit_manifest["cell_summaries"]) == 1
    assert len(audit_manifest["comparison_control_runs"]) == 1
