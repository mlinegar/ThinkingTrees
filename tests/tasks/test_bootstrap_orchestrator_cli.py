from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Sequence


def _arg(cmd: Sequence[str], key: str) -> str:
    idx = list(cmd).index(key)
    return str(cmd[idx + 1])


def _write_jsonl(path: Path, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(n_rows):
            handle.write(json.dumps({"id": idx, "text": f"row_{idx}"}) + "\n")


def _fake_run_command(cmd, *, log_path, dry_run, cwd):  # noqa: ANN001
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("mocked\n", encoding="utf-8")

    script = str(cmd[1])

    if script.endswith("generate_manifesto_lawstress.py"):
        out = Path(_arg(cmd, "--output-dir"))
        train = int(_arg(cmd, "--train-size"))
        val = int(_arg(cmd, "--val-size"))
        test = int(_arg(cmd, "--test-size"))
        total = train + val + test
        _write_jsonl(out / "lawstress_records.jsonl", total)

    elif script.endswith("eval_manifesto_lawstress.py"):
        out = Path(_arg(cmd, "--output-dir"))
        mode = _arg(cmd, "--mode")
        if mode == "summarize_only":
            pred = Path(_arg(cmd, "--predictions-path"))
            _write_jsonl(pred, 5)
        else:
            metrics = {
                "overall": {
                    "c1_pass_rate": 60.0,
                    "c2_pass_rate": 60.0,
                    "c3_pass_rate": 60.0,
                },
                "success": {"overall_pass": True},
            }
            (out / "eval_metrics.json").parent.mkdir(parents=True, exist_ok=True)
            (out / "eval_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            _write_jsonl(out / "eval_results.jsonl", 5)

    elif script.endswith("run_single_manifesto_local_law_tune.py"):
        out = Path(_arg(cmd, "--output-root"))
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "best": {
                "config": {
                    "use_dspy_guidance": False,
                    "score_tolerance_raw": 20.0,
                    "max_attempts": 3,
                    "dspy_guidance_temperature": 0.1,
                    "dspy_guidance_max_tokens": 1200,
                    "summary_temperature": 0.08,
                }
            }
        }
        (out / "candidate_results.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("generate_manifesto_teacher_traces.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        records_path = out / "teacher_trace_records.jsonl"
        _write_jsonl(records_path, 12)
        manifest = {
            "accepted_docs": 12,
            "paths": {
                "records": str(records_path),
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    elif script.endswith("build_teacher_trace_split_views.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "summary_pairs_train.jsonl", 8)
        _write_jsonl(out / "summary_pairs_val.jsonl", 2)
        _write_jsonl(out / "benchmark_docs_train.jsonl", 4)
        _write_jsonl(out / "benchmark_docs_val.jsonl", 1)
        _write_jsonl(out / "benchmark_docs_test.jsonl", 1)
        payload = {
            "counts": {"docs": {"train": 4, "val": 1, "test": 1}},
            "split_ids": {"train": ["a"], "val": ["b"], "test": ["c"]},
        }
        (out / "split_ids.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("build_lawstress_split_views.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "summary_pairs_train.jsonl", 8)
        _write_jsonl(out / "summary_pairs_val.jsonl", 2)
        _write_jsonl(out / "summary_pairs_test.jsonl", 2)
        _write_jsonl(out / "benchmark_docs_train.jsonl", 4)
        _write_jsonl(out / "benchmark_docs_val.jsonl", 1)
        _write_jsonl(out / "benchmark_docs_test.jsonl", 1)
        payload = {
            "counts": {"docs": {"train": 4, "val": 1, "test": 1}},
            "split_ids": {"train": ["a"], "val": ["b"], "test": ["c"]},
        }
        (out / "split_ids.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("bootstrap_lawstress_summarizer.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        module_path = out / "trained_modules" / "unified_g_final.json"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("{}", encoding="utf-8")
        payload = {
            "paths": {
                "unified_g": str(module_path),
            }
        }
        (out / "bootstrap_stats.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("eval_manifesto_teacher_trace_local_laws.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        is_baseline = "baseline" in str(out)
        if is_baseline:
            overall = {
                "c1_pass_rate": 50.0,
                "c2_pass_rate": 50.0,
                "c3_pass_rate": 50.0,
                "avg_law_pass_rate": 50.0,
            }
        else:
            overall = {
                "c1_pass_rate": 57.0,
                "c2_pass_rate": 56.0,
                "c3_pass_rate": 58.0,
                "avg_law_pass_rate": 57.0,
            }
        (out / "eval_metrics.json").write_text(json.dumps({"overall": overall}), encoding="utf-8")
        _write_jsonl(out / "eval_results.jsonl", 3)

    elif script.endswith("eval_lawstress_dspy_module.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        optimized = {
            "metrics": {
                "overall": {
                    "c1_pass_rate": 60.0,
                    "c2_pass_rate": 60.0,
                    "c3_pass_rate": 60.0,
                },
                "success": {"overall_pass": True},
            },
            "groups": {},
        }
        (out / "optimized_eval_metrics.json").write_text(json.dumps(optimized), encoding="utf-8")
        (out / "baseline_eval_metrics.json").write_text(json.dumps(optimized), encoding="utf-8")

    elif script.endswith("train_manifesto_summary_sft.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "completed",
            "artifacts": {
                "adapter_or_model_path": str(out / "model" / "final"),
            },
        }
        (out / "sft_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    return {
        "command": [str(part) for part in cmd],
        "log_path": str(log_path),
        "returncode": 0,
        "dry_run": False,
        "started_at": "2026-03-03T00:00:00+00:00",
        "finished_at": "2026-03-03T00:00:01+00:00",
    }


def test_bootstrap_orchestrator_dry_run_smoke(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_poc")

    output_dir = tmp_path / "bootstrap_dry"
    rc = cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--manifesto-id",
            "dummy_anchor",
            "--dry-run",
            "--disable-genrm",
        ]
    )
    assert rc == 0

    manifest_path = output_dir / "bootstrap_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] in {"completed", "failed_gates"}
    assert manifest["config"]["judge_backend"] == "large_qwen"
    assert manifest["config"]["tournament_backend"] == "disabled"
    assert "stage_a_lawstress" in manifest["phases"]
    assert "stage_b_teacher_traces" in manifest["phases"]
    assert "stage_f_post_sft_eval" in manifest["phases"]


def test_bootstrap_orchestrator_mocked_run_writes_complete_manifest(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_poc")
    monkeypatch.setattr(cli, "_run_command", _fake_run_command)
    monkeypatch.setattr(cli, "_is_server_alive", lambda *args, **kwargs: True)

    output_dir = tmp_path / "bootstrap_mocked"
    rc = cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--manifesto-id",
            "dummy_anchor",
            "--disable-genrm",
        ]
    )
    assert rc == 0

    manifest = json.loads((output_dir / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["gates"]["overall_pass"] is True
    assert manifest["gates"]["post_sft"]["lawstress_success_overall_pass"] is True
    assert (
        manifest["gates"]["post_sft"]["real_anchor_local_law_improvement"]["average_gain_at_least_5"]
        is True
    )
    assert manifest["artifacts"]["backend_provenance"]["judge_backend"] == "large_qwen"
    assert manifest["artifacts"]["backend_provenance"]["tournament_backend"] == "disabled"


def test_bootstrap_orchestrator_mocked_run_skip_real_anchor(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_poc")
    monkeypatch.setattr(cli, "_run_command", _fake_run_command)
    monkeypatch.setattr(cli, "_is_server_alive", lambda *args, **kwargs: True)

    output_dir = tmp_path / "bootstrap_skip_real_anchor"
    rc = cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--disable-genrm",
            "--skip-real-anchor",
            "--student-training-strategy",
            "sft",
        ]
    )
    assert rc == 0

    manifest = json.loads((output_dir / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["phases"]["stage_b_teacher_traces"]["status"] == "skipped"
    assert manifest["phases"]["stage_d_real_anchor_baseline_eval"]["status"] == "skipped"
    assert manifest["gates"]["overall_pass"] is True
    assert manifest["gates"]["post_sft"]["real_anchor_local_law_improvement"]["pass"] is True


def test_bootstrap_orchestrator_rejects_genrm_mode(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_poc")
    output_dir = tmp_path / "bootstrap_reject_genrm"
    rc = cli.main(
        [
            "--output-dir",
            str(output_dir),
            "--no-disable-genrm",
        ]
    )
    assert rc == 2
