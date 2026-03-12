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


def _fake_run_command(cmd, *, log_path, cwd, dry_run, env_overrides=None):  # noqa: ANN001,ARG001
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("mocked\n", encoding="utf-8")

    # Shell invocations (server start/stop) are no-op in tests.
    if not cmd:
        return {"returncode": 0, "log_path": str(log), "command": []}
    if str(cmd[0]) in {"bash", "./scripts/stop_small_servers.sh"}:
        return {
            "command": [str(part) for part in cmd],
            "log_path": str(log),
            "returncode": 0,
            "dry_run": False,
            "started_at": "2026-03-03T00:00:00+00:00",
            "finished_at": "2026-03-03T00:00:01+00:00",
        }

    script = str(cmd[1]) if len(cmd) > 1 else ""

    if script.endswith("generate_manifesto_lawstress.py"):
        out = Path(_arg(cmd, "--output-dir"))
        train = int(_arg(cmd, "--train-size"))
        val = int(_arg(cmd, "--val-size"))
        test = int(_arg(cmd, "--test-size"))
        _write_jsonl(out / "lawstress_records.jsonl", train + val + test)

    elif script.endswith("eval_manifesto_lawstress.py"):
        out = Path(_arg(cmd, "--output-dir"))
        mode = _arg(cmd, "--mode")
        if mode == "summarize_only":
            pred = Path(_arg(cmd, "--predictions-path"))
            _write_jsonl(pred, 5)
        else:
            payload = {
                "overall": {"c1_pass_rate": 50.0, "c2_pass_rate": 50.0, "c3_pass_rate": 50.0, "mae": 0.1},
                "success": {"overall_pass": False},
            }
            (out / "eval_metrics.json").parent.mkdir(parents=True, exist_ok=True)
            (out / "eval_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

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
                    "dspy_guidance_max_tokens": 1600,
                    "summary_temperature": 0.1,
                }
            }
        }
        (out / "candidate_results.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("generate_manifesto_teacher_traces.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        records_path = out / "teacher_trace_records.jsonl"
        _write_jsonl(records_path, 12)
        payload = {"accepted_docs": 12, "paths": {"records": str(records_path)}}
        (out / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("build_teacher_trace_split_views.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out / "summary_pairs_train.jsonl", 8)
        _write_jsonl(out / "summary_pairs_val.jsonl", 2)
        payload = {
            "counts": {"docs": {"train": 8, "val": 2, "test": 2}},
            "split_ids": {"train": ["a"], "val": ["b"], "test": ["c"]},
        }
        (out / "split_ids.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("bootstrap_lawstress_summarizer.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        module_path = out / "trained_modules" / "unified_g_final.json"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("{}", encoding="utf-8")
        payload = {"paths": {"unified_g": str(module_path)}}
        (out / "bootstrap_stats.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("eval_lawstress_dspy_module.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        mode = _arg(cmd, "--mode")
        if mode == "score_only":
            payload = {
                "metrics": {
                    "overall": {"c1_pass_rate": 61.0, "c2_pass_rate": 61.0, "c3_pass_rate": 61.0},
                    "success": {"overall_pass": True},
                },
                "groups": {},
            }
            (out / "optimized_eval_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    elif script.endswith("eval_manifesto_teacher_trace_local_laws.py"):
        out = Path(_arg(cmd, "--output-dir"))
        out.mkdir(parents=True, exist_ok=True)
        mode = _arg(cmd, "--mode")
        if mode == "summarize_only":
            pred = Path(_arg(cmd, "--predictions-path"))
            _write_jsonl(pred, 4)
        else:
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
                    "c1_pass_rate": 56.0,
                    "c2_pass_rate": 56.0,
                    "c3_pass_rate": 56.0,
                    "avg_law_pass_rate": 56.0,
                }
            (out / "eval_metrics.json").write_text(json.dumps({"overall": overall}), encoding="utf-8")

    return {
        "command": [str(part) for part in cmd],
        "log_path": str(log),
        "returncode": 0,
        "dry_run": False,
        "started_at": "2026-03-03T00:00:00+00:00",
        "finished_at": "2026-03-03T00:00:01+00:00",
    }


def test_bootstrap_manual_dry_run_smoke(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_manual")
    output_dir = tmp_path / "bootstrap_manual_dry"
    rc = cli.main(["--output-dir", str(output_dir), "--dry-run"])
    assert rc == 0
    manifest_path = output_dir / "bootstrap_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] in {"completed", "failed_gates"}
    assert manifest["config"]["judge_backend"] == "large_qwen"
    assert manifest["config"]["tournament_backend"] == "disabled"
    assert "stage_a_lawstress" in manifest["phases"]
    assert "stage_b_teacher_traces" in manifest["phases"]
    assert "stage_f_score_only" in manifest["phases"]


def test_bootstrap_manual_mocked_run_writes_complete_manifest(tmp_path: Path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_manual")
    monkeypatch.setattr(cli, "_run_command", _fake_run_command)
    monkeypatch.setattr(cli, "_is_server_alive", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli, "_wait_for_server", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        cli,
        "_get_model_ids",
        lambda *_args, **_kwargs: [cli.DEFAULT_TEACHER_MODEL, cli.DEFAULT_STUDENT_MODEL, cli.DEFAULT_EMBEDDING_MODEL],
    )

    output_dir = tmp_path / "bootstrap_manual_mocked"
    rc = cli.main(["--output-dir", str(output_dir), "--no-dynamic-mode"])
    assert rc == 0

    manifest = json.loads((output_dir / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["gates"]["overall_pass"] is True
    assert manifest["gates"]["post_training"]["lawstress_success_overall_pass"] is True
    assert manifest["gates"]["post_training"]["real_anchor_local_law_improvement"]["pass"] is True
    assert manifest["artifacts"]["backend_provenance"]["judge_backend"] == "large_qwen"
    assert manifest["artifacts"]["backend_provenance"]["tournament_backend"] == "disabled"


def test_bootstrap_manual_rejects_genrm_mode(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.run_manifesto_local_law_bootstrap_manual")
    rc = cli.main(["--output-dir", str(tmp_path / "reject_genrm"), "--no-disable-genrm"])
    assert rc == 2
