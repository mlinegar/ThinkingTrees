from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_microgrid_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run_phase3_gepa_microgrid.py"
    spec = importlib.util.spec_from_file_location("run_phase3_gepa_microgrid_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arg(cmd: list[str], name: str) -> str | None:
    if name not in cmd:
        return None
    idx = cmd.index(name)
    return cmd[idx + 1]


def _write_report(
    output_dir: Path,
    *,
    baseline_dev: float | None = None,
    optimized_dev: float | None = None,
    final_test: float = 0.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "final_test": {
            "pearson_r": final_test,
            "pearson_defined": True,
            "prediction_path": str(output_dir / "per_mfesto_final_test.jsonl"),
        },
        "final_artifacts": {
            "program": str(output_dir / "final_program.json"),
            "scorer": str(output_dir / "scorer_final.json"),
            "g": str(output_dir / "unified_g_final.json"),
        },
    }
    if baseline_dev is not None:
        report["baseline_dev"] = {
            "pearson_r": baseline_dev,
            "pearson_defined": True,
            "prediction_path": str(output_dir / "per_mfesto_baseline_dev.jsonl"),
        }
    if optimized_dev is not None:
        report["optimized_dev"] = {
            "pearson_r": optimized_dev,
            "pearson_defined": True,
            "prediction_path": str(output_dir / "per_mfesto_optimized_dev.jsonl"),
        }
        report["optimized_artifacts"] = {
            "program": str(output_dir / "optimized_program.json"),
            "scorer": str(output_dir / "optimized_scorer.json"),
            "g": str(output_dir / "optimized_unified_g.json"),
        }
    (output_dir / "report.json").write_text(json.dumps(report))


def test_fgf_dry_run_passes_previous_optimized_stage_to_next_stage(tmp_path, monkeypatch):
    module = _load_microgrid_module()
    out = tmp_path / "microgrid"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase3_gepa_microgrid.py",
            "--dry-run",
            "--skip-baseline",
            "--stage-plan",
            "fgf",
            "--output-root",
            str(out),
        ],
    )

    assert module.main() == 0

    manifest = json.loads((out / "microgrid_manifest.json").read_text())
    runs = manifest["runs"]
    assert [run["stage_component"] for run in runs] == ["f", "g", "f"]
    assert _arg(runs[0]["command"], "--optimize-scope") == "f"
    assert _arg(runs[0]["command"], "--init-dir") is None
    assert _arg(runs[1]["command"], "--optimize-scope") == "g"
    assert _arg(runs[1]["command"], "--init-dir") == runs[0]["output_dir"]
    assert _arg(runs[1]["command"], "--init-artifact-kind") == "optimized"
    assert "--init-components-only" in runs[1]["command"]
    assert _arg(runs[2]["command"], "--optimize-scope") == "f"
    assert _arg(runs[2]["command"], "--init-dir") == runs[1]["output_dir"]
    assert _arg(runs[2]["command"], "--init-artifact-kind") == "optimized"
    assert "--init-components-only" in runs[2]["command"]
    assert manifest["args"]["stage_plan"] == "fgf"
    assert manifest["args"]["stage_plan_codename"] == "joint"
    assert manifest["final_stage"]["stage"] == "stage3_f"
    assert manifest["final_stage"]["stage_plan_codename"] == "joint"


def test_partial_initial_f_warm_start_is_only_used_for_first_stage(tmp_path, monkeypatch):
    module = _load_microgrid_module()
    out = tmp_path / "microgrid"
    scorer = tmp_path / "optimized_scorer.json"
    scorer.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase3_gepa_microgrid.py",
            "--dry-run",
            "--skip-baseline",
            "--stage-plan",
            "fg",
            "--init-scorer",
            str(scorer),
            "--output-root",
            str(out),
        ],
    )

    assert module.main() == 0

    manifest = json.loads((out / "microgrid_manifest.json").read_text())
    stage1, stage2 = manifest["runs"]
    assert _arg(stage1["command"], "--init-scorer") == str(scorer)
    assert _arg(stage1["command"], "--init-dir") is None
    assert _arg(stage2["command"], "--init-scorer") is None
    assert _arg(stage2["command"], "--init-dir") == stage1["output_dir"]
    assert _arg(stage2["command"], "--init-artifact-kind") == "optimized"
    assert "--init-components-only" in stage2["command"]


def test_report_both_manifest_records_final_and_dev_best_with_paths(tmp_path):
    module = _load_microgrid_module()
    base = tmp_path / "baseline"
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    _write_report(base, baseline_dev=0.80, final_test=0.10)
    _write_report(stage1, baseline_dev=0.80, optimized_dev=0.60, final_test=0.20)
    _write_report(stage2, baseline_dev=0.60, optimized_dev=0.70, final_test=0.30)
    manifest = {
        "runs": [
            {
                "stage": "baseline",
                "condition_key": "c16000",
                "output_dir": str(base),
                "status": "completed",
                "return_code": 0,
            },
            {
                "stage": "stage1_f",
                "stage_index": 1,
                "stage_component": "f",
                "stage_plan": "fg",
                "stage_plan_codename": "joint",
                "condition_key": "c16000",
                "output_dir": str(stage1),
                "status": "completed",
                "return_code": 0,
            },
            {
                "stage": "stage2_g",
                "stage_index": 2,
                "stage_component": "g",
                "stage_plan": "fg",
                "stage_plan_codename": "joint",
                "condition_key": "c16000",
                "output_dir": str(stage2),
                "status": "completed",
                "return_code": 0,
            },
        ],
    }

    module._refresh_stage_manifest(manifest)

    assert manifest["final_stage"]["stage"] == "stage2_g"
    assert manifest["final_stage"]["stage_plan_codename"] == "joint"
    assert manifest["final_stage"]["artifacts"]["program"].endswith("optimized_program.json")
    assert manifest["dev_best_stage"]["stage"] == "baseline"
    assert manifest["dev_best_stage"]["prediction_paths"]["dev"].endswith(
        "per_mfesto_baseline_dev.jsonl"
    )
    assert len(manifest["condition_summaries"]["c16000"]["stage_history"]) == 3
