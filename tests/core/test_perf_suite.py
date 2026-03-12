"""Tests for scenario-driven performance suite runner."""

from pathlib import Path

from src.benchmark.perf_suite import (
    compare_suite_results,
    load_suite_config,
    render_comparison_markdown,
    run_performance_suite,
    save_suite_results,
)


def test_load_suite_config_infers_defaults(tmp_path: Path):
    cfg_path = tmp_path / "suite.yaml"
    cfg_path.write_text("cases:\n  - id: c1\n    command: ['echo', 'ok']\n", encoding="utf-8")

    cfg = load_suite_config(cfg_path)
    assert cfg["version"] == 1
    assert cfg["name"] == "suite"
    assert cfg["output_root"] == "outputs/performance_suite"
    assert isinstance(cfg["cases"], list)
    assert cfg["cases"][0]["id"] == "c1"


def test_run_performance_suite_dry_run_with_filters(tmp_path: Path):
    cfg = {
        "name": "unit_suite",
        "defaults": {"repeats": 1, "cwd": "."},
        "cases": [
            {
                "id": "micro_case",
                "layer": "micro",
                "enabled": True,
                "command": ["python3", "-c", "print('micro')"],
            },
            {
                "id": "component_case",
                "layer": "component",
                "enabled": False,
                "command": ["python3", "-c", "print('component')"],
            },
        ],
    }

    payload = run_performance_suite(
        cfg,
        output_dir=tmp_path / "run",
        include_layers=["micro"],
        include_disabled=False,
        dry_run=True,
    )

    assert payload["selected_case_ids"] == ["micro_case"]
    assert payload["summary"]["cases_total"] == 1
    assert payload["summary"]["cases_dry_run"] == 1
    assert payload["summary"]["cases_failed"] == 0


def test_run_performance_suite_executes_and_extracts_json(tmp_path: Path):
    cfg = {
        "name": "unit_exec",
        "defaults": {"repeats": 1, "cwd": "."},
        "cases": [
            {
                "id": "json_case",
                "layer": "micro",
                "command": [
                    "python3",
                    "-c",
                    (
                        "import json,sys;"
                        "json.dump(dict(value=3.5, ok=True),open(sys.argv[1],'w',encoding='utf-8'))"
                    ),
                    "{case_dir}/artifact.json",
                ],
                "extractors": [
                    {"type": "json_file", "name": "artifact", "path": "{case_dir}/artifact.json"}
                ],
            }
        ],
    }

    payload = run_performance_suite(cfg, output_dir=tmp_path / "run")
    assert payload["summary"]["cases_failed"] == 0

    case = payload["results"][0]
    rep = case["repeats"][0]
    assert rep["status"] == "ok"
    assert rep["extracts"]["artifact"]["value"] == 3.5
    assert rep["extracts"]["artifact"]["ok"] is True


def test_save_suite_results_writes_json_and_markdown(tmp_path: Path):
    payload = {
        "suite_name": "save_test",
        "generated_at": "2026-02-26T00:00:00+00:00",
        "completed_at": "2026-02-26T00:00:01+00:00",
        "duration_seconds": 1.0,
        "run_dir": str(tmp_path / "run"),
        "results": [
            {
                "id": "case_1",
                "layer": "micro",
                "status": "ok",
                "repeats": [{"status": "ok"}],
                "successful_repeats": 1,
                "failed_repeats": 0,
            }
        ],
    }
    outputs = save_suite_results(payload)
    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    assert "case_1" in outputs["markdown"].read_text(encoding="utf-8")


def test_compare_suite_results_detects_regressions():
    cfg = {
        "cases": [
            {
                "id": "metric_case",
                "metric_rules": [
                    {
                        "name": "higher_better_metric",
                        "path": "extracts.metrics.throughput",
                        "direction": "higher",
                        "max_regression_pct": 5.0,
                    },
                    {
                        "name": "lower_better_metric",
                        "path": "extracts.metrics.latency_ms",
                        "direction": "lower",
                        "max_regression_abs": 10.0,
                    },
                ],
            }
        ]
    }
    baseline = {
        "results": [
            {
                "id": "metric_case",
                "repeats": [
                    {"extracts": {"metrics": {"throughput": 100.0, "latency_ms": 60.0}}},
                    {"extracts": {"metrics": {"throughput": 110.0, "latency_ms": 65.0}}},
                ],
            }
        ]
    }
    candidate = {
        "results": [
            {
                "id": "metric_case",
                "repeats": [
                    {"extracts": {"metrics": {"throughput": 80.0, "latency_ms": 80.0}}},
                    {"extracts": {"metrics": {"throughput": 85.0, "latency_ms": 82.0}}},
                ],
            }
        ]
    }

    comparison = compare_suite_results(cfg, baseline, candidate)
    assert comparison["summary"]["checks_total"] == 2
    assert comparison["summary"]["checks_regression"] == 2

    markdown = render_comparison_markdown(comparison)
    assert "metric_case" in markdown
    assert "regression" in markdown


def test_pipeline_run_extractor_includes_transition_stats(tmp_path: Path):
    cfg = {
        "name": "pipeline_extract",
        "defaults": {"repeats": 1, "cwd": "."},
        "cases": [
            {
                "id": "pipeline_case",
                "layer": "integration",
                "command": [
                    "python3",
                    "-c",
                    (
                        "import json,os,sys;"
                        "d=sys.argv[1];"
                        "os.makedirs(d,exist_ok=True);"
                        "json.dump(dict("
                        "success=True,"
                        "started_at='2026-02-26T00:00:00',"
                        "completed_at='2026-02-26T00:00:10',"
                        "config=dict(task='manifesto_rile',train_samples=1,val_samples=1,test_samples=1),"
                        "train=dict(mae=0.1),"
                        "test=dict(mae=0.2,prediction_distribution=dict(frac_neutral=0.1,n_unique_rounded_4dp=2)),"
                        "conditional_memory=dict(mode='readwrite',hit_rate=0.25,l1_hits=2,l2_hits=3,misses=5,writes=7)"
                        "),open(os.path.join(d,'final_stats.json'),'w',encoding='utf-8'));"
                        "open(os.path.join(d,'run.log'),'w',encoding='utf-8').write("
                        "'Transitioned to genrm mode in 9.5s\\n'"
                        "'Transitioned to task_dp2 mode in 0.6s\\n'"
                        "'Cascading progress: x rate=1.2 items/s y tokens=1,234 z tok/s=77\\n'"
                        ")"
                    ),
                    "{case_dir}/pipeline_run",
                ],
                "extractors": [
                    {"type": "pipeline_run", "name": "pipeline", "path": "{case_dir}/pipeline_run"}
                ],
            }
        ],
    }

    payload = run_performance_suite(cfg, output_dir=tmp_path / "run")
    rep = payload["results"][0]["repeats"][0]
    summary = rep["extracts"]["pipeline"]["summary"]
    stats = summary["gpu_transition_stats"]
    assert stats["count"] == 2
    assert abs(float(stats["max_seconds"]) - 9.5) < 1e-9
    assert abs(float(stats["mean_seconds"]) - 5.05) < 1e-9
    assert abs(float(stats["p95_seconds"]) - 9.055) < 1e-9
    assert summary["throughput"]["peak_tok_per_sec"] == 77.0
