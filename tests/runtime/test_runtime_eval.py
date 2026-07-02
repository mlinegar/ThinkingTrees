from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.runtime.adapters.ruler import _string_match_all, _string_match_part
from src.runtime.contracts import RunPhaseSpec, RunSpec, expand_units
from src.runtime.loop import _extract_context_and_question, _run_problem
from src.runtime.contracts import ModelResponse, NodeContract, ProblemSpec, RunUnit, RuntimeConfig
from src.runtime.memory import TokenCounter, chunk_text_tokens
from src.runtime.repair import SimpleRepairPolicy
from src.runtime.verifier import DeterministicVerifier
from scripts.run_runtime_eval import _load_run_spec


def test_expand_units_runtime_grid_cartesian_product():
    spec = RunSpec(
        run_id="r1",
        created_utc="now",
        output_dir="outputs/evals",
        benchmark={"name": "ruler_synthetic"},
        runtime_defaults={},
        phases=[
            RunPhaseSpec(
                phase_id="P0",
                tasks=["t1", "t2"],
                lengths=[10],
                seeds=[0, 1],
                num_samples=5,
                split="validation",
                methods=["m1", "m2"],
                runtime_grid={"leaf_memory_tokens": [64, 128], "merge_memory_tokens": [256]},
            )
        ],
    )

    units = expand_units(spec)
    # tasks(2) * lengths(1) * seeds(2) * methods(2) * grid(2*1)=16
    assert len(units) == 16
    assert any(u.runtime_overrides.get("leaf_memory_tokens") == 64 for u in units)
    assert any(u.runtime_overrides.get("leaf_memory_tokens") == 128 for u in units)


def test_load_run_spec_accepts_top_level_and_phase_methods(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: longbench_v2",
                "methods: [full_context, retrieval]",
                "scorer:",
                "  endpoint: http://localhost:8000/v1",
                "embedder:",
                "  mock: true",
                "phases:",
                "  - phase_id: P0",
                "    tasks: [all]",
                "    lengths: [4096]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: test",
                "  - phase_id: P1",
                "    tasks: [all]",
                "    lengths: [4096]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: test",
                "    methods: [summary_tree]",
            ]
        ),
        encoding="utf-8",
    )

    spec = _load_run_spec(config_path, output_dir=tmp_path / "outputs", experiment_id="r")

    assert spec.phases[0].methods == ["full_context", "retrieval"]
    assert spec.phases[1].methods == ["summary_tree"]
    assert spec.roles["scorer"]["base_url"] == "http://localhost:8000/v1"
    assert spec.roles["summarizer"]["defaulted_from"] == "scorer"
    assert spec.surfaces["chat_openai"]["base_url"] == "http://localhost:8000/v1"


def test_load_run_spec_rejects_unsupported_public_config_keys(tmp_path: Path) -> None:
    for key in ("answerer", "run", "run_id", "unexpected_key"):
        config_path = tmp_path / f"{key}.yaml"
        if key == "run":
            body = ["run:", "  name: old", "benchmark:", "  name: longbench_v2", "phases: []"]
        elif key == "run_id":
            body = ["run_id: old", "benchmark:", "  name: longbench_v2", "phases: []"]
        elif key == "unexpected_key":
            body = ["unexpected_key: true", "benchmark:", "  name: longbench_v2", "phases: []"]
        else:
            body = [
                "benchmark:",
                "  name: longbench_v2",
                f"{key}:",
                "  endpoint: http://localhost:8000/v1",
                "phases: []",
            ]
        config_path.write_text("\n".join(body), encoding="utf-8")

        try:
            _load_run_spec(config_path, output_dir=tmp_path / "outputs", experiment_id="r")
        except ValueError as exc:
            assert f"Unsupported runtime-eval config key: {key}" in str(exc)
            assert "Supported runtime-eval config keys" in str(exc)
        else:
            raise AssertionError(f"Expected runtime config key to be rejected: {key}")


def test_load_run_spec_rejects_unsupported_phase_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: longbench_v2",
                "methods: [full_context]",
                "scorer:",
                "  endpoint: http://localhost:8000/v1",
                "phases:",
                "  - phase_id: P0",
                "    tasks: [all]",
                "    lengths: [4096]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: test",
                "    modes: [full_context]",
            ]
        ),
        encoding="utf-8",
    )

    try:
        _load_run_spec(config_path, output_dir=tmp_path / "outputs", experiment_id="r")
    except ValueError as exc:
        assert "Unsupported runtime-eval phase key in 'P0': modes" in str(exc)
        assert "Supported phase keys" in str(exc)
    else:
        raise AssertionError("Expected unsupported phase key to be rejected")


def test_ruler_string_match_all():
    preds = ["A B C", "foo bar"]
    refs = [["a", "b"], ["bar"]]
    assert _string_match_all(preds, refs) == 100.0


def test_ruler_string_match_part():
    preds = ["the answer is Paris", "missing"]
    refs = [["paris"], ["rome", "athens"]]
    assert _string_match_part(preds, refs) == 50.0


def test_extract_context_and_question_niah():
    p = ProblemSpec(
        problem_id="p1",
        input_text="Intro...\nCONTEXT\nWhat are all the things?",
        metadata={"ruler_task_type": "niah"},
    )
    ctx, q = _extract_context_and_question(p)
    assert "CONTEXT" in ctx
    assert q.startswith("What ")


def test_runtime_full_uses_leaf_and_merge_budgets_for_max_tokens():
    class DummyAdapter:
        def build_contract(self, problem: ProblemSpec) -> NodeContract:
            return NodeContract(objective="dummy", max_input_tokens=8192, max_output_tokens=9999)

    class SpyBackbone:
        def __init__(self):
            self.max_tokens_calls: list[int] = []

        def generate(self, messages, *, max_tokens: int, **kwargs) -> ModelResponse:  # type: ignore[no-untyped-def]
            self.max_tokens_calls.append(int(max_tokens))
            return ModelResponse(
                text="ok", model_id="spy", prompt_tokens=0, completion_tokens=0, latency_ms=0.0
            )

    counter = TokenCounter()
    verifier = DeterministicVerifier(counter)
    repair = SimpleRepairPolicy()

    runtime = RuntimeConfig(
        method="runtime_full",
        cap_tokens=512,
        safety_tokens=0,
        max_output_tokens=30,
        chunk_tokens=10,
        overlap_tokens=0,
        leaf_memory_tokens=10,
        merge_memory_tokens=20,
        verifier_enabled=True,
        repair_enabled=True,
    )

    unit = RunUnit(
        run_id="r1",
        unit_id="u000001",
        phase_id="P0",
        benchmark="b",
        task_id="t",
        split="validation",
        max_seq_length=10,
        seed=0,
        num_samples=1,
        method="runtime_full",
    )

    problem = ProblemSpec(
        problem_id="p1", input_text=("x " * 100).strip(), query="Q?", references=["ref"]
    )
    chunks = chunk_text_tokens(
        problem.input_text,
        counter=counter,
        chunk_tokens=runtime.chunk_tokens,
        overlap_tokens=runtime.overlap_tokens,
    )
    assert len(chunks) >= 2

    backbone = SpyBackbone()
    pred, cost, events = _run_problem(
        problem=problem,
        adapter=DummyAdapter(),
        unit=unit,
        runtime=runtime,
        backbone=backbone,  # type: ignore[arg-type]
        counter=counter,
        verifier=verifier,
        repair=repair,
    )

    assert pred == "ok"
    assert cost.get("n_calls", 0) == len(backbone.max_tokens_calls)
    assert backbone.max_tokens_calls[-1] == runtime.max_output_tokens  # answer step
    assert backbone.max_tokens_calls[: len(chunks)] == [runtime.leaf_memory_tokens] * len(chunks)
    assert backbone.max_tokens_calls[len(chunks) : -1] == [runtime.merge_memory_tokens] * (
        len(chunks) - 1
    )


def test_runtime_eval_init_and_aggregate_emit_canonical_control_plane(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = tmp_path / "runtime.yaml"
    output_root = tmp_path / "outputs"
    experiment_id = "runtime_smoke"
    config_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: ruler_synthetic",
                "  family: runtime_benchmark",
                "scorer:",
                "  endpoint: http://localhost:8000/v1",
                "  model: demo-model",
                "  engine: vllm",
                "runtime_defaults: {}",
                "phases:",
                "  - phase_id: P0",
                "    tasks: [vt]",
                "    lengths: [1024]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: validation",
                "    methods: [runtime_full]",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_runtime_eval.py",
            "init",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_root),
            "--experiment-id",
            experiment_id,
        ],
        cwd=repo_root,
    )
    run_dir = output_root / experiment_id
    assert (run_dir / "experiment_manifest.json").exists()
    assert (run_dir / "experiment_status.json").exists()
    unit_dir = run_dir / "units" / "u000001"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "metrics_partial.json").write_text(
        json.dumps({"primary_metric": "score", "mean_score": 0.5}),
        encoding="utf-8",
    )
    (unit_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "unit_id": "u000001",
                "phase_id": "P0",
                "benchmark": "ruler_synthetic",
                "task_id": "vt",
                "split": "validation",
                "max_seq_length": 1024,
                "seed": 0,
                "method": "runtime_full",
                "primary_metric": "score",
                "problem_id": "p1",
                "prediction": "ok",
                "references": ["ok"],
                "metrics": {"score": 1.0},
                "cost": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (unit_dir / "steps.jsonl").write_text(
        json.dumps({"event": "step"}) + "\n",
        encoding="utf-8",
    )
    (unit_dir / "calls.jsonl").write_text(
        json.dumps(
            {
                "unit_id": "u000001",
                "method_id": "runtime_full",
                "runner_id": "llm_tree_memory",
                "surface": "chat_openai",
                "usage": {"prompt_tokens": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_runtime_eval.py",
            "aggregate",
            "--experiment-dir",
            str(run_dir),
        ],
        cwd=repo_root,
    )
    assert (run_dir / "artifacts.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "calls.jsonl").exists()
    status = json.loads((run_dir / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    assert "metrics_json" in artifacts["artifacts"]
    assert "merged_calls_jsonl" in artifacts["artifacts"]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_surface_calls"] == 1
    results_lines = [
        line
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert results_lines
    result_payload = json.loads(results_lines[0])
    assert result_payload["method_ref"]["method_id"] == "runtime_full"
    assert result_payload["method_ref"]["model"] == ""
    assert "scorer" in result_payload["method_ref"]["metadata"]["roles"]


def test_runtime_eval_plan_command_reports_methods_and_surfaces(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = tmp_path / "lb.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "_id": "x",
                "domain": "law",
                "sub_domain": "contracts",
                "difficulty": "easy",
                "length": "short",
                "question": "Which option is named?",
                "choice_A": "Alpha",
                "choice_B": "Beta",
                "choice_C": "Gamma",
                "choice_D": "Delta",
                "answer": "A",
                "context": "Alpha is named.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: longbench_v2",
                f"  dataset_path: {fixture}",
                "methods: [full_context, retrieval]",
                "scorer:",
                "  endpoint: http://localhost:8000/v1",
                "  model: mock-model",
                "embedder:",
                "  mock: true",
                "phases:",
                "  - phase_id: P0",
                "    tasks: [all]",
                "    lengths: [4096]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: test",
            ]
        ),
        encoding="utf-8",
    )

    raw = subprocess.check_output(
        [
            sys.executable,
            "scripts/run_runtime_eval.py",
            "plan",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--experiment-id",
            "plan_smoke",
            "--json",
        ],
        cwd=repo_root,
        text=True,
    )
    payload = json.loads(raw)

    assert payload["experiment_id"] == "plan_smoke"
    assert payload["unit_count"] == 2
    assert {method["method_id"] for method in payload["methods"]} == {
        "full_context",
        "retrieval",
    }
    assert "scorer" in payload["roles"]
    assert "summarizer" in payload["roles"]
    assert payload["roles"]["summarizer"]["config"]["defaulted_from"] == "scorer"
    assert "embedder" in payload["roles"]
    assert payload["oracle"]["kind"] == "benchmark_labels"
    assert payload["missing_roles"] == {}
