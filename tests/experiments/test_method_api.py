from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.experiments import (
    ARTIFACT_BEST_CHECKPOINT_PATH,
    ExperimentContext,
    ExperimentMethodSpec,
    ROLE_SCORER,
    SamplingPlan,
    benchmark_ref_from_parts,
    chat_role_ref,
    experiment_method_ref,
    normalize_method_output,
    oracle_ref,
    prefixed_artifact_key,
)


def _refs() -> tuple[object, object]:
    benchmark_ref = benchmark_ref_from_parts(
        family="fixture",
        name="fixture benchmark",
        dataset_id="fixture-data",
    )
    method_ref = experiment_method_ref(
        family="fixture_method",
        variant="v1",
        adapter="test",
        roles={
            ROLE_SCORER: chat_role_ref(
                role=ROLE_SCORER,
                model="fixture-scorer",
                base_url="http://localhost:8000/v1",
            )
        },
        oracle=oracle_ref(kind="benchmark_labels", source="fixture"),
    )
    return benchmark_ref, method_ref


def _context(tmp_path: Path, *, sampling: SamplingPlan | None = None) -> ExperimentContext:
    benchmark_ref, method_ref = _refs()
    return ExperimentContext(
        output_root=tmp_path,
        benchmark_ref=benchmark_ref,
        method_ref=method_ref,
        title="fixture train",
        adapter_id="test_context",
        phases=("train", "evaluate", "predict"),
        sampling=sampling or SamplingPlan(seed=7, split="validation", strategy="fixture"),
        metadata={"suite": "method_api"},
        launch_command=("python", "train.py"),
    )


@dataclass
class _DataclassTrainResult:
    train_count: int
    val_count: int
    output_dir: str
    metadata: dict[str, object] = field(default_factory=dict)
    trained_artifact: object | None = None
    student_model_class: str = "embedding_proxy"


def test_sampling_plan_defaults_and_mapping_normalization() -> None:
    assert SamplingPlan().to_dict() == {
        "seed": None,
        "split": "",
        "strategy": "",
        "sample_budget": None,
        "sampling_probability": None,
        "unit": "",
        "frame": "",
        "metadata": {},
    }

    sampling = SamplingPlan.from_value(
        {
            "seed": "11",
            "split": "test",
            "strategy": "pps",
            "sample_budget": "5",
            "sampling_probability": "0.25",
            "unit": "document",
            "frame": "manifesto",
            "metadata": {"source": "fixture"},
        }
    )
    assert sampling.seed == 11
    assert sampling.sample_budget == 5
    assert sampling.sampling_probability == 0.25
    assert sampling.metadata == {"source": "fixture"}


def test_context_record_writes_sampling_sidecars_and_rows(tmp_path: Path) -> None:
    context = _context(tmp_path)
    artifact_key = prefixed_artifact_key("ctreepo", ARTIFACT_BEST_CHECKPOINT_PATH)

    run = context.record(
        {
            "metrics": {"loss": 0.25, "nested": {"accuracy": 0.75}},
            "artifacts": {artifact_key: str(tmp_path / "model.pt")},
            "metadata": {"scale": 2},
            "model": object(),
            "train_count": 6,
        },
        phase="train",
    )

    assert run.normalized.metrics["loss"] == 0.25
    assert run.normalized.metrics["nested.accuracy"] == 0.75
    assert run.normalized.metrics["train_count"] == 6
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "experiment_status.json").exists()
    assert (tmp_path / "artifacts.json").exists()

    status = json.loads((tmp_path / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["metadata"]["sampling"]["seed"] == 7
    assert status["metadata"]["sampling"]["split"] == "validation"

    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["metric_name"] for row in rows} >= {"loss", "nested.accuracy", "train_count"}
    assert rows[0]["seed"] == 7
    assert rows[0]["split"] == "validation"
    assert rows[0]["method_ref"]["metadata"]["roles"]["scorer"]["model"] == "fixture-scorer"
    assert rows[0]["method_ref"]["metadata"]["oracle"]["kind"] == "benchmark_labels"
    assert rows[0]["benchmark_ref"]["problem_id"] == "fixture"
    assert "family" not in rows[0]["benchmark_ref"]


def test_context_record_normalizes_existing_dataclass_result(tmp_path: Path) -> None:
    output_dir = tmp_path / "trained"
    result = _DataclassTrainResult(
        train_count=10,
        val_count=4,
        output_dir=str(output_dir),
        metadata={"dry_run": True},
        trained_artifact=object(),
    )

    normalized = normalize_method_output(result)
    assert normalized.metrics == {"train_count": 10, "val_count": 4}
    assert normalized.artifacts == {"output_dir": str(output_dir)}
    assert normalized.metadata["student_model_class"] == "embedding_proxy"
    assert normalized.metadata["dry_run"] is True

    run = _context(tmp_path).record(result, state="dry_run")
    status = json.loads((tmp_path / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "dry_run"
    assert run.experiment_spec.method_refs[0].metadata["roles"]["scorer"]["model"]


class _TrainOnlyMethod:
    def __init__(self, spec: ExperimentMethodSpec) -> None:
        self._spec = spec
        self.seen_context = False
        self.seen_config = {}
        self.validation_size = 0

    def method_spec(self) -> ExperimentMethodSpec:
        return self._spec

    def train(self, train_data, validation_data=None, *, context, config=None):
        self.seen_context = context is not None
        self.seen_config = dict(config or {})
        self.validation_size = len(validation_data or [])
        return {
            "metrics": {"train_count": len(train_data), "validation_count": self.validation_size},
            "metadata": {"method": "train_only"},
        }


def test_context_train_uses_context_identity_and_sampling(tmp_path: Path) -> None:
    foreign_benchmark, foreign_method = _refs()
    foreign_spec = ExperimentMethodSpec(
        benchmark_ref=foreign_benchmark,
        method_ref=foreign_method,
        title="foreign",
        phases=("train",),
        metadata={"suite": "foreign"},
    )
    method = _TrainOnlyMethod(foreign_spec)
    context = _context(tmp_path, sampling=SamplingPlan(seed=3, split="train", strategy="uniform"))

    run = context.train(
        method,
        [1, 2, 3],
        validation_data=[4],
        config={"lr": 0.1},
    )

    assert method.seen_context is True
    assert method.seen_config == {"lr": 0.1}
    assert run.normalized.metrics["train_count"] == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["seed"] for row in rows} == {3}
    assert {row["split"] for row in rows} == {"train"}
    assert rows[0]["method_ref"]["method_id"] == "fixture_method"
    assert "family" not in rows[0]["method_ref"]


class _TorchModule:
    def __init__(self) -> None:
        self.mode_calls: list[str] = []

    def train(self, mode: bool = True):
        self.mode_calls.append(f"train:{mode}")
        return self

    def eval(self):
        self.mode_calls.append("eval")
        return self

    def state_dict(self):
        return {"weight": object()}


class _TorchLikeMethod:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.module = _TorchModule()

    def train(self, data, *, context=None):
        assert context is not None
        self.module.train()
        return {
            "metrics": {"loss": 0.1},
            "artifacts": {"checkpoint_path": str(self.checkpoint)},
            "metadata": {"state_dict_keys": list(self.module.state_dict().keys())},
            "model": self,
        }

    def evaluate(self, data, *, split="test", context=None):
        assert context is not None
        self.module.eval()
        return {"metrics": {f"{split}_mae": 0.2}}

    def predict(self, inputs, *, context=None):
        assert context is not None
        self.module.eval()
        return {"metrics": {"n_predictions": len(inputs)}, "metadata": {"kind": "torch_like"}}


def test_torch_like_method_owns_modes_and_artifacts(tmp_path: Path) -> None:
    method = _TorchLikeMethod(tmp_path / "best.pt")
    context = _context(tmp_path, sampling=SamplingPlan(seed=5, split="validation"))

    train_run = context.train(method, [1, 2])
    eval_run = context.evaluate(method, [3])
    pred_run = context.predict(method, [4, 5, 6])

    assert method.module.mode_calls == ["train:True", "eval", "eval"]
    assert train_run.normalized.artifacts["checkpoint_path"].endswith("best.pt")
    assert "model" not in train_run.normalized.metadata
    assert eval_run.normalized.metrics["validation_mae"] == 0.2
    assert pred_run.normalized.metrics["n_predictions"] == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["phase"] for row in rows} == {"train", "evaluate", "predict"}
    assert {row["seed"] for row in rows} == {5}


class _DspyLikeMethod:
    def __init__(self, artifact_path: Path) -> None:
        self.artifact_path = artifact_path

    def train(self, trainset, devset=None, *, context=None):
        assert context is not None
        return {
            "metrics": {"compile_score": 0.8},
            "artifacts": {"compiled_program": str(self.artifact_path)},
            "metadata": context.call_metadata(
                role="summarizer",
                request_kind="dspy_compile",
                runner_id="dspy_gepa",
            ),
        }

    def evaluate(self, examples, *, context=None, split="test"):
        assert context is not None
        return {
            "metrics": {"accuracy": 1.0},
            "metadata": context.call_metadata(role="scorer", request_kind="score", runner_id="dspy_eval"),
        }


def test_dspy_vllm_like_method_preserves_role_metadata(tmp_path: Path) -> None:
    benchmark_ref, _ = _refs()
    method_ref = experiment_method_ref(
        family="summary_tree",
        variant="dspy_gepa",
        adapter="dspy",
        roles={
            "scorer": chat_role_ref(role="scorer", model="vllm-scorer"),
            "summarizer": chat_role_ref(role="summarizer", model="vllm-summarizer"),
        },
        oracle=oracle_ref(kind="teacher", model="teacher"),
    )
    context = ExperimentContext(
        output_root=tmp_path,
        benchmark_ref=benchmark_ref,
        method_ref=method_ref,
        title="dspy train",
        phases=("train", "evaluate"),
        sampling=SamplingPlan(seed=9, split="test", strategy="fixture"),
    )
    method = _DspyLikeMethod(tmp_path / "compiled.json")

    train_run = context.train(method, ["train"], method_kwargs={"devset": ["dev"]})
    eval_run = context.evaluate(method, ["test"])

    roles = train_run.experiment_spec.method_refs[0].metadata["roles"]
    assert roles["scorer"]["model"] == "vllm-scorer"
    assert roles["summarizer"]["model"] == "vllm-summarizer"
    assert train_run.normalized.artifacts["compiled_program"].endswith("compiled.json")
    assert eval_run.normalized.metadata["role"] == "scorer"
    assert eval_run.normalized.metadata["sampling"]["seed"] == 9


class _SklearnLikeProxy:
    def __init__(self) -> None:
        self.sample_weight = None
        self.mean = 0.0

    def fit(self, inputs, targets, sample_weight=None):
        self.sample_weight = sample_weight
        self.mean = sum(targets) / len(targets)
        return {"metrics": {"fit_count": len(inputs), "target_mean": self.mean}}

    def predict(self, inputs):
        return {
            "metrics": {"n_predictions": len(inputs)},
            "metadata": {"predictions": [self.mean for _ in inputs]},
        }


class _SklearnProxyMethod:
    def __init__(self, proxy: _SklearnLikeProxy) -> None:
        self.proxy = proxy

    def train(self, inputs, targets, sample_weight=None):
        return self.proxy.fit(inputs, targets, sample_weight=sample_weight)

    def predict(self, inputs):
        return self.proxy.predict(inputs)


def test_sklearn_like_proxy_adapter_train_predict_with_context(tmp_path: Path) -> None:
    proxy = _SklearnLikeProxy()
    method = _SklearnProxyMethod(proxy)
    context = _context(tmp_path)

    train_run = context.train(
        method,
        [[1], [2]],
        [0.0, 1.0],
        method_kwargs={"sample_weight": [1.0, 2.0]},
    )
    pred_run = context.predict(method, [[3], [4]])

    assert proxy.sample_weight == [1.0, 2.0]
    assert train_run.normalized.metrics["fit_count"] == 2
    assert pred_run.normalized.metadata["predictions"] == [0.5, 0.5]


def test_context_train_does_not_dispatch_to_raw_fit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not support phase 'train'"):
        _context(tmp_path).train(_SklearnLikeProxy(), [[1]], [1.0])


class _PytorchModeOnlyModule:
    def train(self, mode: bool = True):
        return self


def test_context_train_rejects_raw_pytorch_mode_toggle(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="PyTorch mode toggle"):
        _context(tmp_path).train(_PytorchModeOnlyModule())


class _EvaluateOnlyMethod:
    def evaluate(self, data, *, split="test"):
        return {"metrics": {f"{split}_score": len(data)}}


def test_context_evaluate_only_method_runs_without_train(tmp_path: Path) -> None:
    run = _context(tmp_path, sampling=SamplingPlan(split="test")).evaluate(
        _EvaluateOnlyMethod(),
        [1, 2],
    )
    assert run.normalized.metrics["test_score"] == 2


class _BrokenMethod:
    def train(self, data):
        raise ValueError("method boom")


def test_context_train_records_failure_status(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="method boom"):
        _context(tmp_path).train(_BrokenMethod(), [1], metadata={"attempt": 2})

    status = json.loads((tmp_path / "experiment_status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["metadata"]["error"]["type"] == "ValueError"
    assert status["metadata"]["attempt"] == 2
    assert status["metadata"]["sampling"]["seed"] == 7
    assert (tmp_path / "results.jsonl").read_text(encoding="utf-8") == ""


def test_old_helper_names_are_not_public_exports() -> None:
    experiments = importlib.import_module("src.experiments")
    method_api = importlib.import_module("src.experiments.method_api")
    for name in (
        "RunContext",
        "run_method_phase",
        "train_method",
        "evaluate_method",
        "predict_method",
        "fit_trainer",
        "run_trainer_phase",
        "evaluate_trainer",
        "predict_trainer",
        "fit_with_experiment",
        "run_with_experiment",
        "record_method_output",
        "SupportsFit",
        "SupportsArtifacts",
        "SupportsEvaluate",
        "SupportsPredict",
        "SupportsTrain",
        "ExperimentTrainer",
        "METHOD_ADAPTERS",
        "REPORT_PROFILES",
        "RuntimeEvalAdapter",
        "RuntimeUmbrellaScriptAdapter",
        "method_ref_from_markov_full_doc_run",
        "result_rows_from_scalar_metrics",
    ):
        assert not hasattr(experiments, name)
        if name not in {
            "METHOD_ADAPTERS",
            "REPORT_PROFILES",
            "RuntimeEvalAdapter",
            "RuntimeUmbrellaScriptAdapter",
            "method_ref_from_markov_full_doc_run",
            "result_rows_from_scalar_metrics",
        }:
            assert not hasattr(method_api, name)

    public_exports = set(getattr(experiments, "__all__", ()))
    assert {"ExperimentContext", "SamplingPlan", "benchmark_ref_from_parts"} <= public_exports
    assert "METHOD_ADAPTERS" not in public_exports
    assert "SupportsTrain" not in public_exports
