# Experiment Method API

This is the canonical path for adding a new experiment, training routine, or
method lifecycle to the repo.

The public vocabulary is intentionally small:

1. A method owns learning and inference.
2. `ExperimentContext` owns identity, sampling, roles, metrics, artifacts, and status.
3. Canonical sidecars make every experiment discoverable without changing existing
   script outputs.

## Core Objects

- `SamplingPlan`: the sampling design for the experiment. It records `seed`,
  `split`, `strategy`, `sample_budget`, `sampling_probability`, `unit`, `frame`,
  and free-form `metadata`.
- `BenchmarkRef`: what data/task/cell the method ran on.
- `MethodRef`: what method ran. Public role metadata lives under
  `method_ref.metadata.roles`; oracle provenance lives under
  `method_ref.metadata.oracle`.
- `ExperimentContext`: the only public lifecycle object. It exposes `train`,
  `evaluate`, `predict`, `record`, and `call_metadata`.
- `NormalizedMethodOutput`: the serializable view of a train/evaluate/predict result:
  `metrics`, `artifacts`, and `metadata`.
- `ExperimentMethod`: optional class/protocol style for methods with a natural
  lifecycle: `train`, `evaluate`, `predict`, and artifact export.

The trained Python object, estimator, or neural module should stay on the raw
return value. It is not serialized into sidecars.

## Public Roles

Use paper-facing roles in method metadata:

- `scorer`: practical task scorer `f`.
- `summarizer`: compression/summarization map `g`.
- `embedder`: embedding model used for retrieval or proxy features.
- `state_model`: deterministic or learned state realization machinery.
- `oracle`: trusted target/evaluator `f*`, usually benchmark labels or a teacher.

Internal surfaces such as `chat_openai`, `embedding`, and `operator` are
implementation metadata, not user-facing experiment vocabulary.

## Recommended Method Contract

A new experiment method should expose `train(...)` for learned methods and
`evaluate(...)` / `predict(...)` for evaluation-only methods. Return either a
dataclass/object or mapping with these fields when possible:

```python
{
    "metrics": {"val_loss": 0.12, "val_accuracy": 0.83},
    "artifacts": {"checkpoint_path": "outputs/experiment/model.pt"},
    "metadata": {"train_docs": 100, "optimizer": "bootstrap_random_search"},
    "model": trained_model,  # kept in Python, not serialized
}
```

Existing dataclass results are also supported. Numeric top-level dataclass
fields become metrics; fields such as `output_dir`, `model_path`, and
`checkpoint_path` become artifacts; `metadata` is preserved.

If a backend is sklearn-shaped, keep `fit(...)` inside a thin adapter whose
public experiment phase is `train(...)`.

## Wrapping A Method

```python
from src.experiments import (
    ExperimentContext,
    SamplingPlan,
    benchmark_ref_from_parts,
    chat_role_ref,
    experiment_method_ref,
    oracle_ref,
)

benchmark_ref = benchmark_ref_from_parts(
    family="manifesto_rile",
    name="RILE smoke",
    dataset_id="manifesto",
)
method_ref = experiment_method_ref(
    family="summary_tree",
    variant="bootstrap",
    adapter="my_training_script",
    roles={
        "scorer": chat_role_ref(role="scorer", model="qwen3.5-397b"),
        "summarizer": chat_role_ref(role="summarizer", model="nemotron-30b"),
    },
    oracle=oracle_ref(kind="teacher", model="qwen3.5-397b"),
)

context = ExperimentContext(
    output_root="outputs/my_experiment",
    benchmark_ref=benchmark_ref,
    method_ref=method_ref,
    title="summary tree train",
    adapter_id="treepo_training",
    phases=("train", "evaluate"),
    sampling=SamplingPlan(
        seed=0,
        split="train",
        strategy="task_split",
        sample_budget=100,
        unit="document",
        frame="manifesto_train",
    ),
)

train_result = context.train(method, train_data, validation_data=val_data)
eval_result = context.with_sampling({"seed": 0, "split": "test"}).evaluate(method, test_data)
prediction_result = context.predict(method, inputs)
context.record({"metrics": {"accuracy": 0.83}}, phase="evaluate")
```

This writes:

- `experiment_manifest.json`
- `experiment_status.json`
- `artifacts.json`
- `results.jsonl`

Function-shaped code should call the function directly, then call
`context.record(...)` with the returned payload. Object-shaped code should use
`context.train(...)`, `context.evaluate(...)`, and `context.predict(...)`.

Use `context.call_metadata(...)` when handing model calls to a batched client or
callstream scheduler.

## PyTorch And VLLM

PyTorch-like methods should keep native idioms inside the method: call
`module.train()` within the method's `train(...)`, call `module.eval()` within
`evaluate(...)`/`predict(...)`, and record checkpoints as artifact paths rather
than serialized tensors. The experiment wrapper does not infer mode switches.
Passing a raw `nn.Module` as the experiment method is intentionally rejected
when its `train(...)` looks like the PyTorch mode toggle.

vLLM-backed methods usually implement `evaluate`/`predict`; optimizer-driven
wrappers such as DSPy prompt or program optimization implement `train` against
scorer/summarizer roles.

## Artifact Keys

Prefer the shared artifact constants from `src.experiments.artifacts`:

- `summary_json`
- `metrics_json`
- `predictions_jsonl`
- `calls_jsonl`
- `steps_jsonl`
- `final_stats_json`
- `training_result_json`
- `reproducibility_manifest_json`
- `checkpoint_path`
- `best_checkpoint_path`
- `final_checkpoint_path`
- `output_dir`

Method-specific artifacts should be namespaced with
`prefixed_artifact_key("ctreepo", "training_result_json")` rather than
inventing new suffix conventions.

## Implementation Rules

- Do not create a new result schema for each method. Convert to
  `metrics/artifacts/metadata`.
- Do not copy full prompts, contexts, trained objects, tensors, or datasets into
  sidecars. Store compact paths and provenance.
- Keep existing CLIs stable. Add canonical sidecars beside old outputs.
- Add a method adapter to `scripts/run_experiment.py` only when the entrypoint
  needs planning, launching, resume, or collection support.
- If a method emits active model traffic, route calls through the runtime
  callstream or a call trace sink so `calls.jsonl` can be registered.

## Supported Entrypoints

The supported/legacy boundary lives in
`config/runtime_umbrella_entrypoints.yaml`.

- `supported`: first-class scripts that should emit canonical sidecars or be
  covered by the experiment control plane.
- `adapter_covered`: families handled by a central adapter without making every
  historical script a first-class API.
- `legacy_globs`: old simulations, reports, demos, and maintenance scripts that
  remain useful but are not public workflow entrypoints.

Before release, run:

```bash
python scripts/audit_runtime_umbrella_coverage.py --fail-on-unclassified
python scripts/check_repo_release_hygiene.py --json
```

## Relationship To Runtime Eval

Runtime eval is the online benchmark path: method runners call scorer,
summarizer, embedder, and state_model roles through the runtime callstream.
LongBench v2 methods use the same roles:

- `full_context`: scorer predicts from the full context prompt.
- `retrieval`: embedder selects evidence, scorer predicts.
- `summary_tree`: summarizer builds tree summaries, scorer predicts.
- `state_tree`: summarizer/state representation path, scorer predicts.
- `neural_operator`: state_model selects or renders evidence, scorer predicts.

The method API is the offline method path: `train()`, training scripts, Markov
grids, and post-hoc reports use the same role metadata and sidecar contract.

Both paths produce comparable `MethodRef` and `ResultRow` records.
