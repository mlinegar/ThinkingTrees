# Generate-First TreePO Stack (StateTree)

This repo now has a single, unified way to run TreePO across:

- **Exact / symbolic** operators (e.g. the Markov toy lane)
- **Text `/generate`** engines (preferred; e.g. SGLang serving `POST /generate`)
- **Chat** engines (fallback implementation detail)

The canonical path is:

1. `build_treepo_stack(model_spec, contract_spec) -> TreePOStack`
2. `TreePOStack.run_fixed_binary(leaf_spans, ...) -> FixedBinaryStateTreeRunResult`

Legacy `TreeBuilder` is now implemented on top of the same fixed-binary `StateTree`
runner internally (and converts back to `Tree` only at the end), so there’s a
single place where fixed-binary execution lives.

The result contains:

- a `StateTree` (nodes carry `span`, arbitrary internal `state`, and a `rendered` string),
- an operation trace (`StateTreeOperationTrace[]`),
- verifier outputs stored on each `StateNode.audit["law_checks"]...`.

## Core API

- `src/tree/treepo_stack.py`
  - `TreePOModelSpec` (how to run **g**: summarize/merge/resummary)
  - `TreePOContractSpec` (rubric + local-law config + oracle lane)
  - `build_treepo_stack(...)`
- `src/tree/state_tree_runner.py`
  - `run_fixed_binary_state_tree(...)` (canonical runner)
- `src/tree/state_tree.py`
  - `StateNode`, `StateTree` (`StateTree.to_dict()` is JSON-safe; tensors are structural only)

## What changes between Markov vs LLM vs diffusion?

Almost nothing in the calling code:

- You still call `TreePOStack.run_fixed_binary(...)`.
- You still get back a `StateTree` + trace + verifier outputs.

What *does* change is just the `model_spec` and the contract’s oracle lane:

- **Markov** uses `kind="markov_toy_exact"` and the built-in exact verifier lane.
- **Text** lanes require an **oracle lane** (provided oracle or a trained proxy) so the local-law auditor can score/check.
- **Diffusion vs AR** differences live in `engine_options` passed through to `/generate` (prompt templates stay the same by default).

If you explicitly disable all local-law checks (`enable_l1/enable_l2/enable_l3/enable_substitution=False`),
you can build/run a text stack **without** an oracle lane (useful for pure “infrastructure” runs or for offline labeling).

## Quickstart: Markov (exact, no servers)

```python
from src.tree import (
    OracleLaneSpec,
    TreePOContractSpec,
    TreePOModelSpec,
    build_treepo_stack,
)

stack = build_treepo_stack(
    TreePOModelSpec(kind="markov_toy_exact"),
    TreePOContractSpec(
        rubric="(unused for Markov toy)",
        oracle_lane=OracleLaneSpec(kind="markov_exact"),
    ),
)

# Markov spans are lists of tokens (Span=list[str]).
leaf_spans = [["a", "b"], ["b"], ["c", "c"]]
result = stack.run_fixed_binary(leaf_spans)

print("root rendered:", result.tree.final_rendered)
print("root checks:", result.tree.root.audit.get("law_checks"))
```

## Quickstart: `/generate` with SGLang (preferred surface)

Start an SGLang server (example; use your preferred profile/port):

```bash
./scripts/start_sglang.sh nemotron-30b-nvfp4 --port 30000
```

Then run TreePO over `POST /generate`:

```python
from src.tree import (
    OracleLaneSpec,
    TreePOContractSpec,
    TreePOModelSpec,
    build_treepo_stack,
)

stack = build_treepo_stack(
    TreePOModelSpec(
        # engine defaults to "auto" (inferred from base_url when provided).
        model="default",
        # base_url may be:
        # - "http://host:port" (preferred for /generate),
        # - "http://host:port/generate" (will be normalized), or
        # - "http://host:port/v1" (will be normalized for generate-first when possible).
        # it is normalized to avoid "/generate/generate".
        base_url="http://localhost:30000",
        surface="generate",
    ),
    TreePOContractSpec(
        rubric="Preserve named entities, numbers, and causal claims.",
        oracle_lane=OracleLaneSpec(
            kind="provided_scoring_oracle",
            import_path="src.tasks.manifesto.oracle:create_rile_oracle",
            kwargs={},
        ),
    ),
)

leaf_spans = [
    "Alice paid Bob $10 on Tuesday.",
    "Later, Bob refunded Alice $5.",
    "Alice says the refund was incomplete.",
]

result = stack.run_fixed_binary(
    leaf_spans,
    refine_rounds=1,
    sampling_params={"max_tokens": 128, "temperature": 0.2},
    # Diffusion-vs-AR behavior is controlled here (engine-specific).
    engine_options={"dllm_algorithm": "none"},
)

print(result.tree.final_rendered)
print(result.tree.metadata.get("treepo_stack"))
```

### Chat fallback behavior

If you request `surface="generate"` on an engine that does **not** expose `/generate` (e.g. `engine="vllm"`),
`build_treepo_stack` will fall back to chat internally and record:

- `tree.metadata["treepo_stack"]["surface_requested"] == "generate"`
- `tree.metadata["treepo_stack"]["surface"] == "chat_openai"`
- `tree.metadata["treepo_stack"]["surface_fallback_reason"] == "engine_missing_generate_surface"`

## Oracle lanes (text verification)

Text local-law checking is done by `TextAuditorAdapterVerifier`, which runs the
legacy-auditor semantics **directly on `StateTree[str, str]`** (no `StateTree → Tree`
conversion) and stores results nodewise into `StateNode.audit["law_checks"]...`.

### Lane: provided scoring oracle

Use this when you already have a `ScoringOracle` factory in Python.

```python
OracleLaneSpec(
    kind="provided_scoring_oracle",
    import_path="some.module:make_oracle",
    kwargs={"base_url": "http://localhost:8000/v1", "model": "default"},
)
```

### Lane: embedding proxy (train from labeled CSV/JSONL)

If you provide `contract_spec.supervision_source`, the stack will:

1. build and save a canonical `SupervisionDataset` JSON under `outputs/treepo_stack/`,
2. train an embedding-ridge proxy (by default),
3. save the proxy artifact under `outputs/treepo_stack/`,
4. use the trained proxy as the scoring oracle.

```python
from src.tree import SupervisionSourceSpec, OracleLaneSpec, TreePOContractSpec

contract = TreePOContractSpec(
    rubric="Preserve named entities and numbers.",
    supervision_source=SupervisionSourceSpec(
        kind="csv",
        path="data/my_labels.csv",
        text_column="text",
        label_column="label",
        response_signal_min=0.0,
        response_signal_max=1.0,
    ),
    oracle_lane=OracleLaneSpec(
        kind="embedding_proxy",
        embedding_base_url="http://localhost:8003/v1",
        embedding_model="default",
        ridge_lambda=1.0,
        proxy_model_id="my_proxy_v1",
    ),
)
```

## Supervision (scores + preferences)

The canonical container for supervision in this repo is:

- `src/training/supervision/types.py:SupervisionDataset`

It supports:

- **Scalar labels** via `ResponseJudgment.response_signal_value` (and optional vectors via `response_signal_vector`)
- **Comparative / preference supervision** via `comparative_judgments` (groupwise rankings) and binary projections

### Supplying arbitrary scalar scores

You can provide any float labels you want (on a declared scale) by loading/building a `SupervisionDataset`.
The stack’s `SupervisionSourceSpec(kind="csv"|"jsonl")` is a convenience builder for the common “text + scalar label” case.

If you already have a richer supervision artifact, point the stack at it directly:

```python
SupervisionSourceSpec(kind="supervision_dataset_json", path="outputs/my_dataset.json")
```

### Supplying choices / preferences

Preferences are represented as comparative judgments and can be projected into optimizer-friendly formats:

- `SupervisionDataset.project_binary(...)` → pairwise comparisons (DPO/reward training)
- `SupervisionDataset.to_group_grpo_records(...)` → groupwise GRPO format

`build_treepo_stack(...)` does not require preferences, but the new “generate-first” stack is compatible with
the same training supervision surface; you can use the same dataset types regardless of whether `g` is Markov, chat, or `/generate`.

### Collecting comparative supervision from a StateTree run

`TreePOStack.run_fixed_binary(...)` can emit either:

- scalar `response_judgments` (default), or
- groupwise `comparative_judgments` (two candidates per sampled unit).

Use `TreePOSupervisionSpec(supervision_kind="comparative", ...)` to collect comparative records.
In `"requests"` mode, candidates are stored but unranked; you can label them later.

### Automatic oracle score judgments (default)

If a stack is built with an oracle lane (e.g. `provided_scoring_oracle` or `embedding_proxy`), then
`TreePOStack.run_fixed_binary(...)` will automatically emit a tiny supervision dataset by default:

- 1 labeled example per document (the **root** node),
- saved under `outputs/treepo_supervision_auto/`,
- recorded in `result.tree.metadata["treepo_supervision"]`.

Disable it explicitly with:

```python
from src.tree import TreePOSupervisionSpec

result = stack.run_fixed_binary(
    leaf_spans,
    supervision=TreePOSupervisionSpec(mode="off"),
)
```

### Non-blocking label requests (sample now, label later)

If you want to run trees **without any online oracle calls**, but still collect occasional supervision for later
labeling, pass a `TreePOSupervisionSpec` to `TreePOStack.run_fixed_binary(...)`.

Mode `"requests"` emits a `SupervisionDataset` JSON with `response_signal_value=None` that you can label later.

```python
from src.tree import TreePOSupervisionSpec

result = stack.run_fixed_binary(
    leaf_spans,
    document_id="doc_001",
    supervision=TreePOSupervisionSpec(
        mode="requests",
        doc_sample_probability=0.05,  # ~5% of docs
        unit_selector="all",
        max_units=16,
        sampling_strategy="random",  # or "level_weighted" / "content_weighted"
        unit_sampling_probability=1.0,  # optional second-stage gate (like Auditor.sampling_probability); alias: sample_prob
        random_seed=0,
        output_dir="outputs/treepo_supervision_requests",
    ),
)
print(result.tree.metadata.get("treepo_supervision"))
```

DSL-friendly aliases are accepted when passing dict specs:

- `sample_prob` / `sampling_probability` → `unit_sampling_probability`
- `doc_sample_prob` → `doc_sample_probability`

For the offline labeling step, you can also use a policy spec (DSL-friendly):

- `TreePOLabelingPolicySpec(max_labels=..., label_probability=..., random_seed=...)`
- aliases: `sample_prob` / `sampling_probability` → `label_probability`

### Markov node labels (exact, offline)

For the Markov toy lane, you can label nodes *exactly* (no online oracle) by using:

- `TreePOSupervisionSpec(mode="label_now", labeler_kind="markov_toy_changepoints", ...)`

```python
result = stack.run_fixed_binary(
    leaf_spans,  # list[list[str]]
    document_id="markov_doc_001",
    supervision=TreePOSupervisionSpec(
        mode="label_now",
        labeler_kind="markov_toy_changepoints",
        doc_sample_probability=1.0,
        unit_selector="root",
        max_units=1,
        output_dir="outputs/treepo_markov_node_labels",
        response_signal_min=0.0,
        response_signal_max=100.0,
    ),
)
```

To label the emitted dataset later with an oracle (blocking in a separate job), use:

```bash
./venv/bin/python scripts/label_treepo_supervision_dataset.py \
  --input outputs/treepo_supervision_requests/supervision_*.json \
  --oracle-import-path src.tree.auditor:SimpleScorer \
  --max-labels 200 \
  --random-seed 0
```

`--max-labels` (SRSWOR) and `--label-probability` (Bernoulli) both record/maintain
`label_propensity` so IPW weights remain well-defined for partial labeling.

### StateTree node range + relationship labels

`run_fixed_binary_state_tree(...)` populates minimal structural labels on each `StateNode.metadata`:

- `leaf_index` (leaves only)
- `leaf_start_index`, `leaf_end_index` (all nodes; leaf-range coverage)
- `range_label` (e.g., `"0:15"`)
- `parent_id` (all non-root nodes)
- `child_side` (`"root"`, `"left"`, `"right"`)

For long-running / non-blocking labeling, launch it detached:

```bash
./venv/bin/python scripts/long_job.py launch \
  --name treepo_label_requests \
  --job-root outputs/treepo_label_requests_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  -- ./venv/bin/python scripts/label_treepo_supervision_dataset.py \
       --input outputs/treepo_supervision_requests/supervision_*.json \
       --oracle-import-path src.tree.auditor:SimpleScorer \
       --max-labels 200 \
       --random-seed 0
```

## Where verifier outputs live

All verifier outputs are attached to the `StateTree` nodes:

- `StateNode.audit["law_checks"][verifier_name][law_kind] = CheckResult.to_dict()`

This is true for Markov exact checks and for text local-law auditing.

## Recommended usage for new code

- Prefer `build_treepo_stack(...)` + `TreePOStack.run_fixed_binary(...)`.
- Prefer `/generate` (SGLang) as the primary surface; chat is supported as a fallback.
- Treat legacy fixed-binary diffusion engines as deprecated wrappers.
