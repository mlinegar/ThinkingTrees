# Runtime Backbone Spec and LLM Handoff

Date: 2026-02-17
Repository: ThinkingTrees / C-TreePO / SemanticForests

## Purpose

This document formalizes the missing "runtime backbone" layer between:

1. Backbone LLMs (model endpoints and token limits)
2. Problem definitions (benchmarks and tasks)
3. Execution loop (planning, retrieval, generation, verification, repair, commit)

Goal: enable another LLM agent to implement this layer without re-deriving architecture.

## Problem Statement

Current repo strengths:

1. Strong tree abstractions and auditing (`src/tree/*`)
2. Training and preference optimization stack (`src/training/*`)
3. Context-window-aware model plumbing (`src/config/context_window.py`, `src/core/model_detection.py`)
4. Throughput benchmark utilities (`src/benchmark/throughput.py`)
5. Task plugin architecture (`src/tasks/*`)

Current gap:

1. No explicit runtime contract for "agentic long-context behavior from bounded-context calls"
2. No unified interface that binds benchmark instance -> tree state -> per-step execution -> verifier outcomes
3. No standard trace schema for reproducible benchmark runs and ablations

## Scope and Non-Goals

In scope:

1. Runtime layer contracts and modules
2. Control loop/state machine
3. Benchmark adapter interface for RULER, LongBench, ContextBench
4. Logging and metrics sufficient for ablations
5. First vertical slice runnable on one benchmark

Out of scope for first implementation:

1. New model training methods
2. New theorem work
3. Full optimization of all benchmarks at once
4. Large redesign of existing tree math APIs

## Target Architecture

Four-layer stack:

1. Model layer: frozen LLM endpoints + context limits
2. Runtime layer: tree controller + memory packer + verifier + repair policy
3. Task/execution layer: benchmark adapters and optional tool executor
4. Evaluation layer: metrics + trace artifacts + ablation harness

Proposed module layout:

```text
src/runtime/
  backbone.py          # BackboneAdapter protocol + implementations
  contracts.py         # ProblemSpec, NodeContract, RuntimeConfig, StepResult
  state.py             # RuntimeState, node store, provenance refs, frontier
  memory.py            # Context packing and budget-aware prompt assembly
  planner.py           # Frontier/node selection policy
  verifier.py          # Deterministic + model-based checks
  repair.py            # Retry, re-split, retrieve-more policies
  executor.py          # Optional tool actions (for code/external tasks)
  loop.py              # Main runtime control loop
  trace.py             # Structured logging schema and writers
  adapters/
    base.py            # BenchmarkAdapter protocol
    ruler.py           # RULER adapter (first recommended)
    longbench.py       # LongBench adapter
    contextbench.py    # ContextBench adapter
scripts/
  run_runtime_eval.py  # CLI entrypoint for runtime benchmark runs
tests/runtime/
  test_contracts.py
  test_memory.py
  test_loop.py
  test_ruler_adapter.py
```

## Formal Contracts

### 1) ProblemSpec

Defines one benchmark/problem instance independent of model choice.

Required fields:

1. `problem_id: str`
2. `input_text: str` or `input_artifacts: dict[str, Any]`
3. `query: str`
4. `success_metric: str`
5. `success_target: float | dict[str, float]`
6. `constraints: dict[str, Any]`
7. `allowed_actions: list[str]`
8. `metadata: dict[str, Any]`

### 2) NodeContract

Defines what each tree node must preserve/produce.

Required fields:

1. `objective: str` (what this node should achieve)
2. `must_preserve: list[str]` (facts or invariants)
3. `output_schema: dict[str, str]` (expected fields)
4. `acceptance_checks: list[str]` (check names)
5. `max_input_tokens: int`
6. `max_output_tokens: int`

### 3) BackboneAdapter

Model-facing API used by runtime loop.

Required methods:

1. `generate(messages, *, max_tokens, temperature, stop) -> ModelResponse`
2. `score_pair(context, a, b, rubric) -> float | PreferenceLabel`
3. `embed(texts) -> list[list[float]]` (optional but standardized)
4. `max_context() -> int`
5. `model_id() -> str`

Implementation note:

1. Wrap existing vLLM/DSPy integrations rather than replacing them.

### 4) Verifier

Two-stage check strategy:

1. Deterministic checks first: schema, parseability, forbidden patterns, budget compliance
2. Model-based checks second: preservation/consistency scoring

Return type:

1. `VerifierResult { pass: bool, score: float, failures: list[str], evidence: dict }`

### 5) RuntimeState

Persistent state for the control loop.

Required data:

1. `root_id`, `frontier_ids`, `node_graph`
2. `provenance_map` (node -> source spans/evidence IDs)
3. `token_accounting` (per step and cumulative)
4. `budget_state` (remaining step/time/token budgets)
5. `status` (running/success/failed/budget_exhausted)

## Runtime Loop (Reference Semantics)

Single-step semantics:

1. Select next frontier node with planner
2. Build bounded prompt from node state and selected evidence
3. Call backbone model for candidate output/action
4. Verify candidate against node contract
5. On pass: commit node, update parent/merge state
6. On fail: invoke repair policy (retry, re-split, retrieve more, fallback)
7. Emit structured trace event
8. Stop on success predicate or exhausted budget

Reference pseudocode:

```python
while not state.done():
    node_id = planner.select_frontier(state)
    packed = memory.pack(state, node_id, config)
    candidate = backbone.generate(packed.messages, max_tokens=packed.max_output_tokens)
    v = verifier.check(state, node_id, candidate)
    if v.pass_:
        state.commit(node_id, candidate, v)
        state.try_merge_upwards(node_id)
    else:
        repair.apply(state, node_id, candidate, v)
    trace.log_step(state, node_id, packed, candidate, v)
return state.final_result()
```

## Invariants (Must Hold)

1. Per-call context safety: `input_tokens + output_tokens + safety <= model_max_context`
2. Every committed node has provenance references
3. Every failed check is trace-recorded with reason code
4. Every benchmark run emits reproducible config + seed + model metadata
5. Runtime can execute with verifier disabled for ablation mode

## Benchmark Adapter Contract

Each adapter must implement:

1. `load_split(split: str, limit: int | None) -> Iterable[ProblemSpec]`
2. `build_contract(problem: ProblemSpec) -> NodeContract`
3. `score(problem: ProblemSpec, runtime_output: dict) -> dict[str, float]`
4. `primary_metric() -> str`
5. `supports_tools() -> bool`

Adapter mapping for first stage:

1. `RULER`: first priority, controlled length and retrieval stress
2. `LongBench`: second priority, realistic long-doc tasks
3. `ContextBench`: third priority, process-level context retrieval for coding agents

## Metrics and Logging Schema

Per-run metrics:

1. Primary benchmark score
2. Score by length bucket
3. Max per-call input tokens
4. Total input/output tokens
5. Steps per problem
6. Verifier pass rate
7. Repair rate by reason code
8. Wall-clock latency

Required artifact files:

1. `config.json` (full runtime and model settings)
2. `metrics.json` (aggregates)
3. `steps.jsonl` (one event per step)
4. `predictions.jsonl` (problem-level outputs + scores)

Minimum `steps.jsonl` fields:

1. `run_id`
2. `problem_id`
3. `step_idx`
4. `node_id`
5. `action_type`
6. `input_tokens`
7. `output_tokens`
8. `verifier_pass`
9. `failure_codes`
10. `repair_action`
11. `latency_ms`
12. `timestamp_utc`

## Ablation Matrix (Required for Claims)

Must-run comparisons:

1. `flat_prompt_baseline`: no tree runtime, single-shot or naive chunk+concat
2. `runtime_no_verifier`: tree runtime, verifier disabled
3. `runtime_no_repair`: verifier enabled, no recovery actions
4. `runtime_full`: verifier + repair + bounded memory packing

For each condition report:

1. Primary score
2. Score vs length
3. Token/latency cost
4. Failure mode breakdown

## Implementation Plan

Phase 0: bootstrap contracts and no-op loop

1. Create `src/runtime/contracts.py`, `src/runtime/state.py`, `src/runtime/loop.py`
2. Add unit tests for schema/state transitions
3. Add `scripts/run_runtime_eval.py` CLI skeleton

Exit criteria:

1. Dry-run with synthetic `ProblemSpec` succeeds and writes artifacts

Phase 1: RULER vertical slice

1. Implement `src/runtime/adapters/ruler.py`
2. Implement minimal `memory.py`, `planner.py`, `verifier.py`, `repair.py`
3. Integrate existing LLM client via `backbone.py`
4. Run small split and generate metrics artifacts

Exit criteria:

1. End-to-end benchmark run completes with reproducible traces

Phase 2: robustness and ablations

1. Implement ablation flags in CLI
2. Add failure reason taxonomy
3. Add score-vs-length reporting

Exit criteria:

1. Full ablation table produced from one benchmark

Phase 3: adapter expansion

1. Add `longbench.py` adapter
2. Add `contextbench.py` adapter
3. Keep runtime loop unchanged; only adapter/config changes

Exit criteria:

1. Same runtime executable on all three benchmarks

## Acceptance Criteria (Definition of Done)

Engineering DoD:

1. Runtime package has typed contracts and tests
2. CLI supports seed, budget, ablation flags, and output directory
3. Artifacts are sufficient to replay and audit runs
4. Existing training pipeline remains unaffected

Scientific DoD:

1. At least one small model evaluated with strict per-call context cap
2. Tree runtime outperforms flat baseline on length scaling
3. Claim is backed by ablations, not only best-run numbers

## Risks and Mitigations

Risk: runtime adds complexity but no quality gains.
Mitigation: enforce flat baseline and no-verifier ablations early.

Risk: verifier false positives/negatives distort results.
Mitigation: keep deterministic checks explicit and log all failure codes.

Risk: benchmark adapters diverge and fork runtime behavior.
Mitigation: enforce adapter-only differences; keep loop and contracts common.

Risk: token budgeting bugs invalidate long-context claim.
Mitigation: hard context invariant checks and token accounting in every step event.

## Handoff Instructions for Next LLM

Use this order:

1. Implement Phase 0 exactly as defined
2. Submit PR with contracts, state machine, CLI skeleton, tests
3. Implement Phase 1 RULER adapter and minimal runtime components
4. Submit PR with end-to-end artifacts and one ablation (`runtime_full` vs `flat_prompt_baseline`)
5. Expand to remaining ablations, then LongBench and ContextBench adapters

Required implementation discipline:

1. No silent fallback behavior
2. No benchmark-specific logic in core loop
3. Every recovery action must emit a trace event
4. Keep interfaces stable and typed before optimizing performance

## Suggested First Command Sequence (Current CLI)

```bash
source venv/bin/activate
pytest -q tests/runtime/test_runtime_eval.py

# Initialize a run (expands phases -> units.jsonl under one run_id).
python3 scripts/run_runtime_eval.py init \
  --config config/runtime_eval/ruler_8k_freeform.yaml \
  --output-dir outputs/evals \
  --run-id ruler8k_freeform_$(date +%Y%m%d_%H%M%S)

# Run a small smoke subset (no server required).
python3 scripts/run_runtime_eval.py run \
  --run-dir outputs/evals/<RUN_ID> \
  --phase-id S0_smoke \
  --mode runtime_full \
  --mock-llm \
  --max-units 2 \
  --max-problems 5

# Aggregate per-unit artifacts into run-level files.
python3 scripts/run_runtime_eval.py aggregate --run-dir outputs/evals/<RUN_ID>

# For cluster/job arrays, shard deterministically:
#   --shard-index 0 --shard-count 64 --skip-done
```

To run against a real local vLLM server, start it first and then drop `--mock-llm`:

```bash
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --enforce-eager
```

## Open Design Decisions (Resolve in PR 1)

1. Whether `Verifier` should consume raw text spans or only packed prompt context
2. Whether `Planner` is deterministic by default or stochastic with seed
3. Whether repair retries are bounded per-node or global-budget only
4. Whether to support tool execution in Phase 1 or defer to Phase 3

## Copy-Paste Prompt for Next LLM

```text
You are implementing the runtime backbone for ThinkingTrees/C-TreePO/SemanticForests.

Read and follow this spec exactly:
docs/runtime_backbone_handoff_spec.md

Constraints:
1) Preserve existing training pipeline behavior.
2) Do not add benchmark-specific logic into core runtime loop.
3) Implement in phases; complete Phase 0 first.
4) Add tests for contracts and state transitions before benchmark adapters.

Deliverables for this pass:
1) Code changes for current phase
2) Test updates
3) Brief architecture note listing interfaces actually implemented
4) Example run command and expected output artifact paths
5) Explicit list of unresolved open design decisions

Quality bar:
1) Typed interfaces and deterministic behavior under fixed seed
2) Structured JSON/JSONL artifacts for replay and audit
3) No silent fallbacks
```
