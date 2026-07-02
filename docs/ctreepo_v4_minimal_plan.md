# C-TreePO Minimal v4 Plan

Date: 2026-04-13

Related notes:

- `docs/ctreepo_v4_minimal_audit.md`
- `docs/treepo_generate_first_stack.md`
- existing wide-scope configs that are **not** the authority for this plan:
  - `config/markov/tradeoff_pipeline.long_v4.toml`
  - `config/markov/publication_bundle.long_v4.toml`

## Summary

This plan defines the smallest v4 that is still scientifically useful and
actually pointed at the real target workload.

The v4 package will be:

- single-path
- supervision-first
- contract-first
- report-light
- exact-reference-backed

The central design choice is:

- v4 is **not** a Markov-only package with an optional LLM path added later
- v4 is one shared TreePO execution and supervision stack
- the Markov lane is the cheapest exact reference benchmark for correctness
- the text/LLM lane is the real downstream workload the package must serve

Minimal v4 still ships exactly three canonical scientific experiment families:

1. exact reference parity canary
2. controlled multileaf local-law ladder
3. fixed-budget supervision frontier

But those families are built on top of one shared stack:

1. `build_treepo_stack(model_spec, contract_spec) -> TreePOStack`
2. `TreePOStack.run_fixed_binary(leaf_spans, ...) -> FixedBinaryStateTreeRunResult`
3. `emit_supervision_dataset(tree_run_result, supervision_spec) -> SupervisionDataset`

## Locked decisions

The following decisions are fixed for the minimal v4 plan.

### Scope

- v4 core is the unified fixed-binary TreePO stack.
- The same public path must support:
  - an exact/symbolic lane
  - a text/LLM lane
- The exact Markov lane is a reference benchmark and contract lock, not the
  package boundary.
- The text lane should be generate-first. Chat is a fallback implementation
  detail, not a separate product surface.
- v4 does not include LDA, publication bundles, rolling reports, or
  appendix-style report surfaces in the core package.

### Primary API

- The primary public API is the shared stack plus the supervision emitter.
- Experiment wrappers exist, but they are secondary.
- A user should be able to:
  - build a tree stack once
  - run exact or text tasks through the same runner
  - emit canonical supervision rows for PyTorch without importing the legacy v3
    launch/report stack

### Shared-path doctrine

- There is one operator interface.
- There is one `StateTree` representation.
- There is one fixed-binary runner.
- There is one law-verifier surface.
- There is one supervision dataset schema.
- There is one artifact contract per run.
- Engine-specific branching is allowed only inside `TreePOModelSpec`,
  `OracleLaneSpec`, and the operator/oracle adapters.

### Reporting

- The canonical output per run is:
  - `run_manifest.json`
  - `rows.jsonl`
  - `summary.json`
  - `report.md`
- `supervision_dataset.json` is emitted when supervision collection is enabled.
- PDF generation is optional and out of core scope.
- There is no `latest`, `current`, or prefix-based selection logic in the v4
  core.

### Identity

- Every run has one immutable `run_id`.
- Every report surface has one immutable `surface_id`.
- Reports read provenance; they do not rewrite provenance.
- Lane metadata is explicit:
  - `lane_kind`
  - `operator_kind`
  - `engine_surface`
  - `oracle_lane_kind`

### Namespace

- The new implementation lives under `src/ctreepo/v4/`.
- v4 may continue to depend on shared repo-wide infrastructure in `src/core/*`
  and `src/training/supervision/*` where those modules are already the correct
  shared abstraction.
- v4 must not depend on the v3 simulation suite, rolling report, or launcher
  glue.

## Deliverables

Minimal v4 is complete when the repo contains:

### D1. A new v4 package namespace

Required directories:

- `src/ctreepo/v4/core/`
- `src/ctreepo/v4/stack/`
- `src/ctreepo/v4/tasks/markov/`
- `src/ctreepo/v4/supervision/`
- `src/ctreepo/v4/report/`
- `src/ctreepo/v4/experiments/`

### D2. A shared stack public surface

Required public types and functions:

- `TreePOModelSpec`
- `TreePOContractSpec`
- `TreePOLocalLawConfig`
- `OracleLaneSpec`
- `TreePOSupervisionSpec`
- `build_treepo_stack(model_spec, contract_spec) -> TreePOStack`
- `TreePOStack.run_fixed_binary(...) -> FixedBinaryStateTreeRunResult`
- `emit_supervision_dataset(tree_run_result, supervision_spec) -> SupervisionDataset`
- `emit_numeric_rows(supervision_dataset) -> Iterable[dict]`
- `write_v4_run_artifacts(run_result, output_root) -> dict`

### D3. Three canonical scientific experiment families

- exact reference parity canary
- multileaf local-law ladder
- fixed-budget supervision frontier

### D4. One required text/LLM operational surface

- a minimal generate-first text run using the same stack
- explicit oracle-lane wiring or proxy wiring
- the same artifact contract and supervision emission path as the exact lane
- explicit metadata when `/generate` falls back to chat

### D5. One canonical report path

- build rows from a single explicit run root
- emit one summary JSON
- emit one Markdown report
- fail hard on duplicate or contaminated evidence

### D6. A minimal test suite

- shared stack construction
- exact parity contract
- text-lane smoke path
- supervision emission
- duplicate-row hard fail
- multileaf local-law gain sanity
- fixed-budget frontier sanity

## Non-goals

The minimal v4 explicitly does not include:

- LDA family migration
- publication bundles
- rolling report aggregation
- prefix-based `current` reports
- appendix PDF surfaces
- broad parity/capacity/bundle launch matrices
- mutable report manifests
- separate duplicated runners for Markov vs text
- a trainer-specific bespoke artifact schema
- global multi-benchmark package selection logic beyond one exact benchmark
  family and one text-lane smoke surface

## Source-of-truth module mapping

This section is the exact migration map from the current repo to minimal v4.

### Copy directly into v4

These modules already look like the right minimal shared-path kernel.

- `src/tree/state_tree.py`
  - destination: `src/ctreepo/v4/stack/state_tree.py`
- `src/tree/state_tree_runner.py`
  - destination: `src/ctreepo/v4/stack/state_tree_runner.py`
- `src/tree/state_tree_verifiers.py`
  - destination: `src/ctreepo/v4/stack/state_tree_verifiers.py`
- `src/tree/async_operator.py`
  - destination: `src/ctreepo/v4/stack/async_operator.py`
- `src/tree/treepo_stack.py`
  - destination: `src/ctreepo/v4/stack/treepo_stack.py`
- `src/tree/treepo_supervision.py`
  - destination: `src/ctreepo/v4/supervision/treepo_supervision.py`
- `src/tree/generate_prompting.py`
  - destination: `src/ctreepo/v4/stack/generate_prompting.py`
- `src/core/ops_checks.py`
  - destination: `src/ctreepo/v4/core/ops_checks.py`
- `src/ctreepo/sim/core/run_intent.py`
  - destination: `src/ctreepo/v4/core/run_intent.py`
- `src/ctreepo/sim/core/run_config.py`
  - destination: `src/ctreepo/v4/core/run_config.py`
- `src/ctreepo/sim/core/training_selection.py`
  - destination: `src/ctreepo/v4/core/training_selection.py`
- `src/ctreepo/sim/core/fno_arch_config.py`
  - destination: `src/ctreepo/v4/core/fno_arch_config.py`
- `src/ctreepo/sim/core/markov_oracle_metric.py`
  - destination: `src/ctreepo/v4/tasks/markov/oracle_metric.py`
- `src/ctreepo/sim/core/markov_capability.py`
  - destination: `src/ctreepo/v4/tasks/markov/capability.py`
- exact adapter portion of `src/ctreepo/sim/core/markov_theorem_feature_adapter.py`
  - destination: `src/ctreepo/v4/tasks/markov/theorem_adapter.py`

### Reuse in place, do not fork initially

These modules are already the correct repo-wide shared abstraction and should
stay shared unless there is a concrete v4-specific reason to move them later.

- `src/core/engines.py`
- `src/core/inference_engine.py`
- `src/core/scoring.py`
- `src/core/url_utils.py`
- `src/core/supervision_metadata.py`
- `src/training/supervision/types.py`
- `src/training/supervision/builders.py`
- `src/training/supervision/numeric_rows.py`
- `src/training/supervision/adapters.py`
- `src/training/supervision/torch_scalar.py`
- `src/training/supervision/torch_simplex.py`
- `src/training/embedding_proxy.py`

Reason:

- duplicating these would create two competing shared layers
- minimal v4 should cut launch/report spaghetti, not clone good shared
  infrastructure

### Split and re-home

These modules contain real scientific content, but must not be ported whole.

- `src/ctreepo/sim/core/markov_neural_operator_baselines.py`
  - split into:
    - `src/ctreepo/v4/tasks/markov/models/fno_encoder.py`
    - `src/ctreepo/v4/tasks/markov/models/tree_summary_model.py`
    - `src/ctreepo/v4/tasks/markov/models/merge_heads.py`
    - `src/ctreepo/v4/tasks/markov/models/theorem_feature_heads.py`
    - `src/ctreepo/v4/experiments/train_tree_fno.py`
    - `src/ctreepo/v4/tasks/markov/metrics.py`
    - `src/ctreepo/v4/tasks/markov/exact_witness.py`
- `src/ctreepo/sim/core/markov_changepoint_ops_count.py`
  - split into:
    - `src/ctreepo/v4/tasks/markov/dgp.py`
    - `src/ctreepo/v4/tasks/markov/data_bundle.py`
    - `src/ctreepo/v4/tasks/markov/exact_sketch.py`
    - `src/ctreepo/v4/tasks/markov/config.py`
    - `src/ctreepo/v4/tasks/markov/eval.py`
    - `src/ctreepo/v4/tasks/markov/summarize.py`
- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`
  - replace with:
    - `src/ctreepo/v4/experiments/exact_reference_canary.py`
    - `src/ctreepo/v4/experiments/multileaf_local_law_ladder.py`
    - `src/ctreepo/v4/experiments/supervision_frontier.py`
    - `src/ctreepo/v4/report/build_rows.py`
    - `src/ctreepo/v4/report/build_summary.py`

### Reuse selectively

These modules are useful as references, but should not define the new public
core as-is.

- `src/ctreepo/sim/core/tree_reference_presets.py`
  - keep only the parity-canary and standard-ladder recipe values
- `src/ctreepo/sim/core/markov_comparison_surface.py`
  - keep the surface-locking concepts, not the legacy wrapper context
- `src/ctreepo/sim/core/exact_utility_common.py`
  - reuse exact scoring and utility pieces required by the canary and frontier
- `src/ctreepo/sim/core/markov_treepo_preference.py`
  - reuse only if the supervision frontier needs the existing exact utility
    bridge

### Leave in legacy v3

- `src/ctreepo/sim/suite/*`
- `src/ctreepo/sim/report/*` except helpers copied into v4
- `src/ctreepo/sim/runner.py`
- `src/ctreepo/sim/manifest.py`
- `src/ctreepo/sim/resource_queue.py`
- `src/ctreepo/sim/learning_problem.py`
- `src/ctreepo/sim/local_law_backfill.py`
- `src/ctreepo/sim/theory_alignment.py`
- `src/ctreepo/sim/core/markov_alignment_validation.py`
- `src/ctreepo/sim/core/markov_v3_row_contract.py`
- `src/ctreepo/sim/core/markov_tree_fno_validation.py`
- `scripts/run_markov_optimization_tradeoff_pipeline.py`
- `scripts/launch_markov_v3_initial_grid.py`
- `scripts/refresh_markov_v3_rolling_report.py`
- all `publication_bundle*.toml` configs

## v4 package layout

The exact package structure for minimal v4 is:

```text
src/ctreepo/v4/
├── __init__.py
├── core/
│   ├── ops_checks.py
│   ├── run_config.py
│   ├── run_intent.py
│   ├── training_selection.py
│   ├── artifacts.py
│   └── row_contract.py
├── stack/
│   ├── async_operator.py
│   ├── generate_prompting.py
│   ├── state_tree.py
│   ├── state_tree_runner.py
│   ├── state_tree_verifiers.py
│   └── treepo_stack.py
├── tasks/
│   └── markov/
│       ├── benchmark.py
│       ├── capability.py
│       ├── config.py
│       ├── data_bundle.py
│       ├── dgp.py
│       ├── eval.py
│       ├── exact_sketch.py
│       ├── exact_witness.py
│       ├── metrics.py
│       ├── models/
│       │   ├── fno_encoder.py
│       │   ├── merge_heads.py
│       │   ├── theorem_feature_heads.py
│       │   └── tree_summary_model.py
│       ├── oracle_metric.py
│       ├── summarize.py
│       └── theorem_adapter.py
├── supervision/
│   ├── emitters.py
│   ├── numeric_rows.py
│   ├── policy.py
│   └── treepo_supervision.py
├── report/
│   ├── build_rows.py
│   ├── build_summary.py
│   ├── render_markdown.py
│   └── validate_rows.py
└── experiments/
    ├── exact_reference_canary.py
    ├── multileaf_local_law_ladder.py
    ├── supervision_frontier.py
    └── text_stack_smoke.py
```

## Public interfaces

The minimal v4 public surface should be exactly the following.

### Core dataclasses

- `V4RunId`
  - fields:
    - `run_id`
    - `surface_id`
    - `benchmark`
    - `experiment_kind`
    - `lane_kind`
    - `operator_kind`
    - `oracle_lane_kind`
    - `seed`
    - `train_docs`
    - `fixed_leaf_tokens`
- `V4RunManifest`
  - fields:
    - `run_id`
    - `surface_id`
    - `config`
    - `input_artifacts`
    - `git_revision`
    - `created_at`
- `V4RunResult`
  - fields:
    - `manifest`
    - `tree_run_result`
    - `aggregate_rows`
    - `supervision_payloads`
    - `summary`

### Public functions

- `build_treepo_stack(model_spec, contract_spec)`
- `TreePOStack.run_fixed_binary(...)`
- `run_exact_reference_canary(config)`
- `run_multileaf_ladder(config)`
- `run_supervision_frontier(config)`
- `run_text_stack_smoke(config)`
- `emit_supervision_dataset(tree_run_result, supervision_spec)`
- `emit_numeric_rows(supervision_dataset)`
- `write_v4_run_artifacts(run_result, output_root)`

### Artifact schema

Every v4 run must emit:

- `run_manifest.json`
- `rows.jsonl`
- `summary.json`
- `report.md`

Every run that enables supervision collection must also emit:

- `supervision_dataset.json`

Every aggregate row must contain at minimum:

- `run_id`
- `surface_id`
- `lane_kind`
- `operator_kind`
- `engine_surface`
- `oracle_lane_kind`
- `benchmark`
- `experiment_kind`
- `seed`
- `train_docs`
- `fixed_leaf_tokens`
- `effective_leaves_per_doc`
- `parity_mode`
- `local_law_active`
- `budget_total_calls_per_doc`
- `full_doc_budget_share`
- `root_error`
- `leaf_error`
- `merge_error`
- `delta_vs_reference`
- `theorem_evidence_status`
- `contract_status`
- `source_path`

## Canonical experiment definitions

These are the only three scientific experiment families that minimal v4 must
implement.

### Surface 1: Exact reference parity canary

Purpose:

- establish that the shared tree stack can exactly coincide with the reference
  comparator when the tree degenerates to one leaf per document
- run this through the same fixed-binary runner used everywhere else

Fixed defaults:

- lane: exact symbolic
- operator kind: `markov_toy_exact`
- benchmark: `recoverable_v4_t128`
- comparator family: `official_fno`
- tree family: one canonical tree family only
- package semantics: full-root baseline only
- leaf tokens: `128`
- effective leaves per doc: `1`
- train docs: `[1024, 4096, 10240]`
- seeds: `[0, 1]`

Structural confirmation:

- benchmark: `structural_core_v1_t128`
- cell: `r12_seg10to12`
- train docs: `[10240]`
- seeds: `[0]`

Hard contract:

- `parity_mode = "exact_full_doc"`
- `effective_leaves_per_doc = 1`
- local-law terms are recorded but inactive
- tree root error must exactly match the `official_fno` comparator on the same
  benchmark/train-doc/seed cell

Success criterion:

- exact parity passes for all recoverable canary cells
- structural confirmation also passes or is clearly labeled as a failing stress
  surface

### Surface 2: Controlled multileaf local-law ladder

Purpose:

- show whether local-law supervision matters once the tree has real internal
  structure
- keep the architecture and supervision comparison clean

Fixed defaults:

- lane: exact symbolic
- benchmark: `recoverable_v4_t128`
- tree family: same family as the parity canary
- leaf ladder: `[64, 32, 16, 8]`
- train docs: `[1024, 4096, 10240]`
- seeds: `[0, 1]`

Compared configurations:

- `root_only_multileaf`
  - root supervision only
  - no local-law objective
  - no local labels
- `standard_multileaf`
  - standard two-stage tree schedule
  - local-law objective active
  - local labels active

Structural confirmation:

- benchmark: `structural_core_v1_t128`
- cell: `r12_seg10to12`
- leaf ladder: `[64, 32]`
- train docs: `[10240]`
- seeds: `[0]`

Success criterion:

- at least one nondegenerate recoverable geometry shows `standard_multileaf`
  outperforming `root_only_multileaf`
- the surface is labeled as architectural/local-law evidence, not parity
  evidence

### Surface 3: Fixed-budget supervision frontier

Purpose:

- show trees as a supervision mechanism rather than only as an architecture
- keep the output directly consumable by PyTorch

Fixed defaults:

- lane: exact symbolic
- benchmark: `recoverable_v4_t128`
- geometry: `fixed_leaf_tokens = 32`
- train docs: `[1024, 4096, 10240]`
- seeds: `[0, 1]`
- total supervision mass per doc fixed at `1.0`
- local allocation policy: `balanced`

Compared points:

- `root_share = 1.0`
- `root_share = 0.9`
- `root_share = 0.8`
- `root_share = 0.5`

Interpretation:

- `root_share = 1.0` is the pure root baseline
- lower root share redistributes the same total supervision mass into local
  tree labels
- this is the canonical "trees as supervision" surface

Structural confirmation:

- benchmark: `structural_core_v1_t128`
- cell: `r12_seg10to12`
- points: `root_share = 1.0` and `0.8`
- train docs: `[10240]`
- seeds: `[0]`

Success criterion:

- at least one redistributed point beats the matched pure-root point on
  recoverable
- rows stay budget-comparable and do not rely on superset semantics

## Required text/LLM operational gate

This is not a fourth scientific family. It is a required shared-stack
validation surface.

### Text stack smoke

Purpose:

- prove minimal v4 is actually usable for LLM work, not only for Markov
- force the shared stack, verifier, and supervision path to stay lane-agnostic

Required defaults:

- lane: text
- surface requested: `/generate`
- chat fallback recorded if `/generate` is unavailable
- operator built through `TreePOModelSpec(...)`
- oracle lane built through `OracleLaneSpec(...)`
- supervision emitted through `TreePOSupervisionSpec(...)`

Minimal success criterion:

- the run returns a `StateTree`
- law checks attach to nodes when enabled
- supervision emits a valid `SupervisionDataset`
- the artifact contract matches the exact lane
- metadata explicitly records whether the run used `/generate` or chat fallback

## Report contract

The report path is intentionally narrow.

### Allowed behavior

- read one explicit run root
- validate rows
- build one summary JSON
- render one Markdown report

### Forbidden behavior

- choosing the newest directory by prefix
- rewriting source-selection manifests
- merging unrelated roots by path name
- silently resolving duplicate rows by path order
- using `latest` files as the canonical truth mechanism

### Duplicate handling

- duplicate semantic rows are a hard failure
- contaminated rows are a hard failure
- quarantined or legacy rows are never included in headline outputs

## Acceptance tests

The minimal test suite is complete when the following tests exist and pass.

### Shared stack tests

- `test_v4_build_treepo_stack_markov_exact.py`
- `test_v4_build_treepo_stack_text_generate.py`
- `test_v4_text_surface_fallback_metadata.py`
- `test_v4_shared_runner_artifact_schema.py`

### Core theorem and contract tests

- `test_v4_theorem_mapping.py`
  - assert C1/L1, C2/L3, C3/L2 mapping
- `test_v4_exact_markov_adapter.py`
  - require `first` and `last`
  - stable diagnostic key on exact `(count, first, last)`
- `test_v4_run_identity.py`
  - run identity is immutable and deterministic

### Exact parity tests

- `test_v4_exact_reference_canary_recoverable.py`
  - exact match on all recoverable canary cells
- `test_v4_exact_reference_canary_structural.py`
  - structural stress path emits explicit pass/fail labeling

### Multileaf tests

- `test_v4_multileaf_ladder_rows.py`
  - row schema and identity are correct
- `test_v4_multileaf_ladder_gain_sanity.py`
  - standard local-law lane beats root-only on at least one recoverable geometry

### Supervision tests

- `test_v4_text_supervision_emitter_smoke.py`
- `test_v4_supervision_emitter_root_only.py`
- `test_v4_supervision_emitter_local_targets.py`
- `test_v4_supervision_numeric_rows.py`
- `test_v4_supervision_frontier_budget_conservation.py`

### Reporting tests

- `test_v4_report_duplicate_rows_fail.py`
- `test_v4_report_legacy_rows_fail.py`
- `test_v4_report_summary_schema.py`

## Migration phases

Implementation will proceed in seven phases.

### Phase 0: Fence off the minimal-v4 effort

Actions:

- create the new `src/ctreepo/v4/` namespace
- mark this plan note as the project authority
- mark existing `long_v4` configs as legacy/non-authoritative in docs

Exit criterion:

- no new work is added to `long_v4` configs as part of the minimal-v4 build

### Phase 1: Port the shared runtime kernel

Actions:

- copy `StateTree`, the fixed-binary runner, the verifier layer, the async
  operator abstraction, the generate prompting helpers, and the stack builder
- copy theorem vocabulary and run-identity helpers

Exit criterion:

- v4 has one runnable stack surface with no dependency on v3
  suite/runner/report code

### Phase 2: Lock the supervision and artifact path

Actions:

- port `treepo_supervision.py`
- add thin emitters into the shared `src/training/supervision/*` surface
- define immutable `run_manifest.json`, `rows.jsonl`, `summary.json`, and
  `report.md`

Exit criterion:

- both exact and text runs can emit the same artifact schema and supervision
  dataset format

### Phase 3: Lock the text/LLM lane

Actions:

- make generate-first the default public path
- wire oracle lanes and proxy training through the shared stack
- add the text-stack smoke runner and tests

Exit criterion:

- minimal v4 can run a text task, verify locally, and emit supervision without
  special-case runner code

### Phase 4: Extract the Markov exact lane

Actions:

- split the Markov task logic out of `markov_changepoint_ops_count.py`
- isolate benchmark, DGP, exact sketch, oracle metric, and evaluation
- keep the exact benchmark as the correctness lock for the stack

Exit criterion:

- v4 can build exact reference targets without importing the large v3
  diagnostic runner

### Phase 5: Implement the three canonical scientific surfaces

Actions:

- implement exact reference canary
- implement multileaf ladder
- implement fixed-budget supervision frontier
- split only the minimal tree/FNO code needed by these surfaces

Exit criterion:

- each surface runs from one config object to one `V4RunResult`

### Phase 6: Implement the minimal report path

Actions:

- build canonical rows
- validate rows
- build summary JSON
- render Markdown report

Exit criterion:

- one run root produces deterministic `rows.jsonl`, `summary.json`, and
  `report.md`

### Phase 7: Delete or quarantine legacy dependencies from the v4 path

Actions:

- remove any residual dependency on:
  - `run_markov_optimization_tradeoff_pipeline.py`
  - `launch_markov_v3_initial_grid.py`
  - `refresh_markov_v3_rolling_report.py`
  - suite registries
  - mutable report manifests

Exit criterion:

- the minimal-v4 path is fully runnable without importing v3 launch/report glue

## Definition of done

Minimal v4 is done when all of the following are true.

- the new `src/ctreepo/v4/` namespace exists and has a shared stack core
- both exact and text lanes run through the same `build_treepo_stack(...)` plus
  `run_fixed_binary(...)` path
- exact reference parity passes on the recoverable benchmark
- the text stack smoke run emits a valid supervision dataset
- the multileaf ladder shows a real local-law benefit on at least one
  nondegenerate recoverable geometry
- the fixed-budget supervision frontier shows at least one improvement from
  redistributing supervision mass into the tree
- supervision datasets can be emitted directly into the shared PyTorch-facing
  supervision surface
- every run produces one immutable run manifest and one canonical report root
- duplicate or contaminated report rows hard fail
- no v4 public path depends on prefix-latest or mutable report-selection state

## First implementation order

This is the exact execution order to use when coding starts.

1. Scaffold `src/ctreepo/v4/`.
2. Copy the shared stack kernel modules.
3. Wire run identity, artifact writing, and supervision emission.
4. Make the text generate-first smoke path pass.
5. Extract the Markov exact benchmark and make the reference canary pass.
6. Split the minimal tree/FNO comparator code needed by the canary.
7. Implement the multileaf ladder.
8. Implement the fixed-budget supervision frontier.
9. Implement the minimal report path.
10. Add the acceptance tests and remove any remaining dependency on the v3
    launch/report stack.

## What remains before first testing and runs

This is the smallest remaining build scope before we should start serious
testing or experiment launches.

### Must exist before any meaningful v4 run

1. Scaffold `src/ctreepo/v4/` with the final package boundaries.
2. Port the shared stack kernel:
   - `state_tree.py`
   - `state_tree_runner.py`
   - `state_tree_verifiers.py`
   - `async_operator.py`
   - `treepo_stack.py`
   - `generate_prompting.py`
3. Port the minimal identity and artifact layer:
   - run identity
   - immutable manifest writing
   - canonical row contract
4. Port supervision emission:
   - `treepo_supervision.py`
   - thin adapters into the existing shared supervision types
5. Make one text-stack smoke path run end to end.
6. Make one exact Markov reference canary run end to end.

If those six things are not done, testing will mostly measure migration noise
instead of the real v4 design.

### Can wait until after the first runs work

- splitting the remaining Markov comparator code beyond the canary minimum
- the multileaf ladder
- the fixed-budget supervision frontier
- the final Markdown report renderer
- broader acceptance-test coverage

### What to minimize especially

These are the main places where the design should be made smaller before we run
anything serious.

#### 1. Minimize duplicated control planes

Do not build:

- one Markov runner
- one LLM runner
- one separate supervision export path per lane

Keep only:

- one `build_treepo_stack(...)`
- one `run_fixed_binary(...)`
- one supervision emitter
- one artifact writer

#### 2. Minimize copied infrastructure

Do not fork shared modules unless v4 truly needs different semantics.

Reuse in place:

- engine resolution
- inference engine wrappers
- scoring abstractions
- supervision dataset types
- numeric-row emitters
- proxy training helpers

#### 3. Minimize public API surface

The minimal public surface before testing should stay close to:

- `build_treepo_stack(...)`
- `TreePOStack.run_fixed_binary(...)`
- `emit_supervision_dataset(...)`
- `write_v4_run_artifacts(...)`

Everything else should remain private or experimental until the first runs are
stable.

#### 4. Minimize experiment surface count

Before broader testing, support only:

- one text-stack smoke run
- one exact reference canary

Do not revive broad grids, family bundles, or rolling reports yet.

#### 5. Minimize artifact complexity

Before the first runs, emit only:

- `run_manifest.json`
- `rows.jsonl`
- `summary.json`
- `report.md`
- `supervision_dataset.json` when enabled

Do not add:

- rolling summaries
- `latest` selectors
- mutable report manifests
- merged multi-root reports

#### 6. Minimize comparator/model extraction

Only split enough of the old Markov comparator/model code to support:

- exact reference parity
- one minimal tree-vs-reference comparison cell

Do not port the full historical baseline file before the shared stack is alive.

### Practical stop rule

We are ready to start real testing once all of the following are true:

- the shared v4 stack can run one text example
- the shared v4 stack can run one exact Markov example
- both runs emit the same artifact schema
- supervision emission works for both lanes
- no v4 path imports the rolling report or suite machinery

## Efficiency contract for large runs

Minimal v4 should be smaller than v3, but it should not regress throughput for
serious training runs.

The simplification rule is:

- remove duplicated launch, report, and orchestration logic
- keep the batch, residency, and vectorization ideas that actually move
  throughput

### Large Markov training requirements

Before any large Markov sweep, v4 should preserve these behaviors.

#### 1. Structure-aware batching

- batch by structural shape, not arbitrary document order
- keep leaf-count-aware bucketing as the default large-run strategy
- keep explicit runtime knobs for:
  - `bucket_mode`
  - `tree_batch_structural_pad_limit`
  - `tree_batch_auto_queue_min_docs`
  - `tree_batch_auto_queue_min_fill_ratio`

Reason:

- the active xlarge configs already rely on `bucket_mode = "leaf_count_auto_queue"`
- for tree models, padding waste and shape mismatch are first-order throughput
  costs

#### 2. Resident data mode

- support `data_mode = "resident"` for large runs
- preload train, val, and test splits when the run is large enough to justify
  residency
- keep resident-store reuse across training and evaluation passes

Reason:

- the current xlarge configs already use resident mode
- the old Markov baseline records resident-store build time, hit rate, miss
  rate, and steady-state host-to-device transfer cost, which means residency is
  already part of the real throughput story

#### 3. Vectorized batch forward paths

- keep batch leaf encoding
- keep packed tree-work items across documents
- keep fused or packed batch-forward execution for the hot path
- do not make per-document Python loops the default large-run training path

Reason:

- the old Markov baseline already has vectorized leaf encoding, packed tree
  batches, and a fused fixed-binary batch forward path
- minimal v4 should preserve that direction, not regress to the simpler legacy
  loop

#### 4. Device transfer discipline

- keep pinned host staging where it matters
- keep CUDA autocast or equivalent mixed-precision support in the hot path
- separate train and eval batch budgets when that improves occupancy or
  stability

Reason:

- the old baseline already uses pinned staging and autocast in the batch path
- these are low-complexity, high-value throughput features

#### 5. Batch-budget autotuning

- keep one autotune path for large Markov runs
- cache probe results by model, device, and geometry signature
- write the chosen budgets into the run manifest or summary

At minimum, the chosen contract should record:

- train leaf-token budget
- train node budget
- eval leaf-token budget
- eval node budget
- effective max docs
- bucket caps by leaf count
- auto-queue targets by leaf count

Reason:

- the old baseline already emits `autotuned_batch_budgets` and a probe-cache
  profile
- the point of minimal v4 is to make this smaller and clearer, not to throw it
  away and retune blindly on every run

#### 6. Runtime telemetry must survive simplification

Every serious large Markov run should emit:

- `runtime_efficiency`
- `batching_metrics`
- `autotuned_batch_budgets`

At minimum the telemetry should cover:

- mean docs per batch
- mean leaf tokens per batch
- mean nodes per batch
- padding waste ratio
- bucket utilization rate
- forward time
- backward time
- eval time
- resident-store hits and misses
- steady-state host-to-device bytes and time
- autotune cache hits and misses

Reason:

- if we do not carry these metrics forward, we will not know whether a "clean"
  v4 is actually faster or just less observable

### Text and LLM efficiency requirements

The same simplification rule applies to the text lane.

- prefer cross-document request pooling over per-document request submission
- keep the server fed with ready work across docs and levels
- reuse the existing batching ideas from `src/core/batch_processor.py` and
  `src/core/batch_orchestrator.py` where they match the shared stack
- do not create a separate artifact or supervision path for "batched text"

### Anti-patterns to avoid

Do not let v4 regress into:

- per-document hot-path training for large Markov runs
- one-off retuning for every child run in a sweep
- re-encoding identical leaf batches when the resident path can reuse them
- mixing report or launcher concerns into the training loop
- separate batch logic for exact and text lanes at the artifact level

### Practical gate before xlarge runs

Before launching any xlarge Markov sweep, run one pilot and confirm that the
artifacts include:

- `runtime_efficiency`
- `batching_metrics`
- `autotuned_batch_budgets`

and that the pilot uses:

- resident mode
- leaf-count-aware bucketing
- explicit auto-queue and structural-pad settings

## Practical interpretation

This plan is intentionally narrower than the existing v3 and `long_v4`
surfaces.

That is the point.

If a feature does not directly support one of these things:

- the shared TreePO stack
- text/LLM supervision emission
- exact reference parity
- multileaf local-law comparison
- fixed-budget supervision emission
- immutable per-run reporting

it is not part of the minimal v4.
