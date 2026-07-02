# C-TreePO Minimal v4 Audit

Date: 2026-04-13

Related notes:

- `docs/ctreepo_v4_minimal_plan.md`
- `docs/treepo_generate_first_stack.md`

Status:

- [x] Repo-wide read pass over the current v3 simulation stack
- [x] Parallel audit passes for architecture, reporting/collisions, theorem
      alignment, and publication story
- [x] Joint-path audit for the existing TreePO LLM stack
- [x] Initial minimal-v4 cut recommendation
- [ ] Code extraction / implementation
- [ ] Replace v3 report surfaces with the minimal v4 path

## Goal

Port the smallest working part of v3 into a minimal v4 that:

1. preserves the scientifically real tree-vs-reference story,
2. removes single-use orchestration and report spaghetti,
3. keeps theorem-facing honesty explicit,
4. makes trees usable primarily as a PyTorch supervision mechanism,
5. works for exact Markov-style benchmarks and LLM/text runs through one
   shared path.

## Main conclusion

The most important repo fact is:

- the codebase already contains a single joint TreePO path centered on
  `TreePOStack`, `StateTree`, the fixed-binary `StateTree` runner, and
  `TreePOSupervisionSpec`

That means minimal v4 should **not** be designed as:

- a small Markov package now
- plus a second LLM package later

It should be designed as:

- one shared execution path
- one shared verifier path
- one shared supervision-emission path
- one shared artifact contract

Within that joint path:

- Markov is the cheap exact reference benchmark and contract lock
- text/LLM is the real target workload

## Recommended v4 scope

Working recommendation for this note:

- v4 should be **single-path**, not Markov-first in package shape
- the primary public surface should be:
  - `build_treepo_stack(model_spec, contract_spec)`
  - `TreePOStack.run_fixed_binary(...)`
  - supervision emission into the shared PyTorch-facing dataset format
- the minimal reporting surface should be:
  - one immutable run manifest
  - one canonical rows file
  - one canonical JSON summary
  - one canonical Markdown report
- PDF generation, rolling `current` reports, and multi-family publication
  bundles should not be part of the v4 core

## Current v3 shape

There is a real reusable core in v3, but it is buried under a large amount of
launch, compatibility, and report machinery.

Largest entanglement points:

- `src/ctreepo/sim/core/markov_neural_operator_baselines.py` (~17.5k LOC)
- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py` (~13.7k LOC)
- `src/ctreepo/sim/core/markov_changepoint_ops_count.py` (~9.7k LOC)
- `scripts/run_markov_optimization_tradeoff_pipeline.py` (~13.7k LOC)
- `scripts/run_markov_supervision_recovery_parity_grid.py` (~3.4k LOC)
- `scripts/launch_markov_v3_initial_grid.py` (~2.6k LOC)

Main architectural split visible in the repo:

- useful reusable core:
  - fixed-binary tree execution
  - theorem vocabulary and capability types
  - text/exact operator abstraction
  - supervision data contracts
  - small config and identity helpers
- non-core scaffolding:
  - suite registries
  - rolling/latest report machinery
  - publication bundle wrappers
  - mutable report manifests
  - legacy/current quarantine compatibility layers
  - broad multi-surface launch matrices

## Existing joint path already in the repo

The strongest evidence against a Markov-only v4 is the current generate-first
TreePO stack.

Key files:

- `src/tree/treepo_stack.py`
- `src/tree/state_tree.py`
- `src/tree/state_tree_runner.py`
- `src/tree/state_tree_verifiers.py`
- `src/tree/async_operator.py`
- `src/tree/treepo_supervision.py`
- `docs/treepo_generate_first_stack.md`

Repo facts from those files:

- `build_treepo_stack(...)` is already the shared entrypoint
- `TreePOStack.run_fixed_binary(...)` is already the shared runner surface
- exact Markov runs and text/LLM runs already differ mainly through:
  - `TreePOModelSpec`
  - `OracleLaneSpec`
  - engine/oracle adapters
- the supervision path is already generic over `StateTree`, not Markov-specific
- chat is already treated as a fallback behind a generate-first surface

Audit conclusion:

- minimal v4 should extract and simplify this joint path
- minimal v4 should not invent a second runner abstraction for LLMs

## What v4 should keep

### 1. The shared fixed-binary stack

Keep directly:

- `src/tree/state_tree.py`
- `src/tree/state_tree_runner.py`
- `src/tree/state_tree_verifiers.py`
- `src/tree/async_operator.py`
- `src/tree/treepo_stack.py`
- `src/tree/treepo_supervision.py`
- `src/tree/generate_prompting.py`

Reason:

- this is already the cleanest implementation of the "single joint path"
- it is the right place to keep exact and text lanes aligned
- it already carries the right abstraction boundary for operator, verifier, and
  supervision wiring

Minimal v4 rule:

- this becomes the v4 core
- `TreeBuilder`-style legacy wrappers should be treated as compatibility shims,
  not as the primary design surface

### 2. The theorem-facing vocabulary

Keep directly:

- `src/core/ops_checks.py`
- the exact Markov adapter inside
  `src/ctreepo/sim/core/markov_theorem_feature_adapter.py`

Reason:

- this is the canonical Python-side Lean vocabulary
- C1/L1, C2/L3, C3/L2 are already encoded correctly here
- the exact Markov adapter is still the cleanest theorem-facing object in the
  current stack

Minimal v4 rule:

- keep only the exact Markov theorem adapter in the core theorem path
- move coarsened or fiber-bucket adapters behind an explicit experimental
  namespace

### 3. The shared supervision surface

Reuse directly as shared infrastructure:

- `src/training/supervision/types.py`
- `src/training/supervision/builders.py`
- `src/training/supervision/numeric_rows.py`
- `src/training/supervision/adapters.py`
- `src/training/supervision/torch_scalar.py`
- `src/training/supervision/torch_simplex.py`
- `src/tree/compositional_learning.py`

Reason:

- this is already the cleanest path for "trees as supervision" in PyTorch
- it separates supervision shape from model family
- duplicating it into v4 would create a second shared layer for no gain

Minimal v4 rule:

- the tree stack should emit canonical supervision artifacts into this shared
  surface
- v4 should not require its own bespoke trainer API to be useful

### 4. Small config and identity helpers

Keep directly:

- `src/ctreepo/sim/core/run_intent.py`
- `src/ctreepo/sim/core/run_config.py`
- `src/ctreepo/sim/core/training_selection.py`
- `src/ctreepo/sim/core/fno_arch_config.py`
- `src/ctreepo/sim/core/markov_oracle_metric.py`
- `src/ctreepo/sim/core/markov_capability.py`
- the useful surface-locking parts of
  `src/ctreepo/sim/core/markov_comparison_surface.py`

Reason:

- these files are small, explicit, and reusable
- they encode identity and comparability logic that v4 still needs

Minimal v4 rule:

- `run_intent` should become the immutable run-identity backbone
- the report layer must consume run identity, not rediscover it from paths

### 5. The Markov scientific kernel

Keep, but split:

- `src/ctreepo/sim/core/markov_changepoint_ops_count.py`
- `src/ctreepo/sim/core/markov_treepo_preference.py`
- `src/ctreepo/sim/core/exact_utility_common.py`
- parts of `src/ctreepo/sim/core/full_doc_config_codec.py`

Reason:

- this is where the exact benchmark, oracle, sketch, and
  supervision-recovery mechanics live
- these pieces contain the cheapest scientific control surface for correctness

Minimal v4 rule:

- treat this as the reference benchmark lane
- do not let the benchmark lane define the whole package architecture

### 6. The large-run efficiency kernel

Keep conceptually, but re-home into a much smaller training/runtime layer:

- structure-aware tree batching
- resident data mode
- vectorized leaf encoding
- fused or packed batch-forward paths
- pinned host staging
- CUDA autocast
- batch-budget autotuning with probe-cache reuse
- runtime and batching telemetry

Concrete evidence already in the repo:

- current xlarge generated configs use:
  - `data_mode = "resident"`
  - `bucket_mode = "leaf_count_auto_queue"`
  - `tree_batch_structural_pad_limit = 0.5`
  - `tree_batch_auto_queue_min_docs = 8`
  - `tree_batch_auto_queue_min_fill_ratio = 0.5`
- the old Markov baseline emits:
  - `runtime_efficiency`
  - `batching_metrics`
  - `autotuned_batch_budgets`
  - resident-store hit and miss metrics
  - autotune cache and probe metrics

Reason:

- large Markov runs are already throughput-sensitive
- the right simplification is to keep the throughput semantics and delete the
  surrounding launch/report sprawl

Minimal v4 rule:

- preserve the efficiency kernel
- do not preserve the giant monolithic training file shape
- do not let the minimal v4 stack regress to a doc-by-doc hot path for large
  training runs

## What v4 should copy as-is

Best copy-over candidates with minimal or no redesign:

- `src/tree/state_tree.py`
- `src/tree/state_tree_runner.py`
- `src/tree/state_tree_verifiers.py`
- `src/tree/async_operator.py`
- `src/tree/treepo_stack.py`
- `src/tree/treepo_supervision.py`
- `src/tree/generate_prompting.py`
- `src/core/ops_checks.py`
- `src/ctreepo/sim/core/run_intent.py`
- `src/ctreepo/sim/core/run_config.py`
- `src/ctreepo/sim/core/training_selection.py`
- `src/ctreepo/sim/core/fno_arch_config.py`
- `src/ctreepo/sim/core/markov_oracle_metric.py`
- `src/ctreepo/sim/core/markov_capability.py`

Safe to copy selectively:

- the exact Markov theorem adapter from
  `src/ctreepo/sim/core/markov_theorem_feature_adapter.py`
- the one-leaf parity canary recipe values from
  `src/ctreepo/sim/core/tree_reference_presets.py`
- the minimal surface-locking logic from
  `src/ctreepo/sim/core/markov_comparison_surface.py`

## What v4 should reuse in place

Keep these as shared repo-wide dependencies rather than forking them into v4:

- `src/core/engines.py`
- `src/core/inference_engine.py`
- `src/core/scoring.py`
- `src/core/url_utils.py`
- `src/core/supervision_metadata.py`
- `src/training/supervision/*`
- `src/training/embedding_proxy.py`

Reason:

- these are already good shared abstractions
- minimal v4 should remove duplicate control planes, not introduce new ones

## What v4 should split, not copy whole

### 1. `markov_neural_operator_baselines.py`

Do not carry this as a single file.

Split into:

- minimal Markov comparator model files
- minimal training entrypoints for the three canonical surfaces
- minimal metric and witness helpers

Reason:

- the file currently mixes model definition, batching, training, evaluation,
  diagnostics, theorem-feature routing, and report-facing metadata
- this is the biggest source of hidden coupling in the v3 core

Important qualification:

- large-run efficiency should be extracted from this file, not discarded
- the parts worth preserving are:
  - packed tree-work batching
  - fused fixed-binary batch forward
  - resident-store reuse
  - pinned staging
  - mixed precision
  - batch-budget autotuning
  - runtime telemetry
- the parts that should not survive are the broad experiment/report wrappers

### 2. `full_doc_anchor_diagnostics.py`

Do not carry this file whole into v4.

Replace it with:

- a minimal exact reference canary runner
- a minimal ladder runner
- a minimal frontier runner
- a minimal summary emitter

Reason:

- v4 does not need a giant all-in-one diagnostic/report engine just to answer:
  - does one leaf equal the reference?
  - what happens when geometry deepens?
  - do local laws help under a fixed supervision budget?

### 3. `markov_changepoint_ops_count.py`

Keep the scientific content, but split by responsibility:

- DGP and data-bundle construction
- exact sketch and oracle utilities
- model and training config
- experiment execution
- metric summarization

Reason:

- the file is too large to remain the v4 public core intact
- its content is important, but its current shape is not

## What v4 should drop from the core

Drop from the minimal core entirely:

- `src/ctreepo/sim/suite/*`
- `src/ctreepo/sim/report/*` as a package boundary, except selected helpers
  reused in the minimal report
- `src/ctreepo/sim/runner.py`
- `src/ctreepo/sim/manifest.py`
- `src/ctreepo/sim/resource_queue.py`
- `src/ctreepo/sim/expectations.py`
- `src/ctreepo/sim/local_law_backfill.py`
- `src/ctreepo/sim/learning_problem.py`
- `src/ctreepo/sim/theory_alignment.py`
- `scripts/launch_markov_v3_initial_grid.py`
- `scripts/refresh_markov_v3_rolling_report.py`
- `config/markov/publication_bundle*.toml`

Move to a legacy or appendix namespace:

- `src/ctreepo/sim/core/markov_v3_row_contract.py`
- `src/ctreepo/sim/core/markov_alignment_validation.py`
- `src/ctreepo/sim/core/markov_tree_fno_validation.py`
- `src/ctreepo/sim/core/full_doc_anchor_ladder.py`
- `src/ctreepo/sim/core/full_tree_ipw_grid.py`
- broad LDA family launch and report surfaces

Reason:

- these are mostly publication plumbing, compatibility guards, or
  evidence-quarantine machinery for a broader v3 world
- they may still be useful historically, but they should not define the v4
  package

## Reporting and collision diagnosis

Current v3 report problems are structural, not cosmetic.

Observed failure modes:

- report-row identity can collapse distinct runs under the same semantic key
- path ordering can decide the surviving row
- rolling reports choose the newest directory by prefix, not a stable run
  identity
- report manifests are mutable state, not pure provenance
- refresh paths rewrite selected source metadata into the same output root
- stale runs can be backfilled by mtime rather than immutable identity

This is why v3 can make it hard to tell what really happened.

Minimal v4 report rules:

1. every run gets one immutable `run_id`
2. every report surface gets one immutable `surface_id`
3. reports consume an explicit input set, never a prefix scan
4. reports never rewrite the provenance manifest they read from
5. duplicate semantic rows are a hard failure, not a tie-break opportunity
6. there is no `latest` or `current` logic inside the core reporting path

Canonical v4 artifact set per run:

- `run_manifest.json`
- `rows.jsonl`
- `summary.json`
- `report.md`
- `supervision_dataset.json` when supervision collection is enabled

Optional later:

- `report.pdf` generated from `report.md`, but not part of the core truth
  surface

## Minimal evidence package for v4

The smallest convincing v4 package is three scientific experiment families plus
one required text-lane operational gate.

### 1. Exact reference parity canary

Purpose:

- prove the tree can exactly coincide with the reference comparator when there
  is one leaf per document and local-law terms are structurally inactive

Interpretation rule:

- this is a parity and protocol check only
- it is not evidence that local laws help

### 2. Controlled multileaf local-law ladder

Purpose:

- show where tree structure becomes nontrivial and whether local-law
  supervision helps once the tree has real internal structure

Interpretation rule:

- this is the main architectural and local-law comparison surface
- this is where "trees work" should actually be evaluated

### 3. Fixed-budget supervision frontier

Purpose:

- show the tree as a supervision mechanism, not only as an architecture

Interpretation rule:

- if v4 is supposed to matter for PyTorch training, this is the most important
  surface after exact parity

### 4. Text-stack smoke gate

Purpose:

- prove the shared stack is actually usable for LLM work
- stop the minimal-v4 design from drifting back into a Markov-only package

Interpretation rule:

- this is an operational gate, not a headline publication family

## Efficiency implications for v4

The audit recommendation is:

- minimal v4 should be simple in control plane
- not simple in the sense of throwing away batching and residency

Concretely:

- keep one efficiency story for large Markov runs
- keep one efficiency story for text runs
- keep both stories on the same artifact and supervision contract

For large Markov runs, v4 should preserve:

- leaf-count-aware bucketing
- resident preload mode
- vectorized leaf encodes
- packed multi-document tree batches
- cached autotuned batch budgets
- runtime and batching telemetry

For text runs, v4 should preserve:

- cross-document request pooling
- global ready-work batching across docs and levels where applicable
- a generate-first path that does not fork the artifact or supervision story

The main thing to avoid is a false simplification:

- replacing the current high-throughput path with a smaller but slower
  doc-by-doc path
- then compensating for lost throughput with more launch and scheduler
  complexity

## Minimal PyTorch-facing design

Primary design choice:

- trees should be a **supervision adapter**
- not a giant end-to-end experiment runner

Minimal v4 public API should expose something like:

- `build_treepo_stack(...) -> TreePOStack`
- `TreePOStack.run_fixed_binary(...) -> FixedBinaryStateTreeRunResult`
- `emit_supervision_dataset(tree_run_result, spec) -> SupervisionDataset`
- `emit_numeric_rows(supervision_dataset) -> rows`
- `emit_run_summary(run_result) -> summary.json payload`

The supervision payload should be enough for a plain PyTorch trainer to consume
without knowing the whole C-TreePO stack.

Required emitted information:

- run identity: `run_id`, `surface_id`, `benchmark`, `split`
- lane metadata: `lane_kind`, `operator_kind`, `engine_surface`,
  `oracle_lane_kind`
- geometry: `fixed_leaf_tokens`, `leaves_per_doc`, depth metadata
- supervision channel: root, leaf, internal
- target kind: scalar, theorem feature, count, exact sketch, or preference
- value and weight
- propensity if sampling is used
- theorem/proxy label for each target family

Recommended design rules:

- keep the dataset emitter first-class
- keep trainer wrappers thin and optional
- keep the exact and text lanes on the same supervision contract

## Honesty and theorem rules that v4 must preserve

Non-negotiable invariants:

- C1 = L1
- C2 = L3
- C3 = L2
- theorem totals contain only C1/C2/C3
- schedule consistency and schedule spread stay proxy-only
- one-leaf parity is empirical equality, not a Lean theorem
- full-tree IPW remains diagnostic unless explicit CI semantics are added
- legacy/current rows never mix in headline evidence

Minimal tests to preserve:

- theorem-mapping unit test
- exact Markov adapter unit test
- exact one-leaf parity contract test
- text-stack smoke test
- duplicate-row hard-fail test in reporting
- no contaminated or quarantined rows in headline outputs
- supervision-emission smoke test into the shared training/supervision surface

## Proposed v4 package shape

Recommended layout:

- `src/ctreepo/v4/core/`
- `src/ctreepo/v4/stack/`
- `src/ctreepo/v4/tasks/markov/`
- `src/ctreepo/v4/supervision/`
- `src/ctreepo/v4/report/`
- `src/ctreepo/v4/experiments/`

Rough mapping:

- `core/`
  - theorem vocabulary
  - run identity
  - config contracts
- `stack/`
  - `TreePOStack`
  - `StateTree`
  - fixed-binary runner
  - verifiers
  - async operator abstraction
- `tasks/markov/`
  - benchmark
  - DGP
  - oracle
  - exact sketch
  - minimal comparator models
- `supervision/`
  - emitters into `src/training/supervision/*`
- `report/`
  - canonical row builder
  - summary builder
  - Markdown renderer
- `experiments/`
  - `exact_reference_canary.py`
  - `multileaf_local_law_ladder.py`
  - `supervision_frontier.py`
  - `text_stack_smoke.py`

## Practical cut list

### Copy first

- shared fixed-binary stack
- theorem vocabulary
- run-identity helpers
- exact Markov theorem adapter

### Reuse directly

- shared engine, scoring, and supervision modules

### Extract second

- Markov exact benchmark kernel
- minimal comparator model modules
- minimal experiment runners

### Replace third

- rolling/latest reports
- mutable report manifests
- large pipeline launchers
- multi-surface publication bundles

### Leave behind for now

- LDA family shells
- broad publication-progress suites
- archived parity diagnosis flows
- appendix-only report scripts

## Immediate implementation order

1. Create a fresh v4 namespace rather than editing the v3 stack in place.
2. Copy the shared stack kernel first.
3. Port supervision emission and immutable artifact writing.
4. Make a text generate-first smoke run pass through the shared stack.
5. Extract the Markov exact benchmark and make the reference canary pass.
6. Build the multileaf ladder.
7. Build the supervision frontier.
8. Add the minimal JSON and Markdown report path last.

## Bottom line

The right minimal v4 is not "v3 but tidier," and it is not "Markov first,
LLMs later."

It should be:

- one small shared TreePO stack,
- one shared supervision-emission layer into PyTorch,
- one exact reference benchmark lane,
- one required LLM/text operational lane,
- one multileaf local-law ladder,
- one fixed-budget supervision frontier,
- one immutable report surface per run.

Everything else in v3 should be treated as legacy scaffolding unless it is
needed to support one of those seven things.
