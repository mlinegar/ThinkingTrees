# Two-Stage Teacher-First Swap Plan

This note describes the work required before the Python stack should fully
switch to the teacher-first route:

1. learn or cache an expensive surrogate oracle or theorem feature in stage 1;
2. train the tree summary and downstream heads relative to that surrogate in
   stage 2; and
3. evaluate the result using the decomposed Lean-aligned error terms rather than
   only aggregate task MAE / AUC.

The Lean side now has the main ingredients:

- direct surrogate-oracle transfer with additive stage-1 slack
- layered shared-feature transfer with explicit stage-2 transport, stage-2
  fiber error, measurement error, and stage-1 substitution cost
- breakeven and multistage distillation lemmas

The Python side should only cut over once those terms have first-class runtime
representations, metrics, and validation gates.

Current dated implementation/status handoff:

- [tree_neural_teacher_first_handoff_2026-03-23.md](./tree_neural_teacher_first_handoff_2026-03-23.md)

## Practical optimization rule

The runner should not assume there is one universally best surrogate.

In practice, teacher-first selection should optimize over a frontier:

- lower `stage1_substitution_cost`
- lower `stage2_transport_budget`
- lower downstream root/task error
- lower total decomposition-aware bound

The right operational default is:

1. rank candidates by the total bound
2. retain the Pareto frontier over substitution, transport, and downstream
   error
3. promote multiple frontier candidates when they expose materially different
   tradeoffs

This matters because stage-1 root-aware surrogates can improve substitution
cost while slightly worsening transport, and theorem-only surrogates can do the
reverse. The search space therefore needs explicit stage-1 objective knobs,
especially `tree_stage1_root_weight` and the stage-1 checkpoint metric.

## Goal

Make the default tree-neural training route optimize for the large-model /
teacher target in the same way the theory now describes:

- stage 1 approximates the expensive oracle or its learned theorem feature
- stage 2 learns the summary operator relative to that approximation
- evaluation reports the four-term decomposition and the direct task outcome

The Markov setting remains the smoke-test lane, not the conceptual template.

## Current status

What already exists:

- Lean now formalizes:
  - direct teacher-first surrogate transfer
  - arbitrary scores on decoded labels
  - layered two-stage end-to-end decomposition
  - breakeven and multistage distillation tradeoffs
- Python already has:
  - theorem-feature adapters
  - covered-pair supervision
  - shared-feature / shared-feature-adapters routes
  - generic pairwise diagnostics

What is still missing:

- stage 1 is not yet a first-class artifact in the Python training loop
- stage 2 diagnostics do not report the full Lean decomposition
- runners still think in terms of one monolithic training job rather than a
  staged pipeline with cached stage-1 outputs
- defaults still carry too much Markov-shaped decode logic

## Target protocol

### Stage 1

Inputs:

- full-document teacher labels
- sampled node / merge labels when budget allows
- optional paired same/different labels induced by the teacher or surrogate

Outputs:

- a cached surrogate artifact
- enough metadata to replay how it was trained and what budget it used

Artifact contents:

- `surrogate_kind`
- `teacher_target_name`
- `train/val/test splits`
- fitted surrogate parameters or checkpoint path
- calibration statistics
- pair coverage statistics
- root/node labeling coverage
- stage-1 quality metrics

Stage-1 quality metrics:

- root agreement with teacher
- sampled-node agreement with teacher
- pairwise same/different quality
- calibration error or score distortion
- feature distortion or oracle approximation error where available

### Stage 2

Inputs:

- the cached stage-1 surrogate artifact
- tree data
- theorem-feature adapter semantics
- downstream task or label-score objective

Training behavior:

- the tree operator is trained against the stage-1 surrogate, not the raw
  expensive teacher, except where spot-checking or held-out validation is
  intentionally enabled
- `shared_feature` should be the default theory-aligned route
- unfactored full-state root heads remain ablations only

Outputs:

- the trained summary operator
- downstream factored heads
- full decomposition diagnostics

## Lean-to-Python metric map

The runtime should report one metric per main theorem term.

### Direct surrogate-oracle route

Lean term:

- surrogate transport budget
- additive `2 * eps_stage1`

Python metrics:

- `stage2_transport_budget`
- `stage1_surrogate_slack`
- `teacher_first_total_bound`

### Layered shared-feature route

Lean term:

- stage-2 transport
- stage-2 fiber error
- root measurement error
- stage-1 substitution cost

Python metrics:

- `stage2_transport_budget`
- `stage2_fiber_error`
- `root_measurement_error`
- `stage1_substitution_cost`
- `teacher_first_decomposed_bound`

These should be emitted per split and, where possible, per node type:

- root
- labeled internal nodes
- leaves

## Required Python changes

### 1. First-class stage-1 surrogate artifact

Add a stable artifact model and serializer for stage 1.

Needed changes:

- add a dataclass or Pydantic model for stage-1 surrogate outputs
- make stage-1 training runnable independently from stage 2
- support loading a frozen stage-1 artifact into stage-2 runs
- store the teacher budget and labeling policy alongside the artifact

Likely files:

- `src/ctreepo/sim/core/theorem_feature_route.py`
- `src/ctreepo/sim/core/markov_neural_operator_baselines.py`
- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`
- a new helper module for staged artifact IO if the existing route gets too
  crowded

### 2. Stage-2 objective cleanup

The stage-2 route should be explicitly relative to the surrogate artifact.

Needed changes:

- separate "teacher labels" from "surrogate labels" in the data path
- use teacher labels only for held-out evaluation and optional spot checks
- remove residual assumptions that stage 2 directly learns the true theorem
  object
- treat Markov tuple decode as a probe, not the central theorem-facing target

### 3. Four-term diagnostics

Add explicit runtime estimators for:

- `stage2_transport_budget`
- `stage2_fiber_error`
- `root_measurement_error`
- `stage1_substitution_cost`

Also keep:

- downstream task loss
- pairwise same/different metrics
- factored-readout gap

The critical rule is that aggregate task loss should no longer be the only
debugging surface. The decomposition has to be visible in reports and runner
summaries.

### 4. Stage-aware runners

Runners should support:

- stage-1-only jobs
- stage-2-only jobs using a frozen stage-1 artifact
- end-to-end pipelines that explicitly save the stage-1 artifact before stage 2

Needed changes:

- runner manifests record both stages
- report scripts aggregate both stages
- restart / resume logic respects the stage boundary

### 5. Adapter cleanup

Adapters should distinguish three things:

- teacher label space
- surrogate feature or surrogate oracle space
- optional decoded probe space

Needed changes:

- keep `task_readout_target`
- add optional `surrogate_reference_target`
- make canonical decode optional and clearly diagnostic-only unless the adapter
  is truly exact

### 6. Validation gates

Do not switch the default route until all of the following are true:

1. stage-1 artifacts can be trained, saved, loaded, and reused
2. stage-2 runs consume frozen stage-1 artifacts without hidden teacher calls
3. the four decomposition terms are emitted in diagnostics
4. Markov smoke tests pass under the staged route
5. one non-Markov toy adapter passes the staged route
6. runner summaries expose stage-1 and stage-2 metrics separately

## Suggested implementation order

1. Add the stage-1 artifact model and serializer.
2. Make stage-1 training invocable independently.
3. Load frozen stage-1 artifacts into the shared-feature training path.
4. Emit the four decomposition metrics in diagnostics.
5. Update runners and reports to treat stage 1 and stage 2 as distinct jobs.
6. Flip the default route only after the staged validation gates pass.

## Tradeoffs to track

### Why the two-stage route can win

- stage 2 optimizes for a fixed target
- expensive teacher supervision is amortized
- pairwise and nodewise labels can be generated from the surrogate rather than
  requerying the teacher
- large-model quality can be retained while stage 2 is much cheaper

### Why it can fail

- bad stage-1 approximation gets preserved rather than repaired
- stage-1 substitution cost can dominate any stage-2 budget savings
- too many distillation layers can amplify error
- discontinuous downstream score functions are harder to control than smooth
  ones

### Practical decision rule

Use the staged route as the default only when:

- `stage2_transport_budget + stage2_fiber_error + root_measurement_error +
  stage1_substitution_cost`

is consistently more informative than, and typically smaller than, the direct
single-stage error budget on the same held-out tasks.

## Tournament-of-tournaments selection

One practical way to reduce stage-1 substitution cost is not to trust the
stage-1 fitting loss alone.

Instead:

1. train a bracket of candidate stage-1 surrogates;
2. freeze each surrogate as an artifact;
3. run a smaller downstream stage-2 optimization bracket against each frozen
   surrogate; and
4. rank the stage-1 candidates by the best stage-2 result they induce.

This is a useful operational translation of the Lean tradeoff story:

- `stage1_substitution_cost` is the theorem-side quantity we are trying to
  drive down;
- the outer tournament is a search procedure over surrogate artifacts;
- the inner tournament estimates whether a surrogate is actually useful once
  the downstream summary operator is optimized relative to it.

Recommended ranking surface for the outer tournament:

- primary: `teacher_first_total_bound`
- first tie-break: `stage1_substitution_cost`
- second tie-break: downstream task MAE
- third tie-break: `stage2_transport_budget`

This keeps the search aligned with the formal decomposition instead of letting
the outer tournament drift back toward raw task MAE only.

## Immediate follow-up after this plan

The next coding pass should implement:

1. a stage-1 surrogate artifact schema
2. diagnostic estimation of the four Lean-aligned terms
3. runner support for stage-1-only and stage-2-only execution
4. a report page that shows direct task outcome next to the decomposition
