# Tree-Neural Teacher-First Handoff 2026-03-23

This note is the current handoff for the tree-neural / teacher-first lane.
It is written for other LLMs or humans who need to understand:

- what the current theory-aligned objective is
- what was implemented in Lean and Python
- what failed operationally and how it was fixed
- what the best empirical signal is today
- what is still unresolved, especially on GPU throughput

Use this together with:

- [two_stage_teacher_first_swap_plan.md](./two_stage_teacher_first_swap_plan.md)
- [treepo_supervision_optimization.md](./treepo_supervision_optimization.md)

## Executive Summary

The project moved from a Markov-slot-centric theorem surface toward a
teacher-first, theorem-feature route.

The highest-level progression was:

1. generalize the Lean surface from fixed slot sketches to oracle fibers,
   shared theorem features, and two-stage surrogate training;
2. refactor Python around theorem-feature adapters and decomposition-aware
   metrics;
3. diagnose and fix the catastrophic host-RAM blowup in exact evaluation;
4. introduce a factorized score-fiber route aligned to the new Lean product
   state;
5. tighten reliability around grouped stage-2 jobs and async scheduling;
6. begin a true GPU-throughput pass with a fixed-structure fused backend for
   `recoverable_v4`.

Current qualitative status:

- theory alignment is much better than at the start of this effort
- the score-fiber route is currently the most promising modeling direction
- reliability is much better than before the grouped-stage2 deadlock fix
- throughput is still not where it needs to be, but the new fused path is now
  clearly using more GPU memory than the generic bucket path

Current practical status:

- do not treat the old `shared_feature_phi192` route as the only serious route
- do not trust old timing numbers from probes that used numeric
  `CUDA_VISIBLE_DEVICES=0/1/2/3` on this machine
- the next decision point should be based on the clean MIG-UUID fixed-fused
  probe results, not the older underfilled runs

## Theory Progression

### Initial diagnosis

The original modeling problem was that the theorem-facing object was still too
close to a Markov sketch:

- fixed count / first / last surfaces
- `phi` acting as a carrier for that sketch
- C2 mostly expressed as replay or reconstruction, not equivalence on oracle
  fibers

The real target was:

- limited access to a true oracle `f*`
- a learned latent `Phi`
- a summary / merge operator `g` that preserves the relevant oracle structure
- downstream heads factored through `Phi`

### Lean extensions added in this arc

The Lean side was pushed in roughly this order:

1. oracle-fiber and feature-fiber formalization
2. covered-pair / sparse-supervision theorems
3. arbitrary score transport over decoded labels
4. two-stage teacher-first surrogate transfer
5. product-state specialization for a score slice plus a fiber slice

The important conceptual shift is:

- C2 is now about oracle-fiber preservation, not just slot replay
- two-stage teacher-first training is now a first-class theorem route
- the score-fiber factorization is now an explicit product-state specialization,
  not just an informal engineering story

### Main Lean artifacts from this arc

These were the key new or promoted Lean files:

- `OracleFiberRelations.lean`
- `SharedFeatureMultihead.lean`
- `ApproxOracleRecovery.lean`
- `FiberPreservingObjective.lean`
- `LabelScoreObjectives.lean`
- `TwoStageOracleSurrogate.lean`
- `TwoStageLabelScoreObjectives.lean`
- `TwoStageDecomposition.lean`
- `ProductScoreFiber.lean`

High-level theorem surface now available:

- two-stage surrogate transfer with additive stage-1 slack
- layered decomposition into:
  - stage-2 transport
  - stage-2 fiber error
  - root measurement error
  - stage-1 substitution cost
- arbitrary score transport over decoded labels
- product-state specialization where the theorem state is a bounded
  score coordinate plus a fiber state

## Python Progression

### 1. Generic theorem-feature route

The first big Python shift was to stop hard-coding the theorem route to Markov
slot structure.

Main additions:

- theorem-feature adapter layer
- generic same/different pair construction
- shared-feature and shared-feature-adapter routes
- decomposition-aware diagnostics

Key files:

- `src/ctreepo/sim/core/theorem_feature_route.py`
- `src/ctreepo/sim/core/markov_theorem_feature_adapter.py`
- `src/ctreepo/sim/core/markov_neural_operator_baselines.py`
- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`

This was a necessary transition, but it still left the route too close to a
generic shared embedding.

### 2. Teacher-first artifact and decomposition metrics

The next change made stage 1 and stage 2 explicit in the runtime:

- stage-1 artifact save/load
- stage-2 reuse of frozen artifacts
- explicit metrics:
  - `stage2_transport_budget`
  - `stage2_fiber_error`
  - `root_measurement_error`
  - `stage1_substitution_cost`
  - `teacher_first_total_bound`

This made the theory-runtime mapping real enough to optimize against.

### 3. Stage-1 search, halving, and async promotion

The runner stack then moved through:

- explicit teacher-first tournaments
- root-weight sweeps
- halving-style stage-1 pruning
- grouped stage-2 execution
- async per-count promotion

Relevant runners:

- `scripts/run_tree_neural_teacher_first_push.py`
- `scripts/run_tree_neural_teacher_first_scaling_push.py`
- `scripts/run_tree_neural_full_doc_mig.py`

### 4. Score-fiber route

After the generic shared-feature route still looked too expensive and too
embedding-like, the code added a factorized theorem surface:

- theorem state split as `[score | fiber | aux]`
- scalar task head reads the score slice directly
- summary heads consume the full product state
- fiber losses apply only to the fiber slice
- score merge is structured (`gated_affine`)

This is the current best alignment to the Lean product-state story.

### 5. GPU throughput work

The most recent implementation pass added:

- bucketed work-item batching
- pinned staging and nonblocking copies
- a `fixed_fused` backend for fixed-structure `recoverable_v4`
- fixed-fused root prediction and teacher-first node-view support
- fused training inner-loop support for:
  - `shared_feature`
  - `factorized_score_fiber`

The fused path is now the intended fast path for `recoverable_v4`.

## Major Operational Problems And Fixes

### Host-RAM blowup in exact evaluation

This was the biggest stability problem.

Diagnosis:

- saved simulated data was fine
- the blowup happened in stage-1 training/evaluation, not data generation
- the legacy exact evaluator still materialized large structures and pairwise
  comparisons

The decisive fixed-bundle memory bisect is:

- [tree_neural_fixed_bundle_memory_bisect_20260322_053849/memory_bisect_summary.json](../outputs/tree_neural_fixed_bundle_memory_bisect_20260322_053849/memory_bisect_summary.json)

Key result:

- `slotwise_control_legacy_exact`: about `2.39 GiB` peak RSS
- `shared_feature_adapters_exact_selection_legacy`: killed at about `126.17 GiB`
- `shared_feature_adapters_cheap_selection_legacy`: killed at about `230.33 GiB`
- `shared_feature_adapters_cheap_selection_streaming`: completed at about
  `9.46 GiB` observed peak, with the instrumented checkpoint timeline itself
  around `2.21 GiB`

Main fixes:

- streaming exact evaluator
- cheap stage-1 checkpoint selection
- efficient pair-AUC computation

### Grouped stage-2 deadlock

The async scaling runner later hung even after grouped stage-2 summaries were
written.

Cause:

- grouped workers wrote a large JSON payload to stdout
- controller launched them with `stdout=PIPE`
- controller drained only after exit
- pipe filled, worker blocked, controller waited forever

Fix:

- make grouped-stage2 completion file-driven from
  `grouped_stage2_summary.json`
- reduce worker stdout to a tiny completion record
- stop parsing grouped-stage2 completion from a large stdout payload

This was a reliability fix, not a modeling fix.

### MIG placement mistake during probing

One later speed probe used:

- `CUDA_VISIBLE_DEVICES=0`
- `CUDA_VISIBLE_DEVICES=1`
- `CUDA_VISIBLE_DEVICES=2`
- `CUDA_VISIBLE_DEVICES=3`

On this machine, that meant four MIG slices on physical GPU 0, not one
physical GPU each.

Important rule for future runs:

- use MIG UUIDs, not numeric `CUDA_VISIBLE_DEVICES`, when you want explicit
  per-physical-GPU placement

## Empirical Status

### Teacher-first broad halving baseline

The main broad sweep before the score-fiber switch was:

- [tree_neural_teacher_first_halving_overnight_20260322_084727/broad_seeds01/teacher_first_scaling_summary.json](../outputs/tree_neural_teacher_first_halving_overnight_20260322_084727/broad_seeds01/teacher_first_scaling_summary.json)

Best candidates from that run:

| Train docs | Winner | Test root MAE | Mean total bound | Mean stage-1 substitution |
| --- | --- | ---: | ---: | ---: |
| 128 | `teacherfirst_shared_feature_phi192_root0p50` | `0.708` | `0.980` | `0.690` |
| 512 | `teacherfirst_shared_feature_phi192_root0p50` | `0.435` | `1.322` | `0.508` |
| 1024 | `teacherfirst_shared_feature_phi192` | `0.257` | `0.532` | `0.285` |
| 2048 | `teacherfirst_shared_feature_phi192_root0p50` | `0.199` | `0.360` | `0.199` |

Interpretation:

- `phi192` clearly improved with scale
- stage-1 substitution cost dropped with more data
- the route still looked expensive and too embedding-heavy

### Small score-fiber route compare

The first direct compare after adding score-fiber was:

- [scorefiber_route_compare_20260323_small/summary.json](../outputs/scorefiber_route_compare_20260323_small/summary.json)

Key rows:

| Route | Elapsed seconds | Test root MAE | Teacher-first total bound |
| --- | ---: | ---: | ---: |
| `scorefiber_s1_f15` | `100.6` | `0.6902` | `0.8910` |
| `shared_feature_phi192` | `110.1` | `0.6891` | `1.0759` |
| `shared_feature_adapters_phi128` | `125.2` | `0.6895` | `1.5561` |

Interpretation:

- score-fiber was not dramatically faster yet
- but it was already competitive on MAE and better on the decomposition-aware
  bound

### Score-fiber smoke tournament

The first dedicated score-fiber teacher-first smoke was:

- [scorefiber_teacher_first_smoke_20260323/teacher_first_tournament_summary.json](../outputs/scorefiber_teacher_first_smoke_20260323/teacher_first_tournament_summary.json)

Key result:

- `teacherfirst_scorefiber_s1_f15_root0p50` beat the no-root-weight variant in
  the smoke tournament

This was a tiny run, but it reinforced the root-weight signal that showed up
earlier in the shared-feature lane.

### Rapid 1024/2048 overnight

The most important current modeling evidence is from:

- [tree_neural_speed_overnight_20260323_rapid/main_scaling_1024_2048](../outputs/tree_neural_speed_overnight_20260323_rapid/main_scaling_1024_2048)

Important caveat:

- the controller never finalized the top-level summary because of the grouped
  stage-2 deadlock
- however, the per-condition grouped summaries were written and are usable

Current best evidence from those grouped summaries:

#### At 1024 docs

Best raw test MAE:

- `teacherfirst_scorefiber_s1_f15_root0p50__internal_count_dense__judge_t1024`
- `test_root_mae = 0.2814`
- `selection_metric_value = 3.0578`

Best selection metric:

- `teacherfirst_scorefiber_s1_f15_root0p50__internal_full_dense__judge_t1024`
- `test_root_mae = 0.3054`
- `selection_metric_value = 2.3515`

Best `phi192` control:

- `teacherfirst_shared_feature_phi192__internal_count_dense__judge_t1024`
- `test_root_mae = 0.3323`
- `selection_metric_value = 4.2489`

Interpretation:

- score-fiber was better than `phi192` on both raw MAE and selection metric at
  `1024`

#### At 2048 docs

Best raw test MAE:

- `teacherfirst_scorefiber_s1_f15_root0p50__internal_count_dense__judge_t2048`
- `test_root_mae = 0.2044`
- `selection_metric_value = 2.5392`

Best selection metric:

- `teacherfirst_scorefiber_s1_f31_root0p50__internal_full_dense__judge_t2048`
- `test_root_mae = 0.2241`
- `selection_metric_value = 1.5434`

Interpretation:

- score-fiber remained the strongest route at `2048`
- the best raw MAE and the best theorem-style selection metric came from
  different score-fiber variants

### Fast-path GPU status

The latest work is about throughput, not objective redesign.

The clean fixed-fused comparison is currently running under:

- `outputs/fixed_fused_probe_20260323_1024_mig_v2`
- `outputs/fixed_fused_probe_20260323_1024_mig_v3_1epoch`

These runs are intentionally using MIG UUIDs to avoid the earlier placement
mistake.

Current status:

- probes are still running
- final `summary.json` timing outputs were not yet available at the time of
  this note

But the clean memory snapshots already show the fused backend is materially
heavier on-GPU than the generic bucket path:

- score-fiber fixed-fused at `1024`: about `2.8 GiB`
- score-fiber structure-bucket at `1024`: about `0.6 GiB`
- `phi192` fixed-fused at `1024`: about `2.3 GiB`
- `phi192` structure-bucket at `1024`: about `0.6 GiB`

Interpretation:

- the fused backend is finally filling the slices more aggressively
- the remaining question is whether that translates into the wall-clock gain we
  need

## Current Source Of Truth By Topic

### Theory

- Lean files under `lean3/FormalProofs/OPT/`
- especially the product-state and two-stage files listed above

### Modeling / training code

- `src/ctreepo/sim/core/markov_neural_operator_baselines.py`
- `src/ctreepo/sim/core/theorem_feature_route.py`
- `src/ctreepo/sim/core/markov_theorem_feature_adapter.py`

### Diagnostics

- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`

### Orchestration

- `scripts/run_tree_neural_full_doc_mig.py`
- `scripts/run_tree_neural_teacher_first_push.py`
- `scripts/run_tree_neural_teacher_first_scaling_push.py`

## What Is Actually Working

Working well enough to build on:

- teacher-first artifact split
- decomposition-aware runtime metrics
- theorem-feature adapter plumbing
- score-fiber route
- halving / async promotion infrastructure
- grouped-stage2 file-driven completion
- streaming exact evaluator
- fixed-fused backend selection for `recoverable_v4`

Working but still unsatisfactory:

- score-fiber GPU throughput
- fused eval reuse
- full end-to-end overnight throughput at larger scales

Still incomplete:

- final clean timing confirmation for fixed-fused versus structure-bucket
- a validated `1024/2048` overnight sweep on the fused path
- any serious `5000+` launch after the fused path proves itself

## Open Problems

### 1. Throughput is still not good enough

Even after major cleanup, the lane is still too slow for the intended large
nightly sweeps.

The immediate open question is:

- does `fixed_fused` provide enough wall-clock improvement to justify becoming
  the default path for `recoverable_v4`?

### 2. Exact eval is still not fully fused

The exact evaluator now uses fused state construction more effectively, but it
is not yet a complete dense fused tensor evaluator.

### 3. Prefetch / overlap is still shallow

Pinned staging and nonblocking copies are there, but this is not yet a
fully-overlapped prefetch pipeline.

### 4. Oversubscription policy is not yet truly automatic

`eval_workers_per_mig` is partially wired, but it is not yet fully driven by
worker-side autotune results.

## Recommended Next Steps

Priority order:

1. wait for the clean MIG-UUID fixed-fused probe results
2. compare `fixed_fused` versus `structure_bucket` on:
   - wall-clock
   - `gpu_reserved_mem_peak_gb`
   - `train_forward_time_s`
   - `train_backward_time_s`
3. if the fused path is clearly better, make it the default for
   `recoverable_v4` scaling runs
4. rerun a clean `1024/2048` overnight sweep on the fused path
5. only then launch `5000+`

If the fused path still does not give enough speedup, the next engineering
pressure points are:

- more aggressive fused eval
- better overlap / prefetch
- further reduction of Python-side per-doc work inside the training loop

## Practical Notes For Other LLMs

If you pick this work up:

- start from this doc plus the two stable docs linked at the top
- prefer MIG UUIDs to numeric `CUDA_VISIBLE_DEVICES`
- do not trust pre-fix grouped-stage2 controller behavior
- do not trust old probe timings that accidentally shared one physical GPU
- treat the score-fiber route as the current modeling favorite
- treat the fused-path timing probe as the current operational blocker

The main unresolved question is no longer conceptual theorem alignment. It is
whether the current score-fiber implementation can be made fast enough to
justify the larger scaling runs it now seems to deserve.
