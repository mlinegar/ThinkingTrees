# Markov Lean-Aligned Test Ladder Status

Date: 2026-05-01

This note records the current Markov recovery experiment, the Lean target it is
trying to align with, the commands used for the latest simulations, and why the
current evidence is progress but not yet a Lean-aligned recovery claim.

## Short Version

We moved back to Markov because the Lean target is explicit and sharper than the
HLL experiment. The exact theorem-facing state is not just a scalar count. It is
the Markov count sketch state

```text
(count, first, last)
```

with merge

```text
count(left) + count(right) + 1[last(left) != first(right)]
```

The experiment is testing whether the neural tree can learn a state and merge
surface that recovers this theorem-facing behavior. Root changepoint count MAE
is useful, but it is not enough. A model can predict the scalar root count while
still failing to recover the theorem state needed for compositional local laws.

Current status:

- The one-leaf parity canary passes: the tree path matches standalone FNO root
  behavior when the tree degenerates to one leaf.
- The quick multi-leaf v3 rerun completed cleanly and uses the efficient Markov
  tree batching path.
- The learned tree is competitive on scalar root MAE, especially on the
  recoverable setting.
- The theorem-state diagnostics are not yet convincing: endpoint accuracy is
  weak, exact projected root MAE is often much worse than the learned root
  readout, and phi alignment is near zero or negative.
- Therefore the current result is "root target learning works in places", not
  "Lean-aligned exact Markov state recovery".

## Lean Target

The core Lean files are:

- `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean`
- `lean3/FormalProofs/OPT/MarkovPathDGP.lean`
- `lean3/FormalProofs/OPT/MarkovSimulationValidation.lean`
- `lean3/FormalProofs/OPT/MarkovSufficiency.lean`

### Exact Sketch Monoid

`MarkovCountSketchExample.lean` defines the following state shape. This is an
ASCII sketch of the Lean definition:

```lean
inductive MarkovCountSketch (n : Nat) : Type
| empty : MarkovCountSketch n
| nonempty (count : Nat) (first : Fin n) (last : Fin n) : MarkovCountSketch n
```

The monoid merge is:

```lean
empty * b = b
a * empty = a
nonempty c1 f1 l1 * nonempty c2 f2 l2 =
  nonempty (c1 + c2 + join l1 f2) f1 l2
```

where `join l f = 0` if `l = f`, otherwise `1`.

The scalar oracle is

```lean
fstar s = MarkovCountSketch.count s
```

and the exact summarizer is the identity/pure summarizer:

```lean
gExact x = PMF.pure x
```

The important exactness facts are:

- `L1_gExact`: leaf/local law exactness for the exact sketch.
- `L2_gExact`: merge/local law exactness for the exact sketch.
- `exactSketch_root_distortion_zero`: exact root distortion is zero.
- `not_L3_gFlip`: a deliberately bad summarizer that changes count breaks a
  local law.

This is the Lean-side reason the exact Markov sketch is a clean control.

### Raw Markov Paths

`MarkovPathDGP.lean` lifts the exact sketch from theorem-domain sketches to raw
Markov paths. It defines:

```lean
encodePath : MarkovPath n -> MarkovCountSketch n
changepointCount : MarkovPath n -> Nat
countOnlyFeature : MarkovPath n -> Nat
```

The key exact facts are:

- `encodePath_append`: encoding is compositional over path concatenation.
- `count_encodePath`: the encoded sketch count equals the true changepoint
  count.
- `encodePath_congruent`: the exact encoded state is congruent under
  concatenation.
- `local_laws_of_encoded_state`: exact local laws hold for the encoded state.
- `state_exact_on_tree`: any downstream utility of the full exact state is
  preserved on a tree.
- `count_exact_on_tree`: the scalar changepoint count is exact as a consequence
  of the state result.

The negative control is also formal:

- `countOnlyFeature_not_congruent`: count-only is not a congruent feature once
  there are at least two regimes.
- `countOnly_mergeFold_counterexample`: count-only additive merge fails on a
  concrete two-leaf tree.

This matters for the experiment because a low root count MAE can be a count-only
success. It is not by itself evidence that the learned tree recovered the
theorem-facing state.

### Simulation Contract

`MarkovSimulationValidation.lean` gives the simulation contract:

- `markovPath_stochastic_policy_local_laws`: support trees inherit exact local
  laws when the theorem-facing object is the encoded exact state.
- `markovPath_exactTheoremBacked_on_support`: support trees are exact
  theorem-backed.
- `ExactMarkovPathSimulationContract.state_exact_on_support`: the merged exact
  state equals the exact encoding of the original path.
- `ExactMarkovPathSimulationContract.changepoint_count_exact_on_support`: the
  changepoint count is exact on support.
- `markov_countOnly_not_exact_on_all_trees`: count-only cannot certify the
  general topology claim.

The practical alignment target is therefore not "predict the root count". It is
"learn a state and merge whose decoded diagnostics behave like the exact
`MarkovCountSketch` state, while the runtime merge is learned rather than
hard-coded".

## Runtime Experiment Target

The modern Markov tree model under test is `unified_g`. In the intended
paper-facing interpretation:

- Leaf states are learned.
- Merge is learned.
- The exact Markov sketch is an oracle/diagnostic reference only.
- There is no hard-coded exact projected merge in the runtime tree.
- Reports must identify scalar root success separately from theorem-state
  recovery.

In the implementation, the runtime learned merge is identified as:

```text
tree_runtime_merge_kind = learned_unified_g
tree_exact_projected_merge_is_runtime_merge_rate = 0.0
```

The old `tree_score_merge_mode` metadata was misleading for this use case, so it
was removed from the modern v3 Markov pipeline/report surface. The exact
projected Markov merge remains available only as a diagnostic decoding and
oracle comparison, not as the merge implementation we are claiming to learn.

The corrected local-law estimator used in the training/audit ladder is the
usual proxy correction:

```text
proxy + R / pi * (oracle - proxy)
```

That estimator is separate from the Lean exact-state theorem. It is a way to
train or audit under partial local supervision; it does not by itself certify
that the learned representation recovered `(count, first, last)`.

## Commands Run

All commands were run from `/home/mlinegar/ThinkingTrees`.

### One-Leaf Parity Canary

This checks the degenerate tree case: one leaf at 128 tokens should match the
standalone FNO behavior.

```bash
./venv/bin/python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.fno_parity_canary_test_t128.toml \
  --output-root outputs/markov_fno_parity_canary_t128_actual_20260430_235354 \
  --device-mode cuda \
  --migs 0,1 \
  --supervision-recovery-train-docs 1024 \
  --supervision-recovery-seeds 0 \
  --max-workers 2
```

Output:

```text
outputs/markov_fno_parity_canary_t128_actual_20260430_235354
```

Scheduler status:

```text
state = completed
items_total = 6
completed_items = 6
failed_items = 0
```

Parity rows:

| scope | family | package | leaf tokens | test root MAE | val root MAE |
| --- | --- | --- | ---: | ---: | ---: |
| `r12_p079` | `official_fno` | `full100` | 128 | 2.41796875 | 2.1171875 |
| `r12_p079` | `official_fno_sumlen` | `full100` | 128 | 2.42578125 | 2.21875 |
| `r12_p079` | `tree_neural` | `full100` | 128 | 2.41796875 | 2.1171875 |
| `recoverable_v5_t128` | `official_fno` | `full100` | 128 | 1.2890625 | 1.3125 |
| `recoverable_v5_t128` | `official_fno_sumlen` | `full100` | 128 | 1.22265625 | 1.15625 |
| `recoverable_v5_t128` | `tree_neural` | `full100` | 128 | 1.2890625 | 1.3125 |

Interpretation: the tree path is not introducing an obvious one-leaf mismatch.
The `tree_neural` one-leaf row equals `official_fno` on root MAE for both
scopes. This is a necessary sanity check, not a multi-leaf theorem-state
recovery result.

### Full v3 Plan

This was the full seed-0, train-docs-1024 v3 plan inspection.

```bash
./venv/bin/python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.v3.toml \
  --plan-only \
  --device-mode cuda \
  --migs 0,1 \
  --max-workers 2 \
  --output-root /tmp/tt_markov_v3_cuda_fast_plan \
  --supervision-recovery-train-docs 1024 \
  --supervision-recovery-seeds 0
```

The plan resolved cleanly. The full actual run was started first, but hit a
pre-existing aggregation bug before the later fix:

```bash
./venv/bin/python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.v3.toml \
  --output-root outputs/markov_v3_t128_fast_recreate_20260501_001320 \
  --device-mode cuda \
  --migs 0,1 \
  --max-workers 2 \
  --supervision-recovery-train-docs 1024 \
  --supervision-recovery-seeds 0
```

That aborted root is:

```text
outputs/markov_v3_t128_fast_recreate_20260501_001320
```

It should not be treated as a result. The failure was a report aggregation
`KeyError: optimization_root_weight`. The worker summary now carries the
compatibility alias needed by the report path, and a previously failed worker
request was rerun successfully:

```bash
./venv/bin/python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --worker-task outputs/markov_v3_t128_fast_recreate_20260501_001320/supervision_recovery/attempts/20260501_001239_209398/raw/recoverable_v5_t128__train01024__full100__leaf032__g1p00__tree_neural__d0/task.request
```

### Quick Multi-Leaf Recreate

After the fix, we ran a constrained v3 recreate covering the two relevant
scopes, train-docs 1024, one seed, gamma 1.0, and two packages:

```bash
./venv/bin/python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.v3.toml \
  --output-root outputs/markov_v3_t128_fast_quick_recreate_20260501_001705 \
  --device-mode cuda \
  --migs 0,1 \
  --max-workers 2 \
  --supervision-recovery-train-docs 1024 \
  --supervision-recovery-seeds 0 \
  --supervision-recovery-depth-discount-gammas 1.0 \
  --supervision-recovery-packages full100,r100_superset_local_eq_10p0
```

The corresponding plan-only command was:

```bash
./venv/bin/python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.v3.toml \
  --plan-only \
  --device-mode cuda \
  --migs 0,1 \
  --max-workers 2 \
  --output-root /tmp/tt_markov_v3_quick_old_results_plan \
  --supervision-recovery-train-docs 1024 \
  --supervision-recovery-seeds 0 \
  --supervision-recovery-depth-discount-gammas 1.0 \
  --supervision-recovery-packages full100,r100_superset_local_eq_10p0
```

Output:

```text
outputs/markov_v3_t128_fast_quick_recreate_20260501_001705
```

Key files:

```text
outputs/markov_v3_t128_fast_quick_recreate_20260501_001705/scheduler_status.json
outputs/markov_v3_t128_fast_quick_recreate_20260501_001705/supervision_recovery/attempts/20260501_001709_694423/summary.json
outputs/markov_v3_t128_fast_quick_recreate_20260501_001705/tradeoff_report/attempts/20260501_001709_725302/summary.json
outputs/markov_v3_t128_fast_quick_recreate_20260501_001705/tradeoff_report/attempts/20260501_001709_725302/report.md
```

Scheduler status:

```text
state = completed
items_total = 16
completed_items = 16
failed_items = 0
```

Batching evidence:

```text
current_evidence_status = fast_path_engaged_and_likely_materially_helping
runtime_data_mode = resident
runtime_bucket_mode = leaf_count_auto_queue
tree_batch_pack_mode = fixed_fused
max resident_store_hits = 2000
max auto_queue_fused_batches = 3920
auto_queue_generic_fallback_batches = 0
steady_state_h2d_events = 0
```

This matters because Markov trees are very sensitive to the efficient batching
path. This run is not on the slow generic path.

## Quick Multi-Leaf Results

The rows below are the `tree_neural` rows from the quick recreate. All use
`tree_runtime_merge_kind = learned_unified_g`, and all have
`tree_exact_projected_merge_is_runtime_merge_rate = 0.0`.

| scope | package | leaf tokens | test root MAE | root direct count MAE | exact projected root MAE | learned merger gap | leaf first acc | leaf last acc | merge join acc | phi align |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r12_p079` | `full100` | 8 | 2.4007 | 2.4154 | 15.4631 | -13.0477 | 0.0845 | 0.0842 | 0.7536 | -0.0596 |
| `r12_p079` | `full100` | 16 | 2.3958 | 2.5384 | 8.8630 | -6.3247 | 0.0850 | 0.0742 | 0.7946 | -0.1147 |
| `r12_p079` | `full100` | 32 | 2.4097 | 2.4115 | 14.9891 | -12.5776 | 0.0674 | 0.1045 | 0.3776 | -0.0761 |
| `r12_p079` | `r100_superset_local_eq_10p0` | 8 | 2.3885 | 2.4060 | 7.1750 | -4.7690 | 0.0811 | 0.0842 | 0.5307 | -0.1059 |
| `r12_p079` | `r100_superset_local_eq_10p0` | 16 | 2.3926 | 2.6547 | 9.8002 | -7.1455 | 0.0732 | 0.0840 | 0.5547 | -0.0726 |
| `r12_p079` | `r100_superset_local_eq_10p0` | 32 | 2.4005 | 2.4596 | 2.9658 | -0.5062 | 0.0879 | 0.0742 | 0.6302 | -0.0297 |
| `recoverable_v5_t128` | `full100` | 8 | 1.6448 | 1.7831 | 41.9257 | -40.1426 | 0.4666 | 0.4622 | 0.9049 | -0.0155 |
| `recoverable_v5_t128` | `full100` | 16 | 1.6982 | 1.7438 | 36.1403 | -34.3965 | 0.1123 | 0.1450 | 0.9515 | -0.0098 |
| `recoverable_v5_t128` | `full100` | 32 | 0.8918 | 0.8682 | 4.0034 | -3.1352 | 0.4160 | 0.2295 | 0.5117 | 0.0295 |
| `recoverable_v5_t128` | `r100_superset_local_eq_10p0` | 8 | 1.6545 | 2.3209 | 4.0445 | -1.7236 | 0.2561 | 0.2566 | 0.7740 | -0.0154 |
| `recoverable_v5_t128` | `r100_superset_local_eq_10p0` | 16 | 1.0656 | 1.3278 | 5.4823 | -4.1545 | 0.4321 | 0.2822 | 0.7768 | -0.0429 |
| `recoverable_v5_t128` | `r100_superset_local_eq_10p0` | 32 | 0.8750 | 0.9370 | 2.2070 | -1.2700 | 0.2559 | 0.2676 | 0.4922 | -0.0399 |

For context, the one-leaf official FNO rows were:

| scope | official FNO test root MAE |
| --- | ---: |
| `r12_p079` | 2.41796875 |
| `recoverable_v5_t128` | 1.2890625 |

So the quick tree run is scalar-competitive:

- On `r12_p079`, tree rows are around 2.388 to 2.410, close to the one-leaf FNO
  reference of 2.418.
- On `recoverable_v5_t128`, the best tree rows reach about 0.875 to 0.892,
  better than the one-leaf FNO reference of 1.289.

But this is not enough for Lean alignment.

## Why We Are Not There Yet

The Lean story is about theorem-state recovery. The exact state has three
pieces: count, first endpoint, and last endpoint. The scalar count alone is
formally insufficient. The current multi-leaf quick run still mostly looks like
scalar root success rather than exact-state recovery.

Main gaps:

- The exact projected root MAE is often much worse than the learned root readout.
  This means that decoding the learned state as an exact Markov sketch does not
  usually recover a good root count, even when the learned root head does.
- `learned_merger_gap` is negative in every quick tree row. The learned root
  count head is outperforming exact projected decoding from the learned state.
  That is useful for prediction, but it argues against claiming that the state
  itself is an exact Markov sketch surrogate.
- Endpoint recovery is weak. On the structural setting, leaf first/last
  accuracies are around 0.07 to 0.10. On the recoverable setting they improve in
  some rows but are still not consistently high.
- Merge join-bit accuracy can be high in some rows, but it is not stable across
  leaf sizes and packages, and join-bit recovery alone is not full state
  recovery.
- `phi_merge_alignment` is near zero or negative almost everywhere. The only
  positive quick row is small (`0.0295`) and not enough to claim representation
  alignment.
- `leaf_direct_exact_match` and `merge_direct_exact_match` are still null in
  the quick report rows. The report now preserves the theorem-state diagnostic
  fields, but the exact-match diagnostics need to be populated consistently in
  the next run/report path.
- The quick run is intentionally small: train-docs 1024, one seed, gamma 1.0,
  two packages, and leaf tokens 8/16/32. It is a fast recreate, not the full v3
  evidence grid.

The right interpretation is:

```text
The current unified_g tree can learn useful scalar root behavior under the
modern Markov pipeline, and the pipeline now reports that the merge is learned.
It has not yet demonstrated recovery of the Lean theorem state
(count, first, last).
```

## Dimension Note

In the quick multi-leaf ladder, `state_dim = 128` and executed leaf token sizes
are 8, 16, and 32. For those multi-leaf rows, the state dimension is at least
4x the largest leaf input token count. The one-leaf parity canary uses 128
tokens and `state_dim = 128`; that canary is intentionally an FNO/tree parity
check, not the overcomplete multi-leaf recovery setting.

## Verification Commands

Focused test command:

```bash
./venv/bin/python -m pytest \
  tests/ctreepo/test_neural_operator_baselines.py::test_exact_sketch_selection_exact_merge_fallback_detects_serialized_modes \
  tests/ctreepo/test_neural_operator_baselines.py::test_unified_g_exact_projected_label_still_uses_learned_runtime_merge \
  tests/ctreepo/test_neural_operator_baselines.py::test_unified_g_c3_local_law_backprops_through_learned_merge_projector \
  tests/tree/test_markov_optimization_tradeoff_pipeline.py::test_supervision_recovery_aggregation_preserves_theorem_state_diagnostics \
  tests/tree/test_markov_optimization_tradeoff_pipeline.py::test_checked_in_v3_tradeoff_config_builds_plan \
  tests/tree/test_markov_optimization_tradeoff_pipeline.py::test_checked_in_t128_canary_config_builds_plan \
  tests/tree/test_full_doc_anchor_diagnostics.py::test_tree_stage1_expected_layout_metadata_opaque_carrier_exact_sketch \
  tests/ctreepo/test_markov_alignment_validation.py \
  tests/ctreepo/test_simulation_expectations.py -q
```

Result:

```text
24 passed, 2 warnings
```

Alignment fixtures alone:

```bash
./venv/bin/python -m pytest \
  tests/ctreepo/test_markov_alignment_validation.py \
  tests/ctreepo/test_simulation_expectations.py -q
```

Result:

```text
17 passed, 2 warnings
```

Compile check:

```bash
./venv/bin/python -m compileall -q \
  src/ctreepo/sim/core/markov_neural_operator_baselines.py \
  src/ctreepo/sim/core/markov_changepoint_ops_count.py \
  src/ctreepo/sim/core/full_doc_anchor_diagnostics.py \
  src/ctreepo/sim/core/tree_reference_presets.py \
  scripts/run_markov_optimization_tradeoff_pipeline.py \
  scripts/run_markov_supervision_recovery_parity_grid.py \
  tests/ctreepo/test_neural_operator_baselines.py \
  tests/tree/test_markov_optimization_tradeoff_pipeline.py \
  tests/tree/test_full_doc_anchor_diagnostics.py \
  tests/ctreepo/test_simulation_expectations.py \
  tests/ctreepo/test_markov_alignment_validation.py
```

Result:

```text
passed
```

Whitespace/diff check on the touched implementation and tests:

```bash
git diff --check -- \
  src/ctreepo/sim/core/markov_neural_operator_baselines.py \
  src/ctreepo/sim/core/markov_changepoint_ops_count.py \
  src/ctreepo/sim/core/full_doc_anchor_diagnostics.py \
  src/ctreepo/sim/core/tree_reference_presets.py \
  scripts/run_markov_optimization_tradeoff_pipeline.py \
  scripts/run_markov_supervision_recovery_parity_grid.py \
  tests/ctreepo/test_neural_operator_baselines.py \
  tests/tree/test_markov_optimization_tradeoff_pipeline.py \
  tests/tree/test_full_doc_anchor_diagnostics.py \
  tests/ctreepo/test_simulation_expectations.py \
  tests/ctreepo/test_markov_alignment_validation.py
```

Result:

```text
passed
```

## Next Steps

1. Populate `leaf_direct_exact_match` and `merge_direct_exact_match`
   consistently in the summary/report path.
2. Rerun the full v3 52-task grid after the `optimization_root_weight`
   aggregation fix.
3. Treat root MAE, endpoint accuracy, join accuracy, exact projected root MAE,
   direct exact match, and phi alignment as a joint acceptance gate. Do not
   promote a row as Lean-aligned from root MAE alone.
4. Use `scripts/test_markov_exact_progression.py` as the small lab for checking
   whether the merge signal is available before spending on larger grids.
5. If the state diagnostics stay flat, strengthen the theorem-state supervision
   rather than only increasing root supervision. The formal negative control
   says count-only success can hide non-congruence.

## Current Acceptance Status

| Criterion | Status |
| --- | --- |
| One-leaf tree matches standalone FNO behavior | passed in the parity canary |
| Efficient Markov tree batching is engaged | passed in the quick recreate |
| Runtime merge is reported as learned, not exact projected | passed in modern v3 rows |
| Multi-leaf root MAE is competitive | partially passed |
| Multi-leaf theorem-state diagnostics move convincingly | not yet |
| Reports distinguish scalar root success from theorem-state recovery | partially passed; exact-match fields still need population |
| Lean-aligned Markov recovery claim | not yet |
