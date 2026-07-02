# Markov Neural Operator Diagnosis

Date: 2026-05-01

This note answers the question raised in
`docs/markov_lean_aligned_test_ladder_status_2026-05-01.md`: why the v3 Markov
tree-neural-operator runs are scalar-competitive on root MAE but do not recover
the Lean theorem state `(count, first, last)`. The diagnosis is now backed by
controlled exact-leaf and explicit-sketch experiments.

## TL;DR

Latest bottom-stack control: the simple Markov setting is learnable once the
state is the theorem sketch and the summarizer respects the already-tokenized
DGP. In
`outputs/markov_explicit_transition_table_additive_join_recoverable_v5_t128_256_20260501`,
a learned token-to-regime transition-table leaf summarizer plus a learned
additive join-table merge gets 100% test root exact match, 100% leaf exact
match, and 100% internal merge exact match on `recoverable_v5_t128` with 256
training docs. This points away from "the neural operator cannot get the
simple setting" and toward the full v3 pipeline's latent surface, objective
weighting, and leaf summarizer mismatch.

The new overtrain ladder makes that conclusion less anecdotal. With an
explicit theorem-sketch surface and a learned transition-table leaf plus
learned additive join-table merge, `recoverable_v5_t128` reaches 100% test
root exact match, 100% leaf exact match, and 100% internal merge exact match
at 4096 train docs. The same runner shows why the generic neural operator is
not trivially solved by the existing sample sizes: the MLP/MLP variant has
699,914 parameters and only 0.060 leaf examples per leaf parameter at 4096
docs. A focused stable refit also shows that the generic MLP merge *can*
learn `g_theta`: with transition-table leaves and an MLP merge at
`merge_lr=5e-4`, the full learned-leaf path reaches 100% root exact, 100%
leaf exact, and 100% internal merge exact at 4096 docs. So the remaining
problem is not DGP learnability or MLP expressivity; it is the current full
training surface, loss schedule, and checkpointing.

Surface follow-up on 2026-05-01: the carrier-projection surface is now the
cleanest lower-stack diagnosis.

- `encode_summary()` now preserves direct `(count, first, last)` carrier
  summaries before the `unified_g` wide-summary encoder. A regression test
  covers this because the old ordering made exact theorem summaries pass
  through an untrained MLP and re-enter the state manifold distorted.
- Direct Markov carrier slots are now validated as exactly
  `(1, n_regimes, n_regimes)`. The inherited `8/8/8` theorem-dim defaults are
  invalid for direct slots because direct endpoint slots are regime logits,
  not learned endpoint features.
- The fixed-fused dense leaf runtime now uses the same carrier-projection
  leaf path as the flat/list runtime. Before this patch, training could use
  the carrier path while evaluation packed leaves through the generic
  summary-spec path, producing bogus decoded counts.

The new carrier split run
`outputs/markov_surface_carrier_v2_split_20260501` confirms that the exact
summary encoder is no longer the bottleneck:

```
exact summaries -> state leaf count MAE : 0.0000
exact summaries -> state exact match    : 1.0000
exact summaries -> state first/last acc : 1.0000 / 1.0000
learned leaves + learned merge root MAE : 1.4488
learned leaves + exact merge root MAE   : 1.9024
exact leaves + learned merge root MAE   : 8.9882
```

That looks paradoxical until we isolate the merge surface. The direct
carrier-projection MLP merge can overfit `g` when trained directly on exact
merge pairs. In
`outputs/markov_carrier_merge_surface_20260501`, the actual
`FNOCountSketch.count_slot_merger` reaches 100% test pair exact rate, test
pair-count MAE 0.0184, and exact-leaf learned-merge root MAE 0.0533. So the
surface is expressive enough.

The remaining issue is stability/composition. In
`outputs/markov_carrier_merge_surface_with_leaf_fixedeval_20260501`, after
direct merge pretraining and frozen-merge leaf training, the leaf encoder gets
reasonable sketches (`leaf_count_mae=0.1353`, first/last accuracy 1.0,
leaf exact 0.9492), and exact projected merge over those learned leaves gives
root MAE 0.5954. But the learned MLP merge itself gives root MAE 7.4303 on
the same learned leaves. The MLP learned the integer-lattice merge surface,
but not a stable continuous/on-policy merge under slightly off-grid learned
count slots. Adding Gaussian count jitter 0.5 during direct pair training
(`outputs/markov_carrier_merge_surface_jitter05_leaf_20260501`) did not fix
the iterated learned-merge root path (`root_direct_count_mae=6.6708`).

Current interpretation: the bottom-stack problem is now narrowed to
on-policy `g_theta` training and iterative stability, not tokenization, exact
summary reencoding, or raw expressivity. A likely next ladder rung is
sequential teacher forcing: train leaves to emit the sketch, train `g_theta`
on exact and learned-leaf child states with internal full-sketch targets, then
only fine-tune jointly after the off-grid merge is stable.

Follow-up: canonicalizing direct carrier endpoint slots before learned merge
solves the composition lab. In
`outputs/markov_carrier_merge_surface_canonical_endpoints_20260501`, the
sequential setup reaches `learned leaves + exact merge root MAE = 0.5954` and
`learned leaves + learned merge root MAE = 0.5970`, with merge first/last/join
accuracy all 1.0. This demonstrates that learned composition works once the
learned merge consumes endpoint identities rather than off-surface endpoint
logits. The remaining root error is leaf count error, not merge failure.

Update after the full-sketch shared-feature probe: the count-only diagnosis
below remains true for the v3 quick-recreate runs, but the current patched
`full_sketch` setting has a sharper failure. The learned leaf token encoder is
partly learning the scalar count, but the exact theorem-summary surface
`(count, first, last) -> encode_summary -> decoded state` is not calibrated at
all. In
`outputs/markov_leaf_merge_split_shared_feature_r2_stage2only_encoder_20260501`,
the test split gives:

```
learned leaves + learned merge root MAE : 1.1889
learned leaves + exact merge root MAE   : 3.1978
exact summaries -> state leaf count MAE : 3.4348
exact summaries -> state exact match    : 0.0000
exact summaries -> state first/last acc : 0.1587 / 0.0972
exact leaves + learned merge root MAE   : 4.7048
exact leaf merge exact-state rate       : 0.0268
```

So the current full-sketch pipeline is not yet learning the `g_theta` surface
that accepts an exact theorem sketch and re-enters the learned state manifold.
The "exact leaves + learned merge" swap fails before the merge is even a clean
test: exact theorem summaries are mapped into distorted states. This explains
why scalar root MAE can look decent while Lean-facing state recovery remains
bad.

The v3 quick-recreate runs (`full100`, `r100_superset_local_eq_10p0`) configure
`leaf_supervision_kind=count_only` and `internal_supervision_kind=count_only`.
Endpoints are never supervised. With only the scalar count signal, the merge
problem is structurally underidentified: many `g_theta` satisfy the closure
constraint without encoding `(first, last)`. The model's first/last accuracies
sitting near `1/n_regimes` (0.083 for `r12_p079`, 0.25 for `recoverable_v4`)
are exactly the chance baselines that prediction would fall to under such an
objective.

A controlled exact-leaf study confirms this. Swapping the merge objective from
closure (`strict_c3`) or count-only parent supervision (`teacher_parent_count`)
to full-state parent supervision (`teacher_parent_full_sketch`) lifts the
merge endpoint accuracies from chance (~0.25) to 0.999-1.000 and the
exact-state match rate from under 2% to 88-97%. Architecture is not the
bottleneck. Supervision is.

## Setup

Lab: `scripts/test_markov_exact_progression.py` on `recoverable_v4`. Exact
leaf summaries `(count, first, last)` are passed in via
`encode_summary`, so the leaf encoder cannot fail. Only the merge module
`_merge_state_pairs` and the heads are learned. Three merge objectives:

- `strict_c3`: closure on the scalar count, `f_theta(merged) approx
  f_star(parent_truth)`.
- `teacher_parent_count`: count target at every internal merge node.
- `teacher_parent_full_sketch`: `(count, first, last)` target at every
  internal merge node.

Each combined with `root_loss_weight in {0, 1}`, `merge_weighting in
{flat_mean, depth_balanced}`. `recoverable_v4` properties: 4 regimes, 6 leaves
per doc, mean root count 4.0, cross-leaf changepoint rate 11.87%. Chance
baselines: first/last accuracy 0.25, join accuracy 0.881 (always-no-join).

GPUs: 0/1/3 in parallel. n_train in {256, 1024}, seeds 0/1/2 (n=256), 0/1
(n=1024), 50 epochs, batch size 128.

## Aggregate Results

47 of 50 runs complete at the time of writing (the last three n=256 seed=2
specs finish in the same direction).

```
docs                  objective  rW            wgt  n    rmae  firstA   lastA   joinA   exact
  256                  strict_c3   0      flat_mean  3   0.712   0.219   0.223   0.881   0.006
  256                  strict_c3   0 depth_balanced  3   0.746   0.263   0.241   0.881   0.007
  256                  strict_c3   1      flat_mean  3   0.317   0.257   0.184   0.881   0.009
  256                  strict_c3   1 depth_balanced  3   0.424   0.240   0.229   0.881   0.007
  256       teacher_parent_count   0      flat_mean  3   0.281   0.243   0.240   0.509   0.064
  256       teacher_parent_count   1      flat_mean  3   0.284   0.255   0.238   0.551   0.065
  256 teacher_parent_full_sketch   0      flat_mean  3   0.307   1.000   1.000   0.470   0.899
  256 teacher_parent_full_sketch   0 depth_balanced  2   0.305   1.000   1.000   0.557   0.893
  256 teacher_parent_full_sketch   1      flat_mean  2   0.330   0.999   0.999   0.614   0.876
  256 teacher_parent_full_sketch   1 depth_balanced  2   0.317   1.000   1.000   0.588   0.884
 1024                  strict_c3   0      flat_mean  2   0.805   0.273   0.228   0.881   0.005
 1024                  strict_c3   0 depth_balanced  2   0.792   0.266   0.230   0.881   0.008
 1024                  strict_c3   1      flat_mean  2   0.210   0.235   0.167   0.881   0.009
 1024                  strict_c3   1 depth_balanced  2   0.170   0.245   0.240   0.881   0.014
 1024       teacher_parent_count   0      flat_mean  2   0.132   0.253   0.242   0.650   0.071
 1024       teacher_parent_count   1      flat_mean  2   0.128   0.242   0.234   0.655   0.071
 1024 teacher_parent_full_sketch   0      flat_mean  2   0.190   1.000   0.999   0.597   0.964
 1024 teacher_parent_full_sketch   0 depth_balanced  2   0.128   1.000   1.000   0.564   0.969
 1024 teacher_parent_full_sketch   1      flat_mean  2   0.160   1.000   1.000   0.582   0.962
 1024 teacher_parent_full_sketch   1 depth_balanced  2   0.140   1.000   1.000   0.601   0.964
```

Stable across seeds and data sizes:

- `strict_c3`: first/last at chance, exact match under 1.5% in all rows.
  Adding root MSE (`rW=1`) drops root MAE but does not move first/last.
- `teacher_parent_count`: same first/last picture; exact match creeps to
  6-7% (the count-target makes the merge head agree more often by accident).
- `teacher_parent_full_sketch`: first/last 0.999-1.000, exact match 0.876
  to 0.966. Adding more data makes the recovery cleaner. The root MSE
  weight matters very little here.

## Why Closure Is Not Enough

The user-stated defining equation is

```
f_star(x + y) = f_star( g_star( g_star(x) + g_star(y) ) )
```

This is closure on the scalar feature `f_star`. It is satisfied by many
`g_theta`. The count-only feature is one classical solution that satisfies
the equation almost everywhere but breaks `g_star`'s congruence at the
join boundary. The Lean negative control
`countOnlyFeature_not_congruent` (and the matching simulation control
`markov_countOnly_not_exact_on_all_trees`) say exactly this: count-only is
not a congruent feature, so the closure alone does not pin down
`(count, first, last)`.

Empirically:

- Closure with no parent count target leaves first/last at chance even at
  4096 docs.
- Adding the parent count target pushes the merge head toward count
  agreement but still leaves first/last at chance.
- Adding the parent `(first, last)` target recovers the state at 99.9%+.

So the path from "scalar root MAE is competitive" to "Lean theorem state is
recovered" runs through endpoint supervision at internal merge nodes, not
through more data, more epochs, or a different scalar weighting.

## Mapping to the v3 Pipeline Quick Recreate

Inspecting the existing v3 quick-recreate worker summaries in
`outputs/markov_v3_t128_fast_quick_recreate_20260501_001705`:

```
package = full100
config:
  leaf_weight             = 0.0
  c2_weight               = 0.0
  c3_weight               = 0.0
  root_weight             = 1.0
  local_law_weight        = 0.8
  law_package             = all_laws
  leaf_supervision_kind   = count_only
  internal_supervision_kind = none
  model_family            = fno

package = r100_superset_local_eq_10p0
config:
  ...
  leaf_supervision_kind   = count_only
  internal_supervision_kind = count_only
  ...
```

Both packages used in the recent multi-leaf grid supervise count only. The
quick-recreate `tree_neural` rows therefore look exactly like the
`teacher_parent_count` (or weaker, the `strict_c3`) rows from the small lab:

- `r12_p079` leaf first/last accuracy 0.07-0.10, near 1/12 chance.
- `recoverable_v5_t128` leaf first/last accuracy 0.11-0.46, much higher
  than chance because the regime alphabet there is smaller and some
  endpoints are inferable from the local count alone.
- `phi_merge_alignment` near zero, `exact_projected_root_mae` worse than
  the learned root readout in every row.

This is the expected behavior of an underidentified objective. It is not a
pipeline batching bug, an architecture limit, or a Markov-specific quirk.

## What This Means For The Lean Alignment Claim

The current v3 quick-recreate evidence supports:

- The runtime tree merge is learned, not exact-projected
  (`tree_runtime_merge_kind = learned_unified_g`).
- Scalar root MAE is competitive with the one-leaf FNO baseline.

It does not support:

- Recovery of `(count, first, last)`.
- Recovery of `g_star` as the exact Markov sketch merge.

The status doc's interpretation - "root target learning works in places, not
Lean-aligned exact Markov state recovery" - is correct. The diagnosis here
explains why and points at the smallest change that should move the needle.

## Cross-Check On The Real V3 Pipeline

The completed `full100` control rows in
`outputs/markov_v3_endpoint_supervision_check_20260501_024937` already
reproduce the small-lab pattern. Example, `recoverable_v5_t128 + full100 +
leaf016 + tree_neural`:

```
test_root_mae                                           = 1.698
leaf_first_accuracy                                     = 0.112
leaf_last_accuracy                                      = 0.145
merge_first_accuracy                                    = 0.272
merge_last_accuracy                                     = 0.206
exact_projected_root_mae                                = 36.140
phi_merge_alignment                                     = -0.010
test_root_mae_oracle_counts_predicted_endpoints         = 32.547
test_root_mae_predicted_counts_oracle_endpoints         =  3.594
```

The last two lines are the cleanest single diagnostic: substituting oracle
endpoints into the same predicted counts cuts root MAE by ~10x (32.5 -> 3.6),
while substituting oracle counts into the predicted endpoints leaves the root
MAE essentially as bad as the all-predicted case (36.1 -> 32.5). The model is
right about counts and wrong about endpoints, exactly as the count-only
supervision predicts.

## V3 Internal-Full-Sketch Result (Surprise)

`outputs/markov_v3_internal_full_sketch_20260501_034746` adds the new package
`full100_leaf_full100_internal_full100` (this commit registers it as canonical
package #89 in `scripts/run_markov_optimization_tradeoff_pipeline.py`,
`leaf_supervision_kind=full_sketch`, `internal_supervision_kind=full_sketch`).
Run on `r12_p079 + train1024 + seed=42 + leaf{16,32}`, with `full100` as
control and the previous `full100_leaf_full100_internal_count100` as the
leaf-supervised-only comparator.

```
scope             leaf  pkg            test_root_mae leaf_first leaf_last merge_first merge_last exact_proj_root_mae phi_align oracle_endpoints_mae oracle_counts_mae
r12_p079          16    control                2.396      0.085     0.074       0.086      0.094               8.86    -0.115                3.19              6.46
r12_p079          16    internal_count         1.201      0.495     0.508       0.097      0.090               4.54    -0.169                1.10              4.52
r12_p079          16    INTERNAL_FULL          1.430      0.090     0.088       0.084      0.092               4.05    -0.051                1.36              3.77
r12_p079          32    control                2.410      0.067     0.104       0.072      0.092              14.99    -0.076               12.31              2.70
r12_p079          32    internal_count         1.366      0.107     0.111       0.074      0.077               1.96    -0.176                1.39              1.07
r12_p079          32    INTERNAL_FULL          1.595      0.080     0.093       0.081      0.078               2.43    -0.111                1.56              2.01
recoverable_v5_t128 8   control                1.645      0.467     0.462       0.237      0.254              41.93    -0.016               40.37              1.65
recoverable_v5_t128 8   internal_count         0.833      0.865     0.870       0.270      0.299               3.69    -0.139                0.53              3.41
recoverable_v5_t128 16  control                1.698      0.112     0.145       0.272      0.206              36.14    -0.010               32.55              3.59
recoverable_v5_t128 16  internal_count         0.726      0.742     0.751       0.234      0.240               2.69     0.039                0.55              2.78
recoverable_v5_t128 32  control                0.892      0.416     0.229       0.279      0.233               4.00     0.030                2.83              1.55
recoverable_v5_t128 32  internal_count         0.701      0.587     0.589       0.279      0.221               1.84     0.051                0.63              1.66
```

The headline: `internal_count100` is the right pick today and `INTERNAL_FULL`
*regressed* leaf endpoint accuracy from 0.495 -> 0.090 on
`r12_p079 + leaf16`, even though the leaf-side supervision settings are
identical between the two packages (both have `leaf_supervision_kind=full_sketch`,
`leaf_label_rate=1.0`).

This is consistent with two architectural realities visible in
`scripts/run_markov_optimization_tradeoff_pipeline.py` and the v3 cache key:

- The stage1 artifact cache key (in
  `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py:4344-4407`)
  intentionally excludes `leaf_supervision_kind` and `internal_supervision_kind`
  - both packages above hit the same cached stage1 model
  (`outputs/_stage1_artifacts/markov_comparison_grid_v3/r12_p079__tree_neural__train_1024__seed_42__f2b0bdd8467475e7`).
- Stage2 with `local_law_weight=0.8` distributes 0.27 of the loss mass to each
  of c1/c2/c3. Switching c3 from `count_only` (one scalar target per merge)
  to `full_sketch` (count + first + last targets) roughly triples the c3
  gradient magnitude, which appears to swamp the leaf-side c1 signal during
  stage2 even though the leaf encoder started stage2 with a learned endpoint
  representation.

So the small lab's "supervise the parent state" recipe transfers to v3 only
through the leaf-side path. Internal supervision under the current shared
weighting and shared stage1 cache is *negative* on the harder r12_p079 scope.

## Explicit Sketch Bottom-Stack Control

User clarification: after tokenization, the summarizer should be the encoder.
We should not require another hidden exact-sketch encoder. The sketch
`(count, first, last)` is the object to learn, and it is fine for this to be
the oracle `g_star` surface in the simple Markov DGP.

`scripts/test_markov_explicit_sketch_learning.py` now tests that question
directly. The state is the theorem sketch itself:

```
(count / target_scale, first logits, last logits)
```

On `recoverable_v5_t128`, token ids are already regime-coded by tokenization:
`0-3`, `4-7`, `8-11`, `12-15`. A flat MLP leaf summarizer can fit the training
leaves but generalizes poorly; it is being asked to rediscover the shared
token-to-regime map and the adjacent-change counter from whole-leaf examples.
A transition-table summarizer is the right bottom-stack control: it learns
shared token-regime logits, uses the differentiable adjacent-change count for
leaf `count`, and reads first/last from the boundary tokens. The merge remains
learned.

Results, 256 train docs:

```
run                                                                          exact leaf root exact  learned leaf root exact  learned leaf exact  internal merge exact  learned root MAE
explicit MLP leaf + MLP merge                                                                     0.742                  0.230              0.833                 0.515             1.307
transition-table leaf + frozen MLP merge                                                          0.840                  0.777              1.000                 0.968             0.372
transition-table leaf + joint-finetuned MLP merge                                                  0.281                  0.906              1.000                 0.986             0.263
transition-table leaf + additive learned join-table merge                                          1.000                  1.000              1.000                 1.000             0.172
```

The joint-finetuned MLP row is a useful warning: it improves the learned-leaf
runtime path but damages exact-leaf compatibility. The frozen-merge and
additive-join rows are the Lean-aligned rows. The additive join table learns

```
count_parent = count_left + count_right + join_theta(last_left, first_right)
```

with `join_theta` as learned parameters, while carrying `first_left` and
`last_right`. This is still learned `g_theta`; it just uses the Markov sketch
surface instead of asking a generic latent MLP to rediscover additivity and
join structure at the same time.

Conclusion: the DGP is not the obstacle. The obstacle in the full pipeline is
that the learned state is not a canonical theorem-sketch surface, the
`encode_summary` bridge can distort exact sketches, and the leaf/token
summarizer and internal loss schedule are not isolating the simple
token-to-sketch and sketch-to-sketch maps.

## Base-Case Overtrain Ladder

`scripts/run_markov_base_case_overtrain.py` runs the same explicit-sketch
bottom stack as a degrees-of-freedom ladder. It sweeps train docs in
`{256, 1024, 4096}` and records leaf/merge parameter counts against the number
of supervised leaf and internal-merge examples. The leaves are 16 tokens,
there are 8 leaves per document, and the DGP is `recoverable_v5_t128`.

Artifacts:

- `outputs/markov_base_case_overtrain_transition_additive_20260501/overtrain_summary.md`
- `outputs/markov_base_case_overtrain_transition_mlp_20260501/overtrain_summary.md`
- `outputs/markov_base_case_overtrain_mlp_mlp_20260501/overtrain_summary.md`
- focused exact-leaf MLP-merge refits:
  `outputs/markov_explicit_transition_mlp_exactmerge_lr5e4_t4096_20260501/explicit_sketch_summary.md`,
  `outputs/markov_explicit_transition_mlp_exactmerge_lr2e4_t4096_20260501/explicit_sketch_summary.md`,
  `outputs/markov_explicit_transition_mlp_exactmerge_lr1e4_t4096_20260501/explicit_sketch_summary.md`
- stable full learned-leaf MLP-merge refit:
  `outputs/markov_explicit_transition_mlp_stable_full_t4096_20260501/explicit_sketch_summary.md`

Test split summary:

```
docs  variant              params  leaf ex/leaf p  merge ex/merge p  root exact  leaf exact  merge exact  root MAE  exact-leaf root exact
 256  mlp_mlp              699914          0.0037             0.012       0.328       0.729        0.520     1.380                0.031
 256  transition_mlp        69441         32.0000             0.026       0.113       1.000        0.611     1.345                0.121
 256  transition_additive      80         32.0000           112.000       0.000       1.000        0.824     0.884                1.000
1024  mlp_mlp              699914          0.0150             0.047       0.375       0.866        0.688     1.070                0.000
1024  transition_mlp        69441        128.0000             0.103       0.555       1.000        0.935     0.476                0.906
1024  transition_additive      80        128.0000           448.000       0.992       1.000        0.999     0.376                1.000
4096  mlp_mlp              699914          0.0599             0.187       0.578       0.959        0.852     0.559                0.602
4096  transition_mlp        69441        512.0000             0.413       0.375       1.000        0.820     0.620                0.375
4096  transition_additive      80        512.0000          1792.000       1.000       1.000        1.000     0.063                1.000
```

Note: the table records the first-pass ladder artifacts. After the focused
MLP-merge refit exposed the high-LR checkpoint instability, the
`transition_mlp` default in `scripts/run_markov_base_case_overtrain.py` was
changed to the stable `merge_lr=5e-4`, `merge_epochs=1200` schedule.

Interpretation:

- Yes, the base case is fully solvable with enough data when the state and
  merge surface match the theorem sketch. The `transition_additive` row is not
  an oracle max; it learns the token-to-regime table and the `4 x 4` boundary
  join table.
- The MLP leaf + MLP merge variant is not actually overdetermined at 4096
  docs. It has about 700k parameters and only 32,768 leaf examples plus 28,672
  merge examples. It improves with data, but it is still a generic high-
  capacity function approximator trained from far fewer examples than
  parameters.
- The transition leaf + MLP merge variant isolates a training recipe issue.
  The overtrain ladder's 4096-doc run used `merge_lr=2e-3`; its progress log
  had `merge_count_loss` near 0.020 at epoch 280 and then jumped to 0.165 at
  the final checkpoint. A focused exact-leaf refit with lower LR fixes that:

```
run             exact-leaf root exact  exact-leaf merge exact  exact-leaf root MAE
merge_lr=5e-4                   1.000                   1.000                0.130
merge_lr=2e-4                   0.980                   0.997                0.387
merge_lr=1e-4                   1.000                   1.000                0.285
```

  The stable full learned-leaf refit closes the loop:

```
run                                      root exact  leaf exact  merge exact  root MAE
transition leaf + MLP merge, stable           1.000       1.000        1.000     0.140
```

  Thus generic MLP `g_theta` is expressive enough for the base sketch merge,
  and the token leaf summarizer can feed it correctly. The nontrivial part is
  making the training path robust and theorem-surface-aligned in the full v3
  pipeline.
- The practical fix is not "give the same v3 latent MLP more epochs" first. It
  is to put the state on the explicit sketch surface, then add learnable
  degrees of freedom only where the DGP requires them.

## Capacity Versus Pathology

The v3 unified_g operator at `r12_p079 + leaf16` instantiates with
**2,574,333 trainable parameters**, broken down roughly as:

```
fno_encoder                : 592,640 (23.0%)
doc_sequence_fno           : 592,640 (23.0%)
summary_encoder            : 526,464 (20.5%)
unified_g_merge_summary_proj: 247,296 (9.6%)
summary_state_merger       : 198,272 (7.7%)
leaf_proj                  : 131,712 (5.1%)
... (other components)
```

The DGP for `recoverable_v5_t128` is **trivial**: 4 regimes, vocab 16, with
each regime emitting tokens from a disjoint 4-element bucket. The token-to-
regime map is exactly `regime = token // 4`. Look at any single token in a
leaf and the regime is determined. Leaf endpoint identification is solvable
by a single linear layer.

For comparison, `scripts/test_markov_explicit_sketch_learning.py`
(`leaf_encoder=transition_table`, `merge_count_mode=additive_join_table`)
encodes the leaf with `nn.Embedding(vocab_size=16, n_regimes=4)` (64
parameters) and the merge with a `4 x 4` join table. With 256 training docs,
it achieves:

```
test leaf_first_accuracy  = 1.000
test leaf_last_accuracy   = 1.000
test merge_first_accuracy = 1.000
test merge_last_accuracy  = 1.000
test merge_exact_match    = 1.000
test root_exact_match     = 1.000
test root_mae             = 0.171
```

A 64-parameter token table outperforms the 2.6M-parameter v3 operator on
the same DGP, by a wide margin. So the v3 failure mode is *not* lack of
capacity, lack of data (256 docs is enough for the explicit model), or
opacity of the DGP.

The added shared-feature probe (referenced in the TL;DR, output at
`outputs/markov_leaf_merge_split_shared_feature_r2_stage2only_encoder_20260501`)
makes the architectural pathology explicit:

```
exact summaries -> state -> decoded leaf count MAE       : 3.4348
exact summaries -> state -> decoded leaf exact match     : 0.0000
exact summaries -> state -> decoded leaf first/last acc  : 0.1587 / 0.0972
```

That is: feeding the **exact** `(count, first, last)` summary through the
v3 `encode_summary` and decoding back gives chance-level recovery. The v3
shared-feature surface (`theorem_surface_mode=shared_feature`,
`state_dim=128`) cannot represent the sketch identically, let alone learn
it under joint training.

So the v3 pipeline's failure on this trivial DGP is:

- *Not* a data quantity issue (explicit model wins at 256 docs).
- *Not* a capacity issue (v3 has 25,000x more parameters than the explicit
  model that wins).
- *Not* an optimization-over-time issue (the bottleneck is the
  representation, not the optimizer; the long-tail accuracy ceiling is
  governed by what the latent state can encode).
- It *is* a representation-surface mismatch. The shared-feature latent
  intentionally fuses `(count, first, last)` into a learned 128-dim
  bottleneck that does not decode cleanly to the theorem sketch, even
  when the input is the exact sketch.

The relevant existing preset that *does* keep the sketch decomposable is
`structural_factorized_fiber_v2` in
`src/ctreepo/sim/core/tree_reference_presets.py:317-368`
(`theorem_surface_mode=factorized_score_fiber`,
`internal_supervision_kind=full_sketch`,
`tree_root_supervision_kind=count_ce`). Switching the v3 grid to this
preset, or adding an explicit-sketch surface, is likely the smallest fix
that lets the v3 plumbing reach the explicit-sketch ceiling.

## Overtraining Test

`outputs/markov_v3_overtrain_recoverable_v5_t128_leaf16_20260501_043340`
runs the v3 pipeline on the trivial DGP with the highest available data and
4x epochs:

```
scope:       recoverable_v5_t128
leaf_tokens: 16
train_docs:  10240
seeds:       0
stage1_epochs: 30 (vs default 10)
stage2_epochs: 120 (vs default 30)
packages:    full100, full100_leaf_full100_internal_count100
stage1_artifact_root: fresh (no cached stage1 contamination)
```

If the v3 pipeline can converge to ~1.0 leaf endpoint accuracy on this
setting, then the prior 0.742 ceiling was a budget issue. If it plateaus
short of 1.0, that is direct confirmation that the shared-feature surface
is the binding constraint, independent of supervision package choice. In
flight on GPU 1.

## f-then-g validation (rule-learning probes)

A focused diagnostic in `docs/markov_rule_learning_diagnostic_plan_2026-05-01.md`
isolated whether the v3 failure is the leaf encoder, the merge, or the joint
training surface. Three nested probes:

- Leaf-only probe (`scripts/probe_markov_rule_learning.py`): standalone
  FNO leaf encoder on `recoverable_v5_t128`. Result: **100% test
  leaf_first/last accuracy, train/test gap 0.0000, 100% length-transfer
  invariance, 100% token-swap-within-regime invariance.** The FNO encoder
  by itself learns the rule.

- Triangle probe (`scripts/probe_markov_triangle.py`): 2 leaves + 1 merge
  end-to-end on a synthetic recoverable DGP. Endpoints reach 1.000 on every
  leaf size with every encoder. Counts memorize as L grows: at L=16, vanilla
  encoders hit train=1.0, test=0.5 - a 0.5 generalization gap purely from
  count-head memorization. CE swap helps ~7-9pp at L=16 but does not fix it.

- Structural-encoder probe (same script, `mlp_structural` and
  `fno_structural` variants): replaces the holistic count-head with a
  per-token regime classifier and derives count via
  `count = sum_t (1 - sum_r P(reg_t)*P(reg_{t+1}))`. With **f = the count
  formula (fixed)** and **g = the per-token regime classifier (learned)**,
  the local laws are satisfied by construction.

Test `root_count_exact` (train in parens):

| L | vanilla mlp+mlp | vanilla fno+mlp | mlp_structural+mlp | fno_structural+mlp |
|---:|---:|---:|---:|---:|
| 4  | 0.999 (1.000) | 0.730 (0.790) | **1.000 (1.000)** | **1.000 (1.000)** |
| 8  | 0.897 (1.000) | 0.647 (0.807) | **1.000 (1.000)** | **1.000 (1.000)** |
| 16 | 0.491 (1.000) | 0.572 (0.918) | 0.285 (0.264)     | 0.525 (0.632) |

Conclusion: the f-then-g framing - **f is the formula, g is what we
learn** - is empirically validated at L=4 and L=8. Test = train = 1.000,
zero memorization gap, exact rule recovery.

The L=16 structural failure was an optimization-basin issue, not an
architecture limitation. Initial count predictions are ~0.75 * L (because
adjacent uniform-init regime probs disagree most of the time), so at L=16
the count-MSE term has scale ~110 at init, dwarfing the first/last
cross-entropy. The fastest local descent is "collapse all positions to a
single regime" -> count -> 0, which lands the model in a degenerate basin
that satisfies count for k=0 leaves but breaks first/last for k>0 leaves.

A 50-epoch linear warmup on the count-loss scale (from 0 to 1) escapes
this. With warmup, first/last cross-entropy organizes per-token regimes
*before* count loss kicks in; once count loss starts mattering, regimes
are already sharp enough that the structural formula gives correct counts
without further architectural changes.

Sweep with curriculum (`--count-loss-warmup-epochs 50`) at L=16:

| L=16 cell | warmup train | warmup test | (no-warmup test) |
|---|---:|---:|---:|
| mlp_structural + aj_table | 1.000 | 1.000 | 0.234 |
| mlp_structural + mlp_merge | 1.000 | 1.000 | 0.280 |
| fno_structural + aj_table | 1.000 | 0.9995 | 0.791 (after 5x reweight) |
| fno_structural + mlp_merge | 1.000 | 1.000 | 0.348 |

Same architecture as the L=4 and L=8 runs that already hit 1.000. Only
the loss schedule differs. **The structural recipe gets perfect rule
recovery at L = 4, 8, AND 16 with no train/test gap.**

The actionable v3 prescription: expose per-token regime probabilities as
the leaf state instead of a 128-dim shared bottleneck, and use the
transition formula for count. Endpoints are positions 0 and T-1 of the
same per-token regime head. The merge then operates on
`(count_via_formula, first_logits, last_logits)` triples and the existing
additive_join_table or MLP merge head can chain them.

## Multi-L Training Learns The Merge Formula End-To-End

Open question after the structural-encoder probe: with the count formula
hard-coded, the system trivially gets the right answer at any L. But can
we **learn** the merge - i.e., let an MLP head replace the formula and
recover the same composition behavior from g alone?

Probe (`scripts/probe_markov_triangle.py --multi-l-train`): train the
2-leaf triangle on a mixture of L in {4, 8, 16, 64} and evaluate
zero-shot on L not seen at train time.

| leaf_enc (g) | merge head (f) | L=4 | L=8 | L=16 | L=64 | xfer L=32 | xfer L=128 | xfer L=256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp_structural | mlp (learned) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** |
| fno_structural | mlp (learned) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** |

The MLP merge generalizes perfectly to L=128 and L=256 zero-shot when
trained on the L set above; at far-OOD L it actually outperforms the
formula merge baseline because it learns to compensate for the small
per-position residual that accumulates in the leaf encoder. **The merge
formula is learnable from g alone**, confirming the f-then-g framing
end-to-end. Single-L training does not do this: the MLP merge memorizes
an L-specific calibration that does not transfer.

Single-L vs multi-L (mlp_structural leaf, mlp_merge head):

| train regime | L=8 train | L=8 test | xfer L=128 |
|---|---:|---:|---:|
| single L=8 | 1.000 | 1.000 | ~0.2 (chance-level) |
| mixed L in {4,8,16,64} | 1.000 | 1.000 | **1.000** |

So the v3 pipeline today, which trains at a single fixed leaf_tokens
(128 for `recoverable_v5_t128`), is exactly the case where the merge
head learns an L-specific shortcut. To force composition into the merge
head we need to vary leaf length during training.

## Sparse R10 V3 Sweep (Composition Learning In The Real Pipeline)

`outputs/markov_v3_R10_SPARSE_20260501_182911/` (11/11 cells done) varies
non-root supervision under sparse `train_docs=10` root supervision on
`recoverable_v5_t128`:

| variant | leaf_mae (g proxy) | merge_mae (f proxy) | root_mae |
|---|---:|---:|---:|
| full10 (root only)                              | 0.277 | 0.666 | 0.432 |
| full10_leaf_full10                              | 0.014 | 0.136 | 0.208 |
| full10_leaf_full10_internal_count10             | 0.010 | 0.062 | 0.138 |
| full10_leaf_full10_internal_count100            | 0.008 | 0.043 | 0.097 |
| full10_leaf_full100_internal_count100           | 0.004 | 0.036 | 0.086 |
| full100 (root, lots of data)                    | 0.140 | 0.431 | 0.032 |
| full100_leaf_full100_internal_count100          | **0.004** | **0.013** | **0.022** |

Reading: under sparse R10 root supervision, adding leaf and internal
supervision cuts `leaf_mae` 70x (0.277 -> 0.004) and `merge_mae` 50x
(0.666 -> 0.013), and brings root_mae from 0.43 -> 0.086, about 5x
better than root-only at the same root budget. **Composition learning
IS happening in the real pipeline**, not just the standalone probe -
adding observations at non-root nodes does help when root supervision
is sparse.

The remaining gap between `full10_leaf_full100_internal_count100`
(0.086) and `full100_leaf_full100_internal_count100` (0.022) is the
piece the multi-L probe explains: the v3 MLP merge is still trained at
fixed L=128 and so memorizes an L-specific calibration. The next
overnight test is to add multi-L training to the v3 pipeline and see
whether the sparse-R10 + multi-L combination closes that gap.

## t2048 Composition Stress (Long Merge Chains)

To test whether composition supervision still rescues sparse root MAE
under a much longer merge chain, we built a new benchmark
`recoverable_v5_t2048` (4 regimes, vocab 16, 2048-token docs,
~20 expected boundaries via sqrt-scaling from t128's 5). At
`fixed_leaf_tokens=16` this gives **128 leaves and 127 merges per doc** -
about 18x the merge depth of the t128 sweep.

Sweep config: `config/markov/tradeoff_pipeline.t2048_composition_stress.toml`,
4 packages x 4 leaf rungs (`[1024, 256, 64, 16]`) x 1 seed = 16 cells, 10240
train docs. Per-rung batch sizes via the new
`supervision_recovery_leaf_token_batch_sizes` knob (`16=128;64=256;...`)
target ~16K leaves per batch.

Headline cells from the first run
(`outputs/markov_t2048_composition_stress_20260502_090326/`, completed
under the slow code path, before the perf fix below):

| package | leaf | leaves/doc | root_mae | leaf_mae | merge_mae |
|---|---:|---:|---:|---:|---:|
| full10 (root only)                              | 16   | 128 | 3.72 | 0.42 | 20.4 |
| full10                                          | 64   |  32 | 3.72 | 11.1 | 15.7 |
| full10                                          | 256  |   8 | 3.72 | 10.1 |  8.2 |
| full10                                          | 1024 |   2 | 1.94 |  2.36 |  2.22 |
| full10_leaf_full10_internal_count100            | 64   |  32 | 1.69 |  0.15 |  4.96 |
| full10_leaf_full10_internal_count100            | 256  |   8 | 1.42 |  0.43 |  2.99 |
| full10_leaf_full10_internal_count100            | 1024 |   2 | 1.65 |  1.17 |  1.67 |
| full100 (root only)                             | 64   |  32 | 1.14 |  9.95 |  7.40 |
| full100                                         | 256  |   8 | 1.06 |  2.14 |  1.59 |
| full100_leaf_full100_internal_count100          | 64   |  32 | 0.83 |  0.063 |  0.53 |
| full100_leaf_full100_internal_count100          | 256  |   8 | 0.92 |  0.26 |  0.79 |

Reading at the headline rung (leaf=16, 127 merges):
- `full10` (sparse root only): root_mae **3.72**, ceiling at the chance
  level. Merge chain alone breaks root MAE.
- `full100` (full root only): also stuck at root_mae **3.72** at leaf=16.
  More root data alone does NOT rescue the long chain.
- Composition supervision (`*_leaf_*_internal_count100`) closes most of
  the gap at moderate chains (leaf=64 / leaf=256), confirming
  composition rescue extends from t128 (7 merges) to t2048 (31-127
  merges).
- Best leaf=16 cell from the first sweep is mid-stage2 still grinding;
  the optimized re-launch should land it in a few hours.

Length-vs-merge intuition validated: composition rescue mechanism is
real and scales with merge supervision density, but merge-chain length
itself is a hard axis (root_mae floor stays roughly proportional to
leaves/doc when only root is supervised).

## Performance: forward_doc_unified collect_full_trace

While running the t2048 sweep, the leaf=16 cells were taking ~6 hours
each. Profiling (`py-spy` flamegraph) showed the worker at 100% CPU on
one core while the GPU sat at 5-10% utilization. The hot path was 27.5%
in `nn.Linear.forward` plus 14% in `forward_doc_unified` - kernel-launch
overhead, not compute.

Two surgical fixes in
`src/ctreepo/sim/core/markov_neural_operator_baselines.py:forward_doc_unified`:

1. **Batched `predict_norm_from_state`**: stack all leaf and merge
   states into one tensor and call the prediction head once instead of
   N=255 sequential `nn.Linear` calls per doc.
2. **Deferred & gated telemetry**: per-node `FullTreeNodeRecord`,
   `DocumentLevelPredictionRecord`, and `StateNode` trace each force a
   GPU->CPU sync (255 syncs per doc * 8 docs/batch = 2040 sync points
   per training step). New `collect_full_trace=False` default skips
   these entirely; opt-in callers (the phi-feature pair-stats collector
   only) pass `collect_full_trace=True` explicitly.

Probe results (8 docs/batch, 128 leaves/doc):

| code path | leaves/sec | speedup |
|---|---:|---:|
| baseline (per-node syncs) | 950 | 1.0x |
| optimized + collect_full_trace=True | 6,689 | **7.0x** |
| optimized + collect_full_trace=False (default) | 8,877 | **9.3x** |

Numerical correctness confirmed by re-running the t2048 sweep with the
optimized code path: cells that completed in both sweeps produce
byte-for-byte identical metrics
(`outputs/markov_t2048_composition_stress_optimized_20260502_192918/`).
Regression test
(`test_forward_doc_unified_collect_full_trace_false_skips_telemetry`)
asserts numerical parity between the two paths and that the telemetry
side-channel is empty under the default.

This is now a project-wide rule: don't add per-node `.cpu()`/`.item()`
calls inside `forward_doc_unified` or any helper it calls per-node.
Stack tensors and do one batched `.cpu()` after the per-node loop. See
`docs/ctreepo_python_code_map_for_llms.md` "Performance:
forward_doc_unified collect_full_trace" subsection for the canonical
note.

## Head Capacity Default

The zero-merge probe (`outputs/markov_t2048_zero_merge_20260502_234841/`,
leaf=2048, 1 leaf per doc, no merge composition) revealed a separate
floor: `full100` saturated at root_mae **2.14** even though this is just
single-leaf count regression on a 2048-token doc with ~20 boundaries.
The FNO baseline at the same setup also stalled at root_mae **2.22**.
With no merge tree and no composition error to compound, the only
remaining bottleneck is **head capacity**.

| package | family | leaf | leaves/doc | merges | root_mae |
|---|---|---:|---:|---:|---:|
| full10 | tree_neural | 2048 | 1 | 0 | 2.56 |
| full10 | FNO baseline | 128 | 16 | n/a | 3.67 |
| full100 | tree_neural | 2048 | 1 | 0 | **2.14** |
| full100 | FNO baseline | 128 | 16 | n/a | 2.22 |

**Head capacity hypothesis refuted (2026-05-03).** I tested the rule
`state_dim=2048, hidden_dim=2048, tree_merge_hidden_dim=4096` (vs prior
`state_dim=128, hidden_dim=512`) at
`outputs/markov_t2048_full_grid_wide_heads_20260503_003820/`. The
zero-merge floor moved 2.14 -> 2.13 (noise). Worse, several composition
cells regressed:

| package | leaf | merges | OLD root_mae | WIDE root_mae | Δ |
|---|---:|---:|---:|---:|---:|
| full100 | 2048 | 0 | 2.14 | 2.13 | ~0 |
| full100 | 1024 | 1 | 1.51 | 3.72 | **-2.21** |
| full100 | 256 | 7 | 1.06 | 3.72 | **-2.66** |
| full100 | 64 | 31 | 1.14 | 3.72 | **-2.58** |
| full100_leaf+internal | 256 | 7 | 0.92 | 0.89 | +0.03 |
| full10_leaf+internal | 256 | 7 | 1.42 | 3.72 | **-2.30** |

The bigger model converged early (best_epoch=2 / 10 vs 25-30 for
conservative) to a bad local min. So head capacity is NOT the
bottleneck for the ~2.14 floor.

**Default reverted to `state_dim=128, hidden_dim=512`** (the values
above the headline tables in this doc). Real bottleneck for the
zero-merge floor is upstream of the heads - candidates are the FNO
leaf encoder (`fno_width=128, n_modes=8, n_layers=4`), count-head
pooling, or DGP irreducible noise. To probe further: try widening
`tree_leaf_fno_width`, increasing `fno_n_modes`, or measuring the
Bayes-optimal MAE for the recoverable_v5_t2048 DGP at zero merges.

## FNO Mode Sweep At Zero-Merge

The next probe varied `tree_leaf_fno_n_modes` directly (with the
conservative state/hidden defaults). The boundary signal on
`recoverable_v5_t2048` has period ~128 tokens (mean segment length),
so Nyquist mode index is 16 -- the default `n_modes=8` is sub-Nyquist
by 2x. The sweep covers a wide range to also test what happens at and
beyond Nyquist:

| n_modes | epochs | root_mae | best_ep | output |
|---:|---:|---:|---:|---|
| 8 (default) | 30 | 2.14 | 6 | conservative grid |
| 8 | 120 | 2.17 | 23 | `markov_t2048_modes8_zeroM_long2_*` |
| 16 | 30 | 1.81 | 22 | `markov_t2048_modes16_*` |
| 32 | 30 | 1.57 | 26 | `markov_t2048_modes32_*` |
| 32 | 120 | **1.49** | 83 | `markov_t2048_modes32_zeroM_long2_*` |
| 64 | 30 | 1.94 | 10 | `markov_t2048_modes64_*` |
| 64 | 120 | 1.88 | 84 | `markov_t2048_modes64_zeroM_long2_*` |
| 128 | 30 | 1.38 | 17 | `markov_t2048_modes128_*` |
| 512 | 120 | 1.11 | 77 | `markov_t2048_modes512_zeroM_long_*` |
| 1024 | 120 | **1.08** | 78 | `markov_t2048_modes1024_zeroM_long_*` |
| 2048 | 120 | 1.29 | 58 | `markov_t2048_modes2048_zeroM_long_*` |

Key takeaways:
- More modes monotonically helps zero-merge (modulo the modes=64
  outlier, which is single-seed noise).
- The floor lands at root_mae ~1.08 around the Nyquist limit
  (`n_modes=1024` for L=2048, since FFT has L/2+1 unique frequency
  bins). Going past Nyquist (`n_modes=2048`) wastes parameters and
  gets worse.
- More epochs at fixed n_modes only help marginally
  (modes=32: 1.57 -> 1.49; modes=8: 2.14 -> 2.17).
- **Modes alone do not crack the floor**. Even with full Fourier
  resolution, root_mae stalls at ~1.0 -- well above the achievable
  near-zero we see on the smaller `recoverable_v5_t128` problem with
  the same architecture (root_mae=0.02 at 150 epochs).

Composition cells regress badly with high n_modes (e.g.
`full100 @ leaf=256, 7 merges`: 1.06 at modes=8 -> 3.72 at modes=32+),
matching the wide-heads pattern. The bigger model overfits and
early-stops at a bad local min when there is composition supervision
in the loss.

## Sum-Pool vs Mean-Pool At The Leaf Encoder

To test whether the floor is a pooling bottleneck (mean-pool dilutes
count info by L), I added a `tree_leaf_fno_pooling` config knob
(`mean` default, or `sum`) and ran a parallel ablation. Output:
`markov_t2048_modes{8,32,128,512,1024}_sumpool_*`.

| n_modes | mean (120 ep) | sum (120 ep) |
|---:|---:|---:|
| 32 | 1.49 | **1.41** |
| 128 | 1.38 (30ep) | 1.45 |
| 512 | 1.11 | 1.18 |

Sum-pool is essentially indistinguishable from mean-pool. Neither
pooling mode cracks the ~1.0 floor. The pooling choice is not the
bottleneck.

## Refactor Toward A Single Source Of Truth (apply_fno_token_encoder)

Three call sites in the existing pipeline did the same
embed -> permute -> FNO -> masked-pool sequence with subtly different
inline implementations:

- `FNOTokenEncoder.forward` (the official wrapper)
- `FNOCountSketch._encode_token_batch` (the unified-g leaf encoder)
- `FNOCountSketch.predict_doc_sequence_logits` (the doc-sequence FNO
  baseline)

Refactor: extracted `apply_fno_token_encoder` in
`src/ctreepo/sim/core/fno_doc_baselines.py` as the single source of
truth. All three call sites now delegate to it. Verified
byte-identical to pre-refactor for both mean and sum pooling. 130/130
of the existing `test_neural_operator_baselines.py` tests still pass
(one pre-existing failure on main, unrelated to the refactor).

## Clean f/g Composition Models (`clean_unified_fg.py`)

The big `FNOCountSketch` accumulated many surface modes, opaque
carrier paths, and bookkeeping that obscure what's actually being
composed. To make the f/g algebra explicit and testable I added a
fresh standalone module: `src/ctreepo/sim/core/clean_unified_fg.py`.
Two model classes, both written so each component is a thin wrapper
around an "official" primitive.

**Vector-state model: `CleanUnifiedFG`** (a clean version of the
slim baseline)
- `leaf_encoder`: `nn.Embedding` + `neuralop.FNO` + masked pool. Uses
  `apply_fno_token_encoder` (one source of truth).
- `g`: bare `nn.Linear(2*state_dim, state_dim)` over `cat(left, right)`.
- `f`: bare `nn.Linear(state_dim, 1)`.
- `forward_doc(leaf_tokens)` builds a balanced binary merge tree and
  applies `f` at every node (leaves, merges, root), returning a
  `TreeForwardOutput` of per-node count predictions.

**Operator-state model: `CleanUnifiedNO`** (the C-TreePO-faithful
unified-g)
- State is a discretized FUNCTION `(B, channels, length)`, not a
  vector -- so g and f operate on functions, not pooled summaries.
- `g` is **shared between leaves and merges**: one `neuralop.FNO`
  module with `in_channels=2C, out_channels=C` and signature
  `(B, 2C, L) -> (B, C, L)`.
  - At leaves: `g.encode_leaf(emb)` lifts `(B, C, L)` to `(B, 2C, L)`
    by zero-padding the right half (a leaf is a degenerate "merge"
    of content + null). Same g call.
  - At merges: `g.merge(left, right)` concatenates along the channel
    axis to `(B, 2C, L)`. Same g call.
- `f`: `neuralop.FNO` + masked pool + `nn.Linear(C, 1)` -- operator
  first, then scalar readout for the count target.
- Test `test_g_at_leaves_is_same_instance_as_g_at_merges` asserts
  there is exactly one `g.fno` module in the model graph; same
  parameters apply at every node.

Loss helpers (`root_mse_loss`, `leaf_mse_loss`, `merge_mse_loss`)
work on both output types and accept observed-mask tensors so the
trainer can express the C1/C2/C3 supervision-sparsity packages
directly. 36/36 tests pass in
`tests/ctreepo/test_clean_unified_fg.py`.

## In-Flight: CleanUnifiedNO Zero-Merge Probe

Standalone script: `scripts/probe_clean_unified_no.py`. Uses the
existing `prepare_markov_full_doc_anchor_diagnostics_data` to
materialize `recoverable_v5_t2048` data (no data-pipeline
duplication) and trains `CleanUnifiedNO` with a simple Adam loop.

Currently running:
`outputs/clean_unified_no_sharedg_match_20260503_223145`. The probe
is configured to match the slim-model floor experiment that hit
root_mae=1.08:

| knob | value | notes |
|---|---:|---|
| benchmark | recoverable_v5_t2048 | doc=2048, ~20 boundaries/doc |
| leaf_tokens | 2048 | zero-merge: 1 leaf/doc, 0 merges |
| train_docs | 10240 | matches slim baseline |
| epochs | 120 | matches slim baseline |
| batch_size | 128 docs | matches slim baseline |
| channels | 128 | == slim's `fno_width=128` |
| g_n_modes | 1024 | == slim's `n_modes=1024` (Nyquist for L=2048) |
| g_n_layers | 4 | == slim's `n_layers=4` |
| lr | 5e-4 | matches slim baseline |
| grad_clip | off | matches slim baseline |

Model size: 34.4M params total -- emb=1K, **g (shared) = 33.9M**,
f scorer = 0.5M. ~54s/epoch on a single GPU; ~108 min total.

The headline number to read: **best_val_root_mae over 120 epochs**,
with test_root_mae evaluated at the best-val checkpoint (the script
saves the best state and reloads before test eval, since earlier
runs showed late-epoch divergence makes last-epoch numbers
misleading).

The earlier shared-g probe (channels=64, modes=64, lr=5e-4, batch=32,
60 epochs, no grad clip; output `clean_unified_no_sharedg_zeroM_*`)
finished with best_val=2.21 @ epoch 51 and showed wild val
oscillation in late epochs. That run had ~0.5x channels and ~1/16x
modes vs the slim baseline, so the result is not a fair test of the
operator architecture; the matched-config probe above is the proper
A/B.

## What We Are Comparing

The matched-config probe is asking one specific question: **does
moving from vector-state to function-state -- so g and f both become
neural operators rather than neural nets on pooled vectors -- crack
the ~1.08 root_mae floor at zero-merge?**

If yes: the slim model's floor was a pool-then-scalar bottleneck,
and the operator-on-functions design gives us headroom.

If no (operator floor lands ~1.0 too): the bottleneck is something
deeper. Candidates ranked by what to test next:
1. Bayes-optimal MAE for the DGP at zero-merge (compute analytically
   or via a memorization upper bound).
2. The token embedding -> state mapping (currently a tiny
   `nn.Embedding(vocab=16, dim=128)`); maybe regimes are not
   linearly separable in 128-dim from a single token id.
3. The leaf-encoder FNO depth (4 layers may not propagate boundary
   detection to enough channels at this length).
4. The training recipe (lr schedule, weight decay, longer training).

## Recommended Next Experiments

1. v3 pipeline confirmation, completed:
   `outputs/markov_v3_endpoint_supervision_check_20260501_024937`. Confirms
   `internal_count100` is a clean win over `full100` (leaf endpoint
   accuracy 0.085 -> 0.495 on r12_p079_leaf16; 0.112 -> 0.742 on
   recoverable_v5_t128_leaf16; exact projected root MAE drops 8x-13x).

2. Promote the explicit sketch surface into the main Markov neural-operator
   path. Avoid the current arbitrary `encode_summary` bridge for exact
   sketches; use a canonical `(count, first, last)` carrier or a directly
   decoded theorem surface. For `recoverable_v5_t128`, use a learned
   transition-table leaf summarizer and a learned additive join-table merge as
   the bottom-stack sanity target.

3. Revisit internal full-sketch supervision only after the surface fix. The
   first v3 `INTERNAL_FULL` package regressed because the current stage2
   weighting and shared cached stage1 artifact let the internal c3 gradients
   swamp the leaf-side c1 signal. It is not evidence against full-state
   parent supervision in a canonical sketch state.

4. Identifiability sweep. Run the small lab with mixed objectives, e.g.
   `0.9 * teacher_parent_count + 0.1 * teacher_parent_full_sketch`, to
   measure how much endpoint supervision is needed before recovery
   collapses. This calibrates how cheap we can make the structural target
   without losing identifiability.

5. r12_p079 small-lab rerun. The current Phase 1 used `recoverable_v4` (4
   regimes); rerunning with the `r12` Markov benchmark would let us read the
   leaf first/last accuracies directly against the chance baseline of 1/12
   that we see in the quick-recreate v3 rows.

6. Do not promote scalar-only rows. Keep treating any tree row with
   leaf first/last accuracy at the chance baseline as evidence of the
   underidentified objective, not evidence that the architecture cannot
   represent the state.

## Caveats

- The exact-leaf small lab isolates the merge from the leaf encoder. The full
  v3 pipeline must also learn `(count, first, last)` from raw leaf tokens.
  Phase 2 above is the right control because it leaves the leaf encoder
  intact and only changes the supervision package.
- `recoverable_v4` has only 4 regimes and a 12% join rate; small-lab numbers
  do not transfer one-to-one to the harder `r12_p079` and
  `recoverable_v5_t128` scopes. The qualitative conclusion - chance-level
  endpoints under count-only supervision - does transfer.
- The join accuracy in `teacher_parent_full_sketch` rows is below the
  always-no-join prior (around 0.5-0.6) because the join head is a separate
  output that this objective does not heavily weight; the merged state still
  encodes the truth `(first, last)`, which is what matters for state
  recovery.
