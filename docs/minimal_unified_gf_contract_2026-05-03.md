# Minimal Unified g/f Contract

Date: 2026-05-03

This note defines the reference lane for the current unified-g review. The
canonical implementation target is
`src/ctreepo/sim/core/clean_unified_fg.py`, specifically `CleanUnifiedNO`.
`FNOCountSketch` remains production Markov infrastructure, but it is not the
minimal reference implementation.

## Contract

The learned operator should mirror the traditional C-TreePO target

```text
f*(x + y) = f*(g*(g*(x) + g*(y)))
```

with the learned state equations

```text
z_x  = g_theta(embed(x), null)
z_y  = g_theta(embed(y), null)
z_xy = g_theta(z_x, z_y)
score = f_theta(z_xy)
```

The only pre-g learned adapter in the clean neural-operator lane is token
embedding from token ids to a content function. Leaf states must be produced by
`g(leafInput(content))`. Merge states must be produced by the same endomap
through `g(mergeInput(left, right))`. `f` must accept any carrier state - leaf,
internal merge, or root - and return the scalar score/readout.

In the reference API this is expressed directly:

```python
z_x = g(leaf_input(embed(x)))
z_y = g(leaf_input(embed(y)))
z_xy = g(merge_input(z_x, z_y))
score = f(z_xy)
```

`g.encode_leaf(...)` and `g.merge(...)` may exist as compatibility aliases, but
new contract checks should prefer the direct `g(...)` calls so the public
interfaces of `g` and `f` stay aligned.

## Allowed Components

- `token_embedding`: trivial input adapter from token ids to `(B, C, L)`.
- `g`: the single learned state operator. In `CleanUnifiedNO`, this is one FNO
  applied as `g(leafInput(content))` for leaves and
  `g(mergeInput(left, right))` for merges. Both wrappers lower to the same
  `(B, 2C, L) -> (B, C, L)` FNO.
- `f`: the state readout. In `CleanUnifiedNO`, this is FNO -> pool -> Linear,
  applied uniformly to all node states.

## Contextual Sufficiency

The paper-facing interpretation should put `g` front and center as the learned
sufficient state map. The merge/operator role is the implementation surface;
the statistical contract is stronger:

```text
rep(x) = g_theta(leafInput(x))
merge(s, t) = g_theta(mergeInput(s, t))

rep(x) = rep(y)
  ==> for every context (left, right),
      fstar(left * x * right) = fstar(left * y * right)
```

Equivalently, `g_theta(leafInput(x))` should preserve the full contextual
response signature of `x` under downstream composition. The explicit Lean
contract has one carrier space: `leafInput : Raw -> Carrier`,
`mergeInput : Carrier -> Carrier -> Carrier`, one shared endomap
`g : Carrier -> Carrier`, and downstream readouts `f : Carrier -> Y`.
For the common case where raw leaves already live in the carrier, use
`UniformG.onCarrier` with `leafInput = id`. The local laws certify that learned
reductions stay inside the same contextual-response fiber. For the Markov
changepoint witness, the exact `(count, first, last)` sketch is one sufficient
state, but it is not the definition of sufficiency.

The Lean layer for this is
`FormalProofs.OPT.ContextualQuerySufficiency`. Its public surface is:

- `QuerySufficient`
- `ResponseSignature`
- `TwoSidedContextQuery`
- `TwoSidedContextSufficient`
- `UniformG`
- `UniformG.leaf`
- `UniformG.merge`
- `FiniteContextCovers`
- `querySufficient_iff_exists_contextReadout`
- `querySufficient_no_collision_of_distinguished_context`
- `finiteContext_zeroLoss_implies_querySufficient`
- `uniformComposedTwoSidedReadoutExact_implies_twoSidedContextSufficient`
- `uniformComposedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin`

The Markov file re-exports the specialization showing that the existing
`MarkovCountQuerySufficient` theorem is exactly two-sided contextual
sufficiency for `fstar = MarkovCountSketch.count`. It also keeps count-only
Markov state as the generic counterexample and exact `(count, first, last)` as
the validation witness.

This is compatible with Chen et al.,
[Neural Approximate Sufficient Statistics for Implicit Models](https://arxiv.org/abs/2010.10079):
their mutual-information objective motivates learning sufficient statistics;
our deterministic Lean core formalizes the compositional/contextual response
fibers that the learned statistic must preserve. Full Shannon mutual
information is learning motivation here, not part of the formal theorem
surface.

The modern objective pull-through is tracked in
`docs/contextual_sufficiency_modern_lit_2026-05-04.md`. The short version:
borrow NASS/DeepInfoMax-style dependence losses from Chen et al. 2021, borrow
sliced low-dimensional response targets from Chen/Gutmann/Weller 2023, and use
Dirmeier/Albert/Perez-Cruz 2025 SSNL as the closest modern application pattern
where the learned low-dimensional state is the variable on which downstream
inference is performed.

The clean probe now exposes those options through:

```text
--contextual-dependence-objective infonce|regression|dcorr|jsd|dv|wasserstein|none
--response-signature-contexts K
--response-signature-slices M
```

`regression` with response-signature slices is the first recommended objective
after plain contextual MSE.

## Forbidden Shortcuts

- A separate learned leaf encoder that constructs leaf states without calling
  `g(leafInput(content))`.
- A separate learned merge projector that constructs parent states through a
  different operator than the leaf-read `g`.
- A root-only readout path that cannot score leaf and internal states.
- Exact projected merge being reported as the learned unified-g runtime path.
- Labeling `FNOCountSketch(tree_model_version="unified_g")` as the minimal
  reference implementation.

## Production Path Compatibility

`FNOCountSketch` currently has a `tree_model_version="unified_g"` mode, but that
mode is broader than this reference contract. Its leaf path prepares a wide
summary from FNO pooled/endpoint features and runs `summary_encoder`; its merge
path runs a learned `unified_g_merge_summary_proj` over `(left_state,
right_state)` before the same `summary_encoder`. That shares a downstream
encoder, but it is not the same typed `g` call at leaves and merges.

This is acceptable as legacy/production infrastructure. It should not be used
as evidence that the minimal unified-g contract has been implemented.

## Current Probe Guidance

The matched `CleanUnifiedNO` probe with channels=128, g modes=1024, and 4
shared-g layers has about 34M parameters, with almost all capacity in `g`.
As of epoch 36 in `outputs/clean_unified_no_sharedg_match_20260503_223145`,
best validation root MAE was still about 3.46 and later epochs were unstable.

Do not use that 34M run as the next tuning target. First keep the reference lane
small and controlled:

- channels 32 or 64
- g modes 16 to 64
- g layers 2
- AdamW with weight decay
- cosine LR schedule or another explicit decay
- gradient clipping enabled

Large capacity sweeps should wait until the structural contract tests pass and a
small stable probe has a reproducible baseline.

## Clean Grid Smoke, 2026-05-04

Added `scripts/run_clean_unified_no_grid.py` to run compact grids over
`CleanUnifiedNO` without routing through production `FNOCountSketch` presets.
The grid wrapper writes a manifest, per-cell outputs, `grid_summary.csv`, and
`grid_report.md`.

Initial paper-facing smoke:

- Command family: `recoverable_v5_t2048`, train docs 64, epochs 2, channels 32,
  g layers 2, AdamW, weight decay 0.01, cosine LR, gradient clip 1.0.
- Output root: `outputs/clean_unified_no_grid_20260504_0316`.

| leaf tokens | leaves/doc | g modes | test root MAE |
| ---: | ---: | ---: | ---: |
| 2048 | 1 | 8 | 6377.31 |
| 2048 | 1 | 16 | 254.00 |
| 256 | 8 | 8 | 778.19 |
| 256 | 8 | 16 | 44.22 |

Follow-up merge-only capacity smoke:

- Command family: `recoverable_v5_t2048`, fixed leaf tokens 256 (8 leaves/doc),
  train docs 128, epochs 4, g layers 2, AdamW, weight decay 0.01, cosine LR,
  gradient clip 1.0.
- Output root: `outputs/clean_unified_no_grid_followup_20260504_0320`.

| channels | g modes | params | test root MAE |
| ---: | ---: | ---: | ---: |
| 32 | 16 | 65,025 | 9.76 |
| 32 | 32 | 81,409 | 10.30 |
| 64 | 16 | 257,025 | 6.55 |
| 64 | 32 | 322,561 | 9.14 |

Interpretation: the minimal clean lane is trainable under the paper-facing
recoverable Markov setup once the run is slightly less tiny. In this short
regime, `g_n_modes=16` is better than 32, and channels 64 improves over
channels 32. The one-leaf/2048-token cells are unstable in two epochs and should
not be used as the main composition diagnostic.

Channel-count follow-up:

- Literal `channels=2048` (same as total doc tokens) is feasible on the current
  GPU, but it is a 260.15M parameter diagnostic (`g=134.25M`, `f=125.86M`).
  A deliberately tiny run (`train_docs=8`, `eval_docs=16`, `epochs=1`,
  `leaf_tokens=256`) completed at
  `outputs/clean_unified_no_channels_eq_doc_tokens_20260504_0430`, with
  `test_root_mae=142.45`. This should not be treated as a useful tuned point;
  it only proves the literal shape can instantiate and run.
- `channels=256` (same as leaf tokens in the 8-leaf composition setup) is a more
  useful scale point. With `train_docs=128`, `epochs=4`, `leaf_tokens=256`,
  `g_n_modes=16`, and the same AdamW/cosine/clip settings, the run at
  `outputs/clean_unified_no_channels_eq_leaf_tokens_20260504_0431` reached
  `best_val_root_mae=4.30` and `test_root_mae=4.38` at epoch 1. This improves on
  the short-grid `channels=64, g_n_modes=16` result (`test_root_mae=6.55`) but
  shows early overfit after epoch 1.
- Single-leaf follow-up (`leaf_tokens=2048`, one leaf/doc):

| channels | interpretation | train docs | params | best epoch | test root MAE |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | channels = doc tokens | 8 | 260.15M | 1 | 210.70 |
| 128 | channels = train docs | 128 | 1.02M | 4 | 15.35 |
| 256 | larger practical one-leaf scale | 128 | 4.08M | 4 | 3.97 |

Output roots:

- `outputs/clean_unified_no_doc2048_oneleaf_ch2048_20260504_0547`
- `outputs/clean_unified_no_doc2048_oneleaf_ch128_traindocs128_20260504_0548`
- `outputs/clean_unified_no_doc2048_oneleaf_ch256_20260504_0549`

Interpretation: for one-leaf full-document reading, `channels=256` is the best
small practical scale tested so far. Literal `channels=doc_tokens=2048` runs,
but at 260M parameters it is not useful without a much more careful optimizer
and data-size plan.

Longer-doc direct generation follow-up:

`recoverable_v5_t2048` is the largest named benchmark currently wired into the
paper-facing Markov config surface. For larger document lengths,
`scripts/probe_clean_unified_no.py` now supports direct sticky-recoverable
generation via `--doc-tokens`. The default expected boundary count scales as
`5 * sqrt(doc_tokens / 128)`, matching the existing 2048-token design
(`~20` expected boundaries/doc).

Using `leaf_tokens=256`, `channels=256`, `g_n_modes=16`, `g_n_layers=2`,
`train_docs=128`, and the same AdamW/cosine/clip settings:

| doc tokens | leaves/doc | eval docs | target scale | best epoch | test root MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 256 | 42 | 1 | 4.41 |
| 8192 | 32 | 256 | 58 | 1 | 5.32 |
| 16384 | 64 | 128 | 82 | 4 | 7.18 |

Output roots:

- `outputs/clean_unified_no_doc4096_leaf256_ch256_20260504_0514`
- `outputs/clean_unified_no_doc8192_leaf256_ch256_20260504_0515`
- `outputs/clean_unified_no_doc16384_leaf256_ch256_20260504_0516`

Interpretation: the minimal clean lane remains stable as document length and
merge depth increase from 8 leaves/doc at 2048 tokens to 64 leaves/doc at 16k
tokens. Error rises with length, but the rise is modest relative to the target
scale and far below the earlier unstable tiny/low-mode runs.

## Required Checks

- Exactly one `CleanUnifiedG` instance and one `g.fno` module exist.
- `forward_doc` cannot produce leaf states if `g.forward` is unavailable.
- Changing the leaf input path `g(leafInput(content))` changes leaf/root states
  in one-leaf runs.
- Changing the merge input path `g(mergeInput(left, right))` changes root states
  in two-leaf runs.
- Manual replay of
  `f(g(mergeInput(g(leafInput(x)), g(leafInput(y)))))` matches the stored
  two-leaf root prediction.
- Balanced multi-leaf replay through the merge-input wrapper matches every
  stored merge state.
- All stored node scores equal `f(state)` applied to the corresponding states.

## Exactness Push, 2026-05-04

This section records the next experiment wave after the initial clean-lane
contract checks. The motivating concern is that the Markov DGP is not merely
learnable in a loose sense: for the disjoint-palette recoverable family, the
root changepoint count is exactly recoverable from observed token ids. Therefore
the right diagnostic question is not whether the model can beat a naive
baseline; it is why a learned minimal unified-g/f lane is not yet finding the
zero-error rule.

### Exact Witness

Added an exact disjoint-palette witness to `scripts/probe_clean_unified_no.py`
and surfaced its columns in `scripts/run_clean_unified_no_grid.py`.

For `recoverable_v5_t2048`, the witness maps each token id to its palette block
and counts adjacent block changes. On the same prepared data used by the clean
NO probes, this gives:

| split | docs | exact witness root MAE | max abs error |
| --- | ---: | ---: | ---: |
| train | 1024 | 0.0 | 0.0 |
| val | 1024 | 0.0 | 0.0 |
| test | 1024 | 0.0 | 0.0 |

Interpretation: any nonzero learned root MAE on this benchmark is a model,
optimizer, or inductive-bias failure. It is not irreducible DGP noise or missing
observability. Future grid summaries now include
`exact_witness_val_root_mae`, `exact_witness_test_root_mae`, and
`exact_witness_test_max_abs_error`.

### Contextual-Sufficiency Probe Objective

`scripts/probe_clean_unified_no.py` now has an opt-in contextual objective:

```bash
./venv/bin/python scripts/probe_clean_unified_no.py \
  --training-objective contextual_sufficiency \
  --context-samples-per-doc 1 \
  --contextual-loss-weight 1.0 \
  --infomax-loss-weight 0.0
```

The default remains unchanged: `--training-objective root` with
`--context-samples-per-doc 0`.

When enabled, the probe samples fixed-length two-sided context fragments,
encodes each fragment through the same leaf surface `g(leafInput(embed(.)))`,
composes with the same merge surface `g(mergeInput(left_state, right_state))`,
and trains `f` on the exact oracle count for `left + span + right`. An optional
contrastive empirical-response-signature loss can be enabled with
`--infomax-loss-weight` and `--response-signature-contexts`.

The summary now reports contextual MAE/correlation alongside root diagnostics,
constant-predictor baselines, exact witness MAE, exact deterministic surface
MAE, and Markov sketch/boundary diagnostics. Boundary supervision diagnostics
include precision, recall, F1, and predicted-positive rate so an all-zero
boundary predictor is not accidentally treated as successful.

### Completed/Partially Completed Results

Snapshot as of 2026-05-04 around 18:25 UTC. Some grids are still running, so
the tables below record the best completed cells plus useful live tails.

Medium leaf-size grid, 512 train docs, 12 epochs, `channels=256`,
`g_n_modes=16`, `g_n_layers=2`:

| leaf tokens | leaves/doc | best val root MAE | test root MAE |
| ---: | ---: | ---: | ---: |
| 2048 | 1 | 3.513 | 3.700 |
| 1024 | 2 | 3.526 | 3.755 |
| 512 | 4 | 3.463 | 3.685 |
| 256 | 8 | 3.517 | 3.739 |
| 128 | 16 | 3.235 | 3.505 |
| 64 | 32 | 3.038 | 3.246 |

Output roots:

- `outputs/clean_unified_no_leafgrid_medium_20260504_0645/gpu0_large_leaves`
- `outputs/clean_unified_no_leafgrid_medium_20260504_0645/gpu1_small_leaves`

This grid showed that smaller leaves help under the clean contract, moving the
best completed test error from the old ~3.7 one-leaf range to ~3.25 at
`leaf_tokens=64`.

Long literal `channels=2048` one-leaf runs:

| run | train docs | lr | epochs | best val root MAE | test root MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ch2048, train1024 | 1024 | 3e-5 | 80 | 3.446 | 3.719 |
| ch2048, train2048 | 2048 | 1e-5 | 60 | 3.447 | 3.726 |

Output roots:

- `outputs/clean_unified_no_ch2048_oneleaf_lr3e5_train1024_20260504_0641/run`
- `outputs/clean_unified_no_ch2048_oneleaf_lr1e5_train2048_20260504_0641/run`

Interpretation: `channels = doc_tokens = 2048` does not solve the zero-merge
case. Even very large channel count and long one-leaf training remain around
the same ~3.7 test MAE. This strongly suggests that simply giving the FNO more
state channels is not enough for the current architecture/optimizer to discover
the adjacent-palette transition rule.

Direct `doc_tokens=128` exactness sanity grid:

- Output root: `outputs/clean_unified_no_exactness_t128_many_20260504_171158/grid`
- Purpose: check whether the clean NO lane can solve the smaller exact DGP where
  the same palette-block witness is zero.
- Grid: `leaf_tokens={128,64,32,16}`, `channels={64,128,256}`,
  `g_n_modes={8,16,32,64}`, 2048 train docs, 1024 eval docs, 60 epochs.

Best completed cells so far:

| cell | leaf tokens | channels | g modes | best val root MAE | test root MAE | exact witness test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `leaf128_ch128_gm32_seed0` | 128 | 128 | 32 | 0.791 | 0.789 | 0.0 |
| `leaf128_ch256_gm8_seed0` | 128 | 256 | 8 | 0.809 | 0.804 | 0.0 |
| `leaf128_ch64_gm32_seed0` | 128 | 64 | 32 | 0.841 | 0.820 | 0.0 |

Live tail: later cells are still improving within the 60-epoch budget; one
active cell reached `val_root_mae=1.0269` by epoch 44. The headline is that the
clean lane gets below 1.0 on the small exact DGP, but it has not yet reached the
zero witness.

`recoverable_v5_t2048` shallow small-leaf grid:

- Output root: `outputs/clean_unified_no_t2048_smallleaf_many_20260504_171159/grid`
- Purpose: continue the leaf-size result with more train docs/epochs and exact
  witness reporting.
- Grid: `leaf_tokens={256,128,64,32,16}`, `channels={128,256,512}`,
  `g_n_modes={8,16}`, 2048 train docs, 1024 eval docs, 60 epochs.

Best completed cells so far:

| cell | leaf tokens | channels | g modes | best val root MAE | test root MAE | exact witness test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `leaf256_ch128_gm16_seed0` | 256 | 128 | 16 | 1.376 | 1.370 | 0.0 |
| `leaf256_ch256_gm8_seed0` | 256 | 256 | 8 | 1.457 | 1.451 | 0.0 |
| `leaf256_ch128_gm8_seed0` | 256 | 128 | 8 | 1.774 | 1.795 | 0.0 |

Interpretation: extra epochs/data are materially helping. The old 512-doc,
12-epoch `leaf=256` result was ~3.74 test MAE; this grid has already improved
to ~1.37 test MAE under the same minimal contract. The live tails still show
late-epoch improvement, so longer runs are justified.

`recoverable_v5_t2048` deeper small-leaf grid:

- Output root: `outputs/clean_unified_no_t2048_deep_smallleaf_20260504_171430/grid`
- Purpose: test whether deeper shared `g` and deeper operator readout `f`
  improve composition under the strict contract.
- Grid: `leaf_tokens={128,64,32,16}`, `channels={256,512}`,
  `g_n_modes={8,16}`, `g_n_layers=4`, `scorer_n_layers=4`, 2048 train docs,
  1024 eval docs, 60 epochs.

Completed so far:

| cell | leaf tokens | channels | g modes | best val root MAE | test root MAE | exact witness test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `leaf128_ch256_gm8_seed0` | 128 | 256 | 8 | 1.197 | 1.237 | 0.0 |

Live tail: the next deeper cell reached `val_root_mae=1.1640` by epoch 32/60.
This is currently the most promising t2048 direction. It suggests the previous
3.x range was not an unavoidable floor for the clean contract; better leaf
granularity, more epochs, and deeper `g/f` can push substantially lower.

`recoverable_v5_t2048` one-leaf high-mode grid:

- Output root: `outputs/clean_unified_no_t2048_oneleaf_highmodes_20260504_171429/grid`
- Purpose: test whether the zero-merge/full-doc failure is mainly Fourier-mode
  or readout-resolution limited.
- Grid: `leaf_tokens=2048`, `channels={128,256,512}`,
  `g_n_modes={32,64,128}`, `scorer_n_modes=128`, 2048 train docs, 1024 eval
  docs, 80 epochs.

Best completed cells so far:

| cell | channels | g modes | best val root MAE | test root MAE | exact witness test MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `leaf2048_ch128_gm32_seed0` | 128 | 32 | 3.444 | 3.719 | 0.0 |
| `leaf2048_ch128_gm64_seed0` | 128 | 64 | 3.453 | 3.724 | 0.0 |
| `leaf2048_ch128_gm128_seed0` | 128 | 128 | 3.454 | 3.724 | 0.0 |
| `leaf2048_ch256_gm64_seed0` | 256 | 64 | 3.453 | 3.725 | 0.0 |

Interpretation: high Fourier modes and higher readout modes do not crack the
one-leaf path. The one-leaf/full-doc readout appears to be a poor inductive
bias for learning an adjacent-transition count, even though the exact witness
is trivial. Multi-leaf composition is now outperforming one-leaf reading.

### Queued Longer-Epoch Follow-Ups

Added `scripts/wait_for_long_job_then_run.py` so follow-up jobs can wait for a
current launcher before occupying the same GPU. Four queued jobs are active in
the waiting state:

| queued output root | waits for | purpose |
| --- | --- | --- |
| `outputs/clean_unified_no_t128_epochs240_focused_20260504_173829` | t128 60-epoch grid | Run focused t128 cells for 240 epochs to see whether the exact small DGP approaches zero. |
| `outputs/clean_unified_no_t2048_l2_epochs240_focused_20260504_173831` | t2048 shallow grid | Extend the promising shallow small-leaf cells to 240 epochs. |
| `outputs/clean_unified_no_t2048_oneleaf_ch512_gm128_ep240_20260504_173831` | t2048 one-leaf high-mode grid | Give the best high-mode one-leaf hypothesis a longer focused run. |
| `outputs/clean_unified_no_t2048_l4_epochs200_focused_20260504_173830` | t2048 deeper grid | Extend the promising deeper small-leaf cells to 200 epochs. |

### Current Interpretation

The exact witness proves that the target function is available from observed
tokens. The clean `g/f` contract is therefore not being limited by the DGP. The
current experiments point to three working conclusions:

1. One-leaf/full-doc FNO reading is the wrong inductive bias for this exact
   transition-count DGP. More channels, more modes, and more one-leaf epochs
   have not moved it below ~3.7 test MAE.
2. Multi-leaf trees under the strict shared-g contract are substantially better.
   With 2048-token docs, `leaf_tokens=256` has already improved from ~3.7 to
   ~1.37 test MAE when we use more train docs and a 60-epoch schedule.
3. Deeper shared `g/f` with smaller leaves is the best current direction. The
   first completed deeper cell has test MAE ~1.24, and the next live cell has
   already reached ~1.16 validation MAE before finishing.

What we hope to learn from the queued longer runs:

- Whether the t128 exact DGP can be driven close to zero under the clean NO
  contract, or whether even the small problem stalls above zero.
- Whether the t2048 multi-leaf improvements continue past 60 epochs or plateau
  around ~1.0-1.5.
- Whether the one-leaf path is fundamentally stuck under this architecture, or
  just slower.
- Which axis matters more next: leaf size, depth/layers, channels, or a more
  explicit sufficient-state parameterization that still respects
  `g(leafInput(x))`, `g(mergeInput(left, right))`, and `f(carrier)`.
