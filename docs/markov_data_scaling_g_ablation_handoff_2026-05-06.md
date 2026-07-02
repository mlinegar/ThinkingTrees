# Markov Data-Scaling and g-Ablation Handoff (2026-05-06)

Follow-up handoff after the contextual-sufficiency ablation grids
([`markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`](markov_contextual_sufficiency_ablation_handoff_2026-05-05.md)).
This doc records three rounds run on 2026-05-06 — data scaling, g-side
ablation, and FNO-as-g sized properly — plus the engineering changes
that unblocked them. Intended for a fresh LLM/engineer to pick up
without replaying the session.

For the latest top-level snapshot start with
[`markov_sim_status.md`](markov_sim_status.md).

## Short Status

The Markov contextual-sufficiency thread now has a solid empirical
data-scaling picture and a tight g-side ablation:

- **Data scaling (R5)** answered "does more data close the count_mae gap?"
  Yes: `count_only` flexible learner goes 0.82 → 0.12 → 0.027 across
  1024 → 10240 → 102400 docs. Architectural ceiling (`rts`) hits 0.0005
  at 102400 — essentially solved.
- **g-side ablation (R6)** showed g is *not* the bottleneck at this
  scale. The simple `mlp merge + linear decoder + nass_jsd` stack hits
  0.0227 — within noise of the best run; FNO-as-g at length-2 is
  strictly worse; merge-loss (mse / nass_jsd / nasss_jsd) moves
  count_mae by ~10%; decoder_head=linear is a free 19% win over
  decoder_head=mlp.
- **FNO-as-g sized properly (R7)** confirmed FNO ≈ MLP at matched
  rep_dim. Increasing rep_dim from 50 → 256 didn't help either family;
  FNO with deeper/wider variants got slightly worse (optimization friction).
- The residual ~50× gap from the best flexible learner (0.022) to the
  ceiling (0.0005) is **f-side** (encoder mis-specification or
  optimization), not g-side.

Canonical output roots:

- `outputs/markov_fno_round5_data_scaling_20260506_195724/` — data scaling, 30 cells
- `outputs/markov_fno_round5_smoke20_20260506_193236/` — n_iter=20 screening, 12 cells
- `outputs/markov_fno_round6_g_ablation_20260506_222608/` — g-side ablation, 12 cells
- `outputs/markov_fno_round7_repdim_fno_g_20260506_232039/` — rep_dim × FNO-as-g, 8 cells

CSV digests:

- `outputs/markov_fno_round5_data_scaling_20260506_195724/data_scaling_summary.csv`
- `outputs/markov_fno_round6_g_ablation_20260506_222608/round6_g_ablation_summary.csv`
- `outputs/markov_fno_round7_repdim_fno_g_20260506_232039/round7_repdim_fno_g_summary.csv`

## What We Ran

| round | runner | cells | output root | purpose |
|---|---|---:|---|---|
| **R5 smoke20** | hand-launched 12-cell shell loop | 12 | `outputs/markov_fno_round5_smoke20_20260506_193236/` | Cheap n_iter=20 screen of data scaling × family × encoding × leaf to confirm the data-scaling effect before committing 2h to R5 proper. |
| **R5 data scaling** | [`scripts/run_markov_fno_round5_data_scaling.sh`](../scripts/run_markov_fno_round5_data_scaling.sh) | 30 | `outputs/markov_fno_round5_data_scaling_20260506_195724/` | Headline: train_docs ∈ {1024, 10240, 102400} × 3 families × 2 encodings × {64, 128} leaves. n_iter=200. |
| **R6 g-ablation** | [`scripts/run_markov_fno_round6_g_ablation.sh`](../scripts/run_markov_fno_round6_g_ablation.sh) | 12 | `outputs/markov_fno_round6_g_ablation_20260506_222608/` | g-axis ablation at the headline cell: merge_family × merge_loss × decoder_head. |
| **R7 rep_dim × FNO-as-g** | [`scripts/run_markov_fno_round7_repdim_fno_g.sh`](../scripts/run_markov_fno_round7_repdim_fno_g.sh) | 8 | `outputs/markov_fno_round7_repdim_fno_g_20260506_232039/` | Re-test FNO-as-g after fixing the length-2 design — proper FNO over the rep-dim spatial axis at multiple rep_dim. |

All four runs use the same DGP (`paper_hazard_panel_v1_t128`, 12 regimes,
doc length 128) and seed 0. `regime_one_hot` encoding and `count_only`
supervision are the headline f-side; `fully_learned` arch is the
headline g-side.

## Main Results

### R5 data scaling (30 cells, n_iter=200)

`count_mae = theta_count_raw_mae` at test, leaf=64 only shown:

| family | encoding | 1024 | 10240 | 102400 | 100× ratio |
|---|---|---:|---:|---:|---:|
| `fno_count_only` | regime_oh | 0.818 | 0.124 | **0.027** | 30× |
| `fno_count_only` | tokens | 2.522 | 0.811 | 0.049 | 51× |
| `fno_sketch + analytic` | regime_oh | 2.251 | 0.279 | 0.061 | 37× |
| `fno_sketch + analytic` | tokens | 3.725 | 0.562 | 0.058 | 65× |
| `regime_transition_sum` (ceiling) | regime_oh | 1.974 | 0.013 | **0.0005** | 4069× |

Headlines:

1. The `count_only` flexible learner at the headline cell beats
   `sketch+analytic` by ~2.3× at 102400. Same finding as 1024 (Round 4),
   sharpens at scale.
2. The architectural ceiling (`regime_transition_sum`) drops to 0.0005 at
   102400 — essentially the irreducible noise floor. Confirms the model is
   correctly specified; small-data error was 1/√N variance + small-N
   optimization friction.
3. `regime_one_hot` > `tokens` encoding at every scale; gap narrows from
   ~3× at 1024 to ~2× at 102400.
4. `leaf=128` is uniformly worse than `leaf=64` because `fragment_len ==
   doc_len` produces a degenerate tree (no merges → merge head untrained).

### R6 g-side ablation (12 cells, n_iter=200, 102400/leaf=64/regime_oh/count_only)

Sorted by count_mae ascending:

```
merge_family  merge_loss   decoder   count_mae
─────────────────────────────────────────────────
mlp           nass_jsd     linear    0.0227   ← best (tied)
mlp           nasss_jsd    linear    0.0227   ← best (tied)
mlp           mse          linear    0.0230
mlp           mse          mlp       0.0296
mlp           nass_jsd     mlp       0.0301
fno_rep[L=2]  nass_jsd     linear    0.0305
fno_rep[L=2]  nass_jsd     mlp       0.0341
mlp           nasss_jsd    mlp       0.0352
fno_rep[L=2]  nasss_jsd    linear    0.0355
fno_rep[L=2]  mse          linear    0.0370
fno_rep[L=2]  mse          mlp       0.0370
fno_rep[L=2]  nasss_jsd    mlp       0.0446
```

Axis effects (averaged):

| axis | comparison | average count_mae |
|---|---|---|
| decoder_head | linear vs mlp | 0.0285 vs 0.0351 (linear wins, ~19%) |
| merge_family | mlp vs fno_rep [length-2] | 0.0272 vs 0.0364 (mlp wins, ~25%) |
| merge_loss | nass_jsd / mse / nasss_jsd | 0.0293 / 0.0316 / 0.0345 |

`fno_rep` in R6 used a length-2 spatial axis (left vs right) which is
degenerate (only mode 0 = sum and mode 1 = diff). R7 reframes it.

### R7 FNO-as-g, properly sized (8 cells, n_iter=200, headline cell)

`fno_rep` rewritten: spatial axis = rep_dim (state_dim_effective),
channels = (left, right) lifted to `merge_fno_hidden_channels`. With
state_dim=256 and n_modes=32, this is a real FNO with 32 spectral modes
along the rep dim.

```
cell                             merge    rep_dim  modes  layers  hid   count_mae
─────────────────────────────────────────────────────────────────────────────────
mlp__rep050                      mlp        50      —     —       —     0.0224  ← best
mlp__rep128                      mlp       128      —     —       —     0.0269
fno__rep128__m32                 fno       128      32     2      32    0.0273
mlp__rep256                      mlp       256      —     —       —     0.0278
fno__rep256__m32                 fno       256      32     2      32    0.0290
fno__rep050__m16                 fno        50      16     2      32    0.0300
fno__rep256__m64__l3             fno       256      64     3      64    0.0333
fno__rep256__m32__hid64          fno       256      32     2      64    0.0334
```

Findings:

1. Larger rep_dim does *not* help — for either family. MLP goes 0.0224 →
   0.0269 → 0.0278; FNO goes 0.0300 → 0.0273 → 0.0290 across rep_dim ∈
   {50, 128, 256}.
2. At matched rep_dim, FNO ≈ MLP. The "FNO is at least as good given
   enough data" hypothesis holds; FNO does not *exceed* MLP either.
3. Adding FNO capacity (deeper, wider) makes it slightly worse —
   optimization friction at n_iter=200.

Interpretation: the count-only merge is fundamentally low-dim (count +
endpoint regimes ≈ 25 useful dims), so capacity is not the bottleneck.

## What Changed In Code

Primary surface:

- [`src/ctreepo/sim/core/contextual_sbijax.py`](../src/ctreepo/sim/core/contextual_sbijax.py)
- [`scripts/probe_contextual_sbijax.py`](../scripts/probe_contextual_sbijax.py)

### Performance refactor (unblocked 102400-doc scale)

The pre-2026-05-06 training step closed over the full
`train_left_features` / `train_right_features` and ran the FNO encoder
over **all N_train_merges every step**. At 102400 docs / leaf=64 this
allocates ~72 GiB and OOMs. Fix:

- **Mini-batch the merge supervision.** Each step samples
  `merge_batch_size = min(batch_size, n_train_merges)` random merge
  indices; only the corresponding subset is processed. Memory becomes
  O(batch) instead of O(N_train).
- **Move all 10 gathers inside the JIT.** The step takes integer
  indices `idx` and `merge_idx` and gathers train tensors internally;
  saves 10 separate kernel launches per step (~10-20% speedup).
- **Accumulate metrics on device.** A length-7 jnp accumulator is
  updated inside the JIT; one host sync per epoch instead of per step.
- **Drop trailing partial batches.** All R5/R6/R7 cells use `batch_size=128`
  with N_train ∈ {1024, 10240, 102400} which divide evenly, so this is
  effectively a no-op; safety guard for other configs.

Eval-time changes (also fixed at the same time):

- **`_apply_fn_chunked` helper.** `evaluate_contextual_sbijax` and
  `_markov_local_law_eval_metrics` ran `apply_fn(params, all_tokens, …)`
  in one shot, OOMing the FNO at 102400 docs (~3.2 GiB intermediate).
  Now chunks at 1024 rows.
- **N²-collision diagnostic subsampling.** `_collision_rate` and
  `_summary_collision_diagnostics` materialized `(N, N, D)` pairwise
  distance arrays — at N=10⁵ this asks for 1.2 TiB of host RAM.
  Subsample to 4096 rows; pairwise distance is a sanity check, a
  uniform random subset is sufficient.

Parity smoke at 1024 docs: `count_mae = 0.8180` exactly matches the R5
baseline before refactor (bit-equivalent for `mlp+mse+mlp+default rep_dim`).

### New model variants (R6/R7)

| flag | values | purpose |
|---|---|---|
| `--local-law-merge-loss` | `mse` / `nass_jsd` / **`nasss_jsd`** (NEW) | C2 supervision form. `nasss_jsd` slices `merge_target` onto `merge_nasss_n_slices` random unit projections and applies per-slice JSD MI lower bound. |
| `--merge-family` (NEW) | `mlp` / `fno_rep` | Architecture for `g(s_L, s_R)`. `fno_rep` is a 1D FNO with the rep dim as spatial axis and (left, right) lifted to `merge_fno_hidden_channels` channels. |
| `--merge-fno-n-modes` (NEW) | int (default 16) | Spectral modes kept along the rep-dim axis. Capped at `state_dim // 2 + 1`. |
| `--merge-fno-n-layers` (NEW) | int (default 2) | Number of FNO blocks in the merge family. |
| `--merge-fno-hidden-channels` (NEW) | int (default 32) | Lifted channel dim for the FNO merge family. |
| `--decoder-head` (NEW) | `mlp` / `linear` | Decoder readout architecture. `linear` is a single Dense; `mlp` is the existing 2-layer head. |
| `--merge-nasss-n-slices` (NEW) | int (default 16) | Slice count for `nasss_jsd` merge supervision. |

All new fields surface in `provenance` (`merge_family`,
`merge_fno_n_modes`, `merge_fno_n_layers`, `merge_fno_hidden_channels`,
`decoder_head`, `local_law_merge_loss`, `merge_nasss_n_slices`).

### New helper / model factories

- `_make_fno_merge_net` ([`contextual_sbijax.py:2575`](../src/ctreepo/sim/core/contextual_sbijax.py#L2575)) — proper FNO over rep-dim axis. R6 had a length-2 version; the current code is the R7 rewrite.
- `_make_learned_decoder_net` ([`contextual_sbijax.py:2541`](../src/ctreepo/sim/core/contextual_sbijax.py#L2541)) — accepts `head ∈ {mlp, linear}`.
- `_apply_fn_chunked` ([`contextual_sbijax.py:3270`](../src/ctreepo/sim/core/contextual_sbijax.py#L3270)) — chunked eval-time apply_fn wrapper.
- NASSS merge slice matrix built once at fit-time (`merge_nasss_slice_matrix`); per-slice critic uses the existing `_MergeJSDCriticNet` Flax module.

## How To Reproduce

### Generate the 102400-doc bundle (one-time)

The `paper_hazard_panel_v1_t128` bundle that ships in the repo has
10240 train docs. The 102400-doc bundle is generated separately:

```bash
./venv/bin/python scripts/prepare_markov_hazard_panel_data.py \
  --panel-ids paper_hazard_panel_v1_t128 \
  --train-docs 102400 \
  --val-docs 1024 \
  --test-docs 1024 \
  --seed 0 \
  --bundle-root outputs/_bundles/markov_hazard_panels_train102400 \
  --skip-prepared-cache
```

Lands at
`outputs/_bundles/markov_hazard_panels_train102400/paper_hazard_panel_v1_t128/seed_0/base_bundle.json`
(~456 MB JSON, 1-2 min generation).

### R5 — data scaling (30 cells, ~2h on 3 GPUs)

```bash
N_ITER=200 GPUS=0,2,3 bash scripts/run_markov_fno_round5_data_scaling.sh
```

The launcher resolves the right bundle (existing 10240-doc for
train_docs ∈ {1024, 10240}; new 102400-doc for train_docs=102400) and
emits one cell per (family × encoding × leaf × train_docs). Aggregator
writes `data_scaling_summary.csv` at the run root.

### R6 — g-side ablation (12 cells, ~1.5h on 3 GPUs)

```bash
GPUS=0,2,3 bash scripts/run_markov_fno_round6_g_ablation.sh
```

Holds f-side at the R5 winner (jax_fno + count_only + regime_oh +
fully_learned + leaf=64 + train=102400) and varies merge_family ×
merge_loss × decoder_head. Aggregator writes
`round6_g_ablation_summary.csv`.

### R7 — rep_dim × FNO-as-g (8 cells, ~1h on 3 GPUs)

```bash
GPUS=0,2,3 bash scripts/run_markov_fno_round7_repdim_fno_g.sh
```

Holds R6 g-side winner (nass_jsd + linear) and varies rep_dim ∈ {50,
128, 256} × merge_family ∈ {mlp, fno_rep} plus two extra fno_rep
configurations (deeper / wider) at rep_dim=256. Aggregator writes
`round7_repdim_fno_g_summary.csv`.

### Single-cell smoke (1024 parity, ~3 min)

To verify a code change didn't break the headline cell:

```bash
SMOKE_OUT=outputs/markov_fno_smoke_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$SMOKE_OUT"
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
./venv/bin/ctreepo sim run contextual-sbijax \
  --data-source markov \
  --load-data-bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
  --sbijax-trainer learned_local_laws --sbijax-method nasss --sbijax-package-theta markov_exact_sketch \
  --sbijax-input-encoding regime_one_hot --local-law-summary-family jax_fno \
  --local-law-summary-fno-n-modes 32 --local-law-summary-fno-n-layers 3 --local-law-summary-fno-pooling-mode sum \
  --law-architecture fully_learned --c2-merge-target theta \
  --merge-family mlp --decoder-head mlp --local-law-merge-loss mse \
  --learned-merge-hidden-dim 128 --learned-decoder-hidden-dim 128 \
  --train-docs 1024 --val-docs 256 --test-docs 256 \
  --fragment-len 64 --context-samples-per-doc 1 \
  --response-signature-contexts 16 --response-signature-slices 8 \
  --embedding-dim 64 --state-dim 25 --hidden-dim 128 \
  --learning-rate 0.0003 --lr-schedule cosine \
  --n-iter 200 --batch-size 128 \
  --local-law-weight 1.0 --local-law-leaf-weight 1.0 --local-law-merge-weight 1.0 \
  --local-law-idempotence-weight 0.0 --local-law-contextual-weight 1.0 --local-law-package-weight 0.0 \
  --local-law-count-only --local-law-rep-dim 0 \
  --seed 0 --output-root "$SMOKE_OUT"
```

Expected: `theta_count_raw_mae ≈ 0.8180` (Round 5 baseline at this cell).

## Current Interpretation

1. **Data scaling closes most of the count_mae gap.** From 1024 to 102400
   train docs, the headline `count_only` cell goes 0.82 → 0.12 → 0.027
   — a 30× improvement. The architectural ceiling drops 4000× to 0.0005.
2. **g is *not* the bottleneck at 102400.** Across the 12 g-side
   ablations and 8 rep_dim × architecture variants, count_mae stays
   within ~2× (0.022-0.045). The cheap win is `decoder_head=linear`
   (~19% improvement over the R5 default).
3. **The 50× residual gap from best-flexible (0.022) to architectural
   ceiling (0.0005) lives on the f side.** Either encoder mis-specification
   (the leaf FNO can't cleanly compute "sum of regime transitions") or
   optimization (gradient descent doesn't find the exact-zero parameters
   from the contextual signal in n_iter=200).
4. **NASS / NASSS contrastive merge supervision ≈ MSE.** No clear
   advantage at this scale and budget. May pay off at n_iter=4000.
5. **FNO-as-g works at proper sizing** (rep-dim spatial axis with ≥16
   spectral modes), is on par with MLP, and does not improve with more
   capacity in the count_only regime — the problem is fundamentally
   low-dim.

## Recommended Next Experiments

The natural next moves all live on the f side:

1. **Re-run R6/R7 winners at n_iter=4000** at the 102400 cell to see
   whether the 0.022 floor drops further with more compute. Cheap (~5h
   on 3 GPUs for 3-5 cells); answers whether the gap is compute-limited
   or architecture-limited.
2. **f-side architecture sweep at 102400.** Vary leaf-FNO `n_modes`,
   `n_layers`, `hidden_dim` (currently fixed at 32 / 3 / 128). The
   residual is here.
3. **Symmetric vs asymmetric leaf-encoder pooling.** Current
   `--local-law-summary-fno-pooling-mode sum` mixes left/right
   information uniformly; alternatives (mean, max, attention) could
   change first/last-regime recoverability.
4. **Compute-vs-data Pareto.** Hold step-count constant across data
   sizes (n_iter scaled as 1/N) for a true scaling-law comparison
   instead of "fixed epochs through varying data".
5. **Longer documents (`paper_hazard_panel_v1_t2048`)** with leaf=64-256.
   At t=128 the merge tree is shallow; longer docs would test merge
   composition at depth.

## Rounds 8 / 9 / 10 (2026-05-07): f-side, compute, and stacking

R5/R6/R7 left "the residual lives on f" as a deduction. R8 measured it
directly; R9 settled the compute-floor question; R10 stacked the two
winners. Headline: count_mae went from 0.0224 (R7 winner) → **0.0069**
(R10 winner) — a 3.2× drop from one round of f-side and compute work.

### R8 — f-side architecture sweep (12 cells, n_iter=200)

Hold g at the R7 winner; vary the leaf-FNO. All cells at
102400 / leaf=64 / regime_oh / count_only. Sorted by count_mae:

```
cell                 modes  layers  pool   count_mae
m64__l4__psum         64     4      sum    0.0117  ← R8 best
m64__l3__pmean        64     3      mean   0.0143
m64__l3__psum         64     3      sum    0.0165
m32__l4__psum         32     4      sum    0.0197
m32__l3__pmean        32     3      mean   0.0197
m64__l2__psum         64     2      sum    0.0198
m32__l3__psum         32     3      sum    0.0224  ← R7-winner config
m16__l4__psum         16     4      sum    0.0291
m32__l2__psum         32     2      sum    0.0344
m16__l3__pmean        16     3      mean   0.0332
m16__l3__psum         16     3      sum    0.0326
m16__l2__psum         16     2      sum    0.0438
```

Findings:

1. **f IS the bottleneck — direct measurement.** m16 → m64 cuts
   count_mae ~3.7×. Going from the R5/R7 default (m32/l3/sum) to
   m64/l4/sum drops count_mae from 0.0224 to 0.0117 (~2×).
2. Mean pooling beats sum pooling at matched (modes, layers) by
   10-13%.
3. The summary-FNO modes cap at `L//2+1` (33 at fragment_len=64), so
   `m=64` is effectively `m=33`. The improvement vs `m=32` is
   nonetheless real (~26%) — JAX init shape and effective spectral
   coverage both shift slightly.

### R9 — compute scaling at the R7-winner architecture (3 cells)

Hold the R7 winner (m32/l3/sum + mlp+linear+nass_jsd) and vary
`n_iter ∈ {500, 1500, 4000}`. All cells at 102400 / leaf=64.

```
n_iter   count_mae
   200    0.0224  (R7 baseline, reference)
   500    0.0168
  1500    0.0140
  4000    0.0141  ← compute saturated at n_iter≈1500
```

Finding: at the R7 architecture, n_iter=200 was a compute floor;
doubling iters drops mae by ~37%. Past n_iter=1500 there is no further
gain. So R7 was simultaneously compute-limited and architecture-limited.

### R10 — stacked winner (5 cells, n_iter=1500)

Stack the R8 architecture winner (m64/l4/sum) with the R9 compute
budget (n_iter=1500). All cells at 102400 / leaf=64.

```
cell                modes  layers  pool   count_mae
m64__l5__pmean       64     5      mean   0.00690  ← R10 best
m64__l4__pmean       64     4      mean   0.00781
m64__l5__psum        64     5      sum    0.00785
m64__l4__psum        64     4      sum    0.00801
m32__l4__psum        32     4      sum    0.01110  ← control
```

The m32 control confirms the m32 → m64 effect from R8 carries to l=4
(28% drop). Mean pooling continues to beat sum (3-6% at the new
architecture). Going l=4 → l=5 buys another ~3-13% depending on
pooling.

### R11 — compute scaling at the R10-winner architecture (1 cell, n_iter=4000)

Single-cell follow-up to R10: m64/l5/mean at n_iter=4000 vs R10's
n_iter=1500.

```
n_iter   count_mae
  200    0.0224  (R7 baseline)
 1500    0.00690 (R10 winner)
 4000    0.00640 (R11)
```

**The R10-winner architecture is architecture-limited, not
compute-limited.** Going from n_iter=1500 to n_iter=4000 buys ~7%
(within noise). The 0.0069 floor at m64/l5/mean is essentially the
architecture's floor at this f/g configuration. Further reductions need
structural changes (longer fragments past the L/2+1 modes cap, or wider
hidden_dim).

### Trajectory: R5 → R10 in one table

| stage | config | count_mae | × better than 1024 baseline |
|---|---|---:|---:|
| R5 baseline (1024 docs) | m32/l3/sum, R7-default-g, n_iter=200 | 0.82 | 1× |
| R5 winner (102400 docs) | m32/l3/sum, default-g, n_iter=200 | 0.027 | 30× |
| R7 winner (g optimal) | m32/l3/sum, mlp+linear+nass_jsd, n_iter=200 | 0.0224 | 37× |
| R8 winner (f optimal) | m64/l4/sum, R7-g, n_iter=200 | 0.0117 | 70× |
| R9 (compute) | m32/l3/sum, R7-g, n_iter=1500 | 0.0140 | 59× |
| **R10 stacked** | **m64/l5/mean, R7-g, n_iter=1500** | **0.0069** | **119×** |
| **R11 (compute)** | **m64/l5/mean, R7-g, n_iter=4000** | **0.0064** | **128×** |
| Architectural ceiling | regime_transition_sum, n_iter=200 | 0.0005 | 1640× |

After R10 the residual gap to the architectural ceiling is ~14× (down
from ~50× at R7). The four learning axes (data, compute, f-architecture,
g-architecture) are now characterized for t=128 / paper_hazard_panel_v1.

### What R8-R10 changed in code

No new model surface — R8/R9/R10 use the R6/R7 plumbing. The R10 winner
config:
- `--local-law-summary-fno-n-modes 64`
- `--local-law-summary-fno-n-layers 5`
- `--local-law-summary-fno-pooling-mode mean`
- `--merge-family mlp --decoder-head linear --local-law-merge-loss nass_jsd`
- `--n-iter 1500 --batch-size 128`

### How to reproduce R8/R9/R10

```bash
GPUS=0,1 STAMP=$(date -u +%Y%m%d_%H%M%S) bash scripts/run_markov_fno_round8_f_arch.sh
GPUS=2,3 STAMP=$(date -u +%Y%m%d_%H%M%S) bash scripts/run_markov_fno_round9_compute_scaling.sh
GPUS=0,1,2,3 STAMP=$(date -u +%Y%m%d_%H%M%S) bash scripts/run_markov_fno_round10_stacked_winner.sh
GPU=1 STAMP=$(date -u +%Y%m%d_%H%M%S) bash scripts/run_markov_fno_round11_compute_at_r10_winner.sh
```

### CSV digests

- `outputs/markov_fno_round8_f_arch_20260507_011248/round8_f_arch_summary.csv`
- `outputs/markov_fno_round9_compute_scaling_20260507_011250/round9_compute_scaling_summary.csv`
- `outputs/markov_fno_round10_stacked_winner_20260507_185814/round10_stacked_winner_summary.csv`

## Rounds 12 / 13 (2026-05-07): t=2048 longer-DGP generalization

R5-R10 characterized recovery on `paper_hazard_panel_v1_t128` (12-regime,
doc length 128, shallow merge tree). R12 + R13 extend to
`paper_hazard_panel_v1_t2048` (same 12-regime DGP, doc length 2048,
deeper merge trees: leaf=64 → 32 leaves/doc → 5-level merge tree).
Headline: **the architectural ceiling generalizes (rts ≈ 0.001 at
t=2048), the R10 winner architecture transfers, and deeper merge trees
help, not hurt.**

### R12 — t=2048 smoke (4 cells, 10240 docs, batch=16, iter=400, ~7 min/cell)

```
cell                count_mae    interpretation
rts__leaf128        0.000688    architectural ceiling generalizes ≈ t=128
r7base__leaf128     0.154       under-trained at this compute budget
r10win__leaf128     0.249       R10-arch is more compute-sensitive
r10win__leaf256     0.488       even worse with shallower tree at low compute
```

R12 looked alarming — the R10 winner architecture (`m64/l5/mean`) was
*worse* than the R7 baseline (`m32/l3/sum`) at t=2048. Hypothesis at
the time: either under-training or wrong inductive bias for deep trees.
R13 settled it.

### R13 — t=2048 headline (5 cells, 102400 docs, batch=128, iter=200, ~30-50 min/cell)

Matched-to-R5 schedule on the fresh 102400-doc t=2048 bundle.

```
cell                  count_mae   ratio-to-ceiling
rts__leaf128          0.000946    1×        (architectural ceiling)
r10win__leaf64        0.00988    10.4×      ← best flexible learner
r7base__leaf64        0.0178     18.8×
r10win__leaf128       0.0231     24.4×
r7base__leaf128       0.0761     80.4×
```

Findings:

1. **Architectural ceiling at t=2048/leaf=128 = 0.000946**, on par with
   the t=128 ceiling (0.0005). Both are essentially the irreducible
   noise floor.
2. **R10-winner f-architecture transfers** — at every leaf, m64/l5/mean
   beats m32/l3/sum (3-4× better at leaf=128, ~2× at leaf=64).
3. **Deeper merge trees help.** Both architectures land 2-3× lower
   count_mae at leaf=64 (32-leaf tree) than at leaf=128 (16-leaf tree).
   More merges → more supervision per doc.
4. **R12 was a compute floor**, not an inductive-bias mismatch. With
   matched-to-R5 compute (102400 docs, batch=128, iter=200), the
   architecture transfers cleanly. R12 still useful: it shows the
   R10-winner config has higher minimum compute than R7-baseline.
5. **At iter=200, t=2048 is closer to its ceiling than t=128.** R8's
   best-flexible-at-iter=200 was m64/l4/sum=0.0117 → 23× from t=128's
   ceiling. R13's best at the same compute: 0.00988 → 10× from t=2048's
   ceiling. Deeper trees give more per-doc gradient signal.

### Wall-time correction

R12/R13 finished much faster than my pre-launch estimate. The "16×
per-step at t=2048" extrapolation was overstated:

- R12: 4 cells in ~30 min total (~7 min/cell) at batch=16/iter=400.
- R13: 5 cells in ~50 min total (~30-50 min/cell) at batch=128/iter=200,
  which is the same wall as t=128 R5 cells at this schedule.

Most of the t=2048 overhead is absorbed by the mini-batched merge
supervision and the chunked eval-time apply_fn. Per-step at fixed
batch_size scales much less than 16× of the t=128 cost; closer to 3-5×.

### R11 (compute at R10 winner, t=128) — in flight

The single n_iter=4000 cell at the R10-winner architecture started
earlier today (~21:00 UTC) and is still running. Will be appended once
it lands; the question it answers is whether 0.0069 is compute-limited
or architecture-limited at the m64/l5/mean architecture.

### How to reproduce R12 / R13

```bash
# R12 smoke (uses existing 10240-doc t=2048 bundle):
GPUS=2,3 STAMP=$(date -u +%Y%m%d_%H%M%S) bash scripts/run_markov_fno_round12_t2048_screen.sh

# R13 headline (requires the 102400-doc t=2048 bundle, generated via:
#   ./venv/bin/python scripts/prepare_markov_hazard_panel_data.py \
#     --panel-ids paper_hazard_panel_v1_t2048 --train-docs 102400 \
#     --val-docs 1024 --test-docs 1024 --seed 0 \
#     --bundle-root outputs/_bundles/markov_hazard_panels_train102400_t2048 \
#     --skip-prepared-cache )
GPUS=0,2,3 STAMP=$(date -u +%Y%m%d_%H%M%S) bash scripts/run_markov_fno_round13_t2048_headline.sh
```

### CSV digests

- `outputs/markov_fno_round12_t2048_screen_20260507_213741/round12_t2048_screen_summary.csv`
- `outputs/markov_fno_round13_t2048_headline_20260507_222049/round13_t2048_headline_summary.csv`

## Lean Crosswalk (unchanged from 2026-05-05)

The Lean stack still anchors *exactness of the sketch* and *contextual
sufficiency*, not SGD convergence. See
[`markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`](markov_contextual_sufficiency_ablation_handoff_2026-05-05.md)
"Lean Crosswalk" section for the full mapping; nothing changes with
R5/R6/R7 — the empirical claim is that data scaling drives the flexible
learner's count_mae toward the architectural ceiling, but the *exact*
zero result still requires the architectural prior (`rts`) or local-law
supervision over the sketch.

## Cross-Round Cell Counts (running totals)

| round | cells | n_iter | wall (3 GPUs) | what it tests |
|---|---:|---:|---:|---|
| R3 unified | 55 | 4000 | ~2h | f family × encoding × leaf at 1024 docs |
| R3 sbijax-internal | 16 | 4000 | ~30 min | NASS/NASSS aux + nass_jsd merge at 1024 docs |
| R4 sbijax ablations | 51 | 4000 | ~70 min | 4 sbijax tiers (pure, everywhere, weight sweep, learned-decoder) at 1024 docs |
| **R5 smoke20** | **12** | **20** | **~13 min** | **headline data-scaling screen** |
| **R5 data scaling** | **30** | **200** | **~2h** | **train_docs ∈ {1024, 10240, 102400} headline matrix** |
| **R6 g-ablation** | **12** | **200** | **~1.5h** | **merge_family × merge_loss × decoder_head at 102400/leaf=64** |
| **R7 rep_dim × FNO-as-g** | **8** | **200** | **~1h** | **rep_dim ∈ {50, 128, 256} × merge_family at 102400/leaf=64** |
| **R8 f-arch sweep (2026-05-07)** | **12** | **200** | **~35 min** | **leaf-FNO modes × layers × pooling at 102400/leaf=64** |
| **R9 compute scaling (2026-05-07)** | **3** | **500-4000** | **~5h** | **n_iter ∈ {500,1500,4000} at R7-winner architecture** |
| **R10 stacked winner (2026-05-07)** | **5** | **1500** | **~75 min/cell** | **m64/{l4,l5}/{sum,mean} + m32 control at 102400/leaf=64** |
| **R11 compute @ R10-winner (2026-05-07)** | **1** | **4000** | **~3h** | **m64/l5/mean at n_iter=4000 — confirmed R10-arch is architecture-limited (0.0064 vs 0.0069)** |
| **R12 t=2048 smoke (2026-05-07)** | **4** | **400** | **~30 min** | **t=2048 at 10240 docs / batch=16 — under-trained negative** |
| **R13 t=2048 headline (2026-05-07)** | **5** | **200** | **~50 min** | **t=2048 at 102400 docs / batch=128 — paper-grade transfer result** |

Bold rows are the focus of this handoff.
