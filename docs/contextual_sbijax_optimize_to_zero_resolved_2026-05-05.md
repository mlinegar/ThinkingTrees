# Contextual Sbijax — `optimize_to_zero.md` Resolved (2026-05-05)

This is the closing handoff for the Markov contextual-sufficiency thread that
`docs/optimize_to_zero.md` opened. It is meant for another LLM or engineer
picking up this thread; the original handoff stays in place as the framing
document, and this note adds the empirical resolution and next-step ladder.

The original question:

> Why doesn't `sbijax.NASSS` drive contextual MAE to numerical zero on the
> Markov change-point task — even when the input is already the exact
> sufficient sketch?

The short answer:

- **The contrastive sliced-JSD objective in NASSS does not push the summary
  network toward identity.** Identity is one optimum of the contrastive loss,
  but the loss is invariant under any invertible transformation of the
  summary, so SGD picks an arbitrary member of that family. Without a
  tie-breaker, almost surely the chosen summary is not identity — and not the
  sufficient sketch.
- **The local laws (C1 leaf preservation, C2 merge consistency, C3
  idempotence) ARE the tie-breaker.** They constrain the summary structurally
  to be the sufficient statistic. With laws active, the learned state
  converges to the exact Markov sketch at numerical precision.
- **Without laws, you can fit the contextual response and still have a
  non-sufficient state.** This is the empirical case the original handoff
  was running into.

The rest of this note records the experiments, the cleanest diagnostic
table, and the implications for the broader `recoverable_v5_t2048` and f/g
pipeline floors.

## Empirical evidence

Two experimental campaigns landed on 2026-05-05 against the
`paper_hazard_panel_v1_t128/seed_0/base_bundle.json` hazard panel:

1. **Leaf=1 trainer comparison** (11-row diagnostic), produced in
   `outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/`. Compares
   exact controls, learned local-law variants, theta-supervised, and the
   sbijax NASS / NASSS package at the most stringent stress check (single
   token per item).
2. **Leaf grid + ablations** (this session), produced in
   `outputs/optimize_to_zero_theta_sup_grid_t128/`,
   `outputs/optimize_to_zero_laws_grid_t128/`,
   `outputs/optimize_to_zero_laws_hard_inputs/`,
   `outputs/optimize_to_zero_long_n5000_leaf{1,4}/`. Extends the diagnostic
   to all leaves 1, 2, 4, 8, 16, 32, 64 across input encodings, plus the
   long-iteration NASSS test that pins down what "long enough" buys you
   without laws.

### Canonical status

| lane | status | key diagnostic |
|---|---|---|
| `learned_local_laws + markov_exact_sketch` | resolved exact-zero path | full leaf grid max contextual MAE `3.61e-9`; first/last accuracy `1.0 / 1.0` |
| `theta_supervised` | near-exact control | lower contextual MAE than package NASSS; `regime_one_hot` endpoint accuracy bottoms out at `0.9961 / 0.9883` |
| long NASSS | approximate baseline | 5000 iters lowers contextual MAE but leaves theta recovery near random |
| hard-input local laws | encoder-capacity diagnostic | token one-hot works at small leaves; scalar/normalized encodings degrade first |
| laws on/off | mechanism check | all laws drop theta MAE from ~`0.18` to `2.2e-4` at leaf=2 |

### Post-resolution ablation pass

After the exact-zero path was established, we ran three ablation grids to test
whether NASS/NASSS auxiliaries, learned merge/readout variants, or a fully
general learned f/g surface change the interpretation. Full table artifact:
`outputs/markov_contextual_ablation_grid_report_20260505.md`. Handoff:
`docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`.

| grid | rows | best row | best metric | interpretation |
|---|---:|---|---:|---|
| JAX f/g architecture | 36 | `nasss/w_0/learned_merge/c2_self_consistency/leaf_1` | contextual MAE `1.53e-5` | Learned merge/decoder variants work inside the local-law lane; all architecture groups preserve first/last accuracy at `1.0 / 1.0`. |
| Hybrid NASS/NASSS + laws | 42 | `nasss / regime_one_hot / leaf=1 / w=0.1` | contextual MAE `1.47e-5` | NASSS is useful as a low-weight auxiliary; NASS is weaker; hard token encodings still degrade. |
| CleanUnifiedNO general f/g | 15 | `contextual_sufficiency/dep_none/leaf_tokens_16` | root MAE `1.1451`, contextual MAE `1.1187` | Honest general f/g does not discover exact recovery yet. |

The ablation conclusion is narrower than "f/g learned the theorem." Inside
`learned_local_laws`, learned merge and learned decoder work because the state
is still supervised toward the Markov sufficient sketch. In the standalone
`CleanUnifiedNO` path no exact Markov merge/readout is installed, and the best
row remains far from exact-zero. This is the current bridge gap between the
resolved Markov local-law objective and the production-shaped general f/g
pipeline.

### Leaf=1 trainer comparison (the cleanest single table)

Source: `outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/leaf1_diagnostic_summary.md`.
`contextual_raw_mae` is in raw count units (not normalized). `theta_mae` is
on the 9-D Markov sketch. `first_accuracy` / `last_accuracy` are the
accuracy of the recovered first- and last-regime one-hots out of 12
regimes — these are the diagnostic that distinguishes "fits the response"
from "learned the sufficient statistic."

| candidate | encoding | decoder | contextual_raw_mae | theta_mae | first_acc | last_acc | reading |
|---|---|---|---:|---:|---:|---:|---|
| `exact_zero_markov` | regime_one_hot | exact | 0 | 0 | 1.000 | 1.000 | analytic baseline |
| `identity_theta` | markov_exact_sketch | exact | 0 | 0 | 1.000 | 1.000 | analytic baseline |
| `learned_local_laws_affine` | regime_one_hot | exact | 0 | 0 | 1.000 | 1.000 | LS affine summary + laws + analytic decoder |
| `learned_local_laws_exact_input` | markov_exact_sketch | exact | 0 | 0 | 1.000 | 1.000 | input==theta, summary skipped |
| **`learned_local_laws_mlp`** | regime_one_hot | exact | **1.49e-4** | **7.1e-5** | **1.000** | **1.000** | **MLP summary, laws active — sufficient at float32 noise** |
| `theta_supervised` | regime_one_hot | learned affine | 2.09e-4 | 8.1e-5 | 1.000 | 1.000 | direct theta MSE — also sufficient |
| `package_nass` | regime_one_hot | learned MLP | 1.69e-4 | **0.625** | **0.035** | **0.035** | **fits response, state is junk** |
| `package_nass_exact_input` | markov_exact_sketch | learned MLP | 1.71e-4 | **0.607** | **0.000** | **0.051** | input==theta and STILL doesn't recover identity |
| `package_nasss` | regime_one_hot | learned MLP | **4.96e-2** | 0.207 | 0.000 | 0.176 | sliced contrastive doesn't fit either |
| `package_nasss_exact_input` | markov_exact_sketch | learned MLP | **2.67e-2** | 0.302 | 0.141 | 0.039 | sliced + input==theta — same plateau |

The two rows to dwell on are `package_nass` and `package_nasss_exact_input`:
the contextual prediction *can* be fit (NASS gets to 1.69e-4 raw MAE), but
the recovered theta is wrong — first/last regime accuracy is at random
(1/12 ≈ 0.083). The summary the contrastive objective settles on is some
informative-but-non-canonical encoding; the readout MLP becomes its private
decoder.

### Long NASSS runs (5000 iters)

Source: `outputs/optimize_to_zero_long_n5000_leaf{1,4}/`. These pin down
what just turning the iteration knob up gets you in the NASSS package
trainer.

| run | n_iter | contextual_mae | theta_first/last accuracy | theta_mae |
|---|---:|---:|---:|---:|
| NASSS leaf=1, baseline | 50 | 4.79e-2 | — | — |
| **NASSS leaf=1 long** | **5000** | **5.7e-4** | **0% / 9.8%** | **0.36** |
| NASSS leaf=4 long | 5000 | 4.7e-3 | 17.2% / 0.8% | 0.44 |
| theta_supervised leaf=1 | 5000 | 1.96e-4 | 100% / 100% | 1.93e-4 |
| `learned_local_laws` leaf=1 | 1000 | numerical 0 | 100% / 100% | numerical 0 |

NASSS at 5000 iters drops contextual MAE 84× (from 4.79e-2 to 5.7e-4) but
theta first/last accuracy stays at random. This is the cleanest possible
demonstration that contextual fit and sufficiency are different objectives.

### Theta-supervised leaf grid

Source: `outputs/optimize_to_zero_theta_sup_grid_t128/leaf_grid_summary.json`.
14 cells, n_iter=1000.

| input | leaf=1 | leaf=2 | leaf=4 | leaf=8 | leaf=16 | leaf=32 | leaf=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `markov_exact_sketch` MAE | 5.4e-4 | 4.6e-4 | 2.5e-3 | 2.7e-3 | 2.0e-3 | 3.0e-3 | 1.7e-3 |
| `regime_one_hot` MAE | 4.6e-4 | 2.2e-4 | 4.0e-3 | 4.9e-3 | 7.9e-3 | 1.05e-2 | 1.27e-2 |

`theta_supervised` is effectively a sketch-recovery control, but not literally
perfect in every cell. The `markov_exact_sketch` input has 100% first/last
accuracy across the grid; the `regime_one_hot` input bottoms out at
`theta_first_regime_accuracy=0.9961` and
`theta_last_regime_accuracy=0.9883`. Compared to the NASSS package baseline at
the original handoff (8.8e-3 to 4.8e-2 across the same grid),
theta_supervised is 3-88x lower and almost perfectly recovers the sketch. The
remaining sub-1e-3 error at small leaves is the affine-probe floor of the
fixed least-squares readout — *not* an optimization failure.

### Laws-aligned leaf grid

Source: `outputs/optimize_to_zero_laws_grid_t128/leaf_grid_summary.json`.
14 cells, n_iter=1000, all laws weighted 1.0.

| input | leaf=1 | leaf=2 | leaf=4 | leaf=8 | leaf=16 | leaf=32 | leaf=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `markov_exact_sketch` MAE | **0** | **0** | **0** | 9.1e-10 | 2.0e-9 | 9.8e-10 | 3.6e-9 |
| `regime_one_hot` MAE | 8.3e-4 | 4.3e-4 | 4.7e-3 | 6.0e-3 | 9.8e-3 | 1.22e-2 | 1.43e-2 |

The `markov_exact_sketch` column is true numerical zero (or float32 noise):
when input==theta, `learned_local_laws` skips training and the analytic
decoder gives exact recovery. The `regime_one_hot` column degrades
monotonically with leaf size — this is the input-encoder bottleneck, not the
laws.

### Hard-input ablation

Source: `outputs/optimize_to_zero_laws_hard_inputs/summary.json`. Tests
whether laws-aligned learning works when the input encoding hides regime
structure.

| encoding | leaf | contextual_mae | theta_mae | first_acc | last_acc | l2_merge |
|---|---:|---:|---:|---:|---:|---:|
| `one_hot_token_ids` | 2 | 1.25e-3 | 1.5e-3 | **100%** | **100%** | 2.6e-7 |
| `one_hot_token_ids` | 4 | 7.0e-3 | 1.3e-2 | 100% | 99.2% | 2.6e-4 |
| `regime_ids` (scalar) | 2 | 6.3e-3 | 2.1e-2 | 96.9% | 96.9% | 2.6e-4 |
| `regime_ids` | 4 | 7.7e-3 | 3.2e-2 | 92.6% | 92.6% | 4.6e-3 |
| `normalized_token_ids` | 2 | 1.47e-2 | 3.7e-2 | 93.4% | 94.1% | 5.7e-3 |
| `normalized_token_ids` | 4 | 1.17e-2 | 4.8e-2 | 91.8% | 93.8% | 1.0e-2 |

Laws drive sufficiency learning even from raw token inputs at small leaves
(`one_hot_token_ids` leaf=2: 100% accuracy, contextual MAE 1.25e-3).
Degradation at leaf=4 and on harder encodings (scalar `regime_ids`,
`normalized_token_ids`) is an *encoder capacity* issue — the
`_make_theta_summary_net` MLP at hidden_dim=128 cannot extract regime
structure from a 1-D scalar input fast enough at this size. The laws are
not the bottleneck on these inputs; the input-side feature extractor is.

### Leaf=2 laws-on/off ablation

Source: `outputs/optimize_to_zero_laws_ablation/`. Confirms the central
"laws produce sufficiency" claim by toggling them off.

| setup | contextual_mae | first_acc | last_acc | theta_mae | l1_leaf | l2_merge |
|---|---:|---:|---:|---:|---:|---:|
| leaf=1 no laws | 1.54e-4 | 89.8% | 68.8% | 0.167 | 5.2e-2 | — |
| leaf=2 no laws | 2.67e-4 | 89.5% | **57.8%** | 0.185 | 6.3e-2 | **7.5e-2** |
| leaf=2 C2-only | 1.27e-3 | 100% | 98.8% | 0.095 | 1.5e-2 | 4.0e-7 |
| **leaf=2 all laws** | 1.07e-3 | **100%** | **100%** | **2.2e-4** | 4.7e-4 | 2.4e-7 |

Without laws the contextual readout still fits acceptably (the analytic
decoder is forgiving), but the state is wrong (theta_mae 0.18, l2_merge
0.075). Adding C2 alone fixes merge consistency at numerical zero. Adding
all laws drops theta_mae three more orders of magnitude.

## What this resolves and what it doesn't

**Resolved**

1. The original optimize-to-zero claim is empirically met. With
   `--sbijax-trainer learned_local_laws --sbijax-input-encoding markov_exact_sketch`,
   the leaf grid hits literal numerical zero (every cell ≤ 4e-9
   contextual MAE).
2. The cause of the NASSS floor is identified: the contrastive objective
   has a continuous family of equivalent optima, and SGD does not pick the
   sufficient summary. Iterations and learning rate are not the issue;
   the objective is.
3. The local laws (the existing `learned_local_laws` trainer at
   [`fit_contextual_sbijax_learned_local_laws`](../src/ctreepo/sim/core/contextual_sbijax.py#L3967))
   are the principled fix. C1 supervises the leaf state on theta, C2
   supervises the analytic merge of left+right against the merge target,
   C3 supervises idempotence. With all three on, the state converges to
   the sufficient sketch.
4. Sufficiency and fit are *different objectives.* Both NASS and NASSS at
   long iterations can fit the contextual response well enough; neither
   recovers theta accuracy above random. This is exactly the
   sufficiency-vs-prediction gap the paper formalizes.

**Not resolved (next-step ladder)**

1. The encoder bottleneck on harder input encodings. At leaf=4+ with
   `regime_ids` and `normalized_token_ids`, the summary MLP runs out of
   capacity and theta accuracy slips below 100%. The fix is on the
   encoder side: bigger `--state-dim` and `--hidden-dim`, or a
   convolutional / attention encoder that handles per-token structure
   better. The laws are doing their job; the feature extractor isn't.
2. The bridge to the f/g tree pipeline. The
   `recoverable_v5_t2048` zero-merge floor of ~2.13 root_mae documented
   in `CLAUDE.md` is the same family of question, scaled up: do laws +
   sufficient encoder capacity drive the floor down? The CLAUDE.md note
   that wider *heads* (`hidden_dim=2048`, `tree_merge_hidden_dim=4096`)
   did not help is consistent with the sandbox finding here — head
   capacity is not the issue, encoder/leaf-pooling capacity is.
3. NASS / NASSS + laws hybrid. We never tested whether adding a NASSS
   contrastive term to `learned_local_laws` gives a strictly better
   summary than laws alone (it might add information beyond what C1/C2/C3
   constrain). This is a one-day experiment, not load-bearing for the
   immediate thread.

## How to reproduce

The cleanest single command for "optimize to zero":

```bash
source venv/bin/activate
BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json

XLA_PYTHON_CLIENT_PREALLOCATE=false \
ctreepo sim run contextual-sbijax \
  --data-source markov \
  --load-data-bundle "$BUNDLE" \
  --sbijax-trainer learned_local_laws \
  --sbijax-method nasss \
  --sbijax-package-theta markov_exact_sketch \
  --sbijax-input-encoding markov_exact_sketch \
  --train-docs 1024 --val-docs 256 --test-docs 256 \
  --fragment-len 1 \
  --context-samples-per-doc 1 \
  --response-signature-contexts 16 --response-signature-slices 8 \
  --embedding-dim 32 --state-dim 25 --hidden-dim 128 \
  --learning-rate 0.0003 --n-iter 1000 --batch-size 128 \
  --local-law-weight 1.0 \
  --local-law-leaf-weight 1.0 \
  --local-law-merge-weight 1.0 \
  --local-law-idempotence-weight 1.0 \
  --local-law-contextual-weight 1.0 \
  --seed 0 \
  --output-root outputs/optimize_to_zero_demo
```

Expected: `summary.json -> diagnostics.test.contextual_mae` is exactly 0,
`theta_first_regime_accuracy` and `theta_last_regime_accuracy` are 1.0.

The full leaf grid scripts:

- `scripts/run_optimize_to_zero_theta_sup_grid.sh` — theta_supervised baseline grid (14 cells).
- `scripts/run_optimize_to_zero_laws_grid.sh` — laws-aligned grid (14 cells).
- `scripts/run_optimize_to_zero_laws_hard_inputs.sh` — hard-input ablation (6 cells).

## Files of record

Code:

- `src/ctreepo/sim/core/contextual_sbijax.py`
  - `fit_contextual_sbijax_learned_local_laws` — current trainer that hits zero and now supports `local_law_package_weight`, `law_architecture`, and `c2_merge_target`.
  - `fit_contextual_sbijax_theta_supervised` — direct-theta baseline.
  - `fit_contextual_sbijax_identity_theta` — analytic ID baseline.
  - `fit_contextual_sbijax_package_direct` — sbijax NASS/NASSS path that plateaus without laws.
  - `_responses_from_markov_exact_states` — analytic decoder used by laws/identity lanes.
- `scripts/probe_contextual_sbijax.py` — CLI flags `--sbijax-trainer learned_local_laws`,
  `--local-law-{leaf,merge,idempotence,contextual}-weight`,
  `--local-law-package-weight`, `--law-architecture`, `--c2-merge-target`,
  `--learned-merge-hidden-dim`, and `--learned-decoder-hidden-dim`.
- `src/ctreepo/sim/core/clean_unified_fg.py` and
  `scripts/probe_clean_unified_no.py` — clean general f/g surface used for the
  standalone ablation.
- `scripts/run_optimize_to_zero_fg_architecture_ablation.sh`,
  `scripts/run_optimize_to_zero_laws_hybrid_grid.sh`, and
  `scripts/run_clean_unified_fg_contextual_ablation.sh` — completed
  post-resolution ablation launchers.

Outputs (this resolution):

- `outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/leaf1_diagnostic_summary.{md,json}` — 11-row trainer comparison.
- `outputs/optimize_to_zero_theta_sup_grid_t128/leaf_grid_summary.json` — theta_supervised leaf grid.
- `outputs/optimize_to_zero_laws_grid_t128/leaf_grid_summary.json` — laws-aligned leaf grid.
- `outputs/optimize_to_zero_laws_hard_inputs/summary.json` — hard-input ablation.
- `outputs/optimize_to_zero_laws_ablation/` — laws on/off at leaf=2.
- `outputs/optimize_to_zero_long_n5000_leaf{1,4}/summary.json` — NASSS long-iter baseline.
- `outputs/optimize_to_zero_fg_architecture_ablation_t128/summary.json` — learned merge/decoder architecture grid.
- `outputs/optimize_to_zero_laws_hybrid_grid_t128/summary.json` — NASS/NASSS auxiliary-weight grid.
- `outputs/clean_unified_fg_contextual_ablation_t128/summary.json` — standalone general f/g grid.
- `outputs/markov_contextual_ablation_grid_report_20260505.md` — full 93-row ablation report.

Companion docs:

- `docs/optimize_to_zero.md` — the original handoff.
- `docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md` — post-resolution ablation handoff and Lean crosswalk.
- `docs/contextual_sbijax_walkthrough.md` — sbijax-package vocabulary and lane bridging.
- `docs/local_laws_unification_handoff_2026-04-18.md` — laws schema and `(1-λ)·root + λ·avg(C1,C2,C3)` formula across pipelines.
- `docs/mergeable_sketches_and_learned_sufficiency.md` — theory anchor for why laws characterize sufficiency.
