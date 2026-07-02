# Markov Simulation Status (2026-05-07)

This is the current high-level status page for the Markov C-TreePO simulation
thread. It is meant to be the first place to check before deciding what to run
next. More detailed handoffs remain linked below.

## What's New (2026-05-07)

**R8/R9/R10 stacked the f-side and compute-axis winners.** Headline:
count_mae **0.0224 (R7) → 0.0069 (R10)** at 102400 docs / leaf=64 / t=128
— a 3.2× drop in one round. Trajectory from the 1024-doc R5 baseline to
the R10 winner is **0.82 → 0.0069 (119×)**; the architectural ceiling
(`regime_transition_sum`) sits at 0.0005 (1640× from the baseline), so
the residual gap to the ceiling is now ~14× (down from ~50× at R7).

R8 (12 cells, n_iter=200) directly varied the leaf-FNO and showed f IS
the bottleneck — m16/l2/sum=0.0438 → m64/l4/sum=0.0117 (3.7×). Mean
pooling beats sum at matched (modes, layers) by 10-13%.

R9 (3 cells) settled compute at the R7 architecture: n_iter=200 (0.0224)
→ n_iter=1500 (0.0140); n_iter=4000 (0.0141) is flat.

R10 (5 cells, n_iter=1500) stacked the R8 architecture winner with the
R9 compute budget. Best: m64/l5/mean = 0.0069. The m32/l4/sum control
landed at 0.0111 — confirms m32 → m64 carries to l=4.

**R12/R13 generalize the recovery story to t=2048** (longer DGP, deeper
merge trees: leaf=64 → 32 leaves/doc → 5-level tree). At the matched-to-R5
schedule (102400 docs / batch=128 / iter=200) the t=2048 architectural
ceiling lands at **0.000946** (≈ t=128's 0.0005); the R10-winner
architecture (m64/l5/mean) lands at **0.00988 at leaf=64** — within 10×
of the ceiling, which is *closer* than t=128 at the same compute (R8 was
23× from its ceiling). Deeper merge trees help, not hurt: leaf=64 beats
leaf=128 by 2-3× for both architectures. R12 (smoke at batch=16/iter=400)
showed R10-arch *worse* than R7-baseline — this turned out to be a
compute floor, not an architectural mismatch. Full write-up in
[`markov_data_scaling_g_ablation_handoff_2026-05-06.md`](markov_data_scaling_g_ablation_handoff_2026-05-06.md)
("Rounds 8/9/10" and "Rounds 12/13" sections).

R11 settled the compute question for the R10-winner architecture: at
n_iter=4000 the count_mae is **0.0064** vs R10's **0.0069** at
n_iter=1500 — only ~7% better, within noise. **The R10 architecture is
architecture-limited at iter=1500**, not compute-limited. Further
reductions in t=128 count_mae need structural changes (longer fragments
past the L/2+1 modes cap; wider hidden_dim).

## What's New (2026-05-06)

Three new rounds (R5/R6/R7) extended the JAX `learned_local_laws` lane along
the data-scaling and g-architecture axes. **Headline: data scaling closes
most of the count_mae gap.** At 102400 train docs (100× the prior baseline),
the flexible-encoder `count_only` cell hits `count_mae = 0.027` and the
architectural ceiling (`regime_transition_sum`) hits `count_mae = 0.0005`.
The g-side is *not* the bottleneck — across 12 g-axis ablations
(`merge_family × merge_loss × decoder_head`) and 8 rep_dim × FNO-as-g
variants, count_mae stayed within ~2× of the best result. The cheap win is
`decoder_head=linear` (~19% better than the default mlp head). Full
write-up: [`markov_data_scaling_g_ablation_handoff_2026-05-06.md`](markov_data_scaling_g_ablation_handoff_2026-05-06.md).

Engineering changes that unblocked these rounds:

- **Mini-batched merge supervision.** The pre-2026-05-06 step closed over
  the full `train_left/right_features` and ran the FNO over all
  N_train_merges every step → ~72 GiB OOM at 102400 docs. Now samples
  `merge_batch_size = min(batch_size, n_train_merges)` per step.
- **Gathers inside JIT + on-device metric accumulation** — saves 10
  per-step kernel launches and one host sync per step (~10-20% speedup).
- **Chunked eval** (`_apply_fn_chunked`) — apply_fn over the full dataset
  was OOMing the FNO; now chunks at 1024 rows.
- **N²-collision diagnostic subsampling** to 4096 rows — avoids 1.2 TiB
  host RAM blow-up at 100k+ docs.
- **New CLI flags / config:** `--merge-family {mlp, fno_rep}`,
  `--decoder-head {mlp, linear}`, `--local-law-merge-loss
  {mse, nass_jsd, nasss_jsd}`, plus FNO-merge sizing and slice count.
  All surface in `provenance`.

Parity smoke at 1024 docs: `count_mae = 0.8180` matches the pre-refactor
R5 baseline exactly (bit-equivalent for the default `mlp+mse+mlp` path).

## Framework Note (2026-05-05)

DeepMind deprecated `dm-haiku` for new projects in July 2023 in favor of
Flax. After a verification pass on `sbijax 0.3.6` (latest, both PyPI and
GitHub `main`), it still declares `"dm-haiku>=0.0.16"` and defines core
classes like `class NASSNet(hk.Module)` throughout `sbijax/_src/nn/`. Full
haiku removal would require abandoning sbijax (the standard NASS/NASSS
package), which we don't want.

**Resolved partial migration / route cleanup (2026-05-05 evening):**

- `_make_learned_merge_net` and `_make_learned_decoder_net` (purely
  internal, not sbijax-facing) are now `flax.linen` modules.
- The dormant haiku `_make_fno_summary_net` has been deleted.
- `_make_jax_fno_summary_net` is the self-contained JAX FNO summary route:
  flattened leaf features are reshaped to `(B, fragment_len, input_width)`, a
  normalized position channel is added, spectral residual blocks are applied
  with `jax.numpy.fft`, and an enriched pool reads out the Markov sketch shape.
  The old `norax_fno` option is retained only as a compatibility alias for
  `jax_fno`.
- The 3 sbijax-interface module factories
  (`_make_theta_summary_net`, `_make_regime_transition_sum_summary_net`,
  `_make_theta_summary_package_aux_net`) plus the package-density and
  posterior-network factories remain haiku because sbijax's NASS/NASSS
  estimators expect `hk.Transformed` inputs.
- `norax` and `pardax` are reference designs only. They are not required
  dependencies for the Markov route.

**Status:** new route code is split into self-contained JAX and PyTorch
wrappers with a shared output schema. Haiku remains as a transitive sbijax dep
and for summary factories that must return `hk.Transformed` objects. This is
an upstream interface constraint, not a new dependency on external FNO
packages.

## Bottom Line

The original reason for the JAX lane was to use `sbijax`, especially its
package-native neural sufficient summary estimators `sbijax.NASS` and
`sbijax.NASSS`. That remains the right historical framing. The current working
control, however, is the repo-owned JAX `learned_local_laws` lane, because the
package-native NASS/NASSS objectives fit response signatures without reliably
selecting the canonical Markov sufficient state.

`learned_local_laws` recovers the Markov sufficient state when the local-law
objective is active. Three input-encoding rungs are now mapped:

- `markov_exact_sketch` (input is the answer): trivial — decoder is identity.
- `regime_one_hot + regime_transition_sum`: works at all leaf sizes, but the
  encoder hard-codes "count = sum of adjacent inequality MLP" — this is the
  Markov boundary count formula written as architecture. It is a methodological
  control, not a learned-from-input result.
- `regime_one_hot + flat MLP`: degrades with leaf size at small data (1024
  train docs); count MAE 0.14 at leaf=16, 1.00 at leaf=64, 2.25 at leaf=128.

**As of 2026-05-06 this picture extends with data-scaling evidence.** With
100× more training data (102400 docs vs 1024), the flexible learner closes
most of the gap to the architectural ceiling: at leaf=64 / regime_one_hot /
`count_only` supervision, `count_mae` goes 0.82 → 0.12 → 0.027 across
1024 → 10240 → 102400 docs. The architectural ceiling (`regime_transition_sum`)
drops to 0.0005 at 102400 (1/√N variance + small-N optimization friction).
The residual ~50× gap from the best flexible learner (0.022) to the ceiling
(0.0005) is f-side (encoder mis-specification or optimization at finite
n_iter), not g-side — confirmed by 20-cell g-axis ablation. See
[`markov_data_scaling_g_ablation_handoff_2026-05-06.md`](markov_data_scaling_g_ablation_handoff_2026-05-06.md).

The PyTorch/FNO bridge is still open. The 8-hour Round 1 multi-leaf bridge
campaign confirmed `markov_local_laws_fno` plateaus at root MAE `~1.94` at
leaf=32 (best cell) and `markov_node_witness` collapses to constant prediction
across configurations. A Round 2 single-leaf diagnostic is currently running
and showing that the FNO encoder itself is fine (boundary BCE F1≥0.99 at
doc=32–64); the multi-leaf collapse must come from pooling calibration and/or
merge composition under laws-only supervision.

## Self-Contained Route Runners

Current route split:

- JAX route: [`scripts/run_markov_jax_route.py`](../scripts/run_markov_jax_route.py)
  runs package `sbijax.NASS/NASSS` baselines beside repo-owned JAX FNO cells
  (`jax_fno_node_witness`, `jax_fno_local_laws`) and the structured
  `regime_transition_sum` control. It reads the Markov t128 bundle directly
  and does not depend on PyTorch artifacts. The package cells are the
  response-objective baselines; `jax_fno_node_witness` is the direct
  state-target capacity diagnostic; `jax_fno_local_laws` is the pure
  local-law route with learned merge + `c2=self_consistency`, so it does not
  prescribe the merge.
- PyTorch route: [`scripts/run_markov_pytorch_route.py`](../scripts/run_markov_pytorch_route.py)
  wraps `CleanUnifiedNO` / `probe_clean_unified_no.py` for `root`,
  `contextual_none`, `markov_node_witness`, and `markov_local_laws_fno` cells.
  It reads the same bundle directly and does not depend on JAX artifacts.
- Shared schema: [`scripts/markov_route_contract.py`](../scripts/markov_route_contract.py)
  writes `summary.json`, `grid_summary.csv`, and `grid_report.md` with common
  columns including `root_count_mae`, `theta_mae`,
  `theta_first_regime_accuracy`, `theta_last_regime_accuracy`, `eps_leaf`,
  `eps_merge`, `eps_idemp`, `contextual_mae`, `pred_truth_corr`, and
  `pred_std`.

## Reading Order

1. This page.
2. [`docs/contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`](contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md)
3. [`docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`](markov_contextual_sufficiency_ablation_handoff_2026-05-05.md)
4. [`docs/markov_fno_local_law_bridge.md`](markov_fno_local_law_bridge.md)
5. Current result tables:
   - [`outputs/markov_contextual_ablation_grid_report_20260505.md`](../outputs/markov_contextual_ablation_grid_report_20260505.md)
   - [`outputs/regime_one_hot_recovery_grid_20260505_162001/grid_summary.csv`](../outputs/regime_one_hot_recovery_grid_20260505_162001/grid_summary.csv)
   - [`outputs/regime_one_hot_recovery_grid_20260505_162001/grid_report.md`](../outputs/regime_one_hot_recovery_grid_20260505_162001/grid_report.md)

## Original JAX Purpose

Do not lose this context: JAX entered the project as the natural runtime for
`sbijax` neural sufficient summaries, not as a goal in itself.

The original target was:

- use `sbijax.NASS` / `sbijax.NASSS` to learn a low-dimensional sufficient
  statistic from context-response training;
- treat the learned summary as the neural analogue of the theorem-side Markov
  sketch `(count, first, last)`;
- then test whether the learned state satisfies the local-law / contextual
  sufficiency diagnostics.

What we learned:

- package-native NASS/NASSS is valuable as the historical baseline and sometimes
  as a low-weight auxiliary;
- by itself it has a non-identifiability/tie-breaking problem: many summaries
  fit the response objective, and SGD does not necessarily choose the canonical
  sufficient sketch;
- the local laws are the structural tie-breaker that force the state toward the
  Markov sketch;
- the newer `regime_transition_sum` experiment is a structured control for the
  count-extraction bottleneck, not a replacement for the original NASS/NASSS
  neural-summary ambition.

So the clean wording is: **the JAX/sbijax experiment showed that generic neural
sufficient-summary objectives were not enough on this Markov task; adding
local-law structure made the sufficient state identifiable and learnable.**

## Current Lane Status

| lane | backend | current status | interpretation |
|---|---|---|---|
| Exact Markov sketch local laws | JAX + Haiku + sbijax | Resolved / working | `learned_local_laws + markov_exact_sketch` is the exact-zero control. |
| JAX architecture ablations | JAX + Haiku + sbijax | Working inside local-law lane | `analytic`, `learned_merge`, `learned_decoder`, and `fully_learned` variants work when the local-law state signal is active. |
| Package NASS/NASSS | JAX + sbijax package objectives | Useful only as auxiliary | NASSS can help at low weight, but package contrastive objectives alone do not force the canonical sufficient state. |
| `regime_one_hot` with flat MLP summary | JAX + Haiku | Fails at large leaves | First/last are already exact, but count extraction degrades with leaf size. |
| `regime_one_hot` with `regime_transition_sum` summary | JAX + Haiku | Working but architecturally hard-coded | Encoder is "MLP edge score → sum"; this is the Markov boundary count formula written as architecture, not a discovery result. Useful as a methodological control. |
| `CleanUnifiedNO` contextual/general f/g | PyTorch + FNO | Not solved | The prior contextual ablation bottomed out around root MAE `1.1451`; it did not discover the exact Markov law. |
| FNO local-law bridge — Round 1 multi-leaf campaign | PyTorch + FNO | Bridge not solved | Best `markov_local_laws_fno` plateaus at root MAE `1.94` at leaf=32. `markov_node_witness` collapses to constant prediction (`pred_std ≈ 2e-4`). |
| FNO local-law bridge — Round 2 Stage 1 single-leaf diagnostic | PyTorch + FNO | In progress / encoder confirmed working | Boundary BCE F1≥0.99 at doc=32 and doc=64 single-leaf. Witness/laws full-exact rate 0.83–0.99 at doc=32. Refines the failure mode: encoder is fine; pooling calibration and/or merge composition is the bottleneck. |
| t2048 composition stress | PyTorch production-shaped pipeline | Still open | The old `recoverable_v5_t2048` ~`2.13` floor has not yet been moved by the new local-law bridge. |

## What Changed Recently

### JAX `regime_transition_sum`

Implemented `local_law_summary_family="regime_transition_sum"` for
`learned_local_laws`.

The summary family:

- is valid only with `--sbijax-input-encoding regime_one_hot`;
- reshapes flattened regime one-hot features back to
  `(batch, fragment_len, n_regimes + 1)`;
- scores adjacent regime-pair features with a learned MLP;
- sums sigmoid edge scores to estimate normalized transition count;
- learns first/last heads from the first and last one-hot positions;
- outputs the same sketch-shaped state as the exact Markov sketch:
  `(count_norm, first_regime_probs, last_regime_probs)`.

**Caveat (load-bearing for any writeup):** this is the Markov boundary count
formula encoded into the architecture (count = sum of adjacent-pair inequality
MLP outputs). The MLP only has to learn the trivial "left ≠ right" function on
one-hots. It is a successful methodological control proving that *given the
right summary*, the laws + downstream pipeline are correct. It is **not** a
flexible-encoder learnability result, and it should not be reported as
"flexible encoders solve sufficiency under laws."

Primary code / tests:

- [`src/ctreepo/sim/core/contextual_sbijax.py`](../src/ctreepo/sim/core/contextual_sbijax.py)
- [`scripts/probe_contextual_sbijax.py`](../scripts/probe_contextual_sbijax.py)
- [`scripts/run_regime_one_hot_recovery_grid.py`](../scripts/run_regime_one_hot_recovery_grid.py)
- [`tests/ctreepo/test_contextual_sbijax.py`](../tests/ctreepo/test_contextual_sbijax.py)

Verification run:

```bash
./venv/bin/python -m py_compile \
  src/ctreepo/sim/core/contextual_sbijax.py \
  scripts/probe_contextual_sbijax.py \
  scripts/run_regime_one_hot_recovery_grid.py

./venv/bin/python -m pytest \
  tests/ctreepo/test_contextual_sbijax.py \
  -k 'regime_transition_sum or dense_exact_is_exact' -q
```

Result: `4 passed, 28 deselected`.

### Regime-One-Hot Recovery Grid

Main artifact root:

[`outputs/regime_one_hot_recovery_grid_20260505_162001/`](../outputs/regime_one_hot_recovery_grid_20260505_162001/)

Run shape:

- bundle: `paper_hazard_panel_v1_t128`
- train/val/test docs: `10240 / 1024 / 1024`
- input: `regime_one_hot`
- leaves: `1, 2, 4, 8, 16, 32, 64, 128`
- summary families: `mlp`, `regime_transition_sum`
- hidden dims: `64, 128, 256`
- main iterations: `300`
- long followups: `1000`
- batch size: `256`
- seed: `0` for main/long analytic rows
- learned-merge followups: seeds `0, 1, 2` for leaves `32` and `64`

Launcher status:

- `81 / 81` rows completed
- launcher result: `success`

Best main analytic rows:

| leaf | family | hidden | lr | count raw MAE | theta MAE | eps merge |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `regime_transition_sum` | 128 | 0.01 | `0.0` | `2.76e-05` | n/a |
| 2 | `regime_transition_sum` | 128 | 0.01 | `2.08e-05` | `2.69e-05` | `2.93e-05` |
| 4 | `regime_transition_sum` | 256 | 0.003 | `1.10e-04` | `4.63e-05` | `4.80e-05` |
| 8 | `regime_transition_sum` | 64 | 0.01 | `3.40e-04` | `3.66e-05` | `3.72e-05` |
| 16 | `regime_transition_sum` | 256 | 0.003 | `8.70e-04` | `4.68e-05` | `4.71e-05` |
| 32 | `regime_transition_sum` | 256 | 0.003 | `0.00171` | `4.67e-05` | `4.68e-05` |
| 64 | `regime_transition_sum` | 256 | 0.003 | `0.00385` | `4.60e-05` | `4.61e-05` |
| 128 | `regime_transition_sum` | 64 | 0.01 | `0.00572` | `3.75e-05` | `3.75e-05` |

Long analytic followups:

| leaf | count raw MAE | theta MAE | eps merge |
|---:|---:|---:|---:|
| 32 | `0.000182` | `2.65e-06` | `2.65e-06` |
| 64 | `0.000340` | `2.66e-06` | `2.66e-06` |
| 128 | `0.000556` | `1.81e-06` | `1.81e-06` |

Flat MLP baseline, best row by leaf:

| leaf | count raw MAE |
|---:|---:|
| 16 | `0.142` |
| 32 | `0.382` |
| 64 | `1.000` |
| 128 | `2.247` |

This confirms the earlier failure mode: endpoints were not the issue; large-leaf
count extraction was.

Learned-merge followups with `c2_merge_target=self_consistency`:

| leaf | seeds | mean count raw MAE | mean theta MAE | mean eps merge |
|---:|---:|---:|---:|---:|
| 32 | 0, 1, 2 | `0.00202` | `4.98e-05` | `4.99e-05` |
| 64 | 0, 1, 2 | `0.00380` | `5.04e-05` | `5.04e-05` |

The learned-merge rows are stable and good at 300 iterations, though not as
numerically sharp as the 1000-iteration analytic rows.

## FNO Bridge Status

The bridge-design doc is
[`docs/markov_fno_local_law_bridge.md`](markov_fno_local_law_bridge.md).

Current code exposes two PyTorch objectives in
[`scripts/probe_clean_unified_no.py`](../scripts/probe_clean_unified_no.py):

- `markov_node_witness`: direct decoded `(count, first, last)` supervision on
  every leaf and merge state.
- `markov_local_laws_fno`: C1 leaf calibration, C2 relational merge, and C3
  idempotence/range diagnostics. This is the closer analogue to the theorem
  local-law bridge and should not be conflated with direct witness regression.

Important distinction:

- witness success would show the FNO surface has capacity under direct state
  targets;
- local-law success would be the real transfer of the JAX local-law result.

### Round 1: Multi-Leaf Bridge Campaign (concluded — bridge not solved)

Output root:
[`outputs/markov_fno_bridge_8h_20260505_065112/`](../outputs/markov_fno_bridge_8h_20260505_065112/)

48 of 52 cells completed across `markov_local_laws_fno`, `markov_node_witness`,
and root/contextual control objectives at leaves `2, 4, 16, 32, 64, 128, 256,
2048` × channels `64, 128` × g_n_modes `8, 16` × epochs `3-24`. Bundle
`paper_hazard_panel_v1_t128`.

Key results:

- Best `markov_local_laws_fno` cell at t128: `leaf=32, ch=128, gm=16, ep=24`
  → test_root_mae **`1.94`** with leaf first/last `0.93/0.94`, merge first/last
  `0.78/0.80`, but root rounded-count exact rate only `0.089`. The boundary
  signal is being captured, but the count is not being composed correctly.
- t2048 composition stress: best `markov_local_laws_fno` cell `leaf=256` gets
  leaf first/last to `0.987` while root MAE remains `~3.4`; the
  `recoverable_v5_t2048` `~2.13` floor is unmoved.
- `markov_node_witness` cells across all configurations collapsed to constant
  prediction (`pred_std ≈ 2e-4`, `test_root_mae ≈ 3.15`) at multi-leaf scales.
  Even with direct `(count, first, last)` labels at every node, the model did
  not learn under this configuration.

### Round 2 Stage 1: Single-Leaf Encoder Diagnostic (in progress)

Output root:
[`outputs/markov_fno_round2_stage1_20260505_173903/`](../outputs/markov_fno_round2_stage1_20260505_173903/)

After Round 1 confirmed the bridge is unsolved, this stage isolates the leaf
encoder from any merge composition by running at `n_leaves=1`
(`doc_tokens == leaf_tokens`), so root state is the leaf state. Three parallel
lanes on GPUs 0/2/3:

- **boundary**: per-token BCE on `regime[i] != regime[i+1]` via the existing
  `_run_boundary_supervision_ablation`. Tells the encoder where boundaries
  are.
- **witness**: direct `(count, first, last)` labels via
  `markov_node_witness`.
- **trivial_laws**: `markov_local_laws_fno` with `merge_weight=0`,
  `idempotence_weight=0` (C1 leaf calibration only).

Grid: `doc=leaf ∈ {32, 64, 128} × channels ∈ {128, 256} × g_n_modes ∈ {16, 32}`,
4096 train docs, 48 epochs, seed 0.

Partial results (21/36 cells, doc=32 fully done, doc=64 fully done for boundary
and partly for the others, doc=128 just started on boundary):

| lane | cell | test_root_mae | F1 / theta_mae | full_exact / count_mae |
|---|---|---:|---|---|
| boundary | doc=32 ch=128 gm=16 | 1.20 | F1=0.998, BCE=0.002 | precision=0.999, recall=0.996 |
| boundary | doc=32 ch=128 gm=32 | 1.24 | F1=1.000, BCE=0.000 | precision=recall=1.000 |
| boundary | doc=32 ch=256 gm=16 | 1.30 | F1=0.999, BCE=0.001 | (near-perfect) |
| boundary | doc=32 ch=256 gm=32 | 1.27 | F1=1.000, BCE=0.000 | (perfect) |
| boundary | doc=64 ch=128 gm=16 | 1.58 | F1=0.990, BCE=0.008 | (still near-perfect) |
| boundary | doc=64 ch=128 gm=32 | 1.58 | F1=0.996, BCE=0.002 | (still near-perfect) |
| boundary | doc=64 ch=256 gm=16 | 1.88 | F1=0.992, BCE=0.005 | |
| boundary | doc=64 ch=256 gm=32 | 1.63 | F1=0.999, BCE=0.001 | (essentially perfect) |
| witness/trivial | doc=32 ch=128 gm=16 | 0.385 | theta_mae=0.0037 | full_exact=0.826, count_mae=0.295 |
| witness/trivial | doc=32 ch=128 gm=32 | 0.333 | theta_mae=0.0024 | full_exact=0.928, count_mae=0.192 |
| witness/trivial | doc=32 ch=256 gm=16 | 0.216 | theta_mae=0.0028 | full_exact=0.891, count_mae=0.225 |
| witness/trivial | doc=32 ch=256 gm=32 | **0.130** | theta_mae=0.0014 | full_exact=0.992, count_mae=0.112 |
| witness/trivial | doc=64 ch=128 gm=16 | 0.433 | theta_mae=0.0040 | full_exact=0.736, count_mae=0.386 |
| witness | doc=64 ch=128 gm=32 | 0.450 | theta_mae=0.0042 | full_exact=0.777, count_mae=0.359 |
| witness | doc=64 ch=256 gm=16 | **0.276** | theta_mae=0.0028 | full_exact=0.863, count_mae=0.269 |

Key findings so far:

1. **The encoder is fundamentally fine at single leaf.** Boundary BCE F1 stays
   at `0.99–1.00` from doc=32 through doc=64. The earlier multi-leaf collapse
   was not an encoder limitation.
2. **`witness` and `trivial_laws` produce identical numbers at single leaf.**
   That's expected: at `n_leaves=1` the laws-only objective with
   `merge_weight=0` reduces to leaf calibration, exactly what `witness` does.
   They will diverge once we run multi-leaf.
3. **Wider FNO modes consistently help.** `gm=32` beats `gm=16` across all
   lanes (F1 0.998→1.000; full_exact 0.826→0.928; count_mae 0.295→0.192).
4. **Sum-pool calibration degrades with leaf length even when per-token
   classification stays perfect.** F1 is still ≥0.99 at doc=64 but full_exact
   drops 0.83→0.74 and count_mae rises 0.30→0.39. Sum of small per-token
   sigmoid calibration errors compounds linearly with sequence length —
   not an encoder failure but a pooling/calibration failure.
5. **Wider channels (`ch=128 → 256`) buy real calibration improvement at
   doc=64.** witness full_exact: 0.74–0.78 (ch=128) → 0.86 (ch=256, gm=16);
   witness count_mae: 0.36–0.39 (ch=128) → 0.27 (ch=256, gm=16). F1 was already
   near 1.0, so this isn't classification gain — wider channels produce
   better-calibrated sigmoid scores that sum more accurately, exactly the
   prediction of the calibration-error story.
6. **Capacity helps along the calibration margin, not the classification
   margin.** Once per-token F1 is ≥0.99, additional capacity makes the sigmoid
   probabilities tighter (closer to 0.0 or 1.0), reducing the sum-pool
   accumulation error. This sets up Stage 2: if the bottleneck is *how* we
   pool (not *what features* we pool), a structurally better pool should beat
   even wide-channel sum pool.

### Stage 2 (planned, gated on Stage 1 doc=128)

If the encoder still works at doc=128 single-leaf (which the doc=64 trajectory
suggests it will), the right next test is **pooling alternatives**, not
encoder capacity. The plan is to swap the `sum`/`mean` pool in
`apply_fno_token_encoder` for general alternatives and run a multi-leaf grid:

- `max` pool (preserves extremes).
- `endpoint_concat` (concat first valid + last valid token features, projected
  back to width).
- `mean_max_concat` (capture magnitude + extremes).
- `attention` (learned per-position score, weighted sum).

These are general pool choices, not Markov-specific structural priors. Test
grid: leaf=32 ch=128 gm=16 ep=48 with `markov_local_laws_fno` and seeds 0/1/2,
sweep over pool modes. If a pool decisively beats `sum` (root_mae ≪ 1.94 from
the Round 1 best cell), pooling was the multi-leaf bottleneck.

### Stage 3 (planned, gated on Stage 2)

Investigate the multi-leaf witness collapse from Round 1. The hypothesis is
that the root count head and the witness readout decode count from the leaf
state in incompatible ways. Action items: read `_NodeCountReadout` and
`_MarkovWitnessReadout` source, check whether they share parameters, and
either tie them or add an explicit count-from-witness path.

### Updated Interpretation Of The Multi-Leaf Collapse

After Stage 1, the picture is:

- **Encoder layer**: works (F1≥0.99 on per-token boundaries at single leaf,
  including up through doc=64; doc=128 cell now running for confirmation).
- **Sum-pool calibration**: degrades linearly with sequence length even when
  per-token outputs are well-calibrated. Real but not catastrophic at single
  leaf — count_mae ≈ 0.1 (best doc=32 ch=256 gm=32) → 0.27 (best doc=64
  ch=256 gm=16) → unknown at doc=128. Wider channels mitigate it
  significantly (0.39 → 0.27 at doc=64 going ch=128 → ch=256), suggesting the
  error is partly fixable with capacity but the fundamental issue remains
  that a sum-pool of N approximate boundary indicators accumulates error
  linearly in N.
- **Multi-leaf merge composition**: this is where the catastrophic collapse
  likely happens. The per-leaf state already has small calibration error;
  composing N of them via a learned merge MLP under laws-only supervision
  amplifies the error AND the laws-only objective has degenerate solutions
  (predict-mean) that the gradient finds. Stage 2 (pooling) and Stage 3
  (witness/root readout sharing) target this layer.

**Where capacity helps and where it doesn't:** Stage 1 shows that within the
single-leaf regime, more channels and more modes help calibration (sigmoid
scores tighten toward 0/1, sum-pool count is more accurate). But this is a
diminishing-returns axis: F1 saturates near 1.0, so all additional capacity
goes into calibration-tightening only. The Round 1 multi-leaf failures had
ch=128 gm=16 typical, and even the campaign best at ch=128 gm=16 ep=24 hit
root MAE 1.94 — likely the same calibration-accumulation story compounded
over a merge chain. Stage 2 (structurally different pool) is the cleaner
attack than just throwing more width at the existing sum pool.

## Current Interpretation

1. Local-law supervision is the reliable sufficiency selector in the JAX lane.
2. Exact sketch input proves the theorem-side target, but it is not a learned
   observed-input result.
3. `regime_one_hot + regime_transition_sum` is a useful methodological control
   but the encoder hard-codes the Markov boundary-counting formula. Do not
   conflate this with discovery-from-input.
4. The unstructured MLP summary is the wrong architecture for large-leaf
   regime-one-hot counting under flexible learning.
5. Learned merge is viable once the state representation is learnable; at 300
   iterations it is stable but not as numerically tight as analytic merge at
   1000 iterations.
6. **FNO encoder is fine at single leaf** (Round 2 Stage 1: F1≥0.99 on
   per-token boundary BCE at doc=32–64, full_exact 0.83–0.99 on single-leaf
   sketch). The multi-leaf collapse from Round 1 is not a leaf-encoder
   problem.
7. **Pooling calibration error compounds with sequence length.** Even when
   per-token boundary classification is essentially perfect, sum-pooled count
   estimates accumulate error linearly. count_mae ≈ 0.1 at doc=32 grows to
   ≈0.4 at doc=64 even with F1=0.99–1.00.
8. **Multi-leaf merge composition under laws-only supervision is the next
   suspect.** Composing N already-slightly-miscalibrated leaf states through a
   learned merge MLP with degenerate-solution paths in the laws-only landscape
   is the most likely cause of the Round 1 collapse to root_mae≈3.15.

## Recommended Next Steps

1. Finish Round 2 Stage 1 to confirm the encoder still works at doc=128
   single-leaf. Then write up the single-leaf encoder result as the positive
   "FNO can fit per-token boundaries" baseline.
2. Run Round 2 Stage 2 — pooling alternatives (max, endpoint_concat,
   mean_max_concat, attention) — at multi-leaf=32 with `markov_local_laws_fno`
   on the shared t128 bundle. Anchor cell: leaf=32, ch=128, gm=16, ep=48,
   seeds 0/1/2. Plan drafted in `/tmp/round2_notes/stage2_pooling_plan.md`
   pending implementation in `apply_fno_token_encoder`.
3. Round 2 Stage 3 — investigate whether the multi-leaf witness collapse
   stems from `_NodeCountReadout` and `_MarkovWitnessReadout` decoupling.
   Stage 1 already showed witness ≡ trivial_laws at single leaf; the multi-
   leaf divergence is what we want to explain.
4. Do not burn t2048 composition-stress compute until t128
   `markov_local_laws_fno` clearly beats root/contextual baselines.
5. After Round 2, decide whether the next learned observed-input target is
   raw `one_hot_token_ids` or stays on `regime_one_hot`. Raw tokens add the
   regime-discovery problem on top of boundary counting; that's a different
   bottleneck.

## Note On Reporting

When writing about these results, distinguish carefully between:

- Trivial controls: input is the answer (`markov_exact_sketch`).
- Architectural controls: the right summary statistic is hard-coded into the
  encoder (`regime_transition_sum`).
- Flexible-learning results: the encoder must discover the structure from
  gradient signal under a general loss (e.g., raw FNO + `markov_local_laws_fno`,
  or MLP + `regime_one_hot`).

So far only the first two work cleanly. The flexible-learning result is the
research question, not the existing results.

## Reproduction Commands

Status for the completed recovery grid:

```bash
./venv/bin/python scripts/long_job.py status \
  --job-root outputs/regime_one_hot_recovery_grid_20260505_162001/launcher
```

Recovery grid command shape:

```bash
./venv/bin/python scripts/run_regime_one_hot_recovery_grid.py \
  --output-root outputs/regime_one_hot_recovery_grid_20260505_162001 \
  --gpus 0,2,3 \
  --bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
  --train-docs 10240 \
  --val-docs 1024 \
  --test-docs 1024 \
  --summary-families 'regime_transition_sum mlp' \
  --hidden-dims '128 64 256' \
  --leaves '16 32 64 128 8 4 2 1' \
  --mlp-learning-rates '0.0003' \
  --structured-learning-rates '0.01 0.003' \
  --n-iter 300 \
  --long-n-iter 1000 \
  --batch-size 256 \
  --xla-mem-fraction 0.35
```

Use `scripts/long_job.py launch` for future overnight runs so the job survives
after the launching shell exits.
