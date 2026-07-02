# Optimize To Zero Handoff

> **Status (2026-05-05): Resolved.** See
> [`contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`](contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md)
> for the resolution. The short version: the NASSS contrastive objective
> has a continuous family of equivalent optima and SGD does not pick the
> sufficient summary. The local laws (`fit_contextual_sbijax_learned_local_laws`)
> are the principled fix and drive the leaf grid to literal numerical zero
> on `markov_exact_sketch` input. The body of this note remains as-is for
> historical context and as the framing for the next-step ladder; treat
> the "Latest t128 Leaf Grid" table as the *pre-resolution* baseline.
>
> **Post-resolution ablations (2026-05-05):** see
> [`markov_sim_status.md`](markov_sim_status.md) for the current overall
> status snapshot,
> [`markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`](markov_contextual_sufficiency_ablation_handoff_2026-05-05.md)
> and
> [`../outputs/markov_contextual_ablation_grid_report_20260505.md`](../outputs/markov_contextual_ablation_grid_report_20260505.md).
> The ablations confirm the current interpretation: NASSS helps only as an
> auxiliary, learned merge/decoder variants work once local laws supervise the
> sufficient sketch, and standalone `CleanUnifiedNO` general f/g remains far
> from exact-zero. The follow-up regime-one-hot recovery grid adds
> `local_law_summary_family="regime_transition_sum"` and fixes the large-leaf
> count-extraction failure without feeding the exact Markov sketch as input.

This is the continuation note for the Markov contextual-sufficiency thread.
It is meant for another LLM or engineer to resume the work without rebuilding
the session history.

## Current Thesis

Contextual sufficiency is the `f` after `g*` story:

- `g*(x) = g(leafInput(embed(x)))` is the learned item carrier state.
- `f(g*(x))`, or `f` applied after a composed context, should recover the
  query response.
- For the Markov changepoint task, the exact sufficient statistic is the
  sketch `(count, first, last)`.
- The exact sketch is the zero-error witness. If a learned state preserves it,
  the contextual responses are recoverable.

The present bottleneck is not the DGP, target construction, or exact statistic:
the exact-sketch oracle and root witness are zero or numerical zero. The
remaining nonzero error is in learned/package summary recovery.

## Current Status

Implemented and working:

- Generic contextual dataset API around `query(ctx, x)`, not hard-coded
  `left/span/right` semantics.
- Markov two-sided adapter over `Ctx = (left_fragment, right_fragment)`.
- Paper hazard-panel bundles with balanced condition metadata.
- Package-facing CLI: `ctreepo sim run contextual-sbijax`.
- Exact Markov sketch diagnostics:
  - `markov_exact_sketch_oracle`
  - `exact_root_witness`
- Full t128 contextual leaf ladder: `1, 2, 4, 8, 16, 32, 64`.
- Repo-owned learned local-law Markov lane:
  - `--sbijax-trainer learned_local_laws`
  - exact decoder over the theorem sketch `(count, first, last)`
  - dense/sparse/dual local-law observation metadata
  - `eps_leaf`, `eps_merge`, `eps_idemp` diagnostics
  - `--local-law-summary-family affine_probe` for the exact affine
    leaf=1 diagnostic lane.

Primary artifact anchors:

- `outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json`
- `outputs/contextual_sbijax_leaf_grid_t128_actual_20260504_231651/leaf_grid_summary.json`
- `outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/leaf1_diagnostic_summary.md`
- `outputs/optimize_to_zero_laws_grid_t128/leaf_grid_summary.json`
- `outputs/optimize_to_zero_theta_sup_grid_t128/leaf_grid_summary.json`
- `outputs/optimize_to_zero_fg_architecture_ablation_t128/summary.json`
- `outputs/optimize_to_zero_laws_hybrid_grid_t128/summary.json`
- `outputs/clean_unified_fg_contextual_ablation_t128/summary.json`
- `outputs/markov_contextual_ablation_grid_report_20260505.md`
- `docs/contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`
- `docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`

## Latest Leaf=1 Exact-Zero Diagnostic

Run shape:

- bundle: `paper_hazard_panel_v1_t128`, seed 0
- train/val/test docs: `1024/256/256`
- `leaf_tokens=1`, `fragment_len=1`
- package theta: `markov_exact_sketch`
- response contexts/slices: `16/8`
- main input under test: `regime_one_hot`

Summary artifact:

- [leaf1_diagnostic_summary.md](../outputs/contextual_sbijax_leaf1_diagnostic_20260505_012737/leaf1_diagnostic_summary.md)

Key rows:

| candidate | input | decoder | contextual raw MAE | theta MAE | raw count MAE | first acc | last acc | eps leaf |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `exact_zero_markov` | `regime_one_hot` | exact | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| `identity_theta` | `markov_exact_sketch` | exact | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| `learned_local_laws_affine` | `regime_one_hot` | exact | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| `learned_local_laws_exact_input` | `markov_exact_sketch` | exact | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| `learned_local_laws_mlp` | `regime_one_hot` | exact | 1.49e-4 | 7.10e-5 | 1.01e-4 | 1.0 | 1.0 | 7.10e-5 |
| `theta_supervised` | `regime_one_hot` | learned | 2.09e-4 | 8.09e-5 | 1.00e-4 | 1.0 | 1.0 | 8.09e-5 |
| `package_nass` | `regime_one_hot` | learned | 1.69e-4 | 0.625 | 0.939 | 0.035 | 0.035 | 0.625 |
| `package_nasss` | `regime_one_hot` | learned | 4.96e-2 | 0.207 | 0.251 | 0.0 | 0.176 | 0.207 |

Interpretation:

- Leaf=1 is exactly learnable from `regime_one_hot`; exact zero does not depend
  on feeding `markov_exact_sketch` as input.
- The exact-zero learned row is a learned affine map plus canonical projection
  to the theorem-domain sketch. That projection is part of the discrete
  Markov count/endpoint contract, not a learned readout.
- Plain continuous MLP optimization gets very close but leaves small slot
  drift. This is an optimizer/numerical surface issue, not a representability
  issue.
- Package NASS/NASSS should remain approximate baselines. In particular, NASS
  can get low contextual error while failing the state checks badly; exact-zero
  claims must report `theta_mae`, raw count MAE, first/last accuracy, and law
  eps values.

## Latest t128 Leaf Grid

Run shape:

- bundle: `paper_hazard_panel_v1_t128`, seed 0
- train/val/test docs: `1024/256/256`
- trainer: `package`
- method: `nasss`
- package theta: `markov_exact_sketch`
- response contexts/slices: `16/8`
- learned summary dimensions: `state_dim=25`, `hidden_dim=128`
- iterations/batch: `50/128`

| input encoding | leaf | test contextual MAE | test corr | exact oracle MAE | root witness MAE |
|---|---:|---:|---:|---:|---:|
| `markov_exact_sketch` | 1 | 0.047866 | 0.951 | 0 | 0 |
| `markov_exact_sketch` | 2 | 0.015275 | 0.971 | 0 | 0 |
| `markov_exact_sketch` | 4 | 0.012961 | 0.940 | 0 | 0 |
| `markov_exact_sketch` | 8 | 0.008774 | 0.944 | 9.09e-10 | 0 |
| `markov_exact_sketch` | 16 | 0.011024 | 0.872 | 1.97e-09 | 0 |
| `markov_exact_sketch` | 32 | 0.011977 | 0.808 | 1.87e-09 | 0 |
| `markov_exact_sketch` | 64 | 0.010763 | 0.800 | 3.32e-09 | 0 |
| `regime_one_hot` | 1 | 0.058444 | 0.928 | 0 | 0 |
| `regime_one_hot` | 2 | 0.015091 | 0.973 | 0 | 0 |
| `regime_one_hot` | 4 | 0.010168 | 0.954 | 0 | 0 |
| `regime_one_hot` | 8 | 0.009462 | 0.938 | 9.09e-10 | 0 |
| `regime_one_hot` | 16 | 0.008617 | 0.903 | 1.97e-09 | 0 |
| `regime_one_hot` | 32 | 0.010060 | 0.850 | 1.87e-09 | 0 |
| `regime_one_hot` | 64 | 0.009943 | 0.816 | 3.32e-09 | 0 |

Interpretation:

- The exact response path remains clean: oracle/root are zero or numerical zero.
- `sbijax.NASSS` is close but not zero, even when the input is already the exact
  sketch.
- Leaf 1 is the strongest small-item stress check and should not be skipped.

## Reproduction Template

Use the package CLI through the saved hazard-panel bundle. Run this loop for
each input encoding under test.

```bash
source venv/bin/activate

OUT_ROOT=outputs/contextual_sbijax_leaf_grid_t128_next
BUNDLE=outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json

for encoding in markov_exact_sketch regime_one_hot; do
  for leaf in 1 2 4 8 16 32 64; do
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    ctreepo sim run contextual-sbijax \
      --data-source markov \
      --load-data-bundle "$BUNDLE" \
      --sbijax-trainer package \
      --sbijax-method nasss \
      --sbijax-package-theta markov_exact_sketch \
      --sbijax-input-encoding "$encoding" \
      --train-docs 1024 \
      --val-docs 256 \
      --test-docs 256 \
      --fragment-len "$leaf" \
      --context-samples-per-doc 1 \
      --response-signature-contexts 16 \
      --response-signature-slices 8 \
      --embedding-dim 32 \
      --state-dim 25 \
      --hidden-dim 128 \
      --learning-rate 0.0003 \
      --n-iter 50 \
      --batch-size 128 \
      --seed 0 \
      --output-root "$OUT_ROOT/${encoding}/leaf_${leaf}"
  done
done
```

For long runs, launch the same loop through `scripts/long_job.py` and keep the
environment constrained to one GPU:

```bash
python scripts/long_job.py launch \
  --name contextual_sbijax_leaf_grid_t128_next \
  --job-root outputs/contextual_sbijax_leaf_grid_t128_next_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  --replace-existing \
  -- bash -lc 'source venv/bin/activate && ./path/to/grid_loop.sh'
```

Summarize each run into `leaf_grid_summary.json` with one row per
`encoding x leaf`, including at least:

- `test_contextual_mae`
- `test_corr`
- `test_pred_std`
- `test_truth_std`
- `exact_sketch_oracle_mae`
- `root_witness_mae`
- split condition counts

## Current Next Experiment Ladder

Use the leaf=1 diagnostic as a gate before spending time on larger leaves or
HLL/FNO surfaces:

1. Confirm exact controls:
   - `exact_zero_markov`
   - `identity_theta`
2. Confirm learned local-law exact lane:
   - `learned_local_laws` with `--local-law-summary-family affine_probe`
   - `regime_one_hot` input
   - exact decoder
3. Compare continuous optimization against the exact lane:
   - `learned_local_laws` default MLP
   - `theta_supervised`
   - report whether the only remaining error is small count/endpoint slot
     drift.
4. Keep package methods as approximate baselines:
   - `--sbijax-method nasss`
   - `--sbijax-method nass`
5. Compare package activations:
   - `relu`
   - `tanh`
   - `gelu`
   - `swish`
6. Sweep capacity and training:
   - `state_dim`: `25`, `64`, `128`
   - `hidden_dim`: `128`, `256`, `512`
   - `n_iter`: `50`, `100`, `200`
   - `batch_size`: keep `128` first; adjust only if GPU utilization is poor.
7. Sweep contextual supervision:
   - response contexts: `16`, `32`, `64`
   - response slices: `8`, `16`, `32`
8. Compare trainer controls:
   - `package`
   - `theta_supervised`
   - `identity_theta`
   - `learned_local_laws`
9. Compare input encodings in this order:
   - `markov_exact_sketch`
   - `regime_one_hot`
   - `regime_ids`
   - `one_hot_token_ids`
   - `normalized_token_ids`

Only treat a setting as exact-zero if it passes the sketch/law diagnostics. A
low contextual readout error with high `theta_mae` is a failed sufficient-state
recovery run, not an exact-zero run.

## Acceptance Notes

For a credible "optimized to zero" claim:

- The exact oracle/root witness must remain zero or numerical zero.
- Learned/package summary MAE should approach numerical zero on
  `markov_exact_sketch` input first.
- The same setting should then be checked on `regime_one_hot`.
- Leaf `1, 2, 4` are required, not optional. They prevent long-leaf averaging
  from hiding failures.
- Condition counts should remain balanced across the four t128 panel cells.

**Met (2026-05-05):** all five conditions are met by
`--sbijax-trainer learned_local_laws --sbijax-input-encoding markov_exact_sketch`
on the full leaf grid (1, 2, 4, 8, 16, 32, 64). Outputs in
`outputs/optimize_to_zero_laws_grid_t128/`. Resolution write-up in
[`contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`](contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md).
Two remaining open threads (encoder bottleneck on harder input encodings,
and the bridge to the f/g `recoverable_v5_t2048` floor) are documented
there.
