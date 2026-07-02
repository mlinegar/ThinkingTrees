# HLL JAX Local-Law Handoff (2026-05-08)

## Summary

HLL is now wired into the current repo-owned JAX `learned_local_laws`
lane with `package_theta=hll_register_sketch`. Round 1 and Round 2 showed
that the local-law learner can recover HLL registers, and that larger
training sets help register MAE in the same direction as Markov. The open
failure mode was the scalar HLL cardinality readout: register MAE improved,
but raw HLL estimate MAE stayed too high.

The older estimate-aware path existed in the PyTorch/FNO HLL diagnostics as
`readout_arch=hll_formula` / `hll_residual` in
`scripts/run_fno_mergeable_sketch_diagnostic.py`, and as the default
`hll_formula` readout in `scripts/run_hll_sampled_node_rate_grid.py`. The
current JAX lane now ports that idea as an optional auxiliary loss:

```bash
--local-law-hll-estimate-weight <weight>
```

This HLL-only auxiliary penalizes the normalized cardinality estimate implied
by predicted leaf and merge registers. It defaults to `0.0`, so existing
Markov and HLL runs are not forced onto the new objective.

## Code Changes

- `src/ctreepo/sim/core/contextual_sbijax.py`
  - Added `ContextualSBIJAXConfig.local_law_hll_estimate_weight`.
  - Added a differentiable HLL estimate proxy matching the old formula
    behavior: exact on integer registers, smooth on off-lattice states.
  - Added `train_hll_estimate_mse` / `val_hll_estimate_mse` to training
    history.
  - Added provenance fields:
    `local_law_hll_estimate_weight` and
    `local_law_hll_estimate_objective`.
- `scripts/probe_contextual_sbijax.py`
  - Added `--local-law-hll-estimate-weight`.
- `tests/ctreepo/test_contextual_sbijax.py`
  - Tiny HLL local-law smoke now exercises the estimate-aware auxiliary.
- `scripts/run_hll_jax_local_law_round3_estimate_aware.py`
  - New 10-cell Round 3 screen.

## Paper-Notation Realignment

After Round 3, the `learned_local_laws` lane was realigned so future runs can
make the paper notation literal:

```text
g_phi(x)           -> z
g_phi(z_left,z_right) -> z_parent
f_psi(z)           -> theta_hat
R(theta_hat, c)    -> response
```

Enable it with:

```bash
--local-law-explicit-state-decoder \
--local-law-summary-dim <d_z> \
--local-law-state-decoder-head {mlp,linear}
```

This route requires `--law-architecture learned_merge` or
`--law-architecture fully_learned`, because analytic merge is defined on
decoded theorem states, not arbitrary learned summaries. Historical fused
baselines remain available with the default
`--no-local-law-explicit-state-decoder` behavior:

```text
summary_net(x) == theta_hat
```

Diagnostics now report:

- `paper_notation_factorization`
- `g_summary_dim`
- `f_state_decoder_kind`
- `local_law_explicit_state_decoder`

Smoke artifact:

```bash
outputs/hll_explicit_fg_smoke_20260508/summary.json
outputs/markov_explicit_fg_smoke_20260508/summary.json
```

The smoke run used `g_summary_dim=12`, `f_state_decoder_kind=explicit_mlp`,
`merge_network=learned_asymmetric_mlp`, and exact HLL response readout. The
Markov smoke used `g_summary_dim=8` with the same explicit factorization.

## Round 3 Results

Output root:

```bash
outputs/hll_jax_local_law_round3_estimate_aware_20260508_034900
```

All 10 cells exited with code 0. Aggregated CSV:

```bash
outputs/hll_jax_local_law_round3_estimate_aware_20260508_034900/grid_summary.csv
```

Key test metrics:

| train_docs | architecture | est_weight | register_mae | raw_estimate_mae | contextual_mae |
|---:|---|---:|---:|---:|---:|
| 10240 | analytic | 0.0 | 0.01891 | 15.34 | 0.04121 |
| 10240 | analytic | 0.1 | 0.01545 | 5.19 | 0.04003 |
| 10240 | analytic | 1.0 | 0.01887 | 4.18 | 0.03412 |
| 10240 | learned_merge | 1.0 | 0.02007 | 4.46 | 0.03698 |
| 10240 | fully_learned | 1.0 | 0.05329 | 3.80 | 0.03142 |
| 102400 | analytic | 0.0 | 0.01316 | 9.08 | 0.02685 |
| 102400 | analytic | 0.1 | 0.01237 | 3.03 | 0.02552 |
| 102400 | analytic | 1.0 | 0.01603 | 2.86 | 0.02553 |
| 102400 | learned_merge | 1.0 | 0.01533 | 2.95 | 0.02579 |
| 102400 | fully_learned | 1.0 | 0.03747 | 3.00 | 0.02340 |

## Interpretation

Estimate-aware loss works for the failure mode we saw: raw HLL estimate MAE
drops by roughly 3x at 10240 docs and 3x at 102400 docs versus the matched
analytic baseline.

Data scaling still helps. The analytic baseline improves from raw estimate
MAE `15.34` to `9.08` when moving 10240 -> 102400 docs; with the
estimate-aware auxiliary, the best analytic result improves from `4.18` to
`2.86`.

Register recovery and cardinality readout are partly separable. The
fully-learned cells have worse register MAE but better contextual MAE,
especially at 102400 docs. That says the model can learn a state that is
useful for the scalar HLL response without staying close to the exact
register lattice. For C-TreePO/theorem-facing claims, keep reporting
register MAE and law eps separately from raw/cardinality metrics.

One comparability note: the JAX HLL decoder and off-lattice diagnostics now
use the differentiable HLL formula proxy for predicted registers. Exact
integer-register targets are unchanged, but Round 3 raw estimate MAE should
be compared against the Round 3 baseline row, not naively against pre-port
off-lattice raw estimate diagnostics.

## Reproduction

```bash
./venv/bin/python scripts/run_hll_jax_local_law_round3_estimate_aware.py \
  --output-root outputs/hll_jax_local_law_round3_estimate_aware_$(date -u +%Y%m%d_%H%M%S) \
  --plan-only

./venv/bin/python scripts/long_job.py launch \
  --name hll_round3_estaware_s0 \
  --job-root outputs/hll_jax_local_law_round3_estimate_aware_YYYYmmdd_HHMMSS/launcher_shard0 \
  --cwd /home/mlinegar/ThinkingTrees \
  -- env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=1 \
     ./venv/bin/python scripts/run_hll_jax_local_law_round3_estimate_aware.py \
       --output-root outputs/hll_jax_local_law_round3_estimate_aware_YYYYmmdd_HHMMSS \
       --shard-index 0 --num-shards 4
```

Repeat the launch command for shard indices 1-3 with matching
`CUDA_VISIBLE_DEVICES`.
