# Segmented-LDA End-to-End Simulation for C-TreePO

## Goal

Build a simulation that directly supports the end-to-end decomposition:

- upstream topic estimation error,
- midstream summary-learning/calibration error,
- downstream merge/guidance error.

The design preserves the LDA backbone while adding true segmentation structure so OPS-style
tree-local checks are meaningful.

For unified multi-family sweeps (Segment-LDA weight recovery + segmented C-TreePO + Tensor-LDA books),
see `docs/tree_topic_simulation_suite.md`.

## DGP

For each topic `k`:

- `mu_k ~ Dirichlet(beta)`.

For each book `b`:

- `w_b ~ Dirichlet(alpha)`.
- sample segment count `S_b` and segment lengths.
- for each segment:
- choose a dominant topic from `w_b`.
- sample segment topic mixture from a concentrated Dirichlet around dominant topic.
- sample token topics and words.

This creates:

- true token-level topic assignments,
- true boundaries,
- true root topic mixture per book.

## Topic Estimation Modes

This simulation uses the same upstream topic-word estimator interface as the Segment-LDA OPS
weight-recovery benchmark.

`topic_phi_estimator ∈ {true, noisy_theory, tensor_lda, online_tensor_lda, spectral_numpy}`:

- `true`:
  - use true `φ` (upper bound).
- `noisy_theory`:
  - perturb `φ` at a Lean-mirrored Theorem‑5.1-shaped `O(1/√N)` rate to stress-test end-to-end scaling.
- `tensor_lda`:
  - estimate `φ̂` from unlabeled books via centered moments + whitening + tensor power decomposition + recenter.
- `online_tensor_lda`:
  - estimate `φ̂` via burn-in whitening + mini-batch STGD-style factor updates.
- `spectral_numpy`:
  - estimate `φ̂` from training leaf counts only (center + SVD projection + k-means in spectral space).

In outputs, upstream diagnostics are recorded under `topic_meta.*` (e.g. `topic_phi_l2_error_mean`,
`topic_phi_eps_bound`, whitening error, online loss, etc.).

## Policies

- `oracle_proxy`:
- use oracle topic matrix for leaf projection from word counts.

- `estimated_uncalibrated`:
- use estimated/noisy topic matrix, no calibration.

- `estimated_calibrated`:
- apply affine calibration learned from queried training leaves.

- `estimated_calibrated_budgeted`:
- calibrated leaves + eval-time oracle budget on leaves/internal nodes.

- `oracle_tree`:
- true leaf summaries, zero distortion reference.

## Metrics

- Root error:
- `L1` mean/median/p95, `L2` mean.

- OPS local-law proxies:
- C1 leaf discrepancy violation rate.
- C3 merge discrepancy violation rate.

- Query accounting:
- mean leaf/internal/total queries.

- Optional selection-bias audit:
- naive/IPW/DSL estimators for discrepancy and violation-rate populations.

## End-to-End Decomposition

For each book:

- `total = d(root_budgeted, truth_root)`
- `topic = d(root_uncalibrated, root_oracle_proxy)`
- `calib = d(root_calibrated, root_uncalibrated)`
- `guidance = d(root_budgeted, root_calibrated)`
- `oracle_proxy = d(root_oracle_proxy, truth_root)`

with `d = L1`.

Triangle chain:

- `total <= topic + calib + guidance + oracle_proxy`.

The simulation reports means of all terms plus:

- `upper_bound_mean = topic + calib + guidance + oracle_proxy`
- `slack_mean = upper_bound_mean - total`.

## Runners

Single run:

```bash
cd ThinkingTrees
venv/bin/python scripts/run_segmented_lda_ctreepo_simulation.py --json
```

Grid sweep:

```bash
cd ThinkingTrees
venv/bin/python scripts/grid_segmented_lda_ctreepo_simulation.py
```

Command-list builder for `xargs`/`nohup` style parallel sweeps:

```bash
cd ThinkingTrees
venv/bin/python scripts/build_segmented_lda_ctreepo_cmds.py \
  --topic-phi-estimator spectral_numpy \
  --out-cmds logs/segmented_lda_ctreepo_cmds.txt

JOBS=$(nproc)
nohup bash -lc "xargs -P $JOBS -I{} bash -lc \"{}\" < logs/segmented_lda_ctreepo_cmds.txt" \
  > logs/segmented_lda_ctreepo_sweep.log 2>&1 &
```

Phase/line plotting on per-run JSON outputs:

```bash
cd ThinkingTrees
venv/bin/python scripts/plot_segmented_lda_ctreepo_phase.py \
  --input-glob "outputs/segmented_lda_ctreepo/**/*.json" \
  --topic-phi-estimator spectral_numpy \
  --train-docs 1000 \
  --metric decomposition_total_root_l1_mean \
  --aggregate median \
  --output-figure outputs/segmented_lda_ctreepo/phase_td1000.png \
  --output-json outputs/segmented_lda_ctreepo/phase_td1000_report.json

venv/bin/python scripts/plot_segmented_lda_ctreepo_lines.py \
  --input-glob "outputs/segmented_lda_ctreepo/**/*.json" \
  --topic-phi-estimator spectral_numpy \
  --x-axis oracle_cost_ratio \
  --metric budgeted_root_l1_mean \
  --aggregate median \
  --band p10_p90 \
  --log-x \
  --output-figure outputs/segmented_lda_ctreepo/lines_costratio_rootl1.png \
  --output-json outputs/segmented_lda_ctreepo/lines_costratio_rootl1_report.json
```

## Suggested Initial Grid

- `n_books_train`: `64,128,256,512`
- `fixed_leaf_tokens`: `16,32,64`
- `calibration_leaf_query_rate`: `0.05,0.10,0.25,0.50`
- `eval_internal_query_rate`: `0.00,0.10,0.25,0.50,1.00`
- seeds: at least `3`

and hold `topic_phi_estimator="noisy_theory"` first (then swap in `tensor_lda` / `online_tensor_lda`).
