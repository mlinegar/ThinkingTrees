# Tensor-LDA DGP Benchmark for ThinkingTrees / C-TreePO

## Purpose

This benchmark is designed to compare:

- Tensor-LDA-style estimation under the traditional LDA DGP.
- C-TreePO-style tree aggregation and query-budgeted correction on synthetic "books."
- Oracle upper bounds.

This is no longer the cleanest "can the tree recover ordinary LDA?" base case.
That role belongs to `docs/lda_tree_recovery_simulation_spec.md`, which fixes the
exact count-merge control first. This benchmark remains useful as a richer
book/chapter hierarchy after that exact base family is in place.

For unified command generation across this benchmark plus segmented-LDA families,
see `docs/tree_topic_simulation_suite.md`.

It is aligned to the Tensor-LDA paper simulation setup (traditional LDA DGP) and extends it with
tree-local and audit-focused diagnostics that matter for ThinkingTrees.

## DGP (matches traditional LDA simulation assumptions)

For `K` topics and vocabulary size `V`:

- Topic-word distributions: `mu_k ~ Dirichlet(beta)`.
- Book-level topic weights: `w_b ~ Dirichlet(alpha)`.
- Chapter-level topic mixtures: `theta_{b,c} ~ Dirichlet(concentration * w_b)`.
- Token generation for each chapter:
- `z ~ Multinomial(theta_{b,c})`.
- `word ~ Multinomial(mu_z)`.

This gives synthetic books with known latent weights and observed chapter word-count vectors.

## Compared Policies

- `tlda_projection`:
- Tensor-LDA-style baseline at book level (project aggregate book counts onto known topic matrix).

- `ctree_proxy`:
- C-TreePO-style balanced tree over chapter summaries using a deliberately under-supported proxy
  estimator (anchor-word scores).

- `ctree_calibrated`:
- Same tree, but proxy leaf summaries are calibrated with queried training leaves
  (affine map from proxy to oracle chapter mixtures).

- `ctree_calibrated_budgeted`:
- Calibrated tree with evaluation-time query budget:
- leaf-level oracle replacement rate
- internal-node oracle replacement rate
- internal query design: `none|uniform|risk`.

- `ctree_oracle`:
- Oracle upper bound (all leaf/internal summaries exact).

## Reported Metrics

- Root quality:
- mean/median/p95 root `L1` error
- mean root `L2` error
- mean root `L1` error vs latent book weight

- Local-law proxies:
- `C1` violation rate: leaf discrepancy above threshold
- `C3` violation rate: internal merge discrepancy above threshold

- Query accounting:
- mean leaf queries
- mean internal queries
- mean total queries

- Optional selection-bias audit on internal-node discrepancy population:
- naive estimator bias/variance
- IPW bias/variance
- DSL0 and oracle-DSL bias/variance
- IPW violation-rate CI coverage/radius

## Recommended Experiment Matrix

Run sweeps over these axes:

- Sample size axis:
- `n_books_train in {64, 128, 256, 512}`
- `n_books_test in {128, 256}`

- Information axis:
- `tokens_per_chapter in {32, 64, 128, 256}`
- `chapters_per_book in {8, 16, 32}`

- Calibration budget axis:
- `calibration_leaf_query_rate in {0.01, 0.05, 0.10, 0.25, 0.50}`
- `calibration_policy in {uniform, entropy}`

- Evaluation guidance axis:
- `eval_internal_query_rate in {0.00, 0.05, 0.10, 0.25, 0.50, 1.00}`
- `eval_internal_query_design in {uniform, risk}`

- Robustness axis:
- vary `alpha_topic`, `chapter_concentration`, `proxy_noise_std`

## Expected Signatures

- `ctree_calibrated` should improve root error versus `ctree_proxy`.
- Increasing calibration labels should reduce calibrated error.
- Increasing internal query budget should reduce `C3` violations and root error.
- `ctree_oracle` should be a lower bound on achievable root error.
- Under risk-skewed sampling, naive audit estimators should be biased; IPW/DSL should correct bias.

## Lean Linkage

This benchmark maps to the Lean comparison layer:

- `ML.TopicModels.ThinkingTrees.CTreePOGuarantees`
- `ML.TopicModels.ThinkingTrees.CTreePOExtendsTensorLDA`
- `ML.TopicModels.ThinkingTrees.ctreepo_inherits_tensor_guarantees_if_same_model`

and is intended as empirical evidence for the additional C-TreePO augmentation fields:

- query-budget error monotonicity/consistency
- audit-bias control (IPW/DSL style)

## Runner

Use:

```bash
cd ThinkingTrees
venv/bin/python scripts/run_tensor_lda_book_weight_benchmark.py --json
```

Output artifacts:

- JSON summary with full config and policy metrics.
- CSV row suitable for grid aggregation.
