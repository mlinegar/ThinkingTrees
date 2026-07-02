# Local-Law Sampling and IPW Contract

This repository has one canonical implementation for sampled local-law
objectives:

- `treepo/src/treepo/training/local_law.py`
- `local_law_objective_from_losses(...)`
- `local_law_objective_target_mse(...)`
- `sampled_uniform_node_ipw_mean_loss(...)`
- `observed_uniform_node_ipw_mean_loss(...)`

Experiment-specific code, including Markov, HLL, classical datasketches, LDA,
and future learned-sketch runners, should not implement its own IPW arithmetic.
It should build node-level rows and call the master objective.

## Required Row Contract

Every sampled local-law objective must be represented as rows with:

- `oracle_loss`: the loss against the queried oracle/local-law label.
- `proxy_loss`: the proxy loss for corrected/doubly robust modes, or zeros for
  sampled-IPW-only modes.
- `observed`: whether the oracle/local-law row was queried.
- `propensity`: the row's inclusion probability under the actual sampling
  design, in `(0, 1]` for observed rows.
- `node_weights`: optional target-population weights. Use ones for the
  unweighted node mean.
- `depths`: optional depth rows for depth discounting. Use zeros when no depth
  discount is part of the estimand.

The estimator must derive `propensity` from the realized candidate population
and sampling design. Do not reuse display rates such as `R10` directly as
weights unless they equal the actual inclusion probability for every row. For
fixed-size uniform sampling over `N` candidate nodes with `q` draws, each row's
propensity is `q / N`.

## Uniform All-Node Supervision

For the "uniform over all nodes" estimand, the target population is each
logical tree node exactly once. The root is not guaranteed; it has the same
inclusion probability as every other node.

If a model surface exposes both:

- an explicit root row, and
- cumulative merge rows whose final row is also the root,

then drop the final cumulative merge row before constructing the all-node
population. Otherwise the root is double-counted and has the wrong sampling
probability.

For a full binary tree with `L` leaves, the all-node scalar population should
have `2L - 1` rows: `L` leaves, `L - 2` non-root internal rows, and one explicit
root row.

## Estimand Changes Must Be Explicit

The historical root-guaranteed objective and the uniform all-node objective are
different estimands:

- root-guaranteed: root is always observed and local rows are separate
  components.
- uniform all-node: root is one row in the same sampled node population as every
  other node.

For random-supervision `R` grids, `R` is a per-node inclusion probability, not
a per-document quota. A document is allowed to contribute zero supervised nodes.
Do not round up to "at least one node per document"; that changes the estimand
for small trees. Repeated training epochs must reuse a persistent sampled mask
for each document/run so that `R10` means roughly 10% of nodes are ever labeled,
not 10% newly redrawn on every epoch.

When launching runs, record the estimand in config and output metadata, e.g.
`learned_supervision_sampling_policy=uniform_all_nodes`. Do not compare these
rows as if they used the same objective.

## Tests

Changes to sampled supervision should include tests against the master local-law
module, especially `treepo/tests/training/test_local_law.py`. Tests should cover
node weights, zero-sample behavior, and any adapter that converts tree traces or
cumulative merge traces into node rows.
