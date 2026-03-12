# Tree-Relevant LDA Code Review

Date: 2026-03-06

Snapshot reviewed:

- Production root: `outputs/tree_relevant_lda_production_queue_20260306`
- Status at review time: Stage 1 `1536 / 1536` complete, Stage 2 `3965 / 4608` complete
- Status timestamp: `2026-03-06T19:54:16Z`

## Scope

This review covers the current tree-relevant LDA ladder:

- `src/ctreepo/sim/core/lda_tree_utility_vector.py`
- `src/ctreepo/sim/core/leaf_local_mixture_utility.py`
- `scripts/run_lda_tree_utility_vector_simulation.py`
- `scripts/run_leaf_local_mixture_utility_simulation.py`
- `scripts/build_lda_tree_utility_vector_cmds.py`
- `scripts/build_leaf_local_mixture_utility_cmds.py`
- `scripts/report_lda_tree_methods_paper.py`
- `scripts/launch_tree_relevant_lda_sweeps.sh`
- `docs/tree_relevant_lda_simulation_ladder.md`
- `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`
- `lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean`

I also inspected the completed JSON outputs under the production queue and ran small local aggregations over those outputs to check whether the reported story matches the actual slices.

## Executive Summary

The short version is:

1. Stage 1 is correct and well aligned with the intended paper story.
2. Stage 2 is also coherent as a generative story, and it does create a real pooled-vs-leaf gap when `lambda_multiplier > 0` and `local_mixture_concentration` is small.
3. The current Stage 2 report is not isolating that story cleanly. It averages across `lambda_multiplier` in places where `lambda_multiplier` is the key control, and it uses a signed heterogeneity statistic that mostly cancels.
4. The current practical baseline `leaf_ridge_from_u` is not broadly better than the pooled ceiling, but it does beat the pooled model in the strongest completed heterogeneity slice. So the right conclusion is not "Stage 2 does not work"; it is "the Stage 2 world is meaningful, but the current practical estimator is only partially recovering the available gain."

## What The Code Implements

### Stage 1

Stage 1 is the clean positive-control case.

The code in `src/ctreepo/sim/core/lda_tree_utility_vector.py` defines:

- ordinary bag-of-words LDA documents,
- a document-level scalar target `Y_d = r^T U c_d`,
- a mergeable intermediate sketch `u(A) = U c_A`,
- exact tree recovery by leafwise utility-vector addition,
- two compressed recovery families:
  - `count_svd_ceiling`
  - `utility_pca_practical`

This is internally coherent. The target is a function of the full document histogram, and the tree is only a way of computing or compressing a mergeable sketch of that target.

### Stage 2

Stage 2 in `src/ctreepo/sim/core/leaf_local_mixture_utility.py` moves to local latent mixtures:

- `pi_d ~ Dir(alpha)`
- `pi_{d,b} ~ Dir(tau * pi_d)`
- `Y_d = sum_b n_b h(pi_{d,b})`
- `h(pi) = theta^T pi + lambda * pi^T W pi`

This is the first place where leaves matter statistically. The code matches the intended theorem:

- when `lambda_multiplier = 0`, the purely linear part collapses to a pooled document quantity;
- when `lambda_multiplier > 0`, the quadratic part creates a pooled-vs-leaf gap.

The practical tree methods are:

- `leaf_ridge_from_u`: fine base-leaf supervision from the Stage 1 sketch `u(A) = U c_A`
- `coarse_leaf_ridge_from_u`: the same idea at coarser evaluation leaves

The non-tree comparator is:

- `pooled_doc_wrong_model`: infer one document-level topic mixture from full counts, then score the whole document with the single-mixture model

This baseline is intentionally strong: it uses the true topic-word table and the exact utility form.

## Stage 1 Assessment

Stage 1 does what it should do.

From the completed queue:

| Method | Result |
|---|---|
| `tree_exact_utility` | Mean scalar error stays at about `6.9e-16` for every `state_dim` |
| `utility_pca_practical` | Becomes exact at `state_dim = 16`, matching `utility_dim` |
| `count_svd_ceiling` | Becomes exact at `state_dim = 512`, matching `vocab_size` |

Observed mean scalar errors by `state_dim`:

- `utility_pca_practical`: `1.4057` at `4`, `0.0288` at `8`, then machine precision from `16` onward
- `count_svd_ceiling`: `2.1807` at `4`, `0.9031` at `8`, `0.4823` at `128`, `0.1965` at `256`, exact at `512`

Interpretation:

- This is a good positive control for the mergeability story.
- It is not a substantive statistical challenge, and it should not be sold as one.
- It is doing exactly what Stage 1 should do: establish the algebraic baseline cleanly.

## Stage 2 Assessment

Stage 2 makes sense, but the current paper report undersells and partly obscures what it is showing.

### What is working

The Stage 2 world is coherent:

- `leaf_oracle_sum` achieves zero error by construction, so the oracle leafwise target is internally consistent.
- The `lambda = 0` setting behaves like the intended control: the true pooled-vs-leaf gap vanishes.
- The completed queue already contains slices where the leafwise methods beat the pooled model on mean held-out error.

Across completed Stage 2 slices, the clearest mean-win regime is the strongest completed heterogeneity case:

- `lambda = 2`, `tau = 0.25`

All-leaves-labeled means in that slice:

| Leaf size | Pooled | Fine leaf ridge | Coarse leaf ridge |
|---|---:|---:|---:|
| `100%` | `24.5361` | `17.3345` | `15.6611` |
| `50%` | `24.5361` | `17.3345` | `15.9004` |
| `25%` | `24.5361` | `17.3345` | `16.7017` |

Counting completed parameter slices after seed aggregation:

- Completed Stage 2 slices with all three practical methods present: `249`
- Slices where `leaf_ridge_from_u` beats pooled on mean error: `18`
- Slices where `coarse_leaf_ridge_from_u` beats pooled on mean error: `18`

Those `18` winning slices are all the completed `lambda = 2`, `tau = 0.25` slices across leaf-size and budget settings.

### What is not working yet

Outside that strongest heterogeneity regime, the current `leaf_ridge_from_u` practical method is usually worse than the pooled ceiling.

For the all-leaves-labeled rows currently completed:

| lambda | tau | Pooled | Fine leaf ridge | Coarse leaf ridge |
|---:|---:|---:|---:|---:|
| `0` | `0.25` | `3.8311` | `14.3016` | `12.7464` |
| `0` | `1` | `4.4241` | `17.8427` | `13.3691` |
| `0` | `8` | `5.1193` | `25.1843` | `16.7330` |
| `0` | `64` | `5.2830` | `27.6116` | `17.4096` |
| `1` | `0.25` | `12.9240` | `15.3239` | `13.8811` |
| `1` | `1` | `9.1076` | `18.7986` | `14.4359` |
| `1` | `8` | `5.6977` | `26.1220` | `17.8838` |
| `1` | `64` | `5.4890` | `28.3761` | `18.4464` |
| `2` | `0.25` | `24.5361` | `17.3345` | `16.0877` |
| `2` | `1` | `16.0390` | `20.9331` | `16.6882` |
| `2` | `8` | `6.8957` | `28.3698` | `20.4107` |
| `2` | `64` | `6.0355` | `30.5339` | `20.9658` |

Interpretation:

- The current practical estimator is only recovering the leaf advantage in the highest-gap regime.
- Coarse leaf regression is often better than fine leaf regression, which suggests that very fine leaves are adding too much observation noise for the current feature map and ridge fit.
- More Stage 2 queue volume will sharpen estimates, but it is unlikely to change the overall story by itself.

## Key Review Findings

### 1. The current Stage 2 report is averaging across the main causal knob

The Stage 2 markdown table and the tau plot in `scripts/report_lda_tree_methods_paper.py` group by `local_mixture_concentration` but do not stratify by `lambda_multiplier`.

Relevant code:

- `scripts/report_lda_tree_methods_paper.py:218-230`
- `scripts/report_lda_tree_methods_paper.py:353-390`

The resolution plots do the same:

- `scripts/report_lda_tree_methods_paper.py:392-430`

This matters because `lambda_multiplier` is exactly what turns the pooled-vs-leaf gap on and off. Mixing `lambda = 0`, `1`, and `2` into one figure makes the report answer a different question from the one the theory sets up.

Consequence:

- The standard report currently makes pooled look better at every tau.
- The stratified outputs show that leaf methods already win in the strongest completed gap slice.

### 2. The current heterogeneity statistic is the wrong summary to plot

The core code records the signed theoretical gap:

- `heterogeneity_signal_test.append(float(oracle_true - pooled_true))`

Relevant code:

- `src/ctreepo/sim/core/leaf_local_mixture_utility.py:716-769`

The report then plots `hetero_mean_test_gap_signal` against pooled error:

- `scripts/report_lda_tree_methods_paper.py:432-440`
- `scripts/report_lda_tree_methods_paper.py:617-623`

This is a poor diagnostic because the signed quadratic gap cancels when `W_base` has mixed signs.

Empirical check on completed rows:

- correlation between signed mean gap and pooled error: about `-0.060`
- correlation between absolute mean gap and pooled error: about `0.903`

So the report currently has the right mechanism in code but the wrong observable on the plot.

### 3. The pooled baseline is a strong ceiling, not a like-for-like practical comparator

The pooled baseline uses:

- full-document counts,
- the true topic-word table `topics_phi`,
- exact utility parameters `theta_true` and `W_base`,
- model-based mixture inference via `_infer_topic_mixture_from_counts`

Relevant code:

- `src/ctreepo/sim/core/leaf_local_mixture_utility.py:686-700`

The leafwise ridge methods, by contrast, only see the compressed sketch `u(A) = U c_A` and scalar leaf labels:

- `src/ctreepo/sim/core/leaf_local_mixture_utility.py:719-752`
- `src/ctreepo/sim/core/leaf_local_mixture_utility.py:803-830`

This is not a bug, but it is an interpretive hazard. Negative comparisons against this pooled method are conservative. They do not imply that leafwise information is useless; they imply that the current `u(A)` plus ridge pipeline is weaker than a strong pooled oracle ceiling in most regimes.

### 4. `leaf_oracle_sum` is not budget-matched under fixed budgets

`leaf_oracle_sum` always reports full base-leaf query cost:

- `src/ctreepo/sim/core/leaf_local_mixture_utility.py:792-801`

even when `budget_regime = fixed_oracle_budget`.

That makes it a full-information upper bound, not a budget-matched oracle. That is acceptable if stated explicitly, but it should not be read as "the best method under the same budget."

## Bottom Line

Do the simulations make sense?

Yes.

- Stage 1 is clean and correct.
- Stage 2 is also coherent and does create the intended pooled-vs-leaf statistical gap.

Do they currently show off the intended paper story?

Partly.

- Stage 1 already does.
- Stage 2 does not yet do so cleanly in the standard report, mostly because the report is aggregating across the wrong axis and using a weak heterogeneity diagnostic.
- The current practical method is not broadly dominant, but it is not a total miss either: it is already recovering a real gain in the strongest completed heterogeneity regime.

The defensible claim today is:

> Stage 2 establishes a real local-mixture gap beyond pooled-document sufficiency, and the current `leaf_ridge_from_u` practical estimator recovers part of that gain only in the strongest heterogeneity slice.

The claim that is not yet defensible is:

> The current practical tree method is broadly better than pooled document modeling.

## Recommended Next Steps

1. Recut the Stage 2 report stratified by `lambda_multiplier`.
2. Present `lambda = 0` separately as the sanity-control page.
3. Replace the signed gap page with an absolute gap or quadratic-gap magnitude.
4. Add one stronger leafwise baseline, ideally a leafwise plug-in model using inferred leaf mixtures, so recoverability is separated from the current `u(A)` bottleneck.
5. Label `pooled_doc_wrong_model` as a strong wrong-model ceiling and `leaf_oracle_sum` as a full-information upper bound.
6. After the queue finishes, regenerate the report with the corrected stratification before using it for paper-facing claims.
