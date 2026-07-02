# Allocation, Budgeting, and IPW/DSL Weighting

This note is for paper framing only. It does not edit the TeX yet. The goal is
to state the allocation-policy story in a way that is both plain-language and
actually supported by the current Lean theorem surface.

Relevant paper entry point:
- [main_new.tex](/home/mlinegar/ThinkingTrees/paper/ctreepo/main_new.tex)

Relevant current plot/render code:
- [render_markov_sticky_simple_fixed10240_current.py](/home/mlinegar/ThinkingTrees/scripts/render_markov_sticky_simple_fixed10240_current.py)
- [render_markov_sticky_allocation_policy_grid.py](/home/mlinegar/ThinkingTrees/scripts/render_markov_sticky_allocation_policy_grid.py)
- [run_markov_optimization_tradeoff_pipeline.py](/home/mlinegar/ThinkingTrees/scripts/run_markov_optimization_tradeoff_pipeline.py)
- [markov_changepoint_ops_count.py](/home/mlinegar/ThinkingTrees/src/ctreepo/sim/core/markov_changepoint_ops_count.py)

Relevant current outputs:
- sticky final root/leaf bundle:
  [report.md](/home/mlinegar/ThinkingTrees/outputs/markov_v5_simple_current_plots_20260415_233539/report.md)
- sticky allocation-policy bundle in progress:
  [report.md](/home/mlinegar/ThinkingTrees/outputs/markov_v5_sticky_allocation_policy_20260416_000850/report.md)

## 1. The plain-language point we want to make

The reader-facing point is not that "any way of reallocating the supervision
budget over the tree gives the same performance." That claim is too strong.

The safer and more accurate point is:

1. There is a fixed target objective we care about at the root.
2. We cannot afford to label every node, so we choose a sampling/allocation
   design over leaves and internal summaries.
3. If we log the propensities and evaluate with the IPW/DSL machinery, then the
   resulting estimator is design-correct for the weighted objective implied by
   that allocation policy.
4. Different allocation policies can still differ materially in finite-sample
   efficiency, variance, and learning usefulness.

So the right message is:

> The weighting scheme makes the comparison honest. It does not imply that all
> allocations are equally good.

That is the main nuance the paper should explain.

## 2. What the Lean proofs do support

There are really two separate theorem stories.

### 2.1 Exact local-law story

If the local laws hold exactly, then the tree preserves the oracle-relevant
content exactly, and the downstream oracle-indexed objective agrees with the
full-document objective.

Relevant Lean references:
- `one_pass`, `schedule_invariance`, `fold_of_folds` in
  [PAPER_TO_LEAN_MAP.md](/home/mlinegar/ThinkingTrees/lean3/docs/PAPER_TO_LEAN_MAP.md)
- `multi_round_proper`, `dpo_equivalence`, `grpo_equivalence`,
  `grpo_rl_equivalence`
- information-sufficiency bridge:
  `jointTreeSummaryLaw_oracle_factorizationAE_of_localLaws`,
  `jointTreeSummaryLaw_taskRelevantKLIC_zero_ae_of_localLaws`

Paper interpretation:

- In the ideal exact-law regime, the tree representation is not changing the
  target objective at all.
- In that regime, the allocation policy only affects what we choose to observe
  or estimate, not the underlying oracle-equivalent target.

This is the strongest "allocation should not matter" intuition available in the
 current formal story, but it holds only in the exact-preservation regime.

### 2.2 Approximate + sampled story

Once we move to approximate local laws and only sample a subset of nodes, the
formal guarantee changes.

Relevant Lean references:
- approximate local-law budget route in
  [UnifiedOracleRoute.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/OPT/UnifiedOracleRoute.lean):
  `ofApproxLaws`
- IPW unbiasedness in
  [TreeIPW.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/TreeIPW.lean):
  `ipw_preference_loss_connection`
- DSL validity in
  [TreeIPW.lean](/home/mlinegar/ThinkingTrees/lean3/FormalProofs/DSL/TreeIPW.lean):
  `computeDSLBound_valid_from_events`,
  `computeDSLBound_valid_from_events_with_oracleMeasurement`
- certificate transport in
  [PAPER_TO_LEAN_MAP.md](/home/mlinegar/ThinkingTrees/lean3/docs/PAPER_TO_LEAN_MAP.md):
  `tree_gap_bound_transport_upper`,
  `tree_gap_bound_transport_upper_prob`

Paper interpretation:

- Approximate local laws give a budget-bounded distortion route.
- IPW gives design-unbiased estimation of the sampled weighted objective under
  logged positive propensities.
- DSL turns those sampled estimates plus calibration/clipping envelopes into a
  valid bound.

So the theorem-supported claim is:

> If we choose an allocation policy, log the sampling propensities, and use the
> IPW/DSL machinery, then we get a valid estimate and bound for the objective
> induced by that policy.

That is different from saying:

> all allocation policies with the same nominal budget yield the same error.

I do not see that stronger claim proved in Lean.

## 3. What Lean does not currently prove

I do not see a theorem showing that, under a fixed total labeling budget, all
reallocations of that budget across root / leaves / internal summaries recover
the same approximate error or the same finite-sample certificate width.

What is missing for that stronger claim would be something like:

1. a theorem identifying all of those different allocation-weighted objectives
   as the same estimand, or
2. a theorem saying that under the DSL/IPW weighting scheme, their induced gaps
   are equal or asymptotically equivalent, or
3. a variance/efficiency comparison theorem showing when reallocations are
   interchangeable up to lower-order terms.

We do not appear to have that today.

So the paper should not say:

> as long as we follow IPW/DSL weighting, we recover the same budgeted error
> regardless of how we allocate labels over the tree.

That overstates the theorem surface.

## 4. The refined claim we can safely make

This is the version I would recommend using in the paper:

> Reallocating supervision across the tree changes which local summaries receive
> direct feedback, and therefore changes the statistical efficiency of learning
> and auditing. What the IPW/DSL weighting scheme guarantees is that, for any
> chosen allocation policy with logged positive propensities, the resulting
> sampled estimate remains design-correct for the corresponding weighted
> objective. The weighting makes the comparison honest; it does not imply that
> all allocations are equally data-efficient.

An even plainer version:

> The weighting scheme fixes the accounting, not the difficulty.

That is the shortest high-level summary.

## 5. How to explain the budgeting nuance to the reader

There are two distinct comparison types in these plots, and the paper needs to
keep them separate.

### 5.1 Same root budget

`fullXX` means:

- only `XX%` of documents receive root labels
- no local labels are added
- so total full-document-equivalent supervision mass is only `XX/100`

This is a reduced-budget condition.

### 5.2 Same total mass

`*_mass_eq_*` means:

- retain root mass `XX/100`
- reallocate the missing mass `1 - XX/100` to local labels
- keep total full-document-equivalent supervision mass fixed at `1.0`

This is a fixed-total-mass condition.

So:

- `full40` vs `r40_leaf_mass_eq_60p0` is a same-root-budget comparison
- it is not a same-total-mass comparison
- only the mass-preserving policies are comparable to one another at equal total
  mass

This is why we want two separate plot families:

1. replacement view:
   same root budget, different local reallocation choices
2. pure allocation view:
   fixed total mass, compare only the mass-preserving families

## 6. The geometry nuance the reader should know

For the `128`-token documents, leaf size determines how many distinct non-root
summary depths exist.

For the current balanced-tree construction:

- `leaf128`: `1` leaf, no non-root internal depth
- `leaf64`: `2` leaves, still no non-root internal depth distinct from the root
- `leaf32`: `4` leaves, first geometry with `1` non-root internal depth
- `leaf16`: `8` leaves, `2` non-root internal depths
- `leaf8`: `16` leaves, `3` non-root internal depths

This matters because:

- `leaf-only`, `depth-equal`, and `balanced-node` are genuinely different only
  once there are non-root internal layers to allocate mass across
- so `leaf128` and `leaf64` cannot distinguish those internal allocation
  policies
- the first meaningful internal-allocation geometry is `leaf32`

This is already encoded in the launcher logic:
- [launch_markov_sticky_simple_fixed10240_quick.py](/home/mlinegar/ThinkingTrees/scripts/launch_markov_sticky_simple_fixed10240_quick.py)

## 7. What the future allocation-policy figure should say

The future grid plot should support the following argument.

### 7.1 Replacement view

Question:

> If I keep the same root budget `RXX`, where should I spend the missing mass:
> on leaves, spread across internal depths, or across all local node types?

Reader takeaway:

- green root-only line: lower total mass
- colored local-allocation lines: same total mass as `full100`
- this view asks which reallocation policy best converts local supervision into
  root accuracy

### 7.2 Pure allocation view

Question:

> Holding total supervision mass fixed at `1.0`, how should I split that mass
> between root labels and local labels, and among which local layers?

Reader takeaway:

- this is the clean allocation-policy comparison
- root-only `fullXX` should not be drawn as if it were on the same fixed-mass
  footing, except possibly `full100` as the all-root endpoint/reference

## 8. Suggested paper-facing prose

This is candidate prose for later insertion, not a final edit.

### 8.1 Short paragraph version

> Reallocating supervision across the tree changes which summaries receive
> direct feedback, but it does not invalidate the audit as long as the sampling
> propensities are logged and the estimator uses the corresponding IPW/DSL
> weighting. The formal guarantee is design-correctness for the weighted
> objective induced by the chosen allocation policy. This does not mean that all
> allocations are equally good: different policies can have very different
> variance and very different impact on learning, especially once the tree has
> multiple non-root summary depths.

### 8.2 Slightly more explicit version

> The key distinction is between root budget and total supervision mass.
> Conditions such as `full40` keep only the root-label budget fixed and therefore
> use substantially less total supervision than a mass-preserving condition such
> as `r40_*_mass_eq_*`. The latter keeps the total full-document-equivalent mass
> fixed at `1.0` and reallocates the missing `60%` to local summaries. Our
> weighting scheme guarantees that these sampled local signals are accounted for
> correctly; it does not imply that allocating the same total mass to different
> depths of the tree will give identical finite-sample performance.

### 8.3 One-sentence punchline

> The IPW/DSL scheme makes allocation-policy comparisons honest, not trivial.

## 9. Practical recommendation for the paper

For `main_new`, I would recommend saying some version of the following:

1. Exact local-law theory says the tree can preserve the target objective.
2. Approximate local-law + IPW/DSL theory says sampled, weighted audits remain
   valid for the chosen allocation policy.
3. The empirical question is then not whether reallocation is "allowed," but
   which reallocation buys the best root accuracy or tightest certificate under
   a fixed labeling budget.

That is the strongest clean argument I think the current theorem surface
supports.
