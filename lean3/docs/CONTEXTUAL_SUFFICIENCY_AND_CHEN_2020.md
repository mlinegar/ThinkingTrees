# Contextual Sufficiency and Neural Sufficient-Statistic Learning

This note clarifies the position of `FormalProofs.OPT.ContextualQuerySufficiency`
relative to the neural sufficient-statistic literature that motivates the
learning side:

- Chen et al. (2021), *Neural Approximate Sufficient Statistics for Implicit
  Models* (NASS): https://arxiv.org/abs/2010.10079 and
  https://github.com/cyz-ai/neural-approx-ss-lfi
- Chen, Gutmann, Weller (2023), *Is Learning Summary Statistics Necessary for
  Likelihood-free Inference?* (SSS/NASSS):
  https://proceedings.mlr.press/v202/chen23h.html
- Dirmeier, Albert, Perez-Cruz (2025), *Simulation-based Inference for
  High-dimensional Data using Surjective Sequential Neural Likelihood
  Estimation* (SSNL): https://proceedings.mlr.press/v286/dirmeier25a.html
- `sbijax` and `surjectors`, the modern code bases for NASS/NASSS/SNLE and
  surjective flow layers: https://github.com/dirmeier/sbijax and
  https://github.com/dirmeier/surjectors

## What the Lean lane formalizes

The contextual-sufficiency stack is purely deterministic / metric-space.

- **Representation-level sufficiency**
  (`TargetSufficientRepresentation`, `TargetReadoutRealizes`,
  `TargetMeasurable`): a representation preserves target-relevant information
  when its fibers refine the target fibers, equivalently when the target can be
  read out from the representation.
- **Likelihood-family sufficiency**
  (`LikelihoodFamilySufficient`): a statistic is sufficient for an explicit
  likelihood model when collisions preserve every likelihood value for every
  parameter. This is `QuerySufficient` with parameters as contexts.
- **Likelihood-on-state sufficiency**
  (`LikelihoodOnStateFamily`, `LikelihoodFamilySufficientWithin`): if a
  likelihood head is evaluated only through a learned state `z_x`, then `z_x`
  is sufficient for that induced likelihood family. Approximate likelihood
  readouts give deterministic approximate sufficiency with metric slack.
- **Likelihood-free response sufficiency**
  (`LikelihoodFreeResponseSufficient`): a learned representation is sufficient
  for an implicit/simulator/probe target when collisions preserve every probe
  response. This is `QuerySufficient` with probes as contexts.
- **Exact contextual sufficiency** (`QuerySufficient`,
  `TwoSidedContextSufficient`): a representation `rep` is sufficient when its
  fibers refine contextual-response fibers — `rep x = rep y` forces every
  contextual query response to agree exactly.
- **Approximate contextual sufficiency** (`QuerySufficientWithin`,
  `TwoSidedContextSufficientWithin`): collisions of `rep` cost at most a
  metric-space slack `ε` in any contextual query response.
- **Approximate finite-context coverage** (`QuerySufficientWithinOn`,
  `FiniteContextCoversWithin`): if a finite sampled context set controls the
  full response family up to slack, then finite-context approximate sufficiency
  implies full approximate contextual sufficiency.
- **Metric near-collision sufficiency** (`QuerySufficientNearWithin`,
  `ContextReadoutNearPreserving`): continuous learned states can be handled by
  replacing `rep x = rep y` with `dist (rep x) (rep y) <= δ` and adding
  readout-stability hypotheses.
- **Approximate shared-`g` composed-readout bridge**
  (`uniformComposedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin`):
  if one shared `g` plus readout `f` matches the oracle uniformly within `ε`
  across all two-sided contexts, then collisions of the induced leaf state map
  cost at most `2 * ε` in any oracle response. The older
  `composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin`
  theorem is the lower-level algebraic leaf/merge/readout helper.
- **Sliced contextual sufficiency**
  (`SlicedQuerySufficient`, `SlicedQuerySufficientOn`,
  `FiniteSlicesCoverResponseFibers`, `SlicedQuerySufficientWithinOn`): selected
  SSS/NASSS-style slice functions are deterministic probes of the full
  contextual response signature. If collisions preserve those selected slices
  and the slices cover the response fibers, then ordinary contextual
  sufficiency follows. Full coordinate slices and deterministic left-invertible
  slice maps now provide concrete cover witnesses.
- **Markov instance** (`MarkovSufficiency.lean`): the exact `(count, first, last)`
  changepoint sketch satisfies the generic two-sided contextual sufficiency
  condition; count-only summaries do not; sufficiency forces injectivity and
  decoder existence on the exact sketch state. Approximate real-valued count
  corollaries instantiate the generic contextual and sliced bridges.
- **LDA likelihood example** (`BagOfWordsLDARecovery.lean`): the bag-of-words
  histogram factors the ordinary LDA document likelihood family and is exported
  as a `LikelihoodFamilySufficient` instance.
- **Hybrid summary statistics** (`HybridSummarySufficiency.lean`): the product
  summary `(base(x), neural(x))` refines both components; target, likelihood,
  and likelihood-free readouts from the product imply the corresponding
  sufficiency condition. Product sufficiency is equivalent to within-base
  sufficiency, and therefore the neural component separates distinctions left
  inside base-summary fibers.

The core Lean theorems are deterministic statements about set-theoretic /
metric fibers. The random-slice extension adds only event-level probability
transport over selected-slice seeds; it assumes the good selected-slice event
and does not prove an analytic random-direction coverage law.

## What the Lean lane does *not* formalize

- Shannon mutual information `I(g(X); fstar(X))` and infomax inequalities.
- Measure-theoretic conditional MI `I(s; theta | t)` for hybrid summaries.
  The symbolic chain-rule/objective layer is formalized, but not as Shannon
  information over a probability space.
- Variational MI lower bounds (DV, MINE) and the InfoNCE family.
- Analytic random-direction coverage statements for SSS/NASSS. Lean now has an
  event-level probability wrapper for any supplied good selected-slice event.
- PAC-style "low empirical risk implies sufficient with high probability"
  generalization statements.
- SSNL/SNLE likelihood estimation, flow density/Jacobian semantics, posterior
  calibration, or classical posterior-consistency theorems. Set-theoretic
  surjective state factorization, finite Bayes posterior transport, MAP/odds,
  posterior-functional, and posterior-predictive state readouts, mathlib Bayes
  API alignment, and assumption-backed posterior consistency transport are
  formalized.

These are intentionally out of scope, consistent with `INFORMATION_SUFFICIENCY_BRIDGE.md`:
the C-TreePO Lean stack is task-relevant deterministic compression, not a
generic source-coding / information-theory development.

## The Neural Sufficient-Statistic Position

Chen et al. (2020) present infomax learning of approximate sufficient statistics
for **implicit models**: they parameterize a representation `g(x)` and train it
to maximize a variational MI lower bound between `g(x)` and a downstream
oracle/observable. The MI objective stands in for "make `g` retain everything
the oracle could distinguish."

The contextual-sufficiency framing here is the compositional/contextual
extension of that idea:

| | NASS / SSS / SSNL literature | C-TreePO contextual sufficiency |
|---|---|---|
| Object trained | summary / low-dimensional state `S(x)` | same `g` at leaves *and* at merges |
| Sufficiency target | parameter, likelihood family, or oracle/probe variable | family of contextual queries `fstar(L · x · R)` |
| Loss | NASS infomax, SSS slices, SSNL likelihood-on-state | sampled-context contextual loss + dependence auxiliary |
| Theory | MI optimization / likelihood or likelihood-free inference | deterministic representation fibers, readout factorization, ε-slack, and slice bridges |

The 2023 SSS/NASSS point matters operationally: if the contextual response
signature is high-dimensional, learn many low-dimensional random slices instead
of estimating one large MI directly. The 2025 SSNL point matters conceptually:
the learned low-dimensional state is the variable on which downstream inference
is done. In our notation, that is exactly the role of the carrier state
`g(leafInput(x))`.

## Three Status Tiers

| Tier | Meaning | Current items |
|---|---|---|
| Documented inspiration | Literature used for vocabulary, design pressure, or future directions. | SSNL/SNLE likelihood estimation, `surjectors`, MI estimator cautions. |
| Implemented objective/probe | Code path exists and can be run, but the package or estimator internals are not formalized. | `sbijax.NASS`, `sbijax.NASSS`, PyTorch `regression`, `dcorr`, `jsd`, `dv`, `wasserstein`, `infonce`. |
| Machine-checked theorem | Lean formalizes the condition or bridge that the code is meant to satisfy. | `ContextualQuerySufficiency.lean`, `SlicedContextualSufficiency.lean`, `InformationRepresentationSufficiency.lean`, `LikelihoodOnStateSufficiency.lean`, `HybridSummarySufficiency.lean`, `BagOfWordsLDARecovery.lean`, `MarkovSufficiency.lean`. |

The important boundary is that SSS randomness lives in the implemented/probe
tier. The machine-checked statement starts after a finite slice set has been
selected: preservation of those selected slices plus a slice-cover assumption
implies contextual sufficiency.

## 2026-05-05 Markov Ablation Status

The Python Markov contextual-sufficiency ablations are now documented in
`docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md`, with the
full result table in
`outputs/markov_contextual_ablation_grid_report_20260505.md`.

Empirical summary:

- `learned_local_laws` is the exact-zero training lane for the Markov
  `(count, first, last)` sketch.
- Low-weight NASSS is a useful auxiliary inside that lane, but NASS/NASSS
  package objectives alone are not sufficient-state recovery objectives.
- Learned merge and learned decoder variants work when the local-law state
  target remains active.
- Standalone `CleanUnifiedNO` general f/g is an honest learned-operator test
  and remains far from exact-zero in the completed grid.

Lean interpretation: these runs are evidence about optimization, not new Lean
theorems. The checked content is still the exact sketch, contextual/sliced
sufficiency implication surface, and neural-operator/local-law interfaces. Do
not restate the ablation as a convergence theorem for SGD.

## How the Lean bridges connect to a Python training objective

The approximate bridge says: if the trained pipeline reaches contextual loss
`ε` on a sufficiently rich context distribution, the learned `g` is
contextually `2ε`-sufficient. So the Python training objective should:

1. Sample triples `(left, x, right)` from the data distribution.
2. Compute the composed prediction
   `f(g(g(g(left), null), g(g(x), null), g(g(right), null)))`.
3. Compare against `fstar(left · x · right)` (the witness on the concatenated
   sequence — for the Markov DGP, palette-bigram counting on the concatenation).
4. Minimize the per-sample distance.

A dependence term over response-signature vectors is an optional auxiliary loss
motivated by NASS/NASSS; it is not required by the Lean bridge. The current
probe menu is:

- `regression`: SSS-style sliced contextual response prediction.
- `dcorr`: critic-free distance correlation.
- `jsd`: DeepInfoMax/NASS-style shuffled-negative objective.
- `dv`: MINE-style Donsker-Varadhan objective.
- `wasserstein`: Wasserstein dependency objective.
- `infonce`: contrastive response-signature matching.

See `docs/literature/contextual_sufficiency/README.md` and
`docs/contextual_sufficiency_modern_lit_2026-05-04.md` for the paper/code
index.

## Public Lean export surface

All of the relevant theorems are re-exported with paper-facing names in
`FormalProofs/OPT/MainTheorems.lean`:

- `target_sufficient_representation`, `target_readout_realizes`,
  `target_sufficient_iff_exists_readout`, `target_measurable`, and
  `target_sufficient_preserves_target_measurable`
- `likelihood_family_sufficient`,
  `likelihood_family_sufficient_iff_contextual_query_sufficient`, and
  `likelihood_family_sufficient_iff_exists_readout`
- `likelihood_on_state_family`,
  `likelihood_on_state_family_sufficient`,
  `rep_with_state_readout_likelihood_on_state_family_sufficient`, and
  `likelihood_readout_within_implies_likelihood_family_sufficient_within`
- `posterior_on_state_sufficient`,
  `likelihood_sufficient_implies_posterior_sufficient`,
  `surjective_state_posterior_factorization`, and
  `surjective_state_posterior_sufficient_iff_factors`
- `finite_bayes_posterior_sum_eq_one`,
  `likelihood_sufficient_implies_finite_bayes_posterior_sufficient`,
  `finite_bayes_posterior_likelihood_on_state_sufficient`,
  `finite_bayes_posterior_pmf`,
  `finite_bayes_posterior_pmf_to_measure_set`,
  `state_finite_bayes_posterior_pmf`,
  `state_finite_bayes_posterior_pmf_to_measure_set`, and
  `finite_bayes_posterior_pmf_likelihood_on_state_eq_state_pmf`
- `mathlib_conditional_bayes_rule`,
  `mathlib_conditional_probability_finite_fiber_total`,
  `mathlib_kernel_posterior`,
  `mathlib_kernel_posterior_compProd_eq_map_swap`,
  `mathlib_kernel_posterior_with_density_countable`,
  `mathlib_kernel_posterior_eq_with_density`,
  `mathlib_kernel_posterior_rn_deriv`,
  `mathlib_kernel_posterior_unique_ae`,
  `mathlib_pdf_map_eq_with_density`,
  `mathlib_pdf_lintegral_lotus`, and
  `posterior_consistent_iff_mathlib_tendsto_in_measure`
- `finite_posterior_mass_concentrates_at`,
  `finite_posterior_mass_concentrates_at_iff_mathlib_tendsto_in_measure`,
  `finite_bayes_consistency_likelihood_on_state_iff`, and
  `state_readout_finite_bayes_consistency`
- `likelihood_free_response_sufficient`,
  `likelihood_free_response_sufficient_iff_contextual_query_sufficient`, and
  `likelihood_free_response_sufficient_iff_exists_readout`
- `contextual_query_sufficient` (= `QuerySufficient`)
- `contextual_query_sufficient_within` (= `QuerySufficientWithin`)
- `contextual_query_sufficient_within_on`,
  `finite_context_covers_within`, and
  `finite_context_within_implies_contextual_sufficiency_within`
- `contextual_query_sufficient_near_within`,
  `contextual_readout_near_preserving`, and
  `contextual_readout_approx_near_preserving_implies_near_sufficiency`
- `twoSided_context_sufficient` (= `TwoSidedContextSufficient`)
- `twoSided_context_sufficient_within` (= `TwoSidedContextSufficientWithin`)
- `uniform_composable_g` (= `UniformG`), `uniform_composable_leaf`
  (= `UniformG.leaf`), and `uniform_composable_merge` (= `UniformG.merge`)
- `exact_shared_gf_implies_twoSided_contextual_sufficiency`
- `approx_shared_gf_implies_twoSided_contextual_sufficiency`
- algebraic helpers:
  `exact_composed_state_readout_implies_twoSided_contextual_sufficiency`,
  `approx_composed_state_readout_implies_twoSided_contextual_sufficiency`, and
  `exact_composed_state_readout_implies_twoSided_contextual_sufficiency_within_zero`

`UniformG` is endomorphic on one carrier space: leaves are embedded by
`leafInput : Raw -> Carrier`, merge inputs are built by
`mergeInput : Carrier -> Carrier -> Carrier`, and the one learned map is
`g : Carrier -> Carrier`. The readout/oracle head has type `Carrier -> Y`.
The algebraic helpers above still quantify over arbitrary `leaf` and `merge`
maps only as lower-level lemmas; the paper-facing shared-g route uses the
carrier contract.
- `sliced_contextual_signature`, `sliced_query_sufficient`,
  `finite_slices_cover_response_fibers`,
  `finite_sliced_zeroLoss_implies_contextual_sufficiency`, and
  `finite_sliced_within_implies_contextual_sufficiency_within`
- `coordinate_slices_cover_response_fibers`,
  `finite_coordinate_slices_univ_cover_response_fibers`, and
  `left_invertible_slices_cover_response_fibers`
- `likelihood_family_sufficient_no_collision_of_distinguished_likelihood`,
  `bagOfWords_lda_likelihood_family_sufficient`, and
  `twoSided_context_sufficient_iff_likelihood_free_response_sufficient`
- `hybrid_summary`, `hybrid_summary_sufficient_for_base`,
  `hybrid_target_sufficient_iff_within_base_target_sufficient`,
  `hybrid_likelihood_sufficient_neural_separates_likelihood_within_base`,
  `hybrid_likelihood_readout_implies_likelihood_sufficient`, and
  `hybrid_likelihood_on_state_family_sufficient`
- `markov_count_query_sufficient`, the `(count, first, last)` decoder
  theorems, and approximate Markov aliases such as
  `markov_finite_sliced_within_implies_count_query_sufficient_within`.

## When to extend the Lean lane

Extensions worth considering once the Python training side stabilizes:

- Adapter-specific slice-cover lemmas for non-coordinate slice families that do
  not already come with a left inverse.
- Analytic random-direction coverage theorems, if a later proof needs a
  concrete law for the good selected-slice event.
- A stronger Lipschitz algebra for merge/readout composition, beyond the current
  radius-local near-preservation theorem.
- SSNL/SNLE likelihood-estimation or flow density/Jacobian semantics, if those
  become load-bearing rather than inspirational.
- A PAC-style empirical-risk bridge, out of scope unless the formal theory
  becomes load-bearing for paper claims about generalization.
