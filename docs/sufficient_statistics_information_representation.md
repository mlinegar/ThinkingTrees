# Sufficient Statistics, Information, and Representation

This note records the shared vocabulary for the sufficient-statistics lane. It
covers both likelihood-model and likelihood-free versions, while keeping the
formal claim aligned with the current deterministic Lean surface.

## Core Objects

| Symbol | Meaning |
| --- | --- |
| `X` | Raw observation, document, span, or simulator output. |
| `rep(x)` / `z_x` | Learned or hand-built representation / summary statistic. |
| `target(x)` | Task variable, oracle value, response signature, parameter-relevant statistic, or downstream label. |
| `R_K(x)` | Finite contextual response signature `[query(c_i, x)]_{i=1..K}`. |
| `phi^T R_K(x)` | SSS/NASSS-style selected slice target. |
| `D(z_x, R_K(x))` | Dependence proxy objective between a learned state and a response signature. |
| `ell_theta(z_x)` | SSNL/SNLE-style likelihood head evaluated on learned state. |
| `post(z_x)` | NPE/SNPE-style posterior-like readout evaluated on learned state. |
| `p(theta | x)` | Finite/discrete Bayes posterior induced by a fixed prior and likelihood family. |
| `p_n(theta | x)` | Posterior sequence used for assumption-backed consistency or concentration statements. |
| `pi(signal | theta)` | Bayesian-persuasion experiment / signal kernel. |
| `mu_signal`, `posterior(signal)` | Signal distribution and posterior beliefs induced by a finite experiment. |
| `(base(x), neural(x))` | Hybrid summary statistic combining domain and learned summaries. |
| `readout(rep(x))` | Decoder, likelihood head, response head, or downstream inference layer on the representation. |

The central condition is:

```text
rep(x) = rep(y)  =>  target(x) = target(y)
```

Equivalently, `target` factors through `rep`: there exists a readout such that
`readout(rep(x)) = target(x)`. This is the representation-level meaning of
"the summary preserved all information needed for the target."

Lean entry point:

```text
FormalProofs.OPT.InformationRepresentationSufficiency
```

## Likelihood Version

For an explicit likelihood model, the target is the full likelihood family:

```text
likelihood : Theta -> X -> Y
```

A statistic `rep(x)` is sufficient when:

```text
rep(x) = rep(y)  =>  likelihood(theta, x) = likelihood(theta, y)
                    for every theta
```

This is the deterministic fiber form of classical likelihood sufficiency. It
says that no likelihood-relevant information is lost by replacing `x` with
`rep(x)`. In Lean this is:

```text
LikelihoodFamilySufficient rep likelihood
```

and it is exactly `QuerySufficient` with `theta` as the context. The readout
form is:

```text
exists likelihood_readout,
  likelihood_readout(theta, rep(x)) = likelihood(theta, x)
```

This is the formal bridge to classical sufficient statistics and likelihood
factorization without committing to a particular density, domination measure,
or Fisher-Neyman theorem in this tranche.

## Likelihood on State

The SSNL/SNLE pattern is a likelihood model evaluated on a learned lower-
dimensional state:

```text
state : X -> State
state_likelihood : Theta -> State -> Y
likelihood(theta, x) = state_likelihood(theta, state(x))
```

Lean captures the deterministic part:

```text
LikelihoodOnStateFamily state state_likelihood
likelihood_on_state_family_sufficient
```

This says that if likelihood values are functions of `z_x = state(x)`, then
`z_x` is sufficient for that likelihood family. A richer representation is also
sufficient when it has a decoder to `z_x`:

```text
rep_with_state_readout_likelihood_on_state_family_sufficient
```

The approximate version is metric-space based:

```text
likelihood_readout_within_implies_likelihood_family_sufficient_within
```

This is the theorem-backed part of the SSNL analogy. It does not prove that a
surjective flow estimates the true likelihood, that MCMC/VB is consistent, or
that a learned density has calibrated posterior semantics. The set-theoretic
surjective part is now formalized separately: if a surjective state map has
likelihood values constant on state fibers, then the likelihood factors through
a state-space likelihood head (`surjective_state_likelihood_factorization`);
the approximate companion is `surjective_state_likelihood_readout_within`.

## Posterior / Readout on State

The posterior-learning side of SBI often freezes a learned state and trains a
posterior, ratio, moment, interval, or diagnostic readout on that state. Lean now
formalizes only the deterministic part:

```text
state : X -> State
state_posterior : State -> Posterior
posterior(x) = state_posterior(state(x))
```

The theorem-backed claim is:

```text
posterior_on_state_sufficient
```

meaning collisions of `state(x)` preserve the posterior-like object. The module
`PosteriorOnStateSufficiency.lean` also proves:

- `rep_with_state_readout_posterior_on_state_sufficient`: any richer
  representation that decodes the frozen state is sufficient for a posterior
  readout on that state;
- `likelihood_sufficient_implies_posterior_sufficient`: if a posterior-like
  object is explicitly assumed to be determined by the full likelihood family,
  likelihood sufficiency transports to posterior sufficiency;
- `surjective_state_posterior_factorization` and
  `surjective_state_posterior_sufficient_iff_factors`: under set-theoretic
  surjectivity, posterior sufficiency is equivalent to a state-space posterior
  readout;
- `posterior_readout_within_implies_posterior_sufficient_within`: approximate
  readouts give approximate posterior sufficiency with two-sided slack.

This is not a posterior calibration proof, coverage result, or SBI
estimator-consistency statement. It is the deterministic readout/fiber layer
that those methods would need to satisfy.

## Finite Bayes Semantics

`FiniteBayesOnState.lean` adds the finite/discrete Bayes algebra that is safe
to formalize in this tranche. Parameters live in a finite type `Theta`, with a
fixed prior and real-valued likelihood family:

```text
BayesNumerator(theta, x) = prior(theta) * likelihood(theta, x)
BayesEvidence(x) = sum_theta BayesNumerator(theta, x)
BayesPosterior(x)(theta) = BayesNumerator(theta, x) / BayesEvidence(x)
```

The theorem-backed claims are:

- `finite_bayes_posterior_sum_eq_one`: posterior mass sums to one when
  evidence is nonzero;
- `finite_bayes_posterior_map_iff_numerator_map` and its state analogue:
  positive evidence normalization preserves MAP decisions, so MAP can be read
  from unnormalized Bayes numerators;
- `finite_bayes_posterior_odds_eq_numerator_odds` and its state analogue:
  posterior odds cancel the evidence and equal numerator odds;
- `finite_bayes_posterior_expectation_likelihood_on_state_eq_state`: posterior
  expectations and other finite-parameter posterior functionals factor through
  the learned state whenever the likelihood does;
- `finite_bayes_posterior_predictive_likelihood_on_state_eq_state`: finite
  posterior predictive likelihoods factor through learned states when both the
  observed and future likelihoods are state likelihoods;
- `finite_bayes_posterior_risk_likelihood_on_state_eq_state`: posterior Bayes
  risks for arbitrary finite-action losses factor through learned states when
  the likelihood does;
- `finite_bayes_action_likelihood_on_state_iff_state_action`: Bayes-action
  optimality is equivalent before and after replacing observations by the
  learned state;
- `finite_bayes_posterior_set_mass_likelihood_on_state_eq_state` and
  `finite_bayes_credible_at_level_likelihood_on_state_iff_state`: finite
  credible/acceptance-set masses and level claims factor through learned
  states;
- `finite_bayes_posterior_target_eq_inv_one_plus_evidence_ratio_remainder`:
  target posterior mass is exactly the inverse one-plus evidence-ratio
  remainder, giving the algebraic bridge used by the consistency layer;
- `finite_bayes_posterior_determined_by_likelihood`: for a fixed prior, the
  finite Bayes posterior is determined by the likelihood family;
- `finite_bayes_posterior_expectation_determined_by_likelihood`: fixed-prior
  posterior expectations are also determined by the likelihood family;
- `likelihood_sufficient_implies_finite_bayes_posterior_sufficient`: likelihood
  sufficiency transports to finite-Bayes posterior sufficiency;
- `finite_bayes_posterior_likelihood_on_state_sufficient`: if the likelihood is
  evaluated through a state likelihood head, the induced finite Bayes posterior
  is state-sufficient;
- `finite_bayes_posterior_expectation_likelihood_on_state_sufficient`: finite
  posterior expectations are state-sufficient in the likelihood-on-state case;
- `finite_bayes_posterior_predictive_likelihood_on_state_sufficient_observed`:
  for a fixed future observation, finite posterior predictive readouts are
  sufficient in the observed learned state;
- `surjective_state_finite_bayes_posterior_factorization`: under set-theoretic
  surjectivity, finite Bayes posteriors factor through a state-space posterior
  readout.

This is still not a dominated-measure Bayes theorem, continuous posterior
construction, posterior calibration theorem, coverage theorem, MCMC/VB
semantics, density-estimator consistency result, or normalizing-flow
Jacobian/volume-correction theorem.

## Bayesian Persuasion / Information Design Layer

Kamenica and Gentzkow's canonical *Bayesian Persuasion* paper is American
Economic Review 2011, not Econometrica. The formal layer here captures the
finite algebraic core rather than the infinite-state geometric theorem.

For a finite state space and finite signal space, a persuasion experiment is a
signal kernel:

```text
experiment(theta, signal) = Pr(signal | theta)
```

Together with a prior, it induces:

```text
signal_distribution(signal)
  = sum_theta prior(theta) * experiment(theta, signal)

posterior_after_signal(signal)(theta)
  = prior(theta) * experiment(theta, signal)
      / signal_distribution(signal)
```

`BayesianPersuasion.lean` proves:

- `bayesian_persuasion_posterior_eq_finite_bayes`: posterior-after-signal is
  exactly the finite Bayes posterior from `FiniteBayesOnState.lean`;
- `bayesian_persuasion_signal_distribution_probability`: a valid experiment
  and prior induce a probability distribution over signals;
- `bayesian_persuasion_valid_signal_bayes_plausible`: under full support of
  retained signal realizations, induced posteriors are Bayes-plausible, i.e.
  their weighted barycenter is the prior;
- `bayesian_persuasion_valid_signal_scheme_feasible`: valid full-support
  experiments induce feasible finite persuasion schemes;
- `bayesian_persuasion_splitting_experiment_valid`: any Bayes-plausible finite
  posterior decomposition with normalized posterior labels and positive prior
  support defines a valid signal experiment by the standard splitting formula;
- `bayesian_persuasion_splitting_signal_distribution_eq_weight` and
  `bayesian_persuasion_splitting_posterior_eq`: the splitting experiment
  recovers the supplied signal weights and positive-weight posterior labels;
- `bayesian_persuasion_receiver_bayes_action_iff_best_response`: a receiver
  best response is exactly a finite Bayes action for the loss
  `-receiverUtility`;
- `bayesian_persuasion_concavification_iff_optimal_value`: a supplied finite
  concavification witness is exactly an optimal-value witness in the symbolic
  persuasion surface.

`BayesianPersuasionEconomics.lean` then builds this into the existing economic
formulations:

- `bayesian_persuasion_receiver_loss_factors_through_belief`: receiver
  posterior loss over signals factors through the posterior-belief state;
- `bayesian_persuasion_sender_indirect_value_factors_through_belief`: sender
  indirect value with a belief-indexed receiver-action selector factors through
  the posterior-belief state;
- `bayesian_persuasion_signal_indirect_value_eq_scheme_value`: indirect
  experiment value is exactly `PersuasionSchemeValue` for the induced
  posterior distribution;
- `bayesian_persuasion_receiver_obedient_iff_bayes_action`: direct
  recommendation obedience is equivalent to finite Bayes-action optimality for
  negative receiver utility loss on each positive-probability recommendation;
- `bayesian_persuasion_indirect_value_eq_of_same_posterior_distribution`:
  belief-based persuasion value is invariant under same signal-indexed
  posterior distributions.

`BayesianPersuasionDirect.lean` adds the finite direct-recommendation
accounting layer. Given an experiment and a deterministic signal-to-action
rule, it constructs the action-valued recommendation experiment obtained by
pooling all signals that recommend the same action:

```text
recommendation(theta, a)
  = sum_{signal : actionOfSignal(signal) = a}
      experiment(theta, signal)
```

It proves:

- `bayesian_persuasion_direct_recommendation_from_experiment_valid`: this
  pooled recommendation kernel is a valid finite experiment when the original
  experiment is valid;
- `bayesian_persuasion_direct_recommendation_ex_ante_sender_value_eq`: pooling
  preserves ex-ante sender value exactly by finite sum regrouping;
- `bayesian_persuasion_direct_recommendation_sender_value_eq`: under full
  support of original signals and pooled recommendations, the posterior-value
  formulation is preserved as well.

This gives the repo a checked bridge from Bayes semantics to the
information-design intuition: persuasion is optimization over posterior belief
distributions constrained by Bayes plausibility. V1 proves the finite splitting
construction under explicit support/normalization assumptions, but does not
prove compact-action existence, geometric concavification, measurable
selection, tie-breaking existence, direct-revelation existence, or
infinite-state optimal signal existence.

## Mathlib Bayes Alignment

`MathlibBayesBridge.lean` connects the bounded local Bayes layer to the
probability APIs that already exist in mathlib:

- `mathlib_conditional_bayes_rule` aliases mathlib's event-level Bayes theorem
  for conditional probabilities;
- `mathlib_conditional_probability_apply`,
  `mathlib_conditional_probability_condition_twice`,
  `mathlib_conditional_probability_total_complement`, and
  `mathlib_conditional_probability_finite_fiber_total` expose mathlib's
  conditional-probability algebra and finite-fiber law of total probability;
- `mathlib_conditional_expectation`,
  `mathlib_integral_conditional_expectation`,
  `mathlib_conditional_expectation_indicator`,
  `mathlib_rn_deriv_ae_eq_conditional_expectation`, and
  `mathlib_conditional_expectation_independent_eq_integral` expose mathlib's
  conditional-expectation semantics, integral identities, indicator rule,
  Radon-Nikodym bridge, and independence simplification;
- `mathlib_kernel_posterior`,
  `mathlib_kernel_posterior_compProd_eq_map_swap`, and
  `mathlib_kernel_posterior_with_density_countable` expose mathlib's
  kernel/disintegration posterior surface;
- `mathlib_kernel_posterior_eq_with_density`,
  `mathlib_kernel_posterior_rn_deriv`,
  `mathlib_kernel_posterior_unique_ae`,
  `mathlib_kernel_posterior_comp_self`,
  `mathlib_kernel_posterior_posterior`, and
  `mathlib_kernel_posterior_comp` expose the main posterior density,
  Radon-Nikodym, uniqueness, inversion, and composition facts from mathlib;
- `mathlib_has_pdf`, `mathlib_pdf_map_eq_with_density`,
  `mathlib_pdf_map_eq_set_lintegral`, and `mathlib_pdf_lintegral_lotus`
  expose the dominated-density/PDF layer mathlib already proves;
- `finite_bayes_posterior_pmf` and `state_finite_bayes_posterior_pmf` package
  the repo's real-valued finite Bayes posteriors as mathlib `PMF`s under
  nonnegative-prior, nonnegative-likelihood, and positive-evidence assumptions;
- `finite_bayes_posterior_pmf_likelihood_on_state_eq_state_pmf` proves that
  the raw PMF and state PMF are identical when the likelihood family factors
  through the learned state;
- `finite_bayes_posterior_pmf_to_measure_singleton`,
  `finite_bayes_posterior_pmf_to_measure_set`, and their state-space analogues
  identify singleton and arbitrary-event masses of the induced mathlib
  measures; and
- `posterior_consistent_iff_mathlib_tendsto_in_measure` plus
  `finite_posterior_mass_concentrates_at_iff_mathlib_tendsto_in_measure` make
  explicit that the local convergence predicates are mathlib
  `TendstoInMeasure`.

This is alignment and reuse, not a new continuous-state Bayes theorem for the
learned representations. The current theorem-backed continuous semantics are
mathlib's own kernel posterior APIs; the repo-specific learned-state results
remain deterministic sufficiency/factorization plus finite/discrete Bayes
transport.

## Posterior Consistency Layer

`PosteriorConsistency.lean` adds the first consistency vocabulary. It reuses the
repo-wide convergence-in-probability definition from `DSL.AsymptoticCore`:

```text
posterior_consistent
  := posteriorSeq_n -> posteriorLimit in probability
```

For finite parameter spaces, it also records posterior concentration as mass on
the target parameter tending to one:

```text
finite_posterior_mass_concentrates_at
```

The theorem-backed part is transport, not a classical consistency proof:

- `posterior_consistency_of_pointwise_equal`: pointwise-equal posterior
  sequences have the same consistency behavior;
- `finite_posterior_mass_concentration_of_pointwise_equal`: pointwise-equal
  finite posterior sequences have the same concentration behavior;
- `finite_bayes_consistency_likelihood_on_state_iff`: finite-Bayes posterior
  concentration for a likelihood-on-state family is equivalent to concentration
  of the induced state posterior sequence;
- `state_readout_finite_bayes_consistency`: exact state decoders transport
  finite-Bayes posterior concentration.
- `finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition`
  and its state analogue: an assumption bundle with identifiability, prior
  positivity, likelihood-ratio concentration, and evidence-ratio posterior
  transform concentration implies finite posterior mass concentration.

The statistical ingredients that would prove concentration are named assumption
bundles, not proved:

```text
finite_bayes_posterior_consistency_assumption
state_finite_bayes_posterior_consistency_assumption
finite_bayes_likelihood_ratio_consistency_condition
state_finite_bayes_likelihood_ratio_consistency_condition
```

These bundles name identifiability, prior positivity, likelihood-ratio
concentration, and either the final concentration claim or the evidence-ratio
posterior-transform convergence assumption. V1 therefore formalizes the
general framework, the deterministic evidence-ratio bridge, and state/readout
transport of assumed consistency. It still does not prove Schwartz-style
consistency, dominated-measure Bayes, continuous posterior construction,
posterior calibration, coverage, MCMC/VB semantics, density-estimator
consistency, continuous-mapping theorems for every posterior transform, or
SSNL/SNLE convergence.

## Hybrid Summary Version

Makinen et al. phrase hybrid summary learning as adding a learned summary
`s(d)` to an existing/domain summary `t(d)`, targeting extra information beyond
`t`. The paper writes the information target as conditional MI
`I(s; theta | t)`, equivalently maximizing information in the concatenated
summary `[t, s]` while `I(t; theta)` is fixed.

Lean now has a **symbolic** objective layer for this claim, not a
measure-theoretic information theory layer. `HybridInformationObjectives.lean`
defines argmax/argmin predicates, a `HybridMIChainRule` interface for

```text
I((t, s); theta) = I(s; theta | t) + I(t; theta)
```

and proves that maximizing the conditional term is equivalent to maximizing the
joint hybrid term when the base summary is fixed. It also proves optimizer
bridges for EPE/posterior and classifier/JSD losses when the loss is supplied
as a negated or order-reversing information proxy.

The deterministic counterpart is fiber-theoretic:

```text
hybrid(d) = (base(d), neural(d))

base(x) = base(y) and neural(x) = neural(y)
  => target(x) = target(y)
```

That is `WithinBaseTargetSufficient`. It is equivalent to ordinary sufficiency
of the product summary:

```text
hybrid_target_sufficient_iff_within_base_target_sufficient
```

The "extra information beyond the base summary" reading is captured by
separation theorems:

```text
base(x) = base(y) and target(x) != target(y)
  => neural(x) != neural(y)
```

with likelihood and likelihood-free analogues. Approximate hybrid readout
theorems give metric slack versions for learned low-data readouts.

Concrete anchors are now theorem-backed:

- Markov: `(count-only, endpoint residual)` is a hybrid summary sufficient for
  all two-sided changepoint-count queries.
- LDA: `(bagOfWords, neural)` is sufficient for the ordinary bag-of-words LDA
  likelihood family; if a hybrid is sufficient for order/contextual responses,
  the neural component must separate response distinctions left inside
  bag-of-words fibers.

The remaining Makinen empirical claims remain empirical: improved cosmological
constraints, finite-simulation efficiency, and posterior calibration are
documented and probed, not proved.

## Likelihood-Free Version

For implicit models, simulator outputs, and neural summary-statistic learning,
we usually do not have a tractable likelihood. The target becomes a family of
responses or probes:

```text
response : Probe -> X -> Y
```

Examples:

- NASS target variable or oracle-side response.
- SSS/NASSS selected slices of a response signature.
- contextual queries `fstar(left * x * right)`.
- simulator diagnostics or posterior-quality probes.
- downstream labels/readouts used to test whether the state is useful.

The sufficient-representation condition is:

```text
rep(x) = rep(y)  =>  response(probe, x) = response(probe, y)
                    for every probe
```

In Lean this is:

```text
LikelihoodFreeResponseSufficient rep response
```

and it is exactly `QuerySufficient` with probes as contexts. This is the formal
place where NASS/SSS-style learned summaries connect to C-TreePO contextual
sufficiency.

## Dependence Objective Layer

The NASS/dependence-objective line supplies ways to train `z_x`, but those
objectives are not themselves sufficiency theorems. This repo now records the
objective vocabulary explicitly:

```text
proxy(candidate)       -- symbolic dependence / information proxy
loss(candidate)        -- training loss
information(candidate) -- symbolic target information objective
```

`DependenceObjectiveProxies.lean` proves optimizer algebra only:

- an order-reversing loss has the same optima as maximizing its proxy;
- if `|proxy - information| <= epsilon` uniformly, exact proxy maximizers are
  `2 * epsilon`-near-maximizers of the information objective;
- a pointwise lower-bound relation alone is insufficient to transport argmaxes.

Paper-facing aliases map the implemented objective menu to this symbolic layer:

- `mine_dv_loss_min_iff_proxy_max` for MINE/DV;
- `deep_infomax_jsd_loss_min_iff_proxy_max` for Deep InfoMax/JSD;
- `infonce_loss_min_iff_proxy_max` for InfoNCE/CPC;
- `distance_correlation_proxy_max` for distance correlation;
- `wasserstein_dependency_proxy_max` for Wasserstein dependency.

The negative result `lowerBoundProxy_alone_counterexample` is why Poole/Song-
Ermon-style cautions stay visible in the docs: lower bounds, negative sampling,
and finite critic training require extra tightness, order, or estimation
assumptions before they can justify information-optimality claims.

## Information and Representation

The Lean information claim is task-relative, not source-coding-relative.

What we formalize:

- representation fibers refine target fibers;
- target readout/factorization through the representation;
- preservation of every downstream quantity that is measurable with respect to
  the target;
- likelihood-family sufficiency as contextual sufficiency over parameters;
- likelihood-free response sufficiency as contextual sufficiency over probes;
- finite selected-slice bridges for SSS/NASSS-style learning;
- event-level random finite-slice probability transport;
- likelihood-on-state factorization for SSNL/SNLE-style readouts;
- set-theoretic surjective state likelihood factorization;
- posterior/readout-on-state sufficiency and surjective posterior
  factorization;
- finite Bayesian-persuasion Bayes-plausibility and receiver best-response
  algebra;
- symbolic dependence-proxy/loss optimization bridges;
- deterministic uniform-proxy-error near-optimality;
- hybrid product-summary sufficiency for hand-built plus neural summaries;
- approximate finite-context coverage and approximate selected-slice coverage;
- metric near-collision sufficiency for continuous learned representations;
- concrete coordinate-slice and left-invertible slice-cover witnesses;
- an ordinary bag-of-words LDA likelihood-family example;
- approximate Markov count-query corollaries.

What we do not claim in this tranche:

- Shannon mutual information identities;
- variational MI lower-bound correctness;
- InfoNCE/CPC negative-sampling or contrastive-estimator consistency;
- distance-correlation independence theorem;
- optimal-transport duality for Wasserstein dependency;
- analytic random-direction coverage guarantees;
- PAC generalization from empirical loss;
- density/Jacobian semantics, posterior calibration, exact SSNL/SNLE likelihood
  estimation, or classical posterior-consistency theorems beyond the
  assumption-backed transport framework.
- infinite-state Bayesian-persuasion geometry, compact-action existence,
  measurable selection, geometric concavification, or optimal signal existence.

The practical interpretation is:

```text
learning objective  -> tries to discover z_x
Lean theorem        -> states what it means for z_x to be sufficient
diagnostics         -> test whether z_x preserved target/probe information
```

## Current Lean Crosswalk

| Concept | Lean Name |
| --- | --- |
| Representation sufficient for target | `TargetSufficientRepresentation` |
| Target readout from representation | `TargetReadoutRealizes` |
| Factorization/readout equivalence | `targetSufficient_iff_exists_readout` |
| Downstream target-measurability | `TargetMeasurable` |
| Preserve target-measurable quantities | `targetSufficient_preserves_targetMeasurable` |
| Likelihood-family sufficiency | `LikelihoodFamilySufficient` |
| Likelihood-family readout equivalence | `likelihoodFamilySufficient_iff_exists_readout` |
| Likelihood-family no-bad-collision theorem | `likelihood_family_sufficient_no_collision_of_distinguished_likelihood` |
| Likelihood-on-state family | `likelihood_on_state_family` |
| Likelihood-on-state sufficiency | `likelihood_on_state_family_sufficient` |
| Rich representation with state decoder | `rep_with_state_readout_likelihood_on_state_family_sufficient` |
| Approximate likelihood readout bridge | `likelihood_readout_within_implies_likelihood_family_sufficient_within` |
| Bag-of-words LDA likelihood-family sufficiency | `bagOfWords_lda_likelihood_family_sufficient` |
| LDA bag-of-words hybrid likelihood sufficiency | `lda_bow_hybrid_likelihood_sufficient` |
| LDA within-bag response residual separation | `lda_bow_hybrid_neural_separates_response_within_bagOfWords` |
| Likelihood-free response sufficiency | `LikelihoodFreeResponseSufficient` |
| Likelihood-free readout equivalence | `likelihoodFreeResponseSufficient_iff_exists_readout` |
| Two-sided contextual sufficiency as likelihood-free sufficiency | `twoSided_context_sufficient_iff_likelihood_free_response_sufficient` |
| Sliced contextual sufficiency as likelihood-free sufficiency | `sliced_sufficiency_implies_likelihood_free_response_sufficient` |
| Contextual response sufficiency | `QuerySufficient`, `QuerySufficientWithin` |
| Approximate finite-context bridge | `finite_context_within_implies_contextual_sufficiency_within` |
| Near-collision metric sufficiency | `contextual_query_sufficient_near_within` |
| Approximate readout/near-preservation bridge | `contextual_readout_approx_near_preserving_implies_near_sufficiency` |
| SSS/NASSS finite slice bridge | `finiteSliced_zeroLoss_implies_querySufficient` |
| Hybrid product summary | `hybrid_summary` |
| Symbolic hybrid MI chain rule | `hybrid_mi_chain_rule` |
| Conditional MI vs joint MI argmax equivalence | `hybrid_cmi_argmax_iff_joint_mi_argmax` |
| EPE loss minimizer vs information maximizer | `hybrid_epe_loss_min_iff_information_max` |
| Classifier loss minimizer vs information-proxy maximizer | `hybrid_classifier_loss_min_iff_information_proxy_max` |
| Dependence proxy epsilon-argmax | `dependence_proxy_epsilon_argmax` |
| Uniform proxy error to information near-argmax | `uniform_proxy_error_argmax_implies_information_epsilon_argmax` |
| MINE/DV loss minimizer vs proxy maximizer | `mine_dv_loss_min_iff_proxy_max` |
| Deep InfoMax/JSD loss minimizer vs proxy maximizer | `deep_infomax_jsd_loss_min_iff_proxy_max` |
| InfoNCE/CPC loss minimizer vs proxy maximizer | `infonce_loss_min_iff_proxy_max` |
| Distance-correlation symbolic proxy maximizer | `distance_correlation_proxy_max` |
| Wasserstein-dependency symbolic proxy maximizer | `wasserstein_dependency_proxy_max` |
| Within-base target sufficiency | `within_base_target_sufficient` |
| Hybrid iff within-base target sufficiency | `hybrid_target_sufficient_iff_within_base_target_sufficient` |
| Neural separates target distinctions within base fibers | `hybrid_target_sufficient_neural_separates_target_within_base` |
| Hybrid iff within-base likelihood sufficiency | `hybrid_likelihood_sufficient_iff_within_base_likelihood_sufficient` |
| Neural separates likelihood distinctions within base fibers | `hybrid_likelihood_sufficient_neural_separates_likelihood_within_base` |
| Hybrid collision impossible for likelihood-distinct documents | `hybrid_collision_impossible_of_distinguished_likelihood` |
| Hybrid target readout sufficiency | `hybrid_target_readout_implies_target_sufficient` |
| Hybrid likelihood readout sufficiency | `hybrid_likelihood_readout_implies_likelihood_sufficient` |
| Hybrid likelihood-on-state sufficiency | `hybrid_likelihood_on_state_family_sufficient` |
| Approximate hybrid likelihood readout bridge | `hybrid_likelihood_readout_within_implies_within_base_likelihood_sufficient_within` |
| Approximate hybrid response readout bridge | `hybrid_response_readout_within_implies_within_base_response_sufficient_within` |
| Concrete coordinate slice cover | `finite_coordinate_slices_univ_cover_response_fibers` |
| Left-invertible slice cover | `left_invertible_slices_cover_response_fibers` |
| Approximate finite sliced bridge | `finite_sliced_within_implies_contextual_sufficiency_within` |
| Random finite sliced probability transport | `random_finite_sliced_contextual_sufficiency_failure_prob_le` |
| Approximate random finite sliced probability transport | `random_finite_sliced_within_contextual_sufficiency_failure_prob_le` |
| Surjective state likelihood factorization | `surjective_state_likelihood_factorization` |
| Surjective state likelihood iff factorization | `surjective_state_likelihood_sufficient_iff_factors` |
| Approximate surjective state likelihood readout | `surjective_state_likelihood_readout_within` |
| Posterior/readout on state sufficiency | `posterior_on_state_sufficient` |
| Likelihood sufficiency transports to posterior sufficiency | `likelihood_sufficient_implies_posterior_sufficient` |
| Surjective posterior state factorization | `surjective_state_posterior_factorization` |
| Surjective posterior iff factorization | `surjective_state_posterior_sufficient_iff_factors` |
| Approximate posterior readout bridge | `posterior_readout_within_implies_posterior_sufficient_within` |
| Finite Bayes posterior normalization | `finite_bayes_posterior_sum_eq_one` |
| Positive normalization preserves finite Bayes MAP decisions | `finite_bayes_posterior_map_iff_numerator_map`, `state_finite_bayes_posterior_map_iff_numerator_map` |
| Finite Bayes posterior odds cancel evidence | `finite_bayes_posterior_odds_eq_numerator_odds`, `state_finite_bayes_posterior_odds_eq_numerator_odds` |
| Finite Bayes posterior expectation/readout | `finite_bayes_posterior_expectation`, `state_finite_bayes_posterior_expectation` |
| Likelihood-on-state posterior expectation equality | `finite_bayes_posterior_expectation_likelihood_on_state_eq_state` |
| Likelihood-on-state posterior expectation sufficiency | `finite_bayes_posterior_expectation_likelihood_on_state_sufficient` |
| Finite Bayes posterior predictive/readout | `finite_bayes_posterior_predictive`, `state_finite_bayes_posterior_predictive` |
| Likelihood-on-state posterior predictive equality | `finite_bayes_posterior_predictive_likelihood_on_state_eq_state` |
| Likelihood-on-state posterior predictive observed-state sufficiency | `finite_bayes_posterior_predictive_likelihood_on_state_sufficient_observed` |
| Finite Bayes risk and Bayes action semantics | `finite_bayes_posterior_risk`, `state_finite_bayes_posterior_risk`, `finite_bayes_action`, `state_finite_bayes_action` |
| Likelihood-on-state Bayes risk/action transport | `finite_bayes_posterior_risk_likelihood_on_state_eq_state`, `finite_bayes_action_likelihood_on_state_iff_state_action` |
| Finite credible/acceptance-set mass | `finite_bayes_posterior_set_mass`, `state_finite_bayes_posterior_set_mass`, `finite_bayes_credible_at_level`, `state_finite_bayes_credible_at_level` |
| Likelihood-on-state credible/acceptance-set transport | `finite_bayes_posterior_set_mass_likelihood_on_state_eq_state`, `finite_bayes_credible_at_level_likelihood_on_state_iff_state` |
| Evidence-ratio target-posterior algebra | `finite_bayes_posterior_target_eq_inv_one_plus_evidence_ratio_remainder`, `state_finite_bayes_posterior_target_eq_inv_one_plus_evidence_ratio_remainder` |
| Fixed-prior finite Bayes posterior determined by likelihood | `finite_bayes_posterior_determined_by_likelihood` |
| Fixed-prior posterior expectation determined by likelihood | `finite_bayes_posterior_expectation_determined_by_likelihood` |
| Likelihood sufficiency transports to finite-Bayes posterior sufficiency | `likelihood_sufficient_implies_finite_bayes_posterior_sufficient` |
| Finite-Bayes posterior on state sufficiency | `finite_bayes_posterior_likelihood_on_state_sufficient` |
| Surjective finite-Bayes posterior factorization | `surjective_state_finite_bayes_posterior_factorization` |
| Mathlib conditional Bayes rule alias | `mathlib_conditional_bayes_rule` |
| Mathlib conditional probability application formula | `mathlib_conditional_probability_apply` |
| Mathlib condition-twice identity | `mathlib_conditional_probability_condition_twice` |
| Mathlib finite-fiber law of total probability | `mathlib_conditional_probability_finite_fiber_total` |
| Mathlib conditional expectation layer | `mathlib_conditional_expectation`, `mathlib_integral_conditional_expectation`, `mathlib_conditional_expectation_indicator`, `mathlib_rn_deriv_ae_eq_conditional_expectation` |
| Mathlib kernel posterior alias | `mathlib_kernel_posterior` |
| Mathlib countable posterior density/Bayes formula alias | `mathlib_kernel_posterior_with_density_countable` |
| Mathlib posterior density formula under AC | `mathlib_kernel_posterior_eq_with_density` |
| Mathlib posterior RN derivative identity | `mathlib_kernel_posterior_rn_deriv` |
| Mathlib posterior uniqueness a.e. | `mathlib_kernel_posterior_unique_ae` |
| Mathlib posterior inversion/composition facts | `mathlib_kernel_posterior_comp_self`, `mathlib_kernel_posterior_posterior`, `mathlib_kernel_posterior_comp` |
| Mathlib PDF/RN derivative layer | `mathlib_has_pdf`, `mathlib_pdf_map_eq_with_density`, `mathlib_pdf_map_eq_set_lintegral`, `mathlib_pdf_lintegral_lotus` |
| Finite Bayes posterior as mathlib PMF | `finite_bayes_posterior_pmf` |
| Finite Bayes PMF singleton measure mass | `finite_bayes_posterior_pmf_to_measure_singleton` |
| Finite Bayes PMF arbitrary-event measure mass | `finite_bayes_posterior_pmf_to_measure_set` |
| State finite Bayes posterior as mathlib PMF | `state_finite_bayes_posterior_pmf` |
| State finite Bayes PMF arbitrary-event measure mass | `state_finite_bayes_posterior_pmf_to_measure_set` |
| Raw/state finite Bayes PMF equality for likelihood-on-state families | `finite_bayes_posterior_pmf_likelihood_on_state_eq_state_pmf` |
| Posterior consistency in probability | `posterior_consistent` |
| Posterior consistency iff mathlib `TendstoInMeasure` | `posterior_consistent_iff_mathlib_tendsto_in_measure` |
| Mathlib convergence-in-measure bridge helpers | `mathlib_tendsto_in_measure_congr`, `mathlib_tendsto_in_measure_exists_seq_tendsto_ae` |
| Finite posterior mass concentration | `finite_posterior_mass_concentrates_at` |
| Finite posterior mass concentration iff mathlib `TendstoInMeasure` | `finite_posterior_mass_concentrates_at_iff_mathlib_tendsto_in_measure` |
| Finite Bayes posterior consistency assumption bundle | `finite_bayes_posterior_consistency_assumption` |
| State finite Bayes posterior consistency assumption bundle | `state_finite_bayes_posterior_consistency_assumption` |
| Pointwise equality transports posterior consistency | `posterior_consistency_of_pointwise_equal` |
| Pointwise equality transports finite posterior concentration | `finite_posterior_mass_concentration_of_pointwise_equal` |
| Likelihood-on-state finite Bayes concentration equivalence | `finite_bayes_consistency_likelihood_on_state_iff` |
| State readout transports finite Bayes concentration | `state_readout_finite_bayes_consistency` |
| Evidence-ratio consistency sufficient condition | `finite_bayes_likelihood_ratio_consistency_condition`, `finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition` |
| State evidence-ratio consistency sufficient condition | `state_finite_bayes_likelihood_ratio_consistency_condition`, `state_finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition` |
| Bayesian-persuasion finite signal experiment | `bayesian_persuasion_signal_experiment_valid` |
| Signal-induced posterior as finite Bayes posterior | `bayesian_persuasion_posterior_eq_finite_bayes` |
| Signal distribution is a finite probability vector | `bayesian_persuasion_signal_distribution_probability` |
| Bayes-plausibility from valid full-support experiment | `bayesian_persuasion_valid_signal_bayes_plausible` |
| Valid experiment induces feasible persuasion scheme | `bayesian_persuasion_valid_signal_scheme_feasible` |
| Finite splitting construction is a valid experiment | `bayesian_persuasion_splitting_experiment_valid` |
| Splitting construction recovers signal weights | `bayesian_persuasion_splitting_signal_distribution_eq_weight` |
| Splitting construction recovers positive-weight posteriors | `bayesian_persuasion_splitting_posterior_eq` |
| Receiver best response iff finite Bayes action | `bayesian_persuasion_receiver_bayes_action_iff_best_response` |
| Symbolic concavification/optimal value equivalence | `bayesian_persuasion_concavification_iff_optimal_value` |
| Receiver loss factors through posterior belief | `bayesian_persuasion_receiver_loss_factors_through_belief` |
| Sender indirect value factors through posterior belief | `bayesian_persuasion_sender_indirect_value_factors_through_belief` |
| Indirect experiment value equals persuasion-scheme value | `bayesian_persuasion_signal_indirect_value_eq_scheme_value` |
| Direct recommendation obedience iff finite Bayes action | `bayesian_persuasion_receiver_obedient_iff_bayes_action` |
| Persuasion value invariant under same posterior distribution | `bayesian_persuasion_indirect_value_eq_of_same_posterior_distribution` |
| Direct recommendation from arbitrary experiment | `bayesian_persuasion_direct_recommendation_from_experiment` |
| Pooled direct recommendation is a valid experiment | `bayesian_persuasion_direct_recommendation_from_experiment_valid` |
| Pooling preserves ex-ante sender value | `bayesian_persuasion_direct_recommendation_ex_ante_sender_value_eq` |
| Full-support pooling preserves posterior sender value | `bayesian_persuasion_direct_recommendation_sender_value_eq` |
| Markov approximate count-query bridge | `markov_finite_sliced_within_implies_count_query_sufficient_within` |
| Markov count-plus-endpoint hybrid sufficiency | `markov_count_endpoint_hybrid_two_sided_sufficient` |
