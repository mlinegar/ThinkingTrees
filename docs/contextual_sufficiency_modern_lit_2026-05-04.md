# Contextual Sufficiency: Modern Objective Pull-Through

Date: 2026-05-04

This note records the pieces we are pulling into the unified-g/contextual
sufficiency lane. The canonical runtime package for the learned sufficient-state
experiments is now `sbijax==0.3.6`, exposed through the optional
`contextual_sbi` dependency group and the JAX probe
`scripts/probe_contextual_sbijax.py`. The PyTorch `CleanUnifiedNO` probe remains
the baseline/comparison path.

Durable index locations:

- Literature manifest: `docs/literature/contextual_sufficiency/manifest.json`
- Human-readable literature index:
  `docs/literature/contextual_sufficiency/README.md`
- BibTeX keys: `ChenEtAl2021NASS`, `ChenGutmannWeller2023SSS`,
  `DirmeierAlbertPerezCruz2025SSNL`, `DirmeierUlzegaMiraAlbert2024SBIJAX`,
  `Dirmeier2024Surjectors`, `MakinenEtAl2024HybridSummaryStatistics`,
  `BelghaziEtAl2018MINE`, `HjelmEtAl2019DeepInfoMax`,
  `OordEtAl2018CPCInfoNCE`, `SzekelyRizzoBakirov2007DistanceCorrelation`,
  `OzairEtAl2019WassersteinDependency`, `PooleEtAl2019VariationalMIBounds`,
  and `SongErmon2020UnderstandingMIEstimators`
- Downloaded PDFs:
  `docs/literature/contextual_sufficiency/dirmeier_albert_perezcruz_2025_ssnl.pdf`
  and
  `docs/literature/contextual_sufficiency/makinen_et_al_2024_hybrid_summary_statistics.pdf`

## Core Translation

Our target object is not a parameter posterior. It is the contextual response
fiber:

```text
R_K(x) = [fstar(left_i * x * right_i)]_{i=1..K}
z_x = g(leafInput(embed(x)))
```

The learning problem is to make `z_x` preserve `R_K(x)` under many contexts.
This is the compositional analogue of learning neural sufficient statistics for
implicit models.

The common formal vocabulary is now:

```text
TargetSufficientRepresentation rep target:
  rep(x) = rep(y) -> target(x) = target(y)

LikelihoodFamilySufficient rep likelihood:
  rep(x) = rep(y) -> likelihood(theta, x) = likelihood(theta, y)

LikelihoodFreeResponseSufficient rep response:
  rep(x) = rep(y) -> response(probe, x) = response(probe, y)
```

These live in `FormalProofs.OPT.InformationRepresentationSufficiency`. The
likelihood form treats parameters as contexts; the likelihood-free form treats
simulator probes, contextual queries, or response-signature slices as contexts.
Both reduce to the same representation/fiber condition.

## Borrowed vs. Not Claimed

| Source | Borrowed Into This Repo | Not Claimed / Not Formalized | Machine-Checked Surface |
| --- | --- | --- | --- |
| Chen et al. 2021 NASS | Dependence objectives between a learned summary and an oracle-side target. | No Shannon MI theorem, variational-bound proof, estimator-consistency proof, or PAC generalization claim. | `LikelihoodFreeResponseSufficient`, `QuerySufficient`, `QuerySufficientWithin`, composed `g/f` sufficiency bridges, and symbolic dependence-proxy aliases in `DependenceObjectiveProxies.lean`. |
| Chen/Gutmann/Weller 2023 SSS/NASSS | Low-dimensional selected slices `phi^T R_K(x)` as easier response-signature targets. | No analytic random-direction coverage theorem. Once chosen, slices are deterministic Lean functions; probability is only transported from an assumed good selected-slice event. | `finite_sliced_zeroLoss_implies_contextual_sufficiency`, `finite_sliced_within_implies_contextual_sufficiency_within`, coordinate/full-slice cover, left-invertible slice-cover theorems, and `random_finite_sliced_contextual_sufficiency_failure_prob_le`. |
| Dirmeier/Albert/Perez-Cruz 2025 SSNL/SNLE | Framing of `g` as the lower-dimensional state used by downstream likelihood/readout. | No dominated-measure/continuous Bayes theorem for learned states, density/Jacobian semantics, posterior calibration/SNL consistency theorem, or package-level estimator semantics. | `LikelihoodOnStateSufficiency.lean`: `likelihood_on_state_family_sufficient`, state-decoder lifting, and approximate likelihood-readout sufficiency. `SurjectiveLikelihoodOnState.lean`: set-theoretic surjective state factorization. `PosteriorOnStateSufficiency.lean`: deterministic posterior/readout-on-state sufficiency and surjective posterior factorization. `FiniteBayesOnState.lean`: finite/discrete Bayes normalization, MAP/odds invariance, posterior expectations/readouts, posterior predictive readouts, Bayes risks/actions, credible/acceptance-set masses, evidence-ratio target-posterior algebra, likelihood-to-posterior transport, and state factorization. `MathlibBayesBridge.lean`: aliases mathlib conditional Bayes, conditional expectation, finite-fiber total probability, kernel posterior/density/RN APIs, and dominated PDF/RN APIs; packages finite posteriors as `PMF`s; proves set-mass formulas and raw/state PMF equality for likelihood-on-state families; and identifies local consistency with `TendstoInMeasure`. `PosteriorConsistency.lean`: assumption-backed posterior consistency/concentration transport plus evidence-ratio sufficient-condition bundles. |
| Makinen et al. 2024 Hybrid Summary Statistics | Augment hand-built summaries with neural summaries and target information beyond the existing summary. | No measure-theoretic MI, estimator guarantee, strong posterior consistency theorem, or cosmology-performance theorem. | `HybridInformationObjectives.lean`: symbolic MI chain-rule and loss/proxy optimizer equivalences under assumptions. `HybridSummarySufficiency.lean`: product sufficiency, within-base separation, collision-impossibility, and approximate readout bridges. Markov/LDA hybrid anchors are theorem-backed. |
| `sbijax` | Maintained NASS/NASSS package surface and provenance-recorded JAX learner. | No formal semantics for package internals. | Abstract theorem conditions over selected states, queries, slices, likelihood-free probes, and approximate metric slack. |
| `surjectors` | Design reference for possible future density-on-state diagnostics. | No direct training dependency, package semantics, flow-density theorem, or Jacobian theorem. | Set-theoretic surjective state factorization: `surjective_state_likelihood_factorization`, `surjective_state_likelihood_sufficient_iff_factors`, and `surjective_state_likelihood_readout_within`. |

## Pieces To Borrow

### NASS, Chen et al. 2021

Sources:

- Paper: https://arxiv.org/abs/2010.10079
- Code: https://github.com/cyz-ai/neural-approx-ss-lfi
- Modern JAX implementation: `sbijax.NASS`

Useful pattern: learn a summary network with an infomax/dependence objective
against an oracle-side variable. The original code exposes a practical menu:
JSD/DeepInfoMax-style negatives, distance correlation, DV/MINE, and Wasserstein
dependency. We now mirror that menu in the probe as contextual dependence
objectives over `(z_x, R_K(x))`.

Related BibTeX objective keys:

- `BelghaziEtAl2018MINE` for `dv`.
- `HjelmEtAl2019DeepInfoMax` for `jsd`.
- `OordEtAl2018CPCInfoNCE` for `infonce`.
- `SzekelyRizzoBakirov2007DistanceCorrelation` for `dcorr`.
- `OzairEtAl2019WassersteinDependency` for `wasserstein`.
- `PooleEtAl2019VariationalMIBounds` and
  `SongErmon2020UnderstandingMIEstimators` for the high-dimensional MI
  variance/bias caution.

Machine-checked status: `DependenceObjectiveProxies.lean` gives a symbolic
objective layer for this menu. `dv`, `jsd`, and `infonce` are represented as
order-reversing loss/proxy bridges; `dcorr` and `wasserstein` are represented
as direct proxy maximization aliases. The checked bridge to target information
requires a uniform proxy-error assumption:
`uniform_proxy_error_argmax_implies_information_epsilon_argmax`. The checked
caution is `lowerBoundProxy_alone_counterexample`: a lower-bound relation by
itself does not justify proxy-argmax implies information-argmax.

### NASSS / SSS, Chen, Gutmann, Weller 2023

Sources:

- Paper: https://proceedings.mlr.press/v202/chen23h.html
- Code path: `sbijax.NASSS`

Useful pattern: do not estimate high-dimensional MI directly. Sample random
unit-sphere directions and learn low-dimensional sliced targets. In this repo,
the training code may sample directions, but the Lean lane treats the resulting
selected slice functions deterministically. Our equivalent is:

```text
phi_j ~ sphere(K)
y_j = phi_j^T R_K(x)
predict y_j from z_x
```

This is now available in the probe via:

```bash
--infomax-loss-weight <w> \
--contextual-dependence-objective regression \
--response-signature-contexts K \
--response-signature-slices M
```

The formal counterpart is
`FormalProofs.OPT.SlicedContextualSufficiency`: if representation collisions
preserve selected slice values and those selected slices cover the full
response-signature fibers, then `QuerySufficient` follows. The approximate
version replaces exact equality with slice slack `δ` and contextual-response
slack `ε`. Concrete cover assumptions can now be discharged by full coordinate
slices (`finite_coordinate_slices_univ_cover_response_fibers`) or by any
deterministic slice map with a left inverse
(`left_invertible_slices_cover_response_fibers`).

`FormalProofs.OPT.RandomSlicedContextualSufficiency` adds the bounded
probability wrapper. A seed `omega` chooses a representation, finite selected
slice set, and slice family. If the good-seed event fails with probability at
most `eta`, the contextual-sufficiency failure event also has probability at
most `eta`. This transports an assumed coverage event; it does not prove the
analytic probability that random directions cover a task-specific response
fiber.

### SSNL, Dirmeier, Albert, Perez-Cruz 2025

Sources:

- UAI/PMLR: https://proceedings.mlr.press/v286/dirmeier25a.html
- Local PDF:
  `docs/literature/contextual_sufficiency/dirmeier_albert_perezcruz_2025_ssnl.pdf`
- Software: https://github.com/dirmeier/sbijax
- Surjective-flow layers: https://github.com/dirmeier/surjectors

Useful pattern: the representation is not merely an auxiliary summary. It is
the lower-dimensional variable on which the downstream likelihood/readout is
trained. For us this supports the paper framing that `g` is the learned state
map and `f` is the readout over that state. The deterministic Lean counterpart
is:

```text
likelihood(theta, x) = ell_theta(z_x)
=> z_x is LikelihoodFamilySufficient for that likelihood family
```

The theorem names are `likelihood_on_state_family_sufficient`,
`rep_with_state_readout_likelihood_on_state_family_sufficient`, and
`likelihood_readout_within_implies_likelihood_family_sufficient_within`.
`SurjectiveLikelihoodOnState.lean` adds the set-theoretic converse: if a
surjective state map has likelihood values constant on state fibers, then the
likelihood factors through a state-space likelihood head
(`surjective_state_likelihood_factorization`), with an approximate readout
companion (`surjective_state_likelihood_readout_within`). We still do not
formalize density/Jacobian semantics, posterior calibration, or SSNL estimator
consistency.

For posterior/readout workflows, `PosteriorOnStateSufficiency.lean` now captures
the deterministic part:

```text
posterior(x) = post(z_x)
=> z_x is PosteriorSufficient for that posterior-like object
```

It also exports `likelihood_sufficient_implies_posterior_sufficient` under an
explicit "posterior determined by likelihood" assumption, plus
`surjective_state_posterior_factorization` and
`posterior_readout_within_implies_posterior_sufficient_within`. These are
readout/fiber theorems, not coverage, calibration, or strong posterior
consistency theorems.

For Bayes semantics, `FiniteBayesOnState.lean` now covers the finite/discrete
case: a fixed prior and finite parameter type give a normalized
`BayesPosterior`, likelihood-family sufficiency transports to finite-Bayes
posterior sufficiency, positive evidence normalization preserves MAP
decisions, posterior odds cancel the evidence, finite posterior
expectations/readouts and posterior predictive readouts factor through state
likelihoods, and state likelihoods induce finite-Bayes posterior-on-state
readouts. `MathlibBayesBridge.lean` aligns this bounded
layer with mathlib: conditional-probability Bayes, finite-fiber total
probability, kernel posterior/density/RN, and dominated PDF/RN theorem aliases
are public; positive finite posteriors can be packaged as mathlib `PMF`s and
induced measures with singleton and arbitrary-event masses; likelihood-on-state
families have equal raw/state Bayes PMFs; and local posterior consistency is
definitionally mathlib `TendstoInMeasure`. This is intentionally bounded for
repo-specific
learned states: continuous or dominated-measure Bayes theorems,
density/Jacobian semantics, MCMC/VB semantics, posterior calibration, and
estimator consistency remain outside scope.

For posterior consistency, `PosteriorConsistency.lean` now covers the V1
framework: posterior consistency is convergence in probability, finite
posterior concentration is mass on the target parameter tending to one, and
both are transported across pointwise-equal posterior sequences, finite
likelihood-on-state Bayes posteriors, and exact state readouts. This is an
assumption-backed transport layer. The finite likelihood-ratio route is now
captured as evidence-ratio algebra plus `finite_bayes_likelihood_ratio_consistency_condition`
and `finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition`,
with the analytic posterior-transform convergence kept explicit. It does not
prove Schwartz-style consistency, dominated-measure Bayes, estimator
consistency, calibration, coverage, or SSNL/SNLE convergence.

### Hybrid Summary Statistics, 2024

Source:

- Paper: https://arxiv.org/abs/2410.07548
- Local PDF:
  `docs/literature/contextual_sufficiency/makinen_et_al_2024_hybrid_summary_statistics.pdf`

Useful pattern: learn a neural summary `s(d)` to add information beyond a fixed
summary `t(d)`. The paper writes this target as conditional MI
`I(s; theta | t) = I([s, t]; theta) - I(t; theta)` and studies EPE/CE training
objectives for the learned summary. We now formalize the algebraic part
symbolically: `HybridMIChainRule` states the chain rule as an assumption, and
theorems prove that conditional-MI maximizers are joint-MI maximizers up to the
fixed base term. EPE and classifier/JSD losses get optimizer-equivalence
theorems only when the loss is supplied as a negated or order-reversing
information proxy.

For our Markov witness, exact `(count, first, last)` probes remain diagnostics,
not hard-coded learned-state slots. The contextual `sbijax` CLI can now add
Makinen-style finite-response diagnostics with
`--include-hybrid-diagnostics`, reporting base-only, neural-only, and hybrid
collision rates against fixed response signatures.

The deterministic Lean counterpart is product-summary sufficiency:

```text
hybrid(x) = (base(x), neural(x))
```

Theorems show that the hybrid product refines both components, sufficiency of
either component lifts to the hybrid, and target/likelihood/likelihood-free
readouts from the hybrid imply the corresponding sufficiency condition. The
stronger Makinen-facing theorem is within-base: product sufficiency is exactly
the statement that, after fixing `base(x) = base(y)`, equality of the neural
summary preserves the target/likelihood/probe response. Its contrapositive says
the neural summary separates distinctions left unresolved inside base-summary
fibers. Key public aliases include `hybrid_summary`,
`hybrid_cmi_argmax_iff_joint_mi_argmax`,
`hybrid_epe_loss_min_iff_information_max`,
`hybrid_target_sufficient_iff_within_base_target_sufficient`,
`hybrid_likelihood_sufficient_neural_separates_likelihood_within_base`,
`hybrid_collision_impossible_of_distinguished_likelihood`,
`hybrid_likelihood_readout_implies_likelihood_sufficient`, and
`hybrid_likelihood_on_state_family_sufficient`.

Concrete theorem anchors:

- `markov_count_endpoint_hybrid_two_sided_sufficient`: `(count-only, endpoint
  residual)` is sufficient for Markov two-sided changepoint-count queries.
- `lda_bow_hybrid_likelihood_sufficient`: any `(bagOfWords, neural)` hybrid is
  sufficient for the ordinary bag-of-words LDA likelihood family.
- `lda_bow_hybrid_neural_separates_response_within_bagOfWords`: order or
  contextual probe distinctions inside a bag-of-words fiber must be carried by
  the neural component if the hybrid is response-sufficient.

## Probe Objective Menu

The JAX/sbijax lane is the package-disciplined path:

```bash
python scripts/probe_contextual_sbijax.py \
  --training-objective contextual_sufficiency \
  --sbijax-method nasss \
  --context-samples-per-doc 2 \
  --response-signature-contexts 8 \
  --response-signature-slices 4
```

Its run output records `backend_package: sbijax`, the installed `sbijax`,
`jax`, `jaxlib`, and `surjectors` versions, the selected method (`nass` or
`nasss`), and the response-signature dimensions.

The PyTorch comparison probe, `scripts/probe_clean_unified_no.py`, supports:

```text
--contextual-dependence-objective infonce|regression|dcorr|jsd|dv|wasserstein|none
```

Mapping:

- `regression`: sliced contextual-response prediction, closest to 2023 SSS.
- `dcorr`: critic-free dependence objective, closest to the stable NASS
  alternative; Lean alias `distance_correlation_proxy_max`.
- `jsd`: DeepInfoMax/NASS shuffled-negative objective; Lean alias
  `deep_infomax_jsd_loss_min_iff_proxy_max`.
- `dv`: MINE-style Donsker-Varadhan objective; useful but not the default;
  Lean alias `mine_dv_loss_min_iff_proxy_max`.
- `wasserstein`: Wasserstein dependency-style objective; Lean alias
  `wasserstein_dependency_proxy_max`.
- `infonce`: contrastive response-signature matching; Lean alias
  `infonce_loss_min_iff_proxy_max`.

The CLI keeps `--infomax-loss-weight` as the compatibility weight, but the loss
is now better understood as a contextual dependence loss.

These aliases are theorem-backed only at the symbolic optimization layer. They
do not prove Shannon MI identities, DV/JSD/InfoNCE estimator correctness,
negative-sampling consistency, distance-correlation independence, or
optimal-transport duality.

## Recommended Experiment Order

1. Start with contextual MSE only:
   `--training-objective contextual_sufficiency --infomax-loss-weight 0`.
2. Add sliced regression:
   `--contextual-dependence-objective regression --response-signature-contexts 8 --response-signature-slices 4`.
3. Compare `dcorr` at the same context count.
4. Only then compare `jsd`, `infonce`, `wasserstein`, and `dv`.
5. Report Markov sketch decoders as diagnostics: first/last/count accuracy,
   contextual MAE, collision rate, prediction std/correlation, and boundary
   precision/recall/F1.

## Paper Framing

The front-page claim should be:

```text
g learns a contextual sufficient state map.
f is a readout on that state.
The local laws and Lean layer specify the fiber condition.
The NASS/NASSS/SSNL literature supplies modern learning objectives for finding
such state maps without hard-coding Markov-specific slots.
Hybrid summary statistics justify reporting exact sketches as diagnostics or
auxiliary product summaries without redefining the learned state.
```

The Markov `(count, first, last)` sketch remains the validation witness: it
proves the contextual-sufficiency target is real and recoverable in at least one
controlled setting, but it is not the general definition of `g`.

The slice bridge sharpens the SSS/NASSS claim for our setting: random directions
are a learning device, while the deterministic theorem is about the finite
selected slice set actually used by a run. The event-level probability wrapper
then says that any externally justified good selected-slice event transfers to
a contextual-sufficiency failure-probability bound.

The Markov Lean witness now has approximate companions for real-valued count
queries: `exact_markov_sketch_twoSided_context_sufficient_within_zero_real`,
`markov_composed_readout_within_implies_twoSided_context_sufficient_within_real`,
and `markov_finite_sliced_within_implies_count_query_sufficient_within`.
