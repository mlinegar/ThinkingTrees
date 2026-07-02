# Core Proofs (Paper ↔ Lean)

This document is meant to let an *inexperienced* reader do two things:

1. Reconstruct the core arguments **by hand** (pen-and-paper proof skeletons).
2. Jump from each proof step to the **corresponding Lean lemma/definition**.

The Lean code is the source of truth for formal statements; this file is the “guided tour”.

---

## 0. Quick map: main paper results → Lean theorems

The active paper's main theorem statements are in
`paper/ctreepo/sections/v3/08_theory.tex`; appendix extensions are in
`paper/ctreepo/appendix/`.

| Paper result | Paper label | Lean theorem | Lean file |
|---|---:|---|---|
| Inductive Preservation | `thm:one-pass` | `one_pass` | `../FormalProofs/OPT/PreservationTheorems.lean` |
| Schedule invariance | `cor:schedule` | `schedule_invariance` | `../FormalProofs/OPT/PreservationTheorems.lean` |
| Fold-of-folds invariance | `cor:folds` | `fold_of_folds` | `../FormalProofs/OPT/PreservationTheorems.lean` |
| Multi-round preservation | `thm:multi-round` | `multi_round_proper` | `../FormalProofs/OPT/ExpectationTheory.lean` |
| Neural-operator-to-gap bridge | quantitative route into `thm:unified-gap` | `neural_operator_transfer_local_law_budget`, `neural_operator_transfer_method_gap_budget`, `neural_operator_delta_r_transfer_moduli_bound`, `expectedObjectiveGap_via_neuralOperatorTransferModuli`, finite-dimensionalization analogues, and method-specific neural-operator gap wrappers | `../FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean`, `../FormalProofs/OPT/MainTheorems.lean` |
| Preference-objective equivalence | `thm:pref-equiv` | `preference_learning_equivalence`, `same_oracle_measurable_argmin_general_of_loss_eq`, plus `dpo_exact_metric`, `grpo_pl_exact_metric`, `grpo_rl_exact_metric`, `dpo_equivalence`, `grpo_equivalence`, `grpo_rl_equivalence` | `../FormalProofs/OPT/PreferenceLearning.lean`, `../FormalProofs/OPT/PreferenceBounds.lean` |
| Unified preference gap | `thm:unified-gap` | `unified_preference_gap_bounded` | `../FormalProofs/OPT/PreferenceBounds.lean` |
| C2 independence counterexample | `ex:c2-independent` | `ex_c2_independent_formalized`, `thm10_1_L3_not_derivable` | `../FormalProofs/OPT/CounterexampleExistence.lean` |
| Scene-register changepoint sketch | `sec:markov-interlude` | `MarkovCountSketch`, `exactSketch_root_distortion_zero`, `not_L3_gFlip` | `../FormalProofs/OPT/MarkovCountSketchExample.lean` |
| Bayes / posterior-state formal layer | information appendix / SBI scope | `mathlib_conditional_bayes_rule`, `mathlib_kernel_posterior`, `finite_bayes_posterior_map_iff_numerator_map`, `finite_bayes_posterior_expectation_likelihood_on_state_sufficient`, `finite_bayes_posterior_predictive_likelihood_on_state_sufficient_observed`, `finite_bayes_posterior_pmf`, `posterior_consistent_iff_mathlib_tendsto_in_measure`, `finite_bayes_consistency_likelihood_on_state_iff` | `../FormalProofs/OPT/MathlibBayesBridge.lean`, `../FormalProofs/OPT/FiniteBayesOnState.lean`, `../FormalProofs/OPT/PosteriorConsistency.lean` |
| Bayesian persuasion finite information design | Kamenica-Gentzkow AER 2011 crosswalk | `bayesian_persuasion_posterior_eq_finite_bayes`, `bayesian_persuasion_valid_signal_bayes_plausible`, `bayesian_persuasion_valid_signal_scheme_feasible`, `bayesian_persuasion_splitting_experiment_valid`, `bayesian_persuasion_splitting_posterior_eq`, `bayesian_persuasion_receiver_bayes_action_iff_best_response`, `bayesian_persuasion_concavification_iff_optimal_value`, `bayesian_persuasion_receiver_loss_factors_through_belief`, `bayesian_persuasion_sender_indirect_value_factors_through_belief`, `bayesian_persuasion_signal_indirect_value_eq_scheme_value`, `bayesian_persuasion_receiver_obedient_iff_bayes_action`, `bayesian_persuasion_indirect_value_eq_of_same_posterior_distribution`, `bayesian_persuasion_direct_recommendation_from_experiment_valid`, `bayesian_persuasion_direct_recommendation_ex_ante_sender_value_eq`, `bayesian_persuasion_direct_recommendation_sender_value_eq` | `../FormalProofs/OPT/BayesianPersuasion.lean`, `../FormalProofs/OPT/BayesianPersuasionEconomics.lean`, `../FormalProofs/OPT/BayesianPersuasionDirect.lean` |

For a curated Lean entry point that re-exports the key results with documentation, start from:
`../FormalProofs/OPT/MainTheorems.lean`.

---

## 1. Notation map (paper → Lean)

All core objects are defined in `../FormalProofs/OPT/CoreDefinitions.lean` and
`../FormalProofs/OPT/LocalLaws.lean`.

| Concept | Paper notation | Lean identifier |
|---|---|---|
| Document space (monoid) | `Strings` with `concat` | `Strings` with `[Monoid Strings]` |
| Oracle space (pseudo-metric) | `(Y, d_Y)` | `Y` with `[PseudoMetricSpace Y]` |
| Oracle function | `f* : Strings → Y` | `fstar : Strings → Y` |
| Summarizer (randomized) | `g(x)` is a distribution on strings | `Summarizer Strings := Strings → PMF Strings` |
| Distortion | `d_Y(f*(z), f*(x))` | `D fstar z x := dist (fstar z) (fstar x)` |
| One-step expectation | `E_{z~g(x)}[·]` | `Eg g f x` |
| Tree reduction | reduce a merge tree bottom-up | `reduce g T : PMF Strings` |
| Multi-round reduction | `Z^(R)` | `ZR g x R T : PMF Strings` |
| Realized string of a tree | product of leaves | `S T` |
| Internal-node expectation | expectation under `reduce` | `Egu g T f` |

### Local laws: paper C1/C2/C3 vs Lean L1/L2/L3

The Lean names are in `../FormalProofs/OPT/LocalLaws.lean`:

- Paper **C1 (Sufficiency)** = Lean **L1**: leaf summaries preserve the oracle (in expectation).
- Paper **C3 (Merge consistency)** = Lean **L2**: internal-node merges preserve the oracle (in expectation).
- Paper **C2 (Idempotence / on-range stability)** = Lean **L3**: re-summarizing an on-range string is oracle-preserving (in expectation).

The slightly “scrambled” numbering (C2 ↔ L3) is intentional: in Lean, L1/L2 are the
tree-local laws, and L3 is the global “on-range” law.

### Mergeable-Sketch Reduction

The paper’s mergeable-reduction proposition has two Lean branches:

- Strict oracle-output branch: `A3_global` supplies a merge
  `Y → Y → Y`, used by `prop3_mergeable_classical` and
  `ops_reduction_to_classical_mergeable`.
- Classical state-level branch: sketch states merge before readout, captured by
  `sketch_state_level_reduction_to_classical_mergeable` and
  `sketch_state_mergeClosed_of_compatible`.

The negative control `scalarDistinctCount_not_child_cardinality_mergeable`
records why scalar distinct counts are not enough: child cardinalities do not
carry overlap information, while HLL-style register states do.

The literature-specific formal layer is reusable in
`../../FormalProbability/FormalProbability/ML/MergeableSummaries.lean` and
is organized chronologically in
`../../FormalProbability/FormalProbability/ML/MergeableSummaries/LiteratureChronology.lean`.
The standalone import surface for all separately formalized pieces is
`../../FormalProbability/FormalProbability/ML/MergeableSummaries/Literature.lean`.
It is re-exported through `../FormalProofs/OPT/MergeableReduction.lean`.
For the full proof-needs inventory, see
`MERGEABLE_SKETCH_LITERATURE_PROOF_MAP.md`; for the dedicated Gibbons theorem
inventory, see `GIBBONS1996_PROOF_MAP.md`; for the dedicated Gray/Data Cube
inventory, see `GRAY1997_PROOF_MAP.md`.

- Gibbons: `Gibbons1996.theorem_4_1_third_homomorphism`,
  `Gibbons1996.lemma_4_3_homomorphic_iff_kernel_congruent`,
  `Gibbons1996.longestSortedPrefixNat_not_homomorphic`,
  `Gibbons1996.insertionSort_inefficientMerge_homomorphic`,
  `Gibbons1996.mergeSort_eq_insertionSort`,
  `Gibbons1996.Section5RuntimeClaims`,
  `ctreepo_gibbons1996_linearGrowth_linearTime`,
  `ctreepo_gibbons1996_quadraticGrowth_quadraticTime`,
  `ctreepo_gibbons1996_nLogNGrowth_nLogNTime`,
  `ctreepo_gibbons1996_referenceSection5RuntimeClaims`,
  `ctreepo_gibbons1996_third_homomorphism`, and
  `ctreepo_gibbons1996_ordered_schedule_invariance`.
- Gray et al.: `DistributiveAggregate`, `AlgebraicAggregate`, and
  `ctreepo_gray1997_state_level_summary_is_algebraic`,
  `ctreepo_gray1997_cubeMask_card`,
  `ctreepo_gray1997_superAggregateMask_card`,
  `ctreepo_gray1997_cubeAddressD_card`,
  `ctreepo_gray1997_rollupSuperLevel_card`,
  `ctreepo_gray1997_rollupPrefixMask_injective`,
  `ctreepo_gray1997_directRollupUpdates_le_directCubeUpdates`,
  `ctreepo_gray1997_directRollupSuperAggregateUpdates_le_directCubeSuperAggregateUpdates`,
  `ctreepo_gray1997_average_not_distributive_oracle`,
  `ctreepo_gray1997_max_no_scalar_delete_front`, and
  `ctreepo_gray1997_modeBool_state_card_lower_bound`;
  the stronger `ctreepo_gray1997_modeBool_no_finite_state_homomorphic_realization`
  packages the lower bound into a no-finite-state corollary.  The same
  contextual lower-bound pattern is also exposed for the Boolean
  median/majority aggregate through
  `ctreepo_gray1997_medianMajorityBool_no_finite_state_homomorphic_realization`.
- Feldman et al.: `MUDAggregator`, `feldman2006_03_mud_build_permutation_invariant`,
  `Feldman2008.ComputationTree.evalState_eq_build_data`,
  `Feldman2008.StreamingAlgorithm.lemma1_streaming_state_congruence_append`,
  `Feldman2008.StreamingAlgorithm.lemma2_representative_merge_exists`,
  `Feldman2008.deterministic_streaming_to_representative_mud`,
  `Feldman2008.PolylogRate.square`,
  `Feldman2008.polylog_streaming_subset_general_mud`,
  `Feldman2008.theorem2_streaming_to_scm_semantic`,
  `Feldman2008.not_polylog_streaming_of_scm_lower_bound`,
  `Feldman2008.boolVectorEquality_messageA_card_lower`,
  `Feldman2008.boolVectorEquality_bitsA_lower`,
  `Feldman2008.BitAccountedEqualityProtocolFamily.linear_bigO_lower`,
  `Feldman2008.finSetParity_two_vectors_eq`,
  `Feldman2008.finSetParity_scm_lower_bound_of_equality`,
  `Feldman2008.finSetParity_bitAccounted_bitsA_lower`,
  `Feldman2008.BitAccountedFinSetParitySCMFamily.linear_bigO_lower`,
  `Feldman2008.privateCoinEqualityProtocolFromFinSetParity_successCount`,
  `Feldman2008.privateCoinEqualityProtocolFromFinSetParity_computesWithSuccess`,
  `Feldman2008.privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality`,
  `Feldman2008.finiteSetParity_scm_sqrt_lower_bound_of_equality`,
  `Feldman2008.representativeMUDFromStreaming_computesOnAllTrees`,
  `Feldman2008.PublicRandomStreamingFamily.successSet_eq_univ_of_computesSeedwise`,
  `Feldman2008.PublicRandomStreamingFamily.successProbability_eq_one_of_computesSeedwise`,
  `Feldman2008.public_randomness_seedwise_general_mud`,
  `Feldman2008.setParity_symmetric`,
  `Feldman2008.setParity_scm_sqrt_lower_bound_statement`,
  `Feldman2008.symmetricIndexCanonical_mem_domain`,
  `Feldman2008.symmetricIndexCanonical_readout_eq`,
  `Feldman2008.symmetricIndex_promise_symmetric`,
  `Feldman2008.symmetricIndex_scm_linear_lower_bound_statement`,
  `Feldman2008.mud_polylog_subset_streaming`,
  `ctreepo_feldman2008_public_randomness_seedwise_extension_statement`,
  `ctreepo_feldman2008_representativeMUDFromStreaming_computesOnAllTrees`,
  `ctreepo_feldman2008_publicRandom_successSet_eq_univ_of_computesSeedwise`,
  `ctreepo_feldman2008_publicRandom_successProbability_eq_one_of_computesSeedwise`,
  `ctreepo_feldman2008_public_randomness_seedwise_general_mud`,
  `ctreepo_feldman2008_theorem3_private_randomness_separation_statement`,
  `ctreepo_feldman2008_mud_state_level_mergeable`,
  `ctreepo_feldman2008_item_tree_state_eq_build`, and
  `ctreepo_feldman2008_theorem1_deterministic_streaming_to_mud_semantic`.
- Flajolet et al.: `HLLRegisters`,
  `ctreepo_flajolet2007_hll_merge_idempotent`,
  `ctreepo_flajolet2007_hll_state_level_mergeable`,
  `ctreepo_flajolet2007_idealHash_state_level_mergeable`,
  `ctreepo_flajolet2007_randomIdealHash_seedFamily_hierarchical`,
  `ctreepo_flajolet2007_bitsToNat_lt_two_pow_length`,
  `ctreepo_flajolet2007_hashWord_rank_positive`,
  `ctreepo_flajolet2007_hll_buildFromHashes_append`,
  `ctreepo_flajolet2007_hll_rawEstimator_empty`,
  `ctreepo_flajolet2007_hll_relativeStandardError_registerCount`, and
  `ctreepo_flajolet2007_hll_rse_p14_under_one_percent`, and
  `ctreepo_flajolet2007_hll_stochasticEstimatorClaims`,
  `ctreepo_flajolet2007_hll_stochasticEstimatorBigOClaims`,
  `ctreepo_flajolet2007_PoissonizedBySeries`,
  `ctreepo_flajolet2007_PoissonizationDepoissonizationAnalysis`,
  `ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poisson_depoissonization`,
  `ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poissonization_analysis`,
  and `ctreepo_flajolet2007_relativeStandardErrorBigO_of_asymptotic`.
- Agarwal et al.: `StateLevelMergeableSummary`,
  `StateLevelMergeableSummary.QueryCorrect`,
  `ctreepo_agarwal2013_state_level_hierarchical_readout`,
  `ctreepo_agarwal2013_incrementally_maintainable_one_way`,
  `ctreepo_agarwal2013_countMin_state_level_mergeable`,
  `ctreepo_agarwal2013_countMin_merge_not_idempotent_of_pos`,
  `ctreepo_agarwal2013_misraGries_hierarchical`,
  `ctreepo_agarwal2013_executableMisraGries_totalCounterMass_le_length`,
  `ctreepo_agarwal2013_executableMisraGries_positiveCounts`,
  `ctreepo_agarwal2013_executableMisraGries_tracedPotential_le_length`,
  `ctreepo_agarwal2013_executableMisraGries_debt_mul_succ_le_length`,
  `ctreepo_agarwal2013_spaceSaving_hierarchical_of_isomorphism`,
  `ctreepo_agarwal2013_executableSpaceSaving_boundedBy`,
  `ctreepo_agarwal2013_executableSpaceSaving_totalCounterMass_le_length`, and
  `ctreepo_agarwal2013_gk_corollary2_oneWay`,
  `ctreepo_agarwal2013_SameWeightIntervalApproximationSpec`,
  `ctreepo_agarwal2013_stateLevelEpsilonApproximation_tree_error`,
  `ctreepo_agarwal2013_exactStateLevelEpsilonApproximation_tree_error`,
  `ctreepo_agarwal2013_finiteRangeSpace_shattered_card_le_vcDim`,
  `ctreepo_agarwal2013_finiteRangeSpace_trace_card_le_sauerShelah`,
  `ctreepo_agarwal2013_finiteRangeSpace_traceFailureEvent_le_sauerShelah_mul`,
  `ctreepo_agarwal2013_exactRangeSpaceSizedMergeableQuerySketch`,
  `ctreepo_agarwal2013_sameWeightInterval_tree_error_on_equalLength`,
  `ctreepo_agarwal2013_sameWeightHalving_unbiased_interval_count`,
  `ctreepo_agarwal2013_sameWeightHalving_interval_error_abs_le_one`,
  `ctreepo_agarwal2013_sameWeightHalving_hoeffdingDenominator_le`,
  `ctreepo_agarwal2013_sameWeightHalving_root_error_to_epsilon_n_of_scale`,
  `ctreepo_agarwal2013_sameWeightHalving_completeTree_hoeffding_tail`,
  `ctreepo_agarwal2013_sameWeightHalving_completeTree_epsilon_n_tail`,
  `ctreepo_agarwal2013_EpsilonKernelSpec`,
  `ctreepo_agarwal2013_epsilonKernel_hierarchical`,
  `ctreepo_agarwal2013_epsilonKernel_tree_widthError`,
  `ctreepo_agarwal2013_directionalWidth_append`,
  `ctreepo_agarwal2013_directionalWidth_translateStream`,
  `ctreepo_agarwal2013_directionalWidth_scaleStream_of_nonneg`,
  `ctreepo_agarwal2013_exactEpsilonKernel_tree_widthError`, and
  `ctreepo_agarwal2013_hybridTrace_level_monotone`,
  `ctreepo_agarwal2013_hybridRandomBuffer_failure_bound`, and
  `ctreepo_agarwal2013_hybridRandomBuffer_failure_bound_uniform`.
- Executable quantile witnesses: `ctreepo_gk2001_executable_build_n`,
  `ctreepo_gk2001_executable_build_gapMassValid`,
  `ctreepo_kll2016_theorem4_mergeable_variant_of_algorithm`,
  `ctreepo_kll2016_theorem5_optimal_variant_of_algorithm`,
  `ctreepo_kll2016_executable_weightedCount_step`, and
  `ctreepo_kll2016_executable_build_massValid`.

### Bayes and posterior-state layer

The paper's SBI-facing discussion now has a bounded Lean-backed Bayes layer. It
has three pieces:

1. **Mathlib Bayes APIs.** `MathlibBayesBridge.lean` re-exports the relevant
   mathlib probability surfaces under paper-facing names:
   `mathlib_conditional_bayes_rule` for event-level conditional Bayes,
   `mathlib_conditional_probability_condition_twice` and
   `mathlib_conditional_probability_finite_fiber_total` for conditional
   probability algebra,
   `mathlib_conditional_expectation`,
   `mathlib_integral_conditional_expectation`, and
   `mathlib_rn_deriv_ae_eq_conditional_expectation` for conditional
   expectation/Radon-Nikodym semantics, `mathlib_kernel_posterior` and
   `mathlib_kernel_posterior_compProd_eq_map_swap` for kernel/disintegration
   posterior semantics, and `mathlib_kernel_posterior_with_density_countable`
   plus `mathlib_kernel_posterior_eq_with_density` and
   `mathlib_kernel_posterior_rn_deriv` for density/Radon-Nikodym Bayes
   formulas. It also exposes the dominated-density layer through
   `mathlib_has_pdf`, `mathlib_pdf_map_eq_with_density`, and
   `mathlib_pdf_lintegral_lotus`.
2. **Finite Bayes on learned state.** `FiniteBayesOnState.lean` gives the
   discrete formula
   `prior theta * likelihood theta x / evidence x`, proves
   `finite_bayes_posterior_sum_eq_one`, MAP invariance under positive
   normalization (`finite_bayes_posterior_map_iff_numerator_map`), posterior
   odds cancellation (`finite_bayes_posterior_odds_eq_numerator_odds`), finite
   posterior expectations/readouts
   (`finite_bayes_posterior_expectation_likelihood_on_state_sufficient`),
   posterior predictive readouts
   (`finite_bayes_posterior_predictive_likelihood_on_state_sufficient_observed`),
   Bayes risk/action transport
   (`finite_bayes_posterior_risk_likelihood_on_state_eq_state`,
   `finite_bayes_action_likelihood_on_state_iff_state_action`),
   credible/acceptance-set transport
   (`finite_bayes_credible_at_level_likelihood_on_state_iff_state`), and
   evidence-ratio target-posterior algebra
   (`finite_bayes_posterior_target_eq_inv_one_plus_evidence_ratio_remainder`),
   transports likelihood sufficiency to finite-Bayes posterior sufficiency, and provides
   `finite_bayes_posterior_likelihood_on_state_sufficient` plus
   `surjective_state_finite_bayes_posterior_factorization`. The mathlib bridge
   also packages the raw and state posteriors as `PMF`s and proves
   `finite_bayes_posterior_pmf_to_measure_set`,
   `state_finite_bayes_posterior_pmf_to_measure_set`, and
   `finite_bayes_posterior_pmf_likelihood_on_state_eq_state_pmf`.
3. **Consistency transport.** `PosteriorConsistency.lean` records posterior
   consistency as convergence in probability. `MathlibBayesBridge.lean` proves
   `posterior_consistent_iff_mathlib_tendsto_in_measure` and
   `finite_posterior_mass_concentrates_at_iff_mathlib_tendsto_in_measure`, so
   the local predicates are explicitly mathlib `TendstoInMeasure` statements.
   Exact state/readout equalities then transport assumed consistency via
   `finite_bayes_consistency_likelihood_on_state_iff` and
   `state_readout_finite_bayes_consistency`. The finite likelihood-ratio route
   is represented by explicit evidence-ratio condition bundles, including
   `finite_bayes_likelihood_ratio_consistency_condition` and
   `finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition`.

What this does **not** prove: posterior calibration, MCMC/VB convergence,
SSNL density/Jacobian semantics, estimator consistency, or a classical
Schwartz/Doob posterior consistency theorem for learned states. Those remain
named assumptions or future theorem targets.

---

## 2. Core proof chain (what implies what)

At a very high level, the proof flow is:

1. **Local laws** (`L1`, `L2`, `L3`)
2. ⇒ **preservation of oracle distortion** under tree reduction (`one_pass`, `multi_round_proper`)
3. ⇒ **zero distortion on support** (the nonnegative-expectation argument)
4. ⇒ **expected loss invariance** for oracle-measurable losses and oracle-indexed generators
5. ⇒ **training equivalence** for DPO / GRPO-PL / GRPO-RL
6. and (separately) ⇒ **quantitative gap bound** when distortion is nonzero (`unified_preference_gap_bounded`)

The rest of this document unpacks each arrow with a hand-proof skeleton and Lean anchors.

---

## 3. Inductive (one-pass) preservation

**Paper:** Theorem “Inductive Preservation” (`thm:one-pass`).

**Lean statement:** `one_pass` in `../FormalProofs/OPT/PreservationTheorems.lean`.

### What to prove by hand

Fix a merge tree `T` whose leaves multiply to the document `x = S T`.
Define the property for *any* subtree `u` of `T`:

> `P(u)`: the expected distortion at `u` is zero:  
> `E_{z ~ reduce(g,u)}[ d_Y(f*(z), f*(S u)) ] = 0`.

Then prove `P(u)` for every subtree `u` by structural induction.

### Proof skeleton

1. **Base case (leaf):** `u = leaf b`.
   - `reduce(g, leaf b) = g(b)`.
   - `S(leaf b) = b`.
   - `L1` is *exactly* the claim that `Eg g (fun z => D fstar z b) b = 0`.
2. **Inductive case (node):** `u = node u_L u_R`.
   - `L2` asserts that the expected distortion at each realized internal node is 0, i.e.
     `Egu g (node u_L u_R) (fun z => D fstar z (S (node u_L u_R))) = 0`.
   - This is precisely `P(u)` for internal nodes.
3. Apply the result to the root `u = root T` and rewrite `S (root T) = S T = x`.

### Lean anchors

- The induction is packaged as `nodewise_preservation` (subtree-by-subtree), then specialized to the root in `one_pass`:
  - `nodewise_preservation` in `../FormalProofs/OPT/PreservationTheorems.lean`
  - `one_pass` in `../FormalProofs/OPT/PreservationTheorems.lean`
- The corollaries in the paper are proved immediately from “both sides are 0”:
  - `schedule_invariance` in `../FormalProofs/OPT/PreservationTheorems.lean`
  - `fold_of_folds` in `../FormalProofs/OPT/PreservationTheorems.lean`

---

## 4. Multi-round preservation

**Paper:** Theorem “Multi-round preservation” (`thm:multi-round`).

**Lean statement:** `multi_round_proper` in `../FormalProofs/OPT/ExpectationTheory.lean`.

### What to prove by hand

Let `Z^(R)` be the random output after `R` summarization rounds:
`Z^(1) := reduce(g,T)` and `Z^(R+1) := g(Z^(R))`.

Goal: for all `R ≥ 1`,

> `E[ d_Y(f*(Z^(R)), f*(x)) ] = 0`.

### Proof skeleton (induction on R)

1. **Base (R = 1):** this is exactly the one-pass theorem applied to the root.
2. **Step:** assume `E[ d_Y(f*(Z^(R)), f*(x)) ] = 0`. Consider `Z^(R+1) = g(Z^(R))`.
   - Use the fact that `d_Y ≥ 0`. In measure-theoretic terms:
     `E[nonneg] = 0` forces the integrand to be 0 almost surely.
   - `Z^(R)` lies in the range/support of `g` (because it was created by reductions using `g`).
   - Apply **L3** (on-range idempotence): re-summarizing on-range strings preserves the oracle.
   - Conclude the next-round distortion remains 0 in expectation.

### Lean anchors (and a technical note)

- `multi_round_proper` is the “fully rigorous” statement:
  - `multi_round_proper` in `../FormalProofs/OPT/ExpectationTheory.lean`
- You will also see convenience wrappers:
  - `multi_round_bounded` and `multi_round_typeclass` in `../FormalProofs/OPT/ExpectationTheory.lean`
- **Why the Lean proof looks more technical than the paper:** Lean represents expectations over a `PMF` as `∑'` (a `tsum`), so it must prove *summability*. The `*_proper` versions add an explicit bound on distortion to keep everything axiom-free and summable.

---

## 5. From “expected distortion = 0” to “distortion = 0 on support”

Many downstream equivalence theorems want a hypothesis of the form:

> for all `z` in the support of the summarized distribution and `x` in the support of the original distribution,  
> `dist (fstar z) (fstar x) = 0`.

But preservation is often proved first as an *expectation* statement:
`E[D] = 0`.

### Hand lemma (discrete “E[X]=0 ⇒ X=0 a.s.”)

Let `p` be a discrete distribution and `h : α → ℝ` with `h ≥ 0`.
If `E_p[h] = 0`, then for every `a ∈ support(p)`, we must have `h(a) = 0`.

**Proof:** if some `a` has positive mass and `h(a) > 0`, then the expectation would be strictly positive.

### Lean anchors

This reasoning is done explicitly (by contradiction) in several “via ZR” theorems, e.g.:

- DPO: `dpo_gap_zero_of_local_laws_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`
- GRPO-PL: `grpo_equivalence_via_ZR` in `../FormalProofs/OPT/PreferenceLearning.lean`
- GRPO-RL: `grpo_rl_equivalence_via_ZR` in `../FormalProofs/OPT/PreferenceLearning.lean`

When reading those proofs, look for the pattern:

1. assume `dist (fstar z) (fstar x) ≠ 0` for a support point `z`,
2. show it implies the `tsum` defining `Exp` is `> 0`,
3. contradict the previously established `Exp (...) = 0`.

---

## 6. Zero distortion ⇒ expected loss invariance (method-agnostic)

This is the core “bridge” between preservation and learning objectives.

### 6.1 Pairwise (generic) version

**Lean:** `expected_loss_eq_of_zero_dist_generic` in `../FormalProofs/OPT/PreferenceLearning.lean`.

**Hypotheses (what you assume by hand):**

1. `h_zero`: all oracle distances between `μ_Z`-support and `μ_X`-support points are 0,
2. `loss` is **oracle-measurable**: `dist(f*(x),f*(x'))=0 → loss x a = loss x' a`,
3. `gen` is **oracle-indexed**: `dist(f*(x),f*(x'))=0 → gen x = gen x'`.

**Conclusion:**

> The expected loss computed under `μ_X` equals the expected loss computed under `μ_Z`.

### Hand proof skeleton

1. Pick a reference point `x₀ ∈ support(μ_X)` (possible because `PMF.support_nonempty`).
2. Show every `x ∈ support(μ_X)` has `dist(f*(x), f*(x₀)) = 0`.
   - Use a fixed `z₀ ∈ support(μ_Z)` plus triangle inequality and the hypothesis `h_zero`.
3. By oracle-indexedness, `gen x = gen x₀` for all `x` in support.
4. By oracle-measurability, `loss x a = loss x₀ a` for all `x` in support and all `a`.
5. Therefore the inner expectation `E_{a~gen x}[loss x a]` is constant over support,
   so the outer expectation is that constant.
6. Repeat the same argument for `μ_Z` and conclude both expectations are equal.

### 6.2 Groupwise version (k-wise losses)

GRPO-style objectives are groupwise. The same idea appears as:

- `expected_group_loss_eq_of_zero_dist` in `../FormalProofs/OPT/PreferenceLearning.lean`

---

## 7. Instantiations: DPO, GRPO-PL, GRPO-RL

The active paper states these as method instantiations of
`thm:pref-equiv`: prove oracle-measurability / oracle-indexedness, then
apply the generic invariance lemma.

### DPO

- **Paper:** Preference-objective equivalence (`thm:pref-equiv`), DPO row.
- **Lean:** `dpo_equivalence` in `../FormalProofs/OPT/PreferenceBounds.lean`.

Proof idea by hand:

1. Local laws ⇒ `E[D(Z^(R), x)] = 0` (multi-round preservation).
2. Nonnegativity ⇒ `dist(f*(z), f*(x)) = 0` for all `z ∈ support(Z^(R))`.
3. DPO loss is oracle-measurable when `pol` and `pol_ref` are oracle-measurable.
4. The pair generator is oracle-indexed.
5. Apply the generic invariance lemma to conclude equality of expected DPO loss.

### GRPO-PL (Plackett–Luce)

- **Paper:** Preference-objective equivalence (`thm:pref-equiv`), GRPO-PL row.
- **Lean:** `grpo_equivalence` and the same-argmin export
  `grpo_pl_exact_metric` in `../FormalProofs/OPT/PreferenceLearning.lean`.

`grpo_equivalence_via_ZR` shows the same theorem specialized to the `ZR` distribution.

### GRPO-RL (clipping + KL; DeepSeek-R1 style)

- **Paper:** Preference-objective equivalence (`thm:pref-equiv`), GRPO-RL row.
- **Lean:** `grpo_rl_equivalence` and the same-argmin export
  `grpo_rl_exact_metric` in `../FormalProofs/OPT/PreferenceLearning.lean`.

Again, `grpo_rl_equivalence_via_ZR` is the `ZR`-specialized version.

---

## 8. Quantitative bound: the unified preference gap

When distortion is not exactly zero, we want a bound of the form:

> `| E_X[E_gen(X)] - E_Z[E_gen(Z)] | ≤ L · Δ_R`,

where `Δ_R` is the expected oracle distortion between originals and summaries.
In the neural-operator route, `Δ_R` is not treated as a bare free parameter:
external approximation gives a realized-call tolerance, the local-law transfer
bridge converts that tolerance into leaf/merge/idempotence budgets, and the
preference-gap theorem only supplies the final method-transport multiplier.

### Lean statement

- `unified_preference_gap_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.
- Neural-operator bridge wrappers in
  `../FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean`:
  `NeuralOperatorTransferModuli`,
  `NeuralOperatorTransferModuli.localLawBudget`,
  `NeuralOperatorTransferModuli.methodGapBudget`,
  `ApproxNeuralOperatorPreferenceBridge.matchesTransferModuli`,
  `ApproxNeuralOperatorPreferenceBridge.localLawBudget`,
  `ApproxNeuralOperatorPreferenceBridge.localLawBudget_eq_transferModuliBudget`,
  `ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_localLawBudget`,
  `ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_transferModuliBudget`,
  `FDNeuralOperatorPreferenceBridge.matchesTransferModuli`,
  `FDNeuralOperatorPreferenceBridge.localLawBudget`,
  `FDNeuralOperatorPreferenceBridge.localLawBudget_eq_transferModuliBudget`,
  `FDNeuralOperatorPreferenceBridge.delta_R_ZR_le_localLawBudget`,
  `FDNeuralOperatorPreferenceBridge.delta_R_ZR_le_transferModuliBudget`,
  `expectedObjectiveGap_via_neuralOperatorUniformBridge`,
  `expectedObjectiveGap_via_neuralOperatorFDBridge`,
  `expectedObjectiveGap_via_neuralOperatorTransferModuli`,
  `expectedObjectiveGap_via_neuralOperatorFDTransferModuli`,
  `dpo_gap_via_neuralOperatorUniformBridge`,
  `dpo_gap_via_neuralOperatorFDBridge`,
  `grpo_pl_gap_via_neuralOperatorUniformBridge`,
  `grpo_pl_gap_via_neuralOperatorFDBridge`,
  `grpo_rl_gap_via_neuralOperatorUniformBridge`, and
  `grpo_rl_gap_via_neuralOperatorFDBridge`.

### Hand proof skeleton (the coupling argument)

Let `μ_X` and `μ_Z` be distributions on documents, and let `E_gen : Strings → ℝ` be the
“inner expected loss” for a fixed document.

1. Write the difference as a **double sum over the product measure**:
   - `E_X[E_gen] - E_Z[E_gen]`
   - `= ∑_x μ_X(x) E_gen(x) - ∑_z μ_Z(z) E_gen(z)`
   - `= ∑_x ∑_z μ_X(x) μ_Z(z) (E_gen(x) - E_gen(z))`.
2. Take absolute values and apply triangle inequality:
   - `|∑_{x,z} μ_X μ_Z (E_gen(x) - E_gen(z))|`
   - `≤ ∑_{x,z} μ_X μ_Z |E_gen(x) - E_gen(z)|`.
3. Apply the **Lipschitz assumption**:
   - `|E_gen(x) - E_gen(z)| ≤ L · dist(f*(x), f*(z))`.
4. Factor out `L` and identify the remaining quantity as `Δ_R`.
5. If using the neural-operator route, substitute the bridge-produced
   local-law budget for `Δ_R`.

### Lean anchors for the proof steps

The Lean proof follows the same steps, but must also manage `tsum` summability:

- Step (1) is `coupling_expansion_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.
- Steps (2)–(3) are packaged as `coupling_bound_ineq_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.
- The final assembly is `unified_preference_gap_bounded` in `../FormalProofs/OPT/PreferenceBounds.lean`.
- The neural-operator wrappers first call
  `ApproxNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle` or the
  finite-dimensionalization bridge, then apply the same bounded-gap machinery.

---

## 9. Necessity: C2 is independent

**Paper:** Example “Counterexample: C1 and C3 do not imply C2”
(`ex:c2-independent`).

**Lean:** `ex_c2_independent_formalized` in
`../FormalProofs/OPT/CounterexampleExistence.lean`, with the older core lemma
`thm10_1_L3_not_derivable` retained as a smaller anchor.

### Hand proof idea

Construct a summarizer that behaves well on *fresh* inputs, but misbehaves on its own outputs:

- On fresh strings `b`, it returns a canonical representative consistent with the oracle value.
- On some `s ∈ range(g)`, it *flips* (creates a 2-cycle), violating on-range idempotence.

This demonstrates why L3/C2 is a genuine extra condition, not redundant with L1 and L2.

---

## 10. Suggested reading order (Lean)

If you want to follow the proofs directly in Lean, a good order is:

1. `../FormalProofs/OPT/CoreDefinitions.lean`
2. `../FormalProofs/OPT/LocalLaws.lean`
3. `../FormalProofs/OPT/PreservationTheorems.lean`
4. `../FormalProofs/OPT/ExpectationTheory.lean`
5. `../FormalProofs/OPT/PreferenceLearning.lean`
6. `../FormalProofs/OPT/PreferenceBounds.lean`
7. `../FormalProofs/OPT/MainTheorems.lean` (for the polished/curated layer)
