import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.MergeTriangle
import FormalProofs.OPT.OptimizationPerturbation
import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.NeuralOperatorPreferenceBridge
import FormalProofs.OPT.MathlibBayesBridge
import FormalProofs.OPT.FiniteBayesOnState
import FormalProofs.OPT.PosteriorConsistency
import FormalProofs.DSL.TreeIPW
import FormalProofs.DSL.IPWTheory
import FormalProofs.DSL.TreePOEndToEnd

/-!
# Paper Theorems: The C-TreePO Paper-Facing Export Surface

This file declares exactly the names cited by the paper's Lean crosswalk
(`paper/ctreepo/appendix/v13_triangle/E_proof_artifacts.tex`) that previously
lived in `FormalProofs/OPT/MainTheorems.lean`, together with the in-file
declarations they depend on. Its import list is intentionally minimal, so
`lake build FormalProofs.OPT.PaperTheorems` compiles exactly the
paper-relevant closure. The remaining curated exports live in
`FormalProofs/OPT/ExtendedExports.lean`, and `FormalProofs/OPT/MainTheorems.lean`
is a compatibility shim importing both, so all fully-qualified names are
unchanged.

The original `MainTheorems.lean` module documentation follows.

# Main Theorems: Local-to-Global Oracle Preference Learning

This file collects and documents the main results of the formalization. These theorems
establish that **local consistency conditions imply global training equivalence** for
a broad class of preference learning methods including DPO, GRPO, and PPO-style RL.

## The Core Insight

The central contribution is NOT the distillation/gap-composition results (which follow
from triangle inequality). Rather, it is the **local-to-global mechanism**:

1. **Local Laws (L1, L2, L3)**: Testable conditions on a summarizer `g`
2. **Zero Distortion**: L1 + L2 + L3 imply `E[dist(f*(Z), f*(x))] = 0` after R rounds
3. **Oracle-Measurability**: When loss/generator depend on x only through f*(x)
4. **Training Equivalence**: Optimal policies on summarized data = optimal on originals

This means we can **test locally** (audit the summarizer) and **conclude globally**
(training on summaries is as good as training on full documents).

## Theorem Hierarchy

```
                    Local Laws (L1, L2, L3)
                           │
                           ▼
              multi_round_proper (ExpectationTheory)
              E[D(Z^R, x)] = 0 for all R ≥ 1
                           │
                           ▼
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
  preference_learning_equivalence     grpo_rl_equivalence
  (PreferenceLearning.lean)           (PreferenceLearning.lean)
         │                                   │
         ▼                                   ▼
  dpo_gap_zero_of_local_laws         DPO/GRPO training sound
  (PreferenceBounds.lean)            on summarized data
```

## Coverage of Modern Methods

The formalization captures:

| Method | File | Key Theorem |
|--------|------|-------------|
| DPO | PreferenceBounds.lean | `dpo_equivalence` |
| Plackett-Luce GRPO | PreferenceLearning.lean | `grpo_equivalence` |
| GRPO-RL (DeepSeek-R1) | PreferenceLearning.lean | `grpo_rl_equivalence` |
| General pairwise | PreferenceLearning.lean | `preference_learning_equivalence` |
| General group-wise | PreferenceLearning.lean | `expected_group_loss_eq_of_zero_dist` |
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

noncomputable section

namespace MainTheorems

/-!
## Theorem 1: Multi-Round Preservation

**Statement**: If local laws L1, L2, L3 hold for summarizer g on tree T, then
after R rounds of summarization, the expected oracle distortion is exactly zero.

**Significance**: This is the foundational result. It shows that local testable
conditions (which an auditor can check on individual summarizer calls) guarantee
global preservation of oracle information through arbitrary reduction depth.

**Paper Reference**: Theorem 5.1 (Multi-Round Preservation)
-/

/-- **⚠ Per-tree kernel only.** This alias is a literal rename of
`multi_round_proper`: it covers the fixed-tree kernel of the paper's
fixed-partition theorem, not the extension that names it. The actual
extension — deterministic partition rule `Π`, document distribution `μ_X`, and
the tower step — is formalized as `fixed_partition_population` (support and
expectation forms) in `FormalProofs/OPT/MergeTriangle.lean`; cite that for
Appendix C. -/
abbrev fixed_partition_extension_instantiation := @multi_round_proper

/-- Coupling-form distortion equals document-level `Δ_R_ZR` when `μ_X = pure(x)`. -/
abbrev coupling_delta_eq_delta_r_zr := @coupling_Δ_eq_Δ_R_ZR

/-!
## Merge Triangle and Compositional Preservation (curated exports)

The paper's central compositionality law `g(x·y) ~ g(g(x)·g(y))` and the
de-circularized preservation tier live in `FormalProofs/OPT/MergeTriangle.lean`;
these aliases give them stable paper-facing names.
-/

/-- Error-budget union bound (paper Equation `eq:error_budget`). -/
abbrev paper_error_budget_union_bound' := @paper_error_budget_union_bound

/-!
## Stochastic and Deterministic Theorem-Backed Interfaces

The core C-TreePO semantics are stochastic: `Summarizer α = α → PMF α`.
Deterministic operators, including the current neural-operator realization
bridges, enter as the special case `deterministicSummarizer s`.
-/

/-- Paper-facing stochastic exact route: direct local laws for a PMF-valued
summarizer imply exact theorem-backedness. -/
abbrev stochastic_direct_exact_theorem_backed :=
  @FormalProofs.OPT.DirectSummaryExactAssumptions.toExactTheoremBacked

/-- Paper-facing stochastic approximate route: approximate local laws for a
PMF-valued summarizer imply approximate theorem-backedness. -/
abbrev stochastic_direct_approx_theorem_backed :=
  @FormalProofs.OPT.DirectSummaryApproxAssumptions.toApproxTheoremBacked

/-- Embedding of deterministic summary operators into the stochastic
`Summarizer` interface. -/
abbrev deterministic_summarizer_embedding :=
  @deterministicSummarizer

/-- Deterministic exact theorem-backedness from the global preservation
typeclass route. -/
abbrev deterministic_exact_theorem_backed_of_global_preservation :=
  @FormalProofs.OPT.exactTheoremBacked_of_globalPreservation

/-!
## Neural-Operator Bridge Exports

These aliases expose the Lean-backed interface that connects Section 9-style
neural-operator approximation assumptions to the existing approximate
theorem-backed route. These bridges certify deterministic realizers and then
embed them into the stochastic PMF-valued summarizer semantics above.
-/

/-- Paper formula
`ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`. -/
abbrev neural_operator_transfer_local_law_budget :=
  @FormalProofs.OPT.NeuralOperatorTransferModuli.localLawBudget

/-- Paper formula with method transport:
`C_meth * (ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε))`. -/
abbrev neural_operator_transfer_method_gap_budget :=
  @FormalProofs.OPT.NeuralOperatorTransferModuli.methodGapBudget

/-- Paper-form uniform `Δ_R` bound using
`ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`. -/
abbrev neural_operator_delta_r_transfer_moduli_bound :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_transferModuliBudget

/-!
## Application Assumption Packages

Application sections, including the manifesto/RILE workflow, are not separate
theorems. They supply these predicate packages and then instantiate the generic
DPO/GRPO theorem surfaces.
-/

/-- DPO application package: current and reference policies are
oracle-measurable. Pair-generation oracle-indexing is exported separately. -/
abbrev dpo_application_oracle_measurable_policies :=
  @OracleMeasurablePolicies

/-- DPO application package: preference-pair generation is oracle-indexed. -/
abbrev dpo_application_oracle_indexed_pair_generator :=
  @OracleIndexedPairGen

/-- GRPO-PL application package: policy, ranker, and group generator satisfy
the oracle-measurability/indexing conditions used by `grpo_equivalence`. -/
abbrev grpo_pl_application_oracle_measurable_bundle :=
  @GRPOOracleMeasurableBundle

/-- GRPO-RL application package: policy measurability predicate. -/
abbrev grpo_rl_application_oracle_measurable_policy :=
  @GRPOOracleMeasurable

/-- GRPO-RL application package: reward measurability predicate. -/
abbrev grpo_rl_application_oracle_measurable_reward :=
  @OracleMeasurableReward

/-- GRPO-RL application package: group generator oracle-indexing predicate. -/
abbrev grpo_rl_application_oracle_indexed_group_generator :=
  @OracleIndexedGroupGen

/-- Bundled DPO application assumptions: the policies are oracle-measurable
and the preference-pair generator is oracle-indexed. -/
structure DPOApplicationAssumptionBundle {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (gen : PairGenerator Strings A)
    (fstar : Strings → Y) where
  policies : OracleMeasurablePolicies pol pol_ref fstar
  pair_generator : OracleIndexedPairGen gen fstar

/-- A DPO application bundle supplies exactly the premises used by the generic
DPO theorem surfaces. -/
theorem dpo_application_bundle_supplies_premises
    {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {gen : PairGenerator Strings A}
    {fstar : Strings → Y}
    (bundle : DPOApplicationAssumptionBundle pol pol_ref gen fstar) :
    OracleMeasurablePolicies pol pol_ref fstar ∧ OracleIndexedPairGen gen fstar :=
  ⟨bundle.policies, bundle.pair_generator⟩

/-- Bundled GRPO-PL application assumptions: policy, ranker, and group
generator satisfy the oracle-measurability/indexing predicates. -/
structure GRPOPLApplicationAssumptionBundle {Strings A Y : Type*} [PseudoMetricSpace Y]
    {k : ℕ} (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k) (fstar : Strings → Y) where
  oracle_bundle : GRPOOracleMeasurableBundle pol ranker gen fstar

/-- A GRPO-PL application bundle supplies exactly the bundled predicate used by
the GRPO-PL theorem surfaces. -/
theorem grpo_pl_application_bundle_supplies_premises
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    {pol : Policy' Strings A} {ranker : Strings → GroupRanker A k}
    {gen : GroupGenerator Strings A k} {fstar : Strings → Y}
    (bundle : GRPOPLApplicationAssumptionBundle pol ranker gen fstar) :
    GRPOOracleMeasurableBundle pol ranker gen fstar :=
  bundle.oracle_bundle

/-- Bundled GRPO-RL application assumptions: current, old, and reference
policies are oracle-measurable, reward is oracle-measurable, and group
generation is oracle-indexed. -/
structure GRPORLApplicationAssumptionBundle {Strings A Y : Type*} [PseudoMetricSpace Y]
    {k : ℕ} (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (gen : GroupGenerator Strings A k)
    (fstar : Strings → Y) where
  pol_measurable : GRPOOracleMeasurable pol fstar
  old_measurable : GRPOOracleMeasurable pol_old fstar
  ref_measurable : GRPOOracleMeasurable pol_ref fstar
  reward_measurable : OracleMeasurableReward reward fstar
  group_generator : OracleIndexedGroupGen gen fstar

/-- A GRPO-RL application bundle supplies exactly the premises used by the
GRPO-RL theorem surfaces. -/
theorem grpo_rl_application_bundle_supplies_premises
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    {pol pol_old pol_ref : Policy' Strings A}
    {reward : Strings → A → ℝ} {gen : GroupGenerator Strings A k}
    {fstar : Strings → Y}
    (bundle : GRPORLApplicationAssumptionBundle pol pol_old pol_ref reward gen fstar) :
    GRPOOracleMeasurable pol fstar ∧
      GRPOOracleMeasurable pol_old fstar ∧
      GRPOOracleMeasurable pol_ref fstar ∧
      OracleMeasurableReward reward fstar ∧
      OracleIndexedGroupGen gen fstar :=
  ⟨bundle.pol_measurable, bundle.old_measurable, bundle.ref_measurable,
    bundle.reward_measurable, bundle.group_generator⟩

/-!
## Paper Preference Stack

The public preference statement is method-generic: once an application bundle
instantiates the DPO/GRPO theorem premises, exact theorem-backed objectives are
the same on the admissible class. If the paper uses a noisy or approximate
preference object, the residual field records the uniform objective drift.
-/

/-- Paper-facing preference-objective alignment stack.

`fullObjective` is the objective on originals, `summaryObjective` is the
objective on summaries or theorem-backed preference objects, `admissible` is the
policy/model class supplied by the application bundle, and `objective_gap`
records the uniform residual on that class. -/
structure PaperPreferenceStack (Θ : Type*) where
  fullObjective : Θ → ℝ
  summaryObjective : Θ → ℝ
  admissible : Θ → Prop
  residual : ℝ
  objective_gap : ∀ θ, admissible θ → |fullObjective θ - summaryObjective θ| ≤ residual

namespace PaperPreferenceStack

/-- Exact constrained argmin set for a paper preference objective. -/
def argminSet {Θ : Type*} (s : PaperPreferenceStack Θ) (objective : Θ → ℝ) : Set Θ :=
  FormalProofs.OPT.ConstrainedParamEpsilonArgmin objective s.admissible 0

/-- Constrained `ε`-argmin set for a paper preference objective. -/
def epsilonArgminSet {Θ : Type*} (s : PaperPreferenceStack Θ)
    (objective : Θ → ℝ) (ε : ℝ) : Set Θ :=
  FormalProofs.OPT.ConstrainedParamEpsilonArgmin objective s.admissible ε

/-- If the residual is zero, the full and summary constrained argmin sets are
identical. -/
theorem same_argmin_of_zero_residual {Θ : Type*} (s : PaperPreferenceStack Θ)
    (h_residual : s.residual = 0) :
    s.argminSet s.fullObjective = s.argminSet s.summaryObjective := by
  apply Set.Subset.antisymm
  · simpa [argminSet, h_residual, abs_sub_comm] using
      (FormalProofs.OPT.constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
        (lossTrue := s.summaryObjective)
        (lossSur := s.fullObjective)
        (feasible := s.admissible)
        (ε := 0)
        (hclose := by
          intro θ hθ
          simpa [h_residual, abs_sub_comm] using s.objective_gap θ hθ))
  · simpa [argminSet, h_residual] using
      (FormalProofs.OPT.constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
        (lossTrue := s.fullObjective)
        (lossSur := s.summaryObjective)
        (feasible := s.admissible)
        (ε := 0)
        (hclose := by
          intro θ hθ
          simpa [h_residual] using s.objective_gap θ hθ))

/-- Any exact summary-objective minimizer is `2 * residual`-optimal for the full
objective on the same admissible class. -/
theorem summary_argmin_full_epsilon {Θ : Type*} (s : PaperPreferenceStack Θ) :
    s.argminSet s.summaryObjective ⊆
      s.epsilonArgminSet s.fullObjective (2 * s.residual) := by
  simpa [argminSet, epsilonArgminSet] using
    (FormalProofs.OPT.constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
      (lossTrue := s.fullObjective)
      (lossSur := s.summaryObjective)
      (feasible := s.admissible)
      (ε := s.residual)
      (hclose := s.objective_gap))

end PaperPreferenceStack

/-- Public alias: zero residual gives identical full/summary argmins. -/
abbrev paper_preference_stack_same_argmin :=
  @PaperPreferenceStack.same_argmin_of_zero_residual

/-- Public alias: exact summary minimizers are full-objective
`2 * residual`-minimizers. -/
abbrev paper_preference_stack_summary_argmin_full_epsilon :=
  @PaperPreferenceStack.summary_argmin_full_epsilon

/-!
## Audit Robustness Exports

These are the paper-facing anchors for the adversarial-sampling appendix:
logged marginal propensities give HT unbiasedness, while variance control is
reported through the explicit constrained-design proxy.
-/

/-- HT unbiasedness under arbitrary sampling with correct logged marginal
propensities. -/
abbrev ht_unbiased_of_logged_marginals :=
  @htExp_unbiased_of_logged_marginals

/-- Uniform finite-population HT unbiasedness under arbitrary sampling with
correct logged marginal propensities. -/
abbrev ht_uniform_mean_unbiased_of_logged_marginals :=
  @htUniformMean_unbiased_of_logged_marginals

/-- Actual constrained-design variance bound once covariance control is
available. -/
abbrev ht_uniform_mean_variance_bound_constrained :=
  @htUniformMean_variance_bound_of_constrained_design

/-- Bernoulli-product-measure surface for the same variance bound. -/
abbrev ht_uniform_mean_variance_bound_independent_bernoulli :=
  @htUniformMean_variance_bound_of_independent_bernoulli

/-- Independent Bernoulli product sampling satisfies the covariance-control
proxy used by the constrained-design variance theorem. -/
abbrev ht_uniform_mean_covariance_controlled_independent_bernoulli :=
  @htUniformMean_covarianceControlled_independent_bernoulli

/-- TreePO distortion HT unbiasedness under logged marginal propensities. -/
abbrev tree_audit_uniform_distortion_unbiased_logged :=
  @treeAuditUniformDistortion_unbiased_of_logged_marginals

/-- TreePO distortion constrained-design variance bound. -/
abbrev tree_audit_uniform_distortion_variance_bound_constrained :=
  @treeAuditUniformDistortion_variance_bound_of_constrained_design

/-- TreePO distortion Bernoulli-product-measure variance-bound surface. -/
abbrev tree_audit_uniform_distortion_variance_bound_independent_bernoulli :=
  @treeAuditUniformDistortion_variance_bound_of_independent_bernoulli

/-!
## Paper Error Certificate Exports

These aliases make the displayed finite-sample certificate formula a first-class
Lean object: transported local-law distortion, calibration, estimation, and
clipping.
-/

/-- The paper error certificate expands definitionally to the displayed
formula. -/
abbrev paper_error_certificate_formula :=
  @PaperErrorCertificate.totalObjectiveBound_eq_paper_formula

/-- High-probability objective-gap wrapper from calibration, estimation,
clipping, and local-law transport events. -/
abbrev paper_error_certificate_high_prob :=
  @PaperErrorCertificate.high_prob_total_of_events

/-- High-probability objective-gap wrapper for the bundled paper error stack. -/
abbrev paper_error_stack_high_prob :=
  @PaperErrorStack.high_prob_total

/-- Clipped inverse-propensity weight `min(w, w_max)`. -/
abbrev clipped_ipw_weight := @TreeSample.clippedWeight

/-- Deterministic clipping excess `w - min(w, w_max)`. -/
abbrev clipped_ipw_excess := @TreeSample.clippingExcess

/-- Aggregate clipping-bias envelope from total clipped mass. -/
abbrev clipped_ipw_bias_abs_bound := @clippingBiasAbsBound

/-- Deterministic aggregate clipping-bias bound. -/
abbrev clipped_ipw_total_bias_bound :=
  @totalClippingBias_abs_le

/-- Deterministic clipped-vs-unclipped Hajek gap bound. -/
abbrev clipped_hajek_gap_bound :=
  @clippedHajek_abs_diff_le

/-- Relative clipped-mass corollary for the Hajek gap. -/
abbrev clipped_hajek_gap_bound_relative_excess :=
  @clippedHajek_abs_diff_le_of_relative_excess

/-- Unit-range corollary for the clipped Hajek gap. -/
abbrev clipped_hajek_gap_bound_unit :=
  @clippedHajek_abs_diff_le_unit

/-!
### Unified-g Literature Method Certificates

These aliases organize the main paper routes for learning or certifying a
unified `g` sufficient statistic for `f*`.
-/

/-- Mathlib event-level Bayes rule for conditional probabilities. -/
abbrev mathlib_conditional_bayes_rule :=
  @FormalProofs.OPT.mathlib_conditional_bayes_rule

/-- Mathlib finite-fiber law of total probability for a random variable. -/
abbrev mathlib_conditional_probability_finite_fiber_total :=
  @FormalProofs.OPT.mathlib_conditional_probability_finite_fiber_total

/-- Mathlib kernel/disintegration posterior `posterior κ μ`, notation `κ†μ`. -/
abbrev mathlib_kernel_posterior :=
  @FormalProofs.OPT.mathlib_kernel_posterior

/-- Mathlib kernel posterior defining identity for the swapped joint law. -/
abbrev mathlib_kernel_posterior_compProd_eq_map_swap :=
  @FormalProofs.OPT.mathlib_kernel_posterior_compProd_eq_map_swap

/-- Mathlib countable-parameter posterior density/Bayes formula. -/
abbrev mathlib_kernel_posterior_with_density_countable :=
  @FormalProofs.OPT.mathlib_kernel_posterior_with_density_countable

/-- Mathlib posterior density/Bayes formula under an absolute-continuity
assumption. -/
abbrev mathlib_kernel_posterior_eq_with_density :=
  @FormalProofs.OPT.mathlib_kernel_posterior_eq_with_density

/-- Mathlib posterior Radon-Nikodym derivative identity. -/
abbrev mathlib_kernel_posterior_rn_deriv :=
  @FormalProofs.OPT.mathlib_kernel_posterior_rn_deriv

/-- Mathlib posterior uniqueness up to a.e. equality from the swapped joint-law
identity. -/
abbrev mathlib_kernel_posterior_unique_ae :=
  @FormalProofs.OPT.mathlib_kernel_posterior_unique_ae

/-- Mathlib theorem that composing a kernel with its posterior recovers the
prior measure. -/
abbrev mathlib_kernel_posterior_comp_self :=
  @FormalProofs.OPT.mathlib_kernel_posterior_comp_self

/-- Mathlib law-as-with-density theorem for random variables with PDFs. -/
abbrev mathlib_pdf_map_eq_with_density :=
  @FormalProofs.OPT.mathlib_pdf_map_eq_with_density

/-- Mathlib nonnegative LOTUS theorem for PDFs. -/
abbrev mathlib_pdf_lintegral_lotus :=
  @FormalProofs.OPT.mathlib_pdf_lintegral_lotus

/-- Finite Bayes posterior normalizes when evidence is nonzero. -/
abbrev finite_bayes_posterior_sum_eq_one :=
  @FormalProofs.OPT.bayesPosterior_sum_eq_one

/-- Positive normalization preserves finite Bayes MAP decisions. -/
abbrev finite_bayes_posterior_map_iff_numerator_map :=
  @FormalProofs.OPT.bayesPosteriorMAP_iff_bayesNumeratorMAP

/-- Positive state normalization preserves state finite Bayes MAP decisions. -/
abbrev state_finite_bayes_posterior_map_iff_numerator_map :=
  @FormalProofs.OPT.stateBayesPosteriorMAP_iff_stateBayesNumeratorMAP

/-- Posterior odds cancel the evidence and equal Bayes-numerator odds. -/
abbrev finite_bayes_posterior_odds_eq_numerator_odds :=
  @FormalProofs.OPT.bayesPosterior_odds_eq_bayesNumerator_odds

/-- State posterior odds cancel the state evidence and equal state
Bayes-numerator odds. -/
abbrev state_finite_bayes_posterior_odds_eq_numerator_odds :=
  @FormalProofs.OPT.stateBayesPosterior_odds_eq_stateBayesNumerator_odds

/-- Likelihood-on-state finite posterior expectations equal state posterior
expectations. -/
abbrev finite_bayes_posterior_expectation_likelihood_on_state_eq_state :=
  @FormalProofs.OPT.bayesPosteriorExpectation_likelihoodOnState_eq_state

/-- Likelihood-on-state finite posterior expectations are state-sufficient. -/
abbrev finite_bayes_posterior_expectation_likelihood_on_state_sufficient :=
  @FormalProofs.OPT.bayesPosteriorExpectation_likelihoodOnState_sufficient

/-- Likelihood-on-state finite posterior predictive likelihoods equal state
posterior predictive likelihoods. -/
abbrev finite_bayes_posterior_predictive_likelihood_on_state_eq_state :=
  @FormalProofs.OPT.bayesPosteriorPredictive_likelihoodOnState_eq_state

/-- For a fixed future observation, likelihood-on-state finite posterior
predictives are sufficient in the observed learned state. -/
abbrev finite_bayes_posterior_predictive_likelihood_on_state_sufficient_observed :=
  @FormalProofs.OPT.bayesPosteriorPredictive_likelihoodOnState_sufficient_observed

/-- Finite Bayes posterior packaged as a mathlib `PMF` under positivity
assumptions. -/
abbrev finite_bayes_posterior_pmf :=
  @FormalProofs.OPT.BayesPosteriorPMF

/-- The finite Bayes PMF's induced measure has singleton masses matching the
posterior mass function. -/
abbrev finite_bayes_posterior_pmf_to_measure_singleton :=
  @FormalProofs.OPT.bayesPosteriorPMF_toMeasure_singleton

/-- The finite Bayes PMF's induced measure has arbitrary-event masses matching
the finite sum of posterior masses. -/
abbrev finite_bayes_posterior_pmf_to_measure_set :=
  @FormalProofs.OPT.bayesPosteriorPMF_toMeasure_set

/-- State finite Bayes posterior packaged as a mathlib `PMF` under positivity
assumptions. -/
abbrev state_finite_bayes_posterior_pmf :=
  @FormalProofs.OPT.StateBayesPosteriorPMF

/-- The state finite Bayes PMF's induced measure has arbitrary-event masses
matching the finite sum of posterior masses. -/
abbrev state_finite_bayes_posterior_pmf_to_measure_set :=
  @FormalProofs.OPT.stateBayesPosteriorPMF_toMeasure_set

/-- If the likelihood factors through state, the raw finite Bayes PMF equals
the state finite Bayes PMF. -/
abbrev finite_bayes_posterior_pmf_likelihood_on_state_eq_state_pmf :=
  @FormalProofs.OPT.bayesPosteriorPMF_likelihoodOnState_eq_stateBayesPosteriorPMF

/-- Posterior consistency is exactly mathlib `TendstoInMeasure` along
`Filter.atTop`. -/
abbrev posterior_consistent_iff_mathlib_tendsto_in_measure :=
  @FormalProofs.OPT.posteriorConsistent_iff_mathlib_tendstoInMeasure

/-- Finite posterior mass concentration is exactly mathlib `TendstoInMeasure`
for the target parameter's posterior mass. -/
abbrev finite_posterior_mass_concentrates_at_iff_mathlib_tendsto_in_measure :=
  @FormalProofs.OPT.finitePosteriorMassConcentratesAt_iff_mathlib_tendstoInMeasure

/-- For likelihood-on-state finite Bayes, raw posterior concentration is
equivalent to state posterior concentration. -/
abbrev finite_bayes_consistency_likelihood_on_state_iff :=
  @FormalProofs.OPT.finiteBayesConsistency_likelihoodOnState_iff

/-- Exact state readout transports finite Bayes posterior concentration. -/
abbrev state_readout_finite_bayes_consistency :=
  @FormalProofs.OPT.stateReadout_finiteBayesConsistency

end MainTheorems

end
