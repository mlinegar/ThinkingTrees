import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.ConditionalExpectation
import Mathlib.Probability.Density
import Mathlib.Probability.Kernel.Posterior
import Mathlib.Probability.ProbabilityMassFunction.Basic
import FormalProofs.OPT.PosteriorConsistency

/-!
# FormalProofs/OPT/MathlibBayesBridge.lean

Thin alignment layer between the repo's finite/state Bayes surfaces and
mathlib's probability APIs.

The existing OPT Bayes files deliberately stay finite and algebraic: they use
real-valued priors/likelihoods and deterministic state/readout sufficiency.  In
mathlib, the broader Bayesian semantics live in:

* `ProbabilityTheory.cond` and `cond_eq_inv_mul_cond_mul` for event-level
  conditional probability and Bayes' rule;
* `MeasureTheory.condExp` and its uniqueness/integral identities for
  conditional-expectation semantics;
* `ProbabilityTheory.posterior` for kernel/disintegration posterior semantics;
* `MeasureTheory.HasPDF` and `MeasureTheory.pdf` for dominated density
  surfaces via Radon-Nikodym derivatives;
* `PMF` for discrete probability mass functions; and
* `MeasureTheory.TendstoInMeasure` for convergence in probability.

This file does not replace the local finite algebra with a measure-theoretic
Bayes theorem.  It exposes exact aliases to the mathlib APIs, proves the local
posterior-consistency predicates are definitionally mathlib convergence in
measure, and packages finite Bayes posteriors as mathlib `PMF`s when the usual
positivity assumptions are supplied.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Classical
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-! ## Direct mathlib API aliases -/

/-- Mathlib's event-conditioned measure `μ[|s]`. -/
abbrev mathlib_conditional_probability :=
  @ProbabilityTheory.cond

/-- Mathlib's event-level Bayes rule:
`μ[t|s] = (μ s)⁻¹ * μ[s|t] * μ t`. -/
abbrev mathlib_conditional_bayes_rule :=
  @ProbabilityTheory.cond_eq_inv_mul_cond_mul

/-- Mathlib's application formula for conditional probability:
`μ[t|s] = (μ s)⁻¹ * μ (s ∩ t)`. -/
abbrev mathlib_conditional_probability_apply :=
  @ProbabilityTheory.cond_apply

/-- Mathlib's theorem that conditioning on `s` and then `t` is conditioning on
`s ∩ t`. -/
abbrev mathlib_conditional_probability_condition_twice :=
  @ProbabilityTheory.cond_cond_eq_cond_inter

/-- Mathlib's conditional-probability product identity:
`μ[t|s] * μ s = μ (s ∩ t)`. -/
abbrev mathlib_conditional_probability_mul_eq_inter :=
  @ProbabilityTheory.cond_mul_eq_inter

/-- Mathlib's complement form of the law of total probability. -/
abbrev mathlib_conditional_probability_total_complement :=
  @ProbabilityTheory.cond_add_cond_compl_eq

/-- Mathlib's finite-fiber law of total probability for a random variable. -/
abbrev mathlib_conditional_probability_finite_fiber_total :=
  @ProbabilityTheory.sum_meas_smul_cond_fiber

/-- Mathlib's condition ensuring a conditional measure is a probability
measure. -/
abbrev mathlib_conditional_probability_is_probability :=
  @ProbabilityTheory.cond_isProbabilityMeasure

/-- A conditional measure is absolutely continuous with respect to the original
measure. -/
abbrev mathlib_conditional_probability_absolutely_continuous :=
  @ProbabilityTheory.cond_absolutelyContinuous

/-- Mathlib's conditional expectation `μ[f|m]`. -/
abbrev mathlib_conditional_expectation :=
  @MeasureTheory.condExp

/-- Conditional expectation preserves addition, a.e., for integrable
summands. -/
abbrev mathlib_conditional_expectation_add :=
  @MeasureTheory.condExp_add

/-- Conditional expectation of a constant under a finite measure. -/
abbrev mathlib_conditional_expectation_const :=
  @MeasureTheory.condExp_const

/-- Conditional expectation respects a.e.-equal versions of the integrand. -/
abbrev mathlib_conditional_expectation_congr_ae :=
  @MeasureTheory.condExp_congr_ae

/-- Integral preservation for conditional expectation. -/
abbrev mathlib_integral_conditional_expectation :=
  @MeasureTheory.integral_condExp

/-- Set-integral preservation for conditional expectation over measurable
sets. -/
abbrev mathlib_set_integral_conditional_expectation :=
  @MeasureTheory.setIntegral_condExp

/-- Conditional expectations are strongly measurable with respect to the
conditioning sigma-algebra. -/
abbrev mathlib_strongly_measurable_conditional_expectation :=
  @MeasureTheory.stronglyMeasurable_condExp

/-- Conditional expectations are integrable. -/
abbrev mathlib_integrable_conditional_expectation :=
  @MeasureTheory.integrable_condExp

/-- If the function is already strongly measurable for the conditioning
sigma-algebra, conditional expectation returns it. -/
abbrev mathlib_conditional_expectation_of_strongly_measurable :=
  @MeasureTheory.condExp_of_stronglyMeasurable

/-- Conditional expectation commutes with indicators of measurable sets, a.e. -/
abbrev mathlib_conditional_expectation_indicator :=
  @MeasureTheory.condExp_indicator

/-- Conditional expectation as a Radon-Nikodym derivative. -/
abbrev mathlib_rn_deriv_ae_eq_conditional_expectation :=
  @MeasureTheory.rnDeriv_ae_eq_condExp

/-- Conditional expectation of an independent integrand is its integral, a.e. -/
abbrev mathlib_conditional_expectation_independent_eq_integral :=
  @MeasureTheory.condExp_indep_eq

/-- Conditional expectation commutes with scalar multiplication, a.e. -/
abbrev mathlib_conditional_expectation_smul :=
  @MeasureTheory.condExp_smul

/-- Monotonicity of conditional expectation. -/
abbrev mathlib_conditional_expectation_mono :=
  @MeasureTheory.condExp_mono

/-- Mathlib's kernel posterior `posterior κ μ`, notation `κ†μ`. -/
abbrev mathlib_kernel_posterior :=
  @ProbabilityTheory.posterior

/-- Mathlib's defining posterior/disintegration identity:
the data marginal composed with the posterior recovers the swapped joint law. -/
abbrev mathlib_kernel_posterior_compProd_eq_map_swap :=
  @ProbabilityTheory.compProd_posterior_eq_map_swap

/-- Mathlib's countable-parameter posterior density/Bayes formula. -/
abbrev mathlib_kernel_posterior_with_density_countable :=
  @ProbabilityTheory.posterior_eq_withDensity_of_countable

/-- Mathlib's general posterior density formula under an absolute-continuity
assumption. -/
abbrev mathlib_kernel_posterior_eq_with_density :=
  @ProbabilityTheory.posterior_eq_withDensity

/-- Mathlib's Radon-Nikodym derivative identity for posterior kernels. -/
abbrev mathlib_kernel_posterior_rn_deriv :=
  @ProbabilityTheory.rnDeriv_posterior

/-- Symmetric form of the posterior Radon-Nikodym derivative identity. -/
abbrev mathlib_kernel_posterior_rn_deriv_symm :=
  @ProbabilityTheory.rnDeriv_posterior_symm

/-- The posterior is unique, up to the data marginal, among kernels satisfying
the swapped joint-law identity. -/
abbrev mathlib_kernel_posterior_unique_ae :=
  @ProbabilityTheory.ae_eq_posterior_of_compProd_eq

/-- Composing a kernel with its posterior recovers the prior measure. -/
abbrev mathlib_kernel_posterior_comp_self :=
  @ProbabilityTheory.posterior_comp_self

/-- The posterior of the identity kernel is the identity kernel, a.e. -/
abbrev mathlib_kernel_posterior_identity :=
  @ProbabilityTheory.posterior_id

/-- Posterior inversion is involutive up to a.e. equality. -/
abbrev mathlib_kernel_posterior_posterior :=
  @ProbabilityTheory.posterior_posterior

/-- Posterior kernels compose contravariantly up to a.e. equality. -/
abbrev mathlib_kernel_posterior_comp :=
  @ProbabilityTheory.posterior_comp

/-- Mathlib's `HasPDF` class for dominated random variables. -/
abbrev mathlib_has_pdf :=
  @MeasureTheory.HasPDF

/-- Mathlib's probability density function, defined as an RN derivative of the
pushforward law. -/
abbrev mathlib_pdf :=
  @MeasureTheory.pdf

/-- Mathlib's defining equality for `pdf`. -/
abbrev mathlib_pdf_def :=
  @MeasureTheory.pdf_def

/-- Mathlib's characterization of `HasPDF`. -/
abbrev mathlib_has_pdf_iff :=
  @MeasureTheory.hasPDF_iff

/-- If a random variable has a PDF, its law is the reference measure weighted by
that PDF. -/
abbrev mathlib_pdf_map_eq_with_density :=
  @MeasureTheory.map_eq_withDensity_pdf

/-- Setwise density formula for probabilities of measurable sets. -/
abbrev mathlib_pdf_map_eq_set_lintegral :=
  @MeasureTheory.map_eq_setLIntegral_pdf

/-- Nonnegative LOTUS: integrate a function of the random variable through the
PDF. -/
abbrev mathlib_pdf_lintegral_lotus :=
  @MeasureTheory.pdf.lintegral_pdf_mul

/-- Mathlib's discrete probability mass function type. -/
abbrev mathlib_probability_mass_function :=
  PMF

/-- Mathlib's PMF-to-measure construction. -/
abbrev mathlib_pmf_to_measure :=
  @PMF.toMeasure

/-- Mathlib's finite-type PMF measure formula for arbitrary sets. -/
abbrev mathlib_pmf_to_measure_apply_fintype :=
  @PMF.toMeasure_apply_fintype

/-- Mathlib's singleton formula for PMF-induced measures. -/
abbrev mathlib_pmf_to_measure_apply_singleton :=
  @PMF.toMeasure_apply_singleton

/-- Equality of PMF-induced measures is equivalent to equality of PMFs. -/
abbrev mathlib_pmf_to_measure_inj :=
  @PMF.toMeasure_inj

/-- Mathlib's convergence-in-measure predicate, used here as convergence in
probability because the ambient measure is a probability measure. -/
abbrev mathlib_tendsto_in_measure :=
  @MeasureTheory.TendstoInMeasure

/-- Mathlib congruence theorem for convergence in measure. -/
abbrev mathlib_tendsto_in_measure_congr :=
  @MeasureTheory.TendstoInMeasure.congr

/-- Convergence in measure admits a subsequence converging a.e. -/
abbrev mathlib_tendsto_in_measure_exists_seq_tendsto_ae :=
  @MeasureTheory.TendstoInMeasure.exists_seq_tendsto_ae

/-! ## Local definitions are mathlib convergence in measure -/

/-- The local posterior-consistency predicate is exactly mathlib
`TendstoInMeasure` along `Filter.atTop`. -/
theorem posteriorConsistent_iff_mathlib_tendstoInMeasure
    {Ω Posterior : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [PseudoMetricSpace Posterior]
    (posteriorSeq : ℕ → Ω → Posterior)
    (posteriorLimit : Ω → Posterior) :
    PosteriorConsistent μ posteriorSeq posteriorLimit ↔
      MeasureTheory.TendstoInMeasure
        μ
        posteriorSeq
        Filter.atTop
        posteriorLimit :=
  Iff.rfl

/-- Finite posterior mass concentration is exactly mathlib `TendstoInMeasure`
for the scalar posterior mass at the target parameter. -/
theorem finitePosteriorMassConcentratesAt_iff_mathlib_tendstoInMeasure
    {Ω Θ : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    (posteriorSeq : ℕ → Ω → Θ → ℝ)
    (θ0 : Θ) :
    FinitePosteriorMassConcentratesAt μ posteriorSeq θ0 ↔
      MeasureTheory.TendstoInMeasure
        μ
        (fun n ω => posteriorSeq n ω θ0)
        Filter.atTop
        (fun _ => (1 : ℝ)) :=
  Iff.rfl

/-! ## Finite Bayes posteriors as mathlib PMFs -/

variable {X State Θ : Type*} [Fintype Θ]

/-- Finite Bayes posterior masses are nonnegative under nonnegative prior,
nonnegative likelihood, and positive evidence. -/
theorem bayesPosterior_nonneg
    {prior : Θ → ℝ}
    {likelihood : Θ → X → ℝ}
    {x : X}
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ x, 0 ≤ likelihood θ x)
    (hEvidence : 0 < BayesEvidence prior likelihood x) :
    ∀ θ, 0 ≤ BayesPosterior prior likelihood x θ := by
  intro θ
  unfold BayesPosterior BayesNumerator
  exact div_nonneg
    (mul_nonneg (hPrior θ) (hLikelihood θ x))
    (le_of_lt hEvidence)

/-- State-space finite Bayes posterior masses are nonnegative under
nonnegative prior, nonnegative state likelihood, and positive evidence. -/
theorem stateBayesPosterior_nonneg
    {prior : Θ → ℝ}
    {stateLikelihood : Θ → State → ℝ}
    {z : State}
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ z, 0 ≤ stateLikelihood θ z)
    (hEvidence : 0 < StateBayesEvidence prior stateLikelihood z) :
    ∀ θ, 0 ≤ StateBayesPosterior prior stateLikelihood z θ := by
  intro θ
  unfold StateBayesPosterior StateBayesNumerator
  exact div_nonneg
    (mul_nonneg (hPrior θ) (hLikelihood θ z))
    (le_of_lt hEvidence)

/-- Package the local finite Bayes posterior as a mathlib `PMF` when positivity
assumptions make the real-valued posterior a probability mass function. -/
def BayesPosteriorPMF
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ x, 0 ≤ likelihood θ x)
    (hEvidence : 0 < BayesEvidence prior likelihood x) :
    PMF Θ :=
  ⟨fun θ => ENNReal.ofReal (BayesPosterior prior likelihood x θ), by
    have hPostNonneg : ∀ θ, 0 ≤ BayesPosterior prior likelihood x θ :=
      bayesPosterior_nonneg hPrior hLikelihood hEvidence
    have hsumReal :
        (∑ θ : Θ, BayesPosterior prior likelihood x θ) = 1 :=
      bayesPosterior_sum_eq_one
        (prior := prior)
        (likelihood := likelihood)
        (x := x)
        (ne_of_gt hEvidence)
    have hsumENN :
        (∑ θ : Θ, ENNReal.ofReal (BayesPosterior prior likelihood x θ)) = 1 := by
      rw [← ENNReal.ofReal_sum_of_nonneg]
      · simp [hsumReal]
      · intro θ _
        exact hPostNonneg θ
    simpa [hsumENN] using
      (hasSum_fintype
        (fun θ : Θ => ENNReal.ofReal (BayesPosterior prior likelihood x θ)))
  ⟩

/-- The PMF representation has the expected point masses. -/
@[simp]
theorem bayesPosteriorPMF_apply
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ x, 0 ≤ likelihood θ x)
    (hEvidence : 0 < BayesEvidence prior likelihood x)
    (θ : Θ) :
    BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence θ =
      ENNReal.ofReal (BayesPosterior prior likelihood x θ) :=
  rfl

/-- The measure induced by the finite Bayes PMF assigns singleton mass equal to
the corresponding posterior mass. -/
theorem bayesPosteriorPMF_toMeasure_singleton
    [MeasurableSpace Θ]
    [MeasurableSingletonClass Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ x, 0 ≤ likelihood θ x)
    (hEvidence : 0 < BayesEvidence prior likelihood x)
    (θ : Θ) :
    (BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence).toMeasure {θ} =
      ENNReal.ofReal (BayesPosterior prior likelihood x θ) := by
  simpa using
    PMF.toMeasure_apply_singleton
      (BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence)
      θ
      (measurableSet_singleton θ)

/-- The measure induced by the finite Bayes PMF assigns an arbitrary event the
finite sum of posterior masses over that event. -/
theorem bayesPosteriorPMF_toMeasure_set
    [MeasurableSpace Θ]
    [MeasurableSingletonClass Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ x, 0 ≤ likelihood θ x)
    (hEvidence : 0 < BayesEvidence prior likelihood x)
    (s : Set Θ) :
    (BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence).toMeasure s =
      ∑ θ : Θ,
        s.indicator
          (fun θ => ENNReal.ofReal (BayesPosterior prior likelihood x θ))
          θ := by
  calc
    (BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence).toMeasure s
        =
          ∑ θ : Θ,
            s.indicator
              (BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence)
              θ := by
          exact
            PMF.toMeasure_apply_fintype
              (BayesPosteriorPMF prior likelihood x hPrior hLikelihood hEvidence)
              s
    _ =
          ∑ θ : Θ,
            s.indicator
              (fun θ => ENNReal.ofReal (BayesPosterior prior likelihood x θ))
              θ := by
          refine Finset.sum_congr rfl ?_
          intro θ _
          by_cases hθ : θ ∈ s <;> simp [Set.indicator, hθ]

/-- Package the local state-space finite Bayes posterior as a mathlib `PMF`
when positivity assumptions make the real-valued posterior a probability mass
function. -/
def StateBayesPosteriorPMF
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ z, 0 ≤ stateLikelihood θ z)
    (hEvidence : 0 < StateBayesEvidence prior stateLikelihood z) :
    PMF Θ :=
  ⟨fun θ => ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ), by
    have hPostNonneg :
        ∀ θ, 0 ≤ StateBayesPosterior prior stateLikelihood z θ :=
      stateBayesPosterior_nonneg hPrior hLikelihood hEvidence
    have hsumReal :
        (∑ θ : Θ, StateBayesPosterior prior stateLikelihood z θ) = 1 :=
      stateBayesPosterior_sum_eq_one
        (prior := prior)
        (stateLikelihood := stateLikelihood)
        (z := z)
        (ne_of_gt hEvidence)
    have hsumENN :
        (∑ θ : Θ,
          ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ)) = 1 := by
      rw [← ENNReal.ofReal_sum_of_nonneg]
      · simp [hsumReal]
      · intro θ _
        exact hPostNonneg θ
    simpa [hsumENN] using
      (hasSum_fintype
        (fun θ : Θ =>
          ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ)))
  ⟩

/-- The state-space PMF representation has the expected point masses. -/
@[simp]
theorem stateBayesPosteriorPMF_apply
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ z, 0 ≤ stateLikelihood θ z)
    (hEvidence : 0 < StateBayesEvidence prior stateLikelihood z)
    (θ : Θ) :
    StateBayesPosteriorPMF prior stateLikelihood z hPrior hLikelihood hEvidence θ =
      ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ) :=
  rfl

/-- The measure induced by the state-space Bayes PMF assigns singleton mass
equal to the corresponding posterior mass. -/
theorem stateBayesPosteriorPMF_toMeasure_singleton
    [MeasurableSpace Θ]
    [MeasurableSingletonClass Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ z, 0 ≤ stateLikelihood θ z)
    (hEvidence : 0 < StateBayesEvidence prior stateLikelihood z)
    (θ : Θ) :
    (StateBayesPosteriorPMF
      prior
      stateLikelihood
      z
      hPrior
      hLikelihood
      hEvidence).toMeasure {θ} =
      ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ) := by
  simpa using
    PMF.toMeasure_apply_singleton
      (StateBayesPosteriorPMF
        prior
        stateLikelihood
        z
        hPrior
        hLikelihood
        hEvidence)
      θ
      (measurableSet_singleton θ)

/-- The measure induced by the state-space finite Bayes PMF assigns an
arbitrary event the finite sum of posterior masses over that event. -/
theorem stateBayesPosteriorPMF_toMeasure_set
    [MeasurableSpace Θ]
    [MeasurableSingletonClass Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ z, 0 ≤ stateLikelihood θ z)
    (hEvidence : 0 < StateBayesEvidence prior stateLikelihood z)
    (s : Set Θ) :
    (StateBayesPosteriorPMF prior stateLikelihood z hPrior hLikelihood hEvidence).toMeasure s =
      ∑ θ : Θ,
        s.indicator
          (fun θ => ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ))
          θ := by
  calc
    (StateBayesPosteriorPMF prior stateLikelihood z hPrior hLikelihood hEvidence).toMeasure s
        =
          ∑ θ : Θ,
            s.indicator
              (StateBayesPosteriorPMF
                prior
                stateLikelihood
                z
                hPrior
                hLikelihood
                hEvidence)
              θ := by
          exact
            PMF.toMeasure_apply_fintype
              (StateBayesPosteriorPMF
                prior
                stateLikelihood
                z
                hPrior
                hLikelihood
                hEvidence)
              s
    _ =
          ∑ θ : Θ,
            s.indicator
              (fun θ => ENNReal.ofReal (StateBayesPosterior prior stateLikelihood z θ))
              θ := by
          refine Finset.sum_congr rfl ?_
          intro θ _
          by_cases hθ : θ ∈ s <;> simp [Set.indicator, hθ]

/-- For a likelihood family that factors through a learned state, the raw
finite Bayes posterior PMF is exactly the corresponding state-space posterior
PMF.  This is the mathlib-object version of
`bayesPosterior_likelihoodOnState_eq_posteriorOnState`. -/
theorem bayesPosteriorPMF_likelihoodOnState_eq_stateBayesPosteriorPMF
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X)
    (hPrior : ∀ θ, 0 ≤ prior θ)
    (hLikelihood : ∀ θ z, 0 ≤ stateLikelihood θ z)
    (hEvidence :
      0 < BayesEvidence
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x) :
    BayesPosteriorPMF
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        hPrior
        (fun θ x => hLikelihood θ (state x))
        hEvidence
      =
      StateBayesPosteriorPMF
        prior
        stateLikelihood
        (state x)
        hPrior
        hLikelihood
        (by
          simpa [bayesEvidence_likelihoodOnState_eq_stateBayesEvidence]
            using hEvidence) := by
  ext θ
  simp [BayesPosteriorPMF, StateBayesPosteriorPMF, BayesPosterior,
    StateBayesPosterior, BayesNumerator, StateBayesNumerator,
    BayesEvidence, StateBayesEvidence, LikelihoodOnStateFamily]

end FormalProofs.OPT
