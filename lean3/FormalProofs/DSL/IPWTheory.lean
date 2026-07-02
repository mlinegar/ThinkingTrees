import FormalProbability.DSL.IPWTheory
import FormalProofs.OPT.ExpectationTheory
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Probability.Moments.Covariance
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
# FormalProofs/DSL/IPWTheory.lean

This file re-exports the core IPW formalization from `FormalProbability` and
adds TreePO-specific sampling utilities (tree propensities) plus a bridge
lemma connecting Bernoulli HT unbiasedness to `Exp`.

The heavy IPW machinery (weighted samples, Hajek/HT estimators, effective
sample size, variance bounds) lives in:
- `FormalProbability/DSL/IPWTheory.lean`
- `FormalProbability/DSL/SamplingTheory.lean`
- `FormalProbability/DSL/ClusteredVariance.lean`
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal ENNReal MeasureTheory
open MeasureTheory Measure ProbabilityTheory

set_option maxHeartbeats 400000
set_option synthInstance.maxHeartbeats 80000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Section 1: Bernoulli HT Unbiasedness → Exp

We repackage the FormalProbability Bernoulli HT result into the `Exp` notation
used throughout the FormalProofs OPT layer.
-/

section BernoulliExp

open MeasureTheory

variable {ι : Type*} [Fintype ι] [DecidableEq ι]
variable (p : PMF ι)
variable (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)

/-- HT estimator for an `Exp p f` target, using inclusion probabilities pi. -/
def htExpEstimator (f : ι → ℝ) (ω : ι → Bool) : ℝ :=
  htEstimator pi (fun i => (p i).toReal * f i) ω

/-- Exact unbiasedness for `Exp p f` under independent Bernoulli sampling. -/
lemma htExp_unbiased (f : ι → ℝ) :
    ∫ ω, htExpEstimator p pi f ω ∂bernoulliProductMeasure pi hpi_pos hpi_le = Exp p f := by
  classical
  -- FormalProbability gives: E[HT] = ∑ i, y i
  -- Here y i = p(i) * f(i), so the sum is Exp p f.
  simpa [htExpEstimator, Exp, tsum_fintype] using
    (ht_expectation (p := pi) (hp_pos := hpi_pos) (hp_le := hpi_le)
      (y := fun i => (p i).toReal * f i))

/-- Exact HT unbiasedness needs only correct logged marginal propensities, not
independent sampling. This is the design-based form used by the audit
robustness discussion: if the logged propensity `pi i` is the marginal
inclusion probability of unit `i`, inverse-propensity weighting targets
`Exp p f`. -/
theorem htExp_unbiased_of_logged_marginals
    (μ : Measure (ι → Bool)) [IsFiniteMeasure μ]
    (f : ι → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (h_marginal : ∀ i, ∫ ω, indicator i ω ∂μ = pi i) :
    ∫ ω, htExpEstimator p pi f ω ∂μ = Exp p f := by
  classical
  have h_int : ∀ i ∈ (Finset.univ : Finset ι),
      Integrable (fun ω => indicator i ω / pi i * ((p i).toReal * f i)) μ := by
    intro i _
    exact Integrable.of_finite
  have h_sum :
      ∫ ω, htExpEstimator p pi f ω ∂μ =
        ∑ i, ∫ ω, indicator i ω / pi i * ((p i).toReal * f i) ∂μ := by
    simpa [htExpEstimator, htEstimator] using
      (integral_finset_sum (μ := μ) (s := Finset.univ)
        (f := fun i ω => indicator i ω / pi i * ((p i).toReal * f i)) h_int)
  have h_term :
      ∀ i, ∫ ω, indicator i ω / pi i * ((p i).toReal * f i) ∂μ =
        (p i).toReal * f i := by
    intro i
    have hpi_ne : pi i ≠ 0 := ne_of_gt (hpi_pos i)
    calc
      ∫ ω, indicator i ω / pi i * ((p i).toReal * f i) ∂μ
          = (∫ ω, indicator i ω / pi i ∂μ) * ((p i).toReal * f i) := by
              simpa [mul_comm, mul_left_comm, mul_assoc] using
                (integral_mul_const (μ := μ) (r := (p i).toReal * f i)
                  (f := fun ω => indicator i ω / pi i))
      _ = ((∫ ω, indicator i ω ∂μ) / pi i) * ((p i).toReal * f i) := by
              congr 1
              simpa using (integral_div (μ := μ) (r := pi i)
                (f := fun ω => indicator i ω))
      _ = (pi i / pi i) * ((p i).toReal * f i) := by simp [h_marginal i]
      _ = (p i).toReal * f i := by field_simp [hpi_ne]
  calc
    ∫ ω, htExpEstimator p pi f ω ∂μ
        = ∑ i, ∫ ω, indicator i ω / pi i * ((p i).toReal * f i) ∂μ := h_sum
    _ = ∑ i, (p i).toReal * f i := by
          refine Finset.sum_congr rfl ?_
          intro i _
          exact h_term i
    _ = Exp p f := by simp [Exp, tsum_fintype]

/-!
## Audit Robustness: Constrained-Design Variance Proxy

The paper's adversarial-sampling appendix separates two facts:

1. HT unbiasedness only needs correct logged marginal propensities.
2. A variance bound additionally needs an independent Bernoulli design or an
   explicit covariance-control condition.

The definitions below expose that second condition directly. They are stated
for a uniform finite population because that is the appendix estimator
`N^{-1} ∑ᵢ ZᵢYᵢ/πᵢ`.
-/

/-- Uniform finite-population mean. -/
def uniformFiniteMean (y : ι → ℝ) : ℝ :=
  (1 / (Fintype.card ι : ℝ)) * ∑ i, y i

/-- HT estimator for the uniform finite-population mean. -/
def htUniformMeanEstimator (pi : ι → ℝ) (y : ι → ℝ) (ω : ι → Bool) : ℝ :=
  (1 / (Fintype.card ι : ℝ)) * ∑ i, indicator i ω / pi i * y i

/-- The independent-Bernoulli / covariance-controlled variance proxy:
`N^{-2} ∑ᵢ ((1-πᵢ)/πᵢ)Yᵢ²`. -/
def htUniformMeanVarianceProxy (pi : ι → ℝ) (y : ι → ℝ) : ℝ :=
  (1 / (Fintype.card ι : ℝ)^2) *
    ∑ i, ((1 - pi i) / pi i) * (y i)^2

/-- Logged-marginal unbiasedness for the uniform finite-population HT mean. -/
theorem htUniformMean_unbiased_of_logged_marginals
    (μ : Measure (ι → Bool)) [IsFiniteMeasure μ]
    (pi : ι → ℝ) (y : ι → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (h_marginal : ∀ i, ∫ ω, indicator i ω ∂μ = pi i) :
    ∫ ω, htUniformMeanEstimator pi y ω ∂μ = uniformFiniteMean y := by
  classical
  have h_int : ∀ i ∈ (Finset.univ : Finset ι),
      Integrable (fun ω => indicator i ω / pi i * y i) μ := by
    intro i _
    exact Integrable.of_finite
  have h_sum :
      ∫ ω, htUniformMeanEstimator pi y ω ∂μ =
        (1 / (Fintype.card ι : ℝ)) *
          ∑ i, ∫ ω, indicator i ω / pi i * y i ∂μ := by
    simp [htUniformMeanEstimator]
    rw [integral_const_mul]
    congr 1
    exact integral_finset_sum (μ := μ) (s := Finset.univ)
      (f := fun i ω => indicator i ω / pi i * y i) h_int
  have h_term :
      ∀ i, ∫ ω, indicator i ω / pi i * y i ∂μ = y i := by
    intro i
    have hpi_ne : pi i ≠ 0 := ne_of_gt (hpi_pos i)
    calc
      ∫ ω, indicator i ω / pi i * y i ∂μ
          = (∫ ω, indicator i ω / pi i ∂μ) * y i := by
              simpa [mul_comm, mul_left_comm, mul_assoc] using
                (integral_mul_const (μ := μ) (r := y i)
                  (f := fun ω => indicator i ω / pi i))
      _ = ((∫ ω, indicator i ω ∂μ) / pi i) * y i := by
              congr 1
              simpa using (integral_div (μ := μ) (r := pi i)
                (f := fun ω => indicator i ω))
      _ = (pi i / pi i) * y i := by simp [h_marginal i]
      _ = y i := by field_simp [hpi_ne]
  calc
    ∫ ω, htUniformMeanEstimator pi y ω ∂μ
        = (1 / (Fintype.card ι : ℝ)) *
          ∑ i, ∫ ω, indicator i ω / pi i * y i ∂μ := h_sum
    _ = (1 / (Fintype.card ι : ℝ)) * ∑ i, y i := by
          congr 1
          refine Finset.sum_congr rfl ?_
          intro i _
          exact h_term i
    _ = uniformFiniteMean y := by rfl

/-- Covariance-control condition connecting the actual variance of the HT mean
to the usual Bernoulli-design proxy. Independent Bernoulli sampling discharges
this condition; nonpositive cross-covariances also suffice. -/
def HTUniformMeanCovarianceControlled
    (μ : Measure (ι → Bool)) (pi : ι → ℝ) (y : ι → ℝ) : Prop :=
  ProbabilityTheory.variance (htUniformMeanEstimator pi y) μ ≤
    htUniformMeanVarianceProxy pi y

/-- Measurability of a Bernoulli product-coordinate indicator. -/
lemma indicator_aemeasurable_bernoulliProductMeasure
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (i : ι) :
    AEMeasurable (fun ω : ι → Bool => indicator i ω)
      (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  have h_bool : Measurable (fun b : Bool => if b then (1 : ℝ) else 0) :=
    Measurable.of_discrete
  have h_eval : Measurable (fun ω : ι → Bool => ω i) := measurable_pi_apply i
  simpa [indicator] using (h_bool.comp h_eval).aemeasurable

/-- Square-integrability of a Bernoulli product-coordinate indicator. -/
lemma indicator_memLp_bernoulliProductMeasure
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (i : ι) :
    MemLp (fun ω : ι → Bool => indicator i ω) 2
      (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  letI : IsProbabilityMeasure (bernoulliProductMeasure pi hpi_pos hpi_le) := by
    simpa [bernoulliProductMeasure, μi] using
      (Measure.pi.instIsProbabilityMeasure (μ := μi))
  refine MemLp.of_bound
    (indicator_aemeasurable_bernoulliProductMeasure
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) i).aestronglyMeasurable
    1 ?_
  apply ae_of_all
  intro ω
  by_cases h : ω i <;> simp [indicator, h]

/-- The square of a Bernoulli indicator has the same expectation as the
indicator. -/
lemma indicator_sq_integral_bernoulliProductMeasure
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (i : ι) :
    ∫ ω : ι → Bool, (indicator i ω)^2
        ∂bernoulliProductMeasure pi hpi_pos hpi_le = pi i := by
  have hfun : (fun ω : ι → Bool => (indicator i ω)^2) =
      fun ω => indicator i ω := by
    funext ω
    by_cases h : ω i <;> simp [indicator, h]
  rw [hfun]
  exact indicator_expectation (p := pi) (hp_pos := hpi_pos) (hp_le := hpi_le) i

/-- Variance of a Bernoulli product-coordinate indicator. -/
lemma indicator_variance_bernoulliProductMeasure
    (pi : ι → ℝ) (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (i : ι) :
    ProbabilityTheory.variance (fun ω : ι → Bool => indicator i ω)
      (bernoulliProductMeasure pi hpi_pos hpi_le) = pi i - (pi i)^2 := by
  let μ : Measure (ι → Bool) := bernoulliProductMeasure pi hpi_pos hpi_le
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  letI : IsProbabilityMeasure μ := by
    simpa [μ, bernoulliProductMeasure, μi] using
      (Measure.pi.instIsProbabilityMeasure (μ := μi))
  have hmem :=
    indicator_memLp_bernoulliProductMeasure
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) i
  calc
    ProbabilityTheory.variance (fun ω : ι → Bool => indicator i ω)
        (bernoulliProductMeasure pi hpi_pos hpi_le)
        = ∫ ω : ι → Bool, ((fun ω => indicator i ω)^2) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le
          - (∫ ω : ι → Bool, indicator i ω
              ∂bernoulliProductMeasure pi hpi_pos hpi_le)^2 := by
            simpa [μ] using (variance_eq_sub (μ := μ) hmem)
    _ = pi i - (pi i)^2 := by
            simp [indicator_sq_integral_bernoulliProductMeasure
              (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) i,
              indicator_expectation (p := pi) (hp_pos := hpi_pos) (hp_le := hpi_le) i]

/-- Independent Bernoulli product sampling satisfies the variance-proxy control
used by the constrained-design audit robustness theorem. -/
theorem htUniformMean_covarianceControlled_independent_bernoulli
    (pi : ι → ℝ) (y : ι → ℝ)
    (hcard_pos : 0 < (Fintype.card ι : ℝ))
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    HTUniformMeanCovarianceControlled
      (bernoulliProductMeasure pi hpi_pos hpi_le) pi y := by
  classical
  let μ : Measure (ι → Bool) := bernoulliProductMeasure pi hpi_pos hpi_le
  let N : ℝ := Fintype.card ι
  let X : ι → (ι → Bool) → ℝ :=
    fun i ω => (1 / N) * (indicator i ω / pi i * y i)
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  letI : IsProbabilityMeasure μ := by
    simpa [μ, bernoulliProductMeasure, μi] using
      (Measure.pi.instIsProbabilityMeasure (μ := μi))
  have hX_meas : ∀ i, AEMeasurable (X i) μ := by
    intro i
    have h_bool :
        Measurable (fun b : Bool =>
          (1 / N) * (((if b then (1 : ℝ) else 0) / pi i) * y i)) :=
      Measurable.of_discrete
    have h_eval : Measurable (fun ω : ι → Bool => ω i) := measurable_pi_apply i
    simpa [X, indicator, μ] using (h_bool.comp h_eval).aemeasurable
  have hX_mem : ∀ i, MemLp (X i) 2 μ := by
    intro i
    refine MemLp.of_bound (hX_meas i).aestronglyMeasurable
      (‖(1 / N) * (y i / pi i)‖) ?_
    apply ae_of_all
    intro ω
    by_cases h : ω i
    · simp [X, indicator, h, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]
    · have hnonneg : 0 ≤ |N|⁻¹ * (|y i| / |pi i|) := by positivity
      simpa [X, indicator, h, div_eq_mul_inv, Real.norm_eq_abs, abs_mul, abs_div,
        mul_comm, mul_left_comm, mul_assoc] using hnonneg
  have h_ind : iIndepFun X μ := by
    let Xsingle : ι → Bool → ℝ :=
      fun i b => (1 / N) * (((if b then (1 : ℝ) else 0) / pi i) * y i)
    have hsingle : ∀ i, AEMeasurable (Xsingle i) (μi i) := by
      intro i
      exact Measurable.of_discrete.aemeasurable
    have h := iIndepFun_pi (μ := μi) (X := Xsingle) hsingle
    simpa [X, Xsingle, μ, μi, bernoulliProductMeasure, indicator] using h
  have h_pair : Set.Pairwise (↑(Finset.univ : Finset ι))
      fun i j => X i ⟂ᵢ[μ] X j := by
    intro i _ j _ hne
    have hj_not : j ∉ ({i} : Finset ι) := by simp [hne.symm]
    have h :=
      h_ind.indepFun_finset_sum_of_notMem₀ hX_meas
        (s := ({i} : Finset ι)) (i := j) hj_not
    simpa using h
  have hest_eq : htUniformMeanEstimator pi y = fun ω => ∑ i, X i ω := by
    funext ω
    simp [htUniformMeanEstimator, X, N, Finset.mul_sum, mul_assoc]
  have hvar_sum' :
      ProbabilityTheory.variance (∑ i, X i) μ =
        ∑ i, ProbabilityTheory.variance (X i) μ := by
    simpa using
      (IndepFun.variance_sum (μ := μ) (X := X) (s := Finset.univ)
        (fun i _ => hX_mem i) h_pair)
  have hsum_fun : (∑ i, X i) = fun ω => ∑ i, X i ω := by
    ext ω
    simp
  have hvar_sum :
      ProbabilityTheory.variance (fun ω => ∑ i, X i ω) μ =
        ∑ i, ProbabilityTheory.variance (X i) μ := by
    rw [← hsum_fun]
    exact hvar_sum'
  have hvar_i : ∀ i,
      ProbabilityTheory.variance (X i) μ =
        (1 / N^2) * (((1 - pi i) / pi i) * (y i)^2) := by
    intro i
    have hpi_ne : pi i ≠ 0 := ne_of_gt (hpi_pos i)
    have hN_ne : N ≠ 0 := ne_of_gt hcard_pos
    have hX_eq : X i =
        fun ω => ((1 / N) * (y i / pi i)) * indicator i ω := by
      funext ω
      by_cases h : ω i <;>
        simp [X, indicator, h, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]
    calc
      ProbabilityTheory.variance (X i) μ
          = ProbabilityTheory.variance
              (fun ω => ((1 / N) * (y i / pi i)) * indicator i ω) μ := by
              rw [hX_eq]
      _ = ((1 / N) * (y i / pi i))^2 *
            ProbabilityTheory.variance (fun ω => indicator i ω) μ := by
              rw [variance_const_mul]
      _ = ((1 / N) * (y i / pi i))^2 * (pi i - (pi i)^2) := by
              rw [indicator_variance_bernoulliProductMeasure
                (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) i]
      _ = (1 / N^2) * (((1 - pi i) / pi i) * (y i)^2) := by
              field_simp [hpi_ne, hN_ne]
  have heq :
      ProbabilityTheory.variance
          (htUniformMeanEstimator pi y) (bernoulliProductMeasure pi hpi_pos hpi_le) =
        htUniformMeanVarianceProxy pi y := by
    calc
      ProbabilityTheory.variance
          (htUniformMeanEstimator pi y) (bernoulliProductMeasure pi hpi_pos hpi_le)
          = ProbabilityTheory.variance (fun ω => ∑ i, X i ω) μ := by
              rw [hest_eq]
      _ = ∑ i, ProbabilityTheory.variance (X i) μ := hvar_sum
      _ = ∑ i, (1 / N^2) * (((1 - pi i) / pi i) * (y i)^2) := by
              simp [hvar_i]
      _ = htUniformMeanVarianceProxy pi y := by
              simp [htUniformMeanVarianceProxy, N, Finset.mul_sum]
  exact le_of_eq heq

/-- The constrained-design variance proxy is controlled by the minimum
propensity. This is the algebraic core of the appendix bound. -/
theorem htUniformMeanVarianceProxy_le_constrained
    (pi : ι → ℝ) (y : ι → ℝ)
    (pi_min D_max : ℝ)
    (hcard_pos : 0 < (Fintype.card ι : ℝ))
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (hpi_min_pos : 0 < pi_min)
    (hpi_min_le_one : pi_min ≤ 1)
    (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (hD_nonneg : 0 ≤ D_max)
    (hy_bound : ∀ i, |y i| ≤ D_max) :
    htUniformMeanVarianceProxy pi y ≤
      (D_max^2 / (Fintype.card ι : ℝ)) * (1 / pi_min - 1) := by
  classical
  let N : ℝ := (Fintype.card ι : ℝ)
  let C : ℝ := (1 / pi_min - 1) * D_max^2
  have hN_pos : 0 < N := hcard_pos
  have hC_nonneg : 0 ≤ C := by
    have hratio : 0 ≤ 1 / pi_min - 1 := by
      have hone : 1 ≤ 1 / pi_min := one_le_one_div hpi_min_pos hpi_min_le_one
      linarith
    exact mul_nonneg hratio (sq_nonneg D_max)
  have hterm :
      ∀ i, ((1 - pi i) / pi i) * (y i)^2 ≤ C := by
    intro i
    have hratio_nonneg : 0 ≤ (1 - pi i) / pi i := by
      exact div_nonneg (by linarith [hpi_le i]) (le_of_lt (hpi_pos i))
    have hratio_le : (1 - pi i) / pi i ≤ 1 / pi_min - 1 := by
      have hpi_ne : pi i ≠ 0 := ne_of_gt (hpi_pos i)
      have hinv : 1 / pi i ≤ 1 / pi_min :=
        one_div_le_one_div_of_le hpi_min_pos (hpi_min_le i)
      have heq : (1 - pi i) / pi i = 1 / pi i - 1 := by
        field_simp [hpi_ne]
      rw [heq]
      linarith
    have hy_sq : (y i)^2 ≤ D_max^2 := by
      have habs_nonneg : 0 ≤ |y i| := abs_nonneg (y i)
      have hsq := mul_le_mul (hy_bound i) (hy_bound i) habs_nonneg hD_nonneg
      simpa [sq_abs, pow_two] using hsq
    have hy_sq_nonneg : 0 ≤ (y i)^2 := sq_nonneg (y i)
    calc
      ((1 - pi i) / pi i) * (y i)^2
          ≤ (1 / pi_min - 1) * (y i)^2 := by
              exact mul_le_mul_of_nonneg_right hratio_le hy_sq_nonneg
      _ ≤ (1 / pi_min - 1) * D_max^2 := by
              have hratio_min_nonneg : 0 ≤ 1 / pi_min - 1 :=
                le_trans hratio_nonneg hratio_le
              exact mul_le_mul_of_nonneg_left hy_sq hratio_min_nonneg
      _ = C := rfl
  have hsum :
      ∑ i, ((1 - pi i) / pi i) * (y i)^2 ≤ N * C := by
    calc
      ∑ i, ((1 - pi i) / pi i) * (y i)^2
          ≤ ∑ _i : ι, C := by
              exact Finset.sum_le_sum (fun i _ => hterm i)
      _ = N * C := by simp [N]
  have hscale_nonneg : 0 ≤ 1 / N^2 := by positivity
  calc
    htUniformMeanVarianceProxy pi y
        = (1 / N^2) * ∑ i, ((1 - pi i) / pi i) * (y i)^2 := by
            simp [htUniformMeanVarianceProxy, N]
    _ ≤ (1 / N^2) * (N * C) := by
            exact mul_le_mul_of_nonneg_left hsum hscale_nonneg
    _ = (D_max^2 / N) * (1 / pi_min - 1) := by
            have hN_ne : N ≠ 0 := ne_of_gt hN_pos
            calc
              (1 / N^2) * (N * C) = C / N := by
                field_simp [hN_ne]
              _ = ((1 / pi_min - 1) * D_max^2) / N := by
                rfl
              _ = (D_max^2 / N) * (1 / pi_min - 1) := by
                ring

/-- Actual variance bound for any design whose covariance structure is
controlled by the Bernoulli-design proxy. -/
theorem htUniformMean_variance_bound_of_constrained_design
    (μ : Measure (ι → Bool)) (pi : ι → ℝ) (y : ι → ℝ)
    (pi_min D_max : ℝ)
    (hcard_pos : 0 < (Fintype.card ι : ℝ))
    (hcontrol : HTUniformMeanCovarianceControlled μ pi y)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (hpi_min_pos : 0 < pi_min)
    (hpi_min_le_one : pi_min ≤ 1)
    (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (hD_nonneg : 0 ≤ D_max)
    (hy_bound : ∀ i, |y i| ≤ D_max) :
    ProbabilityTheory.variance (htUniformMeanEstimator pi y) μ ≤
      (D_max^2 / (Fintype.card ι : ℝ)) * (1 / pi_min - 1) := by
  exact hcontrol.trans
    (htUniformMeanVarianceProxy_le_constrained
      (pi := pi) (y := y) (pi_min := pi_min) (D_max := D_max)
      hcard_pos hpi_pos hpi_le hpi_min_pos hpi_min_le_one hpi_min_le hD_nonneg hy_bound)

/-- Independent-Bernoulli surface for the audit robustness bound. The sampling
measure is the existing Bernoulli product measure; covariance control is proved
from product independence by `htUniformMean_covarianceControlled_independent_bernoulli`. -/
theorem htUniformMean_variance_bound_of_independent_bernoulli
    (pi : ι → ℝ) (y : ι → ℝ)
    (pi_min D_max : ℝ)
    (hcard_pos : 0 < (Fintype.card ι : ℝ))
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (hpi_min_pos : 0 < pi_min)
    (hpi_min_le_one : pi_min ≤ 1)
    (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (hD_nonneg : 0 ≤ D_max)
    (hy_bound : ∀ i, |y i| ≤ D_max) :
    ProbabilityTheory.variance (htUniformMeanEstimator pi y)
        (bernoulliProductMeasure pi hpi_pos hpi_le) ≤
      (D_max^2 / (Fintype.card ι : ℝ)) * (1 / pi_min - 1) :=
  htUniformMean_variance_bound_of_constrained_design
    (μ := bernoulliProductMeasure pi hpi_pos hpi_le)
    (pi := pi) (y := y) (pi_min := pi_min) (D_max := D_max)
    hcard_pos
    (htUniformMean_covarianceControlled_independent_bernoulli
      (pi := pi) (y := y) hcard_pos hpi_pos hpi_le)
    hpi_pos hpi_le hpi_min_pos hpi_min_le_one hpi_min_le hD_nonneg hy_bound

lemma htExpEstimator_abs_le
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (hpi_pos : ∀ i, 0 < pi i)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ ω, |htExpEstimator p pi f ω| ≤ M / pi_min := by
  intro ω
  classical
  have h_sum :
      |∑ i, indicator i ω / pi i * ((p i).toReal * f i)| ≤
        ∑ i, |indicator i ω / pi i * ((p i).toReal * f i)| := by
    simpa [htExpEstimator, htEstimator] using
      (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset ι))
        (f := fun i => indicator i ω / pi i * ((p i).toReal * f i)))
  have h_term :
      ∀ i, |indicator i ω / pi i * ((p i).toReal * f i)| ≤
        (p i).toReal * (M / pi_min) := by
    intro i
    by_cases hω : ω i
    · have hpi_pos' : 0 < pi i := hpi_pos i
      have h_inv_le : (1 / pi i : ℝ) ≤ 1 / pi_min :=
        one_div_le_one_div_of_le hpi_min_pos (hpi_min_le i)
      have h_p_nonneg : 0 ≤ (p i).toReal := ENNReal.toReal_nonneg
      have h_abs :
          |indicator i ω / pi i * ((p i).toReal * f i)| =
            (p i).toReal * |f i| * (1 / pi i) := by
        simp [indicator, hω, abs_mul, abs_div, abs_of_pos hpi_pos',
          mul_comm, mul_left_comm, mul_assoc]
      calc
        |indicator i ω / pi i * ((p i).toReal * f i)|
            = (p i).toReal * |f i| * (1 / pi i) := h_abs
        _ ≤ (p i).toReal * M * (1 / pi i) := by
          have hf := hbound i
          have h1 : (p i).toReal * |f i| ≤ (p i).toReal * M :=
            mul_le_mul_of_nonneg_left hf h_p_nonneg
          exact mul_le_mul_of_nonneg_right h1 (one_div_nonneg.mpr (le_of_lt hpi_pos'))
        _ ≤ (p i).toReal * M * (1 / pi_min) := by
          have h2 : 0 ≤ (p i).toReal * M := mul_nonneg h_p_nonneg hM
          exact mul_le_mul_of_nonneg_left h_inv_le h2
        _ = (p i).toReal * (M / pi_min) := by
          ring
    · have h_p_nonneg : 0 ≤ (p i).toReal := ENNReal.toReal_nonneg
      have hpi_min_nonneg : 0 ≤ pi_min := le_of_lt hpi_min_pos
      have h_rhs_nonneg : 0 ≤ (p i).toReal * (M / pi_min) :=
        mul_nonneg h_p_nonneg (div_nonneg hM hpi_min_nonneg)
      simpa [indicator, hω] using h_rhs_nonneg
  have h_sum_le :
      ∑ i, |indicator i ω / pi i * ((p i).toReal * f i)| ≤
        ∑ i, (p i).toReal * (M / pi_min) := by
    refine Finset.sum_le_sum ?_
    intro i _; exact h_term i
  have h_sum_p : (∑ i, (p i).toReal) = 1 := by
    simpa [tsum_fintype] using (PMF.toReal_tsum_coe p)
  calc
    |htExpEstimator p pi f ω|
        = |∑ i, indicator i ω / pi i * ((p i).toReal * f i)| := by
            simp [htExpEstimator, htEstimator]
    _ ≤ ∑ i, |indicator i ω / pi i * ((p i).toReal * f i)| := h_sum
    _ ≤ ∑ i, (p i).toReal * (M / pi_min) := h_sum_le
    _ = (M / pi_min) * ∑ i, (p i).toReal := by
          have h :
              ∑ i, (p i).toReal * (M / pi_min) =
                ∑ i, (M / pi_min) * (p i).toReal := by
              refine Finset.sum_congr rfl ?_
              intro i _; ring
          rw [h, Finset.mul_sum]
    _ = M / pi_min := by simp [h_sum_p]

lemma htExpEstimator_abs_sq_le
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (hpi_pos : ∀ i, 0 < pi i)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ ω, |htExpEstimator p pi f ω|^2 ≤ (M / pi_min)^2 := by
  intro ω
  have h :=
    htExpEstimator_abs_le (p := p) (pi := pi) (f := f) (M := M) (hM := hM)
      (hbound := hbound) (hpi_pos := hpi_pos)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) ω
  have hB_nonneg : 0 ≤ M / pi_min :=
    div_nonneg hM (le_of_lt hpi_min_pos)
  have h_sq :
      |htExpEstimator p pi f ω| * |htExpEstimator p pi f ω| ≤
        (M / pi_min) * (M / pi_min) := by
    exact mul_le_mul h h (abs_nonneg _) hB_nonneg
  simpa [pow_two] using h_sq

lemma subgaussian_param_twoB (B : ℝ) (hB_nonneg : 0 ≤ B) :
    ((‖2 * B - -(2 * B)‖₊ / 2) ^ 2 : ℝ) = (2 * B)^2 := by
  have h4 : 0 ≤ 4 * B := by nlinarith [hB_nonneg]
  have hnn : (‖2 * B - -(2 * B)‖₊ : ℝ) = 4 * B := by
    calc
      (‖2 * B - -(2 * B)‖₊ : ℝ)
          = ‖2 * B - -(2 * B)‖ := by
              simp
      _ = |2 * B - -(2 * B)| := by simp [Real.norm_eq_abs]
      _ = |4 * B| := by ring_nf
      _ = 4 * B := by simp [abs_of_nonneg h4]
  have hnonneg : 0 ≤ 2 * B + 2 * B := by nlinarith [hB_nonneg]
  simp [NNReal.coe_pow, NNReal.coe_div, hnn, pow_two, abs_of_nonneg hnonneg]

def htExpTerm (f : ι → ℝ) (i : ι) (ω : ι → Bool) : ℝ :=
  indicator i ω / pi i * ((p i).toReal * f i)

def htExpCenteredTerm (f : ι → ℝ) (i : ι) (ω : ι → Bool) : ℝ :=
  htExpTerm (p := p) (pi := pi) f i ω - (p i).toReal * f i

lemma htExpTerm_measurable (f : ι → ℝ) (i : ι) :
    AEMeasurable (fun ω => htExpTerm (p := p) (pi := pi) f i ω)
      (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  let μ : Measure (ι → Bool) := bernoulliProductMeasure pi hpi_pos hpi_le
  have h_bool :
      Measurable (fun b : Bool =>
        ((if b then (1 : ℝ) else 0) / pi i) * ((p i).toReal * f i)) :=
    Measurable.of_discrete
  have h_eval : Measurable (fun ω : ι → Bool => ω i) := measurable_pi_apply i
  have h_comp :
      Measurable (fun ω : ι → Bool =>
        ((if ω i then (1 : ℝ) else 0) / pi i) * ((p i).toReal * f i)) :=
    h_bool.comp h_eval
  simpa [htExpTerm, indicator, μ] using h_comp.aemeasurable

lemma htExpCenteredTerm_measurable (f : ι → ℝ) (i : ι) :
    AEMeasurable (fun ω => htExpCenteredTerm (p := p) (pi := pi) f i ω)
      (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  have hterm :=
    htExpTerm_measurable (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f i
  have hconst :
      AEMeasurable (fun _ : (ι → Bool) => (p i).toReal * f i)
        (bernoulliProductMeasure pi hpi_pos hpi_le) := aemeasurable_const
  simpa [htExpCenteredTerm] using hterm.sub hconst

lemma htExpTerm_abs_le
    (hpi_pos : ∀ i, 0 < pi i)
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ i ω, |htExpTerm (p := p) (pi := pi) f i ω| ≤ M / pi_min := by
  intro i ω
  by_cases hω : ω i
  · have hpi_pos_i : 0 < pi i := hpi_pos i
    have h_inv_le : (1 / pi i : ℝ) ≤ 1 / pi_min :=
      one_div_le_one_div_of_le hpi_min_pos (hpi_min_le i)
    have h_p_nonneg : 0 ≤ (p i).toReal := ENNReal.toReal_nonneg
    have hp_le_one : (p i).toReal ≤ 1 := by
      have h : p i ≤ ENNReal.ofReal 1 := by
        simpa using (p.coe_le_one i)
      exact ENNReal.toReal_le_of_le_ofReal (by norm_num) h
    calc
      |htExpTerm (p := p) (pi := pi) f i ω|
          = (p i).toReal * |f i| * (1 / pi i) := by
              simp [htExpTerm, indicator, hω, abs_mul, abs_div, abs_of_pos hpi_pos_i,
                mul_comm, mul_left_comm, mul_assoc]
      _ ≤ (p i).toReal * M * (1 / pi i) := by
            have h1 : (p i).toReal * |f i| ≤ (p i).toReal * M :=
              mul_le_mul_of_nonneg_left (hbound i) h_p_nonneg
            exact mul_le_mul_of_nonneg_right h1 (one_div_nonneg.mpr (le_of_lt hpi_pos_i))
      _ ≤ (p i).toReal * M * (1 / pi_min) := by
            have h2 : 0 ≤ (p i).toReal * M := mul_nonneg h_p_nonneg hM
            exact mul_le_mul_of_nonneg_left h_inv_le h2
      _ = (p i).toReal * (M / pi_min) := by
            simp [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]
      _ ≤ 1 * (M / pi_min) := by
            exact mul_le_mul_of_nonneg_right hp_le_one (div_nonneg hM (le_of_lt hpi_min_pos))
      _ = M / pi_min := by simp
  · have hB : 0 ≤ M / pi_min := div_nonneg hM (le_of_lt hpi_min_pos)
    simp [htExpTerm, indicator, hω, hB]

lemma htExpExpectation_abs_le
    (hpi_le : ∀ i, pi i ≤ 1)
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ i, |(p i).toReal * f i| ≤ M / pi_min := by
  intro i
  have h_p_nonneg : 0 ≤ (p i).toReal := ENNReal.toReal_nonneg
  have hp_le_one : (p i).toReal ≤ 1 := by
    have h : p i ≤ ENNReal.ofReal 1 := by
      simpa using (p.coe_le_one i)
    exact ENNReal.toReal_le_of_le_ofReal (by norm_num) h
  have hpi_min_le_one : pi_min ≤ 1 := le_trans (hpi_min_le i) (hpi_le i)
  have h_one_le : 1 ≤ (1 / pi_min) := one_le_one_div hpi_min_pos hpi_min_le_one
  have hM_le_B : M ≤ M / pi_min := by
    have hM_mul : M * 1 ≤ M * (1 / pi_min) := mul_le_mul_of_nonneg_left h_one_le hM
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hM_mul
  calc
    |(p i).toReal * f i| = (p i).toReal * |f i| := by
      simp [abs_mul, h_p_nonneg]
    _ ≤ (p i).toReal * M := by
      exact mul_le_mul_of_nonneg_left (hbound i) h_p_nonneg
    _ ≤ 1 * M := by
      exact mul_le_mul_of_nonneg_right hp_le_one hM
    _ = M := by ring
    _ ≤ M / pi_min := hM_le_B

lemma htExpCenteredTerm_mem_Icc
    (hpi_pos : ∀ i, 0 < pi i) (hpi_le : ∀ i, pi i ≤ 1)
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ i ω,
      htExpCenteredTerm (p := p) (pi := pi) f i ω ∈
        Set.Icc (-(2 * (M / pi_min))) (2 * (M / pi_min)) := by
  intro i ω
  have hX := htExpTerm_abs_le (p := p) (pi := pi) (hpi_pos := hpi_pos)
    (f := f) (M := M) (hM := hM) (hbound := hbound)
    (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i ω
  have hEX := htExpExpectation_abs_le (p := p) (pi := pi) (hpi_le := hpi_le)
    (f := f) (M := M) (hM := hM) (hbound := hbound)
    (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i
  have hY_abs : |htExpCenteredTerm (p := p) (pi := pi) f i ω| ≤ 2 * (M / pi_min) := by
    calc
      |htExpCenteredTerm (p := p) (pi := pi) f i ω|
          = |htExpTerm (p := p) (pi := pi) f i ω - (p i).toReal * f i| := by rfl
      _ ≤ |htExpTerm (p := p) (pi := pi) f i ω| + |(p i).toReal * f i| := by
          simpa [sub_eq_add_neg] using
            (abs_add_le (htExpTerm (p := p) (pi := pi) f i ω) (-(p i).toReal * f i))
      _ ≤ (M / pi_min) + (M / pi_min) := by exact add_le_add hX hEX
      _ = 2 * (M / pi_min) := by ring
  exact (abs_le.mp hY_abs)

lemma htExpCenteredTerm_indep (f : ι → ℝ) :
    iIndepFun (fun i ω => htExpCenteredTerm (p := p) (pi := pi) f i ω)
      (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  let g : ι → Bool → ℝ :=
    fun i b =>
      ((if b then (1 : ℝ) else 0) / pi i) * ((p i).toReal * f i) -
        (p i).toReal * f i
  have hg : ∀ i, AEMeasurable (g i) (μi i) := by
    intro i
    exact (Measurable.of_discrete).aemeasurable
  simpa [htExpCenteredTerm, htExpTerm, μi, bernoulliProductMeasure, indicator, g] using
    (iIndepFun_pi (μ := μi) (X := g) hg)

lemma htExpTerm_mem_Icc
    (hpi_pos : ∀ i, 0 < pi i)
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ i ω,
      htExpTerm (p := p) (pi := pi) f i ω ∈
        Set.Icc (-(M / pi_min)) (M / pi_min) := by
  intro i ω
  have h :=
    htExpTerm_abs_le (p := p) (pi := pi) (hpi_pos := hpi_pos)
      (f := f) (M := M) (hM := hM) (hbound := hbound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i ω
  exact (abs_le.mp h)

lemma htExpTerm_integrable
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) (i : ι) :
    Integrable (fun ω => htExpTerm (p := p) (pi := pi) f i ω)
      (bernoulliProductMeasure pi hpi_pos hpi_le) := by
  let μ : Measure (ι → Bool) := bernoulliProductMeasure pi hpi_pos hpi_le
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  letI : IsProbabilityMeasure μ := by
    simpa [μ, μi, bernoulliProductMeasure] using
      (Measure.pi.instIsProbabilityMeasure (μ := μi))
  apply Integrable.of_mem_Icc (-(M / pi_min)) (M / pi_min)
  · simpa [μ] using
      (htExpTerm_measurable (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f i)
  · exact ae_of_all μ
      (htExpTerm_mem_Icc (p := p) (pi := pi) (hpi_pos := hpi_pos)
        (f := f) (M := M) (hM := hM) (hbound := hbound)
        (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i)

lemma htExpCenteredTerm_mean_zero
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i) :
    ∀ i,
      ∫ ω, htExpCenteredTerm (p := p) (pi := pi) f i ω ∂
        bernoulliProductMeasure pi hpi_pos hpi_le = 0 := by
  intro i
  let μ : Measure (ι → Bool) := bernoulliProductMeasure pi hpi_pos hpi_le
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  letI : IsProbabilityMeasure μ := by
    simpa [μ, μi, bernoulliProductMeasure] using
      (Measure.pi.instIsProbabilityMeasure (μ := μi))
  have hX_integrable :
      Integrable (fun ω => htExpTerm (p := p) (pi := pi) f i ω) μ := by
    dsimp [μ]
    exact
      (htExpTerm_integrable (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
        (f := f) (M := M) (hM := hM) (hbound := hbound)
        (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i)
  have hconst_integrable :
      Integrable (fun _ : (ι → Bool) => (p i).toReal * f i) μ :=
    integrable_const _
  have h_ind : ∫ ω, indicator i ω ∂μ = pi i := by
    simpa [μ] using
      (indicator_expectation (p := pi) (hp_pos := hpi_pos) (hp_le := hpi_le) i)
  have hpi_ne : (pi i) ≠ 0 := ne_of_gt (hpi_pos i)
  have h_int_ind : ∫ ω, indicator i ω / pi i ∂μ = 1 := by
    calc
      ∫ ω, indicator i ω / pi i ∂μ
          = (∫ ω, indicator i ω ∂μ) / pi i := by
              simpa using (integral_div (μ := μ) (r := pi i)
                (f := fun ω => indicator i ω))
      _ = pi i / pi i := by simp [h_ind]
      _ = 1 := by field_simp [hpi_ne]
  have hX_int : ∫ ω, htExpTerm (p := p) (pi := pi) f i ω ∂μ =
      (p i).toReal * f i := by
    calc
      ∫ ω, htExpTerm (p := p) (pi := pi) f i ω ∂μ
          = (∫ ω, indicator i ω / pi i ∂μ) * ((p i).toReal * f i) := by
              simpa [htExpTerm] using
                (integral_mul_const (μ := μ) (r := (p i).toReal * f i)
                  (f := fun ω => indicator i ω / pi i))
      _ = (p i).toReal * f i := by simp [h_int_ind]
  have h_one : μ.real Set.univ = 1 := probReal_univ
  have h_sub :
      MeasureTheory.integral μ
        (fun ω : (ι → Bool) => htExpTerm (p := p) (pi := pi) f i ω - (p i).toReal * f i) =
        MeasureTheory.integral μ (fun ω : (ι → Bool) => htExpTerm (p := p) (pi := pi) f i ω) -
          MeasureTheory.integral μ (fun _ : (ι → Bool) => (p i).toReal * f i) := by
    exact
      (integral_sub (μ := μ) (f := fun ω => htExpTerm (p := p) (pi := pi) f i ω)
        (g := fun _ => (p i).toReal * f i) hX_integrable hconst_integrable)
  calc
    ∫ ω, htExpCenteredTerm (p := p) (pi := pi) f i ω ∂μ
        = MeasureTheory.integral μ
            (fun ω : (ι → Bool) => htExpTerm (p := p) (pi := pi) f i ω - (p i).toReal * f i) := by
              rfl
    _ = MeasureTheory.integral μ (fun ω : (ι → Bool) => htExpTerm (p := p) (pi := pi) f i ω) -
          MeasureTheory.integral μ (fun _ : (ι → Bool) => (p i).toReal * f i) := by
              exact h_sub
    _ = (p i).toReal * f i - (p i).toReal * f i := by
          simp [hX_int, integral_const, h_one]
    _ = 0 := by ring

lemma htExpEstimator_hoeffding_bound
    (f : ι → ℝ) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ i, |f i| ≤ M)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (ε : ℝ) (hε : 0 < ε) :
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≤
      2 * Real.exp (- ε^2 / (8 * (Fintype.card ι) * (M / pi_min)^2)) := by
  classical
  let μ : Measure (ι → Bool) := bernoulliProductMeasure pi hpi_pos hpi_le
  let μi : ι → Measure Bool := fun i => bernoulliMeasure pi hpi_pos hpi_le i
  let B : ℝ := M / pi_min
  have hB_nonneg : 0 ≤ B := div_nonneg hM (le_of_lt hpi_min_pos)
  letI : ∀ i, IsProbabilityMeasure (μi i) := by
    intro i
    dsimp [μi, bernoulliMeasure]
    infer_instance
  letI : IsProbabilityMeasure μ := by
    simpa [μ, μi, bernoulliProductMeasure] using
      (Measure.pi.instIsProbabilityMeasure (μ := μi))
  let Y : ι → (ι → Bool) → ℝ :=
    fun i ω => htExpCenteredTerm (p := p) (pi := pi) f i ω

  -- Measurability / independence / bounds for centered terms
  have hY_meas : ∀ i, AEMeasurable (Y i) μ := by
    intro i
    simpa [Y, μ] using
      (htExpCenteredTerm_measurable (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f i)
  have h_indep_Y : iIndepFun Y μ := by
    simpa [Y, μ] using
      (htExpCenteredTerm_indep (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f)
  have hY_bound_ae :
      ∀ i, ∀ᵐ ω ∂μ, Y i ω ∈ Set.Icc (-(2 * B)) (2 * B) := by
    intro i
    apply ae_of_all μ
    intro ω
    have h :=
      htExpCenteredTerm_mem_Icc (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
        (f := f) (M := M) (hM := hM) (hbound := hbound)
        (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i ω
    simpa [B, Y] using h
  have hY_mean : ∀ i, ∫ ω, Y i ω ∂μ = 0 := by
    intro i
    simpa [Y, μ] using
      (htExpCenteredTerm_mean_zero (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
        (f := f) (M := M) (hM := hM) (hbound := hbound)
        (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) i)

  -- Sub-Gaussian for Y with parameter (2B)^2
  have hY_subG :
      ∀ i, HasSubgaussianMGF (Y i) ((‖2 * B - -(2 * B)‖₊ / 2) ^ 2) μ := by
    intro i
    exact
      hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero (hY_meas i) (hY_bound_ae i) (hY_mean i)

  let c : ℝ≥0 := (‖2 * B - -(2 * B)‖₊ / 2) ^ 2
  have hparam : (c : ℝ) = (2 * B)^2 :=
    subgaussian_param_twoB (B := B) hB_nonneg

  -- Hoeffding right tail
  have h_right : μ.real {ω | ε ≤ ∑ i, Y i ω} ≤
      Real.exp (- ε^2 / (8 * (Fintype.card ι) * B^2)) := by
    have hε_nonneg : 0 ≤ ε := le_of_lt hε
    have h :=
      HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun h_indep_Y
        (c := fun _ => c) (s := Finset.univ) (fun i _ => hY_subG i) hε_nonneg
    have hsum :
        (∑ i : ι, (c : ℝ≥0)) = (Fintype.card ι) * (c : ℝ≥0) := by
      simp [Finset.sum_const, Finset.card_univ]
    have h' : μ.real {ω | ε ≤ ∑ i, Y i ω} ≤
        Real.exp (- ε^2 / (2 * (Fintype.card ι) * (2 * B)^2)) := by
      simpa [hsum, hparam, mul_assoc] using h
    have hden : (2 * (Fintype.card ι) * (2 * B)^2 : ℝ) = 8 * (Fintype.card ι) * B^2 := by
      ring
    simpa [hden] using h'

  -- Hoeffding left tail
  have h_left : μ.real {ω | ε ≤ -∑ i, Y i ω} ≤
      Real.exp (- ε^2 / (8 * (Fintype.card ι) * B^2)) := by
    have hε_nonneg : 0 ≤ ε := le_of_lt hε
    have h_neg :
        ∀ i, HasSubgaussianMGF (fun ω => - (Y i ω))
          ((‖2 * B - -(2 * B)‖₊ / 2) ^ 2) μ :=
      fun i => (hY_subG i).neg
    have h_indep_neg : iIndepFun (fun i ω => - (Y i ω)) μ := by
      have hneg_meas : ∀ i : ι, Measurable (fun x : ℝ => -x) := fun _ => measurable_neg
      exact h_indep_Y.comp (fun _ => (- ·)) hneg_meas
    have h :=
      HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun h_indep_neg
        (c := fun _ => c) (s := Finset.univ) (fun i _ => h_neg i) hε_nonneg
    have hsum :
        (∑ i : ι, (c : ℝ≥0)) = (Fintype.card ι) * (c : ℝ≥0) := by
      simp [Finset.sum_const, Finset.card_univ]
    have h' : μ.real {ω | ε ≤ -∑ i, Y i ω} ≤
        Real.exp (- ε^2 / (2 * (Fintype.card ι) * (2 * B)^2)) := by
      simpa [hsum, hparam, mul_assoc] using h
    have hden : (2 * (Fintype.card ι) * (2 * B)^2 : ℝ) = 8 * (Fintype.card ι) * B^2 := by
      ring
    simpa [hden] using h'

  -- Union bound
  have h_sum : ∀ ω, (∑ i, Y i ω) = htExpEstimator p pi f ω - Exp p f := by
    intro ω
    simp [Y, htExpCenteredTerm, htExpTerm, htExpEstimator, htEstimator, Exp, tsum_fintype,
      Finset.sum_sub_distrib, indicator, mul_comm, mul_left_comm, mul_assoc]
  have h_set_subset :
      {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ⊆
        {ω | ε ≤ ∑ i, Y i ω} ∪ {ω | ε ≤ -∑ i, Y i ω} := by
    intro ω hω
    have hω' : |∑ i, Y i ω| ≥ ε := by
      simpa [h_sum ω] using hω
    simp only [Set.mem_union, Set.mem_setOf_eq]
    rcases le_or_gt (∑ i, Y i ω) 0 with hneg | hpos
    · right
      have : ε ≤ -∑ i, Y i ω := by
        rw [abs_of_nonpos hneg] at hω'
        linarith
      exact this
    · left
      have : ε ≤ ∑ i, Y i ω := by
        rw [abs_of_pos hpos] at hω'
        linarith
      exact this
  have h_bound :
      μ.real {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≤
        μ.real ({ω | ε ≤ ∑ i, Y i ω} ∪ {ω | ε ≤ -∑ i, Y i ω}) := by
    exact measureReal_mono h_set_subset
  calc
    μ.real {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε}
        ≤ μ.real ({ω | ε ≤ ∑ i, Y i ω} ∪ {ω | ε ≤ -∑ i, Y i ω}) := h_bound
    _ ≤ μ.real {ω | ε ≤ ∑ i, Y i ω} + μ.real {ω | ε ≤ -∑ i, Y i ω} := by
        exact measureReal_union_le _ _
    _ ≤ Real.exp (- ε^2 / (8 * (Fintype.card ι) * B^2)) +
        Real.exp (- ε^2 / (8 * (Fintype.card ι) * B^2)) := by
        exact add_le_add h_right h_left
    _ = 2 * Real.exp (- ε^2 / (8 * (Fintype.card ι) * (M / pi_min)^2)) := by
        simp [B, two_mul, mul_comm, mul_left_comm, mul_assoc]

lemma htExpEstimator_hoeffding_bound_unit
    (f : ι → ℝ) (hbound : ∀ i, |f i| ≤ (1 : ℝ))
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (ε : ℝ) (hε : 0 < ε) :
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≤
      2 * Real.exp (- ε^2 / (8 * (Fintype.card ι) * (1 / pi_min)^2)) := by
  have hM : 0 ≤ (1 : ℝ) := by norm_num
  simpa using
    (htExpEstimator_hoeffding_bound (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
      (f := f) (M := 1) (hM := hM) (hbound := hbound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le)
      (ε := ε) (hε := hε))

lemma htExpEstimator_hoeffding_bound_indicator
    (f : ι → ℝ) (h0 : ∀ i, 0 ≤ f i) (h1 : ∀ i, f i ≤ (1 : ℝ))
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (ε : ℝ) (hε : 0 < ε) :
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |htExpEstimator p pi f ω - Exp p f| ≥ ε} ≤
      2 * Real.exp (- ε^2 / (8 * (Fintype.card ι) * (1 / pi_min)^2)) := by
  have hbound : ∀ i, |f i| ≤ (1 : ℝ) := by
    intro i
    have h0i := h0 i
    have h1i := h1 i
    have habs : |f i| = f i := abs_of_nonneg h0i
    simpa [habs] using h1i
  exact
    htExpEstimator_hoeffding_bound_unit (p := p) (pi := pi) (hpi_pos := hpi_pos)
      (hpi_le := hpi_le) (f := f) (hbound := hbound)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le)
      (ε := ε) (hε := hε)


end BernoulliExp

/-!
## Section 2: Self-Normalized (Empirical Bernstein) Bounds

These definitions are design-agnostic and apply to any weighted sample list.
We package the actual concentration inequality as an explicit assumption
structure, following the DSL style used elsewhere in the repo.
-/

section SelfNormalized

/-- Reusable empirical-Bernstein bound package (center + radius). -/
structure EBBound where
  center : ℝ
  radius : ℝ

namespace EBBound

/-- Lower endpoint of the confidence interval. -/
def lower (b : EBBound) : ℝ := b.center - b.radius

/-- Upper endpoint of the confidence interval. -/
def upper (b : EBBound) : ℝ := b.center + b.radius

/-- Closed confidence interval corresponding to the bound. -/
def interval (b : EBBound) : ℝ × ℝ := (b.lower, b.upper)

end EBBound

/-- Build an empirical-Bernstein bound from weighted samples. -/
def empiricalBernsteinBound
    (samples : List (WeightedSample ℝ)) (δ range : ℝ) : EBBound :=
  { center := weightedMean samples
    radius := empiricalBernsteinRadius samples δ range }

/-- CI helper is exactly the interval from `empiricalBernsteinBound`. -/
lemma empiricalBernsteinCI_eq_interval
    (samples : List (WeightedSample ℝ)) (δ range : ℝ) :
    empiricalBernsteinCI samples δ range =
      (empiricalBernsteinBound samples δ range).interval := by
  rfl

/-- ENNReal-form wrapper for `EmpiricalBernsteinAxioms.bound`.

`FormalProbability` exposes this axiom in `μ.real` form. For DSL theorems that
compose via measure union bounds, this converts it back to `μ ≤ ENNReal.ofReal`.
-/
theorem empiricalBernstein_bound_ennreal
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List (WeightedSample ℝ))
    (mean_true range : ℝ)
    (axioms : EmpiricalBernsteinAxioms μ samples mean_true range)
    (δ : ℝ) (hδ_pos : 0 < δ) (hδ_lt : δ < 1) :
    μ {ω | |hajekEstimator (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (samples ω) δ range} ≤ ENNReal.ofReal δ := by
  let S : Set Ω := {ω |
    |hajekEstimator (samples ω) - mean_true| ≥
      empiricalBernsteinRadius (samples ω) δ range}
  have h_real : μ.real S ≤ δ := by
    simpa [S, ge_iff_le] using axioms.bound δ hδ_pos hδ_lt
  have h_toReal : (μ S).toReal ≤ δ := by
    simpa [S, measureReal_def] using h_real
  have h_ofReal : ENNReal.ofReal ((μ S).toReal) ≤ ENNReal.ofReal δ :=
    ENNReal.ofReal_le_ofReal h_toReal
  have hS_ne_top : μ S ≠ ∞ := measure_ne_top μ S
  simpa [S, ENNReal.ofReal_toReal hS_ne_top] using h_ofReal

/-- Direct-event form of empirical-Bernstein concentration (axiom-free interface).

This theorem is intentionally thin: callers provide the EB event bound directly,
which can come from any concentration proof or external certificate. -/
theorem empiricalBernstein_bound_ennreal_of_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List (WeightedSample ℝ))
    (mean_true range : ℝ)
    (δ : ℝ)
    (h_event :
      μ {ω | |hajekEstimator (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (samples ω) δ range} ≤ ENNReal.ofReal δ) :
    μ {ω | |hajekEstimator (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (samples ω) δ range} ≤ ENNReal.ofReal δ := by
  exact h_event

end SelfNormalized

/-!
## Section 2: Three-Stage Propensity for Tree Sampling
-/

/-- Three-stage propensity for tree-based sampling.

Sampling proceeds as:
1. Sample document with probability p_doc
2. Sample node within document with probability p_node_given_doc
3. Sample action/pair at node with probability p_action_given_node

Joint propensity: p = p_doc × p_{n|d} × p_{a|n}
-/
structure TreePropensity where
  p_doc : ℝ
  p_node_given_doc : ℝ
  p_action_given_node : ℝ
  h_doc_pos : 0 < p_doc
  h_node_pos : 0 < p_node_given_doc
  h_action_pos : 0 < p_action_given_node

namespace TreePropensity

/-- Joint propensity: product of all three stages -/
def joint (p : TreePropensity) : ℝ :=
  p.p_doc * p.p_node_given_doc * p.p_action_given_node

lemma joint_pos (p : TreePropensity) : 0 < p.joint := by
  unfold joint
  exact mul_pos (mul_pos p.h_doc_pos p.h_node_pos) p.h_action_pos

/-- Convert TreePropensity to WeightedSample -/
def toWeightedSample (p : TreePropensity) (value : ℝ) : WeightedSample ℝ :=
  ⟨value, p.joint, p.joint_pos⟩

end TreePropensity
