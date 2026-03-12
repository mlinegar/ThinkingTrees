import FormalProbability.DSL.IPWTheory
import FormalProofs.OPT.ExpectationTheory
import Mathlib.Probability.Moments.SubGaussian
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
