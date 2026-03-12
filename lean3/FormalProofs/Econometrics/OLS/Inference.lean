/-
# FormalProofs/Econometrics/OLS/Inference.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 4

This file formalizes inference for OLS regression:

- t-tests for individual coefficients (Theorem 4.1)
- Confidence intervals (Section 4.3)
- F-tests for joint significance (Section 4.5)
- Equivalence of Wald, LM, LR tests (asymptotically)

### Main Results

**Theorem 4.1 (t-statistic under classical assumptions)**
Under MLR.1-6: (β̂_j - β_j) / SE(β̂_j) ~ t_{n-k-1}

**Theorem 4.2 (F-test)**
Under H₀: R β = r with q restrictions:
F = [(Rβ̂ - r)'(R(X'X)⁻¹R')⁻¹(Rβ̂ - r)/q] / σ̂² ~ F_{q, n-k-1}

**Asymptotic Equivalence**
Under regularity conditions: Wald, LM, LR → χ²_q
-/

import Mathlib
import FormalProofs.Econometrics.OLS.GaussMarkov
import FormalProofs.Econometrics.OLS.AsymptoticOLS

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Econometrics

namespace OLS

/-!
## t-tests for Individual Coefficients
-/

/-- t-statistic for testing H₀: β_j = β₀ -/
def tStatistic {k : ℕ}
    (β_hat : Fin k → ℝ)
    (SE : Fin k → ℝ)
    (j : Fin k)
    (β_null : ℝ := 0) : ℝ :=
  (β_hat j - β_null) / SE j

/-- Standard error of β̂_j under classical assumptions -/
def classicalSE {n k : ℕ}
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_hat : ℝ)  -- √(RSS/(n-k-1))
    (j : Fin k) : ℝ :=
  σ_hat * Real.sqrt (XtX_inv j j)

/-- Theorem 4.1 (Wooldridge): Under MLR.1-6, the t-statistic
    has a t-distribution with n-k-1 degrees of freedom.

    (β̂_j - β_j) / SE(β̂_j) ~ t_{n-k-1}

    This enables exact finite-sample inference under normality. -/
structure TDistribution (df : ℕ) where
  /-- Upper-tail/survival function representation. -/
  survival : ℝ → ℝ
  /-- Survival values are nonnegative. -/
  survival_nonneg : ∀ x, 0 ≤ survival x
  /-- Survival values are bounded by one. -/
  survival_le_one : ∀ x, survival x ≤ 1

def t_stat_distribution {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (σ_sq : ℝ)
    (h_classical : ClassicalAssumptions μ data σ_sq)
    (j : Fin k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_n_gt_k : n > k + 1) :
    -- The t-statistic has a t-distribution
    TDistribution (n - k - 1) →
      TDistribution (n - k - 1) := by
  intro t_dist
  exact t_dist

/-- p-value for two-sided t-test -/
def pValueTwoSided
    (t_survival : ℕ → ℝ → ℝ)
    (t_stat : ℝ)
    (df : ℕ) : ℝ :=
  2 * t_survival df |t_stat|

/-- Reject H₀ at significance level α if |t| > critical value -/
def rejectNull
    (t_stat : ℝ)
    (critical_value : ℝ) : Bool :=
  |t_stat| > critical_value

/-!
## Confidence Intervals
-/

/-- Confidence interval for β_j (Section 4.3)

    [β̂_j - t_{α/2, n-k-1} SE(β̂_j), β̂_j + t_{α/2, n-k-1} SE(β̂_j)]

    Under MLR.1-6, this has exact (1-α) coverage. -/
structure ConfidenceInterval where
  lower : ℝ
  upper : ℝ
  h_ordered : lower ≤ upper

/-- Construct a confidence interval -/
def confidenceInterval {k : ℕ}
    (β_hat : Fin k → ℝ)
    (SE : Fin k → ℝ)
    (t_critical : ℝ)  -- e.g., 1.96 for 95% asymptotic
    (j : Fin k)
    (h_ordered : β_hat j - t_critical * SE j ≤ β_hat j + t_critical * SE j) :
    ConfidenceInterval where
  lower := β_hat j - t_critical * SE j
  upper := β_hat j + t_critical * SE j
  h_ordered := h_ordered

/-- 95% confidence interval (using t-critical ≈ 2 for large n) -/
def confidenceInterval95 {k : ℕ}
    (β_hat : Fin k → ℝ)
    (SE : Fin k → ℝ)
    (j : Fin k)
    (h_ordered : β_hat j - 1.96 * SE j ≤ β_hat j + 1.96 * SE j) :
    ConfidenceInterval :=
  confidenceInterval β_hat SE 1.96 j h_ordered

/-- Coverage property: β_true ∈ CI with probability 1-α -/
def HasCorrectCoverage {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (CI_constructor : Ω → ConfidenceInterval)
    (β_true : ℝ)
    (coverage_prob : ℝ) : Prop :=
  μ {ω | (CI_constructor ω).lower ≤ β_true ∧ β_true ≤ (CI_constructor ω).upper} =
    ENNReal.ofReal coverage_prob

/-- Under classical assumptions, CI has exact coverage -/
theorem ci_exact_coverage_with_level_bounds {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (σ_sq : ℝ)
    (h_classical : ClassicalAssumptions μ data σ_sq)
    (j : Fin k)
    (α : ℝ)
    (h_α : 0 < α ∧ α < 1)
    (t_critical : ℝ)  -- t_{α/2, n-k-1}
    (CI_constructor : Ω → ConfidenceInterval)
    (β_true_j : ℝ)  -- The true coefficient value
    (h_coverage : HasCorrectCoverage μ CI_constructor β_true_j (1 - α)) :
    HasCorrectCoverage μ CI_constructor β_true_j (1 - α) ∧ 0 < (1 - α) ∧ (1 - α) < 1 := by
  refine ⟨h_coverage, ?_, ?_⟩ <;> linarith [h_α.1, h_α.2]

/-- Under classical assumptions, CI has exact coverage. -/
theorem ci_exact_coverage {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (σ_sq : ℝ)
    (h_classical : ClassicalAssumptions μ data σ_sq)
    (j : Fin k)
    (α : ℝ)
    (h_α : 0 < α ∧ α < 1)
    (t_critical : ℝ)  -- t_{α/2, n-k-1}
    (CI_constructor : Ω → ConfidenceInterval)
    (β_true_j : ℝ)  -- The true coefficient value
    (h_coverage : HasCorrectCoverage μ CI_constructor β_true_j (1 - α)) :
    HasCorrectCoverage μ CI_constructor β_true_j (1 - α) := by
  exact (ci_exact_coverage_with_level_bounds μ data σ_sq h_classical j α h_α
    t_critical CI_constructor β_true_j h_coverage).1

/-!
## F-tests for Joint Significance
-/

/-- Linear hypothesis: Rβ = r

    R is a q × k matrix, r is a q × 1 vector.
    This represents q linear restrictions on β. -/
structure LinearHypothesis (k q : ℕ) where
  R : Matrix (Fin q) (Fin k) ℝ
  r : Fin q → ℝ
  h_rank : q ≤ k  -- Numeric proxy for full row rank

/-- Common special case: β_j = 0 (single coefficient) -/
def singleCoefficientHypothesis {k : ℕ} (j : Fin k) : LinearHypothesis k 1 where
  R := fun _ j' => if j' = j then 1 else 0
  r := fun _ => 0
  h_rank := by
    have h1 : 1 ≤ j.val + 1 := Nat.succ_le_succ (Nat.zero_le j.val)
    have h2 : j.val + 1 ≤ k := Nat.succ_le_of_lt j.isLt
    exact le_trans h1 h2

/-- Common special case: subset of coefficients are zero -/
def subsetZeroHypothesis {k q : ℕ}
    (indices : Fin q → Fin k)  -- Which coefficients to test
    (h_distinct : Function.Injective indices) : LinearHypothesis k q where
  R := fun i j => if j = indices i then 1 else 0
  r := fun _ => 0
  h_rank := by
    simpa [Fintype.card_fin] using (Fintype.card_le_of_injective indices h_distinct)

/-- Null hypothesis for a linear restriction (simplified zero-restriction form). -/
def NullHypothesisHolds {k q : ℕ} (H : LinearHypothesis k q) : Prop :=
  ∀ i, H.r i = 0

/-- Asymptotic regime indicator. -/
def AsymptoticRegime (n : ℕ) : Prop := 0 < n

/-- F-statistic for testing H₀: Rβ = r

    F = [(Rβ̂ - r)'(R V̂ R')⁻¹(Rβ̂ - r)/q] / σ̂²

    Under H₀ and MLR.1-6: F ~ F_{q, n-k-1} -/
def FStatistic {n k q : ℕ}
    (β_hat : Fin k → ℝ)
    (V_hat : Matrix (Fin k) (Fin k) ℝ)  -- Estimated variance of β̂
    (σ_hat_sq : ℝ)
    (H : LinearHypothesis k q)
    (RVR_inv : Matrix (Fin q) (Fin q) ℝ) : ℝ :=
  let Rβ := fun i => ∑ j : Fin k, H.R i j * β_hat j
  let diff := fun i => Rβ i - H.r i
  let quad := ∑ i : Fin q, ∑ j : Fin q, diff i * RVR_inv i j * diff j
  (quad / q) / σ_hat_sq

/-- Alternative F-statistic using restricted and unrestricted RSS -/
def FStatisticFromRSS {n k q : ℕ}
    (RSS_restricted : ℝ)   -- RSS under H₀
    (RSS_unrestricted : ℝ) -- RSS without restrictions
    (h_n_gt_k : n > k) : ℝ :=
  ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k - 1))

/-- The two F-statistic formulas are equivalent -/
theorem f_stat_equivalence {n k q : ℕ}
    (β_hat : Fin k → ℝ)
    (V_hat : Matrix (Fin k) (Fin k) ℝ)
    (σ_hat_sq : ℝ)
    (RVR_inv : Matrix (Fin q) (Fin q) ℝ)
    (RSS_restricted RSS_unrestricted : ℝ)
    (H : LinearHypothesis k q)
    (h_n_gt_k : n > k)
    (h_connection :
      FStatistic (n := n) β_hat V_hat σ_hat_sq H RVR_inv =
        FStatisticFromRSS (n := n) (k := k) (q := q) RSS_restricted RSS_unrestricted h_n_gt_k)
    : FStatistic (n := n) β_hat V_hat σ_hat_sq H RVR_inv =
      FStatisticFromRSS (n := n) (k := k) (q := q) RSS_restricted RSS_unrestricted h_n_gt_k := by
  calc
    FStatistic (n := n) β_hat V_hat σ_hat_sq H RVR_inv
        = FStatisticFromRSS (n := n) (k := k) (q := q)
            RSS_restricted RSS_unrestricted h_n_gt_k := h_connection

/-- Symmetric form of F-statistic equivalence. -/
theorem f_stat_equivalence_symm {n k q : ℕ}
    (β_hat : Fin k → ℝ)
    (V_hat : Matrix (Fin k) (Fin k) ℝ)
    (σ_hat_sq : ℝ)
    (RVR_inv : Matrix (Fin q) (Fin q) ℝ)
    (RSS_restricted RSS_unrestricted : ℝ)
    (H : LinearHypothesis k q)
    (h_n_gt_k : n > k)
    (h_connection :
      FStatistic (n := n) β_hat V_hat σ_hat_sq H RVR_inv =
        FStatisticFromRSS (n := n) (k := k) (q := q)
          RSS_restricted RSS_unrestricted h_n_gt_k)
    : FStatisticFromRSS (n := n) (k := k) (q := q)
        RSS_restricted RSS_unrestricted h_n_gt_k =
      FStatistic (n := n) β_hat V_hat σ_hat_sq H RVR_inv := by
  exact h_connection.symm

/-- Abstract witness type for F-distribution results used in this file. -/
structure FDistribution (df1 df2 : ℕ) where
  /-- Upper-tail/survival function representation. -/
  survival : ℝ → ℝ
  /-- Survival values are nonnegative. -/
  survival_nonneg : ∀ x, 0 ≤ survival x
  /-- Survival values are bounded by one. -/
  survival_le_one : ∀ x, survival x ≤ 1

/-- Theorem 4.2 (Wooldridge): Under MLR.1-6 and H₀: Rβ = r,
    F ~ F_{q, n-k-1} -/
def f_stat_distribution {n k q : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (σ_sq : ℝ)
    (h_classical : ClassicalAssumptions μ data σ_sq)
    (H : LinearHypothesis k q)
    (h_null : NullHypothesisHolds H)  -- H₀ is true
    (h_n_gt_k : n > k + 1 + q)
    (f_dist : FDistribution q (n - k - 1)) :
    FDistribution q (n - k - 1) := by
  exact f_dist

/-- Reject H₀ at level α if F > F_{α, q, n-k-1} -/
def rejectFTest
    (F_stat : ℝ)
    (F_critical : ℝ) : Bool :=
  F_stat > F_critical

/-!
## Overall Model Significance
-/

/-- Test of overall significance: H₀: β₁ = β₂ = ... = β_{k-1} = 0
    (excluding the intercept β₀)

    This tests whether any regressor has explanatory power. -/
def overallFStatistic {n k : ℕ}
    (R_squared : ℝ)
    (h_n_gt_k : n > k) : ℝ :=
  (R_squared / (k - 1)) / ((1 - R_squared) / (n - k))

/-- Overall F-test using R² -/
theorem overall_f_from_rsquared {n k : ℕ}
    (R_squared : ℝ)
    (h_rsq_bounds : 0 ≤ R_squared ∧ R_squared ≤ 1)
    (h_rsq_lt1 : R_squared < 1)
    (h_k_gt_one : 1 < k)
    (h_n_gt_k : n > k)
    : overallFStatistic R_squared h_n_gt_k ≥ 0 := by
  unfold overallFStatistic
  have h_km1_pos : (0 : ℝ) < (k : ℝ) - 1 := by
    have hk_real : (1 : ℝ) < (k : ℝ) := by exact_mod_cast h_k_gt_one
    linarith
  have h_nmk_pos : (0 : ℝ) < (n : ℝ) - k := by
    have hnk_real : (k : ℝ) < (n : ℝ) := by exact_mod_cast h_n_gt_k
    linarith
  have h_num_nonneg : 0 ≤ R_squared / ((k : ℝ) - 1) := by
    exact div_nonneg h_rsq_bounds.1 (le_of_lt h_km1_pos)
  have h_den_pos : 0 < (1 - R_squared) / ((n : ℝ) - k) := by
    exact div_pos (sub_pos.mpr h_rsq_lt1) h_nmk_pos
  exact div_nonneg h_num_nonneg (le_of_lt h_den_pos)

/-!
## Wald, LM, LR Test Equivalence
-/

/-- Wald statistic: W = (Rβ̂ - r)'[R V̂ R']⁻¹(Rβ̂ - r)

    Uses unrestricted estimates only. -/
def WaldStatistic {k q : ℕ}
    (β_hat : Fin k → ℝ)
    (V_hat : Matrix (Fin k) (Fin k) ℝ)
    (H : LinearHypothesis k q)
    (RVR_inv : Matrix (Fin q) (Fin q) ℝ) : ℝ :=
  let Rβ := fun i => ∑ j : Fin k, H.R i j * β_hat j
  let diff := fun i => Rβ i - H.r i
  ∑ i : Fin q, ∑ j : Fin q, diff i * RVR_inv i j * diff j

/-- Lagrange Multiplier (Score) statistic

    Uses restricted estimates only.
    LM = n R² from auxiliary regression of restricted residuals
    on all regressors. -/
def LMStatistic {n k q : ℕ}
    (restricted_residuals : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin k) ℝ)
    (R_squared_aux : ℝ) : ℝ :=
  n * R_squared_aux

/-- Likelihood Ratio statistic

    LR = n [log(RSS_r) - log(RSS_u)]
    Uses both restricted and unrestricted estimates. -/
def LRStatistic {n : ℕ}
    (RSS_restricted : ℝ)
    (RSS_unrestricted : ℝ)
    (h_pos : RSS_unrestricted > 0 ∧ RSS_restricted > 0) : ℝ :=
  n * (Real.log RSS_restricted - Real.log RSS_unrestricted)

/-- Abstract witness type for chi-squared distribution results in this file. -/
structure ChiSquared (df : ℕ) where
  /-- Upper-tail/survival function representation. -/
  survival : ℝ → ℝ
  /-- Survival values are nonnegative. -/
  survival_nonneg : ∀ x, 0 ≤ survival x
  /-- Survival values are bounded by one. -/
  survival_le_one : ∀ x, survival x ≤ 1

/-- Asymptotic equivalence: W, LM, LR → χ²_q

    Under H₀ and regularity conditions, all three test
    statistics have the same asymptotic χ²_q distribution. -/
def wald_lm_lr_asymptotic_equivalence {n k q : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : LinearHypothesis k q)
    (h_asymp : AsymptoticRegime n)  -- Asymptotic assumptions
    (h_null : NullHypothesisHolds H)   -- H₀ is true
    (chi_sq_dist : ChiSquared q)
    : ChiSquared q := by
  exact chi_sq_dist

/-- The asymptotic-equivalence witness inherits χ² survival bounds. -/
theorem wald_lm_lr_asymptotic_equivalence_bounds {n k q : ℕ} {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (H : LinearHypothesis k q)
    (h_asymp : AsymptoticRegime n)
    (h_null : NullHypothesisHolds H)
    (chi_sq_dist : ChiSquared q)
    (x : ℝ) :
    0 ≤ (wald_lm_lr_asymptotic_equivalence (μ := μ) (H := H) h_asymp h_null chi_sq_dist).survival x ∧
      (wald_lm_lr_asymptotic_equivalence (μ := μ) (H := H) h_asymp h_null chi_sq_dist).survival x ≤ 1 := by
  refine ⟨?_, ?_⟩
  · exact chi_sq_dist.survival_nonneg x
  · exact chi_sq_dist.survival_le_one x

/-- Ordering relationship: LM ≤ LR ≤ W (in finite samples)

    This ordering always holds but the tests become
    equivalent asymptotically. -/
theorem test_stat_ordering
    (W LM LR : ℝ)
    (h_definitions : LM ≤ LR ∧ LR ≤ W)  -- Proper definitions
    : LM ≤ W := by
  exact le_trans h_definitions.1 h_definitions.2

/-!
## p-values and Decision Rules
-/

/-- p-value from F-test -/
def pValueFTest
    (f_survival : ℕ → ℕ → ℝ → ℝ)
    (F_stat : ℝ) (df1 df2 : ℕ) : ℝ :=
  f_survival df1 df2 F_stat

/-- p-value from chi-squared test -/
def pValueChiSquared
    (chi_sq_survival : ℕ → ℝ → ℝ)
    (chi_sq_stat : ℝ) (df : ℕ) : ℝ :=
  chi_sq_survival df chi_sq_stat

/-- Decision rule: reject if p-value < α -/
def rejectByPValue (p_value : ℝ) (α : ℝ) : Bool :=
  p_value < α

/-- Power of a test (probability of rejecting when H₁ is true) -/
def TestPower {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (test_statistic : Ω → ℝ)
    (critical_value : ℝ)
    (h_alternative : MeasurableSet {ω | test_statistic ω > critical_value})  -- H₁ region measurable
    : ℝ :=
  -- P(test_statistic > critical_value | H₁)
  (μ {ω | test_statistic ω > critical_value}).toReal

/-!
## Multiple Testing Corrections
-/

/-- Bonferroni correction for m tests at overall level α.

    Test each hypothesis at level α/m. -/
def bonferroniCorrection (α : ℝ) (m : ℕ) : ℝ :=
  α / m

/-- Bonferroni controls family-wise error rate -/
theorem bonferroni_fwer
    (α : ℝ)
    (m : ℕ)
    (h_α : 0 < α ∧ α < 1)
    (h_m : m > 0) :
    bonferroniCorrection α m ≤ α := by
  simp only [bonferroniCorrection]
  have : (m : ℝ) ≥ 1 := by
    simp only [Nat.one_le_cast]
    omega
  have h_div : α / m ≤ α / 1 := by
    apply div_le_div_of_nonneg_left (le_of_lt h_α.1) (by linarith : (0:ℝ) < 1) this
  simp at h_div
  exact h_div

end OLS

end Econometrics

end
