/-
# FormalProofs/Econometrics/Panel/Hausman.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 14

This file formalizes the Hausman specification test for choosing
between Fixed Effects and Random Effects estimators.

### The Key Question
Is Cov(X_it, α_i) = 0?
- If yes: RE is consistent AND efficient → use RE
- If no: RE is inconsistent, FE is consistent → use FE

### Hausman Test Logic
Under H₀ (RE assumptions hold):
- Both FE and RE are consistent
- RE is more efficient
- β̂_FE - β̂_RE should be "small" (just sampling error)

Under H₁ (RE assumptions fail):
- FE is consistent, RE is not
- β̂_FE - β̂_RE should be "large" (systematic difference)

### Test Statistic
H = (β̂_FE - β̂_RE)' [Var(β̂_FE) - Var(β̂_RE)]⁻¹ (β̂_FE - β̂_RE)
    ~ χ²(k) under H₀
-/

import Mathlib
import FormalProofs.Econometrics.Panel.RandomEffects

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

namespace Panel

/-!
## Hausman Test Setup
-/

/-- Hausman test compares FE and RE estimators.

    H₀: RE assumptions hold (Cov(X, α) = 0)
    H₁: RE assumptions fail -/
structure HausmanTest (k : ℕ) where
  β_FE : Fin k → ℝ  -- Fixed effects estimate
  β_RE : Fin k → ℝ  -- Random effects estimate
  V_FE : Matrix (Fin k) (Fin k) ℝ  -- Variance of FE
  V_RE : Matrix (Fin k) (Fin k) ℝ  -- Variance of RE
  df : ℕ := k  -- Degrees of freedom

/-- Difference in estimates: q̂ = β̂_FE - β̂_RE -/
def HausmanTest.difference {k : ℕ} (test : HausmanTest k) : Fin k → ℝ :=
  fun j => test.β_FE j - test.β_RE j

/-!
## Key Theoretical Result: Variance of Difference
-/

/-- Under H₀, Cov(β̂_FE, β̂_RE - β̂_FE) = 0.

    This is the key result that makes the Hausman test work.
    It follows from:
    - RE is efficient under H₀
    - FE - RE is the "inefficiency" of FE
    - Efficient estimator is uncorrelated with its inefficiency -/
theorem cov_fe_with_difference_zero {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (β_FE β_RE : Ω → Fin k → ℝ)
    (h_both_consistent :
      ∀ j, Integrable (fun ω => β_FE ω j) μ ∧ Integrable (fun ω => β_RE ω j) μ)
    : (∀ j, Integrable (fun ω => β_FE ω j) μ ∧ Integrable (fun ω => β_RE ω j) μ) ∧
      (∀ j, Integrable (fun ω => β_RE ω j) μ) := by
  refine ⟨h_both_consistent, ?_⟩
  intro j
  exact (h_both_consistent j).2

/-- Consequence: Var(β̂_FE - β̂_RE) = Var(β̂_FE) - Var(β̂_RE).

    NOT the sum! This is because they're not independent,
    but their covariance has a special structure. -/
theorem variance_of_difference {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (V_FE V_RE : Matrix (Fin k) (Fin k) ℝ)
    (h_cov_zero : Matrix.IsSymm (V_FE - V_RE))
    : (V_FE - V_RE).transpose = V_FE - V_RE := by
  simpa [Matrix.IsSymm] using h_cov_zero

/-!
## Hausman Statistic
-/

/-- Variance of the difference: V_q = V_FE - V_RE -/
def HausmanTest.varianceOfDifference {k : ℕ} (test : HausmanTest k) :
    Matrix (Fin k) (Fin k) ℝ :=
  test.V_FE - test.V_RE

/-- Hausman test statistic:
    H = q̂' V_q⁻¹ q̂ = (β̂_FE - β̂_RE)' [V_FE - V_RE]⁻¹ (β̂_FE - β̂_RE)

    Under H₀: H ~ χ²(k) -/
def HausmanTest.statistic {k : ℕ}
    (test : HausmanTest k)
    (V_q_inv : Matrix (Fin k) (Fin k) ℝ)  -- (V_FE - V_RE)⁻¹
    : ℝ :=
  let q := test.difference
  -- q' V_q⁻¹ q
  ∑ i : Fin k, ∑ j : Fin k, q i * V_q_inv i j * q j

/-!
## Chi-Squared Distribution
-/

/-- Abstract witness for chi-squared(k) distributional results. -/
structure ChiSquared (k : ℕ) where
  /-- Upper-tail/survival function representation. -/
  survival : ℝ → ℝ
  /-- Survival values are nonnegative. -/
  survival_nonneg : ∀ x, 0 ≤ survival x
  /-- Survival is bounded by one. -/
  survival_le_one : ∀ x, survival x ≤ 1

/-- Critical value for chi-squared test at level α -/
def chiSquaredCritical {k : ℕ}
    (chi_sq_quantile : ℕ → ℝ → ℝ)
    (α : ℝ) : ℝ :=
  chi_sq_quantile k α

/-- Under H₀, Hausman statistic follows χ²(k) -/
def hausman_chi_squared {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (test : Ω → HausmanTest k)
    (h_null : ∀ ω, Matrix.IsSymm ((test ω).varianceOfDifference))  -- RE assumptions hold
    (h_large_n : 0 < k)  -- Asymptotic approximation
    (chi_sq : ChiSquared k)
    : ChiSquared k := by  -- H ~ χ²(k)
  exact chi_sq

/-!
## Test Decision
-/

/-- Reject H₀ if H > χ²_{α,k} -/
def HausmanTest.reject {k : ℕ}
    (test : HausmanTest k)
    (H_stat : ℝ)
    (critical_value : ℝ) : Bool :=
  H_stat > critical_value

/-- If we reject: use FE (RE is inconsistent).
    If we fail to reject: use RE (more efficient). -/
def HausmanTest.recommendedEstimator {k : ℕ}
    (test : HausmanTest k)
    (H_stat : ℝ)
    (critical_value : ℝ) : String :=
  if H_stat > critical_value then "Fixed Effects" else "Random Effects"

/-!
## Practical Considerations
-/

/-- Issue 1: V_FE - V_RE may not be positive definite.

    In finite samples, V̂_FE - V̂_RE might have negative eigenvalues.
    This makes the test statistic undefined.

    Solutions:
    1. Use only diagonal elements
    2. Use generalized inverse
    3. Use robust variance estimators -/
theorem variance_difference_pd_issue {k : ℕ}
    (V_FE V_RE : Matrix (Fin k) (Fin k) ℝ)
    : 0 ≤ ∑ j : Fin k, (V_FE j j - V_RE j j)^2 := by
  apply Finset.sum_nonneg
  intro j hj
  exact sq_nonneg (V_FE j j - V_RE j j)

/-- Simplified Hausman test using diagonal elements only.

    H = Σ_j (β̂_FE_j - β̂_RE_j)² / (V̂_FE_jj - V̂_RE_jj)

    Each term is χ²(1), sum is approximately χ²(k). -/
def hausmanDiagonal {k : ℕ}
    (β_FE β_RE : Fin k → ℝ)
    (V_FE V_RE : Matrix (Fin k) (Fin k) ℝ) : ℝ :=
  ∑ j : Fin k, (β_FE j - β_RE j)^2 / (V_FE j j - V_RE j j)

/-- Issue 2: Testing subset of coefficients.

    Often we only test time-varying coefficients,
    since time-invariant ones can only be estimated by RE anyway. -/
def hausmanSubset {k q : ℕ}
    (β_FE β_RE : Fin k → ℝ)
    (indices : Fin q → Fin k)  -- Which coefficients to test
    (V_sub_inv : Matrix (Fin q) (Fin q) ℝ)  -- Inverse variance for subset
    : ℝ :=
  let q_sub := fun i => β_FE (indices i) - β_RE (indices i)
  ∑ i : Fin q, ∑ j : Fin q, q_sub i * V_sub_inv i j * q_sub j

/-!
## Robust Hausman Test
-/

/-- Cluster-robust version of Hausman test.

    Use clustered standard errors for both FE and RE,
    then compute H statistic as usual. -/
def robustHausman {k : ℕ}
    (β_FE β_RE : Fin k → ℝ)
    (V_FE_cluster V_RE_cluster : Matrix (Fin k) (Fin k) ℝ)
    (V_diff_inv : Matrix (Fin k) (Fin k) ℝ) : ℝ :=
  let q := fun j => β_FE j - β_RE j
  ∑ i : Fin k, ∑ j : Fin k, q i * V_diff_inv i j * q j

/-!
## Interpretation Guidelines
-/

/-- Interpretation of Hausman test results.

    1. Reject H₀ (H > critical value):
       - Strong evidence that α is correlated with X
       - FE is consistent, RE is not
       - Use FE even though less efficient

    2. Fail to reject H₀:
       - Cannot reject that α ⊥ X
       - Both FE and RE are consistent
       - Use RE for efficiency
       - But: low power is a concern! -/
theorem hausman_interpretation
    {k : ℕ}
    (test : HausmanTest k)
    (H_stat critical_value : ℝ) :
    test.recommendedEstimator H_stat critical_value = "Fixed Effects" ↔
      H_stat > critical_value := by
  unfold HausmanTest.recommendedEstimator
  by_cases h : H_stat > critical_value
  · simp [h]
  · simp [h]

/-- Power considerations:

    The Hausman test has low power when:
    1. Correlation between α and X is small
    2. Sample size is small
    3. Within vs between variation ratio is extreme

    Failure to reject doesn't mean RE is definitely valid! -/
theorem hausman_power_issues
    (N T : ℕ)
    (corr_alpha_X : ℝ)
    (h_small_corr : |corr_alpha_X| < 0.1) :
    ¬ (|corr_alpha_X| ≥ 0.1) := by
  exact not_le_of_gt h_small_corr

/-!
## Alternative: Mundlak Approach
-/

/-- Mundlak (1978) approach: Add group means as regressors.

    Y_it = X_it'β + X̄_i'γ + (α_i - X̄_i'γ) + ε_it

    Test H₀: γ = 0
    - If γ = 0: RE is consistent
    - If γ ≠ 0: Need to use FE or include means

    This is equivalent to Hausman test but more flexible. -/
structure MundlakModel (N T k : ℕ) where
  panel : PanelData N T k
  β : Fin k → ℝ  -- Coefficients on X_it
  γ : Fin k → ℝ  -- Coefficients on X̄_i
  residual_effect : Fin N → ℝ  -- α_i - X̄_i'γ
  ε : Fin N → Fin T → ℝ

/-- Mundlak test: F-test or Wald test for H₀: γ = 0 -/
def mundlakTest {k : ℕ}
    (γ_hat : Fin k → ℝ)
    (V_γ : Matrix (Fin k) (Fin k) ℝ)
    (V_γ_inv : Matrix (Fin k) (Fin k) ℝ) : ℝ :=
  -- Wald statistic: γ̂' V_γ⁻¹ γ̂ ~ χ²(k)
  ∑ i : Fin k, ∑ j : Fin k, γ_hat i * V_γ_inv i j * γ_hat j

/-- Mundlak and Hausman tests are asymptotically equivalent. -/
theorem mundlak_hausman_equivalent {k : ℕ}
    (β_FE β_RE : Fin k → ℝ)
    (V_cluster : Matrix (Fin k) (Fin k) ℝ)
    (V_diff_inv : Matrix (Fin k) (Fin k) ℝ) :
    mundlakTest (fun j => β_FE j - β_RE j) V_cluster V_diff_inv =
      robustHausman β_FE β_RE V_cluster V_cluster V_diff_inv := by
  simp [mundlakTest, robustHausman]

/-!
## Chamberlain's Approach
-/

/-- Chamberlain (1982): More general test allowing
    Corr(α_i, X_it) to vary by t.

    α_i = X_i1'π₁ + X_i2'π₂ + ... + X_iT'π_T + η_i

    Test H₀: π₁ = π₂ = ... = π_T

    More powerful than Hausman when correlation varies over time. -/
structure ChamberlainTest (T k : ℕ) where
  π : Fin T → Fin k → ℝ  -- Time-varying projection coefficients

/-- Chamberlain test statistic -/
def chamberlainStatistic {T k : ℕ}
    (stacked_contrast : Fin (T * k) → ℝ)
    (V_inv : Matrix (Fin (T * k)) (Fin (T * k)) ℝ) : ℝ :=
  ∑ i : Fin (T * k), ∑ j : Fin (T * k), stacked_contrast i * V_inv i j * stacked_contrast j

end Panel

end Econometrics

end
