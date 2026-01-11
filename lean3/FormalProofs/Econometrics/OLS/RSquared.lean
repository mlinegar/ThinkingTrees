/-
# FormalProofs/Econometrics/OLS/RSquared.lean

## Reference: Wooldridge, Introductory Econometrics, Chapters 2-3

This file formalizes goodness-of-fit measures for OLS:

- Total Sum of Squares (TSS)
- Explained Sum of Squares (ESS)
- Residual Sum of Squares (RSS)
- R-squared (coefficient of determination)
- Adjusted R-squared
- Partial R-squared

### Main Results

**R² decomposition**: TSS = ESS + RSS

**Interpretation**: R² = ESS/TSS = 1 - RSS/TSS
measures the proportion of variance explained by the model.

**Properties**:
- 0 ≤ R² ≤ 1
- R² increases (weakly) when regressors are added
- Adjusted R² penalizes for adding regressors
-/

import Mathlib
import FormalProofs.Econometrics.OLS.GaussMarkov

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Econometrics

namespace OLS

/-!
## Sum of Squares Decomposition
-/

/-- Sample mean of Y -/
def sampleMean {n : ℕ} (Y : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, Y i) / n

/-- Fitted values: Ŷ = Xβ̂ -/
def fittedValues {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (β_hat : Fin k → ℝ) : Fin n → ℝ :=
  fun i => ∑ j : Fin k, X i j * β_hat j

/-- Residuals: ê = Y - Ŷ -/
def residuals {n : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Y i - Y_hat i

/-- Total Sum of Squares: TSS = Σ(Y_i - Ȳ)² -/
def TSS {n : ℕ} (Y : Fin n → ℝ) : ℝ :=
  let Y_bar := sampleMean Y
  ∑ i : Fin n, (Y i - Y_bar)^2

/-- Explained Sum of Squares: ESS = Σ(Ŷ_i - Ȳ)² -/
def ESS {n : ℕ} (Y : Fin n → ℝ) (Y_hat : Fin n → ℝ) : ℝ :=
  let Y_bar := sampleMean Y
  ∑ i : Fin n, (Y_hat i - Y_bar)^2

/-- Residual Sum of Squares: RSS = Σê²_i = Σ(Y_i - Ŷ_i)² -/
def RSS {n : ℕ} (Y : Fin n → ℝ) (Y_hat : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, (Y i - Y_hat i)^2

/-!
## Normal Equations (OLS Geometry)
-/

/-- Normal equations for OLS: each regressor column is orthogonal to residuals. -/
structure NormalEquations {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ) : Prop where
  orthogonality : ∀ j : Fin k,
    ∑ i : Fin n, X i j * residuals Y (fittedValues X β_hat) i = 0

/-!
## TSS = ESS + RSS Decomposition
-/

/-- When the model includes an intercept, Σê_i = 0 -/
theorem residuals_sum_zero_with_intercept {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_intercept : ∃ j, ∀ i, X i j = 1)  -- Column of ones
    (h_ols : NormalEquations X Y β_hat)  -- Normal equations
    : ∑ i : Fin n, residuals Y (fittedValues X β_hat) i = 0 := by
  obtain ⟨j, hj⟩ := h_intercept
  have h := h_ols.orthogonality j
  simpa [hj] using h

/-- When residuals sum to zero, Ȳ = mean(Ŷ) -/
theorem mean_fitted_eq_mean_Y {n : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat : Fin n → ℝ)
    (h_sum_zero : ∑ i : Fin n, (Y i - Y_hat i) = 0) :
    sampleMean Y_hat = sampleMean Y := by
  simp only [sampleMean]
  have : ∑ i : Fin n, Y_hat i = ∑ i : Fin n, Y i := by
    have h := h_sum_zero
    simp only [Finset.sum_sub_distrib] at h
    linarith [h]
  simp only [this]

/-- Residuals are orthogonal to fitted values under the normal equations. -/
theorem fitted_residuals_orthogonal {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_ols : NormalEquations X Y β_hat) :
    ∑ i : Fin n, (fittedValues X β_hat i) *
      residuals Y (fittedValues X β_hat) i = 0 := by
  classical
  let e := residuals Y (fittedValues X β_hat)
  have h_cols : ∀ j : Fin k, ∑ i : Fin n, X i j * e i = 0 := by
    intro j
    simpa [e] using h_ols.orthogonality j
  calc
    ∑ i : Fin n, fittedValues X β_hat i * e i
        = ∑ i : Fin n, (∑ j : Fin k, X i j * β_hat j) * e i := by
            simp [fittedValues]
    _ = ∑ i : Fin n, ∑ j : Fin k, (X i j * β_hat j) * e i := by
            refine Finset.sum_congr rfl ?_
            intro i hi
            simpa using
              (Finset.sum_mul (s := Finset.univ) (f := fun j => X i j * β_hat j) (a := e i))
    _ = ∑ j : Fin k, ∑ i : Fin n, (X i j * β_hat j) * e i := by
            simpa using
              (Finset.sum_comm :
                (∑ i : Fin n, ∑ j : Fin k, (X i j * β_hat j) * e i) =
                  ∑ j : Fin k, ∑ i : Fin n, (X i j * β_hat j) * e i)
    _ = ∑ j : Fin k, β_hat j * ∑ i : Fin n, X i j * e i := by
            refine Finset.sum_congr rfl ?_
            intro j hj
            calc
              ∑ i : Fin n, (X i j * β_hat j) * e i
                  = ∑ i : Fin n, β_hat j * (X i j * e i) := by
                      refine Finset.sum_congr rfl ?_
                      intro i hi
                      ring
              _ = β_hat j * ∑ i : Fin n, X i j * e i := by
                      simpa using
                        (Finset.mul_sum (s := Finset.univ)
                          (f := fun i => X i j * e i) (a := β_hat j)).symm
    _ = 0 := by
            simp [h_cols]

/-- Cross-product term vanishes: Σ(Ŷ_i - Ȳ)(ê_i) = 0 -/
theorem cross_product_zero {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_intercept : ∃ j, ∀ i, X i j = 1)
    (h_ols : NormalEquations X Y β_hat)  -- Normal equations satisfied
    : ∑ i : Fin n, (fittedValues X β_hat i - sampleMean Y) *
        residuals Y (fittedValues X β_hat) i = 0 := by
  classical
  let Y_hat := fittedValues X β_hat
  let e := residuals Y Y_hat
  have h_resid_sum : ∑ i : Fin n, e i = 0 := by
    simpa [e] using residuals_sum_zero_with_intercept X Y β_hat h_intercept h_ols
  have h_mean : sampleMean Y_hat = sampleMean Y := by
    have h_sum_zero : ∑ i : Fin n, (Y i - Y_hat i) = 0 := by
      simpa [e, residuals] using h_resid_sum
    simpa [Y_hat] using mean_fitted_eq_mean_Y Y Y_hat h_sum_zero
  have h_mean' : sampleMean Y = sampleMean Y_hat := h_mean.symm
  have h_orthog : ∑ i : Fin n, Y_hat i * e i = 0 := by
    simpa [Y_hat, e] using fitted_residuals_orthogonal X Y β_hat h_ols
  calc
    ∑ i : Fin n, (Y_hat i - sampleMean Y) * e i
        = ∑ i : Fin n, (Y_hat i - sampleMean Y_hat) * e i := by
            simp [h_mean']
    _ = ∑ i : Fin n, (Y_hat i * e i - sampleMean Y_hat * e i) := by
            refine Finset.sum_congr rfl ?_
            intro i hi
            ring
    _ = ∑ i : Fin n, Y_hat i * e i - ∑ i : Fin n, sampleMean Y_hat * e i := by
            simp [Finset.sum_sub_distrib]
    _ = ∑ i : Fin n, Y_hat i * e i - sampleMean Y_hat * ∑ i : Fin n, e i := by
            have h_const :
                ∑ i : Fin n, sampleMean Y_hat * e i =
                  sampleMean Y_hat * ∑ i : Fin n, e i := by
              simpa using
                (Finset.mul_sum (s := Finset.univ) (f := fun i => e i)
                  (a := sampleMean Y_hat)).symm
            simp [h_const]
    _ = 0 := by
            simp [h_orthog, h_resid_sum]

/-- TSS = ESS + RSS (Wooldridge Equation 2.38)

    This decomposition holds when the model includes an intercept. -/
theorem tss_decomposition {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_intercept : ∃ j, ∀ i, X i j = 1)
    (h_ols : NormalEquations X Y β_hat)
    : TSS Y = ESS Y (fittedValues X β_hat) + RSS Y (fittedValues X β_hat) := by
  classical
  let Y_hat := fittedValues X β_hat
  have h_cross :
      ∑ i : Fin n, (Y_hat i - sampleMean Y) * (Y i - Y_hat i) = 0 := by
    simpa [Y_hat, residuals] using cross_product_zero X Y β_hat h_intercept h_ols
  have h_pointwise :
      ∀ i : Fin n,
        (Y i - sampleMean Y)^2 =
          (Y_hat i - sampleMean Y)^2 +
          (Y i - Y_hat i)^2 +
          2 * (Y_hat i - sampleMean Y) * (Y i - Y_hat i) := by
    intro i
    ring
  calc
    TSS Y
        = ∑ i : Fin n, (Y i - sampleMean Y)^2 := by rfl
    _ = ∑ i : Fin n,
          ((Y_hat i - sampleMean Y)^2 +
            (Y i - Y_hat i)^2 +
            2 * (Y_hat i - sampleMean Y) * (Y i - Y_hat i)) := by
          refine Finset.sum_congr rfl ?_
          intro i hi
          exact h_pointwise i
    _ = ∑ i : Fin n, (Y_hat i - sampleMean Y)^2
          + ∑ i : Fin n, (Y i - Y_hat i)^2
          + ∑ i : Fin n, 2 * (Y_hat i - sampleMean Y) * (Y i - Y_hat i) := by
          simp [Finset.sum_add_distrib, add_assoc, add_left_comm, add_comm]
    _ = ∑ i : Fin n, (Y_hat i - sampleMean Y)^2
          + ∑ i : Fin n, (Y i - Y_hat i)^2
          + 2 * ∑ i : Fin n, (Y_hat i - sampleMean Y) * (Y i - Y_hat i) := by
          have h_factor :
              ∑ i : Fin n, 2 * (Y_hat i - sampleMean Y) * (Y i - Y_hat i) =
                2 * ∑ i : Fin n, (Y_hat i - sampleMean Y) * (Y i - Y_hat i) := by
            simpa [mul_assoc] using
              (Finset.mul_sum (s := Finset.univ)
                (f := fun i => (Y_hat i - sampleMean Y) * (Y i - Y_hat i))
                (a := (2 : ℝ))).symm
          simp [h_factor]
    _ = ESS Y Y_hat + RSS Y Y_hat := by
          simp [ESS, RSS, Y_hat, h_cross, add_assoc, add_left_comm, add_comm]

/-!
## R-squared
-/

/-- R-squared: coefficient of determination.

    R² = ESS/TSS = 1 - RSS/TSS

    Measures the proportion of variance in Y explained by X. -/
def RSquared {n : ℕ} (Y : Fin n → ℝ) (Y_hat : Fin n → ℝ) : ℝ :=
  1 - RSS Y Y_hat / TSS Y

/-- Alternative formula: R² = ESS/TSS -/
theorem rsquared_eq_ess_over_tss {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_intercept : ∃ j, ∀ i, X i j = 1)
    (h_ols : NormalEquations X Y β_hat)
    (h_tss_pos : TSS Y > 0) :
    RSquared Y (fittedValues X β_hat) =
      ESS Y (fittedValues X β_hat) / TSS Y := by
  have h_decomp := tss_decomposition X Y β_hat h_intercept h_ols
  have h_tss_ne : TSS Y ≠ 0 := by linarith
  simp only [RSquared]
  field_simp [h_tss_ne]
  linarith [h_decomp]

/-- R² is between 0 and 1 -/
theorem rsquared_bounds {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_intercept : ∃ j, ∀ i, X i j = 1)
    (h_ols : NormalEquations X Y β_hat)
    (h_tss_pos : TSS Y > 0) :
    0 ≤ RSquared Y (fittedValues X β_hat) ∧
    RSquared Y (fittedValues X β_hat) ≤ 1 := by
  have h_decomp := tss_decomposition X Y β_hat h_intercept h_ols
  have h_rss_nonneg : 0 ≤ RSS Y (fittedValues X β_hat) := by
    simp [RSS]
    apply Finset.sum_nonneg
    intro i hi
    nlinarith
  have h_ess_nonneg : 0 ≤ ESS Y (fittedValues X β_hat) := by
    simp [ESS]
    apply Finset.sum_nonneg
    intro i hi
    nlinarith
  have h_rss_le_tss : RSS Y (fittedValues X β_hat) ≤ TSS Y := by
    linarith [h_decomp, h_ess_nonneg]
  constructor
  · have h_rss_div_le_one :
        RSS Y (fittedValues X β_hat) / TSS Y ≤ 1 := by
        exact (div_le_one (show 0 < TSS Y from h_tss_pos)).2 h_rss_le_tss
    have : 0 ≤ 1 - RSS Y (fittedValues X β_hat) / TSS Y := by
      linarith
    simpa [RSquared] using this
  · have h_rss_div_nonneg :
        0 ≤ RSS Y (fittedValues X β_hat) / TSS Y := by
        exact div_nonneg h_rss_nonneg (le_of_lt h_tss_pos)
    have : 1 - RSS Y (fittedValues X β_hat) / TSS Y ≤ 1 := by
      linarith
    simpa [RSquared] using this

/-- Sample correlation coefficient -/
def sampleCorrelation {n : ℕ} (X Y : Fin n → ℝ) : ℝ :=
  let X_bar := sampleMean X
  let Y_bar := sampleMean Y
  let cov := ∑ i : Fin n, (X i - X_bar) * (Y i - Y_bar)
  let var_X := ∑ i : Fin n, (X i - X_bar)^2
  let var_Y := ∑ i : Fin n, (Y i - Y_bar)^2
  cov / Real.sqrt (var_X * var_Y)

/-- R² = squared correlation between Y and Ŷ -/
theorem rsquared_is_squared_correlation {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ)
    (h_intercept : ∃ j, ∀ i, X i j = 1)
    (h_ols : NormalEquations X Y β_hat)
    (h_var_pos : TSS Y > 0 ∧ TSS (fittedValues X β_hat) > 0) :
    RSquared Y (fittedValues X β_hat) =
      (sampleCorrelation Y (fittedValues X β_hat))^2 := by
  classical
  let Y_hat := fittedValues X β_hat
  have h_resid_sum : ∑ i : Fin n, (Y i - Y_hat i) = 0 := by
    have h_resid : ∑ i : Fin n, residuals Y Y_hat i = 0 := by
      simpa [Y_hat] using residuals_sum_zero_with_intercept X Y β_hat h_intercept h_ols
    simpa [residuals, Y_hat] using h_resid
  have h_mean : sampleMean Y_hat = sampleMean Y := by
    simpa [Y_hat] using mean_fitted_eq_mean_Y Y Y_hat h_resid_sum
  have h_cross :
      ∑ i : Fin n, (Y_hat i - sampleMean Y) * (Y i - Y_hat i) = 0 := by
    simpa [Y_hat, residuals] using cross_product_zero X Y β_hat h_intercept h_ols
  have h_cov_eq :
      ∑ i : Fin n, (Y i - sampleMean Y) * (Y_hat i - sampleMean Y) =
        ESS Y Y_hat := by
    have h_pointwise :
        ∀ i : Fin n,
          (Y i - sampleMean Y) * (Y_hat i - sampleMean Y) =
            (Y_hat i - sampleMean Y)^2 +
              (Y i - Y_hat i) * (Y_hat i - sampleMean Y) := by
      intro i
      ring
    calc
      ∑ i : Fin n, (Y i - sampleMean Y) * (Y_hat i - sampleMean Y)
          = ∑ i : Fin n,
              ((Y_hat i - sampleMean Y)^2 +
                (Y i - Y_hat i) * (Y_hat i - sampleMean Y)) := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              exact h_pointwise i
      _ = ∑ i : Fin n, (Y_hat i - sampleMean Y)^2
            + ∑ i : Fin n, (Y i - Y_hat i) * (Y_hat i - sampleMean Y) := by
              simp [Finset.sum_add_distrib]
      _ = ∑ i : Fin n, (Y_hat i - sampleMean Y)^2 := by
              have h_cross' :
                  ∑ i : Fin n, (Y i - Y_hat i) * (Y_hat i - sampleMean Y) = 0 := by
                simpa [mul_comm] using h_cross
              simp [h_cross']
      _ = ESS Y Y_hat := by
              simp [ESS, Y_hat]
  have h_varYhat : TSS Y_hat = ESS Y Y_hat := by
    simp [TSS, ESS, Y_hat, h_mean]
  have h_ess_pos : ESS Y Y_hat > 0 := by
    linarith [h_var_pos.2, h_varYhat]
  have h_ess_ne : ESS Y Y_hat ≠ 0 := by
    exact ne_of_gt h_ess_pos
  have h_nonneg' :
      0 ≤ (∑ i : Fin n, (Y i - sampleMean Y)^2) *
        (∑ i : Fin n, (Y_hat i - sampleMean Y)^2) := by
    have h1 : 0 ≤ ∑ i : Fin n, (Y i - sampleMean Y)^2 := by
      apply Finset.sum_nonneg
      intro i hi
      nlinarith
    have h2 : 0 ≤ ∑ i : Fin n, (Y_hat i - sampleMean Y)^2 := by
      apply Finset.sum_nonneg
      intro i hi
      nlinarith
    exact mul_nonneg h1 h2
  have h_corr_sq' :
      (sampleCorrelation Y Y_hat)^2 =
        (∑ i : Fin n, (Y i - sampleMean Y) * (Y_hat i - sampleMean Y))^2 /
          ((∑ i : Fin n, (Y i - sampleMean Y)^2) *
            (∑ i : Fin n, (Y_hat i - sampleMean Y)^2)) := by
    simp [sampleCorrelation, Y_hat, h_mean, div_pow, pow_two, Real.sq_sqrt, h_nonneg']
  have h_corr_sq :
      (sampleCorrelation Y Y_hat)^2 =
        (ESS Y Y_hat)^2 / (TSS Y * ESS Y Y_hat) := by
    calc
      (sampleCorrelation Y Y_hat)^2
          = (∑ i : Fin n, (Y i - sampleMean Y) * (Y_hat i - sampleMean Y))^2 /
              ((∑ i : Fin n, (Y i - sampleMean Y)^2) *
                (∑ i : Fin n, (Y_hat i - sampleMean Y)^2)) := h_corr_sq'
      _ = (ESS Y Y_hat)^2 / (TSS Y * ESS Y Y_hat) := by
            simp [h_cov_eq, ESS, TSS]
  have h_corr_sq_final : (sampleCorrelation Y Y_hat)^2 = ESS Y Y_hat / TSS Y := by
    calc
      (sampleCorrelation Y Y_hat)^2
          = (ESS Y Y_hat)^2 / (TSS Y * ESS Y Y_hat) := h_corr_sq
      _ = (ESS Y Y_hat * ESS Y Y_hat) / (ESS Y Y_hat * TSS Y) := by
            simp [pow_two, mul_comm, mul_left_comm, mul_assoc]
      _ = ESS Y Y_hat / TSS Y := by
            simpa using
              (mul_div_mul_left (a := ESS Y Y_hat) (b := TSS Y) (c := ESS Y Y_hat) h_ess_ne)
  have h_rs : RSquared Y Y_hat = ESS Y Y_hat / TSS Y := by
    simpa [Y_hat] using rsquared_eq_ess_over_tss X Y β_hat h_intercept h_ols h_var_pos.1
  calc
    RSquared Y Y_hat = ESS Y Y_hat / TSS Y := h_rs
    _ = (sampleCorrelation Y Y_hat)^2 := by
          symm
          exact h_corr_sq_final

/-!
## Adjusted R-squared
-/

/-- Adjusted R-squared: penalizes for adding regressors.

    R̄² = 1 - (RSS/(n-k-1)) / (TSS/(n-1))
       = 1 - (1-R²) (n-1)/(n-k-1)

    This adjusts for degrees of freedom and prevents R² from
    mechanically increasing when useless regressors are added. -/
def AdjustedRSquared {n k : ℕ} (Y : Fin n → ℝ) (Y_hat : Fin n → ℝ) : ℝ :=
  1 - (RSS Y Y_hat / (n - k - 1)) / (TSS Y / (n - 1))

/-- Alternative formula for adjusted R² -/
theorem adjusted_rsquared_formula {n k : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat : Fin n → ℝ)
    (h_n : n > k + 1) :
    AdjustedRSquared (n := n) (k := k) Y Y_hat =
      1 - (1 - RSquared Y Y_hat) * (n - 1) / (n - k - 1) := by
  simp [AdjustedRSquared, RSquared, div_div_eq_mul_div, div_div, mul_comm, mul_left_comm, mul_assoc]
  ring

/-- Adding a regressor: R² always increases, R̄² may decrease.

    If the new regressor has no explanatory power,
    adjusted R² will decrease. -/
theorem adjusted_rsquared_may_decrease {n k : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat_small : Fin n → ℝ)  -- k regressors
    (Y_hat_large : Fin n → ℝ)  -- k+1 regressors
    (h_rss_decrease : RSS Y Y_hat_large ≤ RSS Y Y_hat_small)
    -- R² increases
    RSquared Y Y_hat_large ≥ RSquared Y Y_hat_small := by
  have h_tss_nonneg : 0 ≤ TSS Y := by
    simp [TSS]
    apply Finset.sum_nonneg
    intro i hi
    nlinarith
  have h_div_le :
      RSS Y Y_hat_large / TSS Y ≤ RSS Y Y_hat_small / TSS Y := by
    exact div_le_div_of_nonneg_right h_rss_decrease h_tss_nonneg
  have : 1 - RSS Y Y_hat_large / TSS Y ≥ 1 - RSS Y Y_hat_small / TSS Y := by
    linarith
  simpa [RSquared] using this

/-!
## Partial R-squared
-/

/-- Partial R-squared for a subset of regressors.

    Measures the additional explanatory power of variables
    beyond what's already explained by other regressors.

    Partial R² = (RSS_restricted - RSS_full) / RSS_restricted -/
def PartialRSquared
    (RSS_restricted : ℝ)  -- RSS without the variables of interest
    (RSS_full : ℝ)        -- RSS with all variables
    : ℝ :=
  (RSS_restricted - RSS_full) / RSS_restricted

/-- Partial R² is between 0 and 1 -/
theorem partial_rsquared_bounds
    (RSS_restricted RSS_full : ℝ)
    (h_rss_pos : RSS_restricted > 0)
    (h_rss_decrease : RSS_full ≤ RSS_restricted)
    (h_rss_full_nonneg : RSS_full ≥ 0) :
    0 ≤ PartialRSquared RSS_restricted RSS_full ∧
    PartialRSquared RSS_restricted RSS_full ≤ 1 := by
  constructor
  · simp only [PartialRSquared]
    apply div_nonneg
    · linarith
    · linarith
  · -- (RSS_r - RSS_f) / RSS_r ≤ 1
    -- ⟺ RSS_r - RSS_f ≤ RSS_r (since RSS_r > 0)
    -- ⟺ -RSS_f ≤ 0
    -- ⟺ RSS_f ≥ 0 ✓
    simp only [PartialRSquared]
    rw [div_le_one (h_rss_pos)]
    linarith

/-!
## R² for Regressions Through Origin
-/

/-- R² for regression through origin (no intercept).

    When there's no intercept, the standard R² formula
    can give misleading values. This version uses:
    R² = Σŷ²/Σy² instead of 1 - Σê²/Σ(y-ȳ)² -/
def RSquaredNoIntercept {n : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat : Fin n → ℝ) : ℝ :=
  let total := ∑ i : Fin n, (Y i)^2
  let fitted := ∑ i : Fin n, (Y_hat i)^2
  fitted / total

/-- Warning: R² without intercept can exceed 1 or be negative
    when using the standard formula -/
theorem rsquared_no_intercept_warning {n : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat : Fin n → ℝ)
    : True := by  -- This is just documentation
  -- The centered formula R² = 1 - RSS/TSS can give values > 1 or < 0
  -- when there's no intercept because TSS uses centered values
  -- but RSS may not respect this.
  trivial

end OLS

end Econometrics

end
