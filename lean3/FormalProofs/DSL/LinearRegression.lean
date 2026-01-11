/-
# FormalProofs/DSL/LinearRegression.lean

## Paper Reference: Section 3.1, Equation 6

This file formalizes DSL for linear regression:
- Standard OLS on design-adjusted outcomes
- Coefficient interpretation
- Standard errors and inference

### Main Formula (Equation 6)

  β̂_DSL = (X'X)⁻¹X'Ỹ

where Ỹ is the vector of design-adjusted outcomes:
  Ỹᵢ = Ŷᵢ - (Rᵢ/πᵢ)(Ŷᵢ - Yᵢ)

This is simply OLS on the adjusted outcomes.
-/

import FormalProofs.DSL.DSLEstimator
import FormalProofs.DSL.AsymptoticTheory

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## Linear Regression Data
-/

/-- Data for DSL linear regression -/
structure DSLLinearData (n d : ℕ) where
  /-- Covariate matrix X (n × d) -/
  X : Fin n → Fin d → ℝ
  /-- Predicted outcomes Ŷ -/
  Y_pred : Fin n → ℝ
  /-- True outcomes Y (for sampled units; 0 otherwise) -/
  Y_true : Fin n → ℝ
  /-- Sampling indicators R -/
  R : Fin n → SamplingIndicator
  /-- Sampling probabilities π -/
  π : Fin n → ℝ
  /-- Positivity condition -/
  h_π_pos : ∀ i, π i > 0

/-- Compute design-adjusted outcomes for all observations -/
def DSLLinearData.Y_tilde {n d : ℕ} (data : DSLLinearData n d) : Fin n → ℝ :=
  fun i => designAdjustedOutcome (data.Y_pred i) (data.Y_true i) (data.R i) (data.π i)

/-!
## OLS Computation
-/

/-- X'X matrix (Gram matrix) -/
def XtX {n d : ℕ} (X : Fin n → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  fun i j => ∑ k : Fin n, X k i * X k j

/-- X'Y vector -/
def XtY' {n d : ℕ} (X : Fin n → Fin d → ℝ) (Y : Fin n → ℝ) : Fin d → ℝ :=
  fun i => ∑ k : Fin n, X k i * Y k

/-- DSL linear regression estimator.

    β̂_DSL = (X'X)⁻¹X'Ỹ

    This is standard OLS with the adjusted outcomes. -/
def DSLLinearEstimator {n d : ℕ}
    (data : DSLLinearData n d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ) : Fin d → ℝ :=
  let XtY_tilde := XtY' data.X data.Y_tilde
  fun i => ∑ j : Fin d, XtX_inv i j * XtY_tilde j

/-!
## Equivalence to Standard OLS on Adjusted Outcomes
-/

/-- DSL linear regression is OLS on Ỹ.

    This is important: no new software needed! Just compute Ỹ
    and use standard OLS routines. -/
theorem DSL_is_OLS_on_adjusted
    {n d : ℕ}
    (data : DSLLinearData n d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ)
    : DSLLinearEstimator data XtX_inv =
      fun i => ∑ j : Fin d, XtX_inv i j * XtY' data.X data.Y_tilde j := by
  rfl

/-!
## Residuals and Fitted Values
-/

/-- Fitted values from DSL linear regression -/
def fittedValues {n d : ℕ}
    (data : DSLLinearData n d)
    (β_hat : Fin d → ℝ) : Fin n → ℝ :=
  fun i => innerProduct (data.X i) β_hat

/-- Residuals from DSL linear regression -/
def residuals {n d : ℕ}
    (data : DSLLinearData n d)
    (β_hat : Fin d → ℝ) : Fin n → ℝ :=
  fun i => data.Y_tilde i - fittedValues data β_hat i

/-- Residual sum of squares -/
def RSS {n d : ℕ}
    (data : DSLLinearData n d)
    (β_hat : Fin d → ℝ) : ℝ :=
  ∑ i : Fin n, (residuals data β_hat i)^2

/-!
## Standard Errors
-/

/-- Estimated variance of residuals -/
def σ_hat_sq {n d : ℕ}
    (data : DSLLinearData n d)
    (β_hat : Fin d → ℝ) : ℝ :=
  RSS data β_hat / (n - d)

/-- Estimated variance-covariance matrix (homoskedastic case) -/
def varianceMatrix_homoskedastic {n d : ℕ}
    (data : DSLLinearData n d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ)
    (β_hat : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  (σ_hat_sq data β_hat) • XtX_inv

/-- Standard errors (homoskedastic) -/
def standardErrors_homoskedastic {n d : ℕ}
    (data : DSLLinearData n d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ)
    (β_hat : Fin d → ℝ) : Fin d → ℝ :=
  fun i => Real.sqrt (varianceMatrix_homoskedastic data XtX_inv β_hat i i)

/-!
## Heteroskedasticity-Robust Standard Errors
-/

/-- Meat matrix for HC standard errors: X' diag(e²) X -/
def meatMatrix {n d : ℕ}
    (data : DSLLinearData n d)
    (β_hat : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  let e := residuals data β_hat
  fun i j => ∑ k : Fin n, (data.X k i) * (e k)^2 * (data.X k j)

/-- HC (White) robust variance-covariance matrix -/
def varianceMatrix_HC {n d : ℕ}
    (data : DSLLinearData n d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ)
    (β_hat : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  let meat := meatMatrix data β_hat
  XtX_inv * meat * XtX_inv

/-- Standard errors (heteroskedasticity-robust) -/
def standardErrors_HC {n d : ℕ}
    (data : DSLLinearData n d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ)
    (β_hat : Fin d → ℝ) : Fin d → ℝ :=
  fun i => Real.sqrt (varianceMatrix_HC data XtX_inv β_hat i i)

/-!
## Inference
-/

/-- t-statistic for coefficient j -/
def tStatistic {n d : ℕ}
    (β_hat : Fin d → ℝ)
    (SE : Fin d → ℝ)
    (j : Fin d)
    (β_null : ℝ := 0) : ℝ :=
  (β_hat j - β_null) / SE j

/-- 95% confidence interval for coefficient j -/
def confidenceInterval95 {n d : ℕ}
    (β_hat : Fin d → ℝ)
    (SE : Fin d → ℝ)
    (j : Fin d) : ℝ × ℝ :=
  (β_hat j - 1.96 * SE j, β_hat j + 1.96 * SE j)

/-!
## Simple Linear Regression (d = 2)
-/

/-- Simple linear regression: Y = β₀ + β₁X + ε -/
structure SimpleLinearData (n : ℕ) where
  X : Fin n → ℝ          -- Single covariate
  Y_pred : Fin n → ℝ
  Y_true : Fin n → ℝ
  R : Fin n → SamplingIndicator
  π : Fin n → ℝ
  h_π_pos : ∀ i, π i > 0

/-- Convert to full linear data (with intercept) -/
def SimpleLinearData.toLinearData {n : ℕ} (data : SimpleLinearData n) :
    DSLLinearData n 2 where
  X := fun i => fun j => match j with
    | ⟨0, _⟩ => 1        -- Intercept
    | ⟨1, _⟩ => data.X i -- Slope covariate
  Y_pred := data.Y_pred
  Y_true := data.Y_true
  R := data.R
  π := data.π
  h_π_pos := data.h_π_pos

/-- Simple regression slope formula:
    β₁ = ∑(Xᵢ - X̄)(Ỹᵢ - Ỹ̄) / ∑(Xᵢ - X̄)² -/
def simpleRegressionSlope {n : ℕ} (data : SimpleLinearData n) : ℝ :=
  let X_bar := (∑ i : Fin n, data.X i) / n
  let Y_tilde := fun i => designAdjustedOutcome (data.Y_pred i) (data.Y_true i) (data.R i) (data.π i)
  let Y_bar := (∑ i : Fin n, Y_tilde i) / n
  let num := ∑ i : Fin n, (data.X i - X_bar) * (Y_tilde i - Y_bar)
  let denom := ∑ i : Fin n, (data.X i - X_bar)^2
  num / denom

/-!
## Properties
-/

/-- Linear moment function over (X, Y) pairs (d-dimensional covariates). -/
def linearMomentPair {d : ℕ} : MomentFunction ((Fin d → ℝ) × ℝ) d :=
  fun D β => linearMoment D.2 D.1 β

/-- DSL linear regression is consistent under M-estimation conditions. -/
theorem DSL_linear_unbiased
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Con : Type*} {d : ℕ}
    (axioms :
      MEstimationAxioms Ω ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling (Fin d → ℝ) ℝ Con)
    (β_star : Fin d → ℝ)
    (reg : RegularityConditions
      ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) d)
    (h_unbiased :
      MomentUnbiased (DSLMomentFromData (linearMomentPair (d := d))) axioms.E β_star)
    (data_seq : ℕ →
      Ω → List ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est :
      IsMEstimatorSeq (DSLMomentFromData (linearMomentPair (d := d)))
        data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact DSL_consistent μ axioms dbs (linearMomentPair (d := d)) β_star reg h_unbiased data_seq
    β_hat_seq h_est

/-- DSL standard errors provide valid coverage.

    Under Assumption 1, the 95% CI achieves nominal coverage
    asymptotically, regardless of prediction accuracy. -/
theorem DSL_linear_valid_coverage
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Con : Type*} {d : ℕ}
    (axioms :
      MEstimationAxioms Ω ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling (Fin d → ℝ) ℝ Con)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions
      ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) d)
    (h_unbiased :
      MomentUnbiased (DSLMomentFromData (linearMomentPair (d := d))) axioms.E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (h_α : 0 < α ∧ α < 1)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    : AsymptoticCoverage μ CI_seq β_star α := by
  exact DSL_valid_coverage μ axioms coverage_axioms dbs (linearMomentPair (d := d)) β_star V reg
    h_unbiased CI_seq α h_α centered_scaled_seq

end DSL

end
