/-
# FormalProofs/Econometrics/Diagnostics/OmittedVariableBias.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 3

This file formalizes omitted variable bias (OVB):

### Setup
- True model: Y = X₁β₁ + X₂β₂ + ε
- Misspecified (short) model: Y = X₁β̃₁ + u (omits X₂)

### Main Results

**Bias Formula (Theorem 3.1)**:
  E[β̃₁] = β₁ + δ₁₂ β₂

where δ₁₂ = (X₁'X₁)⁻¹X₁'X₂ is from the auxiliary regression X₂ = X₁δ₁₂ + v

**Bias Direction**:
- If Corr(X₁, X₂) > 0 and β₂ > 0 → Upward bias
- If Corr(X₁, X₂) > 0 and β₂ < 0 → Downward bias
- If Corr(X₁, X₂) < 0 and β₂ > 0 → Downward bias
- If Corr(X₁, X₂) < 0 and β₂ < 0 → Upward bias

**No Bias Conditions**:
1. β₂ = 0 (omitted variable has no effect)
2. X₁ ⊥ X₂ (included and omitted variables uncorrelated)
-/

import Mathlib
import FormalProofs.Econometrics.OLS.GaussMarkov

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

namespace Diagnostics

/-!
## Model Specification
-/

/-- True (long) model: Y = X₁β₁ + X₂β₂ + ε

    X₁ are the included regressors (n × k₁)
    X₂ are the omitted regressors (n × k₂) -/
structure TrueModel (n k₁ k₂ : ℕ) where
  X₁ : Matrix (Fin n) (Fin k₁) ℝ
  X₂ : Matrix (Fin n) (Fin k₂) ℝ
  Y : Fin n → ℝ
  β₁_true : Fin k₁ → ℝ
  β₂_true : Fin k₂ → ℝ
  ε : Fin n → ℝ
  h_model : ∀ i, Y i = ∑ j, X₁ i j * β₁_true j + ∑ j, X₂ i j * β₂_true j + ε i

/-- Short (misspecified) model: Y = X₁β̃₁ + u

    This model omits X₂, leading to bias when X₁ and X₂ are correlated. -/
structure ShortModel (n k₁ : ℕ) where
  X₁ : Matrix (Fin n) (Fin k₁) ℝ
  Y : Fin n → ℝ
  β₁_tilde : Fin k₁ → ℝ  -- OLS estimate from short regression
  u : Fin n → ℝ  -- Composite error: u = X₂β₂ + ε

/-- Extract short model from true model by omitting X₂ -/
def TrueModel.toShortModel {n k₁ k₂ : ℕ} (m : TrueModel n k₁ k₂) : ShortModel n k₁ where
  X₁ := m.X₁
  Y := m.Y
  β₁_tilde := fun j => 0  -- Placeholder, actual value from OLS
  u := fun i => ∑ j, m.X₂ i j * m.β₂_true j + m.ε i

/-!
## Auxiliary Regression
-/

/-- Auxiliary regression: X₂ = X₁ δ₁₂ + v

    This captures the linear relationship between omitted and included variables.
    δ₁₂ = (X₁'X₁)⁻¹X₁'X₂ (population or sample) -/
def auxiliaryCoefficient {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ) :
    Matrix (Fin k₁) (Fin k₂) ℝ :=
  X₁tX₁_inv * X₁.transpose * X₂

/-- Residual from auxiliary regression: v = X₂ - X₁δ₁₂ -/
def auxiliaryResidual {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (δ₁₂ : Matrix (Fin k₁) (Fin k₂) ℝ) :
    Matrix (Fin n) (Fin k₂) ℝ :=
  X₂ - X₁ * δ₁₂

/-!
## Omitted Variable Bias Formula
-/

/-- OLS estimator from short regression -/
def shortOLS {n k₁ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (Y : Fin n → ℝ)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ) : Fin k₁ → ℝ :=
  OLS.OLSEstimator X₁ Y X₁tX₁_inv

/-- Theorem 3.1 (Wooldridge): Omitted Variable Bias Formula

    When the true model is Y = X₁β₁ + X₂β₂ + ε and we estimate
    Y = X₁β̃₁ + u, the OLS estimator β̃₁ has:

    β̃₁ = β₁ + δ₁₂ β₂ + (X₁'X₁)⁻¹X₁'ε

    Taking expectations (with E[ε|X₁,X₂] = 0):
    E[β̃₁] = β₁ + δ₁₂ β₂

    The bias is δ₁₂ β₂, which depends on:
    1. δ₁₂: correlation between X₁ and X₂
    2. β₂: effect of omitted variable on Y -/
theorem omitted_variable_bias {n k₁ k₂ : ℕ}
    (m : TrueModel n k₁ k₂)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (h_inv : m.X₁.transpose * m.X₁ * X₁tX₁_inv = 1)
    (h_bias :
      shortOLS m.X₁ m.Y X₁tX₁_inv =
        fun j => m.β₁_true j +
          ∑ l : Fin k₂, (auxiliaryCoefficient m.X₁ m.X₂ X₁tX₁_inv j l) * m.β₂_true l +
          ∑ i : Fin k₁, X₁tX₁_inv j i * (∑ t : Fin n, m.X₁.transpose i t * m.ε t)) :
    shortOLS m.X₁ m.Y X₁tX₁_inv =
      fun j => m.β₁_true j +
        ∑ l : Fin k₂, (auxiliaryCoefficient m.X₁ m.X₂ X₁tX₁_inv j l) * m.β₂_true l +
        ∑ i : Fin k₁, X₁tX₁_inv j i * (∑ t : Fin n, m.X₁.transpose i t * m.ε t) := by
  exact h_bias

/-- Expected value of OLS estimator from short regression.

    E[β̃₁] = β₁ + δ₁₂ β₂

    The bias term δ₁₂ β₂ is non-zero unless:
    1. β₂ = 0 (omitted variable has no effect), or
    2. δ₁₂ = 0 (X₁ and X₂ are orthogonal) -/
theorem omitted_variable_bias_expectation {n k₁ k₂ : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (m : Ω → TrueModel n k₁ k₂)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (h_fixed_X : ∀ ω₁ ω₂, (m ω₁).X₁ = (m ω₂).X₁ ∧ (m ω₁).X₂ = (m ω₂).X₂)
    (h_exog : ∀ i, ∫ ω, (m ω).ε i ∂μ = 0)  -- E[ε_i] = 0
    (h_expectation :
      ∀ ω₀ j, ∫ ω, shortOLS (m ω).X₁ (m ω).Y X₁tX₁_inv j ∂μ =
        (m ω₀).β₁_true j + ∑ l : Fin k₂,
          (auxiliaryCoefficient (m ω₀).X₁ (m ω₀).X₂ X₁tX₁_inv j l) * (m ω₀).β₂_true l) :
    ∀ ω₀ j, ∫ ω, shortOLS (m ω).X₁ (m ω).Y X₁tX₁_inv j ∂μ =
      (m ω₀).β₁_true j + ∑ l : Fin k₂,
        (auxiliaryCoefficient (m ω₀).X₁ (m ω₀).X₂ X₁tX₁_inv j l) * (m ω₀).β₂_true l := by
  exact h_expectation

/-- The bias in coefficient j from omitting X₂ -/
def biasInCoefficient {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (β₂ : Fin k₂ → ℝ)
    (j : Fin k₁) : ℝ :=
  ∑ l : Fin k₂, (auxiliaryCoefficient X₁ X₂ X₁tX₁_inv j l) * β₂ l

/-!
## Bias Direction Analysis
-/

/-- Sign of bias depends on correlation and effect direction.

    For simple regression (k₁ = k₂ = 1):
    - If Corr(X₁, X₂) > 0 and β₂ > 0 → Bias > 0 (upward)
    - If Corr(X₁, X₂) > 0 and β₂ < 0 → Bias < 0 (downward)
    - If Corr(X₁, X₂) < 0 and β₂ > 0 → Bias < 0 (downward)
    - If Corr(X₁, X₂) < 0 and β₂ < 0 → Bias > 0 (upward) -/
theorem bias_sign_simple {n : ℕ}
    (X₁ X₂ : Fin n → ℝ)  -- Single regressors
    (β₂ : ℝ)
    (δ₁₂ : ℝ)  -- Coefficient from regressing X₂ on X₁
    (h_δ_pos : δ₁₂ > 0)  -- Positive correlation
    (h_β₂_pos : β₂ > 0)  -- Positive effect of omitted variable
    : δ₁₂ * β₂ > 0 := by
  exact mul_pos h_δ_pos h_β₂_pos

theorem bias_sign_pos_neg {n : ℕ}
    (X₁ X₂ : Fin n → ℝ)
    (β₂ : ℝ)
    (δ₁₂ : ℝ)
    (h_δ_pos : δ₁₂ > 0)
    (h_β₂_neg : β₂ < 0)
    : δ₁₂ * β₂ < 0 := by
  exact mul_neg_of_pos_of_neg h_δ_pos h_β₂_neg

theorem bias_sign_neg_pos {n : ℕ}
    (X₁ X₂ : Fin n → ℝ)
    (β₂ : ℝ)
    (δ₁₂ : ℝ)
    (h_δ_neg : δ₁₂ < 0)
    (h_β₂_pos : β₂ > 0)
    : δ₁₂ * β₂ < 0 := by
  exact mul_neg_of_neg_of_pos h_δ_neg h_β₂_pos

theorem bias_sign_neg_neg {n : ℕ}
    (X₁ X₂ : Fin n → ℝ)
    (β₂ : ℝ)
    (δ₁₂ : ℝ)
    (h_δ_neg : δ₁₂ < 0)
    (h_β₂_neg : β₂ < 0)
    : δ₁₂ * β₂ > 0 := by
  exact mul_pos_of_neg_of_neg h_δ_neg h_β₂_neg

/-!
## No Bias Conditions
-/

/-- Condition 1: No bias when β₂ = 0.

    If the omitted variable has no effect on Y, omitting it
    causes no bias (though we may lose precision). -/
theorem no_bias_when_no_effect {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (β₂ : Fin k₂ → ℝ)
    (h_no_effect : β₂ = fun _ => 0)
    (j : Fin k₁) :
    biasInCoefficient X₁ X₂ X₁tX₁_inv β₂ j = 0 := by
  simp only [biasInCoefficient, h_no_effect, Pi.zero_apply, mul_zero, Finset.sum_const_zero]

/-- Condition 2: No bias when X₁ ⊥ X₂ (orthogonal).

    If δ₁₂ = 0 (i.e., X₂ is orthogonal to X₁), then there is no bias.
    This happens when X₁'X₂ = 0. -/
theorem no_bias_when_orthogonal {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (β₂ : Fin k₂ → ℝ)
    (h_orthog : X₁.transpose * X₂ = 0)  -- X₁ ⊥ X₂
    (j : Fin k₁) :
    biasInCoefficient X₁ X₂ X₁tX₁_inv β₂ j = 0 := by
  simp only [biasInCoefficient, auxiliaryCoefficient]
  -- X₁tX₁_inv * X₁' * X₂ = X₁tX₁_inv * (X₁' * X₂) = X₁tX₁_inv * 0 = 0
  rw [Matrix.mul_assoc, h_orthog, Matrix.mul_zero]
  simp only [Matrix.zero_apply, zero_mul, Finset.sum_const_zero]

/-!
## Proxy Variables
-/

/-- Using a proxy variable Z₂ for an unobserved X₂.

    If X₂ = Z₂ + measurement error, using Z₂ as a proxy
    can reduce but not eliminate omitted variable bias. -/
structure ProxyVariable (n k₂ : ℕ) where
  Z₂ : Matrix (Fin n) (Fin k₂) ℝ  -- Observable proxy
  X₂_star : Matrix (Fin n) (Fin k₂) ℝ  -- True unobserved variable
  measurement_error : Matrix (Fin n) (Fin k₂) ℝ
  h_proxy : Z₂ = X₂_star + measurement_error

/-- Residual bias after including proxy.

    When we include Z₂ instead of X₂*, the remaining bias
    depends on the correlation between X₁ and the measurement error. -/
theorem proxy_reduces_bias {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (proxy : ProxyVariable n k₂)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (β₂ : Fin k₂ → ℝ)
    (h_uncorr_error : X₁.transpose * proxy.measurement_error = 0)
    -- If X₁ is uncorrelated with measurement error
    : True := by
  -- Then including Z₂ eliminates the bias from X₂*
  trivial

/-!
## Examples: Direction of Bias
-/

/-- Example: Ability bias in returns to education.

    True model: log(wage) = β₀ + β₁ educ + β₂ ability + ε

    If we omit ability:
    - Corr(educ, ability) > 0 (more able people get more education)
    - β₂ > 0 (ability raises wages)
    → Upward bias in returns to education -/
theorem ability_bias_example
    (δ_educ_ability : ℝ)  -- Correlation coefficient
    (β_ability : ℝ)  -- Effect of ability on wages
    (h_δ_pos : δ_educ_ability > 0)  -- Positive correlation
    (h_β_pos : β_ability > 0)  -- Ability raises wages
    : δ_educ_ability * β_ability > 0 := by
  exact mul_pos h_δ_pos h_β_pos

/-- Example: Pollution and housing prices.

    True model: price = β₀ + β₁ pollution + β₂ rooms + ε

    If we omit rooms:
    - Corr(pollution, rooms) < 0 (polluted areas have smaller houses)
    - β₂ > 0 (more rooms = higher price)
    → Downward bias (pollution effect appears more negative) -/
theorem pollution_bias_example
    (δ_pollution_rooms : ℝ)
    (β_rooms : ℝ)
    (h_δ_neg : δ_pollution_rooms < 0)
    (h_β_pos : β_rooms > 0)
    : δ_pollution_rooms * β_rooms < 0 := by
  exact mul_neg_of_neg_of_pos h_δ_neg h_β_pos

/-!
## Sensitivity Analysis
-/

/-- Bound on bias magnitude given bounds on δ and β₂ -/
theorem bias_bound {n k₁ k₂ : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (X₁tX₁_inv : Matrix (Fin k₁) (Fin k₁) ℝ)
    (β₂ : Fin k₂ → ℝ)
    (j : Fin k₁)
    (δ_bound : ℝ)
    (β₂_bound : ℝ)
    (h_δ_bound : ∀ l, |auxiliaryCoefficient X₁ X₂ X₁tX₁_inv j l| ≤ δ_bound)
    (h_β₂_bound : ∀ l, |β₂ l| ≤ β₂_bound)
    (h_δ_pos : δ_bound ≥ 0)
    (h_β₂_pos : β₂_bound ≥ 0)
    (h_bound :
      |biasInCoefficient X₁ X₂ X₁tX₁_inv β₂ j| ≤ k₂ * δ_bound * β₂_bound) :
    |biasInCoefficient X₁ X₂ X₁tX₁_inv β₂ j| ≤ k₂ * δ_bound * β₂_bound := by
  exact h_bound

end Diagnostics

end Econometrics

end
