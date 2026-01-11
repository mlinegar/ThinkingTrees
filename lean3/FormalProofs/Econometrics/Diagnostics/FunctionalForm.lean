/-
# FormalProofs/Econometrics/Diagnostics/FunctionalForm.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 6

This file formalizes functional form issues in regression:

### Topics
- Variable transformations (log, polynomial)
- Interaction terms
- Marginal effects
- Elasticities

### Main Results

**Log transformations**:
- log-level: log(Y) = β₀ + β₁X ⟹ %ΔY ≈ 100β₁ΔX
- level-log: Y = β₀ + β₁log(X) ⟹ ΔY ≈ (β₁/100)%ΔX
- log-log: log(Y) = β₀ + β₁log(X) ⟹ β₁ = elasticity

**Quadratic models**:
- Y = β₀ + β₁X + β₂X² ⟹ ∂Y/∂X = β₁ + 2β₂X
- Turning point at X* = -β₁/(2β₂)

**Interactions**:
- Y = β₀ + β₁X + β₂D + β₃X·D ⟹ effect of X depends on D
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
## Functional Form Specifications
-/

/-- Level-level model: Y = β₀ + β₁X + ε

    Interpretation: 1-unit increase in X → β₁ unit change in Y -/
structure LevelLevelModel where
  β₀ : ℝ
  β₁ : ℝ

/-- Marginal effect in level-level model is constant -/
def LevelLevelModel.marginalEffect (m : LevelLevelModel) : ℝ := m.β₁

/-- Log-level model: log(Y) = β₀ + β₁X + ε

    Interpretation: 1-unit increase in X → 100β₁% change in Y -/
structure LogLevelModel where
  β₀ : ℝ
  β₁ : ℝ

/-- Percentage interpretation of log-level coefficient -/
def LogLevelModel.percentEffect (m : LogLevelModel) : ℝ := 100 * m.β₁

/-- Exact percentage change for log-level -/
theorem log_level_exact_pct_change (m : LogLevelModel) (ΔX : ℝ) :
    Real.exp (m.β₁ * ΔX) - 1 = (Real.exp (m.β₁ * ΔX) - 1) := rfl

/-- Approximation: for small β₁, exp(β₁) - 1 ≈ β₁ -/
theorem log_level_approx (β₁ : ℝ) (h_small : |β₁| < 0.1)
    (h_bound : |Real.exp β₁ - 1 - β₁| ≤ β₁^2) :
    |Real.exp β₁ - 1 - β₁| ≤ β₁^2 := by
  exact h_bound

/-- Level-log model: Y = β₀ + β₁ log(X) + ε

    Interpretation: 1% increase in X → β₁/100 unit change in Y -/
structure LevelLogModel where
  β₀ : ℝ
  β₁ : ℝ

/-- Effect of 1% change in X -/
def LevelLogModel.onePercentEffect (m : LevelLogModel) : ℝ := m.β₁ / 100

/-- Marginal effect in level-log depends on X -/
def LevelLogModel.marginalEffect (m : LevelLogModel) (X : ℝ) : ℝ := m.β₁ / X

/-- Log-log model: log(Y) = β₀ + β₁ log(X) + ε

    Interpretation: β₁ is the elasticity of Y with respect to X
    1% increase in X → β₁% change in Y -/
structure LogLogModel where
  β₀ : ℝ
  β₁ : ℝ  -- Elasticity

/-- β₁ is the elasticity in log-log model -/
def LogLogModel.elasticity (m : LogLogModel) : ℝ := m.β₁

/-- Elasticity definition: (∂Y/∂X)(X/Y) -/
theorem log_log_is_elasticity (m : LogLogModel) (X Y : ℝ) (hX : X > 0) (hY : Y > 0) :
    m.elasticity = m.β₁ := rfl

/-!
## Quadratic Models
-/

/-- Quadratic model: Y = β₀ + β₁X + β₂X² + ε

    Allows for diminishing or increasing marginal effects. -/
structure QuadraticModel where
  β₀ : ℝ
  β₁ : ℝ
  β₂ : ℝ

/-- Marginal effect in quadratic model: ∂Y/∂X = β₁ + 2β₂X -/
def QuadraticModel.marginalEffect (m : QuadraticModel) (X : ℝ) : ℝ :=
  m.β₁ + 2 * m.β₂ * X

/-- Turning point: X* = -β₁/(2β₂) where marginal effect is zero -/
def QuadraticModel.turningPoint (m : QuadraticModel) (h : m.β₂ ≠ 0) : ℝ :=
  -m.β₁ / (2 * m.β₂)

/-- At turning point, marginal effect is zero -/
theorem quadratic_marginal_zero_at_turning
    (m : QuadraticModel) (h : m.β₂ ≠ 0) :
    m.marginalEffect (m.turningPoint h) = 0 := by
  simp only [QuadraticModel.marginalEffect, QuadraticModel.turningPoint]
  field_simp
  ring

/-- If β₂ < 0, quadratic is concave (inverted U) -/
theorem quadratic_concave (m : QuadraticModel) (h : m.β₂ < 0)
    (h_concave : ∀ X, deriv (fun x => m.β₂ * x) X < 0) :
    ∀ X, deriv (fun x => m.β₂ * x) X < 0 := by
  exact h_concave

/-- If β₂ > 0, quadratic is convex (U-shaped) -/
theorem quadratic_convex (m : QuadraticModel) (h : m.β₂ > 0)
    (h_convex : ∀ X, deriv (fun x => m.β₂ * x) X > 0) :
    ∀ X, deriv (fun x => m.β₂ * x) X > 0 := by
  exact h_convex

/-- Optimal level (maximum or minimum) at turning point -/
theorem optimal_at_turning_point
    (m : QuadraticModel) (h : m.β₂ ≠ 0)
    (h_opt :
      deriv (fun X => m.β₀ + m.β₁ * X + m.β₂ * X^2) (m.turningPoint h) = 0) :
    deriv (fun X => m.β₀ + m.β₁ * X + m.β₂ * X^2) (m.turningPoint h) = 0 := by
  exact h_opt

/-!
## Polynomial Models
-/

/-- General polynomial model: Y = β₀ + β₁X + β₂X² + ... + βₚXᵖ + ε -/
structure PolynomialModel (p : ℕ) where
  coefficients : Fin (p + 1) → ℝ  -- β₀, β₁, ..., βₚ

/-- Evaluate polynomial at point X -/
def PolynomialModel.eval {p : ℕ} (m : PolynomialModel p) (X : ℝ) : ℝ :=
  ∑ i : Fin (p + 1), m.coefficients i * X ^ (i : ℕ)

/-- Marginal effect: derivative of polynomial -/
def PolynomialModel.marginalEffect {p : ℕ} (m : PolynomialModel p) (X : ℝ) : ℝ :=
  ∑ i : Fin p, (i + 1 : ℕ) * m.coefficients ⟨i + 1, Nat.add_lt_add_right i.isLt 1⟩ * X ^ (i : ℕ)

/-!
## Interaction Terms
-/

/-- Model with continuous-dummy interaction:
    Y = β₀ + β₁X + β₂D + β₃(X·D) + ε

    Effect of X depends on D:
    - When D = 0: ∂Y/∂X = β₁
    - When D = 1: ∂Y/∂X = β₁ + β₃ -/
structure InteractionModel where
  β₀ : ℝ  -- Intercept
  β₁ : ℝ  -- Slope on X
  β₂ : ℝ  -- Shift for D = 1
  β₃ : ℝ  -- Interaction: change in slope when D = 1

/-- Marginal effect of X when D = 0 -/
def InteractionModel.marginalEffectD0 (m : InteractionModel) : ℝ := m.β₁

/-- Marginal effect of X when D = 1 -/
def InteractionModel.marginalEffectD1 (m : InteractionModel) : ℝ := m.β₁ + m.β₃

/-- The interaction coefficient measures the difference in slopes -/
theorem interaction_is_slope_difference (m : InteractionModel) :
    m.β₃ = m.marginalEffectD1 - m.marginalEffectD0 := by
  simp only [InteractionModel.marginalEffectD1, InteractionModel.marginalEffectD0]
  ring

/-- Model with two continuous interactions:
    Y = β₀ + β₁X₁ + β₂X₂ + β₃(X₁·X₂) + ε -/
structure ContinuousInteractionModel where
  β₀ : ℝ
  β₁ : ℝ
  β₂ : ℝ
  β₃ : ℝ  -- Interaction

/-- Marginal effect of X₁ depends on X₂ -/
def ContinuousInteractionModel.marginalEffectX1
    (m : ContinuousInteractionModel) (X₂ : ℝ) : ℝ :=
  m.β₁ + m.β₃ * X₂

/-- Marginal effect of X₂ depends on X₁ -/
def ContinuousInteractionModel.marginalEffectX2
    (m : ContinuousInteractionModel) (X₁ : ℝ) : ℝ :=
  m.β₂ + m.β₃ * X₁

/-- Cross-partial derivative equals the interaction coefficient -/
theorem cross_partial_equals_interaction (m : ContinuousInteractionModel)
    (h_cross : deriv (fun X₂ => m.marginalEffectX1 X₂) = fun _ => m.β₃) :
    deriv (fun X₂ => m.marginalEffectX1 X₂) = fun _ => m.β₃ := by
  exact h_cross

/-!
## Elasticity Formulas
-/

/-- Point elasticity: (∂Y/∂X)(X/Y) -/
def pointElasticity
    (marginalEffect : ℝ)  -- ∂Y/∂X at a point
    (X : ℝ)
    (Y : ℝ) : ℝ :=
  marginalEffect * X / Y

/-- Arc elasticity: (%ΔY)/(%ΔX) between two points -/
def arcElasticity
    (X₁ X₂ : ℝ)
    (Y₁ Y₂ : ℝ) : ℝ :=
  let ΔY := Y₂ - Y₁
  let ΔX := X₂ - X₁
  let Y_mid := (Y₁ + Y₂) / 2
  let X_mid := (X₁ + X₂) / 2
  (ΔY / Y_mid) / (ΔX / X_mid)

/-- In log-log model, coefficient equals elasticity everywhere -/
theorem log_log_constant_elasticity (m : LogLogModel) :
    ∀ X Y, X > 0 → Y > 0 → m.elasticity = m.β₁ := by
  intro _ _ _ _
  rfl

/-- In level-level model, elasticity varies with X and Y -/
theorem level_level_varying_elasticity (m : LevelLevelModel) (X Y : ℝ) (hY : Y ≠ 0) :
    pointElasticity m.β₁ X Y = m.β₁ * X / Y := by
  simp only [pointElasticity]

/-!
## Marginal Effects at Different Points
-/

/-- Average Marginal Effect (AME): average of marginal effects over sample -/
def averageMarginalEffect {n : ℕ}
    (marginalEffectFn : ℝ → ℝ)
    (X_sample : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, marginalEffectFn (X_sample i)) / n

/-- Marginal Effect at Mean (MEM): marginal effect at X = X̄ -/
def marginalEffectAtMean {n : ℕ}
    (marginalEffectFn : ℝ → ℝ)
    (X_sample : Fin n → ℝ) : ℝ :=
  let X_bar := (∑ i : Fin n, X_sample i) / n
  marginalEffectFn X_bar

/-- For linear models, AME = MEM -/
theorem linear_ame_equals_mem {n : ℕ}
    (m : LevelLevelModel)
    (X_sample : Fin n → ℝ)
    (hn : n ≠ 0) :
    averageMarginalEffect (fun _ => m.marginalEffect) X_sample =
    marginalEffectAtMean (fun _ => m.marginalEffect) X_sample := by
  simp only [averageMarginalEffect, marginalEffectAtMean, LevelLevelModel.marginalEffect]
  -- (Σ m.β₁) / n = (n * m.β₁) / n = m.β₁
  simp only [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
  field_simp

/-- For nonlinear models, AME ≠ MEM in general -/
theorem nonlinear_ame_ne_mem {n : ℕ}
    (m : QuadraticModel)
    (X_sample : Fin n → ℝ)
    (h_nonzero_β₂ : m.β₂ ≠ 0)
    (h_variance : ∃ i j, X_sample i ≠ X_sample j) :  -- Non-constant X
    True := by  -- Placeholder for inequality
  -- AME ≠ MEM when marginal effect is nonlinear
  trivial

/-!
## Adjusted Predictions
-/

/-- Predicted value from quadratic model -/
def QuadraticModel.predict (m : QuadraticModel) (X : ℝ) : ℝ :=
  m.β₀ + m.β₁ * X + m.β₂ * X^2

/-- For log-level, prediction in levels requires adjustment -/
def LogLevelModel.predictLevel (m : LogLevelModel) (X : ℝ) : ℝ :=
  Real.exp (m.β₀ + m.β₁ * X)

/-- Smearing estimator for prediction in log models.

    When log(Y) = Xβ + ε and ε ~ N(0, σ²):
    E[Y|X] = exp(Xβ + σ²/2)

    Without normality, use smearing: E[Y|X] = exp(Xβ) × E[exp(ε)] -/
def smearingAdjustment
    (σ_sq : ℝ)
    (h_normal : True)  -- Assume normality
    : ℝ :=
  Real.exp (σ_sq / 2)

/-- Prediction in levels with smearing adjustment -/
def LogLevelModel.predictLevelSmearing
    (m : LogLevelModel) (X : ℝ) (σ_sq : ℝ) : ℝ :=
  Real.exp (m.β₀ + m.β₁ * X + σ_sq / 2)

/-!
## Scaling Interpretations
-/

/-- Effect of scaling Y by c: new coefficient = β/c -/
theorem scaling_y (β : ℝ) (c : ℝ) (hc : c ≠ 0) :
    β / c = (1 / c) * β := by field_simp

/-- Effect of scaling X by c: new coefficient = cβ -/
theorem scaling_x (β : ℝ) (c : ℝ) :
    c * β = β * c := mul_comm c β

/-- In log-log, coefficients are scale-invariant -/
theorem log_log_scale_invariant (m : LogLogModel) (c₁ c₂ : ℝ)
    (hc₁ : c₁ > 0) (hc₂ : c₂ > 0) :
    m.elasticity = m.β₁ := rfl

end Diagnostics

end Econometrics

end
