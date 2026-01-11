import FormalProofs.DSL.CoreDefinitions

/-!
# FormalProofs/DSL/MomentFunctions.lean

## Paper Reference: Section 2.3 - Generalized Linear Models

This file formalizes the moment function framework for generalized linear models (GLMs):
- General moment function definition
- Linear regression moments
- Logistic regression moments
- Convex optimization characterization

### Key Concept

The DSL framework applies to any estimator defined by a moment condition:
  E[m(D; β*)] = 0

where m is the moment function. For GLMs, this arises from the first-order
conditions of likelihood maximization or M-estimation.

### Main GLM Moment Functions

| Model | Loss ℓ(D; β) | Moment m(D; β) |
|-------|--------------|----------------|
| Linear Regression | (Y - X'β)² | (Y - X'β)X |
| Logistic Regression | -Y·X'β + log(1+e^{X'β}) | (Y - expit(X'β))X |
| Poisson | -Y·X'β + e^{X'β} | (Y - e^{X'β})X |
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## General Moment Function
-/

/-- A moment function maps data and parameters to a score vector.
    m(D; β) : Data → Params → Score
    At the true parameter β*, we have E[m(D; β*)] = 0 -/
def MomentFunction (Data : Type*) (d : ℕ) :=
  Data → (Fin d → ℝ) → (Fin d → ℝ)

/-- Bundle of data for moment evaluation -/
structure DataPoint (Obs Mis : Type*) (d : ℕ) where
  d_obs : Obs
  d_mis : Mis
  Y : ℝ               -- Outcome variable
  X : Fin d → ℝ       -- Covariates

/-!
## Linear Regression Moment
-/

/-- Linear regression moment function.
    m(D; β) = (Y - X'β) · X

    This is the score from least squares: ∇_β (Y - X'β)² = -2(Y - X'β)X -/
def linearMoment {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) : Fin d → ℝ :=
  fun i => (Y - innerProduct X β) * X i

/-- Linear moment function as a MomentFunction -/
def linearMomentFn {Obs Mis : Type*} {d : ℕ}
    (outcome : Obs → Mis → ℝ)
    (covariates : Obs → Mis → Fin d → ℝ) :
    MomentFunction (DataPoint Obs Mis d) d :=
  fun dp β => linearMoment dp.Y dp.X β

/-- Residual for linear regression -/
def linearResidual {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) : ℝ :=
  Y - innerProduct X β

/-- Linear moment is residual times covariates -/
lemma linearMoment_eq_residual_mul {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) (i : Fin d) :
    linearMoment Y X β i = linearResidual Y X β * X i := rfl

/-!
## Logistic Regression Moment
-/

/-- Logistic regression moment function.
    m(D; β) = (Y - expit(X'β)) · X

    This is the score from logistic likelihood maximization. -/
def logisticMoment {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) : Fin d → ℝ :=
  fun i => (Y - expit (innerProduct X β)) * X i

/-- Logistic moment function as a MomentFunction -/
def logisticMomentFn {Obs Mis : Type*} {d : ℕ}
    (outcome : Obs → Mis → ℝ)
    (covariates : Obs → Mis → Fin d → ℝ) :
    MomentFunction (DataPoint Obs Mis d) d :=
  fun dp β => logisticMoment dp.Y dp.X β

/-- Logistic residual: Y - predicted probability -/
def logisticResidual {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) : ℝ :=
  Y - expit (innerProduct X β)

/-- Logistic moment is residual times covariates -/
lemma logisticMoment_eq_residual_mul {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) (i : Fin d) :
    logisticMoment Y X β i = logisticResidual Y X β * X i := rfl

/-!
## Poisson Regression Moment
-/

/-- Poisson regression moment function.
    m(D; β) = (Y - exp(X'β)) · X -/
def poissonMoment {d : ℕ} (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) : Fin d → ℝ :=
  fun i => (Y - Real.exp (innerProduct X β)) * X i

/-!
## Category Proportion (Simple Case)
-/

/-- Simple moment for estimating a proportion.
    For Y ∈ {0, 1}, the moment is (Y - μ) where μ is the proportion.
    This is a special case of linear regression with X = 1. -/
def proportionMoment (Y : ℝ) (μ : ℝ) : ℝ := Y - μ

/-!
## Moment Properties
-/

/-- A moment function is unbiased at β* if E[m(D; β*)] = 0 -/
def MomentUnbiased {Data : Type*} {d : ℕ}
    (m : MomentFunction Data d)
    (E : (Data → Fin d → ℝ) → Fin d → ℝ)  -- Expectation operator
    (β_star : Fin d → ℝ) : Prop :=
  ∀ i, E (fun D => m D β_star) i = 0

/-- A moment function satisfies regularity if ∂E[m]/∂β is invertible at β* -/
def MomentRegular {Data : Type*} {d : ℕ}
    (m : MomentFunction Data d)
    (jacobian : Matrix (Fin d) (Fin d) ℝ)  -- E[∂m/∂β] evaluated at β*
    : Prop :=
  jacobian.det ≠ 0

/-!
## Predicted vs True Moments
-/

/-- Moment evaluated with predicted missing variables -/
def momentPredicted {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred : Mis) (β : Fin d → ℝ) : Fin d → ℝ :=
  m (d_obs, d_mis_pred) β

/-- Moment evaluated with true missing variables -/
def momentTrue {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_true : Mis) (β : Fin d → ℝ) : Fin d → ℝ :=
  m (d_obs, d_mis_true) β

/-- Prediction error in moment function -/
def momentError {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_true : Mis) (β : Fin d → ℝ) : Fin d → ℝ :=
  fun i => momentPredicted m d_obs d_mis_pred β i - momentTrue m d_obs d_mis_true β i

/-!
## Simple Linear Regression (d=2 case)
-/

/-- Simple linear regression: Y = β₀ + β₁X + ε -/
def simpleLinearMoment (Y X : ℝ) (β₀ β₁ : ℝ) : Fin 2 → ℝ :=
  fun i => match i with
    | ⟨0, _⟩ => Y - (β₀ + β₁ * X)           -- Intercept moment
    | ⟨1, _⟩ => (Y - (β₀ + β₁ * X)) * X     -- Slope moment

/-!
## Fixed Effects Moment
-/

/-- Fixed effects linear regression moment.
    Y_it = α_i + X_it'β + ε_it
    After within-transformation: Ỹ_it = X̃_it'β + ε̃_it -/
def fixedEffectsMoment {d : ℕ}
    (Y_tilde : ℝ)      -- Within-transformed outcome
    (X_tilde : Fin d → ℝ)  -- Within-transformed covariates
    (β : Fin d → ℝ) : Fin d → ℝ :=
  linearMoment Y_tilde X_tilde β

end DSL

end
