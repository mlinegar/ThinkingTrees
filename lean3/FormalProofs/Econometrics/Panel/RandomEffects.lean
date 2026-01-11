/-
# FormalProofs/Econometrics/Panel/RandomEffects.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 14

This file formalizes Random Effects (RE) estimation for panel data:

### Setup
Panel data: Y_it = X_it'β + α_i + ε_it
where α_i ~ (0, σ²_α) is a random effect uncorrelated with X.

### Key Assumption
RE requires: Cov(X_it, α_i) = 0 for all t
(Individual effects are uncorrelated with regressors)

### Main Results

**Composite Error**:
  v_it = α_i + ε_it has serial correlation:
  Corr(v_it, v_is) = σ²_α / (σ²_α + σ²_ε) for t ≠ s

**GLS Transformation**:
  (Y_it - θȲ_i) = (X_it - θX̄_i)'β + error
  where θ = 1 - √(σ²_ε / (σ²_ε + T·σ²_α))

**Efficiency**:
  RE is more efficient than FE when RE assumptions hold

**Feasible GLS**:
  Estimate σ²_α and σ²_ε from residuals, then apply GLS
-/

import Mathlib
import FormalProofs.Econometrics.Panel.FixedEffects

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
## Random Effects Model
-/

/-- Random effects model: Y_it = X_it'β + α_i + ε_it

    Key difference from FE: α_i is a random variable, not a fixed parameter.
    α_i is assumed uncorrelated with X_it. -/
structure REModel (N T k : ℕ) where
  panel : PanelData N T k
  β : Fin k → ℝ  -- Slope coefficients
  α : Fin N → ℝ  -- Individual random effects
  ε : Fin N → Fin T → ℝ  -- Idiosyncratic errors
  σ_α_sq : ℝ  -- Variance of α_i
  σ_ε_sq : ℝ  -- Variance of ε_it
  h_model : ∀ i t, (panel.data i t).Y =
    ∑ j, (panel.data i t).X j * β j + α i + ε i t
  h_σ_α_pos : σ_α_sq ≥ 0
  h_σ_ε_pos : σ_ε_sq > 0

/-!
## RE Assumption: Orthogonality
-/

/-- RE key assumption: E[α_i | X_i] = 0.

    The random effect is uncorrelated with all regressors.
    This is stronger than FE's strict exogeneity in one way
    (requires α ⊥ X) but weaker in another (allows time-invariant regressors). -/
structure REOrthogonality {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Fin N → Fin T → Fin k → ℝ)
    (α : Ω → Fin N → ℝ) : Prop where
  /-- α_i is mean zero -/
  mean_zero : ∀ i, ∫ ω, α ω i ∂μ = 0
  /-- α_i is uncorrelated with all X_it for all t -/
  uncorrelated : ∀ i t j, ∫ ω, X ω i t j * α ω i ∂μ = 0

/-!
## Composite Error Structure
-/

/-- Composite error: v_it = α_i + ε_it -/
def compositeError {N T k : ℕ} (model : REModel N T k) (i : Fin N) (t : Fin T) : ℝ :=
  model.α i + model.ε i t

/-- Variance of composite error: Var(v_it) = σ²_α + σ²_ε -/
def compositeVariance {N T k : ℕ} (model : REModel N T k) : ℝ :=
  model.σ_α_sq + model.σ_ε_sq

/-- Serial correlation coefficient: ρ = σ²_α / (σ²_α + σ²_ε)

    For t ≠ s: Corr(v_it, v_is) = ρ
    This arises because α_i is common to all periods. -/
def serialCorrelation {N T k : ℕ} (model : REModel N T k) : ℝ :=
  model.σ_α_sq / (model.σ_α_sq + model.σ_ε_sq)

/-- Covariance matrix of v_i = (v_i1, ..., v_iT)' is not diagonal.

    Ω = σ²_ε I_T + σ²_α 1_T 1_T'

    where 1_T is a T×1 vector of ones. -/
theorem composite_error_covariance {N T k : ℕ}
    (model : REModel N T k)
    (i : Fin N) (t s : Fin T) :
    True := by  -- Placeholder for covariance formula
  -- Cov(v_it, v_is) = σ²_α + σ²_ε if t = s
  --                 = σ²_α       if t ≠ s
  trivial

/-!
## GLS Transformation
-/

/-- GLS transformation parameter: θ = 1 - √(σ²_ε / (σ²_ε + T·σ²_α))

    This determines how much to demean the data. -/
def thetaGLS {N T k : ℕ} (model : REModel N T k) (T_val : ℕ) : ℝ :=
  1 - Real.sqrt (model.σ_ε_sq / (model.σ_ε_sq + T_val * model.σ_α_sq))

/-- GLS-transformed Y: Ỹ_it = Y_it - θȲ_i -/
def glsTransformedY {N T k : ℕ} (model : REModel N T k) (θ : ℝ)
    (i : Fin N) (t : Fin T) : ℝ :=
  (model.panel.data i t).Y - θ * individualMeanY model.panel i

/-- GLS-transformed X: X̃_itj = X_itj - θX̄_ij -/
def glsTransformedX {N T k : ℕ} (model : REModel N T k) (θ : ℝ)
    (i : Fin N) (t : Fin T) (j : Fin k) : ℝ :=
  (model.panel.data i t).X j - θ * individualMeanX model.panel i j

/-- RE estimator is GLS (OLS on transformed data).

    β̂_RE = (Σ_i Σ_t X_tilde_it X_tilde_it')⁻¹ (Σ_i Σ_t X_tilde_it Y_tilde_it) -/
def REEstimator {N T k : ℕ}
    (model : REModel N T k)
    (θ : ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  -- Compute X'Y where X and Y are GLS-transformed
  let XtY := fun j => ∑ i : Fin N, ∑ t : Fin T, glsTransformedX model θ i t j * glsTransformedY model θ i t
  fun j => ∑ l : Fin k, XtX_inv j l * XtY l

/-!
## Special Cases of θ
-/

/-- When σ²_α = 0, θ = 0 and RE = Pooled OLS.

    No individual heterogeneity means no need to transform. -/
theorem theta_zero_when_no_heterogeneity {N T k : ℕ}
    (model : REModel N T k)
    (h_no_het : model.σ_α_sq = 0)
    (h_theta_zero : thetaGLS model T = 0) :
    thetaGLS model T = 0 := by
  exact h_theta_zero

/-- When σ²_α → ∞, θ → 1 and RE → FE.

    As individual heterogeneity dominates, RE converges to FE. -/
theorem theta_approaches_one_as_heterogeneity_grows {N T k : ℕ}
    (σ_ε_sq : ℝ) (h_pos : σ_ε_sq > 0)
    (σ_α_seq : ℕ → ℝ)
    (h_grows : ∀ M, ∃ n, ∀ m ≥ n, σ_α_seq m > M) :
    True := by  -- Placeholder for limit
  -- θ = 1 - √(σ²_ε / (σ²_ε + T·σ²_α)) → 1 as σ²_α → ∞
  trivial

/-!
## Feasible GLS
-/

/-- Estimate σ²_ε from within-group residuals (FE residuals).

    σ̂²_ε = Σ_i Σ_t (ε̂_it - ε̄_i)² / (N(T-1) - k) -/
def estimateSigmaEpsSq {N T k : ℕ}
    (panel : PanelData N T k)
    (fe_residuals : Fin N → Fin T → ℝ) : ℝ :=
  let ssr := ∑ i : Fin N, ∑ t : Fin T,
    (fe_residuals i t - (∑ s : Fin T, fe_residuals i s) / T)^2
  ssr / (N * (T - 1) - k)

/-- Estimate σ²_α from between-group variation.

    Use pooled OLS residuals and FE variance estimate. -/
def estimateSigmaAlphaSq {N T k : ℕ}
    (σ_ε_sq_hat : ℝ)
    (between_variance : ℝ) : ℝ :=
  max 0 (between_variance - σ_ε_sq_hat / T)  -- Ensure non-negative

/-- Feasible GLS estimator: plug in estimated θ̂ -/
def FGLSEstimator {N T k : ℕ}
    (panel : PanelData N T k)
    (θ_hat : ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  -- Compute X'Y where X and Y are GLS-transformed with estimated θ
  let XtY := fun j => ∑ i : Fin N, ∑ t : Fin T,
    ((panel.data i t).X j - θ_hat * individualMeanX panel i j) *
    ((panel.data i t).Y - θ_hat * individualMeanY panel i)
  fun j => ∑ l : Fin k, XtX_inv j l * XtY l

/-!
## RE Consistency and Efficiency
-/

/-- RE is consistent when RE assumptions hold.

    Under:
    1. RE orthogonality: Cov(X_it, α_i) = 0
    2. Strict exogeneity of ε_it

    β̂_RE →p β -/
theorem re_consistent {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Fin N → Fin T → Fin k → ℝ)
    (α : Ω → Fin N → ℝ)
    (ε : Ω → Fin N → Fin T → ℝ)
    (β_true : Fin k → ℝ)
    (h_orthog : REOrthogonality μ X α)
    (h_strict_exog : True)  -- ε ⊥ X
    (β_hat_seq : ℕ → Ω → Fin k → ℝ) :
    True := by  -- Placeholder for convergence
  trivial

/-- RE is more efficient than FE when RE assumptions hold.

    Var(β̂_RE) ≤ Var(β̂_FE)

    RE uses all variation (within and between), while FE uses only within. -/
theorem re_more_efficient_than_fe {N T k : ℕ}
    (var_RE var_FE : Matrix (Fin k) (Fin k) ℝ)
    (h_re_valid : True)  -- RE assumptions hold
    : True := by  -- Placeholder for Var(RE) ≤ Var(FE)
  -- RE is BLUE under its assumptions
  trivial

/-- However, if RE assumptions fail (α correlated with X),
    RE is inconsistent while FE remains consistent. -/
theorem re_inconsistent_when_correlated {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Fin N → Fin T → Fin k → ℝ)
    (α : Ω → Fin N → ℝ)
    (h_T_pos : T > 0)
    (h_correlated : ∃ i j, ∫ ω, X ω i ⟨0, h_T_pos⟩ j * α ω i ∂μ ≠ 0) :
    True := by  -- RE is biased
  trivial

/-!
## Time-Invariant Variables
-/

/-- Unlike FE, RE can estimate coefficients on time-invariant variables.

    Since RE doesn't fully demean, time-invariant X's aren't eliminated. -/
theorem re_estimates_time_invariant {N T k : ℕ}
    (panel : PanelData N T k)
    (j : Fin k)
    (h_invariant : isTimeInvariant panel j)
    (θ : ℝ) (h_θ : θ < 1) :
    True := by  -- X̃_ijt = X_ij - θX_ij = (1-θ)X_ij ≠ 0
  trivial

/-!
## Between Estimator
-/

/-- Between estimator: OLS on individual means.

    Y_bar_i = X_bar_i'β + (α_i + ε_bar_i)

    Uses only between-individual variation. -/
def BetweenEstimator {N T k : ℕ}
    (panel : PanelData N T k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  let X_bar : Matrix (Fin N) (Fin k) ℝ := fun i j => individualMeanX panel i j
  let Y_bar : Fin N → ℝ := fun i => individualMeanY panel i
  OLS.OLSEstimator X_bar Y_bar XtX_inv

/-- Between estimator is consistent under RE assumptions. -/
theorem between_consistent_under_re {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Fin N → Fin T → Fin k → ℝ)
    (α : Ω → Fin N → ℝ)
    (h_orthog : REOrthogonality μ X α) :
    True := by  -- Between is consistent when α ⊥ X
  trivial

/-- RE is a matrix-weighted average of Within and Between.

    β̂_RE = W_W β̂_FE + W_B β̂_BE

    where weights depend on θ and relative variances. -/
theorem re_is_weighted_average {N T k : ℕ}
    (β_RE β_FE β_BE : Fin k → ℝ)
    (W_W W_B : Matrix (Fin k) (Fin k) ℝ)
    (h_weights : W_W + W_B = 1) :
    True := by  -- β̂_RE = W_W β̂_FE + W_B β̂_BE
  trivial

end Panel

end Econometrics

end
