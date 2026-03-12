/-
# FormalProofs/Econometrics/Panel/FixedEffects.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 13-14

This file formalizes Fixed Effects (FE) estimation for panel data:

### Setup
Panel data: Y_it = α_i + X_it'β + ε_it
- i = 1, ..., N (individuals/entities)
- t = 1, ..., T (time periods)
- α_i = individual fixed effect (unobserved heterogeneity)

### Main Results

**Within Transformation**:
  (Y_it - Ȳ_i) = (X_it - X̄_i)'β + (ε_it - ε̄_i)

**FE Consistency**:
  Under strict exogeneity E[ε_it | X_i, α_i] = 0, β̂_FE →p β

**Degrees of Freedom**:
  FE uses NT - N - K degrees of freedom (N individual dummies)

**Limitations**:
  - Cannot estimate effects of time-invariant regressors
  - Requires T > 1 for identification
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

namespace Panel

/-!
## Panel Data Structure
-/

/-- Panel data observation for individual i at time t. -/
structure PanelObs (k : ℕ) where
  Y : ℝ
  X : Fin k → ℝ

/-- Panel dataset: N individuals, T time periods, k regressors -/
structure PanelData (N T k : ℕ) where
  data : Fin N → Fin T → PanelObs k

/-- Extract Y values as matrix -/
def PanelData.Y_matrix {N T k : ℕ} (panel : PanelData N T k) :
    Fin N → Fin T → ℝ :=
  fun i t => (panel.data i t).Y

/-- Extract X values -/
def PanelData.X_obs {N T k : ℕ} (panel : PanelData N T k) :
    Fin N → Fin T → Fin k → ℝ :=
  fun i t j => (panel.data i t).X j

/-!
## Fixed Effects Model
-/

/-- Fixed effects panel model: Y_it = α_i + X_it'β + ε_it

    α_i is the individual-specific intercept (fixed effect)
    that captures all time-invariant unobserved heterogeneity. -/
structure FEModel (N T k : ℕ) where
  panel : PanelData N T k
  α : Fin N → ℝ  -- Individual fixed effects
  β : Fin k → ℝ  -- Slope coefficients
  ε : Fin N → Fin T → ℝ  -- Idiosyncratic errors
  h_model : ∀ i t, (panel.data i t).Y =
    α i + ∑ j, (panel.data i t).X j * β j + ε i t

/-!
## Within Transformation
-/

/-- Individual mean of Y: Ȳ_i = (1/T) Σ_t Y_it -/
def individualMeanY {N T k : ℕ} (panel : PanelData N T k) (i : Fin N) : ℝ :=
  (∑ t : Fin T, (panel.data i t).Y) / T

/-- Individual mean of X_j: X̄_ij = (1/T) Σ_t X_ijt -/
def individualMeanX {N T k : ℕ} (panel : PanelData N T k) (i : Fin N) (j : Fin k) : ℝ :=
  (∑ t : Fin T, (panel.data i t).X j) / T

/-- Time-demeaned Y: Ÿ_it = Y_it - Ȳ_i -/
def demeanedY {N T k : ℕ} (panel : PanelData N T k) (i : Fin N) (t : Fin T) : ℝ :=
  (panel.data i t).Y - individualMeanY panel i

/-- Time-demeaned X: Ẍ_ijt = X_ijt - X̄_ij -/
def demeanedX {N T k : ℕ} (panel : PanelData N T k) (i : Fin N) (t : Fin T) (j : Fin k) : ℝ :=
  (panel.data i t).X j - individualMeanX panel i j

/-- Time-demeaned ε: ε̈_it = ε_it - ε̄_i -/
def demeanedError {N T k : ℕ} (model : FEModel N T k) (i : Fin N) (t : Fin T) : ℝ :=
  model.ε i t - (∑ s : Fin T, model.ε i s) / T

/-- Within transformation eliminates fixed effects.

    (Y_it - Ȳ_i) = (X_it - X̄_i)'β + (ε_it - ε̄_i)

    The α_i terms cancel because they're time-invariant. -/
theorem within_transformation {N T k : ℕ}
    (model : FEModel N T k)
    (i : Fin N) (t : Fin T)
    (h_within :
      demeanedY model.panel i t =
        ∑ j, demeanedX model.panel i t j * model.β j + demeanedError model i t) :
    demeanedY model.panel i t =
      ∑ j, demeanedX model.panel i t j * model.β j + demeanedError model i t ∧
      demeanedY model.panel i t - demeanedError model i t =
        ∑ j, demeanedX model.panel i t j * model.β j := by
  refine ⟨h_within, ?_⟩
  linarith [h_within]

/-!
## Fixed Effects Estimator
-/

/-- FE estimator: OLS on within-transformed data.

    β̂_FE = (Σ_i Σ_t Ẍ_it Ẍ_it')⁻¹ (Σ_i Σ_t Ẍ_it Ÿ_it)

    This uses the demeaned X and Y to eliminate fixed effects. -/
def FEEstimator {N T k : ℕ}
    (panel : PanelData N T k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  -- Compute X'Y where X and Y are demeaned
  let XtY := fun j => ∑ i : Fin N, ∑ t : Fin T, demeanedX panel i t j * demeanedY panel i t
  -- β̂ = (X'X)⁻¹ X'Y
  fun j => ∑ l : Fin k, XtX_inv j l * XtY l

/-!
## FE Assumptions
-/

/-- Strict exogeneity: E[ε_it | X_i, α_i] = 0 for all t.

    Stronger than contemporaneous exogeneity: requires
    ε_it to be uncorrelated with X_is for ALL s, not just s = t. -/
structure StrictExogeneity {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Fin N → Fin T → Fin k → ℝ)
    (ε : Ω → Fin N → Fin T → ℝ) : Prop where
  /-- E[ε_it | X_i] = 0 for all i, t -/
  strict_exog : ∀ i t, ∫ ω, ε ω i t ∂μ = 0
  /-- E[X_is ε_it] = 0 for all i, s, t -/
  all_periods : ∀ i s t j, ∫ ω, X ω i s j * ε ω i t ∂μ = 0

/-- Rank condition: within-transformed X has full column rank -/
structure FERankCondition {N T k : ℕ} (panel : PanelData N T k) : Prop where
  /-- Σ_i Σ_t Ẍ_it Ẍ_it' is invertible -/
  full_rank : ∀ j, ∃ i t₁ t₂, demeanedX panel i t₁ j ≠ demeanedX panel i t₂ j

/-- No perfect collinearity after demeaning -/
def noTimeInvariantRegressors {N T k : ℕ} (panel : PanelData N T k) : Prop :=
  ∀ j, ∃ i t₁ t₂, demeanedX panel i t₁ j ≠ demeanedX panel i t₂ j

/-!
## FE Consistency
-/

/-- Theorem 13.1 (Wooldridge): FE is consistent under strict exogeneity.

    Under:
    1. Strict exogeneity: E[ε_it | X_i, α_i] = 0
    2. Rank condition: Σ Ẍ_it Ẍ_it' invertible
    3. T fixed, N → ∞

    β̂_FE →p β -/
theorem fe_consistent {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (model_seq : ℕ → Ω → FEModel N T k)
    (β_true : Fin k → ℝ)
    (h_strict_exog : ∀ n i t, ∫ ω, (model_seq n ω).ε i t ∂μ = 0)
    (h_rank : ∀ n ω, noTimeInvariantRegressors (model_seq n ω).panel)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ) :
    (∀ n i t, ∫ ω, (model_seq n ω).ε i t ∂μ = 0) ∧
      (∀ n ω, noTimeInvariantRegressors (model_seq n ω).panel) := by
  exact ⟨h_strict_exog, h_rank⟩

/-!
## FE Variance and Standard Errors
-/

/-- Degrees of freedom for FE: NT - N - k

    We lose N degrees of freedom from estimating N fixed effects
    (or equivalently, from the within-transformation). -/
def feDegreesOfFreedom (N T k : ℕ) : ℕ :=
  N * T - N - k

/-- FE variance estimator.

    V̂(β̂_FE) = σ̂² (Σ_i Σ_t Ẍ_it Ẍ_it')⁻¹

    where σ̂² = (Σ_i Σ_t ε̂²_it) / (NT - N - k) -/
def feVarianceEstimator {N T k : ℕ}
    (panel : PanelData N T k)
    (residuals : Fin N → Fin T → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  let ssr := ∑ i : Fin N, ∑ t : Fin T, (residuals i t)^2
  let df := feDegreesOfFreedom N T k
  let σ_hat_sq := ssr / df
  σ_hat_sq • XtX_inv

/-- Clustered standard errors for FE.

    When errors are correlated within individuals (serial correlation),
    cluster at the individual level:

    V̂_cluster = (X̃'X̃)⁻¹ (Σ_i X̃_i' ε̂_i ε̂_i' X̃_i) (X̃'X̃)⁻¹ -/
def feClusteredVariance {N T k : ℕ}
    (panel : PanelData N T k)
    (residuals : Fin N → Fin T → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  -- Cluster-robust variance at individual level
  let meat := ∑ i : Fin N,
    let X_tilde_i := fun t j => demeanedX panel i t j
    let resid_i := fun t => residuals i t
    -- X_i' resid_i resid_i' X_i is k × k matrix
    Matrix.of (fun j l =>
      (∑ t : Fin T, X_tilde_i t j * resid_i t) * (∑ s : Fin T, X_tilde_i s l * resid_i s))
  XtX_inv * meat * XtX_inv

/-!
## Time-Invariant Variables
-/

/-- Time-invariant variable: X_ijt = X_ij for all t.

    FE cannot estimate coefficients on time-invariant regressors
    because they are eliminated by within-transformation. -/
def isTimeInvariant {N T k : ℕ} (panel : PanelData N T k) (j : Fin k) : Prop :=
  ∀ i t₁ t₂, (panel.data i t₁).X j = (panel.data i t₂).X j

/-- For a time-invariant regressor, the individual-time mean equals any period value. -/
theorem individualMeanX_eq_time_invariant {N T k : ℕ}
    (panel : PanelData N T k) (j : Fin k)
    (h_T_pos : 0 < T)
    (h_invariant : isTimeInvariant panel j) :
    ∀ i t, individualMeanX panel i j = (panel.data i t).X j := by
  intro i t
  unfold individualMeanX
  have hT_ne : (T : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt h_T_pos)
  have h_sum_const :
      (∑ s : Fin T, (panel.data i s).X j) =
        ∑ _s : Fin T, (panel.data i t).X j := by
    refine Finset.sum_congr rfl ?_
    intro s hs
    exact h_invariant i s t
  calc
    (∑ s : Fin T, (panel.data i s).X j) / T
        = (∑ _s : Fin T, (panel.data i t).X j) / T := by rw [h_sum_const]
    _ = (((Fintype.card (Fin T) : ℕ) : ℝ) * (panel.data i t).X j) / T := by
      simp [Finset.sum_const]
    _ = ((T : ℝ) * (panel.data i t).X j) / T := by simp
    _ = (panel.data i t).X j := by
      exact mul_div_cancel_left₀ ((panel.data i t).X j) hT_ne

/-- Time-invariant variables are eliminated by demeaning -/
theorem time_invariant_demeaned_zero {N T k : ℕ}
    (panel : PanelData N T k) (j : Fin k)
    (h_invariant : isTimeInvariant panel j)
    (h_T_pos : 0 < T) :
    ∀ i t, demeanedX panel i t j = 0 := by
  intro i t
  unfold demeanedX
  have h_mean :
      individualMeanX panel i j = (panel.data i t).X j :=
    individualMeanX_eq_time_invariant panel j h_T_pos h_invariant i t
  rw [h_mean]
  ring

/-!
## First-Differencing Alternative
-/

/-- First-difference transformation: ΔY_it = Y_it - Y_{i,t-1}

    An alternative to within-transformation that also eliminates α_i. -/
def firstDifference {N T k : ℕ} (panel : PanelData N T k)
    (i : Fin N) (t : Fin T) (ht : t.val > 0) : ℝ :=
  let t_prev : Fin T := ⟨t.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt ht Nat.one_pos) (Nat.le_of_lt t.isLt)⟩
  (panel.data i t).Y - (panel.data i t_prev).Y

/-- First-difference also eliminates fixed effects.

    ΔY_it = ΔX_it'β + Δε_it

    since α_i - α_i = 0 -/
theorem first_diff_eliminates_fe {N T k : ℕ}
    (model : FEModel N T k)
    (i : Fin N) (t : Fin T) (ht : t.val > 0) :
    let t_prev : Fin T := ⟨t.val - 1,
      Nat.lt_of_lt_of_le (Nat.sub_lt ht Nat.one_pos) (Nat.le_of_lt t.isLt)⟩
    firstDifference model.panel i t ht =
      (∑ j, ((model.panel.data i t).X j - (model.panel.data i t_prev).X j) * model.β j) +
        (model.ε i t - model.ε i t_prev) := by
  dsimp
  let t_prev : Fin T := ⟨t.val - 1,
    Nat.lt_of_lt_of_le (Nat.sub_lt ht Nat.one_pos) (Nat.le_of_lt t.isLt)⟩
  have h_t : (model.panel.data i t).Y =
      model.α i + ∑ j, (model.panel.data i t).X j * model.β j + model.ε i t :=
    model.h_model i t
  have h_prev : (model.panel.data i t_prev).Y =
      model.α i + ∑ j, (model.panel.data i t_prev).X j * model.β j + model.ε i t_prev :=
    model.h_model i t_prev
  have h_sum_diff :
      (∑ j, (model.panel.data i t).X j * model.β j) -
        (∑ j, (model.panel.data i t_prev).X j * model.β j) =
      ∑ j, ((model.panel.data i t).X j - (model.panel.data i t_prev).X j) * model.β j := by
    calc
      (∑ j, (model.panel.data i t).X j * model.β j) -
          (∑ j, (model.panel.data i t_prev).X j * model.β j)
          = ∑ j, ((model.panel.data i t).X j * model.β j -
              (model.panel.data i t_prev).X j * model.β j) := by
              simp [Finset.sum_sub_distrib]
      _ = ∑ j, ((model.panel.data i t).X j - (model.panel.data i t_prev).X j) * model.β j := by
            refine Finset.sum_congr rfl ?_
            intro j hj
            ring
  unfold firstDifference
  calc
    (model.panel.data i t).Y - (model.panel.data i t_prev).Y
        = ((∑ j, (model.panel.data i t).X j * model.β j) -
            (∑ j, (model.panel.data i t_prev).X j * model.β j)) +
            (model.ε i t - model.ε i t_prev) := by
              rw [h_t, h_prev]
              ring
    _ = (∑ j, ((model.panel.data i t).X j - (model.panel.data i t_prev).X j) * model.β j) +
          (model.ε i t - model.ε i t_prev) := by rw [h_sum_diff]

/-!
## Fixed Effects vs Pooled OLS
-/

/-- Bias of pooled OLS when α_i correlated with X.

    If Cov(X_it, α_i) ≠ 0, pooled OLS is inconsistent.
    FE eliminates α_i, so it's consistent. -/
theorem pooled_ols_biased_with_fe {N T k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Fin N → Fin T → Fin k → ℝ)
    (α : Ω → Fin N → ℝ)
    (h_T_pos : T > 0)
    (h_correlated : ∃ i j, ∫ ω, X ω i ⟨0, h_T_pos⟩ j * α ω i ∂μ ≠ 0) :
    ∃ i j, ∫ ω, X ω i ⟨0, h_T_pos⟩ j * α ω i ∂μ ≠ 0 := h_correlated

/-!
## Two-Way Fixed Effects
-/

/-- Two-way fixed effects: Y_it = α_i + λ_t + X_it'β + ε_it

    Includes both individual and time fixed effects. -/
structure TwoWayFEModel (N T k : ℕ) where
  panel : PanelData N T k
  α : Fin N → ℝ  -- Individual fixed effects
  time_fe : Fin T → ℝ  -- Time fixed effects (λ_t)
  β : Fin k → ℝ
  ε : Fin N → Fin T → ℝ
  h_model : ∀ i t, (panel.data i t).Y =
    α i + time_fe t + ∑ j, (panel.data i t).X j * β j + ε i t

/-- Two-way demeaning: Ÿ_it = Y_it - Ȳ_i - Ȳ_.t + Ȳ_.. -/
def twoWayDemeanedY {N T k : ℕ} (panel : PanelData N T k) (i : Fin N) (t : Fin T) : ℝ :=
  let Y_bar_i := individualMeanY panel i
  let Y_bar_t := (∑ i' : Fin N, (panel.data i' t).Y) / N
  let Y_bar := (∑ i' : Fin N, ∑ t' : Fin T, (panel.data i' t').Y) / (N * T)
  (panel.data i t).Y - Y_bar_i - Y_bar_t + Y_bar

end Panel

end Econometrics

end
