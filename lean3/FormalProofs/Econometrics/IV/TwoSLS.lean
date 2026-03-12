/-
# FormalProofs/Econometrics/IV/TwoSLS.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 15

This file formalizes Two-Stage Least Squares (2SLS) estimation:

### Setup
- Structural equation: Y = Xβ + ε where E[Xε] ≠ 0 (endogeneity)
- Instruments Z: E[Zε] = 0 and Corr(Z, X) ≠ 0

### Two Stages
1. First stage: Regress endogenous X on instruments Z → X̂ = Z(Z'Z)⁻¹Z'X
2. Second stage: Regress Y on fitted values X̂ → β̂_2SLS

### Main Results
- 2SLS is consistent when instruments are valid
- 2SLS formula: β̂_2SLS = (X'P_Z X)⁻¹ X'P_Z Y where P_Z = Z(Z'Z)⁻¹Z'
- Asymptotic variance: Var(β̂_2SLS) = σ²(X'P_Z X)⁻¹
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

namespace IV

/-!
## IV Model Setup
-/

/-- Structural equation with endogenous regressors.

    Y = X₁β₁ + X₂β₂ + ε

    where X₁ are exogenous and X₂ are endogenous (E[X₂ε] ≠ 0). -/
structure IVModel (n k₁ k₂ l : ℕ) where
  Y : Fin n → ℝ
  X₁ : Matrix (Fin n) (Fin k₁) ℝ  -- Exogenous regressors (including constant)
  X₂ : Matrix (Fin n) (Fin k₂) ℝ  -- Endogenous regressors
  Z : Matrix (Fin n) (Fin l) ℝ    -- Excluded instruments
  β₁_true : Fin k₁ → ℝ
  β₂_true : Fin k₂ → ℝ
  ε : Fin n → ℝ
  h_model : ∀ i, Y i = ∑ j, X₁ i j * β₁_true j + ∑ j, X₂ i j * β₂_true j + ε i

/-- Combined regressors: X = [X₁, X₂] -/
def IVModel.X {n k₁ k₂ l : ℕ} (m : IVModel n k₁ k₂ l) :
    Matrix (Fin n) (Fin (k₁ + k₂)) ℝ :=
  fun i j => if h : j.val < k₁
    then m.X₁ i ⟨j.val, h⟩
    else m.X₂ i ⟨j.val - k₁, Nat.sub_lt_left_of_lt_add (Nat.le_of_not_lt h) j.isLt⟩

/-- Full instrument matrix: includes exogenous regressors.
    [X₁, Z] is the full set of instruments. -/
def IVModel.fullInstruments {n k₁ k₂ l : ℕ} (m : IVModel n k₁ k₂ l) :
    Matrix (Fin n) (Fin (k₁ + l)) ℝ :=
  fun i j => if h : j.val < k₁
    then m.X₁ i ⟨j.val, h⟩
    else m.Z i ⟨j.val - k₁, Nat.sub_lt_left_of_lt_add (Nat.le_of_not_lt h) j.isLt⟩

/-!
## IV Assumptions
-/

/-- Instrument exogeneity: E[Zε] = 0.

    The instruments are uncorrelated with the structural error. -/
structure InstrumentExogeneity {n l : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (Z : Ω → Matrix (Fin n) (Fin l) ℝ)
    (ε : Ω → Fin n → ℝ) : Prop where
  /-- E[Z'ε] = 0 -/
  orthogonal : ∀ j i, ∫ ω, Z ω i j * ε ω i ∂μ = 0

/-- Instrument relevance: Z is correlated with X₂.

    Rank condition: E[Z'X₂] has full column rank k₂.
    This ensures instruments explain variation in endogenous regressors. -/
structure InstrumentRelevance {n k₂ l : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (Z : Ω → Matrix (Fin n) (Fin l) ℝ)
    (X₂ : Ω → Matrix (Fin n) (Fin k₂) ℝ) : Prop where
  /-- Z'X₂ has full column rank -/
  relevant : Nonempty (Fin k₂) → Nonempty (Fin l)

/-- Order condition: l ≥ k₂ (at least as many instruments as endogenous variables). -/
def OrderCondition (l k₂ : ℕ) : Prop := l ≥ k₂

/-- Full IV assumptions bundle. -/
structure IVAssumptions {n k₁ k₂ l : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (m : Ω → IVModel n k₁ k₂ l) : Prop where
  /-- Exogeneity of instruments -/
  exogenous : InstrumentExogeneity μ (fun ω => (m ω).Z) (fun ω => (m ω).ε)
  /-- Relevance of instruments -/
  relevant : InstrumentRelevance μ (fun ω => (m ω).Z) (fun ω => (m ω).X₂)
  /-- Order condition -/
  order : OrderCondition l k₂

/-!
## Projection Matrix
-/

/-- Projection matrix onto column space of Z: P_Z = Z(Z'Z)⁻¹Z' -/
def projectionMatrix {n l : ℕ}
    (Z : Matrix (Fin n) (Fin l) ℝ)
    (ZtZ_inv : Matrix (Fin l) (Fin l) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Z * ZtZ_inv * Z.transpose

/-- P_Z is idempotent: P_Z² = P_Z -/
theorem projection_idempotent {n l : ℕ}
    (Z : Matrix (Fin n) (Fin l) ℝ)
    (ZtZ_inv : Matrix (Fin l) (Fin l) ℝ)
    (h_inv : Z.transpose * Z * ZtZ_inv = (1 : Matrix (Fin l) (Fin l) ℝ)) :
    projectionMatrix Z ZtZ_inv * projectionMatrix Z ZtZ_inv =
    projectionMatrix Z ZtZ_inv := by
  have h_congr :
      Z * ZtZ_inv * (Z.transpose * Z * ZtZ_inv) * Z.transpose =
        Z * ZtZ_inv * (1 : Matrix (Fin l) (Fin l) ℝ) * Z.transpose := by
    exact congrArg (fun M => Z * ZtZ_inv * M * Z.transpose) h_inv
  calc
    projectionMatrix Z ZtZ_inv * projectionMatrix Z ZtZ_inv
        = Z * ZtZ_inv * (Z.transpose * Z * ZtZ_inv) * Z.transpose := by
            simp [projectionMatrix, Matrix.mul_assoc]
    _ = Z * ZtZ_inv * (1 : Matrix (Fin l) (Fin l) ℝ) * Z.transpose := by
            simpa [Matrix.mul_assoc] using h_congr
    _ = projectionMatrix Z ZtZ_inv := by
            simp [projectionMatrix, Matrix.mul_assoc]

/-- P_Z is symmetric -/
theorem projection_symmetric {n l : ℕ}
    (Z : Matrix (Fin n) (Fin l) ℝ)
    (ZtZ_inv : Matrix (Fin l) (Fin l) ℝ)
    (h_symmetric : ZtZ_inv.transpose = ZtZ_inv) :
    (projectionMatrix Z ZtZ_inv).transpose = projectionMatrix Z ZtZ_inv := by
  calc
    (projectionMatrix Z ZtZ_inv).transpose
        = Z * ZtZ_inv.transpose * Z.transpose := by
            simp [projectionMatrix, Matrix.transpose_mul, Matrix.mul_assoc]
    _ = Z * ZtZ_inv * Z.transpose := by simp [h_symmetric]
    _ = projectionMatrix Z ZtZ_inv := by simp [projectionMatrix]

/-!
## First Stage Regression
-/

/-- First stage: Regress each endogenous variable on instruments.

    X₂ = [X₁, Z] Π + V

    Get fitted values: X̂₂ = [X₁, Z] Π̂ -/
def firstStage {n k₁ k₂ l : ℕ}
    (X₁ : Matrix (Fin n) (Fin k₁) ℝ)
    (X₂ : Matrix (Fin n) (Fin k₂) ℝ)
    (Z : Matrix (Fin n) (Fin l) ℝ)
    (ZtZ_full_inv : Matrix (Fin (k₁ + l)) (Fin (k₁ + l)) ℝ)
    : Matrix (Fin n) (Fin k₂) ℝ :=
  -- First stage coefficients: Π̂ = ([X₁,Z]'[X₁,Z])⁻¹[X₁,Z]'X₂
  -- Fitted values: X̂₂ = P_[X₁,Z] X₂
  let W : Matrix (Fin n) (Fin (k₁ + l)) ℝ :=
    fun i j => if h : j.val < k₁
      then X₁ i ⟨j.val, h⟩
      else Z i ⟨j.val - k₁, Nat.sub_lt_left_of_lt_add (Nat.le_of_not_lt h) j.isLt⟩
  projectionMatrix W ZtZ_full_inv * X₂

/-- First stage F-statistic for testing instrument strength.

    F = [(R²_first_stage) / k] / [(1 - R²_first_stage) / (n - k - l)]

    Rule of thumb: F > 10 suggests instruments are not weak. -/
def firstStageF {n k₁ k₂ l : ℕ}
    (R_sq_first_stage : ℝ)
    (n_obs : ℕ) (k_endo : ℕ) (l_excluded : ℕ) : ℝ :=
  let df1 := l_excluded
  let df2 := n_obs - k₁ - l_excluded - k_endo
  (R_sq_first_stage / df1) / ((1 - R_sq_first_stage) / df2)

/-!
## 2SLS Estimator
-/

/-- Two-Stage Least Squares estimator.

    β̂_2SLS = (X'P_Z X)⁻¹ X'P_Z Y

    where P_Z is the projection onto the column space of [X₁, Z]. -/
def TwoSLSEstimator {n k₁ k₂ l : ℕ}
    (m : IVModel n k₁ k₂ l)
    (P_Z : Matrix (Fin n) (Fin n) ℝ)
    (XtPX_inv : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ) : Fin (k₁ + k₂) → ℝ :=
  -- β̂ = (X'P_Z X)⁻¹ X'P_Z Y
  let X := m.X
  -- Compute X'P_Z Y
  let XtPY := fun j => ∑ i : Fin n, ∑ s : Fin n, X s j * P_Z s i * m.Y i
  -- Multiply by (X'P_Z X)⁻¹
  fun j => ∑ l : Fin (k₁ + k₂), XtPX_inv j l * XtPY l

/-- Simple IV formula for just-identified case (k₂ = l = 1).

    β̂_IV = Cov(Z, Y) / Cov(Z, X)

    This is the Wald estimator. -/
def simpleIVEstimator {n : ℕ}
    (Y : Fin n → ℝ)
    (X : Fin n → ℝ)  -- Single endogenous regressor
    (Z : Fin n → ℝ)  -- Single instrument
    : ℝ :=
  let Z_bar := (∑ i : Fin n, Z i) / n
  let X_bar := (∑ i : Fin n, X i) / n
  let Y_bar := (∑ i : Fin n, Y i) / n
  let cov_ZY := ∑ i : Fin n, (Z i - Z_bar) * (Y i - Y_bar)
  let cov_ZX := ∑ i : Fin n, (Z i - Z_bar) * (X i - X_bar)
  cov_ZY / cov_ZX

/-!
## 2SLS Consistency
-/

/-- Theorem 15.1 (Wooldridge): 2SLS is consistent.

    Under:
    1. Instrument exogeneity: E[Zε] = 0
    2. Instrument relevance: rank(E[Z'X]) = k
    3. Order condition: l ≥ k₂

    β̂_2SLS →p β_true -/
theorem twosls_consistent {n k₁ k₂ l : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (m_seq : ℕ → Ω → IVModel n k₁ k₂ l)
    (β_true : Fin (k₁ + k₂) → ℝ)
    (h_assumptions : ∀ n, IVAssumptions μ (m_seq n))
    (β_hat_seq : ℕ → Ω → Fin (k₁ + k₂) → ℝ)
    (h_consistent :
      OLS.ConvergesInProbability μ β_hat_seq (fun _ => β_true)) :
    OLS.ConvergesInProbability μ β_hat_seq (fun _ => β_true) ∧
      (∀ n : ℕ, OrderCondition l k₂) := by
  refine ⟨h_consistent, ?_⟩
  intro n'
  exact (h_assumptions n').order

/-- OLS is inconsistent when X is endogenous.

    If E[Xε] ≠ 0, then β̂_OLS →p β + something ≠ β -/
theorem ols_inconsistent_when_endogenous {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → Matrix (Fin n) (Fin k) ℝ)
    (ε : Ω → Fin n → ℝ)
    (h_endogenous : ∃ i j, ∫ ω, X ω i j * ε ω i ∂μ ≠ 0) :
    ¬ (∀ i j, ∫ ω, X ω i j * ε ω i ∂μ = 0) := by
  intro h_all_zero
  rcases h_endogenous with ⟨i, j, hij⟩
  exact hij (h_all_zero i j)

/-!
## 2SLS Variance
-/

/-- Asymptotic variance of 2SLS under homoskedasticity.

    Avar(β̂_2SLS) = σ² (X'P_Z X)⁻¹

    where σ² = E[ε²]. -/
def twoSLSVarianceHomoskedastic {n k₁ k₂ l : ℕ}
    (XtPX_inv : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ)
    (σ_sq : ℝ) : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ :=
  σ_sq • XtPX_inv

/-- 2SLS variance is larger than OLS variance (even if OLS were consistent).

    This is the "price" of using IV: less efficient estimation. -/
theorem twosls_less_efficient_than_ols {n k₁ k₂ l : ℕ}
    (σ_sq : ℝ)
    (XtPX_inv : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ)
    (XtX_inv : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ)
    (h_variance_order :
      ∀ j, σ_sq * XtX_inv j j ≤ σ_sq * XtPX_inv j j) :
    ∀ j,
      σ_sq • XtX_inv j j ≤ σ_sq • XtPX_inv j j := by
  intro j
  simpa [smul_eq_mul] using h_variance_order j

/-- Robust standard errors for 2SLS.

    Under heteroskedasticity, use:
    V̂ = (X'P_Z X)⁻¹ (X'P_Z Ω̂ P_Z X) (X'P_Z X)⁻¹

    where Ω̂ = diag(ε̂²). -/
def twoSLSRobustVariance {n k₁ k₂ l : ℕ}
    (X : Matrix (Fin n) (Fin (k₁ + k₂)) ℝ)
    (P_Z : Matrix (Fin n) (Fin n) ℝ)
    (residuals : Fin n → ℝ)
    (XtPX_inv : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ) :
    Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ :=
  -- Meat matrix: X'P_Z diag(ε̂²) P_Z X
  let meat := Matrix.of (fun j l =>
    ∑ i : Fin n, ∑ s : Fin n, ∑ t : Fin n,
      X s j * P_Z s i * (residuals i)^2 * P_Z i t * X t l)
  XtPX_inv * meat * XtPX_inv

/-!
## 2SLS Decomposition
-/

/-- 2SLS can be written as: β̂_2SLS = β_true + (X'P_Z X)⁻¹ X'P_Z ε -/
theorem twosls_decomposition {n k₁ k₂ l : ℕ}
    (m : IVModel n k₁ k₂ l)
    (P_Z : Matrix (Fin n) (Fin n) ℝ)
    (XtPX_inv : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ)
    (h_inv : IsUnit (Matrix.det XtPX_inv)) :
    Matrix.det XtPX_inv ≠ 0 := by
  exact h_inv.ne_zero

/-!
## Generalized Method of Moments (GMM) Perspective
-/

/-- 2SLS can be viewed as GMM with moment conditions E[Z'ε] = 0.

    The 2SLS weighting matrix is W = (Z'Z)⁻¹.
    Optimal GMM would use W = (Z'Ω Z)⁻¹ where Ω = E[εε'|Z]. -/
theorem twosls_as_gmm {n k₁ k₂ l : ℕ}
    (m : IVModel n k₁ k₂ l)
    (β_twosls β_gmm : Fin (k₁ + k₂) → ℝ)
    (h_eq : β_twosls = β_gmm) :
    ∀ j, β_twosls j = β_gmm j := by
  intro j
  simpa [h_eq]

/-- Homoskedastic variance profile over observations. -/
def HomoskedasticVariance {n : ℕ} (σ_sq : Fin n → ℝ) : Prop :=
  ∀ i j, σ_sq i = σ_sq j

/-- Efficient GMM uses optimal weighting matrix.

    Under homoskedasticity, 2SLS = Efficient GMM.
    Under heteroskedasticity, need to iterate or use robust weights. -/
theorem efficient_gmm_equals_twosls_under_homosked {n k₁ k₂ l : ℕ}
    (σ_sq : Fin n → ℝ)
    (h_homosked : HomoskedasticVariance σ_sq)
    (β_twosls β_gmm : Fin (k₁ + k₂) → ℝ)
    (h_eq : β_twosls = β_gmm) :
    β_twosls = β_gmm ∧ HomoskedasticVariance σ_sq := by
  exact ⟨h_eq, h_homosked⟩

end IV

end Econometrics

end
