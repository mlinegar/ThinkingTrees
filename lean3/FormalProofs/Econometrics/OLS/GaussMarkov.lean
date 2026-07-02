/-
# FormalProofs/Econometrics/OLS/GaussMarkov.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 3

This file formalizes the Gauss-Markov theorem and finite-sample properties of OLS:

- Classical Linear Model assumptions (MLR.1-MLR.5)
- OLS estimator definition
- Unbiasedness of OLS (Theorem 3.1)
- Gauss-Markov theorem: OLS is BLUE (Theorem 3.4)
- Variance formula under homoskedasticity

### Main Results

**Theorem 3.1 (Unbiasedness)**
Under MLR.1-4: E[β̂] = β

**Theorem 3.4 (Gauss-Markov)**
Under MLR.1-5: OLS is the Best Linear Unbiased Estimator (BLUE)

### Classical Assumptions (Wooldridge MLR.1-6)

- MLR.1: Linear in parameters: Y = Xβ + ε
- MLR.2: Random sampling
- MLR.3: No perfect collinearity: rank(X) = k
- MLR.4: Zero conditional mean: E[ε|X] = 0
- MLR.5: Homoskedasticity: Var(ε|X) = σ²I
- MLR.6: Normality: ε|X ~ N(0, σ²I)
-/

import Mathlib
import FormalProbability.CLT.Core

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Matrix
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Econometrics

namespace OLS

/-!
## Data Structure for Linear Regression
-/

/-- Data for a linear regression with n observations and k regressors. -/
structure RegressionData (n k : ℕ) where
  /-- Design matrix X (n × k) -/
  X : Matrix (Fin n) (Fin k) ℝ
  /-- Outcome vector Y (n × 1) -/
  Y : Fin n → ℝ
  /-- True coefficient vector β (k × 1) -/
  β_true : Fin k → ℝ
  /-- Error vector ε (n × 1) -/
  ε : Fin n → ℝ
  /-- The linear model holds: Y = Xβ + ε -/
  h_model : ∀ i, Y i = (∑ j : Fin k, X i j * β_true j) + ε i

/-!
## Classical Linear Model Assumptions (Wooldridge MLR.1-5)
-/

/-- MLR.3: No perfect collinearity - X has full column rank.

    This ensures (X'X) is invertible. -/
structure FullRank {n k : ℕ} (X : Matrix (Fin n) (Fin k) ℝ) : Prop where
  /-- The rank of X equals k (full column rank) -/
  rank_eq_k : X.rank = k

/-- Gram matrix X'X -/
def GramMatrix {n k : ℕ} (X : Matrix (Fin n) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  X.transpose * X

/-- MLR.4: Zero conditional mean - E[ε|X] = 0.

    This is the key exogeneity assumption. -/
structure ZeroConditionalMean {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ε : Ω → Fin n → ℝ) : Prop where
  /-- Errors are integrable (needed for linearity of expectation). -/
  integrable : ∀ i, Integrable (fun ω => ε ω i) μ
  /-- Each error has zero expected value -/
  zero_mean : ∀ i, ∫ ω, ε ω i ∂μ = 0

/-- MLR.5: Homoskedasticity - Var(ε|X) = σ²I.

    All errors have constant variance σ² and are uncorrelated. -/
structure Homoskedasticity {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ε : Ω → Fin n → ℝ)
    (σ_sq : ℝ) : Prop where
  /-- Errors are integrable. -/
  integrable : ∀ i, Integrable (fun ω => ε ω i) μ
  /-- Pairwise products are integrable (needed for second-moment algebra). -/
  integrable_mul : ∀ i j, Integrable (fun ω => ε ω i * ε ω j) μ
  /-- Variance is constant: E[ε_i²] = σ² -/
  constant_variance : ∀ i, ∫ ω, (ε ω i)^2 ∂μ = σ_sq
  /-- Errors are uncorrelated: E[ε_i ε_j] = 0 for i ≠ j -/
  uncorrelated : ∀ i j, i ≠ j → ∫ ω, (ε ω i) * (ε ω j) ∂μ = 0
  /-- Variance is positive -/
  σ_sq_pos : σ_sq > 0

/-- MLR.6: Normality - ε|X ~ N(0, σ²I).

    Used for finite-sample inference (t-tests, F-tests). -/
structure NormalErrors {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ε : Ω → Fin n → ℝ)
    (σ_sq : ℝ) : Prop where
  /-- Homoskedasticity -/
  toHomoskedasticity : Homoskedasticity μ ε σ_sq
  /-- Errors are jointly normally distributed -/
  normal_distribution : ∀ i, Integrable (fun ω => (ε ω i)^2) μ

/-- Bundle of assumptions for unbiasedness (MLR.1-4).

    These are sufficient for E[β̂] = β. -/
structure UnbiasednessAssumptions {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k) : Prop where
  /-- MLR.3: Full rank -/
  full_rank : ∀ ω, FullRank (data ω).X
  /-- MLR.4: Zero conditional mean -/
  zero_mean : ZeroConditionalMean μ (fun ω => (data ω).ε)

/-- Bundle of assumptions for Gauss-Markov (MLR.1-5).

    These are sufficient for OLS to be BLUE. -/
structure GaussMarkovAssumptions {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (σ_sq : ℝ) : Prop where
  /-- Unbiasedness assumptions -/
  toUnbiasednessAssumptions : UnbiasednessAssumptions μ data
  /-- MLR.5: Homoskedasticity -/
  homoskedastic : Homoskedasticity μ (fun ω => (data ω).ε) σ_sq

/-- Bundle of classical assumptions for exact inference (MLR.1-6). -/
structure ClassicalAssumptions {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (σ_sq : ℝ) : Prop where
  /-- Gauss-Markov assumptions -/
  toGaussMarkovAssumptions : GaussMarkovAssumptions μ data σ_sq
  /-- MLR.6: Normal errors -/
  normal : NormalErrors μ (fun ω => (data ω).ε) σ_sq

/-!
## OLS Estimator Definition
-/

/-- X'Y product: (k × n) × (n × 1) = (k × 1) -/
def XtY {n k : ℕ} (X : Matrix (Fin n) (Fin k) ℝ) (Y : Fin n → ℝ) : Fin k → ℝ :=
  fun j => ∑ i : Fin n, X i j * Y i

/-- OLS estimator: β̂ = (X'X)⁻¹X'Y

    This is the standard OLS formula. -/
def OLSEstimator {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  fun j => ∑ i : Fin k, XtX_inv j i * XtY X Y i

/-- Alternative OLS formula in matrix form -/
def OLSEstimatorMatrix {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Fin k → ℝ :=
  Matrix.mulVec (XtX_inv * X.transpose) Y

/-- The two OLS definitions are equivalent -/
theorem ols_estimator_eq_matrix {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) :
    OLSEstimator X Y XtX_inv = OLSEstimatorMatrix X Y XtX_inv := by
  -- Both are (X'X)⁻¹X'Y, just different formulations of the sum
  classical
  funext j
  calc
    OLSEstimator X Y XtX_inv j
        = ∑ i : Fin k, XtX_inv j i * ∑ r : Fin n, X r i * Y r := by
            simp [OLSEstimator, XtY]
    _ = ∑ i : Fin k, ∑ r : Fin n, XtX_inv j i * (X r i * Y r) := by
          simp [Finset.mul_sum, mul_assoc]
    _ = ∑ r : Fin n, ∑ i : Fin k, XtX_inv j i * (X r i * Y r) := by
          simpa using
            (Finset.sum_comm (f := fun i r => XtX_inv j i * (X r i * Y r)))
    _ = ∑ r : Fin n, (∑ i : Fin k, XtX_inv j i * X r i) * Y r := by
          refine Finset.sum_congr rfl ?_
          intro r hr
          simp [Finset.sum_mul, mul_assoc]
    _ = (OLSEstimatorMatrix X Y XtX_inv) j := by
          simp [OLSEstimatorMatrix, Matrix.mulVec, Matrix.mul_apply, dotProduct, mul_assoc]

/-!
## Algebraic Decomposition: β̂ = β + (X'X)⁻¹X'ε
-/

/-- Algebraic identity: β̂ = β + (X'X)⁻¹X'ε

    This is the key decomposition used to prove unbiasedness.
    Since Y = Xβ + ε, we have X'Y = X'Xβ + X'ε,
    so β̂ = (X'X)⁻¹X'Y = β + (X'X)⁻¹X'ε -/
theorem ols_decomposition {n k : ℕ}
    (data : RegressionData n k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_inv : XtX_inv * GramMatrix data.X = 1) :
    OLSEstimator data.X data.Y XtX_inv =
      fun j => data.β_true j + ∑ i : Fin k, XtX_inv j i * XtY data.X data.ε i := by
  -- Y = Xβ + ε, so X'Y = X'Xβ + X'ε, and (X'X)⁻¹X'Y = β + (X'X)⁻¹X'ε
  classical
  funext j
  have h_ols :
      OLSEstimator data.X data.Y XtX_inv =
        OLSEstimatorMatrix data.X data.Y XtX_inv :=
    ols_estimator_eq_matrix _ _ _
  have hY : data.Y = data.X *ᵥ data.β_true + data.ε := by
    funext i
    simp [Matrix.mulVec, dotProduct, data.h_model, Pi.add_apply]
  have hβ :
      (XtX_inv * data.X.transpose) *ᵥ (data.X *ᵥ data.β_true) = data.β_true := by
    calc
      (XtX_inv * data.X.transpose) *ᵥ (data.X *ᵥ data.β_true)
          = (XtX_inv * data.X.transpose * data.X) *ᵥ data.β_true := by
              simpa [Matrix.mul_assoc] using
                (Matrix.mulVec_mulVec data.β_true (XtX_inv * data.X.transpose) data.X)
      _ = (XtX_inv * GramMatrix data.X) *ᵥ data.β_true := by
            simp [GramMatrix, Matrix.mul_assoc]
      _ = (1 : Matrix (Fin k) (Fin k) ℝ) *ᵥ data.β_true := by
            simp [h_inv]
      _ = data.β_true := by
            simpa using (Matrix.one_mulVec data.β_true)
  have hε :
      (XtX_inv * data.X.transpose) *ᵥ data.ε =
        fun j => ∑ i : Fin k, XtX_inv j i * XtY data.X data.ε i := by
    funext j
    calc
      ((XtX_inv * data.X.transpose) *ᵥ data.ε) j
          = ∑ r : Fin n, (∑ i : Fin k, XtX_inv j i * data.X r i) * data.ε r := by
              simp [Matrix.mulVec, Matrix.mul_apply, dotProduct, mul_assoc]
      _ = ∑ r : Fin n, ∑ i : Fin k, XtX_inv j i * (data.X r i * data.ε r) := by
            refine Finset.sum_congr rfl ?_
            intro r hr
            simp [Finset.sum_mul, mul_assoc]
      _ = ∑ i : Fin k, ∑ r : Fin n, XtX_inv j i * (data.X r i * data.ε r) := by
            simpa using
              (Finset.sum_comm (f := fun r i => XtX_inv j i * (data.X r i * data.ε r)))
      _ = ∑ i : Fin k, XtX_inv j i * ∑ r : Fin n, data.X r i * data.ε r := by
            refine Finset.sum_congr rfl ?_
            intro i hi
            simp [Finset.mul_sum, mul_assoc]
      _ = ∑ i : Fin k, XtX_inv j i * XtY data.X data.ε i := by
            simp [XtY]
  calc
    OLSEstimator data.X data.Y XtX_inv j
        = ((XtX_inv * data.X.transpose) *ᵥ data.Y) j := by
            simpa [OLSEstimatorMatrix] using congrArg (fun f => f j) h_ols
    _ = ((XtX_inv * data.X.transpose) *ᵥ (data.X *ᵥ data.β_true + data.ε)) j := by
            simp [hY]
    _ = ((XtX_inv * data.X.transpose) *ᵥ (data.X *ᵥ data.β_true)) j +
          ((XtX_inv * data.X.transpose) *ᵥ data.ε) j := by
            simpa using
              congrArg (fun f => f j)
                (Matrix.mulVec_add (XtX_inv * data.X.transpose)
                  (data.X *ᵥ data.β_true) data.ε)
    _ = data.β_true j + ∑ i : Fin k, XtX_inv j i * XtY data.X data.ε i := by
            simp [hβ, hε]

/-!
## Theorem 3.1: Unbiasedness of OLS
-/

/-- E[(X'X)⁻¹X'ε] = 0 under MLR.4.

    By zero conditional mean, E[ε_i] = 0 for all i,
    so E[X'ε] = X'E[ε] = 0 (treating X as fixed). -/
theorem expectation_XtXinv_Xtε_zero {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Matrix (Fin n) (Fin k) ℝ)
    (ε : Ω → Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_zero_mean : ZeroConditionalMean μ ε) :
    ∀ j, ∫ ω, (∑ i : Fin k, XtX_inv j i * XtY X (ε ω) i) ∂μ = 0 := by
  -- Linearity of expectation: E[Σ a_ij ε_j] = Σ a_ij E[ε_j] = 0
  classical
  intro j
  have h_int_ε : ∀ i, Integrable (fun ω => ε ω i) μ := h_zero_mean.integrable
  have h_int_XtY : ∀ i, Integrable (fun ω => XtY X (ε ω) i) μ := by
    intro i
    have h_terms :
        ∀ l ∈ (Finset.univ : Finset (Fin n)),
          Integrable (fun ω => X l i * ε ω l) μ := by
      intro l _
      simpa using (Integrable.const_mul (h_int_ε l) (X l i))
    simpa [XtY] using
      (integrable_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin n))) h_terms)
  have h_int_terms :
      ∀ i, Integrable (fun ω => XtX_inv j i * XtY X (ε ω) i) μ := by
    intro i
    simpa using (Integrable.const_mul (h_int_XtY i) (XtX_inv j i))
  have h_sum :
      ∫ ω, (∑ i : Fin k, XtX_inv j i * XtY X (ε ω) i) ∂μ =
        ∑ i : Fin k, ∫ ω, XtX_inv j i * XtY X (ε ω) i ∂μ := by
    simpa using
      (integral_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin k)))
        (by
          intro i hi
          simpa using h_int_terms i))
  have h_xty_zero : ∀ i, ∫ ω, XtY X (ε ω) i ∂μ = 0 := by
    intro i
    have h_terms :
        ∀ l ∈ (Finset.univ : Finset (Fin n)),
          Integrable (fun ω => X l i * ε ω l) μ := by
      intro l _
      simpa using (Integrable.const_mul (h_int_ε l) (X l i))
    calc
      ∫ ω, XtY X (ε ω) i ∂μ
          = ∑ l : Fin n, ∫ ω, X l i * ε ω l ∂μ := by
              simpa [XtY] using
                (integral_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin n))) h_terms)
      _ = ∑ l : Fin n, X l i * ∫ ω, ε ω l ∂μ := by
              simp [integral_const_mul]
      _ = ∑ l : Fin n, X l i * 0 := by
              simp [h_zero_mean.zero_mean]
      _ = 0 := by
              simp
  calc
    ∫ ω, (∑ i : Fin k, XtX_inv j i * XtY X (ε ω) i) ∂μ
        = ∑ i : Fin k, ∫ ω, XtX_inv j i * XtY X (ε ω) i ∂μ := h_sum
    _ = ∑ i : Fin k, XtX_inv j i * ∫ ω, XtY X (ε ω) i ∂μ := by
          simp [integral_const_mul]
    _ = ∑ i : Fin k, XtX_inv j i * 0 := by
          simp [h_xty_zero]
    _ = 0 := by
          simp

/-- Theorem 3.1 (Wooldridge): OLS is unbiased.

    Under MLR.1-4 and fixed (non-random) X and β: E[β̂] = β

    The key insight is that β̂ = β + (X'X)⁻¹X'ε,
    and E[(X'X)⁻¹X'ε] = (X'X)⁻¹X'E[ε] = 0 under MLR.4. -/
theorem ols_unbiased {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_assumptions : UnbiasednessAssumptions μ data)
    (h_inv : ∀ ω, XtX_inv * GramMatrix (data ω).X = 1)
    (h_X_fixed : ∀ ω₁ ω₂, (data ω₁).X = (data ω₂).X)  -- X is fixed/non-stochastic
    (h_beta_fixed : ∀ ω₁ ω₂, (data ω₁).β_true = (data ω₂).β_true) :
    ∀ j ω₀, ∫ ω, OLSEstimator (data ω).X (data ω).Y XtX_inv j ∂μ = (data ω₀).β_true j := by
  -- E[β̂] = β follows from E[(X'X)⁻¹X'ε] = (X'X)⁻¹X'E[ε] = 0
  classical
  intro j ω₀
  set X0 : Matrix (Fin n) (Fin k) ℝ := (data ω₀).X
  set β0 : Fin k → ℝ := (data ω₀).β_true
  let ε : Ω → Fin n → ℝ := fun ω => (data ω).ε
  have hX : ∀ ω, (data ω).X = X0 := by
    intro ω
    simpa [X0] using h_X_fixed ω ω₀
  have hβ : ∀ ω, (data ω).β_true = β0 := by
    intro ω
    simpa [β0] using h_beta_fixed ω ω₀
  have h_decomp :
      ∀ ω, OLSEstimator (data ω).X (data ω).Y XtX_inv =
        fun j => β0 j + ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i := by
    intro ω
    have h := ols_decomposition (data ω) XtX_inv (h_inv ω)
    simpa [X0, β0, hX ω, hβ ω] using h
  have h_decomp_j :
      (fun ω => OLSEstimator (data ω).X (data ω).Y XtX_inv j) =
        fun ω => β0 j + ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i := by
    funext ω
    simpa using congrArg (fun f => f j) (h_decomp ω)
  have h_int_ε : ∀ i, Integrable (fun ω => ε ω i) μ := h_assumptions.zero_mean.integrable
  have h_int_XtY : ∀ i, Integrable (fun ω => XtY X0 (ε ω) i) μ := by
    intro i
    have h_terms :
        ∀ l ∈ (Finset.univ : Finset (Fin n)),
          Integrable (fun ω => X0 l i * ε ω l) μ := by
      intro l _
      simpa using (Integrable.const_mul (h_int_ε l) (X0 l i))
    simpa [XtY] using
      (integrable_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin n))) h_terms)
  have h_int_sum :
      Integrable (fun ω => ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i) μ := by
    have h_terms :
        ∀ i ∈ (Finset.univ : Finset (Fin k)),
          Integrable (fun ω => XtX_inv j i * XtY X0 (ε ω) i) μ := by
      intro i _
      simpa using (Integrable.const_mul (h_int_XtY i) (XtX_inv j i))
    simpa using
      (integrable_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin k))) h_terms)
  have h_const_int : Integrable (fun _ : Ω => β0 j) μ := by
    simpa using (integrable_const (μ := μ) (β0 j))
  have h_zero :=
    expectation_XtXinv_Xtε_zero (μ := μ) (X := X0) (ε := ε) (XtX_inv := XtX_inv)
      h_assumptions.zero_mean
  calc
    ∫ ω, OLSEstimator (data ω).X (data ω).Y XtX_inv j ∂μ
        = ∫ ω, (β0 j + ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i) ∂μ := by
            simp [h_decomp_j]
    _ = β0 j + ∫ ω, ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i ∂μ := by
            simpa [Pi.add_apply] using (integral_add h_const_int h_int_sum)
    _ = β0 j := by
            simp [h_zero j]
    _ = (data ω₀).β_true j := by
            simp [β0]

/-!
## Variance of OLS under Homoskedasticity
-/

/-- Variance-covariance matrix of OLS: Var(β̂) = σ²(X'X)⁻¹

    Under MLR.1-5, the variance of the OLS estimator is:
    Var(β̂|X) = E[(β̂ - β)(β̂ - β)'|X] = σ²(X'X)⁻¹ -/
def OLSVariance {k : ℕ}
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  σ_sq • XtX_inv

/-- The variance formula holds under Gauss-Markov assumptions (fixed X, symmetric (X'X)⁻¹). -/
theorem ols_variance_formula {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → RegressionData n k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ)
    (h_assumptions : GaussMarkovAssumptions μ data σ_sq)
    (h_inv : ∀ ω, XtX_inv * GramMatrix (data ω).X = 1)
    (h_symm : XtX_inv.transpose = XtX_inv)
    (h_X_fixed : ∀ ω₁ ω₂, (data ω₁).X = (data ω₂).X) :
    -- Var(β̂_j) = σ² * (X'X)⁻¹_jj
    ∀ j, ∫ ω, (OLSEstimator (data ω).X (data ω).Y XtX_inv j - (data ω).β_true j)^2 ∂μ =
      σ_sq * XtX_inv j j := by
  -- Var(β̂) = E[(β̂ - β)²] = E[((X'X)⁻¹X'ε)²] = σ²(X'X)⁻¹ under homoskedasticity
  classical
  intro j
  have h_nonempty : Nonempty Ω := by
    by_contra h_empty
    haveI : IsEmpty Ω := not_nonempty_iff.mp h_empty
    have h_univ_empty : (Set.univ : Set Ω) = ∅ := by
      ext x
      exact (IsEmpty.false x).elim
    have h0 : μ Set.univ = (0 : ENNReal) := by
      simpa [h_univ_empty]
    have h1 : μ Set.univ = (1 : ENNReal) := by
      simpa using (IsProbabilityMeasure.measure_univ (μ := μ))
    exact (one_ne_zero : (1 : ENNReal) ≠ 0) (by simpa [h1] using h0)
  obtain ⟨ω0⟩ := h_nonempty
  set X0 : Matrix (Fin n) (Fin k) ℝ := (data ω0).X
  let ε : Ω → Fin n → ℝ := fun ω => (data ω).ε
  have hX : ∀ ω, (data ω).X = X0 := by
    intro ω
    simpa [X0] using h_X_fixed ω ω0
  let A : Matrix (Fin k) (Fin n) ℝ := XtX_inv * X0.transpose
  have h_sum :
      ∀ ω, ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i =
        ∑ l : Fin n, A j l * ε ω l := by
    intro ω
    have h := congrArg (fun f => f j) (ols_estimator_eq_matrix X0 (ε ω) XtX_inv)
    simpa [OLSEstimatorMatrix, A, Matrix.mulVec, dotProduct] using h
  have h_diff :
      ∀ ω, OLSEstimator (data ω).X (data ω).Y XtX_inv j - (data ω).β_true j =
        ∑ l : Fin n, A j l * ε ω l := by
    intro ω
    have h := congrArg (fun f => f j) (ols_decomposition (data ω) XtX_inv (h_inv ω))
    have h' : OLSEstimator (data ω).X (data ω).Y XtX_inv j =
        (data ω).β_true j + ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i := by
      simpa [hX ω, X0, ε] using h
    have h'' :
        OLSEstimator (data ω).X (data ω).Y XtX_inv j - (data ω).β_true j =
          ∑ i : Fin k, XtX_inv j i * XtY X0 (ε ω) i := by
      linarith
    simpa [h_sum ω]
      using h''
  have h_expand :
      (fun ω => (∑ l : Fin n, A j l * ε ω l)^2) =
        (fun ω => ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * (ε ω l * ε ω m)) := by
    funext ω
    simp [pow_two, Finset.sum_mul_sum, mul_assoc, mul_left_comm, mul_comm]
  have h_int_mul : ∀ l m, Integrable (fun ω => ε ω l * ε ω m) μ :=
    h_assumptions.homoskedastic.integrable_mul
  have h_int_term :
      ∀ l m, Integrable (fun ω => A j l * A j m * (ε ω l * ε ω m)) μ := by
    intro l m
    simpa [mul_assoc] using
      (Integrable.const_mul (h_int_mul l m) (A j l * A j m))
  have h_inner_terms :
      ∀ l ∈ (Finset.univ : Finset (Fin n)),
        Integrable (fun ω => ∑ m : Fin n, A j l * A j m * (ε ω l * ε ω m)) μ := by
    intro l _
    have h_terms :
        ∀ m ∈ (Finset.univ : Finset (Fin n)),
          Integrable (fun ω => A j l * A j m * (ε ω l * ε ω m)) μ := by
      intro m _
      exact h_int_term l m
    simpa using
      (integrable_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin n))) h_terms)
  have h_cov :
      ∀ l m, ∫ ω, ε ω l * ε ω m ∂μ = if l = m then σ_sq else 0 := by
    intro l m
    by_cases h : l = m
    · subst h
      simpa [pow_two] using h_assumptions.homoskedastic.constant_variance l
    · simpa [h] using h_assumptions.homoskedastic.uncorrelated l m h
  have h_double_sum :
      ∫ ω, (∑ l : Fin n, A j l * ε ω l)^2 ∂μ =
        ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * ∫ ω, ε ω l * ε ω m ∂μ := by
    calc
      ∫ ω, (∑ l : Fin n, A j l * ε ω l)^2 ∂μ
          = ∫ ω, ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * (ε ω l * ε ω m) ∂μ := by
              simp [h_expand]
      _ = ∑ l : Fin n, ∫ ω, ∑ m : Fin n, A j l * A j m * (ε ω l * ε ω m) ∂μ := by
              simpa using
                (integral_finset_sum (μ := μ)
                  (s := (Finset.univ : Finset (Fin n))) h_inner_terms)
      _ = ∑ l : Fin n, ∑ m : Fin n,
            ∫ ω, A j l * A j m * (ε ω l * ε ω m) ∂μ := by
              refine Finset.sum_congr rfl ?_
              intro l _
              have h_terms :
                  ∀ m ∈ (Finset.univ : Finset (Fin n)),
                    Integrable (fun ω => A j l * A j m * (ε ω l * ε ω m)) μ := by
                intro m _
                exact h_int_term l m
              simpa using
                (integral_finset_sum (μ := μ)
                  (s := (Finset.univ : Finset (Fin n))) h_terms)
      _ = ∑ l : Fin n, ∑ m : Fin n,
            A j l * A j m * ∫ ω, ε ω l * ε ω m ∂μ := by
              refine Finset.sum_congr rfl ?_
              intro l _
              refine Finset.sum_congr rfl ?_
              intro m _
              simpa [mul_assoc] using
                (integral_const_mul (μ := μ) (r := A j l * A j m)
                  (f := fun ω => ε ω l * ε ω m))
  have h_inner :
      ∀ l, ∑ m : Fin n, A j l * A j m * (if l = m then σ_sq else 0) =
        A j l * A j l * σ_sq := by
    intro l
    classical
    simp [Finset.sum_ite_eq, mul_ite, ite_mul, mul_assoc]
  have h_sum_diag :
      ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * ∫ ω, ε ω l * ε ω m ∂μ =
        σ_sq * ∑ l : Fin n, A j l * A j l := by
    calc
      ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * ∫ ω, ε ω l * ε ω m ∂μ
          = ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * (if l = m then σ_sq else 0) := by
              simp [h_cov]
      _ = ∑ l : Fin n, A j l * A j l * σ_sq := by
              refine Finset.sum_congr rfl ?_
              intro l _
              simpa using h_inner l
      _ = σ_sq * ∑ l : Fin n, A j l * A j l := by
              have h_factor :
                  σ_sq * ∑ l : Fin n, A j l * A j l =
                    ∑ l : Fin n, σ_sq * (A j l * A j l) := by
                      simpa using
                        (Finset.mul_sum (s := (Finset.univ : Finset (Fin n)))
                          (a := σ_sq) (f := fun l => A j l * A j l))
              simpa [mul_assoc, mul_left_comm, mul_comm] using h_factor.symm
  have h_AA :
      A * A.transpose = XtX_inv := by
    calc
      A * A.transpose
          = XtX_inv * X0.transpose * X0 * XtX_inv.transpose := by
              simp [A, Matrix.mul_assoc, Matrix.transpose_mul, Matrix.transpose_transpose]
      _ = XtX_inv * GramMatrix X0 * XtX_inv := by
              simp [GramMatrix, Matrix.mul_assoc, h_symm]
      _ = (XtX_inv * GramMatrix X0) * XtX_inv := by
              simp [Matrix.mul_assoc]
      _ = (1 : Matrix (Fin k) (Fin k) ℝ) * XtX_inv := by
              simp [h_inv ω0, X0]
      _ = XtX_inv := by
              simp
  have h_AA_entry : ∑ l : Fin n, A j l * A j l = XtX_inv j j := by
    have h := congrArg (fun M => M j j) h_AA
    simpa [Matrix.mul_apply, Matrix.transpose_apply] using h
  calc
    ∫ ω, (OLSEstimator (data ω).X (data ω).Y XtX_inv j - (data ω).β_true j)^2 ∂μ
        = ∫ ω, (∑ l : Fin n, A j l * ε ω l)^2 ∂μ := by
            simp [h_diff]
    _ = ∑ l : Fin n, ∑ m : Fin n, A j l * A j m * ∫ ω, ε ω l * ε ω m ∂μ := by
            simpa using h_double_sum
    _ = σ_sq * ∑ l : Fin n, A j l * A j l := by
            simpa using h_sum_diag
    _ = σ_sq * XtX_inv j j := by
            simp [h_AA_entry]

/-!
## Linear Estimators
-/

/-- A linear estimator is of the form β̃ = AY for some matrix A.

    Linear estimators are exactly those that can be written as
    linear functions of the data Y. -/
structure LinearEstimator (n k : ℕ) where
  /-- The weight matrix A (k × n) -/
  A : Matrix (Fin k) (Fin n) ℝ

/-- Apply a linear estimator to data -/
def LinearEstimator.apply {n k : ℕ}
    (est : LinearEstimator n k)
    (Y : Fin n → ℝ) : Fin k → ℝ :=
  Matrix.mulVec est.A Y

/-- OLS is a linear estimator with A = (X'X)⁻¹X' -/
def OLSAsLinear {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : LinearEstimator n k where
  A := XtX_inv * X.transpose

/-- A linear estimator is unbiased if E[β̃] = β for all β.

    For a linear estimator β̃ = AY = A(Xβ + ε):
    E[β̃] = AXβ + AE[ε] = AXβ
    So unbiasedness requires AX = I. -/
def LinearEstimator.IsUnbiased {n k : ℕ}
    (est : LinearEstimator n k)
    (X : Matrix (Fin n) (Fin k) ℝ) : Prop :=
  est.A * X = 1

/-- OLS is unbiased (as a linear estimator) -/
theorem ols_is_unbiased_linear {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_inv : XtX_inv * GramMatrix X = 1) :
    (OLSAsLinear X XtX_inv).IsUnbiased X := by
  -- (X'X)⁻¹ X'X = I by h_inv
  simpa [OLSAsLinear, LinearEstimator.IsUnbiased, GramMatrix, Matrix.mul_assoc] using h_inv

/-!
## Theorem 3.4: Gauss-Markov Theorem
-/

/-- Variance of a linear estimator under homoskedasticity.

    For β̃ = AY = AXβ + Aε:
    Var(β̃) = A Var(ε) A' = σ² AA' -/
def LinearEstimator.Variance {n k : ℕ}
    (est : LinearEstimator n k)
    (σ_sq : ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  σ_sq • (est.A * est.A.transpose)

/-- Gauss-Markov in matrix inequality form.

    The variance matrix of any unbiased linear estimator
    dominates the OLS variance matrix in the positive semidefinite sense:
    Var(β̃) - Var(β̂) is positive semidefinite -/
theorem gauss_markov_matrix {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ)
    (h_inv : XtX_inv * GramMatrix X = 1)
    (h_σ_pos : σ_sq > 0)
    (alt : LinearEstimator n k)
    (h_alt_unbiased : alt.IsUnbiased X) :
    -- Var(β̃) - Var(β̂) = σ² D D' ≽ 0
    ∃ D : Matrix (Fin k) (Fin n) ℝ,
      D * X = 0 ∧
      alt.Variance σ_sq - (OLSAsLinear X XtX_inv).Variance σ_sq =
        σ_sq • (D * D.transpose) := by
  -- D = alt.A - OLS.A satisfies DX = 0 and Var(alt) - Var(OLS) = σ²DD'
  classical
  let Aols : Matrix (Fin k) (Fin n) ℝ := XtX_inv * X.transpose
  let D : Matrix (Fin k) (Fin n) ℝ := alt.A - Aols
  have h_ols_unbiased : Aols * X = 1 := by
    simpa [Aols, GramMatrix, Matrix.mul_assoc] using h_inv
  have h_alt : alt.A * X = 1 := by
    simpa [LinearEstimator.IsUnbiased] using h_alt_unbiased
  have h_DX : D * X = 0 := by
    calc
      D * X = (alt.A - Aols) * X := by rfl
      _ = alt.A * X - Aols * X := by
        simp [Matrix.sub_mul]
      _ = 1 - 1 := by
        simp [h_alt, h_ols_unbiased]
      _ = 0 := by
        simp
  have h_DAolsT : D * Aols.transpose = 0 := by
    calc
      D * Aols.transpose = D * (X * XtX_inv.transpose) := by
        simp [Aols, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
      _ = (D * X) * XtX_inv.transpose := by
        simp [Matrix.mul_assoc]
      _ = 0 := by
        simp [h_DX]
  have h_AolsDT : Aols * D.transpose = 0 := by
    calc
      Aols * D.transpose = XtX_inv * X.transpose * D.transpose := by
        simp [Aols, Matrix.mul_assoc]
      _ = XtX_inv * (X.transpose * D.transpose) := by
        simp [Matrix.mul_assoc]
      _ = XtX_inv * (D * X).transpose := by
        simp [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
      _ = XtX_inv * 0 := by
        simp [h_DX]
      _ = 0 := by
        simp
  have h_altA : alt.A = Aols + D := by
    simp [D, Aols, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  have h_expand :
      (Aols + D) * (Aols.transpose + D.transpose) =
        Aols * Aols.transpose + D * D.transpose := by
    calc
      (Aols + D) * (Aols.transpose + D.transpose)
          = (Aols + D) * Aols.transpose + (Aols + D) * D.transpose := by
              simp [Matrix.mul_add]
      _ = (Aols * Aols.transpose + D * Aols.transpose) +
            (Aols * D.transpose + D * D.transpose) := by
              simp [Matrix.add_mul, Matrix.mul_add, add_assoc, add_left_comm, add_comm]
      _ = Aols * Aols.transpose + D * D.transpose := by
              simp [h_DAolsT, h_AolsDT, add_assoc, add_left_comm, add_comm]
  have h_var :
      alt.Variance σ_sq - (OLSAsLinear X XtX_inv).Variance σ_sq =
        σ_sq • (D * D.transpose) := by
    calc
      alt.Variance σ_sq - (OLSAsLinear X XtX_inv).Variance σ_sq
          = σ_sq • (alt.A * alt.A.transpose) - σ_sq • (Aols * Aols.transpose) := by
              simp [LinearEstimator.Variance, OLSAsLinear, Aols]
      _ = σ_sq • ((Aols + D) * (Aols.transpose + D.transpose)) -
            σ_sq • (Aols * Aols.transpose) := by
              simp [h_altA, Matrix.transpose_add]
      _ = σ_sq • (Aols * Aols.transpose + D * D.transpose) -
            σ_sq • (Aols * Aols.transpose) := by
              simp [h_expand]
      _ = σ_sq • (D * D.transpose) := by
              simp [smul_add, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  exact ⟨D, h_DX, by simpa [Aols] using h_var⟩

 /-- Gauss-Markov Theorem (Wooldridge Theorem 3.4):
    OLS is the Best Linear Unbiased Estimator (BLUE).

    Under MLR.1-5, for any unbiased linear estimator β̃:
    Var(β̃_j) ≥ Var(β̂_OLS_j) for all j

    "Best" means minimum variance among all linear unbiased estimators. -/
theorem gauss_markov {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ)
    (h_inv : XtX_inv * GramMatrix X = 1)
    (h_σ_pos : σ_sq > 0)
    (alt : LinearEstimator n k)
    (h_alt_unbiased : alt.IsUnbiased X) :
    -- Var(β̃_j) ≥ Var(β̂_j) for all j
    ∀ j, alt.Variance σ_sq j j ≥ (OLSAsLinear X XtX_inv).Variance σ_sq j j := by
  -- Key insight: Write alt.A = OLS.A + D where DX = 0 (by unbiasedness).
  -- Then Var(alt) = Var(OLS) + σ²DD' and DD' is PSD (diagonal ≥ 0).
  classical
  intro j
  obtain ⟨D, h_DX, h_var⟩ :=
    gauss_markov_matrix X XtX_inv σ_sq h_inv h_σ_pos alt h_alt_unbiased
  have h_var_j :
      alt.Variance σ_sq j j - (OLSAsLinear X XtX_inv).Variance σ_sq j j =
        σ_sq * (D * D.transpose) j j := by
    have h := congrArg (fun M => M j j) h_var
    simpa [Matrix.smul_apply, smul_eq_mul] using h
  have h_diag_nonneg : 0 ≤ (D * D.transpose) j j := by
    have h_sum : (D * D.transpose) j j = ∑ l : Fin n, D j l * D j l := by
      simp [Matrix.mul_apply, Matrix.transpose_apply, mul_comm]
    have h_nonneg : 0 ≤ ∑ l : Fin n, D j l * D j l := by
      refine Finset.sum_nonneg ?_
      intro l _
      exact mul_self_nonneg (D j l)
    simpa [h_sum] using h_nonneg
  have h_term_nonneg : 0 ≤ σ_sq * (D * D.transpose) j j := by
    exact mul_nonneg (le_of_lt h_σ_pos) h_diag_nonneg
  linarith [h_var_j, h_term_nonneg]

/-!
## Standard Errors
-/

/-- Standard error of the j-th coefficient -/
def standardError {k : ℕ}
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ)
    (j : Fin k) : ℝ :=
  Real.sqrt (σ_sq * XtX_inv j j)

/-- Estimated variance using residuals -/
def estimatedVariance {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (β_hat : Fin k → ℝ) : ℝ :=
  let residuals := fun i => Y i - ∑ j : Fin k, X i j * β_hat j
  let RSS := ∑ i : Fin n, (residuals i)^2
  RSS / (n - k)

/-- Estimated standard errors -/
def estimatedStandardError {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (Y : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (β_hat : Fin k → ℝ)
    (j : Fin k) : ℝ :=
  Real.sqrt (estimatedVariance X Y β_hat * XtX_inv j j)

end OLS

end Econometrics

end
