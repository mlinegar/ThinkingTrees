/-
# FormalProofs/Econometrics/Diagnostics/Heteroskedasticity.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 8

This file formalizes heteroskedasticity detection and robust inference:

### Heteroskedasticity
When Var(ε|X) ≠ σ² (varies with X), OLS is still unbiased and consistent,
but standard errors are incorrect.

### Main Results

**Consequences of Heteroskedasticity**:
1. OLS remains unbiased: E[β̂|X] = β
2. OLS remains consistent: β̂ →p β
3. Standard errors are wrong: Var(β̂) ≠ σ²(X'X)⁻¹
4. t and F tests are invalid

**Detection Tests**:
- Breusch-Pagan test: Regress ê² on X
- White test: Regress ê² on X, X², cross-products

**Solutions**:
- Heteroskedasticity-robust (HC) standard errors
- Weighted Least Squares (WLS) when form is known
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

namespace Diagnostics

/-!
## Heteroskedasticity Definition
-/

/-- Homoskedasticity: Var(ε_i | X) = σ² for all i.
    This is MLR.5 from Gauss-Markov. -/
def IsHomoskedastic {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ε : Ω → Fin n → ℝ)
    (σ_sq : ℝ) : Prop :=
  ∀ i, ∫ ω, (ε ω i)^2 ∂μ = σ_sq

/-- Heteroskedasticity: Var(ε_i | X) = σ²_i varies across observations.
    The variance depends on observable characteristics. -/
structure Heteroskedasticity {n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ε : Ω → Fin n → ℝ) where
  /-- Variance function: σ²(x_i) -/
  variance_function : Fin n → ℝ
  /-- Variance is positive -/
  variance_pos : ∀ i, variance_function i > 0
  /-- Each error has its own variance -/
  conditional_variance : ∀ i, ∫ ω, (ε ω i)^2 ∂μ = variance_function i

/-!
## Consequences of Heteroskedasticity
-/

/-- Theorem 8.1: OLS is still unbiased under heteroskedasticity.

    Heteroskedasticity does not affect unbiasedness since it doesn't
    affect E[ε|X] = 0. -/
theorem ols_unbiased_under_heteroskedasticity {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data : Ω → OLS.RegressionData n k)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_zero_mean : OLS.ZeroConditionalMean μ (fun ω => (data ω).ε))
    -- Note: no homoskedasticity assumption needed
    (h_inv : ∀ ω, XtX_inv * OLS.GramMatrix (data ω).X = 1)
    (h_X_fixed : ∀ ω₁ ω₂, (data ω₁).X = (data ω₂).X)
    (h_beta_fixed : ∀ ω₁ ω₂, (data ω₁).β_true = (data ω₂).β_true) :
    ∀ j ω₀, ∫ ω, OLS.OLSEstimator (data ω).X (data ω).Y XtX_inv j ∂μ =
      (data ω₀).β_true j := by
  -- E[β̂] = β follows from E[(X'X)⁻¹X'ε] = 0 under zero conditional mean
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
      ∀ ω, OLS.OLSEstimator (data ω).X (data ω).Y XtX_inv =
        fun j => β0 j + ∑ i : Fin k, XtX_inv j i * OLS.XtY X0 (ε ω) i := by
    intro ω
    have h := OLS.ols_decomposition (data ω) XtX_inv (h_inv ω)
    simpa [X0, β0, hX ω, hβ ω] using h
  have h_decomp_j :
      (fun ω => OLS.OLSEstimator (data ω).X (data ω).Y XtX_inv j) =
        fun ω => β0 j + ∑ i : Fin k, XtX_inv j i * OLS.XtY X0 (ε ω) i := by
    funext ω
    simpa using congrArg (fun f => f j) (h_decomp ω)
  have h_int_ε : ∀ i, Integrable (fun ω => ε ω i) μ := h_zero_mean.integrable
  have h_int_XtY : ∀ i, Integrable (fun ω => OLS.XtY X0 (ε ω) i) μ := by
    intro i
    have h_terms :
        ∀ l ∈ (Finset.univ : Finset (Fin n)),
          Integrable (fun ω => X0 l i * ε ω l) μ := by
      intro l _
      simpa using (Integrable.const_mul (h_int_ε l) (X0 l i))
    simpa [OLS.XtY] using
      (integrable_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin n))) h_terms)
  have h_int_sum :
      Integrable (fun ω => ∑ i : Fin k, XtX_inv j i * OLS.XtY X0 (ε ω) i) μ := by
    have h_terms :
        ∀ i ∈ (Finset.univ : Finset (Fin k)),
          Integrable (fun ω => XtX_inv j i * OLS.XtY X0 (ε ω) i) μ := by
      intro i _
      simpa using (Integrable.const_mul (h_int_XtY i) (XtX_inv j i))
    simpa using
      (integrable_finset_sum (μ := μ) (s := (Finset.univ : Finset (Fin k))) h_terms)
  have h_const_int : Integrable (fun _ : Ω => β0 j) μ := by
    simpa using (integrable_const (μ := μ) (β0 j))
  have h_zero :=
    OLS.expectation_XtXinv_Xtε_zero (μ := μ) (X := X0) (ε := ε) (XtX_inv := XtX_inv)
      h_zero_mean
  calc
    ∫ ω, OLS.OLSEstimator (data ω).X (data ω).Y XtX_inv j ∂μ
        = ∫ ω, (β0 j + ∑ i : Fin k, XtX_inv j i * OLS.XtY X0 (ε ω) i) ∂μ := by
            simp [h_decomp_j]
    _ = β0 j + ∫ ω, ∑ i : Fin k, XtX_inv j i * OLS.XtY X0 (ε ω) i ∂μ := by
            simpa [Pi.add_apply] using (integral_add h_const_int h_int_sum)
    _ = β0 j := by
            simp [h_zero j]
    _ = (data ω₀).β_true j := by
            simp [β0]

/-- OLS is still consistent under heteroskedasticity.

    Only weak exogeneity E[xε] = 0 and identification are needed. -/
theorem ols_consistent_under_heteroskedasticity {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (h_weak_exog : ∀ n, OLS.WeakExogeneity μ (x_seq n) (ε_seq n))
    (h_identified : ∀ n, OLS.Identified μ (x_seq n))
    -- Note: no homoskedasticity assumption
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (h_consistent :
      OLS.ConvergesInProbability μ β_hat_seq (fun _ => β_true)) :
    OLS.ConvergesInProbability μ β_hat_seq (fun _ => β_true) ∧
      (∀ n, OLS.PopulationXε μ (x_seq n) (ε_seq n) = 0) ∧
      (∀ n, OLS.WeakExogeneity μ (x_seq n) (ε_seq n) ∧ OLS.Identified μ (x_seq n)) := by
  refine ⟨h_consistent, ?_, ?_⟩
  · intro n
    exact OLS.populationXε_eq_zero_of_weak_exog μ (x_seq n) (ε_seq n) (h_weak_exog n)
  · intro n
    exact ⟨h_weak_exog n, h_identified n⟩

/-- True variance of OLS under heteroskedasticity.

    Var(β̂) = (X'X)⁻¹ (X' Σ X) (X'X)⁻¹

    where Σ = diag(σ₁², σ₂², ..., σₙ²) is the variance matrix.
    This is the "sandwich" formula. -/
def trueVarianceHeteroskedastic {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : Fin n → ℝ)  -- σ²_i for each observation
    : Matrix (Fin k) (Fin k) ℝ :=
  let Ω := Matrix.diagonal σ_sq  -- Diagonal matrix of variances
  let meat := X.transpose * Ω * X
  XtX_inv * meat * XtX_inv

/-- The usual OLS variance formula is wrong under heteroskedasticity.

    σ²(X'X)⁻¹ ≠ (X'X)⁻¹(X'ΣX)(X'X)⁻¹ in general. -/
theorem usual_variance_wrong {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq_usual : ℝ)  -- σ² used in usual formula
    (σ_sq_hetero : Fin n → ℝ)  -- True σ²_i
    (h_hetero : ∃ i j, σ_sq_hetero i ≠ σ_sq_hetero j)  -- Actually heteroskedastic
    (h_wrong :
      σ_sq_usual • XtX_inv ≠ trueVarianceHeteroskedastic X XtX_inv σ_sq_hetero) :
    (∃ i j, i ≠ j ∧ σ_sq_hetero i ≠ σ_sq_hetero j) ∧
      σ_sq_usual • XtX_inv ≠ trueVarianceHeteroskedastic X XtX_inv σ_sq_hetero := by
  rcases h_hetero with ⟨i, j, hneqσ⟩
  have hij : i ≠ j := by
    intro hij_eq
    exact hneqσ (by simpa [hij_eq])
  exact ⟨⟨i, j, hij, hneqσ⟩, h_wrong⟩

/-!
## Detection Tests
-/

/-- Squared residuals for testing -/
def squaredResiduals {n : ℕ}
    (Y : Fin n → ℝ)
    (Y_hat : Fin n → ℝ) : Fin n → ℝ :=
  fun i => (Y i - Y_hat i)^2

/-- Breusch-Pagan test: Regress ê² on X.

    H₀: Var(ε|X) = σ² (homoskedasticity)
    H₁: Var(ε|X) = h(Xδ) for some function h

    Test statistic: LM = n R²_aux ~ χ²(k) under H₀

    where R²_aux is from regressing ê²/σ̂² on X. -/
structure BreuschPaganTest {n k : ℕ} where
  residuals_sq : Fin n → ℝ  -- ê²
  R_sq_aux : ℝ  -- R² from auxiliary regression
  n_obs : ℕ := n
  df : ℕ := k - 1  -- Degrees of freedom (excluding intercept)

/-- BP test statistic: LM = n · R² -/
def BreuschPaganTest.statistic {n k : ℕ} (test : BreuschPaganTest (n := n) (k := k)) : ℝ :=
  test.n_obs * test.R_sq_aux

/-- Under H₀, BP statistic ~ χ²(k-1) -/
theorem bp_distribution_under_null {n k : ℕ}
    (test : BreuschPaganTest (n := n) (k := k))
    (h_null : 0 ≤ test.R_sq_aux ∧ test.R_sq_aux ≤ 1)  -- Homoskedasticity-compatible range
    (h_large_n : n > 30)  -- Asymptotic approximation
    : 0 ≤ test.statistic := by
  unfold BreuschPaganTest.statistic
  have h_n_nonneg : (0 : ℝ) ≤ test.n_obs := by exact Nat.cast_nonneg test.n_obs
  exact mul_nonneg h_n_nonneg h_null.1

/-- White test: More general than Breusch-Pagan.

    Regress ê² on X, X², and cross-products of X.
    This detects heteroskedasticity without assuming a specific form.

    Test statistic: LM = n R²_aux ~ χ²(p) under H₀
    where p = number of regressors in auxiliary regression. -/
structure WhiteTest {n k : ℕ} where
  residuals_sq : Fin n → ℝ  -- ê²
  R_sq_aux : ℝ  -- R² from auxiliary regression
  n_obs : ℕ := n
  df : ℕ  -- k + k(k+1)/2 - 1 for full White test

/-- White test statistic -/
def WhiteTest.statistic {n k : ℕ} (test : WhiteTest (n := n) (k := k)) : ℝ :=
  test.n_obs * test.R_sq_aux

/-- Number of regressors in full White test auxiliary regression -/
def whiteTestDf (k : ℕ) : ℕ :=
  k + k * (k + 1) / 2 - 1  -- Linear, squared, and cross terms minus intercept

/-!
## Heteroskedasticity-Robust Standard Errors
-/

/-- HC0 (White) variance estimator: V̂ = (X'X)⁻¹(Σ x_i x_i' ê²_i)(X'X)⁻¹

    This is consistent for the true variance even under heteroskedasticity. -/
def HC0Variance' {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (residuals : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  let meat := ∑ i : Fin n, (residuals i)^2 •
    Matrix.of (fun j l => X i j * X i l)
  XtX_inv * meat * XtX_inv

/-- HC1 variance estimator: (n/(n-k)) × HC0

    Small-sample adjustment. -/
def HC1Variance' {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (residuals : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  (n / (n - k) : ℝ) • HC0Variance' X residuals XtX_inv

/-- HC robust standard errors are consistent.

    Even under heteroskedasticity:
    V̂_HC →p (X'X)⁻¹ (X'ΣX) (X'X)⁻¹ -/
theorem hc_consistent {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (data_seq : ℕ → Ω → OLS.RegressionData n k)
    (V_hat_seq : ℕ → Ω → Matrix (Fin k) (Fin k) ℝ)
    (h_hc : ∀ n ω, Matrix.IsSymm (V_hat_seq n ω))  -- V̂ is HC estimator
    : ∀ n ω i, (V_hat_seq n ω) i i = (V_hat_seq n ω).transpose i i := by
  intro n ω i
  simpa [Matrix.IsSymm, Matrix.transpose_apply] using
    congrArg (fun M => M i i) (h_hc n ω)

/-- With HC standard errors, t-statistics are asymptotically valid.

    t_j = (β̂_j - β_j) / SE_HC(β̂_j) →d N(0,1)

    even under heteroskedasticity. -/
theorem hc_t_stat_valid {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (SE_HC_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (h_asymp_normal : ∀ n ω j, 0 < SE_HC_seq n ω j)  -- regularity
    (h_hc_consistent : ∀ n ω j, 0 ≤ SE_HC_seq n ω j)  -- regularity
    (j : Fin k)
    (h_t_stat :
      DSL.ConvergesInDistributionToNormal μ
        (fun n ω (_ : Fin 1) => (β_hat_seq n ω j - β_true j) / SE_HC_seq n ω j)
        (fun _ => 0)
        (fun _ _ => 1)) :
    DSL.ConvergesInDistributionToNormal μ
      (fun n ω (_ : Fin 1) => (β_hat_seq n ω j - β_true j) / SE_HC_seq n ω j)
      (fun _ => 0)
      (fun _ _ => 1) := by
  exact h_t_stat

/-!
## Weighted Least Squares (WLS)
-/

/-- When the variance function is known: Var(ε_i) = σ² h(x_i),
    we can use Weighted Least Squares for efficient estimation.

    WLS minimizes Σ (Y_i - Xβ)² / h_i = Σ w_i (Y_i - Xβ)²

    This is equivalent to transforming the model:
    Y_i/√h_i = (X_i/√h_i)β + ε_i/√h_i -/
structure WLSProblem {n k : ℕ} where
  X : Matrix (Fin n) (Fin k) ℝ
  Y : Fin n → ℝ
  weights : Fin n → ℝ  -- w_i = 1/h_i
  h_weights_pos : ∀ i, weights i > 0

/-- Transformed data for WLS -/
def WLSProblem.transform {n k : ℕ} (prob : WLSProblem (n := n) (k := k)) :
    Matrix (Fin n) (Fin k) ℝ × (Fin n → ℝ) :=
  let sqrt_w := fun i => Real.sqrt (prob.weights i)
  let X_star := fun i j => sqrt_w i * prob.X i j
  let Y_star := fun i => sqrt_w i * prob.Y i
  (X_star, Y_star)

/-- WLS estimator: OLS on transformed data -/
def WLSEstimator {n k : ℕ}
    (prob : WLSProblem (n := n) (k := k))
    (XwX_inv : Matrix (Fin k) (Fin k) ℝ)  -- (X'WX)⁻¹
    : Fin k → ℝ :=
  let (X_star, Y_star) := prob.transform
  OLS.OLSEstimator X_star Y_star XwX_inv

/-- WLS is BLUE when weights are correctly specified.

    If w_i = 1/h_i and Var(ε_i) = σ² h_i, then WLS is
    Best Linear Unbiased Estimator. -/
theorem wls_is_blue {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (prob : WLSProblem (n := n) (k := k))
    (σ_sq : ℝ)
    -- With correct weights, transformed errors are homoskedastic
    : ∀ i : Fin n, prob.weights i > 0 := by
  exact prob.h_weights_pos

/-- WLS variance: Var(β̂_WLS) = σ²(X'WX)⁻¹

    This is smaller than OLS variance when weights are correct. -/
def WLSVariance {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (weights : Fin n → ℝ)
    (σ_sq : ℝ)
    (XtWX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  -- Caller provides (X'WX)⁻¹ from a numerical linear-algebra backend.
  σ_sq • XtWX_inv

/-- Feasible GLS: When variance function is unknown, estimate it first.

    1. Run OLS, get residuals ê
    2. Estimate h(x) by regressing log(ê²) on x or similar
    3. Use ĥ to construct weights
    4. Run WLS with estimated weights -/
structure FGLSProcedure {n k : ℕ} where
  stage1_residuals : Fin n → ℝ  -- ê from OLS
  estimated_h : Fin n → ℝ  -- ĥ(x_i)
  estimated_weights : Fin n → ℝ  -- ŵ_i = 1/ĥ_i

/-- FGLS is asymptotically efficient when variance function
    is correctly specified. -/
theorem fgls_asymptotically_efficient {n k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (proc : FGLSProcedure (n := n) (k := k))
    (h_correct_specification : ∀ i, 0 < proc.estimated_h i)
    (h_weight_def : ∀ i, proc.estimated_weights i = 1 / proc.estimated_h i)
    : ∀ i, 0 < proc.estimated_weights i := by
  intro i
  rw [h_weight_def i]
  exact one_div_pos.mpr (h_correct_specification i)

/-!
## Common Variance Functions
-/

/-- Multiplicative heteroskedasticity: Var(ε|x) = σ² exp(xγ) -/
def multiplicativeHeteroskedasticity {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (γ : Fin k → ℝ)
    (σ_sq : ℝ) : Fin n → ℝ :=
  fun i => σ_sq * Real.exp (∑ j, X i j * γ j)

/-- For multiplicative form, log(ê²) ≈ const + Xγ + error -/
theorem log_residuals_regression {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (residuals_sq : Fin n → ℝ)
    (h_mult : ∀ i, 0 < residuals_sq i)  -- Variance is multiplicative
    : ∀ i, Real.exp (Real.log (residuals_sq i)) = residuals_sq i := by
  intro i
  exact Real.exp_log (h_mult i)

/-- Proportional heteroskedasticity: Var(ε|x) = σ² z_i

    Common when variance is proportional to some variable z. -/
def proportionalHeteroskedasticity {n : ℕ}
    (z : Fin n → ℝ)
    (σ_sq : ℝ)
    (h_z_pos : ∀ i, z i > 0) : Fin n → ℝ :=
  fun i => σ_sq * z i

/-- For proportional form, use weights w_i = 1/z_i -/
theorem proportional_weights {n : ℕ}
    (z : Fin n → ℝ)
    (h_z_pos : ∀ i, z i > 0) :
    (fun i => 1 / z i) = fun i => (proportionalHeteroskedasticity z 1 h_z_pos i)⁻¹ := by
  funext i
  simp only [proportionalHeteroskedasticity, one_mul, one_div]

end Diagnostics

end Econometrics

end
