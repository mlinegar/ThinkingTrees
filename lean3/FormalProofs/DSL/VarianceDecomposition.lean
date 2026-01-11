/-
# FormalProofs/DSL/VarianceDecomposition.lean

## Paper Reference: Appendix B.4, Section 4 (Power Analysis)

This file formalizes the variance decomposition and power analysis for DSL:
- Variance as a function of n (expert annotations)
- Effect of prediction quality on variance
- Power calculations
- Sample size requirements

### Key Results

1. **Increasing n reduces variance**: More expert annotations → smaller SEs
2. **Better predictions reduce variance**: Higher accuracy → smaller SEs
3. **Valid inference regardless of accuracy**: Unlike naive approach

### Variance Decomposition

Var(β̂_DSL) ≈ Var_full + (1/n - 1/N) · Var_correction

where Var_correction depends on prediction accuracy.
-/

import FormalProofs.DSL.AsymptoticTheory

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
## Variance as Function of n
-/

/-- Variance of DSL estimator as a function of sample sizes.

    Parameters:
    - N: total population size
    - n: number of expert-coded documents
    - σ²_Y: variance of true outcome
    - σ²_pred: prediction error variance
    - ρ: correlation between prediction error and covariates -/
structure DSLVarianceParams where
  N : ℕ                    -- Total population
  n : ℕ                    -- Expert-coded sample
  σ_Y_sq : ℝ               -- Var(Y)
  σ_pred_sq : ℝ            -- Var(Ŷ - Y)
  h_n_le_N : n ≤ N
  h_N_pos : N > 0
  h_n_pos : n > 0

/-- Sampling probability π = n/N -/
def DSLVarianceParams.π (p : DSLVarianceParams) : ℝ :=
  (p.n : ℝ) / (p.N : ℝ)

/-- Approximate variance formula for DSL (simplified).

    Var(β̂_DSL) ≈ σ²_Y/N + (1/n - 1/N) · σ²_pred

    The first term is the irreducible variance (what you'd get with all labels).
    The second term is the penalty for using predictions on unlabeled data. -/
def approximateVariance (p : DSLVarianceParams) : ℝ :=
  p.σ_Y_sq / p.N + (1 / p.n - 1 / p.N) * p.σ_pred_sq

/-- Variance decreases with n.

    ∂Var/∂n = -σ²_pred/n² < 0

    More expert annotations always reduce variance. -/
theorem variance_decreases_with_n
    (p1 p2 : DSLVarianceParams)
    (h_same_N : p1.N = p2.N)
    (h_same_σY : p1.σ_Y_sq = p2.σ_Y_sq)
    (h_same_σpred : p1.σ_pred_sq = p2.σ_pred_sq)
    (h_n : p1.n < p2.n)
    (h_σ_nonneg : p1.σ_pred_sq ≥ 0)
    : approximateVariance p2 ≤ approximateVariance p1 := by
  unfold approximateVariance
  have hpos_n1 : (0 : ℝ) < p1.n := Nat.cast_pos.mpr p1.h_n_pos
  have hle_n : (p1.n : ℝ) ≤ p2.n := Nat.cast_le.mpr (Nat.le_of_lt h_n)
  have h_div : (1 : ℝ) / p2.n ≤ 1 / p1.n :=
    one_div_le_one_div_of_le hpos_n1 hle_n
  have h_sub : (1 : ℝ) / p2.n - 1 / p1.N ≤ 1 / p1.n - 1 / p1.N :=
    sub_le_sub_right h_div _
  have h_mul :
      (1 / p2.n - 1 / p1.N) * p1.σ_pred_sq ≤
      (1 / p1.n - 1 / p1.N) * p1.σ_pred_sq := by
    exact mul_le_mul_of_nonneg_right h_sub h_σ_nonneg
  have h_mul' :
      (1 / p2.n - 1 / p2.N) * p2.σ_pred_sq ≤
      (1 / p1.n - 1 / p2.N) * p2.σ_pred_sq := by
    simpa [h_same_N, h_same_σpred] using h_mul
  rw [h_same_N, h_same_σY, h_same_σpred]
  have h_add := add_le_add_left h_mul' (p2.σ_Y_sq / p2.N)
  simpa [add_comm, add_left_comm, add_assoc] using h_add

/-- Limiting variance as n → N (full sample).

    When all documents are expert-coded:
    Var(β̂_DSL) → σ²_Y/N

    This is the irreducible variance from sampling variability. -/
theorem variance_limit_full_sample
    (σ_Y_sq σ_pred_sq : ℝ) (N : ℕ) (hN : N > 0)
    : approximateVariance ⟨N, N, σ_Y_sq, σ_pred_sq, le_refl N, hN, hN⟩ = σ_Y_sq / N := by
  unfold approximateVariance
  simp

/-!
## Effect of Prediction Quality
-/

/-- Variance decreases with better predictions.

    Better predictions → smaller σ²_pred → smaller variance.

    This is the efficiency gain from using good LLMs. -/
theorem variance_decreases_with_accuracy
    (p1 p2 : DSLVarianceParams)
    (h_same_N : p1.N = p2.N)
    (h_same_n : p1.n = p2.n)
    (h_same_σY : p1.σ_Y_sq = p2.σ_Y_sq)
    (h_better : p2.σ_pred_sq ≤ p1.σ_pred_sq)  -- Better predictions
    (h_pos_factor : (1 : ℝ) / p1.n - 1 / p1.N ≥ 0)
    : approximateVariance p2 ≤ approximateVariance p1 := by
  unfold approximateVariance
  rw [h_same_N, h_same_n, h_same_σY]
  have h_factor : (1 : ℝ) / p2.n - 1 / p2.N ≥ 0 := by
    simpa [h_same_n, h_same_N] using h_pos_factor
  have h_mul :
      (1 / p2.n - 1 / p2.N) * p2.σ_pred_sq ≤
      (1 / p2.n - 1 / p2.N) * p1.σ_pred_sq :=
    mul_le_mul_of_nonneg_left h_better h_factor
  have h_add := add_le_add_left h_mul (p2.σ_Y_sq / p2.N)
  simpa [add_comm, add_left_comm, add_assoc] using h_add

/-- Perfect predictions give minimum variance.

    When Ŷ = Y (perfect predictions), σ²_pred = 0 and:
    Var(β̂_DSL) = σ²_Y/N

    regardless of n. -/
theorem perfect_predictions_minimum_variance
    (p : DSLVarianceParams) (h_perfect : p.σ_pred_sq = 0)
    : approximateVariance p = p.σ_Y_sq / p.N := by
  unfold approximateVariance
  rw [h_perfect]
  ring

/-!
## Power Analysis
-/

/-- Standard error for a given parameter configuration -/
def standardErrorFromVariance (v : ℝ) : ℝ :=
  Real.sqrt v

/-- z-statistic for hypothesis testing -/
def zStatistic (β_hat β_null SE : ℝ) : ℝ :=
  (β_hat - β_null) / SE

/-- Power function: probability of rejecting null when alternative is true.

    Power = P(|Z| > z_{α/2} | β = β_alt)
          = P(Z > z_{α/2} - (β_alt - β_null)/SE) + P(Z < -z_{α/2} - (β_alt - β_null)/SE)

    For one-sided: Power ≈ Φ((β_alt - β_null)/SE - z_α) -/
def powerApproximate
    (β_alt β_null : ℝ)  -- Alternative and null values
    (SE : ℝ)            -- Standard error
    (z_alpha : ℝ)       -- Critical value
    : ℝ :=
  -- Simplified: non-centrality parameter
  let ncp := (β_alt - β_null) / SE
  -- Power ≈ 1 - Φ(z_alpha - ncp) for one-sided
  -- Placeholder: actual CDF would need Mathlib probability
  if ncp > z_alpha then 0.8 else 0.5  -- Simplified

/-- Required n for given power target.

    Solve for n such that Power(n) ≥ target

    This is the key sample size calculation for DSL studies. -/
def requiredSampleSize
    (β_alt β_null : ℝ)
    (σ_Y_sq σ_pred_sq : ℝ)
    (N : ℕ)
    (target_power : ℝ)
    (z_alpha : ℝ)
    : ℕ :=
  -- Simplified: would need actual inverse calculation
  -- n ≥ σ²_pred / (σ²_Y/z² - ...) roughly
  N / 10  -- Placeholder

/-!
## Comparison with Full Labeling
-/

/-- Relative efficiency of DSL compared to full labeling.

    RE = Var_full / Var_DSL

    When RE > 1, DSL is less efficient (needs more N for same power).
    The efficiency loss depends on n/N and prediction quality. -/
def relativeEfficiency (p : DSLVarianceParams) : ℝ :=
  let var_full := p.σ_Y_sq / p.N
  let var_DSL := approximateVariance p
  var_full / var_DSL

/-- Relative efficiency improves with better predictions.

    As σ²_pred → 0, RE → 1 (DSL as efficient as full labeling). -/
theorem relative_efficiency_improves_with_accuracy
    (p1 p2 : DSLVarianceParams)
    (h_same : p1.N = p2.N ∧ p1.n = p2.n ∧ p1.σ_Y_sq = p2.σ_Y_sq)
    (h_better : p2.σ_pred_sq < p1.σ_pred_sq)
    (h_pos : p1.σ_pred_sq > 0)
    (h_pos2 : p2.σ_pred_sq ≥ 0)
    (h_Y_nonneg : p1.σ_Y_sq ≥ 0)
    (h_var_pos2 : 0 < approximateVariance p2)
    : relativeEfficiency p2 ≥ relativeEfficiency p1 := by
  rcases h_same with ⟨hN, hn, hσY⟩
  have h_pos_factor : (1 : ℝ) / p1.n - 1 / p1.N ≥ 0 := by
    have hpos_n : (0 : ℝ) < p1.n := Nat.cast_pos.mpr p1.h_n_pos
    have hle_n : (p1.n : ℝ) ≤ p1.N := Nat.cast_le.mpr p1.h_n_le_N
    have h_div : (1 : ℝ) / p1.N ≤ 1 / p1.n :=
      one_div_le_one_div_of_le hpos_n hle_n
    linarith
  have h_better' : p2.σ_pred_sq ≤ p1.σ_pred_sq := le_of_lt h_better
  have h_var_le : approximateVariance p2 ≤ approximateVariance p1 :=
    variance_decreases_with_accuracy p1 p2 hN hn hσY h_better' h_pos_factor
  have h_inv : (1 : ℝ) / approximateVariance p1 ≤ 1 / approximateVariance p2 :=
    one_div_le_one_div_of_le h_var_pos2 h_var_le
  have h_num_nonneg : 0 ≤ p1.σ_Y_sq / p1.N := by
    have hN_pos : (0 : ℝ) < p1.N := Nat.cast_pos.mpr p1.h_N_pos
    exact div_nonneg h_Y_nonneg (le_of_lt hN_pos)
  unfold relativeEfficiency
  -- Use monotonicity of multiplication with nonnegative numerator.
  have h_mul : (p1.σ_Y_sq / p1.N) * (1 / approximateVariance p1) ≤
      (p1.σ_Y_sq / p1.N) * (1 / approximateVariance p2) :=
    mul_le_mul_of_nonneg_left h_inv h_num_nonneg
  simpa [hN, hn, hσY, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using h_mul

/-!
## Practical Sample Size Guidelines
-/

/-- Rule of thumb: n ≈ 10% of N is often sufficient.

    When prediction accuracy is high (>90%), using n ≈ 0.1N
    typically gives power comparable to full labeling.

    This is a rough heuristic; actual n depends on:
    - Prediction accuracy
    - Effect size
    - Desired power -/
def ruleOfThumb_n (N : ℕ) : ℕ :=
  max (N / 10) 100  -- At least 100, or 10% of N

/-- Sample size needed for 80% power (rough formula).

    n ≈ (z_{1-α/2} + z_{1-β})² · σ²_pred / δ²

    where δ = β_alt - β_null is the effect size. -/
def sampleSizeForPower80
    (effect_size : ℝ)   -- δ = β_alt - β_null
    (σ_pred_sq : ℝ)     -- Prediction error variance
    : ℝ :=
  let z_alpha := 1.96  -- 5% significance
  let z_beta := 0.84   -- 80% power
  (z_alpha + z_beta)^2 * σ_pred_sq / effect_size^2

/-!
## Variance Bounds
-/

/-- Lower bound on DSL variance.

    Var(β̂_DSL) ≥ σ²_Y/N

    This is achieved in the limit as n → N or σ²_pred → 0. -/
theorem variance_lower_bound (p : DSLVarianceParams) (h_nonneg : p.σ_pred_sq ≥ 0)
    : approximateVariance p ≥ p.σ_Y_sq / p.N := by
  unfold approximateVariance
  have hpos_n : (0 : ℝ) < p.n := Nat.cast_pos.mpr p.h_n_pos
  have hle_n : (p.n : ℝ) ≤ p.N := Nat.cast_le.mpr p.h_n_le_N
  have h_div : (1 : ℝ) / p.N ≤ 1 / p.n :=
    one_div_le_one_div_of_le hpos_n hle_n
  have h_factor : 0 ≤ (1 : ℝ) / p.n - 1 / p.N := by linarith
  have h_mul : 0 ≤ (1 / p.n - 1 / p.N) * p.σ_pred_sq :=
    mul_nonneg h_factor h_nonneg
  linarith

/-- Upper bound on DSL variance (when n is small).

    As n → 1, variance can be large if σ²_pred is large.
    This motivates using reasonable sample sizes. -/
theorem variance_upper_bound_small_n (p : DSLVarianceParams) (h_n_small : p.n = 1)
    : approximateVariance p = p.σ_Y_sq / p.N + (1 - 1 / p.N) * p.σ_pred_sq := by
  unfold approximateVariance
  rw [h_n_small]
  simp

end DSL

end
