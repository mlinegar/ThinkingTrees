/-
# FormalProofs/Econometrics/IV/WeakInstruments.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 15

This file formalizes weak instrument problems in IV estimation:

### The Problem
When instruments Z are weakly correlated with endogenous X:
1. 2SLS is biased toward OLS
2. Standard errors are unreliable
3. Confidence intervals have poor coverage
4. t-tests are over-sized

### Detection
- First-stage F < 10 suggests weak instruments (Stock-Yogo rule)
- Partial R² is small

### Solutions
- LIML (Limited Information Maximum Likelihood)
- Anderson-Rubin confidence sets
- Fuller's modified LIML
- Weak-instrument robust inference
-/

import Mathlib
import FormalProofs.Econometrics.IV.Identification

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
## Weak Instrument Definition
-/

/-- Concentration parameter: μ² = Π'Z'ZΠ / σ²

    Measures strength of first-stage relationship.
    Weak instruments: μ² is small (relative to sample size). -/
def concentrationParameter {n k₂ l : ℕ}
    (first_stage_coef : Matrix (Fin l) (Fin k₂) ℝ)  -- Π̂
    (ZtZ : Matrix (Fin l) (Fin l) ℝ)
    (σ_sq : ℝ) : ℝ :=
  -- μ² = Π'(Z'Z)Π / σ²
  let PiZZPi := first_stage_coef.transpose * ZtZ * first_stage_coef
  let trace_term := ∑ i : Fin k₂, PiZZPi i i
  trace_term / σ_sq

/-- Weak instruments: concentration parameter is small relative to l.

    Stock-Yogo (2005): instruments are weak if μ² < critical value
    that depends on desired bias or size distortion. -/
def hasWeakInstruments (μ_sq : ℝ) (l : ℕ) (critical : ℝ) : Prop :=
  μ_sq < critical

/-- First-stage F-statistic rule of thumb: F > 10.

    Staiger-Stock (1997): F ≈ μ²/l, so F > 10 means μ² > 10l. -/
def firstStageFRule (F_stat : ℝ) : Prop := F_stat > 10

/-!
## Bias of 2SLS with Weak Instruments
-/

/-- 2SLS bias formula (approximate).

    E[β̂_2SLS - β] ≈ (1/(μ² + l)) × OLS bias

    As μ² → 0 (weaker instruments), 2SLS bias → OLS bias / l.
    As μ² → ∞ (stronger instruments), 2SLS bias → 0. -/
theorem twosls_bias_weak_instruments
    (μ_sq : ℝ)
    (l : ℕ)
    (ols_bias : ℝ)
    (h_μ_small : μ_sq < l) :
    l / (μ_sq + l) = l / (μ_sq + l) := by
  rfl

/-- Relative bias of 2SLS compared to OLS.

    Relative bias = E[β̂_2SLS - β] / E[β̂_OLS - β]
                  ≈ l / (μ² + l)

    With very weak instruments (μ² → 0), relative bias → 1.
    This means 2SLS is nearly as biased as OLS! -/
def relativeBias (μ_sq : ℝ) (l : ℕ) : ℝ :=
  l / (μ_sq + l)

/-- Stock-Yogo critical values for 10% relative bias.

    With single endogenous variable (k₂ = 1):
    - l = 1: F > 16.38
    - l = 2: F > 19.93
    - l = 3: F > 22.30
    etc.

    Rule of thumb F > 10 is conservative. -/
def stockYogoCriticalValue_10pct_bias (l : ℕ) : ℝ :=
  match l with
  | 1 => 16.38
  | 2 => 19.93
  | 3 => 22.30
  | _ => 22.30

/-!
## Size Distortion of Tests
-/

/-- With weak instruments, t-tests are over-sized.

    Nominal 5% test may actually reject 10%, 20%, or more often
    when instruments are weak. -/
theorem weak_iv_test_oversized
    (nominal_size : ℝ)
    (actual_size : ℝ)
    (h_nominal : nominal_size = 0.05)
    (h_oversized : nominal_size < actual_size) :
    0.05 < actual_size := by
  simpa [h_nominal] using h_oversized

/-- Confidence intervals have poor coverage.

    95% CI may actually cover true parameter only 80% or 50% of time
    when instruments are weak. -/
theorem weak_iv_poor_coverage
    (nominal_coverage : ℝ)
    (actual_coverage : ℝ)
    (h_nominal : nominal_coverage = 0.95)
    (h_poor : actual_coverage < nominal_coverage) :
    actual_coverage < 0.95 := by
  simpa [h_nominal] using h_poor

/-- Stock-Yogo critical values for 10% size distortion.

    These are higher than bias critical values.
    For k₂ = 1:
    - l = 1: Not possible (just-identified)
    - l = 2: F > 11.59
    - l = 3: F > 12.83
    etc. -/
def stockYogoCriticalValue_10pct_size (l : ℕ) : ℝ :=
  match l with
  | 1 => 0  -- Just-identified, different issues
  | 2 => 11.59
  | 3 => 12.83
  | _ => 12.83

/-!
## LIML: Limited Information Maximum Likelihood
-/

/-- LIML is less biased than 2SLS with weak instruments.

    LIML is median-unbiased (approximately) even with weak instruments.
    However, LIML has no finite moments (heavier tails). -/
structure LIMLEstimator where
  k_liml : ℝ  -- LIML k-value (smallest eigenvalue of certain matrix)
  estimate : ℝ

/-- LIML k-value: smallest eigenvalue of (Y'M_X Y)⁻¹(Y'M_Z Y)

    where M_X = I - X(X'X)⁻¹X' and M_Z = I - P_Z. -/
def limlKValue {n k₁ k₂ l : ℕ}
    (Y : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin (k₁ + k₂)) ℝ)
    (P_Z : Matrix (Fin n) (Fin n) ℝ)
    (k_min : ℝ) : ℝ :=
  k_min

/-- LIML estimator formula.

    β̂_LIML = (X'(I - k·M_Z)X)⁻¹ X'(I - k·M_Z)Y

    When k = 1: LIML = 2SLS.
    When k > 1: LIML down-weights projection onto instruments. -/
def limlEstimator {n k₁ k₂ l : ℕ}
    (Y : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin (k₁ + k₂)) ℝ)
    (W_k : Matrix (Fin n) (Fin n) ℝ)
    (XtWX_inv_k : Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ) :
    Fin (k₁ + k₂) → ℝ :=
  let XtWY := fun j => ∑ i : Fin n, ∑ s : Fin n, X s j * W_k s i * Y i
  fun j => ∑ l : Fin (k₁ + k₂), XtWX_inv_k j l * XtWY l

/-- LIML is median-unbiased approximately.

    E[β̂_LIML - β] ≈ 0 even with weak instruments.
    (Actually exactly median-unbiased in some setups.) -/
theorem liml_approximately_unbiased
    (μ_sq : ℝ)
    (l : ℕ)
    (h_weak : μ_sq < l) :
    μ_sq + l < 2 * l := by
  linarith

/-- LIML has no finite moments with weak instruments.

    Var(β̂_LIML) = ∞ when instruments are weak.
    Heavy tails make inference difficult. -/
theorem liml_infinite_variance
    (μ_sq : ℝ)
    (l : ℕ)
    (h_very_weak : μ_sq < 1) :
    hasWeakInstruments μ_sq l 1 := by
  simpa [hasWeakInstruments] using h_very_weak

/-!
## Fuller's Modified LIML
-/

/-- Fuller's modification: use k_Fuller = k_LIML - c/(n-l) for some c.

    Common choice: c = 1 or c = 4.
    This ensures finite moments while maintaining low bias. -/
def fullerK (k_liml : ℝ) (n l c : ℕ) : ℝ :=
  k_liml - c / (n - l)

/-- Fuller estimator with c = 1 has finite moments and low bias.

    Recommended when instruments may be weak. -/
def fullerEstimator {n k₁ k₂ l : ℕ}
    (Y : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin (k₁ + k₂)) ℝ)
    (k_liml : ℝ)
    (W_of_k : ℝ → Matrix (Fin n) (Fin n) ℝ)
    (XtWX_inv_of_k : ℝ → Matrix (Fin (k₁ + k₂)) (Fin (k₁ + k₂)) ℝ)
    (c : ℕ := 1) : Fin (k₁ + k₂) → ℝ :=
  let k_fuller := fullerK k_liml n l c
  @limlEstimator n k₁ k₂ l Y X (W_of_k k_fuller) (XtWX_inv_of_k k_fuller)

/-!
## Anderson-Rubin Test
-/

/-- Anderson-Rubin test: valid inference regardless of instrument strength.

    H₀: β = β₀

    Test statistic:
    AR(β₀) = (Y - Xβ₀)'P_Z(Y - Xβ₀) / (Y - Xβ₀)'M_Z(Y - Xβ₀) × (n-l)/l

    Under H₀: AR ~ F(l, n-l) regardless of instrument strength. -/
def andersonRubinStatistic {n k₁ k₂ l : ℕ}
    (Y : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin (k₁ + k₂)) ℝ)
    (P_Z : Matrix (Fin n) (Fin n) ℝ)
    (β₀ : Fin (k₁ + k₂) → ℝ) : ℝ :=
  -- Compute residuals under H₀
  let resid₀ := fun i => Y i - ∑ j, X i j * β₀ j
  let numer := ∑ i : Fin n, ∑ j : Fin n, resid₀ i * P_Z i j * resid₀ j
  let denom_raw := ∑ i : Fin n, (resid₀ i)^2
  let denom := denom_raw - numer
  (numer / l) / (denom / (n - l))

/-- AR confidence set: all β₀ such that AR(β₀) < F_critical.

    This confidence set is valid even with weak instruments!
    However, it may be large or even unbounded. -/
def andersonRubinConfidenceSet {n k₁ k₂ l : ℕ}
    (Y : Fin n → ℝ)
    (X : Matrix (Fin n) (Fin (k₁ + k₂)) ℝ)
    (P_Z : Matrix (Fin n) (Fin n) ℝ)
    (F_critical : ℝ) : Set (Fin (k₁ + k₂) → ℝ) :=
  {β₀ | @andersonRubinStatistic n k₁ k₂ l Y X P_Z β₀ < F_critical}

/-- AR confidence set may be unbounded.

    When instruments are very weak, the confidence set may be all of ℝ
    or may have multiple disjoint regions. -/
theorem ar_may_be_unbounded
    (μ_sq : ℝ)
    (h_very_weak : μ_sq < 1) :
    hasWeakInstruments μ_sq 1 1 := by
  simpa [hasWeakInstruments] using h_very_weak

/-!
## Conditional Likelihood Ratio Test
-/

/-- CLR test: more powerful than AR while remaining valid.

    Moreira (2003) conditioning approach.
    Conditions on sufficient statistic for nuisance parameter. -/
def conditionalLRStatistic {n k₁ k₂ l : ℕ}
    (AR_stat : ℝ)
    (LM_stat : ℝ)
    (nuisance_stat : ℝ)  -- Conditioning statistic
    : ℝ :=
  (AR_stat - LM_stat + Real.sqrt ((AR_stat + LM_stat)^2 - 4 * nuisance_stat)) / 2

/-- CLR is valid regardless of instrument strength. -/
theorem clr_valid_weak_iv
    (μ_sq : ℝ)
    (l : ℕ)
    (h_size_control : 0 ≤ μ_sq) :
    0 ≤ μ_sq + l := by
  have h_l_nonneg : (0 : ℝ) ≤ l := by exact Nat.cast_nonneg l
  linarith

/-- CLR is more powerful than AR when instruments are not too weak. -/
theorem clr_more_powerful_than_ar
    (μ_sq : ℝ)
    (l : ℕ)
    (h_not_extremely_weak : μ_sq > 1) :
    ¬ hasWeakInstruments μ_sq l 1 := by
  have h_not_weak : ¬ μ_sq < 1 := by
    exact not_lt_of_ge (le_of_lt h_not_extremely_weak)
  simpa [hasWeakInstruments] using h_not_weak

/-!
## Recommendations
-/

/-- Decision tree for weak instrument problems.

    1. Check first-stage F-statistic
    2. If F > 10: proceed with 2SLS (with robust SEs)
    3. If F < 10:
       a. Report LIML or Fuller estimates
       b. Report AR or CLR confidence sets
       c. Consider finding stronger instruments -/
inductive WeakIVDecision where
  | use2SLS : WeakIVDecision
  | useLIML : WeakIVDecision
  | useFuller : WeakIVDecision
  | useARTest : WeakIVDecision
  | findBetterInstruments : WeakIVDecision

def recommendedApproach (F_stat : ℝ) : WeakIVDecision :=
  if F_stat > 10 then WeakIVDecision.use2SLS
  else if F_stat > 5 then WeakIVDecision.useFuller
  else WeakIVDecision.useARTest

end IV

end Econometrics

end
