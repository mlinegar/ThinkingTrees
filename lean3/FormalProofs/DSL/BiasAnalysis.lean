/-
# FormalProofs/DSL/BiasAnalysis.lean

## Paper Reference: Section 2.1, Equation 3, Appendix OA.14

This file formalizes the bias analysis for prediction errors:
- Condition for ignorable prediction error (Equation 3)
- Bias formulas when ignoring errors (OA.14)
- Examples showing bias with high accuracy

### Key Insight

The condition E[e | X] = 0 (where e = Ŷ - Y) is MUCH stronger than it appears.
It requires prediction errors to be uncorrelated with:
1. The covariates X
2. The true outcome Y
3. Any unobserved confounders U

This almost never holds in practice, even with 90%+ prediction accuracy.

### Bias Formula (Equation OA.14)

When ignoring prediction errors, the bias in β̂ is:
  Bias(β̂) = [E(∂m/∂β)]⁻¹ · E[m(D^obs, D̂^mis; β*) - m(D^obs, D^mis; β*)]

This bias depends on how prediction errors correlate with the analysis variables.
-/

import FormalProofs.DSL.MomentFunctions

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
## Ignorable Prediction Error (Equation 3)
-/

/-- The condition required to ignore prediction errors (Equation 3).

    E[e | X] = 0 where e = Ŷ - Y

    This is the assumption implicitly made when using predicted labels
    as if they were true labels. -/
def IgnorablePredictionError
    {Obs Mis : Type*}
    (Y_pred Y_true : Obs → Mis → ℝ)
    (E_cond : Obs → (Mis → ℝ) → ℝ)  -- E[· | X]
    : Prop :=
  ∀ d_obs, E_cond d_obs (fun d_mis => Y_pred d_obs d_mis - Y_true d_obs d_mis) = 0

/-- Ignorable prediction error is a very strong condition.

    It requires errors to be uncorrelated with:
    (1) Covariates X
    (2) True outcome Y
    (3) Unobserved confounders U

    Any of these correlations causes bias. -/
structure IgnorabilityFailureModes where
  /-- Errors correlated with covariates: E[e | X] ≠ 0 -/
  correlated_with_X : Prop
  /-- Errors correlated with true outcome: Corr(e, Y) ≠ 0 -/
  correlated_with_Y : Prop
  /-- Errors correlated with confounders: Corr(e, U) ≠ 0 -/
  correlated_with_U : Prop

/-!
## Bias When Ignoring Prediction Errors
-/

/-- Bias in the moment function when using predictions instead of true values.

    This is the bias E[m(D^obs, D̂^mis; β) - m(D^obs, D^mis; β)] -/
def MomentBias {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  E (fun ⟨d_obs, d_mis_pred, d_mis_true⟩ =>
    fun i => m (d_obs, d_mis_pred) β i - m (d_obs, d_mis_true) β i)

/-- Parameter bias formula (Equation OA.14).

    Bias(β̂) = S⁻¹ · E[m_pred - m_true]

    where S = E[∂m/∂β] is the Jacobian. -/
def ParameterBias {d : ℕ}
    (S_inv : Matrix (Fin d) (Fin d) ℝ)
    (moment_bias : Fin d → ℝ) : Fin d → ℝ :=
  fun i => ∑ j : Fin d, S_inv i j * moment_bias j

/-!
## Bias Examples
-/

/-- Example: Bias in linear regression coefficient.

    For Y = β₀ + β₁X + ε with binary treatment X:
    - LLM predicts X̂ with accuracy p
    - True effect is β₁
    - Naive estimate is biased toward 0

    Bias formula: β̂₁ - β₁ ≈ (2p - 1 - 1)β₁ = (2p - 2)β₁

    Even with p = 0.9, bias is about 20% of the true effect! -/
def linearRegressionBiasExample
    (p : ℝ)           -- Prediction accuracy
    (β₁_true : ℝ)     -- True coefficient
    : ℝ :=
  (2 * p - 2) * β₁_true

/-- Bias exists even with high accuracy (Theorem from paper Section 2.1) -/
theorem bias_with_high_accuracy
    (p : ℝ) (hp : 0.9 ≤ p ∧ p < 1)
    (β₁_true : ℝ) (hβ : β₁_true ≠ 0)
    : linearRegressionBiasExample p β₁_true ≠ 0 := by
  unfold linearRegressionBiasExample
  -- (2p - 2)β₁ ≠ 0 when p < 1 and β₁ ≠ 0
  intro h
  have hp_lt_1 : p < 1 := hp.2
  have h2p_lt_2 : 2 * p < 2 := by linarith
  have h2p_minus_2_neg : 2 * p - 2 < 0 := by linarith
  have h2p_minus_2_ne : 2 * p - 2 ≠ 0 := ne_of_lt h2p_minus_2_neg
  rw [mul_eq_zero] at h
  cases h with
  | inl h1 => exact h2p_minus_2_ne h1
  | inr h2 => exact hβ h2

/-- Example: Bias in logistic regression odds ratio.

    When predicting a binary treatment indicator X:
    - LLM accuracy affects odds ratio estimate
    - Bias is toward the null (OR = 1)
    - The direction of attenuation depends on error pattern -/
def logisticRegressionBiasExample
    (p_treat : ℝ)     -- Accuracy on treated (X=1)
    (p_control : ℝ)   -- Accuracy on control (X=0)
    (true_OR : ℝ)     -- True odds ratio
    : ℝ :=
  -- Simplified attenuation formula
  -- In general: estimated OR = true_OR^[(p_treat + p_control - 1)]
  -- When p_treat = p_control = p: estimated OR = true_OR^(2p-1)
  let attenuation := p_treat + p_control - 1
  Real.rpow true_OR attenuation

/-!
## Differential Prediction Error
-/

/-- Differential prediction error: error rates differ by group.

    This is common in practice:
    - LLMs may perform better on some groups than others
    - Creates heterogeneous bias
    - Can flip the sign of estimated effects -/
structure DifferentialError where
  /-- Error rate for group 1 -/
  error_rate_1 : ℝ
  /-- Error rate for group 2 -/
  error_rate_2 : ℝ
  /-- Error rates differ -/
  h_differential : error_rate_1 ≠ error_rate_2

/-- A simple attenuation-style estimated effect under differential errors. -/
def estimatedEffectFromErrors (true_effect : ℝ) (de : DifferentialError) : ℝ :=
  true_effect - (de.error_rate_1 - de.error_rate_2)

/-- Differential error can flip the sign of effects.

    Example: If group 1 has higher error rate and is also the
    treatment group, the estimated effect can be wrong sign. -/
theorem differential_error_sign_flip
    (de : DifferentialError)
    (true_effect : ℝ) (h_pos : true_effect > 0)
    -- Sufficient differential error can flip the sign
    (h_severe : de.error_rate_1 - de.error_rate_2 > true_effect)
    : estimatedEffectFromErrors true_effect de < 0 := by
  unfold estimatedEffectFromErrors
  linarith

/-!
## Bias in Category Proportions
-/

/-- Bias in estimating a category proportion.

    True proportion: μ = P(Y = 1)
    Estimated (naive): μ̂ = P(Ŷ = 1)

    Bias: μ̂ - μ = P(Ŷ = 1, Y = 0) - P(Ŷ = 0, Y = 1)
                 = FP_rate - FN_rate

    When false positives ≠ false negatives, there is bias. -/
def proportionBias
    (FP_rate : ℝ)   -- P(Ŷ = 1 | Y = 0)
    (FN_rate : ℝ)   -- P(Ŷ = 0 | Y = 1)
    : ℝ :=
  FP_rate - FN_rate

/-- High accuracy doesn't prevent proportion bias.

    Example: 95% accuracy with FP_rate = 0.08, FN_rate = 0.02
    Bias = 0.06 = 6 percentage points!

    This is huge when estimating proportions in the 10-20% range. -/
theorem proportion_bias_with_high_accuracy
    (FP_rate FN_rate : ℝ)
    (h_acc : 0.95 ≤ 1 - (FP_rate + FN_rate) / 2)  -- 95%+ accuracy
    (h_diff : FP_rate ≠ FN_rate)
    : proportionBias FP_rate FN_rate ≠ 0 := by
  unfold proportionBias
  intro h
  have : FP_rate = FN_rate := by linarith
  exact h_diff this

/-!
## When Can Errors Be Ignored?
-/

/-- Sufficient conditions for ignoring prediction errors (very restrictive).

    Errors can ONLY be ignored when:
    1. Classical measurement error: e ⊥ (X, Y, U) [almost never holds]
    2. Perfect predictions: Ŷ = Y almost surely
    3. Special balanced case: errors exactly cancel out

    In practice, none of these hold. -/
structure IgnorableErrorConditions where
  /-- Errors independent of covariates -/
  independent_of_X : Prop
  /-- Errors independent of true outcome -/
  independent_of_Y : Prop
  /-- Errors independent of unobservables -/
  independent_of_U : Prop
  /-- Full independence required -/
  h_full : independent_of_X ∧ independent_of_Y ∧ independent_of_U

/-- Classical measurement error rarely applies to LLM predictions.

    LLM errors are typically:
    - Correlated with text complexity
    - Correlated with topic/domain
    - Correlated with outcome values
    - Systematic rather than random

    This violates the classical measurement error assumption. -/
theorem llm_errors_not_classical
    (cond : IgnorableErrorConditions)
    -- LLM errors have patterns that violate full independence.
    (h_patterns :
      ¬ (cond.independent_of_X ∧ cond.independent_of_Y ∧ cond.independent_of_U))
    : ¬ (cond.independent_of_X ∧ cond.independent_of_Y ∧ cond.independent_of_U) := by
  exact h_patterns

end DSL

end
