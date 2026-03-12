import FormalProofs.DSL.BiasAnalysis
import FormalProofs.DSL.DSLEstimator
import FormalProofs.DSL.IPWMeasurementError

/-!
# FormalProofs/DSL/NonclassicalExpectationMismatch.lean

This file formalizes the stronger intuition behind nonclassical error:

1. the relevant conditional expectation may depend on a richer information set
   than the analyst is using; and
2. if the design-adjusted outcome is formed with a propensity that ignores that
   richer information set, its expectation retains a residual bias term.

So the problem is not only "wrong weights". It is also "wrong sigma-field" /
wrong conditioning information.
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

/-- Observed-only ignorability for a proxy-indexed error family. The analyst is
conditioning only on the observed document state. -/
def ObservedIgnorablePredictionErrorWithProxy
    {Obs Proxy Mis : Type*}
    (Y_pred Y_true : Obs → Proxy → Mis → ℝ)
    (E_obs : Obs → (Mis → ℝ) → ℝ) : Prop :=
  ∀ d_obs proxy,
    E_obs d_obs (fun d_mis => Y_pred d_obs proxy d_mis - Y_true d_obs proxy d_mis) = 0

/-- Proxy-aware ignorability: the mean prediction error vanishes even after
conditioning on the richer information set `(X, proxy)`. -/
def ProxyAwareIgnorablePredictionError
    {Obs Proxy Mis : Type*}
    (Y_pred Y_true : Obs → Proxy → Mis → ℝ)
    (E_proxy : Obs → Proxy → (Mis → ℝ) → ℝ) : Prop :=
  ∀ d_obs proxy,
    E_proxy d_obs proxy (fun d_mis => Y_pred d_obs proxy d_mis - Y_true d_obs proxy d_mis) = 0

/-- The observed-only and proxy-aware information sets disagree about the mean
prediction error on at least one event. This is the precise sense in which some
expectations are "misformed" by the wrong information set. -/
def ProxyExpectationMismatch
    {Obs Proxy Mis : Type*}
    (Y_pred Y_true : Obs → Proxy → Mis → ℝ)
    (E_obs : Obs → (Mis → ℝ) → ℝ)
    (E_proxy : Obs → Proxy → (Mis → ℝ) → ℝ) : Prop :=
  ∃ d_obs proxy,
    E_obs d_obs (fun d_mis => Y_pred d_obs proxy d_mis - Y_true d_obs proxy d_mis) ≠
      E_proxy d_obs proxy (fun d_mis => Y_pred d_obs proxy d_mis - Y_true d_obs proxy d_mis)

/-- If observed-only ignorability holds but proxy-aware ignorability fails, then
the analyst is using the wrong conditional expectation on at least one
conditioning event. -/
theorem proxyExpectationMismatch_of_observedIgnorable_and_not_proxyAwareIgnorable
    {Obs Proxy Mis : Type*}
    {Y_pred Y_true : Obs → Proxy → Mis → ℝ}
    {E_obs : Obs → (Mis → ℝ) → ℝ}
    {E_proxy : Obs → Proxy → (Mis → ℝ) → ℝ}
    (h_obs : ObservedIgnorablePredictionErrorWithProxy Y_pred Y_true E_obs)
    (h_not_proxy : ¬ ProxyAwareIgnorablePredictionError Y_pred Y_true E_proxy) :
    ProxyExpectationMismatch Y_pred Y_true E_obs E_proxy := by
  unfold ProxyAwareIgnorablePredictionError at h_not_proxy
  push_neg at h_not_proxy
  rcases h_not_proxy with ⟨d_obs, proxy, h_proxy⟩
  refine ⟨d_obs, proxy, ?_⟩
  rw [h_obs d_obs proxy]
  intro h_eq
  exact h_proxy h_eq.symm

/-- Under observed-only ignorability, proxy-aware ignorability is equivalent to
there being no expectation mismatch between the observed-only and proxy-aware
information sets. -/
theorem proxyAwareIgnorable_iff_noExpectationMismatch_of_observedIgnorable
    {Obs Proxy Mis : Type*}
    {Y_pred Y_true : Obs → Proxy → Mis → ℝ}
    {E_obs : Obs → (Mis → ℝ) → ℝ}
    {E_proxy : Obs → Proxy → (Mis → ℝ) → ℝ}
    (h_obs : ObservedIgnorablePredictionErrorWithProxy Y_pred Y_true E_obs) :
    ProxyAwareIgnorablePredictionError Y_pred Y_true E_proxy ↔
      ¬ ProxyExpectationMismatch Y_pred Y_true E_obs E_proxy := by
  constructor
  · intro h_proxy
    intro h_mismatch
    rcases h_mismatch with ⟨d_obs, proxy, hneq⟩
    apply hneq
    rw [h_obs d_obs proxy, h_proxy d_obs proxy]
  · intro h_no_mismatch
    intro d_obs proxy
    by_contra hneq
    apply h_no_mismatch
    refine ⟨d_obs, proxy, ?_⟩
    rw [h_obs d_obs proxy]
    intro h_eq
    exact hneq h_eq.symm

/-- Design-adjusted outcomes formed with a user-supplied propensity `π_used`,
while the true sampling expectation is governed by `π_true`, have an exact
residual expectation term. -/
theorem designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
    (Y_pred Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_true R π_used - Y_true) =
      (1 - π_true / π_used) * (Y_pred - Y_true) := by
  have h1 :
      E_cond (fun R => designAdjustedOutcome Y_pred Y_true R π_used - Y_true) =
        E_cond (fun R => (1 - R.toReal / π_used) * (Y_pred - Y_true)) := by
    congr 1
    ext R
    unfold designAdjustedOutcome SamplingIndicator.toReal
    cases R with
    | false => simp
    | true =>
        simp
        ring
  rw [h1]
  calc
    E_cond (fun R => (1 - R.toReal / π_used) * (Y_pred - Y_true))
        = E_cond (fun R => (Y_pred - Y_true) * 1 + (-(Y_pred - Y_true) / π_used) * R.toReal) := by
            congr 1
            ext R
            ring
    _ = (Y_pred - Y_true) * E_cond (fun _ => 1) +
          (-(Y_pred - Y_true) / π_used) * E_cond (fun R => R.toReal) := by
            rw [hE_linear]
    _ = (Y_pred - Y_true) * 1 + (-(Y_pred - Y_true) / π_used) * π_true := by
            rw [hE_1, hE_R]
    _ = (1 - π_true / π_used) * (Y_pred - Y_true) := by
            ring

/-- Correctly using the true propensity removes the residual expectation term. -/
theorem designAdjustedOutcome_unbiased_of_matched_propensity
    (Y_pred Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_match : π_true = π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_true R π_used - Y_true) = 0 := by
  rw [designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
    Y_pred Y_true π_true π_used hπ_used E_cond hE_R hE_1 hE_linear, h_match]
  have hπ_ne : π_used ≠ 0 := ne_of_gt hπ_used
  rw [div_self hπ_ne]
  ring

/-- If the used propensity differs from the true one and the prediction error is
nonzero, the design-adjusted expectation is nonzero. This is the expectation
level statement of "the estimator is misformed". -/
theorem designAdjustedOutcome_expectation_ne_zero_of_mismatch_and_error
    (Y_pred Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_mismatch : π_true ≠ π_used)
    (h_error : Y_pred ≠ Y_true)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_true R π_used - Y_true) ≠ 0 := by
  rw [designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
    Y_pred Y_true π_true π_used hπ_used E_cond hE_R hE_1 hE_linear]
  intro h_zero
  rw [mul_eq_zero] at h_zero
  cases h_zero with
  | inl h_factor =>
      have hπ_ne : π_used ≠ 0 := ne_of_gt hπ_used
      field_simp [hπ_ne] at h_factor
      apply h_mismatch
      linarith
  | inr h_err =>
      apply h_error
      linarith

/-- Full scalar decomposition: if the design-adjusted outcome is formed using a
possibly noisy oracle label `Y_oracle` and a misspecified propensity `π_used`,
the expectation splits into a sampling-law residual plus an oracle measurement
error term. -/
theorem designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
    (Y_pred Y_oracle Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_true) =
      (1 - π_true / π_used) * (Y_pred - Y_oracle) + (Y_oracle - Y_true) := by
  calc
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_true)
        = E_cond (fun R =>
            1 * (designAdjustedOutcome Y_pred Y_oracle R π_used - Y_oracle) +
              (Y_oracle - Y_true) * 1) := by
              congr 1
              ext R
              ring
    _ = 1 * E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_oracle) +
          (Y_oracle - Y_true) * E_cond (fun _ => 1) := by
            rw [hE_linear]
    _ = 1 * ((1 - π_true / π_used) * (Y_pred - Y_oracle)) +
          (Y_oracle - Y_true) * 1 := by
            rw [designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
              Y_pred Y_oracle π_true π_used hπ_used E_cond hE_R hE_1 hE_linear, hE_1]
    _ = (1 - π_true / π_used) * (Y_pred - Y_oracle) + (Y_oracle - Y_true) := by
            ring

/-- Exact-oracle special case of
`designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError`. -/
theorem designAdjustedOutcome_expectation_eq_residual_of_exactOracle
    (Y_pred Y_oracle Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_oracle_exact : Y_oracle = Y_true)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_true) =
      (1 - π_true / π_used) * (Y_pred - Y_true) := by
  rw [designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
    Y_pred Y_oracle Y_true π_true π_used hπ_used E_cond hE_R hE_1 hE_linear, h_oracle_exact]
  ring

/-- If the sampling propensity is adjusted correctly, the only remaining scalar
expectation error is the oracle measurement error term. -/
theorem designAdjustedOutcome_expectation_eq_oracleMeasurementError_of_adjusted
    (Y_pred Y_oracle Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_match : π_true = π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_true) =
      Y_oracle - Y_true := by
  rw [designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
    Y_pred Y_oracle Y_true π_true π_used hπ_used E_cond hE_R hE_1 hE_linear, h_match]
  have hπ_ne : π_used ≠ 0 := ne_of_gt hπ_used
  rw [div_self hπ_ne]
  ring

/-- If both the propensity is adjusted correctly and the oracle label is exact,
the scalar design-adjusted expectation is unbiased. -/
theorem designAdjustedOutcome_unbiased_of_adjusted_and_exactOracle
    (Y_pred Y_oracle Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_match : π_true = π_used)
    (h_oracle_exact : Y_oracle = Y_true)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_true) = 0 := by
  rw [designAdjustedOutcome_expectation_eq_oracleMeasurementError_of_adjusted
    Y_pred Y_oracle Y_true π_true π_used hπ_used h_match E_cond hE_R hE_1 hE_linear, h_oracle_exact]
  ring

/-- Absolute-value version of the full scalar decomposition. -/
theorem abs_designAdjustedOutcome_expectation_le_residual_plus_oracleMeasurementError
    (Y_pred Y_oracle Y_true π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    |E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R π_used - Y_true)| ≤
      |(1 - π_true / π_used) * (Y_pred - Y_oracle)| + |Y_oracle - Y_true| := by
  rw [designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
    Y_pred Y_oracle Y_true π_true π_used hπ_used E_cond hE_R hE_1 hE_linear]
  exact abs_add_le _ _

/-- Proxy-aware version: the residual expectation term is indexed by the actual
proxy-aware propensity, not the observed-only propensity the analyst might have
plugged in. -/
theorem proxyAware_designAdjustedOutcome_expectation_eq_residual
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_true R (pi_used.π d_obs q) - Y_true) =
      (1 - pi_actual.π d_obs (proxy d_obs) q / pi_used.π d_obs q) * (Y_pred - Y_true) := by
  exact designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
    Y_pred Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) E_cond hE_R hE_1 hE_linear

/-- If the propensity plugged into the design-adjusted outcome matches the
actual proxy-aware design, the expectation is unbiased even without any
classical-error assumption. -/
theorem proxyAware_designAdjustedOutcome_unbiased_of_adjusted
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (h_adjusted : pi_used.π d_obs q = pi_actual.π d_obs (proxy d_obs) q)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_true R (pi_used.π d_obs q) - Y_true) = 0 := by
  exact designAdjustedOutcome_unbiased_of_matched_propensity
    Y_pred Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) h_adjusted.symm
    E_cond hE_R hE_1 hE_linear

/-- If the actual proxy-aware design differs from the propensity used by the
analyst, and prediction error is nonzero, then the design-adjusted expectation
is nonzero on that conditioning event. -/
theorem proxyAware_designAdjustedOutcome_expectation_ne_zero_of_mismatch_and_error
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (h_mismatch : pi_actual.π d_obs (proxy d_obs) q ≠ pi_used.π d_obs q)
    (h_error : Y_pred ≠ Y_true)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_true R (pi_used.π d_obs q) - Y_true) ≠ 0 := by
  exact designAdjustedOutcome_expectation_ne_zero_of_mismatch_and_error
    Y_pred Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) h_mismatch h_error
    E_cond hE_R hE_1 hE_linear

/-- Proxy-aware full scalar decomposition with a noisy oracle label. -/
theorem proxyAware_designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_oracle Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R (pi_used.π d_obs q) - Y_true) =
      (1 - pi_actual.π d_obs (proxy d_obs) q / pi_used.π d_obs q) * (Y_pred - Y_oracle) +
        (Y_oracle - Y_true) := by
  exact designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
    Y_pred Y_oracle Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) E_cond hE_R hE_1 hE_linear

/-- Proxy-aware exact-oracle corollary. -/
theorem proxyAware_designAdjustedOutcome_expectation_eq_residual_of_exactOracle
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_oracle Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (h_oracle_exact : Y_oracle = Y_true)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R (pi_used.π d_obs q) - Y_true) =
      (1 - pi_actual.π d_obs (proxy d_obs) q / pi_used.π d_obs q) * (Y_pred - Y_true) := by
  exact designAdjustedOutcome_expectation_eq_residual_of_exactOracle
    Y_pred Y_oracle Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) h_oracle_exact
    E_cond hE_R hE_1 hE_linear

/-- Proxy-aware adjusted-propensity corollary: the remaining scalar error is
exactly the oracle measurement error term. -/
theorem proxyAware_designAdjustedOutcome_expectation_eq_oracleMeasurementError_of_adjusted
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_oracle Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (h_adjusted : pi_used.π d_obs q = pi_actual.π d_obs (proxy d_obs) q)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R (pi_used.π d_obs q) - Y_true) =
      Y_oracle - Y_true := by
  exact designAdjustedOutcome_expectation_eq_oracleMeasurementError_of_adjusted
    Y_pred Y_oracle Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) h_adjusted.symm
    E_cond hE_R hE_1 hE_linear

/-- Proxy-aware adjusted-propensity + exact-oracle corollary: unbiasedness is
fully restored. -/
theorem proxyAware_designAdjustedOutcome_unbiased_of_adjusted_and_exactOracle
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (Y_pred Y_oracle Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (h_adjusted : pi_used.π d_obs q = pi_actual.π d_obs (proxy d_obs) q)
    (h_oracle_exact : Y_oracle = Y_true)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = pi_actual.π d_obs (proxy d_obs) q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => designAdjustedOutcome Y_pred Y_oracle R (pi_used.π d_obs q) - Y_true) = 0 := by
  exact designAdjustedOutcome_unbiased_of_adjusted_and_exactOracle
    Y_pred Y_oracle Y_true (pi_actual.π d_obs (proxy d_obs) q) (pi_used.π d_obs q)
    (pi_used.positivity d_obs q) h_adjusted.symm h_oracle_exact
    E_cond hE_R hE_1 hE_linear

/-- Full moment-level decomposition: misspecified proxy-aware sampling plus
oracle measurement error. -/
theorem DSLMoment_expectation_eq_residual_plus_oracleMeasurementError
    {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_oracle d_mis_true : Mis)
    (β : Fin d → ℝ) (i : Fin d)
    (π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R =>
      DSLMoment m d_obs d_mis_pred d_mis_oracle R π_used β i -
        m (d_obs, d_mis_true) β i) =
      (1 - π_true / π_used) *
        (m (d_obs, d_mis_pred) β i - m (d_obs, d_mis_oracle) β i) +
      (m (d_obs, d_mis_oracle) β i - m (d_obs, d_mis_true) β i) := by
  simpa [DSLMoment] using
    (designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError
      (Y_pred := m (d_obs, d_mis_pred) β i)
      (Y_oracle := m (d_obs, d_mis_oracle) β i)
      (Y_true := m (d_obs, d_mis_true) β i)
      (π_true := π_true) (π_used := π_used)
      hπ_used E_cond hE_R hE_1 hE_linear)

/-- Exact-oracle moment corollary. -/
theorem DSLMoment_expectation_eq_residual_of_exactOracle
    {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_true : Mis)
    (β : Fin d → ℝ) (i : Fin d)
    (π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R =>
      DSLMoment m d_obs d_mis_pred d_mis_true R π_used β i -
        m (d_obs, d_mis_true) β i) =
      (1 - π_true / π_used) *
        (m (d_obs, d_mis_pred) β i - m (d_obs, d_mis_true) β i) := by
  simpa [DSLMoment] using
    (designAdjustedOutcome_expectation_eq_residual_of_exactOracle
      (Y_pred := m (d_obs, d_mis_pred) β i)
      (Y_oracle := m (d_obs, d_mis_true) β i)
      (Y_true := m (d_obs, d_mis_true) β i)
      (π_true := π_true) (π_used := π_used)
      hπ_used rfl E_cond hE_R hE_1 hE_linear)

/-- If the propensity is adjusted correctly, the remaining moment expectation is
exactly the oracle measurement error term. -/
theorem DSLMoment_expectation_eq_oracleMeasurementError_of_adjusted
    {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_oracle d_mis_true : Mis)
    (β : Fin d → ℝ) (i : Fin d)
    (π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_match : π_true = π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R =>
      DSLMoment m d_obs d_mis_pred d_mis_oracle R π_used β i -
        m (d_obs, d_mis_true) β i) =
      m (d_obs, d_mis_oracle) β i - m (d_obs, d_mis_true) β i := by
  simpa [DSLMoment] using
    (designAdjustedOutcome_expectation_eq_oracleMeasurementError_of_adjusted
      (Y_pred := m (d_obs, d_mis_pred) β i)
      (Y_oracle := m (d_obs, d_mis_oracle) β i)
      (Y_true := m (d_obs, d_mis_true) β i)
      (π_true := π_true) (π_used := π_used)
      hπ_used h_match E_cond hE_R hE_1 hE_linear)

/-- With correct propensity and exact oracle labels, the moment equation is
unbiased. -/
theorem DSLMoment_unbiased_of_adjusted_and_exactOracle
    {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_true : Mis)
    (β : Fin d → ℝ) (i : Fin d)
    (π_true π_used : ℝ)
    (hπ_used : 0 < π_used)
    (h_match : π_true = π_used)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = π_true)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R =>
      DSLMoment m d_obs d_mis_pred d_mis_true R π_used β i -
        m (d_obs, d_mis_true) β i) = 0 := by
  simpa [DSLMoment] using
    (designAdjustedOutcome_unbiased_of_adjusted_and_exactOracle
      (Y_pred := m (d_obs, d_mis_pred) β i)
      (Y_oracle := m (d_obs, d_mis_true) β i)
      (Y_true := m (d_obs, d_mis_true) β i)
      (π_true := π_true) (π_used := π_used)
      hπ_used h_match rfl E_cond hE_R hE_1 hE_linear)

end DSL
