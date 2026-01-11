/-
FormalProofs/AuditCore.lean

Shared confidence margin and sample complexity definitions for audit proofs.
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Confidence Margin and Sample Complexity

The confidence margin from Hoeffding's inequality for n iid samples
from a distribution with values in [0,1]:

  P(|p_hat - p| >= eps) <= 2 * exp(-2 * n * eps^2)

Solving for eps given delta = 2 * exp(-2 * n * eps^2):
  eps = sqrt(ln(2/delta) / (2 * n))
-/

/-- Confidence margin from Hoeffding: sqrt(ln(2/delta) / (2 * n))

Given n samples and confidence parameter delta, this is the margin eps
such that P(|p_hat - p| >= eps) <= delta for Hoeffding's inequality. -/
def confidence_margin (delta : Real) (n : Nat) : Real :=
  Real.sqrt (Real.log (2 / delta) / (2 * n))

/-- Sample complexity for (eps, delta)-guarantee: ceil(ln(2/delta) / (2 * eps^2))

The minimum number of samples needed to achieve margin eps
with confidence 1 - delta. -/
def sample_complexity (eps delta : Real) : Nat :=
  Nat.ceil (Real.log (2 / delta) / (2 * eps^2))

/-- Confidence margin is non-negative when parameters are valid -/
lemma confidence_margin_nonneg (delta : Real) (n : Nat) (h_delta : 0 < delta) (h_delta' : delta < 2)
    (h_n : 0 < n) : 0 <= confidence_margin delta n := by
  unfold confidence_margin
  apply Real.sqrt_nonneg

/- Sample complexity gives required margin.

When n >= sample_complexity(eps, delta), the confidence margin is at most eps.
This is a direct consequence of the definition: sample_complexity inverts
the margin formula. -/
lemma sample_complexity_gives_margin (eps delta : Real) (h_eps : 0 < eps) (h_delta : 0 < delta)
    (h_delta' : delta < 2) (n : Nat) (h_n : n >= sample_complexity eps delta) :
    confidence_margin delta n <= eps := by
  unfold confidence_margin sample_complexity at *
  -- n >= ceil(ln(2/delta)/(2 * eps^2)) implies ln(2/delta)/(2n) <= eps^2
  have h_log_pos : 0 < Real.log (2 / delta) := by
    apply Real.log_pos
    rw [one_lt_div h_delta]
    linarith
  have h_ratio_pos : 0 < Real.log (2 / delta) / (2 * eps^2) := by positivity
  have h_n_pos : 0 < n := Nat.lt_of_lt_of_le (Nat.ceil_pos.mpr h_ratio_pos) h_n
  have h_n_ge : Real.log (2 / delta) / (2 * eps^2) <= n := by
    have h_ceil :
        Real.log (2 / delta) / (2 * eps^2) <=
          (Nat.ceil (Real.log (2 / delta) / (2 * eps^2)) : Real) := by
      simpa using (Nat.le_ceil (Real.log (2 / delta) / (2 * eps^2)))
    have h_ceil_le : (Nat.ceil (Real.log (2 / delta) / (2 * eps^2)) : Real) <= n := by
      exact_mod_cast h_n
    exact le_trans h_ceil h_ceil_le
  -- From h_n_ge, derive ln(2/delta) <= 2 * n * eps^2
  have h_ineq : Real.log (2 / delta) <= 2 * n * eps^2 := by
    have h2eps : 0 < 2 * eps^2 := by positivity
    calc Real.log (2 / delta) = (Real.log (2 / delta) / (2 * eps^2)) * (2 * eps^2) := by
            field_simp [ne_of_gt h2eps]
      _ <= n * (2 * eps^2) := by nlinarith
      _ = 2 * n * eps^2 := by ring
  -- Therefore ln(2/delta)/(2n) <= eps^2
  have h_ratio_le : Real.log (2 / delta) / (2 * n) <= eps^2 := by
    have h2n_pos : 0 < 2 * (n : Real) := by positivity
    apply (div_le_iff₀ h2n_pos).2
    nlinarith [h_ineq]
  -- Finally, sqrt(ln(2/delta)/(2n)) <= eps
  apply Real.sqrt_le_iff.mpr
  exact And.intro (le_of_lt h_eps) h_ratio_le
