import FormalProofs.DSL.MomentFunctions

/-!
# FormalProofs/DSL/MultinomialLogistic.lean

## Paper Reference: Appendix B.3 (Multinomial Logistic Regression)

This file formalizes the multinomial logit moment function:
- Baseline category indexed as 0
- Parameters for remaining categories k = 1..K

Moment (per k):
  m_k = (1{Y = k} - rho_k) * X
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

/-- Multinomial logit denominator: 1 + sum_k exp(X' beta_k). -/
def multinomialDenom {p n : ℕ}
    (x : Fin p → ℝ) (beta : Fin n → Fin p → ℝ) : ℝ :=
  1 + ∑ k : Fin n, Real.exp (innerProduct x (beta k))

/-- Multinomial logit probability for category k (non-baseline). -/
def multinomialRho {p n : ℕ}
    (x : Fin p → ℝ) (beta : Fin n → Fin p → ℝ) (k : Fin n) : ℝ :=
  Real.exp (innerProduct x (beta k)) / multinomialDenom x beta

/-- Indicator for category k, with baseline category = 0.

We encode outcomes as `Fin (n+1)`:
- 0 = baseline
- succ k = non-baseline category k
-/
def multinomialIndicator {n : ℕ} (y : Fin (n + 1)) (k : Fin n) : ℝ :=
  if y = Fin.succ k then 1 else 0

/-- Multinomial logit moment function (Appendix B.3).

Returns a vector for each non-baseline category k.
-/
def multinomialLogitMoment {p n : ℕ}
    (y : Fin (n + 1)) (x : Fin p → ℝ) (beta : Fin n → Fin p → ℝ) :
    Fin n → Fin p → ℝ :=
  fun k j => (multinomialIndicator y k - multinomialRho x beta k) * x j

end DSL

end
