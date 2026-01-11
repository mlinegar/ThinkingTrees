import FormalProofs.DSL.MomentFunctions

/-!
# FormalProofs/DSL/InstrumentalVariables.lean

## Paper Reference: Appendix B.3 (Instrumental Variables)

Moment (two-stage least squares, single instrument):
  m(D; beta) = (Y - beta0 - beta1 * T - X' * beta2) * (1, Z, X)

This file encodes the residual and the corresponding moment vector.
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

/-- IV data point: outcome, treatment, instrument, covariates. -/
structure IVData (p : ℕ) where
  Y : ℝ
  T : ℝ
  Z : ℝ
  X : Fin p → ℝ

/-- IV parameters: intercept, treatment coefficient, covariate coefficients. -/
structure IVParams (p : ℕ) where
  beta0 : ℝ
  beta1 : ℝ
  beta2 : Fin p → ℝ

/-- IV residual: Y - beta0 - beta1 * T - X' beta2. -/
def ivResidual {p : ℕ} (d : IVData p) (beta : IVParams p) : ℝ :=
  d.Y - beta.beta0 - beta.beta1 * d.T - innerProduct d.X beta.beta2

/-- IV moment vector: residual times instrument vector (1, Z, X). -/
def ivMoment {p : ℕ} (d : IVData p) (beta : IVParams p) :
    ℝ × ℝ × (Fin p → ℝ) :=
  let e := ivResidual d beta
  (e, e * d.Z, fun j => e * d.X j)

end DSL

end
