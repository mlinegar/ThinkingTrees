import FormalProofs.DSL.MomentFunctions

/-!
# FormalProofs/DSL/RegressionDiscontinuity.lean

## Paper Reference: Appendix B.3 (Regression Discontinuity)

Local linear regression at cutoff c with bandwidth h and kernel K:
- Above cutoff: moments m1
- Below cutoff: moments m0
RD effect = beta1_0 - beta0_0 (difference in intercepts).
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

/-- Kernel function type. -/
abbrev Kernel := ℝ → ℝ

/-- RD data point with outcome and running variable. -/
structure RDData where
  Y : ℝ
  X : ℝ

/-- Local linear design vector: (1, X - c). -/
def rdDesignVec (d : RDData) (c : ℝ) : Fin 2 → ℝ :=
  fun j => match j with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => d.X - c

/-- RD residual for local linear regression. -/
def rdResidual (d : RDData) (c : ℝ) (beta : Fin 2 → ℝ) : ℝ :=
  d.Y - (beta 0 + beta 1 * (d.X - c))

/-- Weight for observations above the cutoff. -/
def rdWeightAbove (d : RDData) (c h : ℝ) (K : Kernel) : ℝ :=
  if c < d.X ∧ d.X < c + h then K ((d.X - c) / h) else 0

/-- Weight for observations below the cutoff. -/
def rdWeightBelow (d : RDData) (c h : ℝ) (K : Kernel) : ℝ :=
  if c - h < d.X ∧ d.X < c then K ((d.X - c) / h) else 0

/-- Moment above the cutoff (Appendix B.3). -/
def rdMomentAbove (d : RDData) (c h : ℝ) (K : Kernel) (beta : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let w := rdWeightAbove d c h K
  fun j => w * rdResidual d c beta * rdDesignVec d c j

/-- Moment below the cutoff (Appendix B.3). -/
def rdMomentBelow (d : RDData) (c h : ℝ) (K : Kernel) (beta : Fin 2 → ℝ) : Fin 2 → ℝ :=
  let w := rdWeightBelow d c h K
  fun j => w * rdResidual d c beta * rdDesignVec d c j

/-- RD effect: difference in intercepts above vs below cutoff. -/
def rdEffect (betaAbove betaBelow : Fin 2 → ℝ) : ℝ :=
  betaAbove 0 - betaBelow 0

end DSL

end
