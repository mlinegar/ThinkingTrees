import Mathlib

/-!
# FormalProofs/OPT/RootLocalObjective.lean

Nominal root/local objective.

The paper-facing objective uses a fixed analyst-chosen local-law share
`Lambda`. Oracle disagreement is handled inside the local-law loss supplied to
this objective, not by changing `Lambda`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-- Nominal root/local objective with the user-facing local-law share
`Lambda`: `(1 - Lambda) * rootLoss + Lambda * lawLoss`. -/
def nominalRootLocalObjective
    (Lambda rootLoss lawLoss : ℝ) : ℝ :=
  (1 - Lambda) * rootLoss + Lambda * lawLoss

end FormalProofs.OPT
