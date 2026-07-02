import FormalProofs.OPT.TwoStageOracleSurrogate

/-!
# FormalProofs/OPT/ProxyOracleGap.lean

Proxy/oracle gap facts used by the root and local-law routes.
-/

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

namespace FormalProofs.OPT

variable {Root : Type*}
variable {Y : Type*} [BoundedMetricSpace Y]

/-- Root-pair gap induced by evaluating a pairwise root comparison through
`fhat` instead of the target oracle `fstar`. -/
def rootPairBias
    (fstar fhat : Root → Y) (x y : Root) : ℝ :=
  dist (fhat x) (fhat y) - dist (fstar x) (fstar y)

/-- Uniform oracle approximation is symmetric after swapping `fstar` and
`fhat`, because metric distance is symmetric. -/
theorem uniformOracleApproximation_symm
    [Monoid Root]
    {fstar fhat : Root → Y} {eps : ℝ≥0}
    (hApprox : UniformOracleApproximation fstar fhat eps) :
    UniformOracleApproximation fhat fstar eps := by
  intro x
  simpa [dist_comm] using hApprox x

/-- A root-pair distance measured through `fhat` differs from the corresponding
true-oracle root-pair distance by at most the two-sided oracle-recovery slack. -/
theorem rootPairBias_abs_le_oracleRecoverySlack
    [Monoid Root]
    {fstar fhat : Root → Y} {eps : ℝ≥0}
    (hApprox : UniformOracleApproximation fstar fhat eps)
    (x y : Root) :
    |rootPairBias fstar fhat x y| ≤ OracleRecoverySlack eps := by
  have hTrueLe :
      dist (fstar x) (fstar y) ≤
        dist (fhat x) (fhat y) + 2 * (eps : ℝ) :=
    trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
      (hApprox := hApprox) (x := x) (x' := y)
  have hSurLe :
      dist (fhat x) (fhat y) ≤
        dist (fstar x) (fstar y) + 2 * (eps : ℝ) :=
    trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
      (fstar := fhat) (fhat := fstar)
      (hApprox := uniformOracleApproximation_symm (Root := Root) hApprox)
      (x := x) (x' := y)
  rw [abs_le]
  constructor
  · simp [rootPairBias, OracleRecoverySlack]
    linarith
  · simp [rootPairBias, OracleRecoverySlack]
    linarith

end FormalProofs.OPT
