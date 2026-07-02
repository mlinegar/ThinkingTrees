import FormalProofs.OPT.DiscountedTreeMetaObjective

/-!
# FormalProofs/OPT/NodeLocalLawAggregate.lean

Depth-discounted node local-law aggregation.
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

variable {Strings : Type*}

/-- Discounted aggregate of node-level local-law losses measured through
`fhat`. The aggregation convention is inherited from `discountedTreeMetaLoss`. -/
def discountedSurrogateNodeLawLoss
    (gammaDepth : ℝ)
    (nodeSurrogateLoss : BinTree Strings → ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth nodeSurrogateLoss T

/-- Discounted aggregate of the corresponding node-level local-law losses that
would be measured through `fstar`. -/
def discountedTrueNodeLawLoss
    (gammaDepth : ℝ)
    (nodeTrueLoss : BinTree Strings → ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth nodeTrueLoss T

/-- Discounted aggregate node-bias envelope. -/
def discountedNodeBiasBound
    (gammaDepth : ℝ)
    (nodeBiasBound : BinTree Strings → ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth nodeBiasBound T

theorem discountedNodeBiasBound_nonneg
    (gammaDepth : ℝ)
    (nodeBiasBound : BinTree Strings → ℝ)
    (T : BinTree Strings)
    (hGamma : 0 ≤ gammaDepth)
    (hBiasNonneg : ∀ U : BinTree Strings, 0 ≤ nodeBiasBound U) :
    0 ≤ discountedNodeBiasBound gammaDepth nodeBiasBound T := by
  exact discountedTreeMetaLoss_nonneg gammaDepth nodeBiasBound T hGamma hBiasNonneg

/-- If every node's surrogate-vs-true law loss is bounded by a nodewise bias
envelope, then the discounted aggregate surrogate-vs-true law loss is bounded by
the discounted aggregate envelope. -/
theorem discountedNodeLawLoss_abs_sub_le_biasBound
    (gammaDepth : ℝ)
    (nodeSurrogateLoss nodeTrueLoss nodeBiasBound : BinTree Strings → ℝ)
    (T : BinTree Strings)
    (hGamma : 0 ≤ gammaDepth)
    (hNode :
      ∀ U : BinTree Strings,
        |nodeSurrogateLoss U - nodeTrueLoss U| ≤ nodeBiasBound U)
    (hBiasNonneg : ∀ U : BinTree Strings, 0 ≤ nodeBiasBound U) :
    |discountedSurrogateNodeLawLoss gammaDepth nodeSurrogateLoss T -
      discountedTrueNodeLawLoss gammaDepth nodeTrueLoss T| ≤
      discountedNodeBiasBound gammaDepth nodeBiasBound T := by
  induction T with
  | leaf a =>
      simpa [discountedSurrogateNodeLawLoss, discountedTrueNodeLawLoss,
        discountedNodeBiasBound] using hNode (BinTree.leaf a)
  | node TL TR ihL ihR =>
      let root : BinTree Strings := BinTree.node TL TR
      let sRoot := nodeSurrogateLoss root
      let tRoot := nodeTrueLoss root
      let sL := discountedTreeMetaLoss gammaDepth nodeSurrogateLoss TL
      let tL := discountedTreeMetaLoss gammaDepth nodeTrueLoss TL
      let sR := discountedTreeMetaLoss gammaDepth nodeSurrogateLoss TR
      let tR := discountedTreeMetaLoss gammaDepth nodeTrueLoss TR
      let bRoot := nodeBiasBound root
      let bL := discountedTreeMetaLoss gammaDepth nodeBiasBound TL
      let bR := discountedTreeMetaLoss gammaDepth nodeBiasBound TR
      have hRoot : |sRoot - tRoot| ≤ bRoot := hNode root
      have hL : |sL - tL| ≤ bL := by
        simpa [sL, tL, bL, discountedSurrogateNodeLawLoss,
          discountedTrueNodeLawLoss, discountedNodeBiasBound] using ihL
      have hR : |sR - tR| ≤ bR := by
        simpa [sR, tR, bR, discountedSurrogateNodeLawLoss,
          discountedTrueNodeLawLoss, discountedNodeBiasBound] using ihR
      have hGL : |gammaDepth * (sL - tL)| ≤ gammaDepth * bL := by
        have hmul := mul_le_mul_of_nonneg_left hL hGamma
        simpa [abs_mul, abs_of_nonneg hGamma] using hmul
      have hGR : |gammaDepth * (sR - tR)| ≤ gammaDepth * bR := by
        have hmul := mul_le_mul_of_nonneg_left hR hGamma
        simpa [abs_mul, abs_of_nonneg hGamma] using hmul
      have hTriangle :
          |(sRoot - tRoot) + gammaDepth * (sL - tL) +
              gammaDepth * (sR - tR)| ≤
            |sRoot - tRoot| + |gammaDepth * (sL - tL)| +
              |gammaDepth * (sR - tR)| := by
        calc
          |(sRoot - tRoot) + gammaDepth * (sL - tL) +
              gammaDepth * (sR - tR)|
              ≤ |(sRoot - tRoot) + gammaDepth * (sL - tL)| +
                  |gammaDepth * (sR - tR)| := by
                    exact abs_add_le _ _
          _ ≤ |sRoot - tRoot| + |gammaDepth * (sL - tL)| +
                  |gammaDepth * (sR - tR)| := by
                    have h := abs_add_le (sRoot - tRoot) (gammaDepth * (sL - tL))
                    linarith
      have hDecomp :
          (sRoot + gammaDepth * sL + gammaDepth * sR) -
            (tRoot + gammaDepth * tL + gammaDepth * tR) =
          (sRoot - tRoot) + gammaDepth * (sL - tL) +
            gammaDepth * (sR - tR) := by
        ring
      simp [discountedSurrogateNodeLawLoss, discountedTrueNodeLawLoss,
        discountedNodeBiasBound, discountedTreeMetaLoss_node]
      calc
        |(sRoot + gammaDepth * sL + gammaDepth * sR) -
            (tRoot + gammaDepth * tL + gammaDepth * tR)|
            = |(sRoot - tRoot) + gammaDepth * (sL - tL) +
                gammaDepth * (sR - tR)| := by rw [hDecomp]
        _ ≤ |sRoot - tRoot| + |gammaDepth * (sL - tL)| +
              |gammaDepth * (sR - tR)| := hTriangle
        _ ≤ bRoot + gammaDepth * bL + gammaDepth * bR := by
              linarith

end FormalProofs.OPT
