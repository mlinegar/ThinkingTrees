import FormalProofs.OPT.NodeLocalLawAggregate
import FormalProofs.OPT.RootLocalObjective
import FormalProofs.DSL.NonclassicalExpectationMismatch

/-!
# FormalProofs/OPT/NodeAIPWLocalLawAdjustment.lean

Proxy plus node-oracle AIPW node-law aggregation.

Every node has a proxy local-law loss measured through `fhat`; sampled nodes
also expose the corresponding oracle local-law loss measured through `fstar`.
The node oracle channel is used as an inverse-propensity residual correction to
the proxy channel, so both channels target one true discounted node-law
estimand rather than two separate local objectives.
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

/-! ## Pointwise AIPW node law -/

/-- Per-node proxy-plus-oracle adjusted law loss. This is the DSL adjusted
outcome with `Y_pred = proxy` and `Y_true = oracle`:
`proxy + R / pi * (oracle - proxy)`. -/
def nodeAIPWAdjustedLawLoss
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  DSL.designAdjustedOutcome
    (nodeProxyLoss T) (nodeOracleLoss T) (nodeObserved T) (nodePi T)

theorem nodeAIPWAdjustedLawLoss_eq_proxy_of_unsampled
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : nodeObserved T = false) :
    nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nodeProxyLoss T := by
  unfold nodeAIPWAdjustedLawLoss
  rw [hObserved]
  exact DSL.designAdjustedOutcome_unsampled
    (nodeProxyLoss T) (nodeOracleLoss T) (nodePi T)

theorem nodeAIPWAdjustedLawLoss_eq_oracle_of_sampled_pi_one
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : nodeObserved T = true)
    (hPi : nodePi T = 1) :
    nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nodeOracleLoss T := by
  unfold nodeAIPWAdjustedLawLoss
  rw [hObserved, hPi]
  exact DSL.designAdjustedOutcome_full_sample
    (nodeProxyLoss T) (nodeOracleLoss T)

/-! ## Discounted AIPW node law -/

/-- Discounted aggregate of AIPW-adjusted node local-law losses. -/
def discountedAIPWNodeLawLoss
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi) T

private theorem discountedTreeMetaLoss_congr_all
    {α : Type*}
    (gammaDepth : ℝ)
    (nodeLoss1 nodeLoss2 : BinTree α -> ℝ)
    (T : BinTree α)
    (h : ∀ U : BinTree α, nodeLoss1 U = nodeLoss2 U) :
    discountedTreeMetaLoss gammaDepth nodeLoss1 T =
      discountedTreeMetaLoss gammaDepth nodeLoss2 T := by
  induction T with
  | leaf a =>
      simpa using h (BinTree.leaf a)
  | node TL TR ihL ihR =>
      rw [discountedTreeMetaLoss_node, discountedTreeMetaLoss_node,
        h (BinTree.node TL TR), ihL, ihR]

theorem discountedAIPWNodeLawLoss_eq_proxy_of_all_unsampled
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = false) :
    discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedSurrogateNodeLawLoss gammaDepth nodeProxyLoss T := by
  unfold discountedAIPWNodeLawLoss discountedSurrogateNodeLawLoss
  exact discountedTreeMetaLoss_congr_all gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
    nodeProxyLoss T
    (fun U => nodeAIPWAdjustedLawLoss_eq_proxy_of_unsampled
      nodeProxyLoss nodeOracleLoss nodeObserved nodePi U (hObserved U))

theorem discountedAIPWNodeLawLoss_eq_oracle_of_all_sampled_pi_one
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = true)
    (hPi : ∀ U : BinTree Strings, nodePi U = 1) :
    discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  unfold discountedAIPWNodeLawLoss discountedTrueNodeLawLoss
  exact discountedTreeMetaLoss_congr_all gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
    nodeOracleLoss T
    (fun U => nodeAIPWAdjustedLawLoss_eq_oracle_of_sampled_pi_one
      nodeProxyLoss nodeOracleLoss nodeObserved nodePi U (hObserved U) (hPi U))

/-! ## Scalar econometric residual identities -/

/-- With matched propensities, the scalar AIPW adjusted node law is unbiased for
the oracle node law, regardless of proxy error. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) = 0 := by
  exact DSL.designAdjustedOutcome_unbiased_of_matched_propensity
    proxyLoss oracleLoss piTrue piUsed hPiUsed hMatch E_cond hE_R hE_1 hE_linear

/-- With a misspecified propensity, the exact scalar residual is the propensity
mismatch factor times proxy-oracle error. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_residual_of_misspecified_propensity
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) =
      (1 - piTrue / piUsed) * (proxyLoss - oracleLoss) := by
  exact DSL.designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
    proxyLoss oracleLoss piTrue piUsed hPiUsed E_cond hE_R hE_1 hE_linear

/-- Discounted aggregate envelope for adjusted-node-law error. -/
def discountedAIPWNodeErrorBound
    (gammaDepth : ℝ)
    (nodeErrorBound : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedNodeBiasBound gammaDepth nodeErrorBound T

theorem discountedAIPWNodeLawLoss_abs_sub_le_errorBound
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss nodeErrorBound : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hGamma : 0 ≤ gammaDepth)
    (hNode :
      ∀ U : BinTree Strings,
        |nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi U -
          nodeOracleLoss U| ≤ nodeErrorBound U)
    (hErrorNonneg : ∀ U : BinTree Strings, 0 ≤ nodeErrorBound U) :
    |discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T -
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T| ≤
      discountedAIPWNodeErrorBound gammaDepth nodeErrorBound T := by
  simpa [discountedAIPWNodeLawLoss, discountedAIPWNodeErrorBound]
    using
      discountedNodeLawLoss_abs_sub_le_biasBound gammaDepth
        (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
        nodeOracleLoss nodeErrorBound T hGamma hNode hErrorNonneg

/-- Nominal root/local objective using the adjusted node-law channel. -/
def nominalAIPWNodeObjective
    (Lambda gammaDepth rootLoss : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  nominalRootLocalObjective Lambda rootLoss
    (discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
      nodeObserved nodePi T)

end FormalProofs.OPT
