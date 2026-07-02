import FormalProofs.OPT.UnifiedLocalLawAdjustment

/-!
# FormalProofs/OPT/DoublyRobustLocalLawAdjustment.lean

Doubly robust local-law adjustment.

This module makes explicit the classical AIPW/DSL double-robust reading of the
unified local-law channel. The adjusted node-law outcome is unbiased when either
the logged propensity matches the true node-oracle sampling law, or the proxy
local-law loss is already equal to the oracle local-law loss.

The first route is inherited from `UnifiedLocalLawAdjustment`; this file adds
the exact-proxy route and names the combined disjunction.
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

/-! ## Scalar doubly robust residual cancellation -/

/-- If the proxy loss is exact, the explicit propensity-mismatch residual is
zero for any true/used propensity pair. -/
theorem propensityMismatchResidual_eq_zero_of_exact_proxy
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss = 0 := by
  unfold propensityMismatchResidual
  rw [hExact]
  ring

/-- The explicit propensity residual vanishes if either the propensity is
matched or the proxy local-law loss is exact. -/
theorem propensityMismatchResidual_eq_zero_of_matched_or_exact_proxy
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss) :
    propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss = 0 := by
  cases hDR with
  | inl hMatch =>
      exact propensityMismatchResidual_eq_zero_of_matched
        piTrue piUsed proxyLoss oracleLoss hPiUsed hMatch
  | inr hExact =>
      exact propensityMismatchResidual_eq_zero_of_exact_proxy
        piTrue piUsed proxyLoss oracleLoss hExact

/-- If the proxy outcome equals the oracle outcome, the DSL adjusted outcome is
the oracle outcome for any sampling indicator and any used propensity. -/
theorem designAdjustedOutcome_eq_true_of_exact_proxy
    (proxyLoss oracleLoss : ℝ)
    (R : DSL.SamplingIndicator)
    (piUsed : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed = oracleLoss := by
  subst proxyLoss
  unfold DSL.designAdjustedOutcome
  cases R <;> simp

/-- Exact proxy local laws remove the scalar AIPW residual even when the
propensity used for weighting is misspecified. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_exact_proxy
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (hExact : proxyLoss = oracleLoss)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) = 0 := by
  rw [nodeAIPWAdjustedLawLoss_expectation_eq_propensityMismatchResidual
    proxyLoss oracleLoss piTrue piUsed hPiUsed E_cond hE_R hE_1 hE_linear]
  exact propensityMismatchResidual_eq_zero_of_exact_proxy
    piTrue piUsed proxyLoss oracleLoss hExact

/-- Scalar doubly robust unbiasedness: the AIPW/DSL residual is zero if either
the propensity is matched or the proxy local-law loss is exact. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity_or_exact_proxy
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) = 0 := by
  cases hDR with
  | inl hMatch =>
      exact nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity
        proxyLoss oracleLoss piTrue piUsed hPiUsed hMatch E_cond hE_R hE_1
        hE_linear
  | inr hExact =>
      exact nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_exact_proxy
        proxyLoss oracleLoss piTrue piUsed hPiUsed hExact E_cond hE_R hE_1
        hE_linear

/-! ## Pointwise and discounted exact-proxy endpoints -/

/-- If the proxy node loss equals the oracle node loss at a node, the adjusted
node loss equals the oracle node loss for any observation state and propensity. -/
theorem nodeAIPWAdjustedLawLoss_eq_oracle_of_exact_proxy
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hExact : nodeProxyLoss T = nodeOracleLoss T) :
    nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nodeOracleLoss T := by
  unfold nodeAIPWAdjustedLawLoss
  exact designAdjustedOutcome_eq_true_of_exact_proxy
    (nodeProxyLoss T) (nodeOracleLoss T) (nodeObserved T) (nodePi T) hExact

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

/-- If every proxy node loss equals its oracle counterpart, the discounted AIPW
node law is exactly the true-oracle discounted node law. -/
theorem discountedAIPWNodeLawLoss_eq_oracle_of_exact_proxy
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hExact : ∀ U : BinTree Strings, nodeProxyLoss U = nodeOracleLoss U) :
    discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  unfold discountedAIPWNodeLawLoss discountedTrueNodeLawLoss
  exact discountedTreeMetaLoss_congr_all gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
    nodeOracleLoss T
    (fun U => nodeAIPWAdjustedLawLoss_eq_oracle_of_exact_proxy
      nodeProxyLoss nodeOracleLoss nodeObserved nodePi U (hExact U))

/-- The unified adjusted local-law estimate reduces to the true-oracle
discounted law when the proxy local law is exact at every node. -/
theorem unifiedAdjustedLocalLawEstimate_eq_oracle_of_exact_proxy
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hExact : ∀ U : BinTree Strings, nodeProxyLoss U = nodeOracleLoss U) :
    unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  unfold unifiedAdjustedLocalLawEstimate
  exact discountedAIPWNodeLawLoss_eq_oracle_of_exact_proxy
    gammaDepth nodeProxyLoss nodeOracleLoss nodeObserved nodePi T hExact

/-! ## Envelope simplifications under doubly robust residual cancellation -/

/-- If the explicit propensity residual is zero, the adjusted local-law envelope
is the same as the no-propensity-residual envelope. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    (ipwEstimate gapMargin zScore se oracleSlack propensityResidual : ℝ)
    (hResidual : propensityResidual = 0) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        propensityResidual =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  simp [adjustedLocalLawEnvelope, hResidual]

/-- Matched propensities remove the explicit propensity-residual term from the
adjusted local-law envelope. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_matched_propensity
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss) =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  exact adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    ipwEstimate gapMargin zScore se oracleSlack
    (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss)
    (propensityMismatchResidual_eq_zero_of_matched
      piTrue piUsed proxyLoss oracleLoss hPiUsed hMatch)

/-- Exact proxy local laws remove the explicit propensity-residual term from the
adjusted local-law envelope. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_exact_proxy
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss) =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  exact adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    ipwEstimate gapMargin zScore se oracleSlack
    (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss)
    (propensityMismatchResidual_eq_zero_of_exact_proxy
      piTrue piUsed proxyLoss oracleLoss hExact)

/-- The adjusted local-law envelope drops the explicit propensity-residual term
under either classical doubly robust route. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_matched_propensity_or_exact_proxy
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss) =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  exact adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    ipwEstimate gapMargin zScore se oracleSlack
    (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss)
    (propensityMismatchResidual_eq_zero_of_matched_or_exact_proxy
      piTrue piUsed proxyLoss oracleLoss hPiUsed hDR)

end FormalProofs.OPT
