import FormalProofs.OPT.NodeAIPWLocalLawAdjustment

/-!
# FormalProofs/OPT/UnifiedLocalLawAdjustment.lean

Unified local-law adjustment equation.

This module names the final paper-facing local-law adjustment layer. Node
local-law losses measured through `fhat` are corrected by node-oracle residuals
when those observations are present and aggregated down the tree with the
existing depth discount. The paper-facing theorem path supplies this corrected
local-law loss to `nominalRootLocalObjective` with the nominal lambda.

The DSL/IPW certificate components remain scalar inputs here. In particular,
the propensity-misspecification residual is explicit, so it is zero in the
matched-propensity design case and visible otherwise.
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

/-! ## Unified adjusted local-law estimate -/

/-- The unified adjusted local-law estimate:
`sum_v gammaDepth^depth(v) * (proxy_v + R_v/pi_v * (oracle_v - proxy_v))`.
It is definitionally the discounted AIPW node-law aggregate. -/
def unifiedAdjustedLocalLawEstimate
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
    nodeObserved nodePi T

theorem unifiedAdjustedLocalLawEstimate_eq_proxy_of_all_unsampled
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = false) :
    unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedSurrogateNodeLawLoss gammaDepth nodeProxyLoss T := by
  exact discountedAIPWNodeLawLoss_eq_proxy_of_all_unsampled
    gammaDepth nodeProxyLoss nodeOracleLoss nodeObserved nodePi T hObserved

theorem unifiedAdjustedLocalLawEstimate_eq_oracle_of_all_sampled_pi_one
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = true)
    (hPi : ∀ U : BinTree Strings, nodePi U = 1) :
    unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  exact discountedAIPWNodeLawLoss_eq_oracle_of_all_sampled_pi_one
    gammaDepth nodeProxyLoss nodeOracleLoss nodeObserved nodePi T hObserved hPi

/-! ## Scalar certificate envelope -/

/-- Explicit residual left by propensity misspecification in the scalar
design-adjusted local-law outcome. -/
def propensityMismatchResidual
    (piTrue piUsed proxyLoss oracleLoss : ℝ) : ℝ :=
  (1 - piTrue / piUsed) * (proxyLoss - oracleLoss)

theorem propensityMismatchResidual_eq_zero_of_matched
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed) :
    propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss = 0 := by
  unfold propensityMismatchResidual
  rw [hMatch]
  rw [div_self (ne_of_gt hPiUsed)]
  ring

/-- The scalar AIPW residual identity is exactly the named propensity-mismatch
residual. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_propensityMismatchResidual
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) =
      propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss := by
  exact nodeAIPWAdjustedLawLoss_expectation_eq_residual_of_misspecified_propensity
    proxyLoss oracleLoss piTrue piUsed hPiUsed E_cond hE_R hE_1 hE_linear

/-- Paper-facing scalar local-law error envelope:
IPW/AIPW point estimate plus proxy-oracle gap margin, sampling uncertainty,
oracle slack, and explicit propensity residual. -/
def adjustedLocalLawEnvelope
    (ipwEstimate gapMargin zScore se oracleSlack propensityResidual : ℝ) : ℝ :=
  ipwEstimate + gapMargin + zScore * se + 2 * oracleSlack +
    propensityResidual

/-! ## Nominal unified objective -/

/-- Paper-facing nominal root/local objective with the corrected local-law
estimate as the local channel. -/
def unifiedLocalLawNominalObjective
    (Lambda gammaDepth rootLoss : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  nominalRootLocalObjective Lambda rootLoss
    (unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
      nodeObserved nodePi T)

theorem unifiedLocalLawNominalObjective_eq_nominal
    {gammaDepth rootLoss : ℝ}
    {nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ}
    {nodeObserved : BinTree Strings -> DSL.SamplingIndicator}
    {nodePi : BinTree Strings -> ℝ}
    {T : BinTree Strings}
    (Lambda : ℝ) :
    unifiedLocalLawNominalObjective Lambda gammaDepth rootLoss
        nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nominalRootLocalObjective Lambda rootLoss
        (unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
          nodeObserved nodePi T) := by
  rfl

end FormalProofs.OPT
