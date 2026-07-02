import FormalProofs.OPT.DoublyRobustLocalLawAdjustment
import FormalProofs.OPT.MergeableProjection
import FormalProofs.OPT.RootLocalObjective

/-!
# FormalProofs/OPT/DoublyRobustMinimizationObjective.lean

Full doubly robust minimization objective.

This module names the active optimization surface after the local-law channel
has been corrected by the DSL/AIPW doubly robust adjustment. The root supervised
channel is combined with the adjusted discounted local-law channel using the
nominal root/local weight `Lambda`.
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

variable {Param Strings : Type*}

/-! ## Proxy-oracle gap loss vocabulary -/

/-- Pointwise gap loss between a learned proxy `fhat` and the target oracle
`fstar`. Empirical or population averages of this scalar are the intended
instantiations of the proxy-oracle gap channel below. -/
def oracleGapLossAt {X Y : Type*} [PseudoMetricSpace Y]
    (fstar fhat : X -> Y) (x : X) : ℝ :=
  dist (fhat x) (fstar x)

theorem oracleGapLossAt_nonneg {X Y : Type*} [PseudoMetricSpace Y]
    (fstar fhat : X -> Y) (x : X) :
    0 ≤ oracleGapLossAt fstar fhat x := by
  exact dist_nonneg

/-! ## Problem data -/

/-- Generic data for the full doubly robust minimization problem.

`Param` is the candidate predictor/policy/summarizer class being optimized.
For each candidate, the local-law channel is measured by a proxy loss at every
node and corrected by node-oracle observations when they are available. -/
structure DoublyRobustMinimizationProblem (Param Strings : Type*) where
  oracleGapLoss : Param -> ℝ
  rootLoss : Param -> ℝ
  nodeProxyLoss : Param -> BinTree Strings -> ℝ
  nodeOracleLoss : Param -> BinTree Strings -> ℝ
  nodeObserved : Param -> BinTree Strings -> DSL.SamplingIndicator
  nodePi : Param -> BinTree Strings -> ℝ
  gammaDepth : ℝ
  Lambda : ℝ
  oracleGapWeight : ℝ
  rootWeight : ℝ

/-! ## Objective components -/

/-- Proxy-oracle gap channel contribution for penalizing the discrepancy
between the learned proxy measurement `fhat` and the oracle `fstar`. -/
def drOracleGapChannelLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) : ℝ :=
  problem.oracleGapWeight * problem.oracleGapLoss theta

/-- Root supervised channel contribution before root/local lambda mixing. -/
def drRootChannelLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) : ℝ :=
  problem.rootWeight * problem.rootLoss theta

/-- Discounted doubly robust adjusted local-law estimate for one candidate. -/
def drAdjustedLocalLawLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  unifiedAdjustedLocalLawEstimate problem.gammaDepth
    (problem.nodeProxyLoss theta) (problem.nodeOracleLoss theta)
    (problem.nodeObserved theta) (problem.nodePi theta) T

/-! ## Local-law bias decomposition -/

/-- Node-level proxy-oracle local-law bias. Positive values mean the proxy law
loss is larger than the oracle law loss at that node. -/
def drNodeLocalLawBias
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  problem.nodeProxyLoss theta T - problem.nodeOracleLoss theta T

/-- The residual multiplier left after applying the node-oracle correction:
`1 - R/pi`. -/
def drNodeBiasResidualFactor
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  1 - (problem.nodeObserved theta T).toReal / problem.nodePi theta T

/-- Node residual bias after correction:
`(1 - R/pi) * (proxy law loss - oracle law loss)`. -/
def drNodeBiasResidual
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  drNodeBiasResidualFactor problem theta T *
    drNodeLocalLawBias problem theta T

/-- Discounted residual-bias aggregate for the corrected local-law channel. -/
def drDiscountedNodeBiasResidual
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss problem.gammaDepth
    (drNodeBiasResidual problem theta) T

/-- Bias-form local-law channel:
oracle local-law objective plus the discounted residual proxy-oracle bias. -/
def drBiasFormLocalLawLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  discountedTrueNodeLawLoss problem.gammaDepth
      (problem.nodeOracleLoss theta) T +
    drDiscountedNodeBiasResidual problem theta T

theorem drNodeAIPWAdjustedLawLoss_eq_oracle_plus_biasResidual
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) :
    nodeAIPWAdjustedLawLoss (problem.nodeProxyLoss theta)
        (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
        (problem.nodePi theta) T =
      problem.nodeOracleLoss theta T +
        drNodeBiasResidual problem theta T := by
  unfold nodeAIPWAdjustedLawLoss drNodeBiasResidual
    drNodeBiasResidualFactor drNodeLocalLawBias
  unfold DSL.designAdjustedOutcome DSL.SamplingIndicator.toReal
  cases hObs : problem.nodeObserved theta T <;> simp [hObs]
  all_goals ring

theorem drAdjustedLocalLawLoss_eq_biasForm
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) :
    drAdjustedLocalLawLoss problem theta T =
      drBiasFormLocalLawLoss problem theta T := by
  induction T with
  | leaf a =>
      unfold drAdjustedLocalLawLoss unifiedAdjustedLocalLawEstimate
        discountedAIPWNodeLawLoss drBiasFormLocalLawLoss
        discountedTrueNodeLawLoss drDiscountedNodeBiasResidual
      simp [drNodeAIPWAdjustedLawLoss_eq_oracle_plus_biasResidual]
  | node TL TR ihL ihR =>
      unfold drAdjustedLocalLawLoss unifiedAdjustedLocalLawEstimate
        discountedAIPWNodeLawLoss drBiasFormLocalLawLoss
        discountedTrueNodeLawLoss drDiscountedNodeBiasResidual at ihL ihR ⊢
      simp [drNodeAIPWAdjustedLawLoss_eq_oracle_plus_biasResidual]
      rw [ihL, ihR]
      ring

/-! ## Nominal root/local objective -/

/-- Root/local part of the full objective for an explicit local-law scalar. -/
def drRootLocalValueWithLocalLaw
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (lawLoss : ℝ) : ℝ :=
  nominalRootLocalObjective problem.Lambda
    (drRootChannelLoss problem theta) lawLoss

/-- Full objective for an explicit local-law scalar. -/
def drMinimizationValueWithLocalLaw
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (lawLoss : ℝ) : ℝ :=
  drOracleGapChannelLoss problem theta +
    drRootLocalValueWithLocalLaw problem theta lawLoss

/-- Full objective for explicit envelope and local-law scalars.
The envelope is retained only as a statement convenience for residual-envelope
lemmas; it does not alter the root/local weight. -/
def drMinimizationValueWithEnvelopeAndLocalLaw
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (_envelope lawLoss : ℝ) : ℝ :=
  drMinimizationValueWithLocalLaw problem theta lawLoss

/-- Exact doubly robust minimization objective:
proxy-oracle gap plus root/local lambda mixing, where the local-law loss is the
discounted AIPW/DSL adjusted node-law estimate. -/
def drMinimizationValue
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  drMinimizationValueWithLocalLaw problem theta
    (drAdjustedLocalLawLoss problem theta T)

/-- Full objective written directly in bias form. -/
def drBiasFormMinimizationValue
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  drMinimizationValueWithLocalLaw problem theta
    (drBiasFormLocalLawLoss problem theta T)

theorem drMinimizationValue_eq_oracleGap_plus_nominal
    {problem : DoublyRobustMinimizationProblem Param Strings}
    {theta : Param} {T : BinTree Strings} :
    drMinimizationValue problem theta T =
      drOracleGapChannelLoss problem theta +
        nominalRootLocalObjective problem.Lambda
          (drRootChannelLoss problem theta)
          (drAdjustedLocalLawLoss problem theta T) := by
  rfl

theorem drMinimizationValue_eq_biasForm
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) :
    drMinimizationValue problem theta T =
      drBiasFormMinimizationValue problem theta T := by
  unfold drMinimizationValue drBiasFormMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_biasForm]

/-! ## Local-law endpoints inside the full objective -/

theorem drAdjustedLocalLawLoss_eq_proxy_of_all_unsampled
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = false) :
    drAdjustedLocalLawLoss problem theta T =
      discountedSurrogateNodeLawLoss problem.gammaDepth
        (problem.nodeProxyLoss theta) T := by
  unfold drAdjustedLocalLawLoss
  exact unifiedAdjustedLocalLawEstimate_eq_proxy_of_all_unsampled
    problem.gammaDepth (problem.nodeProxyLoss theta)
    (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
    (problem.nodePi theta) T hObserved

theorem drMinimizationValue_eq_proxy_of_all_unsampled
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = false) :
    drMinimizationValue problem theta T =
      drMinimizationValueWithLocalLaw problem theta
        (discountedSurrogateNodeLawLoss problem.gammaDepth
          (problem.nodeProxyLoss theta) T) := by
  unfold drMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_proxy_of_all_unsampled
    problem theta T hObserved]

theorem drAdjustedLocalLawLoss_eq_oracle_of_all_sampled_pi_one
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = true)
    (hPi : ∀ U : BinTree Strings, problem.nodePi theta U = 1) :
    drAdjustedLocalLawLoss problem theta T =
      discountedTrueNodeLawLoss problem.gammaDepth
        (problem.nodeOracleLoss theta) T := by
  unfold drAdjustedLocalLawLoss
  exact unifiedAdjustedLocalLawEstimate_eq_oracle_of_all_sampled_pi_one
    problem.gammaDepth (problem.nodeProxyLoss theta)
    (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
    (problem.nodePi theta) T hObserved hPi

theorem drMinimizationValue_eq_oracle_of_all_sampled_pi_one
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = true)
    (hPi : ∀ U : BinTree Strings, problem.nodePi theta U = 1) :
    drMinimizationValue problem theta T =
      drMinimizationValueWithLocalLaw problem theta
        (discountedTrueNodeLawLoss problem.gammaDepth
          (problem.nodeOracleLoss theta) T) := by
  unfold drMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_oracle_of_all_sampled_pi_one
    problem theta T hObserved hPi]

theorem drAdjustedLocalLawLoss_eq_oracle_of_exact_proxy
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hExact :
      ∀ U : BinTree Strings,
        problem.nodeProxyLoss theta U = problem.nodeOracleLoss theta U) :
    drAdjustedLocalLawLoss problem theta T =
      discountedTrueNodeLawLoss problem.gammaDepth
        (problem.nodeOracleLoss theta) T := by
  unfold drAdjustedLocalLawLoss
  exact unifiedAdjustedLocalLawEstimate_eq_oracle_of_exact_proxy
    problem.gammaDepth (problem.nodeProxyLoss theta)
    (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
    (problem.nodePi theta) T hExact

theorem drMinimizationValue_eq_oracle_of_exact_proxy
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hExact :
      ∀ U : BinTree Strings,
        problem.nodeProxyLoss theta U = problem.nodeOracleLoss theta U) :
    drMinimizationValue problem theta T =
      drMinimizationValueWithLocalLaw problem theta
        (discountedTrueNodeLawLoss problem.gammaDepth
          (problem.nodeOracleLoss theta) T) := by
  unfold drMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_oracle_of_exact_proxy
    problem theta T hExact]

/-! ## Propensity-residual objective simplifications -/

theorem drMinimizationValueWithEnvelope_eq_no_propensity_residual_of_matched
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param)
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss lawLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed) :
    drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack
          (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss))
        lawLoss =
      drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack 0)
        lawLoss := by
  rfl

theorem drMinimizationValueWithEnvelope_eq_no_propensity_residual_of_exact_proxy
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param)
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss lawLoss : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack
          (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss))
        lawLoss =
      drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack 0)
        lawLoss := by
  rfl

theorem drMinimizationValueWithEnvelope_eq_no_propensity_residual_of_dr
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param)
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss lawLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss) :
    drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack
          (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss))
        lawLoss =
      drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack 0)
        lawLoss := by
  rfl

/-! ## Feasible minimization -/

/-- Feasible minimizer for the full doubly robust objective. -/
def IsDRLocalLawMinimizer
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (feasible : Set Param)
    (theta : Param) (T : BinTree Strings) : Prop :=
  IsMergeableProjection feasible
    (fun theta' : Param => drMinimizationValue problem theta' T) theta

/-- Projection-style alias for later mergeability-gap transfer results. -/
abbrev IsDRLocalLawProjection :=
  @IsDRLocalLawMinimizer

theorem drLocalLawMinimizer_mem
    {problem : DoublyRobustMinimizationProblem Param Strings}
    {feasible : Set Param} {theta : Param} {T : BinTree Strings}
    (hmin : IsDRLocalLawMinimizer problem feasible theta T) :
    theta ∈ feasible :=
  hmin.mem

theorem drLocalLawMinimizer_value_le
    {problem : DoublyRobustMinimizationProblem Param Strings}
    {feasible : Set Param} {theta theta' : Param} {T : BinTree Strings}
    (hmin : IsDRLocalLawMinimizer problem feasible theta T)
    (hfeasible : theta' ∈ feasible) :
    drMinimizationValue problem theta T ≤
      drMinimizationValue problem theta' T :=
  hmin.risk_le theta' hfeasible

end FormalProofs.OPT
