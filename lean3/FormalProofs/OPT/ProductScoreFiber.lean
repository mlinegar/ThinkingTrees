import FormalProofs.OPT.SharedFeatureMultihead
import FormalProofs.OPT.TwoStageLabelScoreObjectives

/-!
# FormalProofs/OPT/ProductScoreFiber.lean

Explicit product-state specialization for the factorized score-fiber route.

This file packages the intended implementation shape:

- a learned theorem-bearing feature of the form `ℝ × FiberState`,
- an exact scalar score readout from the first coordinate, and
- an approximate summary/readout route through the full product state.

The main purpose is to expose a direct theorem surface for the structured
`factorized_score_fiber` implementation, without changing the broader shared-
feature and two-stage theory.
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

variable {Strings : Type*}

/-- The exact scalar score route for a product-state theorem feature is just the
first-coordinate projection. -/
theorem scoreReadoutFactorsThroughProductScoreFiber_firstCoordinate
    {Score Fiber : Type*}
    {feature : Strings → Score × Fiber}
    {scoreReadout : Strings → Score}
    (hScore : ∀ x : Strings, scoreReadout x = (feature x).1) :
    ReadoutFactorsThroughFeature feature scoreReadout := by
  refine ⟨fun z => z.1, ?_⟩
  intro x
  exact hScore x

section SharedFeatureRoute

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Score Fiber Summary : Type*}
variable [MetricSpace Score] [PseudoMetricSpace Fiber] [PseudoMetricSpace Summary]

/-- One learned product-state theorem feature supports an exact scalar score
readout and an approximate summary readout simultaneously. -/
theorem paired_approxOracleRecoversReadouts_of_productScoreFiber
    {fstar : Strings → Y}
    {feature : Strings → Score × Fiber}
    {scoreReadout : Strings → Score}
    {summaryReadout : Strings → Summary}
    {ε_fiber ε_summary : ℝ≥0}
    {L_score L_summary : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hScore : ∀ x : Strings, scoreReadout x = (feature x).1)
    (hScoreLip : LipschitzWith L_score (fun z : Score × Fiber => z.1))
    (hSummaryApproxFactor : ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : (Score × Fiber) → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover) :
    ApproxOracleRecoversFeature fstar scoreReadout (L_score * ε_fiber) ∧
      ApproxOracleRecoversFeature fstar summaryReadout
        (L_summary * ε_fiber + 2 * ε_summary) := by
  have hScoreFactor :
      ReadoutFactorsThroughFeature feature scoreReadout :=
    scoreReadoutFactorsThroughProductScoreFiber_firstCoordinate
      (feature := feature) (scoreReadout := scoreReadout) hScore
  have hScoreApprox :
      ApproxReadoutFactorsThroughFeature feature scoreReadout 0 :=
    readoutFactorsThroughFeature_implies_approx hScoreFactor 0
  have hScoreWitness :
      ∃ recover : (Score × Fiber) → Score,
        (∀ x : Strings, dist (scoreReadout x) (recover (feature x)) ≤ (0 : ℝ)) ∧
        LipschitzWith L_score recover := by
    refine ⟨fun z => z.1, ?_, hScoreLip⟩
    intro x
    rw [hScore x]
    simp
  simpa using
    (paired_approxOracleRecoversReadouts_of_sharedFeature
      (fstar := fstar)
      (feature := feature)
      (taskReadout := scoreReadout)
      (summaryReadout := summaryReadout)
      (ε_fiber := ε_fiber)
      (ε_task := 0)
      (ε_summary := ε_summary)
      (L_task := L_score)
      (L_summary := L_summary)
      (hApproxRecover := hApproxRecover)
      (hTaskApproxFactor := hScoreApprox)
      (hTaskLip := hScoreWitness)
      (hSummaryApproxFactor := hSummaryApproxFactor)
      (hSummaryLip := hSummaryLip))

end SharedFeatureRoute

section TwoStageLabelScoreRoute

variable [Monoid Strings]
variable {Score Fiber : Type*} [BoundedMetricSpace Score] [BoundedMetricSpace Fiber]
variable {Label : Type*}

instance instBoundedMetricSpaceProdScoreFiber : BoundedMetricSpace (Score × Fiber) where
  toMetricSpace := Prod.metricSpaceMax
  diameterBound :=
    max (BoundedMetricSpace.diameterBound (α := Score))
      (BoundedMetricSpace.diameterBound (α := Fiber))
  diameterBound_pos := by
    exact lt_of_lt_of_le
      (BoundedMetricSpace.diameterBound_pos (α := Score))
      (le_max_left _ _)
  dist_le_diameterBound := by
    intro x y
    rw [Prod.dist_eq]
    exact max_le_max
      (BoundedMetricSpace.dist_le x.1 y.1)
      (BoundedMetricSpace.dist_le x.2 y.2)

/-- Full two-stage label-score decomposition specialized to a learned product
state `Score × FiberState`. This is the theorem-facing form of the
teacher-first + factorized score-fiber route. -/
theorem expected_labelScoreUtility_two_stage_end_to_end_bound_of_productScoreFiber
    (phiHat : Strings → Score × Fiber)
    (feature_ref feature2 featureHat2 : Strings → Score × Fiber)
    (labelOf : (Score × Fiber) → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K₂ : ℝ≥0) (ε₂ : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hR : R ≥ 1)
    (hApprox2 : ApproxTheoremBacked g T phiHat)
    (hApproxRecover2 : ApproxOracleRecoversFeature phiHat feature2 ε₂)
    (hFeatureLip2 : FeatureLipschitzFromOracle phiHat feature2 K₂)
    (hL1 : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L1)
    (hL2 : OracleUtilityLipschitz2 (labelScoreUtility labelOf score) L2)
    (hU : OracleUtilityBoundedAt (labelScoreUtility labelOf score) (feature2 x) U)
    (hbound : ∀ z, D phiHat z x ≤ 1)
    (hbound_global : ∀ w z, D phiHat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g phiHat (p.bind g) ≤ pIdemp g phiHat p) :
    |Exp (ZR g x R T)
        (fun z => labelScoreUtility labelOf score (feature2 z) (featureHat2 x)) -
        labelScoreUtility labelOf score (feature_ref x) (feature_ref x)| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) +
      ((L1 : ℝ) + (L2 : ℝ)) * dist (feature2 x) (feature_ref x) := by
  simpa using
    (expected_labelScoreUtility_two_stage_end_to_end_bound
      (phiHat := phiHat)
      (feature_ref := feature_ref)
      (feature2 := feature2)
      (featureHat2 := featureHat2)
      (labelOf := labelOf)
      (score := score)
      (g := g) (x := x) (R := R) (T := T)
      (K₂ := K₂) (ε₂ := ε₂) (L1 := L1) (L2 := L2) (U := U)
      (hp := hp) (hR := hR)
      (hApprox2 := hApprox2)
      (hApproxRecover2 := hApproxRecover2)
      (hFeatureLip2 := hFeatureLip2)
      (hL1 := hL1) (hL2 := hL2) (hU := hU)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

end TwoStageLabelScoreRoute

end FormalProofs.OPT
