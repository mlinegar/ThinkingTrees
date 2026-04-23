import FormalProofs.OPT.LabelScoreObjectives
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.OPT.TwoStageDecomposition

/-!
# FormalProofs/OPT/TwoStageLabelScoreObjectives.lean

Two-stage transport theorems specialized to arbitrary scores on decoded labels.

This file makes two routes explicit.

1. **Direct surrogate-oracle route**:
   stage 1 learns a surrogate oracle `f̂` that is uniformly close to the true
   oracle `f*`, and stage 2 learns a tree summary relative to `f̂`.

2. **Layered shared-feature route**:
   stage 1 learns a surrogate theorem feature / oracle `φ̂`, stage 2 learns the
   tree relative to `φ̂`, and downstream task/summary heads evaluate arbitrary
   scores on labels decoded from a learned feature.

The first route captures the simple teacher-first pipeline. The second route is
the more general decomposition that exposes the tradeoff terms explicitly:
transport budget, stage-2 fiber error, measurement error, and stage-1
substitution cost.
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

variable {Strings : Type*} [Monoid Strings]

section DirectSurrogateOracle

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Label : Type*}

/-- Exact theorem-backedness for a learned surrogate oracle yields a direct bound
for arbitrary scores on labels decoded from the true oracle output. -/
theorem expected_trueLabelScoreUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
    (fstar fhat : Strings → Y)
    (labelOf : Y → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L : ℝ≥0) (ε : ℝ≥0)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hL : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L) :
    |Exp (ZR g x R T) (fun z => labelScoreUtility labelOf score (fstar z) (fstar x)) -
        labelScoreUtility labelOf score (fstar x) (fstar x)| ≤
      (L : ℝ) * (2 * (ε : ℝ)) := by
  simpa [labelScoreUtility] using
    (expected_trueOracleUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (u := labelScoreUtility labelOf score)
      (L := L) (hp := hp) (hExact := hExact) (hR := hR)
      (hApprox := hApprox) (hL := hL))

/-- Approximate theorem-backedness for a learned surrogate oracle yields a true
label-score gap bounded by the surrogate transport budget plus the additive
stage-1 surrogate slack. -/
theorem expected_trueLabelScoreUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
    (fstar fhat : Strings → Y)
    (labelOf : Y → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L : ℝ≥0) (ε : ℝ≥0)
    (hp : S T = x)
    (hApproxBacked : ApproxTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hL : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    |Exp (ZR g x R T) (fun z => labelScoreUtility labelOf score (fstar z) (fstar x)) -
        labelScoreUtility labelOf score (fstar x) (fstar x)| ≤
      (L : ℝ) *
        ((hApproxBacked.approxLocalLaws.epsLeaf +
          hApproxBacked.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
        2 * (ε : ℝ)) := by
  simpa [labelScoreUtility] using
    (expected_trueOracleUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (u := labelScoreUtility labelOf score)
      (L := L) (hp := hp) (hApproxBacked := hApproxBacked)
      (hR := hR) (hApprox := hApprox) (hL := hL)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

end DirectSurrogateOracle

section LayeredSharedFeature

variable {Φ : Type*} [BoundedMetricSpace Φ]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]
variable {Label : Type*}

/-- Stage-2 transport in the surrogate feature space, specialized to arbitrary
scores on decoded labels. -/
theorem expected_labelScoreUtility_bound_in_surrogateFeatureSpace
    (phiHat : Strings → Φ)
    (feature2 featureHat2 : Strings → Feature)
    (labelOf : Feature → Label)
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
        labelScoreUtility labelOf score (feature2 x) (feature2 x)| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) := by
  simpa [labelScoreUtility] using
    (stage2_utility_bound_in_phiHat_space
      (phiHat := phiHat)
      (feature2 := feature2) (featureHat2 := featureHat2)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      (K₂ := K₂) (ε₂ := ε₂) (L1 := L1) (L2 := L2) (U := U)
      (hp := hp) (hR := hR) (hApprox2 := hApprox2)
      (hApproxRecover2 := hApproxRecover2)
      (hFeatureLip2 := hFeatureLip2)
      (hL1 := hL1) (hL2 := hL2) (hU := hU)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

/-- Full two-stage end-to-end decomposition for arbitrary label-score
objectives. This is the general teacher-first route with an explicit stage-1
substitution term. -/
theorem expected_labelScoreUtility_two_stage_end_to_end_bound
    (phiHat : Strings → Φ)
    (feature_ref : Strings → Feature)
    (feature2 featureHat2 : Strings → Feature)
    (labelOf : Feature → Label)
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
  simpa [labelScoreUtility] using
    (two_stage_full_end_to_end_bound
      (phiHat := phiHat)
      (feature_ref := feature_ref)
      (feature2 := feature2) (featureHat2 := featureHat2)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      (K₂ := K₂) (ε₂ := ε₂) (L1 := L1) (L2 := L2) (U := U)
      (hp := hp) (hR := hR) (hApprox2 := hApprox2)
      (hApproxRecover2 := hApproxRecover2)
      (hFeatureLip2 := hFeatureLip2)
      (hL1 := hL1) (hL2 := hL2) (hU := hU)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

end LayeredSharedFeature

end FormalProofs.OPT
