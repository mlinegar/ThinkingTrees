import FormalProofs.OPT.TheoremBackingMeasurementError
import FormalProofs.OPT.TheoremBackingApproxMeasurementError

/-!
# FormalProofs/OPT/LabelScoreObjectives.lean

Generic score objectives on decoded labels.

The existing `OracleUtility2` machinery already supports arbitrary real-valued
objectives on a theorem-bearing feature. This file packages the common and more
interpretable special case where a learned theorem feature `Φ` is first decoded
to a label and then evaluated by an arbitrary score function on label pairs.

This is the right abstraction for downstream settings where the oracle exposes
labels or scores over labels rather than only same/different class structure.
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

/-- A generic label-score objective obtained by decoding a label from the
theorem-bearing feature and then scoring the resulting label pair. -/
def labelScoreUtility
    {Feature Label : Type*}
    (labelOf : Feature → Label) (score : Label → Label → ℝ) :
    OracleUtility2 Feature :=
  fun y y' => score (labelOf y) (labelOf y')

section Exact

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Label : Type*} [Encodable Feature]

/-- Exact theorem-backed transport for arbitrary scores on decoded labels. -/
theorem expected_labelScoreUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (labelOf : Feature → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature) :
    Exp (ZR g x R T)
      (fun z => labelScoreUtility labelOf score (feature z) (feature x)) =
      labelScoreUtility labelOf score (feature x) (feature x) := by
  simpa [labelScoreUtility] using
    (expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover)

section ExactMeasurementError

variable [PseudoMetricSpace Feature]

/-- Exact theorem-backed transport for arbitrary label-score objectives with a
noisy observation of the truth feature. -/
theorem expected_labelScoreUtility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (labelOf : Feature → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (hL : OracleUtilityLipschitz2 (labelScoreUtility labelOf score) L)
    (hU : OracleUtilityBoundedAt (labelScoreUtility labelOf score) (feature x) U) :
    |Exp (ZR g x R T)
        (fun z => labelScoreUtility labelOf score (feature z) (featureHat x)) -
        labelScoreUtility labelOf score (feature x) (feature x)| ≤
      (L : ℝ) * dist (featureHat x) (feature x) := by
  simpa [labelScoreUtility] using
    (expected_feature_utility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature) (featureHat := featureHat)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      (L := L) (U := U)
      hp hExact hR hRecover hL hU)

end ExactMeasurementError
end Exact

section Approximate

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Label : Type*} [BoundedPseudoMetricSpace Feature]

/-- Approximate theorem-backed transport for arbitrary scores on decoded
labels, under the same feature-Lipschitz and utility regularity conditions as
the general latent-feature transport theorem. -/
theorem expected_labelScoreUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (labelOf : Feature → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L1)
    (hL2 : OracleUtilityLipschitz2 (labelScoreUtility labelOf score) L2)
    (hU : OracleUtilityBoundedAt (labelScoreUtility labelOf score) (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T)
        (fun z => labelScoreUtility labelOf score (feature z) (featureHat x)) -
        labelScoreUtility labelOf score (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  simpa [labelScoreUtility] using
    (expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
      (fstar := fstar) (feature := feature) (featureHat := featureHat)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      (K := K) (L1 := L1) (L2 := L2) (U := U)
      hp hApprox hR hFeatureLip hL1 hL2 hU hbound hbound_global h_mono)

end Approximate

end FormalProofs.OPT
