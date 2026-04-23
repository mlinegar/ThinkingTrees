import FormalProofs.OPT.FeatureFiberLaws
import FormalProofs.OPT.ReadoutAlignment

/-!
# FormalProofs/OPT/FeatureClassObjectives.lean

Feature-class objectives for learned theorem features.

The point of this file is modest: it packages a class/objective layer on top of
the existing theorem-backed feature transport results, so the same feature `Φ`
can support both downstream scalar readouts and class-style supervision.

The hard indicator objectives below are intentionally simple. Exact transport is
automatic once the oracle identifies the feature. Approximate transport is
available whenever the chosen class objective admits the same Lipschitz and
boundedness hypotheses already used by
`TheoremBackingApproxMeasurementError.lean`.
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

/-- Hard same-class objective on a feature space. -/
def sameFeatureClassUtility
    {Feature Class : Type*} [DecidableEq Class]
    (classOf : Feature → Class) : OracleUtility2 Feature :=
  fun y y' => if classOf y = classOf y' then 1 else 0

/-- Hard different-class objective on a feature space. -/
def differentFeatureClassUtility
    {Feature Class : Type*} [DecidableEq Class]
    (classOf : Feature → Class) : OracleUtility2 Feature :=
  fun y y' => if classOf y ≠ classOf y' then 1 else 0

section Exact

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Class : Type*} [Encodable Feature] [DecidableEq Class]

/-- Exact theorem-backed reductions preserve same-class feature objectives
whenever the oracle identifies the theorem-bearing feature. -/
theorem expected_sameFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature) :
    Exp (ZR g x R T)
      (fun z => sameFeatureClassUtility classOf (feature z) (feature x)) =
      sameFeatureClassUtility classOf (feature x) (feature x) := by
  simpa [sameFeatureClassUtility] using
    (expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature)
      (u := sameFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover)

/-- Exact theorem-backed reductions preserve different-class feature objectives
whenever the oracle identifies the theorem-bearing feature. -/
theorem expected_differentFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature) :
    Exp (ZR g x R T)
      (fun z => differentFeatureClassUtility classOf (feature z) (feature x)) =
      differentFeatureClassUtility classOf (feature x) (feature x) := by
  simpa [differentFeatureClassUtility] using
    (expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature)
      (u := differentFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover)

end Exact

section Approximate

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Class : Type*} [BoundedPseudoMetricSpace Feature] [DecidableEq Class]

/-- Approximate theorem-backed transport for same-class feature objectives. The
indicator-style objective itself is fixed here; the quantitative assumptions are
the usual Lipschitz/boundedness ones already required by the approximate
feature-transport theorem. -/
theorem expected_sameFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 (sameFeatureClassUtility classOf) L1)
    (hL2 : OracleUtilityLipschitz2 (sameFeatureClassUtility classOf) L2)
    (hU : OracleUtilityBoundedAt (sameFeatureClassUtility classOf) (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T)
        (fun z => sameFeatureClassUtility classOf (feature z) (featureHat x)) -
        sameFeatureClassUtility classOf (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  simpa [sameFeatureClassUtility] using
    (expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
      (fstar := fstar) (feature := feature) (featureHat := featureHat)
      (u := sameFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      (K := K) (L1 := L1) (L2 := L2) (U := U)
      hp hApprox hR hFeatureLip hL1 hL2 hU hbound hbound_global h_mono)

/-- Approximate theorem-backed transport for different-class feature
objectives. -/
theorem expected_differentFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 (differentFeatureClassUtility classOf) L1)
    (hL2 : OracleUtilityLipschitz2 (differentFeatureClassUtility classOf) L2)
    (hU : OracleUtilityBoundedAt (differentFeatureClassUtility classOf) (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T)
        (fun z => differentFeatureClassUtility classOf (feature z) (featureHat x)) -
        differentFeatureClassUtility classOf (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  simpa [differentFeatureClassUtility] using
    (expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
      (fstar := fstar) (feature := feature) (featureHat := featureHat)
      (u := differentFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      (K := K) (L1 := L1) (L2 := L2) (U := U)
      hp hApprox hR hFeatureLip hL1 hL2 hU hbound hbound_global h_mono)

end Approximate

end FormalProofs.OPT
