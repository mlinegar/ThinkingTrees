import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.OracleUtility

/-!
# FormalProofs/OPT/TheoremBackingMeasurementError.lean

Bridge the exact theorem-backed regime to latent-state and measurement-error views.

The key additional assumption is that a latent state / feature is **identified by
the oracle**: zero oracle distortion implies the same latent state. Under that
assumption:

1. exact theorem-backedness for the oracle implies exact transport for any loss
   or preference program indexed by the latent state;
2. exact latent-state utility preservation is recovered as a corollary; and
3. if the truth-state is observed through a noisy proxy, the only remaining gap
   is a pure measurement-error term.
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

open Set

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

/-- The oracle identifies a latent state exactly if oracle-equality forces
latent-state equality. -/
def OracleRecoversFeature
    {Feature : Type*}
    (fstar : Strings → Y) (feature : Strings → Feature) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → feature x = feature x'

section EncodedFeature

variable {Feature α : Type*} [Encodable Feature]

/-- If the oracle identifies a feature exactly, then zero oracle distortion
implies zero distortion for the encoded-feature oracle. -/
lemma encodedOracle_zero_of_oracleRecoversFeature
    {fstar : Strings → Y} {feature : Strings → Feature}
    (hRecover : OracleRecoversFeature fstar feature)
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    dist ((encodedOracle (Strings := Strings) feature) x)
      ((encodedOracle (Strings := Strings) feature) x') = 0 := by
  have hEq : feature x = feature x' := hRecover x x' hzero
  apply dist_eq_zero.mpr
  simp [encodedOracle, hEq]

/-- Exact theorem-backedness for `fstar`, plus exact oracle recovery of a latent
feature, implies exact transport for any loss oracle-measurable with respect to
that latent feature. -/
theorem expected_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (loss : Strings → α → ℝ)
    (gen : Strings → PMF α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (h_meas : OracleMeasurableLossGeneric loss (encodedOracle (Strings := Strings) feature))
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric loss (PMF.pure x) gen =
    ExpectedLossGeneric loss (ZR g x R T) gen := by
  exact expected_loss_eq_of_zero_dist_generic
    (fstar := encodedOracle (Strings := Strings) feature)
    (loss := loss)
    (gen := gen)
    (μ_X := PMF.pure x)
    (μ_Z := ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact encodedOracle_zero_of_oracleRecoversFeature hRecover
        (zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz))
    h_meas
    h_gen

/-- Preference-program version of
`expected_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature`. -/
theorem expected_pref_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (loss : PrefLoss Strings α)
    (gen : PrefGen Strings α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (h_meas : OracleMeasurablePrefLoss loss (encodedOracle (Strings := Strings) feature))
    (h_gen : OracleIndexedGenComb gen (encodedOracle (Strings := Strings) feature)) :
    ExpectedPrefLoss loss (PMF.pure x) gen =
    ExpectedPrefLoss loss (ZR g x R T) gen := by
  exact expected_pref_loss_eq_of_zero_dist
    (fstar := encodedOracle (Strings := Strings) feature)
    (loss := loss)
    (gen := gen)
    (μ_X := PMF.pure x)
    (μ_Z := ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact encodedOracle_zero_of_oracleRecoversFeature hRecover
        (zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz))
    h_meas
    h_gen

/-- Feature-indexed objectives are preserved exactly whenever the feature is
identified by the oracle and the reduction is exact theorem-backed. -/
theorem featureIndexedObjective_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (objective : Feature → α → ℝ)
    (gen : Strings → PMF α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective) (PMF.pure x) gen =
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective) (ZR g x R T) gen := by
  exact expected_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar := fstar)
    (feature := feature)
    (loss := featureIndexedObjective (Strings := Strings) feature objective)
    (gen := gen)
    (g := g) (x := x) (R := R) (T := T)
    hp hExact hR hRecover
    (oracleMeasurableLossGeneric_of_featureIndexedObjective
      (Strings := Strings) (feature := feature) (objective := objective))
    h_gen

/-- Direct supervised-state learning is preserved exactly whenever the latent
state is identified by the oracle. -/
theorem supervisedStateExpectedLoss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (loss : Feature → Feature → ℝ)
    (gen : Strings → PMF Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) feature loss) (PMF.pure x) gen =
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) feature loss) (ZR g x R T) gen := by
  simpa [supervisedStateLoss, featureIndexedObjective] using
    (featureIndexedObjective_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar)
      (feature := feature)
      (objective := fun y a => loss a y)
      (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover h_gen)

end EncodedFeature

section MeasurementError

variable {Feature : Type*} [Encodable Feature]

/-- Constant unit generator used to read scalar expectations through the generic
expected-loss interface without universe headaches from `PUnit`. -/
def unitExampleGenerator (Strings : Type*) : Strings → PMF Unit :=
  fun _ => PMF.pure ()

/-- The constant unit generator is oracle-indexed for every oracle. -/
lemma oracleIndexedGenGeneric_unitExampleGenerator
    (fstar : Strings → ℝ) :
    OracleIndexedGenGeneric
      (unitExampleGenerator Strings) fstar := by
  intro x x' hdist
  simp [unitExampleGenerator]

/-- Trivial-generator expected loss is just expectation of the corresponding
document score. -/
lemma expectedLossGeneric_unitExampleGenerator_eq_Exp
    (μ : PMF Strings) (loss : Strings → Unit → ℝ) :
    ExpectedLossGeneric loss μ (unitExampleGenerator Strings) =
      Exp μ (fun x => loss x ()) := by
  unfold ExpectedLossGeneric Exp unitExampleGenerator
  refine tsum_congr ?_
  intro x
  simp

/-- Exact theorem-backedness plus exact oracle recovery implies exact preservation
of latent-state utilities evaluated against the true latent state. -/
theorem expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature) :
    Exp (ZR g x R T) (fun z => u (feature z) (feature x)) =
      u (feature x) (feature x) := by
  have h_eq :=
    featureIndexedObjective_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar)
      (feature := feature)
      (objective := fun y (_ : Unit) => u y (feature x))
      (gen := unitExampleGenerator Strings)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover
      (oracleIndexedGenGeneric_unitExampleGenerator
        (fstar := encodedOracle (Strings := Strings) feature))
  rw [expectedLossGeneric_unitExampleGenerator_eq_Exp,
      expectedLossGeneric_unitExampleGenerator_eq_Exp,
      Exp_pure] at h_eq
  simpa [featureIndexedObjective] using h_eq.symm

/-- If the latent state is identified exactly by the oracle, then noisy
observation of the truth-state appears purely as a measurement-error term. The
transport term vanishes. -/
theorem expected_feature_utility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    [PseudoMetricSpace Feature]
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (hL : OracleUtilityLipschitz2 u L)
    (hU : OracleUtilityBoundedAt u (feature x) U) :
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)| ≤
      (L : ℝ) * dist (featureHat x) (feature x) := by
  have h_noise :
      |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
          Exp (ZR g x R T) (fun z => u (feature z) (feature x))| ≤
        (L : ℝ) * dist (featureHat x) (feature x) :=
    expected_utility_noise_bound_pmf_bounded
      (p := ZR g x R T) (x := x)
      (fstar := feature) (fhat := featureHat)
      (u := u) (L := L) (U := U)
      hL hU
  have h_exact :
      Exp (ZR g x R T) (fun z => u (feature z) (feature x)) =
        u (feature x) (feature x) :=
    expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature) (u := u)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover
  simpa [h_exact] using h_noise

end MeasurementError

end FormalProofs.OPT
