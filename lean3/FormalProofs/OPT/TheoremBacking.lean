import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.OracleUtility
import FormalProofs.OPT.ApproximateLocalLaws

/-!
# FormalProofs/OPT/TheoremBacking.lean

Pure consolidation pass (2026-07-02). No proof, statement, name, or docstring
content was edited; source-file content is included verbatim, each chunk wrapped
in an anonymous `section ... end` so file-scope `open scoped` / `set_option`
lines cannot leak between chunks.

The consolidation cluster was planned as four files:

- `FormalProofs/OPT/TheoremBackingAssumptions.lean`
- `FormalProofs/OPT/TheoremBackingStructure.lean`
- `FormalProofs/OPT/TheoremBackingMeasurementError.lean`
- `FormalProofs/OPT/TheoremBackingApproxMeasurementError.lean`

This file consolidates the upper two:

- `FormalProofs/OPT/TheoremBackingMeasurementError.lean`
- `FormalProofs/OPT/TheoremBackingApproxMeasurementError.lean`

`TheoremBackingAssumptions.lean` and `TheoremBackingStructure.lean` keep their
original content and are NOT consolidated here, because
`FormalProofs/OPT/TheoremBackingConsequences.lean` (out of scope for this pass)
imports `TheoremBackingStructure` while `TheoremBackingMeasurementError` imports
`TheoremBackingConsequences`: folding all four into one module would force an
import cycle `TheoremBacking → TheoremBackingConsequences →
TheoremBackingStructure (shim) → TheoremBacking`. Since this file imports
`TheoremBackingConsequences`, it transitively re-exports the full cluster
(`Assumptions`, `Structure`, `Consequences`, and the two chunks below), so
importers of any of the four modules can be retargeted to this module in the
follow-up pass. `TheoremBackingConsequences` itself must never be retargeted to
import this module.
-/

/-! ## From TheoremBackingMeasurementError.lean -/

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

section

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

end -- closes the source file's `noncomputable section`

end -- closes the consolidation wrapper `section`

/-! ## From TheoremBackingApproxMeasurementError.lean -/

/-!
# FormalProofs/OPT/TheoremBackingApproxMeasurementError.lean

Approximate theorem-backed transport for latent-state utilities with measurement
error.

This file formalizes the next step after the exact measurement-error bridge:

1. the summarizer is only approximately theorem-backed for an oracle `fstar`;
2. the latent state / feature is Lipschitz with respect to that oracle; and
3. utilities on the latent state may also be evaluated against a noisy state
   proxy.

The resulting bound decomposes into:

- a transport term inherited from approximate theorem-backedness, pushed through
  the feature-Lipschitz constant; and
- a pure measurement-error term on the noisy state proxy.
-/

section

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
variable {Y : Type*} [BoundedMetricSpace Y]

/-- A latent state / feature map is Lipschitz through the oracle if feature
distance is controlled by oracle distance. -/
def FeatureLipschitzFromOracle
    {Feature : Type*} [BoundedPseudoMetricSpace Feature]
    (fstar : Strings → Y) (feature : Strings → Feature) (K : ℝ≥0) : Prop :=
  ∀ x x', dist (feature x) (feature x') ≤ (K : ℝ) * dist (fstar x) (fstar x')

section FeatureTransport

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- If a latent feature is Lipschitz through the oracle, expected feature
distortion is controlled by expected oracle distortion. -/
theorem feature_distortion_le_of_featureLipschitzFromOracle
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (feature : Strings → Feature)
    (K : ℝ≥0)
    (hLip : FeatureLipschitzFromOracle fstar feature K) :
    Exp p (fun z => D feature z x) ≤ (K : ℝ) * Exp p (fun z => D fstar z x) := by
  let M_feature : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Feature)
  have hM_feature : 0 ≤ M_feature := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Feature)
  have hbound_feature : ∀ z, D feature z x ≤ M_feature := by
    intro z
    unfold D M_feature
    exact BoundedPseudoMetricSpace.dist_le (feature z) (feature x)
  have hsum_feature :
      Summable (fun z => (p z).toReal * D feature z x) :=
    summable_D_of_bounded p feature x M_feature hM_feature hbound_feature
  let M_oracle : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM_oracle : 0 ≤ M_oracle := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have hbound_oracle : ∀ z, D fstar z x ≤ M_oracle := by
    intro z
    unfold D M_oracle
    exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
  have hsum_oracle :
      Summable (fun z => (p z).toReal * D fstar z x) :=
    summable_D_of_bounded p fstar x M_oracle hM_oracle hbound_oracle
  have hsum_scaled :
      Summable (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) := by
    have hEq :
        (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) =
          (fun z => (K : ℝ) * ((p z).toReal * D fstar z x)) := by
      funext z
      ring
    rw [hEq]
    exact hsum_oracle.mul_left (K : ℝ)
  have hmono :
      Exp p (fun z => D feature z x) ≤
        Exp p (fun z => (K : ℝ) * D fstar z x) :=
    Exp_mono' p
      (fun z => D feature z x)
      (fun z => (K : ℝ) * D fstar z x)
      (fun z => by
        unfold D
        simpa using hLip z x)
      hsum_feature
      hsum_scaled
  calc
    Exp p (fun z => D feature z x) ≤ Exp p (fun z => (K : ℝ) * D fstar z x) := hmono
    _ = (K : ℝ) * Exp p (fun z => D fstar z x) := by
      unfold Exp
      have hEq :
          (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) =
            (fun z => (K : ℝ) * ((p z).toReal * D fstar z x)) := by
        funext z
        ring
      rw [hEq, tsum_mul_left]

/-- Approximate theorem-backedness on the oracle, pushed through a feature map
that is Lipschitz in the oracle, yields a latent-state utility bound with an
additive measurement-error term. -/
theorem expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  let budget : ℝ :=
    hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
      ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp
  let M_feature : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Feature)
  have hM_feature : 0 ≤ M_feature := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Feature)
  have hbound_feature : ∀ z, D feature z x ≤ M_feature := by
    intro z
    unfold D M_feature
    exact BoundedPseudoMetricSpace.dist_le (feature z) (feature x)
  have hD_feature :
      Summable (fun z => (ZR g x R T z).toReal * D feature z x) :=
    summable_D_of_bounded (ZR g x R T) feature x M_feature hM_feature hbound_feature
  have h_noise_transport :
      |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
          u (feature x) (feature x)| ≤
        (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) :=
    expected_utility_bound_with_noise_ZR
      (g := g) (T := T) (x := x) (R := R)
      (fstar := feature) (fhat := featureHat)
      (u := u) (L1 := L1) (L2 := L2) (U := U)
      hL1 hL2 hU hD_feature
  have h_feature_dist :
      Exp (ZR g x R T) (fun z => D feature z x) ≤
        (K : ℝ) * Δ_R_ZR g x R T fstar := by
    simpa [Δ_R_ZR] using
      (feature_distortion_le_of_featureLipschitzFromOracle
        (p := ZR g x R T) (x := x) (fstar := fstar) (feature := feature)
        (K := K) hFeatureLip)
  have h_budget :
      Δ_R_ZR g x R T fstar ≤ budget :=
    Δ_R_ZR_le_of_approx_bundle g T fstar x R hp hR hbound hbound_global h_mono
      hApprox.approxLocalLaws
  have h_transport :
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) ≤
        (L1 : ℝ) * (K : ℝ) * budget := by
    have hL1_nonneg : 0 ≤ (L1 : ℝ) := by exact_mod_cast L1.property
    have hK_nonneg : 0 ≤ (K : ℝ) := by exact_mod_cast K.property
    calc
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x)
          ≤ (L1 : ℝ) * ((K : ℝ) * Δ_R_ZR g x R T fstar) := by
              exact mul_le_mul_of_nonneg_left h_feature_dist hL1_nonneg
      _ ≤ (L1 : ℝ) * ((K : ℝ) * budget) := by
              apply mul_le_mul_of_nonneg_left
              exact mul_le_mul_of_nonneg_left h_budget hK_nonneg
              exact hL1_nonneg
      _ = (L1 : ℝ) * (K : ℝ) * budget := by ring
  calc
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)| ≤
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := h_noise_transport
    _ ≤ (L1 : ℝ) * (K : ℝ) * budget +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          exact add_le_add h_transport le_rfl

end FeatureTransport

end FormalProofs.OPT

end -- closes the source file's `noncomputable section`

end -- closes the consolidation wrapper `section`
