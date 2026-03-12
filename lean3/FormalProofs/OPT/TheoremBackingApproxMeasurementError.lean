import FormalProofs.OPT.TheoremBackingMeasurementError
import FormalProofs.OPT.ApproximateLocalLaws

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
