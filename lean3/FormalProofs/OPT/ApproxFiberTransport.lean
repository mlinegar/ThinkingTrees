import FormalProofs.OPT.ApproxOracleRecovery
import FormalProofs.OPT.TheoremBackingApproxMeasurementError
import FormalProofs.OPT.LipschitzReadoutFactorization
import FormalProofs.OPT.TwoStageOracleSurrogate

/-!
# FormalProofs/OPT/ApproxFiberTransport.lean

Combined transport bounds when both theorem-backedness and feature recovery
are approximate.

This file unifies three independent error sources into a single additive bound:

1. **Transport budget** from `ApproxLocalLawsBundle` — how well g satisfies
   the local laws L1/L2/L3.
2. **Fiber error** from `ApproxOracleRecoversFeature` — how well the learned
   feature φ captures f*-equivalence classes.
3. **Readout error** from `ApproxReadoutFactorsThroughFeature` — how well
   downstream heads factor through φ.

The main theorem `expected_utility_bound_approx_fiber` gives:

  |E[u(φ(Z), φ̂(x))] - u(φ(x), φ(x))| ≤
      L₁ · K · (transport_budget)       -- from approximate local laws
    + L₁ · ε_fiber                       -- from approximate fiber preservation
    + L₂ · dist(φ̂(x), φ(x))            -- from measurement error

The fiber error term is the new contribution; the other two terms match the
existing `TheoremBackingApproxMeasurementError` bound structure.
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

section ExactBackingApproxFiber

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Exact theorem-backed transport with approximate fiber recovery.

When the summarizer is exactly theorem-backed but the feature map only
approximately preserves fibers, the utility gap is bounded by the Lipschitz
constant times the fiber error plus measurement error.

This is complementary to the existing
`expected_feature_utility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature`
which requires exact `OracleRecoversFeature`. -/
theorem expected_utility_bound_exactBacked_approxFiber
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (ε_fiber : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature x) U) :
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)| ≤
      (L1 : ℝ) * (ε_fiber : ℝ) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
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
  -- The key insight: with exact theorem-backing, every z in ZR support has
  -- dist(f*(z), f*(x)) = 0, so approximate fiber recovery gives
  -- dist(feature(z), feature(x)) ≤ ε_fiber for each z.
  have h_fiber_bound :
      Exp (ZR g x R T) (fun z => D feature z x) ≤ (ε_fiber : ℝ) := by
    exact Exp_le_const_of_support
      (p := ZR g x R T)
      (f := fun z => D feature z x)
      (c := (ε_fiber : ℝ))
      (M := M_feature)
      (hc := by exact_mod_cast ε_fiber.property)
      (hM := hM_feature)
      (hsupport := by
        intro z hz
        unfold D
        exact hApproxRecover z x
          (zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz))
      (hf_nonneg := by
        intro z
        exact dist_nonneg)
      (hf_bound := hbound_feature)
  have hmul_fiber :
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) ≤
        (L1 : ℝ) * (ε_fiber : ℝ) := by
    exact mul_le_mul_of_nonneg_left h_fiber_bound (by exact_mod_cast L1.property)
  calc
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)|
      ≤ (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := h_noise_transport
    _ ≤ (L1 : ℝ) * (ε_fiber : ℝ) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          linarith

end ExactBackingApproxFiber

section FullApprox

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Main theorem: combined transport bound with approximate theorem-backing,
approximate fiber preservation, and measurement error.

The three error terms decompose additively:
- L₁ · K · (leaf + merge + (R-1)·idemp)  from approximate local laws
- L₁ · ε_fiber                            from approximate fiber preservation
- L₂ · dist(φ̂(x), φ(x))                  from measurement error

This extends `expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz`
with the additional fiber-error term. When ε_fiber = 0 (exact fiber recovery),
the first and third terms match the existing bound. -/
theorem expected_utility_bound_approx_fiber
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K : ℝ≥0) (ε_fiber : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
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
      (L1 : ℝ) * (ε_fiber : ℝ) +
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
  -- Step 1: Split into transport + noise terms
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
  -- Step 2: Bound feature distortion using Lipschitz + fiber error
  -- D(feature, z, x) ≤ K · D(fstar, z, x) + ε_fiber
  -- (triangle: if D(fstar,z,x) = 0 then ε_fiber; otherwise K · D(fstar,z,x))
  have h_feature_pointwise :
      ∀ z, D feature z x ≤ (K : ℝ) * D fstar z x + (ε_fiber : ℝ) := by
    intro z
    unfold D
    by_cases h : dist (fstar z) (fstar x) = 0
    · calc dist (feature z) (feature x)
          ≤ (ε_fiber : ℝ) := hApproxRecover z x h
        _ ≤ (K : ℝ) * dist (fstar z) (fstar x) + (ε_fiber : ℝ) := by
            linarith [mul_nonneg (show 0 ≤ (K : ℝ) from by exact_mod_cast K.property)
              (show 0 ≤ dist (fstar z) (fstar x) from dist_nonneg)]
    · calc dist (feature z) (feature x)
          ≤ (K : ℝ) * dist (fstar z) (fstar x) := hFeatureLip z x
        _ ≤ (K : ℝ) * dist (fstar z) (fstar x) + (ε_fiber : ℝ) := by
            exact le_add_of_nonneg_right (by exact_mod_cast ε_fiber.property)
  -- Step 3: Take expectations
  let M_oracle : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM_oracle : 0 ≤ M_oracle := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have hbound_oracle : ∀ z, D fstar z x ≤ M_oracle := by
    intro z
    unfold D M_oracle
    exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
  have hD_oracle :
      Summable (fun z => (ZR g x R T z).toReal * D fstar z x) :=
    summable_D_of_bounded (ZR g x R T) fstar x M_oracle hM_oracle hbound_oracle
  have hKD_summable :
      Summable (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x)) := by
    have hEq :
        (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x)) =
          (fun z => (K : ℝ) * ((ZR g x R T z).toReal * D fstar z x)) := by
      funext z
      ring
    rw [hEq]
    exact hD_oracle.mul_left (K : ℝ)
  have hconst_summable :
      Summable (fun z => (ZR g x R T z).toReal * (ε_fiber : ℝ)) :=
    PMF.summable_coe_real_mul_of_bounded
      (ZR g x R T) (fun _ => (ε_fiber : ℝ)) (ε_fiber : ℝ)
      (by exact_mod_cast ε_fiber.property)
      (fun _ => by
        have hε : 0 ≤ (ε_fiber : ℝ) := by exact_mod_cast ε_fiber.property
        simp [abs_of_nonneg hε])
  have hsum_stage :
      Summable
        (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x + (ε_fiber : ℝ))) := by
    have hEq :
        (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x + (ε_fiber : ℝ))) =
          (fun z =>
            (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x) +
              (ZR g x R T z).toReal * (ε_fiber : ℝ)) := by
      funext z
      ring
    rw [hEq]
    exact Summable.add hKD_summable hconst_summable
  have h_Exp_bound :
      Exp (ZR g x R T) (fun z => D feature z x) ≤
        (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ) := by
    calc Exp (ZR g x R T) (fun z => D feature z x)
        ≤ Exp (ZR g x R T) (fun z => (K : ℝ) * D fstar z x + (ε_fiber : ℝ)) := by
          apply Exp_mono'
          · intro z
            exact h_feature_pointwise z
          · exact hD_feature
          · exact hsum_stage
      _ = (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ) := by
          rw [Exp_add (ZR g x R T)
            (fun z => (K : ℝ) * D fstar z x)
            (fun _ => (ε_fiber : ℝ))
            hKD_summable
            hconst_summable]
          have hExp_scale :
              Exp (ZR g x R T) (fun z => (K : ℝ) * D fstar z x) =
                (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) := by
            unfold Exp
            have hEq :
                (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x)) =
                  (fun z => (K : ℝ) * ((ZR g x R T z).toReal * D fstar z x)) := by
              funext z
              ring
            rw [hEq, tsum_mul_left]
          rw [hExp_scale, Exp_const]
  -- Step 4: Use approximate local law budget
  have h_budget :
      Δ_R_ZR g x R T fstar ≤ budget :=
    Δ_R_ZR_le_of_approx_bundle g T fstar x R hp hR hbound hbound_global h_mono
      hApprox.approxLocalLaws
  -- Step 5: Combine
  have hL1_nonneg : 0 ≤ (L1 : ℝ) := by exact_mod_cast L1.property
  have hK_nonneg : 0 ≤ (K : ℝ) := by exact_mod_cast K.property
  have hmul_exp :
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) ≤
        (L1 : ℝ) * ((K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ)) := by
    exact mul_le_mul_of_nonneg_left h_Exp_bound hL1_nonneg
  have hbudget_scaled :
      (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) ≤ (K : ℝ) * budget := by
    apply mul_le_mul_of_nonneg_left
    simpa [Δ_R_ZR, budget] using h_budget
    exact hK_nonneg
  have hmul_budget :
      (L1 : ℝ) * ((K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ)) ≤
        (L1 : ℝ) * ((K : ℝ) * budget + (ε_fiber : ℝ)) := by
    apply mul_le_mul_of_nonneg_left
    linarith
    exact hL1_nonneg
  calc
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)|
      ≤ (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := h_noise_transport
    _ ≤ (L1 : ℝ) * ((K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ)) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          linarith
    _ ≤ (L1 : ℝ) * ((K : ℝ) * budget + (ε_fiber : ℝ)) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          linarith
    _ = (L1 : ℝ) * (K : ℝ) * budget +
        (L1 : ℝ) * (ε_fiber : ℝ) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by ring

end FullApprox

end FormalProofs.OPT
