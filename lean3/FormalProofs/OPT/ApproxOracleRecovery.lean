import FormalProofs.OPT.TheoremBackingMeasurementError
import FormalProofs.OPT.ReadoutAlignment

/-!
# FormalProofs/OPT/ApproxOracleRecovery.lean

Approximate oracle recovery: ε-relaxation of `OracleRecoversFeature`.

The exact predicate `OracleRecoversFeature` requires that zero oracle distortion
implies *propositional equality* of the latent feature. In practice, a learned
feature map φ may only approximately satisfy this: oracle-equivalent inputs are
mapped to nearby (but not identical) feature vectors.

This file introduces `ApproxOracleRecoversFeature`, which replaces equality with
an ε-ball condition in a metric on the Feature space. The two key results are:

1. **Zero reduction**: ε = 0 recovers exact `OracleRecoversFeature` when
   Feature is a metric space (T0 separation).
2. **Lipschitz composition**: if a readout is Lipschitz in the feature, then
   ε-approximate feature recovery implies (L·ε)-approximate readout recovery.

This is complementary to `FeatureLipschitzFromOracle` in
`TheoremBackingApproxMeasurementError.lean`, which controls feature distance
when oracle distance is *nonzero*. Here we control feature distance when oracle
distance is *zero* but the learned feature is imperfect.
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
variable {Y : Type*} [BoundedMetricSpace Y]

/-- ε-approximate oracle recovery: when oracle distance is zero, feature distance
is at most ε. This is the metric relaxation of `OracleRecoversFeature`. -/
def ApproxOracleRecoversFeature
    {Feature : Type*} [PseudoMetricSpace Feature]
    (fstar : Strings → Y) (feature : Strings → Feature) (ε : ℝ≥0) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → dist (feature x) (feature x') ≤ (ε : ℝ)

/-- Exact oracle recovery restricted to a covered pair relation. This is the
right interface when only some oracle-equivalent pairs are labeled. -/
def OracleRecoversFeatureOn
    {Feature : Type*}
    (covered : Strings → Strings → Prop)
    (fstar : Strings → Y) (feature : Strings → Feature) : Prop :=
  ∀ x x', covered x x' → dist (fstar x) (fstar x') = 0 → feature x = feature x'

/-- Approximate oracle recovery restricted to a covered pair relation. -/
def ApproxOracleRecoversFeatureOn
    {Feature : Type*} [PseudoMetricSpace Feature]
    (covered : Strings → Strings → Prop)
    (fstar : Strings → Y) (feature : Strings → Feature) (ε : ℝ≥0) : Prop :=
  ∀ x x', covered x x' → dist (fstar x) (fstar x') = 0 →
    dist (feature x) (feature x') ≤ (ε : ℝ)

section Reduction

variable {Feature : Type*} [MetricSpace Feature]

/-- ε = 0 approximate recovery implies exact `OracleRecoversFeature`. -/
theorem approxOracleRecoversFeature_zero_implies_oracleRecoversFeature
    {fstar : Strings → Y} {feature : Strings → Feature}
    (h : ApproxOracleRecoversFeature fstar feature 0) :
    OracleRecoversFeature fstar feature := by
  intro x x' hzero
  have hdist : dist (feature x) (feature x') = 0 := by
    simpa using h x x' hzero
  exact dist_eq_zero.mp hdist

/-- Exact `OracleRecoversFeature` implies ε-approximate recovery for any ε. -/
theorem oracleRecoversFeature_implies_approxOracleRecoversFeature
    {fstar : Strings → Y} {feature : Strings → Feature}
    (h : OracleRecoversFeature fstar feature) (ε : ℝ≥0) :
    ApproxOracleRecoversFeature fstar feature ε := by
  intro x x' hzero
  have hEq := h x x' hzero
  rw [hEq]
  simpa using ε.property

end Reduction

section RestrictedRecovery

variable {Feature : Type*}

/-- Global exact recovery implies exact recovery on any covered relation. -/
theorem oracleRecoversFeature_implies_on
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature}
    (h : OracleRecoversFeature fstar feature) :
    OracleRecoversFeatureOn covered fstar feature := by
  intro x x' _ hzero
  exact h x x' hzero

end RestrictedRecovery

section Monotonicity

variable {Feature : Type*} [PseudoMetricSpace Feature]

/-- Approximate recovery is monotone in ε: tighter guarantees imply looser ones. -/
theorem approxOracleRecoversFeature_mono
    {fstar : Strings → Y} {feature : Strings → Feature}
    {ε₁ ε₂ : ℝ≥0} (hle : ε₁ ≤ ε₂)
    (h : ApproxOracleRecoversFeature fstar feature ε₁) :
    ApproxOracleRecoversFeature fstar feature ε₂ := by
  intro x x' hzero
  calc dist (feature x) (feature x') ≤ (ε₁ : ℝ) := h x x' hzero
    _ ≤ (ε₂ : ℝ) := by exact_mod_cast hle

end Monotonicity

section RestrictedApproxRecovery

variable {Feature : Type*} [PseudoMetricSpace Feature]

/-- Global approximate recovery implies approximate recovery on any covered
relation. -/
theorem approxOracleRecoversFeature_implies_on
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature} {ε : ℝ≥0}
    (h : ApproxOracleRecoversFeature fstar feature ε) :
    ApproxOracleRecoversFeatureOn covered fstar feature ε := by
  intro x x' _ hzero
  exact h x x' hzero

/-- Restricted approximate recovery is monotone in ε. -/
theorem approxOracleRecoversFeatureOn_mono
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature}
    {ε₁ ε₂ : ℝ≥0} (hle : ε₁ ≤ ε₂)
    (h : ApproxOracleRecoversFeatureOn covered fstar feature ε₁) :
    ApproxOracleRecoversFeatureOn covered fstar feature ε₂ := by
  intro x x' hCovered hzero
  calc dist (feature x) (feature x') ≤ (ε₁ : ℝ) := h x x' hCovered hzero
    _ ≤ (ε₂ : ℝ) := by exact_mod_cast hle

end RestrictedApproxRecovery

section RestrictedReduction

variable {Feature : Type*} [MetricSpace Feature]

/-- Zero restricted approximate recovery implies restricted exact recovery. -/
theorem approxOracleRecoversFeatureOn_zero_implies_oracleRecoversFeatureOn
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature}
    (h : ApproxOracleRecoversFeatureOn covered fstar feature 0) :
    OracleRecoversFeatureOn covered fstar feature := by
  intro x x' hCovered hzero
  have hdist : dist (feature x) (feature x') = 0 := by
    simpa using h x x' hCovered hzero
  exact dist_eq_zero.mp hdist

/-- Restricted exact recovery implies restricted approximate recovery for any
ε ≥ 0. -/
theorem oracleRecoversFeatureOn_implies_approxOracleRecoversFeatureOn
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature}
    (h : OracleRecoversFeatureOn covered fstar feature) (ε : ℝ≥0) :
    ApproxOracleRecoversFeatureOn covered fstar feature ε := by
  intro x x' hCovered hzero
  have hEq := h x x' hCovered hzero
  rw [hEq]
  simpa using ε.property

end RestrictedReduction

section Composition

variable {Feature Readout : Type*}
variable [PseudoMetricSpace Feature] [PseudoMetricSpace Readout]

/-- Lipschitz readout composed with ε-approximate feature recovery gives
(L · ε)-approximate readout recovery. -/
theorem approxOracleRecoversReadout_of_approxRecover_and_lipschitzRecover
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε : ℝ≥0} {L : ℝ≥0}
    (hRecover : ApproxOracleRecoversFeature fstar feature ε)
    (hFactor : ∃ recover : Feature → Readout,
      (∀ x : Strings, readout x = recover (feature x)) ∧
      LipschitzWith L recover) :
    ApproxOracleRecoversFeature fstar readout (L * ε) := by
  rcases hFactor with ⟨recover, hRecoverEq, hLip⟩
  intro x x' hzero
  rw [hRecoverEq x, hRecoverEq x']
  have h_feat_dist := hRecover x x' hzero
  calc dist (recover (feature x)) (recover (feature x'))
      ≤ (L : ℝ) * dist (feature x) (feature x') := hLip.dist_le_mul _ _
    _ ≤ (L : ℝ) * (ε : ℝ) := by
        apply mul_le_mul_of_nonneg_left h_feat_dist
        exact_mod_cast L.property
    _ = ((L * ε : ℝ≥0) : ℝ) := by push_cast; ring

/-- If a readout factors through a feature (exact factorization) and the feature
has ε-approximate recovery, the readout has (L·ε)-approximate recovery for any
L-Lipschitz recovery map. -/
theorem approxOracleRecoversReadout_of_factored_and_approxRecover
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε : ℝ≥0} {L : ℝ≥0}
    (hRecover : ApproxOracleRecoversFeature fstar feature ε)
    (hFactor : ReadoutFactorsThroughFeature feature readout)
    (hLip : ∃ recover : Feature → Readout,
      (∀ x : Strings, readout x = recover (feature x)) ∧
      LipschitzWith L recover) :
    ApproxOracleRecoversFeature fstar readout (L * ε) :=
  approxOracleRecoversReadout_of_approxRecover_and_lipschitzRecover hRecover hLip

end Composition

section RestrictedComposition

variable {Feature Readout : Type*}
variable [PseudoMetricSpace Feature] [PseudoMetricSpace Readout]

/-- Lipschitz readout composed with restricted ε-approximate feature recovery
gives restricted approximate readout recovery on the same covered relation. -/
theorem approxOracleRecoversReadoutOn_of_approxRecoverOn_and_lipschitzRecover
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε : ℝ≥0} {L : ℝ≥0}
    (hRecover : ApproxOracleRecoversFeatureOn covered fstar feature ε)
    (hFactor : ∃ recover : Feature → Readout,
      (∀ x : Strings, readout x = recover (feature x)) ∧
      LipschitzWith L recover) :
    ApproxOracleRecoversFeatureOn covered fstar readout (L * ε) := by
  rcases hFactor with ⟨recover, hRecoverEq, hLip⟩
  intro x x' hCovered hzero
  rw [hRecoverEq x, hRecoverEq x']
  have h_feat_dist := hRecover x x' hCovered hzero
  calc dist (recover (feature x)) (recover (feature x'))
      ≤ (L : ℝ) * dist (feature x) (feature x') := hLip.dist_le_mul _ _
    _ ≤ (L : ℝ) * (ε : ℝ) := by
        apply mul_le_mul_of_nonneg_left h_feat_dist
        exact_mod_cast L.property
    _ = ((L * ε : ℝ≥0) : ℝ) := by push_cast; ring

/-- Restricted exact factorization plus restricted approximate recovery yields
restricted approximate readout recovery. -/
theorem approxOracleRecoversReadoutOn_of_factored_and_approxRecoverOn
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε : ℝ≥0} {L : ℝ≥0}
    (hRecover : ApproxOracleRecoversFeatureOn covered fstar feature ε)
    (hFactor : ReadoutFactorsThroughFeature feature readout)
    (hLip : ∃ recover : Feature → Readout,
      (∀ x : Strings, readout x = recover (feature x)) ∧
      LipschitzWith L recover) :
    ApproxOracleRecoversFeatureOn covered fstar readout (L * ε) :=
  approxOracleRecoversReadoutOn_of_approxRecoverOn_and_lipschitzRecover hRecover hLip

end RestrictedComposition

section Contrapositive

variable {Feature Readout : Type*}
variable [PseudoMetricSpace Feature] [PseudoMetricSpace Readout]

/-- If a readout separates two oracle-equivalent states by more than L·ε,
then no L-Lipschitz recovery map can reconcile this with ε-approximate
feature recovery. -/
theorem readout_separation_bound_of_approxRecover
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε : ℝ≥0} {L : ℝ≥0}
    (hRecover : ApproxOracleRecoversFeature fstar feature ε)
    (hFactor : ∃ recover : Feature → Readout,
      (∀ x : Strings, readout x = recover (feature x)) ∧
      LipschitzWith L recover)
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    dist (readout x) (readout x') ≤ ((L * ε : ℝ≥0) : ℝ) :=
  approxOracleRecoversReadout_of_approxRecover_and_lipschitzRecover
    hRecover hFactor x x' hzero

end Contrapositive

end FormalProofs.OPT
