import FormalProofs.OPT.ReadoutAlignment
import FormalProofs.OPT.ApproxOracleRecovery

/-!
# FormalProofs/OPT/LipschitzReadoutFactorization.lean

Approximate readout factorization through a theorem-bearing feature.

The exact predicate `ReadoutFactorsThroughFeature` requires that the readout is
*exactly* a post-processing of the feature. In practice, learned readout heads
may only approximately factor through the learned feature φ — e.g., because a
root MLP has residual connections that bypass φ.

This file introduces `ApproxReadoutFactorsThroughFeature`, where the readout is
ε-close to some function of the feature. The main results are:

1. **Zero reduction**: ε = 0 recovers exact `ReadoutFactorsThroughFeature`.
2. **Fiber bound**: approximate factorization bounds readout variation on
   feature fibers.
3. **Combined bound**: when both feature recovery and readout factorization
   are approximate, the total readout error on oracle fibers is additive.
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

/-- A readout approximately factors through a feature if it is uniformly ε-close
to some function of the feature. -/
def ApproxReadoutFactorsThroughFeature
    {Feature Readout : Type*} [PseudoMetricSpace Readout]
    (feature : Strings → Feature)
    (readout : Strings → Readout) (ε : ℝ≥0) : Prop :=
  ∃ recover : Feature → Readout,
    ∀ x : Strings, dist (readout x) (recover (feature x)) ≤ (ε : ℝ)

section Reduction

variable {Feature Readout : Type*} [MetricSpace Readout]

/-- Exact factorization implies ε-approximate factorization for any ε ≥ 0. -/
theorem readoutFactorsThroughFeature_implies_approx
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    (h : ReadoutFactorsThroughFeature feature readout) (ε : ℝ≥0) :
    ApproxReadoutFactorsThroughFeature feature readout ε := by
  rcases h with ⟨recover, hRecover⟩
  refine ⟨recover, ?_⟩
  intro x
  rw [hRecover x]
  simpa using ε.property

/-- ε = 0 approximate factorization implies exact factorization. -/
theorem approxReadoutFactorsThroughFeature_zero_implies_exact
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    (h : ApproxReadoutFactorsThroughFeature feature readout 0) :
    ReadoutFactorsThroughFeature feature readout := by
  rcases h with ⟨recover, hRecover⟩
  refine ⟨recover, ?_⟩
  intro x
  have hdist : dist (readout x) (recover (feature x)) = 0 := by
    simpa using hRecover x
  exact dist_eq_zero.mp hdist

end Reduction

section FiberBound

variable {Feature Readout : Type*} [PseudoMetricSpace Readout]

/-- Monotonicity: tighter approximation implies looser. -/
theorem approxReadoutFactorsThroughFeature_mono
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε₁ ε₂ : ℝ≥0} (hle : ε₁ ≤ ε₂)
    (h : ApproxReadoutFactorsThroughFeature feature readout ε₁) :
    ApproxReadoutFactorsThroughFeature feature readout ε₂ := by
  rcases h with ⟨recover, hRecover⟩
  refine ⟨recover, ?_⟩
  intro x
  calc dist (readout x) (recover (feature x)) ≤ (ε₁ : ℝ) := hRecover x
    _ ≤ (ε₂ : ℝ) := by exact_mod_cast hle

/-- Approximate factorization bounds readout variation on feature fibers:
if feature(x) = feature(x'), then readout(x) and readout(x') differ by at
most 2ε. -/
theorem approxReadoutFactorsThroughFeature_fiber_bound
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε : ℝ≥0}
    (h : ApproxReadoutFactorsThroughFeature feature readout ε)
    {x x' : Strings} (hEq : feature x = feature x') :
    dist (readout x) (readout x') ≤ 2 * (ε : ℝ) := by
  rcases h with ⟨recover, hRecover⟩
  calc dist (readout x) (readout x')
      ≤ dist (readout x) (recover (feature x)) +
        dist (recover (feature x)) (readout x') := dist_triangle _ _ _
    _ = dist (readout x) (recover (feature x)) +
        dist (recover (feature x')) (readout x') := by rw [hEq]
    _ ≤ dist (readout x) (recover (feature x)) +
        dist (readout x') (recover (feature x')) := by
          have hRight :
              dist (recover (feature x')) (readout x') ≤
                dist (readout x') (recover (feature x')) := by
            rw [dist_comm]
          exact add_le_add_right hRight _
    _ ≤ (ε : ℝ) + (ε : ℝ) := add_le_add (hRecover x) (hRecover x')
    _ = 2 * (ε : ℝ) := by ring

end FiberBound

section CombinedBound

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Readout : Type*}
variable [PseudoMetricSpace Feature] [PseudoMetricSpace Readout]

/-- Combined bound: ε-approximate feature recovery plus δ-approximate readout
factorization through an L-Lipschitz recover map gives a total readout distance
bound of (L·ε + 2δ) on oracle fibers.

The two error sources are independent and decompose additively:
- L·ε from imperfect fiber preservation by the feature map
- 2δ from imperfect factorization of the readout through the feature -/
theorem combined_readout_bound_on_oracle_fibers
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε_fiber : ℝ≥0} {ε_readout : ℝ≥0} {L : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hApproxFactor : ApproxReadoutFactorsThroughFeature feature readout ε_readout)
    (hLip : ∃ recover : Feature → Readout,
      (∀ x : Strings, dist (readout x) (recover (feature x)) ≤ (ε_readout : ℝ)) ∧
      LipschitzWith L recover)
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    dist (readout x) (readout x') ≤
      (L : ℝ) * (ε_fiber : ℝ) + 2 * (ε_readout : ℝ) := by
  rcases hLip with ⟨recover, hRecoverBound, hRecoverLip⟩
  have h_feat := hApproxRecover x x' hzero
  have hLeft : dist (readout x) (recover (feature x)) ≤ (ε_readout : ℝ) :=
    hRecoverBound x
  have hMid :
      dist (recover (feature x)) (recover (feature x')) ≤
        (L : ℝ) * dist (feature x) (feature x') :=
    hRecoverLip.dist_le_mul _ _
  have hRight :
      dist (recover (feature x')) (readout x') ≤ (ε_readout : ℝ) := by
    rw [dist_comm]
    exact hRecoverBound x'
  have hFiberScaled :
      (L : ℝ) * dist (feature x) (feature x') ≤ (L : ℝ) * (ε_fiber : ℝ) := by
    apply mul_le_mul_of_nonneg_left h_feat
    exact_mod_cast L.property
  calc dist (readout x) (readout x')
      ≤ dist (readout x) (recover (feature x)) +
        dist (recover (feature x)) (recover (feature x')) +
        dist (recover (feature x')) (readout x') := by
          calc dist (readout x) (readout x')
              ≤ dist (readout x) (recover (feature x)) +
                dist (recover (feature x)) (readout x') := dist_triangle _ _ _
            _ ≤ dist (readout x) (recover (feature x)) +
                (dist (recover (feature x)) (recover (feature x')) +
                dist (recover (feature x')) (readout x')) := by
                  exact add_le_add_right
                    (dist_triangle
                      (recover (feature x))
                      (recover (feature x'))
                      (readout x'))
                    _
            _ = dist (readout x) (recover (feature x)) +
                dist (recover (feature x)) (recover (feature x')) +
                dist (recover (feature x')) (readout x') := by ring
    _ ≤ (ε_readout : ℝ) +
        (L : ℝ) * dist (feature x) (feature x') +
        (ε_readout : ℝ) := by
          have hLeftMid :
              dist (readout x) (recover (feature x)) +
                  dist (recover (feature x)) (recover (feature x')) ≤
                (ε_readout : ℝ) + (L : ℝ) * dist (feature x) (feature x') := by
            exact add_le_add hLeft hMid
          have hTotal :
              (dist (readout x) (recover (feature x)) +
                  dist (recover (feature x)) (recover (feature x'))) +
                  dist (recover (feature x')) (readout x') ≤
                ((ε_readout : ℝ) + (L : ℝ) * dist (feature x) (feature x')) +
                  (ε_readout : ℝ) := by
            exact add_le_add hLeftMid hRight
          simpa [add_assoc] using hTotal
    _ ≤ (ε_readout : ℝ) + (L : ℝ) * (ε_fiber : ℝ) + (ε_readout : ℝ) := by
          have hStep :
              ((ε_readout : ℝ) + (L : ℝ) * dist (feature x) (feature x')) +
                  (ε_readout : ℝ) ≤
                ((ε_readout : ℝ) + (L : ℝ) * (ε_fiber : ℝ)) +
                  (ε_readout : ℝ) := by
            exact add_le_add_left
              (add_le_add_right hFiberScaled (ε_readout : ℝ))
              (ε_readout : ℝ)
          simpa [add_assoc] using hStep
    _ = (L : ℝ) * (ε_fiber : ℝ) + 2 * (ε_readout : ℝ) := by ring

end CombinedBound

section RestrictedCombinedBound

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Readout : Type*}
variable [PseudoMetricSpace Feature] [PseudoMetricSpace Readout]

/-- Restricted version of `combined_readout_bound_on_oracle_fibers` for a
covered pair relation. This is the natural form when only some oracle-labeled
pairs are available. -/
theorem combined_readout_bound_on_covered_oracle_fibers
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    {ε_fiber : ℝ≥0} {ε_readout : ℝ≥0} {L : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeatureOn covered fstar feature ε_fiber)
    (hApproxFactor : ApproxReadoutFactorsThroughFeature feature readout ε_readout)
    (hLip : ∃ recover : Feature → Readout,
      (∀ x : Strings, dist (readout x) (recover (feature x)) ≤ (ε_readout : ℝ)) ∧
      LipschitzWith L recover)
    {x x' : Strings}
    (hCovered : covered x x')
    (hzero : dist (fstar x) (fstar x') = 0) :
    dist (readout x) (readout x') ≤
      (L : ℝ) * (ε_fiber : ℝ) + 2 * (ε_readout : ℝ) := by
  rcases hLip with ⟨recover, hRecoverBound, hRecoverLip⟩
  have h_feat := hApproxRecover x x' hCovered hzero
  have hLeft : dist (readout x) (recover (feature x)) ≤ (ε_readout : ℝ) :=
    hRecoverBound x
  have hMid :
      dist (recover (feature x)) (recover (feature x')) ≤
        (L : ℝ) * dist (feature x) (feature x') :=
    hRecoverLip.dist_le_mul _ _
  have hRight :
      dist (recover (feature x')) (readout x') ≤ (ε_readout : ℝ) := by
    rw [dist_comm]
    exact hRecoverBound x'
  have hFiberScaled :
      (L : ℝ) * dist (feature x) (feature x') ≤ (L : ℝ) * (ε_fiber : ℝ) := by
    apply mul_le_mul_of_nonneg_left h_feat
    exact_mod_cast L.property
  calc dist (readout x) (readout x')
      ≤ dist (readout x) (recover (feature x)) +
        dist (recover (feature x)) (recover (feature x')) +
        dist (recover (feature x')) (readout x') := by
          calc dist (readout x) (readout x')
              ≤ dist (readout x) (recover (feature x)) +
                dist (recover (feature x)) (readout x') := dist_triangle _ _ _
            _ ≤ dist (readout x) (recover (feature x)) +
                (dist (recover (feature x)) (recover (feature x')) +
                dist (recover (feature x')) (readout x')) := by
                  exact add_le_add_right
                    (dist_triangle
                      (recover (feature x))
                      (recover (feature x'))
                      (readout x'))
                    _
            _ = dist (readout x) (recover (feature x)) +
                dist (recover (feature x)) (recover (feature x')) +
                dist (recover (feature x')) (readout x') := by ring
    _ ≤ (ε_readout : ℝ) +
        (L : ℝ) * dist (feature x) (feature x') +
        (ε_readout : ℝ) := by
          have hLeftMid :
              dist (readout x) (recover (feature x)) +
                  dist (recover (feature x)) (recover (feature x')) ≤
                (ε_readout : ℝ) + (L : ℝ) * dist (feature x) (feature x') := by
            exact add_le_add hLeft hMid
          have hTotal :
              (dist (readout x) (recover (feature x)) +
                  dist (recover (feature x)) (recover (feature x'))) +
                  dist (recover (feature x')) (readout x') ≤
                ((ε_readout : ℝ) + (L : ℝ) * dist (feature x) (feature x')) +
                  (ε_readout : ℝ) := by
            exact add_le_add hLeftMid hRight
          simpa [add_assoc] using hTotal
    _ ≤ (ε_readout : ℝ) + (L : ℝ) * (ε_fiber : ℝ) + (ε_readout : ℝ) := by
          have hStep :
              ((ε_readout : ℝ) + (L : ℝ) * dist (feature x) (feature x')) +
                  (ε_readout : ℝ) ≤
                ((ε_readout : ℝ) + (L : ℝ) * (ε_fiber : ℝ)) +
                  (ε_readout : ℝ) := by
            exact add_le_add_left
              (add_le_add_right hFiberScaled (ε_readout : ℝ))
              (ε_readout : ℝ)
          simpa [add_assoc] using hStep
    _ = (L : ℝ) * (ε_fiber : ℝ) + 2 * (ε_readout : ℝ) := by ring

end RestrictedCombinedBound

end FormalProofs.OPT
