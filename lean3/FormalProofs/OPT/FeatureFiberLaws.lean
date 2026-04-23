import FormalProofs.OPT.TheoremBackingApproxMeasurementError
import FormalProofs.OPT.TheoremBackingStructure

/-!
# FormalProofs/OPT/FeatureFiberLaws.lean

Feature-fiber restatements of the theorem-backed local-law route.

This file makes explicit a formulation that is often more natural for learned
latent features:

- C2 can be read as preservation of a **feature fiber** rather than raw latent
  equality;
- exact theorem-backedness gives support-level feature-fiber preservation on
  leaves, merges, and on-range resummaries whenever the oracle identifies the
  feature; and
- approximate theorem-backedness inherits the existing oracle-to-feature
  distortion bound through `FeatureLipschitzFromOracle`.
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

/-- Two inputs lie in the same feature fiber when the theorem-bearing feature
identifies them. -/
def SameFeatureFiber
    {Feature : Type*}
    (feature : Strings → Feature) (x x' : Strings) : Prop :=
  feature x = feature x'

theorem sameFeatureFiber_refl
    {Feature : Type*} (feature : Strings → Feature) (x : Strings) :
    SameFeatureFiber feature x x := by
  rfl

theorem sameFeatureFiber_symm
    {Feature : Type*} {feature : Strings → Feature} {x x' : Strings}
    (h : SameFeatureFiber feature x x') :
    SameFeatureFiber feature x' x := by
  simpa [SameFeatureFiber] using h.symm

theorem sameFeatureFiber_trans
    {Feature : Type*} {feature : Strings → Feature} {x y z : Strings}
    (hxy : SameFeatureFiber feature x y)
    (hyz : SameFeatureFiber feature y z) :
    SameFeatureFiber feature x z := by
  simpa [SameFeatureFiber] using Eq.trans hxy hyz

section EncodedFeature

variable {Feature : Type*} [Encodable Feature]

/-- Exact bridge: encoded-feature zero distortion is equivalent to lying in the
same feature fiber. -/
theorem sameFeatureFiber_iff_encodedOracle_zero
    {feature : Strings → Feature} {x x' : Strings} :
    SameFeatureFiber feature x x' ↔
      dist ((encodedOracle (Strings := Strings) feature) x)
        ((encodedOracle (Strings := Strings) feature) x') = 0 := by
  constructor
  · intro hFiber
    apply dist_eq_zero.mpr
    simp [SameFeatureFiber] at hFiber
    simpa [encodedOracle, hFiber]
  · intro hzero
    have hEq :
        (encodedOracle (Strings := Strings) feature) x =
          (encodedOracle (Strings := Strings) feature) x' :=
      dist_eq_zero.mp hzero
    have hCodeReal :
        ((Encodable.encode (feature x) : ℕ) : ℝ) =
          ((Encodable.encode (feature x') : ℕ) : ℝ) := by
      simpa [encodedOracle] using hEq
    have hCode :
        Encodable.encode (feature x) = Encodable.encode (feature x') := by
      exact_mod_cast hCodeReal
    simpa [SameFeatureFiber] using Encodable.encode_injective hCode

end EncodedFeature

section ExactSupport

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature : Type*}

/-- Exact oracle recovery turns zero oracle distortion into feature-fiber
preservation. -/
theorem sameFeatureFiber_of_zero_oracle_dist
    {fstar : Strings → Y} {feature : Strings → Feature}
    (hRecover : OracleRecoversFeature fstar feature)
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    SameFeatureFiber feature x x' := by
  simpa [SameFeatureFiber] using hRecover x x' hzero

/-- Leaf-support version: realized leaf summaries stay inside one feature fiber
whenever the oracle identifies that feature. -/
theorem leaf_support_sameFeatureFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {feature : Strings → Feature}
    (hExact : ExactTheoremBacked g T fstar)
    (hRecover : OracleRecoversFeature fstar feature)
    {b z : Strings}
    (hb : b ∈ leaves T)
    (hz : z ∈ (g b).support) :
    SameFeatureFiber feature z b := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z b = 0 := hSupport.1 b hb z hz
  exact sameFeatureFiber_of_zero_oracle_dist
    (fstar := fstar) (feature := feature) hRecover (by simpa [D] using hzeroD)

/-- Merge-support version: every realized internal reduction stays inside the
feature fiber of its raw subtree whenever the oracle identifies that feature. -/
theorem merge_support_sameFeatureFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {feature : Strings → Feature}
    (hExact : ExactTheoremBacked g T fstar)
    (hRecover : OracleRecoversFeature fstar feature)
    {p : BinTree Strings × BinTree Strings} {z : Strings}
    (hp : p ∈ internal_nodes T)
    (hz : z ∈ (reduce g (BinTree.node p.1 p.2)).support) :
    SameFeatureFiber feature z (S (BinTree.node p.1 p.2)) := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z (S (BinTree.node p.1 p.2)) = 0 := hSupport.2.1 p hp z hz
  exact sameFeatureFiber_of_zero_oracle_dist
    (fstar := fstar) (feature := feature) hRecover (by simpa [D] using hzeroD)

/-- On-range idempotence version: re-summaries stay inside the same feature
fiber as the already-realized theorem object. -/
theorem idempotent_support_sameFeatureFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {feature : Strings → Feature}
    (hExact : ExactTheoremBacked g T fstar)
    (hRecover : OracleRecoversFeature fstar feature)
    {Z z : Strings}
    (hRange : InRange g Z)
    (hz : z ∈ (g Z).support) :
    SameFeatureFiber feature z Z := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z Z = 0 := hSupport.2.2 Z hRange z hz
  exact sameFeatureFiber_of_zero_oracle_dist
    (fstar := fstar) (feature := feature) hRecover (by simpa [D] using hzeroD)

/-- Multi-round support version: every realized reduction under `ZR` remains in
the feature fiber of the original document. -/
theorem zr_support_sameFeatureFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y} {feature : Strings → Feature}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    {z : Strings}
    (hz : z ∈ (ZR g x R T).support) :
    SameFeatureFiber feature z x := by
  exact sameFeatureFiber_of_zero_oracle_dist
    (fstar := fstar) (feature := feature) hRecover
    (zero_distortion_on_ZR_support_of_exactTheoremBacked
      (hp := hp) (hExact := hExact) (hR := hR) z hz)

end ExactSupport

section Approximate

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Approximate bridge: feature-fiber distortion is controlled by oracle
distortion whenever the feature is Lipschitz through the oracle. -/
theorem expected_featureFiberDistortion_le_of_featureLipschitzFromOracle
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (feature : Strings → Feature)
    (K : ℝ≥0)
    (hLip : FeatureLipschitzFromOracle fstar feature K) :
    Exp p (fun z => D feature z x) ≤ (K : ℝ) * Exp p (fun z => D fstar z x) := by
  simpa using
    (feature_distortion_le_of_featureLipschitzFromOracle
      (p := p) (x := x) (fstar := fstar) (feature := feature) (K := K) hLip)

end Approximate

end FormalProofs.OPT
