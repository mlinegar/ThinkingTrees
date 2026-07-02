import FormalProofs.OPT.TheoremBacking
import FormalProofs.OPT.TheoremBackingConsequences

/-!
# FormalProofs/OPT/OracleFibers.lean

Consolidated (2026-07-02) from the oracle-fiber cluster, laws/relations layer:
`ReadoutAlignment`, `FeatureFiberLaws`, `LabelScoreObjectives`.

Each original file is preserved verbatim as one section below; the original
modules remain as import shims. The objectives layer of the same cluster lives
in `FormalProofs/OPT/OracleFiberObjectives.lean` (split forced by import
cycles through `ApproxOracleRecovery`/`LipschitzReadoutFactorization`, which
import `ReadoutAlignment`).
-/

/-! ## From FormalProofs/OPT/ReadoutAlignment.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/ReadoutAlignment.lean

Readout-side alignment lemmas for theorem-backed operators.

This file isolates a distinction that matters for learned tree systems:

- a **same-surface** root objective is routed directly through the same
  theorem-bearing feature used by the local laws;
- a more general auxiliary root readout is still theory-aligned if it
  **factors through** that theorem feature; and
- if a proposed root head separates two inputs that the theorem feature
  identifies, then it cannot be justified by the same theorem-bearing route.

These lemmas are intentionally architecture-agnostic. They do not mention
specific neural operators or specific sketches; they only express the minimal
structural assumptions needed for the exact theorem-backed transport layer to
apply to downstream root objectives.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*}

/-- A downstream readout is theory-aligned with a theorem-bearing feature if it
is obtained by post-processing that feature. -/
def ReadoutFactorsThroughFeature
    {Feature Readout : Type*}
    (feature : Strings → Feature)
    (readout : Strings → Readout) : Prop :=
  ∃ recover : Feature → Readout, ∀ x : Strings, readout x = recover (feature x)

/-- Minimal alignment: the root readout uses exactly the same feature surface as
the theorem-facing object. -/
def SameReadoutSurface
    {Feature : Type*}
    (feature readout : Strings → Feature) : Prop :=
  ∀ x : Strings, readout x = feature x

/-- Same-surface routing is the minimal special case of feature factorization. -/
theorem sameReadoutSurface_implies_factorsThroughFeature
    {Feature : Type*}
    {feature readout : Strings → Feature}
    (hSame : SameReadoutSurface feature readout) :
    ReadoutFactorsThroughFeature feature readout := by
  refine ⟨id, ?_⟩
  intro x
  simpa using hSame x

/-- Any theory-aligned readout is constant on fibers of the theorem-bearing
feature. -/
theorem readoutFactorsThroughFeature_respects_feature_fibers
    {Feature Readout : Type*}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    (hFactor : ReadoutFactorsThroughFeature feature readout) :
    ∀ {x x' : Strings}, feature x = feature x' → readout x = readout x' := by
  rcases hFactor with ⟨recover, hRecover⟩
  intro x x' hEq
  rw [hRecover x, hRecover x', hEq]

/-- Two simultaneous heads remain theory-aligned whenever each head factors
through the same theorem-bearing feature. This packages the common
"task-readout + summary-readout" case into one factorization witness. -/
theorem pairedReadoutFactorsThroughFeature
    {Feature Readout₁ Readout₂ : Type*}
    {feature : Strings → Feature}
    {readout₁ : Strings → Readout₁}
    {readout₂ : Strings → Readout₂}
    (hFactor₁ : ReadoutFactorsThroughFeature feature readout₁)
    (hFactor₂ : ReadoutFactorsThroughFeature feature readout₂) :
    ReadoutFactorsThroughFeature feature
      (fun x => (readout₁ x, readout₂ x)) := by
  rcases hFactor₁ with ⟨recover₁, hRecover₁⟩
  rcases hFactor₂ with ⟨recover₂, hRecover₂⟩
  refine ⟨fun y => (recover₁ y, recover₂ y), ?_⟩
  intro x
  simp [hRecover₁ x, hRecover₂ x]

/-- Contrapositive: if a root head distinguishes two states that the theorem
feature identifies, then it is not theory-aligned via that feature. -/
theorem not_readoutFactorsThroughFeature_of_distinguished_feature_fibers
    {Feature Readout : Type*}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    (hSep : ∃ x x' : Strings, feature x = feature x' ∧ readout x ≠ readout x') :
    ¬ ReadoutFactorsThroughFeature feature readout := by
  intro hFactor
  rcases hSep with ⟨x, x', hEq, hNe⟩
  exact hNe (readoutFactorsThroughFeature_respects_feature_fibers hFactor hEq)

section OracleRecovery

variable {Y : Type*} [BoundedMetricSpace Y]

/-- If the oracle identifies a theorem-bearing feature exactly, then every
readout that factors through that feature is also oracle-identified. -/
theorem oracleRecoversReadout_of_oracleRecoversFeature_and_factorization
    {Feature Readout : Type*}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {readout : Strings → Readout}
    (hRecover : OracleRecoversFeature fstar feature)
    (hFactor : ReadoutFactorsThroughFeature feature readout) :
    OracleRecoversFeature fstar readout := by
  rcases hFactor with ⟨recover, hRecoverReadout⟩
  intro x x' hzero
  rw [hRecoverReadout x, hRecoverReadout x']
  rw [hRecover x x' hzero]

end OracleRecovery

section Transport

variable [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Readout α : Type*}
variable [Encodable Readout]

/-- Exact theorem-backed transport for any loss indexed by a root readout that
factors through the theorem-bearing feature. -/
theorem expected_loss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (readout : Strings → Readout)
    (loss : Strings → α → ℝ)
    (gen : Strings → PMF α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (hFactor : ReadoutFactorsThroughFeature feature readout)
    (h_meas : OracleMeasurableLossGeneric loss (encodedOracle (Strings := Strings) readout))
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) readout)) :
    ExpectedLossGeneric loss (PMF.pure x) gen =
      ExpectedLossGeneric loss (ZR g x R T) gen := by
  exact
    expected_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar)
      (feature := readout)
      (loss := loss)
      (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR
      (oracleRecoversReadout_of_oracleRecoversFeature_and_factorization
        (fstar := fstar) (feature := feature) (readout := readout) hRecover hFactor)
      h_meas h_gen

/-- Direct supervised root-state learning is preserved exactly whenever the
supervised target factors through the theorem-bearing feature. -/
theorem supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (readout : Strings → Readout)
    (loss : Readout → Readout → ℝ)
    (gen : Strings → PMF Readout)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature)
    (hFactor : ReadoutFactorsThroughFeature feature readout)
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) readout)) :
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) readout loss) (PMF.pure x) gen =
      ExpectedLossGeneric
        (supervisedStateLoss (Strings := Strings) readout loss) (ZR g x R T) gen := by
  exact
    supervisedStateExpectedLoss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar)
      (feature := readout)
      (loss := loss)
      (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR
      (oracleRecoversReadout_of_oracleRecoversFeature_and_factorization
        (fstar := fstar) (feature := feature) (readout := readout) hRecover hFactor)
      h_gen

/-- Same-surface root routing is the minimal theory-aligned special case of the
factored-readout transport theorem. -/
theorem supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_sameSurface
    [Encodable Feature]
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
  exact
    supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout
      (fstar := fstar)
      (feature := feature)
      (readout := feature)
      (loss := loss)
      (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover
      (sameReadoutSurface_implies_factorsThroughFeature (feature := feature) (readout := feature)
        (by intro _x; rfl))
      h_gen

end Transport

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/FeatureFiberLaws.lean (consolidated 2026-07-02) -/

section

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

end

end

/-! ## From FormalProofs/OPT/LabelScoreObjectives.lean (consolidated 2026-07-02) -/

section

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

end

end
