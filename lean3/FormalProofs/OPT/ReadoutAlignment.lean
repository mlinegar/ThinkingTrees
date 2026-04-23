import FormalProofs.OPT.TheoremBackingMeasurementError

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
