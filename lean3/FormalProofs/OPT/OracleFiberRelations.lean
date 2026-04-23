import FormalProofs.OPT.ApproxOracleRecovery
import FormalProofs.OPT.TheoremBackingConsequences

/-!
# FormalProofs/OPT/OracleFiberRelations.lean

Relation-first restatement of theorem-backedness around oracle fibers.

The point of this file is to make the intended object explicit before any
particular learned feature map is chosen:

- the primitive equivalence relation is "same oracle fiber";
- exact and approximate feature recovery are just ways of realizing that
  relation with a learned theorem feature; and
- exact theorem-backedness keeps realized reductions inside one oracle fiber.
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

/-- Two inputs lie in the same oracle fiber when the oracle identifies them. -/
def SameOracleFiber
    (fstar : Strings → Y) (x x' : Strings) : Prop :=
  dist (fstar x) (fstar x') = 0

theorem sameOracleFiber_refl
    (fstar : Strings → Y) (x : Strings) :
    SameOracleFiber fstar x x := by
  simp [SameOracleFiber]

theorem sameOracleFiber_symm
    {fstar : Strings → Y} {x x' : Strings}
    (h : SameOracleFiber fstar x x') :
    SameOracleFiber fstar x' x := by
  simpa [SameOracleFiber, dist_comm] using h

theorem sameOracleFiber_trans
    {fstar : Strings → Y} {x y z : Strings}
    (hxy : SameOracleFiber fstar x y)
    (hyz : SameOracleFiber fstar y z) :
    SameOracleFiber fstar x z := by
  have hxyEq : fstar x = fstar y := dist_eq_zero.mp hxy
  have hyzEq : fstar y = fstar z := dist_eq_zero.mp hyz
  simpa [SameOracleFiber, hxyEq, hyzEq]

section Recovery

variable {Feature : Type*}

/-- Exact oracle recovery is exactly the statement that the learned theorem
feature is constant on oracle fibers. -/
theorem oracleRecoversFeature_iff_respects_sameOracleFiber
    {fstar : Strings → Y} {feature : Strings → Feature} :
    OracleRecoversFeature fstar feature ↔
      ∀ {x x' : Strings}, SameOracleFiber fstar x x' → feature x = feature x' := by
  constructor
  · intro hRecover x x' hFiber
    exact hRecover x x' hFiber
  · intro hFiber x x' hzero
    exact hFiber hzero

end Recovery

section ApproxRecovery

variable {Feature : Type*} [PseudoMetricSpace Feature]

/-- Approximate oracle recovery is exactly the statement that the learned
theorem feature has bounded diameter on each oracle fiber. -/
theorem approxOracleRecoversFeature_iff_bounded_on_sameOracleFiber
    {fstar : Strings → Y} {feature : Strings → Feature} {ε : ℝ≥0} :
    ApproxOracleRecoversFeature fstar feature ε ↔
      ∀ {x x' : Strings}, SameOracleFiber fstar x x' →
        dist (feature x) (feature x') ≤ (ε : ℝ) := by
  constructor
  · intro hRecover x x' hFiber
    exact hRecover x x' hFiber
  · intro hBound x x' hzero
    exact hBound hzero

end ApproxRecovery

section ExactSupport

/-- Leaf-support version: realized leaf summaries stay in the same oracle fiber
as their source leaf. -/
theorem leaf_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {b z : Strings}
    (hExact : ExactTheoremBacked g T fstar)
    (hb : b ∈ leaves T)
    (hz : z ∈ (g b).support) :
    SameOracleFiber fstar z b := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z b = 0 := hSupport.1 b hb z hz
  simpa [SameOracleFiber, D] using hzeroD

/-- Merge-support version: every realized internal reduction stays in the same
oracle fiber as its raw subtree. -/
theorem merge_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {p : BinTree Strings × BinTree Strings} {z : Strings}
    (hExact : ExactTheoremBacked g T fstar)
    (hp : p ∈ internal_nodes T)
    (hz : z ∈ (reduce g (BinTree.node p.1 p.2)).support) :
    SameOracleFiber fstar z (S (BinTree.node p.1 p.2)) := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z (S (BinTree.node p.1 p.2)) = 0 := hSupport.2.1 p hp z hz
  simpa [SameOracleFiber, D] using hzeroD

/-- On-range idempotence version: re-summaries stay in the same oracle fiber as
the already-realized theorem object. -/
theorem idempotent_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {Z z : Strings}
    (hExact : ExactTheoremBacked g T fstar)
    (hRange : InRange g Z)
    (hz : z ∈ (g Z).support) :
    SameOracleFiber fstar z Z := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z Z = 0 := hSupport.2.2 Z hRange z hz
  simpa [SameOracleFiber, D] using hzeroD

/-- Multi-round support version: every realized reduction under `ZR` remains in
the same oracle fiber as the original document. -/
theorem zr_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    {z : Strings}
    (hz : z ∈ (ZR g x R T).support) :
    SameOracleFiber fstar z x := by
  exact zero_distortion_on_ZR_support_of_exactTheoremBacked
    (hp := hp) (hExact := hExact) (hR := hR) z hz

end ExactSupport

end FormalProofs.OPT
