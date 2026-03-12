import FormalProofs.OPT.GlobalAssumptions

/-!
# FormalProofs/OPT/SketchFlipMergeBridge.lean

Bridge lemmas between the C-TreePO global merge condition (`A2_global`) and
Sketch-Flip-Merge style "extra target" requirements (Corollary 4.11 shape).

The key point formalized here:
if one deterministic merge route `g (g u * g v)` is required to preserve two
different binary targets, then those targets must already be oracle-equivalent.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- Additional target condition using the same deterministic merge route.

This is the C-TreePO analogue of adding an extra condition like
Corollary 4.11's condition (4) on top of an existing merge condition. -/
def SameRouteAltTarget
    (g : Strings → Strings)
    (fstar : Strings → Y)
    (alt : Strings → Strings → Strings) : Prop :=
  ∀ u v : Strings, D fstar (alt u v) (g (g u * g v)) = 0

/-- If one deterministic merge route preserves both concatenation and an alternate
target, then the two targets are oracle-equivalent pointwise. -/
theorem same_route_two_targets_force_oracle_equiv
    {g : Strings → Strings} {fstar : Strings → Y}
    {alt : Strings → Strings → Strings}
    (hA2 : A2_global g fstar)
    (hAlt : SameRouteAltTarget g fstar alt) :
    ∀ u v : Strings, D fstar (u * v) (alt u v) = 0 := by
  intro u v
  have hConcat : D fstar (u * v) (g (g u * g v)) = 0 := hA2 u v
  have hAlt' : D fstar (g (g u * g v)) (alt u v) = 0 := by
    rw [D_symm]
    exact hAlt u v
  apply le_antisymm _ dist_nonneg
  calc
    D fstar (u * v) (alt u v)
      ≤ D fstar (u * v) (g (g u * g v)) + D fstar (g (g u * g v)) (alt u v) :=
        D_triangle fstar (u * v) (g (g u * g v)) (alt u v)
    _ = 0 + 0 := by rw [hConcat, hAlt']
    _ = 0 := by ring

/-- Contrapositive form: if the oracle distinguishes concatenation from an
alternate target on any input pair, both same-route conditions cannot hold. -/
theorem no_two_distinguished_targets_on_one_route
    {g : Strings → Strings} {fstar : Strings → Y}
    {alt : Strings → Strings → Strings}
    (hA2 : A2_global g fstar)
    (hSep : ∃ u v : Strings, D fstar (u * v) (alt u v) ≠ 0) :
    ¬ SameRouteAltTarget g fstar alt := by
  intro hAlt
  rcases hSep with ⟨u, v, hne⟩
  exact hne (same_route_two_targets_force_oracle_equiv (g := g) (fstar := fstar)
    (alt := alt) hA2 hAlt u v)

/-- Typeclass-packaged variant: under global preservation, any oracle-distinguishable
alternate target is incompatible with the same-route requirement. -/
theorem global_preservation_rejects_distinguished_alt_target
    {g : Strings → Strings} {fstar : Strings → Y}
    [inst : GlobalPreservation g fstar]
    {alt : Strings → Strings → Strings}
    (hSep : ∃ u v : Strings, D fstar (u * v) (alt u v) ≠ 0) :
    ¬ SameRouteAltTarget g fstar alt := by
  exact no_two_distinguished_targets_on_one_route (g := g) (fstar := fstar)
    (alt := alt) inst.a2 hSep

