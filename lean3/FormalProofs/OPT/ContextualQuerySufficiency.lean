import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Topology.MetricSpace.Basic
import FormalProofs.OPT.UniformG

/-!
# FormalProofs/OPT/ContextualQuerySufficiency.lean

This file isolates the deterministic core behind learning `g` as a sufficient
state map.

The general object is a family of contextual queries
`query : Ctx → X → Y`. A representation `rep : X → Rep` is sufficient exactly
when every collision of `rep` stays inside the same contextual-response fiber:
for every context `c`, `query c x = query c y`.

The Markov `(count, first, last)` sketch is only one validation witness for
this interface. The generic definition below is problem-independent and keeps
the theorem surface focused on contextual response preservation, not on
hard-coded Markov slots.
-/

set_option linter.mathlibStandardSet false

open scoped Nat
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X Ctx Rep Y State Carrier : Type*}

/-- The full contextual-response signature of an input. -/
def ResponseSignature (query : Ctx → X → Y) (x : X) : Ctx → Y :=
  fun c => query c x

/-- A representation is query-sufficient when its fibers refine contextual
response fibers. In words: if `rep` cannot distinguish `x` and `y`, no
downstream context/query response can distinguish them either. -/
def QuerySufficient (rep : X → Rep) (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → ∀ c : Ctx, query c x = query c y

/-- A readout realizes all contextual responses from the representation. -/
def ContextReadoutRealizes
    (rep : X → Rep)
    (query : Ctx → X → Y)
    (readout : Rep → Ctx → Y) : Prop :=
  ∀ x c, readout (rep x) c = query c x

/-- The full response signature is sufficient by construction. -/
theorem responseSignatures_querySufficient
    (query : Ctx → X → Y) :
    QuerySufficient (ResponseSignature query) query := by
  intro x y hxy c
  exact congrFun hxy c

/-- Sufficiency is equivalently preservation of response signatures across
representation collisions. -/
theorem querySufficient_iff_responseSignature_respects_rep
    {rep : X → Rep}
    {query : Ctx → X → Y} :
    QuerySufficient rep query ↔
      ∀ ⦃x y : X⦄,
        rep x = rep y →
          ResponseSignature query x = ResponseSignature query y := by
  constructor
  · intro hSuff x y hxy
    funext c
    exact hSuff hxy c
  · intro hSig x y hxy c
    exact congrFun (hSig hxy) c

/-- A sufficient representation cannot collapse two inputs separated by any
contextual query response. -/
theorem querySufficient_no_collision_of_distinguished_context
    {rep : X → Rep}
    {query : Ctx → X → Y}
    (hSuff : QuerySufficient rep query)
    {x y : X}
    {c : Ctx}
    (hSep : query c x ≠ query c y) :
    rep x ≠ rep y := by
  intro hxy
  exact hSep (hSuff hxy c)

/-- A query-sufficient representation is exactly one from which some contextual
readout can recover every query response. The default branch is irrelevant off
the image of `rep`; the theorem therefore only needs `X` inhabited. -/
theorem querySufficient_iff_exists_contextReadout
    [Inhabited X]
    {rep : X → Rep}
    {query : Ctx → X → Y} :
    QuerySufficient rep query ↔
      ∃ readout : Rep → Ctx → Y,
        ContextReadoutRealizes rep query readout := by
  constructor
  · intro hSuff
    classical
    let readout : Rep → Ctx → Y := fun r c =>
      if h : ∃ x, rep x = r then query c (Classical.choose h) else query c default
    refine ⟨readout, ?_⟩
    intro x c
    unfold readout
    have hx : ∃ x', rep x' = rep x := ⟨x, rfl⟩
    simp [hx]
    exact hSuff (Classical.choose_spec hx) c
  · rintro ⟨readout, hReadout⟩ x y hxy c
    calc
      query c x = readout (rep x) c := (hReadout x c).symm
      _ = readout (rep y) c := by rw [hxy]
      _ = query c y := hReadout y c

/-- Finite-context sufficiency: collisions preserve responses on a sampled
finite context set. This is the theorem-level counterpart of zero empirical
contextual loss. -/
def QuerySufficientOn
    [DecidableEq Ctx]
    (contexts : Finset Ctx)
    (rep : X → Rep)
    (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → ∀ c ∈ contexts, query c x = query c y

/-- A finite context set covers the full response-signature fibers if equality
on that finite set implies equality under every context. -/
def FiniteContextCovers
    [DecidableEq Ctx]
    (contexts : Finset Ctx)
    (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    (∀ c ∈ contexts, query c x = query c y) →
      ∀ c : Ctx, query c x = query c y

/-- If the sampled contexts cover the true response-signature fibers, zero
finite-context collision loss implies full query sufficiency. -/
theorem finiteContext_zeroLoss_implies_querySufficient
    [DecidableEq Ctx]
    {contexts : Finset Ctx}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    (hCover : FiniteContextCovers contexts query)
    (hZero : QuerySufficientOn contexts rep query) :
    QuerySufficient rep query := by
  intro x y hxy c
  exact hCover (fun c' hc' => hZero hxy c' hc') c

section ApproximateFiniteContexts

variable [PseudoMetricSpace Y]

/-- Approximate finite-context sufficiency: collisions preserve responses up to
`ε` on a sampled finite context set. -/
def QuerySufficientWithinOn
    [DecidableEq Ctx]
    (contexts : Finset Ctx)
    (ε : ℝ)
    (rep : X → Rep)
    (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    rep x = rep y →
      ∀ c ∈ contexts, dist (query c x) (query c y) ≤ ε

/-- Approximate finite context cover: closeness on a finite context set implies
closeness for every contextual response. -/
def FiniteContextCoversWithin
    [DecidableEq Ctx]
    (contexts : Finset Ctx)
    (ε δ : ℝ)
    (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    (∀ c ∈ contexts, dist (query c x) (query c y) ≤ ε) →
      ∀ c : Ctx, dist (query c x) (query c y) ≤ δ

end ApproximateFiniteContexts

section TwoSided

variable [Monoid X]

/-- Two-sided compositional context query:
`query (left, right) x = fstar (left * x * right)`. -/
def TwoSidedContextQuery (fstar : X → Y) : (X × X) → X → Y :=
  fun ctx x => fstar (ctx.1 * x * ctx.2)

/-- A representation is sufficient for all two-sided compositional contexts. -/
def TwoSidedContextSufficient
    (rep : X → Rep)
    (fstar : X → Y) : Prop :=
  QuerySufficient rep (TwoSidedContextQuery fstar)

/-- Leaf state induced by a shared endomorphic `g` on one carrier space. -/
abbrev uniformComposableLeaf
    (G : UniformG X Carrier) : X → Carrier :=
  UniformG.leaf G

/-- Merge state induced by the same shared endomorphic `g` on that carrier. -/
abbrev uniformComposableMerge
    (G : UniformG X Carrier) : Carrier → Carrier → Carrier :=
  UniformG.merge G

/-- Two-sided specialization of the no-bad-collision theorem. -/
theorem twoSidedContextSufficient_no_collision_of_distinguished_context
    {rep : X → Rep}
    {fstar : X → Y}
    (hSuff : TwoSidedContextSufficient rep fstar)
    {x y left right : X}
    (hSep : fstar (left * x * right) ≠ fstar (left * y * right)) :
    rep x ≠ rep y := by
  exact querySufficient_no_collision_of_distinguished_context
    (rep := rep)
    (query := TwoSidedContextQuery fstar)
    hSuff
    (c := (left, right))
    hSep

/-- Exact `leaf -> merge -> merge -> readout` behavior implies the leaf state is
two-sided context-sufficient. This is the algebraic state-fold lemma; the
shared-`g` specialization is `uniformComposedTwoSidedReadoutExact_implies...`
below. -/
theorem composedTwoSidedReadoutExact_implies_twoSidedContextSufficient
    {leaf : X → State}
    {merge : State → State → State}
    {readout : State → Y}
    {fstar : X → Y}
    (hExact :
      ∀ left x right,
        readout (merge (merge (leaf left) (leaf x)) (leaf right)) =
          fstar (left * x * right)) :
    TwoSidedContextSufficient leaf fstar := by
  intro x y hxy ctx
  rcases ctx with ⟨left, right⟩
  calc
    fstar (left * x * right) =
        readout (merge (merge (leaf left) (leaf x)) (leaf right)) := by
          exact (hExact left x right).symm
    _ = readout (merge (merge (leaf left) (leaf y)) (leaf right)) := by
          rw [hxy]
    _ = fstar (left * y * right) := hExact left y right

/-- Shared-`g` two-sided exactness implies contextual sufficiency of the induced
leaf representation. The same `G.g` is used to make leaf states and merge
states. -/
theorem uniformComposedTwoSidedReadoutExact_implies_twoSidedContextSufficient
    {G : UniformG X Carrier}
    {readout : Carrier → Y}
    {fstar : X → Y}
    (hExact :
      ∀ left x right,
        readout
          (uniformComposableMerge G
            (uniformComposableMerge G
              (uniformComposableLeaf G left)
              (uniformComposableLeaf G x))
            (uniformComposableLeaf G right)) =
          fstar (left * x * right)) :
    TwoSidedContextSufficient (uniformComposableLeaf G) fstar :=
  composedTwoSidedReadoutExact_implies_twoSidedContextSufficient
    (leaf := uniformComposableLeaf G)
    (merge := uniformComposableMerge G)
    (readout := readout)
    (fstar := fstar)
    hExact

end TwoSided

/-! ## Approximate sufficiency

The deterministic core above describes exact-collision sufficiency:
`rep x = rep y` forces every contextual query response to agree on the nose.
A learned `g` will not satisfy that exactly. The bridge below treats the
approximate case: if a composed state/readout pipeline reproduces the oracle
within `ε` uniformly across two-sided contexts, then collisions of the leaf
state map cost at most `2 * ε` in any oracle response. The shared-`g` theorem
packages the same bridge with one `G.g` applied at both leaf and merge sites.

Mutual-information-flavored objectives (Chen et al., 2010.10079; InfoNCE, MINE)
remain the *learning* motivation; the Lean lane stays in deterministic
metric-space slack tracking. See `lean3/docs/INFORMATION_SUFFICIENCY_BRIDGE.md`
for the scope rationale. -/

section ApproximateSufficiency

variable [PseudoMetricSpace Y]

/-- Approximate query sufficiency: representation collisions cost at most `ε`
in any contextual query response. The exact `QuerySufficient` definition is the
`ε = 0` instance up to indistinguishability of the metric. -/
def QuerySufficientWithin
    (ε : ℝ)
    (rep : X → Rep)
    (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → ∀ c : Ctx, dist (query c x) (query c y) ≤ ε

/-- The full response signature realizes zero-error approximate sufficiency. -/
theorem responseSignatures_querySufficientWithin_zero
    (query : Ctx → X → Y) :
    QuerySufficientWithin (Rep := Ctx → Y) 0 (ResponseSignature query) query := by
  intro x y hxy c
  have hEq : query c x = query c y := congrFun hxy c
  rw [hEq, dist_self]

/-- Approximate sufficiency is monotone in the slack budget. -/
theorem querySufficientWithin_mono
    {ε ε' : ℝ}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    (hLe : ε ≤ ε')
    (hSuff : QuerySufficientWithin ε rep query) :
    QuerySufficientWithin ε' rep query := by
  intro x y hxy c
  exact (hSuff hxy c).trans hLe

/-- Exact sufficiency implies zero-error approximate sufficiency. -/
theorem querySufficient_implies_querySufficientWithin_zero
    {rep : X → Rep}
    {query : Ctx → X → Y}
    (hSuff : QuerySufficient rep query) :
    QuerySufficientWithin 0 rep query := by
  intro x y hxy c
  have hEq : query c x = query c y := hSuff hxy c
  rw [hEq, dist_self]

/-- If sampled contexts cover all contextual responses within slack, then
finite-context collision control implies approximate query sufficiency. -/
theorem finiteContext_within_implies_querySufficientWithin
    [DecidableEq Ctx]
    {contexts : Finset Ctx}
    {ε δ : ℝ}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    (hCover : FiniteContextCoversWithin contexts ε δ query)
    (hWithin : QuerySufficientWithinOn contexts ε rep query) :
    QuerySufficientWithin δ rep query := by
  intro x y hxy c
  exact hCover (fun c' hc' => hWithin hxy c' hc') c

end ApproximateSufficiency

/-! ## Metric / near-collision sufficiency -/

section NearCollisionSufficiency

variable [PseudoMetricSpace Rep] [PseudoMetricSpace Y]

/-- Metric contextual sufficiency: if two representations are within `δ`, then
all contextual responses are within `ε`. This is the continuous-state analogue
of exact-collision sufficiency for learned representations. -/
def QuerySufficientNearWithin
    (δ ε : ℝ)
    (rep : X → Rep)
    (query : Ctx → X → Y) : Prop :=
  ∀ ⦃x y : X⦄,
    dist (rep x) (rep y) ≤ δ →
      ∀ c : Ctx, dist (query c x) (query c y) ≤ ε

/-- Approximate realization of contextual responses by a readout on the
representation state. -/
def ContextReadoutRealizesWithin
    (ε : ℝ)
    (rep : X → Rep)
    (query : Ctx → X → Y)
    (readout : Rep → Ctx → Y) : Prop :=
  ∀ x c, dist (readout (rep x) c) (query c x) ≤ ε

/-- Radius-local readout stability: states within `δ` have readouts within
`η` for every context. This is the theorem-level form of a Lipschitz-style
continuity assumption, without committing to a global Lipschitz constant. -/
def ContextReadoutNearPreserving
    (δ η : ℝ)
    (readout : Rep → Ctx → Y) : Prop :=
  ∀ ⦃r s : Rep⦄,
    dist r s ≤ δ →
      ∀ c : Ctx, dist (readout r c) (readout s c) ≤ η

/-- Exact readout plus radius-local readout stability implies metric contextual
sufficiency. -/
theorem contextReadoutNearPreserving_implies_querySufficientNearWithin
    {δ η : ℝ}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    {readout : Rep → Ctx → Y}
    (hRealizes : ContextReadoutRealizes rep query readout)
    (hNear : ContextReadoutNearPreserving δ η readout) :
    QuerySufficientNearWithin δ η rep query := by
  intro x y hxy c
  calc
    dist (query c x) (query c y)
        = dist (readout (rep x) c) (readout (rep y) c) := by
          rw [(hRealizes x c).symm, (hRealizes y c).symm]
    _ ≤ η := hNear hxy c

/-- Approximate readout plus radius-local readout stability implies metric
contextual sufficiency with the readout error paid on both sides. -/
theorem contextReadoutApproxNearPreserving_implies_querySufficientNearWithin
    {δ ε η : ℝ}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    {readout : Rep → Ctx → Y}
    (hApprox : ContextReadoutRealizesWithin ε rep query readout)
    (hNear : ContextReadoutNearPreserving δ η readout) :
    QuerySufficientNearWithin δ (ε + (η + ε)) rep query := by
  intro x y hxy c
  have hx : dist (query c x) (readout (rep x) c) ≤ ε := by
    simpa [dist_comm] using hApprox x c
  have hy : dist (readout (rep y) c) (query c y) ≤ ε := hApprox y c
  have hxyReadout : dist (readout (rep x) c) (readout (rep y) c) ≤ η :=
    hNear hxy c
  have hTriReadout :
      dist (readout (rep x) c) (query c y) ≤
        dist (readout (rep x) c) (readout (rep y) c) +
          dist (readout (rep y) c) (query c y) :=
    dist_triangle _ _ _
  calc
    dist (query c x) (query c y)
        ≤ dist (query c x) (readout (rep x) c) +
          dist (readout (rep x) c) (query c y) := dist_triangle _ _ _
    _ ≤ dist (query c x) (readout (rep x) c) +
          (dist (readout (rep x) c) (readout (rep y) c) +
            dist (readout (rep y) c) (query c y)) := by
          exact add_le_add_right hTriReadout (dist (query c x) (readout (rep x) c))
    _ ≤ ε + (η + ε) := add_le_add hx (add_le_add hxyReadout hy)

end NearCollisionSufficiency

section ApproximateTwoSided

variable [Monoid X] [PseudoMetricSpace Y]

/-- Two-sided compositional approximate contextual sufficiency:
`leaf` collisions cost at most `ε` in any `fstar (left * x * right)` response. -/
def TwoSidedContextSufficientWithin
    (ε : ℝ)
    (rep : X → Rep)
    (fstar : X → Y) : Prop :=
  QuerySufficientWithin ε rep (TwoSidedContextQuery fstar)

/-- Bridge from approximate composed state/readout behavior to approximate
two-sided contextual sufficiency. This is the algebraic state-fold lemma; use
`uniformComposedTwoSidedReadoutWithinEps_implies...` for the shared-`g`
specialization. -/
theorem composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
    {ε : ℝ}
    {leaf : X → State}
    {merge : State → State → State}
    {readout : State → Y}
    {fstar : X → Y}
    (hApprox :
      ∀ left x right,
        dist (readout (merge (merge (leaf left) (leaf x)) (leaf right)))
             (fstar (left * x * right)) ≤ ε) :
    TwoSidedContextSufficientWithin (2 * ε) leaf fstar := by
  intro x y hxy ctx
  rcases ctx with ⟨left, right⟩
  set rx := readout (merge (merge (leaf left) (leaf x)) (leaf right)) with hrx_def
  set ry := readout (merge (merge (leaf left) (leaf y)) (leaf right)) with hry_def
  have hxApprox : dist rx (fstar (left * x * right)) ≤ ε := hApprox left x right
  have hyApprox : dist ry (fstar (left * y * right)) ≤ ε := hApprox left y right
  have hRxy : rx = ry := by
    simp [hrx_def, hry_def, hxy]
  show dist (fstar (left * x * right)) (fstar (left * y * right)) ≤ 2 * ε
  calc dist (fstar (left * x * right)) (fstar (left * y * right))
      ≤ dist (fstar (left * x * right)) rx + dist rx (fstar (left * y * right)) :=
            dist_triangle _ _ _
    _ = dist rx (fstar (left * x * right)) + dist rx (fstar (left * y * right)) := by
            rw [dist_comm rx (fstar (left * x * right))]
    _ = dist rx (fstar (left * x * right)) + dist ry (fstar (left * y * right)) := by
            rw [hRxy]
    _ ≤ ε + ε := add_le_add hxApprox hyApprox
    _ = 2 * ε := by ring

/-- Shared-`g` approximate two-sided bridge. The same endomorphic `G.g` is used
at leaf and merge sites on one carrier space. -/
theorem uniformComposedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
    {ε : ℝ}
    {G : UniformG X Carrier}
    {readout : Carrier → Y}
    {fstar : X → Y}
    (hApprox :
      ∀ left x right,
        dist
          (readout
            (uniformComposableMerge G
              (uniformComposableMerge G
                (uniformComposableLeaf G left)
                (uniformComposableLeaf G x))
              (uniformComposableLeaf G right)))
          (fstar (left * x * right)) ≤ ε) :
    TwoSidedContextSufficientWithin (2 * ε) (uniformComposableLeaf G) fstar :=
  composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
    (ε := ε)
    (leaf := uniformComposableLeaf G)
    (merge := uniformComposableMerge G)
    (readout := readout)
    (fstar := fstar)
    hApprox

/-- The exact-collision bridge is the `ε = 0` instance of the approximate one:
if a composed pipeline matches the oracle exactly across all two-sided contexts,
then leaf collisions preserve the oracle response (recovering the existing
`composedTwoSidedReadoutExact_implies_twoSidedContextSufficient`). -/
theorem composedTwoSidedReadoutExact_implies_twoSidedContextSufficientWithin_zero
    {leaf : X → State}
    {merge : State → State → State}
    {readout : State → Y}
    {fstar : X → Y}
    (hExact :
      ∀ left x right,
        readout (merge (merge (leaf left) (leaf x)) (leaf right)) =
          fstar (left * x * right)) :
    TwoSidedContextSufficientWithin 0 leaf fstar := by
  have hApprox :
      ∀ left x right,
        dist (readout (merge (merge (leaf left) (leaf x)) (leaf right)))
             (fstar (left * x * right)) ≤ 0 := by
    intro left x right
    rw [hExact left x right, dist_self]
  have h2eps :=
    composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
      (ε := 0) (leaf := leaf) (merge := merge) (readout := readout)
      (fstar := fstar) hApprox
  -- 2 * 0 = 0
  simpa using h2eps

end ApproximateTwoSided

end FormalProofs.OPT
