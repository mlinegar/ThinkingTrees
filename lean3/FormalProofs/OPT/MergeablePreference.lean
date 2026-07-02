import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.OPT.CounterexampleExistence
import FormalProofs.OPT.InformationSufficiency

/-!
# FormalProofs/OPT/MergeablePreference.lean

## What this file is — and what it is not

This file is the **scalar-oracle characterization layer** of C-TreePO.  It
asks and answers:

> *Which preferences `f* : Strings → Y` admit a single binary merge
> `M : Y → Y → Y` on the oracle values themselves?*

This is the **narrowest, most concrete** characterization of the framework's
scope.  When it succeeds, you can implement the readout as a stateless
scalar fold; when it fails, you cannot, but you may still be able to keep a
richer *state* and read out at the root.

For the **broader state-level characterization** — which is what the C-TreePO
paper actually claims as its scope, and which subsumes this layer — see
`OPT/PreferenceScope.lean`.  The two layers are related as follows:

| Layer | Lean file | Slogan |
|---|---|---|
| **State-level** (broad) | `OPT/PreferenceScope.lean` (`MergeablePreferenceShape`) | `σ(x·y) = merge(σ(x), σ(y))` and `f* = F ∘ σ` |
| **Scalar-oracle** (narrow) | this file (`MergeablePreference`) | `f*(x·y) = M(f*(g x), f*(g y))`; specialization with `σ = f*`, `F = id` |

The state-level layer permits arbitrary nonlinear readouts `F` on top of an
additively or mergeably composable state `σ`.  This file's negative results
are about *premature scalar collapse* — what fails when you insist on
merging only final oracle values.  They do **not** rule out preferences
that admit a richer mergeable state.

## Equivalent characterizations (paper-friendly TFAE)

The scalar-oracle layer admits a tight chain of equivalences.  At the
**existential level** (over the choice of summarizer `g`), for any
scalar oracle `fstar : Strings → ℝ`, the following are equivalent:

| # | Statement | Lean witness |
|---|---|---|
| (1) | `CTreePOExpressible fstar` — some `g` makes it a `MergeablePreference` | definitional |
| (2) | **Concat determinacy.** `∀ u₁,v₁,u₂,v₂`, pairwise oracle agreement on the parts forces oracle agreement on the concats | `ConcatDeterminacy fstar` |
| (3) | **Global merge exists.** `∃ M : ℝ → ℝ → ℝ` with `fstar(u·v) = M(fstar u, fstar v)` for all `u, v` | `ctreepoExpressible_iff_exists_global_merge` |
| (4) | **Schedule invariance.** `∃ g, M` such that the M-fold of `fstar ∘ g` over every binary tree `T` over the leaves equals `fstar(S(T))` | `schedule_invariance` (forward); `ctreepo_gibbons1996_third_homomorphism` (converse) |

The equivalence `(1) ↔ (2)` is `ctreepoExpressible_iff_concatDeterminacy`
— the *completeness theorem*.  It upgrades
`not_ctreepoExpressible_of_concat_witness` from a sufficient
unexpressibility test to **the only one needed**: a scalar preference is
C-TreePO-expressible iff it has *no* concat-witness violation.

The equivalence `(2) ↔ (3)` is `ctreepoExpressible_iff_exists_global_merge`.
Forward direction takes `M := buildMerge fstar` (Classical choice over
preimages, well-defined by determinacy).  Backward is direct rewriting.

The equivalence `(3) ↔ (4)` is the **third homomorphism theorem**
(Gibbons 1996), already in the project as
`ctreepo_gibbons1996_third_homomorphism` and the forward direction
`schedule_invariance` / `fold_of_folds`.  Together they say: a function
is mergeable iff its scalar fold is invariant under all order-preserving
re-bracketings of the leaves.

So the **single-sentence characterization** of the scalar-oracle layer is:

> *`fstar` is C-TreePO-expressible iff its tree fold is schedule-
> invariant — equivalently, iff `fstar(u·v)` is determined by `(fstar u,
> fstar v)`, with no exceptions.*

At the **per-summarizer level** (with `g` fixed), the tight equivalences
are:

| # | Statement | Lean witness |
|---|---|---|
| (i) | `MergeablePreference g fstar` | definitional |
| (ii) | `A1_global g fstar ∧ A2_global g fstar ∧ A3_global g fstar` | `mergeablePreference_iff_axioms` |
| (iii) | **Doob-Dynkin sufficiency + edge-mergeability + congruence** — `g` is sufficient for `fstar` AND `∃ M` distance-congruent recovering `fstar(u·v)` from `(fstar(g u), fstar(g v))` | `mergeablePreference_iff_axioms` (unfolds to the three-prong) |

**Honesty note on the Doob-Dynkin existential.** The pseudometric
`FactorsThroughSummary` predicate is the Doob-Dynkin existential `∃ h,
dist(h(g s), fstar s) = 0`.  This is implied by `A1_global` (with the
canonical decoder `h := fstar`) but is *strictly weaker* without further
constraints on `h`.  The full equivalence `A1_global ↔ FactorsThroughSummary`
holds only when restricted to canonical decoders.  See Section 3.

For the **broader state-level layer** (`MergeablePreferenceShape` in
`PreferenceScope.lean`), the parallel TFAE replaces "single global scalar
merge" with "single state-level merge plus a (possibly nonlinear) readout
`F`" — strictly extending what the scalar layer expresses.  The
canonical sentence becomes:

> *`fstar` is C-TreePO-expressible at the state level iff there exist
> a sufficient state `σ` and a readout `F` with `f* = F ∘ σ` and
> `σ(x·y) = merge(σ(x), σ(y))`.*

Most "interesting" preferences (threshold-AND, HLL distinct count, LDA
likelihood, boundary-sensitive scoring) sit at the state level but
*not* the scalar level — they fail the scalar TFAE precisely because
the per-leaf scalar collapses too early.

## What the scalar-oracle layer expresses

A preference `f*` is C-TreePO-expressible **at the scalar-oracle layer** iff
there is a sketch `g` such that:

* **(C1, sufficiency)** You can read off `f*(s)` from `g(s)` alone — i.e.
  `f*` factors through `g`, up to oracle distance zero (Doob-Dynkin).
* **(C3 + A3, edge-wise mergeability)** There is a *single* binary operator
  `M : Y → Y → Y` (congruent in distance) that recovers `f*(u·v)` from the
  pair `(f*(g u), f*(g v))`.
* **(C2, idempotence)** Re-summarizing an already-summarized string is
  inert.  This drops out of (C1) automatically (`A1_implies_L3` in
  `GlobalAssumptions.lean`); it is *not* an independent requirement.

When `Y = ℝ` and `M = +`, this collapses to **additively-separable utility**
in the sense of Debreu (1959, *Theory of Value*, Ch. 4).  When `Y = ℝ` and
`M = max`, you get Leontief/extremal aggregation.  When `Y` is itself a
sketch state, you recover classical mergeable summaries (Agarwal et al.
2013).  See `mergeablePreference_of_additiveSeparable` for the formal
Debreu instance.

**Important reframing.** Debreu's classical additivity attaches to the
*final utility*: `f*(x·y) = f*(x) + f*(y)`.  The state-level layer relaxes
this to *additivity of the sufficient state*: `σ(x·y) = σ(x) + σ(y)`, with
arbitrary nonlinear `F : σ ↦ f*`.  See `AdditivelySeparableThroughState`
in `PreferenceScope.lean`.  Most "interesting" preferences (threshold-AND,
HLL distinct count, LDA likelihood, boundary-sensitive scoring) fit the
state-level layer but *not* the scalar-oracle layer.

## What the scalar-oracle layer does NOT express

The framework **cannot** express, *as a scalar oracle*, preferences that
depend on:

* **Cross-boundary token interactions** (e.g. bigrams spanning a cut) —
  per-leaf scalar summaries cannot retain the side-of-cut tokens needed to
  reconstruct cross-boundary statistics.  Section 5b
  (`not_ctreepoExpressible_cross_boundary_bigram`).
  *State-level workaround:* keep boundary tokens in the merge state
  (`supported_boundary_interaction` in `PreferenceScope.lean`).
* **Conjunctive thresholds across cuts** (e.g.
  `u = 𝟙{count_L ≥ k₁ ∧ count_R ≥ k₂}`) — once `f*` thresholds, distinct
  underlying counts collapse to the same scalar, so the global merge `M`
  is over-determined.  Section 5a
  (`not_ctreepoExpressible_threshold_and`).
  *State-level workaround:* keep both counts in the state and threshold
  only at the readout (`supported_nonseparable_complementarity`).
* **Rank statistics and set cardinalities** (median, IQR, mode, distinct
  count, etc.) — these need cross-child information that no congruent
  scalar `M` can recover.  Section 5c
  (`not_ctreepoExpressible_distinct_count`).
  *State-level workaround:* keep an HLL register array, multiset, or
  histogram and query at the root (`hll_state_level_preference_shape`,
  `supported_histogram_state_any_utility`).

`min`, `max`, `sum`, `count`, additive log-likelihoods, and any
bag-of-words feature with linear weights all fit the scalar-oracle layer
directly — these are the "easy" positive cases.

## The clean negative criterion

The workhorse `not_ctreepoExpressible_of_concat_witness` (Section 5) is
the simplest scalar-oracle non-expressibility test:

> If there exist `u₁, v₁, u₂, v₂` with `f*(u₁) = f*(u₂)`, `f*(v₁) = f*(v₂)`,
> but `f*(u₁·v₁) ≠ f*(u₂·v₂)`, then no summarizer makes `f*` a
> `MergeablePreference`.

All three counterexamples in Section 5 are 5-line corollaries.

## Honesty note on the "iff factorization"

The user-facing slogan "`f*` factors through `g`" is *necessary but not
sufficient* for `MergeablePreference g fstar`.  Two distinct gaps exist:

1. The merge axioms `A2 + A3` are independent of pure sufficiency.  A
   function whose oracle value happens to be readable from its summary may
   still fail to admit a global congruent merge.
2. The existential decoder `∃ h, dist(h(g s), fstar s) = 0` form of
   "factorization" is *strictly weaker* than `A1_global g fstar`, because
   an arbitrary `h` need not coincide with `fstar` on the range of `g`.
   Only the canonical choice `h := fstar` recovers `A1_global` for free.

Both gaps are documented at the relevant theorem statements.

## Doob-Dynkin bridge

Section 7 connects the pseudometric `FactorsThroughSummary` (this file) to
the measure-theoretic `OracleIndexedConditionalDensity`
(`InformationSufficiency.lean`):

> If `g` is sufficient for `fstar` (`A1_global`) and `Y` is a `MetricSpace`,
> then any oracle-indexed conditional density `p(x, y) = pbar(fstar x, y)`
> is also invariant under `g`: `p(x, y) = p(g x, y)`.

The `MetricSpace` (rather than `PseudoMetricSpace`) hypothesis is
essential — pseudometric `dist = 0` does not collapse to equality.  All
concrete oracle types in the project carry `MetricSpace`, so this is not
a binding restriction in practice.

## Cross-references

* **Broader state-level characterization:** `OPT/PreferenceScope.lean` —
  `MergeablePreferenceShape`, `AdditivelySeparableThroughState`, the
  supported-nonseparable examples (`supported_nonseparable_complementarity`,
  `supported_boundary_interaction`, `supported_histogram_state_any_utility`,
  `supported_lda_likelihood_histogram_utility`), and a `Scalar-Oracle
  Boundary` section that wraps this file's exports as paper-aligned
  abbrevs.
* **Empirical Python suite:** `docs/nonseparable_preference_suite_spec.md`
  exercises the same DGPs we prove unexpressible at the scalar layer.
* **Information-sufficiency bridge:** `lean3/docs/INFORMATION_SUFFICIENCY_BRIDGE.md`
  documents the measure-theoretic Doob-Dynkin layer.
* **Classical sketches:** `OPT/ClassicalSketchLocalLaws.lean` instantiates
  the local laws for sketch state spaces;
  `scalarDistinctCount_not_child_cardinality_mergeable` there is a
  closely-related two-element-universe version of Section 5c.
* **Conceptual companion:** `docs/preference_scope_ctreepo.md` is the
  prose companion to `PreferenceScope.lean`, with the diagnostic checklist
  and the supported-nonseparable examples at the state level.
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

/-! ## Section 2 — The characterization structure -/

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- **Mergeable preference certificate.**

A bundled witness that the summarizer `g` together with the oracle `fstar`
satisfies the three global preservation axioms.  This is the formal
counterpart of "the preference is C-TreePO-expressible via `g`".

Field naming uses the *paper's* vocabulary (`sufficiency`, `twoRoute`,
`mergeExists`) rather than the Lean A1/A2/A3 names so the structure is
readable without cross-referencing the global-assumptions file.

The fourth law (C2 / idempotence on summary range) is intentionally absent:
`A1_global` already implies it (`A1_implies_L3` in
`GlobalAssumptions.lean`).  Listing it would be redundant. -/
structure MergeablePreference
    (g : Strings → Strings) (fstar : Strings → Y) : Prop where
  /-- C1: Oracle sufficiency — `g` preserves `fstar` up to distance zero. -/
  sufficiency : A1_global g fstar
  /-- C3: Two-route compatibility — `fstar(u·v) ≈ fstar(g(g u · g v))`. -/
  twoRoute    : A2_global g fstar
  /-- A3: A single global congruent merge `M : Y → Y → Y` exists. -/
  mergeExists : A3_global g fstar

/-- **Existential characterization.**  A preference is *C-TreePO-expressible*
iff *some* summarizer makes it a `MergeablePreference`.  Negative results
target this predicate, since to rule out the framework we must rule out
*every* summarizer. -/
def CTreePOExpressible (fstar : Strings → Y) : Prop :=
  ∃ g : Strings → Strings, MergeablePreference g fstar

/-- The certificate version implies the existential predicate. -/
theorem CTreePOExpressible.of_mergeable
    {g : Strings → Strings} {fstar : Strings → Y}
    (h : MergeablePreference g fstar) : CTreePOExpressible fstar :=
  ⟨g, h⟩

/-- The structure unfolded as a literal conjunction of the three axioms.
This is the trivial "characterization" — it is just the structure constructor
in propositional form.  Useful for callers who want to inhabit the structure
from existing axiom-shaped hypotheses. -/
theorem mergeablePreference_iff_axioms
    {g : Strings → Strings} {fstar : Strings → Y} :
    MergeablePreference g fstar ↔
      A1_global g fstar ∧ A2_global g fstar ∧ A3_global g fstar := by
  constructor
  · intro h; exact ⟨h.sufficiency, h.twoRoute, h.mergeExists⟩
  · rintro ⟨h1, h2, h3⟩; exact ⟨h1, h2, h3⟩

/-! ## Section 3 — Doob-Dynkin factorization

The Doob-Dynkin lemma (in our setting) says: `fstar` is determined by
`g(·)` iff there is a function `h` with `fstar = h ∘ g`.  In the
pseudometric setting the equality is up to distance zero. -/

/-- `fstar` factors through `g` (Doob-Dynkin form): there exists a decoder
`h : Strings → Y` such that `h(g s)` agrees with `fstar s` up to oracle
distance zero.

This is the Doob-Dynkin half of the C1 condition: it captures
"`fstar` depends on `s` only through `g s`".  See the honesty note in
the file docstring for why this is *strictly weaker* than `A1_global`. -/
def FactorsThroughSummary (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∃ h : Strings → Y, ∀ s : Strings, dist (h (g s)) (fstar s) = 0

/-- Easy direction of Doob-Dynkin: `A1_global` (and hence
`MergeablePreference`) implies the existence of a decoder.  The canonical
choice is `h := fstar` itself. -/
theorem factorsThroughSummary_of_a1_global
    {g : Strings → Strings} {fstar : Strings → Y}
    (hA1 : A1_global g fstar) :
    FactorsThroughSummary g fstar := by
  refine ⟨fstar, ?_⟩
  intro s
  have := hA1 s
  unfold D at this
  exact this

/-- Easy direction, packaged for `MergeablePreference`. -/
theorem factorsThroughSummary_of_mergeable
    {g : Strings → Strings} {fstar : Strings → Y}
    (h : MergeablePreference g fstar) :
    FactorsThroughSummary g fstar :=
  factorsThroughSummary_of_a1_global h.sufficiency

/-! Why we do not state the converse `FactorsThroughSummary → A1_global`:
For an arbitrary decoder `h`, the triangle inequality only gives

  `dist(fstar(g s), fstar s) ≤ dist(fstar(g s), h(g s)) + dist(h(g s), fstar s)`

The second term is `0` by hypothesis, but the first term equates `fstar` and
`h` on the range of `g`, which is exactly what `A1_global` would let us
conclude — circular.  The converse holds only when `h` is chosen as `fstar`,
in which case it is trivial.  Equivalently: the Doob-Dynkin existential is
equivalent to `A1_global` *up to choice of canonical decoder*, which is
itself a substantive choice. -/

/-! ## Section 4 — Additive separability instance (the Debreu connection)

**Reference:** Debreu, G. (1959), *Theory of Value*, Ch. 4 — additively
separable utility.  This is interpretive only; we do *not* derive
additivity from preference axioms here.  We only show that an *assumed*
additivity yields a `MergeablePreference` instance. -/

/-- Concrete additivity hypothesis on a summarizer:
`fstar(u·v) = fstar(g u) + fstar(g v)`.  This packages C1 (`A1_global`)
together with the additive merge identity. -/
def AdditiveSeparableSummarizer
    {Strings : Type*} [Monoid Strings]
    (g : Strings → Strings) (fstar : Strings → ℝ) : Prop :=
  A1_global g fstar ∧
  ∀ u v : Strings, fstar (u * v) = fstar (g u) + fstar (g v)

/-- An additively-separable summarizer is a mergeable preference, with
`M := (· + ·)`. -/
theorem mergeablePreference_of_additiveSeparable
    {Strings : Type*} [Monoid Strings]
    {g : Strings → Strings} {fstar : Strings → ℝ}
    (h : AdditiveSeparableSummarizer g fstar) :
    MergeablePreference g fstar := by
  obtain ⟨hA1, hAdd⟩ := h
  -- A1 collapses `dist = 0` to ℝ-equality via `eq_of_dist_eq_zero`.
  have a1_eq : ∀ s, fstar (g s) = fstar s := fun s =>
    eq_of_dist_eq_zero (hA1 s)
  refine ⟨hA1, ?_, ?_⟩
  · -- A2: fstar(u*v) = fstar(g(g u * g v))
    intro u v
    have h2 : fstar (g (g u * g v)) = fstar (g u) + fstar (g v) := by
      rw [a1_eq (g u * g v), hAdd (g u) (g v), a1_eq (g u), a1_eq (g v)]
    show dist (fstar (u * v)) (fstar (g (g u * g v))) = 0
    rw [hAdd u v, h2]; exact dist_self _
  · -- A3: take M := (· + ·)
    refine ⟨(· + ·), ?_, ?_⟩
    · intro u v
      have h2 : fstar (g (g u * g v)) = fstar (g u) + fstar (g v) := by
        rw [a1_eq (g u * g v), hAdd (g u) (g v), a1_eq (g u), a1_eq (g v)]
      show dist (fstar (g (g u * g v))) (fstar (g u) + fstar (g v)) = 0
      rw [h2]; exact dist_self _
    · intro y₁ y₁' y₂ y₂' hd1 hd2
      have e1 : y₁ = y₁' := eq_of_dist_eq_zero hd1
      have e2 : y₂ = y₂' := eq_of_dist_eq_zero hd2
      rw [e1, e2]; exact dist_self _

/-- **Corollary.** Any additively-separable utility is C-TreePO-expressible. -/
theorem ctreepoExpressible_of_additiveSeparable
    {Strings : Type*} [Monoid Strings]
    {g : Strings → Strings} {fstar : Strings → ℝ}
    (h : AdditiveSeparableSummarizer g fstar) :
    CTreePOExpressible fstar :=
  ⟨g, mergeablePreference_of_additiveSeparable h⟩

/-! ## Section 5 — Counterexamples (non-expressibility)

The general lemma below is the workhorse: any preference that distinguishes
two concatenations whose components have matching oracle values cannot
admit a global congruent merge.  Concrete DGPs from
`docs/nonseparable_preference_suite_spec.md` are 1-line corollaries. -/

/-- **General non-mergeability lemma.**  If there exist `u₁, v₁, u₂, v₂`
such that the oracle agrees pairwise on the components but disagrees on
the concatenations, then the preference is not C-TreePO-expressible.

The intuition: any candidate global merge `M` would have to satisfy
`M(fstar(g u₁), fstar(g v₁)) = fstar(u₁·v₁)` and
`M(fstar(g u₂), fstar(g v₂)) = fstar(u₂·v₂)`.  But by `A1_global` the
arguments to `M` are pairwise equal in `ℝ`, while the right-hand sides
differ — contradiction by congruence of `M`.

This is the formal statement of "merge-determined oracle": expressibility
requires `fstar(u·v)` to depend on `(u, v)` only through
`(fstar(g u), fstar(g v))`. -/
theorem not_ctreepoExpressible_of_concat_witness
    {Strings : Type*} [Monoid Strings]
    {fstar : Strings → ℝ}
    {u₁ v₁ u₂ v₂ : Strings}
    (hu : fstar u₁ = fstar u₂)
    (hv : fstar v₁ = fstar v₂)
    (hne : fstar (u₁ * v₁) ≠ fstar (u₂ * v₂)) :
    ¬ CTreePOExpressible fstar := by
  rintro ⟨g, hMP⟩
  obtain ⟨hA1, hA2, hA3⟩ := mergeablePreference_iff_axioms.mp hMP
  obtain ⟨M, hMeq, _hMcong⟩ := hA3
  -- A1, A2, and the M-agreement clause of A3 all collapse `dist = 0` to
  -- ℝ-equality via `eq_of_dist_eq_zero`.
  have a1_eq : ∀ s, fstar (g s) = fstar s := fun s =>
    eq_of_dist_eq_zero (hA1 s)
  have a2_eq : ∀ u v : Strings, fstar (u * v) = fstar (g (g u * g v)) :=
    fun u v => eq_of_dist_eq_zero (hA2 u v)
  have m_eq : ∀ u v : Strings,
      fstar (g (g u * g v)) = M (fstar (g u)) (fstar (g v)) :=
    fun u v => eq_of_dist_eq_zero (hMeq u v)
  -- Step 4: assemble.  fstar(uᵢ·vᵢ) = M(fstar(g uᵢ), fstar(g vᵢ))
  --                                  = M(fstar uᵢ, fstar vᵢ) by a1_eq
  have lhs1 : fstar (u₁ * v₁) = M (fstar u₁) (fstar v₁) := by
    rw [a2_eq u₁ v₁, m_eq u₁ v₁, a1_eq u₁, a1_eq v₁]
  have lhs2 : fstar (u₂ * v₂) = M (fstar u₂) (fstar v₂) := by
    rw [a2_eq u₂ v₂, m_eq u₂ v₂, a1_eq u₂, a1_eq v₂]
  -- Step 5: by hu, hv the right-hand sides of lhs1 and lhs2 are equal.
  have h_rhs_eq : M (fstar u₁) (fstar v₁) = M (fstar u₂) (fstar v₂) := by
    rw [hu, hv]
  -- Final contradiction.
  exact hne (lhs1.trans (h_rhs_eq.trans lhs2.symm))

/-! ### Concat-determinacy and the completeness theorem

The workhorse lemma above says: if some `u₁,v₁,u₂,v₂` violates concat
determinacy, the preference is *not* expressible.  The result of this
subsection is the **converse**: the negative criterion is *complete* for
the scalar-oracle layer.  A scalar `fstar : Strings → ℝ` is C-TreePO-
expressible iff it has *no* such violation.

This turns `not_ctreepoExpressible_of_concat_witness` from a *test that
sometimes rules things out* into the *complete classification* of scalar-
oracle-expressible preferences. -/

/-- **Concat determinacy.**  The oracle of a concatenation is determined
by the oracle values of the parts.  This is the "no collision" form of
the negative criterion: pairwise oracle agreement on components forces
oracle agreement on concats. -/
def ConcatDeterminacy {Strings : Type*} [Mul Strings] (fstar : Strings → ℝ) : Prop :=
  ∀ u₁ v₁ u₂ v₂ : Strings,
    fstar u₁ = fstar u₂ → fstar v₁ = fstar v₂ →
      fstar (u₁ * v₁) = fstar (u₂ * v₂)

/-- Build a global scalar merge from a `ConcatDeterminacy` witness.  For
`(a, b)` in the image of `fstar × fstar`, picks any preimages and
returns `fstar` of their concatenation; outside the image, returns `0`.
Well-definedness on the image follows from `ConcatDeterminacy`. -/
noncomputable def buildMerge {Strings : Type*} [Mul Strings]
    (fstar : Strings → ℝ) : ℝ → ℝ → ℝ := fun a b =>
  if h : (∃ u : Strings, fstar u = a) ∧ (∃ v : Strings, fstar v = b) then
    fstar (h.1.choose * h.2.choose)
  else 0

/-- The key lemma about `buildMerge`: under `ConcatDeterminacy`, it
recovers the oracle of any concatenation from the oracle values of the
parts. -/
lemma buildMerge_apply
    {Strings : Type*} [Mul Strings] {fstar : Strings → ℝ}
    (hCD : ConcatDeterminacy fstar) (u v : Strings) :
    buildMerge fstar (fstar u) (fstar v) = fstar (u * v) := by
  unfold buildMerge
  have hex : (∃ u' : Strings, fstar u' = fstar u) ∧
              (∃ v' : Strings, fstar v' = fstar v) :=
    ⟨⟨u, rfl⟩, ⟨v, rfl⟩⟩
  rw [dif_pos hex]
  exact hCD _ _ _ _ hex.1.choose_spec hex.2.choose_spec

/-- **Completeness of the negative criterion.**  In the scalar-oracle
setting `Y = ℝ`, `CTreePOExpressible fstar` is *equivalent* to
`ConcatDeterminacy fstar`.  This makes
`not_ctreepoExpressible_of_concat_witness` not just a sufficient
unexpressibility test but the **only one needed**: a scalar preference
fails to be C-TreePO-expressible iff it has a concat-witness violation.

Forward direction: contrapositive of
`not_ctreepoExpressible_of_concat_witness`.

Backward direction: take `g := id`, `M := buildMerge fstar`.  Concat
determinacy makes `M` well-defined and recovers the oracle on every
concatenation. -/
theorem ctreepoExpressible_iff_concatDeterminacy
    {Strings : Type*} [Monoid Strings] {fstar : Strings → ℝ} :
    CTreePOExpressible fstar ↔ ConcatDeterminacy fstar := by
  constructor
  · -- Forward: contrapositive of the workhorse lemma.
    intro hExpr u₁ v₁ u₂ v₂ hu hv
    by_contra hne
    exact not_ctreepoExpressible_of_concat_witness hu hv hne hExpr
  · -- Backward: construct g := id and M := buildMerge fstar.
    intro hCD
    refine ⟨id, ?_, ?_, ?_⟩
    · intro z; show dist (fstar z) (fstar z) = 0; exact dist_self _
    · intro u v
      show dist (fstar (u * v)) (fstar (u * v)) = 0; exact dist_self _
    · refine ⟨buildMerge fstar, ?_, ?_⟩
      · intro u v
        show dist (fstar (u * v)) (buildMerge fstar (fstar u) (fstar v)) = 0
        rw [buildMerge_apply hCD]; exact dist_self _
      · intro y₁ y₁' y₂ y₂' hd1 hd2
        have e1 : y₁ = y₁' := eq_of_dist_eq_zero hd1
        have e2 : y₂ = y₂' := eq_of_dist_eq_zero hd2
        rw [e1, e2]; exact dist_self _

/-- **Negative form of the completeness theorem.**  A scalar preference
is *not* C-TreePO-expressible iff *some* concat-witness violates
determinacy.  Direct corollary of
`ctreepoExpressible_iff_concatDeterminacy`. -/
theorem not_ctreepoExpressible_iff_exists_concat_witness
    {Strings : Type*} [Monoid Strings] {fstar : Strings → ℝ} :
    ¬ CTreePOExpressible fstar ↔
    ∃ u₁ v₁ u₂ v₂ : Strings,
      fstar u₁ = fstar u₂ ∧ fstar v₁ = fstar v₂ ∧
        fstar (u₁ * v₁) ≠ fstar (u₂ * v₂) := by
  rw [ctreepoExpressible_iff_concatDeterminacy]
  unfold ConcatDeterminacy
  push_neg
  rfl

/-! ### A "merge function exists" reformulation

Equivalent to concat determinacy and to `CTreePOExpressible`: there is a
single binary function `M : ℝ × ℝ → ℝ` such that `fstar(u·v) = M(fstar u,
fstar v)` for all `u, v`.  This is the slogan-form of the scalar-oracle
characterization. -/

/-- A scalar preference is C-TreePO-expressible iff there is a global
binary `M : ℝ × ℝ → ℝ` that recovers `fstar(u·v)` from `(fstar u, fstar v)`. -/
theorem ctreepoExpressible_iff_exists_global_merge
    {Strings : Type*} [Monoid Strings] {fstar : Strings → ℝ} :
    CTreePOExpressible fstar ↔
    ∃ M : ℝ → ℝ → ℝ, ∀ u v : Strings, fstar (u * v) = M (fstar u) (fstar v) := by
  rw [ctreepoExpressible_iff_concatDeterminacy]
  constructor
  · intro hCD
    exact ⟨buildMerge fstar, fun u v => (buildMerge_apply hCD u v).symm⟩
  · rintro ⟨M, hM⟩ u₁ v₁ u₂ v₂ hu hv
    rw [hM u₁ v₁, hM u₂ v₂, hu, hv]

/-! ### 5a — Threshold AND counterexample

Matches `dgp1_complementarity_and` from
`docs/nonseparable_preference_suite_spec.md`: `u = 𝟙{c_left ≥ k₁ ∧ c_right ≥ k₂}`.
Carrier monoid: pairs of natural numbers under componentwise addition,
viewed multiplicatively via `Multiplicative`. -/

namespace ThresholdAND

/-- Pair-of-counts carrier as an additive type. -/
abbrev Counts : Type := ℕ × ℕ

/-- The monoid we want is multiplicative pairs of `(ℕ, +)`.  We use the
`Multiplicative` wrapper so `*` denotes componentwise addition. -/
abbrev TStrings : Type := Multiplicative Counts

/-- Project back to the additive view for defining `fstar`. -/
def toCounts (s : TStrings) : Counts := Multiplicative.toAdd s

/-- Inject a pair of counts into `TStrings`. -/
def ofCounts (c : Counts) : TStrings := Multiplicative.ofAdd c

/-- The threshold-AND oracle.  Returns 1 when both counts meet thresholds
`k₁, k₂` and 0 otherwise. -/
def thrFstar (k₁ k₂ : ℕ) (s : TStrings) : ℝ :=
  if k₁ ≤ (toCounts s).1 ∧ k₂ ≤ (toCounts s).2 then 1 else 0

/-- Computes the count of an `ofCounts` value. -/
@[simp] lemma toCounts_ofCounts (c : Counts) : toCounts (ofCounts c) = c := rfl

/-- `*` on `TStrings` is componentwise addition of underlying counts. -/
@[simp] lemma toCounts_mul (s t : TStrings) :
    toCounts (s * t) = (toCounts s + toCounts t : Counts) := rfl

/-- The non-expressibility theorem for threshold AND.

For any thresholds `k₁, k₂ ≥ 1`, no summarizer can make `thrFstar k₁ k₂`
into a mergeable preference.  Witnesses: take leaves with counts
`(k₁, 0)` and `(0, k₂)` (oracle 0); their concat has counts `(k₁, k₂)`
(oracle 1).  Compare against the doubled witnesses `(k₁, 0)·(k₁, 0)`
which has counts `(2k₁, 0)` (still oracle 0).  The oracle agrees on the
components but disagrees on the concats. -/
theorem not_ctreepoExpressible_threshold_and
    (k₁ k₂ : ℕ) (hk₁ : 1 ≤ k₁) (hk₂ : 1 ≤ k₂) :
    ¬ CTreePOExpressible (thrFstar k₁ k₂) := by
  -- Witnesses: u₁ = u₂ = (k₁, 0); v₁ = (0, k₂); v₂ = (k₁, 0).
  -- fstar u₁ = fstar u₂ = 0 (second coord 0 < k₂).
  -- fstar v₁ = fstar v₂ = 0 (first coord 0 < k₁ for v₁; second coord 0 for v₂).
  -- fstar (u₁ * v₁) = 1 (counts (k₁, k₂)).
  -- fstar (u₂ * v₂) = 0 (counts (2k₁, 0); second coord 0 < k₂).
  have h_u1 : thrFstar k₁ k₂ (ofCounts (k₁, 0)) = 0 := by
    unfold thrFstar; simp [toCounts_ofCounts]; omega
  have h_v1 : thrFstar k₁ k₂ (ofCounts (0, k₂)) = 0 := by
    unfold thrFstar; simp [toCounts_ofCounts]; omega
  have h_v2 : thrFstar k₁ k₂ (ofCounts (k₁, 0)) = 0 := h_u1
  have h_uv1 : thrFstar k₁ k₂ (ofCounts (k₁, 0) * ofCounts (0, k₂)) = 1 := by
    unfold thrFstar
    simp [toCounts_mul, toCounts_ofCounts, hk₁, hk₂]
  have h_uv2 : thrFstar k₁ k₂ (ofCounts (k₁, 0) * ofCounts (k₁, 0)) = 0 := by
    unfold thrFstar; simp [toCounts_mul, toCounts_ofCounts]; omega
  refine not_ctreepoExpressible_of_concat_witness
    (Strings := TStrings) (fstar := thrFstar k₁ k₂)
    (u₁ := ofCounts (k₁, 0)) (v₁ := ofCounts (0, k₂))
    (u₂ := ofCounts (k₁, 0)) (v₂ := ofCounts (k₁, 0))
    ?_ ?_ ?_
  · rfl
  · linarith
  · linarith

end ThresholdAND

/-! ### 5b — Cross-boundary bigram counterexample

Matches `dgp2_boundary_interaction` from
`docs/nonseparable_preference_suite_spec.md`: utility includes a
cross-boundary bigram count.  We use the simplest model — a `List Bool`
free monoid with utility = number of adjacent `(true, false)` pairs. -/

namespace CrossBoundaryBigram

/-- The free monoid on `Bool`, i.e. lists of `Bool` with concatenation. -/
abbrev BgStr : Type := FreeMonoid Bool

/-- Inject a list to the free monoid. -/
def ofList (xs : List Bool) : BgStr := xs

/-- Number of adjacent `(true, false)` pairs in a list of booleans. -/
def bigramCount : List Bool → ℕ
  | [] => 0
  | [_] => 0
  | true :: false :: rest => 1 + bigramCount (false :: rest)
  | _ :: (b :: rest) => bigramCount (b :: rest)

/-- The bigram oracle, as a real-valued function on `BgStr`. -/
def bigramFstar (s : BgStr) : ℝ := (bigramCount s.toList : ℝ)

@[simp] lemma bigramCount_singleton (b : Bool) : bigramCount [b] = 0 := rfl

@[simp] lemma bigramCount_true_false : bigramCount [true, false] = 1 := by
  unfold bigramCount; rfl

@[simp] lemma bigramCount_true_true : bigramCount [true, true] = 0 := by
  unfold bigramCount; rfl

@[simp] lemma bigramCount_false_false : bigramCount [false, false] = 0 := by
  unfold bigramCount; rfl

/-- For `FreeMonoid Bool`, `*` is `++` and `toList` strips the wrapper. -/
@[simp] lemma toList_mul (s t : BgStr) : (s * t).toList = s.toList ++ t.toList := rfl

@[simp] lemma toList_ofList (xs : List Bool) : (ofList xs).toList = xs := rfl

/-- The non-expressibility theorem for cross-boundary bigrams.

Witnesses: leaves `[true]` and `[false]` (each with bigram count 0).
Comparison: the concat `[true]·[false] = [true, false]` has count 1, while
the concat `[true]·[true] = [true, true]` has count 0.  The oracle agrees
on the components but disagrees on the concats. -/
theorem not_ctreepoExpressible_cross_boundary_bigram :
    ¬ CTreePOExpressible bigramFstar := by
  -- u₁ = u₂ = ofList [true]; v₁ = ofList [false]; v₂ = ofList [true].
  have h_u1 : bigramFstar (ofList [true]) = 0 := by
    show ((bigramCount [true] : ℕ) : ℝ) = 0; simp
  have h_v1 : bigramFstar (ofList [false]) = 0 := by
    show ((bigramCount [false] : ℕ) : ℝ) = 0; simp
  have h_v2 : bigramFstar (ofList [true]) = 0 := h_u1
  have h_uv1 : bigramFstar (ofList [true] * ofList [false]) = 1 := by
    show ((bigramCount ([true] ++ [false]) : ℕ) : ℝ) = 1
    show ((bigramCount [true, false] : ℕ) : ℝ) = 1
    simp
  have h_uv2 : bigramFstar (ofList [true] * ofList [true]) = 0 := by
    show ((bigramCount ([true] ++ [true]) : ℕ) : ℝ) = 0
    show ((bigramCount [true, true] : ℕ) : ℝ) = 0
    simp
  refine not_ctreepoExpressible_of_concat_witness
    (Strings := BgStr) (fstar := bigramFstar)
    (u₁ := ofList [true]) (v₁ := ofList [false])
    (u₂ := ofList [true]) (v₂ := ofList [true])
    ?_ ?_ ?_
  · rfl
  · linarith
  · linarith

end CrossBoundaryBigram

/-! ### 5c — Rank-statistic counterexample (distinct count)

The canonical "scalar fails / sketch succeeds" pattern from the streaming
literature.  We use *number of distinct elements* as the representative
rank statistic; the same witness pattern rules out **median, IQR, mode,
and any other quantile- or set-cardinality-based oracle**.

The state-level workaround is well-established: keep an HLL register array
(or the full multiset) until the root, then read off the cardinality.  See
`hll_state_level_preference_shape` in `OPT/PreferenceScope.lean` and
`scalarDistinctCount_not_child_cardinality_mergeable` in
`OPT/ClassicalSketchLocalLaws.lean` (a closely-related result over a
two-element universe).  This subsection is the
`MergeablePreference`-shaped statement: no scalar `M` can recover the
distinct count of `u·v` from the distinct counts of `g u` and `g v`. -/

namespace DistinctCount

/-- Free monoid on `ℕ` — finite multisets with concatenation. -/
abbrev DcStr : Type := FreeMonoid ℕ

/-- Inject a list to the free monoid. -/
def ofList (xs : List ℕ) : DcStr := xs

/-- Number of *distinct* elements in a list. -/
def distinctCount (xs : List ℕ) : ℕ := xs.toFinset.card

/-- The distinct-count oracle, real-valued. -/
def distinctFstar (s : DcStr) : ℝ := (distinctCount s.toList : ℝ)

@[simp] lemma toList_ofList (xs : List ℕ) : (ofList xs).toList = xs := rfl

@[simp] lemma toList_mul (s t : DcStr) :
    (s * t).toList = s.toList ++ t.toList := rfl

/-- The non-expressibility theorem for distinct-count.

Witnesses:
- `u₁ = [3, 5]` and `u₂ = [3, 7]` both have 2 distinct elements;
- `v₁ = v₂ = [5]` both have 1 distinct element;
- `u₁ * v₁ = [3, 5, 5]` has 2 distinct elements (`{3, 5}`);
- `u₂ * v₂ = [3, 7, 5]` has 3 distinct elements (`{3, 5, 7}`).

Same component oracle values, distinct concat oracle values ⇒ no scalar
merge `M` exists.

The same pattern (matching component oracles, mismatched concat oracles
under any `g` consistent with `A1`) rules out *median, IQR, mode, and any
other rank statistic*. -/
theorem not_ctreepoExpressible_distinct_count :
    ¬ CTreePOExpressible distinctFstar := by
  -- Concrete component values, evaluated by `decide` on ℕ then cast to ℝ.
  have e_u1 : distinctCount [3, 5] = 2 := by decide
  have e_u2 : distinctCount [3, 7] = 2 := by decide
  have e_v  : distinctCount [5]    = 1 := by decide
  have e_uv1 : distinctCount [3, 5, 5] = 2 := by decide
  have e_uv2 : distinctCount [3, 7, 5] = 3 := by decide
  refine not_ctreepoExpressible_of_concat_witness
    (Strings := DcStr) (fstar := distinctFstar)
    (u₁ := ofList [3, 5]) (v₁ := ofList [5])
    (u₂ := ofList [3, 7]) (v₂ := ofList [5])
    ?_ ?_ ?_
  · show ((distinctCount [3, 5] : ℕ) : ℝ) = ((distinctCount [3, 7] : ℕ) : ℝ)
    rw [e_u1, e_u2]
  · rfl
  · show ((distinctCount [3, 5, 5] : ℕ) : ℝ) ≠ ((distinctCount [3, 7, 5] : ℕ) : ℝ)
    rw [e_uv1, e_uv2]; norm_num

end DistinctCount

/-! ## Section 7 — Doob-Dynkin bridge to `OracleIndexedConditionalDensity`

The pseudometric `FactorsThroughSummary` predicate is the *abstract*
Doob-Dynkin condition; `InformationSufficiency.lean` carries its
*measure-theoretic* counterpart `OracleIndexedConditionalDensity`, which
says a conditional density `p : Strings → Obs → ℝ` factors through
`fstar` (i.e. `p(x, y) = pbar(fstar x, y)`).

The bridge is short: if `fstar` is invariant under `g` (which is exactly
`A1_global` in `MetricSpace Y`, where `dist = 0 ↔ ·=·`), then any
oracle-indexed conditional density is *also* invariant under `g`.  This
makes precise the conceptual story:

> `g` is a sufficient statistic for `fstar`, and `fstar` is a sufficient
> statistic for any oracle-indexed `p`; therefore `g` is a sufficient
> statistic for `p`.

Why we need `MetricSpace Y` rather than `PseudoMetricSpace Y`: in a
pseudometric `dist x y = 0` does not force `x = y`, so we cannot rewrite
`pbar (fstar x) _` to `pbar (fstar (g x)) _` using only `A1_global`.
Concrete oracle types in the project (`ℝ`, `ℝᵈ`, finite vectors of `ℝ`)
all carry `MetricSpace`, so this is not a binding restriction in
practice. -/

/-- **Doob-Dynkin transfer.**  If `fstar` is invariant under `g`
(`A1_global`), then any oracle-indexed conditional density `p` is also
invariant under `g`. -/
theorem oracleIndexedConditionalDensity_invariant_under_summary
    {Strings Obs : Type*} [Monoid Strings] [MeasurableSpace Obs]
    {Y : Type*} [MetricSpace Y]
    {g : Strings → Strings} {fstar : Strings → Y}
    (hA1 : A1_global g fstar)
    {p : Strings → Obs → ℝ}
    (hp : OracleIndexedConditionalDensity p fstar) :
    ∀ x y, p x y = p (g x) y := by
  obtain ⟨pbar, hpbar⟩ := hp
  intro x y
  rw [hpbar x y, hpbar (g x) y]
  -- A1_global g fstar : D fstar (g s) s = 0, i.e. dist (fstar (g s)) (fstar s) = 0.
  -- In MetricSpace Y, this collapses to fstar (g x) = fstar x.
  have heq : fstar (g x) = fstar x := eq_of_dist_eq_zero (hA1 x)
  rw [heq]

/-- Bundled: if `g` is a `MergeablePreference`-summarizer for `fstar`, then
any oracle-indexed conditional density is invariant under `g`. -/
theorem oracleIndexedConditionalDensity_invariant_under_mergeablePreference
    {Strings Obs : Type*} [Monoid Strings] [MeasurableSpace Obs]
    {Y : Type*} [MetricSpace Y]
    {g : Strings → Strings} {fstar : Strings → Y}
    (h : MergeablePreference g fstar)
    {p : Strings → Obs → ℝ}
    (hp : OracleIndexedConditionalDensity p fstar) :
    ∀ x y, p x y = p (g x) y :=
  oracleIndexedConditionalDensity_invariant_under_summary h.sufficiency hp

/-! ## Section 6 — Paper-numbered aliases & inhabitable example -/

/-- **Paper alias for the characterization slogan.**  Re-exports the
mergeable-preference structure under a paper-aligned name for citation
in the LaTeX appendix. -/
abbrev PaperPreferenceCharacterization := @MergeablePreference

/-- **Paper alias for the existence wrapper.** -/
abbrev PaperCTreePOExpressible := @CTreePOExpressible

/-- **Paper alias for the threshold-AND counterexample (DGP-1). -/
theorem paper_dgp1_threshold_and_not_expressible
    (k₁ k₂ : ℕ) (hk₁ : 1 ≤ k₁) (hk₂ : 1 ≤ k₂) :
    ¬ CTreePOExpressible (ThresholdAND.thrFstar k₁ k₂) :=
  ThresholdAND.not_ctreepoExpressible_threshold_and k₁ k₂ hk₁ hk₂

/-- **Paper alias for the cross-boundary bigram counterexample (DGP-2). -/
theorem paper_dgp2_cross_boundary_bigram_not_expressible :
    ¬ CTreePOExpressible CrossBoundaryBigram.bigramFstar :=
  CrossBoundaryBigram.not_ctreepoExpressible_cross_boundary_bigram

/-- **Paper alias for the rank-statistic counterexample (distinct count).**
The same witness pattern rules out median, IQR, mode, and any other
quantile/cardinality-based scalar oracle. -/
theorem paper_rank_statistic_distinct_count_not_expressible :
    ¬ CTreePOExpressible DistinctCount.distinctFstar :=
  DistinctCount.not_ctreepoExpressible_distinct_count

/-- **Paper alias for the completeness theorem.**  The scalar-oracle
characterization is a true biconditional: a preference is C-TreePO-
expressible iff it satisfies concat determinacy.  This is the central
"TFAE" of the scalar layer. -/
theorem paper_scalar_layer_completeness
    {Strings : Type*} [Monoid Strings] {fstar : Strings → ℝ} :
    CTreePOExpressible fstar ↔ ConcatDeterminacy fstar :=
  ctreepoExpressible_iff_concatDeterminacy

/-- **Paper alias for the global-merge reformulation.**  Equivalent
slogan: `fstar` is expressible iff `fstar(u·v)` factors through
`(fstar u, fstar v)` via a single global `M`. -/
theorem paper_scalar_layer_global_merge
    {Strings : Type*} [Monoid Strings] {fstar : Strings → ℝ} :
    CTreePOExpressible fstar ↔
    ∃ M : ℝ → ℝ → ℝ, ∀ u v : Strings, fstar (u * v) = M (fstar u) (fstar v) :=
  ctreepoExpressible_iff_exists_global_merge

/-! ### Inhabitable-witness example

To confirm the structure is *inhabitable* and the negative results are not
vacuous, exhibit a trivial mergeable preference: the constant oracle
`fstar := fun _ => (0 : ℝ)` together with the identity summarizer.
Constants are trivially additively separable with `M = (· + ·)` since
`0 = 0 + 0`. -/

example : MergeablePreference (Strings := Multiplicative ℕ)
    (id) (fun _ => (0 : ℝ)) := by
  refine ⟨?_, ?_, ?_⟩
  · intro _; show dist (0 : ℝ) 0 = 0; exact dist_self _
  · intro _ _; show dist (0 : ℝ) 0 = 0; exact dist_self _
  · refine ⟨(· + ·), ?_, ?_⟩
    · intro _ _; show dist (0 : ℝ) (0 + 0) = 0; simp
    · intro y₁ y₁' y₂ y₂' hd1 hd2
      have e1 : y₁ = y₁' := eq_of_dist_eq_zero hd1
      have e2 : y₂ = y₂' := eq_of_dist_eq_zero hd2
      rw [e1, e2]; exact dist_self _

example : CTreePOExpressible (fun _ : Multiplicative ℕ => (0 : ℝ)) := by
  refine ⟨id, ?_, ?_, ?_⟩
  · intro _; show dist (0 : ℝ) 0 = 0; exact dist_self _
  · intro _ _; show dist (0 : ℝ) 0 = 0; exact dist_self _
  · refine ⟨(· + ·), ?_, ?_⟩
    · intro _ _; show dist (0 : ℝ) (0 + 0) = 0; simp
    · intro y₁ y₁' y₂ y₂' hd1 hd2
      have e1 : y₁ = y₁' := eq_of_dist_eq_zero hd1
      have e2 : y₂ = y₂' := eq_of_dist_eq_zero hd2
      rw [e1, e2]; exact dist_self _

end FormalProofs.OPT

end
