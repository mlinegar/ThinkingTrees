import FormalProofs.OPT.CoreDefinitions
import FormalProofs.OPT.LocalLaws
import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreservationTheorems

/-!
# FormalProofs/OPT/MergeTriangle.lean

## The Merge Triangle and Compositional Preservation

**Paper Reference:** Section 3 (Consistency Conditions), Theorem 1, Corollaries 1-2,
Appendix C (fixed partition), and the paper's Lemma (Merge Triangle).

This module makes the paper's central compositionality law a first-class Lean object:

    **Merge Triangle (paper C3, link 2):  g(x·y) ~ g(g(x)·g(y))**

summarizing a concatenation is oracle-equivalent to merging the summaries and
summarizing once more. Everything in the preservation tier is built on top of it.

### Why this module exists (de-circularization)

The legacy law `L2` (`LocalLaws.lean`) asserts zero expected distortion for the
**full recursive reduction of each subtree**; instantiated at the root it already
*is* the conclusion of `one_pass`. This module replaces that packaging with
genuinely one-call local laws, stated **support-wise (almost surely)** so they
are immune to the `tsum`-summability convention, and proves root preservation by
honest structural induction (`reduce_support_oracle_equiv`). The bridge theorem
`L2_of_local` then *derives* the legacy `L2` from the local laws, so every legacy
theorem downstream of `L2` inherits a non-circular entry point
(`localLawsBundle_of_local`).

### Law dictionary (paper label ↔ this module ↔ legacy Lean)

| Paper | This module | Legacy | Reading |
|-------|-------------|--------|---------|
| C1 (leaf sufficiency) | `LeafSufficiency` | `L1` | `g(b) ~ b` for realized leaves |
| C2 (idempotence) | `RangeIdempotence` | `L3` | `g(s) ~ s` on `range(g)` |
| C3 link 1 (joint faithfulness) | `MergeSufficiency` (realized inputs), `SpanMergeSufficiency` (spans) | — | `g(u·v) ~ u·v` |
| C3 link 2 (**the triangle**) | `MergeTriangle` | — | `g(x·y) ~ g(g(x)·g(y))` |
| Assumption `ass:context` | `ContextCompatible` | — (previously unformalized) | `~` is a congruence for `·` |

The legacy aliases `C1 := L1`, `C2 := L3`, `C3 := L2` in `LocalLaws.lean` are kept
for compatibility; new work should use the names above.

### Main results

- `mergeTriangle_of_local` — the triangle is *derivable* from the audited local
  laws (C1 at the pieces + C3-link-1 + context compatibility).
- `merge_faithful_of_triangle` — what the triangle buys: together with joint
  faithfulness, the merged summary determines the oracle value of the raw span.
- `tree_triangle` — the n-ary generalization: the hierarchical reduction of any
  subtree is oracle-equivalent to a *single* summary call on its raw span.
- `reduce_support_oracle_equiv` / `one_pass_of_local` — Theorem 1 by real
  induction: local laws + context compatibility ⇒ zero root distortion.
- `L2_of_local`, `localLawsBundle_of_local` — bridges into the legacy stack.
- `multi_round_support_of_local` / `multi_round_of_local` — Theorem 2 from local
  laws, with **no boundedness hypothesis** (the support-wise form needs none).
- `schedule_invariance_of_local`, `fold_of_folds_of_local` — honest versions of
  Corollaries 1-2: the same-partition hypothesis is used, and the two-level fold
  structure is explicit (`graft`).
- `population_preservation` / `population_loss_transport` — the tower step over
  a document distribution and a deterministic partition rule (fixed-partition
  extension, Appendix C).
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Oracle equivalence

`OracleEquiv fstar x y` says the oracle cannot distinguish `x` from `y`:
the (pseudo)distance between their oracle values is zero. This is the `~`
of the paper's Section 3.
-/

/-- Oracle equivalence of strings: `x ~ y` iff `dist (f* x) (f* y) = 0`. -/
def OracleEquiv (fstar : Strings → Y) (x y : Strings) : Prop :=
  dist (fstar x) (fstar y) = 0

namespace OracleEquiv

variable {fstar : Strings → Y}

@[refl] lemma refl (fstar : Strings → Y) (x : Strings) : OracleEquiv fstar x x :=
  dist_self _

lemma symm {x y : Strings} (h : OracleEquiv fstar x y) : OracleEquiv fstar y x := by
  unfold OracleEquiv at h ⊢
  rw [dist_comm]
  exact h

lemma trans {x y z : Strings} (hxy : OracleEquiv fstar x y) (hyz : OracleEquiv fstar y z) :
    OracleEquiv fstar x z := by
  unfold OracleEquiv at hxy hyz ⊢
  have h₁ := dist_triangle (fstar x) (fstar y) (fstar z)
  have h₂ := dist_nonneg (x := fstar x) (y := fstar z)
  linarith

/-- Oracle equivalence is exactly zero distortion. -/
lemma iff_D_eq_zero {x y : Strings} : OracleEquiv fstar x y ↔ D fstar x y = 0 :=
  Iff.rfl

end OracleEquiv

/-- Oracle equivalence of two summary *distributions*: every realization of one
side is oracle-equivalent to every realization of the other. Both sides
therefore determine one and the same oracle value. -/
def PMFOracleEquiv (fstar : Strings → Y) (p q : PMF Strings) : Prop :=
  ∀ z ∈ p.support, ∀ w ∈ q.support, OracleEquiv fstar z w

lemma PMFOracleEquiv.symm {fstar : Strings → Y} {p q : PMF Strings}
    (h : PMFOracleEquiv fstar p q) : PMFOracleEquiv fstar q p :=
  fun w hw z hz => (h z hz w hw).symm

/-!
## Local laws, stated almost surely

Support-wise (a.s.) statements sidestep the `tsum` convention entirely: a law
holds iff the property holds at every realization the summarizer can actually
produce. Bridges to the legacy expectation-zero forms are proved below.
-/

/-- `g` is faithful at input `x`: every realization of `g x` is oracle-equivalent
to `x`. This is the pointwise sufficiency primitive; all three C-laws are
instances of it at different input classes. -/
def SufficientAt (g : Summarizer Strings) (fstar : Strings → Y) (x : Strings) : Prop :=
  ∀ z ∈ (g x).support, OracleEquiv fstar z x

/-- **Paper C1 (leaf sufficiency), a.s. form:** `g(b) ~ b` at every realized leaf. -/
def LeafSufficiency (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ b ∈ leaves T, SufficientAt g fstar b

/-- **Paper C2 (on-range idempotence), a.s. form:** `g(s) ~ s` for every `s` the
summarizer can output. -/
def RangeIdempotence (g : Summarizer Strings) (fstar : Strings → Y) : Prop :=
  ∀ z, InRange g z → SufficientAt g fstar z

/-- **Paper C3 link 1 on realized merge inputs, a.s. form:** at every internal
node, `g` is faithful on each concatenation of realized child summaries. This is
the audited one-call merge law: one `g` call per merge, no recursion in the
hypothesis. -/
def MergeSufficiency (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ p ∈ internal_nodes T,
    ∀ zL ∈ (reduce g p.1).support, ∀ zR ∈ (reduce g p.2).support,
      SufficientAt g fstar (zL * zR)

/-- **Paper C3 link 1 at raw spans, a.s. form:** `g(u·v) ~ u·v` where `u, v` are
the raw spans of the children of each internal node. -/
def SpanMergeSufficiency (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ p ∈ internal_nodes T, SufficientAt g fstar (S p.1 * S p.2)

/-- **Paper Assumption `ass:context` (context compatibility):** oracle
equivalence is a congruence for concatenation. Replacing either side of a
concatenation by an oracle-equivalent string preserves oracle equivalence of
the whole. Previously this assumption had no Lean counterpart. -/
def ContextCompatible (fstar : Strings → Y) : Prop :=
  ∀ ⦃u u' v v' : Strings⦄, OracleEquiv fstar u u' → OracleEquiv fstar v v' →
    OracleEquiv fstar (u * v) (u' * v')

/-!
## The Merge Triangle

The paper's central compositionality law, `g(x·y) ~ g(g(x)·g(y))`, as an
equivalence of the two summary distributions.
-/

/-- Merge-of-summaries: summarize `x` and `y` separately, concatenate the
summaries, and summarize once more. This is `g(g(x)·g(y))` read as a
distribution over outputs. -/
def mergeOfSummaries (g : Summarizer Strings) (x y : Strings) : PMF Strings :=
  (g x).bind fun zx => (g y).bind fun zy => g (zx * zy)

/-- **The Merge Triangle (paper C3, link 2): `g(x·y) ~ g(g(x)·g(y))`.**

Summarizing the concatenation directly and merging the two summaries produce
oracle-equivalent outputs, almost surely on both sides. -/
def MergeTriangle (g : Summarizer Strings) (fstar : Strings → Y) (x y : Strings) : Prop :=
  PMFOracleEquiv fstar (g (x * y)) (mergeOfSummaries g x y)

/-- Membership in the support of `mergeOfSummaries`: a realization of
`g(g(x)·g(y))` is exactly a `g`-output on some concatenation of realized
summaries of `x` and `y`. -/
lemma mem_support_mergeOfSummaries_iff (g : Summarizer Strings) (x y w : Strings) :
    w ∈ (mergeOfSummaries g x y).support ↔
      ∃ zx ∈ (g x).support, ∃ zy ∈ (g y).support, w ∈ (g (zx * zy)).support := by
  unfold mergeOfSummaries
  simp [PMF.mem_support_bind_iff]

/-- The merge-of-summaries distribution is definitionally the hierarchical
reduction of the two-leaf tree: the triangle is the two-leaf case of the
whole preservation tier. -/
lemma mergeOfSummaries_eq_reduce_two_leaf (g : Summarizer Strings) (x y : Strings) :
    mergeOfSummaries g x y = reduce g (BinTree.node (BinTree.leaf x) (BinTree.leaf y)) :=
  rfl

/-- **The triangle is derivable from the audited local laws.** If `g` is
faithful at `x`, at `y`, on the raw concatenation `x·y`, and on every realized
concatenation of summaries, and the oracle is context-compatible, then
`g(x·y) ~ g(g(x)·g(y))`. -/
theorem mergeTriangle_of_local (g : Summarizer Strings) (fstar : Strings → Y) (x y : Strings)
    (hctx : ContextCompatible fstar)
    (hx : SufficientAt g fstar x) (hy : SufficientAt g fstar y)
    (hxy : SufficientAt g fstar (x * y))
    (hmerge : ∀ zx ∈ (g x).support, ∀ zy ∈ (g y).support, SufficientAt g fstar (zx * zy)) :
    MergeTriangle g fstar x y := by
  intro w hw w' hw'
  have hwxy : OracleEquiv fstar w (x * y) := hxy w hw
  rw [mem_support_mergeOfSummaries_iff] at hw'
  obtain ⟨zx, hzx, zy, hzy, hw'⟩ := hw'
  have h1 : OracleEquiv fstar w' (zx * zy) := hmerge zx hzx zy hzy w' hw'
  have h2 : OracleEquiv fstar (zx * zy) (x * y) := hctx (hx zx hzx) (hy zy hzy)
  exact hwxy.trans ((h1.trans h2).symm)

/-- **What the triangle buys:** together with joint faithfulness
(`g(x·y) ~ x·y`), the triangle transports the oracle value of the raw span to
the merged summary: every realization of `g(g(x)·g(y))` is oracle-equivalent to
`x·y`. The pivot uses the fact that `g (x·y)` has nonempty support. -/
theorem merge_faithful_of_triangle (g : Summarizer Strings) (fstar : Strings → Y) (x y : Strings)
    (htri : MergeTriangle g fstar x y) (hxy : SufficientAt g fstar (x * y)) :
    ∀ w' ∈ (mergeOfSummaries g x y).support, OracleEquiv fstar w' (x * y) := by
  intro w' hw'
  obtain ⟨w, hw⟩ := (g (x * y)).support_nonempty
  exact ((htri w hw w' hw').symm).trans (hxy w hw)

/-!
## Law restriction lemmas
-/

lemma leafSufficiency_left {g : Summarizer Strings} {L R : BinTree Strings}
    {fstar : Strings → Y} (h : LeafSufficiency g (BinTree.node L R) fstar) :
    LeafSufficiency g L fstar :=
  fun b hb => h b (List.mem_append_left _ hb)

lemma leafSufficiency_right {g : Summarizer Strings} {L R : BinTree Strings}
    {fstar : Strings → Y} (h : LeafSufficiency g (BinTree.node L R) fstar) :
    LeafSufficiency g R fstar :=
  fun b hb => h b (List.mem_append_right _ hb)

lemma mergeSufficiency_left {g : Summarizer Strings} {L R : BinTree Strings}
    {fstar : Strings → Y} (h : MergeSufficiency g (BinTree.node L R) fstar) :
    MergeSufficiency g L fstar :=
  fun p hp => h p (List.mem_cons_of_mem _ (List.mem_append_left _ hp))

lemma mergeSufficiency_right {g : Summarizer Strings} {L R : BinTree Strings}
    {fstar : Strings → Y} (h : MergeSufficiency g (BinTree.node L R) fstar) :
    MergeSufficiency g R fstar :=
  fun p hp => h p (List.mem_cons_of_mem _ (List.mem_append_right _ hp))

/-- Leaves of the subtree at an internal node are leaves of the whole tree. -/
lemma leaves_mem_of_internal {α : Type*} :
    ∀ (T : BinTree α), ∀ p ∈ internal_nodes T,
      ∀ b ∈ leaves (BinTree.node p.1 p.2), b ∈ leaves T
  | BinTree.leaf _ => by
      intro p hp
      simp [internal_nodes] at hp
  | BinTree.node L R => by
      intro p hp b hb
      rcases List.mem_cons.mp hp with hpe | hp'
      · subst hpe
        exact hb
      · rcases List.mem_append.mp hp' with hL | hR
        · exact List.mem_append_left _ (leaves_mem_of_internal L p hL b hb)
        · exact List.mem_append_right _ (leaves_mem_of_internal R p hR b hb)

/-- Internal nodes of the subtree at an internal node are internal nodes of the
whole tree. -/
lemma internal_nodes_mem_of_internal {α : Type*} :
    ∀ (T : BinTree α), ∀ p ∈ internal_nodes T,
      ∀ q ∈ internal_nodes (BinTree.node p.1 p.2), q ∈ internal_nodes T
  | BinTree.leaf _ => by
      intro p hp
      simp [internal_nodes] at hp
  | BinTree.node L R => by
      intro p hp q hq
      rcases List.mem_cons.mp hp with hpe | hp'
      · subst hpe
        exact hq
      · rcases List.mem_append.mp hp' with hL | hR
        · exact List.mem_cons_of_mem _
            (List.mem_append_left _ (internal_nodes_mem_of_internal L p hL q hq))
        · exact List.mem_cons_of_mem _
            (List.mem_append_right _ (internal_nodes_mem_of_internal R p hR q hq))

/-- Restriction of the per-tree laws to the subtree rooted at an internal node. -/
lemma internal_node_subtree_laws {g : Summarizer Strings} {fstar : Strings → Y}
    (T : BinTree Strings) (p : BinTree Strings × BinTree Strings)
    (hp : p ∈ internal_nodes T)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar) :
    LeafSufficiency g (BinTree.node p.1 p.2) fstar ∧
      MergeSufficiency g (BinTree.node p.1 p.2) fstar :=
  ⟨fun b hb => h1 b (leaves_mem_of_internal T p hp b hb),
   fun q hq => h2 q (internal_nodes_mem_of_internal T p hp q hq)⟩

/-!
## Compositional preservation (Theorem 1, de-circularized)
-/

/-- **Compositional preservation, support form (Theorem 1 core).** Under leaf
sufficiency (C1), the one-call merge law (C3 link 1 on realized inputs), and
context compatibility (`ass:context`), every realization of the hierarchical
reduction is oracle-equivalent to the raw span. Proved by structural induction
on the tree; no hypothesis mentions the behavior of whole-subtree reductions. -/
theorem reduce_support_oracle_equiv (g : Summarizer Strings) (fstar : Strings → Y)
    (hctx : ContextCompatible fstar) :
    ∀ T : BinTree Strings, LeafSufficiency g T fstar → MergeSufficiency g T fstar →
      ∀ z ∈ (reduce g T).support, OracleEquiv fstar z (S T)
  | BinTree.leaf b => by
      intro h1 _h2 z hz
      exact h1 b (by simp [leaves]) z hz
  | BinTree.node L R => by
      intro h1 h2 z hz
      have hz' : ∃ zL ∈ (reduce g L).support, ∃ zR ∈ (reduce g R).support,
          z ∈ (g (zL * zR)).support := by
        have hred : reduce g (BinTree.node L R)
            = (reduce g L).bind (fun sL => (reduce g R).bind (fun sR => g (sL * sR))) := rfl
        rw [hred] at hz
        simpa [PMF.mem_support_bind_iff] using hz
      obtain ⟨zL, hzL, zR, hzR, hz⟩ := hz'
      have hL : OracleEquiv fstar zL (S L) :=
        reduce_support_oracle_equiv g fstar hctx L
          (leafSufficiency_left h1) (mergeSufficiency_left h2) zL hzL
      have hR : OracleEquiv fstar zR (S R) :=
        reduce_support_oracle_equiv g fstar hctx R
          (leafSufficiency_right h1) (mergeSufficiency_right h2) zR hzR
      have hmerge : OracleEquiv fstar z (zL * zR) :=
        h2 (L, R) (List.mem_cons_self) zL hzL zR hzR z hz
      have hspan : OracleEquiv fstar (zL * zR) (S L * S R) := hctx hL hR
      exact hmerge.trans hspan

/-- **The tree triangle (n-ary generalization of the Merge Triangle).** At every
internal node, the hierarchical reduction of the subtree is oracle-equivalent to
a *single* summary call on the raw span:
`g(S T_L · S T_R) ~ reduce g (node T_L T_R)`. The two-leaf instance is exactly
`MergeTriangle` (via `mergeOfSummaries_eq_reduce_two_leaf`). -/
theorem tree_triangle (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar)
    (hspan : SpanMergeSufficiency g T fstar) :
    ∀ p ∈ internal_nodes T,
      PMFOracleEquiv fstar (g (S p.1 * S p.2)) (reduce g (BinTree.node p.1 p.2)) := by
  intro p hp w hw z hz
  have hsub := internal_node_subtree_laws (g := g) (fstar := fstar) T p hp h1 h2
  have hzspan : OracleEquiv fstar z (S (BinTree.node p.1 p.2)) :=
    reduce_support_oracle_equiv g fstar hctx (BinTree.node p.1 p.2) hsub.1 hsub.2 z hz
  have hSnode : S (BinTree.node p.1 p.2) = S p.1 * S p.2 := rfl
  rw [hSnode] at hzspan
  exact (hspan p hp w hw).trans hzspan.symm

/-- The two-leaf case of `tree_triangle` *is* the Merge Triangle. -/
theorem mergeTriangle_of_tree_triangle (g : Summarizer Strings) (fstar : Strings → Y)
    (x y : Strings) (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g (BinTree.node (BinTree.leaf x) (BinTree.leaf y)) fstar)
    (h2 : MergeSufficiency g (BinTree.node (BinTree.leaf x) (BinTree.leaf y)) fstar)
    (hspan : SpanMergeSufficiency g (BinTree.node (BinTree.leaf x) (BinTree.leaf y)) fstar) :
    MergeTriangle g fstar x y := by
  have h := tree_triangle g fstar (BinTree.node (BinTree.leaf x) (BinTree.leaf y)) hctx h1 h2
    hspan ((BinTree.leaf x), (BinTree.leaf y)) (List.mem_cons_self)
  unfold MergeTriangle
  rw [mergeOfSummaries_eq_reduce_two_leaf]
  exact h

/-!
## Bridges to the legacy expectation-zero laws

Support-wise laws imply the legacy `tsum`-form laws unconditionally (a term
that vanishes on the support makes every summand zero). Together with
`ContextCompatible`, they produce a full legacy `LocalLawsBundle`, so every
downstream theorem (`one_pass`, `multi_round_proper`, the preference
equivalences, the gap bounds) now has a genuinely local, non-circular entry
point.
-/

/-- If a function vanishes on the support of a PMF, its expectation is zero
(no summability needed — the series is identically 0). -/
lemma Exp_eq_zero_of_zero_on_support {α : Type*} (p : PMF α) (f : α → ℝ)
    (h : ∀ z ∈ p.support, f z = 0) : Exp p f = 0 := by
  unfold Exp
  have hz : ∀ z, (p z).toReal * f z = 0 := by
    intro z
    by_cases hzs : z ∈ p.support
    · rw [h z hzs, mul_zero]
    · rw [PMF.mem_support_iff, not_not] at hzs
      rw [hzs]
      simp
  rw [tsum_congr hz]
  exact tsum_zero

/-- Leaf sufficiency (a.s.) implies the legacy `L1`. -/
theorem L1_of_leafSufficiency (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) (h : LeafSufficiency g T fstar) : L1 g T fstar := by
  intro b hb
  exact Exp_eq_zero_of_zero_on_support (g b) _ (fun z hz => h b hb z hz)

/-- Range idempotence (a.s.) implies the legacy `L3`. -/
theorem L3_of_rangeIdempotence (g : Summarizer Strings) (fstar : Strings → Y)
    (h : RangeIdempotence g fstar) : L3 g fstar := by
  intro Z hZ
  exact Exp_eq_zero_of_zero_on_support (g Z) _ (fun z hz => h Z hZ z hz)

/-- **De-circularization bridge:** the genuinely local laws imply the legacy
subtree-level `L2`. Every legacy theorem taking `L2` (in particular `one_pass`,
`multi_round_proper`, and everything downstream) therefore now has a
non-circular entry point. -/
theorem L2_of_local (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar) :
    L2 g T fstar := by
  intro p hp
  have hsub := internal_node_subtree_laws (g := g) (fstar := fstar) T p hp h1 h2
  have hsupp := reduce_support_oracle_equiv g fstar hctx (BinTree.node p.1 p.2) hsub.1 hsub.2
  show Egu g (BinTree.node p.1 p.2) (fun z => D fstar z (S (BinTree.node p.1 p.2))) = 0
  unfold Egu
  have hz : ∀ z, (reduce g (BinTree.node p.1 p.2) z).toReal
      * D fstar z (S (BinTree.node p.1 p.2)) = 0 := by
    intro z
    by_cases hzs : z ∈ (reduce g (BinTree.node p.1 p.2)).support
    · rw [show D fstar z (S (BinTree.node p.1 p.2)) = 0 from hsupp z hzs, mul_zero]
    · rw [PMF.mem_support_iff, not_not] at hzs
      rw [hzs]
      simp
  rw [tsum_congr hz]
  exact tsum_zero

/-- **Master bridge:** local laws + range idempotence + context compatibility
produce a full legacy `LocalLawsBundle`. -/
theorem localLawsBundle_of_local (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar)
    (h3 : RangeIdempotence g fstar) : LocalLawsBundle g T fstar :=
  ⟨L1_of_leafSufficiency g T fstar h1,
   L2_of_local g T fstar hctx h1 h2,
   L3_of_rangeIdempotence g fstar h3⟩

/-- **One-pass preservation from local laws (Theorem 1, non-circular).**
Zero expected root distortion, derived by structural induction from one-call
local laws and context compatibility. Unlike the legacy `one_pass`, no
hypothesis refers to the behavior of whole-subtree reductions. -/
theorem one_pass_of_local (g : Summarizer Strings) (T : BinTree Strings) (x : Strings)
    (fstar : Strings → Y) (hp : S T = x) (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar) :
    Egu g (root T) (fun z => D fstar z x) = 0 := by
  subst hp
  have hsupp := reduce_support_oracle_equiv g fstar hctx T h1 h2
  show Egu g T (fun z => D fstar z (S T)) = 0
  unfold Egu
  have hz : ∀ z, (reduce g T z).toReal * D fstar z (S T) = 0 := by
    intro z
    by_cases hzs : z ∈ (reduce g T).support
    · rw [show D fstar z (S T) = 0 from hsupp z hzs, mul_zero]
    · rw [PMF.mem_support_iff, not_not] at hzs
      rw [hzs]
      simp
  rw [tsum_congr hz]
  exact tsum_zero

/-!
## Multi-round preservation from local laws (Theorem 2, no boundedness)
-/

/-- **Multi-round preservation, support form (Theorem 2 core).** Under the local
laws plus range idempotence (C2), every realization of `Z^(R)` is
oracle-equivalent to the document. Proved with **no boundedness hypothesis**:
the support-wise form needs none. -/
theorem multi_round_support_of_local (g : Summarizer Strings) (T : BinTree Strings)
    (x : Strings) (fstar : Strings → Y) (hp : S T = x) (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar)
    (h3 : RangeIdempotence g fstar) :
    ∀ R : ℕ, R ≥ 1 → ∀ z ∈ (ZR g x R T).support, OracleEquiv fstar z x := by
  intro R
  induction R with
  | zero => omega
  | succ n ih =>
      intro _hR
      by_cases hn : n = 0
      · subst hn
        intro z hz
        have hze := reduce_support_oracle_equiv g fstar hctx T h1 h2 z hz
        rwa [hp] at hze
      · have hn1 : n ≥ 1 := Nat.one_le_iff_ne_zero.mpr hn
        have hbind : ZR g x (n + 1) T = (ZR g x n T).bind g :=
          ZR_succ_eq_bind g x n T hn1
        intro z hz
        rw [hbind, PMF.mem_support_bind_iff] at hz
        obtain ⟨w, hw, hz⟩ := hz
        have hwx : OracleEquiv fstar w x := ih hn1 w hw
        have hwr : InRange g w := ZR_support_in_range g x n T hn1 w hw
        have hzw : OracleEquiv fstar z w := h3 w hwr z hz
        exact hzw.trans hwx

/-- **Multi-round preservation from local laws (Theorem 2, expectation form).**
Zero expected distortion after any number of rounds, with no boundedness
hypothesis — the paper's Recompression Stack conclusion from genuinely local
premises. -/
theorem multi_round_of_local (g : Summarizer Strings) (T : BinTree Strings)
    (x : Strings) (fstar : Strings → Y) (R : ℕ) (hR : R ≥ 1) (hp : S T = x)
    (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar)
    (h3 : RangeIdempotence g fstar) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
  Exp_eq_zero_of_zero_on_support _ _
    (fun z hz => multi_round_support_of_local g T x fstar hp hctx h1 h2 h3 R hR z hz)

/-!
## Honest schedule invariance and fold-of-folds (Corollaries 1-2)
-/

/-- The raw span is the ordered product of the leaves. -/
lemma S_eq_leaves_prod : ∀ T : BinTree Strings, S T = (leaves T).prod
  | BinTree.leaf b => by simp [S, leaves]
  | BinTree.node L R => by
      rw [show S (BinTree.node L R) = S L * S R from rfl,
        show leaves (BinTree.node L R) = leaves L ++ leaves R from rfl,
        List.prod_append, S_eq_leaves_prod L, S_eq_leaves_prod R]

/-- **Schedule invariance from local laws (Corollary 1, honest form).** Two
schedules over the *same ordered leaves* (the hypothesis is used: it forces the
spans to coincide) produce oracle-equivalent reductions: every realization of
either schedule is oracle-equivalent to every realization of the other. -/
theorem schedule_invariance_of_local (g : Summarizer Strings) (fstar : Strings → Y)
    (T T' : BinTree Strings) (h_l : leaves T = leaves T')
    (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g T fstar) (h2 : MergeSufficiency g T fstar)
    (h1' : LeafSufficiency g T' fstar) (h2' : MergeSufficiency g T' fstar) :
    PMFOracleEquiv fstar (reduce g T) (reduce g T') := by
  have hspan : S T = S T' := by
    rw [S_eq_leaves_prod, S_eq_leaves_prod, h_l]
  intro z hz w hw
  have hzT := reduce_support_oracle_equiv g fstar hctx T h1 h2 z hz
  have hwT := reduce_support_oracle_equiv g fstar hctx T' h1' h2' w hw
  rw [← hspan] at hwT
  exact hzT.trans hwT.symm

/-- Flatten a two-level plan — a tree whose leaves are themselves trees (folds) —
into the composite tree it denotes. The outer shape is the over-fold
parenthesization; each leaf is a within-fold parenthesization. -/
def graft {α : Type*} : BinTree (BinTree α) → BinTree α
  | BinTree.leaf t => t
  | BinTree.node U V => BinTree.node (graft U) (graft V)

/-- The leaves of a grafted plan are the concatenated leaves of its folds. -/
lemma graft_leaves {α : Type*} : ∀ U : BinTree (BinTree α),
    leaves (graft U) = (leaves U).flatMap leaves
  | BinTree.leaf t => by simp [graft, leaves]
  | BinTree.node U V => by
      rw [show graft (BinTree.node U V) = BinTree.node (graft U) (graft V) from rfl,
        show leaves (BinTree.node (graft U) (graft V))
          = leaves (graft U) ++ leaves (graft V) from rfl,
        graft_leaves U, graft_leaves V,
        show leaves (BinTree.node U V) = leaves U ++ leaves V from rfl,
        List.flatMap_append]

/-- **Fold-of-folds invariance from local laws (Corollary 2, honest form).**
Two two-level plans (fold, then merge fold summaries) over folds carrying the
same ordered leaves are oracle-equivalent, regardless of the within-fold or
over-fold parenthesizations. The fold structure is explicit: `U, U'` are plans
whose leaves are the folds. -/
theorem fold_of_folds_of_local (g : Summarizer Strings) (fstar : Strings → Y)
    (U U' : BinTree (BinTree Strings))
    (h_l : (leaves U).flatMap leaves = (leaves U').flatMap leaves)
    (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g (graft U) fstar) (h2 : MergeSufficiency g (graft U) fstar)
    (h1' : LeafSufficiency g (graft U') fstar) (h2' : MergeSufficiency g (graft U') fstar) :
    PMFOracleEquiv fstar (reduce g (graft U)) (reduce g (graft U')) := by
  apply schedule_invariance_of_local g fstar (graft U) (graft U') _ hctx h1 h2 h1' h2'
  rw [graft_leaves, graft_leaves, h_l]

/-- Fold-of-folds preservation: a two-level plan with the right span preserves
the oracle value (specializes `one_pass_of_local` to the grafted tree). -/
theorem fold_of_folds_preserves_of_local (g : Summarizer Strings) (fstar : Strings → Y)
    (U : BinTree (BinTree Strings)) (x : Strings) (hp : S (graft U) = x)
    (hctx : ContextCompatible fstar)
    (h1 : LeafSufficiency g (graft U) fstar) (h2 : MergeSufficiency g (graft U) fstar) :
    Egu g (root (graft U)) (fun z => D fstar z x) = 0 :=
  one_pass_of_local g (graft U) x fstar hp hctx h1 h2

/-!
## Fixed-partition extension (Appendix C): the tower step

The paper's Theorem quantifies over a deterministic partition rule `Π` mapping
each document to a tree, and takes the outer expectation over the document
distribution. These theorems supply exactly that step, previously absent.
-/

/-- **Population preservation, support form.** For a document distribution
`μX` and a deterministic partition rule `Tpi` whose trees realize their
documents, the local laws at every document give oracle equivalence of every
realization at every document in the population. -/
theorem population_preservation (g : Summarizer Strings) (fstar : Strings → Y)
    (μX : PMF Strings) (Tpi : Strings → BinTree Strings) (R : ℕ) (hR : R ≥ 1)
    (hp : ∀ x ∈ μX.support, S (Tpi x) = x)
    (hctx : ContextCompatible fstar)
    (h1 : ∀ x ∈ μX.support, LeafSufficiency g (Tpi x) fstar)
    (h2 : ∀ x ∈ μX.support, MergeSufficiency g (Tpi x) fstar)
    (h3 : RangeIdempotence g fstar) :
    ∀ x ∈ μX.support, ∀ z ∈ (ZR g x R (Tpi x)).support, OracleEquiv fstar z x :=
  fun x hx =>
    multi_round_support_of_local g (Tpi x) x fstar (hp x hx) hctx (h1 x hx) (h2 x hx) h3 R hR

/-- **Fixed-partition extension (Appendix C), expectation form.** The population
expected distortion — outer expectation over documents, inner expectation over
the summarizer — is exactly zero. This is the tower step the paper's Theorem
adds on top of the per-tree kernel. -/
theorem fixed_partition_population (g : Summarizer Strings) (fstar : Strings → Y)
    (μX : PMF Strings) (Tpi : Strings → BinTree Strings) (R : ℕ) (hR : R ≥ 1)
    (hp : ∀ x ∈ μX.support, S (Tpi x) = x)
    (hctx : ContextCompatible fstar)
    (h1 : ∀ x ∈ μX.support, LeafSufficiency g (Tpi x) fstar)
    (h2 : ∀ x ∈ μX.support, MergeSufficiency g (Tpi x) fstar)
    (h3 : RangeIdempotence g fstar) :
    ∑' x, (μX x).toReal * Exp (ZR g x R (Tpi x)) (fun z => D fstar z x) = 0 := by
  have hterm : ∀ x, (μX x).toReal * Exp (ZR g x R (Tpi x)) (fun z => D fstar z x) = 0 := by
    intro x
    by_cases hx : x ∈ μX.support
    · have h0 : Exp (ZR g x R (Tpi x)) (fun z => D fstar z x) = 0 :=
        Exp_eq_zero_of_zero_on_support _ _
          (fun z hz => population_preservation g fstar μX Tpi R hR hp hctx h1 h2 h3 x hx z hz)
      rw [h0, mul_zero]
    · rw [PMF.mem_support_iff, not_not] at hx
      rw [hx]
      simp
  rw [tsum_congr hterm]
  exact tsum_zero

/-- **Population loss transport (the tower step for preference equivalence).**
Any loss that respects oracle equivalence has the same expectation on the
summary as on the document, at every document in the population. Combined over
`μX`, the population summary objective equals the population document
objective — the population-level half of Theorem `thm:pref-equiv`. -/
theorem population_loss_transport (g : Summarizer Strings) (fstar : Strings → Y)
    (μX : PMF Strings) (Tpi : Strings → BinTree Strings) (R : ℕ) (hR : R ≥ 1)
    (hp : ∀ x ∈ μX.support, S (Tpi x) = x)
    (hctx : ContextCompatible fstar)
    (h1 : ∀ x ∈ μX.support, LeafSufficiency g (Tpi x) fstar)
    (h2 : ∀ x ∈ μX.support, MergeSufficiency g (Tpi x) fstar)
    (h3 : RangeIdempotence g fstar)
    (Loss : Strings → ℝ)
    (hLoss : ∀ z x, OracleEquiv fstar z x → Loss z = Loss x) :
    ∀ x ∈ μX.support, Exp (ZR g x R (Tpi x)) Loss = Loss x := by
  intro x hx
  have hconst : ∀ z ∈ (ZR g x R (Tpi x)).support, Loss z = Loss x :=
    fun z hz =>
      hLoss z x (population_preservation g fstar μX Tpi R hR hp hctx h1 h2 h3 x hx z hz)
  unfold Exp
  have hterm : ∀ z, ((ZR g x R (Tpi x)) z).toReal * Loss z
      = ((ZR g x R (Tpi x)) z).toReal * Loss x := by
    intro z
    by_cases hz : z ∈ (ZR g x R (Tpi x)).support
    · rw [hconst z hz]
    · rw [PMF.mem_support_iff, not_not] at hz
      rw [hz]
      simp
  rw [tsum_congr hterm, tsum_mul_right, PMF.toReal_tsum_coe, one_mul]

/-!
## Error-budget union bound (paper Equation `eq:error_budget`)
-/

/-- **Error-budget union bound.** For any finite family of violation events,
the probability of at least one violation is bounded by the sum of the
per-event probabilities. Instantiate the index set with
`leaves ⊕ merges ⊕ rounds` to obtain the paper's display. -/
theorem paper_error_budget_union_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) {ι : Type*} (units : Finset ι) (violation : ι → Set Ω) :
    μ (⋃ i ∈ units, violation i) ≤ ∑ i ∈ units, μ (violation i) :=
  MeasureTheory.measure_biUnion_finset_le units violation

end
