import FormalProofs.OPT.BigramSketch

/-!
# FormalProofs/OPT/TopicBigramOracle.lean

## Mergeable oracle example: topic unigrams + topic bigrams

This file specializes the `BigramSketch` mergeability pattern to the oracle used in the
Segment‑LDA OPS weight‑recovery simulation:

`f⋆(span) = ⟨θ, topicCounts(span)⟩ + λ · ⟨W, topicBigrams(span)⟩`.

Key point:
- Unigram counts are leaf-additive.
- Bigram counts require **one token of boundary metadata** (`first`/`last`) to be mergeable.

This exactly matches the “minimal exact sketch” story used in the simulation code.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped BigOperators

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section TopicBigramOracle

variable {α : Type*} [DecidableEq α]

/-- A mergeable sketch carrying topic unigrams and (mergeable) topic bigrams. -/
structure UniBigramSketch (α : Type*) where
  unigrams : Multiset α
  bigrams : BigramSketch α

@[ext] lemma UniBigramSketch.ext {s t : UniBigramSketch α}
    (h_uni : s.unigrams = t.unigrams) (h_bi : s.bigrams = t.bigrams) : s = t := by
  cases s
  cases t
  cases h_uni
  cases h_bi
  rfl

/-- Construct the sketch for a list of topics. -/
def uniBigramSketch (xs : List α) : UniBigramSketch α :=
  ⟨(xs : Multiset α), bigramSketch xs⟩

/-- Merge operation for sketches (associative via `mergeSketch`). -/
def mergeUniBigramSketch (s t : UniBigramSketch α) : UniBigramSketch α :=
  ⟨s.unigrams + t.unigrams, mergeSketch s.bigrams t.bigrams⟩

theorem uniBigramSketch_append (xs ys : List α) :
    uniBigramSketch (xs ++ ys) = mergeUniBigramSketch (uniBigramSketch xs) (uniBigramSketch ys) := by
  apply UniBigramSketch.ext
  · -- unigrams: multiset counts add under append
    change ((xs ++ ys : List α) : Multiset α) = (xs : Multiset α) + (ys : Multiset α)
    simpa using (Multiset.coe_add xs ys)
  · -- bigrams: use the `BigramSketch` merge lemma
    simp [uniBigramSketch, mergeUniBigramSketch, bigramSketch_append]

/-- Linear oracle score computed from a `UniBigramSketch`. -/
def oracleFromSketch (θ : α → ℝ) (W : (α × α) → ℝ) (lam : ℝ) (s : UniBigramSketch α) : ℝ :=
  (s.unigrams.map θ).sum + lam * (s.bigrams.pairs.map W).sum

/-- The same oracle score computed directly from a topic sequence. -/
def oracleOnList (θ : α → ℝ) (W : (α × α) → ℝ) (lam : ℝ) (xs : List α) : ℝ :=
  oracleFromSketch θ W lam (uniBigramSketch xs)

theorem oracleOnList_append (θ : α → ℝ) (W : (α × α) → ℝ) (lam : ℝ) (xs ys : List α) :
    oracleOnList θ W lam (xs ++ ys)
      = oracleFromSketch θ W lam (mergeUniBigramSketch (uniBigramSketch xs) (uniBigramSketch ys)) := by
  simp [oracleOnList, uniBigramSketch_append (xs := xs) (ys := ys)]

end TopicBigramOracle

end FormalProofs.OPT
