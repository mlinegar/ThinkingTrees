import FormalProofs.OPT.SketchSummaryOperators

/-!
# FormalProofs/OPT/BagOfWordsLDARecovery.lean

## Exact bag-of-words LDA recovery via mergeable count sketches

This file formalizes the clean base case for the tree-based LDA story.

In ordinary bag-of-words LDA, the document-level latent variable is a single mixture `π_d`,
so the sufficient statistic for the document likelihood is the document histogram. That gives
an exact mergeable sketch:

- leaf sketch: word histogram on the leaf,
- merge: histogram addition,
- root sketch: the full-document histogram, independent of tree shape.

Consequences:

1. any downstream utility depending only on the histogram is preserved exactly by the tree;
2. the ordinary bag-of-words LDA document likelihood is preserved exactly as a corollary.

This is the right formal base case before introducing learned compression or local-mixture
extensions where leaves become statistically informative.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section BagOfWords

variable {α : Type*}

/-- Bag-of-words feature map: forget token order and keep only the multiset of words. -/
def bagOfWords (xs : List α) : Multiset α :=
  (xs : Multiset α)

@[simp] theorem bagOfWords_nil : bagOfWords ([] : List α) = (0 : Multiset α) := rfl

@[simp] theorem bagOfWords_cons (x : α) (xs : List α) :
    bagOfWords (x :: xs) = ({x} : Multiset α) + bagOfWords xs := rfl

@[simp] theorem bagOfWords_append (xs ys : List α) :
    bagOfWords (xs ++ ys) = bagOfWords xs + bagOfWords ys := by
  change ((xs ++ ys : List α) : Multiset α) = (xs : Multiset α) + (ys : Multiset α)
  exact Multiset.coe_add xs ys

@[simp] theorem bagOfWords_toList (m : Multiset α) :
    bagOfWords m.toList = m := by
  change ((m.toList : List α) : Multiset α) = m
  exact Multiset.coe_toList m

/-- Exact count sketch for bag-of-words documents. -/
def countSketchOperator : SketchOperator (List α) (Multiset α) where
  encode := bagOfWords
  merge := (· + ·)
  decode := Multiset.toList

/-- Tree reduction of the exact count sketch returns the full-document histogram. -/
theorem sketchReduce_countSketch_eq_bagOfWords (T : BinTree (List α)) :
    sketchReduce (countSketchOperator (α := α)) T = bagOfWords (S T) := by
  induction T with
  | leaf b =>
      simp [sketchReduce, countSketchOperator, bagOfWords, S]
  | node TL TR ihL ihR =>
      have ihL' :
          sketchReduce { encode := bagOfWords, merge := fun x1 x2 => x1 + x2, decode := Multiset.toList } TL
            = bagOfWords (S TL) := by
        simpa [countSketchOperator] using ihL
      have ihR' :
          sketchReduce { encode := bagOfWords, merge := fun x1 x2 => x1 + x2, decode := Multiset.toList } TR
            = bagOfWords (S TR) := by
        simpa [countSketchOperator] using ihR
      rw [show S (BinTree.node TL TR) = S TL ++ S TR by rfl]
      simp [sketchReduce, countSketchOperator, bagOfWords_append]
      rw [ihL', ihR']

/-- Decoding the root count sketch preserves the full bag-of-words representation exactly. -/
theorem bagOfWords_sketchSummary_countSketch (T : BinTree (List α)) :
    bagOfWords (sketchSummary (countSketchOperator (α := α)) T) = bagOfWords (S T) := by
  rw [show sketchSummary (countSketchOperator (α := α)) T =
      (sketchReduce (countSketchOperator (α := α)) T).toList by rfl]
  rw [bagOfWords_toList, sketchReduce_countSketch_eq_bagOfWords]

/-- Any downstream utility that depends only on the histogram is exactly preserved by the tree. -/
theorem histogramUtility_exact_on_tree {β : Type*} (u : Multiset α → β) (T : BinTree (List α)) :
    u (bagOfWords (sketchSummary (countSketchOperator (α := α)) T)) = u (bagOfWords (S T)) := by
  exact congrArg u (bagOfWords_sketchSummary_countSketch (α := α) (T := T))

/-- A particularly useful histogram utility: a fixed linear word-weight functional. -/
def weightedWordUtility (w : α → ℝ) (m : Multiset α) : ℝ :=
  ((m.toList.map w).sum)

/-- Fixed linear word-weight utilities are exactly preserved by the count-sketch tree. -/
theorem weightedWordUtility_exact_on_tree (w : α → ℝ) (T : BinTree (List α)) :
    weightedWordUtility w (bagOfWords (sketchSummary (countSketchOperator (α := α)) T))
      = weightedWordUtility w (bagOfWords (S T)) := by
  exact histogramUtility_exact_on_tree (α := α) (u := weightedWordUtility w) (T := T)

end BagOfWords

section LDADocumentLikelihood

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {κ : Type*} [Fintype κ]

/-- Word probability under a fixed topic mixture `π` and topic-word table `φ`. -/
def ldaTokenProb (π : κ → ℝ) (φ : κ → α → ℝ) (w : α) : ℝ :=
  ∑ k : κ, π k * φ k w

/-- Bag-of-words LDA document likelihood as a function of a histogram. -/
def ldaHistogramLikelihood (π : κ → ℝ) (φ : κ → α → ℝ) (m : Multiset α) : ℝ :=
  ∏ w : α, (ldaTokenProb π φ w) ^ m.count w

/-- Document likelihood viewed on token lists through their bag-of-words feature map. -/
def ldaDocumentLikelihood (π : κ → ℝ) (φ : κ → α → ℝ) (xs : List α) : ℝ :=
  ldaHistogramLikelihood π φ (bagOfWords xs)

/-- Histogram addition is the exact merge rule for the ordinary LDA document likelihood. -/
theorem ldaHistogramLikelihood_add (π : κ → ℝ) (φ : κ → α → ℝ)
    (m₁ m₂ : Multiset α) :
    ldaHistogramLikelihood π φ (m₁ + m₂)
      = ldaHistogramLikelihood π φ m₁ * ldaHistogramLikelihood π φ m₂ := by
  classical
  simp [ldaHistogramLikelihood, Multiset.count_add, pow_add, Finset.prod_mul_distrib]

/-- The exact count sketch recovers the same bag-of-words LDA likelihood at the root. -/
theorem ldaDocumentLikelihood_exact_on_tree (π : κ → ℝ) (φ : κ → α → ℝ)
    (T : BinTree (List α)) :
    ldaDocumentLikelihood π φ (sketchSummary (countSketchOperator (α := α)) T)
      = ldaDocumentLikelihood π φ (S T) := by
  simp [ldaDocumentLikelihood, bagOfWords_sketchSummary_countSketch]

end LDADocumentLikelihood

end FormalProofs.OPT
