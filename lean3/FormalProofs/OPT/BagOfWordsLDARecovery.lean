import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.InformationRepresentationSufficiency
import FormalProofs.OPT.HybridSummarySufficiency
import FormalProofs.OPT.UniformG

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

/-- Deterministic tree fold for bag-of-words observations.

This is the ordinary LDA input space: leaves are histograms/multisets, not raw
text. -/
def bagOfWordsTree : BinTree (Multiset α) → Multiset α
  | BinTree.leaf m => m
  | BinTree.node TL TR => bagOfWordsTree TL + bagOfWordsTree TR

/-- Exact shared-`g` instance for ordinary bag-of-words LDA.

The carrier is the bag-of-words space itself and the shared endomap is `id`;
merge inputs are histogram sums. -/
def bagOfWordsExactG : UniformG (Multiset α) (Multiset α) :=
  UniformG.onCarrier (fun m n : Multiset α => m + n) id

/-- The exact shared `g` folds bag-of-words leaves by histogram addition. -/
theorem bagOfWordsExactG_treeEval_eq_bagOfWordsTree (T : BinTree (Multiset α)) :
    UniformG.treeEval (bagOfWordsExactG (α := α)) T = bagOfWordsTree T := by
  induction T with
  | leaf m =>
      rfl
  | node TL TR ihL ihR =>
      change
        UniformG.treeEval (bagOfWordsExactG (α := α)) TL +
          UniformG.treeEval (bagOfWordsExactG (α := α)) TR =
        bagOfWordsTree TL + bagOfWordsTree TR
      rw [ihL, ihR]

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

/-- Token mass of a bag-of-words histogram, as a real number. -/
def histogramTokenMass (m : Multiset α) : ℝ :=
  ∑ w : α, (m.count w : ℝ)

/-- Word log-probability under a fixed topic mixture and topic-word table. -/
def ldaTokenLogProb (π : κ → ℝ) (φ : κ → α → ℝ) (w : α) : ℝ :=
  Real.log (ldaTokenProb π φ w)

/-- Bag-of-words LDA log-likelihood, ignoring the multinomial coefficient.

This is the log-space version of `ldaHistogramLikelihood`, written directly as
the additive sufficient-statistic functional. -/
def ldaHistogramLogLikelihood (π : κ → ℝ) (φ : κ → α → ℝ) (m : Multiset α) : ℝ :=
  ∑ w : α, (m.count w : ℝ) * ldaTokenLogProb π φ w

/-- Per-token average bag-of-words LDA log-likelihood. -/
def ldaAverageLogLikelihood (π : κ → ℝ) (φ : κ → α → ℝ) (m : Multiset α) : ℝ :=
  ldaHistogramLogLikelihood π φ m / histogramTokenMass m

/-- LDA likelihood readout from a bag-of-words histogram. -/
def ldaLikelihoodReadout (θ : (κ → ℝ) × (κ → α → ℝ)) (m : Multiset α) : ℝ :=
  ldaHistogramLikelihood θ.1 θ.2 m

/-- Ordinary bag-of-words LDA likelihood is recovered exactly by the shared
endomorphic `g` on bag observations. -/
theorem ldaHistogramLikelihood_exact_uniformG
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α)) :
    ldaHistogramLikelihood π φ
        (UniformG.treeEval (bagOfWordsExactG (α := α)) T) =
      ldaHistogramLikelihood π φ (bagOfWordsTree T) := by
  rw [bagOfWordsExactG_treeEval_eq_bagOfWordsTree (T := T)]

/-- LDA document likelihood as a likelihood family over token lists. -/
def ldaLikelihoodFamily (θ : (κ → ℝ) × (κ → α → ℝ)) (xs : List α) : ℝ :=
  ldaDocumentLikelihood θ.1 θ.2 xs

/-- Bag-of-words histograms realize the LDA likelihood family by readout. -/
theorem ldaLikelihoodReadout_realizes_bagOfWords :
    LikelihoodReadoutRealizes
      (X := List α)
      (Rep := Multiset α)
      (Θ := (κ → ℝ) × (κ → α → ℝ))
      (Y := ℝ)
      (bagOfWords (α := α))
      (ldaLikelihoodFamily (κ := κ) (α := α))
      (ldaLikelihoodReadout (κ := κ) (α := α)) := by
  intro θ xs
  simp [ldaLikelihoodReadout, ldaLikelihoodFamily, ldaDocumentLikelihood]

/-- The bag-of-words histogram is sufficient for the ordinary LDA document
likelihood family. -/
theorem bagOfWords_ldaLikelihoodFamilySufficient :
    LikelihoodFamilySufficient
      (X := List α)
      (Rep := Multiset α)
      (Θ := (κ → ℝ) × (κ → α → ℝ))
      (Y := ℝ)
      (bagOfWords (α := α))
      (ldaLikelihoodFamily (κ := κ) (α := α)) := by
  intro xs ys hxy θ
  simp [ldaLikelihoodFamily, ldaDocumentLikelihood, hxy]

/-- Any Makinen-style hybrid that includes bag-of-words as its base statistic
is sufficient for the ordinary LDA document likelihood family. The neural
component may carry residual/order information, but it is not needed for the
bag-of-words LDA likelihood itself. -/
theorem lda_bowHybrid_likelihoodFamilySufficient
    {Neural : Type*}
    (neural : List α → Neural) :
    LikelihoodFamilySufficient
      (X := List α)
      (Rep := Multiset α × Neural)
      (Θ := (κ → ℝ) × (κ → α → ℝ))
      (Y := ℝ)
      (HybridSummary (bagOfWords (α := α)) neural)
      (ldaLikelihoodFamily (κ := κ) (α := α)) :=
  hybridLikelihoodSufficient_of_baseSufficient
    (base := bagOfWords (α := α))
    (neural := neural)
    (likelihood := ldaLikelihoodFamily (κ := κ) (α := α))
    bagOfWords_ldaLikelihoodFamilySufficient

/-- If a hybrid `(bagOfWords, neural)` is response-sufficient for an
order/contextual probe family, then the neural component must separate any
probe-distinct documents that share the same bag-of-words histogram. This is
the deterministic residual-information reading of hybrid summaries for
order-sensitive targets. -/
theorem lda_bowHybrid_neural_separates_response_within_bagOfWords
    {Neural Probe Y : Type*}
    {neural : List α → Neural}
    {response : Probe → List α → Y}
    (hHybrid :
      LikelihoodFreeResponseSufficient
        (HybridSummary (bagOfWords (α := α)) neural)
        response)
    {xs ys : List α}
    (hBow : bagOfWords xs = bagOfWords ys)
    (hResponseDistinct : ∃ p : Probe, response p xs ≠ response p ys) :
    neural xs ≠ neural ys :=
  hybridResponseSufficient_neuralSeparatesResponseWithinBase
    (base := bagOfWords (α := α))
    (neural := neural)
    (response := response)
    hHybrid
    hBow
    hResponseDistinct

/-- Histogram addition is the exact merge rule for the ordinary LDA document likelihood. -/
theorem ldaHistogramLikelihood_add (π : κ → ℝ) (φ : κ → α → ℝ)
    (m₁ m₂ : Multiset α) :
    ldaHistogramLikelihood π φ (m₁ + m₂)
      = ldaHistogramLikelihood π φ m₁ * ldaHistogramLikelihood π φ m₂ := by
  classical
  simp [ldaHistogramLikelihood, Multiset.count_add, pow_add, Finset.prod_mul_distrib]

/-- The bag-of-words tree likelihood factors as the product of leaf likelihoods.

Equivalently, after taking logs and normalizing by token count under the usual
positivity assumptions, per-token log likelihood is a token-weighted average of
leaf per-token log likelihoods. -/
theorem ldaHistogramLikelihood_bagOfWordsTree_eq_leaf_prod
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α)) :
    ldaHistogramLikelihood π φ (bagOfWordsTree T) =
      ((leaves T).map (fun m => ldaHistogramLikelihood π φ m)).prod := by
  induction T with
  | leaf m =>
      simp [bagOfWordsTree, leaves]
  | node TL TR ihL ihR =>
      simp [bagOfWordsTree, leaves, List.map_append, ldaHistogramLikelihood_add, ihL, ihR]

/-- The shared bag-of-words `g` recovers the product-of-leaf LDA likelihood. -/
theorem ldaHistogramLikelihood_uniformG_eq_leaf_prod
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α)) :
    ldaHistogramLikelihood π φ
        (UniformG.treeEval (bagOfWordsExactG (α := α)) T) =
      ((leaves T).map (fun m => ldaHistogramLikelihood π φ m)).prod := by
  rw [bagOfWordsExactG_treeEval_eq_bagOfWordsTree (T := T)]
  exact ldaHistogramLikelihood_bagOfWordsTree_eq_leaf_prod
    (π := π) (φ := φ) (T := T)

/-- Histogram token mass is additive under bag addition. -/
theorem histogramTokenMass_add (m₁ m₂ : Multiset α) :
    histogramTokenMass (m₁ + m₂) =
      histogramTokenMass m₁ + histogramTokenMass m₂ := by
  classical
  simp [histogramTokenMass, Multiset.count_add, Finset.sum_add_distrib]

/-- LDA log-likelihood is additive under bag addition. -/
theorem ldaHistogramLogLikelihood_add
    (π : κ → ℝ) (φ : κ → α → ℝ) (m₁ m₂ : Multiset α) :
    ldaHistogramLogLikelihood π φ (m₁ + m₂) =
      ldaHistogramLogLikelihood π φ m₁ + ldaHistogramLogLikelihood π φ m₂ := by
  classical
  simp [ldaHistogramLogLikelihood, Multiset.count_add, Nat.cast_add,
    add_mul, Finset.sum_add_distrib]

/-- Root token mass is the sum of leaf token masses for a bag-of-words tree. -/
theorem histogramTokenMass_bagOfWordsTree_eq_leaf_sum
    (T : BinTree (Multiset α)) :
    histogramTokenMass (bagOfWordsTree T) =
      ((leaves T).map histogramTokenMass).sum := by
  induction T with
  | leaf m =>
      simp [bagOfWordsTree, leaves]
  | node TL TR ihL ihR =>
      simp [bagOfWordsTree, leaves, List.map_append, histogramTokenMass_add, ihL, ihR]

/-- Root LDA log-likelihood is the sum of leaf LDA log-likelihoods. -/
theorem ldaHistogramLogLikelihood_bagOfWordsTree_eq_leaf_sum
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α)) :
    ldaHistogramLogLikelihood π φ (bagOfWordsTree T) =
      ((leaves T).map (fun m => ldaHistogramLogLikelihood π φ m)).sum := by
  induction T with
  | leaf m =>
      simp [bagOfWordsTree, leaves]
  | node TL TR ihL ihR =>
      simp [bagOfWordsTree, leaves, List.map_append,
        ldaHistogramLogLikelihood_add, ihL, ihR]

/-- The shared bag-of-words `g` recovers the sum-of-leaf LDA log-likelihood. -/
theorem ldaHistogramLogLikelihood_uniformG_eq_leaf_sum
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α)) :
    ldaHistogramLogLikelihood π φ
        (UniformG.treeEval (bagOfWordsExactG (α := α)) T) =
      ((leaves T).map (fun m => ldaHistogramLogLikelihood π φ m)).sum := by
  rw [bagOfWordsExactG_treeEval_eq_bagOfWordsTree (T := T)]
  exact ldaHistogramLogLikelihood_bagOfWordsTree_eq_leaf_sum
    (π := π) (φ := φ) (T := T)

private theorem list_sum_map_mass_mul_normalized
    {ι : Type*} (xs : List ι) (mass value : ι → ℝ)
    (hMass : ∀ x : ι, x ∈ xs → mass x ≠ 0) :
    (xs.map (fun x => mass x * (value x / mass x))).sum =
      (xs.map value).sum := by
  induction xs with
  | nil =>
      simp
  | cons x xs ih =>
      have hx : mass x ≠ 0 := hMass x (by simp)
      have hxs : ∀ y : ι, y ∈ xs → mass y ≠ 0 := by
        intro y hy
        exact hMass y (by simp [hy])
      have hxterm : mass x * (value x / mass x) = value x := by
        field_simp [hx]
      simp [hxterm, ih hxs]

/-- Token-mass weighted average of a leaf bag readout. -/
def tokenWeightedBagAverage
    (T : BinTree (Multiset α)) (q : Multiset α → ℝ) : ℝ :=
  ((leaves T).map (fun m => histogramTokenMass m * q m)).sum /
    histogramTokenMass (bagOfWordsTree T)

/-- Average document log-likelihood is the token-weighted average of leaf
average log-likelihoods. -/
theorem ldaAverageLogLikelihood_bagOfWordsTree_eq_tokenWeightedLeafAverage
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α))
    (hLeaf : ∀ m : Multiset α, m ∈ leaves T → histogramTokenMass m ≠ 0) :
    ldaAverageLogLikelihood π φ (bagOfWordsTree T) =
      tokenWeightedBagAverage T (fun m => ldaAverageLogLikelihood π φ m) := by
  unfold ldaAverageLogLikelihood tokenWeightedBagAverage
  rw [ldaHistogramLogLikelihood_bagOfWordsTree_eq_leaf_sum
    (π := π) (φ := φ) (T := T)]
  rw [← list_sum_map_mass_mul_normalized
    (xs := leaves T)
    (mass := fun m : Multiset α => histogramTokenMass m)
    (value := fun m : Multiset α => ldaHistogramLogLikelihood π φ m)
    hLeaf]

/-- The shared bag-of-words `g` recovers the token-weighted average
log-likelihood decomposition. -/
theorem ldaAverageLogLikelihood_uniformG_eq_tokenWeightedLeafAverage
    (π : κ → ℝ) (φ : κ → α → ℝ) (T : BinTree (Multiset α))
    (hLeaf : ∀ m : Multiset α, m ∈ leaves T → histogramTokenMass m ≠ 0) :
    ldaAverageLogLikelihood π φ
        (UniformG.treeEval (bagOfWordsExactG (α := α)) T) =
      tokenWeightedBagAverage T (fun m => ldaAverageLogLikelihood π φ m) := by
  rw [bagOfWordsExactG_treeEval_eq_bagOfWordsTree (T := T)]
  exact ldaAverageLogLikelihood_bagOfWordsTree_eq_tokenWeightedLeafAverage
    (π := π) (φ := φ) (T := T) hLeaf

/-- The exact count sketch recovers the same bag-of-words LDA likelihood at the root. -/
theorem ldaDocumentLikelihood_exact_on_tree (π : κ → ℝ) (φ : κ → α → ℝ)
    (T : BinTree (List α)) :
    ldaDocumentLikelihood π φ (sketchSummary (countSketchOperator (α := α)) T)
      = ldaDocumentLikelihood π φ (S T) := by
  simp [ldaDocumentLikelihood, bagOfWords_sketchSummary_countSketch]

end LDADocumentLikelihood

end FormalProofs.OPT
