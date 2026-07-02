import FormalProofs.OPT.BagOfWordsLDARecovery
import FormalProofs.OPT.BigramSketch
import FormalProofs.OPT.UniformG

/-!
# FormalProofs/OPT/LDAAggregateStatistics.lean

## Leaf-to-document aggregate statistics for the LDA approximation

This file formalizes the deterministic bookkeeping layer used by the LDA
simulations before any statistical error is introduced.

Each token has

- an observed word `word x`;
- a hard or soft topic-weight vector `topicWeight x`.

For a leaf/span, we record:

- token mass;
- word counts;
- soft topic counts;
- word-topic responsibility counts;
- an outer-product word co-occurrence sketch.

The first four fields merge by addition. The co-occurrence field merges by the
usual count outer-product cross terms. Therefore the root statistic produced by
any binary tree over leaves is exactly the statistic computed on the full
document. Normalized topic proportions are then just a readout of the root
topic counts.

Adjacent word co-occurrences are also included as a separate boundary-carrying
bigram sketch, reusing `BigramSketch.lean`.
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

section AdditiveStatistics

variable {τ α κ : Type*} [DecidableEq α]

/-- Token count, written over `ℝ` to match the floating-point simulation layer. -/
def tokenMass (xs : List τ) : ℝ :=
  (xs.map (fun _ => (1 : ℝ))).sum

/-- Word count vector for a span. -/
def wordMass (word : τ → α) (xs : List τ) (a : α) : ℝ :=
  (xs.map (fun x => if word x = a then (1 : ℝ) else 0)).sum

/-- Soft topic-count vector for a span.

For hard topics, use `hardTopicWeight`; for inferred topics, use posterior
responsibility weights. -/
def topicMass (topicWeight : τ → κ → ℝ) (xs : List τ) (k : κ) : ℝ :=
  (xs.map (fun x => topicWeight x k)).sum

/-- Word-topic responsibility counts. -/
def wordTopicMass (word : τ → α) (topicWeight : τ → κ → ℝ)
    (xs : List τ) (a : α) (k : κ) : ℝ :=
  (xs.map (fun x => if word x = a then topicWeight x k else 0)).sum

/-- Outer-product word co-occurrence approximation: `c(a)c(b)`. -/
def wordCoocOuter (word : τ → α) (xs : List τ) (a b : α) : ℝ :=
  wordMass word xs a * wordMass word xs b

@[simp] theorem tokenMass_append (xs ys : List τ) :
    tokenMass (xs ++ ys) = tokenMass xs + tokenMass ys := by
  simp [tokenMass]

@[simp] theorem wordMass_append (word : τ → α) (xs ys : List τ) (a : α) :
    wordMass word (xs ++ ys) a = wordMass word xs a + wordMass word ys a := by
  simp [wordMass]

@[simp] theorem topicMass_append (topicWeight : τ → κ → ℝ) (xs ys : List τ) (k : κ) :
    topicMass topicWeight (xs ++ ys) k =
      topicMass topicWeight xs k + topicMass topicWeight ys k := by
  simp [topicMass]

@[simp] theorem wordTopicMass_append
    (word : τ → α) (topicWeight : τ → κ → ℝ) (xs ys : List τ) (a : α) (k : κ) :
    wordTopicMass word topicWeight (xs ++ ys) a k =
      wordTopicMass word topicWeight xs a k + wordTopicMass word topicWeight ys a k := by
  simp [wordTopicMass]

/-- Aggregate leaf/document statistic for the deterministic LDA approximation. -/
structure LDAAggregateStats (α κ : Type*) where
  tokenMass : ℝ
  wordMass : α → ℝ
  topicMass : κ → ℝ
  wordTopicMass : α → κ → ℝ
  wordCoocMass : α → α → ℝ

omit [DecidableEq α] in
@[ext] lemma LDAAggregateStats.ext {s t : LDAAggregateStats α κ}
    (hToken : s.tokenMass = t.tokenMass)
    (hWord : ∀ a : α, s.wordMass a = t.wordMass a)
    (hTopic : ∀ k : κ, s.topicMass k = t.topicMass k)
    (hWordTopic : ∀ a : α, ∀ k : κ, s.wordTopicMass a k = t.wordTopicMass a k)
    (hCooc : ∀ a b : α, s.wordCoocMass a b = t.wordCoocMass a b) :
    s = t := by
  cases s with
  | mk sn sw st swt sc =>
      cases t with
      | mk tn tw tt twt tc =>
          simp at hToken hWord hTopic hWordTopic hCooc
          cases hToken
          have hw : sw = tw := funext hWord
          have ht : st = tt := funext hTopic
          have hwt : swt = twt := funext (fun a => funext (hWordTopic a))
          have hc : sc = tc := funext (fun a => funext (hCooc a))
          cases hw
          cases ht
          cases hwt
          cases hc
          rfl

/-- Merge two aggregate LDA statistics. Additive fields add directly; the
outer-product word co-occurrence field receives the two cross-leaf terms. -/
def mergeLDAAggregateStats (s t : LDAAggregateStats α κ) : LDAAggregateStats α κ where
  tokenMass := s.tokenMass + t.tokenMass
  wordMass := fun a => s.wordMass a + t.wordMass a
  topicMass := fun k => s.topicMass k + t.topicMass k
  wordTopicMass := fun a k => s.wordTopicMass a k + t.wordTopicMass a k
  wordCoocMass := fun a b =>
    s.wordCoocMass a b + t.wordCoocMass a b
      + s.wordMass a * t.wordMass b
      + t.wordMass a * s.wordMass b

/-- Full aggregate statistic for a token span. -/
def ldaAggregateStats (word : τ → α) (topicWeight : τ → κ → ℝ)
    (xs : List τ) : LDAAggregateStats α κ where
  tokenMass := tokenMass xs
  wordMass := wordMass word xs
  topicMass := topicMass topicWeight xs
  wordTopicMass := wordTopicMass word topicWeight xs
  wordCoocMass := wordCoocOuter word xs

/-- The LDA aggregate statistic of an appended span is the merge of the two
span statistics. -/
theorem ldaAggregateStats_append
    (word : τ → α) (topicWeight : τ → κ → ℝ) (xs ys : List τ) :
    ldaAggregateStats word topicWeight (xs ++ ys) =
      mergeLDAAggregateStats
        (ldaAggregateStats word topicWeight xs)
        (ldaAggregateStats word topicWeight ys) := by
  apply LDAAggregateStats.ext
  · simp [ldaAggregateStats, mergeLDAAggregateStats]
  · intro a
    simp [ldaAggregateStats, mergeLDAAggregateStats]
  · intro k
    simp [ldaAggregateStats, mergeLDAAggregateStats]
  · intro a k
    simp [ldaAggregateStats, mergeLDAAggregateStats]
  · intro a b
    simp [ldaAggregateStats, mergeLDAAggregateStats, wordCoocOuter]
    ring

/-- Bottom-up tree aggregation of LDA leaf statistics. -/
def ldaAggregateTreeStats (word : τ → α) (topicWeight : τ → κ → ℝ) :
    BinTree (List τ) → LDAAggregateStats α κ
  | BinTree.leaf xs => ldaAggregateStats word topicWeight xs
  | BinTree.node TL TR =>
      mergeLDAAggregateStats
        (ldaAggregateTreeStats word topicWeight TL)
        (ldaAggregateTreeStats word topicWeight TR)

/-- Tree aggregation recovers exactly the full-document aggregate statistic. -/
theorem ldaAggregateTreeStats_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) :
    ldaAggregateTreeStats word topicWeight T =
      ldaAggregateStats word topicWeight (S T) := by
  induction T with
  | leaf xs =>
      rfl
  | node TL TR ihL ihR =>
      change
        mergeLDAAggregateStats
          (ldaAggregateTreeStats word topicWeight TL)
          (ldaAggregateTreeStats word topicWeight TR)
          =
        ldaAggregateStats word topicWeight (S TL ++ S TR)
      rw [ihL, ihR]
      exact (ldaAggregateStats_append word topicWeight (S TL) (S TR)).symm

/-- Root word counts agree with full-document word counts. -/
theorem ldaAggregateTree_wordMass_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) (a : α) :
    (ldaAggregateTreeStats word topicWeight T).wordMass a =
      wordMass word (S T) a := by
  rw [ldaAggregateTreeStats_eq_full]
  rfl

/-- Root soft topic counts agree with full-document soft topic counts. -/
theorem ldaAggregateTree_topicMass_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) (k : κ) :
    (ldaAggregateTreeStats word topicWeight T).topicMass k =
      topicMass topicWeight (S T) k := by
  rw [ldaAggregateTreeStats_eq_full]
  rfl

/-- Root word-topic responsibility counts agree with the full-document counts. -/
theorem ldaAggregateTree_wordTopicMass_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) (a : α) (k : κ) :
    (ldaAggregateTreeStats word topicWeight T).wordTopicMass a k =
      wordTopicMass word topicWeight (S T) a k := by
  rw [ldaAggregateTreeStats_eq_full]
  rfl

/-- Root outer-product word co-occurrences agree with the full-document
outer-product count sketch. -/
theorem ldaAggregateTree_wordCoocMass_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) (a b : α) :
    (ldaAggregateTreeStats word topicWeight T).wordCoocMass a b =
      wordCoocOuter word (S T) a b := by
  rw [ldaAggregateTreeStats_eq_full]
  rfl

/-- Topic proportions are a readout from aggregate topic counts. -/
def topicProportion (s : LDAAggregateStats α κ) (k : κ) : ℝ :=
  s.topicMass k / s.tokenMass

/-- Smoothed topic proportions, matching the usual "prior mass + expected topic
counts" shape used by approximate LDA inference. -/
def smoothedTopicProportion (priorMass : ℝ) (prior : κ → ℝ)
    (s : LDAAggregateStats α κ) (k : κ) : ℝ :=
  (prior k + s.topicMass k) / (priorMass + s.tokenMass)

/-- Empirical word proportion read from an aggregate state. -/
def wordProportion (s : LDAAggregateStats α κ) (a : α) : ℝ :=
  s.wordMass a / s.tokenMass

/-- Empirical joint word-topic mass, normalized by total token mass. -/
def wordTopicJointProportion (s : LDAAggregateStats α κ) (a : α) (k : κ) : ℝ :=
  s.wordTopicMass a k / s.tokenMass

/-- Topic-conditional word distribution read from word-topic responsibility
counts. -/
def wordGivenTopicProportion (s : LDAAggregateStats α κ) (a : α) (k : κ) : ℝ :=
  s.wordTopicMass a k / s.topicMass k

/-- Tree topic proportions are exactly the proportions computed from the full
document aggregate statistic. -/
theorem topicProportion_tree_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) (k : κ) :
    topicProportion (ldaAggregateTreeStats word topicWeight T) k =
      topicProportion (ldaAggregateStats word topicWeight (S T)) k := by
  rw [ldaAggregateTreeStats_eq_full]

/-- The same exactness holds for smoothed topic proportions. -/
theorem smoothedTopicProportion_tree_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ)
    (priorMass : ℝ) (prior : κ → ℝ) (T : BinTree (List τ)) (k : κ) :
    smoothedTopicProportion priorMass prior
        (ldaAggregateTreeStats word topicWeight T) k =
      smoothedTopicProportion priorMass prior
        (ldaAggregateStats word topicWeight (S T)) k := by
  rw [ldaAggregateTreeStats_eq_full]

/-- Hard topic assignment as a soft topic-weight vector. -/
def hardTopicWeight [DecidableEq κ] (topic : τ → κ) : τ → κ → ℝ :=
  fun x k => if topic x = k then (1 : ℝ) else 0

/-- If each token has topic weights summing to one, then aggregate topic mass
sums to token mass. -/
theorem sum_topicMass_eq_tokenMass_of_simplex [Fintype κ]
    (topicWeight : τ → κ → ℝ)
    (hSimplex : ∀ x : τ, (∑ k : κ, topicWeight x k) = 1)
    (xs : List τ) :
    (∑ k : κ, topicMass topicWeight xs k) = tokenMass xs := by
  induction xs with
  | nil =>
      simp [topicMass, tokenMass]
  | cons x xs ih =>
      calc
        (∑ k : κ, topicMass topicWeight (x :: xs) k)
            = ∑ k : κ, (topicWeight x k + topicMass topicWeight xs k) := by
                simp [topicMass]
        _ = (∑ k : κ, topicWeight x k) + (∑ k : κ, topicMass topicWeight xs k) := by
                rw [Finset.sum_add_distrib]
        _ = 1 + tokenMass xs := by
                rw [hSimplex x, ih]
        _ = tokenMass (x :: xs) := by
                simp [tokenMass]

end AdditiveStatistics

section OracleSummaryDecomposition

variable {τ α κ β : Type*} [DecidableEq α]

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

/-- LDA-specialized name for generic bottom-up evaluation of one shared `g`.

The theorem-facing LDA leaves are bag/statistical observations, not raw text.
Token lists above are one concrete realization used to derive those bag
statistics. -/
abbrev ldaGTreeEval
    {Carrier : Type*}
    (G : UniformG (LDAAggregateStats α κ) Carrier) :
    BinTree (LDAAggregateStats α κ) → Carrier :=
  UniformG.treeEval G

/-- Deterministic tree fold for bag-level LDA statistics. -/
def ldaBagTreeStats : BinTree (LDAAggregateStats α κ) → LDAAggregateStats α κ
  | BinTree.leaf s => s
  | BinTree.node TL TR =>
      mergeLDAAggregateStats (ldaBagTreeStats TL) (ldaBagTreeStats TR)

/-- Root token mass is the sum of leaf token masses. -/
theorem ldaBagTreeStats_tokenMass_eq_leaf_sum
    (T : BinTree (LDAAggregateStats α κ)) :
    (ldaBagTreeStats T).tokenMass =
      ((leaves T).map (fun s => s.tokenMass)).sum := by
  induction T with
  | leaf s =>
      simp [ldaBagTreeStats, leaves]
  | node TL TR ihL ihR =>
      simp [ldaBagTreeStats, mergeLDAAggregateStats, leaves, List.map_append, ihL, ihR]

/-- Root word mass is the componentwise sum of leaf word masses. -/
theorem ldaBagTreeStats_wordMass_eq_leaf_sum
    (T : BinTree (LDAAggregateStats α κ)) (a : α) :
    (ldaBagTreeStats T).wordMass a =
      ((leaves T).map (fun s => s.wordMass a)).sum := by
  induction T with
  | leaf s =>
      simp [ldaBagTreeStats, leaves]
  | node TL TR ihL ihR =>
      simp [ldaBagTreeStats, mergeLDAAggregateStats, leaves, List.map_append, ihL, ihR]

/-- Root topic mass is the componentwise sum of leaf topic masses. -/
theorem ldaBagTreeStats_topicMass_eq_leaf_sum
    (T : BinTree (LDAAggregateStats α κ)) (k : κ) :
    (ldaBagTreeStats T).topicMass k =
      ((leaves T).map (fun s => s.topicMass k)).sum := by
  induction T with
  | leaf s =>
      simp [ldaBagTreeStats, leaves]
  | node TL TR ihL ihR =>
      simp [ldaBagTreeStats, mergeLDAAggregateStats, leaves, List.map_append, ihL, ihR]

/-- Root word-topic responsibility mass is the componentwise sum of leaf
word-topic masses. -/
theorem ldaBagTreeStats_wordTopicMass_eq_leaf_sum
    (T : BinTree (LDAAggregateStats α κ)) (a : α) (k : κ) :
    (ldaBagTreeStats T).wordTopicMass a k =
      ((leaves T).map (fun s => s.wordTopicMass a k)).sum := by
  induction T with
  | leaf s =>
      simp [ldaBagTreeStats, leaves]
  | node TL TR ihL ihR =>
      simp [ldaBagTreeStats, mergeLDAAggregateStats, leaves, List.map_append, ihL, ihR]

/-- Token-mass weighted average of a leaf readout. -/
def tokenWeightedLeafAverage
    (T : BinTree (LDAAggregateStats α κ))
    (q : LDAAggregateStats α κ → ℝ) : ℝ :=
  ((leaves T).map (fun s => s.tokenMass * q s)).sum /
    (ldaBagTreeStats T).tokenMass

/-- Topic-mass weighted average of a leaf readout for topic `k`. -/
def topicMassWeightedLeafAverage
    (T : BinTree (LDAAggregateStats α κ)) (k : κ)
    (q : LDAAggregateStats α κ → ℝ) : ℝ :=
  ((leaves T).map (fun s => s.topicMass k * q s)).sum /
    (ldaBagTreeStats T).topicMass k

/-- Document topic proportions are token-weighted averages of leaf topic
proportions. -/
theorem lda_topicProportion_eq_tokenWeightedLeafAverage
    (T : BinTree (LDAAggregateStats α κ)) (k : κ)
    (hLeaf : ∀ s : LDAAggregateStats α κ, s ∈ leaves T → s.tokenMass ≠ 0) :
    topicProportion (ldaBagTreeStats T) k =
      tokenWeightedLeafAverage T (fun s => topicProportion s k) := by
  unfold tokenWeightedLeafAverage topicProportion
  rw [ldaBagTreeStats_topicMass_eq_leaf_sum (T := T) (k := k)]
  rw [← list_sum_map_mass_mul_normalized
    (xs := leaves T)
    (mass := fun s : LDAAggregateStats α κ => s.tokenMass)
    (value := fun s : LDAAggregateStats α κ => s.topicMass k)
    hLeaf]

/-- Document word proportions are token-weighted averages of leaf word
proportions. -/
theorem lda_wordProportion_eq_tokenWeightedLeafAverage
    (T : BinTree (LDAAggregateStats α κ)) (a : α)
    (hLeaf : ∀ s : LDAAggregateStats α κ, s ∈ leaves T → s.tokenMass ≠ 0) :
    wordProportion (ldaBagTreeStats T) a =
      tokenWeightedLeafAverage T (fun s => wordProportion s a) := by
  unfold tokenWeightedLeafAverage wordProportion
  rw [ldaBagTreeStats_wordMass_eq_leaf_sum (T := T) (a := a)]
  rw [← list_sum_map_mass_mul_normalized
    (xs := leaves T)
    (mass := fun s : LDAAggregateStats α κ => s.tokenMass)
    (value := fun s : LDAAggregateStats α κ => s.wordMass a)
    hLeaf]

/-- Document word-topic joint proportions are token-weighted averages of leaf
word-topic joint proportions. -/
theorem lda_wordTopicJointProportion_eq_tokenWeightedLeafAverage
    (T : BinTree (LDAAggregateStats α κ)) (a : α) (k : κ)
    (hLeaf : ∀ s : LDAAggregateStats α κ, s ∈ leaves T → s.tokenMass ≠ 0) :
    wordTopicJointProportion (ldaBagTreeStats T) a k =
      tokenWeightedLeafAverage T (fun s => wordTopicJointProportion s a k) := by
  unfold tokenWeightedLeafAverage wordTopicJointProportion
  rw [ldaBagTreeStats_wordTopicMass_eq_leaf_sum (T := T) (a := a) (k := k)]
  rw [← list_sum_map_mass_mul_normalized
    (xs := leaves T)
    (mass := fun s : LDAAggregateStats α κ => s.tokenMass)
    (value := fun s : LDAAggregateStats α κ => s.wordTopicMass a k)
    hLeaf]

/-- Topic-conditional word proportions are topic-mass weighted averages of leaf
topic-conditional word proportions. -/
theorem lda_wordGivenTopicProportion_eq_topicMassWeightedLeafAverage
    (T : BinTree (LDAAggregateStats α κ)) (a : α) (k : κ)
    (hLeaf : ∀ s : LDAAggregateStats α κ, s ∈ leaves T → s.topicMass k ≠ 0) :
    wordGivenTopicProportion (ldaBagTreeStats T) a k =
      topicMassWeightedLeafAverage T k (fun s => wordGivenTopicProportion s a k) := by
  unfold topicMassWeightedLeafAverage wordGivenTopicProportion
  rw [ldaBagTreeStats_wordTopicMass_eq_leaf_sum (T := T) (a := a) (k := k)]
  rw [← list_sum_map_mass_mul_normalized
    (xs := leaves T)
    (mass := fun s : LDAAggregateStats α κ => s.topicMass k)
    (value := fun s : LDAAggregateStats α κ => s.wordTopicMass a k)
    hLeaf]

/-- Map a token-realization tree to the bag/statistic tree used by the
theorem-facing LDA `g`. -/
def ldaTokenTreeBags (word : τ → α) (topicWeight : τ → κ → ℝ) :
    BinTree (List τ) → BinTree (LDAAggregateStats α κ)
  | BinTree.leaf xs => BinTree.leaf (ldaAggregateStats word topicWeight xs)
  | BinTree.node TL TR =>
      BinTree.node
        (ldaTokenTreeBags word topicWeight TL)
        (ldaTokenTreeBags word topicWeight TR)

/-- Folding the bag/statistic tree generated from token realizations recovers
the same full-document aggregate statistic as the token-level derivation. -/
theorem ldaBagTreeStats_tokenTreeBags_eq_full
    (word : τ → α) (topicWeight : τ → κ → ℝ) (T : BinTree (List τ)) :
    ldaBagTreeStats (ldaTokenTreeBags word topicWeight T) =
      ldaAggregateStats word topicWeight (S T) := by
  induction T with
  | leaf xs =>
      rfl
  | node TL TR ihL ihR =>
      change
        mergeLDAAggregateStats
          (ldaBagTreeStats (ldaTokenTreeBags word topicWeight TL))
          (ldaBagTreeStats (ldaTokenTreeBags word topicWeight TR))
          =
        ldaAggregateStats word topicWeight (S TL ++ S TR)
      rw [ihL, ihR]
      exact (ldaAggregateStats_append word topicWeight (S TL) (S TR)).symm

/-- Unified carrier for the exact LDA `g`.

The carrier contains both bag-level leaf observations and aggregate summary
states. This is the explicit `X` in the theorem-facing shape `g : X → X`,
`f : X → Y`. -/
inductive LDACarrier (α κ : Type*) where
  | bag : LDAAggregateStats α κ → LDACarrier α κ
  | stats : LDAAggregateStats α κ → LDACarrier α κ

/-- Interpret any LDA carrier element as its aggregate statistic. -/
def ldaCarrierStats :
    LDACarrier α κ → LDAAggregateStats α κ
  | LDACarrier.bag s => s
  | LDACarrier.stats s => s

/-- Endomorphic exact LDA summarizer on the unified carrier. -/
def ldaCarrierG :
    LDACarrier α κ → LDACarrier α κ
  | LDACarrier.bag s => LDACarrier.stats s
  | LDACarrier.stats s => LDACarrier.stats s

/-- Carrier-level merge input constructor for exact LDA. -/
def ldaCarrierMergeInput
    (s t : LDACarrier α κ) : LDACarrier α κ :=
  LDACarrier.stats
    (mergeLDAAggregateStats
      (ldaCarrierStats s)
      (ldaCarrierStats t))

@[simp] theorem ldaCarrierStats_g
    (s : LDACarrier α κ) :
    ldaCarrierStats (ldaCarrierG s) = ldaCarrierStats s := by
  cases s <;> rfl

@[simp] theorem ldaCarrierStats_mergeInput
    (s t : LDACarrier α κ) :
    ldaCarrierStats (ldaCarrierMergeInput s t) =
      mergeLDAAggregateStats (ldaCarrierStats s) (ldaCarrierStats t) :=
  rfl

/-- Exact LDA summary operator `g`.

This uses one carrier space `LDACarrier`: leaves enter as bag/statistic
observations, merge inputs enter as carrier-level aggregate states, and the
shared learned map has type `LDACarrier → LDACarrier`. -/
def ldaExactG :
    UniformG (LDAAggregateStats α κ) (LDACarrier α κ) where
  leafInput := LDACarrier.bag
  mergeInput := ldaCarrierMergeInput
  g := ldaCarrierG

/-- The exact `g` summary recovers the full-document LDA aggregate state after
the carrier-state readout. -/
theorem ldaGTreeEval_exact_eq_full
    (T : BinTree (LDAAggregateStats α κ)) :
    ldaCarrierStats (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) T) =
      ldaBagTreeStats T := by
  induction T with
  | leaf s =>
      rfl
  | node TL TR ihL ihR =>
      change
        ldaCarrierStats
          (ldaCarrierG
            (ldaCarrierMergeInput
              (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) TL)
              (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) TR))) =
          mergeLDAAggregateStats (ldaBagTreeStats TL) (ldaBagTreeStats TR)
      rw [ldaCarrierStats_g]
      change
        mergeLDAAggregateStats
          (ldaCarrierStats
            (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) TL))
          (ldaCarrierStats
            (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) TR)) =
          mergeLDAAggregateStats (ldaBagTreeStats TL) (ldaBagTreeStats TR)
      rw [ihL, ihR]

/-- A document-level LDA target `fstar` factors through the aggregate LDA state
when some oracle/readout `f` from aggregate summaries recovers it on every
bag observation. This is the formal version of "doc-level supervision with
`f*` is enough" once the hypothesis class contains the exact LDA summary `g`. -/
def LDAFStarFactorsThroughSummary
    (fstar : LDAAggregateStats α κ → β) : Prop :=
  ∃ f : LDAAggregateStats α κ → β,
    ∀ s : LDAAggregateStats α κ, f s = fstar s

/-- Exact recovery of a document-level target by a summary `g` and an
oracle/readout `f`. -/
def LDAExactOracleSummaryRecovery
    {Carrier : Type*}
    (G : UniformG (LDAAggregateStats α κ) Carrier)
    (f : Carrier → β)
    (fstar : LDAAggregateStats α κ → β) : Prop :=
  ∀ T : BinTree (LDAAggregateStats α κ),
    f (ldaGTreeEval G T) = fstar (ldaBagTreeStats T)

/-- Zero doc-level supervision loss on a training/support predicate. This is a
pure equality version of root supervision against `fstar`. -/
def LDADocSupervisionZero
    {Carrier : Type*}
    (train : BinTree (LDAAggregateStats α κ) → Prop)
    (G : UniformG (LDAAggregateStats α κ) Carrier)
    (f : Carrier → β)
    (fstar : LDAAggregateStats α κ → β) : Prop :=
  ∀ T : BinTree (LDAAggregateStats α κ),
    train T → f (ldaGTreeEval G T) = fstar (ldaBagTreeStats T)

/-- If an LDA target factors through the aggregate state, then the exact
summary `g` and the factorization oracle/readout `f` recover `fstar` exactly on
every document tree. -/
theorem lda_exact_summary_recovers_fstar_of_factorization
    (fstar : LDAAggregateStats α κ → β)
    (hFactor : LDAFStarFactorsThroughSummary (α := α) (κ := κ) fstar) :
    ∃ f : LDACarrier α κ → β,
      LDAExactOracleSummaryRecovery
        (ldaExactG (α := α) (κ := κ))
        f
        fstar := by
  rcases hFactor with ⟨f, hf⟩
  refine ⟨fun s => f (ldaCarrierStats s), ?_⟩
  intro T
  change
    f (ldaCarrierStats (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) T)) =
      fstar (ldaBagTreeStats T)
  rw [ldaGTreeEval_exact_eq_full (T := T)]
  exact hf (ldaBagTreeStats T)

/-- The exact summary/oracle pair also has zero doc-level supervision error on any
training/support predicate. -/
theorem lda_exact_summary_zero_doc_supervision_of_factorization
    (fstar : LDAAggregateStats α κ → β)
    (train : BinTree (LDAAggregateStats α κ) → Prop)
    (hFactor : LDAFStarFactorsThroughSummary (α := α) (κ := κ) fstar) :
    ∃ f : LDACarrier α κ → β,
      LDADocSupervisionZero
        train
        (ldaExactG (α := α) (κ := κ))
        f
        fstar := by
  rcases lda_exact_summary_recovers_fstar_of_factorization
      (α := α) (κ := κ) (fstar := fstar) hFactor with
    ⟨f, hExact⟩
  refine ⟨f, ?_⟩
  intro T _hT
  exact hExact T

/-- There exists an exact summary `g` for every aggregate-state oracle/readout
`f`. This is the concrete realizability theorem for LDA-style targets. -/
theorem exists_lda_exact_g_for_oracle_readout
    (f : LDAAggregateStats α κ → β) :
    ∃ G : UniformG (LDAAggregateStats α κ) (LDACarrier α κ),
      LDAExactOracleSummaryRecovery
        G (fun s : LDACarrier α κ => f (ldaCarrierStats s))
        f := by
  refine ⟨ldaExactG (α := α) (κ := κ), ?_⟩
  intro T
  change
    f (ldaCarrierStats (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) T)) =
      f (ldaBagTreeStats T)
  rw [ldaGTreeEval_exact_eq_full (T := T)]

/-- Concrete target: the `k`th document topic proportion. -/
def ldaTopicProportionFStar (k : κ) :
    LDAAggregateStats α κ → ℝ :=
  fun s => topicProportion s k

/-- The exact summary `g` plus topic-proportion oracle/readout `f` recovers
document topic proportions. -/
theorem lda_topicProportion_exact_summary_recovery (k : κ) :
    LDAExactOracleSummaryRecovery
      (ldaExactG (α := α) (κ := κ))
      (fun s : LDACarrier α κ => topicProportion (ldaCarrierStats s) k)
      (ldaTopicProportionFStar (α := α) (κ := κ) k) := by
  intro T
  change
    topicProportion
        (ldaCarrierStats
          (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) T))
        k =
      topicProportion (ldaBagTreeStats T) k
  rw [ldaGTreeEval_exact_eq_full (T := T)]

/-- Concrete target: smoothed `k`th document topic proportion. -/
def ldaSmoothedTopicProportionFStar
    (priorMass : ℝ) (prior : κ → ℝ) (k : κ) :
    LDAAggregateStats α κ → ℝ :=
  fun s => smoothedTopicProportion priorMass prior s k

/-- The exact summary `g` plus smoothed-topic-proportion oracle/readout `f`
recovers smoothed document topic proportions. -/
theorem lda_smoothedTopicProportion_exact_summary_recovery
    (priorMass : ℝ) (prior : κ → ℝ) (k : κ) :
    LDAExactOracleSummaryRecovery
      (ldaExactG (α := α) (κ := κ))
      (fun s : LDACarrier α κ =>
        smoothedTopicProportion priorMass prior (ldaCarrierStats s) k)
      (ldaSmoothedTopicProportionFStar (α := α) (κ := κ) priorMass prior k) := by
  intro T
  change
    smoothedTopicProportion priorMass prior
        (ldaCarrierStats
          (ldaGTreeEval (ldaExactG (α := α) (κ := κ)) T))
        k =
      smoothedTopicProportion priorMass prior (ldaBagTreeStats T) k
  rw [ldaGTreeEval_exact_eq_full (T := T)]

end OracleSummaryDecomposition

section AdjacentWordCooccurrences

variable {τ α : Type*} [DecidableEq α]

/-- Adjacent-word co-occurrence sketch obtained by mapping tokens to words and
then using the boundary-carrying `BigramSketch`. -/
def wordBigramSketch (word : τ → α) (xs : List τ) : BigramSketch α :=
  bigramSketch (xs.map word)

/-- Word bigram sketches merge exactly across concatenation. -/
theorem wordBigramSketch_append (word : τ → α) (xs ys : List τ) :
    wordBigramSketch word (xs ++ ys) =
      mergeSketch (wordBigramSketch word xs) (wordBigramSketch word ys) := by
  simpa [wordBigramSketch, List.map_append] using
    (bigramSketch_append (xs := xs.map word) (ys := ys.map word))

/-- Bottom-up tree aggregation of adjacent-word bigram sketches. -/
def wordBigramTreeSketch (word : τ → α) : BinTree (List τ) → BigramSketch α
  | BinTree.leaf xs => wordBigramSketch word xs
  | BinTree.node TL TR =>
      mergeSketch (wordBigramTreeSketch word TL) (wordBigramTreeSketch word TR)

/-- Boundary-carrying adjacent-word co-occurrence aggregation recovers the full
document word bigram sketch. -/
theorem wordBigramTreeSketch_eq_full (word : τ → α) (T : BinTree (List τ)) :
    wordBigramTreeSketch word T = wordBigramSketch word (S T) := by
  induction T with
  | leaf xs =>
      rfl
  | node TL TR ihL ihR =>
      change
        mergeSketch (wordBigramTreeSketch word TL) (wordBigramTreeSketch word TR)
          =
        wordBigramSketch word (S TL ++ S TR)
      rw [ihL, ihR]
      exact (wordBigramSketch_append word (S TL) (S TR)).symm

/-- The adjacent-word co-occurrence multiset at the root equals the full-document
adjacent-word co-occurrence multiset. -/
theorem wordBigramTreeSketch_pairs_eq_full (word : τ → α) (T : BinTree (List τ)) :
    (wordBigramTreeSketch word T).pairs =
      (wordBigramSketch word (S T)).pairs := by
  rw [wordBigramTreeSketch_eq_full]

end AdjacentWordCooccurrences

end FormalProofs.OPT
