import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.MarkovCountSketchExample
import FormalProofs.OPT.TopicBigramOracle
import FormalProofs.OPT.BagOfWordsLDARecovery
import FormalProofs.OPT.SketchSummaryOperators

/-!
# FormalProofs/OPT/ExactUtilityTransportInstances.lean

Concrete exact-state instances for the exact utility-transport suite.

These are the theorem-backed exact-control lanes behind the new simulations:

1. Markov changepoint exact state
2. Nonseparable complementarity counts
3. Topic mass / topic-plus-boundary exact sketches
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

section MarkovUtility

open MarkovCountSketch

variable {n : ℕ}
variable {β : Type*}

/-- Count-only exact control utility on Markov sketch states. -/
def markovCountOnlyUtility (truth pred : MarkovCountSketch n) : ℝ :=
  if truth.count = pred.count then 1 else 0

/-- Full exact-state exact control utility on Markov sketch states. -/
def markovCountEndpointsUtility (truth pred : MarkovCountSketch n) : ℝ :=
  if truth = pred then 1 else 0

/-- Any downstream utility on the exact Markov merge state is preserved exactly
by the tree. -/
theorem markovStateUtility_exact_on_tree
    (u : MarkovCountSketch n → β)
    (T : BinTree (MarkovCountSketch n)) :
    u (mergeFold (encode := fun s => s) (merge := (· * ·)) T) = u (S T) := by
  simpa using
    (mergeableStateUtility_exact_on_tree
      (Strings := MarkovCountSketch n)
      (Sketch := MarkovCountSketch n)
      (encode := fun s => s)
      (merge := (· * ·))
      (feature := fun s => s)
      (h_encode := fun _ => rfl)
      (h_merge := fun _ _ => rfl)
      (u := u)
      (T := T))

/-- The Markov count-only control utility reaches its exact optimum on the exact
Markov tree fold. -/
theorem markovCountOnlyUtility_exact_on_tree
    (T : BinTree (MarkovCountSketch n)) :
    markovCountOnlyUtility (truth := S T)
      (mergeFold (encode := fun s => s) (merge := (· * ·)) T) = 1 := by
  rw [mergeFold_eq_feature
    (encode := fun s : MarkovCountSketch n => s)
    (merge := (· * ·))
    (feature := fun s : MarkovCountSketch n => s)
    (h_encode := fun _ => rfl)
    (h_merge := fun _ _ => rfl)
    (T := T)]
  simp [markovCountOnlyUtility]

/-- The Markov endpoint-sensitive exact utility also reaches its optimum on the
exact Markov tree fold. -/
theorem markovCountEndpointsUtility_exact_on_tree
    (T : BinTree (MarkovCountSketch n)) :
    markovCountEndpointsUtility (truth := S T)
      (mergeFold (encode := fun s => s) (merge := (· * ·)) T) = 1 := by
  rw [mergeFold_eq_feature
    (encode := fun s : MarkovCountSketch n => s)
    (merge := (· * ·))
    (feature := fun s : MarkovCountSketch n => s)
    (h_encode := fun _ => rfl)
    (h_merge := fun _ _ => rfl)
    (T := T)]
  simp [markovCountEndpointsUtility]

end MarkovUtility

section ComplementarityUtility

variable {β : Type*}

/-- Two-token synthetic alphabet for the exact complementarity lane. -/
inductive ComplementarityToken
| left
| right
deriving DecidableEq, Repr, Encodable

/-- Exact left/right count state for the complementarity lane. -/
def complementarityCounts : List ComplementarityToken → ℕ × ℕ
  | [] => (0, 0)
  | ComplementarityToken.left :: xs =>
      let c := complementarityCounts xs
      (c.1 + 1, c.2)
  | ComplementarityToken.right :: xs =>
      let c := complementarityCounts xs
      (c.1, c.2 + 1)

/-- Merge rule for left/right count states. -/
def mergeComplementarityCounts (s t : ℕ × ℕ) : ℕ × ℕ :=
  (s.1 + t.1, s.2 + t.2)

@[simp] theorem complementarityCounts_nil :
    complementarityCounts ([] : List ComplementarityToken) = (0, 0) := rfl

@[simp] theorem complementarityCounts_cons_left (xs : List ComplementarityToken) :
    complementarityCounts (ComplementarityToken.left :: xs) =
      ((complementarityCounts xs).1 + 1, (complementarityCounts xs).2) := by
  rfl

@[simp] theorem complementarityCounts_cons_right (xs : List ComplementarityToken) :
    complementarityCounts (ComplementarityToken.right :: xs) =
      ((complementarityCounts xs).1, (complementarityCounts xs).2 + 1) := by
  rfl

theorem complementarityCounts_fst_append
    (xs ys : List ComplementarityToken) :
    (complementarityCounts (xs ++ ys)).1 =
      (complementarityCounts xs).1 + (complementarityCounts ys).1 := by
  induction xs with
  | nil =>
      simp [complementarityCounts]
  | cons x xs ih =>
      cases x <;>
        simp [complementarityCounts, ih, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem complementarityCounts_snd_append
    (xs ys : List ComplementarityToken) :
    (complementarityCounts (xs ++ ys)).2 =
      (complementarityCounts xs).2 + (complementarityCounts ys).2 := by
  induction xs with
  | nil =>
      simp [complementarityCounts]
  | cons x xs ih =>
      cases x <;>
        simp [complementarityCounts, ih, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem complementarityCounts_append
    (xs ys : List ComplementarityToken) :
    mergeComplementarityCounts (complementarityCounts xs) (complementarityCounts ys) =
      complementarityCounts (xs ++ ys) := by
  ext <;>
    simp [mergeComplementarityCounts, complementarityCounts_fst_append,
      complementarityCounts_snd_append, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

/-- Threshold-AND downstream utility for the complementarity lane. -/
def complementarityThresholdUtility (kL kR : ℕ) (s : ℕ × ℕ) : ℝ :=
  if kL ≤ s.1 ∧ kR ≤ s.2 then 1 else 0

/-- Any downstream utility on the exact complementarity state is preserved
exactly by the tree. -/
theorem complementarityStateUtility_exact_on_tree
    (u : ℕ × ℕ → β)
    (T : BinTree (List ComplementarityToken)) :
    u (mergeFold
      (encode := complementarityCounts)
      (merge := mergeComplementarityCounts) T) =
      u (complementarityCounts (S T)) := by
  simpa using
    (mergeableStateUtility_exact_on_tree
      (Strings := List ComplementarityToken)
      (Sketch := ℕ × ℕ)
      (encode := complementarityCounts)
      (merge := mergeComplementarityCounts)
      (feature := complementarityCounts)
      (h_encode := fun _ => rfl)
      (h_merge := complementarityCounts_append)
      (u := u)
      (T := T))

/-- The threshold-AND complementarity utility is exactly preserved by the tree. -/
theorem complementarityThresholdUtility_exact_on_tree
    (kL kR : ℕ)
    (T : BinTree (List ComplementarityToken)) :
    complementarityThresholdUtility kL kR
      (mergeFold
        (encode := complementarityCounts)
        (merge := mergeComplementarityCounts) T) =
      complementarityThresholdUtility kL kR (complementarityCounts (S T)) := by
  exact complementarityStateUtility_exact_on_tree
    (u := complementarityThresholdUtility kL kR) (T := T)

end ComplementarityUtility

section TopicUtility

variable {α : Type*} [DecidableEq α]
variable {β : Type*}

/-- Exact tree fold of the topic unigram+boundary sketch. -/
def topicSketchTree : BinTree (List α) → UniBigramSketch α
  | BinTree.leaf xs => uniBigramSketch xs
  | BinTree.node TL TR => mergeUniBigramSketch (topicSketchTree TL) (topicSketchTree TR)

/-- Folding exact topic unigram+boundary sketches over the tree recovers the
full-document sketch exactly. -/
theorem topicSketchTree_eq_full
    (T : BinTree (List α)) :
    topicSketchTree T = uniBigramSketch (S T) := by
  induction T with
  | leaf xs =>
      rfl
  | node TL TR ihL ihR =>
      simpa [topicSketchTree, S, ihL, ihR] using
        (uniBigramSketch_append (xs := S TL) (ys := S TR)).symm

/-- Any downstream utility on the exact topic unigram+boundary state is
preserved exactly by the tree. -/
theorem topicSketchUtility_exact_on_tree
    (u : UniBigramSketch α → β)
    (T : BinTree (List α)) :
    u (topicSketchTree T) = u (uniBigramSketch (S T)) := by
  rw [topicSketchTree_eq_full]

/-- Any downstream utility depending only on topic mass (bag-of-words control)
is exactly preserved by the topic tree fold. -/
theorem topicMassUtility_exact_on_tree
    (u : Multiset α → β)
    (T : BinTree (List α)) :
    u (topicSketchTree T).unigrams = u (bagOfWords (S T)) := by
  rw [topicSketchTree_eq_full]
  rfl

/-- The exact topic-plus-boundary oracle score used in the boundary-sensitive
topic simulation is exactly preserved by the topic tree fold. -/
theorem topicOracleFromSketch_exact_on_tree
    (θ : α → ℝ) (W : (α × α) → ℝ) (lam : ℝ)
    (T : BinTree (List α)) :
    oracleFromSketch θ W lam (topicSketchTree T) = oracleOnList θ W lam (S T) := by
  rw [topicSketchTree_eq_full]
  rfl

end TopicUtility

end FormalProofs.OPT
