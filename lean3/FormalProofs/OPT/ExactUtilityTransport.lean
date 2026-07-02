import FormalProofs.OPT.CoreDefinitions
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.MarkovCountSketchExample
import FormalProofs.OPT.TopicBigramOracle
import FormalProofs.OPT.BagOfWordsLDARecovery

/-!
# FormalProofs/OPT/ExactUtilityTransport.lean

This file packages the exact-utility transport layer that sits underneath the
new exact utility-transport simulations.

There are two complementary results here:

1. **Oracle-indexed objective transport**:
   any objective that factors through an exact latent/feature oracle inherits
   the generic zero-distortion transport theorem already proved in
   `PreferenceLearning.lean`.
2. **Exact mergeable-state utility preservation**:
   if a latent state is represented by an exact mergeable fold
   (`encode` at leaves, `merge` at internal nodes), then **any utility on that
   exact state** is preserved by the tree exactly.

This is the formal bridge we need for the exact Markov / nonseparable /
boundary-topic simulation lanes:

- the objective family can vary (supervised state loss, reward, pairwise loss,
  group/listwise loss, PPO-style reward optimization);
- the core latent state can still be treated as the theorem-bearing object.
-/

/-! ## Original FormalProofs/OPT/ExactUtilityTransport.lean content -/

section

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

section OracleIndexedObjectives

variable {Strings : Type*} [Monoid Strings]
variable {Feature : Type*} [Encodable Feature]
variable {α : Type*}

/-- An oracle-indexed objective obtained by composing a feature/state oracle
with an objective on that oracle value. -/
def featureIndexedObjective
    (feature : Strings → Feature) (objective : Feature → α → ℝ) :
    Strings → α → ℝ :=
  fun x a => objective (feature x) a

/-- A direct supervised-state loss: compare a predicted feature/state `a` to the
true feature/state of document `x`. -/
def supervisedStateLoss
    (feature : Strings → Feature) (loss : Feature → Feature → ℝ) :
    Strings → Feature → ℝ :=
  fun x a => loss a (feature x)

/-- Constant generator used to view deterministic utility evaluation as a generic
expected loss. -/
def trivialExampleGenerator (Strings : Type*) : Strings → PMF PUnit :=
  fun _ => PMF.pure PUnit.unit

/-- Objectives that depend on a document only through an encoded feature/state
are oracle-measurable for the encoded-feature oracle. -/
lemma oracleMeasurableLossGeneric_of_featureIndexedObjective
    (feature : Strings → Feature) (objective : Feature → α → ℝ) :
    OracleMeasurableLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective)
      (encodedOracle (Strings := Strings) feature) := by
  intro x x' a hdist
  have hreal :
      encodedOracle (Strings := Strings) feature x =
        encodedOracle (Strings := Strings) feature x' := by
    exact dist_eq_zero.mp hdist
  have hcast :
      ((Encodable.encode (feature x) : ℕ) : ℝ) =
        ((Encodable.encode (feature x') : ℕ) : ℝ) := by
    simpa [encodedOracle] using hreal
  have hcode : Encodable.encode (feature x) = Encodable.encode (feature x') := by
    exact_mod_cast hcast
  have hfeature : feature x = feature x' := Encodable.encode_injective hcode
  simp [featureIndexedObjective, hfeature]

/-- The constant unit generator is oracle-indexed for every oracle. -/
lemma oracleIndexedGenGeneric_trivialExampleGenerator
    (fstar : Strings → ℝ) :
    OracleIndexedGenGeneric
      (trivialExampleGenerator Strings) fstar := by
  intro x x' hdist
  simp [trivialExampleGenerator]

/-- Generic oracle-indexed feature/state objective transport.

This is the exact objective-level theorem used by the new utility-transport
suite: once the objective factors through a feature/state oracle, the generic
zero-distortion theorem applies immediately. -/
theorem featureIndexedObjective_eq_of_zero_dist
    (feature : Strings → Feature)
    (objective : Feature → α → ℝ)
    (gen : Strings → PMF α)
    (μ_X μ_Z : PMF Strings)
    (h_zero :
      ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support →
        dist ((encodedOracle (Strings := Strings) feature) z)
          ((encodedOracle (Strings := Strings) feature) x) = 0)
    (h_gen :
      OracleIndexedGenGeneric gen
        (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective) μ_X gen =
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective) μ_Z gen := by
  exact expected_loss_eq_of_zero_dist_generic
    (fstar := encodedOracle (Strings := Strings) feature)
    (loss := featureIndexedObjective (Strings := Strings) feature objective)
    (gen := gen)
    (μ_X := μ_X)
    (μ_Z := μ_Z)
    h_zero
    (oracleMeasurableLossGeneric_of_featureIndexedObjective
      (Strings := Strings) (feature := feature) (objective := objective))
    h_gen

/-- Direct supervised-state learning is a special case of the generic
oracle-indexed objective transport theorem. -/
theorem supervisedStateExpectedLoss_eq_of_zero_dist
    (feature : Strings → Feature)
    (loss : Feature → Feature → ℝ)
    (gen : Strings → PMF Feature)
    (μ_X μ_Z : PMF Strings)
    (h_zero :
      ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support →
        dist ((encodedOracle (Strings := Strings) feature) z)
          ((encodedOracle (Strings := Strings) feature) x) = 0)
    (h_gen :
      OracleIndexedGenGeneric gen
        (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) feature loss) μ_X gen =
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) feature loss) μ_Z gen := by
  simpa [supervisedStateLoss, featureIndexedObjective] using
    (featureIndexedObjective_eq_of_zero_dist
      (Strings := Strings)
      (feature := feature)
      (objective := fun y a => loss a y)
      (gen := gen)
      (μ_X := μ_X)
      (μ_Z := μ_Z)
      h_zero h_gen)

end OracleIndexedObjectives

section ExactUtilities

variable {Action State β : Type*}

/-- Utility obtained by normalizing an error function. -/
def normalizedErrorUtility
    (err : Action → State → ℝ) (scale : ℝ) :
    Action → State → ℝ :=
  fun a y => 1 - err a y / scale

/-- Utility regret against an exact/reference action. -/
def utilityRegret
    (utility : Action → State → ℝ)
    (aStar a : Action) (y : State) : ℝ :=
  utility aStar y - utility a y

/-- For normalized exact-state utilities, regret against an exact action is just
the normalized error gap. -/
theorem normalizedErrorUtility_regret_eq_error_gap
    (err : Action → State → ℝ) (scale : ℝ)
    {aStar a : Action} {y : State}
    (h_exact : err aStar y = 0) :
    utilityRegret (normalizedErrorUtility err scale) aStar a y =
      err a y / scale := by
  unfold utilityRegret normalizedErrorUtility
  rw [h_exact]
  ring

/-- Zero regret coincides with zero error for normalized exact-state utilities. -/
theorem normalizedErrorUtility_zero_regret_iff_zero_error
    (err : Action → State → ℝ) (scale : ℝ)
    {aStar a : Action} {y : State}
    (hscale : 0 < scale)
    (h_exact : err aStar y = 0) :
    utilityRegret (normalizedErrorUtility err scale) aStar a y = 0
      ↔ err a y = 0 := by
  rw [normalizedErrorUtility_regret_eq_error_gap
    (err := err) (scale := scale) (aStar := aStar) (a := a) (y := y) h_exact]
  constructor
  · intro hzero
    have hdiv : err a y / scale = 0 := hzero
    have hs : scale ≠ 0 := by linarith
    exact (div_eq_zero_iff).mp hdiv |>.resolve_right hs
  · intro herr
    simp [herr]

end ExactUtilities

section ExactMergeableState

variable {Strings : Type*} [Monoid Strings]
variable {Sketch β : Type*}

/-- Bottom-up exact fold of a mergeable latent state over a tree. -/
def mergeFold
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch) :
    BinTree Strings → Sketch
  | BinTree.leaf b => encode b
  | BinTree.node TL TR => merge (mergeFold encode merge TL) (mergeFold encode merge TR)

/-- If a latent feature/state is exactly mergeable, tree folding recovers the
same state as computing the feature on the full span directly. -/
theorem mergeFold_eq_feature
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch)
    (feature : Strings → Sketch)
    (h_encode : ∀ x, encode x = feature x)
    (h_merge : ∀ x y, merge (feature x) (feature y) = feature (x * y))
    (T : BinTree Strings) :
    mergeFold encode merge T = feature (S T) := by
  induction T with
  | leaf b =>
      simpa [mergeFold] using h_encode b
  | node TL TR ihL ihR =>
      simp [mergeFold, S, ihL, ihR, h_merge]

/-- Any downstream utility on an exact mergeable latent state is preserved by
tree reduction exactly. This is the exact-control theorem used by the
simulation suite. -/
theorem mergeableStateUtility_exact_on_tree
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch)
    (feature : Strings → Sketch)
    (h_encode : ∀ x, encode x = feature x)
    (h_merge : ∀ x y, merge (feature x) (feature y) = feature (x * y))
    (u : Sketch → β)
    (T : BinTree Strings) :
    u (mergeFold encode merge T) = u (feature (S T)) := by
  rw [mergeFold_eq_feature
    (encode := encode) (merge := merge) (feature := feature) h_encode h_merge T]

end ExactMergeableState

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/ExactUtilityTransportInstances.lean (consolidated 2026-07-02) -/

/-!
# FormalProofs/OPT/ExactUtilityTransportInstances.lean

Concrete exact-state instances for the exact utility-transport suite.

These are the theorem-backed exact-control lanes behind the new simulations:

1. Markov changepoint exact state
2. Nonseparable complementarity counts
3. Topic mass / topic-plus-boundary exact sketches
-/

section

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

end

end
