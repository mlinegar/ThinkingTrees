import FormalProofs.ML.Core
import FormalProofs.ML.ERM

/-!
# FormalProofs/ML/DecisionTree.lean

Decision trees as a hypothesis class.

This is a minimal, algorithm-agnostic formalization:
- tree structure (leaf / split)
- evaluation
- tree operations (map, precompose)
- structural metrics (depth, leaves, nodes)
- paths and routing
- depth-bounded hypothesis classes (including stumps)
- impurity measures and split objectives
- measurability of evaluation
- cost-complexity pruning and depth selection
- axis-aligned splits for real features
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ML

/-!
## Tree Structure
-/

/-- A binary decision tree with boolean splits. -/
inductive DecisionTree (X Y : Type*) : Type _
| leaf : Y → DecisionTree X Y
| node : (X → Bool) → DecisionTree X Y → DecisionTree X Y → DecisionTree X Y

/-- Evaluate a decision tree on an input. -/
def DecisionTree.eval {X Y : Type*} : DecisionTree X Y → X → Y
| DecisionTree.leaf y, _ => y
| DecisionTree.node s tL tR, x => if s x then DecisionTree.eval tL x else DecisionTree.eval tR x

/-!
## Tree Operations
-/

/-- Map leaf values through a function. -/
def DecisionTree.map {X Y Z : Type*} (f : Y → Z) : DecisionTree X Y → DecisionTree X Z
| DecisionTree.leaf y => DecisionTree.leaf (f y)
| DecisionTree.node s tL tR => DecisionTree.node s (DecisionTree.map f tL) (DecisionTree.map f tR)

lemma DecisionTree.eval_map {X Y Z : Type*} (f : Y → Z) (t : DecisionTree X Y) (x : X) :
    DecisionTree.eval (DecisionTree.map f t) x = f (DecisionTree.eval t x) := by
  induction t with
  | leaf y =>
      simp [DecisionTree.map, DecisionTree.eval]
  | node s tL tR ihL ihR =>
      by_cases h : s x
      · simp [DecisionTree.map, DecisionTree.eval, h, ihL, ihR]
      · simp [DecisionTree.map, DecisionTree.eval, h, ihL, ihR]

/-- Precompose inputs: build a tree on `Z` by mapping `Z → X` into the splits. -/
def DecisionTree.precompose {X Y Z : Type*} (g : Z → X) : DecisionTree X Y → DecisionTree Z Y
| DecisionTree.leaf y => DecisionTree.leaf y
| DecisionTree.node s tL tR =>
    DecisionTree.node (fun z => s (g z)) (DecisionTree.precompose g tL) (DecisionTree.precompose g tR)

lemma DecisionTree.eval_precompose {X Y Z : Type*} (g : Z → X) (t : DecisionTree X Y) (z : Z) :
    DecisionTree.eval (DecisionTree.precompose g t) z = DecisionTree.eval t (g z) := by
  induction t <;> simp [DecisionTree.precompose, DecisionTree.eval, *]

/-!
## Structural Metrics
-/

/-- Tree depth (leaf has depth 0). -/
def DecisionTree.depth {X Y : Type*} : DecisionTree X Y → Nat
| DecisionTree.leaf _ => 0
| DecisionTree.node _ tL tR => Nat.succ (Nat.max (DecisionTree.depth tL) (DecisionTree.depth tR))

/-- Number of leaves in the tree. -/
def DecisionTree.numLeaves {X Y : Type*} : DecisionTree X Y → Nat
| DecisionTree.leaf _ => 1
| DecisionTree.node _ tL tR => DecisionTree.numLeaves tL + DecisionTree.numLeaves tR

/-- Total number of nodes (internal + leaves). -/
def DecisionTree.numNodes {X Y : Type*} : DecisionTree X Y → Nat
| DecisionTree.leaf _ => 1
| DecisionTree.node _ tL tR => 1 + DecisionTree.numNodes tL + DecisionTree.numNodes tR

/-- Number of internal (split) nodes. -/
def DecisionTree.numInternalNodes {X Y : Type*} : DecisionTree X Y → Nat
| DecisionTree.leaf _ => 0
| DecisionTree.node _ tL tR => 1 + DecisionTree.numInternalNodes tL + DecisionTree.numInternalNodes tR

lemma DecisionTree.numLeaves_pos {X Y : Type*} (t : DecisionTree X Y) :
    0 < DecisionTree.numLeaves t := by
  induction t with
  | leaf y =>
      simp [DecisionTree.numLeaves]
  | node s tL tR ihL ihR =>
      have hL : 1 ≤ DecisionTree.numLeaves tL := Nat.succ_le_iff.2 ihL
      have hL' : DecisionTree.numLeaves tL ≤
          DecisionTree.numLeaves tL + DecisionTree.numLeaves tR := by
        exact Nat.le_add_right _ _
      have h1 : 1 ≤ DecisionTree.numLeaves tL + DecisionTree.numLeaves tR :=
        le_trans hL hL'
      exact (Nat.succ_le_iff.mp h1)

lemma DecisionTree.numNodes_eq_internal_plus_leaves {X Y : Type*} (t : DecisionTree X Y) :
    DecisionTree.numNodes t =
      DecisionTree.numInternalNodes t + DecisionTree.numLeaves t := by
  induction t with
  | leaf y =>
      simp [DecisionTree.numNodes, DecisionTree.numInternalNodes, DecisionTree.numLeaves]
  | node s tL tR ihL ihR =>
      simp [DecisionTree.numNodes, DecisionTree.numInternalNodes, DecisionTree.numLeaves, ihL, ihR,
        Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

/-!
## Paths and Routing
-/

/-- Path through a binary tree, encoded as left/right decisions (`true` = left). -/
abbrev DecisionTree.Path := List Bool

/-- Follow a path; returns `none` if the path runs past a leaf. -/
def DecisionTree.follow? {X Y : Type*} : DecisionTree X Y → DecisionTree.Path → Option (DecisionTree X Y)
| t, [] => some t
| DecisionTree.leaf _, _::_ => none
| DecisionTree.node _ tL tR, b::p =>
    if b then DecisionTree.follow? tL p else DecisionTree.follow? tR p

/-- Leaf value at a path, if the path is valid. -/
def DecisionTree.leafValue? {X Y : Type*} (t : DecisionTree X Y) (p : DecisionTree.Path) : Option Y :=
  match DecisionTree.follow? t p with
  | some (DecisionTree.leaf y) => some y
  | _ => none

/-- A path is valid if it ends at a leaf. -/
def DecisionTree.ValidPath {X Y : Type*} (t : DecisionTree X Y) (p : DecisionTree.Path) : Prop :=
  ∃ y, DecisionTree.leafValue? t p = some y

/-- Set of valid paths. -/
def DecisionTree.pathSet {X Y : Type*} (t : DecisionTree X Y) : Set DecisionTree.Path :=
  {p | DecisionTree.ValidPath t p}

/-- Route an input through the tree, returning the induced path. -/
def DecisionTree.route {X Y : Type*} : DecisionTree X Y → X → DecisionTree.Path
| DecisionTree.leaf _, _ => []
| DecisionTree.node s tL tR, x =>
    if s x then true :: DecisionTree.route tL x else false :: DecisionTree.route tR x

lemma DecisionTree.follow?_route {X Y : Type*} (t : DecisionTree X Y) (x : X) :
    DecisionTree.follow? t (DecisionTree.route t x) =
      some (DecisionTree.leaf (DecisionTree.eval t x)) := by
  induction t with
  | leaf y =>
      simp [DecisionTree.route, DecisionTree.follow?, DecisionTree.eval]
  | node s tL tR ihL ihR =>
      by_cases h : s x
      · simp [DecisionTree.route, DecisionTree.follow?, DecisionTree.eval, h, ihL]
      · simp [DecisionTree.route, DecisionTree.follow?, DecisionTree.eval, h, ihR]

lemma DecisionTree.leafValue?_route {X Y : Type*} (t : DecisionTree X Y) (x : X) :
    DecisionTree.leafValue? t (DecisionTree.route t x) =
      some (DecisionTree.eval t x) := by
  have h := DecisionTree.follow?_route (t := t) (x := x)
  simp [DecisionTree.leafValue?, h]

/-- Region of inputs routed to a given path. -/
def DecisionTree.pathRegion {X Y : Type*} (t : DecisionTree X Y) (p : DecisionTree.Path) : Set X :=
  {x | DecisionTree.route t x = p}

/-- Region of inputs mapped to a given leaf value. -/
def DecisionTree.leafRegion {X Y : Type*} (t : DecisionTree X Y) (y : Y) : Set X :=
  {x | DecisionTree.eval t x = y}

/-!
## Hypothesis Class View
-/

/-- The hypothesis induced by a decision tree. -/
def treeHypothesis {X Y : Type*} (t : DecisionTree X Y) : Hypothesis X Y :=
  DecisionTree.eval t

/-- Depth-bounded tree class. -/
def DepthBounded {X Y : Type*} (d : Nat) : Set (DecisionTree X Y) :=
  {t | DecisionTree.depth t ≤ d}

/-- Depth-bounded hypothesis class (trees viewed as predictors). -/
def TreeHypothesisClass {X Y : Type*} (d : Nat) : Set (Hypothesis X Y) :=
  {h | ∃ t : DecisionTree X Y, DecisionTree.depth t ≤ d ∧ h = treeHypothesis t}

/-- Decision stumps (depth at most 1). -/
def DecisionStump {X Y : Type*} : Set (DecisionTree X Y) :=
  DepthBounded 1

/-- Stump hypothesis class. -/
def StumpHypothesisClass {X Y : Type*} : Set (Hypothesis X Y) :=
  TreeHypothesisClass 1

/-!
## Impurity and Split Objectives
-/

/-- Weight of a node defined by membership weights. -/
def nodeWeight {n : ℕ} (w : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, w i

/-- Weighted class mass in a node. -/
def classWeight {X Y : Type*} {n : ℕ} [DecidableEq Y]
    (data : Dataset X Y n) (w : Fin n → ℝ) (y : Y) : ℝ :=
  ∑ i : Fin n, if (data i).2 = y then w i else 0

/-- Weighted class probability in a node. -/
def classProb {X Y : Type*} {n : ℕ} [DecidableEq Y]
    (data : Dataset X Y n) (w : Fin n → ℝ) (y : Y) : ℝ :=
  classWeight data w y / nodeWeight w

/-- Gini impurity of a node (classification). -/
def giniImpurity {X Y : Type*} {n : ℕ} [Fintype Y] [DecidableEq Y]
    (data : Dataset X Y n) (w : Fin n → ℝ) : ℝ :=
  1 - ∑ y : Y, (classProb data w y) ^ 2

/-- Entropy impurity of a node (classification). -/
def entropyImpurity {X Y : Type*} {n : ℕ} [Fintype Y] [DecidableEq Y]
    (data : Dataset X Y n) (w : Fin n → ℝ) : ℝ :=
  -∑ y : Y, (classProb data w y) * Real.log (classProb data w y)

/-- Weighted mean of labels in a node (regression). -/
def nodeMean {X : Type*} {n : ℕ} (data : Dataset X ℝ n) (w : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, w i * (data i).2) / nodeWeight w

/-- Weighted squared error impurity (sum of squared residuals). -/
def squaredErrorImpurity {X : Type*} {n : ℕ} (data : Dataset X ℝ n) (w : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, w i * ((data i).2 - nodeMean data w) ^ 2

/-- Left weights induced by a split on a node. -/
def leftWeights {X Y : Type*} {n : ℕ} (data : Dataset X Y n) (w : Fin n → ℝ)
    (s : X → Bool) : Fin n → ℝ :=
  fun i => if s (data i).1 then w i else 0

/-- Right weights induced by a split on a node. -/
def rightWeights {X Y : Type*} {n : ℕ} (data : Dataset X Y n) (w : Fin n → ℝ)
    (s : X → Bool) : Fin n → ℝ :=
  fun i => if s (data i).1 then 0 else w i

/-- Weighted split objective for a given node impurity. -/
def splitObjective {X Y : Type*} {n : ℕ}
    (impurity : Dataset X Y n → (Fin n → ℝ) → ℝ)
    (data : Dataset X Y n) (w : Fin n → ℝ) (s : X → Bool) : ℝ :=
  let wL := leftWeights data w s
  let wR := rightWeights data w s
  (nodeWeight wL / nodeWeight w) * impurity data wL +
    (nodeWeight wR / nodeWeight w) * impurity data wR

/-- Impurity reduction (information gain) from a split. -/
def informationGain {X Y : Type*} {n : ℕ}
    (impurity : Dataset X Y n → (Fin n → ℝ) → ℝ)
    (data : Dataset X Y n) (w : Fin n → ℝ) (s : X → Bool) : ℝ :=
  impurity data w - splitObjective impurity data w s

/-- Split optimality predicate over a candidate class. -/
def IsBestSplit {X Y : Type*} {n : ℕ}
    (S : Set (X → Bool))
    (impurity : Dataset X Y n → (Fin n → ℝ) → ℝ)
    (data : Dataset X Y n) (w : Fin n → ℝ) (s : X → Bool) : Prop :=
  s ∈ S ∧ ∀ s' ∈ S, splitObjective impurity data w s ≤ splitObjective impurity data w s'

/-!
## Measurability
-/

/-- Predicate that all splits in a tree are measurable. -/
def DecisionTree.MeasurableSplits {X Y : Type*} [MeasurableSpace X] :
    DecisionTree X Y → Prop
| DecisionTree.leaf _ => True
| DecisionTree.node s tL tR =>
    MeasurableSet {x : X | s x = true} ∧
      DecisionTree.MeasurableSplits tL ∧ DecisionTree.MeasurableSplits tR

lemma measurable_ite_bool {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    {s : X → Bool} (hs : MeasurableSet {x : X | s x = true})
    {f g : X → Y} (hf : Measurable f) (hg : Measurable g) :
    Measurable (fun x => if s x then f x else g x) := by
  have h : Measurable (fun x => ite (s x = true) (f x) (g x)) :=
    Measurable.ite hs hf hg
  have h_eq :
      (fun x => if s x then f x else g x) =
        (fun x => ite (s x = true) (f x) (g x)) := by
    funext x
    cases hxs : s x <;> simp [hxs]
  simpa [h_eq] using h

/-- Measurability of evaluation under measurable splits. -/
lemma DecisionTree.measurable_eval {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    (t : DecisionTree X Y) (ht : DecisionTree.MeasurableSplits t) :
    Measurable (DecisionTree.eval t) := by
  induction t with
  | leaf y =>
      simp [DecisionTree.eval]
  | node s tL tR ihL ihR =>
      rcases ht with ⟨hs, hL, hR⟩
      have hL' := ihL hL
      have hR' := ihR hR
      have h :=
        measurable_ite_bool (s := s) hs (f := DecisionTree.eval tL) (g := DecisionTree.eval tR) hL' hR'
      simpa [DecisionTree.eval] using h

/-!
## Empirical Risk and Cost-Complexity Pruning
-/

/-- Empirical risk of a decision tree. -/
def treeEmpiricalRisk {X Y : Type*} {n : ℕ}
    (loss : Loss Y) (data : Dataset X Y n) (t : DecisionTree X Y) : ℝ :=
  empiricalRisk loss data (treeHypothesis t)

/-- Cost-complexity objective (CART): empirical risk + alpha * number of leaves. -/
def costComplexityObjective {X Y : Type*} {n : ℕ}
    (alpha : ℝ) (loss : Loss Y) (data : Dataset X Y n) (t : DecisionTree X Y) : ℝ :=
  treeEmpiricalRisk loss data t + alpha * DecisionTree.numLeaves t

/-- Cost-complexity minimizer within a tree class. -/
def IsCostComplexityMinimizer {X Y : Type*} {n : ℕ}
    (T : Set (DecisionTree X Y)) (alpha : ℝ) (loss : Loss Y) (data : Dataset X Y n)
    (t : DecisionTree X Y) : Prop :=
  t ∈ T ∧ ∀ t' ∈ T, costComplexityObjective alpha loss data t ≤
    costComplexityObjective alpha loss data t'

/-- Structural pruning relation: replace subtrees by leaves. -/
inductive DecisionTree.PrunesTo {X Y : Type*} : DecisionTree X Y → DecisionTree X Y → Prop
| refl (t : DecisionTree X Y) : DecisionTree.PrunesTo t t
| node {s : X → Bool} {tL tR tL' tR' : DecisionTree X Y}
    (hL : DecisionTree.PrunesTo tL tL') (hR : DecisionTree.PrunesTo tR tR') :
    DecisionTree.PrunesTo (DecisionTree.node s tL tR) (DecisionTree.node s tL' tR')
| collapse {s : X → Bool} {tL tR : DecisionTree X Y} (y : Y) :
    DecisionTree.PrunesTo (DecisionTree.node s tL tR) (DecisionTree.leaf y)

/-- The pruning family of a tree: all prunings reachable by `PrunesTo`. -/
def DecisionTree.PruningFamily {X Y : Type*} (t : DecisionTree X Y) : Set (DecisionTree X Y) :=
  {t' | DecisionTree.PrunesTo t t'}

/-- Best pruned subtree under cost-complexity objective. -/
def IsCostComplexityPruning {X Y : Type*} {n : ℕ}
    (alpha : ℝ) (loss : Loss Y) (data : Dataset X Y n)
    (t : DecisionTree X Y) (t' : DecisionTree X Y) : Prop :=
  t' ∈ DecisionTree.PruningFamily t ∧
    ∀ t'' ∈ DecisionTree.PruningFamily t,
      costComplexityObjective alpha loss data t' ≤ costComplexityObjective alpha loss data t''

/-- ERM within a fixed depth class. -/
def DepthERM {X Y : Type*} {n : ℕ}
    (d : Nat) (loss : Loss Y) (data : Dataset X Y n) (t : DecisionTree X Y) : Prop :=
  t ∈ DepthBounded d ∧
    ∀ t' ∈ DepthBounded d, treeEmpiricalRisk loss data t ≤ treeEmpiricalRisk loss data t'

/-- Depth-penalized objective for structural risk minimization. -/
def depthPenalizedObjective {X Y : Type*} {n : ℕ}
    (penalty : Nat → ℝ) (loss : Loss Y) (data : Dataset X Y n) (t : DecisionTree X Y) : ℝ :=
  treeEmpiricalRisk loss data t + penalty (DecisionTree.depth t)

/-- Selected depth is any depth whose ERM tree minimizes the penalized objective across depths. -/
def IsSelectedDepth {X Y : Type*} {n : ℕ}
    (penalty : Nat → ℝ) (loss : Loss Y) (data : Dataset X Y n) (d : Nat) : Prop :=
  ∃ t, DepthERM d loss data t ∧
    ∀ d' t', DepthERM d' loss data t' →
      depthPenalizedObjective penalty loss data t ≤ depthPenalizedObjective penalty loss data t'

/-!
## Axis-Aligned Splits (Regression Trees)
-/

/-- Axis-aligned split for feature vectors `Fin d → ℝ`. -/
structure AxisSplit (d : Nat) where
  index : Fin d
  threshold : ℝ

/-- Evaluate an axis-aligned split as a boolean decision. -/
def AxisSplit.apply {d : Nat} (s : AxisSplit d) (x : Fin d → ℝ) : Bool :=
  decide (x s.index ≤ s.threshold)

/-- Split objective specialized to axis-aligned splits. -/
def AxisSplit.objective {d : Nat} {Y : Type*} {n : ℕ}
    (impurity : Dataset (Fin d → ℝ) Y n → (Fin n → ℝ) → ℝ)
    (data : Dataset (Fin d → ℝ) Y n) (w : Fin n → ℝ) (s : AxisSplit d) : ℝ :=
  splitObjective impurity data w s.apply

/-- Optimality predicate for axis-aligned splits. -/
def AxisSplit.IsBest {d : Nat} {Y : Type*} {n : ℕ}
    (S : Set (AxisSplit d))
    (impurity : Dataset (Fin d → ℝ) Y n → (Fin n → ℝ) → ℝ)
    (data : Dataset (Fin d → ℝ) Y n) (w : Fin n → ℝ) (s : AxisSplit d) : Prop :=
  s ∈ S ∧ ∀ s' ∈ S, AxisSplit.objective impurity data w s ≤ AxisSplit.objective impurity data w s'

/-- Left region of an axis-aligned split (feature is below threshold). -/
def AxisSplit.leftRegion {d : Nat} (s : AxisSplit d) : Set (Fin d → ℝ) :=
  {x | x s.index ≤ s.threshold}

/-- Right region of an axis-aligned split (feature is above threshold). -/
def AxisSplit.rightRegion {d : Nat} (s : AxisSplit d) : Set (Fin d → ℝ) :=
  {x | s.threshold < x s.index}

lemma AxisSplit.apply_eq_true {d : Nat} (s : AxisSplit d) (x : Fin d → ℝ) :
    s.apply x = true ↔ x s.index ≤ s.threshold := by
  simp [AxisSplit.apply]

/-- Axis-aligned decision tree on real feature vectors. -/
abbrev AxisAlignedTree (d : Nat) (Y : Type*) := DecisionTree (Fin d → ℝ) Y

/-- Regression tree (real-valued output). -/
abbrev RegressionTree (X : Type*) := DecisionTree X ℝ

end ML
