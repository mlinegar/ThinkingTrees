import FormalProofs.OPT.TreeProperties

/-!
# FormalProofs/OPT/DiscountedTreeMetaObjective.lean

Discounted tree-level meta-objectives.

This file formalizes the weighting idea that appears in learned-optimizer
training: instead of supervising only the terminal loss, weight intermediate
losses along the optimization trajectory. In the tree setting we treat tree
depth as the analogue of trajectory time:

- depth `0` is the root / final aggregate,
- depth `1` is the first layer below the root,
- depth `d` receives weight `γ^d`.

The reinforcement-learning link is made explicit in two steps:

1. `discountedTrajectoryLoss` is the finite-horizon stage-cost objective;
2. `discountedReturn` is the usual discounted-return functional.

Minimizing discounted loss is equivalent to maximizing discounted return with
reward `r_t = -ℓ_t`. We then instantiate the same construction on a binary
tree by extracting the depth-indexed sequence of aggregated node losses.
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

section DiscountedTrajectory

/-- Finite-horizon discounted stage-cost objective. This is the optimization
counterpart of discounted return in reinforcement learning. -/
def discountedTrajectoryLoss (γ : ℝ) : List ℝ → ℝ
| [] => 0
| ℓ :: losses => ℓ + γ * discountedTrajectoryLoss γ losses

/-- Finite-horizon discounted return. -/
def discountedReturn (γ : ℝ) : List ℝ → ℝ
| [] => 0
| r :: rewards => r + γ * discountedReturn γ rewards

/-- Minimizing discounted loss is the same as maximizing discounted return with
reward equal to the negative loss. -/
theorem discountedTrajectoryLoss_eq_neg_discountedReturn_of_negated
    (γ : ℝ) (losses : List ℝ) :
    discountedTrajectoryLoss γ losses =
      - discountedReturn γ (losses.map fun ℓ => -ℓ) := by
  induction losses with
  | nil =>
      simp [discountedTrajectoryLoss, discountedReturn]
  | cons ℓ losses ih =>
      simp [discountedTrajectoryLoss, discountedReturn, ih]
      ring

/-- At `γ = 1`, discounted stage cost reduces to the ordinary finite sum. -/
theorem discountedTrajectoryLoss_one_eq_sum (losses : List ℝ) :
    discountedTrajectoryLoss 1 losses = losses.sum := by
  induction losses with
  | nil =>
      simp [discountedTrajectoryLoss]
  | cons ℓ losses ih =>
      simp [discountedTrajectoryLoss, ih]

end DiscountedTrajectory

section TreeInstantiation

variable {α : Type*}

/-- Elementwise list addition with zero padding. This lets us aggregate node
losses from the left and right subtrees into a single depth-indexed trajectory. -/
def combineLevelLosses : List ℝ → List ℝ → List ℝ
| [], ys => ys
| xs, [] => xs
| x :: xs, y :: ys => (x + y) :: combineLevelLosses xs ys

/-- The discounted trajectory loss of elementwise-combined sequences is the sum
of their discounted trajectory losses. -/
theorem discountedTrajectoryLoss_combineLevelLosses
    (γ : ℝ) (xs ys : List ℝ) :
    discountedTrajectoryLoss γ (combineLevelLosses xs ys) =
      discountedTrajectoryLoss γ xs + discountedTrajectoryLoss γ ys := by
  induction xs generalizing ys with
  | nil =>
      cases ys <;> simp [combineLevelLosses, discountedTrajectoryLoss]
  | cons x xs ih =>
      cases ys with
      | nil =>
          simp [combineLevelLosses, discountedTrajectoryLoss]
      | cons y ys =>
          simp [combineLevelLosses, discountedTrajectoryLoss, ih]
          ring

/-- Elementwise combination commutes with negation. -/
theorem combineLevelLosses_map_neg
    (xs ys : List ℝ) :
    combineLevelLosses (xs.map fun x => -x) (ys.map fun y => -y) =
      (combineLevelLosses xs ys).map (fun z => -z) := by
  induction xs generalizing ys with
  | nil =>
      cases ys <;> simp [combineLevelLosses]
  | cons x xs ih =>
      cases ys with
      | nil =>
          simp [combineLevelLosses]
      | cons y ys =>
          simp [combineLevelLosses, ih]
          ring

/-- Depth-indexed tree loss sequence. The `d`-th entry is the sum of `nodeLoss`
over all nodes at depth `d`, with the root at depth `0`. -/
def treeLevelLosses (nodeLoss : BinTree α → ℝ) (T : BinTree α) : List ℝ :=
  match T with
  | BinTree.leaf _ => [nodeLoss T]
  | BinTree.node TL TR =>
      nodeLoss T :: combineLevelLosses (treeLevelLosses nodeLoss TL) (treeLevelLosses nodeLoss TR)

/-- Discounted tree meta-loss: root has weight `1`, children have weight `γ`,
grandchildren have weight `γ^2`, and so on. -/
def discountedTreeMetaLoss
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (T : BinTree α) : ℝ :=
  discountedTrajectoryLoss γ (treeLevelLosses nodeLoss T)

/-- Tree-level discounted return built from node rewards instead of node losses. -/
def discountedTreeReturn
    (γ : ℝ) (nodeReward : BinTree α → ℝ) (T : BinTree α) : ℝ :=
  discountedReturn γ (treeLevelLosses nodeReward T)

/-- Total undiscounted node loss on a tree. -/
def totalNodeLoss (nodeLoss : BinTree α → ℝ) (T : BinTree α) : ℝ :=
  match T with
  | BinTree.leaf _ => nodeLoss T
  | BinTree.node TL TR =>
      nodeLoss T + totalNodeLoss nodeLoss TL + totalNodeLoss nodeLoss TR

@[simp] theorem treeLevelLosses_leaf
    (nodeLoss : BinTree α → ℝ) (a : α) :
    treeLevelLosses nodeLoss (BinTree.leaf a) = [nodeLoss (BinTree.leaf a)] := by
  simp [treeLevelLosses]

@[simp] theorem treeLevelLosses_node
    (nodeLoss : BinTree α → ℝ) (TL TR : BinTree α) :
    treeLevelLosses nodeLoss (BinTree.node TL TR) =
      nodeLoss (BinTree.node TL TR)
        :: combineLevelLosses (treeLevelLosses nodeLoss TL) (treeLevelLosses nodeLoss TR) := by
  simp [treeLevelLosses]

@[simp] theorem discountedTreeMetaLoss_leaf
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (a : α) :
    discountedTreeMetaLoss γ nodeLoss (BinTree.leaf a) = nodeLoss (BinTree.leaf a) := by
  simp [discountedTreeMetaLoss, discountedTrajectoryLoss]

/-- Recursive decomposition: the root gets weight `1`, and both child subtrees
are discounted by one factor of `γ`. -/
@[simp] theorem discountedTreeMetaLoss_node
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (TL TR : BinTree α) :
    discountedTreeMetaLoss γ nodeLoss (BinTree.node TL TR) =
      nodeLoss (BinTree.node TL TR)
        + γ * discountedTreeMetaLoss γ nodeLoss TL
        + γ * discountedTreeMetaLoss γ nodeLoss TR := by
  simp [discountedTreeMetaLoss, discountedTrajectoryLoss,
    discountedTrajectoryLoss_combineLevelLosses, mul_add, add_assoc]

@[simp] theorem discountedTreeReturn_leaf
    (γ : ℝ) (nodeReward : BinTree α → ℝ) (a : α) :
    discountedTreeReturn γ nodeReward (BinTree.leaf a) = nodeReward (BinTree.leaf a) := by
  simp [discountedTreeReturn, discountedReturn]

/-- Reinforcement-learning bridge on trees: minimizing discounted node losses is
equivalent to maximizing discounted node rewards with reward `-loss`. -/
theorem discountedTreeMetaLoss_eq_neg_discountedTreeReturn_of_negated
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (T : BinTree α) :
    discountedTreeMetaLoss γ nodeLoss T =
      - discountedTreeReturn γ (fun U => -nodeLoss U) T := by
  have hLevels :
      treeLevelLosses (fun U => -nodeLoss U) T =
        (treeLevelLosses nodeLoss T).map (fun ℓ => -ℓ) := by
    induction T with
    | leaf a =>
        simp [treeLevelLosses]
    | node TL TR ihL ihR =>
        simp [treeLevelLosses, ihL, ihR, combineLevelLosses_map_neg]
  unfold discountedTreeMetaLoss discountedTreeReturn
  rw [hLevels]
  simpa using
    discountedTrajectoryLoss_eq_neg_discountedReturn_of_negated γ
      (treeLevelLosses nodeLoss T)

/-- Endpoint `γ = 0`: only the root / final aggregate is supervised. -/
theorem discountedTreeMetaLoss_zero_eq_root
    (nodeLoss : BinTree α → ℝ) (T : BinTree α) :
    discountedTreeMetaLoss 0 nodeLoss T = nodeLoss T := by
  cases T <;> simp [discountedTreeMetaLoss, treeLevelLosses, discountedTrajectoryLoss]

@[simp] theorem totalNodeLoss_leaf
    (nodeLoss : BinTree α → ℝ) (a : α) :
    totalNodeLoss nodeLoss (BinTree.leaf a) = nodeLoss (BinTree.leaf a) := by
  simp [totalNodeLoss]

@[simp] theorem totalNodeLoss_node
    (nodeLoss : BinTree α → ℝ) (TL TR : BinTree α) :
    totalNodeLoss nodeLoss (BinTree.node TL TR) =
      nodeLoss (BinTree.node TL TR)
        + totalNodeLoss nodeLoss TL
        + totalNodeLoss nodeLoss TR := by
  simp [totalNodeLoss]

/-- Endpoint `γ = 1`: every tree level receives full weight, so we recover the
ordinary sum of node losses. -/
theorem discountedTreeMetaLoss_one_eq_totalNodeLoss
    (nodeLoss : BinTree α → ℝ) (T : BinTree α) :
    discountedTreeMetaLoss 1 nodeLoss T = totalNodeLoss nodeLoss T := by
  induction T with
  | leaf a =>
      simp [discountedTreeMetaLoss, totalNodeLoss, discountedTrajectoryLoss]
  | node TL TR ihL ihR =>
      rw [discountedTreeMetaLoss_node, totalNodeLoss_node, ihL, ihR]
      ring

/-- Nonnegativity propagates through the discounted tree objective whenever
`γ ≥ 0`. -/
theorem discountedTreeMetaLoss_nonneg
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (T : BinTree α)
    (hγ : 0 ≤ γ)
    (hNonneg : ∀ U : BinTree α, 0 ≤ nodeLoss U) :
    0 ≤ discountedTreeMetaLoss γ nodeLoss T := by
  induction T with
  | leaf a =>
      simp [discountedTreeMetaLoss, discountedTrajectoryLoss, hNonneg]
  | node TL TR ihL ihR =>
      have hroot : 0 ≤ nodeLoss (BinTree.node TL TR) := hNonneg _
      rw [discountedTreeMetaLoss_node]
      nlinarith

/-- For nonnegative node losses and `0 ≤ γ ≤ 1`, discounting never exceeds the
undiscounted total node loss. -/
theorem discountedTreeMetaLoss_le_totalNodeLoss
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (T : BinTree α)
    (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hNonneg : ∀ U : BinTree α, 0 ≤ nodeLoss U) :
    discountedTreeMetaLoss γ nodeLoss T ≤ totalNodeLoss nodeLoss T := by
  induction T with
  | leaf a =>
      simp [discountedTreeMetaLoss, totalNodeLoss, discountedTrajectoryLoss]
  | node TL TR ihL ihR =>
      have hIhSum :
          discountedTreeMetaLoss γ nodeLoss TL + discountedTreeMetaLoss γ nodeLoss TR ≤
            totalNodeLoss nodeLoss TL + totalNodeLoss nodeLoss TR :=
        add_le_add ihL ihR
      have hSubNonneg :
          0 ≤ discountedTreeMetaLoss γ nodeLoss TL + discountedTreeMetaLoss γ nodeLoss TR := by
        exact add_nonneg
          (discountedTreeMetaLoss_nonneg γ nodeLoss TL hγ0 hNonneg)
          (discountedTreeMetaLoss_nonneg γ nodeLoss TR hγ0 hNonneg)
      have hScale :
          γ * (discountedTreeMetaLoss γ nodeLoss TL + discountedTreeMetaLoss γ nodeLoss TR) ≤
            discountedTreeMetaLoss γ nodeLoss TL + discountedTreeMetaLoss γ nodeLoss TR := by
        have hTmp :=
          mul_le_mul_of_nonneg_right hγ1 hSubNonneg
        simpa using hTmp
      rw [discountedTreeMetaLoss_node, totalNodeLoss]
      nlinarith

/-- For nonnegative node losses and `γ ≥ 0`, discounting keeps at least the
root loss. -/
theorem rootLoss_le_discountedTreeMetaLoss
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (T : BinTree α)
    (hγ : 0 ≤ γ)
    (hNonneg : ∀ U : BinTree α, 0 ≤ nodeLoss U) :
    nodeLoss T ≤ discountedTreeMetaLoss γ nodeLoss T := by
  cases T with
  | leaf a =>
      simp [discountedTreeMetaLoss, discountedTrajectoryLoss]
  | node TL TR =>
      rw [discountedTreeMetaLoss_node]
      have hSubNonneg :
          0 ≤ discountedTreeMetaLoss γ nodeLoss TL + discountedTreeMetaLoss γ nodeLoss TR := by
        exact add_nonneg
          (discountedTreeMetaLoss_nonneg γ nodeLoss TL hγ hNonneg)
          (discountedTreeMetaLoss_nonneg γ nodeLoss TR hγ hNonneg)
      have hTail : 0 ≤ γ * (discountedTreeMetaLoss γ nodeLoss TL + discountedTreeMetaLoss γ nodeLoss TR) :=
        mul_nonneg hγ hSubNonneg
      linarith

/-- Combined bracket: for nonnegative node losses and `0 ≤ γ ≤ 1`, the
discounted tree meta-loss interpolates between the root loss and the full
undiscounted sum of node losses. -/
theorem discountedTreeMetaLoss_sandwich
    (γ : ℝ) (nodeLoss : BinTree α → ℝ) (T : BinTree α)
    (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hNonneg : ∀ U : BinTree α, 0 ≤ nodeLoss U) :
    nodeLoss T ≤ discountedTreeMetaLoss γ nodeLoss T
      ∧ discountedTreeMetaLoss γ nodeLoss T ≤ totalNodeLoss nodeLoss T := by
  exact ⟨rootLoss_le_discountedTreeMetaLoss γ nodeLoss T hγ0 hNonneg,
    discountedTreeMetaLoss_le_totalNodeLoss γ nodeLoss T hγ0 hγ1 hNonneg⟩

end TreeInstantiation

end FormalProofs.OPT
