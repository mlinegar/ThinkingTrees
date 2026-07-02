import Mathlib
import FormalProofs.DSL.NonclassicalExpectationMismatch
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.OPT.TreeProperties
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.OptimizationPerturbation
import FormalProofs.OPT.MergeableProjection

/-!
# FormalProofs/OPT/LocalLawObjectives.lean

Consolidated objective/adjustment ladder (2026-07-02).

Verbatim merge, in intra-cluster dependency order, of the former modules:

1.  `FormalProofs/OPT/RootLocalObjective.lean`
2.  `FormalProofs/OPT/ProxyOracleGap.lean`
3.  `FormalProofs/OPT/DiscountedTreeMetaObjective.lean`
4.  `FormalProofs/OPT/NodeLocalLawAggregate.lean`
5.  `FormalProofs/OPT/NodeAIPWLocalLawAdjustment.lean`
6.  `FormalProofs/OPT/UnifiedLocalLawAdjustment.lean`
7.  `FormalProofs/OPT/DoublyRobustLocalLawAdjustment.lean`
8.  `FormalProofs/OPT/CoverageNormalizedObjective.lean`
9.  `FormalProofs/OPT/DiscountedIPWObjective.lean`
10. `FormalProofs/OPT/DoublyRobustMinimizationObjective.lean` (ACTIVE surface)

Each original module's docstring is preserved inside its section below. The
terminal section, from `DoublyRobustMinimizationObjective`, names the ACTIVE
optimization surface (it matches the Python ObjectiveSpec v1 default).
-/

/-! ## From FormalProofs/OPT/RootLocalObjective.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/RootLocalObjective.lean

Nominal root/local objective.

The paper-facing objective uses a fixed analyst-chosen local-law share
`Lambda`. Oracle disagreement is handled inside the local-law loss supplied to
this objective, not by changing `Lambda`.
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

/-- Nominal root/local objective with the user-facing local-law share
`Lambda`: `(1 - Lambda) * rootLoss + Lambda * lawLoss`. -/
def nominalRootLocalObjective
    (Lambda rootLoss lawLoss : ℝ) : ℝ :=
  (1 - Lambda) * rootLoss + Lambda * lawLoss

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/ProxyOracleGap.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/ProxyOracleGap.lean

Proxy/oracle gap facts used by the root and local-law routes.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Root : Type*}
variable {Y : Type*} [BoundedMetricSpace Y]

/-- Root-pair gap induced by evaluating a pairwise root comparison through
`fhat` instead of the target oracle `fstar`. -/
def rootPairBias
    (fstar fhat : Root → Y) (x y : Root) : ℝ :=
  dist (fhat x) (fhat y) - dist (fstar x) (fstar y)

/-- Uniform oracle approximation is symmetric after swapping `fstar` and
`fhat`, because metric distance is symmetric. -/
theorem uniformOracleApproximation_symm
    [Monoid Root]
    {fstar fhat : Root → Y} {eps : ℝ≥0}
    (hApprox : UniformOracleApproximation fstar fhat eps) :
    UniformOracleApproximation fhat fstar eps := by
  intro x
  simpa [dist_comm] using hApprox x

/-- A root-pair distance measured through `fhat` differs from the corresponding
true-oracle root-pair distance by at most the two-sided oracle-recovery slack. -/
theorem rootPairBias_abs_le_oracleRecoverySlack
    [Monoid Root]
    {fstar fhat : Root → Y} {eps : ℝ≥0}
    (hApprox : UniformOracleApproximation fstar fhat eps)
    (x y : Root) :
    |rootPairBias fstar fhat x y| ≤ OracleRecoverySlack eps := by
  have hTrueLe :
      dist (fstar x) (fstar y) ≤
        dist (fhat x) (fhat y) + 2 * (eps : ℝ) :=
    trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
      (hApprox := hApprox) (x := x) (x' := y)
  have hSurLe :
      dist (fhat x) (fhat y) ≤
        dist (fstar x) (fstar y) + 2 * (eps : ℝ) :=
    trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
      (fstar := fhat) (fhat := fstar)
      (hApprox := uniformOracleApproximation_symm (Root := Root) hApprox)
      (x := x) (x' := y)
  rw [abs_le]
  constructor
  · simp [rootPairBias, OracleRecoverySlack]
    linarith
  · simp [rootPairBias, OracleRecoverySlack]
    linarith

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/DiscountedTreeMetaObjective.lean (consolidated 2026-07-02) -/

section

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

end

end

/-! ## From FormalProofs/OPT/NodeLocalLawAggregate.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/NodeLocalLawAggregate.lean

Depth-discounted node local-law aggregation.
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

variable {Strings : Type*}

/-- Discounted aggregate of node-level local-law losses measured through
`fhat`. The aggregation convention is inherited from `discountedTreeMetaLoss`. -/
def discountedSurrogateNodeLawLoss
    (gammaDepth : ℝ)
    (nodeSurrogateLoss : BinTree Strings → ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth nodeSurrogateLoss T

/-- Discounted aggregate of the corresponding node-level local-law losses that
would be measured through `fstar`. -/
def discountedTrueNodeLawLoss
    (gammaDepth : ℝ)
    (nodeTrueLoss : BinTree Strings → ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth nodeTrueLoss T

/-- Discounted aggregate node-bias envelope. -/
def discountedNodeBiasBound
    (gammaDepth : ℝ)
    (nodeBiasBound : BinTree Strings → ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth nodeBiasBound T

theorem discountedNodeBiasBound_nonneg
    (gammaDepth : ℝ)
    (nodeBiasBound : BinTree Strings → ℝ)
    (T : BinTree Strings)
    (hGamma : 0 ≤ gammaDepth)
    (hBiasNonneg : ∀ U : BinTree Strings, 0 ≤ nodeBiasBound U) :
    0 ≤ discountedNodeBiasBound gammaDepth nodeBiasBound T := by
  exact discountedTreeMetaLoss_nonneg gammaDepth nodeBiasBound T hGamma hBiasNonneg

/-- If every node's surrogate-vs-true law loss is bounded by a nodewise bias
envelope, then the discounted aggregate surrogate-vs-true law loss is bounded by
the discounted aggregate envelope. -/
theorem discountedNodeLawLoss_abs_sub_le_biasBound
    (gammaDepth : ℝ)
    (nodeSurrogateLoss nodeTrueLoss nodeBiasBound : BinTree Strings → ℝ)
    (T : BinTree Strings)
    (hGamma : 0 ≤ gammaDepth)
    (hNode :
      ∀ U : BinTree Strings,
        |nodeSurrogateLoss U - nodeTrueLoss U| ≤ nodeBiasBound U)
    (hBiasNonneg : ∀ U : BinTree Strings, 0 ≤ nodeBiasBound U) :
    |discountedSurrogateNodeLawLoss gammaDepth nodeSurrogateLoss T -
      discountedTrueNodeLawLoss gammaDepth nodeTrueLoss T| ≤
      discountedNodeBiasBound gammaDepth nodeBiasBound T := by
  induction T with
  | leaf a =>
      simpa [discountedSurrogateNodeLawLoss, discountedTrueNodeLawLoss,
        discountedNodeBiasBound] using hNode (BinTree.leaf a)
  | node TL TR ihL ihR =>
      let root : BinTree Strings := BinTree.node TL TR
      let sRoot := nodeSurrogateLoss root
      let tRoot := nodeTrueLoss root
      let sL := discountedTreeMetaLoss gammaDepth nodeSurrogateLoss TL
      let tL := discountedTreeMetaLoss gammaDepth nodeTrueLoss TL
      let sR := discountedTreeMetaLoss gammaDepth nodeSurrogateLoss TR
      let tR := discountedTreeMetaLoss gammaDepth nodeTrueLoss TR
      let bRoot := nodeBiasBound root
      let bL := discountedTreeMetaLoss gammaDepth nodeBiasBound TL
      let bR := discountedTreeMetaLoss gammaDepth nodeBiasBound TR
      have hRoot : |sRoot - tRoot| ≤ bRoot := hNode root
      have hL : |sL - tL| ≤ bL := by
        simpa [sL, tL, bL, discountedSurrogateNodeLawLoss,
          discountedTrueNodeLawLoss, discountedNodeBiasBound] using ihL
      have hR : |sR - tR| ≤ bR := by
        simpa [sR, tR, bR, discountedSurrogateNodeLawLoss,
          discountedTrueNodeLawLoss, discountedNodeBiasBound] using ihR
      have hGL : |gammaDepth * (sL - tL)| ≤ gammaDepth * bL := by
        have hmul := mul_le_mul_of_nonneg_left hL hGamma
        simpa [abs_mul, abs_of_nonneg hGamma] using hmul
      have hGR : |gammaDepth * (sR - tR)| ≤ gammaDepth * bR := by
        have hmul := mul_le_mul_of_nonneg_left hR hGamma
        simpa [abs_mul, abs_of_nonneg hGamma] using hmul
      have hTriangle :
          |(sRoot - tRoot) + gammaDepth * (sL - tL) +
              gammaDepth * (sR - tR)| ≤
            |sRoot - tRoot| + |gammaDepth * (sL - tL)| +
              |gammaDepth * (sR - tR)| := by
        calc
          |(sRoot - tRoot) + gammaDepth * (sL - tL) +
              gammaDepth * (sR - tR)|
              ≤ |(sRoot - tRoot) + gammaDepth * (sL - tL)| +
                  |gammaDepth * (sR - tR)| := by
                    exact abs_add_le _ _
          _ ≤ |sRoot - tRoot| + |gammaDepth * (sL - tL)| +
                  |gammaDepth * (sR - tR)| := by
                    have h := abs_add_le (sRoot - tRoot) (gammaDepth * (sL - tL))
                    linarith
      have hDecomp :
          (sRoot + gammaDepth * sL + gammaDepth * sR) -
            (tRoot + gammaDepth * tL + gammaDepth * tR) =
          (sRoot - tRoot) + gammaDepth * (sL - tL) +
            gammaDepth * (sR - tR) := by
        ring
      simp [discountedSurrogateNodeLawLoss, discountedTrueNodeLawLoss,
        discountedNodeBiasBound, discountedTreeMetaLoss_node]
      calc
        |(sRoot + gammaDepth * sL + gammaDepth * sR) -
            (tRoot + gammaDepth * tL + gammaDepth * tR)|
            = |(sRoot - tRoot) + gammaDepth * (sL - tL) +
                gammaDepth * (sR - tR)| := by rw [hDecomp]
        _ ≤ |sRoot - tRoot| + |gammaDepth * (sL - tL)| +
              |gammaDepth * (sR - tR)| := hTriangle
        _ ≤ bRoot + gammaDepth * bL + gammaDepth * bR := by
              linarith

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/NodeAIPWLocalLawAdjustment.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/NodeAIPWLocalLawAdjustment.lean

Proxy plus node-oracle AIPW node-law aggregation.

Every node has a proxy local-law loss measured through `fhat`; sampled nodes
also expose the corresponding oracle local-law loss measured through `fstar`.
The node oracle channel is used as an inverse-propensity residual correction to
the proxy channel, so both channels target one true discounted node-law
estimand rather than two separate local objectives.
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

variable {Strings : Type*}

/-! ## Pointwise AIPW node law -/

/-- Per-node proxy-plus-oracle adjusted law loss. This is the DSL adjusted
outcome with `Y_pred = proxy` and `Y_true = oracle`:
`proxy + R / pi * (oracle - proxy)`. -/
def nodeAIPWAdjustedLawLoss
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  DSL.designAdjustedOutcome
    (nodeProxyLoss T) (nodeOracleLoss T) (nodeObserved T) (nodePi T)

theorem nodeAIPWAdjustedLawLoss_eq_proxy_of_unsampled
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : nodeObserved T = false) :
    nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nodeProxyLoss T := by
  unfold nodeAIPWAdjustedLawLoss
  rw [hObserved]
  exact DSL.designAdjustedOutcome_unsampled
    (nodeProxyLoss T) (nodeOracleLoss T) (nodePi T)

theorem nodeAIPWAdjustedLawLoss_eq_oracle_of_sampled_pi_one
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : nodeObserved T = true)
    (hPi : nodePi T = 1) :
    nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nodeOracleLoss T := by
  unfold nodeAIPWAdjustedLawLoss
  rw [hObserved, hPi]
  exact DSL.designAdjustedOutcome_full_sample
    (nodeProxyLoss T) (nodeOracleLoss T)

/-! ## Discounted AIPW node law -/

/-- Discounted aggregate of AIPW-adjusted node local-law losses. -/
def discountedAIPWNodeLawLoss
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi) T

private theorem discountedTreeMetaLoss_congr_all
    {α : Type*}
    (gammaDepth : ℝ)
    (nodeLoss1 nodeLoss2 : BinTree α -> ℝ)
    (T : BinTree α)
    (h : ∀ U : BinTree α, nodeLoss1 U = nodeLoss2 U) :
    discountedTreeMetaLoss gammaDepth nodeLoss1 T =
      discountedTreeMetaLoss gammaDepth nodeLoss2 T := by
  induction T with
  | leaf a =>
      simpa using h (BinTree.leaf a)
  | node TL TR ihL ihR =>
      rw [discountedTreeMetaLoss_node, discountedTreeMetaLoss_node,
        h (BinTree.node TL TR), ihL, ihR]

theorem discountedAIPWNodeLawLoss_eq_proxy_of_all_unsampled
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = false) :
    discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedSurrogateNodeLawLoss gammaDepth nodeProxyLoss T := by
  unfold discountedAIPWNodeLawLoss discountedSurrogateNodeLawLoss
  exact discountedTreeMetaLoss_congr_all gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
    nodeProxyLoss T
    (fun U => nodeAIPWAdjustedLawLoss_eq_proxy_of_unsampled
      nodeProxyLoss nodeOracleLoss nodeObserved nodePi U (hObserved U))

theorem discountedAIPWNodeLawLoss_eq_oracle_of_all_sampled_pi_one
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = true)
    (hPi : ∀ U : BinTree Strings, nodePi U = 1) :
    discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  unfold discountedAIPWNodeLawLoss discountedTrueNodeLawLoss
  exact discountedTreeMetaLoss_congr_all gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
    nodeOracleLoss T
    (fun U => nodeAIPWAdjustedLawLoss_eq_oracle_of_sampled_pi_one
      nodeProxyLoss nodeOracleLoss nodeObserved nodePi U (hObserved U) (hPi U))

/-! ## Scalar econometric residual identities -/

/-- With matched propensities, the scalar AIPW adjusted node law is unbiased for
the oracle node law, regardless of proxy error. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) = 0 := by
  exact DSL.designAdjustedOutcome_unbiased_of_matched_propensity
    proxyLoss oracleLoss piTrue piUsed hPiUsed hMatch E_cond hE_R hE_1 hE_linear

/-- With a misspecified propensity, the exact scalar residual is the propensity
mismatch factor times proxy-oracle error. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_residual_of_misspecified_propensity
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) =
      (1 - piTrue / piUsed) * (proxyLoss - oracleLoss) := by
  exact DSL.designAdjustedOutcome_expectation_eq_residual_of_misspecified_propensity
    proxyLoss oracleLoss piTrue piUsed hPiUsed E_cond hE_R hE_1 hE_linear

/-- Discounted aggregate envelope for adjusted-node-law error. -/
def discountedAIPWNodeErrorBound
    (gammaDepth : ℝ)
    (nodeErrorBound : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedNodeBiasBound gammaDepth nodeErrorBound T

theorem discountedAIPWNodeLawLoss_abs_sub_le_errorBound
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss nodeErrorBound : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hGamma : 0 ≤ gammaDepth)
    (hNode :
      ∀ U : BinTree Strings,
        |nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi U -
          nodeOracleLoss U| ≤ nodeErrorBound U)
    (hErrorNonneg : ∀ U : BinTree Strings, 0 ≤ nodeErrorBound U) :
    |discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T -
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T| ≤
      discountedAIPWNodeErrorBound gammaDepth nodeErrorBound T := by
  simpa [discountedAIPWNodeLawLoss, discountedAIPWNodeErrorBound]
    using
      discountedNodeLawLoss_abs_sub_le_biasBound gammaDepth
        (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
        nodeOracleLoss nodeErrorBound T hGamma hNode hErrorNonneg

/-- Nominal root/local objective using the adjusted node-law channel. -/
def nominalAIPWNodeObjective
    (Lambda gammaDepth rootLoss : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  nominalRootLocalObjective Lambda rootLoss
    (discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
      nodeObserved nodePi T)

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/UnifiedLocalLawAdjustment.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/UnifiedLocalLawAdjustment.lean

Unified local-law adjustment equation.

This module names the final paper-facing local-law adjustment layer. Node
local-law losses measured through `fhat` are corrected by node-oracle residuals
when those observations are present and aggregated down the tree with the
existing depth discount. The paper-facing theorem path supplies this corrected
local-law loss to `nominalRootLocalObjective` with the nominal lambda.

The DSL/IPW certificate components remain scalar inputs here. In particular,
the propensity-misspecification residual is explicit, so it is zero in the
matched-propensity design case and visible otherwise.
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

variable {Strings : Type*}

/-! ## Unified adjusted local-law estimate -/

/-- The unified adjusted local-law estimate:
`sum_v gammaDepth^depth(v) * (proxy_v + R_v/pi_v * (oracle_v - proxy_v))`.
It is definitionally the discounted AIPW node-law aggregate. -/
def unifiedAdjustedLocalLawEstimate
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
    nodeObserved nodePi T

theorem unifiedAdjustedLocalLawEstimate_eq_proxy_of_all_unsampled
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = false) :
    unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedSurrogateNodeLawLoss gammaDepth nodeProxyLoss T := by
  exact discountedAIPWNodeLawLoss_eq_proxy_of_all_unsampled
    gammaDepth nodeProxyLoss nodeOracleLoss nodeObserved nodePi T hObserved

theorem unifiedAdjustedLocalLawEstimate_eq_oracle_of_all_sampled_pi_one
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hObserved : ∀ U : BinTree Strings, nodeObserved U = true)
    (hPi : ∀ U : BinTree Strings, nodePi U = 1) :
    unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  exact discountedAIPWNodeLawLoss_eq_oracle_of_all_sampled_pi_one
    gammaDepth nodeProxyLoss nodeOracleLoss nodeObserved nodePi T hObserved hPi

/-! ## Scalar certificate envelope -/

/-- Explicit residual left by propensity misspecification in the scalar
design-adjusted local-law outcome. -/
def propensityMismatchResidual
    (piTrue piUsed proxyLoss oracleLoss : ℝ) : ℝ :=
  (1 - piTrue / piUsed) * (proxyLoss - oracleLoss)

theorem propensityMismatchResidual_eq_zero_of_matched
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed) :
    propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss = 0 := by
  unfold propensityMismatchResidual
  rw [hMatch]
  rw [div_self (ne_of_gt hPiUsed)]
  ring

/-- The scalar AIPW residual identity is exactly the named propensity-mismatch
residual. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_propensityMismatchResidual
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) =
      propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss := by
  exact nodeAIPWAdjustedLawLoss_expectation_eq_residual_of_misspecified_propensity
    proxyLoss oracleLoss piTrue piUsed hPiUsed E_cond hE_R hE_1 hE_linear

/-- Paper-facing scalar local-law error envelope:
IPW/AIPW point estimate plus proxy-oracle gap margin, sampling uncertainty,
oracle slack, and explicit propensity residual. -/
def adjustedLocalLawEnvelope
    (ipwEstimate gapMargin zScore se oracleSlack propensityResidual : ℝ) : ℝ :=
  ipwEstimate + gapMargin + zScore * se + 2 * oracleSlack +
    propensityResidual

/-! ## Nominal unified objective -/

/-- Paper-facing nominal root/local objective with the corrected local-law
estimate as the local channel. -/
def unifiedLocalLawNominalObjective
    (Lambda gammaDepth rootLoss : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings) : ℝ :=
  nominalRootLocalObjective Lambda rootLoss
    (unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
      nodeObserved nodePi T)

theorem unifiedLocalLawNominalObjective_eq_nominal
    {gammaDepth rootLoss : ℝ}
    {nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ}
    {nodeObserved : BinTree Strings -> DSL.SamplingIndicator}
    {nodePi : BinTree Strings -> ℝ}
    {T : BinTree Strings}
    (Lambda : ℝ) :
    unifiedLocalLawNominalObjective Lambda gammaDepth rootLoss
        nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nominalRootLocalObjective Lambda rootLoss
        (unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
          nodeObserved nodePi T) := by
  rfl

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/DoublyRobustLocalLawAdjustment.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/DoublyRobustLocalLawAdjustment.lean

Doubly robust local-law adjustment.

This module makes explicit the classical AIPW/DSL double-robust reading of the
unified local-law channel. The adjusted node-law outcome is unbiased when either
the logged propensity matches the true node-oracle sampling law, or the proxy
local-law loss is already equal to the oracle local-law loss.

The first route is inherited from `UnifiedLocalLawAdjustment`; this file adds
the exact-proxy route and names the combined disjunction.
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

variable {Strings : Type*}

/-! ## Scalar doubly robust residual cancellation -/

/-- If the proxy loss is exact, the explicit propensity-mismatch residual is
zero for any true/used propensity pair. -/
theorem propensityMismatchResidual_eq_zero_of_exact_proxy
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss = 0 := by
  unfold propensityMismatchResidual
  rw [hExact]
  ring

/-- The explicit propensity residual vanishes if either the propensity is
matched or the proxy local-law loss is exact. -/
theorem propensityMismatchResidual_eq_zero_of_matched_or_exact_proxy
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss) :
    propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss = 0 := by
  cases hDR with
  | inl hMatch =>
      exact propensityMismatchResidual_eq_zero_of_matched
        piTrue piUsed proxyLoss oracleLoss hPiUsed hMatch
  | inr hExact =>
      exact propensityMismatchResidual_eq_zero_of_exact_proxy
        piTrue piUsed proxyLoss oracleLoss hExact

/-- If the proxy outcome equals the oracle outcome, the DSL adjusted outcome is
the oracle outcome for any sampling indicator and any used propensity. -/
theorem designAdjustedOutcome_eq_true_of_exact_proxy
    (proxyLoss oracleLoss : ℝ)
    (R : DSL.SamplingIndicator)
    (piUsed : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed = oracleLoss := by
  subst proxyLoss
  unfold DSL.designAdjustedOutcome
  cases R <;> simp

/-- Exact proxy local laws remove the scalar AIPW residual even when the
propensity used for weighting is misspecified. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_exact_proxy
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (hExact : proxyLoss = oracleLoss)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) = 0 := by
  rw [nodeAIPWAdjustedLawLoss_expectation_eq_propensityMismatchResidual
    proxyLoss oracleLoss piTrue piUsed hPiUsed E_cond hE_R hE_1 hE_linear]
  exact propensityMismatchResidual_eq_zero_of_exact_proxy
    piTrue piUsed proxyLoss oracleLoss hExact

/-- Scalar doubly robust unbiasedness: the AIPW/DSL residual is zero if either
the propensity is matched or the proxy local-law loss is exact. -/
theorem nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity_or_exact_proxy
    (proxyLoss oracleLoss piTrue piUsed : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss)
    (E_cond : (DSL.SamplingIndicator -> ℝ) -> ℝ)
    (hE_R : E_cond (fun R => R.toReal) = piTrue)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : DSL.SamplingIndicator -> ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R => DSL.designAdjustedOutcome proxyLoss oracleLoss R piUsed -
      oracleLoss) = 0 := by
  cases hDR with
  | inl hMatch =>
      exact nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity
        proxyLoss oracleLoss piTrue piUsed hPiUsed hMatch E_cond hE_R hE_1
        hE_linear
  | inr hExact =>
      exact nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_exact_proxy
        proxyLoss oracleLoss piTrue piUsed hPiUsed hExact E_cond hE_R hE_1
        hE_linear

/-! ## Pointwise and discounted exact-proxy endpoints -/

/-- If the proxy node loss equals the oracle node loss at a node, the adjusted
node loss equals the oracle node loss for any observation state and propensity. -/
theorem nodeAIPWAdjustedLawLoss_eq_oracle_of_exact_proxy
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hExact : nodeProxyLoss T = nodeOracleLoss T) :
    nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi T =
      nodeOracleLoss T := by
  unfold nodeAIPWAdjustedLawLoss
  exact designAdjustedOutcome_eq_true_of_exact_proxy
    (nodeProxyLoss T) (nodeOracleLoss T) (nodeObserved T) (nodePi T) hExact

-- NOTE (consolidation 2026-07-02): the original file restated the textually
-- identical `private theorem discountedTreeMetaLoss_congr_all` from
-- `NodeAIPWLocalLawAdjustment.lean` here. In the consolidated module the copy
-- from the `NodeAIPWLocalLawAdjustment` section above is in scope, so the
-- duplicate is elided to avoid a duplicate-declaration error.

/-- If every proxy node loss equals its oracle counterpart, the discounted AIPW
node law is exactly the true-oracle discounted node law. -/
theorem discountedAIPWNodeLawLoss_eq_oracle_of_exact_proxy
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hExact : ∀ U : BinTree Strings, nodeProxyLoss U = nodeOracleLoss U) :
    discountedAIPWNodeLawLoss gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  unfold discountedAIPWNodeLawLoss discountedTrueNodeLawLoss
  exact discountedTreeMetaLoss_congr_all gammaDepth
    (nodeAIPWAdjustedLawLoss nodeProxyLoss nodeOracleLoss nodeObserved nodePi)
    nodeOracleLoss T
    (fun U => nodeAIPWAdjustedLawLoss_eq_oracle_of_exact_proxy
      nodeProxyLoss nodeOracleLoss nodeObserved nodePi U (hExact U))

/-- The unified adjusted local-law estimate reduces to the true-oracle
discounted law when the proxy local law is exact at every node. -/
theorem unifiedAdjustedLocalLawEstimate_eq_oracle_of_exact_proxy
    (gammaDepth : ℝ)
    (nodeProxyLoss nodeOracleLoss : BinTree Strings -> ℝ)
    (nodeObserved : BinTree Strings -> DSL.SamplingIndicator)
    (nodePi : BinTree Strings -> ℝ)
    (T : BinTree Strings)
    (hExact : ∀ U : BinTree Strings, nodeProxyLoss U = nodeOracleLoss U) :
    unifiedAdjustedLocalLawEstimate gammaDepth nodeProxyLoss nodeOracleLoss
        nodeObserved nodePi T =
      discountedTrueNodeLawLoss gammaDepth nodeOracleLoss T := by
  unfold unifiedAdjustedLocalLawEstimate
  exact discountedAIPWNodeLawLoss_eq_oracle_of_exact_proxy
    gammaDepth nodeProxyLoss nodeOracleLoss nodeObserved nodePi T hExact

/-! ## Envelope simplifications under doubly robust residual cancellation -/

/-- If the explicit propensity residual is zero, the adjusted local-law envelope
is the same as the no-propensity-residual envelope. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    (ipwEstimate gapMargin zScore se oracleSlack propensityResidual : ℝ)
    (hResidual : propensityResidual = 0) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        propensityResidual =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  simp [adjustedLocalLawEnvelope, hResidual]

/-- Matched propensities remove the explicit propensity-residual term from the
adjusted local-law envelope. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_matched_propensity
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss) =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  exact adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    ipwEstimate gapMargin zScore se oracleSlack
    (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss)
    (propensityMismatchResidual_eq_zero_of_matched
      piTrue piUsed proxyLoss oracleLoss hPiUsed hMatch)

/-- Exact proxy local laws remove the explicit propensity-residual term from the
adjusted local-law envelope. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_exact_proxy
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss) =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  exact adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    ipwEstimate gapMargin zScore se oracleSlack
    (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss)
    (propensityMismatchResidual_eq_zero_of_exact_proxy
      piTrue piUsed proxyLoss oracleLoss hExact)

/-- The adjusted local-law envelope drops the explicit propensity-residual term
under either classical doubly robust route. -/
theorem adjustedLocalLawEnvelope_eq_no_propensity_residual_of_matched_propensity_or_exact_proxy
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss) :
    adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss) =
      adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se oracleSlack
        0 := by
  exact adjustedLocalLawEnvelope_eq_no_propensity_residual_of_zero_residual
    ipwEstimate gapMargin zScore se oracleSlack
    (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss)
    (propensityMismatchResidual_eq_zero_of_matched_or_exact_proxy
      piTrue piUsed proxyLoss oracleLoss hPiUsed hDR)

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/CoverageNormalizedObjective.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/CoverageNormalizedObjective.lean

Coverage-normalized tree objectives for budgeted document supervision.

This module isolates the current bug in the Markov tree trainer:

* the legacy/current document-level objective divides the selected root-loss sum
  by the full batch document count, which introduces a hidden multiplicative
  coverage factor;
* the corrected objective divides by the number of supervised documents, so
  root-label coverage changes variance but not the intended root-vs-local tradeoff;
* under constant inclusion probability, the Horvitz-Thompson document-mean
  estimator is unbiased for the full population document mean.

The file is intentionally finite and elementary: documents live in a finite type,
the supervised subset is a `Finset`, and the stochastic results are stated for a
PMF over subsets with constant marginal inclusion probability.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-- Explicit root-vs-local weight bundle for the tree objective. The intended
tradeoff lives entirely in these weights, not in supervision coverage. -/
structure CoverageNormalizedTreeObjectiveWeights where
  rootWeight : ℝ
  c1Weight : ℝ
  c2Weight : ℝ
  c3Weight : ℝ

section Deterministic

variable {Doc : Type*} [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- Full-population mean of a document-level loss. -/
def documentMean (loss : Doc → ℝ) : ℝ :=
  (∑ i, loss i) / (Fintype.card Doc : ℝ)

/-- Mean of a document-level loss over the supervised subset. Empty subsets map
to `0`; the theorems below use `selected.Nonempty` when normalization matters. -/
def selectedDocumentMean (selected : Finset Doc) (loss : Doc → ℝ) : ℝ :=
  if h : selected.card = 0 then 0 else selected.sum loss / (selected.card : ℝ)

/-- Realized document-supervision coverage rate. -/
def coverageRate (selected : Finset Doc) : ℝ :=
  (selected.card : ℝ) / (Fintype.card Doc : ℝ)

/-- Dense local-law objective. These terms are already normalized at the document
level and should not change when root supervision coverage changes. -/
def denseLocalObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.c1Weight * documentMean c1Loss
    + weights.c2Weight * documentMean c2Loss
    + weights.c3Weight * documentMean c3Loss

/-- Current buggy objective: the supervised root-loss sum is divided by the full
document count, which hides a multiplicative coverage factor. -/
def currentCoverageScaledTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.rootWeight * (selected.sum rootLoss / (Fintype.card Doc : ℝ))
    + denseLocalObjective weights c1Loss c2Loss c3Loss

/-- Corrected objective: the document/root term is normalized by the number of
supervised documents, so coverage changes only the variance of the selected mean. -/
def correctedCoverageNormalizedTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.rootWeight * selectedDocumentMean selected rootLoss
    + denseLocalObjective weights c1Loss c2Loss c3Loss

/-- Full-supervision objective. -/
def fullSupervisionTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.rootWeight * documentMean rootLoss
    + denseLocalObjective weights c1Loss c2Loss c3Loss

/-- Horvitz-Thompson document-mean estimator under a constant inclusion
probability `coverage`. -/
def constantInclusionHTRootMeanOfProb
    (coverage : ℝ) (selected : Finset Doc) (rootLoss : Doc → ℝ) : ℝ :=
  (∑ i, if i ∈ selected then rootLoss i / coverage else 0) / (Fintype.card Doc : ℝ)

/-- HT document-mean estimator where the inclusion probability is instantiated
at the realized coverage rate. Under fixed-size sampling, this agrees exactly
with the selected-subset mean. -/
def constantInclusionHTRootMean
    (selected : Finset Doc) (rootLoss : Doc → ℝ) : ℝ :=
  constantInclusionHTRootMeanOfProb (coverageRate selected) selected rootLoss

lemma selectedDocumentMean_eq_sum_div_card
    (selected : Finset Doc) (loss : Doc → ℝ) (hsel : selected.Nonempty) :
    selectedDocumentMean selected loss = selected.sum loss / (selected.card : ℝ) := by
  have hs : selected.card ≠ 0 := Finset.card_ne_zero.mpr hsel
  simp [selectedDocumentMean, hs]

lemma documentMean_univ_eq_selectedDocumentMean
    (loss : Doc → ℝ) :
    documentMean loss = selectedDocumentMean (Finset.univ : Finset Doc) loss := by
  have hs : (Finset.univ : Finset Doc).Nonempty := Finset.univ_nonempty
  have hcard :
      ((Finset.univ : Finset Doc).card : ℝ) = (Fintype.card Doc : ℝ) := by
    simp
  rw [selectedDocumentMean_eq_sum_div_card _ _ hs]
  simp [documentMean, hcard]

/-- The current objective contains a hidden multiplicative coverage factor on the
root/document term. -/
theorem currentCoverageScaledTreeObjective_eq_coverageRate_mul_selectedRootMean
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ)
    (hsel : selected.Nonempty) :
    currentCoverageScaledTreeObjective weights selected rootLoss c1Loss c2Loss c3Loss
      = coverageRate selected * weights.rootWeight * selectedDocumentMean selected rootLoss
          + denseLocalObjective weights c1Loss c2Loss c3Loss := by
  have hs_nat : selected.card ≠ 0 := Finset.card_ne_zero.mpr hsel
  have hs : (selected.card : ℝ) ≠ 0 := by
    exact_mod_cast hs_nat
  have hdoc : (Fintype.card Doc : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  rw [selectedDocumentMean_eq_sum_div_card _ _ hsel]
  unfold currentCoverageScaledTreeObjective coverageRate
  field_simp [hs, hdoc]

/-- The corrected objective keeps the root/document term at the supervised-subset
mean, removing the hidden coverage multiplier. -/
theorem correctedCoverageNormalizedTreeObjective_eq_rootWeight_mul_selectedRootMean
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) :
    correctedCoverageNormalizedTreeObjective weights selected rootLoss c1Loss c2Loss c3Loss
      = weights.rootWeight * selectedDocumentMean selected rootLoss
          + denseLocalObjective weights c1Loss c2Loss c3Loss := by
  simp [correctedCoverageNormalizedTreeObjective]

/-- At full coverage, the corrected objective coincides with the full-supervision
objective. -/
theorem correctedCoverageNormalizedTreeObjective_eq_fullSupervision_at_fullCoverage
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) :
    correctedCoverageNormalizedTreeObjective weights (Finset.univ : Finset Doc)
        rootLoss c1Loss c2Loss c3Loss
      = fullSupervisionTreeObjective weights rootLoss c1Loss c2Loss c3Loss := by
  rw [correctedCoverageNormalizedTreeObjective, fullSupervisionTreeObjective,
    ← documentMean_univ_eq_selectedDocumentMean (loss := rootLoss)]

/-- With constant inclusion probability set equal to the realized coverage rate,
the HT document-mean estimator collapses to the selected-subset mean. -/
theorem constantInclusionHTRootMean_eq_selectedDocumentMean
    (selected : Finset Doc) (rootLoss : Doc → ℝ) (hsel : selected.Nonempty) :
    constantInclusionHTRootMean selected rootLoss = selectedDocumentMean selected rootLoss := by
  have hs_nat : selected.card ≠ 0 := Finset.card_ne_zero.mpr hsel
  have hs : (selected.card : ℝ) ≠ 0 := by
    exact_mod_cast hs_nat
  have hdoc : (Fintype.card Doc : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  rw [selectedDocumentMean_eq_sum_div_card _ _ hsel]
  unfold constantInclusionHTRootMean constantInclusionHTRootMeanOfProb coverageRate
  rw [Finset.sum_ite_mem]
  simp
  field_simp [hs, hdoc]
  calc
    (selected.card : ℝ) * ∑ i ∈ selected, rootLoss i * (Fintype.card Doc : ℝ) / (selected.card : ℝ)
      = ∑ i ∈ selected, (selected.card : ℝ) * (rootLoss i * (Fintype.card Doc : ℝ) / (selected.card : ℝ)) := by
          simpa using
            (Finset.mul_sum selected
              (fun i => rootLoss i * (Fintype.card Doc : ℝ) / (selected.card : ℝ))
              (a := (selected.card : ℝ)))
    _ = ∑ i ∈ selected, (Fintype.card Doc : ℝ) * rootLoss i := by
          apply Finset.sum_congr rfl
          intro i hi
          field_simp [hs, hdoc]
    _ = (Fintype.card Doc : ℝ) * selected.sum rootLoss := by
          simpa [mul_comm, mul_left_comm, mul_assoc] using
            (Finset.mul_sum selected rootLoss (a := (Fintype.card Doc : ℝ))).symm

/-- Pointwise slack decomposition: the corrected objective differs from the
full-supervision objective only through the selected-vs-population root mean. -/
theorem correctedCoverageNormalizedTreeObjective_sub_fullSupervisionTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) :
    correctedCoverageNormalizedTreeObjective weights selected rootLoss c1Loss c2Loss c3Loss
      - fullSupervisionTreeObjective weights rootLoss c1Loss c2Loss c3Loss
      = weights.rootWeight * (selectedDocumentMean selected rootLoss - documentMean rootLoss) := by
  unfold correctedCoverageNormalizedTreeObjective fullSupervisionTreeObjective
    denseLocalObjective
  ring_nf

end Deterministic

section Stochastic

variable {Doc Θ : Type*} [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- Finite expectation over a PMF on a finite type. -/
def finiteExpectation {α : Type*} [Fintype α] (μ : PMF α) (f : α → ℝ) : ℝ :=
  ∑ a, (μ a).toReal * f a

lemma finiteExpectation_const {α : Type*} [Fintype α] (μ : PMF α) (c : ℝ) :
    finiteExpectation μ (fun _ : α => c) = c := by
  unfold finiteExpectation
  calc
    ∑ x, (μ x).toReal * c = (∑ x, (μ x).toReal) * c := by
      simpa using
        (Finset.sum_mul (Finset.univ : Finset α) (fun x => (μ x).toReal) c).symm
    _ = c := by
      have hmass : ∑ x, (μ x).toReal = (1 : ℝ) := by
        simpa [tsum_fintype] using (PMF.toReal_tsum_coe μ)
      rw [hmass]
      ring

lemma finiteExpectation_add {α : Type*} [Fintype α] (μ : PMF α) (f g : α → ℝ) :
    finiteExpectation μ (fun x => f x + g x) = finiteExpectation μ f + finiteExpectation μ g := by
  unfold finiteExpectation
  simp_rw [mul_add]
  rw [Finset.sum_add_distrib]

lemma finiteExpectation_mul_left {α : Type*} [Fintype α] (μ : PMF α) (a : ℝ) (f : α → ℝ) :
    finiteExpectation μ (fun x => a * f x) = a * finiteExpectation μ f := by
  unfold finiteExpectation
  calc
    ∑ x, (μ x).toReal * (a * f x)
      = ∑ x, a * ((μ x).toReal * f x) := by
          apply Finset.sum_congr rfl
          intro x hx
          ring
    _ = a * ∑ x, (μ x).toReal * f x := by
          simpa using
            (Finset.mul_sum (Finset.univ : Finset α)
              (fun x => (μ x).toReal * f x)
              (a := a)).symm

/-- Expected corrected objective using a constant-inclusion-probability HT root
term. This is the stochastic version of the corrected objective. -/
def expectedCorrectedCoverageNormalizedTreeObjective
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) : Θ → ℝ :=
  fun θ =>
    finiteExpectation μ (fun selected =>
      weights.rootWeight * constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ)
        + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ))

/-- Full-supervision objective as a function of the parameter `θ`. -/
def fullSupervisionTreeObjectiveFn
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) : Θ → ℝ :=
  fun θ =>
    fullSupervisionTreeObjective weights (rootLoss θ) (c1Loss θ) (c2Loss θ) (c3Loss θ)

/-- If each document has constant marginal inclusion probability `coverage`, then
the HT document-mean estimator is unbiased for the full population document mean. -/
theorem finiteExpectation_constantInclusionHTRootMean_eq_documentMean
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (rootLoss : Doc → ℝ)
    (hcoverage : coverage ≠ 0)
    (hmarg :
      ∀ i : Doc, finiteExpectation μ (fun selected => if i ∈ selected then (1 : ℝ) else 0) = coverage) :
    finiteExpectation μ (fun selected => constantInclusionHTRootMeanOfProb coverage selected rootLoss)
      = documentMean rootLoss := by
  classical
  let n : ℝ := Fintype.card Doc
  have hdoc0 : (Fintype.card Doc : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  have hdoc : n ≠ 0 := by
    simpa [n] using hdoc0
  have hmarg' :
      ∀ i : Doc, ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0) = coverage := by
    intro i
    simpa [finiteExpectation] using hmarg i
  unfold finiteExpectation constantInclusionHTRootMeanOfProb documentMean
  calc
    ∑ selected, (μ selected).toReal *
        ((∑ i, if i ∈ selected then rootLoss i / coverage else 0) / n)
      = ((∑ selected, (μ selected).toReal *
            (∑ i, if i ∈ selected then rootLoss i / coverage else 0)) / n) := by
          rw [div_eq_mul_inv]
          simpa [mul_assoc] using
            (Finset.sum_mul (Finset.univ : Finset (Finset Doc))
              (fun selected =>
                (μ selected).toReal * (∑ i, if i ∈ selected then rootLoss i / coverage else 0))
              (n⁻¹)).symm
    _ = ((∑ selected, ∑ i, (μ selected).toReal *
            (if i ∈ selected then rootLoss i / coverage else 0)) / n) := by
          congr 1
          apply Finset.sum_congr rfl
          intro selected hselected
          simpa using
            (Finset.mul_sum (Finset.univ : Finset Doc)
              (fun i => if i ∈ selected then rootLoss i / coverage else 0)
              (a := (μ selected).toReal))
    _ = ((∑ i, ∑ selected, (μ selected).toReal *
            (if i ∈ selected then rootLoss i / coverage else 0)) / n) := by
          congr 1
          simpa using
            (Finset.sum_comm
              (s := (Finset.univ : Finset (Finset Doc)))
              (t := (Finset.univ : Finset Doc))
              (f := fun selected i =>
                (μ selected).toReal *
                  (if i ∈ selected then rootLoss i / coverage else 0)))
    _ = ((∑ i, (rootLoss i / coverage) *
            ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0)) / n) := by
          congr 1
          apply Finset.sum_congr rfl
          intro i hi
          have hfactor :
              ∑ selected, (μ selected).toReal * (if i ∈ selected then rootLoss i / coverage else 0)
                = (rootLoss i / coverage) *
                    ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0) := by
              calc
                ∑ selected, (μ selected).toReal * (if i ∈ selected then rootLoss i / coverage else 0)
                  = ∑ selected, (rootLoss i / coverage) *
                      ((μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0)) := by
                        apply Finset.sum_congr rfl
                        intro selected hselected
                        by_cases hi' : i ∈ selected
                        · simp [hi', mul_assoc, mul_left_comm, mul_comm]
                        · simp [hi']
                _ = (rootLoss i / coverage) *
                      ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0) := by
                        simpa using
                          (Finset.mul_sum (Finset.univ : Finset (Finset Doc))
                            (fun selected =>
                              (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0))
                            (a := (rootLoss i / coverage))).symm
          exact hfactor
    _ = ((∑ i, rootLoss i) / n) := by
          congr 1
          apply Finset.sum_congr rfl
          intro i hi
          rw [hmarg' i]
          field_simp [hcoverage]
    _ = documentMean rootLoss := by
          simp [documentMean, n]

/-- The expected corrected objective matches the full-supervision objective when
the document-supervision design has constant inclusion probability. -/
theorem finiteExpectation_correctedCoverageNormalizedTreeObjective_eq_fullSupervision
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ)
    (hcoverage : coverage ≠ 0)
    (hmarg :
      ∀ i : Doc, finiteExpectation μ (fun selected => if i ∈ selected then (1 : ℝ) else 0) = coverage) :
    expectedCorrectedCoverageNormalizedTreeObjective μ coverage weights rootLoss c1Loss c2Loss c3Loss
      = fullSupervisionTreeObjectiveFn weights rootLoss c1Loss c2Loss c3Loss := by
  funext θ
  have hroot :=
    finiteExpectation_constantInclusionHTRootMean_eq_documentMean
      (μ := μ) (coverage := coverage) (rootLoss := rootLoss θ) hcoverage hmarg
  unfold expectedCorrectedCoverageNormalizedTreeObjective fullSupervisionTreeObjectiveFn
  calc
    finiteExpectation μ (fun selected =>
      weights.rootWeight * constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ)
        + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ))
      = finiteExpectation μ (fun selected =>
          weights.rootWeight * constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ))
        + finiteExpectation μ (fun _ =>
            denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ)) := by
            rw [finiteExpectation_add]
    _ = weights.rootWeight *
          finiteExpectation μ
            (fun selected => constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ))
        + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ) := by
            rw [finiteExpectation_mul_left, finiteExpectation_const]
    _ = weights.rootWeight * documentMean (rootLoss θ)
          + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ) := by
            rw [hroot]
    _ = fullSupervisionTreeObjective weights (rootLoss θ) (c1Loss θ) (c2Loss θ) (c3Loss θ) := by
            simp [fullSupervisionTreeObjective]

/-- Generic same-argmin lemma for pointwise-equal objectives. -/
theorem paramArgmin_eq_of_pointwise_loss_eq
    {Θ : Type*}
    (loss₁ loss₂ : Θ → ℝ)
    (hEq : ∀ θ, loss₁ θ = loss₂ θ) :
    ParamArgmin loss₁ = ParamArgmin loss₂ := by
  ext θ
  simp [ParamArgmin, hEq]

/-- The corrected expected objective has the same parameter argmin set as the
full-supervision objective. Coverage changes only the sampling noise, not the
population objective being optimized. -/
theorem coverageNormalized_expectedObjective_same_paramArgmin
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ)
    (hcoverage : coverage ≠ 0)
    (hmarg :
      ∀ i : Doc, finiteExpectation μ (fun selected => if i ∈ selected then (1 : ℝ) else 0) = coverage) :
    ParamArgmin
        (expectedCorrectedCoverageNormalizedTreeObjective μ coverage weights
          rootLoss c1Loss c2Loss c3Loss)
      = ParamArgmin (fullSupervisionTreeObjectiveFn weights rootLoss c1Loss c2Loss c3Loss) := by
  apply paramArgmin_eq_of_pointwise_loss_eq
  intro θ
  have hEq := congrArg (fun f => f θ)
    (finiteExpectation_correctedCoverageNormalizedTreeObjective_eq_fullSupervision
      (μ := μ) (coverage := coverage) (weights := weights)
      (rootLoss := rootLoss) (c1Loss := c1Loss) (c2Loss := c2Loss) (c3Loss := c3Loss)
      hcoverage hmarg)
  simpa using hEq

end Stochastic

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/DiscountedIPWObjective.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/DiscountedIPWObjective.lean

Bridge between discounted tree-style objectives and IPW / Horvitz-Thompson
estimation.

The key point is simple: if each supervision component is estimated unbiasedly
with HT/IPW, then any fixed linear weighting scheme applied to those components
remains unbiased. This covers:

- depth discounting with weights `γ^d`,
- the current root / C1 / C2 / C3 weighting scheme, and
- combined schemes obtained by taking a product index such as depth × channel.

So adding a reinforcement-learning-style discount factor does not break the
design-based logic. It only changes the deterministic coefficients in front of
already unbiased component estimators.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section GenericWeightedIPW

variable {Doc Depth Θ : Type*}
variable [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]
variable [Fintype Depth] [DecidableEq Depth]

/-- Finite expectation commutes with pushforward of a finite PMF. -/
lemma finiteExpectation_map
    {α β : Type*} [Fintype α] [Fintype β]
    (μ : PMF α) (g : α → β) (f : β → ℝ) :
    finiteExpectation (PMF.map g μ) f =
      finiteExpectation μ (fun a => f (g a)) := by
  classical
  have hμ :
      μ = PMF.ofFintype (fun a => μ a) (by simpa [tsum_fintype] using μ.tsum_coe) := by
    ext a
    simp
  rw [hμ, PMF.map_ofFintype]
  unfold finiteExpectation
  calc
    ∑ x, ((∑ a with g a = x, μ a)).toReal * f x
      = ∑ x, (∑ a with g a = x, (μ a).toReal) * f x := by
          apply Finset.sum_congr rfl
          intro x hx
          congr 1
          rw [ENNReal.toReal_sum]
          intro a ha
          exact PMF.apply_ne_top _ _
    _ = ∑ x, ∑ a with g a = x, (μ a).toReal * f x := by
          apply Finset.sum_congr rfl
          intro x hx
          calc
            (∑ a with g a = x, (μ a).toReal) * f x
              = f x * ∑ a with g a = x, (μ a).toReal := by ring
            _ = ∑ a with g a = x, f x * (μ a).toReal := by
                  exact Finset.mul_sum (s := Finset.univ.filter fun a => g a = x)
                    (f := fun a => (μ a).toReal) (a := f x)
            _ = ∑ a with g a = x, (μ a).toReal * f x := by
                  apply Finset.sum_congr rfl
                  intro a ha
                  ring
    _ = ∑ x, ∑ a with g a = x, (μ a).toReal * f (g a) := by
          apply Finset.sum_congr rfl
          intro x hx
          apply Finset.sum_congr rfl
          intro a ha
          simp at ha
          simp [ha]
    _ = ∑ a, (μ a).toReal * f (g a) := by
          simpa using
            (Finset.sum_fiberwise_of_maps_to
              (s := (Finset.univ : Finset α))
              (t := (Finset.univ : Finset β))
              (g := g)
              (f := fun a => (μ a).toReal * f (g a))
              (h := fun a ha => by simp))

omit [DecidableEq Doc] [Nonempty Doc] in
/-- Finite expectation commutes with finite sums over an external index. -/
lemma finiteExpectation_sum
    (μ : PMF (Depth → Finset Doc))
    (f : Depth → (Depth → Finset Doc) → ℝ) :
    finiteExpectation μ (fun selected => ∑ d, f d selected) =
      ∑ d, finiteExpectation μ (fun selected => f d selected) := by
  unfold finiteExpectation
  calc
    ∑ selected, (μ selected).toReal * ∑ d, f d selected
      = ∑ selected, ∑ d, (μ selected).toReal * f d selected := by
          apply Finset.sum_congr rfl
          intro selected hselected
          rw [Finset.mul_sum]
    _ = ∑ d, ∑ selected, (μ selected).toReal * f d selected := by
          simpa using
            (Finset.sum_comm
              (s := (Finset.univ : Finset (Depth → Finset Doc)))
              (t := (Finset.univ : Finset Depth))
              (f := fun selected d => (μ selected).toReal * f d selected))
    _ = ∑ d, finiteExpectation μ (fun selected => f d selected) := by
          rfl

/-- Generic population objective built as a finite weighted sum of
document-level component means. The index type can stand for supervision
channels, tree depths, or depth × channel pairs. -/
def fullWeightedDocumentObjective
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ) : Θ → ℝ :=
  fun θ => ∑ d, weights d * documentMean (fun i => componentLoss θ d i)

/-- IPW-corrected version of the same objective. For each component `d`, the
logged subset `selected d` is corrected with the HT mean estimator. -/
def expectedIPWWeightedDocumentObjective
    (μ : PMF (Depth → Finset Doc))
    (coverage : Depth → ℝ)
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ) : Θ → ℝ :=
  fun θ =>
    finiteExpectation μ (fun selected =>
      ∑ d, weights d *
        constantInclusionHTRootMeanOfProb (coverage d) (selected d)
          (fun i => componentLoss θ d i))

/-- If each component has constant marginal inclusion probability, the expected
IPW-weighted objective equals the full population objective. -/
theorem expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective
    (μ : PMF (Depth → Finset Doc))
    (coverage : Depth → ℝ)
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ)
    (hcoverage : ∀ d : Depth, coverage d ≠ 0)
    (hmarg :
      ∀ d : Depth, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    expectedIPWWeightedDocumentObjective μ coverage weights componentLoss =
      fullWeightedDocumentObjective weights componentLoss := by
  funext θ
  unfold expectedIPWWeightedDocumentObjective fullWeightedDocumentObjective
  rw [finiteExpectation_sum]
  simp_rw [finiteExpectation_mul_left]
  apply Finset.sum_congr rfl
  intro d hd
  have hroot :
      finiteExpectation μ
          (fun x => constantInclusionHTRootMeanOfProb (coverage d) (x d)
            (fun i => componentLoss θ d i)) =
        documentMean (fun i => componentLoss θ d i) := by
    rw [← finiteExpectation_map
      (μ := μ)
      (g := fun x : Depth → Finset Doc => x d)
      (f := fun selected =>
        constantInclusionHTRootMeanOfProb (coverage d) selected
          (fun i => componentLoss θ d i))]
    exact finiteExpectation_constantInclusionHTRootMean_eq_documentMean
      (μ := PMF.map (fun x : Depth → Finset Doc => x d) μ)
      (coverage := coverage d)
      (rootLoss := fun i => componentLoss θ d i)
      (hcoverage := hcoverage d)
      (hmarg := by
        intro i
        rw [finiteExpectation_map
          (μ := μ)
          (g := fun x : Depth → Finset Doc => x d)
          (f := fun selected : Finset Doc => if i ∈ selected then (1 : ℝ) else 0)]
        exact hmarg d i)
  rw [hroot]

/-- Pointwise-equal expected objectives have the same parameter argmin set. -/
theorem ipwWeightedObjective_same_paramArgmin
    (μ : PMF (Depth → Finset Doc))
    (coverage : Depth → ℝ)
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ)
    (hcoverage : ∀ d : Depth, coverage d ≠ 0)
    (hmarg :
      ∀ d : Depth, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    ParamArgmin (expectedIPWWeightedDocumentObjective μ coverage weights componentLoss) =
      ParamArgmin (fullWeightedDocumentObjective weights componentLoss) := by
  apply paramArgmin_eq_of_pointwise_loss_eq
  intro θ
  have hEq := congrArg (fun f => f θ)
    (expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective
      (μ := μ) (coverage := coverage) (weights := weights)
      (componentLoss := componentLoss) hcoverage hmarg)
  simpa using hEq

end GenericWeightedIPW

section DiscountedSpecialization

variable {Doc Θ : Type*}
variable [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- RL-style discount weights indexed by finite depth. -/
def discountedDepthWeights {n : ℕ} (γ : ℝ) : Fin n → ℝ :=
  fun d => γ ^ (d : ℕ)

/-- Population objective with depth discounting. -/
def fullDiscountedDocumentObjective {n : ℕ}
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ) : Θ → ℝ :=
  fullWeightedDocumentObjective (discountedDepthWeights γ) depthLoss

/-- IPW-corrected discounted objective. -/
def expectedIPWDiscountedDocumentObjective {n : ℕ}
    (μ : PMF (Fin n → Finset Doc))
    (coverage : Fin n → ℝ)
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ) : Θ → ℝ :=
  expectedIPWWeightedDocumentObjective μ coverage (discountedDepthWeights γ) depthLoss

/-- Discounting by `γ^d` preserves HT/IPW unbiasedness under constant marginal
inclusion probabilities at each depth. -/
theorem expectedIPWDiscountedDocumentObjective_eq_fullDiscountedDocumentObjective
    {n : ℕ}
    (μ : PMF (Fin n → Finset Doc))
    (coverage : Fin n → ℝ)
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ)
    (hcoverage : ∀ d : Fin n, coverage d ≠ 0)
    (hmarg :
      ∀ d : Fin n, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    expectedIPWDiscountedDocumentObjective μ coverage γ depthLoss =
      fullDiscountedDocumentObjective γ depthLoss := by
  exact expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective
    (μ := μ) (coverage := coverage) (weights := discountedDepthWeights γ)
    (componentLoss := depthLoss) hcoverage hmarg

/-- Therefore the IPW-corrected discounted objective and the full discounted
objective have the same parameter argmin set. -/
theorem ipwDiscountedObjective_same_paramArgmin
    {n : ℕ}
    (μ : PMF (Fin n → Finset Doc))
    (coverage : Fin n → ℝ)
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ)
    (hcoverage : ∀ d : Fin n, coverage d ≠ 0)
    (hmarg :
      ∀ d : Fin n, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    ParamArgmin (expectedIPWDiscountedDocumentObjective μ coverage γ depthLoss) =
      ParamArgmin (fullDiscountedDocumentObjective γ depthLoss) := by
  exact ipwWeightedObjective_same_paramArgmin
    (μ := μ) (coverage := coverage) (weights := discountedDepthWeights γ)
    (componentLoss := depthLoss) hcoverage hmarg

end DiscountedSpecialization

section CurrentWeightingScheme

variable {Doc Θ : Type*}
variable [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- The current tree-training supervision channels. This packages the existing
root / C1 / C2 / C3 weighting scheme as an instance of the generic weighted-IPW
surface. -/
inductive TreeSupervisionChannel
| root
| c1
| c2
| c3
deriving DecidableEq, Fintype

/-- Explicit equivalence used to expand finite sums over the four supervision
channels. -/
def treeSupervisionChannelEquivFin4 : TreeSupervisionChannel ≃ Fin 4 where
  toFun
    | .root => 0
    | .c1 => 1
    | .c2 => 2
    | .c3 => 3
  invFun
    | ⟨0, _⟩ => .root
    | ⟨1, _⟩ => .c1
    | ⟨2, _⟩ => .c2
    | ⟨3, _⟩ => .c3
  left_inv := by
    intro c
    cases c <;> rfl
  right_inv := by
    intro i
    rcases i with ⟨i, hi⟩
    have hi' : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
    rcases hi' with rfl | rfl | rfl | rfl <;> rfl

/-- Closed-form expansion of a sum over the four supervision channels. -/
lemma sum_treeSupervisionChannel (f : TreeSupervisionChannel → ℝ) :
    ∑ c, f c = f .root + f .c1 + f .c2 + f .c3 := by
  let e := treeSupervisionChannelEquivFin4
  calc
    ∑ c, f c = ∑ i : Fin 4, f (e.symm i) := by
      symm
      exact Fintype.sum_equiv e (fun c => f c) (fun i => f (e.symm i)) (by intro x; simp [e])
    _ = f .root + f .c1 + f .c2 + f .c3 := by
      simp [e, treeSupervisionChannelEquivFin4, Fin.sum_univ_four]

/-- Convert the existing tree-objective weight bundle into a generic channel
weight function. -/
def channelWeightOfCoverageNormalized
    (weights : CoverageNormalizedTreeObjectiveWeights) :
    TreeSupervisionChannel → ℝ
| .root => weights.rootWeight
| .c1 => weights.c1Weight
| .c2 => weights.c2Weight
| .c3 => weights.c3Weight

/-- Package the existing root / C1 / C2 / C3 document losses into a single
generic component-loss family. -/
def channelLossFamily
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) :
    Θ → TreeSupervisionChannel → Doc → ℝ
| θ, .root, i => rootLoss θ i
| θ, .c1, i => c1Loss θ i
| θ, .c2, i => c2Loss θ i
| θ, .c3, i => c3Loss θ i

omit [DecidableEq Doc] [Nonempty Doc] in
/-- The current full-supervision tree objective is exactly the generic weighted
document objective instantiated at the four supervision channels. -/
theorem fullWeightedDocumentObjective_eq_fullSupervisionTreeObjectiveFn
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) :
    fullWeightedDocumentObjective
        (channelWeightOfCoverageNormalized weights)
        (channelLossFamily rootLoss c1Loss c2Loss c3Loss)
      = fullSupervisionTreeObjectiveFn weights rootLoss c1Loss c2Loss c3Loss := by
  funext θ
  rw [fullWeightedDocumentObjective, sum_treeSupervisionChannel]
  simp [channelWeightOfCoverageNormalized, channelLossFamily,
    fullSupervisionTreeObjectiveFn, fullSupervisionTreeObjective,
    denseLocalObjective, documentMean]
  ring

end CurrentWeightingScheme

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/DoublyRobustMinimizationObjective.lean (consolidated 2026-07-02)

This terminal section is the ACTIVE optimization surface: `drMinimizationValue`
names the objective actually optimized, matching the Python ObjectiveSpec v1
default convex mixing `(1 - λ_eff) · root + λ_eff · corrected`. -/

section

/-!
# FormalProofs/OPT/DoublyRobustMinimizationObjective.lean

Full doubly robust minimization objective.

This module names the active optimization surface after the local-law channel
has been corrected by the DSL/AIPW doubly robust adjustment. The root supervised
channel is combined with the adjusted discounted local-law channel using the
nominal root/local weight `Lambda`.
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

variable {Param Strings : Type*}

/-! ## Proxy-oracle gap loss vocabulary -/

/-- Pointwise gap loss between a learned proxy `fhat` and the target oracle
`fstar`. Empirical or population averages of this scalar are the intended
instantiations of the proxy-oracle gap channel below. -/
def oracleGapLossAt {X Y : Type*} [PseudoMetricSpace Y]
    (fstar fhat : X -> Y) (x : X) : ℝ :=
  dist (fhat x) (fstar x)

theorem oracleGapLossAt_nonneg {X Y : Type*} [PseudoMetricSpace Y]
    (fstar fhat : X -> Y) (x : X) :
    0 ≤ oracleGapLossAt fstar fhat x := by
  exact dist_nonneg

/-! ## Problem data -/

/-- Generic data for the full doubly robust minimization problem.

`Param` is the candidate predictor/policy/summarizer class being optimized.
For each candidate, the local-law channel is measured by a proxy loss at every
node and corrected by node-oracle observations when they are available. -/
structure DoublyRobustMinimizationProblem (Param Strings : Type*) where
  oracleGapLoss : Param -> ℝ
  rootLoss : Param -> ℝ
  nodeProxyLoss : Param -> BinTree Strings -> ℝ
  nodeOracleLoss : Param -> BinTree Strings -> ℝ
  nodeObserved : Param -> BinTree Strings -> DSL.SamplingIndicator
  nodePi : Param -> BinTree Strings -> ℝ
  gammaDepth : ℝ
  Lambda : ℝ
  oracleGapWeight : ℝ
  rootWeight : ℝ

/-! ## Objective components -/

/-- Proxy-oracle gap channel contribution for penalizing the discrepancy
between the learned proxy measurement `fhat` and the oracle `fstar`. -/
def drOracleGapChannelLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) : ℝ :=
  problem.oracleGapWeight * problem.oracleGapLoss theta

/-- Root supervised channel contribution before root/local lambda mixing. -/
def drRootChannelLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) : ℝ :=
  problem.rootWeight * problem.rootLoss theta

/-- Discounted doubly robust adjusted local-law estimate for one candidate. -/
def drAdjustedLocalLawLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  unifiedAdjustedLocalLawEstimate problem.gammaDepth
    (problem.nodeProxyLoss theta) (problem.nodeOracleLoss theta)
    (problem.nodeObserved theta) (problem.nodePi theta) T

/-! ## Local-law bias decomposition -/

/-- Node-level proxy-oracle local-law bias. Positive values mean the proxy law
loss is larger than the oracle law loss at that node. -/
def drNodeLocalLawBias
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  problem.nodeProxyLoss theta T - problem.nodeOracleLoss theta T

/-- The residual multiplier left after applying the node-oracle correction:
`1 - R/pi`. -/
def drNodeBiasResidualFactor
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  1 - (problem.nodeObserved theta T).toReal / problem.nodePi theta T

/-- Node residual bias after correction:
`(1 - R/pi) * (proxy law loss - oracle law loss)`. -/
def drNodeBiasResidual
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  drNodeBiasResidualFactor problem theta T *
    drNodeLocalLawBias problem theta T

/-- Discounted residual-bias aggregate for the corrected local-law channel. -/
def drDiscountedNodeBiasResidual
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  discountedTreeMetaLoss problem.gammaDepth
    (drNodeBiasResidual problem theta) T

/-- Bias-form local-law channel:
oracle local-law objective plus the discounted residual proxy-oracle bias. -/
def drBiasFormLocalLawLoss
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  discountedTrueNodeLawLoss problem.gammaDepth
      (problem.nodeOracleLoss theta) T +
    drDiscountedNodeBiasResidual problem theta T

theorem drNodeAIPWAdjustedLawLoss_eq_oracle_plus_biasResidual
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) :
    nodeAIPWAdjustedLawLoss (problem.nodeProxyLoss theta)
        (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
        (problem.nodePi theta) T =
      problem.nodeOracleLoss theta T +
        drNodeBiasResidual problem theta T := by
  unfold nodeAIPWAdjustedLawLoss drNodeBiasResidual
    drNodeBiasResidualFactor drNodeLocalLawBias
  unfold DSL.designAdjustedOutcome DSL.SamplingIndicator.toReal
  cases hObs : problem.nodeObserved theta T <;> simp [hObs]
  all_goals ring

theorem drAdjustedLocalLawLoss_eq_biasForm
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) :
    drAdjustedLocalLawLoss problem theta T =
      drBiasFormLocalLawLoss problem theta T := by
  induction T with
  | leaf a =>
      unfold drAdjustedLocalLawLoss unifiedAdjustedLocalLawEstimate
        discountedAIPWNodeLawLoss drBiasFormLocalLawLoss
        discountedTrueNodeLawLoss drDiscountedNodeBiasResidual
      simp [drNodeAIPWAdjustedLawLoss_eq_oracle_plus_biasResidual]
  | node TL TR ihL ihR =>
      unfold drAdjustedLocalLawLoss unifiedAdjustedLocalLawEstimate
        discountedAIPWNodeLawLoss drBiasFormLocalLawLoss
        discountedTrueNodeLawLoss drDiscountedNodeBiasResidual at ihL ihR ⊢
      simp [drNodeAIPWAdjustedLawLoss_eq_oracle_plus_biasResidual]
      rw [ihL, ihR]
      ring

/-! ## Nominal root/local objective -/

/-- Root/local part of the full objective for an explicit local-law scalar. -/
def drRootLocalValueWithLocalLaw
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (lawLoss : ℝ) : ℝ :=
  nominalRootLocalObjective problem.Lambda
    (drRootChannelLoss problem theta) lawLoss

/-- Full objective for an explicit local-law scalar. -/
def drMinimizationValueWithLocalLaw
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (lawLoss : ℝ) : ℝ :=
  drOracleGapChannelLoss problem theta +
    drRootLocalValueWithLocalLaw problem theta lawLoss

/-- Full objective for explicit envelope and local-law scalars.
The envelope is retained only as a statement convenience for residual-envelope
lemmas; it does not alter the root/local weight. -/
def drMinimizationValueWithEnvelopeAndLocalLaw
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (_envelope lawLoss : ℝ) : ℝ :=
  drMinimizationValueWithLocalLaw problem theta lawLoss

/-- Exact doubly robust minimization objective:
proxy-oracle gap plus root/local lambda mixing, where the local-law loss is the
discounted AIPW/DSL adjusted node-law estimate. -/
def drMinimizationValue
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  drMinimizationValueWithLocalLaw problem theta
    (drAdjustedLocalLawLoss problem theta T)

/-- Full objective written directly in bias form. -/
def drBiasFormMinimizationValue
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) : ℝ :=
  drMinimizationValueWithLocalLaw problem theta
    (drBiasFormLocalLawLoss problem theta T)

theorem drMinimizationValue_eq_oracleGap_plus_nominal
    {problem : DoublyRobustMinimizationProblem Param Strings}
    {theta : Param} {T : BinTree Strings} :
    drMinimizationValue problem theta T =
      drOracleGapChannelLoss problem theta +
        nominalRootLocalObjective problem.Lambda
          (drRootChannelLoss problem theta)
          (drAdjustedLocalLawLoss problem theta T) := by
  rfl

theorem drMinimizationValue_eq_biasForm
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings) :
    drMinimizationValue problem theta T =
      drBiasFormMinimizationValue problem theta T := by
  unfold drMinimizationValue drBiasFormMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_biasForm]

/-! ## Local-law endpoints inside the full objective -/

theorem drAdjustedLocalLawLoss_eq_proxy_of_all_unsampled
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = false) :
    drAdjustedLocalLawLoss problem theta T =
      discountedSurrogateNodeLawLoss problem.gammaDepth
        (problem.nodeProxyLoss theta) T := by
  unfold drAdjustedLocalLawLoss
  exact unifiedAdjustedLocalLawEstimate_eq_proxy_of_all_unsampled
    problem.gammaDepth (problem.nodeProxyLoss theta)
    (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
    (problem.nodePi theta) T hObserved

theorem drMinimizationValue_eq_proxy_of_all_unsampled
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = false) :
    drMinimizationValue problem theta T =
      drMinimizationValueWithLocalLaw problem theta
        (discountedSurrogateNodeLawLoss problem.gammaDepth
          (problem.nodeProxyLoss theta) T) := by
  unfold drMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_proxy_of_all_unsampled
    problem theta T hObserved]

theorem drAdjustedLocalLawLoss_eq_oracle_of_all_sampled_pi_one
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = true)
    (hPi : ∀ U : BinTree Strings, problem.nodePi theta U = 1) :
    drAdjustedLocalLawLoss problem theta T =
      discountedTrueNodeLawLoss problem.gammaDepth
        (problem.nodeOracleLoss theta) T := by
  unfold drAdjustedLocalLawLoss
  exact unifiedAdjustedLocalLawEstimate_eq_oracle_of_all_sampled_pi_one
    problem.gammaDepth (problem.nodeProxyLoss theta)
    (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
    (problem.nodePi theta) T hObserved hPi

theorem drMinimizationValue_eq_oracle_of_all_sampled_pi_one
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hObserved :
      ∀ U : BinTree Strings, problem.nodeObserved theta U = true)
    (hPi : ∀ U : BinTree Strings, problem.nodePi theta U = 1) :
    drMinimizationValue problem theta T =
      drMinimizationValueWithLocalLaw problem theta
        (discountedTrueNodeLawLoss problem.gammaDepth
          (problem.nodeOracleLoss theta) T) := by
  unfold drMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_oracle_of_all_sampled_pi_one
    problem theta T hObserved hPi]

theorem drAdjustedLocalLawLoss_eq_oracle_of_exact_proxy
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hExact :
      ∀ U : BinTree Strings,
        problem.nodeProxyLoss theta U = problem.nodeOracleLoss theta U) :
    drAdjustedLocalLawLoss problem theta T =
      discountedTrueNodeLawLoss problem.gammaDepth
        (problem.nodeOracleLoss theta) T := by
  unfold drAdjustedLocalLawLoss
  exact unifiedAdjustedLocalLawEstimate_eq_oracle_of_exact_proxy
    problem.gammaDepth (problem.nodeProxyLoss theta)
    (problem.nodeOracleLoss theta) (problem.nodeObserved theta)
    (problem.nodePi theta) T hExact

theorem drMinimizationValue_eq_oracle_of_exact_proxy
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param) (T : BinTree Strings)
    (hExact :
      ∀ U : BinTree Strings,
        problem.nodeProxyLoss theta U = problem.nodeOracleLoss theta U) :
    drMinimizationValue problem theta T =
      drMinimizationValueWithLocalLaw problem theta
        (discountedTrueNodeLawLoss problem.gammaDepth
          (problem.nodeOracleLoss theta) T) := by
  unfold drMinimizationValue
  rw [drAdjustedLocalLawLoss_eq_oracle_of_exact_proxy
    problem theta T hExact]

/-! ## Propensity-residual objective simplifications -/

theorem drMinimizationValueWithEnvelope_eq_no_propensity_residual_of_matched
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param)
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss lawLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hMatch : piTrue = piUsed) :
    drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack
          (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss))
        lawLoss =
      drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack 0)
        lawLoss := by
  rfl

theorem drMinimizationValueWithEnvelope_eq_no_propensity_residual_of_exact_proxy
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param)
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss lawLoss : ℝ)
    (hExact : proxyLoss = oracleLoss) :
    drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack
          (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss))
        lawLoss =
      drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack 0)
        lawLoss := by
  rfl

theorem drMinimizationValueWithEnvelope_eq_no_propensity_residual_of_dr
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (theta : Param)
    (ipwEstimate gapMargin zScore se oracleSlack : ℝ)
    (piTrue piUsed proxyLoss oracleLoss lawLoss : ℝ)
    (hPiUsed : 0 < piUsed)
    (hDR : piTrue = piUsed ∨ proxyLoss = oracleLoss) :
    drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack
          (propensityMismatchResidual piTrue piUsed proxyLoss oracleLoss))
        lawLoss =
      drMinimizationValueWithEnvelopeAndLocalLaw problem theta
        (adjustedLocalLawEnvelope ipwEstimate gapMargin zScore se
          oracleSlack 0)
        lawLoss := by
  rfl

/-! ## Feasible minimization -/

/-- Feasible minimizer for the full doubly robust objective. -/
def IsDRLocalLawMinimizer
    (problem : DoublyRobustMinimizationProblem Param Strings)
    (feasible : Set Param)
    (theta : Param) (T : BinTree Strings) : Prop :=
  IsMergeableProjection feasible
    (fun theta' : Param => drMinimizationValue problem theta' T) theta

/-- Projection-style alias for later mergeability-gap transfer results. -/
abbrev IsDRLocalLawProjection :=
  @IsDRLocalLawMinimizer

theorem drLocalLawMinimizer_mem
    {problem : DoublyRobustMinimizationProblem Param Strings}
    {feasible : Set Param} {theta : Param} {T : BinTree Strings}
    (hmin : IsDRLocalLawMinimizer problem feasible theta T) :
    theta ∈ feasible :=
  hmin.mem

theorem drLocalLawMinimizer_value_le
    {problem : DoublyRobustMinimizationProblem Param Strings}
    {feasible : Set Param} {theta theta' : Param} {T : BinTree Strings}
    (hmin : IsDRLocalLawMinimizer problem feasible theta T)
    (hfeasible : theta' ∈ feasible) :
    drMinimizationValue problem theta T ≤
      drMinimizationValue problem theta' T :=
  hmin.risk_le theta' hfeasible

end FormalProofs.OPT

end

end
