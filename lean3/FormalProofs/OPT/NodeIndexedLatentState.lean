import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.OracleFiberObjectives

/-!
# FormalProofs/OPT/NodeIndexedLatentState.lean

Exact tree-induction lemmas for node-indexed / context-conditioned latent-state
families.

This file formalizes the setting where:

- leaves use learned extractors `f_v`,
- internal nodes use learned merge/reconcile operators `g_v`, and
- both families are indexed by the realized tree node.

The main theorem says that if every leaf extractor is exact for the canonical
span-state map `Φ`, and every internal merge operator is exact on canonical
child states, then the bottom-up node-indexed evaluation recovers `Φ (S T)`
exactly on every tree.

This is the Lean version of the statement that an exact node-indexed family of
learned operators computes the same latent object as the global tree state.
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

variable {Strings : Type*} [Monoid Strings]
variable {State β : Type*}

/-- Leaf-level state extractor family. -/
abbrev LeafStateFamily (Strings State : Type*) := Strings → State

/-- Internal-node merge family indexed by the realized subtree. -/
abbrev MergeStateFamily (Strings State : Type*) [Monoid Strings] :=
  BinTree Strings → State → State → State

/-- Bottom-up evaluation of a node-indexed latent-state family on a tree. -/
def nodeIndexedStateEval
    (leafFn : LeafStateFamily Strings State)
    (mergeFn : MergeStateFamily Strings State) :
    BinTree Strings → State
  | BinTree.leaf b => leafFn b
  | BinTree.node TL TR =>
      mergeFn (BinTree.node TL TR)
        (nodeIndexedStateEval leafFn mergeFn TL)
        (nodeIndexedStateEval leafFn mergeFn TR)

/-- Exact leaf-state condition: each learned leaf extractor equals the canonical
span-state map on realized leaves. -/
def NodeIndexedLeafExact
    (leafFn : LeafStateFamily Strings State)
    (feature : Strings → State) : Prop :=
  ∀ b, leafFn b = feature b

/-- Exact merge-state condition: each node-indexed merge operator maps canonical
child states to the canonical parent-span state. -/
def NodeIndexedMergeExact
    (mergeFn : MergeStateFamily Strings State)
    (feature : Strings → State) : Prop :=
  ∀ TL TR,
    mergeFn (BinTree.node TL TR) (feature (S TL)) (feature (S TR)) =
      feature (S (BinTree.node TL TR))

/-- Exact node-indexed state evaluation recovers the canonical span state by
tree induction. -/
theorem nodeIndexedStateEval_eq_feature_of_exact
    (leafFn : LeafStateFamily Strings State)
    (mergeFn : MergeStateFamily Strings State)
    (feature : Strings → State)
    (h_leaf : NodeIndexedLeafExact leafFn feature)
    (h_merge : NodeIndexedMergeExact mergeFn feature)
    (T : BinTree Strings) :
    nodeIndexedStateEval leafFn mergeFn T = feature (S T) := by
  induction T with
  | leaf b =>
      simpa [nodeIndexedStateEval, NodeIndexedLeafExact] using h_leaf b
  | node TL TR ihL ihR =>
      simpa [nodeIndexedStateEval, S, ihL, ihR] using h_merge TL TR

/-- Any downstream readout of an exact node-indexed latent-state family agrees
with the readout of the canonical span-state map on the full tree span. -/
theorem nodeIndexedStateUtility_exact_on_tree
    (leafFn : LeafStateFamily Strings State)
    (mergeFn : MergeStateFamily Strings State)
    (feature : Strings → State)
    (h_leaf : NodeIndexedLeafExact leafFn feature)
    (h_merge : NodeIndexedMergeExact mergeFn feature)
    (u : State → β)
    (T : BinTree Strings) :
    u (nodeIndexedStateEval leafFn mergeFn T) = u (feature (S T)) := by
  rw [nodeIndexedStateEval_eq_feature_of_exact
    (leafFn := leafFn) (mergeFn := mergeFn) (feature := feature)
    h_leaf h_merge T]

/-- If the node-indexed merge family is in fact a constant merge operator, the
node-indexed evaluator collapses to the standard exact merge-fold. -/
theorem nodeIndexedStateEval_eq_mergeFold_of_constant_merge
    (leafFn : LeafStateFamily Strings State)
    (merge : State → State → State)
    (mergeFn : MergeStateFamily Strings State)
    (h_mergeFn : ∀ T sL sR, mergeFn T sL sR = merge sL sR)
    (T : BinTree Strings) :
    nodeIndexedStateEval leafFn mergeFn T = mergeFold leafFn merge T := by
  induction T with
  | leaf b =>
      rfl
  | node TL TR ihL ihR =>
      simp [nodeIndexedStateEval, mergeFold, ihL, ihR, h_mergeFn]

/-- Specialization of the exact mergeable-state theorem to node-indexed families
whose realized operators collapse to one exact mergeable latent-state route. -/
theorem nodeIndexedStateUtility_exact_on_tree_of_mergeable_feature
    (leafFn : LeafStateFamily Strings State)
    (mergeFn : MergeStateFamily Strings State)
    (feature : Strings → State)
    (merge : State → State → State)
    (h_leaf : ∀ x, leafFn x = feature x)
    (h_mergeFn : ∀ T sL sR, mergeFn T sL sR = merge sL sR)
    (h_feature_merge : ∀ x y, merge (feature x) (feature y) = feature (x * y))
    (u : State → β)
    (T : BinTree Strings) :
    u (nodeIndexedStateEval leafFn mergeFn T) = u (feature (S T)) := by
  rw [nodeIndexedStateEval_eq_mergeFold_of_constant_merge
    (leafFn := leafFn) (merge := merge) (mergeFn := mergeFn) h_mergeFn T]
  exact mergeableStateUtility_exact_on_tree
    (encode := leafFn) (merge := merge) (feature := feature)
    h_leaf h_feature_merge u T

section Approximate

variable {Y : Type*} [BoundedMetricSpace Y]
variable [BoundedPseudoMetricSpace State]

/-- Approximate theorem-backed transport specialized to an exact node-indexed
latent-state family. If the learned per-node state operators recover the
canonical latent state exactly, then the existing approximate-local-law /
feature-fiber transport theorem controls the root-state utility relative to the
node-indexed state itself. -/
theorem expected_nodeIndexedStateUtility_bound_of_exactNodeIndexed_and_approxBacked
    (fstar : Strings → Y)
    (feature : Strings → State)
    (leafFn : LeafStateFamily Strings State)
    (mergeFn : MergeStateFamily Strings State)
    (u : OracleUtility2 State)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K ε_fiber L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hNodeLeaf : NodeIndexedLeafExact leafFn feature)
    (hNodeMerge : NodeIndexedMergeExact mergeFn feature)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T) (fun z => u (feature z) (nodeIndexedStateEval leafFn mergeFn T)) -
        u (nodeIndexedStateEval leafFn mergeFn T) (nodeIndexedStateEval leafFn mergeFn T)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε_fiber : ℝ) := by
  have hNodeEq : nodeIndexedStateEval leafFn mergeFn T = feature x := by
    calc
      nodeIndexedStateEval leafFn mergeFn T = feature (S T) :=
        nodeIndexedStateEval_eq_feature_of_exact
          (leafFn := leafFn) (mergeFn := mergeFn) (feature := feature)
          hNodeLeaf hNodeMerge T
      _ = feature x := by simpa [hp]
  have hMain :=
    expected_utility_bound_approx_fiber
      (fstar := fstar)
      (feature := feature)
      (featureHat := feature)
      (u := u)
      (g := g) (x := x) (R := R) (T := T)
      (K := K) (ε_fiber := ε_fiber) (L1 := L1) (L2 := L2) (U := U)
      hp hApprox hR hApproxRecover hFeatureLip hL1 hL2 hU hbound hbound_global h_mono
  simpa [hNodeEq, dist_self] using hMain

end Approximate

end FormalProofs.OPT
