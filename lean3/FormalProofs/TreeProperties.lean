/-
FormalProofs/TreeProperties.lean

Properties and lemmas about binary trees:
- Relationship between list lengths and counts
- Structural properties of binary trees
-/

import FormalProofs.CoreDefinitions

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

/-!
## Tree Count Lemmas
-/

/-- List length equals count for leaves -/
lemma leaves_length_eq {α : Type*} (T : BinTree α) : (leaves T).length = numLeaves T := by
  induction T with
  | leaf b => simp [leaves, numLeaves]
  | node T_L T_R ih_L ih_R => simp [leaves, numLeaves, ih_L, ih_R]

/-- List length equals count for internal nodes -/
lemma internal_nodes_length_eq {α : Type*} (T : BinTree α) :
    (internal_nodes T).length = numInternalNodes T := by
  induction T with
  | leaf b => simp [internal_nodes, numInternalNodes]
  | node T_L T_R ih_L ih_R =>
    simp only [internal_nodes, List.length_cons, List.length_append, ih_L, ih_R, numInternalNodes]
    ring

/-- A binary tree always has at least one leaf -/
lemma numLeaves_pos {α : Type*} (T : BinTree α) : 0 < numLeaves T := by
  induction T with
  | leaf _ => simp [numLeaves]
  | node T_L T_R ih_L _ => simp [numLeaves]; omega

/-- Standard binary tree property: internal nodes = leaves - 1 -/
lemma internal_eq_leaves_minus_one {α : Type*} (T : BinTree α) :
    numInternalNodes T = numLeaves T - 1 := by
  induction T with
  | leaf b => simp [numLeaves, numInternalNodes]
  | node T_L T_R ih_L ih_R =>
    simp only [numLeaves, numInternalNodes]
    have h1 : 0 < numLeaves T_L := numLeaves_pos T_L
    have h2 : 0 < numLeaves T_R := numLeaves_pos T_R
    omega

end
