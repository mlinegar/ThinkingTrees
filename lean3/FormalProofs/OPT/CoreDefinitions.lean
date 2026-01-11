/-
FormalProofs/CoreDefinitions.lean

Core definitions for the summarization formalization:
- Summarizer type and expectation operator Eg
- Distortion measure D
- Binary tree type and operations (S, reduce, ZR)
- Tree accessors (leaves, internal_nodes, root, subtrees)
-/

import Mathlib

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Summarizer and Expectation
-/

/-- A summarizer maps elements to probability distributions over the same type -/
def Summarizer (α : Type*) := α → PMF α

/-- Expectation of f under the summarizer g applied at x -/
noncomputable def Eg {α : Type*} (g : Summarizer α) (f : α → ℝ) (x : α) : ℝ :=
  ∑' z, (g x z).toReal * f z

/-!
## Distortion
-/

variable {Y : Type*} [PseudoMetricSpace Y]

/-- Distortion: distance between oracle values -/
def D {α : Type*} (fstar : α → Y) (z x : α) : ℝ :=
  dist (fstar z) (fstar x)

/-!
## Binary Tree and Hierarchical Reduction
-/

/-- Binary tree with values at leaves -/
inductive BinTree (α : Type*) : Type _
| leaf : α → BinTree α
| node : BinTree α → BinTree α → BinTree α

variable {Strings : Type*} [Monoid Strings]

/-- Realize a tree as a string by multiplying leaves left-to-right -/
def S : BinTree Strings → Strings
| BinTree.leaf b => b
| BinTree.node T_L T_R => S T_L * S T_R

variable (g : Summarizer Strings)

/-- Hierarchical reduction: summarize tree bottom-up -/
def reduce : BinTree Strings → PMF Strings
| BinTree.leaf b => g b
| BinTree.node T_L T_R => (reduce T_L).bind (fun s_L => (reduce T_R).bind (fun s_R => g (s_L * s_R)))

/-- Multi-round reduction: apply summarization R times -/
def ZR (x : Strings) (R : ℕ) (T : BinTree Strings) : PMF Strings :=
  match R with
  | 0 => PMF.pure x -- Base case (paper uses 1-indexing)
  | 1 => reduce g T
  | n + 1 => (ZR x n T).bind g

/-!
## Tree Accessors
-/

/-- The root of a tree (identity function, for clarity) -/
def root {α : Type*} (T : BinTree α) : BinTree α := T

/-- List of all leaf values in left-to-right order -/
def leaves {α : Type*} : BinTree α → List α
| BinTree.leaf b => [b]
| BinTree.node T_L T_R => leaves T_L ++ leaves T_R

/-- List of all internal nodes as (left, right) subtree pairs -/
def internal_nodes {α : Type*} : BinTree α → List (BinTree α × BinTree α)
| BinTree.leaf _ => []
| BinTree.node T_L T_R => (T_L, T_R) :: (internal_nodes T_L ++ internal_nodes T_R)

/-- Count leaves in a binary tree -/
def numLeaves {α : Type*} : BinTree α → ℕ
| BinTree.leaf _ => 1
| BinTree.node T_L T_R => numLeaves T_L + numLeaves T_R

/-- Count internal nodes in a binary tree -/
def numInternalNodes {α : Type*} : BinTree α → ℕ
| BinTree.leaf _ => 0
| BinTree.node T_L T_R => 1 + numInternalNodes T_L + numInternalNodes T_R

/-- List of all subtrees (including the tree itself) -/
def subtrees {α : Type*} : BinTree α → List (BinTree α)
| BinTree.leaf b => [BinTree.leaf b]
| BinTree.node T_L T_R => BinTree.node T_L T_R :: (subtrees T_L ++ subtrees T_R)

end
