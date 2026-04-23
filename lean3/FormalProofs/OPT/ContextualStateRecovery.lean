import FormalProofs.OPT.CoreDefinitions

/-!
# FormalProofs/OPT/ContextualStateRecovery.lean

Exact-state recovery for subtree-indexed / context-conditioned operator families.

This file captures the setting where the learned leaf and merge operators are
not globally shared, but instead depend on the current node context. Rather
than introducing a separate node-ID system, we use the current subtree itself
as the canonical index and then derive the more ergonomic context-conditioned
form as a corollary.

The guiding use case is a C-tree whose semantic object is a canonical latent
state `feature (S T)`, while node-local learned functions implement:

- leaf extraction conditioned on node context, and
- merge / reconciliation conditioned on node context.

If those local operators are exact on the canonical latent state, then the full
tree fold recovers that state exactly by induction.
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

section SubtreeIndexedState

variable {Strings : Type*} [Monoid Strings]
variable {State β : Type*}

/-- Exact state fold where the leaf and merge operators may depend on the
current subtree. The subtree itself is the node index. -/
def subtreeIndexedStateFold
    (encodeAt : BinTree Strings → Strings → State)
    (mergeAt : BinTree Strings → State → State → State) :
    BinTree Strings → State
  | BinTree.leaf b =>
      encodeAt (BinTree.leaf b) b
  | BinTree.node TL TR =>
      mergeAt (BinTree.node TL TR)
        (subtreeIndexedStateFold encodeAt mergeAt TL)
        (subtreeIndexedStateFold encodeAt mergeAt TR)

/-- Exact leaf recovery for subtree-indexed state operators. -/
def SubtreeIndexedLeafExact
    (encodeAt : BinTree Strings → Strings → State)
    (feature : Strings → State) : Prop :=
  ∀ b, encodeAt (BinTree.leaf b) b = feature b

/-- Exact merge recovery for subtree-indexed state operators, assuming child
states are already canonical. -/
def SubtreeIndexedMergeExact
    (mergeAt : BinTree Strings → State → State → State)
    (feature : Strings → State) : Prop :=
  ∀ TL TR,
    mergeAt (BinTree.node TL TR) (feature (S TL)) (feature (S TR)) =
      feature (S (BinTree.node TL TR))

/-- If node-indexed leaf and merge operators are exact on the canonical latent
state, the whole subtree-indexed fold recovers that state exactly. -/
theorem subtreeIndexedStateFold_eq_feature
    (encodeAt : BinTree Strings → Strings → State)
    (mergeAt : BinTree Strings → State → State → State)
    (feature : Strings → State)
    (h_leaf : SubtreeIndexedLeafExact encodeAt feature)
    (h_merge : SubtreeIndexedMergeExact mergeAt feature)
    (T : BinTree Strings) :
    subtreeIndexedStateFold encodeAt mergeAt T = feature (S T) := by
  induction T with
  | leaf b =>
      simpa [subtreeIndexedStateFold, SubtreeIndexedLeafExact] using h_leaf b
  | node TL TR ihL ihR =>
      simpa [subtreeIndexedStateFold, S, ihL, ihR] using h_merge TL TR

/-- Any downstream readout of an exact subtree-indexed state fold is preserved
exactly on the tree. -/
theorem subtreeIndexedStateUtility_exact_on_tree
    (encodeAt : BinTree Strings → Strings → State)
    (mergeAt : BinTree Strings → State → State → State)
    (feature : Strings → State)
    (h_leaf : SubtreeIndexedLeafExact encodeAt feature)
    (h_merge : SubtreeIndexedMergeExact mergeAt feature)
    (u : State → β)
    (T : BinTree Strings) :
    u (subtreeIndexedStateFold encodeAt mergeAt T) = u (feature (S T)) := by
  rw [subtreeIndexedStateFold_eq_feature
    (encodeAt := encodeAt) (mergeAt := mergeAt) (feature := feature)
    h_leaf h_merge T]

end SubtreeIndexedState

section ContextConditionedState

variable {Strings : Type*} [Monoid Strings]
variable {Context State β : Type*}

/-- Context-conditioned state fold obtained by evaluating a context map on the
current subtree and passing that context to shared operator families. -/
def contextConditionedStateFold
    (ctx : BinTree Strings → Context)
    (encode : Context → Strings → State)
    (merge : Context → State → State → State) :
    BinTree Strings → State :=
  subtreeIndexedStateFold
    (fun T b => encode (ctx T) b)
    (fun T sL sR => merge (ctx T) sL sR)

/-- Exact leaf recovery for context-conditioned state operators. -/
def ContextConditionedLeafExact
    (ctx : BinTree Strings → Context)
    (encode : Context → Strings → State)
    (feature : Strings → State) : Prop :=
  ∀ b, encode (ctx (BinTree.leaf b)) b = feature b

/-- Exact merge recovery for context-conditioned state operators. -/
def ContextConditionedMergeExact
    (ctx : BinTree Strings → Context)
    (merge : Context → State → State → State)
    (feature : Strings → State) : Prop :=
  ∀ TL TR,
    merge (ctx (BinTree.node TL TR)) (feature (S TL)) (feature (S TR)) =
      feature (S (BinTree.node TL TR))

/-- Exact recovery theorem for operator families of the form
`encode (ctx T)` / `merge (ctx T)`. This is the direct Lean version of
node-context-conditioned learned `f` / `g` functions. -/
theorem contextConditionedStateFold_eq_feature
    (ctx : BinTree Strings → Context)
    (encode : Context → Strings → State)
    (merge : Context → State → State → State)
    (feature : Strings → State)
    (h_leaf : ContextConditionedLeafExact ctx encode feature)
    (h_merge : ContextConditionedMergeExact ctx merge feature)
    (T : BinTree Strings) :
    contextConditionedStateFold ctx encode merge T = feature (S T) := by
  exact subtreeIndexedStateFold_eq_feature
    (encodeAt := fun T b => encode (ctx T) b)
    (mergeAt := fun T sL sR => merge (ctx T) sL sR)
    (feature := feature)
    (h_leaf := h_leaf)
    (h_merge := h_merge)
    T

/-- Any downstream readout of an exact context-conditioned state fold is
preserved exactly on the tree. -/
theorem contextConditionedStateUtility_exact_on_tree
    (ctx : BinTree Strings → Context)
    (encode : Context → Strings → State)
    (merge : Context → State → State → State)
    (feature : Strings → State)
    (h_leaf : ContextConditionedLeafExact ctx encode feature)
    (h_merge : ContextConditionedMergeExact ctx merge feature)
    (u : State → β)
    (T : BinTree Strings) :
    u (contextConditionedStateFold ctx encode merge T) = u (feature (S T)) := by
  rw [contextConditionedStateFold_eq_feature
    (ctx := ctx) (encode := encode) (merge := merge) (feature := feature)
    h_leaf h_merge T]

end ContextConditionedState

end FormalProofs.OPT
