import FormalProofs.OPT.LocalLaws
import FormalProofs.OPT.TreeProperties

/-!
# FormalProofs/PreservationTheorems.lean

## Paper Reference: Section 3 (Theorems & Corollaries)

This file contains the core preservation theorems from Section 3 of the paper:

### Key Theorems

| Paper | Lean Theorem | Description |
|-------|--------------|-------------|
| **Theorem 1** (Inductive Preservation) | `one_pass` | L1 + L2 → zero distortion at root |
| **Corollary 1** (Schedule Invariance) | `schedule_invariance` | Same expected oracle |
| **Corollary 2** (Fold-of-Folds) | `fold_of_folds` | Two-level hierarchical plans preserve oracle |

### Technical Lemmas

- `nodewise_preservation`: Distortion is 0 at every subtree (inductive foundation)
- `reduce_support_in_range`: Support of hierarchical reduction is in range(g)
- `ZR_support_in_range`: Support of multi-round reduction is in range(g)
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Nodewise Preservation
-/

/-- If L1 and L2 hold for T, then for any subtree u of T, the expected distortion is 0 -/
theorem nodewise_preservation (g : Summarizer Strings) (T : BinTree Strings) (u : BinTree Strings) (fstar : Strings → Y)
  (h_sub : u ∈ subtrees T) (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
  Egu g u (fun z => D fstar z (S u)) = 0 := by
    revert h_sub h1 h2;
    rintro hu h1 h2;
    induction' T with T_L T_R hT_L hT_R generalizing u;
    · cases hu;
      · convert h1 T_L ?_;
        simp only [leaves, List.mem_singleton];
      · contradiction;
    · -- u is in subtrees of T_R or hT_L
      by_cases hu_T_R : u ∈ subtrees T_R;
      · apply hT_R u hu_T_R;
        · intro b hb;
          apply h1 b;
          exact List.mem_append_left _ hb;
        · intro p hp;
          apply h2;
          exact List.mem_cons_of_mem _ ( List.mem_append_left _ hp );
      · by_cases hu_hT_L : u ∈ subtrees hT_L;
        · rename_i ih;
          apply ih u hu_hT_L;
          · intro b hb;
            exact h1 b ( by
              exact List.mem_append_right _ hb );
          · intro p hp;
            apply h2;
            simp [internal_nodes] at hp ⊢;
            exact Or.inr <| Or.inr hp;
        · cases hu;
          · exact h2 ( T_R, hT_L ) ( by simp +decide [ internal_nodes ] );
          · cases List.mem_append.mp ‹_› <;> tauto

/-!
## One-Pass Preservation
-/

/-- **One-Pass Preservation (Paper: Theorem 1)**

**Paper Reference:** Section 3, Theorem 1 (Inductive Preservation)

If L1 (leaf sufficiency) and L2 (merge consistency) hold at every realized node,
then the expected distortion at the root is 0:
  `E[D(Z^(1), x)] = 0`

This is the base case for multi-round preservation (Theorem 2 = `multi_round_bounded`). -/
theorem one_pass (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (fstar : Strings → Y)
  (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
  Egu g (root T) (fun z => D fstar z x) = 0 := by
    have h_root : root T ∈ subtrees T := by
      cases T <;> tauto;
    have := nodewise_preservation g T ( root T ) fstar h_root h1 h2; aesop;

/-!
## Single-Leaf Degenerate Case
-/

/-- **L2 is vacuously true for a single-leaf tree.**

When the tree is just `BinTree.leaf b`, `internal_nodes T = []`,
so the universal quantifier in L2 ranges over an empty set. -/
theorem L2_vacuous_of_leaf (g : Summarizer Strings) (b : Strings) (fstar : Strings → Y) :
  L2 g (BinTree.leaf b) fstar := by
    intro p hp; simp [internal_nodes] at hp

/-- **Single-leaf one-pass preservation.**

For a single-leaf tree, only L1 is needed: L2 holds vacuously,
so if the leaf encoder preserves the oracle, root distortion is zero.
This is the formal basis for the tree–FNO parity claim: when the
tree has one leaf spanning the full document, it computes `g(b)` and
the preservation guarantee depends only on `g` being L1-faithful. -/
theorem single_leaf_one_pass (g : Summarizer Strings) (b : Strings) (fstar : Strings → Y)
  (h1 : L1 g (BinTree.leaf b) fstar) :
  Egu g (root (BinTree.leaf b)) (fun z => D fstar z (S (BinTree.leaf b))) = 0 :=
    one_pass g (BinTree.leaf b) (S (BinTree.leaf b)) fstar rfl h1 (L2_vacuous_of_leaf g b fstar)

/-- **Single-leaf reduction is just the encoder.**

`reduce g (BinTree.leaf b) = g b` holds definitionally. -/
theorem single_leaf_reduces_to_encoder (g : Summarizer Strings) (b : Strings) :
  reduce g (BinTree.leaf b) = g b := by rfl

/-!
## Schedule Invariance
-/

/-- **Schedule Invariance (Paper: Corollary 1)**

**Paper Reference:** Section 3, Corollary 1

For any fixed partition, every full binary tree on the leaves yields the same
expected oracle whenever L1 and L2 hold on all realized edges. In particular,
balanced reductions and daisy chains are interchangeable.

Both trees produce expected distortion 0, so they are trivially equal. -/
theorem schedule_invariance (g : Summarizer Strings) (T T' : BinTree Strings) (fstar : Strings → Y)
  (_h_l : leaves T = leaves T') (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h1' : L1 g T' fstar) (h2' : L2 g T' fstar) :
  Egu g (root T) (fun z => D fstar z (S T)) = Egu g (root T') (fun z => D fstar z (S T')) := by
    have h_eq_dist : Egu g (root T) (fun z => D fstar z (S T)) = 0 ∧ Egu g (root T') (fun z => D fstar z (S T')) = 0 := by
      exact ⟨ one_pass g T ( S T ) fstar rfl h1 h2, one_pass g T' ( S T' ) fstar rfl h1' h2' ⟩;
    rw [ h_eq_dist.1, h_eq_dist.2 ]

/-!
## Fold-of-Folds Invariance
-/

/-- **Fold-of-Folds Invariance (Paper: Corollary 2)**

**Paper Reference:** Section 3, Corollary 2

Consider any two-level plan that first reduces contiguous "folds" and then
reduces the fold summaries. If L1 and L2 hold on every realized edge and
L3 holds on the intermediate summaries, the composite schedule preserves
f* in expectation regardless of the within-fold or over-fold parenthesizations.

Note: This is a specialized version of `one_pass`. The main theorem `multi_round_bounded`
provides the full multi-round guarantee using L1, L2, and L3. -/
theorem fold_of_folds (g : Summarizer Strings) (T_comp : BinTree Strings) (x : Strings) (fstar : Strings → Y)
  (hp : S T_comp = x) (h1 : L1 g T_comp fstar) (h2 : L2 g T_comp fstar) (_h3 : L3 g fstar) :
  Egu g (root T_comp) (fun z => D fstar z x) = 0 := by
    exact one_pass g T_comp x fstar hp h1 h2

/-!
## Support Properties
-/

/-- Any string in the support of reduce g T is in the range of g -/
theorem reduce_support_in_range (g : Summarizer Strings) (T : BinTree Strings) :
  ∀ z ∈ (reduce g T).support, InRange g z := by
    intro z hz;
    induction' T with T_L T_R ih_L ih_R;
    · exact ⟨ T_L, hz ⟩;
    · unfold reduce at hz;
      contrapose! hz; simp_all +decide [ PMF.support ] ;
      intro x hx y hy; unfold InRange at hz; aesop;

/-- For R >= 1, any string in the support of ZR is in the range of g -/
theorem ZR_support_in_range (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings) (hR : R ≥ 1) :
  ∀ z ∈ (ZR g x R T).support, InRange g z := by
    induction' hR with R hR ih generalizing x T;
    · simp only [ZR]; exact reduce_support_in_range g T;
    · intro z hz;
      obtain ⟨y, hy⟩ : ∃ y ∈ (ZR g x R T).support, z ∈ (g y).support := by
        have h_bind : (ZR g x (Nat.succ R) T) = (ZR g x R T).bind g := by
          cases R <;> tauto;
        contrapose! hz; aesop;
      exact ⟨ y, hy.2 ⟩

end
