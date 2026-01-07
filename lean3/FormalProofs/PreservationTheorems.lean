/-
FormalProofs/PreservationTheorems.lean

Core preservation theorems:
- Nodewise preservation: L1 & L2 → distortion 0 on subtrees
- One-pass: L1 & L2 → distortion 0 at root
- Schedule invariance: Trees with same leaves have equal distortion
- Fold-of-folds: L1 & L2 & L3 → distortion 0
- Support properties for reduce and ZR
-/

import FormalProofs.LocalLaws
import FormalProofs.TreeProperties

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

/-- If L1 and L2 hold, then the expected distortion of the reduction of the root is 0 -/
theorem one_pass (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (fstar : Strings → Y)
  (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
  Egu g (root T) (fun z => D fstar z x) = 0 := by
    have h_root : root T ∈ subtrees T := by
      cases T <;> tauto;
    have := nodewise_preservation g T ( root T ) fstar h_root h1 h2; aesop;

/-!
## Schedule Invariance
-/

/-- If two trees have the same leaves and L1/L2 hold for both, their expected distortions are equal (both 0) -/
theorem schedule_invariance (g : Summarizer Strings) (T T' : BinTree Strings) (fstar : Strings → Y)
  (_h_l : leaves T = leaves T') (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h1' : L1 g T' fstar) (h2' : L2 g T' fstar) :
  Egu g (root T) (fun z => D fstar z (S T)) = Egu g (root T') (fun z => D fstar z (S T')) := by
    have h_eq_dist : Egu g (root T) (fun z => D fstar z (S T)) = 0 ∧ Egu g (root T') (fun z => D fstar z (S T')) = 0 := by
      exact ⟨ one_pass g T ( S T ) fstar rfl h1 h2, one_pass g T' ( S T' ) fstar rfl h1' h2' ⟩;
    rw [ h_eq_dist.1, h_eq_dist.2 ]

/-!
## Fold-of-Folds Invariance
-/

/-- If L1, L2, and L3 hold, then the expected distortion is 0 -/
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
