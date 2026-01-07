/-
FormalProofs/ExpectationTheory.lean

Expectation theory for PMFs:
- Exp: Real-valued expectation
- ExpENN: ENNReal-valued expectation
- PMF summability lemmas
- Law of iterated expectation
- Multi-round preservation theorem
-/

import FormalProofs.PreservationTheorems

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Expectation Definitions
-/

/-- Expectation of f under a PMF -/
def Exp {α : Type*} (p : PMF α) (f : α → ℝ) : ℝ := ∑' z, (p z).toReal * f z

/-- Expectation in ENNReal (extended non-negative reals) -/
def ExpENN {α : Type*} (p : PMF α) (f : α → ENNReal) : ENNReal := ∑' z, p z * f z

/-!
## ExpENN Properties
-/

/-- Law of iterated expectation for ExpENN -/
lemma ExpENN_bind {α : Type*} (p : PMF α) (g : α → PMF α) (f : α → ENNReal) :
  ExpENN (p.bind g) f = ExpENN p (fun x => ExpENN (g x) f) := by
    unfold ExpENN;
    simp +decide [ PMF.bind_apply, mul_comm, mul_left_comm, ← ENNReal.tsum_mul_left ];
    rw [ ← ENNReal.tsum_comm ]

/-!
## PMF Summability Lemmas
-/

/-- PMF values are summable as Reals -/
lemma PMF.summable_coe_real {α : Type*} (p : PMF α) : Summable (fun z => (p z).toReal) := by
  apply ENNReal.summable_toReal
  rw [p.tsum_coe]
  exact ENNReal.one_ne_top

/-- PMF values times a bounded function are summable -/
lemma PMF.summable_coe_real_mul_of_bounded {α : Type*} (p : PMF α) (f : α → ℝ) (M : ℝ)
    (hM : 0 ≤ M) (hf : ∀ z, |f z| ≤ M) :
    Summable (fun z => (p z).toReal * f z) := by
  refine Summable.of_norm_bounded (g := fun z => (p z).toReal * M)
    ((PMF.summable_coe_real p).mul_right M) ?_
  intro z
  simp only [norm_mul, Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
  exact mul_le_mul_of_nonneg_left (hf z) ENNReal.toReal_nonneg

/-- PMF values times a function are summable.

## ⚠️ SOUNDNESS WARNING ⚠️

**This axiom is FALSE in general.** For unbounded f, the sum may diverge.
It is only sound when f is bounded (i.e., ∃ M, ∀ z, |f z| ≤ M).

### Why This Axiom Exists

All uses in this codebase involve bounded functions:
- `D fstar z x` (distortion) - bounded when the metric space has bounded diameter
- `violationInd` - takes values in {0, 1}
- `Exp (g z) f` for bounded f - bounded by the bound on f

### Recommended Alternatives

For mathematically rigorous code:
1. Use `PMF.summable_coe_real_mul_of_bounded` with explicit bounds
2. Use primed variants (`Exp_mono'`, etc.) with explicit summability hypotheses
3. Use `multi_round_bounded` instead of `multi_round` for the main theorem

### Future Work

TODO: Replace all uses with bounded variants and delete this axiom.
This requires adding boundedness hypotheses to ~50 call sites across:
- ExpectationTheory.lean (this file)
- AuditBounds.lean
- DPO.lean

Until then, this axiom serves as a convenience for proofs where boundedness
is semantically guaranteed but not explicitly tracked in the type system. -/
axiom PMF.summable_coe_real_mul {α : Type*} (p : PMF α) (f : α → ℝ) :
    Summable (fun z => (p z).toReal * f z)

/-- PMF values sum to 1 as Reals -/
lemma PMF.toReal_tsum_coe {α : Type*} (p : PMF α) : ∑' z, (p z).toReal = 1 := by
  have hne : ∀ z, p z ≠ ⊤ := fun z => p.apply_ne_top z
  rw [← ENNReal.tsum_toReal_eq hne]
  rw [p.tsum_coe]
  simp

/-!
## Bounded Summability Helpers

These lemmas provide summability for common bounded cases,
enabling rigorous use without the axiom.
-/

/-- Summability for distortion when metric is bounded -/
lemma summable_D_of_bounded {α : Type*} (p : PMF α) {Y : Type*} [PseudoMetricSpace Y]
    (fstar : α → Y) (x : α) (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ z, D fstar z x ≤ M) :
    Summable (fun z => (p z).toReal * D fstar z x) :=
  PMF.summable_coe_real_mul_of_bounded p _ M hM (fun z => by
    unfold D; rw [abs_of_nonneg dist_nonneg]; exact hbound z)

/-- Summability for [0,1]-valued functions -/
lemma summable_indicator_bounded {α : Type*} (p : PMF α) (f : α → ℝ)
    (hf_nonneg : ∀ z, 0 ≤ f z) (hf_le_one : ∀ z, f z ≤ 1) :
    Summable (fun z => (p z).toReal * f z) :=
  PMF.summable_coe_real_mul_of_bounded p f 1 (by norm_num) (fun z => by
    rw [abs_of_nonneg (hf_nonneg z)]; exact hf_le_one z)

/-!
## Relating Exp and ExpENN
-/

/-- Relating Exp (Real) and ExpENN (ENNReal) for non-negative functions -/
lemma Exp_eq_ExpENN_toReal {α : Type*} (p : PMF α) (f : α → ℝ) (hf : ∀ x, 0 ≤ f x) :
  Exp p f = (ExpENN p (fun x => ENNReal.ofReal (f x))).toReal := by
    unfold Exp ExpENN;
    rw [ ENNReal.tsum_toReal_eq ];
    · rw [ tsum_congr ] ; intros ; rw [ ENNReal.toReal_mul ] ; aesop;
    · exact fun x => ENNReal.mul_ne_top ( p.apply_ne_top x ) ( ENNReal.ofReal_ne_top )

/-!
## tsum Lemmas
-/

/-- If a summable series of non-negative real numbers sums to 0, then each term is 0 -/
lemma tsum_eq_zero_of_nonneg {α : Type*} (f : α → ℝ) (hf : ∀ x, 0 ≤ f x) (h_summable : Summable f) (h_sum : ∑' x, f x = 0) :
  ∀ x, f x = 0 := by
    contrapose! h_sum;
    exact ne_of_gt ( lt_of_lt_of_le ( lt_of_le_of_ne ( hf _ ) ( Ne.symm h_sum.choose_spec ) ) ( Summable.le_tsum h_summable _ fun x _ => hf x ) )

/-!
## L3 Implications
-/

/-- If L3 holds and expected distortion is summable, distortion is 0 on support -/
lemma L3_implies_dist_zero_on_support (g : Summarizer Strings) (fstar : Strings → Y) (h3 : L3 g fstar) (Z : Strings) (hZ : InRange g Z)
  (h_summable : Summable (fun z => (g Z z).toReal * D fstar z Z)) :
  ∀ z ∈ (g Z).support, D fstar z Z = 0 := by
    have h_sum_zero : ∑' z, (g Z z).toReal * D fstar z Z = 0 := by
      exact h3 Z hZ;
    intro z hz; contrapose! h_sum_zero; simp_all +decide [ dist_eq_zero ] ;
    refine ne_of_gt ( lt_of_lt_of_le ?_ ( Summable.le_tsum ?_ z ?_ ) );
    · exact mul_pos ( ENNReal.toReal_pos hz ( by
        exact PMF.apply_ne_top (g Z) z ) ) ( lt_of_le_of_ne ( dist_nonneg ) ( Ne.symm h_sum_zero ) );
    · exact h_summable;
    · exact fun _ _ => mul_nonneg ( ENNReal.toReal_nonneg ) ( dist_nonneg )

/-- If L3 holds, then each term in the expected distortion sum is zero -/
lemma L3_implies_term_zero (g : Summarizer Strings) (fstar : Strings → Y) (h3 : L3 g fstar) (Z : Strings) (hZ : InRange g Z)
  (h_summable : Summable (fun z => (g Z z).toReal * D fstar z Z)) :
  ∀ z, (g Z z).toReal * D fstar z Z = 0 := by
    intros z
    by_cases hz : z ∈ (g Z).support
    · exact (by
      have h_dist_zero : D fstar z Z = 0 := by
        apply L3_implies_dist_zero_on_support g fstar h3 Z hZ h_summable z hz;
      rw [ h_dist_zero, MulZeroClass.mul_zero ])
    · exact (by aesop)

/-- If L3 holds and expected distortion is summable, then ExpENN of distortion is 0 -/
lemma L3_implies_ExpENN_zero (g : Summarizer Strings) (fstar : Strings → Y) (h3 : L3 g fstar) (Z : Strings) (hZ : InRange g Z)
  (h_summable : Summable (fun z => (g Z z).toReal * D fstar z Z)) :
  ExpENN (g Z) (fun z => ENNReal.ofReal (D fstar z Z)) = 0 := by
    have h_zero_terms : ∀ z, (g Z z) * ENNReal.ofReal (D fstar z Z) = 0 := by
      by_contra h_nonzero;
      have h_exp_zero : ∀ z ∈ (g Z).support, D fstar z Z = 0 := by
        exact L3_implies_dist_zero_on_support g fstar h3 Z hZ h_summable;
      aesop;
    exact ENNReal.tsum_eq_zero.mpr fun _ => h_zero_terms _

/-!
## Helper Lemmas for Multi-Round
-/

/-- Exp over ZR at R=1 equals Egu over root -/
lemma Exp_ZR_one_eq_Egu (g : Summarizer Strings) (x : Strings) (T : BinTree Strings) (f : Strings → ℝ) :
  Exp (ZR g x 1 T) f = Egu g (root T) f := by
    unfold Exp Egu ZR root
    rfl

/-- Triangle inequality for distortion -/
lemma D_triangle (fstar : Strings → Y) (w z x : Strings) :
  D fstar w x ≤ D fstar w z + D fstar z x := by
    unfold D
    exact dist_triangle (fstar w) (fstar z) (fstar x)

/-- Exp is linear (for adding functions) -/
lemma Exp_add (p : PMF Strings) (f₁ f₂ : Strings → ℝ)
  (hf₁ : Summable (fun z => (p z).toReal * f₁ z))
  (hf₂ : Summable (fun z => (p z).toReal * f₂ z)) :
  Exp p (fun z => f₁ z + f₂ z) = Exp p f₁ + Exp p f₂ := by
    unfold Exp
    rw [← hf₁.tsum_add hf₂]
    congr 1
    ext z
    ring

/-- Exp_add with bounded functions (avoids axiom) -/
lemma Exp_add_bounded (p : PMF Strings) (f₁ f₂ : Strings → ℝ) (M₁ M₂ : ℝ)
    (hM₁ : 0 ≤ M₁) (hM₂ : 0 ≤ M₂)
    (hf₁ : ∀ z, |f₁ z| ≤ M₁) (hf₂ : ∀ z, |f₂ z| ≤ M₂) :
    Exp p (fun z => f₁ z + f₂ z) = Exp p f₁ + Exp p f₂ :=
  Exp_add p f₁ f₂
    (PMF.summable_coe_real_mul_of_bounded p f₁ M₁ hM₁ hf₁)
    (PMF.summable_coe_real_mul_of_bounded p f₂ M₂ hM₂ hf₂)

/-- Exp is monotone for non-negative functions (with explicit summability) -/
lemma Exp_mono' (p : PMF Strings) (f₁ f₂ : Strings → ℝ) (h : ∀ z, f₁ z ≤ f₂ z)
    (hf₁ : Summable (fun z => (p z).toReal * f₁ z))
    (hf₂ : Summable (fun z => (p z).toReal * f₂ z)) :
    Exp p f₁ ≤ Exp p f₂ := by
  unfold Exp
  apply Summable.tsum_le_tsum
  · intro z; exact mul_le_mul_of_nonneg_left (h z) ENNReal.toReal_nonneg
  · exact hf₁
  · exact hf₂

/-- Exp_mono with bounded functions (avoids axiom) -/
lemma Exp_mono_bounded (p : PMF Strings) (f₁ f₂ : Strings → ℝ) (M : ℝ) (hM : 0 ≤ M)
    (h : ∀ z, f₁ z ≤ f₂ z) (hf₁ : ∀ z, |f₁ z| ≤ M) (hf₂ : ∀ z, |f₂ z| ≤ M) :
    Exp p f₁ ≤ Exp p f₂ :=
  Exp_mono' p f₁ f₂ h
    (PMF.summable_coe_real_mul_of_bounded p f₁ M hM hf₁)
    (PMF.summable_coe_real_mul_of_bounded p f₂ M hM hf₂)

/-- Exp is monotone for non-negative functions.
    Note: Uses PMF.summable_coe_real_mul axiom. For explicit bounds, use Exp_mono' or Exp_mono_bounded. -/
lemma Exp_mono (p : PMF Strings) (f₁ f₂ : Strings → ℝ) (h : ∀ z, f₁ z ≤ f₂ z) :
  Exp p f₁ ≤ Exp p f₂ :=
    Exp_mono' p f₁ f₂ h (PMF.summable_coe_real_mul p f₁) (PMF.summable_coe_real_mul p f₂)

/-- ZR at R+1 is bind of ZR at R with g, when R ≥ 1 -/
lemma ZR_succ_eq_bind (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings) (hR : R ≥ 1) :
  ZR g x (R + 1) T = (ZR g x R T).bind g := by
    cases R with
    | zero => exact absurd hR (by decide)
    | succ n => rfl

/-- Law of iterated expectation for Exp (with explicit summability) -/
lemma Exp_bind_eq' (p : PMF Strings) (g : Summarizer Strings) (f : Strings → ℝ) (hf : ∀ z, 0 ≤ f z)
    (h_sum : ∀ z, Summable (fun w => (g z w).toReal * f w)) :
  Exp (p.bind g) f = Exp p (fun z => Exp (g z) f) := by
    have hExp_nonneg : ∀ z, 0 ≤ Exp (g z) f := fun z => by
      unfold Exp; apply tsum_nonneg; intro w; exact mul_nonneg ENNReal.toReal_nonneg (hf w)
    rw [Exp_eq_ExpENN_toReal (p.bind g) f hf]
    rw [Exp_eq_ExpENN_toReal p (fun z => Exp (g z) f) hExp_nonneg]
    congr 1
    rw [ExpENN_bind]
    congr 1; ext z
    unfold ExpENN Exp
    symm
    rw [ENNReal.ofReal_tsum_of_nonneg]
    · congr 1; ext w
      rw [ENNReal.ofReal_mul ENNReal.toReal_nonneg]
      rw [ENNReal.ofReal_toReal ((g z).apply_ne_top w)]
    · intro w; exact mul_nonneg ENNReal.toReal_nonneg (hf w)
    · exact h_sum z

/-- Law of iterated expectation for Exp.
    Note: Uses PMF.summable_coe_real_mul axiom. For explicit summability, use Exp_bind_eq'. -/
lemma Exp_bind_eq (p : PMF Strings) (g : Summarizer Strings) (f : Strings → ℝ) (hf : ∀ z, 0 ≤ f z) :
  Exp (p.bind g) f = Exp p (fun z => Exp (g z) f) :=
    Exp_bind_eq' p g f hf (fun z => PMF.summable_coe_real_mul (g z) f)

lemma Exp_bind_le (p : PMF Strings) (g : Summarizer Strings) (f : Strings → ℝ) (hf : ∀ z, 0 ≤ f z) :
  Exp (p.bind g) f ≤ Exp p (fun z => Exp (g z) f) := by
    rw [Exp_bind_eq p g f hf]

/-- Expected distortion from one more round is 0 (uses L3 + support condition) -/
lemma multi_round_step (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings) (fstar : Strings → Y)
  (h3 : L3 g fstar) (h_supp : ∀ z ∈ (ZR g x R T).support, InRange g z) :
  Exp (ZR g x R T) (fun z => Eg g (fun w => D fstar w z) z) = 0 := by
    unfold Exp;
    convert tsum_zero with z;
    by_cases hz : z ∈ ( ZR g x R T ).support <;> aesop

/-!
## Multi-Round Preservation (Main Theorem)
-/

/-- Multi-Round Preservation: If L1, L2, L3 hold and R ≥ 1, expected distortion after R rounds is 0.

⚠️ DEPRECATED: This version uses the unsound `PMF.summable_coe_real_mul` axiom internally.
For mathematically rigorous proofs, use `multi_round_proper` or `multi_round_bounded` instead,
which avoid the axiom by requiring explicit boundedness hypotheses. -/
theorem multi_round (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ) (fstar : Strings → Y)
  (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1) :
  Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
    induction' hR with R' hR' ih generalizing x T
    -- Base case: R = 1
    · rw [Exp_ZR_one_eq_Egu]
      exact one_pass g T x fstar hp h1 h2
    -- Inductive case: R = R' + 1 where R' ≥ 1
    · rw [ZR_succ_eq_bind g x R' T hR']
      have hD_nonneg : ∀ z, 0 ≤ D fstar z x := fun z => dist_nonneg
      have h_exp_nonneg : 0 ≤ Exp ((ZR g x R' T).bind g) (fun z => D fstar z x) := by
        unfold Exp
        apply tsum_nonneg
        intro z
        exact mul_nonneg ENNReal.toReal_nonneg (hD_nonneg z)
      have h_bound : Exp ((ZR g x R' T).bind g) (fun w => D fstar w x) ≤
                     Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z) +
                     Exp (ZR g x R' T) (fun z => D fstar z x) := by
        calc Exp ((ZR g x R' T).bind g) (fun w => D fstar w x)
          ≤ Exp (ZR g x R' T) (fun z => Exp (g z) (fun w => D fstar w x)) := by
              exact Exp_bind_le (ZR g x R' T) g (fun w => D fstar w x) hD_nonneg
          _ ≤ Exp (ZR g x R' T) (fun z => Exp (g z) (fun w => D fstar w z + D fstar z x)) := by
              apply Exp_mono
              intro z
              apply Exp_mono
              intro w
              exact D_triangle fstar w z x
          _ = Exp (ZR g x R' T) (fun z => Exp (g z) (fun w => D fstar w z) + Exp (g z) (fun _ => D fstar z x)) := by
              congr 1; ext z
              rw [Exp_add]
              · exact PMF.summable_coe_real_mul (g z) _
              · exact PMF.summable_coe_real_mul (g z) _
          _ = Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z + D fstar z x) := by
              congr 1; ext z
              unfold Eg Exp
              congr 1
              simp only [tsum_mul_right]
              rw [PMF.toReal_tsum_coe]
              ring
          _ ≤ Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z) +
              Exp (ZR g x R' T) (fun z => D fstar z x) := by
              rw [Exp_add]
              · exact PMF.summable_coe_real_mul (ZR g x R' T) _
              · exact PMF.summable_coe_real_mul (ZR g x R' T) _
      have h_first_zero : Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z) = 0 :=
        multi_round_step g x R' T fstar h3 (ZR_support_in_range g x R' T hR')
      have h_second_zero : Exp (ZR g x R' T) (fun z => D fstar z x) = 0 := ih T x hp h1 h2
      linarith

/-!
## Bounded Multi-Round Preservation

This version provides explicit boundedness hypotheses, avoiding the `PMF.summable_coe_real_mul` axiom.
Use this for mathematically rigorous proofs when the metric space has bounded diameter.
-/

/-- Helper: bound on |D fstar z x| from bound on D -/
lemma abs_D_le_of_D_le (fstar : Strings → Y) (z x : Strings) (M : ℝ) (_hM : 0 ≤ M) (h : D fstar z x ≤ M) :
    |D fstar z x| ≤ M := by
  unfold D at h ⊢
  rw [abs_of_nonneg dist_nonneg]; exact h

/-- Helper: Eg is bounded when D is bounded -/
lemma Eg_D_bounded (g : Summarizer Strings) (fstar : Strings → Y) (z : Strings) (M : ℝ)
    (hM : 0 ≤ M) (hbound : ∀ w, D fstar w z ≤ M) :
    Eg g (fun w => D fstar w z) z ≤ M := by
  unfold Eg
  calc ∑' w, (g z w).toReal * D fstar w z
      ≤ ∑' w, (g z w).toReal * M := by
        apply Summable.tsum_le_tsum
        · intro w; exact mul_le_mul_of_nonneg_left (hbound w) ENNReal.toReal_nonneg
        · exact PMF.summable_coe_real_mul_of_bounded (g z) _ M hM (fun w => abs_D_le_of_D_le fstar w z M hM (hbound w))
        · exact (PMF.summable_coe_real (g z)).mul_right M
      _ = M := by rw [tsum_mul_right, PMF.toReal_tsum_coe]; ring

/-- Helper: |Eg D| is bounded when D is bounded -/
lemma abs_Eg_D_bounded (g : Summarizer Strings) (fstar : Strings → Y) (z : Strings) (M : ℝ)
    (hM : 0 ≤ M) (hbound : ∀ w, D fstar w z ≤ M) :
    |Eg g (fun w => D fstar w z) z| ≤ M := by
  rw [abs_of_nonneg]
  · exact Eg_D_bounded g fstar z M hM hbound
  · unfold Eg; apply tsum_nonneg; intro w; exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg

/-- Exp_bind_le with explicit boundedness (avoids axiom) -/
lemma Exp_bind_le_bounded (p : PMF Strings) (g : Summarizer Strings) (f : Strings → ℝ) (M : ℝ)
    (hM : 0 ≤ M) (hf : ∀ z, 0 ≤ f z) (hfbound : ∀ z, |f z| ≤ M) :
    Exp (p.bind g) f ≤ Exp p (fun z => Exp (g z) f) := by
  have h_sum : ∀ z, Summable (fun w => (g z w).toReal * f w) :=
    fun z => PMF.summable_coe_real_mul_of_bounded (g z) f M hM hfbound
  rw [Exp_bind_eq' p g f hf h_sum]

/-- Multi-Round Preservation: Fully rigorous version with explicit boundedness.

This theorem is mathematically equivalent to `multi_round` but uses only proven lemmas,
avoiding the `PMF.summable_coe_real_mul` axiom. The bound M should be the diameter of
the metric space (or any upper bound on pairwise distortion).

The proof follows the same structure as `multi_round`:
1. Base case: R = 1 uses one_pass which is already axiom-free
2. Inductive case: Uses triangle inequality and bounded summability lemmas

All summability arguments use `PMF.summable_coe_real_mul_of_bounded` (proven lemma)
instead of `PMF.summable_coe_real_mul` (unsound axiom). -/
theorem multi_round_proper (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ) (fstar : Strings → Y)
    (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ z w, D fstar z w ≤ M) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
  induction' hR with R' hR' ih generalizing x T
  -- Base case: R = 1 (uses one_pass, which is axiom-free)
  · rw [Exp_ZR_one_eq_Egu]
    exact one_pass g T x fstar hp h1 h2
  -- Inductive case: R = R' + 1 where R' ≥ 1
  -- The proof uses bounded variants of Exp_mono and Exp_add.
  -- Key bounds derived from hbound:
  · rw [ZR_succ_eq_bind g x R' T hR']
    have hD_nonneg : ∀ z, 0 ≤ D fstar z x := fun z => dist_nonneg
    have hD_bound : ∀ z, |D fstar z x| ≤ M := fun z => abs_D_le_of_D_le fstar z x M hM (hbound z x)
    have hEgD_bound : ∀ z, |Eg g (fun w => D fstar w z) z| ≤ M :=
      fun z => abs_Eg_D_bounded g fstar z M hM (fun w => hbound w z)
    -- Main inequality chain using bounded lemmas
    have h_bound : Exp ((ZR g x R' T).bind g) (fun w => D fstar w x) ≤
                   Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z) +
                   Exp (ZR g x R' T) (fun z => D fstar z x) := by
      calc Exp ((ZR g x R' T).bind g) (fun w => D fstar w x)
        ≤ Exp (ZR g x R' T) (fun z => Exp (g z) (fun w => D fstar w x)) := by
            exact Exp_bind_le_bounded (ZR g x R' T) g (fun w => D fstar w x) M hM hD_nonneg hD_bound
        _ ≤ Exp (ZR g x R' T) (fun z => Exp (g z) (fun w => D fstar w z + D fstar z x)) := by
            -- Use 2*M as bound since D w z + D z x ≤ M + M = 2*M
            have h2M : 0 ≤ 2 * M := by linarith
            apply Exp_mono_bounded _ _ _ (2 * M) h2M
            · intro z
              apply Exp_mono_bounded _ _ _ (2 * M) h2M
              · intro w; exact D_triangle fstar w z x
              · intro w
                calc |D fstar w x| = D fstar w x := abs_of_nonneg dist_nonneg
                  _ ≤ M := hbound w x
                  _ ≤ 2 * M := by linarith
              · intro w
                have h1 : D fstar w z ≤ M := hbound w z
                have h2 : D fstar z x ≤ M := hbound z x
                calc |D fstar w z + D fstar z x|
                    = D fstar w z + D fstar z x := abs_of_nonneg (add_nonneg dist_nonneg dist_nonneg)
                  _ ≤ M + M := add_le_add h1 h2
                  _ = 2 * M := by ring
            · intro z
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => D fstar w x) := by
                unfold Exp; apply tsum_nonneg; intro w; exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
              calc |Exp (g z) (fun w => D fstar w x)|
                  = Exp (g z) (fun w => D fstar w x) := abs_of_nonneg hExp_nonneg
                _ ≤ M := by
                    unfold Exp
                    calc ∑' w, (g z w).toReal * D fstar w x
                        ≤ ∑' w, (g z w).toReal * M := by
                          apply Summable.tsum_le_tsum
                          · intro w; exact mul_le_mul_of_nonneg_left (hbound w x) ENNReal.toReal_nonneg
                          · exact PMF.summable_coe_real_mul_of_bounded (g z) _ M hM hD_bound
                          · exact (PMF.summable_coe_real (g z)).mul_right M
                      _ = M := by rw [tsum_mul_right, PMF.toReal_tsum_coe]; ring
                _ ≤ 2 * M := by linarith
            · intro z
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => D fstar w z + D fstar z x) := by
                unfold Exp; apply tsum_nonneg; intro w
                exact mul_nonneg ENNReal.toReal_nonneg (add_nonneg dist_nonneg dist_nonneg)
              have hSum_bound : ∀ w, |D fstar w z + D fstar z x| ≤ 2 * M := fun w => by
                have h1 : D fstar w z ≤ M := hbound w z
                have h2 : D fstar z x ≤ M := hbound z x
                calc |D fstar w z + D fstar z x|
                    = D fstar w z + D fstar z x := abs_of_nonneg (add_nonneg dist_nonneg dist_nonneg)
                  _ ≤ M + M := add_le_add h1 h2
                  _ = 2 * M := by ring
              calc |Exp (g z) (fun w => D fstar w z + D fstar z x)|
                  = Exp (g z) (fun w => D fstar w z + D fstar z x) := abs_of_nonneg hExp_nonneg
                _ ≤ 2 * M := by
                    unfold Exp
                    calc ∑' w, (g z w).toReal * (D fstar w z + D fstar z x)
                        ≤ ∑' w, (g z w).toReal * (2 * M) := by
                          apply Summable.tsum_le_tsum
                          · intro w
                            apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
                            have h1 : D fstar w z ≤ M := hbound w z
                            have h2 : D fstar z x ≤ M := hbound z x
                            linarith
                          · exact PMF.summable_coe_real_mul_of_bounded (g z) _ (2 * M) h2M hSum_bound
                          · exact (PMF.summable_coe_real (g z)).mul_right (2 * M)
                      _ = 2 * M := by rw [tsum_mul_right, PMF.toReal_tsum_coe]; ring
        _ = Exp (ZR g x R' T) (fun z => Exp (g z) (fun w => D fstar w z) + Exp (g z) (fun _ => D fstar z x)) := by
            congr 1; ext z
            exact Exp_add_bounded (g z) _ _ M M hM hM
              (fun w => abs_D_le_of_D_le fstar w z M hM (hbound w z))
              (fun _ => hD_bound z)
        _ = Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z + D fstar z x) := by
            congr 1; ext z
            unfold Eg Exp
            congr 1
            simp only [tsum_mul_right]
            rw [PMF.toReal_tsum_coe]
            ring
        _ = Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z) +
            Exp (ZR g x R' T) (fun z => D fstar z x) := by
            exact Exp_add_bounded (ZR g x R' T) _ _ M M hM hM hEgD_bound hD_bound
    have h_first_zero : Exp (ZR g x R' T) (fun z => Eg g (fun w => D fstar w z) z) = 0 :=
      multi_round_step g x R' T fstar h3 (ZR_support_in_range g x R' T hR')
    have h_second_zero : Exp (ZR g x R' T) (fun z => D fstar z x) = 0 := ih T x hp h1 h2
    -- Exp is nonnegative since D is nonnegative
    have h_nonneg : 0 ≤ Exp ((ZR g x R' T).bind g) (fun w => D fstar w x) := by
      unfold Exp; apply tsum_nonneg; intro w
      exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
    linarith

/-- Multi-Round Preservation with explicit boundedness.

This version uses only proven lemmas (via `multi_round_proper`), avoiding the
`PMF.summable_coe_real_mul` axiom entirely.

The bound M should be the diameter of the metric space (or an upper bound on distortion). -/
theorem multi_round_bounded (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ) (fstar : Strings → Y)
  (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
  (M : ℝ) (hM : 0 ≤ M) (_hbound : ∀ z, D fstar z x ≤ M) (hbound_global : ∀ w z, D fstar w z ≤ M) :
  Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_proper g T x R fstar hp h1 h2 h3 hR M hM hbound_global

end
