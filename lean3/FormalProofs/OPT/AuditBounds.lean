/-
FormalProofs/AuditBounds.lean

Audit Bounds from Section 7:
- violationInd: indicator for positive distortion
- ViolationProb: expected violation indicator
- pLeafAvg, pMergeAvg: average violation rates
- Connection to preservation theorems
-/

import Mathlib
import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.TreeProperties

/-! ## From FormalProofs/OPT/AuditCore.lean (consolidated 2026-07-02) -/

/-
FormalProofs/AuditCore.lean

Shared confidence margin and sample complexity definitions for audit proofs.
-/

section

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

/-!
## Confidence Margin and Sample Complexity

The confidence margin from Hoeffding's inequality for n iid samples
from a distribution with values in [0,1]:

  P(|p_hat - p| >= eps) <= 2 * exp(-2 * n * eps^2)

Solving for eps given delta = 2 * exp(-2 * n * eps^2):
  eps = sqrt(ln(2/delta) / (2 * n))
-/

/-- Confidence margin from Hoeffding: sqrt(ln(2/delta) / (2 * n))

Given n samples and confidence parameter delta, this is the margin eps
such that P(|p_hat - p| >= eps) <= delta for Hoeffding's inequality. -/
def confidence_margin (delta : Real) (n : Nat) : Real :=
  Real.sqrt (Real.log (2 / delta) / (2 * n))

/-- Sample complexity for (eps, delta)-guarantee: ceil(ln(2/delta) / (2 * eps^2))

The minimum number of samples needed to achieve margin eps
with confidence 1 - delta. -/
def sample_complexity (eps delta : Real) : Nat :=
  Nat.ceil (Real.log (2 / delta) / (2 * eps^2))

/-- Confidence margin is non-negative when parameters are valid -/
lemma confidence_margin_nonneg (delta : Real) (n : Nat) (h_delta : 0 < delta) (h_delta' : delta < 2)
    (h_n : 0 < n) : 0 <= confidence_margin delta n := by
  unfold confidence_margin
  apply Real.sqrt_nonneg

/- Sample complexity gives required margin.

When n >= sample_complexity(eps, delta), the confidence margin is at most eps.
This is a direct consequence of the definition: sample_complexity inverts
the margin formula. -/
lemma sample_complexity_gives_margin (eps delta : Real) (h_eps : 0 < eps) (h_delta : 0 < delta)
    (h_delta' : delta < 2) (n : Nat) (h_n : n >= sample_complexity eps delta) :
    confidence_margin delta n <= eps := by
  unfold confidence_margin sample_complexity at *
  -- n >= ceil(ln(2/delta)/(2 * eps^2)) implies ln(2/delta)/(2n) <= eps^2
  have h_log_pos : 0 < Real.log (2 / delta) := by
    apply Real.log_pos
    rw [one_lt_div h_delta]
    linarith
  have h_ratio_pos : 0 < Real.log (2 / delta) / (2 * eps^2) := by positivity
  have h_n_pos : 0 < n := Nat.lt_of_lt_of_le (Nat.ceil_pos.mpr h_ratio_pos) h_n
  have h_n_ge : Real.log (2 / delta) / (2 * eps^2) <= n := by
    have h_ceil :
        Real.log (2 / delta) / (2 * eps^2) <=
          (Nat.ceil (Real.log (2 / delta) / (2 * eps^2)) : Real) := by
      simpa using (Nat.le_ceil (Real.log (2 / delta) / (2 * eps^2)))
    have h_ceil_le : (Nat.ceil (Real.log (2 / delta) / (2 * eps^2)) : Real) <= n := by
      exact_mod_cast h_n
    exact le_trans h_ceil h_ceil_le
  -- From h_n_ge, derive ln(2/delta) <= 2 * n * eps^2
  have h_ineq : Real.log (2 / delta) <= 2 * n * eps^2 := by
    have h2eps : 0 < 2 * eps^2 := by positivity
    calc Real.log (2 / delta) = (Real.log (2 / delta) / (2 * eps^2)) * (2 * eps^2) := by
            field_simp [ne_of_gt h2eps]
      _ <= n * (2 * eps^2) := by nlinarith
      _ = 2 * n * eps^2 := by ring
  -- Therefore ln(2/delta)/(2n) <= eps^2
  have h_ratio_le : Real.log (2 / delta) / (2 * n) <= eps^2 := by
    have h2n_pos : 0 < 2 * (n : Real) := by positivity
    apply (div_le_iff₀ h2n_pos).2
    nlinarith [h_ineq]
  -- Finally, sqrt(ln(2/delta)/(2n)) <= eps
  apply Real.sqrt_le_iff.mpr
  exact And.intro (le_of_lt h_eps) h_ratio_le


end

end

/-! ## Original FormalProofs/OPT/AuditBounds.lean content -/

section

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
## Section 11: Audit Bounds (Union Bound)

This section formalizes the Audit Bounds from Section 7 of the paper.
It provides probabilistic upper bounds on the deviation probability using union bounds.

Key result: P[D(Z^R, X) > 0] ≤ N * p_leaf + M * p_merge + (R-1) * p_idemp

Where:
- p_leaf = average violation probability at leaves
- p_merge = average violation probability at merges
- p_idemp = idempotence violation probability
-/

section AuditBounds

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-
Violation indicator: 1 if distortion is positive, 0 otherwise.
This is the core building block for probability bounds.
-/
def violationInd (fstar : Strings → Y) (z x : Strings) : ℝ :=
  if D fstar z x > 0 then 1 else 0

lemma violationInd_nonneg (fstar : Strings → Y) (z x : Strings) :
    0 ≤ violationInd fstar z x := by
  unfold violationInd
  split_ifs <;> linarith

lemma violationInd_le_one (fstar : Strings → Y) (z x : Strings) :
    violationInd fstar z x ≤ 1 := by
  unfold violationInd
  split_ifs <;> linarith

lemma violationInd_eq_zero_iff (fstar : Strings → Y) (z x : Strings) :
    violationInd fstar z x = 0 ↔ D fstar z x = 0 := by
  unfold violationInd D
  simp only [ite_eq_right_iff, one_ne_zero]
  constructor
  · intro h
    by_contra hne
    have hpos : dist (fstar z) (fstar x) > 0 := lt_of_le_of_ne dist_nonneg (Ne.symm hne)
    exact h hpos
  · intro h
    simp [h]

lemma violationInd_eq_one_iff (fstar : Strings → Y) (z x : Strings) :
    violationInd fstar z x = 1 ↔ D fstar z x > 0 := by
  unfold violationInd
  constructor
  · intro h
    by_contra hle
    push_neg at hle
    have : D fstar z x = 0 := le_antisymm hle dist_nonneg
    simp [this] at h
  · intro h
    simp [h]

/-
Violation probability: Expected value of indicator over a PMF.
This equals P[D(Z, x) > 0] where Z ~ p.
-/
def ViolationProb (fstar : Strings → Y) (p : PMF Strings) (x : Strings) : ℝ :=
  Exp p (fun z => violationInd fstar z x)

lemma ViolationProb_nonneg (fstar : Strings → Y) (p : PMF Strings) (x : Strings) :
    0 ≤ ViolationProb fstar p x := by
  unfold ViolationProb Exp
  apply tsum_nonneg
  intro z
  exact mul_nonneg ENNReal.toReal_nonneg (violationInd_nonneg fstar z x)

lemma ViolationProb_le_one (fstar : Strings → Y) (p : PMF Strings) (x : Strings) :
    ViolationProb fstar p x ≤ 1 := by
  unfold ViolationProb
  have h := Exp_mono_bounded p (fun z => violationInd fstar z x) (fun _ => 1) 1 (by norm_num)
              (fun z => violationInd_le_one fstar z x)
              (fun z => by rw [abs_of_nonneg (violationInd_nonneg fstar z x)]; exact violationInd_le_one fstar z x)
              (fun _ => by norm_num)
  calc Exp p (fun z => violationInd fstar z x)
      ≤ Exp p (fun _ => 1) := h
      _ = 1 := by
        unfold Exp
        simp only [mul_one]
        exact PMF.toReal_tsum_coe p

/-
-- COMMENTED OUT: Deprecated lemmas using unsound axiom
-- Use the bounded versions below instead

/-- When expected distortion is 0, violation probability is 0.
This is the key lemma connecting preservation theorems to audit bounds.

⚠️ DEPRECATED: Uses unsound `PMF.summable_coe_real_mul` axiom.

**Migration path**: Add `[BoundedPseudoMetricSpace Y]` to get diameter bound M,
then use `summable_distortion_pmf_bounded p fstar x M hM hbound`. -/
@[deprecated "Use bounded summability helpers" (since := "2026-01-10")]
lemma ViolationProb_eq_zero_of_Exp_D_eq_zero (fstar : Strings → Y) (p : PMF Strings) (x : Strings)
    (h : Exp p (fun z => D fstar z x) = 0) :
    ViolationProb fstar p x = 0 := by
  unfold ViolationProb Exp at *
  -- Since D ≥ 0 and E[D] = 0, we have each term is 0
  have hD_nonneg : ∀ z, 0 ≤ (p z).toReal * D fstar z x :=
    fun z => mul_nonneg ENNReal.toReal_nonneg dist_nonneg
  have hD_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    PMF.summable_coe_real_mul p (fun z => D fstar z x)
  have hD_zero : ∀ z, (p z).toReal * D fstar z x = 0 :=
    tsum_eq_zero_of_nonneg _ hD_nonneg hD_summable h
  -- Each term of violation sum is 0 because D = 0 on support
  convert tsum_zero with z
  have hz := hD_zero z
  by_cases hp : (p z).toReal = 0
  · simp [hp]
  · have hD : D fstar z x = 0 := by
      have := mul_eq_zero.mp hz
      cases this with
      | inl h => exact absurd h hp
      | inr h => exact h
    simp only []
    rw [(violationInd_eq_zero_iff fstar z x).mpr hD]
    simp

/-- Converse: ViolationProb = 0 implies Exp D = 0.
Since violationInd(z,x) = 0 iff D(z,x) = 0, and both are non-negative,
ViolationProb = 0 means D = 0 on support, hence Exp D = 0.

⚠️ DEPRECATED: Uses unsound `PMF.summable_coe_real_mul` axiom.

**Migration path**: Add `[BoundedPseudoMetricSpace Y]` and use
`summable_indicator_bounded` for violationInd (which is in [0,1]). -/
@[deprecated "Use bounded summability helpers" (since := "2026-01-10")]
lemma Exp_D_eq_zero_of_ViolationProb_eq_zero (fstar : Strings → Y) (p : PMF Strings) (x : Strings)
    (h : ViolationProb fstar p x = 0) :
    Exp p (fun z => D fstar z x) = 0 := by
  unfold ViolationProb Exp at *
  -- ViolationProb = 0 means the tsum of non-negative indicator terms is 0
  have hInd_nonneg : ∀ z, 0 ≤ (p z).toReal * violationInd fstar z x :=
    fun z => mul_nonneg ENNReal.toReal_nonneg (violationInd_nonneg fstar z x)
  have hInd_summable : Summable (fun z => (p z).toReal * violationInd fstar z x) :=
    PMF.summable_coe_real_mul p (fun z => violationInd fstar z x)
  have hInd_zero : ∀ z, (p z).toReal * violationInd fstar z x = 0 :=
    tsum_eq_zero_of_nonneg _ hInd_nonneg hInd_summable h
  -- Now show Exp D = 0
  convert tsum_zero with z
  have hz := hInd_zero z
  by_cases hp : (p z).toReal = 0
  · simp [hp]
  · have hInd : violationInd fstar z x = 0 := by
      have := mul_eq_zero.mp hz
      cases this with
      | inl h => exact absurd h hp
      | inr h => exact h
    have hD : D fstar z x = 0 := (violationInd_eq_zero_iff fstar z x).mp hInd
    simp [hD]
-/

/-- When expected distortion is 0, violation probability is 0.
This version uses BoundedPseudoMetricSpace to derive summability. -/
lemma ViolationProb_eq_zero_of_Exp_D_eq_zero_bounded {Strings Y : Type*} [Monoid Strings]
    [BoundedPseudoMetricSpace Y] (fstar : Strings → Y) (p : PMF Strings) (x : Strings)
    (h : Exp p (fun z => D fstar z x) = 0) :
    ViolationProb fstar p x = 0 := by
  unfold ViolationProb Exp at *
  let M := BoundedPseudoMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedPseudoMetricSpace.diameterBound_nonneg
  have hD_bound : ∀ z, |D fstar z x| ≤ M := fun z =>
    BoundedPseudoMetricSpace.abs_dist_le (fstar z) (fstar x)
  have hD_nonneg : ∀ z, 0 ≤ (p z).toReal * D fstar z x :=
    fun z => mul_nonneg ENNReal.toReal_nonneg dist_nonneg
  have hD_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    PMF.summable_coe_real_mul_of_bounded p _ M hM hD_bound
  have hD_zero : ∀ z, (p z).toReal * D fstar z x = 0 :=
    tsum_eq_zero_of_nonneg _ hD_nonneg hD_summable h
  convert tsum_zero with z
  have hz := hD_zero z
  by_cases hp : (p z).toReal = 0
  · simp [hp]
  · have hD : D fstar z x = 0 := by
      have := mul_eq_zero.mp hz
      cases this with
      | inl h => exact absurd h hp
      | inr h => exact h
    simp only []
    rw [(violationInd_eq_zero_iff fstar z x).mpr hD]
    simp

/-- Converse: ViolationProb = 0 implies Exp D = 0.
This version uses BoundedPseudoMetricSpace to derive summability. -/
lemma Exp_D_eq_zero_of_ViolationProb_eq_zero_bounded {Strings Y : Type*} [Monoid Strings]
    [BoundedPseudoMetricSpace Y] (fstar : Strings → Y) (p : PMF Strings) (x : Strings)
    (h : ViolationProb fstar p x = 0) :
    Exp p (fun z => D fstar z x) = 0 := by
  unfold ViolationProb Exp at *
  -- violationInd is in [0,1], so use 1 as the bound
  have hInd_nonneg : ∀ z, 0 ≤ (p z).toReal * violationInd fstar z x :=
    fun z => mul_nonneg ENNReal.toReal_nonneg (violationInd_nonneg fstar z x)
  have hInd_bound : ∀ z, |violationInd fstar z x| ≤ 1 := fun z => by
    rw [abs_of_nonneg (violationInd_nonneg fstar z x)]
    exact violationInd_le_one fstar z x
  have hInd_summable : Summable (fun z => (p z).toReal * violationInd fstar z x) :=
    PMF.summable_coe_real_mul_of_bounded p _ 1 (by norm_num) hInd_bound
  have hInd_zero : ∀ z, (p z).toReal * violationInd fstar z x = 0 :=
    tsum_eq_zero_of_nonneg _ hInd_nonneg hInd_summable h
  let M := BoundedPseudoMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedPseudoMetricSpace.diameterBound_nonneg
  have hD_bound : ∀ z, |D fstar z x| ≤ M := fun z =>
    BoundedPseudoMetricSpace.abs_dist_le (fstar z) (fstar x)
  have hD_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    PMF.summable_coe_real_mul_of_bounded p _ M hM hD_bound
  convert tsum_zero with z
  have hz := hInd_zero z
  by_cases hp : (p z).toReal = 0
  · simp [hp]
  · have hInd : violationInd fstar z x = 0 := by
      have := mul_eq_zero.mp hz
      cases this with
      | inl h => exact absurd h hp
      | inr h => exact h
    have hD : D fstar z x = 0 := (violationInd_eq_zero_iff fstar z x).mp hInd
    simp [hD]

/-- Helper: foldl with + over non-negative terms is monotonic in init -/
lemma foldl_add_ge_init {α : Type*} (f : α → ℝ) (l : List α) (init : ℝ)
    (hf : ∀ a ∈ l, 0 ≤ f a) :
    init ≤ l.foldl (fun acc a => acc + f a) init := by
  induction l generalizing init with
  | nil => exact le_refl init
  | cons x xs ih =>
    simp only [List.foldl_cons]
    have hx : 0 ≤ f x := hf x (by simp)
    calc init
        ≤ init + f x := by linarith
      _ ≤ xs.foldl (fun acc a => acc + f a) (init + f x) :=
          ih (init + f x) (fun a ha => hf a (by simp [ha]))

/-- When a foldl sum of non-negative terms equals 0, each term is 0.

Proof: By induction on the list. The sum of non-negative terms equals 0
iff each term is 0. This is a standard result from analysis. -/
lemma foldl_add_eq_zero_implies_all_zero {α : Type*} (f : α → ℝ) (l : List α)
    (hf : ∀ a ∈ l, 0 ≤ f a)
    (hsum : l.foldl (fun acc a => acc + f a) 0 = 0) :
    ∀ a ∈ l, f a = 0 := by
  induction l with
  | nil => intro a ha; simp at ha
  | cons x xs ih =>
    intro a ha
    simp only [List.foldl_cons, zero_add] at hsum
    -- foldl xs (f x) = 0, and f x ≤ foldl xs (f x), and f x ≥ 0, so f x = 0
    have hge : f x ≤ xs.foldl (fun acc a => acc + f a) (f x) :=
      foldl_add_ge_init f xs (f x) (fun a ha => hf a (by simp [ha]))
    have hfx : f x = 0 := by
      have h1 : f x ≥ 0 := hf x (by simp)
      have h2 : f x ≤ 0 := by rw [hsum] at hge; exact hge
      linarith
    -- Now foldl xs 0 = foldl xs (f x) = 0 since f x = 0
    have hxs : xs.foldl (fun acc a => acc + f a) 0 = 0 := by
      simp only [← hfx, hsum]
    simp only [List.mem_cons] at ha
    cases ha with
    | inl heq => rw [heq]; exact hfx
    | inr hmem =>
      exact ih (fun b hb => hf b (by simp [hb])) hxs a hmem

/-
Average leaf violation rate: E_g[1{D(g(B), B) > 0}] averaged over leaves.
-/
def pLeafAvg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) : ℝ :=
  let ls := leaves T
  if ls.length = 0 then 0
  else (1 / ls.length) * ls.foldl (fun acc b => acc + ViolationProb fstar (g b) b) 0

/-
Average merge violation rate.
-/
def pMergeAvg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) : ℝ :=
  let nodes := internal_nodes T
  if nodes.length = 0 then 0
  else (1 / nodes.length) * nodes.foldl
    (fun acc (pair : BinTree Strings × BinTree Strings) =>
      let (T_L, T_R) := pair
      acc + ViolationProb fstar (reduce g (BinTree.node T_L T_R)) (S (BinTree.node T_L T_R))) 0

/-!
## Auxiliary lemmas for audit bounds

The full union bound theorem requires additional infrastructure for
tracking violations across tree structure. We provide the key building
blocks here.
-/

/-
-- COMMENTED OUT: More deprecated lemmas
/-- If L1 holds (zero expected distortion at leaves), leaf violation probability is 0.

    ⚠️ DEPRECATED: Uses unsound axiom. Use `leaf_violation_zero_of_L1_bounded` instead. -/
@[deprecated "Use leaf_violation_zero_of_L1_bounded" (since := "2026-01-10")]
lemma leaf_violation_zero_of_L1 (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (h1 : L1 g T fstar) :
    ∀ b ∈ leaves T, ViolationProb fstar (g b) b = 0 := by
  intro b hb
  apply ViolationProb_eq_zero_of_Exp_D_eq_zero
  exact h1 b hb

/-- If L2 holds (zero expected distortion at merges), merge violation probability is 0.

    ⚠️ DEPRECATED: Uses unsound axiom. Use `merge_violation_zero_of_L2_bounded` instead. -/
@[deprecated "Use merge_violation_zero_of_L2_bounded" (since := "2026-01-10")]
lemma merge_violation_zero_of_L2 (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (h2 : L2 g T fstar) :
    ∀ pair ∈ internal_nodes T,
      ViolationProb fstar (reduce g (BinTree.node pair.1 pair.2)) (S (BinTree.node pair.1 pair.2)) = 0 := by
  intro pair hpair
  apply ViolationProb_eq_zero_of_Exp_D_eq_zero
  -- L2 says Egu g (node T_L T_R) (D ... (S (node T_L T_R))) = 0
  -- Egu g T f = ∑' z, (reduce g T z).toReal * f z = Exp (reduce g T) f
  -- So L2 gives exactly what we need (definitionally equal)
  exact h2 pair hpair

/-- Main audit lemma: When L1, L2, L3 all hold, total violation probability is 0.

    ⚠️ DEPRECATED: Uses unsound axiom. Use `audit_bound_zero_bounded` instead. -/
@[deprecated "Use audit_bound_zero_bounded" (since := "2026-01-10")]
theorem audit_bound_zero (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings)
    (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar)
    (R : ℕ) (hR : R ≥ 1) :
    ViolationProb fstar (ZR g x R T) x = 0 := by
  apply ViolationProb_eq_zero_of_Exp_D_eq_zero
  exact multi_round g T x R fstar hp h1 h2 h3 hR
-/

/-- If L1 holds (zero expected distortion at leaves), leaf violation probability is 0.
    Bounded version using BoundedPseudoMetricSpace. -/
lemma leaf_violation_zero_of_L1_bounded {Strings Y : Type*} [Monoid Strings]
    [BoundedPseudoMetricSpace Y] (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (h1 : L1 g T fstar) :
    ∀ b ∈ leaves T, ViolationProb fstar (g b) b = 0 := by
  intro b hb
  apply ViolationProb_eq_zero_of_Exp_D_eq_zero_bounded
  exact h1 b hb

/-- If L2 holds (zero expected distortion at merges), merge violation probability is 0.
    Bounded version using BoundedPseudoMetricSpace. -/
lemma merge_violation_zero_of_L2_bounded {Strings Y : Type*} [Monoid Strings]
    [BoundedPseudoMetricSpace Y] (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (h2 : L2 g T fstar) :
    ∀ pair ∈ internal_nodes T,
      ViolationProb fstar (reduce g (BinTree.node pair.1 pair.2)) (S (BinTree.node pair.1 pair.2)) = 0 := by
  intro pair hpair
  apply ViolationProb_eq_zero_of_Exp_D_eq_zero_bounded
  exact h2 pair hpair

/-- Main audit lemma: When L1, L2, L3 all hold, total violation probability is 0.
    Bounded version using BoundedPseudoMetricSpace. -/
theorem audit_bound_zero_bounded {Strings Y : Type*} [Monoid Strings]
    [BoundedPseudoMetricSpace Y] (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings)
    (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar)
    (R : ℕ) (hR : R ≥ 1) :
    ViolationProb fstar (ZR g x R T) x = 0 := by
  apply ViolationProb_eq_zero_of_Exp_D_eq_zero_bounded
  exact multi_round_typeclass g T x R fstar hp h1 h2 h3 hR

/-!
## Quantitative Union Bound (Theorem 8.1)

This is the key theorem providing bounds even when local laws hold only approximately.
The bound uses the number of leaves N, internal nodes M, and rounds R.

Key result:
- Δ₁ ≤ N · p_leaf + M · p_merge
- Δ_R ≤ N · p_leaf + M · p_merge + (R-1) · p_idemp  (for R ≥ 2)

where Δ_R = E[D(Z^R, X)] is the expected distortion after R rounds.
-/

/-- Sum of leaf violation probabilities -/
def totalLeafViolation (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) : ℝ :=
  (leaves T).foldl (fun acc b => acc + ViolationProb fstar (g b) b) 0

/-- Sum of merge violation probabilities -/
def totalMergeViolation (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) : ℝ :=
  (internal_nodes T).foldl
    (fun acc (pair : BinTree Strings × BinTree Strings) =>
      let (T_L, T_R) := pair
      acc + ViolationProb fstar (reduce g (BinTree.node T_L T_R)) (S (BinTree.node T_L T_R))) 0

/-- Idempotence violation probability: P[D(g(Z), Z) > 0] for Z in range -/
def pIdemp (g : Summarizer Strings) (fstar : Strings → Y)
    (p : PMF Strings) : ℝ :=
  Exp p (fun z => Exp (g z) (fun w => violationInd fstar w z))

/-- pIdemp is non-negative -/
lemma pIdemp_nonneg (g : Summarizer Strings) (fstar : Strings → Y) (p : PMF Strings) :
    0 ≤ pIdemp g fstar p := by
  unfold pIdemp Exp
  apply tsum_nonneg
  intro z
  apply mul_nonneg ENNReal.toReal_nonneg
  apply tsum_nonneg
  intro w
  apply mul_nonneg ENNReal.toReal_nonneg
  exact violationInd_nonneg fstar w z

/-- Helper: foldl with + preserves non-negativity -/
lemma foldl_add_nonneg {α : Type*} (f : α → ℝ) (l : List α) (init : ℝ)
    (hinit : 0 ≤ init) (hf : ∀ a ∈ l, 0 ≤ f a) :
    0 ≤ l.foldl (fun acc a => acc + f a) init := by
  induction l generalizing init with
  | nil => exact hinit
  | cons a as ih =>
    simp only [List.foldl_cons]
    apply ih
    · have ha : a ∈ a :: as := by simp
      linarith [hf a ha]
    · intro b hb
      exact hf b (by simp [hb])

/-- Total violation probabilities are non-negative -/
lemma totalLeafViolation_nonneg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) : 0 ≤ totalLeafViolation g fstar T := by
  unfold totalLeafViolation
  apply foldl_add_nonneg
  · linarith
  · intro b _
    exact ViolationProb_nonneg fstar (g b) b

lemma totalMergeViolation_nonneg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) : 0 ≤ totalMergeViolation g fstar T := by
  unfold totalMergeViolation
  apply foldl_add_nonneg
  · linarith
  · intro pair _
    exact ViolationProb_nonneg fstar _ _

/-- Helper: foldl add shifts by initial value -/
lemma foldl_add_shift {α : Type*} (f : α → ℝ) (l : List α) (init : ℝ) :
    l.foldl (fun acc a => acc + f a) init = init + l.foldl (fun acc a => acc + f a) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl_cons, zero_add]
    rw [ih (init + f x), ih (f x)]
    ring

/-- Helper: foldl add over concatenated lists -/
lemma foldl_add_append {α : Type*} (f : α → ℝ) (l₁ l₂ : List α) (init : ℝ) :
    (l₁ ++ l₂).foldl (fun acc a => acc + f a) init =
    l₂.foldl (fun acc a => acc + f a) (l₁.foldl (fun acc a => acc + f a) init) := by
  induction l₁ generalizing init with
  | nil => simp
  | cons x xs ih => simp only [List.cons_append, List.foldl_cons]; exact ih (init + f x)

/-- Helper: foldl add starting from 0 over concatenated lists splits -/
lemma foldl_add_append_zero {α : Type*} (f : α → ℝ) (l₁ l₂ : List α) :
    (l₁ ++ l₂).foldl (fun acc a => acc + f a) 0 =
    l₁.foldl (fun acc a => acc + f a) 0 + l₂.foldl (fun acc a => acc + f a) 0 := by
  rw [foldl_add_append, foldl_add_shift]

/-- totalLeafViolation decomposes over tree structure -/
lemma totalLeafViolation_node (g : Summarizer Strings) (fstar : Strings → Y)
    (T_L T_R : BinTree Strings) :
    totalLeafViolation g fstar (BinTree.node T_L T_R) =
    totalLeafViolation g fstar T_L + totalLeafViolation g fstar T_R := by
  unfold totalLeafViolation
  simp only [leaves]
  exact foldl_add_append_zero _ _ _

/-- totalMergeViolation decomposes over tree structure -/
lemma totalMergeViolation_node (g : Summarizer Strings) (fstar : Strings → Y)
    (T_L T_R : BinTree Strings) :
    totalMergeViolation g fstar (BinTree.node T_L T_R) =
    ViolationProb fstar (reduce g (BinTree.node T_L T_R)) (S (BinTree.node T_L T_R)) +
    totalMergeViolation g fstar T_L + totalMergeViolation g fstar T_R := by
  unfold totalMergeViolation
  simp only [internal_nodes, List.foldl_cons, zero_add]
  rw [foldl_add_append]
  -- Now have: foldl (foldl VP (internal_nodes T_L)) (internal_nodes T_R)
  -- Apply shift to first foldl: foldl VP l = VP + foldl 0 l
  rw [foldl_add_shift (f := fun a => ViolationProb fstar (reduce g (BinTree.node a.1 a.2))
      (S (BinTree.node a.1 a.2))) (internal_nodes T_L)]
  -- Now have: foldl (VP + foldl 0 T_L) T_R
  rw [foldl_add_shift (f := fun a => ViolationProb fstar (reduce g (BinTree.node a.1 a.2))
      (S (BinTree.node a.1 a.2))) (internal_nodes T_R)]

/-- Expected distortion is bounded by violation probability (for bounded metrics) -/
lemma Exp_D_le_ViolationProb (fstar : Strings → Y) (p : PMF Strings) (x : Strings)
    (hbound : ∀ z, D fstar z x ≤ 1) :
    Exp p (fun z => D fstar z x) ≤ ViolationProb fstar p x := by
  unfold Exp ViolationProb violationInd
  -- Use bounded summability since D ≤ 1 and violationInd ∈ {0, 1}
  have h_abs_D : ∀ z, |D fstar z x| ≤ 1 := fun z => by
    unfold D; rw [abs_of_nonneg dist_nonneg]; exact hbound z
  have h_abs_viol : ∀ z, |if D fstar z x > 0 then (1 : ℝ) else 0| ≤ 1 := fun z => by
    split_ifs <;> simp
  apply Summable.tsum_le_tsum
  · intro z
    by_cases h : D fstar z x > 0
    · simp [h]
      calc (p z).toReal * D fstar z x
          ≤ (p z).toReal * 1 := by
            apply mul_le_mul_of_nonneg_left (hbound z) ENNReal.toReal_nonneg
        _ = (p z).toReal := by ring
    · push_neg at h
      have hD0 : D fstar z x = 0 := le_antisymm h dist_nonneg
      simp [hD0]
  · exact PMF.summable_coe_real_mul_of_bounded p _ 1 (by norm_num) h_abs_D
  · exact PMF.summable_coe_real_mul_of_bounded p _ 1 (by norm_num) h_abs_viol

/-- Triangle inequality for violation probability:
    If D(a, c) > 0, then either D(a, b) > 0 or D(b, c) > 0 -/
lemma violation_triangle (fstar : Strings → Y) (a b c : Strings)
    (h : violationInd fstar a c = 1) :
    violationInd fstar a b = 1 ∨ violationInd fstar b c = 1 := by
  -- Extract that D(a, c) > 0 from violation indicator
  have hDac : D fstar a c > 0 := (violationInd_eq_one_iff fstar a c).mp h
  -- Contrapositive: if both D(a,b) = 0 and D(b,c) = 0, then D(a,c) = 0
  by_contra hne
  push_neg at hne
  have hab : violationInd fstar a b ≠ 1 := hne.1
  have hbc : violationInd fstar b c ≠ 1 := hne.2
  -- violationInd ≠ 1 means D = 0 (since indicator is 0 or 1)
  have hDab : D fstar a b = 0 := by
    by_contra hne
    have hpos : D fstar a b > 0 := lt_of_le_of_ne dist_nonneg (Ne.symm hne)
    have := (violationInd_eq_one_iff fstar a b).mpr hpos
    exact hab this
  have hDbc : D fstar b c = 0 := by
    by_contra hne
    have hpos : D fstar b c > 0 := lt_of_le_of_ne dist_nonneg (Ne.symm hne)
    have := (violationInd_eq_one_iff fstar b c).mpr hpos
    exact hbc this
  -- Triangle inequality: D(a,c) ≤ D(a,b) + D(b,c) = 0
  unfold D at *
  have htri := dist_triangle (fstar a) (fstar b) (fstar c)
  linarith

/-- D is bounded by violation indicator when metric is bounded by 1 -/
lemma D_le_violationInd_of_bounded (fstar : Strings → Y) (z x : Strings)
    (hbound : D fstar z x ≤ 1) :
    D fstar z x ≤ violationInd fstar z x := by
  unfold violationInd
  by_cases h : D fstar z x > 0
  · simp [h]; exact hbound
  · push_neg at h
    have hD0 : D fstar z x = 0 := le_antisymm h dist_nonneg
    simp [hD0]

/-- Triangle inequality for violation indicators -/
lemma violationInd_triangle_le (fstar : Strings → Y) (a b c : Strings) :
    violationInd fstar a c ≤ violationInd fstar a b + violationInd fstar b c := by
  by_cases h : violationInd fstar a c = 0
  · simp [h]; exact add_nonneg (violationInd_nonneg fstar a b) (violationInd_nonneg fstar b c)
  · -- violationInd a c = 1 (since it's not 0 and can only be 0 or 1)
    have h1 : violationInd fstar a c = 1 := by
      -- violationInd is defined as if-then-else with values 0 or 1
      unfold violationInd at h ⊢
      split_ifs with hD
      · rfl
      · simp at h; exact absurd h hD
    rw [h1]
    have htri := violation_triangle fstar a b c h1
    cases htri with
    | inl hab =>
      calc (1 : ℝ) = 1 + 0 := by ring
        _ ≤ violationInd fstar a b + violationInd fstar b c := by
          apply add_le_add
          · exact le_of_eq hab.symm
          · exact violationInd_nonneg fstar b c
    | inr hbc =>
      calc (1 : ℝ) = 0 + 1 := by ring
        _ ≤ violationInd fstar a b + violationInd fstar b c := by
          apply add_le_add
          · exact violationInd_nonneg fstar a b
          · exact le_of_eq hbc.symm

/-!
## Idempotence Monotonicity from L3

When L3 holds (expected distortion is 0 for any oracle in range), we can prove
that applying the summarizer one more time produces zero idempotence violation.
This is because elements in the range of g stay in the range, and L3 guarantees
zero distortion for in-range elements.
-/

/-- Elements in the support of p.bind g are in the range of g -/
lemma bind_support_InRange (g : Summarizer Strings) (p : PMF Strings)
    (z : Strings) (hz : z ∈ (p.bind g).support) : InRange g z := by
  rw [PMF.mem_support_bind_iff] at hz
  obtain ⟨x, _, hzx⟩ := hz
  exact ⟨x, hzx⟩

/-- When L3 holds, inner expectation of violation indicator is 0 for in-range z -/
lemma inner_exp_zero_of_L3 (g : Summarizer Strings) (fstar : Strings → Y)
    (h3 : L3 g fstar) (z : Strings) (hz : InRange g z)
    (h_summable : Summable (fun w => (g z w).toReal * D fstar w z)) :
    Exp (g z) (fun w => violationInd fstar w z) = 0 := by
  have h_dist_zero : ∀ w ∈ (g z).support, D fstar w z = 0 :=
    L3_implies_dist_zero_on_support g fstar h3 z hz h_summable
  unfold Exp
  convert tsum_zero with w
  by_cases hw : w ∈ (g z).support
  · have h_viol_zero : violationInd fstar w z = 0 := by
      rw [violationInd_eq_zero_iff]
      exact h_dist_zero w hw
    simp [h_viol_zero]
  · simp [PMF.mem_support_iff] at hw
    simp [hw]

/-- When L3 holds, pIdemp of p.bind g is 0 -/
theorem pIdemp_bind_zero_of_L3 (g : Summarizer Strings) (fstar : Strings → Y)
    (h3 : L3 g fstar)
    (h_summable : ∀ z, InRange g z → Summable (fun w => (g z w).toReal * D fstar w z)) :
    ∀ p : PMF Strings, pIdemp g fstar (p.bind g) = 0 := by
  intro p
  unfold pIdemp Exp
  convert tsum_zero with z
  simp only
  by_cases hz : z ∈ (p.bind g).support
  · have hz_range : InRange g z := bind_support_InRange g p z hz
    have h_inner_zero : Exp (g z) (fun w => violationInd fstar w z) = 0 :=
      inner_exp_zero_of_L3 g fstar h3 z hz_range (h_summable z hz_range)
    unfold Exp at h_inner_zero
    simp only at h_inner_zero
    rw [h_inner_zero]
    ring
  · simp only [PMF.mem_support_iff, ne_eq, not_not] at hz
    rw [hz]
    simp

/-- Idempotence Monotonicity from L3: p.bind g has zero pIdemp, hence ≤ pIdemp(p) -/
theorem idemp_monotone_from_L3 (g : Summarizer Strings) (fstar : Strings → Y)
    (h3 : L3 g fstar)
    (h_summable : ∀ z, InRange g z → Summable (fun w => (g z w).toReal * D fstar w z)) :
    ∀ p : PMF Strings, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p := by
  intro p
  rw [pIdemp_bind_zero_of_L3 g fstar h3 h_summable p]
  exact pIdemp_nonneg g fstar p

/-- pIdemp for ZR is bounded by pIdemp for reduce.
    Requires hypothesis that idempotence violation decreases under bind. -/
lemma pIdemp_ZR_le_reduce (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    pIdemp g fstar (ZR g x R T) ≤ pIdemp g fstar (reduce g T) := by
  induction R with
  | zero => exact absurd hR (by decide)
  | succ R' ih =>
    cases R' with
    | zero =>
      -- R = 1: ZR g x 1 T = reduce g T
      simp only [ZR]
      exact le_refl _
    | succ R'' =>
      -- R = R'' + 2: ZR g x (R''+2) T = (ZR g x (R''+1) T).bind g
      simp only [ZR]
      calc pIdemp g fstar ((ZR g x (R'' + 1) T).bind g)
          ≤ pIdemp g fstar (ZR g x (R'' + 1) T) := h_mono _
        _ ≤ pIdemp g fstar (reduce g T) := ih (by omega)

/-!
## Quantitative Union Bound Theorem

The theorem states that expected distortion after R rounds is bounded by the sum of
violation probabilities at leaves, internal nodes, and idempotence violations.

Proof sketch:
1. If D(Z^R, X) > 0, then at some step in the computation, a local violation occurred
2. By contrapositive of the preservation theorems, if all local laws held exactly,
   distortion would be 0
3. Union bound gives: P[D > 0] ≤ ∑ P[local violations]
4. Expected distortion (for bounded metrics) is bounded by violation probability
-/

/-- One-round distortion bound (Theorem 8.1, R=1 case)

This theorem states that expected distortion after one round of reduction is bounded
by the sum of leaf and merge violation probabilities.

Proof strategy (tree induction):

**Base case (BinTree.leaf b):**
- `reduce g (leaf b) = g b`
- `totalLeafViolation g fstar (leaf b) = ViolationProb fstar (g b) b`
- `totalMergeViolation g fstar (leaf b) = 0`
- By `Exp_D_le_ViolationProb`: `Exp (g b) (D fstar · b) ≤ ViolationProb fstar (g b) b` ✓

**Inductive case (BinTree.node T_L T_R):**
- `reduce g (node T_L T_R) = (reduce g T_L).bind (fun s_L => (reduce g T_R).bind (fun s_R => g (s_L * s_R)))`
- Use triangle inequality: `D(z, x) ≤ D(z, s_L * s_R) + D(s_L * s_R, x)`
- The first term (merge distortion) is bounded by the merge violation at this node
- For `D(s_L * s_R, x)`, decompose using `x = S T_L * S T_R`:
  - `D(s_L * s_R, x) ≤ D(s_L, S T_L) + D(s_R, S T_R)` (by metric properties on the oracle)
- Apply inductive hypothesis to bound distortions from subtrees
- Sum gives: leaves violations from T_L + T_R + merge violations from T_L + T_R + this node

The full formalization requires:
1. Helper lemmas decomposing totalLeafViolation/totalMergeViolation over tree structure
2. Careful tracking of how Exp distributes over bind
3. Triangle inequality for D applied at each merge step
-/
theorem union_bound_one_round (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (_hp : S T = x)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1) :
    Exp (reduce g T) (fun z => D fstar z x) ≤
    totalLeafViolation g fstar T + totalMergeViolation g fstar T := by
  induction T generalizing x with
  | leaf b =>
    -- Base case: reduce g (leaf b) = g b
    simp only [reduce, totalLeafViolation, totalMergeViolation, leaves, internal_nodes,
               List.foldl_cons, List.foldl_nil, zero_add, add_zero]
    have hx : x = b := by simp only [S] at _hp; exact _hp.symm
    rw [hx]
    exact Exp_D_le_ViolationProb fstar (g b) b (fun z => hbound_global z b)
  | node T_L T_R ih_L ih_R =>
    -- Inductive case: The key insight is that ViolationProb(reduce node, S node) is
    -- included in totalMergeViolation, so we can use Exp_D_le_ViolationProb.
    rw [totalLeafViolation_node, totalMergeViolation_node]
    -- Extract key identity: x = S T_L * S T_R
    have hx : x = S T_L * S T_R := by simp only [S] at _hp; exact _hp.symm
    have hx_eq : x = S (BinTree.node T_L T_R) := by simp only [S]; exact hx
    -- Step 1: Use Exp_D_le_ViolationProb for the main bound
    have h_main : Exp (reduce g (BinTree.node T_L T_R)) (fun z => D fstar z x) ≤
        ViolationProb fstar (reduce g (BinTree.node T_L T_R)) (S (BinTree.node T_L T_R)) := by
      rw [hx_eq]
      exact Exp_D_le_ViolationProb fstar (reduce g (BinTree.node T_L T_R))
        (S (BinTree.node T_L T_R)) (fun z => by rw [← hx_eq]; exact hbound z)
    -- Step 2: Use pre-existing non-negativity lemmas
    have h_merge_nonneg_L := totalMergeViolation_nonneg g fstar T_L
    have h_merge_nonneg_R := totalMergeViolation_nonneg g fstar T_R
    have h_leaf_nonneg_L := totalLeafViolation_nonneg g fstar T_L
    have h_leaf_nonneg_R := totalLeafViolation_nonneg g fstar T_R
    -- Combine the inequalities
    linarith

/-- Multi-round distortion bound (Theorem 8.1) - bounded version.

    This version uses bounded summability helpers instead of the unsound axiom.
    All deprecated calls are replaced with their bounded alternatives.

    Note: The proof follows the same structure as union_bound_multi_round but with
    explicit bound proofs instead of the axiom. -/
theorem union_bound_multi_round_bounded (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x) (R : ℕ) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    Exp (ZR g x R T) (fun z => D fstar z x) ≤
    totalLeafViolation g fstar T + totalMergeViolation g fstar T +
    (R - 1) * pIdemp g fstar (reduce g T) := by
  -- Derive bound hypotheses: |D fstar w z| = D fstar w z since D ≥ 0
  have hD_abs_bound : ∀ w z, |D fstar w z| ≤ 1 := fun w z => by
    have h1 : D fstar w z ≥ 0 := dist_nonneg
    rw [abs_of_nonneg h1]; exact hbound_global w z
  have hD_abs_bound_x : ∀ z, |D fstar z x| ≤ 1 := fun z => by
    have h1 : D fstar z x ≥ 0 := dist_nonneg
    rw [abs_of_nonneg h1]; exact hbound z
  induction R with
  | zero => exact absurd hR (by decide)
  | succ R' ih =>
    cases R' with
    | zero =>
      simp only [ZR]
      have h_coeff : ((0 + 1 : ℕ) : ℝ) - 1 = 0 := by norm_num
      rw [h_coeff, zero_mul, add_zero]
      exact union_bound_one_round g fstar T x hp hbound hbound_global
    | succ R'' =>
      simp only [ZR]
      have hD_nonneg : ∀ w, 0 ≤ D fstar w x := fun _ => dist_nonneg
      -- Use Exp_bind_eq' with summability proof
      have h_summable_inner : ∀ z, Summable fun w => ((g z) w).toReal * D fstar w x :=
        fun z => PMF.summable_coe_real_mul_of_bounded (g z) _ 1 (by norm_num) hD_abs_bound_x
      rw [Exp_bind_eq' (ZR g x (R'' + 1) T) g (fun w => D fstar w x) hD_nonneg h_summable_inner]
      -- Apply triangle inequality using Exp_mono_bounded
      calc Exp (ZR g x (R'' + 1) T) (fun z => Exp (g z) (fun w => D fstar w x))
          ≤ Exp (ZR g x (R'' + 1) T) (fun z => Exp (g z) (fun w => D fstar w z + D fstar z x)) := by
            -- Bound for inner Exp values
            have h_inner_bound : ∀ z, |Exp (g z) (fun w => D fstar w x)| ≤ 1 := fun z => by
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => D fstar w x) := by
                unfold Exp; apply tsum_nonneg; intro w
                exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
              rw [abs_of_nonneg hExp_nonneg]
              calc Exp (g z) (fun w => D fstar w x)
                  ≤ Exp (g z) (fun _ => 1) := by
                    apply Exp_mono_bounded (g z) _ _ 1 (by norm_num)
                      (fun _ => hbound _) hD_abs_bound_x (fun _ => by norm_num)
                _ = 1 := by unfold Exp; simp only [mul_one]; exact PMF.toReal_tsum_coe (g z)
            have h_inner_bound' : ∀ z, |Exp (g z) (fun w => D fstar w z + D fstar z x)| ≤ 2 := fun z => by
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => D fstar w z + D fstar z x) := by
                unfold Exp; apply tsum_nonneg; intro w
                apply mul_nonneg ENNReal.toReal_nonneg
                have h1 : D fstar w z ≥ 0 := dist_nonneg
                have h2 : D fstar z x ≥ 0 := dist_nonneg
                linarith
              rw [abs_of_nonneg hExp_nonneg]
              have hsum_bound : ∀ w, D fstar w z + D fstar z x ≤ 2 := fun w => by
                linarith [hbound_global w z, hbound z]
              have hsum_abs_bound : ∀ w, |D fstar w z + D fstar z x| ≤ 2 := fun w => by
                have h1 : D fstar w z ≥ 0 := dist_nonneg
                have h2 : D fstar z x ≥ 0 := dist_nonneg
                rw [abs_of_nonneg (by linarith)]
                exact hsum_bound w
              calc Exp (g z) (fun w => D fstar w z + D fstar z x)
                  ≤ Exp (g z) (fun _ => 2) := by
                    apply Exp_mono_bounded (g z) _ _ 2 (by norm_num) hsum_bound hsum_abs_bound (fun _ => by norm_num)
                _ = 2 := by
                    unfold Exp; simp only [mul_one]
                    calc ∑' w, ((g z) w).toReal * 2
                        = 2 * ∑' w, ((g z) w).toReal := by rw [tsum_mul_right]; ring
                      _ = 2 * 1 := by rw [PMF.toReal_tsum_coe (g z)]
                      _ = 2 := by ring
            -- Bounds for outer Exp_mono_bounded
            have hf1_abs : ∀ z, |Exp (g z) (fun w => D fstar w x)| ≤ 2 := fun z =>
              le_trans (h_inner_bound z) (by norm_num : (1 : ℝ) ≤ 2)
            apply Exp_mono_bounded (ZR g x (R'' + 1) T) _ _ 2 (by norm_num)
            · intro z; apply Exp_mono_bounded (g z) _ _ 2 (by norm_num)
              · intro w; exact D_triangle fstar w z x
              · intro w; exact le_trans (hD_abs_bound_x w) (by norm_num : (1 : ℝ) ≤ 2)
              · intro w
                have h1 : D fstar w z ≥ 0 := dist_nonneg
                have h2 : D fstar z x ≥ 0 := dist_nonneg
                rw [abs_of_nonneg (by linarith)]
                linarith [hbound_global w z, hbound z]
            · exact hf1_abs
            · exact h_inner_bound'
        _ = Exp (ZR g x (R'' + 1) T) (fun z => Exp (g z) (fun w => D fstar w z) + D fstar z x) := by
            congr 1
            ext z
            have hDwz_summable : Summable (fun w => ((g z) w).toReal * D fstar w z) :=
              PMF.summable_coe_real_mul_of_bounded (g z) _ 1 (by norm_num) (fun w => hD_abs_bound w z)
            have hconst_summable : Summable (fun w => ((g z) w).toReal * D fstar z x) :=
              PMF.summable_coe_real_mul_of_bounded (g z) _ 1 (by norm_num) (fun _ => hD_abs_bound_x z)
            rw [Exp_add (g z) (fun w => D fstar w z) (fun _ => D fstar z x) hDwz_summable hconst_summable]
            congr 1
            unfold Exp
            conv_rhs => rw [← one_mul (D fstar z x), ← PMF.toReal_tsum_coe (g z)]
            rw [tsum_mul_right]
        _ = Exp (ZR g x (R'' + 1) T) (fun z => Exp (g z) (fun w => D fstar w z)) +
            Exp (ZR g x (R'' + 1) T) (fun z => D fstar z x) := by
            have h_bound_EExp : ∀ z, |Exp (g z) (fun w => D fstar w z)| ≤ 1 := fun z => by
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => D fstar w z) := by
                unfold Exp; apply tsum_nonneg; intro w
                exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
              rw [abs_of_nonneg hExp_nonneg]
              calc Exp (g z) (fun w => D fstar w z)
                  ≤ Exp (g z) (fun _ => 1) := by
                    apply Exp_mono_bounded (g z) _ _ 1 (by norm_num) (fun w => hbound_global w z)
                      (fun w => hD_abs_bound w z) (fun _ => by norm_num)
                _ = 1 := by unfold Exp; simp only [mul_one]; exact PMF.toReal_tsum_coe (g z)
            have h1_summable : Summable (fun z => ((ZR g x (R'' + 1) T) z).toReal *
                Exp (g z) (fun w => D fstar w z)) :=
              PMF.summable_coe_real_mul_of_bounded _ _ 1 (by norm_num) h_bound_EExp
            have h2_summable : Summable (fun z => ((ZR g x (R'' + 1) T) z).toReal * D fstar z x) :=
              PMF.summable_coe_real_mul_of_bounded _ _ 1 (by norm_num) hD_abs_bound_x
            exact Exp_add _ _ _ h1_summable h2_summable
        _ ≤ pIdemp g fstar (ZR g x (R'' + 1) T) +
            Exp (ZR g x (R'' + 1) T) (fun z => D fstar z x) := by
            have h_violInd_bound : ∀ w z, |violationInd fstar w z| ≤ 1 := fun w z => by
              rw [abs_of_nonneg (violationInd_nonneg fstar w z)]
              exact violationInd_le_one fstar w z
            have h_EviolInd_bound : ∀ z, |Exp (g z) (fun w => violationInd fstar w z)| ≤ 1 := fun z => by
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => violationInd fstar w z) := by
                unfold Exp; apply tsum_nonneg; intro w
                exact mul_nonneg ENNReal.toReal_nonneg (violationInd_nonneg fstar w z)
              rw [abs_of_nonneg hExp_nonneg]
              calc Exp (g z) (fun w => violationInd fstar w z)
                  ≤ Exp (g z) (fun _ => 1) := by
                    apply Exp_mono_bounded (g z) _ _ 1 (by norm_num) (fun w => violationInd_le_one fstar w z)
                      (fun w => h_violInd_bound w z) (fun _ => by norm_num)
                _ = 1 := by unfold Exp; simp only [mul_one]; exact PMF.toReal_tsum_coe (g z)
            have h_EExp_bound : ∀ z, |Exp (g z) (fun w => D fstar w z)| ≤ 1 := fun z => by
              have hExp_nonneg : 0 ≤ Exp (g z) (fun w => D fstar w z) := by
                unfold Exp; apply tsum_nonneg; intro w
                exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
              rw [abs_of_nonneg hExp_nonneg]
              calc Exp (g z) (fun w => D fstar w z)
                  ≤ Exp (g z) (fun _ => 1) := by
                    apply Exp_mono_bounded (g z) _ _ 1 (by norm_num) (fun w => hbound_global w z)
                      (fun w => hD_abs_bound w z) (fun _ => by norm_num)
                _ = 1 := by unfold Exp; simp only [mul_one]; exact PMF.toReal_tsum_coe (g z)
            have h_le : Exp (ZR g x (R'' + 1) T) (fun z => Exp (g z) (fun w => D fstar w z)) ≤
                        pIdemp g fstar (ZR g x (R'' + 1) T) := by
              unfold pIdemp
              apply Exp_mono_bounded (ZR g x (R'' + 1) T) _ _ 1 (by norm_num)
              · intro z
                apply Exp_mono_bounded (g z) _ _ 1 (by norm_num)
                · intro w; exact D_le_violationInd_of_bounded fstar w z (hbound_global w z)
                · intro w; exact hD_abs_bound w z
                · intro w; exact h_violInd_bound w z
              · exact h_EExp_bound
              · exact h_EviolInd_bound
            linarith
        _ ≤ pIdemp g fstar (reduce g T) +
            Exp (ZR g x (R'' + 1) T) (fun z => D fstar z x) := by
            have h_le := pIdemp_ZR_le_reduce g fstar T x (R'' + 1) (by omega) h_mono
            linarith
        _ ≤ pIdemp g fstar (reduce g T) +
            (totalLeafViolation g fstar T + totalMergeViolation g fstar T +
             (R'' + 1 - 1) * pIdemp g fstar (reduce g T)) := by
            have ih_bound := ih (by omega)
            simp only [ZR] at ih_bound
            have h_coerce : (↑(R'' + 1) - 1 : ℝ) = (↑R'' + 1 - 1 : ℝ) := by push_cast; ring
            rw [h_coerce] at ih_bound
            linarith
        _ = totalLeafViolation g fstar T + totalMergeViolation g fstar T +
            ((R'' + 2 : ℕ) - 1) * pIdemp g fstar (reduce g T) := by
            have h1 : (↑R'' + 1 - 1 : ℝ) = (R'' : ℝ) := by ring
            have h2 : (↑(R'' + 2) - 1 : ℝ) = (↑R'' + 1 : ℝ) := by push_cast; ring
            rw [h1, h2]
            ring

/-
-- COMMENTED OUT: Deprecated theorem using unsound axiom
-- Use union_bound_zero_implies_distortion_zero_bounded instead

/-- Qualitative corollary: When all violations are 0, distortion is 0.

Note: The idempotence hypothesis is strengthened to require zero violation for ALL
elements in InRange g, not just those in support(reduce g T). This is the correct
mathematical condition for L3 to hold globally.

⚠️ DEPRECATED: Uses unsound axiom. Use `union_bound_zero_implies_distortion_zero_bounded` instead. -/
@[deprecated "Use union_bound_zero_implies_distortion_zero_bounded" (since := "2026-01-10")]
theorem union_bound_zero_implies_distortion_zero (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x) (R : ℕ) (hR : R ≥ 1)
    (hLeaf : totalLeafViolation g fstar T = 0)
    (hMerge : totalMergeViolation g fstar T = 0)
    (hIdemp : ∀ Z, InRange g Z → ViolationProb fstar (g Z) Z = 0) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
  -- When all violations are 0, the local laws L1, L2, L3 hold
  -- and we can apply multi_round directly
  have h1 : L1 g T fstar := by
    intro b hb
    -- totalLeafViolation = 0 implies each leaf violation = 0
    unfold totalLeafViolation at hLeaf
    have hVP_nonneg : ∀ c ∈ leaves T, 0 ≤ ViolationProb fstar (g c) c :=
      fun c _ => ViolationProb_nonneg fstar (g c) c
    have hVP_zero : ViolationProb fstar (g b) b = 0 :=
      foldl_add_eq_zero_implies_all_zero _ _ hVP_nonneg hLeaf b hb
    -- ViolationProb = 0 implies Exp D = 0
    have hExp_zero := Exp_D_eq_zero_of_ViolationProb_eq_zero fstar (g b) b hVP_zero
    -- Eg g f b = Exp (g b) f by definition
    unfold Eg
    exact hExp_zero
  have h2 : L2 g T fstar := by
    intro pair hpair
    -- totalMergeViolation = 0 implies each merge violation = 0
    unfold totalMergeViolation at hMerge
    let f := fun (p : BinTree Strings × BinTree Strings) =>
      ViolationProb fstar (reduce g (BinTree.node p.1 p.2)) (S (BinTree.node p.1 p.2))
    have hVP_nonneg : ∀ p ∈ internal_nodes T, 0 ≤ f p :=
      fun p _ => ViolationProb_nonneg fstar _ _
    have hVP_zero : f pair = 0 :=
      foldl_add_eq_zero_implies_all_zero f _ hVP_nonneg hMerge pair hpair
    -- ViolationProb = 0 implies Exp D = 0
    have hExp_zero := Exp_D_eq_zero_of_ViolationProb_eq_zero fstar
      (reduce g (BinTree.node pair.1 pair.2)) (S (BinTree.node pair.1 pair.2)) hVP_zero
    -- Egu g T f = Exp (reduce g T) f by definition
    unfold Egu
    exact hExp_zero
  have h3 : L3 g fstar := by
    intro Z hZ
    -- With strengthened hIdemp, this follows directly
    have hVP_zero := hIdemp Z hZ
    exact Exp_D_eq_zero_of_ViolationProb_eq_zero fstar (g Z) Z hVP_zero
  exact multi_round g T x R fstar hp h1 h2 h3 hR
-/

/-- Qualitative corollary: When all violations are 0, distortion is 0.
    Bounded version using BoundedPseudoMetricSpace.

Note: The idempotence hypothesis is strengthened to require zero violation for ALL
elements in InRange g, not just those in support(reduce g T). This is the correct
mathematical condition for L3 to hold globally. -/
theorem union_bound_zero_implies_distortion_zero_bounded {Strings Y : Type*} [Monoid Strings]
    [BoundedPseudoMetricSpace Y] (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x) (R : ℕ) (hR : R ≥ 1)
    (hLeaf : totalLeafViolation g fstar T = 0)
    (hMerge : totalMergeViolation g fstar T = 0)
    (hIdemp : ∀ Z, InRange g Z → ViolationProb fstar (g Z) Z = 0) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
  -- When all violations are 0, the local laws L1, L2, L3 hold
  -- and we can apply multi_round_typeclass directly
  have h1 : L1 g T fstar := by
    intro b hb
    -- totalLeafViolation = 0 implies each leaf violation = 0
    unfold totalLeafViolation at hLeaf
    have hVP_nonneg : ∀ c ∈ leaves T, 0 ≤ ViolationProb fstar (g c) c :=
      fun c _ => ViolationProb_nonneg fstar (g c) c
    have hVP_zero : ViolationProb fstar (g b) b = 0 :=
      foldl_add_eq_zero_implies_all_zero _ _ hVP_nonneg hLeaf b hb
    -- ViolationProb = 0 implies Exp D = 0 (using bounded version)
    have hExp_zero := Exp_D_eq_zero_of_ViolationProb_eq_zero_bounded fstar (g b) b hVP_zero
    -- Eg g f b = Exp (g b) f by definition
    unfold Eg
    exact hExp_zero
  have h2 : L2 g T fstar := by
    intro pair hpair
    -- totalMergeViolation = 0 implies each merge violation = 0
    unfold totalMergeViolation at hMerge
    let f := fun (p : BinTree Strings × BinTree Strings) =>
      ViolationProb fstar (reduce g (BinTree.node p.1 p.2)) (S (BinTree.node p.1 p.2))
    have hVP_nonneg : ∀ p ∈ internal_nodes T, 0 ≤ f p :=
      fun p _ => ViolationProb_nonneg fstar _ _
    have hVP_zero : f pair = 0 :=
      foldl_add_eq_zero_implies_all_zero f _ hVP_nonneg hMerge pair hpair
    -- ViolationProb = 0 implies Exp D = 0 (using bounded version)
    have hExp_zero := Exp_D_eq_zero_of_ViolationProb_eq_zero_bounded fstar
      (reduce g (BinTree.node pair.1 pair.2)) (S (BinTree.node pair.1 pair.2)) hVP_zero
    -- Egu g T f = Exp (reduce g T) f by definition
    unfold Egu
    exact hExp_zero
  have h3 : L3 g fstar := by
    intro Z hZ
    -- With strengthened hIdemp, this follows directly (using bounded version)
    have hVP_zero := hIdemp Z hZ
    exact Exp_D_eq_zero_of_ViolationProb_eq_zero_bounded fstar (g Z) Z hVP_zero
  exact multi_round_typeclass g T x R fstar hp h1 h2 h3 hR

end AuditBounds

end

end

/-! ## From FormalProofs/OPT/AuditSizes.lean (consolidated 2026-07-02) -/

/-
FormalProofs/OPT/AuditSizes.lean

Count-based audit bounds and sample-size scaling with tree size.

This file is intentionally "combinatorial": it rewrites the existing union bounds
from `AuditBounds.lean` in terms of tree counts (`numLeaves`, `numInternalNodes`)
and the *average* violation rates (`pLeafAvg`, `pMergeAvg`).

It also contains a small sample-complexity lemma showing how Hoeffding margins
scale when the total error budget is split across `N` leaves (or merges).
-/

section

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

section AuditSizes

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Turning totals into (count × average)

`AuditBounds.lean` defines:
- `totalLeafViolation` / `totalMergeViolation` as explicit sums over leaves/internal nodes
- `pLeafAvg` / `pMergeAvg` as the corresponding averages

The lemmas below connect them to the *structural* counts (`numLeaves`, `numInternalNodes`).
-/

/-- Total leaf violation equals `(numLeaves) × (average leaf violation)`. -/
lemma totalLeafViolation_eq_numLeaves_mul_pLeafAvg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) :
    totalLeafViolation g fstar T = (numLeaves T : ℝ) * pLeafAvg g fstar T := by
  have hpos : 0 < numLeaves T := numLeaves_pos T
  have hnum_ne : numLeaves T ≠ 0 := Nat.ne_of_gt hpos
  have hlen_ne : (leaves T).length ≠ 0 := by
    have : 0 < (leaves T).length := by
      simpa [leaves_length_eq T] using hpos
    exact Nat.ne_of_gt this
  have hN_ne : (numLeaves T : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt hpos)
  -- Unfold and simplify the `if` in `pLeafAvg` using non-emptiness.
  unfold totalLeafViolation pLeafAvg
  simp [hlen_ne, leaves_length_eq, hnum_ne]

/-- Total merge violation equals `(numInternalNodes) × (average merge violation)`. -/
lemma totalMergeViolation_eq_numInternal_mul_pMergeAvg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) :
    totalMergeViolation g fstar T = (numInternalNodes T : ℝ) * pMergeAvg g fstar T := by
  by_cases hlen : (internal_nodes T).length = 0
  · -- No internal nodes: both sides are 0.
    have hnodes : internal_nodes T = [] := List.length_eq_zero_iff.mp hlen
    have hnum : numInternalNodes T = 0 := by
      -- `internal_nodes_length_eq` gives `length = numInternalNodes`.
      simpa [internal_nodes_length_eq T] using hlen
    unfold totalMergeViolation pMergeAvg
    simp [hnodes, hlen, hnum]
  · -- Nonempty internal nodes: same algebra as leaves.
    have hlen_pos : 0 < (internal_nodes T).length := Nat.pos_of_ne_zero hlen
    have hlen_ne : (internal_nodes T).length ≠ 0 := Nat.ne_of_gt hlen_pos
    have hnum_ne_nat : numInternalNodes T ≠ 0 := by
      intro h0
      apply hlen
      simp [internal_nodes_length_eq T, h0]
    have hM_ne : (numInternalNodes T : ℝ) ≠ 0 := by
      exact_mod_cast hnum_ne_nat
    unfold totalMergeViolation pMergeAvg
    simp [hlen_ne, internal_nodes_length_eq, hnum_ne_nat]

/-- Internal nodes = leaves - 1, cast to `ℝ`. -/
lemma numInternalNodes_real_eq_numLeaves_real_sub_one {α : Type*} (T : BinTree α) :
    (numInternalNodes T : ℝ) = (numLeaves T : ℝ) - 1 := by
  have hle : 1 ≤ numLeaves T := Nat.succ_le_of_lt (numLeaves_pos T)
  calc
    (numInternalNodes T : ℝ)
        = (numLeaves T - 1 : ℕ) := by
            exact_mod_cast (internal_eq_leaves_minus_one T)
    _ = (numLeaves T : ℝ) - 1 := by
          simpa using (Nat.cast_sub hle)

/-- Union bound (R=1) rewritten using counts and average violation rates. -/
theorem union_bound_one_round_counts (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1) :
    Exp (reduce g T) (fun z => D fstar z x) ≤
      (numLeaves T : ℝ) * pLeafAvg g fstar T +
      (numInternalNodes T : ℝ) * pMergeAvg g fstar T := by
  have h := union_bound_one_round g fstar T x hp hbound hbound_global
  calc
    Exp (reduce g T) (fun z => D fstar z x)
        ≤ totalLeafViolation g fstar T + totalMergeViolation g fstar T := h
    _ = (numLeaves T : ℝ) * pLeafAvg g fstar T +
        (numInternalNodes T : ℝ) * pMergeAvg g fstar T := by
          simp [totalLeafViolation_eq_numLeaves_mul_pLeafAvg,
            totalMergeViolation_eq_numInternal_mul_pMergeAvg]

/-- Union bound (R rounds) rewritten using counts and average violation rates. -/
theorem union_bound_multi_round_counts (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x)
    (R : ℕ) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    Exp (ZR g x R T) (fun z => D fstar z x) ≤
      (numLeaves T : ℝ) * pLeafAvg g fstar T +
      (numInternalNodes T : ℝ) * pMergeAvg g fstar T +
      (R - 1) * pIdemp g fstar (reduce g T) := by
  have h := union_bound_multi_round_bounded g fstar T x hp R hR hbound hbound_global h_mono
  calc
    Exp (ZR g x R T) (fun z => D fstar z x) ≤
        totalLeafViolation g fstar T + totalMergeViolation g fstar T +
          (R - 1) * pIdemp g fstar (reduce g T) := h
    _ = (numLeaves T : ℝ) * pLeafAvg g fstar T +
        (numInternalNodes T : ℝ) * pMergeAvg g fstar T +
          (R - 1) * pIdemp g fstar (reduce g T) := by
          simp [totalLeafViolation_eq_numLeaves_mul_pLeafAvg,
            totalMergeViolation_eq_numInternal_mul_pMergeAvg]

/-!
## Sample complexity scaling with the number of leaves

`AuditCore.sample_complexity` inverts Hoeffding's margin:
if `n ≥ sample_complexity eps δ` then `confidence_margin δ n ≤ eps`.

When using a union bound with `N` leaves, to make `N * margin ≤ ε` it suffices
to require `margin ≤ ε / N`.
-/

/-- If `n` is chosen for margin `ε/N`, then `N * margin ≤ ε`. -/
lemma leaf_scaled_margin (T : BinTree Strings) (ε δ : ℝ)
    (hε : 0 < ε) (hδ : 0 < δ) (hδ' : δ < 2)
    (n : ℕ) (hn : n ≥ sample_complexity (ε / (numLeaves T : ℝ)) δ) :
    (numLeaves T : ℝ) * confidence_margin δ n ≤ ε := by
  have hN_pos : 0 < (numLeaves T : ℝ) := by
    exact_mod_cast (numLeaves_pos T)
  have hN_ne : (numLeaves T : ℝ) ≠ 0 := ne_of_gt hN_pos
  have hε' : 0 < ε / (numLeaves T : ℝ) := div_pos hε hN_pos
  have hcm : confidence_margin δ n ≤ ε / (numLeaves T : ℝ) :=
    sample_complexity_gives_margin (ε / (numLeaves T : ℝ)) δ hε' hδ hδ' n hn
  have hmul :
      (numLeaves T : ℝ) * confidence_margin δ n ≤
        (numLeaves T : ℝ) * (ε / (numLeaves T : ℝ)) :=
    mul_le_mul_of_nonneg_left hcm (le_of_lt hN_pos)
  calc
    (numLeaves T : ℝ) * confidence_margin δ n
        ≤ (numLeaves T : ℝ) * (ε / (numLeaves T : ℝ)) := hmul
    _ = ε := by
        field_simp [hN_ne]

end AuditSizes

/-!
## Corpus-level size relationships (expectations)

If documents are sampled from a PMF `μ` and each document induces a tree (a chunking),
then we can talk about *expected* tree sizes across the corpus.

We prove the structural identity
`E[numInternalNodes] = E[numLeaves] - 1`
under a mild summability hypothesis (needed only to legally split `tsum`).
-/

section CorpusExpectations

variable {Doc : Type*} {α : Type*}

/-- Expected number of leaves under a document distribution. -/
def expectedNumLeaves (μ : PMF Doc) (treeOf : Doc → BinTree α) : ℝ :=
  Exp μ (fun d => (numLeaves (treeOf d) : ℝ))

/-- Expected number of internal nodes under a document distribution. -/
def expectedNumInternalNodes (μ : PMF Doc) (treeOf : Doc → BinTree α) : ℝ :=
  Exp μ (fun d => (numInternalNodes (treeOf d) : ℝ))

/-- Expected internal nodes equal expected leaves minus one.

The only non-structural assumption is a summability hypothesis enabling `tsum` linearity. -/
lemma expectedNumInternalNodes_eq_expectedNumLeaves_minus_one
    (μ : PMF Doc) (treeOf : Doc → BinTree α)
    (h_summable : Summable (fun d => (μ d).toReal * (numLeaves (treeOf d) : ℝ))) :
    expectedNumInternalNodes μ treeOf = expectedNumLeaves μ treeOf - 1 := by
  have hsum_const : Summable (fun d => (μ d).toReal * (-1 : ℝ)) := by
    apply PMF.summable_coe_real_mul_of_bounded μ (fun _ => (-1 : ℝ)) 1 (by norm_num)
    intro _; simp

  unfold expectedNumInternalNodes expectedNumLeaves Exp
  -- Rewrite `numInternalNodes` pointwise as `numLeaves - 1`.
  have hrewrite :
      (fun d => (μ d).toReal * (numInternalNodes (treeOf d) : ℝ)) =
        fun d => (μ d).toReal * ((numLeaves (treeOf d) : ℝ) - 1) := by
    funext d
    simp [numInternalNodes_real_eq_numLeaves_real_sub_one]
  simp [hrewrite]

  -- Expand subtraction inside the `tsum` and split.
  calc
    (∑' d, (μ d).toReal * ((numLeaves (treeOf d) : ℝ) - 1))
        = ∑' d, ((μ d).toReal * (numLeaves (treeOf d) : ℝ) + (μ d).toReal * (-1 : ℝ)) := by
            refine tsum_congr ?_
            intro d
            ring
    _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) +
          ∑' d, (μ d).toReal * (-1 : ℝ) := by
          simpa using (Summable.tsum_add h_summable hsum_const)
    _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + (-1) := by
          have hmul_to_neg :
              (∑' d, (μ d).toReal * (-1 : ℝ)) = ∑' d, -(μ d).toReal := by
            refine tsum_congr ?_
            intro d
            ring
          have htsum_neg : (∑' d, -(μ d).toReal) = (-1 : ℝ) := by
            -- `∑ μ = 1` for PMFs.
            simpa [PMF.toReal_tsum_coe μ] using
              (tsum_neg (L := SummationFilter.unconditional Doc) (f := fun d : Doc => (μ d).toReal))
          calc
            (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + (∑' d, (μ d).toReal * (-1 : ℝ))
                = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + ∑' d, -(μ d).toReal := by
                    rw [hmul_to_neg]
            _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + (-1) := by
                    rw [htsum_neg]
    _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) - 1 := by
          ring

end CorpusExpectations

end

end
