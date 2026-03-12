import Mathlib

/-!
# FormalProofs/OPT/CutBudgetGuidance.lean

## Cut-budgeted boundary selection: approaching the oracle optimum

This file formalizes a deterministic guarantee behind the cut-budgeted changepoint simulations.

### High-level statement

Fix a maximum cut budget `K`. Let `r : α → ℝ` be the true per-position reward, and `r̂ : α → ℝ`
an estimated reward. If the estimation error is uniformly bounded:

`∀ t, |r t - r̂ t| ≤ ε` with `ε ≥ 0`,

and `Ĉ` maximizes `CutObj r̂` among cut sets of size `≤ K`, then the **oracle gap**
relative to any reference cut set `C⋆` with `|C⋆| ≤ K` satisfies:

`CutObj r C⋆ - CutObj r Ĉ ≤ 2Kε`.

In particular, taking `C⋆` to be the **oracle optimal** cut set for `r` yields:
`CutObj r C⋆ - CutObj r Ĉ ≤ 2Kε`, so as guidance drives `ε → 0`, the gap vanishes.

### Connection to Hamming loss for `{+1,-1}` boundary rewards

For a true boundary set `B : Finset α`, define the reward `TrueCutReward B` to be `+1` on `B`
and `-1` off `B`. Then the cut objective is exactly a constant minus the Hamming loss:

`CutObj (TrueCutReward B) C = (B.card : ℝ) - (HammingLoss B C : ℝ)`.

So the same `2Kε` bound translates directly into a Hamming-loss optimality gap.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

section CutObjective

variable {α : Type*}

/-- Objective for a cut set: sum of per-position rewards. -/
def CutObj (r : α → ℝ) (C : Finset α) : ℝ :=
  ∑ t ∈ C, r t

lemma cutObj_sub_eq_sum_sub (r rhat : α → ℝ) (C : Finset α) :
    CutObj r C - CutObj rhat C = ∑ t ∈ C, (r t - rhat t) := by
  classical
  simp [CutObj, Finset.sum_sub_distrib]

lemma abs_cutObj_sub_le_sum_abs (r rhat : α → ℝ) (C : Finset α) :
    |CutObj r C - CutObj rhat C| ≤ ∑ t ∈ C, |r t - rhat t| := by
  classical
  calc
    |CutObj r C - CutObj rhat C|
        = |∑ t ∈ C, (r t - rhat t)| := by
            simp [cutObj_sub_eq_sum_sub]
    _ ≤ ∑ t ∈ C, |r t - rhat t| := by
            simpa using
              (Finset.abs_sum_le_sum_abs
                (s := C)
                (f := fun t => r t - rhat t))

lemma abs_cutObj_sub_le_card_mul (r rhat : α → ℝ) (C : Finset α) (ε : ℝ)
    (hε : ∀ t, |r t - rhat t| ≤ ε) :
    |CutObj r C - CutObj rhat C| ≤ (C.card : ℝ) * ε := by
  classical
  calc
    |CutObj r C - CutObj rhat C|
        = |∑ t ∈ C, (r t - rhat t)| := by
            simp [cutObj_sub_eq_sum_sub]
    _ ≤ ∑ t ∈ C, |r t - rhat t| := by
            simpa using
              (Finset.abs_sum_le_sum_abs
                (s := C)
                (f := fun t => r t - rhat t))
    _ ≤ ∑ t ∈ C, ε := by
            refine Finset.sum_le_sum ?_
            intro t ht
            exact hε t
    _ = (C.card : ℝ) * ε := by
            simp

/-- **Oracle gap bound (sum-absolute-error form).**

This is the same argument as `oracle_gap_le_of_uniform_reward_error`, but without a uniform
sup-norm bound. Instead, the gap is controlled by the total absolute reward errors on the two
cut sets. This matches the simulation-side bound:

`Σ_{t∈C⋆} |r̂_t - r_t| + Σ_{t∈Ĉ} |r̂_t - r_t|`.
-/
theorem oracle_gap_le_sum_abs_reward_errors
    (r rhat : α → ℝ)
    {Cstar Chat : Finset α}
    (hChat_opt_at_Cstar : CutObj rhat Cstar ≤ CutObj rhat Chat) :
    CutObj r Cstar - CutObj r Chat ≤
      (∑ t ∈ Cstar, |r t - rhat t|) + (∑ t ∈ Chat, |r t - rhat t|) := by
  classical
  have hmid : CutObj rhat Cstar - CutObj rhat Chat ≤ 0 := sub_nonpos.mpr hChat_opt_at_Cstar

  have h1 : CutObj r Cstar - CutObj rhat Cstar ≤ ∑ t ∈ Cstar, |r t - rhat t| := by
    have habs := abs_cutObj_sub_le_sum_abs (r := r) (rhat := rhat) (C := Cstar)
    exact le_trans (le_abs_self _) habs

  have h3 : CutObj rhat Chat - CutObj r Chat ≤ ∑ t ∈ Chat, |r t - rhat t| := by
    have habs' := abs_cutObj_sub_le_sum_abs (r := rhat) (rhat := r) (C := Chat)
    have habs : |CutObj rhat Chat - CutObj r Chat| ≤ ∑ t ∈ Chat, |r t - rhat t| := by
      simpa [abs_sub_comm] using habs'
    exact le_trans (le_abs_self _) habs

  have hdecomp :
      CutObj r Cstar - CutObj r Chat =
        (CutObj r Cstar - CutObj rhat Cstar) +
        (CutObj rhat Cstar - CutObj rhat Chat) +
        (CutObj rhat Chat - CutObj r Chat) := by
    ring

  rw [hdecomp]
  linarith

/-- **Oracle gap bound** for selecting the best cut set under an estimated reward.

If `r̂` is uniformly close to `r` (in sup norm) and both the oracle reference set `C⋆`
and the estimated optimizer `Ĉ` respect the budget `K`, then:

`CutObj r C⋆ - CutObj r Ĉ ≤ 2Kε`.
-/
theorem oracle_gap_le_of_uniform_reward_error
    (r rhat : α → ℝ)
    (K : ℕ)
    (ε : ℝ)
    (hε_nonneg : 0 ≤ ε)
    (hε : ∀ t, |r t - rhat t| ≤ ε)
    {Cstar Chat : Finset α}
    (hCstar : Cstar.card ≤ K)
    (hChat : Chat.card ≤ K)
    (hChat_opt : ∀ C : Finset α, C.card ≤ K → CutObj rhat C ≤ CutObj rhat Chat) :
    CutObj r Cstar - CutObj r Chat ≤ (2:ℝ) * (K:ℝ) * ε := by
  classical
  -- Approximation error on Cstar.
  have h1 : CutObj r Cstar - CutObj rhat Cstar ≤ (Cstar.card : ℝ) * ε := by
    have habs := abs_cutObj_sub_le_card_mul (r := r) (rhat := rhat) (C := Cstar) (ε := ε) hε
    exact le_trans (le_abs_self _) habs

  -- Approximation error on Chat (swap roles using `abs_sub_comm`).
  have hε' : ∀ t, |rhat t - r t| ≤ ε := by
    intro t
    simpa [abs_sub_comm] using (hε t)

  have h3 : CutObj rhat Chat - CutObj r Chat ≤ (Chat.card : ℝ) * ε := by
    have habs := abs_cutObj_sub_le_card_mul (r := rhat) (rhat := r) (C := Chat) (ε := ε) hε'
    exact le_trans (le_abs_self _) habs

  -- Estimated optimality: Chat maximizes rhat under the budget.
  have h2 : CutObj rhat Cstar ≤ CutObj rhat Chat := by
    simpa using (hChat_opt Cstar hCstar)

  have hmid : CutObj rhat Cstar - CutObj rhat Chat ≤ 0 := by
    exact sub_nonpos.mpr h2

  -- Decompose the true gap into three pieces.
  have hdecomp :
      CutObj r Cstar - CutObj r Chat =
        (CutObj r Cstar - CutObj rhat Cstar) +
        (CutObj rhat Cstar - CutObj rhat Chat) +
        (CutObj rhat Chat - CutObj r Chat) := by
    ring

  -- First bound using the three inequalities.
  have hgap1 : CutObj r Cstar - CutObj r Chat ≤ (Cstar.card : ℝ) * ε + 0 + (Chat.card : ℝ) * ε := by
    rw [hdecomp]
    linarith

  -- Now use card ≤ K.
  have hCstarK : (Cstar.card : ℝ) ≤ (K : ℝ) := by
    exact_mod_cast hCstar
  have hChatK : (Chat.card : ℝ) ≤ (K : ℝ) := by
    exact_mod_cast hChat

  calc
    CutObj r Cstar - CutObj r Chat
        ≤ (Cstar.card : ℝ) * ε + 0 + (Chat.card : ℝ) * ε := hgap1
    _ ≤ (K : ℝ) * ε + 0 + (K : ℝ) * ε := by
          have hCstar_mul : (Cstar.card : ℝ) * ε ≤ (K : ℝ) * ε := by
            exact mul_le_mul_of_nonneg_right hCstarK hε_nonneg
          have hChat_mul : (Chat.card : ℝ) * ε ≤ (K : ℝ) * ε := by
            exact mul_le_mul_of_nonneg_right hChatK hε_nonneg
          linarith
    _ = (2:ℝ) * (K:ℝ) * ε := by
          ring

end CutObjective

section BoundaryReward

variable {α : Type*} [DecidableEq α]

/-- The `{+1,-1}` reward for boundary detection: `+1` on true boundaries `B`, `-1` elsewhere. -/
def TrueCutReward (B : Finset α) : α → ℝ :=
  fun t => if t ∈ B then (1:ℝ) else (-1:ℝ)

/-- Hamming loss between a predicted cut set `C` and the true boundary set `B`. -/
def HammingLoss (B C : Finset α) : ℕ :=
  (C \ B).card + (B \ C).card

lemma cutObj_trueCutReward_eq_card_inter_sub_card_sdiff (B C : Finset α) :
    CutObj (TrueCutReward B) C = ((C ∩ B).card : ℝ) - ((C \ B).card : ℝ) := by
  classical
  simp [CutObj, TrueCutReward]
  have hdisj : Disjoint (C \ B) (C ∩ B) := Finset.disjoint_sdiff_inter C B
  calc
    (∑ x ∈ C, if x ∈ B then (1:ℝ) else (-1:ℝ))
        = ∑ x ∈ (C \ B ∪ C ∩ B), if x ∈ B then (1:ℝ) else (-1:ℝ) := by
            simp [Finset.sdiff_union_inter]
    _ = (∑ x ∈ C \ B, if x ∈ B then (1:ℝ) else (-1:ℝ)) +
        ∑ x ∈ C ∩ B, if x ∈ B then (1:ℝ) else (-1:ℝ) := by
            simpa [Finset.sum_union, hdisj, add_comm, add_left_comm, add_assoc]
    _ = ((C ∩ B).card : ℝ) - ((C \ B).card : ℝ) := by
            have h_sdiff :
                (∑ x ∈ C \ B, if x ∈ B then (1:ℝ) else (-1:ℝ)) = -((C \ B).card : ℝ) := by
              calc
                (∑ x ∈ C \ B, if x ∈ B then (1:ℝ) else (-1:ℝ))
                    = ∑ x ∈ C \ B, (-1:ℝ) := by
                        refine Finset.sum_congr rfl ?_
                        intro x hx
                        have hxB : x ∉ B := (Finset.mem_sdiff.mp hx).2
                        simp [hxB]
                _ = -((C \ B).card : ℝ) := by
                        simp
            have h_inter :
                (∑ x ∈ C ∩ B, if x ∈ B then (1:ℝ) else (-1:ℝ)) = ((C ∩ B).card : ℝ) := by
              calc
                (∑ x ∈ C ∩ B, if x ∈ B then (1:ℝ) else (-1:ℝ))
                    = ∑ x ∈ C ∩ B, (1:ℝ) := by
                        refine Finset.sum_congr rfl ?_
                        intro x hx
                        have hxB : x ∈ B := (Finset.mem_inter.mp hx).2
                        simp [hxB]
                _ = ((C ∩ B).card : ℝ) := by
                        simp
            simp [h_sdiff, h_inter, sub_eq_add_neg, add_comm, add_left_comm, add_assoc]

lemma card_eq_inter_add_sdiff (s t : Finset α) :
    s.card = (s ∩ t).card + (s \ t).card := by
  classical
  have h : s \ t ∪ s ∩ t = s := Finset.sdiff_union_inter s t
  have hdisj : Disjoint (s \ t) (s ∩ t) := Finset.disjoint_sdiff_inter s t
  have := congrArg Finset.card h
  simpa [Finset.card_union_of_disjoint hdisj, add_comm, add_left_comm, add_assoc] using this

lemma card_sub_sdiff_eq_inter (s t : Finset α) :
    (s.card : ℝ) - (s \ t).card = (s ∩ t).card := by
  classical
  have hnat : s.card = (s ∩ t).card + (s \ t).card := card_eq_inter_add_sdiff (s := s) (t := t)
  have hreal : (s.card : ℝ) = (s ∩ t).card + (s \ t).card := by
    exact_mod_cast hnat
  linarith

lemma cardB_sub_hamming_eq_inter_sub_sdiff (B C : Finset α) :
    (B.card : ℝ) - (HammingLoss B C : ℝ) = (C ∩ B).card - (C \ B).card := by
  classical
  have : (HammingLoss B C : ℝ) = (C \ B).card + (B \ C).card := by
    simp [HammingLoss, Nat.cast_add]
  calc
    (B.card : ℝ) - (HammingLoss B C : ℝ)
        = (B.card : ℝ) - ((C \ B).card + (B \ C).card) := by
            simpa [this]
    _ = ((B.card : ℝ) - (B \ C).card) - (C \ B).card := by
            ring
    _ = (B ∩ C).card - (C \ B).card := by
            simpa [card_sub_sdiff_eq_inter (s := B) (t := C)]
    _ = (C ∩ B).card - (C \ B).card := by
            simp [Finset.inter_comm]

lemma cutObj_trueCutReward_eq_cardB_sub_hamming (B C : Finset α) :
    CutObj (TrueCutReward B) C = (B.card : ℝ) - (HammingLoss B C : ℝ) := by
  classical
  calc
    CutObj (TrueCutReward B) C
        = (C ∩ B).card - (C \ B).card := cutObj_trueCutReward_eq_card_inter_sub_card_sdiff (B := B) (C := C)
    _ = (B.card : ℝ) - (HammingLoss B C : ℝ) := by
          simpa using (cardB_sub_hamming_eq_inter_sub_sdiff (B := B) (C := C)).symm

/-- Hamming-loss gap bound (sum-absolute-error form), matching the simulation-side quantity

`Σ_{t∈C⋆} |r̂_t - r_t| + Σ_{t∈Ĉ} |r̂_t - r_t|`.
-/
theorem hamming_gap_le_sum_abs_reward_errors
    (B : Finset α)
    (rhat : α → ℝ)
    {Cstar Chat : Finset α}
    (hChat_opt_at_Cstar : CutObj rhat Cstar ≤ CutObj rhat Chat) :
    (HammingLoss B Chat : ℝ) - (HammingLoss B Cstar : ℝ) ≤
      (∑ t ∈ Cstar, |TrueCutReward B t - rhat t|) + (∑ t ∈ Chat, |TrueCutReward B t - rhat t|) := by
  classical
  have hobj :=
    oracle_gap_le_sum_abs_reward_errors
      (r := TrueCutReward B) (rhat := rhat)
      (Cstar := Cstar) (Chat := Chat)
      hChat_opt_at_Cstar

  have hrewrite :
      CutObj (TrueCutReward B) Cstar - CutObj (TrueCutReward B) Chat =
        (HammingLoss B Chat : ℝ) - (HammingLoss B Cstar : ℝ) := by
    simp [cutObj_trueCutReward_eq_cardB_sub_hamming, sub_eq_add_neg, add_comm, add_left_comm, add_assoc]

  simpa [hrewrite] using hobj

/-- Hamming-loss gap bound obtained by combining `{+1,-1}` rewards with `oracle_gap_le_of_uniform_reward_error`. -/
theorem hamming_gap_le_of_uniform_reward_error
    (B : Finset α)
    (rhat : α → ℝ)
    (K : ℕ)
    (ε : ℝ)
    (hε_nonneg : 0 ≤ ε)
    (hε : ∀ t, |TrueCutReward B t - rhat t| ≤ ε)
    {Cstar Chat : Finset α}
    (hCstar : Cstar.card ≤ K)
    (hChat : Chat.card ≤ K)
    (hChat_opt : ∀ C : Finset α, C.card ≤ K → CutObj rhat C ≤ CutObj rhat Chat) :
    (HammingLoss B Chat : ℝ) - (HammingLoss B Cstar : ℝ) ≤ (2:ℝ) * (K:ℝ) * ε := by
  classical
  have hobj :=
    oracle_gap_le_of_uniform_reward_error
      (r := TrueCutReward B) (rhat := rhat)
      (K := K) (ε := ε)
      hε_nonneg hε
      (Cstar := Cstar) (Chat := Chat)
      hCstar hChat hChat_opt

  -- Rewrite the objective gap as a Hamming-loss gap.
  have hrewrite :
      CutObj (TrueCutReward B) Cstar - CutObj (TrueCutReward B) Chat =
        (HammingLoss B Chat : ℝ) - (HammingLoss B Cstar : ℝ) := by
    -- Use `CutObj = const - Hamming` on both sides.
    simp [cutObj_trueCutReward_eq_cardB_sub_hamming, sub_eq_add_neg, add_comm, add_left_comm, add_assoc]

  -- Transfer the bound.
  simpa [hrewrite] using hobj

end BoundaryReward
