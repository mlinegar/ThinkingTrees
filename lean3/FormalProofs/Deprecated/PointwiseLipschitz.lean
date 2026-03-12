import FormalProofs.OPT.PreferenceLearning

/-!
# FormalProofs/Deprecated/PointwiseLipschitz.lean

## Deprecated Pointwise Lipschitz Lemmas

**Status:** These lemmas are DEPRECATED and NOT USED by the main theorems.

**Why deprecated:** The pointwise Lipschitz bounds for GRPO-PL and GRPO-RL are
**not provable** for the `dist > 0` case because rankings are discontinuous at ties.
When `dist(fstar x, fstar z) > 0`, the ranker may give different ranks for x and z,
causing the loss to jump discontinuously.

**What replaced them:** The main theorems now use explicit axioms for **expected**
Lipschitz bounds:
- `ExpectedGRPOLossLipschitz` for GRPO-PL
- `ExpectedGRPORLLossLipschitz` for GRPO-RL

These axioms are justified by the **Random Utility Model** assumption: under
continuous underlying utilities with noise (e.g., Gumbel → Plackett-Luce),
ties have measure zero, so the expected loss is continuous.

**Why kept:** These lemmas document the mathematical structure and prove the
`dist = 0` case correctly. They may be useful for understanding the theory
or for future work if stronger assumptions are adopted.

See `PreferenceBounds.lean` Section "Random Utility Model Foundation" for details.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise
open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

set_option maxHeartbeats 400000

noncomputable section

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {A : Type*}

/-!
## GRPO-PL Pointwise Lipschitz (DEPRECATED)
-/

/-- Plackett-Luce loss bound with different ranks (DEPRECATED - NOT PROVABLE).

**Status:** This lemma is NOT PROVABLE for the `dist > 0` case because rankings
are discontinuous at ties. It is preserved for documentation but the main theorems
now use `ExpectedGRPOLossLipschitz` instead, which works in expectation over the
Random Utility Model noise.

**Why unprovable:** When dist(fstar x, fstar z) > 0, the ranker may give different
ranks for x and z. Since rankings are discrete, the loss can jump discontinuously
even for arbitrarily small oracle distances. This makes pointwise Lipschitz impossible.

**Why expected version works:** Under the Random Utility Model assumption, rankings
arise from continuous utilities plus noise. Ties (where rankings change) have
measure zero, so the expected loss is continuous by dominated convergence.

The `dist = 0` case is fully proved; the `dist > 0` case is intentionally excluded. -/
lemma PlackettLuceLoss_lipschitz_general {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) (L : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (x z : Strings) (group : Fin k → A)
    (h_dist : dist (fstar x) (fstar z) = 0) :
    |GRPOLossPointwise pol x group (ranker x group) -
     GRPOLossPointwise pol z group (ranker z group)| ≤ L * dist (fstar x) (fstar z) := by
  -- dist = 0 → ranker and policy values are equal → loss difference is 0
  have h_ranker_eq : ranker x = ranker z := h_ranker x z h_dist
  have h_pol_eq : ∀ a, pol x a = pol z a := by
    intro a
    have := h_pol_lip x z a
    simp only [h_dist, mul_zero] at this
    exact eq_of_abs_sub_nonpos this
  -- Losses are equal
  have h_loss_eq : GRPOLossPointwise pol x group (ranker x group) =
                   GRPOLossPointwise pol z group (ranker z group) := by
    unfold GRPOLossPointwise PlackettLuceLogProb
    simp only [h_ranker_eq, h_pol_eq]
  simp only [h_loss_eq, sub_self, abs_zero, h_dist, mul_zero, le_refl]

/-- GRPO-PL loss Lipschitz bound (DEPRECATED).

Delegates to `PlackettLuceLoss_lipschitz_general`, which is only stated for `dist = 0`.
Use `ExpectedGRPOLossLipschitz` instead for the expected version. -/
lemma grpo_pl_loss_lipschitz_bound {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y)
    (L_grpo : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (x z : Strings) (group : Fin k → A)
    (h_dist : dist (fstar x) (fstar z) = 0) :
    |GRPOLossPointwise pol x group (ranker x group) -
     GRPOLossPointwise pol z group (ranker z group)| ≤
    L_grpo * dist (fstar x) (fstar z) :=
  PlackettLuceLoss_lipschitz_general pol ranker fstar L_grpo h_pol_lip h_ranker x z group h_dist

/-!
## GRPO-RL Pointwise Lipschitz (DEPRECATED)
-/

/-- GRPO-RL loss is Lipschitz when policies and rewards are Lipschitz (DEPRECATED - NOT PROVABLE).

**Status:** Similar to `PlackettLuceLoss_lipschitz_general`, this lemma is only stated
for the `dist = 0` case. The main theorems now use `ExpectedGRPORLLossLipschitz` instead.

**Mathematical content (for reference):** The GRPO-RL loss composes several Lipschitz components:
1. **Policy ratios**: pol(x,a)/pol_old(x,a) - ratio of Lipschitz functions
2. **Z-score advantages**: (r - mean)/std - involves reward Lipschitz bound
3. **Clipping**: max(1-ε, min(1+ε, r)) - 1-Lipschitz
4. **KL penalty**: log-ratio of policies - composition of Lipschitz functions

The `dist = 0` case is fully proved; the `dist > 0` case is intentionally excluded. -/
lemma GRPORLLoss_lipschitz_general {Strings A Y : Type*} [PseudoMetricSpace Y] (k : ℕ)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ) (fstar : Strings → Y)
    (L : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (x z : Strings) (group : Fin k → A)
    (h_dist : dist (fstar x) (fstar z) = 0) :
    |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
     GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| ≤
    L * dist (fstar x) (fstar z) := by
  -- dist = 0 → all component functions return equal values → loss equal
  have h_pol_eq : ∀ a, pol x a = pol z a := by
    intro a
    have := h_pol_lip x z a
    simp only [h_dist, mul_zero] at this
    exact eq_of_abs_sub_nonpos this
  have h_old_eq : ∀ a, pol_old x a = pol_old z a := by
    intro a
    have := h_old_lip x z a
    simp only [h_dist, mul_zero] at this
    exact eq_of_abs_sub_nonpos this
  have h_ref_eq : ∀ a, pol_ref x a = pol_ref z a := by
    intro a
    have := h_ref_lip x z a
    simp only [h_dist, mul_zero] at this
    exact eq_of_abs_sub_nonpos this
  have h_reward_eq : ∀ a, reward x a = reward z a := by
    intro a
    have := h_reward_lip x z a
    simp only [h_dist, mul_zero] at this
    exact eq_of_abs_sub_nonpos this
  -- All components equal → loss equal
  have h_loss_eq : GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group =
                   GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group := by
    simp only [GRPORLLossPointwise, GRPOGroupKL, GRPOKLEstimator, GRPOAdvantage,
               GRPOGroupMean, GRPOGroupStd, GRPOClip]
    simp only [h_pol_eq, h_old_eq, h_ref_eq, h_reward_eq]
  simp only [h_loss_eq, sub_self, abs_zero, h_dist, mul_zero, le_refl]

/-- GRPO-RL loss Lipschitz bound (DEPRECATED).

Delegates to `GRPORLLoss_lipschitz_general`, which is only stated for `dist = 0`.
Use `ExpectedGRPORLLossLipschitz` instead for the expected version. -/
lemma grpo_rl_loss_lipschitz_bound (k : ℕ)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ) (fstar : Strings → Y)
    (L_grpo_rl : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (x z : Strings) (group : Fin k → A)
    (h_dist : dist (fstar x) (fstar z) = 0) :
    |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
     GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| ≤
    L_grpo_rl * dist (fstar x) (fstar z) :=
  GRPORLLoss_lipschitz_general k pol pol_old pol_ref reward eps beta fstar L_grpo_rl
    h_pol_lip h_old_lip h_ref_lip h_reward_lip x z group h_dist

end
