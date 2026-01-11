import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.TrainingPipeline

/-!
# Main Theorems: Local-to-Global Oracle Preference Learning

This file collects and documents the main results of the formalization. These theorems
establish that **local consistency conditions imply global training equivalence** for
a broad class of preference learning methods including DPO, GRPO, and PPO-style RL.

## The Core Insight

The central contribution is NOT the distillation/gap-composition results (which follow
from triangle inequality). Rather, it is the **local-to-global mechanism**:

1. **Local Laws (L1, L2, L3)**: Testable conditions on a summarizer `g`
2. **Zero Distortion**: L1 + L2 + L3 imply `E[dist(f*(Z), f*(x))] = 0` after R rounds
3. **Oracle-Measurability**: When loss/generator depend on x only through f*(x)
4. **Training Equivalence**: Optimal policies on summarized data = optimal on originals

This means we can **test locally** (audit the summarizer) and **conclude globally**
(training on summaries is as good as training on full documents).

## Theorem Hierarchy

```
                    Local Laws (L1, L2, L3)
                           │
                           ▼
              multi_round_proper (ExpectationTheory)
              E[D(Z^R, x)] = 0 for all R ≥ 1
                           │
                           ▼
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
  preference_learning_equivalence     grpo_rl_equivalence
  (PreferenceLearning.lean)           (PreferenceLearning.lean)
         │                                   │
         ▼                                   ▼
  dpo_gap_zero_of_local_laws         DPO/GRPO training sound
  (DPO.lean)                         on summarized data
```

## Coverage of Modern Methods

The formalization captures:

| Method | File | Key Theorem |
|--------|------|-------------|
| DPO | DPO.lean | `dpo_equivalence` |
| Plackett-Luce GRPO | PreferenceLearning.lean | `grpo_equivalence` |
| GRPO-RL (DeepSeek-R1) | PreferenceLearning.lean | `grpo_rl_equivalence` |
| General pairwise | PreferenceLearning.lean | `preference_learning_equivalence` |
| General group-wise | PreferenceLearning.lean | `expected_group_loss_eq_of_zero_dist` |

-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

noncomputable section

namespace MainTheorems

/-!
## Theorem 1: Multi-Round Preservation

**Statement**: If local laws L1, L2, L3 hold for summarizer g on tree T, then
after R rounds of summarization, the expected oracle distortion is exactly zero.

**Significance**: This is the foundational result. It shows that local testable
conditions (which an auditor can check on individual summarizer calls) guarantee
global preservation of oracle information through arbitrary reduction depth.

**Paper Reference**: Theorem 5.1 (Multi-Round Preservation)
-/

/-- Multi-round preservation: local laws imply zero expected distortion.

This is the core theorem enabling local-to-global inference. When a summarizer
satisfies L1 (leaf sufficiency), L2 (merge consistency), and L3 (idempotence),
the expected distortion after R rounds of tree reduction is exactly 0.

Mathematical statement:
  L1 ∧ L2 ∧ L3 ∧ R ≥ 1 ⟹ E_{z~ZR(g,x,R,T)}[dist(f*(z), f*(x))] = 0

Proof: Induction on R, using L1+L2 for base case and L3 for inductive step. -/
abbrev multi_round_preservation := @multi_round_proper

/-!
## Theorem 2: Preference Learning Equivalence

**Statement**: When a preference loss is oracle-measurable and the pair/group
generator is oracle-indexed, zero distortion implies equal expected loss.

**Significance**: This abstracts over ALL preference learning methods. Any method
where the loss depends on documents only through oracle values inherits the
equivalence property.
-/

/-- General preference learning equivalence under zero distortion.

When summaries preserve oracle values (dist(f*(z), f*(x)) = 0 for all z in
summary support), any oracle-measurable preference learning method achieves
identical expected loss on summaries vs. originals.

This theorem abstracts over:
- DPO (pairwise, Bradley-Terry)
- GRPO (k-wise, Plackett-Luce)
- GRPO-RL (clipped surrogate + KL)
- Any future method satisfying oracle-measurability -/
abbrev preference_learning_equiv := @preference_learning_equivalence

/-!
## Theorem 3: DPO Training Soundness

**Statement**: When local laws hold, DPO training on summarized data produces
the same optimal policy as training on original data.

**Significance**: Concrete instantiation for the widely-used DPO method.
-/

/-- DPO equivalence: local laws imply identical training outcomes.

The gap between DPO loss on original data and DPO loss on summarized data
is exactly zero when L1, L2, L3 hold.

Corollary: argmin_{π measurable} L_DPO(π; X) = argmin_{π measurable} L_DPO(π; Z^R) -/
abbrev dpo_training_sound := @dpo_equivalence

/-!
## Theorem 4: GRPO-RL Equivalence (DeepSeek-R1 Style)

**Statement**: The GRPO-RL objective (clipped surrogate + KL penalty) is
equivalent on original vs. summarized data when oracle-measurability holds.

**Significance**: Captures the exact objective used by DeepSeek-R1:
  J_GRPO(θ) = E[1/G Σ min(r_i·A_i, clip(r_i)·A_i) - β·D_KL(π_θ || π_ref)]
where A_i = (reward_i - mean) / std (z-score normalized advantage).
-/

/-- GRPO-RL equivalence: DeepSeek-R1 style training is sound on summaries.

The GRPO-RL loss includes:
- Group sampling (k candidates per prompt)
- Z-score normalized advantages: A_i = (r_i - mean) / std
- PPO-style clipping: min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
- KL penalty: β · D_KL(π_θ || π_ref)

When policies, rewards, and group generators are oracle-measurable,
training on summaries equals training on originals. -/
abbrev grpo_rl_training_sound := @grpo_rl_equivalence

/-!
## Theorem 5: Listwise GRPO Equivalence (Plackett-Luce)

**Statement**: GRPO with Plackett-Luce ranking loss (k > 2 group comparisons)
is equivalent on original vs. summarized data.

**Significance**: Generalizes DPO (k=2, Bradley-Terry) to arbitrary k.
-/

/-- Plackett-Luce GRPO equivalence: listwise ranking is sound on summaries.

The Plackett-Luce model generalizes Bradley-Terry from pairs to rankings:
  P(ranking) = ∏_{i=1}^{k} exp(s_i) / Σ_{j≥i} exp(s_j)

When policy and ranker are oracle-measurable, GRPO training on
summarized data equals training on original data. -/
abbrev grpo_plackett_luce_sound := @grpo_equivalence

/-!
## Corollary: Gap Composition

The distillation/two-stage results in TrainingPipeline.lean are corollaries
that compose the above theorems via triangle inequality.

These are "trivial" in the sense that they follow from standard analysis once
the deep theorems above are established.
-/

/-- Two-stage gap bound: gaps compose additively.

For a two-stage pipeline (Oracle → Teacher → Student):
  |L_S(orig) - L_L(orig)| ≤ 2·ε_stage1 + ε_stage2

When local laws hold exactly, ε_stage1 = 0, giving pure distillation gap. -/
abbrev gap_composition := @training_path_gap_bound

/-!
## Unified Framework: The Common Mathematical Core

All preference learning gap bounds (DPO, GRPO-PL, GRPO-RL, and future methods)
follow from a **single unified template**:

```
Gap ≤ Lipschitz_Constant × Expected_Distortion
```

**Theorem (Unified Preference Gap):** For any expected loss E[L] over a
distribution μ where the inner expectation E_gen is L-Lipschitz in oracle distance,

  |E_X[E_gen] - E_Z[E_gen]| ≤ L × Δ_R

where Δ_R = E_{x,z}[dist(f*(x), f*(z))] is the expected distortion.

**Proof Structure (Method-Agnostic):**
1. `coupling_expansion_bounded` rewrites E_X - E_Z as double sum over product measure
2. Pointwise Lipschitz bound controls each term: |E_gen(x) - E_gen(z)| ≤ L⋅dist(...)
3. `coupling_bound_ineq_bounded` + Fubini gives the final bound

**Method-Specific Instantiations:**

| Method | Lipschitz Constant | E_gen Structure |
|--------|-------------------|-----------------|
| DPO | L = 2\|β\|L_pol | Expected -log σ over pairs |
| GRPO-PL | L = L_grpo | Expected Plackett-Luce over k-groups |
| GRPO-RL | L = L_grpo_rl | Expected clipped advantage + KL over groups |

The unified theorem captures the shared mathematical structure, while the
specific instantiations demonstrate how different loss functions plug into
the framework.
-/

/-- Unified preference gap theorem.

This is the mathematical core shared by all preference learning methods.
Any expected loss with a Lipschitz inner expectation satisfies the standard
gap bound: Gap ≤ Lipschitz × Distortion.

Instantiations:
- DPO: L = 2|β|L_pol
- GRPO-PL: L = L_grpo
- GRPO-RL: L = L_grpo_rl -/
abbrev unified_gap := @unified_preference_gap_bounded

end MainTheorems

end
