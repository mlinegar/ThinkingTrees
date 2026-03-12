import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic

/-!
# FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean

## Why leaves matter only once the target is nonlinear in local mixtures

This file formalizes the key distinction behind the new tree-relevant LDA simulation ladder.

- In Stage 1, the target is a linear function of the bag-of-words histogram, so exact additive
  mergeability means leaves carry no extra statistical information.
- In Stage 2, the target is a sum of **local** nonlinear utilities

  `h(π) = θᵀ π + λ πᵀ W π`

  over latent base leaves. The linear part still averages exactly, but the quadratic part does not
  collapse to the utility of the pooled mean mixture in general.

The theorem below isolates that structure precisely: the pooled-vs-leaf gap is *exactly* the
quadratic gap, scaled by `λ`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section LeafLocalMixture

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Weighted mean of local topic mixtures across leaves. -/
def weightedMean (ω : ι → ℝ) (π : ι → κ → ℝ) : κ → ℝ :=
  fun k => ∑ b : ι, ω b * π b k

/-- Linear utility on a topic mixture. -/
def linearUtility (θ : κ → ℝ) (π : κ → ℝ) : ℝ :=
  ∑ k : κ, θ k * π k

/-- Quadratic utility on a topic mixture, written as a bilinear form. -/
def quadraticUtility (W : κ → κ → ℝ) (π : κ → ℝ) : ℝ :=
  ∑ i : κ, ∑ j : κ, π i * W i j * π j

/-- Affine-plus-quadratic utility used by the local-mixture simulations. -/
def affineQuadraticUtility (θ : κ → ℝ) (W : κ → κ → ℝ) (lam : ℝ) (π : κ → ℝ) : ℝ :=
  linearUtility θ π + lam * quadraticUtility W π

/-- The linear part commutes exactly with leaf averaging. -/
theorem linearUtility_weightedMean_eq_sum (ω : ι → ℝ) (π : ι → κ → ℝ) (θ : κ → ℝ) :
    linearUtility θ (weightedMean ω π)
      = ∑ b : ι, ω b * linearUtility θ (π b) := by
  classical
  simp [linearUtility, weightedMean, Finset.mul_sum, Finset.sum_mul, mul_assoc, mul_left_comm, mul_comm]
  rw [Finset.sum_comm]

/-- The pooled-vs-leaf gap is carried entirely by the quadratic part of the utility. -/
theorem affineQuadratic_gap_eq_quadratic_gap
    (ω : ι → ℝ) (π : ι → κ → ℝ) (θ : κ → ℝ) (W : κ → κ → ℝ) (lam : ℝ) :
    (∑ b : ι, ω b * affineQuadraticUtility θ W lam (π b))
      - affineQuadraticUtility θ W lam (weightedMean ω π)
      =
      lam *
        ((∑ b : ι, ω b * quadraticUtility W (π b))
          - quadraticUtility W (weightedMean ω π)) := by
  classical
  simp [
    affineQuadraticUtility,
    linearUtility_weightedMean_eq_sum (ω := ω) (π := π) (θ := θ),
    Finset.sum_add_distrib,
    Finset.mul_sum,
    sub_eq_add_neg,
    mul_add,
    add_comm,
    add_left_comm,
    add_assoc,
    left_distrib,
    right_distrib,
    mul_assoc,
  ]
  simpa [mul_assoc, mul_left_comm, mul_comm]

/-- When the nonlinear coefficient is zero, pooled and leafwise utilities coincide exactly. -/
theorem affineQuadratic_gap_zero_lambda
    (ω : ι → ℝ) (π : ι → κ → ℝ) (θ : κ → ℝ) (W : κ → κ → ℝ) :
    (∑ b : ι, ω b * affineQuadraticUtility θ W 0 (π b))
      - affineQuadraticUtility θ W 0 (weightedMean ω π) = 0 := by
  simpa using
    affineQuadratic_gap_eq_quadratic_gap (ω := ω) (π := π) (θ := θ) (W := W) (lam := (0 : ℝ))

end LeafLocalMixture

end FormalProofs.OPT
