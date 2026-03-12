import FormalProofs.OPT.PreferenceBounds

/-!
# FormalProofs/OPT/RUMSufficientConditions.lean

Sufficient conditions that imply the expected-group Lipschitz assumption used in
GRPO/GRPO-RL gap bounds.
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

variable {Strings : Type*}
variable {A : Type*}
variable {Y : Type*} [PseudoMetricSpace Y]
variable {k : ℕ}

/-- Stronger, pointwise version of expected-group Lipschitz. -/
def PointwiseGroupLossLipschitz
    (loss : Strings → (Fin k → A) → ℝ)
    (fstar : Strings → Y) (L : NNReal) : Prop :=
  ∀ group x z, |loss x group - loss z group| ≤ (L : ℝ) * dist (fstar x) (fstar z)

/-- Pointwise Lipschitz of group loss implies expected-group Lipschitz under any
group PMF. This discharges the RUM-style assumption from a stronger primitive
condition. -/
theorem expected_group_loss_lipschitz_of_pointwise
    (loss : Strings → (Fin k → A) → ℝ)
    (fstar : Strings → Y)
    (g : PMF (Fin k → A))
    (L : NNReal)
    (h_ptwise : PointwiseGroupLossLipschitz (k := k) loss fstar L)
    (x z : Strings)
    (h_sum_x : Summable (fun group : Fin k → A => (g group).toReal * loss x group))
    (h_sum_z : Summable (fun group : Fin k → A => (g group).toReal * loss z group))
    :
    ExpectedGroupLossLipschitz (loss := loss) (fstar := fstar) (g := g) (L := L) (x := x) (z := z) := by
  unfold ExpectedGroupLossLipschitz RUM.ExpectedGroupLossLipschitz
  have h_rewrite :
      |∑' group : Fin k → A, (g group).toReal * loss x group -
        ∑' group : Fin k → A, (g group).toReal * loss z group| =
      |∑' group : Fin k → A, (g group).toReal * (loss x group - loss z group)| := by
    rw [← Summable.tsum_sub h_sum_x h_sum_z]
    congr 1
    apply tsum_congr
    intro group
    ring
  rw [h_rewrite]
  let K : ℝ := (L : ℝ) * dist (fstar x) (fstar z)
  have hK : 0 ≤ K := by
    exact mul_nonneg (NNReal.coe_nonneg L) dist_nonneg
  have h_summable :
      Summable (fun group : Fin k → A => (g group).toReal * (loss x group - loss z group)) :=
    PMF.summable_coe_real_mul_of_bounded
      g (fun group => loss x group - loss z group) K hK
      (fun group => by
        simpa [K] using h_ptwise group x z)
  have h_abs :
      |∑' group : Fin k → A, (g group).toReal * (loss x group - loss z group)| ≤
      ∑' group : Fin k → A, |(g group).toReal * (loss x group - loss z group)| :=
    abs_tsum_le_tsum_abs' _ h_summable h_summable.abs
  have h_term :
      ∀ group : Fin k → A,
        |(g group).toReal * (loss x group - loss z group)| ≤ (g group).toReal * K := by
    intro group
    rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
    exact mul_le_mul_of_nonneg_left (by simpa [K] using h_ptwise group x z) ENNReal.toReal_nonneg
  have h_tsum_le :
      ∑' group : Fin k → A, |(g group).toReal * (loss x group - loss z group)| ≤
      ∑' group : Fin k → A, (g group).toReal * K := by
    apply Summable.tsum_le_tsum
    · exact h_term
    · exact h_summable.abs
    · exact PMF.summable_coe_real_mul_of_bounded g (fun _ => K) K hK (fun _ => by
        rw [abs_of_nonneg hK])
  have h_tsum_eq :
      (∑' group : Fin k → A, (g group).toReal * K) = K := by
    have hmul :
        (fun group : Fin k → A => (g group).toReal * K) =
        (fun group : Fin k → A => K * (g group).toReal) := by
      funext group
      ring
    rw [hmul, tsum_mul_left, PMF.toReal_tsum_coe g]
    ring
  calc
    |∑' group : Fin k → A, (g group).toReal * (loss x group - loss z group)|
      ≤ ∑' group : Fin k → A, |(g group).toReal * (loss x group - loss z group)| := h_abs
    _ ≤ ∑' group : Fin k → A, (g group).toReal * K := h_tsum_le
    _ = K := h_tsum_eq
    _ = (L : ℝ) * dist (fstar x) (fstar z) := by simp [K]

/-- Interface-friendly finite-index version: when the group index type is finite,
summability obligations are discharged automatically. -/
theorem expected_group_loss_lipschitz_of_pointwise_finite
    [Finite (Fin k → A)]
    (loss : Strings → (Fin k → A) → ℝ)
    (fstar : Strings → Y)
    (g : PMF (Fin k → A))
    (L : NNReal)
    (h_ptwise : PointwiseGroupLossLipschitz (k := k) loss fstar L)
    (x z : Strings) :
    ExpectedGroupLossLipschitz (loss := loss) (fstar := fstar) (g := g) (L := L) (x := x) (z := z) := by
  exact expected_group_loss_lipschitz_of_pointwise
    (k := k) (loss := loss) (fstar := fstar) (g := g) (L := L) h_ptwise x z
    (h_sum_x := by
      simpa using
        (Summable.of_finite
          (f := fun group : Fin k → A => (g group).toReal * loss x group)))
    (h_sum_z := by
      simpa using
        (Summable.of_finite
          (f := fun group : Fin k → A => (g group).toReal * loss z group)))

end FormalProofs.OPT
