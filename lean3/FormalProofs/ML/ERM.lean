import FormalProofs.ML.Core

/-!
# FormalProofs/ML/ERM.lean

Empirical risk minimization (ERM) over finite datasets.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ML

/-!
## Empirical Risk
-/

/-- Empirical risk: average loss over a finite dataset. -/
def empiricalRisk {X Y : Type*} {n : ℕ}
    (loss : Loss Y) (data : Dataset X Y n) (h : Hypothesis X Y) : ℝ :=
  (n : ℝ)⁻¹ * ∑ i : Fin n, lossOn loss h (data i)

/-- Empirical risk is nonnegative when the loss is nonnegative. -/
lemma empiricalRisk_nonneg {X Y : Type*} {n : ℕ}
    (loss : Loss Y) (data : Dataset X Y n) (h : Hypothesis X Y)
    (h_nonneg : ∀ yhat y, 0 ≤ loss yhat y) :
    0 ≤ empiricalRisk loss data h := by
  unfold empiricalRisk
  have h_sum : 0 ≤ ∑ i : Fin n, lossOn loss h (data i) := by
    apply Finset.sum_nonneg
    intro i _
    exact h_nonneg _ _
  have h_n : 0 ≤ (n : ℝ)⁻¹ := by
    exact inv_nonneg.mpr (by exact_mod_cast (Nat.zero_le n))
  exact mul_nonneg h_n h_sum

/-!
## ERM Predicate
-/

/-- ERM predicate: h minimizes empirical risk over the hypothesis class H. -/
def ERM {X Y : Type*} {n : ℕ}
    (H : Set (Hypothesis X Y)) (loss : Loss Y) (data : Dataset X Y n)
    (h : Hypothesis X Y) : Prop :=
  h ∈ H ∧ ∀ h' ∈ H, empiricalRisk loss data h ≤ empiricalRisk loss data h'

/-!
## Generalization Gap
-/

/-- Generalization gap: population risk minus empirical risk. -/
def generalizationGap {X Y : Type*} {n : ℕ} [MeasurableSpace X] [MeasurableSpace Y]
    (mu : Measure (X × Y)) [IsProbabilityMeasure mu]
    (loss : Loss Y) (data : Dataset X Y n) (h : Hypothesis X Y) : ℝ :=
  expectedRisk mu loss h - empiricalRisk loss data h

end ML
