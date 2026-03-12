import Mathlib.MeasureTheory.Function.StronglyMeasurable.AEStronglyMeasurable
import Mathlib.Probability.Independence.Integration

import FormalProofs.CLT.Core

/-!
# FormalProofs/CLT/GeneratingFunctions.lean

Probability generating functions and basic independence properties.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped Topology

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ProbabilityTheory

open MeasureTheory

/-!
## Probability generating function
-/

/-- Probability generating function for an `ℕ`-valued random variable. -/
def pgf {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) (X : Omega → ℕ) (z : ℂ) : ℂ :=
  ∫ omega, (z ^ (X omega)) ∂mu

/-!
## PGF of sums of independent random variables
-/

lemma pgf_add_of_indep {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {X Y : Omega → ℕ} (hX : AEMeasurable X mu) (hY : AEMeasurable Y mu)
    (hXY : X ⟂ᵢ[mu] Y) (z : ℂ) :
    pgf mu (fun omega => X omega + Y omega) z = pgf mu X z * pgf mu Y z := by
  have hpow_meas : Measurable (fun n : ℕ => z ^ n) := by
    simpa using (measurable_of_countable (fun n : ℕ => z ^ n))
  have hpow_aes_X : AEStronglyMeasurable (fun n : ℕ => z ^ n) (mu.map X) :=
    hpow_meas.aestronglyMeasurable
  have hpow_aes_Y : AEStronglyMeasurable (fun n : ℕ => z ^ n) (mu.map Y) :=
    hpow_meas.aestronglyMeasurable
  have hprod :=
    (IndepFun.integral_fun_comp_mul_comp (μ := mu) (X := X) (Y := Y)
      (f := fun n : ℕ => z ^ n) (g := fun n : ℕ => z ^ n)
      hXY hX hY hpow_aes_X hpow_aes_Y)
  calc
    pgf mu (fun omega => X omega + Y omega) z
        = ∫ omega, z ^ (X omega + Y omega) ∂mu := rfl
    _ = ∫ omega, z ^ (X omega) * z ^ (Y omega) ∂mu := by
        simp [pow_add]
    _ = (∫ omega, z ^ (X omega) ∂mu) * ∫ omega, z ^ (Y omega) ∂mu := by
        simpa [Function.comp] using hprod
    _ = pgf mu X z * pgf mu Y z := rfl

end ProbabilityTheory
