import FormalProofs.OPT.SerflingAudit
import FormalProofs.OPT.AdversarialChunkingExample

set_option linter.mathlibStandardSet false

open scoped Real NNReal
open MeasureTheory ProbabilityTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-! Compatibility wrappers exposing Serfling/Azuma interfaces under
`FormalProofs.OPT.*` while preserving canonical theorem sources in `_root_.OPT.*`. -/

section SerflingAzumaCompat

variable {Ω : Type*}
variable {m mΩ : MeasurableSpace Ω} {hm : m ≤ mΩ}
variable [StandardBorelSpace Ω]
variable {μ : Measure Ω} [IsFiniteMeasure μ]
variable {X : Ω → ℝ} {a b : ℝ}

lemma hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero
    (hX : Measurable X)
    (hb : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b)
    (hcond : μ[X|m] =ᵐ[μ.trim hm] 0) :
    HasCondSubgaussianMGF m hm X ((‖b - a‖₊ / 2) ^ 2) μ := by
  exact _root_.OPT.hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero
    (hm := hm) (hX := hX) (hb := hb) (hcond := hcond)

end SerflingAzumaCompat

section AzumaFromBoundsCompat

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} [StandardBorelSpace Ω]
variable {μ : Measure Ω} [IsZeroOrProbabilityMeasure μ]
variable {ℱ : Filtration ℕ mΩ}
variable {Y : ℕ → Ω → ℝ}

theorem azuma_hoeffding_of_mem_Icc_of_condExp_eq_zero
    (a b : ℕ → ℝ)
    (h_meas : ∀ i, Measurable (Y i))
    (hY0 : Y 0 = 0)
    (n : ℕ)
    (h_adapted : Adapted ℱ Y)
    (h_bound : ∀ i, i < n - 1 → ∀ᵐ ω ∂μ, Y (i + 1) ω ∈ Set.Icc (a (i + 1)) (b (i + 1)))
    (h_cond : ∀ i, i < n - 1 → μ[Y (i + 1)|ℱ i] =ᵐ[μ] 0)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ ∑ i ∈ Finset.range n, Y i ω}
      ≤ Real.exp
        (-ε ^ 2 /
          (2 * ∑ i ∈ Finset.range n, (if i = 0 then (0 : ℝ≥0) else (‖b i - a i‖₊ / 2) ^ 2))) := by
  exact _root_.OPT.azuma_hoeffding_of_mem_Icc_of_condExp_eq_zero
    (μ := μ) (ℱ := ℱ) (Y := Y)
    (a := a) (b := b)
    (h_meas := h_meas) (hY0 := hY0) (n := n)
    (h_adapted := h_adapted) (h_bound := h_bound) (h_cond := h_cond)
    (hε := hε)

theorem azuma_hoeffding_abs_of_mem_Icc_of_condExp_eq_zero
    (a b : ℕ → ℝ)
    (h_meas : ∀ i, Measurable (Y i))
    (hY0 : Y 0 = 0)
    (n : ℕ)
    (h_adapted : Adapted ℱ Y)
    (h_bound : ∀ i, i < n - 1 → ∀ᵐ ω ∂μ, Y (i + 1) ω ∈ Set.Icc (a (i + 1)) (b (i + 1)))
    (h_cond : ∀ i, i < n - 1 → μ[Y (i + 1)|ℱ i] =ᵐ[μ] 0)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ |∑ i ∈ Finset.range n, Y i ω|}
      ≤ 2 * Real.exp
        (-ε ^ 2 /
          (2 * ∑ i ∈ Finset.range n, (if i = 0 then (0 : ℝ≥0) else (‖b i - a i‖₊ / 2) ^ 2))) := by
  exact _root_.OPT.azuma_hoeffding_abs_of_mem_Icc_of_condExp_eq_zero
    (μ := μ) (ℱ := ℱ) (Y := Y)
    (a := a) (b := b)
    (h_meas := h_meas) (hY0 := hY0) (n := n)
    (h_adapted := h_adapted) (h_bound := h_bound) (h_cond := h_cond)
    (hε := hε)

abbrev RandomPermutationWOR (Ω : Type*) : Type _ := _root_.OPT.RandomPermutationWOR Ω

theorem azuma_hoeffding_abs_of_random_permutation
    {Ω : Type*} {mΩ : MeasurableSpace Ω} [StandardBorelSpace Ω]
    {μ : Measure Ω} [IsZeroOrProbabilityMeasure μ]
    {ℱ : Filtration ℕ mΩ}
    (model : RandomPermutationWOR Ω)
    (center : ℝ)
    (h_sampled_meas :
      ∀ i : Fin model.m, Measurable (fun ω => model.sampledValue i ω))
    (h_adapted : Adapted ℱ (model.increments center))
    (h_cond : ∀ i : ℕ, i < model.m → μ[model.increments center (i + 1)|ℱ i] =ᵐ[μ] 0)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ |∑ i ∈ Finset.range (model.m + 1), model.increments center i ω|}
      ≤ 2 * Real.exp
        (-ε ^ 2 /
          (2 * ∑ i ∈ Finset.range (model.m + 1),
            (if i = 0 then (0 : ℝ≥0)
             else (‖model.upperStep center i - model.lowerStep center i‖₊ / 2) ^ 2))) := by
  exact _root_.OPT.azuma_hoeffding_abs_of_random_permutation
    (μ := μ) (ℱ := ℱ)
    (model := model) (center := center)
    (h_sampled_meas := h_sampled_meas)
    (h_adapted := h_adapted) (h_cond := h_cond)
    (hε := hε)

end AzumaFromBoundsCompat

section AdversarialCompat

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} [StandardBorelSpace Ω]
variable {μ : Measure Ω} [IsZeroOrProbabilityMeasure μ]
variable {ℱ : Filtration ℕ mΩ}

abbrev AdversarialChunkingInstance
    (model : DSL.NonUniformWithoutReplacementIPWLeafAuditFromCondExp Ω mΩ μ ℱ) : Type _ :=
  _root_.OPT.AdversarialChunkingInstance (model := model)

namespace AdversarialChunkingInstance

variable {model : DSL.NonUniformWithoutReplacementIPWLeafAuditFromCondExp Ω mΩ μ ℱ}
variable (inst : AdversarialChunkingInstance (model := model))

theorem small_radius_event_le_eta :
    μ.real {ω | inst.radius ω ≤ inst.ε} ≤ inst.η := by
  exact _root_.OPT.AdversarialChunkingInstance.small_radius_event_le_eta
    (μ := μ) (inst := inst)

theorem failure_bound :
    μ.real
      {ω |
        inst.radius ω ≤
          |(model.toNonUniformIPWLeafAudit.partialSum ω / (model.m : ℝ))|}
      ≤ inst.η + 2 * Real.exp
          (-((model.m : ℝ) * inst.ε) ^ 2 / (2 * model.stepVarianceProxy)) := by
  exact _root_.OPT.AdversarialChunkingInstance.failure_bound
    (μ := μ) (ℱ := ℱ) (inst := inst)

end AdversarialChunkingInstance

end AdversarialCompat

end FormalProofs.OPT
