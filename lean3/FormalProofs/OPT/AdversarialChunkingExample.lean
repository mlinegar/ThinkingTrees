import FormalProbability.DSL.SamplingConcentration

/-!
# Adversarial Chunking Example (Non-Uniform WOR, Self-Normalized Bound)

This file gives a concrete "adversarial chunking" theorem layer on top of the
existing non-uniform without-replacement concentration stack from
`FormalProbability/DSL/SamplingConcentration.lean`.

Interpretation:
- a chunking policy may place high-importance information into a small subset of
  chunks that are also hard to sample (adversarial overlap),
- we model this via a "good event" where the confidence radius is not too small,
- and a "bad event" probability bound `η`,
- then compose that with the existing Azuma-based mean-tail transfer theorem.
-/

set_option linter.mathlibStandardSet false

open scoped Real NNReal
open MeasureTheory ProbabilityTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace OPT

section AdversarialChunking

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} [StandardBorelSpace Ω]
variable {μ : Measure Ω} [IsZeroOrProbabilityMeasure μ]
variable {ℱ : Filtration ℕ mΩ}

/-- Adversarial chunking assumptions for a self-normalized confidence radius.

`goodEvent` can encode situations like:
- no extreme chunker misspecification on this draw,
- no severe concentration collapse in effective sample mass.

On `goodEvent`, the radius is strictly larger than `ε`; outside it, we pay `η`. -/
structure AdversarialChunkingInstance
    (model : DSL.NonUniformWithoutReplacementIPWLeafAuditFromCondExp Ω mΩ μ ℱ) where
  radius : Ω → ℝ
  goodEvent : Set Ω
  ε : ℝ
  η : ℝ
  hε : 0 ≤ ε
  hm_pos : 0 < model.m
  h_radius_good : ∀ ω, ω ∈ goodEvent → ε < radius ω
  h_bad_mass : μ.real (goodEventᶜ) ≤ η

namespace AdversarialChunkingInstance

variable {model : DSL.NonUniformWithoutReplacementIPWLeafAuditFromCondExp Ω mΩ μ ℱ}
variable (inst : AdversarialChunkingInstance (model := model))

/-- Radius-small event is controlled by the bad-event mass bound `η`. -/
theorem small_radius_event_le_eta :
    μ.real {ω | inst.radius ω ≤ inst.ε} ≤ inst.η := by
  have hsmall_to_compl :
      μ.real {ω | inst.radius ω ≤ inst.ε} ≤ μ.real (inst.goodEventᶜ) :=
    DSL.radius_small_event_le_good_compl
      (μ := μ) (R := inst.radius) (ε := inst.ε) (G := inst.goodEvent) inst.h_radius_good
  exact hsmall_to_compl.trans inst.h_bad_mass

/-- Full adversarial chunking failure-event bound (mean scale).

This is the explicit composed statement:
1) control `P(radius ≤ ε)` by the bad-event mass `η`,
2) control `P(ε ≤ |mean error|)` by the non-uniform WOR Azuma transfer,
3) combine them by the radius-event decomposition theorem. -/
theorem failure_bound :
    μ.real
      {ω |
        inst.radius ω ≤
          |(model.toNonUniformIPWLeafAudit.partialSum ω / (model.m : ℝ))|}
      ≤ inst.η + 2 * Real.exp
          (-((model.m : ℝ) * inst.ε) ^ 2 / (2 * model.stepVarianceProxy)) := by
  have hsmall : μ.real {ω | inst.radius ω ≤ inst.ε} ≤ inst.η :=
    inst.small_radius_event_le_eta (μ := μ)
  simpa using
    (model.radius_failure_bound_of_small_radius_and_mean_abs_tail_auto
      (μ := μ) (ℱ := ℱ)
      (radius := inst.radius)
      inst.hm_pos (ε := inst.ε) (η := inst.η) inst.hε hsmall)

end AdversarialChunkingInstance

end AdversarialChunking

end OPT
