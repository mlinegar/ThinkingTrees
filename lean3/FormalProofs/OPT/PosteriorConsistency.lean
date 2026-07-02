import FormalProofs.OPT.FiniteBayesOnState
import FormalProofs.DSL.AsymptoticCore

/-!
# FormalProofs/OPT/PosteriorConsistency.lean

Assumption-backed posterior consistency scaffolding.

This module deliberately formalizes the transport layer, not a classical
posterior-consistency theorem.  It reuses the repo-wide convergence-in-
probability notion from `DSL.AsymptoticCore`, records finite/discrete posterior
mass concentration, and proves that exact state/readout equalities preserve the
consistency statements.

What is theorem-backed here:

* pointwise-equal posterior sequences have the same consistency behavior;
* finite posterior mass concentration is preserved by pointwise equality;
* finite Bayes posteriors for likelihood-on-state families are exactly the
  corresponding state-space finite Bayes posteriors; and
* concentration transfers across exact state decoders.

What remains an explicit assumption:

* identifiability, prior positivity, likelihood-ratio separation, estimator
  consistency, and any classical Schwartz/doob-style posterior consistency
  theorem.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped Topology
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Ω X Rep State Θ Posterior : Type*}

/-- Posterior consistency is convergence in probability of posterior-like
objects in a posterior metric space. -/
def PosteriorConsistent
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [PseudoMetricSpace Posterior]
    (posteriorSeq : ℕ → Ω → Posterior)
    (posteriorLimit : Ω → Posterior) : Prop :=
  DSL.ConvergesInProbability μ posteriorSeq posteriorLimit

/-- Finite-parameter posterior concentration at a true parameter `θ0`: the
posterior mass assigned to `θ0` converges in probability to one. -/
def FinitePosteriorMassConcentratesAt
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    (posteriorSeq : ℕ → Ω → Θ → ℝ)
    (θ0 : Θ) : Prop :=
  DSL.ConvergesInProbability μ
    (fun n ω => posteriorSeq n ω θ0)
    (fun _ => (1 : ℝ))

/-- Finite Bayes posterior sequence induced by observations `dataSeq`. -/
def FiniteBayesPosteriorSeq
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X) :
    ℕ → Ω → Θ → ℝ :=
  fun n ω => BayesPosterior prior likelihood (dataSeq n ω)

/-- State-space finite Bayes posterior sequence induced by states `stateSeq`. -/
def StateFiniteBayesPosteriorSeq
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State) :
    ℕ → Ω → Θ → ℝ :=
  fun n ω => StateBayesPosterior prior stateLikelihood (stateSeq n ω)

/-- Evidence-ratio remainder sequence for finite Bayes observations.  The
target posterior mass is pointwise `(1 + remainder)⁻¹`. -/
def FiniteBayesEvidenceRatioRemainderSeq
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ) :
    ℕ → Ω → ℝ :=
  fun n ω =>
    BayesEvidenceRatioRemainder prior likelihood (dataSeq n ω) θ0

/-- State-space evidence-ratio remainder sequence. -/
def StateFiniteBayesEvidenceRatioRemainderSeq
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ) :
    ℕ → Ω → ℝ :=
  fun n ω =>
    StateBayesEvidenceRatioRemainder prior stateLikelihood (stateSeq n ω) θ0

/-- Deterministic finite-parameter identifiability surface.  This is a named
assumption in V1, not a route to an automatic posterior consistency theorem. -/
def FiniteBayesIdentifiable
    (likelihood : Θ → X → ℝ)
    (θ0 : Θ) : Prop :=
  ∀ ⦃θ : Θ⦄, θ ≠ θ0 → ∃ x : X, likelihood θ x ≠ likelihood θ0 x

/-- Prior mass at the target parameter is positive. -/
def PriorPositiveAt
    (prior : Θ → ℝ)
    (θ0 : Θ) : Prop :=
  0 < prior θ0

/-- Likelihood-ratio concentration toward the target parameter.  This is a
symbolic regularity condition used by the consistency assumption bundle. -/
def LikelihoodRatioConcentratesAt
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ) : Prop :=
  ∀ ⦃θ : Θ⦄, θ ≠ θ0 →
    DSL.ConvergesInProbability μ
      (fun n ω => likelihood θ (dataSeq n ω) /
        likelihood θ0 (dataSeq n ω))
      (fun _ => (0 : ℝ))

/-- State-space identifiability surface for finite Bayes consistency. -/
def StateFiniteBayesIdentifiable
    (stateLikelihood : Θ → State → ℝ)
    (θ0 : Θ) : Prop :=
  ∀ ⦃θ : Θ⦄, θ ≠ θ0 → ∃ z : State,
    stateLikelihood θ z ≠ stateLikelihood θ0 z

/-- State-space likelihood-ratio concentration toward the target parameter. -/
def StateLikelihoodRatioConcentratesAt
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ) : Prop :=
  ∀ ⦃θ : Θ⦄, θ ≠ θ0 →
    DSL.ConvergesInProbability μ
      (fun n ω => stateLikelihood θ (stateSeq n ω) /
        stateLikelihood θ0 (stateSeq n ω))
      (fun _ => (0 : ℝ))

/-- The regularity condition actually needed to turn an evidence-ratio
remainder into finite posterior mass concentration: the posterior transform of
the remainder converges to one.  V1 keeps the analytic continuous-mapping
argument as an assumption. -/
def FiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ) : Prop :=
  DSL.ConvergesInProbability μ
    (fun n ω =>
      (1 +
        FiniteBayesEvidenceRatioRemainderSeq
          prior
          likelihood
          dataSeq
          θ0
          n
          ω)⁻¹)
    (fun _ => (1 : ℝ))

/-- State-space posterior-transform concentration for the evidence-ratio
remainder. -/
def StateFiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ) : Prop :=
  DSL.ConvergesInProbability μ
    (fun n ω =>
      (1 +
        StateFiniteBayesEvidenceRatioRemainderSeq
          prior
          stateLikelihood
          stateSeq
          θ0
          n
          ω)⁻¹)
    (fun _ => (1 : ℝ))

/-- A finite-parameter likelihood-ratio/evidence-ratio sufficient-condition
bundle.  It records the usual statistical ingredients and the exact
posterior-transform convergence assumption needed in this V1 framework. -/
structure FiniteBayesLikelihoodRatioConsistencyCondition
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ) : Prop where
  identifiable : FiniteBayesIdentifiable likelihood θ0
  prior_positive : PriorPositiveAt prior θ0
  likelihood_ratio_concentrates :
    LikelihoodRatioConcentratesAt μ likelihood dataSeq θ0
  posterior_transform_concentrates :
    FiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne
      μ
      prior
      likelihood
      dataSeq
      θ0

/-- State-space finite-parameter likelihood-ratio/evidence-ratio
sufficient-condition bundle. -/
structure StateFiniteBayesLikelihoodRatioConsistencyCondition
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ) : Prop where
  identifiable : StateFiniteBayesIdentifiable stateLikelihood θ0
  prior_positive : PriorPositiveAt prior θ0
  likelihood_ratio_concentrates :
    StateLikelihoodRatioConcentratesAt μ stateLikelihood stateSeq θ0
  posterior_transform_concentrates :
    StateFiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne
      μ
      prior
      stateLikelihood
      stateSeq
      θ0

/-- Evidence-ratio posterior-transform concentration implies finite posterior
mass concentration, by the deterministic Bayes identity
`posterior θ0 = (1 + remainder)⁻¹`. -/
theorem finiteBayesPosteriorMassConcentration_of_evidenceRatioTransform
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ)
    (hTransform :
      FiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne
        μ
        prior
        likelihood
        dataSeq
        θ0) :
    FinitePosteriorMassConcentratesAt μ
      (FiniteBayesPosteriorSeq prior likelihood dataSeq)
      θ0 := by
  unfold FinitePosteriorMassConcentratesAt at *
  have hEq :
      (fun n ω =>
        FiniteBayesPosteriorSeq prior likelihood dataSeq n ω θ0) =
        (fun n ω =>
          (1 +
            FiniteBayesEvidenceRatioRemainderSeq
              prior
              likelihood
              dataSeq
              θ0
              n
              ω)⁻¹) := by
    funext n ω
    simp [FiniteBayesPosteriorSeq, FiniteBayesEvidenceRatioRemainderSeq,
      bayesPosterior_target_eq_inv_one_plus_evidenceRatioRemainder]
  simpa [FiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne, hEq]
    using hTransform

/-- The finite likelihood-ratio/evidence-ratio condition gives finite posterior
mass concentration. -/
theorem finiteBayesPosteriorMassConcentration_of_likelihoodRatioCondition
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ)
    (hCondition :
      FiniteBayesLikelihoodRatioConsistencyCondition
        μ
        prior
        likelihood
        dataSeq
        θ0) :
    FinitePosteriorMassConcentratesAt μ
      (FiniteBayesPosteriorSeq prior likelihood dataSeq)
      θ0 :=
  finiteBayesPosteriorMassConcentration_of_evidenceRatioTransform
    μ
    prior
    likelihood
    dataSeq
    θ0
    hCondition.posterior_transform_concentrates

/-- State-space evidence-ratio posterior-transform concentration implies state
finite posterior mass concentration. -/
theorem stateFiniteBayesPosteriorMassConcentration_of_evidenceRatioTransform
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ)
    (hTransform :
      StateFiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne
        μ
        prior
        stateLikelihood
        stateSeq
        θ0) :
    FinitePosteriorMassConcentratesAt μ
      (StateFiniteBayesPosteriorSeq prior stateLikelihood stateSeq)
      θ0 := by
  unfold FinitePosteriorMassConcentratesAt at *
  have hEq :
      (fun n ω =>
        StateFiniteBayesPosteriorSeq prior stateLikelihood stateSeq n ω θ0) =
        (fun n ω =>
          (1 +
            StateFiniteBayesEvidenceRatioRemainderSeq
              prior
              stateLikelihood
              stateSeq
              θ0
              n
              ω)⁻¹) := by
    funext n ω
    simp [StateFiniteBayesPosteriorSeq,
      StateFiniteBayesEvidenceRatioRemainderSeq,
      stateBayesPosterior_target_eq_inv_one_plus_evidenceRatioRemainder]
  simpa [StateFiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne, hEq]
    using hTransform

/-- The state-space likelihood-ratio/evidence-ratio condition gives finite
posterior mass concentration on the state posterior sequence. -/
theorem stateFiniteBayesPosteriorMassConcentration_of_likelihoodRatioCondition
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ)
    (hCondition :
      StateFiniteBayesLikelihoodRatioConsistencyCondition
        μ
        prior
        stateLikelihood
        stateSeq
        θ0) :
    FinitePosteriorMassConcentratesAt μ
      (StateFiniteBayesPosteriorSeq prior stateLikelihood stateSeq)
      θ0 :=
  stateFiniteBayesPosteriorMassConcentration_of_evidenceRatioTransform
    μ
    prior
    stateLikelihood
    stateSeq
    θ0
    hCondition.posterior_transform_concentrates

/-- Assumption bundle for finite/discrete Bayes posterior consistency on raw
observations.  The final `concentrates` field is the theorem-level property;
the preceding fields name the regularity ingredients rather than proving them. -/
structure FiniteBayesPosteriorConsistencyAssumption
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ) : Prop where
  identifiable : FiniteBayesIdentifiable likelihood θ0
  prior_positive : PriorPositiveAt prior θ0
  likelihood_ratio_concentrates :
    LikelihoodRatioConcentratesAt μ likelihood dataSeq θ0
  concentrates :
    FinitePosteriorMassConcentratesAt μ
      (FiniteBayesPosteriorSeq prior likelihood dataSeq)
      θ0

/-- Assumption bundle for finite/discrete Bayes posterior consistency on learned
states. -/
structure StateFiniteBayesPosteriorConsistencyAssumption
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (stateSeq : ℕ → Ω → State)
    (θ0 : Θ) : Prop where
  identifiable : StateFiniteBayesIdentifiable stateLikelihood θ0
  prior_positive : PriorPositiveAt prior θ0
  likelihood_ratio_concentrates :
    StateLikelihoodRatioConcentratesAt μ stateLikelihood stateSeq θ0
  concentrates :
    FinitePosteriorMassConcentratesAt μ
      (StateFiniteBayesPosteriorSeq prior stateLikelihood stateSeq)
      θ0

/-- Generic learned-posterior consistency assumption for an estimator/head
sequence. -/
def PosteriorEstimatorConsistencyAssumption
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [PseudoMetricSpace Posterior]
    (posteriorSeq : ℕ → Ω → Posterior)
    (posteriorLimit : Ω → Posterior) : Prop :=
  PosteriorConsistent μ posteriorSeq posteriorLimit

/-- Pointwise equality of posterior sequences and limits preserves posterior
consistency. -/
theorem posteriorConsistency_of_pointwise_equal
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [PseudoMetricSpace Posterior]
    {posteriorSeq₁ posteriorSeq₂ : ℕ → Ω → Posterior}
    {posteriorLimit₁ posteriorLimit₂ : Ω → Posterior}
    (hSeq : ∀ n ω, posteriorSeq₁ n ω = posteriorSeq₂ n ω)
    (hLimit : ∀ ω, posteriorLimit₁ ω = posteriorLimit₂ ω)
    (hConsistent : PosteriorConsistent μ posteriorSeq₁ posteriorLimit₁) :
    PosteriorConsistent μ posteriorSeq₂ posteriorLimit₂ := by
  have hSeqEq : posteriorSeq₁ = posteriorSeq₂ := by
    funext n ω
    exact hSeq n ω
  have hLimitEq : posteriorLimit₁ = posteriorLimit₂ := by
    funext ω
    exact hLimit ω
  simpa [PosteriorConsistent, hSeqEq, hLimitEq] using hConsistent

/-- Pointwise equality of finite posterior sequences preserves concentration
of mass at a target parameter. -/
theorem finitePosteriorMassConcentration_of_pointwise_equal
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    {posteriorSeq₁ posteriorSeq₂ : ℕ → Ω → Θ → ℝ}
    {θ0 : Θ}
    (hSeq : ∀ n ω θ, posteriorSeq₁ n ω θ = posteriorSeq₂ n ω θ)
    (hConcentrates :
      FinitePosteriorMassConcentratesAt μ posteriorSeq₁ θ0) :
    FinitePosteriorMassConcentratesAt μ posteriorSeq₂ θ0 := by
  unfold FinitePosteriorMassConcentratesAt at *
  have hMassEq :
      (fun n ω => posteriorSeq₁ n ω θ0) =
        (fun n ω => posteriorSeq₂ n ω θ0) := by
    funext n ω
    exact hSeq n ω θ0
  simpa [hMassEq] using hConcentrates

/-- Finite Bayes posterior sequence for a likelihood-on-state family is exactly
the state-space finite Bayes posterior sequence. -/
theorem finiteBayesPosteriorSeq_likelihoodOnState_eq_stateSeq
    [Fintype Θ]
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (dataSeq : ℕ → Ω → X) :
    FiniteBayesPosteriorSeq
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        dataSeq
      =
        StateFiniteBayesPosteriorSeq
          prior
          stateLikelihood
          (fun n ω => state (dataSeq n ω)) := by
  funext n ω θ
  simp [FiniteBayesPosteriorSeq, StateFiniteBayesPosteriorSeq,
    BayesPosterior, StateBayesPosterior, BayesNumerator,
    StateBayesNumerator, BayesEvidence, StateBayesEvidence,
    LikelihoodOnStateFamily]

/-- Finite-Bayes concentration for a likelihood-on-state family is equivalent
to concentration for the induced state posterior sequence. -/
theorem finiteBayesConsistency_likelihoodOnState_iff
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (dataSeq : ℕ → Ω → X)
    (θ0 : Θ) :
    FinitePosteriorMassConcentratesAt μ
        (FiniteBayesPosteriorSeq
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          dataSeq)
        θ0
      ↔
      FinitePosteriorMassConcentratesAt μ
        (StateFiniteBayesPosteriorSeq
          prior
          stateLikelihood
          (fun n ω => state (dataSeq n ω)))
        θ0 := by
  constructor
  · intro hRaw
    exact finitePosteriorMassConcentration_of_pointwise_equal
      (μ := μ)
      (posteriorSeq₁ :=
        FiniteBayesPosteriorSeq
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          dataSeq)
      (posteriorSeq₂ :=
        StateFiniteBayesPosteriorSeq
          prior
          stateLikelihood
          (fun n ω => state (dataSeq n ω)))
      (θ0 := θ0)
      (by
        intro n ω θ
        exact congrFun
          (congrFun
            (congrFun
              (finiteBayesPosteriorSeq_likelihoodOnState_eq_stateSeq
                (prior := prior)
                (state := state)
                (stateLikelihood := stateLikelihood)
                (dataSeq := dataSeq))
              n)
            ω)
          θ)
      hRaw
  · intro hState
    exact finitePosteriorMassConcentration_of_pointwise_equal
      (μ := μ)
      (posteriorSeq₁ :=
        StateFiniteBayesPosteriorSeq
          prior
          stateLikelihood
          (fun n ω => state (dataSeq n ω)))
      (posteriorSeq₂ :=
        FiniteBayesPosteriorSeq
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          dataSeq)
      (θ0 := θ0)
      (by
        intro n ω θ
        exact (congrFun
          (congrFun
            (congrFun
              (finiteBayesPosteriorSeq_likelihoodOnState_eq_stateSeq
                (prior := prior)
                (state := state)
                (stateLikelihood := stateLikelihood)
                (dataSeq := dataSeq))
              n)
            ω)
          θ).symm)
      hState

/-- If a representation sequence exactly decodes the state sequence, then
finite-Bayes posterior concentration on the state transfers to the decoded
representation state. -/
theorem stateReadout_finiteBayesConsistency
    [MeasurableSpace Ω]
    (μ : Measure Ω)
    [IsProbabilityMeasure μ]
    [Fintype Θ]
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    {repSeq : ℕ → Ω → Rep}
    {stateSeq : ℕ → Ω → State}
    {decodeState : Rep → State}
    {θ0 : Θ}
    (hReadout : ∀ n ω, decodeState (repSeq n ω) = stateSeq n ω)
    (hConcentrates :
      FinitePosteriorMassConcentratesAt μ
        (StateFiniteBayesPosteriorSeq prior stateLikelihood stateSeq)
        θ0) :
    FinitePosteriorMassConcentratesAt μ
      (StateFiniteBayesPosteriorSeq
        prior
        stateLikelihood
        (fun n ω => decodeState (repSeq n ω)))
      θ0 := by
  exact finitePosteriorMassConcentration_of_pointwise_equal
    (μ := μ)
    (posteriorSeq₁ :=
      StateFiniteBayesPosteriorSeq prior stateLikelihood stateSeq)
    (posteriorSeq₂ :=
      StateFiniteBayesPosteriorSeq
        prior
        stateLikelihood
        (fun n ω => decodeState (repSeq n ω)))
    (θ0 := θ0)
    (by
      intro n ω θ
      simp [StateFiniteBayesPosteriorSeq, hReadout n ω])
    hConcentrates

end FormalProofs.OPT
