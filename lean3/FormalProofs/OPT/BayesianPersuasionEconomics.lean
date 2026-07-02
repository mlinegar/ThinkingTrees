import FormalProofs.OPT.BayesianPersuasion
import FormalProofs.OPT.PreferenceScope

/-!
# FormalProofs/OPT/BayesianPersuasionEconomics.lean

Economic companion layer for the finite Bayesian-persuasion surface.

`BayesianPersuasion.lean` carries the Bayes algebra: signal experiments,
posterior beliefs, Bayes plausibility, splitting, Bayes actions, and symbolic
concavification.  This file connects those objects to the repo's existing
economic/formal-preference vocabulary:

* posterior beliefs are the economic state induced by a signal experiment;
* receiver posterior loss and sender indirect value factor through that belief
  state;
* direct recommendation obedience is receiver best-response optimality at the
  induced posterior, equivalently finite Bayes-action optimality for negative
  utility loss;
* sender experiment value is the weighted expected utility from induced
  posteriors and chosen receiver actions; and
* two experiments with the same induced posterior distribution have the same
  indirect persuasion value.

This remains the finite, assumption-backed persuasion layer.  It does not prove
the infinite-state direct-revelation principle, measurable selection,
compact-action existence, geometric concavification, or equilibrium existence.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {State Signal Action : Type*}

/-! ## Posterior beliefs as economic states -/

/-- The posterior-belief state induced by a persuasion experiment. -/
def PersuasionBeliefState
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ) :
    Signal → (State → ℝ) :=
  fun σ => PosteriorAfterSignal prior experiment σ

/-- Receiver posterior loss at a belief is negative expected utility. -/
def ReceiverPosteriorLoss
    [Fintype State]
    (receiverUtility : Action → State → ℝ)
    (belief : State → ℝ)
    (action : Action) : ℝ :=
  - ReceiverExpectedUtility receiverUtility belief action

/-- Receiver posterior loss, viewed as a signal-indexed decision loss, factors
through the posterior-belief state of the experiment. -/
theorem receiverPosteriorLoss_factorsThroughBeliefState
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (receiverUtility : Action → State → ℝ) :
    LossFactorsThroughState
      (PersuasionBeliefState prior experiment)
      (fun σ action =>
        ReceiverPosteriorLoss
          receiverUtility
          (PosteriorAfterSignal prior experiment σ)
          action) := by
  refine ⟨fun belief action => ReceiverPosteriorLoss receiverUtility belief action, ?_⟩
  intro σ action
  rfl

/-! ## Receiver best-response selectors and sender indirect value -/

/-- A belief-indexed action selector that always returns a receiver best
response. -/
def ReceiverBestResponseSelector
    [Fintype State]
    (receiverUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action) : Prop :=
  ∀ belief : State → ℝ,
    ReceiverOptimalAction receiverUtility belief (actionOfBelief belief)

/-- A belief-indexed action selector that implements optimistic tie-breaking
for the sender among receiver best responses. -/
def SenderTieBreakingSelector
    [Fintype State]
    (receiverUtility : Action → State → ℝ)
    (senderUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action) : Prop :=
  ∀ belief : State → ℝ,
    SenderPreferredReceiverBestResponse
      receiverUtility
      senderUtility
      belief
      (actionOfBelief belief)

/-- Sender-preferred tie-breaking is in particular a receiver best-response
selector. -/
theorem senderTieBreakingSelector_receiverBestResponseSelector
    [Fintype State]
    {receiverUtility : Action → State → ℝ}
    {senderUtility : Action → State → ℝ}
    {actionOfBelief : (State → ℝ) → Action}
    (hTie :
      SenderTieBreakingSelector
        receiverUtility
        senderUtility
        actionOfBelief) :
    ReceiverBestResponseSelector receiverUtility actionOfBelief := by
  intro belief
  exact (hTie belief).1

/-- Sender indirect value induced by a belief-indexed receiver-action
selector. -/
def SenderIndirectValueOfSelector
    [Fintype State]
    (senderUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action)
    (belief : State → ℝ) : ℝ :=
  SenderExpectedUtility senderUtility belief (actionOfBelief belief)

/-- Sender indirect value at signals factors through the posterior-belief state
whenever the receiver action is selected from the posterior belief. -/
theorem senderIndirectValue_factorsThroughBeliefState
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (senderUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action) :
    PreferenceFactorsThroughState
      (PersuasionBeliefState prior experiment)
      (fun σ =>
        SenderIndirectValueOfSelector
          senderUtility
          actionOfBelief
          (PosteriorAfterSignal prior experiment σ)) := by
  refine
    ⟨fun belief =>
      SenderIndirectValueOfSelector senderUtility actionOfBelief belief, ?_⟩
  intro σ
  rfl

/-! ## Experiment value and persuasion-scheme value -/

/-- Sender value of a concrete signal experiment and signal-indexed receiver
action rule. -/
def SignalExperimentSenderValue
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (senderUtility : Action → State → ℝ) : ℝ :=
  ∑ σ : Signal,
    SignalDistribution prior experiment σ *
      SenderExpectedUtility
        senderUtility
        (PosteriorAfterSignal prior experiment σ)
        (actionOfSignal σ)

/-- Sender value of a signal experiment when receiver actions are selected as a
function of the induced posterior belief. -/
def SignalExperimentIndirectValue
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (senderUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action) : ℝ :=
  PersuasionSchemeValue
    (SignalDistribution prior experiment)
    (PosteriorAfterSignal prior experiment)
    (SenderIndirectValueOfSelector senderUtility actionOfBelief)

/-- The indirect experiment value is exactly the persuasion-scheme value of the
experiment-induced distribution over posterior beliefs. -/
theorem signalExperimentIndirectValue_eq_persuasionSchemeValue
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (senderUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action) :
    SignalExperimentIndirectValue
        prior experiment senderUtility actionOfBelief =
      PersuasionSchemeValue
        (SignalDistribution prior experiment)
        (PosteriorAfterSignal prior experiment)
        (SenderIndirectValueOfSelector senderUtility actionOfBelief) := by
  rfl

/-- If a signal-indexed action rule is generated by a belief-indexed selector,
then concrete experiment value equals indirect experiment value. -/
theorem signalExperimentSenderValue_eq_indirectValue_of_selector
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (senderUtility : Action → State → ℝ)
    (actionOfSignal : Signal → Action)
    (actionOfBelief : (State → ℝ) → Action)
    (hAction :
      ∀ σ : Signal,
        actionOfSignal σ =
          actionOfBelief (PosteriorAfterSignal prior experiment σ)) :
    SignalExperimentSenderValue
        prior experiment actionOfSignal senderUtility =
      SignalExperimentIndirectValue
        prior experiment senderUtility actionOfBelief := by
  unfold SignalExperimentSenderValue SignalExperimentIndirectValue
  unfold PersuasionSchemeValue SenderIndirectValueOfSelector
  refine Finset.sum_congr rfl ?_
  intro σ _
  rw [hAction σ]

/-! ## Direct recommendations / obedience -/

/-- Direct recommendation obedience for a recommendation experiment whose
signals are receiver actions: every positive-probability recommendation is a
receiver best response at the posterior induced by that recommendation. -/
def ReceiverObedientRecommendation
    [Fintype State]
    [Fintype Action]
    (prior : State → ℝ)
    (recommendation : State → Action → ℝ)
    (receiverUtility : Action → State → ℝ) : Prop :=
  ∀ a : Action,
    SignalDistribution prior recommendation a ≠ 0 →
      ReceiverOptimalAction
        receiverUtility
        (PosteriorAfterSignal prior recommendation a)
        a

/-- Direct recommendation obedience is exactly finite Bayes-action optimality
for negative receiver utility loss on every positive-probability
recommendation. -/
theorem receiverObedientRecommendation_iff_bayesAction_negativeUtility
    [Fintype State]
    [Fintype Action]
    (prior : State → ℝ)
    (recommendation : State → Action → ℝ)
    (receiverUtility : Action → State → ℝ) :
    ReceiverObedientRecommendation prior recommendation receiverUtility ↔
      ∀ a : Action,
        SignalDistribution prior recommendation a ≠ 0 →
          BayesAction
            prior
            recommendation
            a
            (fun act θ => - receiverUtility act θ)
            a := by
  constructor
  · intro hObed a hMass
    have hOpt :
        ReceiverOptimalAction
          receiverUtility
          (BayesPosterior prior recommendation a)
          a := by
      simpa [posteriorAfterSignal_eq_bayesPosterior prior recommendation a]
        using hObed a hMass
    exact
      (bayesAction_negativeReceiverUtility_iff_receiverOptimalAction
        (prior := prior)
        (likelihood := recommendation)
        (obs := a)
        (receiverUtility := receiverUtility)
        (action := a)).mpr hOpt
  · intro hBayes a hMass
    have hOpt :
        ReceiverOptimalAction
          receiverUtility
          (BayesPosterior prior recommendation a)
          a :=
      (bayesAction_negativeReceiverUtility_iff_receiverOptimalAction
        (prior := prior)
        (likelihood := recommendation)
        (obs := a)
        (receiverUtility := receiverUtility)
        (action := a)).mp (hBayes a hMass)
    simpa [posteriorAfterSignal_eq_bayesPosterior prior recommendation a]
      using hOpt

/-- Sender value of an obedient/direct-recommendation experiment, before
imposing obedience.  The obedience constraint is carried separately by
`ReceiverObedientRecommendation`. -/
def DirectRecommendationSenderValue
    [Fintype State]
    [Fintype Action]
    (prior : State → ℝ)
    (recommendation : State → Action → ℝ)
    (senderUtility : Action → State → ℝ) : ℝ :=
  SignalExperimentSenderValue
    prior
    recommendation
    (fun a : Action => a)
    senderUtility

/-- Direct recommendation value is the concrete signal-experiment value with
the identity action rule on action-valued signals. -/
theorem directRecommendationSenderValue_eq_signalExperimentSenderValue
    [Fintype State]
    [Fintype Action]
    (prior : State → ℝ)
    (recommendation : State → Action → ℝ)
    (senderUtility : Action → State → ℝ) :
    DirectRecommendationSenderValue prior recommendation senderUtility =
      SignalExperimentSenderValue
        prior
        recommendation
        (fun a : Action => a)
        senderUtility := by
  rfl

/-! ## Posterior-distribution equivalence of experiments -/

/-- Two experiments are economically equivalent for belief-based persuasion
value when they induce the same signal probabilities and posterior beliefs
signal by signal.  This is a same-index finite version of equality of
distributions over posterior beliefs. -/
def SamePosteriorDistribution
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment₁ experiment₂ : State → Signal → ℝ) : Prop :=
  ∀ σ : Signal,
    SignalDistribution prior experiment₁ σ =
        SignalDistribution prior experiment₂ σ ∧
      PosteriorAfterSignal prior experiment₁ σ =
        PosteriorAfterSignal prior experiment₂ σ

/-- Same posterior distribution is reflexive. -/
theorem samePosteriorDistribution_refl
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ) :
    SamePosteriorDistribution prior experiment experiment := by
  intro σ
  exact ⟨rfl, rfl⟩

/-- Belief-based indirect persuasion value depends only on the induced
posterior distribution, not on the experiment's internal kernel once the
signal-indexed weights and posteriors agree. -/
theorem signalExperimentIndirectValue_eq_of_samePosteriorDistribution
    [Fintype State]
    [Fintype Signal]
    {prior : State → ℝ}
    {experiment₁ experiment₂ : State → Signal → ℝ}
    (senderUtility : Action → State → ℝ)
    (actionOfBelief : (State → ℝ) → Action)
    (hSame : SamePosteriorDistribution prior experiment₁ experiment₂) :
    SignalExperimentIndirectValue
        prior experiment₁ senderUtility actionOfBelief =
      SignalExperimentIndirectValue
        prior experiment₂ senderUtility actionOfBelief := by
  unfold SignalExperimentIndirectValue PersuasionSchemeValue
  unfold SenderIndirectValueOfSelector
  refine Finset.sum_congr rfl ?_
  intro σ _
  rcases hSame σ with ⟨hWeight, hPosterior⟩
  rw [hWeight, hPosterior]

/-- Concrete signal-experiment sender value is also invariant under same
signal-indexed posterior distribution when the action rule is the same on both
experiments. -/
theorem signalExperimentSenderValue_eq_of_samePosteriorDistribution
    [Fintype State]
    [Fintype Signal]
    {prior : State → ℝ}
    {experiment₁ experiment₂ : State → Signal → ℝ}
    (senderUtility : Action → State → ℝ)
    (actionOfSignal : Signal → Action)
    (hSame : SamePosteriorDistribution prior experiment₁ experiment₂) :
    SignalExperimentSenderValue
        prior experiment₁ actionOfSignal senderUtility =
      SignalExperimentSenderValue
        prior experiment₂ actionOfSignal senderUtility := by
  unfold SignalExperimentSenderValue
  refine Finset.sum_congr rfl ?_
  intro σ _
  rcases hSame σ with ⟨hWeight, hPosterior⟩
  rw [hWeight, hPosterior]

end FormalProofs.OPT
