import FormalProofs.OPT.BayesianPersuasionEconomics

/-!
# FormalProofs/OPT/BayesianPersuasionDirect.lean

Finite direct-recommendation layer for Bayesian persuasion.

Given a finite signal experiment and a deterministic receiver action rule, this
module constructs the action-valued direct recommendation experiment obtained by
pooling all signals that recommend the same action:

`rec(theta, a) = sum_{sigma : actionOfSignal sigma = a} experiment(theta, sigma)`.

The formalized claims are intentionally finite:

* the pooled recommendation kernel is a valid signal experiment whenever the
  original experiment is valid;
* ex-ante sender value is preserved exactly by the pooling construction; and
* when the original and pooled experiments have full support, the posterior
  value formulation agrees with the ex-ante value formulation, so posterior
  sender value is preserved too.

Obedience itself remains a receiver-optimality condition on the direct
recommendation posterior.  The stronger theorem that signalwise optimality
survives arbitrary pooling is not asserted here; it needs the usual finite
convexity/nonnegativity argument and tie-handling hypotheses.  This file
provides the direct-revelation accounting surface used by that next lemma.
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

/-- Pool an arbitrary finite signal experiment through a deterministic
signal-to-action rule to obtain an action-valued direct recommendation
experiment. -/
def DirectRecommendationFromExperiment
    [Fintype Signal]
    [DecidableEq Action]
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (θ : State)
    (a : Action) : ℝ :=
  ∑ σ : Signal,
    if a = actionOfSignal σ then experiment θ σ else 0

/-- Pooled direct recommendations are valid finite experiments whenever the
original signal experiment is valid. -/
theorem directRecommendationFromExperiment_valid
    [Fintype Signal]
    [Fintype Action]
    [DecidableEq Action]
    {experiment : State → Signal → ℝ}
    (actionOfSignal : Signal → Action)
    (hExp : SignalExperimentValid State Signal experiment) :
    SignalExperimentValid State Action
      (DirectRecommendationFromExperiment experiment actionOfSignal) where
  nonneg := by
    intro θ a
    unfold DirectRecommendationFromExperiment
    refine Finset.sum_nonneg ?_
    intro σ _
    by_cases h : a = actionOfSignal σ
    · simp [h, hExp.nonneg θ σ]
    · simp [h]
  sum_one := by
    intro θ
    unfold DirectRecommendationFromExperiment
    calc
      (∑ a : Action,
          ∑ σ : Signal,
            if a = actionOfSignal σ then experiment θ σ else 0)
          =
        ∑ σ : Signal,
          ∑ a : Action,
            if a = actionOfSignal σ then experiment θ σ else 0 := by
            rw [Finset.sum_comm]
      _ = ∑ σ : Signal, experiment θ σ := by
            refine Finset.sum_congr rfl ?_
            intro σ _
            simp
      _ = 1 := hExp.sum_one θ

/-! ## Ex-ante sender value and pooling preservation -/

/-- Ex-ante sender value of a signal experiment and deterministic
signal-indexed action rule, written before posterior normalization. -/
def SignalExperimentExAnteSenderValue
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (senderUtility : Action → State → ℝ) : ℝ :=
  ∑ θ : State,
    prior θ *
      (∑ σ : Signal,
        experiment θ σ * senderUtility (actionOfSignal σ) θ)

/-- Ex-ante sender value of a direct recommendation experiment. -/
def DirectRecommendationExAnteSenderValue
    [Fintype State]
    [Fintype Action]
    (prior : State → ℝ)
    (recommendation : State → Action → ℝ)
    (senderUtility : Action → State → ℝ) : ℝ :=
  ∑ θ : State,
    prior θ *
      (∑ a : Action,
        recommendation θ a * senderUtility a θ)

/-- Inner finite regrouping identity behind direct-recommendation value
preservation. -/
theorem directRecommendationFromExperiment_inner_senderValue_eq
    [Fintype Signal]
    [Fintype Action]
    [DecidableEq Action]
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (senderUtility : Action → State → ℝ)
    (θ : State) :
    (∑ a : Action,
        DirectRecommendationFromExperiment experiment actionOfSignal θ a *
          senderUtility a θ) =
      ∑ σ : Signal,
        experiment θ σ * senderUtility (actionOfSignal σ) θ := by
  unfold DirectRecommendationFromExperiment
  calc
    (∑ a : Action,
        (∑ σ : Signal,
          if a = actionOfSignal σ then experiment θ σ else 0) *
          senderUtility a θ)
        =
      ∑ a : Action,
        ∑ σ : Signal,
          (if a = actionOfSignal σ then experiment θ σ else 0) *
            senderUtility a θ := by
          refine Finset.sum_congr rfl ?_
          intro a _
          rw [Finset.sum_mul]
    _ =
      ∑ σ : Signal,
        ∑ a : Action,
          (if a = actionOfSignal σ then experiment θ σ else 0) *
            senderUtility a θ := by
          rw [Finset.sum_comm]
    _ =
      ∑ σ : Signal,
        experiment θ σ * senderUtility (actionOfSignal σ) θ := by
          refine Finset.sum_congr rfl ?_
          intro σ _
          simp

/-- Pooling a finite signal experiment through its induced action rule preserves
ex-ante sender value exactly. -/
theorem directRecommendationFromExperiment_exAnte_senderValue_eq
    [Fintype State]
    [Fintype Signal]
    [Fintype Action]
    [DecidableEq Action]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (senderUtility : Action → State → ℝ) :
    DirectRecommendationExAnteSenderValue
        prior
        (DirectRecommendationFromExperiment experiment actionOfSignal)
        senderUtility =
      SignalExperimentExAnteSenderValue
        prior
        experiment
        actionOfSignal
        senderUtility := by
  unfold DirectRecommendationExAnteSenderValue SignalExperimentExAnteSenderValue
  refine Finset.sum_congr rfl ?_
  intro θ _
  rw [directRecommendationFromExperiment_inner_senderValue_eq]

/-! ## Posterior-value agreement under full support -/

/-- Posterior sender value equals ex-ante sender value when every signal has
nonzero induced probability. -/
theorem signalExperimentSenderValue_eq_exAnteSenderValue
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (senderUtility : Action → State → ℝ)
    (hFull : SignalDistributionFullSupport prior experiment) :
    SignalExperimentSenderValue
        prior
        experiment
        actionOfSignal
        senderUtility =
      SignalExperimentExAnteSenderValue
        prior
        experiment
        actionOfSignal
        senderUtility := by
  unfold SignalExperimentSenderValue SignalExperimentExAnteSenderValue
  unfold SenderExpectedUtility PosteriorAfterSignal
  calc
    (∑ σ : Signal,
        SignalDistribution prior experiment σ *
          ∑ θ : State,
            (prior θ * experiment θ σ /
              SignalDistribution prior experiment σ) *
              senderUtility (actionOfSignal σ) θ)
        =
      ∑ σ : Signal,
        ∑ θ : State,
          SignalDistribution prior experiment σ *
            ((prior θ * experiment θ σ /
              SignalDistribution prior experiment σ) *
              senderUtility (actionOfSignal σ) θ) := by
          refine Finset.sum_congr rfl ?_
          intro σ _
          rw [Finset.mul_sum]
    _ =
      ∑ σ : Signal,
        ∑ θ : State,
          prior θ * experiment θ σ *
            senderUtility (actionOfSignal σ) θ := by
          refine Finset.sum_congr rfl ?_
          intro σ _
          refine Finset.sum_congr rfl ?_
          intro θ _
          field_simp [hFull σ]
    _ =
      ∑ θ : State,
        ∑ σ : Signal,
          prior θ * experiment θ σ *
            senderUtility (actionOfSignal σ) θ := by
          rw [Finset.sum_comm]
    _ =
      ∑ θ : State,
        prior θ *
          ∑ σ : Signal,
            experiment θ σ *
              senderUtility (actionOfSignal σ) θ := by
          refine Finset.sum_congr rfl ?_
          intro θ _
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro σ _
          ring

/-- Posterior sender value of a direct recommendation experiment equals its
ex-ante sender value under full support of recommendation probabilities. -/
theorem directRecommendationSenderValue_eq_exAnteSenderValue
    [Fintype State]
    [Fintype Action]
    (prior : State → ℝ)
    (recommendation : State → Action → ℝ)
    (senderUtility : Action → State → ℝ)
    (hFull : SignalDistributionFullSupport prior recommendation) :
    DirectRecommendationSenderValue prior recommendation senderUtility =
      DirectRecommendationExAnteSenderValue
        prior
        recommendation
        senderUtility := by
  unfold DirectRecommendationSenderValue
  exact
    signalExperimentSenderValue_eq_exAnteSenderValue
      (prior := prior)
      (experiment := recommendation)
      (actionOfSignal := fun a : Action => a)
      (senderUtility := senderUtility)
      hFull

/-- Full-support posterior-value version of direct-recommendation value
preservation. -/
theorem directRecommendationFromExperiment_senderValue_eq
    [Fintype State]
    [Fintype Signal]
    [Fintype Action]
    [DecidableEq Action]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (actionOfSignal : Signal → Action)
    (senderUtility : Action → State → ℝ)
    (hSignalFull : SignalDistributionFullSupport prior experiment)
    (hRecommendationFull :
      SignalDistributionFullSupport
        prior
        (DirectRecommendationFromExperiment experiment actionOfSignal)) :
    DirectRecommendationSenderValue
        prior
        (DirectRecommendationFromExperiment experiment actionOfSignal)
        senderUtility =
      SignalExperimentSenderValue
        prior
        experiment
        actionOfSignal
        senderUtility := by
  rw [
    directRecommendationSenderValue_eq_exAnteSenderValue
      (prior := prior)
      (recommendation :=
        DirectRecommendationFromExperiment experiment actionOfSignal)
      (senderUtility := senderUtility)
      hRecommendationFull,
    signalExperimentSenderValue_eq_exAnteSenderValue
      (prior := prior)
      (experiment := experiment)
      (actionOfSignal := actionOfSignal)
      (senderUtility := senderUtility)
      hSignalFull,
    directRecommendationFromExperiment_exAnte_senderValue_eq
      (prior := prior)
      (experiment := experiment)
      (actionOfSignal := actionOfSignal)
      (senderUtility := senderUtility)
  ]

end FormalProofs.OPT
