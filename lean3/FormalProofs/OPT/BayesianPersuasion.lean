import Mathlib.Algebra.BigOperators.Ring.Finset
import FormalProofs.OPT.FiniteBayesOnState
import FormalProofs.OPT.PreferenceScope

/-!
# FormalProofs/OPT/BayesianPersuasion.lean

Finite Bayesian-persuasion surface for the Kamenica--Gentzkow claim pattern.

The goal is deliberately bounded.  States and signal realizations are finite,
beliefs and experiments are real-valued finite kernels, and Bayes updating is
the same normalized finite Bayes posterior already used in
`FiniteBayesOnState`.  We formalize the algebraic core:

* a signal experiment induces a signal distribution and posterior beliefs;
* posterior beliefs are exactly finite Bayes posteriors for the signal
  likelihood;
* under positive signal support, induced posteriors are Bayes-plausible: their
  weighted barycenter is the prior;
* receiver best responses are Bayes actions for the negative receiver utility
  loss; and
* the paper's concavification theorem is represented as a symbolic optimal
  value/witness interface over Bayes-plausible posterior distributions.

This does not prove the geometric splitting lemma, compact-action existence,
upper semicontinuity, measurable-selection/tie-breaking, or infinite-state
concavification.  Those are explicit analytic assumptions outside this finite
transport layer.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {State Signal Action I : Type*}

/-! ## Finite beliefs, experiments, and induced posteriors -/

/-- A finite real-valued probability vector. -/
structure IsFiniteProbability
    {α : Type*}
    [Fintype α]
    (p : α → ℝ) : Prop where
  nonneg : ∀ a : α, 0 ≤ p a
  sum_one : (∑ a : α, p a) = 1

/-- A finite Bayesian-persuasion experiment: for every state, the signal kernel
is a probability vector. -/
structure SignalExperimentValid
    (State Signal : Type*)
    [Fintype Signal]
    (experiment : State → Signal → ℝ) : Prop where
  nonneg : ∀ (θ : State) (σ : Signal), 0 ≤ experiment θ σ
  sum_one : ∀ θ : State, (∑ σ : Signal, experiment θ σ) = 1

/-- Distribution over signal realizations induced by a prior and experiment. -/
def SignalDistribution
    [Fintype State]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (σ : Signal) : ℝ :=
  ∑ θ : State, prior θ * experiment θ σ

/-- Posterior belief after a signal realization.  This is the finite Bayes
posterior using the experiment as the signal likelihood. -/
def PosteriorAfterSignal
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (σ : Signal)
    (θ : State) : ℝ :=
  prior θ * experiment θ σ / SignalDistribution prior experiment σ

/-- Positive-support assumption for all retained signal realizations.  In a
finite model one can instead drop zero-probability signals; this predicate is
the clean algebraic interface for the all-signals statement. -/
def SignalDistributionFullSupport
    [Fintype State]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ) : Prop :=
  ∀ σ : Signal, SignalDistribution prior experiment σ ≠ 0

/-- The posterior after a persuasion signal is definitionally the finite Bayes
posterior for observation type `Signal` and likelihood `experiment`. -/
theorem posteriorAfterSignal_eq_bayesPosterior
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (σ : Signal) :
    PosteriorAfterSignal prior experiment σ =
      BayesPosterior prior experiment σ := by
  funext θ
  simp [PosteriorAfterSignal, BayesPosterior, BayesNumerator,
    BayesEvidence, SignalDistribution]

/-- The induced signal distribution has nonnegative mass under nonnegative
prior and nonnegative experiment probabilities. -/
theorem signalDistribution_nonneg
    [Fintype State]
    [Fintype Signal]
    {prior : State → ℝ}
    {experiment : State → Signal → ℝ}
    (hPrior : IsFiniteProbability prior)
    (hExp : SignalExperimentValid State Signal experiment)
    (σ : Signal) :
    0 ≤ SignalDistribution prior experiment σ := by
  unfold SignalDistribution
  exact Finset.sum_nonneg
    (fun θ _ => mul_nonneg (hPrior.nonneg θ) (hExp.nonneg θ σ))

/-- The induced signal distribution sums to one. -/
theorem signalDistribution_sum_eq_one
    [Fintype State]
    [Fintype Signal]
    {prior : State → ℝ}
    {experiment : State → Signal → ℝ}
    (hPrior : IsFiniteProbability prior)
    (hExp : SignalExperimentValid State Signal experiment) :
    (∑ σ : Signal, SignalDistribution prior experiment σ) = 1 := by
  unfold SignalDistribution
  calc
    (∑ σ : Signal, ∑ θ : State, prior θ * experiment θ σ)
        = ∑ θ : State, ∑ σ : Signal, prior θ * experiment θ σ := by
          rw [Finset.sum_comm]
    _ = ∑ θ : State, prior θ * (∑ σ : Signal, experiment θ σ) := by
          refine Finset.sum_congr rfl ?_
          intro θ _
          rw [Finset.mul_sum]
    _ = ∑ θ : State, prior θ * 1 := by
          refine Finset.sum_congr rfl ?_
          intro θ _
          rw [hExp.sum_one θ]
    _ = ∑ θ : State, prior θ := by
          simp
    _ = 1 := hPrior.sum_one

/-- The induced signal distribution is a finite probability vector. -/
theorem signalDistribution_isFiniteProbability
    [Fintype State]
    [Fintype Signal]
    {prior : State → ℝ}
    {experiment : State → Signal → ℝ}
    (hPrior : IsFiniteProbability prior)
    (hExp : SignalExperimentValid State Signal experiment) :
    IsFiniteProbability (SignalDistribution prior experiment) where
  nonneg := signalDistribution_nonneg hPrior hExp
  sum_one := signalDistribution_sum_eq_one hPrior hExp

/-! ## Bayes plausibility and concavification vocabulary -/

/-- A finite posterior distribution is Bayes-plausible when its barycenter is
the prior.  This is the finite belief-distribution condition used in the
Kamenica--Gentzkow concavification theorem. -/
def BayesPlausiblePosteriorDistribution
    [Fintype State]
    [Fintype I]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ) : Prop :=
  ∀ θ : State, (∑ i : I, weight i * posterior i θ) = prior θ

/-- A finite persuasion scheme: a probability distribution over posterior
beliefs whose barycenter is the prior.  We keep posterior validity separate
from barycenter validity so zero-weight or externally supplied posterior
labels do not force extra normalization obligations in this symbolic layer. -/
def PersuasionSchemeFeasible
    [Fintype State]
    [Fintype I]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ) : Prop :=
  IsFiniteProbability weight ∧
    BayesPlausiblePosteriorDistribution prior weight posterior

/-- Stronger finite persuasion scheme where each posterior label is itself a
finite probability vector.  This is the input needed for the finite splitting
construction. -/
structure PersuasionSchemeBeliefFeasible
    [Fintype State]
    [Fintype I]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ) : Prop where
  weight_prob : IsFiniteProbability weight
  posterior_prob : ∀ i : I, IsFiniteProbability (posterior i)
  bayes_plausible :
    BayesPlausiblePosteriorDistribution prior weight posterior

/-- The finite splitting construction: implement a Bayes-plausible
posterior-decomposition by using posterior labels as signals and setting
`Pr(signal=i | theta) = weight_i * posterior_i(theta) / prior(theta)`.
Positive prior support is supplied to the theorems using this construction. -/
def SplittingExperiment
    [Fintype State]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ)
    (θ : State)
    (i : I) : ℝ :=
  weight i * posterior i θ / prior θ

/-- Finite splitting lemma, construction direction: a Bayes-plausible
probability distribution over posterior beliefs with positive prior support
defines a valid signal experiment. -/
theorem splittingExperiment_valid_of_bayesPlausible
    [Fintype State]
    [Fintype I]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ)
    (hPriorPos : ∀ θ : State, 0 < prior θ)
    (hScheme : PersuasionSchemeBeliefFeasible prior weight posterior) :
    SignalExperimentValid State I
      (SplittingExperiment prior weight posterior) where
  nonneg := by
    intro θ i
    unfold SplittingExperiment
    exact div_nonneg
      (mul_nonneg
        (hScheme.weight_prob.nonneg i)
        ((hScheme.posterior_prob i).nonneg θ))
      (le_of_lt (hPriorPos θ))
  sum_one := by
    intro θ
    unfold SplittingExperiment
    calc
      (∑ i : I, weight i * posterior i θ / prior θ)
          = (∑ i : I, weight i * posterior i θ) / prior θ := by
            simp [div_eq_mul_inv, Finset.sum_mul]
      _ = prior θ / prior θ := by
            rw [hScheme.bayes_plausible θ]
      _ = 1 := by
            field_simp [ne_of_gt (hPriorPos θ)]

/-- The splitting experiment induces exactly the supplied signal weights. -/
theorem signalDistribution_splittingExperiment_eq_weight
    [Fintype State]
    [Fintype I]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ)
    (hPriorPos : ∀ θ : State, 0 < prior θ)
    (hScheme : PersuasionSchemeBeliefFeasible prior weight posterior)
    (i : I) :
    SignalDistribution
        prior
        (SplittingExperiment prior weight posterior)
        i =
      weight i := by
  unfold SignalDistribution SplittingExperiment
  calc
    (∑ θ : State, prior θ * (weight i * posterior i θ / prior θ))
        = ∑ θ : State, weight i * posterior i θ := by
          refine Finset.sum_congr rfl ?_
          intro θ _
          field_simp [ne_of_gt (hPriorPos θ)]
    _ = weight i * (∑ θ : State, posterior i θ) := by
          rw [Finset.mul_sum]
    _ = weight i := by
          rw [(hScheme.posterior_prob i).sum_one, mul_one]

/-- Signals from the splitting construction recover the supplied posterior
beliefs on every positive-weight signal. -/
theorem posteriorAfterSignal_splittingExperiment_eq_posterior
    [Fintype State]
    [Fintype I]
    (prior : State → ℝ)
    (weight : I → ℝ)
    (posterior : I → State → ℝ)
    (hPriorPos : ∀ θ : State, 0 < prior θ)
    (hScheme : PersuasionSchemeBeliefFeasible prior weight posterior)
    {i : I}
    (hWeight : weight i ≠ 0) :
    PosteriorAfterSignal
        prior
        (SplittingExperiment prior weight posterior)
        i =
      posterior i := by
  funext θ
  unfold PosteriorAfterSignal
  rw [signalDistribution_splittingExperiment_eq_weight
      (prior := prior)
      (weight := weight)
      (posterior := posterior)
      (hPriorPos := hPriorPos)
      (hScheme := hScheme)
      (i := i)]
  unfold SplittingExperiment
  field_simp [ne_of_gt (hPriorPos θ), hWeight]

/-- Paper-facing Bayes-plausibility theorem: a valid finite experiment with
full-support signal distribution induces Bayes-plausible posteriors. -/
theorem validSignalExperiment_bayesPlausible_of_fullSupport
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (hExp : SignalExperimentValid State Signal experiment)
    (hFull : SignalDistributionFullSupport prior experiment) :
    BayesPlausiblePosteriorDistribution
      prior
      (SignalDistribution prior experiment)
      (PosteriorAfterSignal prior experiment) := by
  intro θ
  unfold PosteriorAfterSignal
  calc
    (∑ σ : Signal,
        SignalDistribution prior experiment σ *
          (prior θ * experiment θ σ /
            SignalDistribution prior experiment σ))
        = ∑ σ : Signal, prior θ * experiment θ σ := by
          refine Finset.sum_congr rfl ?_
          intro σ _
          field_simp [hFull σ]
    _ = prior θ * (∑ σ : Signal, experiment θ σ) := by
          rw [Finset.mul_sum]
    _ = prior θ := by
          rw [hExp.sum_one θ, mul_one]

/-- A valid finite experiment induces a feasible persuasion scheme over its
signal-indexed posterior beliefs under full signal support. -/
theorem validSignalExperiment_persuasionSchemeFeasible_of_fullSupport
    [Fintype State]
    [Fintype Signal]
    (prior : State → ℝ)
    (experiment : State → Signal → ℝ)
    (hPrior : IsFiniteProbability prior)
    (hExp : SignalExperimentValid State Signal experiment)
    (hFull : SignalDistributionFullSupport prior experiment) :
    PersuasionSchemeFeasible
      prior
      (SignalDistribution prior experiment)
      (PosteriorAfterSignal prior experiment) := by
  exact
    ⟨signalDistribution_isFiniteProbability hPrior hExp,
      validSignalExperiment_bayesPlausible_of_fullSupport
        prior experiment hExp hFull⟩

/-! ## Receiver best responses and Bayes actions -/

/-- Generic finite argmax predicate for action scores. -/
def IsFiniteArgmax
    {α : Type*}
    (score : α → ℝ)
    (a : α) : Prop :=
  ∀ b : α, score b ≤ score a

/-- Receiver expected utility at a belief. -/
def ReceiverExpectedUtility
    [Fintype State]
    (receiverUtility : Action → State → ℝ)
    (belief : State → ℝ)
    (action : Action) : ℝ :=
  ∑ θ : State, belief θ * receiverUtility action θ

/-- Sender expected utility from a receiver action at a belief. -/
def SenderExpectedUtility
    [Fintype State]
    (senderUtility : Action → State → ℝ)
    (belief : State → ℝ)
    (action : Action) : ℝ :=
  ∑ θ : State, belief θ * senderUtility action θ

/-- Receiver best response at a posterior belief. -/
def ReceiverOptimalAction
    [Fintype State]
    (receiverUtility : Action → State → ℝ)
    (belief : State → ℝ)
    (action : Action) : Prop :=
  IsFiniteArgmax
    (fun a => ReceiverExpectedUtility receiverUtility belief a)
    action

/-- Sender-preferred receiver best response, i.e. the optimistic tie-breaking
version of the indirect sender value used in the finite persuasion model. -/
def SenderPreferredReceiverBestResponse
    [Fintype State]
    (receiverUtility : Action → State → ℝ)
    (senderUtility : Action → State → ℝ)
    (belief : State → ℝ)
    (action : Action) : Prop :=
  ReceiverOptimalAction receiverUtility belief action ∧
    ∀ b : Action,
      ReceiverOptimalAction receiverUtility belief b →
        SenderExpectedUtility senderUtility belief b ≤
          SenderExpectedUtility senderUtility belief action

/-- Posterior risk for negative receiver utility is the negative of receiver
expected utility. -/
theorem bayesPosteriorRisk_negativeReceiverUtility_eq_neg_expectedUtility
    [Fintype State]
    {Observation : Type*}
    (prior : State → ℝ)
    (likelihood : State → Observation → ℝ)
    (obs : Observation)
    (receiverUtility : Action → State → ℝ)
    (action : Action) :
    BayesPosteriorRisk
        prior
        likelihood
        obs
        (fun a θ => - receiverUtility a θ)
        action =
      - ReceiverExpectedUtility
          receiverUtility
          (BayesPosterior prior likelihood obs)
          action := by
  unfold BayesPosteriorRisk BayesPosteriorExpectation ReceiverExpectedUtility
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl ?_
  intro θ _
  ring

/-- Receiver best responses are exactly Bayes actions for the negative-utility
loss under the finite Bayes posterior. -/
theorem bayesAction_negativeReceiverUtility_iff_receiverOptimalAction
    [Fintype State]
    {Observation : Type*}
    (prior : State → ℝ)
    (likelihood : State → Observation → ℝ)
    (obs : Observation)
    (receiverUtility : Action → State → ℝ)
    (action : Action) :
    BayesAction
        prior
        likelihood
        obs
        (fun a θ => - receiverUtility a θ)
        action ↔
      ReceiverOptimalAction
        receiverUtility
        (BayesPosterior prior likelihood obs)
        action := by
  constructor
  · intro h b
    have hb := h b
    have hb' :
        - ReceiverExpectedUtility
            receiverUtility
            (BayesPosterior prior likelihood obs)
            action ≤
          - ReceiverExpectedUtility
            receiverUtility
            (BayesPosterior prior likelihood obs)
            b := by
      simpa only [
        bayesPosteriorRisk_negativeReceiverUtility_eq_neg_expectedUtility]
        using hb
    exact (neg_le_neg_iff.mp hb')
  · intro h b
    have hb := h b
    have hneg := neg_le_neg hb
    simpa only [
      bayesPosteriorRisk_negativeReceiverUtility_eq_neg_expectedUtility]
      using hneg

/-! ## Symbolic concavification / optimal-value surface -/

/-- Value of a finite persuasion scheme for a belief-indexed indirect sender
value. -/
def PersuasionSchemeValue
    [Fintype State]
    [Fintype I]
    (weight : I → ℝ)
    (posterior : I → State → ℝ)
    (senderIndirectValue : (State → ℝ) → ℝ) : ℝ :=
  ∑ i : I, weight i * senderIndirectValue (posterior i)

/-- A finite concavification witness: `opt` upper-bounds every Bayes-plausible
posterior decomposition of the prior and is achieved by some finite
decomposition.  This is the theorem-ready shape of the Kamenica--Gentzkow
concavification claim; analytic existence/geometry can be supplied separately. -/
structure ConcavificationWitness
    [Fintype State]
    (prior : State → ℝ)
    (senderIndirectValue : (State → ℝ) → ℝ)
    (opt : ℝ) : Prop where
  upper_bound :
    ∀ {J : Type} [Fintype J],
      ∀ (weight : J → ℝ) (posterior : J → State → ℝ),
        PersuasionSchemeFeasible prior weight posterior →
          PersuasionSchemeValue weight posterior senderIndirectValue ≤ opt
  achieved :
    ∃ (J : Type) (_ : Fintype J),
      ∃ (weight : J → ℝ) (posterior : J → State → ℝ),
        PersuasionSchemeFeasible prior weight posterior ∧
          PersuasionSchemeValue weight posterior senderIndirectValue = opt

/-- Optimal persuasion value predicate.  Kept as a separate public name because
papers usually state the result as an optimal-value theorem and then identify
that value with the concavification. -/
def IsOptimalPersuasionValue
    [Fintype State]
    (prior : State → ℝ)
    (senderIndirectValue : (State → ℝ) → ℝ)
    (opt : ℝ) : Prop :=
  ConcavificationWitness prior senderIndirectValue opt

/-- Symbolic concavification theorem surface: once a finite concavification
witness is supplied, it is exactly an optimal persuasion value witness. -/
theorem concavificationWitness_iff_optimalPersuasionValue
    [Fintype State]
    (prior : State → ℝ)
    (senderIndirectValue : (State → ℝ) → ℝ)
    (opt : ℝ) :
    ConcavificationWitness prior senderIndirectValue opt ↔
      IsOptimalPersuasionValue prior senderIndirectValue opt := by
  rfl

end FormalProofs.OPT

/-! ## From FormalProofs/OPT/BayesianPersuasionEconomics.lean (consolidated 2026-07-02) -/

section

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

end -- noncomputable section (Economics chunk)
end -- anonymous section (Economics chunk)

/-! ## From FormalProofs/OPT/BayesianPersuasionDirect.lean (consolidated 2026-07-02) -/

section

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

end -- noncomputable section (Direct chunk)
end -- anonymous section (Direct chunk)
