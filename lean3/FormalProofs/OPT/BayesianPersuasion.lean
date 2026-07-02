import Mathlib.Algebra.BigOperators.Ring.Finset
import FormalProofs.OPT.FiniteBayesOnState

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
