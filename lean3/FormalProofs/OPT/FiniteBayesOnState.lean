import Mathlib.Algebra.BigOperators.Ring.Finset
import FormalProofs.OPT.PosteriorOnStateSufficiency

/-!
# FormalProofs/OPT/FiniteBayesOnState.lean

Finite/discrete Bayes semantics for posterior-on-state claims.

This is the bounded Bayes layer that fits the current sufficient-statistics
formalization.  Parameters live in a finite type, priors and likelihoods are
ordinary real-valued functions, and the posterior is the normalized Bayes
numerator

`prior θ * likelihood θ x / evidence x`.

The file proves algebraic/fiber facts only:

* the posterior is determined by the likelihood family for a fixed prior;
* likelihood-family sufficiency implies posterior sufficiency;
* likelihood-on-state families induce posterior-on-state readouts; and
* under a surjective state, Bayes posteriors factor through a state readout.

This is not a theorem about calibrated posterior inference, posterior
consistency, MCMC/VB, density estimation, or a dominated-measure Bayes theorem.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X Rep State Θ : Type*} [Fintype Θ]

/-- Bayes numerator for a fixed prior and likelihood family. -/
def BayesNumerator
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (θ : Θ)
    (x : X) : ℝ :=
  prior θ * likelihood θ x

/-- Finite evidence / marginal likelihood. -/
def BayesEvidence
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X) : ℝ :=
  ∑ θ : Θ, BayesNumerator prior likelihood θ x

/-- Finite Bayes posterior as a real-valued parameter-indexed function.  When
the evidence is zero, Lean's real division gives a value, but normalization
theorems below explicitly assume nonzero evidence. -/
def BayesPosterior
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X) :
    Θ → ℝ :=
  fun θ => BayesNumerator prior likelihood θ x /
    BayesEvidence prior likelihood x

/-- State-space Bayes numerator for a likelihood head on state. -/
def StateBayesNumerator
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (θ : Θ)
    (z : State) : ℝ :=
  prior θ * stateLikelihood θ z

/-- State-space finite evidence / marginal likelihood. -/
def StateBayesEvidence
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State) : ℝ :=
  ∑ θ : Θ, StateBayesNumerator prior stateLikelihood θ z

/-- State-space finite Bayes posterior. -/
def StateBayesPosterior
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State) :
    Θ → ℝ :=
  fun θ => StateBayesNumerator prior stateLikelihood θ z /
    StateBayesEvidence prior stateLikelihood z

/-- Bayes posteriors normalize to one when the evidence is nonzero. -/
theorem bayesPosterior_sum_eq_one
    {prior : Θ → ℝ}
    {likelihood : Θ → X → ℝ}
    {x : X}
    (hEvidence : BayesEvidence prior likelihood x ≠ 0) :
    (∑ θ : Θ, BayesPosterior prior likelihood x θ) = 1 := by
  let evidence := BayesEvidence prior likelihood x
  have hEvidence' : evidence ≠ 0 := by
    simpa [evidence] using hEvidence
  calc
    (∑ θ : Θ, BayesPosterior prior likelihood x θ)
        = ∑ θ : Θ, evidence⁻¹ * BayesNumerator prior likelihood θ x := by
          simp [BayesPosterior, evidence, div_eq_mul_inv, mul_comm]
    _ = evidence⁻¹ * (∑ θ : Θ, BayesNumerator prior likelihood θ x) := by
          rw [Finset.mul_sum]
    _ = evidence⁻¹ * evidence := by
          simp [evidence, BayesEvidence]
    _ = 1 := by
          field_simp [hEvidence']

/-- State-space Bayes posteriors normalize to one when the state evidence is
nonzero. -/
theorem stateBayesPosterior_sum_eq_one
    {prior : Θ → ℝ}
    {stateLikelihood : Θ → State → ℝ}
    {z : State}
    (hEvidence : StateBayesEvidence prior stateLikelihood z ≠ 0) :
    (∑ θ : Θ, StateBayesPosterior prior stateLikelihood z θ) = 1 := by
  let evidence := StateBayesEvidence prior stateLikelihood z
  have hEvidence' : evidence ≠ 0 := by
    simpa [evidence] using hEvidence
  calc
    (∑ θ : Θ, StateBayesPosterior prior stateLikelihood z θ)
        = ∑ θ : Θ, evidence⁻¹ * StateBayesNumerator prior stateLikelihood θ z := by
          simp [StateBayesPosterior, evidence, div_eq_mul_inv, mul_comm]
    _ = evidence⁻¹ *
          (∑ θ : Θ, StateBayesNumerator prior stateLikelihood θ z) := by
          rw [Finset.mul_sum]
    _ = evidence⁻¹ * evidence := by
          simp [evidence, StateBayesEvidence]
    _ = 1 := by
          field_simp [hEvidence']

/-! ## Finite Bayes decision/readout semantics -/

/-- A finite maximizer predicate for real-valued parameter scores.  This is the
finite Bayes/MAP analogue of the symbolic argmax vocabulary used by the
information-objective modules, kept local to avoid importing the MI layer into
the Bayes algebra. -/
def IsFiniteScoreMAP
    (score : Θ → ℝ)
    (θhat : Θ) : Prop :=
  ∀ θ : Θ, score θ ≤ score θhat

/-- A finite minimizer predicate for real-valued action scores. -/
def IsFiniteScoreArgmin
    {Action : Type*}
    (score : Action → ℝ)
    (ahat : Action) : Prop :=
  ∀ a : Action, score ahat ≤ score a

/-- MAP for the unnormalized Bayes numerator. -/
def BayesNumeratorMAP
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (θhat : Θ) : Prop :=
  IsFiniteScoreMAP (fun θ => BayesNumerator prior likelihood θ x) θhat

/-- MAP for the normalized finite Bayes posterior. -/
def BayesPosteriorMAP
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (θhat : Θ) : Prop :=
  IsFiniteScoreMAP (BayesPosterior prior likelihood x) θhat

/-- MAP for the unnormalized state-space Bayes numerator. -/
def StateBayesNumeratorMAP
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (θhat : Θ) : Prop :=
  IsFiniteScoreMAP (fun θ => StateBayesNumerator prior stateLikelihood θ z) θhat

/-- MAP for the normalized state-space finite Bayes posterior. -/
def StateBayesPosteriorMAP
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (θhat : Θ) : Prop :=
  IsFiniteScoreMAP (StateBayesPosterior prior stateLikelihood z) θhat

/-- MAP decisions are unchanged by normalizing Bayes numerators by a positive
evidence term. -/
theorem bayesPosteriorMAP_iff_bayesNumeratorMAP
    {prior : Θ → ℝ}
    {likelihood : Θ → X → ℝ}
    {x : X}
    {θhat : Θ}
    (hEvidence : 0 < BayesEvidence prior likelihood x) :
    BayesPosteriorMAP prior likelihood x θhat ↔
      BayesNumeratorMAP prior likelihood x θhat := by
  constructor
  · intro hPost θ
    unfold BayesPosteriorMAP IsFiniteScoreMAP BayesPosterior at hPost
    have hDiv := hPost θ
    have hMul := mul_le_mul_of_nonneg_right hDiv (le_of_lt hEvidence)
    field_simp [ne_of_gt hEvidence] at hMul
    exact hMul
  · intro hNum θ
    unfold BayesNumeratorMAP IsFiniteScoreMAP at hNum
    unfold BayesPosterior
    have h := hNum θ
    have hScaled := mul_le_mul_of_nonneg_right
      h
      (inv_nonneg.mpr (le_of_lt hEvidence))
    simpa [div_eq_mul_inv] using hScaled

/-- State-space MAP decisions are unchanged by normalizing state Bayes
numerators by a positive state evidence term. -/
theorem stateBayesPosteriorMAP_iff_stateBayesNumeratorMAP
    {prior : Θ → ℝ}
    {stateLikelihood : Θ → State → ℝ}
    {z : State}
    {θhat : Θ}
    (hEvidence : 0 < StateBayesEvidence prior stateLikelihood z) :
    StateBayesPosteriorMAP prior stateLikelihood z θhat ↔
      StateBayesNumeratorMAP prior stateLikelihood z θhat := by
  constructor
  · intro hPost θ
    unfold StateBayesPosteriorMAP IsFiniteScoreMAP StateBayesPosterior at hPost
    have hDiv := hPost θ
    have hMul := mul_le_mul_of_nonneg_right hDiv (le_of_lt hEvidence)
    field_simp [ne_of_gt hEvidence] at hMul
    exact hMul
  · intro hNum θ
    unfold StateBayesNumeratorMAP IsFiniteScoreMAP at hNum
    unfold StateBayesPosterior
    have h := hNum θ
    have hScaled := mul_le_mul_of_nonneg_right
      h
      (inv_nonneg.mpr (le_of_lt hEvidence))
    simpa [div_eq_mul_inv] using hScaled

/-- Posterior odds equal Bayes-numerator odds when the evidence is nonzero. -/
theorem bayesPosterior_odds_eq_bayesNumerator_odds
    {prior : Θ → ℝ}
    {likelihood : Θ → X → ℝ}
    {x : X}
    {θ θ0 : Θ}
    (hEvidence : BayesEvidence prior likelihood x ≠ 0) :
    BayesPosterior prior likelihood x θ /
        BayesPosterior prior likelihood x θ0 =
      BayesNumerator prior likelihood θ x /
        BayesNumerator prior likelihood θ0 x := by
  unfold BayesPosterior
  by_cases hDen : BayesNumerator prior likelihood θ0 x = 0
  · simp [hDen]
  · field_simp [hEvidence, hDen]

/-- State posterior odds equal state Bayes-numerator odds when the state
evidence is nonzero. -/
theorem stateBayesPosterior_odds_eq_stateBayesNumerator_odds
    {prior : Θ → ℝ}
    {stateLikelihood : Θ → State → ℝ}
    {z : State}
    {θ θ0 : Θ}
    (hEvidence : StateBayesEvidence prior stateLikelihood z ≠ 0) :
    StateBayesPosterior prior stateLikelihood z θ /
        StateBayesPosterior prior stateLikelihood z θ0 =
      StateBayesNumerator prior stateLikelihood θ z /
        StateBayesNumerator prior stateLikelihood θ0 z := by
  unfold StateBayesPosterior
  by_cases hDen : StateBayesNumerator prior stateLikelihood θ0 z = 0
  · simp [hDen]
  · field_simp [hEvidence, hDen]

/-- A finite posterior expectation/readout under the Bayes posterior.  This
covers posterior means and other finite-parameter posterior functionals. -/
def BayesPosteriorExpectation
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (stat : Θ → ℝ) : ℝ :=
  ∑ θ : Θ, BayesPosterior prior likelihood x θ * stat θ

/-- State-space posterior expectation/readout under the state Bayes posterior. -/
def StateBayesPosteriorExpectation
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (stat : Θ → ℝ) : ℝ :=
  ∑ θ : Θ, StateBayesPosterior prior stateLikelihood z θ * stat θ

/-- Posterior expectations for likelihood-on-state families are exactly the
corresponding state posterior expectations. -/
theorem bayesPosteriorExpectation_likelihoodOnState_eq_state
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X)
    (stat : Θ → ℝ) :
    BayesPosteriorExpectation
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        stat =
      StateBayesPosteriorExpectation
        prior
        stateLikelihood
        (state x)
        stat := by
  unfold BayesPosteriorExpectation StateBayesPosteriorExpectation
  refine Finset.sum_congr rfl ?_
  intro θ _
  simp [BayesPosterior, StateBayesPosterior, BayesNumerator,
    StateBayesNumerator, BayesEvidence, StateBayesEvidence,
    LikelihoodOnStateFamily]

/-- If the likelihood factors through state, every finite posterior
expectation/readout is state-sufficient. -/
theorem bayesPosteriorExpectation_likelihoodOnState_sufficient
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (stat : Θ → ℝ) :
    PosteriorSufficient
      state
      (fun x =>
        BayesPosteriorExpectation
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          x
          stat) := by
  intro x y hxy
  change
    BayesPosteriorExpectation
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        stat =
      BayesPosteriorExpectation
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        y
        stat
  rw [bayesPosteriorExpectation_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := x)
      (stat := stat),
    bayesPosteriorExpectation_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := y)
      (stat := stat),
    hxy]

/-- Finite posterior predictive likelihood for a future observation. -/
def BayesPosteriorPredictive
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (observed : X)
    (future : X) : ℝ :=
  BayesPosteriorExpectation
    prior
    likelihood
    observed
    (fun θ => likelihood θ future)

/-- State-space finite posterior predictive likelihood for a future state. -/
def StateBayesPosteriorPredictive
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (observed : State)
    (future : State) : ℝ :=
  StateBayesPosteriorExpectation
    prior
    stateLikelihood
    observed
    (fun θ => stateLikelihood θ future)

/-- Posterior predictive likelihoods for likelihood-on-state families are
exactly the corresponding state posterior predictive likelihoods. -/
theorem bayesPosteriorPredictive_likelihoodOnState_eq_state
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (observed future : X) :
    BayesPosteriorPredictive
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        observed
        future =
      StateBayesPosteriorPredictive
        prior
        stateLikelihood
        (state observed)
        (state future) := by
  unfold BayesPosteriorPredictive StateBayesPosteriorPredictive
  exact
    bayesPosteriorExpectation_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := observed)
      (stat := fun θ => stateLikelihood θ (state future))

/-- For any fixed future observation, the finite posterior predictive under a
likelihood-on-state family is sufficient in the observed learned state. -/
theorem bayesPosteriorPredictive_likelihoodOnState_sufficient_observed
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (future : X) :
    PosteriorSufficient
      state
      (fun observed =>
        BayesPosteriorPredictive
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          observed
          future) := by
  intro x y hxy
  change
    BayesPosteriorPredictive
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        future =
      BayesPosteriorPredictive
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        y
        future
  rw [bayesPosteriorPredictive_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (observed := x)
      (future := future),
    bayesPosteriorPredictive_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (observed := y)
      (future := future),
    hxy]

/-! ## Bayes risk, Bayes actions, and credible sets -/

/-- Posterior Bayes risk of an action under a finite posterior. -/
def BayesPosteriorRisk
    {Action : Type*}
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (loss : Action → Θ → ℝ)
    (action : Action) : ℝ :=
  BayesPosteriorExpectation
    prior
    likelihood
    x
    (fun θ => loss action θ)

/-- State-space posterior Bayes risk of an action. -/
def StateBayesPosteriorRisk
    {Action : Type*}
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (loss : Action → Θ → ℝ)
    (action : Action) : ℝ :=
  StateBayesPosteriorExpectation
    prior
    stateLikelihood
    z
    (fun θ => loss action θ)

/-- A finite Bayes action minimizes posterior Bayes risk. -/
def BayesAction
    {Action : Type*}
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (loss : Action → Θ → ℝ)
    (action : Action) : Prop :=
  IsFiniteScoreArgmin
    (fun a => BayesPosteriorRisk prior likelihood x loss a)
    action

/-- A state-space finite Bayes action minimizes state posterior Bayes risk. -/
def StateBayesAction
    {Action : Type*}
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (loss : Action → Θ → ℝ)
    (action : Action) : Prop :=
  IsFiniteScoreArgmin
    (fun a => StateBayesPosteriorRisk prior stateLikelihood z loss a)
    action

/-- Bayes risks for likelihood-on-state families are exactly the corresponding
state posterior risks. -/
theorem bayesPosteriorRisk_likelihoodOnState_eq_state
    {Action : Type*}
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X)
    (loss : Action → Θ → ℝ)
    (action : Action) :
    BayesPosteriorRisk
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        loss
        action =
      StateBayesPosteriorRisk
        prior
        stateLikelihood
        (state x)
        loss
        action :=
  bayesPosteriorExpectation_likelihoodOnState_eq_state
    (prior := prior)
    (state := state)
    (stateLikelihood := stateLikelihood)
    (x := x)
    (stat := fun θ => loss action θ)

/-- For any fixed action, likelihood-on-state Bayes risk is sufficient in the
observed learned state. -/
theorem bayesPosteriorRisk_likelihoodOnState_sufficient
    {Action : Type*}
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (loss : Action → Θ → ℝ)
    (action : Action) :
    PosteriorSufficient
      state
      (fun x =>
        BayesPosteriorRisk
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          x
          loss
          action) := by
  intro x y hxy
  change
    BayesPosteriorRisk
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        loss
        action =
      BayesPosteriorRisk
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        y
        loss
        action
  rw [bayesPosteriorRisk_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := x)
      (loss := loss)
      (action := action),
    bayesPosteriorRisk_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := y)
      (loss := loss)
      (action := action),
    hxy]

/-- Bayes-action optimality transports exactly across likelihood-on-state
factorization. -/
theorem bayesAction_likelihoodOnState_iff_stateBayesAction
    {Action : Type*}
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X)
    (loss : Action → Θ → ℝ)
    (action : Action) :
    BayesAction
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        loss
        action
      ↔
      StateBayesAction
        prior
        stateLikelihood
        (state x)
        loss
        action := by
  constructor
  · intro h a
    calc
      StateBayesPosteriorRisk prior stateLikelihood (state x) loss action
          =
            BayesPosteriorRisk
              prior
              (LikelihoodOnStateFamily state stateLikelihood)
              x
              loss
              action := by
            rw [bayesPosteriorRisk_likelihoodOnState_eq_state
              (prior := prior)
              (state := state)
              (stateLikelihood := stateLikelihood)
              (x := x)
              (loss := loss)
              (action := action)]
      _ ≤
            BayesPosteriorRisk
              prior
              (LikelihoodOnStateFamily state stateLikelihood)
              x
              loss
              a := h a
      _ =
          StateBayesPosteriorRisk prior stateLikelihood (state x) loss a := by
            rw [bayesPosteriorRisk_likelihoodOnState_eq_state
              (prior := prior)
              (state := state)
              (stateLikelihood := stateLikelihood)
              (x := x)
              (loss := loss)
              (action := a)]
  · intro h a
    calc
      BayesPosteriorRisk
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          x
          loss
          action =
          StateBayesPosteriorRisk prior stateLikelihood (state x) loss action := by
            rw [bayesPosteriorRisk_likelihoodOnState_eq_state
              (prior := prior)
              (state := state)
              (stateLikelihood := stateLikelihood)
              (x := x)
              (loss := loss)
              (action := action)]
      _ ≤ StateBayesPosteriorRisk prior stateLikelihood (state x) loss a := h a
      _ =
          BayesPosteriorRisk
            prior
            (LikelihoodOnStateFamily state stateLikelihood)
            x
            loss
            a := by
            rw [bayesPosteriorRisk_likelihoodOnState_eq_state
              (prior := prior)
              (state := state)
              (stateLikelihood := stateLikelihood)
              (x := x)
              (loss := loss)
              (action := a)]

/-- Posterior mass assigned to an event in the finite parameter space. -/
def BayesPosteriorSetMass
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (event : Set Θ) : ℝ :=
  ∑ θ : Θ, event.indicator (BayesPosterior prior likelihood x) θ

/-- State-space posterior mass assigned to an event in the finite parameter
space. -/
def StateBayesPosteriorSetMass
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (event : Set Θ) : ℝ :=
  ∑ θ : Θ, event.indicator (StateBayesPosterior prior stateLikelihood z) θ

/-- A finite credible/acceptance set predicate at a requested posterior mass
level. -/
def BayesCredibleAtLevel
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (event : Set Θ)
    (level : ℝ) : Prop :=
  level ≤ BayesPosteriorSetMass prior likelihood x event

/-- State-space finite credible/acceptance set predicate. -/
def StateBayesCredibleAtLevel
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (event : Set Θ)
    (level : ℝ) : Prop :=
  level ≤ StateBayesPosteriorSetMass prior stateLikelihood z event

/-- Posterior event masses for likelihood-on-state families are exactly the
corresponding state posterior event masses. -/
theorem bayesPosteriorSetMass_likelihoodOnState_eq_state
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X)
    (event : Set Θ) :
    BayesPosteriorSetMass
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        event =
      StateBayesPosteriorSetMass
        prior
        stateLikelihood
        (state x)
        event := by
  unfold BayesPosteriorSetMass StateBayesPosteriorSetMass
  refine Finset.sum_congr rfl ?_
  intro θ _
  by_cases hθ : θ ∈ event <;>
    simp [Set.indicator, hθ, BayesPosterior, StateBayesPosterior,
      BayesNumerator, StateBayesNumerator, BayesEvidence,
      StateBayesEvidence, LikelihoodOnStateFamily]

/-- Likelihood-on-state credible/acceptance-set claims are equivalent to the
state-space claim. -/
theorem bayesCredibleAtLevel_likelihoodOnState_iff_state
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X)
    (event : Set Θ)
    (level : ℝ) :
    BayesCredibleAtLevel
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        event
        level
      ↔
      StateBayesCredibleAtLevel
        prior
        stateLikelihood
        (state x)
        event
        level := by
  unfold BayesCredibleAtLevel StateBayesCredibleAtLevel
  rw [bayesPosteriorSetMass_likelihoodOnState_eq_state
    (prior := prior)
    (state := state)
    (stateLikelihood := stateLikelihood)
    (x := x)
    (event := event)]

/-- For any fixed parameter event, likelihood-on-state posterior event mass is
sufficient in the observed learned state. -/
theorem bayesPosteriorSetMass_likelihoodOnState_sufficient
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (event : Set Θ) :
    PosteriorSufficient
      state
      (fun x =>
        BayesPosteriorSetMass
          prior
          (LikelihoodOnStateFamily state stateLikelihood)
          x
          event) := by
  intro x y hxy
  change
    BayesPosteriorSetMass
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        x
        event =
      BayesPosteriorSetMass
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
        y
        event
  rw [bayesPosteriorSetMass_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := x)
      (event := event),
    bayesPosteriorSetMass_likelihoodOnState_eq_state
      (prior := prior)
      (state := state)
      (stateLikelihood := stateLikelihood)
      (x := y)
      (event := event),
    hxy]

/-! ## Evidence-ratio concentration algebra -/

/-- Evidence-to-target-numerator remainder.  If this remainder is close to
zero, then target posterior mass is close to one through the transform
`r ↦ (1 + r)⁻¹`. -/
def BayesEvidenceRatioRemainder
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (θ0 : Θ) : ℝ :=
  BayesEvidence prior likelihood x /
    BayesNumerator prior likelihood θ0 x - 1

/-- State-space evidence-to-target-numerator remainder. -/
def StateBayesEvidenceRatioRemainder
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (θ0 : Θ) : ℝ :=
  StateBayesEvidence prior stateLikelihood z /
    StateBayesNumerator prior stateLikelihood θ0 z - 1

/-- Target posterior mass is the inverse one-plus evidence-ratio remainder.
This is a deterministic finite Bayes identity; convergence of the remainder is
handled by the posterior-consistency layer as an explicit regularity
assumption. -/
theorem bayesPosterior_target_eq_inv_one_plus_evidenceRatioRemainder
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (x : X)
    (θ0 : Θ) :
    BayesPosterior prior likelihood x θ0 =
      (1 + BayesEvidenceRatioRemainder prior likelihood x θ0)⁻¹ := by
  unfold BayesPosterior BayesEvidenceRatioRemainder
  have h :
      1 + (BayesEvidence prior likelihood x /
          BayesNumerator prior likelihood θ0 x - 1) =
        BayesEvidence prior likelihood x /
          BayesNumerator prior likelihood θ0 x := by
    ring
  rw [h]
  exact (inv_div
    (BayesEvidence prior likelihood x)
    (BayesNumerator prior likelihood θ0 x)).symm

/-- State-space target posterior mass is the inverse one-plus state
evidence-ratio remainder. -/
theorem stateBayesPosterior_target_eq_inv_one_plus_evidenceRatioRemainder
    (prior : Θ → ℝ)
    (stateLikelihood : Θ → State → ℝ)
    (z : State)
    (θ0 : Θ) :
    StateBayesPosterior prior stateLikelihood z θ0 =
      (1 + StateBayesEvidenceRatioRemainder prior stateLikelihood z θ0)⁻¹ := by
  unfold StateBayesPosterior StateBayesEvidenceRatioRemainder
  have h :
      1 + (StateBayesEvidence prior stateLikelihood z /
          StateBayesNumerator prior stateLikelihood θ0 z - 1) =
        StateBayesEvidence prior stateLikelihood z /
          StateBayesNumerator prior stateLikelihood θ0 z := by
    ring
  rw [h]
  exact (inv_div
    (StateBayesEvidence prior stateLikelihood z)
    (StateBayesNumerator prior stateLikelihood θ0 z)).symm

/-- The finite Bayes posterior is determined by the likelihood family for a
fixed prior. -/
theorem bayesPosterior_determinedByLikelihood
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ) :
    PosteriorDeterminedByLikelihood
      likelihood
      (BayesPosterior prior likelihood) := by
  intro x y hLike
  funext θ
  have hNum :
      ∀ θ' : Θ,
        BayesNumerator prior likelihood θ' x =
          BayesNumerator prior likelihood θ' y := by
    intro θ'
    simp [BayesNumerator, hLike θ']
  have hEvidence :
      BayesEvidence prior likelihood x =
        BayesEvidence prior likelihood y := by
    unfold BayesEvidence
    exact Finset.sum_congr rfl (fun θ' _ => hNum θ')
  unfold BayesPosterior
  rw [hNum θ, hEvidence]

/-- A finite posterior expectation is determined by the likelihood family for a
fixed prior and statistic. -/
theorem bayesPosteriorExpectation_determinedByLikelihood
    (prior : Θ → ℝ)
    (likelihood : Θ → X → ℝ)
    (stat : Θ → ℝ) :
    PosteriorDeterminedByLikelihood
      likelihood
      (fun x => BayesPosteriorExpectation prior likelihood x stat) := by
  intro x y hLike
  unfold BayesPosteriorExpectation
  refine Finset.sum_congr rfl ?_
  intro θ _
  rw [bayesPosterior_determinedByLikelihood prior likelihood hLike]

/-- Likelihood-family sufficiency implies finite-Bayes posterior sufficiency for
any fixed prior. -/
theorem likelihoodSufficient_implies_bayesPosteriorSufficient
    {prior : Θ → ℝ}
    {likelihood : Θ → X → ℝ}
    {rep : X → Rep}
    (hLike : LikelihoodFamilySufficient rep likelihood) :
    PosteriorSufficient rep (BayesPosterior prior likelihood) :=
  likelihoodSufficient_implies_posteriorSufficient
    hLike
    (bayesPosterior_determinedByLikelihood prior likelihood)

/-- Evidence for a likelihood-on-state family is the state-space evidence. -/
theorem bayesEvidence_likelihoodOnState_eq_stateBayesEvidence
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ)
    (x : X) :
    BayesEvidence prior (LikelihoodOnStateFamily state stateLikelihood) x =
      StateBayesEvidence prior stateLikelihood (state x) := by
  unfold BayesEvidence StateBayesEvidence BayesNumerator
    StateBayesNumerator LikelihoodOnStateFamily
  rfl

/-- A likelihood-on-state family induces exactly a posterior-on-state readout
with the state-space Bayes posterior. -/
theorem bayesPosterior_likelihoodOnState_eq_posteriorOnState
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ) :
    BayesPosterior
        prior
        (LikelihoodOnStateFamily state stateLikelihood)
      =
        PosteriorOnState
          state
          (StateBayesPosterior prior stateLikelihood) := by
  funext x θ
  simp [BayesPosterior, PosteriorOnState, StateBayesPosterior,
    BayesNumerator, StateBayesNumerator, BayesEvidence, StateBayesEvidence,
    LikelihoodOnStateFamily]

/-- Bayes posterior sufficiency for likelihood-on-state families. -/
theorem bayesPosterior_likelihoodOnState_sufficient
    (prior : Θ → ℝ)
    (state : X → State)
    (stateLikelihood : Θ → State → ℝ) :
    PosteriorSufficient
      state
      (BayesPosterior
        prior
        (LikelihoodOnStateFamily state stateLikelihood)) :=
  likelihoodSufficient_implies_bayesPosteriorSufficient
    (prior := prior)
    (likelihood := LikelihoodOnStateFamily state stateLikelihood)
    (rep := state)
    (likelihoodOnState_family_sufficient state stateLikelihood)

/-- A richer representation with a decoder to the learned state is sufficient
for the finite Bayes posterior induced by a state likelihood. -/
theorem repWithStateReadout_bayesPosterior_likelihoodOnState_sufficient
    {prior : Θ → ℝ}
    {state : X → State}
    {stateLikelihood : Θ → State → ℝ}
    {rep : X → Rep}
    {decodeState : Rep → State}
    (hState : TargetReadoutRealizes rep state decodeState) :
    PosteriorSufficient
      rep
      (BayesPosterior
        prior
        (LikelihoodOnStateFamily state stateLikelihood)) := by
  rw [bayesPosterior_likelihoodOnState_eq_posteriorOnState
    (prior := prior) (state := state) (stateLikelihood := stateLikelihood)]
  exact repWithStateReadout_posteriorOnState_sufficient
    hState
    (StateBayesPosterior prior stateLikelihood)

/-- Under a surjective state map, state-fiber likelihood sufficiency gives a
state-space Bayes posterior readout. -/
theorem surjectiveState_bayesPosterior_factorization
    {prior : Θ → ℝ}
    {state : X → State}
    {likelihood : Θ → X → ℝ}
    (hSurj : SurjectiveStateMap state)
    (hLike : LikelihoodFamilySufficient state likelihood) :
    PosteriorFactorsThroughState state (BayesPosterior prior likelihood) :=
  surjectiveState_posterior_factorization
    hSurj
    (likelihoodSufficient_implies_bayesPosteriorSufficient
      (prior := prior) hLike)

end FormalProofs.OPT
