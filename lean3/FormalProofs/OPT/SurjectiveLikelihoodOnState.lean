import FormalProofs.OPT.LikelihoodOnStateSufficiency

/-!
# FormalProofs/OPT/SurjectiveLikelihoodOnState.lean

Deterministic SSNL/surjector-style state factorization.

`LikelihoodOnStateSufficiency.lean` proves that a likelihood already written as
`ell theta (state x)` is sufficient with respect to `state`. This file proves
the converse under a surjective state map: if the likelihood family is constant
on state fibers, then it factors through a state-space likelihood head.

The word "surjective" here is purely set-theoretic. This file does not
formalize normalizing-flow density corrections, Jacobians, estimator
consistency, MCMC/VB semantics, posterior calibration, or package internals.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X State Θ Y : Type*}

/-- A state map whose image covers the whole state space. This is the
set-theoretic fragment of a surjective state / surjector claim. -/
def SurjectiveStateMap (state : X → State) : Prop :=
  Function.Surjective state

/-- A likelihood family factors through a state if a state-space likelihood head
realizes all likelihood values from `state x`. -/
def LikelihoodFactorsThroughState
    (state : X → State)
    (likelihood : Θ → X → Y) : Prop :=
  ∃ stateLikelihood : Θ → State → Y,
    LikelihoodReadoutRealizes state likelihood stateLikelihood

/-- Exact factorization through a surjective state. If likelihood values are
constant on state fibers, the state-space likelihood head can be chosen by any
preimage of the state. -/
theorem surjectiveState_likelihood_factorization
    {state : X → State}
    {likelihood : Θ → X → Y}
    (hSurj : SurjectiveStateMap state)
    (hSuff : LikelihoodFamilySufficient state likelihood) :
    LikelihoodFactorsThroughState state likelihood := by
  let stateLikelihood : Θ → State → Y :=
    fun θ z => likelihood θ (Classical.choose (hSurj z))
  refine ⟨stateLikelihood, ?_⟩
  intro θ x
  dsimp [stateLikelihood]
  have hFiber : state (Classical.choose (hSurj (state x))) = state x :=
    Classical.choose_spec (hSurj (state x))
  exact hSuff hFiber θ

/-- For a surjective state map, likelihood-family sufficiency is equivalent to
factorization through a state-space likelihood head. -/
theorem surjectiveState_likelihoodSufficient_iff_factors
    {state : X → State}
    {likelihood : Θ → X → Y}
    (hSurj : SurjectiveStateMap state) :
    LikelihoodFamilySufficient state likelihood ↔
      LikelihoodFactorsThroughState state likelihood := by
  constructor
  · intro hSuff
    exact surjectiveState_likelihood_factorization hSurj hSuff
  · rintro ⟨stateLikelihood, hReadout⟩ x y hxy θ
    calc
      likelihood θ x = stateLikelihood θ (state x) := (hReadout θ x).symm
      _ = stateLikelihood θ (state y) := by rw [hxy]
      _ = likelihood θ y := hReadout θ y

section Approximate

variable [PseudoMetricSpace Y]

/-- Approximate factorization through a surjective state. If likelihood values
are within `ε` on every state fiber, choosing one preimage per state gives a
state-space likelihood readout within `ε` on the whole image. -/
theorem surjectiveState_likelihoodReadoutWithin
    {ε : ℝ}
    {state : X → State}
    {likelihood : Θ → X → Y}
    (hSurj : SurjectiveStateMap state)
    (hWithin : LikelihoodFamilySufficientWithin ε state likelihood) :
    ∃ stateLikelihood : Θ → State → Y,
      LikelihoodReadoutRealizesWithin ε state likelihood stateLikelihood := by
  let stateLikelihood : Θ → State → Y :=
    fun θ z => likelihood θ (Classical.choose (hSurj z))
  refine ⟨stateLikelihood, ?_⟩
  intro θ x
  dsimp [stateLikelihood]
  have hFiber : state (Classical.choose (hSurj (state x))) = state x :=
    Classical.choose_spec (hSurj (state x))
  exact hWithin hFiber θ

end Approximate

end FormalProofs.OPT
