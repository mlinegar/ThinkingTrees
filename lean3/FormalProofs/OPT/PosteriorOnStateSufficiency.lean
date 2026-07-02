import FormalProofs.OPT.SurjectiveLikelihoodOnState

/-!
# FormalProofs/OPT/PosteriorOnStateSufficiency.lean

Deterministic posterior/readout-on-state surface for SBI-style workflows.

Many SBI papers train a summary/state first and then train a posterior,
likelihood, ratio, or diagnostic readout on top of that state.  This file only
formalizes the deterministic readout/factorization part:

* if a posterior-like object is evaluated only through `state x`, then `state`
  is sufficient for that object;
* if a fixed prior/model makes the posterior object a function of the full
  likelihood family, likelihood-family sufficiency transports to posterior
  sufficiency; and
* under a surjective state map, posterior sufficiency is equivalent to
  factorization through a state-space posterior readout.

No Bayes theorem, posterior calibration, MCMC/VB semantics, coverage guarantee,
or estimator consistency is claimed here.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X Rep State Θ Like Posterior : Type*}

/-- Posterior/readout sufficiency is target sufficiency for a posterior-like
object.  `Posterior` can be a symbolic distribution, posterior moment, posterior
interval, ratio response, or any other readout target. -/
def PosteriorSufficient
    (rep : X → Rep)
    (posterior : X → Posterior) : Prop :=
  TargetSufficientRepresentation rep posterior

/-- A posterior-like object has a readout from the representation. -/
def PosteriorReadoutRealizes
    (rep : X → Rep)
    (posterior : X → Posterior)
    (readout : Rep → Posterior) : Prop :=
  TargetReadoutRealizes rep posterior readout

/-- A posterior-like object evaluated through a learned state. -/
def PosteriorOnState
    (state : X → State)
    (statePosterior : State → Posterior) :
    X → Posterior :=
  fun x => statePosterior (state x)

/-- A posterior-like object factors through a state if a state-space readout
realizes it on every observation. -/
def PosteriorFactorsThroughState
    (state : X → State)
    (posterior : X → Posterior) : Prop :=
  ∃ statePosterior : State → Posterior,
    PosteriorReadoutRealizes state posterior statePosterior

/-- A posterior-like object produced from `state x` is sufficient with respect
to that state. This is the deterministic posterior-on-state bridge. -/
theorem posteriorOnState_sufficient
    (state : X → State)
    (statePosterior : State → Posterior) :
    PosteriorSufficient state (PosteriorOnState state statePosterior) := by
  intro x y hxy
  simp [PosteriorOnState, hxy]

/-- If a richer representation can decode the frozen state, then it is
sufficient for any posterior-like object evaluated through that state. -/
theorem repWithStateReadout_posteriorOnState_sufficient
    {rep : X → Rep}
    {state : X → State}
    {decodeState : Rep → State}
    (hState : TargetReadoutRealizes rep state decodeState)
    (statePosterior : State → Posterior) :
    PosteriorSufficient rep (PosteriorOnState state statePosterior) := by
  intro x y hxy
  simp [PosteriorOnState, ← hState x, ← hState y, hxy]

/-- A posterior-like object is determined by a likelihood family when equality
of all likelihood values forces equality of the posterior object. This keeps the
Bayesian step as an explicit deterministic assumption. -/
def PosteriorDeterminedByLikelihood
    (likelihood : Θ → X → Like)
    (posterior : X → Posterior) : Prop :=
  ∀ ⦃x y : X⦄, (∀ θ : Θ, likelihood θ x = likelihood θ y) →
    posterior x = posterior y

/-- Likelihood-family sufficiency transports to posterior/readout sufficiency
whenever the posterior-like object is determined by that likelihood family. -/
theorem likelihoodSufficient_implies_posteriorSufficient
    {rep : X → Rep}
    {likelihood : Θ → X → Like}
    {posterior : X → Posterior}
    (hLike : LikelihoodFamilySufficient rep likelihood)
    (hPosterior : PosteriorDeterminedByLikelihood likelihood posterior) :
    PosteriorSufficient rep posterior := by
  intro x y hxy
  exact hPosterior (hLike hxy)

/-- If a posterior-like object is determined by a state-likelihood family, then
the state is posterior-sufficient. -/
theorem likelihoodOnState_implies_posteriorSufficient
    {state : X → State}
    {stateLikelihood : Θ → State → Like}
    {posterior : X → Posterior}
    (hPosterior :
      PosteriorDeterminedByLikelihood
        (LikelihoodOnStateFamily state stateLikelihood)
        posterior) :
    PosteriorSufficient state posterior :=
  likelihoodSufficient_implies_posteriorSufficient
    (likelihoodOnState_family_sufficient state stateLikelihood)
    hPosterior

/-- Exact factorization of a posterior-like object through a surjective state.
If posterior values are constant on state fibers, a state-space readout can be
chosen by any preimage of each state. -/
theorem surjectiveState_posterior_factorization
    {state : X → State}
    {posterior : X → Posterior}
    (hSurj : SurjectiveStateMap state)
    (hSuff : PosteriorSufficient state posterior) :
    PosteriorFactorsThroughState state posterior := by
  let statePosterior : State → Posterior :=
    fun z => posterior (Classical.choose (hSurj z))
  refine ⟨statePosterior, ?_⟩
  intro x
  dsimp [PosteriorReadoutRealizes, TargetReadoutRealizes, statePosterior]
  have hFiber : state (Classical.choose (hSurj (state x))) = state x :=
    Classical.choose_spec (hSurj (state x))
  exact hSuff hFiber

/-- For a surjective state map, posterior sufficiency is equivalent to
factorization through a state-space posterior readout. -/
theorem surjectiveState_posteriorSufficient_iff_factors
    {state : X → State}
    {posterior : X → Posterior}
    (hSurj : SurjectiveStateMap state) :
    PosteriorSufficient state posterior ↔
      PosteriorFactorsThroughState state posterior := by
  constructor
  · intro hSuff
    exact surjectiveState_posterior_factorization hSurj hSuff
  · rintro ⟨statePosterior, hReadout⟩ x y hxy
    calc
      posterior x = statePosterior (state x) := (hReadout x).symm
      _ = statePosterior (state y) := by rw [hxy]
      _ = posterior y := hReadout y

section Approximate

variable [PseudoMetricSpace Posterior]

/-- Approximate posterior/readout sufficiency: representation collisions
preserve the posterior-like object up to metric slack. -/
def PosteriorSufficientWithin
    (ε : ℝ)
    (rep : X → Rep)
    (posterior : X → Posterior) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → dist (posterior x) (posterior y) ≤ ε

/-- Approximate posterior readout from a representation. -/
def PosteriorReadoutRealizesWithin
    (ε : ℝ)
    (rep : X → Rep)
    (posterior : X → Posterior)
    (readout : Rep → Posterior) : Prop :=
  ∀ x, dist (readout (rep x)) (posterior x) ≤ ε

/-- An approximate posterior readout implies approximate posterior sufficiency,
paying the readout error on both collapsed inputs. -/
theorem posteriorReadoutWithin_implies_posteriorSufficientWithin
    {ε : ℝ}
    {rep : X → Rep}
    {posterior : X → Posterior}
    {readout : Rep → Posterior}
    (hReadout : PosteriorReadoutRealizesWithin ε rep posterior readout) :
    PosteriorSufficientWithin (ε + ε) rep posterior := by
  intro x y hxy
  have hLeft : dist (posterior x) (readout (rep x)) ≤ ε := by
    simpa [dist_comm] using hReadout x
  have hRight : dist (readout (rep x)) (posterior y) ≤ ε := by
    rw [hxy]
    exact hReadout y
  calc
    dist (posterior x) (posterior y)
        ≤ dist (posterior x) (readout (rep x)) +
            dist (readout (rep x)) (posterior y) := by
          exact dist_triangle _ _ _
    _ ≤ ε + ε := add_le_add hLeft hRight

/-- Approximate factorization through a surjective state. If posterior values
are within `ε` on state fibers, choosing one preimage per state gives a
state-space posterior readout within `ε` on the state image. -/
theorem surjectiveState_posteriorReadoutWithin
    {ε : ℝ}
    {state : X → State}
    {posterior : X → Posterior}
    (hSurj : SurjectiveStateMap state)
    (hWithin : PosteriorSufficientWithin ε state posterior) :
    ∃ statePosterior : State → Posterior,
      PosteriorReadoutRealizesWithin ε state posterior statePosterior := by
  let statePosterior : State → Posterior :=
    fun z => posterior (Classical.choose (hSurj z))
  refine ⟨statePosterior, ?_⟩
  intro x
  dsimp [PosteriorReadoutRealizesWithin, statePosterior]
  have hFiber : state (Classical.choose (hSurj (state x))) = state x :=
    Classical.choose_spec (hSurj (state x))
  exact hWithin hFiber

end Approximate

end FormalProofs.OPT
