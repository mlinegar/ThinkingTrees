import FormalProofs.OPT.InformationRepresentationSufficiency

/-!
# FormalProofs/OPT/LikelihoodOnStateSufficiency.lean

Deterministic theorem surface for SSNL/SNLE-style "likelihood on learned
state" claims.

The SSNL paper motivates learning a lower-dimensional state jointly with a
surrogate likelihood. This file does not formalize posterior consistency,
surjective normalizing flows, or density-estimation training. It isolates the
part that belongs in the current Lean lane:

* if a likelihood family is evaluated only through a state `z = state x`, then
  `state` is a likelihood-family sufficient representation;
* any richer representation from which that state can be read out is also
  sufficient for the state-likelihood family;
* approximate likelihood readouts give deterministic approximate likelihood
  sufficiency with metric slack.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X State Rep Θ Y : Type*}

/-! ## Exact likelihood on state -/

/-- A likelihood family obtained by first mapping an observation to a learned
state and then evaluating a state-space likelihood head. -/
def LikelihoodOnStateFamily
    (state : X → State)
    (stateLikelihood : Θ → State → Y) :
    Θ → X → Y :=
  fun θ x => stateLikelihood θ (state x)

/-- The state-space likelihood head is a likelihood readout from the learned
state. -/
theorem likelihoodOnState_readout_realizes
    (state : X → State)
    (stateLikelihood : Θ → State → Y) :
    LikelihoodReadoutRealizes
      state
      (LikelihoodOnStateFamily state stateLikelihood)
      stateLikelihood := by
  intro θ x
  rfl

/-- Any likelihood family evaluated only through a state is sufficient with
respect to that state. This is the deterministic SSNL/SNLE bridge. -/
theorem likelihoodOnState_family_sufficient
    (state : X → State)
    (stateLikelihood : Θ → State → Y) :
    LikelihoodFamilySufficient
      state
      (LikelihoodOnStateFamily state stateLikelihood) := by
  intro x y hxy θ
  simp [LikelihoodOnStateFamily, hxy]

/-- If a richer representation has a decoder to the learned state, then it is
sufficient for every likelihood family evaluated through that state. -/
theorem repWithStateReadout_likelihoodOnState_family_sufficient
    {rep : X → Rep}
    {state : X → State}
    {decodeState : Rep → State}
    (hState : TargetReadoutRealizes rep state decodeState)
    (stateLikelihood : Θ → State → Y) :
    LikelihoodFamilySufficient
      rep
      (LikelihoodOnStateFamily state stateLikelihood) := by
  intro x y hxy θ
  simp [LikelihoodOnStateFamily, ← hState x, ← hState y, hxy]

/-- A state sufficient for a likelihood-on-state family cannot collapse two
documents that the state-likelihood head distinguishes. -/
theorem likelihoodOnState_no_collision_of_likelihood_distinct
    {state : X → State}
    {stateLikelihood : Θ → State → Y}
    {x y : X}
    (hSep :
      ∃ θ : Θ,
        LikelihoodOnStateFamily state stateLikelihood θ x ≠
          LikelihoodOnStateFamily state stateLikelihood θ y) :
    state x ≠ state y := by
  exact
    likelihoodFamilySufficient_no_collision_of_distinguished_likelihood
      (likelihoodOnState_family_sufficient state stateLikelihood)
      hSep

/-! ## Approximate likelihood on state -/

/-- Approximate likelihood-family sufficiency: representation collisions preserve
every likelihood value up to metric slack. -/
def LikelihoodFamilySufficientWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (rep : X → Rep)
    (likelihood : Θ → X → Y) : Prop :=
  QuerySufficientWithin ε rep likelihood

/-- A likelihood readout realizes the likelihood family up to metric slack. -/
def LikelihoodReadoutRealizesWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (rep : X → Rep)
    (likelihood : Θ → X → Y)
    (readout : Θ → Rep → Y) : Prop :=
  ∀ θ x, dist (readout θ (rep x)) (likelihood θ x) ≤ ε

/-- An approximate likelihood readout implies approximate likelihood-family
sufficiency, paying the readout error on both collapsed inputs. -/
theorem likelihoodReadoutWithin_implies_likelihoodFamilySufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {rep : X → Rep}
    {likelihood : Θ → X → Y}
    {readout : Θ → Rep → Y}
    (hReadout : LikelihoodReadoutRealizesWithin ε rep likelihood readout) :
    LikelihoodFamilySufficientWithin (ε + ε) rep likelihood := by
  intro x y hxy θ
  have hLeft : dist (likelihood θ x) (readout θ (rep x)) ≤ ε := by
    simpa [dist_comm] using hReadout θ x
  have hRight : dist (readout θ (rep x)) (likelihood θ y) ≤ ε := by
    rw [hxy]
    exact hReadout θ y
  calc
    dist (likelihood θ x) (likelihood θ y)
        ≤ dist (likelihood θ x) (readout θ (rep x)) +
            dist (readout θ (rep x)) (likelihood θ y) := by
          exact dist_triangle _ _ _
    _ ≤ ε + ε := add_le_add hLeft hRight

/-- Approximate likelihood readout for a state-space likelihood head implies
approximate sufficiency for the induced likelihood-on-state family. -/
theorem stateLikelihoodReadoutWithin_implies_likelihoodOnStateSufficientWithin
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {rep : X → Rep}
    {state : X → State}
    {stateLikelihood : Θ → State → Y}
    {readout : Θ → Rep → Y}
    (hReadout :
      LikelihoodReadoutRealizesWithin
        ε
        rep
        (LikelihoodOnStateFamily state stateLikelihood)
        readout) :
    LikelihoodFamilySufficientWithin
      (ε + ε)
      rep
      (LikelihoodOnStateFamily state stateLikelihood) :=
  likelihoodReadoutWithin_implies_likelihoodFamilySufficientWithin hReadout

end FormalProofs.OPT
