import FormalProofs.OPT.SlicedContextualSufficiency

/-!
# FormalProofs/OPT/InformationRepresentationSufficiency.lean

This module gives a small common vocabulary for sufficient statistics,
representations, and likelihood / likelihood-free targets.

The file is deliberately deterministic. It does not formalize Shannon mutual
information, posterior consistency, PAC generalization, or a simulator law.
Instead, it isolates the information condition shared by those settings:

* a representation `rep x` is sufficient for a target when collisions of
  `rep` cannot change the target;
* equivalently, the target can be read out from the representation;
* a likelihood-model sufficient statistic is the same condition applied to the
  whole likelihood family `θ ↦ likelihood θ x`;
* a likelihood-free sufficient representation is the same condition applied to
  a family of simulator/query/probe responses.

This is the theorem surface that NASS/SSS-style objectives try to learn in
Python, while the probabilistic estimator details remain outside Lean.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X Rep Target Downstream Θ Probe Y : Type*}

/-- A representation is sufficient for a target when representation collisions
stay inside target fibers. This is the representation/information form of
"no task-relevant information was lost." -/
def TargetSufficientRepresentation
    (rep : X → Rep)
    (target : X → Target) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → target x = target y

/-- A readout realizes a target from a representation. -/
def TargetReadoutRealizes
    (rep : X → Rep)
    (target : X → Target)
    (readout : Rep → Target) : Prop :=
  ∀ x, readout (rep x) = target x

/-- The target itself is sufficient by construction. -/
theorem target_self_sufficient
    (target : X → Target) :
    TargetSufficientRepresentation target target := by
  intro x y hxy
  exact hxy

/-- A sufficient representation cannot collapse two target-distinct inputs. -/
theorem targetSufficient_no_collision_of_distinguished_target
    {rep : X → Rep}
    {target : X → Target}
    (hSuff : TargetSufficientRepresentation rep target)
    {x y : X}
    (hSep : target x ≠ target y) :
    rep x ≠ rep y := by
  intro hxy
  exact hSep (hSuff hxy)

/-- A representation is target-sufficient iff the target has a readout from that
representation. The default branch is irrelevant off the image of `rep`. -/
theorem targetSufficient_iff_exists_readout
    [Inhabited X]
    {rep : X → Rep}
    {target : X → Target} :
    TargetSufficientRepresentation rep target ↔
      ∃ readout : Rep → Target,
        TargetReadoutRealizes rep target readout := by
  constructor
  · intro hSuff
    classical
    let readout : Rep → Target := fun r =>
      if h : ∃ x, rep x = r then target (Classical.choose h) else target default
    refine ⟨readout, ?_⟩
    intro x
    unfold readout
    have hx : ∃ x', rep x' = rep x := ⟨x, rfl⟩
    simp [hx]
    exact hSuff (Classical.choose_spec hx)
  · rintro ⟨readout, hReadout⟩ x y hxy
    calc
      target x = readout (rep x) := (hReadout x).symm
      _ = readout (rep y) := by rw [hxy]
      _ = target y := hReadout y

/-- A downstream quantity is target-measurable when it is constant on target
fibers. -/
def TargetMeasurable
    (target : X → Target)
    (downstream : X → Downstream) : Prop :=
  ∀ ⦃x y : X⦄, target x = target y → downstream x = downstream y

/-- If `rep` is sufficient for `target`, then every target-measurable downstream
quantity is also preserved across representation collisions. -/
theorem targetSufficient_preserves_targetMeasurable
    {rep : X → Rep}
    {target : X → Target}
    {downstream : X → Downstream}
    (hSuff : TargetSufficientRepresentation rep target)
    (hMeas : TargetMeasurable target downstream) :
    TargetSufficientRepresentation rep downstream := by
  intro x y hxy
  exact hMeas (hSuff hxy)

/-! ## Likelihood-model sufficiency -/

/-- Classical likelihood-family sufficiency: a representation preserves the full
likelihood value for every parameter `θ`. -/
def LikelihoodFamilySufficient
    (rep : X → Rep)
    (likelihood : Θ → X → Y) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → ∀ θ : Θ, likelihood θ x = likelihood θ y

/-- A likelihood readout factors the whole likelihood family through the
representation. -/
def LikelihoodReadoutRealizes
    (rep : X → Rep)
    (likelihood : Θ → X → Y)
    (readout : Θ → Rep → Y) : Prop :=
  ∀ θ x, readout θ (rep x) = likelihood θ x

/-- Likelihood-family sufficiency is exactly contextual query sufficiency with
the parameter as context. -/
theorem likelihoodFamilySufficient_iff_querySufficient
    {rep : X → Rep}
    {likelihood : Θ → X → Y} :
    LikelihoodFamilySufficient rep likelihood ↔
      QuerySufficient rep likelihood := by
  rfl

/-- Likelihood-family sufficiency is equivalent to factoring every likelihood
through the representation. -/
theorem likelihoodFamilySufficient_iff_exists_readout
    [Inhabited X]
    {rep : X → Rep}
    {likelihood : Θ → X → Y} :
    LikelihoodFamilySufficient rep likelihood ↔
      ∃ readout : Θ → Rep → Y,
        LikelihoodReadoutRealizes rep likelihood readout := by
  constructor
  · intro hSuff
    have hQuery : QuerySufficient rep likelihood := hSuff
    rcases (querySufficient_iff_exists_contextReadout
      (rep := rep) (query := likelihood)).mp hQuery with ⟨readout, hReadout⟩
    refine ⟨fun θ r => readout r θ, ?_⟩
    intro θ x
    exact hReadout x θ
  · rintro ⟨readout, hReadout⟩ x y hxy θ
    calc
      likelihood θ x = readout θ (rep x) := (hReadout θ x).symm
      _ = readout θ (rep y) := by rw [hxy]
      _ = likelihood θ y := hReadout θ y

/-- A likelihood-family sufficient representation cannot collapse inputs that
some parameter setting assigns different likelihood values. -/
theorem likelihoodFamilySufficient_no_collision_of_distinguished_likelihood
    {rep : X → Rep}
    {likelihood : Θ → X → Y}
    (hSuff : LikelihoodFamilySufficient rep likelihood)
    {x y : X}
    (hSep : ∃ θ : Θ, likelihood θ x ≠ likelihood θ y) :
    rep x ≠ rep y := by
  intro hxy
  rcases hSep with ⟨θ, hθ⟩
  exact hθ (hSuff hxy θ)

/-! ## Likelihood-free / implicit-model sufficiency -/

/-- Likelihood-free sufficiency: a representation preserves every response in a
family of simulator, probe, posterior, or contextual-query targets. `Probe` is
whatever replaces an explicit likelihood parameter in the implicit-model lane. -/
def LikelihoodFreeResponseSufficient
    (rep : X → Rep)
    (response : Probe → X → Y) : Prop :=
  ∀ ⦃x y : X⦄, rep x = rep y → ∀ p : Probe, response p x = response p y

/-- A likelihood-free response readout factors the response family through the
representation. -/
def LikelihoodFreeReadoutRealizes
    (rep : X → Rep)
    (response : Probe → X → Y)
    (readout : Probe → Rep → Y) : Prop :=
  ∀ p x, readout p (rep x) = response p x

/-- Likelihood-free response sufficiency is exactly contextual query sufficiency
with probes as contexts. -/
theorem likelihoodFreeResponseSufficient_iff_querySufficient
    {rep : X → Rep}
    {response : Probe → X → Y} :
    LikelihoodFreeResponseSufficient rep response ↔
      QuerySufficient rep response := by
  rfl

/-- Likelihood-free response sufficiency is equivalent to readout factorization
through the learned representation. -/
theorem likelihoodFreeResponseSufficient_iff_exists_readout
    [Inhabited X]
    {rep : X → Rep}
    {response : Probe → X → Y} :
    LikelihoodFreeResponseSufficient rep response ↔
      ∃ readout : Probe → Rep → Y,
        LikelihoodFreeReadoutRealizes rep response readout := by
  constructor
  · intro hSuff
    have hQuery : QuerySufficient rep response := hSuff
    rcases (querySufficient_iff_exists_contextReadout
      (rep := rep) (query := response)).mp hQuery with ⟨readout, hReadout⟩
    refine ⟨fun p r => readout r p, ?_⟩
    intro p x
    exact hReadout x p
  · rintro ⟨readout, hReadout⟩ x y hxy p
    calc
      response p x = readout p (rep x) := (hReadout p x).symm
      _ = readout p (rep y) := by rw [hxy]
      _ = response p y := hReadout p y

/-- A likelihood-free sufficient representation cannot collapse inputs that
some probe distinguishes. -/
theorem likelihoodFreeResponseSufficient_no_collision_of_distinguished_probe
    {rep : X → Rep}
    {response : Probe → X → Y}
    (hSuff : LikelihoodFreeResponseSufficient rep response)
    {x y : X}
    (hSep : ∃ p : Probe, response p x ≠ response p y) :
    rep x ≠ rep y := by
  intro hxy
  rcases hSep with ⟨p, hp⟩
  exact hp (hSuff hxy p)

/-- Two-sided contextual sufficiency is likelihood-free response sufficiency
with `(left, right)` contexts as probes. -/
theorem twoSidedContextSufficient_iff_likelihoodFreeResponseSufficient
    {X Rep Y : Type*}
    [Monoid X]
    {rep : X → Rep}
    {fstar : X → Y} :
    TwoSidedContextSufficient rep fstar ↔
      LikelihoodFreeResponseSufficient rep (TwoSidedContextQuery fstar) := by
  rfl

/-- Sliced sufficiency plus slice coverage gives likelihood-free response
sufficiency for the underlying response family. -/
theorem slicedQuerySufficient_implies_likelihoodFreeResponseSufficient
    {Slice SliceVal : Type*}
    {rep : X → Rep}
    {response : Probe → X → Y}
    {slice : Slice → (Probe → Y) → SliceVal}
    (hCover : SlicesCoverResponseFibers response slice)
    (hSliced : SlicedQuerySufficient rep response slice) :
    LikelihoodFreeResponseSufficient rep response := by
  have hQuery : QuerySufficient rep response :=
    slicedQuerySufficient_implies_querySufficient hCover hSliced
  exact
    (likelihoodFreeResponseSufficient_iff_querySufficient
      (rep := rep) (response := response)).mpr hQuery

/-- Finite sliced sufficiency plus finite slice coverage gives
likelihood-free response sufficiency for the underlying response family. -/
theorem finiteSliced_zeroLoss_implies_likelihoodFreeResponseSufficient
    {Slice SliceVal : Type*}
    {selected : Finset Slice}
    {rep : X → Rep}
    {response : Probe → X → Y}
    {slice : Slice → (Probe → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibers selected response slice)
    (hSliced : SlicedQuerySufficientOn selected rep response slice) :
    LikelihoodFreeResponseSufficient rep response := by
  have hQuery : QuerySufficient rep response :=
    finiteSliced_zeroLoss_implies_querySufficient hCover hSliced
  exact
    (likelihoodFreeResponseSufficient_iff_querySufficient
      (rep := rep) (response := response)).mpr hQuery

end FormalProofs.OPT
