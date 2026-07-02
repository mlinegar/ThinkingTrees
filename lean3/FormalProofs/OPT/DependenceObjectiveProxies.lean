import FormalProofs.OPT.HybridInformationObjectives

/-!
# FormalProofs/OPT/DependenceObjectiveProxies.lean

Symbolic objective layer for the dependence-objective literature behind the
NASS/contextual probe menu.

This file does **not** formalize Shannon mutual information, variational
estimator consistency, negative-sampling asymptotics, distance-correlation
independence theorems, or optimal-transport duality.  It only records the
optimization algebra that is safe to reuse across the paper line:

* if a loss reverses a chosen proxy objective, loss minimizers are proxy
  maximizers;
* if a proxy uniformly approximates a target information objective, exact proxy
  maximizers are near-maximizers of the target objective; and
* a variational lower-bound relation alone is too weak to transport argmaxes.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Candidate : Type*}

/-- A point is an `epsilon`-argmax when every other candidate is within
`epsilon` of its objective value.  No probabilistic or estimator semantics are
attached to this predicate. -/
def IsEpsilonArgmax
    (epsilon : ℝ)
    (objective : Candidate → ℝ)
    (candidate : Candidate) : Prop :=
  ∀ other : Candidate, objective other ≤ objective candidate + epsilon

/-- Uniform deterministic proxy error between a dependence proxy and the target
information objective it is meant to approximate. -/
def UniformProxyError
    (epsilon : ℝ)
    (proxy information : Candidate → ℝ) : Prop :=
  ∀ candidate : Candidate, |proxy candidate - information candidate| ≤ epsilon

/-- A variational/lower-bound proxy relation.  This is intentionally weaker
than an order-equivalence or uniform-error assumption. -/
def VariationalLowerBoundProxy
    (proxy information : Candidate → ℝ) : Prop :=
  ∀ candidate : Candidate, proxy candidate ≤ information candidate

/-- Exact proxy maximizers are zero-slack epsilon maximizers of the same proxy. -/
theorem dependenceProxy_exactArgmax_is_epsilonArgmax_zero
    {proxy : Candidate → ℝ}
    {candidate : Candidate}
    (hMax : IsArgmax proxy candidate) :
    IsEpsilonArgmax 0 proxy candidate := by
  intro other
  simpa using hMax other

/-- Uniform proxy error plus exact proxy optimality yields near-optimality for
the target information objective, with the deterministic two-sided slack
`epsilon + epsilon`. -/
theorem uniformProxyError_argmax_implies_informationEpsilonArgmax
    {epsilon : ℝ}
    {proxy information : Candidate → ℝ}
    {candidate : Candidate}
    (hError : UniformProxyError epsilon proxy information)
    (hMax : IsArgmax proxy candidate) :
    IsEpsilonArgmax (epsilon + epsilon) information candidate := by
  intro other
  have hOtherAbs := abs_le.mp (hError other)
  have hCandidateAbs := abs_le.mp (hError candidate)
  have hProxyMax := hMax other
  linarith

/-- If a dependence proxy has the same pairwise order as the symbolic
information objective, then the proxy and information argmax sets coincide. -/
theorem objectiveOrderEquivalent_argmax_iff
    {proxy information : Candidate → ℝ}
    (hOrder : ObjectiveOrderEquivalent proxy information)
    {candidate : Candidate} :
    IsArgmax proxy candidate ↔ IsArgmax information candidate := by
  constructor
  · intro hMax other
    exact (hOrder other candidate).mp (hMax other)
  · intro hMax other
    exact (hOrder other candidate).mpr (hMax other)

/-- Generic dependence-objective bridge: if the training loss reverses the
chosen proxy order, loss minimizers are exactly proxy maximizers. -/
theorem dependenceLoss_argmin_iff_proxyArgmax
    {loss proxy : Candidate → ℝ}
    (hOrder : LossOrderReversesInformation loss proxy)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax proxy candidate :=
  lossArgmin_iff_informationArgmax_of_orderReverses hOrder

/-- MINE/Donsker-Varadhan-style losses are optimizer-equivalent to maximizing
their symbolic proxy when the loss is supplied as an order-reversing proxy
loss.  This is not an estimator-consistency theorem. -/
theorem mineDV_lossArgmin_iff_proxyArgmax
    {loss proxy : Candidate → ℝ}
    (hOrder : LossOrderReversesInformation loss proxy)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax proxy candidate :=
  dependenceLoss_argmin_iff_proxyArgmax hOrder

/-- Deep InfoMax / JSD classifier losses are optimizer-equivalent to maximizing
their symbolic proxy under the same order-reversal assumption. -/
theorem deepInfoMaxJSD_lossArgmin_iff_proxyArgmax
    {loss proxy : Candidate → ℝ}
    (hOrder : LossOrderReversesInformation loss proxy)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax proxy candidate :=
  dependenceLoss_argmin_iff_proxyArgmax hOrder

/-- InfoNCE/CPC-style contrastive losses are optimizer-equivalent to maximizing
their symbolic proxy under an order-reversal assumption.  Negative-sampling
consistency and bound tightness are outside this theorem. -/
theorem infoNCE_lossArgmin_iff_proxyArgmax
    {loss proxy : Candidate → ℝ}
    (hOrder : LossOrderReversesInformation loss proxy)
    {candidate : Candidate} :
    IsArgmin loss candidate ↔ IsArgmax proxy candidate :=
  dependenceLoss_argmin_iff_proxyArgmax hOrder

/-- Distance-correlation-style objectives enter the Lean surface as direct
symbolic dependence proxies, not as a formal independence characterization. -/
theorem distanceCorrelation_proxyMax
    {proxy : Candidate → ℝ}
    {candidate : Candidate}
    (hMax : IsArgmax proxy candidate) :
    IsArgmax proxy candidate :=
  hMax

/-- Wasserstein-dependency-style objectives enter the Lean surface as direct
symbolic dependence proxies, not as an optimal-transport duality theorem. -/
theorem wassersteinDependency_proxyMax
    {proxy : Candidate → ℝ}
    {candidate : Candidate}
    (hMax : IsArgmax proxy candidate) :
    IsArgmax proxy candidate :=
  hMax

/-- Boundary theorem: being a pointwise lower bound on information is not
enough to transport argmaxes.  A constant proxy is a lower bound for the
Boolean information objective below, but its selected proxy maximizer is not an
information maximizer. -/
theorem lowerBoundProxy_alone_counterexample :
    ∃ (proxy information : Bool → ℝ) (candidate : Bool),
      VariationalLowerBoundProxy proxy information ∧
      IsArgmax proxy candidate ∧
      ¬ IsArgmax information candidate := by
  refine ⟨(fun _ => 0), (fun b => if b then (1 : ℝ) else 0), false, ?_⟩
  constructor
  · intro b
    cases b <;> simp
  constructor
  · intro other
    cases other <;> simp
  · intro hMax
    have hTrue := hMax true
    simp at hTrue
    linarith

end FormalProofs.OPT
