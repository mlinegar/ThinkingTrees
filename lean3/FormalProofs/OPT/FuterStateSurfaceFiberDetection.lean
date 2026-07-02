import FormalProofs.OPT.PreferenceScope

/-!
# FormalProofs/OPT/FuterStateSurfaceFiberDetection.lean

Abstract Lean surface for David Futer's 2013 theorem
"Fiber detection for state surfaces".

This file does **not** prove the 3-manifold-topology theorem.  A direct proof
would require a library for link diagrams, Kauffman states, state surfaces,
Murasugi sums, fibrations of link complements, and Jones-polynomial
coefficients.  Instead, this file records the theorem and corollary as precise
typed statements and exposes the detector schema that makes the result relevant
to the C-TreePO preference-scope discussion.

Important terminology warning: Futer's "fiber" is a topological fiber surface
in a fibration over `S¹`.  C-TreePO's "state fiber" is a preimage/equivalence
class of a state map `sigma`.  These are not the same mathematical object.  The
shared pattern is detection: a global property is decided by an associated
combinatorial/state certificate.
-/

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT
namespace Futer2013

universe uLink uDiagram uState uSurface uGraph uObj uCert

/-! ## Abstract topology vocabulary -/

/-- Type carriers needed to state Futer's theorem without building a full knot
theory library. -/
structure StateSurfaceTypes where
  Link : Type uLink
  Diagram : Type uDiagram
  KauffmanState : Type uState
  StateSurface : Type uSurface
  StateGraph : Type uGraph

/-- One diagram/state/surface/reduced-graph configuration. -/
structure StateSurfaceInstance (U : StateSurfaceTypes) where
  link : U.Link
  diagram : U.Diagram
  state : U.KauffmanState
  surface : U.StateSurface
  reducedGraph : U.StateGraph

/-- Predicate vocabulary for state-surface fiber detection.  The predicates are
parameters because this module states the theorem abstractly rather than
formalizing low-dimensional topology. -/
structure StateSurfacePredicates (U : StateSurfaceTypes) where
  isLinkDiagramOf : U.Diagram → U.Link → Prop
  isKauffmanStateOf : U.KauffmanState → U.Diagram → Prop
  isHomogeneousState : U.KauffmanState → U.Diagram → Prop
  isStateSurfaceFor : U.StateSurface → U.Diagram → U.KauffmanState → Prop
  boundaryIsLink : U.StateSurface → U.Link → Prop
  reducedStateGraphOf : U.StateGraph → U.Diagram → U.KauffmanState → Prop
  isTree : U.StateGraph → Prop
  isFiberedWithFiberSurface : U.Link → U.StateSurface → Prop
  isConnectedDiagram : U.Diagram → Prop
  isAAdequateDiagram : U.Diagram → Prop
  isBAdequateDiagram : U.Diagram → Prop
  allAStateSurface : U.Diagram → U.StateSurface
  allBStateSurface : U.Diagram → U.StateSurface
  nextToLastJonesCoeffZero : U.Link → Prop
  secondJonesCoeffZero : U.Link → Prop

variable {U : StateSurfaceTypes}

/-- Futer 2013, Theorem 1, as a Lean proposition:
for a homogeneous state `sigma`, the link complement fibers with fiber
`S_sigma` iff the associated reduced graph `G'_sigma` is a tree. -/
def theorem1_statement (P : StateSurfacePredicates U) : Prop :=
  ∀ I : StateSurfaceInstance U,
    P.isLinkDiagramOf I.diagram I.link →
    P.isKauffmanStateOf I.state I.diagram →
    P.isHomogeneousState I.state I.diagram →
    P.isStateSurfaceFor I.surface I.diagram I.state →
    P.boundaryIsLink I.surface I.link →
    P.reducedStateGraphOf I.reducedGraph I.diagram I.state →
    (P.isFiberedWithFiberSurface I.link I.surface ↔ P.isTree I.reducedGraph)

/-- Futer 2013, Corollary 2, A-adequate half, as a Lean proposition:
for a connected A-adequate diagram, the next-to-last Jones coefficient
vanishes exactly when the all-A state surface is a fiber surface. -/
def corollary2_A_statement (P : StateSurfacePredicates U) : Prop :=
  ∀ (K : U.Link) (D : U.Diagram),
    P.isLinkDiagramOf D K →
    P.isConnectedDiagram D →
    P.isAAdequateDiagram D →
    (P.isFiberedWithFiberSurface K (P.allAStateSurface D) ↔
      P.nextToLastJonesCoeffZero K)

/-- Futer 2013, Corollary 2, B-adequate half, as a Lean proposition:
for a connected B-adequate diagram, the second Jones coefficient vanishes
exactly when the all-B state surface is a fiber surface. -/
def corollary2_B_statement (P : StateSurfacePredicates U) : Prop :=
  ∀ (K : U.Link) (D : U.Diagram),
    P.isLinkDiagramOf D K →
    P.isConnectedDiagram D →
    P.isBAdequateDiagram D →
    (P.isFiberedWithFiberSurface K (P.allBStateSurface D) ↔
      P.secondJonesCoeffZero K)

/-! ## Detector schema -/

/-- A detector problem: each object has an associated certificate, and a global
property is equivalent to a certificate-level predicate. -/
structure DetectorProblem where
  Object : Type uObj
  Certificate : Type uCert
  property : Object → Prop
  certificateOf : Object → Certificate
  certificateProperty : Certificate → Prop

/-- The certificate is an exact detector for the object-level property. -/
def ExactDetector (D : DetectorProblem) : Prop :=
  ∀ x : D.Object, D.property x ↔ D.certificateProperty (D.certificateOf x)

/-- Valid input package for Futer's Theorem 1: a homogeneous state surface
together with the hypotheses needed to apply the theorem. -/
structure ValidHomogeneousStateSurface
    (P : StateSurfacePredicates U) where
  inst : StateSurfaceInstance U
  hDiagram : P.isLinkDiagramOf inst.diagram inst.link
  hState : P.isKauffmanStateOf inst.state inst.diagram
  hHomogeneous : P.isHomogeneousState inst.state inst.diagram
  hSurface : P.isStateSurfaceFor inst.surface inst.diagram inst.state
  hBoundary : P.boundaryIsLink inst.surface inst.link
  hReducedGraph : P.reducedStateGraphOf inst.reducedGraph inst.diagram inst.state

/-- The detector problem induced by Futer's Theorem 1:
objects are valid homogeneous state-surface configurations; certificates are
reduced state graphs; the object property is "fibered with this fiber surface";
the certificate property is "is a tree". -/
def futer_theorem1_detector_problem
    (P : StateSurfacePredicates U) : DetectorProblem where
  Object := ValidHomogeneousStateSurface P
  Certificate := U.StateGraph
  property := fun X => P.isFiberedWithFiberSurface X.inst.link X.inst.surface
  certificateOf := fun X => X.inst.reducedGraph
  certificateProperty := P.isTree

/-- Futer's Theorem 1 says exactly that the reduced state graph is an exact
detector for fiberedness of homogeneous state surfaces. -/
theorem theorem1_yields_exact_detector
    {P : StateSurfacePredicates U}
    (h : theorem1_statement P) :
    ExactDetector (futer_theorem1_detector_problem P) := by
  intro X
  exact h X.inst X.hDiagram X.hState X.hHomogeneous
    X.hSurface X.hBoundary X.hReducedGraph

/-! ## C-TreePO analogue: state-level detector -/

/-- C-TreePO's exact state-factored predicate route also forms an exact
detector: the certificate is `sigma x`, and the property is a predicate on that
state.  This is the precise analogy to Futer's reduced-graph detector, without
identifying topological fibers with state fibers. -/
def state_factored_detector_problem
    {Doc State : Type*}
    (sigma : Doc → State) (propertyOnState : State → Prop) :
    DetectorProblem where
  Object := Doc
  Certificate := State
  property := fun x => propertyOnState (sigma x)
  certificateOf := sigma
  certificateProperty := propertyOnState

/-- State-factored predicates are exactly detected by their state certificate. -/
theorem state_factored_detector_exact
    {Doc State : Type*}
    (sigma : Doc → State) (propertyOnState : State → Prop) :
    ExactDetector (state_factored_detector_problem sigma propertyOnState) := by
  intro x
  simp [state_factored_detector_problem]

end Futer2013
end FormalProofs.OPT

end
