import FormalProofs.ML.KovachkiFiniteDimensionalization

/-!
# FormalProofs/ML/NeuralOperatorApproximation.lean

Section 9-style approximation interfaces for neural operators.

The paper only uses two theorem surfaces:

- uniform approximation on compact realized-call sets;
- optionally, an `L²`-type approximation surface.

This file packages those surfaces as reusable predicates. The mathematical proof
that a specific neural-operator class satisfies these predicates remains an
external input from the cited approximation theory.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ML

/-- Uniform operator approximation on a compact realized-call set. -/
def UniformOperatorApproxOnCompact
    {A U : Type*} [PseudoMetricSpace U]
    (target approx : A → U)
    (K : CompactRealizedCallSet A)
    (ε : ℝ) : Prop :=
  ∀ a, a ∈ K.carrier → dist (target a) (approx a) ≤ ε

/-- Abstract `L²`-type approximation surface. The concrete Bochner-risk or
empirical proxy is supplied externally as `l2Risk`. -/
def L2OperatorApproxOnMeasure
    {A U : Type*}
    (l2Risk : (A → U) → (A → U) → ℝ)
    (target approx : A → U)
    (ε : ℝ) : Prop :=
  l2Risk target approx ≤ ε

/-- Section 9-style universal approximation on compact realized-call sets for a
chosen neural-operator class. `AdmissibleTarget` packages the continuity /
regularity side conditions from the external approximation theorem. -/
def NeuralOperatorUniversalApproxUniform
    {A U : Type*} [PseudoMetricSpace U]
    (NeuralOperators : Set (A → U))
    (AdmissibleTarget : (A → U) → Prop) : Prop :=
  ∀ target (K : CompactRealizedCallSet A) ε, AdmissibleTarget target → K.isCompact → 0 < ε →
    ∃ approx, approx ∈ NeuralOperators ∧
      UniformOperatorApproxOnCompact target approx K ε

/-- Optional Section 9-style `L²` universal approximation interface. -/
def NeuralOperatorUniversalApproxL2
    {A U : Type*}
    (NeuralOperators : Set (A → U))
    (AdmissibleTarget : (A → U) → Prop)
    (l2Risk : (A → U) → (A → U) → ℝ) : Prop :=
  ∀ target ε, AdmissibleTarget target → 0 < ε →
    ∃ approx, approx ∈ NeuralOperators ∧
      L2OperatorApproxOnMeasure l2Risk target approx ε

/-- Convenience wrapper: extract a uniform compact-set approximation witness
from the universal-approximation interface. -/
theorem uniformApproxOnCompact_of_universalApprox
    {A U : Type*} [PseudoMetricSpace U]
    {NeuralOperators : Set (A → U)}
    {AdmissibleTarget : (A → U) → Prop}
    (hApprox : NeuralOperatorUniversalApproxUniform NeuralOperators AdmissibleTarget)
    {target : A → U} {K : CompactRealizedCallSet A} {ε : ℝ}
    (hTarget : AdmissibleTarget target)
    (hK : K.isCompact)
    (hε : 0 < ε) :
    ∃ approx, approx ∈ NeuralOperators ∧
      UniformOperatorApproxOnCompact target approx K ε :=
  hApprox target K ε hTarget hK hε

/-- Convenience wrapper for the optional `L²` approximation surface. -/
theorem l2ApproxOnMeasure_of_universalApprox
    {A U : Type*}
    {NeuralOperators : Set (A → U)}
    {AdmissibleTarget : (A → U) → Prop}
    {l2Risk : (A → U) → (A → U) → ℝ}
    (hApprox : NeuralOperatorUniversalApproxL2 NeuralOperators AdmissibleTarget l2Risk)
    {target : A → U} {ε : ℝ}
    (hTarget : AdmissibleTarget target)
    (hε : 0 < ε) :
    ∃ approx, approx ∈ NeuralOperators ∧
      L2OperatorApproxOnMeasure l2Risk target approx ε :=
  hApprox target ε hTarget hε

/-!
## Kovachki Lemma 22 Bridge

The finite-dimensionalization theorem produces an explicit realized operator of
the form `G_{J'} ∘ φ ∘ F_J`.  The existing C-TreePO bridge only needs the
resulting uniform compact-set approximation predicate, so this section converts
the Lemma-22 witness into that predicate.
-/

/-- A finite-dimensionalization witness directly induces uniform approximation
on the corresponding realized-call compact. -/
theorem uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalization
    {X Y : Type*}
    [NormedAddCommGroup X] [NormedSpace ℝ X]
    [NormedAddCommGroup Y] [NormedSpace ℝ Y]
    {target : X → Y} {K : CompactRealizedCallSet X} {ε : ℝ}
    (hFD : FiniteDimensionalizationOnCompact target K.carrier ε) :
    UniformOperatorApproxOnCompact target hFD.realized K ε :=
  hFD.uniform_error

/-- Existence form of the Lemma-22 bridge: after finite-dimensionalization there
exists a realized finite-dimensional operator satisfying the compact-uniform
approximation predicate used by the theorem-backedness bridge. -/
theorem uniformApproxOnCompact_exists_of_kovachkiFiniteDimensionalization
    {X Y : Type*}
    [NormedAddCommGroup X] [NormedSpace ℝ X]
    [NormedAddCommGroup Y] [NormedSpace ℝ Y]
    {target : X → Y} {K : CompactRealizedCallSet X} {ε : ℝ}
    (hFD : FiniteDimensionalizationOnCompact target K.carrier ε) :
    ∃ approx : X → Y, UniformOperatorApproxOnCompact target approx K ε :=
  ⟨hFD.realized, uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalization hFD⟩

/-- Uniform-continuity version of the Lemma-22 bridge. For Lipschitz or
globally uniformly continuous target operators, the Lemma-21 stability premise
is discharged by `kovachkiLemma21Stability_of_uniformContinuous`, so the
finite-dimensionalization theorem directly yields the compact-set
approximation predicate used downstream by C-TreePO. -/
theorem uniformApproxOnCompact_exists_of_kovachkiFiniteDimensionalization_of_uniformContinuous
    {X Y : Type*}
    [NormedAddCommGroup X] [NormedSpace ℝ X]
    [NormedAddCommGroup Y] [NormedSpace ℝ Y]
    [CompleteSpace X] [CompleteSpace Y]
    {target : X → Y} {K : CompactRealizedCallSet X} {ε : ℝ}
    (hX : HasApproximationProperty X)
    (hY : HasApproximationProperty Y)
    (hTarget : UniformContinuous target)
    (hK : IsCompact K.carrier)
    (hε : 0 < ε) :
    ∃ approx : X → Y, UniformOperatorApproxOnCompact target approx K ε :=
  uniformApproxOnCompact_exists_of_kovachkiFiniteDimensionalization
    (kovachki_finiteDimensionalization_on_compact_of_uniformContinuous
      hX hY hTarget hK hε)

/-- Type-erased finite-dimensionalization certificate on an arbitrary
realized-call compact.  This is the shape used by the C-TreePO bridge after the
Banach-space Lemma-22 witness has been transported through the paper's chosen
representation of a call site. -/
structure KovachkiFiniteDimensionalizedApproxOnCompact
    {A U : Type*} [PseudoMetricSpace U]
    (target : A → U) (K : CompactRealizedCallSet A) (ε : ℝ) where
  realized : A → U
  lemma22_certificate : Prop
  error_bound : UniformOperatorApproxOnCompact target realized K ε

/-- Extract the uniform compact-set approximation predicate from the type-erased
Kovachki finite-dimensionalization certificate. -/
theorem uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalizedApprox
    {A U : Type*} [PseudoMetricSpace U]
    {target : A → U} {K : CompactRealizedCallSet A} {ε : ℝ}
    (hFD : KovachkiFiniteDimensionalizedApproxOnCompact target K ε) :
    UniformOperatorApproxOnCompact target hFD.realized K ε :=
  hFD.error_bound

end ML
