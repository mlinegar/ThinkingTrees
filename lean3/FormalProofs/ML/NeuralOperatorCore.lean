import Mathlib

/-!
# FormalProofs/ML/NeuralOperatorCore.lean

Abstract neural-operator interfaces used by the C-TreePO manuscript.

This file deliberately formalizes only the portion of the neural-operator story
that the paper actually needs:

- a family of operators with shared parameters;
- discretizations and nested refinements;
- compact realized-call sets;
- discretization invariance as a uniform-on-compact convergence interface.

The Banach-space and approximation-theoretic proofs from the neural-operator
literature are treated as external mathematical inputs. The purpose of this
file is to give the paper a precise in-repo interface for those inputs.
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

/-- A parameterized family of operators from `A` to `U`. -/
abbrev OperatorFamily (A U Θ : Type*) := Θ → A → U

/-- A level-specific discretization of a domain. The carrier is left abstract:
the paper only uses the existence of discretization levels, not a specific
encoding of mesh points. -/
structure Discretization (Domain : Type*) where
  carrier : Set Domain

/-- A nested discretization schedule. The `converges` field is intentionally
abstract: the Section 9-style approximation layer only needs to record that a
refinement notion has been fixed. -/
structure DiscreteRefinement (Domain : Type*) where
  level : ℕ → Discretization Domain
  nested : ∀ n, (level n).carrier ⊆ (level (n + 1)).carrier
  converges : Prop

/-- A realized-call set that is assumed compact by external analysis. We keep
the compactness witness abstract because the current paper uses compactness only
as an approximation-theoretic side condition, not as an object manipulated by
later proofs. -/
structure CompactRealizedCallSet (A : Type*) where
  carrier : Set A
  isCompact : Prop

/-- A discretized realization of an operator family indexed by refinement level. -/
abbrev DiscretizedOperatorFamily (A U Θ : Type*) := ℕ → Θ → A → U

/-- Discretization invariance in the exact form C-TreePO uses: for fixed
parameters, discretized realizations converge uniformly on every compact
realized-call set. -/
def DiscretizationInvariantFamily
    {A U Θ : Type*} [PseudoMetricSpace U]
    (G : OperatorFamily A U Θ)
    (refinement : DiscreteRefinement A)
    (realize : DiscretizedOperatorFamily A U Θ) : Prop :=
  ∀ θ (K : CompactRealizedCallSet A) ε, K.isCompact → 0 < ε →
    ∃ N : ℕ, ∀ n ≥ N, ∀ a, a ∈ K.carrier → dist (realize n θ a) (G θ a) ≤ ε

/-- Restrict an operator family to a fixed parameter. -/
def fixedOperator
    {A U Θ : Type*}
    (G : OperatorFamily A U Θ) (θ : Θ) : A → U :=
  G θ

/-- The realized-call compact attached to a singleton input. -/
def singletonCompactRealizedCallSet {A : Type*} (a : A) : CompactRealizedCallSet A where
  carrier := {x | x = a}
  isCompact := True

end ML
