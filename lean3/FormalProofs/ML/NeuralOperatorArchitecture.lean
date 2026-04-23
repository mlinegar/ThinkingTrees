import FormalProofs.ML.NeuralOperatorApproximation
import FormalProbability.ML.NeuralOperatorArchitectureCore

/-!
# FormalProofs/ML/NeuralOperatorArchitecture.lean

Equation-(6)-style neural-operator architecture surfaces.

The reusable equation-(6) architecture primitives live in
`FormalProbability.ML.NeuralOperatorArchitectureCore`; this file keeps the
ThinkingTrees-facing names and adds the C-TreePO risk/refinement interfaces
that depend on local approximation definitions.
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

/-! ## Equation-(6)-style finite compositions -/

/-- ThinkingTrees-facing alias for the reusable homogeneous neural-operator
layer core. -/
abbrev NeuralOperatorLayer :=
  NeuralOperators.ArchCore.NeuralOperatorLayer

namespace NeuralOperatorLayer

/-- Apply a neural-operator layer. -/
def apply {State : Type*} (layer : NeuralOperatorLayer State) : State -> State :=
  NeuralOperators.ArchCore.NeuralOperatorLayer.apply layer

end NeuralOperatorLayer

/-- ThinkingTrees-facing alias for the reusable equation-(6) architecture
shape `Q o layer_T o ... o layer_1 o P`. -/
abbrev Equation6NeuralOperator :=
  NeuralOperators.ArchCore.Equation6NeuralOperator

namespace Equation6NeuralOperator

/-- Hidden representation after all equation-(6) layers have been applied. -/
def hidden {A Hidden U : Type*}
    (G : Equation6NeuralOperator A Hidden U) (a : A) : Hidden :=
  NeuralOperators.ArchCore.Equation6NeuralOperator.hidden G a

/-- The realized operator `A -> U` represented by the architecture. -/
def realize {A Hidden U : Type*}
    (G : Equation6NeuralOperator A Hidden U) : A -> U :=
  NeuralOperators.ArchCore.Equation6NeuralOperator.realize G

/-- With no hidden layers, the architecture is just `Q o P`. -/
theorem realize_nil {A Hidden U : Type*}
    (P : A -> Hidden) (Q : Hidden -> U) :
    realize ({ lift := P, layers := [], project := Q } :
      Equation6NeuralOperator A Hidden U) = fun a => Q (P a) := by
  rfl

end Equation6NeuralOperator

/-- The class of operators representable by an equation-(6)-style architecture
with a fixed hidden space. -/
abbrev Equation6NeuralOperatorClass (A Hidden U : Type*) : Set (A -> U) :=
  NeuralOperators.ArchCore.Equation6NeuralOperatorClass A Hidden U

/-- Every equation-(6) architecture realizes an element of the corresponding
architecture class. -/
theorem equation6Realization_mem_class {A Hidden U : Type*}
    (arch : Equation6NeuralOperator A Hidden U) :
    arch.realize ∈ Equation6NeuralOperatorClass A Hidden U :=
  NeuralOperators.ArchCore.Equation6NeuralOperator.mem_class arch

/-! ## Kovachki risk/refinement interfaces used by the paper -/

/-- Equation (3)-style L2/Bochner risk upper bound, represented through the
abstract `l2Risk` functional used in `NeuralOperatorApproximation.lean`. -/
abbrev KovachkiEquation3RiskLE {A U : Type*}
    (l2Risk : (A -> U) -> (A -> U) -> ℝ)
    (target approx : A -> U) (eps : ℝ) : Prop :=
  L2OperatorApproxOnMeasure l2Risk target approx eps

/-- Equation (5)-style uniform risk on compact sets. -/
abbrev KovachkiEquation5RiskLE {A U : Type*} [PseudoMetricSpace U]
    (target approx : A -> U)
    (K : CompactRealizedCallSet A) (eps : ℝ) : Prop :=
  UniformOperatorApproxOnCompact target approx K eps

/-- Definition 4-style discretization invariance, re-exported with a
Kovachki-facing name. -/
abbrev KovachkiDefinition4DiscretizationInvariant
    {A U Θ : Type*} [PseudoMetricSpace U]
    (G : OperatorFamily A U Θ)
    (refinement : DiscreteRefinement A)
    (realize : DiscretizedOperatorFamily A U Θ) : Prop :=
  DiscretizationInvariantFamily G refinement realize

/-- Universal uniform approximation for the equation-(6) class. This is the
in-repo interface corresponding to Kovachki Theorem 11 after fixing the hidden
space and target admissibility side conditions. -/
def Equation6UniversalApproxUniform
    {A Hidden U : Type*} [PseudoMetricSpace U]
    (AdmissibleTarget : (A -> U) -> Prop) : Prop :=
  NeuralOperatorUniversalApproxUniform
    (Equation6NeuralOperatorClass A Hidden U)
    AdmissibleTarget

/-- Universal L2/Bochner approximation for the equation-(6) class. This is the
in-repo interface corresponding to Kovachki Theorem 13 after fixing the hidden
space and target admissibility side conditions. -/
def Equation6UniversalApproxL2
    {A Hidden U : Type*}
    (AdmissibleTarget : (A -> U) -> Prop)
    (l2Risk : (A -> U) -> (A -> U) -> ℝ) : Prop :=
  NeuralOperatorUniversalApproxL2
    (Equation6NeuralOperatorClass A Hidden U)
    AdmissibleTarget
    l2Risk

end ML
