import FormalProofs.ML.TransformerAsNeuralOperator
import FormalProbability.ML.NeuralOperatorFNOCore

/-!
# FormalProofs/ML/FNOFormalization.lean

ThinkingTrees-facing Fourier neural-operator formalization.

The reusable low-dependency FNO development lives in
`FormalProbability.ML.NeuralOperatorFNOCore`.  This module gives C-TreePO and
Semantic Forests code a stable local bridge to the parts we use most often:
finite-mode FNO formula classes, truncation-tail approximation certificates,
equation-(6) neural-operator routes, direct `L2_mu` routes, and transformer/FNO
envelopes.
-/

set_option linter.mathlibStandardSet false

open scoped Classical BigOperators

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ML

namespace FNOFormalization

/-! ## Local aliases for reusable FormalProbability FNO surfaces -/

abbrev AddCircleFNOFormula (T : ℝ) [Fact (0 < T)] :=
  NeuralOperators.FNOCore.AddCircleFNOFourierMultiplierFormula T

abbrev TorusFNOFormula (d : Type*) [Fintype d] :=
  NeuralOperators.FNOCore.TorusFNOFourierMultiplierFormula d

abbrev FNOClass (D V : Type*) : Nat -> Set ((D -> V) -> D -> V) :=
  fun _n : Nat => NeuralOperators.FNOCore.FNOLayerClass D V

abbrev AddCircleFNOClass (T : ℝ) [Fact (0 < T)] :
    Nat -> Set ((AddCircle T -> ℂ) -> AddCircle T -> ℂ) :=
  FNOClass (AddCircle T) ℂ

abbrev TorusFNOClass (d : Type*) [Fintype d] :
    Nat -> Set ((UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ) :=
  FNOClass (UnitAddTorus d) ℂ

abbrev AddCircleFNOTruncationTailSource
    (T : ℝ) [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    (target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ) :=
  NeuralOperators.FNOCore.AddCircleFNOTruncationTailSource T target

abbrev TorusFNOTruncationTailSource
    (d : Type*) [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    (target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ) :=
  NeuralOperators.FNOCore.TorusFNOTruncationTailSource d target

/-! ## Additive-circle FNO bridge -/

/-- ThinkingTrees-facing certificate for an additive-circle FNO truncation
route.  The single field is intentionally the reusable FormalProbability
source; this module only provides local projections and aliases. -/
structure AddCircleFNOFormalization
    (T : ℝ) [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    (target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ) where
  tailSource : AddCircleFNOTruncationTailSource T target

namespace AddCircleFNOFormalization

def toTheorem11Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    (H : AddCircleFNOFormalization T target) :
    NeuralOperators.FNOCore.Theorem11Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (AddCircleFNOClass T) target where
  obligations := H.tailSource.toTheorem11Obligations

def toTheorem12Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    (H : AddCircleFNOFormalization T target) :
    NeuralOperators.FNOCore.Theorem12Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (AddCircleFNOClass T) target where
  obligations := H.tailSource.toTheorem12Obligations

def toEquation6Theorem11Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    (H : AddCircleFNOFormalization T target) :
    NeuralOperators.FNOCore.Theorem11Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (fun _n : Nat =>
        NeuralOperators.FNOCore.Equation6NeuralOperatorClass
          (AddCircle T -> ℂ) (AddCircle T -> ℂ) (AddCircle T -> ℂ)) target where
  obligations := H.tailSource.toEquation6Theorem11Obligations

def toEquation6Theorem12Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    (H : AddCircleFNOFormalization T target) :
    NeuralOperators.FNOCore.Theorem12Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (fun _n : Nat =>
        NeuralOperators.FNOCore.Equation6NeuralOperatorClass
          (AddCircle T -> ℂ) (AddCircle T -> ℂ) (AddCircle T -> ℂ)) target where
  obligations := H.tailSource.toEquation6Theorem12Obligations

def toTransformerEnvelopeTheorem11Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    (H : AddCircleFNOFormalization T target)
    (TransformerClass :
      Nat -> Set ((AddCircle T -> ℂ) -> AddCircle T -> ℂ)) :
    NeuralOperators.FNOCore.Theorem11Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (NeuralOperators.FNOCore.FNOTransformerEnvelopeClass (AddCircle T) ℂ TransformerClass) target where
  obligations :=
    H.tailSource.toFNOTransformerEnvelopeTheorem11Obligations TransformerClass

def toTransformerEnvelopeTheorem12Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    (H : AddCircleFNOFormalization T target)
    (TransformerClass :
      Nat -> Set ((AddCircle T -> ℂ) -> AddCircle T -> ℂ)) :
    NeuralOperators.FNOCore.Theorem12Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (NeuralOperators.FNOCore.FNOTransformerEnvelopeClass (AddCircle T) ℂ TransformerClass) target where
  obligations :=
    H.tailSource.toFNOTransformerEnvelopeTheorem12Obligations TransformerClass

def toTheorem13Assumptions
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    [MeasurableSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    {μ : MeasureTheory.Measure (AddCircle T -> ℂ)}
    (H : AddCircleFNOFormalization T target)
    (measure : NeuralOperators.FNOCore.PaperMeasureAssumptions
      (AddCircle T -> ℂ) (AddCircle T -> ℂ) target μ)
    (tail : NeuralOperators.FNOCore.DirectCompactTailSkeleton
      (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (AddCircleFNOClass T) target μ) :
    NeuralOperators.FNOCore.Theorem13Assumptions (AddCircle T -> ℂ) (AddCircle T -> ℂ)
      (AddCircleFNOClass T) target μ where
  measure := measure
  theorem11 := H.toTheorem11Assumptions
  tail := tail

theorem directL2RiskBochner_le_tailBound
    {T : ℝ} [Fact (0 < T)] [PseudoMetricSpace (AddCircle T -> ℂ)]
    [MeasurableSpace (AddCircle T -> ℂ)]
    {target : (AddCircle T -> ℂ) -> AddCircle T -> ℂ}
    {μ : MeasureTheory.Measure (AddCircle T -> ℂ)}
    [MeasureTheory.IsProbabilityMeasure μ]
    (H : AddCircleFNOFormalization T target)
    (n : Nat) :
    NeuralOperators.FNOCore.DirectL2RiskBochner μ target ((H.tailSource.approximant n).displayedMap)
      <= H.tailSource.tailBound n :=
  H.tailSource.directL2RiskBochner_le_tailBound (μ := μ) n

end AddCircleFNOFormalization

/-! ## Torus FNO bridge -/

/-- ThinkingTrees-facing certificate for a torus FNO truncation route. -/
structure TorusFNOFormalization
    (d : Type*) [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    (target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ) where
  tailSource : TorusFNOTruncationTailSource d target

namespace TorusFNOFormalization

def toTheorem11Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    (H : TorusFNOFormalization d target) :
    NeuralOperators.FNOCore.Theorem11Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (TorusFNOClass d) target where
  obligations := H.tailSource.toTheorem11Obligations

def toTheorem12Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    (H : TorusFNOFormalization d target) :
    NeuralOperators.FNOCore.Theorem12Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (TorusFNOClass d) target where
  obligations := H.tailSource.toTheorem12Obligations

def toEquation6Theorem11Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    (H : TorusFNOFormalization d target) :
    NeuralOperators.FNOCore.Theorem11Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (fun _n : Nat =>
        NeuralOperators.FNOCore.Equation6NeuralOperatorClass
          (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)) target where
  obligations := H.tailSource.toEquation6Theorem11Obligations

def toEquation6Theorem12Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    (H : TorusFNOFormalization d target) :
    NeuralOperators.FNOCore.Theorem12Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (fun _n : Nat =>
        NeuralOperators.FNOCore.Equation6NeuralOperatorClass
          (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)) target where
  obligations := H.tailSource.toEquation6Theorem12Obligations

def toTransformerEnvelopeTheorem11Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    (H : TorusFNOFormalization d target)
    (TransformerClass :
      Nat -> Set ((UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ)) :
    NeuralOperators.FNOCore.Theorem11Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (NeuralOperators.FNOCore.FNOTransformerEnvelopeClass (UnitAddTorus d) ℂ TransformerClass) target where
  obligations :=
    H.tailSource.toFNOTransformerEnvelopeTheorem11Obligations TransformerClass

def toTransformerEnvelopeTheorem12Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    (H : TorusFNOFormalization d target)
    (TransformerClass :
      Nat -> Set ((UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ)) :
    NeuralOperators.FNOCore.Theorem12Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (NeuralOperators.FNOCore.FNOTransformerEnvelopeClass (UnitAddTorus d) ℂ TransformerClass) target where
  obligations :=
    H.tailSource.toFNOTransformerEnvelopeTheorem12Obligations TransformerClass

def toTheorem13Assumptions
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    [MeasurableSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    {μ : MeasureTheory.Measure (UnitAddTorus d -> ℂ)}
    (H : TorusFNOFormalization d target)
    (measure : NeuralOperators.FNOCore.PaperMeasureAssumptions
      (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ) target μ)
    (tail : NeuralOperators.FNOCore.DirectCompactTailSkeleton
      (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (TorusFNOClass d) target μ) :
    NeuralOperators.FNOCore.Theorem13Assumptions (UnitAddTorus d -> ℂ) (UnitAddTorus d -> ℂ)
      (TorusFNOClass d) target μ where
  measure := measure
  theorem11 := H.toTheorem11Assumptions
  tail := tail

theorem directL2RiskBochner_le_tailBound
    {d : Type*} [Fintype d] [PseudoMetricSpace (UnitAddTorus d -> ℂ)]
    [MeasurableSpace (UnitAddTorus d -> ℂ)]
    {target : (UnitAddTorus d -> ℂ) -> UnitAddTorus d -> ℂ}
    {μ : MeasureTheory.Measure (UnitAddTorus d -> ℂ)}
    [MeasureTheory.IsProbabilityMeasure μ]
    (H : TorusFNOFormalization d target)
    (n : Nat) :
    NeuralOperators.FNOCore.DirectL2RiskBochner μ target ((H.tailSource.approximant n).displayedMap)
      <= H.tailSource.tailBound n :=
  H.tailSource.directL2RiskBochner_le_tailBound (μ := μ) n

end TorusFNOFormalization

end FNOFormalization

end ML

end
