import Mathlib.Probability.StrongLaw

import FormalProofs.CLT.Core

/-!
# FormalProofs/CLT/WeakLaw.lean

Weak law of large numbers derived from the strong law.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped Topology

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ProbabilityTheory

open MeasureTheory
open Filter

lemma weak_law_of_large_numbers {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    (X : ℕ → Omega → ℝ) (hIID : IID mu X) (hInt : Integrable (X 0) mu) :
    ConvergesInProbability mu (fun n => sampleMean X n) (fun _ => expectation mu (X 0)) := by
  have hpair : Pairwise (fun i j => X i ⟂ᵢ[mu] X j) := by
    intro i j hij
    exact (hIID.1.indepFun hij)
  have hident : ∀ i, IdentDistrib (X i) (X 0) mu mu := hIID.2
  have h_ae :=
    (strong_law_ae (X := X) (μ := mu) hInt hpair hident)
  have h_ae' :
      ∀ᵐ omega ∂mu,
        Tendsto (fun n : ℕ => sampleMean X n omega) atTop
          (𝓝 (expectation mu (X 0))) := by
    simpa [sampleMean, expectation, smul_eq_mul] using h_ae
  have hInt_i : ∀ i, Integrable (X i) mu := fun i => (hident i).integrable_iff.2 hInt
  have hAesm : ∀ i, AEStronglyMeasurable (X i) mu := fun i => (hInt_i i).aestronglyMeasurable
  have hAesm_mean : ∀ n, AEStronglyMeasurable (sampleMean X n) mu := by
    intro n
    classical
    have hsum_ae :
        AEMeasurable (fun omega => ∑ i ∈ Finset.range n, X i omega) mu := by
      refine Finset.aemeasurable_fun_sum _ ?_
      intro i _hi
      exact (hAesm i).aemeasurable
    have hsum :
        AEStronglyMeasurable (fun omega => ∑ i ∈ Finset.range n, X i omega) mu :=
      hsum_ae.aestronglyMeasurable
    simpa [sampleMean, smul_eq_mul] using hsum.const_smul ((n : ℝ)⁻¹)
  exact tendstoInMeasure_of_tendsto_ae hAesm_mean h_ae'

end ProbabilityTheory
