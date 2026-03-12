import Mathlib.MeasureTheory.Measure.NullMeasurable
import Mathlib.MeasureTheory.Measure.CharacteristicFunction
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Density

import FormalProofs.CLT.Core

/-!
# FormalProofs/CLT/ProbabilityLaws.lean

Basic probability-law lemmas for random variables:
- laws (pushforward measures)
- law of total probability
- transfer theorem (expectation under the law)
- joint law for independent random variables
- absolute continuity of laws via densities
- characteristic function of a random variable
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

/-!
## Laws of random variables
-/

/-- Law (pushforward measure) of a random variable. -/
def law {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha]
    (mu : Measure Omega) (X : Omega → Alpha) : Measure Alpha :=
  Measure.map X mu

/-- Joint law of two random variables. -/
def jointLaw {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha]
    {Beta : Type*} [MeasurableSpace Beta]
    (mu : Measure Omega) (X : Omega → Alpha) (Y : Omega → Beta) : Measure (Alpha × Beta) :=
  Measure.map (fun omega => (X omega, Y omega)) mu

/-- Characteristic function of a real-valued random variable. -/
def charFunLaw {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) (X : Omega → ℝ) (t : ℝ) : ℂ :=
  charFun (law mu X) t

lemma law_isProbabilityMeasure {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {X : Omega → Alpha} (hX : AEMeasurable X mu) :
    IsProbabilityMeasure (law mu X) := by
  simpa [law] using (Measure.isProbabilityMeasure_map (μ := mu) hX)

lemma law_absolutelyContinuous_of_hasPDF {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha]
    (mu : Measure Omega) {X : Omega → Alpha} {nu : Measure Alpha}
    [MeasureTheory.HasPDF X mu nu] :
    law mu X ≪ nu := by
  simpa [law] using
    (MeasureTheory.HasPDF.absolutelyContinuous (X := X))

/-!
## Transfer theorem (Law of the unconscious statistician)
-/

lemma transfer_theorem {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {X : Omega → Alpha} (hX : AEMeasurable X mu)
    {g : Alpha → ℝ} (hg : AEStronglyMeasurable g (law mu X)) :
    expectation mu (fun omega => g (X omega)) =
      ∫ x, g x ∂(law mu X) := by
  have h := integral_map hX hg
  simpa [expectation, law] using h.symm

/-!
## Joint law for independent random variables
-/

lemma jointLaw_eq_prod_law {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha]
    {Beta : Type*} [MeasurableSpace Beta]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {X : Omega → Alpha} {Y : Omega → Beta}
    (hX : AEMeasurable X mu) (hY : AEMeasurable Y mu) (hXY : X ⟂ᵢ[mu] Y) :
    jointLaw mu X Y = (law mu X).prod (law mu Y) := by
  simpa [jointLaw, law] using
    (indepFun_iff_map_prod_eq_prod_map_map hX hY).1 hXY

/-!
## Law of total probability
-/

lemma law_of_total_probability {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {Iota : Type*} [Countable Iota]
    {A : Set Omega} (hA : MeasurableSet A)
    {B : Iota → Set Omega} (hB : ∀ i, MeasurableSet (B i))
    (hdisj : Pairwise (fun i j => Disjoint (B i) (B j)))
    (hcover : (⋃ i, B i) = Set.univ) :
    mu A = ∑' i, mu (A ∩ B i) := by
  have hdisj' : Pairwise (fun i j => Disjoint (A ∩ B i) (A ∩ B j)) := by
    intro i j hij
    refine Set.disjoint_left.2 ?_
    intro x hx hy
    exact (Set.disjoint_left.1 (hdisj hij)) hx.2 hy.2
  have hmeas : ∀ i, MeasurableSet (A ∩ B i) := fun i => hA.inter (hB i)
  calc
    mu A = mu (A ∩ Set.univ) := by simp
    _ = mu (A ∩ ⋃ i, B i) := by simp [hcover]
    _ = mu (⋃ i, A ∩ B i) := by simp [Set.inter_iUnion]
    _ = ∑' i, mu (A ∩ B i) := by
      simpa using (MeasureTheory.measure_iUnion hdisj' hmeas)

end ProbabilityTheory
