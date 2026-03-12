import FormalProofs.DSL.TreeIPW
import Mathlib.Tactic

/-!
# FormalProofs/DSL/SectionIPWTwoStage.lean

Stage 3 uses a two-stage Bernoulli design:

1. sample documents with propensity `pi_doc`,
2. sample analysis sections within sampled documents with propensity `pi_sec`.

The implementation works with the joint propensity `pi_doc * pi_sec`. This file
records the elementary algebra behind that design and packages the existing
`TreeIPW` Horvitz-Thompson theorem under a Stage-3-oriented name.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal
open MeasureTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.DSL

section TwoStageSectionSampling

variable {Doc Section : Type*} [Fintype Doc] [Fintype Section]

abbrev DocSectionUnit (Doc Section : Type*) := Doc × Section

/-- Joint inclusion probability for document-then-section Bernoulli sampling. -/
def jointSectionPropensity
    (piDoc piSection : DocSectionUnit Doc Section → ℝ)
    (u : DocSectionUnit Doc Section) : ℝ :=
  piDoc u * piSection u

theorem jointSectionPropensity_pos
    (piDoc piSection : DocSectionUnit Doc Section → ℝ)
    (hDoc : ∀ u, 0 < piDoc u)
    (hSection : ∀ u, 0 < piSection u)
    (u : DocSectionUnit Doc Section) :
    0 < jointSectionPropensity piDoc piSection u := by
  dsimp [jointSectionPropensity]
  exact mul_pos (hDoc u) (hSection u)

theorem jointSectionPropensity_le_one
    (piDoc piSection : DocSectionUnit Doc Section → ℝ)
    (hDoc : ∀ u, piDoc u ≤ 1)
    (hSection : ∀ u, piSection u ≤ 1)
    (hDocNonneg : ∀ u, 0 ≤ piDoc u)
    (hSectionNonneg : ∀ u, 0 ≤ piSection u)
    (u : DocSectionUnit Doc Section) :
    jointSectionPropensity piDoc piSection u ≤ 1 := by
  dsimp [jointSectionPropensity]
  nlinarith [hDoc u, hSection u, hDocNonneg u, hSectionNonneg u]

end TwoStageSectionSampling

section TreeAlias

variable {Strings Node A : Type*} {k : ℕ}
variable [Fintype Strings] [Fintype Node] [Fintype A]
variable [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]

/-- Stage-3 name for the TreePO HT unbiasedness theorem on `(doc, analysis-section, group)` units. -/
theorem tokenWeighted_doc_section_ht_unbiased
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ)
    (pi : TreePopulation.TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    ∫ ω, htExpEstimator (p := TreePopulation.treeUnitPMF model) (pi := pi) (TreePopulation.treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss := by
  simpa using
    TreePopulation.ipw_preference_loss_connection_tree
      (model := model) (loss := loss) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)

end TreeAlias

end FormalProofs.DSL
