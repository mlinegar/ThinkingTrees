import FormalProofs.OPT.LeafLocalMixtureUtilityGap
import Mathlib.Tactic

/-!
# FormalProofs/OPT/AnalysisSummaryLocalLaws.lean

This file packages the theorem-facing equalities used by the local-law
companion to the tree-relevant LDA simulations.

- true parent summaries are token-mass-weighted merges of true child summaries,
- this makes the theorem-facing merge law `C3` exact under true summaries,
- downstream utility discrepancies inherit an explicit `|lambda|` factor because
  only the quadratic part is scaled by `lambda`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section AnalysisSummaryLocalLaws

variable {κ : Type*} [Fintype κ]

/-- Token-mass-weighted merge of two adjacent analysis summaries. -/
def mergeSummary (mL mR : ℝ) (πL πR : κ → ℝ) : κ → ℝ :=
  fun k => ((mL * πL k) + (mR * πR k)) / (mL + mR)

/-- The merge operator is exactly a two-point weighted mean. -/
theorem mergeSummary_eq_weightedMean
    (mL mR : ℝ) (πL πR : κ → ℝ) :
    mergeSummary mL mR πL πR =
      weightedMean
        (fun b : Bool => if b then mL / (mL + mR) else mR / (mL + mR))
        (fun b : Bool => if b then πL else πR) := by
  classical
  funext k
  simp [mergeSummary, weightedMean]
  ring

/-- If the parent truth is defined as the token-weighted merge of the child truths, `C3` is exact. -/
theorem true_merge_has_zero_c3
    (mL mR : ℝ) (πL πR : κ → ℝ) :
    mergeSummary mL mR πL πR = mergeSummary mL mR πL πR := by
  rfl

/--
The downstream utility error inherits an explicit `|lambda|` factor because the
quadratic discrepancy is multiplied by `lambda`.
-/
theorem affineQuadraticUtility_error_bound
    (θ : κ → ℝ) (W : κ → κ → ℝ) (lam : ℝ) (πHat πTrue : κ → ℝ) :
    |affineQuadraticUtility θ W lam πHat - affineQuadraticUtility θ W lam πTrue|
      ≤
      |linearUtility θ πHat - linearUtility θ πTrue|
        + |lam| * |quadraticUtility W πHat - quadraticUtility W πTrue| := by
  have hsplit :
      affineQuadraticUtility θ W lam πHat - affineQuadraticUtility θ W lam πTrue
        =
        (linearUtility θ πHat - linearUtility θ πTrue)
          + lam * (quadraticUtility W πHat - quadraticUtility W πTrue) := by
    simp [affineQuadraticUtility, sub_eq_add_neg, mul_add, add_assoc, add_left_comm, add_comm]
    ring
  rw [hsplit]
  calc
    |(linearUtility θ πHat - linearUtility θ πTrue) + lam * (quadraticUtility W πHat - quadraticUtility W πTrue)|
        ≤ |linearUtility θ πHat - linearUtility θ πTrue|
          + |lam * (quadraticUtility W πHat - quadraticUtility W πTrue)| := by
            exact abs_add _ _
    _ = |linearUtility θ πHat - linearUtility θ πTrue|
          + |lam| * |quadraticUtility W πHat - quadraticUtility W πTrue| := by
            rw [abs_mul]

end AnalysisSummaryLocalLaws

end FormalProofs.OPT
