import FormalProofs.OPT.LeafLocalMixtureUtilityGap
import Mathlib.Tactic

/-!
# FormalProofs/OPT/AnalysisPartitionMismatch.lean

Stage 3 inserts an explicit analysis partition between the latent local mixtures
and the estimator. This file records the exact overlap-operator identities used by
the mismatch suite:

- analysis sections are overlap-weighted averages of latent section mixtures,
- the pooled-vs-analysis gap is still carried entirely by the quadratic term,
- `lambda = 0` collapses every mismatch gap,
- exact alignment gives zero mismatch gap for all `lambda`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators

set_option maxHeartbeats 200000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section AnalysisPartition

variable {ι α κ : Type*} [Fintype ι] [Fintype α] [Fintype κ]

/-- Overlap-induced analysis mixture: average latent mixtures with analysis weights `ρ a`. -/
def analysisMixture (ρ : α → ι → ℝ) (π : ι → κ → ℝ) : α → κ → ℝ :=
  fun a => weightedMean (ρ a) π

/-- Exact alignment overlap: each analysis section selects one latent section. -/
def identityOverlap (e : α ≃ ι) : α → ι → ℝ :=
  fun a b => if e a = b then 1 else 0

theorem analysisMixture_identityOverlap
    (e : α ≃ ι) (π : ι → κ → ℝ) :
    analysisMixture (identityOverlap e) π = fun a => π (e a) := by
  classical
  funext a
  funext k
  simp [analysisMixture, identityOverlap, weightedMean]

/-- The analysis-partition gap is still carried entirely by the quadratic term. -/
theorem analysis_gap_eq_quadratic_gap
    (ω : α → ℝ) (ρ : α → ι → ℝ) (π : ι → κ → ℝ) (θ : κ → ℝ) (W : κ → κ → ℝ) (lam : ℝ) :
    (∑ a : α, ω a * affineQuadraticUtility θ W lam (analysisMixture ρ π a))
      - affineQuadraticUtility θ W lam (weightedMean ω (analysisMixture ρ π)) =
      lam *
        ((∑ a : α, ω a * quadraticUtility W (analysisMixture ρ π a))
          - quadraticUtility W (weightedMean ω (analysisMixture ρ π))) := by
  simpa [analysisMixture] using
    affineQuadratic_gap_eq_quadratic_gap
      (ω := ω) (π := analysisMixture ρ π) (θ := θ) (W := W) (lam := lam)

/-- Any mismatch gap disappears when `lambda = 0`. -/
theorem analysis_gap_zero_lambda
    (ω : α → ℝ) (ρ : α → ι → ℝ) (π : ι → κ → ℝ) (θ : κ → ℝ) (W : κ → κ → ℝ) :
    (∑ a : α, ω a * affineQuadraticUtility θ W 0 (analysisMixture ρ π a))
      - affineQuadraticUtility θ W 0 (weightedMean ω (analysisMixture ρ π)) = 0 := by
  simpa [analysisMixture] using
    affineQuadratic_gap_zero_lambda
      (ω := ω) (π := analysisMixture ρ π) (θ := θ) (W := W)

/-- Under exact alignment, the analysis-partition target equals the latent target. -/
theorem aligned_partition_zero_gap
    (e : α ≃ ι) (ω : α → ℝ) (π : ι → κ → ℝ) (θ : κ → ℝ) (W : κ → κ → ℝ) (lam : ℝ) :
    (∑ a : α, ω a * affineQuadraticUtility θ W lam (analysisMixture (identityOverlap e) π a))
      =
      ∑ a : α, ω a * affineQuadraticUtility θ W lam (π (e a)) := by
  simp [analysisMixture_identityOverlap]

end AnalysisPartition

end FormalProofs.OPT
