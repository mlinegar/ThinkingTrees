import FormalProofs.OPT.OracleFiberRelations
import FormalProofs.OPT.LipschitzReadoutFactorization
import FormalProofs.OPT.FiberPreservingObjective

/-!
# FormalProofs/OPT/SharedFeatureMultihead.lean

Approximate multi-head guarantees for a shared theorem feature `Φ`.

This file packages the regime that matters for learned tree systems:

- one learned theorem-bearing feature `Φ`,
- multiple downstream heads that only approximately factor through `Φ`, and
- quantitative stability of each head on oracle fibers and on exact theorem-
  backed tree reductions.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*}
variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Task Summary : Type*}
variable [PseudoMetricSpace Feature] [PseudoMetricSpace Task] [PseudoMetricSpace Summary]

/-- If two heads approximately factor through the same learned theorem feature,
then both heads are quantitatively stable on every oracle fiber. -/
theorem paired_approxReadoutBound_on_sameOracleFiber_of_sharedFeature
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {taskReadout : Strings → Task}
    {summaryReadout : Strings → Summary}
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Feature → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Feature → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover)
    {x x' : Strings}
    (hFiber : SameOracleFiber fstar x x') :
    dist (taskReadout x) (taskReadout x') ≤
        (L_task : ℝ) * (ε_fiber : ℝ) + 2 * (ε_task : ℝ) ∧
      dist (summaryReadout x) (summaryReadout x') ≤
        (L_summary : ℝ) * (ε_fiber : ℝ) + 2 * (ε_summary : ℝ) := by
  constructor
  · exact combined_readout_bound_on_oracle_fibers
      (hApproxRecover := hApproxRecover)
      (hApproxFactor := hTaskApproxFactor)
      (hLip := hTaskLip)
      hFiber
  · exact combined_readout_bound_on_oracle_fibers
      (hApproxRecover := hApproxRecover)
      (hApproxFactor := hSummaryApproxFactor)
      (hLip := hSummaryLip)
      hFiber

/-- Covered-pair version of the shared-feature bound. This is the natural form
when only a labeled pair relation is available. -/
theorem paired_approxReadoutBound_on_coveredSameOracleFiber_of_sharedFeature
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {taskReadout : Strings → Task}
    {summaryReadout : Strings → Summary}
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeatureOn covered fstar feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Feature → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Feature → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover)
    {x x' : Strings}
    (hCovered : covered x x')
    (hFiber : SameOracleFiber fstar x x') :
    dist (taskReadout x) (taskReadout x') ≤
        (L_task : ℝ) * (ε_fiber : ℝ) + 2 * (ε_task : ℝ) ∧
      dist (summaryReadout x) (summaryReadout x') ≤
        (L_summary : ℝ) * (ε_fiber : ℝ) + 2 * (ε_summary : ℝ) := by
  constructor
  · exact combined_readout_bound_on_covered_oracle_fibers
      (hApproxRecover := hApproxRecover)
      (hApproxFactor := hTaskApproxFactor)
      (hLip := hTaskLip)
      hCovered hFiber
  · exact combined_readout_bound_on_covered_oracle_fibers
      (hApproxRecover := hApproxRecover)
      (hApproxFactor := hSummaryApproxFactor)
      (hLip := hSummaryLip)
      hCovered hFiber

/-- The previous paired oracle-fiber bound can be read as approximate oracle
recovery for both heads simultaneously. -/
theorem paired_approxOracleRecoversReadouts_of_sharedFeature
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {taskReadout : Strings → Task}
    {summaryReadout : Strings → Summary}
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Feature → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Feature → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover) :
    ApproxOracleRecoversFeature fstar taskReadout
        (L_task * ε_fiber + 2 * ε_task) ∧
      ApproxOracleRecoversFeature fstar summaryReadout
        (L_summary * ε_fiber + 2 * ε_summary) := by
  constructor
  · intro x x' hFiber
    exact (paired_approxReadoutBound_on_sameOracleFiber_of_sharedFeature
      (hApproxRecover := hApproxRecover)
      (hTaskApproxFactor := hTaskApproxFactor)
      (hTaskLip := hTaskLip)
      (hSummaryApproxFactor := hSummaryApproxFactor)
      (hSummaryLip := hSummaryLip)
      hFiber).1
  · intro x x' hFiber
    exact (paired_approxReadoutBound_on_sameOracleFiber_of_sharedFeature
      (hApproxRecover := hApproxRecover)
      (hTaskApproxFactor := hTaskApproxFactor)
      (hTaskLip := hTaskLip)
      (hSummaryApproxFactor := hSummaryApproxFactor)
      (hSummaryLip := hSummaryLip)
      hFiber).2

/-- Covered-pair version of simultaneous approximate oracle recovery for both
heads through the same theorem feature. -/
theorem paired_approxOracleRecoversReadoutsOn_of_sharedFeature
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {taskReadout : Strings → Task}
    {summaryReadout : Strings → Summary}
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeatureOn covered fstar feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Feature → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Feature → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover) :
    ApproxOracleRecoversFeatureOn covered fstar taskReadout
        (L_task * ε_fiber + 2 * ε_task) ∧
      ApproxOracleRecoversFeatureOn covered fstar summaryReadout
        (L_summary * ε_fiber + 2 * ε_summary) := by
  constructor
  · intro x x' hCovered hFiber
    exact (paired_approxReadoutBound_on_coveredSameOracleFiber_of_sharedFeature
      (hApproxRecover := hApproxRecover)
      (hTaskApproxFactor := hTaskApproxFactor)
      (hTaskLip := hTaskLip)
      (hSummaryApproxFactor := hSummaryApproxFactor)
      (hSummaryLip := hSummaryLip)
      hCovered hFiber).1
  · intro x x' hCovered hFiber
    exact (paired_approxReadoutBound_on_coveredSameOracleFiber_of_sharedFeature
      (hApproxRecover := hApproxRecover)
      (hTaskApproxFactor := hTaskApproxFactor)
      (hTaskLip := hTaskLip)
      (hSummaryApproxFactor := hSummaryApproxFactor)
      (hSummaryLip := hSummaryLip)
      hCovered hFiber).2

section ExactBacking

variable [Monoid Strings]

/-- Under exact theorem-backed reduction, every realized `ZR` summary inherits
the paired approximate readout bound relative to the original document. -/
theorem zr_support_paired_approxReadoutBound_of_exactTheoremBacked_and_sharedFeature
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {taskReadout : Strings → Task}
    {summaryReadout : Strings → Summary}
    {g : Summarizer Strings} {x : Strings} {R : ℕ} {T : BinTree Strings}
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Feature → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Feature → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover)
    {z : Strings}
    (hz : z ∈ (ZR g x R T).support) :
    dist (taskReadout z) (taskReadout x) ≤
        (L_task : ℝ) * (ε_fiber : ℝ) + 2 * (ε_task : ℝ) ∧
      dist (summaryReadout z) (summaryReadout x) ≤
        (L_summary : ℝ) * (ε_fiber : ℝ) + 2 * (ε_summary : ℝ) := by
  exact paired_approxReadoutBound_on_sameOracleFiber_of_sharedFeature
    (hApproxRecover := hApproxRecover)
    (hTaskApproxFactor := hTaskApproxFactor)
    (hTaskLip := hTaskLip)
    (hSummaryApproxFactor := hSummaryApproxFactor)
    (hSummaryLip := hSummaryLip)
    (zr_support_sameOracleFiber_of_exactTheoremBacked
      (hp := hp) (hExact := hExact) (hR := hR) hz)

/-- Covered-pair version of the exact-theorem-backed `ZR` support result. This
lets one use a sparse covered relation together with exact theorem-backedness. -/
theorem zr_support_paired_approxReadoutBound_of_exactTheoremBacked_and_sharedFeature_on
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → Feature}
    {taskReadout : Strings → Task}
    {summaryReadout : Strings → Summary}
    {g : Summarizer Strings} {x : Strings} {R : ℕ} {T : BinTree Strings}
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeatureOn covered fstar feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Feature → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Feature → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover)
    (hCoveredSupport : ∀ {z : Strings}, z ∈ (ZR g x R T).support → covered z x)
    {z : Strings}
    (hz : z ∈ (ZR g x R T).support) :
    dist (taskReadout z) (taskReadout x) ≤
        (L_task : ℝ) * (ε_fiber : ℝ) + 2 * (ε_task : ℝ) ∧
      dist (summaryReadout z) (summaryReadout x) ≤
        (L_summary : ℝ) * (ε_fiber : ℝ) + 2 * (ε_summary : ℝ) := by
  exact paired_approxReadoutBound_on_coveredSameOracleFiber_of_sharedFeature
    (hApproxRecover := hApproxRecover)
    (hTaskApproxFactor := hTaskApproxFactor)
    (hTaskLip := hTaskLip)
    (hSummaryApproxFactor := hSummaryApproxFactor)
    (hSummaryLip := hSummaryLip)
    (hCovered := hCoveredSupport hz)
    (hFiber := zr_support_sameOracleFiber_of_exactTheoremBacked
      (hp := hp) (hExact := hExact) (hR := hR) hz)

end ExactBacking

section CoveredContrastive

variable {FeatureC TaskC SummaryC : Type*}
variable [BoundedMetricSpace FeatureC]
variable [PseudoMetricSpace TaskC] [PseudoMetricSpace SummaryC]

/-- Zero contrastive risk on a covered pair distribution forces the shared
theorem feature to collapse each covered oracle fiber, so any approximately
factored task and summary heads are stable on those covered fibers. -/
theorem paired_approxReadoutBound_on_coveredSameOracleFiber_of_zero_contrastiveRisk
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y}
    {feature : Strings → FeatureC}
    {taskReadout : Strings → TaskC}
    {summaryReadout : Strings → SummaryC}
    {μ : PMF (Strings × Strings)}
    {margin : ℝ}
    {ε_task ε_summary : ℝ≥0}
    (hSupport : CoveredSameFiberSupportCoverage covered fstar μ)
    (hZero : populationContrastiveFiberRisk fstar feature margin μ = 0)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature feature taskReadout ε_task)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature feature summaryReadout ε_summary)
    {x x' : Strings}
    (hCovered : covered x x')
    (hFiber : SameOracleFiber fstar x x') :
    dist (taskReadout x) (taskReadout x') ≤ 2 * (ε_task : ℝ) ∧
      dist (summaryReadout x) (summaryReadout x') ≤ 2 * (ε_summary : ℝ) := by
  have hRecoverOn : OracleRecoversFeatureOn covered fstar feature :=
    oracleRecoversFeatureOn_of_zero_contrastive_risk
      (covered := covered) (fstar := fstar) (feature := feature)
      (margin := margin) (μ := μ) hSupport hZero
  have hEq : feature x = feature x' := hRecoverOn x x' hCovered hFiber
  constructor
  · exact approxReadoutFactorsThroughFeature_fiber_bound
      (h := hTaskApproxFactor) hEq
  · exact approxReadoutFactorsThroughFeature_fiber_bound
      (h := hSummaryApproxFactor) hEq

end CoveredContrastive

end FormalProofs.OPT
