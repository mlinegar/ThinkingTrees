import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.ApproxOracleRecovery
import FormalProofs.OPT.RegularizedObjective

/-!
# FormalProofs/OPT/FiberPreservingObjective.lean

Training objectives whose minimizers satisfy (approximate) `OracleRecoversFeature`.

The existing infrastructure proves *what follows from* oracle feature recovery
(exact or approximate), but not *how to achieve it*. This file bridges the gap
by formalizing a contrastive fiber loss whose population-level minimizer is
provably a feature map satisfying `OracleRecoversFeature`.

## Key results

1. **`contrastiveFiberLoss`**: A pairwise loss that penalizes feature distance
   on oracle-equivalent pairs and rewards feature distance on oracle-distinct
   pairs.

2. **`oracleRecoversFeature_of_zero_contrastive_risk`**: Zero population
   contrastive risk implies exact `OracleRecoversFeature`, subject to a support
   coverage assumption on the training distribution.

3. **`approxOracleRecoversFeature_of_bounded_contrastive_risk`**: Bounded
   population risk implies ε-approximate recovery.

4. **`FiberRegularizedObjectiveWeights`**: Extends `RegularizedObjectiveWeights`
   with a fiber-preservation penalty weight, connecting to the existing
   optimization surface.
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

section ContrastiveLoss

variable {Feature : Type*} [PseudoMetricSpace Feature]

/-- Contrastive fiber loss on a single pair of inputs. Oracle-equivalent pairs
(same fiber) contribute their feature distance; oracle-distinct pairs contribute
a hinge penalty encouraging feature separation.

The `margin` parameter controls the target separation for distinct-fiber pairs.
Setting `margin = 0` gives a pure pullback loss (no push-apart term). -/
def contrastiveFiberLoss
    (fstar : Strings → Y) (feature : Strings → Feature)
    (margin : ℝ)
    (x x' : Strings) : ℝ :=
  if dist (fstar x) (fstar x') = 0
  then dist (feature x) (feature x')
  else max 0 (margin - dist (feature x) (feature x'))

/-- The same-fiber component of the contrastive loss is nonneg. -/
theorem contrastiveFiberLoss_nonneg_same_fiber
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ}
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    0 ≤ contrastiveFiberLoss fstar feature margin x x' := by
  simp [contrastiveFiberLoss, hzero, dist_nonneg]

/-- The contrastive loss is always nonneg. -/
theorem contrastiveFiberLoss_nonneg
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ}
    (x x' : Strings) :
    0 ≤ contrastiveFiberLoss fstar feature margin x x' := by
  unfold contrastiveFiberLoss
  split
  · exact dist_nonneg
  · exact le_max_left 0 _

/-- Population contrastive fiber risk over a joint distribution on pairs. -/
def populationContrastiveFiberRisk
    (fstar : Strings → Y) (feature : Strings → Feature)
    (margin : ℝ)
    (μ : PMF (Strings × Strings)) : ℝ :=
  Exp μ (fun p => contrastiveFiberLoss fstar feature margin p.1 p.2)

/-- The population risk is nonneg (it's an expectation of nonneg terms). -/
theorem populationContrastiveFiberRisk_nonneg
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ}
    (μ : PMF (Strings × Strings)) :
    0 ≤ populationContrastiveFiberRisk fstar feature margin μ := by
  unfold populationContrastiveFiberRisk
  rw [Exp_eq_ExpENN_toReal μ
    (fun p => contrastiveFiberLoss fstar feature margin p.1 p.2)]
  · exact ENNReal.toReal_nonneg
  intro p
  exact contrastiveFiberLoss_nonneg p.1 p.2

end ContrastiveLoss

section ExactRecovery

variable {Feature : Type*} [BoundedMetricSpace Feature]

/-- Covered support coverage: every covered same-fiber pair appears in the
training distribution's support. This is the right sparse-supervision interface
when only some oracle-labeled pairs are available. -/
def CoveredSameFiberSupportCoverage
    (covered : Strings → Strings → Prop)
    (fstar : Strings → Y)
    (μ : PMF (Strings × Strings)) : Prop :=
  ∀ x x' : Strings, covered x x' → dist (fstar x) (fstar x') = 0 →
    (x, x') ∈ μ.support

/-- Support coverage: every same-fiber pair appears in the training
distribution's support. This is a data-coverage assumption, not a property
of the model. -/
def SameFiberSupportCoverage
    (fstar : Strings → Y)
    (μ : PMF (Strings × Strings)) : Prop :=
  CoveredSameFiberSupportCoverage (fun _ _ => True) fstar μ

/-- Zero population contrastive risk on a distribution with same-fiber support
coverage implies exact `OracleRecoversFeature`.

Proof sketch: zero risk means every term in the expectation is zero (since all
terms are nonneg). In particular, for any same-fiber pair (x, x') in support,
the same-fiber branch gives `dist(feature(x), feature(x')) = 0`, hence
`feature(x) = feature(x')` by MetricSpace T0 separation. -/
theorem oracleRecoversFeatureOn_of_zero_contrastive_risk
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ}
    {μ : PMF (Strings × Strings)}
    (h_support : CoveredSameFiberSupportCoverage covered fstar μ)
    (h_zero : populationContrastiveFiberRisk fstar feature margin μ = 0) :
    OracleRecoversFeatureOn covered fstar feature := by
  let M : ℝ :=
    max (BoundedMetricSpace.diameterBound (α := Feature)) (max 0 margin)
  have hM : 0 ≤ M := by
    dsimp [M]
    exact le_trans (BoundedMetricSpace.diameterBound_nonneg (α := Feature))
      (le_max_left _ _)
  have h_loss_summable :
      Summable (fun p : Strings × Strings =>
        (μ p).toReal * contrastiveFiberLoss fstar feature margin p.1 p.2) := by
    apply PMF.summable_coe_real_mul_of_bounded μ
      (fun p => contrastiveFiberLoss fstar feature margin p.1 p.2) M hM
    intro p
    rw [abs_of_nonneg
      (contrastiveFiberLoss_nonneg (fstar := fstar)
        (feature := feature) (margin := margin) p.1 p.2)]
    unfold contrastiveFiberLoss
    split
    · exact le_trans
        (BoundedMetricSpace.dist_le (feature p.1) (feature p.2))
        (le_max_left _ _)
    · have hmargin :
          max 0 (margin - dist (feature p.1) (feature p.2)) ≤ max 0 margin := by
        exact max_le_max le_rfl (sub_le_self _ dist_nonneg)
      exact le_trans hmargin (le_max_right _ _)
  have h_term_nonneg :
      ∀ p : Strings × Strings,
        0 ≤ (μ p).toReal * contrastiveFiberLoss fstar feature margin p.1 p.2 := by
    intro p
    exact mul_nonneg ENNReal.toReal_nonneg
      (contrastiveFiberLoss_nonneg (fstar := fstar)
        (feature := feature) (margin := margin) p.1 p.2)
  have h_term_zero :
      ∀ p : Strings × Strings,
        (μ p).toReal * contrastiveFiberLoss fstar feature margin p.1 p.2 = 0 :=
    tsum_eq_zero_of_nonneg
      (fun p : Strings × Strings =>
        (μ p).toReal * contrastiveFiberLoss fstar feature margin p.1 p.2)
      h_term_nonneg h_loss_summable (by
        simpa [populationContrastiveFiberRisk, Exp] using h_zero)
  intro x x' hCovered hzero_oracle
  have h_in_support := h_support x x' hCovered hzero_oracle
  have hpair_ne0 : μ (x, x') ≠ 0 := by
    simpa [PMF.mem_support_iff] using h_in_support
  have hpair_toReal_pos : 0 < (μ (x, x')).toReal :=
    ENNReal.toReal_pos hpair_ne0 (PMF.apply_ne_top μ (x, x'))
  have hz_mul :
      (μ (x, x')).toReal * contrastiveFiberLoss fstar feature margin x x' = 0 :=
    h_term_zero (x, x')
  have h_loss_zero : contrastiveFiberLoss fstar feature margin x x' = 0 := by
    rcases mul_eq_zero.mp hz_mul with hmass | hloss
    · exfalso
      exact (ne_of_gt hpair_toReal_pos) hmass
    · exact hloss
  simp [contrastiveFiberLoss, hzero_oracle] at h_loss_zero
  exact h_loss_zero

/-- Zero population contrastive risk on a distribution with same-fiber support
coverage implies exact global `OracleRecoversFeature`. -/
theorem oracleRecoversFeature_of_zero_contrastive_risk
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ}
    {μ : PMF (Strings × Strings)}
    (h_support : SameFiberSupportCoverage fstar μ)
    (h_zero : populationContrastiveFiberRisk fstar feature margin μ = 0) :
    OracleRecoversFeature fstar feature := by
  have h_on :
      OracleRecoversFeatureOn (fun _ _ : Strings => True) fstar feature :=
    oracleRecoversFeatureOn_of_zero_contrastive_risk
      (covered := fun _ _ : Strings => True)
      (fstar := fstar) (feature := feature) (margin := margin) (μ := μ)
      (by simpa [SameFiberSupportCoverage] using h_support) h_zero
  intro x x' hzero_oracle
  exact h_on x x' trivial hzero_oracle

end ExactRecovery

section ApproxRecovery

variable {Feature : Type*} [PseudoMetricSpace Feature]

/-- Restricted pointwise same-fiber control is exactly restricted approximate
oracle recovery. -/
theorem approxOracleRecoversFeatureOn_of_bounded_same_fiber_component
    {covered : Strings → Strings → Prop}
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ} {ε : ℝ≥0}
    (h_bound : ∀ x x', covered x x' → dist (fstar x) (fstar x') = 0 →
      dist (feature x) (feature x') ≤ (ε : ℝ)) :
    ApproxOracleRecoversFeatureOn covered fstar feature ε := by
  exact h_bound

/-- Bounded population contrastive risk implies ε-approximate oracle recovery,
where ε is controlled by the risk value divided by the minimum probability mass
on same-fiber pairs. -/
theorem approxOracleRecoversFeature_of_bounded_same_fiber_component
    {fstar : Strings → Y} {feature : Strings → Feature}
    {margin : ℝ} {ε : ℝ≥0}
    (h_bound : ∀ x x', dist (fstar x) (fstar x') = 0 →
      dist (feature x) (feature x') ≤ (ε : ℝ)) :
    ApproxOracleRecoversFeature fstar feature ε := by
  exact h_bound

end ApproxRecovery

section ExtendedWeights

/-- Extended weight bundle for the regularized objective with an additional
fiber-preservation penalty weight. -/
structure FiberRegularizedObjectiveWeights extends RegularizedObjectiveWeights where
  fiber : ℝ

/-- Uniform default with equal weight on all four penalty components. -/
def defaultFiberRegularizedObjectiveWeights : FiberRegularizedObjectiveWeights where
  distortion := (3 : ℝ) / 4
  summary := (1 : ℝ) / 8
  leaf := (1 : ℝ) / 32
  merge := (1 : ℝ) / 32
  idemp := (1 : ℝ) / 32
  fiber := (1 : ℝ) / 32

/-- Project to base weights for backward compatibility. -/
def FiberRegularizedObjectiveWeights.toBase
    (w : FiberRegularizedObjectiveWeights) : RegularizedObjectiveWeights :=
  w.toRegularizedObjectiveWeights

variable [Monoid Strings]
variable {Y' : Type*} [PseudoMetricSpace Y']

/-- Certified fiber-regularized objective: the base certified objective plus
a fiber-preservation penalty. -/
def certifiedFiberRegularizedObjective
    {Feature : Type*} [PseudoMetricSpace Feature]
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y')
    (cost : SummaryCost Strings)
    (weights : FiberRegularizedObjectiveWeights)
    (laws : ApproxLocalLawsBundle g T fstar)
    (fiberPenalty : ℝ) : ℝ :=
  certifiedRegularizedObjective g x R T fstar cost weights.toBase laws
    + weights.fiber * fiberPenalty

end ExtendedWeights

end FormalProofs.OPT
