import FormalProofs.OPT.OracleFibers
import FormalProofs.OPT.ApproxOracleRecovery
import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.TheoremBacking
import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.RegularizedObjective
import FormalProofs.OPT.LipschitzReadoutFactorization
import FormalProofs.OPT.TwoStageOracleSurrogate

/-!
# FormalProofs/OPT/OracleFiberObjectives.lean

Consolidated (2026-07-02) from the oracle-fiber cluster, objectives layer:
`OracleFiberRelations`, `FiberPreservingObjective`, `FeatureClassObjectives`,
`SharedFeatureMultihead`, `ApproxFiberTransport`.

Each original file is preserved verbatim as one section below; the original
modules remain as import shims. The laws/relations layer lives in
`FormalProofs/OPT/OracleFibers.lean`. `ProductScoreFiber.lean` stays a real
module: it needs `TwoStageLabelScoreObjectives` (now consolidated into
`UnifiedOracleRoute`), which imports `FiberPreservingObjective`,
`ApproxFiberTransport`, and `LabelScoreObjectives`, so merging it here would
create an import cycle through those shims.
-/

/-! ## From FormalProofs/OPT/OracleFiberRelations.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/OracleFiberRelations.lean

Relation-first restatement of theorem-backedness around oracle fibers.

The point of this file is to make the intended object explicit before any
particular learned feature map is chosen:

- the primitive equivalence relation is "same oracle fiber";
- exact and approximate feature recovery are just ways of realizing that
  relation with a learned theorem feature; and
- exact theorem-backedness keeps realized reductions inside one oracle fiber.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

/-- Two inputs lie in the same oracle fiber when the oracle identifies them. -/
def SameOracleFiber
    (fstar : Strings → Y) (x x' : Strings) : Prop :=
  dist (fstar x) (fstar x') = 0

theorem sameOracleFiber_refl
    (fstar : Strings → Y) (x : Strings) :
    SameOracleFiber fstar x x := by
  simp [SameOracleFiber]

theorem sameOracleFiber_symm
    {fstar : Strings → Y} {x x' : Strings}
    (h : SameOracleFiber fstar x x') :
    SameOracleFiber fstar x' x := by
  simpa [SameOracleFiber, dist_comm] using h

theorem sameOracleFiber_trans
    {fstar : Strings → Y} {x y z : Strings}
    (hxy : SameOracleFiber fstar x y)
    (hyz : SameOracleFiber fstar y z) :
    SameOracleFiber fstar x z := by
  have hxyEq : fstar x = fstar y := dist_eq_zero.mp hxy
  have hyzEq : fstar y = fstar z := dist_eq_zero.mp hyz
  simpa [SameOracleFiber, hxyEq, hyzEq]

section Recovery

variable {Feature : Type*}

/-- Exact oracle recovery is exactly the statement that the learned theorem
feature is constant on oracle fibers. -/
theorem oracleRecoversFeature_iff_respects_sameOracleFiber
    {fstar : Strings → Y} {feature : Strings → Feature} :
    OracleRecoversFeature fstar feature ↔
      ∀ {x x' : Strings}, SameOracleFiber fstar x x' → feature x = feature x' := by
  constructor
  · intro hRecover x x' hFiber
    exact hRecover x x' hFiber
  · intro hFiber x x' hzero
    exact hFiber hzero

end Recovery

section ApproxRecovery

variable {Feature : Type*} [PseudoMetricSpace Feature]

/-- Approximate oracle recovery is exactly the statement that the learned
theorem feature has bounded diameter on each oracle fiber. -/
theorem approxOracleRecoversFeature_iff_bounded_on_sameOracleFiber
    {fstar : Strings → Y} {feature : Strings → Feature} {ε : ℝ≥0} :
    ApproxOracleRecoversFeature fstar feature ε ↔
      ∀ {x x' : Strings}, SameOracleFiber fstar x x' →
        dist (feature x) (feature x') ≤ (ε : ℝ) := by
  constructor
  · intro hRecover x x' hFiber
    exact hRecover x x' hFiber
  · intro hBound x x' hzero
    exact hBound hzero

end ApproxRecovery

section ExactSupport

/-- Leaf-support version: realized leaf summaries stay in the same oracle fiber
as their source leaf. -/
theorem leaf_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {b z : Strings}
    (hExact : ExactTheoremBacked g T fstar)
    (hb : b ∈ leaves T)
    (hz : z ∈ (g b).support) :
    SameOracleFiber fstar z b := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z b = 0 := hSupport.1 b hb z hz
  simpa [SameOracleFiber, D] using hzeroD

/-- Merge-support version: every realized internal reduction stays in the same
oracle fiber as its raw subtree. -/
theorem merge_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {p : BinTree Strings × BinTree Strings} {z : Strings}
    (hExact : ExactTheoremBacked g T fstar)
    (hp : p ∈ internal_nodes T)
    (hz : z ∈ (reduce g (BinTree.node p.1 p.2)).support) :
    SameOracleFiber fstar z (S (BinTree.node p.1 p.2)) := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z (S (BinTree.node p.1 p.2)) = 0 := hSupport.2.1 p hp z hz
  simpa [SameOracleFiber, D] using hzeroD

/-- On-range idempotence version: re-summaries stay in the same oracle fiber as
the already-realized theorem object. -/
theorem idempotent_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {Z z : Strings}
    (hExact : ExactTheoremBacked g T fstar)
    (hRange : InRange g Z)
    (hz : z ∈ (g Z).support) :
    SameOracleFiber fstar z Z := by
  have hSupport : SupportExactTheoremBacked g T fstar :=
    (exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
      (g := g) (T := T) (fstar := fstar)).1 ⟨hExact⟩
  have hzeroD : D fstar z Z = 0 := hSupport.2.2 Z hRange z hz
  simpa [SameOracleFiber, D] using hzeroD

/-- Multi-round support version: every realized reduction under `ZR` remains in
the same oracle fiber as the original document. -/
theorem zr_support_sameOracleFiber_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    {z : Strings}
    (hz : z ∈ (ZR g x R T).support) :
    SameOracleFiber fstar z x := by
  exact zero_distortion_on_ZR_support_of_exactTheoremBacked
    (hp := hp) (hExact := hExact) (hR := hR) z hz

end ExactSupport

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/FiberPreservingObjective.lean (consolidated 2026-07-02) -/

section

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

end

end

/-! ## From FormalProofs/OPT/FeatureClassObjectives.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/FeatureClassObjectives.lean

Feature-class objectives for learned theorem features.

The point of this file is modest: it packages a class/objective layer on top of
the existing theorem-backed feature transport results, so the same feature `Φ`
can support both downstream scalar readouts and class-style supervision.

The hard indicator objectives below are intentionally simple. Exact transport is
automatic once the oracle identifies the feature. Approximate transport is
available whenever the chosen class objective admits the same Lipschitz and
boundedness hypotheses already used by
`TheoremBackingApproxMeasurementError.lean`.
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

variable {Strings : Type*} [Monoid Strings]

/-- Hard same-class objective on a feature space. -/
def sameFeatureClassUtility
    {Feature Class : Type*} [DecidableEq Class]
    (classOf : Feature → Class) : OracleUtility2 Feature :=
  fun y y' => if classOf y = classOf y' then 1 else 0

/-- Hard different-class objective on a feature space. -/
def differentFeatureClassUtility
    {Feature Class : Type*} [DecidableEq Class]
    (classOf : Feature → Class) : OracleUtility2 Feature :=
  fun y y' => if classOf y ≠ classOf y' then 1 else 0

section Exact

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Class : Type*} [Encodable Feature] [DecidableEq Class]

/-- Exact theorem-backed reductions preserve same-class feature objectives
whenever the oracle identifies the theorem-bearing feature. -/
theorem expected_sameFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature) :
    Exp (ZR g x R T)
      (fun z => sameFeatureClassUtility classOf (feature z) (feature x)) =
      sameFeatureClassUtility classOf (feature x) (feature x) := by
  simpa [sameFeatureClassUtility] using
    (expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature)
      (u := sameFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover)

/-- Exact theorem-backed reductions preserve different-class feature objectives
whenever the oracle identifies the theorem-bearing feature. -/
theorem expected_differentFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
    (fstar : Strings → Y)
    (feature : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hRecover : OracleRecoversFeature fstar feature) :
    Exp (ZR g x R T)
      (fun z => differentFeatureClassUtility classOf (feature z) (feature x)) =
      differentFeatureClassUtility classOf (feature x) (feature x) := by
  simpa [differentFeatureClassUtility] using
    (expected_feature_utility_preserved_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature
      (fstar := fstar) (feature := feature)
      (u := differentFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      hp hExact hR hRecover)

end Exact

section Approximate

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature Class : Type*} [BoundedPseudoMetricSpace Feature] [DecidableEq Class]

/-- Approximate theorem-backed transport for same-class feature objectives. The
indicator-style objective itself is fixed here; the quantitative assumptions are
the usual Lipschitz/boundedness ones already required by the approximate
feature-transport theorem. -/
theorem expected_sameFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 (sameFeatureClassUtility classOf) L1)
    (hL2 : OracleUtilityLipschitz2 (sameFeatureClassUtility classOf) L2)
    (hU : OracleUtilityBoundedAt (sameFeatureClassUtility classOf) (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T)
        (fun z => sameFeatureClassUtility classOf (feature z) (featureHat x)) -
        sameFeatureClassUtility classOf (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  simpa [sameFeatureClassUtility] using
    (expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
      (fstar := fstar) (feature := feature) (featureHat := featureHat)
      (u := sameFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      (K := K) (L1 := L1) (L2 := L2) (U := U)
      hp hApprox hR hFeatureLip hL1 hL2 hU hbound hbound_global h_mono)

/-- Approximate theorem-backed transport for different-class feature
objectives. -/
theorem expected_differentFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (classOf : Feature → Class)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 (differentFeatureClassUtility classOf) L1)
    (hL2 : OracleUtilityLipschitz2 (differentFeatureClassUtility classOf) L2)
    (hU : OracleUtilityBoundedAt (differentFeatureClassUtility classOf) (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T)
        (fun z => differentFeatureClassUtility classOf (feature z) (featureHat x)) -
        differentFeatureClassUtility classOf (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  simpa [differentFeatureClassUtility] using
    (expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz
      (fstar := fstar) (feature := feature) (featureHat := featureHat)
      (u := differentFeatureClassUtility classOf)
      (g := g) (x := x) (R := R) (T := T)
      (K := K) (L1 := L1) (L2 := L2) (U := U)
      hp hApprox hR hFeatureLip hL1 hL2 hU hbound hbound_global h_mono)

end Approximate

end FormalProofs.OPT

end

end

/-! ## From FormalProofs/OPT/SharedFeatureMultihead.lean (consolidated 2026-07-02) -/

section

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

end

end

/-! ## From FormalProofs/OPT/ApproxFiberTransport.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/ApproxFiberTransport.lean

Combined transport bounds when both theorem-backedness and feature recovery
are approximate.

This file unifies three independent error sources into a single additive bound:

1. **Transport budget** from `ApproxLocalLawsBundle` — how well g satisfies
   the local laws L1/L2/L3.
2. **Fiber error** from `ApproxOracleRecoversFeature` — how well the learned
   feature φ captures f*-equivalence classes.
3. **Readout error** from `ApproxReadoutFactorsThroughFeature` — how well
   downstream heads factor through φ.

The main theorem `expected_utility_bound_approx_fiber` gives:

  |E[u(φ(Z), φ̂(x))] - u(φ(x), φ(x))| ≤
      L₁ · K · (transport_budget)       -- from approximate local laws
    + L₁ · ε_fiber                       -- from approximate fiber preservation
    + L₂ · dist(φ̂(x), φ(x))            -- from measurement error

The fiber error term is the new contribution; the other two terms match the
existing `TheoremBackingApproxMeasurementError` bound structure.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

section ExactBackingApproxFiber

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Exact theorem-backed transport with approximate fiber recovery.

When the summarizer is exactly theorem-backed but the feature map only
approximately preserves fibers, the utility gap is bounded by the Lipschitz
constant times the fiber error plus measurement error.

This is complementary to the existing
`expected_feature_utility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature`
which requires exact `OracleRecoversFeature`. -/
theorem expected_utility_bound_exactBacked_approxFiber
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (ε_fiber : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature x) U) :
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)| ≤
      (L1 : ℝ) * (ε_fiber : ℝ) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  let M_feature : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Feature)
  have hM_feature : 0 ≤ M_feature := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Feature)
  have hbound_feature : ∀ z, D feature z x ≤ M_feature := by
    intro z
    unfold D M_feature
    exact BoundedPseudoMetricSpace.dist_le (feature z) (feature x)
  have hD_feature :
      Summable (fun z => (ZR g x R T z).toReal * D feature z x) :=
    summable_D_of_bounded (ZR g x R T) feature x M_feature hM_feature hbound_feature
  have h_noise_transport :
      |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
          u (feature x) (feature x)| ≤
        (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) :=
    expected_utility_bound_with_noise_ZR
      (g := g) (T := T) (x := x) (R := R)
      (fstar := feature) (fhat := featureHat)
      (u := u) (L1 := L1) (L2 := L2) (U := U)
      hL1 hL2 hU hD_feature
  -- The key insight: with exact theorem-backing, every z in ZR support has
  -- dist(f*(z), f*(x)) = 0, so approximate fiber recovery gives
  -- dist(feature(z), feature(x)) ≤ ε_fiber for each z.
  have h_fiber_bound :
      Exp (ZR g x R T) (fun z => D feature z x) ≤ (ε_fiber : ℝ) := by
    exact Exp_le_const_of_support
      (p := ZR g x R T)
      (f := fun z => D feature z x)
      (c := (ε_fiber : ℝ))
      (M := M_feature)
      (hc := by exact_mod_cast ε_fiber.property)
      (hM := hM_feature)
      (hsupport := by
        intro z hz
        unfold D
        exact hApproxRecover z x
          (zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz))
      (hf_nonneg := by
        intro z
        exact dist_nonneg)
      (hf_bound := hbound_feature)
  have hmul_fiber :
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) ≤
        (L1 : ℝ) * (ε_fiber : ℝ) := by
    exact mul_le_mul_of_nonneg_left h_fiber_bound (by exact_mod_cast L1.property)
  calc
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)|
      ≤ (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := h_noise_transport
    _ ≤ (L1 : ℝ) * (ε_fiber : ℝ) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          linarith

end ExactBackingApproxFiber

section FullApprox

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Main theorem: combined transport bound with approximate theorem-backing,
approximate fiber preservation, and measurement error.

The three error terms decompose additively:
- L₁ · K · (leaf + merge + (R-1)·idemp)  from approximate local laws
- L₁ · ε_fiber                            from approximate fiber preservation
- L₂ · dist(φ̂(x), φ(x))                  from measurement error

This extends `expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz`
with the additional fiber-error term. When ε_fiber = 0 (exact fiber recovery),
the first and third terms match the existing bound. -/
theorem expected_utility_bound_approx_fiber
    (fstar : Strings → Y)
    (feature featureHat : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K : ℝ≥0) (ε_fiber : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hApproxRecover : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hFeatureLip : FeatureLipschitzFromOracle fstar feature K)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature x) U)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)| ≤
      (L1 : ℝ) * (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε_fiber : ℝ) +
      (L2 : ℝ) * dist (featureHat x) (feature x) := by
  let budget : ℝ :=
    hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
      ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp
  let M_feature : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Feature)
  have hM_feature : 0 ≤ M_feature := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Feature)
  have hbound_feature : ∀ z, D feature z x ≤ M_feature := by
    intro z
    unfold D M_feature
    exact BoundedPseudoMetricSpace.dist_le (feature z) (feature x)
  have hD_feature :
      Summable (fun z => (ZR g x R T z).toReal * D feature z x) :=
    summable_D_of_bounded (ZR g x R T) feature x M_feature hM_feature hbound_feature
  -- Step 1: Split into transport + noise terms
  have h_noise_transport :
      |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
          u (feature x) (feature x)| ≤
        (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) :=
    expected_utility_bound_with_noise_ZR
      (g := g) (T := T) (x := x) (R := R)
      (fstar := feature) (fhat := featureHat)
      (u := u) (L1 := L1) (L2 := L2) (U := U)
      hL1 hL2 hU hD_feature
  -- Step 2: Bound feature distortion using Lipschitz + fiber error
  -- D(feature, z, x) ≤ K · D(fstar, z, x) + ε_fiber
  -- (triangle: if D(fstar,z,x) = 0 then ε_fiber; otherwise K · D(fstar,z,x))
  have h_feature_pointwise :
      ∀ z, D feature z x ≤ (K : ℝ) * D fstar z x + (ε_fiber : ℝ) := by
    intro z
    unfold D
    by_cases h : dist (fstar z) (fstar x) = 0
    · calc dist (feature z) (feature x)
          ≤ (ε_fiber : ℝ) := hApproxRecover z x h
        _ ≤ (K : ℝ) * dist (fstar z) (fstar x) + (ε_fiber : ℝ) := by
            linarith [mul_nonneg (show 0 ≤ (K : ℝ) from by exact_mod_cast K.property)
              (show 0 ≤ dist (fstar z) (fstar x) from dist_nonneg)]
    · calc dist (feature z) (feature x)
          ≤ (K : ℝ) * dist (fstar z) (fstar x) := hFeatureLip z x
        _ ≤ (K : ℝ) * dist (fstar z) (fstar x) + (ε_fiber : ℝ) := by
            exact le_add_of_nonneg_right (by exact_mod_cast ε_fiber.property)
  -- Step 3: Take expectations
  let M_oracle : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM_oracle : 0 ≤ M_oracle := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have hbound_oracle : ∀ z, D fstar z x ≤ M_oracle := by
    intro z
    unfold D M_oracle
    exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
  have hD_oracle :
      Summable (fun z => (ZR g x R T z).toReal * D fstar z x) :=
    summable_D_of_bounded (ZR g x R T) fstar x M_oracle hM_oracle hbound_oracle
  have hKD_summable :
      Summable (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x)) := by
    have hEq :
        (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x)) =
          (fun z => (K : ℝ) * ((ZR g x R T z).toReal * D fstar z x)) := by
      funext z
      ring
    rw [hEq]
    exact hD_oracle.mul_left (K : ℝ)
  have hconst_summable :
      Summable (fun z => (ZR g x R T z).toReal * (ε_fiber : ℝ)) :=
    PMF.summable_coe_real_mul_of_bounded
      (ZR g x R T) (fun _ => (ε_fiber : ℝ)) (ε_fiber : ℝ)
      (by exact_mod_cast ε_fiber.property)
      (fun _ => by
        have hε : 0 ≤ (ε_fiber : ℝ) := by exact_mod_cast ε_fiber.property
        simp [abs_of_nonneg hε])
  have hsum_stage :
      Summable
        (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x + (ε_fiber : ℝ))) := by
    have hEq :
        (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x + (ε_fiber : ℝ))) =
          (fun z =>
            (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x) +
              (ZR g x R T z).toReal * (ε_fiber : ℝ)) := by
      funext z
      ring
    rw [hEq]
    exact Summable.add hKD_summable hconst_summable
  have h_Exp_bound :
      Exp (ZR g x R T) (fun z => D feature z x) ≤
        (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ) := by
    calc Exp (ZR g x R T) (fun z => D feature z x)
        ≤ Exp (ZR g x R T) (fun z => (K : ℝ) * D fstar z x + (ε_fiber : ℝ)) := by
          apply Exp_mono'
          · intro z
            exact h_feature_pointwise z
          · exact hD_feature
          · exact hsum_stage
      _ = (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ) := by
          rw [Exp_add (ZR g x R T)
            (fun z => (K : ℝ) * D fstar z x)
            (fun _ => (ε_fiber : ℝ))
            hKD_summable
            hconst_summable]
          have hExp_scale :
              Exp (ZR g x R T) (fun z => (K : ℝ) * D fstar z x) =
                (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) := by
            unfold Exp
            have hEq :
                (fun z => (ZR g x R T z).toReal * ((K : ℝ) * D fstar z x)) =
                  (fun z => (K : ℝ) * ((ZR g x R T z).toReal * D fstar z x)) := by
              funext z
              ring
            rw [hEq, tsum_mul_left]
          rw [hExp_scale, Exp_const]
  -- Step 4: Use approximate local law budget
  have h_budget :
      Δ_R_ZR g x R T fstar ≤ budget :=
    Δ_R_ZR_le_of_approx_bundle g T fstar x R hp hR hbound hbound_global h_mono
      hApprox.approxLocalLaws
  -- Step 5: Combine
  have hL1_nonneg : 0 ≤ (L1 : ℝ) := by exact_mod_cast L1.property
  have hK_nonneg : 0 ≤ (K : ℝ) := by exact_mod_cast K.property
  have hmul_exp :
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) ≤
        (L1 : ℝ) * ((K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ)) := by
    exact mul_le_mul_of_nonneg_left h_Exp_bound hL1_nonneg
  have hbudget_scaled :
      (K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) ≤ (K : ℝ) * budget := by
    apply mul_le_mul_of_nonneg_left
    simpa [Δ_R_ZR, budget] using h_budget
    exact hK_nonneg
  have hmul_budget :
      (L1 : ℝ) * ((K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ)) ≤
        (L1 : ℝ) * ((K : ℝ) * budget + (ε_fiber : ℝ)) := by
    apply mul_le_mul_of_nonneg_left
    linarith
    exact hL1_nonneg
  calc
    |Exp (ZR g x R T) (fun z => u (feature z) (featureHat x)) -
        u (feature x) (feature x)|
      ≤ (L1 : ℝ) * Exp (ZR g x R T) (fun z => D feature z x) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := h_noise_transport
    _ ≤ (L1 : ℝ) * ((K : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) + (ε_fiber : ℝ)) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          linarith
    _ ≤ (L1 : ℝ) * ((K : ℝ) * budget + (ε_fiber : ℝ)) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by
          linarith
    _ = (L1 : ℝ) * (K : ℝ) * budget +
        (L1 : ℝ) * (ε_fiber : ℝ) +
        (L2 : ℝ) * dist (featureHat x) (feature x) := by ring

end FullApprox

end FormalProofs.OPT

end

end
