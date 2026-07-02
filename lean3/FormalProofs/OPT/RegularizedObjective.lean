import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.NeuralOperatorTheoremBridge

/-!
# FormalProofs/OPT/RegularizedObjective.lean

Regularized oracle-risk objectives for summarizer selection.

This file makes the optimization surface explicit:

- a population oracle-risk term based on `Δ_R_ZR`,
- a generic summary-cost term,
- a certified local-law penalty built from `ApproxLocalLawsBundle`,
- minimizer predicates for regularized and constrained regularized objectives,
- upper bounds obtained by substituting the approximate-local-law distortion bound.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-! ## No-cost empirical training objective -/

/-- Empirical objective used for the paper-facing learned-tree story.

The parameter type is intentionally generic: in applications it may be a pair
`(θ, φ)` of state-map and score-map parameters, or a prompt/program choice. The
fields are losses already estimated on the training sample. This objective has
no summary-cost term; summary cost remains available in the more general
population objective below, but is not part of the present empirical training
surface. -/
structure NoCostLearnedTreeObjective (Param : Type*) where
  calibrationLoss : Param → ℝ
  /-- Gold-standard root prediction loss. Kept as `rootLoss` for API
  compatibility with earlier objective statements. -/
  rootLoss : Param → ℝ
  c1Loss : Param → ℝ
  c3Loss : Param → ℝ
  c2Loss : Param → ℝ
  calibrationWeight : ℝ
  rootWeight : ℝ
  c1Weight : ℝ
  c3Weight : ℝ
  c2Weight : ℝ

namespace NoCostLearnedTreeObjective

variable {Param : Type*}

/-- Paper-facing name for the gold-standard root prediction loss. -/
def goldLoss (obj : NoCostLearnedTreeObjective Param) : Param → ℝ :=
  obj.rootLoss

/-- Paper-facing name for the gold-standard root prediction weight. -/
def goldWeight (obj : NoCostLearnedTreeObjective Param) : ℝ :=
  obj.rootWeight

/-- Local-law part of the no-cost empirical objective. -/
def localLawPenalty (obj : NoCostLearnedTreeObjective Param) (p : Param) : ℝ :=
  obj.c1Weight * obj.c1Loss p
    + obj.c3Weight * obj.c3Loss p
    + obj.c2Weight * obj.c2Loss p

/-- Calibration plus gold-standard root-fit part of the no-cost empirical
objective. -/
def supervisedPenalty (obj : NoCostLearnedTreeObjective Param) (p : Param) : ℝ :=
  obj.calibrationWeight * obj.calibrationLoss p
    + obj.goldWeight * obj.goldLoss p

/-- Total no-cost empirical objective:
calibration + gold-standard root fit + C1/C3/C2 local-law penalties. -/
def value (obj : NoCostLearnedTreeObjective Param) (p : Param) : ℝ :=
  obj.supervisedPenalty p + obj.localLawPenalty p

/-- Expanded form of the no-cost empirical objective. -/
theorem value_eq
    (obj : NoCostLearnedTreeObjective Param) (p : Param) :
    obj.value p =
      obj.calibrationWeight * obj.calibrationLoss p
        + obj.rootWeight * obj.rootLoss p
        + obj.c1Weight * obj.c1Loss p
        + obj.c3Weight * obj.c3Loss p
        + obj.c2Weight * obj.c2Loss p := by
  simp [value, supervisedPenalty, localLawPenalty, goldWeight, goldLoss]
  ring_nf

/-- Expanded paper-facing form of the no-cost empirical objective, with the
root prediction loss named as gold-standard loss. -/
theorem value_eq_gold_form
    (obj : NoCostLearnedTreeObjective Param) (p : Param) :
    obj.value p =
      obj.calibrationWeight * obj.calibrationLoss p
        + obj.goldWeight * obj.goldLoss p
        + obj.c1Weight * obj.c1Loss p
        + obj.c3Weight * obj.c3Loss p
        + obj.c2Weight * obj.c2Loss p := by
  simpa [goldWeight, goldLoss] using value_eq (obj := obj) p

/-- If the local-law weights are zero, the objective reduces to calibration
and gold-standard root fit. -/
theorem value_eq_supervisedPenalty_of_zero_localLawWeights
    (obj : NoCostLearnedTreeObjective Param) (p : Param)
    (h1 : obj.c1Weight = 0) (h3 : obj.c3Weight = 0)
    (h2 : obj.c2Weight = 0) :
    obj.value p = obj.supervisedPenalty p := by
  simp [value, localLawPenalty, h1, h3, h2]

/-- If the calibration/root weights are zero, the objective is pure local-law
projection pressure. -/
theorem value_eq_localLawPenalty_of_zero_supervisedWeights
    (obj : NoCostLearnedTreeObjective Param) (p : Param)
    (hcal : obj.calibrationWeight = 0) (hroot : obj.rootWeight = 0) :
    obj.value p = obj.localLawPenalty p := by
  simp [value, supervisedPenalty, goldWeight, goldLoss, hcal, hroot]

/-- Nonnegativity under nonnegative weights and nonnegative component losses. -/
theorem value_nonneg
    (obj : NoCostLearnedTreeObjective Param) (p : Param)
    (hWcal : 0 ≤ obj.calibrationWeight)
    (hWroot : 0 ≤ obj.rootWeight)
    (hW1 : 0 ≤ obj.c1Weight)
    (hW3 : 0 ≤ obj.c3Weight)
    (hW2 : 0 ≤ obj.c2Weight)
    (hLcal : 0 ≤ obj.calibrationLoss p)
    (hLroot : 0 ≤ obj.rootLoss p)
    (hL1 : 0 ≤ obj.c1Loss p)
    (hL3 : 0 ≤ obj.c3Loss p)
    (hL2 : 0 ≤ obj.c2Loss p) :
    0 ≤ obj.value p := by
  rw [value_eq]
  positivity

end NoCostLearnedTreeObjective

/-- Generic cost on produced summaries (e.g. length, bits, latency proxy). -/
abbrev SummaryCost (Strings : Type*) := Strings → ℝ

/-- Weight bundle for the regularized oracle objective.

These are optimization weights, usually denoted by a vector such as `λ` in the
paper.  They control how strongly training penalizes distortion, summary cost,
and local-law residuals.  They are not certification thresholds; epsilon-level
certification is expressed separately by the local-law error in
`ApproxLocalLawsBundle.CertifiedAtEpsilon`. -/
structure RegularizedObjectiveWeights where
  distortion : ℝ
  summary : ℝ
  leaf : ℝ
  merge : ℝ
  idemp : ℝ

@[ext] theorem RegularizedObjectiveWeights.ext
    {w₁ w₂ : RegularizedObjectiveWeights}
    (hDistortion : w₁.distortion = w₂.distortion)
    (hSummary : w₁.summary = w₂.summary)
    (hLeaf : w₁.leaf = w₂.leaf)
    (hMerge : w₁.merge = w₂.merge)
    (hIdemp : w₁.idemp = w₂.idemp) :
    w₁ = w₂ := by
  cases w₁
  cases w₂
  simp_all

/-- Relative shares used inside the local-law part of the regularizer. -/
structure LawComponentShares where
  leaf : ℝ
  merge : ℝ
  idemp : ℝ

/-- Uniform split across leaf / merge / idempotence law terms. -/
def uniformLawComponentShares : LawComponentShares where
  leaf := (1 : ℝ) / 3
  merge := (1 : ℝ) / 3
  idemp := (1 : ℝ) / 3

/-- One-parameter frontier from the legacy summary-only endpoint
(`lawStrength = 0`) to the fully law-regularized endpoint (`lawStrength = 1`).

`regularizerWeight` controls the total mass moved away from the distortion term,
while `lawStrength` reallocates that regularizer mass from the summary-budget
term toward the local-law penalties. -/
def frontierRegularizedObjectiveWeights
    (regularizerWeight lawStrength : ℝ)
    (shares : LawComponentShares) : RegularizedObjectiveWeights where
  distortion := 1 - regularizerWeight
  summary := regularizerWeight * (1 - lawStrength)
  leaf := regularizerWeight * lawStrength * shares.leaf
  merge := regularizerWeight * lawStrength * shares.merge
  idemp := regularizerWeight * lawStrength * shares.idemp

/-- Fixed simulation-facing default: `0.75` on global oracle distortion, and
the remaining `0.25` split equally between summary-budget pressure and the
three local-law penalties. -/
def simulationDefaultRegularizedObjectiveWeights : RegularizedObjectiveWeights where
  distortion := (3 : ℝ) / 4
  summary := (1 : ℝ) / 8
  leaf := (1 : ℝ) / 24
  merge := (1 : ℝ) / 24
  idemp := (1 : ℝ) / 24

/-- The fixed simulation default is the frontier point with regularizer weight
`0.25`, law strength `0.5`, and uniform law shares. -/
theorem simulationDefaultRegularizedObjectiveWeights_eq_frontier :
    simulationDefaultRegularizedObjectiveWeights =
      frontierRegularizedObjectiveWeights ((1 : ℝ) / 4) ((1 : ℝ) / 2)
        uniformLawComponentShares := by
  ext <;> norm_num [simulationDefaultRegularizedObjectiveWeights,
    frontierRegularizedObjectiveWeights, uniformLawComponentShares]

/-- Expected summary cost under the `ZR` output distribution. -/
def expectedSummaryCost
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (cost : SummaryCost Strings) : ℝ :=
  Exp (ZR g x R T) cost

/-- Certified local-law penalty built from an audited / approximate law bundle.
The idempotence term is scaled by `R - 1`, matching its contribution to
multi-round distortion bounds. -/
def certifiedLawPenalty
    (R : ℕ)
    (weights : RegularizedObjectiveWeights)
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fstar) : ℝ :=
  weights.leaf * laws.epsLeaf
    + weights.merge * laws.epsMerge
    + weights.idemp * (((R : ℝ) - 1) * laws.epsIdemp)

/-- Population oracle-risk objective: distortion plus summary cost. -/
def oracleRiskObjective
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights) : ℝ :=
  weights.distortion * Δ_R_ZR g x R T fstar
    + weights.summary * expectedSummaryCost g x R T cost

/-- Certified regularized objective: population oracle risk plus the approximate
local-law penalty. This is the optimization surface used for model selection /
hyperparameter comparison when only audited local-law budgets are available. -/
def certifiedRegularizedObjective
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (laws : ApproxLocalLawsBundle g T fstar) : ℝ :=
  oracleRiskObjective g x R T fstar cost weights
    + certifiedLawPenalty R weights laws

/-- Hard budget version of the same design problem. -/
structure RegularizedObjectiveConstraints where
  summaryMax : ℝ
  epsLeafMax : ℝ
  epsMergeMax : ℝ
  epsIdempMax : ℝ

/-- Feasibility for the constrained formulation. -/
def SatisfiesRegularizedConstraints
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (laws : ApproxLocalLawsBundle g T fstar)
    (constraints : RegularizedObjectiveConstraints) : Prop :=
  expectedSummaryCost g x R T cost ≤ constraints.summaryMax
    ∧ laws.epsLeaf ≤ constraints.epsLeafMax
    ∧ laws.epsMerge ≤ constraints.epsMergeMax
    ∧ laws.epsIdemp ≤ constraints.epsIdempMax

/-- Unconstrained minimizer predicate for the certified regularized objective. -/
def IsCertifiedRegularizedMinimizer
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar) : Prop :=
  ∀ g : Summarizer Strings,
    ∀ laws : ApproxLocalLawsBundle g T fstar,
      certifiedRegularizedObjective gStar x R T fstar cost weights lawsStar ≤
        certifiedRegularizedObjective g x R T fstar cost weights laws

/-- Constrained minimizer predicate for the certified regularized objective. -/
def IsConstrainedCertifiedRegularizedMinimizer
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (constraints : RegularizedObjectiveConstraints)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar) : Prop :=
  SatisfiesRegularizedConstraints gStar x R T fstar cost lawsStar constraints
    ∧ ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          SatisfiesRegularizedConstraints g x R T fstar cost laws constraints →
            certifiedRegularizedObjective gStar x R T fstar cost weights lawsStar ≤
              certifiedRegularizedObjective g x R T fstar cost weights laws

/-- The simulation-facing default weights are nonnegative. -/
theorem simulationDefaultRegularizedObjectiveWeights_nonneg :
    0 ≤ simulationDefaultRegularizedObjectiveWeights.distortion
      ∧ 0 ≤ simulationDefaultRegularizedObjectiveWeights.summary
      ∧ 0 ≤ simulationDefaultRegularizedObjectiveWeights.leaf
      ∧ 0 ≤ simulationDefaultRegularizedObjectiveWeights.merge
      ∧ 0 ≤ simulationDefaultRegularizedObjectiveWeights.idemp := by
  repeat' constructor <;> norm_num [simulationDefaultRegularizedObjectiveWeights]

/-- Nonnegativity of the frontier weights under the obvious box constraints. -/
theorem frontierRegularizedObjectiveWeights_nonneg
    (regularizerWeight lawStrength : ℝ)
    (shares : LawComponentShares)
    (hRegNonneg : 0 ≤ regularizerWeight) (hRegLeOne : regularizerWeight ≤ 1)
    (hLawNonneg : 0 ≤ lawStrength) (hLawLeOne : lawStrength ≤ 1)
    (hLeaf : 0 ≤ shares.leaf) (hMerge : 0 ≤ shares.merge) (hIdemp : 0 ≤ shares.idemp) :
    0 ≤ (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).distortion
      ∧ 0 ≤ (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).summary
      ∧ 0 ≤ (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).leaf
      ∧ 0 ≤ (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).merge
      ∧ 0 ≤ (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).idemp := by
  constructor
  · dsimp [frontierRegularizedObjectiveWeights]
    linarith
  constructor
  · dsimp [frontierRegularizedObjectiveWeights]
    exact mul_nonneg hRegNonneg (sub_nonneg.mpr hLawLeOne)
  constructor
  · dsimp [frontierRegularizedObjectiveWeights]
    exact mul_nonneg (mul_nonneg hRegNonneg hLawNonneg) hLeaf
  constructor
  · dsimp [frontierRegularizedObjectiveWeights]
    exact mul_nonneg (mul_nonneg hRegNonneg hLawNonneg) hMerge
  · dsimp [frontierRegularizedObjectiveWeights]
    exact mul_nonneg (mul_nonneg hRegNonneg hLawNonneg) hIdemp

/-- If the law shares sum to one, the whole frontier weight vector sums to one. -/
theorem frontierRegularizedObjectiveWeights_total_mass
    (regularizerWeight lawStrength : ℝ)
    (shares : LawComponentShares)
    (hShares : shares.leaf + shares.merge + shares.idemp = 1) :
    (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).distortion
      + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).summary
      + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).leaf
      + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).merge
      + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).idemp
      = 1 := by
  calc
    (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).distortion
        + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).summary
        + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).leaf
        + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).merge
        + (frontierRegularizedObjectiveWeights regularizerWeight lawStrength shares).idemp
      =
        1 - regularizerWeight * lawStrength
          + regularizerWeight * lawStrength
              * (shares.leaf + shares.merge + shares.idemp) := by
          dsimp [frontierRegularizedObjectiveWeights]
          ring
    _ = 1 - regularizerWeight * lawStrength + regularizerWeight * lawStrength * 1 := by
          rw [hShares]
    _ = 1 := by
          ring

/-- `lawStrength = 0` recovers the legacy summary-only endpoint. -/
theorem frontierRegularizedObjectiveWeights_zero_lawStrength
    (regularizerWeight : ℝ) (shares : LawComponentShares) :
    frontierRegularizedObjectiveWeights regularizerWeight 0 shares =
      { distortion := 1 - regularizerWeight
        summary := regularizerWeight
        leaf := 0
        merge := 0
        idemp := 0 } := by
  ext <;> simp [frontierRegularizedObjectiveWeights]

/-- `lawStrength = 1` removes the summary-budget part of the regularizer. -/
theorem frontierRegularizedObjectiveWeights_one_lawStrength
    (regularizerWeight : ℝ) (shares : LawComponentShares) :
    frontierRegularizedObjectiveWeights regularizerWeight 1 shares =
      { distortion := 1 - regularizerWeight
        summary := 0
        leaf := regularizerWeight * shares.leaf
        merge := regularizerWeight * shares.merge
        idemp := regularizerWeight * shares.idemp } := by
  ext <;> simp [frontierRegularizedObjectiveWeights]

/-- Oracle/projection objective weights.  The parameter `lam` moves mass from
the oracle-distortion term to the local-law projection penalty, while `shares`
chooses the split across C1/C2/C3. -/
def oracleProjectionObjectiveWeights
    (lam : ℝ) (shares : LawComponentShares) : RegularizedObjectiveWeights where
  distortion := 1 - lam
  summary := 0
  leaf := lam * shares.leaf
  merge := lam * shares.merge
  idemp := lam * shares.idemp

/-- At `lam = 0`, the oracle/projection weights are pure oracle distortion. -/
theorem oracleProjectionObjectiveWeights_lam_zero
    (shares : LawComponentShares) :
    oracleProjectionObjectiveWeights 0 shares =
      { distortion := 1
        summary := 0
        leaf := 0
        merge := 0
        idemp := 0 } := by
  ext <;> simp [oracleProjectionObjectiveWeights]

/-- At `lam = 1`, the oracle/projection weights are pure local-law projection
penalty, with no oracle-distortion mass. -/
theorem oracleProjectionObjectiveWeights_lam_one
    (shares : LawComponentShares) :
    oracleProjectionObjectiveWeights 1 shares =
      { distortion := 0
        summary := 0
        leaf := shares.leaf
        merge := shares.merge
        idemp := shares.idemp } := by
  ext <;> simp [oracleProjectionObjectiveWeights]

/-- Nonnegativity of the oracle/projection weights under the standard simplex
constraints. -/
theorem oracleProjectionObjectiveWeights_nonneg
    (lam : ℝ) (shares : LawComponentShares)
    (hLamNonneg : 0 ≤ lam) (hLamLeOne : lam ≤ 1)
    (hLeaf : 0 ≤ shares.leaf) (hMerge : 0 ≤ shares.merge) (hIdemp : 0 ≤ shares.idemp) :
    0 ≤ (oracleProjectionObjectiveWeights lam shares).distortion
      ∧ 0 ≤ (oracleProjectionObjectiveWeights lam shares).summary
      ∧ 0 ≤ (oracleProjectionObjectiveWeights lam shares).leaf
      ∧ 0 ≤ (oracleProjectionObjectiveWeights lam shares).merge
      ∧ 0 ≤ (oracleProjectionObjectiveWeights lam shares).idemp := by
  constructor
  · dsimp [oracleProjectionObjectiveWeights]
    linarith
  constructor
  · dsimp [oracleProjectionObjectiveWeights]
    exact le_rfl
  constructor
  · dsimp [oracleProjectionObjectiveWeights]
    exact mul_nonneg hLamNonneg hLeaf
  constructor
  · dsimp [oracleProjectionObjectiveWeights]
    exact mul_nonneg hLamNonneg hMerge
  · dsimp [oracleProjectionObjectiveWeights]
    exact mul_nonneg hLamNonneg hIdemp

/-- If the local-law shares sum to one, the oracle/projection weights also have
unit total mass. -/
theorem oracleProjectionObjectiveWeights_total_mass
    (lam : ℝ) (shares : LawComponentShares)
    (hShares : shares.leaf + shares.merge + shares.idemp = 1) :
    (oracleProjectionObjectiveWeights lam shares).distortion
      + (oracleProjectionObjectiveWeights lam shares).summary
      + (oracleProjectionObjectiveWeights lam shares).leaf
      + (oracleProjectionObjectiveWeights lam shares).merge
      + (oracleProjectionObjectiveWeights lam shares).idemp
      = 1 := by
  calc
    (oracleProjectionObjectiveWeights lam shares).distortion
        + (oracleProjectionObjectiveWeights lam shares).summary
        + (oracleProjectionObjectiveWeights lam shares).leaf
        + (oracleProjectionObjectiveWeights lam shares).merge
        + (oracleProjectionObjectiveWeights lam shares).idemp
      = 1 - lam + lam * (shares.leaf + shares.merge + shares.idemp) := by
          dsimp [oracleProjectionObjectiveWeights]
          ring
    _ = 1 - lam + lam * 1 := by
          rw [hShares]
    _ = 1 := by
          ring

/-- The paper-facing oracle/projection objective:

`(1-lam) * oracle risk + lam * local-law projection penalty`.

The idempotence term carries the same `(R-1)` round scaling as the existing
approximate-gap theorem. -/
def oracleProjectionObjective
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (lam : ℝ) (shares : LawComponentShares)
    (laws : ApproxLocalLawsBundle g T fstar) : ℝ :=
  (1 - lam) * Δ_R_ZR g x R T fstar
    + lam *
      (shares.leaf * laws.epsLeaf
        + shares.merge * laws.epsMerge
        + shares.idemp * (((R : ℝ) - 1) * laws.epsIdemp))

/-- The direct oracle/projection objective is exactly the certified regularized
objective with zero summary-cost weight and `oracleProjectionObjectiveWeights`. -/
theorem oracleProjectionObjective_eq_certifiedRegularizedObjective
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (lam : ℝ) (shares : LawComponentShares)
    (laws : ApproxLocalLawsBundle g T fstar) :
    oracleProjectionObjective g x R T fstar lam shares laws =
      certifiedRegularizedObjective g x R T fstar cost
        (oracleProjectionObjectiveWeights lam shares) laws := by
  simp [oracleProjectionObjective, certifiedRegularizedObjective,
    oracleRiskObjective, certifiedLawPenalty, oracleProjectionObjectiveWeights]
  ring

/-- At `lam = 0`, the objective is pure oracle risk. -/
theorem oracleProjectionObjective_lam_zero
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (shares : LawComponentShares)
    (laws : ApproxLocalLawsBundle g T fstar) :
    oracleProjectionObjective g x R T fstar 0 shares laws =
      Δ_R_ZR g x R T fstar := by
  simp [oracleProjectionObjective]

/-- At `lam = 1`, the objective is pure local-law projection penalty. -/
theorem oracleProjectionObjective_lam_one
    (g : Summarizer Strings)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (shares : LawComponentShares)
    (laws : ApproxLocalLawsBundle g T fstar) :
    oracleProjectionObjective g x R T fstar 1 shares laws =
      shares.leaf * laws.epsLeaf
        + shares.merge * laws.epsMerge
        + shares.idemp * (((R : ℝ) - 1) * laws.epsIdemp) := by
  simp [oracleProjectionObjective]

/-- Oracle-risk objective bounded by the approximate local-law bundle. -/
theorem oracleRiskObjective_le_of_approx_bundle
    (g : Summarizer Strings)
    (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (h_dist_nonneg : 0 ≤ weights.distortion)
    (laws : ApproxLocalLawsBundle g T fstar) :
    oracleRiskObjective g x R T fstar cost weights ≤
      weights.summary * expectedSummaryCost g x R T cost
        + weights.distortion
            * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  have hΔ :
      Δ_R_ZR g x R T fstar ≤
        laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp :=
    Δ_R_ZR_le_of_approx_bundle g T fstar x R hp hR hbound hbound_global h_mono laws
  unfold oracleRiskObjective
  have hmul :
      weights.distortion * Δ_R_ZR g x R T fstar ≤
        weights.distortion *
          (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    exact mul_le_mul_of_nonneg_left hΔ h_dist_nonneg
  linarith

/-- Certified regularized objective bounded by the summary-cost term plus the
combined approximate-local-law envelope. -/
theorem certifiedRegularizedObjective_le_of_approx_bundle
    (g : Summarizer Strings)
    (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (h_dist_nonneg : 0 ≤ weights.distortion)
    (laws : ApproxLocalLawsBundle g T fstar) :
    certifiedRegularizedObjective g x R T fstar cost weights laws ≤
      weights.summary * expectedSummaryCost g x R T cost
        + (weights.distortion + weights.leaf) * laws.epsLeaf
        + (weights.distortion + weights.merge) * laws.epsMerge
        + (weights.distortion + weights.idemp) * (((R : ℝ) - 1) * laws.epsIdemp) := by
  have hbase :=
    oracleRiskObjective_le_of_approx_bundle g T fstar x R hp hR hbound hbound_global h_mono
      cost weights h_dist_nonneg laws
  unfold certifiedRegularizedObjective certifiedLawPenalty
  have hadd :
      oracleRiskObjective g x R T fstar cost weights
        + (weights.leaf * laws.epsLeaf
            + weights.merge * laws.epsMerge
            + weights.idemp * (((R : ℝ) - 1) * laws.epsIdemp))
      ≤
      (weights.summary * expectedSummaryCost g x R T cost
        + weights.distortion
            * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
        + (weights.leaf * laws.epsLeaf
            + weights.merge * laws.epsMerge
            + weights.idemp * (((R : ℝ) - 1) * laws.epsIdemp)) := by
    exact add_le_add hbase le_rfl
  refine le_trans hadd ?_
  ring_nf
  exact le_rfl

/-- Neural-operator bridge for the oracle/projection objective: uniform
approximation on realized-call compacts plus transfer assumptions yields the
approximate local-law bundle, and the existing approximate-gap theorem then
bounds the oracle part by the same symbolic C1/C2/C3 budgets. -/
theorem oracleProjectionObjective_le_of_uniformApproxExactTheoremBacked
    [PseudoMetricSpace Strings]
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y} {ε : ℝ}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    (hBridge : NeuralOperatorTheoremBridgeAssumptions sStar sApprox T fstar ε)
    (x : Strings) (R : ℕ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p)
    (lam : ℝ) (shares : LawComponentShares)
    (hLamOracle : 0 ≤ 1 - lam) :
    let laws :=
      approxLocalLawsBundle_of_uniformApproxExactTheoremBacked hExact hBridge
    oracleProjectionObjective
        (deterministicSummarizer sApprox) x R T fstar lam shares laws ≤
      (1 - lam) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp)
        + lam *
          (shares.leaf * laws.epsLeaf
            + shares.merge * laws.epsMerge
            + shares.idemp * (((R : ℝ) - 1) * laws.epsIdemp)) := by
  intro laws
  have hΔ :
      Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
        laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp :=
    Δ_R_ZR_le_of_approx_bundle
      (deterministicSummarizer sApprox) T fstar x R hp hR hbound hbound_global h_mono laws
  unfold oracleProjectionObjective
  exact add_le_add
    (mul_le_mul_of_nonneg_left hΔ hLamOracle)
    le_rfl

end FormalProofs.OPT
