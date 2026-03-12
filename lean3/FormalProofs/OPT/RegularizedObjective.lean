import FormalProofs.OPT.ApproximateLocalLaws

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

/-- Generic cost on produced summaries (e.g. length, bits, latency proxy). -/
abbrev SummaryCost (Strings : Type*) := Strings → ℝ

/-- Weight bundle for the regularized oracle objective. -/
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

end FormalProofs.OPT
