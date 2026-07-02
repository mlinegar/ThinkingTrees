import FormalProofs.OPT.NeuralOperatorSpaces

/-!
# Mergeable Projection

This file records the paper-facing distinction between exact recovery of a
mergeable law and projection onto a chosen mergeable-summary class.  It does
not attempt to prove that any particular external sketch implementation is
non-mergeable; it only formalizes the optimization object used when exact
membership is unavailable.
-/

set_option linter.mathlibStandardSet false
set_option relaxedAutoImplicit false
set_option autoImplicit false

open scoped BigOperators
open scoped Classical

noncomputable section

namespace FormalProofs
namespace OPT

/-- Finite empirical risk of a candidate summary law on a finite support. -/
def FiniteEmpiricalRisk {Hyp Sample : Type*} [DecidableEq Sample]
    (support : Finset Sample) (loss : Hyp → Sample → ℝ) (h : Hyp) : ℝ :=
  support.sum (fun x => loss h x)

/-- A candidate `h` is the projection of a target onto a mergeable class `C`
when it lies in `C` and has no larger risk than any other member of `C`. -/
structure IsMergeableProjection {Hyp : Type*}
    (C : Set Hyp) (risk : Hyp → ℝ) (h : Hyp) : Prop where
  mem : h ∈ C
  risk_le : ∀ h' : Hyp, h' ∈ C → risk h ≤ risk h'

/-- Projection candidates are no worse than any other candidate in the
mergeable class. -/
theorem mergeableProjection_risk_le {Hyp : Type*}
    {C : Set Hyp} {risk : Hyp → ℝ} {h h' : Hyp}
    (hp : IsMergeableProjection C risk h) (hh' : h' ∈ C) :
    risk h ≤ risk h' :=
  hp.risk_le h' hh'

/-- If the mergeable class contains an exact zero-risk representative and risk
is nonnegative on that class, any projection has zero residual gap. -/
theorem mergeableProjection_zero_of_exact {Hyp : Type*}
    {C : Set Hyp} {risk : Hyp → ℝ} {h : Hyp}
    (hp : IsMergeableProjection C risk h)
    (hnonneg : ∀ h' : Hyp, h' ∈ C → 0 ≤ risk h')
    (hexact : ∃ h₀ : Hyp, h₀ ∈ C ∧ risk h₀ = 0) :
    risk h = 0 := by
  rcases hexact with ⟨h₀, h₀_mem, h₀_zero⟩
  apply le_antisymm
  · simpa [h₀_zero] using hp.risk_le h₀ h₀_mem
  · exact hnonneg h hp.mem

/-- If every member of the mergeable class has positive risk, the residual
projection gap is structural for every projection. -/
theorem mergeableProjection_structural_gap {Hyp : Type*}
    {C : Set Hyp} {risk : Hyp → ℝ} {h : Hyp}
    (hp : IsMergeableProjection C risk h)
    (hpos : ∀ h' : Hyp, h' ∈ C → 0 < risk h') :
    0 < risk h :=
  hpos h hp.mem

/-- A paper-facing projection gap: `γ` is the attained minimum risk over the
mergeable class `C`.  When the target oracle is not exactly mergeable in the
chosen class, this is the structural approximation error of the projected
mergeable target. -/
structure IsProjectionGap {Hyp : Type*}
    (C : Set Hyp) (risk : Hyp → ℝ) (γ : ℝ) : Prop where
  lower : ∀ h : Hyp, h ∈ C → γ ≤ risk h
  attained : ∃ h : Hyp, h ∈ C ∧ risk h = γ

/-- Any projection candidate realizes the attained projection gap. -/
theorem mergeableProjection_risk_eq_projectionGap {Hyp : Type*}
    {C : Set Hyp} {risk : Hyp → ℝ} {h : Hyp} {γ : ℝ}
    (hp : IsMergeableProjection C risk h)
    (hgap : IsProjectionGap C risk γ) :
    risk h = γ := by
  rcases hgap.attained with ⟨h₀, h₀_mem, h₀_risk⟩
  apply le_antisymm
  · simpa [h₀_risk] using hp.risk_le h₀ h₀_mem
  · exact hgap.lower h hp.mem

/-- If every member of the mergeable class has positive risk, any attained
projection gap is positive. -/
theorem projectionGap_positive_of_structural {Hyp : Type*}
    {C : Set Hyp} {risk : Hyp → ℝ} {γ : ℝ}
    (hgap : IsProjectionGap C risk γ)
    (hpos : ∀ h : Hyp, h ∈ C → 0 < risk h) :
    0 < γ := by
  rcases hgap.attained with ⟨h₀, h₀_mem, h₀_risk⟩
  simpa [h₀_risk] using hpos h₀ h₀_mem

/-! ## Neural-operator projections onto mergeable/local-law sets -/

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- The exact learnable mergeable set: a chosen neural-operator class
intersected with the exact C1/C2/C3 local-law set on a fixed tree and a fixed
score map. In the teacher-first route, the score map supplied here is usually
the selected learned teacher `fhat`, not the unavailable local true oracle.
This is a set of learnable operators, not a linear subspace. -/
def LearnableExactMergeableSet
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings) :
    Set (NeuralOperatorSpaces.NeuralOperator Strings) :=
  NeuralOperatorSpaces.ExactLocalLawNeuralOperators C fstar T

/-- The approximate learnable mergeable set: a chosen neural-operator class
intersected with the approximate C1/C2/C3 local-law set for the currently
selected score map. In LLM/no-local-oracle settings, this is the set induced by
the learned teacher `fhat` and explicit residual budgets. -/
def LearnableApproxMergeableSet
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings)
    (εLeaf εMerge εIdemp : ℝ) :
    Set (NeuralOperatorSpaces.NeuralOperator Strings) :=
  NeuralOperatorSpaces.ApproxLocalLawNeuralOperators
    C fstar T εLeaf εMerge εIdemp

/-- Alias emphasizing that the local-law-compatible set is relative to the
chosen score map, which may be a learned teacher. -/
def LearnableExactMergeableSetForScore :=
  @LearnableExactMergeableSet

/-- Alias emphasizing that approximate mergeability is measured against the
chosen score map, usually `fhat` in the teacher-first route. -/
def LearnableApproxMergeableSetForScore :=
  @LearnableApproxMergeableSet

/-- A learned exact mergeable projection is a risk minimizer over the
representable neural operators that satisfy the exact local laws. -/
def IsLearnedMergeableProjection
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings)
    (risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ)
    (g : NeuralOperatorSpaces.NeuralOperator Strings) : Prop :=
  IsMergeableProjection (LearnableExactMergeableSet C fstar T) risk g

/-- A learned approximate mergeable projection is a risk minimizer over the
representable neural operators whose local-law residuals fit the supplied
budgets. -/
def IsLearnedApproxMergeableProjection
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings)
    (εLeaf εMerge εIdemp : ℝ)
    (risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ)
    (g : NeuralOperatorSpaces.NeuralOperator Strings) : Prop :=
  IsMergeableProjection
    (LearnableApproxMergeableSet C fstar T εLeaf εMerge εIdemp) risk g

/-- The attained structural gap after projecting an oracle-risk objective onto
the exact learnable mergeable set for the chosen score map.  If this gap is
positive, the learned tree is estimating the best mergeable representative,
not recovering the original oracle exactly. -/
def IsLearnedExactProjectionGap
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings)
    (risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ)
    (γ : ℝ) : Prop :=
  IsProjectionGap (LearnableExactMergeableSet C fstar T) risk γ

/-- The attained structural gap after projecting onto an approximate
local-law-compatible set. -/
def IsLearnedApproxProjectionGap
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings)
    (εLeaf εMerge εIdemp : ℝ)
    (risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ)
    (γ : ℝ) : Prop :=
  IsProjectionGap
    (LearnableApproxMergeableSet C fstar T εLeaf εMerge εIdemp) risk γ

/-- Projection candidates minimize risk over the exact learnable mergeable
set. -/
theorem learnedMergeableProjection_risk_le
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g g' : NeuralOperatorSpaces.NeuralOperator Strings}
    (hp : IsLearnedMergeableProjection C fstar T risk g)
    (hg' : g' ∈ LearnableExactMergeableSet C fstar T) :
    risk g ≤ risk g' :=
  mergeableProjection_risk_le hp hg'

/-- Projection candidates minimize risk over the approximate learnable
mergeable set. -/
theorem learnedApproxMergeableProjection_risk_le
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {εLeaf εMerge εIdemp : ℝ}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g g' : NeuralOperatorSpaces.NeuralOperator Strings}
    (hp : IsLearnedApproxMergeableProjection
      C fstar T εLeaf εMerge εIdemp risk g)
    (hg' : g' ∈ LearnableApproxMergeableSet
      C fstar T εLeaf εMerge εIdemp) :
    risk g ≤ risk g' :=
  mergeableProjection_risk_le hp hg'

/-- Exact-set projections realize the exact learnable projection gap. -/
theorem learnedMergeableProjection_risk_eq_projectionGap
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g : NeuralOperatorSpaces.NeuralOperator Strings} {γ : ℝ}
    (hp : IsLearnedMergeableProjection C fstar T risk g)
    (hgap : IsLearnedExactProjectionGap C fstar T risk γ) :
    risk g = γ :=
  mergeableProjection_risk_eq_projectionGap hp hgap

/-- Approximate-set projections realize the approximate learnable projection
gap. -/
theorem learnedApproxMergeableProjection_risk_eq_projectionGap
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {εLeaf εMerge εIdemp : ℝ}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g : NeuralOperatorSpaces.NeuralOperator Strings} {γ : ℝ}
    (hp : IsLearnedApproxMergeableProjection
      C fstar T εLeaf εMerge εIdemp risk g)
    (hgap : IsLearnedApproxProjectionGap
      C fstar T εLeaf εMerge εIdemp risk γ) :
    risk g = γ :=
  mergeableProjection_risk_eq_projectionGap hp hgap

/-- If the exact learnable mergeable set contains a zero-risk representative
and risk is nonnegative on that set, every learned projection has zero
residual gap. -/
theorem learnedMergeableProjection_zero_of_exact
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g : NeuralOperatorSpaces.NeuralOperator Strings}
    (hp : IsLearnedMergeableProjection C fstar T risk g)
    (hnonneg :
      ∀ g' : NeuralOperatorSpaces.NeuralOperator Strings,
        g' ∈ LearnableExactMergeableSet C fstar T → 0 ≤ risk g')
    (hexact :
      ∃ g₀ : NeuralOperatorSpaces.NeuralOperator Strings,
        g₀ ∈ LearnableExactMergeableSet C fstar T ∧ risk g₀ = 0) :
    risk g = 0 :=
  mergeableProjection_zero_of_exact hp hnonneg hexact

/-- Approximate-set version: if the approximate learnable mergeable set
contains a zero-risk representative and risk is nonnegative on that set, every
projection has zero residual gap. -/
theorem learnedApproxMergeableProjection_zero_of_exact
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {εLeaf εMerge εIdemp : ℝ}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g : NeuralOperatorSpaces.NeuralOperator Strings}
    (hp : IsLearnedApproxMergeableProjection
      C fstar T εLeaf εMerge εIdemp risk g)
    (hnonneg :
      ∀ g' : NeuralOperatorSpaces.NeuralOperator Strings,
        g' ∈ LearnableApproxMergeableSet C fstar T εLeaf εMerge εIdemp →
          0 ≤ risk g')
    (hexact :
      ∃ g₀ : NeuralOperatorSpaces.NeuralOperator Strings,
        g₀ ∈ LearnableApproxMergeableSet C fstar T εLeaf εMerge εIdemp
          ∧ risk g₀ = 0) :
    risk g = 0 :=
  mergeableProjection_zero_of_exact hp hnonneg hexact

/-- If every member of the exact learnable mergeable set has positive risk,
the remaining projection gap is structural for every learned projection. -/
theorem learnedMergeableProjection_structural_gap
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g : NeuralOperatorSpaces.NeuralOperator Strings}
    (hp : IsLearnedMergeableProjection C fstar T risk g)
    (hpos :
      ∀ g' : NeuralOperatorSpaces.NeuralOperator Strings,
        g' ∈ LearnableExactMergeableSet C fstar T → 0 < risk g') :
    0 < risk g :=
  mergeableProjection_structural_gap hp hpos

/-- Approximate-set version of the structural-gap statement. -/
theorem learnedApproxMergeableProjection_structural_gap
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {fstar : Strings → Y} {T : BinTree Strings}
    {εLeaf εMerge εIdemp : ℝ}
    {risk : NeuralOperatorSpaces.NeuralOperator Strings → ℝ}
    {g : NeuralOperatorSpaces.NeuralOperator Strings}
    (hp : IsLearnedApproxMergeableProjection
      C fstar T εLeaf εMerge εIdemp risk g)
    (hpos :
      ∀ g' : NeuralOperatorSpaces.NeuralOperator Strings,
        g' ∈ LearnableApproxMergeableSet C fstar T εLeaf εMerge εIdemp →
          0 < risk g') :
    0 < risk g :=
  mergeableProjection_structural_gap hp hpos

/-! ## Local-law weights as an endpoint projection -/

/-- A class-restricted minimizer of the oracle-plus-projection objective over
a neural-operator class `C`. This is a minimizer over the representable class,
not a claim that arbitrary neural operators are mergeable. -/
def IsClassRestrictedBalancedMinimizer
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (obj : NeuralOperatorSpaces.OraclePlusProjection Strings Y)
    (g : NeuralOperatorSpaces.NeuralOperator Strings) : Prop :=
  g ∈ C ∧
    ∀ g' : NeuralOperatorSpaces.NeuralOperator Strings,
      g' ∈ C →
        NeuralOperatorSpaces.OraclePlusProjection.balancedObjective obj g ≤
          NeuralOperatorSpaces.OraclePlusProjection.balancedObjective obj g'

/-- At the `λ = 1` endpoint, a faithful and nonnegative projection penalty
forces any class-restricted minimizer into the exact learnable local-law set,
provided the class contains at least one zero-penalty representative.

This formalizes the safe endpoint claim only: local-law weights penalize
residual distance to the mergeable/local-law set. No convergence claim is made
for arbitrary intermediate `0 < λ < 1` without additional optimizer or
convexity assumptions. -/
theorem classRestrictedBalancedMinimizer_mem_exactLocalLawNeuralOperators_of_lam_one
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (obj : NeuralOperatorSpaces.OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings)
    (g : NeuralOperatorSpaces.NeuralOperator Strings)
    (hmin : IsClassRestrictedBalancedMinimizer C obj g)
    (hLam : obj.lam = 1)
    (hFaith : NeuralOperatorSpaces.FaithfulProjectionPenalty obj fstar T)
    (hNonneg :
      ∀ g' : NeuralOperatorSpaces.NeuralOperator Strings,
        g' ∈ C → 0 ≤ obj.projectionPenalty g')
    (hZero :
      ∃ g₀ : NeuralOperatorSpaces.NeuralOperator Strings,
        g₀ ∈ C ∧ obj.projectionPenalty g₀ = 0) :
    g ∈ NeuralOperatorSpaces.ExactLocalLawNeuralOperators C fstar T := by
  rcases hmin with ⟨hgC, hmin_le⟩
  rcases hZero with ⟨g₀, hg₀C, hg₀Zero⟩
  have hobj_le :
      NeuralOperatorSpaces.OraclePlusProjection.balancedObjective obj g ≤
        NeuralOperatorSpaces.OraclePlusProjection.balancedObjective obj g₀ :=
    hmin_le g₀ hg₀C
  have hgPenalty_le_zero : obj.projectionPenalty g ≤ 0 := by
    rw [NeuralOperatorSpaces.OraclePlusProjection.balanced_objective_lam_one
        obj g hLam,
      NeuralOperatorSpaces.OraclePlusProjection.balanced_objective_lam_one
        obj g₀ hLam,
      hg₀Zero] at hobj_le
    exact hobj_le
  have hgPenalty_zero : obj.projectionPenalty g = 0 :=
    le_antisymm hgPenalty_le_zero (hNonneg g hgC)
  exact ⟨hgC, (hFaith.zero_iff_in_subspace g).mp hgPenalty_zero⟩

end OPT
end FormalProofs
