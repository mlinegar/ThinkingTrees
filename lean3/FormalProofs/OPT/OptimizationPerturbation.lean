import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.RegularizedObjective
import FormalProofs.OPT.TrainingPipeline
import FormalProofs.OPT.AdaptiveChunkingBridge

/-!
# FormalProofs/OPT/OptimizationPerturbation.lean

Uniform objective-perturbation lemmas for argmins and minimizers.

This file packages the optimization-side consequence of bounded oracle
measurement error:

* if a surrogate objective is uniformly within `ε` of the true objective,
  then any exact minimizer of the surrogate is a `2ε`-minimizer of the truth;
* the same statement has oracle-measurable and constrained variants; and
* these generic results instantiate cleanly to DPO under exact theorem-backed
  reduction and to the certified regularized summarizer-selection objective.
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

/-- `ε`-argmin set for an unconstrained objective. -/
def ParamEpsilonArgmin {Θ : Type*} (loss : Θ → ℝ) (ε : ℝ) : Set Θ :=
  {θ | ∀ θ', loss θ ≤ loss θ' + ε}

/-- Pointwise-slack argmin set for an unconstrained objective. The slack paid in
the comparison `θ` versus `θ'` is `eps θ + eps θ'`. -/
def ParamPointwiseEpsilonArgmin {Θ : Type*} (loss : Θ → ℝ) (eps : Θ → ℝ) : Set Θ :=
  {θ | ∀ θ', loss θ ≤ loss θ' + eps θ + eps θ'}

/-- `ε`-argmin set for a constrained objective. -/
def ConstrainedParamEpsilonArgmin {Θ : Type*}
    (loss : Θ → ℝ) (feasible : Θ → Prop) (ε : ℝ) : Set Θ :=
  {θ | feasible θ ∧ ∀ θ', feasible θ' → loss θ ≤ loss θ' + ε}

/-- Pointwise-slack constrained argmin set. -/
def ConstrainedParamPointwiseEpsilonArgmin {Θ : Type*}
    (loss : Θ → ℝ) (feasible : Θ → Prop) (eps : Θ → ℝ) : Set Θ :=
  {θ | feasible θ ∧ ∀ θ', feasible θ' → loss θ ≤ loss θ' + eps θ + eps θ'}

/-- Oracle-measurable `ε`-argmin set for a generic parameterized objective. -/
def OracleMeasurableParamEpsilonArgmin {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (loss : Θ → ℝ) (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y) (ε : ℝ) : Set Θ :=
  {θ | isMeasurable θ fstar ∧
      ∀ θ', isMeasurable θ' fstar → loss θ ≤ loss θ' + ε}

/-- Oracle-measurable pointwise-slack `ε`-argmin set for a generic objective. -/
def OracleMeasurableParamPointwiseEpsilonArgmin {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (loss : Θ → ℝ) (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y) (eps : Θ → ℝ) : Set Θ :=
  {θ | isMeasurable θ fstar ∧
      ∀ θ', isMeasurable θ' fstar → loss θ ≤ loss θ' + eps θ + eps θ'}

/-- Policy-level `ε`-argmin set. -/
def PolicyEpsilonArgmin {Strings A : Type*}
    (loss : Policy Strings A → ℝ) (ε : ℝ) : Set (Policy Strings A) :=
  ParamEpsilonArgmin loss ε

/-- Oracle-measurable policy-level `ε`-argmin set. -/
def OracleMeasurablePolicyEpsilonArgmin {Strings A Y : Type*} [PseudoMetricSpace Y]
    (loss : Policy Strings A → ℝ) (fstar : Strings → Y) (ε : ℝ) :
    Set (Policy Strings A) :=
  OracleMeasurableParamEpsilonArgmin loss DPO.OracleMeasurable fstar ε

/-- Oracle-measurable policy-level pointwise-slack `ε`-argmin set. -/
def OracleMeasurablePolicyPointwiseEpsilonArgmin {Strings A Y : Type*} [PseudoMetricSpace Y]
    (loss : Policy Strings A → ℝ) (fstar : Strings → Y) (eps : Policy Strings A → ℝ) :
    Set (Policy Strings A) :=
  OracleMeasurableParamPointwiseEpsilonArgmin loss DPO.OracleMeasurable fstar eps

/-- Uniformly perturbing an unconstrained objective by at most `ε` turns exact
surrogate minimizers into `2ε`-minimizers for the true objective. -/
theorem paramArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
    {Θ : Type*}
    (lossTrue lossSur : Θ → ℝ)
    (ε : ℝ)
    (hclose : ∀ θ, |lossTrue θ - lossSur θ| ≤ ε) :
    ParamArgmin lossSur ⊆ ParamEpsilonArgmin lossTrue (2 * ε) := by
  intro θ hθ
  dsimp [ParamArgmin, ParamEpsilonArgmin] at hθ ⊢
  intro θ'
  have hθclose := hclose θ
  have hθ'close := hclose θ'
  have hθupper : lossTrue θ ≤ lossSur θ + ε := by
    have h := abs_le.mp hθclose
    linarith
  have hθ'lower : lossSur θ' ≤ lossTrue θ' + ε := by
    have h := abs_le.mp hθ'close
    linarith
  linarith [hθ θ', hθupper, hθ'lower]

/-- Non-uniform objective perturbations turn exact surrogate minimizers into a
pointwise-slack argmin set for the true objective. -/
theorem paramArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
    {Θ : Type*}
    (lossTrue lossSur : Θ → ℝ)
    (eps : Θ → ℝ)
    (hclose : ∀ θ, |lossTrue θ - lossSur θ| ≤ eps θ) :
    ParamArgmin lossSur ⊆ ParamPointwiseEpsilonArgmin lossTrue eps := by
  intro θ hθ
  dsimp [ParamArgmin, ParamPointwiseEpsilonArgmin] at hθ ⊢
  intro θ'
  have hθclose := hclose θ
  have hθ'close := hclose θ'
  have hθupper : lossTrue θ ≤ lossSur θ + eps θ := by
    have h := abs_le.mp hθclose
    linarith
  have hθ'lower : lossSur θ' ≤ lossTrue θ' + eps θ' := by
    have h := abs_le.mp hθ'close
    linarith
  linarith [hθ θ', hθupper, hθ'lower]

/-- Uniformly perturbing a constrained objective by at most `ε` turns exact
surrogate constrained minimizers into constrained `2ε`-minimizers for the true
objective. -/
theorem constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
    {Θ : Type*}
    (lossTrue lossSur : Θ → ℝ)
    (feasible : Θ → Prop)
    (ε : ℝ)
    (hclose : ∀ θ, feasible θ → |lossTrue θ - lossSur θ| ≤ ε) :
    ConstrainedParamEpsilonArgmin lossSur feasible 0 ⊆
      ConstrainedParamEpsilonArgmin lossTrue feasible (2 * ε) := by
  intro θ hθ
  rcases hθ with ⟨hfeas, hmin⟩
  constructor
  · exact hfeas
  · intro θ' hfeas'
    have hθclose := hclose θ hfeas
    have hθ'close := hclose θ' hfeas'
    have hθupper : lossTrue θ ≤ lossSur θ + ε := by
      have h := abs_le.mp hθclose
      linarith
    have hθ'lower : lossSur θ' ≤ lossTrue θ' + ε := by
      have h := abs_le.mp hθ'close
      linarith
    linarith [hmin θ' hfeas', hθupper, hθ'lower]

/-- Non-uniform perturbation version of
`constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation`. -/
theorem constrainedParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
    {Θ : Type*}
    (lossTrue lossSur : Θ → ℝ)
    (feasible : Θ → Prop)
    (eps : Θ → ℝ)
    (hclose : ∀ θ, feasible θ → |lossTrue θ - lossSur θ| ≤ eps θ) :
    ConstrainedParamEpsilonArgmin lossSur feasible 0 ⊆
      ConstrainedParamPointwiseEpsilonArgmin lossTrue feasible eps := by
  intro θ hθ
  rcases hθ with ⟨hfeas, hmin⟩
  constructor
  · exact hfeas
  · intro θ' hfeas'
    have hθclose := hclose θ hfeas
    have hθ'close := hclose θ' hfeas'
    have hθupper : lossTrue θ ≤ lossSur θ + eps θ := by
      have h := abs_le.mp hθclose
      linarith
    have hθ'lower : lossSur θ' ≤ lossTrue θ' + eps θ' := by
      have h := abs_le.mp hθ'close
      linarith
    linarith [hmin θ' hfeas', hθupper, hθ'lower]

/-- Uniformly perturbing an oracle-measurable objective by at most `ε` turns
exact surrogate minimizers into oracle-measurable `2ε`-minimizers for the true
objective. -/
theorem oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (lossTrue lossSur : Θ → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (ε : ℝ)
    (hclose : ∀ θ, isMeasurable θ fstar → |lossTrue θ - lossSur θ| ≤ ε) :
    OracleMeasurableParamArgmin lossSur isMeasurable fstar ⊆
      OracleMeasurableParamEpsilonArgmin lossTrue isMeasurable fstar (2 * ε) := by
  intro θ hθ
  rcases hθ with ⟨hMeas, hmin⟩
  constructor
  · exact hMeas
  · intro θ' hMeas'
    have hθclose := hclose θ hMeas
    have hθ'close := hclose θ' hMeas'
    have hθupper : lossTrue θ ≤ lossSur θ + ε := by
      have h := abs_le.mp hθclose
      linarith
    have hθ'lower : lossSur θ' ≤ lossTrue θ' + ε := by
      have h := abs_le.mp hθ'close
      linarith
    linarith [hmin θ' hMeas', hθupper, hθ'lower]

/-- Non-uniform perturbation version of
`oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation`. -/
theorem oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (lossTrue lossSur : Θ → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (eps : Θ → ℝ)
    (hclose : ∀ θ, isMeasurable θ fstar → |lossTrue θ - lossSur θ| ≤ eps θ) :
    OracleMeasurableParamArgmin lossSur isMeasurable fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar eps := by
  intro θ hθ
  rcases hθ with ⟨hMeas, hmin⟩
  constructor
  · exact hMeas
  · intro θ' hMeas'
    have hθclose := hclose θ hMeas
    have hθ'close := hclose θ' hMeas'
    have hθupper : lossTrue θ ≤ lossSur θ + eps θ := by
      have h := abs_le.mp hθclose
      linarith
    have hθ'lower : lossSur θ' ≤ lossTrue θ' + eps θ' := by
      have h := abs_le.mp hθ'close
      linarith
    linarith [hmin θ' hMeas', hθupper, hθ'lower]

/-- Two-stage uniform perturbation calculus: if `lossTrue` is uniformly close to
an intermediate oracle objective and that oracle objective is uniformly close to
the surrogate objective, then surrogate minimizers are `2(ε₁+ε₂)`-minimizers
for truth. -/
theorem oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_two_stage_loss_perturbation
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (lossTrue lossOracle lossSur : Θ → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (ε_oracle ε_transport : ℝ)
    (hcloseOracle :
      ∀ θ, isMeasurable θ fstar → |lossTrue θ - lossOracle θ| ≤ ε_oracle)
    (hcloseTransport :
      ∀ θ, isMeasurable θ fstar → |lossOracle θ - lossSur θ| ≤ ε_transport) :
    OracleMeasurableParamArgmin lossSur isMeasurable fstar ⊆
      OracleMeasurableParamEpsilonArgmin lossTrue isMeasurable fstar
        (2 * (ε_oracle + ε_transport)) := by
  have hcloseTotal :
      ∀ θ, isMeasurable θ fstar →
        |lossTrue θ - lossSur θ| ≤ ε_oracle + ε_transport := by
    intro θ hMeas
    have h1 := hcloseOracle θ hMeas
    have h2 := hcloseTransport θ hMeas
    have h_triangle :
        |lossTrue θ - lossSur θ| ≤
          |lossTrue θ - lossOracle θ| + |lossOracle θ - lossSur θ| := by
      have hdecomp :
          lossTrue θ - lossSur θ =
            (lossTrue θ - lossOracle θ) + (lossOracle θ - lossSur θ) := by
        ring
      rw [hdecomp]
      exact abs_add_le _ _
    linarith
  simpa [two_mul, add_assoc, add_left_comm, add_comm] using
    (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
      (lossTrue := lossTrue) (lossSur := lossSur)
      (isMeasurable := isMeasurable) (fstar := fstar)
      (ε := ε_oracle + ε_transport) hcloseTotal)

/-- Non-uniform two-stage perturbation calculus. -/
theorem oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_two_stage_loss_perturbation
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (lossTrue lossOracle lossSur : Θ → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (epsOracle epsTransport : Θ → ℝ)
    (hcloseOracle :
      ∀ θ, isMeasurable θ fstar → |lossTrue θ - lossOracle θ| ≤ epsOracle θ)
    (hcloseTransport :
      ∀ θ, isMeasurable θ fstar → |lossOracle θ - lossSur θ| ≤ epsTransport θ) :
    OracleMeasurableParamArgmin lossSur isMeasurable fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar
        (fun θ => epsOracle θ + epsTransport θ) := by
  have hcloseTotal :
      ∀ θ, isMeasurable θ fstar →
        |lossTrue θ - lossSur θ| ≤ epsOracle θ + epsTransport θ := by
    intro θ hMeas
    have h1 := hcloseOracle θ hMeas
    have h2 := hcloseTransport θ hMeas
    have h_triangle :
        |lossTrue θ - lossSur θ| ≤
          |lossTrue θ - lossOracle θ| + |lossOracle θ - lossSur θ| := by
      have hdecomp :
          lossTrue θ - lossSur θ =
            (lossTrue θ - lossOracle θ) + (lossOracle θ - lossSur θ) := by
        ring
      rw [hdecomp]
      exact abs_add_le _ _
    linarith
  simpa [add_assoc, add_left_comm, add_comm] using
    (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
      (lossTrue := lossTrue) (lossSur := lossSur)
      (isMeasurable := isMeasurable) (fstar := fstar)
      (eps := fun θ => epsOracle θ + epsTransport θ) hcloseTotal)

/-- Expected objective induced by a stochastic adaptive tree policy. -/
noncomputable def ExpectedAdaptiveTreeObjective {Strings Θ : Type*}
    (τ : StochasticAdaptiveTreeMap Strings) (x : Strings)
    (lossTree : Θ → BinTree Strings → ℝ) (θ : Θ) : ℝ :=
  Exp (τ x) (fun T => lossTree θ T)

/-- If the expected absolute pointwise tree gap is finite, then the gap between
the true objective and the expected tree objective is bounded by that expected
absolute gap. -/
theorem abs_sub_expectedAdaptiveTreeObjective_le_expected_abs_sub
    {Strings Θ : Type*}
    (τ : StochasticAdaptiveTreeMap Strings) (x : Strings)
    (lossTrue : Θ → ℝ)
    (lossTree : Θ → BinTree Strings → ℝ)
    (θ : Θ)
    (hsumm_abs :
      Summable (fun T => (τ x T).toReal * |lossTrue θ - lossTree θ T|)) :
    |lossTrue θ - ExpectedAdaptiveTreeObjective τ x lossTree θ| ≤
      Exp (τ x) (fun T => |lossTrue θ - lossTree θ T|) := by
  have hsum_const :
      Summable (fun T => (τ x T).toReal * lossTrue θ) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) (fun _ => lossTrue θ) |lossTrue θ|
      (abs_nonneg _)
      (fun _ => by simp)
  have hsum_abs_signed :
      Summable (fun T =>
        |(τ x T).toReal * (lossTrue θ - lossTree θ T)|) := by
    simpa [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg] using hsumm_abs
  have hsum_signed :
      Summable (fun T => (τ x T).toReal * (lossTrue θ - lossTree θ T)) :=
    Summable.of_abs hsum_abs_signed
  have hsum_tree :
      Summable (fun T => (τ x T).toReal * lossTree θ T) := by
    have hEq :
        (fun T => (τ x T).toReal * lossTree θ T) =
          fun T =>
            (τ x T).toReal * lossTrue θ -
              ((τ x T).toReal * (lossTrue θ - lossTree θ T)) := by
      funext T
      ring
    rw [hEq]
    exact hsum_const.sub hsum_signed
  have hconst_tsum :
      ∑' T, (τ x T).toReal * lossTrue θ = lossTrue θ := by
    rw [tsum_mul_right, PMF.toReal_tsum_coe (τ x)]
    ring
  have hdecomp :
      lossTrue θ - ∑' T, (τ x T).toReal * lossTree θ T =
        ∑' T, (τ x T).toReal * (lossTrue θ - lossTree θ T) := by
    calc
      lossTrue θ - ∑' T, (τ x T).toReal * lossTree θ T
          = (∑' T, (τ x T).toReal * lossTrue θ) -
              ∑' T, (τ x T).toReal * lossTree θ T := by
                rw [hconst_tsum]
      _ = ∑' T, ((τ x T).toReal * lossTrue θ - (τ x T).toReal * lossTree θ T) := by
            rw [Summable.tsum_sub hsum_const hsum_tree]
      _ = ∑' T, (τ x T).toReal * (lossTrue θ - lossTree θ T) := by
            congr 1
            ext T
            ring
  have htsum :
      |∑' T, (τ x T).toReal * (lossTrue θ - lossTree θ T)| ≤
        ∑' T, |(τ x T).toReal * (lossTrue θ - lossTree θ T)| :=
    abs_tsum_le_tsum_abs' _ hsum_signed hsum_abs_signed
  calc
    |lossTrue θ - ExpectedAdaptiveTreeObjective τ x lossTree θ|
        = |∑' T, (τ x T).toReal * (lossTrue θ - lossTree θ T)| := by
            unfold ExpectedAdaptiveTreeObjective Exp
            rw [hdecomp]
    _ ≤ ∑' T, |(τ x T).toReal * (lossTrue θ - lossTree θ T)| := htsum
    _ = Exp (τ x) (fun T => |lossTrue θ - lossTree θ T|) := by
          unfold Exp
          congr 1
          ext T
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]

/-- Expected-tree surrogate perturbation with parameter-dependent slack. -/
theorem oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (τ : StochasticAdaptiveTreeMap Strings) (x : Strings)
    (lossTrue : Θ → ℝ)
    (lossTree : Θ → BinTree Strings → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (eps : Θ → ℝ)
    (hsumm_abs :
      ∀ θ, isMeasurable θ fstar →
        Summable (fun T => (τ x T).toReal * |lossTrue θ - lossTree θ T|))
    (hclose :
      ∀ θ, isMeasurable θ fstar →
        Exp (τ x) (fun T => |lossTrue θ - lossTree θ T|) ≤ eps θ) :
    OracleMeasurableParamArgmin
        (ExpectedAdaptiveTreeObjective τ x lossTree) isMeasurable fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar eps := by
  have hgap :
      ∀ θ, isMeasurable θ fstar →
        |lossTrue θ - ExpectedAdaptiveTreeObjective τ x lossTree θ| ≤ eps θ := by
    intro θ hMeas
    exact le_trans
      (abs_sub_expectedAdaptiveTreeObjective_le_expected_abs_sub
        (τ := τ) (x := x) (lossTrue := lossTrue) (lossTree := lossTree)
        (θ := θ) (hsumm_abs := hsumm_abs θ hMeas))
      (hclose θ hMeas)
  exact oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
    (lossTrue := lossTrue)
    (lossSur := ExpectedAdaptiveTreeObjective τ x lossTree)
    (isMeasurable := isMeasurable)
    (fstar := fstar)
    (eps := eps)
    hgap

/-- Uniform-slack expected-tree surrogate perturbation. -/
theorem oracleMeasurableParamArgmin_subset_epsilonArgmin_of_expectedTree_loss_perturbation
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (τ : StochasticAdaptiveTreeMap Strings) (x : Strings)
    (lossTrue : Θ → ℝ)
    (lossTree : Θ → BinTree Strings → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (ε : ℝ)
    (hsumm_abs :
      ∀ θ, isMeasurable θ fstar →
        Summable (fun T => (τ x T).toReal * |lossTrue θ - lossTree θ T|))
    (hclose :
      ∀ θ, isMeasurable θ fstar →
        Exp (τ x) (fun T => |lossTrue θ - lossTree θ T|) ≤ ε) :
    OracleMeasurableParamArgmin
        (ExpectedAdaptiveTreeObjective τ x lossTree) isMeasurable fstar ⊆
      OracleMeasurableParamEpsilonArgmin lossTrue isMeasurable fstar (2 * ε) := by
  have hgap :
      ∀ θ, isMeasurable θ fstar →
        |lossTrue θ - ExpectedAdaptiveTreeObjective τ x lossTree θ| ≤ ε := by
    intro θ hMeas
    exact le_trans
      (abs_sub_expectedAdaptiveTreeObjective_le_expected_abs_sub
        (τ := τ) (x := x) (lossTrue := lossTrue) (lossTree := lossTree)
        (θ := θ) (hsumm_abs := hsumm_abs θ hMeas))
      (hclose θ hMeas)
  exact oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
    (lossTrue := lossTrue)
    (lossSur := ExpectedAdaptiveTreeObjective τ x lossTree)
    (isMeasurable := isMeasurable)
    (fstar := fstar)
    (ε := ε)
    hgap

section DPO

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {A : Type*}

/-- If the true policy objective is uniformly within `ε` of the oracle-indexed
DPO objective, then any oracle-measurable DPO minimizer on `ZR` is a
`2ε`-minimizer for the true objective on the original example. -/
theorem dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar)
    (hclose :
      ∀ pol, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ ε) :
    OracleMeasurablePolicyArgmin
        (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar ⊆
      OracleMeasurablePolicyEpsilonArgmin lossTrue fstar (2 * ε) := by
  intro pol hpol
  have hcloseZR :
      ∀ pol, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤ ε := by
    intro pol hMeas
    have hEq :
        ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
          ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen :=
      dpo_equivalence_via_ZR_of_exactTheoremBacked
        (fstar := fstar) (pol := pol) (pol_ref := pol_ref) (β := β)
        (gen := gen) (g := g) (x := x) (R := R) (T := T)
        hp hExact hR hMeas h_meas_ref h_gen
    simpa [hEq] using hclose pol hMeas
  simpa [OracleMeasurablePolicyArgmin, OracleMeasurablePolicyEpsilonArgmin]
    using
      (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
        (lossTrue := lossTrue)
        (lossSur := fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
        (isMeasurable := DPO.OracleMeasurable)
        (fstar := fstar) (ε := ε) hcloseZR hpol)

/-- Pointwise form of
`dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement`. -/
theorem dpo_true_loss_le_best_plus_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (polStar pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar)
    (hclose :
      ∀ pol, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ ε)
    (hArgmin :
      polStar ∈ OracleMeasurablePolicyArgmin
        (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar) :
    ∀ pol, DPO.OracleMeasurable pol fstar → lossTrue polStar ≤ lossTrue pol + 2 * ε := by
  have hNear :
      polStar ∈ OracleMeasurablePolicyEpsilonArgmin lossTrue fstar (2 * ε) :=
    dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
      (fstar := fstar) (lossTrue := lossTrue) (pol_ref := pol_ref)
      (gen := gen) (g := g) (x := x) (R := R) (T := T)
      (β := β) (ε := ε)
      hp hExact hR h_meas_ref h_gen hclose hArgmin
  exact hNear.2

/-- Exact theorem-backed DPO argmin transfer with policy-dependent oracle
measurement error. -/
theorem dpo_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (epsOracle : Policy Strings A → ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar)
    (hclose :
      ∀ pol, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ epsOracle pol) :
    OracleMeasurablePolicyArgmin
        (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar ⊆
      OracleMeasurablePolicyPointwiseEpsilonArgmin lossTrue fstar epsOracle := by
  intro pol hpol
  have hcloseZR :
      ∀ pol, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤ epsOracle pol := by
    intro pol hMeas
    have hEq :
        ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
          ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen :=
      dpo_equivalence_via_ZR_of_exactTheoremBacked
        (fstar := fstar) (pol := pol) (pol_ref := pol_ref) (β := β)
        (gen := gen) (g := g) (x := x) (R := R) (T := T)
        hp hExact hR hMeas h_meas_ref h_gen
    simpa [hEq] using hclose pol hMeas
  simpa [OracleMeasurablePolicyArgmin, OracleMeasurablePolicyPointwiseEpsilonArgmin]
    using
      (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
        (lossTrue := lossTrue)
        (lossSur := fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
        (isMeasurable := DPO.OracleMeasurable)
        (fstar := fstar) (eps := epsOracle) hcloseZR hpol)

/-- Approximate-DPO version: if every oracle-measurable policy in a class has a
uniform theorem-backed transport bound from `PMF.pure x` to `ZR`, and the true
objective is uniformly within `ε` of the oracle-indexed objective on
`PMF.pure x`, then DPO argmins on `ZR` are `2(ε + δ)`-optimal for truth. -/
theorem dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_class : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy Strings A, ∀ x' (p : A × A),
        |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_all_meas : ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar)
    (h_all_lip : ∀ pol : Policy Strings A, PolicyLipschitz pol pol_ref fstar L_class)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (ε : ℝ)
    (hclose :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ ε) :
    OracleMeasurablePolicyArgmin
        (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar ⊆
      OracleMeasurablePolicyEpsilonArgmin lossTrue fstar
        (2 * (ε + 2 * |β| * (L_class : ℝ) *
          (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))) := by
  have htransport :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
          ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
          2 * |β| * (L_class : ℝ) *
            (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    intro pol hMeas
    exact dpo_gap_via_approx_bundle
      (fstar := fstar) (pol := pol) (pol_ref := pol_ref) (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      (β := β) (L_pol := L_class)
      hp hR
      D_max hD_max h_dist_bound hbound hbound_global
      Loss_max hLoss_max (hLoss_bound pol)
      hMeas h_meas_ref (h_all_lip pol) h_gen_fixed h_mono laws
  simpa [OracleMeasurablePolicyArgmin, OracleMeasurablePolicyEpsilonArgmin]
    using
      (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_two_stage_loss_perturbation
        (lossTrue := lossTrue)
        (lossOracle := fun pol => ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
        (lossSur := fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
        (isMeasurable := DPO.OracleMeasurable)
        (fstar := fstar)
        (ε_oracle := ε)
      (ε_transport := 2 * |β| * (L_class : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
      hclose htransport)

/-- Approximate-bundle DPO argmin transfer with policy-dependent oracle
measurement error. -/
theorem dpo_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_class : NNReal)
    (epsOracle : Policy Strings A → ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy Strings A, ∀ x' (p : A × A),
        |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_all_meas : ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar)
    (h_all_lip : ∀ pol : Policy Strings A, PolicyLipschitz pol pol_ref fstar L_class)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (hclose :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ epsOracle pol) :
    OracleMeasurablePolicyArgmin
        (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar ⊆
      OracleMeasurablePolicyPointwiseEpsilonArgmin lossTrue fstar
        (fun pol =>
          epsOracle pol +
            2 * |β| * (L_class : ℝ) *
              (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp)) := by
  have htransport :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
          ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
          2 * |β| * (L_class : ℝ) *
            (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    intro pol hMeas
    exact dpo_gap_via_approx_bundle
      (fstar := fstar) (pol := pol) (pol_ref := pol_ref) (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      (β := β) (L_pol := L_class)
      hp hR
      D_max hD_max h_dist_bound hbound hbound_global
      Loss_max hLoss_max (hLoss_bound pol)
      hMeas h_meas_ref (h_all_lip pol) h_gen_fixed h_mono laws
  simpa [OracleMeasurablePolicyArgmin, OracleMeasurablePolicyPointwiseEpsilonArgmin,
    add_assoc, add_left_comm, add_comm] using
      (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_two_stage_loss_perturbation
        (lossTrue := lossTrue)
        (lossOracle := fun pol => ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
        (lossSur := fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
        (isMeasurable := DPO.OracleMeasurable)
        (fstar := fstar)
        (epsOracle := epsOracle)
        (epsTransport := fun _ =>
          2 * |β| * (L_class : ℝ) *
            (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
        hclose htransport)

/-- Audit-event version of
`dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement`.
Once a nodewise empirical audit certifies an approximate bundle, the same
near-optimality guarantee holds on that event. -/
theorem dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_nodewiseEmpiricalAudit_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_class : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy Strings A, ∀ x' (p : A × A),
        |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_all_meas : ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar)
    (h_all_lip : ∀ pol : Policy Strings A, PolicyLipschitz pol pol_ref fstar L_class)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (audit : NodewiseEmpiricalAuditWithConfidence g T fstar)
    (h_event : audit.event)
    (ε : ℝ)
    (hclose :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ ε) :
    OracleMeasurablePolicyArgmin
        (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar ⊆
      OracleMeasurablePolicyEpsilonArgmin lossTrue fstar
        (2 * (ε + 2 * |β| * (L_class : ℝ) *
          ((approx_bundle_of_nodewise_empirical_confidence_event g T fstar audit h_event).epsLeaf +
           (approx_bundle_of_nodewise_empirical_confidence_event g T fstar audit h_event).epsMerge +
           ((R : ℝ) - 1) *
             (approx_bundle_of_nodewise_empirical_confidence_event g T fstar audit h_event).epsIdemp))) := by
  exact dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement
    (fstar := fstar) (lossTrue := lossTrue) (pol_ref := pol_ref) (gen := gen)
    (g := g) (x := x) (R := R) (T := T)
    (β := β) (L_class := L_class) hp hR
    D_max hD_max h_dist_bound hbound hbound_global
    Loss_max hLoss_max hLoss_bound
    h_meas_ref h_all_meas h_all_lip h_gen_fixed h_mono
    (approx_bundle_of_nodewise_empirical_confidence_event g T fstar audit h_event)
    ε hclose

/-- Tree-level DPO optimizer transfer: if a stochastic adaptive tree policy has
approximate local laws and the oracle-to-truth gap is controlled by a
tree-dependent envelope, then exact minimizers of the expected tree objective
are near-optimal for the true objective. -/
theorem dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_class : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy Strings A, ∀ x' (p : A × A),
        |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_class_lip :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        PolicyLipschitz pol pol_ref fstar L_class)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx :
      StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (oracle_err : Policy Strings A → BinTree Strings → ℝ)
    (h_oracle :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        ∀ T,
          |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ oracle_err pol T)
    (hsumm_true :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        Summable (fun T =>
          (τ x T).toReal *
            |lossTrue pol - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_oracle :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        Summable (fun T => (τ x T).toReal * oracle_err pol T))
    (hsumm_gap :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        Summable (fun T =>
          (τ x T).toReal *
            |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
              ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_class : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    OracleMeasurablePolicyArgmin
        (ExpectedAdaptiveTreeObjective τ x
          (fun pol T => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)) fstar ⊆
      OracleMeasurablePolicyPointwiseEpsilonArgmin lossTrue fstar
        (fun pol =>
          Exp (τ x) (oracle_err pol) +
            Exp (τ x) (fun T =>
              2 * |β| * (L_class : ℝ) *
                (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))) := by
  exact oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation
    (τ := τ) (x := x)
    (lossTrue := lossTrue)
    (lossTree := fun pol T => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
    (isMeasurable := DPO.OracleMeasurable)
    (fstar := fstar)
    (eps := fun pol =>
      Exp (τ x) (oracle_err pol) +
        Exp (τ x) (fun T =>
          2 * |β| * (L_class : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))
    (hsumm_abs := hsumm_true)
    (hclose := by
      intro pol hMeas
      exact Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
        fstar pol pol_ref gen g τ β L_class
        D_max hD_max h_dist_bound
        Loss_max hLoss_max (hLoss_bound pol)
        hMeas h_meas_ref (h_class_lip pol hMeas) h_gen_fixed
        hbound hbound_global h_mono
        ε_leaf ε_merge ε_idemp
        h_sound h_approx x R hR
        (lossTrue pol) (oracle_err pol)
        (h_oracle pol hMeas)
        (hsumm_true pol hMeas)
        (hsumm_oracle pol hMeas)
        (hsumm_gap pol hMeas)
        hsumm_budget)

end DPO

section GRPOPL

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {A : Type*}
variable {k : ℕ}

/-- Exact theorem-backed GRPO-PL argmin transfer under uniform oracle
measurement error. -/
theorem grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (hclose :
      ∀ pol, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ ε) :
    OracleMeasurableParamArgmin
        (fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
        (fun pol f => GRPOOracleMeasurable pol f) fstar ⊆
      OracleMeasurableParamEpsilonArgmin
        lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar (2 * ε) := by
  intro pol hpol
  have hcloseZR :
      ∀ pol, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤ ε := by
    intro pol hMeas
    have hEq :
        ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
          ExpectedGRPOLoss pol ranker (ZR g x R T) gen :=
      grpo_equivalence_via_ZR_of_exactTheoremBacked
        (fstar := fstar) (pol := pol) (ranker := ranker) (gen := gen)
        (g := g) (x := x) (R := R) (T := T)
        hp hExact hR hMeas h_ranker h_gen
    simpa [hEq] using hclose pol hMeas
  simpa using
    (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
      (lossTrue := lossTrue)
      (lossSur := fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
      (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
      (fstar := fstar) (ε := ε) hcloseZR hpol)

/-- Exact theorem-backed GRPO-PL argmin transfer with policy-dependent oracle
measurement error. -/
theorem grpo_pl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (epsOracle : Policy' Strings A → ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (hclose :
      ∀ pol, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ epsOracle pol) :
    OracleMeasurableParamArgmin
        (fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
        (fun pol f => GRPOOracleMeasurable pol f) fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin
        lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar epsOracle := by
  intro pol hpol
  have hcloseZR :
      ∀ pol, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤ epsOracle pol := by
    intro pol hMeas
    have hEq :
        ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
          ExpectedGRPOLoss pol ranker (ZR g x R T) gen :=
      grpo_equivalence_via_ZR_of_exactTheoremBacked
        (fstar := fstar) (pol := pol) (ranker := ranker) (gen := gen)
        (g := g) (x := x) (R := R) (T := T)
        hp hExact hR hMeas h_ranker h_gen
    simpa [hEq] using hclose pol hMeas
  simpa using
    (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
      (lossTrue := lossTrue)
      (lossSur := fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
      (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
      (fstar := fstar) (eps := epsOracle) hcloseZR hpol)

/-- Approximate-bundle GRPO-PL argmin transfer under uniform oracle measurement
error. -/
theorem grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_all_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo
          (h_all_lip pol) h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (ε : ℝ)
    (hclose :
      ∀ pol : Policy' Strings A, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ ε) :
    OracleMeasurableParamArgmin
        (fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
        (fun pol f => GRPOOracleMeasurable pol f) fstar ⊆
      OracleMeasurableParamEpsilonArgmin
        lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar
        (2 * (ε + (L_grpo : ℝ) *
          (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))) := by
  have htransport :
      ∀ pol : Policy' Strings A, GRPOOracleMeasurable pol fstar →
        |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
          ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
          (L_grpo : ℝ) *
            (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    intro pol _hMeas
    exact grpo_pl_gap_via_approx_bundle (k := k)
      (fstar := fstar) (pol := pol) (ranker := ranker) (gen := gen)
      (g := g) (x := x) (R := R) (T := T) (L_grpo := L_grpo)
      D_max hD_max h_dist_bound
      Loss_max hLoss_max (hLoss_bound pol)
      (h_all_lip pol) h_ranker (h_rum pol) h_gen_fixed
      hp hR hbound hbound_global h_mono laws
  simpa using
    (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_two_stage_loss_perturbation
      (lossTrue := lossTrue)
      (lossOracle := fun pol => ExpectedGRPOLoss pol ranker (PMF.pure x) gen)
      (lossSur := fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
      (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
      (fstar := fstar)
      (ε_oracle := ε)
      (ε_transport := (L_grpo : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
      hclose htransport)

/-- Approximate-bundle GRPO-PL argmin transfer with policy-dependent oracle
measurement error. -/
theorem grpo_pl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo : NNReal)
    (epsOracle : Policy' Strings A → ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_all_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo
          (h_all_lip pol) h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (hclose :
      ∀ pol : Policy' Strings A, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ epsOracle pol) :
    OracleMeasurableParamArgmin
        (fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
        (fun pol f => GRPOOracleMeasurable pol f) fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin
        lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar
        (fun pol =>
          epsOracle pol +
            (L_grpo : ℝ) * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp)) := by
  have htransport :
      ∀ pol : Policy' Strings A, GRPOOracleMeasurable pol fstar →
        |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
          ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
          (L_grpo : ℝ) * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    intro pol _hMeas
    exact grpo_pl_gap_via_approx_bundle (k := k)
      (fstar := fstar) (pol := pol) (ranker := ranker) (gen := gen)
      (g := g) (x := x) (R := R) (T := T) (L_grpo := L_grpo)
      D_max hD_max h_dist_bound
      Loss_max hLoss_max (hLoss_bound pol)
      (h_all_lip pol) h_ranker (h_rum pol) h_gen_fixed
      hp hR hbound hbound_global h_mono laws
  simpa [add_assoc, add_left_comm, add_comm] using
    (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_two_stage_loss_perturbation
      (lossTrue := lossTrue)
      (lossOracle := fun pol => ExpectedGRPOLoss pol ranker (PMF.pure x) gen)
      (lossSur := fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
      (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
      (fstar := fstar)
      (epsOracle := epsOracle)
      (epsTransport := fun _ =>
        (L_grpo : ℝ) * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
      hclose htransport)

/-- Tree-level GRPO-PL optimizer transfer for stochastic adaptive tree
policies with tree-indexed oracle-measurement uncertainty. -/
theorem grpo_pl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_class_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo
          (h_class_lip pol) h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx :
      StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (oracle_err : Policy' Strings A → BinTree Strings → ℝ)
    (h_oracle :
      ∀ pol : Policy' Strings A, ∀ T,
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ oracle_err pol T)
    (hsumm_true :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |lossTrue pol - ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_oracle :
      ∀ pol : Policy' Strings A,
        Summable (fun T => (τ x T).toReal * oracle_err pol T))
    (hsumm_gap :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
              ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    OracleMeasurableParamArgmin
        (ExpectedAdaptiveTreeObjective τ x
          (fun pol T => ExpectedGRPOLoss pol ranker (ZR g x R T) gen))
        (fun pol f => GRPOOracleMeasurable pol f) fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin
        lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar
        (fun pol =>
          Exp (τ x) (oracle_err pol) +
            Exp (τ x) (fun T =>
              (L_grpo : ℝ) *
                (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))) := by
  exact oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation
    (τ := τ) (x := x)
    (lossTrue := lossTrue)
    (lossTree := fun pol T => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
    (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
    (fstar := fstar)
    (eps := fun pol =>
      Exp (τ x) (oracle_err pol) +
        Exp (τ x) (fun T =>
          (L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))
    (hsumm_abs := by
      intro pol _hMeas
      exact hsumm_true pol)
    (hclose := by
      intro pol _hMeas
      exact Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
        (k := k)
        fstar pol ranker gen g τ L_grpo
        D_max hD_max h_dist_bound
        Loss_max hLoss_max (hLoss_bound pol)
        (h_class_lip pol) h_ranker (h_rum pol) h_gen_fixed
        hbound hbound_global h_mono
        ε_leaf ε_merge ε_idemp
        h_sound h_approx x R hR
        (lossTrue pol) (oracle_err pol)
        (h_oracle pol)
        (hsumm_true pol)
        (hsumm_oracle pol)
        (hsumm_gap pol)
        hsumm_budget)

end GRPOPL

section GRPORL

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {A : Type*}
variable {k : ℕ}

/-- Exact theorem-backed GRPO-RL argmin transfer under uniform oracle
measurement error. -/
theorem grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (hclose :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤ ε) :
    OracleMeasurableParamArgmin
        (fun pol =>
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar ⊆
      OracleMeasurableParamEpsilonArgmin
        lossTrue
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar (2 * ε) := by
  intro pol hpol
  have hcloseZR :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤ ε := by
    intro pol hMeas
    have hEq :
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen :=
      grpo_rl_equivalence_via_ZR_of_exactTheoremBacked
        (fstar := fstar) (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta)
        (gen := gen) (g := g) (x := x) (R := R) (T := T)
        hp hExact hR hMeas h_gen
    simpa [hEq] using hclose pol hMeas
  simpa using
    (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
      (lossTrue := lossTrue)
      (lossSur := fun pol =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
      (isMeasurable := fun pol f =>
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
      (fstar := fstar) (ε := ε) hcloseZR hpol)

/-- Exact theorem-backed GRPO-RL argmin transfer with policy-dependent oracle
measurement error. -/
theorem grpo_rl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (epsOracle : Policy' Strings A → ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (hclose :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤
            epsOracle pol) :
    OracleMeasurableParamArgmin
        (fun pol =>
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin
        lossTrue
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar epsOracle := by
  intro pol hpol
  have hcloseZR :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
            epsOracle pol := by
    intro pol hMeas
    have hEq :
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen :=
      grpo_rl_equivalence_via_ZR_of_exactTheoremBacked
        (fstar := fstar) (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta)
        (gen := gen) (g := g) (x := x) (R := R) (T := T)
        hp hExact hR hMeas h_gen
    simpa [hEq] using hclose pol hMeas
  simpa using
    (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_loss_perturbation
      (lossTrue := lossTrue)
      (lossSur := fun pol =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
      (isMeasurable := fun pol f =>
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
      (fstar := fstar) (eps := epsOracle) hcloseZR hpol)

/-- Approximate-bundle GRPO-RL argmin transfer under uniform oracle
measurement error. -/
theorem grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo_rl : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_all_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x')
          L_grpo_rl (h_all_lip pol) h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (ε_true : ℝ)
    (hclose :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤ ε_true) :
    OracleMeasurableParamArgmin
        (fun pol =>
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar ⊆
      OracleMeasurableParamEpsilonArgmin
        lossTrue
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar
        (2 * (ε_true + (L_grpo_rl : ℝ) *
          (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))) := by
  have htransport :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
            (L_grpo_rl : ℝ) *
              (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    intro pol _hMeas
    exact grpo_rl_gap_via_approx_bundle (k := k)
      (fstar := fstar) (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta) (gen := gen)
      (g := g) (x := x) (R := R) (T := T) (L_grpo_rl := L_grpo_rl)
      D_max hD_max h_dist_bound
      Loss_max hLoss_max (hLoss_bound pol)
      (h_all_lip pol) h_old_lip h_ref_lip h_reward_lip
      (h_rum pol) h_gen_fixed
      hp hR hbound hbound_global h_mono laws
  simpa using
    (oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_two_stage_loss_perturbation
      (lossTrue := lossTrue)
      (lossOracle := fun pol =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen)
      (lossSur := fun pol =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
      (isMeasurable := fun pol f =>
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
      (fstar := fstar)
      (ε_oracle := ε_true)
      (ε_transport := (L_grpo_rl : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
      hclose htransport)

/-- Approximate-bundle GRPO-RL argmin transfer with policy-dependent oracle
measurement error. -/
theorem grpo_rl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo_rl : NNReal)
    (epsOracle : Policy' Strings A → ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_all_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x')
          L_grpo_rl (h_all_lip pol) h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (hclose :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤
            epsOracle pol) :
    OracleMeasurableParamArgmin
        (fun pol =>
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin
        lossTrue
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar
        (fun pol =>
          epsOracle pol +
            (L_grpo_rl : ℝ) *
              (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp)) := by
  have htransport :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
            (L_grpo_rl : ℝ) *
              (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
    intro pol _hMeas
    exact grpo_rl_gap_via_approx_bundle (k := k)
      (fstar := fstar) (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta) (gen := gen)
      (g := g) (x := x) (R := R) (T := T) (L_grpo_rl := L_grpo_rl)
      D_max hD_max h_dist_bound
      Loss_max hLoss_max (hLoss_bound pol)
      (h_all_lip pol) h_old_lip h_ref_lip h_reward_lip
      (h_rum pol) h_gen_fixed
      hp hR hbound hbound_global h_mono laws
  simpa [add_assoc, add_left_comm, add_comm] using
    (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_two_stage_loss_perturbation
      (lossTrue := lossTrue)
      (lossOracle := fun pol =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen)
      (lossSur := fun pol =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
      (isMeasurable := fun pol f =>
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
      (fstar := fstar)
      (epsOracle := epsOracle)
      (epsTransport := fun _ =>
        (L_grpo_rl : ℝ) *
          (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp))
      hclose htransport)

/-- Tree-level GRPO-RL optimizer transfer for stochastic adaptive tree
policies with tree-indexed oracle-measurement uncertainty. -/
theorem grpo_rl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (L_grpo_rl : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_class_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x')
          L_grpo_rl (h_class_lip pol) h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx :
      StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (oracle_err : Policy' Strings A → BinTree Strings → ℝ)
    (h_oracle :
      ∀ pol : Policy' Strings A, ∀ T,
        |lossTrue pol -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤
          oracle_err pol T)
    (hsumm_true :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |lossTrue pol -
                ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_oracle :
      ∀ pol : Policy' Strings A,
        Summable (fun T => (τ x T).toReal * oracle_err pol T))
    (hsumm_gap :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo_rl : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    OracleMeasurableParamArgmin
        (ExpectedAdaptiveTreeObjective τ x
          (fun pol T =>
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen))
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar ⊆
      OracleMeasurableParamPointwiseEpsilonArgmin
        lossTrue
        (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
        fstar
        (fun pol =>
          Exp (τ x) (oracle_err pol) +
            Exp (τ x) (fun T =>
              (L_grpo_rl : ℝ) *
                (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))) := by
  exact oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation
    (τ := τ) (x := x)
    (lossTrue := lossTrue)
    (lossTree := fun pol T =>
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
    (isMeasurable := fun pol f =>
      OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
    (fstar := fstar)
    (eps := fun pol =>
      Exp (τ x) (oracle_err pol) +
        Exp (τ x) (fun T =>
          (L_grpo_rl : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))
    (hsumm_abs := by
      intro pol _hMeas
      exact hsumm_true pol)
    (hclose := by
      intro pol _hMeas
      exact Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
        (k := k)
        fstar pol pol_old pol_ref reward eps beta gen g τ L_grpo_rl
        D_max hD_max h_dist_bound
        Loss_max hLoss_max (hLoss_bound pol)
        (h_class_lip pol) h_old_lip h_ref_lip h_reward_lip (h_rum pol) h_gen_fixed
        hbound hbound_global h_mono
        ε_leaf ε_merge ε_idemp
        h_sound h_approx x R hR
        (lossTrue pol) (oracle_err pol)
        (h_oracle pol)
        (hsumm_true pol)
        (hsumm_oracle pol)
        (hsumm_gap pol)
        hsumm_budget)

end GRPORL

section HighProbabilityTransfer

open MeasureTheory Set

variable {Ω : Type*} [MeasurableSpace Ω]

/-- If a good event fails with probability at most `δ` and the bad event is
impossible on the good event, then the bad event also has probability at most
`δ`. This is the generic event-lifting step behind the high-probability
optimizer perturbation corollaries. -/
theorem failure_event_le_of_good_event_implication
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good bad : Set Ω)
    (δ : ENNReal)
    (h_imp : ∀ ω, ω ∈ good → ω ∉ bad)
    (h_good : μ goodᶜ ≤ δ) :
    μ bad ≤ δ := by
  have h_subset : bad ⊆ goodᶜ := by
    intro ω hbad hωgood
    exact h_imp ω hωgood hbad
  exact le_trans (measure_mono h_subset) h_good

/-- High-probability lift of oracle-measurable argmin transfer: if an optimizer
selection rule lands in the surrogate argmin set on a good event and the good
event itself fails with probability at most `δ`, then the failure probability of
the transported near-optimality statement is also at most `δ`. -/
theorem oracleMeasurableParamArgmin_failure_prob_le_of_good_event_transfer
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (choice : Ω → Θ)
    (lossTrue lossSur : Θ → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (ε : ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        choice ω ∈ OracleMeasurableParamArgmin lossSur isMeasurable fstar)
    (h_transfer :
      OracleMeasurableParamArgmin lossSur isMeasurable fstar ⊆
        OracleMeasurableParamEpsilonArgmin lossTrue isMeasurable fstar ε) :
    μ {ω | choice ω ∉
        OracleMeasurableParamEpsilonArgmin lossTrue isMeasurable fstar ε} ≤ δ := by
  exact failure_event_le_of_good_event_implication
    (μ := μ)
    (good := good)
    (bad := {ω | choice ω ∉
      OracleMeasurableParamEpsilonArgmin lossTrue isMeasurable fstar ε})
    (δ := δ)
    (h_imp := by
      intro ω hωgood hbad
      exact hbad (h_transfer (h_argmin ω hωgood)))
    h_good

/-- Pointwise-slack variant of
`oracleMeasurableParamArgmin_failure_prob_le_of_good_event_transfer`. -/
theorem oracleMeasurableParamArgmin_failure_prob_le_of_good_event_pointwiseTransfer
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (choice : Ω → Θ)
    (lossTrue lossSur : Θ → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (eps : Θ → ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        choice ω ∈ OracleMeasurableParamArgmin lossSur isMeasurable fstar)
    (h_transfer :
      OracleMeasurableParamArgmin lossSur isMeasurable fstar ⊆
        OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar eps) :
    μ {ω | choice ω ∉
        OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar eps} ≤ δ := by
  exact failure_event_le_of_good_event_implication
    (μ := μ)
    (good := good)
    (bad := {ω | choice ω ∉
      OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar eps})
    (δ := δ)
    (h_imp := by
      intro ω hωgood hbad
      exact hbad (h_transfer (h_argmin ω hωgood)))
    h_good

/-- High-probability expected-tree transfer: if a choice rule lands in the
argmin set of an expected tree objective on a good event, and that expected
tree objective is pointwise close to the true objective, then failure of the
transported pointwise near-optimality statement has probability no larger than
the failure probability of the good event. -/
theorem oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (choice : Ω → Θ)
    (τ : StochasticAdaptiveTreeMap Strings) (x : Strings)
    (lossTrue : Θ → ℝ)
    (lossTree : Θ → BinTree Strings → ℝ)
    (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (eps : Θ → ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        choice ω ∈ OracleMeasurableParamArgmin
          (ExpectedAdaptiveTreeObjective τ x lossTree) isMeasurable fstar)
    (hsumm_abs :
      ∀ θ, isMeasurable θ fstar →
        Summable (fun T => (τ x T).toReal * |lossTrue θ - lossTree θ T|))
    (hclose :
      ∀ θ, isMeasurable θ fstar →
        Exp (τ x) (fun T => |lossTrue θ - lossTree θ T|) ≤ eps θ) :
    μ {ω | choice ω ∉
        OracleMeasurableParamPointwiseEpsilonArgmin lossTrue isMeasurable fstar eps} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_pointwiseTransfer
    (μ := μ)
    (good := good)
    (choice := choice)
    (lossTrue := lossTrue)
    (lossSur := ExpectedAdaptiveTreeObjective τ x lossTree)
    (isMeasurable := isMeasurable)
    (fstar := fstar)
    (eps := eps)
    (δ := δ)
    h_good
    h_argmin
    (oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation
      (τ := τ) (x := x)
      (lossTrue := lossTrue)
      (lossTree := lossTree)
      (isMeasurable := isMeasurable)
      (fstar := fstar)
      (eps := eps)
      hsumm_abs hclose)

section DPO

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {A : Type*}

/-- High-probability exact-DPO corollary: if a selection rule returns an
oracle-measurable DPO argmin on a good event and that event fails with
probability at most `δ`, then the selected policy fails to be `2ε`-optimal for
the true objective with probability at most `δ`. -/
theorem dpo_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (polSel : Ω → Policy Strings A)
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        polSel ω ∈ OracleMeasurablePolicyArgmin
          (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen) fstar)
    (hclose :
      ∀ pol, DPO.OracleMeasurable pol fstar →
        |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ ε) :
    μ {ω | polSel ω ∉ OracleMeasurablePolicyEpsilonArgmin lossTrue fstar (2 * ε)} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_transfer
    (μ := μ)
    (good := good)
    (choice := polSel)
    (lossTrue := lossTrue)
    (lossSur := fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
    (isMeasurable := DPO.OracleMeasurable)
    (fstar := fstar)
    (ε := 2 * ε)
    (δ := δ)
    h_good
    h_argmin
    (dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
      (fstar := fstar) (lossTrue := lossTrue) (pol_ref := pol_ref)
      (gen := gen) (g := g) (x := x) (R := R) (T := T)
      (β := β) (ε := ε)
      hp hExact hR h_meas_ref h_gen hclose)

/-- High-probability tree-level DPO optimizer transfer under stochastic
adaptive approximate local laws and tree-indexed oracle measurement. -/
theorem dpo_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (polSel : Ω → Policy Strings A)
    (fstar : Strings → Y)
    (lossTrue : Policy Strings A → ℝ)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_class : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy Strings A, ∀ x' (p : A × A),
        |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_class_lip :
      ∀ pol : Policy Strings A, DPO.OracleMeasurable pol fstar →
        PolicyLipschitz pol pol_ref fstar L_class)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx :
      StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (oracle_err : Policy Strings A → BinTree Strings → ℝ)
    (h_oracle :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        ∀ T,
          |lossTrue pol - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ oracle_err pol T)
    (hsumm_true :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        Summable (fun T =>
          (τ x T).toReal *
            |lossTrue pol - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_oracle :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        Summable (fun T => (τ x T).toReal * oracle_err pol T))
    (hsumm_gap :
      ∀ pol : Policy Strings A, ∀ hMeas : DPO.OracleMeasurable pol fstar,
        Summable (fun T =>
          (τ x T).toReal *
            |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
              ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_class : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))))
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        polSel ω ∈ OracleMeasurablePolicyArgmin
          (ExpectedAdaptiveTreeObjective τ x
            (fun pol T => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)) fstar) :
    μ {ω | polSel ω ∉
        OracleMeasurablePolicyPointwiseEpsilonArgmin lossTrue fstar
          (fun pol =>
            Exp (τ x) (oracle_err pol) +
              Exp (τ x) (fun T =>
                2 * |β| * (L_class : ℝ) *
                  (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer
    (μ := μ)
    (good := good)
    (choice := polSel)
    (τ := τ) (x := x)
    (lossTrue := lossTrue)
    (lossTree := fun pol T => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
    (isMeasurable := DPO.OracleMeasurable)
    (fstar := fstar)
    (eps := fun pol =>
      Exp (τ x) (oracle_err pol) +
        Exp (τ x) (fun T =>
          2 * |β| * (L_class : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))
    (δ := δ)
    h_good
    h_argmin
    hsumm_true
    (by
      intro pol hMeas
      exact Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
        fstar pol pol_ref gen g τ β L_class
        D_max hD_max h_dist_bound
        Loss_max hLoss_max (hLoss_bound pol)
        hMeas h_meas_ref (h_class_lip pol hMeas) h_gen_fixed
        hbound hbound_global h_mono
        ε_leaf ε_merge ε_idemp
        h_sound h_approx x R hR
        (lossTrue pol) (oracle_err pol)
        (h_oracle pol hMeas)
        (hsumm_true pol hMeas)
        (hsumm_oracle pol hMeas)
        (hsumm_gap pol hMeas)
        hsumm_budget)

end DPO

section GRPOPL

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {A : Type*}
variable {k : ℕ}

/-- High-probability exact-GRPO-PL corollary. -/
theorem grpo_pl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (polSel : Ω → Policy' Strings A)
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        polSel ω ∈ OracleMeasurableParamArgmin
          (fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
          (fun pol f => GRPOOracleMeasurable pol f) fstar)
    (hclose :
      ∀ pol, GRPOOracleMeasurable pol fstar →
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ ε) :
    μ {ω | polSel ω ∉
        OracleMeasurableParamEpsilonArgmin
          lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar (2 * ε)} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_transfer
    (μ := μ)
    (good := good)
    (choice := polSel)
    (lossTrue := lossTrue)
    (lossSur := fun pol => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
    (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
    (fstar := fstar)
    (ε := 2 * ε)
    (δ := δ)
    h_good
    h_argmin
    (grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
      (fstar := fstar) (lossTrue := lossTrue) (ranker := ranker) (gen := gen)
      (g := g) (x := x) (R := R) (T := T)
      (ε := ε) hp hExact hR h_ranker h_gen hclose)

/-- High-probability tree-level GRPO-PL optimizer transfer under stochastic
adaptive approximate local laws and tree-indexed oracle measurement. -/
theorem grpo_pl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (polSel : Ω → Policy' Strings A)
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_class_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo
          (h_class_lip pol) h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx :
      StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (oracle_err : Policy' Strings A → BinTree Strings → ℝ)
    (h_oracle :
      ∀ pol : Policy' Strings A, ∀ T,
        |lossTrue pol - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ oracle_err pol T)
    (hsumm_true :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |lossTrue pol - ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_oracle :
      ∀ pol : Policy' Strings A,
        Summable (fun T => (τ x T).toReal * oracle_err pol T))
    (hsumm_gap :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
              ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))))
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        polSel ω ∈ OracleMeasurableParamArgmin
          (ExpectedAdaptiveTreeObjective τ x
            (fun pol T => ExpectedGRPOLoss pol ranker (ZR g x R T) gen))
          (fun pol f => GRPOOracleMeasurable pol f) fstar) :
    μ {ω | polSel ω ∉
        OracleMeasurableParamPointwiseEpsilonArgmin
          lossTrue (fun pol f => GRPOOracleMeasurable pol f) fstar
          (fun pol =>
            Exp (τ x) (oracle_err pol) +
              Exp (τ x) (fun T =>
                (L_grpo : ℝ) *
                  (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer
    (μ := μ)
    (good := good)
    (choice := polSel)
    (τ := τ) (x := x)
    (lossTrue := lossTrue)
    (lossTree := fun pol T => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
    (isMeasurable := fun pol f => GRPOOracleMeasurable pol f)
    (fstar := fstar)
    (eps := fun pol =>
      Exp (τ x) (oracle_err pol) +
        Exp (τ x) (fun T =>
          (L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))
    (δ := δ)
    h_good
    h_argmin
    (by
      intro pol _hMeas
      exact hsumm_true pol)
    (by
      intro pol _hMeas
      exact Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
        (k := k)
        fstar pol ranker gen g τ L_grpo
        D_max hD_max h_dist_bound
        Loss_max hLoss_max (hLoss_bound pol)
        (h_class_lip pol) h_ranker (h_rum pol) h_gen_fixed
        hbound hbound_global h_mono
        ε_leaf ε_merge ε_idemp
        h_sound h_approx x R hR
        (lossTrue pol) (oracle_err pol)
        (h_oracle pol)
        (hsumm_true pol)
        (hsumm_oracle pol)
        (hsumm_gap pol)
        hsumm_budget)

end GRPOPL

section GRPORL

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]
variable {A : Type*}
variable {k : ℕ}

/-- High-probability exact-GRPO-RL corollary. -/
theorem grpo_rl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (polSel : Ω → Policy' Strings A)
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (ε : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        polSel ω ∈ OracleMeasurableParamArgmin
          (fun pol =>
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
          (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
          fstar)
    (hclose :
      ∀ pol,
        OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar →
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤ ε) :
    μ {ω | polSel ω ∉
        OracleMeasurableParamEpsilonArgmin
          lossTrue
          (fun pol f => OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
          fstar (2 * ε)} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_transfer
    (μ := μ)
    (good := good)
    (choice := polSel)
    (lossTrue := lossTrue)
    (lossSur := fun pol =>
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
    (isMeasurable := fun pol f =>
      OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
    (fstar := fstar)
    (ε := 2 * ε)
    (δ := δ)
    h_good
    h_argmin
    (grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement
      (fstar := fstar) (lossTrue := lossTrue) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta)
      (gen := gen) (g := g) (x := x) (R := R) (T := T)
      (ε := ε) hp hExact hR h_gen hclose)

/-- High-probability tree-level GRPO-RL optimizer transfer under stochastic
adaptive approximate local laws and tree-indexed oracle measurement. -/
theorem grpo_rl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (polSel : Ω → Policy' Strings A)
    (fstar : Strings → Y)
    (lossTrue : Policy' Strings A → ℝ)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound :
      ∀ pol : Policy' Strings A, ∀ x' (group : Fin k → A),
        |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_class_lip : ∀ pol : Policy' Strings A, GRPOPolicyLipschitz pol fstar L_grpo)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo)
    (h_rum :
      ∀ pol : Policy' Strings A, ∀ x' z',
        ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x')
          L_grpo (h_class_lip pol) h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx :
      StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (oracle_err : Policy' Strings A → BinTree Strings → ℝ)
    (h_oracle :
      ∀ pol : Policy' Strings A, ∀ T,
          |lossTrue pol -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤
            oracle_err pol T)
    (hsumm_true :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |lossTrue pol -
                ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_oracle :
      ∀ pol : Policy' Strings A,
        Summable (fun T => (τ x T).toReal * oracle_err pol T))
    (hsumm_gap :
      ∀ pol : Policy' Strings A,
        Summable (fun T =>
          (τ x T).toReal *
            |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))))
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_argmin :
      ∀ ω, ω ∈ good →
        polSel ω ∈ OracleMeasurableParamArgmin
          (ExpectedAdaptiveTreeObjective τ x
            (fun pol T =>
              ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen))
          (fun pol f =>
            OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
          fstar) :
    μ {ω | polSel ω ∉
        OracleMeasurableParamPointwiseEpsilonArgmin
          lossTrue
          (fun pol f =>
            OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
          fstar
          (fun pol =>
            Exp (τ x) (oracle_err pol) +
              Exp (τ x) (fun T =>
                (L_grpo : ℝ) *
                  (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))} ≤ δ := by
  exact oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer
    (μ := μ)
    (good := good)
    (choice := polSel)
    (τ := τ) (x := x)
    (lossTrue := lossTrue)
    (lossTree := fun pol T =>
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
    (isMeasurable := fun pol f =>
      OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta f)
    (fstar := fstar)
    (eps := fun pol =>
      Exp (τ x) (oracle_err pol) +
        Exp (τ x) (fun T =>
          (L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))
    (δ := δ)
    h_good
    h_argmin
    (by
      intro pol _hMeas
      exact hsumm_true pol)
    (by
      intro pol _hMeas
      exact Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
        (k := k)
        fstar pol pol_old pol_ref reward eps beta gen g τ L_grpo
        D_max hD_max h_dist_bound
        Loss_max hLoss_max (hLoss_bound pol)
        (h_class_lip pol) h_old_lip h_ref_lip h_reward_lip (h_rum pol) h_gen_fixed
        hbound hbound_global h_mono
        ε_leaf ε_merge ε_idemp
        h_sound h_approx x R hR
        (lossTrue pol) (oracle_err pol)
        (h_oracle pol)
        (hsumm_true pol)
        (hsumm_oracle pol)
        (hsumm_gap pol)
        hsumm_budget)

end GRPORL

end HighProbabilityTransfer

section RegularizedSelection

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- `ε`-optimal version of `IsCertifiedRegularizedMinimizer` for a true
objective on summarizer/law pairs. -/
def IsCertifiedRegularizedEpsilonMinimizer
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (ε : ℝ) : Prop :=
  ∀ g : Summarizer Strings,
    ∀ laws : ApproxLocalLawsBundle g T fstar,
      trueObjective gStar lawsStar ≤ trueObjective g laws + ε

/-- Pointwise-slack `ε`-optimal version of `IsCertifiedRegularizedMinimizer`. -/
def IsCertifiedRegularizedPointwiseEpsilonMinimizer
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (eps : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ) : Prop :=
  ∀ g : Summarizer Strings,
    ∀ laws : ApproxLocalLawsBundle g T fstar,
      trueObjective gStar lawsStar ≤ trueObjective g laws + eps gStar lawsStar + eps g laws

/-- Constrained `ε`-optimal version of
`IsConstrainedCertifiedRegularizedMinimizer`. -/
def IsConstrainedCertifiedRegularizedEpsilonMinimizer
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (constraints : RegularizedObjectiveConstraints)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (ε : ℝ) : Prop :=
  SatisfiesRegularizedConstraints gStar x R T fstar cost lawsStar constraints
    ∧ ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          SatisfiesRegularizedConstraints g x R T fstar cost laws constraints →
            trueObjective gStar lawsStar ≤ trueObjective g laws + ε

/-- Pointwise-slack constrained `ε`-optimal version of
`IsConstrainedCertifiedRegularizedMinimizer`. -/
def IsConstrainedCertifiedRegularizedPointwiseEpsilonMinimizer
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (constraints : RegularizedObjectiveConstraints)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (eps : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ) : Prop :=
  SatisfiesRegularizedConstraints gStar x R T fstar cost lawsStar constraints
    ∧ ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          SatisfiesRegularizedConstraints g x R T fstar cost laws constraints →
            trueObjective gStar lawsStar ≤ trueObjective g laws + eps gStar lawsStar + eps g laws

/-- Uniformly perturbing the certified regularized objective by at most `ε`
turns an exact certified minimizer into a `2ε`-minimizer for the true
objective. -/
theorem certifiedRegularized_epsilonMinimizer_of_uniform_perturbation
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (ε : ℝ)
    (hMin : IsCertifiedRegularizedMinimizer x R T fstar cost weights gStar lawsStar)
    (hclose :
      ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          |trueObjective g laws -
              certifiedRegularizedObjective g x R T fstar cost weights laws| ≤ ε) :
    IsCertifiedRegularizedEpsilonMinimizer
      x R T fstar trueObjective gStar lawsStar (2 * ε) := by
  intro g laws
  have hStar := hclose gStar lawsStar
  have hOther := hclose g laws
  have hMin' := hMin g laws
  linarith [abs_le.mp hStar, abs_le.mp hOther, hMin']

/-- Constrained version of
`certifiedRegularized_epsilonMinimizer_of_uniform_perturbation`. -/
theorem constrainedCertifiedRegularized_epsilonMinimizer_of_uniform_perturbation
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (constraints : RegularizedObjectiveConstraints)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (ε : ℝ)
    (hMin :
      IsConstrainedCertifiedRegularizedMinimizer
        x R T fstar cost weights constraints gStar lawsStar)
    (hclose :
      ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          SatisfiesRegularizedConstraints g x R T fstar cost laws constraints →
            |trueObjective g laws -
                certifiedRegularizedObjective g x R T fstar cost weights laws| ≤ ε) :
    IsConstrainedCertifiedRegularizedEpsilonMinimizer
      x R T fstar cost constraints trueObjective gStar lawsStar (2 * ε) := by
  constructor
  · exact hMin.1
  · intro g laws hFeas
    have hStar := hclose gStar lawsStar hMin.1
    have hOther := hclose g laws hFeas
    have hMin' := hMin.2 g laws hFeas
    linarith [abs_le.mp hStar, abs_le.mp hOther, hMin']

/-- Non-uniform perturbation turns an exact certified-regularized minimizer
into a pointwise-slack minimizer for the true objective. -/
theorem certifiedRegularized_pointwiseEpsilonMinimizer_of_nonuniform_perturbation
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (eps :
      ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (hMin : IsCertifiedRegularizedMinimizer x R T fstar cost weights gStar lawsStar)
    (hclose :
      ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          |trueObjective g laws -
              certifiedRegularizedObjective g x R T fstar cost weights laws| ≤ eps g laws) :
    IsCertifiedRegularizedPointwiseEpsilonMinimizer
      x R T fstar trueObjective gStar lawsStar eps := by
  intro g laws
  have hStar := hclose gStar lawsStar
  have hOther := hclose g laws
  have hMin' := hMin g laws
  linarith [abs_le.mp hStar, abs_le.mp hOther, hMin']

/-- Non-uniform constrained perturbation turns an exact constrained
certified-regularized minimizer into a pointwise-slack constrained minimizer
for the true objective. -/
theorem constrainedCertifiedRegularized_pointwiseEpsilonMinimizer_of_nonuniform_perturbation
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (constraints : RegularizedObjectiveConstraints)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gStar : Summarizer Strings)
    (lawsStar : ApproxLocalLawsBundle gStar T fstar)
    (eps :
      ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (hMin :
      IsConstrainedCertifiedRegularizedMinimizer
        x R T fstar cost weights constraints gStar lawsStar)
    (hclose :
      ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          SatisfiesRegularizedConstraints g x R T fstar cost laws constraints →
            |trueObjective g laws -
                certifiedRegularizedObjective g x R T fstar cost weights laws| ≤ eps g laws) :
    IsConstrainedCertifiedRegularizedPointwiseEpsilonMinimizer
      x R T fstar cost constraints trueObjective gStar lawsStar eps := by
  constructor
  · exact hMin.1
  · intro g laws hFeas
    have hStar := hclose gStar lawsStar hMin.1
    have hOther := hclose g laws hFeas
    have hMin' := hMin.2 g laws hFeas
    linarith [abs_le.mp hStar, abs_le.mp hOther, hMin']

open MeasureTheory Set

/-- High-probability wrapper for certified-regularized selection under a
uniform objective perturbation envelope. -/
theorem certifiedRegularized_epsilonMinimizer_failure_prob_le_of_good_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (gSel : Ω → Summarizer Strings)
    (lawsSel : ∀ ω, ApproxLocalLawsBundle (gSel ω) T fstar)
    (ε : ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_min :
      ∀ ω, ω ∈ good →
        IsCertifiedRegularizedMinimizer x R T fstar cost weights (gSel ω) (lawsSel ω))
    (hclose :
      ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          |trueObjective g laws -
              certifiedRegularizedObjective g x R T fstar cost weights laws| ≤ ε) :
    μ {ω | ¬ IsCertifiedRegularizedEpsilonMinimizer
        x R T fstar trueObjective (gSel ω) (lawsSel ω) (2 * ε)} ≤ δ := by
  exact failure_event_le_of_good_event_implication
    (μ := μ)
    (good := good)
    (bad := {ω | ¬ IsCertifiedRegularizedEpsilonMinimizer
      x R T fstar trueObjective (gSel ω) (lawsSel ω) (2 * ε)})
    (δ := δ)
    (h_imp := by
      intro ω hωgood hbad
      exact hbad
        (certifiedRegularized_epsilonMinimizer_of_uniform_perturbation
          (x := x) (R := R) (T := T) (fstar := fstar)
          (trueObjective := trueObjective)
          (cost := cost) (weights := weights)
          (gStar := gSel ω) (lawsStar := lawsSel ω)
          (ε := ε)
          (hMin := h_min ω hωgood)
          (hclose := hclose)))
    h_good

/-- High-probability constrained wrapper for certified-regularized selection
under a uniform objective perturbation envelope. -/
theorem constrainedCertifiedRegularized_epsilonMinimizer_failure_prob_le_of_good_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (cost : SummaryCost Strings)
    (weights : RegularizedObjectiveWeights)
    (constraints : RegularizedObjectiveConstraints)
    (trueObjective : ∀ g : Summarizer Strings, ApproxLocalLawsBundle g T fstar → ℝ)
    (gSel : Ω → Summarizer Strings)
    (lawsSel : ∀ ω, ApproxLocalLawsBundle (gSel ω) T fstar)
    (ε : ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_min :
      ∀ ω, ω ∈ good →
        IsConstrainedCertifiedRegularizedMinimizer
          x R T fstar cost weights constraints (gSel ω) (lawsSel ω))
    (hclose :
      ∀ g : Summarizer Strings,
        ∀ laws : ApproxLocalLawsBundle g T fstar,
          SatisfiesRegularizedConstraints g x R T fstar cost laws constraints →
            |trueObjective g laws -
                certifiedRegularizedObjective g x R T fstar cost weights laws| ≤ ε) :
    μ {ω | ¬ IsConstrainedCertifiedRegularizedEpsilonMinimizer
        x R T fstar cost constraints trueObjective (gSel ω) (lawsSel ω) (2 * ε)} ≤ δ := by
  exact failure_event_le_of_good_event_implication
    (μ := μ)
    (good := good)
    (bad := {ω | ¬ IsConstrainedCertifiedRegularizedEpsilonMinimizer
      x R T fstar cost constraints trueObjective (gSel ω) (lawsSel ω) (2 * ε)})
    (δ := δ)
    (h_imp := by
      intro ω hωgood hbad
      exact hbad
        (constrainedCertifiedRegularized_epsilonMinimizer_of_uniform_perturbation
          (x := x) (R := R) (T := T) (fstar := fstar)
          (cost := cost) (weights := weights) (constraints := constraints)
          (trueObjective := trueObjective)
          (gStar := gSel ω) (lawsStar := lawsSel ω)
          (ε := ε)
          (hMin := h_min ω hωgood)
          (hclose := hclose)))
    h_good

end RegularizedSelection

end FormalProofs.OPT
