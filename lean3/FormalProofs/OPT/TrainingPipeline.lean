import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.PreferenceLearning

/-!
# FormalProofs/TrainingPipeline.lean

## Multi-Stage Training Gap Composition

This file composes training gaps across multi-stage pipelines via triangle inequality.

### Abstract Framework

The file provides **abstract gap composition** that works for ANY preference learning method:
- `training_path_gap_bound_abstract`: Generic gap bound using abstract loss functional
- Works for DPO, GRPO-PL, GRPO-RL, and future methods

### Concrete Instantiations

- `training_path_gap_bound`: DPO-specific version (original)
- `TrainingPathBundle`: DPO-specific bundle for convenience

## Training Pipeline Structure

```
Stage 1: Oracle f* → Large Policy pol_L (via DPO/GRPO preference learning)
Stage 2: pol_L → Small Policy pol_S (via distillation or further RL)
```

### This File's Contribution

Given that each stage has bounded gap (from the deep theorems):
- ε₁ = |L(pol_L; original) - L(pol_L; summarized)|
- ε₂ = |L(pol_S; summarized) - L(pol_L; summarized)|

We show gaps compose additively:
  |L(pol_S; original) - L(pol_L; original)| ≤ 2·ε₁ + ε₂

When local laws hold exactly, ε₁ = 0, giving pure distillation gap.

### Main Results

1. **training_path_gap_bound_abstract**: Abstract gap composition for any preference method
2. **distillation_exact**: Zero gap when student exactly matches teacher
3. **distillation_gap**: Bounded gap with Lipschitz conditions
4. **training_path_gap_bound**: DPO-specific gap bound (instantiation of abstract)
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

section TrainingPipeline

open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

variable {A : Type*}
variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Abstract Gap Composition

This section provides method-agnostic gap composition that works for ANY preference
learning method. The key insight is that the triangle inequality argument doesn't
depend on the specific form of the loss - only on having a loss functional.
-/

/-- **Abstract Training Path Gap Bound**

Given any loss functional `ExpLoss : Θ → PMF Strings → ℝ`, if we have:
- Stage 1 gap: |L(θ_L, μ_X) - L(θ_L, μ_Z)| ≤ ε₁
- Stage 2 gap: |L(θ_S, μ_Z) - L(θ_L, μ_Z)| ≤ ε₂
- Student equivalence: |L(θ_S, μ_X) - L(θ_S, μ_Z)| ≤ ε₁

Then by triangle inequality:
  |L(θ_S, μ_X) - L(θ_L, μ_X)| ≤ 2·ε₁ + ε₂

This works for DPO, GRPO, or any future preference learning method. -/
theorem training_path_gap_bound_abstract {Θ : Type*}
    (ExpLoss : Θ → PMF Strings → ℝ)
    (θ_L θ_S : Θ)
    (μ_X μ_Z : PMF Strings)
    (eps_stage1 eps_stage2 : ℝ)
    (h_stage1 : |ExpLoss θ_L μ_X - ExpLoss θ_L μ_Z| ≤ eps_stage1)
    (h_stage2 : |ExpLoss θ_S μ_Z - ExpLoss θ_L μ_Z| ≤ eps_stage2)
    (h_S_equiv : |ExpLoss θ_S μ_X - ExpLoss θ_S μ_Z| ≤ eps_stage1) :
    |ExpLoss θ_S μ_X - ExpLoss θ_L μ_X| ≤ 2 * eps_stage1 + eps_stage2 := by
  let L_X_S := ExpLoss θ_S μ_X
  let L_Z_S := ExpLoss θ_S μ_Z
  let L_X_L := ExpLoss θ_L μ_X
  let L_Z_L := ExpLoss θ_L μ_Z
  have h1 : |L_X_S - L_Z_S| ≤ eps_stage1 := h_S_equiv
  have h2 : |L_Z_S - L_Z_L| ≤ eps_stage2 := h_stage2
  have h3 : |L_Z_L - L_X_L| ≤ eps_stage1 := by rw [abs_sub_comm]; exact h_stage1
  have triangle1 : |L_X_S - L_Z_S + (L_Z_S - L_Z_L)| ≤ |L_X_S - L_Z_S| + |L_Z_S - L_Z_L| :=
    abs_add_le _ _
  have triangle2 : |(L_X_S - L_Z_S + (L_Z_S - L_Z_L)) + (L_Z_L - L_X_L)| ≤
      |L_X_S - L_Z_S + (L_Z_S - L_Z_L)| + |L_Z_L - L_X_L| := abs_add_le _ _
  calc |L_X_S - L_X_L|
       = |(L_X_S - L_Z_S + (L_Z_S - L_Z_L)) + (L_Z_L - L_X_L)| := by ring_nf
     _ ≤ |L_X_S - L_Z_S + (L_Z_S - L_Z_L)| + |L_Z_L - L_X_L| := triangle2
     _ ≤ |L_X_S - L_Z_S| + |L_Z_S - L_Z_L| + |L_Z_L - L_X_L| := by linarith [triangle1]
     _ ≤ eps_stage1 + eps_stage2 + eps_stage1 := by linarith
     _ = 2 * eps_stage1 + eps_stage2 := by ring

/-- Abstract training-path gap bound with optional oracle measurement on the
final objective. The exact-oracle regime is recovered by setting both oracle
error terms to `0`. -/
theorem training_path_gap_bound_abstract_with_oracleMeasurement {Θ : Type*}
    (TrueLoss OracleLoss : Θ → PMF Strings → ℝ)
    (θ_L θ_S : Θ)
    (μ_X μ_Z : PMF Strings)
    (eps_stage1 eps_stage2 oracle_err_L oracle_err_S : ℝ)
    (h_oracle_L : |TrueLoss θ_L μ_X - OracleLoss θ_L μ_X| ≤ oracle_err_L)
    (h_oracle_S : |TrueLoss θ_S μ_X - OracleLoss θ_S μ_X| ≤ oracle_err_S)
    (h_stage1 : |OracleLoss θ_L μ_X - OracleLoss θ_L μ_Z| ≤ eps_stage1)
    (h_stage2 : |OracleLoss θ_S μ_Z - OracleLoss θ_L μ_Z| ≤ eps_stage2)
    (h_S_equiv : |OracleLoss θ_S μ_X - OracleLoss θ_S μ_Z| ≤ eps_stage1) :
    |TrueLoss θ_S μ_X - TrueLoss θ_L μ_X| ≤
      oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by
  have h_core :
      |OracleLoss θ_S μ_X - OracleLoss θ_L μ_X| ≤ 2 * eps_stage1 + eps_stage2 :=
    training_path_gap_bound_abstract OracleLoss θ_L θ_S μ_X μ_Z eps_stage1 eps_stage2
      h_stage1 h_stage2 h_S_equiv
  let L_true_S := TrueLoss θ_S μ_X
  let L_oracle_S := OracleLoss θ_S μ_X
  let L_true_L := TrueLoss θ_L μ_X
  let L_oracle_L := OracleLoss θ_L μ_X
  have hL : |L_oracle_L - L_true_L| ≤ oracle_err_L := by
    rw [abs_sub_comm]
    exact h_oracle_L
  have h_triangle1 :
      |L_true_S - L_true_L| ≤
        |L_true_S - L_oracle_S| + |L_oracle_S - L_true_L| := by
    have hdecomp : L_true_S - L_true_L = (L_true_S - L_oracle_S) + (L_oracle_S - L_true_L) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  have h_triangle2 :
      |L_oracle_S - L_true_L| ≤
        |L_oracle_S - L_oracle_L| + |L_oracle_L - L_true_L| := by
    have hdecomp : L_oracle_S - L_true_L = (L_oracle_S - L_oracle_L) + (L_oracle_L - L_true_L) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  calc
    |TrueLoss θ_S μ_X - TrueLoss θ_L μ_X|
        = |L_true_S - L_true_L| := by rfl
    _ ≤ |L_true_S - L_oracle_S| + |L_oracle_S - L_true_L| := h_triangle1
    _ ≤ |L_true_S - L_oracle_S| + (|L_oracle_S - L_oracle_L| + |L_oracle_L - L_true_L|) := by
          linarith [h_triangle2]
    _ ≤ oracle_err_S + ((2 * eps_stage1 + eps_stage2) + oracle_err_L) := by
          linarith [h_oracle_S, h_core, hL]
    _ = oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by ring

/-- GRPO gap composition: uses the abstract theorem with GRPO loss functional.

This shows GRPO training gaps compose additively, just like DPO. -/
theorem grpo_training_path_gap_bound {k : ℕ}
    (pol_L pol_S : Policy' Strings A)
    (ranker_L ranker_S : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (eps_stage1 eps_stage2 : ℝ)
    (h_stage1 : |ExpectedGRPOLoss pol_L ranker_L μ_X gen -
                 ExpectedGRPOLoss pol_L ranker_L μ_Z gen| ≤ eps_stage1)
    (h_stage2 : |ExpectedGRPOLoss pol_S ranker_S μ_Z gen -
                 ExpectedGRPOLoss pol_L ranker_L μ_Z gen| ≤ eps_stage2)
    (h_S_equiv : |ExpectedGRPOLoss pol_S ranker_S μ_X gen -
                  ExpectedGRPOLoss pol_S ranker_S μ_Z gen| ≤ eps_stage1) :
    |ExpectedGRPOLoss pol_S ranker_S μ_X gen -
     ExpectedGRPOLoss pol_L ranker_L μ_X gen| ≤ 2 * eps_stage1 + eps_stage2 := by
  -- Use abstract theorem with Θ = (Policy' Strings A) × (Strings → GroupRanker A k)
  let ExpLoss := fun θ : Policy' Strings A × (Strings → GroupRanker A k) =>
    fun μ => ExpectedGRPOLoss θ.1 θ.2 μ gen
  exact training_path_gap_bound_abstract ExpLoss (pol_L, ranker_L) (pol_S, ranker_S)
    μ_X μ_Z eps_stage1 eps_stage2 h_stage1 h_stage2 h_S_equiv

/-- GRPO training-path gap bound with separate oracle-measurement terms for the
teacher and student objectives on the original distribution. -/
theorem grpo_training_path_gap_bound_with_oracleMeasurement {k : ℕ}
    (pol_L pol_S : Policy' Strings A)
    (ranker_L ranker_S : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (loss_true_L loss_true_S : ℝ)
    (eps_stage1 eps_stage2 oracle_err_L oracle_err_S : ℝ)
    (h_oracle_L : |loss_true_L - ExpectedGRPOLoss pol_L ranker_L μ_X gen| ≤ oracle_err_L)
    (h_oracle_S : |loss_true_S - ExpectedGRPOLoss pol_S ranker_S μ_X gen| ≤ oracle_err_S)
    (h_stage1 : |ExpectedGRPOLoss pol_L ranker_L μ_X gen -
                 ExpectedGRPOLoss pol_L ranker_L μ_Z gen| ≤ eps_stage1)
    (h_stage2 : |ExpectedGRPOLoss pol_S ranker_S μ_Z gen -
                 ExpectedGRPOLoss pol_L ranker_L μ_Z gen| ≤ eps_stage2)
    (h_S_equiv : |ExpectedGRPOLoss pol_S ranker_S μ_X gen -
                  ExpectedGRPOLoss pol_S ranker_S μ_Z gen| ≤ eps_stage1) :
    |loss_true_S - loss_true_L| ≤
      oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by
  have h_core :
      |ExpectedGRPOLoss pol_S ranker_S μ_X gen -
        ExpectedGRPOLoss pol_L ranker_L μ_X gen| ≤
        2 * eps_stage1 + eps_stage2 :=
    grpo_training_path_gap_bound pol_L pol_S ranker_L ranker_S gen μ_X μ_Z
      eps_stage1 eps_stage2 h_stage1 h_stage2 h_S_equiv
  have hL :
      |ExpectedGRPOLoss pol_L ranker_L μ_X gen - loss_true_L| ≤ oracle_err_L := by
    rw [abs_sub_comm]
    exact h_oracle_L
  have h_triangle1 :
      |loss_true_S - loss_true_L| ≤
        |loss_true_S - ExpectedGRPOLoss pol_S ranker_S μ_X gen| +
          |ExpectedGRPOLoss pol_S ranker_S μ_X gen - loss_true_L| := by
    have hdecomp :
        loss_true_S - loss_true_L =
          (loss_true_S - ExpectedGRPOLoss pol_S ranker_S μ_X gen) +
            (ExpectedGRPOLoss pol_S ranker_S μ_X gen - loss_true_L) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  have h_triangle2 :
      |ExpectedGRPOLoss pol_S ranker_S μ_X gen - loss_true_L| ≤
        |ExpectedGRPOLoss pol_S ranker_S μ_X gen -
          ExpectedGRPOLoss pol_L ranker_L μ_X gen| +
        |ExpectedGRPOLoss pol_L ranker_L μ_X gen - loss_true_L| := by
    have hdecomp :
        ExpectedGRPOLoss pol_S ranker_S μ_X gen - loss_true_L =
          (ExpectedGRPOLoss pol_S ranker_S μ_X gen -
            ExpectedGRPOLoss pol_L ranker_L μ_X gen) +
          (ExpectedGRPOLoss pol_L ranker_L μ_X gen - loss_true_L) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  calc
    |loss_true_S - loss_true_L|
        ≤ |loss_true_S - ExpectedGRPOLoss pol_S ranker_S μ_X gen| +
            |ExpectedGRPOLoss pol_S ranker_S μ_X gen - loss_true_L| := h_triangle1
    _ ≤ |loss_true_S - ExpectedGRPOLoss pol_S ranker_S μ_X gen| +
          (|ExpectedGRPOLoss pol_S ranker_S μ_X gen -
              ExpectedGRPOLoss pol_L ranker_L μ_X gen| +
            |ExpectedGRPOLoss pol_L ranker_L μ_X gen - loss_true_L|) := by
          linarith [h_triangle2]
    _ ≤ oracle_err_S + ((2 * eps_stage1 + eps_stage2) + oracle_err_L) := by
          linarith [h_oracle_S, h_core, hL]
    _ = oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by ring

/-!
## Distillation Loss
-/

-- Abstract distillation loss: measures divergence between teacher and student policies.
variable (distillLoss : Policy Strings A → Policy Strings A → Strings → ℝ)

/-- Expected distillation loss over a document distribution. -/
noncomputable def ExpectedDistillationLoss
    (distillLoss : Policy Strings A → Policy Strings A → Strings → ℝ)
    (pol_L pol_S : Policy Strings A) (mu : PMF Strings) : ℝ :=
  ∑' x, (mu x).toReal * distillLoss pol_L pol_S x

/-- Oracle-measurable distillation loss. -/
def OracleMeasurableDistillationLoss {Strings A Y : Type*} [PseudoMetricSpace Y]
    (distillLoss : Policy Strings A → Policy Strings A → Strings → ℝ)
    (pol_L pol_S : Policy Strings A) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → distillLoss pol_L pol_S x = distillLoss pol_L pol_S x'

/-!
## Optimality Definitions
-/

/-- A policy is optimal for oracle f* with respect to a preference loss functional. -/
def OptimalForOracle {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol : Policy Strings A) (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ) : Prop :=
  DPO.OracleMeasurable pol fstar ∧
  ∀ pol', DPO.OracleMeasurable pol' fstar → prefLoss pol ≤ prefLoss pol'

/-- A policy is ε-optimal for oracle f*. -/
def EpsilonOptimalForOracle {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol : Policy Strings A) (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ) (eps : ℝ) : Prop :=
  DPO.OracleMeasurable pol fstar ∧
  ∃ pol_opt, OptimalForOracle pol_opt fstar prefLoss ∧ |prefLoss pol - prefLoss pol_opt| ≤ eps

/-!
## Distillation Theorems
-/

/-- **Distillation Exact**: Student exactly matching teacher preserves optimality. -/
theorem distillation_exact {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (pol_L pol_S : Policy Strings A) (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ)
    (h_L_opt : OptimalForOracle pol_L fstar prefLoss)
    (h_exact : ∀ x a, pol_S x a = pol_L x a) :
    OptimalForOracle pol_S fstar prefLoss := by
  constructor
  · intro x x' a hdist
    rw [h_exact, h_exact]
    exact h_L_opt.1 x x' a hdist
  · intro pol' hpol'
    have h_loss_eq : prefLoss pol_S = prefLoss pol_L := by
      congr 1
      funext x a
      exact h_exact x a
    rw [h_loss_eq]
    exact h_L_opt.2 pol' hpol'

/-- Lipschitz condition for preference loss w.r.t. policy changes. -/
def LipschitzPrefLoss
    (distillLoss : Policy Strings A → Policy Strings A → Strings → ℝ)
    (prefLoss : Policy Strings A → ℝ) (mu : PMF Strings) (L : ℝ≥0) : Prop :=
  ∀ pol pol', |prefLoss pol - prefLoss pol'| ≤
    (L : ℝ) * ExpectedDistillationLoss distillLoss pol pol' mu

/-- **Distillation Gap**: Bounded gap when distillation loss is bounded. -/
theorem distillation_gap
    (pol_L pol_S : Policy Strings A) (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ)
    (mu : PMF Strings)
    (L_dist : ℝ≥0)
    (h_L_opt : OptimalForOracle pol_L fstar prefLoss)
    (h_S_meas : DPO.OracleMeasurable pol_S fstar)
    (h_lip : LipschitzPrefLoss distillLoss prefLoss mu L_dist)
    (eps : ℝ) (heps : 0 ≤ eps)
    (h_distill : ExpectedDistillationLoss distillLoss pol_L pol_S mu ≤ eps)
    (h_symm :
      ExpectedDistillationLoss distillLoss pol_S pol_L mu =
      ExpectedDistillationLoss distillLoss pol_L pol_S mu) :
    |prefLoss pol_S - prefLoss pol_L| ≤ (L_dist : ℝ) * eps := by
  have h_distill' : ExpectedDistillationLoss distillLoss pol_S pol_L mu ≤ eps := by
    simpa [h_symm] using h_distill
  calc |prefLoss pol_S - prefLoss pol_L|
       ≤ (L_dist : ℝ) * ExpectedDistillationLoss distillLoss pol_S pol_L mu := h_lip pol_S pol_L
     _ ≤ (L_dist : ℝ) * eps := by
         apply mul_le_mul_of_nonneg_left h_distill' (NNReal.coe_nonneg L_dist)

/-- Corollary: Student is ε-optimal when distillation loss is bounded. -/
theorem distillation_epsilon_optimal
    (pol_L pol_S : Policy Strings A) (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ)
    (mu : PMF Strings)
    (L_dist : ℝ≥0)
    (h_L_opt : OptimalForOracle pol_L fstar prefLoss)
    (h_S_meas : DPO.OracleMeasurable pol_S fstar)
    (h_lip : LipschitzPrefLoss distillLoss prefLoss mu L_dist)
    (eps : ℝ) (heps : 0 ≤ eps)
    (h_distill : ExpectedDistillationLoss distillLoss pol_L pol_S mu ≤ eps)
    (h_symm :
      ExpectedDistillationLoss distillLoss pol_S pol_L mu =
      ExpectedDistillationLoss distillLoss pol_L pol_S mu) :
    EpsilonOptimalForOracle pol_S fstar prefLoss ((L_dist : ℝ) * eps) := by
  constructor
  · exact h_S_meas
  · use pol_L
    constructor
    · exact h_L_opt
    · exact distillation_gap distillLoss pol_L pol_S fstar prefLoss mu L_dist
        h_L_opt h_S_meas h_lip eps heps h_distill h_symm

/-- Any oracle-measurable policy with bounded loss gap to an oracle-optimal
reference policy is `ε`-optimal. -/
theorem epsilonOptimal_of_reference_gap
    (pol_opt pol : Policy Strings A) (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ)
    (eps : ℝ)
    (h_opt : OptimalForOracle pol_opt fstar prefLoss)
    (h_meas : DPO.OracleMeasurable pol fstar)
    (h_gap : |prefLoss pol - prefLoss pol_opt| ≤ eps) :
    EpsilonOptimalForOracle pol fstar prefLoss eps := by
  constructor
  · exact h_meas
  · exact ⟨pol_opt, h_opt, h_gap⟩

/-- High-probability lift of `EpsilonOptimalForOracle`: if a good event fails
with probability at most `δ` and every realization in that good event yields an
`ε(ω)`-optimal policy, then failure of `ε(ω)`-optimality also has probability
at most `δ`. -/
theorem epsilonOptimal_failure_prob_le_of_good_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (pol : Ω → Policy Strings A)
    (fstar : Strings → Y)
    (prefLoss : Policy Strings A → ℝ)
    (eps : Ω → ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_eps :
      ∀ ω, ω ∈ good →
        EpsilonOptimalForOracle (pol ω) fstar prefLoss (eps ω)) :
    μ {ω | ¬ EpsilonOptimalForOracle (pol ω) fstar prefLoss (eps ω)} ≤ δ := by
  have h_subset :
      {ω | ¬ EpsilonOptimalForOracle (pol ω) fstar prefLoss (eps ω)} ⊆ goodᶜ := by
    intro ω hbad hωgood
    exact hbad (h_eps ω hωgood)
  exact le_trans (measure_mono h_subset) h_good

/-- Named training-path wrapper around `epsilonOptimal_failure_prob_le_of_good_event`.
This is the probabilistic form used when the two-stage oracle-measurement
certificate is only known on a confidence event. -/
theorem training_path_epsilon_optimal_failure_prob_le_with_oracleMeasurement
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (good : Set Ω)
    (pol_S : Ω → Policy Strings A)
    (fstar : Strings → Y)
    (trueLoss : Policy Strings A → ℝ)
    (eps : Ω → ℝ)
    (δ : ENNReal)
    (h_good : μ goodᶜ ≤ δ)
    (h_cert :
      ∀ ω, ω ∈ good →
        EpsilonOptimalForOracle (pol_S ω) fstar trueLoss (eps ω)) :
    μ {ω | ¬ EpsilonOptimalForOracle (pol_S ω) fstar trueLoss (eps ω)} ≤ δ := by
  exact epsilonOptimal_failure_prob_le_of_good_event
    (μ := μ) (good := good) (pol := pol_S)
    (fstar := fstar) (prefLoss := trueLoss) (eps := eps) (δ := δ)
    h_good h_cert

/-!
## Training Path Gap Bound
-/

/-- **Training Path Gap Bound**: Gaps compose additively across stages.

    The proof uses the triangle inequality:
    |L_S(orig) - L_L(orig)|
    ≤ |L_S(orig) - L_S(ZR)| + |L_S(ZR) - L_L(ZR)| + |L_L(ZR) - L_L(orig)|
    ≤ eps_S_equiv + eps_stage2 + eps_stage1

    When pol_S is oracle-measurable and local laws hold exactly, eps_S_equiv = 0,
    giving the bound eps_stage1 + eps_stage2. -/
theorem training_path_gap_bound {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol_L pol_S : Policy Strings A)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (beta : ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (eps_stage1 eps_stage2 : ℝ)
    -- Stage 1: pol_L has bounded gap between original and ZR
    (h_stage1 : |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen -
                 ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ eps_stage1)
    -- Stage 2: pol_S differs from pol_L by bounded amount on ZR
    (h_stage2 : |ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen -
                 ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ eps_stage2)
    -- pol_S also satisfies DPO equivalence (oracle-measurable + local laws)
    (h_S_equiv : |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
                  ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen| ≤ eps_stage1) :
    |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
     ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤ 2 * eps_stage1 + eps_stage2 := by
  -- Let L_X_S = ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen
  -- Let L_Z_S = ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen
  -- Let L_X_L = ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen
  -- Let L_Z_L = ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen
  -- We want: |L_X_S - L_X_L| ≤ 2*eps_stage1 + eps_stage2
  -- Triangle: |L_X_S - L_X_L| ≤ |L_X_S - L_Z_S| + |L_Z_S - L_Z_L| + |L_Z_L - L_X_L|
  let L_X_S := ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen
  let L_Z_S := ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen
  let L_X_L := ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen
  let L_Z_L := ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen
  have h1 : |L_X_S - L_Z_S| ≤ eps_stage1 := h_S_equiv
  have h2 : |L_Z_S - L_Z_L| ≤ eps_stage2 := h_stage2
  have h3 : |L_Z_L - L_X_L| ≤ eps_stage1 := by
    rw [abs_sub_comm]
    exact h_stage1
  have triangle1 : |L_X_S - L_Z_S + (L_Z_S - L_Z_L)| ≤ |L_X_S - L_Z_S| + |L_Z_S - L_Z_L| :=
    abs_add_le _ _
  have triangle2 : |(L_X_S - L_Z_S + (L_Z_S - L_Z_L)) + (L_Z_L - L_X_L)| ≤
      |L_X_S - L_Z_S + (L_Z_S - L_Z_L)| + |L_Z_L - L_X_L| := abs_add_le _ _
  calc |L_X_S - L_X_L|
       = |(L_X_S - L_Z_S + (L_Z_S - L_Z_L)) + (L_Z_L - L_X_L)| := by ring_nf
     _ ≤ |L_X_S - L_Z_S + (L_Z_S - L_Z_L)| + |L_Z_L - L_X_L| := triangle2
     _ ≤ |L_X_S - L_Z_S| + |L_Z_S - L_Z_L| + |L_Z_L - L_X_L| := by linarith [triangle1]
     _ ≤ eps_stage1 + eps_stage2 + eps_stage1 := by linarith
     _ = 2 * eps_stage1 + eps_stage2 := by ring

/-- DPO training-path gap bound with separate oracle-measurement terms for the
teacher and student objectives on the original distribution. -/
theorem training_path_gap_bound_with_oracleMeasurement {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol_L pol_S : Policy Strings A)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (beta : ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (loss_true_L loss_true_S : ℝ)
    (eps_stage1 eps_stage2 oracle_err_L oracle_err_S : ℝ)
    (h_oracle_L :
      |loss_true_L - ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤ oracle_err_L)
    (h_oracle_S :
      |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| ≤ oracle_err_S)
    (h_stage1 : |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen -
                 ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ eps_stage1)
    (h_stage2 : |ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen -
                 ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ eps_stage2)
    (h_S_equiv : |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
                  ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen| ≤ eps_stage1) :
    |loss_true_S - loss_true_L| ≤
      oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by
  have h_core :
      |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
        ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤
        2 * eps_stage1 + eps_stage2 :=
    training_path_gap_bound fstar pol_L pol_S pol_ref gen beta g x R T
      eps_stage1 eps_stage2 h_stage1 h_stage2 h_S_equiv
  have hL :
      |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen - loss_true_L| ≤ oracle_err_L := by
    rw [abs_sub_comm]
    exact h_oracle_L
  have h_triangle1 :
      |loss_true_S - loss_true_L| ≤
        |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| +
          |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen - loss_true_L| := by
    have hdecomp :
        loss_true_S - loss_true_L =
          (loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen) +
            (ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen - loss_true_L) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  have h_triangle2 :
      |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen - loss_true_L| ≤
        |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
          ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| +
        |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen - loss_true_L| := by
    have hdecomp :
        ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen - loss_true_L =
          (ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
            ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen) +
          (ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen - loss_true_L) := by
      ring
    rw [hdecomp]
    exact abs_add_le _ _
  calc
    |loss_true_S - loss_true_L|
        ≤ |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| +
            |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen - loss_true_L| := h_triangle1
    _ ≤ |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| +
          (|ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
              ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| +
            |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen - loss_true_L|) := by
          linarith [h_triangle2]
    _ ≤ oracle_err_S + ((2 * eps_stage1 + eps_stage2) + oracle_err_L) := by
          linarith [h_oracle_S, h_core, hL]
    _ = oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by ring

/-!
## Training Path Bundle
-/

/-- Bundle containing all conditions for two-stage training path analysis. -/
structure TrainingPathBundle {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (pol_L pol_S : Policy Strings A) (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A) (beta : ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y) where
  local_laws : LocalLawsBundle g T fstar
  hp : S T = x
  hR : R ≥ 1
  teacher_measurable : DPO.OracleMeasurable pol_L fstar
  student_measurable : DPO.OracleMeasurable pol_S fstar
  ref_measurable : DPO.OracleMeasurable pol_ref fstar
  pair_indexed : OracleIndexedPairGen gen fstar
  stage1_gap : ℝ
  stage2_gap : ℝ
  h_stage1 : |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen -
              ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ stage1_gap
  h_stage2 : |ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen -
              ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ stage2_gap
  h_S_equiv : |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
               ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen| ≤ stage1_gap

/-- Training path gap bound using bundled conditions.
    Returns bound 2 * stage1_gap + stage2_gap via triangle inequality. -/
theorem training_path_bundle_gap {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    {pol_L pol_S pol_ref : Policy Strings A}
    {gen : PairGenerator Strings A} {beta : ℝ}
    {g : Summarizer Strings} {x : Strings} {R : ℕ} {T : BinTree Strings}
    {fstar : Strings → Y}
    (bundle : TrainingPathBundle pol_L pol_S pol_ref gen beta g x R T fstar) :
    |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
     ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤
    2 * bundle.stage1_gap + bundle.stage2_gap :=
  training_path_gap_bound fstar pol_L pol_S pol_ref gen beta g x R T
    bundle.stage1_gap bundle.stage2_gap bundle.h_stage1 bundle.h_stage2 bundle.h_S_equiv

/-- Bundle-driven DPO training-path gap with optional oracle measurement on the
teacher and student objectives. -/
theorem training_path_bundle_gap_with_oracleMeasurement
    {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    {pol_L pol_S pol_ref : Policy Strings A}
    {gen : PairGenerator Strings A} {beta : ℝ}
    {g : Summarizer Strings} {x : Strings} {R : ℕ} {T : BinTree Strings}
    {fstar : Strings → Y}
    (bundle : TrainingPathBundle pol_L pol_S pol_ref gen beta g x R T fstar)
    (loss_true_L loss_true_S oracle_err_L oracle_err_S : ℝ)
    (h_oracle_L :
      |loss_true_L - ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤ oracle_err_L)
    (h_oracle_S :
      |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| ≤ oracle_err_S) :
    |loss_true_S - loss_true_L| ≤
      oracle_err_S + oracle_err_L + 2 * bundle.stage1_gap + bundle.stage2_gap :=
  training_path_gap_bound_with_oracleMeasurement fstar pol_L pol_S pol_ref gen beta g x R T
    loss_true_L loss_true_S bundle.stage1_gap bundle.stage2_gap oracle_err_L oracle_err_S
    h_oracle_L h_oracle_S bundle.h_stage1 bundle.h_stage2 bundle.h_S_equiv

/-- If the teacher is oracle-optimal for the true objective, then the student is
`ε`-optimal for that true objective whenever the two-stage noisy training path
gap is bounded. -/
theorem training_path_epsilon_optimal_with_oracleMeasurement
    {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (trueLoss : Policy Strings A → ℝ)
    (pol_L pol_S : Policy Strings A)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (beta : ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (loss_true_L loss_true_S : ℝ)
    (eps_stage1 eps_stage2 oracle_err_L oracle_err_S : ℝ)
    (h_L_opt : OptimalForOracle pol_L fstar trueLoss)
    (h_S_meas : DPO.OracleMeasurable pol_S fstar)
    (h_true_L : trueLoss pol_L = loss_true_L)
    (h_true_S : trueLoss pol_S = loss_true_S)
    (h_oracle_L :
      |loss_true_L - ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤ oracle_err_L)
    (h_oracle_S :
      |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| ≤ oracle_err_S)
    (h_stage1 : |ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen -
                 ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ eps_stage1)
    (h_stage2 : |ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen -
                 ExpectedDPOLoss pol_L pol_ref beta (ZR g x R T) gen| ≤ eps_stage2)
    (h_S_equiv : |ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen -
                  ExpectedDPOLoss pol_S pol_ref beta (ZR g x R T) gen| ≤ eps_stage1) :
    EpsilonOptimalForOracle pol_S fstar trueLoss
      (oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2) := by
  have h_gap :
      |trueLoss pol_S - trueLoss pol_L| ≤
        oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2 := by
    simpa [h_true_S, h_true_L] using
      (training_path_gap_bound_with_oracleMeasurement
        (fstar := fstar) (pol_L := pol_L) (pol_S := pol_S)
        (pol_ref := pol_ref) (gen := gen) (beta := beta)
        (g := g) (x := x) (R := R) (T := T)
        (loss_true_L := loss_true_L) (loss_true_S := loss_true_S)
        (eps_stage1 := eps_stage1) (eps_stage2 := eps_stage2)
        (oracle_err_L := oracle_err_L) (oracle_err_S := oracle_err_S)
        h_oracle_L h_oracle_S h_stage1 h_stage2 h_S_equiv)
  exact epsilonOptimal_of_reference_gap pol_L pol_S fstar trueLoss
    (oracle_err_S + oracle_err_L + 2 * eps_stage1 + eps_stage2)
    h_L_opt h_S_meas h_gap

/-- Bundle-driven version of
`training_path_epsilon_optimal_with_oracleMeasurement`. -/
theorem training_path_bundle_epsilon_optimal_with_oracleMeasurement
    {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    {pol_L pol_S pol_ref : Policy Strings A}
    {gen : PairGenerator Strings A} {beta : ℝ}
    {g : Summarizer Strings} {x : Strings} {R : ℕ} {T : BinTree Strings}
    {fstar : Strings → Y}
    (trueLoss : Policy Strings A → ℝ)
    (bundle : TrainingPathBundle pol_L pol_S pol_ref gen beta g x R T fstar)
    (loss_true_L loss_true_S oracle_err_L oracle_err_S : ℝ)
    (h_L_opt : OptimalForOracle pol_L fstar trueLoss)
    (h_S_meas : DPO.OracleMeasurable pol_S fstar)
    (h_true_L : trueLoss pol_L = loss_true_L)
    (h_true_S : trueLoss pol_S = loss_true_S)
    (h_oracle_L :
      |loss_true_L - ExpectedDPOLoss pol_L pol_ref beta (PMF.pure x) gen| ≤ oracle_err_L)
    (h_oracle_S :
      |loss_true_S - ExpectedDPOLoss pol_S pol_ref beta (PMF.pure x) gen| ≤ oracle_err_S) :
    EpsilonOptimalForOracle pol_S fstar trueLoss
      (oracle_err_S + oracle_err_L + 2 * bundle.stage1_gap + bundle.stage2_gap) := by
  exact training_path_epsilon_optimal_with_oracleMeasurement
    (fstar := fstar) (trueLoss := trueLoss)
    (pol_L := pol_L) (pol_S := pol_S) (pol_ref := pol_ref)
    (gen := gen) (beta := beta)
    (g := g) (x := x) (R := R) (T := T)
    (loss_true_L := loss_true_L) (loss_true_S := loss_true_S)
    (eps_stage1 := bundle.stage1_gap) (eps_stage2 := bundle.stage2_gap)
    (oracle_err_L := oracle_err_L) (oracle_err_S := oracle_err_S)
    h_L_opt h_S_meas h_true_L h_true_S
    h_oracle_L h_oracle_S
    bundle.h_stage1 bundle.h_stage2 bundle.h_S_equiv

/-!
## Connection to DPO Gap Bounds
-/

/-- Stage 1 gap from DPO union bound. -/
def Stage1GapFromUnionBound {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (pol_L pol_ref : Policy Strings A) (gen : PairGenerator Strings A) (beta : ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (L_pol : ℝ≥0)
    (leafViol mergeViol pIdemp : ℝ) : ℝ :=
  2 * |beta| * (L_pol : ℝ) * (leafViol + mergeViol + (R - 1) * pIdemp)

end TrainingPipeline

end
