import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.AuditBounds

/-!
# FormalProofs/PreferenceLearning.lean

## Abstract Framework for Preference Learning

This file provides an abstract framework for preference learning methods that generalizes
beyond DPO (Direct Preference Optimization) to encompass modern methods like:
- GRPO (Group Relative Policy Optimization)
- PPO (Proximal Policy Optimization)
- RLHF (Reinforcement Learning from Human Feedback)

### Key Insight

The core theorem of oracle-preserving summarization applies to ANY preference learning
method where:
1. The loss function is oracle-measurable (depends on documents only through f*(x))
2. Preference pairs/groups are generated in an oracle-indexed manner
3. The loss satisfies regularity conditions (Lipschitz) for quantitative bounds

### Paper Reference

This file generalizes Section 6 of the paper. The original DPO-specific theorems are
preserved in PreferenceBounds.lean (formerly DPO.lean) as concrete instantiations of the abstract framework defined here.

### Structure

1. **Abstract Definitions**: `PreferenceLearningMethod` typeclass
2. **General Pair Generator**: `PairGenerator`, `OracleIndexedPairGen` (method-agnostic)
3. **Coupling Lemmas**: Abstract coupling bounds (imported from PreferenceBounds.lean (formerly DPO.lean))
4. **Main Theorems**: `preference_learning_equivalence` - the generalized zero-gap theorem

### Connection to PreferenceBounds.lean (formerly DPO.lean)

PreferenceBounds.lean (formerly DPO.lean) provides:
- `DPOMethod`: A concrete instance of `PreferenceLearningMethod`
- All existing DPO theorems remain valid and are now understood as instantiations

### Usage

To prove preference learning equivalence for a new method (e.g., GRPO):
1. Define a `PreferenceLearningMethod` instance for your method
2. Prove your loss function is oracle-measurable
3. Apply the abstract theorems from this file
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

section PreferenceLearning

open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

-- Action space for policies/responses
variable {A : Type*}

-- Document and oracle spaces
variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Layer 0: Method-Agnostic Infrastructure

This section provides foundational definitions and theorems that apply to
ANY preference learning method. These are independent of whether we use
pairwise (DPO) or group (GRPO) comparisons.

Key concepts:
- **Lipschitz continuity**: Quantitative bounds on how loss changes with oracle value
- **Expected loss**: Generic expectation over document distributions
- **Gap bounds**: Relating Lipschitz continuity to training gap

This infrastructure is shared by all preference methods (DPO, GRPO-PL, GRPO-RL, etc.)
-/

/-- Generic Lipschitz loss: loss difference bounded by L * oracle distance.
This is the most general form - α can be any type (pairs, groups, etc.).
For pairwise losses, use the specialized `LipschitzLoss`.
For group losses, use `LipschitzGroupLoss`. -/
def LipschitzLossGeneric {Strings α Y : Type*} [PseudoMetricSpace Y]
    (loss : Strings → α → ℝ) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ x x' a, |loss x a - loss x' a| ≤ (L : ℝ) * dist (fstar x) (fstar x')

/-- Generic expected loss over a document distribution and generator.
This is the objective being optimized during training. -/
noncomputable def ExpectedLossGeneric {Strings α : Type*} [Monoid Strings]
    (loss : Strings → α → ℝ) (μ : PMF Strings) (gen : Strings → PMF α) : ℝ :=
  ∑' x, (μ x).toReal * ∑' a, (gen x a).toReal * loss x a

/-!
## Layer 1: Pair Generator (Method-Agnostic)

The pair generator produces preference pairs conditioned on a document.
This is shared across all pairwise preference learning methods.

For group-based methods (GRPO), one would define a `GroupGenerator` analog.
-/

/-- Pair generator: samples (a_w, a_ℓ) preference pairs conditioned on document x.
This is method-agnostic - DPO, RLHF, etc. all use this structure for pairwise comparisons. -/
def PairGenerator (Strings A : Type*) := Strings → PMF (A × A)

/-- Oracle-indexed pair generator: pair generation depends on document only through oracle value.
This is the key condition enabling equivalence under oracle-preserving summarization. -/
def OracleIndexedPairGen {Strings A Y : Type*} [PseudoMetricSpace Y]
    (gen : PairGenerator Strings A) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → gen x = gen x'

/-!
## Abstract Preference Learning Method

A preference learning method is characterized by:
1. A parameter space (e.g., policies for DPO, reward parameters for RLHF)
2. A pointwise loss function over preference pairs
3. Oracle-measurability conditions
4. Regularity conditions (Lipschitz) for quantitative bounds
-/

/-- Abstract preference learning method signature.

This typeclass captures the common structure across preference learning methods:
- DPO: parameters are (policy, reference_policy), loss is -log(σ(logit))
- GRPO: parameters include group comparison weights
- PPO: parameters include value function and policy

The key requirement is that all methods must support oracle-measurability analysis. -/
class PreferenceLearningMethod (Strings A Y : Type*) [PseudoMetricSpace Y] where
  /-- The parameter space for this method (e.g., policy pairs for DPO) -/
  Param : Type*
  /-- Pointwise loss on a preference pair -/
  pointwiseLoss : Param → Strings → A → A → ℝ
  /-- Oracle-measurability: loss depends on x only through f*(x) -/
  oracleMeasurable : Param → (Strings → Y) → Prop
  /-- Lipschitz condition for quantitative gap bounds -/
  lipschitzCondition : Param → (Strings → Y) → ℝ≥0 → Prop

/-- Expected loss under a preference learning method.
This is the objective being optimized during training. -/
noncomputable def expectedLoss {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (M : PreferenceLearningMethod Strings A Y)
    (θ : M.Param) (μ : PMF Strings) (gen : PairGenerator Strings A) : ℝ :=
  ∑' x, (μ x).toReal * ∑' p, (gen x p).toReal * M.pointwiseLoss θ x p.1 p.2

/-!
## Oracle-Measurable Loss Equality

The fundamental lemma: when oracle values are preserved (zero distortion),
expected loss is identical for any oracle-measurable preference learning method.
-/

/-- Generic oracle-measurable loss predicate for arbitrary types. -/
def OracleMeasurableLossGeneric {Strings α Y : Type*} [PseudoMetricSpace Y]
    (loss : Strings → α → ℝ) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → loss x a = loss x' a

/-- Generic oracle-indexed generator predicate. -/
def OracleIndexedGenGeneric {Strings α Y : Type*} [PseudoMetricSpace Y]
    (gen : Strings → PMF α) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → gen x = gen x'

/-- **Fundamental Lemma:** Generic zero-distortion implies equal expected loss.

This is the unified theorem from which all specific preference learning
equivalence results (DPO, GRPO, etc.) can be derived. When oracle values
are preserved (dist(f*(z), f*(x)) = 0), the expected loss is identical
regardless of whether we compute it on originals or summaries. -/
lemma expected_loss_eq_of_zero_dist_generic {Strings α Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : Strings → α → ℝ)
    (gen : Strings → PMF α)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurableLossGeneric loss fstar)
    (h_gen : OracleIndexedGenGeneric gen fstar) :
    ∑' x, (μ_X x).toReal * ∑' a, (gen x a).toReal * loss x a =
    ∑' z, (μ_Z z).toReal * ∑' a, (gen z a).toReal * loss z a := by
  obtain ⟨x₀, hx₀⟩ := μ_X.support_nonempty
  let L₀ := fun a : α => (gen x₀ a).toReal * loss x₀ a
  have h_X_eq : ∀ x, x ∈ μ_X.support →
      ∑' a, (gen x a).toReal * loss x a = ∑' a, L₀ a := by
    intro x hx
    apply tsum_congr
    intro a
    obtain ⟨z₀, hz₀⟩ := μ_Z.support_nonempty
    have hd1 : dist (fstar z₀) (fstar x₀) = 0 := h_zero z₀ x₀ hz₀ hx₀
    have hd2 : dist (fstar z₀) (fstar x) = 0 := h_zero z₀ x hz₀ hx
    have hd : dist (fstar x) (fstar x₀) = 0 := by
      have h_tri := dist_triangle (fstar x) (fstar z₀) (fstar x₀)
      have h1 : dist (fstar x) (fstar z₀) = 0 := by rw [dist_comm]; exact hd2
      have h2 : dist (fstar z₀) (fstar x₀) = 0 := hd1
      rw [h1, h2] at h_tri
      simp only [zero_add] at h_tri
      linarith [dist_nonneg (α := Y) (x := fstar x) (y := fstar x₀)]
    have h_gen_eq : gen x = gen x₀ := h_gen x x₀ hd
    have h_loss_eq : loss x a = loss x₀ a := h_meas x x₀ a hd
    rw [h_gen_eq, h_loss_eq]
  have h_Z_eq : ∀ z, z ∈ μ_Z.support →
      ∑' a, (gen z a).toReal * loss z a = ∑' a, L₀ a := by
    intro z hz
    apply tsum_congr
    intro a
    have hd : dist (fstar z) (fstar x₀) = 0 := h_zero z x₀ hz hx₀
    have h_gen_eq : gen z = gen x₀ := h_gen z x₀ hd
    have h_loss_eq : loss z a = loss x₀ a := h_meas z x₀ a hd
    rw [h_gen_eq, h_loss_eq]
  have h_X : ∑' x, (μ_X x).toReal * ∑' a, (gen x a).toReal * loss x a = ∑' a, L₀ a := by
    have h_eq : ∀ x, (μ_X x).toReal * ∑' a, (gen x a).toReal * loss x a =
                (μ_X x).toReal * ∑' a, L₀ a := by
      intro x
      by_cases hx : x ∈ μ_X.support
      · rw [h_X_eq x hx]
      · have h_zero_app : μ_X x = 0 := (μ_X.apply_eq_zero_iff x).mpr hx
        simp only [h_zero_app, ENNReal.toReal_zero, zero_mul]
    simp_rw [h_eq]
    have h_factor : (fun x => (μ_X x).toReal * ∑' a, L₀ a) =
                    (fun x => (∑' a, L₀ a) * (μ_X x).toReal) := by ext x; ring
    rw [h_factor, tsum_mul_left, PMF.toReal_tsum_coe, mul_one]
  have h_Z : ∑' z, (μ_Z z).toReal * ∑' a, (gen z a).toReal * loss z a = ∑' a, L₀ a := by
    have h_eq : ∀ z, (μ_Z z).toReal * ∑' a, (gen z a).toReal * loss z a =
                (μ_Z z).toReal * ∑' a, L₀ a := by
      intro z
      by_cases hz : z ∈ μ_Z.support
      · rw [h_Z_eq z hz]
      · have h_zero_app : μ_Z z = 0 := (μ_Z.apply_eq_zero_iff z).mpr hz
        simp only [h_zero_app, ENNReal.toReal_zero, zero_mul]
    simp_rw [h_eq]
    have h_factor : (fun z => (μ_Z z).toReal * ∑' a, L₀ a) =
                    (fun z => (∑' a, L₀ a) * (μ_Z z).toReal) := by ext z; ring
    rw [h_factor, tsum_mul_left, PMF.toReal_tsum_coe, mul_one]
  rw [h_X, h_Z]

/-- Oracle-measurable loss: loss values are equal when oracle values are equal.
This is the key property enabling loss equivalence under summarization. -/
def OracleMeasurableLoss {Strings A Y : Type*} [PseudoMetricSpace Y]
    (loss : Strings → A → A → ℝ) (fstar : Strings → Y) : Prop :=
  ∀ x x' a_w a_ℓ, dist (fstar x) (fstar x') = 0 → loss x a_w a_ℓ = loss x' a_w a_ℓ

/-- Key lemma: Zero distortion implies equal expected loss for oracle-measurable methods.

This is the pairwise specialization of `expected_loss_eq_of_zero_dist_generic`.
When summaries preserve oracle values (dist(f*(z), f*(x)) = 0), the expected
loss on summaries equals the expected loss on originals.

**Mathematical Statement:**
If ∀ z ∈ supp(μ_Z), x ∈ supp(μ_X): dist(f*(z), f*(x)) = 0, then:
  E_{μ_X}[L(θ; x, gen)] = E_{μ_Z}[L(θ; z, gen)]
-/
lemma expected_loss_eq_of_zero_dist_general {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurableLoss loss fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    ∑' x, (μ_X x).toReal * ∑' p, (gen x p).toReal * loss x p.1 p.2 =
    ∑' z, (μ_Z z).toReal * ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  -- Convert curried loss to uncurried form for generic lemma
  let loss' : Strings → (A × A) → ℝ := fun x p => loss x p.1 p.2
  have h_meas' : OracleMeasurableLossGeneric loss' fstar :=
    fun x x' p hdist => h_meas x x' p.1 p.2 hdist
  exact expected_loss_eq_of_zero_dist_generic fstar loss' gen μ_X μ_Z h_zero h_meas' h_pair

/-!
## Abstract Same-Argmin Definition

The key result: when expected losses are equal, the argmin sets (optimal parameters) are identical.
-/

/-- Parameter argmin: the set of parameters that minimize a loss functional. -/
def ParamArgmin {Θ : Type*} (loss : Θ → ℝ) : Set Θ :=
  {θ | ∀ θ', loss θ ≤ loss θ'}

/-- Oracle-measurable parameter argmin: minimizers among oracle-measurable parameters. -/
def OracleMeasurableParamArgmin {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (loss : Θ → ℝ) (isMeasurable : Θ → (Strings → Y) → Prop) (fstar : Strings → Y) : Set Θ :=
  {θ | isMeasurable θ fstar ∧ ∀ θ', isMeasurable θ' fstar → loss θ ≤ loss θ'}

/-- Two loss functions have the same oracle-measurable argmin. -/
def SameOracleMeasurableArgminGeneral {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (loss₁ loss₂ : Θ → ℝ) (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y) : Prop :=
  OracleMeasurableParamArgmin loss₁ isMeasurable fstar =
  OracleMeasurableParamArgmin loss₂ isMeasurable fstar

/-- If two losses agree on every oracle-measurable parameter, then they have the
same oracle-measurable argmin set. -/
theorem same_oracle_measurable_argmin_general_of_loss_eq
    {Strings Y Θ : Type*} [PseudoMetricSpace Y]
    (loss₁ loss₂ : Θ → ℝ) (isMeasurable : Θ → (Strings → Y) → Prop)
    (fstar : Strings → Y)
    (h_eq : ∀ θ, isMeasurable θ fstar → loss₁ θ = loss₂ θ) :
    SameOracleMeasurableArgminGeneral loss₁ loss₂ isMeasurable fstar := by
  unfold SameOracleMeasurableArgminGeneral OracleMeasurableParamArgmin
  ext θ
  constructor
  · intro ⟨h_meas, h_min⟩
    constructor
    · exact h_meas
    · intro θ' h_meas'
      rw [← h_eq θ h_meas, ← h_eq θ' h_meas']
      exact h_min θ' h_meas'
  · intro ⟨h_meas, h_min⟩
    constructor
    · exact h_meas
    · intro θ' h_meas'
      rw [h_eq θ h_meas, h_eq θ' h_meas']
      exact h_min θ' h_meas'

/-!
## Main Theorem: Preference Learning Equivalence

When local laws hold (zero expected distortion), preference learning on summarized
data is equivalent to preference learning on original data.
-/

/-- **Main Theorem: Preference Learning Equivalence under Local Laws**

This generalizes Theorem 6.1 from PreferenceBounds.lean (formerly DPO.lean) to arbitrary preference learning methods.

When the summarization satisfies local laws (L1, L2, L3) ensuring zero expected
distortion, any oracle-measurable preference learning method achieves the same
expected loss on summarized data as on original data.

**Paper Reference:** Generalization of Section 6, Theorem 6.1

**Corollary:** The optimal parameters (argmin) are identical for both distributions.
-/
theorem preference_learning_equivalence {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    -- Zero distortion (implied by local laws in practice)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    -- Oracle-measurability
    (h_meas : OracleMeasurableLoss loss fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    ∑' x, (μ_X x).toReal * ∑' p, (gen x p).toReal * loss x p.1 p.2 =
    ∑' z, (μ_Z z).toReal * ∑' p, (gen z p).toReal * loss z p.1 p.2 :=
  expected_loss_eq_of_zero_dist_general fstar loss gen μ_X μ_Z h_zero h_meas h_pair

/-- Zero gap version: the absolute difference is exactly zero.
This is the form used in most PreferenceBounds.lean (formerly DPO.lean) theorems. -/
theorem preference_learning_gap_zero {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurableLoss loss fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    |∑' x, (μ_X x).toReal * ∑' p, (gen x p).toReal * loss x p.1 p.2 -
     ∑' z, (μ_Z z).toReal * ∑' p, (gen z p).toReal * loss z p.1 p.2| = 0 := by
  rw [preference_learning_equivalence fstar loss gen μ_X μ_Z h_zero h_meas h_pair]
  simp

/-!
## Connection to ZR (Multi-Round Reduction)

When μ_X = pure(x) and μ_Z = ZR(g, x, R, T), the zero-distortion hypothesis
is satisfied when local laws L1, L2, L3 hold. This is proven in PreferenceBounds.lean (formerly DPO.lean)
via `dpo_gap_zero_of_local_laws` and now applies to all preference methods.
-/

/-- Connecting abstract theorem to ZR.
This shows how the abstract framework applies to the tree-based summarization setting. -/
theorem preference_learning_equivalence_via_ZR {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : Strings → A → A → ℝ)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    -- Local laws ensure zero distortion
    (hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    -- Oracle-measurability
    (h_meas : OracleMeasurableLoss loss fstar)
    (h_pair : OracleIndexedPairGen gen fstar)
    -- Boundedness for multi_round_proper (axiom-free)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ∑' x', (PMF.pure x x').toReal * ∑' p, (gen x' p).toReal * loss x' p.1 p.2 =
    ∑' z, (ZR g x R T z).toReal * ∑' p, (gen z p).toReal * loss z p.1 p.2 := by
  apply preference_learning_equivalence fstar loss gen (PMF.pure x) (ZR g x R T)
  -- Prove zero distortion from local laws
  intro z x' hz hx'
  simp only [PMF.support_pure, Set.mem_singleton_iff] at hx'
  rw [hx']
  -- Use multi_round_proper to get E[D] = 0
  have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_proper g T x R fstar hp h1 h2 h3 hR M hM hbound
  -- E[D] = 0 with D ≥ 0 implies D = 0 on support
  unfold D at h_exp_zero
  by_contra h_dist_ne_zero
  have h_dist_pos : 0 < dist (fstar z) (fstar x) :=
    lt_of_le_of_ne dist_nonneg (Ne.symm h_dist_ne_zero)
  have h_term_pos : 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
    apply mul_pos
    · exact ENNReal.toReal_pos hz (PMF.apply_ne_top _ _)
    · exact h_dist_pos
  have h_summable : Summable (fun z => (ZR g x R T z).toReal * dist (fstar z) (fstar x)) :=
    summable_D_of_bounded (ZR g x R T) fstar x M hM (fun z => hbound z x)
  have h_sum_pos : 0 < ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
    calc 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := h_term_pos
         _ ≤ ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
             apply Summable.le_tsum h_summable z
             intro i _
             exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
  unfold Exp at h_exp_zero
  linarith [h_exp_zero]
  -- Other hypotheses
  exact h_meas
  exact h_pair

/-!
## Lipschitz Gap Bound (Quantitative Version)

For quantitative gap bounds (when distortion is not exactly zero), see PreferenceBounds.lean (formerly DPO.lean)
which provides:
- `dpo_gap`: The original DPO-specific gap bound
- `dpo_gap_typeclass`: Uses BoundedMetricSpace for automatic bounds

The coupling argument in PreferenceBounds.lean (formerly DPO.lean) can be adapted to any preference learning method
by instantiating the abstract framework. The key lemmas (`coupling_expansion`,
`coupling_bound_ineq`) are proven there.
-/

/-- Lipschitz loss: the loss difference is bounded by a constant times the oracle distance.
This is the quantitative analog of `OracleMeasurableLoss`. -/
def LipschitzLoss {Strings A Y : Type*} [PseudoMetricSpace Y]
    (loss : Strings → A → A → ℝ) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ x x' a_w a_ℓ, |loss x a_w a_ℓ - loss x' a_w a_ℓ| ≤ (L : ℝ) * dist (fstar x) (fstar x')

/-!
## GRPO Variant 1: Listwise Ranking (Plackett-Luce)

GRPO generalizes DPO from pairwise comparisons (k=2) to group rankings (k > 2).
Instead of comparing two actions at a time, GRPO ranks k candidates.

**Distinction from GRPO-RL**: This section uses Plackett-Luce listwise ranking loss,
which directly models the probability of a ranking. The next section (GRPO-RL)
formalizes the clipped-surrogate objective with z-score advantages used by DeepSeek-R1.

Both variants satisfy oracle-measurability and thus inherit training equivalence.

### Mathematical Structure

- **Group Generator**: Samples k candidates conditioned on document x
- **Ranking Function**: Maps groups to ordinal rankings
- **Plackett-Luce Loss**: P(ranking) = ∏ exp(s_i) / Σ_{j≥i} exp(s_j)

### Key Result

The same oracle-measurability conditions that ensure DPO equivalence also
ensure GRPO equivalence: when local laws hold, training on summarized data
gives the same optimal policy as training on original data.
-/

/-- Group generator: samples k candidates conditioned on document x.
    Generalizes PairGenerator from k=2 to arbitrary k.
    The output is a PMF over functions from Fin k to A (k-tuples of actions). -/
def GroupGenerator (Strings A : Type*) (k : ℕ) := Strings → PMF (Fin k → A)

/-- Oracle-indexed group generator: group generation depends on document only through oracle value.
    This is the key condition enabling equivalence under oracle-preserving summarization. -/
def OracleIndexedGroupGen {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (gen : GroupGenerator Strings A k) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → gen x = gen x'

/-- L1-Lipschitz group generator: generator shift bounded by oracle distance.

We measure generator shift by the L1 distance between PMFs:
  ∑_g |gen x g - gen x' g| ≤ L * dist(f*(x), f*(x')).

This is the minimal stability condition needed to extend TreePO bounds
to doc-dependent group generators. -/
def GroupGeneratorLipschitzL1 {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    [Fintype A]
    (gen : GroupGenerator Strings A k) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ x x', (∑ g, |(gen x g).toReal - (gen x' g).toReal|) ≤ (L : ℝ) * dist (fstar x) (fstar x')

/-- Group loss: loss on a k-sized group of actions. -/
def GroupLoss (Strings A : Type*) (k : ℕ) := Strings → (Fin k → A) → ℝ

/-- Oracle-measurable group loss: loss depends on document only through oracle value. -/
def OracleMeasurableGroupLoss {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (loss : GroupLoss Strings A k) (fstar : Strings → Y) : Prop :=
  ∀ x x' group, dist (fstar x) (fstar x') = 0 → loss x group = loss x' group

/-- Lipschitz group loss: group loss difference bounded by L * oracle distance.
This is the quantitative analog of `OracleMeasurableGroupLoss`, and the group
analog of `LipschitzLoss` for pairwise losses.

When a group loss is Lipschitz, we get quantitative gap bounds for training
on summarized data vs. original data (see `group_gap_lipschitz`). -/
def LipschitzGroupLoss {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (loss : GroupLoss Strings A k) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ x x' group, |loss x group - loss x' group| ≤ (L : ℝ) * dist (fstar x) (fstar x')

/-- Expected group loss over a distribution and group generator. -/
noncomputable def ExpectedGroupLoss {Strings A : Type*} [Monoid Strings] {k : ℕ}
    (loss : GroupLoss Strings A k) (μ : PMF Strings) (gen : GroupGenerator Strings A k) : ℝ :=
  ∑' x, (μ x).toReal * ∑' g, (gen x g).toReal * loss x g

/-- Oracle-measurable group loss equality under zero distortion.
This is a specialization of `expected_loss_eq_of_zero_dist_generic` for group generators. -/
theorem expected_group_loss_eq_of_zero_dist {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (loss : GroupLoss Strings A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurableGroupLoss loss fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGroupLoss loss μ_X gen = ExpectedGroupLoss loss μ_Z gen := by
  unfold ExpectedGroupLoss
  exact expected_loss_eq_of_zero_dist_generic fstar loss gen μ_X μ_Z h_zero h_meas h_gen

/-- Group ranking function: maps a group of k actions to their ordinal ranks (1 to k).
    The ranking encodes preferences: rank 1 is most preferred, rank k is least preferred. -/
def GroupRanker (A : Type*) (k : ℕ) := (Fin k → A) → (Fin k → ℕ)

/-- Oracle-indexed ranker: the ranking depends on document only through oracle value. -/
def OracleIndexedRanker {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (ranker : Strings → GroupRanker A k) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → ranker x = ranker x'

/-- Policy type (shared with DPO) -/
def Policy' (Strings A : Type*) := Strings → A → ℝ

/-- Plackett-Luce log-likelihood for a ranking.

    The Plackett-Luce model gives the probability of a ranking as a product:
    P(ranking) = ∏_{i=1}^{k} exp(score(a_i)) / ∑_{j=i}^{k} exp(score(a_j))

    where a_i is the action ranked in position i.

    The GRPO loss is the negative log-likelihood. -/
noncomputable def PlackettLuceLogProb {A : Type*} {k : ℕ}
    (scores : Fin k → ℝ) (ranks : Fin k → ℕ) : ℝ :=
  -- Sum over positions: log P(a_i chosen from remaining)
  -- Note: This is a simplified version; full implementation would order by ranks
  ∑ i : Fin k, (scores i - Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores j) else 0))

/-- GRPO pointwise loss: negative Plackett-Luce log-likelihood.

    Given a policy π and a ranked group of actions, the loss encourages
    the policy to assign higher probabilities to higher-ranked actions. -/
noncomputable def GRPOLossPointwise {Strings A : Type*} {k : ℕ}
    (pol : Policy' Strings A) (x : Strings) (group : Fin k → A) (ranks : Fin k → ℕ) : ℝ :=
  -PlackettLuceLogProb (A := A) (fun i => pol x (group i)) ranks

/-- Oracle-measurable policy (for GRPO context) -/
def GRPOOracleMeasurable {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol : Policy' Strings A) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → pol x a = pol x' a

/-- Oracle-measurable GRPO loss: when policy and ranker are oracle-measurable,
    the GRPO loss depends on document only through oracle value. -/
def OracleMeasurableGRPOLoss {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) : Prop :=
  ∀ x x' group, dist (fstar x) (fstar x') = 0 →
    GRPOLossPointwise pol x group (ranker x group) =
    GRPOLossPointwise pol x' group (ranker x' group)

/-- Oracle-measurable GRPO loss follows from oracle-measurable policy and ranker. -/
lemma grpo_loss_oracle_measurable {Strings A Y : Type*} [MetricSpace Y] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k) (fstar : Strings → Y)
    (h_pol : GRPOOracleMeasurable pol fstar)
    (h_ranker : OracleIndexedRanker ranker fstar) :
    OracleMeasurableGRPOLoss pol ranker fstar := by
  intro x x' group hdist
  unfold GRPOLossPointwise PlackettLuceLogProb
  have h_pol_eq : ∀ i, pol x (group i) = pol x' (group i) := by
    intro i
    exact h_pol x x' (group i) hdist
  have h_ranker_eq : ranker x group = ranker x' group := by
    have h := h_ranker x x' hdist
    simp only [h]
  simp only [h_ranker_eq]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  simp only [h_pol_eq]

/-!
### GRPO Lipschitz Infrastructure

These definitions and lemmas connect GRPO to the abstract Lipschitz framework.
-/

/-- Lipschitz policy for GRPO: policy differences bounded by L * oracle distance.
Parallel to `PolicyLipschitz` for DPO. -/
def GRPOPolicyLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol : Policy' Strings A) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ x x' a, |pol x a - pol x' a| ≤ (L : ℝ) * dist (fstar x) (fstar x')

lemma grpo_policy_lipschitz_mono {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol : Policy' Strings A} {fstar : Strings → Y} {L L' : ℝ≥0}
    (h : GRPOPolicyLipschitz pol fstar L) (hL : L ≤ L') :
    GRPOPolicyLipschitz pol fstar L' := by
  intro x x' a
  have hL' : (L : ℝ) ≤ (L' : ℝ) := by
    exact_mod_cast hL
  have hmul :
      (L : ℝ) * dist (fstar x) (fstar x') ≤
      (L' : ℝ) * dist (fstar x) (fstar x') := by
    exact mul_le_mul_of_nonneg_right hL' dist_nonneg
  exact le_trans (h x x' a) hmul

/-- GRPO-PL loss satisfies the abstract LipschitzGroupLoss predicate.

This is the bridge lemma connecting GRPO-specific infrastructure to the abstract framework.
The Lipschitz constant depends on the policy Lipschitz constant and the group size k.

**Note**: The full proof requires showing that Plackett-Luce log-likelihood is Lipschitz
in the scores, which follows from the log-sum-exp being 1-Lipschitz. -/
lemma grpo_pl_loss_satisfies_lipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) (L_grpo : ℝ≥0)
    (h_bound : ∀ x x' group,
      |GRPOLossPointwise pol x group (ranker x group) -
       GRPOLossPointwise pol x' group (ranker x' group)| ≤
      L_grpo * dist (fstar x) (fstar x')) :
    LipschitzGroupLoss (fun x g => GRPOLossPointwise pol x g (ranker x g)) fstar L_grpo :=
  fun x x' group => h_bound x x' group

/-- Expected GRPO loss over a distribution and group generator. -/
noncomputable def ExpectedGRPOLoss {Strings A : Type*} [Monoid Strings] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (μ : PMF Strings) (gen : GroupGenerator Strings A k) : ℝ :=
  ∑' x, (μ x).toReal * ∑' g, (gen x g).toReal * GRPOLossPointwise pol x g (ranker x g)

/-- **GRPO Equivalence Theorem**

    Generalizes DPO equivalence (k=2) to arbitrary k.

    When local laws hold (zero expected distortion), GRPO training on
    summarized data equals GRPO training on original data.

    This is the k>2 analog of `preference_learning_equivalence`. -/
theorem grpo_equivalence {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    -- Zero distortion: summaries have same oracle value as originals
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    -- Oracle-measurability conditions
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPOLoss pol ranker μ_X gen = ExpectedGRPOLoss pol ranker μ_Z gen := by
  have h_meas :
      OracleMeasurableGroupLoss (fun x g => GRPOLossPointwise pol x g (ranker x g)) fstar := by
    intro x x' g hdist
    have h := grpo_loss_oracle_measurable pol ranker fstar h_pol h_ranker
    exact h x x' g hdist
  have h_eq := expected_group_loss_eq_of_zero_dist fstar
    (fun x g => GRPOLossPointwise pol x g (ranker x g)) gen μ_X μ_Z h_zero h_meas h_gen
  simpa [ExpectedGroupLoss, ExpectedGRPOLoss] using h_eq

/-- GRPO-PL same-argmin form of `grpo_equivalence`.

For fixed ranker and group generator, if summaries and originals are
oracle-equivalent on support, then the full-document and summary GRPO-PL losses
have the same argmin set among oracle-measurable policies. -/
theorem grpo_pl_exact_metric {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    SameOracleMeasurableArgminGeneral
      (fun pol : Policy' Strings A => ExpectedGRPOLoss pol ranker μ_X gen)
      (fun pol : Policy' Strings A => ExpectedGRPOLoss pol ranker μ_Z gen)
      (fun pol fstar => GRPOOracleMeasurable pol fstar)
      fstar := by
  apply same_oracle_measurable_argmin_general_of_loss_eq
  intro pol h_pol
  exact grpo_equivalence (Y := Y) fstar pol ranker gen μ_X μ_Z
    h_zero h_pol h_ranker h_gen

/-- GRPO equivalence via ZR: connecting to tree-based summarization. -/
theorem grpo_equivalence_via_ZR {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    -- Local laws ensure zero distortion
    (hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    -- Oracle-measurability conditions
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    -- Boundedness for multi_round_proper
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
    ExpectedGRPOLoss pol ranker (ZR g x R T) gen := by
  apply grpo_equivalence (Y := Y) fstar pol ranker gen (PMF.pure x) (ZR g x R T)
  -- Prove zero distortion from local laws (same as preference_learning_equivalence_via_ZR)
  intro z x' hz hx'
  simp only [PMF.support_pure, Set.mem_singleton_iff] at hx'
  rw [hx']
  have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_proper g T x R fstar hp h1 h2 h3 hR M hM hbound
  unfold D at h_exp_zero
  by_contra h_dist_ne_zero
  have h_dist_pos : 0 < dist (fstar z) (fstar x) :=
    lt_of_le_of_ne dist_nonneg (Ne.symm h_dist_ne_zero)
  have h_term_pos : 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
    apply mul_pos
    · exact ENNReal.toReal_pos hz (PMF.apply_ne_top _ _)
    · exact h_dist_pos
  have h_summable : Summable (fun z => (ZR g x R T z).toReal * dist (fstar z) (fstar x)) :=
    summable_D_of_bounded (ZR g x R T) fstar x M hM (fun z => hbound z x)
  have h_sum_pos : 0 < ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
    calc 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := h_term_pos
         _ ≤ ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
             apply Summable.le_tsum h_summable z
             intro i _
             exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
  unfold Exp at h_exp_zero
  linarith [h_exp_zero]
  -- Other hypotheses
  exact h_pol
  exact h_ranker
  exact h_gen

/-!
## GRPO Variant 2: Clipped-Surrogate + KL (DeepSeek-R1 Style)

This formalizes the group-based RL objective used by DeepSeek-R1 (equations 11-13).

**DeepSeek-R1 GRPO Objective**:
```
J_GRPO(θ) = E[ 1/G Σᵢ min(rᵢ·Aᵢ, clip(rᵢ, 1-ε, 1+ε)·Aᵢ) - β·D_KL(π_θ || π_ref) ]
```
where:
- rᵢ = π_θ(oᵢ|q) / π_θ_old(oᵢ|q) is the policy ratio
- Aᵢ = (reward_i - mean) / std is the z-score normalized advantage
- clip(r, 1-ε, 1+ε) is PPO-style clipping for stability
- D_KL is the unbiased KL estimator: q/p - log(q/p) - 1

**Key difference from Variant 1**: This uses reward-based advantages and clipping,
while Variant 1 uses Plackett-Luce ranking likelihood directly.

**Oracle-measurability**: When policies and rewards depend on x only through f*(x),
the GRPO-RL loss inherits this property, ensuring training equivalence.
-/

/-- Oracle-measurable reward: reward depends on document only through oracle value. -/
def OracleMeasurableReward {Strings A Y : Type*} [PseudoMetricSpace Y]
    (reward : Strings → A → ℝ) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → reward x a = reward x' a

/-- PPO-style clipping for GRPO ratios. -/
noncomputable def GRPOClip (ratio eps : ℝ) : ℝ :=
  max (1 - eps) (min (1 + eps) ratio)

/-- Mean of group rewards. -/
noncomputable def GRPOGroupMean {k : ℕ} (r : Fin k → ℝ) : ℝ :=
  (1 / (k : ℝ)) * ∑ i, r i

/-- Standard deviation of group rewards. -/
noncomputable def GRPOGroupStd {k : ℕ} (r : Fin k → ℝ) : ℝ :=
  Real.sqrt ((1 / (k : ℝ)) * ∑ i, (r i - GRPOGroupMean r)^2)

/-- Advantage normalized within a group. -/
noncomputable def GRPOAdvantage {k : ℕ} (r : Fin k → ℝ) (i : Fin k) : ℝ :=
  (r i - GRPOGroupMean r) / GRPOGroupStd r

/-- Unbiased KL estimator used by GRPO. -/
noncomputable def GRPOKLEstimator (p q : ℝ) : ℝ :=
  q / p - Real.log (q / p) - 1

/-- Group-wise KL penalty. -/
noncomputable def GRPOGroupKL {Strings A : Type*} {k : ℕ}
    (pol pol_ref : Policy' Strings A) (x : Strings) (group : Fin k → A) : ℝ :=
  (1 / (k : ℝ)) * ∑ i, GRPOKLEstimator (pol x (group i)) (pol_ref x (group i))

/-- GRPO pointwise loss: negative clipped-surrogate objective with KL penalty. -/
noncomputable def GRPORLLossPointwise {Strings A : Type*} {k : ℕ}
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (x : Strings) (group : Fin k → A) : ℝ :=
  let ratio : Fin k → ℝ := fun i => pol x (group i) / pol_old x (group i)
  let rewards : Fin k → ℝ := fun i => reward x (group i)
  let adv : Fin k → ℝ := fun i => GRPOAdvantage rewards i
  let surrogate :=
    (1 / (k : ℝ)) * ∑ i, min (ratio i * adv i) (GRPOClip (ratio i) eps * adv i)
  let kl := GRPOGroupKL pol pol_ref x group
  (-(surrogate - beta * kl))

/-- Oracle-measurable GRPO-RL loss. -/
def OracleMeasurableGRPORLLoss {Strings A Y : Type*} [PseudoMetricSpace Y] (k : ℕ)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (fstar : Strings → Y) : Prop :=
  ∀ x x' (group : Fin k → A), dist (fstar x) (fstar x') = 0 →
    GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group =
    GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group

/-- GRPO-RL loss is oracle-measurable when policies and rewards are. -/
lemma grpo_rl_loss_oracle_measurable {Strings A Y : Type*} [MetricSpace Y] (k : ℕ)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ) (fstar : Strings → Y)
    (h_pol : GRPOOracleMeasurable pol fstar)
    (h_old : GRPOOracleMeasurable pol_old fstar)
    (h_ref : GRPOOracleMeasurable pol_ref fstar)
    (h_reward : OracleMeasurableReward reward fstar) :
    OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar := by
  intro x x' group hdist
  have h_pol_eq : ∀ i, pol x (group i) = pol x' (group i) := by
    intro i
    exact h_pol x x' (group i) hdist
  have h_old_eq : ∀ i, pol_old x (group i) = pol_old x' (group i) := by
    intro i
    exact h_old x x' (group i) hdist
  have h_ref_eq : ∀ i, pol_ref x (group i) = pol_ref x' (group i) := by
    intro i
    exact h_ref x x' (group i) hdist
  have h_reward_eq : ∀ i, reward x (group i) = reward x' (group i) := by
    intro i
    exact h_reward x x' (group i) hdist
  simp [GRPORLLossPointwise, GRPOGroupKL, GRPOKLEstimator, GRPOAdvantage, GRPOGroupMean,
    GRPOGroupStd, GRPOClip, h_pol_eq, h_old_eq, h_ref_eq, h_reward_eq]

/-- Lipschitz reward for GRPO-RL: reward differences bounded by L * oracle distance. -/
def RewardLipschitzGRPO {Strings A Y : Type*} [PseudoMetricSpace Y]
    (reward : Strings → A → ℝ) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ x x' a, |reward x a - reward x' a| ≤ (L : ℝ) * dist (fstar x) (fstar x')

/-- GRPO-RL loss satisfies the abstract LipschitzGroupLoss predicate.

This is the bridge lemma for GRPO-RL connecting to the abstract framework.
The Lipschitz constant depends on policy and reward Lipschitz constants, group size k,
clipping parameter eps, and KL penalty coefficient beta.

**Note**: The full proof requires careful analysis of the clipped-surrogate objective
and z-score normalization, which are both Lipschitz in their inputs. -/
lemma grpo_rl_loss_satisfies_lipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] (k : ℕ)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ) (fstar : Strings → Y)
    (L_grpo_rl : ℝ≥0)
    (h_bound : ∀ x x' (group : Fin k → A),
      |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
       GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x' group| ≤
      L_grpo_rl * dist (fstar x) (fstar x')) :
    LipschitzGroupLoss (k := k) (fun x g => GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x g)
      fstar L_grpo_rl :=
  fun x x' group => h_bound x x' group

/-- Expected GRPO-RL loss over a distribution and group generator. -/
noncomputable def ExpectedGRPORLLoss {Strings A : Type*} [Monoid Strings] {k : ℕ}
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (μ : PMF Strings) (gen : GroupGenerator Strings A k) : ℝ :=
  ExpectedGroupLoss (fun x g => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x g) μ gen

/-- GRPO-RL equivalence under zero distortion. -/
theorem grpo_rl_equivalence {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] (k : ℕ)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_X gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_Z gen := by
  let loss : GroupLoss Strings A k := fun x g => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x g
  have h_meas' : OracleMeasurableGroupLoss loss fstar := by
    intro x x' g hdist
    exact h_meas x x' g hdist
  have h_eq := expected_group_loss_eq_of_zero_dist fstar loss gen μ_X μ_Z h_zero h_meas' h_gen
  simpa [ExpectedGroupLoss, ExpectedGRPORLLoss, loss] using h_eq

/-- GRPO-RL same-argmin form of `grpo_rl_equivalence`.

For fixed old/reference policies, reward, and group generator, if summaries and
originals are oracle-equivalent on support, then the full-document and summary
GRPO-RL losses have the same argmin set among oracle-measurable current
policies. -/
theorem grpo_rl_exact_metric {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_old : GRPOOracleMeasurable pol_old fstar)
    (h_ref : GRPOOracleMeasurable pol_ref fstar)
    (h_reward : OracleMeasurableReward reward fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    SameOracleMeasurableArgminGeneral
      (fun pol : Policy' Strings A =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_X gen)
      (fun pol : Policy' Strings A =>
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_Z gen)
      (fun pol fstar => GRPOOracleMeasurable pol fstar)
      fstar := by
  apply same_oracle_measurable_argmin_general_of_loss_eq
  intro pol h_pol
  have h_meas :
      OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar :=
    grpo_rl_loss_oracle_measurable k pol pol_old pol_ref reward eps beta fstar
      h_pol h_old h_ref h_reward
  exact grpo_rl_equivalence (Y := Y) k fstar pol pol_old pol_ref reward eps beta
    gen μ_X μ_Z h_zero h_meas h_gen

/-- GRPO-RL equivalence via ZR: connecting to tree-based summarization. -/
theorem grpo_rl_equivalence_via_ZR {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] (k : ℕ)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    -- Local laws ensure zero distortion
    (hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    -- Oracle-measurability conditions
    (h_meas : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar)
    -- Boundedness for multi_round_proper
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen := by
  apply grpo_rl_equivalence (Y := Y) (k := k) fstar pol pol_old pol_ref reward eps beta gen
    (PMF.pure x) (ZR g x R T)
  -- Prove zero distortion from local laws (same as grpo_equivalence_via_ZR)
  intro z x' hz hx'
  simp only [PMF.support_pure, Set.mem_singleton_iff] at hx'
  rw [hx']
  have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_proper g T x R fstar hp h1 h2 h3 hR M hM hbound
  unfold D at h_exp_zero
  by_contra h_dist_ne_zero
  have h_dist_pos : 0 < dist (fstar z) (fstar x) :=
    lt_of_le_of_ne dist_nonneg (Ne.symm h_dist_ne_zero)
  have h_term_pos : 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
    apply mul_pos
    · exact ENNReal.toReal_pos hz (PMF.apply_ne_top _ _)
    · exact h_dist_pos
  have h_summable : Summable (fun z => (ZR g x R T z).toReal * dist (fstar z) (fstar x)) :=
    summable_D_of_bounded (ZR g x R T) fstar x M hM (fun z => hbound z x)
  have h_sum_pos : 0 < ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
    calc 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := h_term_pos
         _ ≤ ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
             apply Summable.le_tsum h_summable z
             intro i _
             exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
  unfold Exp at h_exp_zero
  linarith [h_exp_zero]
  -- Other hypotheses
  exact h_meas
  exact h_gen

/-- Bundle for GRPO oracle-measurability conditions. -/
structure GRPOOracleMeasurableBundle {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k) (fstar : Strings → Y) where
  pol_measurable : GRPOOracleMeasurable pol fstar
  ranker_indexed : OracleIndexedRanker ranker fstar
  gen_indexed : OracleIndexedGroupGen gen fstar

/-- GRPO equivalence with bundled conditions. -/
theorem grpo_equivalence_bundle {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (bundle : GRPOOracleMeasurableBundle (Y := Y) pol ranker gen fstar) :
    ExpectedGRPOLoss pol ranker μ_X gen = ExpectedGRPOLoss pol ranker μ_Z gen :=
  grpo_equivalence (Y := Y) fstar pol ranker gen μ_X μ_Z h_zero
    bundle.pol_measurable bundle.ranker_indexed bundle.gen_indexed




end PreferenceLearning

/-!
## Section 4: DPO (Direct Preference Optimization) Instance

This section provides the concrete DPO instantiation of the abstract preference learning
framework. DPO uses the Bradley-Terry-Luce (BTL) preference model with sigmoid loss.

### Key Definitions
- `Policy`: Action distribution conditioned on documents
- `DPOLossPointwise`: The DPO loss function
- `DPO.OracleMeasurable`: Policy depends on document only through oracle value

### Main Theorems
- `dpo_equivalence`: DPO training on summaries equals training on originals when local laws hold
- `dpo_gap`: Quantitative Lipschitz bounds on training gap

This section was originally in PreferenceBounds.lean (formerly DPO.lean) and has been consolidated here.
-/

section DPO

open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

-- Action space for policies
variable {A : Type*}

-- Document and oracle spaces (reusing from earlier sections)
variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
### Bradley-Terry-Luce Preference Model

The BTL axiom states that preference probabilities depend on the document
only through its oracle value f*(x).
-/

/-- Reward function indexed by oracle value: R_y(a) gives the reward for action a at oracle value y -/
def RewardFunction (Y A : Type*) := Y → A → ℝ

/-- Reward-based DPO loss: -log σ(β · (R_y(a_w) - R_y(a_ℓ))). -/
noncomputable def DPOLossReward {Y A : Type*} (R : RewardFunction Y A) (β : ℝ)
    (y : Y) (a_w a_ℓ : A) : ℝ :=
  -Real.log (Real.sigmoid (β * (R y a_w - R y a_ℓ)))

/-- BTL (Bradley-Terry-Luce) Preference Axiom:
    Preference probability depends on document only through f*(x).
    P(a_w ≻ a_ℓ | X = x) = σ(β · (R_{f*(x)}(a_w) - R_{f*(x)}(a_ℓ)))
    where R is the per-oracle reward family and β > 0 is temperature. -/
def BTLPreference {Strings Y A : Type*} [PseudoMetricSpace Y]
    (R : RewardFunction Y A) (β : ℝ) (fstar : Strings → Y)
    (prefProb : Strings → A → A → ℝ) : Prop :=
  ∀ x a_w a_ℓ, prefProb x a_w a_ℓ = Real.sigmoid (β * (R (fstar x) a_w - R (fstar x) a_ℓ))

/-- BTL preferences are oracle-measurable: same oracle value implies same preference probability. -/
lemma btl_oracle_measurable {Strings Y A : Type*} [MetricSpace Y]
    {R : RewardFunction Y A} {β : ℝ} {fstar : Strings → Y}
    {prefProb : Strings → A → A → ℝ}
    (hBTL : BTLPreference R β fstar prefProb)
    {x x' : Strings} (h_eq : dist (fstar x) (fstar x') = 0)
    (a_w a_ℓ : A) :
    prefProb x a_w a_ℓ = prefProb x' a_w a_ℓ := by
  rw [hBTL x a_w a_ℓ, hBTL x' a_w a_ℓ]
  have h_fstar_eq : fstar x = fstar x' := dist_eq_zero.mp h_eq
  rw [h_fstar_eq]

/-!
### Policy Type and Log-Ratio
-/

/-- Policy: maps documents and actions to probabilities.
NOTE: We use 'pol' instead of 'π' to avoid conflict with Real.pi -/
def Policy (Strings A : Type*) := Strings → A → ℝ

/-- Log-Ratio of Policies: log(pol(a|x)/pol_ref(a|x)) -/
noncomputable def LogRatio {Strings A : Type*} (pol pol_ref : Policy Strings A)
    (x : Strings) (a : A) : ℝ :=
  Real.log (pol x a) - Real.log (pol_ref x a)

/-- DPO Logit: Λ(x; a_w, a_ℓ) = β · (log-ratio(a_w) - log-ratio(a_ℓ)) -/
noncomputable def DPOLogit {Strings A : Type*} (pol pol_ref : Policy Strings A) (β : ℝ)
    (x : Strings) (a_w a_ℓ : A) : ℝ :=
  β * (LogRatio pol pol_ref x a_w - LogRatio pol pol_ref x a_ℓ)

/-- Pointwise DPO Loss: L(x; a_w, a_ℓ) = -log σ(Λ(x; a_w, a_ℓ)) -/
noncomputable def DPOLossPointwise {Strings A : Type*} (pol pol_ref : Policy Strings A) (β : ℝ)
    (x : Strings) (a_w a_ℓ : A) : ℝ :=
  -Real.log (Real.sigmoid (DPOLogit pol pol_ref β x a_w a_ℓ))

/-- Oracle-Measurable Policy: policy depends on x only through f*(x). -/
def DPO.OracleMeasurable {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol : Policy Strings A) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → pol x a = pol x' a

/-- Both policies in a DPO comparison are oracle-measurable. -/
structure OracleMeasurablePolicies {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (fstar : Strings → Y) where
  pol_measurable : DPO.OracleMeasurable pol fstar
  ref_measurable : DPO.OracleMeasurable pol_ref fstar

namespace OracleMeasurablePolicies

variable {Strings A Y : Type*} [PseudoMetricSpace Y]
variable {pol pol_ref : Policy Strings A} {fstar : Strings → Y}

/-- Construct from individual proofs -/
def mk' (h_pol : DPO.OracleMeasurable pol fstar) (h_ref : DPO.OracleMeasurable pol_ref fstar) :
    OracleMeasurablePolicies pol pol_ref fstar :=
  ⟨h_pol, h_ref⟩

end OracleMeasurablePolicies

/-- Policy is positive on the support of the pair generator. -/
def PositiveOnSupport {Strings A : Type*} (pol : Policy Strings A)
    (gen : PairGenerator Strings A) : Prop :=
  ∀ x p, p ∈ (gen x).support → 0 < pol x p.1 ∧ 0 < pol x p.2

/-- Policy Argmin: the set of policies that minimize a loss functional. -/
def PolicyArgmin {Strings A : Type*} (loss : Policy Strings A → ℝ) : Set (Policy Strings A) :=
  {pol | ∀ pol', loss pol ≤ loss pol'}

/-- Same Argmin: two loss functions have the same argmin. -/
def SameArgmin {Strings A : Type*} (loss₁ loss₂ : Policy Strings A → ℝ) : Prop :=
  PolicyArgmin loss₁ = PolicyArgmin loss₂

/-- Expected DPO Loss: E_{x,p}[L(x; a_w, a_ℓ)] -/
noncomputable def ExpectedDPOLoss {Strings A : Type*} (pol pol_ref : Policy Strings A) (β : ℝ)
    (μ : PMF Strings) (gen : PairGenerator Strings A) : ℝ :=
  ∑' x, (μ x).toReal * ∑' p, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2

/-- Policy-Lipschitz Log-Ratio: Log-ratio is Lipschitz in oracle space. -/
def PolicyLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (fstar : Strings → Y) (L : ℝ≥0) : Prop :=
  ∀ a x x', |LogRatio pol pol_ref x a - LogRatio pol pol_ref x' a| ≤ (L : ℝ) * dist (fstar x) (fstar x')

/-- Reward-Lipschitz: Reward function is Lipschitz in oracle space. -/
def RewardLipschitz {A Y : Type*} [PseudoMetricSpace Y] (R : Y → A → ℝ) (L : ℝ≥0) : Prop :=
  ∀ a y y', |R y a - R y' a| ≤ (L : ℝ) * dist y y'

/-!
### Bridge Lemma: DPO Loss Satisfies Abstract Lipschitz

This lemma connects the DPO-specific `PolicyLipschitz` to the abstract `LipschitzLoss`.
The proof relies on the fact that DPO loss is -log(σ(logit)), where:
- Logit is linear in log-ratios (hence Lipschitz with constant 2|β|L when log-ratios are L-Lipschitz)
- -log(σ(·)) is 1-Lipschitz (proven in PreferenceBounds.lean via derivative bounds)

This gives overall Lipschitz constant 2|β|L for the DPO loss.

**Note**: The full proof is in PreferenceBounds.lean (`dpo_loss_pointwise_lipschitz`).
This lemma provides the bridge to the abstract framework.
-/

/-- DPO loss satisfies the abstract LipschitzLoss predicate when policy log-ratios are Lipschitz.

This is the bridge lemma connecting DPO-specific infrastructure to the abstract framework.
Combined with `coupling_bound_ineq` from PreferenceBounds.lean, this yields quantitative
gap bounds for DPO training.

The Lipschitz constant L_dpo should satisfy L_dpo ≥ 2|β|L where L is the policy Lipschitz constant.
In practice, L_dpo = 2|β|L (proven in PreferenceBounds.lean via `dpo_loss_pointwise_lipschitz`). -/
lemma dpo_loss_satisfies_lipschitz {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (β : ℝ) (fstar : Strings → Y)
    (L_dpo : ℝ≥0)  -- The overall Lipschitz constant for DPO loss
    (h_bound : ∀ x x' a_w a_ℓ,
      |DPOLossPointwise pol pol_ref β x a_w a_ℓ - DPOLossPointwise pol pol_ref β x' a_w a_ℓ| ≤
      L_dpo * dist (fstar x) (fstar x')) :
    LipschitzLoss (fun x a_w a_ℓ => DPOLossPointwise pol pol_ref β x a_w a_ℓ) fstar L_dpo :=
  fun x x' a_w a_ℓ => h_bound x x' a_w a_ℓ

/-!
### Helper Lemmas for Oracle-Measurable Policies
-/

lemma oracle_meas_eq {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol : Policy Strings A} {fstar : Strings → Y}
    (h_meas : DPO.OracleMeasurable pol fstar) {x x' : Strings}
    (h_dist : dist (fstar x) (fstar x') = 0) (a : A) :
    pol x a = pol x' a := h_meas x x' a h_dist

lemma log_ratio_eq_of_oracle_eq {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {fstar : Strings → Y}
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    {x x' : Strings} (h_dist : dist (fstar x) (fstar x') = 0) (a : A) :
    LogRatio pol pol_ref x a = LogRatio pol pol_ref x' a := by
  unfold LogRatio
  rw [oracle_meas_eq h_meas_pol h_dist a, oracle_meas_eq h_meas_ref h_dist a]

lemma dpo_logit_eq_of_oracle_eq {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {fstar : Strings → Y} {β : ℝ}
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    {x x' : Strings} (h_dist : dist (fstar x) (fstar x') = 0)
    (a_w a_ℓ : A) :
    DPOLogit pol pol_ref β x a_w a_ℓ = DPOLogit pol pol_ref β x' a_w a_ℓ := by
  unfold DPOLogit
  rw [log_ratio_eq_of_oracle_eq h_meas_pol h_meas_ref h_dist a_w,
      log_ratio_eq_of_oracle_eq h_meas_pol h_meas_ref h_dist a_ℓ]

lemma dpo_loss_eq_of_oracle_eq {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {fstar : Strings → Y} {β : ℝ}
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    {x x' : Strings} (h_dist : dist (fstar x) (fstar x') = 0)
    (a_w a_ℓ : A) :
    DPOLossPointwise pol pol_ref β x a_w a_ℓ = DPOLossPointwise pol pol_ref β x' a_w a_ℓ := by
  unfold DPOLossPointwise
  rw [dpo_logit_eq_of_oracle_eq h_meas_pol h_meas_ref h_dist a_w a_ℓ]

/-- Bridge lemma: DPO loss satisfies the general OracleMeasurableLoss predicate. -/
lemma dpo_loss_oracle_measurable {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (β : ℝ) (fstar : Strings → Y)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar) :
    OracleMeasurableLoss (fun x a_w a_ℓ => DPOLossPointwise pol pol_ref β x a_w a_ℓ) fstar :=
  fun x x' a_w a_ℓ hdist => dpo_loss_eq_of_oracle_eq h_meas_pol h_meas_ref hdist a_w a_ℓ

/-!
### Bounded DPO Loss Infrastructure

The DPO loss is bounded when the logit (input to sigmoid) is bounded.
This enables us to use the proven `PMF.summable_coe_real_mul_of_bounded` instead of the axiom.

Key bounds:
- DPOLossPointwise = -log(sigmoid(logit))
- sigmoid(x) ∈ (0, 1) for all x
- When |logit| ≤ L: -log(sigmoid) ≤ log(1 + e^L)
-/

/-- The DPO loss bound given a logit bound L: log(1 + e^L).

When |DPOLogit| ≤ L, we have |DPOLossPointwise| ≤ dpoLossBound L.
This is because:
- DPOLossPointwise = -log(sigmoid(logit))
- sigmoid is monotonically increasing
- sigmoid(-L) = 1/(1 + e^L)
- -log(sigmoid(-L)) = log(1 + e^L)
-/
noncomputable def dpoLossBound (L : ℝ) : ℝ := Real.log (1 + Real.exp L)

lemma dpoLossBound_nonneg (L : ℝ) : 0 ≤ dpoLossBound L := by
  unfold dpoLossBound
  apply Real.log_nonneg
  have h : 0 < Real.exp L := Real.exp_pos L
  linarith

/-- DPO loss is non-negative: -log(sigmoid(x)) ≥ 0 since sigmoid(x) ≤ 1. -/
lemma dpo_loss_nonneg {Strings A : Type*} (pol pol_ref : Policy Strings A) (β : ℝ)
    (x : Strings) (a_w a_ℓ : A) :
    0 ≤ DPOLossPointwise pol pol_ref β x a_w a_ℓ := by
  unfold DPOLossPointwise
  rw [neg_nonneg]
  apply Real.log_nonpos
  · exact le_of_lt (Real.sigmoid_pos _)
  · exact le_of_lt (Real.sigmoid_lt_one _)

/-- DPO loss is bounded when the logit is bounded.

When |DPOLogit| ≤ L, we have DPOLossPointwise ≤ log(1 + e^L).

Proof sketch:
- sigmoid is monotonically increasing
- When logit ≥ -L, sigmoid(logit) ≥ sigmoid(-L) = 1/(1 + e^L)
- So -log(sigmoid) ≤ log(1 + e^L)
-/
lemma dpo_loss_le_of_logit_bounded {Strings A : Type*}
    (pol pol_ref : Policy Strings A) (β : ℝ) (x : Strings) (a_w a_ℓ : A)
    (L : ℝ) (hL : 0 ≤ L)
    (h_logit_bound : |DPOLogit pol pol_ref β x a_w a_ℓ| ≤ L) :
    DPOLossPointwise pol pol_ref β x a_w a_ℓ ≤ dpoLossBound L := by
  unfold DPOLossPointwise dpoLossBound
  -- Need: -log(sigmoid(logit)) ≤ log(1 + e^L)
  -- Equivalently: log(sigmoid(logit)) ≥ -log(1 + e^L) = log(1/(1 + e^L))
  -- Equivalently: sigmoid(logit) ≥ 1/(1 + e^L)
  -- Since sigmoid is monotone and logit ≥ -L, this follows from sigmoid(-L) = 1/(1 + e^L)
  have h_abs := h_logit_bound
  rw [abs_le] at h_abs
  have h_lower := h_abs.1
  have h_exp_pos : 0 < 1 + Real.exp L := by linarith [Real.exp_pos L]
  -- sigmoid(logit) ≥ sigmoid(-L) since sigmoid is monotone increasing
  have h_sigmoid_mono : Real.sigmoid (DPOLogit pol pol_ref β x a_w a_ℓ) ≥ Real.sigmoid (-L) :=
    Real.sigmoid_strictMono.monotone h_lower
  -- sigmoid(-L) = 1/(1 + e^L)
  have h_sigmoid_neg : Real.sigmoid (-L) = 1 / (1 + Real.exp L) := by
    unfold Real.sigmoid
    simp only [neg_neg, one_div]
  rw [h_sigmoid_neg] at h_sigmoid_mono
  -- Now: sigmoid(logit) ≥ 1/(1 + e^L) > 0
  have h_sigmoid_pos : 0 < 1 / (1 + Real.exp L) := by positivity
  -- -log(sigmoid) ≤ log(1 + e^L) ⟺ log(sigmoid) ≥ -log(1 + e^L)
  rw [neg_le]
  -- log(sigmoid) ≥ log(1/(1 + e^L)) since sigmoid ≥ 1/(1 + e^L) and log is monotone
  have h_log_mono := Real.log_le_log h_sigmoid_pos h_sigmoid_mono
  -- log(1/(1 + e^L)) = -log(1 + e^L)
  have h_log_inv : Real.log (1 / (1 + Real.exp L)) = -Real.log (1 + Real.exp L) := by
    rw [one_div, Real.log_inv]
  linarith

/-- Absolute value bound for DPO loss when logit is bounded. -/
lemma abs_dpo_loss_le_of_logit_bounded {Strings A : Type*}
    (pol pol_ref : Policy Strings A) (β : ℝ) (x : Strings) (a_w a_ℓ : A)
    (L : ℝ) (hL : 0 ≤ L)
    (h_logit_bound : |DPOLogit pol pol_ref β x a_w a_ℓ| ≤ L) :
    |DPOLossPointwise pol pol_ref β x a_w a_ℓ| ≤ dpoLossBound L := by
  rw [abs_of_nonneg (dpo_loss_nonneg pol pol_ref β x a_w a_ℓ)]
  exact dpo_loss_le_of_logit_bounded pol pol_ref β x a_w a_ℓ L hL h_logit_bound

/-!
### Summability Helper Lemmas (Bounded Versions)

These lemmas use `PMF.summable_coe_real_mul_of_bounded` instead of the axiom.
-/

/-- Summability for DPO loss with explicit logit bound. -/
lemma summable_dpo_loss_pointwise_bounded {Strings A : Type*}
    (g : PMF (A × A)) (pol pol_ref : Policy Strings A) (β : ℝ) (x : Strings)
    (L : ℝ) (hL : 0 ≤ L)
    (h_logit_bound : ∀ a_w a_ℓ, |DPOLogit pol pol_ref β x a_w a_ℓ| ≤ L) :
    Summable (fun p => (g p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2) :=
  PMF.summable_coe_real_mul_of_bounded g _ (dpoLossBound L) (dpoLossBound_nonneg L)
    (fun p => abs_dpo_loss_le_of_logit_bounded pol pol_ref β x p.1 p.2 L hL (h_logit_bound p.1 p.2))

/-!
### Additional Bounded Summability Lemmas
-/

/-- Bounded summability for expected pair losses. -/
lemma summable_expected_pair_loss_bounded {Strings : Type*}
    (μ : PMF Strings) (E_pair : Strings → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ x, |E_pair x| ≤ M) :
    Summable (fun x => (μ x).toReal * E_pair x) :=
  PMF.summable_coe_real_mul_of_bounded μ E_pair M hM hbound

/-- Bounded summability for pair generator DPO losses. -/
lemma summable_pair_gen_dpo_bounded {Strings A : Type*}
    (gen : PairGenerator Strings A) (pol pol_ref : Policy Strings A) (β : ℝ) (x : Strings)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ p : A × A, |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M) :
    Summable (fun p : A × A => (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2) :=
  PMF.summable_coe_real_mul_of_bounded (gen x) _ M hM hbound

/-- Bounded summability for coupling inner sums. -/
lemma summable_coupling_inner_bounded {α : Type*} (μ_Z : PMF α) (f : α → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ z, |f z| ≤ M) :
    Summable (fun z => (μ_Z z).toReal * f z) :=
  PMF.summable_coe_real_mul_of_bounded μ_Z f M hM hbound

/-- Bounded summability for coupling outer sums. -/
lemma summable_coupling_outer_bounded {α : Type*} (μ_X : PMF α) (inner_sum : α → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ x, |inner_sum x| ≤ M) :
    Summable (fun x => (μ_X x).toReal * inner_sum x) :=
  PMF.summable_coe_real_mul_of_bounded μ_X inner_sum M hM hbound

/-- Bounded summability for distortion (uses ExpectationTheory). -/
lemma summable_distortion_pmf_bounded {Strings Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (g : PMF Strings) (fstar : Strings → Y) (x : Strings)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ z, D fstar z x ≤ M) :
    Summable (fun z => (g z).toReal * D fstar z x) :=
  summable_D_of_bounded g fstar x M hM hbound

-- Deprecated summability helpers removed; use the `_bounded` variants above.

/-!
### Oracle-Measurable Policy Argmin Definitions
-/

def OracleMeasurablePolicyArgmin {Strings A Y : Type*} [PseudoMetricSpace Y]
    (loss : Policy Strings A → ℝ) (fstar : Strings → Y) : Set (Policy Strings A) :=
  {pol | DPO.OracleMeasurable pol fstar ∧ ∀ pol', DPO.OracleMeasurable pol' fstar → loss pol ≤ loss pol'}

def SameOracleMeasurableArgmin {Strings A Y : Type*} [PseudoMetricSpace Y]
    (loss₁ loss₂ : Policy Strings A → ℝ) (fstar : Strings → Y) : Prop :=
  OracleMeasurablePolicyArgmin loss₁ fstar = OracleMeasurablePolicyArgmin loss₂ fstar

/-!
### Core DPO Equivalence Theorems
-/

/-- Key lemma: Zero distortion implies equal expected loss for oracle-measurable policies. -/
lemma expected_loss_eq_of_zero_dist {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β μ_X gen = ExpectedDPOLoss pol pol_ref β μ_Z gen := by
  unfold ExpectedDPOLoss
  exact expected_loss_eq_of_zero_dist_general fstar
    (fun x a_w a_ℓ => DPOLossPointwise pol pol_ref β x a_w a_ℓ)
    gen μ_X μ_Z h_zero
    (dpo_loss_oracle_measurable pol pol_ref β fstar h_meas_pol h_meas_ref)
    h_pair

private lemma same_argmin_of_loss_eq {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ)
    (h_dist_eq : ∀ pol, DPO.OracleMeasurable pol fstar →
        ExpectedDPOLoss pol pol_ref β μ_X gen = ExpectedDPOLoss pol pol_ref β μ_Z gen) :
    SameOracleMeasurableArgmin
      (fun pol => ExpectedDPOLoss pol pol_ref β μ_X gen)
      (fun pol => ExpectedDPOLoss pol pol_ref β μ_Z gen)
      fstar := by
  unfold SameOracleMeasurableArgmin OracleMeasurablePolicyArgmin
  ext pol
  constructor
  · intro ⟨h_meas_pol, h_min⟩
    constructor
    · exact h_meas_pol
    · intro pol' h_meas_pol'
      simp only
      rw [← h_dist_eq pol h_meas_pol, ← h_dist_eq pol' h_meas_pol']
      exact h_min pol' h_meas_pol'
  · intro ⟨h_meas_pol, h_min⟩
    constructor
    · exact h_meas_pol
    · intro pol' h_meas_pol'
      simp only
      rw [h_dist_eq pol h_meas_pol, h_dist_eq pol' h_meas_pol']
      exact h_min pol' h_meas_pol'

theorem dpo_exact_metric {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ)
    (h_oracle_eq : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    SameOracleMeasurableArgmin
      (fun pol => ExpectedDPOLoss pol pol_ref β μ_X gen)
      (fun pol => ExpectedDPOLoss pol pol_ref β μ_Z gen)
      fstar := by
  apply same_argmin_of_loss_eq fstar pol_ref gen μ_X μ_Z β
  intro pol h_meas_pol
  exact expected_loss_eq_of_zero_dist fstar pol pol_ref gen μ_X μ_Z β h_oracle_eq h_meas_pol h_meas_ref h_pair

theorem dpo_exact {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ)
    (_h_oracle_eq : ∀ z x, z ∈ (μ_Z).support → x ∈ (μ_X).support → dist (fstar z) (fstar x) = 0)
    (_h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (_h_pair : OracleIndexedPairGen gen fstar)
    (h_dist_eq : ∀ pol, DPO.OracleMeasurable pol fstar →
        ExpectedDPOLoss pol pol_ref β μ_X gen = ExpectedDPOLoss pol pol_ref β μ_Z gen) :
    SameOracleMeasurableArgmin
      (fun pol => ExpectedDPOLoss pol pol_ref β μ_X gen)
      (fun pol => ExpectedDPOLoss pol pol_ref β μ_Z gen)
      fstar :=
  same_argmin_of_loss_eq fstar pol_ref gen μ_X μ_Z β h_dist_eq

end DPO

/-!
## Section 5: Preference Combinators

A small algebra for composing preference generators and losses. The key idea is
to treat preference collection as a PMF-monad program. This section enables:

1. Oracle-indexed generators closed under `map` and `bind` (nesting)
2. Oracle-measurable losses yield invariant expected loss under zero distortion
3. A monadic DSL (PrefProgram) for building complex preference generators

### Design Note

The predicates `PrefGen`, `PrefLoss`, etc. provide a streamlined interface that
complements the main predicates (`OracleIndexedPairGen`, `OracleMeasurableLoss`).
They are equivalent but optimized for compositional reasoning.
-/

section PreferenceCombinators

open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
### Generic Preference Accumulation Types
-/

/-- A preference generator maps documents to distributions over comparison data. -/
def PrefGen (Strings α : Type*) := Strings → PMF α

/-- A preference loss assigns a real loss to each document-comparison pair. -/
def PrefLoss (Strings α : Type*) := Strings → α → ℝ

/-- Oracle-indexed generator: generator depends on x only through f*(x). -/
def OracleIndexedGenComb {Strings α Y : Type*} [PseudoMetricSpace Y]
    (gen : PrefGen Strings α) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → gen x = gen x'

/-- Oracle-indexed kernel: nested generator depends on x only through f*(x). -/
def OracleIndexedKernel {Strings α β Y : Type*} [PseudoMetricSpace Y]
    (k : Strings → α → PMF β) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → k x a = k x' a

/-- Oracle-measurable preference loss: loss depends on x only through f*(x). -/
def OracleMeasurablePrefLoss {Strings α Y : Type*} [PseudoMetricSpace Y]
    (loss : PrefLoss Strings α) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → loss x a = loss x' a

/-- Expected preference loss under a document distribution and generator. -/
noncomputable def ExpectedPrefLoss {Strings α : Type*} [Monoid Strings]
    (loss : PrefLoss Strings α) (μ : PMF Strings) (gen : PrefGen Strings α) : ℝ :=
  ∑' x, (μ x).toReal * ∑' a, (gen x a).toReal * loss x a

/-!
### Nesting: Map/Bind for Generators

Oracle-indexed generators are closed under functorial and monadic operations.
-/

/-- Map a function over a preference generator. -/
def PrefGen.map {Strings α β : Type*} (f : α → β) (gen : PrefGen Strings α) : PrefGen Strings β :=
  fun x => PMF.map f (gen x)

/-- Bind (monadic composition) for preference generators. -/
def PrefGen.bind {Strings α β : Type*} (gen : PrefGen Strings α)
    (k : Strings → α → PMF β) : PrefGen Strings β :=
  fun x => (gen x).bind (k x)

/-- Map preserves oracle-indexedness. -/
lemma oracle_indexed_map {Strings α β Y : Type*} [PseudoMetricSpace Y]
    (f : α → β) (gen : PrefGen Strings α) (fstar : Strings → Y)
    (h_gen : OracleIndexedGenComb gen fstar) :
    OracleIndexedGenComb (PrefGen.map f gen) fstar := by
  intro x x' hdist
  ext b
  simp [PrefGen.map, PMF.map_apply, h_gen x x' hdist]

/-- Bind preserves oracle-indexedness when both generator and kernel are oracle-indexed. -/
lemma oracle_indexed_bind {Strings α β Y : Type*} [PseudoMetricSpace Y]
    (gen : PrefGen Strings α) (k : Strings → α → PMF β) (fstar : Strings → Y)
    (h_gen : OracleIndexedGenComb gen fstar)
    (h_k : OracleIndexedKernel k fstar) :
    OracleIndexedGenComb (PrefGen.bind gen k) fstar := by
  intro x x' hdist
  have h_k' : ∀ a, k x a = k x' a := by
    intro a
    exact h_k x x' a hdist
  ext b
  simp [PrefGen.bind, PMF.bind_apply, h_gen x x' hdist, h_k']

/-!
### Generic Equivalence Theorem

The core theorem: zero distortion implies equal expected loss.
-/

/-- Zero distortion implies equal expected preference loss for oracle-measurable losses
and oracle-indexed generators. -/
lemma expected_pref_loss_eq_of_zero_dist {Strings α Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : PrefLoss Strings α)
    (gen : PrefGen Strings α)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_gen : OracleIndexedGenComb gen fstar) :
    ExpectedPrefLoss loss μ_X gen = ExpectedPrefLoss loss μ_Z gen := by
  obtain ⟨x₀, hx₀⟩ := μ_X.support_nonempty
  let L₀ := fun a : α => (gen x₀ a).toReal * loss x₀ a
  have h_X_eq : ∀ x, x ∈ μ_X.support →
      ∑' a, (gen x a).toReal * loss x a = ∑' a, L₀ a := by
    intro x hx
    apply tsum_congr
    intro a
    obtain ⟨z₀, hz₀⟩ := μ_Z.support_nonempty
    have hd1 : dist (fstar z₀) (fstar x₀) = 0 := h_zero z₀ x₀ hz₀ hx₀
    have hd2 : dist (fstar z₀) (fstar x) = 0 := h_zero z₀ x hz₀ hx
    have hd : dist (fstar x) (fstar x₀) = 0 := by
      have h_tri := dist_triangle (fstar x) (fstar z₀) (fstar x₀)
      have h1 : dist (fstar x) (fstar z₀) = 0 := by rw [dist_comm]; exact hd2
      have h2 : dist (fstar z₀) (fstar x₀) = 0 := hd1
      rw [h1, h2] at h_tri
      simp only [zero_add] at h_tri
      linarith [dist_nonneg (α := Y) (x := fstar x) (y := fstar x₀)]
    have h_gen_eq : gen x = gen x₀ := h_gen x x₀ hd
    have h_loss_eq : loss x a = loss x₀ a := h_meas x x₀ a hd
    rw [h_gen_eq, h_loss_eq]
  have h_Z_eq : ∀ z, z ∈ μ_Z.support →
      ∑' a, (gen z a).toReal * loss z a = ∑' a, L₀ a := by
    intro z hz
    apply tsum_congr
    intro a
    have hd : dist (fstar z) (fstar x₀) = 0 := h_zero z x₀ hz hx₀
    have h_gen_eq : gen z = gen x₀ := h_gen z x₀ hd
    have h_loss_eq : loss z a = loss x₀ a := h_meas z x₀ a hd
    rw [h_gen_eq, h_loss_eq]
  have h_X : ∑' x, (μ_X x).toReal * ∑' a, (gen x a).toReal * loss x a =
             ∑' a, L₀ a := by
    have h_eq : ∀ x, (μ_X x).toReal * ∑' a, (gen x a).toReal * loss x a =
                (μ_X x).toReal * ∑' a, L₀ a := by
      intro x
      by_cases hx : x ∈ μ_X.support
      · rw [h_X_eq x hx]
      · have h_zero_app : μ_X x = 0 := (μ_X.apply_eq_zero_iff x).mpr hx
        simp only [h_zero_app, ENNReal.toReal_zero, zero_mul]
    simp_rw [h_eq]
    have h_factor : (fun x => (μ_X x).toReal * ∑' a, L₀ a) =
                    (fun x => (∑' a, L₀ a) * (μ_X x).toReal) := by ext x; ring
    rw [h_factor, tsum_mul_left, PMF.toReal_tsum_coe, mul_one]
  have h_Z : ∑' z, (μ_Z z).toReal * ∑' a, (gen z a).toReal * loss z a =
             ∑' a, L₀ a := by
    have h_eq : ∀ z, (μ_Z z).toReal * ∑' a, (gen z a).toReal * loss z a =
                (μ_Z z).toReal * ∑' a, L₀ a := by
      intro z
      by_cases hz : z ∈ μ_Z.support
      · rw [h_Z_eq z hz]
      · have h_zero_app : μ_Z z = 0 := (μ_Z.apply_eq_zero_iff z).mpr hz
        simp only [h_zero_app, ENNReal.toReal_zero, zero_mul]
    simp_rw [h_eq]
    have h_factor : (fun z => (μ_Z z).toReal * ∑' a, L₀ a) =
                    (fun z => (∑' a, L₀ a) * (μ_Z z).toReal) := by ext z; ring
    rw [h_factor, tsum_mul_left, PMF.toReal_tsum_coe, mul_one]
  unfold ExpectedPrefLoss
  rw [h_X, h_Z]

/-!
### PrefProgram: Nested Preference Programs

A monadic DSL for building complex preference generators from simple ones.
This enables compositional construction of preference collection schemes.
-/

variable {α β : Type*}

/-- An inductive type representing preference programs that can sample from
    oracle-indexed distributions and compose them. -/
inductive PrefProgram (Strings : Type*) (α : Type*) : Type _ where
  | pure : α → PrefProgram Strings α
  | sample : {β : Type*} → (Strings → PMF β) → (β → PrefProgram Strings α) → PrefProgram Strings α

/-- Run a preference program to produce a preference generator. -/
def PrefProgram.run {Strings α : Type*} : PrefProgram Strings α → Strings → PMF α
  | PrefProgram.pure a => fun _ => PMF.pure a
  | PrefProgram.sample gen k => fun x => (gen x).bind (fun a => PrefProgram.run (k a) x)

/-- Oracle-indexedness predicate for preference programs. -/
def OracleIndexedProgram {Strings α Y : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) : PrefProgram Strings α → Prop
  | PrefProgram.pure _ => True
  | PrefProgram.sample gen k => OracleIndexedGenComb gen fstar ∧ ∀ a, OracleIndexedProgram fstar (k a)

/-- Running an oracle-indexed program yields an oracle-indexed generator. -/
lemma oracle_indexed_run {Strings α Y : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (prog : PrefProgram Strings α)
    (h_prog : OracleIndexedProgram fstar prog) :
    OracleIndexedGenComb (PrefProgram.run prog) fstar := by
  induction prog with
  | pure a =>
      intro x x' hdist
      rfl
  | sample gen k ih =>
      rcases h_prog with ⟨h_gen, h_k⟩
      have h_kernel : OracleIndexedKernel (fun x a => PrefProgram.run (k a) x) fstar := by
        intro x x' a hdist
        have h_run := ih a (h_k a)
        exact h_run x x' hdist
      exact oracle_indexed_bind (gen := gen) (k := fun x a => PrefProgram.run (k a) x)
        (fstar := fstar) h_gen h_kernel

/-- Expected loss for a preference program. -/
noncomputable def ExpectedPrefLossProg {Strings α : Type*} [Monoid Strings]
    (loss : PrefLoss Strings α) (μ : PMF Strings) (prog : PrefProgram Strings α) : ℝ :=
  ExpectedPrefLoss loss μ (PrefProgram.run prog)

/-- Zero distortion implies equal expected loss for preference programs. -/
lemma expected_pref_loss_prog_eq_of_zero_dist {Strings α Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (loss : PrefLoss Strings α)
    (μ_X μ_Z : PMF Strings)
    (prog : PrefProgram Strings α)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_prog : OracleIndexedProgram fstar prog) :
    ExpectedPrefLossProg loss μ_X prog = ExpectedPrefLossProg loss μ_Z prog := by
  have h_gen : OracleIndexedGenComb (PrefProgram.run prog) fstar :=
    oracle_indexed_run fstar prog h_prog
  have h_eq := expected_pref_loss_eq_of_zero_dist (fstar := fstar)
    (loss := loss) (gen := PrefProgram.run prog) (μ_X := μ_X) (μ_Z := μ_Z)
    h_zero h_meas h_gen
  simpa [ExpectedPrefLossProg] using h_eq

/-!
### DPO, PPO, and GRPO as Instances

These lemmas show that DPO, GRPO-PL, and GRPO-RL losses can be viewed through
the combinator framework, deriving equivalence theorems as corollaries.
-/

variable {A : Type*}

/-- Convert a binary loss to a pair loss. -/
def PairLoss (loss : Strings → A → A → ℝ) : PrefLoss Strings (A × A) :=
  fun x p => loss x p.1 p.2

/-- DPO expected loss matches the generic expected preference loss formulation. -/
lemma expected_pref_loss_dpo_eq {Strings A : Type*} [Monoid Strings]
    (pol pol_ref : Policy Strings A) (β : ℝ)
    (μ : PMF Strings) (gen : PairGenerator Strings A) :
    ExpectedPrefLoss (PairLoss (fun x a_w a_ℓ =>
      DPOLossPointwise pol pol_ref β x a_w a_ℓ)) μ gen =
    ExpectedDPOLoss pol pol_ref β μ gen := by
  rfl

/-- DPO equivalence derived via preference combinators. -/
lemma dpo_equivalence_via_pref {Strings A Y : Type*}
    [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A) (β : ℝ)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β μ_X gen = ExpectedDPOLoss pol pol_ref β μ_Z gen := by
  have h_loss : OracleMeasurablePrefLoss
      (PairLoss (fun x a_w a_ℓ => DPOLossPointwise pol pol_ref β x a_w a_ℓ)) fstar := by
    intro x x' p hdist
    exact dpo_loss_eq_of_oracle_eq h_meas_pol h_meas_ref hdist p.1 p.2
  have h_gen' : OracleIndexedGenComb gen fstar := fun x x' hdist => h_gen x x' hdist
  have h_eq := expected_pref_loss_eq_of_zero_dist (fstar := fstar)
    (loss := PairLoss (fun x a_w a_ℓ => DPOLossPointwise pol pol_ref β x a_w a_ℓ))
    (gen := gen) (μ_X := μ_X) (μ_Z := μ_Z) h_zero h_loss h_gen'
  simpa [ExpectedPrefLoss, PairLoss, ExpectedDPOLoss] using h_eq

/-- GRPO Plackett-Luce equivalence derived via preference combinators. -/
lemma grpo_equivalence_via_pref {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPOLoss pol ranker μ_X gen = ExpectedGRPOLoss pol ranker μ_Z gen := by
  have h_meas : OracleMeasurablePrefLoss
      (fun x g => GRPOLossPointwise pol x g (ranker x g)) fstar := by
    intro x x' g hdist
    have h := grpo_loss_oracle_measurable pol ranker fstar h_pol h_ranker
    exact h x x' g hdist
  have h_gen' : OracleIndexedGenComb gen fstar := fun x x' hdist => h_gen x x' hdist
  have h_eq := expected_pref_loss_eq_of_zero_dist (fstar := fstar)
    (loss := fun x g => GRPOLossPointwise pol x g (ranker x g))
    (gen := gen) (μ_X := μ_X) (μ_Z := μ_Z) h_zero h_meas h_gen'
  simpa [ExpectedPrefLoss, ExpectedGRPOLoss] using h_eq

/-- GRPO-RL (DeepSeek-R1 style) equivalence derived via preference combinators. -/
lemma grpo_rl_equivalence_via_pref {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y] {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (h_zero : ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support → dist (fstar z) (fstar x) = 0)
    (h_pol : GRPOOracleMeasurable pol fstar)
    (h_old : GRPOOracleMeasurable pol_old fstar)
    (h_ref : GRPOOracleMeasurable pol_ref fstar)
    (h_reward : OracleMeasurableReward reward fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_X gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_Z gen := by
  have h_loss : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar :=
    grpo_rl_loss_oracle_measurable k pol pol_old pol_ref reward eps beta fstar
      h_pol h_old h_ref h_reward
  have h_meas : OracleMeasurablePrefLoss
      (fun x (g : Fin k → A) => GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x g) fstar := by
    intro x x' g hdist
    exact h_loss x x' g hdist
  have h_gen' : OracleIndexedGenComb gen fstar := fun x x' hdist => h_gen x x' hdist
  have h_eq := expected_pref_loss_eq_of_zero_dist (fstar := fstar)
    (loss := fun x (g : Fin k → A) => GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x g)
    (gen := gen) (μ_X := μ_X) (μ_Z := μ_Z) h_zero h_meas h_gen'
  simpa [ExpectedPrefLoss, ExpectedGRPORLLoss] using h_eq

end PreferenceCombinators

end
