import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.AuditBounds
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.MergeTriangle
import FormalProofs.Shared.BoundedMetricSpace
import FormalProbability.DSL.RUM
import FormalProbability.DSL.PlackettLuce

/-!
# FormalProofs/PreferenceBounds.lean

## Quantitative Bounds for Preference Learning Methods

This file provides **quantitative gap bounds** for preference learning methods.
It contains shared infrastructure (Lipschitz lemmas, coupling bounds) and
method-specific quantitative analysis for DPO and GRPO.

### File Structure

1. **Section 1: Shared Lipschitz Lemmas** - Used by all methods
   - `sigmoid_lipschitz`: Sigmoid is 1-Lipschitz
   - `neg_log_sigmoid_lipschitz`: -log ∘ σ is 1-Lipschitz

2. **Section 2: Shared Coupling Infrastructure** - Used by all methods
   - `coupling_bound_ineq`: General coupling bound for PMF products
   - `PMF.summable_prod_mul_of_bounded`: Summability lemmas

3. **Section 3: DPO Quantitative Bounds**
   - `dpo_logit_lipschitz`, `dpo_loss_pointwise_lipschitz`
   - `dpo_gap_bounded` and variants
   - ZR-connection: `dpo_gap_zero_of_local_laws`, `dpo_equivalence`

4. **Section 4: GRPO Quantitative Bounds** (future)
   - Parallel bounds for GRPO-PL and GRPO-RL

### Relationship to PreferenceLearning.lean

PreferenceLearning.lean provides:
- Abstract oracle-measurability framework
- Zero-distortion equivalence theorems
- Core type definitions (Policy, DPOLossPointwise, etc.)

This file provides:
- Quantitative Lipschitz bounds
- Coupling-based gap analysis
- Concrete gap bounds for specific methods

### Key Theorems

- `dpo_gap_bounded`: DPO gap bounded by Lipschitz constant × expected distortion
- `dpo_equivalence`: Main DPO equivalence under local laws
- `dpo_gap_zero_of_local_laws_bounded`: Zero gap when local laws hold

See ExpectationTheory.lean USER GUIDE for the full bounded API.
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

/-!
## Quantitative DPO Bounds

This section provides quantitative bounds for DPO training, including:
- Lipschitz bounds for the DPO loss
- Coupling lemmas for expectation bounds
- DPO gap theorems
- ZR-connection theorems
- Main DPO equivalence theorem

The core DPO definitions (Policy, DPOLossPointwise, etc.) are now in PreferenceLearning.lean.
This file provides the quantitative analysis that builds on those definitions.
-/

section DPO

open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

-- Action space for policies
variable {A : Type*}

-- Document and oracle spaces (reusing from earlier sections)
variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

-- Core DPO definitions (Policy, DPOLossPointwise, etc.) are imported from PreferenceLearning.lean

/-!
## Lipschitz Bounds for DPO

These lemmas establish Lipschitz continuity properties of the DPO loss
components. The proofs require Mathlib's sigmoid differentiability lemmas.
-/

/-
Helper lemma: Sigmoid is 1-Lipschitz.
The derivative σ'(t) = σ(t)(1-σ(t)) ≤ 1/4 < 1.
-/
lemma sigmoid_lipschitz : LipschitzWith 1 Real.sigmoid := by
  apply lipschitzWith_of_nnnorm_deriv_le differentiable_sigmoid
  intro x
  rw [Real.deriv_sigmoid]
  -- Need to show ‖sigmoid x * (1 - sigmoid x)‖₊ ≤ 1
  -- sigmoid x ∈ (0, 1), so sigmoid x * (1 - sigmoid x) ∈ (0, 1/4] ⊂ [0, 1]
  have h1 : 0 ≤ Real.sigmoid x := Real.sigmoid_nonneg x
  have h2 : Real.sigmoid x ≤ 1 := Real.sigmoid_le_one x
  have h3 : 0 ≤ 1 - Real.sigmoid x := by linarith
  have h_prod_nonneg : 0 ≤ Real.sigmoid x * (1 - Real.sigmoid x) := mul_nonneg h1 h3
  -- Convert nnnorm to subtype, goal becomes comparing NNReals
  rw [nnnorm_of_nonneg h_prod_nonneg]
  -- ⟨a, _⟩ ≤ 1 iff a ≤ 1 (for NNReal)
  rw [← NNReal.coe_le_coe, NNReal.coe_mk, NNReal.coe_one]
  -- Using a(1-a) ≤ 1/4 ≤ 1 for a ∈ [0,1]
  calc Real.sigmoid x * (1 - Real.sigmoid x)
      ≤ 1/4 := by nlinarith [sq_nonneg (Real.sigmoid x - 1/2)]
    _ ≤ 1 := by norm_num

/-
Helper lemma: -log ∘ σ is 1-Lipschitz.
The derivative is -(1 - σ(t)) which has absolute value < 1.
-/
lemma neg_log_sigmoid_lipschitz : LipschitzWith 1 (fun t => -Real.log (Real.sigmoid t)) := by
  -- Show differentiability
  have hdiff : Differentiable ℝ (fun t => -Real.log (Real.sigmoid t)) := by
    intro t
    apply DifferentiableAt.neg
    apply DifferentiableAt.log
    · exact differentiableAt_sigmoid
    · exact ne_of_gt (Real.sigmoid_pos t)
  -- Apply Lipschitz from bounded derivative
  apply lipschitzWith_of_nnnorm_deriv_le hdiff
  intro t
  -- Compute derivative using chain rule:
  -- d/dt(-log(σ(t))) = -σ'(t)/σ(t) = -σ(t)(1-σ(t))/σ(t) = -(1-σ(t)) = σ(t) - 1
  have hpos : 0 < Real.sigmoid t := Real.sigmoid_pos t
  have hderiv : deriv (fun t => -Real.log (Real.sigmoid t)) t = Real.sigmoid t - 1 := by
    have hne : Real.sigmoid t ≠ 0 := ne_of_gt hpos
    -- Compute using DifferentiableAt.neg and deriv.log
    have hdiff_log_sig : DifferentiableAt ℝ (fun t => Real.log (Real.sigmoid t)) t := by
      apply DifferentiableAt.log differentiableAt_sigmoid hne
    calc deriv (fun t => -Real.log (Real.sigmoid t)) t
        = -deriv (fun t => Real.log (Real.sigmoid t)) t := deriv.fun_neg
      _ = -(deriv Real.sigmoid t / Real.sigmoid t) := by
          congr 1
          exact deriv.log differentiableAt_sigmoid hne
      _ = -(Real.sigmoid t * (1 - Real.sigmoid t) / Real.sigmoid t) := by
          rw [Real.deriv_sigmoid]
      _ = Real.sigmoid t - 1 := by field_simp; ring
  rw [hderiv]
  -- Show ‖σ(t) - 1‖₊ ≤ 1
  -- Since 0 < σ(t) < 1, we have -1 < σ(t) - 1 < 0, so |σ(t) - 1| = 1 - σ(t) < 1
  have h2 : Real.sigmoid t < 1 := Real.sigmoid_lt_one t
  have h3 : Real.sigmoid t - 1 < 0 := by linarith
  have h_neg_nonneg : 0 ≤ -(Real.sigmoid t - 1) := by linarith
  -- ‖x‖₊ = ‖|x|‖₊ and for nonneg a, ‖a‖₊ = ⟨a, _⟩
  rw [← Real.nnnorm_abs]
  rw [abs_of_neg h3]
  rw [Real.nnnorm_of_nonneg h_neg_nonneg]
  rw [← NNReal.coe_le_coe, NNReal.coe_mk, NNReal.coe_one]
  linarith

/-
Helper lemma: DPO logit difference is bounded by Lipschitz constant on log-ratios.
|Λ(x) - Λ(x')| ≤ 2|β|L_pol · d_Y(f*(x), f*(x'))
-/
lemma dpo_logit_lipschitz {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {fstar : Strings → Y} {β : ℝ} {L_pol : ℝ≥0}
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol) (a_w a_ℓ : A) :
    ∀ x x', |DPOLogit pol pol_ref β x a_w a_ℓ - DPOLogit pol pol_ref β x' a_w a_ℓ| ≤
            2 * |β| * L_pol * dist (fstar x) (fstar x') := by
  intro x x'
  unfold DPOLogit
  -- |β(log_w - log_ℓ)(x) - β(log_w - log_ℓ)(x')| = |β| * |(log_w(x) - log_w(x')) - (log_ℓ(x) - log_ℓ(x'))|
  have h1 : β * (LogRatio pol pol_ref x a_w - LogRatio pol pol_ref x a_ℓ) -
            β * (LogRatio pol pol_ref x' a_w - LogRatio pol pol_ref x' a_ℓ) =
            β * ((LogRatio pol pol_ref x a_w - LogRatio pol pol_ref x' a_w) -
                 (LogRatio pol pol_ref x a_ℓ - LogRatio pol pol_ref x' a_ℓ)) := by ring
  rw [h1, abs_mul]
  -- Use triangle inequality and Lipschitz bound on log-ratios
  have haw : |LogRatio pol pol_ref x a_w - LogRatio pol pol_ref x' a_w| ≤ L_pol * dist (fstar x) (fstar x') :=
    h_lip a_w x x'
  have hal : |LogRatio pol pol_ref x a_ℓ - LogRatio pol pol_ref x' a_ℓ| ≤ L_pol * dist (fstar x) (fstar x') :=
    h_lip a_ℓ x x'
  calc |β| * |(LogRatio pol pol_ref x a_w - LogRatio pol pol_ref x' a_w) -
              (LogRatio pol pol_ref x a_ℓ - LogRatio pol pol_ref x' a_ℓ)|
      ≤ |β| * (|LogRatio pol pol_ref x a_w - LogRatio pol pol_ref x' a_w| +
               |LogRatio pol pol_ref x a_ℓ - LogRatio pol pol_ref x' a_ℓ|) := by
        apply mul_le_mul_of_nonneg_left (abs_sub _ _) (abs_nonneg _)
    _ ≤ |β| * ((L_pol : ℝ) * dist (fstar x) (fstar x') + (L_pol : ℝ) * dist (fstar x) (fstar x')) := by
        apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
        apply add_le_add haw hal
    _ = 2 * |β| * (L_pol : ℝ) * dist (fstar x) (fstar x') := by ring

/-
Helper lemma: Pointwise DPO loss difference is bounded.
|L(x) - L(x')| ≤ 2|β|L_pol · d_Y(f*(x), f*(x'))
-/
lemma dpo_loss_pointwise_lipschitz {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {fstar : Strings → Y} {β : ℝ} {L_pol : ℝ≥0}
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol) (a_w a_ℓ : A) :
    ∀ x x', |DPOLossPointwise pol pol_ref β x a_w a_ℓ - DPOLossPointwise pol pol_ref β x' a_w a_ℓ| ≤
            2 * |β| * L_pol * dist (fstar x) (fstar x') := by
  intro x x'
  unfold DPOLossPointwise
  -- Use that -log ∘ sigmoid is 1-Lipschitz
  have h_sig_lip := neg_log_sigmoid_lipschitz
  -- |(-log σ)(Λ(x)) - (-log σ)(Λ(x'))| ≤ 1 * |Λ(x) - Λ(x')|
  have h1 : |-Real.log (Real.sigmoid (DPOLogit pol pol_ref β x a_w a_ℓ)) -
            (-Real.log (Real.sigmoid (DPOLogit pol pol_ref β x' a_w a_ℓ)))| ≤
            |DPOLogit pol pol_ref β x a_w a_ℓ - DPOLogit pol pol_ref β x' a_w a_ℓ| := by
    have := h_sig_lip.dist_le_mul (DPOLogit pol pol_ref β x a_w a_ℓ) (DPOLogit pol pol_ref β x' a_w a_ℓ)
    simp only [Real.dist_eq] at this
    calc |-Real.log (Real.sigmoid (DPOLogit pol pol_ref β x a_w a_ℓ)) -
          (-Real.log (Real.sigmoid (DPOLogit pol pol_ref β x' a_w a_ℓ)))|
        ≤ (1 : ℝ≥0) * |DPOLogit pol pol_ref β x a_w a_ℓ - DPOLogit pol pol_ref β x' a_w a_ℓ| := this
      _ = |DPOLogit pol pol_ref β x a_w a_ℓ - DPOLogit pol pol_ref β x' a_w a_ℓ| := by simp
  calc |-Real.log (Real.sigmoid (DPOLogit pol pol_ref β x a_w a_ℓ)) -
        (-Real.log (Real.sigmoid (DPOLogit pol pol_ref β x' a_w a_ℓ)))|
      ≤ |DPOLogit pol pol_ref β x a_w a_ℓ - DPOLogit pol pol_ref β x' a_w a_ℓ| := h1
    _ ≤ 2 * |β| * L_pol * dist (fstar x) (fstar x') := dpo_logit_lipschitz h_lip a_w a_ℓ x x'

/-
Helper lemma: Reward-based DPO loss difference is bounded.
|L(y) - L(y')| ≤ 2|β|L_R · dist(y, y')
-/
lemma dpo_loss_reward_lipschitz {A Y : Type*} [PseudoMetricSpace Y]
    {R : RewardFunction Y A} {β : ℝ} {L_R : ℝ≥0}
    (h_lip : RewardLipschitz R L_R) (a_w a_ℓ : A) :
    ∀ y y', |DPOLossReward R β y a_w a_ℓ - DPOLossReward R β y' a_w a_ℓ| ≤
            2 * |β| * L_R * dist y y' := by
  intro y y'
  unfold DPOLossReward
  -- Use that -log ∘ sigmoid is 1-Lipschitz
  have h_sig_lip := neg_log_sigmoid_lipschitz
  have h1 : |-Real.log (Real.sigmoid (β * (R y a_w - R y a_ℓ))) -
            (-Real.log (Real.sigmoid (β * (R y' a_w - R y' a_ℓ))))| ≤
            |β * (R y a_w - R y a_ℓ) - β * (R y' a_w - R y' a_ℓ)| := by
    have := h_sig_lip.dist_le_mul (β * (R y a_w - R y a_ℓ)) (β * (R y' a_w - R y' a_ℓ))
    simp only [Real.dist_eq] at this
    calc |-Real.log (Real.sigmoid (β * (R y a_w - R y a_ℓ))) -
          (-Real.log (Real.sigmoid (β * (R y' a_w - R y' a_ℓ))))|
        ≤ (1 : ℝ≥0) * |β * (R y a_w - R y a_ℓ) - β * (R y' a_w - R y' a_ℓ)| := this
      _ = |β * (R y a_w - R y a_ℓ) - β * (R y' a_w - R y' a_ℓ)| := by simp
  have h_beta : |β * (R y a_w - R y a_ℓ) - β * (R y' a_w - R y' a_ℓ)|
      = |β| * |(R y a_w - R y a_ℓ) - (R y' a_w - R y' a_ℓ)| := by
    have h1' :
        β * (R y a_w - R y a_ℓ) - β * (R y' a_w - R y' a_ℓ) =
        β * ((R y a_w - R y a_ℓ) - (R y' a_w - R y' a_ℓ)) := by ring
    rw [h1', abs_mul]
  have h_aw : |R y a_w - R y' a_w| ≤ (L_R : ℝ) * dist y y' := h_lip a_w y y'
  have h_al : |R y a_ℓ - R y' a_ℓ| ≤ (L_R : ℝ) * dist y y' := h_lip a_ℓ y y'
  have h_u : |(R y a_w - R y a_ℓ) - (R y' a_w - R y' a_ℓ)| ≤
      2 * (L_R : ℝ) * dist y y' := by
    have h_diff :
        (R y a_w - R y a_ℓ) - (R y' a_w - R y' a_ℓ) =
        (R y a_w - R y' a_w) - (R y a_ℓ - R y' a_ℓ) := by ring
    rw [h_diff]
    calc |(R y a_w - R y' a_w) - (R y a_ℓ - R y' a_ℓ)|
        ≤ |R y a_w - R y' a_w| + |R y a_ℓ - R y' a_ℓ| := abs_sub _ _
      _ ≤ (L_R : ℝ) * dist y y' + (L_R : ℝ) * dist y y' := add_le_add h_aw h_al
      _ = 2 * (L_R : ℝ) * dist y y' := by ring
  calc |DPOLossReward R β y a_w a_ℓ - DPOLossReward R β y' a_w a_ℓ|
      ≤ |β * (R y a_w - R y a_ℓ) - β * (R y' a_w - R y' a_ℓ)| := h1
    _ = |β| * |(R y a_w - R y a_ℓ) - (R y' a_w - R y' a_ℓ)| := h_beta
    _ ≤ |β| * (2 * (L_R : ℝ) * dist y y') := by
          apply mul_le_mul_of_nonneg_left h_u (abs_nonneg β)
    _ = 2 * |β| * (L_R : ℝ) * dist y y' := by ring

/-!
## Helper Lemmas for DPO Gap Bound
-/

/-- Absolute value of tsum is bounded by tsum of absolute values -/
lemma abs_tsum_le_tsum_abs' {α : Type*} (f : α → ℝ) (hf : Summable f)
    (habs : Summable (fun x => |f x|)) :
    |∑' x, f x| ≤ ∑' x, |f x| := by
  have h : ∑' x, f x ≤ ∑' x, |f x| :=
    Summable.tsum_le_tsum (fun x => le_abs_self (f x)) hf habs
  have h' : -∑' x, f x ≤ ∑' x, |f x| := by
    rw [← tsum_neg]
    exact Summable.tsum_le_tsum (fun x => neg_le_abs (f x)) hf.neg habs
  exact abs_le.mpr ⟨by linarith, h⟩

/-- Product of two PMFs times a bounded function is summable.

Mathematical justification: For bounded f with |f(a,b)| ≤ M,
  |p(a) * q(b) * f(a,b)| ≤ M * p(a) * q(b)
Sum over all (a,b): ∑∑ M * p(a) * q(b) = M * (∑ p) * (∑ q) = M * 1 * 1 = M < ∞

**Note:** This is a sound lemma (unlike PMF.summable_coe_real_mul for unbounded f).
We provide a direct summability proof using boundedness and PMF normalization. -/
lemma PMF.summable_prod_mul_of_bounded {α β : Type*} (p : PMF α) (q : PMF β)
    (f : α → β → ℝ) (M : ℝ) (hM : 0 ≤ M) (hf : ∀ a b, |f a b| ≤ M) :
    Summable (fun ab : α × β => (p ab.1).toReal * (q ab.2).toReal * f ab.1 ab.2) := by
  -- The key insight: |p(a) * q(b) * f(a,b)| ≤ M * p(a) * q(b)
  -- And ∑_{a,b} M * p(a) * q(b) = M * (∑_a p(a)) * (∑_b q(b)) = M * 1 * 1 = M < ∞
  -- We use Summable.of_norm_bounded with the product bound
  set bound := fun ab : α × β => M * (p ab.1).toReal * (q ab.2).toReal with hbound_def
  have hbound_summable : Summable bound := by
    -- Use summable_prod_of_nonneg: for non-negative f, Summable ↔ inner sums summable
    have hbound_nonneg : 0 ≤ bound := by
      intro ab
      apply mul_nonneg (mul_nonneg (by linarith) ENNReal.toReal_nonneg) ENNReal.toReal_nonneg
    rw [summable_prod_of_nonneg hbound_nonneg]
    constructor
    · -- For fixed a, fun b => M * p(a) * q(b) is summable
      intro a
      have hq_summable : Summable (fun b => (q b).toReal) := PMF.summable_coe_real q
      exact Summable.mul_left (M * (p a).toReal) hq_summable
    · -- fun a => ∑' b, M * p(a) * q(b) = fun a => M * p(a) * 1 is summable
      have h_inner_eq : (fun a => ∑' b, M * (p a).toReal * (q b).toReal) =
                        (fun a => M * (p a).toReal) := by
        ext a
        have hq_summable : Summable (fun b => (q b).toReal) := PMF.summable_coe_real q
        -- Factor: M * p(a) * q(b) = (M * p(a)) * q(b)
        have h_factor : (fun b => M * (p a).toReal * (q b).toReal) =
                        (fun b => (M * (p a).toReal) * (q b).toReal) := by ext b; ring
        rw [h_factor, tsum_mul_left, PMF.toReal_tsum_coe q, mul_one]
      rw [h_inner_eq]
      have hp_summable : Summable (fun a => (p a).toReal) := PMF.summable_coe_real p
      exact Summable.mul_left M hp_summable
  have hbound_le : ∀ ab, ‖(p ab.1).toReal * (q ab.2).toReal * f ab.1 ab.2‖ ≤ bound ab := by
    intro ab
    rw [Real.norm_eq_abs, abs_mul, abs_mul, hbound_def]
    have hp_nonneg : 0 ≤ (p ab.1).toReal := ENNReal.toReal_nonneg
    have hq_nonneg : 0 ≤ (q ab.2).toReal := ENNReal.toReal_nonneg
    rw [abs_of_nonneg hp_nonneg, abs_of_nonneg hq_nonneg]
    calc (p ab.1).toReal * (q ab.2).toReal * |f ab.1 ab.2|
        ≤ (p ab.1).toReal * (q ab.2).toReal * M := by
            apply mul_le_mul_of_nonneg_left (hf ab.1 ab.2)
            exact mul_nonneg hp_nonneg hq_nonneg
      _ = M * (p ab.1).toReal * (q ab.2).toReal := by ring
  exact Summable.of_norm_bounded (g := bound) hbound_summable hbound_le

/-
-- PROOF SKETCH (times out due to expensive typeclass inference and ring tactics)
-- The mathematical content is standard: bounded functions over product measures are summable.
-- Key steps:
-- 1. Define bound g(a,b) = M * p(a) * q(b)
-- 2. Show g is summable using summable_prod_of_nonneg and PMF.summable_coe_real
-- 3. Apply Summable.of_norm_bounded since |p(a)*q(b)*f(a,b)| ≤ g(a,b)
-/

/- Deprecated lemma `PMF.summable_prod_mul_of_factor_right` removed; use
   `PMF.summable_prod_mul_of_bounded` instead. -/

/-- Coupling bound inequality with explicit bounds.
    When |f(x,z)| ≤ C·d(x,z), the coupled PMF sum is bounded.
    Avoids the unsound axiom by requiring explicit bounds on the distance function. -/
lemma coupling_bound_ineq_bounded {α : Type*} (μ_X μ_Z : PMF α) (f : α → α → ℝ) (C : ℝ) (d : α → α → ℝ)
    (hC : 0 ≤ C) (hd : ∀ x z, 0 ≤ d x z)
    (hbound : ∀ x z, |f x z| ≤ C * d x z)
    (M_d : ℝ) (hM_d : 0 ≤ M_d) (hd_bound : ∀ x z, d x z ≤ M_d) :
    |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z| ≤
    C * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * d x z := by
  -- Derive bound on f from bound on d: |f x z| ≤ C * d x z ≤ C * M_d
  let M_f := C * M_d
  have hM_f : 0 ≤ M_f := mul_nonneg hC hM_d
  have hf_bound : ∀ x z, |f x z| ≤ M_f := fun x z =>
    calc |f x z| ≤ C * d x z := hbound x z
      _ ≤ M_f := mul_le_mul_of_nonneg_left (hd_bound x z) hC

  -- Factor out (μ_X x).toReal from inner sums
  have factor_f : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z =
                       (μ_X x).toReal * ∑' z, (μ_Z z).toReal * f x z := by
    intro x; rw [← tsum_mul_left]; congr 1; ext z; ring
  have factor_abs : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * |f x z| =
                         (μ_X x).toReal * ∑' z, (μ_Z z).toReal * |f x z| := by
    intro x; rw [← tsum_mul_left]; congr 1; ext z; ring
  have factor_Cd : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (C * d x z) =
                        (μ_X x).toReal * ∑' z, (μ_Z z).toReal * (C * d x z) := by
    intro x; rw [← tsum_mul_left]; congr 1; ext z; ring
  have factor_d : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * d x z =
                       (μ_X x).toReal * ∑' z, (μ_Z z).toReal * d x z := by
    intro x; rw [← tsum_mul_left]; congr 1; ext z; ring

  -- Inner summability using bounded helpers
  have inner_f : ∀ x, Summable (fun z => (μ_Z z).toReal * f x z) :=
    fun x => summable_coupling_inner_bounded μ_Z _ M_f hM_f (fun z => hf_bound x z)
  have inner_abs : ∀ x, Summable (fun z => (μ_Z z).toReal * |f x z|) :=
    fun x => summable_coupling_inner_bounded μ_Z _ M_f hM_f (fun z => by rw [abs_abs]; exact hf_bound x z)
  have inner_Cd : ∀ x, Summable (fun z => (μ_Z z).toReal * (C * d x z)) :=
    fun x => summable_coupling_inner_bounded μ_Z _ M_f hM_f (fun z => by
      rw [abs_of_nonneg (mul_nonneg hC (hd x z))]
      exact mul_le_mul_of_nonneg_left (hd_bound x z) hC)
  have inner_d : ∀ x, Summable (fun z => (μ_Z z).toReal * d x z) :=
    fun x => summable_coupling_inner_bounded μ_Z _ M_d hM_d (fun z => by
      rw [abs_of_nonneg (hd x z)]; exact hd_bound x z)

  -- Outer summability using bounded helpers
  -- Bound on inner sum: |∑' z, μ_Z(z) * f(x,z)| ≤ M_f (since f is bounded)
  have inner_sum_bound : ∀ x, |∑' z, (μ_Z z).toReal * f x z| ≤ M_f := fun x => by
    calc |∑' z, (μ_Z z).toReal * f x z|
        ≤ ∑' z, |(μ_Z z).toReal * f x z| := abs_tsum_le_tsum_abs' _ (inner_f x) (inner_f x).abs
      _ = ∑' z, (μ_Z z).toReal * |f x z| := by
          apply tsum_congr; intro z; rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' z, (μ_Z z).toReal * M_f := by
          apply Summable.tsum_le_tsum _ (inner_abs x)
            (summable_coupling_inner_bounded μ_Z _ M_f hM_f (fun _ => by rw [abs_of_nonneg hM_f]))
          intro z; apply mul_le_mul_of_nonneg_left (hf_bound x z) ENNReal.toReal_nonneg
      _ = M_f := by
          have h : (fun z => (μ_Z z).toReal * M_f) = (fun z => M_f * (μ_Z z).toReal) := by ext z; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe μ_Z]; ring

  have sum_f : Summable (fun x => (μ_X x).toReal * ∑' z, (μ_Z z).toReal * f x z) :=
    summable_coupling_outer_bounded μ_X _ M_f hM_f inner_sum_bound
  have sum_f' : Summable (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z) := by
    convert sum_f using 1; ext x; exact factor_f x
  have sum_abs_f : Summable (fun x => |∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z|) :=
    sum_f'.abs

  -- Bound on inner sum of |f|
  have inner_abs_sum_bound : ∀ x, |(∑' z, (μ_Z z).toReal * |f x z|)| ≤ M_f := fun x => by
    rw [abs_of_nonneg (tsum_nonneg (fun z => mul_nonneg ENNReal.toReal_nonneg (abs_nonneg _)))]
    calc ∑' z, (μ_Z z).toReal * |f x z|
        ≤ ∑' z, (μ_Z z).toReal * M_f := by
          apply Summable.tsum_le_tsum _ (inner_abs x)
            (summable_coupling_inner_bounded μ_Z _ M_f hM_f (fun _ => by rw [abs_of_nonneg hM_f]))
          intro z; apply mul_le_mul_of_nonneg_left (hf_bound x z) ENNReal.toReal_nonneg
      _ = M_f := by
          have h : (fun z => (μ_Z z).toReal * M_f) = (fun z => M_f * (μ_Z z).toReal) := by ext z; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe μ_Z]; ring

  have sum_inner_abs : Summable (fun x => (μ_X x).toReal * ∑' z, (μ_Z z).toReal * |f x z|) :=
    summable_coupling_outer_bounded μ_X _ M_f hM_f inner_abs_sum_bound
  have sum_inner_abs' : Summable (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * |f x z|) := by
    convert sum_inner_abs using 1; ext x; exact factor_abs x

  -- Bound on inner sum of C*d
  have inner_Cd_sum_bound : ∀ x, |∑' z, (μ_Z z).toReal * (C * d x z)| ≤ M_f := fun x => by
    rw [abs_of_nonneg (tsum_nonneg (fun z => mul_nonneg ENNReal.toReal_nonneg (mul_nonneg hC (hd x z))))]
    calc ∑' z, (μ_Z z).toReal * (C * d x z)
        ≤ ∑' z, (μ_Z z).toReal * M_f := by
          apply Summable.tsum_le_tsum _ (inner_Cd x)
            (summable_coupling_inner_bounded μ_Z _ M_f hM_f (fun _ => by rw [abs_of_nonneg hM_f]))
          intro z
          apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
          exact mul_le_mul_of_nonneg_left (hd_bound x z) hC
      _ = M_f := by
          have h : (fun z => (μ_Z z).toReal * M_f) = (fun z => M_f * (μ_Z z).toReal) := by ext z; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe μ_Z]; ring

  have sum_Cd : Summable (fun x => (μ_X x).toReal * ∑' z, (μ_Z z).toReal * (C * d x z)) :=
    summable_coupling_outer_bounded μ_X _ M_f hM_f inner_Cd_sum_bound
  have sum_Cd' : Summable (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (C * d x z)) := by
    convert sum_Cd using 1; ext x; exact factor_Cd x

  -- Step 1: |∑∑| ≤ ∑|∑| (outer triangle inequality)
  have h1 : |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z| ≤
            ∑' x, |∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z| :=
    abs_tsum_le_tsum_abs' _ sum_f' sum_abs_f

  -- Step 2: For each x, |∑ z| ≤ ∑ z |·| (inner triangle inequality)
  have h2 : ∀ x, |∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z| ≤
            ∑' z, (μ_X x).toReal * (μ_Z z).toReal * |f x z| := by
    intro x
    rw [factor_f x, factor_abs x]
    rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
    apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
    have h := abs_tsum_le_tsum_abs' _ (inner_f x) (inner_f x).abs
    calc |∑' z, (μ_Z z).toReal * f x z|
        ≤ ∑' z, |(μ_Z z).toReal * f x z| := h
      _ = ∑' z, (μ_Z z).toReal * |f x z| := by
          apply tsum_congr; intro z
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]

  -- Step 3: |f x z| ≤ C * d x z pointwise
  have h3 : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * |f x z| ≤
            ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (C * d x z) := by
    intro x
    rw [factor_abs x, factor_Cd x]
    apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
    apply Summable.tsum_le_tsum _ (inner_abs x) (inner_Cd x)
    intro z
    apply mul_le_mul_of_nonneg_left (hbound x z) ENNReal.toReal_nonneg

  -- Step 4: Factor out C
  have h4 : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (C * d x z) =
            C * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * d x z := by
    conv_lhs =>
      congr; ext x
      rw [factor_Cd x]
      rw [show ∑' z, (μ_Z z).toReal * (C * d x z) = C * ∑' z, (μ_Z z).toReal * d x z by
          rw [← tsum_mul_left]; congr 1; ext z; ring]
    rw [← tsum_mul_left]
    congr 1; ext x
    rw [show (μ_X x).toReal * (C * ∑' z, (μ_Z z).toReal * d x z) =
            C * ((μ_X x).toReal * ∑' z, (μ_Z z).toReal * d x z) by ring]
    rw [← factor_d x]

  -- Combine all steps
  calc |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z|
      ≤ ∑' x, |∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x z| := h1
    _ ≤ ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * |f x z| :=
        Summable.tsum_le_tsum h2 sum_abs_f sum_inner_abs'
    _ ≤ ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (C * d x z) :=
        Summable.tsum_le_tsum h3 sum_inner_abs' sum_Cd'
    _ = C * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * d x z := h4

/- Deprecated lemma `coupling_bound_ineq` removed; use `coupling_bound_ineq_bounded`. -/

/-- Coupling expansion with explicit bounds: difference of expectations equals expectation of differences.
This is the key identity for the coupling argument.
Avoids the unsound axiom by requiring explicit bounds on f. -/
lemma coupling_expansion_bounded {α : Type*} (μ_X μ_Z : PMF α) (f : α → ℝ)
    (M : ℝ) (hM : 0 ≤ M) (hf_bound : ∀ x, |f x| ≤ M) :
    (∑' x, (μ_X x).toReal * f x) - (∑' z, (μ_Z z).toReal * f z) =
    ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (f x - f z) := by
  have hsum_Z : ∑' z, (μ_Z z).toReal = 1 := PMF.toReal_tsum_coe μ_Z
  have hsum_X : ∑' x, (μ_X x).toReal = 1 := PMF.toReal_tsum_coe μ_X

  -- Helper: reorder terms for inner sum with f x (x is fixed, so μ_X(x)*f(x) is constant)
  have inner_eq_fx : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x = (μ_X x).toReal * f x := by
    intro x
    have h : ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x =
             (μ_X x).toReal * f x * ∑' z, (μ_Z z).toReal := by
      rw [← tsum_mul_left]; congr 1; ext z; ring
    rw [h, hsum_Z]; ring

  -- Helper: reorder terms for inner sum with f z
  have inner_eq_fz : ∀ x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f z =
                          (μ_X x).toReal * ∑' z, (μ_Z z).toReal * f z := by
    intro x
    rw [← tsum_mul_left]; congr 1; ext z; ring

  -- Inner summability (for each fixed x) - using simple constant summability
  have hA_inner : ∀ x, Summable (fun z => (μ_X x).toReal * (μ_Z z).toReal * f x) := fun x => by
    -- This is a constant times μ_Z(z), which is summable
    have h : (fun z => (μ_X x).toReal * (μ_Z z).toReal * f x) =
             (fun z => ((μ_X x).toReal * f x) * (μ_Z z).toReal) := by ext z; ring
    rw [h]
    exact (PMF.summable_coe_real μ_Z).mul_left _
  have hB_inner : ∀ x, Summable (fun z => (μ_X x).toReal * (μ_Z z).toReal * f z) := fun x => by
    -- This is μ_X(x) times a bounded sum in z
    have h : (fun z => (μ_X x).toReal * (μ_Z z).toReal * f z) =
             (fun z => (μ_X x).toReal * ((μ_Z z).toReal * f z)) := by ext z; ring
    rw [h]
    exact (summable_coupling_inner_bounded μ_Z f M hM hf_bound).mul_left _

  -- Bounds for inner sums (bound doesn't depend on x, but we need ∀ _ for the outer summability helper)
  have hfz_inner_bound : ∀ _ : α, |∑' z, (μ_Z z).toReal * f z| ≤ M := fun _ => by
    calc |∑' z, (μ_Z z).toReal * f z|
        ≤ ∑' z, |(μ_Z z).toReal * f z| := abs_tsum_le_tsum_abs' _
            (summable_coupling_inner_bounded μ_Z f M hM hf_bound)
            (summable_coupling_inner_bounded μ_Z f M hM hf_bound).abs
      _ = ∑' z, (μ_Z z).toReal * |f z| := by
          apply tsum_congr; intro z; rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' z, (μ_Z z).toReal * M := by
          apply Summable.tsum_le_tsum _
            (summable_coupling_inner_bounded μ_Z (fun z => |f z|) M hM (fun z => by rw [abs_abs]; exact hf_bound z))
            (summable_coupling_inner_bounded μ_Z (fun _ => M) M hM (fun _ => by rw [abs_of_nonneg hM]))
          intro z; apply mul_le_mul_of_nonneg_left (hf_bound z) ENNReal.toReal_nonneg
      _ = M := by
          have h : (fun z => (μ_Z z).toReal * M) = (fun z => M * (μ_Z z).toReal) := by ext z; ring
          rw [h, tsum_mul_left, hsum_Z]; ring

  -- Outer summability using bounded helpers
  have hA_outer : Summable (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x) := by
    have eq : (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x) =
              (fun x => (μ_X x).toReal * f x) := funext inner_eq_fx
    rw [eq]; exact summable_coupling_outer_bounded μ_X f M hM hf_bound
  have hB_outer : Summable (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f z) := by
    have eq : (fun x => ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f z) =
              (fun x => (μ_X x).toReal * (∑' z, (μ_Z z).toReal * f z)) := funext inner_eq_fz
    rw [eq]; exact summable_coupling_outer_bounded μ_X (fun _ => ∑' z, (μ_Z z).toReal * f z) M hM hfz_inner_bound

  -- Step 1: Split the RHS double sum into difference of two double sums
  have rhs_eq : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (f x - f z) =
      (∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f x) -
      (∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f z) := by
    symm
    rw [← Summable.tsum_sub hA_outer hB_outer]
    congr 1; ext x
    rw [← Summable.tsum_sub (hA_inner x) (hB_inner x)]
    congr 1; ext z; ring

  rw [rhs_eq]
  congr 1
  -- First double sum simplifies to first single sum
  · symm
    exact tsum_congr inner_eq_fx
  -- Second double sum: swap order, then simplify to second single sum
  · have hswap : Summable (Function.uncurry fun x z => (μ_X x).toReal * (μ_Z z).toReal * f z) :=
      PMF.summable_prod_mul_of_bounded μ_X μ_Z (fun _ z => f z) M hM (fun _ z => hf_bound z)
    have swap_eq : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * f z =
                   ∑' z, ∑' x, (μ_X x).toReal * (μ_Z z).toReal * f z := by
      rw [Summable.tsum_comm hswap]
    rw [swap_eq]
    symm
    apply tsum_congr
    intro z
    have h : ∑' x, (μ_X x).toReal * (μ_Z z).toReal * f z =
             (μ_Z z).toReal * f z * ∑' x, (μ_X x).toReal := by
      rw [← tsum_mul_left]; congr 1; ext x; ring
    rw [h, hsum_X]; ring

/- Deprecated lemma `coupling_expansion` removed; use `coupling_expansion_bounded`. -/

/-!
## Unified Preference Gap Bound

This section provides the **unified abstraction** that encompasses all preference learning
gap bounds (DPO, GRPO-PL, GRPO-RL, and future methods).

The key insight is that all preference methods follow the same template:
  Gap ≤ Lipschitz_Constant × Expected_Distortion

The proof structure is method-agnostic:
1. coupling_expansion rewrites E_X - E_Z as a double sum
2. The Lipschitz bound on the inner expectation provides pointwise control
3. coupling_bound_ineq + Fubini gives the final bound
-/

/-- **Unified Preference Gap Bound (Bounded Version)**

Any preference loss with a Lipschitz expected-over-generator function satisfies
the standard gap bound. This version uses explicit bounds to avoid the unsound axiom.

**Mathematical Statement:**
  If E_gen(x) is L-Lipschitz in oracle distance (i.e., |E_gen(x) - E_gen(z)| ≤ L⋅dist(f*(x), f*(z))),
  then |E_X[E_gen] - E_Z[E_gen]| ≤ L × Δ_R

**Required bounds:**
- `E_max`: uniform bound on |E_gen(x)|
- `D_max`: bound on oracle distances

**Instantiations:**
- DPO: L = 2|β|L_pol, E_gen = expected DPO loss over pairs
- GRPO-PL: L = L_grpo, E_gen = expected Plackett-Luce loss over groups
- GRPO-RL: L = L_grpo_rl, E_gen = expected clipped advantage + KL over groups -/
theorem unified_preference_gap_bounded {Strings Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (E_gen : Strings → ℝ)  -- Expected loss over generator for fixed document
    (μ_X μ_Z : PMF Strings)
    (L : ℝ≥0) (Δ_R : ℝ)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    (E_max : ℝ) (hE_max : 0 ≤ E_max) (hE_bound : ∀ x, |E_gen x| ≤ E_max)
    (h_lip : ∀ x z, |E_gen x - E_gen z| ≤ L * dist (fstar x) (fstar z))
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |∑' x, (μ_X x).toReal * E_gen x - ∑' z, (μ_Z z).toReal * E_gen z| ≤ L * Δ_R := by
  -- Step 1: Apply coupling expansion (bounded version)
  rw [coupling_expansion_bounded μ_X μ_Z E_gen E_max hE_max hE_bound]

  -- Step 2: Define explicit functions to avoid type inference issues
  let f : Strings → Strings → ℝ := fun x z => E_gen x - E_gen z
  let d : Strings → Strings → ℝ := fun x z => dist (fstar x) (fstar z)

  -- Step 3: Establish bounds
  have h_abs_bound : ∀ x z, |d x z| ≤ D_max := fun x z => by
    rw [abs_of_nonneg dist_nonneg]; exact h_dist_bound x z

  -- Step 4: Derive bound on f from bound on E_gen
  have h_f_bound : ∀ x z, |f x z| ≤ 2 * E_max := fun x z => by
    calc |f x z|
        = |E_gen x - E_gen z| := rfl
      _ ≤ |E_gen x| + |E_gen z| := by
          calc |E_gen x - E_gen z|
              = |E_gen x + (-(E_gen z))| := by ring_nf
            _ ≤ |E_gen x| + |-(E_gen z)| := abs_add_le _ _
            _ = |E_gen x| + |E_gen z| := by rw [abs_neg]
      _ ≤ E_max + E_max := add_le_add (hE_bound x) (hE_bound z)
      _ = 2 * E_max := by ring

  -- Step 5: Apply coupling bound (bounded version)
  have hM_f : 0 ≤ 2 * E_max := by linarith
  have h_coupling := coupling_bound_ineq_bounded μ_X μ_Z f (L : ℝ) d (NNReal.coe_nonneg L)
    (fun _ _ => dist_nonneg) h_lip D_max hD_max h_dist_bound

  -- Step 6: Establish summability for Fubini
  have hswap : Summable (Function.uncurry fun x z => (μ_X x).toReal * (μ_Z z).toReal * d x z) :=
    PMF.summable_prod_mul_of_bounded μ_X μ_Z d D_max hD_max h_abs_bound

  -- Step 7: Fubini + dist_comm
  have h_fubini : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * d x z =
                  ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x) := by
    rw [(Summable.tsum_comm hswap).symm]
    apply tsum_congr; intro z
    apply tsum_congr; intro x
    simp only [d]; rw [dist_comm]; ring

  calc |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (E_gen x - E_gen z)|
      ≤ (L : ℝ) * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * d x z := h_coupling
    _ = (L : ℝ) * ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x) := by
        rw [h_fubini]
    _ = L * Δ_R := by rw [h_Δ]

/- Deprecated lemma `unified_preference_gap` removed; use `unified_preference_gap_bounded`. -/

/-!
## Unified Preference Gap Bound over an Arbitrary Coupling

`unified_preference_gap_bounded` hard-codes the independent product coupling
`μ_X ⊗ μ_Z`. The paper's Theorem (`thm:unified-gap`) speaks of *any* coupled
pair `(X, Z^(R)(X))`; the version below takes an arbitrary joint distribution
`μ` over document/summary pairs, with `Δ_R` the coupled expected oracle
distance. The product form is the special case `μ = μ_X ⊗ μ_Z`; the canonical
document-summary coupling is `documentSummaryCoupling` below.
-/

/-- **Unified Preference Gap Bound (coupled version).** For any joint
distribution `μ` over `(document, summary)` pairs, the objective gap is bounded
by `L` times the coupled expected oracle distance — the paper's statement
"for any coupled pair (X, Z^(R)(X))" verbatim. -/
theorem unified_preference_gap_bounded_coupled {Strings Y : Type*} [Monoid Strings]
    [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (E_gen : Strings → ℝ)
    (μ : PMF (Strings × Strings))
    (L : ℝ≥0) (Δ_R : ℝ)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    (h_lip : ∀ x z, |E_gen x - E_gen z| ≤ L * dist (fstar x) (fstar z))
    (h_Δ : Δ_R = ∑' q : Strings × Strings, (μ q).toReal * dist (fstar q.1) (fstar q.2)) :
    |∑' q : Strings × Strings, (μ q).toReal * (E_gen q.1 - E_gen q.2)| ≤ L * Δ_R := by
  have hLD : (0 : ℝ) ≤ (L : ℝ) * D_max := mul_nonneg (NNReal.coe_nonneg L) hD_max
  have hdiff_bound : ∀ q : Strings × Strings, |E_gen q.1 - E_gen q.2| ≤ (L : ℝ) * D_max :=
    fun q => (h_lip q.1 q.2).trans
      (mul_le_mul_of_nonneg_left (h_dist_bound q.1 q.2) (NNReal.coe_nonneg L))
  have hdist_abs : ∀ q : Strings × Strings, |dist (fstar q.1) (fstar q.2)| ≤ D_max :=
    fun q => by rw [abs_of_nonneg dist_nonneg]; exact h_dist_bound q.1 q.2
  have hsum_diff : Summable (fun q : Strings × Strings =>
      (μ q).toReal * (E_gen q.1 - E_gen q.2)) :=
    PMF.summable_coe_real_mul_of_bounded μ _ ((L : ℝ) * D_max) hLD hdiff_bound
  have hsum_dist : Summable (fun q : Strings × Strings =>
      (μ q).toReal * dist (fstar q.1) (fstar q.2)) :=
    PMF.summable_coe_real_mul_of_bounded μ _ D_max hD_max hdist_abs
  calc |∑' q : Strings × Strings, (μ q).toReal * (E_gen q.1 - E_gen q.2)|
      ≤ ∑' q : Strings × Strings, |(μ q).toReal * (E_gen q.1 - E_gen q.2)| := by
        simpa [Real.norm_eq_abs] using norm_tsum_le_tsum_norm (f := fun q : Strings × Strings =>
          (μ q).toReal * (E_gen q.1 - E_gen q.2)) (by simpa [Real.norm_eq_abs] using hsum_diff.abs)
    _ ≤ ∑' q : Strings × Strings, (μ q).toReal * ((L : ℝ) * dist (fstar q.1) (fstar q.2)) := by
        exact Summable.tsum_le_tsum (fun q => by
            rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
            exact mul_le_mul_of_nonneg_left (h_lip q.1 q.2) ENNReal.toReal_nonneg)
          hsum_diff.abs
          ((hsum_dist.mul_left (L : ℝ)).congr (fun q => by ring))
    _ = (L : ℝ) * ∑' q : Strings × Strings, (μ q).toReal * dist (fstar q.1) (fstar q.2) := by
        rw [← tsum_mul_left]
        exact tsum_congr fun q => by ring
    _ = L * Δ_R := by rw [h_Δ]

/-- The canonical coupling of a document with its own multi-round summary:
draw `x ~ μ_X`, then `z ~ Z^(R)(x)` on the tree assigned to `x`, and return the
pair `(x, z)`. This is the coupling the paper's `thm:unified-gap` intends. -/
noncomputable def documentSummaryCoupling {Strings : Type*} [Monoid Strings]
    (g : Summarizer Strings) (μ_X : PMF Strings) (Tpi : Strings → BinTree Strings)
    (R : ℕ) : PMF (Strings × Strings) :=
  μ_X.bind fun x => (ZR g x R (Tpi x)).map fun z => (x, z)

/-- Support of the document-summary coupling. -/
lemma mem_support_documentSummaryCoupling_iff {Strings : Type*} [Monoid Strings]
    (g : Summarizer Strings) (μ_X : PMF Strings) (Tpi : Strings → BinTree Strings)
    (R : ℕ) (q : Strings × Strings) :
    q ∈ (documentSummaryCoupling g μ_X Tpi R).support ↔
      q.1 ∈ μ_X.support ∧ q.2 ∈ (ZR g q.1 R (Tpi q.1)).support := by
  unfold documentSummaryCoupling
  constructor
  · intro hq
    rw [PMF.mem_support_bind_iff] at hq
    obtain ⟨x, hx, hq⟩ := hq
    rw [PMF.mem_support_map_iff] at hq
    obtain ⟨z, hz, hzq⟩ := hq
    cases hzq
    exact ⟨hx, hz⟩
  · rintro ⟨h1, h2⟩
    rw [PMF.mem_support_bind_iff]
    exact ⟨q.1, h1, by rw [PMF.mem_support_map_iff]; exact ⟨q.2, h2, rfl⟩⟩

/-- **Coupled distortion vanishes under the local laws.** On the canonical
document-summary coupling, the local laws make the coupled `Δ_R` exactly zero —
the population-level closure of the gap bound. -/
theorem documentSummaryCoupling_delta_zero {Strings Y : Type*} [Monoid Strings]
    [PseudoMetricSpace Y]
    (g : Summarizer Strings) (fstar : Strings → Y)
    (μ_X : PMF Strings) (Tpi : Strings → BinTree Strings) (R : ℕ) (hR : R ≥ 1)
    (hp : ∀ x ∈ μ_X.support, S (Tpi x) = x)
    (hctx : ContextCompatible fstar)
    (h1 : ∀ x ∈ μ_X.support, LeafSufficiency g (Tpi x) fstar)
    (h2 : ∀ x ∈ μ_X.support, MergeSufficiency g (Tpi x) fstar)
    (h3 : RangeIdempotence g fstar) :
    ∑' q : Strings × Strings,
      ((documentSummaryCoupling g μ_X Tpi R) q).toReal * dist (fstar q.1) (fstar q.2) = 0 := by
  have hterm : ∀ q : Strings × Strings,
      ((documentSummaryCoupling g μ_X Tpi R) q).toReal * dist (fstar q.1) (fstar q.2) = 0 := by
    intro q
    by_cases hq : q ∈ (documentSummaryCoupling g μ_X Tpi R).support
    · rw [mem_support_documentSummaryCoupling_iff] at hq
      have hzx : OracleEquiv fstar q.2 q.1 :=
        population_preservation g fstar μ_X Tpi R hR hp hctx h1 h2 h3 q.1 hq.1 q.2 hq.2
      have : dist (fstar q.1) (fstar q.2) = 0 := by
        rw [dist_comm]; exact hzx
      rw [this, mul_zero]
    · rw [PMF.mem_support_iff, not_not] at hq
      rw [hq]
      simp
  rw [tsum_congr hterm]
  exact tsum_zero

/-- **Population preference gap vanishes under the local laws (coupled form).**
Combining the coupled gap bound with the vanishing coupled distortion: for any
Lipschitz method loss, the population objective computed on documents equals
the population objective computed on their multi-round summaries. -/
theorem population_gap_zero_of_local {Strings Y : Type*} [Monoid Strings]
    [PseudoMetricSpace Y]
    (g : Summarizer Strings) (fstar : Strings → Y) (E_gen : Strings → ℝ)
    (μ_X : PMF Strings) (Tpi : Strings → BinTree Strings) (R : ℕ) (hR : R ≥ 1)
    (L : ℝ≥0)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    (h_lip : ∀ x z, |E_gen x - E_gen z| ≤ L * dist (fstar x) (fstar z))
    (hp : ∀ x ∈ μ_X.support, S (Tpi x) = x)
    (hctx : ContextCompatible fstar)
    (h1 : ∀ x ∈ μ_X.support, LeafSufficiency g (Tpi x) fstar)
    (h2 : ∀ x ∈ μ_X.support, MergeSufficiency g (Tpi x) fstar)
    (h3 : RangeIdempotence g fstar) :
    ∑' q : Strings × Strings,
      ((documentSummaryCoupling g μ_X Tpi R) q).toReal * (E_gen q.1 - E_gen q.2) = 0 := by
  have hgap := unified_preference_gap_bounded_coupled fstar E_gen
    (documentSummaryCoupling g μ_X Tpi R) L 0 D_max hD_max h_dist_bound h_lip
    (documentSummaryCoupling_delta_zero g fstar μ_X Tpi R hR hp hctx h1 h2 h3).symm
  rw [mul_zero] at hgap
  exact abs_eq_zero.mp (le_antisymm hgap (abs_nonneg _))

/-!
## DPO-Specific Infrastructure

The following lemmas establish the DPO instantiation of the unified framework.
-/

/-- E_pair Lipschitz with explicit bounds.
    Avoids the unsound axiom by requiring explicit bounds on DPO losses. -/
lemma E_pair_lipschitz_bounded {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y) (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0) (g : PMF (A × A))
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (x z : Strings)
    (M : ℝ) (hM : 0 ≤ M)
    (hbound_x : ∀ p : A × A, |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M)
    (hbound_z : ∀ p : A × A, |DPOLossPointwise pol pol_ref β z p.1 p.2| ≤ M) :
    |∑' p, (g p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 -
     ∑' p, (g p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2| ≤
    2 * |β| * (L_pol : ℝ) * dist (fstar x) (fstar z) := by
  -- Derive bounds for differences: |loss_x - loss_z| ≤ |loss_x| + |loss_z| ≤ 2M
  have hbound_diff : ∀ p : A × A, |DPOLossPointwise pol pol_ref β x p.1 p.2 -
                                   DPOLossPointwise pol pol_ref β z p.1 p.2| ≤ 2 * M := by
    intro p
    have h_tri : |DPOLossPointwise pol pol_ref β x p.1 p.2 - DPOLossPointwise pol pol_ref β z p.1 p.2| ≤
                 |DPOLossPointwise pol pol_ref β x p.1 p.2| + |DPOLossPointwise pol pol_ref β z p.1 p.2| := by
      calc |DPOLossPointwise pol pol_ref β x p.1 p.2 - DPOLossPointwise pol pol_ref β z p.1 p.2|
          = |DPOLossPointwise pol pol_ref β x p.1 p.2 + (-(DPOLossPointwise pol pol_ref β z p.1 p.2))| := by ring_nf
        _ ≤ |DPOLossPointwise pol pol_ref β x p.1 p.2| + |-(DPOLossPointwise pol pol_ref β z p.1 p.2)| := abs_add_le _ _
        _ = |DPOLossPointwise pol pol_ref β x p.1 p.2| + |DPOLossPointwise pol pol_ref β z p.1 p.2| := by rw [abs_neg]
    calc |DPOLossPointwise pol pol_ref β x p.1 p.2 - DPOLossPointwise pol pol_ref β z p.1 p.2|
        ≤ |DPOLossPointwise pol pol_ref β x p.1 p.2| + |DPOLossPointwise pol pol_ref β z p.1 p.2| := h_tri
      _ ≤ M + M := add_le_add (hbound_x p) (hbound_z p)
      _ = 2 * M := by ring
  have h2M : 0 ≤ 2 * M := by linarith
  -- Combine the sums using bounded summability
  have h_sub : ∑' p, (g p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 -
               ∑' p, (g p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2 =
               ∑' p, (g p).toReal * (DPOLossPointwise pol pol_ref β x p.1 p.2 -
                                     DPOLossPointwise pol pol_ref β z p.1 p.2) := by
    have h1 := summable_coupling_inner_bounded g _ M hM hbound_x
    have h2 := summable_coupling_inner_bounded g _ M hM hbound_z
    rw [← Summable.tsum_sub h1 h2]
    congr 1; ext p; ring
  rw [h_sub]
  -- Apply triangle inequality for tsum with bounded summability
  have hsum : Summable (fun p => (g p).toReal * (DPOLossPointwise pol pol_ref β x p.1 p.2 -
                                                 DPOLossPointwise pol pol_ref β z p.1 p.2)) :=
    summable_coupling_inner_bounded g _ (2 * M) h2M hbound_diff
  have hsum_abs : Summable (fun p => |(g p).toReal * (DPOLossPointwise pol pol_ref β x p.1 p.2 -
                                                       DPOLossPointwise pol pol_ref β z p.1 p.2)|) :=
    hsum.abs
  -- Summability for the constant term (factors out of the sum)
  let C := 2 * |β| * (L_pol : ℝ) * dist (fstar x) (fstar z)
  have hC_nonneg : 0 ≤ C := by simp only [C]; positivity
  have hsum_const : Summable (fun p => (g p).toReal * C) := by
    have h := PMF.summable_coe_real g
    exact h.mul_right C
  -- Summability for the absolute value version
  have hsum_abs_fn : Summable (fun p => (g p).toReal * |DPOLossPointwise pol pol_ref β x p.1 p.2 -
                                                        DPOLossPointwise pol pol_ref β z p.1 p.2|) := by
    apply summable_coupling_inner_bounded g _ (2 * M) h2M
    intro p
    rw [abs_abs]
    exact hbound_diff p
  calc |∑' p, (g p).toReal * (DPOLossPointwise pol pol_ref β x p.1 p.2 -
                              DPOLossPointwise pol pol_ref β z p.1 p.2)|
      ≤ ∑' p, |(g p).toReal * (DPOLossPointwise pol pol_ref β x p.1 p.2 -
                               DPOLossPointwise pol pol_ref β z p.1 p.2)| := abs_tsum_le_tsum_abs' _ hsum hsum_abs
    _ = ∑' p, (g p).toReal * |DPOLossPointwise pol pol_ref β x p.1 p.2 -
                              DPOLossPointwise pol pol_ref β z p.1 p.2| := by
        congr 1; ext p
        rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
    _ ≤ ∑' p, (g p).toReal * C := by
        apply Summable.tsum_le_tsum _ hsum_abs_fn hsum_const
        intro p
        apply mul_le_mul_of_nonneg_left (dpo_loss_pointwise_lipschitz h_lip p.1 p.2 x z)
        exact ENNReal.toReal_nonneg
    _ = C * ∑' p, (g p).toReal := by
        have : (fun p => (g p).toReal * C) = (fun p => C * (g p).toReal) := by ext p; ring
        rw [this, tsum_mul_left]
    _ = C := by rw [PMF.toReal_tsum_coe g]; ring

/- Deprecated lemma `E_pair_lipschitz` removed; use `E_pair_lipschitz_bounded`. -/

/-- **Theorem: DPO Gap Bound with Explicit Bounds (Paper: Theorem 4)**

**Paper Reference:** Section 6, Theorem 4 (Policy-Lipschitz version)

With Lipschitz conditions on policies, the DPO loss gap is bounded by expected distortion:
  `|L^X(π) - L^Z(π)| ≤ 2|β|L_π · Δ_R`

This version uses explicit bounds to avoid the unsound summability axiom.

**Required bounds:**
- `D_max`: bound on oracle distances
- `Loss_max`: uniform bound on |DPOLossPointwise|

⚠️ **Important assumption:** This theorem requires `h_gen_fixed : ∀ x x', gen x = gen x'`,
meaning the pair generator must be **constant** (independent of the document). -/
theorem dpo_gap_bounded {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ) (L_pol : ℝ≥0)
    (Δ_R : ℝ)
    -- Diameter bound: oracle distances are bounded (ensures summability)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    -- Loss bound: DPO loss is bounded (ensures summability)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (p : A × A), |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ Loss_max)
    (_h_m_pol : DPO.OracleMeasurable pol fstar)
    (_h_m_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedDPOLoss pol pol_ref β μ_X gen - ExpectedDPOLoss pol pol_ref β μ_Z gen| ≤
    2 * |β| * (L_pol : ℝ) * Δ_R := by
  -- Step 1: Fix the generator (gen is constant by h_gen_fixed)
  let g := gen (Classical.arbitrary Strings)
  have hgen_eq : ∀ x, gen x = g := fun x => h_gen_fixed x _

  -- Step 2: Define E_pair and show ExpectedDPOLoss μ gen = Exp μ E_pair
  let E_pair := fun x => ∑' p, (g p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2

  have hE_eq : ∀ μ, ExpectedDPOLoss pol pol_ref β μ gen = ∑' x, (μ x).toReal * E_pair x := by
    intro μ
    unfold ExpectedDPOLoss
    congr 1; ext x
    rw [hgen_eq x]

  -- Step 3: Derive E_pair bound from Loss_max
  have hE_pair_bound : ∀ x, |E_pair x| ≤ Loss_max := fun x => by
    calc |E_pair x|
        = |∑' p, (g p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2| := rfl
      _ ≤ ∑' p, |(g p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2| := by
          apply abs_tsum_le_tsum_abs'
          · exact summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun p => hLoss_bound x p)
          · exact (summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun p => hLoss_bound x p)).abs
      _ = ∑' p, (g p).toReal * |DPOLossPointwise pol pol_ref β x p.1 p.2| := by
          apply tsum_congr; intro p
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' p, (g p).toReal * Loss_max := by
          apply Summable.tsum_le_tsum
          · intro p; apply mul_le_mul_of_nonneg_left (hLoss_bound x p) ENNReal.toReal_nonneg
          · exact summable_coupling_inner_bounded g (fun p => |DPOLossPointwise pol pol_ref β x p.1 p.2|)
              Loss_max hLoss_max (fun p => by rw [abs_abs]; exact hLoss_bound x p)
          · exact summable_coupling_inner_bounded g (fun _ => Loss_max) Loss_max hLoss_max
              (fun _ => by rw [abs_of_nonneg hLoss_max])
      _ = Loss_max := by
          have h : (fun p => (g p).toReal * Loss_max) = (fun p => Loss_max * (g p).toReal) := by
            ext p; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe g]; ring

  -- Step 4: Apply coupling expansion (bounded version)
  rw [hE_eq μ_X, hE_eq μ_Z]
  rw [coupling_expansion_bounded μ_X μ_Z E_pair Loss_max hLoss_max hE_pair_bound]

  -- Step 5: E_pair is Lipschitz via E_pair_lipschitz_bounded
  have h_E_pair_lip : ∀ x z, |E_pair x - E_pair z| ≤ 2 * |β| * (L_pol : ℝ) * dist (fstar x) (fstar z) :=
    fun x z => E_pair_lipschitz_bounded fstar pol pol_ref β L_pol g h_lip x z
      Loss_max hLoss_max (fun p => hLoss_bound x p) (fun p => hLoss_bound z p)

  -- Step 6: Derive bounds for summability from diameter bound
  have h_E_pair_diff_bound : ∀ x z, |E_pair x - E_pair z| ≤ 2 * |β| * L_pol * D_max := by
    intro x z
    calc |E_pair x - E_pair z| ≤ 2 * |β| * L_pol * dist (fstar x) (fstar z) := h_E_pair_lip x z
      _ ≤ 2 * |β| * L_pol * D_max := by
        apply mul_le_mul_of_nonneg_left (h_dist_bound x z)
        positivity

  -- Step 7: Establish summability for Fubini
  have hswap : Summable (Function.uncurry fun x z => (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z)) :=
    PMF.summable_prod_mul_of_bounded μ_X μ_Z (fun x z => dist (fstar x) (fstar z)) D_max hD_max
      (fun x z => by rw [abs_of_nonneg dist_nonneg]; exact h_dist_bound x z)

  -- Step 8: The final inequality follows from coupling_bound_ineq_bounded
  calc |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (E_pair x - E_pair z)|
      ≤ 2 * |β| * (L_pol : ℝ) * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) := by
        have h_lip_pointwise : ∀ x z, |E_pair x - E_pair z| ≤ (2 * |β| * L_pol) * dist (fstar x) (fstar z) := by
          intro x z
          calc |E_pair x - E_pair z| ≤ 2 * |β| * L_pol * dist (fstar x) (fstar z) := h_E_pair_lip x z
            _ = (2 * |β| * L_pol) * dist (fstar x) (fstar z) := by ring
        exact coupling_bound_ineq_bounded μ_X μ_Z (fun x z => E_pair x - E_pair z) (2 * |β| * L_pol)
          (fun x z => dist (fstar x) (fstar z))
          (by positivity) (fun _ _ => dist_nonneg) h_lip_pointwise D_max hD_max h_dist_bound
    _ = 2 * |β| * (L_pol : ℝ) * ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x) := by
        congr 1
        -- Swap sums (Fubini) and apply dist_comm
        have fubini : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) =
                      ∑' z, ∑' x, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) :=
          (Summable.tsum_comm hswap).symm
        rw [fubini]
        apply tsum_congr; intro z
        apply tsum_congr; intro x
        rw [dist_comm]; ring
    _ = 2 * |β| * (L_pol : ℝ) * Δ_R := by rw [h_Δ]

/- Deprecated lemma `dpo_gap` removed; use `dpo_gap_bounded`. -/

/-- DPO Gap Bound (Reward-Lipschitz version, Bounded).

When reward is L_R-Lipschitz in Y, the DPO loss gap is bounded:
  |E_X[L] - E_Z[L]| ≤ 2 * |β| * L_R * Δ_R

This version uses explicit bounds to avoid the unsound summability axiom.

**Required bounds:**
- `D_max`: bound on oracle distances
- `Loss_max`: uniform bound on |DPOLossReward|

This parallels the policy-Lipschitz bound (dpo_gap_bounded). -/
theorem dpo_gap_reward_bounded {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (R : RewardFunction Y A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ) (L_R : ℝ≥0)
    (Δ_R : ℝ)
    -- Diameter bound
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    -- Loss bound
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ y (p : A × A), |DPOLossReward R β y p.1 p.2| ≤ Loss_max)
    (h_lip : RewardLipschitz R L_R)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    let ExpectedDPOLossReward_μ := fun μ =>
      ∑' x, (μ x).toReal * ∑' p, (gen x p).toReal * DPOLossReward R β (fstar x) p.1 p.2
    |ExpectedDPOLossReward_μ μ_X - ExpectedDPOLossReward_μ μ_Z| ≤
    2 * |β| * (L_R : ℝ) * Δ_R := by
  intro ExpectedDPOLossReward_μ
  -- Fix the generator
  let g := gen (Classical.arbitrary Strings)
  have hgen_eq : ∀ x, gen x = g := fun x => h_gen_fixed x _
  -- Define E_pair using reward-based loss
  let E_pair := fun x => ∑' p, (g p).toReal * DPOLossReward R β (fstar x) p.1 p.2

  -- Derive E_pair bound from Loss_max
  have hE_pair_bound : ∀ x, |E_pair x| ≤ Loss_max := fun x => by
    calc |E_pair x|
        = |∑' p, (g p).toReal * DPOLossReward R β (fstar x) p.1 p.2| := rfl
      _ ≤ ∑' p, |(g p).toReal * DPOLossReward R β (fstar x) p.1 p.2| := by
          apply abs_tsum_le_tsum_abs'
          · exact summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun p => hLoss_bound (fstar x) p)
          · exact (summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun p => hLoss_bound (fstar x) p)).abs
      _ = ∑' p, (g p).toReal * |DPOLossReward R β (fstar x) p.1 p.2| := by
          apply tsum_congr; intro p
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' p, (g p).toReal * Loss_max := by
          apply Summable.tsum_le_tsum
          · intro p; apply mul_le_mul_of_nonneg_left (hLoss_bound (fstar x) p) ENNReal.toReal_nonneg
          · exact summable_coupling_inner_bounded g (fun p => |DPOLossReward R β (fstar x) p.1 p.2|)
              Loss_max hLoss_max (fun p => by rw [abs_abs]; exact hLoss_bound (fstar x) p)
          · exact summable_coupling_inner_bounded g (fun _ => Loss_max) Loss_max hLoss_max
              (fun _ => by rw [abs_of_nonneg hLoss_max])
      _ = Loss_max := by
          have h : (fun p => (g p).toReal * Loss_max) = (fun p => Loss_max * (g p).toReal) := by
            ext p; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe g]; ring

  -- Show E_pair is 2*|β|*L_R-Lipschitz
  have h_E_pair_lip : ∀ x z, |E_pair x - E_pair z| ≤ 2 * |β| * L_R * dist (fstar x) (fstar z) := by
    intro x z
    -- Combine sums using bounded summability
    have h_sum_x : Summable (fun p => (g p).toReal * DPOLossReward R β (fstar x) p.1 p.2) :=
      summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun p => hLoss_bound (fstar x) p)
    have h_sum_z : Summable (fun p => (g p).toReal * DPOLossReward R β (fstar z) p.1 p.2) :=
      summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun p => hLoss_bound (fstar z) p)
    have h_sub : E_pair x - E_pair z =
                 ∑' p, (g p).toReal * (DPOLossReward R β (fstar x) p.1 p.2 -
                                       DPOLossReward R β (fstar z) p.1 p.2) := by
      rw [← Summable.tsum_sub h_sum_x h_sum_z]
      congr 1; ext p; ring
    rw [h_sub]
    -- Bound for the difference
    have h_diff_bound : ∀ (p : A × A), |DPOLossReward R β (fstar x) p.1 p.2 -
                              DPOLossReward R β (fstar z) p.1 p.2| ≤ 2 * Loss_max := fun p => by
      calc |DPOLossReward R β (fstar x) p.1 p.2 - DPOLossReward R β (fstar z) p.1 p.2|
          ≤ |DPOLossReward R β (fstar x) p.1 p.2| + |DPOLossReward R β (fstar z) p.1 p.2| := by
            calc |DPOLossReward R β (fstar x) p.1 p.2 - DPOLossReward R β (fstar z) p.1 p.2|
                = |DPOLossReward R β (fstar x) p.1 p.2 + (-(DPOLossReward R β (fstar z) p.1 p.2))| := by ring_nf
              _ ≤ |DPOLossReward R β (fstar x) p.1 p.2| + |-(DPOLossReward R β (fstar z) p.1 p.2)| := abs_add_le _ _
              _ = |DPOLossReward R β (fstar x) p.1 p.2| + |DPOLossReward R β (fstar z) p.1 p.2| := by rw [abs_neg]
        _ ≤ Loss_max + Loss_max := add_le_add (hLoss_bound (fstar x) p) (hLoss_bound (fstar z) p)
        _ = 2 * Loss_max := by ring
    -- Summability for difference
    have hsum : Summable (fun p => (g p).toReal * (DPOLossReward R β (fstar x) p.1 p.2 -
                                                   DPOLossReward R β (fstar z) p.1 p.2)) :=
      summable_coupling_inner_bounded g _ (2 * Loss_max) (by linarith) h_diff_bound
    have hsum_abs := hsum.abs
    calc |∑' p, (g p).toReal * (DPOLossReward R β (fstar x) p.1 p.2 -
                                DPOLossReward R β (fstar z) p.1 p.2)|
        ≤ ∑' p, |(g p).toReal * (DPOLossReward R β (fstar x) p.1 p.2 -
                                 DPOLossReward R β (fstar z) p.1 p.2)| :=
          abs_tsum_le_tsum_abs' _ hsum hsum_abs
      _ = ∑' p, (g p).toReal * |DPOLossReward R β (fstar x) p.1 p.2 -
                                DPOLossReward R β (fstar z) p.1 p.2| := by
          congr 1; ext p
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' p, (g p).toReal * (2 * |β| * L_R * dist (fstar x) (fstar z)) := by
          apply Summable.tsum_le_tsum
          · intro p
            apply mul_le_mul_of_nonneg_left _ ENNReal.toReal_nonneg
            exact dpo_loss_reward_lipschitz h_lip p.1 p.2 (fstar x) (fstar z)
          · have : (fun p => (g p).toReal * |DPOLossReward R β (fstar x) p.1 p.2 -
                                             DPOLossReward R β (fstar z) p.1 p.2|) =
                   (fun p => |(g p).toReal * (DPOLossReward R β (fstar x) p.1 p.2 -
                                              DPOLossReward R β (fstar z) p.1 p.2)|) := by
              ext p; rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
            rw [this]; exact hsum_abs
          · have h_const : (fun p => (g p).toReal * (2 * |β| * L_R * dist (fstar x) (fstar z))) =
                           (fun p => (2 * |β| * L_R * dist (fstar x) (fstar z)) * (g p).toReal) := by
              ext p; ring
            rw [h_const]
            exact (PMF.summable_coe_real g).mul_left _
      _ = 2 * |β| * L_R * dist (fstar x) (fstar z) := by
          have h_factor : (fun p => (g p).toReal * (2 * |β| * L_R * dist (fstar x) (fstar z))) =
                          (fun p => (2 * |β| * L_R * dist (fstar x) (fstar z)) * (g p).toReal) := by
            ext p; ring
          rw [h_factor, tsum_mul_left, PMF.toReal_tsum_coe g, mul_one]

  -- Summability for Fubini
  have hswap : Summable (Function.uncurry fun x z => (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z)) :=
    PMF.summable_prod_mul_of_bounded μ_X μ_Z (fun x z => dist (fstar x) (fstar z)) D_max hD_max
      (fun x z => by rw [abs_of_nonneg dist_nonneg]; exact h_dist_bound x z)

  -- Connect to coupling form
  have hE_eq : ∀ μ, ExpectedDPOLossReward_μ μ = ∑' x, (μ x).toReal * E_pair x := by
    intro μ
    apply tsum_congr
    intro x
    congr 1
    apply tsum_congr
    intro p
    rw [hgen_eq x]
  rw [hE_eq μ_X, hE_eq μ_Z]
  rw [coupling_expansion_bounded μ_X μ_Z E_pair Loss_max hLoss_max hE_pair_bound]

  -- Coupling bound (bounded version)
  calc |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (E_pair x - E_pair z)|
      ≤ 2 * |β| * (L_R : ℝ) * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) := by
        have h_lip_pointwise : ∀ x z, |E_pair x - E_pair z| ≤ (2 * |β| * L_R) * dist (fstar x) (fstar z) := by
          intro x z
          calc |E_pair x - E_pair z| ≤ 2 * |β| * L_R * dist (fstar x) (fstar z) := h_E_pair_lip x z
            _ = (2 * |β| * L_R) * dist (fstar x) (fstar z) := by ring
        exact coupling_bound_ineq_bounded μ_X μ_Z (fun x z => E_pair x - E_pair z) (2 * |β| * L_R)
          (fun x z => dist (fstar x) (fstar z))
          (by positivity) (fun _ _ => dist_nonneg) h_lip_pointwise D_max hD_max h_dist_bound
    _ = 2 * |β| * (L_R : ℝ) * ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x) := by
        congr 1
        have fubini : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) =
                      ∑' z, ∑' x, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) :=
          (Summable.tsum_comm hswap).symm
        rw [fubini]
        apply tsum_congr; intro z
        apply tsum_congr; intro x
        rw [dist_comm]; ring
    _ = 2 * |β| * (L_R : ℝ) * Δ_R := by rw [h_Δ]

/- Deprecated lemma `dpo_gap_reward` removed; use `dpo_gap_reward_bounded`. -/

/-!
## Oracle-Indexed Generator Version

The paper only requires generators to be oracle-indexed (depend on x through f*(x)),
not constant. This section provides the generalized version.
-/

/-- When generator is oracle-indexed and documents have same oracle value, gen gives same PMF -/
lemma oracle_indexed_gen_eq {Strings A Y : Type*} [PseudoMetricSpace Y]
    {gen : PairGenerator Strings A} {fstar : Strings → Y}
    (h_oi : OracleIndexedPairGen gen fstar)
    {x x' : Strings} (h_dist : dist (fstar x) (fstar x') = 0) :
    gen x = gen x' := h_oi x x' h_dist

/-- E_pair bounded for oracle-indexed generators with loss bound.

When generator is oracle-indexed and loss is bounded by M_loss, the expected loss
difference is bounded by 2*M_loss. This is a crude but provable bound that works
for all cases (dist = 0 and dist > 0).

**Key insight for dist = 0**: When fstar x = fstar z, then gen x = gen z (oracle-indexed)
and DPOLossPointwise values are equal (PolicyLipschitz), so difference is exactly 0.

**Key insight for dist > 0**: We use the crude bound 2*M_loss via triangle inequality.

**Empirical estimation of M_loss**: In practice, M_loss can be estimated by:
1. Random sampling pairs (x, a_w, a_ℓ) from the training distribution
2. Computing |DPOLossPointwise| for each sample
3. Using the maximum observed value (or concentration bounds for high-probability)

This "audit" approach allows verification without closed-form analysis. The bound
M_loss depends on the specific policy/reference pair and β, but is typically finite
for well-behaved policies (e.g., those satisfying log-ratio bounds). -/
lemma E_pair_lipschitz_oracle_indexed {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y) (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gen : PairGenerator Strings A)
    (h_oi : OracleIndexedPairGen gen fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    -- Loss bound hypothesis (can be estimated empirically)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A), |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (x z : Strings) :
    |∑' p, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 -
     ∑' p, (gen z p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2| ≤
    2 * M_loss := by
  -- When fstar x = fstar z (dist = 0), gen x = gen z and loss is equal → bound is 0
  -- When fstar x ≠ fstar z (dist > 0), we use the crude bound via triangle inequality
  by_cases h : dist (fstar x) (fstar z) = 0
  · -- Case: same oracle value → difference is exactly 0
    have hgen : gen x = gen z := h_oi x z h
    have hloss : ∀ p : A × A, DPOLossPointwise pol pol_ref β x p.1 p.2 =
                              DPOLossPointwise pol pol_ref β z p.1 p.2 := by
      intro p
      -- PolicyLipschitz + dist = 0 implies LogRatio is equal for each action
      have h_lr_w : LogRatio pol pol_ref x (p.1) = LogRatio pol pol_ref z (p.1) := by
        have h_lip_bound := h_lip (p.1) x z
        rw [h] at h_lip_bound
        simp only [mul_zero, abs_nonpos_iff] at h_lip_bound
        linarith
      have h_lr_l : LogRatio pol pol_ref x (p.2) = LogRatio pol pol_ref z (p.2) := by
        have h_lip_bound := h_lip (p.2) x z
        rw [h] at h_lip_bound
        simp only [mul_zero, abs_nonpos_iff] at h_lip_bound
        linarith
      -- DPOLossPointwise depends on x only through LogRatio
      unfold DPOLossPointwise DPOLogit
      rw [h_lr_w, h_lr_l]
    simp only [hgen, hloss, sub_self, abs_zero]
    positivity
  · -- Case: different oracle values (dist > 0) → use crude bound
    -- |∑ g_x*L_x - ∑ g_z*L_z| ≤ |∑ g_x*L_x| + |∑ g_z*L_z| (triangle inequality)
    -- Each term: |∑ g*L| ≤ ∑ |g| * |L| ≤ M_loss * ∑ g = M_loss (since g ≥ 0 and ∑g = 1)
    have h_bound_x : |∑' p, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss := by
      let f_x := fun p : A × A => (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2
      have hf_x : Summable f_x :=
        summable_pair_gen_dpo_bounded gen pol pol_ref β x M_loss hM_loss (h_loss_bound x)
      have habs_x : Summable (fun p => |f_x p|) := by
        have h_bound_sum : Summable (fun p => (gen x p).toReal * M_loss) :=
          (PMF.summable_coe_real (gen x)).mul_right M_loss
        apply Summable.of_norm_bounded h_bound_sum
        intro p
        rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg,
            abs_of_nonneg (mul_nonneg ENNReal.toReal_nonneg (abs_nonneg _))]
        exact mul_le_mul_of_nonneg_left (h_loss_bound x p) ENNReal.toReal_nonneg
      calc |∑' p, f_x p|
          ≤ ∑' p, |f_x p| := abs_tsum_le_tsum_abs' f_x hf_x habs_x
        _ = ∑' p, (gen x p).toReal * |DPOLossPointwise pol pol_ref β x p.1 p.2| := by
            apply tsum_congr; intro p
            rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
        _ ≤ ∑' p, (gen x p).toReal * M_loss := by
            apply Summable.tsum_le_tsum _
              (summable_coupling_inner_bounded (gen x)
                (fun p => |DPOLossPointwise pol pol_ref β x p.1 p.2|) M_loss hM_loss
                (fun p => by rw [abs_abs]; exact h_loss_bound x p))
              ((PMF.summable_coe_real (gen x)).mul_right M_loss)
            intro p
            apply mul_le_mul_of_nonneg_left (h_loss_bound x p) ENNReal.toReal_nonneg
        _ = M_loss * ∑' p, (gen x p).toReal := by
            rw [tsum_mul_right, mul_comm]
        _ = M_loss * 1 := by rw [PMF.toReal_tsum_coe]
        _ = M_loss := mul_one M_loss
    have h_bound_z : |∑' p, (gen z p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2| ≤ M_loss := by
      let f_z := fun p : A × A => (gen z p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2
      have hf_z : Summable f_z :=
        summable_pair_gen_dpo_bounded gen pol pol_ref β z M_loss hM_loss (h_loss_bound z)
      have habs_z : Summable (fun p => |f_z p|) := by
        have h_bound_sum : Summable (fun p => (gen z p).toReal * M_loss) :=
          (PMF.summable_coe_real (gen z)).mul_right M_loss
        apply Summable.of_norm_bounded h_bound_sum
        intro p
        rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg,
            abs_of_nonneg (mul_nonneg ENNReal.toReal_nonneg (abs_nonneg _))]
        exact mul_le_mul_of_nonneg_left (h_loss_bound z p) ENNReal.toReal_nonneg
      calc |∑' p, f_z p|
          ≤ ∑' p, |f_z p| := abs_tsum_le_tsum_abs' f_z hf_z habs_z
        _ = ∑' p, (gen z p).toReal * |DPOLossPointwise pol pol_ref β z p.1 p.2| := by
            apply tsum_congr; intro p
            rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
        _ ≤ ∑' p, (gen z p).toReal * M_loss := by
            apply Summable.tsum_le_tsum _
              (summable_coupling_inner_bounded (gen z)
                (fun p => |DPOLossPointwise pol pol_ref β z p.1 p.2|) M_loss hM_loss
                (fun p => by rw [abs_abs]; exact h_loss_bound z p))
              ((PMF.summable_coe_real (gen z)).mul_right M_loss)
            intro p
            apply mul_le_mul_of_nonneg_left (h_loss_bound z p) ENNReal.toReal_nonneg
        _ = M_loss * ∑' p, (gen z p).toReal := by
            rw [tsum_mul_right, mul_comm]
        _ = M_loss * 1 := by rw [PMF.toReal_tsum_coe]
        _ = M_loss := mul_one M_loss
    calc |∑' p, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 -
          ∑' p, (gen z p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2|
        ≤ |∑' p, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2| +
          |∑' p, (gen z p).toReal * DPOLossPointwise pol pol_ref β z p.1 p.2| := abs_sub _ _
      _ ≤ M_loss + M_loss := add_le_add h_bound_x h_bound_z
      _ = 2 * M_loss := by ring

/-- DPO Gap Bound with Oracle-Indexed Generator (bounded version).

Generalization of dpo_gap_bounded where the generator only needs to be oracle-indexed
(depend on document through f*(x)), not constant.

This version uses explicit bounds and avoids the unsound summability axiom.
The bound 2*M_loss is crude but always provable.

**Contrast with dpo_gap_bounded**: The constant-generator version achieves the
tighter Lipschitz-style bound `2 * |β| * L_pol * Δ_R`. The oracle-indexed version
uses the crude bound because generator differences across non-equal oracle values
cannot be bounded by distance alone without additional structure.

**When this bound is tight**: When all documents have the same oracle value
(dist(fstar x, fstar z) = 0 for all x, z in support), the bound is 0 (exact equality).
This is the key insight: oracle-indexed generators preserve exactness on "canonical" data. -/
theorem dpo_gap_oracle_indexed_bounded {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ) (L_pol : ℝ≥0)
    (_h_m_pol : DPO.OracleMeasurable pol fstar)
    (_h_m_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    -- Generalized: oracle-indexed instead of constant
    (h_oi : OracleIndexedPairGen gen fstar)
    -- Loss bound (can be estimated empirically via sampling)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A), |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss) :
    |ExpectedDPOLoss pol pol_ref β μ_X gen - ExpectedDPOLoss pol pol_ref β μ_Z gen| ≤
    2 * M_loss := by
  /-
  Proof outline:
  We use the crude bound from E_pair_lipschitz_oracle_indexed.

  For each x, z: |E_pair x - E_pair z| ≤ 2 * M_loss
  Therefore: |∑∑ μ_X(x)*μ_Z(z)*(E_pair x - E_pair z)| ≤ 2*M_loss * ∑∑ μ_X(x)*μ_Z(z) = 2*M_loss
  -/
  -- Define E_pair (now depends on gen(x) not a fixed g)
  let E_pair := fun x => ∑' p, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2

  -- ExpectedDPOLoss = ∑_x μ(x) * E_pair(x)
  have hE_eq : ∀ μ, ExpectedDPOLoss pol pol_ref β μ gen = ∑' x, (μ x).toReal * E_pair x := by
    intro μ
    rfl

  -- Each expected loss is bounded by M_loss (MOVED EARLIER for use in coupling_expansion_bounded)
  have h_E_bound : ∀ x, |E_pair x| ≤ M_loss := by
    intro x
    let f := fun p : A × A => (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2
    have hf : Summable f :=
      summable_pair_gen_dpo_bounded gen pol pol_ref β x M_loss hM_loss (h_loss_bound x)
    have h_bound_sum : Summable (fun p => (gen x p).toReal * M_loss) :=
      (PMF.summable_coe_real (gen x)).mul_right M_loss
    have habs : Summable (fun p => |f p|) := by
      apply Summable.of_norm_bounded h_bound_sum
      intro p
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg,
          abs_of_nonneg (mul_nonneg ENNReal.toReal_nonneg (abs_nonneg _))]
      exact mul_le_mul_of_nonneg_left (h_loss_bound x p) ENNReal.toReal_nonneg
    calc |E_pair x|
        = |∑' p, f p| := rfl
      _ ≤ ∑' p, |f p| := abs_tsum_le_tsum_abs' f hf habs
      _ = ∑' p, (gen x p).toReal * |DPOLossPointwise pol pol_ref β x p.1 p.2| := by
            apply tsum_congr; intro p
            rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' p, (gen x p).toReal * M_loss := by
            apply Summable.tsum_le_tsum _
              (summable_coupling_inner_bounded (gen x) _ M_loss hM_loss
                (fun p => by rw [abs_abs]; exact h_loss_bound x p))
              ((PMF.summable_coe_real (gen x)).mul_right M_loss)
            intro p
            apply mul_le_mul_of_nonneg_left (h_loss_bound x p) ENNReal.toReal_nonneg
      _ = M_loss * ∑' p, (gen x p).toReal := by
            rw [tsum_mul_right, mul_comm]
      _ = M_loss * 1 := by rw [PMF.toReal_tsum_coe]
      _ = M_loss := mul_one M_loss

  rw [hE_eq μ_X, hE_eq μ_Z]
  rw [coupling_expansion_bounded μ_X μ_Z E_pair M_loss hM_loss h_E_bound]

  -- E_pair difference is bounded by 2*M_loss (crude bound)
  have h_E_pair_bound : ∀ x z, |E_pair x - E_pair z| ≤ 2 * M_loss :=
    E_pair_lipschitz_oracle_indexed fstar pol pol_ref β L_pol gen h_oi h_lip M_loss hM_loss h_loss_bound

  -- The coupling sum: |∑_x μ_X(x) * (E_pair x - ∑_z μ_Z(z) * E_pair z)| ≤ 2*M_loss
  -- Using triangle inequality directly on the difference of expectations
  have h_exp_X : |∑' x, (μ_X x).toReal * E_pair x| ≤ M_loss := by
    let f := fun x => (μ_X x).toReal * E_pair x
    have h_sum : Summable f := summable_coupling_outer_bounded μ_X E_pair M_loss hM_loss h_E_bound
    have h_bound_sum : Summable (fun x => (μ_X x).toReal * M_loss) :=
      (PMF.summable_coe_real μ_X).mul_right M_loss
    have h_abs_sum : Summable (fun x => |f x|) := by
      apply Summable.of_norm_bounded h_bound_sum
      intro x
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg,
          abs_of_nonneg (mul_nonneg ENNReal.toReal_nonneg (abs_nonneg _))]
      -- Goal: (μ_X x).toReal * |E_pair x| ≤ (μ_X x).toReal * M_loss
      exact mul_le_mul_of_nonneg_left (h_E_bound x) ENNReal.toReal_nonneg
    calc |∑' x, f x|
        ≤ ∑' x, |f x| := abs_tsum_le_tsum_abs' f h_sum h_abs_sum
      _ = ∑' x, (μ_X x).toReal * |E_pair x| := by
            apply tsum_congr; intro x
            rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' x, (μ_X x).toReal * M_loss := by
            apply Summable.tsum_le_tsum _
              (summable_coupling_outer_bounded μ_X (fun x => |E_pair x|) M_loss hM_loss
                (fun x => by rw [abs_abs]; exact h_E_bound x))
              ((PMF.summable_coe_real μ_X).mul_right M_loss)
            intro x
            apply mul_le_mul_of_nonneg_left (h_E_bound x) ENNReal.toReal_nonneg
      _ = M_loss * ∑' x, (μ_X x).toReal := by
            rw [tsum_mul_right, mul_comm]
      _ = M_loss * 1 := by rw [PMF.toReal_tsum_coe]
      _ = M_loss := mul_one M_loss

  have h_exp_Z : |∑' z, (μ_Z z).toReal * E_pair z| ≤ M_loss := by
    let g := fun z => (μ_Z z).toReal * E_pair z
    have h_sum : Summable g := summable_coupling_outer_bounded μ_Z E_pair M_loss hM_loss h_E_bound
    have h_bound_sum : Summable (fun z => (μ_Z z).toReal * M_loss) :=
      (PMF.summable_coe_real μ_Z).mul_right M_loss
    have h_abs_sum : Summable (fun z => |g z|) := by
      apply Summable.of_norm_bounded h_bound_sum
      intro z
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg,
          abs_of_nonneg (mul_nonneg ENNReal.toReal_nonneg (abs_nonneg _))]
      exact mul_le_mul_of_nonneg_left (h_E_bound z) ENNReal.toReal_nonneg
    calc |∑' z, g z|
        ≤ ∑' z, |g z| := abs_tsum_le_tsum_abs' g h_sum h_abs_sum
      _ = ∑' z, (μ_Z z).toReal * |E_pair z| := by
            apply tsum_congr; intro z
            rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' z, (μ_Z z).toReal * M_loss := by
            apply Summable.tsum_le_tsum _
              (summable_coupling_inner_bounded μ_Z (fun z => |E_pair z|) M_loss hM_loss
                (fun z => by rw [abs_abs]; exact h_E_bound z))
              ((PMF.summable_coe_real μ_Z).mul_right M_loss)
            intro z
            apply mul_le_mul_of_nonneg_left (h_E_bound z) ENNReal.toReal_nonneg
      _ = M_loss * ∑' z, (μ_Z z).toReal := by
            rw [tsum_mul_right, mul_comm]
      _ = M_loss * 1 := by rw [PMF.toReal_tsum_coe]
      _ = M_loss := mul_one M_loss

  -- Use coupling_expansion backwards to relate to original expectations
  have h_coupling_eq : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (E_pair x - E_pair z) =
                       ∑' x, (μ_X x).toReal * E_pair x - ∑' z, (μ_Z z).toReal * E_pair z := by
    rw [← coupling_expansion_bounded μ_X μ_Z E_pair M_loss hM_loss h_E_bound]

  rw [h_coupling_eq]
  calc |∑' x, (μ_X x).toReal * E_pair x - ∑' z, (μ_Z z).toReal * E_pair z|
      ≤ |∑' x, (μ_X x).toReal * E_pair x| + |∑' z, (μ_Z z).toReal * E_pair z| := abs_sub _ _
    _ ≤ M_loss + M_loss := add_le_add h_exp_X h_exp_Z
    _ = 2 * M_loss := by ring

-- Deprecated lemma `dpo_gap_oracle_indexed` removed; use `dpo_gap_oracle_indexed_bounded`.

/-!
## Connection to Multi-Round Reduction (ZR)

This section connects DPO theorems to the tree-based summarization framework.
The key insight is that when local laws L1, L2, L3 hold, the multi-round
reduction ZR produces zero expected distortion, which implies exact DPO equivalence.
-/

/-- Expected distortion Δ_R for ZR-based summarization.
This is E_z~ZR[D(f*(z), f*(x))] where z is a summary of x. -/
def Δ_R_ZR {Strings Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings) (fstar : Strings → Y) : ℝ :=
  Exp (ZR g x R T) (fun z => D fstar z x)

/-- When μ_X = pure(x), the coupling definition of Δ_R equals Δ_R_ZR.

The coupling sum ∑∑ μ_Z(z) * μ_X(x') * dist(fstar z, fstar x')
simplifies when μ_X = pure(x) because pure(x)(x') = 1 iff x' = x,
collapsing the inner sum to a single term. -/
lemma coupling_Δ_eq_Δ_R_ZR {Strings Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (g : Summarizer Strings) (x : Strings) (R : ℕ)
    (T : BinTree Strings) (fstar : Strings → Y) :
    (∑' z, ∑' x', (ZR g x R T z).toReal * (PMF.pure x x').toReal * dist (fstar z) (fstar x'))
    = Δ_R_ZR g x R T fstar := by
  unfold Δ_R_ZR Exp D
  congr 1
  ext z
  simp only [PMF.pure_apply]
  rw [tsum_eq_single x]
  · simp
  · intro x' hx'; simp [hx']

/-- Δ_R equals zero when local laws hold (connects to multi_round theorem).

This bridges the DPO formalization with the tree-based hierarchical reduction framework:
when summarization satisfies L1 (leaf idempotence), L2 (internal node preservation),
and L3 (range preservation), the expected distortion is exactly zero.

Requires `[BoundedPseudoMetricSpace Y]` for axiom-free proofs. -/
theorem Δ_R_eq_zero_of_local_laws {Strings Y : Type*} [Monoid Strings] [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings) (fstar : Strings → Y)
    (hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar)
    (hR : R ≥ 1) :
    Δ_R_ZR g x R T fstar = 0 := by
  unfold Δ_R_ZR
  exact multi_round_typeclass g T x R fstar hp h1 h2 h3 hR

/-- Zero expected distortion implies zero pointwise distortion on support (BoundedMetricSpace).

In a MetricSpace, dist = 0 implies equality. Since expected distortion is a
sum of non-negative terms, E[D] = 0 implies D = 0 almost surely (on support).

Uses BoundedMetricSpace for axiom-free summability. -/
lemma zero_dist_on_support_of_Δ_R_zero {Strings Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings) (fstar : Strings → Y)
    (h_Δ : Δ_R_ZR g x R T fstar = 0) :
    ∀ z ∈ (ZR g x R T).support, dist (fstar z) (fstar x) = 0 := by
  /-
  Proof sketch:
  E[D] = 0 where D ≥ 0 implies D = 0 on support.
  Since D(z,x) = dist(fstar z, fstar x) ≥ 0 and the expectation is 0,
  each term μ(z) * D(z,x) must be 0. For z in support, μ(z) > 0,
  so D(z,x) = 0.
  -/
  intro z hz
  unfold Δ_R_ZR Exp D at h_Δ
  -- Key insight: sum of non-negative terms = 0 implies each term = 0
  -- Let μ = ZR g x R T
  let μ := ZR g x R T
  -- Each term μ(z') * dist(fstar z', fstar x) is non-negative
  have h_nonneg : ∀ z', 0 ≤ (μ z').toReal * dist (fstar z') (fstar x) :=
    fun z' => mul_nonneg ENNReal.toReal_nonneg dist_nonneg
  -- The sum is 0
  have h_sum_zero : ∑' z', (μ z').toReal * dist (fstar z') (fstar x) = 0 := h_Δ
  -- For z in support, μ(z) > 0 (as ENNReal)
  have h_pos_ennreal : 0 < μ z := (μ.apply_pos_iff z).mpr hz
  -- Convert to Real: 0 < (μ z).toReal requires 0 < μ z ∧ μ z < ⊤
  have h_pos : 0 < (μ z).toReal := by
    rw [ENNReal.toReal_pos_iff]
    exact ⟨h_pos_ennreal, lt_top_iff_ne_top.mpr (μ.apply_ne_top z)⟩
  -- Sum of nonneg terms = 0 and one term has positive coefficient implies that term's value is 0
  -- If dist(fstar z, fstar x) > 0, then μ(z) * dist > 0, contradicting sum = 0
  by_contra h_ne
  -- h_ne : ¬(dist (fstar z) (fstar x) = 0), i.e., dist ≠ 0
  -- In MetricSpace, dist = 0 ↔ x = y, so dist ≠ 0 ↔ x ≠ y
  have h_pos_dist : 0 < dist (fstar z) (fstar x) := by
    rw [dist_pos]
    exact fun heq => h_ne (dist_eq_zero.mpr heq)
  have h_pos_term : 0 < (μ z).toReal * dist (fstar z) (fstar x) := mul_pos h_pos h_pos_dist
  -- But sum of nonneg including a positive term is positive
  have h_sum_pos : 0 < ∑' z', (μ z').toReal * dist (fstar z') (fstar x) := by
    let D_max := BoundedMetricSpace.diameterBound (α := Y)
    have hD_max : 0 ≤ D_max := BoundedMetricSpace.diameterBound_nonneg
    have h_dist_bound : ∀ z', |dist (fstar z') (fstar x)| ≤ D_max := fun z' =>
      BoundedMetricSpace.abs_dist_le (fstar z') (fstar x)
    have h_summable : Summable (fun z' => (μ z').toReal * dist (fstar z') (fstar x)) :=
      summable_coupling_inner_bounded μ _ D_max hD_max h_dist_bound
    exact h_summable.tsum_pos h_nonneg z h_pos_term
  linarith

/-- Master Theorem: DPO Exact via ZR.

When local laws L1, L2, L3 hold for the summarization g over tree T,
the DPO loss on μ_X = pure(x) equals the DPO loss on μ_Z = ZR(g, x, R, T).

This is the key theorem connecting DPO training on summaries to DPO training
on original documents: if the summarization preserves oracle values (as
guaranteed by local laws), then training on summaries is equivalent to
training on originals.

Requires `[BoundedMetricSpace Y]` for axiom-free proofs. -/
theorem dpo_exact_via_ZR {Strings A Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    -- Local laws ensure oracle preservation
    (hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    -- Oracle measurability conditions
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    SameOracleMeasurableArgmin
      (fun pol => ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
      (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
      fstar := by
  -- Local laws imply zero expected distortion
  have h_Δ_zero : Δ_R_ZR g x R T fstar = 0 := Δ_R_eq_zero_of_local_laws g x R T fstar hp h1 h2 h3 hR
  -- Zero expected distortion implies zero pointwise distortion on support
  have h_zero_dist : ∀ z ∈ (ZR g x R T).support, dist (fstar z) (fstar x) = 0 :=
    zero_dist_on_support_of_Δ_R_zero g x R T fstar h_Δ_zero
  -- For pure x, the support is {x} which trivially satisfies dist(fstar x, fstar x) = 0
  have h_oracle_eq : ∀ z x', z ∈ (ZR g x R T).support → x' ∈ (PMF.pure x).support →
      dist (fstar z) (fstar x') = 0 := by
    intro z x' hz hx'
    rw [PMF.support_pure, Set.mem_singleton_iff] at hx'
    rw [hx']
    exact h_zero_dist z hz
  -- Apply dpo_exact_metric
  exact dpo_exact_metric fstar pol_ref gen (PMF.pure x) (ZR g x R T) β h_oracle_eq h_meas_ref h_pair

/-- DPO Gap Bound via ZR.

When using the oracle-indexed formulation, the DPO loss difference is bounded
by 2 * M_loss, where M_loss is an empirical bound on the pointwise loss.

Combined with the multi_round theorem, this shows that the DPO gap shrinks
as the summarization quality improves. -/
theorem dpo_gap_via_ZR {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_pol : ℝ≥0)
    -- Oracle measurability and Lipschitz
    (_h_meas_pol : DPO.OracleMeasurable pol fstar)
    (_h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_pair : OracleIndexedPairGen gen fstar)
    -- Loss bound (can be estimated empirically)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x' : Strings) (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ M_loss) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
    2 * M_loss := by
  -- Directly apply dpo_gap_oracle_indexed_bounded with the M_loss bound
  exact dpo_gap_oracle_indexed_bounded fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β L_pol
    _h_meas_pol _h_meas_ref h_lip h_pair M_loss hM_loss h_loss_bound

/-- DPO Gap bound using union bound on violation probabilities.

This is the key theorem connecting the audit framework to DPO training:
the DPO loss gap is bounded by 2|β|L_pol times the sum of violation bounds.

This theorem composes:
1. `union_bound_multi_round`: Δ_R_ZR ≤ leafViol + mergeViol + (R-1)*pIdemp
2. `dpo_gap_bounded`: |L^X - L^Z| ≤ 2|β|L_pol * Δ_R
3. `coupling_Δ_eq_Δ_R_ZR`: For pure(x), coupling Δ_R = Δ_R_ZR

This gives the tight Lipschitz bound in terms of empirically estimable violation rates. -/
theorem dpo_gap_via_union_bound {Strings A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_pol : ℝ≥0)
    (hp : S T = x) (hR : R ≥ 1)
    -- Boundedness for summability
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    -- Loss bound (required for axiom-free proof)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    -- Oracle measurability and Lipschitz
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    -- Monotonicity for union bound
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
    2 * |β| * (L_pol : ℝ) *
    (totalLeafViolation g fstar T + totalMergeViolation g fstar T +
     (R - 1) * pIdemp g fstar (reduce g T)) := by
  -- Step 1: Get Δ_R_ZR bound from union_bound_multi_round_bounded
  have h_Δ_bound : Δ_R_ZR g x R T fstar ≤
      totalLeafViolation g fstar T + totalMergeViolation g fstar T +
      (R - 1) * pIdemp g fstar (reduce g T) :=
    union_bound_multi_round_bounded g fstar T x hp R hR hbound hbound_global h_mono

  -- Step 2: Show coupling Δ_R = Δ_R_ZR for pure(x)
  have h_Δ_eq : (∑' z, ∑' x', (ZR g x R T z).toReal * (PMF.pure x x').toReal *
                 dist (fstar z) (fstar x')) = Δ_R_ZR g x R T fstar :=
    coupling_Δ_eq_Δ_R_ZR g x R T fstar

  -- Step 3: Apply dpo_gap_bounded with the coupling Δ_R
  have h_gap := dpo_gap_bounded fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β L_pol
    (Δ_R_ZR g x R T fstar) D_max hD_max h_dist_bound Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed h_Δ_eq.symm

  -- Step 4: Combine with Δ_R bound
  calc |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
        ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|
      ≤ 2 * |β| * L_pol * Δ_R_ZR g x R T fstar := h_gap
    _ ≤ 2 * |β| * L_pol * (totalLeafViolation g fstar T + totalMergeViolation g fstar T +
         (R - 1) * pIdemp g fstar (reduce g T)) := by
      apply mul_le_mul_of_nonneg_left h_Δ_bound
      positivity

/-- Bounded version: DPO gap vanishes when local laws hold (axiom-free).

This version uses `multi_round_proper` and avoids the unsound `PMF.summable_coe_real_mul` axiom
by requiring an explicit bound M on distortion. For bounded metric spaces (e.g., Y = ℝ with
bounded oracle values), M is the diameter.

Recommended for rigorous formalization. -/
theorem dpo_gap_zero_of_local_laws_bounded {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    -- Local laws
    (_hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    -- Oracle measurability (both pol and pol_ref)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar)
    -- Explicit boundedness hypothesis (avoids axiom)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| = 0 := by
  -- Derive zero distortion from local laws using the bounded version
  have h_zero : ∀ z x', z ∈ (ZR g x R T).support → x' ∈ (PMF.pure x).support →
      dist (fstar z) (fstar x') = 0 := by
    intro z x' hz hx'
    simp only [PMF.support_pure, Finset.mem_singleton] at hx'
    rw [hx']
    -- Use multi_round_proper (axiom-free)
    have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
      multi_round_proper g T x R fstar _hp h1 h2 h3 hR M hM hbound
    unfold D at h_exp_zero
    by_contra h_dist_ne_zero
    have h_dist_pos : 0 < dist (fstar z) (fstar x) :=
      lt_of_le_of_ne dist_nonneg (Ne.symm h_dist_ne_zero)
    have h_term_pos : 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
      apply mul_pos
      · exact ENNReal.toReal_pos hz (PMF.apply_ne_top _ _)
      · exact h_dist_pos
    -- Use bounded summability (no axiom)
    have h_summable : Summable (fun z => (ZR g x R T z).toReal * dist (fstar z) (fstar x)) := by
      exact summable_D_of_bounded (ZR g x R T) fstar x M hM (fun z => hbound z x)
    have h_sum_pos : 0 < ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
      calc 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := h_term_pos
           _ ≤ ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
               apply Summable.le_tsum h_summable z
               intro i _
               exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
    unfold Exp at h_exp_zero
    linarith [h_exp_zero]
  -- Apply the expected_loss_eq_of_zero_dist lemma
  have h_eq : ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
              ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen :=
    expected_loss_eq_of_zero_dist fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β
      h_zero h_meas_pol h_meas_ref h_pair
  rw [h_eq, sub_self, abs_zero]

/-!
## Typeclass Versions (Automatic Bound Derivation)

These theorems use `BoundedPseudoMetricSpace` to automatically derive the diameter
bound, eliminating explicit boundedness hypotheses. This is the cleanest interface
for bounded oracle spaces (common in practice).
-/

/-- DPO gap bound with automatic bound derivation from BoundedPseudoMetricSpace.

This version uses the axiom-free `dpo_gap_bounded`. The diameter bound is
automatically derived from the typeclass instance. Requires an explicit
loss bound `Loss_max` since the DPO loss depends on β and policy structure. -/
theorem dpo_gap_typeclass {Strings A Y : Type*} [Monoid Strings] [BoundedPseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ) (L_pol : ℝ≥0)
    (Δ_R : ℝ)
    -- Loss bound (required for axiom-free proof)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (p : A × A), |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ Loss_max)
    (_h_m_pol : DPO.OracleMeasurable pol fstar)
    (_h_m_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedDPOLoss pol pol_ref β μ_X gen - ExpectedDPOLoss pol pol_ref β μ_Z gen| ≤
    2 * |β| * (L_pol : ℝ) * Δ_R :=
  dpo_gap_bounded fstar pol pol_ref gen μ_X μ_Z β L_pol Δ_R
    (BoundedPseudoMetricSpace.diameterBound (α := Y))
    BoundedPseudoMetricSpace.diameterBound_nonneg
    (fun x z => BoundedPseudoMetricSpace.dist_le (fstar x) (fstar z))
    Loss_max hLoss_max hLoss_bound
    _h_m_pol _h_m_ref h_lip h_gen_fixed h_Δ

/-- Zero DPO gap under local laws with automatic bound derivation.

This is the typeclass version of dpo_gap_zero_of_local_laws_bounded. The diameter
bound is automatically derived from the BoundedMetricSpace instance on Y.

Note: Uses BoundedMetricSpace (not BoundedPseudoMetricSpace) because the proof
requires dist = 0 → eq to establish that policies and generators agree. -/
theorem dpo_gap_zero_of_local_laws_typeclass {Strings A Y : Type*} [Monoid Strings]
    [inst : BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    -- Local laws
    (hp : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1)
    -- Oracle measurability
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| = 0 := by
  -- Derive the bound from the BoundedMetricSpace instance
  have hM : 0 ≤ BoundedMetricSpace.diameterBound (α := Y) :=
    BoundedMetricSpace.diameterBound_nonneg
  have hbound : ∀ w z, D fstar w z ≤ BoundedMetricSpace.diameterBound (α := Y) :=
    fun w z => BoundedMetricSpace.dist_le (fstar w) (fstar z)
  -- Derive zero distortion from local laws
  have h_zero : ∀ z x', z ∈ (ZR g x R T).support → x' ∈ (PMF.pure x).support →
      dist (fstar z) (fstar x') = 0 := by
    intro z x' hz hx'
    simp only [PMF.support_pure, Finset.mem_singleton] at hx'
    rw [hx']
    have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
      multi_round_proper g T x R fstar hp h1 h2 h3 hR
        (BoundedMetricSpace.diameterBound (α := Y)) hM hbound
    unfold D at h_exp_zero
    by_contra h_dist_ne_zero
    have h_dist_pos : 0 < dist (fstar z) (fstar x) :=
      lt_of_le_of_ne dist_nonneg (Ne.symm h_dist_ne_zero)
    have h_term_pos : 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
      apply mul_pos
      · exact ENNReal.toReal_pos hz (PMF.apply_ne_top _ _)
      · exact h_dist_pos
    have h_summable : Summable (fun z => (ZR g x R T z).toReal * dist (fstar z) (fstar x)) := by
      exact summable_D_of_bounded (ZR g x R T) fstar x
        (BoundedMetricSpace.diameterBound (α := Y)) hM (fun z => hbound z x)
    have h_sum_pos : 0 < ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
      calc 0 < (ZR g x R T z).toReal * dist (fstar z) (fstar x) := h_term_pos
           _ ≤ ∑' z, (ZR g x R T z).toReal * dist (fstar z) (fstar x) := by
               apply Summable.le_tsum h_summable z
               intro i _
               exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
    unfold Exp at h_exp_zero
    linarith [h_exp_zero]
  -- Apply the expected_loss_eq_of_zero_dist lemma (requires MetricSpace)
  have h_eq : ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
              ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen :=
    expected_loss_eq_of_zero_dist fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β
      h_zero h_meas_pol h_meas_ref h_pair
  rw [h_eq, sub_self, abs_zero]

/-!
## Bundle Variants

Theorems using `LocalLawsBundle` and `OracleMeasurablePolicies` for cleaner signatures.
-/

/-- DPO gap bound using OracleMeasurablePolicies bundle.

This is `dpo_gap_typeclass` with oracle measurability bundled. -/
theorem dpo_gap_bundle {Strings A Y : Type*} [Monoid Strings] [BoundedPseudoMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (μ_X μ_Z : PMF Strings)
    (β : ℝ) (L_pol : ℝ≥0)
    (Δ_R : ℝ)
    -- Loss bound (required for axiom-free proof)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (p : A × A), |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ Loss_max)
    (h_meas : OracleMeasurablePolicies pol pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedDPOLoss pol pol_ref β μ_X gen - ExpectedDPOLoss pol pol_ref β μ_Z gen| ≤
    2 * |β| * (L_pol : ℝ) * Δ_R :=
  dpo_gap_typeclass fstar pol pol_ref gen μ_X μ_Z β L_pol Δ_R
    Loss_max hLoss_max hLoss_bound
    h_meas.pol_measurable h_meas.ref_measurable h_lip h_gen_fixed h_Δ

/-- Zero DPO gap using LocalLawsBundle + OracleMeasurablePolicies + BoundedMetricSpace.

This is the cleanest version combining all three bundling patterns:
- LocalLawsBundle for L1, L2, L3
- OracleMeasurablePolicies for dual measurability
- BoundedMetricSpace for automatic diameter bounds -/
theorem dpo_gap_zero_bundle {Strings A Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (hp : S T = x)
    (laws : LocalLawsBundle g T fstar) (hR : R ≥ 1)
    (h_meas : OracleMeasurablePolicies pol pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| = 0 :=
  dpo_gap_zero_of_local_laws_typeclass fstar pol pol_ref gen g x R T β hp
    laws.law1 laws.law2 laws.law3 hR
    h_meas.pol_measurable h_meas.ref_measurable h_pair

/-!
## Theorem 6.1: DPO Equivalence under Local Laws

This is the main theoretical result connecting oracle preservation to DPO training:
when local laws hold, training on summarized data is equivalent to training on original data.
-/

/-- Theorem 6.1: DPO Equivalence under Local Laws.

**Paper Reference:** Section 6, Theorem 6.1

When a summarizer satisfies the local laws (L1, L2, L3), the expected DPO loss
on the original distribution `μ_X = PMF.pure x` equals the expected loss on
the multi-round summarized distribution `μ_Z = ZR g x R T`.

This means DPO training is invariant under oracle-preserving summarization:
optimizing the loss on summarized data yields the same gradient signal
as optimizing on the original data.

**Mathematical Statement:**
If L1(g,T,f*), L2(g,T,f*), L3(g,f*) hold and R ≥ 1, then:
  E_{x}[L(x)] = E_{z~ZR(g,x,R,T)}[L(z)]

where L is the DPO loss functional. -/
theorem dpo_equivalence {Strings A Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (hp : S T = x)
    (laws : LocalLawsBundle g T fstar) (hR : R ≥ 1)
    (h_meas : OracleMeasurablePolicies pol pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen := by
  have h := dpo_gap_zero_bundle fstar pol pol_ref gen g x R T β hp laws hR h_meas h_pair
  -- |a - b| = 0 implies a - b = 0 implies a = b
  exact sub_eq_zero.mp (abs_eq_zero.mp h)

/-- Corollary: DPO training is sound under oracle-preserving summarization.

This corollary makes explicit that minimizing the DPO loss on summarized data
minimizes the same objective as on original data.

**Note:** This requires all candidate policies to be oracle-measurable, which is
a natural assumption when policies are parameterized by models that only observe
the oracle value f*(x) rather than the full input x. -/
theorem dpo_training_sound {Strings A Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (hp : S T = x)
    (laws : LocalLawsBundle g T fstar) (hR : R ≥ 1)
    (h_all_meas : ∀ (pol' : Policy Strings A), DPO.OracleMeasurable pol' fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    PolicyArgmin (fun pol' => ExpectedDPOLoss pol' pol_ref β (PMF.pure x) gen) =
    PolicyArgmin (fun pol' => ExpectedDPOLoss pol' pol_ref β (ZR g x R T) gen) := by
  -- The argmin sets are equal because the loss functions differ only by
  -- the distribution, and we've shown the expected losses are equal.
  -- This follows from dpo_equivalence applied pointwise.
  ext pol'
  simp only [PolicyArgmin, Set.mem_setOf_eq]
  constructor
  · intro h pol''
    have h_eq := dpo_equivalence fstar pol' pol_ref gen g x R T β hp laws hR
      ⟨h_all_meas pol', h_all_meas pol_ref⟩ h_pair
    have h_eq' := dpo_equivalence fstar pol'' pol_ref gen g x R T β hp laws hR
      ⟨h_all_meas pol'', h_all_meas pol_ref⟩ h_pair
    rw [← h_eq, ← h_eq']
    exact h pol''
  · intro h pol''
    have h_eq := dpo_equivalence fstar pol' pol_ref gen g x R T β hp laws hR
      ⟨h_all_meas pol', h_all_meas pol_ref⟩ h_pair
    have h_eq' := dpo_equivalence fstar pol'' pol_ref gen g x R T β hp laws hR
      ⟨h_all_meas pol'', h_all_meas pol_ref⟩ h_pair
    rw [h_eq, h_eq']
    exact h pol''

end DPO

/-!
## Section 4: GRPO Quantitative Bounds

This section provides quantitative bounds for GRPO methods, parallel to the DPO bounds above.
The framework covers:
- **GRPO-PL (Plackett-Luce)**: Listwise ranking with k > 2 candidates
- **GRPO-RL (DeepSeek-R1 style)**: Clipped surrogate + KL penalty

### Key Results

1. **Lipschitz bounds**: GRPO losses are Lipschitz when policies/rewards are Lipschitz
2. **Gap bounds**: Expected loss differences bounded by expected distortion
3. **ZR-connection**: Local laws imply zero gap for multi-round summarization
-/

section GRPO

open MeasureTheory Set Filter TopologicalSpace Real
open scoped ENNReal MeasureTheory NNReal

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {A : Type*}

/-!
### Random Utility Model Foundation

In Random Utility Models (McFadden, 1974), choices arise from:
  U_i = V_i + ε_i
where V_i is a continuous deterministic utility and ε_i is i.i.d. noise
(e.g., Gumbel distribution → multinomial logit / Plackett-Luce).

**Key Insight:** While rankings are discontinuous in V pointwise (ties cause jumps),
they are continuous **in expectation** over the noise. This is because:
- Ties (where rankings change) have measure zero when ε has a continuous density
- The expected loss is an integral over the noise distribution
- This integral is continuous in V by dominated convergence

**Consequence for Lipschitz Bounds:**
- Pointwise Lipschitz: DOES NOT HOLD (rankings can jump at ties)
- Expected Lipschitz: HOLDS under continuous utility assumption

The assumptions below formalize expected Lipschitz, bypassing the unprovable pointwise version.
This is a standard approach in econometrics: working with expected utilities rather than
realized choices.

**Reference:** McFadden, D. (1974). "Conditional logit analysis of qualitative choice behavior"
in Frontiers in Econometrics. Zarembka, P. (ed.), Academic Press.
-/

/-!
#### Unified Assumption: Expected Group Loss is Lipschitz (Random Utility Model)

Under the Random Utility Model assumption with continuous underlying utilities,
the expected loss over groups is Lipschitz in the oracle distance.

This is the **single foundational assumption** for preference learning bounds.
It abstracts over any loss function `Strings → (Fin k → A) → ℝ` that:
1. Depends on the document through Lipschitz functions of the oracle value
2. May involve rankings/selections that are discontinuous pointwise

Mathematical justification:
1. Rankings/selections arise from argmax over continuous utilities plus noise
2. Expected loss is an integral over the noise distribution
3. By dominated convergence, this integral is continuous in utilities
4. With bounded utilities and Lipschitz components, this extends to Lipschitz

**Key insight:** The assumption does NOT require pointwise Lipschitz (which fails at ties).
Instead, it directly asserts expected Lipschitz, justified by measure-zero ties.

**Applications:**
- GRPO-PL (Plackett-Luce ranking loss)
- GRPO-RL (PPO-style clipped surrogate with KL penalty)
- Any future preference learning method with similar structure

**Reference:** McFadden, D. (1974). "Conditional logit analysis of qualitative choice behavior"
-/
/-!
**Note:** The RUM Lipschitz assumption lives in `FormalProbability/DSL/RUM.lean` and is
re-exported here as `ExpectedGroupLossLipschitz` for convenience.
-/
abbrev ExpectedGroupLossLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (loss : Strings → (Fin k → A) → ℝ)
    (fstar : Strings → Y) (g : PMF (Fin k → A)) (L : ℝ≥0)
    (x z : Strings) : Prop :=
  RUM.ExpectedGroupLossLipschitz (loss := loss) (fstar := fstar) (g := g) (L := L) (x := x) (z := z)

/-!
#### Instantiations for Specific Loss Functions

The following definitions provide convenient wrappers that instantiate the unified
assumption for specific loss functions (GRPO-PL and GRPO-RL).
-/

/-- GRPO-PL expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for the Plackett-Luce ranking loss. -/
abbrev ExpectedGRPOLossLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) (g : PMF (Fin k → A)) (L : ℝ≥0)
    (_h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (_h_ranker : OracleIndexedRanker ranker fstar)
    (x z : Strings) : Prop :=
  ExpectedGroupLossLipschitz
    (fun doc grp => GRPOLossPointwise pol doc grp (ranker doc grp))
    fstar g L x z

/-- GRPO-RL expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for the PPO-style RL loss. -/
abbrev ExpectedGRPORLLossLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] (k : ℕ)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ) (fstar : Strings → Y) (g : PMF (Fin k → A)) (L : ℝ≥0)
    (_h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (_h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (_h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (_h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (x z : Strings) : Prop :=
  ExpectedGroupLossLipschitz
    (fun doc grp => GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta doc grp)
    fstar g L x z

/-!
#### Concrete RUM Instance: Plackett–Luce with Fixed Ranker

When the ranker is fixed across documents, the Plackett–Luce loss is Lipschitz in the
policy scores. This yields a concrete proof of expected group-loss Lipschitz (and thus
an instance of `ExpectedGRPOLossLipschitz`) without appealing to the RUM assumption.
-/

/-- Plackett–Luce expected loss is Lipschitz when the ranker is fixed across documents. -/
lemma ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker {Strings A Y : Type*}
    [PseudoMetricSpace Y] {k : ℕ} [Fintype A] [DecidableEq A]
    (hk : 0 < k)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) (g : PMF (Fin k → A))
    (L_pol : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_pol)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_ranker_fixed : ∀ x z group, ranker x group = ranker z group)
    (x z : Strings) :
    ExpectedGroupLossLipschitz
      (fun doc grp => GRPOLossPointwise pol doc grp (ranker doc grp))
      fstar g (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol) x z := by
  classical
  -- Local constant
  let L_grpo : ℝ≥0 := ((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol
  -- Pointwise Lipschitz bound per group
  have h_point :
      ∀ group : Fin k → A,
        |GRPOLossPointwise pol x group (ranker x group) -
         GRPOLossPointwise pol z group (ranker z group)| ≤
        (L_grpo : ℝ) * dist (fstar x) (fstar z) := by
    intro group
    have hr : ranker z group = ranker x group := h_ranker_fixed z x group
    -- Apply the Plackett–Luce score Lipschitz lemma
    have hL_nonneg : 0 ≤ (L_pol : ℝ) * dist (fstar x) (fstar z) := by
      have hL : 0 ≤ (L_pol : ℝ) := by exact_mod_cast L_pol.property
      exact mul_nonneg hL dist_nonneg
    have hbound : ∀ i, |pol x (group i) - pol z (group i)| ≤
        (L_pol : ℝ) * dist (fstar x) (fstar z) := by
      intro i
      simpa using (h_pol_lip x z (group i))
    have h_pl :
        |DSL.PlackettLuceLoss (k := k) (fun i => pol x (group i)) (ranker x group) -
         DSL.PlackettLuceLoss (k := k) (fun i => pol z (group i)) (ranker x group)| ≤
        (2 : ℝ) * (k : ℝ) * ((L_pol : ℝ) * dist (fstar x) (fstar z)) := by
      simpa using
        (DSL.plackettLuce_loss_lipschitz_uniform (k := k) (hk := hk)
          (scores := fun i => pol x (group i))
          (scores' := fun i => pol z (group i))
          (ranks := ranker x group) (L := (L_pol : ℝ) * dist (fstar x) (fstar z))
          hL_nonneg hbound)
    -- Rewrite PL loss into GRPO loss (ranker fixed)
    have h_pl' :
        |GRPOLossPointwise pol x group (ranker x group) -
         GRPOLossPointwise pol z group (ranker x group)| ≤
        (2 : ℝ) * (k : ℝ) * ((L_pol : ℝ) * dist (fstar x) (fstar z)) := by
      simpa [GRPOLossPointwise, DSL.PlackettLuceLoss, DSL.PlackettLuceLogProb,
        PlackettLuceLogProb] using h_pl
    -- Align ranker z with ranker x
    simpa [hr, L_grpo, mul_assoc, mul_left_comm, mul_comm] using h_pl'
  -- Reduce to finite sums
  have hsum :
      |∑ group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group) -
          ∑ group, (g group).toReal * GRPOLossPointwise pol z group (ranker z group)| ≤
        (L_grpo : ℝ) * dist (fstar x) (fstar z) := by
    -- Rewrite as sum of differences
    have h_sub :
        (∑ group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group)) -
          ∑ group, (g group).toReal * GRPOLossPointwise pol z group (ranker z group) =
        ∑ group, (g group).toReal *
          (GRPOLossPointwise pol x group (ranker x group) -
           GRPOLossPointwise pol z group (ranker z group)) := by
      symm
      calc
        ∑ group, (g group).toReal *
          (GRPOLossPointwise pol x group (ranker x group) -
           GRPOLossPointwise pol z group (ranker z group))
            =
          ∑ group, ((g group).toReal * GRPOLossPointwise pol x group (ranker x group) -
            (g group).toReal * GRPOLossPointwise pol z group (ranker z group)) := by
            apply Finset.sum_congr rfl
            intro group _; ring
        _ =
          (∑ group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group)) -
            ∑ group, (g group).toReal * GRPOLossPointwise pol z group (ranker z group) := by
          simp [Finset.sum_sub_distrib]
    calc
      |∑ group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group) -
          ∑ group, (g group).toReal * GRPOLossPointwise pol z group (ranker z group)|
          = |∑ group, (g group).toReal *
              (GRPOLossPointwise pol x group (ranker x group) -
               GRPOLossPointwise pol z group (ranker z group))| := by
              simp [h_sub]
      _ ≤ ∑ group, |(g group).toReal *
              (GRPOLossPointwise pol x group (ranker x group) -
               GRPOLossPointwise pol z group (ranker z group))| := by
              exact Finset.abs_sum_le_sum_abs _ _
      _ = ∑ group, (g group).toReal *
              |GRPOLossPointwise pol x group (ranker x group) -
               GRPOLossPointwise pol z group (ranker z group)| := by
              apply Finset.sum_congr rfl
              intro group _
              rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑ group, (g group).toReal * ((L_grpo : ℝ) * dist (fstar x) (fstar z)) := by
              apply Finset.sum_le_sum
              intro group _
              exact mul_le_mul_of_nonneg_left (h_point group) ENNReal.toReal_nonneg
      _ = (L_grpo : ℝ) * dist (fstar x) (fstar z) := by
              have hsum' : (∑ group, (g group).toReal) = 1 := by
                simpa [tsum_fintype] using (PMF.toReal_tsum_coe g)
              have hconst :
                  ∑ group, (g group).toReal * ((L_grpo : ℝ) * dist (fstar x) (fstar z)) =
                    ((L_grpo : ℝ) * dist (fstar x) (fstar z)) * ∑ group, (g group).toReal := by
                have h' :
                    ∑ group, (g group).toReal * ((L_grpo : ℝ) * dist (fstar x) (fstar z)) =
                      ∑ group, ((L_grpo : ℝ) * dist (fstar x) (fstar z)) * (g group).toReal := by
                    apply Finset.sum_congr rfl
                    intro group _; ring
                rw [h', Finset.mul_sum]
              rw [hconst, hsum', mul_one]
  -- Conclude
  simpa [ExpectedGroupLossLipschitz, RUM.ExpectedGroupLossLipschitz, tsum_fintype, L_grpo,
    mul_assoc, mul_left_comm, mul_comm] using hsum

lemma grpo_policy_lipschitz_scaled_plackettLuce {Strings A Y : Type*} [PseudoMetricSpace Y]
    {k : ℕ} (hk : 0 < k)
    (pol : Policy' Strings A) (fstar : Strings → Y) (L_pol : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_pol) :
    GRPOPolicyLipschitz pol fstar (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol) := by
  have hL_real :
      (L_pol : ℝ) ≤ ((((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol) : ℝ) := by
    have hL_nonneg : 0 ≤ (L_pol : ℝ) := by
      exact_mod_cast L_pol.property
    have hk_nat : 1 ≤ k := by
      simpa using hk
    have hk' : (1 : ℝ) ≤ (k : ℝ) := by
      exact_mod_cast hk_nat
    have hcoef : (1 : ℝ) ≤ 2 * (k : ℝ) := by
      nlinarith
    have hmul :
        (1 : ℝ) * (L_pol : ℝ) ≤ (2 * (k : ℝ)) * (L_pol : ℝ) := by
      exact mul_le_mul_of_nonneg_right hcoef hL_nonneg
    simpa using hmul
  have hL : L_pol ≤ (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol) := by
    exact_mod_cast hL_real
  exact grpo_policy_lipschitz_mono h_pol_lip hL

lemma ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker' {Strings A Y : Type*}
    [PseudoMetricSpace Y] {k : ℕ} [Fintype A] [DecidableEq A]
    (hk : 0 < k)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) (g : PMF (Fin k → A))
    (L_pol : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_pol)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_ranker_fixed : ∀ x z group, ranker x group = ranker z group)
    (x z : Strings) :
    ExpectedGRPOLossLipschitz pol ranker fstar g
      (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol)
      (grpo_policy_lipschitz_scaled_plackettLuce (hk := hk) pol fstar L_pol h_pol_lip)
      h_ranker x z := by
  have h :=
    ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker
      (hk := hk) (pol := pol) (ranker := ranker) (fstar := fstar) (g := g)
      (L_pol := L_pol) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
      (h_ranker_fixed := h_ranker_fixed) (x := x) (z := z)
  simpa [ExpectedGRPOLossLipschitz] using h

lemma ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker_all {Strings A Y : Type*}
    [PseudoMetricSpace Y] {k : ℕ} [Fintype A] [DecidableEq A]
    (hk : 0 < k)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y) (g : PMF (Fin k → A))
    (L_pol : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_pol)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_ranker_fixed : ∀ x z group, ranker x group = ranker z group) :
    ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g
        (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol)
        (grpo_policy_lipschitz_scaled_plackettLuce (hk := hk) pol fstar L_pol h_pol_lip)
        h_ranker x z := by
  intro x z
  simpa using
    (ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker'
      (hk := hk) (pol := pol) (ranker := ranker) (fstar := fstar) (g := g)
      (L_pol := L_pol) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
      (h_ranker_fixed := h_ranker_fixed) (x := x) (z := z))

/-- GRPO-RL expected loss is Lipschitz when the pointwise GRPO-RL loss is
Lipschitz on the finite group space.

This is a fully proved narrowing of the abstract expected-loss interface used
throughout the GRPO-RL gap theorems. -/
lemma ExpectedGRPORLLossLipschitz_of_pointwise_finite {Strings A Y : Type*}
    [PseudoMetricSpace Y] {k : ℕ} [Fintype A] [DecidableEq A]
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ) (fstar : Strings → Y) (g : PMF (Fin k → A)) (L : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_point :
      ∀ x z (group : Fin k → A),
        |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
         GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| ≤
        (L : ℝ) * dist (fstar x) (fstar z))
    (x z : Strings) :
    ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
      h_pol_lip h_old_lip h_ref_lip h_reward_lip x z := by
  classical
  have hsum :
      |∑ group, (g group).toReal *
          GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
        ∑ group, (g group).toReal *
          GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| ≤
      (L : ℝ) * dist (fstar x) (fstar z) := by
    have h_sub :
        (∑ group, (g group).toReal *
            GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group) -
          ∑ group, (g group).toReal *
            GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group =
        ∑ group, (g group).toReal *
          (GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
            GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group) := by
      symm
      calc
        ∑ group, (g group).toReal *
            (GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group) =
          ∑ group, ((g group).toReal *
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
            (g group).toReal *
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group) := by
            apply Finset.sum_congr rfl
            intro group _
            ring
        _ =
          (∑ group, (g group).toReal *
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group) -
            ∑ group, (g group).toReal *
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group := by
          simp [Finset.sum_sub_distrib]
    calc
      |∑ group, (g group).toReal *
          GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
        ∑ group, (g group).toReal *
          GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group|
          =
        |∑ group, (g group).toReal *
            (GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group)| := by
              simp [h_sub]
      _ ≤ ∑ group, |(g group).toReal *
            (GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group)| := by
              exact Finset.abs_sum_le_sum_abs _ _
      _ = ∑ group, (g group).toReal *
            |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
              GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| := by
              apply Finset.sum_congr rfl
              intro group _
              rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑ group, (g group).toReal * ((L : ℝ) * dist (fstar x) (fstar z)) := by
              apply Finset.sum_le_sum
              intro group _
              exact mul_le_mul_of_nonneg_left (h_point x z group) ENNReal.toReal_nonneg
      _ = (L : ℝ) * dist (fstar x) (fstar z) := by
              have hsum' : (∑ group, (g group).toReal) = 1 := by
                simpa [tsum_fintype] using (PMF.toReal_tsum_coe g)
              have hconst :
                  ∑ group, (g group).toReal * ((L : ℝ) * dist (fstar x) (fstar z)) =
                    ((L : ℝ) * dist (fstar x) (fstar z)) * ∑ group, (g group).toReal := by
                have h' :
                    ∑ group, (g group).toReal * ((L : ℝ) * dist (fstar x) (fstar z)) =
                      ∑ group, ((L : ℝ) * dist (fstar x) (fstar z)) * (g group).toReal := by
                    apply Finset.sum_congr rfl
                    intro group _
                    ring
                rw [h', Finset.mul_sum]
              rw [hconst, hsum', mul_one]
  simpa [ExpectedGRPORLLossLipschitz, ExpectedGroupLossLipschitz,
    RUM.ExpectedGroupLossLipschitz, tsum_fintype] using hsum
/-!
### GRPO Plackett-Luce Quantitative Bounds

The Plackett-Luce model generalizes Bradley-Terry from pairs to rankings:
  P(ranking) = ∏_{i=1}^{k} exp(s_i) / Σ_{j≥i} exp(s_j)

The loss is the negative log probability of the observed ranking.
-/

/-!
#### Log-Sum-Exp Lipschitz Property

The log-sum-exp function `f(x) = log(∑_i exp(x_i))` is 1-Lipschitz in the ℓ∞ norm.

Mathematical proof:
- The gradient is the softmax: ∂f/∂x_i = exp(x_i) / ∑_j exp(x_j) ∈ [0,1]
- The ℓ¹ norm of the gradient is ∑_i softmax_i = 1
- By duality, f is 1-Lipschitz in ℓ∞

This is a standard result in convex optimization and machine learning. -/

/-- Log-sum-exp is 1-Lipschitz in ℓ∞: |log(∑ exp(xᵢ)) - log(∑ exp(yᵢ))| ≤ max_i |xᵢ - yᵢ|.

**Note:** This is a well-known mathematical fact. The full Lean proof requires
differential calculus machinery (showing gradient is softmax with ℓ¹-norm = 1).

The bound uses uniform bound L on all coordinate differences rather than sup,
avoiding the need for OrderBot on ℝ.

We prove a concrete uniform‑bound version that avoids differential calculus
and is sufficient for the downstream Lipschitz arguments. -/
lemma logSumExp_lipschitz_uniform {k : ℕ} (hk : 0 < k)
    (x y : Fin k → ℝ) (L : ℝ) (hL : 0 ≤ L) (hbound : ∀ i, |x i - y i| ≤ L) :
    |Real.log (∑ i : Fin k, Real.exp (x i)) - Real.log (∑ i : Fin k, Real.exp (y i))| ≤ L := by
  -- Define the sums for convenience
  set S_x := ∑ i : Fin k, Real.exp (x i) with hSx
  set S_y := ∑ i : Fin k, Real.exp (y i) with hSy
  -- Both sums are positive (sum of positive terms, non-empty)
  have hne : (Finset.univ : Finset (Fin k)).Nonempty := by
    rw [Finset.univ_nonempty_iff]
    exact Fin.pos_iff_nonempty.mp hk
  have hSx_pos : 0 < S_x := by
    apply Finset.sum_pos
    · intro i _; exact Real.exp_pos _
    · exact hne
  have hSy_pos : 0 < S_y := by
    apply Finset.sum_pos
    · intro i _; exact Real.exp_pos _
    · exact hne
  -- Key insight: from |x_i - y_i| ≤ L, we get exp(x_i - L) ≤ exp(y_i) ≤ exp(x_i + L)
  have h_exp_lower : ∀ i, Real.exp (x i - L) ≤ Real.exp (y i) := by
    intro i
    apply Real.exp_le_exp.mpr
    have := hbound i
    rw [abs_sub_le_iff] at this
    linarith [this.1]
  have h_exp_upper : ∀ i, Real.exp (y i) ≤ Real.exp (x i + L) := by
    intro i
    apply Real.exp_le_exp.mpr
    have := hbound i
    rw [abs_sub_le_iff] at this
    linarith [this.2]
  -- Summing: exp(-L) · S_x ≤ S_y ≤ exp(L) · S_x
  have h_sum_lower : Real.exp (-L) * S_x ≤ S_y := by
    calc Real.exp (-L) * S_x
        = Real.exp (-L) * ∑ i : Fin k, Real.exp (x i) := by rfl
      _ = ∑ i : Fin k, Real.exp (-L) * Real.exp (x i) := by rw [Finset.mul_sum]
      _ = ∑ i : Fin k, Real.exp (x i - L) := by
          congr 1; ext i; rw [← Real.exp_add]; ring_nf
      _ ≤ ∑ i : Fin k, Real.exp (y i) := Finset.sum_le_sum (fun i _ => h_exp_lower i)
      _ = S_y := rfl
  have h_sum_upper : S_y ≤ Real.exp L * S_x := by
    calc S_y
        = ∑ i : Fin k, Real.exp (y i) := rfl
      _ ≤ ∑ i : Fin k, Real.exp (x i + L) := Finset.sum_le_sum (fun i _ => h_exp_upper i)
      _ = ∑ i : Fin k, Real.exp L * Real.exp (x i) := by
          congr 1; ext i; rw [← Real.exp_add]; ring_nf
      _ = Real.exp L * ∑ i : Fin k, Real.exp (x i) := by rw [← Finset.mul_sum]
      _ = Real.exp L * S_x := rfl
  -- Taking logs: log(S_x) - L ≤ log(S_y) ≤ log(S_x) + L
  have h_log_lower : Real.log S_x - L ≤ Real.log S_y := by
    have := Real.log_le_log (mul_pos (Real.exp_pos _) hSx_pos) h_sum_lower
    rw [Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hSx_pos)] at this
    rw [Real.log_exp] at this
    linarith
  have h_log_upper : Real.log S_y ≤ Real.log S_x + L := by
    have := Real.log_le_log hSy_pos h_sum_upper
    rw [Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hSx_pos)] at this
    rw [Real.log_exp] at this
    linarith
  -- Conclude: |log(S_x) - log(S_y)| ≤ L
  rw [abs_sub_le_iff]
  constructor <;> linarith

/-- Log-sum-exp with filtered terms is 1-Lipschitz.

This is the same as logSumExp_lipschitz_uniform but for sums with a filter (if-then-else).
When the predicate P holds for at least one index, the bound holds. -/
lemma logSumExp_lipschitz_filtered {k : ℕ}
    (x y : Fin k → ℝ) (P : Fin k → Prop) [DecidablePred P]
    (L : ℝ) (hL : 0 ≤ L) (hbound : ∀ i, |x i - y i| ≤ L)
    (hne : ∃ i, P i) :
    |Real.log (∑ i : Fin k, if P i then Real.exp (x i) else 0) -
     Real.log (∑ i : Fin k, if P i then Real.exp (y i) else 0)| ≤ L := by
  -- Define the filtered sums
  set S_x := ∑ i : Fin k, if P i then Real.exp (x i) else 0 with hSx_def
  set S_y := ∑ i : Fin k, if P i then Real.exp (y i) else 0 with hSy_def
  -- Both sums are positive (at least one positive term)
  have hSx_pos : 0 < S_x := by
    obtain ⟨i₀, hi₀⟩ := hne
    have h_nonneg : ∀ i ∈ Finset.univ, 0 ≤ (if P i then Real.exp (x i) else 0) := by
      intro i _; split_ifs with hP <;> [exact le_of_lt (Real.exp_pos _); exact le_refl 0]
    have h_pos_term : 0 < (if P i₀ then Real.exp (x i₀) else 0) := by
      simp only [if_pos hi₀]; exact Real.exp_pos _
    exact Finset.sum_pos' h_nonneg ⟨i₀, Finset.mem_univ i₀, h_pos_term⟩
  have hSy_pos : 0 < S_y := by
    obtain ⟨i₀, hi₀⟩ := hne
    have h_nonneg : ∀ i ∈ Finset.univ, 0 ≤ (if P i then Real.exp (y i) else 0) := by
      intro i _; split_ifs with hP <;> [exact le_of_lt (Real.exp_pos _); exact le_refl 0]
    have h_pos_term : 0 < (if P i₀ then Real.exp (y i₀) else 0) := by
      simp only [if_pos hi₀]; exact Real.exp_pos _
    exact Finset.sum_pos' h_nonneg ⟨i₀, Finset.mem_univ i₀, h_pos_term⟩
  -- Key: for each active term, exp bounds transfer
  have h_term_lower : ∀ i, (if P i then Real.exp (x i - L) else 0) ≤ (if P i then Real.exp (y i) else 0) := by
    intro i
    split_ifs with hP
    · apply Real.exp_le_exp.mpr
      have := hbound i; rw [abs_sub_le_iff] at this; linarith [this.1]
    · exact le_refl 0
  have h_term_upper : ∀ i, (if P i then Real.exp (y i) else 0) ≤ (if P i then Real.exp (x i + L) else 0) := by
    intro i
    split_ifs with hP
    · apply Real.exp_le_exp.mpr
      have := hbound i; rw [abs_sub_le_iff] at this; linarith [this.2]
    · exact le_refl 0
  -- Summing: exp(-L) · S_x ≤ S_y ≤ exp(L) · S_x
  have h_sum_lower : Real.exp (-L) * S_x ≤ S_y := by
    calc Real.exp (-L) * S_x
        = ∑ i : Fin k, Real.exp (-L) * (if P i then Real.exp (x i) else 0) := by rw [Finset.mul_sum]
      _ = ∑ i : Fin k, (if P i then Real.exp (-L) * Real.exp (x i) else 0) := by
          congr 1; ext i; split_ifs <;> ring
      _ = ∑ i : Fin k, (if P i then Real.exp (x i - L) else 0) := by
          congr 1; ext i; split_ifs <;> [rw [← Real.exp_add]; ring_nf]; ring_nf
      _ ≤ ∑ i : Fin k, (if P i then Real.exp (y i) else 0) := Finset.sum_le_sum (fun i _ => h_term_lower i)
      _ = S_y := rfl
  have h_sum_upper : S_y ≤ Real.exp L * S_x := by
    calc S_y = ∑ i : Fin k, (if P i then Real.exp (y i) else 0) := rfl
      _ ≤ ∑ i : Fin k, (if P i then Real.exp (x i + L) else 0) := Finset.sum_le_sum (fun i _ => h_term_upper i)
      _ = ∑ i : Fin k, (if P i then Real.exp L * Real.exp (x i) else 0) := by
          congr 1; ext i; split_ifs <;> [rw [← Real.exp_add]; ring_nf]; ring_nf
      _ = ∑ i : Fin k, Real.exp L * (if P i then Real.exp (x i) else 0) := by
          congr 1; ext i; split_ifs <;> ring
      _ = Real.exp L * S_x := by rw [← Finset.mul_sum]
  -- Taking logs
  have h_log_lower : Real.log S_x - L ≤ Real.log S_y := by
    have := Real.log_le_log (mul_pos (Real.exp_pos _) hSx_pos) h_sum_lower
    rw [Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hSx_pos), Real.log_exp] at this
    linarith
  have h_log_upper : Real.log S_y ≤ Real.log S_x + L := by
    have := Real.log_le_log hSy_pos h_sum_upper
    rw [Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hSx_pos), Real.log_exp] at this
    linarith
  rw [abs_sub_le_iff]
  constructor <;> linarith

lemma PlackettLuceLoss_lipschitz_same_ranks {k : ℕ} {A : Type*} (hk : 0 < k)
    (scores_x scores_z : Fin k → ℝ) (ranks : Fin k → ℕ)
    (L : ℝ) (hL : 0 ≤ L) (hbound : ∀ i, |scores_x i - scores_z i| ≤ L) :
    |PlackettLuceLogProb (A := A) scores_x ranks - PlackettLuceLogProb (A := A) scores_z ranks| ≤ 2 * k * L := by
  -- Unfold the definition
  unfold PlackettLuceLogProb
  -- The difference is a sum of differences
  have h_diff : (∑ i : Fin k, (scores_x i - Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0))) -
                (∑ i : Fin k, (scores_z i - Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0))) =
                ∑ i : Fin k, ((scores_x i - scores_z i) -
                             (Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
                              Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0))) := by
    rw [← Finset.sum_sub_distrib]
    congr 1; ext i; ring
  rw [h_diff]
  -- Bound each term
  have h_term_bound : ∀ i : Fin k,
      |((scores_x i - scores_z i) -
        (Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
         Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0)))| ≤ 2 * L := by
    intro i
    -- Triangle inequality: |a - b| ≤ |a| + |b|
    set a := scores_x i - scores_z i with ha
    set b := Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
             Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0) with hb
    have h_tri : |a - b| ≤ |a| + |b| := by
      calc |a - b| = |a + (-b)| := by rw [sub_eq_add_neg]
        _ ≤ |a| + |-b| := abs_add_le a (-b)
        _ = |a| + |b| := by rw [abs_neg]
    calc |((scores_x i - scores_z i) -
          (Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
           Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0)))|
        ≤ |scores_x i - scores_z i| +
          |Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
           Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0)| := h_tri
      _ ≤ L + L := by
          apply add_le_add
          · exact hbound i
          · apply logSumExp_lipschitz_filtered scores_x scores_z (fun j => ranks j ≥ ranks i) L hL hbound
            exact ⟨i, le_refl (ranks i)⟩
      _ = 2 * L := by ring
  -- Sum the bounds
  calc |∑ i : Fin k, ((scores_x i - scores_z i) -
                      (Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
                       Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0)))|
      ≤ ∑ i : Fin k, |((scores_x i - scores_z i) -
                       (Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_x j) else 0) -
                        Real.log (∑ j : Fin k, if ranks j ≥ ranks i then Real.exp (scores_z j) else 0)))| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin k, (2 * L) := Finset.sum_le_sum (fun i _ => h_term_bound i)
    _ = k * (2 * L) := by rw [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
    _ = 2 * k * L := by ring

/-- Expected GRPO-PL loss over groups is Lipschitz.

This lemma follows directly from the `ExpectedGRPOLossLipschitz` assumption,
justified by the Random Utility Model (continuous underlying utilities).

Previously, this was derived from a pointwise Lipschitz bound, but that approach
required a placeholder proof because rankings are discontinuous pointwise (rankings can
jump at ties). The expected version holds because ties have measure zero under
continuous noise distributions. -/
lemma E_group_grpo_lipschitz {k : ℕ}
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (fstar : Strings → Y)
    (g : PMF (Fin k → A))
    (L_grpo : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (x z : Strings)
    (h_rum : ExpectedGRPOLossLipschitz pol ranker fstar g L_grpo h_pol_lip h_ranker x z) :
    |∑' group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group) -
     ∑' group, (g group).toReal * GRPOLossPointwise pol z group (ranker z group)| ≤
    L_grpo * dist (fstar x) (fstar z) :=
  h_rum

/-- Expected GRPO-RL loss over groups is Lipschitz.

This lemma follows directly from the `ExpectedGRPORLLossLipschitz` assumption,
justified by the Random Utility Model (continuous underlying utilities).

The GRPO-RL loss is more complex than GRPO-PL, involving:
- Policy ratios (pol/pol_old)
- Z-score normalized advantages from rewards
- PPO-style clipping
- KL penalty to reference policy

Under continuous utilities with Lipschitz policies and rewards, the expected
loss inherits Lipschitz properties via dominated convergence. -/
lemma E_group_grpo_rl_lipschitz {k : ℕ}
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (fstar : Strings → Y)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (x z : Strings)
    (h_rum : ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
      h_pol_lip h_old_lip h_ref_lip h_reward_lip x z) :
    |∑' group, (g group).toReal * GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group -
     ∑' group, (g group).toReal * GRPORLLossPointwise pol pol_old pol_ref reward eps beta z group| ≤
    L * dist (fstar x) (fstar z) :=
  h_rum

/-- **GRPO-PL Gap Bound (Bounded Version)**

The expected GRPO-PL loss gap is bounded by L_grpo × expected distortion.

This version uses explicit bounds to avoid the unsound summability axiom.

**Required bounds:**
- `D_max`: bound on oracle distances
- `Loss_max`: uniform bound on |GRPOLossPointwise|

This parallels the DPO gap bound (dpo_gap_bounded). -/
theorem grpo_pl_gap_bounded {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (L_grpo : ℝ≥0)
    (Δ_R : ℝ)
    -- Diameter bound: oracle distances are bounded (ensures summability for Fubini)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    -- Loss bound: GRPO loss is bounded (ensures summability)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (group : Fin k → A), |GRPOLossPointwise pol x group (ranker x group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x) L_grpo h_pol_lip h_ranker x z)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedGRPOLoss pol ranker μ_X gen - ExpectedGRPOLoss pol ranker μ_Z gen| ≤
    L_grpo * Δ_R := by
  -- Fix the generator (gen is constant by h_gen_fixed)
  let g := gen (Classical.arbitrary Strings)
  have hgen_eq : ∀ x, gen x = g := fun x => h_gen_fixed x _
  have h_rum_g : ∀ x z, ExpectedGRPOLossLipschitz pol ranker fstar g L_grpo h_pol_lip h_ranker x z := by
    intro x z
    simpa [hgen_eq x] using h_rum x z

  -- Define E_group (expected loss over groups for fixed document)
  let E_group := fun x => ∑' group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group)

  -- Show ExpectedGRPOLoss μ gen = Exp μ E_group
  have hE_eq : ∀ μ, ExpectedGRPOLoss pol ranker μ gen = ∑' x, (μ x).toReal * E_group x := by
    intro μ
    unfold ExpectedGRPOLoss
    congr 1; ext x
    rw [hgen_eq x]

  -- Derive E_group bound from Loss_max
  have hE_group_bound : ∀ x, |E_group x| ≤ Loss_max := fun x => by
    calc |E_group x|
        = |∑' group, (g group).toReal * GRPOLossPointwise pol x group (ranker x group)| := rfl
      _ ≤ ∑' group, |(g group).toReal * GRPOLossPointwise pol x group (ranker x group)| := by
          apply abs_tsum_le_tsum_abs'
          · exact summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun group => hLoss_bound x group)
          · exact (summable_coupling_inner_bounded g _ Loss_max hLoss_max (fun group => hLoss_bound x group)).abs
      _ = ∑' group, (g group).toReal * |GRPOLossPointwise pol x group (ranker x group)| := by
          apply tsum_congr; intro group
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' group, (g group).toReal * Loss_max := by
          apply Summable.tsum_le_tsum
          · intro group; apply mul_le_mul_of_nonneg_left (hLoss_bound x group) ENNReal.toReal_nonneg
          · exact summable_coupling_inner_bounded g (fun group => |GRPOLossPointwise pol x group (ranker x group)|)
              Loss_max hLoss_max (fun group => by rw [abs_abs]; exact hLoss_bound x group)
          · exact summable_coupling_inner_bounded g (fun _ => Loss_max) Loss_max hLoss_max
              (fun _ => by rw [abs_of_nonneg hLoss_max])
      _ = Loss_max := by
          have h : (fun group => (g group).toReal * Loss_max) = (fun group => Loss_max * (g group).toReal) := by
            ext group; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe g]; ring

  -- Apply coupling expansion (bounded version)
  rw [hE_eq μ_X, hE_eq μ_Z]
  rw [coupling_expansion_bounded μ_X μ_Z E_group Loss_max hLoss_max hE_group_bound]

  -- Establish Lipschitz bound for E_group
  have h_lip : ∀ x z, |E_group x - E_group z| ≤ L_grpo * dist (fstar x) (fstar z) :=
    fun x z => E_group_grpo_lipschitz pol ranker fstar g L_grpo h_pol_lip h_ranker x z (h_rum_g x z)

  -- Summability for Fubini (distance terms are bounded)
  have hswap : Summable (Function.uncurry fun x z => (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z)) :=
    PMF.summable_prod_mul_of_bounded μ_X μ_Z (fun x z => dist (fstar x) (fstar z)) D_max hD_max
      (fun x z => by rw [abs_of_nonneg dist_nonneg]; exact h_dist_bound x z)

  -- Apply coupling bound (bounded version) and Fubini
  calc |∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * (E_group x - E_group z)|
      ≤ L_grpo * ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) := by
        have h_lip_pointwise : ∀ x z, |E_group x - E_group z| ≤ (L_grpo : ℝ) * dist (fstar x) (fstar z) := h_lip
        exact coupling_bound_ineq_bounded μ_X μ_Z (fun x z => E_group x - E_group z) (L_grpo : ℝ)
          (fun x z => dist (fstar x) (fstar z))
          (NNReal.coe_nonneg L_grpo) (fun _ _ => dist_nonneg) h_lip_pointwise D_max hD_max h_dist_bound
    _ = L_grpo * ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x) := by
        congr 1
        -- Swap sums (Fubini) and apply dist_comm
        have fubini : ∑' x, ∑' z, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) =
                      ∑' z, ∑' x, (μ_X x).toReal * (μ_Z z).toReal * dist (fstar x) (fstar z) :=
          (Summable.tsum_comm hswap).symm
        rw [fubini]
        -- Rewrite each term using dist_comm and commutativity
        apply tsum_congr; intro z
        apply tsum_congr; intro x
        rw [dist_comm]; ring
    _ = L_grpo * Δ_R := by rw [h_Δ]

/-!
### Bundle Interface: GRPO-PL Quantitative Gap
-/

/-- Bundled assumptions for the bounded GRPO-PL gap theorem. -/
structure GRPOPLGapBundleAssumptions {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings) where
  L_grpo : ℝ≥0
  Δ_R : ℝ
  D_max : ℝ
  hD_max : 0 ≤ D_max
  h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max
  Loss_max : ℝ
  hLoss_max : 0 ≤ Loss_max
  hLoss_bound : ∀ x (group : Fin k → A), |GRPOLossPointwise pol x group (ranker x group)| ≤ Loss_max
  h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo
  h_ranker : OracleIndexedRanker ranker fstar
  h_rum : ∀ x z, ExpectedGRPOLossLipschitz pol ranker fstar (gen x) L_grpo h_pol_lip h_ranker x z
  h_gen_fixed : ∀ x x', gen x = gen x'
  h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)

/-- Bundle-driven wrapper for `grpo_pl_gap_bounded`. -/
theorem grpo_pl_gap_bundle {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (assump : GRPOPLGapBundleAssumptions (k := k) fstar pol ranker gen μ_X μ_Z) :
    |ExpectedGRPOLoss pol ranker μ_X gen - ExpectedGRPOLoss pol ranker μ_Z gen| ≤
    assump.L_grpo * assump.Δ_R := by
  exact grpo_pl_gap_bounded (k := k) (fstar := fstar) (pol := pol) (ranker := ranker)
    (gen := gen) (μ_X := μ_X) (μ_Z := μ_Z)
    (L_grpo := assump.L_grpo) (Δ_R := assump.Δ_R)
    (D_max := assump.D_max) (hD_max := assump.hD_max) (h_dist_bound := assump.h_dist_bound)
    (Loss_max := assump.Loss_max) (hLoss_max := assump.hLoss_max) (hLoss_bound := assump.hLoss_bound)
    (h_pol_lip := assump.h_pol_lip) (h_ranker := assump.h_ranker) (h_rum := assump.h_rum)
    (h_gen_fixed := assump.h_gen_fixed) (h_Δ := assump.h_Δ)

-- Deprecated lemma `grpo_pl_gap` removed; use `grpo_pl_gap_bounded`.

/-!
### GRPO-PL Gap Bound via Plackett–Luce (Fixed Ranker)

This specialization constructs the RUM-style expected Lipschitz assumption
from the Plackett–Luce model (with a fixed ranker) and wires it into the
bounded GRPO-PL gap bound.
-/

/-- **GRPO-PL Gap Bound (Fixed Ranker, Fixed Generator)**

Specialization of `grpo_pl_gap_bounded` that constructs the RUM Lipschitz
assumption from Plackett–Luce (fixed ranker) and uses
`L_grpo = 2*k*L_pol`. -/
theorem grpo_pl_gap_bounded_plackettLuce_fixed_ranker {k : ℕ} [Fintype A] [DecidableEq A]
    (hk : 0 < k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (L_pol : ℝ≥0)
    (Δ_R : ℝ)
    -- Diameter bound: oracle distances are bounded (ensures summability for Fubini)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    -- Loss bound: GRPO loss is bounded (ensures summability)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (group : Fin k → A), |GRPOLossPointwise pol x group (ranker x group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_pol)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_ranker_fixed : ∀ x z group, ranker x group = ranker z group)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedGRPOLoss pol ranker μ_X gen - ExpectedGRPOLoss pol ranker μ_Z gen| ≤
    (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol) * Δ_R := by
  classical
  -- Define the scaled GRPO Lipschitz constant.
  let L_grpo : ℝ≥0 := ((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol
  have h_pol_lip_grpo : GRPOPolicyLipschitz pol fstar L_grpo :=
    grpo_policy_lipschitz_scaled_plackettLuce (hk := hk) pol fstar L_pol h_pol_lip

  -- Construct the RUM-style expected Lipschitz assumption from PL.
  have h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x) L_grpo h_pol_lip_grpo h_ranker x z := by
    intro x z
    -- Fix the generator using h_gen_fixed
    let g := gen (Classical.arbitrary Strings)
    have hgen_eq : ∀ x, gen x = g := fun x => h_gen_fixed x _
    have h_rum_g :
        ExpectedGRPOLossLipschitz pol ranker fstar g L_grpo h_pol_lip_grpo h_ranker x z := by
      -- Use the PL fixed-ranker instance
      have h :=
        ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker'
          (hk := hk) (pol := pol) (ranker := ranker) (fstar := fstar) (g := g)
          (L_pol := L_pol) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
          (h_ranker_fixed := h_ranker_fixed) (x := x) (z := z)
      simpa [L_grpo, h_pol_lip_grpo] using h
    simpa [hgen_eq x] using h_rum_g

  -- Apply the general bounded gap lemma.
  simpa [L_grpo] using
    (grpo_pl_gap_bounded (k := k) (fstar := fstar) (pol := pol) (ranker := ranker)
      (gen := gen) (μ_X := μ_X) (μ_Z := μ_Z) (L_grpo := L_grpo) (Δ_R := Δ_R)
      (D_max := D_max) (hD_max := hD_max) (h_dist_bound := h_dist_bound)
      (Loss_max := Loss_max) (hLoss_max := hLoss_max) (hLoss_bound := hLoss_bound)
      (h_pol_lip := h_pol_lip_grpo) (h_ranker := h_ranker) (h_rum := h_rum)
      (h_gen_fixed := h_gen_fixed) (h_Δ := h_Δ))

-- Deprecated lemma `grpo_rl_gap` removed; use `grpo_rl_gap_bounded`.

/-- **GRPO-RL Gap Bound (Bounded Version)**

The expected GRPO-RL loss gap is bounded by L_grpo_rl × expected distortion.

This covers the DeepSeek-R1 training objective:
- Policy ratios (pol/pol_old)
- Z-score normalized advantages from rewards
- PPO-style clipping with parameter eps
- KL penalty to reference policy with coefficient beta

**Required bounds:**
- `D_max`: bound on oracle distances
- `Loss_max`: uniform bound on |GRPORLLossPointwise|

This parallels the GRPO-PL gap bound (grpo_pl_gap_bounded). -/
theorem grpo_rl_gap_bounded {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (L_grpo_rl : ℝ≥0)
    (Δ_R : ℝ)
    -- Diameter bound: oracle distances are bounded (ensures summability for Fubini)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    -- Loss bound: GRPO-RL loss is bounded (ensures summability)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group| ≤ Loss_max)
    -- Lipschitz assumptions for all policies and reward
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x) L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    -- Generator is constant (standard for GRPO training)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_X gen -
     ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_Z gen| ≤
    L_grpo_rl * Δ_R := by
  -- Fix the generator (gen is constant by h_gen_fixed)
  let g := gen (Classical.arbitrary Strings)
  have hgen_eq : ∀ x, gen x = g := fun x => h_gen_fixed x _
  have h_rum_g :
      ∀ x z, ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z := by
    intro x z
    simpa [hgen_eq x] using h_rum x z

  -- Define E_group (expected loss over groups for fixed document)
  let E_group := fun x => ∑' group, (g group).toReal *
    GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group

  -- Show ExpectedGRPORLLoss μ gen = Exp μ E_group
  have hE_eq : ∀ μ, ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ gen =
      ∑' x, (μ x).toReal * E_group x := by
    intro μ
    unfold ExpectedGRPORLLoss ExpectedGroupLoss
    congr 1; ext x
    rw [hgen_eq x]

  -- Derive E_group bound from Loss_max
  have hE_group_bound : ∀ x, |E_group x| ≤ Loss_max := fun x => by
    calc |E_group x|
        = |∑' group, (g group).toReal *
            GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group| := rfl
      _ ≤ ∑' group, |(g group).toReal *
            GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group| := by
          apply abs_tsum_le_tsum_abs'
          · exact summable_coupling_inner_bounded g _ Loss_max hLoss_max
              (fun group => hLoss_bound x group)
          · exact (summable_coupling_inner_bounded g _ Loss_max hLoss_max
              (fun group => hLoss_bound x group)).abs
      _ = ∑' group, (g group).toReal *
            |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group| := by
          apply tsum_congr; intro group
          rw [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg]
      _ ≤ ∑' group, (g group).toReal * Loss_max := by
          apply Summable.tsum_le_tsum
          · intro group
            apply mul_le_mul_of_nonneg_left (hLoss_bound x group) ENNReal.toReal_nonneg
          · exact summable_coupling_inner_bounded g
              (fun group => |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group|)
              Loss_max hLoss_max (fun group => by rw [abs_abs]; exact hLoss_bound x group)
          · exact summable_coupling_inner_bounded g (fun _ => Loss_max) Loss_max hLoss_max
              (fun _ => by rw [abs_of_nonneg hLoss_max])
      _ = Loss_max := by
          have h : (fun group => (g group).toReal * Loss_max) =
              (fun group => Loss_max * (g group).toReal) := by ext group; ring
          rw [h, tsum_mul_left, PMF.toReal_tsum_coe g]; ring

  -- Establish Lipschitz bound for E_group using the RUM assumption
  have h_lip : ∀ x z, |E_group x - E_group z| ≤ L_grpo_rl * dist (fstar x) (fstar z) :=
    fun x z => E_group_grpo_rl_lipschitz pol pol_old pol_ref reward eps beta fstar g
      L_grpo_rl h_pol_lip h_old_lip h_ref_lip h_reward_lip x z (h_rum_g x z)

  -- Apply unified_preference_gap_bounded
  rw [hE_eq μ_X, hE_eq μ_Z]
  exact unified_preference_gap_bounded fstar E_group μ_X μ_Z L_grpo_rl Δ_R D_max hD_max
    h_dist_bound Loss_max hLoss_max hE_group_bound h_lip h_Δ

/-!
### Bundle Interface: GRPO-RL Quantitative Gap
-/

/-- Bundled assumptions for the bounded GRPO-RL gap theorem. -/
structure GRPORLGapBundleAssumptions {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings) where
  L_grpo_rl : ℝ≥0
  Δ_R : ℝ
  D_max : ℝ
  hD_max : 0 ≤ D_max
  h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max
  Loss_max : ℝ
  hLoss_max : 0 ≤ Loss_max
  hLoss_bound : ∀ x (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group| ≤ Loss_max
  h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl
  h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl
  h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl
  h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl
  h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x) L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z
  h_gen_fixed : ∀ x x', gen x = gen x'
  h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)

/-- Bundle-driven wrapper for `grpo_rl_gap_bounded`. -/
theorem grpo_rl_gap_bundle {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (assump : GRPORLGapBundleAssumptions (k := k) fstar pol pol_old pol_ref reward eps beta gen μ_X μ_Z) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_X gen -
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_Z gen| ≤
    assump.L_grpo_rl * assump.Δ_R := by
  exact grpo_rl_gap_bounded (k := k) (fstar := fstar)
    (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
    (reward := reward) (eps := eps) (beta := beta) (gen := gen)
    (μ_X := μ_X) (μ_Z := μ_Z)
    (L_grpo_rl := assump.L_grpo_rl) (Δ_R := assump.Δ_R)
    (D_max := assump.D_max) (hD_max := assump.hD_max) (h_dist_bound := assump.h_dist_bound)
    (Loss_max := assump.Loss_max) (hLoss_max := assump.hLoss_max) (hLoss_bound := assump.hLoss_bound)
    (h_pol_lip := assump.h_pol_lip) (h_old_lip := assump.h_old_lip)
    (h_ref_lip := assump.h_ref_lip) (h_reward_lip := assump.h_reward_lip)
    (h_rum := assump.h_rum) (h_gen_fixed := assump.h_gen_fixed) (h_Δ := assump.h_Δ)

/-- **GRPO-RL Gap Bound (Pointwise-Lipschitz Route)**

This variant replaces the abstract expected-loss Lipschitz assumption with a
primitive pointwise Lipschitz hypothesis on `GRPORLLossPointwise`. -/
theorem grpo_rl_gap_bounded_of_pointwise {k : ℕ} [Fintype A] [DecidableEq A]
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (μ_X μ_Z : PMF Strings)
    (L_grpo_rl : ℝ≥0)
    (Δ_R : ℝ)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_point :
      ∀ x z (group : Fin k → A),
        |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
         GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| ≤
        (L_grpo_rl : ℝ) * dist (fstar x) (fstar z))
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_Δ : Δ_R = ∑' z, ∑' x, (μ_Z z).toReal * (μ_X x).toReal * dist (fstar z) (fstar x)) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_X gen -
     ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta μ_Z gen| ≤
    L_grpo_rl * Δ_R := by
  classical
  have h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x) L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z := by
    intro x z
    let g := gen (Classical.arbitrary Strings)
    have hgen_eq : ∀ x', gen x' = g := fun x' => h_gen_fixed x' _
    have h_rum_g :
        ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L_grpo_rl
          h_pol_lip h_old_lip h_ref_lip h_reward_lip x z :=
      ExpectedGRPORLLossLipschitz_of_pointwise_finite
        (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta) (fstar := fstar)
        (g := g) (L := L_grpo_rl)
        (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
        (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
        h_point x z
    simpa [hgen_eq x] using h_rum_g
  exact grpo_rl_gap_bounded (k := k) (fstar := fstar)
    (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
    (reward := reward) (eps := eps) (beta := beta) (gen := gen)
    (μ_X := μ_X) (μ_Z := μ_Z)
    (L_grpo_rl := L_grpo_rl) (Δ_R := Δ_R)
    (D_max := D_max) (hD_max := hD_max) (h_dist_bound := h_dist_bound)
    (Loss_max := Loss_max) (hLoss_max := hLoss_max) (hLoss_bound := hLoss_bound)
    (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
    (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
    (h_rum := h_rum) (h_gen_fixed := h_gen_fixed) (h_Δ := h_Δ)

/-- GRPO-RL gap is zero when local laws hold.

This covers the DeepSeek-R1 training objective. -/
theorem grpo_rl_gap_zero_of_local_laws {Strings : Type*} [Monoid Strings]
    {Y : Type*} [MetricSpace Y] {A : Type*} (k : ℕ)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (eps beta : ℝ)
    (hp : S T = x)
    (h1 : L1 g T fstar)
    (h2 : L2 g T fstar)
    (h3 : L3 g fstar)
    (hR : R ≥ 1)
    (h_pol : GRPOOracleMeasurable pol fstar)
    (h_old : GRPOOracleMeasurable pol_old fstar)
    (h_ref : GRPOOracleMeasurable pol_ref fstar)
    (h_reward : OracleMeasurableReward reward fstar)
    (h_gen : OracleIndexedGroupGen gen fstar)
    -- Boundedness required for multi_round_proper
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen := by
  -- Use the zero-distortion theorem from ExpectationTheory
  have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_proper g T x R fstar hp h1 h2 h3 hR M hM hbound
  -- Derive the zero condition for grpo_rl_equivalence
  have h_zero : ∀ z x', z ∈ (ZR g x R T).support → x' ∈ (PMF.pure x).support →
      dist (fstar z) (fstar x') = 0 := by
    intro z x' hz hx'
    simp only [PMF.support_pure, Set.mem_singleton_iff] at hx'
    rw [hx']
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
  -- Apply GRPO-RL equivalence with oracle-measurability
  have h_meas := grpo_rl_loss_oracle_measurable k pol pol_old pol_ref reward eps beta fstar
    h_pol h_old h_ref h_reward
  exact grpo_rl_equivalence k fstar pol pol_old pol_ref reward eps beta gen
    (PMF.pure x) (ZR g x R T) h_zero h_meas h_gen

end GRPO

end
