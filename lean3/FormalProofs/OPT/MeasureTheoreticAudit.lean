/-
FormalProofs/MeasureTheoreticAudit.lean

Measure-Theoretic Audit Framework using Hoeffding's Inequality

This file provides a fully rigorous measure-theoretic formalization of the
empirical audit framework, connecting to Mathlib's Hoeffding inequality.

## Main Results

* `ViolationProb_eq_integral`: Connect discrete ViolationProb to integral
* `iidSampleMeasure`: Product measure for n iid samples
* `hasSubgaussianMGF_centeredViolationInd`: Violation indicators are sub-Gaussian
* `hoeffding_violation_rate_bound`: Hoeffding bound on empirical deviation
* `audit_certification_measure_theoretic`: Main certification theorem

## References

* Mathlib.Probability.Moments.SubGaussian for sub-Gaussian theory
* Mathlib.Probability.Independence.Basic for independence
* Mathlib.Probability.ProbabilityMassFunction for PMF infrastructure
-/

import FormalProofs.OPT.AuditCore
import FormalProofs.OPT.AuditBounds
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.ProbabilityMassFunction.Integrals

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal ENNReal MeasureTheory
open MeasureTheory Measure ProbabilityTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Section 1: PMF to Measure Bridge

Mathlib already provides `PMF.toMeasure` and `PMF.toMeasure.isProbabilityMeasure`.
We just need to connect our `Exp` and `ViolationProb` to integrals.
-/

/-- Connection between discrete expectation Exp and measure-theoretic integral.

For a PMF p and bounded measurable function f, the discrete expectation
equals the integral with respect to p.toMeasure. -/
lemma Exp_eq_integral [MeasurableSpace Strings] [MeasurableSingletonClass Strings]
    (p : PMF Strings) (f : Strings → ℝ) (hf : Integrable f p.toMeasure) :
    Exp p f = ∫ z, f z ∂p.toMeasure := by
  unfold Exp
  rw [PMF.integral_eq_tsum p f hf]
  -- For ℝ, smul is multiplication, so tsum matches
  simp only [smul_eq_mul]

/-- ViolationProb equals the integral of violationInd -/
lemma ViolationProb_eq_integral [MeasurableSpace Strings] [MeasurableSingletonClass Strings]
    (p : PMF Strings) (fstar : Strings → Y) (x : Strings)
    (hf : Integrable (fun z => violationInd fstar z x) p.toMeasure) :
    ViolationProb fstar p x = ∫ z, violationInd fstar z x ∂p.toMeasure := by
  unfold ViolationProb
  exact Exp_eq_integral p (fun z => violationInd fstar z x) hf

/-!
## Section 2: iid Sampling Space

For n iid samples from a distribution μ, we use Mathlib's product measure.
The sample space is `Fin n → Strings` with the product measure `μ^n`.
-/

/-- Product measure for n iid samples from μ -/
def iidSampleMeasure {α : Type*} [MeasurableSpace α] (μ : Measure α) (n : ℕ) :
    Measure (Fin n → α) :=
  Measure.pi (fun _ => μ)

/-- The i-th sample projection extracts the i-th coordinate -/
def sampleProjection {α : Type*} {n : ℕ} (i : Fin n) : (Fin n → α) → α :=
  fun ω => ω i

/-- Sample projections are measurable -/
lemma measurable_sampleProjection {α : Type*} [MeasurableSpace α] {n : ℕ} (i : Fin n) :
    Measurable (sampleProjection i : (Fin n → α) → α) :=
  measurable_pi_apply i

/-- The iid sample measure is a probability measure when μ is -/
instance iidSampleMeasure_isProbabilityMeasure {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [IsProbabilityMeasure μ] (n : ℕ) :
    IsProbabilityMeasure (iidSampleMeasure μ n) := by
  unfold iidSampleMeasure
  infer_instance

/-!
## Section 3: Independence of Sample Projections

The key property: under the product measure, the sample projections are independent.
-/

/-!
## Section 4: Violation Indicator Properties
-/

/-- violationInd is bounded in [0, 1] -/
lemma violationInd_mem_Icc (fstar : Strings → Y) (z x : Strings) :
    violationInd fstar z x ∈ Set.Icc 0 1 := by
  constructor
  · exact violationInd_nonneg (fstar := fstar) (z := z) (x := x)
  · exact violationInd_le_one (fstar := fstar) (z := z) (x := x)

/-- Centered violation indicator: V - p where p is the true violation probability -/
def centeredViolationInd (fstar : Strings → Y) (x : Strings) (p_true : ℝ)
    {n : ℕ} (i : Fin n) (ω : Fin n → Strings) : ℝ :=
  violationInd fstar (ω i) x - p_true

/-- Centered violation indicator is bounded in [-1, 1]

Since violationInd ∈ [0,1] and p_true ∈ [0,1], we have:
  centeredViolationInd = violationInd - p_true ∈ [-1, 1]
-/
lemma centeredViolationInd_mem_Icc (fstar : Strings → Y) (x : Strings)
    (p_true : ℝ) (hp_nonneg : 0 ≤ p_true) (hp_le_one : p_true ≤ 1)
    {n : ℕ} (i : Fin n) (ω : Fin n → Strings) :
    centeredViolationInd fstar x p_true i ω ∈ Set.Icc (-1) 1 := by
  unfold centeredViolationInd
  have hv := violationInd_mem_Icc fstar (ω i) x
  constructor
  · linarith [hv.1, hp_le_one]
  · linarith [hv.2, hp_nonneg]

/-!
## Section 5: Sub-Gaussian Property

The centered violation indicators are sub-Gaussian because they are bounded.
We use Mathlib's `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero`.
-/

/-- Under product measure, integral of function depending only on i-th coordinate
equals the marginal integral.

This is a standard property of product measures: if f depends only on the i-th
coordinate, then ∫ f(ω_i) dμ^n = ∫ f dμ.

NOTE: This is a standard result about product measures requiring careful handling of
`MeasureTheory.integral_pi_eq_inner` and related lemmas. -/
lemma integral_proj_eq_marginal {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [IsProbabilityMeasure μ] {n : ℕ} (i : Fin n) (f : α → ℝ)
    (hf : Integrable f μ) :
    ∫ ω, f (ω i) ∂(iidSampleMeasure μ n) = ∫ z, f z ∂μ := by
  -- The key insight: (Measure.pi μ).map (eval i) = μ i for probability measures
  -- Then: ∫ f(ω i) d(Measure.pi μ) = ∫ f d(map (eval i) (Measure.pi μ)) = ∫ f dμ
  unfold iidSampleMeasure
  -- measurePreserving_eval requires [∀ i, IsProbabilityMeasure (μ i)]
  letI h_prob : ∀ (j : Fin n), IsProbabilityMeasure ((fun (_ : Fin n) => μ) j) := fun _ => ‹_›
  have h_meas_pres : MeasurePreserving (Function.eval i) (Measure.pi (fun _ => μ)) μ := by
    convert measurePreserving_eval (μ := fun (_ : Fin n) => μ) i
  -- By integral_map: ∫ (f ∘ eval i) d(Measure.pi μ) = ∫ f d((Measure.pi μ).map (eval i))
  -- By measurePreserving: (Measure.pi μ).map (eval i) = μ
  -- Hence: ∫ f(ω i) d(Measure.pi μ) = ∫ f dμ
  have h_eval_meas : Measurable (Function.eval i : (Fin n → α) → α) := measurable_pi_apply i
  -- Need AEStronglyMeasurable f (map (eval i) (Measure.pi ...))
  -- By measurePreserving, this map equals μ, so we can use hf.aestronglyMeasurable
  have h_aesm : AEStronglyMeasurable f ((Measure.pi (fun _ => μ)).map (Function.eval i)) := by
    rw [h_meas_pres.map_eq]
    exact hf.aestronglyMeasurable
  calc ∫ ω, f (ω i) ∂Measure.pi (fun _ => μ)
      = ∫ ω, f (Function.eval i ω) ∂Measure.pi (fun _ => μ) := rfl
    _ = ∫ z, f z ∂(Measure.pi (fun _ => μ)).map (Function.eval i) := by
        rw [← integral_map h_eval_meas.aemeasurable h_aesm]
    _ = ∫ z, f z ∂μ := by rw [h_meas_pres.map_eq]

/-- Integrability is preserved under composition with coordinate projection.

If f is integrable under μ, then f ∘ (· i) is integrable under the product measure μ^n.
This follows from the fact that the marginal of a product measure is the original measure.

NOTE: Full proof requires showing Measure.map (· i) (iidSampleMeasure μ n) = μ
for probability measures, which involves Mathlib's `pi_map_eval` lemma. -/
lemma integrable_proj_of_integrable {α : Type*} [MeasurableSpace α]
    (μ : Measure α) [IsProbabilityMeasure μ] {n : ℕ} (i : Fin n) (f : α → ℝ)
    (hf : Integrable f μ) :
    Integrable (fun ω => f (ω i)) (iidSampleMeasure μ n) := by
  unfold iidSampleMeasure
  -- Use MeasurePreserving.integrable_comp_of_integrable
  letI h_prob : ∀ (j : Fin n), IsProbabilityMeasure ((fun (_ : Fin n) => μ) j) := fun _ => ‹_›
  have h_meas_pres : MeasurePreserving (Function.eval i) (Measure.pi (fun _ => μ)) μ := by
    convert measurePreserving_eval (μ := fun (_ : Fin n) => μ) i
  -- (f ∘ eval i) is integrable under Measure.pi by MeasurePreserving.integrable_comp_of_integrable
  exact h_meas_pres.integrable_comp_of_integrable hf

/-- The centered violation indicator has expectation zero by construction.

This is because E[V_i - p] = E[V_i] - p = p - p = 0 where p is the true violation probability. -/
lemma integral_centeredViolationInd_eq_zero
    [MeasurableSpace Strings] [MeasurableSingletonClass Strings]
    (μ : Measure Strings) [IsProbabilityMeasure μ]
    (fstar : Strings → Y) (x : Strings)
    (p_true : ℝ) (hp : p_true = ∫ z, violationInd fstar z x ∂μ)
    (hf : Integrable (fun z => violationInd fstar z x) μ)
    {n : ℕ} (hn : 0 < n) (i : Fin n) :
    ∫ ω, centeredViolationInd fstar x p_true i ω ∂(iidSampleMeasure μ n) = 0 := by
  unfold centeredViolationInd
  -- Use marginal property: ∫ V(ω_i) dμ^n = ∫ V dμ = p
  have h_marginal : ∫ ω, violationInd fstar (ω i) x ∂(iidSampleMeasure μ n) =
      ∫ z, violationInd fstar z x ∂μ :=
    integral_proj_eq_marginal μ i (fun z => violationInd fstar z x) hf
  -- Now compute: ∫ (V(ω_i) - p) = ∫ V(ω_i) - p = p - p = 0
  rw [integral_sub]
  · -- The integral of V(ω_i) equals p, and integral of constant p over probability measure is p
    have h_const : ∫ _, p_true ∂(iidSampleMeasure μ n) = p_true := by
      rw [integral_const, smul_eq_mul]
      have h_univ : (iidSampleMeasure μ n) Set.univ = 1 := measure_univ
      rw [Measure.real, h_univ, ENNReal.toReal_one, one_mul]
    rw [h_marginal, h_const, ← hp]
    ring
  · -- Integrability of V(ω_i) follows from integrability under marginal
    exact integrable_proj_of_integrable μ i (fun z => violationInd fstar z x) hf
  · -- Integrability of constant
    exact integrable_const p_true

/-!
## Section 6: Empirical Violation Rate
-/

/-- Empirical violation rate: average of n violation indicators -/
def empiricalViolationRate (fstar : Strings → Y) (x : Strings) (n : ℕ)
    (ω : Fin n → Strings) : ℝ :=
  (∑ i : Fin n, violationInd fstar (ω i) x) / n

/-- The deviation set {ω | |p̂(ω) - p| ≥ ε} is null-measurable.

This follows from the fact that empiricalViolationRate is a finite sum of indicator functions
composed with coordinate projections, and {x | c ≤ |f x|} is measurable for measurable f.

**Proof structure:**
1. MeasurableSet → NullMeasurableSet ✓
2. empiricalViolationRate is measurable (finite sum of indicator functions) ✓
3. Subtraction by constant, abs, division preserve measurability ✓
4. {|f| ≥ ε} is measurable for measurable f ✓

We discharge measurability via `[Countable Strings]`, which gives a discrete
measurable structure and makes the indicator condition measurable. -/
lemma deviationSet_nullMeasurable {n : ℕ} [MeasurableSpace Strings] [MeasurableSingletonClass Strings]
    [Countable Strings]
    (μ : Measure (Fin n → Strings)) (fstar : Strings → Y) (x : Strings) (p_true ε : ℝ) :
    NullMeasurableSet {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε} μ := by
  -- Convert to MeasurableSet first, then to NullMeasurableSet
  apply MeasurableSet.nullMeasurableSet
  -- The set {|f - c| ≥ ε} is measurable for measurable f: Real → Real
  -- empiricalViolationRate is measurable because it's a finite sum of indicator functions
  -- composed with measurable projections (measurable_pi_apply)
  have h_meas : Measurable (fun ω => |empiricalViolationRate fstar x n ω - p_true|) := by
    apply Measurable.abs
    apply Measurable.sub_const
    unfold empiricalViolationRate
    apply Measurable.div_const
    -- Finite sum of measurable functions is measurable
    apply Finset.measurable_sum
    intro i _
    -- violationInd fstar (ω i) x is constant on measurable fibers
    -- The key insight: violationInd is an indicator function (values in {0, 1})
    -- For MeasurableSingletonClass, we need the condition set to be measurable
    unfold violationInd D
    -- if-then-else of constants is measurable if the condition set is measurable
    apply Measurable.ite
    · -- {ω | dist (fstar (ω i)) (fstar x) > 0} is measurable
      -- For MeasurableSingletonClass, we can express this as a union of singletons.
      -- The set S = {z : Strings | dist (fstar z) (fstar x) > 0} is fixed, and we need
      -- {ω | ω i ∈ S} which is the preimage under the i-th projection.
      -- For MeasurableSingletonClass, S = ⋃ {z : z ∈ S} is a union of measurable singletons.
      --
      -- Technical approach: We show the preimage is measurable using that
      -- measurable_pi_apply gives measurable projections, and the target set
      -- is expressible in terms of the Borel structure on ℝ.
      --
      -- Key insight: The condition {dist (fstar (ω i)) (fstar x) > 0} defines a set
      -- in Strings, and the preimage under (· i) is measurable for any set in Strings
      -- when we have MeasurableSingletonClass (all singletons are measurable).
      have h_set : MeasurableSet {z : Strings | (0 : ℝ) < dist (fstar z) (fstar x)} := by
        -- With [MeasurableSingletonClass Strings] and [Countable Strings],
        -- we have DiscreteMeasurableSpace Strings, so all sets are measurable
        exact MeasurableSet.of_discrete
      exact h_set.preimage (measurable_pi_apply i)
    · exact measurable_const
    · exact measurable_const
  -- {f ≥ c} is measurable for measurable f
  exact h_meas measurableSet_Ici

/-- Connection between empirical rate and sum of centered indicators -/
lemma empiricalViolationRate_eq_p_plus_sum_centered
    (fstar : Strings → Y) (x : Strings) (p_true : ℝ) (n : ℕ) (hn : 0 < n)
    (ω : Fin n → Strings) :
    empiricalViolationRate fstar x n ω =
      p_true + (∑ i : Fin n, centeredViolationInd fstar x p_true i ω) / n := by
  unfold empiricalViolationRate centeredViolationInd
  have hn_ne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hn)
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  -- Rewrite the sum of (V_i - p) as (∑ V_i) - n*p
  have h_sum : ∑ i : Fin n, (violationInd fstar (ω i) x - p_true) =
      (∑ i : Fin n, violationInd fstar (ω i) x) - n * p_true := by
    rw [Finset.sum_sub_distrib]
    simp only [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
  rw [h_sum]
  field_simp
  ring

/-!
## Section 7: Main Hoeffding Application

This is the main technical result: applying Mathlib's Hoeffding inequality
to get a bound on the deviation of the empirical violation rate.
-/

/-- Core Hoeffding inequality for iid bounded random variables.

This lemma captures the standard Hoeffding bound: for n independent random variables
X_i ∈ [0,1] with common mean p, the empirical mean p̂ = (1/n)∑X_i satisfies:

  P(|p̂ - p| ≥ ε) ≤ 2 * exp(-2nε²)

## Proof Strategy (from Mathlib's sub-Gaussian machinery)
1. Center: Y_i = X_i - p has E[Y_i] = 0 and Y_i ∈ [-p, 1-p] ⊆ [-1, 1]
2. The range is 1, so Y_i is sub-Gaussian with c = (1/2)² = 1/4
3. Apply `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero` to get sub-Gaussian property
4. By `measure_sum_ge_le_of_iIndepFun`: P(∑Y_i ≥ t) ≤ exp(-t²/(2nc))
5. Setting t = nε and c = 1/4: P(∑Y_i ≥ nε) ≤ exp(-2nε²)
6. By symmetry (or applying to -Y_i): P(|∑Y_i| ≥ nε) ≤ 2exp(-2nε²)
7. Since |p̂ - p| = |∑Y_i|/n, we get P(|p̂ - p| ≥ ε) ≤ 2exp(-2nε²)

## Mathlib References
- `hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero`: bounded mean-zero → sub-Gaussian
- `measure_sum_ge_le_of_iIndepFun`: Hoeffding for independent sub-Gaussian sums
- `iIndepFun_pi`: coordinate projections from product measure are independent

## Status
Fully proved in this file by reducing to Mathlib sub-Gaussian concentration
results and performing the finite-sum/empirical-mean algebraic rewrites. -/
lemma hoeffding_iid_bounded {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (n : ℕ) (hn : 0 < n)
    (X : Fin n → Ω → ℝ)  -- n random variables
    (p : ℝ)              -- common mean
    (hX_bound : ∀ i ω, X i ω ∈ Set.Icc 0 1)           -- bounded in [0,1]
    (hX_mean : ∀ i, ∫ ω, X i ω ∂μ = p)                 -- common mean p
    (hX_meas : ∀ i, AEMeasurable (X i) μ)             -- measurability
    (hX_indep : iIndepFun X μ)                        -- independence
    (ε : ℝ) (hε : 0 < ε) :
    μ.real {ω | |((∑ i : Fin n, X i ω) / n) - p| ≥ ε} ≤ 2 * Real.exp (-2 * n * ε^2) := by
  -- ============================================================
  -- STEP 1: Define centered variables Y_i = X_i - p
  -- ============================================================
  let Y : Fin n → Ω → ℝ := fun i ω => X i ω - p

  -- ============================================================
  -- STEP 2: Properties of Y_i
  -- ============================================================

  -- 2a: Y_i are AEMeasurable
  have hY_meas : ∀ i, AEMeasurable (Y i) μ := fun i => (hX_meas i).sub_const p

  -- 2b: Y_i ∈ [-p, 1-p] (ae, but we have pointwise)
  have hY_bound_ae : ∀ i, ∀ᵐ ω ∂μ, Y i ω ∈ Set.Icc (-p) (1 - p) := fun i => by
    apply ae_of_all μ
    intro ω
    have hX := hX_bound i ω  -- X i ω ∈ [0, 1]
    constructor
    · linarith [hX.1]   -- X ≥ 0 → X - p ≥ -p
    · linarith [hX.2]   -- X ≤ 1 → X - p ≤ 1 - p

  -- 2c: E[Y_i] = 0
  have hY_mean : ∀ i, ∫ ω, Y i ω ∂μ = 0 := fun i => by
    simp only [Y]
    -- Need integrability of X_i to split the integral
    have hX_integrable : Integrable (X i) μ := by
      apply Integrable.of_mem_Icc 0 1 (hX_meas i)
      exact ae_of_all μ (hX_bound i)
    rw [integral_sub hX_integrable (integrable_const p)]
    rw [hX_mean i, integral_const, smul_eq_mul]
    -- μ.real Set.univ = 1 for probability measures
    have h_one : μ.real Set.univ = 1 := probReal_univ
    rw [h_one, one_mul]
    ring

  -- ============================================================
  -- STEP 3: Y_i are independent (centering preserves independence)
  -- ============================================================
  have hY_indep : iIndepFun Y μ := by
    -- Y i = (· - p) ∘ X i, and (· - p) : ℝ → ℝ is measurable
    have h_sub_meas : ∀ i : Fin n, Measurable (fun x : ℝ => x - p) :=
      fun _ => measurable_sub_const p
    exact hX_indep.comp (fun _ => (· - p)) h_sub_meas

  -- ============================================================
  -- STEP 4: Each Y_i is sub-Gaussian with parameter 1/4
  -- ============================================================
  -- Range: b - a = (1-p) - (-p) = 1
  -- ‖b - a‖₊ / 2 = 1/2
  -- Parameter: (1/2)² = 1/4
  have hY_subG : ∀ i, HasSubgaussianMGF (Y i) ((1/4 : ℝ≥0)) μ := fun i => by
    have h := hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
                (hY_meas i) (hY_bound_ae i) (hY_mean i)
    -- Need to show: ‖(1-p) - (-p)‖₊ / 2 = 1/2, so param = 1/4
    -- ‖1‖₊ / 2 = 1/2, (1/2)² = 1/4
    convert h using 1
    simp only [sub_neg_eq_add]
    -- (1 - p) + p = 1, so ‖1‖₊ = 1
    have h1 : ‖(1 : ℝ) - p + p‖₊ = 1 := by simp [nnnorm_one]
    rw [h1]
    norm_num

  -- ============================================================
  -- STEP 5: Apply Hoeffding for right tail (∑Y ≥ nε)
  -- ============================================================
  have h_right : μ.real {ω | (n : ℝ) * ε ≤ ∑ i : Fin n, Y i ω} ≤
                 Real.exp (-2 * n * ε^2) := by
    -- Use measure_sum_ge_le_of_iIndepFun with:
    --   s = Finset.univ : Finset (Fin n)
    --   c i = 1/4 for all i
    --   threshold = n * ε
    have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
    have h_nε_nonneg : 0 ≤ (n : ℝ) * ε := mul_nonneg (le_of_lt hn_pos) (le_of_lt hε)
    have h := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hY_indep (c := fun _ => (1/4 : ℝ≥0))
              (s := Finset.univ) (fun i _ => hY_subG i) h_nε_nonneg
    -- Convert the finset sum to what we need:
    -- ∑ i ∈ Finset.univ, 1/4 = n * (1/4) = n/4
    simp only [Finset.sum_const, Finset.card_fin] at h
    -- The bound is: exp(-(nε)² / (2 * n * (1/4))) = exp(-(nε)² * 4 / (2n)) = exp(-2nε²)
    -- The sets are equal by definition
    have h_set_eq : {ω | (n : ℝ) * ε ≤ ∑ i : Fin n, Y i ω} =
                    {ω | (n : ℝ) * ε ≤ ∑ i ∈ Finset.univ, Y i ω} := by rfl
    rw [h_set_eq]
    refine h.trans_eq ?_
    -- Algebraic simplification: -(n*ε)² / (2 * n * (1/4)) = -2 * n * ε²
    congr 1
    -- Handle NNReal coercion: n • (1/4 : ℝ≥0) coerced to ℝ
    simp only [nsmul_eq_mul]
    -- ↑(↑n * (1/4 : ℝ≥0)) = (n : ℝ) * (1/4 : ℝ)
    have h_coerce : (↑(↑n * (1/4 : ℝ≥0)) : ℝ) = (n : ℝ) * (1/4 : ℝ) := by
      rw [NNReal.coe_mul, NNReal.coe_natCast]
      simp only [NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat]
    rw [h_coerce]
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    field_simp
    ring

  -- ============================================================
  -- STEP 6: Apply Hoeffding for left tail (-∑Y ≥ nε, i.e., ∑Y ≤ -nε)
  -- ============================================================
  have h_left : μ.real {ω | (n : ℝ) * ε ≤ -(∑ i : Fin n, Y i ω)} ≤
                Real.exp (-2 * n * ε^2) := by
    -- Use HasSubgaussianMGF.neg: if Y_i is sub-Gaussian, so is -Y_i
    have hNegY_subG : ∀ i, HasSubgaussianMGF (fun ω => -(Y i ω)) ((1/4 : ℝ≥0)) μ :=
      fun i => (hY_subG i).neg
    -- -Y are also independent
    have hNegY_indep : iIndepFun (fun i ω => -(Y i ω)) μ := by
      have h_neg_meas : ∀ i : Fin n, Measurable (fun x : ℝ => -x) := fun _ => measurable_neg
      exact hY_indep.comp (fun _ => (- ·)) h_neg_meas
    have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
    have h_nε_nonneg : 0 ≤ (n : ℝ) * ε := mul_nonneg (le_of_lt hn_pos) (le_of_lt hε)
    have h := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hNegY_indep (c := fun _ => (1/4 : ℝ≥0))
              (s := Finset.univ) (fun i _ => hNegY_subG i) h_nε_nonneg
    simp only [Finset.sum_const, Finset.card_fin] at h
    -- The sets are equal by rearrangement
    have h_set_eq : {ω | (n : ℝ) * ε ≤ -(∑ i : Fin n, Y i ω)} =
                    {ω | (n : ℝ) * ε ≤ ∑ i ∈ Finset.univ, (fun i ω => -(Y i ω)) i ω} := by
      ext ω
      simp only [Set.mem_setOf_eq, Finset.sum_neg_distrib]
    rw [h_set_eq]
    refine h.trans_eq ?_
    congr 1
    simp only [nsmul_eq_mul]
    have h_coerce : (↑(↑n * (1/4 : ℝ≥0)) : ℝ) = (n : ℝ) * (1/4 : ℝ) := by
      rw [NNReal.coe_mul, NNReal.coe_natCast]
      simp only [NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat]
    rw [h_coerce]
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    field_simp
    ring

  -- ============================================================
  -- STEP 7: Union bound
  -- ============================================================
  -- |∑Y| ≥ nε ⟺ (∑Y ≥ nε) ∨ (∑Y ≤ -nε) ⟺ (∑Y ≥ nε) ∨ (-∑Y ≥ nε)
  have h_set_subset : {ω | |∑ i : Fin n, Y i ω| ≥ (n : ℝ) * ε} ⊆
      {ω | (n : ℝ) * ε ≤ ∑ i : Fin n, Y i ω} ∪ {ω | (n : ℝ) * ε ≤ -(∑ i : Fin n, Y i ω)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, ge_iff_le] at hω
    simp only [Set.mem_union, Set.mem_setOf_eq]
    rcases le_or_gt (∑ i : Fin n, Y i ω) 0 with h_neg | h_pos
    · right
      rw [abs_of_nonpos h_neg] at hω
      linarith
    · rcases le_or_gt (n * ε) (∑ i : Fin n, Y i ω) with h | h
      · left; exact h
      · right
        rw [abs_of_pos h_pos] at hω
        linarith

  -- Apply measureReal monotonicity and union bound
  have hn_pos_outer : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hn_ne_outer : (n : ℝ) ≠ 0 := ne_of_gt hn_pos_outer
  calc μ.real {ω | |((∑ i : Fin n, X i ω) / n) - p| ≥ ε}
      = μ.real {ω | |∑ i : Fin n, Y i ω| ≥ (n : ℝ) * ε} := by
        -- (∑X/n - p) = (∑(X-p))/n = (∑Y)/n
        -- |∑Y/n| ≥ ε ⟺ |∑Y| ≥ n*ε
        congr 1
        ext ω
        simp only [Set.mem_setOf_eq, ge_iff_le]
        -- Show (∑X)/n - p = (∑Y)/n
        have h_eq : (∑ i : Fin n, X i ω) / n - p = (∑ i : Fin n, Y i ω) / n := by
          simp only [Y, Finset.sum_sub_distrib]
          rw [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
          field_simp
        rw [h_eq, abs_div, Nat.abs_cast]
        -- ε ≤ |∑Y| / n ↔ n * ε ≤ |∑Y|
        rw [le_div_iff₀ hn_pos_outer]
        ring_nf
    _ ≤ μ.real ({ω | (n : ℝ) * ε ≤ ∑ i, Y i ω} ∪ {ω | (n : ℝ) * ε ≤ -(∑ i, Y i ω)}) := by
        exact measureReal_mono h_set_subset
    _ ≤ μ.real {ω | (n : ℝ) * ε ≤ ∑ i, Y i ω} + μ.real {ω | (n : ℝ) * ε ≤ -(∑ i, Y i ω)} := by
        exact measureReal_union_le _ _
    _ ≤ Real.exp (-2 * n * ε^2) + Real.exp (-2 * n * ε^2) := by
        gcongr
    _ = 2 * Real.exp (-2 * n * ε^2) := by ring

/-- Backward-compatible alias for older imports. -/
lemma hoeffding_iid_bounded_axiom {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (n : ℕ) (hn : 0 < n)
    (X : Fin n → Ω → ℝ)
    (p : ℝ)
    (hX_bound : ∀ i ω, X i ω ∈ Set.Icc 0 1)
    (hX_mean : ∀ i, ∫ ω, X i ω ∂μ = p)
    (hX_meas : ∀ i, AEMeasurable (X i) μ)
    (hX_indep : iIndepFun X μ)
    (ε : ℝ) (hε : 0 < ε) :
    μ.real {ω | |((∑ i : Fin n, X i ω) / n) - p| ≥ ε} ≤ 2 * Real.exp (-2 * n * ε^2) := by
  exact hoeffding_iid_bounded μ n hn X p hX_bound hX_mean hX_meas hX_indep ε hε

/-- Hoeffding bound on empirical violation rate deviation.

With n iid samples, the probability that the empirical rate deviates
from the true rate by more than ε is bounded by 2*exp(-2nε²).

This follows from Mathlib's `measure_sum_range_ge_le_of_iIndepFun` applied
to the centered violation indicators, which are sub-Gaussian with parameter 1/4.
-/
theorem hoeffding_violation_rate_bound
    [MeasurableSpace Strings] [MeasurableSingletonClass Strings] [Countable Strings]
    (fstar : Strings → Y) (x : Strings) (n : ℕ) (hn : 0 < n)
    (μ : Measure Strings) [IsProbabilityMeasure μ]
    (p_true : ℝ) (hp_def : p_true = ∫ z, violationInd fstar z x ∂μ)
    (hp_nonneg : 0 ≤ p_true) (hp_le_one : p_true ≤ 1)
    (hf : Integrable (fun z => violationInd fstar z x) μ)
    (ε : ℝ) (hε : 0 < ε) :
    (iidSampleMeasure μ n).real {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε}
      ≤ 2 * Real.exp (-2 * n * ε^2) := by
  -- Define the n violation indicators as random variables on the product space
  let X : Fin n → (Fin n → Strings) → ℝ := fun i ω => violationInd fstar (ω i) x
  -- These are bounded in [0,1]
  have hX_bound : ∀ i ω, X i ω ∈ Set.Icc 0 1 := fun i ω => violationInd_mem_Icc fstar (ω i) x
  -- They have common mean p_true (by marginal property)
  have hX_mean : ∀ i, ∫ ω, X i ω ∂(iidSampleMeasure μ n) = p_true := fun i => by
    simp only [X]
    rw [integral_proj_eq_marginal μ i (fun z => violationInd fstar z x) hf]
    exact hp_def.symm
  -- X_i are measurable (composition of measurable evaluation and violationInd)
  -- NOTE: Full proof requires showing violationInd fstar · x is measurable, which needs
  -- either fstar measurable or Strings countable. For discrete spaces this holds.
  have hX_meas : ∀ i, AEMeasurable (X i) (iidSampleMeasure μ n) := fun i => by
    simp only [X]
    -- X i ω = violationInd fstar (ω i) x = (violationInd fstar · x) ∘ (eval i)
    -- With [Countable Strings] + [MeasurableSingletonClass Strings], we have DiscreteMeasurableSpace
    -- Any function from a discrete space is measurable
    have h_viol_meas : Measurable (fun z : Strings => violationInd fstar z x) := Measurable.of_discrete
    exact (h_viol_meas.comp (measurable_pi_apply i)).aemeasurable
  -- X_i are independent under the product measure (by iIndepFun_pi)
  have hX_indep : iIndepFun X (iidSampleMeasure μ n) := by
    -- This follows from iIndepFun_pi: coordinate projections from product measure are independent
    -- X i ω = f(ω i) where f = violationInd fstar · x
    -- By iIndepFun_pi, (fun i ω => g i (ω i)) are independent for any g
    unfold iidSampleMeasure
    -- Apply iIndepFun_pi with constant function family: X_i = f for all i
    let f : Strings → ℝ := fun z => violationInd fstar z x
    have hf_meas : ∀ i : Fin n, AEMeasurable f μ := fun _ =>
      Measurable.of_discrete.aemeasurable
    -- iIndepFun_pi gives: (fun i ω => f (ω i)) are independent under Measure.pi (fun _ => μ)
    convert iIndepFun_pi (μ := fun _ : Fin n => μ) (X := fun _ => f) hf_meas using 1
  -- The empirical violation rate is exactly the sample mean of X
  have h_rate : ∀ ω, empiricalViolationRate fstar x n ω = (∑ i : Fin n, X i ω) / n := by
    intro ω
    rfl
  -- Apply Hoeffding's inequality
  simp only [h_rate]
  exact hoeffding_iid_bounded (iidSampleMeasure μ n) n hn X p_true hX_bound hX_mean hX_meas hX_indep ε hε

/-!
## Section 8: Final Audit Certification Theorem
-/

/-- Key lemma: 2*exp(-2n*ε²) = δ when ε = confidence_margin δ n.

This algebraic fact connects the Hoeffding bound to the confidence parameter. -/
lemma hoeffding_bound_eq_delta (δ : ℝ) (n : ℕ) (hn : 0 < n) (hδ : 0 < δ) (hδ' : δ < 2) :
    2 * Real.exp (-2 * n * (confidence_margin δ n)^2) = δ := by
  unfold confidence_margin
  have hn_pos : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
  have h2n_pos : (0 : ℝ) < 2 * n := by positivity
  have h2n_ne : (2 : ℝ) * n ≠ 0 := ne_of_gt h2n_pos
  have h_log_pos : 0 < Real.log (2 / δ) := by
    apply Real.log_pos
    rw [one_lt_div hδ]
    linarith
  have h_ratio_nonneg : 0 ≤ Real.log (2 / δ) / (2 * n) := by positivity
  -- sqrt(ln(2/δ)/(2n))² = ln(2/δ)/(2n)
  rw [Real.sq_sqrt h_ratio_nonneg]
  -- -2n * (ln(2/δ)/(2n)) = -ln(2/δ)
  have h_simplify : -2 * ↑n * (Real.log (2 / δ) / (2 * ↑n)) = -Real.log (2 / δ) := by
    field_simp
  rw [h_simplify]
  -- exp(-ln(2/δ)) = δ/2
  rw [Real.exp_neg, Real.exp_log (by positivity : 0 < 2 / δ)]
  -- 2 * (δ/2) = δ
  field_simp

/-- Complement bound: probability of the complement set.

For a probability measure μ with null-measurable S, μ.real(Sᶜ) = 1 - μ.real(S).
This follows from Mathlib's `measureReal_compl₀` and `probReal_univ`. -/
lemma prob_complement_bound {n : ℕ} [MeasurableSpace Strings] [MeasurableSingletonClass Strings]
    (μ : Measure (Fin n → Strings)) [IsProbabilityMeasure μ] (S : Set (Fin n → Strings))
    (hS : NullMeasurableSet S μ := by infer_instance) :
    μ.real Sᶜ = 1 - μ.real S := by
  rw [measureReal_compl₀ hS, probReal_univ]

/-- Measure-theoretic audit certification theorem.

With n iid samples from the summarization distribution, the empirical violation rate p̂
satisfies: P(p_true ≤ p̂ + ε) ≥ 1 - δ, where ε = confidence_margin δ n.

This is the formal version of Hoeffding's inequality applied to our audit framework,
providing rigorous probabilistic guarantees for empirical auditing.
-/
theorem audit_certification_measure_theoretic
    [MeasurableSpace Strings] [MeasurableSingletonClass Strings] [Countable Strings]
    (fstar : Strings → Y) (x : Strings)
    (μ : Measure Strings) [IsProbabilityMeasure μ]
    (p_true : ℝ) (hp_def : p_true = ∫ z, violationInd fstar z x ∂μ)
    (hp_nonneg : 0 ≤ p_true) (hp_le_one : p_true ≤ 1)
    (hf : Integrable (fun z => violationInd fstar z x) μ)
    (n : ℕ) (hn : 0 < n)
    (δ : ℝ) (hδ : 0 < δ) (hδ' : δ < 2) :
    let ε := confidence_margin δ n
    (iidSampleMeasure μ n).real {ω | p_true ≤ empiricalViolationRate fstar x n ω + ε} ≥ 1 - δ := by
  intro ε
  -- From hoeffding_violation_rate_bound: P(|p̂ - p| ≥ ε) ≤ 2*exp(-2nε²) = δ
  have h_margin_pos : 0 < ε := by
    show 0 < confidence_margin δ n
    unfold confidence_margin
    apply Real.sqrt_pos.mpr
    apply div_pos
    · apply Real.log_pos
      rw [one_lt_div hδ]
      linarith
    · exact mul_pos (by norm_num : (0:ℝ) < 2) (Nat.cast_pos.mpr hn)
  have h_hoeffding := hoeffding_violation_rate_bound fstar x n hn μ p_true hp_def hp_nonneg
    hp_le_one hf ε h_margin_pos
  have h_bound_eq := hoeffding_bound_eq_delta δ n hn hδ hδ'
  -- So P(|p̂ - p| ≥ ε) ≤ δ
  have h_deviation_bound : (iidSampleMeasure μ n).real
      {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε} ≤ δ := by
    calc (iidSampleMeasure μ n).real {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε}
        ≤ 2 * Real.exp (-2 * n * ε^2) := h_hoeffding
      _ = δ := h_bound_eq
  -- The complement: P(|p̂ - p| < ε) ≥ 1 - δ
  -- Since |p̂ - p| < ε implies p - ε < p̂, which implies p ≤ p̂ + ε
  -- We have: {|p̂ - p| < ε} ⊆ {p ≤ p̂ + ε}
  have h_subset : {ω | |empiricalViolationRate fstar x n ω - p_true| < ε} ⊆
      {ω | p_true ≤ empiricalViolationRate fstar x n ω + ε} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have := abs_sub_lt_iff.mp hω
    linarith [this.1]
  -- P({p ≤ p̂ + ε}) ≥ P({|p̂ - p| < ε}) ≥ 1 - δ
  have h_compl : {ω | |empiricalViolationRate fstar x n ω - p_true| < ε} =
      {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε}ᶜ := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_le]
  -- Use monotonicity of measure and complement bound
  have h_compl_bound : (iidSampleMeasure μ n).real
      {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε}ᶜ ≥ 1 - δ := by
    -- μ.real Sᶜ = 1 - μ.real S ≥ 1 - δ when μ.real S ≤ δ
    -- The set is null-measurable because it's defined by a measurable condition
    have h_measurable : NullMeasurableSet
        {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε} (iidSampleMeasure μ n) :=
      deviationSet_nullMeasurable (iidSampleMeasure μ n) fstar x p_true ε
    have h_compl_eq := prob_complement_bound (iidSampleMeasure μ n)
        {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε} h_measurable
    rw [h_compl_eq]
    linarith
  -- Chain the inequalities
  calc (iidSampleMeasure μ n).real {ω | p_true ≤ empiricalViolationRate fstar x n ω + ε}
      ≥ (iidSampleMeasure μ n).real {ω | |empiricalViolationRate fstar x n ω - p_true| < ε} := by
        apply measureReal_mono h_subset
    _ = (iidSampleMeasure μ n).real
          {ω | |empiricalViolationRate fstar x n ω - p_true| ≥ ε}ᶜ := by rw [h_compl]
    _ ≥ 1 - δ := h_compl_bound

/-!
## Section 9: Connection to Original Audit Framework

Connect the measure-theoretic results back to the PMF-based audit framework.
-/

/-- Audit certification for PMF distributions.

Specialization of the measure-theoretic theorem to PMF.toMeasure. -/
theorem audit_certification_pmf
    [MeasurableSpace Strings] [MeasurableSingletonClass Strings] [Countable Strings]
    (fstar : Strings → Y) (x : Strings)
    (p : PMF Strings)
    (hf : Integrable (fun z => violationInd fstar z x) p.toMeasure)
    (hp_nonneg : 0 ≤ ViolationProb fstar p x)
    (hp_le_one : ViolationProb fstar p x ≤ 1)
    (n : ℕ) (hn : 0 < n)
    (δ : ℝ) (hδ : 0 < δ) (hδ' : δ < 2) :
    let ε := confidence_margin δ n
    let p_true := ViolationProb fstar p x
    (iidSampleMeasure p.toMeasure n).real
      {ω | p_true ≤ empiricalViolationRate fstar x n ω + ε} ≥ 1 - δ := by
  -- Apply audit_certification_measure_theoretic with μ = p.toMeasure
  have hp_def : ViolationProb fstar p x = ∫ z, violationInd fstar z x ∂p.toMeasure :=
    ViolationProb_eq_integral p fstar x hf
  exact audit_certification_measure_theoretic fstar x p.toMeasure
    (ViolationProb fstar p x) hp_def hp_nonneg hp_le_one hf n hn δ hδ hδ'

end
