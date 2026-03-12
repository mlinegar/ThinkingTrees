import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.MeasureTheory.Measure.CharacteristicFunction
import Mathlib.MeasureTheory.Measure.IntegralCharFun
import Mathlib.MeasureTheory.Measure.Portmanteau
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.Order.Filter.AtTopBot.Basic
import Mathlib.Order.Interval.Set.UnorderedInterval
import Mathlib.Topology.Algebra.Module.Cardinality
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.Instances.ENNReal.Lemmas
import Mathlib.Topology.Order.OrderClosed

import FormalProofs.CLT.HellySelection

/-!
# FormalProofs/CLT/LevyContinuity.lean

Lemmas toward a Lévy-style continuity theorem for characteristic functions.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped Topology
open scoped Interval

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ProbabilityTheory

open MeasureTheory
open BoundedContinuousFunction
open Filter

def TightSeq (μs : ℕ → ProbabilityMeasure ℝ) : Prop :=
  ∀ ε > 0, ∃ r > 0, ∀ᶠ n in atTop, (μs n : Measure ℝ).real {x | r < |x|} ≤ ε

theorem tendsto_charFun_of_tendsto_probabilityMeasure
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ}
    (h : Tendsto μs atTop (𝓝 μ)) :
    ∀ t : ℝ, Tendsto (fun n => charFun (μs n : Measure ℝ) t) atTop
      (𝓝 (charFun (μ : Measure ℝ) t)) := by
  intro t
  have h' :=
    (ProbabilityMeasure.tendsto_iff_forall_integral_rclike_tendsto (𝕜 := ℂ)).1 h
  specialize h' (innerProbChar t)
  simpa [charFun_eq_integral_innerProbChar] using h'

lemma continuous_charFun {μ : Measure ℝ} [IsFiniteMeasure μ] :
    Continuous (fun t => charFun μ t) := by
  refine continuous_iff_continuousAt.2 ?_
  intro t0
  have h_meas :
      ∀ t : ℝ, AEStronglyMeasurable (fun x : ℝ => Complex.exp (t * x * Complex.I)) μ := by
    intro t
    exact (by fun_prop : Measurable fun x : ℝ => Complex.exp (t * x * Complex.I)).aestronglyMeasurable
  have h_bound :
      ∀ t : ℝ, ∀ᵐ x : ℝ ∂μ, ‖Complex.exp (t * x * Complex.I)‖ ≤ (1 : ℝ) := by
    intro t
    refine ae_of_all _ ?_
    intro x
    have hnorm : ‖Complex.exp (t * x * Complex.I)‖ = 1 := by
      have hx : (↑t : ℂ) * ↑x = (↑(t * x) : ℂ) := by
        simp
      rw [hx]
      exact Complex.norm_exp_ofReal_mul_I (t * x)
    simp [hnorm]
  have h_lim :
      ∀ᵐ x : ℝ ∂μ, Tendsto (fun t : ℝ => Complex.exp (t * x * Complex.I)) (𝓝 t0)
        (𝓝 (Complex.exp (t0 * x * Complex.I))) := by
    refine ae_of_all _ ?_
    intro x
    have hcont : Continuous fun t : ℝ => Complex.exp (t * x * Complex.I) := by
      fun_prop
    exact hcont.tendsto t0
  have h_tendsto :
      Tendsto (fun t : ℝ => ∫ x, Complex.exp (t * x * Complex.I) ∂μ) (𝓝 t0)
        (𝓝 (∫ x, Complex.exp (t0 * x * Complex.I) ∂μ)) := by
    refine
      MeasureTheory.tendsto_integral_filter_of_dominated_convergence
        (μ := μ) (l := 𝓝 t0) (bound := fun _ : ℝ => (1 : ℝ))
        (Filter.Eventually.of_forall h_meas) (Filter.Eventually.of_forall h_bound)
        (integrable_const (μ := μ) (c := (1 : ℝ))) h_lim
  simpa [ContinuousAt, charFun_apply_real, mul_comm, mul_left_comm, mul_assoc] using h_tendsto

lemma tendsto_intervalIntegral_one_sub_charFun_of_tendsto
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ} {a b : ℝ} (hab : a ≤ b)
    (h : ∀ t : ℝ, Tendsto (fun n => charFun (μs n : Measure ℝ) t) atTop
      (𝓝 (charFun (μ : Measure ℝ) t))) :
    Tendsto (fun n => ∫ t in a..b, (1 - charFun (μs n : Measure ℝ) t)) atTop
      (𝓝 (∫ t in a..b, (1 - charFun (μ : Measure ℝ) t))) := by
  have h_meas :
      ∀ n, AEStronglyMeasurable (fun t : ℝ => (1 - charFun (μs n : Measure ℝ) t))
        (volume.restrict (Set.Ioc a b)) := by
    intro n
    have hcont : Continuous fun t : ℝ => charFun (μs n : Measure ℝ) t := by
      simpa using (continuous_charFun (μ := (μs n : Measure ℝ)))
    have hmeas : Measurable fun t : ℝ => (1 - charFun (μs n : Measure ℝ) t) := by
      simpa using (measurable_const.sub hcont.measurable)
    exact hmeas.aestronglyMeasurable
  have h_bound :
      ∀ n, ∀ᵐ t ∂volume.restrict (Set.Ioc a b), ‖1 - charFun (μs n : Measure ℝ) t‖ ≤ (2 : ℝ) := by
    intro n
    refine ae_of_all _ ?_
    intro t
    simpa using (norm_one_sub_charFun_le_two (μ := (μs n : Measure ℝ)) (t := t))
  have h_lim :
      ∀ᵐ t ∂volume.restrict (Set.Ioc a b),
        Tendsto (fun n => 1 - charFun (μs n : Measure ℝ) t) atTop
          (𝓝 (1 - charFun (μ : Measure ℝ) t)) := by
    refine ae_of_all _ ?_
    intro t
    simpa using (tendsto_const_nhds.sub (h t))
  have h_tendsto_set :
      Tendsto
        (fun n => ∫ t, (1 - charFun (μs n : Measure ℝ) t) ∂(volume.restrict (Set.Ioc a b))) atTop
        (𝓝 (∫ t, (1 - charFun (μ : Measure ℝ) t) ∂(volume.restrict (Set.Ioc a b)))) := by
    refine
      MeasureTheory.tendsto_integral_of_dominated_convergence (μ := volume.restrict (Set.Ioc a b))
        (bound := fun _ : ℝ => (2 : ℝ)) h_meas
        (integrable_const (μ := volume.restrict (Set.Ioc a b)) (c := (2 : ℝ))) h_bound h_lim
  simpa [intervalIntegral.integral_of_le hab] using h_tendsto_set

lemma exists_small_one_sub_charFun
    {μ : ProbabilityMeasure ℝ} {ε : ℝ} (hε : 0 < ε) :
    ∃ δ > 0, ∀ t : ℝ, |t| < δ → ‖1 - charFun (μ : Measure ℝ) t‖ < ε := by
  have hcont : Continuous fun t : ℝ => (1 - charFun (μ : Measure ℝ) t) := by
    simpa using (continuous_const.sub (continuous_charFun (μ := (μ : Measure ℝ))))
  have h_tendsto :
      Tendsto (fun t : ℝ => (1 - charFun (μ : Measure ℝ) t)) (𝓝 (0 : ℝ)) (𝓝 (0 : ℂ)) := by
    have h0 : (1 - charFun (μ : Measure ℝ) (0 : ℝ)) = (0 : ℂ) := by
      simp [charFun_zero, IsProbabilityMeasure.measure_univ]
    simpa [h0] using hcont.tendsto (0 : ℝ)
  have h_ball :
      {t : ℝ | dist ((1 - charFun (μ : Measure ℝ) t)) (0 : ℂ) < ε} ∈ (𝓝 (0 : ℝ)) := by
    have h_pre :
        (fun t : ℝ => (1 - charFun (μ : Measure ℝ) t)) ⁻¹' Metric.ball (0 : ℂ) ε ∈
          (𝓝 (0 : ℝ)) := by
      exact (tendsto_def.1 h_tendsto) _ (Metric.ball_mem_nhds (0 : ℂ) hε)
    simpa [Metric.ball] using h_pre
  rcases Metric.mem_nhds_iff.mp h_ball with ⟨δ, hδ, hδprop⟩
  refine ⟨δ, hδ, ?_⟩
  intro t ht
  have ht' : t ∈ Metric.ball (0 : ℝ) δ := by
    simpa [Metric.mem_ball, dist_eq_norm, Real.norm_eq_abs] using ht
  have hdist : dist ((1 - charFun (μ : Measure ℝ) t)) (0 : ℂ) < ε := hδprop ht'
  simpa [dist_eq_norm] using hdist

lemma exists_large_r_integral_bound
    {μ : ProbabilityMeasure ℝ} {ε : ℝ} (hε : 0 < ε) :
    ∃ r > 0, (r / 2) * ‖∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), 1 - charFun (μ : Measure ℝ) t‖ ≤ ε := by
  have hε' : 0 < ε / 2 := by linarith
  rcases exists_small_one_sub_charFun (μ := μ) hε' with ⟨δ, hδ, hδbound⟩
  -- Choose `r` large enough so that `2 / r < δ`.
  let r : ℝ := max 1 (4 / δ)
  have hr_pos : 0 < r := by
    have h1 : (0 : ℝ) < 1 := by norm_num
    exact lt_of_lt_of_le h1 (le_max_left _ _)
  have h_two_over_r : 2 / r ≤ δ / 2 := by
    have h_r_ge : r ≥ 4 / δ := by exact le_max_right _ _
    have hδpos : 0 < δ := hδ
    have hδpos' : 0 < 4 / δ := by exact div_pos (by norm_num) hδpos
    have h_mul : 2 / r ≤ 2 / (4 / δ) := by
      have h_inv : 1 / r ≤ 1 / (4 / δ) := by
        exact one_div_le_one_div_of_le hδpos' h_r_ge
      have h_mul' : 2 * (1 / r) ≤ 2 * (1 / (4 / δ)) := by
        exact (mul_le_mul_of_nonneg_left h_inv (by norm_num : (0 : ℝ) ≤ 2))
      simpa [div_eq_mul_inv] using h_mul'
    have hcalc : 2 / (4 / δ) = δ / 2 := by
      field_simp [hδpos.ne']
      ring
    simpa [hcalc] using h_mul
  have h_two_over_r_lt : 2 / r < δ := by
    have hδpos : 0 < δ := hδ
    linarith
  have h_bound :
      ∀ t ∈ Ι (-(2 * r⁻¹)) (2 * r⁻¹), ‖1 - charFun (μ : Measure ℝ) t‖ ≤ ε / 2 := by
    intro t ht
    have hab : (-(2 * r⁻¹)) ≤ (2 * r⁻¹) := by
      have hpos : 0 ≤ 2 * r⁻¹ := by
        have : 0 ≤ r⁻¹ := by
          exact inv_nonneg.mpr (le_of_lt hr_pos)
        nlinarith
      linarith
    have ht' : t ∈ Set.Ioc (-(2 * r⁻¹)) (2 * r⁻¹) := by
      simpa [Set.uIoc_of_le hab] using ht
    have ht_abs : |t| < δ := by
      have ht_abs_le : |t| ≤ 2 * r⁻¹ := by
        refine abs_le.mpr ?_
        constructor
        · have hlow : (-(2 * r⁻¹)) < t := ht'.1
          exact le_of_lt hlow
        · exact ht'.2
      have ht_abs_le' : |t| ≤ 2 / r := by
        simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using ht_abs_le
      exact lt_of_le_of_lt ht_abs_le' h_two_over_r_lt
    exact (le_of_lt (hδbound t ht_abs))
  have h_integral_bound :
      ‖∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), 1 - charFun (μ : Measure ℝ) t‖
        ≤ (ε / 2) * |(2 * r⁻¹) - (-(2 * r⁻¹))| := by
    exact intervalIntegral.norm_integral_le_of_norm_le_const (h := h_bound)
  have habs : |(2 * r⁻¹) - (-(2 * r⁻¹))| = 4 / r := by
    have hr_nonneg : 0 ≤ r := le_of_lt hr_pos
    have hnonneg : 0 ≤ 4 * r⁻¹ := by
      have : 0 ≤ r⁻¹ := inv_nonneg.mpr hr_nonneg
      nlinarith
    have hcalc : (2 * r⁻¹) - (-(2 * r⁻¹)) = 4 * r⁻¹ := by
      ring
    calc
      |(2 * r⁻¹) - (-(2 * r⁻¹))| = |4 * r⁻¹| := by
        simp [hcalc]
      _ = 4 / r := by
        simp [div_eq_mul_inv, abs_of_nonneg hnonneg]
  have h_final :
      (r / 2) * ‖∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), 1 - charFun (μ : Measure ℝ) t‖ ≤ ε := by
    have hr_nonneg : 0 ≤ r / 2 := by
      nlinarith [hr_pos]
    calc
      (r / 2) * ‖∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), 1 - charFun (μ : Measure ℝ) t‖
          ≤ (r / 2) * ((ε / 2) * |(2 * r⁻¹) - (-(2 * r⁻¹))|) := by
              exact mul_le_mul_of_nonneg_left h_integral_bound hr_nonneg
      _ = (r / 2) * ((ε / 2) * (4 / r)) := by
              rw [habs]
      _ = ε := by
              have hr' : r ≠ 0 := hr_pos.ne'
              have h_cancel : (r / 2) * (4 / r) = 2 := by
                calc
                  (r / 2) * (4 / r) = (r * 4) / (2 * r) := by
                    ring
                  _ = (r * 4) / (r * 2) := by
                    ring
                  _ = 4 / 2 := by
                    simpa [mul_comm, mul_left_comm, mul_assoc] using
                      (mul_div_mul_left (4 : ℝ) 2 hr')
                  _ = 2 := by
                    ring
              calc
                (r / 2) * ((ε / 2) * (4 / r))
                    = (ε / 2) * ((r / 2) * (4 / r)) := by
                        ring
                _ = (ε / 2) * 2 := by
                        simp [h_cancel]
                _ = ε := by
                        ring
  exact ⟨r, hr_pos, h_final⟩

theorem tightSeq_of_charFun_tendsto
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ}
    (h : ∀ t : ℝ, Tendsto (fun n => charFun (μs n : Measure ℝ) t) atTop
      (𝓝 (charFun (μ : Measure ℝ) t))) :
    TightSeq μs := by
  intro ε hε
  have hε' : 0 < ε / 2 := by linarith
  rcases exists_large_r_integral_bound (μ := μ) hε' with ⟨r, hr, hμr⟩
  have hab : (-(2 * r⁻¹)) ≤ (2 * r⁻¹) := by
    have hpos : 0 ≤ 2 * r⁻¹ := by
      have : 0 ≤ r⁻¹ := by
        exact inv_nonneg.mpr (le_of_lt hr)
      nlinarith
    linarith
  have h_tendsto :
      Tendsto
        (fun n =>
          ∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μs n : Measure ℝ) t)) atTop
        (𝓝 (∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μ : Measure ℝ) t))) := by
    exact tendsto_intervalIntegral_one_sub_charFun_of_tendsto (hab := hab) h
  have h_tendsto_norm :
      Tendsto
        (fun n =>
          ‖(∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μs n : Measure ℝ) t))
            - ∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μ : Measure ℝ) t)‖) atTop
        (𝓝 (0 : ℝ)) := by
    exact (tendsto_iff_norm_sub_tendsto_zero).1 h_tendsto
  have h_eventually :
      ∀ᶠ n in atTop,
        ‖(∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μs n : Measure ℝ) t))
          - ∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μ : Measure ℝ) t)‖ < ε / r := by
    have hpos : 0 < ε / r := by
      have hrpos : 0 < r := hr
      positivity
    exact (tendsto_order.1 h_tendsto_norm).2 _ hpos
  refine ⟨r, hr, ?_⟩
  filter_upwards [h_eventually] with n hn
  set A : ℂ :=
    ∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μs n : Measure ℝ) t)
  set B : ℂ :=
    ∫ t in (-(2 * r⁻¹))..(2 * r⁻¹), (1 - charFun (μ : Measure ℝ) t)
  have h_norm_bound :
      ‖A‖ ≤ ‖B‖ + ε / r := by
    have htri : ‖A‖ ≤ ‖A - B‖ + ‖B‖ := by
      have := norm_add_le (A - B) B
      simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this
    have hA : ‖A - B‖ < ε / r := by
      simpa [A, B] using hn
    linarith
  have h_integral_bound :
      (r / 2) * ‖A‖ ≤ ε := by
    have hμr' : (r / 2) * ‖B‖ ≤ ε / 2 := by
      have hμr' := hμr
      simp [B] at hμr'
      exact hμr'
    have h_r : (r / 2) * (ε / r) = ε / 2 := by
      field_simp [hr.ne']
    have h_r_le : (r / 2) * (ε / r) ≤ ε / 2 := by
      simp [h_r]
    calc
      (r / 2) * ‖A‖ ≤ (r / 2) * (‖B‖ + ε / r) := by
              gcongr
      _ = (r / 2) * ‖B‖ + (r / 2) * (ε / r) := by
              ring
      _ ≤ ε / 2 + ε / 2 := by
              exact add_le_add hμr' h_r_le
      _ = ε := by ring
  have h_tail :
      (μs n : Measure ℝ).real {x | r < |x|} ≤
        (r / 2) * ‖A‖ := by
    simpa [A, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
      (MeasureTheory.measureReal_abs_gt_le_integral_charFun (μ := (μs n : Measure ℝ)) hr)
  exact h_tail.trans h_integral_bound

lemma tightSeq_subseq {μs : ℕ → ProbabilityMeasure ℝ} (h : TightSeq μs) {s : ℕ → ℕ}
    (hs : StrictMono s) :
    TightSeq (μs ∘ s) := by
  intro ε hε
  rcases h ε hε with ⟨r, hr, h_eventually⟩
  rcases (eventually_atTop.1 h_eventually) with ⟨N, hN⟩
  have hs_tendsto : Tendsto s atTop atTop := StrictMono.tendsto_atTop hs
  rcases (tendsto_atTop_atTop.1 hs_tendsto N) with ⟨N', hN'⟩
  refine ⟨r, hr, ?_⟩
  refine (eventually_atTop.2 ⟨N', ?_⟩)
  intro n hn
  exact hN (s n) (hN' n hn)

lemma eventually_cdf_ge_one_sub_of_tightSeq
    {μs : ℕ → ProbabilityMeasure ℝ} (h : TightSeq μs) {ε : ℝ} (hε : 0 < ε) :
    ∃ r > 0, ∀ᶠ n in atTop, (1 - ε) ≤ cdf (μs n) r := by
  rcases h ε hε with ⟨r, hr, h_eventually⟩
  refine ⟨r, hr, ?_⟩
  filter_upwards [h_eventually] with n hn
  have h_tail :
      (μs n : Measure ℝ).real (Set.Ioi r) ≤ (μs n : Measure ℝ).real {x | r < |x|} := by
    refine measureReal_mono ?_
    intro x hx
    have hx' : r < |x| := by
      have hr_pos : 0 < r := hr
      have hx_pos : r < x := hx
      have hx_abs : |x| = x := abs_of_pos (lt_trans hr_pos hx_pos)
      simpa [hx_abs] using hx_pos
    exact hx'
  have h_cdf :
      cdf (μs n) r = 1 - (μs n : Measure ℝ).real (Set.Ioi r) := by
    have h_add := probReal_add_probReal_compl (μ := (μs n : Measure ℝ))
      (s := Set.Iic r) (h := measurableSet_Iic)
    have h_add' :
        (μs n : Measure ℝ).real (Set.Iic r) + (μs n : Measure ℝ).real (Set.Ioi r) = 1 := by
      simpa [Set.compl_Iic] using h_add
    linarith [h_add', cdf_eq_real (μ := (μs n : Measure ℝ)) r]
  have h_bound : (μs n : Measure ℝ).real (Set.Ioi r) ≤ ε := h_tail.trans hn
  linarith [h_cdf, h_bound]

lemma eventually_cdf_le_of_tightSeq
    {μs : ℕ → ProbabilityMeasure ℝ} (h : TightSeq μs) {ε : ℝ} (hε : 0 < ε) :
    ∃ r > 0, ∀ᶠ n in atTop, cdf (μs n) (-r - 1) ≤ ε := by
  rcases h ε hε with ⟨r, hr, h_eventually⟩
  refine ⟨r, hr, ?_⟩
  filter_upwards [h_eventually] with n hn
  have h_tail :
      (μs n : Measure ℝ).real (Set.Iic (-r - 1)) ≤
        (μs n : Measure ℝ).real {x | r < |x|} := by
    refine measureReal_mono ?_
    intro x hx
    have hx' : r < |x| := by
      have hx_le : x ≤ -r - 1 := hx
      have hx_neg : x < 0 := by linarith
      have hx_abs : |x| = -x := abs_of_neg hx_neg
      have hx_pos : r + 1 ≤ -x := by linarith
      have hx_pos' : r < -x := by linarith
      simpa [hx_abs] using hx_pos'
    exact hx'
  have h_cdf :
      cdf (μs n) (-r - 1) = (μs n : Measure ℝ).real (Set.Iic (-r - 1)) := by
    simpa using (cdf_eq_real (μ := (μs n : Measure ℝ)) (-r - 1))
  exact (by linarith [h_cdf, h_tail, hn])

lemma tendsto_ratLimit_atTop_one_of_tightSeq
    {μs : ℕ → ProbabilityMeasure ℝ} (hT : TightSeq μs) {s : ℕ → ℕ} (hs : StrictMono s)
    {F : ℚ → ℝ} (hF_nonneg : ∀ q, 0 ≤ F q) (hF_le_one : ∀ q, F q ≤ 1)
    (hF_mono : Monotone F)
    (hF_tendsto : ∀ q : ℚ,
      Tendsto (fun n => cdf (μs (s n)) (q : ℝ)) atTop (𝓝 (F q))) :
    Tendsto (ratLimit F) atTop (𝓝 1) := by
  have hT' : TightSeq (μs ∘ s) := tightSeq_subseq hT hs
  refine (tendsto_order.2 ⟨?_, ?_⟩)
  · intro a ha
    have hε : 0 < (1 - a) / 2 := by linarith
    rcases eventually_cdf_ge_one_sub_of_tightSeq (μs := μs ∘ s) hT' hε with ⟨r, hr, h_event⟩
    obtain ⟨q, hq⟩ := exists_rat_gt r
    have h_event' : ∀ᶠ n in atTop, (1 - (1 - a) / 2) ≤ cdf (μs (s n)) (q : ℝ) := by
      filter_upwards [h_event] with n hn
      have h_mono := monotone_cdf (μ := (μs (s n) : Measure ℝ))
      have hrq : (r : ℝ) ≤ (q : ℚ) := le_of_lt hq
      exact hn.trans (h_mono hrq)
    have h_ge : (1 - (1 - a) / 2) ≤ F q :=
      le_of_tendsto_of_tendsto tendsto_const_nhds (hF_tendsto q) h_event'
    have h_gt : a < F q := by linarith
    refine (eventually_atTop.2 ⟨(q : ℝ), ?_⟩)
    intro x hx
    have h_q_le : F q ≤ ratLimit F x :=
      le_ratLimit_of_mono (F := F) hF_mono (x := x) (q := q) hx
    exact lt_of_lt_of_le h_gt h_q_le
  · intro a ha
    have h_le : ∀ x, ratLimit F x ≤ 1 :=
      fun x => ratLimit_le_one (F := F) hF_nonneg hF_le_one x
    exact Filter.Eventually.of_forall (fun x => (h_le x).trans_lt ha)

lemma tendsto_ratLimit_atBot_zero_of_tightSeq
    {μs : ℕ → ProbabilityMeasure ℝ} (hT : TightSeq μs) {s : ℕ → ℕ} (hs : StrictMono s)
    {F : ℚ → ℝ} (hF_nonneg : ∀ q, 0 ≤ F q)
    (hF_tendsto : ∀ q : ℚ,
      Tendsto (fun n => cdf (μs (s n)) (q : ℝ)) atTop (𝓝 (F q))) :
    Tendsto (ratLimit F) atBot (𝓝 0) := by
  have hT' : TightSeq (μs ∘ s) := tightSeq_subseq hT hs
  refine (tendsto_order.2 ⟨?_, ?_⟩)
  · intro a ha
    have h_nonneg : ∀ x, 0 ≤ ratLimit F x := ratLimit_nonneg (F := F) hF_nonneg
    exact Filter.Eventually.of_forall (fun x => lt_of_lt_of_le ha (h_nonneg x))
  · intro a ha
    have hε : 0 < a / 2 := by linarith
    rcases eventually_cdf_le_of_tightSeq (μs := μs ∘ s) hT' hε with ⟨r, hr, h_event⟩
    obtain ⟨q, hq⟩ := exists_rat_lt (-r - 1)
    have h_event' : ∀ᶠ n in atTop, cdf (μs (s n)) (q : ℝ) ≤ a / 2 := by
      filter_upwards [h_event] with n hn
      have h_mono := monotone_cdf (μ := (μs (s n) : Measure ℝ))
      have hq_le : (q : ℝ) ≤ -r - 1 := le_of_lt hq
      exact (h_mono hq_le).trans hn
    have h_le : F q ≤ a / 2 :=
      le_of_tendsto_of_tendsto (hF_tendsto q) tendsto_const_nhds h_event'
    have h_lt : F q < a := by linarith
    refine (eventually_atBot.2 ⟨(q : ℝ) - 1, ?_⟩)
    intro x hx
    have hxq : x < (q : ℚ) := by linarith
    have h_le_x : ratLimit F x ≤ F q :=
      ratLimit_le_of_lt (F := F) hF_nonneg (x := x) (q := q) hxq
    exact lt_of_le_of_lt h_le_x h_lt

theorem tightSeq_subseq_tendsto_cdf
    {μs : ℕ → ProbabilityMeasure ℝ} (hT : TightSeq μs) :
    ∃ s : ℕ → ℕ, StrictMono s ∧ ∃ μ : ProbabilityMeasure ℝ,
      ∀ x, ContinuousAt (cdf μ) x →
        Tendsto (fun n => cdf (μs (s n)) x) atTop (𝓝 (cdf μ x)) := by
  classical
  rcases exists_subseq_tendsto_cdf_rat μs with ⟨s, hs, F, hF_bounds, hF_tendsto⟩
  have hF_nonneg : ∀ q, 0 ≤ F q := fun q => (hF_bounds q).1
  have hF_le_one : ∀ q, F q ≤ 1 := fun q => (hF_bounds q).2
  have hF_mono : Monotone F :=
    monotone_limit_cdf_rat (μs := μs) (s := s) (F := F) hF_tendsto
  let G : StieltjesFunction ℝ := ratStieltjes F hF_nonneg
  have hG_bot : Tendsto G atBot (𝓝 0) := by
    simpa [G] using
      (tendsto_ratLimit_atBot_zero_of_tightSeq (μs := μs) hT hs hF_nonneg hF_tendsto)
  have hG_top : Tendsto G atTop (𝓝 1) := by
    simpa [G] using
      (tendsto_ratLimit_atTop_one_of_tightSeq (μs := μs) hT hs hF_nonneg hF_le_one hF_mono
        hF_tendsto)
  have h_prob : IsProbabilityMeasure G.measure :=
    StieltjesFunction.isProbabilityMeasure (f := G) hG_bot hG_top
  let μ : ProbabilityMeasure ℝ := ⟨G.measure, h_prob⟩
  have h_cdf : cdf μ = G := by
    simpa using (cdf_measure_stieltjesFunction G hG_bot hG_top)
  refine ⟨s, hs, μ, ?_⟩
  intro x hx
  have hx' : ContinuousAt (ratLimit F) x := by
    simpa [h_cdf, G] using hx
  have h_tend :=
    tendsto_cdf_of_tendsto_cdf_rat (μs := μs) (s := s) (F := F) hF_nonneg hF_tendsto hx'
  simpa [h_cdf, G] using h_tend

def cdfContSet (μ : ProbabilityMeasure ℝ) : Set ℝ :=
  {x | ContinuousAt (cdf μ) x}

lemma dense_cdfContSet (μ : ProbabilityMeasure ℝ) : Dense (cdfContSet μ) := by
  have hcount' :
      Set.Countable {x | Function.leftLim (cdf μ) x ≠ cdf μ x} := by
    simpa using (StieltjesFunction.countable_leftLim_ne (cdf μ))
  have hsubset :
      {x | ¬ ContinuousAt (cdf μ) x} ⊆ {x | Function.leftLim (cdf μ) x ≠ cdf μ x} := by
    intro x hx
    have hiff :
        ContinuousAt (cdf μ) x ↔ Function.leftLim (cdf μ) x = cdf μ x := by
      have hmono := monotone_cdf (μ := (μ : Measure ℝ))
      have hright : Function.rightLim (cdf μ) x = cdf μ x := by
        simpa using (StieltjesFunction.rightLim_eq (cdf μ) x)
      simpa [hright] using
        (hmono.continuousAt_iff_leftLim_eq_rightLim (x := x))
    intro h_eq
    exact hx (hiff.2 h_eq)
  have hcount : Set.Countable {x | ¬ ContinuousAt (cdf μ) x} :=
    hcount'.mono hsubset
  simpa [cdfContSet, Set.compl_setOf, not_not] using
    (Set.Countable.dense_compl (E := ℝ) (𝕜 := ℝ) hcount)

def cdfContIocSet (μ : ProbabilityMeasure ℝ) : Set (Set ℝ) :=
  {S | ∃ᵉ (a ∈ cdfContSet μ) (b ∈ cdfContSet μ), a < b ∧ Set.Ioc a b = S}

lemma tendsto_measure_Ioc_of_tendsto_cdf
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ} {a b : ℝ}
    (ha : ContinuousAt (cdf μ) a) (hb : ContinuousAt (cdf μ) b)
    (h : ∀ x, ContinuousAt (cdf μ) x →
      Tendsto (fun n => cdf (μs n) x) atTop (𝓝 (cdf μ x))) :
    Tendsto (fun n => μs n (Set.Ioc a b)) atTop (𝓝 (μ (Set.Ioc a b))) := by
  have hμs :
      ∀ n, ((μs n : Measure ℝ) (Set.Ioc a b)) =
        ENNReal.ofReal (cdf (μs n) b - cdf (μs n) a) := by
    intro n
    simpa [measure_cdf] using
      (StieltjesFunction.measure_Ioc (f := cdf (μs n : Measure ℝ)) a b)
  have hμ :
      ((μ : Measure ℝ) (Set.Ioc a b)) = ENNReal.ofReal (cdf μ b - cdf μ a) := by
    simpa [measure_cdf] using
      (StieltjesFunction.measure_Ioc (f := cdf (μ : Measure ℝ)) a b)
  have h_diff :
      Tendsto (fun n => cdf (μs n) b - cdf (μs n) a) atTop
        (𝓝 (cdf μ b - cdf μ a)) := by
    exact (h b hb).sub (h a ha)
  have h_tendsto :=
    (ENNReal.tendsto_ofReal h_diff :
      Tendsto (fun n => ENNReal.ofReal (cdf (μs n) b - cdf (μs n) a)) atTop
        (𝓝 (ENNReal.ofReal (cdf μ b - cdf μ a))))
  have h_tendsto_ENN :
      Tendsto (fun n => (μs n : Measure ℝ) (Set.Ioc a b)) atTop
        (𝓝 ((μ : Measure ℝ) (Set.Ioc a b))) := by
    simpa [hμs, hμ] using h_tendsto
  have hlim : ((μ : Measure ℝ) (Set.Ioc a b)) ≠ (⊤ : ENNReal) := by
    exact measure_ne_top (μ := (μ : Measure ℝ)) (Set.Ioc a b)
  have h_tendsto_NN :
      Tendsto
        (fun n => ENNReal.toNNReal ((μs n : Measure ℝ) (Set.Ioc a b))) atTop
        (𝓝 (((μ : Measure ℝ) (Set.Ioc a b)).toNNReal)) := by
    exact (ENNReal.tendsto_toNNReal hlim).comp h_tendsto_ENN
  have h_tendsto' :
      Tendsto (fun n => μs n (Set.Ioc a b)) atTop (𝓝 (μ (Set.Ioc a b))) := by
    simpa [ProbabilityMeasure.coeFn_def] using h_tendsto_NN
  exact h_tendsto'

lemma tendsto_probabilityMeasure_of_tendsto_cdf_cont
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ}
    (h : ∀ x, ContinuousAt (cdf μ) x →
      Tendsto (fun n => cdf (μs n) x) atTop (𝓝 (cdf μ x))) :
    Tendsto μs atTop (𝓝 μ) := by
  classical
  have hPi : IsPiSystem (cdfContIocSet μ) := by
    simpa [cdfContIocSet] using (isPiSystem_Ioc_mem (s := cdfContSet μ) (t := cdfContSet μ))
  have hmeas : ∀ s ∈ cdfContIocSet μ, MeasurableSet s := by
    intro s hs
    rcases hs with ⟨a, ha, b, hb, hab, rfl⟩
    exact measurableSet_Ioc
  have h_dense : Dense (cdfContSet μ) := dense_cdfContSet μ
  have h_basis :
      ∀ (u : Set ℝ), IsOpen u → ∀ x ∈ u,
        ∃ s ∈ cdfContIocSet μ, s ∈ 𝓝 x ∧ s ⊆ u := by
    intro u hu x hx
    rcases (mem_nhds_iff_exists_Ioo_subset).1 (hu.mem_nhds hx) with ⟨l, u', hx_mem, hsub⟩
    have hlx : l < x := hx_mem.1
    have hxu : x < u' := hx_mem.2
    rcases h_dense.exists_between hlx with ⟨a, haC, haIoo⟩
    rcases h_dense.exists_between hxu with ⟨b, hbC, hbIoo⟩
    have ha_lt_x : a < x := haIoo.2
    have hx_lt_b : x < b := hbIoo.1
    have hab : a < b := lt_trans ha_lt_x hx_lt_b
    have h_nhds : Set.Ioc a b ∈ 𝓝 x := by
      have hIoo : Set.Ioo a b ∈ 𝓝 x := Ioo_mem_nhds ha_lt_x hx_lt_b
      exact mem_of_superset hIoo (by intro y hy; exact ⟨hy.1, le_of_lt hy.2⟩)
    have h_sub' : Set.Ioc a b ⊆ Set.Ioo l u' := by
      intro y hy
      refine ⟨lt_trans haIoo.1 hy.1, ?_⟩
      exact lt_of_le_of_lt hy.2 hbIoo.2
    refine ⟨Set.Ioc a b, ?_, h_nhds, h_sub'.trans hsub⟩
    exact ⟨a, haC, b, hbC, hab, rfl⟩
  have h_tendsto_sets :
      ∀ s ∈ cdfContIocSet μ, Tendsto (fun n => μs n s) atTop (𝓝 (μ s)) := by
    intro s hs
    rcases hs with ⟨a, ha, b, hb, hab, rfl⟩
    exact tendsto_measure_Ioc_of_tendsto_cdf (μs := μs) (μ := μ) ha hb h
  exact IsPiSystem.tendsto_probabilityMeasure_of_tendsto_of_mem hPi hmeas h_basis h_tendsto_sets

lemma tendsto_of_subseq_subseq_tendsto {α : Type*} [TopologicalSpace α] {u : ℕ → α} {a : α}
    (h : ∀ s : ℕ → ℕ, StrictMono s →
      ∃ t : ℕ → ℕ, StrictMono t ∧ Tendsto (u ∘ s ∘ t) atTop (𝓝 a)) :
    Tendsto u atTop (𝓝 a) := by
  classical
  by_contra hnot
  rcases (not_tendsto_iff_exists_frequently_notMem).1 hnot with ⟨s, hs_mem, hs_freq⟩
  rcases extraction_of_frequently_atTop hs_freq with ⟨φ, hφ_mono, hφ⟩
  rcases h φ hφ_mono with ⟨ψ, hψ_mono, h_tendsto⟩
  have h_eventually := h_tendsto.eventually_mem hs_mem
  rcases (eventually_atTop.1 h_eventually) with ⟨N, hN⟩
  have h_in : u (φ (ψ N)) ∈ s := hN N (le_rfl)
  have h_out : u (φ (ψ N)) ∉ s := hφ (ψ N)
  exact (h_out h_in).elim

theorem tendsto_probabilityMeasure_of_tendsto_charFun
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ}
    (h : ∀ t : ℝ, Tendsto (fun n => charFun (μs n : Measure ℝ) t) atTop
      (𝓝 (charFun (μ : Measure ℝ) t))) :
    Tendsto μs atTop (𝓝 μ) := by
  have hT : TightSeq μs := tightSeq_of_charFun_tendsto h
  have h_subseq :
      ∀ s : ℕ → ℕ, StrictMono s →
        ∃ t : ℕ → ℕ, StrictMono t ∧ Tendsto (μs ∘ s ∘ t) atTop (𝓝 μ) := by
    intro s hs
    have hT' : TightSeq (μs ∘ s) := tightSeq_subseq hT hs
    rcases tightSeq_subseq_tendsto_cdf (μs := μs ∘ s) hT' with ⟨t, ht, μ', h_cdf⟩
    have h_tendsto' :
        Tendsto (μs ∘ s ∘ t) atTop (𝓝 μ') := by
      refine tendsto_probabilityMeasure_of_tendsto_cdf_cont (μs := μs ∘ s ∘ t) (μ := μ') ?_
      intro x hx
      exact h_cdf x hx
    have h_char_sub :
        ∀ r : ℝ,
          Tendsto (fun n => charFun (μs (s (t n)) : Measure ℝ) r) atTop
            (𝓝 (charFun (μ : Measure ℝ) r)) := by
      intro r
      have hst : StrictMono (s ∘ t) := hs.comp ht
      exact (h r).comp (StrictMono.tendsto_atTop hst)
    have h_char_sub' :
        ∀ r : ℝ,
          Tendsto (fun n => charFun (μs (s (t n)) : Measure ℝ) r) atTop
            (𝓝 (charFun (μ' : Measure ℝ) r)) := by
      intro r
      exact tendsto_charFun_of_tendsto_probabilityMeasure (μs := μs ∘ s ∘ t) (μ := μ') h_tendsto' r
    have h_char_eq : charFun (μ' : Measure ℝ) = charFun (μ : Measure ℝ) := by
      ext r
      exact tendsto_nhds_unique (h_char_sub' r) (h_char_sub r)
    have h_measure : (μ' : Measure ℝ) = (μ : Measure ℝ) := by
      exact Measure.ext_of_charFun h_char_eq
    have h_mu : μ' = μ := by
      exact ProbabilityMeasure.toMeasure_injective h_measure
    refine ⟨t, ht, ?_⟩
    simpa [h_mu] using h_tendsto'
  exact tendsto_of_subseq_subseq_tendsto h_subseq

end ProbabilityTheory
