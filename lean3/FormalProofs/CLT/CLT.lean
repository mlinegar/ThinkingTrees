import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Probability.Independence.CharacteristicFunction
import Mathlib.Probability.IdentDistrib
import Mathlib.Analysis.SpecialFunctions.Complex.LogBounds
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.MeasureTheory.Function.L1Space.Integrable
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.MeasureTheory.Measure.Dirac
import Mathlib.Data.Real.Sqrt

import FormalProofs.CLT.Normal
import FormalProofs.CLT.LevyContinuity

/-!
# FormalProofs/Probability/CLT.lean

Building blocks toward a first-principles CLT proof via characteristic functions.

Key results:
* `central_limit_theorem_iid_bounded` (bounded i.i.d. CLT).
* `CharFunCLTScale_of_integrable_abs_pow3` (Lyapunov p=3 scaling).
* `CharFunCLTScale_of_integrable_sq` (finite-variance scaling).
* `central_limit_theorem_iid_finite_variance` (classical i.i.d. CLT).
* `central_limit_theorem_iid_of_charFunScale` (CLT from characteristic-function scaling).
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Topology

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ProbabilityTheory

open MeasureTheory
open Filter

instance : OpensMeasurableSpace ℝ := by infer_instance
example : TopologicalSpace ℝ := by infer_instance
example : TopologicalSpace (ProbabilityMeasure ℝ) := by infer_instance

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

def charFunRV (X : Ω → ℝ) (t : ℝ) : ℂ :=
  charFun (μ.map X) t

def sumRV (X : ℕ → Ω → ℝ) (s : Finset ℕ) : Ω → ℝ :=
  Finset.sum s (fun i => X i)

lemma charFun_sum_two {X Y : Ω → ℝ}
    (hX : AEMeasurable X μ) (hY : AEMeasurable Y μ) (hXY : X ⟂ᵢ[μ] Y) :
    charFun (μ.map (fun ω => X ω + Y ω)) = charFun (μ.map X) * charFun (μ.map Y) := by
  simpa using (IndepFun.charFun_map_add_eq_mul (P := μ) hX hY hXY)

lemma charFun_sum_finset {X : ℕ → Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (s : Finset ℕ) (t : ℝ) :
    charFun (μ.map (sumRV X s)) t =
      Finset.prod s (fun i => charFun (μ.map (X i)) t) := by
  classical
  refine Finset.induction_on s ?base ?step
  ·
    have hmap : μ.map (0 : Ω → ℝ) = Measure.dirac (0 : ℝ) := by
      calc
        μ.map (0 : Ω → ℝ) = (μ Set.univ) • Measure.dirac (0 : ℝ) := by
          exact (Measure.map_const μ (0 : ℝ))
        _ = Measure.dirac (0 : ℝ) := by
          rw [IsProbabilityMeasure.measure_univ, one_smul]
    simp [sumRV, hmap]
  · intro a s ha h_ind
    have h_indep_sum :
        (X a) ⟂ᵢ[μ] (sumRV X s) := by
      have h :=
        h_indep.indepFun_finset_sum_of_notMem h_meas (s := s) (i := a) ha
      simpa [sumRV] using h.symm
    have hX : AEMeasurable (X a) μ := (h_meas a).aemeasurable
    have hSum : AEMeasurable (sumRV X s) μ := by
      simpa [sumRV] using
        (Finset.aemeasurable_sum (s := s) (f := X) (fun i hi => (h_meas i).aemeasurable))
    have h_char_add :
        charFun (μ.map (X a + sumRV X s)) =
          charFun (μ.map (X a)) * charFun (μ.map (sumRV X s)) := by
      simpa [sumRV] using (IndepFun.charFun_map_add_eq_mul (P := μ) hX hSum h_indep_sum)
    -- Use sum over insert and the induction hypothesis.
    have h_sum_insert :
        sumRV X (insert a s) = X a + sumRV X s := by
      simpa [sumRV] using (Finset.sum_insert (s := s) (a := a) (f := fun i => X i) ha)
    calc
      charFun (μ.map (sumRV X (insert a s))) t
          = charFun (μ.map (X a + sumRV X s)) t := by
              simpa using congrArg (fun f => charFun (μ.map f) t) h_sum_insert
      _ = (charFun (μ.map (X a)) * charFun (μ.map (sumRV X s))) t := by
              simpa using congrArg (fun f => f t) h_char_add
      _ = charFun (μ.map (X a)) t *
            Finset.prod s (fun i => charFun (μ.map (X i)) t) := by
              simp [Pi.mul_apply, h_ind]
      _ = Finset.prod (insert a s) (fun i => charFun (μ.map (X i)) t) := by
              simp [Finset.prod_insert, ha, mul_comm, mul_left_comm, mul_assoc]

/-- Sum over the first `n` indices. -/
def sumRVRange (X : ℕ → Ω → ℝ) (n : ℕ) : Ω → ℝ :=
  sumRV X (Finset.range n)

/-- Normalized sum `(1 / √n) * ∑_{i < n} X_i`. -/
def normalizedSum (X : ℕ → Ω → ℝ) (n : ℕ) : Ω → ℝ :=
  fun ω => (Real.sqrt (n : ℝ))⁻¹ * sumRVRange X n ω

lemma measurable_sumRVRange {X : ℕ → Ω → ℝ}
    (h_meas : ∀ i, Measurable (X i)) (n : ℕ) :
    Measurable (sumRVRange X n) := by
  have h :=
    (Finset.measurable_sum (s := Finset.range n) (f := fun i ω => X i ω)
      (fun i hi => h_meas i))
  have h_eq :
      sumRVRange X n = fun ω => Finset.sum (Finset.range n) (fun i => X i ω) := by
    funext ω
    simp [sumRVRange, sumRV]
  simpa [h_eq] using h

lemma measurable_normalizedSum {X : ℕ → Ω → ℝ}
    (h_meas : ∀ i, Measurable (X i)) (n : ℕ) :
    Measurable (normalizedSum X n) := by
  simpa [normalizedSum] using
    (measurable_const_mul ((Real.sqrt (n : ℝ))⁻¹)).comp (measurable_sumRVRange h_meas n)

lemma variance_nonneg_of_integral_sq_eq {X : Ω → ℝ} {σ2 : ℝ}
    (h_var : ∫ x, x ^ 2 ∂ μ.map X = σ2) : 0 ≤ σ2 := by
  have h_nonneg : 0 ≤ ∫ x, x ^ 2 ∂ μ.map X := by
    refine integral_nonneg ?_
    intro x
    exact sq_nonneg x
  simpa [h_var] using h_nonneg

lemma cdf_continuousAt_measure_singleton_zero {μ : ProbabilityMeasure ℝ} {x : ℝ}
    (hx : ContinuousAt (cdf μ) x) : (μ : Measure ℝ) {x} = 0 := by
  have hmono := monotone_cdf (μ := (μ : Measure ℝ))
  have hright : Function.rightLim (cdf μ) x = cdf μ x := by
    simpa using (StieltjesFunction.rightLim_eq (cdf μ) x)
  have hleft : Function.leftLim (cdf μ) x = cdf μ x := by
    have hiff := (hmono.continuousAt_iff_leftLim_eq_rightLim (x := x))
    have h_eq := hiff.1 hx
    simpa [hright] using h_eq
  have h_measure :
      (μ : Measure ℝ) {x} = ENNReal.ofReal (cdf μ x - Function.leftLim (cdf μ) x) := by
    simpa [measure_cdf] using
      (StieltjesFunction.measure_singleton (f := cdf (μ : Measure ℝ)) x)
  simp [h_measure, hleft]

lemma tendsto_cdf_of_tendsto_probabilityMeasure
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ}
    (hμ : Tendsto μs atTop (𝓝 μ)) {x : ℝ} (hx : ContinuousAt (cdf μ) x) :
    Tendsto (fun n => cdf (μs n) x) atTop (𝓝 (cdf μ x)) := by
  have h_frontier : (μ : Measure ℝ) (frontier (Set.Iic x)) = 0 := by
    have h_singleton : (μ : Measure ℝ) {x} = 0 :=
      cdf_continuousAt_measure_singleton_zero (μ := μ) hx
    have h_frontier' : frontier (Set.Iic x) = ({x} : Set ℝ) := by
      simp [frontier_Iic]
    simpa [h_frontier'] using h_singleton
  have h_tendsto_measure :
      Tendsto (fun n => (μs n : Measure ℝ) (Set.Iic x)) atTop
        (𝓝 ((μ : Measure ℝ) (Set.Iic x))) := by
    exact ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto' (μs := μs) hμ h_frontier
  have h_tendsto_real :
      Tendsto (fun n => ((μs n : Measure ℝ) (Set.Iic x)).toReal) atTop
        (𝓝 (((μ : Measure ℝ) (Set.Iic x)).toReal)) := by
    exact (ENNReal.tendsto_toReal (measure_ne_top (μ := (μ : Measure ℝ)) (Set.Iic x))).comp
      h_tendsto_measure
  simpa [cdf_eq_real, measureReal_def] using h_tendsto_real

lemma charFun_sum_range {X : ℕ → Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (n : ℕ) (t : ℝ) :
    charFun (μ.map (sumRVRange X n)) t =
      Finset.prod (Finset.range n) (fun i => charFun (μ.map (X i)) t) := by
  simpa [sumRVRange] using
    (charFun_sum_finset (X := X) h_indep h_meas (s := Finset.range n) (t := t))

lemma charFun_sum_range_iid {X : ℕ → Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (n : ℕ) (t : ℝ) :
    charFun (μ.map (sumRVRange X n)) t =
      (charFun (μ.map (X 0)) t) ^ n := by
  have h_eq : ∀ i, charFun (μ.map (X i)) t = charFun (μ.map (X 0)) t := by
    intro i
    have h_map : μ.map (X i) = μ.map (X 0) := (h_ident i).map_eq
    simp [h_map]
  calc
    charFun (μ.map (sumRVRange X n)) t
        = Finset.prod (Finset.range n) (fun i => charFun (μ.map (X i)) t) := by
            simpa using (charFun_sum_range h_indep h_meas (n := n) (t := t))
    _ = Finset.prod (Finset.range n) (fun _ => charFun (μ.map (X 0)) t) := by
            refine Finset.prod_congr rfl ?_
            intro i hi
            simp [h_eq i]
    _ = (charFun (μ.map (X 0)) t) ^ n := by
            simp [Finset.prod_const, Finset.card_range]

lemma charFun_normalized_sum {X : ℕ → Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (n : ℕ) (t : ℝ) :
    charFun (μ.map (normalizedSum X n)) t =
      Finset.prod (Finset.range n)
        (fun i => charFun (μ.map (X i)) (t / Real.sqrt (n : ℝ))) := by
  classical
  set r : ℝ := (Real.sqrt (n : ℝ))⁻¹
  have h_sum_meas : Measurable (sumRVRange X n) := by
    have h :=
      (Finset.measurable_sum (s := Finset.range n) (f := fun i ω => X i ω)
        (fun i hi => h_meas i))
    have h_eq :
        sumRVRange X n = fun ω => Finset.sum (Finset.range n) (fun i => X i ω) := by
      funext ω
      simp [sumRVRange, sumRV]
    simpa [h_eq] using h
  have h_map :
      μ.map (normalizedSum X n) =
        (μ.map (sumRVRange X n)).map (fun x => r * x) := by
    have h := Measure.map_map (μ := μ) (f := sumRVRange X n) (g := fun x => r * x)
      (measurable_const_mul r) h_sum_meas
    simpa [normalizedSum, sumRVRange, Function.comp, r] using h.symm
  calc
    charFun (μ.map (normalizedSum X n)) t
        = charFun ((μ.map (sumRVRange X n)).map (fun x => r * x)) t := by
            simp [h_map]
    _ = charFun (μ.map (sumRVRange X n)) (r * t) := by
            simpa using (charFun_map_mul (μ := μ.map (sumRVRange X n)) r t)
    _ = Finset.prod (Finset.range n) (fun i => charFun (μ.map (X i)) (r * t)) := by
            simpa using (charFun_sum_range h_indep h_meas (n := n) (t := r * t))
    _ = Finset.prod (Finset.range n)
          (fun i => charFun (μ.map (X i)) (t / Real.sqrt (n : ℝ))) := by
            simp [r, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]

lemma charFun_normalized_sum_iid {X : ℕ → Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (n : ℕ) (t : ℝ) :
    charFun (μ.map (normalizedSum X n)) t =
      (charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ))) ^ n := by
  have h_eq :
      ∀ i, charFun (μ.map (X i)) (t / Real.sqrt (n : ℝ)) =
        charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ)) := by
    intro i
    have h_map : μ.map (X i) = μ.map (X 0) := (h_ident i).map_eq
    simp [h_map]
  calc
    charFun (μ.map (normalizedSum X n)) t
        = Finset.prod (Finset.range n)
            (fun i => charFun (μ.map (X i)) (t / Real.sqrt (n : ℝ))) := by
            simpa using (charFun_normalized_sum h_indep h_meas (n := n) (t := t))
    _ = Finset.prod (Finset.range n)
          (fun _ => charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ))) := by
            refine Finset.prod_congr rfl ?_
            intro i hi
            simp [h_eq i]
    _ = (charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ))) ^ n := by
            simp [Finset.prod_const, Finset.card_range]

/-- Second-order characteristic function scaling hypothesis for CLT. -/
def CharFunCLTScale (μ : Measure Ω) (X : Ω → ℝ) (σ2 : ℝ) : Prop :=
  ∀ t : ℝ,
    Tendsto
      (fun n : ℕ =>
        (n : ℂ) * (charFun (μ.map X) (t / Real.sqrt (n : ℝ)) - 1))
      atTop
      (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2))

lemma exp_remainder_bound :
    ∃ C δ, 0 < δ ∧ 0 ≤ C ∧ ∀ z : ℂ, ‖z‖ < δ →
      ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ C * ‖z‖ ^ 3 := by
  have hBigO :
      (fun z : ℂ => Complex.exp z - ∑ i ∈ Finset.range 3, z ^ i / (Nat.factorial i)) =O[𝓝 (0 : ℂ)] (· ^ 3) := by
    simpa using (Complex.exp_sub_sum_range_isBigO_pow 3)
  rcases hBigO.bound with ⟨C, hC⟩
  set C' : ℝ := max C 0
  have hC' : 0 ≤ C' := by simp [C']
  have hC_bound : ∀ᶠ z in 𝓝 (0 : ℂ),
      ‖Complex.exp z - (∑ i ∈ Finset.range 3, z ^ i / (Nat.factorial i))‖ ≤ C' * ‖z‖ ^ 3 := by
    refine hC.mono ?_
    intro z hz
    have hC_le : C ≤ C' := by simp [C']
    have hz_nonneg : 0 ≤ ‖z‖ ^ 3 := by positivity
    have hz' :
        ‖Complex.exp z - (∑ i ∈ Finset.range 3, z ^ i / (Nat.factorial i))‖ ≤ C * ‖z‖ ^ 3 := by
      simpa [norm_pow] using hz
    exact hz'.trans (mul_le_mul_of_nonneg_right hC_le hz_nonneg)
  rcases (Metric.eventually_nhds_iff.mp hC_bound) with ⟨δ, hδ, hδprop⟩
  refine ⟨C', δ, hδ, hC', ?_⟩
  intro z hz
  have hz' : dist z (0 : ℂ) < δ := by
    simpa [dist_eq_norm] using hz
  specialize hδprop hz'
  -- Simplify the truncated exponential series.
  have hsum :
      (∑ i ∈ Finset.range 3, z ^ i / (Nat.factorial i)) = 1 + z + z ^ 2 / 2 := by
    have hsum' :
        (∑ i ∈ Finset.range 3, z ^ i / (Nat.factorial i)) = z * z / 2 + (z + 1) := by
      simp [Finset.range_add_one, Finset.sum_insert, Finset.sum_range_succ, Nat.factorial,
        Nat.factorial_succ, Nat.succ_eq_add_one, pow_succ, mul_comm, mul_left_comm, mul_assoc]
    calc
      (∑ i ∈ Finset.range 3, z ^ i / (Nat.factorial i)) = z * z / 2 + (z + 1) := hsum'
      _ = 1 + z + z ^ 2 / 2 := by ring
  -- Use the big-O bound with the simplified sum.
  simpa [hsum, pow_succ, pow_two, pow_three] using hδprop

lemma exp_remainder_bound_global_imag :
    ∃ C, 0 ≤ C ∧ ∀ y : ℝ,
      ‖Complex.exp ((y : ℂ) * Complex.I)
          - (1 + (y : ℂ) * Complex.I + ((y : ℂ) * Complex.I) ^ 2 / 2)‖ ≤
        C * |y| ^ 3 := by
  rcases exp_remainder_bound with ⟨C0, δ, hδ, hC0nonneg, hC0bound⟩
  set Cbig : ℝ := 2 / δ ^ 3 + 1 / δ ^ 2 + 1 / (2 * δ)
  set C : ℝ := max C0 Cbig
  have hCnonneg : 0 ≤ C := by
    exact le_trans hC0nonneg (le_max_left _ _)
  refine ⟨C, hCnonneg, ?_⟩
  intro y
  set z : ℂ := (y : ℂ) * Complex.I
  have hz : ‖z‖ = |y| := by
    simp [z, Complex.norm_mul, Complex.norm_I]
  by_cases hsmall : |y| < δ
  · have hz' : ‖z‖ < δ := by
      simpa [hz] using hsmall
    have hC0le : C0 ≤ C := le_max_left _ _
    have hbound := hC0bound z hz'
    calc
      ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ C0 * ‖z‖ ^ 3 := hbound
      _ ≤ C * ‖z‖ ^ 3 := by
            exact mul_le_mul_of_nonneg_right hC0le (by positivity)
      _ = C * |y| ^ 3 := by
            simp [hz]
  · have hlarge : δ ≤ |y| := le_of_not_gt hsmall
    have hnorm_exp : ‖Complex.exp z‖ = 1 := by
      simp [z]
    have hnorm_poly :
        ‖1 + z + z ^ 2 / 2‖ ≤ 1 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
      have h1 : ‖1 + z + z ^ 2 / 2‖ ≤ ‖(1 : ℂ)‖ + ‖z + z ^ 2 / 2‖ := by
        simpa [add_assoc] using (norm_add_le (1 : ℂ) (z + z ^ 2 / 2))
      have h2 : ‖z + z ^ 2 / 2‖ ≤ ‖z‖ + ‖z ^ 2 / 2‖ := by
        exact norm_add_le z (z ^ 2 / 2)
      have hpow : ‖z ^ 2 / 2‖ = ‖z‖ ^ 2 / 2 := by
        simp [div_eq_mul_inv, norm_mul, norm_pow, mul_comm, mul_left_comm, mul_assoc]
      calc
        ‖1 + z + z ^ 2 / 2‖ ≤ ‖(1 : ℂ)‖ + ‖z + z ^ 2 / 2‖ := h1
        _ ≤ ‖(1 : ℂ)‖ + (‖z‖ + ‖z ^ 2 / 2‖) := by
              simpa [add_assoc, add_left_comm, add_comm] using
                (add_le_add_left h2 ‖(1 : ℂ)‖)
        _ = 1 + ‖z‖ + ‖z ^ 2 / 2‖ := by
              simp [add_assoc, add_left_comm, add_comm]
        _ = 1 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
              simp [hpow]
    have htri :
        ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ ‖Complex.exp z‖ + ‖1 + z + z ^ 2 / 2‖ := by
      simpa using (norm_sub_le (Complex.exp z) (1 + z + z ^ 2 / 2))
    have hbig :
        ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤
          2 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
      calc
        ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖
            ≤ ‖Complex.exp z‖ + ‖1 + z + z ^ 2 / 2‖ := htri
        _ ≤ 1 + (1 + ‖z‖ + ‖z‖ ^ 2 / 2) := by
              exact add_le_add (by simp [hnorm_exp]) hnorm_poly
        _ = 2 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
              ring
    have hCbig :
        2 + ‖z‖ + ‖z‖ ^ 2 / 2 ≤ Cbig * ‖z‖ ^ 3 := by
      have hδpos : 0 < δ := hδ
      have hδne : δ ≠ 0 := ne_of_gt hδpos
      have hzpos : 0 ≤ ‖z‖ := norm_nonneg _
      have hlarge' : δ ≤ ‖z‖ := by
        simpa [hz] using hlarge
      have h1 : (2 : ℝ) ≤ (2 / δ ^ 3) * ‖z‖ ^ 3 := by
        have hpow : δ ^ 3 ≤ ‖z‖ ^ 3 := by
          exact pow_le_pow_left₀ (by positivity) hlarge' 3
        calc
          (2 : ℝ) = (2 / δ ^ 3) * δ ^ 3 := by
            field_simp [hδne]
          _ ≤ (2 / δ ^ 3) * ‖z‖ ^ 3 := by
              exact mul_le_mul_of_nonneg_left hpow (by positivity)
      have h2 : ‖z‖ ≤ (1 / δ ^ 2) * ‖z‖ ^ 3 := by
        have hpow : δ ^ 2 ≤ ‖z‖ ^ 2 := by
          exact pow_le_pow_left₀ (by positivity) hlarge' 2
        have hmul : δ ^ 2 * ‖z‖ ≤ ‖z‖ ^ 3 := by
          calc
            δ ^ 2 * ‖z‖ ≤ ‖z‖ ^ 2 * ‖z‖ := by
                exact mul_le_mul_of_nonneg_right hpow hzpos
            _ = ‖z‖ ^ 3 := by ring
        calc
          ‖z‖ = (1 / δ ^ 2) * (δ ^ 2 * ‖z‖) := by
            field_simp [hδne]
          _ ≤ (1 / δ ^ 2) * ‖z‖ ^ 3 := by
              exact mul_le_mul_of_nonneg_left hmul (by positivity)
      have h3 : ‖z‖ ^ 2 / 2 ≤ (1 / (2 * δ)) * ‖z‖ ^ 3 := by
        have hmul : δ * ‖z‖ ^ 2 ≤ ‖z‖ ^ 3 := by
          calc
            δ * ‖z‖ ^ 2 ≤ ‖z‖ * ‖z‖ ^ 2 := by
                exact mul_le_mul_of_nonneg_right hlarge' (by positivity)
            _ = ‖z‖ ^ 3 := by ring
        calc
          ‖z‖ ^ 2 / 2 = (1 / (2 * δ)) * (δ * ‖z‖ ^ 2) := by
            field_simp [hδne]
          _ ≤ (1 / (2 * δ)) * ‖z‖ ^ 3 := by
              exact mul_le_mul_of_nonneg_left hmul (by positivity)
      have hsum :
          2 + ‖z‖ + ‖z‖ ^ 2 / 2 ≤
            (2 / δ ^ 3) * ‖z‖ ^ 3 + (1 / δ ^ 2) * ‖z‖ ^ 3 + (1 / (2 * δ)) * ‖z‖ ^ 3 := by
        linarith [h1, h2, h3]
      calc
        2 + ‖z‖ + ‖z‖ ^ 2 / 2 ≤
            (2 / δ ^ 3) * ‖z‖ ^ 3 + (1 / δ ^ 2) * ‖z‖ ^ 3 + (1 / (2 * δ)) * ‖z‖ ^ 3 := hsum
        _ = Cbig * ‖z‖ ^ 3 := by
            simp [Cbig, mul_add, add_mul, add_assoc, add_left_comm, add_comm]
    have hCbig_le : Cbig ≤ C := le_max_right _ _
    calc
      ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ 2 + ‖z‖ + ‖z‖ ^ 2 / 2 := hbig
      _ ≤ Cbig * ‖z‖ ^ 3 := hCbig
      _ ≤ C * ‖z‖ ^ 3 := by
            exact mul_le_mul_of_nonneg_right hCbig_le (by positivity)
      _ = C * |y| ^ 3 := by
            simp [hz]

lemma exp_remainder_bound_global_imag_sq :
    ∃ C, 0 ≤ C ∧ ∀ y : ℝ,
      ‖Complex.exp ((y : ℂ) * Complex.I)
          - (1 + (y : ℂ) * Complex.I + ((y : ℂ) * Complex.I) ^ 2 / 2)‖ ≤
        C * |y| ^ 2 := by
  rcases exp_remainder_bound with ⟨C0, δ, hδ, hC0nonneg, hC0bound⟩
  set Csmall : ℝ := C0 * δ
  set Cbig : ℝ := 2 / δ ^ 2 + 1 / δ + 1 / 2
  set C : ℝ := max Csmall Cbig
  have hCnonneg : 0 ≤ C := by
    have hCsmall_nonneg : 0 ≤ Csmall := by
      have hδnonneg : 0 ≤ δ := le_of_lt hδ
      exact mul_nonneg hC0nonneg hδnonneg
    exact le_trans hCsmall_nonneg (le_max_left _ _)
  refine ⟨C, hCnonneg, ?_⟩
  intro y
  set z : ℂ := (y : ℂ) * Complex.I
  have hz : ‖z‖ = |y| := by
    simp [z, Complex.norm_mul, Complex.norm_I]
  by_cases hsmall : |y| < δ
  · have hz' : ‖z‖ < δ := by
      simpa [hz] using hsmall
    have hCsmall_le : Csmall ≤ C := le_max_left _ _
    have hbound := hC0bound z hz'
    have hpow : ‖z‖ ^ 3 ≤ δ * ‖z‖ ^ 2 := by
      have hle : ‖z‖ ≤ δ := le_of_lt hz'
      have hnonneg : 0 ≤ ‖z‖ ^ 2 := by positivity
      have hmul : ‖z‖ ^ 2 * ‖z‖ ≤ ‖z‖ ^ 2 * δ := by
        exact mul_le_mul_of_nonneg_left hle hnonneg
      simpa [pow_succ, pow_two, mul_comm, mul_left_comm, mul_assoc] using hmul
    calc
      ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ C0 * ‖z‖ ^ 3 := hbound
      _ ≤ C0 * (δ * ‖z‖ ^ 2) := by
            exact mul_le_mul_of_nonneg_left hpow hC0nonneg
      _ = Csmall * ‖z‖ ^ 2 := by
            ring
      _ ≤ C * ‖z‖ ^ 2 := by
            exact mul_le_mul_of_nonneg_right hCsmall_le (by positivity)
      _ = C * |y| ^ 2 := by
            simp [hz]
  · have hlarge : δ ≤ |y| := le_of_not_gt hsmall
    have hnorm_exp : ‖Complex.exp z‖ = 1 := by
      simp [z]
    have hnorm_poly :
        ‖1 + z + z ^ 2 / 2‖ ≤ 1 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
      have h1 : ‖1 + z + z ^ 2 / 2‖ ≤ ‖(1 : ℂ)‖ + ‖z + z ^ 2 / 2‖ := by
        simpa [add_assoc] using (norm_add_le (1 : ℂ) (z + z ^ 2 / 2))
      have h2 : ‖z + z ^ 2 / 2‖ ≤ ‖z‖ + ‖z ^ 2 / 2‖ := by
        exact norm_add_le z (z ^ 2 / 2)
      have hpow : ‖z ^ 2 / 2‖ = ‖z‖ ^ 2 / 2 := by
        simp [div_eq_mul_inv, norm_mul, norm_pow, mul_comm, mul_left_comm, mul_assoc]
      calc
        ‖1 + z + z ^ 2 / 2‖ ≤ ‖(1 : ℂ)‖ + ‖z + z ^ 2 / 2‖ := h1
        _ ≤ ‖(1 : ℂ)‖ + (‖z‖ + ‖z ^ 2 / 2‖) := by
              simpa [add_assoc, add_left_comm, add_comm] using
                (add_le_add_left h2 ‖(1 : ℂ)‖)
        _ = 1 + ‖z‖ + ‖z ^ 2 / 2‖ := by
              simp [add_assoc, add_left_comm, add_comm]
        _ = 1 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
              simp [hpow]
    have htri :
        ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ ‖Complex.exp z‖ + ‖1 + z + z ^ 2 / 2‖ := by
      simpa using (norm_sub_le (Complex.exp z) (1 + z + z ^ 2 / 2))
    have hbig :
        ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤
          2 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
      calc
        ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖
            ≤ ‖Complex.exp z‖ + ‖1 + z + z ^ 2 / 2‖ := htri
        _ ≤ 1 + (1 + ‖z‖ + ‖z‖ ^ 2 / 2) := by
              exact add_le_add (by simp [hnorm_exp]) hnorm_poly
        _ = 2 + ‖z‖ + ‖z‖ ^ 2 / 2 := by
              ring
    have hCbig :
        2 + ‖z‖ + ‖z‖ ^ 2 / 2 ≤ Cbig * ‖z‖ ^ 2 := by
      have hδpos : 0 < δ := hδ
      have hδne : δ ≠ 0 := ne_of_gt hδpos
      have hzpos : 0 ≤ ‖z‖ := norm_nonneg _
      have hlarge' : δ ≤ ‖z‖ := by
        simpa [hz] using hlarge
      have h1 : (2 : ℝ) ≤ (2 / δ ^ 2) * ‖z‖ ^ 2 := by
        have hpow : δ ^ 2 ≤ ‖z‖ ^ 2 := by
          exact pow_le_pow_left₀ (by positivity) hlarge' 2
        calc
          (2 : ℝ) = (2 / δ ^ 2) * δ ^ 2 := by
            field_simp [hδne]
          _ ≤ (2 / δ ^ 2) * ‖z‖ ^ 2 := by
              exact mul_le_mul_of_nonneg_left hpow (by positivity)
      have h2 : ‖z‖ ≤ (1 / δ) * ‖z‖ ^ 2 := by
        have hmul : δ * ‖z‖ ≤ ‖z‖ ^ 2 := by
          calc
            δ * ‖z‖ ≤ ‖z‖ * ‖z‖ := by
                exact mul_le_mul_of_nonneg_right hlarge' hzpos
            _ = ‖z‖ ^ 2 := by
                ring
        calc
          ‖z‖ = (1 / δ) * (δ * ‖z‖) := by
            field_simp [hδne]
          _ ≤ (1 / δ) * ‖z‖ ^ 2 := by
              exact mul_le_mul_of_nonneg_left hmul (by positivity)
      have h3 : ‖z‖ ^ 2 / 2 ≤ (1 / 2) * ‖z‖ ^ 2 := by
        have h_eq : ‖z‖ ^ 2 / 2 = (1 / 2) * ‖z‖ ^ 2 := by
          ring
        simp [h_eq]
      have hsum :
          2 + ‖z‖ + ‖z‖ ^ 2 / 2 ≤
            (2 / δ ^ 2) * ‖z‖ ^ 2 + (1 / δ) * ‖z‖ ^ 2 + (1 / 2) * ‖z‖ ^ 2 := by
        linarith [h1, h2, h3]
      calc
        2 + ‖z‖ + ‖z‖ ^ 2 / 2 ≤
            (2 / δ ^ 2) * ‖z‖ ^ 2 + (1 / δ) * ‖z‖ ^ 2 + (1 / 2) * ‖z‖ ^ 2 := hsum
        _ = Cbig * ‖z‖ ^ 2 := by
            simp [Cbig, mul_add, add_mul, add_assoc, add_left_comm, add_comm]
    have hCbig_le : Cbig ≤ C := le_max_right _ _
    calc
      ‖Complex.exp z - (1 + z + z ^ 2 / 2)‖ ≤ 2 + ‖z‖ + ‖z‖ ^ 2 / 2 := hbig
      _ ≤ Cbig * ‖z‖ ^ 2 := hCbig
      _ ≤ C * ‖z‖ ^ 2 := by
            exact mul_le_mul_of_nonneg_right hCbig_le (by positivity)
      _ = C * |y| ^ 2 := by
            simp [hz]

lemma integrable_of_norm_bounded {α : Type*} [MeasurableSpace α] {μ : Measure α}
    [IsFiniteMeasure μ]
    {f : α → ℂ} (hf : AEStronglyMeasurable f μ)
    {C : ℝ} (hbound : ∀ᵐ x ∂μ, ‖f x‖ ≤ C) :
    Integrable f μ := by
  exact Integrable.mono' (integrable_const (μ := μ) (c := C)) hf hbound

lemma abs_le_one_add_abs_sq (x : ℝ) : |x| ≤ 1 + |x| ^ 2 := by
  by_cases h : |x| ≤ 1
  · have hnonneg : 0 ≤ |x| ^ 2 := by positivity
    exact h.trans (le_add_of_nonneg_right hnonneg)
  · have hge : 1 ≤ |x| := le_of_not_ge h
    have hpow : |x| ≤ |x| ^ 2 := by
      exact le_self_pow₀ hge (by decide : (2 : ℕ) ≠ 0)
    exact hpow.trans (le_add_of_nonneg_left (by positivity))

lemma abs_le_one_add_abs_pow3 (x : ℝ) : |x| ≤ 1 + |x| ^ 3 := by
  by_cases h : |x| ≤ 1
  · have hnonneg : 0 ≤ |x| ^ 3 := by positivity
    exact h.trans (le_add_of_nonneg_right hnonneg)
  · have hge : 1 ≤ |x| := le_of_not_ge h
    have hpow : |x| ≤ |x| ^ 3 := by
      exact le_self_pow₀ hge (by decide : (3 : ℕ) ≠ 0)
    exact hpow.trans (le_add_of_nonneg_left (by positivity))

lemma abs_sq_le_one_add_abs_pow3 (x : ℝ) : |x| ^ 2 ≤ 1 + |x| ^ 3 := by
  by_cases h : |x| ≤ 1
  · have hpow : |x| ^ 2 ≤ 1 := by
      simpa using (pow_le_pow_left₀ (abs_nonneg x) h 2)
    have hnonneg : 0 ≤ |x| ^ 3 := by positivity
    exact hpow.trans (le_add_of_nonneg_right hnonneg)
  · have hge : 1 ≤ |x| := le_of_not_ge h
    have hnonneg : 0 ≤ |x| ^ 2 := by positivity
    have hmul : |x| ^ 2 ≤ |x| ^ 2 * |x| := by
      have hmul' : |x| ^ 2 * 1 ≤ |x| ^ 2 * |x| := by
        exact mul_le_mul_of_nonneg_left hge hnonneg
      simpa using hmul'
    have hpow : |x| ^ 2 ≤ |x| ^ 3 := by
      simpa [pow_succ, pow_two, mul_comm, mul_left_comm, mul_assoc] using hmul
    exact hpow.trans (le_add_of_nonneg_left (by positivity))

lemma tendsto_inv_sqrt_nat :
    Tendsto (fun n : ℕ => (Real.sqrt (n : ℝ))⁻¹) atTop (𝓝 (0 : ℝ)) := by
  have h :
      Tendsto (fun x : ℝ => x ^ (-(1 / 2 : ℝ))) atTop (𝓝 0) :=
    tendsto_rpow_neg_atTop (by norm_num : 0 < (1 / 2 : ℝ))
  have h' :
      Tendsto (fun n : ℕ => (n : ℝ) ^ (-(1 / 2 : ℝ))) atTop (𝓝 0) :=
    h.comp tendsto_natCast_atTop_atTop
  refine (tendsto_congr ?_).mpr h'
  intro n
  have hn : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
  calc
    (Real.sqrt (n : ℝ))⁻¹ = ((n : ℝ) ^ (1 / 2 : ℝ))⁻¹ := by
      simp [Real.sqrt_eq_rpow]
    _ = (n : ℝ) ^ (-(1 / 2 : ℝ)) := by
      symm
      simp [Real.rpow_neg hn]

theorem CharFunCLTScale_of_bounded {X : Ω → ℝ} {σ2 M : ℝ}
    (h_meas : Measurable X)
    (h_bound : ∀ᵐ ω ∂μ, |X ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map X = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map X = σ2) :
    CharFunCLTScale μ X σ2 := by
  classical
  set ν : Measure ℝ := μ.map X
  have hprob : IsProbabilityMeasure ν := Measure.isProbabilityMeasure_map (by fun_prop)
  have h_mean' : ∫ x, x ∂ν = 0 := by
    simpa [ν] using h_mean
  have h_var' : ∫ x, x ^ 2 ∂ν = σ2 := by
    simpa [ν] using h_var
  have h_bound' : ∀ᵐ x ∂ν, |x| ≤ |M| := by
    have hX : AEMeasurable X μ := h_meas.aemeasurable
    have h' : ∀ᵐ x ∂μ, |X x| ≤ |M| := by
      filter_upwards [h_bound] with x hx
      exact le_trans hx (le_abs_self M)
    have hset : MeasurableSet {x : ℝ | |x| ≤ |M|} :=
      measurableSet_le measurable_abs measurable_const
    exact (MeasureTheory.ae_map_iff (μ := μ) (f := X) hX hset).2 h'
  have h_int_x : Integrable (fun x : ℝ => (x : ℂ)) ν := by
    refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => (x : ℂ)) (C := |M|) ?_ ?_
    · exact (by fun_prop : Measurable fun x : ℝ => (x : ℂ)).aestronglyMeasurable
    · filter_upwards [h_bound'] with x hx
      simpa [Complex.norm_real, Real.norm_eq_abs] using hx
  have h_int_x2 : Integrable (fun x : ℝ => (x ^ 2 : ℂ)) ν := by
    refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => (x ^ 2 : ℂ)) (C := |M| ^ 2) ?_ ?_
    · exact (by fun_prop : Measurable fun x : ℝ => (x ^ 2 : ℂ)).aestronglyMeasurable
    · filter_upwards [h_bound'] with x hx
      have hx' : |x| ^ 2 ≤ |M| ^ 2 := by
        exact pow_le_pow_left₀ (abs_nonneg x) hx 2
      simpa [Complex.norm_real, Real.norm_eq_abs, abs_pow] using hx'
  rcases exp_remainder_bound with ⟨C, δ, hδ, hCnonneg, hCbound⟩
  intro t
  let u : ℕ → ℝ := fun n => t / Real.sqrt (n : ℝ)
  let z : ℕ → ℝ → ℂ := fun n x => ((u n * x) : ℂ) * Complex.I
  let rem : ℕ → ℝ → ℂ := fun n x =>
    Complex.exp (z n x) - ((1 : ℂ) + z n x + (z n x) ^ 2 / 2)
  have h_mean_c : ∫ x, (x : ℂ) ∂ν = (0 : ℂ) := by
    have h_cast : ((∫ x, x ∂ν : ℝ) : ℂ) = (0 : ℂ) :=
      congrArg (fun r : ℝ => (r : ℂ)) h_mean'
    have h_int : ∫ x, (x : ℂ) ∂ν = ((∫ x, x ∂ν : ℝ) : ℂ) :=
      integral_complex_ofReal (μ := ν) (f := fun x : ℝ => x)
    calc
      ∫ x, (x : ℂ) ∂ν = ((∫ x, x ∂ν : ℝ) : ℂ) := h_int
      _ = (0 : ℂ) := h_cast
  have h_var_c : ∫ x, (x ^ 2 : ℂ) ∂ν = (σ2 : ℂ) := by
    have h_cast : ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) = (σ2 : ℂ) :=
      congrArg (fun r : ℝ => (r : ℂ)) h_var'
    have h_int : ∫ x, (x ^ 2 : ℂ) ∂ν = ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) := by
      simpa [pow_two] using
        (integral_complex_ofReal (μ := ν) (f := fun x : ℝ => x * x))
    calc
      ∫ x, (x ^ 2 : ℂ) ∂ν = ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) := h_int
      _ = (σ2 : ℂ) := h_cast
  have h_u_tendsto : Tendsto (fun n : ℕ => u n) atTop (𝓝 0) := by
    have h_sqrt : Tendsto (fun n : ℕ => (Real.sqrt (n : ℝ))⁻¹) atTop (𝓝 (0 : ℝ)) :=
      tendsto_inv_sqrt_nat
    simpa [u, div_eq_mul_inv] using (tendsto_const_nhds.mul h_sqrt)
  have h_abs_u : Tendsto (fun n : ℕ => |u n|) atTop (𝓝 (0 : ℝ)) := by
    simpa using (continuous_abs.tendsto 0).comp h_u_tendsto
  have h_abs_u_mul : Tendsto (fun n : ℕ => |u n| * |M|) atTop (𝓝 (0 : ℝ)) := by
    simpa [mul_comm] using h_abs_u.const_mul |M|
  have h_u_small : ∀ᶠ n : ℕ in atTop, |u n| * |M| < δ := by
    exact (tendsto_order.1 h_abs_u_mul).2 δ hδ
  have h_rem_bound_ae :
      ∀ᶠ n : ℕ in atTop, ∀ᵐ x ∂ν, ‖rem n x‖ ≤ C * |u n| ^ 3 * |M| ^ 3 := by
    refine h_u_small.mono ?_
    intro n hn
    filter_upwards [h_bound'] with x hx
    have hxmul : |u n| * |x| ≤ |u n| * |M| := by
      exact mul_le_mul_of_nonneg_left hx (abs_nonneg (u n))
    have hz' : |u n * x| < δ := by
      exact lt_of_le_of_lt (by simpa [abs_mul] using hxmul) hn
    have hz : ‖(z n x : ℂ)‖ < δ := by
      simpa [z, norm_mul, Complex.norm_I, Complex.norm_real, Real.norm_eq_abs, abs_mul] using hz'
    have hrem : ‖rem n x‖ ≤ C * ‖(z n x : ℂ)‖ ^ 3 := by
      simpa [rem] using hCbound (z n x) hz
    have hnorm_le : ‖(z n x : ℂ)‖ ≤ |u n| * |M| := by
      simpa [z, norm_mul, Complex.norm_I, Complex.norm_real, Real.norm_eq_abs, abs_mul] using hxmul
    have hpow_le : ‖(z n x : ℂ)‖ ^ 3 ≤ (|u n| * |M|) ^ 3 := by
      exact pow_le_pow_left₀ (by positivity) hnorm_le 3
    calc
      ‖rem n x‖ ≤ C * ‖(z n x : ℂ)‖ ^ 3 := hrem
      _ ≤ C * (|u n| * |M|) ^ 3 := by
        exact mul_le_mul_of_nonneg_left hpow_le hCnonneg
      _ = C * |u n| ^ 3 * |M| ^ 3 := by
        ring
  have h_rem_norm_bound :
      ∀ᶠ n : ℕ in atTop, ‖∫ x, rem n x ∂ν‖ ≤ C * |u n| ^ 3 * |M| ^ 3 := by
    refine h_rem_bound_ae.mono ?_
    intro n h_ae
    have h_int_rem : Integrable (fun x => rem n x) ν := by
      -- Bound via exponential and polynomial terms.
      have h_exp : Integrable (fun x : ℝ => Complex.exp (z n x)) ν := by
        refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => Complex.exp (z n x))
          (C := 1) ?_ ?_
        · exact (by fun_prop : Measurable fun x : ℝ => Complex.exp (z n x)).aestronglyMeasurable
        · filter_upwards with x
          have hnorm : ‖Complex.exp (z n x)‖ = 1 := by
            simpa [z] using (Complex.norm_exp_ofReal_mul_I (u n * x))
          exact le_of_eq hnorm
      have h_z : Integrable (fun x => z n x) ν := by
        simpa [z, mul_assoc, mul_left_comm, mul_comm] using (h_int_x.const_mul (u n * Complex.I))
      have h_z2 : Integrable (fun x => (z n x) ^ 2 / 2) ν := by
        have h_const :
            Integrable (fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ)) ν := by
          exact h_int_x2.const_mul ((u n * Complex.I) ^ 2 / 2)
        have h_eq :
            (fun x : ℝ => (z n x) ^ 2 / 2) =
              fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
          funext x
          ring
        simpa [h_eq] using h_const
      have h_poly : Integrable (fun x => (1 : ℂ) + z n x + (z n x) ^ 2 / 2) ν := by
        have h1 : Integrable (fun _ : ℝ => (1 : ℂ)) ν := by
          simp
        have h12 : Integrable (fun x => (1 : ℂ) + z n x) ν := h1.add h_z
        exact h12.add h_z2
      simpa [rem] using h_exp.sub h_poly
    have h_int_norm : Integrable (fun x => ‖rem n x‖) ν := h_int_rem.norm
    have h_bound_const : Integrable (fun _ => C * |u n| ^ 3 * |M| ^ 3) ν := by
      exact integrable_const (μ := ν) (c := C * |u n| ^ 3 * |M| ^ 3)
    have h_le :
        ∫ x, ‖rem n x‖ ∂ν ≤ ∫ x, C * |u n| ^ 3 * |M| ^ 3 ∂ν := by
      exact integral_mono_ae h_int_norm h_bound_const h_ae
    have h_const :
        ∫ x, C * |u n| ^ 3 * |M| ^ 3 ∂ν = C * |u n| ^ 3 * |M| ^ 3 := by
      simp [integral_const, hprob.measure_univ]
    calc
      ‖∫ x, rem n x ∂ν‖ ≤ ∫ x, ‖rem n x‖ ∂ν :=
        norm_integral_le_integral_norm _
      _ ≤ ∫ x, C * |u n| ^ 3 * |M| ^ 3 ∂ν := h_le
      _ = C * |u n| ^ 3 * |M| ^ 3 := h_const
  have h_decomp :
      ∀ n : ℕ, (charFun ν (u n) - 1) =
        (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν)
          + ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν)
          + ∫ x, rem n x ∂ν := by
    intro n
    have h_exp : Integrable (fun x : ℝ => Complex.exp (z n x)) ν := by
      refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => Complex.exp (z n x))
        (C := 1) ?_ ?_
      · exact (by fun_prop : Measurable fun x : ℝ => Complex.exp (z n x)).aestronglyMeasurable
      · filter_upwards with x
        have hnorm : ‖Complex.exp (z n x)‖ = 1 := by
          simpa [z] using (Complex.norm_exp_ofReal_mul_I (u n * x))
        exact le_of_eq hnorm
    have h_z : Integrable (fun x => z n x) ν := by
      simpa [z, mul_assoc, mul_left_comm, mul_comm] using (h_int_x.const_mul (u n * Complex.I))
    have h_z2 : Integrable (fun x => (z n x) ^ 2 / 2) ν := by
      have h_const :
          Integrable (fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ)) ν := by
        exact h_int_x2.const_mul ((u n * Complex.I) ^ 2 / 2)
      have h_eq :
          (fun x : ℝ => (z n x) ^ 2 / 2) =
            fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
        funext x
        ring
      simpa [h_eq] using h_const
    have h_poly : Integrable (fun x => (1 : ℂ) + z n x + (z n x) ^ 2 / 2) ν := by
      have h1 : Integrable (fun _ : ℝ => (1 : ℂ)) ν := by
        simp
      have h12 : Integrable (fun x => (1 : ℂ) + z n x) ν := h1.add h_z
      exact h12.add h_z2
    have h_rem : Integrable (fun x => rem n x) ν := by
      simpa [rem] using h_exp.sub h_poly
    have h_exp_sub : ∀ x : ℝ, Complex.exp (z n x) - (1 : ℂ) =
        z n x + (z n x) ^ 2 / 2 + rem n x := by
      intro x
      simp [rem, add_assoc, add_left_comm, add_comm, sub_eq_add_neg]
    have h_char :
        charFun ν (u n) - 1 =
          ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := by
      have h_sub : ∫ x, Complex.exp (z n x) - (1 : ℂ) ∂ν =
          ∫ x, Complex.exp (z n x) ∂ν - ∫ x, (1 : ℂ) ∂ν := by
        exact integral_sub h_exp (integrable_const (μ := ν) (c := (1 : ℂ)))
      calc
        charFun ν (u n) - 1
            = ∫ x, Complex.exp (z n x) ∂ν - ∫ x, (1 : ℂ) ∂ν := by
                simp [charFun_apply_real, z, mul_assoc, mul_left_comm, mul_comm,
                  integral_const, hprob.measure_univ]
        _ = ∫ x, Complex.exp (z n x) - 1 ∂ν := by
                simpa using h_sub.symm
        _ = ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := by
                refine integral_congr_ae ?_
                exact ae_of_all _ (fun x => h_exp_sub x)
    have h_split :
        ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν
          = ∫ x, z n x ∂ν
            + ∫ x, (z n x) ^ 2 / 2 ∂ν
            + ∫ x, rem n x ∂ν := by
      have h12 : ∫ x, z n x + (z n x) ^ 2 / 2 ∂ν =
          ∫ x, z n x ∂ν + ∫ x, (z n x) ^ 2 / 2 ∂ν := by
        simpa using (integral_add h_z h_z2)
      have h123 : ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν =
          ∫ x, z n x + (z n x) ^ 2 / 2 ∂ν + ∫ x, rem n x ∂ν := by
        simpa [add_assoc] using (integral_add (h_z.add h_z2) h_rem)
      simpa [h12, add_assoc] using h123
    have h_int_z : ∫ x, z n x ∂ν = (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν) := by
      have h_eq : (fun x : ℝ => z n x) = fun x : ℝ => (u n * Complex.I) * (x : ℂ) := by
        funext x
        ring
      simpa [h_eq] using
        (integral_const_mul (μ := ν) (r := (u n * Complex.I)) (f := fun x : ℝ => (x : ℂ)))
    have h_int_z2 :
        ∫ x, (z n x) ^ 2 / 2 ∂ν =
          ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν) := by
      have h_eq :
          (fun x : ℝ => (z n x) ^ 2 / 2) =
            fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
        funext x
        ring
      simpa [h_eq] using
        (integral_const_mul (μ := ν) (r := (u n * Complex.I) ^ 2 / 2)
          (f := fun x : ℝ => (x ^ 2 : ℂ)))
    calc
      charFun ν (u n) - 1 =
          ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := h_char
      _ = ∫ x, z n x ∂ν
            + ∫ x, (z n x) ^ 2 / 2 ∂ν
            + ∫ x, rem n x ∂ν := h_split
      _ = (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν)
            + ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν)
            + ∫ x, rem n x ∂ν := by
            simp [h_int_z, h_int_z2]
  have h_simpl :
      ∀ n : ℕ, charFun ν (u n) - 1 =
        (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ) + ∫ x, rem n x ∂ν := by
    intro n
    have h := h_decomp n
    have h' :
        charFun ν (u n) - 1 =
          ((u n * Complex.I) ^ 2 / 2) * (σ2 : ℂ) + ∫ x, rem n x ∂ν := by
      simpa [h_mean_c, h_var_c, add_assoc, add_left_comm, add_comm] using h
    have h_coeff : ((u n * Complex.I) ^ 2 / 2 : ℂ) = (-(u n) ^ 2 / 2 : ℂ) := by
      calc
        ((u n * Complex.I) ^ 2 / 2 : ℂ)
            = ((u n * Complex.I) * (u n * Complex.I)) / 2 := by
                simp [pow_two]
        _ = ((u n * u n) * (Complex.I * Complex.I)) / 2 := by
                ring
        _ = (-(u n * u n)) / 2 := by
                simp [Complex.I_mul_I, mul_assoc, mul_left_comm, mul_comm]
        _ = (-(u n) ^ 2 / 2 : ℂ) := by
                simp [pow_two]
    simpa [h_coeff] using h'
  have h_rem_tendsto :
      Tendsto (fun n : ℕ => (n : ℂ) * ∫ x, rem n x ∂ν) atTop (𝓝 (0 : ℂ)) := by
    have h_bound :
        ∀ᶠ n : ℕ in atTop, ‖(n : ℂ) * ∫ x, rem n x ∂ν‖ ≤
          (C * |t| ^ 3 * |M| ^ 3) / Real.sqrt (n : ℝ) := by
      refine h_rem_norm_bound.mono ?_
      intro n h
      have hnorm : ‖(n : ℂ) * ∫ x, rem n x ∂ν‖ = (n : ℝ) * ‖∫ x, rem n x ∂ν‖ := by
        have hn : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
        simp [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hn]
      have h_abs_u : |u n| = |t| / Real.sqrt (n : ℝ) := by
        simp [u, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
      have h_u_pow : |u n| ^ 3 = |t| ^ 3 / (Real.sqrt (n : ℝ)) ^ 3 := by
        simpa [h_abs_u] using (div_pow |t| (Real.sqrt (n : ℝ)) 3)
      have h_nu :
          (n : ℝ) * (C * |u n| ^ 3 * |M| ^ 3) =
            (C * |t| ^ 3 * |M| ^ 3) / Real.sqrt (n : ℝ) := by
        have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
          exact Real.sq_sqrt (by positivity)
        calc
          (n : ℝ) * (C * |u n| ^ 3 * |M| ^ 3)
              = (n : ℝ) * (C * (|t| ^ 3 / (Real.sqrt (n : ℝ)) ^ 3) * |M| ^ 3) := by
                  simp [h_u_pow]
          _ = C * |t| ^ 3 * |M| ^ 3 * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 3) := by
                  ring
          _ = C * |t| ^ 3 * |M| ^ 3 / Real.sqrt (n : ℝ) := by
                  have h' : (n : ℝ) / (Real.sqrt (n : ℝ)) ^ 3 = 1 / Real.sqrt (n : ℝ) := by
                    by_cases hs : Real.sqrt (n : ℝ) = 0
                    · simp [hs]
                    ·
                      have hsq' : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
                        exact Real.sq_sqrt (by positivity)
                      field_simp [hs, hsq', pow_succ, mul_assoc, mul_left_comm, mul_comm]
                      simp [hsq']
                  simp [h', div_eq_mul_inv, mul_assoc]
          _ = (C * |t| ^ 3 * |M| ^ 3) / Real.sqrt (n : ℝ) := by
                  ring
      calc
        ‖(n : ℂ) * ∫ x, rem n x ∂ν‖
            = (n : ℝ) * ‖∫ x, rem n x ∂ν‖ := hnorm
        _ ≤ (n : ℝ) * (C * |u n| ^ 3 * |M| ^ 3) := by
            exact mul_le_mul_of_nonneg_left h (Nat.cast_nonneg n)
        _ = (C * |t| ^ 3 * |M| ^ 3) / Real.sqrt (n : ℝ) := h_nu
    have h_tendsto :
        Tendsto (fun n : ℕ => (C * |t| ^ 3 * |M| ^ 3) / Real.sqrt (n : ℝ))
          atTop (𝓝 (0 : ℝ)) := by
      have h_sqrt := tendsto_inv_sqrt_nat
      simpa [div_eq_mul_inv] using (tendsto_const_nhds.mul h_sqrt)
    have h_norm_tendsto :
        Tendsto (fun n : ℕ => ‖(n : ℂ) * ∫ x, rem n x ∂ν‖) atTop (𝓝 (0 : ℝ)) := by
      refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h_tendsto ?_ h_bound
      exact Filter.Eventually.of_forall (fun n => norm_nonneg _)
    -- Upgrade from norm convergence to complex convergence.
    exact (tendsto_iff_norm_sub_tendsto_zero).2 <| by
      simpa using h_norm_tendsto
  have h_main :
      Tendsto
        (fun n : ℕ => (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    have h_nu2 : ∀ᶠ n : ℕ in atTop, (n : ℂ) * (u n) ^ 2 = (t : ℂ) ^ 2 := by
      refine (eventually_ge_atTop 1).mono ?_
      intro n hn
      have hn0 : (n : ℝ) ≠ 0 := by
        exact_mod_cast (Nat.succ_le_iff.mp hn).ne'
      have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
        exact Real.sq_sqrt (by positivity)
      have h_real : (n : ℝ) * (u n) ^ 2 = t ^ 2 := by
        have h_u2 : (u n) ^ 2 = t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2 := by
          simpa [u] using (div_pow t (Real.sqrt (n : ℝ)) 2)
        calc
          (n : ℝ) * (u n) ^ 2 = (n : ℝ) * (t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
              simp [h_u2]
          _ = t ^ 2 := by
              calc
                (n : ℝ) * (t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2)
                    = t ^ 2 * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 2) := by
                        ring
                _ = t ^ 2 * ((Real.sqrt (n : ℝ)) ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
                        simp [hsq]
                _ = t ^ 2 := by
                        simp [div_self, hn0]
      exact_mod_cast h_real
    refine (tendsto_congr' ?_).mpr tendsto_const_nhds
    refine h_nu2.mono ?_
    intro n hn
    calc
      (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
          = (-(σ2 : ℂ) / 2) * ((n : ℂ) * (u n) ^ 2) := by
              ring
      _ = (-(σ2 : ℂ) / 2) * (t : ℂ) ^ 2 := by
              simp [hn]
      _ = (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
              ring
  have h_decomp' :
      ∀ n : ℕ, (n : ℂ) * (charFun ν (u n) - 1) =
        (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
          + (n : ℂ) * ∫ x, rem n x ∂ν := by
    intro n
    simp [h_simpl n, mul_add, mul_assoc, mul_left_comm, mul_comm]
  have h_sum := h_main.add h_rem_tendsto
  have h_sum' :
      Tendsto
        (fun n : ℕ =>
          (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
            + (n : ℂ) * ∫ x, rem n x ∂ν) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    simpa using h_sum
  refine (tendsto_congr ?_).1 h_sum'
  intro n
  simpa using (h_decomp' n).symm

theorem CharFunCLTScale_of_integrable_abs_pow3 {X : Ω → ℝ} {σ2 : ℝ}
    (h_meas : Measurable X)
    (h_int_abs3 : Integrable (fun x : ℝ => |x| ^ 3) (μ.map X))
    (h_mean : ∫ x, x ∂ μ.map X = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map X = σ2) :
    CharFunCLTScale μ X σ2 := by
  classical
  set ν : Measure ℝ := μ.map X
  have hprob : IsProbabilityMeasure ν := Measure.isProbabilityMeasure_map (by fun_prop)
  have h_mean' : ∫ x, x ∂ν = 0 := by
    simpa [ν] using h_mean
  have h_var' : ∫ x, x ^ 2 ∂ν = σ2 := by
    simpa [ν] using h_var
  have h_int_abs3' : Integrable (fun x : ℝ => |x| ^ 3) ν := by
    simpa [ν] using h_int_abs3
  have h_int_bound : Integrable (fun x : ℝ => 1 + |x| ^ 3) ν := by
    have h1 : Integrable (fun _ : ℝ => (1 : ℝ)) ν := by
      simp
    exact h1.add h_int_abs3'
  have h_int_x' : Integrable (fun x : ℝ => (x : ℂ)) ν := by
    refine Integrable.mono' h_int_bound
      (by fun_prop : AEStronglyMeasurable (fun x : ℝ => (x : ℂ)) ν) ?_
    refine Filter.Eventually.of_forall ?_
    intro x
    have h_le : |x| ≤ 1 + |x| ^ 3 := abs_le_one_add_abs_pow3 x
    simpa [Complex.norm_real, Real.norm_eq_abs] using h_le
  have h_int_x2' : Integrable (fun x : ℝ => (x ^ 2 : ℂ)) ν := by
    refine Integrable.mono' h_int_bound
      (by fun_prop : AEStronglyMeasurable (fun x : ℝ => (x ^ 2 : ℂ)) ν) ?_
    refine Filter.Eventually.of_forall ?_
    intro x
    have h_le : |x| ^ 2 ≤ 1 + |x| ^ 3 := abs_sq_le_one_add_abs_pow3 x
    have h_norm : ‖(x ^ 2 : ℂ)‖ = |x| ^ 2 := by
      calc
        ‖(x ^ 2 : ℂ)‖ = |x ^ 2| := by
          simp [Complex.norm_real, Real.norm_eq_abs]
        _ = |x| ^ 2 := by
          simp [pow_two]
    calc
      ‖(x ^ 2 : ℂ)‖ = |x| ^ 2 := h_norm
      _ ≤ 1 + |x| ^ 3 := h_le
  have h_mean_c : ∫ x, (x : ℂ) ∂ν = (0 : ℂ) := by
    have h_cast : ((∫ x, x ∂ν : ℝ) : ℂ) = (0 : ℂ) :=
      congrArg (fun r : ℝ => (r : ℂ)) h_mean'
    have h_int : ∫ x, (x : ℂ) ∂ν = ((∫ x, x ∂ν : ℝ) : ℂ) :=
      integral_complex_ofReal (μ := ν) (f := fun x : ℝ => x)
    calc
      ∫ x, (x : ℂ) ∂ν = ((∫ x, x ∂ν : ℝ) : ℂ) := h_int
      _ = (0 : ℂ) := h_cast
  have h_var_c : ∫ x, (x ^ 2 : ℂ) ∂ν = (σ2 : ℂ) := by
    have h_cast : ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) = (σ2 : ℂ) :=
      congrArg (fun r : ℝ => (r : ℂ)) h_var'
    have h_int : ∫ x, (x ^ 2 : ℂ) ∂ν = ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) := by
      simpa [pow_two] using
        (integral_complex_ofReal (μ := ν) (f := fun x : ℝ => x * x))
    calc
      ∫ x, (x ^ 2 : ℂ) ∂ν = ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) := h_int
      _ = (σ2 : ℂ) := h_cast
  rcases exp_remainder_bound_global_imag with ⟨C, hCnonneg, hCbound⟩
  intro t
  let u : ℕ → ℝ := fun n => t / Real.sqrt (n : ℝ)
  let z : ℕ → ℝ → ℂ := fun n x => ((u n * x) : ℂ) * Complex.I
  let rem : ℕ → ℝ → ℂ := fun n x =>
    Complex.exp (z n x) - ((1 : ℂ) + z n x + (z n x) ^ 2 / 2)
  have h_rem_bound_ae :
      ∀ n : ℕ, ∀ᵐ x ∂ν, ‖rem n x‖ ≤ |x| ^ 3 * (C * |u n| ^ 3) := by
    intro n
    refine ae_of_all _ ?_
    intro x
    have hbound := hCbound (u n * x)
    simpa [rem, z, abs_mul, mul_pow, mul_comm, mul_left_comm, mul_assoc] using hbound
  have h_rem_norm_bound :
      ∀ n : ℕ, ‖∫ x, rem n x ∂ν‖ ≤ C * |u n| ^ 3 * ∫ x, |x| ^ 3 ∂ν := by
    intro n
    have h_int_rem : Integrable (fun x => rem n x) ν := by
      have h_exp : Integrable (fun x : ℝ => Complex.exp (z n x)) ν := by
        refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => Complex.exp (z n x))
          (C := 1) ?_ ?_
        · exact (by fun_prop : Measurable fun x : ℝ => Complex.exp (z n x)).aestronglyMeasurable
        · filter_upwards with x
          have hnorm : ‖Complex.exp (z n x)‖ = 1 := by
            simpa [z] using (Complex.norm_exp_ofReal_mul_I (u n * x))
          exact le_of_eq hnorm
      have h_z : Integrable (fun x => z n x) ν := by
        simpa [z, mul_assoc, mul_left_comm, mul_comm] using (h_int_x'.const_mul (u n * Complex.I))
      have h_z2 : Integrable (fun x => (z n x) ^ 2 / 2) ν := by
        have h_const :
            Integrable (fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ)) ν := by
          exact h_int_x2'.const_mul ((u n * Complex.I) ^ 2 / 2)
        have h_eq :
            (fun x : ℝ => (z n x) ^ 2 / 2) =
              fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
          funext x
          ring
        simpa [h_eq] using h_const
      have h_poly : Integrable (fun x => (1 : ℂ) + z n x + (z n x) ^ 2 / 2) ν := by
        have h1 : Integrable (fun _ : ℝ => (1 : ℂ)) ν := by
          simp
        have h12 : Integrable (fun x => (1 : ℂ) + z n x) ν := h1.add h_z
        exact h12.add h_z2
      simpa [rem] using h_exp.sub h_poly
    have h_int_norm : Integrable (fun x => ‖rem n x‖) ν := h_int_rem.norm
    have h_bound_const : Integrable (fun x => |x| ^ 3 * (C * |u n| ^ 3)) ν := by
      simpa using h_int_abs3'.mul_const (C * |u n| ^ 3)
    have h_le :
        ∫ x, ‖rem n x‖ ∂ν ≤ ∫ x, |x| ^ 3 * (C * |u n| ^ 3) ∂ν := by
      exact integral_mono_ae h_int_norm h_bound_const (h_rem_bound_ae n)
    have h_const :
        ∫ x, |x| ^ 3 * (C * |u n| ^ 3) ∂ν =
          C * |u n| ^ 3 * ∫ x, |x| ^ 3 ∂ν := by
      calc
        ∫ x, |x| ^ 3 * (C * |u n| ^ 3) ∂ν =
            (∫ x, |x| ^ 3 ∂ν) * (C * |u n| ^ 3) := by
              simpa using
                (integral_mul_const (μ := ν) (r := (C * |u n| ^ 3)) (f := fun x : ℝ => |x| ^ 3))
        _ = C * |u n| ^ 3 * ∫ x, |x| ^ 3 ∂ν := by
              ring
    calc
      ‖∫ x, rem n x ∂ν‖ ≤ ∫ x, ‖rem n x‖ ∂ν :=
        norm_integral_le_integral_norm _
      _ ≤ ∫ x, |x| ^ 3 * (C * |u n| ^ 3) ∂ν := h_le
      _ = C * |u n| ^ 3 * ∫ x, |x| ^ 3 ∂ν := h_const
  have h_decomp :
      ∀ n : ℕ, (charFun ν (u n) - 1) =
        (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν)
          + ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν)
          + ∫ x, rem n x ∂ν := by
    intro n
    have h_exp : Integrable (fun x : ℝ => Complex.exp (z n x)) ν := by
      refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => Complex.exp (z n x))
        (C := 1) ?_ ?_
      · exact (by fun_prop : Measurable fun x : ℝ => Complex.exp (z n x)).aestronglyMeasurable
      · filter_upwards with x
        have hnorm : ‖Complex.exp (z n x)‖ = 1 := by
          simpa [z] using (Complex.norm_exp_ofReal_mul_I (u n * x))
        exact le_of_eq hnorm
    have h_z : Integrable (fun x => z n x) ν := by
      simpa [z, mul_assoc, mul_left_comm, mul_comm] using (h_int_x'.const_mul (u n * Complex.I))
    have h_z2 : Integrable (fun x => (z n x) ^ 2 / 2) ν := by
      have h_const :
          Integrable (fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ)) ν := by
        exact h_int_x2'.const_mul ((u n * Complex.I) ^ 2 / 2)
      have h_eq :
          (fun x : ℝ => (z n x) ^ 2 / 2) =
            fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
        funext x
        ring
      simpa [h_eq] using h_const
    have h_poly : Integrable (fun x => (1 : ℂ) + z n x + (z n x) ^ 2 / 2) ν := by
      have h1 : Integrable (fun _ : ℝ => (1 : ℂ)) ν := by
        simp
      have h12 : Integrable (fun x => (1 : ℂ) + z n x) ν := h1.add h_z
      exact h12.add h_z2
    have h_rem : Integrable (fun x => rem n x) ν := by
      simpa [rem] using h_exp.sub h_poly
    have h_exp_sub : ∀ x : ℝ, Complex.exp (z n x) - (1 : ℂ) =
        z n x + (z n x) ^ 2 / 2 + rem n x := by
      intro x
      simp [rem, add_assoc, add_left_comm, add_comm, sub_eq_add_neg]
    have h_char :
        charFun ν (u n) - 1 =
          ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := by
      have h_sub : ∫ x, Complex.exp (z n x) - (1 : ℂ) ∂ν =
          ∫ x, Complex.exp (z n x) ∂ν - ∫ x, (1 : ℂ) ∂ν := by
        exact integral_sub h_exp (integrable_const (μ := ν) (c := (1 : ℂ)))
      calc
        charFun ν (u n) - 1
            = ∫ x, Complex.exp (z n x) ∂ν - ∫ x, (1 : ℂ) ∂ν := by
                simp [charFun_apply_real, z, mul_assoc, mul_left_comm, mul_comm,
                  integral_const, hprob.measure_univ]
        _ = ∫ x, Complex.exp (z n x) - 1 ∂ν := by
                simpa using h_sub.symm
        _ = ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := by
                refine integral_congr_ae ?_
                exact ae_of_all _ (fun x => h_exp_sub x)
    have h_split :
        ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν
          = ∫ x, z n x ∂ν
            + ∫ x, (z n x) ^ 2 / 2 ∂ν
            + ∫ x, rem n x ∂ν := by
      have h12 : ∫ x, z n x + (z n x) ^ 2 / 2 ∂ν =
          ∫ x, z n x ∂ν + ∫ x, (z n x) ^ 2 / 2 ∂ν := by
        simpa using (integral_add h_z h_z2)
      have h123 : ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν =
          ∫ x, z n x + (z n x) ^ 2 / 2 ∂ν + ∫ x, rem n x ∂ν := by
        simpa [add_assoc] using (integral_add (h_z.add h_z2) h_rem)
      simpa [h12, add_assoc] using h123
    have h_int_z : ∫ x, z n x ∂ν = (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν) := by
      have h_eq : (fun x : ℝ => z n x) = fun x : ℝ => (u n * Complex.I) * (x : ℂ) := by
        funext x
        ring
      simpa [h_eq] using
        (integral_const_mul (μ := ν) (r := (u n * Complex.I)) (f := fun x : ℝ => (x : ℂ)))
    have h_int_z2 :
        ∫ x, (z n x) ^ 2 / 2 ∂ν =
          ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν) := by
      have h_eq :
          (fun x : ℝ => (z n x) ^ 2 / 2) =
            fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
        funext x
        ring
      simpa [h_eq] using
        (integral_const_mul (μ := ν) (r := (u n * Complex.I) ^ 2 / 2)
          (f := fun x : ℝ => (x ^ 2 : ℂ)))
    calc
      charFun ν (u n) - 1 =
          ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := h_char
      _ = ∫ x, z n x ∂ν
            + ∫ x, (z n x) ^ 2 / 2 ∂ν
            + ∫ x, rem n x ∂ν := h_split
      _ = (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν)
            + ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν)
            + ∫ x, rem n x ∂ν := by
            simp [h_int_z, h_int_z2]
  have h_simpl :
      ∀ n : ℕ, charFun ν (u n) - 1 =
        (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ) + ∫ x, rem n x ∂ν := by
    intro n
    have h := h_decomp n
    have h' :
        charFun ν (u n) - 1 =
          ((u n * Complex.I) ^ 2 / 2) * (σ2 : ℂ) + ∫ x, rem n x ∂ν := by
      simpa [h_mean_c, h_var_c, add_assoc, add_left_comm, add_comm] using h
    have h_coeff : ((u n * Complex.I) ^ 2 / 2 : ℂ) = (-(u n) ^ 2 / 2 : ℂ) := by
      calc
        ((u n * Complex.I) ^ 2 / 2 : ℂ)
            = ((u n * Complex.I) * (u n * Complex.I)) / 2 := by
                simp [pow_two]
        _ = ((u n * u n) * (Complex.I * Complex.I)) / 2 := by
                ring
        _ = (-(u n * u n)) / 2 := by
                simp [Complex.I_mul_I, mul_assoc, mul_left_comm, mul_comm]
        _ = (-(u n) ^ 2 / 2 : ℂ) := by
                simp [pow_two]
    simpa [h_coeff] using h'
  have h_rem_tendsto :
      Tendsto (fun n : ℕ => (n : ℂ) * ∫ x, rem n x ∂ν) atTop (𝓝 (0 : ℂ)) := by
    set K : ℝ := ∫ x, |x| ^ 3 ∂ν
    have h_bound :
        ∀ n : ℕ, ‖(n : ℂ) * ∫ x, rem n x ∂ν‖ ≤
          (C * |t| ^ 3 * K) / Real.sqrt (n : ℝ) := by
      intro n
      have hnorm : ‖(n : ℂ) * ∫ x, rem n x ∂ν‖ = (n : ℝ) * ‖∫ x, rem n x ∂ν‖ := by
        have hn : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
        simp [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hn]
      have h_abs_u : |u n| = |t| / Real.sqrt (n : ℝ) := by
        simp [u, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
      have h_u_pow : |u n| ^ 3 = |t| ^ 3 / (Real.sqrt (n : ℝ)) ^ 3 := by
        simpa [h_abs_u] using (div_pow |t| (Real.sqrt (n : ℝ)) 3)
      have h_rem' : ‖∫ x, rem n x ∂ν‖ ≤ C * |u n| ^ 3 * K := by
        simpa [K] using h_rem_norm_bound n
      have h_nu :
          (n : ℝ) * (C * |u n| ^ 3 * K) =
            (C * |t| ^ 3 * K) / Real.sqrt (n : ℝ) := by
        have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
          exact Real.sq_sqrt (by positivity)
        calc
          (n : ℝ) * (C * |u n| ^ 3 * K)
              = (n : ℝ) * (C * (|t| ^ 3 / (Real.sqrt (n : ℝ)) ^ 3) * K) := by
                  simp [h_u_pow]
          _ = C * |t| ^ 3 * K * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 3) := by
                  ring
          _ = C * |t| ^ 3 * K / Real.sqrt (n : ℝ) := by
                  have h' : (n : ℝ) / (Real.sqrt (n : ℝ)) ^ 3 = 1 / Real.sqrt (n : ℝ) := by
                    by_cases hs : Real.sqrt (n : ℝ) = 0
                    · simp [hs]
                    ·
                      have hsq' : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
                        exact Real.sq_sqrt (by positivity)
                      field_simp [hs, hsq', pow_succ, mul_assoc, mul_left_comm, mul_comm]
                      simp [hsq']
                  simp [h', div_eq_mul_inv, mul_assoc]
          _ = (C * |t| ^ 3 * K) / Real.sqrt (n : ℝ) := by
                  ring
      calc
        ‖(n : ℂ) * ∫ x, rem n x ∂ν‖
            = (n : ℝ) * ‖∫ x, rem n x ∂ν‖ := hnorm
        _ ≤ (n : ℝ) * (C * |u n| ^ 3 * K) := by
            exact mul_le_mul_of_nonneg_left h_rem' (Nat.cast_nonneg n)
        _ = (C * |t| ^ 3 * K) / Real.sqrt (n : ℝ) := h_nu
    have h_tendsto :
        Tendsto (fun n : ℕ => (C * |t| ^ 3 * K) / Real.sqrt (n : ℝ))
          atTop (𝓝 (0 : ℝ)) := by
      have h_sqrt := tendsto_inv_sqrt_nat
      simpa [div_eq_mul_inv] using (tendsto_const_nhds.mul h_sqrt)
    have h_norm_tendsto :
        Tendsto (fun n : ℕ => ‖(n : ℂ) * ∫ x, rem n x ∂ν‖) atTop (𝓝 (0 : ℝ)) := by
      refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h_tendsto ?_
        (Filter.Eventually.of_forall h_bound)
      exact Filter.Eventually.of_forall (fun n => norm_nonneg _)
    exact (tendsto_iff_norm_sub_tendsto_zero).2 <| by
      simpa using h_norm_tendsto
  have h_main :
      Tendsto
        (fun n : ℕ => (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    have h_nu2 : ∀ᶠ n : ℕ in atTop, (n : ℂ) * (u n) ^ 2 = (t : ℂ) ^ 2 := by
      refine (eventually_ge_atTop 1).mono ?_
      intro n hn
      have hn0 : (n : ℝ) ≠ 0 := by
        exact_mod_cast (Nat.succ_le_iff.mp hn).ne'
      have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
        exact Real.sq_sqrt (by positivity)
      have h_real : (n : ℝ) * (u n) ^ 2 = t ^ 2 := by
        have h_u2 : (u n) ^ 2 = t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2 := by
          simpa [u] using (div_pow t (Real.sqrt (n : ℝ)) 2)
        calc
          (n : ℝ) * (u n) ^ 2 = (n : ℝ) * (t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
              simp [h_u2]
          _ = t ^ 2 := by
              calc
                (n : ℝ) * (t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2)
                    = t ^ 2 * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 2) := by
                        ring
                _ = t ^ 2 * ((Real.sqrt (n : ℝ)) ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
                        simp [hsq]
                _ = t ^ 2 := by
                        simp [div_self, hn0]
      exact_mod_cast h_real
    refine (tendsto_congr' ?_).mpr tendsto_const_nhds
    refine h_nu2.mono ?_
    intro n hn
    calc
      (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
          = (-(σ2 : ℂ) / 2) * ((n : ℂ) * (u n) ^ 2) := by
              ring
      _ = (-(σ2 : ℂ) / 2) * (t : ℂ) ^ 2 := by
              simp [hn]
      _ = (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
              ring
  have h_decomp' :
      ∀ n : ℕ, (n : ℂ) * (charFun ν (u n) - 1) =
        (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
          + (n : ℂ) * ∫ x, rem n x ∂ν := by
    intro n
    simp [h_simpl n, mul_add, mul_assoc, mul_left_comm, mul_comm]
  have h_sum := h_main.add h_rem_tendsto
  have h_sum' :
      Tendsto
        (fun n : ℕ =>
          (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
            + (n : ℂ) * ∫ x, rem n x ∂ν) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    simpa using h_sum
  refine (tendsto_congr ?_).1 h_sum'
  intro n
  simpa using (h_decomp' n).symm

theorem CharFunCLTScale_of_integrable_sq {X : Ω → ℝ} {σ2 : ℝ}
    (h_meas : Measurable X)
    (h_int_x2 : Integrable (fun x : ℝ => x ^ 2) (μ.map X))
    (h_mean : ∫ x, x ∂ μ.map X = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map X = σ2) :
    CharFunCLTScale μ X σ2 := by
  classical
  set ν : Measure ℝ := μ.map X
  have hprob : IsProbabilityMeasure ν := Measure.isProbabilityMeasure_map (by fun_prop)
  have h_mean' : ∫ x, x ∂ν = 0 := by
    simpa [ν] using h_mean
  have h_var' : ∫ x, x ^ 2 ∂ν = σ2 := by
    simpa [ν] using h_var
  have h_int_x2' : Integrable (fun x : ℝ => x ^ 2) ν := by
    simpa [ν] using h_int_x2
  have h_abs_sq_eq : (fun x : ℝ => |x| ^ 2) = fun x : ℝ => x ^ 2 := by
    funext x
    calc
      |x| ^ 2 = |x ^ 2| := by
        symm
        simp [pow_two]
      _ = x ^ 2 := by
        exact abs_of_nonneg (sq_nonneg x)
  have h_int_abs_sq : Integrable (fun x : ℝ => |x| ^ 2) ν := by
    simpa [← h_abs_sq_eq] using h_int_x2'
  have h_int_bound : Integrable (fun x : ℝ => 1 + |x| ^ 2) ν := by
    have h1 : Integrable (fun _ : ℝ => (1 : ℝ)) ν := by
      simp
    exact h1.add h_int_abs_sq
  have h_int_x' : Integrable (fun x : ℝ => (x : ℂ)) ν := by
    refine Integrable.mono' h_int_bound
      (by fun_prop : AEStronglyMeasurable (fun x : ℝ => (x : ℂ)) ν) ?_
    refine Filter.Eventually.of_forall ?_
    intro x
    have h_le : |x| ≤ 1 + |x| ^ 2 := abs_le_one_add_abs_sq x
    simpa [Complex.norm_real, Real.norm_eq_abs] using h_le
  have h_int_x2c' : Integrable (fun x : ℝ => (x ^ 2 : ℂ)) ν := by
    refine Integrable.mono' h_int_bound
      (by fun_prop : AEStronglyMeasurable (fun x : ℝ => (x ^ 2 : ℂ)) ν) ?_
    refine Filter.Eventually.of_forall ?_
    intro x
    have h_le : |x| ^ 2 ≤ 1 + |x| ^ 2 := by
      exact le_add_of_nonneg_left (by positivity)
    have h_norm : ‖(x ^ 2 : ℂ)‖ = |x| ^ 2 := by
      calc
        ‖(x ^ 2 : ℂ)‖ = |x ^ 2| := by
          simp [Complex.norm_real, Real.norm_eq_abs]
        _ = |x| ^ 2 := by
          simp [pow_two]
    calc
      ‖(x ^ 2 : ℂ)‖ = |x| ^ 2 := h_norm
      _ ≤ 1 + |x| ^ 2 := h_le
  have h_mean_c : ∫ x, (x : ℂ) ∂ν = (0 : ℂ) := by
    have h_cast : ((∫ x, x ∂ν : ℝ) : ℂ) = (0 : ℂ) :=
      congrArg (fun r : ℝ => (r : ℂ)) h_mean'
    have h_int : ∫ x, (x : ℂ) ∂ν = ((∫ x, x ∂ν : ℝ) : ℂ) :=
      integral_complex_ofReal (μ := ν) (f := fun x : ℝ => x)
    calc
      ∫ x, (x : ℂ) ∂ν = ((∫ x, x ∂ν : ℝ) : ℂ) := h_int
      _ = (0 : ℂ) := h_cast
  have h_var_c : ∫ x, (x ^ 2 : ℂ) ∂ν = (σ2 : ℂ) := by
    have h_cast : ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) = (σ2 : ℂ) :=
      congrArg (fun r : ℝ => (r : ℂ)) h_var'
    have h_int : ∫ x, (x ^ 2 : ℂ) ∂ν = ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) := by
      simpa [pow_two] using
        (integral_complex_ofReal (μ := ν) (f := fun x : ℝ => x * x))
    calc
      ∫ x, (x ^ 2 : ℂ) ∂ν = ((∫ x, x ^ 2 ∂ν : ℝ) : ℂ) := h_int
      _ = (σ2 : ℂ) := h_cast
  rcases exp_remainder_bound with ⟨C0, δ, hδ, hC0nonneg, hC0bound⟩
  rcases exp_remainder_bound_global_imag_sq with ⟨C, hCnonneg, hCbound⟩
  intro t
  let u : ℕ → ℝ := fun n => t / Real.sqrt (n : ℝ)
  let z : ℕ → ℝ → ℂ := fun n x => ((u n * x) : ℂ) * Complex.I
  let rem : ℕ → ℝ → ℂ := fun n x =>
    Complex.exp (z n x) - ((1 : ℂ) + z n x + (z n x) ^ 2 / 2)
  have h_u_tendsto : Tendsto (fun n : ℕ => u n) atTop (𝓝 0) := by
    have h_sqrt : Tendsto (fun n : ℕ => (Real.sqrt (n : ℝ))⁻¹) atTop (𝓝 (0 : ℝ)) :=
      tendsto_inv_sqrt_nat
    simpa [u, div_eq_mul_inv] using (tendsto_const_nhds.mul h_sqrt)
  have h_abs_u : Tendsto (fun n : ℕ => |u n|) atTop (𝓝 (0 : ℝ)) := by
    simpa using (continuous_abs.tendsto 0).comp h_u_tendsto
  have h_nu3_tendsto :
      Tendsto (fun n : ℕ => (n : ℝ) * |u n| ^ 3) atTop (𝓝 (0 : ℝ)) := by
    have h_abs_u' : ∀ n : ℕ, |u n| = |t| / Real.sqrt (n : ℝ) := by
      intro n
      simp [u, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
    have h_u_pow : ∀ n : ℕ, |u n| ^ 3 = |t| ^ 3 / (Real.sqrt (n : ℝ)) ^ 3 := by
      intro n
      simpa [h_abs_u' n] using (div_pow |t| (Real.sqrt (n : ℝ)) 3)
    have h_nu :
        ∀ n : ℕ, (n : ℝ) * |u n| ^ 3 = |t| ^ 3 / Real.sqrt (n : ℝ) := by
      intro n
      have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
        exact Real.sq_sqrt (by positivity)
      calc
        (n : ℝ) * |u n| ^ 3 =
            (n : ℝ) * (|t| ^ 3 / (Real.sqrt (n : ℝ)) ^ 3) := by
              simp [h_u_pow n]
        _ = |t| ^ 3 * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 3) := by
              ring
        _ = |t| ^ 3 / Real.sqrt (n : ℝ) := by
              have h' : (n : ℝ) / (Real.sqrt (n : ℝ)) ^ 3 = 1 / Real.sqrt (n : ℝ) := by
                by_cases hs : Real.sqrt (n : ℝ) = 0
                · simp [hs]
                ·
                  have hsq' : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
                    exact Real.sq_sqrt (by positivity)
                  field_simp [hs, hsq', pow_succ, mul_assoc, mul_left_comm, mul_comm]
                  simp [hsq']
              simp [h', div_eq_mul_inv, mul_assoc]
    have h_eq : (fun n : ℕ => (n : ℝ) * |u n| ^ 3) =
        fun n : ℕ => |t| ^ 3 / Real.sqrt (n : ℝ) := by
      funext n
      exact h_nu n
    have h_sqrt : Tendsto (fun n : ℕ => (Real.sqrt (n : ℝ))⁻¹) atTop (𝓝 (0 : ℝ)) :=
      tendsto_inv_sqrt_nat
    simpa [h_eq, div_eq_mul_inv] using (tendsto_const_nhds.mul h_sqrt)
  have h_decomp :
      ∀ n : ℕ, (charFun ν (u n) - 1) =
        (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν)
          + ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν)
          + ∫ x, rem n x ∂ν := by
    intro n
    have h_exp : Integrable (fun x : ℝ => Complex.exp (z n x)) ν := by
      refine integrable_of_norm_bounded (μ := ν) (f := fun x : ℝ => Complex.exp (z n x))
        (C := 1) ?_ ?_
      · exact (by fun_prop : Measurable fun x : ℝ => Complex.exp (z n x)).aestronglyMeasurable
      · filter_upwards with x
        have hnorm : ‖Complex.exp (z n x)‖ = 1 := by
          simpa [z] using (Complex.norm_exp_ofReal_mul_I (u n * x))
        exact le_of_eq hnorm
    have h_z : Integrable (fun x => z n x) ν := by
      simpa [z, mul_assoc, mul_left_comm, mul_comm] using (h_int_x'.const_mul (u n * Complex.I))
    have h_z2 : Integrable (fun x => (z n x) ^ 2 / 2) ν := by
      have h_const :
          Integrable (fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ)) ν := by
        exact h_int_x2c'.const_mul ((u n * Complex.I) ^ 2 / 2)
      have h_eq :
          (fun x : ℝ => (z n x) ^ 2 / 2) =
            fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
        funext x
        ring
      simpa [h_eq] using h_const
    have h_poly : Integrable (fun x => (1 : ℂ) + z n x + (z n x) ^ 2 / 2) ν := by
      have h1 : Integrable (fun _ : ℝ => (1 : ℂ)) ν := by
        simp
      have h12 : Integrable (fun x => (1 : ℂ) + z n x) ν := h1.add h_z
      exact h12.add h_z2
    have h_rem : Integrable (fun x => rem n x) ν := by
      simpa [rem] using h_exp.sub h_poly
    have h_exp_sub : ∀ x : ℝ, Complex.exp (z n x) - (1 : ℂ) =
        z n x + (z n x) ^ 2 / 2 + rem n x := by
      intro x
      simp [rem, add_assoc, add_left_comm, add_comm, sub_eq_add_neg]
    have h_char :
        charFun ν (u n) - 1 =
          ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := by
      have h_sub : ∫ x, Complex.exp (z n x) - (1 : ℂ) ∂ν =
          ∫ x, Complex.exp (z n x) ∂ν - ∫ x, (1 : ℂ) ∂ν := by
        exact integral_sub h_exp (integrable_const (μ := ν) (c := (1 : ℂ)))
      calc
        charFun ν (u n) - 1
            = ∫ x, Complex.exp (z n x) ∂ν - ∫ x, (1 : ℂ) ∂ν := by
                simp [charFun_apply_real, z, mul_assoc, mul_left_comm, mul_comm,
                  integral_const, hprob.measure_univ]
        _ = ∫ x, Complex.exp (z n x) - 1 ∂ν := by
                simpa using h_sub.symm
        _ = ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := by
                refine integral_congr_ae ?_
                exact ae_of_all _ (fun x => h_exp_sub x)
    have h_split :
        ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν
          = ∫ x, z n x ∂ν
            + ∫ x, (z n x) ^ 2 / 2 ∂ν
            + ∫ x, rem n x ∂ν := by
      have h12 : ∫ x, z n x + (z n x) ^ 2 / 2 ∂ν =
          ∫ x, z n x ∂ν + ∫ x, (z n x) ^ 2 / 2 ∂ν := by
        simpa using (integral_add h_z h_z2)
      have h123 : ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν =
          ∫ x, z n x + (z n x) ^ 2 / 2 ∂ν + ∫ x, rem n x ∂ν := by
        simpa [add_assoc] using (integral_add (h_z.add h_z2) h_rem)
      simpa [h12, add_assoc] using h123
    have h_int_z : ∫ x, z n x ∂ν = (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν) := by
      have h_eq : (fun x : ℝ => z n x) = fun x : ℝ => (u n * Complex.I) * (x : ℂ) := by
        funext x
        ring
      simpa [h_eq] using
        (integral_const_mul (μ := ν) (r := (u n * Complex.I)) (f := fun x : ℝ => (x : ℂ)))
    have h_int_z2 :
        ∫ x, (z n x) ^ 2 / 2 ∂ν =
          ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν) := by
      have h_eq :
          (fun x : ℝ => (z n x) ^ 2 / 2) =
            fun x : ℝ => ((u n * Complex.I) ^ 2 / 2) * (x ^ 2 : ℂ) := by
        funext x
        ring
      simpa [h_eq] using
        (integral_const_mul (μ := ν) (r := (u n * Complex.I) ^ 2 / 2)
          (f := fun x : ℝ => (x ^ 2 : ℂ)))
    calc
      charFun ν (u n) - 1 =
          ∫ x, z n x + (z n x) ^ 2 / 2 + rem n x ∂ν := h_char
      _ = ∫ x, z n x ∂ν
            + ∫ x, (z n x) ^ 2 / 2 ∂ν
            + ∫ x, rem n x ∂ν := h_split
      _ = (u n * Complex.I) * (∫ x, (x : ℂ) ∂ν)
            + ((u n * Complex.I) ^ 2 / 2) * (∫ x, (x ^ 2 : ℂ) ∂ν)
            + ∫ x, rem n x ∂ν := by
            simp [h_int_z, h_int_z2]
  have h_simpl :
      ∀ n : ℕ, charFun ν (u n) - 1 =
        (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ) + ∫ x, rem n x ∂ν := by
    intro n
    have h := h_decomp n
    have h' :
        charFun ν (u n) - 1 =
          ((u n * Complex.I) ^ 2 / 2) * (σ2 : ℂ) + ∫ x, rem n x ∂ν := by
      simpa [h_mean_c, h_var_c, add_assoc, add_left_comm, add_comm] using h
    have h_coeff : ((u n * Complex.I) ^ 2 / 2 : ℂ) = (-(u n) ^ 2 / 2 : ℂ) := by
      calc
        ((u n * Complex.I) ^ 2 / 2 : ℂ)
            = ((u n * Complex.I) * (u n * Complex.I)) / 2 := by
                simp [pow_two]
        _ = ((u n * u n) * (Complex.I * Complex.I)) / 2 := by
                ring
        _ = (-(u n * u n)) / 2 := by
                simp [Complex.I_mul_I, mul_assoc, mul_left_comm, mul_comm]
        _ = (-(u n) ^ 2 / 2 : ℂ) := by
                simp [pow_two]
    simpa [h_coeff] using h'
  have h_rem_tendsto :
      Tendsto (fun n : ℕ => (n : ℂ) * ∫ x, rem n x ∂ν) atTop (𝓝 (0 : ℂ)) := by
    let F : ℕ → ℝ → ℂ := fun n x => (n : ℂ) * rem n x
    let bound : ℝ → ℝ := fun x => C * |t| ^ 2 * |x| ^ 2
    have h_meas : ∀ n, AEStronglyMeasurable (F n) ν := by
      intro n
      exact (by fun_prop : Measurable (F n)).aestronglyMeasurable
    have h_bound :
        ∀ n, ∀ᵐ x ∂ν, ‖F n x‖ ≤ bound x := by
      intro n
      refine ae_of_all _ ?_
      intro x
      have hrem : ‖rem n x‖ ≤ C * |u n * x| ^ 2 := by
        simpa [rem, z] using hCbound (u n * x)
      have hrem' : ‖rem n x‖ ≤ C * |u n| ^ 2 * |x| ^ 2 := by
        simpa [abs_mul, mul_pow, mul_comm, mul_left_comm, mul_assoc] using hrem
      have hnorm : ‖F n x‖ = (n : ℝ) * ‖rem n x‖ := by
        have hn : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
        simp [F, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hn]
      have h_nu2_bound : (n : ℝ) * |u n| ^ 2 ≤ |t| ^ 2 := by
        by_cases hn : n = 0
        · subst hn
          have hnonneg : 0 ≤ |t| ^ 2 := by positivity
          simpa [u] using hnonneg
        · have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast hn
          have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
            exact Real.sq_sqrt (by positivity)
          have h_abs_u : |u n| = |t| / Real.sqrt (n : ℝ) := by
            simp [u, abs_div, abs_of_nonneg (Real.sqrt_nonneg _)]
          have h_u_pow : |u n| ^ 2 = |t| ^ 2 / (Real.sqrt (n : ℝ)) ^ 2 := by
            simpa [h_abs_u] using (div_pow |t| (Real.sqrt (n : ℝ)) 2)
          have h_eq : (n : ℝ) * |u n| ^ 2 = |t| ^ 2 := by
            calc
              (n : ℝ) * |u n| ^ 2
                  = (n : ℝ) * (|t| ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
                      simp [h_u_pow]
              _ = |t| ^ 2 := by
                    calc
                      (n : ℝ) * (|t| ^ 2 / (Real.sqrt (n : ℝ)) ^ 2)
                          = |t| ^ 2 * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 2) := by
                              ring
                      _ = |t| ^ 2 * ((Real.sqrt (n : ℝ)) ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
                              simp [hsq]
                      _ = |t| ^ 2 := by
                              simp [div_self, hn0]
          exact h_eq.le
      calc
        ‖F n x‖ = (n : ℝ) * ‖rem n x‖ := hnorm
        _ ≤ (n : ℝ) * (C * |u n| ^ 2 * |x| ^ 2) := by
            exact mul_le_mul_of_nonneg_left hrem' (Nat.cast_nonneg n)
        _ = (C * |x| ^ 2) * ((n : ℝ) * |u n| ^ 2) := by
            ring
        _ ≤ (C * |x| ^ 2) * |t| ^ 2 := by
            exact mul_le_mul_of_nonneg_left h_nu2_bound (by positivity)
        _ = bound x := by
            ring
    have h_abs_t_sq : |t| ^ 2 = t ^ 2 := by
      calc
        |t| ^ 2 = |t ^ 2| := by
          symm
          simp [pow_two]
        _ = t ^ 2 := by
          exact abs_of_nonneg (sq_nonneg t)
    have h_bound_eq : bound = fun x : ℝ => (C * t ^ 2) * x ^ 2 := by
      funext x
      calc
        bound x = C * |t| ^ 2 * |x| ^ 2 := by rfl
        _ = C * t ^ 2 * x ^ 2 := by
              simp [h_abs_sq_eq, h_abs_t_sq, mul_comm, mul_left_comm, mul_assoc]
    have h_int_bound : Integrable bound ν := by
      simpa [h_bound_eq] using (h_int_x2'.const_mul (C * t ^ 2))
    have h_lim :
        ∀ᵐ x ∂ν, Tendsto (fun n : ℕ => F n x) atTop (𝓝 (0 : ℂ)) := by
      refine ae_of_all _ ?_
      intro x
      have h_abs_u_mul : Tendsto (fun n : ℕ => |u n| * |x|) atTop (𝓝 (0 : ℝ)) := by
        simpa [mul_comm] using h_abs_u.const_mul |x|
      have h_u_small : ∀ᶠ n : ℕ in atTop, |u n| * |x| < δ := by
        exact (tendsto_order.1 h_abs_u_mul).2 δ hδ
      have h_bound_eventually :
          ∀ᶠ n : ℕ in atTop, ‖F n x‖ ≤ (C0 * |x| ^ 3) * ((n : ℝ) * |u n| ^ 3) := by
        refine h_u_small.mono ?_
        intro n hn
        have hz' : ‖(z n x : ℂ)‖ < δ := by
          have hmul : |u n * x| < δ := by
            simpa [abs_mul] using hn
          simpa [z, norm_mul, Complex.norm_I, Complex.norm_real, Real.norm_eq_abs, abs_mul] using hmul
        have hrem : ‖rem n x‖ ≤ C0 * ‖(z n x : ℂ)‖ ^ 3 := by
          simpa [rem] using hC0bound (z n x) hz'
        have hrem' : ‖rem n x‖ ≤ C0 * |u n| ^ 3 * |x| ^ 3 := by
          have hz_norm : ‖(z n x : ℂ)‖ = |u n * x| := by
            simp [z, norm_mul, Complex.norm_I, Complex.norm_real, Real.norm_eq_abs, abs_mul]
          calc
            ‖rem n x‖ ≤ C0 * ‖(z n x : ℂ)‖ ^ 3 := hrem
            _ = C0 * |u n * x| ^ 3 := by
                  simp [hz_norm]
            _ = C0 * (|u n| ^ 3 * |x| ^ 3) := by
                  simp [abs_mul, mul_pow, mul_comm, mul_left_comm, mul_assoc]
            _ = C0 * |u n| ^ 3 * |x| ^ 3 := by
                  ring
        have hnorm : ‖F n x‖ = (n : ℝ) * ‖rem n x‖ := by
          have hn' : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
          simp [F, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hn']
        calc
          ‖F n x‖ = (n : ℝ) * ‖rem n x‖ := hnorm
          _ ≤ (n : ℝ) * (C0 * |u n| ^ 3 * |x| ^ 3) := by
              exact mul_le_mul_of_nonneg_left hrem' (Nat.cast_nonneg n)
          _ = (C0 * |x| ^ 3) * ((n : ℝ) * |u n| ^ 3) := by
              ring
      have h_bound_tendsto :
          Tendsto (fun n : ℕ => (C0 * |x| ^ 3) * ((n : ℝ) * |u n| ^ 3)) atTop (𝓝 (0 : ℝ)) := by
        simpa using (tendsto_const_nhds.mul h_nu3_tendsto)
      have h_norm_tendsto :
          Tendsto (fun n : ℕ => ‖F n x‖) atTop (𝓝 (0 : ℝ)) := by
        refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds
          h_bound_tendsto ?_ h_bound_eventually
        exact Filter.Eventually.of_forall (fun n => norm_nonneg _)
      exact (tendsto_iff_norm_sub_tendsto_zero).2 <| by
        simpa using h_norm_tendsto
    have h_tendsto :
        Tendsto (fun n : ℕ => ∫ x, F n x ∂ν) atTop (𝓝 (∫ x, (0 : ℂ) ∂ν)) := by
      refine
        MeasureTheory.tendsto_integral_of_dominated_convergence (μ := ν) (bound := bound)
          h_meas h_int_bound h_bound h_lim
    have h_tendsto' :
        Tendsto (fun n : ℕ => ∫ x, F n x ∂ν) atTop (𝓝 (0 : ℂ)) := by
      simpa using h_tendsto
    refine (tendsto_congr ?_).1 h_tendsto'
    intro n
    simpa [F] using
      (integral_const_mul (μ := ν) (r := (n : ℂ)) (f := fun x => rem n x))
  have h_main :
      Tendsto
        (fun n : ℕ => (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    have h_nu2 : ∀ᶠ n : ℕ in atTop, (n : ℂ) * (u n) ^ 2 = (t : ℂ) ^ 2 := by
      refine (eventually_ge_atTop 1).mono ?_
      intro n hn
      have hn0 : (n : ℝ) ≠ 0 := by
        exact_mod_cast (Nat.succ_le_iff.mp hn).ne'
      have hsq : (Real.sqrt (n : ℝ)) ^ 2 = (n : ℝ) := by
        exact Real.sq_sqrt (by positivity)
      have h_real : (n : ℝ) * (u n) ^ 2 = t ^ 2 := by
        have h_u2 : (u n) ^ 2 = t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2 := by
          simpa [u] using (div_pow t (Real.sqrt (n : ℝ)) 2)
        calc
          (n : ℝ) * (u n) ^ 2 = (n : ℝ) * (t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
              simp [h_u2]
          _ = t ^ 2 := by
              calc
                (n : ℝ) * (t ^ 2 / (Real.sqrt (n : ℝ)) ^ 2)
                    = t ^ 2 * ((n : ℝ) / (Real.sqrt (n : ℝ)) ^ 2) := by
                        ring
                _ = t ^ 2 * ((Real.sqrt (n : ℝ)) ^ 2 / (Real.sqrt (n : ℝ)) ^ 2) := by
                        simp [hsq]
                _ = t ^ 2 := by
                        simp [div_self, hn0]
      exact_mod_cast h_real
    refine (tendsto_congr' ?_).mpr tendsto_const_nhds
    refine h_nu2.mono ?_
    intro n hn
    calc
      (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
          = (-(σ2 : ℂ) / 2) * ((n : ℂ) * (u n) ^ 2) := by
              ring
      _ = (-(σ2 : ℂ) / 2) * (t : ℂ) ^ 2 := by
              simp [hn]
      _ = (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
              ring
  have h_decomp' :
      ∀ n : ℕ, (n : ℂ) * (charFun ν (u n) - 1) =
        (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
          + (n : ℂ) * ∫ x, rem n x ∂ν := by
    intro n
    simp [h_simpl n, mul_add, mul_assoc, mul_left_comm, mul_comm]
  have h_sum := h_main.add h_rem_tendsto
  have h_sum' :
      Tendsto
        (fun n : ℕ =>
          (n : ℂ) * (-(u n) ^ 2 / 2 : ℂ) * (σ2 : ℂ)
            + (n : ℂ) * ∫ x, rem n x ∂ν) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    simpa using h_sum
  refine (tendsto_congr ?_).1 h_sum'
  intro n
  simpa using (h_decomp' n).symm

theorem tendsto_charFun_normalized_sum_iid {X : ℕ → Ω → ℝ} {σ2 : ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_scale : CharFunCLTScale μ (X 0) σ2) (t : ℝ) :
    Tendsto (fun n => charFun (μ.map (normalizedSum X n)) t) atTop
      (𝓝 (Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2))) := by
  let g : ℕ → ℂ :=
    fun n => charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ)) - 1
  have hg :
      Tendsto (fun n : ℕ => (n : ℂ) * g n) atTop
        (𝓝 (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
    simpa [g] using h_scale t
  have hpow :
      Tendsto (fun n : ℕ => (1 + g n) ^ n) atTop
        (𝓝 (Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2))) := by
    simpa using
      (Complex.tendsto_one_add_pow_exp_of_tendsto (g := g)
        (t := (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) hg)
  refine (tendsto_congr ?_).1 hpow
  intro n
  have h_char :
      charFun (μ.map (normalizedSum X n)) t =
        (charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ))) ^ n := by
    simpa using
      (charFun_normalized_sum_iid h_indep h_meas h_ident (n := n) (t := t))
  -- Rewrite `1 + g n` to the characteristic function term.
  have h_one :
      (1 + g n) = charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ)) := by
    simp [g, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  calc
    (1 + g n) ^ n = (charFun (μ.map (X 0)) (t / Real.sqrt (n : ℝ))) ^ n := by
      simp [h_one]
    _ = charFun (μ.map (normalizedSum X n)) t := by
      simpa using h_char.symm

theorem tendsto_probabilityMeasure_normalized_sum_iid {X : ℕ → Ω → ℝ} {σ2 : ℝ}
    (hσ2 : 0 ≤ σ2)
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_scale : CharFunCLTScale μ (X 0) σ2) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩))⟩ :
            ProbabilityMeasure ℝ))) := by
  classical
  letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance
  let μs : ℕ → ProbabilityMeasure ℝ :=
    fun n =>
      ⟨μ.map (normalizedSum X n),
        Measure.isProbabilityMeasure_map (μ := μ)
          ((measurable_normalizedSum h_meas n).aemeasurable)⟩
  let μlim : ProbabilityMeasure ℝ :=
    ⟨gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩,
      by
        simpa using
          (inferInstance :
            IsProbabilityMeasure (gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩))⟩
  have h_char_lim :
      ∀ t : ℝ,
        charFun (μlim : Measure ℝ) t =
          Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
    intro t
    have h :=
      charFun_gaussianReal (μ := (0 : ℝ)) (v := (⟨σ2, hσ2⟩ : NNReal)) t
    have h' :
        charFun (μlim : Measure ℝ) t =
          Complex.exp (-( (σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) := by
      simpa [μlim, mul_comm, mul_left_comm, mul_assoc] using h
    have h_exp :
        (-( (σ2 : ℂ) * (t : ℂ) ^ 2 / 2)) =
          (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2) := by
      ring
    simpa [h_exp] using h'
  have h_char_tendsto :
      ∀ t : ℝ,
        Tendsto (fun n => charFun (μs n : Measure ℝ) t) atTop
          (𝓝 (charFun (μlim : Measure ℝ) t)) := by
    intro t
    have h_tendsto :
        Tendsto (fun n => charFun (μ.map (normalizedSum X n)) t) atTop
          (𝓝 (Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2))) := by
      simpa using
        (tendsto_charFun_normalized_sum_iid (X := X) (σ2 := σ2)
          h_indep h_meas h_ident h_scale t)
    have h_tendsto' :
        Tendsto (fun n => charFun (μs n : Measure ℝ) t) atTop
          (𝓝 (Complex.exp (-(σ2 : ℂ) * (t : ℂ) ^ 2 / 2))) := by
      simpa [μs] using h_tendsto
    simpa [h_char_lim t] using h_tendsto'
  simpa [μs, μlim] using
    (tendsto_probabilityMeasure_of_tendsto_charFun (μs := μs) (μ := μlim) h_char_tendsto)

theorem tendsto_probabilityMeasure_normalized_sum_iid_of_bounded {X : ℕ → Ω → ℝ} {σ2 M : ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_bound : ∀ᵐ ω ∂μ, |X 0 ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map (X 0) = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = σ2) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨gaussianReal (0 : ℝ) ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ)
                    ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩))⟩ :
            ProbabilityMeasure ℝ))) := by
  classical
  letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance
  have hσ2 : 0 ≤ σ2 :=
    variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var
  have h_scale : CharFunCLTScale μ (X 0) σ2 :=
    CharFunCLTScale_of_bounded (μ := μ) (X := X 0) (σ2 := σ2) (M := M)
      (h_meas := h_meas 0) (h_bound := h_bound) (h_mean := h_mean) (h_var := h_var)
  simpa [hσ2] using
    (tendsto_probabilityMeasure_normalized_sum_iid (μ := μ) (X := X) (σ2 := σ2) hσ2
      h_indep h_meas h_ident h_scale)

theorem tendsto_cdf_normalized_sum_iid_of_bounded {X : ℕ → Ω → ℝ} {σ2 M : ℝ} {x : ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_bound : ∀ᵐ ω ∂μ, |X 0 ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map (X 0) = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = σ2)
    (hx :
      ContinuousAt (cdf (gaussianReal (0 : ℝ)
        ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩)) x) :
    Tendsto (fun n => cdf (μ.map (normalizedSum X n)) x) atTop
      (𝓝
        (cdf (gaussianReal (0 : ℝ)
          ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩) x)) := by
  have hσ2 : 0 ≤ σ2 :=
    variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var
  let μs : ℕ → ProbabilityMeasure ℝ :=
    fun n =>
      ⟨μ.map (normalizedSum X n),
        Measure.isProbabilityMeasure_map (μ := μ)
          ((measurable_normalizedSum h_meas n).aemeasurable)⟩
  let μlim : ProbabilityMeasure ℝ :=
    ⟨gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩,
      by
        simpa using
          (inferInstance :
            IsProbabilityMeasure (gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩))⟩
  have h_tendsto : Tendsto μs atTop (𝓝 μlim) := by
    simpa [μs, μlim] using
      (tendsto_probabilityMeasure_normalized_sum_iid_of_bounded (μ := μ) (X := X) (σ2 := σ2)
        (M := M) h_indep h_meas h_ident h_bound h_mean h_var)
  have h_tendsto_cdf :=
    tendsto_cdf_of_tendsto_probabilityMeasure (μs := μs) (μ := μlim) h_tendsto
      (by simpa [μlim, hσ2] using hx)
  simpa [μs, μlim] using h_tendsto_cdf

theorem tendstoInDistribution_normalized_sum_iid {X : ℕ → Ω → ℝ} {σ2 : ℝ} {Z : Ω → ℝ}
    (hσ2 : 0 ≤ σ2)
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_scale : CharFunCLTScale μ (X 0) σ2)
    (hZ : AEMeasurable Z μ)
    (hZlaw : μ.map Z = gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩) :
    TendstoInDistribution (fun n => normalizedSum X n) atTop Z μ := by
  refine ⟨?_, hZ, ?_⟩
  · intro n
    exact (measurable_normalizedSum h_meas n).aemeasurable
  · have h_tendsto :=
      tendsto_probabilityMeasure_normalized_sum_iid (μ := μ) (X := X) (σ2 := σ2) hσ2
        h_indep h_meas h_ident h_scale
    have h_eq :
        (⟨μ.map Z, Measure.isProbabilityMeasure_map hZ⟩ : ProbabilityMeasure ℝ) =
          (⟨gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩))⟩ :
            ProbabilityMeasure ℝ) := by
      apply ProbabilityMeasure.toMeasure_injective
      simp [hZlaw]
    simpa [h_eq] using h_tendsto

theorem tendstoInDistribution_normalized_sum_iid_of_bounded {X : ℕ → Ω → ℝ} {σ2 M : ℝ}
    {Z : Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_bound : ∀ᵐ ω ∂μ, |X 0 ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map (X 0) = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = σ2)
    (hZ : AEMeasurable Z μ)
    (hZlaw :
      μ.map Z =
        gaussianReal (0 : ℝ)
          ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩) :
    TendstoInDistribution (fun n => normalizedSum X n) atTop Z μ := by
  have hσ2 : 0 ≤ σ2 :=
    variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var
  have h_scale : CharFunCLTScale μ (X 0) σ2 :=
    CharFunCLTScale_of_bounded (μ := μ) (X := X 0) (σ2 := σ2) (M := M)
      (h_meas := h_meas 0) (h_bound := h_bound) (h_mean := h_mean) (h_var := h_var)
  have hZlaw' : μ.map Z = gaussianReal (0 : ℝ) ⟨σ2, hσ2⟩ := by
    simpa [hσ2] using hZlaw
  exact tendstoInDistribution_normalized_sum_iid (μ := μ) (X := X) (σ2 := σ2) hσ2
    h_indep h_meas h_ident h_scale hZ hZlaw'

/-- Classical i.i.d. CLT assumptions (mean 0, variance σ²). -/
structure CLTAssumptions (μ : Measure Ω) (X : ℕ → Ω → ℝ) (σ2 : ℝ) : Prop where
  h_indep : iIndepFun X μ
  h_meas : ∀ i, Measurable (X i)
  h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ
  h_mean : ∫ x, x ∂ μ.map (X 0) = 0
  h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = σ2

/-- Central limit theorem for i.i.d. bounded variables: convergence of laws. -/
theorem central_limit_theorem_iid_bounded {X : ℕ → Ω → ℝ} {σ2 M : ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_bound : ∀ᵐ ω ∂μ, |X 0 ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map (X 0) = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = σ2) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨gaussianReal (0 : ℝ)
              ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ)
                    ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩))⟩ :
            ProbabilityMeasure ℝ))) := by
  exact
    tendsto_probabilityMeasure_normalized_sum_iid_of_bounded (μ := μ) (X := X) (σ2 := σ2)
      (M := M) h_indep h_meas h_ident h_bound h_mean h_var

/-- Classical CLT statement via CDF convergence at continuity points. -/
theorem central_limit_theorem_cdf_iid_bounded {X : ℕ → Ω → ℝ} {σ2 M : ℝ} {x : ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_bound : ∀ᵐ ω ∂μ, |X 0 ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map (X 0) = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = σ2)
    (hx :
      ContinuousAt (cdf (gaussianReal (0 : ℝ)
        ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩)) x) :
    Tendsto (fun n => cdf (μ.map (normalizedSum X n)) x) atTop
      (𝓝
        (cdf (gaussianReal (0 : ℝ)
          ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h_var⟩) x)) := by
  exact
    tendsto_cdf_normalized_sum_iid_of_bounded (μ := μ) (X := X) (σ2 := σ2) (M := M)
      (x := x) h_indep h_meas h_ident h_bound h_mean h_var hx

/-- CLT from characteristic-function scaling; the remaining step for the full finite-variance CLT. -/
theorem central_limit_theorem_iid_of_charFunScale {X : ℕ → Ω → ℝ} {σ2 : ℝ}
    (h : CLTAssumptions μ X σ2)
    (h_scale : CharFunCLTScale μ (X 0) σ2) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h.h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨gaussianReal (0 : ℝ)
              ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ)
                    ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var⟩))⟩ :
            ProbabilityMeasure ℝ))) := by
  have hσ2 : 0 ≤ σ2 :=
    variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var
  exact
    tendsto_probabilityMeasure_normalized_sum_iid (μ := μ) (X := X) (σ2 := σ2) hσ2
      h.h_indep h.h_meas h.h_ident h_scale

/-- CLT under a finite third absolute moment (Lyapunov p=3). -/
theorem central_limit_theorem_iid_abs_pow3 {X : ℕ → Ω → ℝ} {σ2 : ℝ}
    (h : CLTAssumptions μ X σ2)
    (h_int_abs3 : Integrable (fun x : ℝ => |x| ^ 3) (μ.map (X 0))) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h.h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨gaussianReal (0 : ℝ)
              ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ)
                    ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var⟩))⟩ :
            ProbabilityMeasure ℝ))) := by
  have h_scale : CharFunCLTScale μ (X 0) σ2 :=
    CharFunCLTScale_of_integrable_abs_pow3 (μ := μ) (X := X 0) (σ2 := σ2)
      (h_meas := h.h_meas 0) h_int_abs3 h.h_mean h.h_var
  exact central_limit_theorem_iid_of_charFunScale (μ := μ) (X := X) (σ2 := σ2) h h_scale

/-- CLT under a finite second moment (classical i.i.d. CLT). -/
theorem central_limit_theorem_iid_finite_variance {X : ℕ → Ω → ℝ} {σ2 : ℝ}
    (h : CLTAssumptions μ X σ2)
    (h_int_x2 : Integrable (fun x : ℝ => x ^ 2) (μ.map (X 0))) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h.h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨gaussianReal (0 : ℝ)
              ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var⟩,
            by
              simpa using
                (inferInstance :
                  IsProbabilityMeasure (gaussianReal (0 : ℝ)
                    ⟨σ2, variance_nonneg_of_integral_sq_eq (μ := μ) (X := X 0) h.h_var⟩))⟩ :
            ProbabilityMeasure ℝ))) := by
  have h_scale : CharFunCLTScale μ (X 0) σ2 :=
    CharFunCLTScale_of_integrable_sq (μ := μ) (X := X 0) (σ2 := σ2)
      (h_meas := h.h_meas 0) h_int_x2 h.h_mean h.h_var
  exact central_limit_theorem_iid_of_charFunScale (μ := μ) (X := X) (σ2 := σ2) h h_scale

theorem tendsto_probabilityMeasure_normalized_sum_iid_stdNormal {X : ℕ → Ω → ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_scale : CharFunCLTScale μ (X 0) (1 : ℝ)) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨stdNormalMeasure, by infer_instance⟩ : ProbabilityMeasure ℝ))) := by
  classical
  letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance
  have hσ2 : 0 ≤ (1 : ℝ) := by norm_num
  have h_tendsto :=
    (tendsto_probabilityMeasure_normalized_sum_iid (μ := μ) (X := X) (σ2 := (1 : ℝ)) hσ2
      h_indep h_meas h_ident h_scale)
  have hσ2' : (⟨(1 : ℝ), hσ2⟩ : NNReal) = (1 : NNReal) := by
    ext
    simp
  have h_eq :
      (⟨gaussianReal (0 : ℝ) ⟨(1 : ℝ), hσ2⟩,
        by
          simpa using
            (inferInstance :
              IsProbabilityMeasure (gaussianReal (0 : ℝ) ⟨(1 : ℝ), hσ2⟩))⟩ :
        ProbabilityMeasure ℝ) =
        (⟨stdNormalMeasure, by infer_instance⟩ : ProbabilityMeasure ℝ) := by
    apply ProbabilityMeasure.toMeasure_injective
    simp [stdNormalMeasure, hσ2']
  simpa [h_eq] using h_tendsto

/-- Bounded i.i.d. CLT specialized to the standard normal limit. -/
theorem central_limit_theorem_iid_bounded_stdNormal {X : ℕ → Ω → ℝ} {M : ℝ}
    (h_indep : iIndepFun X μ)
    (h_meas : ∀ i, Measurable (X i))
    (h_ident : ∀ i, IdentDistrib (X i) (X 0) μ μ)
    (h_bound : ∀ᵐ ω ∂μ, |X 0 ω| ≤ M)
    (h_mean : ∫ x, x ∂ μ.map (X 0) = 0)
    (h_var : ∫ x, x ^ 2 ∂ μ.map (X 0) = (1 : ℝ)) :
    (letI : TopologicalSpace (ProbabilityMeasure ℝ) := inferInstance;
      Tendsto
        (fun n =>
          (⟨μ.map (normalizedSum X n),
            Measure.isProbabilityMeasure_map (μ := μ)
              ((measurable_normalizedSum h_meas n).aemeasurable)⟩ :
            ProbabilityMeasure ℝ)) atTop
        (@nhds (ProbabilityMeasure ℝ) (inferInstance)
          (⟨stdNormalMeasure, by infer_instance⟩ : ProbabilityMeasure ℝ))) := by
  have h_scale : CharFunCLTScale μ (X 0) (1 : ℝ) :=
    CharFunCLTScale_of_bounded (μ := μ) (X := X 0) (σ2 := (1 : ℝ)) (M := M)
      (h_meas := h_meas 0) (h_bound := h_bound) (h_mean := h_mean) (h_var := h_var)
  simpa using
    (tendsto_probabilityMeasure_normalized_sum_iid_stdNormal (μ := μ) (X := X)
      h_indep h_meas h_ident h_scale)

end ProbabilityTheory
