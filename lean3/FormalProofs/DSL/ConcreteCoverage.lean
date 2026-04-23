import FormalProofs.DSL.AsymptoticCore
import FormalProbability.CLT.CLT
import FormalProbability.CLT.LevyContinuity
import Mathlib.Probability.Distributions.Gaussian.Real

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped Topology
open MeasureTheory
open ProbabilityTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-- The standard normal law, packaged as a `ProbabilityMeasure`. -/
def stdNormalProbabilityMeasure : ProbabilityMeasure ℝ :=
  ⟨ProbabilityTheory.stdNormalMeasure, by infer_instance⟩

lemma continuousAt_cdf_of_measure_singleton_zero {μ : ProbabilityMeasure ℝ} {x : ℝ}
    (hx : (μ : Measure ℝ) {x} = 0) : ContinuousAt (ProbabilityTheory.cdf μ) x := by
  have hmono := ProbabilityTheory.monotone_cdf (μ := (μ : Measure ℝ))
  have hright :
      Function.rightLim (ProbabilityTheory.cdf μ) x = ProbabilityTheory.cdf μ x := by
    simpa using (StieltjesFunction.rightLim_eq (ProbabilityTheory.cdf μ) x)
  have h_measure :
      (μ : Measure ℝ) {x} = ENNReal.ofReal
        (ProbabilityTheory.cdf μ x - Function.leftLim (ProbabilityTheory.cdf μ) x) := by
    simpa [measure_cdf] using
      (StieltjesFunction.measure_singleton (f := ProbabilityTheory.cdf (μ : Measure ℝ)) x)
  have h_ofReal_zero :
      ENNReal.ofReal
        (ProbabilityTheory.cdf μ x - Function.leftLim (ProbabilityTheory.cdf μ) x) = 0 := by
    rw [← h_measure]
    exact hx
  have h_nonneg :
      0 ≤ ProbabilityTheory.cdf μ x - Function.leftLim (ProbabilityTheory.cdf μ) x := by
    exact sub_nonneg.mpr (hmono.leftLim_le (x := x) le_rfl)
  have h_sub_zero :
      ProbabilityTheory.cdf μ x - Function.leftLim (ProbabilityTheory.cdf μ) x = 0 := by
    apply le_antisymm
    · exact ENNReal.ofReal_eq_zero.mp h_ofReal_zero
    · exact h_nonneg
  have hleft :
      Function.leftLim (ProbabilityTheory.cdf μ) x = ProbabilityTheory.cdf μ x := by
    linarith
  have hiff := hmono.continuousAt_iff_leftLim_eq_rightLim (x := x)
  apply hiff.2
  simpa [hleft, hright]

lemma stdNormal_measure_singleton_zero (x : ℝ) :
    ((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ) {x} = 0 := by
  letI : NoAtoms ProbabilityTheory.stdNormalMeasure := by
    simpa [ProbabilityTheory.stdNormalMeasure] using
      (ProbabilityTheory.noAtoms_gaussianReal (μ := (0 : ℝ)) (v := (1 : NNReal)) one_ne_zero)
  change ProbabilityTheory.stdNormalMeasure {x} = 0
  simp

lemma continuousAt_stdNormal_cdf (x : ℝ) :
    ContinuousAt (ProbabilityTheory.cdf stdNormalProbabilityMeasure) x := by
  exact continuousAt_cdf_of_measure_singleton_zero (μ := stdNormalProbabilityMeasure)
    (stdNormal_measure_singleton_zero x)

/-- Concrete cdf convergence of one-dimensional laws to the standard normal law. -/
def CDFConvergesToStdNormal (μs : ℕ → ProbabilityMeasure ℝ) : Prop :=
  ∀ x, Filter.Tendsto (fun n => ProbabilityTheory.cdf (μs n) x) Filter.atTop
    (𝓝 (ProbabilityTheory.cdf stdNormalProbabilityMeasure x))

/-- The law sequence induced by a measurable real-valued statistic sequence. -/
def LawSeq1D {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (stat_seq : ℕ → Ω → ℝ)
    (h_stat_meas : ∀ n, Measurable (stat_seq n)) :
    ℕ → ProbabilityMeasure ℝ :=
  fun n =>
    ⟨μ.map (stat_seq n),
      Measure.isProbabilityMeasure_map (μ := μ) ((h_stat_meas n).aemeasurable)⟩

/-- Portmanteau-style interval convergence from convergence in distribution of
real-valued statistics. This is the 1D building block used below after
projecting multivariate limits to coordinates. -/
lemma tendsto_measure_Icc_of_tendstoInDistribution
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : ℕ → Ω → ℝ) (Z : Ω → ℝ)
    (h_dist : MeasureTheory.TendstoInDistribution X Filter.atTop Z μ)
    {a b : ℝ} (hab : a ≤ b)
    (ha : (μ.map Z) {a} = 0)
    (hb : (μ.map Z) {b} = 0) :
    Filter.Tendsto (fun n => μ {ω | X n ω ∈ Set.Icc a b}) Filter.atTop
      (𝓝 ((μ.map Z) (Set.Icc a b))) := by
  let μs : ℕ → ProbabilityMeasure ℝ :=
    fun n => ⟨μ.map (X n), Measure.isProbabilityMeasure_map (μ := μ)
      (h_dist.forall_aemeasurable n)⟩
  let μlim : ProbabilityMeasure ℝ :=
    ⟨μ.map Z, Measure.isProbabilityMeasure_map (μ := μ) h_dist.aemeasurable_limit⟩
  have hμs_tendsto : Filter.Tendsto μs Filter.atTop (𝓝 μlim) := by
    simpa [μs, μlim] using h_dist.tendsto
  have h_frontier :
      (((μlim : ProbabilityMeasure ℝ) : Measure ℝ) (frontier (Set.Icc a b))) = 0 := by
    change (μ.map Z) (frontier (Set.Icc a b)) = 0
    rw [frontier_Icc hab]
    have h_pair :
        ({a, b} : Set ℝ) = ({a} : Set ℝ) ∪ ({b} : Set ℝ) := by
      ext x
      simp [or_comm]
    rw [h_pair]
    exact measure_union_null ha hb
  have h_rect :
      Filter.Tendsto
        (fun n => (((μs n : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b)))
        Filter.atTop
        (𝓝 ((((μlim : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b)))) := by
    exact ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto'
      (μs_lim := hμs_tendsto) h_frontier
  have h_map :
      ∀ n,
        (((μs n : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b)) =
          μ {ω | X n ω ∈ Set.Icc a b} := by
    intro n
    simpa [μs] using
      (Measure.map_apply_of_aemeasurable
        (μ := μ) (f := X n) (h_dist.forall_aemeasurable n) measurableSet_Icc)
  simpa [μlim] using
    (Filter.Tendsto.congr' (Filter.Eventually.of_forall h_map) h_rect)

lemma tendstoInDistribution_fin_apply
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (X : ℕ → Ω → Fin d → ℝ) (Z : Ω → Fin d → ℝ)
    (h_dist : MeasureTheory.TendstoInDistribution X Filter.atTop Z μ)
    (i : Fin d) :
    MeasureTheory.TendstoInDistribution
      (fun n ω => X n ω i) Filter.atTop (fun ω => Z ω i) μ := by
  let evali : (Fin d → ℝ) → ℝ := fun x => x i
  have hcont : Continuous evali := by
    simpa [evali] using
      (continuous_apply i : Continuous fun x : Fin d → ℝ => x i)
  simpa [evali, Function.comp] using h_dist.continuous_comp hcont

lemma tendsto_measure_Icc_of_tendsto_cdf
    {μs : ℕ → ProbabilityMeasure ℝ} {μ : ProbabilityMeasure ℝ} {a b : ℝ}
    (hab : a ≤ b)
    (ha : ContinuousAt (ProbabilityTheory.cdf μ) a)
    (hb : ContinuousAt (ProbabilityTheory.cdf μ) b)
    (h :
      ∀ x, ContinuousAt (ProbabilityTheory.cdf μ) x →
        Filter.Tendsto (fun n => ProbabilityTheory.cdf (μs n) x) Filter.atTop
          (𝓝 (ProbabilityTheory.cdf μ x))) :
    Filter.Tendsto (fun n => ((μs n : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b))
      Filter.atTop
      (𝓝 (((μ : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b))) := by
  have hμs : Filter.Tendsto μs Filter.atTop (𝓝 μ) :=
    ProbabilityTheory.tendsto_probabilityMeasure_of_tendsto_cdf_cont h
  have ha0 : (((μ : ProbabilityMeasure ℝ) : Measure ℝ) {a}) = 0 :=
    ProbabilityTheory.cdf_continuousAt_measure_singleton_zero (μ := μ) ha
  have hb0 : (((μ : ProbabilityMeasure ℝ) : Measure ℝ) {b}) = 0 :=
    ProbabilityTheory.cdf_continuousAt_measure_singleton_zero (μ := μ) hb
  have h_frontier :
      (((μ : ProbabilityMeasure ℝ) : Measure ℝ) (frontier (Set.Icc a b))) = 0 := by
    rw [frontier_Icc hab]
    have h_pair :
        ({a, b} : Set ℝ) = ({a} : Set ℝ) ∪ ({b} : Set ℝ) := by
      ext x
      simp [or_comm]
    rw [h_pair]
    exact measure_union_null ha0 hb0
  exact ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto'
    (μs_lim := hμs) h_frontier

lemma tendsto_measure_Icc_of_tendsto_cdf_stdNormal
    {μs : ℕ → ProbabilityMeasure ℝ} {a b : ℝ} (hab : a ≤ b)
    (h : CDFConvergesToStdNormal μs) :
    Filter.Tendsto (fun n => ((μs n : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b))
      Filter.atTop
      (𝓝 (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ) (Set.Icc a b))) := by
  exact tendsto_measure_Icc_of_tendsto_cdf (μ := stdNormalProbabilityMeasure) hab
    (continuousAt_stdNormal_cdf a) (continuousAt_stdNormal_cdf b)
    (fun x _ => h x)

/-- A reusable first-principles 1D coverage route:
if a real-valued statistic has laws whose cdfs converge to the standard normal,
and confidence-interval coverage is equivalent to that statistic lying in a
fixed interval, then asymptotic coverage follows without any abstract coverage
axiom. -/
theorem asymptoticCoverage_oneDim_of_cdfConvergesToStdNormal_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (stat_seq : ℕ → Ω → ℝ)
    (CI_seq : ℕ → Ω → Fin 1 → ℝ × ℝ)
    (β_star : Fin 1 → ℝ)
    (α a b : ℝ)
    (hab : a ≤ b)
    (h_stat_meas : ∀ n, Measurable (stat_seq n))
    (h_cdf : CDFConvergesToStdNormal (LawSeq1D μ stat_seq h_stat_meas))
    (h_event_eq :
      ∀ n,
        {ω | β_star 0 ∈ Set.Icc (CI_seq n ω 0).1 (CI_seq n ω 0).2} =
          {ω | stat_seq n ω ∈ Set.Icc a b})
    (h_calibration :
      (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
        (Set.Icc a b)) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  intro i
  fin_cases i
  have h_interval :
      Filter.Tendsto
        (fun n =>
          (((LawSeq1D μ stat_seq h_stat_meas n : ProbabilityMeasure ℝ) :
            Measure ℝ) (Set.Icc a b)))
        Filter.atTop
        (𝓝
          ((((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
            (Set.Icc a b)))) := by
    exact tendsto_measure_Icc_of_tendsto_cdf_stdNormal
      (μs := LawSeq1D μ stat_seq h_stat_meas) (a := a) (b := b) hab h_cdf
  have h_map :
      ∀ n,
        (((LawSeq1D μ stat_seq h_stat_meas n : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc a b)) =
          μ {ω | stat_seq n ω ∈ Set.Icc a b} := by
    intro n
    simpa [LawSeq1D] using
      (Measure.map_apply (μ := μ) (f := stat_seq n) (h_stat_meas n) measurableSet_Icc)
  have h_target_stat :
      Filter.Tendsto
        (fun n =>
          μ {ω | stat_seq n ω ∈ Set.Icc a b})
        Filter.atTop
        (𝓝
          ((((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
            (Set.Icc a b)))) := by
    simpa [h_map] using h_interval
  have h_target :
      Filter.Tendsto
        (fun n =>
          μ {ω | β_star 0 ∈ Set.Icc (CI_seq n ω 0).1 (CI_seq n ω 0).2})
        Filter.atTop
        (𝓝
          ((((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
            (Set.Icc a b)))) := by
    exact Filter.Tendsto.congr'
      (Filter.Eventually.of_forall fun n => by rw [← h_event_eq n])
      h_target_stat
  simpa [h_calibration] using h_target

/-- Any coordinatewise limit witness yields asymptotic coverage directly. -/
theorem CoordinateCoverageLimitWitness.asymptoticCoverage
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    {stat_seq : ℕ → Ω → Fin d → ℝ}
    {CI_seq : ℕ → Ω → Fin d → ℝ × ℝ}
    {β_star : Fin d → ℝ}
    {α : ℝ}
    (h_cov :
      CoordinateCoverageLimitWitness μ stat_seq CI_seq β_star α) :
    AsymptoticCoverage μ CI_seq β_star α := by
  intro i
  have h_stat :
      Filter.Tendsto
        (fun n => μ {ω | stat_seq n ω i ∈ Set.Icc (h_cov.events.lower i) (h_cov.events.upper i)})
        Filter.atTop
        (𝓝 ((μ.map (fun ω => h_cov.limit_stat ω i))
          (Set.Icc (h_cov.events.lower i) (h_cov.events.upper i)))) := by
    exact tendsto_measure_Icc_of_tendstoInDistribution
      (μ := μ)
      (X := fun n ω => stat_seq n ω i)
      (Z := fun ω => h_cov.limit_stat ω i)
      (h_dist := h_cov.coord_tendsto i)
      (hab := h_cov.events.interval i)
      (ha := (h_cov.boundary_zero i).1)
      (hb := (h_cov.boundary_zero i).2)
  have h_target :
      Filter.Tendsto
        (fun n => μ {ω | stat_seq n ω i ∈ Set.Icc (h_cov.events.lower i) (h_cov.events.upper i)})
        Filter.atTop
        (𝓝 (ENNReal.ofReal (1 - α))) := by
    simpa [h_cov.calibration i] using h_stat
  exact Filter.Tendsto.congr'
    (Filter.Eventually.of_forall fun n => by
      rw [← h_cov.events.event_eq n i])
    h_target

/-- A normal-coverage construction discharges the abstract coverage transfer
layer from first principles. -/
theorem NormalCoverageConstruction.asymptoticCoverage
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    {centered_scaled_seq : ℕ → Ω → Fin d → ℝ}
    {CI_seq : ℕ → Ω → Fin d → ℝ × ℝ}
    {β_star : Fin d → ℝ}
    {α : ℝ}
    {V : Matrix (Fin d) (Fin d) ℝ}
    (h_cov :
      NormalCoverageConstruction μ centered_scaled_seq CI_seq β_star α V)
    (h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V) :
    AsymptoticCoverage μ CI_seq β_star α := by
  rcases h_norm with ⟨Z, hZ_normal, hZ_dist⟩
  let h_limit :
      CoordinateCoverageLimitWitness μ h_cov.stat_seq CI_seq β_star α :=
    { limit_stat := h_cov.limit_stat Z
      events := h_cov.events
      coord_tendsto := h_cov.coord_tendsto hZ_normal hZ_dist
      boundary_zero := h_cov.boundary_zero hZ_normal
      calibration := h_cov.calibration hZ_normal }
  exact h_limit.asymptoticCoverage (μ := μ)

/-- Multivariate coordinatewise coverage from weak convergence of the full
studentized statistic vector plus explicit coordinate-strip event identities. -/
theorem asymptoticCoverage_of_tendstoInDistribution_of_coordIcc_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (stat_seq : ℕ → Ω → Fin d → ℝ)
    (Z : Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_dist : MeasureTheory.TendstoInDistribution stat_seq Filter.atTop Z μ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | stat_seq n ω i ∈ Set.Icc (lower i) (upper i)})
    (h_boundary_zero :
      ∀ i,
        (μ.map (fun ω => Z ω i)) {lower i} = 0 ∧
          (μ.map (fun ω => Z ω i)) {upper i} = 0)
    (h_calibration :
      ∀ i, (μ.map (fun ω => Z ω i)) (Set.Icc (lower i) (upper i)) =
        ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  let h_limit :
      CoordinateCoverageLimitWitness μ stat_seq CI_seq β_star α :=
    { limit_stat := Z
      events :=
        { lower := lower
          upper := upper
          interval := h_interval
          event_eq := h_event_eq }
      coord_tendsto := fun i => tendstoInDistribution_fin_apply μ stat_seq Z h_dist i
      boundary_zero := h_boundary_zero
      calibration := h_calibration }
  exact h_limit.asymptoticCoverage (μ := μ)

/-- Multivariate coordinatewise coverage when the weak limit has standard-normal
coordinate marginals. This is the concrete multivariate analogue of the 1D
Wald-coverage route. -/
theorem asymptoticCoverage_of_tendstoInDistribution_of_coordStdNormal_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (stat_seq : ℕ → Ω → Fin d → ℝ)
    (Z : Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_dist : MeasureTheory.TendstoInDistribution stat_seq Filter.atTop Z μ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | stat_seq n ω i ∈ Set.Icc (lower i) (upper i)})
    (h_coord_stdNormal :
      ∀ i,
        μ.map (fun ω => Z ω i) =
          ((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ))
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  refine asymptoticCoverage_of_tendstoInDistribution_of_coordIcc_of_eventEq
    (μ := μ) (stat_seq := stat_seq) (Z := Z) (CI_seq := CI_seq)
    (β_star := β_star) (α := α) (lower := lower) (upper := upper)
    (h_dist := h_dist) (h_interval := h_interval) (h_event_eq := h_event_eq)
    (h_boundary_zero := ?_) (h_calibration := ?_)
  · intro i
    constructor
    · rw [h_coord_stdNormal i]
      exact stdNormal_measure_singleton_zero (lower i)
    · rw [h_coord_stdNormal i]
      exact stdNormal_measure_singleton_zero (upper i)
  · intro i
    rw [h_coord_stdNormal i]
    exact h_calibration i

end DSL
