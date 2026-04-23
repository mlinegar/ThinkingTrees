import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.Probability.Distributions.Gaussian.Real

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Topology
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-- MGF characterization hook for a multivariate normal limit. -/
def NormalMGFCharacteristic {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (Z : Ω → Fin d → ℝ) (mean : Fin d → ℝ)
    (variance : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  ∀ t : Fin d → ℝ,
    Integrable (fun ω => Real.exp (∑ i, t i * Z ω i)) μ ∧
      (∫ ω, Real.exp (∑ i, t i * Z ω i) ∂μ) =
        Real.exp
          ((∑ i, t i * mean i) +
            (1 / 2 : ℝ) * ∑ i, ∑ j, t i * variance i j * t j)

/-- Convergence in probability (mathlib: `TendstoInMeasure`). -/
def ConvergesInProbability {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {E : Type*} [PseudoMetricSpace E]
    (seq : ℕ → Ω → E) (limit : Ω → E) : Prop :=
  MeasureTheory.TendstoInMeasure μ seq Filter.atTop limit

/-- Abstract predicate for a normal limit.

Mathlib does not yet provide a packaged multivariate normal distribution,
so we record normality as an explicit assumption. -/
structure NormalLimit {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (Z : Ω → Fin d → ℝ) (mean : Fin d → ℝ)
    (variance : Matrix (Fin d) (Fin d) ℝ) : Prop where
  /-- Abstract normal-law characterization (MGF form). -/
  law_characterization : NormalMGFCharacteristic μ Z mean variance
  /-- Each coordinate variance is nonnegative. -/
  variance_diag_nonneg : ∀ i, 0 ≤ variance i i
  /-- Coordinate marginals are the corresponding univariate Gaussian laws. -/
  coord_gaussian :
    ∀ i,
      μ.map (fun ω => Z ω i) =
        ProbabilityTheory.gaussianReal (mean i)
          ⟨variance i i, variance_diag_nonneg i⟩

/-- Convergence in distribution to a normal limit (mathlib: `TendstoInDistribution`). -/
def ConvergesInDistributionToNormal {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (seq : ℕ → Ω → Fin d → ℝ)
    (mean : Fin d → ℝ)
    (variance : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  ∃ Z : Ω → Fin d → ℝ,
    NormalLimit μ Z mean variance ∧
      MeasureTheory.TendstoInDistribution seq Filter.atTop Z μ

/-- Asymptotic coverage of a confidence interval sequence. -/
def AsymptoticCoverage {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (α : ℝ) : Prop :=
  ∀ i, Filter.Tendsto
    (fun n =>
      μ {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2})
    Filter.atTop (𝓝 (ENNReal.ofReal (1 - α)))

/-- Event-level witness reducing interval coverage to a coordinatewise strip
event for a statistic sequence. -/
structure CoverageEventWitness {Ω : Type*} [MeasurableSpace Ω]
    {d : ℕ}
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (stat_seq : ℕ → Ω → Fin d → ℝ) where
  /-- Lower endpoint of the target interval in the limiting statistic scale. -/
  lower : Fin d → ℝ
  /-- Upper endpoint of the target interval in the limiting statistic scale. -/
  upper : Fin d → ℝ
  /-- The limiting interval is nonempty in every coordinate. -/
  interval : ∀ i, lower i ≤ upper i
  /-- Coverage for coordinate `i` is exactly the event that the transformed
  statistic falls in the corresponding interval. -/
  event_eq :
    ∀ n i,
      {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
        {ω | stat_seq n ω i ∈ Set.Icc (lower i) (upper i)}

/-- Fully generic first-principles coordinatewise coverage witness. It only
requires coordinatewise convergence in distribution of the relevant statistic,
plus boundary-nullness and calibration of the limiting interval events. -/
structure CoordinateCoverageLimitWitness {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (stat_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (α : ℝ) where
  /-- Limiting statistic on the same probability space. -/
  limit_stat : Ω → Fin d → ℝ
  /-- Event-level reduction of coverage to interval membership. -/
  events : CoverageEventWitness CI_seq β_star stat_seq
  /-- Coordinatewise convergence in distribution of the statistic. -/
  coord_tendsto :
    ∀ i,
      MeasureTheory.TendstoInDistribution
        (fun n ω => stat_seq n ω i) Filter.atTop (fun ω => limit_stat ω i) μ
  /-- The limiting law puts no mass on the interval boundaries. -/
  boundary_zero :
    ∀ i,
      (μ.map (fun ω => limit_stat ω i)) {events.lower i} = 0 ∧
        (μ.map (fun ω => limit_stat ω i)) {events.upper i} = 0
  /-- The limiting interval probability equals the target nominal coverage. -/
  calibration :
    ∀ i,
      (μ.map (fun ω => limit_stat ω i)) (Set.Icc (events.lower i) (events.upper i)) =
        ENNReal.ofReal (1 - α)

/-- Generic coverage-construction interface from asymptotic normality. Instead
of postulating coverage for arbitrary confidence intervals, it asks for the
actual mathematical ingredients needed to derive coverage from first
principles: a transformed statistic, the coverage-event identity, coordinatewise
limit convergence, boundary-nullness, and calibration. -/
structure NormalCoverageConstruction {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (α : ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ) where
  /-- Statistic whose interval event characterizes coverage. -/
  stat_seq : ℕ → Ω → Fin d → ℝ
  /-- Transformation of a normal-limit witness into the statistic limit. -/
  limit_stat : (Ω → Fin d → ℝ) → Ω → Fin d → ℝ
  /-- Event-level reduction of coverage to interval membership. -/
  events : CoverageEventWitness CI_seq β_star stat_seq
  /-- Coordinatewise convergence of the transformed statistic, derived from any
  normal-limit witness for the centered/scaled estimator. -/
  coord_tendsto :
    ∀ {Z : Ω → Fin d → ℝ},
      NormalLimit μ Z (fun _ => 0) V →
      MeasureTheory.TendstoInDistribution centered_scaled_seq Filter.atTop Z μ →
      ∀ i,
        MeasureTheory.TendstoInDistribution
          (fun n ω => stat_seq n ω i) Filter.atTop (fun ω => limit_stat Z ω i) μ
  /-- Boundary-nullness for the transformed limit law. -/
  boundary_zero :
    ∀ {Z : Ω → Fin d → ℝ},
      NormalLimit μ Z (fun _ => 0) V →
      ∀ i,
        (μ.map (fun ω => limit_stat Z ω i)) {events.lower i} = 0 ∧
          (μ.map (fun ω => limit_stat Z ω i)) {events.upper i} = 0
  /-- Calibration of the transformed limit interval event. -/
  calibration :
    ∀ {Z : Ω → Fin d → ℝ},
      NormalLimit μ Z (fun _ => 0) V →
      ∀ i,
        (μ.map (fun ω => limit_stat Z ω i)) (Set.Icc (events.lower i) (events.upper i)) =
          ENNReal.ofReal (1 - α)

/-- Wald-style coverage transfer from asymptotic normality. -/
def CoverageFromAsymptoticNormal {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ) : Prop :=
  ∀ (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (α : ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ),
    ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V →
    AsymptoticCoverage μ CI_seq β_star α

/-- Backward-compatible name for the coverage transfer assumption. -/
abbrev CoverageAxioms {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ) : Prop :=
  CoverageFromAsymptoticNormal μ d

end DSL
