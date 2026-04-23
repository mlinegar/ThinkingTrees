import FormalProofs.DSL.TreeIPW
import FormalProofs.OPT.ApproximateLocalLaws

/-!
# FormalProofs/DSL/RuntimeCertificates.lean

Implementation-facing certificate objects and checkers for the TreePO / DSL
runtime surface.

The goal of this file is narrow:
- package the already-proved `computeDSLBound` validity theorems as soundness
  statements for stored runtime artifacts, and
- do the same for stored nodewise local-law audit artifacts.

The checkers validate stored fields rather than introducing a parallel proof
surface.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal
open MeasureTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-- Stored runtime artifact for a computed DSL / TreePO bound. -/
structure RuntimeDSLArtifact where
  samples : List TreeSample
  N : ℕ
  M : ℕ
  R : ℕ
  cal : Option CalibrationSet
  z : ℝ
  bound : DSLBound

namespace RuntimeDSLArtifact

/-- The artifact fields agree with the DSL / TreePO runtime definitions. -/
structure Certifies (art : RuntimeDSLArtifact) : Prop where
  gap_estimate_eq :
    art.bound.gap_estimate = ipwUnionBound art.samples art.N art.M art.R
  se_eq :
    art.bound.se = ipwUnionBoundSE art.samples art.N art.M art.R
  bias_margin_eq :
    art.bound.bias_margin =
      match art.cal with
      | some c => judgeCalibrationErrorBound c art.z
      | none => 0
  confidence_level_eq :
    art.bound.confidence_level = 0.95
  z_score_eq :
    art.bound.z_score = art.z
  z_nonneg :
    0 ≤ art.z
  gap_nonneg :
    0 ≤ ipwUnionBound art.samples art.N art.M art.R

/-- Boolean checker for runtime DSL bound artifacts. -/
def check (art : RuntimeDSLArtifact) : Bool :=
  decide (Certifies art)

theorem check_eq_true_iff (art : RuntimeDSLArtifact) :
    art.check = true ↔ Certifies art := by
  unfold check
  exact decide_eq_true_iff

theorem bound_eq_computeDSLBound
    (art : RuntimeDSLArtifact) (h : Certifies art) :
    art.bound = computeDSLBound art.samples art.N art.M art.R art.cal art.z := by
  cases art with
  | mk samples N M R cal z bound =>
      cases bound with
      | mk gap_estimate se bias_margin confidence_level z_score =>
          cases h with
          | mk gap_estimate_eq se_eq bias_margin_eq confidence_level_eq z_score_eq z_nonneg gap_nonneg =>
              cases cal with
              | none =>
                  change
                    DSLBound.mk gap_estimate se bias_margin confidence_level z_score =
                      DSLBound.mk (ipwUnionBound samples N M R) (ipwUnionBoundSE samples N M R) 0 0.95 z
                  simpa [DSLBound.mk.injEq] using
                    (show gap_estimate = ipwUnionBound samples N M R ∧
                        se = ipwUnionBoundSE samples N M R ∧
                        bias_margin = 0 ∧
                        confidence_level = 0.95 ∧
                        z_score = z from
                      ⟨gap_estimate_eq, se_eq, by simpa using bias_margin_eq,
                        confidence_level_eq, z_score_eq⟩)
              | some c =>
                  change
                    DSLBound.mk gap_estimate se bias_margin confidence_level z_score =
                      DSLBound.mk
                        (ipwUnionBound samples N M R)
                        (ipwUnionBoundSE samples N M R)
                        (judgeCalibrationErrorBound c z)
                        0.95 z
                  simpa [DSLBound.mk.injEq] using
                    (show gap_estimate = ipwUnionBound samples N M R ∧
                        se = ipwUnionBoundSE samples N M R ∧
                        bias_margin = judgeCalibrationErrorBound c z ∧
                        confidence_level = 0.95 ∧
                        z_score = z from
                      ⟨gap_estimate_eq, se_eq, by simpa using bias_margin_eq,
                        confidence_level_eq, z_score_eq⟩)

/-- A checked runtime artifact certifies the pointwise DSL upper bound from a
stored confidence-interval membership fact and calibration envelope. -/
theorem upperBound_of_interval_membership_of_check
    (art : RuntimeDSLArtifact)
    (h_check : art.check = true)
    (gap_oracle gap_judge : ℝ)
    (h_est : gap_judge ∈ Set.Icc
      (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).1
      (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).2)
    (h_cal :
      |gap_oracle - gap_judge| ≤
        match art.cal with
        | some c => judgeCalibrationErrorBound c art.z
        | none => 0) :
    |gap_oracle| ≤ art.bound.upperBound := by
  have h_cert : Certifies art := (check_eq_true_iff art).mp h_check
  have h_bound := bound_eq_computeDSLBound art h_cert
  simpa [h_bound] using
    (dsl_upperBound_of_interval_membership
      (samples := art.samples) (N := art.N) (M := art.M) (R := art.R)
      (cal := art.cal) (z := art.z)
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (h_est := h_est) (h_cal := h_cal)
      (h_est_nonneg := h_cert.gap_nonneg)
      (h_z := h_cert.z_nonneg))

/-- Oracle-measurement version of
`upperBound_of_interval_membership_of_check`. -/
theorem upperBound_of_interval_membership_with_oracleMeasurement_of_check
    (art : RuntimeDSLArtifact)
    (h_check : art.check = true)
    (gap_true gap_oracle gap_judge : ℝ)
    (oracle_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_est : gap_judge ∈ Set.Icc
      (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).1
      (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).2)
    (h_cal :
      |gap_oracle - gap_judge| ≤
        match art.cal with
        | some c => judgeCalibrationErrorBound c art.z
        | none => 0) :
    |gap_true| ≤ art.bound.upperBound + oracle_err := by
  have h_cert : Certifies art := (check_eq_true_iff art).mp h_check
  have h_bound := bound_eq_computeDSLBound art h_cert
  simpa [h_bound] using
    (dsl_upperBound_of_interval_membership_with_oracleMeasurement
      (samples := art.samples) (N := art.N) (M := art.M) (R := art.R)
      (cal := art.cal) (z := art.z)
      (gap_true := gap_true) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (oracle_err := oracle_err)
      (h_oracle := h_oracle)
      (h_est := h_est) (h_cal := h_cal)
      (h_est_nonneg := h_cert.gap_nonneg)
      (h_z := h_cert.z_nonneg))

/-- Event-level soundness of a checked runtime artifact. -/
theorem valid_from_events_of_check
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (art : RuntimeDSLArtifact)
    (h_check : art.check = true)
    (gap_oracle gap_judge : Ω → ℝ)
    (δ_cal δ_est : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥
        match art.cal with
        | some c => judgeCalibrationErrorBound c art.z
        | none => 0} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - ipwUnionBound art.samples art.N art.M art.R| ≥
        art.z * ipwUnionBoundSE art.samples art.N art.M art.R} ≤ δ_est) :
    μ {ω | |gap_oracle ω| ≥ art.bound.upperBound} ≤ δ_cal + δ_est := by
  have h_cert : Certifies art := (check_eq_true_iff art).mp h_check
  have h_bound := bound_eq_computeDSLBound art h_cert
  simpa [h_bound] using
    (computeDSLBound_valid_from_events
      (μ := μ)
      (samples := art.samples) (N := art.N) (M := art.M) (R := art.R)
      (cal := art.cal) (z := art.z)
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (δ_cal := δ_cal) (δ_est := δ_est)
      (h_est_nonneg := h_cert.gap_nonneg)
      (h_cal := h_cal) (h_est := h_est))

/-- Event-level soundness of a checked runtime artifact in the regime with an
additional oracle-measurement envelope. -/
theorem valid_from_events_with_oracleMeasurement_of_check
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (art : RuntimeDSLArtifact)
    (h_check : art.check = true)
    (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (δ_oracle δ_cal δ_est : ENNReal)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ oracle_err} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥
        match art.cal with
        | some c => judgeCalibrationErrorBound c art.z
        | none => 0} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - ipwUnionBound art.samples art.N art.M art.R| ≥
        art.z * ipwUnionBoundSE art.samples art.N art.M art.R} ≤ δ_est) :
    μ {ω | |gap_true ω| ≥ art.bound.upperBound + oracle_err} ≤
      δ_oracle + δ_cal + δ_est := by
  have h_cert : Certifies art := (check_eq_true_iff art).mp h_check
  have h_bound := bound_eq_computeDSLBound art h_cert
  simpa [h_bound] using
    (computeDSLBound_valid_from_events_with_oracleMeasurement
      (μ := μ)
      (samples := art.samples) (N := art.N) (M := art.M) (R := art.R)
      (cal := art.cal) (z := art.z)
      (gap_true := gap_true) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (oracle_err := oracle_err)
      (δ_oracle := δ_oracle) (δ_cal := δ_cal) (δ_est := δ_est)
      (h_est_nonneg := h_cert.gap_nonneg)
      (h_oracle := h_oracle)
      (h_cal := h_cal)
      (h_est := h_est))

/-- Joint interval-event soundness of a checked runtime artifact. -/
theorem valid_from_joint_interval_event_of_check
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (art : RuntimeDSLArtifact)
    (h_check : art.check = true)
    (gap_oracle gap_judge : Ω → ℝ)
    (q : ENNReal)
    (h_event : q ≤ μ {ω |
      gap_judge ω ∈ Set.Icc
        (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).1
        (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).2 ∧
      |gap_oracle ω - gap_judge ω| ≤
        match art.cal with
        | some c => judgeCalibrationErrorBound c art.z
        | none => 0}) :
    q ≤ μ {ω | |gap_oracle ω| ≤ art.bound.upperBound} := by
  have h_cert : Certifies art := (check_eq_true_iff art).mp h_check
  have h_bound := bound_eq_computeDSLBound art h_cert
  simpa [h_bound] using
    (computeDSLBound_valid_from_joint_interval_event
      (μ := μ)
      (samples := art.samples) (N := art.N) (M := art.M) (R := art.R)
      (cal := art.cal) (z := art.z)
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (q := q)
      (h_event := h_event)
      (h_est_nonneg := h_cert.gap_nonneg)
      (h_z := h_cert.z_nonneg))

/-- Oracle-measurement version of
`valid_from_joint_interval_event_of_check`. -/
theorem valid_from_joint_interval_event_with_oracleMeasurement_of_check
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (art : RuntimeDSLArtifact)
    (h_check : art.check = true)
    (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (q : ENNReal)
    (h_event : q ≤ μ {ω |
      |gap_true ω - gap_oracle ω| ≤ oracle_err ∧
      gap_judge ω ∈ Set.Icc
        (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).1
        (ipwUnionBoundConfidenceInterval art.samples art.N art.M art.R art.z).2 ∧
      |gap_oracle ω - gap_judge ω| ≤
        match art.cal with
        | some c => judgeCalibrationErrorBound c art.z
        | none => 0}) :
    q ≤ μ {ω | |gap_true ω| ≤ art.bound.upperBound + oracle_err} := by
  have h_cert : Certifies art := (check_eq_true_iff art).mp h_check
  have h_bound := bound_eq_computeDSLBound art h_cert
  simpa [h_bound] using
    (computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement
      (μ := μ)
      (samples := art.samples) (N := art.N) (M := art.M) (R := art.R)
      (cal := art.cal) (z := art.z)
      (gap_true := gap_true) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (oracle_err := oracle_err)
      (q := q)
      (h_event := h_event)
      (h_est_nonneg := h_cert.gap_nonneg)
      (h_z := h_cert.z_nonneg))

end RuntimeDSLArtifact

/-- Stored runtime artifact for a nodewise local-law audit. -/
structure RuntimeNodewiseAuditArtifact
    {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    (g : Summarizer Strings)
    (T : BinTree Strings)
    (fstar : Strings → Y) where
  cert : FormalProofs.OPT.NodewiseEmpiricalAuditCertificate g T fstar
  upper : FormalProofs.OPT.AuditedApproxUpperBounds g T fstar

namespace RuntimeNodewiseAuditArtifact

/-- The stored audited upper bounds agree with the nodewise empirical audit
certificate used to construct them. -/
structure Certifies
    {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Summarizer Strings}
    {T : BinTree Strings}
    {fstar : Strings → Y}
    (art : RuntimeNodewiseAuditArtifact g T fstar) : Prop where
  upper_eq :
    art.upper =
      FormalProofs.OPT.audited_upper_bounds_of_nodewise_empirical_certificate
        g T fstar art.cert

/-- Boolean checker for a stored nodewise audit artifact. -/
def check
    {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Summarizer Strings}
    {T : BinTree Strings}
    {fstar : Strings → Y}
    (art : RuntimeNodewiseAuditArtifact g T fstar) : Bool :=
  decide (Certifies art)

theorem check_eq_true_iff
    {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Summarizer Strings}
    {T : BinTree Strings}
    {fstar : Strings → Y}
    (art : RuntimeNodewiseAuditArtifact g T fstar) :
    art.check = true ↔ Certifies art := by
  unfold check
  exact decide_eq_true_iff

/-- A checked nodewise audit artifact recovers the exact audited upper-bound
package encoded by the empirical certificate. -/
theorem audited_upper_bounds_eq_of_check
    {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Summarizer Strings}
    {T : BinTree Strings}
    {fstar : Strings → Y}
    (art : RuntimeNodewiseAuditArtifact g T fstar)
    (h_check : art.check = true) :
    art.upper =
      FormalProofs.OPT.audited_upper_bounds_of_nodewise_empirical_certificate
        g T fstar art.cert :=
  (check_eq_true_iff art).mp h_check |>.upper_eq

/-- A checked nodewise audit artifact transports directly to the approximate
local-law bundle already exposed by the OPT surface. -/
theorem approx_bundle_eq_of_check
    {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Summarizer Strings}
    {T : BinTree Strings}
    {fstar : Strings → Y}
    (art : RuntimeNodewiseAuditArtifact g T fstar)
    (h_check : art.check = true) :
    FormalProofs.OPT.approx_bundle_of_audited_upper_bounds g T fstar art.upper =
      FormalProofs.OPT.approx_bundle_of_nodewise_empirical_certificate
        g T fstar art.cert := by
  rw [audited_upper_bounds_eq_of_check art h_check]
  rfl

end RuntimeNodewiseAuditArtifact

end DSL
