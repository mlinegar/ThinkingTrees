import FormalProofs.DSL.Honesty
import FormalProofs.DSL.DocumentStructure
import FormalProofs.DSL.TreePOEndToEnd
import FormalProofs.OPT.InfluenceWeightedLocalLaws

/-!
# Unified Learning Certificate

This file packages the paper-facing certificate for the unified learned
chunker / learned local-law / oracle-audit pipeline.

The honesty and top-level-unit assumptions live in `DSL.Honesty`.  The
certificate below is deliberately small: it records the four error terms that
remain after those assumptions justify using held-out or cross-fitted
estimates, and proves the final deterministic and high-probability envelopes.
-/

namespace DSL

open MeasureTheory ProbabilityTheory
open scoped Classical BigOperators NNReal ENNReal

noncomputable section

/-- Paper-facing error certificate for a unified learning run.

`reportedEstimate` is the held-out/cross-fitted estimate that is reported in
the table or certificate.  The four radii correspond to:

* local-law residual / transported distortion,
* oracle-to-judge calibration,
* honest statistical estimation,
* clipping or reporting post-processing.
-/
structure UnifiedLearningErrorCertificate where
  reportedEstimate : ℝ
  localLawRadius : ℝ
  calibrationRadius : ℝ
  estimationRadius : ℝ
  clippingRadius : ℝ

namespace UnifiedLearningErrorCertificate

/-- The final scalar radius reported by the unified learning certificate. -/
def totalBound (c : UnifiedLearningErrorCertificate) : ℝ :=
  |c.reportedEstimate| + c.localLawRadius + c.calibrationRadius +
    c.estimationRadius + c.clippingRadius

@[simp]
theorem totalBound_eq (c : UnifiedLearningErrorCertificate) :
    c.totalBound =
      |c.reportedEstimate| + c.localLawRadius + c.calibrationRadius +
        c.estimationRadius + c.clippingRadius := rfl

/-- Current paper certificate as a unified certificate with an explicit reported
estimate.  This is the compatibility bridge from the older end-to-end
`PaperErrorCertificate` surface. -/
def ofPaperErrorCertificate
    (reportedEstimate : ℝ) (c : PaperErrorCertificate) :
    UnifiedLearningErrorCertificate :=
  { reportedEstimate := reportedEstimate
    localLawRadius := c.transportedDistortion
    calibrationRadius := c.calibration
    estimationRadius := c.estimation
    clippingRadius := c.clipping }

theorem ofPaperErrorCertificate_totalBound
    (reportedEstimate : ℝ) (c : PaperErrorCertificate) :
    (ofPaperErrorCertificate reportedEstimate c).totalBound =
      |reportedEstimate| + c.totalObjectiveBound := by
  simp [ofPaperErrorCertificate, totalBound, PaperErrorCertificate.totalObjectiveBound,
    PaperErrorCertificate.transportedDistortion]
  ring

end UnifiedLearningErrorCertificate

/-- Canonical paper-facing certificate name for the unified learning procedure. -/
abbrev CurrentPaperErrorCertificate := UnifiedLearningErrorCertificate

/-- Generic provenance record for a radius: the event that the absolute error
exceeds the radius has probability at most `delta`. -/
structure RadiusEventEvidence {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (error : Ω → ℝ) (radius : ℝ) where
  delta : ENNReal
  event_bound : μ {ω | |error ω| ≥ radius} ≤ delta

/-- Component-radius provenance for the final unified certificate. -/
structure UnifiedLearningComponentEvidence {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : Ω → ℝ) where
  localLaw :
    RadiusEventEvidence μ (fun ω => targetGap ω - oracleGap ω) c.localLawRadius
  calibration :
    RadiusEventEvidence μ (fun ω => oracleGap ω - judgeGap ω) c.calibrationRadius
  estimation :
    RadiusEventEvidence μ (fun ω => judgeGap ω - estimateBeforeClip ω) c.estimationRadius
  clipping :
    RadiusEventEvidence μ
      (fun ω => estimateBeforeClip ω - c.reportedEstimate) c.clippingRadius

namespace UnifiedLearningComponentEvidence

/-- Total failure probability advertised by the component provenance records. -/
def totalDelta {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} {c : UnifiedLearningErrorCertificate}
    {targetGap oracleGap judgeGap estimateBeforeClip : Ω → ℝ}
    (e : UnifiedLearningComponentEvidence μ c targetGap oracleGap judgeGap estimateBeforeClip) :
    ENNReal :=
  e.localLaw.delta + e.calibration.delta + e.estimation.delta + e.clipping.delta

end UnifiedLearningComponentEvidence

/-- Paper-level assumptions bundled for the unified procedure.  The arithmetic
certificate theorems below do not need all fields computationally; the fields
state what a paper claim must have fixed before using the certificate. -/
structure UnifiedLearningPaperAssumptions
    {Ω Case Truth Row βc βg βo Artifact γ ArtifactId : Type*}
    [MeasurableSpace Ω] [MeasurableSpace Case] [MeasurableSpace Truth] where
  μ : Measure Ω
  probability : IsProbabilityMeasure μ
  X : ℕ → Ω → Case
  truth : Case → Truth
  top_level_sampling : TopLevelIID μ X truth ∨ TopLevelExchangeable μ X truth
  splits : ThreeLayerSplit Case
  parent : ParentOf Case Row
  train_chunker : List Row → βc
  train_g : List Row → βg
  train_oracle : List Row → βo
  artifact : Artifact
  eval_fn : Artifact → List Row → γ
  honesty :
    UnifiedLearningHonesty splits parent train_chunker train_g train_oracle artifact eval_fn
  chunking : ChunkPartitionContract Case
  manifest : RunManifestContract Case Row ArtifactId
  manifest_parent : manifest.parent = parent
  manifest_roles : ManifestRolesConsistent splits manifest
  manifest_supports_valid : ManifestSupportsValid chunking.unit manifest

/-- Deterministic final-gap certificate from the four certified components. -/
theorem unified_learning_abs_gap_le_totalBound
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : ℝ)
    (h_local : |targetGap - oracleGap| ≤ c.localLawRadius)
    (h_calibration : |oracleGap - judgeGap| ≤ c.calibrationRadius)
    (h_estimation : |judgeGap - estimateBeforeClip| ≤ c.estimationRadius)
    (h_clipping : |estimateBeforeClip - c.reportedEstimate| ≤ c.clippingRadius) :
    |targetGap| ≤ c.totalBound := by
  let eLocal := targetGap - oracleGap
  let eCalibration := oracleGap - judgeGap
  let eEstimation := judgeGap - estimateBeforeClip
  let eClipping := estimateBeforeClip - c.reportedEstimate
  have h_decomp :
      targetGap =
        eLocal + eCalibration + eEstimation + eClipping + c.reportedEstimate := by
    dsimp [eLocal, eCalibration, eEstimation, eClipping]
    ring
  have htri1 :
      |eLocal + eCalibration + eEstimation + eClipping + c.reportedEstimate| ≤
        |eLocal + eCalibration + eEstimation + eClipping| + |c.reportedEstimate| :=
    abs_add_le _ _
  have htri2 :
      |eLocal + eCalibration + eEstimation + eClipping| ≤
        |eLocal + eCalibration + eEstimation| + |eClipping| := by
    simpa [add_assoc] using abs_add_le (eLocal + eCalibration + eEstimation) eClipping
  have htri3 :
      |eLocal + eCalibration + eEstimation| ≤
        |eLocal + eCalibration| + |eEstimation| := by
    simpa [add_assoc] using abs_add_le (eLocal + eCalibration) eEstimation
  have htri4 :
      |eLocal + eCalibration| ≤ |eLocal| + |eCalibration| :=
    abs_add_le _ _
  have htri :
      |targetGap| ≤
        |eLocal| + |eCalibration| + |eEstimation| + |eClipping| +
          |c.reportedEstimate| := by
    rw [h_decomp]
    linarith
  dsimp [eLocal, eCalibration, eEstimation, eClipping] at htri
  dsimp [UnifiedLearningErrorCertificate.totalBound]
  linarith

/-- Same deterministic certificate, with the explicit honesty contract in the
statement.  Honesty is what licenses the four component bounds as held-out or
cross-fitted quantities; the arithmetic envelope itself is deterministic. -/
theorem unified_learning_honest_certificate
    {Case Row βc βg βo Artifact γ : Type*}
    (splits : ThreeLayerSplit Case)
    (parent : Row → Case)
    (train_chunker : List Row → βc)
    (train_g : List Row → βg)
    (train_oracle : List Row → βo)
    (artifact : Artifact)
    (eval_fn : Artifact → List Row → γ)
    (_h_honesty :
      UnifiedLearningHonesty splits parent train_chunker train_g train_oracle artifact eval_fn)
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : ℝ)
    (h_local : |targetGap - oracleGap| ≤ c.localLawRadius)
    (h_calibration : |oracleGap - judgeGap| ≤ c.calibrationRadius)
    (h_estimation : |judgeGap - estimateBeforeClip| ≤ c.estimationRadius)
    (h_clipping : |estimateBeforeClip - c.reportedEstimate| ≤ c.clippingRadius) :
    |targetGap| ≤ c.totalBound :=
  unified_learning_abs_gap_le_totalBound
    c targetGap oracleGap judgeGap estimateBeforeClip
    h_local h_calibration h_estimation h_clipping

/-- Deterministic final-gap certificate when the local-law component is supplied
by the influence-weighted finite-sample certificate. -/
theorem unified_learning_abs_gap_le_totalBound_from_influence
    {AuditRow : Type*} [Fintype AuditRow]
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : ℝ)
    (lambda trueResidual proxyResidual : AuditRow → ℝ)
    (iw :
      FormalProofs.OPT.InfluenceWeightedErrorCertificate
        (|targetGap - oracleGap|) lambda trueResidual proxyResidual)
    (h_local_radius : iw.totalBound ≤ c.localLawRadius)
    (h_calibration : |oracleGap - judgeGap| ≤ c.calibrationRadius)
    (h_estimation : |judgeGap - estimateBeforeClip| ≤ c.estimationRadius)
    (h_clipping : |estimateBeforeClip - c.reportedEstimate| ≤ c.clippingRadius) :
    |targetGap| ≤ c.totalBound := by
  have h_local :
      |targetGap - oracleGap| ≤ c.localLawRadius := by
    exact le_trans
      (FormalProofs.OPT.InfluenceWeightedErrorCertificate.rootError_le_totalBound iw)
      h_local_radius
  exact unified_learning_abs_gap_le_totalBound
    c targetGap oracleGap judgeGap estimateBeforeClip
    h_local h_calibration h_estimation h_clipping

/-- Final deterministic paper certificate for the unified learning procedure.

This is the named end-to-end corollary: top-level sampling, three-layer
honesty, admissible chunk/manifest structure, and the influence-weighted
local-law certificate are all explicit assumptions.  The proof then reduces to
the deterministic unified certificate. -/
theorem unified_learning_final_paper_certificate
    {Ω Case Truth Row βc βg βo Artifact γ ArtifactId AuditRow : Type*}
    [MeasurableSpace Ω] [MeasurableSpace Case] [MeasurableSpace Truth]
    [Fintype AuditRow]
    (_ctx :
      UnifiedLearningPaperAssumptions
        (Ω := Ω) (Case := Case) (Truth := Truth) (Row := Row)
        (βc := βc) (βg := βg) (βo := βo)
        (Artifact := Artifact) (γ := γ) (ArtifactId := ArtifactId))
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : ℝ)
    (lambda trueResidual proxyResidual : AuditRow → ℝ)
    (iw :
      FormalProofs.OPT.InfluenceWeightedErrorCertificate
        (|targetGap - oracleGap|) lambda trueResidual proxyResidual)
    (h_local_radius : iw.totalBound ≤ c.localLawRadius)
    (h_calibration : |oracleGap - judgeGap| ≤ c.calibrationRadius)
    (h_estimation : |judgeGap - estimateBeforeClip| ≤ c.estimationRadius)
    (h_clipping : |estimateBeforeClip - c.reportedEstimate| ≤ c.clippingRadius) :
    |targetGap| ≤ c.totalBound :=
  unified_learning_abs_gap_le_totalBound_from_influence
    c targetGap oracleGap judgeGap estimateBeforeClip
    lambda trueResidual proxyResidual iw h_local_radius
    h_calibration h_estimation h_clipping

/-- High-probability envelope for the same certificate.  If each component
failure event has probability at most its advertised delta, then the event that
the target gap exceeds the reported total bound has probability at most the sum
of the component deltas. -/
theorem unified_learning_certificate_high_prob
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : Ω → ℝ)
    (δ_local δ_calibration δ_estimation δ_clipping : ENNReal)
    (h_local :
      μ {ω | |targetGap ω - oracleGap ω| ≥ c.localLawRadius} ≤ δ_local)
    (h_calibration :
      μ {ω | |oracleGap ω - judgeGap ω| ≥ c.calibrationRadius} ≤ δ_calibration)
    (h_estimation :
      μ {ω | |judgeGap ω - estimateBeforeClip ω| ≥ c.estimationRadius} ≤
        δ_estimation)
    (h_clipping :
      μ {ω | |estimateBeforeClip ω - c.reportedEstimate| ≥ c.clippingRadius} ≤
        δ_clipping) :
    μ {ω | c.totalBound < |targetGap ω|} ≤
      δ_local + δ_calibration + δ_estimation + δ_clipping := by
  let E_local : Set Ω := {ω | |targetGap ω - oracleGap ω| ≥ c.localLawRadius}
  let E_calibration : Set Ω :=
    {ω | |oracleGap ω - judgeGap ω| ≥ c.calibrationRadius}
  let E_estimation : Set Ω :=
    {ω | |judgeGap ω - estimateBeforeClip ω| ≥ c.estimationRadius}
  let E_clipping : Set Ω :=
    {ω | |estimateBeforeClip ω - c.reportedEstimate| ≥ c.clippingRadius}
  let E_bad : Set Ω := {ω | c.totalBound < |targetGap ω|}
  have h_subset :
      E_bad ⊆ E_local ∪ (E_calibration ∪ (E_estimation ∪ E_clipping)) := by
    intro ω hbad
    by_contra hnot
    have hnot_local : ω ∉ E_local := by
      intro hmem
      exact hnot (Or.inl hmem)
    have hnot_calibration : ω ∉ E_calibration := by
      intro hmem
      exact hnot (Or.inr (Or.inl hmem))
    have hnot_estimation : ω ∉ E_estimation := by
      intro hmem
      exact hnot (Or.inr (Or.inr (Or.inl hmem)))
    have hnot_clipping : ω ∉ E_clipping := by
      intro hmem
      exact hnot (Or.inr (Or.inr (Or.inr hmem)))
    have hlocal_lt : |targetGap ω - oracleGap ω| < c.localLawRadius := by
      have : ¬ |targetGap ω - oracleGap ω| ≥ c.localLawRadius := by
        simpa [E_local] using hnot_local
      exact lt_of_not_ge this
    have hcalibration_lt : |oracleGap ω - judgeGap ω| < c.calibrationRadius := by
      have : ¬ |oracleGap ω - judgeGap ω| ≥ c.calibrationRadius := by
        simpa [E_calibration] using hnot_calibration
      exact lt_of_not_ge this
    have hestimation_lt :
        |judgeGap ω - estimateBeforeClip ω| < c.estimationRadius := by
      have : ¬ |judgeGap ω - estimateBeforeClip ω| ≥ c.estimationRadius := by
        simpa [E_estimation] using hnot_estimation
      exact lt_of_not_ge this
    have hclipping_lt :
        |estimateBeforeClip ω - c.reportedEstimate| < c.clippingRadius := by
      have : ¬ |estimateBeforeClip ω - c.reportedEstimate| ≥ c.clippingRadius := by
        simpa [E_clipping] using hnot_clipping
      exact lt_of_not_ge this
    have hbound :
        |targetGap ω| ≤ c.totalBound :=
      unified_learning_abs_gap_le_totalBound
        c (targetGap ω) (oracleGap ω) (judgeGap ω) (estimateBeforeClip ω)
        (le_of_lt hlocal_lt) (le_of_lt hcalibration_lt)
        (le_of_lt hestimation_lt) (le_of_lt hclipping_lt)
    exact not_lt_of_ge hbound hbad
  have h_bad_measure :
      μ E_bad ≤ μ (E_local ∪ (E_calibration ∪ (E_estimation ∪ E_clipping))) :=
    measure_mono h_subset
  have h_union :
      μ (E_local ∪ (E_calibration ∪ (E_estimation ∪ E_clipping))) ≤
        μ E_local + μ E_calibration + μ E_estimation + μ E_clipping := by
    calc
      μ (E_local ∪ (E_calibration ∪ (E_estimation ∪ E_clipping))) ≤
          μ E_local + μ (E_calibration ∪ (E_estimation ∪ E_clipping)) :=
        measure_union_le (μ := μ) _ _
      _ ≤ μ E_local + (μ E_calibration + μ (E_estimation ∪ E_clipping)) := by
        exact add_le_add le_rfl
          (measure_union_le (μ := μ) (s := E_calibration) (t := E_estimation ∪ E_clipping))
      _ ≤ μ E_local + (μ E_calibration + (μ E_estimation + μ E_clipping)) := by
        exact add_le_add le_rfl
          (add_le_add le_rfl
            (measure_union_le (μ := μ) (s := E_estimation) (t := E_clipping)))
      _ = μ E_local + μ E_calibration + μ E_estimation + μ E_clipping := by
        ac_rfl
  have h_components :
      μ E_local + μ E_calibration + μ E_estimation + μ E_clipping ≤
        δ_local + δ_calibration + δ_estimation + δ_clipping := by
    simpa [add_assoc] using
      add_le_add (add_le_add h_local h_calibration)
        (add_le_add h_estimation h_clipping)
  calc
    μ {ω | c.totalBound < |targetGap ω|} = μ E_bad := rfl
    _ ≤ μ (E_local ∪ (E_calibration ∪ (E_estimation ∪ E_clipping))) := h_bad_measure
    _ ≤ μ E_local + μ E_calibration + μ E_estimation + μ E_clipping := h_union
    _ ≤ δ_local + δ_calibration + δ_estimation + δ_clipping := h_components

/-- High-probability certificate discharged from the component-radius
provenance records. -/
theorem unified_learning_component_evidence_high_prob
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : Ω → ℝ)
    (e :
      UnifiedLearningComponentEvidence μ c targetGap oracleGap judgeGap estimateBeforeClip) :
    μ {ω | c.totalBound < |targetGap ω|} ≤ e.totalDelta := by
  exact unified_learning_certificate_high_prob
    (μ := μ)
    (c := c)
    (targetGap := targetGap)
    (oracleGap := oracleGap)
    (judgeGap := judgeGap)
    (estimateBeforeClip := estimateBeforeClip)
    (δ_local := e.localLaw.delta)
    (δ_calibration := e.calibration.delta)
    (δ_estimation := e.estimation.delta)
    (δ_clipping := e.clipping.delta)
    (h_local := by simpa using e.localLaw.event_bound)
    (h_calibration := by simpa using e.calibration.event_bound)
    (h_estimation := by simpa using e.estimation.event_bound)
    (h_clipping := by simpa using e.clipping.event_bound)

/-- Final high-probability paper certificate for the unified learning
procedure, with top-level sampling, honesty, manifest, and component-radius
provenance all bundled in the statement. -/
theorem unified_learning_final_paper_certificate_high_prob
    {Ω Case Truth Row βc βg βo Artifact γ ArtifactId : Type*}
    [MeasurableSpace Ω] [MeasurableSpace Case] [MeasurableSpace Truth]
    (ctx :
      UnifiedLearningPaperAssumptions
        (Ω := Ω) (Case := Case) (Truth := Truth) (Row := Row)
        (βc := βc) (βg := βg) (βo := βo)
        (Artifact := Artifact) (γ := γ) (ArtifactId := ArtifactId))
    (c : UnifiedLearningErrorCertificate)
    (targetGap oracleGap judgeGap estimateBeforeClip : Ω → ℝ)
    (e :
      UnifiedLearningComponentEvidence ctx.μ c targetGap oracleGap judgeGap estimateBeforeClip) :
    ctx.μ {ω | c.totalBound < |targetGap ω|} ≤ e.totalDelta :=
  unified_learning_component_evidence_high_prob
    ctx.μ c targetGap oracleGap judgeGap estimateBeforeClip e

end

end DSL
