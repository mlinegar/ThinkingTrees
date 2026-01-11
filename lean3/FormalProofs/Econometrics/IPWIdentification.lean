import FormalProofs.Econometrics.Assumptions
import FormalProofs.Econometrics.PropensityScore
import Mathlib.MeasureTheory.Function.ConditionalExpectation.PullOut

/-!
# FormalProofs/Econometrics/IPWIdentification.lean

Identification of causal estimands via IPW.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped BigOperators
open scoped MeasureTheory
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Econometrics

/-!
## IPW Functional
-/

/-- IPW functional using a propensity score random variable e(ω).

This is the population analog of the sample IPW estimator:
  E[ D * Y / e - (1 - D) * Y / (1 - e) ].
-/
def ipwFunctionalRV {Ω : Type*} [MeasurableSpace Ω]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (D : Ω → Treatment) (Y : Ω → ℝ) (e : Ω → ℝ) : ℝ :=
  ProbabilityTheory.expectation mu (fun ω =>
    treatmentIndicator D ω * Y ω / e ω
      - (1 - treatmentIndicator D ω) * Y ω / (1 - e ω))

/-- IPW functional when the propensity score is represented as e(X). -/
def ipwFunctional {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (X : Ω → α) (D : Ω → Treatment) (Y : Ω → ℝ)
    (ps : PropensityScore mu X D) : ℝ :=
  ipwFunctionalRV mu D Y ps.value

/-!
## Assumption Bundle
-/

/-- Canonical assumption bundle for IPW identification. -/
structure IPWAssumptions {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (X : Ω → α) (D : Ω → Treatment) (Y : Ω → ℝ)
    (po : PotentialOutcomes Ω) where
  propensity : PropensityScore mu X D
  consistency : Consistency D po Y
  unconfoundedness : Unconfoundedness mu X D po
  positivity : Positivity mu X propensity.e
  measurableX : Measurable X
  measurableD : Measurable D

/-!
## Integrability Conditions
-/

/-- Integrability conditions needed for the IPW identification proof. -/
structure IPWIntegrability {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (X : Ω → α) (D : Ω → Treatment) (po : PotentialOutcomes Ω)
    (ps : PropensityScore mu X D) : Prop where
  dy1 : Integrable (fun ω => treatmentIndicator D ω * po.y1 ω) mu
  dy0 : Integrable (fun ω => (1 - treatmentIndicator D ω) * po.y0 ω) mu
  ipw_y1 : Integrable (fun ω => treatmentIndicator D ω * po.y1 ω / ps.value ω) mu
  ipw_y0 : Integrable (fun ω => (1 - treatmentIndicator D ω) * po.y0 ω / (1 - ps.value ω)) mu

/-!
## Identification Statement (as a predicate)
-/

/-- IPW identification of the ATE as a formal predicate.

Under consistency, unconfoundedness, and positivity, this predicate should hold. -/
def IPWIdentifiesATE {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (X : Ω → α) (D : Ω → Treatment) (Y : Ω → ℝ)
    (po : PotentialOutcomes Ω) (ps : PropensityScore mu X D) : Prop :=
  ATE mu po = ipwFunctional mu X D Y ps

/-- Identification predicate using the bundled assumptions. -/
def IPWIdentifiesATE_fromAssumptions {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (X : Ω → α) (D : Ω → Treatment) (Y : Ω → ℝ)
    (po : PotentialOutcomes Ω) (assm : IPWAssumptions mu X D Y po) : Prop :=
  IPWIdentifiesATE mu X D Y po assm.propensity

/-!
## Reusable Lemmas for IPW

These helper results capture the small, repeated analytic steps that appear
throughout the identification proof.
-/

lemma treatmentIndicator_norm_le_one {Ω : Type*} (D : Ω → Treatment) :
    ∀ ω, ‖treatmentIndicator D ω‖ ≤ (1 : ℝ) := by
  intro ω
  cases h : D ω <;> simp [treatmentIndicator, Treatment.toReal, h]

lemma one_minus_treatmentIndicator_norm_le_one {Ω : Type*} (D : Ω → Treatment) :
    ∀ ω, ‖1 - treatmentIndicator D ω‖ ≤ (1 : ℝ) := by
  intro ω
  cases h : D ω <;> simp [treatmentIndicator, Treatment.toReal, h]

lemma ae_inv_mul_cancel_right {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (a f g : Ω → ℝ) (hfg : f =ᵐ[μ] g) (hne : ∀ᵐ ω ∂μ, g ω ≠ 0) :
    (fun ω => (1 / f ω) * (a ω * g ω)) =ᵐ[μ] a := by
  filter_upwards [hfg, hne] with ω hfgω hneω
  have hne' : g ω ≠ 0 := hneω
  calc
    (1 / f ω) * (a ω * g ω) = (g ω)⁻¹ * (a ω * g ω) := by
      simp [hfgω]
    _ = a ω * ((g ω)⁻¹ * g ω) := by
      simp [mul_comm, mul_left_comm, mul_assoc]
    _ = a ω := by
      simp [hne', mul_assoc]

/-!
## Identification Theorem
-/

private lemma sigmaXD_le {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    {X : Ω → α} {D : Ω → Treatment}
    (hX : Measurable X) (hD : Measurable D) :
    sigmaXD X D ≤ (inferInstance : MeasurableSpace Ω) := by
  refine sup_le ?_ ?_
  · exact hX.comap_le
  · exact hD.comap_le

/-- **IPW Identification (population)**.

Under consistency, unconfoundedness, and integrability (plus measurability and sigma-finiteness
for conditional expectations), the ATE equals the IPW functional. -/
theorem ipw_identifies_ATE
    {Ω α : Type*} [MeasurableSpace Ω] [MeasurableSpace α]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (X : Ω → α) (D : Ω → Treatment) (Y : Ω → ℝ)
    (po : PotentialOutcomes Ω)
    (assm : IPWAssumptions mu X D Y po)
    [SigmaFinite (mu.trim assm.measurableX.comap_le)]
    [SigmaFinite (mu.trim (sigmaXD_le assm.measurableX assm.measurableD))]
    (hint : IPWIntegrability mu X D po assm.propensity) :
    IPWIdentifiesATE mu X D Y po assm.propensity := by
  classical
  -- Shorthand notation
  let d : Ω → ℝ := treatmentIndicator D
  let e : Ω → ℝ := assm.propensity.value
  have hmXD : sigmaXD X D ≤ (inferInstance : MeasurableSpace Ω) :=
    sigmaXD_le assm.measurableX assm.measurableD
  have hmX : sigmaX X ≤ (inferInstance : MeasurableSpace Ω) :=
    assm.measurableX.comap_le
  /-
  ### Step 0: Measurability and Integrability Primitives

  We set up the sigma-algebra relationships and the basic integrability facts
  needed to apply conditional expectation lemmas.
  -/
  -- Measurability helpers
  have hD_mD : Measurable[sigmaD D] D := by
    simpa [sigmaD] using (comap_measurable (f := D))
  have hD_mXD : Measurable[sigmaXD X D] D :=
    hD_mD.mono (by exact le_sup_right) le_rfl
  have hd_mXD : Measurable[sigmaXD X D] d := by
    have h_toReal : Measurable (Treatment.toReal) := by
      simpa using (measurable_of_countable (f := Treatment.toReal))
    simpa [d] using h_toReal.comp hD_mXD
  have hd_mΩ : Measurable d :=
    hd_mXD.mono hmXD le_rfl
  have he_mX : Measurable[sigmaX X] e := assm.propensity.value_measurable
  have h_inv_e_mX : StronglyMeasurable[sigmaX X] (fun ω => 1 / e ω) := by
    exact (measurable_const.div he_mX).stronglyMeasurable
  have h_inv_one_minus_e_mX : StronglyMeasurable[sigmaX X] (fun ω => 1 / (1 - e ω)) := by
    have h_one_minus : Measurable[sigmaX X] (fun ω => 1 - e ω) := by
      simpa using (measurable_const.sub he_mX)
    exact (measurable_const.div h_one_minus).stronglyMeasurable
  /-
  ### Step 1: Integrability and 0/1 Bounds for the Treatment Indicator
  -/
  have hY1_int : Integrable po.y1 mu := assm.unconfoundedness.integrable.y1_integrable
  have hY0_int : Integrable po.y0 mu := assm.unconfoundedness.integrable.y0_integrable
  have hDY1_int : Integrable (fun ω => d ω * po.y1 ω) mu := hint.dy1
  have hDY0_int : Integrable (fun ω => (1 - d ω) * po.y0 ω) mu := hint.dy0
  have hIPW1_int : Integrable (fun ω => d ω * po.y1 ω / e ω) mu := hint.ipw_y1
  have hIPW0_int : Integrable (fun ω => (1 - d ω) * po.y0 ω / (1 - e ω)) mu := hint.ipw_y0
  have h_bdd : ∀ᵐ ω ∂mu, ‖d ω‖ ≤ (1 : ℝ) := by
    simpa [d] using (Filter.Eventually.of_forall (treatmentIndicator_norm_le_one D))
  have hd_int : Integrable d mu := by
    have h_d_meas : AEStronglyMeasurable d mu := hd_mΩ.aestronglyMeasurable
    have h_bdd' : ∀ᵐ ω ∂mu, ‖d ω‖ ≤ ‖(1 : ℝ)‖ := by
      simpa using h_bdd
    exact Integrable.mono (integrable_const (1 : ℝ)) h_d_meas h_bdd'
  have h_one_minus_d_int : Integrable (fun ω => 1 - d ω) mu := by
    simpa using (integrable_const (1 : ℝ)).sub hd_int
  /-
  ### Step 2: Positivity and the Propensity Score RV
  -/
  have h_pos : ∀ᵐ ω ∂mu, 0 < e ω ∧ e ω < 1 := by
    simpa [Positivity, PropensityScore.value, e] using assm.positivity
  have h_ps_ae : e =ᵐ[mu] propensityScoreRV mu X D := by
    simpa [PropensityScore.value, e] using assm.propensity.ae_eq
  have h_ne : ∀ᵐ ω ∂mu, e ω ≠ 0 := by
    exact h_pos.mono (fun _ h => ne_of_gt h.1)
  have h_one_minus_ne : ∀ᵐ ω ∂mu, 1 - e ω ≠ 0 := by
    exact h_pos.mono (fun _ h => by linarith)
  /-
  ### Step 3: Consistency (Observed Outcome Substitution)

  Replace the observed outcome with the potential outcomes and use the
  fact that D is 0/1 to remove cross-terms.
  -/
  have h_ipw_simplify :
      (fun ω => d ω * Y ω / e ω - (1 - d ω) * Y ω / (1 - e ω))
        = fun ω => d ω * po.y1 ω / e ω - (1 - d ω) * po.y0 ω / (1 - e ω) := by
    funext ω
    have hY : Y ω = observedOutcome D po ω := by
      simpa using congrArg (fun f => f ω) assm.consistency.obs_eq
    have hY' : Y ω = (if D ω then po.y1 ω else po.y0 ω) := by
      simpa [observedOutcome_eq_by_cases] using hY
    cases h : D ω <;> simp [d, treatmentIndicator, Treatment.toReal, h, hY']
  /-
  ### Step 4: ATE as an Expectation Difference
  -/
  have h_ate :
      ATE mu po =
        ProbabilityTheory.expectation mu po.y1
        - ProbabilityTheory.expectation mu po.y0 := by
    simp [ATE, treatmentEffect, ProbabilityTheory.expectation, integral_sub hY1_int hY0_int]
  /-
  ### Step 5: IPW Recovers E[Y1]

  This is the standard IPW identity:
    E[D * Y1 / e(X)] = E[Y1].

  We use the pull-out property of conditional expectation, the tower property,
  unconfoundedness, and then cancel the propensity score.
  -/
  have h_ipw_y1 :
      ProbabilityTheory.expectation mu (fun ω => d ω * po.y1 ω / e ω) =
        ProbabilityTheory.expectation mu po.y1 := by
    have h_condexp_pull :
        mu[fun ω => (1 / e ω) * (d ω * po.y1 ω) | sigmaX X]
          =ᵐ[mu] fun ω => (1 / e ω) * mu[fun ω => d ω * po.y1 ω | sigmaX X] ω := by
      have h_mul : Integrable (fun ω => (1 / e ω) * (d ω * po.y1 ω)) mu := by
        simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hIPW1_int
      have h :=
        condExp_mul_of_stronglyMeasurable_left (μ := mu) (m := sigmaX X)
          (f := fun ω => 1 / e ω) (g := fun ω => d ω * po.y1 ω) h_inv_e_mX h_mul hDY1_int
      simpa using h
    have h_pull_d :
        mu[fun ω => d ω * po.y1 ω | sigmaXD X D]
          =ᵐ[mu] fun ω => d ω * mu[po.y1 | sigmaXD X D] ω := by
      have h :=
        condExp_mul_of_stronglyMeasurable_left (μ := mu) (m := sigmaXD X D)
          (f := d) (g := po.y1) hd_mXD.stronglyMeasurable hDY1_int hY1_int
      simpa using h
    have h_y1_mul :
        (fun ω => d ω * mu[po.y1 | sigmaXD X D] ω)
          =ᵐ[mu] fun ω => d ω * mu[po.y1 | sigmaX X] ω := by
      exact Filter.EventuallyEq.fun_mul (Filter.EventuallyEq.rfl)
        assm.unconfoundedness.y1_condexp_eq
    have h_step :
        mu[mu[fun ω => d ω * po.y1 ω | sigmaXD X D] | sigmaX X]
          =ᵐ[mu] mu[fun ω => d ω * mu[po.y1 | sigmaX X] ω | sigmaX X] := by
      exact condExp_congr_ae (h_pull_d.trans h_y1_mul)
    have h_tower :
        mu[mu[fun ω => d ω * po.y1 ω | sigmaXD X D] | sigmaX X]
          =ᵐ[mu] mu[fun ω => d ω * po.y1 ω | sigmaX X] := by
      simpa using (condExp_condExp_of_le (μ := mu)
        (f := fun ω => d ω * po.y1 ω) (m₁ := sigmaX X) (m₂ := sigmaXD X D)
        (m₀ := inferInstance) le_sup_left hmXD)
    have hy1_mX : StronglyMeasurable[sigmaX X] (fun ω => mu[po.y1 | sigmaX X] ω) := by
      exact (stronglyMeasurable_condExp (μ := mu) (m := sigmaX X) (f := po.y1))
    have hy1_int : Integrable (fun ω => mu[po.y1 | sigmaX X] ω) mu := by
      simpa using (integrable_condExp (μ := mu) (m := sigmaX X) (f := po.y1))
    have h_dg_int : Integrable (fun ω => d ω * mu[po.y1 | sigmaX X] ω) mu := by
      have h_d_meas : AEStronglyMeasurable d mu := hd_mΩ.aestronglyMeasurable
      exact Integrable.bdd_mul (hg := hy1_int) h_d_meas (by simpa using h_bdd)
    have h_pull_y1 :
        mu[fun ω => d ω * mu[po.y1 | sigmaX X] ω | sigmaX X]
          =ᵐ[mu] fun ω => mu[d | sigmaX X] ω * mu[po.y1 | sigmaX X] ω := by
      exact condExp_mul_of_stronglyMeasurable_right (μ := mu) (m := sigmaX X)
        hy1_mX h_dg_int hd_int
    have h_condexp_dy1 :
        mu[fun ω => d ω * po.y1 ω | sigmaX X]
          =ᵐ[mu] fun ω => mu[po.y1 | sigmaX X] ω * propensityScoreRV mu X D ω := by
      have h1 :
          mu[fun ω => d ω * po.y1 ω | sigmaX X]
            =ᵐ[mu] mu[mu[fun ω => d ω * po.y1 ω | sigmaXD X D] | sigmaX X] := by
        exact h_tower.symm
      have h2 :
          mu[mu[fun ω => d ω * po.y1 ω | sigmaXD X D] | sigmaX X]
            =ᵐ[mu] mu[fun ω => d ω * mu[po.y1 | sigmaX X] ω | sigmaX X] := h_step
      have h3 :
          mu[fun ω => d ω * mu[po.y1 | sigmaX X] ω | sigmaX X]
            =ᵐ[mu] fun ω => mu[d | sigmaX X] ω * mu[po.y1 | sigmaX X] ω := h_pull_y1
      have h4 :
          (fun ω => mu[d | sigmaX X] ω * mu[po.y1 | sigmaX X] ω)
            =ᵐ[mu] fun ω => mu[po.y1 | sigmaX X] ω * propensityScoreRV mu X D ω := by
        refine Filter.Eventually.of_forall ?_
        intro ω
        simp [propensityScoreRV, d, mul_comm, mul_left_comm, mul_assoc]
      exact h1.trans (h2.trans (h3.trans h4))
    have h_tmp :
        mu[fun ω => d ω * po.y1 ω / e ω | sigmaX X]
          =ᵐ[mu] fun ω => (1 / e ω) * (mu[po.y1 | sigmaX X] ω * propensityScoreRV mu X D ω) := by
      have h_condexp_pull' :
          mu[fun ω => d ω * po.y1 ω / e ω | sigmaX X]
            =ᵐ[mu] fun ω => (1 / e ω) * mu[fun ω => d ω * po.y1 ω | sigmaX X] ω := by
        simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using h_condexp_pull
      refine h_condexp_pull'.trans ?_
      exact Filter.EventuallyEq.mul (Filter.EventuallyEq.rfl) h_condexp_dy1
    have h_cancel :
        (fun ω => (1 / e ω) * (mu[po.y1 | sigmaX X] ω * propensityScoreRV mu X D ω))
          =ᵐ[mu] fun ω => mu[po.y1 | sigmaX X] ω := by
      have h_ps_ne : ∀ᵐ ω ∂mu, propensityScoreRV mu X D ω ≠ 0 := by
        filter_upwards [h_ps_ae, h_ne] with ω hω hne
        simpa [hω] using hne
      exact ae_inv_mul_cancel_right (μ := mu)
        (a := fun ω => mu[po.y1 | sigmaX X] ω)
        (f := e) (g := propensityScoreRV mu X D) h_ps_ae h_ps_ne
    have h_condexp_eq :
        mu[fun ω => d ω * po.y1 ω / e ω | sigmaX X]
          =ᵐ[mu] fun ω => mu[po.y1 | sigmaX X] ω := h_tmp.trans h_cancel
    have h_eq_int :
        ∫ ω, (mu[fun ω => d ω * po.y1 ω / e ω | sigmaX X]) ω ∂mu =
          ∫ ω, (mu[po.y1 | sigmaX X]) ω ∂mu := by
      exact integral_congr_ae h_condexp_eq
    calc
      ProbabilityTheory.expectation mu (fun ω => d ω * po.y1 ω / e ω)
          = ∫ ω, (mu[fun ω => d ω * po.y1 ω / e ω | sigmaX X]) ω ∂mu := by
              simp [ProbabilityTheory.expectation, integral_condExp (m := sigmaX X) (μ := mu) hmX]
      _ = ∫ ω, (mu[po.y1 | sigmaX X]) ω ∂mu := h_eq_int
      _ = ProbabilityTheory.expectation mu po.y1 := by
          simp [ProbabilityTheory.expectation, integral_condExp (m := sigmaX X) (μ := mu) hmX]
  /-
  ### Step 6: IPW Recovers E[Y0]

  The proof mirrors the previous step, with (1 - D) and (1 - e(X)).
  -/
  have h_ipw_y0 :
      ProbabilityTheory.expectation mu (fun ω => (1 - d ω) * po.y0 ω / (1 - e ω)) =
        ProbabilityTheory.expectation mu po.y0 := by
    have h_condexp_pull :
        mu[fun ω => (1 / (1 - e ω)) * ((1 - d ω) * po.y0 ω) | sigmaX X]
          =ᵐ[mu] fun ω => (1 / (1 - e ω)) * mu[fun ω => (1 - d ω) * po.y0 ω | sigmaX X] ω := by
      have h_mul :
          Integrable (fun ω => (1 / (1 - e ω)) * ((1 - d ω) * po.y0 ω)) mu := by
        simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hIPW0_int
      have h :=
        condExp_mul_of_stronglyMeasurable_left (μ := mu) (m := sigmaX X)
          (f := fun ω => 1 / (1 - e ω)) (g := fun ω => (1 - d ω) * po.y0 ω)
          h_inv_one_minus_e_mX h_mul hDY0_int
      simpa using h
    have h_pull_d :
        mu[fun ω => (1 - d ω) * po.y0 ω | sigmaXD X D]
          =ᵐ[mu] fun ω => (1 - d ω) * mu[po.y0 | sigmaXD X D] ω := by
      have h :=
        condExp_mul_of_stronglyMeasurable_left (μ := mu) (m := sigmaXD X D)
          (f := fun ω => 1 - d ω) (g := po.y0)
          (measurable_const.sub hd_mXD).stronglyMeasurable hDY0_int hY0_int
      simpa using h
    have h_y0_mul :
        (fun ω => (1 - d ω) * mu[po.y0 | sigmaXD X D] ω)
          =ᵐ[mu] fun ω => (1 - d ω) * mu[po.y0 | sigmaX X] ω := by
      exact Filter.EventuallyEq.fun_mul (Filter.EventuallyEq.rfl)
        assm.unconfoundedness.y0_condexp_eq
    have h_step :
        mu[mu[fun ω => (1 - d ω) * po.y0 ω | sigmaXD X D] | sigmaX X]
          =ᵐ[mu] mu[fun ω => (1 - d ω) * mu[po.y0 | sigmaX X] ω | sigmaX X] := by
      exact condExp_congr_ae (h_pull_d.trans h_y0_mul)
    have h_tower :
        mu[mu[fun ω => (1 - d ω) * po.y0 ω | sigmaXD X D] | sigmaX X]
          =ᵐ[mu] mu[fun ω => (1 - d ω) * po.y0 ω | sigmaX X] := by
      simpa using (condExp_condExp_of_le (μ := mu)
        (f := fun ω => (1 - d ω) * po.y0 ω) (m₁ := sigmaX X) (m₂ := sigmaXD X D)
        (m₀ := inferInstance) le_sup_left hmXD)
    have hy0_mX : StronglyMeasurable[sigmaX X] (fun ω => mu[po.y0 | sigmaX X] ω) := by
      exact (stronglyMeasurable_condExp (μ := mu) (m := sigmaX X) (f := po.y0))
    have hy0_int : Integrable (fun ω => mu[po.y0 | sigmaX X] ω) mu := by
      simpa using (integrable_condExp (μ := mu) (m := sigmaX X) (f := po.y0))
    have h_dg_int : Integrable (fun ω => (1 - d ω) * mu[po.y0 | sigmaX X] ω) mu := by
      have h_d_meas : AEStronglyMeasurable (fun ω => 1 - d ω) mu := by
        exact (measurable_const.sub hd_mΩ).aestronglyMeasurable
      have h_bdd' : ∀ᵐ ω ∂mu, ‖1 - d ω‖ ≤ (1 : ℝ) := by
        simpa [d] using (Filter.Eventually.of_forall (one_minus_treatmentIndicator_norm_le_one D))
      exact Integrable.bdd_mul (hg := hy0_int) h_d_meas (by simpa using h_bdd')
    have h_pull_y0 :
        mu[fun ω => (1 - d ω) * mu[po.y0 | sigmaX X] ω | sigmaX X]
          =ᵐ[mu] fun ω => mu[fun ω => 1 - d ω | sigmaX X] ω * mu[po.y0 | sigmaX X] ω := by
      exact condExp_mul_of_stronglyMeasurable_right (μ := mu) (m := sigmaX X)
        hy0_mX h_dg_int h_one_minus_d_int
    have h_condexp_one_minus_d :
        mu[fun ω => 1 - d ω | sigmaX X] =ᵐ[mu] fun ω => 1 - mu[d | sigmaX X] ω := by
      simpa [condExp_const (μ := mu) (m := sigmaX X) hmX (1 : ℝ)] using
        (condExp_sub (μ := mu) (m := sigmaX X)
          (f := fun _ : Ω => (1 : ℝ)) (g := d) (integrable_const (1 : ℝ)) hd_int)
    have h_condexp_dy0 :
        mu[fun ω => (1 - d ω) * po.y0 ω | sigmaX X]
          =ᵐ[mu] fun ω => mu[po.y0 | sigmaX X] ω * (1 - propensityScoreRV mu X D ω) := by
      have h1 :
          mu[fun ω => (1 - d ω) * po.y0 ω | sigmaX X]
            =ᵐ[mu] mu[mu[fun ω => (1 - d ω) * po.y0 ω | sigmaXD X D] | sigmaX X] := by
        exact h_tower.symm
      have h2 :
          mu[mu[fun ω => (1 - d ω) * po.y0 ω | sigmaXD X D] | sigmaX X]
            =ᵐ[mu] mu[fun ω => (1 - d ω) * mu[po.y0 | sigmaX X] ω | sigmaX X] := h_step
      have h3 :
          mu[fun ω => (1 - d ω) * mu[po.y0 | sigmaX X] ω | sigmaX X]
            =ᵐ[mu] fun ω => mu[fun ω => 1 - d ω | sigmaX X] ω * mu[po.y0 | sigmaX X] ω := h_pull_y0
      have h4 :
          (fun ω => mu[fun ω => 1 - d ω | sigmaX X] ω * mu[po.y0 | sigmaX X] ω)
            =ᵐ[mu] fun ω => mu[po.y0 | sigmaX X] ω * (1 - propensityScoreRV mu X D ω) := by
        refine (Filter.EventuallyEq.mul h_condexp_one_minus_d (Filter.EventuallyEq.rfl)).trans ?_
        refine Filter.Eventually.of_forall ?_
        intro ω
        have h_prop : propensityScoreRV mu X D ω = mu[d | sigmaX X] ω := by
          simp [propensityScoreRV, d]
        calc
          (1 - mu[d | sigmaX X] ω) * mu[po.y0 | sigmaX X] ω
              = mu[po.y0 | sigmaX X] ω * (1 - mu[d | sigmaX X] ω) := by
                  exact mul_comm _ _
          _ = mu[po.y0 | sigmaX X] ω * (1 - propensityScoreRV mu X D ω) := by
                rw [h_prop]
      exact h1.trans (h2.trans (h3.trans h4))
    have h_tmp :
        mu[fun ω => (1 - d ω) * po.y0 ω / (1 - e ω) | sigmaX X]
          =ᵐ[mu] fun ω => (1 / (1 - e ω)) * (mu[po.y0 | sigmaX X] ω * (1 - propensityScoreRV mu X D ω)) := by
      have h_condexp_pull' :
          mu[fun ω => (1 - d ω) * po.y0 ω / (1 - e ω) | sigmaX X]
            =ᵐ[mu] fun ω => (1 / (1 - e ω)) * mu[fun ω => (1 - d ω) * po.y0 ω | sigmaX X] ω := by
        simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using h_condexp_pull
      refine h_condexp_pull'.trans ?_
      exact Filter.EventuallyEq.mul (Filter.EventuallyEq.rfl) h_condexp_dy0
    have h_cancel :
        (fun ω =>
          (1 / (1 - e ω)) * (mu[po.y0 | sigmaX X] ω * (1 - propensityScoreRV mu X D ω)))
          =ᵐ[mu] fun ω => mu[po.y0 | sigmaX X] ω := by
      have h_ps_sub : (fun ω => 1 - e ω) =ᵐ[mu] fun ω => 1 - propensityScoreRV mu X D ω := by
        filter_upwards [h_ps_ae] with ω hω
        simp [hω]
      have h_ps_sub_ne : ∀ᵐ ω ∂mu, 1 - propensityScoreRV mu X D ω ≠ 0 := by
        filter_upwards [h_ps_sub, h_one_minus_ne] with ω hω hne
        simpa [hω] using hne
      exact ae_inv_mul_cancel_right (μ := mu)
        (a := fun ω => mu[po.y0 | sigmaX X] ω)
        (f := fun ω => 1 - e ω) (g := fun ω => 1 - propensityScoreRV mu X D ω)
        h_ps_sub h_ps_sub_ne
    have h_condexp_eq :
        mu[fun ω => (1 - d ω) * po.y0 ω / (1 - e ω) | sigmaX X]
          =ᵐ[mu] fun ω => mu[po.y0 | sigmaX X] ω := h_tmp.trans h_cancel
    have h_eq_int :
        ∫ ω, (mu[fun ω => (1 - d ω) * po.y0 ω / (1 - e ω) | sigmaX X]) ω ∂mu =
          ∫ ω, (mu[po.y0 | sigmaX X]) ω ∂mu := by
      exact integral_congr_ae h_condexp_eq
    calc
      ProbabilityTheory.expectation mu (fun ω => (1 - d ω) * po.y0 ω / (1 - e ω))
          = ∫ ω, (mu[fun ω => (1 - d ω) * po.y0 ω / (1 - e ω) | sigmaX X]) ω ∂mu := by
              simp [ProbabilityTheory.expectation, integral_condExp (m := sigmaX X) (μ := mu) hmX]
      _ = ∫ ω, (mu[po.y0 | sigmaX X]) ω ∂mu := h_eq_int
      _ = ProbabilityTheory.expectation mu po.y0 := by
          simp [ProbabilityTheory.expectation, integral_condExp (m := sigmaX X) (μ := mu) hmX]
  /-
  ### Step 7: Combine to Identify the ATE
  -/
  have h_ipw :
      ipwFunctional mu X D Y assm.propensity =
        ProbabilityTheory.expectation mu (fun ω => d ω * po.y1 ω / e ω) -
        ProbabilityTheory.expectation mu (fun ω => (1 - d ω) * po.y0 ω / (1 - e ω)) := by
    unfold ipwFunctional ipwFunctionalRV
    have h_ipw_simplify' :
        (fun ω => treatmentIndicator D ω * Y ω / e ω -
          (1 - treatmentIndicator D ω) * Y ω / (1 - e ω)) =
          fun ω => treatmentIndicator D ω * po.y1 ω / e ω -
            (1 - treatmentIndicator D ω) * po.y0 ω / (1 - e ω) := by
      simpa [d] using h_ipw_simplify
    have hIPW1_int' : Integrable (fun ω => treatmentIndicator D ω * po.y1 ω / e ω) mu := by
      simpa [d] using hIPW1_int
    have hIPW0_int' :
        Integrable (fun ω => (1 - treatmentIndicator D ω) * po.y0 ω / (1 - e ω)) mu := by
      simpa [d] using hIPW0_int
    have h_ipw' :
        ProbabilityTheory.expectation mu (fun ω => treatmentIndicator D ω * Y ω / e ω -
          (1 - treatmentIndicator D ω) * Y ω / (1 - e ω)) =
          ProbabilityTheory.expectation mu (fun ω => treatmentIndicator D ω * po.y1 ω / e ω) -
            ProbabilityTheory.expectation mu
              (fun ω => (1 - treatmentIndicator D ω) * po.y0 ω / (1 - e ω)) := by
      calc
        ProbabilityTheory.expectation mu (fun ω => treatmentIndicator D ω * Y ω / e ω -
          (1 - treatmentIndicator D ω) * Y ω / (1 - e ω))
            =
            ProbabilityTheory.expectation mu
              (fun ω => treatmentIndicator D ω * po.y1 ω / e ω -
                (1 - treatmentIndicator D ω) * po.y0 ω / (1 - e ω)) := by
                  simp [ProbabilityTheory.expectation, h_ipw_simplify']
        _ =
            ProbabilityTheory.expectation mu (fun ω => treatmentIndicator D ω * po.y1 ω / e ω) -
              ProbabilityTheory.expectation mu
                (fun ω => (1 - treatmentIndicator D ω) * po.y0 ω / (1 - e ω)) := by
                  simp [ProbabilityTheory.expectation, integral_sub hIPW1_int' hIPW0_int']
    simpa [d, e] using h_ipw'
  -- Final assembly
  calc
    ATE mu po
        = ProbabilityTheory.expectation mu po.y1
          - ProbabilityTheory.expectation mu po.y0 := h_ate
    _ = ipwFunctional mu X D Y assm.propensity := by
      simp [h_ipw, h_ipw_y1.symm, h_ipw_y0.symm]

end Econometrics
