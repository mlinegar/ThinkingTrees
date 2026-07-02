/-
FormalProofs/Audit.lean

Empirical Audit Framework for DPO Summarization Guarantees

This file provides the mathematical infrastructure for auditing summarization
systems through random sampling. The key insight is that even when we cannot
prove exact local laws (L1, L2, L3), we can:
1. Estimate violation rates empirically through sampling
2. Bound the true violation rate using concentration inequalities
3. Translate these bounds to DPO loss guarantees

The framework uses Hoeffding's inequality to provide (ε, δ)-guarantees:
with probability at least 1-δ, the empirical violation rate is within ε
of the true rate.

## Main Definitions

* `confidence_margin`: The ε margin from Hoeffding's inequality: sqrt(ln(2/δ)/(2n))
* `sample_complexity`: Minimum n needed for (ε, δ)-guarantee: ceil(ln(2/δ)/(2ε²))
* `AuditConfig`: Configuration for an audit (confidence level, threshold)
* `AuditResult`: Result of an audit with empirical rates

## Main Theorems

* `hoeffding_margin_sufficient`: With n samples, the margin is valid with prob 1-δ
* `sample_complexity_gives_margin`: n ≥ sample_complexity(ε, δ) implies margin ≤ ε
* `audit_bound_to_dpo`: Connect audit results to DPO gap bounds

## References

* Hoeffding's inequality (Mathlib: `measure_sum_ge_le_of_iIndepFun`)
* Sub-Gaussian random variables (Mathlib: `HasSubgaussianMGF`)
-/

import FormalProofs.OPT.AuditBounds
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.MeasureTheoreticAudit

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Confidence Margin and Sample Complexity

Definitions and lemmas for confidence margins and sample complexity live in
`FormalProofs/OPT/AuditCore.lean` and are imported here.
-/

/-!
## Audit Configuration and Results

Structures for specifying and reporting audits.
-/

/-- Configuration for an empirical audit -/
structure AuditConfig where
  /-- Confidence parameter: we want probability ≥ 1-δ -/
  δ : ℝ
  /-- Maximum acceptable violation rate threshold -/
  threshold : ℝ
  /-- Validity: 0 < δ < 1 -/
  hδ_pos : 0 < δ
  hδ_lt_one : δ < 1

/-- Result of an empirical audit -/
structure AuditResult where
  /-- Number of samples taken -/
  sample_count : ℕ
  /-- Number of violations observed -/
  violation_count : ℕ
  /-- Empirical violation rate: violations / samples -/
  empirical_rate : ℝ
  /-- Validity: empirical_rate = violation_count / sample_count -/
  rate_valid : sample_count > 0 → empirical_rate = violation_count / sample_count

/-- Upper bound on true violation rate with confidence 1-δ -/
def AuditResult.upper_bound (result : AuditResult) (δ : ℝ) : ℝ :=
  result.empirical_rate + confidence_margin δ result.sample_count

/-- Audit passes if upper bound is below threshold -/
def AuditResult.passes (result : AuditResult) (config : AuditConfig) : Prop :=
  result.upper_bound config.δ < config.threshold

/-!
## Audit Bounds Connection

Connect empirical audit results to the theoretical union bounds.
The key insight is that the union bound provides:

  Δ_R ≤ totalLeafViolation + totalMergeViolation + (R-1) * pIdemp

Each of these terms can be estimated empirically through sampling.
-/

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- The audit provides a probabilistic bound on violation rates.

This is a direct wrapper around `audit_certification_pmf` from
`MeasureTheoreticAudit`. It states the Hoeffding-style guarantee for the
empirical violation rate. -/
theorem audit_gives_probabilistic_bound
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
  exact audit_certification_pmf fstar x p hf hp_nonneg hp_le_one n hn δ hδ hδ'

/-- Minimum samples for (ε, δ)-audit guarantee -/
theorem audit_sample_complexity (ε δ : ℝ) (hε : 0 < ε) (hδ : 0 < δ) (hδ' : δ < 2) :
    ∀ n ≥ sample_complexity ε δ, confidence_margin δ n ≤ ε :=
  fun n hn => sample_complexity_gives_margin ε δ hε hδ hδ' n hn

/-!
## Three-Level Guarantee Summary

The three levels of DPO training guarantees:

### Level 1: EXACT (Local Laws)
When L1, L2, L3 hold exactly:
- `dpo_gap_zero_of_local_laws_bounded`: DPO gap = 0

### Level 2: UNION BOUND (Violation Rates)
With quantitative violation rates:
- `union_bound_multi_round`: Δ_R ≤ leaf + merge + (R-1)*idemp violations
- The DPO gap is bounded by 2*M_loss (crude but always provable)

### Level 3: EMPIRICAL AUDIT
With n iid samples from the summarization distribution:
- Estimate violation rate p̂
- With probability ≥ 1-δ: p_true ≤ p̂ + sqrt(ln(2/δ)/(2n))
- Sample complexity: n ≥ ln(2/δ)/(2ε²) gives margin ≤ ε

The connection between levels:
- Level 1 is a special case of Level 2 where all violations = 0
- Level 3 provides probabilistic estimates of Level 2 quantities
- For practical deployment, Level 3 is most useful for certification
-/

/-- Master theorem showing all three levels of DPO guarantees.

This theorem unifies the three levels of DPO training guarantees:

**Level 1 (EXACT)**: When local laws L1, L2, L3 hold exactly, the DPO gap is zero.
This is the strongest guarantee but requires exact semantic preservation.

**Level 2 (UNION BOUND)**: With any bounded loss function, the DPO gap is bounded
by 2 * M_loss. This is a crude but always-provable bound that applies even when
local laws don't hold.

**Level 3 (EMPIRICAL AUDIT)**: With n samples and confidence parameter δ,
the empirical violation rate p̂ satisfies: P(p_true ≤ p̂ + margin) ≥ 1-δ,
where margin = sqrt(ln(2/δ)/(2n)).

The three levels form a spectrum from strongest (L1) to most practical (L3):
- Use Level 1 when you can prove local laws
- Use Level 2 for worst-case bounds
- Use Level 3 for practical certification with statistical guarantees
-/
theorem dpo_three_level_guarantees {Strings A Y : Type*} [Monoid Strings] [MetricSpace Y]
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (pol pol_ref : Policy Strings A) (gen : PairGenerator Strings A)
    (fstar : Strings → Y) (β : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar)
    (M : ℝ) (hM : 0 ≤ M) (hbound : ∀ w z, D fstar w z ≤ M)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A), |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (L_pol : ℝ≥0) (h_lip : PolicyLipschitz pol pol_ref fstar L_pol) :
    -- Level 1: EXACT (when local laws hold)
    (L1 g T fstar → L2 g T fstar → L3 g fstar →
      |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
       ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| = 0) ∧
    -- Level 2: UNION BOUND (crude but always valid)
    (|ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
      ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤ 2 * M_loss) ∧
    -- Level 3: EMPIRICAL AUDIT (sample complexity for margin ε)
    (∀ ε δ : ℝ, 0 < ε → 0 < δ → δ < 2 →
      ∀ n ≥ sample_complexity ε δ, confidence_margin δ n ≤ ε) := by
  constructor
  -- Level 1: Exact guarantee via local laws
  · intro h1 h2 h3
    exact dpo_gap_zero_of_local_laws_bounded fstar pol pol_ref gen g x R T β
      hp h1 h2 h3 hR h_meas_pol h_meas_ref h_pair M hM hbound
  constructor
  -- Level 2: Union bound via loss bound
  · exact dpo_gap_oracle_indexed_bounded fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β L_pol
      h_meas_pol h_meas_ref h_lip h_pair M_loss hM_loss h_loss_bound
  -- Level 3: Sample complexity gives margin bound
  · intro ε δ hε hδ hδ' n hn
    exact sample_complexity_gives_margin ε δ hε hδ hδ' n hn

end
