import FormalProofs.DSL.SamplingTheory

/-!
# FormalProofs/DSL/IPWMeasurementError.lean

Measurement-error view of inverse-probability weighting.

The key point is simple:

- if the actual sampling design uses inclusion probability `π_true`, but
- the analyst weights by `1 / π_used`,

then the Horvitz-Thompson cancellation becomes

`E[R / π_used] = π_true / π_used`,

not `1`.

So ignoring a nonclassical / proxy-aware sampling mechanism in the IPW component
creates a multiplicative bias factor. Correcting the IPW propensities to match
the actual sampling law restores exact cancellation.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-- Horvitz-Thompson style weight built from a user-supplied propensity,
regardless of the true sampling design. -/
def htWeightWithPropensity {Obs Con : Type*}
    (pi_used : SamplingProbability Obs Con)
    (R : SamplingIndicator) (d_obs : Obs) (q : Con) : ℝ :=
  if R then 1 / pi_used d_obs q else 0

/-- Under the true design, using a misspecified propensity `pi_used` produces
the exact multiplicative bias factor `π_true / π_used`. -/
theorem misspecified_ht_weight_expectation_eq_ratio
    {Obs Mis Con : Type*}
    (dbs_true : DesignBasedSampling Obs Mis Con)
    (pi_used : ValidSamplingProbability Obs Con)
    (d_obs : Obs) (q : Con) :
    dbs_true.prob d_obs q *
        htWeightWithPropensity pi_used.π true d_obs q +
      (1 - dbs_true.prob d_obs q) *
        htWeightWithPropensity pi_used.π false d_obs q =
      dbs_true.prob d_obs q / pi_used.π d_obs q := by
  simp [htWeightWithPropensity, div_eq_mul_inv]

/-- Exact HT cancellation holds iff the propensity used in the weight matches
the true sampling propensity at that unit. -/
theorem misspecified_ht_weight_expectation_eq_one_iff
    {Obs Mis Con : Type*}
    (dbs_true : DesignBasedSampling Obs Mis Con)
    (pi_used : ValidSamplingProbability Obs Con)
    (d_obs : Obs) (q : Con) :
    dbs_true.prob d_obs q *
        htWeightWithPropensity pi_used.π true d_obs q +
      (1 - dbs_true.prob d_obs q) *
        htWeightWithPropensity pi_used.π false d_obs q = 1 ↔
      dbs_true.prob d_obs q = pi_used.π d_obs q := by
  rw [misspecified_ht_weight_expectation_eq_ratio]
  constructor
  · intro h
    have hmult := congrArg (fun t : ℝ => t * pi_used.π d_obs q) h
    have h_pos : pi_used.π d_obs q ≠ 0 := ne_of_gt (pi_used.positivity d_obs q)
    simp [div_eq_mul_inv, h_pos] at hmult
    simpa [mul_comm] using hmult
  · intro hmatch
    rw [hmatch]
    have h_pos : pi_used.π d_obs q ≠ 0 := ne_of_gt (pi_used.positivity d_obs q)
    field_simp [h_pos]

/-- Correcting the IPW propensity to the true design restores exact HT
cancellation. -/
theorem ht_weight_expectation_eq_one_of_adjusted
    {Obs Mis Con : Type*}
    (dbs_true : DesignBasedSampling Obs Mis Con)
    (pi_used : ValidSamplingProbability Obs Con)
    (h_adjusted : ∀ d_obs q, pi_used.π d_obs q = dbs_true.prob d_obs q)
    (d_obs : Obs) (q : Con) :
    dbs_true.prob d_obs q *
        htWeightWithPropensity pi_used.π true d_obs q +
      (1 - dbs_true.prob d_obs q) *
        htWeightWithPropensity pi_used.π false d_obs q = 1 := by
  rw [misspecified_ht_weight_expectation_eq_one_iff]
  exact (h_adjusted d_obs q).symm

/-- A proxy-aware sampling design can depend on an error-prone or model-based
signal in addition to the observed document state. -/
def ProxyAwareSamplingProbability (Observed Proxy Content : Type*) :=
  Observed → Proxy → Content → ℝ

/-- Valid proxy-aware sampling probabilities. -/
structure ProxyAwareValidSamplingProbability
    (Observed Proxy Content : Type*) where
  π : ProxyAwareSamplingProbability Observed Proxy Content
  positivity : ∀ (d_obs : Observed) (proxy : Proxy) (q : Content), π d_obs proxy q > 0
  bounded : ∀ (d_obs : Observed) (proxy : Proxy) (q : Content), π d_obs proxy q ≤ 1

namespace ProxyAwareValidSamplingProbability

/-- Collapse a proxy-aware design to an observed-state design by plugging in the
realized proxy mapping. -/
def asObserved {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy) : ValidSamplingProbability Obs Con where
  π := fun d_obs q => pi_actual.π d_obs (proxy d_obs) q
  positivity := fun d_obs q => pi_actual.positivity d_obs (proxy d_obs) q
  bounded := fun d_obs q => pi_actual.bounded d_obs (proxy d_obs) q

end ProxyAwareValidSamplingProbability

/-- Using a propensity that ignores a proxy-aware selection mechanism produces
the exact ratio between the actual proxy-aware propensity and the propensity
used in the weight. -/
theorem proxyAware_ht_weight_expectation_eq_ratio
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (d_obs : Obs) (q : Con) :
    (pi_actual.π d_obs (proxy d_obs) q) *
        htWeightWithPropensity pi_used.π true d_obs q +
      (1 - pi_actual.π d_obs (proxy d_obs) q) *
        htWeightWithPropensity pi_used.π false d_obs q =
      pi_actual.π d_obs (proxy d_obs) q / pi_used.π d_obs q := by
  simpa using
    (misspecified_ht_weight_expectation_eq_ratio
      (Obs := Obs) (Mis := Unit) (Con := Con)
      (dbs_true := {
        π := pi_actual.asObserved proxy
        known_by_design := ()
      })
      (pi_used := pi_used)
      (d_obs := d_obs) (q := q))

/-- Exact HT cancellation under a proxy-aware design is recovered once the IPW
propensity is adjusted to the realized proxy-aware sampling law. -/
theorem proxyAware_ht_weight_expectation_eq_one_of_adjusted
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (h_adjusted : ∀ d_obs q, pi_used.π d_obs q = pi_actual.π d_obs (proxy d_obs) q)
    (d_obs : Obs) (q : Con) :
    (pi_actual.π d_obs (proxy d_obs) q) *
        htWeightWithPropensity pi_used.π true d_obs q +
      (1 - pi_actual.π d_obs (proxy d_obs) q) *
        htWeightWithPropensity pi_used.π false d_obs q = 1 := by
  rw [proxyAware_ht_weight_expectation_eq_ratio]
  rw [h_adjusted d_obs q]
  have h_pos_actual : pi_actual.π d_obs (proxy d_obs) q ≠ 0 :=
    ne_of_gt (pi_actual.positivity d_obs (proxy d_obs) q)
  field_simp [h_pos_actual]

/-- If the actual proxy-aware propensity differs from the propensity used in the
weight, the HT cancellation factor is not one. This is the precise formal
failure mode behind "ignore the error mechanism in IPW and you are in trouble". -/
theorem proxyAware_ht_weight_expectation_ne_one_of_mismatch
    {Obs Proxy Con : Type*}
    (pi_actual : ProxyAwareValidSamplingProbability Obs Proxy Con)
    (proxy : Obs → Proxy)
    (pi_used : ValidSamplingProbability Obs Con)
    (d_obs : Obs) (q : Con)
    (h_mismatch : pi_actual.π d_obs (proxy d_obs) q ≠ pi_used.π d_obs q) :
    (pi_actual.π d_obs (proxy d_obs) q) *
        htWeightWithPropensity pi_used.π true d_obs q +
      (1 - pi_actual.π d_obs (proxy d_obs) q) *
        htWeightWithPropensity pi_used.π false d_obs q ≠ 1 := by
  rw [proxyAware_ht_weight_expectation_eq_ratio]
  intro hratio
  have hmult := congrArg (fun t : ℝ => t * pi_used.π d_obs q) hratio
  have h_pos : pi_used.π d_obs q ≠ 0 := ne_of_gt (pi_used.positivity d_obs q)
  simp [div_eq_mul_inv, h_pos] at hmult
  exact h_mismatch hmult

end DSL
