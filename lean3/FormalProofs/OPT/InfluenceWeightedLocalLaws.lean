import FormalProofs.OPT.CoreDefinitions

/-!
# Influence-Weighted Local-Law Certificates

This file isolates the "no adversarially hidden needles" condition discussed in
the C-TreePO paper response.  The theorem layer is intentionally generic over a
finite audit-row type: a row can be a C1 leaf check, a C3 merge check, or a
round-indexed C2/idempotence check.

The main idea is:

* local residuals are weighted by an influence function `lambda`;
* the audit policy has logged row propensities `pi`;
* consequential rows must have enough audit overlap, measured by
  `sum lambda^2 / pi` and `max lambda / pi`;
* if root error is controlled by the influence-weighted local residual mass,
  then any finite-sample upper bound on that mass immediately upper-bounds root
  error.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-!
## Audit rows and local residuals
-/

/-- Local-law channel for a finite audit row. -/
inductive LocalLawChannel
  | c1_leaf
  | c2_idempotence
  | c3_merge
  deriving DecidableEq, Repr

/-- A paper-facing audit row: channel plus an abstract node and round index.

For a one-pass tree certificate, the round index can be `Unit`.  For repeated
summarization, C2 rows can use a genuine round index while C1/C3 rows reuse a
dummy value. -/
structure LocalLawAuditRow (Node Round : Type*) where
  channel : LocalLawChannel
  node : Node
  round : Round
  deriving Repr

/-- Deterministic local-law residual measured by an oracle/readout `f`.

For stochastic summarizers, instantiate `out` as an expected residual row or use
this file's generic residual interface directly. -/
def deterministicLocalLawResidual
    {AuditRow Strings Y : Type*} [PseudoMetricSpace Y]
    (f : Strings -> Y) (out target : AuditRow -> Strings) (a : AuditRow) : ℝ :=
  D f (out a) (target a)

/-- Influence-weighted local-law mass: `sum_a lambda(a) r(a)`. -/
def weightedLocalLawMass {AuditRow : Type*} [Fintype AuditRow]
    (lambda residual : AuditRow -> ℝ) : ℝ :=
  ∑ a, lambda a * residual a

/-- Horvitz-Thompson summand for an influence-weighted audit row. -/
def influenceHTSummand {AuditRow : Type*}
    (lambda pi residual : AuditRow -> ℝ) (a : AuditRow) : ℝ :=
  lambda a / pi a * residual a

/-- Empirical HT estimate from a finite list of sampled rows.

The caller is responsible for ensuring that the list was sampled from the
logged design.  Concentration is supplied separately as an event/assumption. -/
def empiricalInfluenceHT {AuditRow : Type*}
    (lambda pi residual : AuditRow -> ℝ) (sample : List AuditRow) : ℝ :=
  (1 / (sample.length : ℝ)) *
    (sample.map (fun a => influenceHTSummand lambda pi residual a)).sum

/-!
## Audit-overlap quantities
-/

/-- Influence-weighted design effect proxy: `sum_a lambda(a)^2 / pi(a)`. -/
def influenceDesignEffect {AuditRow : Type*} [Fintype AuditRow]
    (lambda pi : AuditRow -> ℝ) : ℝ :=
  ∑ a, (lambda a)^2 / pi a

/-- Influence-weighted worst importance ratio predicate. -/
def influenceWorstRatioBound {AuditRow : Type*}
    (lambda pi : AuditRow -> ℝ) (W : ℝ) : Prop :=
  forall a, lambda a / pi a <= W

/-- Audit overlap for consequential rows.

This is the formal "no adversarially hidden needles" condition: rows with
nonzero influence must have positive logged propensity, and the design effect
and worst importance ratio are bounded. -/
structure InfluenceWeightedAuditOverlap
    {AuditRow : Type*} [Fintype AuditRow]
    (lambda pi : AuditRow -> ℝ) (Dlambda Wlambda : ℝ) : Prop where
  lambda_nonneg : forall a, 0 <= lambda a
  pi_pos : forall a, 0 < pi a
  pi_le_one : forall a, pi a <= 1
  design_effect_le : influenceDesignEffect lambda pi <= Dlambda
  worst_ratio_le : influenceWorstRatioBound lambda pi Wlambda

namespace InfluenceWeightedAuditOverlap

/-- Consequential rows have positive audit probability. -/
theorem consequential_row_positive
    {AuditRow : Type*} [Fintype AuditRow]
    {lambda pi : AuditRow -> ℝ} {Dlambda Wlambda : ℝ}
    (h : InfluenceWeightedAuditOverlap lambda pi Dlambda Wlambda)
    (a : AuditRow) (_h_consequential : 0 < lambda a) :
    0 < pi a :=
  h.pi_pos a

end InfluenceWeightedAuditOverlap

/-!
## Calibration transfer from proxy local laws to true local laws
-/

/-- Uniform oracle/readout calibration implies each deterministic true local
residual is bounded by the proxy residual plus `2 * eps`. -/
theorem deterministicLocalLawResidual_le_proxy_plus_two_calibration
    {AuditRow Strings Y : Type*} [PseudoMetricSpace Y]
    (fstar fhat : Strings -> Y)
    (out target : AuditRow -> Strings)
    (eps : ℝ)
    (h_cal : forall s, dist (fstar s) (fhat s) <= eps)
    (a : AuditRow) :
    deterministicLocalLawResidual fstar out target a <=
      deterministicLocalLawResidual fhat out target a + 2 * eps := by
  unfold deterministicLocalLawResidual D
  have h1 :
      dist (fstar (out a)) (fstar (target a)) <=
        dist (fstar (out a)) (fhat (out a)) +
          dist (fhat (out a)) (fstar (target a)) :=
    dist_triangle _ _ _
  have h2 :
      dist (fhat (out a)) (fstar (target a)) <=
        dist (fhat (out a)) (fhat (target a)) +
          dist (fhat (target a)) (fstar (target a)) :=
    dist_triangle _ _ _
  have h_out : dist (fstar (out a)) (fhat (out a)) <= eps := h_cal (out a)
  have h_target : dist (fhat (target a)) (fstar (target a)) <= eps := by
    simpa [dist_comm] using h_cal (target a)
  linarith

/-- If each true residual is bounded by the proxy residual plus `eta`, then the
influence-weighted true mass is bounded by proxy mass plus
`eta * sum lambda`. -/
theorem weightedLocalLawMass_le_proxy_plus_uniform_slack
    {AuditRow : Type*} [Fintype AuditRow]
    (lambda trueResidual proxyResidual : AuditRow -> ℝ)
    (eta : ℝ)
    (h_lambda_nonneg : forall a, 0 <= lambda a)
    (h_row : forall a, trueResidual a <= proxyResidual a + eta) :
    weightedLocalLawMass lambda trueResidual <=
      weightedLocalLawMass lambda proxyResidual + eta * (∑ a, lambda a) := by
  classical
  unfold weightedLocalLawMass
  calc
    ∑ a, lambda a * trueResidual a
        <= ∑ a, lambda a * (proxyResidual a + eta) := by
          refine Finset.sum_le_sum ?_
          intro a _ha
          exact mul_le_mul_of_nonneg_left (h_row a) (h_lambda_nonneg a)
    _ = ∑ a, (lambda a * proxyResidual a + lambda a * eta) := by
          simp [mul_add]
    _ = (∑ a, lambda a * proxyResidual a) + ∑ a, lambda a * eta := by
          simp [Finset.sum_add_distrib]
    _ = (∑ a, lambda a * proxyResidual a) + eta * (∑ a, lambda a) := by
          simp [Finset.mul_sum, mul_comm]

/-- Proxy-to-true weighted local-law mass transfer under uniform calibration. -/
theorem weightedOracleMass_le_proxy_plus_calibration
    {AuditRow Strings Y : Type*} [Fintype AuditRow] [PseudoMetricSpace Y]
    (lambda : AuditRow -> ℝ)
    (fstar fhat : Strings -> Y)
    (out target : AuditRow -> Strings)
    (eps : ℝ)
    (h_lambda_nonneg : forall a, 0 <= lambda a)
    (h_cal : forall s, dist (fstar s) (fhat s) <= eps) :
    weightedLocalLawMass lambda
        (deterministicLocalLawResidual fstar out target) <=
      weightedLocalLawMass lambda
        (deterministicLocalLawResidual fhat out target) +
        (2 * eps) * (∑ a, lambda a) := by
  exact weightedLocalLawMass_le_proxy_plus_uniform_slack
    (lambda := lambda)
    (trueResidual := deterministicLocalLawResidual fstar out target)
    (proxyResidual := deterministicLocalLawResidual fhat out target)
    (eta := 2 * eps)
    h_lambda_nonneg
    (deterministicLocalLawResidual_le_proxy_plus_two_calibration
      (fstar := fstar) (fhat := fhat) (out := out) (target := target)
      (eps := eps) h_cal)

/-!
## Root-error certificates
-/

/-- The local-to-global propagation assumption: root/document error is bounded
by influence-weighted local-law residual mass. -/
def RootErrorControlledByInfluenceMass
    {AuditRow : Type*} [Fintype AuditRow]
    (rootError : ℝ) (lambda residual : AuditRow -> ℝ) : Prop :=
  rootError <= weightedLocalLawMass lambda residual

/-- Deterministic certificate: an upper bound on influence-weighted local-law
mass is an upper bound on root error. -/
theorem rootError_le_of_influence_weighted_mass_upper
    {AuditRow : Type*} [Fintype AuditRow]
    {rootError estimate statRadius calibrationRadius : ℝ}
    {lambda residual : AuditRow -> ℝ}
    (h_root : RootErrorControlledByInfluenceMass rootError lambda residual)
    (h_mass :
      weightedLocalLawMass lambda residual <=
        estimate + statRadius + calibrationRadius) :
    rootError <= estimate + statRadius + calibrationRadius :=
  le_trans h_root h_mass

/-- Proxy certificate: combine propagation, proxy finite-sample estimation, and
calibration transfer. -/
theorem rootError_le_proxy_estimate_plus_stat_plus_calibration
    {AuditRow : Type*} [Fintype AuditRow]
    {rootError estimate statRadius calibrationRadius : ℝ}
    {lambda trueResidual proxyResidual : AuditRow -> ℝ}
    (h_root : RootErrorControlledByInfluenceMass rootError lambda trueResidual)
    (h_cal_mass :
      weightedLocalLawMass lambda trueResidual <=
        weightedLocalLawMass lambda proxyResidual + calibrationRadius)
    (h_est :
      weightedLocalLawMass lambda proxyResidual <= estimate + statRadius) :
    rootError <= estimate + statRadius + calibrationRadius := by
  have h_true :
      weightedLocalLawMass lambda trueResidual <=
        estimate + statRadius + calibrationRadius := by
    linarith
  exact rootError_le_of_influence_weighted_mass_upper
    (h_root := h_root) (h_mass := h_true)

/-- A packaged influence-weighted finite-sample certificate.  The statistical
radius is typically instantiated using `influenceDesignEffect` and the worst
ratio bound from `InfluenceWeightedAuditOverlap`. -/
structure InfluenceWeightedErrorCertificate
    {AuditRow : Type*} [Fintype AuditRow]
    (rootError : ℝ) (lambda trueResidual proxyResidual : AuditRow -> ℝ) where
  estimate : ℝ
  statRadius : ℝ
  calibrationRadius : ℝ
  root_control :
    RootErrorControlledByInfluenceMass rootError lambda trueResidual
  calibration_control :
    weightedLocalLawMass lambda trueResidual <=
      weightedLocalLawMass lambda proxyResidual + calibrationRadius
  estimation_control :
    weightedLocalLawMass lambda proxyResidual <= estimate + statRadius

namespace InfluenceWeightedErrorCertificate

/-- The displayed certificate bound:
`estimate + statRadius + calibrationRadius`. -/
def totalBound
    {AuditRow : Type*} [Fintype AuditRow]
    {rootError : ℝ} {lambda trueResidual proxyResidual : AuditRow -> ℝ}
    (c : InfluenceWeightedErrorCertificate rootError lambda trueResidual proxyResidual) : ℝ :=
  c.estimate + c.statRadius + c.calibrationRadius

/-- The packaged certificate bounds root/document error. -/
theorem rootError_le_totalBound
    {AuditRow : Type*} [Fintype AuditRow]
    {rootError : ℝ} {lambda trueResidual proxyResidual : AuditRow -> ℝ}
    (c : InfluenceWeightedErrorCertificate rootError lambda trueResidual proxyResidual) :
    rootError <= c.totalBound := by
  exact rootError_le_proxy_estimate_plus_stat_plus_calibration
    (h_root := c.root_control)
    (h_cal_mass := c.calibration_control)
    (h_est := c.estimation_control)

end InfluenceWeightedErrorCertificate

end FormalProofs.OPT

end
