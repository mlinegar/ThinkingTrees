import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.MeasureTheoreticAudit

/-!
# FormalProofs/OPT/ApproximateLocalLaws.lean

Approximate (`ε`) local-law layer for realistic non-perfect summarizers.

This file uses the existing audit/union-bound machinery:
- local leaf/merge/idempotence violations are represented as additive budgets,
- those budgets propagate to a bound on `Δ_R_ZR`,
- and then into DPO / GRPO-PL / GRPO-RL objective gaps.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- Approximate leaf law: total leaf violation budget is at most `ε`. -/
def L1ε (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) (ε : ℝ) : Prop :=
  totalLeafViolation g fstar T ≤ ε

/-- Approximate merge law: total merge violation budget is at most `ε`. -/
def L2ε (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) (ε : ℝ) : Prop :=
  totalMergeViolation g fstar T ≤ ε

/-- Approximate idempotence law at the reduction distribution: budget at most `ε`. -/
def L3ε (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) (ε : ℝ) : Prop :=
  pIdemp g fstar (reduce g T) ≤ ε

/-- Nodewise approximate leaf law: each realized leaf has its own violation budget. -/
def L1εNode (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε_leaf : Strings → ℝ) : Prop :=
  ∀ b, b ∈ leaves T → ViolationProb fstar (g b) b ≤ ε_leaf b

/-- Nodewise approximate merge law: each realized internal merge has its own budget. -/
def L2εNode (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ) : Prop :=
  ∀ p, p ∈ internal_nodes T →
    let (T_L, T_R) := p
    ViolationProb fstar (reduce g (BinTree.node T_L T_R)) (S (BinTree.node T_L T_R)) ≤ ε_merge p

/-- Aggregate leaf budget induced by nodewise budgets. -/
def leafBudgetOf (T : BinTree Strings) (ε_leaf : Strings → ℝ) : ℝ :=
  (leaves T).foldl (fun acc b => acc + ε_leaf b) 0

/-- Aggregate merge budget induced by nodewise budgets. -/
def mergeBudgetOf (T : BinTree Strings)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ) : ℝ :=
  (internal_nodes T).foldl (fun acc p => acc + ε_merge p) 0

/-- Folded additive form equals `init + (map f).sum`. -/
lemma foldl_add_eq_init_add_sum_map {α : Type*} (l : List α) (f : α → ℝ) (init : ℝ) :
    l.foldl (fun acc a => acc + f a) init = init + (l.map f).sum := by
  induction l generalizing init with
  | nil => simp
  | cons a as ih =>
      simp [List.foldl_cons, ih, add_assoc]

/-- Folded additive form from zero equals `(map f).sum`. -/
lemma foldl_add_eq_sum_map {α : Type*} (l : List α) (f : α → ℝ) :
    l.foldl (fun acc a => acc + f a) 0 = (l.map f).sum := by
  simpa using foldl_add_eq_init_add_sum_map (l := l) (f := f) (init := 0)

/-- Pointwise bounds over a list imply bounds on folded additive totals. -/
lemma foldl_add_le_foldl_add {α : Type*} (l : List α) (f g : α → ℝ)
    (hfg : ∀ a ∈ l, f a ≤ g a) :
    l.foldl (fun acc a => acc + f a) 0 ≤ l.foldl (fun acc a => acc + g a) 0 := by
  rw [foldl_add_eq_sum_map, foldl_add_eq_sum_map]
  exact List.sum_le_sum (f := fun a => f a) (g := fun a => g a) hfg

/-- Nodewise leaf budgets imply the aggregate `L1ε` law. -/
theorem L1ε_of_nodewise
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε_leaf : Strings → ℝ)
    (h_node : L1εNode g T fstar ε_leaf) :
    L1ε g T fstar (leafBudgetOf (T := T) ε_leaf) := by
  unfold L1ε leafBudgetOf totalLeafViolation
  exact foldl_add_le_foldl_add (l := leaves T)
    (f := fun b => ViolationProb fstar (g b) b)
    (g := fun b => ε_leaf b)
    (fun b hb => h_node b hb)

/-- Nodewise merge budgets imply the aggregate `L2ε` law. -/
theorem L2ε_of_nodewise
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ)
    (h_node : L2εNode g T fstar ε_merge) :
    L2ε g T fstar (mergeBudgetOf (T := T) ε_merge) := by
  unfold L2ε mergeBudgetOf totalMergeViolation
  exact foldl_add_le_foldl_add (l := internal_nodes T)
    (f := fun p =>
      ViolationProb fstar (reduce g (BinTree.node p.1 p.2)) (S (BinTree.node p.1 p.2)))
    (g := fun p => ε_merge p)
    (fun p hp => by
      simpa using h_node p hp)

/-- Bundle of approximate local-law budgets. -/
structure ApproxLocalLawsBundle (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  epsLeaf : ℝ
  epsMerge : ℝ
  epsIdemp : ℝ
  law1 : L1ε g T fstar epsLeaf
  law2 : L2ε g T fstar epsMerge
  law3 : L3ε g T fstar epsIdemp

namespace ApproxLocalLawsBundle

/-- Paper-facing local-law error from checked C1/C2/C3 residuals.
The three stored fields remain the measured residual components; this is the
single local-law quantity compared against a target `ε`, or combined with
calibration slack for teacher-first certification. -/
def localLawError
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fstar) (R : ℕ) : ℝ :=
  laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp

/-- Backward-compatible name for `localLawError`. Older theorem statements used
``rootErrorBudget'' for the same composed local-law quantity. -/
def rootErrorBudget
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fstar) (R : ℕ) : ℝ :=
  laws.localLawError R

/-- The legacy and paper-facing names denote the same quantity. -/
theorem localLawError_eq_rootErrorBudget
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fstar) (R : ℕ) :
    laws.localLawError R = laws.rootErrorBudget R := rfl

/-- The bundle certifies a tree at target task error `ε` when the composed
local-law error is at most `ε`. -/
def CertifiedAtEpsilon
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fstar) (R : ℕ) (ε : ℝ) : Prop :=
  laws.localLawError R ≤ ε

/-- Total certified error after adding a two-sided scorer-calibration slack. -/
def totalCertifiedError
    {g : Summarizer Strings} {T : BinTree Strings} {fhat : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fhat) (R : ℕ) (εf : ℝ) : ℝ :=
  laws.localLawError R + 2 * εf

/-- Teacher-first certification predicate: local-law error plus calibration
slack is at most the final target `ε`. -/
def CertifiedWithCalibration
    {g : Summarizer Strings} {T : BinTree Strings} {fhat : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fhat) (R : ℕ) (εf ε : ℝ) : Prop :=
  laws.totalCertifiedError R εf ≤ ε

end ApproxLocalLawsBundle

/-- Build an aggregate approximate-local-law bundle from nodewise leaf/merge budgets
and a global idempotence budget. -/
def approx_bundle_of_nodewise
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε_leaf : Strings → ℝ)
    (ε_merge : BinTree Strings × BinTree Strings → ℝ)
    (ε_idemp : ℝ)
    (h1_node : L1εNode g T fstar ε_leaf)
    (h2_node : L2εNode g T fstar ε_merge)
    (h3 : L3ε g T fstar ε_idemp) :
    ApproxLocalLawsBundle g T fstar where
  epsLeaf := leafBudgetOf (T := T) ε_leaf
  epsMerge := mergeBudgetOf (T := T) ε_merge
  epsIdemp := ε_idemp
  law1 := L1ε_of_nodewise g T fstar ε_leaf h1_node
  law2 := L2ε_of_nodewise g T fstar ε_merge h2_node
  law3 := h3

/-- Audit-level aggregate upper bounds for approximate local laws on a fixed tree. -/
structure AuditedApproxUpperBounds (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  epsLeaf : ℝ
  epsMerge : ℝ
  epsIdemp : ℝ
  leaf_cert : totalLeafViolation g fstar T ≤ epsLeaf
  merge_cert : totalMergeViolation g fstar T ≤ epsMerge
  idemp_cert : pIdemp g fstar (reduce g T) ≤ epsIdemp

/-- Convert audited aggregate upper bounds into an approximate local-law bundle. -/
def approx_bundle_of_audited_upper_bounds
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (audit : AuditedApproxUpperBounds g T fstar) :
    ApproxLocalLawsBundle g T fstar where
  epsLeaf := audit.epsLeaf
  epsMerge := audit.epsMerge
  epsIdemp := audit.epsIdemp
  law1 := audit.leaf_cert
  law2 := audit.merge_cert
  law3 := audit.idemp_cert

/-- Confidence-event wrapper: on event `E`, audited upper bounds hold. -/
structure AuditedBoundsWithConfidence (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  event : Prop
  cert : event → AuditedApproxUpperBounds g T fstar

/-- If the audit confidence event holds, we can transfer to an approximate
local-law bundle. -/
def approx_bundle_of_audited_confidence_event
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (audit : AuditedBoundsWithConfidence g T fstar)
    (h_event : audit.event) :
    ApproxLocalLawsBundle g T fstar := by
  exact approx_bundle_of_audited_upper_bounds g T fstar (audit.cert h_event)

/-- Nodewise empirical audit certificate with Hoeffding/Serfling-style margins.
The margin term is abstracted as `confidence_margin`, matching the concentration
interface from `MeasureTheoreticAudit` and reusable with WOR certificates. -/
structure NodewiseEmpiricalAuditCertificate (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  empLeaf : Strings → ℝ
  empMerge : BinTree Strings × BinTree Strings → ℝ
  empIdemp : ℝ
  deltaLeaf : ℝ
  nLeaf : ℕ
  deltaMerge : ℝ
  nMerge : ℕ
  deltaIdemp : ℝ
  nIdemp : ℕ
  leaf_upper :
    ∀ b, b ∈ leaves T →
      ViolationProb fstar (g b) b ≤ empLeaf b + confidence_margin deltaLeaf nLeaf
  merge_upper :
    ∀ p, p ∈ internal_nodes T →
      ViolationProb fstar (reduce g (BinTree.node p.1 p.2)) (S (BinTree.node p.1 p.2))
        ≤ empMerge p + confidence_margin deltaMerge nMerge
  idemp_upper :
    pIdemp g fstar (reduce g T) ≤ empIdemp + confidence_margin deltaIdemp nIdemp

/-- Convert a nodewise empirical audit certificate into aggregate audited upper bounds. -/
def audited_upper_bounds_of_nodewise_empirical_certificate
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (cert : NodewiseEmpiricalAuditCertificate g T fstar) :
    AuditedApproxUpperBounds g T fstar where
  epsLeaf := leafBudgetOf (T := T) (fun b => cert.empLeaf b + confidence_margin cert.deltaLeaf cert.nLeaf)
  epsMerge := mergeBudgetOf (T := T) (fun p => cert.empMerge p + confidence_margin cert.deltaMerge cert.nMerge)
  epsIdemp := cert.empIdemp + confidence_margin cert.deltaIdemp cert.nIdemp
  leaf_cert := L1ε_of_nodewise g T fstar
    (fun b => cert.empLeaf b + confidence_margin cert.deltaLeaf cert.nLeaf)
    cert.leaf_upper
  merge_cert := L2ε_of_nodewise g T fstar
    (fun p => cert.empMerge p + confidence_margin cert.deltaMerge cert.nMerge)
    cert.merge_upper
  idemp_cert := cert.idemp_upper

/-- One-step lift: empirical nodewise certificates induce an approximate-local-law
bundle directly. -/
def approx_bundle_of_nodewise_empirical_certificate
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (cert : NodewiseEmpiricalAuditCertificate g T fstar) :
    ApproxLocalLawsBundle g T fstar :=
  approx_bundle_of_audited_upper_bounds g T fstar
    (audited_upper_bounds_of_nodewise_empirical_certificate g T fstar cert)

/-- Confidence-event wrapper for nodewise empirical audit certificates. -/
structure NodewiseEmpiricalAuditWithConfidence (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  event : Prop
  cert : event → NodewiseEmpiricalAuditCertificate g T fstar

/-- Under the empirical-audit confidence event, recover an approximate local-law
bundle. -/
def approx_bundle_of_nodewise_empirical_confidence_event
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (audit : NodewiseEmpiricalAuditWithConfidence g T fstar)
    (h_event : audit.event) :
    ApproxLocalLawsBundle g T fstar :=
  approx_bundle_of_nodewise_empirical_certificate g T fstar (audit.cert h_event)

/-- Aggregate violation budget implied by approximate local laws. -/
theorem violation_budget_bound_of_approx_local_laws
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) (R : ℕ)
    (hR : R ≥ 1)
    (ε_leaf ε_merge ε_idemp : ℝ)
    (h1 : L1ε g T fstar ε_leaf)
    (h2 : L2ε g T fstar ε_merge)
    (h3 : L3ε g T fstar ε_idemp) :
    totalLeafViolation g fstar T + totalMergeViolation g fstar T +
      ((R : ℝ) - 1) * pIdemp g fstar (reduce g T)
      ≤ ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp := by
  have h12 : totalLeafViolation g fstar T + totalMergeViolation g fstar T ≤ ε_leaf + ε_merge :=
    add_le_add h1 h2
  have hcoef : 0 ≤ (R : ℝ) - 1 := by
    have hR' : (1 : ℝ) ≤ (R : ℝ) := by exact_mod_cast hR
    linarith
  have h3' :
      ((R : ℝ) - 1) * pIdemp g fstar (reduce g T) ≤
      ((R : ℝ) - 1) * ε_idemp := by
    exact mul_le_mul_of_nonneg_left h3 hcoef
  exact add_le_add h12 h3'

/-- `Δ_R_ZR` bound from approximate local laws and the audit union bound. -/
theorem Δ_R_ZR_le_of_approx_local_laws
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : ℝ)
    (h1 : L1ε g T fstar ε_leaf)
    (h2 : L2ε g T fstar ε_merge)
    (h3 : L3ε g T fstar ε_idemp) :
    Δ_R_ZR g x R T fstar ≤ ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp := by
  have h_union :
      Δ_R_ZR g x R T fstar ≤
      totalLeafViolation g fstar T + totalMergeViolation g fstar T +
      ((R : ℝ) - 1) * pIdemp g fstar (reduce g T) := by
    simpa [Δ_R_ZR] using
      (union_bound_multi_round_bounded g fstar T x hp R hR hbound hbound_global h_mono)
  exact le_trans h_union
    (violation_budget_bound_of_approx_local_laws g T fstar R hR ε_leaf ε_merge ε_idemp h1 h2 h3)

/-- Bundle-driven variant of `Δ_R_ZR` bound under approximate local laws. -/
theorem Δ_R_ZR_le_of_approx_bundle
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar) :
    Δ_R_ZR g x R T fstar ≤
      laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp := by
  exact Δ_R_ZR_le_of_approx_local_laws g T fstar x R hp hR hbound hbound_global h_mono
    laws.epsLeaf laws.epsMerge laws.epsIdemp laws.law1 laws.law2 laws.law3

/-- Bundle-driven local-law-error variant of the `Δ_R_ZR` bound. -/
theorem Δ_R_ZR_le_localLawError_of_approx_bundle
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar) :
    Δ_R_ZR g x R T fstar ≤ laws.localLawError R := by
  simpa [ApproxLocalLawsBundle.localLawError] using
    Δ_R_ZR_le_of_approx_bundle
      g T fstar x R hp hR hbound hbound_global h_mono laws

/-- Epsilon-certificate variant: if the composed local-law error from the
approximate local-law bundle is at most `ε`, then the realized tree distortion
is at most `ε`. -/
theorem Δ_R_ZR_le_of_approx_bundle_certifiedAtEpsilon
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (ε : ℝ) (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (hcert : laws.CertifiedAtEpsilon R ε) :
    Δ_R_ZR g x R T fstar ≤ ε := by
  calc
    Δ_R_ZR g x R T fstar ≤ laws.localLawError R :=
      Δ_R_ZR_le_localLawError_of_approx_bundle
        g T fstar x R hp hR hbound hbound_global h_mono laws
    _ ≤ ε := hcert

section Objectives

variable {A : Type*}

/-- DPO gap bound specialized to approximate local-law budgets. -/
theorem dpo_gap_via_approx_local_laws
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_pol : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x x', gen x = gen x')
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : ℝ)
    (h1 : L1ε g T fstar ε_leaf)
    (h2 : L2ε g T fstar ε_merge)
    (h3 : L3ε g T fstar ε_idemp) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
    2 * |β| * (L_pol : ℝ) * (ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp) := by
  have h_gap :
      |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
       ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
      2 * |β| * (L_pol : ℝ) *
      (totalLeafViolation g fstar T + totalMergeViolation g fstar T +
       ((R : ℝ) - 1) * pIdemp g fstar (reduce g T)) := by
    exact dpo_gap_via_union_bound fstar pol pol_ref gen g x R T β L_pol hp hR
      D_max hD_max h_dist_bound hbound hbound_global
      Loss_max hLoss_max hLoss_bound
      h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono
  have h_budget :
      totalLeafViolation g fstar T + totalMergeViolation g fstar T +
        ((R : ℝ) - 1) * pIdemp g fstar (reduce g T)
        ≤ ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp :=
    violation_budget_bound_of_approx_local_laws g T fstar R hR ε_leaf ε_merge ε_idemp h1 h2 h3
  calc
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
      ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|
        ≤ 2 * |β| * (L_pol : ℝ) *
            (totalLeafViolation g fstar T + totalMergeViolation g fstar T +
             ((R : ℝ) - 1) * pIdemp g fstar (reduce g T)) := h_gap
    _ ≤ 2 * |β| * (L_pol : ℝ) * (ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp) := by
      exact mul_le_mul_of_nonneg_left h_budget (by
        have hL : 0 ≤ (L_pol : ℝ) := NNReal.coe_nonneg L_pol
        nlinarith [abs_nonneg β, hL])

/-- GRPO-PL gap bound specialized to approximate local-law budgets. -/
theorem grpo_pl_gap_via_approx_local_laws {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A), |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : ℝ)
    (h1 : L1ε g T fstar ε_leaf)
    (h2 : L2ε g T fstar ε_merge)
    (h3 : L3ε g T fstar ε_idemp) :
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
     ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
    (L_grpo : ℝ) * (ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp) := by
  have h_Δ :
      Δ_R_ZR g x R T fstar =
      ∑' z, ∑' x',
        (ZR g x R T z).toReal * (PMF.pure x x').toReal * dist (fstar z) (fstar x') := by
    simpa using (coupling_Δ_eq_Δ_R_ZR g x R T fstar).symm
  have h_gap :
      |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
       ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
      L_grpo * Δ_R_ZR g x R T fstar := by
    simpa using
      (grpo_pl_gap_bounded (k := k) (fstar := fstar) (pol := pol) (ranker := ranker)
        (gen := gen) (μ_X := PMF.pure x) (μ_Z := ZR g x R T) (L_grpo := L_grpo)
        (Δ_R := Δ_R_ZR g x R T fstar)
        (D_max := D_max) (hD_max := hD_max) (h_dist_bound := h_dist_bound)
        (Loss_max := Loss_max) (hLoss_max := hLoss_max) (hLoss_bound := hLoss_bound)
        (h_pol_lip := h_pol_lip) (h_ranker := h_ranker) (h_rum := h_rum)
        (h_gen_fixed := h_gen_fixed) (h_Δ := h_Δ))
  have h_Δ_bound :
      Δ_R_ZR g x R T fstar ≤ ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp :=
    Δ_R_ZR_le_of_approx_local_laws g T fstar x R hp hR hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp h1 h2 h3
  calc
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
      ExpectedGRPOLoss pol ranker (ZR g x R T) gen|
        ≤ (L_grpo : ℝ) * Δ_R_ZR g x R T fstar := h_gap
    _ ≤ (L_grpo : ℝ) * (ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp) := by
      exact mul_le_mul_of_nonneg_left h_Δ_bound (NNReal.coe_nonneg _)

/-- GRPO-RL gap bound specialized to approximate local-law budgets. -/
theorem grpo_rl_gap_via_approx_local_laws {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo_rl : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum : ∀ x' z',
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x') L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : ℝ)
    (h1 : L1ε g T fstar ε_leaf)
    (h2 : L2ε g T fstar ε_merge)
    (h3 : L3ε g T fstar ε_idemp) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
     ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
    (L_grpo_rl : ℝ) * (ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp) := by
  have h_Δ :
      Δ_R_ZR g x R T fstar =
      ∑' z, ∑' x',
        (ZR g x R T z).toReal * (PMF.pure x x').toReal * dist (fstar z) (fstar x') := by
    simpa using (coupling_Δ_eq_Δ_R_ZR g x R T fstar).symm
  have h_gap :
      |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
       ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
      L_grpo_rl * Δ_R_ZR g x R T fstar := by
    simpa using
      (grpo_rl_gap_bounded (k := k) (fstar := fstar)
        (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta) (gen := gen)
        (μ_X := PMF.pure x) (μ_Z := ZR g x R T)
        (L_grpo_rl := L_grpo_rl) (Δ_R := Δ_R_ZR g x R T fstar)
        (D_max := D_max) (hD_max := hD_max) (h_dist_bound := h_dist_bound)
        (Loss_max := Loss_max) (hLoss_max := hLoss_max) (hLoss_bound := hLoss_bound)
        (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
        (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
        (h_rum := h_rum) (h_gen_fixed := h_gen_fixed) (h_Δ := h_Δ))
  have h_Δ_bound :
      Δ_R_ZR g x R T fstar ≤ ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp :=
    Δ_R_ZR_le_of_approx_local_laws g T fstar x R hp hR hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp h1 h2 h3
  calc
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|
        ≤ (L_grpo_rl : ℝ) * Δ_R_ZR g x R T fstar := h_gap
    _ ≤ (L_grpo_rl : ℝ) * (ε_leaf + ε_merge + ((R : ℝ) - 1) * ε_idemp) := by
      exact mul_le_mul_of_nonneg_left h_Δ_bound (NNReal.coe_nonneg _)

/-- Bundle-driven DPO gap bound under approximate local laws. -/
theorem dpo_gap_via_approx_bundle
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_pol : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
    2 * |β| * (L_pol : ℝ) *
      (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  exact dpo_gap_via_approx_local_laws fstar pol pol_ref gen g x R T β L_pol hp hR
    D_max hD_max h_dist_bound hbound hbound_global
    Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono
    laws.epsLeaf laws.epsMerge laws.epsIdemp laws.law1 laws.law2 laws.law3

/-- Confidence-event lift: audited approximate-local-law bundle implies a DPO gap
bound on that event. -/
theorem dpo_gap_via_audited_confidence_event
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ) (L_pol : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (audit : AuditedBoundsWithConfidence g T fstar)
    (h_event : audit.event) :
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
     ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
    2 * |β| * (L_pol : ℝ) *
      ((audit.cert h_event).epsLeaf + (audit.cert h_event).epsMerge +
        ((R : ℝ) - 1) * (audit.cert h_event).epsIdemp) := by
  exact dpo_gap_via_approx_bundle fstar pol pol_ref gen g x R T β L_pol hp hR
    D_max hD_max h_dist_bound hbound hbound_global
    Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono
    (audit.cert h_event |> approx_bundle_of_audited_upper_bounds g T fstar)

/-- Bundle-driven GRPO-PL gap bound under approximate local laws. -/
theorem grpo_pl_gap_via_approx_bundle {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A), |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar) :
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
     ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
    (L_grpo : ℝ) * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  exact grpo_pl_gap_via_approx_local_laws (k := k) fstar pol ranker gen g x R T L_grpo
    D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_pol_lip h_ranker h_rum h_gen_fixed
    hp hR hbound hbound_global h_mono
    laws.epsLeaf laws.epsMerge laws.epsIdemp laws.law1 laws.law2 laws.law3

/-- Confidence-event lift: audited approximate-local-law bundle implies a GRPO-PL
gap bound on that event. -/
theorem grpo_pl_gap_via_audited_confidence_event {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A), |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (audit : AuditedBoundsWithConfidence g T fstar)
    (h_event : audit.event) :
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
     ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
    (L_grpo : ℝ) *
      ((audit.cert h_event).epsLeaf + (audit.cert h_event).epsMerge +
        ((R : ℝ) - 1) * (audit.cert h_event).epsIdemp) := by
  exact grpo_pl_gap_via_approx_bundle (k := k) fstar pol ranker gen g x R T L_grpo
    D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_pol_lip h_ranker h_rum h_gen_fixed
    hp hR hbound hbound_global h_mono
    (audit.cert h_event |> approx_bundle_of_audited_upper_bounds g T fstar)

/-- Bundle-driven GRPO-RL gap bound under approximate local laws. -/
theorem grpo_rl_gap_via_approx_bundle {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo_rl : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum : ∀ x' z',
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x') L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
     ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
    (L_grpo_rl : ℝ) * (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  exact grpo_rl_gap_via_approx_local_laws (k := k) fstar pol pol_old pol_ref reward eps beta gen g x R T
    L_grpo_rl D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_pol_lip h_old_lip h_ref_lip h_reward_lip
    h_rum h_gen_fixed
    hp hR hbound hbound_global h_mono
    laws.epsLeaf laws.epsMerge laws.epsIdemp laws.law1 laws.law2 laws.law3

/-- Confidence-event lift: audited approximate-local-law bundle implies a GRPO-RL
gap bound on that event. -/
theorem grpo_rl_gap_via_audited_confidence_event {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L_grpo_rl : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPORLLossPointwise pol pol_old pol_ref reward eps beta x' group| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo_rl)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L_grpo_rl)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L_grpo_rl)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L_grpo_rl)
    (h_rum : ∀ x' z',
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar (gen x') L_grpo_rl
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (audit : AuditedBoundsWithConfidence g T fstar)
    (h_event : audit.event) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
     ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
    (L_grpo_rl : ℝ) *
      ((audit.cert h_event).epsLeaf + (audit.cert h_event).epsMerge +
        ((R : ℝ) - 1) * (audit.cert h_event).epsIdemp) := by
  exact grpo_rl_gap_via_approx_bundle (k := k) fstar pol pol_old pol_ref reward eps beta gen g x R T
    L_grpo_rl D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_pol_lip h_old_lip h_ref_lip h_reward_lip
    h_rum h_gen_fixed
    hp hR hbound hbound_global h_mono
    (audit.cert h_event |> approx_bundle_of_audited_upper_bounds g T fstar)

end Objectives

end FormalProofs.OPT
