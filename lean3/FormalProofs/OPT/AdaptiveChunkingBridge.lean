import FormalProofs.OPT.ApproximateLocalLaws

/-!
# FormalProofs/OPT/AdaptiveChunkingBridge.lean

Bridge layer from adaptive chunking policies (`x ↦ T(x)`) to the fixed-tree
theorems already proved in OPT.
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

/-- Adaptive tree policy selecting a reduction tree per document. -/
abbrev AdaptiveTreeMap (Strings : Type*) := Strings → BinTree Strings

/-- Stochastic adaptive tree policy selecting a distribution over trees per document. -/
abbrev StochasticAdaptiveTreeMap (Strings : Type*) := Strings → PMF (BinTree Strings)

/-- Adaptive chunking is sound if every selected tree reconstructs the document. -/
def AdaptiveChunkingSound (τ : AdaptiveTreeMap Strings) : Prop :=
  ∀ x, S (τ x) = x

/-- Adaptive local laws: each selected tree satisfies a local-law bundle. -/
def AdaptiveLocalLaws (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : AdaptiveTreeMap Strings) : Prop :=
  ∀ x, LocalLawsBundle g (τ x) fstar

/-- Stochastic adaptive local laws: every support tree satisfies a local-law bundle. -/
def StochasticAdaptiveLocalLaws (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings) : Prop :=
  ∀ x T, T ∈ (τ x).support → LocalLawsBundle g T fstar

/-- Adaptive approximate local laws with per-document budgets. -/
def AdaptiveApproxLocalLaws (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : AdaptiveTreeMap Strings)
    (ε_leaf ε_merge ε_idemp : Strings → ℝ) : Prop :=
  ∀ x,
    L1ε g (τ x) fstar (ε_leaf x) ∧
    L2ε g (τ x) fstar (ε_merge x) ∧
    L3ε g (τ x) fstar (ε_idemp x)

/-- Stochastic adaptive chunking is sound if every support tree reconstructs input. -/
def StochasticAdaptiveChunkingSound (τ : StochasticAdaptiveTreeMap Strings) : Prop :=
  ∀ x T, T ∈ (τ x).support → S T = x

/-- Stochastic adaptive approximate laws with per-(document,tree) budgets. -/
def StochasticAdaptiveApproxLocalLaws (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ) : Prop :=
  ∀ x T, T ∈ (τ x).support →
    L1ε g T fstar (ε_leaf x T) ∧
    L2ε g T fstar (ε_merge x T) ∧
    L3ε g T fstar (ε_idemp x T)

/-- Fixed-tree multi-round theorem instantiated along an adaptive tree policy. -/
theorem multi_round_typeclass_of_adaptive
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : AdaptiveTreeMap Strings)
    (h_sound : AdaptiveChunkingSound τ)
    (h_laws : AdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (x : Strings) (R : ℕ) (hR : R ≥ 1) :
    Exp (ZR g x R (τ x)) (fun z => D fstar z x) = 0 := by
  exact multi_round_typeclass g (τ x) x R fstar (h_sound x) (h_laws x).law1 (h_laws x).law2
    (h_laws x).law3 hR

/-- Fixed-tree multi-round theorem instantiated at any support tree of a
stochastic adaptive policy. -/
theorem multi_round_typeclass_of_stochastic_adaptive
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_laws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree Strings) (hT : T ∈ (τ x).support) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
  have h_bundle : LocalLawsBundle g T fstar := h_laws x T hT
  exact multi_round_typeclass g T x R fstar (h_sound x T hT)
    h_bundle.law1 h_bundle.law2 h_bundle.law3 hR

/-- DPO equivalence instantiated along an adaptive tree policy. -/
theorem dpo_equivalence_of_adaptive
    {A : Type*} {Y : Type*} [BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (τ : AdaptiveTreeMap Strings)
    (β : ℝ)
    (h_sound : AdaptiveChunkingSound τ)
    (h_laws : AdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (h_meas : OracleMeasurablePolicies pol pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β (ZR g x R (τ x)) gen := by
  exact dpo_equivalence fstar pol pol_ref gen g x R (τ x) β
    (h_sound x) (h_laws x) hR h_meas h_pair

/-- DPO equivalence instantiated at any support tree of a stochastic adaptive
policy. -/
theorem dpo_equivalence_of_stochastic_adaptive
    {A : Type*} {Y : Type*} [BoundedMetricSpace Y]
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_laws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (h_meas : OracleMeasurablePolicies pol pol_ref fstar)
    (h_pair : OracleIndexedPairGen gen fstar)
    (T : BinTree Strings) (hT : T ∈ (τ x).support) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen := by
  have h_bundle : LocalLawsBundle g T fstar := h_laws x T hT
  exact dpo_equivalence fstar pol pol_ref gen g x R T β
    (h_sound x T hT) h_bundle hR h_meas h_pair

/-- Approximate adaptive laws imply a per-document `Δ_R_ZR` bound. -/
theorem Δ_R_ZR_le_of_adaptive_approx_local_laws
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : AdaptiveTreeMap Strings)
    (ε_leaf ε_merge ε_idemp : Strings → ℝ)
    (h_sound : AdaptiveChunkingSound τ)
    (h_approx : AdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (x : Strings) (R : ℕ) (hR : R ≥ 1) :
    Δ_R_ZR g x R (τ x) fstar ≤
      ε_leaf x + ε_merge x + ((R : ℝ) - 1) * ε_idemp x := by
  have hx := h_approx x
  exact Δ_R_ZR_le_of_approx_local_laws g (τ x) fstar x R (h_sound x) hR
    (hbound x) hbound_global h_mono
    (ε_leaf x) (ε_merge x) (ε_idemp x) hx.1 hx.2.1 hx.2.2

/-- Bundle-driven deterministic adaptive bound with cleaner interface. -/
theorem Δ_R_ZR_le_of_adaptive_approx_bundle
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : AdaptiveTreeMap Strings)
    (h_sound : AdaptiveChunkingSound τ)
    (h_approx : ∀ x, ApproxLocalLawsBundle g (τ x) fstar)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (x : Strings) (R : ℕ) (hR : R ≥ 1) :
    Δ_R_ZR g x R (τ x) fstar ≤
      (h_approx x).epsLeaf + (h_approx x).epsMerge + ((R : ℝ) - 1) * (h_approx x).epsIdemp := by
  exact Δ_R_ZR_le_of_approx_bundle g (τ x) fstar x R (h_sound x) hR
    (hbound x) hbound_global h_mono (h_approx x)

/-- Stochastic adaptive approximate laws imply per-support-tree `Δ_R_ZR` bounds. -/
theorem Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (T : BinTree Strings) (hT : T ∈ (τ x).support) :
    Δ_R_ZR g x R T fstar ≤
      ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T := by
  have hTlaw := h_approx x T hT
  exact Δ_R_ZR_le_of_approx_local_laws g T fstar x R (h_sound x T hT) hR
    (hbound x) hbound_global h_mono
    (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
    hTlaw.1 hTlaw.2.1 hTlaw.2.2

/-- Expected stochastic-policy distortion proxy vanishes when every support tree
satisfies exact local laws. -/
theorem Exp_Δ_R_ZR_eq_zero_of_stochastic_adaptive_local_laws
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_laws : StochasticAdaptiveLocalLaws (g := g) (fstar := fstar) τ)
    (x : Strings) (R : ℕ) (hR : R ≥ 1) :
    Exp (τ x) (fun T => Δ_R_ZR g x R T fstar) = 0 := by
  unfold Exp
  have hzero : ∀ T, (τ x T).toReal * Δ_R_ZR g x R T fstar = 0 := by
    intro T
    by_cases hT : T ∈ (τ x).support
    · have h_bundle : LocalLawsBundle g T fstar := h_laws x T hT
      have hΔ : Δ_R_ZR g x R T fstar = 0 :=
        Δ_R_eq_zero_of_local_laws g x R T fstar (h_sound x T hT)
          h_bundle.law1 h_bundle.law2 h_bundle.law3 hR
      simp [hΔ]
    · have hτ : τ x T = 0 := by
        simpa [PMF.mem_support_iff] using hT
      simp [hτ]
  calc
    ∑' T, (τ x T).toReal * Δ_R_ZR g x R T fstar
        = ∑' T, (0 : ℝ) := by
          apply tsum_congr
          intro T
          exact hzero T
    _ = 0 := by simp

/-- Expected stochastic-policy distortion proxy is bounded by expected per-support
approximate budgets. -/
theorem Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (hsumm_Δ :
      Summable (fun T => (τ x T).toReal * Δ_R_ZR g x R T fstar))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))) :
    Exp (τ x) (fun T => Δ_R_ZR g x R T fstar) ≤
      Exp (τ x) (fun T => ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  unfold Exp
  apply Summable.tsum_le_tsum
  · intro T
    by_cases hT : T ∈ (τ x).support
    · have hΔ :
        Δ_R_ZR g x R T fstar ≤
          ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T :=
        Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws g fstar τ ε_leaf ε_merge ε_idemp
          h_sound h_approx hbound hbound_global h_mono x R hR T hT
      exact mul_le_mul_of_nonneg_left hΔ ENNReal.toReal_nonneg
    · have hτ : τ x T = 0 := by
        simpa [PMF.mem_support_iff] using hT
      simp [hτ]
  · exact hsumm_Δ
  · exact hsumm_budget

/-- Bounded-wrapper variant of expected stochastic-policy `Δ_R_ZR` control that
eliminates explicit summability obligations. -/
theorem Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws_bounded
    (g : Summarizer Strings) (fstar : Strings → Y)
    (τ : StochasticAdaptiveTreeMap Strings)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (M_Δ M_budget : ℝ) (hM_Δ : 0 ≤ M_Δ) (hM_budget : 0 ≤ M_budget)
    (hΔ_abs : ∀ T, |Δ_R_ZR g x R T fstar| ≤ M_Δ)
    (hbudget_abs :
      ∀ T, |ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T| ≤ M_budget) :
    Exp (τ x) (fun T => Δ_R_ZR g x R T fstar) ≤
      Exp (τ x) (fun T => ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) := by
  have hsumm_Δ :
      Summable (fun T => (τ x T).toReal * Δ_R_ZR g x R T fstar) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_Δ hM_Δ hΔ_abs
  have hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_budget hM_budget hbudget_abs
  exact Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws g fstar τ ε_leaf ε_merge ε_idemp
    h_sound h_approx hbound hbound_global h_mono x R hR hsumm_Δ hsumm_budget

/-- Lift an expected stochastic-adaptive oracle-indexed gap bound to a true
target gap bound by appending one oracle-measurement term. -/
theorem Exp_loss_gap_le_of_stochastic_adaptive_oracleMeasurement
    (τ : StochasticAdaptiveTreeMap Strings)
    (x : Strings)
    (loss_true loss_oracle : ℝ)
    (loss_tree budget : BinTree Strings → ℝ)
    (oracle_err : ℝ)
    (h_oracle : |loss_true - loss_oracle| ≤ oracle_err)
    (h_gap :
      Exp (τ x) (fun T => |loss_oracle - loss_tree T|) ≤ Exp (τ x) budget)
    (hsumm_true :
      Summable (fun T => (τ x T).toReal * |loss_true - loss_tree T|))
    (hsumm_gap :
      Summable (fun T => (τ x T).toReal * |loss_oracle - loss_tree T|)) :
    Exp (τ x) (fun T => |loss_true - loss_tree T|) ≤
      oracle_err + Exp (τ x) budget := by
  unfold Exp
  have h_oracle_nonneg : 0 ≤ oracle_err := le_trans (abs_nonneg _) h_oracle
  have hsumm_const :
      Summable (fun T => (τ x T).toReal * oracle_err) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) (fun _ => oracle_err) |oracle_err|
      (abs_nonneg _) (fun _ => by simpa [abs_of_nonneg h_oracle_nonneg])
  have hsumm_rhs :
      Summable (fun T => (τ x T).toReal * (oracle_err + |loss_oracle - loss_tree T|)) := by
    simpa [mul_add, add_comm, add_left_comm, add_assoc] using hsumm_const.add hsumm_gap
  have hpoint :
      ∀ T, (τ x T).toReal * |loss_true - loss_tree T| ≤
        (τ x T).toReal * (oracle_err + |loss_oracle - loss_tree T|) := by
    intro T
    have htri :
        |loss_true - loss_tree T| ≤ oracle_err + |loss_oracle - loss_tree T| := by
      have hdecomp :
          loss_true - loss_tree T =
            (loss_true - loss_oracle) + (loss_oracle - loss_tree T) := by ring
      calc
        |loss_true - loss_tree T|
            = |(loss_true - loss_oracle) + (loss_oracle - loss_tree T)| := by rw [hdecomp]
        _ ≤ |loss_true - loss_oracle| + |loss_oracle - loss_tree T| := abs_add_le _ _
        _ ≤ oracle_err + |loss_oracle - loss_tree T| := by linarith [h_oracle]
    exact mul_le_mul_of_nonneg_left htri ENNReal.toReal_nonneg
  have hmain := Summable.tsum_le_tsum hpoint hsumm_true hsumm_rhs
  calc
    ∑' T, (τ x T).toReal * |loss_true - loss_tree T|
        ≤ ∑' T, (τ x T).toReal * (oracle_err + |loss_oracle - loss_tree T|) := hmain
    _ = ∑' T, ((τ x T).toReal * oracle_err + (τ x T).toReal * |loss_oracle - loss_tree T|) := by
          congr 1
          ext T
          ring
    _ = (∑' T, (τ x T).toReal * oracle_err) +
          (∑' T, (τ x T).toReal * |loss_oracle - loss_tree T|) := by
            rw [Summable.tsum_add hsumm_const hsumm_gap]
    _ ≤ (∑' T, (τ x T).toReal * oracle_err) + Exp (τ x) budget := by
          exact add_le_add (le_refl _) h_gap
    _ = oracle_err + Exp (τ x) budget := by
          rw [tsum_mul_right, PMF.toReal_tsum_coe (τ x)]
          ring

/-- Tree-indexed oracle-measurement lift: if the gap between the true target and
an oracle-indexed target is itself bounded by a tree-dependent envelope, then
the expected true-target gap is controlled by the expected oracle envelope plus
the expected transport budget. This is the stochastic-adaptive form of
non-uniform oracle measurement error. -/
theorem Exp_loss_gap_le_of_stochastic_adaptive_pointwiseOracleMeasurement
    (τ : StochasticAdaptiveTreeMap Strings)
    (x : Strings)
    (loss_true : ℝ)
    (loss_oracle loss_tree oracle_err budget : BinTree Strings → ℝ)
    (h_oracle : ∀ T, |loss_true - loss_oracle T| ≤ oracle_err T)
    (h_gap :
      Exp (τ x) (fun T => |loss_oracle T - loss_tree T|) ≤ Exp (τ x) budget)
    (hsumm_true :
      Summable (fun T => (τ x T).toReal * |loss_true - loss_tree T|))
    (hsumm_oracle :
      Summable (fun T => (τ x T).toReal * oracle_err T))
    (hsumm_gap :
      Summable (fun T => (τ x T).toReal * |loss_oracle T - loss_tree T|))
    (hsumm_budget :
      Summable (fun T => (τ x T).toReal * budget T)) :
    Exp (τ x) (fun T => |loss_true - loss_tree T|) ≤
      Exp (τ x) oracle_err + Exp (τ x) budget := by
  unfold Exp
  have hsumm_rhs :
      Summable (fun T => (τ x T).toReal * (oracle_err T + |loss_oracle T - loss_tree T|)) := by
    simpa [mul_add, add_comm, add_left_comm, add_assoc] using hsumm_oracle.add hsumm_gap
  have hpoint :
      ∀ T, (τ x T).toReal * |loss_true - loss_tree T| ≤
        (τ x T).toReal * (oracle_err T + |loss_oracle T - loss_tree T|) := by
    intro T
    have htri :
        |loss_true - loss_tree T| ≤ oracle_err T + |loss_oracle T - loss_tree T| := by
      have hdecomp :
          loss_true - loss_tree T =
            (loss_true - loss_oracle T) + (loss_oracle T - loss_tree T) := by ring
      calc
        |loss_true - loss_tree T|
            = |(loss_true - loss_oracle T) + (loss_oracle T - loss_tree T)| := by rw [hdecomp]
        _ ≤ |loss_true - loss_oracle T| + |loss_oracle T - loss_tree T| := abs_add_le _ _
        _ ≤ oracle_err T + |loss_oracle T - loss_tree T| := by linarith [h_oracle T]
    exact mul_le_mul_of_nonneg_left htri ENNReal.toReal_nonneg
  have hmain := Summable.tsum_le_tsum hpoint hsumm_true hsumm_rhs
  calc
    ∑' T, (τ x T).toReal * |loss_true - loss_tree T|
        ≤ ∑' T, (τ x T).toReal * (oracle_err T + |loss_oracle T - loss_tree T|) := hmain
    _ = ∑' T, ((τ x T).toReal * oracle_err T + (τ x T).toReal * |loss_oracle T - loss_tree T|) := by
          congr 1
          ext T
          ring
    _ = (∑' T, (τ x T).toReal * oracle_err T) +
          (∑' T, (τ x T).toReal * |loss_oracle T - loss_tree T|) := by
            rw [Summable.tsum_add hsumm_oracle hsumm_gap]
    _ ≤ Exp (τ x) oracle_err + Exp (τ x) budget := by
          exact add_le_add le_rfl h_gap

section Objectives

variable {A : Type*}

/-- Expected DPO gap under a stochastic adaptive policy with approximate local
laws on support trees. -/
theorem Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_pol : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
        ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) ≤
    Exp (τ x) (fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  unfold Exp
  apply Summable.tsum_le_tsum
  · intro T
    by_cases hT : T ∈ (τ x).support
    · have hTlaw := h_approx x T hT
      have hgap :
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤
          2 * |β| * (L_pol : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) :=
        dpo_gap_via_approx_local_laws fstar pol pol_ref gen g x R T β L_pol
          (h_sound x T hT) hR
          D_max hD_max h_dist_bound
          (hbound x) hbound_global
          Loss_max hLoss_max hLoss_bound
          h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono
          (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
          hTlaw.1 hTlaw.2.1 hTlaw.2.2
      exact mul_le_mul_of_nonneg_left hgap ENNReal.toReal_nonneg
    · have hτ : τ x T = 0 := by
        simpa [PMF.mem_support_iff] using hT
      simp [hτ]
  · exact hsumm_gap
  · exact hsumm_budget

/-- Bounded-wrapper expected DPO gap theorem that discharges summability by
uniform bounds. -/
theorem Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_bounded
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (M_gap M_budget : ℝ) (hM_gap : 0 ≤ M_gap) (hM_budget : 0 ≤ M_budget)
    (hgap_abs :
      ∀ T,
        |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
          ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤ M_gap)
    (hbudget_abs :
      ∀ T,
        |2 * |β| * (L_pol : ℝ) *
          (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)| ≤ M_budget) :
    Exp (τ x) (fun T =>
      |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
        ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) ≤
    Exp (τ x) (fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_gap hM_gap
      (fun T => by
        simpa [abs_of_nonneg (abs_nonneg _)] using hgap_abs T)
  have hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_pol : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_budget hM_budget hbudget_abs
  exact Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws fstar pol pol_ref gen g τ β L_pol
    D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed
    hbound hbound_global h_mono
    ε_leaf ε_merge ε_idemp
    h_sound h_approx x R hR hsumm_gap hsumm_budget

/-- Expected DPO gap under a stochastic adaptive policy, lifted from the
oracle-indexed target to an arbitrary true target within oracle measurement
error `oracle_err`. -/
theorem Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ oracle_err)
    (hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_pol : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) ≤
    oracle_err + Exp (τ x) (fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have h_gap :=
    Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws fstar pol pol_ref gen g τ β L_pol
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_meas_pol h_meas_ref h_lip h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp
      h_sound h_approx x R hR hsumm_gap hsumm_budget
  exact Exp_loss_gap_le_of_stochastic_adaptive_oracleMeasurement
    (τ := τ) (x := x)
    (loss_true := loss_true)
    (loss_oracle := ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
    (loss_tree := fun T => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
    (budget := fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))
    (oracle_err := oracle_err)
    h_oracle h_gap hsumm_true hsumm_gap

/-- Expected DPO gap under a stochastic adaptive policy with a tree-dependent
oracle-measurement envelope. This exposes uncertainty from the oracle-to-truth
gap as an expected additional term rather than a single global scalar. -/
theorem Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true : ℝ)
    (oracle_err : BinTree Strings → ℝ)
    (h_oracle :
      ∀ T, |loss_true - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ oracle_err T)
    (hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_oracle :
      Summable (fun T => (τ x T).toReal * oracle_err T))
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_pol : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) ≤
    Exp (τ x) oracle_err + Exp (τ x) (fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have h_gap :=
    Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws fstar pol pol_ref gen g τ β L_pol
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_meas_pol h_meas_ref h_lip h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp
      h_sound h_approx x R hR hsumm_gap hsumm_budget
  exact Exp_loss_gap_le_of_stochastic_adaptive_pointwiseOracleMeasurement
    (τ := τ) (x := x)
    (loss_true := loss_true)
    (loss_oracle := fun _ => ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
    (loss_tree := fun T => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
    (oracle_err := oracle_err)
    (budget := fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))
    h_oracle h_gap hsumm_true hsumm_oracle hsumm_gap hsumm_budget

/-- Bounded-wrapper DPO stochastic-adaptive theorem with oracle measurement. -/
theorem Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_bounded_with_oracleMeasurement
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
    (β : ℝ) (L_pol : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A), |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen| ≤ oracle_err)
    (M_true_gap M_gap M_budget : ℝ)
    (hM_true_gap : 0 ≤ M_true_gap) (hM_gap : 0 ≤ M_gap) (hM_budget : 0 ≤ M_budget)
    (htrue_gap_abs :
      ∀ T,
        |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤ M_true_gap)
    (hgap_abs :
      ∀ T,
        |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
          ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen| ≤ M_gap)
    (hbudget_abs :
      ∀ T,
        |2 * |β| * (L_pol : ℝ) *
          (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)| ≤ M_budget) :
    Exp (τ x) (fun T =>
      |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) ≤
    oracle_err + Exp (τ x) (fun T =>
      2 * |β| * (L_pol : ℝ) *
        (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true - ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_true_gap hM_true_gap
      (fun T => by simpa [abs_of_nonneg (abs_nonneg _)] using htrue_gap_abs T)
  have hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
            ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen|) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_gap hM_gap
      (fun T => by simpa [abs_of_nonneg (abs_nonneg _)] using hgap_abs T)
  have hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          (2 * |β| * (L_pol : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))) :=
    PMF.summable_coe_real_mul_of_bounded (τ x) _ M_budget hM_budget hbudget_abs
  exact Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement
    fstar pol pol_ref gen g τ β L_pol
    D_max hD_max h_dist_bound
    Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed
    hbound hbound_global h_mono
    ε_leaf ε_merge ε_idemp
    h_sound h_approx x R hR
    loss_true oracle_err h_oracle
    hsumm_true hsumm_gap hsumm_budget

/-- Expected GRPO-PL gap under a stochastic adaptive policy with approximate
local laws on support trees. -/
theorem Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws
    {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
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
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
            ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
        ExpectedGRPOLoss pol ranker (ZR g x R T) gen|) ≤
    Exp (τ x) (fun T =>
      (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  unfold Exp
  apply Summable.tsum_le_tsum
  · intro T
    by_cases hT : T ∈ (τ x).support
    · have hTlaw := h_approx x T hT
      have hgap :
          |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
            ExpectedGRPOLoss pol ranker (ZR g x R T) gen| ≤
          (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) :=
        grpo_pl_gap_via_approx_local_laws (k := k) fstar pol ranker gen g x R T L_grpo
          D_max hD_max h_dist_bound
          Loss_max hLoss_max hLoss_bound
          h_pol_lip h_ranker h_rum h_gen_fixed
          (h_sound x T hT) hR
          (hbound x) hbound_global h_mono
          (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
          hTlaw.1 hTlaw.2.1 hTlaw.2.2
      exact mul_le_mul_of_nonneg_left hgap ENNReal.toReal_nonneg
    · have hτ : τ x T = 0 := by
        simpa [PMF.mem_support_iff] using hT
      simp [hτ]
  · exact hsumm_gap
  · exact hsumm_budget

/-- Expected GRPO-PL gap under a stochastic adaptive policy, lifted from the
oracle-indexed target to an arbitrary true target within oracle measurement
error `oracle_err`. -/
theorem Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement
    {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
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
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ oracle_err)
    (hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true - ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
            ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |loss_true - ExpectedGRPOLoss pol ranker (ZR g x R T) gen|) ≤
    oracle_err + Exp (τ x) (fun T =>
      (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have h_gap :=
    Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws
      (k := k) fstar pol ranker gen g τ L_grpo
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_pol_lip h_ranker h_rum h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp
      h_sound h_approx x R hR hsumm_gap hsumm_budget
  exact Exp_loss_gap_le_of_stochastic_adaptive_oracleMeasurement
    (τ := τ) (x := x)
    (loss_true := loss_true)
    (loss_oracle := ExpectedGRPOLoss pol ranker (PMF.pure x) gen)
    (loss_tree := fun T => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
    (budget := fun T =>
      (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))
    (oracle_err := oracle_err)
    h_oracle h_gap hsumm_true hsumm_gap

/-- Expected GRPO-PL gap under a stochastic adaptive policy with a tree-dependent
oracle-measurement envelope. -/
theorem Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    {k : ℕ}
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
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
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true : ℝ)
    (oracle_err : BinTree Strings → ℝ)
    (h_oracle :
      ∀ T, |loss_true - ExpectedGRPOLoss pol ranker (PMF.pure x) gen| ≤ oracle_err T)
    (hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true - ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_oracle :
      Summable (fun T => (τ x T).toReal * oracle_err T))
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
            ExpectedGRPOLoss pol ranker (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |loss_true - ExpectedGRPOLoss pol ranker (ZR g x R T) gen|) ≤
    Exp (τ x) oracle_err + Exp (τ x) (fun T =>
      (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have h_gap :=
    Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws
      (k := k) fstar pol ranker gen g τ L_grpo
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_pol_lip h_ranker h_rum h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp
      h_sound h_approx x R hR hsumm_gap hsumm_budget
  exact Exp_loss_gap_le_of_stochastic_adaptive_pointwiseOracleMeasurement
    (τ := τ) (x := x)
    (loss_true := loss_true)
    (loss_oracle := fun _ => ExpectedGRPOLoss pol ranker (PMF.pure x) gen)
    (loss_tree := fun T => ExpectedGRPOLoss pol ranker (ZR g x R T) gen)
    (oracle_err := oracle_err)
    (budget := fun T =>
      (L_grpo : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))
    h_oracle h_gap hsumm_true hsumm_oracle hsumm_gap hsumm_budget

/-- Expected GRPO-RL gap under a stochastic adaptive policy with approximate
local laws on support trees. -/
theorem Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws
    {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
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
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo_rl : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|) ≤
    Exp (τ x) (fun T =>
      (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  unfold Exp
  apply Summable.tsum_le_tsum
  · intro T
    by_cases hT : T ∈ (τ x).support
    · have hTlaw := h_approx x T hT
      have hgap :
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen| ≤
          (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T) :=
        grpo_rl_gap_via_approx_local_laws (k := k) fstar pol pol_old pol_ref reward eps beta gen g x R T
          L_grpo_rl D_max hD_max h_dist_bound
          Loss_max hLoss_max hLoss_bound
          h_pol_lip h_old_lip h_ref_lip h_reward_lip
          h_rum h_gen_fixed
          (h_sound x T hT) hR
          (hbound x) hbound_global h_mono
          (ε_leaf x T) (ε_merge x T) (ε_idemp x T)
          hTlaw.1 hTlaw.2.1 hTlaw.2.2
      exact mul_le_mul_of_nonneg_left hgap ENNReal.toReal_nonneg
    · have hτ : τ x T = 0 := by
        simpa [PMF.mem_support_iff] using hT
      simp [hτ]
  · exact hsumm_gap
  · exact hsumm_budget

/-- Expected GRPO-RL gap under a stochastic adaptive policy, lifted from the
oracle-indexed target to an arbitrary true target within oracle measurement
error `oracle_err`. -/
theorem Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement
    {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
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
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true -
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤ oracle_err)
    (hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo_rl : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |loss_true -
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|) ≤
    oracle_err + Exp (τ x) (fun T =>
      (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have h_gap :=
    Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws
      (k := k) fstar pol pol_old pol_ref reward eps beta gen g τ L_grpo_rl
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_pol_lip h_old_lip h_ref_lip h_reward_lip
      h_rum h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp
      h_sound h_approx x R hR hsumm_gap hsumm_budget
  exact Exp_loss_gap_le_of_stochastic_adaptive_oracleMeasurement
    (τ := τ) (x := x)
    (loss_true := loss_true)
    (loss_oracle := ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen)
    (loss_tree := fun T =>
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
    (budget := fun T =>
      (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))
    (oracle_err := oracle_err)
    h_oracle h_gap hsumm_true hsumm_gap

/-- Expected GRPO-RL gap under a stochastic adaptive policy with a tree-dependent
oracle-measurement envelope. -/
theorem Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement
    {k : ℕ}
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings)
    (τ : StochasticAdaptiveTreeMap Strings)
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
    (hbound : ∀ x z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (ε_leaf ε_merge ε_idemp : Strings → BinTree Strings → ℝ)
    (h_sound : StochasticAdaptiveChunkingSound τ)
    (h_approx : StochasticAdaptiveApproxLocalLaws (g := g) (fstar := fstar) τ ε_leaf ε_merge ε_idemp)
    (x : Strings) (R : ℕ) (hR : R ≥ 1)
    (loss_true : ℝ)
    (oracle_err : BinTree Strings → ℝ)
    (h_oracle :
      ∀ T, |loss_true -
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen| ≤ oracle_err T)
    (hsumm_true :
      Summable (fun T =>
        (τ x T).toReal *
          |loss_true -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_oracle :
      Summable (fun T => (τ x T).toReal * oracle_err T))
    (hsumm_gap :
      Summable (fun T =>
        (τ x T).toReal *
          |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
            ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|))
    (hsumm_budget :
      Summable (fun T =>
        (τ x T).toReal *
          ((L_grpo_rl : ℝ) *
            (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)))) :
    Exp (τ x) (fun T =>
      |loss_true -
        ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen|) ≤
    Exp (τ x) oracle_err + Exp (τ x) (fun T =>
      (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T)) := by
  have h_gap :=
    Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws
      (k := k) fstar pol pol_old pol_ref reward eps beta gen g τ L_grpo_rl
      D_max hD_max h_dist_bound
      Loss_max hLoss_max hLoss_bound
      h_pol_lip h_old_lip h_ref_lip h_reward_lip
      h_rum h_gen_fixed
      hbound hbound_global h_mono
      ε_leaf ε_merge ε_idemp
      h_sound h_approx x R hR hsumm_gap hsumm_budget
  exact Exp_loss_gap_le_of_stochastic_adaptive_pointwiseOracleMeasurement
    (τ := τ) (x := x)
    (loss_true := loss_true)
    (loss_oracle := fun _ =>
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen)
    (loss_tree := fun T =>
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen)
    (oracle_err := oracle_err)
    (budget := fun T =>
      (L_grpo_rl : ℝ) * (ε_leaf x T + ε_merge x T + ((R : ℝ) - 1) * ε_idemp x T))
    h_oracle h_gap hsumm_true hsumm_oracle hsumm_gap hsumm_budget

end Objectives

end FormalProofs.OPT
