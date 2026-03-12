import FormalProofs.OPT.TheoremBackingStructure
import FormalProofs.OPT.PreferenceLearning

/-!
# FormalProofs/OPT/TheoremBackingConsequences.lean

Consequences of exact theorem-backedness for multi-round reduction and downstream
objective equivalence.

This file makes explicit a chain that was already latent in the development:

1. `ExactTheoremBacked` packages `L1/L2/L3`.
2. `L1/L2/L3` imply exact multi-round zero distortion on `ZR`.
3. Zero distortion implies equality of any oracle-measurable expected loss,
   including DPO, GRPO, GRPO-RL, and compositional preference programs.

The boundary is important: these consequences apply to objectives indexed by the
same oracle `fstar`. Utilities that depend on a richer exact latent state are
handled separately by `ExactUtilityTransport.lean`.
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

open Set

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

/-- If the expected oracle distortion under a PMF is zero, then distortion is zero
at every support point of that PMF. -/
lemma dist_zero_on_support_of_Exp_zero
    (p : PMF Strings) (fstar : Strings → Y) (x : Strings)
    (h_exp_zero : Exp p (fun z => D fstar z x) = 0) :
    ∀ z ∈ p.support, dist (fstar z) (fstar x) = 0 := by
  let M : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have hbound : ∀ z, D fstar z x ≤ M := by
    intro z
    simpa [M, D] using (BoundedMetricSpace.dist_le (fstar z) (fstar x))
  have h_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    summable_D_of_bounded p fstar x M hM hbound
  have h_term_zero : ∀ z, (p z).toReal * D fstar z x = 0 :=
    tsum_eq_zero_of_nonneg
      (fun z => (p z).toReal * D fstar z x)
      (fun z => mul_nonneg ENNReal.toReal_nonneg dist_nonneg)
      h_summable
      (by simpa [Exp] using h_exp_zero)
  intro z hz
  have hz_ne0 : p z ≠ 0 := by
    simpa [PMF.mem_support_iff] using hz
  have hz_toReal_pos : 0 < (p z).toReal :=
    ENNReal.toReal_pos hz_ne0 (PMF.apply_ne_top p z)
  have hz_mul : (p z).toReal * D fstar z x = 0 := h_term_zero z
  rcases mul_eq_zero.mp hz_mul with hz_toReal | hz_dist
  · exfalso
    exact (ne_of_gt hz_toReal_pos) hz_toReal
  · simpa [D] using hz_dist

/-- Exact theorem-backedness implies supportwise zero distortion for the multi-round
reduction distribution `ZR`. -/
theorem zero_distortion_on_ZR_support_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1) :
    ∀ z ∈ (ZR g x R T).support, dist (fstar z) (fstar x) = 0 := by
  have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_typeclass g T x R fstar hp
      hExact.localLaws.law1 hExact.localLaws.law2 hExact.localLaws.law3 hR
  exact dist_zero_on_support_of_Exp_zero (p := ZR g x R T) (fstar := fstar) (x := x) h_exp_zero

/-- Exact theorem-backedness implies the multi-round zero-distortion theorem. -/
theorem multi_round_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
  exact multi_round_typeclass g T x R fstar hp
    hExact.localLaws.law1 hExact.localLaws.law2 hExact.localLaws.law3 hR

/-- Exact theorem-backedness is enough for any oracle-measurable expected-loss
objective on `PMF.pure x` versus the multi-round reduction `ZR`. -/
theorem expected_loss_eq_via_ZR_of_exactTheoremBacked
    {α : Type*}
    (fstar : Strings → Y)
    (loss : Strings → α → ℝ)
    (gen : Strings → PMF α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurableLossGeneric loss fstar)
    (h_gen : OracleIndexedGenGeneric gen fstar) :
    ExpectedLossGeneric loss (PMF.pure x) gen =
    ExpectedLossGeneric loss (ZR g x R T) gen := by
  exact expected_loss_eq_of_zero_dist_generic fstar loss gen (PMF.pure x) (ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_gen

/-- Exact theorem-backedness is enough for any compositional preference loss on
`PMF.pure x` versus `ZR`. -/
theorem expected_pref_loss_eq_via_ZR_of_exactTheoremBacked
    {α : Type*}
    (fstar : Strings → Y)
    (loss : PrefLoss Strings α)
    (gen : PrefGen Strings α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_gen : OracleIndexedGenComb gen fstar) :
    ExpectedPrefLoss loss (PMF.pure x) gen =
    ExpectedPrefLoss loss (ZR g x R T) gen := by
  exact expected_pref_loss_eq_of_zero_dist (fstar := fstar)
    (loss := loss) (gen := gen) (μ_X := PMF.pure x) (μ_Z := ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_gen

/-- Exact theorem-backedness is enough for nested preference programs built from
oracle-indexed preference samplers. -/
theorem expected_pref_loss_prog_eq_via_ZR_of_exactTheoremBacked
    {α : Type*}
    (fstar : Strings → Y)
    (loss : PrefLoss Strings α)
    (prog : PrefProgram Strings α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_prog : OracleIndexedProgram fstar prog) :
    ExpectedPrefLossProg loss (PMF.pure x) prog =
    ExpectedPrefLossProg loss (ZR g x R T) prog := by
  exact expected_pref_loss_prog_eq_of_zero_dist (fstar := fstar)
    (loss := loss) (μ_X := PMF.pure x) (μ_Z := ZR g x R T) (prog := prog)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_prog

section DPO

variable {A : Type*}

/-- DPO expected-loss equivalence follows directly from exact theorem-backedness. -/
theorem dpo_equivalence_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A) (β : ℝ)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen := by
  exact expected_loss_eq_of_zero_dist fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas_pol
    h_meas_ref
    h_gen

/-- DPO exact argmin preservation over `PMF.pure x` versus `ZR` follows from exact
theorem-backedness. -/
theorem dpo_exact_metric_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar) :
    SameOracleMeasurableArgmin
      (fun pol => ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
      (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
      fstar := by
  exact dpo_exact_metric fstar pol_ref gen (PMF.pure x) (ZR g x R T) β
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas_ref
    h_gen

end DPO

section GRPO

variable {A : Type*}
variable {k : ℕ}

/-- GRPO-Plackett-Luce expected-loss equivalence follows directly from exact
theorem-backedness. -/
theorem grpo_equivalence_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
    ExpectedGRPOLoss pol ranker (ZR g x R T) gen := by
  exact grpo_equivalence (Y := Y) fstar pol ranker gen (PMF.pure x) (ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_pol
    h_ranker
    h_gen

/-- GRPO-RL expected-loss equivalence follows directly from exact theorem-backedness. -/
theorem grpo_rl_equivalence_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen := by
  exact grpo_rl_equivalence (Y := Y) (k := k) fstar pol pol_old pol_ref reward eps beta gen
    (PMF.pure x) (ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_gen

end GRPO

end FormalProofs.OPT
