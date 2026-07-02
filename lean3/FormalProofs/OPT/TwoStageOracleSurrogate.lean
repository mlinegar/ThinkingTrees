import FormalProofs.OPT.TheoremBacking
import FormalProofs.OPT.OracleUtility

/-!
# FormalProofs/OPT/TwoStageOracleSurrogate.lean

Two-stage oracle-surrogate route:

1. learn an expensive surrogate oracle `f̂` that approximates the true oracle `f*`;
2. learn a tree summary `g` relative to `f̂` rather than directly relative to `f*`.

This file packages the reduction-side consequence of that design. If the tree
is exact or approximately theorem-backed for `f̂`, and `f̂` is uniformly close
to `f*`, then distortion and Lipschitz oracle utilities for `f*` inherit a
clean additive slack from the stage-1 surrogate error.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

/-- Stage-1 surrogate-oracle approximation: every document score produced by the
learned oracle `f̂` is within `ε` of the true oracle `f*`. -/
def UniformOracleApproximation
    (fstar fhat : Strings → Y) (ε : ℝ≥0) : Prop :=
  ∀ x : Strings, dist (fhat x) (fstar x) ≤ (ε : ℝ)

/-- Paper-facing alias for uniform oracle recovery: the learned oracle/readout
`fhat` approximates the true oracle `fstar` within `ε_orc`. -/
def OracleRecoveredWithin
    (fstar fhat : Strings → Y) (ε_orc : ℝ≥0) : Prop :=
  UniformOracleApproximation fstar fhat ε_orc

/-- The two-sided slack paid when transferring a pairwise comparison from a
learned oracle/readout back to the true oracle. -/
def OracleRecoverySlack (ε_orc : ℝ≥0) : ℝ :=
  2 * (ε_orc : ℝ)

/-- Total true-oracle budget obtained from a local-law budget measured through a
learned oracle/readout and a uniform oracle-recovery error. -/
def TotalOracleRecoveryBudget (E : ℝ) (ε_orc : ℝ≥0) : ℝ :=
  E + OracleRecoverySlack ε_orc

/-- Expectation of a constant under a PMF is the constant itself. -/
lemma Exp_const
    {α : Type*}
    (p : PMF α) (c : ℝ) :
    Exp p (fun _ => c) = c := by
  unfold Exp
  calc
    ∑' z, (p z).toReal * c = (∑' z, (p z).toReal) * c := by
      rw [tsum_mul_right]
    _ = 1 * c := by rw [PMF.toReal_tsum_coe]
    _ = c := by ring

/-- If a function is supportwise bounded above by a nonnegative constant, its
expectation is bounded by the same constant. -/
lemma Exp_le_const_of_support
    {α : Type*}
    (p : PMF α) (f : α → ℝ) (c M : ℝ)
    (hc : 0 ≤ c)
    (hM : 0 ≤ M)
    (hsupport : ∀ z ∈ p.support, f z ≤ c)
    (hf_nonneg : ∀ z, 0 ≤ f z)
    (hf_bound : ∀ z, f z ≤ M) :
    Exp p f ≤ c := by
  have hf_summable :
      Summable (fun z => (p z).toReal * f z) :=
    PMF.summable_coe_real_mul_of_bounded p f M hM (fun z => by
      rw [abs_of_nonneg (hf_nonneg z)]
      exact hf_bound z)
  have hc_summable :
      Summable (fun z => (p z).toReal * c) :=
    PMF.summable_coe_real_mul_of_bounded p (fun _ => c) c hc (fun _ => by
      simp [abs_of_nonneg hc])
  have hmono :
      Exp p f ≤ Exp p (fun _ => c) := by
    unfold Exp
    apply Summable.tsum_le_tsum
    · intro z
      by_cases hz : z ∈ p.support
      · exact mul_le_mul_of_nonneg_left (hsupport z hz) ENNReal.toReal_nonneg
      · simp [PMF.mem_support_iff] at hz
        simp [hz]
    · exact hf_summable
    · exact hc_summable
  simpa [Exp_const] using hmono

/-- Surrogate distortion plus a uniform stage-1 oracle approximation imply a
true-oracle distortion bound with additive `2ε` slack. -/
theorem trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (hApprox : UniformOracleApproximation fstar fhat ε)
    {x x' : Strings} :
    dist (fstar x) (fstar x') ≤ dist (fhat x) (fhat x') + 2 * (ε : ℝ) := by
  have hx : dist (fstar x) (fhat x) ≤ (ε : ℝ) := by
    simpa [dist_comm] using hApprox x
  have hx' : dist (fhat x') (fstar x') ≤ (ε : ℝ) := hApprox x'
  calc
    dist (fstar x) (fstar x')
        ≤ dist (fstar x) (fhat x) + dist (fhat x) (fstar x') := by
            exact dist_triangle _ _ _
    _ ≤ dist (fstar x) (fhat x) +
          (dist (fhat x) (fhat x') + dist (fhat x') (fstar x')) := by
            have htri :=
              add_le_add_left
                (dist_triangle (fhat x) (fhat x') (fstar x'))
                (dist (fstar x) (fhat x))
            simpa [add_assoc, add_left_comm, add_comm] using htri
    _ = dist (fstar x) (fhat x) + dist (fhat x) (fhat x') +
          dist (fhat x') (fstar x') := by ring
    _ ≤ (ε : ℝ) + dist (fhat x) (fhat x') + (ε : ℝ) := by
          linarith
    _ = dist (fhat x) (fhat x') + 2 * (ε : ℝ) := by ring

/-- Exact surrogate-fiber preservation implies true-oracle closeness up to the
stage-1 surrogate slack. -/
theorem sameSurrogateFiber_implies_trueOracleClose_of_uniformOracleApproximation
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (hApprox : UniformOracleApproximation fstar fhat ε)
    {x x' : Strings}
    (hFiber : dist (fhat x) (fhat x') = 0) :
    dist (fstar x) (fstar x') ≤ 2 * (ε : ℝ) := by
  calc
    dist (fstar x) (fstar x')
        ≤ dist (fhat x) (fhat x') + 2 * (ε : ℝ) :=
          trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
            (hApprox := hApprox)
    _ = 2 * (ε : ℝ) := by simp [hFiber]

/-- Exact theorem-backedness for the surrogate oracle keeps every realized
summary within `2ε` of the true oracle. -/
theorem zr_support_trueOracleDist_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    {z : Strings}
    (hz : z ∈ (ZR g x R T).support) :
    dist (fstar z) (fstar x) ≤ 2 * (ε : ℝ) := by
  exact sameSurrogateFiber_implies_trueOracleClose_of_uniformOracleApproximation
    (hApprox := hApprox)
    (zero_distortion_on_ZR_support_of_exactTheoremBacked
      (hp := hp) (hExact := hExact) (hR := hR) z hz)

/-- Exact theorem-backedness for the surrogate oracle implies a uniform bound on
the expected true-oracle distortion. -/
theorem Δ_R_ZR_true_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε) :
    Δ_R_ZR g x R T fstar ≤ 2 * (ε : ℝ) := by
  let p : PMF Strings := ZR g x R T
  let M : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have h_nonneg : ∀ z, 0 ≤ D fstar z x := by
    intro z
    exact dist_nonneg
  have h_bound : ∀ z, D fstar z x ≤ M := by
    intro z
    unfold D M
    exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
  unfold Δ_R_ZR
  exact Exp_le_const_of_support
    (p := p)
    (f := fun z => D fstar z x)
    (c := 2 * (ε : ℝ))
    (M := M)
    (hc := by
      have hε : 0 ≤ (ε : ℝ) := by exact_mod_cast ε.property
      nlinarith)
    (hM := hM)
    (hsupport := by
      intro z hz
      simpa [D] using
        (zr_support_trueOracleDist_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
          (hp := hp) (hExact := hExact) (hR := hR) (hApprox := hApprox) hz))
    (hf_nonneg := h_nonneg)
    (hf_bound := h_bound)

/-- Any Lipschitz utility on the true oracle inherits the `2ε` transport bound
from exact theorem-backedness on the surrogate oracle. -/
theorem expected_trueOracleUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (u : OracleUtility2 Y)
    (L : ℝ≥0)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hL : OracleUtilityLipschitz1 u L) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) -
        u (fstar x) (fstar x)| ≤
      (L : ℝ) * (2 * (ε : ℝ)) := by
  have hUtility :=
    expected_utility_bound_ZR
      (g := g) (T := T) (x := x) (R := R)
      (fstar := fstar) (u := u) (L := L) hL
  have hDist :=
    Δ_R_ZR_true_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (hp := hp) (hExact := hExact) (hR := hR) (hApprox := hApprox)
  calc
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) -
        u (fstar x) (fstar x)|
        ≤ (L : ℝ) * Δ_R_ZR g x R T fstar := by
            simpa [Δ_R_ZR] using hUtility
    _ ≤ (L : ℝ) * (2 * (ε : ℝ)) := by
          apply mul_le_mul_of_nonneg_left hDist
          exact_mod_cast L.property

/-- Approximate theorem-backedness on the surrogate oracle implies a true-oracle
distortion bound with additive `2ε` surrogate slack. -/
theorem Δ_R_ZR_true_le_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (hp : S T = x)
    (hApproxBacked : ApproxTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    Δ_R_ZR g x R T fstar ≤
      (hApproxBacked.approxLocalLaws.epsLeaf +
        hApproxBacked.approxLocalLaws.epsMerge +
        ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
      2 * (ε : ℝ) := by
  let p : PMF Strings := ZR g x R T
  let M : ℝ := BoundedMetricSpace.diameterBound (α := Y) + 2 * (ε : ℝ)
  have hM_nonneg : 0 ≤ M := by
    dsimp [M]
    have hdiam : 0 ≤ BoundedMetricSpace.diameterBound (α := Y) :=
      BoundedMetricSpace.diameterBound_nonneg (α := Y)
    have hε : 0 ≤ (ε : ℝ) := by exact_mod_cast ε.property
    nlinarith
  have h_dist_mono :
      Δ_R_ZR g x R T fstar ≤
        Exp p (fun z => D fhat z x + 2 * (ε : ℝ)) := by
    unfold Δ_R_ZR
    apply Exp_mono_bounded (p := p) (M := M)
    · exact hM_nonneg
    · intro z
      simpa [D] using
        (trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
          (hApprox := hApprox) (x := z) (x' := x))
    · intro z
      have h_nonneg : 0 ≤ D fstar z x := dist_nonneg
      rw [abs_of_nonneg h_nonneg]
      have hdist : D fstar z x ≤ BoundedMetricSpace.diameterBound (α := Y) := by
        unfold D
        exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
      dsimp [M]
      have hε : 0 ≤ (ε : ℝ) := by exact_mod_cast ε.property
      exact le_trans hdist (by linarith)
    · intro z
      have h_nonneg :
          0 ≤ D fhat z x + 2 * (ε : ℝ) := by
        have hε : 0 ≤ (ε : ℝ) := by exact_mod_cast ε.property
        have hdist_nonneg : 0 ≤ D fhat z x := dist_nonneg
        linarith
      rw [abs_of_nonneg h_nonneg]
      have hdist : D fhat z x ≤ BoundedMetricSpace.diameterBound (α := Y) := by
        unfold D
        exact BoundedMetricSpace.dist_le (fhat z) (fhat x)
      dsimp [M]
      have hStep := add_le_add_right hdist (2 * (ε : ℝ))
      simpa [add_assoc, add_left_comm, add_comm] using hStep
  have hDhat_summable :
      Summable (fun z => (p z).toReal * D fhat z x) := by
    let Mhat : ℝ := BoundedMetricSpace.diameterBound (α := Y)
    have hMhat : 0 ≤ Mhat := BoundedMetricSpace.diameterBound_nonneg (α := Y)
    have hbound_hat : ∀ z, D fhat z x ≤ Mhat := by
      intro z
      unfold D Mhat
      exact BoundedMetricSpace.dist_le (fhat z) (fhat x)
    exact summable_D_of_bounded p fhat x Mhat hMhat hbound_hat
  have hconst_summable :
      Summable (fun z => (p z).toReal * (2 * (ε : ℝ))) :=
    PMF.summable_coe_real_mul_of_bounded p (fun _ => 2 * (ε : ℝ)) (2 * (ε : ℝ))
      (by
        have hε : 0 ≤ (ε : ℝ) := by exact_mod_cast ε.property
        nlinarith)
      (fun _ => by
        have hε : 0 ≤ (ε : ℝ) := by exact_mod_cast ε.property
        simp [abs_of_nonneg, hε])
  have hExp_add :
      Exp p (fun z => D fhat z x + 2 * (ε : ℝ)) =
        Δ_R_ZR g x R T fhat + 2 * (ε : ℝ) := by
    rw [Exp_add (p := p)
      (f₁ := fun z => D fhat z x)
      (f₂ := fun _ => 2 * (ε : ℝ))
      hDhat_summable
      hconst_summable]
    simp [Δ_R_ZR, p, Exp_const]
  have hBudget :
      Δ_R_ZR g x R T fhat ≤
        hApproxBacked.approxLocalLaws.epsLeaf +
          hApproxBacked.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp :=
    Δ_R_ZR_le_of_approx_bundle
      g T fhat x R hp hR hbound hbound_global h_mono
      hApproxBacked.approxLocalLaws
  calc
    Δ_R_ZR g x R T fstar ≤ Exp p (fun z => D fhat z x + 2 * (ε : ℝ)) :=
      h_dist_mono
    _ = Δ_R_ZR g x R T fhat + 2 * (ε : ℝ) := hExp_add
    _ ≤ (hApproxBacked.approxLocalLaws.epsLeaf +
          hApproxBacked.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
        2 * (ε : ℝ) := by
          linarith

/-- Approximate theorem-backedness on the surrogate oracle yields a true-oracle
utility gap bound: transport budget on the surrogate plus additive stage-1
surrogate slack. -/
theorem expected_trueOracleUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar fhat : Strings → Y}
    {ε : ℝ≥0}
    (u : OracleUtility2 Y)
    (L : ℝ≥0)
    (hp : S T = x)
    (hApproxBacked : ApproxTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hL : OracleUtilityLipschitz1 u L)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) -
        u (fstar x) (fstar x)| ≤
      (L : ℝ) *
        ((hApproxBacked.approxLocalLaws.epsLeaf +
          hApproxBacked.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
        2 * (ε : ℝ)) := by
  have hUtility :=
    expected_utility_bound_ZR
      (g := g) (T := T) (x := x) (R := R)
      (fstar := fstar) (u := u) (L := L) hL
  have hDist :=
    Δ_R_ZR_true_le_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (hp := hp)
      (hApproxBacked := hApproxBacked)
      (hR := hR)
      (hApprox := hApprox)
      (hbound := hbound)
      (hbound_global := hbound_global)
      (h_mono := h_mono)
  calc
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) -
        u (fstar x) (fstar x)|
        ≤ (L : ℝ) * Δ_R_ZR g x R T fstar := by
            simpa [Δ_R_ZR] using hUtility
    _ ≤ (L : ℝ) *
        ((hApproxBacked.approxLocalLaws.epsLeaf +
          hApproxBacked.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
        2 * (ε : ℝ)) := by
          apply mul_le_mul_of_nonneg_left hDist
          exact_mod_cast L.property

end FormalProofs.OPT
