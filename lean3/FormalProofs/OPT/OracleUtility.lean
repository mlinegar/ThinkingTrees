import FormalProofs.OPT.PreservationTheorems
import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceBounds

/-!
# FormalProofs/OPT/OracleUtility.lean

## Oracle Utility Transport (Stochastic Trees)

This file records the **theorem statement** that connects oracle-valued utilities
to stochastic tree reductions without assuming separability.

Key idea:
- A utility is a score on **oracle labels** (Y), not on document pieces.
- The summarizer `g` is stochastic (`Summarizer Strings`), so the tree reduction
  yields a **distribution** over summaries.
- Under a Lipschitz utility assumption, the expected utility gap is controlled
  by expected oracle distortion `D`.

This is the bridge we need for:
- RUM-style utility representations on oracle values
- Tree-based aggregation without additivity/separability
- DPO/PPO bounds that already scale with expected distortion
- Sampling-based recovery of oracle utility via IPW

Paper reference: Section 3 (Preservation) + Section 6/8 (Preference/Gaps).
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*}

section Pseudo

variable [PseudoMetricSpace Y]

/-!
## Oracle Utilities
-/

/-- Oracle-level utility (score) on labels. -/
abbrev OracleUtility (Y : Type*) := Y → ℝ

/-- Oracle-level evaluation on (predicted label, true label). -/
abbrev OracleUtility2 (Y : Type*) := Y → Y → ℝ

/-- Document utility induced by oracle labels. -/
def DocUtility (fstar : Strings → Y) (u : OracleUtility Y) : Strings → ℝ :=
  fun x => u (fstar x)

/-- Summary evaluation against a document's oracle label. -/
def DocSummaryUtility (fstar : Strings → Y) (u : OracleUtility2 Y) (x z : Strings) : ℝ :=
  u (fstar z) (fstar x)

/-- Lipschitz utility on the oracle space (no separability assumed). -/
def OracleUtilityLipschitz (u : OracleUtility Y) (L : ℝ≥0) : Prop :=
  ∀ y y', |u y - u y'| ≤ (L : ℝ) * dist y y'

/-- Lipschitz in the first argument (utility vs. truth label fixed). -/
def OracleUtilityLipschitz1 (u : OracleUtility2 Y) (L : ℝ≥0) : Prop :=
  ∀ y y' y0, |u y y0 - u y' y0| ≤ (L : ℝ) * dist y y'

/-- Lipschitz in the second argument (utility vs. truth label varies). -/
def OracleUtilityLipschitz2 (u : OracleUtility2 Y) (L : ℝ≥0) : Prop :=
  ∀ y y0 y0', |u y y0 - u y y0'| ≤ (L : ℝ) * dist y0 y0'

/-- Utility bounded at a fixed truth label. -/
def OracleUtilityBoundedAt (u : OracleUtility2 Y) (y0 : Y) (U : ℝ) : Prop :=
  0 ≤ U ∧ ∀ y, |u y y0| ≤ U

/-- Example utility: 1-distance to the true label. -/
def oneMinusDist2 : OracleUtility2 Y := fun y ytrue => 1 - dist y ytrue

/-- `oneMinusDist2` is 1-Lipschitz in its first argument. -/
lemma oneMinusDist2_lipschitz1 : OracleUtilityLipschitz1 (oneMinusDist2 (Y := Y)) (1 : ℝ≥0) := by
  intro y y' y0
  -- |(1 - d(y,y0)) - (1 - d(y',y0))| = |d(y',y0) - d(y,y0)| ≤ d(y,y')
  have h1 : dist y y0 - dist y' y0 ≤ dist y y' := by
    -- dist y y0 ≤ dist y y' + dist y' y0
    have := dist_triangle y y' y0
    linarith
  have h2 : dist y' y0 - dist y y0 ≤ dist y y' := by
    -- dist y' y0 ≤ dist y' y + dist y y0
    have h := dist_triangle y' y y0
    have h' : dist y' y0 - dist y y0 ≤ dist y' y := by
      linarith
    simpa [dist_comm] using h'
  have habs : |dist y' y0 - dist y y0| ≤ dist y y' := by
    refine abs_le.mpr ?_
    constructor
    · linarith [h1]
    · exact h2
  -- Rewrite the target
  have hrewrite :
      |(1 - dist y y0) - (1 - dist y' y0)| = |dist y' y0 - dist y y0| := by
    ring_nf
  -- L = 1
  simpa [oneMinusDist2, hrewrite] using habs

/-!
## Measurement Error (Noisy Truth Labels)
-/

/-- Pointwise measurement-error bound (truth label perturbed). -/
lemma utility_noise_pointwise
    (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz2 u L)
    (y y0 y0' : Y) :
    |u y y0 - u y y0'| ≤ (L : ℝ) * dist y0 y0' := by
  exact hL y y0 y0'

/-- Expected measurement-error bound for a fixed document `x` and summary distribution `p`.

This version assumes summability of the baseline oracle utility at the true label. -/
theorem expected_utility_noise_bound_pmf
    (p : PMF Strings) (x : Strings)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz2 u L)
    (hsum_u2 : Summable (fun z => (p z).toReal * u (fstar z) (fstar x))) :
    |Exp p (fun z => u (fstar z) (fhat x)) -
      Exp p (fun z => u (fstar z) (fstar x))| ≤
      (L : ℝ) * dist (fhat x) (fstar x) := by
  classical
  -- Abbreviations
  let u1 : Strings → ℝ := fun z => u (fstar z) (fhat x)
  let u2 : Strings → ℝ := fun z => u (fstar z) (fstar x)
  let C : ℝ := (L : ℝ) * dist (fhat x) (fstar x)
  have hC_nonneg : 0 ≤ C := by
    exact mul_nonneg (by exact_mod_cast L.property) dist_nonneg
  have hdiff : ∀ z, |u1 z - u2 z| ≤ C := by
    intro z
    have h := hL (fstar z) (fhat x) (fstar x)
    simpa [u1, u2, C] using h
  -- Summability for the difference and for u1
  have hsum_diff : Summable (fun z => (p z).toReal * (u1 z - u2 z)) :=
    PMF.summable_coe_real_mul_of_bounded p (fun z => u1 z - u2 z) C hC_nonneg
      (fun z => by simpa using hdiff z)
  have hsum_u1 : Summable (fun z => (p z).toReal * u1 z) := by
    have hsum_add :
        Summable (fun z => (p z).toReal * (u1 z - u2 z) + (p z).toReal * u2 z) :=
      hsum_diff.add hsum_u2
    have hEq :
        (fun z => (p z).toReal * (u1 z - u2 z) + (p z).toReal * u2 z) =
        (fun z => (p z).toReal * u1 z) := by
      funext z
      ring
    simpa [hEq] using hsum_add
  -- Rewrite the gap
  have hrewrite :
      Exp p (fun z => u1 z) - Exp p (fun z => u2 z) =
        ∑' z, (p z).toReal * (u1 z - u2 z) := by
    unfold Exp
    rw [← Summable.tsum_sub hsum_u1 hsum_u2]
    refine tsum_congr ?_
    intro z
    ring
  -- Apply abs_tsum_le_tsum_abs
  have hsum_abs :
      Summable (fun z => (p z).toReal * |u1 z - u2 z|) :=
    PMF.summable_coe_real_mul_of_bounded p (fun z => |u1 z - u2 z|) C hC_nonneg
      (fun z => by
        have h := hdiff z
        simpa [abs_abs] using h)
  have hsum_abs_f : Summable (fun z => |(p z).toReal * (u1 z - u2 z)|) := by
    simpa [abs_mul, abs_of_nonneg ENNReal.toReal_nonneg] using hsum_abs
  have habs :
      |∑' z, (p z).toReal * (u1 z - u2 z)| ≤
        ∑' z, (p z).toReal * |u1 z - u2 z| := by
    simpa using
      (abs_tsum_le_tsum_abs'
        (fun z => (p z).toReal * (u1 z - u2 z)) hsum_diff hsum_abs_f)
  -- Bound the RHS by C
  have hsum_const :
      Summable (fun z => (p z).toReal * C) :=
    PMF.summable_coe_real_mul_of_bounded p (fun _ => C) C hC_nonneg
      (fun _ => by
        have hC : |C| = C := abs_of_nonneg hC_nonneg
        simp [hC])
  have hterm :
      ∀ z, (p z).toReal * |u1 z - u2 z| ≤ (p z).toReal * C := by
    intro z
    exact mul_le_mul_of_nonneg_left (hdiff z) ENNReal.toReal_nonneg
  have hsum_le :
      ∑' z, (p z).toReal * |u1 z - u2 z| ≤ ∑' z, (p z).toReal * C :=
    Summable.tsum_le_tsum hterm hsum_abs hsum_const
  have hsum_p : ∑' z, (p z).toReal = 1 := PMF.toReal_tsum_coe p
  have hsum_C : ∑' z, (p z).toReal * C = C := by
    calc
      ∑' z, (p z).toReal * C = (∑' z, (p z).toReal) * C := by
        simp [tsum_mul_right, hsum_p]
      _ = C := by simp [hsum_p]
  calc
    |Exp p (fun z => u1 z) - Exp p (fun z => u2 z)|
        = |∑' z, (p z).toReal * (u1 z - u2 z)| := by
            simp [hrewrite]
    _ ≤ ∑' z, (p z).toReal * |u1 z - u2 z| := habs
    _ ≤ ∑' z, (p z).toReal * C := hsum_le
    _ = C := hsum_C

/-! A boundedness corollary (summability is automatic). -/
theorem expected_utility_noise_bound_pmf_bounded
    (p : PMF Strings) (x : Strings)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0) (U : ℝ)
    (hL : OracleUtilityLipschitz2 u L)
    (hU : OracleUtilityBoundedAt u (fstar x) U) :
    |Exp p (fun z => u (fstar z) (fhat x)) -
      Exp p (fun z => u (fstar z) (fstar x))| ≤
      (L : ℝ) * dist (fhat x) (fstar x) := by
  have hsum_u2 :
      Summable (fun z => (p z).toReal * u (fstar z) (fstar x)) :=
    PMF.summable_coe_real_mul_of_bounded p (fun z => u (fstar z) (fstar x)) U hU.1
      (fun z => by
        have h := hU.2 (fstar z)
        simpa using h)
  simpa using
    (expected_utility_noise_bound_pmf (p := p) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L := L) hL hsum_u2)

/-! ZR-specialized measurement error bounds. -/

theorem expected_utility_noise_bound_ZR_summable
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz2 u L)
    (hsum_u2 : Summable (fun z => (ZR g x R T z).toReal * u (fstar z) (fstar x))) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fhat x)) -
      Exp (ZR g x R T) (fun z => u (fstar z) (fstar x))| ≤
      (L : ℝ) * dist (fhat x) (fstar x) := by
  simpa using
    (expected_utility_noise_bound_pmf (p := ZR g x R T) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L := L) hL hsum_u2)

theorem expected_utility_noise_bound_ZR
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0) (U : ℝ)
    (hL : OracleUtilityLipschitz2 u L)
    (hU : OracleUtilityBoundedAt u (fstar x) U) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fhat x)) -
      Exp (ZR g x R T) (fun z => u (fstar z) (fstar x))| ≤
      (L : ℝ) * dist (fhat x) (fstar x) := by
  simpa using
    (expected_utility_noise_bound_pmf_bounded (p := ZR g x R T) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L := L) (U := U) hL hU)

/-!
## Main Statement: Utility Transport Bound (No Boundedness)
-/

/-- **Utility Transport Bound (PMF Form, summability + bounded utility).**

This version does not assume a bounded oracle space. Instead it assumes:
1. `u` is bounded at the fixed truth label `f*(x)`, and
2. the distortion expectation is summable.
-/
theorem expected_utility_bound_pmf
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0) (U : ℝ)
    (hL : OracleUtilityLipschitz1 u L)
    (hU : OracleUtilityBoundedAt u (fstar x) U)
    (hD : Summable (fun z => (p z).toReal * D fstar z x)) :
    |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Exp p (fun z => D fstar z x) := by
  classical
  -- Abbreviations
  let uz : Strings → ℝ := fun z => u (fstar z) (fstar x)
  let ux : ℝ := u (fstar x) (fstar x)
  have hU_nonneg : 0 ≤ U := hU.1
  have hU_z : ∀ z, |uz z| ≤ U := by
    intro z; simpa [uz] using hU.2 (fstar z)
  have hU_x : |ux| ≤ U := by
    simpa [ux] using hU.2 (fstar x)
  have hdiff_bound : ∀ z, |uz z - ux| ≤ (2 : ℝ) * U := by
    intro z
    have h1 : |uz z - ux| ≤ |uz z| + |ux| := by
      simpa [sub_eq_add_neg] using (abs_add_le (uz z) (-ux))
    have h2 : |uz z| + |ux| ≤ U + U := add_le_add (hU_z z) hU_x
    calc
      |uz z - ux| ≤ U + U := le_trans h1 h2
      _ = (2 : ℝ) * U := by ring
  -- Summability for the needed series
  have hsum_uz : Summable (fun z => (p z).toReal * uz z) :=
    PMF.summable_coe_real_mul_of_bounded p (fun z => uz z) U hU_nonneg
      (fun z => by simpa using hU_z z)
  have hsum_ux : Summable (fun z => (p z).toReal * ux) :=
    PMF.summable_coe_real_mul_of_bounded p (fun _ => ux) U hU_nonneg
      (fun _ => by simpa using hU_x)
  -- Rewrite the gap as a single tsum
  have hrewrite :
      Exp p (fun z => uz z) - ux =
        ∑' z, (p z).toReal * (uz z - ux) := by
    -- Expand Exp
    unfold Exp
    -- Replace ux with its expectation under p
    have hsum_p : ∑' z, (p z).toReal = 1 := PMF.toReal_tsum_coe p
    have hconst : ∑' z, (p z).toReal * ux = ux := by
      calc
        ∑' z, (p z).toReal * ux
            = (∑' z, (p z).toReal) * ux := by
                simpa using (tsum_mul_right (f := fun z => (p z).toReal) (a := ux))
        _ = ux := by simp [hsum_p]
    calc
      ∑' z, (p z).toReal * uz z - ux
          = ∑' z, (p z).toReal * uz z - ∑' z, (p z).toReal * ux := by
              rw [hconst]
      _ = ∑' z, ((p z).toReal * uz z - (p z).toReal * ux) := by
              rw [← Summable.tsum_sub hsum_uz hsum_ux]
      _ = ∑' z, (p z).toReal * (uz z - ux) := by
              refine tsum_congr ?_
              intro z
              ring
  -- Apply abs_tsum_le_tsum_abs
  let f : Strings → ℝ := fun z => (p z).toReal * (uz z - ux)
  have hsum_f : Summable f := by
    have hsum_diff : Summable (fun z => (p z).toReal * uz z - (p z).toReal * ux) :=
      hsum_uz.sub hsum_ux
    simpa [f, mul_sub] using hsum_diff
  have hsum_abs_f : Summable (fun z => |f z|) := by
    have h2U_nonneg : 0 ≤ (2 : ℝ) * U := by linarith
    have hsum_abs :
        Summable (fun z => (p z).toReal * |uz z - ux|) :=
      PMF.summable_coe_real_mul_of_bounded p (fun z => |uz z - ux|)
        ((2 : ℝ) * U) h2U_nonneg (fun z => by
          simpa [abs_abs] using hdiff_bound z)
    simpa [f, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg] using hsum_abs
  have habs : |∑' z, f z| ≤ ∑' z, |f z| :=
    abs_tsum_le_tsum_abs' f hsum_f hsum_abs_f
  -- Bound the RHS by Lipschitz + distortion
  have hsum_LD :
      Summable (fun z => (p z).toReal * ((L : ℝ) * D fstar z x)) := by
    -- scale the given summable distortion series
    have hsum_scaled : Summable (fun z => (L : ℝ) * ((p z).toReal * D fstar z x)) :=
      hD.mul_left (L : ℝ)
    -- rewrite to match the target
    simpa [mul_left_comm, mul_assoc] using hsum_scaled
  have hLipschitz_sum :
      ∑' z, |f z| ≤ ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x) := by
    have hterm :
        ∀ z, |f z| ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
      intro z
      have hnonneg : 0 ≤ (p z).toReal := ENNReal.toReal_nonneg
      have hL' : |uz z - ux| ≤ (L : ℝ) * D fstar z x := by
        simpa [uz, ux, D] using (hL (fstar z) (fstar x) (fstar x))
      calc
        |f z| = (p z).toReal * |uz z - ux| := by
                simp [f, abs_mul, abs_of_nonneg hnonneg]
        _ ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
                exact mul_le_mul_of_nonneg_left hL' hnonneg
    refine Summable.tsum_le_tsum hterm ?_ hsum_LD
    simpa [f, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg] using hsum_abs_f
  -- Put everything together
  calc
    |Exp p (fun z => uz z) - ux|
        = |∑' z, f z| := by
            simp [hrewrite, f]
    _ ≤ ∑' z, |f z| := habs
    _ ≤ ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x) := hLipschitz_sum
    _ = (L : ℝ) * ∑' z, (p z).toReal * D fstar z x := by
          -- factor out L
          calc
            ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x)
                = ∑' z, (L : ℝ) * ((p z).toReal * D fstar z x) := by
                    refine tsum_congr ?_
                    intro z
                    ring
            _ = (L : ℝ) * ∑' z, (p z).toReal * D fstar z x := by
                    simpa using (tsum_mul_left (f := fun z => (p z).toReal * D fstar z x) (a := (L : ℝ)))
    _ = (L : ℝ) * Exp p (fun z => D fstar z x) := by
          rfl

/-!
## Utility Transport Bound (Summability-Only Version)

This variant removes the boundedness assumption and instead assumes
summability of the oracle-utility series at the true label. -/

theorem expected_utility_bound_pmf_summable
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L)
    (hsum_uz : Summable (fun z => (p z).toReal * u (fstar z) (fstar x)))
    (hD : Summable (fun z => (p z).toReal * D fstar z x)) :
    |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Exp p (fun z => D fstar z x) := by
  classical
  -- Abbreviations
  let uz : Strings → ℝ := fun z => u (fstar z) (fstar x)
  let ux : ℝ := u (fstar x) (fstar x)
  -- Summability for the constant term
  have hsum_ux : Summable (fun z => (p z).toReal * ux) :=
    PMF.summable_coe_real_mul_of_bounded p (fun _ => ux) |ux|
      (abs_nonneg _) (fun _ => by simp)
  -- Rewrite the gap as a single tsum
  have hrewrite :
      Exp p (fun z => uz z) - ux =
        ∑' z, (p z).toReal * (uz z - ux) := by
    unfold Exp
    have hsum_p : ∑' z, (p z).toReal = 1 := PMF.toReal_tsum_coe p
    have hconst : ∑' z, (p z).toReal * ux = ux := by
      calc
        ∑' z, (p z).toReal * ux
            = (∑' z, (p z).toReal) * ux := by
                simpa using (tsum_mul_right (f := fun z => (p z).toReal) (a := ux))
        _ = ux := by simp [hsum_p]
    calc
      ∑' z, (p z).toReal * uz z - ux
          = ∑' z, (p z).toReal * uz z - ∑' z, (p z).toReal * ux := by
              rw [hconst]
      _ = ∑' z, ((p z).toReal * uz z - (p z).toReal * ux) := by
              rw [← Summable.tsum_sub hsum_uz hsum_ux]
      _ = ∑' z, (p z).toReal * (uz z - ux) := by
              refine tsum_congr ?_
              intro z
              ring
  -- Apply abs_tsum_le_tsum_abs
  let f : Strings → ℝ := fun z => (p z).toReal * (uz z - ux)
  have hsum_f : Summable f := by
    have hsum_diff : Summable (fun z => (p z).toReal * uz z - (p z).toReal * ux) :=
      hsum_uz.sub hsum_ux
    simpa [f, mul_sub] using hsum_diff
  -- Summability of |f z| via domination by the Lipschitz distortion term.
  have hsum_LD :
      Summable (fun z => (p z).toReal * ((L : ℝ) * D fstar z x)) := by
    have hsum_scaled : Summable (fun z => (L : ℝ) * ((p z).toReal * D fstar z x)) :=
      hD.mul_left (L : ℝ)
    simpa [mul_left_comm, mul_assoc] using hsum_scaled
  have hsum_abs_f : Summable (fun z => |f z|) := by
    refine Summable.of_nonneg_of_le (fun _ => abs_nonneg _) ?_ hsum_LD
    intro z
    have hnonneg : 0 ≤ (p z).toReal := ENNReal.toReal_nonneg
    have hL' : |uz z - ux| ≤ (L : ℝ) * D fstar z x := by
      simpa [uz, ux, D] using (hL (fstar z) (fstar x) (fstar x))
    calc
      |f z| = (p z).toReal * |uz z - ux| := by
              simp [f, abs_mul, abs_of_nonneg hnonneg]
      _ ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
              exact mul_le_mul_of_nonneg_left hL' hnonneg
  have habs : |∑' z, f z| ≤ ∑' z, |f z| :=
    abs_tsum_le_tsum_abs' f hsum_f hsum_abs_f
  -- Bound the RHS by Lipschitz + distortion
  have hLipschitz_sum :
      ∑' z, |f z| ≤ ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x) := by
    have hterm :
        ∀ z, |f z| ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
      intro z
      have hnonneg : 0 ≤ (p z).toReal := ENNReal.toReal_nonneg
      have hL' : |uz z - ux| ≤ (L : ℝ) * D fstar z x := by
        simpa [uz, ux, D] using (hL (fstar z) (fstar x) (fstar x))
      calc
        |f z| = (p z).toReal * |uz z - ux| := by
                simp [f, abs_mul, abs_of_nonneg hnonneg]
        _ ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
                exact mul_le_mul_of_nonneg_left hL' hnonneg
    refine Summable.tsum_le_tsum hterm hsum_abs_f hsum_LD
  -- Put everything together
  calc
    |Exp p (fun z => uz z) - ux|
        = |∑' z, f z| := by
            simp [hrewrite, f]
    _ ≤ ∑' z, |f z| := habs
    _ ≤ ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x) := hLipschitz_sum
    _ = (L : ℝ) * ∑' z, (p z).toReal * D fstar z x := by
          calc
            ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x)
                = ∑' z, (L : ℝ) * ((p z).toReal * D fstar z x) := by
                    refine tsum_congr ?_
                    intro z
                    ring
            _ = (L : ℝ) * ∑' z, (p z).toReal * D fstar z x := by
                    simpa using (tsum_mul_left (f := fun z => (p z).toReal * D fstar z x) (a := (L : ℝ)))
    _ = (L : ℝ) * Exp p (fun z => D fstar z x) := by
          rfl

/-- Utility transport bound specialized to ZR distributions (summability + bounded utility). -/
theorem expected_utility_bound_ZR_summable
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0) (U : ℝ)
    (hL : OracleUtilityLipschitz1 u L)
    (hU : OracleUtilityBoundedAt u (fstar x) U)
    (hD : Summable (fun z => (ZR g x R T z).toReal * D fstar z x)) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) := by
  simpa using
    (expected_utility_bound_pmf (p := ZR g x R T) (x := x)
      (fstar := fstar) (u := u) (L := L) (U := U) hL hU hD)

/-- Utility transport bound for ZR distributions (summability only, no boundedness). -/
theorem expected_utility_bound_ZR_summable_unbounded
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L)
    (hsum_uz : Summable (fun z => (ZR g x R T z).toReal * u (fstar z) (fstar x)))
    (hD : Summable (fun z => (ZR g x R T z).toReal * D fstar z x)) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) := by
  simpa using
    (expected_utility_bound_pmf_summable (p := ZR g x R T) (x := x)
      (fstar := fstar) (u := u) (L := L) hL hsum_uz hD)

/-!
## Big Picture: Transport + Measurement Error
-/

/-- Combined transport + measurement error bound (PMF form).

This ties together:
- transport from summaries to the document (Lipschitz in the first argument), and
- label noise in the truth label (Lipschitz in the second argument). -/
theorem expected_utility_bound_with_noise_pmf
    (p : PMF Strings) (x : Strings)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y)
    (L1 L2 : ℝ≥0) (U : ℝ)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (fstar x) U)
    (hD : Summable (fun z => (p z).toReal * D fstar z x)) :
    |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
      (L1 : ℝ) * Exp p (fun z => D fstar z x) +
      (L2 : ℝ) * dist (fhat x) (fstar x) := by
  -- Summability for the baseline oracle utility at the true label.
  have hsum_u2 :
      Summable (fun z => (p z).toReal * u (fstar z) (fstar x)) :=
    PMF.summable_coe_real_mul_of_bounded p (fun z => u (fstar z) (fstar x)) U hU.1
      (fun z => by
        have h := hU.2 (fstar z)
        simpa using h)
  -- Two pieces: measurement error + transport
  have h_noise :
      |Exp p (fun z => u (fstar z) (fhat x)) -
        Exp p (fun z => u (fstar z) (fstar x))| ≤
        (L2 : ℝ) * dist (fhat x) (fstar x) :=
    expected_utility_noise_bound_pmf (p := p) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L := L2) hL2 hsum_u2
  have h_transport :
      |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
        (L1 : ℝ) * Exp p (fun z => D fstar z x) :=
    expected_utility_bound_pmf (p := p) (x := x) (fstar := fstar)
      (u := u) (L := L1) (U := U) hL1 hU hD
  -- Triangle inequality
  have htriangle :
      |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
        |Exp p (fun z => u (fstar z) (fhat x)) -
            Exp p (fun z => u (fstar z) (fstar x))| +
        |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| := by
    have h :
        Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x) =
          (Exp p (fun z => u (fstar z) (fhat x)) -
              Exp p (fun z => u (fstar z) (fstar x))) +
          (Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)) := by
      ring
    calc
      |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)|
          = |(Exp p (fun z => u (fstar z) (fhat x)) -
                Exp p (fun z => u (fstar z) (fstar x))) +
              (Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x))| := by
            rw [h]
      _ ≤ |Exp p (fun z => u (fstar z) (fhat x)) -
              Exp p (fun z => u (fstar z) (fstar x))| +
            |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| := by
            exact abs_add_le _ _
  calc
    |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
        |Exp p (fun z => u (fstar z) (fhat x)) -
            Exp p (fun z => u (fstar z) (fstar x))| +
        |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| := htriangle
    _ ≤ (L2 : ℝ) * dist (fhat x) (fstar x) +
        (L1 : ℝ) * Exp p (fun z => D fstar z x) := by
          exact add_le_add h_noise h_transport
    _ = (L1 : ℝ) * Exp p (fun z => D fstar z x) +
        (L2 : ℝ) * dist (fhat x) (fstar x) := by
          ring

/-! Summability-only version (no boundedness). -/

theorem expected_utility_bound_with_noise_pmf_summable
    (p : PMF Strings) (x : Strings)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y)
    (L1 L2 : ℝ≥0)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hsum_u2 : Summable (fun z => (p z).toReal * u (fstar z) (fstar x)))
    (hD : Summable (fun z => (p z).toReal * D fstar z x)) :
    |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
      (L1 : ℝ) * Exp p (fun z => D fstar z x) +
      (L2 : ℝ) * dist (fhat x) (fstar x) := by
  have h_noise :
      |Exp p (fun z => u (fstar z) (fhat x)) -
        Exp p (fun z => u (fstar z) (fstar x))| ≤
        (L2 : ℝ) * dist (fhat x) (fstar x) :=
    expected_utility_noise_bound_pmf (p := p) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L := L2) hL2 hsum_u2
  have h_transport :
      |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
        (L1 : ℝ) * Exp p (fun z => D fstar z x) :=
    expected_utility_bound_pmf_summable (p := p) (x := x) (fstar := fstar)
      (u := u) (L := L1) hL1 hsum_u2 hD
  -- Triangle inequality
  have htriangle :
      |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
        |Exp p (fun z => u (fstar z) (fhat x)) -
            Exp p (fun z => u (fstar z) (fstar x))| +
        |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| := by
    have h :
        Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x) =
          (Exp p (fun z => u (fstar z) (fhat x)) -
              Exp p (fun z => u (fstar z) (fstar x))) +
          (Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)) := by
      ring
    calc
      |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)|
          = |(Exp p (fun z => u (fstar z) (fhat x)) -
                Exp p (fun z => u (fstar z) (fstar x))) +
              (Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x))| := by
            rw [h]
      _ ≤ |Exp p (fun z => u (fstar z) (fhat x)) -
              Exp p (fun z => u (fstar z) (fstar x))| +
            |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| := by
            exact abs_add_le _ _
  calc
    |Exp p (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
        |Exp p (fun z => u (fstar z) (fhat x)) -
            Exp p (fun z => u (fstar z) (fstar x))| +
        |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| := htriangle
    _ ≤ (L2 : ℝ) * dist (fhat x) (fstar x) +
        (L1 : ℝ) * Exp p (fun z => D fstar z x) := by
          exact add_le_add h_noise h_transport
    _ = (L1 : ℝ) * Exp p (fun z => D fstar z x) +
        (L2 : ℝ) * dist (fhat x) (fstar x) := by
          ring

/-! ZR-specialized transport + measurement error bounds. -/

theorem expected_utility_bound_with_noise_ZR_summable
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y)
    (L1 L2 : ℝ≥0) (U : ℝ)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (fstar x) U)
    (hD : Summable (fun z => (ZR g x R T z).toReal * D fstar z x)) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) +
      (L2 : ℝ) * dist (fhat x) (fstar x) := by
  simpa using
    (expected_utility_bound_with_noise_pmf (p := ZR g x R T) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L1 := L1) (L2 := L2) (U := U)
      hL1 hL2 hU hD)

theorem expected_utility_bound_with_noise_ZR
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y)
    (L1 L2 : ℝ≥0) (U : ℝ)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (fstar x) U)
    (hD : Summable (fun z => (ZR g x R T z).toReal * D fstar z x)) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) +
      (L2 : ℝ) * dist (fhat x) (fstar x) := by
  simpa using
    (expected_utility_bound_with_noise_ZR_summable (g := g) (T := T) (x := x) (R := R)
      (fstar := fstar) (fhat := fhat) (u := u) (L1 := L1) (L2 := L2) (U := U)
      hL1 hL2 hU hD)

theorem expected_utility_bound_with_noise_ZR_summable_unbounded
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y)
    (L1 L2 : ℝ≥0)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hsum_u2 : Summable (fun z => (ZR g x R T z).toReal * u (fstar z) (fstar x)))
    (hD : Summable (fun z => (ZR g x R T z).toReal * D fstar z x)) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fhat x)) - u (fstar x) (fstar x)| ≤
      (L1 : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) +
      (L2 : ℝ) * dist (fhat x) (fstar x) := by
  simpa using
    (expected_utility_bound_with_noise_pmf_summable (p := ZR g x R T) (x := x)
      (fstar := fstar) (fhat := fhat) (u := u) (L1 := L1) (L2 := L2)
      hL1 hL2 hsum_u2 hD)

end Pseudo

section Bounded

variable [BoundedPseudoMetricSpace Y]

/-!
## Main Statement: Utility Transport Bound
-/

/-- **Utility Transport Bound (PMF Form)**.

General version: for any distribution over summaries, the expected utility gap
is bounded by expected oracle distortion.
-/
theorem expected_utility_bound_pmf_bounded
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) :
    |Exp p (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Exp p (fun z => D fstar z x) := by
  classical
  -- Abbreviations
  let uz : Strings → ℝ := fun z => u (fstar z) (fstar x)
  let ux : ℝ := u (fstar x) (fstar x)
  let M : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Y)
  have hL_nonneg : 0 ≤ (L : ℝ) := by exact_mod_cast L.property
  have hsum_p : ∑' z, (p z).toReal = 1 := by
    simpa using (PMF.toReal_tsum_coe p)
  have hdist_bound : ∀ z, D fstar z x ≤ M := by
    intro z
    unfold D
    exact BoundedPseudoMetricSpace.dist_le (fstar z) (fstar x)
  have hLzx : ∀ z, |uz z - ux| ≤ (L : ℝ) * D fstar z x := by
    intro z
    simpa [uz, ux, D] using (hL (fstar z) (fstar x) (fstar x))
  have hLzx_M : ∀ z, |uz z - ux| ≤ (L : ℝ) * M := by
    intro z
    exact le_trans (hLzx z) (mul_le_mul_of_nonneg_left (hdist_bound z) hL_nonneg)
  -- Bound uz for summability
  have hbound_uz : ∀ z, |uz z| ≤ |ux| + (L : ℝ) * M := by
    intro z
    have htri : |uz z| ≤ |uz z - ux| + |ux| := by
      have h' : (uz z - ux) + ux = uz z := by ring
      have h := abs_add_le (uz z - ux) ux
      simpa [h'] using h
    calc
      |uz z| ≤ |uz z - ux| + |ux| := htri
      _ ≤ (L : ℝ) * M + |ux| := by
            exact add_le_add (hLzx_M z) (le_of_eq rfl)
      _ = |ux| + (L : ℝ) * M := by ring
  -- Summability for the needed series
  have hsum_uz : Summable (fun z => (p z).toReal * uz z) :=
    PMF.summable_coe_real_mul_of_bounded p (fun z => uz z)
      (|ux| + (L : ℝ) * M)
      (by
        apply add_nonneg
        · exact abs_nonneg _
        · exact mul_nonneg hL_nonneg hM)
      (fun z => by simpa using hbound_uz z)
  have hsum_ux : Summable (fun z => (p z).toReal * ux) :=
    PMF.summable_coe_real_mul_of_bounded p (fun _ => ux) |ux|
      (abs_nonneg _)
      (fun z => by simp)
  -- Rewrite the gap as a single tsum
  have hrewrite :
      Exp p (fun z => uz z) - ux =
        ∑' z, (p z).toReal * (uz z - ux) := by
    -- Expand Exp
    unfold Exp
    -- Replace ux with its expectation under p
    have hconst : ∑' z, (p z).toReal * ux = ux := by
      calc
        ∑' z, (p z).toReal * ux
            = (∑' z, (p z).toReal) * ux := by
                simpa using (tsum_mul_right (f := fun z => (p z).toReal) (a := ux))
        _ = ux := by simp [hsum_p]
    calc
      ∑' z, (p z).toReal * uz z - ux
          = ∑' z, (p z).toReal * uz z - ∑' z, (p z).toReal * ux := by
              rw [hconst]
      _ = ∑' z, ((p z).toReal * uz z - (p z).toReal * ux) := by
              rw [← Summable.tsum_sub hsum_uz hsum_ux]
      _ = ∑' z, (p z).toReal * (uz z - ux) := by
              refine tsum_congr ?_
              intro z
              ring
  -- Apply abs_tsum_le_tsum_abs
  let f : Strings → ℝ := fun z => (p z).toReal * (uz z - ux)
  have hsum_f : Summable f := by
    -- f is difference of summable series
    have hsum_diff : Summable (fun z => (p z).toReal * uz z - (p z).toReal * ux) :=
      hsum_uz.sub hsum_ux
    -- rewrite to match f
    simpa [f, mul_sub] using hsum_diff
  have hsum_abs_f : Summable (fun z => |f z|) := by
    -- Use boundedness via Lipschitz + diameter
    have hLM : 0 ≤ (L : ℝ) * M := mul_nonneg hL_nonneg hM
    have hsum_abs :
        Summable (fun z => (p z).toReal * |uz z - ux|) :=
      PMF.summable_coe_real_mul_of_bounded p (fun z => |uz z - ux|)
        ((L : ℝ) * M) hLM (fun z => by
          simpa [abs_abs] using hLzx_M z)
    simpa [f, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg] using hsum_abs
  have habs : |∑' z, f z| ≤ ∑' z, |f z| :=
    abs_tsum_le_tsum_abs' f hsum_f hsum_abs_f
  -- Bound the RHS by Lipschitz + distortion
  have hsum_LD :
      Summable (fun z => (p z).toReal * ((L : ℝ) * D fstar z x)) := by
    have hLM : 0 ≤ (L : ℝ) * M := mul_nonneg hL_nonneg hM
    exact PMF.summable_coe_real_mul_of_bounded p
      (fun z => (L : ℝ) * D fstar z x) ((L : ℝ) * M) hLM
      (fun z => by
        have hD_nonneg : 0 ≤ D fstar z x := dist_nonneg
        have hD_bound : D fstar z x ≤ M := hdist_bound z
        have h1 : |(L : ℝ) * D fstar z x| = (L : ℝ) * D fstar z x := by
          exact abs_of_nonneg (mul_nonneg hL_nonneg hD_nonneg)
        -- use bound on D
        calc
          |(L : ℝ) * D fstar z x|
              = (L : ℝ) * D fstar z x := h1
          _ ≤ (L : ℝ) * M := mul_le_mul_of_nonneg_left hD_bound hL_nonneg)
  have hLipschitz_sum :
      ∑' z, |f z| ≤ ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x) := by
    -- rewrite |f z| and apply termwise bound
    have hterm :
        ∀ z, |f z| ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
      intro z
      have hnonneg : 0 ≤ (p z).toReal := ENNReal.toReal_nonneg
      have hL' : |uz z - ux| ≤ (L : ℝ) * D fstar z x := hLzx z
      calc
        |f z| = (p z).toReal * |uz z - ux| := by
                simp [f, abs_mul, abs_of_nonneg hnonneg]
        _ ≤ (p z).toReal * ((L : ℝ) * D fstar z x) := by
                exact mul_le_mul_of_nonneg_left hL' hnonneg
    -- Use summable bound
    refine Summable.tsum_le_tsum hterm ?_ hsum_LD
    -- show summable of left side
    simpa [f, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg] using hsum_abs_f
  -- Put everything together
  calc
    |Exp p (fun z => uz z) - ux|
        = |∑' z, f z| := by
            simp [hrewrite, f]
    _ ≤ ∑' z, |f z| := habs
    _ ≤ ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x) := hLipschitz_sum
    _ = (L : ℝ) * ∑' z, (p z).toReal * D fstar z x := by
          -- factor out L
          have hsum_D : Summable (fun z => (p z).toReal * D fstar z x) :=
            summable_D_of_bounded p fstar x M hM hdist_bound
          -- use tsum_mul_left
          calc
            ∑' z, (p z).toReal * ((L : ℝ) * D fstar z x)
                = ∑' z, (L : ℝ) * ((p z).toReal * D fstar z x) := by
                    refine tsum_congr ?_
                    intro z
                    ring
            _ = (L : ℝ) * ∑' z, (p z).toReal * D fstar z x := by
                    simpa using (tsum_mul_left (f := fun z => (p z).toReal * D fstar z x) (a := (L : ℝ)))
    _ = (L : ℝ) * Exp p (fun z => D fstar z x) := by
          rfl

/-- **Utility Transport Bound (Statement)**.

For any Lipschitz oracle utility `u`, the expected utility gap between a
stochastic tree summary and the original document is bounded by expected
oracle distortion. No separability is assumed.

This is the utility-level analogue of the distortion-based bounds used
throughout the OPT theory (DPO/GRPO/PPO).
-/
theorem expected_utility_bound
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) (hx : S T = x) :
    |Egu g (root T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Egu g (root T) (fun z => D fstar z x) := by
  -- Reduce to the PMF form
  simpa [Egu, root] using
    (expected_utility_bound_pmf_bounded (p := reduce g (root T)) (x := x)
      (fstar := fstar) (u := u) (L := L) hL)

/-- Utility transport bound for multi-round summaries (ZR distribution). -/
theorem expected_utility_bound_ZR
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) :
    |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤
      (L : ℝ) * Exp (ZR g x R T) (fun z => D fstar z x) := by
  simpa using
    (expected_utility_bound_pmf_bounded (p := ZR g x R T) (x := x)
      (fstar := fstar) (u := u) (L := L) hL)

/-!
## Corollary: One-Pass Preservation (L1 + L2)
-/

/-- If L1 and L2 hold on a tree, expected oracle utility is preserved (one pass). -/
theorem expected_utility_preserved_one_pass
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) (hx : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
    Egu g (root T) (fun z => u (fstar z) (fstar x)) = u (fstar x) (fstar x) := by
  -- Use expected_utility_bound + one_pass (expected distortion = 0).
  have hdist0 : Egu g (root T) (fun z => D fstar z x) = 0 :=
    one_pass g T x fstar hx h1 h2
  have hgap :
      |Egu g (root T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤ 0 := by
    have h := expected_utility_bound g T x fstar u L hL hx
    simpa [hdist0] using h
  have hgap' : |Egu g (root T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| = 0 := by
    exact le_antisymm hgap (abs_nonneg _)
  have hzero :
      Egu g (root T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x) = 0 := by
    exact abs_eq_zero.mp hgap'
  linarith

/-!
## Corollary: Multi-Round Preservation (L1 + L2 + L3)
-/

/-- If L1, L2, L3 hold, expected oracle utility is preserved after R rounds. -/
theorem expected_utility_preserved_multi_round
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y) (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) (hx : S T = x)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) (hR : R ≥ 1) :
    Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) = u (fstar x) (fstar x) := by
  have hdist0 : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_typeclass g T x R fstar hx h1 h2 h3 hR
  have hgap :
      |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| ≤ 0 := by
    have h := expected_utility_bound_ZR g T x R fstar u L hL
    simpa [hdist0] using h
  have hgap' :
      |Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x)| = 0 := by
    exact le_antisymm hgap (abs_nonneg _)
  have hzero :
      Exp (ZR g x R T) (fun z => u (fstar z) (fstar x)) - u (fstar x) (fstar x) = 0 := by
    exact abs_eq_zero.mp hgap'
  linarith

/-!
## Bridge: Utility Gap as a Unified Preference Gap (Bounded)
-/

/-- Utility gap is a special case of the unified preference gap bound. -/
theorem utility_gap_unified_gap_pure
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (u : OracleUtility2 Y)
    (L : ℝ≥0) (U : ℝ)
    (hL : OracleUtilityLipschitz1 u L)
    (hU : OracleUtilityBoundedAt u (fstar x) U) :
    |u (fstar x) (fstar x) - ∑' z, (p z).toReal * u (fstar z) (fstar x)| ≤
      (L : ℝ) * ∑' z, (p z).toReal * dist (fstar z) (fstar x) := by
  classical
  let E_gen : Strings → ℝ := fun doc => u (fstar doc) (fstar x)
  let μ_X : PMF Strings := PMF.pure x
  let μ_Z : PMF Strings := p
  let D_max : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Y)
  have hD_max : 0 ≤ D_max := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Y)
  have h_dist_bound : ∀ x z, dist (fstar x) (fstar z) ≤ D_max :=
    fun x z => BoundedPseudoMetricSpace.dist_le (fstar x) (fstar z)
  have hE_max : 0 ≤ U := hU.1
  have hE_bound : ∀ x', |E_gen x'| ≤ U := by
    intro x'; simpa [E_gen] using hU.2 (fstar x')
  have h_lip : ∀ x' z, |E_gen x' - E_gen z| ≤ L * dist (fstar x') (fstar z) := by
    intro x' z
    simpa [E_gen] using (hL (fstar x') (fstar z) (fstar x))
  -- Coupling Δ_R specialized to pure(x)
  let Δ_R : ℝ := ∑' z, (μ_Z z).toReal * dist (fstar z) (fstar x)
  have h_Δ :
      Δ_R =
        ∑' z, ∑' x', (μ_Z z).toReal * (μ_X x').toReal * dist (fstar z) (fstar x') := by
    unfold Δ_R μ_X μ_Z
    refine tsum_congr ?_
    intro z
    simp only [PMF.pure_apply]
    rw [tsum_eq_single x]
    · simp
    · intro x' hx'; simp [hx']
  -- Apply unified gap bound
  have h_gap :=
    unified_preference_gap_bounded fstar E_gen μ_X μ_Z L Δ_R D_max hD_max
      h_dist_bound U hE_max hE_bound h_lip h_Δ
  -- Simplify the pure(x) term
  have h_pure : ∑' x', (μ_X x').toReal * E_gen x' = E_gen x := by
    unfold μ_X
    simp only [PMF.pure_apply]
    rw [tsum_eq_single x]
    · simp [E_gen]
    · intro x' hx'; simp [hx', E_gen]
  -- Finish
  simpa [μ_Z, E_gen, h_pure, Δ_R, sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h_gap

end Bounded

section Pseudo

variable [PseudoMetricSpace Y]

/-!
## Bridge: Utility Loss as Expected Group Loss (Lipschitz)
-/

/-- A fixed-truth utility loss is an instance of expected group-loss Lipschitz. -/
lemma expectedGroupLossLipschitz_of_oracleUtility
    {A : Type*} {k : ℕ}
    (g : PMF (Fin k → A)) (fstar : Strings → Y)
    (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) (x z : Strings) :
    ExpectedGroupLossLipschitz
      (loss := fun doc (_ : Fin k → A) => u (fstar doc) (fstar x))
      fstar g L x z := by
  -- Expand the expected loss (group-independent) and apply Lipschitz.
  have hsumg : ∑' grp, (g grp).toReal = 1 := PMF.toReal_tsum_coe g
  have hconst_x :
      ∑' grp, (g grp).toReal * u (fstar x) (fstar x) = u (fstar x) (fstar x) := by
    calc
      ∑' grp, (g grp).toReal * u (fstar x) (fstar x)
          = (∑' grp, (g grp).toReal) * u (fstar x) (fstar x) := by
              simpa using (tsum_mul_right (f := fun grp => (g grp).toReal) (a := u (fstar x) (fstar x)))
      _ = u (fstar x) (fstar x) := by simp [hsumg]
  have hconst_z :
      ∑' grp, (g grp).toReal * u (fstar z) (fstar x) = u (fstar z) (fstar x) := by
    calc
      ∑' grp, (g grp).toReal * u (fstar z) (fstar x)
          = (∑' grp, (g grp).toReal) * u (fstar z) (fstar x) := by
              simpa using (tsum_mul_right (f := fun grp => (g grp).toReal) (a := u (fstar z) (fstar x)))
      _ = u (fstar z) (fstar x) := by simp [hsumg]
  -- Reduce to Lipschitz in first argument
  have hL' : |u (fstar x) (fstar x) - u (fstar z) (fstar x)| ≤
      (L : ℝ) * dist (fstar x) (fstar z) := by
    simpa [dist_comm] using (hL (fstar x) (fstar z) (fstar x))
  -- Assemble
  unfold ExpectedGroupLossLipschitz RUM.ExpectedGroupLossLipschitz
  simp [hconst_x, hconst_z, hL']

end Pseudo

end
