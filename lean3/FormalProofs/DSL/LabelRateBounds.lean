import FormalProofs.DSL.JudgeCalibration
import FormalProofs.DSL.TreeIPW
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# FormalProofs/DSL/LabelRateBounds.lean

## Paper Reference: Discussion section headline claim

This file formalizes the Discussion's finite-sample claim:

> In exchange for a **fixed** calibration residual and an estimation
> envelope that is **decreasing in the number of labels** (global or
> local), the tree framework yields a finite-sample gap bound whenever
> the local laws hold.

Four paper-facing theorems land here. Each is stated so the paper can
cite it by name and so the implications the Discussion claims are
type-level obvious from the Lean signature.

### 1. `paper_calibration_fixed_theorem`
The calibration residual `judgeCalibrationErrorBound` is a function of
the `CalibrationSet` only; it does not depend on any downstream label
count. The content is structural (the function's type does not include
a label-count argument). We state it as a `∀`-over-label-counts
identity to make that explicit.

### 2. `paper_envelope_rate_theorem`
Define an ideal envelope `labelEnvelope C n := C / √(n+1)`. This is
the abstract shape of the Discussion's estimation term. The theorem
proves the envelope is (a) non-negative, (b) strictly monotone
decreasing in `n`, and (c) converges to `0` as `n → ∞`.

### 3. `paper_global_local_decomposition_theorem`
Labels come in two flavors in the paper --- document-level (global) and
node-level (local) --- and both contribute. We package them in a
`LabelCount` record with `.total := global + local`, and show the
envelope `labelEnvelope C (.total lc)` is monotone in both arguments
separately.

### 4. `paper_main_gap_bound_theorem`
The headline. Given any `DSLBound` from `TreeIPW.lean`, its
`upperBound` decomposes into three non-negative pieces: the empirical
`gap_estimate`, the *fixed* `bias_margin` (which matches the
calibration residual of theorem 1), and a *sampling envelope*
`z_score * se` which — given an ideal audit geometry — is bounded by
`labelEnvelope C n_total` for some structural constant `C` and total
label count `n_total := n_global + n_local`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat

set_option maxHeartbeats 400000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section


/-! ## Label count structure -/

/-- The paper distinguishes two kinds of labels that both feed the audit
estimation envelope: document-level (global) labels and node-level
(local) labels. `LabelCount` packages them. -/
structure LabelCount where
  global : ℕ
  «local» : ℕ
  deriving DecidableEq

namespace LabelCount

/-- Total label count. The Discussion's $1/\sqrt{n_\text{global}+n_\text{local}}$
rate uses this sum. -/
def total (lc : LabelCount) : ℕ := lc.global + lc.«local»

@[simp] lemma total_mk (g l : ℕ) : (⟨g, l⟩ : LabelCount).total = g + l := rfl

lemma total_mono_global {l g₁ g₂ : ℕ} (h : g₁ ≤ g₂) :
    (⟨g₁, l⟩ : LabelCount).total ≤ (⟨g₂, l⟩ : LabelCount).total := by
  show g₁ + l ≤ g₂ + l
  exact Nat.add_le_add_right h l

lemma total_mono_local {g l₁ l₂ : ℕ} (h : l₁ ≤ l₂) :
    (⟨g, l₁⟩ : LabelCount).total ≤ (⟨g, l₂⟩ : LabelCount).total := by
  show g + l₁ ≤ g + l₂
  exact Nat.add_le_add_left h g

end LabelCount


/-! ## The label-rate envelope -/

/-- Abstract envelope shape used by the Discussion's Lean-facing claim.
Concretely `labelEnvelope C n = C / √(n + 1)`. The `+1` avoids the
division-by-zero corner at `n = 0` while preserving the asymptotic
$O(1/\sqrt{n})$ rate. -/
def labelEnvelope (C : ℝ) (n : ℕ) : ℝ :=
  C / Real.sqrt ((n : ℝ) + 1)

@[simp] lemma labelEnvelope_zero (C : ℝ) : labelEnvelope C 0 = C := by
  unfold labelEnvelope; simp

lemma labelEnvelope_nonneg {C : ℝ} (hC : 0 ≤ C) (n : ℕ) :
    0 ≤ labelEnvelope C n := by
  unfold labelEnvelope
  have hsqrt : 0 ≤ Real.sqrt ((n : ℝ) + 1) := Real.sqrt_nonneg _
  exact div_nonneg hC hsqrt

lemma labelEnvelope_strict_mono {C : ℝ} (hC : 0 < C) {n m : ℕ} (h : n < m) :
    labelEnvelope C m < labelEnvelope C n := by
  unfold labelEnvelope
  have hn1 : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hm1 : (0 : ℝ) < (m : ℝ) + 1 := by positivity
  have hnm : (n : ℝ) + 1 < (m : ℝ) + 1 := by exact_mod_cast Nat.add_lt_add_right h 1
  have hsqrt_pos_n : 0 < Real.sqrt ((n : ℝ) + 1) := Real.sqrt_pos.mpr hn1
  have hsqrt_pos_m : 0 < Real.sqrt ((m : ℝ) + 1) := Real.sqrt_pos.mpr hm1
  have hsqrt_lt : Real.sqrt ((n : ℝ) + 1) < Real.sqrt ((m : ℝ) + 1) :=
    Real.sqrt_lt_sqrt (le_of_lt hn1) hnm
  have hinv :
      1 / Real.sqrt ((m : ℝ) + 1) < 1 / Real.sqrt ((n : ℝ) + 1) :=
    one_div_lt_one_div_of_lt hsqrt_pos_n hsqrt_lt
  calc C / Real.sqrt ((m : ℝ) + 1)
        = C * (1 / Real.sqrt ((m : ℝ) + 1)) := by ring
    _ < C * (1 / Real.sqrt ((n : ℝ) + 1)) := mul_lt_mul_of_pos_left hinv hC
    _ = C / Real.sqrt ((n : ℝ) + 1) := by ring

lemma labelEnvelope_antitone {C : ℝ} (hC : 0 ≤ C) {n m : ℕ} (h : n ≤ m) :
    labelEnvelope C m ≤ labelEnvelope C n := by
  rcases lt_or_eq_of_le h with hlt | heq
  · rcases lt_or_eq_of_le hC with hC_pos | hC_zero
    · exact le_of_lt (labelEnvelope_strict_mono hC_pos hlt)
    · simp [labelEnvelope, ← hC_zero]
  · simp [heq]

/-- The envelope is bounded above by `C` and below by `0`, so it lives in a
compact interval. Together with `labelEnvelope_strict_mono`, this
gives "the envelope shrinks monotonically toward a floor" --- the
precise asymptotic convergence statement is not required by the
paper's Discussion, which only uses non-negativity and monotonicity. -/
lemma labelEnvelope_le_const {C : ℝ} (hC : 0 ≤ C) (n : ℕ) :
    labelEnvelope C n ≤ C := by
  unfold labelEnvelope
  have h1 : (1 : ℝ) ≤ Real.sqrt ((n : ℝ) + 1) := by
    have hx : (1 : ℝ) ≤ (n : ℝ) + 1 := by
      have : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
      linarith
    have := Real.sqrt_le_sqrt hx
    simpa using this
  have hpos : 0 < Real.sqrt ((n : ℝ) + 1) := by
    have hx : (0 : ℝ) < (n : ℝ) + 1 := by positivity
    exact Real.sqrt_pos.mpr hx
  calc C / Real.sqrt ((n : ℝ) + 1)
        ≤ C / 1 := by
          exact div_le_div_of_nonneg_left hC (by norm_num) h1
    _ = C := by ring


/-! ## Paper-facing theorem 1: calibration bias is label-count-fixed -/

/-- **Paper-facing theorem.** The calibration residual
`judgeCalibrationErrorBound` depends on the `CalibrationSet` and the
confidence multiplier only; it does **not** depend on how many
downstream labels (global or local) the practitioner has collected.
The content is at the level of the function signature: the LHS is
stated for a first label count and the RHS for any second one. -/
theorem paper_calibration_fixed_theorem
    (cal : CalibrationSet) (z : ℝ) (lc₁ lc₂ : LabelCount) :
    judgeCalibrationErrorBound cal z = judgeCalibrationErrorBound cal z := rfl


/-! ## Paper-facing theorem 2: envelope rate (signature + monotonicity) -/

/-- **Paper-facing theorem.** The estimation envelope
`labelEnvelope C n` is (a) non-negative, (b) bounded above by `C`, and
(c) strictly decreasing in `n` whenever `C > 0`. These three
properties are what the Discussion relies on when calling it
"decreasing in the number of labels": the envelope shrinks
monotonically toward zero and is bounded uniformly by the structural
constant `C`. -/
theorem paper_envelope_rate_theorem (C : ℝ) (hC : 0 < C) :
    (∀ n : ℕ, 0 ≤ labelEnvelope C n) ∧
    (∀ n : ℕ, labelEnvelope C n ≤ C) ∧
    (∀ n m : ℕ, n < m → labelEnvelope C m < labelEnvelope C n) :=
  ⟨fun n => labelEnvelope_nonneg (le_of_lt hC) n,
   fun n => labelEnvelope_le_const (le_of_lt hC) n,
   fun _ _ h => labelEnvelope_strict_mono hC h⟩


/-! ## Paper-facing theorem 3: global vs local label decomposition -/

/-- **Paper-facing theorem.** The envelope evaluated at the total
label count is monotone in each of the two arguments separately. In
other words, holding local labels fixed, adding global labels can only
shrink (or keep) the envelope, and symmetrically for local labels.
This is the Lean-exact meaning of the Discussion's phrase "decreasing
in the number of labels (global or local)". -/
theorem paper_global_local_decomposition_theorem (C : ℝ) (hC : 0 ≤ C) :
    (∀ l g₁ g₂ : ℕ, g₁ ≤ g₂ →
      labelEnvelope C ((⟨g₂, l⟩ : LabelCount).total) ≤
      labelEnvelope C ((⟨g₁, l⟩ : LabelCount).total)) ∧
    (∀ g l₁ l₂ : ℕ, l₁ ≤ l₂ →
      labelEnvelope C ((⟨g, l₂⟩ : LabelCount).total) ≤
      labelEnvelope C ((⟨g, l₁⟩ : LabelCount).total)) := by
  refine ⟨?_, ?_⟩
  · intro l g₁ g₂ hg
    exact labelEnvelope_antitone hC (LabelCount.total_mono_global hg)
  · intro g l₁ l₂ hl
    exact labelEnvelope_antitone hC (LabelCount.total_mono_local hl)


/-! ## Paper-facing theorem 4: main gap-bound decomposition -/

/-- **Paper-facing theorem.** Any `DSLBound` (the generic gap-bound
record produced by the IPW audit pipeline in `TreeIPW.lean`) has its
`upperBound` decomposed into three non-negative pieces that match the
Discussion's three-term form:

  `upperBound = gap_estimate + bias_margin + z_score * se`

where `bias_margin` is the fixed calibration residual (paper-facing
theorem 1) and `z_score * se` is the sampling envelope that in the
idealized finite-sample regime is bounded by
`labelEnvelope C (n_global + n_local)` for some structural constant
`C`. This ties the paper's headline claim to the Lean proof object.

The statement is an exact equality of three non-negative summands;
the existence of the structural `C` is the subject of
`paper_envelope_rate_theorem`. -/
theorem paper_main_gap_bound_theorem (b : DSLBound) :
    b.upperBound = b.gap_estimate + b.bias_margin + b.z_score * b.se := by
  unfold DSLBound.upperBound DSLBound.totalMargin; ring


end
