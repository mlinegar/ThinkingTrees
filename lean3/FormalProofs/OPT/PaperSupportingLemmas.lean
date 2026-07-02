import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.TheoremBackingConsequences
import Mathlib.Order.Interval.Finset.Fin

/-!
# FormalProofs/OPT/PaperSupportingLemmas.lean

## Lean Backing for the Paper's Appendix-B Supporting Lemmas and the Sketch Boundary

**Paper Reference:** Appendix B, "Supporting Lemmas for the Gap Bound"
(`app:supporting-lemmas`) and "Sketch Sufficiency Boundary" (`app:proof-m-lt-k`):

- `lem:sigmoid-lip` — the sigmoid `σ(t) = (1 + e^{-t})⁻¹` is 1-Lipschitz;
- `lem:neglogsig-lip` — `t ↦ -log σ(t)` is 1-Lipschitz;
- `lem:dpo-lip` — the pointwise DPO loss is `2|β|·L_pol`-Lipschitz in oracle distance;
- `lem:dpo-oracle-meas` — the expected DPO loss factors through the oracle `f*`;
- `lem:zero-dist-support` — zero expected distortion forces pointwise-zero
  distortion on the support of the summary distribution;
- `prop:m_lt_k` — the failure boundary for top-`m` sketches against the
  threshold target `τ_k` when `m < k`.

This module is a thin **completeness layer**: it gives each of these six
LaTeX-only items a named, paper-facing Lean theorem so the Appendix E
crosswalk can cite them.  Where an exact equivalent already exists in the
repository, the paper-facing name is a documented re-export of the existing
theorem (no duplication of proofs):

| Paper label | Here | Backing |
|---|---|---|
| lem:sigmoid-lip | `paper_sigmoid_lipschitz` (+ abs corollary) | re-export of `sigmoid_lipschitz` (`OPT/PreferenceBounds.lean`) |
| lem:neglogsig-lip | `paper_neg_log_sigmoid_lipschitz` (+ abs corollary) | re-export of `neg_log_sigmoid_lipschitz` (`OPT/PreferenceBounds.lean`) |
| lem:dpo-lip | `paper_dpo_loss_pointwise_lipschitz` | re-export of `dpo_loss_pointwise_lipschitz` (`OPT/PreferenceBounds.lean`); abstract two-score form `paper_neg_log_sigmoid_comp_lipschitz` is new |
| lem:dpo-oracle-meas | `paper_dpo_loss_oracle_measurable` | re-export of `dpo_loss_oracle_measurable` (`OPT/PreferenceLearning.lean`); the expected-loss factorization `paper_expected_dpo_loss_factors_through_oracle` is new glue |
| lem:zero-dist-support | `paper_zero_dist_support` / `paper_zero_dist_support_ZR` | re-export of `FormalProofs.OPT.dist_zero_on_support_of_Exp_zero` (`OPT/TheoremBackingConsequences.lean`); general nonnegative-`φ` form `paper_zero_on_support_of_Exp_eq_zero` is new |
| prop:m_lt_k | `paper_m_lt_k_sketch_state_collision`, `paper_m_lt_k_no_estimator` | new (this file) |

The only genuinely new mathematical content is the `prop:m_lt_k`
formalization; everything else is either a re-export or a small composition
of already-proved lemmas.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped NNReal

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-!
## Paper: lem:sigmoid-lip — the sigmoid is 1-Lipschitz

The paper's `σ(t) = 1/(1 + e^{-t})` is mathlib's `Real.sigmoid`
(`(1 + exp (-t))⁻¹`), the same function used by the repository's DPO stack.
-/

/-- **Paper: lem:sigmoid-lip.** The sigmoid `σ(t) = (1 + e^{-t})⁻¹` is
1-Lipschitz.  Re-export of `sigmoid_lipschitz` from
`OPT/PreferenceBounds.lean` (proved there via `|σ'(t)| = σ(t)(1-σ(t)) ≤ 1/4`). -/
theorem paper_sigmoid_lipschitz : LipschitzWith 1 Real.sigmoid :=
  sigmoid_lipschitz

/-- **Paper: lem:sigmoid-lip**, `dist`-free form:
`|σ(t₁) - σ(t₂)| ≤ |t₁ - t₂|` for all reals.  Direct corollary of
`paper_sigmoid_lipschitz`. -/
theorem paper_sigmoid_abs_sub_le (t₁ t₂ : ℝ) :
    |Real.sigmoid t₁ - Real.sigmoid t₂| ≤ |t₁ - t₂| := by
  have h := sigmoid_lipschitz.dist_le_mul t₁ t₂
  simpa [Real.dist_eq] using h

/-!
## Paper: lem:neglogsig-lip — `-log ∘ σ` is 1-Lipschitz
-/

/-- **Paper: lem:neglogsig-lip.** The map `t ↦ -log σ(t)` is 1-Lipschitz.
Re-export of `neg_log_sigmoid_lipschitz` from `OPT/PreferenceBounds.lean`
(proved there via `|d/dt(-log σ(t))| = |σ(t) - 1| < 1`). -/
theorem paper_neg_log_sigmoid_lipschitz :
    LipschitzWith 1 (fun t => -Real.log (Real.sigmoid t)) :=
  neg_log_sigmoid_lipschitz

/-- **Paper: lem:neglogsig-lip**, `dist`-free form:
`|(-log σ)(t₁) - (-log σ)(t₂)| ≤ |t₁ - t₂|`. -/
theorem paper_neg_log_sigmoid_abs_sub_le (t₁ t₂ : ℝ) :
    |(-Real.log (Real.sigmoid t₁)) - (-Real.log (Real.sigmoid t₂))| ≤ |t₁ - t₂| := by
  have h := neg_log_sigmoid_lipschitz.dist_le_mul t₁ t₂
  simp only [Real.dist_eq] at h
  calc |(-Real.log (Real.sigmoid t₁)) - (-Real.log (Real.sigmoid t₂))|
      ≤ (1 : ℝ≥0) * |t₁ - t₂| := h
    _ = |t₁ - t₂| := by simp

/-!
## Paper: lem:dpo-lip — the pointwise DPO loss is Lipschitz in oracle distance

Two forms are provided.  The abstract form takes two arbitrary score
functions `g_w g_l : Strings → ℝ` (the paper's per-action policy log-ratios
`x ↦ log(π_θ(a|x)/π_ref(a|x))` for the preferred/dispreferred actions), each
`L`-Lipschitz in the oracle pseudometric `dist (f* x) (f* x')`, and bounds
the composite `-log σ(β·(g_w - g_l))`.  The DPO-named instantiation is a
re-export of the existing `dpo_loss_pointwise_lipschitz`.
-/

/-- **Paper: lem:dpo-lip (abstract form).** If two score functions `g_w`,
`g_l` are each `L`-Lipschitz in the oracle pseudometric
`d(x, x') = dist (f* x) (f* x')`, then the pointwise DPO-style loss
`x ↦ -log σ(β·(g_w x - g_l x))` is `2|β|·L`-Lipschitz in that pseudometric.
The proof composes the triangle inequality on the two score terms with
`lem:neglogsig-lip` (`paper_neg_log_sigmoid_lipschitz`). -/
theorem paper_neg_log_sigmoid_comp_lipschitz {Strings Y : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (g_w g_l : Strings → ℝ) (β L : ℝ)
    (h_w : ∀ x x', |g_w x - g_w x'| ≤ L * dist (fstar x) (fstar x'))
    (h_l : ∀ x x', |g_l x - g_l x'| ≤ L * dist (fstar x) (fstar x'))
    (x x' : Strings) :
    |(-Real.log (Real.sigmoid (β * (g_w x - g_l x)))) -
      (-Real.log (Real.sigmoid (β * (g_w x' - g_l x'))))| ≤
      2 * |β| * L * dist (fstar x) (fstar x') := by
  -- Step 1: `-log ∘ σ` is 1-Lipschitz (lem:neglogsig-lip).
  have h_sig : |(-Real.log (Real.sigmoid (β * (g_w x - g_l x)))) -
      (-Real.log (Real.sigmoid (β * (g_w x' - g_l x'))))| ≤
      |β * (g_w x - g_l x) - β * (g_w x' - g_l x')| := by
    have h := neg_log_sigmoid_lipschitz.dist_le_mul
      (β * (g_w x - g_l x)) (β * (g_w x' - g_l x'))
    simp only [Real.dist_eq] at h
    calc |(-Real.log (Real.sigmoid (β * (g_w x - g_l x)))) -
        (-Real.log (Real.sigmoid (β * (g_w x' - g_l x'))))|
        ≤ (1 : ℝ≥0) * |β * (g_w x - g_l x) - β * (g_w x' - g_l x')| := h
      _ = |β * (g_w x - g_l x) - β * (g_w x' - g_l x')| := by simp
  -- Step 2: triangle inequality on the two score differences.
  have h_split : β * (g_w x - g_l x) - β * (g_w x' - g_l x') =
      β * ((g_w x - g_w x') - (g_l x - g_l x')) := by ring
  calc |(-Real.log (Real.sigmoid (β * (g_w x - g_l x)))) -
      (-Real.log (Real.sigmoid (β * (g_w x' - g_l x'))))|
      ≤ |β * (g_w x - g_l x) - β * (g_w x' - g_l x')| := h_sig
    _ = |β| * |(g_w x - g_w x') - (g_l x - g_l x')| := by rw [h_split, abs_mul]
    _ ≤ |β| * (|g_w x - g_w x'| + |g_l x - g_l x'|) :=
        mul_le_mul_of_nonneg_left (abs_sub _ _) (abs_nonneg β)
    _ ≤ |β| * (L * dist (fstar x) (fstar x') + L * dist (fstar x) (fstar x')) :=
        mul_le_mul_of_nonneg_left (add_le_add (h_w x x') (h_l x x')) (abs_nonneg β)
    _ = 2 * |β| * L * dist (fstar x) (fstar x') := by ring

/-- **Paper: lem:dpo-lip (DPO instantiation).** If the policy log-ratios are
`L_pol`-Lipschitz in oracle distance (`PolicyLipschitz`), the pointwise DPO
loss is `2|β|·L_pol`-Lipschitz.  Re-export of `dpo_loss_pointwise_lipschitz`
from `OPT/PreferenceBounds.lean`. -/
theorem paper_dpo_loss_pointwise_lipschitz {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {fstar : Strings → Y} {β : ℝ} {L_pol : ℝ≥0}
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol) (a_w a_ℓ : A) :
    ∀ x x', |DPOLossPointwise pol pol_ref β x a_w a_ℓ -
        DPOLossPointwise pol pol_ref β x' a_w a_ℓ| ≤
      2 * |β| * L_pol * dist (fstar x) (fstar x') :=
  dpo_loss_pointwise_lipschitz h_lip a_w a_ℓ

/-!
## Paper: lem:dpo-oracle-meas — the expected DPO loss factors through `f*`

The paper's hypotheses are (i) both policies oracle-measurable
(`DPO.OracleMeasurable`: `dist (f* x) (f* x') = 0` implies equal policy
values) and (ii) the pair generator oracle-indexed (`OracleIndexedPairGen`).
The conclusion is that the expected DPO loss
`E_{(a_w,a_l) ~ gen(x)}[ℓ_DPO(x, a_w, a_l)]` depends on `x` only through
`f*(x)`, stated in the repository's pointwise convention: oracle-equivalent
documents receive equal expected losses.
-/

/-- **Paper: lem:dpo-oracle-meas (pointwise-loss layer).** Oracle-measurable
policies make the pointwise DPO loss oracle-measurable.  Re-export of
`dpo_loss_oracle_measurable` from `OPT/PreferenceLearning.lean`. -/
theorem paper_dpo_loss_oracle_measurable {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (β : ℝ) (fstar : Strings → Y)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar) :
    OracleMeasurableLoss (fun x a_w a_ℓ => DPOLossPointwise pol pol_ref β x a_w a_ℓ) fstar :=
  dpo_loss_oracle_measurable pol pol_ref β fstar h_meas_pol h_meas_ref

/-- **Paper: lem:dpo-oracle-meas.** With oracle-measurable policies and an
oracle-indexed pair generator, the expected DPO loss factors through the
oracle: whenever `dist (f* x) (f* x') = 0`, the generator distributions and
all pointwise losses coincide, so the inner expectations (the integrand of
`ExpectedDPOLoss`) are equal. -/
theorem paper_expected_dpo_loss_factors_through_oracle {Strings A Y : Type*}
    [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (β : ℝ) (fstar : Strings → Y)
    (gen : PairGenerator Strings A)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar)
    {x x' : Strings} (h_dist : dist (fstar x) (fstar x') = 0) :
    ∑' p : A × A, (gen x p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 =
      ∑' p : A × A, (gen x' p).toReal * DPOLossPointwise pol pol_ref β x' p.1 p.2 := by
  have hg : gen x = gen x' := h_gen x x' h_dist
  rw [hg]
  exact tsum_congr fun p => by
    rw [dpo_loss_eq_of_oracle_eq h_meas_pol h_meas_ref h_dist p.1 p.2]

/-!
## Paper: lem:zero-dist-support — zero expected distortion is pointwise on support
-/

/-- **Paper: lem:zero-dist-support (general form).** If a nonnegative function
`φ` has zero expectation under a PMF `μ` (and the weighted series is
summable, which rules out Lean's `tsum = 0` convention for non-summable
series), then `φ` vanishes at every point of `μ.support`. -/
theorem paper_zero_on_support_of_Exp_eq_zero {α : Type*} (μ : PMF α) (φ : α → ℝ)
    (hφ : ∀ z, 0 ≤ φ z)
    (hsum : Summable (fun z => (μ z).toReal * φ z))
    (h : Exp μ φ = 0) :
    ∀ z ∈ μ.support, φ z = 0 := by
  have h_term : ∀ z, (μ z).toReal * φ z = 0 :=
    tsum_eq_zero_of_nonneg _
      (fun z => mul_nonneg ENNReal.toReal_nonneg (hφ z)) hsum
      (by simpa [Exp] using h)
  intro z hz
  have hz_ne0 : μ z ≠ 0 := by simpa [PMF.mem_support_iff] using hz
  have hz_pos : 0 < (μ z).toReal :=
    ENNReal.toReal_pos hz_ne0 (PMF.apply_ne_top μ z)
  rcases mul_eq_zero.mp (h_term z) with h0 | h0
  · exact absurd h0 (ne_of_gt hz_pos)
  · exact h0

/-- **Paper: lem:zero-dist-support (distortion specialization).** With
`φ z = dist (f* z, f* x)` (`= D fstar z x`) and a bounded oracle metric
(which supplies summability), zero expected distortion forces
`dist (f* z, f* x) = 0` at every support point.  Re-export of
`FormalProofs.OPT.dist_zero_on_support_of_Exp_zero` from
`OPT/TheoremBackingConsequences.lean`. -/
theorem paper_zero_dist_support {Strings Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (μ : PMF Strings) (fstar : Strings → Y) (x : Strings)
    (h : Exp μ (fun z => D fstar z x) = 0) :
    ∀ z ∈ μ.support, dist (fstar z) (fstar x) = 0 :=
  dist_zero_on_support_of_Exp_zero μ fstar x h

/-- **Paper: lem:zero-dist-support (ZR form).** The paper instantiates `μ` as
the distribution of the multi-round summary `Z^{(R)}(x)`; this is that
instantiation verbatim. -/
theorem paper_zero_dist_support_ZR {Strings Y : Type*} [Monoid Strings] [BoundedMetricSpace Y]
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (fstar : Strings → Y)
    (h : Exp (ZR g x R T) (fun z => D fstar z x) = 0) :
    ∀ z ∈ (ZR g x R T).support, dist (fstar z) (fstar x) = 0 :=
  dist_zero_on_support_of_Exp_zero (ZR g x R T) fstar x h

/-!
## Paper: prop:m_lt_k — failure boundary for top-`m` sketches

The paper's proposition: for the threshold target `τ_k(r) = 1{C(r) ≥ k}` on
binary indicators `r ∈ {0,1}^n` with `C(r) = Σ_j r_j`, a top-`m` sketch with
`m < k` admits inputs with identical sketch state but different target
values, so no estimator on the sketch state can compute `τ_k`.

**Concrete faithful model.** A top-`m` sketch of binary indicators (the
CMS/heavy-hitters-style "retain the `m` largest entries" state of the
paper's mechanism discussion) keeps `min(m, C(r))` unit entries: when the
document has at least `m` ones, all `m` retained slots are ones and every
excess one is discarded, so the retained state is determined by
`min(m, C(r))` alone.  We therefore model the sketch state as
`topMState m r := min m (C r)`.  The paper's witnesses are used verbatim:
`r⁻` with exactly `m` leading ones and `r⁺` with exactly `k` leading ones
(both of common length `n ≥ k`) satisfy `S_m(r⁻) = S_m(r⁺) = m` while
`τ_k(r⁻) = 0 ≠ 1 = τ_k(r⁺)`.  The paper's side condition `k ≥ 1` is implied
by `m < k` for natural `m`.
-/

/-- Number of unit indicators: the paper's `C(r) = Σ_j r_j` for `r ∈ {0,1}^n`. -/
def onesCount {n : ℕ} (r : Fin n → Bool) : ℕ :=
  (Finset.univ.filter fun j => r j = true).card

/-- Threshold target `τ_k(r) = 1{C(r) ≥ k}` (paper: prop:m_lt_k). -/
def thresholdTarget (k : ℕ) {n : ℕ} (r : Fin n → Bool) : Bool :=
  decide (k ≤ onesCount r)

/-- Top-`m` sketch state for binary indicators: retains `min(m, C(r))` unit
entries.  This is the state a top-`m` retention sketch actually exposes on
`{0,1}`-valued inputs (see the section docstring). -/
def topMState (m : ℕ) {n : ℕ} (r : Fin n → Bool) : ℕ :=
  min m (onesCount r)

/-- The paper's witness family: the document with exactly `c` leading unit
indicators, `(1,…,1,0,…,0)` with `c` ones. -/
def prefixOnes (n c : ℕ) : Fin n → Bool :=
  fun j => decide ((j : ℕ) < c)

/-- `prefixOnes n c` has exactly `c` unit indicators when `c ≤ n`. -/
lemma onesCount_prefixOnes {n c : ℕ} (hc : c ≤ n) :
    onesCount (prefixOnes n c) = c := by
  unfold onesCount prefixOnes
  rcases eq_or_lt_of_le hc with rfl | hlt
  · -- `c = n`: every index qualifies.
    have huniv : (Finset.univ.filter fun j : Fin c => decide ((j : ℕ) < c) = true) =
        Finset.univ := by
      apply Finset.filter_true_of_mem
      intro j _
      exact decide_eq_true j.isLt
    rw [huniv, Finset.card_univ, Fintype.card_fin]
  · -- `c < n`: the qualifying indices are exactly `Finset.Iio ⟨c, hlt⟩`.
    have hIio : (Finset.univ.filter fun j : Fin n => decide ((j : ℕ) < c) = true) =
        Finset.Iio (⟨c, hlt⟩ : Fin n) := by
      ext j
      simp [Fin.lt_def]
    rw [hIio]
    simp

/-- **Paper: prop:m_lt_k (state collision).** For `m < k ≤ n` there exist two
documents with identical top-`m` sketch state but different threshold-target
values: the paper's `r⁻` (exactly `m` ones) and `r⁺` (exactly `k` ones). -/
theorem paper_m_lt_k_sketch_state_collision {n k m : ℕ}
    (hmk : m < k) (hkn : k ≤ n) :
    ∃ r r' : Fin n → Bool,
      topMState m r = topMState m r' ∧
        thresholdTarget k r ≠ thresholdTarget k r' := by
  have hmn : m ≤ n := (hmk.le).trans hkn
  refine ⟨prefixOnes n m, prefixOnes n k, ?_, ?_⟩
  · -- Both sketch states equal `m`: `min m m = m = min m k` since `m ≤ k`.
    unfold topMState
    rw [onesCount_prefixOnes hmn, onesCount_prefixOnes hkn, min_self,
      min_eq_left hmk.le]
  · -- Targets differ: `τ_k(r⁻) = 1{k ≤ m} = 0` but `τ_k(r⁺) = 1{k ≤ k} = 1`.
    have h1 : thresholdTarget k (prefixOnes n m) = false := by
      unfold thresholdTarget
      rw [onesCount_prefixOnes hmn]
      simp only [decide_eq_false_iff_not]
      omega
    have h2 : thresholdTarget k (prefixOnes n k) = true := by
      unfold thresholdTarget
      rw [onesCount_prefixOnes hkn]
      simp
    rw [h1, h2]
    exact Bool.false_ne_true

/-- **Paper: prop:m_lt_k (no-estimator corollary).** No estimator reading
only the top-`m` sketch state can compute the threshold target `τ_k` when
`m < k ≤ n`: any candidate `est` disagrees with `τ_k` on one of the two
collision witnesses.  This is the paper's "no estimator restricted to that
sketch class can recover `τ_k` without ambiguity," and the structural
(more-data-cannot-fix-it) failure mode motivating the local-law audit. -/
theorem paper_m_lt_k_no_estimator {n k m : ℕ} (hmk : m < k) (hkn : k ≤ n) :
    ∀ est : ℕ → Bool,
      ¬ (∀ r : Fin n → Bool, est (topMState m r) = thresholdTarget k r) := by
  intro est hest
  obtain ⟨r, r', hstate, htarget⟩ := paper_m_lt_k_sketch_state_collision hmk hkn
  apply htarget
  rw [← hest r, ← hest r', hstate]

end FormalProofs.OPT
