import FormalProofs.OPT.AgarwalNesting
import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.NeuralOperatorSpaces
import FormalProofs.OPT.NeuralOperatorPreferenceBridge
import FormalProofs.OPT.PreservationTheorems
import FormalProofs.OPT.TheoremBackingStructure
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.DSL.TreeIPW
import FormalProofs.DSL.LabelRateBounds

/-!
# FormalProofs/OPT/LocalLawNesting.lean

Paper-facing theorem names for the claim that mergeable summaries nest inside
the C-TreePO local-law interface.

There are two related, but distinct, statements.

* Same-tree inclusion: for a fixed C-Tree topology `T`, a sketch/codec whose
  leaves, merges, and summary compatibility are exact supplies a
  `LocalLawsBundle` on that same `T`.
* Schedule bridge: if the local laws hold on two tree topologies with the same
  leaves, C-TreePO gives the classical schedule-invariance conclusion at the
  oracle/readout level.  This is not a byte-equality theorem for states.

State-level Agarwal summaries are handled relationally: validity is propagated
up a `MergeTree`, and readout correctness is applied only at the root.
-/

set_option linter.mathlibStandardSet false
set_option linter.unusedSectionVars false
set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

open ML.MergeableSummary
open MeasureTheory

namespace FormalProofs.OPT

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-! ## C-Tree validity recovered from local laws -/

/-- A produced C-Tree state `z` is a valid summary for represented input `x`
when it has zero oracle distortion against `x`.  This is the C-TreePO analogue
of Agarwal's "valid `S(D, ε)` summary" relation, with size deliberately
kept out of the predicate. -/
def CTreeSummaryValid (fstar : Strings → Y) (x z : Strings) : Prop :=
  D fstar z x = 0

/-- Epsilon-valid C-Tree summary: the produced state has oracle distortion at
most the stated task tolerance. -/
def CTreeSummaryValidWithin
    (fstar : Strings → Y) (ε : ℝ) (x z : Strings) : Prop :=
  D fstar z x ≤ ε

/-- A summary distribution is exactly valid when every point in its support is
valid for the represented input. -/
def CTreeSummaryPMFValid
    (fstar : Strings → Y) (x : Strings) (p : PMF Strings) : Prop :=
  ∀ z, z ∈ p.support → CTreeSummaryValid fstar x z

/-- Supportwise epsilon-validity: every realized output of the summary
distribution is within the requested task tolerance. -/
def CTreeSummaryPMFValidWithin
    (fstar : Strings → Y) (ε : ℝ) (x : Strings) (p : PMF Strings) : Prop :=
  ∀ z, z ∈ p.support → CTreeSummaryValidWithin fstar ε x z

/-- Expected epsilon-validity: the summary distribution's expected oracle
distortion is bounded by `ε`.  This is the guarantee produced by aggregate
approximate local-law audits; it is weaker than supportwise epsilon-validity. -/
def CTreeSummaryPMFExpectedValidWithin
    (fstar : Strings → Y) (ε : ℝ) (x : Strings) (p : PMF Strings) : Prop :=
  Exp p (fun z => D fstar z x) ≤ ε

/-! ## Agarwal valid-summary sets -/

/-- Agarwal-style valid-summary set for a represented stream `xs`.

The paper's `S(D, ε)` notation is represented in Lean by an explicit validity
relation `valid`; the error parameter may be closed over by that relation. -/
def AgarwalValidSummarySet {α State : Type*}
    (valid : Stream α → State → Prop) (xs : Stream α) : Set State :=
  {s | valid xs s}

/-- Agarwal-style valid-summary set with the explicit `k(|D|, ε)` size profile
included.  Local laws can supply validity; the size bound remains an external
Agarwal hypothesis. -/
def AgarwalValidSizedSummarySet {α State : Type*}
    (valid : Stream α → State → Prop)
    (size : State → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ) (xs : Stream α) : Set State :=
  {s | valid xs s ∧ (size s : ℝ) ≤ profile ε xs.length}

/-- Zero expected distortion for a bounded oracle metric implies the PMF is
supported on exactly valid summaries. -/
theorem ctreeSummaryPMFValid_of_expected_zero
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (fstar : Strings → Y) (x : Strings) (p : PMF Strings)
    (h_exp : Exp p (fun z => D fstar z x) = 0) :
    CTreeSummaryPMFValid fstar x p := by
  let M : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Y)
  have hbound : ∀ z, D fstar z x ≤ M := by
    intro z
    unfold D
    exact BoundedPseudoMetricSpace.dist_le (fstar z) (fstar x)
  have h_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    summable_D_of_bounded (p := p) (fstar := fstar) (x := x) M hM hbound
  have h_term_zero : ∀ z, (p z).toReal * D fstar z x = 0 :=
    tsum_eq_zero_of_nonneg
      (fun z => (p z).toReal * D fstar z x)
      (fun z => mul_nonneg ENNReal.toReal_nonneg dist_nonneg)
      h_summable
      (by simpa [Exp] using h_exp)
  intro z hz
  have hz_ne0 : p z ≠ 0 := by
    simpa [PMF.mem_support_iff] using hz
  have hz_toReal_pos : 0 < (p z).toReal :=
    ENNReal.toReal_pos hz_ne0 (PMF.apply_ne_top p z)
  have hz_mul : (p z).toReal * D fstar z x = 0 := h_term_zero z
  rcases mul_eq_zero.mp hz_mul with hz_toReal | hz_dist
  · exfalso
    exact (ne_of_gt hz_toReal_pos) hz_toReal
  · exact hz_dist

/-- Exact C1 and C3/L2 local laws imply that the one-pass root distribution is
supported on valid summaries for the represented tree input. -/
theorem exactLocalLaws_root_validSummaryPMF
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
    CTreeSummaryPMFValid fstar (S T) (reduce g T) := by
  have h_exp : Exp (reduce g T) (fun z => D fstar z (S T)) = 0 := by
    simpa [Exp, Egu, root] using
      one_pass g T (S T) fstar rfl h1 h2
  exact ctreeSummaryPMFValid_of_expected_zero fstar (S T) (reduce g T) h_exp

/-- Exact C1/C2/C3 local laws imply that every multi-round `ZR` output is a
valid summary for the represented input. -/
theorem exactLocalLaws_multiround_validSummaryPMF
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar) :
    CTreeSummaryPMFValid fstar x (ZR g x R T) := by
  have h_exp : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_typeclass g T x R fstar hp
      laws.law1 laws.law2 laws.law3 hR
  exact ctreeSummaryPMFValid_of_expected_zero fstar x (ZR g x R T) h_exp

/-- Exact root validity gives containment of the C-Tree root support in the
Agarwal valid-summary set, once an external bridge from oracle-validity to the
chosen Agarwal validity relation is supplied. -/
theorem exactLocalLaws_root_support_subset_agarwalValidSummarySet
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    {α : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (valid : Stream α → Strings → Prop) (xs : Stream α)
    (h_to_valid :
      ∀ z, CTreeSummaryValid fstar (S T) z → valid xs z)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
    (reduce g T).support ⊆ AgarwalValidSummarySet valid xs := by
  intro z hz
  have h_ctree : CTreeSummaryValid fstar (S T) z :=
    exactLocalLaws_root_validSummaryPMF g T fstar h1 h2 z hz
  simpa [AgarwalValidSummarySet] using h_to_valid z h_ctree

/-- Multi-round support-containment version for Agarwal valid-summary sets. -/
theorem exactLocalLaws_multiround_support_subset_agarwalValidSummarySet
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    {α : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (valid : Stream α → Strings → Prop) (xs : Stream α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    (h_to_valid : ∀ z, CTreeSummaryValid fstar x z → valid xs z) :
    (ZR g x R T).support ⊆ AgarwalValidSummarySet valid xs := by
  intro z hz
  have h_ctree : CTreeSummaryValid fstar x z :=
    exactLocalLaws_multiround_validSummaryPMF
      g T x R fstar hp hR laws z hz
  simpa [AgarwalValidSummarySet] using h_to_valid z h_ctree

/-- Exact root support-containment in Agarwal's sized `S(D, ε)` set.  The size
profile is not derived from local laws; it is the supplied Agarwal size
hypothesis. -/
theorem exactLocalLaws_root_support_subset_agarwalValidSizedSummarySet
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    {α : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (valid : Stream α → Strings → Prop)
    (size : Strings → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ) (xs : Stream α)
    (h_to_valid :
      ∀ z, CTreeSummaryValid fstar (S T) z → valid xs z)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
    (reduce g T).support ⊆
      AgarwalValidSizedSummarySet valid size profile ε xs := by
  intro z hz
  have h_ctree : CTreeSummaryValid fstar (S T) z :=
    exactLocalLaws_root_validSummaryPMF g T fstar h1 h2 z hz
  have hvalid : valid xs z := h_to_valid z h_ctree
  have hsized :
      valid xs z ∧ (size z : ℝ) ≤ profile ε xs.length :=
    ⟨hvalid, h_size xs z hvalid⟩
  simpa [AgarwalValidSizedSummarySet] using hsized

/-- Multi-round support-containment in Agarwal's sized `S(D, ε)` set. -/
theorem exactLocalLaws_multiround_support_subset_agarwalValidSizedSummarySet
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    {α : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (valid : Stream α → Strings → Prop)
    (size : Strings → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ) (xs : Stream α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    (h_to_valid : ∀ z, CTreeSummaryValid fstar x z → valid xs z)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε) :
    (ZR g x R T).support ⊆
      AgarwalValidSizedSummarySet valid size profile ε xs := by
  intro z hz
  have h_ctree : CTreeSummaryValid fstar x z :=
    exactLocalLaws_multiround_validSummaryPMF
      g T x R fstar hp hR laws z hz
  have hvalid : valid xs z := h_to_valid z h_ctree
  have hsized :
      valid xs z ∧ (size z : ℝ) ≤ profile ε xs.length :=
    ⟨hvalid, h_size xs z hvalid⟩
  simpa [AgarwalValidSizedSummarySet] using hsized

/-- C2/L3 is the stable re-entry law: once a state is in the summarizer's range,
resummarizing it is supported on states that remain valid for that state. -/
theorem exactLocalLaws_resummary_valid
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (fstar : Strings → Y)
    (h3 : L3 g fstar) (Z : Strings) (hZ : InRange g Z) :
    CTreeSummaryPMFValid fstar Z (g Z) := by
  intro z hz
  exact L3_implies_dist_zero_on_support_typeclass
    g fstar h3 Z hZ z hz

/-- Supportwise approximate local laws are strong enough to recover
supportwise epsilon-validity at the one-pass root.  This is intentionally a
stronger premise than the aggregate audit budget. -/
theorem supportwiseApproxLocalLaws_root_validSummaryPMF
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε : ℝ)
    (h_support :
      CTreeSummaryPMFValidWithin fstar ε (S T) (reduce g T)) :
    CTreeSummaryPMFValidWithin fstar ε (S T) (reduce g T) :=
  h_support

/-- Supportwise approximate multi-round validity is a separate hypothesis; it
does not follow from aggregate expected-error certification alone. -/
theorem supportwiseApproxLocalLaws_multiround_validSummaryPMF
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y) (ε : ℝ)
    (h_support :
      CTreeSummaryPMFValidWithin fstar ε x (ZR g x R T)) :
    CTreeSummaryPMFValidWithin fstar ε x (ZR g x R T) :=
  h_support

/-- Aggregate approximate local laws recover expected epsilon-validity with the
composed local-law budget.  No supportwise epsilon-validity is claimed here. -/
theorem approxLocalLaws_expectedValidSummaryPMF
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar) :
    CTreeSummaryPMFExpectedValidWithin
      fstar (laws.localLawError R) x (ZR g x R T) :=
  Δ_R_ZR_le_localLawError_of_approx_bundle
    g T fstar x R hp hR hbound hbound_global h_mono laws

/-- If the aggregate approximate local-law bundle is certified at target
`ε`, then the multi-round summary distribution is expected-`ε` valid. -/
theorem approxLocalLaws_certifiedAtEpsilon_expectedValidSummaryPMF
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (x : Strings) (R : ℕ) (ε : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p)
    (laws : ApproxLocalLawsBundle g T fstar)
    (hcert : laws.CertifiedAtEpsilon R ε) :
    CTreeSummaryPMFExpectedValidWithin fstar ε x (ZR g x R T) :=
  Δ_R_ZR_le_of_approx_bundle_certifiedAtEpsilon
    g T fstar x R ε hp hR hbound hbound_global h_mono laws hcert

/-! ## Fixed-epsilon learned NO/FNO route -/

/-- A learned summary is epsilon-good for a fixed tree/input when its root
distribution has expected oracle distortion at most the target `ε`. -/
def EpsilonGoodLearnedSummary
    (g : Summarizer Strings) (T : BinTree Strings)
    (x : Strings) (R : ℕ) (fstar : Strings → Y) (ε : ℝ) : Prop :=
  CTreeSummaryPMFExpectedValidWithin fstar ε x (ZR g x R T)

/-- Fixed-epsilon neural-operator route.

If an exact ideal summarizer in a neural-operator class is uniformly
approximated by a realized neural operator, and the resulting transferred
local-law budget is below the target `εCert`, then the realized C-Tree summary
is expected-`εCert` valid.  This is a certification theorem, not an optimizer
existence theorem. -/
theorem neuralOperator_fixedEpsilon_expectedValidSummaryPMF
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y} {εNO : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar εNO)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p)
    (hcert : H.localLawBudget R ≤ εCert) :
    EpsilonGoodLearnedSummary
      (deterministicSummarizer sApprox) T x R fstar εCert := by
  have hcert_laws :
      H.toApproxLocalLawsBundle.CertifiedAtEpsilon R εCert := by
    simpa [ApproxNeuralOperatorPreferenceBridge.localLawBudget,
      ApproxLocalLawsBundle.CertifiedAtEpsilon,
      ApproxLocalLawsBundle.localLawError] using hcert
  exact approxLocalLaws_certifiedAtEpsilon_expectedValidSummaryPMF
    (deterministicSummarizer sApprox) T fstar x R εCert
    hp hR hbound hbound_global h_mono
    H.toApproxLocalLawsBundle hcert_laws

/-- Fixed-epsilon finite-dimensionalization/FNO route.

The FNO/finite-dimensionalization bridge first produces the same approximate
local-law bundle as the uniform neural-operator route; once its budget is below
`εCert`, the realized tree is expected-`εCert` valid. -/
theorem fno_fixedEpsilon_expectedValidSummaryPMF
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y} {εNO : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar εNO)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p)
    (hcert : H.localLawBudget R ≤ εCert) :
    EpsilonGoodLearnedSummary
      (deterministicSummarizer sApprox) T x R fstar εCert := by
  have hcert_laws :
      H.toApproxLocalLawsBundle.CertifiedAtEpsilon R εCert := by
    simpa [FDNeuralOperatorPreferenceBridge.localLawBudget,
      FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle,
      ApproxLocalLawsBundle.CertifiedAtEpsilon,
      ApproxLocalLawsBundle.localLawError] using hcert
  exact approxLocalLaws_certifiedAtEpsilon_expectedValidSummaryPMF
    (deterministicSummarizer sApprox) T fstar x R εCert
    hp hR hbound hbound_global h_mono
    H.toApproxLocalLawsBundle hcert_laws

/-- Fixed-epsilon calibrated neural-operator route.

When local laws are certified through a learned score `fhat`, the true-oracle
epsilon-good statement follows once the neural-operator local-law budget plus
the two-sided oracle-recovery slack is below the target `εCert`. -/
theorem calibratedNeuralOperator_fixedEpsilon_expectedValidSummaryPMF
    {Yb : Type*} [BoundedMetricSpace Yb]
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar fhat : Strings → Yb}
    {εNO : ℝ} {ε_orc : NNReal}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat εNO)
    (hRec : OracleRecoveredWithin fstar fhat ε_orc)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p)
    (hcert : TotalOracleRecoveryBudget (H.localLawBudget R) ε_orc ≤ εCert) :
    EpsilonGoodLearnedSummary
      (deterministicSummarizer sApprox) T x R fstar εCert := by
  simpa [EpsilonGoodLearnedSummary, CTreeSummaryPMFExpectedValidWithin,
    Δ_R_ZR, OracleRecoveredWithin, TotalOracleRecoveryBudget,
    OracleRecoverySlack] using
    trueOracle_delta_R_ZR_le_epsilon_of_calibrated_neuralOperatorBridge
      (H := H) hRec x R εCert hp hR hbound hbound_global h_mono hcert

/-- Fixed-epsilon calibrated FNO/finite-dimensionalization route. -/
theorem calibratedFNO_fixedEpsilon_expectedValidSummaryPMF
    {Yb : Type*} [BoundedMetricSpace Yb]
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar fhat : Strings → Yb}
    {εNO : ℝ} {ε_orc : NNReal}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat εNO)
    (hRec : OracleRecoveredWithin fstar fhat ε_orc)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p)
    (hcert : TotalOracleRecoveryBudget (H.localLawBudget R) ε_orc ≤ εCert) :
    EpsilonGoodLearnedSummary
      (deterministicSummarizer sApprox) T x R fstar εCert := by
  simpa [EpsilonGoodLearnedSummary, CTreeSummaryPMFExpectedValidWithin,
    Δ_R_ZR, OracleRecoveredWithin, TotalOracleRecoveryBudget,
    OracleRecoverySlack] using
    trueOracle_delta_R_ZR_le_epsilon_of_calibrated_neuralOperatorFDBridge
      (H := H) hRec x R εCert hp hR hbound hbound_global h_mono hcert

/-- Oracle-recovery version of the fixed-epsilon neural-operator route.

If the learned oracle/readout recovers the true oracle within `ε_orc`, then the
realized tree is expected-good at the explicit budget
`TotalOracleRecoveryBudget (localLawBudget) ε_orc`. -/
theorem neuralOperator_oracleRecovery_expectedValidSummaryPMF
    {Yb : Type*} [BoundedMetricSpace Yb]
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar fhat : Strings → Yb}
    {εNO : ℝ} {ε_orc : NNReal}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat εNO)
    (hRec : OracleRecoveredWithin fstar fhat ε_orc)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    EpsilonGoodLearnedSummary
      (deterministicSummarizer sApprox) T x R fstar
      (TotalOracleRecoveryBudget (H.localLawBudget R) ε_orc) := by
  simpa [EpsilonGoodLearnedSummary, CTreeSummaryPMFExpectedValidWithin,
    Δ_R_ZR, OracleRecoveredWithin, TotalOracleRecoveryBudget,
    OracleRecoverySlack] using
    trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge
      (H := H) hRec x R hp hR hbound hbound_global h_mono

/-- Oracle-recovery version of the fixed-epsilon FNO route. -/
theorem fno_oracleRecovery_expectedValidSummaryPMF
    {Yb : Type*} [BoundedMetricSpace Yb]
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar fhat : Strings → Yb}
    {εNO : ℝ} {ε_orc : NNReal}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat εNO)
    (hRec : OracleRecoveredWithin fstar fhat ε_orc)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    EpsilonGoodLearnedSummary
      (deterministicSummarizer sApprox) T x R fstar
      (TotalOracleRecoveryBudget (H.localLawBudget R) ε_orc) := by
  simpa [EpsilonGoodLearnedSummary, CTreeSummaryPMFExpectedValidWithin,
    Δ_R_ZR, OracleRecoveredWithin, TotalOracleRecoveryBudget,
    OracleRecoverySlack] using
    trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge
      (H := H) hRec x R hp hR hbound hbound_global h_mono

/-! ## Expanded DSL/IPW certificates -/

/-- Canonical paper-facing certified tree error: the DSL/IPW upper bound plus
the two-sided oracle-recovery slack. -/
def CertifiedTreeError (b : DSLBound) (ε_orc : NNReal) : ℝ :=
  b.upperBound + OracleRecoverySlack ε_orc

/-- Expanded form of `CertifiedTreeError`, with the sampled local-law estimate,
the existing DSL/IPW calibration and sampling margins, and the oracle-recovery
slack displayed separately.  The paper writes Lean's `b.z_score` as a
non-overloaded confidence multiplier `c_α`. -/
def CertifiedTreeErrorExpanded (b : DSLBound) (ε_orc : NNReal) : ℝ :=
  b.gap_estimate + b.bias_margin + b.z_score * b.se + 2 * (ε_orc : ℝ)

/-- The compact DSL/IPW certificate is exactly the expanded `DSLBound.upperBound`
expression plus oracle-recovery slack. -/
theorem CertifiedTreeError_eq_expanded
    (b : DSLBound) (ε_orc : NNReal) :
    CertifiedTreeError b ε_orc = CertifiedTreeErrorExpanded b ε_orc := by
  rw [CertifiedTreeError, CertifiedTreeErrorExpanded,
    DSLBound.upperBound, DSLBound.totalMargin, OracleRecoverySlack]
  ring

/-- If the observed DSL/IPW standard-error term is bounded by a label-count envelope, the
full certified tree error is bounded by the same expanded expression with the
envelope in place of that term.  This is an interpretive sample-size
corollary, not a learning-rate theorem for the summarizer. -/
theorem CertifiedTreeError_le_labelEnvelope_of_se_le
    (b : DSLBound) (ε_orc : NNReal) (C : ℝ) (n_eff : ℕ)
    (hse : b.z_score * b.se ≤ labelEnvelope C n_eff) :
    CertifiedTreeError b ε_orc ≤
      b.gap_estimate + labelEnvelope C n_eff + b.bias_margin +
        2 * (ε_orc : ℝ) := by
  rw [CertifiedTreeError, DSLBound.upperBound, DSLBound.totalMargin,
    OracleRecoverySlack]
  linarith

/-- Explicit empirical-Bernstein radius for the TreePO union-bound estimator.
This is the appendix alternative to the Wald/SE term in `DSLBound.upperBound`. -/
def UnionBoundEBRadius
    (samples : List TreeSample) (N M R : ℕ)
    (δ_leaf δ_merge δ_idemp : ℝ) : ℝ :=
  (N : ℝ) *
      empiricalBernsteinRadius (toWeightedSamples (leafSamples samples))
        δ_leaf 1
    + (M : ℝ) *
      empiricalBernsteinRadius (toWeightedSamples (mergeSamples samples))
        δ_merge 1
    + ((R - 1 : ℕ) : ℝ) *
      empiricalBernsteinRadius (toWeightedSamples (resummarySamples samples))
        δ_idemp 1

/-- Componentwise empirical-Bernstein events imply a union-bound error event
with the explicit `UnionBoundEBRadius`. -/
theorem ipwUnionBound_empirical_bernstein_from_components_expanded
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List TreeSample)
    (N M R : ℕ)
    (mean_leaf mean_merge mean_idemp : ℝ)
    (δ_leaf δ_merge δ_idemp : ℝ)
    (h_leaf_event :
      μ {ω | |ipwLeafViolationRate (samples ω) - mean_leaf| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (leafSamples (samples ω))) δ_leaf 1}
        ≤ ENNReal.ofReal δ_leaf)
    (h_merge_event :
      μ {ω | |ipwMergeViolationRate (samples ω) - mean_merge| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (mergeSamples (samples ω))) δ_merge 1}
        ≤ ENNReal.ofReal δ_merge)
    (h_idemp_event :
      μ {ω | |ipwIdempViolationRate (samples ω) - mean_idemp| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (resummarySamples (samples ω))) δ_idemp 1}
        ≤ ENNReal.ofReal δ_idemp)
    (hR_one_le : 1 ≤ R)
    (hcoeff : 0 < N ∨ 0 < M ∨ 1 < R) :
    μ {ω |
        |ipwUnionBound (samples ω) N M R -
          ((N : ℝ) * mean_leaf + (M : ℝ) * mean_merge +
            ((R - 1 : ℕ) : ℝ) * mean_idemp)| ≥
          UnionBoundEBRadius (samples ω) N M R
            δ_leaf δ_merge δ_idemp} ≤
      ENNReal.ofReal δ_leaf + ENNReal.ofReal δ_merge +
        ENNReal.ofReal δ_idemp := by
  simpa [UnionBoundEBRadius] using
    (ipwUnionBound_empirical_bernstein_from_components
      (μ := μ) (samples := samples) (N := N) (M := M) (R := R)
      (mean_leaf := mean_leaf) (mean_merge := mean_merge)
      (mean_idemp := mean_idemp) (δ_leaf := δ_leaf)
      (δ_merge := δ_merge) (δ_idemp := δ_idemp)
      h_leaf_event h_merge_event h_idemp_event hR_one_le hcoeff)

/-- Existential fixed-epsilon neural-operator wrapper.

Any learning/approximation result that returns a realized neural operator with a
certified local-law budget below the target immediately yields an epsilon-good
learned C-Tree summary. -/
theorem exists_neuralOperator_fixedEpsilon_goodLearnedSummaryPMF
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y}
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (hlearn : ∃ (sApprox : Strings → Strings) (εNO : ℝ),
      ∃ H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar εNO,
        H.localLawBudget R ≤ εCert ∧
        (∀ p,
          pIdemp (deterministicSummarizer sApprox) fstar
            (p.bind (deterministicSummarizer sApprox)) ≤
          pIdemp (deterministicSummarizer sApprox) fstar p)) :
    ∃ g : Summarizer Strings,
      EpsilonGoodLearnedSummary g T x R fstar εCert := by
  rcases hlearn with ⟨sApprox, εNO, H, hcert, h_mono_sApprox⟩
  exact ⟨deterministicSummarizer sApprox,
    neuralOperator_fixedEpsilon_expectedValidSummaryPMF
      H x R εCert hp hR hbound hbound_global h_mono_sApprox hcert⟩

/-- Existential fixed-epsilon FNO wrapper. -/
theorem exists_fno_fixedEpsilon_goodLearnedSummaryPMF
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y}
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (hlearn : ∃ (sApprox : Strings → Strings) (εNO : ℝ),
      ∃ H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar εNO,
        H.localLawBudget R ≤ εCert ∧
        (∀ p,
          pIdemp (deterministicSummarizer sApprox) fstar
            (p.bind (deterministicSummarizer sApprox)) ≤
          pIdemp (deterministicSummarizer sApprox) fstar p)) :
    ∃ g : Summarizer Strings,
      EpsilonGoodLearnedSummary g T x R fstar εCert := by
  rcases hlearn with ⟨sApprox, εNO, H, hcert, h_mono_sApprox⟩
  exact ⟨deterministicSummarizer sApprox,
    fno_fixedEpsilon_expectedValidSummaryPMF
      H x R εCert hp hR hbound hbound_global h_mono_sApprox hcert⟩

/-- Existential calibrated fixed-epsilon neural-operator wrapper. -/
theorem exists_calibratedNeuralOperator_fixedEpsilon_goodLearnedSummaryPMF
    {Yb : Type*} [BoundedMetricSpace Yb]
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar : Strings → Strings}
    {T : BinTree Strings} {fstar fhat : Strings → Yb}
    {ε_orc : NNReal}
    (hRec : OracleRecoveredWithin fstar fhat ε_orc)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (hlearn : ∃ (sApprox : Strings → Strings) (εNO : ℝ),
      ∃ H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat εNO,
        TotalOracleRecoveryBudget (H.localLawBudget R) ε_orc ≤ εCert ∧
        (∀ p,
          pIdemp (deterministicSummarizer sApprox) fhat
            (p.bind (deterministicSummarizer sApprox)) ≤
          pIdemp (deterministicSummarizer sApprox) fhat p)) :
    ∃ g : Summarizer Strings,
      EpsilonGoodLearnedSummary g T x R fstar εCert := by
  rcases hlearn with ⟨sApprox, εNO, H, hcert, h_mono_sApprox⟩
  exact ⟨deterministicSummarizer sApprox,
    calibratedNeuralOperator_fixedEpsilon_expectedValidSummaryPMF
      H hRec x R εCert hp hR hbound hbound_global h_mono_sApprox hcert⟩

/-- Existential calibrated fixed-epsilon FNO wrapper. -/
theorem exists_calibratedFNO_fixedEpsilon_goodLearnedSummaryPMF
    {Yb : Type*} [BoundedMetricSpace Yb]
    [PseudoMetricSpace Strings]
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar : Strings → Strings}
    {T : BinTree Strings} {fstar fhat : Strings → Yb}
    {ε_orc : NNReal}
    (hRec : OracleRecoveredWithin fstar fhat ε_orc)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (hlearn : ∃ (sApprox : Strings → Strings) (εNO : ℝ),
      ∃ H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat εNO,
        TotalOracleRecoveryBudget (H.localLawBudget R) ε_orc ≤ εCert ∧
        (∀ p,
          pIdemp (deterministicSummarizer sApprox) fhat
            (p.bind (deterministicSummarizer sApprox)) ≤
          pIdemp (deterministicSummarizer sApprox) fhat p)) :
    ∃ g : Summarizer Strings,
      EpsilonGoodLearnedSummary g T x R fstar εCert := by
  rcases hlearn with ⟨sApprox, εNO, H, hcert, h_mono_sApprox⟩
  exact ⟨deterministicSummarizer sApprox,
    calibratedFNO_fixedEpsilon_expectedValidSummaryPMF
      H hRec x R εCert hp hR hbound hbound_global h_mono_sApprox hcert⟩

/-- A root support point certified by exact local laws becomes an Agarwal
`ValidSizedState` only after an external map from oracle-validity to the chosen
state-validity relation and an external size profile are supplied. -/
def exactLocalLaws_root_validSizedState_of_external_sizeProfile
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    {α : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (valid : Stream α → Strings → Prop)
    (size : Strings → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ) (xs : Stream α)
    {z : Strings} (hz : z ∈ (reduce g T).support)
    (h_to_valid : CTreeSummaryValid fstar (S T) z → valid xs z)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε)
    (h1 : L1 g T fstar) (h2 : L2 g T fstar) :
    Agarwal2013Full.ValidSizedState valid size profile ε xs := by
  have h_ctree : CTreeSummaryValid fstar (S T) z :=
    exactLocalLaws_root_validSummaryPMF g T fstar h1 h2 z hz
  have hvalid : valid xs z := h_to_valid h_ctree
  exact ⟨z, hvalid, h_size xs z hvalid⟩

/-- Multi-round version of the sized wrapper: local laws provide oracle/readout
validity; the Agarwal size guarantee still comes only from the supplied
profile hypothesis. -/
def exactLocalLaws_multiround_validSizedState_of_external_sizeProfile
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    {α : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (valid : Stream α → Strings → Prop)
    (size : Strings → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ) (xs : Stream α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    {z : Strings} (hz : z ∈ (ZR g x R T).support)
    (h_to_valid : CTreeSummaryValid fstar x z → valid xs z)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε) :
    Agarwal2013Full.ValidSizedState valid size profile ε xs := by
  have h_ctree : CTreeSummaryValid fstar x z :=
    exactLocalLaws_multiround_validSummaryPMF
      g T x R fstar hp hR laws z hz
  have hvalid : valid xs z := h_to_valid h_ctree
  exact ⟨z, hvalid, h_size xs z hvalid⟩

/-! ## Exact equivalence theorem -/

/-- Exact readout equivalence between the two presentations.

If an exact C-Tree and a classical state-level mergeable summary represent the
same input, then every C-Tree multi-round support point has the same oracle
value as the classical merged root readout.  This is the strongest equivalence
available without an additional invertible codec/state-identity hypothesis. -/
theorem exactLocalLaws_stateLevelReadout_equivalence
    {Y : Type*} [BoundedMetricSpace Y]
    {α State : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (A : StateLevelMergeableSummary α State Y)
    (oracle : Stream α → Y)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query)
    (t : MergeTree α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    (h_same_input : oracle (MergeTree.data t) = fstar x) :
    ∀ z ∈ (ZR g x R T).support,
      fstar z = A.query (MergeTree.eval A.build A.merge t) := by
  intro z hz
  have h_ctree : CTreeSummaryValid fstar x z :=
    exactLocalLaws_multiround_validSummaryPMF
      g T x R fstar hp hR laws z hz
  have h_zx : fstar z = fstar x := by
    exact dist_eq_zero.mp (by simpa [CTreeSummaryValid, D] using h_ctree)
  have h_classical :
      A.query (MergeTree.eval A.build A.merge t) =
        oracle (MergeTree.data t) :=
    stateLevelMergeableSummary_readout_of_mergeTree A oracle h_query t
  calc
    fstar z = fstar x := h_zx
    _ = oracle (MergeTree.data t) := h_same_input.symm
    _ = A.query (MergeTree.eval A.build A.merge t) := h_classical.symm

/-- Set-level exact readout equivalence.

If exact local laws hold and `xs` represents the same input as the C-Tree, then
every realized C-Tree output has the same oracle value as every Agarwal-valid
summary state for `xs`.  This is a statement about the whole valid-summary set,
not only about the particular state obtained from one merge tree. -/
theorem exactLocalLaws_agarwalValidSummarySet_readout_equivalence
    {Y : Type*} [BoundedMetricSpace Y]
    {α State : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (valid : Stream α → State → Prop)
    (query : State → Y)
    (oracle : Stream α → Y)
    (xs : Stream α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    (h_same_input : oracle xs = fstar x)
    (h_query : ∀ xs s, valid xs s → query s = oracle xs) :
    ∀ z ∈ (ZR g x R T).support,
      ∀ s ∈ AgarwalValidSummarySet valid xs,
        fstar z = query s := by
  intro z hz s hs
  have h_ctree : CTreeSummaryValid fstar x z :=
    exactLocalLaws_multiround_validSummaryPMF
      g T x R fstar hp hR laws z hz
  have h_zx : fstar z = fstar x := by
    exact dist_eq_zero.mp (by simpa [CTreeSummaryValid, D] using h_ctree)
  have hs_valid : valid xs s := by
    simpa [AgarwalValidSummarySet] using hs
  have h_classical : query s = oracle xs :=
    h_query xs s hs_valid
  calc
    fstar z = fstar x := h_zx
    _ = oracle xs := h_same_input.symm
    _ = query s := h_classical.symm

/-- Sized-set variant of exact readout equivalence.  The size component is part
of the Agarwal set membership, while readout equality uses only validity. -/
theorem exactLocalLaws_agarwalValidSizedSummarySet_readout_equivalence
    {Y : Type*} [BoundedMetricSpace Y]
    {α State : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (valid : Stream α → State → Prop)
    (size : State → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ)
    (query : State → Y)
    (oracle : Stream α → Y)
    (xs : Stream α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    (h_same_input : oracle xs = fstar x)
    (h_query : ∀ xs s, valid xs s → query s = oracle xs) :
    ∀ z ∈ (ZR g x R T).support,
      ∀ s ∈ AgarwalValidSizedSummarySet valid size profile ε xs,
        fstar z = query s := by
  intro z hz s hs
  have hs_pair :
      valid xs s ∧ (size s : ℝ) ≤ profile ε xs.length := by
    simpa [AgarwalValidSizedSummarySet] using hs
  exact exactLocalLaws_agarwalValidSummarySet_readout_equivalence
    g T x R fstar valid query oracle xs hp hR laws h_same_input h_query
    z hz s (by simpa [AgarwalValidSummarySet] using hs_pair.1)

/-- State-level summary specialization of the set-level theorem: every valid
Agarwal state for the represented stream has the same readout value as every
exact C-Tree output. -/
theorem exactLocalLaws_stateLevelValidSummarySet_readout_equivalence
    {Y : Type*} [BoundedMetricSpace Y]
    {α State : Type*}
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (A : StateLevelMergeableSummary α State Y)
    (oracle : Stream α → Y)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query)
    (xs : Stream α)
    (hp : S T = x) (hR : R ≥ 1)
    (laws : LocalLawsBundle g T fstar)
    (h_same_input : oracle xs = fstar x) :
    ∀ z ∈ (ZR g x R T).support,
      ∀ s ∈ AgarwalValidSummarySet A.valid xs,
        fstar z = A.query s :=
  exactLocalLaws_agarwalValidSummarySet_readout_equivalence
    g T x R fstar A.valid A.query oracle xs
    hp hR laws h_same_input h_query

/-! ## Same-tree local-law inclusion -/

/-- Same-tree sketch inclusion: exact sketch-local witnesses give C1/C3/C2 on
the fixed C-Tree topology `T`. -/
theorem sameTree_sketch_to_local_laws
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (T : BinTree Strings) :
    LocalLawsBundle (sketchSummarizer op) T fstar :=
  local_laws_of_sketch
    (op := op) (fstar := fstar) (T := T)
    h_leaf h_merge h_compat

/-- Exact sketch/codec assumptions nest in exact theorem-backed local laws on
the same tree. -/
theorem sketchCodec_nests_exactLocalLaws
    {op : SketchOperator Strings Sketch} {fstar : Strings → Y}
    (assumptions : SketchCodecExactAssumptions op fstar)
    (T : BinTree Strings) :
    ExactTheoremBacked (sketchSummarizer op) T fstar :=
  assumptions.toExactTheoremBacked T

/-- Approximate sketch/codec assumptions nest in the approximate local-law
interface on the same tree.  This is a `def`, not a theorem, because the
approximate bundle stores numeric budget data. -/
def sketchCodecApprox_nests_approxLocalLaws
    {op : SketchOperator Strings Sketch} {T : BinTree Strings}
    {fstar : Strings → Y}
    (assumptions : SketchCodecApproxAssumptions op T fstar) :
    ApproxTheoremBacked (sketchSummarizer op) T fstar :=
  assumptions.toApproxTheoremBacked

/-! ## State-level Agarwal nesting -/

/-- Package Agarwal et al.'s original state-level ingredients as the Lean
`StateLevelMergeableSummary` interface: a summary method builds valid states,
and the state merge sends valid summaries for `D₁` and `D₂` to a valid summary
for `D₁ ++ D₂`. -/
def agarwalOriginal_to_stateLevelSummary
    {α State Pref : Type*}
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge) :
    StateLevelMergeableSummary α State Pref where
  build := build
  merge := merge
  query := query
  valid := valid
  build_valid := h_build
  merge_valid := h_merge

/-- Direct original-Agarwal validity theorem: build-validity plus mergeability
propagates validity to the root of any binary merge tree. -/
theorem agarwalOriginal_validity_of_mergeTree
    {α State : Type*}
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (t : MergeTree α) :
    valid (MergeTree.data t) (MergeTree.eval build merge t) :=
  (hierarchical_of_full
    (V := ({ build := build, valid := valid, build_valid := h_build } :
      ValidSketch α State))
    (merge := merge)
    h_merge) t

/-- Original-Agarwal hypotheses instantiate the exact relational C-TreePO
shape: validity is the invariant, and readout is applied only at the root. -/
theorem agarwalOriginal_nests_relational_ctreepo
    {α State Pref : Type*}
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_query : ∀ xs s, valid xs s → query s = oracle xs) :
    RelationalMergeablePreferenceShape build merge valid query oracle where
  build_valid := h_build
  merge_valid := h_merge
  readout_valid := h_query

/-- Direct original-Agarwal readout theorem: after any allowed state-merge tree,
the root query equals the oracle on the represented concatenated stream. -/
theorem agarwalOriginal_readout_of_mergeTree
    {α State Pref : Type*}
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_query : ∀ xs s, valid xs s → query s = oracle xs)
    (t : MergeTree α) :
    query (MergeTree.eval build merge t) = oracle (MergeTree.data t) :=
  h_query (MergeTree.data t) (MergeTree.eval build merge t)
    (agarwalOriginal_validity_of_mergeTree
      build merge valid h_build h_merge t)

/-- The exact local-law C1/L1 condition follows directly from the original
Agarwal validity and valid-state readout assumptions. -/
theorem agarwalOriginal_C1_localLaw
    {α State Pref : Type*}
    (build : Stream α → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_query : ∀ xs s, valid xs s → query s = oracle xs) :
    LocalLaws.C1 valid build oracle query := by
  intro xs
  refine ⟨h_build xs, ?_⟩
  exact h_query xs (build xs) (h_build xs)

/-- The exact local-law C3/L2 condition is precisely Agarwal mergeability plus
valid-state readout correctness. -/
theorem agarwalOriginal_C3_localLaw
    {α State Pref : Type*}
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (h_merge : MergeClosed valid merge)
    (h_query : ∀ xs s, valid xs s → query s = oracle xs) :
    LocalLaws.C3 valid merge oracle query := by
  intro xs ys sx sy hsx hsy
  have hvalid : valid (xs ++ ys) (merge sx sy) :=
    h_merge xs ys sx sy hsx.1 hsy.1
  refine ⟨hvalid, ?_⟩
  exact h_query (xs ++ ys) (merge sx sy) hvalid

/-- C1/C3 local laws derived from the original Agarwal hypotheses recover the
same root-readout theorem. -/
theorem agarwalOriginal_localLaw_readout_of_mergeTree
    {α State Pref : Type*}
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_query : ∀ xs s, valid xs s → query s = oracle xs)
    (t : MergeTree α) :
    LocalLaws.decide build merge query t = oracle (MergeTree.data t) :=
  LocalLaws.decide_eq_oracle_of_C1_C3
    oracle valid build merge query
    (agarwalOriginal_C1_localLaw build valid query oracle h_build h_query)
    (agarwalOriginal_C3_localLaw merge valid query oracle h_merge h_query)
    t

/-- Original Agarwal fixed-`ε` size validity for a freshly built state:
`build(D)` is a valid `S(D, ε)` summary satisfying the size profile. -/
def agarwalOriginal_build_valid_sized_state
    {α State : Type*}
    (build : Stream α → State)
    (valid : Stream α → State → Prop)
    (size : State → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε)
    (xs : Stream α) :
    Agarwal2013Full.ValidSizedState valid size profile ε xs :=
  Agarwal2013Full.buildValidSizedState
    (hbuild := h_build)
    (hsize := h_size)
    xs

/-- Original Agarwal fixed-`ε` mergeability for sized states: merging valid
`S(D₁, ε)` and `S(D₂, ε)` states gives a valid `S(D₁ ++ D₂, ε)` state with
the same size-profile guarantee. -/
def agarwalOriginal_merge_valid_sized_state
    {α State : Type*}
    (valid : Stream α → State → Prop)
    (merge : State → State → State)
    (size : State → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ)
    (h_merge : MergeClosed valid merge)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε)
    {xs ys : Stream α}
    (sx : Agarwal2013Full.ValidSizedState valid size profile ε xs)
    (sy : Agarwal2013Full.ValidSizedState valid size profile ε ys) :
    Agarwal2013Full.ValidSizedState valid size profile ε (xs ++ ys) :=
  Agarwal2013Full.mergeValidSizedState
    (hmerge := h_merge)
    (hsize := h_size)
    sx sy

/-- Original Agarwal fixed-`ε` tree theorem: evaluating any full state-merge
tree yields a valid sized `S(D, ε)` summary for the represented stream. -/
def agarwalOriginal_mergeTree_valid_sized_state
    {α State : Type*}
    (build : Stream α → State)
    (valid : Stream α → State → Prop)
    (merge : State → State → State)
    (size : State → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε)
    (t : MergeTree α) :
    Agarwal2013Full.ValidSizedState valid size profile ε
      (MergeTree.data t) :=
  Agarwal2013Full.mergeTree_validSizedState
    (hbuild := h_build)
    (hmerge := h_merge)
    (hsize := h_size)
    t

/-- The original Agarwal `k(|D|, ε)` size bound at the root of a merge tree. -/
theorem agarwalOriginal_mergeTree_size_bound
    {α State : Type*}
    (build : Stream α → State)
    (valid : Stream α → State → Prop)
    (merge : State → State → State)
    (size : State → Nat)
    (profile : Agarwal2013.SizeProfile)
    (ε : ℝ)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_size : Agarwal2013Full.ValidStateSizeProfile valid size profile ε)
    (t : MergeTree α) :
    (size (MergeTree.eval build merge t) : ℝ) ≤
      profile ε (MergeTree.data t).length :=
  (agarwalOriginal_mergeTree_valid_sized_state
    build valid merge size profile ε h_build h_merge h_size t).size_bound

/-- Original Agarwal epsilon-readout assumptions instantiate C-TreePO's
epsilon relational shape without turning the statement into deterministic
state equality. -/
theorem agarwalOriginal_nests_epsilon_relational_ctreepo
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (ε : ℝ)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect valid oracle query ε) :
    EpsilonRelationalMergeablePreferenceShape
      build merge valid query oracle ε where
  build_valid := h_build
  merge_valid := h_merge
  readout_valid := h_query

/-- Direct original-Agarwal epsilon-readout theorem at the root of any state
merge tree. -/
theorem agarwalOriginal_readout_error_of_mergeTree
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (build : Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (query : State → Pref)
    (oracle : Stream α → Pref)
    (ε : ℝ)
    (h_build : ∀ xs : Stream α, valid xs (build xs))
    (h_merge : MergeClosed valid merge)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect valid oracle query ε)
    (t : MergeTree α) :
    dist (query (MergeTree.eval build merge t))
      (oracle (MergeTree.data t)) ≤ ε :=
  h_query (MergeTree.data t) (MergeTree.eval build merge t)
    (agarwalOriginal_validity_of_mergeTree
      build merge valid h_build h_merge t)

/-- State-level Agarwal summaries instantiate C-TreePO's relational
state/readout shape. -/
theorem stateLevelSummary_nests_relational_ctreepo
    {α State Pref : Type*}
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query) :
    RelationalMergeablePreferenceShape A.build A.merge A.valid A.query oracle :=
  stateLevelMergeableSummary_relationalShape A oracle h_query

/-- State-level Agarwal summaries read out the target after merging along any
allowed merge tree. -/
theorem stateLevelSummary_root_readout_nests_ctreepo
    {α State Pref : Type*}
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query)
    (t : MergeTree α) :
    A.query (MergeTree.eval A.build A.merge t) =
      oracle (MergeTree.data t) :=
  stateLevelMergeableSummary_readout_of_mergeTree A oracle h_query t

/-- Epsilon state-level summaries instantiate the approximate relational
C-TreePO shape. -/
theorem epsilonStateLevelSummary_nests_relational_ctreepo
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref) (ε : ℝ)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect A.valid oracle A.query ε) :
    EpsilonRelationalMergeablePreferenceShape
      A.build A.merge A.valid A.query oracle ε :=
  stateLevelMergeableSummary_epsilonRelationalShape A oracle ε h_query

/-- Epsilon state-level summaries read out within the target task error after
merging along any allowed merge tree. -/
theorem epsilonStateLevelSummary_root_readout_nests_ctreepo
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref) (ε : ℝ)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect A.valid oracle A.query ε)
    (t : MergeTree α) :
    dist (A.query (MergeTree.eval A.build A.merge t))
      (oracle (MergeTree.data t)) ≤ ε :=
  stateLevelMergeableSummary_readout_error_of_mergeTree A oracle ε h_query t

/-- Randomized state-level summaries preserve their original root-success
probability when viewed as C-TreePO readout statements. -/
theorem randomizedStateLevelSummary_nests_relational_ctreepo
    {Ω α State Pref : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (readout : State → Pref)
    (oracle : Stream α → Pref)
    (p : ℝ)
    (h_success :
      ∀ t : MergeTree α,
        Agarwal2013Full.RandomizedTreeSuccess μ build valid merge t p)
    (h_query : ∀ xs s, valid xs s → readout s = oracle xs)
    (t : MergeTree α) :
    RandomizedTreeReadoutSuccess μ build merge readout oracle t p :=
  randomizedMergeableSummary_readout_success_of_mergeTree
    μ build merge valid readout oracle p h_success h_query t

/-- Randomized epsilon summaries preserve both their original probability and
their epsilon readout guarantee in C-TreePO form. -/
theorem randomizedEpsilonStateLevelSummary_nests_relational_ctreepo
    {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (readout : State → Pref)
    (oracle : Stream α → Pref)
    (ε p : ℝ)
    (h_success :
      ∀ t : MergeTree α,
        Agarwal2013Full.RandomizedTreeSuccess μ build valid merge t p)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect valid oracle readout ε)
    (t : MergeTree α) :
    RandomizedTreeEpsilonReadoutSuccess μ build merge readout oracle t ε p :=
  randomizedMergeableSummary_epsilon_readout_success_of_mergeTree
    μ build merge valid readout oracle ε p h_success h_query t

/-! ## Function-space/subspace inclusion -/

/-- Exact mergeable-sketch operators are contained in the exact local-law
subspace for every fixed tree. -/
theorem mergeableSketch_nests_exactLocalLawSubspace
    (fstar : Strings → Y) (T : BinTree Strings) :
    NeuralOperatorSpaces.MergeableSketchSummaryClass fstar ⊆
      NeuralOperatorSpaces.ExactLocalLawSubspace fstar T :=
  NeuralOperatorSpaces.mergeableSketchSummaryClass_subset_exactLocalLawSubspace
    fstar T

/-- The overlap between a chosen neural-operator class and the mergeable-sketch
class is contained in the exact local-law neural-operator subspace. -/
theorem mergeableSketchOverlap_nests_exactLocalLawNeuralOperators
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings) :
    NeuralOperatorSpaces.NeuralOperatorMergeableSketchOverlap C fstar ⊆
      NeuralOperatorSpaces.ExactLocalLawNeuralOperators C fstar T :=
  NeuralOperatorSpaces.mergeableSketch_overlap_subset_exactLocalLawNeuralOperators
    C fstar T

/-! ## Schedule bridges -/

/-- C-TreePO schedule bridge: if C1/C3 hold on two topologies with the same
leaves, both schedules have the same zero oracle distortion. -/
theorem localLaw_schedule_bridge
    (g : Summarizer Strings) (T T' : BinTree Strings) (fstar : Strings → Y)
    (h_leaves : leaves T = leaves T')
    (laws : LocalLawsBundle g T fstar)
    (laws' : LocalLawsBundle g T' fstar) :
    Egu g (root T) (fun z => D fstar z (S T)) =
      Egu g (root T') (fun z => D fstar z (S T')) :=
  schedule_invariance g T T' fstar h_leaves
    laws.law1 laws.law2 laws'.law1 laws'.law2

/-- Fold-of-folds bridge: C2 supplies the stable re-entry condition for
summaries reused across a two-level reduction plan. -/
theorem localLaw_foldOfFolds_bridge
    (g : Summarizer Strings) (T_comp : BinTree Strings)
    (x : Strings) (fstar : Strings → Y)
    (hp : S T_comp = x)
    (laws : LocalLawsBundle g T_comp fstar) :
    Egu g (root T_comp) (fun z => D fstar z x) = 0 :=
  fold_of_folds g T_comp x fstar hp laws.law1 laws.law2 laws.law3

/-- Ordered classical bridge: for free-monoid homomorphisms, rebracketing
ordered leaves gives identical state evaluation; no commutativity is used. -/
theorem orderedClassicalSchedule_bridge
    {α S : Type*}
    (h : Stream α → S) (combine : S → S → S)
    (h_hom : OrderedListHomomorphism h combine)
    (t₁ t₂ : MergeTree α)
    (h_data : MergeTree.data t₁ = MergeTree.data t₂) :
    MergeTree.eval h combine t₁ = MergeTree.eval h combine t₂ :=
  ctreepo_gibbons1996_ordered_schedule_invariance
    h combine h_hom t₁ t₂ h_data

end FormalProofs.OPT

end
