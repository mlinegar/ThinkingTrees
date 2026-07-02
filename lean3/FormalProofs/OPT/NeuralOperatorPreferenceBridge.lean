import FormalProofs.OPT.NeuralOperatorTheoremBridge
import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.NeuralOperatorSpaces
import FormalProofs.OPT.TwoStageOracleSurrogate

/-!
# FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean

Bridge from neural-operator theorem backing to preference objectives.

This file composes existing surfaces rather than reproving them:

* `NeuralOperatorTheoremBridge` turns uniform or finite-dimensionalized
  neural-operator approximation into approximate local-law bundles.
* `TheoremBackingConsequences` turns exact theorem-backedness into exact
  DPO/GRPO/preference-program equivalence.
* `TwoStageOracleSurrogate` turns theorem-backedness for a calibrated readout
  `fhat` into a true-oracle bound for `fstar` with additive `2ε_f` slack.
* `ApproximateLocalLaws` turns approximate local-law bundles into quantitative
  DPO/GRPO gaps.
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

variable {Strings : Type*} [Monoid Strings] [PseudoMetricSpace Strings]
variable {Y : Type*}

/-- Paper-facing aggregate transfer moduli for the neural-operator route.

The upstream operator theorem supplies a realized-call tolerance `ε`.  These
three moduli are the aggregate leaf, merge, and idempotence budgets used in the
paper formula
`ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`, after any nodewise leaf/merge
budgets have already been summed over the realized tree. -/
structure NeuralOperatorTransferModuli where
  omegaLeaf : ℝ → ℝ
  omegaMerge : ℝ → ℝ
  omegaIdemp : ℝ → ℝ

namespace NeuralOperatorTransferModuli

/-- The paper's neural-operator-first local-law budget for `R` re-summary
rounds. -/
def localLawBudget (ω : NeuralOperatorTransferModuli) (ε : ℝ) (R : ℕ) : ℝ :=
  ω.omegaLeaf ε + ω.omegaMerge ε + ((R : ℝ) - 1) * ω.omegaIdemp ε

/-- The paper's generic method-gap budget: method transport constant times the
neural-operator/local-law budget. -/
def methodGapBudget
    (ω : NeuralOperatorTransferModuli) (ε : ℝ) (R : ℕ) (C_meth : ℝ) : ℝ :=
  C_meth * ω.localLawBudget ε R

end NeuralOperatorTransferModuli

/-- Exact neural-operator preference bridge: a deterministic summarizer belongs
to a chosen neural-operator class and already satisfies exact theorem-backed
local laws. -/
structure ExactNeuralOperatorPreferenceBridge
    [BoundedMetricSpace Y]
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (s : Strings → Strings) (T : BinTree Strings) (fstar : Strings → Y) where
  mem_class : s ∈ C
  exact : ExactTheoremBacked (deterministicSummarizer s) T fstar

/-- Approximate neural-operator preference bridge: an exact ideal summarizer is
approximated by a realized neural operator, with explicit approximation-to-law
transfer assumptions. -/
structure ApproxNeuralOperatorPreferenceBridge
    [PseudoMetricSpace Y]
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (sStar sApprox : Strings → Strings) (T : BinTree Strings)
    (fstar : Strings → Y) (ε : ℝ) where
  ideal_mem_class : sStar ∈ C
  realized_mem_class : sApprox ∈ C
  exact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar
  uniformBridge : NeuralOperatorTheoremBridgeAssumptions sStar sApprox T fstar ε

namespace ApproxNeuralOperatorPreferenceBridge

variable [PseudoMetricSpace Y]

def toApproxLocalLawsBundle
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε) :
    ApproxLocalLawsBundle (deterministicSummarizer sApprox) T fstar :=
  approxLocalLawsBundle_of_uniformApproxExactTheoremBacked H.exact H.uniformBridge

def toApproxTheoremBacked
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε) :
    ApproxTheoremBacked (deterministicSummarizer sApprox) T fstar :=
  approxTheoremBacked_of_uniformApproxExactTheoremBacked H.exact H.uniformBridge

/-- The aggregate local-law bundle produced by the bridge matches the paper's
transfer-modulus notation at tolerance `ε`. -/
def matchesTransferModuli
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli) : Prop :=
  let laws := H.toApproxLocalLawsBundle
  laws.epsLeaf = ω.omegaLeaf ε ∧
    laws.epsMerge = ω.omegaMerge ε ∧
    laws.epsIdemp = ω.omegaIdemp ε

/-- The realized local-law budget produced by the neural-operator bridge for
`R` re-summary rounds. This is the Lean-facing version of the paper's
`ε_leaf + ε_merge + (R-1) ε_idemp` quantity. -/
def localLawBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (R : ℕ) : ℝ :=
  let laws := H.toApproxLocalLawsBundle
  laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp

/-- If the bridge budgets are written as transfer moduli, the realized budget is
exactly the paper formula
`ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`. -/
theorem localLawBudget_eq_transferModuliBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (R : ℕ) :
    H.localLawBudget R = ω.localLawBudget ε R := by
  rcases hω with ⟨hLeaf, hMerge, hIdemp⟩
  simp [localLawBudget, NeuralOperatorTransferModuli.localLawBudget,
    hLeaf, hMerge, hIdemp]

/-- The neural-operator realization budget controls document-level
`Δ_R_ZR`. This isolates the first quantitative step used by the paper:
operator approximation plus transfer moduli produce a local-law distortion
budget before any method-specific preference transport is applied. -/
theorem delta_R_ZR_le_localLawBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      H.localLawBudget R := by
  simpa [localLawBudget] using
    Δ_R_ZR_le_of_approx_bundle
      (deterministicSummarizer sApprox) T fstar x R hp hR
      hbound hbound_global h_mono H.toApproxLocalLawsBundle

/-- Paper-form version of the `Δ_R` bound: once the neural-operator bridge
budgets are expressed by transfer moduli, document-level distortion is bounded
by `ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`. -/
theorem delta_R_ZR_le_transferModuliBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      ω.localLawBudget ε R := by
  calc
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
        H.localLawBudget R :=
      H.delta_R_ZR_le_localLawBudget x R hp hR hbound hbound_global h_mono
    _ = ω.localLawBudget ε R :=
      H.localLawBudget_eq_transferModuliBudget ω hω R

/-- Epsilon-target version: a neural-operator realization is certified for
`fstar` at target `εCert` whenever its composed local-law budget is at most
`εCert`.  The approximation parameter `ε` stored in `H` controls the upstream
operator approximation; `εCert` is the paper-facing certification threshold. -/
theorem delta_R_ZR_le_epsilon_of_localLawBudget_le
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p)
    (hcert : H.localLawBudget R ≤ εCert) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤ εCert :=
  (H.delta_R_ZR_le_localLawBudget x R hp hR hbound hbound_global h_mono).trans hcert

end ApproxNeuralOperatorPreferenceBridge

/-! ## Teacher-first local-law route -/

/-- Pointwise teacher-residual transfer.  If a learned score `fhat` uniformly
approximates the true oracle `fstar`, then any local residual measured through
`fhat` bounds the corresponding true-oracle residual with two calibration
charges, one for each endpoint of the comparison. -/
theorem trueResidual_le_teacherResidual_add_two_calibration
    [BoundedMetricSpace Y]
    {fstar fhat : Strings → Y} {εf : NNReal}
    (hCal : UniformOracleApproximation fstar fhat εf)
    (a b : Strings) :
    dist (fstar a) (fstar b) ≤
      dist (fhat a) (fhat b) + 2 * (εf : ℝ) :=
  trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
    (fstar := fstar) (fhat := fhat) (ε := εf)
    (hApprox := hCal)

/-- Componentwise local-law residuals for the teacher-first route.

The `teacher*` fields are the residual components measured in the learned
teacher score. The `true*` fields are the corresponding components in the true
oracle score. This structure formalizes the componentwise interpretation of
law-error transfer; the root-prediction certification theorem below uses the
sharper end-to-end transfer and pays calibration slack only once. -/
structure TeacherFirstComponentLawErrors where
  teacherLeaf : ℝ
  teacherMerge : ℝ
  teacherIdemp : ℝ
  trueLeaf : ℝ
  trueMerge : ℝ
  trueIdemp : ℝ

namespace TeacherFirstComponentLawErrors

/-- Local-law error measured through the learned teacher. -/
def teacherLocalLawError (E : TeacherFirstComponentLawErrors) (R : ℕ) : ℝ :=
  E.teacherLeaf + E.teacherMerge + ((R : ℝ) - 1) * E.teacherIdemp

/-- Local-law error measured through the true oracle. -/
def trueLocalLawError (E : TeacherFirstComponentLawErrors) (R : ℕ) : ℝ :=
  E.trueLeaf + E.trueMerge + ((R : ℝ) - 1) * E.trueIdemp

/-- Componentwise calibration slack for the unweighted C1/C3/C2 composition:
one C1 term, one C3 term, and `R - 1` C2 terms. -/
def componentwiseCalibrationSlack (R : ℕ) (εf : ℝ) : ℝ :=
  2 * ((R : ℝ) + 1) * εf

/-- If each true-oracle component is bounded by the corresponding teacher
component plus two calibration errors, then the true local-law error is bounded
by the teacher local-law error plus the componentwise calibration slack. -/
theorem trueLocalLawError_le_teacherLocalLawError_add_componentwiseCalibration
    (E : TeacherFirstComponentLawErrors) (R : ℕ) (εf : ℝ)
    (hR : R ≥ 1)
    (hLeaf : E.trueLeaf ≤ E.teacherLeaf + 2 * εf)
    (hMerge : E.trueMerge ≤ E.teacherMerge + 2 * εf)
    (hIdemp : E.trueIdemp ≤ E.teacherIdemp + 2 * εf) :
    E.trueLocalLawError R ≤
      E.teacherLocalLawError R + componentwiseCalibrationSlack R εf := by
  unfold trueLocalLawError teacherLocalLawError componentwiseCalibrationSlack
  have hcoef : 0 ≤ ((R : ℝ) - 1) := by
    exact sub_nonneg.mpr (by exact_mod_cast hR)
  have hIdemp_scaled :
      ((R : ℝ) - 1) * E.trueIdemp ≤
        ((R : ℝ) - 1) * (E.teacherIdemp + 2 * εf) :=
    mul_le_mul_of_nonneg_left hIdemp hcoef
  nlinarith

end TeacherFirstComponentLawErrors

/-- Teacher-first route for settings where local true-oracle calls are not
available during training.

Stage 1 learns a score map `fhat` from available global labels and calibrates
it to the true oracle `fstar`. Stage 2 trains or audits a state map `g` using
local laws measured through `fhat`. The final theorem adds the two-sided
calibration slack to the local-law error. -/
structure TeacherFirstLocalLawRoute
    [BoundedMetricSpace Y]
    (g : Summarizer Strings) (T : BinTree Strings)
    (fstar fhat : Strings → Y) (εf : NNReal) where
  calibration : UniformOracleApproximation fstar fhat εf
  laws : ApproxLocalLawsBundle g T fhat

namespace TeacherFirstLocalLawRoute

variable [BoundedMetricSpace Y]

/-- The local-law error measured in the learned-teacher score `fhat`. -/
def localLawError
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf) (R : ℕ) : ℝ :=
  H.laws.localLawError R

/-- Backward-compatible name for `localLawError`. -/
def rootErrorBudget
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf) (R : ℕ) : ℝ :=
  H.localLawError R

/-- Total certified error for the teacher-first route: local-law error plus
the two-sided calibration slack. -/
def totalCertifiedError
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf) (R : ℕ) : ℝ :=
  H.localLawError R + 2 * (εf : ℝ)

/-- A teacher-first route is approximately theorem-backed for the learned
teacher `fhat`. -/
def toApproxTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf) :
    ApproxTheoremBacked g T fhat :=
  ApproxTheoremBacked.ofApproxLocalLaws H.laws

/-- Teacher-first certification: local-law distortion for the learned teacher
plus the two-sided calibration slack bounds true-oracle tree distortion. -/
theorem trueOracle_delta_R_ZR_le_rootErrorBudget_add_calibration
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    Δ_R_ZR g x R T fstar ≤
      H.rootErrorBudget R + 2 * (εf : ℝ) := by
  simpa [rootErrorBudget, ApproxLocalLawsBundle.rootErrorBudget,
    toApproxTheoremBacked] using
    (Δ_R_ZR_true_le_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (g := g) (T := T) (x := x) (R := R)
      (fstar := fstar) (fhat := fhat) (ε := εf)
      (hp := hp) (hApproxBacked := H.toApproxTheoremBacked)
      (hR := hR) (hApprox := H.calibration)
      (hbound := hbound) (hbound_global := hbound_global)
      (h_mono := h_mono))

/-- Teacher-first certification stated in the paper-facing total-error
notation. -/
theorem trueOracle_delta_R_ZR_le_totalCertifiedError
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    Δ_R_ZR g x R T fstar ≤ H.totalCertifiedError R := by
  simpa [totalCertifiedError, rootErrorBudget, localLawError] using
    H.trueOracle_delta_R_ZR_le_rootErrorBudget_add_calibration
      x R hp hR hbound hbound_global h_mono

/-- Epsilon-target teacher-first certification.  The training weights used to
fit `g` and `fhat` are separate from this final target threshold. -/
theorem trueOracle_delta_R_ZR_le_epsilon
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p)
    (hcert : H.rootErrorBudget R + 2 * (εf : ℝ) ≤ εCert) :
    Δ_R_ZR g x R T fstar ≤ εCert :=
  (H.trueOracle_delta_R_ZR_le_rootErrorBudget_add_calibration
      x R hp hR hbound hbound_global h_mono).trans hcert

/-- Epsilon-target teacher-first certification using the total certified error
notation. -/
theorem trueOracle_delta_R_ZR_le_epsilon_of_totalCertifiedError
    {g : Summarizer Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {εf : NNReal}
    (H : TeacherFirstLocalLawRoute g T fstar fhat εf)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p)
    (hcert : H.totalCertifiedError R ≤ εCert) :
    Δ_R_ZR g x R T fstar ≤ εCert :=
  (H.trueOracle_delta_R_ZR_le_totalCertifiedError
      x R hp hR hbound hbound_global h_mono).trans hcert

end TeacherFirstLocalLawRoute

/-- Finite-dimensionalization version of the approximate bridge.  The stored
finite-dimensionalization bridge is converted to the uniform bridge when the
preference theorem is applied. -/
structure FDNeuralOperatorPreferenceBridge
    [PseudoMetricSpace Y]
    (C : NeuralOperatorSpaces.NeuralOperatorClass Strings)
    (sStar sApprox : Strings → Strings) (T : BinTree Strings)
    (fstar : Strings → Y) (ε : ℝ) where
  ideal_mem_class : sStar ∈ C
  realized_mem_class : sApprox ∈ C
  exact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar
  fdBridge :
    NeuralOperatorFiniteDimensionalizationBridgeAssumptions
      sStar sApprox T fstar ε

namespace FDNeuralOperatorPreferenceBridge

variable [PseudoMetricSpace Y]

def toApproxBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε) :
    ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε where
  ideal_mem_class := H.ideal_mem_class
  realized_mem_class := H.realized_mem_class
  exact := H.exact
  uniformBridge := H.fdBridge.toUniformBridge

def toApproxLocalLawsBundle
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε) :
    ApproxLocalLawsBundle (deterministicSummarizer sApprox) T fstar :=
  H.toApproxBridge.toApproxLocalLawsBundle

def toApproxTheoremBacked
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε) :
    ApproxTheoremBacked (deterministicSummarizer sApprox) T fstar :=
  H.toApproxBridge.toApproxTheoremBacked

/-- Finite-dimensionalization bridge budgets written in the paper's aggregate
transfer-modulus notation. -/
def matchesTransferModuli
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli) : Prop :=
  H.toApproxBridge.matchesTransferModuli ω

/-- The realized local-law budget produced by the finite-dimensionalization
neural-operator bridge for `R` re-summary rounds. -/
def localLawBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (R : ℕ) : ℝ :=
  let laws := H.toApproxLocalLawsBundle
  laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp

/-- Finite-dimensionalization version of the paper formula for the realized
local-law budget. -/
theorem localLawBudget_eq_transferModuliBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (R : ℕ) :
    H.localLawBudget R = ω.localLawBudget ε R := by
  simpa [localLawBudget, matchesTransferModuli,
    FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    H.toApproxBridge.localLawBudget_eq_transferModuliBudget ω hω R

/-- Finite-dimensionalization version of the neural-operator realization-budget
bound on document-level `Δ_R_ZR`. -/
theorem delta_R_ZR_le_localLawBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      H.localLawBudget R := by
  simpa [localLawBudget, FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_localLawBudget
      (H := H.toApproxBridge) x R hp hR hbound hbound_global h_mono

/-- Finite-dimensionalization paper-form `Δ_R` bound using transfer moduli. -/
theorem delta_R_ZR_le_transferModuliBudget
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      ω.localLawBudget ε R := by
  calc
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
        H.localLawBudget R :=
      H.delta_R_ZR_le_localLawBudget x R hp hR hbound hbound_global h_mono
    _ = ω.localLawBudget ε R :=
      H.localLawBudget_eq_transferModuliBudget ω hω R

/-- Epsilon-target finite-dimensionalization version: the realized tree is
certified at target `εCert` when the composed local-law budget is at most
`εCert`. -/
theorem delta_R_ZR_le_epsilon_of_localLawBudget_le
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p)
    (hcert : H.localLawBudget R ≤ εCert) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤ εCert :=
  (H.delta_R_ZR_le_localLawBudget x R hp hR hbound hbound_global h_mono).trans hcert

end FDNeuralOperatorPreferenceBridge

section CalibratedReadout

variable [BoundedMetricSpace Y]

/-- Calibrated-readout neural-operator bridge.

If an ideal state-level/theorem-backed operator is exact for the learned
readout `fhat`, the realized neural operator approximates that ideal with the
usual local-law transfer budget, and `fhat` uniformly approximates the true
oracle `fstar`, then the realized summarizer's true-oracle distortion is
bounded by the local-law budget plus the two-sided calibration slack `2ε_f`.

This is the theorem-facing version of the learning route: learn/calibrate
`fhat` against `fstar`, then learn `g` by projecting toward the state-level
local-law subspace for `fhat`. -/
theorem trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      H.localLawBudget R + 2 * (εf : ℝ) := by
  simpa [ApproxNeuralOperatorPreferenceBridge.localLawBudget] using
    (Δ_R_ZR_true_le_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (g := deterministicSummarizer sApprox)
      (T := T)
      (x := x)
      (R := R)
      (fstar := fstar)
      (fhat := fhat)
      (ε := εf)
      (hp := hp)
      (hApproxBacked := H.toApproxTheoremBacked)
      (hR := hR)
      (hApprox := hCal)
      (hbound := hbound)
      (hbound_global := hbound_global)
      (h_mono := h_mono))

/-- Transfer-modulus form of
`trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge`. -/
theorem trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge_transferModuli
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      ω.localLawBudget ε R + 2 * (εf : ℝ) := by
  calc
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
        H.localLawBudget R + 2 * (εf : ℝ) :=
      trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge
        (H := H) hCal x R hp hR hbound hbound_global h_mono
    _ = ω.localLawBudget ε R + 2 * (εf : ℝ) := by
      rw [H.localLawBudget_eq_transferModuliBudget ω hω R]

/-- Epsilon-target calibrated version: if the realized local-law budget plus
two-sided scorer-calibration slack is at most `εCert`, then the true-oracle
tree distortion is certified at target `εCert`.  The optimizer loss weights
used to learn the operator are separate from this certification threshold. -/
theorem trueOracle_delta_R_ZR_le_epsilon_of_calibrated_neuralOperatorBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p)
    (hcert : H.localLawBudget R + 2 * (εf : ℝ) ≤ εCert) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤ εCert :=
  (trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge
      (H := H) hCal x R hp hR hbound hbound_global h_mono).trans hcert

/-- Lipschitz true-oracle utility version of the calibrated neural-operator
bridge. -/
theorem expected_trueOracleUtility_bound_via_calibrated_neuralOperatorBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (u : OracleUtility2 Y) (L : NNReal)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hL : OracleUtilityLipschitz1 u L)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    |Exp (ZR (deterministicSummarizer sApprox) x R T)
        (fun z => u (fstar z) (fstar x)) -
      u (fstar x) (fstar x)| ≤
      (L : ℝ) * (H.localLawBudget R + 2 * (εf : ℝ)) := by
  simpa [ApproxNeuralOperatorPreferenceBridge.localLawBudget] using
    (expected_trueOracleUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (g := deterministicSummarizer sApprox)
      (T := T)
      (x := x)
      (R := R)
      (fstar := fstar)
      (fhat := fhat)
      (ε := εf)
      (u := u)
      (L := L)
      (hp := hp)
      (hApproxBacked := H.toApproxTheoremBacked)
      (hR := hR)
      (hApprox := hCal)
      (hL := hL)
      (hbound := hbound)
      (hbound_global := hbound_global)
      (h_mono := h_mono))

/-- Finite-dimensionalization version of the calibrated true-oracle
distortion bridge. -/
theorem trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      H.localLawBudget R + 2 * (εf : ℝ) := by
  simpa [FDNeuralOperatorPreferenceBridge.localLawBudget,
    FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    (trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge
      (H := H.toApproxBridge) hCal x R hp hR hbound hbound_global h_mono)

/-- Transfer-modulus form of the finite-dimensionalization calibrated bridge. -/
theorem trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge_transferModuli
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
      ω.localLawBudget ε R + 2 * (εf : ℝ) := by
  calc
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
        H.localLawBudget R + 2 * (εf : ℝ) :=
      trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge
        (H := H) hCal x R hp hR hbound hbound_global h_mono
    _ = ω.localLawBudget ε R + 2 * (εf : ℝ) := by
      rw [H.localLawBudget_eq_transferModuliBudget ω hω R]

/-- Epsilon-target finite-dimensionalization calibrated version: if the
finite-dimensionalized realized local-law budget plus two-sided
scorer-calibration slack is at most `εCert`, then the true-oracle tree
distortion is certified at target `εCert`. -/
theorem trueOracle_delta_R_ZR_le_epsilon_of_calibrated_neuralOperatorFDBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (x : Strings) (R : ℕ) (εCert : ℝ)
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p)
    (hcert : H.localLawBudget R + 2 * (εf : ℝ) ≤ εCert) :
    Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤ εCert :=
  (trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge
      (H := H) hCal x R hp hR hbound hbound_global h_mono).trans hcert

/-- Finite-dimensionalization version of the calibrated true-oracle utility
bridge. -/
theorem expected_trueOracleUtility_bound_via_calibrated_neuralOperatorFDBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar fhat : Strings → Y} {ε : ℝ} {εf : NNReal}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fhat ε)
    (hCal : UniformOracleApproximation fstar fhat εf)
    (u : OracleUtility2 Y) (L : NNReal)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (hL : OracleUtilityLipschitz1 u L)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fhat
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fhat p) :
    |Exp (ZR (deterministicSummarizer sApprox) x R T)
        (fun z => u (fstar z) (fstar x)) -
      u (fstar x) (fstar x)| ≤
      (L : ℝ) * (H.localLawBudget R + 2 * (εf : ℝ)) := by
  simpa [FDNeuralOperatorPreferenceBridge.localLawBudget,
    FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    (expected_trueOracleUtility_bound_via_calibrated_neuralOperatorBridge
      (H := H.toApproxBridge) hCal u L x R hp hR hL
      hbound hbound_global h_mono)

end CalibratedReadout

section Exact

variable [BoundedMetricSpace Y]

/-- Exact neural-operator theorem-backedness transports any generic expected
loss whose generator and loss are oracle-indexed. -/
theorem expectedLossGeneric_eq_via_neuralOperatorExactBridge
    {α : Type*}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {s : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (H : ExactNeuralOperatorPreferenceBridge C s T fstar)
    (loss : Strings → α → ℝ) (gen : Strings → PMF α)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas : OracleMeasurableLossGeneric loss fstar)
    (h_gen : OracleIndexedGenGeneric gen fstar) :
    ExpectedLossGeneric loss (PMF.pure x) gen =
    ExpectedLossGeneric loss (ZR (deterministicSummarizer s) x R T) gen :=
  expected_loss_eq_via_ZR_of_exactTheoremBacked
    (fstar := fstar) (loss := loss) (gen := gen)
    (g := deterministicSummarizer s) (x := x) (R := R) (T := T)
    hp H.exact hR h_meas h_gen

/-- Exact neural-operator theorem-backedness transports any compositional
preference loss with an oracle-indexed generator. -/
theorem expectedPrefLoss_eq_via_neuralOperatorExactBridge
    {α : Type*}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {s : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (H : ExactNeuralOperatorPreferenceBridge C s T fstar)
    (loss : PrefLoss Strings α) (gen : PrefGen Strings α)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_gen : OracleIndexedGenComb gen fstar) :
    ExpectedPrefLoss loss (PMF.pure x) gen =
    ExpectedPrefLoss loss (ZR (deterministicSummarizer s) x R T) gen :=
  expected_pref_loss_eq_via_ZR_of_exactTheoremBacked
    (fstar := fstar) (loss := loss) (gen := gen)
    (g := deterministicSummarizer s) (x := x) (R := R) (T := T)
    hp H.exact hR h_meas h_gen

/-- Exact neural-operator theorem-backedness transports nested preference
programs built from oracle-indexed samplers. -/
theorem expectedPrefLossProg_eq_via_neuralOperatorExactBridge
    {α : Type*}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {s : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (H : ExactNeuralOperatorPreferenceBridge C s T fstar)
    (loss : PrefLoss Strings α) (prog : PrefProgram Strings α)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_prog : OracleIndexedProgram fstar prog) :
    ExpectedPrefLossProg loss (PMF.pure x) prog =
    ExpectedPrefLossProg loss (ZR (deterministicSummarizer s) x R T) prog :=
  expected_pref_loss_prog_eq_via_ZR_of_exactTheoremBacked
    (fstar := fstar) (loss := loss) (prog := prog)
    (g := deterministicSummarizer s) (x := x) (R := R) (T := T)
    hp H.exact hR h_meas h_prog

/-- Exact neural-operator theorem-backedness gives DPO zero-gap transport. -/
theorem dpo_equivalence_via_neuralOperatorExactBridge
    {A : Type*}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {s : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (H : ExactNeuralOperatorPreferenceBridge C s T fstar)
    (pol pol_ref : Policy Strings A) (β : ℝ)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β
      (ZR (deterministicSummarizer s) x R T) gen :=
  dpo_equivalence_via_ZR_of_exactTheoremBacked
    (fstar := fstar) pol pol_ref β gen
    (g := deterministicSummarizer s) (x := x) (R := R) (T := T)
    hp H.exact hR h_meas_pol h_meas_ref h_gen

/-- Exact neural-operator theorem-backedness gives GRPO-PL zero-gap transport. -/
theorem grpo_pl_equivalence_via_neuralOperatorExactBridge
    {A : Type*} {k : ℕ}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {s : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (H : ExactNeuralOperatorPreferenceBridge C s T fstar)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
    ExpectedGRPOLoss pol ranker
      (ZR (deterministicSummarizer s) x R T) gen :=
  grpo_equivalence_via_ZR_of_exactTheoremBacked
    (fstar := fstar) pol ranker gen
    (g := deterministicSummarizer s) (x := x) (R := R) (T := T)
    hp H.exact hR h_pol h_ranker h_gen

/-- Exact neural-operator theorem-backedness gives GRPO-RL zero-gap transport. -/
theorem grpo_rl_equivalence_via_neuralOperatorExactBridge
    {A : Type*} {k : ℕ}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {s : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (H : ExactNeuralOperatorPreferenceBridge C s T fstar)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ)
    (hp : S T = x) (hR : R ≥ 1)
    (h_meas : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
      (PMF.pure x) gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
      (ZR (deterministicSummarizer s) x R T) gen :=
  grpo_rl_equivalence_via_ZR_of_exactTheoremBacked
    (fstar := fstar) pol pol_old pol_ref reward eps beta gen
    (g := deterministicSummarizer s) (x := x) (R := R) (T := T)
    hp H.exact hR h_meas h_gen

end Exact

section Approximate

variable [PseudoMetricSpace Y]

/-- Generic Lipschitz expected-objective bridge for any preference-style method:
if the realized neural operator yields an approximate local-law bundle and
`E_gen` is Lipschitz in oracle distance, then the objective gap is bounded by
the same C1/C2/C3 budget. -/
theorem expectedObjectiveGap_via_neuralOperatorUniformBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (E_gen : Strings → ℝ)
    (x : Strings) (R : ℕ) (L : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (E_max : ℝ) (hE_max : 0 ≤ E_max)
    (hE_bound : ∀ x', |E_gen x'| ≤ E_max)
    (h_lip : ∀ x' z', |E_gen x' - E_gen z'| ≤ L * dist (fstar x') (fstar z'))
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |∑' x', (PMF.pure x x').toReal * E_gen x' -
      ∑' z, (ZR (deterministicSummarizer sApprox) x R T z).toReal * E_gen z| ≤
      (L : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  intro laws
  have hΔ :
      Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar =
        ∑' z, ∑' x',
          (ZR (deterministicSummarizer sApprox) x R T z).toReal *
            (PMF.pure x x').toReal * dist (fstar z) (fstar x') := by
    simpa using
      (coupling_Δ_eq_Δ_R_ZR
        (deterministicSummarizer sApprox) x R T fstar).symm
  have hgap :
      |∑' x', (PMF.pure x x').toReal * E_gen x' -
        ∑' z, (ZR (deterministicSummarizer sApprox) x R T z).toReal * E_gen z| ≤
        (L : ℝ) * Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar := by
    exact unified_preference_gap_bounded
      fstar E_gen (PMF.pure x) (ZR (deterministicSummarizer sApprox) x R T)
      L (Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar)
      D_max hD_max h_dist_bound E_max hE_max hE_bound h_lip hΔ
  have hBudget :
      Δ_R_ZR (deterministicSummarizer sApprox) x R T fstar ≤
        laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp :=
    Δ_R_ZR_le_of_approx_bundle
      (deterministicSummarizer sApprox) T fstar x R hp hR
      hbound hbound_global h_mono laws
  exact le_trans hgap
    (mul_le_mul_of_nonneg_left hBudget (NNReal.coe_nonneg L))

/-- Finite-dimensionalization variant of the generic Lipschitz bridge. -/
theorem expectedObjectiveGap_via_neuralOperatorFDBridge
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (E_gen : Strings → ℝ)
    (x : Strings) (R : ℕ) (L : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (E_max : ℝ) (hE_max : 0 ≤ E_max)
    (hE_bound : ∀ x', |E_gen x'| ≤ E_max)
    (h_lip : ∀ x' z', |E_gen x' - E_gen z'| ≤ L * dist (fstar x') (fstar z'))
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |∑' x', (PMF.pure x x').toReal * E_gen x' -
      ∑' z, (ZR (deterministicSummarizer sApprox) x R T z).toReal * E_gen z| ≤
      (L : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  simpa [FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    expectedObjectiveGap_via_neuralOperatorUniformBridge
      (H := H.toApproxBridge) E_gen x R L
      D_max hD_max h_dist_bound E_max hE_max hE_bound h_lip
      hp hR hbound hbound_global h_mono

/-- Paper-form generic objective bridge.  The upstream neural-operator tolerance
`ε` is first converted to transfer-modulus budgets, then multiplied by the
method transport constant. -/
theorem expectedObjectiveGap_via_neuralOperatorTransferModuli
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (E_gen : Strings → ℝ)
    (x : Strings) (R : ℕ) (L : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (E_max : ℝ) (hE_max : 0 ≤ E_max)
    (hE_bound : ∀ x', |E_gen x'| ≤ E_max)
    (h_lip : ∀ x' z', |E_gen x' - E_gen z'| ≤ L * dist (fstar x') (fstar z'))
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    |∑' x', (PMF.pure x x').toReal * E_gen x' -
      ∑' z, (ZR (deterministicSummarizer sApprox) x R T z).toReal * E_gen z| ≤
      ω.methodGapBudget ε R (L : ℝ) := by
  rcases hω with ⟨hLeaf, hMerge, hIdemp⟩
  have hGap :=
    expectedObjectiveGap_via_neuralOperatorUniformBridge
      (H := H) E_gen x R L
      D_max hD_max h_dist_bound E_max hE_max hE_bound h_lip
      hp hR hbound hbound_global h_mono
  simpa [NeuralOperatorTransferModuli.localLawBudget,
    NeuralOperatorTransferModuli.methodGapBudget, hLeaf, hMerge, hIdemp] using hGap

/-- Finite-dimensionalization version of the paper-form generic objective
bridge. -/
theorem expectedObjectiveGap_via_neuralOperatorFDTransferModuli
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (ω : NeuralOperatorTransferModuli)
    (hω : H.matchesTransferModuli ω)
    (E_gen : Strings → ℝ)
    (x : Strings) (R : ℕ) (L : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (E_max : ℝ) (hE_max : 0 ≤ E_max)
    (hE_bound : ∀ x', |E_gen x'| ≤ E_max)
    (h_lip : ∀ x' z', |E_gen x' - E_gen z'| ≤ L * dist (fstar x') (fstar z'))
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    |∑' x', (PMF.pure x x').toReal * E_gen x' -
      ∑' z, (ZR (deterministicSummarizer sApprox) x R T z).toReal * E_gen z| ≤
      ω.methodGapBudget ε R (L : ℝ) := by
  simpa [FDNeuralOperatorPreferenceBridge.matchesTransferModuli] using
    expectedObjectiveGap_via_neuralOperatorTransferModuli
      (H := H.toApproxBridge) ω hω E_gen x R L
      D_max hD_max h_dist_bound E_max hE_max hE_bound h_lip
      hp hR hbound hbound_global h_mono

/-- DPO quantitative gap via the uniform neural-operator bridge. -/
theorem dpo_gap_via_neuralOperatorUniformBridge
    {A : Type*}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ) (β : ℝ) (L_pol : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A),
      |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
      ExpectedDPOLoss pol pol_ref β
        (ZR (deterministicSummarizer sApprox) x R T) gen| ≤
      2 * |β| * (L_pol : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  intro laws
  exact dpo_gap_via_approx_bundle fstar pol pol_ref gen
    (deterministicSummarizer sApprox) x R T β L_pol hp hR
    D_max hD_max h_dist_bound hbound hbound_global
    Loss_max hLoss_max hLoss_bound
    h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono laws

/-- DPO quantitative gap via the finite-dimensionalization neural-operator bridge. -/
theorem dpo_gap_via_neuralOperatorFDBridge
    {A : Type*}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (pol pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (x : Strings) (R : ℕ) (β : ℝ) (L_pol : NNReal)
    (hp : S T = x) (hR : R ≥ 1)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (p : A × A),
      |DPOLossPointwise pol pol_ref β x' p.1 p.2| ≤ Loss_max)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen -
      ExpectedDPOLoss pol pol_ref β
        (ZR (deterministicSummarizer sApprox) x R T) gen| ≤
      2 * |β| * (L_pol : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  simpa [FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    dpo_gap_via_neuralOperatorUniformBridge
      (H := H.toApproxBridge) pol pol_ref gen x R β L_pol hp hR
      D_max hD_max h_dist_bound hbound hbound_global
      Loss_max hLoss_max hLoss_bound
      h_meas_pol h_meas_ref h_lip h_gen_fixed h_mono

/-- GRPO-Plackett-Luce quantitative gap via the uniform neural-operator bridge. -/
theorem grpo_pl_gap_via_neuralOperatorUniformBridge
    {A : Type*} {k : ℕ}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ) (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo
        h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
      ExpectedGRPOLoss pol ranker
        (ZR (deterministicSummarizer sApprox) x R T) gen| ≤
      (L_grpo : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  intro laws
  exact grpo_pl_gap_via_approx_bundle
    (k := k) fstar pol ranker gen
    (deterministicSummarizer sApprox) x R T L_grpo
    D_max hD_max h_dist_bound Loss_max hLoss_max hLoss_bound
    h_pol_lip h_ranker h_rum h_gen_fixed hp hR hbound hbound_global
    h_mono laws

/-- GRPO-Plackett-Luce quantitative gap via the finite-dimensionalization
neural-operator bridge. -/
theorem grpo_pl_gap_via_neuralOperatorFDBridge
    {A : Type*} {k : ℕ}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ) (L_grpo : NNReal)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ w z, dist (fstar w) (fstar z) ≤ D_max)
    (Loss_max : ℝ) (hLoss_max : 0 ≤ Loss_max)
    (hLoss_bound : ∀ x' (group : Fin k → A),
      |GRPOLossPointwise pol x' group (ranker x' group)| ≤ Loss_max)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x' z',
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x') L_grpo
        h_pol_lip h_ranker x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |ExpectedGRPOLoss pol ranker (PMF.pure x) gen -
      ExpectedGRPOLoss pol ranker
        (ZR (deterministicSummarizer sApprox) x R T) gen| ≤
      (L_grpo : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  simpa [FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    grpo_pl_gap_via_neuralOperatorUniformBridge
      (H := H.toApproxBridge) pol ranker gen x R L_grpo
      D_max hD_max h_dist_bound Loss_max hLoss_max hLoss_bound
      h_pol_lip h_ranker h_rum h_gen_fixed hp hR hbound hbound_global h_mono

/-- GRPO-RL quantitative gap via the uniform neural-operator bridge. -/
theorem grpo_rl_gap_via_neuralOperatorUniformBridge
    {A : Type*} {k : ℕ}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : ApproxNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ) (L_grpo_rl : NNReal)
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
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta
        fstar (gen x') L_grpo_rl h_pol_lip h_old_lip h_ref_lip
        h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
        (ZR (deterministicSummarizer sApprox) x R T) gen| ≤
      (L_grpo_rl : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  intro laws
  exact grpo_rl_gap_via_approx_bundle
    (k := k) fstar pol pol_old pol_ref reward eps beta gen
    (deterministicSummarizer sApprox) x R T L_grpo_rl
    D_max hD_max h_dist_bound Loss_max hLoss_max hLoss_bound
    h_pol_lip h_old_lip h_ref_lip h_reward_lip h_rum h_gen_fixed
    hp hR hbound hbound_global h_mono laws

/-- GRPO-RL quantitative gap via the finite-dimensionalization neural-operator
bridge. -/
theorem grpo_rl_gap_via_neuralOperatorFDBridge
    {A : Type*} {k : ℕ}
    {C : NeuralOperatorSpaces.NeuralOperatorClass Strings}
    {sStar sApprox : Strings → Strings} {T : BinTree Strings}
    {fstar : Strings → Y} {ε : ℝ}
    (H : FDNeuralOperatorPreferenceBridge C sStar sApprox T fstar ε)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (x : Strings) (R : ℕ) (L_grpo_rl : NNReal)
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
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta
        fstar (gen x') L_grpo_rl h_pol_lip h_old_lip h_ref_lip
        h_reward_lip x' z')
    (h_gen_fixed : ∀ x' x'', gen x' = gen x'')
    (hp : S T = x) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p,
      pIdemp (deterministicSummarizer sApprox) fstar
        (p.bind (deterministicSummarizer sApprox)) ≤
      pIdemp (deterministicSummarizer sApprox) fstar p) :
    let laws := H.toApproxLocalLawsBundle
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen -
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta
        (ZR (deterministicSummarizer sApprox) x R T) gen| ≤
      (L_grpo_rl : ℝ) *
        (laws.epsLeaf + laws.epsMerge + ((R : ℝ) - 1) * laws.epsIdemp) := by
  simpa [FDNeuralOperatorPreferenceBridge.toApproxLocalLawsBundle] using
    grpo_rl_gap_via_neuralOperatorUniformBridge
      (H := H.toApproxBridge) pol pol_old pol_ref reward eps beta gen
      x R L_grpo_rl D_max hD_max h_dist_bound Loss_max hLoss_max
      hLoss_bound h_pol_lip h_old_lip h_ref_lip h_reward_lip h_rum
      h_gen_fixed hp hR hbound hbound_global h_mono

end Approximate

end FormalProofs.OPT
