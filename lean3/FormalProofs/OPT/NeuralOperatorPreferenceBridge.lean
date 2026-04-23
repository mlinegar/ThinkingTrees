import FormalProofs.OPT.NeuralOperatorTheoremBridge
import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.NeuralOperatorSpaces

/-!
# FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean

Bridge from neural-operator theorem backing to preference objectives.

This file composes existing surfaces rather than reproving them:

* `NeuralOperatorTheoremBridge` turns uniform or finite-dimensionalized
  neural-operator approximation into approximate local-law bundles.
* `TheoremBackingConsequences` turns exact theorem-backedness into exact
  DPO/GRPO/preference-program equivalence.
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

end ApproxNeuralOperatorPreferenceBridge

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

end FDNeuralOperatorPreferenceBridge

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
