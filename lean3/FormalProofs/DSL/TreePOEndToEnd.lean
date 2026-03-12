import FormalProofs.DSL.TreeIPW

/-!
# FormalProofs/DSL/TreePOEndToEnd.lean

End-to-end TreePO certificates that chain:

1. Bernoulli Horvitz-Thompson (HT) unbiasedness for the sampled tree objective,
2. Method-specific TreePO gap bounds (DPO / GRPO-PL / GRPO-RL) expressed via
   the HT estimator of tree distortion.

These theorems are "one-stop" statements for downstream use. They package the
core chain already proved in `DSL/TreeIPW.lean` into method-level certificates.
-/

set_option linter.mathlibStandardSet false

open MeasureTheory
open scoped BigOperators Real Nat Classical Pointwise NNReal

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-! ## Generic Building Blocks -/

/-- TreePO objective unbiasedness under Bernoulli HT sampling. -/
theorem treepo_objective_unbiased
    {Strings Node A : Type*} {k : ℕ}
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss :=
  ipw_preference_loss_connection_tree model loss pi hpi_pos hpi_le

/-- Tree distortion unbiasedness under Bernoulli HT sampling. -/
theorem treepo_distortion_unbiased
    {Strings Node A Y : Type*} {k : ℕ}
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    [PseudoMetricSpace Y]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      ExpectedTreeDistortion model fstar :=
  ipw_tree_distortion_unbiased model fstar pi hpi_pos hpi_le

/-- Generic end-to-end bridge: if a "true" target loss is within `oracle_err`
of the oracle-indexed target loss, and the oracle-indexed target loss is within
`tree_bound` of the tree objective, then the true target loss is within
`oracle_err + tree_bound` of the tree objective. -/
theorem treepo_loss_gap_with_oracleMeasurement
    (loss_true loss_oracle loss_tree oracle_err tree_bound : ℝ)
    (h_oracle : |loss_true - loss_oracle| ≤ oracle_err)
    (h_tree : |loss_oracle - loss_tree| ≤ tree_bound) :
    |loss_true - loss_tree| ≤ oracle_err + tree_bound := by
  have h :=
    treepo_gap_with_oracleMeasurement_calibration_and_estimation
      (gap_true := loss_true - loss_tree)
      (gap_oracle := loss_oracle - loss_tree)
      (gap_judge := loss_oracle - loss_tree)
      (gap_est := 0)
      (oracle_err := oracle_err)
      (cal_err := 0)
      (est_err := tree_bound)
      (h_oracle := by
        simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using h_oracle)
      (h_cal := by simp)
      (h_est := by simpa using h_tree)
  simpa [sub_eq_add_neg] using h

/-- Exact-oracle convenience corollary of `treepo_loss_gap_with_oracleMeasurement`. -/
theorem treepo_loss_gap_of_exactOracle
    (loss_true loss_oracle loss_tree tree_bound : ℝ)
    (h_oracle_exact : loss_true = loss_oracle)
    (h_tree : |loss_oracle - loss_tree| ≤ tree_bound) :
    |loss_true - loss_tree| ≤ tree_bound := by
  have h_oracle : |loss_true - loss_oracle| ≤ 0 := by
    rw [h_oracle_exact]
    simp
  have h :=
    treepo_loss_gap_with_oracleMeasurement
      (loss_true := loss_true) (loss_oracle := loss_oracle) (loss_tree := loss_tree)
      (oracle_err := 0) (tree_bound := tree_bound)
      h_oracle h_tree
  simpa using h

/-! ## Method Certificates -/

/-- End-to-end TreePO certificate for DPO:
objective unbiasedness + IPW distortion gap bound. -/
theorem dpo_treepo_end_to_end_certificate
    {Strings Node A Y : Type*}
    [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u,
      model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    let loss : Strings → (Fin 1 → (A × A)) → ℝ :=
      fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
      (2 * |β| * (L_pol : ℝ)) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  refine ⟨?_, ?_⟩
  · exact ipw_preference_loss_connection_tree model _ pi hpi_pos hpi_le
  · simpa using
      (dpo_tree_gap_bounded_ipw (model := model) (fstar := fstar)
        (pol := pol) (pol_ref := pol_ref) (β := β) (L_pol := L_pol)
        (gpair := gpair) (h_group := h_group) (h_lip := h_lip)
        (M_loss := M_loss) (hM_loss := hM_loss) (h_loss_bound := h_loss_bound)
        (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le))

/-- End-to-end DPO certificate with an optional oracle-measurement layer above
the usual oracle-indexed DPO target. -/
theorem dpo_treepo_end_to_end_certificate_with_oracleMeasurement
    {Strings Node A Y : Type*}
    [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u,
      model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair)| ≤ oracle_err) :
    let loss : Strings → (Fin 1 → (A × A)) → ℝ :=
      fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|loss_true - OPT.ExpectedTreePreferenceLoss model loss| ≤
      oracle_err +
        (2 * |β| * (L_pol : ℝ)) *
          ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  rcases dpo_treepo_end_to_end_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (pol_ref := pol_ref) (β := β) (L_pol := L_pol)
      (gpair := gpair) (h_group := h_group) (h_lip := h_lip)
      (M_loss := M_loss) (hM_loss := hM_loss) (h_loss_bound := h_loss_bound)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) with ⟨h_obj, h_gap⟩
  refine ⟨h_obj, ?_⟩
  exact treepo_loss_gap_with_oracleMeasurement
    (loss_true := loss_true)
    (loss_oracle := ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair))
    (loss_tree := OPT.ExpectedTreePreferenceLoss model
      (fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2))
    (oracle_err := oracle_err)
    (tree_bound := (2 * |β| * (L_pol : ℝ)) *
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le)
    h_oracle h_gap

/-- End-to-end TreePO certificate for GRPO-PL with constant group generator:
objective unbiasedness + IPW distortion gap bound. -/
theorem grpo_pl_treepo_end_to_end_certificate
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L h_pol_lip h_ranker x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPOLossPointwise pol x group (ranker x group)
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  refine ⟨?_, ?_⟩
  · exact ipw_preference_loss_connection_tree model _ pi hpi_pos hpi_le
  · simpa using
      (grpo_pl_tree_gap_bounded_ipw (model := model) (fstar := fstar)
        (pol := pol) (ranker := ranker) (g := g) (L := L)
        (h_group := h_group) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
        (h_rum := h_rum) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le))

/-- End-to-end GRPO-PL certificate with an optional oracle-measurement layer
above the oracle-indexed GRPO target. -/
theorem grpo_pl_treepo_end_to_end_certificate_with_oracleMeasurement
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L h_pol_lip h_ranker x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedGRPOLoss pol ranker model.docDist (fun _ => g)| ≤ oracle_err) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPOLossPointwise pol x group (ranker x group)
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|loss_true - OPT.ExpectedTreePreferenceLoss model loss| ≤
      oracle_err +
        (L : ℝ) *
          ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  rcases grpo_pl_treepo_end_to_end_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (ranker := ranker) (g := g) (L := L)
      (h_group := h_group) (h_pol_lip := h_pol_lip)
      (h_ranker := h_ranker) (h_rum := h_rum)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) with ⟨h_obj, h_gap⟩
  refine ⟨h_obj, ?_⟩
  exact treepo_loss_gap_with_oracleMeasurement
    (loss_true := loss_true)
    (loss_oracle := ExpectedGRPOLoss pol ranker model.docDist (fun _ => g))
    (loss_tree := OPT.ExpectedTreePreferenceLoss model
      (fun x group => GRPOLossPointwise pol x group (ranker x group)))
    (oracle_err := oracle_err)
    (tree_bound := (L : ℝ) *
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le)
    h_oracle h_gap

/-- End-to-end TreePO certificate for GRPO-PL with span-induced doc-dependent group generator:
objective unbiasedness + IPW distortion gap bound. -/
theorem grpo_pl_treepo_end_to_end_gen_certificate
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A] [DecidableEq A]
    [DecidableEq Strings] [DecidableEq Node]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (L_grpo L_gen : ℝ≥0)
    (M : ℝ) (hM : 0 ≤ M)
    (h_group : ∀ u, model.groupGen u = gen (model.nodeSpan u))
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x) L_grpo h_pol_lip h_ranker x z)
    (h_loss_bound : ∀ x (group : Fin k → A),
      |GRPOLossPointwise pol x group (ranker x group)| ≤ M)
    (h_gen_lip : GroupGeneratorLipschitzL1 gen fstar L_gen)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPOLossPointwise pol x group (ranker x group)
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|ExpectedGRPOLoss pol ranker model.docDist gen -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
      ((L_grpo : ℝ) + M * (L_gen : ℝ)) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  refine ⟨?_, ?_⟩
  · exact ipw_preference_loss_connection_tree model _ pi hpi_pos hpi_le
  · simpa using
      (grpo_pl_tree_gap_bounded_ipw_gen (model := model) (fstar := fstar)
        (pol := pol) (ranker := ranker) (gen := gen)
        (L_grpo := L_grpo) (L_gen := L_gen) (M := M) (hM := hM)
        (h_group := h_group) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
        (h_rum := h_rum) (h_loss_bound := h_loss_bound) (h_gen_lip := h_gen_lip)
        (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le))

/-- End-to-end GRPO-PL certificate for span-induced generators with an optional
oracle-measurement layer above the oracle-indexed GRPO target. -/
theorem grpo_pl_treepo_end_to_end_gen_certificate_with_oracleMeasurement
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A] [DecidableEq A]
    [DecidableEq Strings] [DecidableEq Node]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (L_grpo L_gen : ℝ≥0)
    (M : ℝ) (hM : 0 ≤ M)
    (h_group : ∀ u, model.groupGen u = gen (model.nodeSpan u))
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_grpo)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar (gen x) L_grpo h_pol_lip h_ranker x z)
    (h_loss_bound : ∀ x (group : Fin k → A),
      |GRPOLossPointwise pol x group (ranker x group)| ≤ M)
    (h_gen_lip : GroupGeneratorLipschitzL1 gen fstar L_gen)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true - ExpectedGRPOLoss pol ranker model.docDist gen| ≤ oracle_err) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPOLossPointwise pol x group (ranker x group)
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|loss_true - OPT.ExpectedTreePreferenceLoss model loss| ≤
      oracle_err +
        ((L_grpo : ℝ) + M * (L_gen : ℝ)) *
          ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  rcases grpo_pl_treepo_end_to_end_gen_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (ranker := ranker) (gen := gen)
      (L_grpo := L_grpo) (L_gen := L_gen) (M := M) (hM := hM)
      (h_group := h_group) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
      (h_rum := h_rum) (h_loss_bound := h_loss_bound) (h_gen_lip := h_gen_lip)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) with ⟨h_obj, h_gap⟩
  refine ⟨h_obj, ?_⟩
  exact treepo_loss_gap_with_oracleMeasurement
    (loss_true := loss_true)
    (loss_oracle := ExpectedGRPOLoss pol ranker model.docDist gen)
    (loss_tree := OPT.ExpectedTreePreferenceLoss model
      (fun x group => GRPOLossPointwise pol x group (ranker x group)))
    (oracle_err := oracle_err)
    (tree_bound := ((L_grpo : ℝ) + M * (L_gen : ℝ)) *
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le)
    h_oracle h_gap

/-- End-to-end TreePO certificate for GRPO-RL:
objective unbiasedness + IPW distortion gap bound. -/
theorem grpo_rl_treepo_end_to_end_certificate
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  refine ⟨?_, ?_⟩
  · exact ipw_preference_loss_connection_tree model _ pi hpi_pos hpi_le
  · simpa using
      (grpo_rl_tree_gap_bounded_ipw (model := model) (fstar := fstar)
        (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta)
        (g := g) (L := L) (h_group := h_group)
        (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip) (h_ref_lip := h_ref_lip)
        (h_reward_lip := h_reward_lip) (h_rum := h_rum)
        (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le))

/-- End-to-end GRPO-RL certificate with an optional oracle-measurement layer
above the oracle-indexed GRPO-RL target. -/
theorem grpo_rl_treepo_end_to_end_certificate_with_oracleMeasurement
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ)
    (eps beta : ℝ)
    (g : PMF (Fin k → A))
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L)
    (h_old_lip : GRPOPolicyLipschitz pol_old fstar L)
    (h_ref_lip : GRPOPolicyLipschitz pol_ref fstar L)
    (h_reward_lip : RewardLipschitzGRPO reward fstar L)
    (h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (loss_true oracle_err : ℝ)
    (h_oracle :
      |loss_true -
          ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g)| ≤
        oracle_err) :
    let loss : Strings → (Fin k → A) → ℝ :=
      fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group
    (∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss) ∧
    (|loss_true - OPT.ExpectedTreePreferenceLoss model loss| ≤
      oracle_err +
        (L : ℝ) *
          ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
                (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le) := by
  classical
  dsimp
  rcases grpo_rl_treepo_end_to_end_certificate
      (model := model) (fstar := fstar)
      (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta)
      (g := g) (L := L) (h_group := h_group)
      (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
      (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
      (h_rum := h_rum)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) with ⟨h_obj, h_gap⟩
  refine ⟨h_obj, ?_⟩
  exact treepo_loss_gap_with_oracleMeasurement
    (loss_true := loss_true)
    (loss_oracle := ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g))
    (loss_tree := OPT.ExpectedTreePreferenceLoss model
      (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group))
    (oracle_err := oracle_err)
    (tree_bound := (L : ℝ) *
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le)
    h_oracle h_gap
