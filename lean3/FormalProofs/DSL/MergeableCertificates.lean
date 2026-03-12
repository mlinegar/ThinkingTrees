import FormalProofs.DSL.TreeIPW
import FormalProbability.ML.MergeableSummaries.GK
import FormalProbability.ML.MergeableSummaries.KLL

/-!
# FormalProofs/DSL/MergeableCertificates.lean

Certificate transport lemmas for plugging mergeable-sketch upper bounds into
existing TreePO/IPW gap certificates.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

open ML.MergeableSummary

/-- Deterministic certificate transport: substitute any upper bound on `Δ`. -/
theorem tree_gap_bound_transport_upper
    {gap C Δ U : ℝ}
    (hC : 0 ≤ C)
    (h_gap : |gap| ≤ C * Δ)
    (h_upper : Δ ≤ U) :
    |gap| ≤ C * U := by
  exact le_trans h_gap (mul_le_mul_of_nonneg_left h_upper hC)

/-- Event-conditional transport used for high-probability upper-bound events. -/
theorem tree_gap_bound_transport_upper_prob
    {Ω : Type*}
    (E : Set Ω)
    {gap C : ℝ}
    {Δ U : Ω → ℝ}
    (hC : 0 ≤ C)
    (h_gap : ∀ ω ∈ E, |gap| ≤ C * Δ ω)
    (h_upper : ∀ ω ∈ E, Δ ω ≤ U ω) :
    ∀ ω ∈ E, |gap| ≤ C * U ω := by
  intro ω hω
  exact tree_gap_bound_transport_upper
    (gap := gap) (C := C) (Δ := Δ ω) (U := U ω)
    hC (h_gap ω hω) (h_upper ω hω)

/-- DPO TreePO certificate after substituting an upper bound on the IPW distortion term. -/
theorem dpo_tree_gap_bounded_by_sketch_upper
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node (A × A) 1)
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A)
    (β : ℝ) (L_pol : ℝ≥0)
    (gpair : PMF (A × A))
    (h_group : ∀ u, model.groupGen u = PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair)
    (h_lip : PolicyLipschitz pol pol_ref fstar L_pol)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (U : ℝ)
    (h_upper :
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le ≤ U) :
    |ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2)| ≤
      (2 * |β| * (L_pol : ℝ)) * U := by
  have h_ipw :
      |ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2)| ≤
      (2 * |β| * (L_pol : ℝ)) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le :=
    dpo_tree_gap_bounded_ipw (model := model) (fstar := fstar)
      (pol := pol) (pol_ref := pol_ref) (β := β) (L_pol := L_pol)
      (gpair := gpair) (h_group := h_group) (h_lip := h_lip)
      (M_loss := M_loss) (hM_loss := hM_loss) (h_loss_bound := h_loss_bound)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  have hC : 0 ≤ 2 * |β| * (L_pol : ℝ) := by positivity
  exact tree_gap_bound_transport_upper hC h_ipw h_upper

/-- GRPO-PL TreePO certificate after substituting an upper bound on IPW distortion. -/
theorem grpo_pl_tree_gap_bounded_by_sketch_upper
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
    (U : ℝ)
    (h_upper :
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le ≤ U) :
    |ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPOLossPointwise pol x group (ranker x group))| ≤
      (L : ℝ) * U := by
  have h_ipw :
      |ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPOLossPointwise pol x group (ranker x group))| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le :=
    grpo_pl_tree_gap_bounded_ipw (model := model) (fstar := fstar)
      (pol := pol) (ranker := ranker) (g := g) (L := L)
      (h_group := h_group) (h_pol_lip := h_pol_lip)
      (h_ranker := h_ranker) (h_rum := h_rum)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  have hL : 0 ≤ (L : ℝ) := by positivity
  exact tree_gap_bound_transport_upper hL h_ipw h_upper

/-- GRPO-RL TreePO certificate after substituting an upper bound on IPW distortion. -/
theorem grpo_rl_tree_gap_bounded_by_sketch_upper
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
    (U : ℝ)
    (h_upper :
      ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le ≤ U) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group)| ≤
      (L : ℝ) * U := by
  have h_ipw :
      |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group)| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le :=
    grpo_rl_tree_gap_bounded_ipw (model := model) (fstar := fstar)
      (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta)
      (g := g) (L := L) (h_group := h_group)
      (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
      (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
      (h_rum := h_rum)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  have hL : 0 ≤ (L : ℝ) := by positivity
  exact tree_gap_bound_transport_upper hL h_ipw h_upper

/-- KLL algorithms provide hierarchical mergeability (fixed randomness). -/
theorem kll_hierarchical_mergeability_available
    {Ω α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    (A : ML.MergeableSummary.KLL.Algorithm Ω α μ) :
    ∀ ω : Ω, HierarchicalMergeable (A.build ω) A.valid A.merge := by
  exact ML.MergeableSummary.KLL.hierarchical_mergeability_of_algorithm A

/-- GK algorithms provide one-way mergeability in the Agarwal et al. interface. -/
theorem gk_one_way_mergeability_available
    {α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (A : ML.MergeableSummary.GK.Algorithm α) :
    ML.MergeableSummary.GK.corollary2_statement A := by
  exact ML.MergeableSummary.GK.algorithm_implies_gk_one_way_mergeability A

/-- One-way chunk-fold ingestion: sequential absorption preserves validity. -/
theorem one_way_chunk_fold_valid
    {α S : Type*}
    (V : ValidSketch α S)
    (mergeInto : S → Stream α → S)
    (h_one_way : OneWayMergeable V mergeInto)
    (xs0 : Stream α) (s0 : S)
    (hs0 : V.valid xs0 s0) :
    ∀ chunks : List (Stream α),
      V.valid (xs0 ++ List.flatten chunks) (chunks.foldl mergeInto s0) := by
  intro chunks
  induction chunks generalizing xs0 s0 with
  | nil =>
      simpa [List.flatten] using hs0
  | cons ys yss ih =>
      have hs1 : V.valid (xs0 ++ ys) (mergeInto s0 ys) :=
        h_one_way xs0 ys s0 hs0
      simpa [List.foldl, List.flatten, List.append_assoc] using
        ih (xs0 := xs0 ++ ys) (s0 := mergeInto s0 ys) hs1

/-- GK chunk-fold ingestion corollary (one-way merge setting, not full merge trees). -/
theorem gk_chunk_fold_ingestion_available
    {α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (A : ML.MergeableSummary.GK.Algorithm α)
    (xs0 : Stream α) (s0 : ML.MergeableSummary.GK.Summary α)
    (hs0 : ML.MergeableSummary.GK.validFor A.ε xs0 s0) :
    ∀ chunks : List (Stream α),
      ML.MergeableSummary.GK.validFor A.ε
        (xs0 ++ List.flatten chunks) (chunks.foldl A.mergeInto s0) := by
  let V : ValidSketch α (ML.MergeableSummary.GK.Summary α) :=
    { build := A.build
      valid := fun xs s => ML.MergeableSummary.GK.validFor A.ε xs s
      build_valid := by
        intro xs
        exact A.build_valid xs }
  have h_one_way_raw : ML.MergeableSummary.GK.corollary2_statement A :=
    gk_one_way_mergeability_available A
  have h_one_way : OneWayMergeable V A.mergeInto := by
    simpa [V, ML.MergeableSummary.GK.corollary2_statement] using h_one_way_raw
  simpa [V] using
    (one_way_chunk_fold_valid
      (V := V) (mergeInto := A.mergeInto)
      h_one_way xs0 s0 hs0)
