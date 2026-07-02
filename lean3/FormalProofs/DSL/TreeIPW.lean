import FormalProofs.DSL.JudgeCalibration
import FormalProofs.DSL.Honesty
import FormalProofs.DSL.CoreDefinitions
import FormalProofs.OPT.AuditBounds
import FormalProofs.OPT.SamplingModel
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.OracleUtility
import FormalProofs.DSL.IPWTheory

/-!
# FormalProofs/TreeIPW.lean

## Integration: IPW Framework for Tree-Based Preference Learning

This file integrates the Design-Based Supervised Learning (DSL) framework with
the existing tree summarization theory. It connects:

1. IPW estimates (Hajek estimator) to ViolationProb
2. Clustered standard errors to tree-level uncertainty
3. Judge calibration to training bounds
4. The full RLHF/DSL bound for tree preference learning

See RLHF_DSL_BANDIT_NOTES.md for the conceptual model.

### Key Results

- `TreeSample`: A logged sample from tree-based sampling
- `ipw_violation_rate`: IPW estimate of violation probability
- `ipw_union_bound_connection`: Links IPW estimates to existing union bound
- `dsl_bound`: Master theorem for design-based preference learning

### Method Agnosticism

This framework is agnostic to the downstream training method (DPO, GRPO, SFT, etc.).
The DSL/IPW machinery handles:
- Population-valid estimates via propensity weighting
- Uncertainty quantification via clustered SEs
- Surrogate error control via judge calibration

The training objective can be any preference learning loss.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal
open MeasureTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Section 2.5: Finite TreePO Population (HT Unbiasedness)

We model the TreePO sampling space as a finite population of triples
(document, node, group) and build the joint PMF explicitly. This lets us
apply the Bernoulli Horvitz–Thompson (HT) unbiasedness lemma directly.
-/

section TreePopulation

variable {Strings Node A : Type*} {k : ℕ}
variable [Fintype Strings] [Fintype Node] [Fintype A]
variable [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]

abbrev TreeUnit (Strings Node A : Type*) (k : ℕ) := Strings × Node × (Fin k → A)

/-- Joint PMF over tree units (doc → node → group). -/
noncomputable def treeUnitPMF
    (model : OPT.TreePreferenceSamplingModel Strings Node A k) :
    PMF (TreeUnit Strings Node A k) :=
  model.docDist.bind (fun x =>
    (model.nodeSampler x).bind (fun u =>
      PMF.map (fun g => (x, u, g)) (model.groupGen u)))

lemma treeUnitPMF_apply
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (x : Strings) (u : Node) (g : Fin k → A) :
    treeUnitPMF model (x, u, g) =
      model.docDist x * model.nodeSampler x u * model.groupGen u g := by
  classical
  unfold treeUnitPMF
  -- Expand the joint PMF and collapse each sum via `sum_ite_eq`.
  simp [PMF.bind_apply, PMF.map_apply, tsum_fintype]
  -- now finish by collapsing the nested finite sums
  have hmid :
      ∀ a,
        (∑ a1,
            (model.nodeSampler a) a1 *
              ∑ a2, if x = a ∧ u = a1 ∧ g = a2 then (model.groupGen a1) a2 else 0) =
          if x = a then (model.nodeSampler a) u * (model.groupGen u) g else 0 := by
    intro a
    by_cases hxa : x = a
    · subst hxa
      have hinner' :
          ∀ a1,
            (∑ a2, if u = a1 ∧ g = a2 then (model.groupGen a1) a2 else 0) =
              if u = a1 then (model.groupGen a1) g else 0 := by
        intro a1
        by_cases hua : u = a1
        · simp [hua]
        · simp [hua]
      simp [hinner', mul_ite]
    · simp [hxa]
  calc
    ∑ a,
        model.docDist a *
          ∑ a_1, (model.nodeSampler a) a_1 *
            ∑ a_2,
              (if x = a ∧ u = a_1 ∧ g = a_2 then (model.groupGen a_1) a_2 else 0)
        =
      ∑ a,
        model.docDist a *
          (if x = a then (model.nodeSampler a) u * (model.groupGen u) g else 0) := by
      simp [hmid]
    _ =
      model.docDist x * ((model.nodeSampler x) u * (model.groupGen u) g) := by
      simp [mul_ite]
    _ =
      model.docDist x * (model.nodeSampler x) u * (model.groupGen u) g := by
      simp [mul_assoc]

/-- Loss on a tree unit. -/
def treeUnitLoss
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ) :
    TreeUnit Strings Node A k → ℝ :=
  fun t => loss (model.nodeSpan t.2.1) t.2.2

/-- Exp over the joint PMF equals the nested TreePO expectation. -/
lemma Exp_treeUnitPMF_eq_expected
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ) :
    Exp (treeUnitPMF model) (treeUnitLoss model loss) =
      OPT.ExpectedTreePreferenceLoss model loss := by
  classical
  -- Expand definitions; `tsum_fintype` turns each `tsum` into a finite sum.
  unfold treeUnitLoss OPT.ExpectedTreePreferenceLoss Exp
  simp [tsum_fintype]
  -- Rewrite the joint expectation into iterated sums over (doc, node, group).
  have hLeft :
      (∑ z,
          (treeUnitPMF model z).toReal * loss (model.nodeSpan z.2.1) z.2.2) =
        ∑ x, ∑ u, ∑ g,
          (model.docDist x).toReal * (model.nodeSampler x u).toReal *
            (model.groupGen u g).toReal * loss (model.nodeSpan u) g := by
    simp [Fintype.sum_prod_type, treeUnitPMF_apply, ENNReal.toReal_mul, mul_assoc]
  have hRight :
      (∑ x,
          (model.docDist x).toReal *
            ∑ u,
              (model.nodeSampler x u).toReal *
                ∑ g, (model.groupGen u g).toReal * loss (model.nodeSpan u) g) =
        ∑ x, ∑ u, ∑ g,
          (model.docDist x).toReal * (model.nodeSampler x u).toReal *
            (model.groupGen u g).toReal * loss (model.nodeSpan u) g := by
    simp [Finset.mul_sum, mul_assoc]
  calc
    ∑ z, (treeUnitPMF model z).toReal * loss (model.nodeSpan z.2.1) z.2.2
        =
      ∑ x, ∑ u, ∑ g,
        (model.docDist x).toReal * (model.nodeSampler x u).toReal *
          (model.groupGen u g).toReal * loss (model.nodeSpan u) g := hLeft
    _ =
      ∑ x, (model.docDist x).toReal *
        ∑ u, (model.nodeSampler x u).toReal *
          ∑ g, (model.groupGen u g).toReal * loss (model.nodeSpan u) g := by
      symm
      exact hRight

/-- TreePO IPW unbiasedness under Bernoulli sampling. -/
theorem ipw_preference_loss_connection_tree
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi) (treeUnitLoss model loss) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      OPT.ExpectedTreePreferenceLoss model loss := by
  -- Bernoulli HT unbiasedness + identification of the target expectation.
  simpa [Exp_treeUnitPMF_eq_expected (model := model) (loss := loss)] using
    (htExp_unbiased (p := treeUnitPMF model) (pi := pi)
      (hpi_pos := hpi_pos) (hpi_le := hpi_le) (treeUnitLoss model loss))

/-- Citeable alias: the full realized node population is the TreePO finite population. -/
abbrev ExpectedFullNodePopulationPreferenceLoss := @OPT.ExpectedTreePreferenceLoss

/-- Citeable alias for the corresponding IPW unbiasedness statement. -/
abbrev full_node_population_preference_loss_unbiased := @ipw_preference_loss_connection_tree

end TreePopulation

/-!
## Section 2.6: Tree Distortion + IPW Gap Bridge

We define a tree-level distortion on the joint tree unit space and show
that the Bernoulli HT estimator is unbiased for its expectation. This
lets us translate any Lipschitz gap bound expressed in terms of expected
tree distortion into an equivalent bound in terms of the IPW estimator.
-/

section TreeDistortionIPW

variable {Strings Node A Y : Type*} {k : ℕ}
variable [Fintype Strings] [Fintype Node] [Fintype A]
variable [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
variable [PseudoMetricSpace Y]

/-- Tree-level distortion: compare oracle values at node span vs document. -/
def treeDistortion
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y) :
    TreeUnit Strings Node A k → ℝ :=
  fun t => D fstar (model.nodeSpan t.2.1) t.1

/-- Expected tree distortion under the joint tree-unit PMF. -/
def ExpectedTreeDistortion
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y) : ℝ :=
  Exp (treeUnitPMF model) (treeDistortion model fstar)

/-- Expected tree distortion reduces to the doc→node coupling (group cancels). -/
lemma ExpectedTreeDistortion_eq_docnode
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y) :
    ExpectedTreeDistortion model fstar =
      ∑' x, (model.docDist x).toReal *
        ∑' u, D fstar (model.nodeSpan u) x *
          (model.nodeSampler x u).toReal := by
  classical
  unfold ExpectedTreeDistortion treeDistortion Exp
  -- Expand the joint PMF and sum out groups (PMF sums to 1).
  simp [Fintype.sum_prod_type, treeUnitPMF_apply, tsum_fintype, ENNReal.toReal_mul,
    mul_assoc, mul_left_comm, mul_comm] -- expand to finite triple sum
  -- Sum over groups collapses to 1.
  have hsumg : ∀ u, (∑ g, (model.groupGen u g).toReal) = 1 := by
    intro u
    simpa [tsum_fintype] using (PMF.toReal_tsum_coe (model.groupGen u))
  -- Factor out the group sum.
  calc
    ∑ x,
        ∑ u,
          ∑ g,
            D fstar (model.nodeSpan u) x *
              ((model.docDist x).toReal * ((model.nodeSampler x u).toReal * (model.groupGen u g).toReal)) =
        ∑ x,
          ∑ u,
            (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) *
              (∑ g, (model.groupGen u g).toReal) := by
      refine Finset.sum_congr rfl ?_
      intro x hx
      refine Finset.sum_congr rfl ?_
      intro u hu
      -- pull constants outside the g-sum
      have hconst :
          ∑ g,
              D fstar (model.nodeSpan u) x *
                ((model.docDist x).toReal * ((model.nodeSampler x u).toReal * (model.groupGen u g).toReal)) =
            (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) *
              ∑ g, (model.groupGen u g).toReal := by
        -- rewrite summand and use sum_mul
        have h1 :
            ∑ g, (model.groupGen u g).toReal *
                (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) =
              (∑ g, (model.groupGen u g).toReal) *
                (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) := by
          simpa using
            (Finset.sum_mul (s := Finset.univ)
              (f := fun g => (model.groupGen u g).toReal)
              (a := D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal)).symm
        -- swap factors to match the original summand
        calc
          ∑ g,
              D fstar (model.nodeSpan u) x *
                ((model.docDist x).toReal * ((model.nodeSampler x u).toReal * (model.groupGen u g).toReal))
              =
            ∑ g, (model.groupGen u g).toReal *
                (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) := by
                  refine Finset.sum_congr rfl ?_
                  intro g hg
                  ring
          _ =
            (∑ g, (model.groupGen u g).toReal) *
                (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) := h1
          _ =
            (D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal) *
              (∑ g, (model.groupGen u g).toReal) := by
              ring
      exact hconst
    _ = ∑ x, ∑ u, D fstar (model.nodeSpan u) x * (model.docDist x).toReal * (model.nodeSampler x u).toReal := by
      simp [hsumg]
    _ = ∑ x, (model.docDist x).toReal * ∑ u, D fstar (model.nodeSpan u) x * (model.nodeSampler x u).toReal := by
      simp [Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm]

/-- Expected tree loss when a document-level expected loss `E_gen` is evaluated on node spans. -/
def ExpectedTreeEgen
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (E_gen : Strings → ℝ) : ℝ :=
  ∑' x, (model.docDist x).toReal *
    ∑' u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u)

/-- If the group generator is constant, `ExpectedTreePreferenceLoss` reduces to `ExpectedTreeEgen`. -/
lemma ExpectedTreePreferenceLoss_eq_Egen
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ)
    (g : PMF (Fin k → A))
    (h_group : ∀ u, model.groupGen u = g) :
    OPT.ExpectedTreePreferenceLoss model loss =
      ExpectedTreeEgen model (fun x => ∑' group, (g group).toReal * loss x group) := by
  classical
  unfold OPT.ExpectedTreePreferenceLoss ExpectedTreeEgen
  simp [h_group, tsum_fintype, mul_assoc, mul_left_comm, mul_comm]

/-- If the node group generator is induced by a doc-level generator,
`ExpectedTreePreferenceLoss` reduces to `ExpectedTreeEgen`. -/
lemma ExpectedTreePreferenceLoss_eq_Egen_nodeSpan
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ)
    (gen : GroupGenerator Strings A k)
    (h_group : ∀ u, model.groupGen u = gen (model.nodeSpan u)) :
    OPT.ExpectedTreePreferenceLoss model loss =
      ExpectedTreeEgen model (fun x => ∑' group, (gen x group).toReal * loss x group) := by
  classical
  unfold OPT.ExpectedTreePreferenceLoss ExpectedTreeEgen
  simp [h_group, tsum_fintype, mul_assoc, mul_left_comm, mul_comm]

/-- Tree gap bound from a Lipschitz expected loss, using the doc→node coupling. -/
lemma tree_gap_bounded_from_lipschitz
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (E_gen : Strings → ℝ)
    (L : ℝ≥0)
    (h_lip : ∀ x z, |E_gen x - E_gen z| ≤ (L : ℝ) * dist (fstar x) (fstar z)) :
    |∑' x, (model.docDist x).toReal * E_gen x -
        ExpectedTreeEgen model E_gen| ≤
      (L : ℝ) * ExpectedTreeDistortion model fstar := by
  classical
  -- Rewrite the goal as finite sums.
  simp [ExpectedTreeEgen, ExpectedTreeDistortion_eq_docnode, tsum_fintype]
  -- Expand the difference and rewrite with nodeSampler sum = 1.
  have hsum1 : ∀ x, (∑ u, (model.nodeSampler x u).toReal) = 1 := by
    intro x
    simpa [tsum_fintype] using (PMF.toReal_tsum_coe (model.nodeSampler x))
  have hsum_weight : ∀ x, ∑ u, (model.nodeSampler x u).toReal * E_gen x = E_gen x := by
    intro x
    calc
      ∑ u, (model.nodeSampler x u).toReal * E_gen x
          = (∑ u, (model.nodeSampler x u).toReal) * E_gen x := by
              -- use sum_mul and commutativity
              simpa [mul_comm] using
                (Finset.sum_mul (s := Finset.univ)
                  (f := fun u => (model.nodeSampler x u).toReal) (a := E_gen x)).symm
      _ = 1 * E_gen x := by simp [hsum1]
      _ = E_gen x := by simp
  -- Convert the difference to a double sum over x,u of (E_gen x - E_gen span u).
  have hdiff :
      (∑ x, (model.docDist x).toReal * E_gen x -
        ∑ x, (model.docDist x).toReal * ∑ u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u)) =
      ∑ x, (model.docDist x).toReal *
        ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u)) := by
    -- Expand RHS and use sum weights = 1
    have hR :
        ∑ x, (model.docDist x).toReal *
            ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u)) =
          ∑ x, (model.docDist x).toReal * E_gen x -
            ∑ x, (model.docDist x).toReal *
              ∑ u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u) := by
      calc
        ∑ x, (model.docDist x).toReal *
            ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u)) =
          ∑ x, (model.docDist x).toReal *
            (∑ u, (model.nodeSampler x u).toReal * E_gen x -
             ∑ u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u)) := by
            simp [Finset.sum_sub_distrib, sub_mul, mul_comm, mul_left_comm, mul_assoc]
        _ =
          ∑ x, (model.docDist x).toReal * ∑ u, (model.nodeSampler x u).toReal * E_gen x -
            ∑ x, (model.docDist x).toReal *
              ∑ u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u) := by
            simp [Finset.sum_sub_distrib, mul_sub]
        _ =
          ∑ x, (model.docDist x).toReal * E_gen x -
            ∑ x, (model.docDist x).toReal *
              ∑ u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u) := by
            simp [hsum_weight]
    exact hR.symm
  -- Bound by sum of absolute values and Lipschitz constant.
  calc
    |∑ x, (model.docDist x).toReal * E_gen x -
        ∑ x, (model.docDist x).toReal * ∑ u, (model.nodeSampler x u).toReal * E_gen (model.nodeSpan u)| =
        |∑ x, (model.docDist x).toReal *
            ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))| := by
      simp [hdiff]
    _ ≤ ∑ x, |(model.docDist x).toReal *
            ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))| := by
      simpa using (Finset.abs_sum_le_sum_abs (s := Finset.univ)
        (f := fun x => (model.docDist x).toReal *
          ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))))
    _ ≤ ∑ x, (model.docDist x).toReal *
          ∑ u, (model.nodeSampler x u).toReal * ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u))) := by
      -- Pull abs inside, then apply Lipschitz bound.
      refine Finset.sum_le_sum ?_
      intro x hx
      have h_nonneg : 0 ≤ (model.docDist x).toReal := ENNReal.toReal_nonneg
      -- Use |a*b| = a*|b| for a ≥ 0, then Lipschitz.
      have h_abs :
          |(model.docDist x).toReal *
              ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))| =
          (model.docDist x).toReal *
            |∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))| := by
        simp [abs_mul, h_nonneg]
      -- Now bound inner abs by sum of abs, then Lipschitz.
      have h_inner :
          |∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))| ≤
            ∑ u, (model.nodeSampler x u).toReal * ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u))) := by
        -- bound by sum of abs and apply Lipschitz pointwise
        refine (Finset.abs_sum_le_sum_abs (s := Finset.univ)
          (f := fun u => (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u)))).trans ?_
        refine Finset.sum_le_sum ?_
        intro u hu
        have hq : 0 ≤ (model.nodeSampler x u).toReal := ENNReal.toReal_nonneg
        have h_lip' := h_lip x (model.nodeSpan u)
        calc
          |(model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))|
              = (model.nodeSampler x u).toReal * |E_gen x - E_gen (model.nodeSpan u)| := by
                simp [abs_mul, hq]
          _ ≤ (model.nodeSampler x u).toReal * ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u))) := by
                exact mul_le_mul_of_nonneg_left h_lip' hq
      -- combine
      calc
        |(model.docDist x).toReal *
            ∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))|
            =
          (model.docDist x).toReal *
            |∑ u, (model.nodeSampler x u).toReal * (E_gen x - E_gen (model.nodeSpan u))| := h_abs
        _ ≤ (model.docDist x).toReal *
            ∑ u, (model.nodeSampler x u).toReal * ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u))) := by
            exact mul_le_mul_of_nonneg_left h_inner h_nonneg
    _ = (L : ℝ) * ∑ x, (model.docDist x).toReal *
          ∑ u, D fstar (model.nodeSpan u) x * ((model.nodeSampler x) u).toReal := by
      simp [D, dist_comm, mul_left_comm, mul_assoc, mul_comm, Finset.mul_sum]

/-- Expected loss Lipschitz with a doc-dependent generator.

Decomposes the shift into:
1) change in loss under fixed generator gen x, and
2) change in generator under fixed loss at z. -/
lemma expected_group_loss_lipschitz_gen_shift
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    [Fintype A] [DecidableEq A]
    (loss : Strings → (Fin k → A) → ℝ)
    (gen : GroupGenerator Strings A k)
    (fstar : Strings → Y)
    (L_loss L_gen : ℝ≥0)
    (M : ℝ) (hM : 0 ≤ M)
    (h_loss_bound : ∀ x g, |loss x g| ≤ M)
    (h_loss_lip : ∀ x z,
      |∑ g, (gen x g).toReal * loss x g -
        ∑ g, (gen x g).toReal * loss z g| ≤
        (L_loss : ℝ) * dist (fstar x) (fstar z))
    (h_gen_lip : GroupGeneratorLipschitzL1 gen fstar L_gen) :
    ∀ x z,
      |∑ g, (gen x g).toReal * loss x g -
        ∑ g, (gen z g).toReal * loss z g| ≤
        ((L_loss : ℝ) + M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by
  classical
  intro x z
  set Ex : ℝ := ∑ g, (gen x g).toReal * loss x g
  set Ez : ℝ := ∑ g, (gen z g).toReal * loss z g
  set C : ℝ := ∑ g, (gen x g).toReal * loss z g
  have hsplit : Ex - Ez = (Ex - C) + (C - Ez) := by ring
  have h1 : Ex - C = ∑ g, (gen x g).toReal * (loss x g - loss z g) := by
    calc
      Ex - C
          = ∑ g, (gen x g).toReal * loss x g - ∑ g, (gen x g).toReal * loss z g := by
              simp [Ex, C]
      _ = ∑ g, (gen x g).toReal * (loss x g - loss z g) := by
              symm
              simp [Finset.sum_sub_distrib, mul_sub]
  have h2 : C - Ez = ∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g := by
    calc
      C - Ez
          = ∑ g, (gen x g).toReal * loss z g - ∑ g, (gen z g).toReal * loss z g := by
              simp [C, Ez]
      _ = ∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g := by
              symm
              simp [Finset.sum_sub_distrib, sub_mul]
  have h_triangle :
      |Ex - Ez| ≤
        |∑ g, (gen x g).toReal * (loss x g - loss z g)| +
        |∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g| := by
    calc
      |Ex - Ez| = |(∑ g, (gen x g).toReal * (loss x g - loss z g)) +
        (∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g)| := by
          simp [hsplit, h1, h2]
      _ ≤ _ := by exact abs_add_le _ _
  have h_term1 :
      |∑ g, (gen x g).toReal * (loss x g - loss z g)| ≤
        (L_loss : ℝ) * dist (fstar x) (fstar z) := by
    have h := h_loss_lip x z
    simpa [Finset.sum_sub_distrib, mul_sub] using h
  have h_term2' :
      |∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g| ≤
        ∑ g, |(gen x g).toReal - (gen z g).toReal| * M := by
    refine (Finset.abs_sum_le_sum_abs (s := Finset.univ)
      (f := fun g => ((gen x g).toReal - (gen z g).toReal) * loss z g)).trans ?_
    refine Finset.sum_le_sum ?_
    intro g _
    have hq : 0 ≤ |(gen x g).toReal - (gen z g).toReal| := abs_nonneg _
    have hbound := h_loss_bound z g
    calc
      |((gen x g).toReal - (gen z g).toReal) * loss z g|
          = |(gen x g).toReal - (gen z g).toReal| * |loss z g| := by
              simp [abs_mul]
      _ ≤ |(gen x g).toReal - (gen z g).toReal| * M := by
              exact mul_le_mul_of_nonneg_left hbound hq
  have h_term2 :
      |∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g| ≤
        (M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by
    have h_sum :
        ∑ g, |(gen x g).toReal - (gen z g).toReal| * M =
          M * ∑ g, |(gen x g).toReal - (gen z g).toReal| := by
      have h' :
          ∑ g, |(gen x g).toReal - (gen z g).toReal| * M =
            ∑ g, M * |(gen x g).toReal - (gen z g).toReal| := by
        refine Finset.sum_congr rfl ?_
        intro g _; ring
      rw [h', Finset.mul_sum]
    have h_gen := h_gen_lip x z
    calc
      |∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g|
          ≤ ∑ g, |(gen x g).toReal - (gen z g).toReal| * M := h_term2'
      _ = M * ∑ g, |(gen x g).toReal - (gen z g).toReal| := h_sum
      _ ≤ M * ((L_gen : ℝ) * dist (fstar x) (fstar z)) := by
            exact mul_le_mul_of_nonneg_left h_gen hM
      _ = (M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by ring
  calc
    |∑ g, (gen x g).toReal * loss x g -
        ∑ g, (gen z g).toReal * loss z g|
        = |Ex - Ez| := by simp [Ex, Ez]
    _ ≤ |∑ g, (gen x g).toReal * (loss x g - loss z g)| +
          |∑ g, ((gen x g).toReal - (gen z g).toReal) * loss z g| := h_triangle
    _ ≤ (L_loss : ℝ) * dist (fstar x) (fstar z) +
        (M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by
          exact add_le_add h_term1 h_term2
    _ = ((L_loss : ℝ) + M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by ring

/-!
### Oracle Utility Gap (Tree Sampling)
-/

/-- Document-level expected oracle utility (truth label = f*(x)). -/
def ExpectedDocOracleUtility
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y) (u : OracleUtility2 Y) : ℝ :=
  ∑' x, (model.docDist x).toReal * u (fstar x) (fstar x)

/-- Expected doc-level label noise (truth label perturbation). -/
def ExpectedDocNoise
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y) : ℝ :=
  ∑' x, (model.docDist x).toReal * dist (fhat x) (fstar x)

/-- Tree-level expected oracle utility (node span vs. document truth). -/
def ExpectedTreeOracleUtility
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y) (u : OracleUtility2 Y) : ℝ :=
  ∑' x, (model.docDist x).toReal *
    ∑' u_node, (model.nodeSampler x u_node).toReal * u (fstar (model.nodeSpan u_node)) (fstar x)

/-- Citeable alias: document-level oracle target `y_doc = f*(x)`. -/
abbrev ExpectedDocumentOracleUtility := @ExpectedDocOracleUtility

/-- Citeable alias: tree/final-summary-side oracle utility target. -/
abbrev ExpectedFinalSummaryOracleUtility := @ExpectedTreeOracleUtility

/-- Document-level expected oracle utility with noisy truth labels. -/
def ExpectedDocOracleUtilityNoise
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y) : ℝ :=
  ∑' x, (model.docDist x).toReal * u (fstar x) (fhat x)

/-- Tree-level expected oracle utility with noisy truth labels. -/
def ExpectedTreeOracleUtilityNoise
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y) (u : OracleUtility2 Y) : ℝ :=
  ∑' x, (model.docDist x).toReal *
    ∑' u_node, (model.nodeSampler x u_node).toReal * u (fstar (model.nodeSpan u_node)) (fhat x)

/-- Oracle utility gap bound via expected tree distortion. -/
lemma tree_oracle_utility_gap_bounded
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L) :
    |ExpectedDocOracleUtility model fstar u -
        ExpectedTreeOracleUtility model fstar u| ≤
      (L : ℝ) * ExpectedTreeDistortion model fstar := by
  classical
  -- Rewrite the goal as finite sums.
  simp [ExpectedDocOracleUtility, ExpectedTreeOracleUtility,
        ExpectedTreeDistortion_eq_docnode, tsum_fintype]
  -- Expand the difference and rewrite with nodeSampler sum = 1.
  have hsum1 : ∀ x, (∑ u_node, (model.nodeSampler x u_node).toReal) = 1 := by
    intro x
    simpa [tsum_fintype] using (PMF.toReal_tsum_coe (model.nodeSampler x))
  have hsum_weight : ∀ x, ∑ u_node, (model.nodeSampler x u_node).toReal * u (fstar x) (fstar x) =
      u (fstar x) (fstar x) := by
    intro x
    calc
      ∑ u_node, (model.nodeSampler x u_node).toReal * u (fstar x) (fstar x)
          = (∑ u_node, (model.nodeSampler x u_node).toReal) * u (fstar x) (fstar x) := by
              simpa [mul_comm] using
                (Finset.sum_mul (s := Finset.univ)
                  (f := fun u_node => (model.nodeSampler x u_node).toReal) (a := u (fstar x) (fstar x))).symm
      _ = 1 * u (fstar x) (fstar x) := by simp [hsum1]
      _ = u (fstar x) (fstar x) := by simp
  -- Convert the difference to a double sum over x,u of (u(x,x) - u(span,x)).
  have hdiff :
      (∑ x, (model.docDist x).toReal * u (fstar x) (fstar x) -
        ∑ x, (model.docDist x).toReal *
          ∑ u_node, (model.nodeSampler x u_node).toReal * u (fstar (model.nodeSpan u_node)) (fstar x)) =
      ∑ x, (model.docDist x).toReal *
        ∑ u_node, (model.nodeSampler x u_node).toReal *
          (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x)) := by
    -- Expand RHS and use sum weights = 1
    have hR :
        ∑ x, (model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x)) =
          ∑ x, (model.docDist x).toReal * u (fstar x) (fstar x) -
            ∑ x, (model.docDist x).toReal *
              ∑ u_node, (model.nodeSampler x u_node).toReal *
                u (fstar (model.nodeSpan u_node)) (fstar x) := by
      calc
        ∑ x, (model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x)) =
          ∑ x, (model.docDist x).toReal *
            (∑ u_node, (model.nodeSampler x u_node).toReal * u (fstar x) (fstar x) -
             ∑ u_node, (model.nodeSampler x u_node).toReal *
                u (fstar (model.nodeSpan u_node)) (fstar x)) := by
            simp [Finset.sum_sub_distrib, sub_mul, mul_comm, mul_left_comm, mul_assoc]
        _ =
          ∑ x, (model.docDist x).toReal * ∑ u_node, (model.nodeSampler x u_node).toReal *
                u (fstar x) (fstar x) -
            ∑ x, (model.docDist x).toReal *
              ∑ u_node, (model.nodeSampler x u_node).toReal *
                u (fstar (model.nodeSpan u_node)) (fstar x) := by
            simp [Finset.sum_sub_distrib, mul_sub]
        _ =
          ∑ x, (model.docDist x).toReal * u (fstar x) (fstar x) -
            ∑ x, (model.docDist x).toReal *
              ∑ u_node, (model.nodeSampler x u_node).toReal *
                u (fstar (model.nodeSpan u_node)) (fstar x) := by
            simp [hsum_weight]
    exact hR.symm
  -- Bound by sum of absolute values and Lipschitz constant.
  calc
    |∑ x, (model.docDist x).toReal * u (fstar x) (fstar x) -
        ∑ x, (model.docDist x).toReal *
          ∑ u_node, (model.nodeSampler x u_node).toReal * u (fstar (model.nodeSpan u_node)) (fstar x)| =
        |∑ x, (model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))| := by
      simp [hdiff]
    _ ≤ ∑ x, |(model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))| := by
      simpa using (Finset.abs_sum_le_sum_abs (s := Finset.univ)
        (f := fun x => (model.docDist x).toReal *
          ∑ u_node, (model.nodeSampler x u_node).toReal *
            (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))))
    _ ≤ ∑ x, (model.docDist x).toReal *
          ∑ u_node, (model.nodeSampler x u_node).toReal *
            ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u_node))) := by
      -- Pull abs inside, then apply Lipschitz bound.
      refine Finset.sum_le_sum ?_
      intro x hx
      have h_nonneg : 0 ≤ (model.docDist x).toReal := ENNReal.toReal_nonneg
      -- Use |a*b| = a*|b| for a ≥ 0, then Lipschitz.
      have h_abs :
          |(model.docDist x).toReal *
              ∑ u_node, (model.nodeSampler x u_node).toReal *
                (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))| =
          (model.docDist x).toReal *
            |∑ u_node, (model.nodeSampler x u_node).toReal *
                (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))| := by
        simp [abs_mul, h_nonneg]
      -- Now bound inner abs by sum of abs, then Lipschitz.
      have h_inner :
          |∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))| ≤
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u_node))) := by
        -- bound by sum of abs and apply Lipschitz pointwise
        refine (Finset.abs_sum_le_sum_abs (s := Finset.univ)
          (f := fun u_node =>
            (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x)))).trans ?_
        refine Finset.sum_le_sum ?_
        intro u_node hu
        have hq : 0 ≤ (model.nodeSampler x u_node).toReal := ENNReal.toReal_nonneg
        have h_lip' :
            |u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x)| ≤
              (L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u_node)) := by
          -- Lipschitz in the first argument (truth fixed at f*(x))
          simpa using (hL (fstar x) (fstar (model.nodeSpan u_node)) (fstar x))
        calc
          |(model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))|
              = (model.nodeSampler x u_node).toReal *
                  |u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x)| := by
                simp [abs_mul, hq]
          _ ≤ (model.nodeSampler x u_node).toReal *
                ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u_node))) := by
                exact mul_le_mul_of_nonneg_left h_lip' hq
      -- combine
      calc
        |(model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))|
            =
          (model.docDist x).toReal *
            |∑ u_node, (model.nodeSampler x u_node).toReal *
                (u (fstar x) (fstar x) - u (fstar (model.nodeSpan u_node)) (fstar x))| := h_abs
        _ ≤ (model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              ((L : ℝ) * dist (fstar x) (fstar (model.nodeSpan u_node))) := by
            exact mul_le_mul_of_nonneg_left h_inner h_nonneg
    _ = (L : ℝ) * ∑ x, (model.docDist x).toReal *
          ∑ u_node, D fstar (model.nodeSpan u_node) x * ((model.nodeSampler x) u_node).toReal := by
      simp [D, dist_comm, mul_left_comm, mul_assoc, mul_comm, Finset.mul_sum]

/-- Doc-level noise bound for oracle utility (truth label perturbed). -/
lemma expected_doc_oracle_utility_noise_bound
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y)
    (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz2 u L) :
    |ExpectedDocOracleUtilityNoise model fstar fhat u -
        ExpectedDocOracleUtility model fstar u| ≤
      (L : ℝ) * ExpectedDocNoise model fstar fhat := by
  classical
  -- Reduce to finite sums.
  simp [ExpectedDocOracleUtilityNoise, ExpectedDocOracleUtility,
    ExpectedDocNoise, tsum_fintype]
  -- Rewrite the difference as a single sum.
  have hdiff :
      (∑ x, (model.docDist x).toReal * u (fstar x) (fhat x)) -
        ∑ x, (model.docDist x).toReal * u (fstar x) (fstar x) =
      ∑ x, (model.docDist x).toReal *
        (u (fstar x) (fhat x) - u (fstar x) (fstar x)) := by
    simp [Finset.sum_sub_distrib, mul_sub]
  -- Apply abs_sum_le_sum_abs and Lipschitz2 pointwise.
  calc
    |∑ x, (model.docDist x).toReal * u (fstar x) (fhat x) -
        ∑ x, (model.docDist x).toReal * u (fstar x) (fstar x)| =
        |∑ x, (model.docDist x).toReal *
            (u (fstar x) (fhat x) - u (fstar x) (fstar x))| := by
          simp [hdiff]
    _ ≤ ∑ x, |(model.docDist x).toReal *
            (u (fstar x) (fhat x) - u (fstar x) (fstar x))| := by
          simpa using (Finset.abs_sum_le_sum_abs (s := Finset.univ)
            (f := fun x =>
              (model.docDist x).toReal *
                (u (fstar x) (fhat x) - u (fstar x) (fstar x))))
    _ ≤ ∑ x, (model.docDist x).toReal *
          ((L : ℝ) * dist (fhat x) (fstar x)) := by
          refine Finset.sum_le_sum ?_
          intro x hx
          have h_nonneg : 0 ≤ (model.docDist x).toReal := ENNReal.toReal_nonneg
          have h_lip :
              |u (fstar x) (fhat x) - u (fstar x) (fstar x)| ≤
                (L : ℝ) * dist (fhat x) (fstar x) := by
            simpa using (hL (fstar x) (fhat x) (fstar x))
          calc
            |(model.docDist x).toReal *
                (u (fstar x) (fhat x) - u (fstar x) (fstar x))| =
                (model.docDist x).toReal *
                  |u (fstar x) (fhat x) - u (fstar x) (fstar x)| := by
                simp [abs_mul, h_nonneg]
            _ ≤ (model.docDist x).toReal *
                  ((L : ℝ) * dist (fhat x) (fstar x)) := by
                exact mul_le_mul_of_nonneg_left h_lip h_nonneg
    _ = (L : ℝ) * ∑ x, (model.docDist x).toReal * dist (fhat x) (fstar x) := by
          simp [Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc]

/-- Tree-level noise bound for oracle utility (truth label perturbed). -/
lemma expected_tree_oracle_utility_noise_bound
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y)
    (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz2 u L) :
    |ExpectedTreeOracleUtilityNoise model fstar fhat u -
        ExpectedTreeOracleUtility model fstar u| ≤
      (L : ℝ) * ExpectedDocNoise model fstar fhat := by
  classical
  -- Reduce to finite sums.
  simp [ExpectedTreeOracleUtilityNoise, ExpectedTreeOracleUtility,
    ExpectedDocNoise, tsum_fintype]
  -- Sum over nodes is 1.
  have hsum1 : ∀ x, (∑ u, (model.nodeSampler x u).toReal) = 1 := by
    intro x
    simpa [tsum_fintype] using (PMF.toReal_tsum_coe (model.nodeSampler x))
  -- Rewrite the difference as a single sum.
  have hdiff :
      (∑ x, (model.docDist x).toReal *
        ∑ u_node, (model.nodeSampler x u_node).toReal *
          u (fstar (model.nodeSpan u_node)) (fhat x)) -
        ∑ x, (model.docDist x).toReal *
          ∑ u_node, (model.nodeSampler x u_node).toReal *
            u (fstar (model.nodeSpan u_node)) (fstar x) =
      ∑ x, (model.docDist x).toReal *
        ∑ u_node, (model.nodeSampler x u_node).toReal *
          (u (fstar (model.nodeSpan u_node)) (fhat x) -
            u (fstar (model.nodeSpan u_node)) (fstar x)) := by
    simp [Finset.sum_sub_distrib, mul_sub]
  -- Bound by Lipschitz2 + collapse node sum.
  calc
    |∑ x, (model.docDist x).toReal *
        ∑ u_node, (model.nodeSampler x u_node).toReal *
          u (fstar (model.nodeSpan u_node)) (fhat x) -
      ∑ x, (model.docDist x).toReal *
        ∑ u_node, (model.nodeSampler x u_node).toReal *
          u (fstar (model.nodeSpan u_node)) (fstar x)| =
        |∑ x, (model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar (model.nodeSpan u_node)) (fhat x) -
                u (fstar (model.nodeSpan u_node)) (fstar x))| := by
          simp [hdiff]
    _ ≤ ∑ x, |(model.docDist x).toReal *
            ∑ u_node, (model.nodeSampler x u_node).toReal *
              (u (fstar (model.nodeSpan u_node)) (fhat x) -
                u (fstar (model.nodeSpan u_node)) (fstar x))| := by
          simpa using (Finset.abs_sum_le_sum_abs (s := Finset.univ)
            (f := fun x =>
              (model.docDist x).toReal *
                ∑ u_node, (model.nodeSampler x u_node).toReal *
                  (u (fstar (model.nodeSpan u_node)) (fhat x) -
                    u (fstar (model.nodeSpan u_node)) (fstar x))))
    _ ≤ ∑ x, (model.docDist x).toReal *
          ∑ u_node, (model.nodeSampler x u_node).toReal *
            ((L : ℝ) * dist (fhat x) (fstar x)) := by
          refine Finset.sum_le_sum ?_
          intro x hx
          have h_nonneg : 0 ≤ (model.docDist x).toReal := ENNReal.toReal_nonneg
          have h_inner :
              |∑ u_node, (model.nodeSampler x u_node).toReal *
                  (u (fstar (model.nodeSpan u_node)) (fhat x) -
                    u (fstar (model.nodeSpan u_node)) (fstar x))| ≤
                ∑ u_node, (model.nodeSampler x u_node).toReal *
                  ((L : ℝ) * dist (fhat x) (fstar x)) := by
            refine (Finset.abs_sum_le_sum_abs (s := Finset.univ)
              (f := fun u_node =>
                (model.nodeSampler x u_node).toReal *
                  (u (fstar (model.nodeSpan u_node)) (fhat x) -
                    u (fstar (model.nodeSpan u_node)) (fstar x)))).trans ?_
            refine Finset.sum_le_sum ?_
            intro u_node hu
            have hq : 0 ≤ (model.nodeSampler x u_node).toReal := ENNReal.toReal_nonneg
            have h_lip :
                |u (fstar (model.nodeSpan u_node)) (fhat x) -
                  u (fstar (model.nodeSpan u_node)) (fstar x)| ≤
                  (L : ℝ) * dist (fhat x) (fstar x) := by
              simpa using (hL (fstar (model.nodeSpan u_node)) (fhat x) (fstar x))
            calc
              |(model.nodeSampler x u_node).toReal *
                  (u (fstar (model.nodeSpan u_node)) (fhat x) -
                    u (fstar (model.nodeSpan u_node)) (fstar x))| =
                  (model.nodeSampler x u_node).toReal *
                    |u (fstar (model.nodeSpan u_node)) (fhat x) -
                      u (fstar (model.nodeSpan u_node)) (fstar x)| := by
                    simp [abs_mul, hq]
              _ ≤ (model.nodeSampler x u_node).toReal *
                    ((L : ℝ) * dist (fhat x) (fstar x)) := by
                    exact mul_le_mul_of_nonneg_left h_lip hq
          calc
            |(model.docDist x).toReal *
                ∑ u_node, (model.nodeSampler x u_node).toReal *
                  (u (fstar (model.nodeSpan u_node)) (fhat x) -
                    u (fstar (model.nodeSpan u_node)) (fstar x))| =
              (model.docDist x).toReal *
                |∑ u_node, (model.nodeSampler x u_node).toReal *
                    (u (fstar (model.nodeSpan u_node)) (fhat x) -
                      u (fstar (model.nodeSpan u_node)) (fstar x))| := by
                simp [abs_mul, h_nonneg]
            _ ≤ (model.docDist x).toReal *
                ∑ u_node, (model.nodeSampler x u_node).toReal *
                  ((L : ℝ) * dist (fhat x) (fstar x)) := by
                exact mul_le_mul_of_nonneg_left h_inner h_nonneg
    _ = ∑ x, (model.docDist x).toReal *
          ((L : ℝ) * dist (fhat x) (fstar x)) := by
          refine Finset.sum_congr rfl ?_
          intro x hx
          calc
            (model.docDist x).toReal *
                ∑ u_node, (model.nodeSampler x u_node).toReal *
                  ((L : ℝ) * dist (fhat x) (fstar x)) =
              (model.docDist x).toReal *
                ((∑ u_node, (model.nodeSampler x u_node).toReal) *
                  ((L : ℝ) * dist (fhat x) (fstar x))) := by
                have hsum :
                    ∑ u_node, (model.nodeSampler x u_node).toReal *
                      ((L : ℝ) * dist (fhat x) (fstar x)) =
                    (∑ u_node, (model.nodeSampler x u_node).toReal) *
                      ((L : ℝ) * dist (fhat x) (fstar x)) := by
                  simpa using
                    (Finset.sum_mul (s := Finset.univ)
                      (f := fun u_node => (model.nodeSampler x u_node).toReal)
                      (a := (L : ℝ) * dist (fhat x) (fstar x))).symm
                simp [hsum]
            _ = (model.docDist x).toReal * ((L : ℝ) * dist (fhat x) (fstar x)) := by
                simp [hsum1 x]
    _ = (L : ℝ) * ∑ x, (model.docDist x).toReal * dist (fhat x) (fstar x) := by
          -- factor out L
          calc
            ∑ x, (model.docDist x).toReal * ((L : ℝ) * dist (fhat x) (fstar x)) =
              ∑ x, (L : ℝ) * ((model.docDist x).toReal * dist (fhat x) (fstar x)) := by
                apply Finset.sum_congr rfl
                intro x hx
                ring
            _ = (L : ℝ) * ∑ x, (model.docDist x).toReal * dist (fhat x) (fstar x) := by
                simpa using
                  (Finset.mul_sum (s := Finset.univ)
                    (f := fun x => (model.docDist x).toReal * dist (fhat x) (fstar x))
                    (a := (L : ℝ))).symm

/-- End-to-end oracle utility bound with noisy truth labels. -/
theorem tree_oracle_utility_gap_noisy_bounded
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y)
    (u : OracleUtility2 Y) (L1 L2 : ℝ≥0)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2) :
    |ExpectedDocOracleUtility model fstar u -
        ExpectedTreeOracleUtilityNoise model fstar fhat u| ≤
      (L1 : ℝ) * ExpectedTreeDistortion model fstar +
      (L2 : ℝ) * ExpectedDocNoise model fstar fhat := by
  have h_tree :
      |ExpectedDocOracleUtility model fstar u -
          ExpectedTreeOracleUtility model fstar u| ≤
        (L1 : ℝ) * ExpectedTreeDistortion model fstar :=
    tree_oracle_utility_gap_bounded (model := model) (fstar := fstar) (u := u) (L := L1) hL1
  have h_noise :
      |ExpectedTreeOracleUtility model fstar u -
          ExpectedTreeOracleUtilityNoise model fstar fhat u| ≤
        (L2 : ℝ) * ExpectedDocNoise model fstar fhat := by
    simpa [abs_sub_comm] using
      (expected_tree_oracle_utility_noise_bound (model := model) (fstar := fstar)
        (fhat := fhat) (u := u) (L := L2) hL2)
  have htriangle :
      |ExpectedDocOracleUtility model fstar u -
          ExpectedTreeOracleUtilityNoise model fstar fhat u| ≤
        |ExpectedDocOracleUtility model fstar u -
            ExpectedTreeOracleUtility model fstar u| +
        |ExpectedTreeOracleUtility model fstar u -
            ExpectedTreeOracleUtilityNoise model fstar fhat u| := by
    have h :
        ExpectedDocOracleUtility model fstar u -
            ExpectedTreeOracleUtilityNoise model fstar fhat u =
          (ExpectedDocOracleUtility model fstar u -
              ExpectedTreeOracleUtility model fstar u) +
          (ExpectedTreeOracleUtility model fstar u -
              ExpectedTreeOracleUtilityNoise model fstar fhat u) := by
      ring
    calc
      |ExpectedDocOracleUtility model fstar u -
          ExpectedTreeOracleUtilityNoise model fstar fhat u| =
          |(ExpectedDocOracleUtility model fstar u -
              ExpectedTreeOracleUtility model fstar u) +
            (ExpectedTreeOracleUtility model fstar u -
              ExpectedTreeOracleUtilityNoise model fstar fhat u)| := by
            rw [h]
      _ ≤ |ExpectedDocOracleUtility model fstar u -
            ExpectedTreeOracleUtility model fstar u| +
          |ExpectedTreeOracleUtility model fstar u -
            ExpectedTreeOracleUtilityNoise model fstar fhat u| := by
            exact abs_add_le _ _
  calc
    |ExpectedDocOracleUtility model fstar u -
        ExpectedTreeOracleUtilityNoise model fstar fhat u| ≤
        |ExpectedDocOracleUtility model fstar u -
            ExpectedTreeOracleUtility model fstar u| +
        |ExpectedTreeOracleUtility model fstar u -
            ExpectedTreeOracleUtilityNoise model fstar fhat u| := htriangle
    _ ≤ (L1 : ℝ) * ExpectedTreeDistortion model fstar +
        (L2 : ℝ) * ExpectedDocNoise model fstar fhat := by
          exact add_le_add h_tree h_noise

/-- IPW unbiasedness for expected tree distortion. -/
theorem ipw_tree_distortion_unbiased
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω
      ∂bernoulliProductMeasure pi hpi_pos hpi_le =
      ExpectedTreeDistortion model fstar := by
  simpa [ExpectedTreeDistortion, treeDistortion] using
    (htExp_unbiased (p := treeUnitPMF model) (pi := pi)
      (hpi_pos := hpi_pos) (hpi_le := hpi_le) (treeDistortion model fstar))

/-- Uniform bound on the HT estimator for tree distortion. -/
theorem ipw_tree_distortion_abs_le
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ t, |treeDistortion model fstar t| ≤ D_max) :
    ∀ ω,
      |htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω| ≤ D_max / pi_min := by
  intro ω
  simpa using
    (htExpEstimator_abs_le (p := treeUnitPMF model) (pi := pi)
      (f := treeDistortion model fstar) (M := D_max) (hM := hD_max)
      (hbound := h_dist_bound) (hpi_pos := hpi_pos)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) ω)

/-- Pointwise second-moment bound for tree distortion HT estimator. -/
theorem ipw_tree_distortion_abs_sq_le
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (D_max : ℝ) (hD_max : 0 ≤ D_max)
    (h_dist_bound : ∀ t, |treeDistortion model fstar t| ≤ D_max) :
    ∀ ω,
      |htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω|^2 ≤ (D_max / pi_min)^2 := by
  intro ω
  simpa using
    (htExpEstimator_abs_sq_le (p := treeUnitPMF model) (pi := pi)
      (f := treeDistortion model fstar) (M := D_max) (hM := hD_max)
      (hbound := h_dist_bound) (hpi_pos := hpi_pos)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le) ω)

/-!
## Audit Robustness Wrappers

The appendix robustness note uses the uniform finite population of realized
audit units. These wrappers instantiate the generic HT robustness results for
TreePO distortion units.
-/

/-- Uniform finite-population HT estimator for TreePO distortion audit units. -/
abbrev treeAuditUniformDistortionEstimator
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ) :
    (TreeUnit Strings Node A k → Bool) → ℝ :=
  htUniformMeanEstimator pi (treeDistortion model fstar)

/-- Uniform finite-population variance proxy for TreePO distortion audit units. -/
abbrev treeAuditUniformDistortionVarianceProxy
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ) : ℝ :=
  htUniformMeanVarianceProxy pi (treeDistortion model fstar)

/-- TreePO distortion version of logged-marginal HT unbiasedness. -/
theorem treeAuditUniformDistortion_unbiased_of_logged_marginals
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (μ : Measure (TreeUnit Strings Node A k → Bool)) [IsFiniteMeasure μ]
    (hpi_pos : ∀ i, 0 < pi i)
    (h_marginal : ∀ i, ∫ ω, indicator i ω ∂μ = pi i) :
    ∫ ω, treeAuditUniformDistortionEstimator model fstar pi ω ∂μ =
      uniformFiniteMean (treeDistortion model fstar) :=
  htUniformMean_unbiased_of_logged_marginals
    (μ := μ) (pi := pi) (y := treeDistortion model fstar)
    hpi_pos h_marginal

/-- TreePO distortion version of the constrained-design variance bound. -/
theorem treeAuditUniformDistortion_variance_bound_of_constrained_design
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (μ : Measure (TreeUnit Strings Node A k → Bool))
    (pi : TreeUnit Strings Node A k → ℝ)
    (pi_min D_max : ℝ)
    (hcard_pos : 0 < (Fintype.card (TreeUnit Strings Node A k) : ℝ))
    (hcontrol :
      HTUniformMeanCovarianceControlled μ pi (treeDistortion model fstar))
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (hpi_min_pos : 0 < pi_min)
    (hpi_min_le_one : pi_min ≤ 1)
    (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (hD_nonneg : 0 ≤ D_max)
    (h_dist_bound : ∀ t, |treeDistortion model fstar t| ≤ D_max) :
    ProbabilityTheory.variance
        (treeAuditUniformDistortionEstimator model fstar pi) μ ≤
      (D_max^2 / (Fintype.card (TreeUnit Strings Node A k) : ℝ)) *
        (1 / pi_min - 1) :=
  htUniformMean_variance_bound_of_constrained_design
    (μ := μ) (pi := pi) (y := treeDistortion model fstar)
    (pi_min := pi_min) (D_max := D_max)
    hcard_pos hcontrol hpi_pos hpi_le hpi_min_pos hpi_min_le_one
    hpi_min_le hD_nonneg h_dist_bound

/-- TreePO distortion version specialized to the Bernoulli product sampling
measure used by the existing IPW theory. -/
theorem treeAuditUniformDistortion_variance_bound_of_independent_bernoulli
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (pi_min D_max : ℝ)
    (hcard_pos : 0 < (Fintype.card (TreeUnit Strings Node A k) : ℝ))
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (hpi_min_pos : 0 < pi_min)
    (hpi_min_le_one : pi_min ≤ 1)
    (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (hD_nonneg : 0 ≤ D_max)
    (h_dist_bound : ∀ t, |treeDistortion model fstar t| ≤ D_max) :
    ProbabilityTheory.variance
        (treeAuditUniformDistortionEstimator model fstar pi)
        (bernoulliProductMeasure pi hpi_pos hpi_le) ≤
      (D_max^2 / (Fintype.card (TreeUnit Strings Node A k) : ℝ)) *
        (1 / pi_min - 1) :=
  htUniformMean_variance_bound_of_independent_bernoulli
    (pi := pi) (y := treeDistortion model fstar)
    (pi_min := pi_min) (D_max := D_max)
    hcard_pos hpi_pos hpi_le hpi_min_pos hpi_min_le_one
    hpi_min_le hD_nonneg h_dist_bound

/-- Bridge lemma: any Lipschitz gap bound in terms of expected tree distortion
    can be rewritten in terms of the IPW estimator's expectation. -/
theorem tree_gap_bounded_by_ipw
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (gap L : ℝ)
    (h_gap : |gap| ≤ L * ExpectedTreeDistortion model fstar) :
    |gap| ≤
      L * ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  have h_ipw' :
      ExpectedTreeDistortion model fstar =
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
    simpa [ExpectedTreeDistortion, treeDistortion] using
      (ipw_tree_distortion_unbiased (model := model) (fstar := fstar)
        (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)).symm
  calc
    |gap| ≤ L * ExpectedTreeDistortion model fstar := h_gap
    _ = L * ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
          (treeDistortion model fstar) ω
        ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
      simp [h_ipw']

/-- Tree gap bound via IPW for a doc-level generator induced at nodes by `nodeSpan`. -/
theorem tree_gap_bounded_ipw_gen
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (loss : Strings → (Fin k → A) → ℝ)
    (gen : GroupGenerator Strings A k)
    (L : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = gen (model.nodeSpan u))
    (h_lip : ∀ x z,
      |(∑' group, (gen x group).toReal * loss x group) -
        (∑' group, (gen z group).toReal * loss z group)| ≤
        (L : ℝ) * dist (fstar x) (fstar z))
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    |ExpectedGroupLoss loss model.docDist gen -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  classical
  let E_gen : Strings → ℝ :=
    fun x => ∑' group, (gen x group).toReal * loss x group
  have h_gap0 :
      |∑' x, (model.docDist x).toReal * E_gen x - ExpectedTreeEgen model E_gen| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar :=
    tree_gap_bounded_from_lipschitz (model := model) (fstar := fstar)
      (E_gen := E_gen) (L := L) (h_lip := by
        intro x z
        simpa [E_gen] using h_lip x z)
  have h_base :
      ExpectedGroupLoss loss model.docDist gen =
        ∑' x, (model.docDist x).toReal * E_gen x := by
    simp [ExpectedGroupLoss, E_gen]
  have h_tree :
      OPT.ExpectedTreePreferenceLoss model loss = ExpectedTreeEgen model E_gen := by
    simpa [E_gen] using
      (ExpectedTreePreferenceLoss_eq_Egen_nodeSpan
        (model := model) (loss := loss) (gen := gen) (h_group := h_group))
  have h_gap :
      |ExpectedGroupLoss loss model.docDist gen -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar := by
    simpa [h_base, h_tree] using h_gap0
  exact tree_gap_bounded_by_ipw (model := model) (fstar := fstar) (pi := pi)
    (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (gap := ExpectedGroupLoss loss model.docDist gen -
      OPT.ExpectedTreePreferenceLoss model loss)
    (L := (L : ℝ)) h_gap

/-- Oracle utility gap bound via IPW estimator of tree distortion. -/
theorem tree_oracle_utility_gap_bounded_ipw
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (u : OracleUtility2 Y) (L : ℝ≥0)
    (hL : OracleUtilityLipschitz1 u L)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    |ExpectedDocOracleUtility model fstar u -
        ExpectedTreeOracleUtility model fstar u| ≤
      (L : ℝ) * ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  have h_gap :
      |ExpectedDocOracleUtility model fstar u -
          ExpectedTreeOracleUtility model fstar u| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar :=
    tree_oracle_utility_gap_bounded (model := model) (fstar := fstar) (u := u) (L := L) hL
  exact tree_gap_bounded_by_ipw (model := model) (fstar := fstar)
    (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (gap := ExpectedDocOracleUtility model fstar u - ExpectedTreeOracleUtility model fstar u)
    (L := L) h_gap

/-- Citeable alias emphasizing document labels vs. tree/final-summary utility. -/
abbrev document_label_vs_tree_oracle_utility_gap_bounded_ipw :=
  @tree_oracle_utility_gap_bounded_ipw

/-- End-to-end oracle utility bound with noisy truth labels (IPW form). -/
theorem tree_oracle_utility_gap_noisy_bounded_ipw
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar fhat : Strings → Y)
    (u : OracleUtility2 Y) (L1 L2 : ℝ≥0)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    |ExpectedDocOracleUtility model fstar u -
        ExpectedTreeOracleUtilityNoise model fstar fhat u| ≤
      (L1 : ℝ) * ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
            (treeDistortion model fstar) ω
          ∂bernoulliProductMeasure pi hpi_pos hpi_le +
      (L2 : ℝ) * ExpectedDocNoise model fstar fhat := by
  have h_gap :
      |ExpectedDocOracleUtility model fstar u -
          ExpectedTreeOracleUtilityNoise model fstar fhat u| ≤
        (L1 : ℝ) * ExpectedTreeDistortion model fstar +
        (L2 : ℝ) * ExpectedDocNoise model fstar fhat :=
    tree_oracle_utility_gap_noisy_bounded (model := model) (fstar := fstar)
      (fhat := fhat) (u := u) (L1 := L1) (L2 := L2) hL1 hL2
  have h_ipw :
      ExpectedTreeDistortion model fstar =
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
    simpa [ExpectedTreeDistortion, treeDistortion] using
      (ipw_tree_distortion_unbiased (model := model) (fstar := fstar)
        (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)).symm
  simpa [h_ipw] using h_gap

/-- Citeable decomposition: full-document truth vs. sampled tree-unit estimation. -/
abbrev document_truth_vs_sampled_tree_estimation_decomposition :=
  @tree_oracle_utility_gap_noisy_bounded_ipw

/-- GRPO-RL TreePO gap bound via IPW (constant group generator). -/
theorem grpo_pl_tree_gap_bounded_ipw
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
    |ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPOLossPointwise pol x group (ranker x group))| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  classical
  let loss : Strings → (Fin k → A) → ℝ :=
    fun x group => GRPOLossPointwise pol x group (ranker x group)
  let E_group : Strings → ℝ :=
    fun x => ∑' group, (g group).toReal * loss x group
  have h_lip_E : ∀ x z, |E_group x - E_group z| ≤ (L : ℝ) * dist (fstar x) (fstar z) :=
    fun x z =>
      E_group_grpo_lipschitz (pol := pol) (ranker := ranker) (fstar := fstar) (g := g)
        (L_grpo := L) h_pol_lip h_ranker x z (h_rum x z)
  have h_gap0 :
      |∑' x, (model.docDist x).toReal * E_group x - ExpectedTreeEgen model E_group| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar :=
    tree_gap_bounded_from_lipschitz (model := model) (fstar := fstar)
      (E_gen := E_group) (L := L) h_lip_E
  have h_base :
      ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) =
        ∑' x, (model.docDist x).toReal * E_group x := by
    simp [ExpectedGRPOLoss, E_group, loss]
  have h_tree :
      OPT.ExpectedTreePreferenceLoss model loss = ExpectedTreeEgen model E_group := by
    simpa [loss, E_group] using
      (ExpectedTreePreferenceLoss_eq_Egen (model := model) (loss := loss) (g := g) h_group)
  have h_gap :
      |ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar := by
    simpa [h_base, h_tree] using h_gap0
  exact tree_gap_bounded_by_ipw (model := model) (fstar := fstar) (pi := pi)
    (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (gap := ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
      OPT.ExpectedTreePreferenceLoss model loss)
    (L := (L : ℝ)) h_gap

/-- GRPO-PL TreePO gap bound via IPW (doc-dependent generator induced by node spans). -/
theorem grpo_pl_tree_gap_bounded_ipw_gen
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
    |ExpectedGRPOLoss pol ranker model.docDist gen -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPOLossPointwise pol x group (ranker x group))| ≤
      ((L_grpo : ℝ) + M * (L_gen : ℝ)) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  classical
  let loss : Strings → (Fin k → A) → ℝ :=
    fun x group => GRPOLossPointwise pol x group (ranker x group)
  have h_loss_lip :
      ∀ x z,
        |∑ group, (gen x group).toReal * loss x group -
          ∑ group, (gen x group).toReal * loss z group| ≤
          (L_grpo : ℝ) * dist (fstar x) (fstar z) := by
    intro x z
    have h :=
      E_group_grpo_lipschitz (pol := pol) (ranker := ranker) (fstar := fstar) (g := gen x)
        (L_grpo := L_grpo) h_pol_lip h_ranker x z (h_rum x z)
    simpa [loss, tsum_fintype] using h
  have h_lip :
      ∀ x z,
        |∑ group, (gen x group).toReal * loss x group -
          ∑ group, (gen z group).toReal * loss z group| ≤
          ((L_grpo : ℝ) + M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by
    intro x z
    exact expected_group_loss_lipschitz_gen_shift
      (loss := loss) (gen := gen) (fstar := fstar) (L_loss := L_grpo) (L_gen := L_gen)
      (M := M) (hM := hM) (h_loss_bound := h_loss_bound) (h_loss_lip := h_loss_lip)
      (h_gen_lip := h_gen_lip) x z
  have h_lip' :
      ∀ x z,
        |∑' group, (gen x group).toReal * loss x group -
          ∑' group, (gen z group).toReal * loss z group| ≤
          ((L_grpo : ℝ) + M * (L_gen : ℝ)) * dist (fstar x) (fstar z) := by
    intro x z
    simpa [tsum_fintype] using h_lip x z
  have hL_total : 0 ≤ (L_grpo : ℝ) + M * (L_gen : ℝ) := by
    have hL1 : 0 ≤ (L_grpo : ℝ) := by exact_mod_cast L_grpo.property
    have hL2 : 0 ≤ (L_gen : ℝ) := by exact_mod_cast L_gen.property
    have hM' : 0 ≤ M * (L_gen : ℝ) := mul_nonneg hM hL2
    exact add_nonneg hL1 hM'
  let L_total : ℝ≥0 := ⟨(L_grpo : ℝ) + M * (L_gen : ℝ), hL_total⟩
  have h_tree :=
    tree_gap_bounded_ipw_gen (model := model) (fstar := fstar) (loss := loss)
      (gen := gen) (L := L_total)
      (h_group := h_group) (h_lip := h_lip') (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)
  -- rewrite ExpectedGroupLoss to ExpectedGRPOLoss
  simpa [ExpectedGRPOLoss, ExpectedGroupLoss, loss, L_total] using h_tree

/-- GRPO-PL TreePO gap bound via IPW (fixed ranker, constant group generator). -/
theorem grpo_pl_tree_gap_bounded_ipw_plackettLuce_fixed_ranker
    {Strings Node A Y : Type*} [Monoid Strings] [PseudoMetricSpace Y]
    {k : ℕ} [Fintype Strings] [Fintype Node] [Fintype A]
    [DecidableEq Strings] [DecidableEq Node] [DecidableEq A]
    (hk : 0 < k)
    (model : OPT.TreePreferenceSamplingModel Strings Node A k)
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (g : PMF (Fin k → A))
    (L_pol : ℝ≥0)
    (h_group : ∀ u, model.groupGen u = g)
    (h_pol_lip : GRPOPolicyLipschitz pol fstar L_pol)
    (h_ranker : OracleIndexedRanker ranker fstar)
    (h_ranker_fixed : ∀ x z group, ranker x group = ranker z group)
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    |ExpectedGRPOLoss pol ranker model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPOLossPointwise pol x group (ranker x group))| ≤
      (((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  classical
  let L_grpo : ℝ≥0 := ((2 : ℝ≥0) * (k : ℝ≥0)) * L_pol
  have h_pol_lip_grpo : GRPOPolicyLipschitz pol fstar L_grpo :=
    grpo_policy_lipschitz_scaled_plackettLuce (hk := hk) pol fstar L_pol h_pol_lip
  have h_rum : ∀ x z,
      ExpectedGRPOLossLipschitz pol ranker fstar g L_grpo h_pol_lip_grpo h_ranker x z := by
    intro x z
    simpa [L_grpo, h_pol_lip_grpo] using
      (ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker_all
        (hk := hk) (pol := pol) (ranker := ranker) (fstar := fstar) (g := g)
        (L_pol := L_pol) (h_pol_lip := h_pol_lip) (h_ranker := h_ranker)
        (h_ranker_fixed := h_ranker_fixed) x z)
  simpa [L_grpo] using
    (grpo_pl_tree_gap_bounded_ipw (model := model) (fstar := fstar) (pol := pol)
      (ranker := ranker) (g := g) (L := L_grpo) (h_group := h_group)
      (h_pol_lip := h_pol_lip_grpo) (h_ranker := h_ranker) (h_rum := h_rum)
      (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le))

--
-- Existing GRPO-RL TreePO lemma below.
--
theorem grpo_rl_tree_gap_bounded_ipw
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
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group)| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  classical
  let loss : Strings → (Fin k → A) → ℝ :=
    fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group
  let E_group : Strings → ℝ :=
    fun x => ∑' group, (g group).toReal * loss x group
  have h_lip_E : ∀ x z, |E_group x - E_group z| ≤ (L : ℝ) * dist (fstar x) (fstar z) :=
    fun x z =>
      E_group_grpo_rl_lipschitz (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
        (reward := reward) (eps := eps) (beta := beta) (fstar := fstar) (g := g)
        (L := L) h_pol_lip h_old_lip h_ref_lip h_reward_lip x z (h_rum x z)
  have h_gap0 :
      |∑' x, (model.docDist x).toReal * E_group x - ExpectedTreeEgen model E_group| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar :=
    tree_gap_bounded_from_lipschitz (model := model) (fstar := fstar)
      (E_gen := E_group) (L := L) h_lip_E
  have h_base :
      ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) =
        ∑' x, (model.docDist x).toReal * E_group x := by
    simp [ExpectedGRPORLLoss, ExpectedGroupLoss, E_group, loss]
  have h_tree :
      OPT.ExpectedTreePreferenceLoss model loss = ExpectedTreeEgen model E_group := by
    simpa [loss, E_group] using
      (ExpectedTreePreferenceLoss_eq_Egen (model := model) (loss := loss) (g := g) h_group)
  have h_gap :
      |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
        (L : ℝ) * ExpectedTreeDistortion model fstar := by
    simpa [h_base, h_tree] using h_gap0
  exact tree_gap_bounded_by_ipw (model := model) (fstar := fstar) (pi := pi)
    (hpi_pos := hpi_pos) (hpi_le := hpi_le)
    (gap := ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
      OPT.ExpectedTreePreferenceLoss model loss)
    (L := (L : ℝ)) h_gap

/-- GRPO-RL TreePO gap bound via IPW from a primitive pointwise loss-Lipschitz
hypothesis on the finite group space. -/
theorem grpo_rl_tree_gap_bounded_ipw_of_pointwise
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
    (h_point :
      ∀ x z (group : Fin k → A),
        |GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta x group -
         GRPORLLossPointwise (k := k) pol pol_old pol_ref reward eps beta z group| ≤
        (L : ℝ) * dist (fstar x) (fstar z))
    (pi : TreeUnit Strings Node A k → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    |ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta model.docDist (fun _ => g) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => GRPORLLossPointwise pol pol_old pol_ref reward eps beta x group)| ≤
      (L : ℝ) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  have h_rum : ∀ x z,
      ExpectedGRPORLLossLipschitz k pol pol_old pol_ref reward eps beta fstar g L
        h_pol_lip h_old_lip h_ref_lip h_reward_lip x z := by
    intro x z
    exact ExpectedGRPORLLossLipschitz_of_pointwise_finite
      (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
      (reward := reward) (eps := eps) (beta := beta) (fstar := fstar)
      (g := g) (L := L)
      (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
      (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
      h_point x z
  exact grpo_rl_tree_gap_bounded_ipw
    (model := model) (fstar := fstar)
    (pol := pol) (pol_old := pol_old) (pol_ref := pol_ref)
    (reward := reward) (eps := eps) (beta := beta)
    (g := g) (L := L) (h_group := h_group)
    (h_pol_lip := h_pol_lip) (h_old_lip := h_old_lip)
    (h_ref_lip := h_ref_lip) (h_reward_lip := h_reward_lip)
    (h_rum := h_rum)
    (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le)

/-- DPO TreePO gap bound via IPW (pairs encoded as singleton groups). -/
theorem dpo_tree_gap_bounded_ipw
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
    -- Loss bound for DPO (used by E_pair_lipschitz_bounded)
    (M_loss : ℝ) (hM_loss : 0 ≤ M_loss)
    (h_loss_bound : ∀ (x : Strings) (p : A × A),
      |DPOLossPointwise pol pol_ref β x p.1 p.2| ≤ M_loss)
    (pi : TreeUnit Strings Node (A × A) 1 → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1) :
    |ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
        OPT.ExpectedTreePreferenceLoss model
          (fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2)| ≤
      (2 * |β| * (L_pol : ℝ)) *
        ∫ ω, htExpEstimator (p := treeUnitPMF model) (pi := pi)
              (treeDistortion model fstar) ω
            ∂bernoulliProductMeasure pi hpi_pos hpi_le := by
  classical
  let loss : Strings → (Fin 1 → (A × A)) → ℝ :=
    fun x group => DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2
  let E_pair : Strings → ℝ :=
    fun x => ∑' p, (gpair p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2
  -- Lipschitz constant for DPO expected loss
  let Ldpo : ℝ≥0 := (⟨2 * |β|, by nlinarith [abs_nonneg β]⟩) * L_pol
  have h_lip_E : ∀ x z, |E_pair x - E_pair z| ≤ (Ldpo : ℝ) * dist (fstar x) (fstar z) := by
    intro x z
    have h :=
      E_pair_lipschitz_bounded fstar pol pol_ref β L_pol gpair h_lip x z M_loss hM_loss
        (fun p => h_loss_bound x p) (fun p => h_loss_bound z p)
    have h' : |E_pair x - E_pair z| ≤ 2 * |β| * (L_pol : ℝ) * dist (fstar x) (fstar z) := by
      simpa [E_pair] using h
    -- rewrite to Ldpo
    simpa [Ldpo, mul_assoc] using h'
  -- Tree gap using Lipschitz E_pair
  have h_gap0 :
      |∑' x, (model.docDist x).toReal * E_pair x - ExpectedTreeEgen model E_pair| ≤
        (Ldpo : ℝ) * ExpectedTreeDistortion model fstar :=
    tree_gap_bounded_from_lipschitz (model := model) (fstar := fstar)
      (E_gen := E_pair) (L := Ldpo) h_lip_E
  -- ExpectedDPOLoss as Exp over E_pair
  have h_base :
      ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) =
        ∑' x, (model.docDist x).toReal * E_pair x := by
    simp [ExpectedDPOLoss, E_pair]
  -- Tree loss equals ExpectedTreeEgen with E_pair via constant groupGen
  have h_tree :
      OPT.ExpectedTreePreferenceLoss model loss = ExpectedTreeEgen model E_pair := by
    -- collapse the group generator (constant via h_group)
    let g1 : PMF (Fin 1 → (A × A)) := PMF.map (fun p : A × A => (fun _ : Fin 1 => p)) gpair
    have h_group' : ∀ u, model.groupGen u = g1 := by
      intro u; simpa [g1] using h_group u
    have h_tree1 :
        OPT.ExpectedTreePreferenceLoss model loss =
          ExpectedTreeEgen model (fun x => ∑' group, (g1 group).toReal * loss x group) :=
      ExpectedTreePreferenceLoss_eq_Egen (model := model) (loss := loss) (g := g1) h_group'
    -- show the inner expected loss equals E_pair
    have h_Egen_eq : (fun x => ∑ group, (g1 group).toReal * loss x group) = E_pair := by
      funext x
      classical
      -- collapse the mapped PMF over singleton groups
      have h_g1_toReal :
          ∀ group : Fin 1 → (A × A), (g1 group).toReal = (gpair (group 0)).toReal := by
        intro group
        have h_g1 : g1 group = gpair (group 0) := by
          -- equality of constant functions is equality of values
          have h_eq_fun :
              ∀ p : A × A, (group = (fun _ : Fin 1 => p)) ↔ group 0 = p := by
            intro p
            constructor
            · intro h; exact congrArg (fun f => f 0) h
            · intro h; funext i; fin_cases i; simp [h]
          -- rewrite the mapped PMF and collapse the sum
          simp [g1, PMF.map_apply, h_eq_fun, eq_comm, tsum_ite_eq]
        simp [h_g1]
      -- change variables using the `Fin 1` equivalence
      let e : (Fin 1 → (A × A)) ≃ (A × A) := Equiv.funUnique (Fin 1) (A × A)
      have hsum :
          ∑ group, (g1 group).toReal * loss x group =
            ∑ p, (gpair p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 := by
        have h1 :
            ∑ group, (g1 group).toReal * loss x group =
              ∑ group : Fin 1 → (A × A), (gpair (group 0)).toReal *
                DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2 := by
          simp [h_g1_toReal, loss]
        have h2 :
            ∑ group : Fin 1 → (A × A), (gpair (group 0)).toReal *
                DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2 =
              ∑ p, (gpair p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2 := by
          refine Fintype.sum_equiv e
              (f := fun group =>
                (gpair (group 0)).toReal *
                  DPOLossPointwise pol pol_ref β x (group 0).1 (group 0).2)
              (g := fun p => (gpair p).toReal * DPOLossPointwise pol pol_ref β x p.1 p.2) ?_
          intro group
          simp [e]
        simpa [h1] using h2
      simp [E_pair, tsum_fintype, hsum]
    simpa [tsum_fintype, h_Egen_eq] using h_tree1
  have h_gap :
      |ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
        OPT.ExpectedTreePreferenceLoss model loss| ≤
        (Ldpo : ℝ) * ExpectedTreeDistortion model fstar := by
    simpa [h_base, h_tree] using h_gap0
  -- Bridge to IPW estimator
  have h_ipw :=
    tree_gap_bounded_by_ipw (model := model) (fstar := fstar) (pi := pi)
      (hpi_pos := hpi_pos) (hpi_le := hpi_le)
      (gap := ExpectedDPOLoss pol pol_ref β model.docDist (fun _ => gpair) -
        OPT.ExpectedTreePreferenceLoss model loss)
      (L := (Ldpo : ℝ)) h_gap
  -- simplify constant Ldpo
  simpa [Ldpo, mul_assoc] using h_ipw

end TreeDistortionIPW

/-!
## Section 1: Tree Sample Structure
-/

/-- Type of node in the tree: leaf, internal (merge), or re-summarization -/
inductive NodeType where
  | leaf : NodeType
  | merge : NodeType
  | resummary : NodeType
deriving DecidableEq, Repr

/-- A logged sample from the tree-based sampling process.

This structure captures all information needed for IPW evaluation:
- Location: document, node, action IDs
- Outcome: oracle or judge score
- Propensity: sampling probability at each stage
- Metadata: policy version, oracle vs judge label -/
structure TreeSample where
  doc_id : String
  node_id : String
  action_id : String
  node_type : NodeType
  outcome : ℝ                   -- Generic outcome (e.g., loss or violation indicator)
  propensity : TreePropensity   -- Three-stage sampling probability
  policy_version : ℕ
  is_oracle_labeled : Bool      -- True if labeled by oracle, false if judge

namespace TreeSample

/-- Joint propensity for this sample -/
def jointPropensity (s : TreeSample) : ℝ := s.propensity.joint

/-- Weight for this sample (inverse propensity) -/
def weight (s : TreeSample) : ℝ := 1 / s.jointPropensity

/-- Convert to WeightedSample for IPW computation -/
def toWeightedSample (s : TreeSample) : WeightedSample ℝ :=
  ⟨s.outcome, s.jointPropensity, s.propensity.joint_pos⟩

/-- Is this a leaf node sample? -/
def isLeaf (s : TreeSample) : Bool := s.node_type == NodeType.leaf

/-- Is this a merge node sample? -/
def isMerge (s : TreeSample) : Bool := s.node_type == NodeType.merge

/-- Is this a re-summarization sample? -/
def isResummary (s : TreeSample) : Bool := s.node_type == NodeType.resummary

end TreeSample

/-!
## Section 1b: Label Provenance and Robust Propensity Handling

In applied settings we may have mixed truth-label sources (human annotations
and trusted dataset labels), and sampling probabilities may be heterogeneous
or not exactly known for every labeled unit. This section adds lightweight
metadata and a safe weighting fallback.
-/

/-- Logged joint propensity metadata with mandatory positivity floor.

- `logged = some p`: a per-unit propensity (by design or estimated/logged).
- `logged = none`: no unit-level propensity available; use the floor fallback.
- `floor`: strict lower bound used for safe weighting. -/
structure LoggedJointPropensity where
  logged : Option ℝ
  floor : ℝ
  h_floor_pos : 0 < floor
  h_logged_nonneg : ∀ p : ℝ, logged = some p → 0 ≤ p

namespace LoggedJointPropensity

/-- Effective propensity used for weighting under partial propensity knowledge. -/
def effective (m : LoggedJointPropensity) : ℝ :=
  match m.logged with
  | some p => max m.floor p
  | none => m.floor

lemma floor_le_effective (m : LoggedJointPropensity) :
    m.floor ≤ m.effective := by
  unfold effective
  cases h : m.logged with
  | none =>
      simp [h]
  | some p =>
      simp [h]

lemma effective_pos (m : LoggedJointPropensity) : 0 < m.effective := by
  have h_floor : m.floor ≤ m.effective := m.floor_le_effective
  exact lt_of_lt_of_le m.h_floor_pos h_floor

/-- If a logged propensity is available and dominates the floor, the effective
propensity equals the logged propensity exactly. -/
lemma effective_eq_logged_of_floor_le (m : LoggedJointPropensity)
    {p : ℝ} (h_logged : m.logged = some p) (h_floor_le : m.floor ≤ p) :
    m.effective = p := by
  simp [effective, h_logged, max_eq_right h_floor_le]

/-- If no logged propensity is available, the effective propensity is exactly
the floor fallback. -/
lemma effective_eq_floor_of_none (m : LoggedJointPropensity)
    (h_logged : m.logged = none) :
    m.effective = m.floor := by
  simp [effective, h_logged]

end LoggedJointPropensity

/-- Tree sample plus provenance metadata for truth/approx labels and
robust propensity handling. -/
structure TreeSampleWithProvenance where
  base : TreeSample
  truth_source : Option DSL.TruthLabelSource
  approx_source : Option DSL.ApproxLabelSource
  propensity_meta : LoggedJointPropensity

namespace TreeSampleWithProvenance

/-- Effective joint propensity used for robust weighting. -/
def effectiveJointPropensity (s : TreeSampleWithProvenance) : ℝ :=
  s.propensity_meta.effective

lemma effectiveJointPropensity_pos (s : TreeSampleWithProvenance) :
    0 < s.effectiveJointPropensity :=
  s.propensity_meta.effective_pos

/-- Robust inverse-propensity weight. -/
def weight (s : TreeSampleWithProvenance) : ℝ :=
  1 / s.effectiveJointPropensity

/-- Convert to WeightedSample using robust effective propensity. -/
def toWeightedSample (s : TreeSampleWithProvenance) : WeightedSample ℝ :=
  ⟨s.base.outcome, s.effectiveJointPropensity, s.effectiveJointPropensity_pos⟩

/-- Robust weight is bounded by the inverse floor. -/
lemma weight_le_inv_floor (s : TreeSampleWithProvenance) :
    s.weight ≤ 1 / s.propensity_meta.floor := by
  unfold weight effectiveJointPropensity
  exact one_div_le_one_div_of_le
    s.propensity_meta.h_floor_pos
    s.propensity_meta.floor_le_effective

/-- If the logged joint propensity is the actual propensity and it dominates the
floor, robust weighting uses that actual propensity exactly. -/
lemma effectiveJointPropensity_eq_logged
    (s : TreeSampleWithProvenance)
    {p : ℝ}
    (h_logged : s.propensity_meta.logged = some p)
    (h_floor_le : s.propensity_meta.floor ≤ p) :
    s.effectiveJointPropensity = p := by
  exact s.propensity_meta.effective_eq_logged_of_floor_le h_logged h_floor_le

/-- Whether the sample carries any truth label source metadata. -/
def hasTruthLabel (s : TreeSampleWithProvenance) : Bool :=
  s.truth_source.isSome

/-- Whether the truth label source is human annotation. -/
def isHumanTruth (s : TreeSampleWithProvenance) : Bool :=
  match s.truth_source with
  | some src => decide (src = DSL.TruthLabelSource.human)
  | _ => false

/-- Whether the truth label source is a trusted dataset label. -/
def isDatasetTruth (s : TreeSampleWithProvenance) : Bool :=
  match s.truth_source with
  | some src => decide (src = DSL.TruthLabelSource.dataset)
  | _ => false

end TreeSampleWithProvenance

/-!
## Section 2.4: Honest Sample Splitting (Train vs Eval)

We provide helpers for filtering logged samples to the evaluation split.
This supports "honest" inference where the tree/summarizer is learned on
training documents and the evaluation estimator uses only held-out docs.
-/

/-- Sample split over documents. -/
abbrev DocSplit := DSL.SampleSplit String

/-- Training subset of TreeSamples (by doc_id). -/
def trainSamples (split : DocSplit) (samples : List TreeSample) : List TreeSample :=
  DSL.filterTrain split (fun s => s.doc_id) samples

/-- Evaluation subset of TreeSamples (by doc_id). -/
def evalSamples (split : DocSplit) (samples : List TreeSample) : List TreeSample :=
  DSL.filterEval split (fun s => s.doc_id) samples

/-- K-fold evaluation subset for TreeSamples (by doc_id). -/
def evalSamplesFold (split : DSL.KFoldSplit String) (k : Fin split.K)
    (samples : List TreeSample) : List TreeSample :=
  DSL.filterEvalFold split k (fun s => s.doc_id) samples

/-!
## Section 2.5: Robust Sample Collections (Mixed Truth Sources)
-/

/-- Convert provenance-aware tree samples to weighted samples using robust
effective propensities. -/
def toWeightedSamplesRobust
    (samples : List TreeSampleWithProvenance) : List (WeightedSample ℝ) :=
  samples.map TreeSampleWithProvenance.toWeightedSample

/-- Robust IPW estimate of violation rate for provenance-aware samples. -/
def ipwViolationRateRobust
    (samples : List TreeSampleWithProvenance) : ℝ :=
  if samples.isEmpty then 0
  else hajekEstimator (toWeightedSamplesRobust samples)

/-- Honest (eval-only) robust IPW violation rate. -/
def honestIPWViolationRateRobust (split : DocSplit)
    (samples : List TreeSampleWithProvenance) : ℝ :=
  ipwViolationRateRobust (DSL.filterEval split (fun s => s.base.doc_id) samples)

/-- K-fold honest robust IPW violation rate (average over folds). -/
def kFoldIPWViolationRateRobust (split : DSL.KFoldSplit String)
    (samples : List TreeSampleWithProvenance) : ℝ :=
  (∑ k,
      ipwViolationRateRobust
        (DSL.filterEvalFold split k (fun s => s.base.doc_id) samples)) / (split.K : ℝ)

/-- Samples with any truth-label source metadata. -/
def truthLabeledSamples
    (samples : List TreeSampleWithProvenance) : List TreeSampleWithProvenance :=
  samples.filter TreeSampleWithProvenance.hasTruthLabel

/-- Samples whose truth source is human annotation. -/
def humanTruthSamples
    (samples : List TreeSampleWithProvenance) : List TreeSampleWithProvenance :=
  samples.filter TreeSampleWithProvenance.isHumanTruth

/-- Samples whose truth source is trusted dataset labels. -/
def datasetTruthSamples
    (samples : List TreeSampleWithProvenance) : List TreeSampleWithProvenance :=
  samples.filter TreeSampleWithProvenance.isDatasetTruth

/-!
## Section 2: Sample Collections and Filtering
-/

/-- Filter samples by node type -/
def filterByType (samples : List TreeSample) (t : NodeType) : List TreeSample :=
  samples.filter (fun s => s.node_type == t)

/-- Get leaf samples -/
def leafSamples (samples : List TreeSample) : List TreeSample :=
  filterByType samples NodeType.leaf

/-- Get merge samples -/
def mergeSamples (samples : List TreeSample) : List TreeSample :=
  filterByType samples NodeType.merge

/-- Get re-summarization samples -/
def resummarySamples (samples : List TreeSample) : List TreeSample :=
  filterByType samples NodeType.resummary

/-- Convert tree samples to weighted samples -/
def toWeightedSamples (samples : List TreeSample) : List (WeightedSample ℝ) :=
  samples.map TreeSample.toWeightedSample

/-!
## Section 3: IPW Estimates of Violation Rates
-/

/-- IPW estimate of violation rate for a set of samples.

Uses Hajek estimator: μ̂ = (Σ w_i y_i) / (Σ w_i)

The outcome is coded as 1 = violation, 0 = no violation.
So the estimate is the violation probability. -/
def ipwViolationRate (samples : List TreeSample) : ℝ :=
  if samples.isEmpty then 0
  else hajekEstimator (toWeightedSamples samples)

/-- Honest (eval-only) IPW violation rate. -/
def honestIPWViolationRate (split : DocSplit) (samples : List TreeSample) : ℝ :=
  ipwViolationRate (evalSamples split samples)

/-- K-fold honest IPW violation rate (average over folds). -/
def kFoldIPWViolationRate (split : DSL.KFoldSplit String)
    (samples : List TreeSample) : ℝ :=
  (∑ k, ipwViolationRate (evalSamplesFold split k samples)) / (split.K : ℝ)

/-- IPW estimate of leaf violation rate: p̂_leaf -/
def ipwLeafViolationRate (samples : List TreeSample) : ℝ :=
  ipwViolationRate (leafSamples samples)

/-- IPW estimate of merge violation rate: p̂_merge -/
def ipwMergeViolationRate (samples : List TreeSample) : ℝ :=
  ipwViolationRate (mergeSamples samples)

/-- IPW estimate of idempotence violation rate: p̂_idemp -/
def ipwIdempViolationRate (samples : List TreeSample) : ℝ :=
  ipwViolationRate (resummarySamples samples)

/-!
## Section 3b: IPW Estimates of Preference Losses
-/

/-- A logged sample for preference-loss evaluation at a tree node.

This is the TreePO analog of a weighted preference example:
- document and node identity
- k-wise candidate group
- observed loss value
- logged three-stage propensity -/
structure TreePreferenceSample (Strings Node A : Type*) (k : ℕ) where
  doc : Strings
  node : Node
  group : Fin k → A
  loss : ℝ
  propensity : TreePropensity
  policy_version : ℕ
  is_oracle_labeled : Bool

namespace TreePreferenceSample

/-- Convert to WeightedSample for IPW computation. -/
def toWeightedSample {Strings Node A : Type*} {k : ℕ}
    (s : TreePreferenceSample Strings Node A k) : WeightedSample ℝ :=
  ⟨s.loss, s.propensity.joint, s.propensity.joint_pos⟩

end TreePreferenceSample

/-- Preference-loss sample with provenance metadata and robust propensity info. -/
structure TreePreferenceSampleWithProvenance (Strings Node A : Type*) (k : ℕ) where
  base : TreePreferenceSample Strings Node A k
  truth_source : Option DSL.TruthLabelSource
  approx_source : Option DSL.ApproxLabelSource
  propensity_meta : LoggedJointPropensity

namespace TreePreferenceSampleWithProvenance

/-- Effective joint propensity used for robust weighting. -/
def effectiveJointPropensity {Strings Node A : Type*} {k : ℕ}
    (s : TreePreferenceSampleWithProvenance Strings Node A k) : ℝ :=
  s.propensity_meta.effective

lemma effectiveJointPropensity_pos {Strings Node A : Type*} {k : ℕ}
    (s : TreePreferenceSampleWithProvenance Strings Node A k) :
    0 < s.effectiveJointPropensity :=
  s.propensity_meta.effective_pos

/-- If the logged joint propensity is the actual propensity and it dominates the
floor, robust preference weighting uses that actual propensity exactly. -/
lemma effectiveJointPropensity_eq_logged {Strings Node A : Type*} {k : ℕ}
    (s : TreePreferenceSampleWithProvenance Strings Node A k)
    {p : ℝ}
    (h_logged : s.propensity_meta.logged = some p)
    (h_floor_le : s.propensity_meta.floor ≤ p) :
    s.effectiveJointPropensity = p := by
  exact s.propensity_meta.effective_eq_logged_of_floor_le h_logged h_floor_le

/-- Convert to WeightedSample using robust effective propensity. -/
def toWeightedSample {Strings Node A : Type*} {k : ℕ}
    (s : TreePreferenceSampleWithProvenance Strings Node A k) : WeightedSample ℝ :=
  ⟨s.base.loss, s.effectiveJointPropensity, s.effectiveJointPropensity_pos⟩

end TreePreferenceSampleWithProvenance

/-- Convert preference samples to weighted samples. -/
def toWeightedPrefSamples {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSample Strings Node A k)) : List (WeightedSample ℝ) :=
  samples.map TreePreferenceSample.toWeightedSample

/-- Convert provenance-aware preference samples to weighted samples with robust
effective propensities. -/
def toWeightedPrefSamplesRobust {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSampleWithProvenance Strings Node A k)) :
    List (WeightedSample ℝ) :=
  samples.map TreePreferenceSampleWithProvenance.toWeightedSample

/-- IPW estimate of preference loss for a set of samples. -/
def ipwPreferenceLoss {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSample Strings Node A k)) : ℝ :=
  if samples.isEmpty then 0
  else hajekEstimator (toWeightedPrefSamples samples)

/-- Honest (eval-only) IPW preference loss. -/
def honestIPWPreferenceLoss {Strings Node A : Type*} {k : ℕ}
    (split : DSL.SampleSplit Strings)
    (samples : List (TreePreferenceSample Strings Node A k)) : ℝ :=
  ipwPreferenceLoss (DSL.filterEval split (fun s => s.doc) samples)

/-- K-fold honest IPW preference loss (average over folds). -/
def kFoldIPWPreferenceLoss {Strings Node A : Type*} {k : ℕ}
    (split : DSL.KFoldSplit Strings)
    (samples : List (TreePreferenceSample Strings Node A k)) : ℝ :=
  (∑ kf, ipwPreferenceLoss (DSL.filterEvalFold split kf (fun s => s.doc) samples)) /
    (split.K : ℝ)

/-- Robust IPW estimate of preference loss for provenance-aware samples. -/
def ipwPreferenceLossRobust {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSampleWithProvenance Strings Node A k)) : ℝ :=
  if samples.isEmpty then 0
  else hajekEstimator (toWeightedPrefSamplesRobust samples)

/-- Honest (eval-only) robust IPW preference loss. -/
def honestIPWPreferenceLossRobust {Strings Node A : Type*} {k : ℕ}
    (split : DSL.SampleSplit Strings)
    (samples : List (TreePreferenceSampleWithProvenance Strings Node A k)) : ℝ :=
  ipwPreferenceLossRobust (DSL.filterEval split (fun s => s.base.doc) samples)

/-- K-fold honest robust IPW preference loss (average over folds). -/
def kFoldIPWPreferenceLossRobust {Strings Node A : Type*} {k : ℕ}
    (split : DSL.KFoldSplit Strings)
    (samples : List (TreePreferenceSampleWithProvenance Strings Node A k)) : ℝ :=
  (∑ kf, ipwPreferenceLossRobust (DSL.filterEvalFold split kf (fun s => s.base.doc) samples)) /
    (split.K : ℝ)

/-!
## K-Fold Concentration Wrappers (Honest Aggregation)
-/

/-- K-fold aggregation of deviation bounds for IPW violation rates. -/
theorem kFoldIPWViolationRate_bound
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.KFoldSplit String) (hK : 0 < split.K)
    (samples : Ω → List TreeSample)
    (mean : Fin split.K → ℝ) (r : Fin split.K → ℝ)
    (δ : Fin split.K → ENNReal)
    (hδ : ∀ k,
      μ {ω | |ipwViolationRate (evalSamplesFold split k (samples ω)) - mean k| ≥ r k} ≤ δ k) :
    μ {ω |
        |(∑ k, (ipwViolationRate (evalSamplesFold split k (samples ω)) - mean k)) / (split.K : ℝ)| ≥
          (∑ k, r k) / (split.K : ℝ)} ≤
      ∑' k, δ k := by
  simpa using
    (DSL.kfold_avg_bound (μ := μ) (hK := hK)
      (eval := fun k ω => ipwViolationRate (evalSamplesFold split k (samples ω)))
      (mean := mean) (r := fun k _ => r k) (δ := δ) (hδ := hδ))

/-- K-fold aggregation of deviation bounds for IPW preference losses. -/
theorem kFoldIPWPreferenceLoss_bound
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.KFoldSplit Strings) (hK : 0 < split.K)
    (samples : Ω → List (TreePreferenceSample Strings Node A k))
    (mean : Fin split.K → ℝ) (r : Fin split.K → ℝ)
    (δ : Fin split.K → ENNReal)
    (hδ : ∀ j,
      μ {ω |
          |ipwPreferenceLoss (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)) - mean j| ≥
            r j} ≤ δ j) :
    μ {ω |
        |(∑ j,
            (ipwPreferenceLoss (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)) - mean j)) /
          (split.K : ℝ)| ≥
          (∑ j, r j) / (split.K : ℝ)} ≤
      ∑' j, δ j := by
  simpa using
    (DSL.kfold_avg_bound (μ := μ) (hK := hK)
      (eval := fun j ω =>
        ipwPreferenceLoss (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)))
      (mean := mean) (r := fun j _ => r j) (δ := δ) (hδ := hδ))

/-- IPW preference loss equals Hajek estimator when samples are nonempty. -/
lemma ipwPreferenceLoss_eq_hajek {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSample Strings Node A k)) (h_nonempty : samples ≠ []) :
    ipwPreferenceLoss samples = hajekEstimator (toWeightedPrefSamples samples) := by
  cases samples with
  | nil => cases h_nonempty rfl
  | cons s ss =>
      simp [ipwPreferenceLoss, toWeightedPrefSamples]

/-- Empirical Bernstein radius for preference-loss IPW. -/
def ipwPreferenceEmpiricalBernsteinRadius {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSample Strings Node A k)) (δ range : ℝ) : ℝ :=
  empiricalBernsteinRadius (toWeightedPrefSamples samples) δ range

/-- Empirical Bernstein CI for preference-loss IPW. -/
def ipwPreferenceEmpiricalBernsteinCI {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSample Strings Node A k)) (δ range : ℝ) : ℝ × ℝ :=
  empiricalBernsteinCI (toWeightedPrefSamples samples) δ range

/-- Empirical Bernstein bound package for preference-loss IPW. -/
def ipwPreferenceEmpiricalBernsteinBound {Strings Node A : Type*} {k : ℕ}
    (samples : List (TreePreferenceSample Strings Node A k)) (δ range : ℝ) : EBBound :=
  empiricalBernsteinBound (toWeightedPrefSamples samples) δ range

/-- Empirical Bernstein concentration for IPW preference loss from a direct event bound. -/
theorem ipw_preference_loss_empirical_bernstein
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List (TreePreferenceSample Strings Node A k))
    (mean_true range : ℝ)
    (h_nonempty : ∀ ω, samples ω ≠ [])
    (δ : ℝ)
    (h_eb :
      μ {ω | |hajekEstimator (toWeightedPrefSamples (samples ω)) - mean_true| ≥
        empiricalBernsteinRadius (toWeightedPrefSamples (samples ω)) δ range} ≤ ENNReal.ofReal δ) :
    μ {ω | |ipwPreferenceLoss (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (toWeightedPrefSamples (samples ω)) δ range}
      ≤ ENNReal.ofReal δ := by
  have h_eq : ∀ ω,
      ipwPreferenceLoss (samples ω) = hajekEstimator (toWeightedPrefSamples (samples ω)) := by
    intro ω
    exact ipwPreferenceLoss_eq_hajek (samples ω) (h_nonempty ω)
  simpa [h_eq] using h_eb

/-- Compatibility wrapper: derive the direct event bound from EB axioms. -/
theorem ipw_preference_loss_empirical_bernstein_from_axioms
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List (TreePreferenceSample Strings Node A k))
    (mean_true range : ℝ)
    (h_nonempty : ∀ ω, samples ω ≠ [])
    (axioms :
      EmpiricalBernsteinAxioms μ (fun ω => toWeightedPrefSamples (samples ω)) mean_true range)
    (δ : ℝ) (hδ_pos : 0 < δ) (hδ_lt : δ < 1) :
    μ {ω | |ipwPreferenceLoss (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (toWeightedPrefSamples (samples ω)) δ range}
      ≤ ENNReal.ofReal δ := by
  have h_eb :=
    empiricalBernstein_bound_ennreal (μ := μ)
      (samples := fun ω => toWeightedPrefSamples (samples ω))
      (mean_true := mean_true) (range := range)
      (axioms := axioms)
      (δ := δ) (hδ_pos := hδ_pos) (hδ_lt := hδ_lt)
  exact ipw_preference_loss_empirical_bernstein (μ := μ) (samples := samples)
    (mean_true := mean_true) (range := range) (h_nonempty := h_nonempty) (δ := δ) h_eb

/-- Honest split: empirical Bernstein bound for eval-only IPW preference loss. -/
theorem honest_ipw_preference_empirical_bernstein
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.SampleSplit Strings)
    (samples : Ω → List (TreePreferenceSample Strings Node A k))
    (mean_true range : ℝ)
    (h_nonempty : ∀ ω, DSL.filterEval split (fun s => s.doc) (samples ω) ≠ [])
    (δ : ℝ)
    (h_eb :
      μ {ω | |hajekEstimator
          (toWeightedPrefSamples (DSL.filterEval split (fun s => s.doc) (samples ω))) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedPrefSamples (DSL.filterEval split (fun s => s.doc) (samples ω))) δ range}
        ≤ ENNReal.ofReal δ) :
    μ {ω | |honestIPWPreferenceLoss split (samples ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedPrefSamples (DSL.filterEval split (fun s => s.doc) (samples ω))) δ range}
      ≤ ENNReal.ofReal δ := by
  have h_eq : ∀ ω,
      honestIPWPreferenceLoss split (samples ω) =
        hajekEstimator (toWeightedPrefSamples
          (DSL.filterEval split (fun s => s.doc) (samples ω))) := by
    intro ω
    by_cases h_empty : DSL.filterEval split (fun s => s.doc) (samples ω) = []
    · have := (h_nonempty ω) h_empty
      exact (this.elim)
    · simp [honestIPWPreferenceLoss, ipwPreferenceLoss, h_empty]
  simpa [h_eq] using h_eb

/-- Compatibility wrapper: derive honest preference-loss EB event from EB axioms. -/
theorem honest_ipw_preference_empirical_bernstein_from_axioms
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.SampleSplit Strings)
    (samples : Ω → List (TreePreferenceSample Strings Node A k))
    (mean_true range : ℝ)
    (h_nonempty : ∀ ω, DSL.filterEval split (fun s => s.doc) (samples ω) ≠ [])
    (axioms :
      EmpiricalBernsteinAxioms μ
        (fun ω => toWeightedPrefSamples (DSL.filterEval split (fun s => s.doc) (samples ω)))
        mean_true range)
    (δ : ℝ) (hδ_pos : 0 < δ) (hδ_lt : δ < 1) :
    μ {ω | |honestIPWPreferenceLoss split (samples ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedPrefSamples (DSL.filterEval split (fun s => s.doc) (samples ω))) δ range}
      ≤ ENNReal.ofReal δ := by
  have h_eb :=
    empiricalBernstein_bound_ennreal (μ := μ)
      (samples := fun ω =>
        toWeightedPrefSamples (DSL.filterEval split (fun s => s.doc) (samples ω)))
      (mean_true := mean_true) (range := range)
      (axioms := axioms)
      (δ := δ) (hδ_pos := hδ_pos) (hδ_lt := hδ_lt)
  exact honest_ipw_preference_empirical_bernstein (μ := μ) (split := split) (samples := samples)
    (mean_true := mean_true) (range := range) (h_nonempty := h_nonempty) (δ := δ) h_eb

/-- Connection theorem: Bernoulli HT estimator is unbiased for `Exp p loss`.

This is the formal IPW unbiasedness statement used by TreePO. It is stated in
terms of the Bernoulli product measure over inclusion indicators and the
HT estimator from FormalProbability.

To instantiate for TreePO, take:
- `ι = Strings × Node × (Fin k → A)` (finite support)
- `p` as the joint PMF from the tree sampling model
- `loss` as the tree-level loss function with PMF weight folded into `p`.
-/
theorem ipw_preference_loss_connection
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (p : PMF ι) (pi : ι → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (loss : ι → ℝ) :
    ∫ ω, htExpEstimator p pi loss ω ∂bernoulliProductMeasure pi hpi_pos hpi_le = Exp p loss := by
  simpa using
    (htExp_unbiased (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) loss)

/-!
## Section 4: Connection to Union Bound
-/

/-- IPW estimate of the union bound.

This is the IPW version of the audit bound:
  Δ̂ = N × p̂_leaf + M × p̂_merge + (R-1) × p̂_idemp

Where N, M, R are tree structure parameters. -/
def ipwUnionBound (samples : List TreeSample)
    (N M R : ℕ) : ℝ :=
  N * ipwLeafViolationRate samples +
  M * ipwMergeViolationRate samples +
  (R - 1) * ipwIdempViolationRate samples

/-- Connection theorem: Bernoulli HT estimator is unbiased for `Exp p violationInd`. -/
theorem ipw_violation_rate_connection
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (p : PMF ι) (pi : ι → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (violationInd : ι → ℝ) :
    ∫ ω, htExpEstimator p pi violationInd ω ∂bernoulliProductMeasure pi hpi_pos hpi_le = Exp p violationInd := by
  simpa using
    (htExp_unbiased (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) violationInd)

/-!
## Section 3c: Concentration for Violation Rates
-/

/-- Hoeffding bound for HT estimation of a violation indicator. -/
theorem ipw_violation_rate_hoeffding_bound
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (p : PMF ι) (pi : ι → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (violationInd : ι → ℝ)
    (h0 : ∀ i, 0 ≤ violationInd i)
    (h1 : ∀ i, violationInd i ≤ (1 : ℝ))
    (pi_min : ℝ) (hpi_min_pos : 0 < pi_min) (hpi_min_le : ∀ i, pi_min ≤ pi i)
    (ε : ℝ) (hε : 0 < ε) :
    (bernoulliProductMeasure pi hpi_pos hpi_le).real
      {ω | |htExpEstimator p pi violationInd ω - Exp p violationInd| ≥ ε} ≤
      2 * Real.exp (- ε^2 / (8 * (Fintype.card ι) * (1 / pi_min)^2)) := by
  simpa using
    (htExpEstimator_hoeffding_bound_indicator (p := p) (pi := pi)
      (hpi_pos := hpi_pos) (hpi_le := hpi_le)
      (f := violationInd) (h0 := h0) (h1 := h1)
      (pi_min := pi_min) (hpi_min_pos := hpi_min_pos) (hpi_min_le := hpi_min_le)
      (ε := ε) (hε := hε))

/-- IPW violation rate equals Hajek estimator when samples are nonempty. -/
lemma ipwViolationRate_eq_hajek (samples : List TreeSample) (h_nonempty : samples ≠ []) :
    ipwViolationRate samples = hajekEstimator (toWeightedSamples samples) := by
  cases samples with
  | nil => cases h_nonempty rfl
  | cons s ss =>
      simp [ipwViolationRate, toWeightedSamples]

/-- Empirical Bernstein radius for violation-rate IPW (range = 1). -/
def ipwViolationEmpiricalBernsteinRadius (samples : List TreeSample) (δ : ℝ) : ℝ :=
  empiricalBernsteinRadius (toWeightedSamples samples) δ 1

/-- Empirical Bernstein CI for violation-rate IPW (range = 1). -/
def ipwViolationEmpiricalBernsteinCI (samples : List TreeSample) (δ : ℝ) : ℝ × ℝ :=
  empiricalBernsteinCI (toWeightedSamples samples) δ 1

/-- Empirical Bernstein bound package for violation-rate IPW. -/
def ipwViolationEmpiricalBernsteinBound (samples : List TreeSample) (δ : ℝ) : EBBound :=
  empiricalBernsteinBound (toWeightedSamples samples) δ 1

/-- Empirical Bernstein concentration for the IPW violation rate from a direct event bound. -/
theorem ipw_violation_rate_empirical_bernstein
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List TreeSample)
    (mean_true : ℝ)
    (h_nonempty : ∀ ω, samples ω ≠ [])
    (δ : ℝ)
    (h_eb :
      μ {ω | |hajekEstimator (toWeightedSamples (samples ω)) - mean_true| ≥
        empiricalBernsteinRadius (toWeightedSamples (samples ω)) δ 1} ≤ ENNReal.ofReal δ) :
    μ {ω | |ipwViolationRate (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (toWeightedSamples (samples ω)) δ 1}
      ≤ ENNReal.ofReal δ := by
  have h_eq : ∀ ω,
      ipwViolationRate (samples ω) = hajekEstimator (toWeightedSamples (samples ω)) := by
    intro ω
    exact ipwViolationRate_eq_hajek (samples ω) (h_nonempty ω)
  simpa [h_eq] using h_eb

/-- Compatibility wrapper: derive violation-rate EB event from EB axioms. -/
theorem ipw_violation_rate_empirical_bernstein_from_axioms
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List TreeSample)
    (mean_true : ℝ)
    (h_nonempty : ∀ ω, samples ω ≠ [])
    (axioms :
      EmpiricalBernsteinAxioms μ (fun ω => toWeightedSamples (samples ω)) mean_true 1)
    (δ : ℝ) (hδ_pos : 0 < δ) (hδ_lt : δ < 1) :
    μ {ω | |ipwViolationRate (samples ω) - mean_true| ≥
        empiricalBernsteinRadius (toWeightedSamples (samples ω)) δ 1}
      ≤ ENNReal.ofReal δ := by
  have h_eb :=
    empiricalBernstein_bound_ennreal (μ := μ)
      (samples := fun ω => toWeightedSamples (samples ω))
      (mean_true := mean_true) (range := 1)
      (axioms := axioms)
      (δ := δ) (hδ_pos := hδ_pos) (hδ_lt := hδ_lt)
  exact ipw_violation_rate_empirical_bernstein (μ := μ) (samples := samples)
    (mean_true := mean_true) (h_nonempty := h_nonempty) (δ := δ) h_eb

/-- Honest split: empirical Bernstein bound for eval-only IPW violation rate. -/
theorem honest_ipw_violation_empirical_bernstein
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DocSplit)
    (samples : Ω → List TreeSample)
    (mean_true : ℝ)
    (h_nonempty : ∀ ω, evalSamples split (samples ω) ≠ [])
    (δ : ℝ)
    (h_eb :
      μ {ω | |hajekEstimator (toWeightedSamples (evalSamples split (samples ω))) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (evalSamples split (samples ω))) δ 1} ≤ ENNReal.ofReal δ) :
    μ {ω | |honestIPWViolationRate split (samples ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (evalSamples split (samples ω))) δ 1}
      ≤ ENNReal.ofReal δ := by
  have h_eq : ∀ ω,
      honestIPWViolationRate split (samples ω) =
        hajekEstimator (toWeightedSamples (evalSamples split (samples ω))) := by
    intro ω
    by_cases h_empty : evalSamples split (samples ω) = []
    · have := (h_nonempty ω) h_empty
      exact (this.elim)
    · simp [honestIPWViolationRate, ipwViolationRate, h_empty]
  simpa [h_eq] using h_eb

/-- Compatibility wrapper: derive honest violation-rate EB event from EB axioms. -/
theorem honest_ipw_violation_empirical_bernstein_from_axioms
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DocSplit)
    (samples : Ω → List TreeSample)
    (mean_true : ℝ)
    (h_nonempty : ∀ ω, evalSamples split (samples ω) ≠ [])
    (axioms :
      EmpiricalBernsteinAxioms μ
        (fun ω => toWeightedSamples (evalSamples split (samples ω))) mean_true 1)
    (δ : ℝ) (hδ_pos : 0 < δ) (hδ_lt : δ < 1) :
    μ {ω | |honestIPWViolationRate split (samples ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (evalSamples split (samples ω))) δ 1}
      ≤ ENNReal.ofReal δ := by
  have h_eb :=
    empiricalBernstein_bound_ennreal (μ := μ)
      (samples := fun ω => toWeightedSamples (evalSamples split (samples ω)))
      (mean_true := mean_true) (range := 1)
      (axioms := axioms)
      (δ := δ) (hδ_pos := hδ_pos) (hδ_lt := hδ_lt)
  exact honest_ipw_violation_empirical_bernstein (μ := μ) (split := split) (samples := samples)
    (mean_true := mean_true) (h_nonempty := h_nonempty) (δ := δ) h_eb


/-- K-fold empirical Bernstein radius for IPW violation rates. -/
def kFoldViolationEmpiricalBernsteinRadius
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (δ : Fin split.K → ℝ) : ℝ :=
  (∑ k,
      empiricalBernsteinRadius
        (toWeightedSamples (evalSamplesFold split k samples)) (δ k) 1) /
    (split.K : ℝ)

/-- K-fold empirical Bernstein CI for IPW violation rates. -/
def kFoldViolationEmpiricalBernsteinCI
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (δ : Fin split.K → ℝ) : ℝ × ℝ :=
  let μhat := kFoldIPWViolationRate split samples
  let r := kFoldViolationEmpiricalBernsteinRadius split samples δ
  (μhat - r, μhat + r)

/-- K-fold empirical Bernstein bound package for IPW violation rates. -/
def kFoldViolationEmpiricalBernsteinBound
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (δ : Fin split.K → ℝ) : EBBound :=
  { center := kFoldIPWViolationRate split samples
    radius := kFoldViolationEmpiricalBernsteinRadius split samples δ }

/-- K-fold empirical Bernstein radius for IPW preference losses. -/
def kFoldPreferenceEmpiricalBernsteinRadius {Strings Node A : Type*} {k : ℕ}
    (split : DSL.KFoldSplit Strings)
    (samples : List (TreePreferenceSample Strings Node A k))
    (δ : Fin split.K → ℝ) (range : ℝ) : ℝ :=
  (∑ j,
      empiricalBernsteinRadius
        (toWeightedPrefSamples
          (DSL.filterEvalFold split j (fun s => s.doc) samples))
        (δ j) range) /
    (split.K : ℝ)

/-- K-fold empirical Bernstein CI for IPW preference losses. -/
def kFoldPreferenceEmpiricalBernsteinCI {Strings Node A : Type*} {k : ℕ}
    (split : DSL.KFoldSplit Strings)
    (samples : List (TreePreferenceSample Strings Node A k))
    (δ : Fin split.K → ℝ) (range : ℝ) : ℝ × ℝ :=
  let μhat := kFoldIPWPreferenceLoss split samples
  let r := kFoldPreferenceEmpiricalBernsteinRadius split samples δ range
  (μhat - r, μhat + r)

/-- K-fold empirical Bernstein bound package for IPW preference losses. -/
def kFoldPreferenceEmpiricalBernsteinBound {Strings Node A : Type*} {k : ℕ}
    (split : DSL.KFoldSplit Strings)
    (samples : List (TreePreferenceSample Strings Node A k))
    (δ : Fin split.K → ℝ) (range : ℝ) : EBBound :=
  { center := kFoldIPWPreferenceLoss split samples
    radius := kFoldPreferenceEmpiricalBernsteinRadius split samples δ range }

/-- K-fold empirical Bernstein bound for IPW violation rates. -/
theorem kFoldIPWViolationRate_empirical_bernstein
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.KFoldSplit String) (hK : 0 < split.K)
    (samples : Ω → List TreeSample)
    (mean : Fin split.K → ℝ)
    (δ : Fin split.K → ℝ)
    (hδ_fold :
      ∀ k,
        μ {ω |
            |ipwViolationRate (evalSamplesFold split k (samples ω)) - mean k| ≥
              empiricalBernsteinRadius
                (toWeightedSamples (evalSamplesFold split k (samples ω))) (δ k) 1} ≤
          ENNReal.ofReal (δ k)) :
    μ {ω |
        |(∑ k,
            (ipwViolationRate (evalSamplesFold split k (samples ω)) - mean k)) /
          (split.K : ℝ)| ≥
          kFoldViolationEmpiricalBernsteinRadius split (samples ω) δ} ≤
      ∑' k, ENNReal.ofReal (δ k) := by
  have h :=
    DSL.kfold_avg_bound (μ := μ) (hK := hK)
      (eval := fun k ω => ipwViolationRate (evalSamplesFold split k (samples ω)))
      (mean := mean)
      (r := fun k ω =>
        empiricalBernsteinRadius (toWeightedSamples (evalSamplesFold split k (samples ω)))
          (δ k) 1)
      (δ := fun k => ENNReal.ofReal (δ k))
      (hδ := hδ_fold)
  simpa [kFoldViolationEmpiricalBernsteinRadius] using h

/-- K-fold empirical Bernstein bound for IPW preference losses. -/
theorem kFoldIPWPreferenceLoss_empirical_bernstein
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.KFoldSplit Strings) (hK : 0 < split.K)
    (samples : Ω → List (TreePreferenceSample Strings Node A k))
    (mean : Fin split.K → ℝ) (range : ℝ)
    (δ : Fin split.K → ℝ)
    (hδ_fold :
      ∀ j,
        μ {ω |
            |ipwPreferenceLoss
                (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)) - mean j| ≥
              empiricalBernsteinRadius
                (toWeightedPrefSamples
                  (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)))
                (δ j) range} ≤
          ENNReal.ofReal (δ j)) :
    μ {ω |
        |(∑ j,
            (ipwPreferenceLoss
              (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)) - mean j)) /
          (split.K : ℝ)| ≥
          kFoldPreferenceEmpiricalBernsteinRadius split (samples ω) δ range} ≤
      ∑' j, ENNReal.ofReal (δ j) := by
  have h :=
    DSL.kfold_avg_bound (μ := μ) (hK := hK)
      (eval := fun j ω =>
        ipwPreferenceLoss (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)))
      (mean := mean)
      (r := fun j ω =>
        empiricalBernsteinRadius
          (toWeightedPrefSamples
            (DSL.filterEvalFold split j (fun s => s.doc) (samples ω)))
          (δ j) range)
      (δ := fun j => ENNReal.ofReal (δ j))
      (hδ := hδ_fold)
  simpa [kFoldPreferenceEmpiricalBernsteinRadius] using h

/-- K-fold honest IPW union bound (average over folds). -/
def kFoldIPWUnionBound
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (N M R : ℕ) : ℝ :=
  (∑ k, ipwUnionBound (evalSamplesFold split k samples) N M R) /
    (split.K : ℝ)

/-- K-fold empirical Bernstein radius for TreePO union bound. -/
def kFoldUnionBoundEmpiricalBernsteinRadius
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (N M R : ℕ)
    (δ_leaf δ_merge δ_idemp : Fin split.K → ℝ) : ℝ :=
  (∑ k,
      ((N : ℝ) * empiricalBernsteinRadius
          (toWeightedSamples (leafSamples (evalSamplesFold split k samples)))
          (δ_leaf k) 1
    + (M : ℝ) * empiricalBernsteinRadius
          (toWeightedSamples (mergeSamples (evalSamplesFold split k samples)))
          (δ_merge k) 1
    + ((R - 1 : ℕ) : ℝ) * empiricalBernsteinRadius
          (toWeightedSamples (resummarySamples (evalSamplesFold split k samples)))
          (δ_idemp k) 1)) /
    (split.K : ℝ)

/-- K-fold empirical Bernstein CI for TreePO union bound. -/
def kFoldUnionBoundEmpiricalBernsteinCI
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (N M R : ℕ)
    (δ_leaf δ_merge δ_idemp : Fin split.K → ℝ) : ℝ × ℝ :=
  let μhat := kFoldIPWUnionBound split samples N M R
  let r := kFoldUnionBoundEmpiricalBernsteinRadius split samples N M R
    δ_leaf δ_merge δ_idemp
  (μhat - r, μhat + r)

/-- K-fold empirical Bernstein bound package for TreePO union bound. -/
def kFoldUnionBoundEmpiricalBernsteinBound
    (split : DSL.KFoldSplit String)
    (samples : List TreeSample)
    (N M R : ℕ)
    (δ_leaf δ_merge δ_idemp : Fin split.K → ℝ) : EBBound :=
  { center := kFoldIPWUnionBound split samples N M R
    radius := kFoldUnionBoundEmpiricalBernsteinRadius split samples N M R
      δ_leaf δ_merge δ_idemp }

/-- K-fold empirical Bernstein bound for TreePO union bound. -/
theorem kFoldIPWUnionBound_empirical_bernstein
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.KFoldSplit String) (hK : 0 < split.K)
    (samples : Ω → List TreeSample)
    (N M R : ℕ)
    (mean_leaf mean_merge mean_idemp : Fin split.K → ℝ)
    (δ_leaf δ_merge δ_idemp : Fin split.K → ℝ)
    (hδ_union :
      ∀ k,
        μ {ω |
            |ipwUnionBound (evalSamplesFold split k (samples ω)) N M R -
              ((N : ℝ) * mean_leaf k + (M : ℝ) * mean_merge k +
                ((R - 1 : ℕ) : ℝ) * mean_idemp k)| ≥
            (N : ℝ) * empiricalBernsteinRadius
                (toWeightedSamples (leafSamples (evalSamplesFold split k (samples ω))))
                (δ_leaf k) 1
          + (M : ℝ) * empiricalBernsteinRadius
                (toWeightedSamples (mergeSamples (evalSamplesFold split k (samples ω))))
                (δ_merge k) 1
          + ((R - 1 : ℕ) : ℝ) * empiricalBernsteinRadius
                (toWeightedSamples (resummarySamples (evalSamplesFold split k (samples ω))))
                (δ_idemp k) 1} ≤
          ENNReal.ofReal (δ_leaf k) + ENNReal.ofReal (δ_merge k) +
            ENNReal.ofReal (δ_idemp k)) :
    μ {ω |
        |(∑ k,
            (ipwUnionBound (evalSamplesFold split k (samples ω)) N M R -
              ((N : ℝ) * mean_leaf k + (M : ℝ) * mean_merge k +
                ((R - 1 : ℕ) : ℝ) * mean_idemp k))) / (split.K : ℝ)| ≥
          kFoldUnionBoundEmpiricalBernsteinRadius split (samples ω) N M R
            δ_leaf δ_merge δ_idemp} ≤
      ∑' k, (ENNReal.ofReal (δ_leaf k) + ENNReal.ofReal (δ_merge k) +
        ENNReal.ofReal (δ_idemp k)) := by
  have h :=
    DSL.kfold_avg_bound (μ := μ) (hK := hK)
      (eval := fun k ω =>
        ipwUnionBound (evalSamplesFold split k (samples ω)) N M R)
      (mean := fun k =>
        (N : ℝ) * mean_leaf k + (M : ℝ) * mean_merge k +
          ((R - 1 : ℕ) : ℝ) * mean_idemp k)
      (r := fun k ω =>
        (N : ℝ) * empiricalBernsteinRadius
            (toWeightedSamples (leafSamples (evalSamplesFold split k (samples ω))))
            (δ_leaf k) 1
      + (M : ℝ) * empiricalBernsteinRadius
            (toWeightedSamples (mergeSamples (evalSamplesFold split k (samples ω))))
            (δ_merge k) 1
      + ((R - 1 : ℕ) : ℝ) * empiricalBernsteinRadius
            (toWeightedSamples (resummarySamples (evalSamplesFold split k (samples ω))))
            (δ_idemp k) 1)
      (δ := fun k =>
        ENNReal.ofReal (δ_leaf k) + ENNReal.ofReal (δ_merge k) +
          ENNReal.ofReal (δ_idemp k))
      (hδ := hδ_union)
  simpa [kFoldUnionBoundEmpiricalBernsteinRadius] using h

/-- Component-wise empirical Bernstein bounds imply a TreePO union-bound EB guarantee. -/
theorem ipwUnionBound_empirical_bernstein_from_components
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : Ω → List TreeSample)
    (N M R : ℕ)
    (mean_leaf mean_merge mean_idemp : ℝ)
    (δ_leaf δ_merge δ_idemp : ℝ)
    (h_leaf_event :
      μ {ω | |ipwLeafViolationRate (samples ω) - mean_leaf| ≥
        empiricalBernsteinRadius (toWeightedSamples (leafSamples (samples ω))) δ_leaf 1}
        ≤ ENNReal.ofReal δ_leaf)
    (h_merge_event :
      μ {ω | |ipwMergeViolationRate (samples ω) - mean_merge| ≥
        empiricalBernsteinRadius (toWeightedSamples (mergeSamples (samples ω))) δ_merge 1}
        ≤ ENNReal.ofReal δ_merge)
    (h_idemp_event :
      μ {ω | |ipwIdempViolationRate (samples ω) - mean_idemp| ≥
        empiricalBernsteinRadius (toWeightedSamples (resummarySamples (samples ω))) δ_idemp 1}
        ≤ ENNReal.ofReal δ_idemp)
    (hR_one_le : 1 ≤ R)
    (hcoeff : 0 < N ∨ 0 < M ∨ 1 < R) :
    μ {ω |
        |ipwUnionBound (samples ω) N M R -
          ((N : ℝ) * mean_leaf + (M : ℝ) * mean_merge + ((R - 1 : ℕ) : ℝ) * mean_idemp)| ≥
          (N : ℝ) * empiricalBernsteinRadius
              (toWeightedSamples (leafSamples (samples ω))) δ_leaf 1
        + (M : ℝ) * empiricalBernsteinRadius
              (toWeightedSamples (mergeSamples (samples ω))) δ_merge 1
        + ((R - 1 : ℕ) : ℝ) * empiricalBernsteinRadius
              (toWeightedSamples (resummarySamples (samples ω))) δ_idemp 1} ≤
      ENNReal.ofReal δ_leaf + ENNReal.ofReal δ_merge + ENNReal.ofReal δ_idemp := by
  let rLeaf : Ω → ℝ := fun ω =>
    empiricalBernsteinRadius (toWeightedSamples (leafSamples (samples ω))) δ_leaf 1
  let rMerge : Ω → ℝ := fun ω =>
    empiricalBernsteinRadius (toWeightedSamples (mergeSamples (samples ω))) δ_merge 1
  let rIdemp : Ω → ℝ := fun ω =>
    empiricalBernsteinRadius (toWeightedSamples (resummarySamples (samples ω))) δ_idemp 1
  let ALeaf : Set Ω := {ω | |ipwLeafViolationRate (samples ω) - mean_leaf| ≥ rLeaf ω}
  let AMerge : Set Ω := {ω | |ipwMergeViolationRate (samples ω) - mean_merge| ≥ rMerge ω}
  let AIdemp : Set Ω := {ω | |ipwIdempViolationRate (samples ω) - mean_idemp| ≥ rIdemp ω}
  let AUnion : Set Ω := ALeaf ∪ (AMerge ∪ AIdemp)
  let E : Set Ω := {ω |
    |ipwUnionBound (samples ω) N M R -
      ((N : ℝ) * mean_leaf + (M : ℝ) * mean_merge + ((R - 1 : ℕ) : ℝ) * mean_idemp)| ≥
      (N : ℝ) * rLeaf ω + (M : ℝ) * rMerge ω + ((R - 1 : ℕ) : ℝ) * rIdemp ω}
  have h_leaf :
      μ ALeaf ≤ ENNReal.ofReal δ_leaf := by
    simpa [ALeaf, rLeaf] using h_leaf_event
  have h_merge :
      μ AMerge ≤ ENNReal.ofReal δ_merge := by
    simpa [AMerge, rMerge] using h_merge_event
  have h_idemp :
      μ AIdemp ≤ ENNReal.ofReal δ_idemp := by
    simpa [AIdemp, rIdemp] using h_idemp_event
  have h_subset : E ⊆ AUnion := by
    intro ω hω
    by_contra hω_not
    have h_not_leaf : ω ∉ ALeaf := by
      intro hω'
      exact hω_not (Or.inl hω')
    have h_not_merge : ω ∉ AMerge := by
      intro hω'
      exact hω_not (Or.inr (Or.inl hω'))
    have h_not_idemp : ω ∉ AIdemp := by
      intro hω'
      exact hω_not (Or.inr (Or.inr hω'))
    have h_leaf_lt :
        |ipwLeafViolationRate (samples ω) - mean_leaf| < rLeaf ω := by
      exact lt_of_not_ge (by simpa [ALeaf] using h_not_leaf)
    have h_merge_lt :
        |ipwMergeViolationRate (samples ω) - mean_merge| < rMerge ω := by
      exact lt_of_not_ge (by simpa [AMerge] using h_not_merge)
    set dx : ℝ := ipwLeafViolationRate (samples ω) - mean_leaf
    set dy : ℝ := ipwMergeViolationRate (samples ω) - mean_merge
    set dId : ℝ := ipwIdempViolationRate (samples ω) - mean_idemp
    have h_idemp_lt : |dId| < rIdemp ω := by
      exact lt_of_not_ge (by simpa [AIdemp, dId] using h_not_idemp)
    set cR : ℝ := ((R - 1 : ℕ) : ℝ)
    set dz : ℝ := cR * dId
    have hω' :
        |(N : ℝ) * dx + (M : ℝ) * dy + dz| ≥
          (N : ℝ) * rLeaf ω + (M : ℝ) * rMerge ω + cR * rIdemp ω := by
      have hRcast : cR = (R : ℝ) - 1 := by
        dsimp [cR]
        simpa using (Nat.cast_sub (R := ℝ) hR_one_le)
      have h_expand :
          ipwUnionBound (samples ω) N M R -
              ((N : ℝ) * mean_leaf + (M : ℝ) * mean_merge + cR * mean_idemp) =
            (N : ℝ) * dx + (M : ℝ) * dy + dz := by
        simp [ipwUnionBound, dx, dy, dId, dz]
        rw [hRcast]
        simp [sub_eq_add_neg, mul_add, add_mul, add_assoc, add_left_comm, add_comm, mul_assoc]
      simpa [E, rLeaf, rMerge, rIdemp, cR, h_expand] using hω
    have hN_nonneg : (0 : ℝ) ≤ (N : ℝ) := by exact_mod_cast Nat.zero_le N
    have hM_nonneg : (0 : ℝ) ≤ (M : ℝ) := by exact_mod_cast Nat.zero_le M
    have hC_nonneg : (0 : ℝ) ≤ cR := by
      have hC_nonneg' : (0 : ℝ) ≤ ((R - 1 : ℕ) : ℝ) := by
        exact_mod_cast Nat.zero_le (R - 1)
      simpa [cR] using hC_nonneg'
    have htri :
        |(N : ℝ) * dx + (M : ℝ) * dy + dz| ≤
          (N : ℝ) * |dx| + (M : ℝ) * |dy| + |dz| := by
      have h1 :
          |(N : ℝ) * dx + (M : ℝ) * dy + dz| ≤
            |(N : ℝ) * dx| + |(M : ℝ) * dy| + |dz| := by
        calc
          |(N : ℝ) * dx + (M : ℝ) * dy + dz|
              = |((N : ℝ) * dx + (M : ℝ) * dy) + dz| := by ring
          _ ≤ |(N : ℝ) * dx + (M : ℝ) * dy| + |dz| := abs_add_le _ _
          _ ≤ (|(N : ℝ) * dx| + |(M : ℝ) * dy|) + |dz| := by
                gcongr
                exact abs_add_le _ _
          _ = |(N : ℝ) * dx| + |(M : ℝ) * dy| + |dz| := by ring
      have hAbsN : |(N : ℝ) * dx| = (N : ℝ) * |dx| := by
        simpa [abs_of_nonneg hN_nonneg] using (abs_mul (N : ℝ) dx)
      have hAbsM : |(M : ℝ) * dy| = (M : ℝ) * |dy| := by
        simpa [abs_of_nonneg hM_nonneg] using (abs_mul (M : ℝ) dy)
      simpa [hAbsN, hAbsM] using h1
    have hMerge_le : (M : ℝ) * |dy| ≤ (M : ℝ) * rMerge ω := by
      exact mul_le_mul_of_nonneg_left (le_of_lt h_merge_lt) hM_nonneg
    have hAbsDz : |dz| = cR * |dId| := by
      simpa [dz, abs_of_nonneg hC_nonneg, mul_assoc, mul_comm, mul_left_comm] using
        (abs_mul cR dId)
    have hIdemp_le : |dz| ≤ cR * rIdemp ω := by
      calc
        |dz| = cR * |dId| := hAbsDz
        _ ≤ cR * rIdemp ω := by
              exact mul_le_mul_of_nonneg_left (le_of_lt h_idemp_lt) hC_nonneg
    have h_sum_lt :
        (N : ℝ) * |dx| + (M : ℝ) * |dy| + |dz| <
          (N : ℝ) * rLeaf ω + (M : ℝ) * rMerge ω + cR * rIdemp ω := by
      rcases hcoeff with hN_pos | hM_pos | hR_pos
      · have hN_pos' : (0 : ℝ) < (N : ℝ) := by exact_mod_cast hN_pos
        have hLeaf_lt_scaled : (N : ℝ) * |dx| < (N : ℝ) * rLeaf ω := by
          exact mul_lt_mul_of_pos_left h_leaf_lt hN_pos'
        nlinarith [hLeaf_lt_scaled, hMerge_le, hIdemp_le]
      · have hM_pos' : (0 : ℝ) < (M : ℝ) := by exact_mod_cast hM_pos
        have hLeaf_le : (N : ℝ) * |dx| ≤ (N : ℝ) * rLeaf ω := by
          exact mul_le_mul_of_nonneg_left (le_of_lt h_leaf_lt) hN_nonneg
        have hMerge_lt_scaled : (M : ℝ) * |dy| < (M : ℝ) * rMerge ω := by
          exact mul_lt_mul_of_pos_left h_merge_lt hM_pos'
        nlinarith [hLeaf_le, hMerge_lt_scaled, hIdemp_le]
      · have hR_pos_nat : 0 < (R - 1 : ℕ) := Nat.sub_pos_of_lt hR_pos
        have hC_pos : (0 : ℝ) < cR := by
          have hC_pos' : (0 : ℝ) < ((R - 1 : ℕ) : ℝ) := by
            exact_mod_cast hR_pos_nat
          simpa [cR] using hC_pos'
        have hLeaf_le : (N : ℝ) * |dx| ≤ (N : ℝ) * rLeaf ω := by
          exact mul_le_mul_of_nonneg_left (le_of_lt h_leaf_lt) hN_nonneg
        have hIdemp_lt_scaled : |dz| < cR * rIdemp ω := by
          calc
            |dz| = cR * |dId| := hAbsDz
            _ < cR * rIdemp ω := by
                  exact mul_lt_mul_of_pos_left h_idemp_lt hC_pos
        nlinarith [hLeaf_le, hMerge_le, hIdemp_lt_scaled]
    have h_final_lt :
        |(N : ℝ) * dx + (M : ℝ) * dy + dz| <
          (N : ℝ) * rLeaf ω + (M : ℝ) * rMerge ω + cR * rIdemp ω :=
      lt_of_le_of_lt htri h_sum_lt
    exact (not_lt_of_ge hω') h_final_lt
  have hE_le_union :
      μ E ≤ μ AUnion := by
    exact measure_mono h_subset
  have h_union :
      μ AUnion ≤ μ ALeaf + μ AMerge + μ AIdemp := by
    calc
      μ AUnion
          ≤ μ ALeaf + μ (AMerge ∪ AIdemp) := by
            simpa [AUnion] using (measure_union_le (μ := μ) ALeaf (AMerge ∪ AIdemp))
      _ ≤ μ ALeaf + (μ AMerge + μ AIdemp) := by
            simpa [add_assoc, add_comm, add_left_comm] using
              (add_le_add_left (measure_union_le (μ := μ) AMerge AIdemp) (μ ALeaf))
      _ = μ ALeaf + μ AMerge + μ AIdemp := by simp [add_assoc]
  have h_comp :
      μ ALeaf + μ AMerge + μ AIdemp ≤
        ENNReal.ofReal δ_leaf + ENNReal.ofReal δ_merge + ENNReal.ofReal δ_idemp := by
    exact add_le_add (add_le_add h_leaf h_merge) h_idemp
  calc
    μ {ω |
        |ipwUnionBound (samples ω) N M R -
          ((N : ℝ) * mean_leaf + (M : ℝ) * mean_merge + ((R - 1 : ℕ) : ℝ) * mean_idemp)| ≥
          (N : ℝ) * empiricalBernsteinRadius
              (toWeightedSamples (leafSamples (samples ω))) δ_leaf 1
        + (M : ℝ) * empiricalBernsteinRadius
              (toWeightedSamples (mergeSamples (samples ω))) δ_merge 1
        + ((R - 1 : ℕ) : ℝ) * empiricalBernsteinRadius
              (toWeightedSamples (resummarySamples (samples ω))) δ_idemp 1}
        = μ E := by
          simp [E, rLeaf, rMerge, rIdemp]
    _ ≤ μ AUnion := hE_le_union
    _ ≤ μ ALeaf + μ AMerge + μ AIdemp := h_union
    _ ≤ ENNReal.ofReal δ_leaf + ENNReal.ofReal δ_merge + ENNReal.ofReal δ_idemp := h_comp

/-- K-fold TreePO union-bound EB guarantee from per-fold component event bounds. -/
theorem kFoldIPWUnionBound_empirical_bernstein_from_components
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : DSL.KFoldSplit String) (hK : 0 < split.K)
    (samples : Ω → List TreeSample)
    (N M R : ℕ)
    (mean_leaf mean_merge mean_idemp : Fin split.K → ℝ)
    (δ_leaf δ_merge δ_idemp : Fin split.K → ℝ)
    (h_leaf_fold_event :
      ∀ k,
        μ {ω | |ipwLeafViolationRate (evalSamplesFold split k (samples ω)) - mean_leaf k| ≥
          empiricalBernsteinRadius
            (toWeightedSamples (leafSamples (evalSamplesFold split k (samples ω))))
            (δ_leaf k) 1}
          ≤ ENNReal.ofReal (δ_leaf k))
    (h_merge_fold_event :
      ∀ k,
        μ {ω | |ipwMergeViolationRate (evalSamplesFold split k (samples ω)) - mean_merge k| ≥
          empiricalBernsteinRadius
            (toWeightedSamples (mergeSamples (evalSamplesFold split k (samples ω))))
            (δ_merge k) 1}
          ≤ ENNReal.ofReal (δ_merge k))
    (h_idemp_fold_event :
      ∀ k,
        μ {ω | |ipwIdempViolationRate (evalSamplesFold split k (samples ω)) - mean_idemp k| ≥
          empiricalBernsteinRadius
            (toWeightedSamples (resummarySamples (evalSamplesFold split k (samples ω))))
            (δ_idemp k) 1}
          ≤ ENNReal.ofReal (δ_idemp k))
    (hR_one_le : 1 ≤ R)
    (hcoeff : 0 < N ∨ 0 < M ∨ 1 < R) :
    μ {ω |
        |(∑ k,
            (ipwUnionBound (evalSamplesFold split k (samples ω)) N M R -
              ((N : ℝ) * mean_leaf k + (M : ℝ) * mean_merge k +
                ((R - 1 : ℕ) : ℝ) * mean_idemp k))) / (split.K : ℝ)| ≥
          kFoldUnionBoundEmpiricalBernsteinRadius split (samples ω) N M R
            δ_leaf δ_merge δ_idemp} ≤
      ∑' k, (ENNReal.ofReal (δ_leaf k) + ENNReal.ofReal (δ_merge k) +
        ENNReal.ofReal (δ_idemp k)) := by
  have hδ_union :
      ∀ k,
        μ {ω |
            |ipwUnionBound (evalSamplesFold split k (samples ω)) N M R -
              ((N : ℝ) * mean_leaf k + (M : ℝ) * mean_merge k +
                ((R - 1 : ℕ) : ℝ) * mean_idemp k)| ≥
            (N : ℝ) * empiricalBernsteinRadius
                (toWeightedSamples (leafSamples (evalSamplesFold split k (samples ω))))
                (δ_leaf k) 1
          + (M : ℝ) * empiricalBernsteinRadius
                (toWeightedSamples (mergeSamples (evalSamplesFold split k (samples ω))))
                (δ_merge k) 1
          + ((R - 1 : ℕ) : ℝ) * empiricalBernsteinRadius
                (toWeightedSamples (resummarySamples (evalSamplesFold split k (samples ω))))
                (δ_idemp k) 1} ≤
          ENNReal.ofReal (δ_leaf k) + ENNReal.ofReal (δ_merge k) +
            ENNReal.ofReal (δ_idemp k) := by
    intro k
    have h :=
      ipwUnionBound_empirical_bernstein_from_components (μ := μ)
        (samples := fun ω => evalSamplesFold split k (samples ω))
        (N := N) (M := M) (R := R)
        (mean_leaf := mean_leaf k) (mean_merge := mean_merge k) (mean_idemp := mean_idemp k)
        (δ_leaf := δ_leaf k) (δ_merge := δ_merge k) (δ_idemp := δ_idemp k)
        (h_leaf_event := h_leaf_fold_event k)
        (h_merge_event := h_merge_fold_event k)
        (h_idemp_event := h_idemp_fold_event k)
        (hR_one_le := hR_one_le) (hcoeff := hcoeff)
    simpa using h
  exact
    kFoldIPWUnionBound_empirical_bernstein (μ := μ)
      (split := split) (hK := hK) (samples := samples)
      (N := N) (M := M) (R := R)
      (mean_leaf := mean_leaf) (mean_merge := mean_merge) (mean_idemp := mean_idemp)
      (δ_leaf := δ_leaf) (δ_merge := δ_merge) (δ_idemp := δ_idemp)
      (hδ_union := hδ_union)

/-- Linearity of expectation for a weighted union bound under Bernoulli HT estimators. -/
theorem ipw_union_bound_connection
    {ι : Type*} [Fintype ι] [DecidableEq ι]
    (p : PMF ι) (pi : ι → ℝ)
    (hpi_pos : ∀ i, 0 < pi i)
    (hpi_le : ∀ i, pi i ≤ 1)
    (f_leaf f_merge f_idemp : ι → ℝ) (N M R : ℕ) :
    (N : ℝ) * (∫ ω, htExpEstimator p pi f_leaf ω ∂bernoulliProductMeasure pi hpi_pos hpi_le) +
    (M : ℝ) * (∫ ω, htExpEstimator p pi f_merge ω ∂bernoulliProductMeasure pi hpi_pos hpi_le) +
    ((R - 1 : ℕ) : ℝ) *
      (∫ ω, htExpEstimator p pi f_idemp ω ∂bernoulliProductMeasure pi hpi_pos hpi_le)
      =
    (N : ℝ) * Exp p f_leaf +
    (M : ℝ) * Exp p f_merge +
    ((R - 1 : ℕ) : ℝ) * Exp p f_idemp := by
  classical
  have h_leaf :=
    htExp_unbiased (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f_leaf
  have h_merge :=
    htExp_unbiased (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f_merge
  have h_idemp :=
    htExp_unbiased (p := p) (pi := pi) (hpi_pos := hpi_pos) (hpi_le := hpi_le) f_idemp
  simp [h_leaf, h_merge, h_idemp]

/-!
## Section 5: Clustered Standard Errors for Tree Samples
-/

/-- Group tree samples by document (cluster) -/
def groupTreeSamplesByDoc (samples : List TreeSample) : List (Cluster ℝ) :=
  let doc_ids := (samples.map TreeSample.doc_id).eraseDups
  doc_ids.map (fun doc_id =>
    let doc_samples := samples.filter (fun s => s.doc_id == doc_id)
    ⟨doc_id, toWeightedSamples doc_samples⟩)

/-- Clustered standard error for IPW violation rate estimate -/
def ipwViolationSE (samples : List TreeSample) : ℝ :=
  let clusters := groupTreeSamplesByDoc samples
  let mu_hat := ipwViolationRate samples
  clusteredSE clusters mu_hat

lemma ipwViolationSE_nonneg (samples : List TreeSample) :
    0 ≤ ipwViolationSE samples := by
  unfold ipwViolationSE
  exact clusteredSE_nonneg _ _

/-- Standard error for the union bound estimate -/
def ipwUnionBoundSE (samples : List TreeSample) (N M R : ℕ) : ℝ :=
  -- Simplified: use delta method approximation
  -- Full version would propagate uncertainties through the linear combination
  let se_leaf := ipwViolationSE (leafSamples samples)
  let se_merge := ipwViolationSE (mergeSamples samples)
  let se_idemp := ipwViolationSE (resummarySamples samples)
  Real.sqrt ((N : ℝ)^2 * se_leaf^2 + (M : ℝ)^2 * se_merge^2 + ((R - 1 : ℕ) : ℝ)^2 * se_idemp^2)

lemma ipwUnionBoundSE_nonneg (samples : List TreeSample) (N M R : ℕ) :
    0 ≤ ipwUnionBoundSE samples N M R := by
  unfold ipwUnionBoundSE
  exact Real.sqrt_nonneg _

/-- Confidence interval around the IPW union-bound estimate. -/
def ipwUnionBoundConfidenceInterval
    (samples : List TreeSample) (N M R : ℕ) (z : ℝ := 1.96) : ℝ × ℝ :=
  confidenceInterval (ipwUnionBound samples N M R) (ipwUnionBoundSE samples N M R) z

/-!
## Section 5.5: Calibration Integration
-/

/-- Combine a TreePO gap bound with a calibration error bound. -/
theorem treepo_gap_with_calibration
    (gap_oracle gap_judge tree_bound cal_err : ℝ)
    (h_tree : |gap_judge| ≤ tree_bound)
    (h_cal : |gap_oracle - gap_judge| ≤ cal_err) :
    |gap_oracle| ≤ tree_bound + cal_err := by
  have htriangle :
      |gap_oracle| ≤ |gap_judge| + |gap_oracle - gap_judge| := by
    have h : gap_judge + (gap_oracle - gap_judge) = gap_oracle := by ring
    calc
      |gap_oracle| = |gap_judge + (gap_oracle - gap_judge)| := by
        rw [h]
      _ ≤ |gap_judge| + |gap_oracle - gap_judge| := by
            exact abs_add_le _ _
  calc
    |gap_oracle| ≤ |gap_judge| + |gap_oracle - gap_judge| := htriangle
    _ ≤ tree_bound + cal_err := by
          exact add_le_add h_tree h_cal

/-- Combine TreePO gap bound with calibration error from RMSE-style assumption. -/
theorem treepo_gap_with_calibration_rmse
    (gap_oracle gap_judge tree_bound : ℝ)
    (cal : CalibrationSet) (z : ℝ := 1.96)
    (h_tree : |gap_judge| ≤ tree_bound)
    (h_rmse : |gap_oracle - gap_judge| ≤ 2 * judgeRMSE cal)
    (h_z : 0 ≤ z) :
    |gap_oracle| ≤ tree_bound + judgeCalibrationErrorBound cal z := by
  have h_cal :
      |gap_oracle - gap_judge| ≤ 2 * (absbiasUpperBound cal z + judgeStd cal) :=
    surrogate_bound_from_rmse (cal := cal) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) (z := z) h_rmse h_z
  have h :=
    treepo_gap_with_calibration (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (tree_bound := tree_bound) (cal_err := 2 * (absbiasUpperBound cal z + judgeStd cal))
      h_tree h_cal
  simpa [judgeCalibrationErrorBound] using h

/-- Combine TreePO gap bound with calibration error derived from a Lipschitz PMF model. -/
theorem treepo_gap_with_calibration_lipschitz_pmf
    {Ω : Type*} [Fintype Ω]
    (p : PMF Ω) (oracle judge : Ω → ℝ) (G : ℝ → ℝ)
    (cal : CalibrationSet) (z : ℝ := 1.96)
    (hL : GapLipschitz G (2 : ℝ≥0))
    (h_rmse_upper :
      Real.sqrt (∑' ω, (p ω).toReal * (oracle ω - judge ω)^2) ≤
        absbiasUpperBound cal z + judgeStd cal)
    (tree_bound : ℝ)
    (h_tree : |gapJudge p G judge| ≤ tree_bound) :
    |gapOracle p G oracle| ≤ tree_bound + judgeCalibrationErrorBound cal z := by
  have h_cal :
      |gapOracle p G oracle - gapJudge p G judge| ≤ judgeCalibrationErrorBound cal z :=
    surrogate_bound_pmf_calibration2 (p := p) (oracle := oracle) (judge := judge)
      (G := G) (cal := cal) (z := z) hL h_rmse_upper
  exact treepo_gap_with_calibration
    (gap_oracle := gapOracle p G oracle)
    (gap_judge := gapJudge p G judge)
    (tree_bound := tree_bound)
    (cal_err := judgeCalibrationErrorBound cal z)
    h_tree h_cal

/-- Compatibility wrapper: derive Lipschitz PMF calibration bridge from `CalibrationAxioms`. -/
theorem treepo_gap_with_calibration_lipschitz_pmf_from_axioms
    {Ω : Type*} [Fintype Ω]
    (p : PMF Ω) (oracle judge : Ω → ℝ) (G : ℝ → ℝ)
    (cal : CalibrationSet) (z : ℝ := 1.96)
    (hL : GapLipschitz G (2 : ℝ≥0))
    (cal_axioms : CalibrationAxioms p oracle judge cal z)
    (tree_bound : ℝ)
    (h_tree : |gapJudge p G judge| ≤ tree_bound) :
    |gapOracle p G oracle| ≤ tree_bound + judgeCalibrationErrorBound cal z := by
  exact treepo_gap_with_calibration_lipschitz_pmf
    (p := p) (oracle := oracle) (judge := judge) (G := G)
    (cal := cal) (z := z) (hL := hL)
    (h_rmse_upper := cal_axioms)
    (tree_bound := tree_bound) h_tree

/-- Calibration + sampling-estimation worst-case envelope. -/
theorem treepo_gap_with_calibration_and_estimation
    (gap_oracle gap_judge gap_est cal_err est_err : ℝ)
    (h_cal : |gap_oracle - gap_judge| ≤ cal_err)
    (h_est : |gap_judge - gap_est| ≤ est_err) :
    |gap_oracle| ≤ |gap_est| + cal_err + est_err := by
  let d_oj : ℝ := gap_oracle - gap_judge
  let d_je : ℝ := gap_judge - gap_est
  have h_oracle_judge :
      |gap_oracle| ≤ |gap_judge| + |d_oj| := by
    have h_decomp : gap_oracle = gap_judge + d_oj := by
      simp [d_oj]
    calc
      |gap_oracle| = |gap_judge + d_oj| := by rw [h_decomp]
      _ ≤ |gap_judge| + |d_oj| := abs_add_le _ _
  have h_judge_est :
      |gap_judge| ≤ |gap_est| + |d_je| := by
    have h_decomp : gap_judge = gap_est + d_je := by
      simp [d_je]
    calc
      |gap_judge| = |gap_est + d_je| := by rw [h_decomp]
      _ ≤ |gap_est| + |d_je| := abs_add_le _ _
  have h_triangle :
      |gap_oracle| ≤ |gap_est| + |d_je| + |d_oj| := by
    calc
      |gap_oracle| ≤ |gap_judge| + |d_oj| := h_oracle_judge
      _ ≤ (|gap_est| + |d_je|) + |d_oj| := by
            exact add_le_add h_judge_est (le_refl _)
      _ = |gap_est| + |d_je| + |d_oj| := by ring
  calc
    |gap_oracle| ≤ |gap_est| + |d_je| + |d_oj| := h_triangle
    _ = |gap_est| + |gap_judge - gap_est| + |gap_oracle - gap_judge| := by
          simp [d_je, d_oj]
    _ ≤ |gap_est| + est_err + cal_err := by linarith
    _ = |gap_est| + cal_err + est_err := by ring

/-- Oracle-measurement + calibration + sampling-estimation worst-case envelope.

This is the general four-layer decomposition

`true target -> oracle target -> judge target -> estimated target`.

The usual exact-oracle regime is the special case `oracle_err = 0`. -/
theorem treepo_gap_with_oracleMeasurement_calibration_and_estimation
    (gap_true gap_oracle gap_judge gap_est oracle_err cal_err est_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_cal : |gap_oracle - gap_judge| ≤ cal_err)
    (h_est : |gap_judge - gap_est| ≤ est_err) :
    |gap_true| ≤ |gap_est| + oracle_err + cal_err + est_err := by
  have h_tail :
      |gap_oracle| ≤ |gap_est| + cal_err + est_err :=
    treepo_gap_with_calibration_and_estimation
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (gap_est := gap_est) (cal_err := cal_err) (est_err := est_err)
      h_cal h_est
  let d_to : ℝ := gap_true - gap_oracle
  have h_bridge :
      |gap_true| ≤ |gap_oracle| + oracle_err := by
    have h_decomp : gap_true = gap_oracle + d_to := by
      simp [d_to]
    calc
      |gap_true| = |gap_oracle + d_to| := by rw [h_decomp]
      _ ≤ |gap_oracle| + |d_to| := abs_add_le _ _
      _ = |gap_oracle| + |gap_true - gap_oracle| := by simp [d_to]
      _ ≤ |gap_oracle| + oracle_err := by
            exact add_le_add (le_refl _) h_oracle
  calc
    |gap_true| ≤ |gap_oracle| + oracle_err := h_bridge
    _ ≤ (|gap_est| + cal_err + est_err) + oracle_err := by
          exact add_le_add h_tail (le_refl _)
    _ = |gap_est| + oracle_err + cal_err + est_err := by ring

/-- Exact-oracle convenience corollary of the general four-layer envelope. -/
theorem treepo_gap_with_exactOracle_calibration_and_estimation
    (gap_true gap_oracle gap_judge gap_est cal_err est_err : ℝ)
    (h_oracle_exact : gap_true = gap_oracle)
    (h_cal : |gap_oracle - gap_judge| ≤ cal_err)
    (h_est : |gap_judge - gap_est| ≤ est_err) :
    |gap_true| ≤ |gap_est| + cal_err + est_err := by
  have h_oracle : |gap_true - gap_oracle| ≤ 0 := by
    rw [h_oracle_exact]
    simp
  have h :=
    treepo_gap_with_oracleMeasurement_calibration_and_estimation
      (gap_true := gap_true) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) (gap_est := gap_est)
      (oracle_err := 0) (cal_err := cal_err) (est_err := est_err)
      h_oracle h_cal h_est
  simpa using h

/-- Calibration + sampling + clipping worst-case envelope. -/
theorem treepo_gap_with_calibration_estimation_clipping
    (gap_oracle gap_judge gap_est gap_clip cal_err est_err clip_err : ℝ)
    (h_cal : |gap_oracle - gap_judge| ≤ cal_err)
    (h_est : |gap_judge - gap_est| ≤ est_err)
    (h_clip : |gap_est - gap_clip| ≤ clip_err) :
    |gap_oracle| ≤ |gap_clip| + cal_err + est_err + clip_err := by
  have h_from_est :
      |gap_oracle| ≤ |gap_est| + cal_err + est_err :=
    treepo_gap_with_calibration_and_estimation
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (gap_est := gap_est) (cal_err := cal_err) (est_err := est_err)
      h_cal h_est
  have h_est_from_clip : |gap_est| ≤ |gap_clip| + clip_err := by
    let d_ec : ℝ := gap_est - gap_clip
    have h_triangle : |gap_est| ≤ |gap_clip| + |gap_est - gap_clip| := by
      have h_decomp : gap_est = gap_clip + d_ec := by
        simp [d_ec]
      calc
        |gap_est| = |gap_clip + d_ec| := by rw [h_decomp]
        _ ≤ |gap_clip| + |d_ec| := by exact abs_add_le _ _
        _ = |gap_clip| + |gap_est - gap_clip| := by simp [d_ec]
    linarith
  have h_step :
      |gap_est| + cal_err + est_err ≤ (|gap_clip| + clip_err) + cal_err + est_err := by
    linarith
  calc
    |gap_oracle| ≤ |gap_est| + cal_err + est_err := h_from_est
    _ ≤ (|gap_clip| + clip_err) + cal_err + est_err := h_step
    _ = |gap_clip| + cal_err + est_err + clip_err := by ring

/-- Oracle-measurement + calibration + sampling + clipping worst-case envelope. -/
theorem treepo_gap_with_oracleMeasurement_calibration_estimation_clipping
    (gap_true gap_oracle gap_judge gap_est gap_clip oracle_err cal_err est_err clip_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_cal : |gap_oracle - gap_judge| ≤ cal_err)
    (h_est : |gap_judge - gap_est| ≤ est_err)
    (h_clip : |gap_est - gap_clip| ≤ clip_err) :
    |gap_true| ≤ |gap_clip| + oracle_err + cal_err + est_err + clip_err := by
  have h_tail :
      |gap_oracle| ≤ |gap_clip| + cal_err + est_err + clip_err :=
    treepo_gap_with_calibration_estimation_clipping
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (gap_est := gap_est) (gap_clip := gap_clip)
      (cal_err := cal_err) (est_err := est_err) (clip_err := clip_err)
      h_cal h_est h_clip
  let d_to : ℝ := gap_true - gap_oracle
  have h_bridge :
      |gap_true| ≤ |gap_oracle| + oracle_err := by
    have h_decomp : gap_true = gap_oracle + d_to := by
      simp [d_to]
    calc
      |gap_true| = |gap_oracle + d_to| := by rw [h_decomp]
      _ ≤ |gap_oracle| + |d_to| := abs_add_le _ _
      _ = |gap_oracle| + |gap_true - gap_oracle| := by simp [d_to]
      _ ≤ |gap_oracle| + oracle_err := by
            exact add_le_add (le_refl _) h_oracle
  calc
    |gap_true| ≤ |gap_oracle| + oracle_err := h_bridge
    _ ≤ (|gap_clip| + cal_err + est_err + clip_err) + oracle_err := by
          exact add_le_add h_tail (le_refl _)
    _ = |gap_clip| + oracle_err + cal_err + est_err + clip_err := by ring

/-!
## Section 6: Master DSL Bound
-/

/-- The full DSL bound for tree-based preference learning.

This is the master theorem combining:
1. IPW estimate of union bound
2. Clustered standard error for uncertainty
3. Judge calibration error (if using judge instead of oracle)

The bound states that with high probability (e.g., 95%):
  |gap_oracle - gap_estimated| ≤ margin

where margin includes:
- SE margin for sampling uncertainty
- Bias margin for judge calibration (if applicable) -/
structure DSLBound where
  gap_estimate : ℝ              -- IPW estimate of gap (union bound)
  se : ℝ                        -- Clustered standard error
  bias_margin : ℝ               -- Judge calibration error margin (0 if oracle-labeled)
  confidence_level : ℝ          -- e.g., 0.95
  z_score : ℝ                   -- e.g., 1.96 for 95%

namespace DSLBound

/-- Total margin: z × SE + bias -/
def totalMargin (b : DSLBound) : ℝ :=
  b.z_score * b.se + b.bias_margin

/-- Upper bound on true gap -/
def upperBound (b : DSLBound) : ℝ :=
  b.gap_estimate + b.totalMargin

/-- Lower bound on true gap -/
def lowerBound (b : DSLBound) : ℝ :=
  b.gap_estimate - b.totalMargin

/-- Confidence interval -/
def confidenceInterval (b : DSLBound) : ℝ × ℝ :=
  (b.lowerBound, b.upperBound)

end DSLBound

/-- Compute DSL bound from samples.

This is the main entry point for DSL-based evaluation. -/
def computeDSLBound (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ := 1.96) : DSLBound :=
  let gap_est := ipwUnionBound samples N M R
  let se := ipwUnionBoundSE samples N M R
  let bias_margin := match cal with
    | some c => judgeCalibrationErrorBound c z
    | none => 0  -- Oracle-labeled, no calibration needed
  { gap_estimate := gap_est
    se := se
    bias_margin := bias_margin
    confidence_level := 0.95
    z_score := z }

/-- DSL upper bound implied by TreePO gap + calibration error. -/
theorem dsl_upperBound_treepo
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_oracle gap_judge : ℝ)
    (h_tree : |gap_judge| ≤ ipwUnionBound samples N M R)
    (h_cal :
      |gap_oracle - gap_judge| ≤
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0)
    (h_z : 0 ≤ z)
    (h_se : 0 ≤ ipwUnionBoundSE samples N M R) :
    |gap_oracle| ≤ (computeDSLBound samples N M R cal z).upperBound := by
  classical
  cases cal with
  | none =>
      have h_cal' : |gap_oracle - gap_judge| ≤ 0 := by simpa using h_cal
      have h_base :
          |gap_oracle| ≤ ipwUnionBound samples N M R + 0 :=
        treepo_gap_with_calibration (gap_oracle := gap_oracle) (gap_judge := gap_judge)
          (tree_bound := ipwUnionBound samples N M R) (cal_err := 0) h_tree h_cal'
      have h_nonneg : 0 ≤ z * ipwUnionBoundSE samples N M R := by
        exact mul_nonneg h_z h_se
      have h_step :
          ipwUnionBound samples N M R + 0 ≤
            ipwUnionBound samples N M R + (z * ipwUnionBoundSE samples N M R + 0) := by
        linarith [h_nonneg]
      have h_final := le_trans h_base h_step
      simpa [computeDSLBound, DSLBound.upperBound, DSLBound.totalMargin] using h_final
  | some c =>
      have h_cal' :
          |gap_oracle - gap_judge| ≤ judgeCalibrationErrorBound c z := by
        simpa using h_cal
      have h_base :
          |gap_oracle| ≤ ipwUnionBound samples N M R + judgeCalibrationErrorBound c z :=
        treepo_gap_with_calibration (gap_oracle := gap_oracle) (gap_judge := gap_judge)
          (tree_bound := ipwUnionBound samples N M R)
          (cal_err := judgeCalibrationErrorBound c z) h_tree h_cal'
      have h_nonneg : 0 ≤ z * ipwUnionBoundSE samples N M R := by
        exact mul_nonneg h_z h_se
      have h_step :
          ipwUnionBound samples N M R + judgeCalibrationErrorBound c z ≤
            ipwUnionBound samples N M R +
              (z * ipwUnionBoundSE samples N M R + judgeCalibrationErrorBound c z) := by
        linarith [h_nonneg]
      have h_final := le_trans h_base h_step
      simpa [computeDSLBound, DSLBound.upperBound, DSLBound.totalMargin] using h_final

/-- DSL upper bound implied by TreePO gap + calibration error + an additional
oracle-measurement envelope. The exact-oracle regime is the special case
`oracle_err = 0`. -/
theorem dsl_upperBound_treepo_with_oracleMeasurement
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_true gap_oracle gap_judge : ℝ)
    (oracle_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_tree : |gap_judge| ≤ ipwUnionBound samples N M R)
    (h_cal :
      |gap_oracle - gap_judge| ≤
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0)
    (h_z : 0 ≤ z)
    (h_se : 0 ≤ ipwUnionBoundSE samples N M R) :
    |gap_true| ≤ (computeDSLBound samples N M R cal z).upperBound + oracle_err := by
  have h_base :
      |gap_oracle| ≤ (computeDSLBound samples N M R cal z).upperBound :=
    dsl_upperBound_treepo
      (samples := samples) (N := N) (M := M) (R := R)
      (cal := cal) (z := z)
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      h_tree h_cal h_z h_se
  let d_to : ℝ := gap_true - gap_oracle
  have h_bridge :
      |gap_true| ≤ |gap_oracle| + oracle_err := by
    have h_decomp : gap_true = gap_oracle + d_to := by
      simp [d_to]
    calc
      |gap_true| = |gap_oracle + d_to| := by rw [h_decomp]
      _ ≤ |gap_oracle| + |d_to| := abs_add_le _ _
      _ = |gap_oracle| + |gap_true - gap_oracle| := by simp [d_to]
      _ ≤ |gap_oracle| + oracle_err := by
            exact add_le_add (le_refl _) h_oracle
  calc
    |gap_true| ≤ |gap_oracle| + oracle_err := h_bridge
    _ ≤ (computeDSLBound samples N M R cal z).upperBound + oracle_err := by
          exact add_le_add h_base (le_refl _)

/-!
## Section 6.5: Calibrated TreePO DSL Entry Point
-/

/-- DSL upper bound for TreePO using an explicit RMSE calibration envelope. -/
theorem dsl_upperBound_treepo_calibrated_pmf
    {Ω : Type*} [Fintype Ω]
    (p : PMF Ω) (oracle judge : Ω → ℝ) (G : ℝ → ℝ)
    (cal : CalibrationSet) (z : ℝ := 1.96)
    (h_rmse_upper :
      Real.sqrt (∑' ω, (p ω).toReal * (oracle ω - judge ω)^2) ≤
        absbiasUpperBound cal z + judgeStd cal)
    (hL : GapLipschitz G (2 : ℝ≥0))
    (samples : List TreeSample) (N M R : ℕ)
    (h_tree : |gapJudge p G judge| ≤ ipwUnionBound samples N M R)
    (h_z : 0 ≤ z) (h_se : 0 ≤ ipwUnionBoundSE samples N M R) :
    |gapOracle p G oracle| ≤ (computeDSLBound samples N M R (some cal) z).upperBound := by
  have h_cal :
      |gapOracle p G oracle - gapJudge p G judge| ≤ judgeCalibrationErrorBound cal z :=
    surrogate_bound_pmf_calibration2 (p := p) (oracle := oracle) (judge := judge)
      (G := G) (cal := cal) (z := z) hL h_rmse_upper
  -- Use the generic TreePO → DSL upper bound.
  simpa using
    (dsl_upperBound_treepo (samples := samples) (N := N) (M := M) (R := R)
      (cal := some cal) (z := z)
      (gap_oracle := gapOracle p G oracle) (gap_judge := gapJudge p G judge)
      (h_tree := h_tree) (h_cal := h_cal) (h_z := h_z) (h_se := h_se))

/-- Compatibility wrapper: recover calibrated PMF DSL bound from `CalibrationAxioms`. -/
theorem dsl_upperBound_treepo_calibrated_pmf_from_axioms
    {Ω : Type*} [Fintype Ω]
    (p : PMF Ω) (oracle judge : Ω → ℝ) (G : ℝ → ℝ)
    (cal : CalibrationSet) (z : ℝ := 1.96)
    (cal_axioms : CalibrationAxioms p oracle judge cal z)
    (hL : GapLipschitz G (2 : ℝ≥0))
    (samples : List TreeSample) (N M R : ℕ)
    (h_tree : |gapJudge p G judge| ≤ ipwUnionBound samples N M R)
    (h_z : 0 ≤ z) (h_se : 0 ≤ ipwUnionBoundSE samples N M R) :
    |gapOracle p G oracle| ≤ (computeDSLBound samples N M R (some cal) z).upperBound := by
  exact dsl_upperBound_treepo_calibrated_pmf
    (p := p) (oracle := oracle) (judge := judge) (G := G)
    (cal := cal) (z := z)
    (h_rmse_upper := cal_axioms)
    (hL := hL)
    (samples := samples) (N := N) (M := M) (R := R)
    h_tree h_z h_se

/-- DSL bound expressed directly around the estimated gap (`gap_estimate`). -/
theorem dsl_abs_gap_bound_from_estimate
    (b : DSLBound) (gap_oracle gap_judge : ℝ)
    (h_est : |gap_judge - b.gap_estimate| ≤ b.z_score * b.se)
    (h_cal : |gap_oracle - gap_judge| ≤ b.bias_margin) :
    |gap_oracle| ≤ |b.gap_estimate| + b.totalMargin := by
  have h :=
    treepo_gap_with_calibration_and_estimation
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (gap_est := b.gap_estimate) (cal_err := b.bias_margin)
      (est_err := b.z_score * b.se) h_cal h_est
  simpa [DSLBound.totalMargin, add_assoc, add_left_comm, add_comm] using h

/-- DSL bound expressed directly around the estimated gap, with an optional
oracle-measurement envelope on top of the usual oracle→judge→estimate chain. -/
theorem dsl_abs_gap_bound_from_estimate_with_oracleMeasurement
    (b : DSLBound) (gap_true gap_oracle gap_judge : ℝ)
    (oracle_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_est : |gap_judge - b.gap_estimate| ≤ b.z_score * b.se)
    (h_cal : |gap_oracle - gap_judge| ≤ b.bias_margin) :
    |gap_true| ≤ |b.gap_estimate| + b.totalMargin + oracle_err := by
  have h :=
    treepo_gap_with_oracleMeasurement_calibration_and_estimation
      (gap_true := gap_true) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) (gap_est := b.gap_estimate)
      (oracle_err := oracle_err)
      (cal_err := b.bias_margin) (est_err := b.z_score * b.se)
      h_oracle h_cal h_est
  simpa [DSLBound.totalMargin, add_assoc, add_left_comm, add_comm] using h

/-- DSL upper bound from estimate-space assumptions and nonnegative estimate. -/
theorem dsl_upperBound_from_estimate
    (b : DSLBound) (gap_oracle gap_judge : ℝ)
    (h_est : |gap_judge - b.gap_estimate| ≤ b.z_score * b.se)
    (h_cal : |gap_oracle - gap_judge| ≤ b.bias_margin)
    (h_est_nonneg : 0 ≤ b.gap_estimate) :
    |gap_oracle| ≤ b.upperBound := by
  have h_abs :=
    dsl_abs_gap_bound_from_estimate (b := b) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) h_est h_cal
  simpa [DSLBound.upperBound, abs_of_nonneg h_est_nonneg] using h_abs

/-- DSL upper bound from estimate-space assumptions with an additional oracle
measurement-error envelope. Exact oracle labels recover `oracle_err = 0`. -/
theorem dsl_upperBound_from_estimate_with_oracleMeasurement
    (b : DSLBound) (gap_true gap_oracle gap_judge : ℝ)
    (oracle_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_est : |gap_judge - b.gap_estimate| ≤ b.z_score * b.se)
    (h_cal : |gap_oracle - gap_judge| ≤ b.bias_margin)
    (h_est_nonneg : 0 ≤ b.gap_estimate) :
    |gap_true| ≤ b.upperBound + oracle_err := by
  have h_abs :=
    dsl_abs_gap_bound_from_estimate_with_oracleMeasurement
      (b := b) (gap_true := gap_true) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) (oracle_err := oracle_err)
      h_oracle h_est h_cal
  simpa [DSLBound.upperBound, DSLBound.totalMargin, abs_of_nonneg h_est_nonneg,
    add_assoc, add_left_comm, add_comm] using h_abs

/-- If the judge-side gap estimate lies in the clustered confidence interval
around the IPW union-bound estimate, then the computed DSL bound is pointwise
valid once the calibration envelope is supplied. -/
theorem dsl_upperBound_of_interval_membership
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_oracle gap_judge : ℝ)
    (h_est : gap_judge ∈ Set.Icc
      (ipwUnionBoundConfidenceInterval samples N M R z).1
      (ipwUnionBoundConfidenceInterval samples N M R z).2)
    (h_cal :
      |gap_oracle - gap_judge| ≤
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0)
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_z : 0 ≤ z) :
    |gap_oracle| ≤ (computeDSLBound samples N M R cal z).upperBound := by
  have h_radius : 0 ≤ z * ipwUnionBoundSE samples N M R :=
    mul_nonneg h_z (ipwUnionBoundSE_nonneg samples N M R)
  have h_est' :
      |gap_judge - ipwUnionBound samples N M R| ≤
        z * ipwUnionBoundSE samples N M R := by
    simpa [ipwUnionBoundConfidenceInterval] using
      (mem_confidenceInterval_iff_abs_sub_le
        (theta := gap_judge)
        (mu_hat := ipwUnionBound samples N M R)
        (se := ipwUnionBoundSE samples N M R)
        (z := z)
        h_radius).mp h_est
  simpa [computeDSLBound] using
    (dsl_upperBound_from_estimate
      (b := computeDSLBound samples N M R cal z)
      (gap_oracle := gap_oracle)
      (gap_judge := gap_judge)
      (h_est := h_est')
      (h_cal := h_cal)
      (h_est_nonneg := h_est_nonneg))

/-- Oracle-measurement version of `dsl_upperBound_of_interval_membership`. -/
theorem dsl_upperBound_of_interval_membership_with_oracleMeasurement
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_true gap_oracle gap_judge : ℝ)
    (oracle_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_est : gap_judge ∈ Set.Icc
      (ipwUnionBoundConfidenceInterval samples N M R z).1
      (ipwUnionBoundConfidenceInterval samples N M R z).2)
    (h_cal :
      |gap_oracle - gap_judge| ≤
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0)
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_z : 0 ≤ z) :
    |gap_true| ≤ (computeDSLBound samples N M R cal z).upperBound + oracle_err := by
  have h_radius : 0 ≤ z * ipwUnionBoundSE samples N M R :=
    mul_nonneg h_z (ipwUnionBoundSE_nonneg samples N M R)
  have h_est' :
      |gap_judge - ipwUnionBound samples N M R| ≤
        z * ipwUnionBoundSE samples N M R := by
    simpa [ipwUnionBoundConfidenceInterval] using
      (mem_confidenceInterval_iff_abs_sub_le
        (theta := gap_judge)
        (mu_hat := ipwUnionBound samples N M R)
        (se := ipwUnionBoundSE samples N M R)
        (z := z)
        h_radius).mp h_est
  simpa [computeDSLBound] using
    (dsl_upperBound_from_estimate_with_oracleMeasurement
      (b := computeDSLBound samples N M R cal z)
      (gap_true := gap_true)
      (gap_oracle := gap_oracle)
      (gap_judge := gap_judge)
      (oracle_err := oracle_err)
      (h_oracle := h_oracle)
      (h_est := h_est')
      (h_cal := h_cal)
      (h_est_nonneg := h_est_nonneg))

/-- Event-level validity of `computeDSLBound` from a joint confidence-interval
membership event and a calibration event. -/
theorem computeDSLBound_valid_from_joint_interval_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_oracle gap_judge : Ω → ℝ)
    (q : ENNReal)
    (h_event : q ≤ μ {ω |
      gap_judge ω ∈ Set.Icc
        (ipwUnionBoundConfidenceInterval samples N M R z).1
        (ipwUnionBoundConfidenceInterval samples N M R z).2 ∧
      |gap_oracle ω - gap_judge ω| ≤
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0})
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_z : 0 ≤ z) :
    q ≤ μ {ω | |gap_oracle ω| ≤ (computeDSLBound samples N M R cal z).upperBound} := by
  cases cal with
  | none =>
      simp at h_event ⊢
      have h_subset :
          {ω |
            gap_judge ω ∈ Set.Icc
              (ipwUnionBoundConfidenceInterval samples N M R z).1
              (ipwUnionBoundConfidenceInterval samples N M R z).2 ∧
            gap_oracle ω - gap_judge ω = 0} ⊆
            {ω | |gap_oracle ω| ≤ (computeDSLBound samples N M R none z).upperBound} := by
        intro ω hω
        rcases hω with ⟨h_estω, h_calω⟩
        exact dsl_upperBound_of_interval_membership
          (samples := samples) (N := N) (M := M) (R := R)
          (cal := none) (z := z)
          (gap_oracle := gap_oracle ω) (gap_judge := gap_judge ω)
          (h_est := h_estω) (h_cal := by simpa [abs_eq_zero] using h_calω)
          (h_est_nonneg := h_est_nonneg) (h_z := h_z)
      exact le_trans h_event (measure_mono h_subset)
  | some c =>
      simp at h_event ⊢
      have h_subset :
          {ω |
            gap_judge ω ∈ Set.Icc
              (ipwUnionBoundConfidenceInterval samples N M R z).1
              (ipwUnionBoundConfidenceInterval samples N M R z).2 ∧
            |gap_oracle ω - gap_judge ω| ≤ judgeCalibrationErrorBound c z} ⊆
            {ω | |gap_oracle ω| ≤ (computeDSLBound samples N M R (some c) z).upperBound} := by
        intro ω hω
        rcases hω with ⟨h_estω, h_calω⟩
        exact dsl_upperBound_of_interval_membership
          (samples := samples) (N := N) (M := M) (R := R)
          (cal := some c) (z := z)
          (gap_oracle := gap_oracle ω) (gap_judge := gap_judge ω)
          (h_est := h_estω) (h_cal := h_calω)
          (h_est_nonneg := h_est_nonneg) (h_z := h_z)
      exact le_trans h_event (measure_mono h_subset)

/-- Oracle-measurement version of
`computeDSLBound_valid_from_joint_interval_event`. -/
theorem computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (q : ENNReal)
    (h_event : q ≤ μ {ω |
      |gap_true ω - gap_oracle ω| ≤ oracle_err ∧
      gap_judge ω ∈ Set.Icc
        (ipwUnionBoundConfidenceInterval samples N M R z).1
        (ipwUnionBoundConfidenceInterval samples N M R z).2 ∧
      |gap_oracle ω - gap_judge ω| ≤
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0})
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_z : 0 ≤ z) :
    q ≤ μ {ω |
      |gap_true ω| ≤ (computeDSLBound samples N M R cal z).upperBound + oracle_err} := by
  cases cal with
  | none =>
      simp at h_event ⊢
      have h_subset :
          {ω |
            |gap_true ω - gap_oracle ω| ≤ oracle_err ∧
            gap_judge ω ∈ Set.Icc
              (ipwUnionBoundConfidenceInterval samples N M R z).1
              (ipwUnionBoundConfidenceInterval samples N M R z).2 ∧
            gap_oracle ω - gap_judge ω = 0} ⊆
            {ω |
              |gap_true ω| ≤ (computeDSLBound samples N M R none z).upperBound + oracle_err} := by
        intro ω hω
        rcases hω with ⟨h_oracleω, h_estω, h_calω⟩
        exact dsl_upperBound_of_interval_membership_with_oracleMeasurement
          (samples := samples) (N := N) (M := M) (R := R)
          (cal := none) (z := z)
          (gap_true := gap_true ω) (gap_oracle := gap_oracle ω)
          (gap_judge := gap_judge ω)
          (oracle_err := oracle_err)
          (h_oracle := h_oracleω)
          (h_est := h_estω) (h_cal := by simpa [abs_eq_zero] using h_calω)
          (h_est_nonneg := h_est_nonneg) (h_z := h_z)
      exact le_trans h_event (measure_mono h_subset)
  | some c =>
      simp at h_event ⊢
      have h_subset :
          {ω |
            |gap_true ω - gap_oracle ω| ≤ oracle_err ∧
            gap_judge ω ∈ Set.Icc
              (ipwUnionBoundConfidenceInterval samples N M R z).1
              (ipwUnionBoundConfidenceInterval samples N M R z).2 ∧
            |gap_oracle ω - gap_judge ω| ≤ judgeCalibrationErrorBound c z} ⊆
            {ω |
              |gap_true ω| ≤ (computeDSLBound samples N M R (some c) z).upperBound + oracle_err} := by
        intro ω hω
        rcases hω with ⟨h_oracleω, h_estω, h_calω⟩
        exact dsl_upperBound_of_interval_membership_with_oracleMeasurement
          (samples := samples) (N := N) (M := M) (R := R)
          (cal := some c) (z := z)
          (gap_true := gap_true ω) (gap_oracle := gap_oracle ω)
          (gap_judge := gap_judge ω)
          (oracle_err := oracle_err)
          (h_oracle := h_oracleω)
          (h_est := h_estω) (h_cal := h_calω)
          (h_est_nonneg := h_est_nonneg) (h_z := h_z)
      exact le_trans h_event (measure_mono h_subset)

/-- DSL bound around a clipped estimate with an explicit clipping envelope. -/
theorem dsl_abs_gap_bound_from_clipped_estimate
    (b : DSLBound) (gap_oracle gap_judge gap_clip clip_err : ℝ)
    (h_est : |gap_judge - b.gap_estimate| ≤ b.z_score * b.se)
    (h_cal : |gap_oracle - gap_judge| ≤ b.bias_margin)
    (h_clip : |b.gap_estimate - gap_clip| ≤ clip_err) :
    |gap_oracle| ≤ |gap_clip| + b.totalMargin + clip_err := by
  have h :=
    treepo_gap_with_calibration_estimation_clipping
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (gap_est := b.gap_estimate) (gap_clip := gap_clip)
      (cal_err := b.bias_margin) (est_err := b.z_score * b.se) (clip_err := clip_err)
      h_cal h_est h_clip
  simpa [DSLBound.totalMargin, add_assoc, add_left_comm, add_comm] using h

/-- Clipped-estimate DSL envelope with an additional oracle-measurement term. -/
theorem dsl_abs_gap_bound_from_clipped_estimate_with_oracleMeasurement
    (b : DSLBound) (gap_true gap_oracle gap_judge gap_clip clip_err oracle_err : ℝ)
    (h_oracle : |gap_true - gap_oracle| ≤ oracle_err)
    (h_est : |gap_judge - b.gap_estimate| ≤ b.z_score * b.se)
    (h_cal : |gap_oracle - gap_judge| ≤ b.bias_margin)
    (h_clip : |b.gap_estimate - gap_clip| ≤ clip_err) :
    |gap_true| ≤ |gap_clip| + b.totalMargin + oracle_err + clip_err := by
  have h :=
    treepo_gap_with_oracleMeasurement_calibration_estimation_clipping
      (gap_true := gap_true) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) (gap_est := b.gap_estimate) (gap_clip := gap_clip)
      (oracle_err := oracle_err)
      (cal_err := b.bias_margin) (est_err := b.z_score * b.se) (clip_err := clip_err)
      h_oracle h_cal h_est h_clip
  simpa [DSLBound.totalMargin, add_assoc, add_left_comm, add_comm] using h

/-- If `|x|` exceeds `|y|` by `b`, then `|x-y|` is at least `b`. -/
lemma abs_sub_ge_of_abs_ge_abs_plus (x y b : ℝ)
    (h : |x| ≥ |y| + b) :
    |x - y| ≥ b := by
  have htri : |x| ≤ |x - y| + |y| := by
    have hdecomp : (x - y) + y = x := by ring
    have htri' : |(x - y) + y| ≤ |x - y| + |y| := abs_add_le _ _
    simpa [hdecomp] using htri'
  have hsum : |y| + b ≤ |x - y| + |y| := le_trans h htri
  linarith

/-- One-shot high-probability envelope from calibration + estimation errors. -/
theorem dsl_abs_gap_bound_from_estimate_high_prob
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (gap_oracle gap_judge gap_est : Ω → ℝ)
    (b_cal b_est : ℝ)
    (δ_cal δ_est : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - gap_est ω| ≥ b_est} ≤ δ_est) :
    μ {ω | |gap_oracle ω| ≥ |gap_est ω| + b_cal + b_est} ≤
      δ_cal + δ_est := by
  let Acal : Set Ω := {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal}
  let Aest : Set Ω := {ω | |gap_judge ω - gap_est ω| ≥ b_est}
  let Atot : Set Ω := {ω | |gap_oracle ω| ≥ |gap_est ω| + b_cal + b_est}
  have h_subset : Atot ⊆ Acal ∪ Aest := by
    intro ω hω
    have h_atot : |gap_oracle ω| ≥ |gap_est ω| + (b_cal + b_est) := by
      simpa [Atot, add_assoc] using hω
    have h_diff :
        |gap_oracle ω - gap_est ω| ≥ b_cal + b_est :=
      abs_sub_ge_of_abs_ge_abs_plus (x := gap_oracle ω) (y := gap_est ω)
        (b := b_cal + b_est) h_atot
    by_cases hA : |gap_oracle ω - gap_judge ω| ≥ b_cal
    · exact Or.inl (by simpa [Acal] using hA)
    · by_cases hB : |gap_judge ω - gap_est ω| ≥ b_est
      · exact Or.inr (by simpa [Aest] using hB)
      · have hA_lt : |gap_oracle ω - gap_judge ω| < b_cal := lt_of_not_ge hA
        have hB_lt : |gap_judge ω - gap_est ω| < b_est := lt_of_not_ge hB
        have htri :
            |gap_oracle ω - gap_est ω| ≤
              |gap_oracle ω - gap_judge ω| + |gap_judge ω - gap_est ω| := by
          have hdecomp :
              gap_oracle ω - gap_est ω =
                (gap_oracle ω - gap_judge ω) + (gap_judge ω - gap_est ω) := by ring
          rw [hdecomp]
          exact abs_add_le _ _
        have hlt_sum :
            |gap_oracle ω - gap_est ω| < b_cal + b_est := by
          exact lt_of_le_of_lt htri (add_lt_add hA_lt hB_lt)
        exact False.elim ((not_lt_of_ge h_diff) hlt_sum)
  have h_mono :
      μ Atot ≤ μ (Acal ∪ Aest) := measure_mono h_subset
  have h_union :
      μ (Acal ∪ Aest) ≤ μ Acal + μ Aest := measure_union_le (μ := μ) Acal Aest
  calc
    μ {ω | |gap_oracle ω| ≥ |gap_est ω| + b_cal + b_est}
        = μ Atot := by rfl
    _ ≤ μ (Acal ∪ Aest) := h_mono
    _ ≤ μ Acal + μ Aest := h_union
    _ ≤ δ_cal + δ_est := add_le_add h_cal h_est

/-- One-shot high-probability envelope from calibration + estimation + clipping. -/
theorem dsl_abs_gap_bound_from_clipped_estimate_high_prob
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (gap_oracle gap_judge gap_est gap_clip : Ω → ℝ)
    (b_cal b_est b_clip : ℝ)
    (δ_cal δ_est δ_clip : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - gap_est ω| ≥ b_est} ≤ δ_est)
    (h_clip :
      μ {ω | |gap_est ω - gap_clip ω| ≥ b_clip} ≤ δ_clip) :
    μ {ω | |gap_oracle ω| ≥ |gap_clip ω| + b_cal + b_est + b_clip} ≤
      δ_cal + δ_est + δ_clip := by
  let eCal : Ω → ℝ := fun ω => gap_oracle ω - gap_judge ω
  let eEst : Ω → ℝ := fun ω => gap_judge ω - gap_est ω
  let eClip : Ω → ℝ := fun ω => gap_est ω - gap_clip ω
  let Acal : Set Ω := {ω | |eCal ω| ≥ b_cal}
  let Aest : Set Ω := {ω | |eEst ω| ≥ b_est}
  let Aclip : Set Ω := {ω | |eClip ω| ≥ b_clip}
  let Atot : Set Ω := {ω | |gap_oracle ω| ≥ |gap_clip ω| + b_cal + b_est + b_clip}
  have h_subset : Atot ⊆ Acal ∪ Aest ∪ Aclip := by
    intro ω hω
    have h_atot : |gap_oracle ω| ≥ |gap_clip ω| + (b_cal + b_est + b_clip) := by
      simpa [Atot, add_assoc] using hω
    have h_diff :
        |gap_oracle ω - gap_clip ω| ≥ b_cal + b_est + b_clip :=
      abs_sub_ge_of_abs_ge_abs_plus (x := gap_oracle ω) (y := gap_clip ω)
        (b := b_cal + b_est + b_clip) h_atot
    have h_event :
        ω ∈ {ω |
          |eCal ω + eEst ω + eClip ω| ≥
            (fun _ => b_cal) ω + (fun _ => b_est) ω + (fun _ => b_clip) ω} := by
      have hsum :
          eCal ω + eEst ω + eClip ω = gap_oracle ω - gap_clip ω := by
        simp [eCal, eEst, eClip]
      have hrad :
          (fun _ => b_cal) ω + (fun _ => b_est) ω + (fun _ => b_clip) ω =
            b_cal + b_est + b_clip := by ring
      simpa [hsum, hrad] using h_diff
    have h_union_event :=
      DSL.threeLayer_error_event_subset
        (e_chunk := eCal) (e_sum := eEst) (e_oracle := eClip)
        (r_chunk := fun _ => b_cal) (r_sum := fun _ => b_est) (r_oracle := fun _ => b_clip)
        h_event
    simpa [Acal, Aest, Aclip] using h_union_event
  have h_mono : μ Atot ≤ μ (Acal ∪ Aest ∪ Aclip) := measure_mono h_subset
  have h_union :
      μ (Acal ∪ Aest ∪ Aclip) ≤ μ Acal + μ Aest + μ Aclip := by
    calc
      μ (Acal ∪ Aest ∪ Aclip)
          = μ ((Acal ∪ Aest) ∪ Aclip) := by simp [Set.union_assoc]
      _ ≤ μ (Acal ∪ Aest) + μ Aclip := measure_union_le (μ := μ) (Acal ∪ Aest) Aclip
      _ ≤ (μ Acal + μ Aest) + μ Aclip := by
            exact add_le_add (measure_union_le (μ := μ) Acal Aest) (le_refl _)
      _ = μ Acal + μ Aest + μ Aclip := by ring
  have h_sum :
      μ Acal + μ Aest + μ Aclip ≤ δ_cal + δ_est + δ_clip := by
    exact add_le_add (add_le_add h_cal h_est) h_clip
  calc
    μ {ω | |gap_oracle ω| ≥ |gap_clip ω| + b_cal + b_est + b_clip}
        = μ Atot := by rfl
    _ ≤ μ (Acal ∪ Aest ∪ Aclip) := h_mono
    _ ≤ μ Acal + μ Aest + μ Aclip := h_union
    _ ≤ δ_cal + δ_est + δ_clip := h_sum

/-- One-shot high-probability envelope from oracle measurement + calibration +
estimation errors. This is the event-level form of the general
`true -> oracle -> judge -> estimate` chain. -/
theorem dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (gap_true gap_oracle gap_judge gap_est : Ω → ℝ)
    (b_oracle b_cal b_est : ℝ)
    (δ_oracle δ_cal δ_est : ENNReal)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ b_oracle} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - gap_est ω| ≥ b_est} ≤ δ_est) :
    μ {ω | |gap_true ω| ≥ |gap_est ω| + b_oracle + b_cal + b_est} ≤
      δ_oracle + δ_cal + δ_est := by
  simpa [add_assoc, add_left_comm, add_comm] using
    (dsl_abs_gap_bound_from_clipped_estimate_high_prob
      (μ := μ)
      (gap_oracle := gap_true) (gap_judge := gap_oracle)
      (gap_est := gap_judge) (gap_clip := gap_est)
      (b_cal := b_oracle) (b_est := b_cal) (b_clip := b_est)
      (δ_cal := δ_oracle) (δ_est := δ_cal) (δ_clip := δ_est)
      h_oracle h_cal h_est)

/-- Total-failure-budget form of the two-component one-shot envelope. -/
theorem dsl_abs_gap_bound_from_estimate_high_prob_total
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (gap_oracle gap_judge gap_est : Ω → ℝ)
    (b_cal b_est : ℝ)
    (δ_cal δ_est δ_total : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - gap_est ω| ≥ b_est} ≤ δ_est)
    (h_split : δ_cal + δ_est ≤ δ_total) :
    μ {ω | |gap_oracle ω| ≥ |gap_est ω| + b_cal + b_est} ≤ δ_total := by
  exact le_trans
    (dsl_abs_gap_bound_from_estimate_high_prob
      (μ := μ) (gap_oracle := gap_oracle) (gap_judge := gap_judge) (gap_est := gap_est)
      (b_cal := b_cal) (b_est := b_est)
      (δ_cal := δ_cal) (δ_est := δ_est)
      h_cal h_est)
    h_split

/-- Total-failure-budget form of the three-component oracle-measurement
envelope. -/
theorem dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement_total
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (gap_true gap_oracle gap_judge gap_est : Ω → ℝ)
    (b_oracle b_cal b_est : ℝ)
    (δ_oracle δ_cal δ_est δ_total : ENNReal)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ b_oracle} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - gap_est ω| ≥ b_est} ≤ δ_est)
    (h_split : δ_oracle + δ_cal + δ_est ≤ δ_total) :
    μ {ω | |gap_true ω| ≥ |gap_est ω| + b_oracle + b_cal + b_est} ≤ δ_total := by
  exact le_trans
    (dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement
      (μ := μ)
      (gap_true := gap_true) (gap_oracle := gap_oracle)
      (gap_judge := gap_judge) (gap_est := gap_est)
      (b_oracle := b_oracle) (b_cal := b_cal) (b_est := b_est)
      (δ_oracle := δ_oracle) (δ_cal := δ_cal) (δ_est := δ_est)
      h_oracle h_cal h_est)
    h_split

/-- Total-failure-budget form of the three-component one-shot envelope. -/
theorem dsl_abs_gap_bound_from_clipped_estimate_high_prob_total
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (gap_oracle gap_judge gap_est gap_clip : Ω → ℝ)
    (b_cal b_est b_clip : ℝ)
    (δ_cal δ_est δ_clip δ_total : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b_cal} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - gap_est ω| ≥ b_est} ≤ δ_est)
    (h_clip :
      μ {ω | |gap_est ω - gap_clip ω| ≥ b_clip} ≤ δ_clip)
    (h_split : δ_cal + δ_est + δ_clip ≤ δ_total) :
    μ {ω | |gap_oracle ω| ≥ |gap_clip ω| + b_cal + b_est + b_clip} ≤ δ_total := by
  exact le_trans
    (dsl_abs_gap_bound_from_clipped_estimate_high_prob
      (μ := μ) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (gap_est := gap_est) (gap_clip := gap_clip)
      (b_cal := b_cal) (b_est := b_est) (b_clip := b_clip)
      (δ_cal := δ_cal) (δ_est := δ_est) (δ_clip := δ_clip)
      h_cal h_est h_clip)
    h_split

/-- High-probability validity of a `DSLBound` from calibration and estimation events. -/
theorem dsl_bound_valid_from_events
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : DSLBound) (gap_oracle gap_judge : Ω → ℝ)
    (δ_cal δ_est : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b.bias_margin} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - b.gap_estimate| ≥ b.z_score * b.se} ≤ δ_est)
    (h_est_nonneg : 0 ≤ b.gap_estimate) :
    μ {ω | |gap_oracle ω| ≥ b.upperBound} ≤ δ_cal + δ_est := by
  have h_core :=
    dsl_abs_gap_bound_from_estimate_high_prob
      (μ := μ)
      (gap_oracle := gap_oracle)
      (gap_judge := gap_judge)
      (gap_est := fun _ => b.gap_estimate)
      (b_cal := b.bias_margin)
      (b_est := b.z_score * b.se)
      (δ_cal := δ_cal) (δ_est := δ_est)
      (h_cal := h_cal)
      (h_est := by simpa using h_est)
  simpa [DSLBound.upperBound, DSLBound.totalMargin, abs_of_nonneg h_est_nonneg,
    add_assoc, add_left_comm, add_comm] using h_core

/-- High-probability validity of a `DSLBound` for a true target that is only
approximately represented by the oracle target. Exact oracle labels are the
special case `oracle_err = 0`. -/
theorem dsl_bound_valid_from_events_with_oracleMeasurement
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : DSLBound) (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (δ_oracle δ_cal δ_est : ENNReal)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ oracle_err} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b.bias_margin} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - b.gap_estimate| ≥ b.z_score * b.se} ≤ δ_est)
    (h_est_nonneg : 0 ≤ b.gap_estimate) :
    μ {ω | |gap_true ω| ≥ b.upperBound + oracle_err} ≤ δ_oracle + δ_cal + δ_est := by
  have h_core :=
    dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement
      (μ := μ)
      (gap_true := gap_true)
      (gap_oracle := gap_oracle)
      (gap_judge := gap_judge)
      (gap_est := fun _ => b.gap_estimate)
      (b_oracle := oracle_err)
      (b_cal := b.bias_margin)
      (b_est := b.z_score * b.se)
      (δ_oracle := δ_oracle)
      (δ_cal := δ_cal)
      (δ_est := δ_est)
      (h_oracle := h_oracle)
      (h_cal := h_cal)
      (h_est := by simpa using h_est)
  simpa [DSLBound.upperBound, DSLBound.totalMargin, abs_of_nonneg h_est_nonneg,
    add_assoc, add_left_comm, add_comm] using h_core

/-- Total-failure-budget form of `dsl_bound_valid_from_events`. -/
theorem dsl_bound_valid_from_events_total
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : DSLBound) (gap_oracle gap_judge : Ω → ℝ)
    (δ_cal δ_est δ_total : ENNReal)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b.bias_margin} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - b.gap_estimate| ≥ b.z_score * b.se} ≤ δ_est)
    (h_est_nonneg : 0 ≤ b.gap_estimate)
    (h_split : δ_cal + δ_est ≤ δ_total) :
    μ {ω | |gap_oracle ω| ≥ b.upperBound} ≤ δ_total := by
  exact le_trans
    (dsl_bound_valid_from_events (μ := μ) (b := b)
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (δ_cal := δ_cal) (δ_est := δ_est)
      h_cal h_est h_est_nonneg)
    h_split

/-- Total-failure-budget form of
`dsl_bound_valid_from_events_with_oracleMeasurement`. -/
theorem dsl_bound_valid_from_events_with_oracleMeasurement_total
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : DSLBound) (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (δ_oracle δ_cal δ_est δ_total : ENNReal)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ oracle_err} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥ b.bias_margin} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - b.gap_estimate| ≥ b.z_score * b.se} ≤ δ_est)
    (h_est_nonneg : 0 ≤ b.gap_estimate)
    (h_split : δ_oracle + δ_cal + δ_est ≤ δ_total) :
    μ {ω | |gap_true ω| ≥ b.upperBound + oracle_err} ≤ δ_total := by
  exact le_trans
    (dsl_bound_valid_from_events_with_oracleMeasurement
      (μ := μ) (b := b)
      (gap_true := gap_true) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (oracle_err := oracle_err)
      (δ_oracle := δ_oracle) (δ_cal := δ_cal) (δ_est := δ_est)
      h_oracle h_cal h_est h_est_nonneg)
    h_split

/-- High-probability validity for `computeDSLBound` under explicit event assumptions. -/
theorem computeDSLBound_valid_from_events
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_oracle gap_judge : Ω → ℝ)
    (δ_cal δ_est : ENNReal)
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - ipwUnionBound samples N M R| ≥
        z * ipwUnionBoundSE samples N M R} ≤ δ_est) :
    μ {ω | |gap_oracle ω| ≥ (computeDSLBound samples N M R cal z).upperBound} ≤
      δ_cal + δ_est := by
  simpa [computeDSLBound] using
    (dsl_bound_valid_from_events (μ := μ)
      (b := computeDSLBound samples N M R cal z)
      (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (δ_cal := δ_cal) (δ_est := δ_est)
      h_cal h_est h_est_nonneg)

/-- High-probability validity for `computeDSLBound` with an explicit oracle
measurement envelope on top of the usual calibration and estimation events. -/
theorem computeDSLBound_valid_from_events_with_oracleMeasurement
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (δ_oracle δ_cal δ_est : ENNReal)
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ oracle_err} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - ipwUnionBound samples N M R| ≥
        z * ipwUnionBoundSE samples N M R} ≤ δ_est) :
    μ {ω | |gap_true ω| ≥ (computeDSLBound samples N M R cal z).upperBound + oracle_err} ≤
      δ_oracle + δ_cal + δ_est := by
  simpa [computeDSLBound] using
    (dsl_bound_valid_from_events_with_oracleMeasurement (μ := μ)
      (b := computeDSLBound samples N M R cal z)
      (gap_true := gap_true) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
      (oracle_err := oracle_err)
      (δ_oracle := δ_oracle) (δ_cal := δ_cal) (δ_est := δ_est)
      h_oracle h_cal h_est h_est_nonneg)

/-- DSL guarantee theorem (non-tautological): event-level validity for the computed bound. -/
theorem dsl_bound_valid
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_oracle gap_judge : Ω → ℝ)
    (δ_cal δ_est : ENNReal)
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - ipwUnionBound samples N M R| ≥
        z * ipwUnionBoundSE samples N M R} ≤ δ_est) :
    μ {ω | |gap_oracle ω| ≥ (computeDSLBound samples N M R cal z).upperBound} ≤
      δ_cal + δ_est :=
  computeDSLBound_valid_from_events
    (μ := μ) (samples := samples) (N := N) (M := M) (R := R)
    (cal := cal) (z := z) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
    (δ_cal := δ_cal) (δ_est := δ_est)
    h_est_nonneg h_cal h_est

/-- DSL guarantee theorem for the more general regime where the oracle target
may itself be measured with bounded error. -/
theorem dsl_bound_valid_with_oracleMeasurement
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples : List TreeSample) (N M R : ℕ)
    (cal : Option CalibrationSet) (z : ℝ)
    (gap_true gap_oracle gap_judge : Ω → ℝ)
    (oracle_err : ℝ)
    (δ_oracle δ_cal δ_est : ENNReal)
    (h_est_nonneg : 0 ≤ ipwUnionBound samples N M R)
    (h_oracle :
      μ {ω | |gap_true ω - gap_oracle ω| ≥ oracle_err} ≤ δ_oracle)
    (h_cal :
      μ {ω | |gap_oracle ω - gap_judge ω| ≥
        match cal with
        | some c => judgeCalibrationErrorBound c z
        | none => 0} ≤ δ_cal)
    (h_est :
      μ {ω | |gap_judge ω - ipwUnionBound samples N M R| ≥
        z * ipwUnionBoundSE samples N M R} ≤ δ_est) :
    μ {ω | |gap_true ω| ≥ (computeDSLBound samples N M R cal z).upperBound + oracle_err} ≤
      δ_oracle + δ_cal + δ_est :=
  computeDSLBound_valid_from_events_with_oracleMeasurement
    (μ := μ) (samples := samples) (N := N) (M := M) (R := R)
    (cal := cal) (z := z)
    (gap_true := gap_true) (gap_oracle := gap_oracle) (gap_judge := gap_judge)
    (oracle_err := oracle_err)
    (δ_oracle := δ_oracle) (δ_cal := δ_cal) (δ_est := δ_est)
    h_est_nonneg h_oracle h_cal h_est

/-!
## Section 6.6: Sequential Audit Stopping

These wrappers turn any fixed-horizon event family with a scheduled failure
budget into an anytime-valid bound for a data-dependent stopping rule. The
stopping rule itself need not be measurable for the set-theoretic inclusion
argument used below.
-/

/-- A scheduled family of bad-event bounds implies a bound on the union over all
times. -/
theorem scheduled_iUnion_bound
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (E : ℕ → Set Ω)
    (δ : ℕ → ENNReal)
    (hE : ∀ n, μ (E n) ≤ δ n) :
    μ (⋃ n, E n) ≤ ∑' n, δ n := by
  have h_union : μ (⋃ n, E n) ≤ ∑' n, μ (E n) := measure_iUnion_le (μ := μ) (s := E)
  exact h_union.trans (ENNReal.tsum_le_tsum hE)

/-- If every horizon-specific bad event is scheduled with its own failure budget,
then the bad event evaluated at an arbitrary stopping time inherits the sum of
those budgets. -/
theorem stopped_event_bound_of_scheduled_events
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    (E : ℕ → Set Ω)
    (τ : Ω → ℕ)
    (δ : ℕ → ENNReal)
    (hE : ∀ n, μ (E n) ≤ δ n) :
    μ {ω | ω ∈ E (τ ω)} ≤ ∑' n, δ n := by
  have h_subset : {ω | ω ∈ E (τ ω)} ⊆ ⋃ n, E n := by
    intro ω hω
    exact Set.mem_iUnion.mpr ⟨τ ω, hω⟩
  exact (measure_mono h_subset).trans (scheduled_iUnion_bound μ E δ hE)

/-- Anytime-valid IPW violation-rate empirical-Bernstein bound from a scheduled
family of fixed-horizon events. -/
theorem stopped_ipw_violation_rate_empirical_bernstein
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples_seq : ℕ → Ω → List TreeSample)
    (τ : Ω → ℕ)
    (mean_true : ℝ)
    (δ : ℕ → ℝ)
    (h_event :
      ∀ n,
        μ {ω |
          |ipwViolationRate (samples_seq n ω) - mean_true| ≥
            empiricalBernsteinRadius
              (toWeightedSamples (samples_seq n ω)) (δ n) 1} ≤
          ENNReal.ofReal (δ n)) :
    μ {ω |
      |ipwViolationRate (samples_seq (τ ω) ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (samples_seq (τ ω) ω)) (δ (τ ω)) 1} ≤
      ∑' n, ENNReal.ofReal (δ n) := by
  exact stopped_event_bound_of_scheduled_events
    (μ := μ)
    (E := fun n => {ω |
      |ipwViolationRate (samples_seq n ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (samples_seq n ω)) (δ n) 1})
    (τ := τ)
    (δ := fun n => ENNReal.ofReal (δ n))
    h_event

/-- Anytime-valid IPW violation-rate empirical-Bernstein bound from the
existing fixed-horizon EB axioms interface. -/
theorem stopped_ipw_violation_rate_empirical_bernstein_from_axioms
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples_seq : ℕ → Ω → List TreeSample)
    (τ : Ω → ℕ)
    (mean_true : ℝ)
    (h_nonempty : ∀ n ω, samples_seq n ω ≠ [])
    (axioms :
      ∀ n,
        EmpiricalBernsteinAxioms μ
          (fun ω => toWeightedSamples (samples_seq n ω)) mean_true 1)
    (δ : ℕ → ℝ)
    (hδ_pos : ∀ n, 0 < δ n)
    (hδ_lt : ∀ n, δ n < 1) :
    μ {ω |
      |ipwViolationRate (samples_seq (τ ω) ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedSamples (samples_seq (τ ω) ω)) (δ (τ ω)) 1} ≤
      ∑' n, ENNReal.ofReal (δ n) := by
  refine stopped_ipw_violation_rate_empirical_bernstein
    (μ := μ) (samples_seq := samples_seq) (τ := τ)
    (mean_true := mean_true) (δ := δ) ?_
  intro n
  exact ipw_violation_rate_empirical_bernstein_from_axioms
    (μ := μ) (samples := samples_seq n) (mean_true := mean_true)
    (h_nonempty := h_nonempty n) (axioms := axioms n)
    (δ := δ n) (hδ_pos := hδ_pos n) (hδ_lt := hδ_lt n)

/-- Anytime-valid IPW preference-loss empirical-Bernstein bound from a
scheduled family of fixed-horizon events. -/
theorem stopped_ipw_preference_loss_empirical_bernstein
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples_seq : ℕ → Ω → List (TreePreferenceSample Strings Node A k))
    (τ : Ω → ℕ)
    (mean_true range : ℝ)
    (δ : ℕ → ℝ)
    (h_event :
      ∀ n,
        μ {ω |
          |ipwPreferenceLoss (samples_seq n ω) - mean_true| ≥
            empiricalBernsteinRadius
              (toWeightedPrefSamples (samples_seq n ω)) (δ n) range} ≤
          ENNReal.ofReal (δ n)) :
    μ {ω |
      |ipwPreferenceLoss (samples_seq (τ ω) ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedPrefSamples (samples_seq (τ ω) ω)) (δ (τ ω)) range} ≤
      ∑' n, ENNReal.ofReal (δ n) := by
  exact stopped_event_bound_of_scheduled_events
    (μ := μ)
    (E := fun n => {ω |
      |ipwPreferenceLoss (samples_seq n ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedPrefSamples (samples_seq n ω)) (δ n) range})
    (τ := τ)
    (δ := fun n => ENNReal.ofReal (δ n))
    h_event

/-- Anytime-valid IPW preference-loss empirical-Bernstein bound from the
existing fixed-horizon EB axioms interface. -/
theorem stopped_ipw_preference_loss_empirical_bernstein_from_axioms
    {Ω Strings Node A : Type*} {k : ℕ} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (samples_seq : ℕ → Ω → List (TreePreferenceSample Strings Node A k))
    (τ : Ω → ℕ)
    (mean_true range : ℝ)
    (h_nonempty : ∀ n ω, samples_seq n ω ≠ [])
    (axioms :
      ∀ n,
        EmpiricalBernsteinAxioms μ
          (fun ω => toWeightedPrefSamples (samples_seq n ω)) mean_true range)
    (δ : ℕ → ℝ)
    (hδ_pos : ∀ n, 0 < δ n)
    (hδ_lt : ∀ n, δ n < 1) :
    μ {ω |
      |ipwPreferenceLoss (samples_seq (τ ω) ω) - mean_true| ≥
        empiricalBernsteinRadius
          (toWeightedPrefSamples (samples_seq (τ ω) ω)) (δ (τ ω)) range} ≤
      ∑' n, ENNReal.ofReal (δ n) := by
  refine stopped_ipw_preference_loss_empirical_bernstein
    (μ := μ) (samples_seq := samples_seq) (τ := τ)
    (mean_true := mean_true) (range := range) (δ := δ) ?_
  intro n
  exact ipw_preference_loss_empirical_bernstein_from_axioms
    (μ := μ) (samples := samples_seq n) (mean_true := mean_true) (range := range)
    (h_nonempty := h_nonempty n) (axioms := axioms n)
    (δ := δ n) (hδ_pos := hδ_pos n) (hδ_lt := hδ_lt n)

/-!
## Section 7: Practical Diagnostics
-/

/-- Effective sample size for tree samples -/
def treeEffectiveSampleSize (samples : List TreeSample) : ℝ :=
  effectiveSampleSize (toWeightedSamples samples)

/-- Check if effective sample size is adequate.

Rule of thumb: n_eff / n should be > 0.5.
If lower, weights are too variable; increase exploration. -/
def hasAdequateTreeNeff (samples : List TreeSample) (threshold : ℝ := 0.5) : Bool :=
  let n := samples.length
  let n_eff := treeEffectiveSampleSize samples
  threshold * n ≤ n_eff

/-- Check if there are enough clusters (documents) for reliable inference.

Rule of thumb: need at least 30 clusters for normal approximation. -/
def hasEnoughTreeClusters (samples : List TreeSample) (threshold : ℕ := 30) : Bool :=
  let doc_ids := (samples.map TreeSample.doc_id).eraseDups
  threshold ≤ doc_ids.length

/-- Check maximum weight (indicates exploration adequacy).

If max weight is too large, some samples dominate; increase exploration. -/
def maxTreeWeight (samples : List TreeSample) : ℝ :=
  maxWeight (toWeightedSamples samples)

/-- Check if weights are bounded adequately.

Rule of thumb: max_weight should be < 10 × mean_weight -/
def hasAdequateWeightBound (samples : List TreeSample) (multiplier : ℝ := 10) : Bool :=
  if samples.isEmpty then true
  else
    let weights := toWeightedSamples samples
    let max_w := maxWeight weights
    let mean_w := sumWeights weights / samples.length
    max_w ≤ multiplier * mean_w

/-!
## Section 8: Oracle/Judge Split Analysis
-/

/-- Separate oracle-labeled and judge-labeled samples -/
def splitByLabel (samples : List TreeSample) : List TreeSample × List TreeSample :=
  (samples.filter TreeSample.is_oracle_labeled,
   samples.filter (fun s => !s.is_oracle_labeled))

/-- Oracle sample proportion (for monitoring) -/
def oracleProportion (samples : List TreeSample) : ℝ :=
  if samples.isEmpty then 0
  else
    let (oracle_samples, _) := splitByLabel samples
    oracle_samples.length / samples.length

/-!
## Section 9: Exploration Floor Utilities
-/

/-- Apply exploration floor to adaptive probability.

This ensures p_mixture ≥ eps * p_uniform, which bounds weights. -/
def applyExplorationFloor (p_adaptive p_uniform eps : ℝ) : ℝ :=
  mixtureProbability p_adaptive p_uniform eps

/-- Compute required exploration epsilon to achieve target weight bound.

If max_weight_target = W, and p_uniform_min = p_min:
  eps ≥ 1 / (W × p_min) -/
def requiredEpsilon (max_weight_target p_uniform_min : ℝ) : ℝ :=
  if max_weight_target ≤ 0 || p_uniform_min ≤ 0 then 1
  else 1 / (max_weight_target * p_uniform_min)

/-!
## Section 9.5: Clipping Bias and Error Decomposition

These definitions and lemmas formalize the computational bias/variance tradeoff
from weight clipping. They are deterministic finite-sample statements.
-/

namespace TreeSample

/-- Clipped IPW weight `min(w, w_max)`. -/
def clippedWeight (s : TreeSample) (w_max : ℝ) : ℝ :=
  min s.weight w_max

/-- Clipping excess `w - min(w, w_max)` (nonnegative). -/
def clippingExcess (s : TreeSample) (w_max : ℝ) : ℝ :=
  s.weight - s.clippedWeight w_max

lemma clippedWeight_le_weight (s : TreeSample) (w_max : ℝ) :
    s.clippedWeight w_max ≤ s.weight := by
  unfold clippedWeight
  exact min_le_left _ _

lemma clippingExcess_nonneg (s : TreeSample) (w_max : ℝ) :
    0 ≤ s.clippingExcess w_max := by
  unfold clippingExcess
  exact sub_nonneg.mpr (s.clippedWeight_le_weight w_max)

/-- Sample-level clipping distortion term `(w - w_clip) * y`. -/
def clippingBiasTerm (s : TreeSample) (w_max : ℝ) : ℝ :=
  s.clippingExcess w_max * s.outcome

/-- Deterministic absolute bound on sample-level clipping bias. -/
lemma clippingBiasTerm_abs_le (s : TreeSample) (w_max M : ℝ)
    (hM : 0 ≤ M) (h_outcome : |s.outcome| ≤ M) :
    |s.clippingBiasTerm w_max| ≤ M * s.clippingExcess w_max := by
  have h_excess_nonneg : 0 ≤ s.clippingExcess w_max := s.clippingExcess_nonneg w_max
  calc
    |s.clippingBiasTerm w_max|
        = |s.clippingExcess w_max * s.outcome| := by rfl
    _ = |s.clippingExcess w_max| * |s.outcome| := by simpa [abs_mul]
    _ = s.clippingExcess w_max * |s.outcome| := by
          simp [abs_of_nonneg h_excess_nonneg]
    _ ≤ s.clippingExcess w_max * M := by
          exact mul_le_mul_of_nonneg_left h_outcome h_excess_nonneg
    _ = M * s.clippingExcess w_max := by ring

end TreeSample

/-- Total clipping excess over a sample list. -/
def totalClippingExcess (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  (samples.map (fun s => s.clippingExcess w_max)).sum

/-- Total clipping bias term over a sample list. -/
def totalClippingBias (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  (samples.map (fun s => s.clippingBiasTerm w_max)).sum

/-- Deterministic absolute clipping-bias envelope `M * Σ (w - w_clip)`. -/
def clippingBiasAbsBound (samples : List TreeSample) (w_max M : ℝ) : ℝ :=
  M * totalClippingExcess samples w_max

/-- Aggregate clipping-bias bound under a uniform outcome envelope `|y| ≤ M`. -/
lemma totalClippingBias_abs_le (samples : List TreeSample) (w_max M : ℝ)
    (hM : 0 ≤ M)
    (h_outcome : ∀ s ∈ samples, |s.outcome| ≤ M) :
    |totalClippingBias samples w_max| ≤ clippingBiasAbsBound samples w_max M := by
  induction samples with
  | nil =>
      simp [totalClippingBias, clippingBiasAbsBound, totalClippingExcess]
  | cons s ss ih =>
      have hs_outcome : |s.outcome| ≤ M := h_outcome s (by simp)
      have hs_bound :
          |s.clippingBiasTerm w_max| ≤ M * s.clippingExcess w_max :=
        s.clippingBiasTerm_abs_le w_max M hM hs_outcome
      have h_outcome_tail : ∀ t ∈ ss, |t.outcome| ≤ M := by
        intro t ht
        exact h_outcome t (by simp [ht])
      have h_tail :
          |totalClippingBias ss w_max| ≤ clippingBiasAbsBound ss w_max M :=
        ih h_outcome_tail
      calc
        |totalClippingBias (s :: ss) w_max|
            = |s.clippingBiasTerm w_max + totalClippingBias ss w_max| := by
                simp [totalClippingBias]
        _ ≤ |s.clippingBiasTerm w_max| + |totalClippingBias ss w_max| := by
              exact abs_add_le _ _
        _ ≤ M * s.clippingExcess w_max + clippingBiasAbsBound ss w_max M := by
              exact add_le_add hs_bound h_tail
        _ = clippingBiasAbsBound (s :: ss) w_max M := by
              simp [clippingBiasAbsBound, totalClippingExcess, mul_add, add_comm, add_left_comm,
                add_assoc]

/-- Mean clipping bias term. -/
def meanClippingBias (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  if h : samples.length = 0 then 0
  else totalClippingBias samples w_max / samples.length

/-- Clipping variance around the mean clipping bias term. -/
def clippingVariance (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  if h : samples.length = 0 then 0
  else
    let b := meanClippingBias samples w_max
    ((samples.map (fun s => (s.clippingBiasTerm w_max - b)^2)).sum) / samples.length

/-- Clipping MSE: bias² + variance (definition-level decomposition). -/
def clippingMSE (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  (meanClippingBias samples w_max)^2 + clippingVariance samples w_max

/-- Clipping MSE decomposition is definitional. -/
theorem clippingMSE_decomposition (samples : List TreeSample) (w_max : ℝ) :
    clippingMSE samples w_max =
      (meanClippingBias samples w_max)^2 + clippingVariance samples w_max := by
  rfl

/-!
### Clipped Hajek: Deterministic Bias Controls
-/

/-- Raw weighted outcome sum `Σ w_i y_i`. -/
def rawWeightedOutcomeSum (samples : List TreeSample) : ℝ :=
  (samples.map (fun s => s.weight * s.outcome)).sum

/-- Clipped weighted outcome sum `Σ min(w_i, w_max) y_i`. -/
def clippedWeightedOutcomeSum (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  (samples.map (fun s => s.clippedWeight w_max * s.outcome)).sum

/-- Raw denominator `Σ w_i`. -/
def rawWeightSum (samples : List TreeSample) : ℝ :=
  (samples.map TreeSample.weight).sum

/-- Clipped denominator `Σ min(w_i, w_max)`. -/
def clippedWeightSum (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  (samples.map (fun s => s.clippedWeight w_max)).sum

/-- Unclipped Hajek ratio. -/
def rawHajekEstimator (samples : List TreeSample) : ℝ :=
  rawWeightedOutcomeSum samples / rawWeightSum samples

/-- Clipped Hajek ratio. -/
def clippedHajekEstimator (samples : List TreeSample) (w_max : ℝ) : ℝ :=
  clippedWeightedOutcomeSum samples w_max / clippedWeightSum samples w_max

lemma totalClippingExcess_nonneg (samples : List TreeSample) (w_max : ℝ) :
    0 ≤ totalClippingExcess samples w_max := by
  induction samples with
  | nil =>
      simp [totalClippingExcess]
  | cons s ss ih =>
      simp [totalClippingExcess]
      exact add_nonneg (s.clippingExcess_nonneg w_max) ih

lemma rawWeightSum_sub_clippedWeightSum_eq_totalClippingExcess
    (samples : List TreeSample) (w_max : ℝ) :
    rawWeightSum samples - clippedWeightSum samples w_max =
      totalClippingExcess samples w_max := by
  induction samples with
  | nil =>
      simp [rawWeightSum, clippedWeightSum, totalClippingExcess]
  | cons s ss ih =>
      calc
        rawWeightSum (s :: ss) - clippedWeightSum (s :: ss) w_max
            = (s.weight + rawWeightSum ss) -
                (s.clippedWeight w_max + clippedWeightSum ss w_max) := by
                  simp [rawWeightSum, clippedWeightSum]
        _ = (s.weight - s.clippedWeight w_max) +
              (rawWeightSum ss - clippedWeightSum ss w_max) := by ring
        _ = s.clippingExcess w_max + totalClippingExcess ss w_max := by
              simp [ih, TreeSample.clippingExcess]
        _ = totalClippingExcess (s :: ss) w_max := by
              simp [totalClippingExcess]

lemma rawWeightSum_eq_clipped_plus_excess
    (samples : List TreeSample) (w_max : ℝ) :
    rawWeightSum samples =
      clippedWeightSum samples w_max + totalClippingExcess samples w_max := by
  have h := rawWeightSum_sub_clippedWeightSum_eq_totalClippingExcess samples w_max
  linarith

lemma rawWeightSum_pos_of_clippedWeightSum_pos
    (samples : List TreeSample) (w_max : ℝ)
    (h_clip_pos : 0 < clippedWeightSum samples w_max) :
    0 < rawWeightSum samples := by
  have h_eq : rawWeightSum samples =
      clippedWeightSum samples w_max + totalClippingExcess samples w_max :=
    rawWeightSum_eq_clipped_plus_excess samples w_max
  have h_excess_nonneg : 0 ≤ totalClippingExcess samples w_max :=
    totalClippingExcess_nonneg samples w_max
  have h_ge : clippedWeightSum samples w_max ≤ rawWeightSum samples := by
    rw [h_eq]
    linarith
  exact lt_of_lt_of_le h_clip_pos h_ge

lemma rawWeightedOutcomeSum_sub_clippedWeightedOutcomeSum_eq_totalClippingBias
    (samples : List TreeSample) (w_max : ℝ) :
    rawWeightedOutcomeSum samples - clippedWeightedOutcomeSum samples w_max =
      totalClippingBias samples w_max := by
  induction samples with
  | nil =>
      simp [rawWeightedOutcomeSum, clippedWeightedOutcomeSum, totalClippingBias]
  | cons s ss ih =>
      calc
        rawWeightedOutcomeSum (s :: ss) - clippedWeightedOutcomeSum (s :: ss) w_max
            = (s.weight * s.outcome + rawWeightedOutcomeSum ss) -
                (s.clippedWeight w_max * s.outcome + clippedWeightedOutcomeSum ss w_max) := by
                  simp [rawWeightedOutcomeSum, clippedWeightedOutcomeSum]
        _ = (s.weight * s.outcome - s.clippedWeight w_max * s.outcome) +
              (rawWeightedOutcomeSum ss - clippedWeightedOutcomeSum ss w_max) := by ring
        _ = s.clippingBiasTerm w_max + totalClippingBias ss w_max := by
              rw [TreeSample.clippingBiasTerm, TreeSample.clippingExcess, ih]
              ring
        _ = totalClippingBias (s :: ss) w_max := by
              simp [totalClippingBias]

lemma rawWeightedOutcomeSum_abs_le
    (samples : List TreeSample) (M : ℝ) (hM : 0 ≤ M)
    (h_outcome : ∀ s ∈ samples, |s.outcome| ≤ M) :
    |rawWeightedOutcomeSum samples| ≤ M * rawWeightSum samples := by
  induction samples with
  | nil =>
      simp [rawWeightedOutcomeSum, rawWeightSum]
  | cons s ss ih =>
      have hs_outcome : |s.outcome| ≤ M := h_outcome s (by simp)
      have h_outcome_tail : ∀ t ∈ ss, |t.outcome| ≤ M := by
        intro t ht
        exact h_outcome t (by simp [ht])
      have hs_term :
          |s.weight * s.outcome| ≤ M * s.weight := by
        have h_weight_pos : 0 < s.weight := (TreeSample.toWeightedSample s).weight_pos
        calc
          |s.weight * s.outcome|
              = |s.weight| * |s.outcome| := by simp [abs_mul]
          _ = s.weight * |s.outcome| := by simp [abs_of_pos h_weight_pos]
          _ ≤ s.weight * M := by
                exact mul_le_mul_of_nonneg_left hs_outcome
                  (le_of_lt h_weight_pos)
          _ = M * s.weight := by ring
      have h_tail :
          |rawWeightedOutcomeSum ss| ≤ M * rawWeightSum ss :=
        ih h_outcome_tail
      calc
        |rawWeightedOutcomeSum (s :: ss)|
            = |s.weight * s.outcome + rawWeightedOutcomeSum ss| := by
                simp [rawWeightedOutcomeSum]
        _ ≤ |s.weight * s.outcome| + |rawWeightedOutcomeSum ss| := by
              exact abs_add_le _ _
        _ ≤ M * s.weight + M * rawWeightSum ss := by
              exact add_le_add hs_term h_tail
        _ = M * rawWeightSum (s :: ss) := by
              simp [rawWeightSum, mul_add, add_comm, add_left_comm, add_assoc]

lemma rawMinusClippedOutcome_abs_le
    (samples : List TreeSample) (w_max M : ℝ)
    (hM : 0 ≤ M)
    (h_outcome : ∀ s ∈ samples, |s.outcome| ≤ M) :
    |rawWeightedOutcomeSum samples - clippedWeightedOutcomeSum samples w_max| ≤
      M * totalClippingExcess samples w_max := by
  have h_bias :
      |totalClippingBias samples w_max| ≤ clippingBiasAbsBound samples w_max M :=
    totalClippingBias_abs_le samples w_max M hM h_outcome
  have h_eq :
      rawWeightedOutcomeSum samples - clippedWeightedOutcomeSum samples w_max =
        totalClippingBias samples w_max :=
    rawWeightedOutcomeSum_sub_clippedWeightedOutcomeSum_eq_totalClippingBias samples w_max
  simpa [h_eq, clippingBiasAbsBound]
    using h_bias

lemma clippedHajek_minus_rawHajek_decompose
    (samples : List TreeSample) (w_max : ℝ)
    (h_clip_ne : clippedWeightSum samples w_max ≠ 0)
    (h_raw_ne : rawWeightSum samples ≠ 0) :
    clippedHajekEstimator samples w_max - rawHajekEstimator samples =
      (clippedWeightedOutcomeSum samples w_max - rawWeightedOutcomeSum samples) /
        clippedWeightSum samples w_max +
      rawWeightedOutcomeSum samples *
        (rawWeightSum samples - clippedWeightSum samples w_max) /
        (clippedWeightSum samples w_max * rawWeightSum samples) := by
  unfold clippedHajekEstimator rawHajekEstimator
  field_simp [h_clip_ne, h_raw_ne]
  ring

/-- Deterministic clipped-vs-unclipped Hajek gap bound.

This controls clipping-induced estimator bias in terms of:
- outcome envelope `M`,
- total clipped mass `Σ(w - w_clip)`,
- clipped denominator `Σ w_clip`. -/
theorem clippedHajek_abs_diff_le
    (samples : List TreeSample) (w_max M : ℝ)
    (hM : 0 ≤ M)
    (h_outcome : ∀ s ∈ samples, |s.outcome| ≤ M)
    (h_clip_pos : 0 < clippedWeightSum samples w_max) :
    |clippedHajekEstimator samples w_max - rawHajekEstimator samples| ≤
      2 * M * totalClippingExcess samples w_max / clippedWeightSum samples w_max := by
  let Δ : ℝ := totalClippingExcess samples w_max
  let Dc : ℝ := clippedWeightSum samples w_max
  let D : ℝ := rawWeightSum samples
  let N : ℝ := rawWeightedOutcomeSum samples
  let Nc : ℝ := clippedWeightedOutcomeSum samples w_max
  have hΔ_nonneg : 0 ≤ Δ := by
    simpa [Δ] using totalClippingExcess_nonneg samples w_max
  have hDc_pos : 0 < Dc := by simpa [Dc] using h_clip_pos
  have hD_pos : 0 < D := by
    simpa [D, Dc, Δ] using
      rawWeightSum_pos_of_clippedWeightSum_pos samples w_max h_clip_pos
  have hDc_ne : Dc ≠ 0 := ne_of_gt hDc_pos
  have hD_ne : D ≠ 0 := ne_of_gt hD_pos
  have hN_abs : |N| ≤ M * D := by
    simpa [N, D] using rawWeightedOutcomeSum_abs_le samples M hM h_outcome
  have h_num_diff :
      |Nc - N| ≤ M * Δ := by
    have h :=
      rawMinusClippedOutcome_abs_le samples w_max M hM h_outcome
    simpa [N, Nc, Δ, abs_sub_comm] using h
  have h_den_diff : D - Dc = Δ := by
    simpa [D, Dc, Δ] using
      rawWeightSum_sub_clippedWeightSum_eq_totalClippingExcess samples w_max
  have h_den_abs : |D - Dc| = Δ := by
    simp [h_den_diff, abs_of_nonneg hΔ_nonneg]
  have h_decomp :
      clippedHajekEstimator samples w_max - rawHajekEstimator samples =
        (Nc - N) / Dc + N * (D - Dc) / (Dc * D) := by
    simpa [N, Nc, D, Dc] using
      clippedHajek_minus_rawHajek_decompose samples w_max hDc_ne hD_ne
  have h_first :
      |(Nc - N) / Dc| ≤ (M * Δ) / Dc := by
    calc
      |(Nc - N) / Dc| = |Nc - N| / Dc := by
          simp [abs_div, abs_of_pos hDc_pos]
      _ ≤ (M * Δ) / Dc := by
          exact div_le_div_of_nonneg_right h_num_diff (le_of_lt hDc_pos)
  have h_second :
      |N * (D - Dc) / (Dc * D)| ≤ (M * Δ) / Dc := by
    have hDen_pos : 0 < Dc * D := mul_pos hDc_pos hD_pos
    calc
      |N * (D - Dc) / (Dc * D)|
          = |N| * |D - Dc| / (Dc * D) := by
              simp [abs_mul, abs_div, abs_of_pos hDen_pos, mul_assoc, mul_left_comm, mul_comm]
      _ = |N| * Δ / (Dc * D) := by simp [h_den_abs]
      _ ≤ (M * D) * Δ / (Dc * D) := by
            have hmul : |N| * Δ ≤ (M * D) * Δ :=
              mul_le_mul_of_nonneg_right hN_abs hΔ_nonneg
            exact div_le_div_of_nonneg_right hmul (le_of_lt hDen_pos)
      _ = (M * Δ) / Dc := by
            field_simp [hD_ne]
  calc
    |clippedHajekEstimator samples w_max - rawHajekEstimator samples|
        = |(Nc - N) / Dc + N * (D - Dc) / (Dc * D)| := by
            simp [h_decomp]
    _ ≤ |(Nc - N) / Dc| + |N * (D - Dc) / (Dc * D)| := by
          exact abs_add_le _ _
    _ ≤ (M * Δ) / Dc + (M * Δ) / Dc := by
          exact add_le_add h_first h_second
    _ = 2 * M * Δ / Dc := by ring
    _ = 2 * M * totalClippingExcess samples w_max / clippedWeightSum samples w_max := by
          simp [Δ, Dc]

/-- Relative clipping-mass corollary:
if `Σ(w - w_clip) ≤ ρ Σw_clip`, then clipping bias is at most `2Mρ`. -/
theorem clippedHajek_abs_diff_le_of_relative_excess
    (samples : List TreeSample) (w_max M ρ : ℝ)
    (hM : 0 ≤ M) (hρ : 0 ≤ ρ)
    (h_outcome : ∀ s ∈ samples, |s.outcome| ≤ M)
    (h_clip_pos : 0 < clippedWeightSum samples w_max)
    (h_rel : totalClippingExcess samples w_max ≤ ρ * clippedWeightSum samples w_max) :
    |clippedHajekEstimator samples w_max - rawHajekEstimator samples| ≤ 2 * M * ρ := by
  have h_main :=
    clippedHajek_abs_diff_le samples w_max M hM h_outcome h_clip_pos
  have h_factor_nonneg : 0 ≤ (2 * M) / clippedWeightSum samples w_max := by
    exact div_nonneg (mul_nonneg (by norm_num) hM) (le_of_lt h_clip_pos)
  calc
    |clippedHajekEstimator samples w_max - rawHajekEstimator samples|
        ≤ 2 * M * totalClippingExcess samples w_max / clippedWeightSum samples w_max := h_main
    _ = ((2 * M) / clippedWeightSum samples w_max) * totalClippingExcess samples w_max := by
          ring
    _ ≤ ((2 * M) / clippedWeightSum samples w_max) *
          (ρ * clippedWeightSum samples w_max) := by
            exact mul_le_mul_of_nonneg_left h_rel h_factor_nonneg
    _ = 2 * M * ρ := by
          field_simp [ne_of_gt h_clip_pos]

/-- Unit-range outcome corollary (`|y| ≤ 1`). -/
theorem clippedHajek_abs_diff_le_unit
    (samples : List TreeSample) (w_max : ℝ)
    (h_outcome : ∀ s ∈ samples, |s.outcome| ≤ 1)
    (h_clip_pos : 0 < clippedWeightSum samples w_max) :
    |clippedHajekEstimator samples w_max - rawHajekEstimator samples| ≤
      2 * totalClippingExcess samples w_max / clippedWeightSum samples w_max := by
  have h :=
    clippedHajek_abs_diff_le samples w_max 1 (by norm_num) h_outcome h_clip_pos
  simpa using h

/-!
## Section 10: Summary Statistics
-/

/-- Summary of IPW analysis results -/
structure IPWAnalysisSummary where
  n_samples : ℕ
  n_documents : ℕ
  n_eff : ℝ
  n_eff_ratio : ℝ               -- n_eff / n
  max_weight : ℝ
  oracle_proportion : ℝ
  violation_rate_leaf : ℝ
  violation_rate_merge : ℝ
  violation_rate_idemp : ℝ
  union_bound_estimate : ℝ
  union_bound_se : ℝ
  is_adequate : Bool

/-- Compute summary statistics for tree samples -/
def analyzeTreeSamples (samples : List TreeSample) (N M R : ℕ) : IPWAnalysisSummary :=
  let n := samples.length
  let doc_ids := (samples.map TreeSample.doc_id).eraseDups
  let n_eff := treeEffectiveSampleSize samples
  let n_eff_ratio := if n = 0 then 0 else n_eff / n
  let max_w := maxTreeWeight samples
  let oracle_prop := oracleProportion samples
  let p_leaf := ipwLeafViolationRate samples
  let p_merge := ipwMergeViolationRate samples
  let p_idemp := ipwIdempViolationRate samples
  let ub := ipwUnionBound samples N M R
  let se := ipwUnionBoundSE samples N M R
  let adequate := hasAdequateTreeNeff samples && hasEnoughTreeClusters samples
  { n_samples := n
    n_documents := doc_ids.length
    n_eff := n_eff
    n_eff_ratio := n_eff_ratio
    max_weight := max_w
    oracle_proportion := oracle_prop
    violation_rate_leaf := p_leaf
    violation_rate_merge := p_merge
    violation_rate_idemp := p_idemp
    union_bound_estimate := ub
    union_bound_se := se
    is_adequate := adequate }

/-!
## Section 11: Validity Properties
-/

/-- IPW violation rate is non-negative when all outcomes are non-negative.

In practice, outcomes are violation indicators (0 or 1), which are always non-negative. -/
lemma ipwViolationRate_nonneg (samples : List TreeSample)
    (h_nonneg : ∀ s ∈ samples, 0 ≤ s.outcome) :
    0 ≤ ipwViolationRate samples := by
  unfold ipwViolationRate
  by_cases h : samples.isEmpty
  · simp [h]
  · simp only [h, ↓reduceIte]
    apply hajekEstimator_nonneg
    · -- samples is non-empty, so toWeightedSamples samples is non-empty
      unfold toWeightedSamples
      intro heq
      simp only [List.map_eq_nil_iff] at heq
      rw [heq] at h
      simp at h
    · intro ws hws
      simp only [List.mem_map, toWeightedSamples] at hws
      obtain ⟨ts, hts, rfl⟩ := hws
      exact h_nonneg ts hts

/-- Union bound is non-negative when all outcomes are non-negative. -/
lemma ipwUnionBound_nonneg (samples : List TreeSample) (N M R : ℕ)
    (hR : 1 ≤ R) (h_nonneg : ∀ s ∈ samples, 0 ≤ s.outcome) :
    0 ≤ ipwUnionBound samples N M R := by
  unfold ipwUnionBound
  apply add_nonneg
  apply add_nonneg
  · apply mul_nonneg (Nat.cast_nonneg N)
    apply ipwViolationRate_nonneg
    intro s hs
    simp only [leafSamples, filterByType, List.mem_filter] at hs
    exact h_nonneg s hs.1
  · apply mul_nonneg (Nat.cast_nonneg M)
    apply ipwViolationRate_nonneg
    intro s hs
    simp only [mergeSamples, filterByType, List.mem_filter] at hs
    exact h_nonneg s hs.1
  · apply mul_nonneg
    · have h2 : (1 : ℝ) ≤ (R : ℝ) := Nat.one_le_cast.mpr hR
      linarith
    · apply ipwViolationRate_nonneg
      intro s hs
      simp only [resummarySamples, filterByType, List.mem_filter] at hs
      exact h_nonneg s hs.1

lemma DSLBound.totalMargin_nonneg (b : DSLBound)
    (h_z : 0 ≤ b.z_score) (h_se : 0 ≤ b.se) (h_bias : 0 ≤ b.bias_margin) :
    0 ≤ b.totalMargin := by
  unfold totalMargin
  apply add_nonneg
  · exact mul_nonneg h_z h_se
  · exact h_bias

end
