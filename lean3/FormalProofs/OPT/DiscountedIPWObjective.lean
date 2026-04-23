import FormalProofs.OPT.CoverageNormalizedObjective
import FormalProofs.OPT.DiscountedTreeMetaObjective

/-!
# FormalProofs/OPT/DiscountedIPWObjective.lean

Bridge between discounted tree-style objectives and IPW / Horvitz-Thompson
estimation.

The key point is simple: if each supervision component is estimated unbiasedly
with HT/IPW, then any fixed linear weighting scheme applied to those components
remains unbiased. This covers:

- depth discounting with weights `γ^d`,
- the current root / C1 / C2 / C3 weighting scheme, and
- combined schemes obtained by taking a product index such as depth × channel.

So adding a reinforcement-learning-style discount factor does not break the
design-based logic. It only changes the deterministic coefficients in front of
already unbiased component estimators.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section GenericWeightedIPW

variable {Doc Depth Θ : Type*}
variable [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]
variable [Fintype Depth] [DecidableEq Depth]

/-- Finite expectation commutes with pushforward of a finite PMF. -/
lemma finiteExpectation_map
    {α β : Type*} [Fintype α] [Fintype β]
    (μ : PMF α) (g : α → β) (f : β → ℝ) :
    finiteExpectation (PMF.map g μ) f =
      finiteExpectation μ (fun a => f (g a)) := by
  classical
  have hμ :
      μ = PMF.ofFintype (fun a => μ a) (by simpa [tsum_fintype] using μ.tsum_coe) := by
    ext a
    simp
  rw [hμ, PMF.map_ofFintype]
  unfold finiteExpectation
  calc
    ∑ x, ((∑ a with g a = x, μ a)).toReal * f x
      = ∑ x, (∑ a with g a = x, (μ a).toReal) * f x := by
          apply Finset.sum_congr rfl
          intro x hx
          congr 1
          rw [ENNReal.toReal_sum]
          intro a ha
          exact PMF.apply_ne_top _ _
    _ = ∑ x, ∑ a with g a = x, (μ a).toReal * f x := by
          apply Finset.sum_congr rfl
          intro x hx
          calc
            (∑ a with g a = x, (μ a).toReal) * f x
              = f x * ∑ a with g a = x, (μ a).toReal := by ring
            _ = ∑ a with g a = x, f x * (μ a).toReal := by
                  exact Finset.mul_sum (s := Finset.univ.filter fun a => g a = x)
                    (f := fun a => (μ a).toReal) (a := f x)
            _ = ∑ a with g a = x, (μ a).toReal * f x := by
                  apply Finset.sum_congr rfl
                  intro a ha
                  ring
    _ = ∑ x, ∑ a with g a = x, (μ a).toReal * f (g a) := by
          apply Finset.sum_congr rfl
          intro x hx
          apply Finset.sum_congr rfl
          intro a ha
          simp at ha
          simp [ha]
    _ = ∑ a, (μ a).toReal * f (g a) := by
          simpa using
            (Finset.sum_fiberwise_of_maps_to
              (s := (Finset.univ : Finset α))
              (t := (Finset.univ : Finset β))
              (g := g)
              (f := fun a => (μ a).toReal * f (g a))
              (h := fun a ha => by simp))

omit [DecidableEq Doc] [Nonempty Doc] in
/-- Finite expectation commutes with finite sums over an external index. -/
lemma finiteExpectation_sum
    (μ : PMF (Depth → Finset Doc))
    (f : Depth → (Depth → Finset Doc) → ℝ) :
    finiteExpectation μ (fun selected => ∑ d, f d selected) =
      ∑ d, finiteExpectation μ (fun selected => f d selected) := by
  unfold finiteExpectation
  calc
    ∑ selected, (μ selected).toReal * ∑ d, f d selected
      = ∑ selected, ∑ d, (μ selected).toReal * f d selected := by
          apply Finset.sum_congr rfl
          intro selected hselected
          rw [Finset.mul_sum]
    _ = ∑ d, ∑ selected, (μ selected).toReal * f d selected := by
          simpa using
            (Finset.sum_comm
              (s := (Finset.univ : Finset (Depth → Finset Doc)))
              (t := (Finset.univ : Finset Depth))
              (f := fun selected d => (μ selected).toReal * f d selected))
    _ = ∑ d, finiteExpectation μ (fun selected => f d selected) := by
          rfl

/-- Generic population objective built as a finite weighted sum of
document-level component means. The index type can stand for supervision
channels, tree depths, or depth × channel pairs. -/
def fullWeightedDocumentObjective
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ) : Θ → ℝ :=
  fun θ => ∑ d, weights d * documentMean (fun i => componentLoss θ d i)

/-- IPW-corrected version of the same objective. For each component `d`, the
logged subset `selected d` is corrected with the HT mean estimator. -/
def expectedIPWWeightedDocumentObjective
    (μ : PMF (Depth → Finset Doc))
    (coverage : Depth → ℝ)
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ) : Θ → ℝ :=
  fun θ =>
    finiteExpectation μ (fun selected =>
      ∑ d, weights d *
        constantInclusionHTRootMeanOfProb (coverage d) (selected d)
          (fun i => componentLoss θ d i))

/-- If each component has constant marginal inclusion probability, the expected
IPW-weighted objective equals the full population objective. -/
theorem expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective
    (μ : PMF (Depth → Finset Doc))
    (coverage : Depth → ℝ)
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ)
    (hcoverage : ∀ d : Depth, coverage d ≠ 0)
    (hmarg :
      ∀ d : Depth, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    expectedIPWWeightedDocumentObjective μ coverage weights componentLoss =
      fullWeightedDocumentObjective weights componentLoss := by
  funext θ
  unfold expectedIPWWeightedDocumentObjective fullWeightedDocumentObjective
  rw [finiteExpectation_sum]
  simp_rw [finiteExpectation_mul_left]
  apply Finset.sum_congr rfl
  intro d hd
  have hroot :
      finiteExpectation μ
          (fun x => constantInclusionHTRootMeanOfProb (coverage d) (x d)
            (fun i => componentLoss θ d i)) =
        documentMean (fun i => componentLoss θ d i) := by
    rw [← finiteExpectation_map
      (μ := μ)
      (g := fun x : Depth → Finset Doc => x d)
      (f := fun selected =>
        constantInclusionHTRootMeanOfProb (coverage d) selected
          (fun i => componentLoss θ d i))]
    exact finiteExpectation_constantInclusionHTRootMean_eq_documentMean
      (μ := PMF.map (fun x : Depth → Finset Doc => x d) μ)
      (coverage := coverage d)
      (rootLoss := fun i => componentLoss θ d i)
      (hcoverage := hcoverage d)
      (hmarg := by
        intro i
        rw [finiteExpectation_map
          (μ := μ)
          (g := fun x : Depth → Finset Doc => x d)
          (f := fun selected : Finset Doc => if i ∈ selected then (1 : ℝ) else 0)]
        exact hmarg d i)
  rw [hroot]

/-- Pointwise-equal expected objectives have the same parameter argmin set. -/
theorem ipwWeightedObjective_same_paramArgmin
    (μ : PMF (Depth → Finset Doc))
    (coverage : Depth → ℝ)
    (weights : Depth → ℝ)
    (componentLoss : Θ → Depth → Doc → ℝ)
    (hcoverage : ∀ d : Depth, coverage d ≠ 0)
    (hmarg :
      ∀ d : Depth, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    ParamArgmin (expectedIPWWeightedDocumentObjective μ coverage weights componentLoss) =
      ParamArgmin (fullWeightedDocumentObjective weights componentLoss) := by
  apply paramArgmin_eq_of_pointwise_loss_eq
  intro θ
  have hEq := congrArg (fun f => f θ)
    (expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective
      (μ := μ) (coverage := coverage) (weights := weights)
      (componentLoss := componentLoss) hcoverage hmarg)
  simpa using hEq

end GenericWeightedIPW

section DiscountedSpecialization

variable {Doc Θ : Type*}
variable [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- RL-style discount weights indexed by finite depth. -/
def discountedDepthWeights {n : ℕ} (γ : ℝ) : Fin n → ℝ :=
  fun d => γ ^ (d : ℕ)

/-- Population objective with depth discounting. -/
def fullDiscountedDocumentObjective {n : ℕ}
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ) : Θ → ℝ :=
  fullWeightedDocumentObjective (discountedDepthWeights γ) depthLoss

/-- IPW-corrected discounted objective. -/
def expectedIPWDiscountedDocumentObjective {n : ℕ}
    (μ : PMF (Fin n → Finset Doc))
    (coverage : Fin n → ℝ)
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ) : Θ → ℝ :=
  expectedIPWWeightedDocumentObjective μ coverage (discountedDepthWeights γ) depthLoss

/-- Discounting by `γ^d` preserves HT/IPW unbiasedness under constant marginal
inclusion probabilities at each depth. -/
theorem expectedIPWDiscountedDocumentObjective_eq_fullDiscountedDocumentObjective
    {n : ℕ}
    (μ : PMF (Fin n → Finset Doc))
    (coverage : Fin n → ℝ)
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ)
    (hcoverage : ∀ d : Fin n, coverage d ≠ 0)
    (hmarg :
      ∀ d : Fin n, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    expectedIPWDiscountedDocumentObjective μ coverage γ depthLoss =
      fullDiscountedDocumentObjective γ depthLoss := by
  exact expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective
    (μ := μ) (coverage := coverage) (weights := discountedDepthWeights γ)
    (componentLoss := depthLoss) hcoverage hmarg

/-- Therefore the IPW-corrected discounted objective and the full discounted
objective have the same parameter argmin set. -/
theorem ipwDiscountedObjective_same_paramArgmin
    {n : ℕ}
    (μ : PMF (Fin n → Finset Doc))
    (coverage : Fin n → ℝ)
    (γ : ℝ)
    (depthLoss : Θ → Fin n → Doc → ℝ)
    (hcoverage : ∀ d : Fin n, coverage d ≠ 0)
    (hmarg :
      ∀ d : Fin n, ∀ i : Doc,
        finiteExpectation μ (fun selected => if i ∈ selected d then (1 : ℝ) else 0) = coverage d) :
    ParamArgmin (expectedIPWDiscountedDocumentObjective μ coverage γ depthLoss) =
      ParamArgmin (fullDiscountedDocumentObjective γ depthLoss) := by
  exact ipwWeightedObjective_same_paramArgmin
    (μ := μ) (coverage := coverage) (weights := discountedDepthWeights γ)
    (componentLoss := depthLoss) hcoverage hmarg

end DiscountedSpecialization

section CurrentWeightingScheme

variable {Doc Θ : Type*}
variable [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- The current tree-training supervision channels. This packages the existing
root / C1 / C2 / C3 weighting scheme as an instance of the generic weighted-IPW
surface. -/
inductive TreeSupervisionChannel
| root
| c1
| c2
| c3
deriving DecidableEq, Fintype

/-- Explicit equivalence used to expand finite sums over the four supervision
channels. -/
def treeSupervisionChannelEquivFin4 : TreeSupervisionChannel ≃ Fin 4 where
  toFun
    | .root => 0
    | .c1 => 1
    | .c2 => 2
    | .c3 => 3
  invFun
    | ⟨0, _⟩ => .root
    | ⟨1, _⟩ => .c1
    | ⟨2, _⟩ => .c2
    | ⟨3, _⟩ => .c3
  left_inv := by
    intro c
    cases c <;> rfl
  right_inv := by
    intro i
    rcases i with ⟨i, hi⟩
    have hi' : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
    rcases hi' with rfl | rfl | rfl | rfl <;> rfl

/-- Closed-form expansion of a sum over the four supervision channels. -/
lemma sum_treeSupervisionChannel (f : TreeSupervisionChannel → ℝ) :
    ∑ c, f c = f .root + f .c1 + f .c2 + f .c3 := by
  let e := treeSupervisionChannelEquivFin4
  calc
    ∑ c, f c = ∑ i : Fin 4, f (e.symm i) := by
      symm
      exact Fintype.sum_equiv e (fun c => f c) (fun i => f (e.symm i)) (by intro x; simp [e])
    _ = f .root + f .c1 + f .c2 + f .c3 := by
      simp [e, treeSupervisionChannelEquivFin4, Fin.sum_univ_four]

/-- Convert the existing tree-objective weight bundle into a generic channel
weight function. -/
def channelWeightOfCoverageNormalized
    (weights : CoverageNormalizedTreeObjectiveWeights) :
    TreeSupervisionChannel → ℝ
| .root => weights.rootWeight
| .c1 => weights.c1Weight
| .c2 => weights.c2Weight
| .c3 => weights.c3Weight

/-- Package the existing root / C1 / C2 / C3 document losses into a single
generic component-loss family. -/
def channelLossFamily
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) :
    Θ → TreeSupervisionChannel → Doc → ℝ
| θ, .root, i => rootLoss θ i
| θ, .c1, i => c1Loss θ i
| θ, .c2, i => c2Loss θ i
| θ, .c3, i => c3Loss θ i

omit [DecidableEq Doc] [Nonempty Doc] in
/-- The current full-supervision tree objective is exactly the generic weighted
document objective instantiated at the four supervision channels. -/
theorem fullWeightedDocumentObjective_eq_fullSupervisionTreeObjectiveFn
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) :
    fullWeightedDocumentObjective
        (channelWeightOfCoverageNormalized weights)
        (channelLossFamily rootLoss c1Loss c2Loss c3Loss)
      = fullSupervisionTreeObjectiveFn weights rootLoss c1Loss c2Loss c3Loss := by
  funext θ
  rw [fullWeightedDocumentObjective, sum_treeSupervisionChannel]
  simp [channelWeightOfCoverageNormalized, channelLossFamily,
    fullSupervisionTreeObjectiveFn, fullSupervisionTreeObjective,
    denseLocalObjective, documentMean]
  ring

end CurrentWeightingScheme

end FormalProofs.OPT
