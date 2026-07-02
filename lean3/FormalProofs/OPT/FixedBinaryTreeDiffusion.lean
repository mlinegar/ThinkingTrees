import FormalProofs.OPT.OracleFiberObjectives
import FormalProofs.OPT.MarkovPathDGP

/-!
# FormalProofs/OPT/FixedBinaryTreeDiffusion.lean

Packaging layer for the fixed-binary-tree diffusion story.

This file does not introduce a new stochastic state space. Instead it exposes a
paired interface built from already-formalized components:

- `TextCheckpoint`: the round-indexed text checkpoint `ZR g x r T`;
- `LatentCheckpoint`: the exact bottom-up latent fold `mergeFold encode merge T`;
- `FixedBinaryTreeDiffusionSpec`: a bundled fixed-tree specification carrying
  the deterministic tree soundness witness `S T = x`, the text summarizer, and
  the latent theorem feature.

The theorems here are intentionally thin wrappers around existing results:

- exact and approximate text-checkpoint preservation;
- exact latent-checkpoint correctness for mergeable states; and
- exact / bounded readout transport through the theorem feature.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*} [Monoid Strings]

/-- Round-indexed text checkpoint on a fixed binary tree. -/
abbrev TextCheckpoint
    (g : Summarizer Strings)
    (x : Strings)
    (T : BinTree Strings)
    (r : ℕ) : PMF Strings :=
  ZR g x r T

/-- Exact latent checkpoint obtained by the bottom-up mergeable fold. -/
abbrev LatentCheckpoint
    {Sketch : Type*}
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch)
    (T : BinTree Strings) : Sketch :=
  mergeFold encode merge T

section Spec

variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Bundled fixed-binary-tree diffusion specification.

`BinTree` is the in-scope fixed binary tree object; the only tree-level
correctness witness required here is that the tree span reduces back to the
document `x`. -/
structure FixedBinaryTreeDiffusionSpec where
  g : Summarizer Strings
  x : Strings
  T : BinTree Strings
  fstar : Strings → Y
  feature : Strings → Sketch
  encode : Strings → Sketch
  merge : Sketch → Sketch → Sketch
  sound : S T = x

namespace FixedBinaryTreeDiffusionSpec

/-- The round-indexed text checkpoint carried by a fixed-tree diffusion spec. -/
abbrev textCheckpoint
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (r : ℕ) : PMF Strings :=
  TextCheckpoint spec.g spec.x spec.T r

/-- The exact latent checkpoint carried by a fixed-tree diffusion spec. -/
abbrev latentCheckpoint
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch)) :
    Sketch :=
  LatentCheckpoint spec.encode spec.merge spec.T

end FixedBinaryTreeDiffusionSpec

end Spec

section TextCheckpoint

variable {Y : Type*} [BoundedPseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Exact text-checkpoint preservation on a fixed binary tree, packaged through
`LocalLawsBundle`. -/
theorem textCheckpoint_distortion_zero_of_localLaws
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (laws : LocalLawsBundle spec.g spec.T spec.fstar)
    {r : ℕ}
    (hr : r ≥ 1) :
    Δ_R_ZR spec.g spec.x r spec.T spec.fstar = 0 := by
  exact
    Δ_R_eq_zero_of_local_laws
      spec.g spec.x r spec.T spec.fstar spec.sound
      laws.law1 laws.law2 laws.law3 hr

end TextCheckpoint

section ApproxTextCheckpoint

variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Approximate text-checkpoint distortion budget on a fixed binary tree,
packaged through `ApproxLocalLawsBundle`. -/
theorem textCheckpoint_distortion_le_of_approxLocalLaws
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (laws : ApproxLocalLawsBundle spec.g spec.T spec.fstar)
    {r : ℕ}
    (hr : r ≥ 1)
    (hbound : ∀ z, D spec.fstar z spec.x ≤ 1)
    (hbound_global : ∀ w z, D spec.fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp spec.g spec.fstar (p.bind spec.g) ≤ pIdemp spec.g spec.fstar p) :
    Δ_R_ZR spec.g spec.x r spec.T spec.fstar ≤
      laws.epsLeaf + laws.epsMerge + ((r : ℝ) - 1) * laws.epsIdemp := by
  exact
    Δ_R_ZR_le_of_approx_bundle
      spec.g spec.T spec.fstar spec.x r spec.sound hr
      hbound hbound_global h_mono laws

end ApproxTextCheckpoint

section LatentCheckpoint

variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch β : Type*}

/-- Exact latent-checkpoint correctness on a fixed binary tree. -/
theorem latentCheckpoint_eq_feature_of_exactMergeable
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (h_encode : ∀ s : Strings, spec.encode s = spec.feature s)
    (h_merge : ∀ s t : Strings,
      spec.merge (spec.feature s) (spec.feature t) = spec.feature (s * t)) :
    spec.latentCheckpoint = spec.feature spec.x := by
  rw [FixedBinaryTreeDiffusionSpec.latentCheckpoint, LatentCheckpoint]
  rw [mergeFold_eq_feature
    (encode := spec.encode)
    (merge := spec.merge)
    (feature := spec.feature)
    h_encode h_merge spec.T]
  simpa [spec.sound]

/-- Any downstream utility on the exact latent checkpoint agrees with the
utility of the theorem feature at the fixed-tree root. -/
theorem latentCheckpoint_utility_eq_root_of_exactMergeable
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (u : Sketch → β)
    (h_encode : ∀ s : Strings, spec.encode s = spec.feature s)
    (h_merge : ∀ s t : Strings,
      spec.merge (spec.feature s) (spec.feature t) = spec.feature (s * t)) :
    u spec.latentCheckpoint = u (spec.feature spec.x) := by
  rw [FixedBinaryTreeDiffusionSpec.latentCheckpoint, LatentCheckpoint]
  rw [mergeableStateUtility_exact_on_tree
    (encode := spec.encode)
    (merge := spec.merge)
    (feature := spec.feature)
    h_encode h_merge
    (u := u)
    (T := spec.T)]
  simpa [spec.sound]

end LatentCheckpoint

section PairedTransport

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Sketch Readout α : Type*}
variable [Encodable Readout]

/-- Exact text-checkpoint transport for any loss indexed by a readout that
factors through the theorem feature. -/
theorem factoredReadout_expectedLoss_eq_via_textCheckpoint_of_localLaws
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (readout : Strings → Readout)
    (loss : Strings → α → ℝ)
    (gen : Strings → PMF α)
    (laws : LocalLawsBundle spec.g spec.T spec.fstar)
    {r : ℕ}
    (hr : r ≥ 1)
    (hRecover : OracleRecoversFeature spec.fstar spec.feature)
    (hFactor : ReadoutFactorsThroughFeature spec.feature readout)
    (h_meas : OracleMeasurableLossGeneric loss (encodedOracle (Strings := Strings) readout))
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) readout)) :
    ExpectedLossGeneric loss (PMF.pure spec.x) gen =
      ExpectedLossGeneric loss (spec.textCheckpoint r) gen := by
  simpa [FixedBinaryTreeDiffusionSpec.textCheckpoint, TextCheckpoint] using
    (expected_loss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout
      (fstar := spec.fstar)
      (feature := spec.feature)
      (readout := readout)
      (loss := loss)
      (gen := gen)
      (g := spec.g) (x := spec.x) (R := r) (T := spec.T)
      spec.sound
      (ExactTheoremBacked.ofLocalLaws laws)
      hr hRecover hFactor h_meas h_gen)

/-- Exact supervised text-checkpoint transport for factored root readouts. -/
theorem factoredReadout_supervisedLoss_eq_via_textCheckpoint_of_localLaws
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (readout : Strings → Readout)
    (loss : Readout → Readout → ℝ)
    (gen : Strings → PMF Readout)
    (laws : LocalLawsBundle spec.g spec.T spec.fstar)
    {r : ℕ}
    (hr : r ≥ 1)
    (hRecover : OracleRecoversFeature spec.fstar spec.feature)
    (hFactor : ReadoutFactorsThroughFeature spec.feature readout)
    (h_gen : OracleIndexedGenGeneric gen (encodedOracle (Strings := Strings) readout)) :
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) readout loss) (PMF.pure spec.x) gen =
      ExpectedLossGeneric
        (supervisedStateLoss (Strings := Strings) readout loss) (spec.textCheckpoint r) gen := by
  simpa [FixedBinaryTreeDiffusionSpec.textCheckpoint, TextCheckpoint] using
    (supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout
      (fstar := spec.fstar)
      (feature := spec.feature)
      (readout := readout)
      (loss := loss)
      (gen := gen)
      (g := spec.g) (x := spec.x) (R := r) (T := spec.T)
      spec.sound
      (ExactTheoremBacked.ofLocalLaws laws)
      hr hRecover hFactor h_gen)

end PairedTransport

section BoundedPairedTransport

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Sketch Task Summary : Type*}
variable [PseudoMetricSpace Sketch] [PseudoMetricSpace Task] [PseudoMetricSpace Summary]

/-- Quantitative paired-head stability on every realized text checkpoint,
relative to the original document, under exact theorem-backedness and
approximate shared-feature factorization. -/
theorem pairedApproxReadoutBound_on_textCheckpointSupport_of_localLaws
    (spec : FixedBinaryTreeDiffusionSpec (Strings := Strings) (Y := Y) (Sketch := Sketch))
    (taskReadout : Strings → Task)
    (summaryReadout : Strings → Summary)
    (laws : LocalLawsBundle spec.g spec.T spec.fstar)
    {r : ℕ}
    (hr : r ≥ 1)
    {ε_fiber ε_task ε_summary : ℝ≥0}
    {L_task L_summary : ℝ≥0}
    (hApproxRecover : ApproxOracleRecoversFeature spec.fstar spec.feature ε_fiber)
    (hTaskApproxFactor :
      ApproxReadoutFactorsThroughFeature spec.feature taskReadout ε_task)
    (hTaskLip : ∃ recover : Sketch → Task,
      (∀ x : Strings, dist (taskReadout x) (recover (spec.feature x)) ≤ (ε_task : ℝ)) ∧
      LipschitzWith L_task recover)
    (hSummaryApproxFactor :
      ApproxReadoutFactorsThroughFeature spec.feature summaryReadout ε_summary)
    (hSummaryLip : ∃ recover : Sketch → Summary,
      (∀ x : Strings, dist (summaryReadout x) (recover (spec.feature x)) ≤ (ε_summary : ℝ)) ∧
      LipschitzWith L_summary recover)
    {z : Strings}
    (hz : z ∈ (spec.textCheckpoint r).support) :
    dist (taskReadout z) (taskReadout spec.x) ≤
        (L_task : ℝ) * (ε_fiber : ℝ) + 2 * (ε_task : ℝ) ∧
      dist (summaryReadout z) (summaryReadout spec.x) ≤
        (L_summary : ℝ) * (ε_fiber : ℝ) + 2 * (ε_summary : ℝ) := by
  simpa [FixedBinaryTreeDiffusionSpec.textCheckpoint, TextCheckpoint] using
    (zr_support_paired_approxReadoutBound_of_exactTheoremBacked_and_sharedFeature
      (fstar := spec.fstar)
      (feature := spec.feature)
      (taskReadout := taskReadout)
      (summaryReadout := summaryReadout)
      (g := spec.g) (x := spec.x) (R := r) (T := spec.T)
      (hp := spec.sound)
      (hExact := ExactTheoremBacked.ofLocalLaws laws)
      (hR := hr)
      (hApproxRecover := hApproxRecover)
      (hTaskApproxFactor := hTaskApproxFactor)
      (hTaskLip := hTaskLip)
      (hSummaryApproxFactor := hSummaryApproxFactor)
      (hSummaryLip := hSummaryLip)
      hz)

end BoundedPairedTransport

section WorkedExample

/-- Worked fixed-tree latent-checkpoint example on the Markov-path exact state. -/
theorem markovPath_latentCheckpoint_exact_example
    {n : ℕ}
    (T : BinTree (MarkovPath n)) :
    LatentCheckpoint
      (Strings := MarkovPath n)
      (encode := MarkovPath.encodePath (n := n))
      (merge := (· * ·))
      T =
      MarkovPath.encodePath (S T) := by
  simpa [LatentCheckpoint] using
    (mergeFold_eq_feature
      (Strings := MarkovPath n)
      (encode := MarkovPath.encodePath (n := n))
      (merge := (· * ·))
      (feature := MarkovPath.encodePath (n := n))
      (h_encode := fun _ => rfl)
      (h_merge := fun x y => (MarkovPath.encodePath_append (n := n) (xs := x) (ys := y)).symm)
      (T := T))

/-- Worked fixed-tree exact-utility example on the Markov-path theorem state. -/
abbrev markovPath_state_exact_example :=
  @MarkovPath.state_exact_on_tree

/-- Fixed-tree counterexample showing that count-only summaries are not
compositionally sufficient in the Markov family. -/
abbrev markovPath_count_only_counterexample :=
  @MarkovPath.countOnly_mergeFold_counterexample

end WorkedExample

end FormalProofs.OPT
