import FormalProofs.ML.NeuralOperatorApproximation
import FormalProofs.OPT.TheoremBackingAssumptions

/-!
# FormalProofs/OPT/NeuralOperatorTheoremBridge.lean

Bridge from neural-operator approximation to approximate theorem-backedness.

The central point of this file is intentionally modest and explicit:

- an ideal deterministic summarizer `sStar` may already be known to be
  `ExactTheoremBacked`;
- a realized summarizer `sApprox` may approximate `sStar` uniformly on compact
  realized-call sets;
- to turn that approximation statement into audited local-law budgets, one still
  needs explicit transfer assumptions from approximation error to leaf / merge /
  idempotence violation budgets.

Given those ingredients, the existing `ApproxLocalLawsBundle` machinery can be
reused unchanged.
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

open ML

variable {Strings : Type*} [Monoid Strings] [PseudoMetricSpace Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- Compact realized-call containers for the three neural-operator call sites
used in C-TreePO: leaves, internal merges, and the root-level re-summary call. -/
structure NeuralOperatorRealizedCallCompacts
    (T : BinTree Strings) (s : Strings → Strings) where
  leaf : CompactRealizedCallSet Strings
  merge : CompactRealizedCallSet (BinTree Strings × BinTree Strings)
  onRange : CompactRealizedCallSet Strings
  leaf_mem : ∀ b, b ∈ leaves T → b ∈ leaf.carrier
  merge_mem : ∀ p, p ∈ internal_nodes T → p ∈ merge.carrier
  root_onRange : reduceDeterministic s T ∈ onRange.carrier

/-- Leaf exactness extracted from exact theorem-backedness for a deterministic
ideal summarizer. -/
lemma leaf_dist_zero_of_exactTheoremBacked
    {sStar : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    {b : Strings} (hb : b ∈ leaves T) :
    D fstar (sStar b) b = 0 := by
  have hLeaf := hExact.localLaws.law1 b hb
  rw [Eg_deterministic_summaryOp] at hLeaf
  simpa [D] using hLeaf

/-- Merge exactness extracted from exact theorem-backedness for a deterministic
ideal summarizer. -/
lemma merge_dist_zero_of_exactTheoremBacked
    {sStar : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    {p : BinTree Strings × BinTree Strings} (hp : p ∈ internal_nodes T) :
    D fstar (reduceDeterministic sStar (BinTree.node p.1 p.2))
      (S (BinTree.node p.1 p.2)) = 0 := by
  rcases p with ⟨TL, TR⟩
  simpa [Egu_deterministic_summaryOp, D] using
    hExact.localLaws.law2 (TL, TR) hp

/-- On-range exactness extracted from exact theorem-backedness for a
deterministic ideal summarizer. -/
lemma onRange_dist_zero_of_exactTheoremBacked
    {sStar : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    {Z : Strings}
    (hZ : InRange (deterministicSummarizer sStar) Z) :
    D fstar (sStar Z) Z = 0 := by
  have hIdemp := hExact.localLaws.law3 Z hZ
  rw [Eg_deterministic_summaryOp] at hIdemp
  simpa [D] using hIdemp

/-- Explicit bridge assumptions from uniform neural-operator approximation to
approximate theorem-backedness. The transfer fields are where the paper's
Lipschitz / modulus constants live. -/
structure NeuralOperatorTheoremBridgeAssumptions
    (sStar sApprox : Strings → Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε : ℝ) where
  callCompacts : NeuralOperatorRealizedCallCompacts (T := T) (s := sApprox)
  epsLeaf : Strings → ℝ
  epsMerge : BinTree Strings × BinTree Strings → ℝ
  epsIdemp : ℝ
  leafApprox :
    UniformOperatorApproxOnCompact sStar sApprox callCompacts.leaf ε
  mergeApprox :
    UniformOperatorApproxOnCompact
      (fun p : BinTree Strings × BinTree Strings =>
        reduceDeterministic sStar (BinTree.node p.1 p.2))
      (fun p : BinTree Strings × BinTree Strings =>
        reduceDeterministic sApprox (BinTree.node p.1 p.2))
      callCompacts.merge ε
  onRangeApprox :
    UniformOperatorApproxOnCompact sStar sApprox callCompacts.onRange ε
  onRange_target :
    ∀ Z, Z ∈ callCompacts.onRange.carrier →
      InRange (deterministicSummarizer sStar) Z
  leaf_transfer :
    ∀ b, b ∈ callCompacts.leaf.carrier →
      D fstar (sStar b) b = 0 →
      dist (sStar b) (sApprox b) ≤ ε →
      ViolationProb fstar (deterministicSummarizer sApprox b) b ≤ epsLeaf b
  merge_transfer :
    ∀ p, p ∈ callCompacts.merge.carrier →
      D fstar (reduceDeterministic sStar (BinTree.node p.1 p.2))
        (S (BinTree.node p.1 p.2)) = 0 →
      dist
        (reduceDeterministic sStar (BinTree.node p.1 p.2))
        (reduceDeterministic sApprox (BinTree.node p.1 p.2)) ≤ ε →
      ViolationProb fstar
        (reduce (deterministicSummarizer sApprox) (BinTree.node p.1 p.2))
        (S (BinTree.node p.1 p.2)) ≤ epsMerge p
  idemp_transfer :
    ∀ Z, Z ∈ callCompacts.onRange.carrier →
      D fstar (sStar Z) Z = 0 →
      dist (sStar Z) (sApprox Z) ≤ ε →
      ViolationProb fstar (deterministicSummarizer sApprox Z) Z ≤ epsIdemp

/-- Finite-dimensionalization version of the neural-operator bridge
assumptions.  Instead of taking uniform approximation on the three call
compacts as primitive, this bundle takes Lemma-22 witnesses and identifies
their realized encoder--map--decoder operators with the implemented
summarizer/reduction on the relevant compact. -/
structure NeuralOperatorFiniteDimensionalizationBridgeAssumptions
    (sStar sApprox : Strings → Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (ε : ℝ) where
  callCompacts : NeuralOperatorRealizedCallCompacts (T := T) (s := sApprox)
  epsLeaf : Strings → ℝ
  epsMerge : BinTree Strings × BinTree Strings → ℝ
  epsIdemp : ℝ
  leafFD :
    KovachkiFiniteDimensionalizedApproxOnCompact sStar callCompacts.leaf ε
  mergeFD :
    KovachkiFiniteDimensionalizedApproxOnCompact
      (fun p : BinTree Strings × BinTree Strings =>
        reduceDeterministic sStar (BinTree.node p.1 p.2))
      callCompacts.merge ε
  onRangeFD :
    KovachkiFiniteDimensionalizedApproxOnCompact sStar callCompacts.onRange ε
  leaf_realized_eq :
    ∀ b, b ∈ callCompacts.leaf.carrier →
      leafFD.realized b = sApprox b
  merge_realized_eq :
    ∀ p, p ∈ callCompacts.merge.carrier →
      mergeFD.realized p =
        reduceDeterministic sApprox (BinTree.node p.1 p.2)
  onRange_realized_eq :
    ∀ Z, Z ∈ callCompacts.onRange.carrier →
      onRangeFD.realized Z = sApprox Z
  onRange_target :
    ∀ Z, Z ∈ callCompacts.onRange.carrier →
      InRange (deterministicSummarizer sStar) Z
  leaf_transfer :
    ∀ b, b ∈ callCompacts.leaf.carrier →
      D fstar (sStar b) b = 0 →
      dist (sStar b) (sApprox b) ≤ ε →
      ViolationProb fstar (deterministicSummarizer sApprox b) b ≤ epsLeaf b
  merge_transfer :
    ∀ p, p ∈ callCompacts.merge.carrier →
      D fstar (reduceDeterministic sStar (BinTree.node p.1 p.2))
        (S (BinTree.node p.1 p.2)) = 0 →
      dist
        (reduceDeterministic sStar (BinTree.node p.1 p.2))
        (reduceDeterministic sApprox (BinTree.node p.1 p.2)) ≤ ε →
      ViolationProb fstar
        (reduce (deterministicSummarizer sApprox) (BinTree.node p.1 p.2))
        (S (BinTree.node p.1 p.2)) ≤ epsMerge p
  idemp_transfer :
    ∀ Z, Z ∈ callCompacts.onRange.carrier →
      D fstar (sStar Z) Z = 0 →
      dist (sStar Z) (sApprox Z) ≤ ε →
      ViolationProb fstar (deterministicSummarizer sApprox Z) Z ≤ epsIdemp

/-- Convert Lemma-22 finite-dimensionalization witnesses into the uniform
approximation bridge assumptions used by the existing approximate-local-law
proof. -/
def NeuralOperatorFiniteDimensionalizationBridgeAssumptions.toUniformBridge
    {sStar sApprox : Strings → Strings} {T : BinTree Strings} {fstar : Strings → Y}
    {ε : ℝ}
    (hFD :
      NeuralOperatorFiniteDimensionalizationBridgeAssumptions
        sStar sApprox T fstar ε) :
    NeuralOperatorTheoremBridgeAssumptions sStar sApprox T fstar ε where
  callCompacts := hFD.callCompacts
  epsLeaf := hFD.epsLeaf
  epsMerge := hFD.epsMerge
  epsIdemp := hFD.epsIdemp
  leafApprox := by
    intro b hb
    have hApprox :
        dist (sStar b) (hFD.leafFD.realized b) ≤ ε :=
      uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalizedApprox
        hFD.leafFD b hb
    simpa [hFD.leaf_realized_eq b hb] using hApprox
  mergeApprox := by
    intro p hp
    have hApprox :
        dist
          (reduceDeterministic sStar (BinTree.node p.1 p.2))
          (hFD.mergeFD.realized p) ≤ ε :=
      uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalizedApprox
        hFD.mergeFD p hp
    simpa [hFD.merge_realized_eq p hp] using hApprox
  onRangeApprox := by
    intro Z hZ
    have hApprox :
        dist (sStar Z) (hFD.onRangeFD.realized Z) ≤ ε :=
      uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalizedApprox
        hFD.onRangeFD Z hZ
    simpa [hFD.onRange_realized_eq Z hZ] using hApprox
  onRange_target := hFD.onRange_target
  leaf_transfer := hFD.leaf_transfer
  merge_transfer := hFD.merge_transfer
  idemp_transfer := hFD.idemp_transfer

/-- Uniform approximation plus explicit transfer assumptions yields an
approximate-local-law bundle for the realized summarizer. -/
def approxLocalLawsBundle_of_uniformApproxExactTheoremBacked
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y} {ε : ℝ}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    (hBridge : NeuralOperatorTheoremBridgeAssumptions sStar sApprox T fstar ε) :
    ApproxLocalLawsBundle (deterministicSummarizer sApprox) T fstar := by
  have hLeafNode :
      L1εNode (deterministicSummarizer sApprox) T fstar hBridge.epsLeaf := by
    intro b hb
    have hMem : b ∈ hBridge.callCompacts.leaf.carrier :=
      hBridge.callCompacts.leaf_mem b hb
    have hExactLeaf :
        D fstar (sStar b) b = 0 :=
      leaf_dist_zero_of_exactTheoremBacked hExact hb
    have hApproxLeaf :
        dist (sStar b) (sApprox b) ≤ ε :=
      hBridge.leafApprox b hMem
    exact hBridge.leaf_transfer b hMem hExactLeaf hApproxLeaf
  have hMergeNode :
      L2εNode (deterministicSummarizer sApprox) T fstar hBridge.epsMerge := by
    intro p hp
    have hMem : p ∈ hBridge.callCompacts.merge.carrier :=
      hBridge.callCompacts.merge_mem p hp
    have hExactMerge :
        D fstar (reduceDeterministic sStar (BinTree.node p.1 p.2))
          (S (BinTree.node p.1 p.2)) = 0 :=
      merge_dist_zero_of_exactTheoremBacked hExact hp
    have hApproxMerge :
        dist
          (reduceDeterministic sStar (BinTree.node p.1 p.2))
          (reduceDeterministic sApprox (BinTree.node p.1 p.2)) ≤ ε :=
      hBridge.mergeApprox p hMem
    simpa using hBridge.merge_transfer p hMem hExactMerge hApproxMerge
  have hIdemp :
      L3ε (deterministicSummarizer sApprox) T fstar hBridge.epsIdemp := by
    unfold L3ε
    rw [reduce_deterministic_eq_pure]
    rw [pIdemp_pure_eq]
    let Z := reduceDeterministic sApprox T
    have hMem : Z ∈ hBridge.callCompacts.onRange.carrier :=
      hBridge.callCompacts.root_onRange
    have hInRangeStar :
        InRange (deterministicSummarizer sStar) Z :=
      hBridge.onRange_target Z hMem
    have hExactIdemp :
        D fstar (sStar Z) Z = 0 :=
      onRange_dist_zero_of_exactTheoremBacked hExact hInRangeStar
    have hApproxIdemp :
        dist (sStar Z) (sApprox Z) ≤ ε :=
      hBridge.onRangeApprox Z hMem
    simpa [Z] using hBridge.idemp_transfer Z hMem hExactIdemp hApproxIdemp
  exact approx_bundle_of_nodewise
    (g := deterministicSummarizer sApprox)
    (T := T)
    (fstar := fstar)
    (ε_leaf := hBridge.epsLeaf)
    (ε_merge := hBridge.epsMerge)
    (ε_idemp := hBridge.epsIdemp)
    hLeafNode hMergeNode hIdemp

/-- Main bridge: the realized neural operator becomes approximately
theorem-backed once uniform approximation and explicit transfer assumptions are
available on the realized call compacts. -/
def approxTheoremBacked_of_uniformApproxExactTheoremBacked
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y} {ε : ℝ}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    (hBridge : NeuralOperatorTheoremBridgeAssumptions sStar sApprox T fstar ε) :
    ApproxTheoremBacked (deterministicSummarizer sApprox) T fstar :=
  ApproxTheoremBacked.ofApproxLocalLaws
    (approxLocalLawsBundle_of_uniformApproxExactTheoremBacked hExact hBridge)

/-- Lemma-22 finite-dimensionalization bridge: exact theorem-backedness for the
ideal summarizer plus finite-dimensionalized approximations on the three
realized call compacts implies approximate theorem-backedness for the realized
summarizer. -/
def approxTheoremBacked_of_kovachkiFiniteDimensionalization
    {sStar sApprox : Strings → Strings}
    {T : BinTree Strings} {fstar : Strings → Y} {ε : ℝ}
    (hExact : ExactTheoremBacked (deterministicSummarizer sStar) T fstar)
    (hFD :
      NeuralOperatorFiniteDimensionalizationBridgeAssumptions
        sStar sApprox T fstar ε) :
    ApproxTheoremBacked (deterministicSummarizer sApprox) T fstar :=
  approxTheoremBacked_of_uniformApproxExactTheoremBacked
    hExact hFD.toUniformBridge

end FormalProofs.OPT
