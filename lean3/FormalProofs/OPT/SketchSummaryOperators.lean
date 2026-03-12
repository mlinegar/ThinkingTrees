import FormalProofs.OPT.LocalLaws

/-!
# FormalProofs/OPT/SketchSummaryOperators.lean

This file adds a reusable bridge for learned/neural sketch operators and
deterministic summary operators:

- deterministic summary operators (`s : Strings → Strings`) and their induced
  probabilistic summarizer (`PMF.pure`);
- sketch operators (`encode/merge/decode`) with compositional preservation
  assumptions;
- a theorem that turns sketch-level assumptions into a `LocalLawsBundle`.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-- List concatenation as multiplication. -/
instance listMul (α : Type*) : Mul (List α) := ⟨List.append⟩

/-- Empty list as identity. -/
instance listOne (α : Type*) : One (List α) := ⟨([] : List α)⟩

/-- Canonical monoid structure on lists via concatenation. -/
instance listMonoid (α : Type*) : Monoid (List α) where
  mul := List.append
  one := []
  mul_assoc := by
    intro a b c
    exact List.append_assoc a b c
  one_mul := by
    intro a
    show ([] : List α) ++ a = a
    rfl
  mul_one := by
    intro a
    show a ++ ([] : List α) = a
    exact List.append_nil a

/-!
## Deterministic Summary Operators
-/

/-- Deterministic summarizer viewed as a degenerate `PMF` summarizer. -/
def deterministicSummarizer (s : Strings → Strings) : Summarizer Strings :=
  fun x => PMF.pure (s x)

/-- Deterministic tree reduction under a summary operator `s`. -/
def reduceDeterministic (s : Strings → Strings) : BinTree Strings → Strings
| BinTree.leaf b => s b
| BinTree.node TL TR => s (reduceDeterministic s TL * reduceDeterministic s TR)

/-- Pointwise oracle preservation of a deterministic summary operator. -/
def PointwisePreserving (s : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ x, dist (fstar (s x)) (fstar x) = 0

/-- Treewise oracle preservation of deterministic tree reduction. -/
def TreewisePreserving (s : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ T, dist (fstar (reduceDeterministic s T)) (fstar (S T)) = 0

/-- `reduce` under a deterministic summarizer is a pure PMF at `reduceDeterministic`. -/
theorem reduce_deterministic_eq_pure (s : Strings → Strings) (T : BinTree Strings) :
    reduce (deterministicSummarizer s) T = PMF.pure (reduceDeterministic s T) := by
  induction T with
  | leaf b =>
      simp [reduce, deterministicSummarizer, reduceDeterministic]
  | node TL TR ihL ihR =>
      simp [reduce, deterministicSummarizer, reduceDeterministic, ihL, ihR]

/-- Utility lemma: `tsum` under an indicator-pure PMF collapses to one term. -/
lemma tsum_indicator_mul_prop {α : Type*} (b : α) (f : α → ℝ) :
    (∑' z : α,
        (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal * f z) = f b := by
  classical
  have h :
      (fun z : α =>
          (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal * f z) =
        fun z : α => if z = b then f z else 0 := by
    funext z
    by_cases hz : z = b <;> simp [hz]
  simp [h, tsum_ite_eq]

/-- Closed form for `Eg` under a deterministic summarizer. -/
theorem Eg_deterministic_summaryOp (s : Strings → Strings) (f : Strings → ℝ) (x : Strings) :
    Eg (deterministicSummarizer s) f x = f (s x) := by
  unfold Eg deterministicSummarizer
  simpa using (tsum_indicator_mul_prop (b := s x) (f := f))

/-- Closed form for `Egu` under a deterministic summarizer. -/
theorem Egu_deterministic_summaryOp (s : Strings → Strings) (T : BinTree Strings) (f : Strings → ℝ) :
    Egu (deterministicSummarizer s) T f = f (reduceDeterministic s T) := by
  unfold Egu
  rw [reduce_deterministic_eq_pure (s := s) (T := T)]
  simpa using (tsum_indicator_mul_prop (b := reduceDeterministic s T) (f := f))

/-- Pointwise preservation implies L1 for every tree. -/
theorem L1_of_pointwise
    (s : Strings → Strings) (fstar : Strings → Y) (T : BinTree Strings)
    (h_pointwise : PointwisePreserving s fstar) :
    L1 (deterministicSummarizer s) T fstar := by
  intro b hb
  simpa [PointwisePreserving, Eg_deterministic_summaryOp, D] using h_pointwise b

/-- Pointwise preservation on all strings implies L3. -/
theorem L3_of_pointwise
    (s : Strings → Strings) (fstar : Strings → Y)
    (h_pointwise : PointwisePreserving s fstar) :
    L3 (deterministicSummarizer s) fstar := by
  intro Z hZ
  simpa [PointwisePreserving, Eg_deterministic_summaryOp, D] using h_pointwise Z

/-- Treewise preservation implies L2 for every tree. -/
theorem L2_of_treewise
    (s : Strings → Strings) (fstar : Strings → Y) (T : BinTree Strings)
    (h_tree : TreewisePreserving s fstar) :
    L2 (deterministicSummarizer s) T fstar := by
  intro p hp
  rcases p with ⟨TL, TR⟩
  simpa [Egu_deterministic_summaryOp, D] using
    h_tree (BinTree.node TL TR)

/-!
## Sketch Operators (`encode / merge / decode`)
-/

variable {Sketch : Type*}

/-- Abstract sketch operator with encoder, merge, and decoder. -/
structure SketchOperator (Strings Sketch : Type*) where
  encode : Strings → Sketch
  merge : Sketch → Sketch → Sketch
  decode : Sketch → Strings

/-- Identity sketch operator on a monoid string type. -/
def identitySketchOperator (Strings : Type*) [Monoid Strings] :
    SketchOperator Strings Strings where
  encode := fun x => x
  merge := (· * ·)
  decode := fun x => x

/-- A simple non-identity sketch operator that stores an auxiliary multiplicative
track in the second component and decodes from the first component.

This is useful as a generic "learned sketch placeholder" where sketch type differs
from `Strings` but decode semantics remain exact. -/
def pairedSketchOperator (Strings : Type*) [Monoid Strings] :
    SketchOperator Strings (Strings × Strings) where
  encode := fun x => (x, 1)
  merge := fun s t => (s.1 * t.1, s.2 * t.2)
  decode := Prod.fst

/-- A genuinely lossy sketch operator on token lists: only the length is kept.
Decode reconstructs a canonical representative with the same length. -/
def lengthSketchOperator (α : Type*) [Inhabited α] :
    SketchOperator (List α) Nat where
  encode := fun xs => xs.length
  merge := Nat.add
  decode := fun n => List.replicate n default

/-- Bottom-up sketch reduction over a tree. -/
def sketchReduce (op : SketchOperator Strings Sketch) : BinTree Strings → Sketch
| BinTree.leaf b => op.encode b
| BinTree.node TL TR => op.merge (sketchReduce op TL) (sketchReduce op TR)

/-- Decoded sketch summary at the tree root. -/
def sketchSummary (op : SketchOperator Strings Sketch) (T : BinTree Strings) : Strings :=
  op.decode (sketchReduce op T)

/-- Span-level summary induced by one sketch encode/decode pass. -/
def summaryFromSketch (op : SketchOperator Strings Sketch) : Strings → Strings :=
  fun x => op.decode (op.encode x)

/-- Leaf-level oracle preservation of sketch encode/decode. -/
def SketchLeafPreserving (op : SketchOperator Strings Sketch) (fstar : Strings → Y) : Prop :=
  PointwisePreserving (summaryFromSketch op) fstar

/-- Merge compatibility at oracle level:
if decoded child sketches match oracle values of `x` and `y`, then decoded merged
sketch matches oracle value of `x * y`. -/
def SketchMergeCompatible (op : SketchOperator Strings Sketch) (fstar : Strings → Y) : Prop :=
  ∀ sL sR x y,
    dist (fstar (op.decode sL)) (fstar x) = 0 →
    dist (fstar (op.decode sR)) (fstar y) = 0 →
    dist (fstar (op.decode (op.merge sL sR))) (fstar (x * y)) = 0

/-- Compatibility between explicit sketch merge and reduction via summary strings. -/
def SketchSummaryCompatible (op : SketchOperator Strings Sketch) : Prop :=
  ∀ sL sR,
    op.decode (op.merge sL sR) =
      summaryFromSketch op (op.decode sL * op.decode sR)

/-- Feature-encoded scalar oracle.
Useful for proving merge compatibility from equality of feature maps. -/
def encodedOracle {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature) : Strings → ℝ :=
  fun x => (Encodable.encode (feature x) : ℝ)

/-- Identity operator always satisfies leaf-preservation for any oracle map. -/
theorem identitySketch_leaf_preserving (fstar : Strings → Y) :
    SketchLeafPreserving (identitySketchOperator (Strings := Strings)) fstar := by
  intro x
  simp [SketchLeafPreserving, PointwisePreserving, summaryFromSketch, identitySketchOperator]

/-- Identity operator is summary-compatible by definition. -/
theorem identitySketch_summary_compatible :
    SketchSummaryCompatible (identitySketchOperator (Strings := Strings)) := by
  intro sL sR
  rfl

/-- The paired non-identity operator preserves leaves for every oracle map. -/
theorem pairedSketch_leaf_preserving (fstar : Strings → Y) :
    SketchLeafPreserving (pairedSketchOperator (Strings := Strings)) fstar := by
  intro x
  simp [SketchLeafPreserving, PointwisePreserving, summaryFromSketch, pairedSketchOperator]

/-- The paired non-identity operator is summary-compatible by construction. -/
theorem pairedSketch_summary_compatible :
    SketchSummaryCompatible (pairedSketchOperator (Strings := Strings)) := by
  intro sL sR
  rfl

/-- The length sketch is summary-compatible (`decode (sL+sR)` equals decode of
the merged decoded strings). -/
theorem lengthSketch_summary_compatible (α : Type*) [Inhabited α] :
    SketchSummaryCompatible (lengthSketchOperator (α := α)) := by
  intro sL sR
  have hlen : sL + sR =
      List.length (List.replicate sL (default : α) * List.replicate sR (default : α)) := by
    change sL + sR = (List.replicate sL (default : α) ++ List.replicate sR (default : α)).length
    simp [List.length_append]
  simp [SketchSummaryCompatible, summaryFromSketch, lengthSketchOperator, hlen]

/-- The lossy length sketch preserves the encoded length oracle at leaves. -/
theorem lengthSketch_leaf_preserving (α : Type*) [Inhabited α] :
    SketchLeafPreserving (lengthSketchOperator (α := α))
      (encodedOracle (Strings := List α) (fun xs : List α => xs.length)) := by
  intro x
  simp [SketchLeafPreserving, PointwisePreserving, summaryFromSketch,
    lengthSketchOperator, encodedOracle]

/-- The lossy length sketch is merge-compatible for the encoded length oracle. -/
theorem lengthSketch_merge_compatible (α : Type*) [Inhabited α] :
    SketchMergeCompatible (lengthSketchOperator (α := α))
      (encodedOracle (Strings := List α) (fun xs : List α => xs.length)) := by
  intro sL sR x y hx hy
  have hL : (sL : ℝ) = (x.length : ℝ) := by
    exact dist_eq_zero.mp (by simpa [lengthSketchOperator, encodedOracle] using hx)
  have hR : (sR : ℝ) = (y.length : ℝ) := by
    exact dist_eq_zero.mp (by simpa [lengthSketchOperator, encodedOracle] using hy)
  have hL' : sL = x.length := Nat.cast_inj.mp hL
  have hR' : sR = y.length := Nat.cast_inj.mp hR
  apply dist_eq_zero.mpr
  calc
    (encodedOracle (Strings := List α) (fun xs : List α => xs.length))
        ((lengthSketchOperator (α := α)).decode ((lengthSketchOperator (α := α)).merge sL sR))
      = ((sL + sR : Nat) : ℝ) := by
          simp [lengthSketchOperator, encodedOracle]
    _ = ((x ++ y).length : ℝ) := by
          simp [hL', hR', List.length_append]
    _ = ((x * y).length : ℝ) := by
          change ((x ++ y).length : ℝ) = ((x ++ y).length : ℝ)
          rfl
    _ = (encodedOracle (Strings := List α) (fun xs : List α => xs.length)) (x * y) := by
          simp [encodedOracle]

/-- In nontrivial alphabets, the length sketch encoder is not injective. -/
theorem lengthSketch_encode_not_injective (α : Type*) [Inhabited α] [Nontrivial α] :
    ¬ Function.Injective (lengthSketchOperator (α := α)).encode := by
  intro h_inj
  obtain ⟨a, b, hab⟩ := exists_pair_ne α
  have h_eq :
      (lengthSketchOperator (α := α)).encode [a] =
      (lengthSketchOperator (α := α)).encode [b] := by
    simp [lengthSketchOperator]
  have h_lists : ([a] : List α) = [b] := h_inj h_eq
  exact hab (List.singleton_inj.mp h_lists)

/-- If a feature map is congruent under monoid multiplication, then the identity
operator is merge-compatible for the corresponding encoded oracle. -/
theorem identitySketch_merge_compatible_of_feature_congruent
    {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature)
    (h_feature_congr :
      ∀ sL sR x y,
        feature sL = feature x →
        feature sR = feature y →
        feature (sL * sR) = feature (x * y)) :
    SketchMergeCompatible (identitySketchOperator (Strings := Strings))
      (encodedOracle (Strings := Strings) feature) := by
  intro sL sR x y hx hy
  have h_code_L :
      encodedOracle (Strings := Strings) feature sL =
      encodedOracle (Strings := Strings) feature x := dist_eq_zero.mp hx
  have h_code_R :
      encodedOracle (Strings := Strings) feature sR =
      encodedOracle (Strings := Strings) feature y := dist_eq_zero.mp hy
  have h_feat_L : feature sL = feature x := by
    exact Encodable.encode_injective (Nat.cast_inj.mp h_code_L)
  have h_feat_R : feature sR = feature y := by
    exact Encodable.encode_injective (Nat.cast_inj.mp h_code_R)
  have h_feat_mul : feature (sL * sR) = feature (x * y) :=
    h_feature_congr sL sR x y h_feat_L h_feat_R
  have h_code_mul :
      encodedOracle (Strings := Strings) feature (sL * sR) =
      encodedOracle (Strings := Strings) feature (x * y) := by
    exact congrArg (fun t : Feature => (Encodable.encode t : ℝ)) h_feat_mul
  exact dist_eq_zero.mpr h_code_mul

/-- If a feature map is congruent under monoid multiplication, then the paired
non-identity operator is merge-compatible for the corresponding encoded oracle. -/
theorem pairedSketch_merge_compatible_of_feature_congruent
    {Feature : Type*} [Encodable Feature]
    (feature : Strings → Feature)
    (h_feature_congr :
      ∀ sL sR x y,
        feature sL = feature x →
        feature sR = feature y →
        feature (sL * sR) = feature (x * y)) :
    SketchMergeCompatible (pairedSketchOperator (Strings := Strings))
      (encodedOracle (Strings := Strings) feature) := by
  intro sL sR x y hx hy
  have h_code_L :
      encodedOracle (Strings := Strings) feature sL.1 =
      encodedOracle (Strings := Strings) feature x := by
    exact dist_eq_zero.mp (by simpa [pairedSketchOperator] using hx)
  have h_code_R :
      encodedOracle (Strings := Strings) feature sR.1 =
      encodedOracle (Strings := Strings) feature y := by
    exact dist_eq_zero.mp (by simpa [pairedSketchOperator] using hy)
  have h_feat_L : feature sL.1 = feature x := by
    exact Encodable.encode_injective (Nat.cast_inj.mp h_code_L)
  have h_feat_R : feature sR.1 = feature y := by
    exact Encodable.encode_injective (Nat.cast_inj.mp h_code_R)
  have h_feat_mul : feature (sL.1 * sR.1) = feature (x * y) :=
    h_feature_congr sL.1 sR.1 x y h_feat_L h_feat_R
  have h_code_mul :
      encodedOracle (Strings := Strings) feature (sL.1 * sR.1) =
      encodedOracle (Strings := Strings) feature (x * y) := by
    exact congrArg (fun t : Feature => (Encodable.encode t : ℝ)) h_feat_mul
  exact dist_eq_zero.mpr (by simpa [pairedSketchOperator] using h_code_mul)

/-- Sketch compositional assumptions imply tree-level oracle preservation of decoded
sketch summaries. -/
theorem sketch_tree_preserving
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar) :
    ∀ T, dist (fstar (sketchSummary op T)) (fstar (S T)) = 0 := by
  intro T
  induction T with
  | leaf b =>
      simpa [sketchSummary, sketchReduce, S, SketchLeafPreserving, summaryFromSketch] using
        h_leaf b
  | node TL TR ihL ihR =>
      simpa [sketchSummary, sketchReduce, S] using
        h_merge (sketchReduce op TL) (sketchReduce op TR) (S TL) (S TR) ihL ihR

/-- Under `SketchSummaryCompatible`, decoded sketch reduction equals deterministic
reduction under `summaryFromSketch`. -/
theorem sketchSummary_eq_reduceDeterministic
    (op : SketchOperator Strings Sketch)
    (h_compat : SketchSummaryCompatible op) :
    ∀ T, sketchSummary op T = reduceDeterministic (summaryFromSketch op) T := by
  intro T
  induction T with
  | leaf b =>
      rfl
  | node TL TR ihL ihR =>
      calc
        sketchSummary op (BinTree.node TL TR)
            = op.decode (op.merge (sketchReduce op TL) (sketchReduce op TR)) := by
                rfl
        _ = summaryFromSketch op
              (op.decode (sketchReduce op TL) * op.decode (sketchReduce op TR)) := by
              simpa using h_compat (sketchReduce op TL) (sketchReduce op TR)
        _ = summaryFromSketch op (sketchSummary op TL * sketchSummary op TR) := by
              rfl
        _ = summaryFromSketch op
              (reduceDeterministic (summaryFromSketch op) TL *
                reduceDeterministic (summaryFromSketch op) TR) := by
              simp [ihL, ihR]
        _ = reduceDeterministic (summaryFromSketch op) (BinTree.node TL TR) := by
              rfl

/-- Sketch assumptions imply treewise preservation for the deterministic summary
operator induced by one-step encode/decode. -/
theorem treewise_preserving_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op) :
    TreewisePreserving (summaryFromSketch op) fstar := by
  intro T
  have h_tree :=
    sketch_tree_preserving (op := op) (fstar := fstar) h_leaf h_merge T
  have h_eq :=
    sketchSummary_eq_reduceDeterministic (op := op) h_compat T
  simpa [h_eq] using h_tree

/-- Main bridge theorem: sketch-level assumptions produce a full local-law bundle
for the deterministic summarizer induced by one-step encode/decode. -/
theorem local_laws_bundle_of_sketch
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (T : BinTree Strings) :
    LocalLawsBundle (deterministicSummarizer (summaryFromSketch op)) T fstar := by
  refine ⟨?_, ?_, ?_⟩
  · exact L1_of_pointwise (s := summaryFromSketch op) (fstar := fstar) (T := T) h_leaf
  · exact L2_of_treewise (s := summaryFromSketch op) (fstar := fstar) (T := T)
      (treewise_preserving_of_sketch
        (op := op) (fstar := fstar) h_leaf h_merge h_compat)
  · exact L3_of_pointwise (s := summaryFromSketch op) (fstar := fstar) h_leaf

end
