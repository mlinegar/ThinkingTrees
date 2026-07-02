import FormalProofs.ML.TransformerAsNeuralOperator
import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.TheoremBackingAssumptions
import FormalProofs.OPT.ApproximateLocalLaws

/-!
# FormalProofs/OPT/NeuralOperatorSpaces.lean

## Paper Reference: Function-Space Overlap between Neural Operators and
## Mergeable Sketches

This file formalizes the function-space overlap discussed in the paper's
Discussion and in Appendix~I. The ambient class is the space of operators
`Strings → Strings`. Inside this space sit two subclasses:

1. **Neural operators** --- the parametric class that
   \citet{Kovachki23NeuralOperator} equation~(6) describes as the finite
   composition `Q ∘ layer_T ∘ ... ∘ layer_1 ∘ P`. The generic architecture
   and the transformer-attention inclusion surface are formalized in
   `FormalProofs.ML.NeuralOperatorArchitecture` and
   `FormalProofs.ML.TransformerAsNeuralOperator`.

2. **Classical mergeable sketches** --- the structured class captured by
   `SketchOperator` (encode / merge / decode) in
   `SketchSummaryOperators.lean`. The theorem surface is generic: any sketch
   that supplies the `SketchLeafPreserving`, `SketchMergeCompatible`, and
   `SketchSummaryCompatible` witnesses inhabits this class. The current Lean
   instances cover HyperLogLog, Count-Min, KLL/GK-style quantile witnesses,
   and the paper's bigram/Markov-count sketches; Theta/KMV, Bloom, and
   t-digest are examples of the broader systems lane, not claimed as current
   theorem-backed instances here.

Both classes act on the same ambient function-space
(`Strings → Strings`). They are not disjoint: the strict branch of the
paper's Proposition~1 (mechanized as `ops_reduction_to_classical_mergeable`)
proves that any deterministic neural operator satisfying global A1/A2/A3
inhabits the oracle-homomorphism special case. The state-level sketch branch
is represented separately by `MergeableSketchSummaryClass`.

## Paper-facing identifiers

- `NeuralOperator` — the ambient type of string-to-string operators.
- `NeuralOperatorClass` — a named subset of the ambient operator space.
- `CertifiedSubfamily` — intersection of a class with a certified predicate.
- `MergeableNeuralOperator` — the strict oracle-output intersection: neural
  operators that also satisfy A1/A2/A3 and therefore inhabit the
  oracle-homomorphism special case of mergeability.
- `MergeableSketchSummaryClass` — operators induced by an encode/merge/decode
  sketch with exact local-law witnesses.
- `ExactLocalLawNeuralOperators` and `ApproxLocalLawNeuralOperators` — neural
  operators constrained by exact or approximate local-law witnesses.
- `paper_neural_operator_mergeable_overlap_theorem` — the paper-facing
  overlap statement: every `MergeableNeuralOperator` is oracle-level
  mergeable (via the strict branch of Proposition~1).
- `paper_neural_operator_induces_sketch_adapter_theorem` — shows the
  same overlap from the `SketchOperator` side: every
  `MergeableNeuralOperator` induces a `SketchOperator` whose decode
  agrees with the neural operator's functional action.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real NNReal

open FormalProofs.OPT

set_option maxHeartbeats 400000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace NeuralOperatorSpaces

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-! ## The ambient function space -/

/-- A neural operator on strings, in the function-space sense of
\citet{Kovachki23NeuralOperator}: a map `Strings → Strings`. The
parametric subclass given by equation~(6) is dense in the continuous
operators between Banach function spaces by Theorem~11 of the same
paper, so at the type level we do not distinguish the parametric form
from the ambient one. -/
abbrev NeuralOperator (Strings : Type*) : Type _ := Strings → Strings

/-- A named subset of the ambient neural-operator function space. -/
abbrev NeuralOperatorClass (Strings : Type*) : Type _ :=
  Set (NeuralOperator Strings)

/-- A certified subfamily is an operator class intersected with a predicate. -/
def CertifiedSubfamily {α : Type*} (C : Set α) (P : α → Prop) : Set α :=
  { x | x ∈ C ∧ P x }

/-- The class of operators induced by a sketch/codec carrying the exact
sketch-local witnesses needed by `local_laws_bundle_of_sketch`. -/
def MergeableSketchSummaryClass
    (fstar : Strings → Y) : Set (NeuralOperator Strings) :=
  { g | ∃ Sketch : Type, ∃ op : SketchOperator Strings Sketch,
      summaryFromSketch op = g
        ∧ SketchLeafPreserving op fstar
        ∧ SketchMergeCompatible op fstar
        ∧ SketchSummaryCompatible op }

/-- Intersection of a chosen neural-operator class with the exact
mergeable-sketch summary class. -/
def NeuralOperatorMergeableSketchOverlap
    (C : NeuralOperatorClass Strings) (fstar : Strings → Y) :
    Set (NeuralOperator Strings) :=
  CertifiedSubfamily C (fun g => g ∈ MergeableSketchSummaryClass fstar)


/-! ## The intersection: mergeable neural operators -/

/-- A neural operator that simultaneously satisfies the deterministic global
identities A1/A2/A3 of Proposition~1. By
`ops_reduction_to_classical_mergeable`, any such operator is in the strict
oracle-homomorphism mergeable limit. Classical state-level sketch membership is
tracked separately through `MergeableSketchSummaryClass`. -/
structure MergeableNeuralOperator
    (Strings : Type*) [Monoid Strings]
    (Y : Type*) [PseudoMetricSpace Y]
    (fstar : Strings → Y) where
  /-- The underlying operator, viewed as a map `Strings → Strings`. -/
  operator : NeuralOperator Strings
  /-- A1: the operator preserves the oracle pointwise. -/
  hA1 : A1_global operator fstar
  /-- A2: merge-route equivalence at oracle level. -/
  hA2 : A2_global operator fstar
  /-- A3: an oracle-level merge operator exists. -/
  hA3 : A3_global operator fstar


/-! ## Paper-facing overlap theorems -/

/-- **Paper-facing theorem (overlap via strict Proposition 1 branch).** Every
`MergeableNeuralOperator` is oracle-level mergeable. This is the
function-space-operator form of the strict reduction: when a neural operator
satisfies global A1/A2/A3, oracle values themselves compose. -/
theorem paper_neural_operator_mergeable_overlap_theorem
    (fstar : Strings → Y) (g : MergeableNeuralOperator Strings Y fstar) :
    IsMergeableSummary g.operator fstar :=
  ops_reduction_to_classical_mergeable g.operator fstar g.hA1 g.hA2 g.hA3

/-- **Paper-facing theorem (overlap via SketchOperator).** A
`MergeableNeuralOperator` induces a `SketchOperator` whose encoder is the
operator itself, whose merge is $(s,t) \mapsto g(s \cdot t)$, and whose
decoder is the identity. This is the encode/merge/decode adapter used
throughout the classical-sketch literature, instantiated at the neural
operator. -/
def paper_neural_operator_induces_sketch_adapter
    (fstar : Strings → Y) (g : MergeableNeuralOperator Strings Y fstar) :
    SketchOperator Strings Strings where
  encode := g.operator
  merge := fun s t => g.operator (s * t)
  decode := id

/-- The span-level summary of the induced sketch agrees with the neural
operator's own action, `summaryFromSketch = g.operator`. -/
theorem paper_neural_operator_induces_sketch_adapter_decode_eq
    (fstar : Strings → Y) (g : MergeableNeuralOperator Strings Y fstar) :
    summaryFromSketch (paper_neural_operator_induces_sketch_adapter fstar g)
      = g.operator := by
  funext x
  simp [summaryFromSketch, paper_neural_operator_induces_sketch_adapter]


/-! ## Convenience: building MergeableNeuralOperator from a GlobalPreservation bundle -/

/-- If an operator satisfies the `GlobalPreservation` typeclass defined in
`GlobalAssumptions.lean`, it is a `MergeableNeuralOperator`. This gives a
one-line bridge from any A1/A2/A3 instance to the function-space overlap
theorems above. -/
def ofGlobalPreservation
    (fstar : Strings → Y)
    (g : Strings → Strings)
    [ga : GlobalPreservation g fstar] :
    MergeableNeuralOperator Strings Y fstar where
  operator := g
  hA1 := ga.a1
  hA2 := ga.a2
  hA3 := ga.a3


/-! ## The local-law subspace of neural operators

Inside the ambient `NeuralOperator` space we single out the operators
that obey the three local laws on a fixed tree `T`. This subset is
strictly larger than the `MergeableNeuralOperator` intersection (which
requires global A1/A2/A3), and strictly smaller than the ambient space
(which imposes no correctness at all). The subspace has an exact
version (no tolerance) and an approximate version (local-law budgets
$\varepsilon_{\rm leaf}, \varepsilon_{\rm merge}, \varepsilon_{\rm idemp}$).

In the paper's Discussion, training a learned operator is framed as a
two-term balance: an oracle-approximation loss plus a projection penalty
into this subspace, mixed by a local-law weight $\lambda$ and per-law
relative weights $\rho_i$. The objects below formalize that decomposition.
-/

/-- The exact local-law subspace on tree `T` for oracle `fstar`: the set
of deterministic neural operators whose induced summarizer carries an
`ExactTheoremBacked` witness on `T`. -/
def ExactLocalLawSubspace
    (fstar : Strings → Y) (T : BinTree Strings) :
    Set (NeuralOperator Strings) :=
  { g | Nonempty (ExactTheoremBacked (deterministicSummarizer g) T fstar) }

/-- The approximate local-law subspace on tree `T` parameterized by
per-law budgets. Budgets must be non-negative; exact `ExactLocalLawSubspace`
is the $(0, 0, 0)$ limit. -/
def ApproxLocalLawSubspace
    (fstar : Strings → Y) (T : BinTree Strings)
    (εLeaf εMerge εIdemp : ℝ) :
    Set (NeuralOperator Strings) :=
  { g | Nonempty (ApproxTheoremBacked (deterministicSummarizer g) T fstar) ∧
        ∃ bundle : ApproxLocalLawsBundle (deterministicSummarizer g) T fstar,
          bundle.epsLeaf ≤ εLeaf ∧ bundle.epsMerge ≤ εMerge ∧ bundle.epsIdemp ≤ εIdemp }

/-- A neural-operator class projected into the exact local-law subspace. -/
def ExactLocalLawNeuralOperators
    (C : NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings) :
    Set (NeuralOperator Strings) :=
  CertifiedSubfamily C (fun g => g ∈ ExactLocalLawSubspace fstar T)

/-- A neural-operator class projected into an approximate local-law subspace. -/
def ApproxLocalLawNeuralOperators
    (C : NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings)
    (εLeaf εMerge εIdemp : ℝ) :
    Set (NeuralOperator Strings) :=
  CertifiedSubfamily C
    (fun g => g ∈ ApproxLocalLawSubspace fstar T εLeaf εMerge εIdemp)

/-- Alias for the exact local-law overlap between a chosen neural-operator
class and the C1/C2/C3 theorem-backed subspace. -/
def NeuralOperatorExactLocalLawOverlap
    (C : NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings) :
    Set (NeuralOperator Strings) :=
  ExactLocalLawNeuralOperators C fstar T

/-- Membership in the approximate subspace is monotone in the budgets:
enlarging any budget only enlarges the subspace. -/
theorem approxLocalLawSubspace_mono
    (fstar : Strings → Y) (T : BinTree Strings)
    {εLeaf₁ εMerge₁ εIdemp₁ εLeaf₂ εMerge₂ εIdemp₂ : ℝ}
    (hL : εLeaf₁ ≤ εLeaf₂) (hM : εMerge₁ ≤ εMerge₂) (hI : εIdemp₁ ≤ εIdemp₂) :
    ApproxLocalLawSubspace fstar T εLeaf₁ εMerge₁ εIdemp₁
      ⊆ ApproxLocalLawSubspace fstar T εLeaf₂ εMerge₂ εIdemp₂ := by
  intro g hg
  refine ⟨hg.1, ?_⟩
  obtain ⟨bundle, hBL, hBM, hBI⟩ := hg.2
  exact ⟨bundle, le_trans hBL hL, le_trans hBM hM, le_trans hBI hI⟩

/-- Sketch-induced operators with exact sketch-local witnesses sit inside the
exact local-law neural-operator subspace on every tree. -/
theorem mergeableSketchSummaryClass_subset_exactLocalLawSubspace
    (fstar : Strings → Y) (T : BinTree Strings) :
    MergeableSketchSummaryClass fstar ⊆ ExactLocalLawSubspace fstar T := by
  intro g hg
  rcases hg with ⟨Sketch, op, hSummary, hLeaf, hMerge, hCompat⟩
  subst g
  exact ⟨ExactTheoremBacked.ofLocalLaws
    (local_laws_bundle_of_sketch
      (op := op) (fstar := fstar) hLeaf hMerge hCompat T)⟩

/-- Intersection form: if an operator is in a neural-operator class and is also
induced by an exact mergeable sketch, then it belongs to the exact local-law
projection of that neural-operator class. -/
theorem mergeableSketch_overlap_subset_exactLocalLawNeuralOperators
    (C : NeuralOperatorClass Strings)
    (fstar : Strings → Y) (T : BinTree Strings) :
    NeuralOperatorMergeableSketchOverlap C fstar
      ⊆ ExactLocalLawNeuralOperators C fstar T := by
  intro g hg
  exact ⟨hg.1,
    (mergeableSketchSummaryClass_subset_exactLocalLawSubspace
      (fstar := fstar) (T := T)) hg.2⟩


/-! ## Oracle-plus-projection decomposition

The training objective the paper actually uses takes the form
$(1-\lambda)\cdot \ell_{\rm root} + \lambda \cdot \sum_i \rho_i \ell_{C_i} / \sum_i \rho_i$.
We package the two terms as paper-facing objects so the interpretation
"balance between oracle fit and projection into the local-law subspace"
is a type-level statement rather than prose. -/

/-- A weighted decomposition of a training objective into an
oracle-approximation term and a local-law projection penalty. The
$\lambda$ knob trades between them, and the $\rho_i$ knobs weight the
three laws against each other. -/
structure OraclePlusProjection
    (Strings : Type*) (Y : Type*) where
  /-- Oracle approximation loss: typically the root MSE against the
  document-level oracle value. -/
  oracleLoss : (Strings → Strings) → ℝ
  /-- Projection penalty: distance (in aggregated C1/C2/C3-violation sense)
  from the local-law subspace. -/
  projectionPenalty : (Strings → Strings) → ℝ
  /-- Mixing weight $\lambda \in [0, 1]$. -/
  lam : ℝ
  lam_nonneg : 0 ≤ lam
  lam_le_one : lam ≤ 1

namespace OraclePlusProjection

variable {Strings : Type*} {Y : Type*}

/-- The balanced objective: $(1-\lambda) \cdot \text{oracle} + \lambda \cdot \text{projection}$. -/
def balancedObjective (obj : OraclePlusProjection Strings Y)
    (g : Strings → Strings) : ℝ :=
  (1 - obj.lam) * obj.oracleLoss g + obj.lam * obj.projectionPenalty g

/-- At $\lambda = 0$, the objective reduces to the pure oracle loss. -/
theorem balanced_objective_lam_zero
    (obj : OraclePlusProjection Strings Y) (g : Strings → Strings)
    (h : obj.lam = 0) :
    balancedObjective obj g = obj.oracleLoss g := by
  simp [balancedObjective, h]

/-- At $\lambda = 1$, the objective reduces to the pure projection penalty. -/
theorem balanced_objective_lam_one
    (obj : OraclePlusProjection Strings Y) (g : Strings → Strings)
    (h : obj.lam = 1) :
    balancedObjective obj g = obj.projectionPenalty g := by
  simp [balancedObjective, h]

/-- Non-negativity: if both loss components are non-negative, so is the
balanced objective. -/
theorem balanced_objective_nonneg
    (obj : OraclePlusProjection Strings Y) (g : Strings → Strings)
    (hOracle : 0 ≤ obj.oracleLoss g) (hProj : 0 ≤ obj.projectionPenalty g) :
    0 ≤ balancedObjective obj g := by
  unfold balancedObjective
  have h1 : 0 ≤ (1 - obj.lam) * obj.oracleLoss g :=
    mul_nonneg (by linarith [obj.lam_le_one]) hOracle
  have h2 : 0 ≤ obj.lam * obj.projectionPenalty g :=
    mul_nonneg obj.lam_nonneg hProj
  linarith

/-- Monotonicity in $\lambda$ when the projection penalty is at least the
oracle loss: increasing $\lambda$ weights the projection term more, which
increases the balanced objective if the projection term is larger. -/
theorem balanced_objective_lam_monotone
    (obj : OraclePlusProjection Strings Y) (g : Strings → Strings)
    (obj' : OraclePlusProjection Strings Y)
    (hOracle : obj.oracleLoss = obj'.oracleLoss)
    (hProj : obj.projectionPenalty = obj'.projectionPenalty)
    (hLam : obj.lam ≤ obj'.lam)
    (hIneq : obj.oracleLoss g ≤ obj.projectionPenalty g) :
    balancedObjective obj g ≤ balancedObjective obj' g := by
  unfold balancedObjective
  rw [hOracle, hProj] at *
  -- (1 - λ) * a + λ * b ≤ (1 - λ') * a + λ' * b when a ≤ b and λ ≤ λ'
  have ha : obj'.oracleLoss g ≤ obj'.projectionPenalty g := by
    rw [← hOracle, ← hProj]; exact hIneq
  have : (obj'.lam - obj.lam) * (obj'.projectionPenalty g - obj'.oracleLoss g) ≥ 0 :=
    mul_nonneg (by linarith) (by linarith)
  linarith

end OraclePlusProjection


/-! ## The projection interpretation

The "local-law weights" in training are, operationally, nothing more
elaborate than a projection penalty: they push the learned operator toward
the subset of neural operators that satisfy the local laws. This subsection
makes that interpretation a Lean theorem. Under a faithful projection
penalty --- one whose zero set coincides with the exact local-law subspace
--- the zero-penalty operators within any neural-operator class $C$ are
*exactly* the learnable local-law subspace $C \cap \text{ExactLocalLawSubspace}$.
Nothing exotic is happening: we are reweighting the training loss so that
its optima land in the intersection of "what the class can represent" and
"what the local laws allow." -/

/-- The set of operators on which a given `OraclePlusProjection`'s projection
penalty vanishes. Faithful penalties identify this set with the local-law
subspace; the theorems below make that identification precise. -/
def zeroProjectionSet
    (obj : OraclePlusProjection Strings Y) :
    Set (Strings → Strings) :=
  { g | obj.projectionPenalty g = 0 }

/-- Statement that the local-law weights really are a projection: the zero set
of the projection penalty is exactly the exact local-law subspace. -/
def LocalLawWeightsAreProjection
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) : Prop :=
  zeroProjectionSet obj = ExactLocalLawSubspace fstar T

/-- Class-restricted version: inside a chosen neural-operator class `C`, the
zero-penalty operators are exactly the operators in `C` that satisfy the exact
local laws. -/
def LocalLawWeightsAreProjectionOn
    (C : NeuralOperatorClass Strings)
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) : Prop :=
  C ∩ zeroProjectionSet obj = ExactLocalLawNeuralOperators C fstar T

/-- Statement that the approximation-error structure is exactly local-law
structured: zero projection error is equivalent to membership in the exact
local-law subspace. This is the assumption-side reading of
`LocalLawWeightsAreProjection`. -/
def ApproximationErrorStructuredByLocalLaws
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) : Prop :=
  ∀ g, obj.projectionPenalty g = 0 ↔ g ∈ ExactLocalLawSubspace fstar T

/-- Class-restricted approximation-error structure. On the representable class
`C`, zero approximation/projection error is equivalent to exact local-law
membership. -/
def ApproximationErrorStructuredByLocalLawsOn
    (C : NeuralOperatorClass Strings)
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) : Prop :=
  ∀ g, g ∈ C → (obj.projectionPenalty g = 0 ↔ g ∈ ExactLocalLawSubspace fstar T)

/-- An `OraclePlusProjection` whose projection penalty is *faithful* to the
local-law subspace: the penalty is zero on exactly those operators that lie
in the exact local-law subspace on tree `T` for oracle `fstar`. This is the
property that lets us say the $\lambda = 1$ limit of the balanced objective
"is" the indicator of the local-law subspace. -/
structure FaithfulProjectionPenalty
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) : Prop where
  zero_iff_in_subspace :
    ∀ g, obj.projectionPenalty g = 0 ↔ g ∈ ExactLocalLawSubspace fstar T

/-- The projection interpretation and the structured-approximation-error
assumption are the same statement. -/
theorem localLawWeightsAreProjection_iff_approximationErrorStructuredByLocalLaws
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) :
    LocalLawWeightsAreProjection obj fstar T
      ↔ ApproximationErrorStructuredByLocalLaws obj fstar T := by
  constructor
  · intro h g
    change g ∈ zeroProjectionSet obj ↔ g ∈ ExactLocalLawSubspace fstar T
    rw [h]
  · intro h
    ext g
    change obj.projectionPenalty g = 0 ↔ g ∈ ExactLocalLawSubspace fstar T
    exact h g

/-- Representable-class version of the same equivalence. Within a chosen
neural-operator class, saying "local-law weights are a projection" is
equivalent to assuming the approximation error has exactly the C1/C2/C3
zero set on that class. -/
theorem localLawWeightsAreProjectionOn_iff_approximationErrorStructuredByLocalLawsOn
    (C : NeuralOperatorClass Strings)
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) :
    LocalLawWeightsAreProjectionOn C obj fstar T
      ↔ ApproximationErrorStructuredByLocalLawsOn C obj fstar T := by
  constructor
  · intro h g hgC
    have hmem :
        (g ∈ C ∩ zeroProjectionSet obj)
          ↔ g ∈ ExactLocalLawNeuralOperators C fstar T := by
      rw [h]
    constructor
    · intro hzero
      have hleft : g ∈ C ∩ zeroProjectionSet obj := ⟨hgC, hzero⟩
      exact (hmem.mp hleft).2
    · intro hsub
      have hright : g ∈ ExactLocalLawNeuralOperators C fstar T := ⟨hgC, hsub⟩
      exact (hmem.mpr hright).2
  · intro h
    ext g
    constructor
    · intro hg
      exact ⟨hg.1, (h g hg.1).mp hg.2⟩
    · intro hg
      exact ⟨hg.1, (h g hg.1).mpr hg.2⟩

/-- The old faithful-penalty structure is exactly the structured-error
assumption packaged as a structure. -/
theorem faithfulProjectionPenalty_iff_approximationErrorStructuredByLocalLaws
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings) :
    FaithfulProjectionPenalty obj fstar T
      ↔ ApproximationErrorStructuredByLocalLaws obj fstar T := by
  constructor
  · intro h
    exact h.zero_iff_in_subspace
  · intro h
    exact ⟨h⟩

/-- **Paper-facing equality.** Under faithfulness, the zero set of the
projection penalty equals the exact local-law subspace. This is the
one-sentence content of "local-law weights pick out exactly the
law-abiding operators." -/
theorem paper_zeroProjectionSet_eq_exactLocalLawSubspace
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings)
    (h : FaithfulProjectionPenalty obj fstar T) :
    zeroProjectionSet obj = ExactLocalLawSubspace fstar T := by
  ext g
  exact h.zero_iff_in_subspace g

/-- **Paper-facing simplicity theorem.** Within any neural-operator class
$C$, the zero-penalty operators coincide with the *learnable local-law
subspace* $C \cap \text{ExactLocalLawSubspace}$. This is the formal
statement of the interpretation that training with local-law weights is
"just" projecting the neural-operator class onto the subset where the
local laws hold --- nothing more exotic. -/
theorem paper_projection_eq_learnable_local_law_subspace
    (C : NeuralOperatorClass Strings)
    (obj : OraclePlusProjection Strings Y)
    (fstar : Strings → Y) (T : BinTree Strings)
    (h : FaithfulProjectionPenalty obj fstar T) :
    C ∩ zeroProjectionSet obj = ExactLocalLawNeuralOperators C fstar T := by
  ext g
  have hset : zeroProjectionSet obj = ExactLocalLawSubspace fstar T :=
    paper_zeroProjectionSet_eq_exactLocalLawSubspace obj fstar T h
  simp only [Set.mem_inter_iff, ExactLocalLawNeuralOperators,
             CertifiedSubfamily, Set.mem_setOf_eq,
             show (g ∈ zeroProjectionSet obj) ↔ (g ∈ ExactLocalLawSubspace fstar T) from
               by rw [hset]]


/-! ## Paper-facing overlap+decomposition theorems -/

/-- **Paper-facing theorem.** The local-law subspace is strictly larger
than the `MergeableNeuralOperator` intersection: if a neural operator is
in the latter (A1/A2/A3 hold globally), then on any tree `T` it sits
inside the exact local-law subspace. Concretely, a `MergeableNeuralOperator`
induces an `ExactTheoremBacked` witness on every tree via the
`toLocalLawsBundle` bridge of `GlobalAssumptions.lean`. -/
theorem paper_mergeable_subset_exactLocalLawSubspace
    (fstar : Strings → Y)
    (g : MergeableNeuralOperator Strings Y fstar)
    (T : BinTree Strings) :
    g.operator ∈ ExactLocalLawSubspace fstar T := by
  -- The MergeableNeuralOperator's A1/A2/A3 carries a GlobalPreservation
  -- instance, which via `toLocalLawsBundle` gives an `ExactTheoremBacked`
  -- witness on any tree.
  haveI : GlobalPreservation g.operator fstar :=
    { a1 := g.hA1, a2 := g.hA2, a3 := g.hA3 }
  exact ⟨ExactTheoremBacked.ofLocalLaws
    (GlobalPreservation.toLocalLawsBundle (g_det := g.operator)
      (fstar := fstar) T)⟩

/-- A `MergeableNeuralOperator` that lies in a chosen neural-operator class lies
in the class's exact local-law projection. -/
theorem paper_mergeableNeuralOperator_mem_exactLocalLawNeuralOperators
    (C : NeuralOperatorClass Strings)
    (fstar : Strings → Y)
    (g : MergeableNeuralOperator Strings Y fstar)
    (T : BinTree Strings)
    (hC : g.operator ∈ C) :
    g.operator ∈ ExactLocalLawNeuralOperators C fstar T :=
  ⟨hC, paper_mergeable_subset_exactLocalLawSubspace fstar g T⟩

end NeuralOperatorSpaces

end
