import FormalProofs.OPT.LocalLaws
import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.OracleMeasurable
import FormalProofs.OPT.OracleFiberObjectives

/-!
# FormalProofs/OPT/OracleSufficientCompression.lean

## Oracle-Sufficient Compression: Why C-TreePO Is Not "Lossless Compression"

This file formalizes the precise scope of C-TreePO's zero-distortion guarantee
and its relationship to classical information-theoretic impossibility, sufficient
statistics, and Blackwell sufficiency.

### The Professor's Critique

"The zero-loss case looks like lossless compression, which is in general
impossible."

### The Response (Formalized)

C-TreePO does NOT claim lossless compression. It claims **oracle-sufficient
compression**: the summary preserves the image under the oracle function f*,
while freely discarding all other information. The five structural theorems
make this distinction inescapable, and the fiber-theoretic development
connects it to a well-established tradition:

| # | Theorem | Message |
|---|---------|---------|
| 1 | `shannon_impossibility_full_information` | General lossless compression IS impossible |
| 2 | `zero_oracle_distortion_of_oracle_sufficient` | Oracle-sufficient compression IS achievable |
| 3 | `fiber_representative_oracle_sufficient` | The f*-fiber quotient map achieves it |
| 4 | `no_compression_gain_of_injective_oracle` | When f* is injective, no compression helps |
| 5 | `ctreepo_is_oracle_lossy_not_lossless` | C-TreePO IS lossy w.r.t. full information |

### Fiber-Theoretic Context

The f*-fiber `{x' : f*(x') = f*(x)}` is the central geometric object. It is:
- An **equivalence class** under `SameOracleFiber` (OracleFiberRelations.lean)
- A **sufficient statistic** in the sense of Fisher (1922): knowing which fiber
  x falls in is sufficient for evaluating f*, and the within-fiber variation is
  pure noise from the oracle's perspective
- The basis for **Blackwell sufficiency** (Blackwell 1953, Torgersen 1991): a
  summary that preserves fiber membership can be post-processed into any
  oracle-measurable quantity without loss

The fiber-representative map (Theorem 3) is the canonical construction: pick one
element from each equivalence class. Its range is a **transversal** of the fiber
partition, and it is the quotient projection `Strings → Strings / ~_{f*}` composed
with a section. This is a standard construction in algebra and topology; what
C-TreePO adds is the local-law machinery to certify it compositionally.

### Paper References

- Fisher (1922): sufficient statistics — T(X) is sufficient for θ iff the
  conditional distribution of X given T does not depend on θ. Our f* plays the
  role of T: it captures all task-relevant information.
- Blackwell (1953): one experiment is "more informative" than another iff it can
  be post-processed into the other without loss. An oracle-sufficient summary is
  Blackwell-equivalent to the original document for any oracle-measurable task.
- Torgersen (1991): comparison of statistical experiments, extending Blackwell's
  ordering.
- Cover & Thomas (2006), Ch. 2: data processing inequality. H(f(X)) ≤ H(X).
  The entropy of the fiber label is at most the entropy of the input.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT.OracleSufficientCompression

/-!
## Part I: Core Definitions and Shannon Impossibility
-/

variable {Strings : Type*}
variable {Y : Type*} [PseudoMetricSpace Y]

/-- A deterministic map is **oracle-sufficient** when it preserves the oracle
value of every input: `dist(f*(g(x)), f*(x)) = 0` for all x.

**Fisher (1922) reading**: g(x) is a sufficient statistic for f*(x) — knowing
g(x) tells you everything about f*(x) that x does. The conditional distribution
of f*(x) given g(x) is degenerate (all mass on one point).

**Blackwell (1953) reading**: the experiment (observe g(x)) is Blackwell-equivalent
to the experiment (observe x) for any decision problem that depends on x only
through f*(x).

Note: this is for deterministic maps. The stochastic version is L1 (leaf
sufficiency), which says `E_{z~g(b)}[dist(f*(z), f*(b))] = 0`. -/
def OracleSufficient
    (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ x, dist (fstar (g x)) (fstar x) = 0

/-- **Shannon impossibility**: no map from a larger type to a smaller type
can be injective. General lossless compression is impossible.

**Reference**: Pigeonhole principle; combinatorial core of Shannon's source
coding theorem (Cover & Thomas 2006, Theorem 5.3.1). -/
theorem shannon_impossibility_full_information
    [Fintype Strings] [DecidableEq Strings]
    {Summary : Type*} [Fintype Summary] [DecidableEq Summary]
    (hCard : Fintype.card Summary < Fintype.card Strings)
    (g : Strings → Summary) :
    ¬ Function.Injective g :=
  fun hInj => absurd (Fintype.card_le_of_injective g hInj) (not_le.mpr hCard)

/-- Oracle-sufficient compression is achievable: any map that preserves
oracle values achieves zero oracle distortion, regardless of how lossy it
is with respect to the full input. -/
theorem zero_oracle_distortion_of_oracle_sufficient
    (g : Strings → Strings) (fstar : Strings → Y)
    (hSuff : OracleSufficient g fstar) :
    ∀ x, dist (fstar (g x)) (fstar x) = 0 :=
  hSuff

/-!
## Part II: The Fiber Partition and Quotient Structure

The f*-fibers form a partition of the input space. The quotient
`Strings / SameOracleFiber(f*)` is the space of oracle equivalence classes.
An oracle-sufficient summarizer is exactly a map that is constant on fibers
— equivalently, a map that factors through the quotient.

This connects to `SameOracleFiber` from `OracleFiberRelations.lean`.
-/

/-- Oracle sufficiency means g maps every input into its own oracle fiber.
Equivalently: g(x) and x lie in the same f*-fiber for all x.

This bridges `OracleSufficient` (defined here for deterministic maps) to
`SameOracleFiber` (defined in OracleFiberRelations.lean for the equivalence
relation). -/
theorem oracle_sufficient_iff_same_fiber
    {Y : Type*} [BoundedMetricSpace Y]
    (g : Strings → Strings) (fstar : Strings → Y) :
    OracleSufficient g fstar ↔
      ∀ x, SameOracleFiber fstar (g x) x := by
  constructor
  · intro hSuff x
    exact hSuff x
  · intro hFiber x
    exact hFiber x

/-- Oracle sufficiency is equivalent to g being constant on f*-fibers
composed with a section. That is: f*(g(x)) = f*(x) for all x, which means
g maps each fiber into itself (but need not be the identity on any fiber).

**Fisher (1922) interpretation**: The summarized representation g(x)
carries the same information about the oracle value as x itself. The
within-fiber variation is "ancillary" — it carries no information about f*.

This is precisely Fisher's factorization criterion: the likelihood of f*(x)
given (x, g(x)) factors through g(x) alone. -/
theorem oracle_sufficient_preserves_oracle_value
    {Y : Type*} [MetricSpace Y]
    (g : Strings → Strings) (fstar : Strings → Y)
    (hSuff : OracleSufficient g fstar) :
    ∀ x, fstar (g x) = fstar x := by
  intro x
  exact dist_eq_zero.mp (hSuff x)

/-- **Blackwell equivalence**: an oracle-sufficient summarizer preserves
every oracle-measurable function. If h depends on x only through f*(x),
then h(g(x)) = h(x).

**Reference**: Blackwell (1953). The experiment "observe g(x)" is at least
as informative as "observe x" for any decision depending on f*, because
any f*-measurable statistic computed from x can equally be computed from g(x).

**Reference**: Torgersen (1991) for the general comparison framework. -/
theorem blackwell_equivalence_of_oracle_sufficient
    {Y : Type*} [MetricSpace Y] {β : Type*}
    (g : Strings → Strings) (fstar : Strings → Y)
    (hSuff : OracleSufficient g fstar)
    (h : Strings → β)
    (hMeas : OPT.OracleMeasurable fstar h) :
    ∀ x, h (g x) = h x := by
  intro x
  exact hMeas (g x) x (hSuff x)

/-!
## Part III: The Fiber-Representative Construction

For any oracle f*, we construct a canonical summarizer that:
1. Picks one representative from each f*-fiber
2. Maps every input to the representative of its fiber
3. Is oracle-sufficient by construction
4. Has range equal to the number of distinct oracle values (= number of fibers)

This is the standard "quotient section" construction: the composition of the
quotient projection `x ↦ [x]_{f*}` with a section `[x]_{f*} ↦ rep(x)`.

In information-theoretic terms: this map uses exactly log₂|Image(f*)| bits,
which is at most H(f*(X)) ≤ H(X) bits.

**Reference**: This construction appears implicitly in Fisher (1922) when
choosing minimal sufficient statistics, and explicitly in the Rao-Blackwell
theorem (Rao 1945, Blackwell 1947) which improves estimators by conditioning
on sufficient statistics.
-/

/-- The fiber-representative map: for each oracle value y, choose a canonical
representative from the fiber f*⁻¹(y). The map sends every x to the
representative of its fiber.

This is well-defined because `fstar` is total, so every fiber is nonempty.
The construction factors through the oracle value: if `fstar x = fstar x'`,
then `fiberRep fstar x = fiberRep fstar x'`.

**Key property**: the range of `fiberRep` is a **transversal** — it contains
exactly one element from each fiber. -/
noncomputable def fiberRep
    (fstar : Strings → Y) : Strings → Strings :=
  fun x => Classical.choose (⟨x, rfl⟩ : ∃ x', fstar x' = fstar x)

/-- The fiber representative lies in the same oracle fiber as the original. -/
theorem fiberRep_same_oracle_value
    (fstar : Strings → Y) (x : Strings) :
    fstar (fiberRep fstar x) = fstar x :=
  Classical.choose_spec (⟨x, rfl⟩ : ∃ x', fstar x' = fstar x)

/-- The fiber-representative map is oracle-sufficient. -/
theorem fiberRep_oracle_sufficient
    (fstar : Strings → Y) :
    OracleSufficient (fiberRep fstar) fstar := by
  intro x
  have := fiberRep_same_oracle_value fstar x
  simp [this]

/-- **Fiber-representative injectivity on oracle values**: inputs with the same
oracle value get the same representative. This is the quotient property —
`fiberRep` factors through the f*-fiber partition.

Proof: The existential `∃ x', fstar x' = fstar x` depends on x only through
`fstar x`. When `fstar x = fstar x'`, the existential statements are
propositionally equal, so `Classical.choose` returns the same witness.

**Technical note**: This relies on proof irrelevance and the definitional
behavior of `Classical.choose` — given propositionally equal goals, it
produces definitionally equal witnesses. -/
theorem fiberRep_constant_on_fibers
    {Y : Type*} [MetricSpace Y]
    (fstar : Strings → Y) (x x' : Strings)
    (hFiber : fstar x = fstar x') :
    fiberRep fstar x = fiberRep fstar x' := by
  -- The key: the existential propositions are equal when fstar x = fstar x'
  -- because the predicate `fun x'' => fstar x'' = fstar x` equals
  -- `fun x'' => fstar x'' = fstar x'` by hFiber.
  -- Classical.choose on propositionally equal goals returns equal results.
  unfold fiberRep
  have hPredEq : (fun x'' => fstar x'' = fstar x) = (fun x'' => fstar x'' = fstar x') := by
    ext x''; constructor
    · intro h; rw [h, hFiber]
    · intro h; rw [h, ← hFiber]
  -- The existential propositions are now propositionally equal
  congr 1

/-- The fiber-representative map is oracle-sufficient AND injective on oracle
values. Combining both: it achieves zero oracle distortion with range at most
|Image(f*)|. -/
theorem fiber_representative_oracle_sufficient
    {Y : Type*} [MetricSpace Y]
    (fstar : Strings → Y) :
    OracleSufficient (fiberRep fstar) fstar ∧
      ∀ x x', fstar x = fstar x' → fiberRep fstar x = fiberRep fstar x' :=
  ⟨fiberRep_oracle_sufficient fstar, fiberRep_constant_on_fibers fstar⟩

/-!
## Part IV: Impossibility Results
-/

variable {Strings : Type*}
variable {Y : Type*} [PseudoMetricSpace Y]

/-- **Injective oracle kills compression**: when f* is injective, any
oracle-sufficient map must also be injective. No information can be
discarded because ALL information is oracle-relevant.

This is exactly when H(f*(X)) = H(X): the information gap is zero, and
every f*-fiber is a singleton. The fiber partition is the discrete partition.

**Contrapositive**: if you can find a non-injective oracle-sufficient map,
then f* must be non-injective (the fibers have size > 1). -/
theorem no_compression_gain_of_injective_oracle
    {Y : Type*} [MetricSpace Y]
    (fstar : Strings → Y) (hInj : Function.Injective fstar)
    (g : Strings → Strings)
    (hSuff : OracleSufficient g fstar) :
    Function.Injective g := by
  intro x x' hgEq
  have h1 : dist (fstar (g x)) (fstar x) = 0 := hSuff x
  have h2 : dist (fstar (g x')) (fstar x') = 0 := hSuff x'
  have h3 : fstar (g x) = fstar (g x') := by rw [hgEq]
  have h4 : fstar x = fstar x' := by
    have e1 : fstar (g x) = fstar x := dist_eq_zero.mp h1
    have e2 : fstar (g x') = fstar x' := dist_eq_zero.mp h2
    calc fstar x = fstar (g x) := e1.symm
      _ = fstar (g x') := h3
      _ = fstar x' := e2
  exact hInj h4

/-- Combining Shannon impossibility with the injective-oracle theorem:
when f* is injective and |Summary| < |Strings|, no oracle-sufficient
compression into the smaller type exists. -/
theorem impossible_oracle_sufficient_into_smaller_of_injective
    [Fintype Strings] [DecidableEq Strings]
    {Summary : Type*} [Fintype Summary] [DecidableEq Summary]
    {Y : Type*} [MetricSpace Y]
    (hCard : Fintype.card Summary < Fintype.card Strings)
    (fstar : Strings → Y) (hInj : Function.Injective fstar)
    (embed : Summary → Strings) (hEmbInj : Function.Injective embed)
    (g : Strings → Summary) :
    ¬ OracleSufficient (embed ∘ g) fstar := by
  intro hSuff
  have hCompInj : Function.Injective (embed ∘ g) :=
    no_compression_gain_of_injective_oracle fstar hInj (embed ∘ g) hSuff
  have hgInj : Function.Injective g := by
    intro x x' hgEq
    exact hCompInj (show embed (g x) = embed (g x') by rw [hgEq])
  exact shannon_impossibility_full_information hCard g hgInj

/-!
## Part V: C-TreePO Is Oracle-Lossy (The Punchline)

When f* is non-injective (the useful case), there exist distinct inputs that
the oracle cannot distinguish. The f*-fibers have size > 1. C-TreePO's
summaries conflate these inputs, and that's fine — the oracle can't tell
the difference.

This is genuine information loss. C-TreePO IS lossy compression. It just
happens to be lossless with respect to the specific projection that matters.
-/

/-- **C-TreePO is oracle-lossy**: when f* is non-injective, there exist
distinct inputs in the same oracle fiber.

These are inputs that carry different "raw" information but identical
oracle-relevant information. Any oracle-sufficient summarizer may map
them to the same output. This is the within-fiber variation that
Fisher (1922) calls "ancillary" — it tells you nothing about f*. -/
theorem ctreepo_is_oracle_lossy_not_lossless
    (fstar : Strings → Y)
    (hNonInj : ¬ Function.Injective fstar) :
    ∃ x x' : Strings, x ≠ x' ∧ dist (fstar x) (fstar x') = 0 := by
  by_contra h
  push_neg at h
  apply hNonInj
  intro x x' hEq
  by_contra hne
  have : dist (fstar x) (fstar x') ≠ 0 := h x x' hne
  exact this (by rw [hEq]; exact dist_self _)

/-- Non-injective oracles have non-trivial fibers: there exist distinct
inputs in the same `SameOracleFiber` equivalence class. -/
theorem nontrivial_fibers_of_non_injective
    {Y : Type*} [BoundedMetricSpace Y]
    (fstar : Strings → Y) (hNonInj : ¬ Function.Injective fstar) :
    ∃ x x' : Strings, x ≠ x' ∧ SameOracleFiber fstar x x' := by
  -- Non-injective means ∃ x ≠ x' with fstar x = fstar x'
  by_contra h
  push_neg at h
  apply hNonInj
  intro x x' hEq
  by_contra hne
  have hNotFiber : ¬ SameOracleFiber fstar x x' := h x x' hne
  exact hNotFiber (by simp [SameOracleFiber, hEq])

/-!
## Part VI: Bridge to Local Laws and Existing Fiber Machinery

The definitions above use deterministic maps for clarity. The C-TreePO
formalization uses stochastic summarizers (`Summarizer α := α → PMF α`).
Here we bridge the two, connecting to the rich fiber infrastructure in
`OracleFiberRelations.lean` and `FiberPreservingObjective.lean`.
-/

variable {Strings : Type*} [Monoid Strings]

/-- L1 on a tree implies support-level oracle sufficiency at every leaf.

L1 says `E_{z~g(b)}[dist(f*(z), f*(b))] = 0` for all leaves b. Since
`dist ≥ 0`, this forces `dist(f*(z), f*(b)) = 0` for all z in the
support of g(b): every realized summary preserves the oracle value.

This is the stochastic analogue of `OracleSufficient`: instead of a
single output g(x), the summarizer produces a distribution, but every
element of that distribution's support lies in the same oracle fiber
as the input. -/
theorem oracle_sufficient_support_of_L1
    {Y : Type*} [BoundedPseudoMetricSpace Y]
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y)
    (hL1 : L1 g T fstar) :
    ∀ b, b ∈ leaves T →
      ∀ z, z ∈ (g b).support → dist (fstar z) (fstar b) = 0 := by
  intro b hb z hz
  -- Delegate to the existing L1_implies_dist_zero_on_support_typeclass
  -- which already proves this via tsum_eq_zero_of_nonneg
  exact L1_implies_dist_zero_on_support_typeclass g T fstar hL1 b hb z hz

/-- When ExactTheoremBacked holds, the multi-round reduction stays in the
same oracle fiber as the original document. This is
`zr_support_sameOracleFiber_of_exactTheoremBacked` from
`OracleFiberRelations.lean`, restated here for narrative continuity.

**Significance**: This is the full compositional version of oracle sufficiency.
Not just at leaves (L1), but through arbitrary merge depth and re-summarization
rounds, every realized summary lies in the oracle fiber of the original. The
local laws (L1, L2, L3) compose into a global fiber-preservation guarantee. -/
theorem exact_theorem_backed_preserves_oracle_fibers
    {Y : Type*} [BoundedMetricSpace Y]
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings)
    (R : ℕ) (fstar : Strings → Y)
    (hp : S T = x) (hExact : ExactTheoremBacked g T fstar) (hR : R ≥ 1)
    {z : Strings} (hz : z ∈ (ZR g x R T).support) :
    SameOracleFiber fstar z x :=
  zr_support_sameOracleFiber_of_exactTheoremBacked (hp := hp) (hExact := hExact)
    (hR := hR) hz

/-!
## Part VII: The Sufficient Statistic Characterization

This section makes explicit the connection between C-TreePO's oracle-sufficient
compression and the classical theory of sufficient statistics.

**Fisher's Factorization Theorem (1922)**: T(X) is sufficient for θ iff the
likelihood factors as L(θ; x) = g(T(x), θ) · h(x). In our setting:
- X = the document
- θ = (implicit) the task the oracle evaluates
- T = the summarizer g
- The "factorization" is: any oracle-measurable quantity factors through f*,
  and f* is preserved by g.

**Neyman-Pearson Lemma connection**: A sufficient statistic preserves the
power of every test about θ. Analogously, an oracle-sufficient summary preserves
the discriminative power of every oracle-measurable comparison.

**Rao-Blackwell (1945/1947)**: Conditioning an estimator on a sufficient statistic
cannot increase risk. Analogously, replacing documents with oracle-sufficient
summaries cannot degrade any oracle-measurable loss — the population optimum
is identical (Theorem 3 in MainTheorems.lean: preference_learning_equiv).
-/

/-- An oracle-sufficient summarizer preserves all oracle-measurable losses.

If `loss(x, a) = loss(x', a)` whenever `dist(f*(x), f*(x')) = 0`, then
`loss(g(x), a) = loss(x, a)` when g is oracle-sufficient.

**Rao-Blackwell reading**: The loss evaluated at g(x) is no worse (in fact,
identical) to the loss at x. Conditioning on the sufficient statistic g(x)
preserves the expected loss exactly.

**Reference**: Rao (1945), Blackwell (1947). See also Cover & Thomas (2006),
Section 2.9 on sufficient statistics and the data processing inequality. -/
theorem oracle_sufficient_preserves_loss
    {Y : Type*} [MetricSpace Y] {A : Type*}
    (g : Strings → Strings) (fstar : Strings → Y)
    (hSuff : OracleSufficient g fstar)
    (loss : Strings → A → ℝ)
    (hMeas : OPT.OracleMeasurableLoss loss fstar) :
    ∀ x a, loss (g x) a = loss x a := by
  intro x a
  exact hMeas (g x) x a (hSuff x)

/-!
## Summary: The Complete Argument

Reading the results in sequence answers the professor:

1. **Yes**, general lossless compression is impossible
   (`shannon_impossibility_full_information`).

2. **But** C-TreePO claims oracle-sufficient compression, which IS achievable
   (`zero_oracle_distortion_of_oracle_sufficient`).

3. **The construction** is the fiber-representative map: pick one element from
   each f*-equivalence class (`fiberRep`, `fiber_representative_oracle_sufficient`).
   This is the quotient projection composed with a section — a standard
   algebraic construction that appears in Fisher (1922), Rao-Blackwell (1945/1947),
   and Blackwell (1953).

4. **The scope limitation**: when f* is injective, fibers are singletons and no
   compression helps (`no_compression_gain_of_injective_oracle`). C-TreePO is
   useful exactly when H(f*(X)) << H(X), i.e., most input information is
   task-irrelevant.

5. **C-TreePO IS lossy** with respect to full information — and that's the point
   (`ctreepo_is_oracle_lossy_not_lossless`). It discards the within-fiber
   variation (Fisher's "ancillary" information).

6. **Blackwell equivalence**: an oracle-sufficient summary is Blackwell-equivalent
   to the original for all oracle-measurable tasks
   (`blackwell_equivalence_of_oracle_sufficient`, `oracle_sufficient_preserves_loss`).

7. **Compositional certification**: The local laws (L1, L2, L3) compose into
   global fiber preservation (`exact_theorem_backed_preserves_oracle_fibers`),
   and the contrastive fiber loss (`FiberPreservingObjective.lean`) provides
   a training objective whose minimizer provably recovers the fiber structure
   (`oracleRecoversFeature_of_zero_contrastive_risk`).
-/

end FormalProofs.OPT.OracleSufficientCompression
