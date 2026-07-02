import FormalProofs.OPT.MarkovCountSketchExample
import FormalProofs.OPT.InformationSufficiency
import FormalProofs.OPT.ContextualQuerySufficiency
import FormalProofs.OPT.SlicedContextualSufficiency
import FormalProofs.OPT.HybridSummarySufficiency
import Mathlib.Logic.Function.Basic

/-!
# FormalProofs/OPT/MarkovSufficiency.lean

Markov-specific sufficiency consequences for the exact changepoint sketch.

The exact theorem-domain state for the Markov changepoint task is the
`MarkovCountSketch` carrying `(count, first, last)`. This file makes precise the
task-facing claim used by the runtime diagnostics:

- if a summary is sufficient for **all two-sided changepoint-count queries**,
  then collisions are impossible on exact sketch states;
- therefore the summary admits a decoder back into the exact sketch; and
- the count-only control is not sufficient in this sense.

This is the formal bridge behind treating decoded `(count, first, last)`
recovery as a sufficiency witness for the Markov simulations.
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

open MarkovCountSketch

variable {n : ℕ}

/-- A summary is Markov-count sufficient when it determines the changepoint
count of every two-sided context query `left * x * right`. This is the
task-facing sufficiency notion for the exact Markov sketch lane. -/
def MarkovCountQuerySufficient
    {Summary : Type*}
    (summary : MarkovCountSketch n → Summary) : Prop :=
  ∀ left right x y,
    summary x = summary y →
      MarkovCountSketch.count (left * x * right) =
        MarkovCountSketch.count (left * y * right)

/-- The Markov-specific task-facing sufficiency definition is exactly the
generic two-sided contextual sufficiency definition specialized to
`fstar = MarkovCountSketch.count`. -/
theorem markovCountQuerySufficient_iff_twoSidedContextSufficient
    {Summary : Type*}
    (summary : MarkovCountSketch n → Summary) :
    MarkovCountQuerySufficient (n := n) summary ↔
      TwoSidedContextSufficient
        summary
        (fun s : MarkovCountSketch n => MarkovCountSketch.count s) := by
  constructor
  · intro hSuff x y hxy ctx
    rcases ctx with ⟨left, right⟩
    exact hSuff left right x y hxy
  · intro hSuff left right x y hxy
    exact hSuff hxy (left, right)

/-- The exact sketch itself is sufficient for all two-sided changepoint-count
queries. -/
theorem exact_markov_sketch_query_sufficient :
    MarkovCountQuerySufficient (n := n) (fun s : MarkovCountSketch n => s) := by
  intro left right x y hxy
  simpa [hxy]

/-- The exact Markov sketch satisfies the generic two-sided contextual
sufficiency condition. This is the validation witness for the general
contextual-sufficiency interface. -/
theorem exact_markov_sketch_twoSidedContextSufficient :
    TwoSidedContextSufficient
      (fun s : MarkovCountSketch n => s)
      (fun s : MarkovCountSketch n => MarkovCountSketch.count s) := by
  exact
    (markovCountQuerySufficient_iff_twoSidedContextSufficient
      (n := n)
      (fun s : MarkovCountSketch n => s)).mp
      exact_markov_sketch_query_sufficient

/-- Count-only readout on the theorem-domain state. -/
def markovCountOnlySummary (s : MarkovCountSketch n) : ℕ :=
  MarkovCountSketch.count s

/-- Endpoint residual for the Markov exact sketch: count-only keeps the
internal changepoint count, while this residual keeps the boundary regimes
needed by two-sided contextual count queries. -/
def markovEndpointResidual (s : MarkovCountSketch n) :
    Option (Fin n × Fin n) :=
  match s with
  | MarkovCountSketch.empty => none
  | MarkovCountSketch.nonempty _ first last => some (first, last)

/-- Makinen-style hybrid Markov summary: a base count statistic plus the
endpoint residual.  This is definitionally a `HybridSummary`. -/
def markovCountEndpointHybrid (s : MarkovCountSketch n) :
    ℕ × Option (Fin n × Fin n) :=
  HybridSummary
    (markovCountOnlySummary (n := n))
    (markovEndpointResidual (n := n))
    s

/-- The count-plus-endpoint hybrid is injective on exact Markov sketch states. -/
theorem markov_countEndpointHybrid_injective :
    Function.Injective (markovCountEndpointHybrid (n := n)) := by
  intro x y hxy
  cases x with
  | empty =>
      cases y with
      | empty =>
          rfl
      | nonempty cy fy ly =>
          have hend := congrArg Prod.snd hxy
          simp [markovCountEndpointHybrid, HybridSummary, markovEndpointResidual] at hend
  | nonempty cx fx lx =>
      cases y with
      | empty =>
          have hend := congrArg Prod.snd hxy
          simp [markovCountEndpointHybrid, HybridSummary, markovEndpointResidual] at hend
      | nonempty cy fy ly =>
          have hc : cx = cy := by
            have hcount := congrArg Prod.fst hxy
            simpa [
              markovCountEndpointHybrid,
              HybridSummary,
              markovCountOnlySummary,
              MarkovCountSketch.count
            ] using hcount
          have hp : (fx, lx) = (fy, ly) := by
            have hend := congrArg Prod.snd hxy
            simpa [
              markovCountEndpointHybrid,
              HybridSummary,
              markovEndpointResidual
            ] using hend
          have hf : fx = fy := congrArg Prod.fst hp
          have hl : lx = ly := congrArg Prod.snd hp
          subst hc
          subst hf
          subst hl
          rfl

/-- The Markov count-plus-endpoint hybrid is sufficient for all two-sided
changepoint-count queries. -/
theorem markov_countEndpointHybrid_query_sufficient :
    MarkovCountQuerySufficient
      (n := n)
      (markovCountEndpointHybrid (n := n)) := by
  intro left right x y hxy
  have hstate : x = y := markov_countEndpointHybrid_injective (n := n) hxy
  subst y
  rfl

/-- The Markov count-plus-endpoint hybrid satisfies the generic two-sided
contextual sufficiency condition. -/
theorem markov_countEndpointHybrid_twoSidedContextSufficient :
    TwoSidedContextSufficient
      (markovCountEndpointHybrid (n := n))
      (fun s : MarkovCountSketch n => MarkovCountSketch.count s) := by
  exact
    (markovCountQuerySufficient_iff_twoSidedContextSufficient
      (n := n)
      (markovCountEndpointHybrid (n := n))).mp
      markov_countEndpointHybrid_query_sufficient

/-- Real-valued changepoint count target, used for metric approximate
sufficiency statements. -/
def markovCountReal (s : MarkovCountSketch n) : ℝ :=
  (MarkovCountSketch.count s : ℝ)

/-- Approximate Markov-count query sufficiency after embedding counts into
`ℝ`: collisions may change any two-sided count query by at most `ε`. -/
def MarkovCountQuerySufficientWithin
    {Summary : Type*}
    (ε : ℝ)
    (summary : MarkovCountSketch n → Summary) : Prop :=
  ∀ left right x y,
    summary x = summary y →
      dist (markovCountReal (left * x * right))
        (markovCountReal (left * y * right)) ≤ ε

/-- The approximate Markov-specific definition is exactly generic two-sided
approximate contextual sufficiency specialized to real-valued changepoint
counts. -/
theorem markovCountQuerySufficientWithin_iff_twoSidedContextSufficientWithin
    {Summary : Type*}
    (ε : ℝ)
    (summary : MarkovCountSketch n → Summary) :
    MarkovCountQuerySufficientWithin (n := n) ε summary ↔
      TwoSidedContextSufficientWithin
        ε
        summary
        (markovCountReal (n := n)) := by
  constructor
  · intro hSuff x y hxy ctx
    rcases ctx with ⟨left, right⟩
    exact hSuff left right x y hxy
  · intro hSuff left right x y hxy
    exact hSuff hxy (left, right)

/-- The exact Markov sketch is zero-slack sufficient for real-valued two-sided
count queries. -/
theorem exact_markov_sketch_twoSidedContextSufficientWithin_zero_real :
    TwoSidedContextSufficientWithin
      0
      (fun s : MarkovCountSketch n => s)
      (markovCountReal (n := n)) := by
  intro x y hxy ctx
  have h : x = y := by
    simpa using hxy
  subst y
  simp

/-- If a composed state/readout pipeline approximates every real-valued
two-sided Markov count query within `ε`, then the learned leaf state is
`2ε`-sufficient for all such queries. -/
theorem markov_composedReadoutWithin_implies_twoSidedContextSufficientWithin_real
    {State : Type*}
    {ε : ℝ}
    {leaf : MarkovCountSketch n → State}
    {merge : State → State → State}
    {readout : State → ℝ}
    (hApprox :
      ∀ left x right,
        dist (readout (merge (merge (leaf left) (leaf x)) (leaf right)))
             (markovCountReal (n := n) (left * x * right)) ≤ ε) :
    TwoSidedContextSufficientWithin
      (2 * ε)
      leaf
      (markovCountReal (n := n)) := by
  exact
    composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
      (ε := ε)
      (leaf := leaf)
      (merge := merge)
      (readout := readout)
      (fstar := markovCountReal (n := n))
      hApprox

/-- Finite selected-slice approximate control implies approximate Markov-count
query sufficiency when the selected slices cover the real-valued two-sided
count response family. -/
theorem markov_finiteSlicedWithin_implies_countQuerySufficientWithin
    {Summary Slice SliceVal : Type*}
    [PseudoMetricSpace SliceVal]
    {selected : Finset Slice}
    {δ ε : ℝ}
    {summary : MarkovCountSketch n → Summary}
    {slice : Slice → ((MarkovCountSketch n × MarkovCountSketch n) → ℝ) → SliceVal}
    (hCover :
      FiniteSlicesCoverResponseFibersWithin
        selected
        δ
        ε
        (TwoSidedContextQuery (markovCountReal (n := n)))
        slice)
    (hSliced :
      SlicedQuerySufficientWithinOn
        selected
        δ
        summary
        (TwoSidedContextQuery (markovCountReal (n := n)))
        slice) :
    MarkovCountQuerySufficientWithin (n := n) ε summary := by
  have hTwoSided :
      TwoSidedContextSufficientWithin
        ε
        summary
        (markovCountReal (n := n)) :=
    finiteSlicedWithin_implies_querySufficientWithin hCover hSliced
  exact
    (markovCountQuerySufficientWithin_iff_twoSidedContextSufficientWithin
      (n := n) ε summary).mpr hTwoSided

/-- In an alphabet with at least two regimes, there is always a probe regime
different from a given regime. -/
lemma exists_distinct_regime
    (hn : 1 < n)
    (f : Fin n) :
    ∃ b : Fin n, b ≠ f := by
  by_cases h0 : f.val = 0
  · refine ⟨⟨1, hn⟩, ?_⟩
    intro hb
    have hval : (1 : ℕ) = f.val := by
      simpa using congrArg Fin.val hb
    rw [h0] at hval
    norm_num at hval
  · refine ⟨⟨0, lt_trans (by decide : 0 < 1) hn⟩, ?_⟩
    intro hb
    have hval : f.val = 0 := by
      simpa using (congrArg Fin.val hb).symm
    exact h0 hval

/-- Same-summary collisions preserve the raw changepoint count. -/
lemma markov_count_query_collision_preserves_count
    {Summary : Type*}
    {summary : MarkovCountSketch n → Summary}
    (hSuff : MarkovCountQuerySufficient (n := n) summary)
    {x y : MarkovCountSketch n}
    (hxy : summary x = summary y) :
    MarkovCountSketch.count x = MarkovCountSketch.count y := by
  simpa using hSuff 1 1 x y hxy

/-- Same-summary collisions between nonempty exact sketches preserve the first
endpoint. -/
lemma markov_count_query_collision_preserves_first
    {Summary : Type*}
    {summary : MarkovCountSketch n → Summary}
    (hSuff : MarkovCountQuerySufficient (n := n) summary)
    {cx cy : ℕ}
    {fx fy lx ly : Fin n}
    (hxy :
      summary (MarkovCountSketch.nonempty cx fx lx) =
        summary (MarkovCountSketch.nonempty cy fy ly)) :
    fx = fy := by
  have hcount :
      MarkovCountSketch.count (MarkovCountSketch.nonempty cx fx lx) =
        MarkovCountSketch.count (MarkovCountSketch.nonempty cy fy ly) :=
    markov_count_query_collision_preserves_count (n := n) hSuff hxy
  have hc : cx = cy := by
    simpa [MarkovCountSketch.count] using hcount
  by_cases hff : fx = fy
  · exact hff
  · let probe : MarkovCountSketch n := MarkovCountSketch.nonempty 0 fx fx
    have hprobe := hSuff probe 1
      (MarkovCountSketch.nonempty cx fx lx)
      (MarkovCountSketch.nonempty cy fy ly)
      hxy
    have hbad : cx = cy + 1 := by
      simpa [probe, MarkovCountSketch.join, hff] using hprobe
    have : cx = cx + 1 := by simpa [hc] using hbad
    exact False.elim ((Nat.ne_of_lt (Nat.lt_succ_self cx)) this)

/-- Same-summary collisions between nonempty exact sketches preserve the last
endpoint. -/
lemma markov_count_query_collision_preserves_last
    {Summary : Type*}
    {summary : MarkovCountSketch n → Summary}
    (hSuff : MarkovCountQuerySufficient (n := n) summary)
    {cx cy : ℕ}
    {fx fy lx ly : Fin n}
    (hxy :
      summary (MarkovCountSketch.nonempty cx fx lx) =
        summary (MarkovCountSketch.nonempty cy fy ly)) :
    lx = ly := by
  have hcount :
      MarkovCountSketch.count (MarkovCountSketch.nonempty cx fx lx) =
        MarkovCountSketch.count (MarkovCountSketch.nonempty cy fy ly) :=
    markov_count_query_collision_preserves_count (n := n) hSuff hxy
  have hc : cx = cy := by
    simpa [MarkovCountSketch.count] using hcount
  by_cases hll : lx = ly
  · exact hll
  · let probe : MarkovCountSketch n := MarkovCountSketch.nonempty 0 lx lx
    have hprobe := hSuff 1 probe
      (MarkovCountSketch.nonempty cx fx lx)
      (MarkovCountSketch.nonempty cy fy ly)
      hxy
    have hbad : cx = cy + 1 := by
      simpa [probe, MarkovCountSketch.count, MarkovCountSketch.join, hll, Ne.symm hll] using hprobe
    have : cx = cx + 1 := by simpa [hc] using hbad
    exact False.elim ((Nat.ne_of_lt (Nat.lt_succ_self cx)) this)

/-- Markov-count sufficiency is strong enough to recover the entire exact
sketch state. Any collision would violate some two-sided changepoint-count
query. -/
theorem markov_count_query_sufficient_collision_implies_exact_sketch_eq
    {Summary : Type*}
    (hn : 1 < n)
    {summary : MarkovCountSketch n → Summary}
    (hSuff : MarkovCountQuerySufficient (n := n) summary)
    {x y : MarkovCountSketch n}
    (hxy : summary x = summary y) :
    x = y := by
  cases x with
  | empty =>
      cases y with
      | empty =>
          rfl
      | nonempty cy fy ly =>
          obtain ⟨b, hb⟩ := exists_distinct_regime (n := n) hn fy
          let probe : MarkovCountSketch n := MarkovCountSketch.nonempty 0 b b
          have hprobe := hSuff probe 1 MarkovCountSketch.empty
            (MarkovCountSketch.nonempty cy fy ly) hxy
          have hbad : 0 = cy + 1 := by
            simpa [probe, MarkovCountSketch.count, MarkovCountSketch.join, hb, Ne.symm hb] using hprobe
          exact False.elim (Nat.succ_ne_zero cy hbad.symm)
  | nonempty cx fx lx =>
      cases y with
      | empty =>
          obtain ⟨b, hb⟩ := exists_distinct_regime (n := n) hn fx
          let probe : MarkovCountSketch n := MarkovCountSketch.nonempty 0 b b
          have hprobe := hSuff probe 1
            (MarkovCountSketch.nonempty cx fx lx)
            MarkovCountSketch.empty hxy
          have hbad : cx + 1 = 0 := by
            simpa [probe, MarkovCountSketch.count, MarkovCountSketch.join, hb, Ne.symm hb] using hprobe
          exact False.elim (Nat.succ_ne_zero cx hbad)
      | nonempty cy fy ly =>
          have hcount :
              MarkovCountSketch.count (MarkovCountSketch.nonempty cx fx lx) =
                MarkovCountSketch.count (MarkovCountSketch.nonempty cy fy ly) :=
            markov_count_query_collision_preserves_count (n := n) hSuff hxy
          have hc : cx = cy := by
            simpa [MarkovCountSketch.count] using hcount
          have hf : fx = fy :=
            markov_count_query_collision_preserves_first (n := n) hSuff hxy
          have hl : lx = ly :=
            markov_count_query_collision_preserves_last (n := n) hSuff hxy
          subst hc
          subst hf
          subst hl
          rfl

/-- Hence a Markov-count sufficient summary is injective on the exact sketch
state. -/
theorem markov_count_query_sufficient_injective
    {Summary : Type*}
    (hn : 1 < n)
    {summary : MarkovCountSketch n → Summary}
    (hSuff : MarkovCountQuerySufficient (n := n) summary) :
    Function.Injective summary := by
  intro x y hxy
  exact markov_count_query_sufficient_collision_implies_exact_sketch_eq
    (n := n) hn hSuff hxy

/-- Any Markov-count sufficient summary admits a decoder back to the exact
sketch state. This is the theorem-facing meaning of a successful decoded
`(count, first, last)` sufficiency witness. -/
theorem markov_count_query_sufficient_has_decoder
    {Summary : Type*}
    (hn : 1 < n)
    {summary : MarkovCountSketch n → Summary}
    (hSuff : MarkovCountQuerySufficient (n := n) summary) :
    ∃ recover : Summary → MarkovCountSketch n,
      recover ∘ summary = id := by
  refine ⟨Function.invFun summary, ?_⟩
  funext x
  exact Function.leftInverse_invFun
    (markov_count_query_sufficient_injective (n := n) hn hSuff) x

/-- The count-only statistic is not sufficient for the Markov changepoint task:
two spans with the same internal count but different boundary regimes induce
different two-sided count queries. -/
theorem markov_countOnly_not_query_sufficient
    (hn : 1 < n) :
    ¬ MarkovCountQuerySufficient
      (n := n)
      (Summary := ℕ)
      markovCountOnlySummary := by
  intro hSuff
  let a : Fin n := ⟨0, lt_trans (by decide : 0 < 1) hn⟩
  let b : Fin n := ⟨1, hn⟩
  have hab : a ≠ b := by
    intro hEq
    have : (0 : ℕ) = 1 := by simpa [a, b] using congrArg Fin.val hEq
    norm_num at this
  let left : MarkovCountSketch n := MarkovCountSketch.nonempty 0 a a
  let x : MarkovCountSketch n := MarkovCountSketch.nonempty 0 a a
  let y : MarkovCountSketch n := MarkovCountSketch.nonempty 0 b b
  have hxy : markovCountOnlySummary x = markovCountOnlySummary y := by
    rfl
  have hbad := hSuff left 1 x y hxy
  have : (0 : ℕ) = 1 := by
    simpa [markovCountOnlySummary, left, x, y, MarkovCountSketch.count, MarkovCountSketch.join, hab, Ne.symm hab] using hbad
  norm_num at this

/-- Count-only Markov state also fails the generic two-sided contextual
sufficiency condition. -/
theorem markov_countOnly_not_twoSidedContextSufficient
    (hn : 1 < n) :
    ¬ TwoSidedContextSufficient
      (markovCountOnlySummary (n := n))
      (fun s : MarkovCountSketch n => MarkovCountSketch.count s) := by
  intro hSuff
  exact markov_countOnly_not_query_sufficient (n := n) hn
    ((markovCountQuerySufficient_iff_twoSidedContextSufficient
      (n := n)
      (markovCountOnlySummary (n := n))).mpr hSuff)

end FormalProofs.OPT
