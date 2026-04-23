import FormalProofs.OPT.CoreDefinitions
import Mathlib.Data.Finite.Card

/-!
# FormalProofs/OPT/OracleEntropy.lean

## Finite-Support Log-Cardinality Envelope for Oracle Projection

This file intentionally does **not** formalize Shannon entropy. The current
Lean-backed information-theory surface for C-TreePO is the oracle-sufficiency /
score-transport / task-relevant KLIC bridge in `InformationSufficiency.lean`.

What remains useful here is a weaker finite-support statement with no axioms:
the oracle image cannot have larger log-cardinality than the source space, and
any readout that factors through the oracle cannot have larger log-cardinality
than the oracle image itself.

This gives a clean, formally proved replacement for the earlier axiomatized
entropy context while staying honest about scope: it is a combinatorial upper
bound, not a Shannon source-coding theorem.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT.InformationTheory

/-- Log-cardinality stabilized with `max 1`, so empty types also contribute
`0` rather than `log 0`. -/
noncomputable def LogCard (α : Type*) : ℝ :=
  Real.log (((max 1 (Nat.card α) : ℕ) : ℝ))

theorem LogCard_nonneg (α : Type*) : 0 ≤ LogCard α := by
  unfold LogCard
  have hge : (1 : ℝ) ≤ ((max 1 (Nat.card α) : ℕ) : ℝ) := by
    exact_mod_cast (Nat.le_max_left 1 (Nat.card α))
  exact Real.log_nonneg hge

theorem LogCard_mono {α β : Type*} (hcard : Nat.card α ≤ Nat.card β) :
    LogCard α ≤ LogCard β := by
  unfold LogCard
  have hmax : max 1 (Nat.card α) ≤ max 1 (Nat.card β) :=
    max_le_max le_rfl hcard
  have hmax_real :
      (((max 1 (Nat.card α) : ℕ) : ℝ)) ≤ (((max 1 (Nat.card β) : ℕ) : ℝ)) := by
    exact_mod_cast hmax
  exact Real.log_le_log (by positivity) hmax_real

variable {Strings : Type*} [Finite Strings]
variable {Y : Type*}

/-- Pushforward of a source distribution through the oracle. -/
noncomputable def oraclePushforward
    (p : PMF Strings) (fstar : Strings → Y) : PMF Y :=
  p.map fstar

/-- Log-cardinality envelope of the source type. -/
noncomputable def SourceLogCard (α : Type*) : ℝ :=
  LogCard α

/-- Log-cardinality envelope of the oracle image. -/
noncomputable def OracleLogCard (fstar : Strings → Y) : ℝ :=
  LogCard (Set.range fstar)

/-- The combinatorial amount of source-space log-cardinality that may be
discarded while preserving the oracle image. -/
noncomputable def LogCardGap (fstar : Strings → Y) : ℝ :=
  SourceLogCard Strings - OracleLogCard fstar

/-- The oracle image cannot have larger log-cardinality than the source type. -/
theorem OracleLogCard_le_SourceLogCard
    (fstar : Strings → Y) :
    OracleLogCard fstar ≤ SourceLogCard Strings := by
  unfold OracleLogCard SourceLogCard
  exact LogCard_mono (Finite.card_range_le fstar)

/-- The log-cardinality gap is nonnegative. -/
theorem LogCardGap_nonneg
    (fstar : Strings → Y) :
    0 ≤ LogCardGap fstar := by
  unfold LogCardGap
  linarith [OracleLogCard_le_SourceLogCard fstar]

/-- A constant oracle has zero oracle log-cardinality. -/
theorem OracleLogCard_eq_zero_of_constant
    (fstar : Strings → Y) (y₀ : Y)
    (hConst : ∀ x, fstar x = y₀) :
    OracleLogCard fstar = 0 := by
  unfold OracleLogCard LogCard
  have hSub : Subsingleton (Set.range fstar) := by
    refine ⟨?_⟩
    intro a b
    apply Subtype.ext
    rcases a.2 with ⟨x, hx⟩
    rcases b.2 with ⟨x', hx'⟩
    rw [← hx, ← hx', hConst x, hConst x']
  have hCard : Nat.card (Set.range fstar) ≤ 1 :=
    (Finite.card_le_one_iff_subsingleton).2 hSub
  rw [show (max 1 (Nat.card (Set.range fstar)) : ℕ) = 1 by exact max_eq_left hCard]
  simp

/-- If `h` factors through `fstar`, then the range of `h` is no larger than the
range of `fstar`. -/
theorem card_range_le_card_range_of_factors_through
    {Z : Type*}
    (fstar : Strings → Y)
    (h : Strings → Z) (h' : Y → Z)
    (hFactor : ∀ x, h x = h' (fstar x)) :
    Nat.card (Set.range h) ≤ Nat.card (Set.range fstar) := by
  let k : Set.range fstar → Z := fun y => h' y.1
  have hRange :
      Set.range h = Set.range k := by
    ext z
    constructor
    · rintro ⟨x, rfl⟩
      exact ⟨⟨fstar x, ⟨x, rfl⟩⟩, (hFactor x).symm⟩
    · rintro ⟨y, rfl⟩
      rcases y.2 with ⟨x, hx⟩
      refine ⟨x, ?_⟩
      simpa [k, hx] using hFactor x
  rw [hRange]
  exact Finite.card_range_le k

/-- Any readout that factors through the oracle image has log-cardinality
bounded by the oracle log-cardinality envelope. -/
theorem oracleFactored_logCard_le
    {Z : Type*}
    (fstar : Strings → Y)
    (h : Strings → Z) (h' : Y → Z)
    (hFactor : ∀ x, h x = h' (fstar x)) :
    LogCard (Set.range h) ≤ OracleLogCard fstar := by
  unfold OracleLogCard
  exact LogCard_mono <|
    card_range_le_card_range_of_factors_through fstar h h' hFactor

end FormalProofs.OPT.InformationTheory
