import FormalProofs.OPT.LocalLaws

/-!
# FormalProofs/CounterexampleExistence.lean

## Paper Reference: Sections 9-10

This file formalizes the existence results (Section 9) and counterexamples (Section 10):

### Section 9: Existence of Oracle-Preserving Summarizers

- **Proposition 9.1** (`prop9_1_canonical_rep_exists`): Canonical representative construction.
  For any oracle f*, there exists a summarizer g such that D(g(x), x) = 0 for all x,
  achieved by choosing a canonical fiber representative for each oracle value.

- **Proposition 9.2** (`prop9_2_compact_encoding_exists`): Compact encoding construction.
  Given an encoding enc : Y → Strings with fstar(enc(y)) = y, the summarizer
  g(x) := enc(fstar(x)) preserves oracle values.

### Section 10: L3 is Substantive (Counterexample)

- **Theorem 10.1** (`thm10_1_L3_not_derivable`): Stability counterexample.
  Constructs a "bad" summarizer that satisfies L1 on fresh inputs but violates L3,
  demonstrating that on-range idempotence is a substantive requirement.

Key constructions:
- `g_can`: Canonical representative summarizer (Section 9.1)
- `g_bad`: Counterexample summarizer that breaks L3 (Section 10)
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Pointwise
open Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Section 10: Counterexample - Stability is Substantive

This section formalizes the counterexample from Section 10 of the paper.
It constructs a "bad" summarizer g_bad that:
1. Satisfies L1 on all fresh inputs (b ≠ POS, NEG)
2. Fails L3 because g_bad(POS) = NEG, flipping the oracle value

This demonstrates that L3 (on-range idempotence) is a substantive requirement
that cannot be derived from L1 and L2 alone.
-/

section Counterexample

-- For the counterexample, we work with a specific oracle space (Real numbers)
-- and string type with distinguished elements POS and NEG.

variable {Strings' : Type*} [DecidableEq Strings'] [Monoid Strings']

-- Special tokens (axiomatized for the counterexample)
variable (POS NEG : Strings')
variable (fstar' : Strings' → ℝ)

-- Axioms about the tokens
variable (h_pos : fstar' POS = 1)
variable (h_neg : fstar' NEG = 0)
variable (h_distinct : POS ≠ NEG)
-- Binary oracle: fstar only takes values 0 or 1 (standard for classification tasks)
variable (h_binary : ∀ x, fstar' x = 0 ∨ fstar' x = 1)

/-
The "bad" summarizer:
- If input is POS or NEG, return NEG (this breaks idempotence on POS)
- Otherwise, return POS if oracle = 1, else NEG
-/
def g_bad (x : Strings') : Strings' :=
  if x = POS ∨ x = NEG then NEG
  else if fstar' x = 1 then POS
  else NEG

-- Key property: g_bad maps POS to NEG, which breaks idempotence
lemma g_bad_on_POS : g_bad POS NEG fstar' POS = NEG := by
  simp only [g_bad, eq_self_iff_true, true_or, ite_true]

lemma g_bad_on_NEG : g_bad POS NEG fstar' NEG = NEG := by
  simp only [g_bad, eq_self_iff_true, or_true, ite_true]

lemma g_bad_on_fresh (b : Strings') (h_ne_pos : b ≠ POS) (h_ne_neg : b ≠ NEG) :
    g_bad POS NEG fstar' b = if fstar' b = 1 then POS else NEG := by
  simp only [g_bad, h_ne_pos, h_ne_neg, or_self, ite_false]

-- Distortion function for this example (using real distance)
def D_real (fstar'' : Strings' → ℝ) (z x : Strings') : ℝ :=
  |fstar'' z - fstar'' x|

-- Main theorem part 1: g_bad preserves oracle on "fresh" inputs
theorem g_bad_preserves_fresh (b : Strings')
    (h_ne_pos : b ≠ POS) (h_ne_neg : b ≠ NEG)
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0)
    (hbin : ∀ x, fstar' x = 0 ∨ fstar' x = 1) :
    D_real fstar' (g_bad POS NEG fstar' b) b = 0 := by
  rw [g_bad_on_fresh POS NEG fstar' b h_ne_pos h_ne_neg]
  simp only [D_real]
  cases hbin b with
  | inl h0 =>
    simp only [h0]
    norm_num
    simp only [hn, abs_zero]
  | inr h1 =>
    simp only [h1, ite_true, hp, sub_self, abs_zero]

-- Main theorem part 2: g_bad breaks on POS (which is in range)
theorem g_bad_breaks_on_range
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0) :
    D_real fstar' (g_bad POS NEG fstar' POS) POS = 1 := by
  rw [g_bad_on_POS]
  simp only [D_real, hn, hp, zero_sub, abs_neg, abs_one]

-- POS is in the range of g_bad (witnessed by any b with fstar b = 1 and b ≠ POS, NEG)
lemma POS_in_range_g_bad (b : Strings') (h_ne_pos : b ≠ POS) (h_ne_neg : b ≠ NEG)
    (hb : fstar' b = 1) :
    g_bad POS NEG fstar' b = POS := by
  rw [g_bad_on_fresh POS NEG fstar' b h_ne_pos h_ne_neg]
  simp only [hb, ite_true]

-- Conclusion: L3 fails for g_bad because re-summarizing POS flips to NEG
theorem L3_fails_for_g_bad
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0) :
    D_real fstar' (g_bad POS NEG fstar' POS) POS > 0 := by
  rw [g_bad_breaks_on_range POS NEG fstar' hp hn]
  norm_num

/-
Combined statement matching the LaTeX (Section 10, Example 10.1):
1. For all b ≠ POS, NEG: distortion is 0 (L1 holds on fresh inputs)
2. For POS: distortion > 0 (L3 fails because POS is in range but g_bad(POS) = NEG)
-/
theorem stability_counterexample
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0)
    (hbin : ∀ x, fstar' x = 0 ∨ fstar' x = 1) :
    (∀ b : Strings', b ≠ POS → b ≠ NEG → D_real fstar' (g_bad POS NEG fstar' b) b = 0) ∧
    D_real fstar' (g_bad POS NEG fstar' POS) POS > 0 :=
  ⟨fun b h1 h2 => g_bad_preserves_fresh POS NEG fstar' b h1 h2 hp hn hbin,
   L3_fails_for_g_bad POS NEG fstar' hp hn⟩

end Counterexample

/-!
## Section 9: Existence Results

This section formalizes the existence results from Section 9 of the paper.
It proves that oracle-preserving summarizers exist in two ways:
1. Canonical Representative: Using choice to pick fiber representatives
2. Compact Encoding: Given an encoding that inverts fstar
-/

section ExistenceResults

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-
Canonical Representative Construction.
Key idea: for each oracle value y in the image of fstar, pick a canonical string that maps to y.
We use Classical.choose to select a representative from each fiber.
-/

-- InImage predicate: y is in the image of fstar
def InImage (fstar : Strings → Y) (y : Y) : Prop := ∃ x, fstar x = y

-- Every fstar(x) is in the image
lemma fstar_in_image (fstar : Strings → Y) (x : Strings) : InImage fstar (fstar x) :=
  ⟨x, rfl⟩

-- Representative function: given y in image, pick a preimage
noncomputable def rep (fstar : Strings → Y) (y : Y) (hy : InImage fstar y) : Strings :=
  hy.choose

-- The representative maps to y
lemma fstar_rep_eq (fstar : Strings → Y) (y : Y) (hy : InImage fstar y) :
    fstar (rep fstar y hy) = y :=
  hy.choose_spec

-- Canonical summarizer: g_can(x) = rep(fstar(x))
noncomputable def g_can (fstar : Strings → Y) (x : Strings) : Strings :=
  rep fstar (fstar x) (fstar_in_image fstar x)

-- Key property: fstar(g_can(x)) = fstar(x)
lemma fstar_g_can_eq (fstar : Strings → Y) (x : Strings) :
    fstar (g_can fstar x) = fstar x :=
  fstar_rep_eq fstar (fstar x) (fstar_in_image fstar x)

-- Crucial lemma: g_can only depends on fstar(x), not on x
-- When fstar(z) = fstar(x), we have g_can(z) = g_can(x)
-- This uses proof irrelevance: InImage fstar (fstar z) = InImage fstar (fstar x)
-- when fstar z = fstar x, so Classical.choose gives the same result.
lemma g_can_eq_of_fstar_eq (fstar : Strings → Y) (x z : Strings) (h : fstar z = fstar x) :
    g_can fstar z = g_can fstar x := by
  unfold g_can rep
  -- InImage fstar (fstar z) and InImage fstar (fstar x) are the same Prop
  -- since fstar z = fstar x, so by proof irrelevance, choose gives same result
  simp only [h]

/-
Proposition 9.1: Canonical Representative Theorem

There exists a summarizer g : Strings → Strings such that:
1. Oracle preservation: D(f*(g(x)), f*(x)) = 0 for all x
2. Idempotence on range: g(z) = z for all z in range(g)
-/
theorem canonical_rep_exists (fstar : Strings → Y) :
    ∃ g : Strings → Strings,
      (∀ x, D fstar (g x) x = 0) ∧
      (∀ z, (∃ x, g x = z) → g z = z) := by
  use g_can fstar
  constructor
  -- Part 1: Oracle preservation
  · intro x
    unfold D
    rw [fstar_g_can_eq fstar x]
    exact dist_self (fstar x)
  -- Part 2: Idempotence on range
  · intro z ⟨x, hx⟩
    -- z = g_can(x), so fstar(z) = fstar(x)
    have hz : fstar z = fstar x := by rw [← hx]; exact fstar_g_can_eq fstar x
    -- By g_can_eq_of_fstar_eq, g_can(z) = g_can(x) = z
    rw [g_can_eq_of_fstar_eq fstar x z hz, hx]

/-
Proposition 9.2: Compact Encoding

Given an encoding enc : Y → Strings that inverts fstar (i.e., fstar(enc(y)) = y),
we can construct an oracle-preserving summarizer g_enc(x) := enc(fstar(x)).
-/
theorem compact_encoding_exists (fstar : Strings → Y)
    (enc : Y → Strings)
    (h_inv : ∀ y, fstar (enc y) = y) :
    ∃ g : Strings → Strings,
      (∀ x, D fstar (g x) x = 0) ∧
      (∀ z, (∃ x, g x = z) → g z = z) := by
  -- Define g_enc(x) := enc(fstar(x))
  use fun x => enc (fstar x)
  constructor
  -- Part 1: Oracle preservation
  · intro x
    unfold D
    rw [h_inv (fstar x)]
    exact dist_self (fstar x)
  -- Part 2: Idempotence on range
  · intro z ⟨x, hx⟩
    -- z = enc(fstar(x))
    -- g_enc(z) = enc(fstar(z)) = enc(fstar(enc(fstar(x)))) = enc(fstar(x)) = z
    simp only [← hx, h_inv]

/-!
## Paper-Numbered Aliases

These aliases provide explicit theorem names matching the paper's proposition/theorem numbering.
-/

/-- **Proposition 9.1: Canonical Representative Exists**

**Paper Reference:** Section 9, Proposition 9.1

For any oracle f* : Strings → Y, there exists a summarizer g such that:
1. g preserves oracle values: D(g(x), x) = 0 for all x
2. g is idempotent on its range: g(z) = z for all z in range(g)

This is achieved via the canonical representative construction, which picks
a fiber representative for each oracle value using Classical.choose. -/
theorem prop9_1_canonical_rep_exists (fstar : Strings → Y) :
    ∃ g : Strings → Strings,
      (∀ x, D fstar (g x) x = 0) ∧
      (∀ z, (∃ x, g x = z) → g z = z) :=
  canonical_rep_exists fstar

/-- **Proposition 9.2: Compact Encoding Construction**

**Paper Reference:** Section 9, Proposition 9.2

Given an encoding enc : Y → Strings that inverts f* (i.e., f*(enc(y)) = y),
the summarizer g(x) := enc(f*(x)) satisfies:
1. Oracle preservation: D(g(x), x) = 0 for all x
2. Idempotence on range: g(z) = z for all z in range(g)

This provides a constructive alternative to the canonical representative when
an explicit encoding function is available. -/
theorem prop9_2_compact_encoding_exists (fstar : Strings → Y)
    (enc : Y → Strings)
    (h_inv : ∀ y, fstar (enc y) = y) :
    ∃ g : Strings → Strings,
      (∀ x, D fstar (g x) x = 0) ∧
      (∀ z, (∃ x, g x = z) → g z = z) :=
  compact_encoding_exists fstar enc h_inv

end ExistenceResults

section CounterexampleAliases

variable {Strings' : Type*} [DecidableEq Strings'] [Monoid Strings']
variable (POS NEG : Strings')
variable (fstar' : Strings' → ℝ)

/-- **Theorem 10.1: L3 is Not Derivable from L1 and L2**

**Paper Reference:** Section 10, Theorem 10.1

Constructs a counterexample summarizer g_bad that demonstrates L3 (on-range idempotence)
is a substantive requirement that cannot be derived from L1 and L2 alone.

The summarizer g_bad:
- Satisfies L1 on all "fresh" inputs (b ≠ POS, NEG): D(g_bad(b), b) = 0
- Fails L3 on POS (which is in range): D(g_bad(POS), POS) > 0 because g_bad(POS) = NEG

This shows that stability (L3) is required as an independent axiom. -/
theorem thm10_1_L3_not_derivable
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0)
    (hbin : ∀ x, fstar' x = 0 ∨ fstar' x = 1) :
    (∀ b : Strings', b ≠ POS → b ≠ NEG → D_real fstar' (g_bad POS NEG fstar' b) b = 0) ∧
    D_real fstar' (g_bad POS NEG fstar' POS) POS > 0 :=
  stability_counterexample POS NEG fstar' hp hn hbin

/-- `g_bad` always returns one of the two distinguished token summaries. -/
lemma g_bad_value_is_token (x : Strings') :
    g_bad POS NEG fstar' x = POS ∨ g_bad POS NEG fstar' x = NEG := by
  by_cases htok : x = POS ∨ x = NEG
  · simp [g_bad, htok]
  · by_cases hone : fstar' x = 1
    · simp [g_bad, htok, hone]
    · simp [g_bad, htok, hone]

/-- Fresh-input preservation as oracle-value equality rather than zero
distortion. -/
lemma fstar_g_bad_eq_of_fresh
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0)
    (hbin : ∀ x, fstar' x = 0 ∨ fstar' x = 1)
    (b : Strings') (h_ne_pos : b ≠ POS) (h_ne_neg : b ≠ NEG) :
    fstar' (g_bad POS NEG fstar' b) = fstar' b := by
  rw [g_bad_on_fresh POS NEG fstar' b h_ne_pos h_ne_neg]
  cases hbin b with
  | inl h0 =>
      simp [h0, hn]
  | inr h1 =>
      simp [h1, hp]

/-- Public-shape C2 independence counterexample.

This is the Lean-facing version of paper Example `ex:c2-independent`: under
first-token-style concat assumptions, `g_bad` satisfies C1 on fresh inputs and
the fresh-input C3 merge chain, but fails C2/idempotence on its own range. -/
theorem ex_c2_independent_formalized
    (hp : fstar' POS = 1) (hn : fstar' NEG = 0)
    (hbin : ∀ x, fstar' x = 0 ∨ fstar' x = 1)
    (h_mul_left : ∀ a b : Strings', fstar' (a * b) = fstar' a)
    (h_fresh_mul : ∀ u v : Strings',
      u ≠ POS → u ≠ NEG → v ≠ POS → v ≠ NEG →
        (u * v ≠ POS ∧ u * v ≠ NEG))
    (h_token_mul_fresh : ∀ a b : Strings',
      (a = POS ∨ a = NEG) → (b = POS ∨ b = NEG) →
        (a * b ≠ POS ∧ a * b ≠ NEG))
    (h_pos_fresh_exists : ∃ b : Strings', b ≠ POS ∧ b ≠ NEG ∧ fstar' b = 1) :
    (∀ b : Strings', b ≠ POS → b ≠ NEG →
      D_real fstar' (g_bad POS NEG fstar' b) b = 0) ∧
    (∀ u v : Strings', u ≠ POS → u ≠ NEG → v ≠ POS → v ≠ NEG →
      D_real fstar' (g_bad POS NEG fstar' (u * v)) (u * v) = 0 ∧
      D_real fstar'
        (g_bad POS NEG fstar'
          (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v))
        (u * v) = 0) ∧
    POS ∈ Set.range (g_bad POS NEG fstar') ∧
    D_real fstar' (g_bad POS NEG fstar' POS) POS > 0 := by
  constructor
  · intro b hb_pos hb_neg
    exact g_bad_preserves_fresh POS NEG fstar' b hb_pos hb_neg hp hn hbin
  constructor
  · intro u v hu_pos hu_neg hv_pos hv_neg
    constructor
    · have huv_fresh := h_fresh_mul u v hu_pos hu_neg hv_pos hv_neg
      exact g_bad_preserves_fresh POS NEG fstar' (u * v)
        huv_fresh.1 huv_fresh.2 hp hn hbin
    · have hgu_token := g_bad_value_is_token POS NEG fstar' u
      have hgv_token := g_bad_value_is_token POS NEG fstar' v
      have hmerge_fresh :=
        h_token_mul_fresh (g_bad POS NEG fstar' u) (g_bad POS NEG fstar' v)
          hgu_token hgv_token
      have hgu_pres :
          fstar' (g_bad POS NEG fstar' u) = fstar' u :=
        fstar_g_bad_eq_of_fresh POS NEG fstar' hp hn hbin u hu_pos hu_neg
      have hmerge_pres :
          fstar'
              (g_bad POS NEG fstar'
                (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v)) =
            fstar' (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v) :=
        fstar_g_bad_eq_of_fresh POS NEG fstar' hp hn hbin
          (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v)
          hmerge_fresh.1 hmerge_fresh.2
      have h_eq :
          fstar'
              (g_bad POS NEG fstar'
                (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v)) =
            fstar' (u * v) := by
        calc
          fstar'
              (g_bad POS NEG fstar'
                (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v))
              = fstar' (g_bad POS NEG fstar' u * g_bad POS NEG fstar' v) :=
                hmerge_pres
          _ = fstar' (g_bad POS NEG fstar' u) :=
                h_mul_left (g_bad POS NEG fstar' u) (g_bad POS NEG fstar' v)
          _ = fstar' u := hgu_pres
          _ = fstar' (u * v) := (h_mul_left u v).symm
      unfold D_real
      rw [h_eq]
      simp
  · constructor
    · rcases h_pos_fresh_exists with ⟨b, hb_pos, hb_neg, hb_one⟩
      exact ⟨b, POS_in_range_g_bad POS NEG fstar' b hb_pos hb_neg hb_one⟩
    · exact L3_fails_for_g_bad POS NEG fstar' hp hn

end CounterexampleAliases

end
