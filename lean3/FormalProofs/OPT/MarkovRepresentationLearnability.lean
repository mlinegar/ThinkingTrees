import FormalProofs.OPT.MarkovSufficiency

/-!
# FormalProofs/OPT/MarkovRepresentationLearnability.lean

Representation-test reductions for the clean Markov exact-sketch study.

The learnability simulations do not attempt to certify SGD convergence. The
Lean-facing question is instead:

- if a learned representation / readout pair exactly recovers the theorem-domain
  Markov sketch, what downstream guarantees follow?

This file packages the consequences used by the learnability map:

- exact sketch recovery implies Markov query sufficiency;
- exact sketch recovery implies zero root changepoint-count error; and
- approximate exact-sketch error upper-bounds changepoint-count error through a
  simple discrete transport inequality.
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

/-- A representation / readout pair exactly recovers the theorem-domain Markov
sketch state. -/
def MarkovRepresentationExactRecovery
    {Rep : Type*}
    (encode : MarkovCountSketch n → Rep)
    (decode : Rep → MarkovCountSketch n) : Prop :=
  ∀ s, decode (encode s) = s

/-- Exact recovery makes the representation injective on exact sketch states. -/
theorem markov_representation_exact_recovery_injective
    {Rep : Type*}
    {encode : MarkovCountSketch n → Rep}
    {decode : Rep → MarkovCountSketch n}
    (hRecover : MarkovRepresentationExactRecovery (n := n) encode decode) :
    Function.Injective encode := by
  intro x y hxy
  calc
    x = decode (encode x) := by symm; exact hRecover x
    _ = decode (encode y) := by simpa [hxy]
    _ = y := hRecover y

/-- Exact recovery of the theorem-domain sketch implies the representation is
sufficient for every two-sided changepoint-count query. -/
theorem markov_representation_exact_recovery_implies_query_sufficient
    {Rep : Type*}
    {encode : MarkovCountSketch n → Rep}
    {decode : Rep → MarkovCountSketch n}
    (hRecover : MarkovRepresentationExactRecovery (n := n) encode decode) :
    MarkovCountQuerySufficient (n := n) encode := by
  intro left right x y hxy
  have hEq :
      x = y :=
    markov_representation_exact_recovery_injective
      (n := n)
      hRecover
      hxy
  simpa [hEq]

/-- Exact recovery gives a decoder back to the exact Markov sketch by
construction. -/
theorem markov_representation_exact_recovery_has_exact_sketch_decoder
    {Rep : Type*}
    {encode : MarkovCountSketch n → Rep}
    {decode : Rep → MarkovCountSketch n}
    (hRecover : MarkovRepresentationExactRecovery (n := n) encode decode) :
    decode ∘ encode = id := by
  funext s
  exact hRecover s

/-- Discrete count error between two exact sketch states. -/
def markovCountError
    (x y : MarkovCountSketch n) : ℕ :=
  Nat.dist (MarkovCountSketch.count x) (MarkovCountSketch.count y)

/-- A simple discrete exact-sketch error that contains count disagreement plus
endpoint mismatches. This is enough for the learnability map's transport step:
count error is always bounded by exact-sketch error. -/
def markovExactSketchError : MarkovCountSketch n → MarkovCountSketch n → ℕ
  | empty, empty => 0
  | empty, nonempty c _ _ => Nat.dist 0 c + 2
  | nonempty c _ _, empty => Nat.dist c 0 + 2
  | nonempty c₁ f₁ l₁, nonempty c₂ f₂ l₂ =>
      Nat.dist c₁ c₂
        + (if f₁ = f₂ then 0 else 1)
        + (if l₁ = l₂ then 0 else 1)

/-- Count error is bounded by exact-sketch error. -/
theorem markov_count_error_le_exact_sketch_error
    (x y : MarkovCountSketch n) :
    markovCountError (n := n) x y ≤ markovExactSketchError (n := n) x y := by
  cases x with
  | empty =>
      cases y with
      | empty =>
          simp [markovCountError, markovExactSketchError]
      | nonempty c f l =>
          have h : Nat.dist 0 c ≤ Nat.dist 0 c + 2 :=
            Nat.le_add_right (Nat.dist 0 c) 2
          unfold markovCountError markovExactSketchError
          simp
          exact h
  | nonempty c₁ f₁ l₁ =>
      cases y with
      | empty =>
          have h : Nat.dist c₁ 0 ≤ Nat.dist c₁ 0 + 2 :=
            Nat.le_add_right (Nat.dist c₁ 0) 2
          unfold markovCountError markovExactSketchError
          simp
          exact h
      | nonempty c₂ f₂ l₂ =>
          have h :
              Nat.dist c₁ c₂ ≤
                Nat.dist c₁ c₂
                  + ((if f₁ = f₂ then 0 else 1) + (if l₁ = l₂ then 0 else 1)) :=
            Nat.le_add_right
              (Nat.dist c₁ c₂)
              ((if f₁ = f₂ then 0 else 1) + (if l₁ = l₂ then 0 else 1))
          unfold markovCountError markovExactSketchError
          simp [Nat.add_assoc]
          exact h

/-- Exact sketch recovery forces exact-sketch error zero on every recovered
state. -/
theorem markov_representation_exact_recovery_exact_sketch_error_zero
    {Rep : Type*}
    {encode : MarkovCountSketch n → Rep}
    {decode : Rep → MarkovCountSketch n}
    (hRecover : MarkovRepresentationExactRecovery (n := n) encode decode) :
    ∀ s,
      markovExactSketchError (n := n) (decode (encode s)) s = 0 := by
  intro s
  rw [hRecover s]
  cases s <;> simp [markovExactSketchError]

/-- Exact sketch recovery forces zero changepoint-count error at the root. -/
theorem markov_representation_exact_recovery_zero_root_count_error
    {Rep : Type*}
    {encode : MarkovCountSketch n → Rep}
    {decode : Rep → MarkovCountSketch n}
    (hRecover : MarkovRepresentationExactRecovery (n := n) encode decode) :
    ∀ s,
      markovCountError (n := n) (decode (encode s)) s = 0 := by
  intro s
  rw [hRecover s]
  simp [markovCountError]

end FormalProofs.OPT
