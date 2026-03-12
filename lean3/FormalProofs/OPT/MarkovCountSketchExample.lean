import FormalProofs.OPT.PreservationTheorems

/-!
# Markov Changepoint Count Sketch (Worked Example)

This file provides a small, concrete instantiation of the OPS local laws (C1/C3 ≈ L1/L2)
for a mergeable sketch whose oracle is:

`f⋆(x) = (# of changepoints in x)`.

The sketch state is the minimal sufficient mergeable information:

- `count` = changepoint count within the span
- `first` / `last` = endpoint regimes, so a merge can account for the join changepoint

We then:

1. Define the sketch monoid and prove associativity.
2. Instantiate a deterministic summarizer `gExact := PMF.pure` and show `L1` and `L2`.
3. Invoke `one_pass` to conclude zero expected distortion at the root.
4. Define a simple "flip" summarizer and show it violates `L3` (C2), illustrating why
   on-range idempotence is a substantive additional requirement for multi-round results.
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

/-!
## Sketch Monoid
-/

inductive MarkovCountSketch (n : ℕ) : Type
| empty : MarkovCountSketch n
| nonempty (count : ℕ) (first : Fin n) (last : Fin n) : MarkovCountSketch n
deriving DecidableEq, Encodable

namespace MarkovCountSketch

variable {n : ℕ}

def join (l : Fin n) (f : Fin n) : ℕ :=
  if l = f then 0 else 1

def mul : MarkovCountSketch n → MarkovCountSketch n → MarkovCountSketch n
| empty, b => b
| a, empty => a
| nonempty c₁ f₁ l₁, nonempty c₂ f₂ l₂ =>
    nonempty (c₁ + c₂ + join l₁ f₂) f₁ l₂

instance : One (MarkovCountSketch n) := ⟨empty⟩
instance : Mul (MarkovCountSketch n) := ⟨mul⟩

@[simp] lemma one_def : (1 : MarkovCountSketch n) = empty := rfl

@[simp] lemma mul_empty (a : MarkovCountSketch n) : a * empty = a := by
  cases a <;> rfl

@[simp] lemma empty_mul (a : MarkovCountSketch n) : empty * a = a := by
  cases a <;> rfl

@[simp] lemma mul_nonempty_nonempty (c₁ c₂ : ℕ) (f₁ l₁ f₂ l₂ : Fin n) :
    (nonempty c₁ f₁ l₁) * (nonempty c₂ f₂ l₂) =
      nonempty (c₁ + c₂ + join l₁ f₂) f₁ l₂ := rfl

instance : Monoid (MarkovCountSketch n) where
  one := 1
  mul := (· * ·)
  one_mul := by
    intro a; simpa using (empty_mul (n := n) a)
  mul_one := by
    intro a; simpa using (mul_empty (n := n) a)
  mul_assoc := by
    intro a b c
    cases a <;> cases b <;> cases c <;> simp [mul, join] <;> ac_rfl

def count : MarkovCountSketch n → ℕ
| empty => 0
| nonempty c _ _ => c

end MarkovCountSketch

/-!
## Oracle and Exact Summarizer
-/

open MarkovCountSketch

variable {n : ℕ}

def fstar (s : MarkovCountSketch n) : ℝ :=
  (MarkovCountSketch.count s : ℝ)

def gExact : Summarizer (MarkovCountSketch n) :=
  fun x => PMF.pure x

lemma tsum_indicator_mul {α : Type} [DecidableEq α] (b : α) (f : α → ℝ) :
    (∑' z : α, (if z = b then (1 : ENNReal) else 0).toReal * f z) = f b := by
  classical
  have h :
      (fun z : α => (if z = b then (1 : ENNReal) else 0).toReal * f z) =
        fun z : α => if z = b then f z else 0 := by
    funext z
    by_cases hz : z = b <;> simp [hz]
  simpa [h, tsum_ite_eq]

lemma tsum_indicator_mul_prop {α : Type} (b : α) (f : α → ℝ) :
    (∑' z : α, (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal * f z) = f b := by
  classical
  have h :
      (fun z : α =>
          (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal * f z) =
        fun z : α => if z = b then f z else 0 := by
    funext z
    by_cases hz : z = b <;> simp [hz]
  simpa [h, tsum_ite_eq]

lemma reduce_gExact (T : BinTree (MarkovCountSketch n)) :
    reduce (gExact (n := n)) T = PMF.pure (S T) := by
  induction T with
  | leaf b =>
      simp [reduce, gExact, S]
  | node TL TR ihL ihR =>
      simp [reduce, gExact, S, ihL, ihR]

lemma L1_gExact (T : BinTree (MarkovCountSketch n)) : L1 (gExact (n := n)) T (fstar (n := n)) := by
  intro b _hb
  -- Reduce the expectation under `PMF.pure` to a one-term `tsum`.
  simp [Eg, gExact, D, fstar]
  have hsum :
      (∑' z : MarkovCountSketch n,
          (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal *
            dist z.count b.count) =
        dist b.count b.count := by
    simpa using (tsum_indicator_mul_prop (b := b) (f := fun z => dist z.count b.count))
  calc
    (∑' z : MarkovCountSketch n,
        (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal *
          dist z.count b.count)
        = dist b.count b.count := hsum
    _ = 0 := by simp

lemma L2_gExact (T : BinTree (MarkovCountSketch n)) : L2 (gExact (n := n)) T (fstar (n := n)) := by
  intro p hp
  rcases p with ⟨TL, TR⟩
  have hreduce :
      reduce (gExact (n := n)) (BinTree.node TL TR) =
        PMF.pure (S (BinTree.node TL TR)) := by
    simpa using (reduce_gExact (n := n) (T := BinTree.node TL TR))
  -- Egu under a pure reduction is exactly `D(f⋆(S u), f⋆(S u)) = 0`.
  -- The `hp` hypothesis is unused: `gExact` preserves *every* node, not only realized ones.
  simp [Egu, hreduce, Eg, gExact, D, fstar]
  have hsum :
      (∑' z : MarkovCountSketch n,
          (@ite ENNReal (z = S (BinTree.node TL TR))
              (Classical.propDecidable (z = S (BinTree.node TL TR))) 1 0).toReal *
            dist z.count (S (BinTree.node TL TR)).count) =
        dist (S (BinTree.node TL TR)).count (S (BinTree.node TL TR)).count := by
    simpa using
      (tsum_indicator_mul_prop (b := S (BinTree.node TL TR))
        (f := fun z => dist z.count (S (BinTree.node TL TR)).count))
  calc
    (∑' z : MarkovCountSketch n,
        (@ite ENNReal (z = S (BinTree.node TL TR))
            (Classical.propDecidable (z = S (BinTree.node TL TR))) 1 0).toReal *
          dist z.count (S (BinTree.node TL TR)).count)
        = dist (S (BinTree.node TL TR)).count (S (BinTree.node TL TR)).count := hsum
    _ = 0 := by simp

theorem exactSketch_root_distortion_zero (T : BinTree (MarkovCountSketch n)) :
    Egu (gExact (n := n)) (root T) (fun z => D (fstar (n := n)) z (S T)) = 0 := by
  exact one_pass (gExact (n := n)) T (S T) (fstar (n := n)) rfl
    (L1_gExact (n := n) T) (L2_gExact (n := n) T)

/-!
## A Simple L3 Failure ("Flip on Range")
-/

def flip (s : MarkovCountSketch n) : MarkovCountSketch n :=
  match s with
  | MarkovCountSketch.empty => MarkovCountSketch.empty
  | MarkovCountSketch.nonempty c f l => MarkovCountSketch.nonempty (c + 1) f l

def gFlip : Summarizer (MarkovCountSketch n) :=
  fun x => PMF.pure (flip (n := n) x)

theorem not_L3_gFlip (n : ℕ) (hn : 0 < n) :
    ¬ L3 (gFlip (n := n)) (fstar (n := n)) := by
  intro hL3
  let r : Fin n := ⟨0, hn⟩
  let x : MarkovCountSketch n := MarkovCountSketch.nonempty 0 r r
  let Z : MarkovCountSketch n := flip (n := n) x
  have hInRange : InRange (gFlip (n := n)) Z := by
    refine ⟨x, ?_⟩
    simp [gFlip, Z, flip]
  have h0 := hL3 Z hInRange
  -- Compute the L3 distortion explicitly: `gFlip` increments count again.
  have hpos : Eg (gFlip (n := n)) (fun z => D (fstar (n := n)) z Z) Z = 1 := by
    have hsum :
        (∑' z : MarkovCountSketch n,
            (@ite ENNReal (z = flip (n := n) Z)
                (Classical.propDecidable (z = flip (n := n) Z)) 1 0).toReal *
              D (fstar (n := n)) z Z) =
          D (fstar (n := n)) (flip (n := n) Z) Z := by
      simpa using
        (tsum_indicator_mul_prop (b := flip (n := n) Z) (f := fun z => D (fstar (n := n)) z Z))
    calc
      Eg (gFlip (n := n)) (fun z => D (fstar (n := n)) z Z) Z =
          (∑' z : MarkovCountSketch n,
              (@ite ENNReal (z = flip (n := n) Z)
                  (Classical.propDecidable (z = flip (n := n) Z)) 1 0).toReal *
                D (fstar (n := n)) z Z) := by
            simp [Eg, gFlip]
      _ = D (fstar (n := n)) (flip (n := n) Z) Z := hsum
      _ = 1 := by
            simp [D, fstar, Z, x, flip, Real.dist_eq, MarkovCountSketch.count] <;> norm_num
  -- Contradiction: L3 would require this expectation to be 0.
  have : (1 : ℝ) = 0 := by simpa [hpos] using h0
  norm_num at this

end FormalProofs.OPT
