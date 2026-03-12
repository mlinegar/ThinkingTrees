import FormalProofs.OPT.MarkovCountSketchExample
import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.SketchRecovery

/-!
# FormalProofs/OPT/MarkovPathDGP.lean

This file lifts the worked Markov count-sketch example from theorem-domain
states to the actual **document support** used by the Markov changepoint
simulations: finite sequences of latent regimes.

The important distinction is:

- the **Markov transition law** governs which regime sequences are likely;
- the **exact mergeable sketch theorem** only needs support-level algebra.

So we formalize the DGP support as lists of regimes, prove that the exact
Markov sketch is an exact mergeable fold on those lists, and then show that
the count-only statistic used by the undersupported control is *not*
compositionally sufficient.
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

abbrev MarkovPath (n : ℕ) := List (Fin n)

namespace MarkovPath

variable {n : ℕ}

/-- Single-token exact sketch state. -/
def singletonSketch (x : Fin n) : MarkovCountSketch n :=
  MarkovCountSketch.nonempty 0 x x

/-- Exact Markov sketch encoder on realized regime sequences. -/
def encodePath : MarkovPath n → MarkovCountSketch n
  | [] => 1
  | x :: xs => singletonSketch x * encodePath xs

/-- Oracle: number of changepoints in the realized regime sequence. -/
def changepointCount : MarkovPath n → ℕ
  | [] => 0
  | [_] => 0
  | x :: y :: xs => MarkovCountSketch.join x y + changepointCount (y :: xs)

/-- Under-supported control used in the simulation family: only the raw count
is retained. -/
def countOnlyFeature : MarkovPath n → ℕ := changepointCount

@[simp] theorem encodePath_nil :
    encodePath ([] : MarkovPath n) = 1 := rfl

@[simp] theorem encodePath_cons (x : Fin n) (xs : MarkovPath n) :
    encodePath (x :: xs) = singletonSketch x * encodePath xs := rfl

@[simp] theorem changepointCount_nil :
    changepointCount ([] : MarkovPath n) = 0 := rfl

@[simp] theorem changepointCount_singleton (x : Fin n) :
    changepointCount ([x] : MarkovPath n) = 0 := rfl

@[simp] theorem changepointCount_cons_cons (x y : Fin n) (xs : MarkovPath n) :
    changepointCount (x :: y :: xs) =
      MarkovCountSketch.join x y + changepointCount (y :: xs) := rfl

/-- Exact sketch encoding is compositional over concatenation. -/
theorem encodePath_append (xs ys : MarkovPath n) :
    encodePath (xs ++ ys) = encodePath xs * encodePath ys := by
  induction xs with
  | nil =>
      simp [encodePath]
  | cons x xs ih =>
      simp [encodePath, ih, mul_assoc]

/-- Nonempty paths encode to a sketch whose first endpoint is the first token
and whose count is exactly the changepoint count. -/
theorem encodePath_cons_eq (x : Fin n) :
    ∀ xs : MarkovPath n,
      ∃ l, encodePath (x :: xs) = MarkovCountSketch.nonempty (changepointCount (x :: xs)) x l
  | [] => by
      refine ⟨x, ?_⟩
      simp [encodePath, singletonSketch, changepointCount]
  | y :: ys => by
      obtain ⟨l, hl⟩ := encodePath_cons_eq y ys
      refine ⟨l, ?_⟩
      rw [encodePath, hl]
      simp [singletonSketch, changepointCount,
        Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

@[simp] theorem count_encodePath (xs : MarkovPath n) :
    MarkovCountSketch.count (encodePath xs) = changepointCount xs := by
  cases xs with
  | nil =>
      simp [encodePath, changepointCount, MarkovCountSketch.count]
  | cons x xs =>
      obtain ⟨l, hl⟩ := encodePath_cons_eq (x := x) xs
      simpa [MarkovCountSketch.count] using congrArg MarkovCountSketch.count hl

/-- The exact Markov state is a congruent encoded feature on the raw Markov
paths. This is the key support-level condition behind exact local laws. -/
lemma encodePath_congruent :
    ∀ (sL sR x y : MarkovPath n),
      encodePath sL = encodePath x →
      encodePath sR = encodePath y →
      encodePath (sL * sR) = encodePath (x * y) := by
  intro sL sR x y hL hR
  calc
    encodePath (sL * sR) = encodePath sL * encodePath sR := by
      simpa using (encodePath_append (xs := sL) (ys := sR))
    _ = encodePath x * encodePath y := by simpa [hL, hR]
    _ = encodePath (x * y) := by
      simpa using (encodePath_append (xs := x) (ys := y)).symm

/-- Exact local laws on raw Markov paths, viewed through the exact theorem-domain
state `encodePath`. -/
theorem local_laws_of_encoded_state (T : BinTree (MarkovPath n)) :
    LocalLawsBundle
      (sketchSummarizer (identitySketchOperator (Strings := MarkovPath n))) T
      (encodedOracle (Strings := MarkovPath n) (encodePath (n := n))) := by
  simpa using
    (local_laws_of_identity_encoded_feature
      (Strings := MarkovPath n)
      (feature := encodePath (n := n))
      encodePath_congruent
      (T := T))

/-- Any downstream utility on the exact Markov sketch state is preserved exactly
when leaves are raw realized Markov paths and the leaf encoder is `encodePath`. -/
theorem state_exact_on_tree
    {β : Type*}
    (u : MarkovCountSketch n → β)
    (T : BinTree (MarkovPath n)) :
    u (mergeFold (encode := encodePath (n := n)) (merge := (· * ·)) T) =
      u (encodePath (S T)) := by
  simpa using
    (mergeableStateUtility_exact_on_tree
      (Strings := MarkovPath n)
      (Sketch := MarkovCountSketch n)
      (encode := encodePath (n := n))
      (merge := (· * ·))
      (feature := encodePath (n := n))
      (h_encode := fun _ => rfl)
      (h_merge := fun x y => (encodePath_append (xs := x) (ys := y)).symm)
      (u := u)
      (T := T))

/-- In particular, the changepoint-count oracle is exactly preserved on every
tree under the exact Markov sketch. -/
theorem count_exact_on_tree (T : BinTree (MarkovPath n)) :
    MarkovCountSketch.count (mergeFold (encode := encodePath (n := n)) (merge := (· * ·)) T) =
      changepointCount (S T) := by
  rw [mergeFold_eq_feature
    (encode := encodePath (n := n))
    (merge := (· * ·))
    (feature := encodePath (n := n))
    (h_encode := fun _ => rfl)
    (h_merge := fun x y => (encodePath_append (xs := x) (ys := y)).symm)
    (T := T)]
  simpa using (count_encodePath (n := n) (xs := S T))

/-- The count-only statistic is not a congruent feature on Markov paths once the
alphabet has at least two distinct regimes. This is the formal reason the
undersupported baseline is not theorem-backed. -/
theorem countOnlyFeature_not_congruent (hn : 1 < n) :
    ¬ ∀ sL sR x y : MarkovPath n,
      countOnlyFeature (n := n) sL = countOnlyFeature (n := n) x →
      countOnlyFeature (n := n) sR = countOnlyFeature (n := n) y →
      countOnlyFeature (n := n) (sL * sR) = countOnlyFeature (n := n) (x * y) := by
  intro hcongr
  let a : Fin n := ⟨0, lt_trans (by decide : 0 < 1) hn⟩
  let b : Fin n := ⟨1, hn⟩
  have hab : a ≠ b := by
    intro hEq
    have : (0 : ℕ) = 1 := by simpa [a, b] using congrArg Fin.val hEq
    norm_num at this
  have hleafL :
      countOnlyFeature (n := n) ([a] : MarkovPath n) =
        countOnlyFeature (n := n) ([a] : MarkovPath n) := rfl
  have hleafR :
      countOnlyFeature (n := n) ([a] : MarkovPath n) =
        countOnlyFeature (n := n) ([b] : MarkovPath n) := by
    simp [countOnlyFeature, changepointCount]
  have hbad := hcongr [a] [a] [a] [b] hleafL hleafR
  have hbad' :
      countOnlyFeature (n := n) (([a] : MarkovPath n) * ([a] : MarkovPath n)) =
        countOnlyFeature (n := n) (([a] : MarkovPath n) * ([b] : MarkovPath n)) := by
    simpa using hbad
  have hleftc :
      countOnlyFeature (n := n) (([a] : MarkovPath n) * ([a] : MarkovPath n)) = 0 := by
    change changepointCount ([a, a] : MarkovPath n) = 0
    simp [changepointCount, MarkovCountSketch.join]
  have hrightc :
      countOnlyFeature (n := n) (([a] : MarkovPath n) * ([b] : MarkovPath n)) = 1 := by
    change changepointCount ([a, b] : MarkovPath n) = 1
    simp [changepointCount, MarkovCountSketch.join, hab]
  have : (0 : ℕ) = 1 := by
    rw [hleftc, hrightc] at hbad'
    exact hbad'
  norm_num at this

/-- Concrete tree-level counterexample: count-only leaf summaries cannot recover
the true root changepoint count even on a two-leaf tree. -/
theorem countOnly_mergeFold_counterexample (hn : 1 < n) :
    ∃ T : BinTree (MarkovPath n),
      mergeFold (encode := countOnlyFeature (n := n)) (merge := Nat.add) T ≠
        countOnlyFeature (n := n) (S T) := by
  let a : Fin n := ⟨0, lt_trans (by decide : 0 < 1) hn⟩
  let b : Fin n := ⟨1, hn⟩
  have hab : a ≠ b := by
    intro hEq
    have : (0 : ℕ) = 1 := by simpa [a, b] using congrArg Fin.val hEq
    norm_num at this
  let T : BinTree (MarkovPath n) := BinTree.node (BinTree.leaf [a]) (BinTree.leaf [b])
  refine ⟨T, ?_⟩
  have hleft :
      mergeFold (encode := countOnlyFeature (n := n)) (merge := Nat.add) T = 0 := by
    simp [T, mergeFold, countOnlyFeature, changepointCount]
  have hright :
      countOnlyFeature (n := n) (S T) = 1 := by
    change changepointCount ([a, b] : MarkovPath n) = 1
    simp [changepointCount, MarkovCountSketch.join, hab]
  intro hEq
  rw [hleft, hright] at hEq
  norm_num at hEq

end MarkovPath

end FormalProofs.OPT
