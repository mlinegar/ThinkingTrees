import FormalProofs.OPT.WorkedExampleMarkovTree

/-!
# FormalProofs/OPT/MarkovCarrierProjection.lean

This file packages the theorem-facing Markov route where a node carries:

- an arbitrary residual state `ρ`; and
- an exact Markov sketch `(count, first, last)`.

The key point is that the theorem-facing oracle depends only on the sketch
projection. The residual component may evolve arbitrarily as long as merge is
exact on the projected sketch.
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

/-- Abstract carrier state: arbitrary residual payload plus exact Markov sketch. -/
abbrev MarkovCarrierState (ρ : Type) (n : ℕ) := ρ × MarkovCountSketch n

namespace MarkovCarrierState

variable {ρ : Type} [Monoid ρ] {n : ℕ}

/-- The theorem-facing projection `π : H → K`. -/
def proj (x : MarkovCarrierState ρ n) : MarkovCountSketch n := x.2

/-- Carrier merge: residual merge is arbitrary, sketch merge is exact. -/
def mul (a b : MarkovCarrierState ρ n) : MarkovCarrierState ρ n :=
  (a.1 * b.1, a.2 * b.2)

instance : One (MarkovCarrierState ρ n) := ⟨(1, 1)⟩
instance : Mul (MarkovCarrierState ρ n) := ⟨mul⟩

@[simp] lemma proj_mk (r : ρ) (s : MarkovCountSketch n) :
    proj (r, s) = s := rfl

@[simp] lemma proj_mul (a b : MarkovCarrierState ρ n) :
    proj (a * b) = proj a * proj b := rfl

instance : Monoid (MarkovCarrierState ρ n) where
  one := 1
  mul := (· * ·)
  one_mul := by
    intro a
    cases a
    simp [MarkovCarrierState.mul]
  mul_one := by
    intro a
    cases a
    simp [MarkovCarrierState.mul]
  mul_assoc := by
    intro a b c
    cases a with
    | mk ar as =>
      cases b with
      | mk br bs =>
        cases c with
        | mk cr cs =>
          simp [MarkovCarrierState.mul, mul_assoc]

/-- The theorem-facing oracle ignores the residual carrier and reads only the
projected changepoint count. -/
def oracle (x : MarkovCarrierState ρ n) : ℝ :=
  fstar (n := n) (proj x)

/-- Exact carrier summarizer: identity on the full carrier state. -/
def exactSummarizer : Summarizer (MarkovCarrierState ρ n) :=
  fun x => PMF.pure x

lemma reduce_exact (T : BinTree (MarkovCarrierState ρ n)) :
    reduce (exactSummarizer (ρ := ρ) (n := n)) T = PMF.pure (S T) := by
  induction T with
  | leaf b =>
      simp [reduce, exactSummarizer, S]
  | node TL TR ihL ihR =>
      simp [reduce, exactSummarizer, S, ihL, ihR]

lemma L1_exact (T : BinTree (MarkovCarrierState ρ n)) :
    L1 (exactSummarizer (ρ := ρ) (n := n)) T (oracle (ρ := ρ) (n := n)) := by
  intro b hb
  simp [Eg, exactSummarizer, D, oracle]
  have hsum :
      (∑' z : MarkovCarrierState ρ n,
          (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal *
            dist (oracle (ρ := ρ) (n := n) z) (oracle (ρ := ρ) (n := n) b)) =
        dist (oracle (ρ := ρ) (n := n) b) (oracle (ρ := ρ) (n := n) b) := by
    simpa using
      (tsum_indicator_mul_prop
        (b := b)
        (f := fun z : MarkovCarrierState ρ n =>
          dist (oracle (ρ := ρ) (n := n) z) (oracle (ρ := ρ) (n := n) b)))
  calc
    (∑' z : MarkovCarrierState ρ n,
        (@ite ENNReal (z = b) (Classical.propDecidable (z = b)) 1 0).toReal *
          dist (oracle (ρ := ρ) (n := n) z) (oracle (ρ := ρ) (n := n) b))
        = dist (oracle (ρ := ρ) (n := n) b) (oracle (ρ := ρ) (n := n) b) := hsum
    _ = 0 := by simp

lemma L2_exact (T : BinTree (MarkovCarrierState ρ n)) :
    L2 (exactSummarizer (ρ := ρ) (n := n)) T (oracle (ρ := ρ) (n := n)) := by
  intro p hp
  rcases p with ⟨TL, TR⟩
  have hreduce :
      reduce (exactSummarizer (ρ := ρ) (n := n)) (BinTree.node TL TR) =
        PMF.pure (S (BinTree.node TL TR)) := by
    simpa using (reduce_exact (ρ := ρ) (n := n) (T := BinTree.node TL TR))
  simp [Egu, hreduce, Eg, exactSummarizer, D, oracle]
  have hsum :
      (∑' z : MarkovCarrierState ρ n,
          (@ite ENNReal (z = S (BinTree.node TL TR))
              (Classical.propDecidable (z = S (BinTree.node TL TR))) 1 0).toReal *
            dist
              (oracle (ρ := ρ) (n := n) z)
              (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR)))) =
        dist
          (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR)))
          (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR))) := by
    simpa using
      (tsum_indicator_mul_prop
        (b := S (BinTree.node TL TR))
        (f := fun z : MarkovCarrierState ρ n =>
          dist
            (oracle (ρ := ρ) (n := n) z)
            (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR)))))
  calc
    (∑' z : MarkovCarrierState ρ n,
        (@ite ENNReal (z = S (BinTree.node TL TR))
            (Classical.propDecidable (z = S (BinTree.node TL TR))) 1 0).toReal *
          dist
            (oracle (ρ := ρ) (n := n) z)
            (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR))))
        =
      dist
        (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR)))
        (oracle (ρ := ρ) (n := n) (S (BinTree.node TL TR))) := hsum
    _ = 0 := by simp

theorem root_distortion_zero (T : BinTree (MarkovCarrierState ρ n)) :
    Egu
      (exactSummarizer (ρ := ρ) (n := n))
      (root T)
      (fun z => D (oracle (ρ := ρ) (n := n)) z (S T)) = 0 := by
  exact
    one_pass
      (exactSummarizer (ρ := ρ) (n := n))
      T
      (S T)
      (oracle (ρ := ρ) (n := n))
      rfl
      (L1_exact (ρ := ρ) (n := n) T)
      (L2_exact (ρ := ρ) (n := n) T)

/-- Re-summarizing a carrier state preserves C2 whenever the projection is
unchanged. This is the carrier version of "re-encoding a decoded sketch is
inert on range". -/
theorem L3_of_proj_preserving_reencode
    (reencode : MarkovCarrierState ρ n → MarkovCarrierState ρ n)
    (hproj : ∀ x, proj (reencode x) = proj x) :
    L3
      (fun x : MarkovCarrierState ρ n => PMF.pure (reencode x))
      (oracle (ρ := ρ) (n := n)) := by
  intro Z hZ
  simp [Eg, D, oracle]
  have hsum :
      (∑' z : MarkovCarrierState ρ n,
          (@ite ENNReal (z = reencode Z)
              (Classical.propDecidable (z = reencode Z)) 1 0).toReal *
            dist (oracle (ρ := ρ) (n := n) z) (oracle (ρ := ρ) (n := n) Z)) =
        dist
          (oracle (ρ := ρ) (n := n) (reencode Z))
          (oracle (ρ := ρ) (n := n) Z) := by
    simpa using
      (tsum_indicator_mul_prop
        (b := reencode Z)
        (f := fun z : MarkovCarrierState ρ n =>
          dist (oracle (ρ := ρ) (n := n) z) (oracle (ρ := ρ) (n := n) Z)))
  have horacle :
      oracle (ρ := ρ) (n := n) (reencode Z) =
        oracle (ρ := ρ) (n := n) Z := by
    simpa [oracle, fstar] using
      congrArg (fun s : MarkovCountSketch n => (MarkovCountSketch.count s : ℝ))
        (hproj Z)
  calc
    (∑' z : MarkovCarrierState ρ n,
        (@ite ENNReal (z = reencode Z)
            (Classical.propDecidable (z = reencode Z)) 1 0).toReal *
          dist (oracle (ρ := ρ) (n := n) z) (oracle (ρ := ρ) (n := n) Z))
        =
      dist
        (oracle (ρ := ρ) (n := n) (reencode Z))
        (oracle (ρ := ρ) (n := n) Z) := hsum
    _ = 0 := by simp [horacle]

end MarkovCarrierState

/-!
## Worked carrier example

Instantiate the residual payload with `PUnit`. The projected sketch reproduces
the existing 4-leaf Markov worked example exactly.
-/

abbrev UnitMarkovCarrier := MarkovCarrierState PUnit 2

def carrierLeafA : UnitMarkovCarrier := (PUnit.unit, leafA)
def carrierLeafB : UnitMarkovCarrier := (PUnit.unit, leafB)
def carrierLeafC : UnitMarkovCarrier := (PUnit.unit, leafC)
def carrierLeafD : UnitMarkovCarrier := (PUnit.unit, leafD)

def carrierExampleTree : BinTree UnitMarkovCarrier :=
  BinTree.node
    (BinTree.node (BinTree.leaf carrierLeafA) (BinTree.leaf carrierLeafB))
    (BinTree.node (BinTree.leaf carrierLeafC) (BinTree.leaf carrierLeafD))

theorem carrierExampleTree_proj :
    MarkovCarrierState.proj (S carrierExampleTree) = S exampleTree := by
  simp [
    carrierExampleTree,
    exampleTree,
    carrierLeafA,
    carrierLeafB,
    carrierLeafC,
    carrierLeafD,
    MarkovCarrierState.proj,
    MarkovCarrierState.mul,
    S,
  ]

theorem carrierExampleTree_oracle_correct :
    MarkovCarrierState.oracle (ρ := PUnit) (n := 2) (S carrierExampleTree) = 2 := by
  simp [MarkovCarrierState.oracle, carrierExampleTree_proj, tree_oracle_correct]

end FormalProofs.OPT
