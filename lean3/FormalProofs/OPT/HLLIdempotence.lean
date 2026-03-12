import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.SketchSummaryOperators

/-!
# FormalProofs/OPT/HLLIdempotence.lean

This file disentangles two different notions of "idempotence" that are easy to
conflate for mergeable sketches such as HyperLogLog:

1. **Sketch-algebra idempotence**: `merge s s = s`
2. **OPS / L3 re-summary idempotence**: once a theorem-domain summary is on the
   range of the summarizer, re-summarizing it is inert

For HLL-like register states, the algebraic merge really is idempotent:
registers merge by pointwise `max`, and `max r r = r`.

However, that algebraic fact alone does **not** imply OPS `L3`. To get `L3` for
the induced deterministic summary operator `decode ∘ encode`, one also needs a
re-encoding fixed-point property `encode (decode s) = s` on sketch states.

The file therefore proves:

- a general bridge `ReencodeExact -> L3`;
- an approximate bridge from audited on-range re-summary stability to `L3ε`;
- a counterexample showing `merge s s = s` alone is insufficient for `L3`;
- an HLL-style finite-register model where the merge algebra is exactly
  idempotent, associative, and commutative.
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

section ResummaryIdempotence

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Pointwise fixed-point property for a deterministic summary operator. -/
def SummaryFixedPoint (s : Strings → Strings) : Prop :=
  ∀ x, s (s x) = s x

/-- Sketch-level merge idempotence: merging a state with itself does not move it. -/
def MergeIdempotent (op : SketchOperator Strings Sketch) : Prop :=
  ∀ s, op.merge s s = s

/-- Re-encoding exactness: every decoded sketch re-encodes to the same sketch state. -/
def ReencodeExact (op : SketchOperator Strings Sketch) : Prop :=
  ∀ s, op.encode (op.decode s) = s

/-- A deterministic fixed-point summary operator satisfies OPS `L3` for any oracle map. -/
theorem L3_of_summary_fixed_point
    (s : Strings → Strings) (fstar : Strings → Y)
    (h_fix : SummaryFixedPoint s) :
    L3 (deterministicSummarizer s) fstar := by
  intro Z hZ
  rcases hZ with ⟨x, hx⟩
  have hZ_fix : s Z = Z := by
    have hx' : Z = s x := by
      simpa [deterministicSummarizer] using hx
    calc
      s Z = s (s x) := by rw [hx']
      _ = s x := h_fix x
      _ = Z := hx'.symm
  rw [Eg_deterministic_summaryOp]
  rw [hZ_fix]
  simp [D]

/-- Exact re-encoding makes the induced deterministic summary operator a fixed point. -/
theorem summaryFixedPoint_of_reencodeExact
    (op : SketchOperator Strings Sketch)
    (h_reencode : ReencodeExact op) :
    SummaryFixedPoint (summaryFromSketch op) := by
  intro x
  have h :=
    congrArg op.decode (h_reencode (op.encode x))
  simpa [summaryFromSketch] using h

/-- Exact re-encoding is sufficient for OPS `L3` on the induced summary operator. -/
theorem L3_of_reencodeExact
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_reencode : ReencodeExact op) :
    L3 (deterministicSummarizer (summaryFromSketch op)) fstar := by
  exact L3_of_summary_fixed_point
    (s := summaryFromSketch op)
    (fstar := fstar)
    (summaryFixedPoint_of_reencodeExact (op := op) h_reencode)

/-- Approximate on-range re-summary stability stated at the level of violation
probabilities. -/
def OnRangeViolationBound (g : Summarizer Strings) (fstar : Strings → Y)
    (ε : ℝ) : Prop :=
  ∀ Z, InRange g Z → ViolationProb fstar (g Z) Z ≤ ε

/-- `pIdemp` at a pure root summary is just the usual re-summary violation
probability at that summary. -/
theorem pIdemp_pure_eq
    (g : Summarizer Strings) (fstar : Strings → Y) (z : Strings) :
    pIdemp g fstar (PMF.pure z) = ViolationProb fstar (g z) z := by
  unfold pIdemp ViolationProb Exp
  simpa using
    (tsum_indicator_mul_prop
      (b := z)
      (f := fun z => ∑' w, (g z w).toReal * violationInd fstar w z))

/-- Uniform on-range re-summary control implies the approximate idempotence law
`L3ε` for any deterministic summary operator on any tree. -/
theorem L3ε_of_onRangeViolationBound_deterministic
    (s : Strings → Strings) (T : BinTree Strings) (fstar : Strings → Y) (ε : ℝ)
    (h_onRange : OnRangeViolationBound (deterministicSummarizer s) fstar ε) :
    L3ε (deterministicSummarizer s) T fstar ε := by
  unfold L3ε
  rw [reduce_deterministic_eq_pure]
  rw [pIdemp_pure_eq]
  apply h_onRange
  have hSupp :
      reduceDeterministic s T ∈
        (reduce (deterministicSummarizer s) T).support := by
    rw [reduce_deterministic_eq_pure]
    simp
  exact
    (reduce_support_in_range (g := deterministicSummarizer s) (T := T))
      (reduceDeterministic s T) hSupp

end ResummaryIdempotence

section MergeIdempotenceCounterexample

/-- A toy sketch operator with idempotent merge but non-idempotent decode/encode summary. -/
def succMaxOperator : SketchOperator Nat Nat where
  encode := fun n => n
  merge := Nat.max
  decode := Nat.succ

/-- The toy operator's merge is idempotent. -/
theorem succMax_merge_idempotent :
    MergeIdempotent succMaxOperator := by
  intro s
  simp [MergeIdempotent, succMaxOperator]

/-- But the induced theorem-domain summary operator is not a fixed point. -/
theorem succMax_not_summaryFixedPoint :
    ¬ SummaryFixedPoint (summaryFromSketch succMaxOperator) := by
  intro h_fix
  have h := h_fix 0
  norm_num [SummaryFixedPoint, summaryFromSketch, succMaxOperator] at h

/-- Consequently, merge idempotence alone does not imply OPS `L3`. -/
theorem succMax_not_L3 :
    ¬ L3
      (deterministicSummarizer (summaryFromSketch succMaxOperator))
      (fun n : Nat => (n : ℝ)) := by
  intro hL3
  let Z : Nat := 1
  have hInRange :
      InRange
        (deterministicSummarizer (summaryFromSketch succMaxOperator))
        Z := by
    refine ⟨0, ?_⟩
    simp [InRange, deterministicSummarizer, summaryFromSketch, succMaxOperator, Z]
  have h0 := hL3 Z hInRange
  rw [Eg_deterministic_summaryOp] at h0
  norm_num [Z, D, summaryFromSketch, succMaxOperator, Real.dist_eq] at h0

end MergeIdempotenceCounterexample

/-- HLL-style finite register state: a vector of natural-valued registers. -/
structure HLLState (m : ℕ) where
  regs : Fin m → Nat

namespace HLLState

variable {m : ℕ}

@[ext] theorem ext {a b : HLLState m} (h : ∀ i, a.regs i = b.regs i) : a = b := by
  cases a with
  | mk ra =>
      cases b with
      | mk rb =>
          have hfun : ra = rb := funext h
          cases hfun
          rfl

/-- Empty HLL state: all registers are zero. -/
def zero (m : ℕ) : HLLState m :=
  ⟨fun _ => 0⟩

/-- HLL merge: pointwise maximum of register values. -/
def merge (a b : HLLState m) : HLLState m :=
  ⟨fun i => max (a.regs i) (b.regs i)⟩

instance : One (HLLState m) := ⟨zero m⟩

instance : Mul (HLLState m) := ⟨merge⟩

@[simp] theorem regs_one (i : Fin m) : (1 : HLLState m).regs i = 0 := rfl

@[simp] theorem regs_mul (a b : HLLState m) (i : Fin m) :
    (a * b).regs i = max (a.regs i) (b.regs i) := rfl

instance : Monoid (HLLState m) where
  one := 1
  mul := (· * ·)
  one_mul := by
    intro a
    ext i
    simp [zero, merge]
  mul_one := by
    intro a
    ext i
    simp [zero, merge]
  mul_assoc := by
    intro a b c
    ext i
    simp [merge, max_assoc]

/-- HLL merge is commutative. -/
theorem mul_comm (a b : HLLState m) : a * b = b * a := by
  ext i
  simp [merge, max_comm]

/-- HLL merge is exactly idempotent. -/
theorem mul_self (a : HLLState m) : a * a = a := by
  ext i
  simp [merge]

/-- Any HLL merge tree over already-encoded register states is exact. -/
theorem reduceDeterministic_id (T : BinTree (HLLState m)) :
    reduceDeterministic (fun x : HLLState m => x) T = S T := by
  induction T with
  | leaf b =>
      rfl
  | node TL TR ihL ihR =>
      simp [reduceDeterministic, S, ihL, ihR]

end HLLState

section HLLRegisterOperator

variable {m : ℕ}
variable {Y : Type*} [PseudoMetricSpace Y]

/-- The theorem-domain identity operator on HLL register states. -/
abbrev hllRegisterOperator (m : ℕ) :
    SketchOperator (HLLState m) (HLLState m) :=
  identitySketchOperator (Strings := HLLState m)

/-- The HLL register operator has exact re-encoding. -/
theorem hllRegisterOperator_reencodeExact :
    ReencodeExact (hllRegisterOperator m) := by
  intro s
  rfl

/-- The HLL register operator has exactly idempotent merge. -/
theorem hllRegisterOperator_merge_idempotent :
    MergeIdempotent (hllRegisterOperator m) := by
  intro s
  simpa [hllRegisterOperator] using HLLState.mul_self s

/-- Re-summarizing an already-decoded HLL register state is literally inert. -/
theorem hllRegisterOperator_summary_fixed :
    SummaryFixedPoint (summaryFromSketch (hllRegisterOperator m)) := by
  exact summaryFixedPoint_of_reencodeExact
    (op := hllRegisterOperator m)
    hllRegisterOperator_reencodeExact

/-- Therefore HLL register states satisfy OPS `L3` when the theorem-domain objects
are already register states. -/
theorem hllRegisterOperator_L3 (fstar : HLLState m → Y) :
    L3
      (deterministicSummarizer (summaryFromSketch (hllRegisterOperator m)))
      fstar := by
  exact L3_of_reencodeExact
    (op := hllRegisterOperator m)
    (fstar := fstar)
    hllRegisterOperator_reencodeExact

/-- Any tree reduction over already-encoded HLL register states is exact at the root. -/
theorem hllRegisterOperator_root_exact (T : BinTree (HLLState m)) :
    sketchSummary (hllRegisterOperator m) T = S T := by
  calc
    sketchSummary (hllRegisterOperator m) T =
        reduceDeterministic (summaryFromSketch (hllRegisterOperator m)) T := by
          exact sketchSummary_eq_reduceDeterministic
            (op := hllRegisterOperator m)
            (identitySketch_summary_compatible (Strings := HLLState m))
            T
    _ = reduceDeterministic (fun x : HLLState m => x) T := by
          rfl
    _ = S T := HLLState.reduceDeterministic_id (m := m) (T := T)

end HLLRegisterOperator

end FormalProofs.OPT
