import Mathlib

/-!
# FormalProofs/OPT/OracleMeasurable.lean

Lightweight oracle-measurability predicates and closure lemmas.

These are intended as reusable building blocks for OPT/DSL proofs,
without pulling in the full preference-learning stack.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace OPT

/-!
## Oracle-Measurable Functions
-/

/-- A function depends on `x` only through the oracle value `fstar x`. -/
def OracleMeasurable {Strings Y β : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (g : Strings → β) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → g x = g x'

lemma oracleMeasurable_const {Strings Y β : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (c : β) : OracleMeasurable fstar (fun _ => c) := by
  intro _ _ _
  rfl

lemma oracleMeasurable_comp {Strings Y β γ : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (g : Strings → β) (h : β → γ)
    (hg : OracleMeasurable fstar g) :
    OracleMeasurable fstar (fun x => h (g x)) := by
  intro x x' hdist
  simp [hg x x' hdist]

lemma oracleMeasurable_prod {Strings Y β γ : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (g₁ : Strings → β) (g₂ : Strings → γ)
    (h₁ : OracleMeasurable fstar g₁) (h₂ : OracleMeasurable fstar g₂) :
    OracleMeasurable fstar (fun x => (g₁ x, g₂ x)) := by
  intro x x' hdist
  simp [h₁ x x' hdist, h₂ x x' hdist]

/-!
## Oracle-Measurable Losses
-/

/-- Loss oracle-measurability (pointwise in the auxiliary argument). -/
def OracleMeasurableLoss {Strings Y A : Type*} [PseudoMetricSpace Y]
    (loss : Strings → A → ℝ) (fstar : Strings → Y) : Prop :=
  ∀ x x' a, dist (fstar x) (fstar x') = 0 → loss x a = loss x' a

lemma oracleMeasurableLoss_of_factor {Strings Y A : Type*} [PseudoMetricSpace Y]
    (loss : Strings → A → ℝ) (fstar : Strings → Y)
    (L : Y → A → ℝ) (hL : ∀ x a, loss x a = L (fstar x) a)
    (hL_oracle : ∀ y y' a, dist y y' = 0 → L y a = L y' a) :
    OracleMeasurableLoss loss fstar := by
  intro x x' a hdist
  calc
    loss x a = L (fstar x) a := hL x a
    _ = L (fstar x') a := hL_oracle _ _ a hdist
    _ = loss x' a := (hL x' a).symm

/-!
## Oracle-Indexed Generators
-/

/-- A generator depends on `x` only through `fstar x`. -/
def OracleIndexed {Strings Y A : Type*} [PseudoMetricSpace Y]
    (gen : Strings → PMF A) (fstar : Strings → Y) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → gen x = gen x'

lemma oracleIndexed_const {Strings Y A : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (p : PMF A) : OracleIndexed (fun _ => p) fstar := by
  intro _ _ _
  rfl

lemma oracleIndexed_map {Strings Y A B : Type*} [PseudoMetricSpace Y]
    (gen : Strings → PMF A) (fstar : Strings → Y) (f : A → B)
    (hgen : OracleIndexed gen fstar) :
    OracleIndexed (fun x => (gen x).map f) fstar := by
  intro x x' hdist
  simp [hgen x x' hdist]

end OPT
