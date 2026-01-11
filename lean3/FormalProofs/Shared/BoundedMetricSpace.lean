/-
FormalProofs/BoundedMetricSpace.lean

Bounded Pseudo Metric Space typeclass for sound summability proofs.

The paper naturally assumes bounded oracle spaces (finite action spaces, bounded
rewards). This typeclass makes the assumption explicit, enabling:
1. Elimination of the unsound `PMF.summable_coe_real_mul` axiom
2. Automatic derivation of boundedness hypotheses
3. Cleaner theorem statements without repeated `hbound` parameters

## Usage

Replace `[PseudoMetricSpace Y]` with `[BoundedPseudoMetricSpace Y]` in theorem
signatures. The bound `diameterBound` is then available for summability proofs.
-/

import Mathlib

set_option linter.mathlibStandardSet false

/-!
## Bounded Pseudo Metric Space
-/

/-- A pseudo metric space where all distances are bounded by a fixed constant.

This is the key typeclass that enables sound summability for PMF expectations
over distortion functions. The bound `diameterBound` represents the maximum
distance between any two points. -/
class BoundedPseudoMetricSpace (α : Type*) extends PseudoMetricSpace α where
  /-- The diameter bound for the space -/
  diameterBound : ℝ
  /-- The bound is positive -/
  diameterBound_pos : 0 < diameterBound
  /-- All distances are bounded by the diameter -/
  dist_le_diameterBound : ∀ x y : α, dist x y ≤ diameterBound

namespace BoundedPseudoMetricSpace

variable {α : Type*} [BoundedPseudoMetricSpace α]

/-- Non-negative diameter bound -/
lemma diameterBound_nonneg : 0 ≤ diameterBound (α := α) :=
  le_of_lt diameterBound_pos

/-- Absolute value of distance is bounded -/
lemma abs_dist_le (x y : α) : |dist x y| ≤ diameterBound (α := α) := by
  rw [abs_of_nonneg dist_nonneg]
  exact dist_le_diameterBound x y

/-- Distance is bounded (convenience lemma) -/
lemma dist_le (x y : α) : dist x y ≤ diameterBound (α := α) :=
  dist_le_diameterBound x y

end BoundedPseudoMetricSpace

/-!
## Unit-Bounded Pseudo Metric Space

Specialization for spaces with diameter ≤ 1, common for:
- {0,1}-valued violation indicators
- Normalized reward functions
- Probability metrics
-/

/-- A bounded pseudo metric space with diameter exactly 1. -/
class UnitBoundedPseudoMetricSpace (α : Type*) extends BoundedPseudoMetricSpace α where
  diameterBound_eq_one : diameterBound = 1

namespace UnitBoundedPseudoMetricSpace

variable {α : Type*} [inst : UnitBoundedPseudoMetricSpace α]

/-- Distance is bounded by 1 in unit-bounded spaces -/
lemma dist_le_one (x y : α) : dist x y ≤ 1 := by
  have h := @diameterBound_eq_one α inst
  rw [← h]
  exact BoundedPseudoMetricSpace.dist_le x y

end UnitBoundedPseudoMetricSpace

/-!
## Construction Helpers

For creating bounded metric space instances from explicit bounds.
-/

/-- Construct a bounded pseudo metric space from an explicit bound. -/
def BoundedPseudoMetricSpace.ofBound {α : Type*} [PseudoMetricSpace α] (M : ℝ) (hM : 0 < M)
    (hbound : ∀ x y : α, dist x y ≤ M) : BoundedPseudoMetricSpace α where
  diameterBound := M
  diameterBound_pos := hM
  dist_le_diameterBound := hbound

/-- Construct a unit-bounded pseudo metric space from a bound ≤ 1 proof. -/
def UnitBoundedPseudoMetricSpace.ofBoundOne {α : Type*} [PseudoMetricSpace α]
    (hbound : ∀ x y : α, dist x y ≤ 1) : UnitBoundedPseudoMetricSpace α where
  diameterBound := 1
  diameterBound_pos := one_pos
  dist_le_diameterBound := hbound
  diameterBound_eq_one := rfl

/-!
## Bounded Metric Space (proper MetricSpace)

For theorems that need `dist = 0 → eq` (e.g., showing DPO losses are equal when
distortion is zero), we need the full `MetricSpace` axioms, not just `PseudoMetricSpace`.
-/

/-- A metric space where all distances are bounded by a fixed constant.

This is the proper metric space version (dist = 0 ↔ eq), needed for theorems
that derive equality from zero distance. -/
class BoundedMetricSpace (α : Type*) extends MetricSpace α where
  /-- The diameter bound for the space -/
  diameterBound : ℝ
  /-- The bound is positive -/
  diameterBound_pos : 0 < diameterBound
  /-- All distances are bounded by the diameter -/
  dist_le_diameterBound : ∀ x y : α, dist x y ≤ diameterBound

namespace BoundedMetricSpace

variable {α : Type*} [BoundedMetricSpace α]

/-- Non-negative diameter bound -/
lemma diameterBound_nonneg : 0 ≤ diameterBound (α := α) :=
  le_of_lt diameterBound_pos

/-- Distance is bounded (convenience lemma) -/
lemma dist_le (x y : α) : dist x y ≤ diameterBound (α := α) :=
  dist_le_diameterBound x y

/-- Absolute value of distance is bounded -/
lemma abs_dist_le (x y : α) : |dist x y| ≤ diameterBound (α := α) := by
  rw [abs_of_nonneg dist_nonneg]
  exact dist_le_diameterBound x y

end BoundedMetricSpace

/-- A BoundedMetricSpace is automatically a BoundedPseudoMetricSpace. -/
instance BoundedMetricSpace.toBoundedPseudoMetricSpace {α : Type*} [inst : BoundedMetricSpace α] :
    BoundedPseudoMetricSpace α where
  diameterBound := inst.diameterBound
  diameterBound_pos := inst.diameterBound_pos
  dist_le_diameterBound := inst.dist_le_diameterBound

/-- Construct a bounded metric space from an explicit bound. -/
def BoundedMetricSpace.ofBound {α : Type*} [MetricSpace α] (M : ℝ) (hM : 0 < M)
    (hbound : ∀ x y : α, dist x y ≤ M) : BoundedMetricSpace α where
  diameterBound := M
  diameterBound_pos := hM
  dist_le_diameterBound := hbound
