import FormalProofs.OPT.ExpectationTheory

/-!
# FormalProofs/GlobalAssumptions.lean

## Paper Reference: Sections 4-5 (Global Assumptions and Derivations)

This file formalizes the global oracle preservation assumptions and proves they
imply the local laws (L1, L2, L3) for any tree.

### Key Paper Propositions

- **Proposition 1** (`prop1_A1_implies_L1`): A1 (global sufficiency) implies L1
- **Proposition 2** (`prop2_A1_A2_A3_implies_L2`): A1 + A2 + A3 implies L2
- **Strict mergeable limit** (`prop3_mergeable_classical`): A1 + A2 + strict
  oracle-output A3 imply an oracle-level mergeable summary

### Global Assumptions (Lean) vs Conditions (Paper)

The Lean formalization uses STRONGER global assumptions that imply
the paper's local conditions:

| Lean                              | Paper                        | Relationship    |
|-----------------------------------|------------------------------|-----------------|
| A1_global: `∀ z, D(g z, z) = 0`   | C1: sufficiency on leaves    | A1_global → C1  |
| A2_global: two-route identity     | C3: merge consistency (local)| A2_global → C3  |
| A3_global: oracle-output merge exists | strict homomorphism special case | -          |

**Naming correspondence** (see LocalLaws.lean for details):
| Lean Name | Paper Name | Description          |
|-----------|------------|----------------------|
| L1        | C1         | Leaf Sufficiency     |
| L2        | C3         | Merge Consistency    |
| L3        | C2         | Idempotence/On-Range |

### Key Derivation Theorems

- `A1_implies_L1`: A1_global → L1 for any tree
- `A1_A2_A3_implies_L2`: A1 + A2 + A3 → L2 for any tree
- `A1_implies_L3_for_deterministic`: A1 → L3 for deterministic summarizers

### Why Different?

The paper's conditions (C1, C2, C3) are the minimal LOCAL assumptions needed
for the main preservation theorems. The Lean formalization uses GLOBAL
assumptions (A1, A2, A3) which are easier to verify in strict deterministic
settings and immediately imply the local laws for any tree structure.  The A3
used here is stronger than the classical state-level sketch condition: it
requires oracle values, not just hidden sketch states, to admit a merge.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Global Assumptions
-/

/-- A1: Global Sufficiency - distortion is 0 for all strings, not just leaves -/
def A1_global (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ z : Strings, D fstar (g z) z = 0

/-- A2: Global Compatibility - two-route identity -/
def A2_global (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ u v : Strings, D fstar (u * v) (g (g u * g v)) = 0

/-- A3: strict oracle-output merge.

There exists a merge operation on oracle values that agrees with the summary
merge route and respects zero-distance equality.  This is intentionally stronger
than classical mergeable summaries, where bounded sketch states can carry
information unavailable in the final scalar/task readout. -/
def A3_global (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∃ M : Y → Y → Y,
    (∀ u v : Strings, dist (fstar (g (g u * g v))) (M (fstar (g u)) (fstar (g v))) = 0) ∧
    (∀ y₁ y₁' y₂ y₂' : Y, dist y₁ y₁' = 0 → dist y₂ y₂' = 0 →
      dist (M y₁ y₂) (M y₁' y₂') = 0)

/-- InOracleRange: y is in the range of f* ∘ g -/
def InOracleRange (g : Strings → Strings) (fstar : Strings → Y) (y : Y) : Prop :=
  ∃ z : Strings, fstar (g z) = y

/-!
## Global Preservation Typeclass

Bundle A1_global, A2_global, A3_global into a single typeclass for cleaner theorem signatures.
This also enables automatic derivation of LocalLawsBundle for any tree.
-/

/-- A summarizer satisfies all global preservation assumptions.

This typeclass bundles the three global assumptions (A1, A2, A3) that together
imply local laws for any tree structure. It enables:
1. Cleaner theorem signatures (3 hypotheses → 1 typeclass constraint)
2. Automatic derivation of LocalLawsBundle via `toLocalLawsBundle`
3. Consistent API for summarizer properties -/
class GlobalPreservation (g : Strings → Strings) (fstar : Strings → Y) where
  /-- A1: Global sufficiency - distortion is 0 for all strings -/
  a1 : A1_global g fstar
  /-- A2: Global compatibility - two-route identity -/
  a2 : A2_global g fstar
  /-- A3: strict oracle-output merge function exists. -/
  a3 : A3_global g fstar

namespace GlobalPreservation

variable {g : Strings → Strings} {fstar : Strings → Y}

/-- Extract A1_global from typeclass -/
lemma get_A1 [inst : GlobalPreservation g fstar] : A1_global g fstar := inst.a1

/-- Extract A2_global from typeclass -/
lemma get_A2 [inst : GlobalPreservation g fstar] : A2_global g fstar := inst.a2

/-- Extract A3_global from typeclass -/
lemma get_A3 [inst : GlobalPreservation g fstar] : A3_global g fstar := inst.a3

end GlobalPreservation

/-!
## Helper Lemmas
-/

/-- D is symmetric -/
lemma D_symm (fstar : Strings → Y) (x y : Strings) : D fstar x y = D fstar y x := by
  unfold D
  exact dist_comm (fstar x) (fstar y)

/-!
## Concatenation Congruence
-/

/-- Helper: If g preserves oracle values (A1) and D(x, x') = 0, then dist(f*(g x), f*(g x')) = 0.
This is the core triangle-inequality argument used twice in concat_congruence. -/
private lemma dist_fstar_g_eq_zero (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (x x' : Strings) (hx : D fstar x x' = 0) :
    dist (fstar (g x)) (fstar (g x')) = 0 := by
  have hgx : D fstar (g x) x = 0 := hA1 x
  have hgx' : D fstar (g x') x' = 0 := hA1 x'
  unfold D at hgx hgx' hx
  apply le_antisymm _ dist_nonneg
  calc dist (fstar (g x)) (fstar (g x'))
    ≤ dist (fstar (g x)) (fstar x) + dist (fstar x) (fstar (g x')) := dist_triangle _ _ _
    _ ≤ dist (fstar (g x)) (fstar x) +
        (dist (fstar x) (fstar x') + dist (fstar x') (fstar (g x'))) := by
        linarith [dist_triangle (fstar x) (fstar x') (fstar (g x'))]
    _ = 0 + (0 + 0) := by rw [hgx, hx, dist_comm (fstar x') (fstar (g x')), hgx']
    _ = 0 := by ring

/-- If f*(a) ≈ f*(a') and f*(b) ≈ f*(b'), then f*(a⊕b) ≈ f*(a'⊕b') -/
theorem concat_congruence (g : Strings → Strings) (fstar : Strings → Y)
  (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar)
  (a a' b b' : Strings) (ha : D fstar a a' = 0) (hb : D fstar b b' = 0) :
  D fstar (a * b) (a' * b') = 0 := by
    obtain ⟨M, hM1, hM2⟩ := hA3
    set U := g (g a * g b) with hU_def
    set U' := g (g a' * g b') with hU'_def
    have h_tri : D fstar (a * b) (a' * b') ≤
                 D fstar (a * b) U + D fstar U U' + D fstar U' (a' * b') := by
      calc D fstar (a * b) (a' * b')
        ≤ D fstar (a * b) U + D fstar U (a' * b') := D_triangle fstar (a * b) U (a' * b')
        _ ≤ D fstar (a * b) U + (D fstar U U' + D fstar U' (a' * b')) := by
            linarith [D_triangle fstar U U' (a' * b')]
        _ = D fstar (a * b) U + D fstar U U' + D fstar U' (a' * b') := by ring
    have h1 : D fstar (a * b) U = 0 := hA2 a b
    have h3 : D fstar U' (a' * b') = 0 := by rw [D_symm]; exact hA2 a' b'
    have hga : dist (fstar (g a)) (fstar (g a')) = 0 := dist_fstar_g_eq_zero g fstar hA1 a a' ha
    have hgb : dist (fstar (g b)) (fstar (g b')) = 0 := dist_fstar_g_eq_zero g fstar hA1 b b' hb
    have hM_eq : dist (M (fstar (g a)) (fstar (g b))) (M (fstar (g a')) (fstar (g b'))) = 0 :=
      hM2 _ _ _ _ hga hgb
    have h2 : D fstar U U' = 0 := by
      have hU_M : dist (fstar U) (M (fstar (g a)) (fstar (g b))) = 0 := by
        calc dist (fstar U) (M (fstar (g a)) (fstar (g b)))
          = dist (fstar (g (g a * g b))) (M (fstar (g a)) (fstar (g b))) := rfl
          _ = 0 := hM1 a b
      have hU'_M : dist (fstar U') (M (fstar (g a')) (fstar (g b'))) = 0 := hM1 a' b'
      unfold D
      apply le_antisymm _ dist_nonneg
      calc dist (fstar U) (fstar U')
        ≤ dist (fstar U) (M (fstar (g a)) (fstar (g b))) +
          dist (M (fstar (g a)) (fstar (g b))) (fstar U') := dist_triangle _ _ _
        _ ≤ dist (fstar U) (M (fstar (g a)) (fstar (g b))) +
          (dist (M (fstar (g a)) (fstar (g b))) (M (fstar (g a')) (fstar (g b'))) +
           dist (M (fstar (g a')) (fstar (g b'))) (fstar U')) := by
            linarith [dist_triangle (M (fstar (g a)) (fstar (g b)))
                                    (M (fstar (g a')) (fstar (g b')))
                                    (fstar U')]
        _ = 0 + (0 + 0) := by rw [hU_M, hM_eq, dist_comm, hU'_M]
        _ = 0 := by ring
    have h_nonneg : 0 ≤ D fstar (a * b) (a' * b') := dist_nonneg
    linarith

/-!
## Oracle Merge Operation
-/

/-- Extract the merge function from A3 -/
noncomputable def oracleMerge {g : Strings → Strings} {fstar : Strings → Y}
    (hA3 : A3_global g fstar) : Y → Y → Y := hA3.choose

/-- The oracle merge satisfies A3 properties -/
lemma oracleMerge_spec (g : Strings → Strings) (fstar : Strings → Y) (hA3 : A3_global g fstar) :
  (∀ u v : Strings, dist (fstar (g (g u * g v))) (oracleMerge hA3 (fstar (g u)) (fstar (g v))) = 0) ∧
  (∀ y₁ y₁' y₂ y₂' : Y, dist y₁ y₁' = 0 → dist y₂ y₂' = 0 →
    dist (oracleMerge hA3 y₁ y₂) (oracleMerge hA3 y₁' y₂') = 0) :=
  hA3.choose_spec

/-!
## Merge Algebraic Properties
-/

/-- Merge is associative on the oracle range -/
theorem merge_assoc (g : Strings → Strings) (fstar : Strings → Y)
  (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar)
  (y₁ y₂ y₃ : Y)
  (hy₁ : InOracleRange g fstar y₁)
  (hy₂ : InOracleRange g fstar y₂)
  (hy₃ : InOracleRange g fstar y₃) :
  dist (oracleMerge hA3 (oracleMerge hA3 y₁ y₂) y₃)
       (oracleMerge hA3 y₁ (oracleMerge hA3 y₂ y₃)) = 0 := by
    obtain ⟨a, ha⟩ := hy₁
    obtain ⟨b, hb⟩ := hy₂
    obtain ⟨c, hc⟩ := hy₃
    have ⟨hM1, hM2⟩ := oracleMerge_spec g fstar hA3
    have h_string_assoc : (a * b) * c = a * (b * c) := mul_assoc a b c
    set M := oracleMerge hA3
    -- LHS: (y₁ ⊙ y₂) ⊙ y₃
    have h12 : dist (M y₁ y₂) (fstar (g (g a * g b))) = 0 := by
      rw [← ha, ← hb]
      have := hM1 a b
      rw [dist_comm]
      exact this
    have hLHS_step1 : dist (M (M y₁ y₂) y₃) (fstar (g (g (g a * g b) * g c))) = 0 := by
      have h_resp := hM2 (M y₁ y₂) (fstar (g (g a * g b))) y₃ y₃ h12 (dist_self _)
      -- Rewrite y₃ to fstar (g c) in h_resp
      rw [← hc] at h_resp
      apply le_antisymm _ dist_nonneg
      calc dist (M (M y₁ y₂) y₃) (fstar (g (g (g a * g b) * g c)))
        ≤ dist (M (M y₁ y₂) y₃) (M (fstar (g (g a * g b))) (fstar (g c))) +
          dist (M (fstar (g (g a * g b))) (fstar (g c))) (fstar (g (g (g a * g b) * g c))) :=
            dist_triangle _ _ _
        _ = 0 + 0 := by
            rw [← hc, h_resp]
            have := hM1 (g a * g b) c
            rw [dist_comm, this]
        _ = 0 := by ring
    have hLHS_step2 : dist (fstar (g (g (g a * g b) * g c))) (fstar ((a * b) * c)) = 0 := by
      have hAB' : D fstar (g (g a * g b)) (a * b) = 0 := by rw [D_symm]; exact hA2 a b
      have hgc : D fstar (g c) (g c) = 0 := dist_self _
      have step1 : D fstar (g (g (g a * g b) * g c)) ((g (g a * g b)) * g c) = 0 := by
        exact hA1 ((g (g a * g b)) * g c)
      have step2 : D fstar ((g (g a * g b)) * g c) ((a * b) * g c) = 0 :=
        concat_congruence g fstar hA1 hA2 hA3 (g (g a * g b)) (a*b) (g c) (g c) hAB' hgc
      have hc' : D fstar (g c) c = 0 := hA1 c
      have hab_self : D fstar (a*b) (a*b) = 0 := dist_self _
      have step3 : D fstar ((a*b) * g c) ((a*b) * c) = 0 :=
        concat_congruence g fstar hA1 hA2 hA3 (a*b) (a*b) (g c) c hab_self hc'
      unfold D at step1 step2 step3
      apply le_antisymm _ dist_nonneg
      calc dist (fstar (g (g (g a * g b) * g c))) (fstar ((a * b) * c))
        ≤ dist (fstar (g (g (g a * g b) * g c))) (fstar ((g (g a * g b)) * g c)) +
          dist (fstar ((g (g a * g b)) * g c)) (fstar ((a * b) * c)) := dist_triangle _ _ _
        _ ≤ dist (fstar (g (g (g a * g b) * g c))) (fstar ((g (g a * g b)) * g c)) +
          (dist (fstar ((g (g a * g b)) * g c)) (fstar ((a * b) * g c)) +
           dist (fstar ((a * b) * g c)) (fstar ((a * b) * c))) := by
             linarith [dist_triangle (fstar ((g (g a * g b)) * g c))
                                     (fstar ((a * b) * g c))
                                     (fstar ((a * b) * c))]
        _ = 0 + (0 + 0) := by rw [step1, step2, step3]
        _ = 0 := by ring
    -- RHS path
    have hRHS_step1 : dist (M y₁ (M y₂ y₃)) (fstar (g (g a * g (g b * g c)))) = 0 := by
      have h23 : dist (M y₂ y₃) (fstar (g (g b * g c))) = 0 := by
        rw [← hb, ← hc]
        have := hM1 b c
        rw [dist_comm]
        exact this
      have h_resp := hM2 y₁ y₁ (M y₂ y₃) (fstar (g (g b * g c))) (dist_self _) h23
      apply le_antisymm _ dist_nonneg
      calc dist (M y₁ (M y₂ y₃)) (fstar (g (g a * g (g b * g c))))
        ≤ dist (M y₁ (M y₂ y₃)) (M y₁ (fstar (g (g b * g c)))) +
          dist (M y₁ (fstar (g (g b * g c)))) (fstar (g (g a * g (g b * g c)))) :=
            dist_triangle _ _ _
        _ = 0 + dist (M y₁ (fstar (g (g b * g c)))) (fstar (g (g a * g (g b * g c)))) := by
            rw [h_resp]
        _ = 0 + 0 := by
            have := hM1 a (g b * g c)
            rw [← ha]
            rw [dist_comm, this]
        _ = 0 := by ring
    have hRHS_step2 : dist (fstar (g (g a * g (g b * g c)))) (fstar (a * (b * c))) = 0 := by
      have hBC : D fstar (g (g b * g c)) (b * c) = 0 := by rw [D_symm]; exact hA2 b c
      have hga_self : D fstar (g a) (g a) = 0 := dist_self _
      have step1 : D fstar (g (g a * g (g b * g c))) (g a * g (g b * g c)) = 0 := by
        exact hA1 (g a * g (g b * g c))
      have step2 : D fstar (g a * g (g b * g c)) (g a * (b * c)) = 0 :=
        concat_congruence g fstar hA1 hA2 hA3 (g a) (g a) (g (g b * g c)) (b*c) hga_self hBC
      have ha' : D fstar (g a) a = 0 := hA1 a
      have hbc_self : D fstar (b*c) (b*c) = 0 := dist_self _
      have step3 : D fstar (g a * (b * c)) (a * (b * c)) = 0 :=
        concat_congruence g fstar hA1 hA2 hA3 (g a) a (b*c) (b*c) ha' hbc_self
      unfold D at step1 step2 step3
      apply le_antisymm _ dist_nonneg
      calc dist (fstar (g (g a * g (g b * g c)))) (fstar (a * (b * c)))
        ≤ dist (fstar (g (g a * g (g b * g c)))) (fstar (g a * g (g b * g c))) +
          dist (fstar (g a * g (g b * g c))) (fstar (a * (b * c))) := dist_triangle _ _ _
        _ ≤ dist (fstar (g (g a * g (g b * g c)))) (fstar (g a * g (g b * g c))) +
          (dist (fstar (g a * g (g b * g c))) (fstar (g a * (b * c))) +
           dist (fstar (g a * (b * c))) (fstar (a * (b * c)))) := by
             linarith [dist_triangle (fstar (g a * g (g b * g c)))
                                     (fstar (g a * (b * c)))
                                     (fstar (a * (b * c)))]
        _ = 0 + (0 + 0) := by rw [step1, step2, step3]
        _ = 0 := by ring
    -- Combine via string associativity
    apply le_antisymm _ dist_nonneg
    calc dist (M (M y₁ y₂) y₃) (M y₁ (M y₂ y₃))
      ≤ dist (M (M y₁ y₂) y₃) (fstar ((a * b) * c)) +
        dist (fstar ((a * b) * c)) (M y₁ (M y₂ y₃)) := dist_triangle _ _ _
      _ ≤ (dist (M (M y₁ y₂) y₃) (fstar (g (g (g a * g b) * g c))) +
           dist (fstar (g (g (g a * g b) * g c))) (fstar ((a * b) * c))) +
          dist (fstar ((a * b) * c)) (M y₁ (M y₂ y₃)) := by
            linarith [dist_triangle (M (M y₁ y₂) y₃)
                                    (fstar (g (g (g a * g b) * g c)))
                                    (fstar ((a * b) * c))]
      _ = (0 + 0) + dist (fstar ((a * b) * c)) (M y₁ (M y₂ y₃)) := by
            rw [hLHS_step1, hLHS_step2]
      _ = dist (fstar (a * (b * c))) (M y₁ (M y₂ y₃)) := by rw [h_string_assoc]; ring
      _ ≤ dist (fstar (a * (b * c))) (fstar (g (g a * g (g b * g c)))) +
          dist (fstar (g (g a * g (g b * g c)))) (M y₁ (M y₂ y₃)) := dist_triangle _ _ _
      _ = 0 + 0 := by rw [dist_comm, hRHS_step2, dist_comm, hRHS_step1]
      _ = 0 := by ring

/-- f*(g(1)) is a left and right identity for merge on the oracle range -/
theorem merge_identity (g : Strings → Strings) (fstar : Strings → Y)
  (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar)
  (y : Y) (hy : InOracleRange g fstar y) :
  dist (oracleMerge hA3 (fstar (g 1)) y) y = 0 ∧
  dist (oracleMerge hA3 y (fstar (g 1))) y = 0 := by
    obtain ⟨a, ha⟩ := hy
    have ⟨hM1, hM2⟩ := oracleMerge_spec g fstar hA3
    set M := oracleMerge hA3
    constructor
    -- Left identity
    · have h1 : dist (M (fstar (g 1)) y) (fstar (g (g 1 * g a))) = 0 := by
        rw [← ha]
        have := hM1 1 a
        rw [dist_comm]
        exact this
      have h2 : dist (fstar (g (g 1 * g a))) (fstar a) = 0 := by
        have hA2' : D fstar (1 * a) (g (g 1 * g a)) = 0 := hA2 1 a
        unfold D at hA2'
        rw [one_mul] at hA2'
        rw [dist_comm]
        exact hA2'
      have h3 : dist (fstar a) (fstar (g a)) = 0 := by
        have := hA1 a
        unfold D at this
        rw [dist_comm]
        exact this
      apply le_antisymm _ dist_nonneg
      calc dist (M (fstar (g 1)) y) y
        = dist (M (fstar (g 1)) y) (fstar (g a)) := by rw [← ha]
        _ ≤ dist (M (fstar (g 1)) y) (fstar (g (g 1 * g a))) +
            dist (fstar (g (g 1 * g a))) (fstar (g a)) := dist_triangle _ _ _
        _ ≤ dist (M (fstar (g 1)) y) (fstar (g (g 1 * g a))) +
            (dist (fstar (g (g 1 * g a))) (fstar a) + dist (fstar a) (fstar (g a))) := by
              linarith [dist_triangle (fstar (g (g 1 * g a))) (fstar a) (fstar (g a))]
        _ = 0 + (0 + 0) := by rw [h1, h2, h3]
        _ = 0 := by ring
    -- Right identity
    · have h1 : dist (M y (fstar (g 1))) (fstar (g (g a * g 1))) = 0 := by
        rw [← ha]
        have := hM1 a 1
        rw [dist_comm]
        exact this
      have h2 : dist (fstar (g (g a * g 1))) (fstar a) = 0 := by
        have hA2' : D fstar (a * 1) (g (g a * g 1)) = 0 := hA2 a 1
        unfold D at hA2'
        rw [mul_one] at hA2'
        rw [dist_comm]
        exact hA2'
      have h3 : dist (fstar a) (fstar (g a)) = 0 := by
        have := hA1 a
        unfold D at this
        rw [dist_comm]
        exact this
      apply le_antisymm _ dist_nonneg
      calc dist (M y (fstar (g 1))) y
        = dist (M y (fstar (g 1))) (fstar (g a)) := by rw [← ha]
        _ ≤ dist (M y (fstar (g 1))) (fstar (g (g a * g 1))) +
            dist (fstar (g (g a * g 1))) (fstar (g a)) := dist_triangle _ _ _
        _ ≤ dist (M y (fstar (g 1))) (fstar (g (g a * g 1))) +
            (dist (fstar (g (g a * g 1))) (fstar a) + dist (fstar a) (fstar (g a))) := by
              linarith [dist_triangle (fstar (g (g a * g 1))) (fstar a) (fstar (g a))]
        _ = 0 + (0 + 0) := by rw [h1, h2, h3]
        _ = 0 := by ring

/-!
## Global → Local Derivations

These theorems show that the global axioms (A1, A2, A3) imply the local laws (L1, L2, L3)
for any tree, when the summarizer is deterministic (wrapped in PMF.pure).
-/

/-- Helper: Eg for PMF.pure is just evaluation at the point -/
lemma Eg_pure {α : Type*} (f : α → ℝ) (x y : α) :
    Eg (fun _ : α => PMF.pure y) f x = f y := by
  unfold Eg
  simp only [PMF.pure_apply]
  rw [tsum_eq_single y]
  · simp
  · intro z hz
    simp [hz]

/-- Helper: Eg for a deterministic summarizer evaluates at g(x) -/
lemma Eg_deterministic (g_det : Strings → Strings) (f : Strings → ℝ) (x : Strings) :
    Eg (fun y => PMF.pure (g_det y)) f x = f (g_det x) := by
  unfold Eg
  simp only [PMF.pure_apply]
  have h : ∀ z, (if z = g_det x then (1 : ENNReal) else 0).toReal * f z =
           if z = g_det x then f z else 0 := by
    intro z; split_ifs with hz <;> simp
  simp_rw [h]
  rw [tsum_eq_single (g_det x)]
  · simp
  · intro z hz; simp [hz]

/-- A1 (global sufficiency) implies L1 (leaf idempotence) for the deterministic summarizer.

    If g preserves the oracle for all inputs (A1), then for any tree T, the summarizer
    preserves the oracle at each leaf (L1). -/
theorem A1_implies_L1 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) (T : BinTree Strings) :
    L1 (fun x => PMF.pure (g_det x)) T fstar := by
  intro b _hb
  rw [Eg_deterministic]
  exact hA1 b

/-- Deterministic reduction: apply g bottom-up on trees -/
def reduce_det (g : Strings → Strings) : BinTree Strings → Strings
  | BinTree.leaf b => g b
  | BinTree.node T_L T_R => g (reduce_det g T_L * reduce_det g T_R)

/-- Helper: reduce for PMF.pure ∘ g_det is PMF.pure (reduce_det g_det T) -/
lemma reduce_deterministic (g_det : Strings → Strings) (T : BinTree Strings) :
    reduce (fun x => PMF.pure (g_det x)) T = PMF.pure (reduce_det g_det T) := by
  induction T with
  | leaf b => simp [reduce, reduce_det]
  | node T_L T_R ih_L ih_R =>
    simp only [reduce, reduce_det]
    rw [ih_L, ih_R]
    simp only [PMF.pure_bind]

/-- Helper: Egu for PMF.pure ∘ g_det is evaluation at reduce_det g_det T -/
lemma Egu_deterministic (g_det : Strings → Strings) (T : BinTree Strings) (f : Strings → ℝ) :
    Egu (fun x => PMF.pure (g_det x)) T f = f (reduce_det g_det T) := by
  unfold Egu
  rw [reduce_deterministic]
  simp only [PMF.pure_apply]
  have h : ∀ z, (if z = reduce_det g_det T then (1 : ENNReal) else 0).toReal * f z =
           if z = reduce_det g_det T then f z else 0 := by
    intro z; split_ifs with hz <;> simp
  simp_rw [h]
  rw [tsum_eq_single (reduce_det g_det T)]
  · simp
  · intro z hz; simp [hz]

/-- Helper: reduce_det preserves oracle under A1 + A2 + A3 (uses concat_congruence) -/
theorem reduce_det_oracle_eq (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    (T : BinTree Strings) → D fstar (reduce_det g T) (S T) = 0
  | BinTree.leaf b => by
      simp only [reduce_det, S]
      exact hA1 b
  | BinTree.node T_L T_R => by
      simp only [reduce_det, S]
      -- Goal: D fstar (g (reduce T_L * reduce T_R)) (S T_L * S T_R) = 0
      -- By IH: D(reduce T_L, S T_L) = 0 and D(reduce T_R, S T_R) = 0
      have ih_L := reduce_det_oracle_eq g fstar hA1 hA2 hA3 T_L
      have ih_R := reduce_det_oracle_eq g fstar hA1 hA2 hA3 T_R
      -- By concat_congruence: D(reduce_L * reduce_R, S_L * S_R) = 0
      have h_concat := concat_congruence g fstar hA1 hA2 hA3
        (reduce_det g T_L) (S T_L)
        (reduce_det g T_R) (S T_R)
        ih_L ih_R
      -- By A1: D(g(reduce_L * reduce_R), reduce_L * reduce_R) = 0
      have h_A1 := hA1 (reduce_det g T_L * reduce_det g T_R)
      -- Combine via triangle inequality
      unfold D at h_A1 h_concat ⊢
      apply le_antisymm _ dist_nonneg
      calc dist (fstar (g (reduce_det g T_L * reduce_det g T_R))) (fstar (S T_L * S T_R))
          ≤ dist (fstar (g (reduce_det g T_L * reduce_det g T_R)))
                 (fstar (reduce_det g T_L * reduce_det g T_R)) +
            dist (fstar (reduce_det g T_L * reduce_det g T_R)) (fstar (S T_L * S T_R)) :=
              dist_triangle _ _ _
          _ = 0 + 0 := by rw [h_A1, h_concat]
          _ = 0 := by ring

/-- A1 + A2 + A3 implies L2 (parentwise compatibility) for the deterministic summarizer.

    Note: This requires all three global axioms:
    - A1: to ensure g preserves oracle
    - A2: for the two-route identity
    - A3: for concat_congruence (needed to show reduce preserves oracle)

    The key insight is that for a deterministic summarizer, reduce_det preserves
    the oracle value (via reduce_det_oracle_eq), so the L2 condition holds. -/
theorem A1_A2_A3_implies_L2 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) (hA2 : A2_global g_det fstar) (hA3 : A3_global g_det fstar)
    (T : BinTree Strings) :
    L2 (fun x => PMF.pure (g_det x)) T fstar := by
  intro pair hpair
  obtain ⟨T_L, T_R⟩ := pair
  -- Unfold L2 goal
  simp only
  rw [Egu_deterministic]
  -- The goal is now: D fstar (reduce_det g_det (node T_L T_R)) (S (node T_L T_R)) = 0
  exact reduce_det_oracle_eq g_det fstar hA1 hA2 hA3 (BinTree.node T_L T_R)

/-!
## Proposition 3: Classical Mergeable Summary

A deterministic summarizer with global axioms A1, A2, A3 is a "mergeable summary"
in the classical streaming/sketching sense. When the oracle is commutative,
the merge operation is also commutative.
-/

/-- Oracle Commutativity: f*(u·v) = f*(v·u) for all u, v -/
def OracleCommutative (fstar : Strings → Y) : Prop :=
  ∀ u v : Strings, dist (fstar (u * v)) (fstar (v * u)) = 0

/-- Summary Merge Operation: ⊕(s, t) = g(s * t)
    This is the natural merge on summary strings: concatenate then re-summarize. -/
def summaryMerge (g : Strings → Strings) (s t : Strings) : Strings := g (s * t)

/-- Strict oracle-level mergeable summary property.

    A summarizer g is mergeable if:
    1. Merge-oracle equivalence: f*(g(u·v)) = f*(⊕(g(u), g(v)))
    2. Associativity: ⊕(⊕(a,b), c) ≈ ⊕(a, ⊕(b,c)) at the oracle level

This captures the strict homomorphism case where the task value is itself a
valid readout state.  Classical mergeable sketches are more general: their
bounded state can compose even when final query answers cannot. -/
def IsMergeableSummary (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  -- Merge-oracle equivalence: summarizing u·v equals merging summaries
  (∀ u v, D fstar (g (u * v)) (summaryMerge g (g u) (g v)) = 0) ∧
  -- Associativity of merge at oracle level
  (∀ a b c, D fstar (summaryMerge g (summaryMerge g a b) c)
                    (summaryMerge g a (summaryMerge g b c)) = 0)

/-- Strict deterministic summarizer with A1+A2+A3 is oracle-level mergeable.

    This is the strict oracle-homomorphism component of the paper's
    mergeable-reduction proposition.  It should not be read as the full
    classical state-level sketch condition.

    We prove this from the global axioms A1, A2, A3. -/
theorem prop3_mergeable_classical (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    IsMergeableSummary g fstar := by
  constructor
  -- Part 1: Merge-oracle equivalence
  -- Goal: D fstar (g (u * v)) (summaryMerge g (g u) (g v)) = 0
  -- i.e., D fstar (g (u * v)) (g (g u * g v)) = 0
  · intro u v
    unfold summaryMerge
    -- By A1 at (u * v): D fstar (g (u * v)) (u * v) = 0
    have h1 : D fstar (g (u * v)) (u * v) = 0 := hA1 (u * v)
    -- By A2: D fstar (u * v) (g (g u * g v)) = 0
    have h2 : D fstar (u * v) (g (g u * g v)) = 0 := hA2 u v
    -- Triangle inequality gives the result
    apply le_antisymm _ dist_nonneg
    calc D fstar (g (u * v)) (g (g u * g v))
        ≤ D fstar (g (u * v)) (u * v) + D fstar (u * v) (g (g u * g v)) :=
          D_triangle fstar (g (u * v)) (u * v) (g (g u * g v))
      _ = 0 + 0 := by rw [h1, h2]
      _ = 0 := by ring
  -- Part 2: Associativity of merge at oracle level
  -- Goal: D fstar (g (g (a * b) * c)) (g (a * g (b * c))) = 0
  · intro a b c
    unfold summaryMerge
    -- LHS = g(g(a*b) * c), RHS = g(a * g(b*c))
    -- Both should equal f*(a * b * c) at the oracle level
    -- By A1: D fstar (g(g(a*b) * c)) (g(a*b) * c) = 0
    have hL1 : D fstar (g (g (a * b) * c)) (g (a * b) * c) = 0 := hA1 (g (a * b) * c)
    -- By A1: D fstar (g(a*b)) (a*b) = 0
    have hab : D fstar (g (a * b)) (a * b) = 0 := hA1 (a * b)
    -- By A1: D fstar c c = 0 (trivially)
    have hc : D fstar c c = 0 := dist_self _
    -- By concat_congruence: D fstar (g(a*b) * c) ((a*b) * c) = 0
    have hL2 : D fstar (g (a * b) * c) ((a * b) * c) = 0 :=
      concat_congruence g fstar hA1 hA2 hA3 (g (a * b)) (a * b) c c hab hc
    -- Similarly for RHS
    have hR1 : D fstar (g (a * g (b * c))) (a * g (b * c)) = 0 := hA1 (a * g (b * c))
    have hbc : D fstar (g (b * c)) (b * c) = 0 := hA1 (b * c)
    have ha : D fstar a a = 0 := dist_self _
    have hR2 : D fstar (a * g (b * c)) (a * (b * c)) = 0 :=
      concat_congruence g fstar hA1 hA2 hA3 a a (g (b * c)) (b * c) ha hbc
    -- By associativity of monoid: (a * b) * c = a * (b * c)
    have h_assoc : (a * b) * c = a * (b * c) := mul_assoc a b c
    -- Combine all steps
    apply le_antisymm _ dist_nonneg
    calc D fstar (g (g (a * b) * c)) (g (a * g (b * c)))
        ≤ D fstar (g (g (a * b) * c)) ((a * b) * c) +
          D fstar ((a * b) * c) (g (a * g (b * c))) := D_triangle fstar _ _ _
      _ ≤ (D fstar (g (g (a * b) * c)) (g (a * b) * c) +
           D fstar (g (a * b) * c) ((a * b) * c)) +
          D fstar ((a * b) * c) (g (a * g (b * c))) := by
            linarith [D_triangle fstar (g (g (a * b) * c)) (g (a * b) * c) ((a * b) * c)]
      _ = (0 + 0) + D fstar ((a * b) * c) (g (a * g (b * c))) := by rw [hL1, hL2]
      _ = D fstar (a * (b * c)) (g (a * g (b * c))) := by rw [h_assoc]; ring
      _ ≤ D fstar (a * (b * c)) (a * g (b * c)) +
          D fstar (a * g (b * c)) (g (a * g (b * c))) := D_triangle fstar _ _ _
      _ = 0 + 0 := by rw [D_symm, hR2, D_symm, hR1]
      _ = 0 := by ring

/-- Commutativity of merge when oracle is commutative.

    When f*(u·v) = f*(v·u) for all u, v, the summary merge operation
    is also commutative at the oracle level. -/
theorem mergeable_commutative (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (h_comm : OracleCommutative fstar) :
    ∀ a b, D fstar (summaryMerge g a b) (summaryMerge g b a) = 0 := by
  intro a b
  unfold summaryMerge
  -- Goal: D fstar (g (a * b)) (g (b * a)) = 0
  -- By A1: D fstar (g (a * b)) (a * b) = 0
  have h1 : D fstar (g (a * b)) (a * b) = 0 := hA1 (a * b)
  -- By h_comm: dist (fstar (a * b)) (fstar (b * a)) = 0
  have h2 : D fstar (a * b) (b * a) = 0 := h_comm a b
  -- By A1: D fstar (g (b * a)) (b * a) = 0
  have h3 : D fstar (g (b * a)) (b * a) = 0 := hA1 (b * a)
  -- Triangle inequality gives the result
  apply le_antisymm _ dist_nonneg
  calc D fstar (g (a * b)) (g (b * a))
      ≤ D fstar (g (a * b)) (a * b) + D fstar (a * b) (g (b * a)) := D_triangle fstar _ _ _
    _ ≤ D fstar (g (a * b)) (a * b) +
        (D fstar (a * b) (b * a) + D fstar (b * a) (g (b * a))) := by
          linarith [D_triangle fstar (a * b) (b * a) (g (b * a))]
    _ = 0 + (0 + 0) := by rw [h1, h2, D_symm, h3]
    _ = 0 := by ring

/-!
## Global → Local Derivations

These theorems show that the global axioms (A1, A2, A3) imply the local laws (L1, L2, L3)
for any tree, when the summarizer is deterministic (wrapped in PMF.pure).
-/

/-- A1 (global sufficiency) implies L3 (on-range idempotence) for the deterministic summarizer.

    If g preserves the oracle for all inputs (A1), and Z is in the range of g
    (meaning Z = g(x) for some x), then re-summarizing Z still preserves the oracle.

    Key insight: For Z in range, Z = g(x) for some x. Then:
    - By A1 at x: f*(g(x)) = f*(x)
    - By A1 at Z = g(x): f*(g(Z)) = f*(g(g(x))) = f*(g(x)) = f*(Z) -/
theorem A1_implies_L3 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) :
    L3 (fun x => PMF.pure (g_det x)) fstar := by
  intro Z hZ
  rw [Eg_deterministic]
  -- Need: D fstar (g Z) Z = 0
  -- This is exactly A1 applied at Z
  exact hA1 Z

/-- For deterministic summaries, L1 is exactly leafwise global sufficiency on that tree. -/
theorem L1_deterministic_iff_leafwise (g_det : Strings → Strings) (fstar : Strings → Y)
    (T : BinTree Strings) :
    L1 (fun x => PMF.pure (g_det x)) T fstar ↔
      ∀ b, b ∈ leaves T → D fstar (g_det b) b = 0 := by
  constructor
  · intro hL1 b hb
    have h_leaf : Eg (fun x => PMF.pure (g_det x)) (fun z => D fstar z b) b = 0 := hL1 b hb
    simpa [Eg_deterministic] using h_leaf
  · intro h_leaf b hb
    have h_zero : D fstar (g_det b) b = 0 := h_leaf b hb
    simpa [Eg_deterministic] using h_zero

/-- For deterministic summaries, L3 is exactly in-range global sufficiency. -/
theorem L3_deterministic_iff_inRange (g_det : Strings → Strings) (fstar : Strings → Y) :
    L3 (fun x => PMF.pure (g_det x)) fstar ↔
      ∀ Z, InRange (fun x => PMF.pure (g_det x)) Z → D fstar (g_det Z) Z = 0 := by
  constructor
  · intro hL3 Z hZ
    have h_step : Eg (fun x => PMF.pure (g_det x)) (fun z => D fstar z Z) Z = 0 := hL3 Z hZ
    simpa [Eg_deterministic] using h_step
  · intro h_inrange Z hZ
    have h_zero : D fstar (g_det Z) Z = 0 := h_inrange Z hZ
    simpa [Eg_deterministic] using h_zero

/-- Global sufficiency restricted to the summary range of a deterministic summarizer. -/
def A1_on_summary_range (g_det : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ Z, InRange (fun x => PMF.pure (g_det x)) Z → D fstar (g_det Z) Z = 0

/-- Non-surjective converse variant: deterministic L3 is equivalent to A1 on summary range. -/
theorem L3_iff_A1_on_summary_range (g_det : Strings → Strings) (fstar : Strings → Y) :
    L3 (fun x => PMF.pure (g_det x)) fstar ↔ A1_on_summary_range g_det fstar := by
  simpa [A1_on_summary_range] using L3_deterministic_iff_inRange g_det fstar

/-- Global A1 always implies the weaker in-range variant. -/
theorem A1_implies_A1_on_summary_range (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) :
    A1_on_summary_range g_det fstar := by
  intro Z _hZ
  exact hA1 Z

/-- If deterministic L1 holds for every tree, then A1 holds globally. -/
theorem L1_for_all_trees_implies_A1 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hL1 : ∀ T : BinTree Strings, L1 (fun x => PMF.pure (g_det x)) T fstar) :
    A1_global g_det fstar := by
  intro z
  have h_leaf_tree : L1 (fun x => PMF.pure (g_det x)) (BinTree.leaf z) fstar := hL1 (BinTree.leaf z)
  have hz_mem : z ∈ leaves (BinTree.leaf z) := by simp [leaves]
  have h_zero : Eg (fun x => PMF.pure (g_det x)) (fun w => D fstar w z) z = 0 := h_leaf_tree z hz_mem
  simpa [Eg_deterministic] using h_zero

/-- Global A1 is equivalent to deterministic L1 holding on all trees. -/
theorem A1_iff_L1_for_all_trees (g_det : Strings → Strings) (fstar : Strings → Y) :
    A1_global g_det fstar ↔ ∀ T : BinTree Strings, L1 (fun x => PMF.pure (g_det x)) T fstar := by
  constructor
  · intro hA1 T
    exact A1_implies_L1 g_det fstar hA1 T
  · intro h_all
    exact L1_for_all_trees_implies_A1 g_det fstar h_all

/-- Deterministic L3 implies A1 when every string is in the summary range. -/
theorem L3_implies_A1_of_surjective (g_det : Strings → Strings) (fstar : Strings → Y)
    (h_surj : Function.Surjective g_det)
    (hL3 : L3 (fun x => PMF.pure (g_det x)) fstar) :
    A1_global g_det fstar := by
  intro Z
  have h_in_range : InRange (fun x => PMF.pure (g_det x)) Z := by
    rcases h_surj Z with ⟨x, hx⟩
    refine ⟨x, ?_⟩
    rw [PMF.support_pure, Set.mem_singleton_iff]
    exact hx.symm
  have h_step : Eg (fun x => PMF.pure (g_det x)) (fun z => D fstar z Z) Z = 0 := hL3 Z h_in_range
  simpa [Eg_deterministic] using h_step

/-- Under surjectivity, deterministic L3 is equivalent to A1 global sufficiency. -/
theorem A1_iff_L3_of_surjective (g_det : Strings → Strings) (fstar : Strings → Y)
    (h_surj : Function.Surjective g_det) :
    A1_global g_det fstar ↔ L3 (fun x => PMF.pure (g_det x)) fstar := by
  constructor
  · exact A1_implies_L3 g_det fstar
  · intro hL3
    exact L3_implies_A1_of_surjective g_det fstar h_surj hL3

/-- Deterministic L2 on a two-leaf tree is exactly the A2 two-route identity at `(u,v)`. -/
theorem L2_deterministic_two_leaf_iff_A2_pointwise
    (g_det : Strings → Strings) (fstar : Strings → Y) (u v : Strings) :
    L2 (fun x => PMF.pure (g_det x)) (BinTree.node (BinTree.leaf u) (BinTree.leaf v)) fstar ↔
      D fstar (u * v) (g_det (g_det u * g_det v)) = 0 := by
  constructor
  · intro hL2
    have h_local :
        Egu (fun x => PMF.pure (g_det x))
          (BinTree.node (BinTree.leaf u) (BinTree.leaf v))
          (fun z => D fstar z (S (BinTree.node (BinTree.leaf u) (BinTree.leaf v)))) = 0 := by
      exact hL2 (BinTree.leaf u, BinTree.leaf v) (by simp [internal_nodes])
    have h_det : D fstar (g_det (g_det u * g_det v)) (u * v) = 0 := by
      simpa [Egu_deterministic, reduce_det, S] using h_local
    simpa [D_symm] using h_det
  · intro hA2 p hp
    have hp_eq : p = (BinTree.leaf u, BinTree.leaf v) := by
      simpa [internal_nodes] using hp
    subst p
    have h_det : D fstar (g_det (g_det u * g_det v)) (u * v) = 0 := by
      simpa [D_symm] using hA2
    simpa [Egu_deterministic, reduce_det, S] using h_det

/-- A2 implies deterministic L2 on every two-leaf tree. -/
theorem A2_implies_L2_on_two_leaf_trees (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA2 : A2_global g_det fstar) :
    ∀ u v, L2 (fun x => PMF.pure (g_det x)) (BinTree.node (BinTree.leaf u) (BinTree.leaf v)) fstar := by
  intro u v
  exact (L2_deterministic_two_leaf_iff_A2_pointwise g_det fstar u v).2 (hA2 u v)

/-- Deterministic L2 on all two-leaf trees implies A2 global two-route identity. -/
theorem L2_on_two_leaf_trees_implies_A2 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hL2 : ∀ u v, L2 (fun x => PMF.pure (g_det x)) (BinTree.node (BinTree.leaf u) (BinTree.leaf v)) fstar) :
    A2_global g_det fstar := by
  intro u v
  exact (L2_deterministic_two_leaf_iff_A2_pointwise g_det fstar u v).1 (hL2 u v)

/-- Global A2 is equivalent to deterministic L2 holding on all two-leaf trees. -/
theorem A2_iff_L2_on_two_leaf_trees (g_det : Strings → Strings) (fstar : Strings → Y) :
    A2_global g_det fstar ↔
      ∀ u v, L2 (fun x => PMF.pure (g_det x)) (BinTree.node (BinTree.leaf u) (BinTree.leaf v)) fstar := by
  constructor
  · exact A2_implies_L2_on_two_leaf_trees g_det fstar
  · exact L2_on_two_leaf_trees_implies_A2 g_det fstar

/-- Deterministic L2 on all trees implies A2 global two-route identity. -/
theorem L2_on_all_trees_implies_A2 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hL2 : ∀ T : BinTree Strings, L2 (fun x => PMF.pure (g_det x)) T fstar) :
    A2_global g_det fstar := by
  intro u v
  exact (L2_deterministic_two_leaf_iff_A2_pointwise g_det fstar u v).1 (hL2 _)

/-- Under A1 and A3, global A2 is equivalent to deterministic L2 on all trees. -/
theorem A2_iff_L2_on_all_trees_of_A1_A3 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) (hA3 : A3_global g_det fstar) :
    A2_global g_det fstar ↔ ∀ T : BinTree Strings, L2 (fun x => PMF.pure (g_det x)) T fstar := by
  constructor
  · intro hA2 T
    exact A1_A2_A3_implies_L2 g_det fstar hA1 hA2 hA3 T
  · intro hL2
    exact L2_on_all_trees_implies_A2 g_det fstar hL2

/-- Under A1 and A3, checking deterministic L2 on two-leaf trees is equivalent
to checking deterministic L2 on all trees. -/
theorem L2_on_all_trees_iff_two_leaf_trees_of_A1_A3 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) (hA3 : A3_global g_det fstar) :
    (∀ T : BinTree Strings, L2 (fun x => PMF.pure (g_det x)) T fstar) ↔
      (∀ u v, L2 (fun x => PMF.pure (g_det x)) (BinTree.node (BinTree.leaf u) (BinTree.leaf v)) fstar) := by
  constructor
  · intro h_all u v
    exact h_all _
  · intro h_two
    have hA2 : A2_global g_det fstar := L2_on_two_leaf_trees_implies_A2 g_det fstar h_two
    intro T
    exact A1_A2_A3_implies_L2 g_det fstar hA1 hA2 hA3 T

/-- Under surjectivity, deterministic L1-on-all-trees and deterministic L3 are equivalent. -/
theorem L1_on_all_trees_iff_L3_of_surjective
    (g_det : Strings → Strings) (fstar : Strings → Y)
    (h_surj : Function.Surjective g_det) :
    (∀ T : BinTree Strings, L1 (fun x => PMF.pure (g_det x)) T fstar) ↔
      L3 (fun x => PMF.pure (g_det x)) fstar := by
  constructor
  · intro hL1
    have hA1 : A1_global g_det fstar := L1_for_all_trees_implies_A1 g_det fstar hL1
    exact (A1_iff_L3_of_surjective g_det fstar h_surj).1 hA1
  · intro hL3
    have hA1 : A1_global g_det fstar := (A1_iff_L3_of_surjective g_det fstar h_surj).2 hL3
    intro T
    exact A1_implies_L1 g_det fstar hA1 T

/-- Under A3 and surjectivity, `(A1 ∧ A2)` is equivalent to `(L3 ∧ L2-on-all-trees)`. -/
theorem A1_A2_iff_L3_and_L2_on_all_trees_of_A3_surjective
    (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA3 : A3_global g_det fstar) (h_surj : Function.Surjective g_det) :
    (A1_global g_det fstar ∧ A2_global g_det fstar) ↔
      (L3 (fun x => PMF.pure (g_det x)) fstar ∧
        ∀ T : BinTree Strings, L2 (fun x => PMF.pure (g_det x)) T fstar) := by
  constructor
  · intro h
    rcases h with ⟨hA1, hA2⟩
    refine ⟨A1_implies_L3 g_det fstar hA1, ?_⟩
    intro T
    exact A1_A2_A3_implies_L2 g_det fstar hA1 hA2 hA3 T
  · intro h
    rcases h with ⟨hL3, hL2⟩
    have hA1 : A1_global g_det fstar := (A1_iff_L3_of_surjective g_det fstar h_surj).2 hL3
    have hA2 : A2_global g_det fstar := L2_on_all_trees_implies_A2 g_det fstar hL2
    exact ⟨hA1, hA2⟩

/-- Under A3 and surjectivity, `(A1 ∧ A2)` is equivalent to
`(L3 ∧ L2-on-two-leaf-trees)`. -/
theorem A1_A2_iff_L3_and_L2_on_two_leaf_trees_of_A3_surjective
    (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA3 : A3_global g_det fstar) (h_surj : Function.Surjective g_det) :
    (A1_global g_det fstar ∧ A2_global g_det fstar) ↔
      (L3 (fun x => PMF.pure (g_det x)) fstar ∧
        ∀ u v, L2 (fun x => PMF.pure (g_det x)) (BinTree.node (BinTree.leaf u) (BinTree.leaf v)) fstar) := by
  constructor
  · intro h
    rcases h with ⟨hA1, hA2⟩
    refine ⟨A1_implies_L3 g_det fstar hA1, ?_⟩
    intro u v
    exact A1_A2_A3_implies_L2 g_det fstar hA1 hA2 hA3 _
  · intro h
    rcases h with ⟨hL3, hL2⟩
    have hA1 : A1_global g_det fstar := (A1_iff_L3_of_surjective g_det fstar h_surj).2 hL3
    have hA2 : A2_global g_det fstar := L2_on_two_leaf_trees_implies_A2 g_det fstar hL2
    exact ⟨hA1, hA2⟩

/-!
## GlobalPreservation → LocalLawsBundle Derivation

When a summarizer has GlobalPreservation, we can automatically derive
LocalLawsBundle for any tree structure.
-/

namespace GlobalPreservation

/-- Derive LocalLawsBundle for any tree from GlobalPreservation.

This is the key connection: global axioms automatically give local laws for
any tree structure, wrapped in PMF.pure for the deterministic summarizer. -/
def toLocalLawsBundle {Strings : Type*} [Monoid Strings] {Y : Type*} [PseudoMetricSpace Y]
    {g_det : Strings → Strings} {fstar : Strings → Y}
    [inst : GlobalPreservation g_det fstar] (T : BinTree Strings) :
    LocalLawsBundle (fun x => PMF.pure (g_det x)) T fstar where
  law1 := A1_implies_L1 g_det fstar inst.a1 T
  law2 := A1_A2_A3_implies_L2 g_det fstar inst.a1 inst.a2 inst.a3 T
  law3 := A1_implies_L3 g_det fstar inst.a1

end GlobalPreservation

/-!
## Bundle Theorem Variants

Theorems using GlobalPreservation for cleaner signatures.
-/

/-- Concatenation congruence using GlobalPreservation typeclass. -/
theorem concat_congruence_global {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Strings → Strings} {fstar : Strings → Y}
    [inst : GlobalPreservation g fstar]
    (a a' b b' : Strings) (ha : D fstar a a' = 0) (hb : D fstar b b' = 0) :
    D fstar (a * b) (a' * b') = 0 :=
  concat_congruence g fstar inst.a1 inst.a2 inst.a3 a a' b b' ha hb

/-- Merge associativity using GlobalPreservation typeclass. -/
theorem merge_assoc_global {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Strings → Strings} {fstar : Strings → Y}
    [inst : GlobalPreservation g fstar]
    (y₁ y₂ y₃ : Y)
    (hy₁ : InOracleRange g fstar y₁)
    (hy₂ : InOracleRange g fstar y₂)
    (hy₃ : InOracleRange g fstar y₃) :
    dist (oracleMerge inst.a3 (oracleMerge inst.a3 y₁ y₂) y₃)
         (oracleMerge inst.a3 y₁ (oracleMerge inst.a3 y₂ y₃)) = 0 :=
  merge_assoc g fstar inst.a1 inst.a2 inst.a3 y₁ y₂ y₃ hy₁ hy₂ hy₃

/-- Strict oracle-level mergeability using the `GlobalPreservation` typeclass. -/
theorem prop3_mergeable_global {Strings : Type*} [Monoid Strings]
    {Y : Type*} [PseudoMetricSpace Y]
    {g : Strings → Strings} {fstar : Strings → Y}
    [inst : GlobalPreservation g fstar] :
    IsMergeableSummary g fstar :=
  prop3_mergeable_classical g fstar inst.a1 inst.a2 inst.a3

/-!
## Paper-Numbered Theorem Aliases

These provide explicit names matching the paper's proposition numbering.
-/

/-- **Proposition 1: Global Sufficiency Implies Leaf Idempotence**

**Paper Reference:** Section 4, Proposition 1

A1 (global sufficiency: D(g(z), z) = 0 for all z) implies L1 (leaf idempotence)
for any tree T. This shows that the global axiom is strictly stronger than
what's needed for leaf preservation. -/
theorem prop1_A1_implies_L1 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) (T : BinTree Strings) :
    L1 (fun x => PMF.pure (g_det x)) T fstar :=
  A1_implies_L1 g_det fstar hA1 T

/-- **Proposition 2: Global Axioms Imply Internal Node Compatibility**

**Paper Reference:** Section 4, Proposition 2

A1 + A2 + A3 (global sufficiency + compatibility + merge existence) together
imply L2 (internal node idempotence) for any tree T. All three axioms are
necessary for the two-route identity to hold at internal nodes. -/
theorem prop2_A1_A2_A3_implies_L2 (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g_det fstar) (hA2 : A2_global g_det fstar) (hA3 : A3_global g_det fstar)
    (T : BinTree Strings) :
    L2 (fun x => PMF.pure (g_det x)) T fstar :=
  A1_A2_A3_implies_L2 g_det fstar hA1 hA2 hA3 T

end
