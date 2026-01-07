/-
FormalProofs/GlobalAssumptions.lean

Global oracle preservation assumptions:
- A1_global: Oracle sufficiency (distortion 0 for all strings)
- A2_global: Global compatibility (two-route identity)
- A3_global: Merge depends on oracle
- concat_congruence: Concatenation respects equivalence
- oracleMerge: The extracted merge function
- merge_assoc, merge_identity: Algebraic properties
-/

import FormalProofs.ExpectationTheory

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

/-- A3: Merge Depends on Oracle - exists merge function with properties -/
def A3_global (g : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∃ M : Y → Y → Y,
    (∀ u v : Strings, dist (fstar (g (g u * g v))) (M (fstar (g u)) (fstar (g v))) = 0) ∧
    (∀ y₁ y₁' y₂ y₂' : Y, dist y₁ y₁' = 0 → dist y₂ y₂' = 0 →
      dist (M y₁ y₂) (M y₁' y₂') = 0)

/-- InOracleRange: y is in the range of f* ∘ g -/
def InOracleRange (g : Strings → Strings) (fstar : Strings → Y) (y : Y) : Prop :=
  ∃ z : Strings, fstar (g z) = y

/-!
## Helper Lemmas
-/

/-- D is symmetric -/
lemma D_symm (fstar : Strings → Y) (x y : Strings) : D fstar x y = D fstar y x := by
  unfold D
  exact dist_comm (fstar x) (fstar y)

/-- D satisfies triangle inequality -/
lemma D_triangle' (fstar : Strings → Y) (x y z : Strings) :
  D fstar x z ≤ D fstar x y + D fstar y z := by
  unfold D
  exact dist_triangle (fstar x) (fstar y) (fstar z)

/-!
## Concatenation Congruence
-/

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
        ≤ D fstar (a * b) U + D fstar U (a' * b') := D_triangle' fstar (a * b) U (a' * b')
        _ ≤ D fstar (a * b) U + (D fstar U U' + D fstar U' (a' * b')) := by
            linarith [D_triangle' fstar U U' (a' * b')]
        _ = D fstar (a * b) U + D fstar U U' + D fstar U' (a' * b') := by ring
    have h1 : D fstar (a * b) U = 0 := hA2 a b
    have h3 : D fstar U' (a' * b') = 0 := by rw [D_symm]; exact hA2 a' b'
    have hga : dist (fstar (g a)) (fstar (g a')) = 0 := by
      have hga1 : D fstar (g a) a = 0 := hA1 a
      have hga'1 : D fstar (g a') a' = 0 := hA1 a'
      unfold D at hga1 hga'1 ha
      apply le_antisymm _ dist_nonneg
      calc dist (fstar (g a)) (fstar (g a'))
        ≤ dist (fstar (g a)) (fstar a) + dist (fstar a) (fstar (g a')) := dist_triangle _ _ _
        _ ≤ dist (fstar (g a)) (fstar a) +
            (dist (fstar a) (fstar a') + dist (fstar a') (fstar (g a'))) := by
            linarith [dist_triangle (fstar a) (fstar a') (fstar (g a'))]
        _ = 0 + (0 + 0) := by rw [hga1, ha, dist_comm (fstar a') (fstar (g a')), hga'1]
        _ = 0 := by ring
    have hgb : dist (fstar (g b)) (fstar (g b')) = 0 := by
      have hgb1 : D fstar (g b) b = 0 := hA1 b
      have hgb'1 : D fstar (g b') b' = 0 := hA1 b'
      unfold D at hgb1 hgb'1 hb
      apply le_antisymm _ dist_nonneg
      calc dist (fstar (g b)) (fstar (g b'))
        ≤ dist (fstar (g b)) (fstar b) + dist (fstar b) (fstar (g b')) := dist_triangle _ _ _
        _ ≤ dist (fstar (g b)) (fstar b) +
            (dist (fstar b) (fstar b') + dist (fstar b') (fstar (g b'))) := by
            linarith [dist_triangle (fstar b) (fstar b') (fstar (g b'))]
        _ = 0 + (0 + 0) := by rw [hgb1, hb, dist_comm (fstar b') (fstar (g b')), hgb'1]
        _ = 0 := by ring
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

end
