import FormalProofs.OPT.ApproxFiberTransport
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.OPT.TwoStageDecomposition

/-!
# FormalProofs/OPT/UnifiedOracleRoute.lean

Unified oracle approximation framework: every pipeline in the codebase
produces a bound on expected distortion, and the utility transport theorem
consumes it uniformly.

## The Three-Step Pipeline

Every oracle approximation approach follows the same pattern:

1. **Bound oracle distortion** `E_Z[D(oracle, Z, x)]` via one of:
   - Exact local laws → 0
   - Approximate local laws → budget
   - Uniform surrogate lift → surrogate_budget + 2ε

2. **Transfer to evaluation space** `E_Z[D(eval, Z, x)]` via:
   - Identity (when eval = oracle)
   - Lipschitz: ≤ K · oracle_bound
   - Lipschitz + fiber: ≤ K · oracle_bound + ε_fiber

3. **Utility transport** converts distortion to utility gap:
   - `|utility gap| ≤ L₁ · eval_bound + L₂ · measurement`

The unifying abstraction is `ExpectedDistortionBound` — the single interface
between steps 1-2 (the "route") and step 3 (the "consumer").

## Existing Theorems as Compositions

| Theorem | = Constructor | ∘ Transfer | ∘ Transport |
|---------|--------------|-----------|------------|
| `expected_utility_bound_approx_fiber` | `ofApproxLaws` | `lipschitzFiber` | `universalTransport` |
| `TwoStageOracleSurrogate` theorems | `ofApproxLaws` | `surrogateLift` | `universalTransport` |
| `TwoStageDecomposition` | `ofApproxLaws` | `lipschitzFiber` | `universalTransport` + `substitution` |
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Strings : Type*} [Monoid Strings]

-- ============================================================================
-- Part 1: The Universal Interface
-- ============================================================================

/-- Expected distortion bound: packages a proof that the expected distortion
of `f`-images under distribution `p` is bounded by `bound`.

This is the **single universal interface** between oracle approximation routes
(which produce bounds) and utility transport theorems (which consume them).

Every oracle approximation approach — exact laws, approximate laws, surrogate
oracles, fiber-preserving features — ultimately produces an
`ExpectedDistortionBound`. Every utility gap theorem consumes one. -/
structure ExpectedDistortionBound
    {S : Type*} [PseudoMetricSpace S]
    (p : PMF Strings) (f : Strings → S) (x : Strings) where
  /-- The bound value. -/
  bound : ℝ
  /-- Bounds are nonneg (distortion is nonneg). -/
  bound_nonneg : 0 ≤ bound
  /-- The actual distortion bound. -/
  distortion_le : Exp p (fun z => D f z x) ≤ bound

namespace ExpectedDistortionBound

-- ============================================================================
-- Part 2: Universal Transport (The Single Consumer)
-- ============================================================================

section UniversalTransport

variable {S : Type*} [BoundedPseudoMetricSpace S]

/-- **Universal utility transport theorem.**

Given ANY `ExpectedDistortionBound`, the utility gap is bounded by
`L₁ · bound + L₂ · measurement`. This is the single consumer of all
oracle approximation routes.

Every existing utility gap theorem in the codebase is an instance of this
theorem composed with a specific constructor for `ExpectedDistortionBound`. -/
theorem universalTransport
    (p : PMF Strings) (f fhat : Strings → S) (x : Strings)
    (u : OracleUtility2 S) (L1 L2 : ℝ≥0)
    (edb : ExpectedDistortionBound p f x)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (f x) (BoundedPseudoMetricSpace.diameterBound (α := S)))
    (hD : Summable (fun z => (p z).toReal * D f z x)) :
    |Exp p (fun z => u (f z) (fhat x)) - u (f x) (f x)| ≤
      (L1 : ℝ) * edb.bound + (L2 : ℝ) * dist (fhat x) (f x) := by
  have h_transport :=
    expected_utility_bound_with_noise_pmf
      (p := p) (x := x) (fstar := f) (fhat := fhat)
      (u := u) (L1 := L1) (L2 := L2)
      (U := BoundedPseudoMetricSpace.diameterBound (α := S))
      hL1 hL2 hU hD
  calc |Exp p (fun z => u (f z) (fhat x)) - u (f x) (f x)|
      ≤ (L1 : ℝ) * Exp p (fun z => D f z x) +
        (L2 : ℝ) * dist (fhat x) (f x) := h_transport
    _ ≤ (L1 : ℝ) * edb.bound +
        (L2 : ℝ) * dist (fhat x) (f x) := by
          have hL1_nn : (0 : ℝ) ≤ (L1 : ℝ) := by exact_mod_cast L1.property
          have := mul_le_mul_of_nonneg_left edb.distortion_le hL1_nn
          linarith

end UniversalTransport

-- ============================================================================
-- Part 3: Constructors (The Different Producers)
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 3a: From exact local laws (bound = 0)
-- ---------------------------------------------------------------------------

section ExactLaws

variable {Y : Type*} [BoundedPseudoMetricSpace Y]

/-- Exact local laws produce a zero distortion bound.

When L1, L2, L3 all hold exactly, the ZR distribution preserves the oracle
perfectly: `E[D(f*, Z, x)] = 0`. -/
def ofExactLaws
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hp : S T = x) (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar)
    (hR : R ≥ 1) :
    ExpectedDistortionBound (ZR g x R T) fstar x where
  bound := 0
  bound_nonneg := le_refl 0
  distortion_le := by
    have := multi_round_typeclass g T x R fstar hp h1 h2 h3 hR
    linarith

end ExactLaws

-- ---------------------------------------------------------------------------
-- 3b: From approximate local laws (bound = budget)
-- ---------------------------------------------------------------------------

section ApproxLaws

variable {Y : Type*} [BoundedMetricSpace Y]

/-- Approximate local laws produce a budget-bounded distortion bound.

`E[D(f*, Z, x)] ≤ epsLeaf + epsMerge + (R-1) · epsIdemp` -/
def ofApproxLaws
    (g : Summarizer Strings) (T : BinTree Strings) (x : Strings) (R : ℕ)
    (fstar : Strings → Y)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    ExpectedDistortionBound (ZR g x R T) fstar x where
  bound :=
    hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
      ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp
  bound_nonneg := by
    have hbudget :=
      Δ_R_ZR_le_of_approx_bundle
        g T fstar x R hp hR hbound hbound_global h_mono
        hApprox.approxLocalLaws
    have hdist_nonneg : 0 ≤ Δ_R_ZR g x R T fstar := by
      unfold Δ_R_ZR Exp
      apply tsum_nonneg
      intro z; exact mul_nonneg ENNReal.toReal_nonneg dist_nonneg
    linarith
  distortion_le := by
    simpa [Δ_R_ZR] using
      Δ_R_ZR_le_of_approx_bundle
        g T fstar x R hp hR hbound hbound_global h_mono
        hApprox.approxLocalLaws

end ApproxLaws

-- ---------------------------------------------------------------------------
-- 3c: Lipschitz transfer (oracle-space → feature-space)
-- ---------------------------------------------------------------------------

section LipschitzTransfer

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Lipschitz transfer: if feature is K-Lipschitz from oracle, then
feature distortion ≤ K · oracle distortion.

This transfers an `ExpectedDistortionBound` from oracle-space to feature-space.
It is the continuous counterpart of "same-class ⟹ same-feature". -/
def lipschitzTransfer
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (feature : Strings → Feature)
    (K : ℝ≥0)
    (edb_oracle : ExpectedDistortionBound p fstar x)
    (hLip : FeatureLipschitzFromOracle fstar feature K) :
    ExpectedDistortionBound p feature x where
  bound := (K : ℝ) * edb_oracle.bound
  bound_nonneg := mul_nonneg (by exact_mod_cast K.property) edb_oracle.bound_nonneg
  distortion_le := by
    have hK := feature_distortion_le_of_featureLipschitzFromOracle
      p x fstar feature K hLip
    calc Exp p (fun z => D feature z x)
        ≤ (K : ℝ) * Exp p (fun z => D fstar z x) := hK
      _ ≤ (K : ℝ) * edb_oracle.bound := by
          apply mul_le_mul_of_nonneg_left edb_oracle.distortion_le
          exact_mod_cast K.property

end LipschitzTransfer

-- ---------------------------------------------------------------------------
-- 3d: Lipschitz + fiber transfer (the full cross-space transfer)
-- ---------------------------------------------------------------------------

section LipschitzFiberTransfer

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Lipschitz + fiber transfer: when the feature map is K-Lipschitz from the
oracle AND has ε-approximate fiber recovery, the feature distortion bound is
K · oracle_bound + ε_fiber.

This subsumes `lipschitzTransfer` (which is the ε_fiber = 0 case).

The pointwise bound is:
  D(feature, z, x) ≤ K · D(oracle, z, x) + ε_fiber

which holds because:
- When D(oracle, z, x) = 0 (same fiber): ε-recovery gives D(feature) ≤ ε_fiber
- When D(oracle, z, x) > 0 (different fiber): Lipschitz gives D(feature) ≤ K · D(oracle)
- Both ≤ K · D(oracle, z, x) + ε_fiber -/
def lipschitzFiberTransfer
    (p : PMF Strings) (x : Strings)
    (fstar : Strings → Y) (feature : Strings → Feature)
    (K : ℝ≥0) (ε_fiber : ℝ≥0)
    (edb_oracle : ExpectedDistortionBound p fstar x)
    (hLip : FeatureLipschitzFromOracle fstar feature K)
    (hFiber : ApproxOracleRecoversFeature fstar feature ε_fiber) :
    ExpectedDistortionBound p feature x where
  bound := (K : ℝ) * edb_oracle.bound + (ε_fiber : ℝ)
  bound_nonneg := by
    apply add_nonneg
    · exact mul_nonneg (by exact_mod_cast K.property) edb_oracle.bound_nonneg
    · exact_mod_cast ε_fiber.property
  distortion_le := by
    -- Step 1: Pointwise bound
    have h_pointwise : ∀ z, D feature z x ≤ (K : ℝ) * D fstar z x + (ε_fiber : ℝ) := by
      intro z
      unfold D
      by_cases h : dist (fstar z) (fstar x) = 0
      · calc dist (feature z) (feature x)
            ≤ (ε_fiber : ℝ) := hFiber z x h
          _ ≤ (K : ℝ) * dist (fstar z) (fstar x) + (ε_fiber : ℝ) := by
              linarith [mul_nonneg (show 0 ≤ (K : ℝ) from by exact_mod_cast K.property)
                (show 0 ≤ dist (fstar z) (fstar x) from dist_nonneg)]
      · calc dist (feature z) (feature x)
            ≤ (K : ℝ) * dist (fstar z) (fstar x) := hLip z x
          _ ≤ (K : ℝ) * dist (fstar z) (fstar x) + (ε_fiber : ℝ) :=
              le_add_of_nonneg_right (by exact_mod_cast ε_fiber.property)
    -- Step 2: Monotonicity of expectation
    let M_feature : ℝ := BoundedPseudoMetricSpace.diameterBound (α := Feature)
    have hM_feature : 0 ≤ M_feature := BoundedPseudoMetricSpace.diameterBound_nonneg (α := Feature)
    have hbound_feature : ∀ z, D feature z x ≤ M_feature := by
      intro z
      unfold D M_feature
      exact BoundedPseudoMetricSpace.dist_le (feature z) (feature x)
    have hD_feature :=
      summable_D_of_bounded p feature x M_feature hM_feature hbound_feature
    let M_oracle : ℝ := BoundedMetricSpace.diameterBound (α := Y)
    have hM_oracle : 0 ≤ M_oracle := BoundedMetricSpace.diameterBound_nonneg (α := Y)
    have hbound_oracle : ∀ z, D fstar z x ≤ M_oracle := by
      intro z
      unfold D M_oracle
      exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
    have hD_oracle :=
      summable_D_of_bounded p fstar x M_oracle hM_oracle hbound_oracle
    -- Summability of the RHS
    have hRHS_summable :
        Summable (fun z => (p z).toReal * ((K : ℝ) * D fstar z x + (ε_fiber : ℝ))) := by
      have hEq_split :
          (fun z => (p z).toReal * ((K : ℝ) * D fstar z x + (ε_fiber : ℝ))) =
            (fun z => (p z).toReal * ((K : ℝ) * D fstar z x) +
              (p z).toReal * (ε_fiber : ℝ)) := by
        funext z; ring
      rw [hEq_split]
      apply Summable.add
      · have hEq :
            (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) =
              (fun z => (K : ℝ) * ((p z).toReal * D fstar z x)) := by
          funext z; ring
        rw [hEq]
        exact hD_oracle.mul_left (K : ℝ)
      · exact (PMF.summable_coe_real p).mul_right (ε_fiber : ℝ)
    -- Apply Exp_mono'
    have h_mono :=
      Exp_mono' p (fun z => D feature z x)
        (fun z => (K : ℝ) * D fstar z x + (ε_fiber : ℝ))
        h_pointwise hD_feature hRHS_summable
    -- Step 3: Decompose the RHS expectation
    have hS_K : Summable (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) := by
      have hEq :
          (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) =
            (fun z => (K : ℝ) * ((p z).toReal * D fstar z x)) := by
        funext z; ring
      rw [hEq]; exact hD_oracle.mul_left (K : ℝ)
    have hS_eps : Summable (fun z => (p z).toReal * (ε_fiber : ℝ)) :=
      (PMF.summable_coe_real p).mul_right (ε_fiber : ℝ)
    have hRHS_eq :
        Exp p (fun z => (K : ℝ) * D fstar z x + (ε_fiber : ℝ)) =
          (K : ℝ) * Exp p (fun z => D fstar z x) + (ε_fiber : ℝ) := by
      rw [Exp_add p (fun z => (K : ℝ) * D fstar z x) (fun _ => (ε_fiber : ℝ))
        hS_K hS_eps]
      congr 1
      · unfold Exp
        have hEq :
            (fun z => (p z).toReal * ((K : ℝ) * D fstar z x)) =
              (fun z => (K : ℝ) * ((p z).toReal * D fstar z x)) := by
          funext z; ring
        rw [hEq, tsum_mul_left]
      · exact Exp_const p (ε_fiber : ℝ)
    -- Step 4: Combine
    calc Exp p (fun z => D feature z x)
        ≤ Exp p (fun z => (K : ℝ) * D fstar z x + (ε_fiber : ℝ)) := h_mono
      _ = (K : ℝ) * Exp p (fun z => D fstar z x) + (ε_fiber : ℝ) := hRHS_eq
      _ ≤ (K : ℝ) * edb_oracle.bound + (ε_fiber : ℝ) := by
          have hK_nn : (0 : ℝ) ≤ (K : ℝ) := by exact_mod_cast K.property
          have := mul_le_mul_of_nonneg_left edb_oracle.distortion_le hK_nn
          linarith

end LipschitzFiberTransfer

-- ---------------------------------------------------------------------------
-- 3e: Uniform surrogate lift (surrogate-space → true oracle-space)
-- ---------------------------------------------------------------------------

section SurrogateLift

variable {Y : Type*} [BoundedMetricSpace Y]

/-- Uniform surrogate lift: if f̂ uniformly approximates f* within ε, then
any distortion bound for f̂ lifts to a bound for f* with additive 2ε slack.

  E[D(f*, Z, x)] ≤ E[D(f̂, Z, x)] + 2ε ≤ surrogate_bound + 2ε

This is the core of the `TwoStageOracleSurrogate` approach. -/
def surrogateLift
    (p : PMF Strings) (x : Strings)
    (fstar fhat : Strings → Y) (ε : ℝ≥0)
    (edb_surrogate : ExpectedDistortionBound p fhat x)
    (hApprox : UniformOracleApproximation fstar fhat ε) :
    ExpectedDistortionBound p fstar x where
  bound := edb_surrogate.bound + 2 * (ε : ℝ)
  bound_nonneg := by
    apply add_nonneg edb_surrogate.bound_nonneg
    exact mul_nonneg (by norm_num) (by exact_mod_cast ε.property)
  distortion_le := by
    -- Pointwise: D(f*, z, x) ≤ D(f̂, z, x) + 2ε
    have h_pointwise : ∀ z, D fstar z x ≤ D fhat z x + 2 * (ε : ℝ) := by
      intro z
      unfold D
      exact trueOracleDist_le_of_surrogateDist_and_uniformOracleApproximation
        (hApprox := hApprox)
    -- Monotonicity + linearity of expectation
    let M : ℝ := BoundedMetricSpace.diameterBound (α := Y)
    have hM : 0 ≤ M := BoundedMetricSpace.diameterBound_nonneg (α := Y)
    have hbound_fstar : ∀ z, D fstar z x ≤ M := by
      intro z; unfold D M; exact BoundedMetricSpace.dist_le (fstar z) (fstar x)
    have hD_fstar := summable_D_of_bounded p fstar x M hM hbound_fstar
    have hbound_fhat : ∀ z, D fhat z x ≤ M := by
      intro z; unfold D M; exact BoundedMetricSpace.dist_le (fhat z) (fhat x)
    have hD_fhat := summable_D_of_bounded p fhat x M hM hbound_fhat
    have hRHS_summable :
        Summable (fun z => (p z).toReal * (D fhat z x + 2 * (ε : ℝ))) := by
      have hEq_split :
          (fun z => (p z).toReal * (D fhat z x + 2 * (ε : ℝ))) =
            (fun z => (p z).toReal * D fhat z x + (p z).toReal * (2 * (ε : ℝ))) := by
        funext z; ring
      rw [hEq_split]
      apply Summable.add
      · exact hD_fhat
      · exact (PMF.summable_coe_real p).mul_right (2 * (ε : ℝ))
    have h_mono :=
      Exp_mono' p (fun z => D fstar z x) (fun z => D fhat z x + 2 * (ε : ℝ))
        h_pointwise hD_fstar hRHS_summable
    have hRHS_eq :
        Exp p (fun z => D fhat z x + 2 * (ε : ℝ)) =
          Exp p (fun z => D fhat z x) + 2 * (ε : ℝ) := by
      rw [Exp_add p (fun z => D fhat z x) (fun _ => 2 * (ε : ℝ)) hD_fhat
        (by apply Summable.mul_right; exact PMF.summable_coe_real p)]
      simp [Exp_const]
    calc Exp p (fun z => D fstar z x)
        ≤ Exp p (fun z => D fhat z x + 2 * (ε : ℝ)) := h_mono
      _ = Exp p (fun z => D fhat z x) + 2 * (ε : ℝ) := hRHS_eq
      _ ≤ edb_surrogate.bound + 2 * (ε : ℝ) := by
          linarith [edb_surrogate.distortion_le]

end SurrogateLift

-- ============================================================================
-- Part 4: Composition Operations
-- ============================================================================

section Composition

variable {S : Type*} [PseudoMetricSpace S]

/-- Monotonicity: a tighter distortion bound is also valid. -/
def weaken
    (p : PMF Strings) (f : Strings → S) (x : Strings)
    (edb : ExpectedDistortionBound p f x)
    (bound' : ℝ) (h_nonneg : 0 ≤ bound') (h_le : edb.bound ≤ bound') :
    ExpectedDistortionBound p f x where
  bound := bound'
  bound_nonneg := h_nonneg
  distortion_le := le_trans edb.distortion_le h_le

end Composition

-- ============================================================================
-- Part 5: Route Comparison / Dominance
-- ============================================================================

section RouteDominance

variable {S : Type*} [BoundedPseudoMetricSpace S]

/-- Route dominance: if one route produces a tighter distortion bound,
it gives a tighter utility bound for ANY Lipschitz utility.

This is the formal basis for comparing oracle approximation approaches:
the route with the smaller `bound` is uniformly better. -/
theorem route_dominance
    {p : PMF Strings} {f : Strings → S} {x : Strings}
    (fhat : Strings → S) (L1 L2 : ℝ≥0)
    (edb1 edb2 : ExpectedDistortionBound p f x)
    (h_tighter : edb1.bound ≤ edb2.bound) :
    (L1 : ℝ) * edb1.bound + (L2 : ℝ) * dist (fhat x) (f x) ≤
    (L1 : ℝ) * edb2.bound + (L2 : ℝ) * dist (fhat x) (f x) := by
  have hL1_nn : (0 : ℝ) ≤ (L1 : ℝ) := by exact_mod_cast L1.property
  have := mul_le_mul_of_nonneg_left h_tighter hL1_nn
  linarith

end RouteDominance

-- ============================================================================
-- Part 6: Existing Theorems as Instances
-- ============================================================================

section Instances

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- `expected_utility_bound_approx_fiber` is the composition:
  `ofApproxLaws` ∘ `lipschitzFiberTransfer` ∘ `universalTransport`

Given:
- Approximate local laws w.r.t. f* → ODB with budget
- K-Lipschitz + ε-fiber transfer → FDB with K·budget + ε_fiber
- Universal transport → utility gap ≤ L₁·(K·budget + ε_fiber) + L₂·measurement -/
theorem approx_fiber_as_composed_route
    (fstar : Strings → Y) (feature : Strings → Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K : ℝ≥0) (ε_fiber : ℝ≥0)
    (hp : S T = x)
    (hApprox : ApproxTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (hLip : FeatureLipschitzFromOracle fstar feature K)
    (hFiber : ApproxOracleRecoversFeature fstar feature ε_fiber)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    -- Step 1: Oracle distortion bound from approximate laws
    let edb_oracle := ofApproxLaws g T x R fstar hp hApprox hR hbound hbound_global h_mono
    -- Step 2: Feature distortion bound from Lipschitz + fiber
    let edb_feature := lipschitzFiberTransfer (ZR g x R T) x fstar feature K ε_fiber
      edb_oracle hLip hFiber
    -- The composed bound matches the structure of expected_utility_bound_approx_fiber
    edb_feature.bound =
      (K : ℝ) *
        (hApprox.approxLocalLaws.epsLeaf + hApprox.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox.approxLocalLaws.epsIdemp) +
      (ε_fiber : ℝ) := by
  -- The bound is K · budget + ε_fiber by construction
  rfl

/-- The uniform surrogate route from `TwoStageOracleSurrogate` is:
  `ofApproxLaws` (for f̂) ∘ `surrogateLift` (f̂ → f*)

The composed bound is: budget_f̂ + 2ε -/
theorem surrogate_as_composed_route
    (fstar fhat : Strings → Y) (ε : ℝ≥0)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hApproxBacked : ApproxTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    let edb_surrogate := ofApproxLaws g T x R fhat hp hApproxBacked hR hbound hbound_global h_mono
    let edb_true := surrogateLift (ZR g x R T) x fstar fhat ε edb_surrogate hApprox
    edb_true.bound =
      (hApproxBacked.approxLocalLaws.epsLeaf + hApproxBacked.approxLocalLaws.epsMerge +
        ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
      2 * (ε : ℝ) := by
  rfl

end Instances

-- ============================================================================
-- Part 7: Cross-Route Comparison
-- ============================================================================

section CrossRouteComparison

variable {Y : Type*} [BoundedMetricSpace Y]

/-- **Fiber route vs surrogate route comparison.**

The fiber route gives: K · budget + ε_fiber
The surrogate route gives: budget_surrogate + 2ε

The fiber route is tighter when:
  K · budget + ε_fiber ≤ budget_surrogate + 2ε

Key insight: the fiber route can win even with larger K when the fiber error
ε_fiber is small (achieved by good contrastive training) and the budget is
small (tight local laws). The surrogate route wins when the surrogate is
very close (small ε) and K is large.

In the LLM two-stage setting:
- The fiber route corresponds to contrastive feature learning (stage 1 learns
  to preserve oracle equivalence classes)
- The surrogate route corresponds to direct oracle distillation (stage 1 learns
  to reproduce oracle outputs pointwise)
- The fiber route is preferred when the oracle space Y has high dimension or
  complex structure, because fiber preservation is a weaker (easier) goal
  than pointwise reproduction -/
theorem fiber_vs_surrogate_comparison
    (budget_fiber budget_surrogate : ℝ)
    (K : ℝ≥0) (ε_fiber ε_surrogate : ℝ≥0)
    (h_fiber_tighter :
      (K : ℝ) * budget_fiber + (ε_fiber : ℝ) ≤
      budget_surrogate + 2 * (ε_surrogate : ℝ)) :
    (K : ℝ) * budget_fiber + (ε_fiber : ℝ) ≤
    budget_surrogate + 2 * (ε_surrogate : ℝ) :=
  h_fiber_tighter

/-- **Exact route dominates all routes** (trivially: bound = 0).

This formalizes the obvious fact that exact local laws give the best bound,
but at the cost of requiring the strongest assumptions. The approximation
approaches trade bound quality for weaker assumptions. -/
theorem exact_dominates_all
    (p : PMF Strings) {S : Type*} [PseudoMetricSpace S] (f : Strings → S) (x : Strings)
    (edb : ExpectedDistortionBound p f x) :
    (0 : ℝ) ≤ edb.bound :=
  edb.bound_nonneg

end CrossRouteComparison

end ExpectedDistortionBound

end FormalProofs.OPT
