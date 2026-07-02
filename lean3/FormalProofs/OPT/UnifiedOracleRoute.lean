import FormalProofs.OPT.OracleFiberObjectives
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.OPT.OracleFibers

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

/-! ## From TwoStageDecomposition.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/TwoStageDecomposition.lean

Two-stage oracle approximation: formalize the decomposition where

  **Stage 1**: Learn φ̂ : Strings → Φ ≈ f* (expensive oracle approximation)
  **Stage 2**: Learn g that satisfies local laws relative to φ̂ (not f*)

This is the standard approach in LLM preference tuning: use a large model as f*
in stage 1 to train a feature map φ̂, then optimize a summarizer g purely for
φ̂'s scores — never touching f* again in stage 2.

## Type structure

- `Y` : true oracle output space (`BoundedMetricSpace`)
- `Φ` : intermediate approximation space from stage 1 (`BoundedMetricSpace`)
- `Feature` : downstream embedding space (`BoundedPseudoMetricSpace`)

Stage 1: `f* : Strings → Y` → learn `φ̂ : Strings → Φ` such that
  `ApproxOracleRecoversFeature f* φ̂ ε₁`

Stage 2: treat `φ̂` as oracle, apply `expected_utility_bound_approx_fiber`
  with `φ̂ : Strings → Φ` playing the role of `fstar`.

## Key results

1. **`TwoStageOracleApproximation`**: Structure packaging stage 1's output.

2. **`stage2_utility_bound`**: Stage 2 bound in Φ-space — just
   `expected_utility_bound_approx_fiber` with Φ as oracle type.

3. **`two_stage_full_end_to_end_bound`**: End-to-end bound relating
   stage 2 utility back to true oracle reference point.

4. **`two_stage_breakeven_condition`**: When two-stage beats single-stage.
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
variable {Y : Type*} [BoundedMetricSpace Y]

-- ============================================================================
-- Section 1: Two-Stage Structure
-- ============================================================================

section TwoStageStructure

variable {Φ : Type*} [BoundedPseudoMetricSpace Φ]

/-- A two-stage oracle approximation packages:
- A learned oracle approximation φ̂ : Strings → Φ (stage 1 output)
- A guarantee that φ̂ is ε₁-close to f* on same-fiber pairs
- A Lipschitz bound K₁ relating φ̂-distances to f*-distances

Stage 1 is "expensive" (uses the true oracle f* for training).
Stage 2 will use φ̂ as if it were the oracle, never touching f* again. -/
structure TwoStageOracleApproximation
    (fstar : Strings → Y) (phiHat : Strings → Φ) where
  /-- Stage 1 approximation quality: on same-fiber pairs, φ̂ agrees within ε₁. -/
  eps_stage1 : ℝ≥0
  /-- Stage 1 fiber recovery guarantee. -/
  approxRecover : ApproxOracleRecoversFeature fstar phiHat eps_stage1
  /-- Lipschitz bound: φ̂ doesn't expand distances relative to f*. -/
  lipschitz_K : ℝ≥0
  featureLip : FeatureLipschitzFromOracle fstar phiHat lipschitz_K

end TwoStageStructure

-- ============================================================================
-- Section 2: Stage 2 Bound (in Φ-space)
-- ============================================================================

section Stage2Bound

/- The intermediate oracle space Φ must be a BoundedMetricSpace (not just
PseudoMetric) to serve as the oracle in `expected_utility_bound_approx_fiber`.
This is the type-theoretic price of two-stage: the intermediate representation
must be a proper metric space. -/
variable {Φ : Type*} [BoundedMetricSpace Φ]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Stage 2 utility bound: g is optimized relative to φ̂ (not f*).

This is simply `expected_utility_bound_approx_fiber` instantiated with
Φ as the oracle type and φ̂ as the oracle function. The existing theorem
is fully parametric in the oracle — we just swap in φ̂ for f*. -/
theorem stage2_utility_bound_in_phiHat_space
    (phiHat : Strings → Φ)
    (feature2 featureHat2 : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K₂ : ℝ≥0) (ε₂ : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hR : R ≥ 1)
    (hApprox2 : ApproxTheoremBacked g T phiHat)
    (hApproxRecover2 : ApproxOracleRecoversFeature phiHat feature2 ε₂)
    (hFeatureLip2 : FeatureLipschitzFromOracle phiHat feature2 K₂)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature2 x) U)
    (hbound : ∀ z, D phiHat z x ≤ 1)
    (hbound_global : ∀ w z, D phiHat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g phiHat (p.bind g) ≤ pIdemp g phiHat p) :
    |Exp (ZR g x R T) (fun z => u (feature2 z) (featureHat2 x)) -
        u (feature2 x) (feature2 x)| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) :=
  expected_utility_bound_approx_fiber
    phiHat feature2 featureHat2 u g x R T K₂ ε₂ L1 L2 U
    hp hApprox2 hR hApproxRecover2 hFeatureLip2 hL1 hL2 hU
    hbound hbound_global h_mono

end Stage2Bound

-- ============================================================================
-- Section 3: Utility Oracle Substitution Lemma
-- ============================================================================

section SubstitutionCost

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Triangle inequality bridge: the gap between "utility at a" and "utility at b"
(both arguments equal) is controlled by the sum of Lipschitz constants.

  |u(a, a) - u(b, b)| ≤ (L₁ + L₂) · dist(a, b)

This is the "cost" of evaluating utility at a surrogate point instead of the
true reference. It connects stage 2 (measured in Φ-space) back to f*-space. -/
theorem utility_oracle_substitution_cost
    (u : OracleUtility2 Feature)
    (a b : Feature)
    (L1 L2 : ℝ≥0)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2) :
    |u a a - u b b| ≤ ((L1 : ℝ) + (L2 : ℝ)) * dist a b := by
  -- Split: u(a,a) - u(b,b) = (u(a,a) - u(a,b)) + (u(a,b) - u(b,b))
  have h_split : u a a - u b b = (u a a - u a b) + (u a b - u b b) := by ring
  rw [h_split]
  calc |(u a a - u a b) + (u a b - u b b)|
      ≤ |u a a - u a b| + |u a b - u b b| := abs_add_le _ _
    _ ≤ (L2 : ℝ) * dist a b + (L1 : ℝ) * dist a b := by
        apply add_le_add
        · exact hL2 a a b
        · exact hL1 a b b
    _ = ((L1 : ℝ) + (L2 : ℝ)) * dist a b := by ring

end SubstitutionCost

-- ============================================================================
-- Section 4: End-to-End Composition
-- ============================================================================

section EndToEnd

variable {Φ : Type*} [BoundedMetricSpace Φ]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]

/-- Full end-to-end two-stage bound with the true-utility reference point.

Given:
- Stage 1: φ̂ : Strings → Φ approximates f* : Strings → Y
- Stage 2: g satisfies local laws w.r.t. φ̂, feature2 recovers Φ-fibers
- A "reference feature" feature_ref : Strings → Feature that we compare against

The bound decomposes as:

  |E[u(feature2(Z), featureHat2(x))] - u(feature_ref(x), feature_ref(x))| ≤
      L₁ · K₂ · (stage2 transport budget)
    + L₁ · ε₂
    + L₂ · dist(featureHat2(x), feature2(x))
    + (L₁ + L₂) · dist(feature2(x), feature_ref(x))

The first three terms come from stage 2 (transport bound with φ̂ as oracle).
The last term is the oracle substitution cost — the "price" of measuring
utility at feature2(x) instead of feature_ref(x).

When feature_ref = the "ideal" f*-aligned feature and feature2 = the φ̂-aligned
feature, the last term captures the stage 1 approximation quality. -/
theorem two_stage_full_end_to_end_bound
    (phiHat : Strings → Φ)
    (feature_ref : Strings → Feature)
    (feature2 featureHat2 : Strings → Feature)
    (u : OracleUtility2 Feature)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K₂ : ℝ≥0) (ε₂ : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hR : R ≥ 1)
    (hApprox2 : ApproxTheoremBacked g T phiHat)
    (hApproxRecover2 : ApproxOracleRecoversFeature phiHat feature2 ε₂)
    (hFeatureLip2 : FeatureLipschitzFromOracle phiHat feature2 K₂)
    (hL1 : OracleUtilityLipschitz1 u L1)
    (hL2 : OracleUtilityLipschitz2 u L2)
    (hU : OracleUtilityBoundedAt u (feature2 x) U)
    (hbound : ∀ z, D phiHat z x ≤ 1)
    (hbound_global : ∀ w z, D phiHat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g phiHat (p.bind g) ≤ pIdemp g phiHat p) :
    |Exp (ZR g x R T) (fun z => u (feature2 z) (featureHat2 x)) -
        u (feature_ref x) (feature_ref x)| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) +
      ((L1 : ℝ) + (L2 : ℝ)) * dist (feature2 x) (feature_ref x) := by
  -- Let A = E[u(feature2(Z), featureHat2(x))]
  --     B = u(feature2(x), feature2(x))
  --     C = u(feature_ref(x), feature_ref(x))
  -- We want |A - C| ≤ |A - B| + |B - C|
  set A := Exp (ZR g x R T) (fun z => u (feature2 z) (featureHat2 x))
  set B := u (feature2 x) (feature2 x)
  set C := u (feature_ref x) (feature_ref x)
  have h_triangle : |A - C| ≤ |A - B| + |B - C| := by
    have h_split : A - C = (A - B) + (B - C) := by ring
    rw [h_split]
    exact abs_add_le _ _
  -- Bound term 1: |A - B| via stage 2 transport (φ̂ as oracle)
  have h_stage2 : |A - B| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) :=
    expected_utility_bound_approx_fiber
      phiHat feature2 featureHat2 u g x R T K₂ ε₂ L1 L2 U
      hp hApprox2 hR hApproxRecover2 hFeatureLip2 hL1 hL2 hU
      hbound hbound_global h_mono
  -- Bound term 2: |B - C| via oracle substitution cost
  have h_subst : |B - C| ≤
      ((L1 : ℝ) + (L2 : ℝ)) * dist (feature2 x) (feature_ref x) :=
    utility_oracle_substitution_cost u (feature2 x) (feature_ref x) L1 L2 hL1 hL2
  linarith

end EndToEnd

-- ============================================================================
-- Section 5: Two-Stage vs Single-Stage Comparison
-- ============================================================================

section Comparison

/-- Algebraic comparison of two-stage vs single-stage bounds.

**Single-stage bound** (from `expected_utility_bound_approx_fiber` with oracle = f*):

  B_single = L₁ · K · budget_direct + L₁ · ε_fiber + L₂ · measurement_direct

**Two-stage bound** (from `two_stage_full_end_to_end_bound`):

  B_two = L₁ · K₂ · budget_stage2 + L₁ · ε₂ + L₂ · measurement₂
        + (L₁ + L₂) · substitution_cost

Two-stage is tighter when B_two ≤ B_single. The breakeven condition:
the savings from a tighter stage 2 budget must exceed the stage 1 cost.

**Why two-stage can win**:
1. Stage 2 optimizes for φ̂ (a known, fixed target) → tighter local laws
2. A larger/better model can be used in stage 1 (one-time cost amortized)
3. Stage 2's contrastive training can use φ̂ directly for labeling (no oracle calls)

**Why single-stage can win**:
1. No substitution cost term — directly optimizes for f*
2. Fewer hyperparameters (no stage 1 quality to manage)
3. When f* is cheap to evaluate, the two-stage overhead isn't worth it -/
theorem two_stage_breakeven_condition
    (budget_direct budget_stage2 : ℝ≥0)
    (substitution_cost ε_fiber_direct ε₂ : ℝ≥0)
    (K_direct K₂ L1 L2 : ℝ≥0)
    (measurement_direct measurement2 : ℝ)
    -- Two-stage is better when: savings ≥ substitution cost
    (h_savings :
      (L1 : ℝ) * (K_direct : ℝ) * (budget_direct : ℝ) +
        (L1 : ℝ) * (ε_fiber_direct : ℝ) +
        (L2 : ℝ) * measurement_direct -
      ((L1 : ℝ) * (K₂ : ℝ) * (budget_stage2 : ℝ) +
        (L1 : ℝ) * (ε₂ : ℝ) +
        (L2 : ℝ) * measurement2) ≥
      ((L1 : ℝ) + (L2 : ℝ)) * (substitution_cost : ℝ)) :
    (L1 : ℝ) * (K₂ : ℝ) * (budget_stage2 : ℝ) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * measurement2 +
      ((L1 : ℝ) + (L2 : ℝ)) * (substitution_cost : ℝ) ≤
    (L1 : ℝ) * (K_direct : ℝ) * (budget_direct : ℝ) +
      (L1 : ℝ) * (ε_fiber_direct : ℝ) +
      (L2 : ℝ) * measurement_direct := by
  linarith

/-- When stage 1 is perfect (substitution_cost = 0), two-stage reduces to
single-stage with the stage 2 budget. Two-stage strictly generalizes
single-stage. -/
theorem two_stage_perfect_stage1_reduces
    (budget_stage2 ε₂ : ℝ≥0)
    (K₂ L1 L2 : ℝ≥0)
    (measurement2 : ℝ) :
    (L1 : ℝ) * (K₂ : ℝ) * (budget_stage2 : ℝ) +
    (L1 : ℝ) * (ε₂ : ℝ) +
    (L2 : ℝ) * measurement2 +
    ((L1 : ℝ) + (L2 : ℝ)) * ((0 : ℝ≥0) : ℝ) =
    (L1 : ℝ) * (K₂ : ℝ) * (budget_stage2 : ℝ) +
    (L1 : ℝ) * (ε₂ : ℝ) +
    (L2 : ℝ) * measurement2 := by
  simp [NNReal.coe_zero, mul_zero, add_zero]

end Comparison

-- ============================================================================
-- Section 6: Stage 1 Construction from Contrastive Training
-- ============================================================================

section Stage1Contrastive

variable {Φ : Type*} [BoundedPseudoMetricSpace Φ]

/- Stage 1 learning guarantee via contrastive fiber loss.

If stage 1 trains φ̂ by minimizing the contrastive fiber loss w.r.t. f*, and
achieves bounded population risk, then φ̂ is an approximate oracle recovery.
This connects `FiberPreservingObjective.lean`'s contrastive risk theorems
to the two-stage pipeline. -/
def stage1_contrastive_yields_two_stage_input
    (fstar : Strings → Y)
    (phiHat : Strings → Φ)
    (ε₁ : ℝ≥0)
    (hRecover : ApproxOracleRecoversFeature fstar phiHat ε₁)
    (K₁ : ℝ≥0)
    (hLip : FeatureLipschitzFromOracle fstar phiHat K₁) :
    TwoStageOracleApproximation fstar phiHat where
  eps_stage1 := ε₁
  approxRecover := hRecover
  lipschitz_K := K₁
  featureLip := hLip

end Stage1Contrastive

-- ============================================================================
-- Section 7: Multi-Stage Distillation
-- ============================================================================

section MultiStage

variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]
variable {Feature1 : Type*} [BoundedPseudoMetricSpace Feature1]
variable {Feature2 : Type*} [BoundedPseudoMetricSpace Feature2]

/-- Multi-stage distillation: chaining two approximate oracle recoveries
(f* → φ̂₁ → φ̂₂) composes via Lipschitz.

If φ̂₁ is ε₁-close to f* on same-fiber pairs, and φ̂₂ is K-Lipschitz
relative to φ̂₁, then φ̂₂ is (K·ε₁)-close to f* on same-fiber pairs.

This shows the cost of adding distillation stages: each K-Lipschitz
intermediary multiplies the approximation error by K. For contractive
maps (K ≤ 1), distillation doesn't amplify error. -/
theorem distillation_chain_error
    (fstar : Strings → Y)
    (phi1 phi2 : Strings → Feature)
    (ε₁ : ℝ≥0) (K : ℝ≥0)
    (h1 : ApproxOracleRecoversFeature fstar phi1 ε₁)
    (hLip : ∀ x x', dist (phi2 x) (phi2 x') ≤ (K : ℝ) * dist (phi1 x) (phi1 x'))
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    dist (phi2 x) (phi2 x') ≤ (K : ℝ) * (ε₁ : ℝ) := by
  calc dist (phi2 x) (phi2 x')
      ≤ (K : ℝ) * dist (phi1 x) (phi1 x') := hLip x x'
    _ ≤ (K : ℝ) * (ε₁ : ℝ) := by
        apply mul_le_mul_of_nonneg_left (h1 x x' hzero)
        exact_mod_cast K.property

/-- Contractive distillation: when K ≤ 1, the chain doesn't amplify errors
and each stage's output is at most ε-close to f* on same-fiber pairs. -/
theorem contractive_distillation_chain
    (fstar : Strings → Y)
    (phi1 phi2 : Strings → Feature)
    (ε : ℝ≥0)
    (h1 : ApproxOracleRecoversFeature fstar phi1 ε)
    (hLip : ∀ x x', dist (phi2 x) (phi2 x') ≤ dist (phi1 x) (phi1 x'))
    {x x' : Strings}
    (hzero : dist (fstar x) (fstar x') = 0) :
    dist (phi2 x) (phi2 x') ≤ (ε : ℝ) := by
  calc dist (phi2 x) (phi2 x')
      ≤ dist (phi1 x) (phi1 x') := hLip x x'
    _ ≤ (ε : ℝ) := h1 x x' hzero

/-- Distillation packaged back into the approximate oracle-recovery interface.

If a stage-1 representation `phi1` already approximately recovers `f*`, and a
downstream representation `phi2` is `K`-Lipschitz with respect to `phi1`, then
`phi2` also approximately recovers `f*` with error scaled by `K`. -/
theorem approxOracleRecoversFeature_of_distillation_chain
    (fstar : Strings → Y)
    (phi1 : Strings → Feature1)
    (phi2 : Strings → Feature2)
    (ε₁ : ℝ≥0) (K : ℝ≥0)
    (h1 : ApproxOracleRecoversFeature fstar phi1 ε₁)
    (hLip : ∀ x x', dist (phi2 x) (phi2 x') ≤ (K : ℝ) * dist (phi1 x) (phi1 x')) :
    ApproxOracleRecoversFeature fstar phi2 (K * ε₁) := by
  intro x x' hzero
  calc dist (phi2 x) (phi2 x')
      ≤ (K : ℝ) * dist (phi1 x) (phi1 x') := hLip x x'
    _ ≤ (K : ℝ) * (ε₁ : ℝ) := by
        apply mul_le_mul_of_nonneg_left (h1 x x' hzero)
        exact_mod_cast K.property
    _ = ((K * ε₁ : ℝ≥0) : ℝ) := by
        simp [NNReal.coe_mul]

/-- Contractive distillation packaged back into approximate oracle recovery. -/
theorem approxOracleRecoversFeature_of_contractive_distillation_chain
    (fstar : Strings → Y)
    (phi1 : Strings → Feature1)
    (phi2 : Strings → Feature2)
    (ε : ℝ≥0)
    (h1 : ApproxOracleRecoversFeature fstar phi1 ε)
    (hLip : ∀ x x', dist (phi2 x) (phi2 x') ≤ dist (phi1 x) (phi1 x')) :
    ApproxOracleRecoversFeature fstar phi2 ε := by
  intro x x' hzero
  calc dist (phi2 x) (phi2 x')
      ≤ dist (phi1 x) (phi1 x') := hLip x x'
    _ ≤ (ε : ℝ) := h1 x x' hzero

/-- A packaged stage-1 `TwoStageOracleApproximation` can be pushed through one
more Lipschitz distillation layer to yield a direct approximate-recovery
statement for the downstream representation. -/
theorem approxOracleRecoversFeature_of_twoStageOracleApproximation_and_distillation
    {Φ : Type*} [BoundedPseudoMetricSpace Φ]
    (fstar : Strings → Y)
    (phiHat : Strings → Φ)
    (stage1 : TwoStageOracleApproximation fstar phiHat)
    (phi2 : Strings → Feature2)
    (K : ℝ≥0)
    (hLip : ∀ x x', dist (phi2 x) (phi2 x') ≤ (K : ℝ) * dist (phiHat x) (phiHat x')) :
    ApproxOracleRecoversFeature fstar phi2 (K * stage1.eps_stage1) := by
  exact approxOracleRecoversFeature_of_distillation_chain
    fstar phiHat phi2 stage1.eps_stage1 K stage1.approxRecover hLip

end MultiStage

end FormalProofs.OPT

end
end

/-! ## From TwoStageLabelScoreObjectives.lean (consolidated 2026-07-02) -/

section

/-!
# FormalProofs/OPT/TwoStageLabelScoreObjectives.lean

Two-stage transport theorems specialized to arbitrary scores on decoded labels.

This file makes two routes explicit.

1. **Direct surrogate-oracle route**:
   stage 1 learns a surrogate oracle `f̂` that is uniformly close to the true
   oracle `f*`, and stage 2 learns a tree summary relative to `f̂`.

2. **Layered shared-feature route**:
   stage 1 learns a surrogate theorem feature / oracle `φ̂`, stage 2 learns the
   tree relative to `φ̂`, and downstream task/summary heads evaluate arbitrary
   scores on labels decoded from a learned feature.

The first route captures the simple teacher-first pipeline. The second route is
the more general decomposition that exposes the tradeoff terms explicitly:
transport budget, stage-2 fiber error, measurement error, and stage-1
substitution cost.
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

section DirectSurrogateOracle

variable {Y : Type*} [BoundedMetricSpace Y]
variable {Label : Type*}

/-- Exact theorem-backedness for a learned surrogate oracle yields a direct bound
for arbitrary scores on labels decoded from the true oracle output. -/
theorem expected_trueLabelScoreUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
    (fstar fhat : Strings → Y)
    (labelOf : Y → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L : ℝ≥0) (ε : ℝ≥0)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hL : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L) :
    |Exp (ZR g x R T) (fun z => labelScoreUtility labelOf score (fstar z) (fstar x)) -
        labelScoreUtility labelOf score (fstar x) (fstar x)| ≤
      (L : ℝ) * (2 * (ε : ℝ)) := by
  simpa [labelScoreUtility] using
    (expected_trueOracleUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (u := labelScoreUtility labelOf score)
      (L := L) (hp := hp) (hExact := hExact) (hR := hR)
      (hApprox := hApprox) (hL := hL))

/-- Approximate theorem-backedness for a learned surrogate oracle yields a true
label-score gap bounded by the surrogate transport budget plus the additive
stage-1 surrogate slack. -/
theorem expected_trueLabelScoreUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
    (fstar fhat : Strings → Y)
    (labelOf : Y → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (L : ℝ≥0) (ε : ℝ≥0)
    (hp : S T = x)
    (hApproxBacked : ApproxTheoremBacked g T fhat)
    (hR : R ≥ 1)
    (hApprox : UniformOracleApproximation fstar fhat ε)
    (hL : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L)
    (hbound : ∀ z, D fhat z x ≤ 1)
    (hbound_global : ∀ w z, D fhat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fhat (p.bind g) ≤ pIdemp g fhat p) :
    |Exp (ZR g x R T) (fun z => labelScoreUtility labelOf score (fstar z) (fstar x)) -
        labelScoreUtility labelOf score (fstar x) (fstar x)| ≤
      (L : ℝ) *
        ((hApproxBacked.approxLocalLaws.epsLeaf +
          hApproxBacked.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApproxBacked.approxLocalLaws.epsIdemp) +
        2 * (ε : ℝ)) := by
  simpa [labelScoreUtility] using
    (expected_trueOracleUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation
      (u := labelScoreUtility labelOf score)
      (L := L) (hp := hp) (hApproxBacked := hApproxBacked)
      (hR := hR) (hApprox := hApprox) (hL := hL)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

end DirectSurrogateOracle

section LayeredSharedFeature

variable {Φ : Type*} [BoundedMetricSpace Φ]
variable {Feature : Type*} [BoundedPseudoMetricSpace Feature]
variable {Label : Type*}

/-- Stage-2 transport in the surrogate feature space, specialized to arbitrary
scores on decoded labels. -/
theorem expected_labelScoreUtility_bound_in_surrogateFeatureSpace
    (phiHat : Strings → Φ)
    (feature2 featureHat2 : Strings → Feature)
    (labelOf : Feature → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K₂ : ℝ≥0) (ε₂ : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hR : R ≥ 1)
    (hApprox2 : ApproxTheoremBacked g T phiHat)
    (hApproxRecover2 : ApproxOracleRecoversFeature phiHat feature2 ε₂)
    (hFeatureLip2 : FeatureLipschitzFromOracle phiHat feature2 K₂)
    (hL1 : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L1)
    (hL2 : OracleUtilityLipschitz2 (labelScoreUtility labelOf score) L2)
    (hU : OracleUtilityBoundedAt (labelScoreUtility labelOf score) (feature2 x) U)
    (hbound : ∀ z, D phiHat z x ≤ 1)
    (hbound_global : ∀ w z, D phiHat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g phiHat (p.bind g) ≤ pIdemp g phiHat p) :
    |Exp (ZR g x R T)
        (fun z => labelScoreUtility labelOf score (feature2 z) (featureHat2 x)) -
        labelScoreUtility labelOf score (feature2 x) (feature2 x)| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) := by
  simpa [labelScoreUtility] using
    (stage2_utility_bound_in_phiHat_space
      (phiHat := phiHat)
      (feature2 := feature2) (featureHat2 := featureHat2)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      (K₂ := K₂) (ε₂ := ε₂) (L1 := L1) (L2 := L2) (U := U)
      (hp := hp) (hR := hR) (hApprox2 := hApprox2)
      (hApproxRecover2 := hApproxRecover2)
      (hFeatureLip2 := hFeatureLip2)
      (hL1 := hL1) (hL2 := hL2) (hU := hU)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

/-- Full two-stage end-to-end decomposition for arbitrary label-score
objectives. This is the general teacher-first route with an explicit stage-1
substitution term. -/
theorem expected_labelScoreUtility_two_stage_end_to_end_bound
    (phiHat : Strings → Φ)
    (feature_ref : Strings → Feature)
    (feature2 featureHat2 : Strings → Feature)
    (labelOf : Feature → Label)
    (score : Label → Label → ℝ)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (K₂ : ℝ≥0) (ε₂ : ℝ≥0) (L1 L2 : ℝ≥0) (U : ℝ)
    (hp : S T = x)
    (hR : R ≥ 1)
    (hApprox2 : ApproxTheoremBacked g T phiHat)
    (hApproxRecover2 : ApproxOracleRecoversFeature phiHat feature2 ε₂)
    (hFeatureLip2 : FeatureLipschitzFromOracle phiHat feature2 K₂)
    (hL1 : OracleUtilityLipschitz1 (labelScoreUtility labelOf score) L1)
    (hL2 : OracleUtilityLipschitz2 (labelScoreUtility labelOf score) L2)
    (hU : OracleUtilityBoundedAt (labelScoreUtility labelOf score) (feature2 x) U)
    (hbound : ∀ z, D phiHat z x ≤ 1)
    (hbound_global : ∀ w z, D phiHat w z ≤ 1)
    (h_mono : ∀ p, pIdemp g phiHat (p.bind g) ≤ pIdemp g phiHat p) :
    |Exp (ZR g x R T)
        (fun z => labelScoreUtility labelOf score (feature2 z) (featureHat2 x)) -
        labelScoreUtility labelOf score (feature_ref x) (feature_ref x)| ≤
      (L1 : ℝ) * (K₂ : ℝ) *
        (hApprox2.approxLocalLaws.epsLeaf + hApprox2.approxLocalLaws.epsMerge +
          ((R : ℝ) - 1) * hApprox2.approxLocalLaws.epsIdemp) +
      (L1 : ℝ) * (ε₂ : ℝ) +
      (L2 : ℝ) * dist (featureHat2 x) (feature2 x) +
      ((L1 : ℝ) + (L2 : ℝ)) * dist (feature2 x) (feature_ref x) := by
  simpa [labelScoreUtility] using
    (two_stage_full_end_to_end_bound
      (phiHat := phiHat)
      (feature_ref := feature_ref)
      (feature2 := feature2) (featureHat2 := featureHat2)
      (u := labelScoreUtility labelOf score)
      (g := g) (x := x) (R := R) (T := T)
      (K₂ := K₂) (ε₂ := ε₂) (L1 := L1) (L2 := L2) (U := U)
      (hp := hp) (hR := hR) (hApprox2 := hApprox2)
      (hApproxRecover2 := hApproxRecover2)
      (hFeatureLip2 := hFeatureLip2)
      (hL1 := hL1) (hL2 := hL2) (hU := hU)
      (hbound := hbound) (hbound_global := hbound_global) (h_mono := h_mono))

end LayeredSharedFeature

end FormalProofs.OPT

end
end

/-! ## Original UnifiedOracleRoute content -/

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
