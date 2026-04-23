import FormalProofs.OPT.ApproxFiberTransport
import FormalProofs.OPT.FiberPreservingObjective

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
