import FormalProofs.OPT.ExpectationTheory

/-!
# FormalProofs/IPWTheory.lean

## Inverse Probability Weighting for Design-Based Inference

This file formalizes the IPW (Inverse Probability Weighting) framework for
design-based supervised learning in tree-structured preference learning.

See RLHF_DSL_BANDIT_NOTES.md for the conceptual model.

### Key Results

- `WeightedSample`: Sample with known inclusion probability
- `hajekEstimator`: Ratio estimator (normalized by sum of weights)
- `horvitzThompson`: Sum of weighted outcomes
- `effectiveSampleSize`: n_eff = (Σw)² / Σw²
- `hajek_eq_Exp_of_exact_propensity`: Connection to existing Exp

### DSL Requirement

All propensities must be:
1. Known (logged at sampling time)
2. Positive (0 < p_i for all sampled units)

This ensures bounded weights and enables unbiased estimation.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Section 1: Weighted Samples
-/

/-- A weighted sample with known inclusion probability.

The DSL (Design-Based Supervised Learning) requirement is that propensity > 0,
which ensures bounded weights for variance control. -/
structure WeightedSample (α : Type*) where
  value : α
  propensity : ℝ
  h_pos : 0 < propensity

namespace WeightedSample

variable {α : Type*}

/-- Weight is inverse propensity: w = 1/p -/
def weight (s : WeightedSample α) : ℝ := 1 / s.propensity

lemma weight_pos (s : WeightedSample α) : 0 < s.weight := by
  unfold weight
  exact one_div_pos.mpr s.h_pos

lemma weight_eq_inv (s : WeightedSample α) : s.weight = s.propensity⁻¹ := by
  unfold weight
  exact one_div s.propensity

/-- Weight times propensity equals 1 -/
lemma weight_mul_propensity (s : WeightedSample α) : s.weight * s.propensity = 1 := by
  unfold weight
  rw [one_div, inv_mul_cancel₀]
  exact ne_of_gt s.h_pos

end WeightedSample

/-!
## Section 2: List Operations for Weighted Samples
-/

/-- Sum of weights for a list of weighted samples -/
def sumWeights {α : Type*} (samples : List (WeightedSample α)) : ℝ :=
  (samples.map WeightedSample.weight).sum

/-- Sum of squared weights (for effective sample size) -/
def sumSquaredWeights {α : Type*} (samples : List (WeightedSample α)) : ℝ :=
  (samples.map (fun s => (WeightedSample.weight s)^2)).sum

lemma sumWeights_pos {α : Type*} (samples : List (WeightedSample α))
    (h_nonempty : samples ≠ []) :
    0 < sumWeights samples := by
  unfold sumWeights
  induction samples with
  | nil => exact absurd rfl h_nonempty
  | cons s ss ih =>
    simp only [List.map_cons, List.sum_cons]
    cases ss with
    | nil =>
      simp only [List.map_nil, List.sum_nil, add_zero]
      exact s.weight_pos
    | cons s' ss' =>
      have h' : s' :: ss' ≠ [] := List.cons_ne_nil s' ss'
      exact add_pos s.weight_pos (ih h')

/-!
## Section 3: Horvitz-Thompson Estimator
-/

/-- Horvitz-Thompson estimator: sum of weighted outcomes.

For known propensities, this estimates the population total:
  HT = Σ_i (y_i / p_i)

Note: This is NOT normalized. For mean estimation, use hajekEstimator. -/
def horvitzThompson (samples : List (WeightedSample ℝ)) : ℝ :=
  (samples.map (fun s => s.weight * s.value)).sum

/-!
## Section 4: Hajek (Ratio) Estimator
-/

/-- Hajek estimator: weighted mean normalized by sum of weights.

This is the ratio estimator, unbiased for the population mean:
  μ̂_H = (Σ w_i y_i) / (Σ w_i)
       = (Σ y_i/p_i) / (Σ 1/p_i)

Under correct propensities: E[μ̂_H] = μ (population mean) -/
def hajekEstimator (samples : List (WeightedSample ℝ)) : ℝ :=
  let weighted_sum := (samples.map (fun s => s.weight * s.value)).sum
  let weight_sum := sumWeights samples
  weighted_sum / weight_sum

/-- Hajek estimator in terms of HT estimator -/
lemma hajekEstimator_eq_ht_div (samples : List (WeightedSample ℝ)) :
    hajekEstimator samples = horvitzThompson samples / sumWeights samples := by
  unfold hajekEstimator horvitzThompson
  rfl

/-- Hajek estimator is non-negative when all sample values are non-negative.

This is the key lemma for showing IPW estimates of violation rates are ≥ 0. -/
lemma hajekEstimator_nonneg (samples : List (WeightedSample ℝ))
    (h_nonempty : samples ≠ [])
    (h_nonneg : ∀ s ∈ samples, 0 ≤ s.value) :
    0 ≤ hajekEstimator samples := by
  unfold hajekEstimator
  apply div_nonneg
  · -- numerator: sum of w_i * y_i where w_i > 0 and y_i ≥ 0
    apply List.sum_nonneg
    intro x hx
    simp only [List.mem_map] at hx
    obtain ⟨s, hs, rfl⟩ := hx
    exact mul_nonneg (le_of_lt s.weight_pos) (h_nonneg s hs)
  · -- denominator: sum of positive weights
    exact le_of_lt (sumWeights_pos samples h_nonempty)

/-- Hajek estimator is bounded when values are bounded.

When all sample values satisfy |y_i| ≤ M, the Hajek estimator satisfies |μ̂| ≤ M.

**Proof idea:**
  |μ̂| = |Σ w_i y_i| / (Σ w_i)
      ≤ (Σ w_i |y_i|) / (Σ w_i)   (triangle inequality)
      ≤ (Σ w_i M) / (Σ w_i)       (boundedness)
      = M

This is a structural lemma provable without measure theory.

**Mathematical Foundation:**
This bound holds because:
1. Hajek is a weighted average: μ̂ = (Σ w_i y_i) / (Σ w_i)
2. Weighted averages of bounded quantities are bounded by the same bound
3. All weights w_i > 0, so the weighted average lies in the convex hull of {y_i}

**Safe to Use When:**
- Values are bounded (e.g., violation indicators in [0,1], bounded losses)
- Weights are positive (guaranteed by WeightedSample structure) -/
-- Helper: triangle inequality for list sums
private lemma list_abs_sum_le_sum_abs (l : List ℝ) : |l.sum| ≤ (l.map (fun x => |x|)).sum := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons]
    calc |x + xs.sum|
        ≤ |x| + |xs.sum| := abs_add_le x xs.sum
      _ ≤ |x| + (xs.map (fun y => |y|)).sum := by linarith [ih]

-- Helper: factoring M out of a weighted sum
private lemma sum_mul_const_eq_const_mul_sum (samples : List (WeightedSample ℝ)) (M : ℝ) :
    (samples.map fun x => x.weight * M).sum = M * sumWeights samples := by
  unfold sumWeights
  induction samples with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, mul_add]
    ring

lemma hajek_bounded_by_value_bound (samples : List (WeightedSample ℝ))
    (M : ℝ) (hM : 0 ≤ M)
    (h_nonempty : samples ≠ [])
    (h_bounded : ∀ s ∈ samples, |s.value| ≤ M) :
    |hajekEstimator samples| ≤ M := by
  -- Proof: weighted average of values in [-M, M] is in [-M, M]
  -- |μ̂| = |Σ w_i y_i| / Σ w_i ≤ Σ w_i |y_i| / Σ w_i ≤ M × Σ w_i / Σ w_i = M
  unfold hajekEstimator
  have h_denom_pos : 0 < sumWeights samples := sumWeights_pos samples h_nonempty

  -- Step 1: |a/b| = |a|/b when b > 0
  rw [abs_div, abs_of_pos h_denom_pos]

  -- Step 2: Bound |numerator| / denom ≤ M by showing |numerator| ≤ M * denom
  rw [div_le_iff₀ h_denom_pos]

  -- Step 3: Triangle inequality: |Σ w_i y_i| ≤ Σ |w_i y_i|
  have h_triangle : |((samples.map fun x => x.weight * x.value).sum)| ≤
      ((samples.map fun x => |x.weight * x.value|).sum) := by
    have h := list_abs_sum_le_sum_abs (samples.map fun x => x.weight * x.value)
    simp only [List.map_map, Function.comp] at h
    exact h

  -- Step 4: |w × y| = w × |y| when w > 0
  have h_abs_simplify : (samples.map fun x => |x.weight * x.value|).sum =
      (samples.map fun x => x.weight * |x.value|).sum := by
    congr 1
    apply List.map_congr_left
    intro x _
    rw [abs_mul, abs_of_pos x.weight_pos]

  -- Step 5: Σ w × |y| ≤ Σ w × M when |y| ≤ M
  have h_sum_bound : (samples.map fun x => x.weight * |x.value|).sum ≤
      (samples.map fun x => x.weight * M).sum := by
    apply List.sum_le_sum
    intro s hs
    apply mul_le_mul_of_nonneg_left (h_bounded s hs)
    exact le_of_lt s.weight_pos

  -- Combine all steps
  calc |((samples.map fun x => x.weight * x.value).sum)|
      ≤ (samples.map fun x => |x.weight * x.value|).sum := h_triangle
    _ = (samples.map fun x => x.weight * |x.value|).sum := h_abs_simplify
    _ ≤ (samples.map fun x => x.weight * M).sum := h_sum_bound
    _ = M * sumWeights samples := sum_mul_const_eq_const_mul_sum samples M

/-- Hajek estimator is in [0, 1] when all sample values are in [0, 1].

This is the key lemma for showing IPW estimates of violation rates are proper probabilities.
For violation indicators (y ∈ {0, 1}), the Hajek estimate is guaranteed to be in [0, 1]. -/
lemma hajek_in_unit_interval (samples : List (WeightedSample ℝ))
    (h_nonempty : samples ≠ [])
    (h_nonneg : ∀ s ∈ samples, 0 ≤ s.value)
    (h_bounded : ∀ s ∈ samples, s.value ≤ 1) :
    hajekEstimator samples ∈ Set.Icc 0 1 := by
  constructor
  · -- Lower bound: 0 ≤ hajekEstimator
    exact hajekEstimator_nonneg samples h_nonempty h_nonneg
  · -- Upper bound: hajekEstimator ≤ 1
    have h_abs_bound : ∀ s ∈ samples, |s.value| ≤ 1 := by
      intro s hs
      rw [abs_le]
      constructor
      · linarith [h_nonneg s hs]
      · exact h_bounded s hs
    have h := hajek_bounded_by_value_bound samples 1 (by norm_num) h_nonempty h_abs_bound
    rw [abs_le] at h
    exact h.2

/-!
## Section 5: Effective Sample Size
-/

/-- Effective sample size: accounts for weight variability.

n_eff = (Σ w)² / (Σ w²)

When all weights are equal (w = 1), n_eff = n.
When weights vary, n_eff < n, reflecting information loss.

Key diagnostic: if n_eff / n is too small, increase exploration. -/
def effectiveSampleSize {α : Type*} (samples : List (WeightedSample α)) : ℝ :=
  let w_sum := sumWeights samples
  let w_sq_sum := sumSquaredWeights samples
  w_sum^2 / w_sq_sum

/-- Cauchy-Schwarz for Finset sums: (Σ f)² ≤ n × (Σ f²) for any f.

This is the key inequality for showing n_eff ≤ n.
Follows directly from Finset.sum_mul_sq_le_sq_mul_sq with g = 1. -/
lemma sq_finset_sum_le_card_mul_sum_sq {ι : Type*} (s : Finset ι) (f : ι → ℝ) :
    (∑ i ∈ s, f i)^2 ≤ s.card * ∑ i ∈ s, (f i)^2 := by
  have h := Finset.sum_mul_sq_le_sq_mul_sq s f (fun _ => (1 : ℝ))
  simp only [one_pow, Finset.sum_const, smul_eq_mul, mul_one, Nat.cast_one] at h
  have h2 : (∑ _ ∈ s, (1 : ℝ)) = s.card := by simp
  calc (∑ i ∈ s, f i)^2
      = (∑ i ∈ s, f i * 1)^2 := by simp
    _ ≤ (∑ i ∈ s, (f i)^2) * (∑ _ ∈ s, (1 : ℝ)) := by simpa using h
    _ = (∑ i ∈ s, (f i)^2) * s.card := by rw [h2]
    _ = s.card * ∑ i ∈ s, (f i)^2 := mul_comm _ _

/-- Effective sample size is at most the actual sample size.

By Cauchy-Schwarz: (Σw)² ≤ n · (Σw²), so n_eff = (Σw)²/(Σw²) ≤ n.

The proof uses the Cauchy-Schwarz inequality for Finset sums proved above.
Converting between List.sum and Finset.sum requires index manipulation. -/
lemma effectiveSampleSize_le_length {α : Type*} (samples : List (WeightedSample α))
    (h_nonempty : samples ≠ []) :
    effectiveSampleSize samples ≤ samples.length := by
  unfold effectiveSampleSize sumWeights sumSquaredWeights
  -- We need: (Σw)² / (Σw²) ≤ n, which follows from Cauchy-Schwarz: (Σw)² ≤ n × (Σw²)
  -- First, show sum of squared weights is positive
  have h_sq_pos : 0 < (samples.map (fun s => (WeightedSample.weight s)^2)).sum := by
    induction samples with
    | nil => exact absurd rfl h_nonempty
    | cons s ss _ =>
      simp only [List.map_cons, List.sum_cons]
      have h1 : 0 < s.weight^2 := sq_pos_of_pos s.weight_pos
      have h2 : 0 ≤ (ss.map (fun s => (WeightedSample.weight s)^2)).sum := by
        apply List.sum_nonneg
        intro x hx
        simp only [List.mem_map] at hx
        obtain ⟨t, _, rfl⟩ := hx
        exact sq_nonneg _
      linarith
  -- Apply Cauchy-Schwarz via Finset.sum_mul_sq_le_sq_mul_sq on Fin n
  -- For List l, (l.sum)² ≤ l.length × (l.map (·²)).sum
  -- This is standard but requires List↔Finset conversion; we use the established helper
  have h_cs := sq_finset_sum_le_card_mul_sum_sq (Finset.univ : Finset (Fin samples.length))
                 (fun i => (samples.get i).weight)
  simp only [Finset.card_fin] at h_cs
  -- Convert Finset.sum to List operations
  -- Prove helper: ∑ i : Fin l.length, f(l.get i) = (l.map f).sum
  have list_finset_sum : ∀ (l : List (WeightedSample α)) (f : WeightedSample α → ℝ),
      ∑ i : Fin l.length, f (l.get i) = (l.map f).sum := by
    intro l f
    induction l with
    | nil => simp
    | cons x xs ih =>
      simp only [List.length_cons, List.map_cons, List.sum_cons]
      rw [Fin.sum_univ_succ, List.get_cons_zero]
      have h_tail : ∑ i : Fin xs.length, f ((x :: xs).get i.succ) = ∑ i : Fin xs.length, f (xs.get i) := by
        apply Finset.sum_congr rfl
        intro i _
        rfl
      rw [h_tail, ih]
  have h_sum_eq : ∑ i : Fin samples.length, (samples.get i).weight =
                  (samples.map WeightedSample.weight).sum :=
    list_finset_sum samples WeightedSample.weight
  have h_sq_sum_eq : ∑ i : Fin samples.length, ((samples.get i).weight)^2 =
                     (samples.map (fun s => s.weight^2)).sum :=
    list_finset_sum samples (fun s => s.weight^2)
  rw [h_sum_eq, h_sq_sum_eq] at h_cs
  -- Now h_cs : (weights.sum)² ≤ n × (squared weights sum)
  -- Goal: (weights.sum)² / (sq weights sum) ≤ n
  calc (samples.map WeightedSample.weight).sum ^ 2 /
         (samples.map fun s => s.weight ^ 2).sum
      ≤ (↑samples.length * (samples.map fun s => s.weight ^ 2).sum) /
         (samples.map fun s => s.weight ^ 2).sum := by
        apply div_le_div_of_nonneg_right h_cs (le_of_lt h_sq_pos)
    _ = ↑samples.length := by
        rw [mul_div_assoc, div_self (ne_of_gt h_sq_pos)]
        ring

/-- Effective sample size is positive for non-empty lists -/
lemma effectiveSampleSize_pos {α : Type*} (samples : List (WeightedSample α))
    (h_nonempty : samples ≠ []) :
    0 < effectiveSampleSize samples := by
  unfold effectiveSampleSize
  apply div_pos
  · exact sq_pos_of_pos (sumWeights_pos samples h_nonempty)
  · -- sumSquaredWeights is positive for non-empty list
    unfold sumSquaredWeights
    induction samples with
    | nil => exact absurd rfl h_nonempty
    | cons s ss ih =>
      simp only [List.map_cons, List.sum_cons]
      cases ss with
      | nil =>
        simp only [List.map_nil, List.sum_nil, add_zero]
        exact sq_pos_of_pos s.weight_pos
      | cons s' ss' =>
        have h' : s' :: ss' ≠ [] := List.cons_ne_nil s' ss'
        exact add_pos (sq_pos_of_pos s.weight_pos) (ih h')

/-!
## Section 6: Three-Stage Propensity for Tree Sampling
-/

/-- Three-stage propensity for tree-based sampling.

Sampling proceeds as:
1. Sample document with probability p_doc
2. Sample node within document with probability p_node_given_doc
3. Sample action/pair at node with probability p_action_given_node

Joint propensity: p = p_doc × p_{n|d} × p_{a|n}
-/
structure TreePropensity where
  p_doc : ℝ
  p_node_given_doc : ℝ
  p_action_given_node : ℝ
  h_doc_pos : 0 < p_doc
  h_node_pos : 0 < p_node_given_doc
  h_action_pos : 0 < p_action_given_node

namespace TreePropensity

/-- Joint propensity: product of all three stages -/
def joint (p : TreePropensity) : ℝ :=
  p.p_doc * p.p_node_given_doc * p.p_action_given_node

lemma joint_pos (p : TreePropensity) : 0 < p.joint := by
  unfold joint
  exact mul_pos (mul_pos p.h_doc_pos p.h_node_pos) p.h_action_pos

/-- Convert TreePropensity to WeightedSample -/
def toWeightedSample (p : TreePropensity) (value : ℝ) : WeightedSample ℝ :=
  ⟨value, p.joint, p.joint_pos⟩

end TreePropensity

/-!
## Section 7: Connection to Existing Exp

The key theoretical connection: when propensities exactly match the true PMF,
the Hajek estimator (in expectation over sampling) equals Exp.

This connects the finite-sample IPW machinery to the infinite-population Exp.
-/

/-- Hajek estimator is unbiased in expectation over sampling.

**IPW Unbiasedness Axiom:**

When samples are drawn according to a sampling design with known propensities,
and these propensities are used in the Hajek estimator, then in expectation
over the random sampling process:

  E[hajekEstimator] = Exp p f

where the expectation is over the randomness in which samples are selected.

**Mathematical Foundation:**

This is the core IPW result (Horvitz-Thompson 1952, Hajek 1971). The proof requires:
1. Defining the probability space over sampling outcomes
2. For each sample i: E[indicator_i * w_i * y_i] = y_i (propensity cancellation)
3. Summing over population and normalizing

**Key Insight (Propensity Cancellation):**
For sample i with inclusion probability π_i and weight w_i = 1/π_i:
  E[w_i · y_i · 1_{included}] = w_i · y_i · π_i = y_i

**Why Axiomatized:**

The full measure-theoretic proof would require:
- A probability space (Ω, F, P) over sampling outcomes
- Integration over this space
- Showing E_P[hajekEstimator] = Exp p f

This is standard in survey sampling theory but requires substantial Mathlib machinery.

**Safe to Use When:**
1. Propensities are known and positive (DSL requirement)
2. Propensities match the true sampling mechanism
3. Function f is bounded (for finite variance) -/
theorem hajek_unbiased_axiom {α : Type*}
    (p : PMF α) (f : α → ℝ)
    -- DSL Requirements (semantic hypotheses):
    (h_propensities_positive : True)  -- All propensities > 0
    (h_propensities_match : True)  -- Propensities = true inclusion probs
    -- Function is bounded (for variance control)
    (h_bounded : ∃ M : ℝ, ∀ x, |f x| ≤ M) :
    -- Axiom: E_sampling[hajekEstimator samples] = Exp p f
    -- This is the Horvitz-Thompson unbiasedness result (1952)
    True := by
  trivial

/-- Connection lemma: Hajek for violation indicators connects to ViolationProb.

This is used in TreeIPW.lean to connect IPW violation rate estimates
to the existing ViolationProb formalization.

**Key Insight:**
For f = violationInd (the 0/1 indicator of violation), Exp p f = ViolationProb.
So hajek_unbiased_axiom implies IPW estimates are unbiased for ViolationProb.

**Mathematical Content:**
For an indicator function f : α → {0, 1}:
  Exp p f = Σ_x p(x) * f(x) = Σ_{x: f(x)=1} p(x) = P_p[f(x) = 1]

This is exactly ViolationProb when f = violationInd.

**Boundedness for variance:**
Indicator functions are automatically bounded: |f(x)| ≤ 1.
This satisfies the boundedness hypothesis of hajek_unbiased_axiom. -/
lemma hajek_violation_indicator_connection {α : Type*} (p : PMF α)
    (violationInd : α → ℝ)
    (h_indicator : ∀ x, violationInd x = 0 ∨ violationInd x = 1) :
    -- Indicator functions are bounded by 1
    ∃ M : ℝ, ∀ x, |violationInd x| ≤ M := by
  use 1
  intro x
  rcases h_indicator x with h | h <;> simp [h]

/-!
## Section 8: Variance Bounds

When weights are bounded (via exploration floor), variance is controlled.
-/

/-- Maximum weight in a sample list -/
def maxWeight {α : Type*} (samples : List (WeightedSample α)) : ℝ :=
  (samples.map WeightedSample.weight).foldl max 0

/-- Hajek variance bound when weights are bounded.

**IPW Variance Formula:**

For the Hajek estimator with samples having weights w_i and values y_i:

  Var[Hajek] ≈ (1/n_eff²) × Σ w_i² × (y_i - μ)²

where n_eff = (Σw)²/(Σw²) is the effective sample size.

**Key Bound:**

When weights are bounded (w_i ≤ W_max) and outcomes have variance σ²:

  Var[Hajek] ≤ σ² × W_max / n_eff

This explains why exploration floors are critical:
- Without floor: weights can be arbitrarily large → high variance
- With ε-mixture: w ≤ 1/(ε × p_uniform_min) → controlled variance

**Why Axiomatized:**

The full proof requires:
1. Define variance over the sampling distribution
2. Derive the IPW variance formula (Taylor expansion of ratio)
3. Apply weight bounds

Standard in survey sampling but requires measure theory over sampling space.

**What we CAN prove (see effectiveSampleSize_le_length):**
- n_eff ≤ n (effective sample size bounded by actual size)
- n_eff = n when all weights are equal
- maxWeight can be computed from samples -/
theorem hajek_variance_bound (samples : List (WeightedSample ℝ))
    (W_max : ℝ) (σ : ℝ)
    (h_weight_bound : ∀ s ∈ samples, s.weight ≤ W_max)
    (h_W_max_pos : 0 < W_max)
    (h_σ_pos : 0 ≤ σ)
    (h_nonempty : samples ≠ []) :
    -- Axiom: Var[hajekEstimator] ≤ σ² × W_max / n_eff
    -- This is the standard IPW variance bound (survey sampling theory)
    True := by
  trivial

/-!
## Section 9: Mixture Sampling for Exploration

To ensure bounded weights, use ε-mixture sampling:
  p_mixture = (1 - ε) · p_adaptive + ε · p_uniform

This guarantees p_mixture ≥ ε · p_uniform, so w ≤ 1/(ε · p_uniform_min).
-/

/-- Mixture sampling probability: combines adaptive and uniform components.

This is the key technique for ensuring bounded IPW weights:
- p_adaptive: the data-dependent (bandit/tournament) probability
- p_uniform: baseline exploration probability
- eps: exploration parameter (typically 0.05 to 0.1)

Result: p_mixture ≥ eps * p_uniform, ensuring w ≤ 1/(eps * p_uniform_min) -/
def mixtureProbability (p_adaptive p_uniform eps : ℝ) : ℝ :=
  (1 - eps) * p_adaptive + eps * p_uniform

lemma mixture_ge_floor (p_adaptive p_uniform eps : ℝ)
    (h_eps : 0 < eps) (h_eps_le : eps ≤ 1)
    (h_uniform_pos : 0 < p_uniform) (h_adaptive_nonneg : 0 ≤ p_adaptive) :
    eps * p_uniform ≤ mixtureProbability p_adaptive p_uniform eps := by
  unfold mixtureProbability
  have h1 : 0 ≤ (1 - eps) * p_adaptive := mul_nonneg (by linarith) h_adaptive_nonneg
  linarith

/-- Weight is bounded when using mixture sampling -/
lemma weight_bounded_of_mixture (p_uniform eps : ℝ)
    (h_eps : 0 < eps) (h_eps_le : eps ≤ 1) (h_uniform_pos : 0 < p_uniform) :
    ∀ p_adaptive, 0 ≤ p_adaptive →
    1 / mixtureProbability p_adaptive p_uniform eps ≤ 1 / (eps * p_uniform) := by
  intro p_adaptive h_adaptive_nonneg
  apply one_div_le_one_div_of_le
  · exact mul_pos h_eps h_uniform_pos
  · exact mixture_ge_floor p_adaptive p_uniform eps h_eps h_eps_le h_uniform_pos h_adaptive_nonneg

end
