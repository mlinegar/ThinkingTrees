import FormalProofs.DSL.IPWTheory

/-!
# FormalProofs/ClusteredVariance.lean

## Clustered Standard Errors for Correlated Samples

This file formalizes cluster-robust variance estimation for IPW estimators.
In tree-based preference learning, nodes within a document are correlated,
so documents form natural clusters.

See RLHF_DSL_BANDIT_NOTES.md Section 6 for the conceptual model.

### Key Results

- `Cluster`: A cluster of correlated samples (e.g., nodes from one document)
- `clusterResidualSum`: G_d = Σ_{i∈d} w_i(y_i - μ̂)
- `clusteredVariance`: V̂ = (1/n_eff²) · (1/(D-1)) · Σ_d G_d²
- `clusteredSE`: SE = √V̂

### Why Clustered SEs?

Standard SEs assume independent samples. When samples are correlated within
clusters (e.g., nodes within a document), ignoring this correlation leads to:
- Underestimated variance
- Overconfident confidence intervals
- Invalid hypothesis tests

Clustered SEs account for intra-cluster correlation by:
1. Computing residuals at the sample level
2. Summing within clusters
3. Computing variance across cluster sums
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

/-!
## Section 1: Cluster Structure
-/

/-- A cluster of correlated samples.

In tree-based preference learning, each document forms a cluster
containing all sampled nodes from that document.

Samples within a cluster are assumed correlated; clusters are assumed
independent of each other. -/
structure Cluster (α : Type*) where
  cluster_id : String
  samples : List (WeightedSample α)

namespace Cluster

variable {α : Type*}

/-- Number of samples in a cluster -/
def size (c : Cluster α) : ℕ := c.samples.length

/-- Sum of weights within a cluster -/
def sumWeights (c : Cluster α) : ℝ :=
  (c.samples.map WeightedSample.weight).sum

/-- Sum of squared weights within a cluster -/
def sumSquaredWeights (c : Cluster α) : ℝ :=
  (c.samples.map (fun s => (WeightedSample.weight s)^2)).sum

end Cluster

/-!
## Section 2: Residuals and Cluster Sums
-/

/-- Weighted residual for a single sample: w_i * (y_i - μ̂) -/
def weightedResidual (s : WeightedSample ℝ) (mu_hat : ℝ) : ℝ :=
  s.weight * (s.value - mu_hat)

/-- Cluster-level residual sum: G_d = Σ_{i∈d} w_i(y_i - μ̂)

This is the core quantity for cluster-robust variance estimation.
By summing within clusters, we capture the correlation structure. -/
def clusterResidualSum (c : Cluster ℝ) (mu_hat : ℝ) : ℝ :=
  (c.samples.map (fun s => weightedResidual s mu_hat)).sum

/-- Squared cluster residual sum: G_d² -/
def clusterResidualSumSquared (c : Cluster ℝ) (mu_hat : ℝ) : ℝ :=
  (clusterResidualSum c mu_hat)^2

/-!
## Section 3: Clustered Variance Estimator
-/

/-- Sum of weights across all clusters -/
def totalSumWeights (clusters : List (Cluster ℝ)) : ℝ :=
  (clusters.map Cluster.sumWeights).sum

/-- Number of clusters -/
def numClusters (clusters : List (Cluster ℝ)) : ℕ :=
  clusters.length

/-- Clustered (sandwich) variance estimator.

This is the cluster-robust variance formula from the DSL notes:
  V̂ = (1 / (Σw)²) · (1 / (D-1)) · Σ_d G_d²

where:
- Σw = total sum of weights across all samples
- D = number of clusters (documents)
- G_d = cluster-level residual sum

This estimator is consistent under cluster independence. -/
def clusteredVariance (clusters : List (Cluster ℝ)) (mu_hat : ℝ) : ℝ :=
  let D := clusters.length
  let G_d_sq := clusters.map (fun c => clusterResidualSumSquared c mu_hat)
  let w_sum := totalSumWeights clusters
  if D ≤ 1 then 0
  else (1 / w_sum^2) * (1 / (D - 1 : ℝ)) * G_d_sq.sum

/-- Clustered standard error: SE = √V̂ -/
def clusteredSE (clusters : List (Cluster ℝ)) (mu_hat : ℝ) : ℝ :=
  Real.sqrt (clusteredVariance clusters mu_hat)

/-!
## Section 4: Basic Properties
-/

lemma clusteredVariance_nonneg (clusters : List (Cluster ℝ)) (mu_hat : ℝ) :
    0 ≤ clusteredVariance clusters mu_hat := by
  unfold clusteredVariance
  by_cases h : clusters.length ≤ 1
  · simp only [if_pos h, le_refl]
  · simp only [if_neg h]
    apply mul_nonneg
    apply mul_nonneg
    · apply one_div_nonneg.mpr
      exact sq_nonneg _
    · apply one_div_nonneg.mpr
      push_neg at h
      have : (1 : ℝ) ≤ clusters.length := by
        simp only [Nat.one_le_cast]
        omega
      linarith
    · -- Sum of squares is nonneg
      apply List.sum_nonneg
      intro x hx
      simp only [List.mem_map] at hx
      obtain ⟨c, _, rfl⟩ := hx
      exact sq_nonneg _

lemma clusteredSE_nonneg (clusters : List (Cluster ℝ)) (mu_hat : ℝ) :
    0 ≤ clusteredSE clusters mu_hat := by
  unfold clusteredSE
  exact Real.sqrt_nonneg _

/-!
## Section 5: Effective Sample Size for Clustered Data
-/

/-- Flatten clusters to get all samples -/
def allSamples (clusters : List (Cluster ℝ)) : List (WeightedSample ℝ) :=
  clusters.flatMap Cluster.samples

/-- Effective sample size accounting for clustering.

When samples are clustered, the effective sample size is reduced
due to within-cluster correlation. -/
def clusteredEffectiveSampleSize (clusters : List (Cluster ℝ)) : ℝ :=
  effectiveSampleSize (allSamples clusters)

/-- Design effect: ratio of clustered to simple variance.

DEFF = 1 + (m̄ - 1) · ρ

where:
- m̄ = average cluster size
- ρ = intra-cluster correlation coefficient (ICC)

When DEFF > 1, clustering increases variance. -/
def designEffect (icc : ℝ) (avg_cluster_size : ℝ) : ℝ :=
  1 + (avg_cluster_size - 1) * icc

/-- Variance inflation due to clustering.

Var_clustered = Var_iid × DEFF

This quantifies how much larger the variance is compared to
what it would be if samples were independent. -/
theorem variance_inflation_icc (icc : ℝ) (avg_cluster_size : ℝ)
    (h_icc : 0 ≤ icc) (h_icc_le : icc ≤ 1) (h_size : 1 ≤ avg_cluster_size) :
    1 ≤ designEffect icc avg_cluster_size := by
  unfold designEffect
  have h1 : 0 ≤ avg_cluster_size - 1 := by linarith
  have h2 : 0 ≤ (avg_cluster_size - 1) * icc := mul_nonneg h1 h_icc
  linarith

/-!
## Section 6: Validity Theorems
-/

/-- Clustered variance estimate is always non-negative.

If clusters are independent, the clustered variance estimator
is a consistent estimator of the true variance.

**Theoretical Foundation (Liang-Zeger 1986):**

The sandwich (cluster-robust) variance estimator is:
  V̂ = (1 / (Σw)²) × (1 / (D-1)) × Σ_d G_d²

where G_d = Σ_{i∈d} w_i(y_i - μ̂) is the cluster-level residual sum.

**Key Properties:**
1. Consistent: V̂ →ᵖ Var(μ̂) as D → ∞
2. Conservative: E[V̂] ≥ Var(μ̂) for finite D (with HC3 correction)
3. Robust: Valid even if samples within clusters are arbitrarily correlated

**Why It Works:**
- Within-cluster correlation is absorbed by summing residuals within clusters
- The variance is computed across cluster sums, which are independent
- This is the "generalized estimating equations" (GEE) approach

This file proves the structural non-negativity property of the estimator.
Asymptotic consistency under cluster independence is currently documented but
not yet fully formalized in this theorem.

**Reference:** Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis
using generalized linear models. Biometrika, 73(1), 13-22. -/
theorem clustered_se_valid (clusters : List (Cluster ℝ)) (mu_hat : ℝ) :
    -- This theorem formalizes the always-valid structural property.
    0 ≤ clusteredVariance clusters mu_hat := by
  exact clusteredVariance_nonneg clusters mu_hat

/-- Finite-sample correction for small number of clusters.

When D is small (< 50), apply (D / (D-1)) correction for
less biased variance estimation. -/
def finiteSampleCorrection (D : ℕ) : ℝ :=
  if D ≤ 1 then 1 else (D : ℝ) / (D - 1)

/-!
## Section 7: Cluster-Robust Confidence Intervals
-/

/-- 95% confidence interval half-width using clustered SE.

CI = μ̂ ± z_{0.975} × SE ≈ μ̂ ± 1.96 × SE

For small D (< 30), use t-distribution with D-1 degrees of freedom. -/
def confidenceHalfWidth (se : ℝ) (z : ℝ := 1.96) : ℝ :=
  z * se

/-- Confidence interval using clustered SE -/
def confidenceInterval (mu_hat : ℝ) (se : ℝ) (z : ℝ := 1.96) : ℝ × ℝ :=
  let hw := confidenceHalfWidth se z
  (mu_hat - hw, mu_hat + hw)

/-- Clustered confidence interval for Hajek estimator -/
def clusteredConfidenceInterval (clusters : List (Cluster ℝ)) (z : ℝ := 1.96) : ℝ × ℝ :=
  let all := allSamples clusters
  let mu_hat := hajekEstimator all
  let se := clusteredSE clusters mu_hat
  confidenceInterval mu_hat se z

lemma confidenceHalfWidth_nonneg {se z : ℝ}
    (hse : 0 ≤ se) (hz : 0 ≤ z) :
    0 ≤ confidenceHalfWidth se z := by
  unfold confidenceHalfWidth
  exact mul_nonneg hz hse

lemma confidenceInterval_left_le_right
    (mu_hat se z : ℝ) (hse : 0 ≤ se) (hz : 0 ≤ z) :
    (confidenceInterval mu_hat se z).1 ≤ (confidenceInterval mu_hat se z).2 := by
  have hhw : 0 ≤ confidenceHalfWidth se z :=
    confidenceHalfWidth_nonneg hse hz
  unfold confidenceInterval
  linarith

lemma mem_confidenceInterval_center
    (mu_hat se z : ℝ) (hse : 0 ≤ se) (hz : 0 ≤ z) :
    mu_hat ∈ Set.Icc (confidenceInterval mu_hat se z).1
      (confidenceInterval mu_hat se z).2 := by
  have hhw : 0 ≤ confidenceHalfWidth se z :=
    confidenceHalfWidth_nonneg hse hz
  unfold confidenceInterval
  constructor <;> linarith

lemma mem_confidenceInterval_of_abs_sub_le
    (theta mu_hat se z : ℝ)
    (h : |theta - mu_hat| ≤ z * se) :
    theta ∈ Set.Icc (confidenceInterval mu_hat se z).1
      (confidenceInterval mu_hat se z).2 := by
  have h' : -(z * se) ≤ theta - mu_hat ∧ theta - mu_hat ≤ z * se := by
    exact abs_le.mp h
  unfold confidenceInterval confidenceHalfWidth
  constructor <;> linarith

lemma mem_confidenceInterval_iff_abs_sub_le
    (theta mu_hat se z : ℝ)
    (hradius : 0 ≤ z * se) :
    theta ∈ Set.Icc (confidenceInterval mu_hat se z).1
      (confidenceInterval mu_hat se z).2 ↔
      |theta - mu_hat| ≤ z * se := by
  constructor
  · intro hmem
    rcases hmem with ⟨hlo, hhi⟩
    unfold confidenceInterval confidenceHalfWidth at hlo hhi
    refine abs_le.mpr ?_
    constructor <;> linarith
  · intro habs
    exact mem_confidenceInterval_of_abs_sub_le theta mu_hat se z habs

theorem confidenceInterval_coverage_of_error_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω)
    (theta z : ℝ)
    (mu_hat se : Ω → ℝ)
    (q : ENNReal)
    (h_event : q ≤ μ {ω | |theta - mu_hat ω| ≤ z * se ω}) :
    q ≤ μ {ω | theta ∈ Set.Icc
      (confidenceInterval (mu_hat ω) (se ω) z).1
      (confidenceInterval (mu_hat ω) (se ω) z).2} := by
  have h_subset :
      {ω | |theta - mu_hat ω| ≤ z * se ω} ⊆
        {ω | theta ∈ Set.Icc
          (confidenceInterval (mu_hat ω) (se ω) z).1
          (confidenceInterval (mu_hat ω) (se ω) z).2} := by
    intro ω hω
    exact mem_confidenceInterval_of_abs_sub_le theta (mu_hat ω) (se ω) z hω
  exact le_trans h_event (MeasureTheory.measure_mono h_subset)

lemma mem_clusteredConfidenceInterval_of_abs_sub_le
    (clusters : List (Cluster ℝ)) (theta z : ℝ)
    (h : |theta - hajekEstimator (allSamples clusters)| ≤
      z * clusteredSE clusters (hajekEstimator (allSamples clusters))) :
    theta ∈ Set.Icc (clusteredConfidenceInterval clusters z).1
      (clusteredConfidenceInterval clusters z).2 := by
  exact mem_confidenceInterval_of_abs_sub_le theta
    (hajekEstimator (allSamples clusters))
    (clusteredSE clusters (hajekEstimator (allSamples clusters))) z h

theorem clusteredConfidenceInterval_coverage_of_error_event
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω)
    (clusters : Ω → List (Cluster ℝ))
    (theta z : ℝ)
    (q : ENNReal)
    (h_event : q ≤ μ {ω |
      |theta - hajekEstimator (allSamples (clusters ω))| ≤
        z * clusteredSE (clusters ω) (hajekEstimator (allSamples (clusters ω)))}) :
    q ≤ μ {ω | theta ∈ Set.Icc
      (clusteredConfidenceInterval (clusters ω) z).1
      (clusteredConfidenceInterval (clusters ω) z).2} := by
  simpa [clusteredConfidenceInterval] using
    (confidenceInterval_coverage_of_error_event
      (μ := μ)
      (theta := theta)
      (z := z)
      (mu_hat := fun ω => hajekEstimator (allSamples (clusters ω)))
      (se := fun ω => clusteredSE (clusters ω) (hajekEstimator (allSamples (clusters ω))))
      (q := q)
      h_event)

/-!
## Section 8: Diagnostics
-/

/-- Check if there are enough clusters for reliable inference.

Rule of thumb: need at least 30-50 clusters for normal approximation.
With fewer clusters, use t-distribution or bootstrap. -/
def hasEnoughClusters (clusters : List (Cluster ℝ)) (threshold : ℕ := 30) : Bool :=
  threshold ≤ clusters.length

/-- Check if effective sample size is adequate.

Rule of thumb: n_eff / n should be > 0.5.
If lower, weights are too variable; increase exploration. -/
def hasAdequateNeff (clusters : List (Cluster ℝ)) (threshold : ℝ := 0.5) : Bool :=
  let all := allSamples clusters
  let n := all.length
  let n_eff := effectiveSampleSize all
  threshold * n ≤ n_eff

end
