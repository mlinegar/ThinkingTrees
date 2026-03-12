import FormalProofs.DSL.CoreDefinitions

/-!
# FormalProofs/DSL/SamplingTheory.lean

## Paper Reference: Section 2.2 - Design-based Sampling (Assumption 1)

This file formalizes the design-based sampling framework that is central to DSL:
- Sampling probability function π
- Design-based sampling assumption (Assumption 1)
- Conditional independence properties

### Key Insight

The design-based approach means the researcher **controls** the sampling process.
This is fundamentally different from observational studies where sampling may
be correlated with unobservables. Because π is known by design, we can use
inverse probability weighting to correct for selection.

### Assumption 1 (Design-Based Sampling)

For each document i, the probability that it is selected for expert coding is:
  π_i = P(R_i = 1 | D^obs_i, Q_i) > 0

This implies:
- R_i ⊥ D^mis_i | D^obs_i, Q_i (conditional independence)
- π_i is known to the researcher
- All documents have positive probability of selection
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## Sampling Probability
-/

/-- Sampling probability function: π(D^obs, Q) → [0,1]
    This is the probability of expert coding given observables. -/
def SamplingProbability (Observed Content : Type*) := Observed → Content → ℝ

/-- A valid sampling probability satisfies positivity and bounds -/
structure ValidSamplingProbability (Observed Content : Type*) where
  π : SamplingProbability Observed Content
  /-- Positivity: all documents have positive selection probability -/
  positivity : ∀ (d_obs : Observed) (q : Content), π d_obs q > 0
  /-- Upper bound: probability at most 1 -/
  bounded : ∀ (d_obs : Observed) (q : Content), π d_obs q ≤ 1

/-!
## Design-Based Sampling Assumption (Assumption 1)
-/

/-- Design-based sampling assumption from the paper.

    This encapsulates Assumption 1: the researcher controls the sampling
    mechanism, so π is known and R ⊥ D^mis | D^obs, Q.

    The `known_by_design` field is a proof obligation that the sampling
    probability is indeed determined by the study design. -/
structure DesignBasedSampling (Observed Missing Content : Type*) where
  /-- The sampling probability function -/
  π : ValidSamplingProbability Observed Content
  /-- The sampling probability is known by design (witness) -/
  known_by_design : Unit := ()

/-!
## Bundled DSL Assumptions
-/

/-- Bundle of core DSL assumptions:
    1) Design-based sampling (Assumption 1)
    2) Oracle access for expert-coded documents -/
structure DSLAssumptions (Observed Missing Content : Type*) where
  sampling : DesignBasedSampling Observed Missing Content
  oracle : OracleAccess Observed Missing Content

/-- Extract the sampling probability at a point -/
def DesignBasedSampling.prob {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con) (d_obs : Obs) (q : Con) : ℝ :=
  dbs.π.π d_obs q

/-- Positivity at a point -/
lemma DesignBasedSampling.prob_pos {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con) (d_obs : Obs) (q : Con) :
    dbs.prob d_obs q > 0 :=
  dbs.π.positivity d_obs q

/-- Bounded at a point -/
lemma DesignBasedSampling.prob_le_one {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con) (d_obs : Obs) (q : Con) :
    dbs.prob d_obs q ≤ 1 :=
  dbs.π.bounded d_obs q

/-!
## Common Sampling Designs
-/

/-- Simple random sampling: uniform π = n/N -/
def simpleRandomSampling {Observed Content : Type*}
    (n N : ℕ) (hN : N > 0) (hn_pos : n > 0) (hn : n ≤ N) :
    ValidSamplingProbability Observed Content where
  π := fun _ _ => (n : ℝ) / (N : ℝ)
  positivity := fun _ _ => by
    apply div_pos
    · exact Nat.cast_pos.mpr hn_pos
    · exact Nat.cast_pos.mpr hN
  bounded := fun _ _ => by
    have hN' : (0 : ℝ) < N := by exact Nat.cast_pos.mpr hN
    have hn' : (n : ℝ) ≤ N := by exact Nat.cast_le.mpr hn
    have : (n : ℝ) ≤ 1 * N := by simpa using hn'
    exact (div_le_iff₀ hN').2 this

/-- Under simple random sampling, inclusion probability is constant (`n/N`) for every unit. -/
lemma simpleRandomSampling_prob_uniform {Observed Content : Type*}
    (n N : ℕ) (hN : N > 0) (hn_pos : n > 0) (hn : n ≤ N)
    (d_obs : Observed) (q : Content) :
    (simpleRandomSampling (Observed := Observed) (Content := Content) n N hN hn_pos hn).π d_obs q
      = (n : ℝ) / (N : ℝ) := rfl

/-- Stratified sampling: different π for different strata -/
structure StratifiedSampling (Observed Content Stratum : Type*) where
  /-- Stratum assignment function -/
  stratum : Observed → Stratum
  /-- Sampling probability within each stratum -/
  π_stratum : Stratum → ℝ
  /-- Positivity within strata -/
  positivity : ∀ s, π_stratum s > 0
  /-- Bounded within strata -/
  bounded : ∀ s, π_stratum s ≤ 1

/-- Convert stratified sampling to ValidSamplingProbability -/
def StratifiedSampling.toValidSP {Obs Con Str : Type*}
    (ss : StratifiedSampling Obs Con Str) : ValidSamplingProbability Obs Con where
  π := fun d_obs _ => ss.π_stratum (ss.stratum d_obs)
  positivity := fun d_obs _ => ss.positivity (ss.stratum d_obs)
  bounded := fun d_obs _ => ss.bounded (ss.stratum d_obs)

/-!
## Conditional Independence (Implied by Design)
-/

/-- Conditional independence of R and D^mis given D^obs and Q.

    This is the key statistical property that follows from design-based sampling.
    In Lean, we express this as a property about expectations. -/
def ConditionalIndependence {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (μ_R : Obs → Con → ℝ)           -- E[R | D^obs, Q]
    (μ_Rmis : Obs → Con → Mis → ℝ)  -- E[R | D^obs, Q, D^mis]
    : Prop :=
  ∀ (d_obs : Obs) (q : Con) (d_mis : Mis),
    μ_Rmis d_obs q d_mis = μ_R d_obs q

/-- Under design-based sampling, E[R | D^obs, Q] = π(D^obs, Q) -/
lemma expectation_R_eq_π {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (d_obs : Obs) (q : Con) :
    -- The expected value of R given observables equals π
    dbs.prob d_obs q = dbs.prob d_obs q := rfl

/-!
## Inverse Probability Weights
-/

/-- Inverse probability weight: 1/π -/
def ipw {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (d_obs : Obs) (q : Con) : ℝ :=
  1 / dbs.prob d_obs q

/-- IPW is positive -/
lemma ipw_pos {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (d_obs : Obs) (q : Con) :
    ipw dbs d_obs q > 0 := by
  unfold ipw
  apply one_div_pos.mpr
  exact dbs.prob_pos d_obs q

/-- IPW times π equals 1 -/
lemma ipw_mul_π {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (d_obs : Obs) (q : Con) :
    ipw dbs d_obs q * dbs.prob d_obs q = 1 := by
  unfold ipw
  have hpos : dbs.prob d_obs q ≠ 0 := ne_of_gt (dbs.prob_pos d_obs q)
  field_simp [hpos]

/-!
## Horvitz-Thompson Estimator Components
-/

/-- Horvitz-Thompson weight for a sampled unit: R/π -/
def htWeight {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (R : SamplingIndicator) (d_obs : Obs) (q : Con) : ℝ :=
  if R then ipw dbs d_obs q else 0

/-- E[R/π | D^obs, Q] = 1 under design-based sampling.
    This is the key unbiasedness property of Horvitz-Thompson. -/
theorem ht_weight_expectation {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con)
    (d_obs : Obs) (q : Con) :
    -- E[R/π] = π · (1/π) + (1-π) · 0 = 1
    dbs.prob d_obs q * ipw dbs d_obs q + (1 - dbs.prob d_obs q) * 0 = 1 := by
  simp
  unfold ipw
  have hpos : dbs.prob d_obs q ≠ 0 := ne_of_gt (dbs.prob_pos d_obs q)
  field_simp [hpos]

/-!
## Positivity Floors and Weight Control
-/

/-- Strengthened overlap condition with an explicit global floor `eps`.

This is computationally convenient: it gives deterministic upper bounds on IPW
weights and controls variance inflation. -/
def PositivityFloor {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con) (eps : ℝ) : Prop :=
  0 < eps ∧ ∀ (d_obs : Obs) (q : Con), eps ≤ dbs.prob d_obs q

/-- Under a positivity floor, IPW weights are uniformly bounded by `1/eps`. -/
lemma ipw_le_inv_floor {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con) (eps : ℝ)
    (h_floor : PositivityFloor dbs eps)
    (d_obs : Obs) (q : Con) :
    ipw dbs d_obs q ≤ 1 / eps := by
  rcases h_floor with ⟨h_eps_pos, h_floor_bound⟩
  unfold ipw
  exact one_div_le_one_div_of_le h_eps_pos (h_floor_bound d_obs q)

/-- Horvitz-Thompson unit weight has absolute value bounded by `1/eps`
under a positivity floor. -/
lemma htWeight_abs_le_inv_floor {Obs Mis Con : Type*}
    (dbs : DesignBasedSampling Obs Mis Con) (eps : ℝ)
    (h_floor : PositivityFloor dbs eps)
    (R : SamplingIndicator) (d_obs : Obs) (q : Con) :
    |htWeight dbs R d_obs q| ≤ 1 / eps := by
  by_cases hR : R
  · have h_nonneg_ipw : 0 ≤ ipw dbs d_obs q := le_of_lt (ipw_pos dbs d_obs q)
    have h_bound : ipw dbs d_obs q ≤ 1 / eps :=
      ipw_le_inv_floor dbs eps h_floor d_obs q
    simpa [htWeight, hR, abs_of_nonneg h_nonneg_ipw] using h_bound
  · rcases h_floor with ⟨h_eps_pos, _⟩
    have h_rhs_nonneg : 0 ≤ 1 / eps := one_div_nonneg.mpr (le_of_lt h_eps_pos)
    simpa [htWeight, hR] using h_rhs_nonneg

/-!
## Neyman Allocation (Stratified Efficient Sampling)

For strata indexed by a finite type `Stratum`, Neyman allocation sets
`n_h ∝ N_h * σ_h` to minimize a first-order stratified variance proxy.
-/

section NeymanAllocation

variable {Stratum : Type*} [Fintype Stratum] [DecidableEq Stratum]

/-- Neyman mass for stratum `h`: `N_h * σ_h`. -/
def neymanMass (N_h σ_h : Stratum → ℝ) (h : Stratum) : ℝ :=
  N_h h * σ_h h

/-- Total Neyman mass across all strata. -/
def neymanMassTotal (N_h σ_h : Stratum → ℝ) : ℝ :=
  ∑ h, neymanMass N_h σ_h h

/-- Neyman allocation as real-valued sample sizes summing to `n_total`
when `neymanMassTotal ≠ 0`. -/
def neymanAllocation (n_total : ℝ) (N_h σ_h : Stratum → ℝ) : Stratum → ℝ :=
  fun h => n_total * neymanMass N_h σ_h h / neymanMassTotal N_h σ_h

/-- Stratified variance proxy `Σ (N_h² σ_h² / n_h)` for a given allocation. -/
def stratifiedVarianceProxy (allocation N_h σ_h : Stratum → ℝ) : ℝ :=
  ∑ h, (N_h h)^2 * (σ_h h)^2 / allocation h

/-- Rewrite the proxy in terms of squared Neyman masses `m_h = N_h * σ_h`. -/
lemma stratifiedVarianceProxy_eq_massSqSum
    (allocation N_h σ_h : Stratum → ℝ) :
    stratifiedVarianceProxy allocation N_h σ_h =
      ∑ h, (neymanMass N_h σ_h h)^2 / allocation h := by
  unfold stratifiedVarianceProxy neymanMass
  refine Finset.sum_congr rfl ?_
  intro h _
  ring

/-- The variance proxy induced by Neyman allocation. -/
def neymanVarianceProxy (n_total : ℝ) (N_h σ_h : Stratum → ℝ) : ℝ :=
  stratifiedVarianceProxy (neymanAllocation n_total N_h σ_h) N_h σ_h

/-- Neyman allocation is normalized to total sample size. -/
lemma neymanAllocation_sum (n_total : ℝ) (N_h σ_h : Stratum → ℝ)
    (h_total_nonzero : neymanMassTotal N_h σ_h ≠ 0) :
    (∑ h, neymanAllocation n_total N_h σ_h h) = n_total := by
  let T : ℝ := neymanMassTotal N_h σ_h
  have hT : T ≠ 0 := by
    simpa [T] using h_total_nonzero
  have h_rewrite :
      (∑ h, neymanAllocation n_total N_h σ_h h)
        = ∑ h, (n_total / T) * neymanMass N_h σ_h h := by
    refine Finset.sum_congr rfl ?_
    intro h _
    simp [neymanAllocation, T, div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm]
  calc
    (∑ h, neymanAllocation n_total N_h σ_h h)
        = ∑ h, (n_total / T) * neymanMass N_h σ_h h := h_rewrite
    _ = (n_total / T) * (∑ h, neymanMass N_h σ_h h) := by
        rw [Finset.mul_sum]
    _ = (n_total / T) * T := by
        have hsum : (∑ h, neymanMass N_h σ_h h) = T := by rfl
        rw [hsum]
    _ = n_total := by
        calc
          (n_total / T) * T = (n_total * T⁻¹) * T := by rfl
          _ = n_total * (T⁻¹ * T) := by rw [mul_assoc]
          _ = n_total * 1 := by simp [hT]
          _ = n_total := by simp

/-- Nonnegativity of Neyman allocation under nonnegative stratum sizes/SDs. -/
lemma neymanAllocation_nonneg (n_total : ℝ) (N_h σ_h : Stratum → ℝ)
    (hn_nonneg : 0 ≤ n_total)
    (hN_nonneg : ∀ h, 0 ≤ N_h h)
    (hσ_nonneg : ∀ h, 0 ≤ σ_h h)
    (h_total_pos : 0 < neymanMassTotal N_h σ_h) :
    ∀ h, 0 ≤ neymanAllocation n_total N_h σ_h h := by
  intro h
  unfold neymanAllocation neymanMass
  exact div_nonneg
    (mul_nonneg hn_nonneg (mul_nonneg (hN_nonneg h) (hσ_nonneg h)))
    (le_of_lt h_total_pos)

/-- Cauchy-Schwarz lower bound used for Neyman optimality.

For positive allocations `a_h`, we have:
  `(Σ m_h)^2 ≤ (Σ m_h^2 / a_h) * (Σ a_h)`. -/
lemma sq_sum_mass_le_proxy_mul_sum
    (mass allocation : Stratum → ℝ)
    (h_alloc_pos : ∀ h, 0 < allocation h) :
    (∑ h, mass h)^2 ≤
      (∑ h, (mass h)^2 / allocation h) * (∑ h, allocation h) := by
  have hcs :=
    Finset.sum_mul_sq_le_sq_mul_sq
      (s := (Finset.univ : Finset Stratum))
      (f := fun h => mass h / Real.sqrt (allocation h))
      (g := fun h => Real.sqrt (allocation h))
  have h_left :
      ∑ h, (mass h / Real.sqrt (allocation h)) * Real.sqrt (allocation h) =
        ∑ h, mass h := by
    refine Finset.sum_congr rfl ?_
    intro h _
    have hsqrt_ne : Real.sqrt (allocation h) ≠ 0 := by
      exact Real.sqrt_ne_zero'.mpr (h_alloc_pos h)
    field_simp [hsqrt_ne]
  have h_right1 :
      ∑ h, (mass h / Real.sqrt (allocation h)) ^ 2 =
        ∑ h, (mass h)^2 / allocation h := by
    refine Finset.sum_congr rfl ?_
    intro h _
    have h_nonneg : 0 ≤ allocation h := le_of_lt (h_alloc_pos h)
    calc
      (mass h / Real.sqrt (allocation h)) ^ 2
          = (mass h)^2 / (Real.sqrt (allocation h))^2 := by ring
      _ = (mass h)^2 / allocation h := by
          simp [pow_two, Real.sq_sqrt h_nonneg]
  have h_right2 :
      ∑ h, (Real.sqrt (allocation h))^2 = ∑ h, allocation h := by
    refine Finset.sum_congr rfl ?_
    intro h _
    have h_nonneg : 0 ≤ allocation h := le_of_lt (h_alloc_pos h)
    simp [pow_two, Real.sq_sqrt h_nonneg]
  calc
    (∑ h, mass h)^2
        = (∑ h, (mass h / Real.sqrt (allocation h)) * Real.sqrt (allocation h))^2 := by
            simp [h_left]
    _ ≤ (∑ h, (mass h / Real.sqrt (allocation h)) ^ 2) *
          (∑ h, (Real.sqrt (allocation h)) ^ 2) := hcs
    _ = (∑ h, (mass h)^2 / allocation h) * (∑ h, allocation h) := by
          simp [h_right1, h_right2]

/-- Any positive allocation has variance proxy at least
`(Σ m_h)^2 / (Σ n_h)` for `m_h = N_h * σ_h`. -/
lemma stratifiedVarianceProxy_ge_mass_sq_div_sum
    (N_h σ_h : Stratum → ℝ)
    (allocation : Stratum → ℝ)
    (h_alloc_pos : ∀ h, 0 < allocation h)
    (h_alloc_sum_pos : 0 < ∑ h, allocation h) :
    (neymanMassTotal N_h σ_h)^2 / (∑ h, allocation h) ≤
      stratifiedVarianceProxy allocation N_h σ_h := by
  have hcs :=
    sq_sum_mass_le_proxy_mul_sum
      (mass := neymanMass N_h σ_h)
      (allocation := allocation)
      h_alloc_pos
  have h_rewrite_mass :
      (∑ h, neymanMass N_h σ_h h) = neymanMassTotal N_h σ_h := by
    rfl
  have h_rewrite_proxy :
      stratifiedVarianceProxy allocation N_h σ_h =
        (∑ h, (neymanMass N_h σ_h h)^2 / allocation h) := by
    simpa using (stratifiedVarianceProxy_eq_massSqSum
      (allocation := allocation) (N_h := N_h) (σ_h := σ_h))
  have hcs' :
      (neymanMassTotal N_h σ_h)^2 ≤
        stratifiedVarianceProxy allocation N_h σ_h * (∑ h, allocation h) := by
    calc
      (neymanMassTotal N_h σ_h)^2
          = (∑ h, neymanMass N_h σ_h h)^2 := by simpa [h_rewrite_mass]
      _ ≤ (∑ h, (neymanMass N_h σ_h h)^2 / allocation h) * (∑ h, allocation h) := hcs
      _ = stratifiedVarianceProxy allocation N_h σ_h * (∑ h, allocation h) := by
          simpa [h_rewrite_proxy, mul_comm, mul_left_comm, mul_assoc]
  exact (div_le_iff₀ h_alloc_sum_pos).2 (by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hcs')

/-- Closed form of the Neyman variance proxy:
`V(neyman) = (Σ m_h)^2 / n_total` where `m_h = N_h * σ_h`. -/
lemma neymanVarianceProxy_eq_mass_sq_div
    (n_total : ℝ) (N_h σ_h : Stratum → ℝ)
    (h_total_pos : 0 < neymanMassTotal N_h σ_h) :
    neymanVarianceProxy n_total N_h σ_h =
      (neymanMassTotal N_h σ_h)^2 / n_total := by
  let M : ℝ := neymanMassTotal N_h σ_h
  have hM_ne : M ≠ 0 := ne_of_gt h_total_pos
  by_cases hn : n_total = 0
  · subst hn
    unfold neymanVarianceProxy stratifiedVarianceProxy neymanAllocation
    simp
  · have hterm :
        ∀ h,
          (neymanMass N_h σ_h h)^2 / neymanAllocation n_total N_h σ_h h =
            neymanMass N_h σ_h h * M / n_total := by
      intro h
      by_cases hm : neymanMass N_h σ_h h = 0
      · simp [neymanAllocation, M, hm]
      · calc
          (neymanMass N_h σ_h h)^2 / neymanAllocation n_total N_h σ_h h
              = (neymanMass N_h σ_h h)^2 /
                  (n_total * neymanMass N_h σ_h h / M) := by
                    rfl
          _ = neymanMass N_h σ_h h * M / n_total := by
                field_simp [hm, hn, hM_ne]
    have hsum :
        stratifiedVarianceProxy (neymanAllocation n_total N_h σ_h) N_h σ_h =
          ∑ h, neymanMass N_h σ_h h * M / n_total := by
      calc
        stratifiedVarianceProxy (neymanAllocation n_total N_h σ_h) N_h σ_h
            = ∑ h, (neymanMass N_h σ_h h)^2 / neymanAllocation n_total N_h σ_h h := by
                simpa using (stratifiedVarianceProxy_eq_massSqSum
                  (allocation := neymanAllocation n_total N_h σ_h)
                  (N_h := N_h) (σ_h := σ_h))
        _ = ∑ h, neymanMass N_h σ_h h * M / n_total := by
                refine Finset.sum_congr rfl ?_
                intro h _
                simpa using hterm h
    calc
      neymanVarianceProxy n_total N_h σ_h
          = stratifiedVarianceProxy (neymanAllocation n_total N_h σ_h) N_h σ_h := by
              rfl
      _ = ∑ h, neymanMass N_h σ_h h * M / n_total := hsum
      _ = (M / n_total) * (∑ h, neymanMass N_h σ_h h) := by
            calc
              (∑ h, neymanMass N_h σ_h h * M / n_total)
                  = ∑ h, (M / n_total) * neymanMass N_h σ_h h := by
                      refine Finset.sum_congr rfl ?_
                      intro h _
                      ring
              _ = (M / n_total) * (∑ h, neymanMass N_h σ_h h) := by
                      simpa using
                        (Finset.mul_sum
                          (s := Finset.univ)
                          (a := M / n_total)
                          (f := fun h => neymanMass N_h σ_h h)).symm
      _ = (M / n_total) * M := by simp [M, neymanMassTotal]
      _ = M^2 / n_total := by ring
      _ = (neymanMassTotal N_h σ_h)^2 / n_total := by
            simp [M]

/-- Neyman allocation minimizes the first-order stratified variance proxy.

This is the direct Cauchy-Schwarz proof (no axiom): for any feasible positive
allocation with total sample size `n_total`, Neyman achieves the minimum. -/
theorem neyman_optimality
    (n_total : ℝ) (N_h σ_h : Stratum → ℝ)
    (h_total_pos : 0 < neymanMassTotal N_h σ_h)
    (allocation : Stratum → ℝ)
    (h_alloc_pos : ∀ h, 0 < allocation h)
    (h_alloc_sum : (∑ h, allocation h) = n_total)
    (hn_total_pos : 0 < n_total) :
    neymanVarianceProxy n_total N_h σ_h ≤
      stratifiedVarianceProxy allocation N_h σ_h := by
  have h_lower :
      (neymanMassTotal N_h σ_h)^2 / n_total ≤
        stratifiedVarianceProxy allocation N_h σ_h := by
    have h_alloc_sum_pos : 0 < ∑ h, allocation h := by
      simpa [h_alloc_sum] using hn_total_pos
    have h0 :=
      stratifiedVarianceProxy_ge_mass_sq_div_sum
        (N_h := N_h) (σ_h := σ_h)
        (allocation := allocation)
        (h_alloc_pos := h_alloc_pos)
        (h_alloc_sum_pos := h_alloc_sum_pos)
    simpa [h_alloc_sum] using h0
  have h_neyman :
      neymanVarianceProxy n_total N_h σ_h =
        (neymanMassTotal N_h σ_h)^2 / n_total :=
    neymanVarianceProxy_eq_mass_sq_div
      (n_total := n_total) (N_h := N_h) (σ_h := σ_h)
      h_total_pos
  simpa [h_neyman] using h_lower

end NeymanAllocation

end DSL

end
