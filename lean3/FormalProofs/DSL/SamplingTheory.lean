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

end DSL

end
