# Flajolet et al. 2007 Proof Map

Source: Flajolet, Fusy, Gandouet, and Meunier, "HyperLogLog: the analysis of a
near-optimal cardinality estimation algorithm" (AOFA 2007).

Lean module:

- `FormalProbability/ML/MergeableSummaries/Flajolet2007.lean`

## Mechanized Core

| Paper concept | Lean declaration | Status |
|---|---|---|
| Hash word split into register bucket and suffix bits | `Flajolet2007.HashObservation` | Mechanized abstraction |
| Position of first one bit | `Flajolet2007.rho` | Mechanized |
| `rho >= 1` | `Flajolet2007.rho_positive_statement` | Proved |
| Prefix bits parse into `Fin (2^p)` bucket | `Flajolet2007.bitsToNat`, `Flajolet2007.bitsToNat_lt_two_pow_length`, `Flajolet2007.bucketOfPrefix` | Proved |
| Full hash word to bucket/suffix observation | `Flajolet2007.HashWord`, `Flajolet2007.HashWord.bucket`, `Flajolet2007.HashWord.toObservation`, `Flajolet2007.HashWord.rank_pos` | Mechanized/proved |
| Ideal hash family wrapper | `Flajolet2007.IdealHashFamily`, `IdealHashFamily.observation`, `IdealHashFamily.build` | Mechanized deterministic interface for an externally supplied ideal hash |
| Random ideal-hash law wrapper | `Flajolet2007.RandomIdealHashFamily`, `RandomIdealHashFamily.seedFamily`, `RandomIdealHashFamily.seedFamily_build_append`, `RandomIdealHashFamily.seedFamily_hierarchical`, `flajolet2007_10c_randomIdealHash_seedFamily_build_append`, `flajolet2007_10d_randomIdealHash_seedFamily_hierarchical` | Mechanized as a measure-indexed law interface plus seedwise state-level HLL algebra; existence/independence of the ideal source remains external |
| HLL register vector | `HLLRegisters` | Existing mechanized core |
| One-observation update raises one bucket by max | `HLLRegisters.update`, `regs_update_bucket`, `regs_update_of_ne` | Proved |
| Register merge is pointwise max | `HLLRegisters.merge` | Existing |
| Merge associativity / commutativity / idempotence | `flajolet2007_01_hll_merge_associative`, `flajolet2007_02_hll_merge_commutative`, `flajolet2007_03_hll_merge_idempotent` | Proved |
| Build homomorphism over append | `flajolet2007_04_hll_build_append`, `flajolet2007_09_hll_buildFromHashes_append` | Proved |
| State-level mergeability | `flajolet2007_05_hll_state_level_mergeable`, `flajolet2007_10_hll_hash_state_level_mergeable`, `flajolet2007_10b_idealHash_state_level_mergeable` | Proved |
| Bias constant `alpha_m` | `HLLRegisters.alpha` | Defined |
| Harmonic denominator and indicator | `HLLRegisters.inversePowerSum`, `HLLRegisters.indicatorZ` | Defined |
| Raw estimator `alpha_m m^2 Z` | `HLLRegisters.rawEstimator` | Defined |
| Empty-state readout sanity | `flajolet2007_11_hll_indicatorZ_empty`, `flajolet2007_12_hll_rawEstimator_empty` | Proved |
| Linear-counting correction | `HLLRegisters.linearCountingCorrection` | Defined |
| Large-range correction | `HLLRegisters.largeRangeCorrection` | Defined |
| RSE `1.04/sqrt(m)` | `Flajolet2007.relativeStandardErrorOfRegisterCount`, `flajolet2007_13_hll_relativeStandardError_registerCount` | Defined/proved |
| `p = 14` RSE arithmetic | `flajolet2007_hll_rse_p14_exact`, `flajolet2007_hll_rse_p14_under_one_percent`, `flajolet2007_07b_hll_rse_p14_exact`, `flajolet2007_07c_hll_rse_p14_under_one_percent` | Proved: `13/1600 < 1/100` |

## Analytic Citation Schemas

The AOFA paper's deep probability analysis is not asserted as an axiom.  The
Lean surface names the target obligations in mathlib asymptotic language:

| Paper claim | Lean schema |
|---|---|
| Theorem 1(i), asymptotic almost-unbiasedness | `Flajolet2007.AsymptoticallyAlmostUnbiased` |
| Theorem 1(ii), standard-error constant | `Flajolet2007.RelativeStandardErrorAsymptotic` |
| Big-O relaxation of the RSE claim | `Flajolet2007.RelativeStandardErrorBigO` |
| Theorem 1 package | `Flajolet2007.StochasticEstimatorClaims`, `flajolet2007_14_theorem1_stochasticEstimatorClaims` |
| Theorem 1 Big-O consequence package | `Flajolet2007.StochasticEstimatorClaims.toBigOClaims`, `Flajolet2007.hll_stochasticEstimatorBigOClaims`, `flajolet2007_17_relativeStandardErrorBigO_of_asymptotic` |
| Fixed-cardinality expectation formula | `Flajolet2007.FixedCardinalityIndicatorExpectation` |
| Poisson probability mass weight | `Flajolet2007.poissonWeight`, `Flajolet2007.poissonWeight_zero`, `flajolet2007_15a_poissonWeight_zero` |
| Poissonized series transform | `Flajolet2007.poissonizedSeries`, `Flajolet2007.PoissonizedBySeries`, `flajolet2007_15b_PoissonizedBySeries` |
| Poissonized expectation asymptotic | `Flajolet2007.PoissonIndicatorExpectationAsymptotic` |
| Depoissonization transfer | `Flajolet2007.DepoissonizationTransfer` |
| Fixed-cardinality asymptotic from poissonization plus depoissonization | `Flajolet2007.fixedCardinality_asymptotic_of_poisson_depoissonization`, `flajolet2007_16_fixedCardinality_asymptotic_of_poisson_depoissonization` |
| Packaged poissonization/depoissonization analysis | `Flajolet2007.PoissonizationDepoissonizationAnalysis`, `flajolet2007_15c_PoissonizationDepoissonizationAnalysis` |
| Fixed-cardinality asymptotic from the packaged analysis | `Flajolet2007.fixedCardinality_asymptotic_of_poissonization_analysis`, `flajolet2007_16b_fixedCardinality_asymptotic_of_poissonization_analysis` |
| Second moment asymptotic | `Flajolet2007.IndicatorSecondMomentAsymptotic` |
| Variance/RSE asymptotic | `Flajolet2007.VarianceAsymptotic` |

## C-TreePO Bridge Names

`FormalProofs/OPT/MergeableReduction.lean` re-exports the paper-relevant HLL
surface under C-TreePO-facing names:

- `ctreepo_flajolet2007_hll_state_level_mergeable`
- `ctreepo_flajolet2007_hll_buildFromHashes_append`
- `ctreepo_flajolet2007_hll_hash_state_level_mergeable`
- `ctreepo_flajolet2007_idealHash_state_level_mergeable`
- `ctreepo_flajolet2007_randomIdealHash_seedFamily_build_append`
- `ctreepo_flajolet2007_randomIdealHash_seedFamily_hierarchical`
- `ctreepo_flajolet2007_bitsToNat_lt_two_pow_length`
- `ctreepo_flajolet2007_hashWord_rank_positive`
- `ctreepo_flajolet2007_hll_indicatorZ_empty`
- `ctreepo_flajolet2007_hll_rawEstimator_empty`
- `ctreepo_flajolet2007_hll_relativeStandardError_registerCount`
- `ctreepo_flajolet2007_hll_rse_p14_exact`
- `ctreepo_flajolet2007_hll_rse_p14_under_one_percent`
- `ctreepo_flajolet2007_hll_stochasticEstimatorClaims`
- `ctreepo_flajolet2007_hll_stochasticEstimatorBigOClaims`
- `ctreepo_flajolet2007_PoissonizedBySeries`
- `ctreepo_flajolet2007_PoissonizationDepoissonizationAnalysis`
- `ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poisson_depoissonization`
- `ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poissonization_analysis`
- `ctreepo_flajolet2007_relativeStandardErrorBigO_of_asymptotic`

## Scope Boundary

The verified C-TreePO claim is exact: HLL is a mergeable state sketch because
register states merge by pointwise max and readouts happen after the merge.
The deterministic ideal-hash pipeline and a random ideal-hash law wrapper are
mechanized seedwise, so the algebraic HLL claim can be stated under a
measure-indexed hash source.  The probabilistic estimator constants are
expressed as precise Lean obligations.  The Poisson-mixture series package and
the checked composition from poissonized asymptotics plus a depoissonization
transfer to fixed-cardinality asymptotics are mechanized; the actual existence
of an ideal independent hash source, the Mellin-transform estimates, and the
analytic depoissonization proof itself are not mechanized in this pass.
