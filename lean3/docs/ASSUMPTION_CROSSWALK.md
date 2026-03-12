# Assumption Crosswalk

This file records the exact correspondence between paper-level assumptions and
their Lean formalizations.

## Local laws

| Paper name | Meaning | Lean name | File |
|---|---|---|---|
| C1 | Sufficiency | `L1` | `FormalProofs/OPT/LocalLaws.lean` |
| C2 | Idempotence / on-range stability | `L3` | `FormalProofs/OPT/LocalLaws.lean` |
| C3 | Merge consistency | `L2` | `FormalProofs/OPT/LocalLaws.lean` |

The C2/C3 numbering changes because the Lean development uses `L1/L2/L3` in the
order most convenient for the inductive proofs.

## Global assumptions

| Paper name | Meaning | Lean name | File |
|---|---|---|---|
| A1 | Global sufficiency | `A1_GlobalSufficiency` | `FormalProofs/OPT/GlobalAssumptions.lean` |
| A2 | Two-route identity | `A2_TwoRouteIdentity` | `FormalProofs/OPT/GlobalAssumptions.lean` |
| A3 | Merge function existence | `A3_MergeFunction` | `FormalProofs/OPT/GlobalAssumptions.lean` |

## Preference-learning assumptions

| Paper assumption | Lean surface | File |
|---|---|---|
| Oracle-measurable policy | `OracleMeasurablePolicy`, `GRPOOracleMeasurable`, `OracleMeasurableGRPORLLoss` | `FormalProofs/OPT/OracleMeasurable.lean` and method-specific files |
| Oracle-indexed pair generator | `OracleIndexedPairGen` | `FormalProofs/OPT/OracleMeasurable.lean` |
| Oracle-indexed group generator | `OracleIndexedGroupGen` | `FormalProofs/OPT/OracleMeasurable.lean` |
| Oracle-indexed ranker / reward | `OracleIndexedRanker`, `OracleIndexedReward` | `FormalProofs/OPT/PreferenceLearning.lean` and `FormalProofs/OPT/PreferenceBounds.lean` |
| Policy Lipschitz envelope | `PolicyLipschitz`, `GRPOPolicyLipschitz`, `RewardLipschitzGRPO` | `FormalProofs/OPT/PreferenceBounds.lean` |

## Theorem-backed reduction assumptions

| Paper notion | Lean object | File |
|---|---|---|
| Exact theorem-backed reduction | `ExactTheoremBacked` | `FormalProofs/OPT/TheoremBackingAssumptions.lean` |
| Approximate theorem-backed reduction | `ApproxTheoremBacked` | `FormalProofs/OPT/TheoremBackingAssumptions.lean` |
| Direct local-law route | `LocalLawsBundle`, `ApproxLocalLawsBundle` | `FormalProofs/OPT/TheoremBackingAssumptions.lean` |
| Sketch / codec route | `SketchCodecExactAssumptions`, `SketchCodecApproxAssumptions` | `FormalProofs/OPT/TheoremBackingAssumptions.lean` |

## Oracle measurement and adaptive-tree assumptions

| Paper notion | Lean object | File |
|---|---|---|
| Exact oracle identification of a latent feature | `OracleRecoversFeature` | `FormalProofs/OPT/TheoremBackingMeasurementError.lean` |
| Approximate oracle-to-feature control | `FeatureLipschitzFromOracle` | `FormalProofs/OPT/TheoremBackingApproxMeasurementError.lean` |
| Stochastic adaptive soundness | `StochasticAdaptiveChunkingSound` | `FormalProofs/OPT/AdaptiveChunkingBridge.lean` |
| Stochastic adaptive approximate local laws | `StochasticAdaptiveApproxLocalLaws` | `FormalProofs/OPT/AdaptiveChunkingBridge.lean` |

## Axiom policy

The active build uses one axiom:

| Axiom | Meaning | File |
|---|---|---|
| `ExpectedGroupLossLipschitz` | Expected group loss is Lipschitz in oracle distance | `FormalProofs/Axioms.lean` |

This is the only non-derived assumption in the active theorem stack.
