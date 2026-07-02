# Unified Learning Theorem Map

Date: 2026-05-18

This is the paper-facing map for the unified C-TreePO / ThinkingTrees learning
procedure.

## Canonical Objects

| Paper object | Lean name | File |
|---|---|---|
| Top-level IID / exchangeability | `DSL.TopLevelIID`, `DSL.TopLevelExchangeable` | `lean3/FormalProofs/DSL/Honesty.lean` |
| Derived-row parent map | `DSL.ParentOf`, `DSL.DerivedRowHonestyContract` | `lean3/FormalProofs/DSL/Honesty.lean` |
| K-fold honesty | `DSL.KFoldHonestTraining`, `DSL.KFoldHonestEvaluation` | `lean3/FormalProofs/DSL/Honesty.lean` |
| Unified chunker/g/oracle honesty | `DSL.UnifiedLearningHonesty` | `lean3/FormalProofs/DSL/Honesty.lean` |
| Chunker objective | `DSL.ChunkerObjectiveTerms` | `lean3/FormalProofs/DSL/Honesty.lean` |
| Finite support span | `DSL.Span` | `lean3/FormalProofs/DSL/DocumentStructure.lean` |
| Admissible chunk partition | `DSL.AdmissiblePartition` | `lean3/FormalProofs/DSL/DocumentStructure.lean` |
| Chunk partition contract | `DSL.ChunkPartitionContract` | `lean3/FormalProofs/DSL/DocumentStructure.lean` |
| Run manifest contract | `DSL.RunManifestContract` | `lean3/FormalProofs/DSL/DocumentStructure.lean` |
| Manifest role/support validity | `DSL.ManifestRolesConsistent`, `DSL.ManifestSupportsValid` | `lean3/FormalProofs/DSL/DocumentStructure.lean` |
| Influence-weighted audit overlap | `FormalProofs.OPT.InfluenceWeightedAuditOverlap` | `lean3/FormalProofs/OPT/InfluenceWeightedLocalLaws.lean` |
| Influence-weighted local-law certificate | `FormalProofs.OPT.InfluenceWeightedErrorCertificate` | `lean3/FormalProofs/OPT/InfluenceWeightedLocalLaws.lean` |
| Unified final certificate | `DSL.UnifiedLearningErrorCertificate` | `lean3/FormalProofs/DSL/UnifiedLearningCertificate.lean` |
| Component-radius provenance | `DSL.UnifiedLearningComponentEvidence` | `lean3/FormalProofs/DSL/UnifiedLearningCertificate.lean` |
| Bundled paper assumptions | `DSL.UnifiedLearningPaperAssumptions` | `lean3/FormalProofs/DSL/UnifiedLearningCertificate.lean` |

## Final Theorem Surface

The deterministic paper theorem is:

```lean
DSL.unified_learning_final_paper_certificate
```

It assumes:

- top-level sampling: `TopLevelIID ∨ TopLevelExchangeable`;
- three-layer honesty for chunker, `g`, and oracle/readout;
- admissible finite chunk partitions;
- manifest parent IDs, split roles, support spans, artifact lineage, and positive propensities;
- an influence-weighted local-law certificate for the local-law radius;
- calibration, estimation, and clipping component bounds.

It proves:

```text
|target gap| <= |reported estimate|
              + local-law radius
              + calibration radius
              + estimation radius
              + clipping radius
```

The high-probability paper theorem is:

```lean
DSL.unified_learning_final_paper_certificate_high_prob
```

It assumes the same paper context plus `DSL.UnifiedLearningComponentEvidence`,
which records high-probability event bounds for local-law, calibration,
estimation, and clipping errors.

It proves:

```text
Pr(|target gap| > total bound) <=
  delta_local + delta_calibration + delta_estimation + delta_clipping
```

## Compatibility

The older end-to-end surface `DSL.PaperErrorCertificate` remains available in
`lean3/FormalProofs/DSL/TreePOEndToEnd.lean`.  The current canonical certificate
is:

```lean
DSL.UnifiedLearningErrorCertificate
```

with compatibility constructor:

```lean
DSL.UnifiedLearningErrorCertificate.ofPaperErrorCertificate
```

and alias:

```lean
DSL.CurrentPaperErrorCertificate
```

## Paper-Ready Interpretation

The paper should cite the bundled theorem surface for the main claim.  Tables
should report the same components as the Lean certificate:

- reported honest estimate;
- local-law / transported-distortion radius;
- calibration radius;
- honest statistical radius;
- clipping or floor radius;
- total failure probability split.

Rows in empirical manifests should satisfy `RunManifestContract` and include
the fields listed in `docs/unified_learning_procedure.md`.
