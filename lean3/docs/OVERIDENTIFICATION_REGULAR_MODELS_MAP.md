# Overidentification in Regular Models: PDF-to-Lean Map

This note records the paper-facing Lean names exported by
`FormalProofs.Econometrics.Overidentification`.

The generic proofs live in the sibling library `FormalProbability`. The names below are the stable
`FormalProofs` exports intended for paper-level navigation.

## Main Text

| Paper item | Lean export |
| --- | --- |
| Definition 2.1 | `Econometrics.OveridentificationRegularModels.DQMPath` |
| Definition 2.2 | `Econometrics.OveridentificationRegularModels.IsLocallyJustIdentified` |
| Definition 2.2 | `Econometrics.OveridentificationRegularModels.IsLocallyOveridentified` |
| Lemma 2.1(i) | `Econometrics.OveridentificationRegularModels.localJustIdentification_iff_trivialOrthogonalComplement` |
| Lemma 2.1(ii) | `Econometrics.OveridentificationRegularModels.localOveridentification_iff_nontrivialOrthogonalComplement` |
| Lemma 2.1(ii) | `Econometrics.OveridentificationRegularModels.localOveridentification_iff_exists_nonzeroOrthogonalScore` |
| Equations (7)-(9) | `Econometrics.OveridentificationRegularModels.LinearGMMModel` |
| Equation (8) | `Econometrics.OveridentificationRegularModels.gmmResidualMaker` |
| Equation (9) | `Econometrics.OveridentificationRegularModels.gmmOrthogonalRestrictionSpace` |
| Theorem 3.1(i) | `Econometrics.OveridentificationRegularModels.regularEstimators_firstOrderEquivalent` |
| Theorem 3.1(ii) | `Econometrics.OveridentificationRegularModels.distinctRegularEstimators_exist_underOveridentification` |
| Lemma 3.1 / Remark 3.4 direction witness | `Econometrics.OveridentificationRegularModels.localOveridentification_iff_exists_nonzeroOrthogonalScore` |
| Lemma 3.2 | `Econometrics.OveridentificationRegularModels.varianceEquality_onDenseSet_iff_localJustIdentification` |
| Lemma 3.3(ii) | `Econometrics.OveridentificationRegularModels.regularEstimatorDifference_isOrthogonal` |
| Lemma 3.4(i) | `Econometrics.OveridentificationRegularModels.maintainedModelScoreDecomposition` |
| Lemma 3.4(ii) | `Econometrics.OveridentificationRegularModels.fullScoreDecomposition` |
| Remark 3.5 / Eq. (25) | `Econometrics.OveridentificationRegularModels.incrementalSarganOrthogonalCharacterization` |
| Theorem 3.2 / dense-set variance criterion | `Econometrics.OveridentificationRegularModels.varianceEquality_onDenseSet_iff_localJustIdentification` |
| Condition A | `Econometrics.OveridentificationRegularModels.LinearizedConditionalMomentModel` |
| Theorem 4.1 | `Econometrics.OveridentificationRegularModels.conditionalMomentOrthogonalScoreSpace_eq_bot_iff_denseRange` |
| Lemma 4.1 | `Econometrics.OveridentificationRegularModels.conditionalMomentDenseRange_iff_trivialAdjointKernel` |
| Theorem 4.2 | `Econometrics.OveridentificationRegularModels.conditionalMomentFullEfficiency_onDenseSet_iff_denseRange` |
| Lemma 4.2 | `Econometrics.OveridentificationRegularModels.conditionalMomentLocalOveridentification_iff_exists_nonzeroScore` |
| Corollary 4.1 | `Econometrics.OveridentificationRegularModels.conditionalMeanExample_locallyJustIdentified` |
| Corollary 4.1 | `Econometrics.OveridentificationRegularModels.quantileExample_locallyJustIdentified` |
| Example 4.1 | `Econometrics.OveridentificationRegularModels.partiallyLinearExample_locallyOveridentified` |
| Example 4.2 | `Econometrics.OveridentificationRegularModels.conditionalMeanExample_locallyJustIdentified` |
| Example 4.3 | `Econometrics.OveridentificationRegularModels.partiallyLinearExample_locallyOveridentified` |
| Remark 4.1 / efficient second-stage bridge | `Econometrics.OveridentificationRegularModels.conditionalMomentFullEfficiency_onDenseSet_iff_denseRange` |

## Appendix A

| Paper item | Lean export |
| --- | --- |
| Lemma A.1 | `Econometrics.OveridentificationRegularModels.coneLocalJustIdentification_iff_trivialInnerDual` |
| Definition A.1 | `Semiparametric.AsymptoticallyLocallyAdmissible` |
| Lemma A.2 | `Econometrics.OveridentificationRegularModels.coneLocalOveridentification_iff_nontrivialInnerDual` |
| Theorem A.1 | `Econometrics.OveridentificationRegularModels.coneLocalOveridentification_iff_exists_nonzeroInnerDual` |
| Theorem A.2 | `Econometrics.OveridentificationRegularModels.coneLocalOveridentification_iff_nontrivialInnerDual` |
| Theorem A.3 | `Econometrics.OveridentificationRegularModels.noNontrivialInnerDual_underLocalJustIdentification` |

## Appendices B, E, F, G

| Paper item | Lean export |
| --- | --- |
| Theorem B.1(i) | `Econometrics.OveridentificationRegularModels.gaussianShift_coordinateDecomposition` |
| Theorem B.1(ii) | `Econometrics.OveridentificationRegularModels.gaussianShift_scalarizedConvolution` |
| Theorem E.1 | `Econometrics.OveridentificationRegularModels.conditionalMomentFullEfficiency_onDenseSet_iff_denseRange` |
| Lemma F.1 | `Econometrics.OveridentificationRegularModels.conditionalMomentLocalOveridentification_iff_exists_nonzeroScore` |
| Appendix G examples | `Econometrics.OveridentificationRegularModels.partiallyLinearExample_locallyOveridentified` |
| Appendix G examples | `Econometrics.OveridentificationRegularModels.quantileExample_locallyJustIdentified` |

## GMM Smoke Checks

| Paper discussion | Lean export |
| --- | --- |
| `R(P) = 0` iff local just identification | `Econometrics.OveridentificationRegularModels.gmmLocalJustIdentification_iff_residualMaker_zero` |
| Overidentified GMM yields nontrivial test direction | `Econometrics.OveridentificationRegularModels.gmmOveridentified_yieldsNontrivialJDirection` |
| Just-identified GMM kills the orthogonal complement | `Econometrics.OveridentificationRegularModels.gmmJustIdentified_killsOrthogonalComplement` |
| Incremental Sargan specialization | `Econometrics.OveridentificationRegularModels.gmmIncrementalSargan_expansion` |

## Concrete DQM API

These are not separate numbered claims in the PDF. They are the practical pathwise layer added on top
of Definition 2.1 so local identification can be used directly from concrete DQM paths.

| Practical item | Lean export |
| --- | --- |
| Centered score space `L²₀(P)` | `Econometrics.OveridentificationRegularModels.CenteredScoreSpace` |
| DQM-path tangent space | `Econometrics.OveridentificationRegularModels.pathwiseTangentSpace` |
| DQM-path local just identification | `Econometrics.OveridentificationRegularModels.IsPathwiseLocallyJustIdentified` |
| DQM-path local overidentification | `Econometrics.OveridentificationRegularModels.IsPathwiseLocallyOveridentified` |
| Dominated square-root density notation | `Econometrics.OveridentificationRegularModels.squareRootDensity` |
| Canonical local alternatives `h / √n` | `Econometrics.OveridentificationRegularModels.localAlternativeScale` |
| Each DQM path score lies in its generated tangent space | `Econometrics.OveridentificationRegularModels.dqmPathScore_mem_pathwiseTangentSpace` |
| Local log-likelihood ratio vanishes at the base point | `Econometrics.OveridentificationRegularModels.localLogLikelihoodRatio_vanishesAtBase` |
| Tangent-space monotonicity under path-family inclusion | `Econometrics.OveridentificationRegularModels.pathwiseTangentSpace_mono` |
| Union of path families generates the closed supremum tangent space | `Econometrics.OveridentificationRegularModels.pathwiseTangentSpace_union_eq_closedSup` |
| Pathwise just identification iff generated tangent space is top | `Econometrics.OveridentificationRegularModels.pathwiseLocalJustIdentification_iff_generatedTangent_eq_top` |
| Generated-score density iff tangent space is top | `Econometrics.OveridentificationRegularModels.pathwiseGeneratedScores_dense_iff_tangent_eq_top` |
| Pathwise just identification iff trivial orthogonal complement | `Econometrics.OveridentificationRegularModels.pathwiseLocalJustIdentification_iff_trivialOrthogonalComplement` |
| Pathwise overidentification iff nonzero orthogonal direction | `Econometrics.OveridentificationRegularModels.pathwiseLocalOveridentification_iff_exists_nonzeroOrthogonalScore` |
| Pathwise just identification implies first-order equivalence | `Econometrics.OveridentificationRegularModels.pathwiseRegularEstimators_firstOrderEquivalent` |
| First-order expansion along admissible local sequences | `Econometrics.OveridentificationRegularModels.dqmFirstOrderLocalExperimentExpansion_of_tendsto` |
| Absolute-continuity contiguity of local alternatives | `Econometrics.OveridentificationRegularModels.dqmLocalAlternatives_contiguous` |
| Concrete Gaussian-shift experiment induced by a DQM family | `Econometrics.OveridentificationRegularModels.dominatedDqmFamily_gaussianShiftExperiment` |
| Concrete Gaussian-shift orthogonal coordinates vanish | `Econometrics.OveridentificationRegularModels.dqmGaussianShift_orthogonalCoordinates_eq_zero` |
| Concrete Gaussian-shift coordinate decomposition | `Econometrics.OveridentificationRegularModels.dqmGaussianShift_coordinateDecomposition` |
| Concrete Gaussian-shift scalarized convolution | `Econometrics.OveridentificationRegularModels.dqmGaussianShift_scalarizedConvolution` |
| Base `n`-sample product law | `Econometrics.OveridentificationRegularModels.baseSampleMeasure` |
| Product experiment of a DQM path | `Econometrics.OveridentificationRegularModels.sampleMeasure` |
| Coordinatewise sample log-likelihood ratio | `Econometrics.OveridentificationRegularModels.sampleLogLikelihoodRatio` |
| Witness-backed product log-likelihood identity | `Econometrics.OveridentificationRegularModels.ProductLogLikelihoodRatioWitness` |
| Witness-backed quadratic product LAN | `Econometrics.OveridentificationRegularModels.ProductLANWitness` |
| Sample log-likelihood vanishes at the base point | `Econometrics.OveridentificationRegularModels.sampleLogLikelihoodRatio_vanishesAtBase` |
| Witness-backed quadratic LAN expansion | `Econometrics.OveridentificationRegularModels.productLAN_ae_eq_quadratic_plus_remainder` |
| Exponential-tilt local-alternative constructor | `Econometrics.OveridentificationRegularModels.ExponentialTiltPath` |
| Exact one-sample exponential-tilt log-likelihood | `Econometrics.OveridentificationRegularModels.exponentialTilt_localLogLikelihoodRatio_eq` |
| Exact product exponential-tilt log-likelihood | `Econometrics.OveridentificationRegularModels.exponentialTilt_sampleLogLikelihoodRatio_eq` |
| Exact exponential-tilt local-alternative expansion | `Econometrics.OveridentificationRegularModels.exponentialTilt_localAlternativeExpansion` |
| Finite probe family API | `Econometrics.OveridentificationRegularModels.FiniteProbeFamily` |
| Finite-dimensional CLT wrapper | `Econometrics.OveridentificationRegularModels.FiniteProbeCLT` |
| Cramér-Wold witness wrapper | `Econometrics.OveridentificationRegularModels.CramerWoldWitness` |
| Scalar CLT to one-dimensional finite-probe CLT | `Econometrics.OveridentificationRegularModels.scalarCLT_to_finiteProbe_dim1` |
| One-dimensional finite-probe CLT iff scalar CLT | `Econometrics.OveridentificationRegularModels.finiteProbeCLT_dim1_iff_scalar` |
| Finite-probe drift vanishes on orthogonal directions | `Econometrics.OveridentificationRegularModels.finiteProbeDrift_eq_zero_of_orthogonal` |
| Third-lemma witness shift equals finite-probe drift | `Econometrics.OveridentificationRegularModels.thirdLemma_shiftedMean_eq_finiteProbeDrift` |
| GMM-to-DQM associated-family bridge | `Econometrics.OveridentificationRegularModels.LinearGMMDQMAssociatedFamily` |
| GMM local just identification iff pathwise local just identification | `Econometrics.OveridentificationRegularModels.gmmPathwiseLocalJustIdentification_iff` |
| GMM local overidentification iff pathwise local overidentification | `Econometrics.OveridentificationRegularModels.gmmPathwiseLocalOveridentification_iff` |
| GMM local Hansen-`J` drift | `Econometrics.OveridentificationRegularModels.gmmHansenJLocalDrift` |
| GMM local incremental-Sargan drift | `Econometrics.OveridentificationRegularModels.gmmIncrementalSarganLocalDrift` |
| Overidentified GMM yields positive local Hansen-`J` drift | `Econometrics.OveridentificationRegularModels.gmmOveridentified_yieldsPositiveHansenJDrift` |
| Overidentified GMM yields positive local incremental-Sargan drift | `Econometrics.OveridentificationRegularModels.gmmOveridentified_yieldsPositiveIncrementalSarganDrift` |
| Pathwise-overidentified GMM yields positive local Hansen-`J` drift | `Econometrics.OveridentificationRegularModels.gmmPathwiseOveridentified_yieldsPositiveHansenJDrift` |
| Pathwise-overidentified GMM yields positive local incremental-Sargan drift | `Econometrics.OveridentificationRegularModels.gmmPathwiseOveridentified_yieldsPositiveIncrementalSarganDrift` |
| Conditional-moment-to-DQM associated-family bridge | `Econometrics.OveridentificationRegularModels.ConditionalMomentDQMAssociatedFamily` |
| Conditional-moment local just identification iff pathwise local just identification | `Econometrics.OveridentificationRegularModels.conditionalMomentPathwiseLocalJustIdentification_iff` |
| Conditional-moment local overidentification iff pathwise local overidentification | `Econometrics.OveridentificationRegularModels.conditionalMomentPathwiseLocalOveridentification_iff` |
| Conditional-moment orthogonal-score local drift | `Econometrics.OveridentificationRegularModels.conditionalMomentOrthogonalScoreLocalDrift` |
| Pathwise-overidentified conditional moments yield positive score drift | `Econometrics.OveridentificationRegularModels.conditionalMomentPathwiseOveridentified_yieldsPositiveScoreDrift` |
| Constructor-backed conditional-mean associated family | `Econometrics.OveridentificationRegularModels.conditionalMeanAssociatedFamily` |
| Constructor-backed scalar-rescaled associated family | `Econometrics.OveridentificationRegularModels.quantileAssociatedFamily` |
| Conditional mean example in pathwise form | `Econometrics.OveridentificationRegularModels.conditionalMeanExample_pathwiseLocallyJustIdentified` |
| Conditional mean example with tangent-top constructor | `Econometrics.OveridentificationRegularModels.conditionalMeanExample_pathwiseLocallyJustIdentified_of_tangent_eq_top` |
| Quantile-style example in pathwise form | `Econometrics.OveridentificationRegularModels.quantileExample_pathwiseLocallyJustIdentified` |
| Quantile-style example with tangent-top constructor | `Econometrics.OveridentificationRegularModels.quantileExample_pathwiseLocallyJustIdentified_of_tangent_eq_top` |
| Partially linear restriction in pathwise form | `Econometrics.OveridentificationRegularModels.partiallyLinearExample_pathwiseLocallyOveridentified` |

## Build Guard

`FormalProofs.Econometrics.Overidentification.CoverageChecklist` uses `#check` on the exports above.
If one of these names disappears from the build graph, the checklist fails.
