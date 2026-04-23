import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.OptimizationPerturbation

/-!
# FormalProofs/OPT/CoverageNormalizedObjective.lean

Coverage-normalized tree objectives for budgeted document supervision.

This module isolates the current bug in the Markov tree trainer:

* the legacy/current document-level objective divides the selected root-loss sum
  by the full batch document count, which introduces a hidden multiplicative
  coverage factor;
* the corrected objective divides by the number of supervised documents, so
  root-label coverage changes variance but not the intended root-vs-local tradeoff;
* under constant inclusion probability, the Horvitz-Thompson document-mean
  estimator is unbiased for the full population document mean.

The file is intentionally finite and elementary: documents live in a finite type,
the supervised subset is a `Finset`, and the stochastic results are stated for a
PMF over subsets with constant marginal inclusion probability.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-- Explicit root-vs-local weight bundle for the tree objective. The intended
tradeoff lives entirely in these weights, not in supervision coverage. -/
structure CoverageNormalizedTreeObjectiveWeights where
  rootWeight : ℝ
  c1Weight : ℝ
  c2Weight : ℝ
  c3Weight : ℝ

section Deterministic

variable {Doc : Type*} [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- Full-population mean of a document-level loss. -/
def documentMean (loss : Doc → ℝ) : ℝ :=
  (∑ i, loss i) / (Fintype.card Doc : ℝ)

/-- Mean of a document-level loss over the supervised subset. Empty subsets map
to `0`; the theorems below use `selected.Nonempty` when normalization matters. -/
def selectedDocumentMean (selected : Finset Doc) (loss : Doc → ℝ) : ℝ :=
  if h : selected.card = 0 then 0 else selected.sum loss / (selected.card : ℝ)

/-- Realized document-supervision coverage rate. -/
def coverageRate (selected : Finset Doc) : ℝ :=
  (selected.card : ℝ) / (Fintype.card Doc : ℝ)

/-- Dense local-law objective. These terms are already normalized at the document
level and should not change when root supervision coverage changes. -/
def denseLocalObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.c1Weight * documentMean c1Loss
    + weights.c2Weight * documentMean c2Loss
    + weights.c3Weight * documentMean c3Loss

/-- Current buggy objective: the supervised root-loss sum is divided by the full
document count, which hides a multiplicative coverage factor. -/
def currentCoverageScaledTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.rootWeight * (selected.sum rootLoss / (Fintype.card Doc : ℝ))
    + denseLocalObjective weights c1Loss c2Loss c3Loss

/-- Corrected objective: the document/root term is normalized by the number of
supervised documents, so coverage changes only the variance of the selected mean. -/
def correctedCoverageNormalizedTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.rootWeight * selectedDocumentMean selected rootLoss
    + denseLocalObjective weights c1Loss c2Loss c3Loss

/-- Full-supervision objective. -/
def fullSupervisionTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) : ℝ :=
  weights.rootWeight * documentMean rootLoss
    + denseLocalObjective weights c1Loss c2Loss c3Loss

/-- Horvitz-Thompson document-mean estimator under a constant inclusion
probability `coverage`. -/
def constantInclusionHTRootMeanOfProb
    (coverage : ℝ) (selected : Finset Doc) (rootLoss : Doc → ℝ) : ℝ :=
  (∑ i, if i ∈ selected then rootLoss i / coverage else 0) / (Fintype.card Doc : ℝ)

/-- HT document-mean estimator where the inclusion probability is instantiated
at the realized coverage rate. Under fixed-size sampling, this agrees exactly
with the selected-subset mean. -/
def constantInclusionHTRootMean
    (selected : Finset Doc) (rootLoss : Doc → ℝ) : ℝ :=
  constantInclusionHTRootMeanOfProb (coverageRate selected) selected rootLoss

lemma selectedDocumentMean_eq_sum_div_card
    (selected : Finset Doc) (loss : Doc → ℝ) (hsel : selected.Nonempty) :
    selectedDocumentMean selected loss = selected.sum loss / (selected.card : ℝ) := by
  have hs : selected.card ≠ 0 := Finset.card_ne_zero.mpr hsel
  simp [selectedDocumentMean, hs]

lemma documentMean_univ_eq_selectedDocumentMean
    (loss : Doc → ℝ) :
    documentMean loss = selectedDocumentMean (Finset.univ : Finset Doc) loss := by
  have hs : (Finset.univ : Finset Doc).Nonempty := Finset.univ_nonempty
  have hcard :
      ((Finset.univ : Finset Doc).card : ℝ) = (Fintype.card Doc : ℝ) := by
    simp
  rw [selectedDocumentMean_eq_sum_div_card _ _ hs]
  simp [documentMean, hcard]

/-- The current objective contains a hidden multiplicative coverage factor on the
root/document term. -/
theorem currentCoverageScaledTreeObjective_eq_coverageRate_mul_selectedRootMean
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ)
    (hsel : selected.Nonempty) :
    currentCoverageScaledTreeObjective weights selected rootLoss c1Loss c2Loss c3Loss
      = coverageRate selected * weights.rootWeight * selectedDocumentMean selected rootLoss
          + denseLocalObjective weights c1Loss c2Loss c3Loss := by
  have hs_nat : selected.card ≠ 0 := Finset.card_ne_zero.mpr hsel
  have hs : (selected.card : ℝ) ≠ 0 := by
    exact_mod_cast hs_nat
  have hdoc : (Fintype.card Doc : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  rw [selectedDocumentMean_eq_sum_div_card _ _ hsel]
  unfold currentCoverageScaledTreeObjective coverageRate
  field_simp [hs, hdoc]

/-- The corrected objective keeps the root/document term at the supervised-subset
mean, removing the hidden coverage multiplier. -/
theorem correctedCoverageNormalizedTreeObjective_eq_rootWeight_mul_selectedRootMean
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) :
    correctedCoverageNormalizedTreeObjective weights selected rootLoss c1Loss c2Loss c3Loss
      = weights.rootWeight * selectedDocumentMean selected rootLoss
          + denseLocalObjective weights c1Loss c2Loss c3Loss := by
  simp [correctedCoverageNormalizedTreeObjective]

/-- At full coverage, the corrected objective coincides with the full-supervision
objective. -/
theorem correctedCoverageNormalizedTreeObjective_eq_fullSupervision_at_fullCoverage
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) :
    correctedCoverageNormalizedTreeObjective weights (Finset.univ : Finset Doc)
        rootLoss c1Loss c2Loss c3Loss
      = fullSupervisionTreeObjective weights rootLoss c1Loss c2Loss c3Loss := by
  rw [correctedCoverageNormalizedTreeObjective, fullSupervisionTreeObjective,
    ← documentMean_univ_eq_selectedDocumentMean (loss := rootLoss)]

/-- With constant inclusion probability set equal to the realized coverage rate,
the HT document-mean estimator collapses to the selected-subset mean. -/
theorem constantInclusionHTRootMean_eq_selectedDocumentMean
    (selected : Finset Doc) (rootLoss : Doc → ℝ) (hsel : selected.Nonempty) :
    constantInclusionHTRootMean selected rootLoss = selectedDocumentMean selected rootLoss := by
  have hs_nat : selected.card ≠ 0 := Finset.card_ne_zero.mpr hsel
  have hs : (selected.card : ℝ) ≠ 0 := by
    exact_mod_cast hs_nat
  have hdoc : (Fintype.card Doc : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  rw [selectedDocumentMean_eq_sum_div_card _ _ hsel]
  unfold constantInclusionHTRootMean constantInclusionHTRootMeanOfProb coverageRate
  rw [Finset.sum_ite_mem]
  simp
  field_simp [hs, hdoc]
  calc
    (selected.card : ℝ) * ∑ i ∈ selected, rootLoss i * (Fintype.card Doc : ℝ) / (selected.card : ℝ)
      = ∑ i ∈ selected, (selected.card : ℝ) * (rootLoss i * (Fintype.card Doc : ℝ) / (selected.card : ℝ)) := by
          simpa using
            (Finset.mul_sum selected
              (fun i => rootLoss i * (Fintype.card Doc : ℝ) / (selected.card : ℝ))
              (a := (selected.card : ℝ)))
    _ = ∑ i ∈ selected, (Fintype.card Doc : ℝ) * rootLoss i := by
          apply Finset.sum_congr rfl
          intro i hi
          field_simp [hs, hdoc]
    _ = (Fintype.card Doc : ℝ) * selected.sum rootLoss := by
          simpa [mul_comm, mul_left_comm, mul_assoc] using
            (Finset.mul_sum selected rootLoss (a := (Fintype.card Doc : ℝ))).symm

/-- Pointwise slack decomposition: the corrected objective differs from the
full-supervision objective only through the selected-vs-population root mean. -/
theorem correctedCoverageNormalizedTreeObjective_sub_fullSupervisionTreeObjective
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (selected : Finset Doc)
    (rootLoss c1Loss c2Loss c3Loss : Doc → ℝ) :
    correctedCoverageNormalizedTreeObjective weights selected rootLoss c1Loss c2Loss c3Loss
      - fullSupervisionTreeObjective weights rootLoss c1Loss c2Loss c3Loss
      = weights.rootWeight * (selectedDocumentMean selected rootLoss - documentMean rootLoss) := by
  unfold correctedCoverageNormalizedTreeObjective fullSupervisionTreeObjective
    denseLocalObjective
  ring_nf

end Deterministic

section Stochastic

variable {Doc Θ : Type*} [Fintype Doc] [DecidableEq Doc] [Nonempty Doc]

/-- Finite expectation over a PMF on a finite type. -/
def finiteExpectation {α : Type*} [Fintype α] (μ : PMF α) (f : α → ℝ) : ℝ :=
  ∑ a, (μ a).toReal * f a

lemma finiteExpectation_const {α : Type*} [Fintype α] (μ : PMF α) (c : ℝ) :
    finiteExpectation μ (fun _ : α => c) = c := by
  unfold finiteExpectation
  calc
    ∑ x, (μ x).toReal * c = (∑ x, (μ x).toReal) * c := by
      simpa using
        (Finset.sum_mul (Finset.univ : Finset α) (fun x => (μ x).toReal) c).symm
    _ = c := by
      have hmass : ∑ x, (μ x).toReal = (1 : ℝ) := by
        simpa [tsum_fintype] using (PMF.toReal_tsum_coe μ)
      rw [hmass]
      ring

lemma finiteExpectation_add {α : Type*} [Fintype α] (μ : PMF α) (f g : α → ℝ) :
    finiteExpectation μ (fun x => f x + g x) = finiteExpectation μ f + finiteExpectation μ g := by
  unfold finiteExpectation
  simp_rw [mul_add]
  rw [Finset.sum_add_distrib]

lemma finiteExpectation_mul_left {α : Type*} [Fintype α] (μ : PMF α) (a : ℝ) (f : α → ℝ) :
    finiteExpectation μ (fun x => a * f x) = a * finiteExpectation μ f := by
  unfold finiteExpectation
  calc
    ∑ x, (μ x).toReal * (a * f x)
      = ∑ x, a * ((μ x).toReal * f x) := by
          apply Finset.sum_congr rfl
          intro x hx
          ring
    _ = a * ∑ x, (μ x).toReal * f x := by
          simpa using
            (Finset.mul_sum (Finset.univ : Finset α)
              (fun x => (μ x).toReal * f x)
              (a := a)).symm

/-- Expected corrected objective using a constant-inclusion-probability HT root
term. This is the stochastic version of the corrected objective. -/
def expectedCorrectedCoverageNormalizedTreeObjective
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) : Θ → ℝ :=
  fun θ =>
    finiteExpectation μ (fun selected =>
      weights.rootWeight * constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ)
        + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ))

/-- Full-supervision objective as a function of the parameter `θ`. -/
def fullSupervisionTreeObjectiveFn
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ) : Θ → ℝ :=
  fun θ =>
    fullSupervisionTreeObjective weights (rootLoss θ) (c1Loss θ) (c2Loss θ) (c3Loss θ)

/-- If each document has constant marginal inclusion probability `coverage`, then
the HT document-mean estimator is unbiased for the full population document mean. -/
theorem finiteExpectation_constantInclusionHTRootMean_eq_documentMean
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (rootLoss : Doc → ℝ)
    (hcoverage : coverage ≠ 0)
    (hmarg :
      ∀ i : Doc, finiteExpectation μ (fun selected => if i ∈ selected then (1 : ℝ) else 0) = coverage) :
    finiteExpectation μ (fun selected => constantInclusionHTRootMeanOfProb coverage selected rootLoss)
      = documentMean rootLoss := by
  classical
  let n : ℝ := Fintype.card Doc
  have hdoc0 : (Fintype.card Doc : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  have hdoc : n ≠ 0 := by
    simpa [n] using hdoc0
  have hmarg' :
      ∀ i : Doc, ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0) = coverage := by
    intro i
    simpa [finiteExpectation] using hmarg i
  unfold finiteExpectation constantInclusionHTRootMeanOfProb documentMean
  calc
    ∑ selected, (μ selected).toReal *
        ((∑ i, if i ∈ selected then rootLoss i / coverage else 0) / n)
      = ((∑ selected, (μ selected).toReal *
            (∑ i, if i ∈ selected then rootLoss i / coverage else 0)) / n) := by
          rw [div_eq_mul_inv]
          simpa [mul_assoc] using
            (Finset.sum_mul (Finset.univ : Finset (Finset Doc))
              (fun selected =>
                (μ selected).toReal * (∑ i, if i ∈ selected then rootLoss i / coverage else 0))
              (n⁻¹)).symm
    _ = ((∑ selected, ∑ i, (μ selected).toReal *
            (if i ∈ selected then rootLoss i / coverage else 0)) / n) := by
          congr 1
          apply Finset.sum_congr rfl
          intro selected hselected
          simpa using
            (Finset.mul_sum (Finset.univ : Finset Doc)
              (fun i => if i ∈ selected then rootLoss i / coverage else 0)
              (a := (μ selected).toReal))
    _ = ((∑ i, ∑ selected, (μ selected).toReal *
            (if i ∈ selected then rootLoss i / coverage else 0)) / n) := by
          congr 1
          simpa using
            (Finset.sum_comm
              (s := (Finset.univ : Finset (Finset Doc)))
              (t := (Finset.univ : Finset Doc))
              (f := fun selected i =>
                (μ selected).toReal *
                  (if i ∈ selected then rootLoss i / coverage else 0)))
    _ = ((∑ i, (rootLoss i / coverage) *
            ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0)) / n) := by
          congr 1
          apply Finset.sum_congr rfl
          intro i hi
          have hfactor :
              ∑ selected, (μ selected).toReal * (if i ∈ selected then rootLoss i / coverage else 0)
                = (rootLoss i / coverage) *
                    ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0) := by
              calc
                ∑ selected, (μ selected).toReal * (if i ∈ selected then rootLoss i / coverage else 0)
                  = ∑ selected, (rootLoss i / coverage) *
                      ((μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0)) := by
                        apply Finset.sum_congr rfl
                        intro selected hselected
                        by_cases hi' : i ∈ selected
                        · simp [hi', mul_assoc, mul_left_comm, mul_comm]
                        · simp [hi']
                _ = (rootLoss i / coverage) *
                      ∑ selected, (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0) := by
                        simpa using
                          (Finset.mul_sum (Finset.univ : Finset (Finset Doc))
                            (fun selected =>
                              (μ selected).toReal * (if i ∈ selected then (1 : ℝ) else 0))
                            (a := (rootLoss i / coverage))).symm
          exact hfactor
    _ = ((∑ i, rootLoss i) / n) := by
          congr 1
          apply Finset.sum_congr rfl
          intro i hi
          rw [hmarg' i]
          field_simp [hcoverage]
    _ = documentMean rootLoss := by
          simp [documentMean, n]

/-- The expected corrected objective matches the full-supervision objective when
the document-supervision design has constant inclusion probability. -/
theorem finiteExpectation_correctedCoverageNormalizedTreeObjective_eq_fullSupervision
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ)
    (hcoverage : coverage ≠ 0)
    (hmarg :
      ∀ i : Doc, finiteExpectation μ (fun selected => if i ∈ selected then (1 : ℝ) else 0) = coverage) :
    expectedCorrectedCoverageNormalizedTreeObjective μ coverage weights rootLoss c1Loss c2Loss c3Loss
      = fullSupervisionTreeObjectiveFn weights rootLoss c1Loss c2Loss c3Loss := by
  funext θ
  have hroot :=
    finiteExpectation_constantInclusionHTRootMean_eq_documentMean
      (μ := μ) (coverage := coverage) (rootLoss := rootLoss θ) hcoverage hmarg
  unfold expectedCorrectedCoverageNormalizedTreeObjective fullSupervisionTreeObjectiveFn
  calc
    finiteExpectation μ (fun selected =>
      weights.rootWeight * constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ)
        + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ))
      = finiteExpectation μ (fun selected =>
          weights.rootWeight * constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ))
        + finiteExpectation μ (fun _ =>
            denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ)) := by
            rw [finiteExpectation_add]
    _ = weights.rootWeight *
          finiteExpectation μ
            (fun selected => constantInclusionHTRootMeanOfProb coverage selected (rootLoss θ))
        + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ) := by
            rw [finiteExpectation_mul_left, finiteExpectation_const]
    _ = weights.rootWeight * documentMean (rootLoss θ)
          + denseLocalObjective weights (c1Loss θ) (c2Loss θ) (c3Loss θ) := by
            rw [hroot]
    _ = fullSupervisionTreeObjective weights (rootLoss θ) (c1Loss θ) (c2Loss θ) (c3Loss θ) := by
            simp [fullSupervisionTreeObjective]

/-- Generic same-argmin lemma for pointwise-equal objectives. -/
theorem paramArgmin_eq_of_pointwise_loss_eq
    {Θ : Type*}
    (loss₁ loss₂ : Θ → ℝ)
    (hEq : ∀ θ, loss₁ θ = loss₂ θ) :
    ParamArgmin loss₁ = ParamArgmin loss₂ := by
  ext θ
  simp [ParamArgmin, hEq]

/-- The corrected expected objective has the same parameter argmin set as the
full-supervision objective. Coverage changes only the sampling noise, not the
population objective being optimized. -/
theorem coverageNormalized_expectedObjective_same_paramArgmin
    (μ : PMF (Finset Doc))
    (coverage : ℝ)
    (weights : CoverageNormalizedTreeObjectiveWeights)
    (rootLoss c1Loss c2Loss c3Loss : Θ → Doc → ℝ)
    (hcoverage : coverage ≠ 0)
    (hmarg :
      ∀ i : Doc, finiteExpectation μ (fun selected => if i ∈ selected then (1 : ℝ) else 0) = coverage) :
    ParamArgmin
        (expectedCorrectedCoverageNormalizedTreeObjective μ coverage weights
          rootLoss c1Loss c2Loss c3Loss)
      = ParamArgmin (fullSupervisionTreeObjectiveFn weights rootLoss c1Loss c2Loss c3Loss) := by
  apply paramArgmin_eq_of_pointwise_loss_eq
  intro θ
  have hEq := congrArg (fun f => f θ)
    (finiteExpectation_correctedCoverageNormalizedTreeObjective_eq_fullSupervision
      (μ := μ) (coverage := coverage) (weights := weights)
      (rootLoss := rootLoss) (c1Loss := c1Loss) (c2Loss := c2Loss) (c3Loss := c3Loss)
      hcoverage hmarg)
  simpa using hEq

end Stochastic

end FormalProofs.OPT
