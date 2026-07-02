import Mathlib

/-!
# FormalProofs/DSL/CoreDefinitions.lean

## Paper Reference: Egami, Hinck, Stewart, Wei - "Design-based Supervised Learning"

This file formalizes the core definitions for Design-based Supervised Learning (DSL):
- Document types and variable decomposition (D^obs, D^mis)
- Sampling indicators and probabilities
- Prediction functions
- Annotated documents with partial observations

### Key Mathematical Objects (from Paper)

| Paper Notation | Lean Definition | Description |
|----------------|-----------------|-------------|
| D = (D^obs, D^mis) | `VariableDecomposition` | Observed + missing variables |
| R ∈ {0,1} | `SamplingIndicator` | Whether document is expert-coded |
| π_i | `SamplingProbability` | P(R_i = 1 | D^obs, Q) |
| D̂^mis | `Prediction` | Predicted missing variable |
| Q | Document content | Text content for annotation |

### Design Principle

The DSL framework addresses the fundamental problem that prediction errors in
automated text annotation are **non-random** - they correlate with variables
in the downstream analysis. Even with 90%+ accuracy, ignoring these errors
leads to substantial bias and invalid confidence intervals.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Nat
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## Basic Types
-/

/-- Document identifier type -/
def DocId := ℕ
deriving DecidableEq, Inhabited

/-- A document with identifier and content -/
structure Document (Content : Type*) where
  id : DocId
  content : Content
deriving Inhabited

/-- Variable decomposition: observed and missing components.
    D^obs: Always observed (structured data, metadata)
    D^mis: Requires text annotation (labels from text) -/
structure VariableDecomposition (Observed Missing : Type*) where
  observed : Observed   -- D^obs - always available
  missing : Missing     -- D^mis - requires annotation (expert or predicted)

/-- Sampling indicator: whether a document was selected for expert coding -/
abbrev SamplingIndicator := Bool

/-- Convert sampling indicator to real for calculations -/
def SamplingIndicator.toReal (R : SamplingIndicator) : ℝ :=
  if R then 1 else 0

/-!
## Predictions and Annotations
-/

/-- Ground-truth label source.
`human` = expert annotation.
`dataset` = trusted pre-existing label in the source data. -/
inductive TruthLabelSource where
  | human
  | dataset
deriving DecidableEq, Repr, Inhabited

/-- Approximate-label source used for high-throughput scoring.
This captures the teacher/proxy split and embedding-based proxies. -/
inductive ApproxLabelSource where
  | teacher_lm
  | proxy_lm
  | embedding_model
deriving DecidableEq, Repr, Inhabited

/-- Label payload that can carry both truth labels and approximations.
Truth labels may come from either humans or trusted dataset labels. -/
structure LabelObservation (Missing : Type*) where
  truth_label : Option Missing
  truth_source : Option TruthLabelSource
  approx_label : Option Missing
  approx_source : Option ApproxLabelSource
  h_truth_source :
    truth_label.isSome = truth_source.isSome

namespace LabelObservation

/-- Whether a truth label is available. -/
def hasTruthLabel {Missing : Type*} (l : LabelObservation Missing) : Bool :=
  l.truth_label.isSome

/-- Whether the truth label came from human annotation. -/
def isHumanTruth {Missing : Type*} (l : LabelObservation Missing) : Bool :=
  decide (l.truth_source = some TruthLabelSource.human)

/-- Whether the truth label came from a trusted dataset label. -/
def isDatasetTruth {Missing : Type*} (l : LabelObservation Missing) : Bool :=
  decide (l.truth_source = some TruthLabelSource.dataset)

lemma hasTruthLabel_iff_source {Missing : Type*} (l : LabelObservation Missing) :
    l.hasTruthLabel = true ↔ l.truth_source.isSome = true := by
  unfold hasTruthLabel
  constructor
  · intro h
    have h_some : l.truth_label.isSome = true := by
      simpa using h
    rw [l.h_truth_source] at h_some
    exact h_some
  · intro h
    have h_some : l.truth_label.isSome = true := by
      rw [l.h_truth_source]
      exact h
    simpa using h_some

end LabelObservation

/-- A prediction for a missing variable value -/
def Prediction (Missing : Type*) := Missing

/-- Annotated document with all observation states.
    This represents a document that may or may not have been expert-coded. -/
structure AnnotatedDocument (Observed Missing Content : Type*) where
  doc_id : DocId
  d_obs : Observed                    -- D^obs: always observed
  content : Content                   -- Q: document content / features
  d_mis_true : Option Missing         -- D^mis: expert annotation (None if R=0)
  d_mis_pred : Missing                -- D̂^mis: ML/LLM prediction (always available)
  sampled : SamplingIndicator         -- R: whether expert-coded
  h_sampled : sampled = true → d_mis_true.isSome  -- Consistency condition

/-!
## Oracle Access (Expert Coding)
-/

/-- Oracle access assumption: when a document is sampled, the expert label
    matches the oracle label computed from the document content. -/
structure OracleAccess (Observed Missing Content : Type*) where
  oracle : Content → Missing
  agrees_on_sampled :
    ∀ doc : AnnotatedDocument Observed Missing Content,
      doc.sampled = true → doc.d_mis_true = some (oracle doc.content)

/-- Population of N documents -/
structure Population (Observed Missing Content : Type*) where
  documents : List (AnnotatedDocument Observed Missing Content)
  n_total : ℕ                         -- N: total population size
  n_labeled : ℕ                       -- n: number with expert labels
  h_size : documents.length = n_total
  h_labeled : (documents.filter (·.sampled)).length = n_labeled

/-!
## Outcome Variables
-/

/-- An outcome variable that may depend on missing variables.
    In the paper's notation, this is Y (the response variable). -/
structure OutcomeSpec (Observed Missing : Type*) where
  /-- True outcome: depends on both observed and missing -/
  true_outcome : Observed → Missing → ℝ
  /-- Predicted outcome: depends on observed and predicted missing -/
  predicted_outcome : Observed → Missing → ℝ
  /-- For simple cases where predicted and true use same function -/
  h_same_form : predicted_outcome = true_outcome := by rfl

/-- Compute the true outcome Y for an annotated document (requires R=1) -/
def OutcomeSpec.Y {Obs Mis Con : Type*} (spec : OutcomeSpec Obs Mis)
    (doc : AnnotatedDocument Obs Mis Con) (h_sampled : doc.sampled = true) : ℝ :=
  match doc.d_mis_true with
  | some d_mis => spec.true_outcome doc.d_obs d_mis
  | none => 0  -- Should not happen if h_sampled consistency holds

/-- Compute the predicted outcome Ŷ for any document -/
def OutcomeSpec.Y_hat {Obs Mis Con : Type*} (spec : OutcomeSpec Obs Mis)
    (doc : AnnotatedDocument Obs Mis Con) : ℝ :=
  spec.predicted_outcome doc.d_obs doc.d_mis_pred

lemma OutcomeSpec.Y_eq_oracle {Obs Mis Con : Type*} (spec : OutcomeSpec Obs Mis)
    (oracle : OracleAccess Obs Mis Con)
    (doc : AnnotatedDocument Obs Mis Con) (h_sampled : doc.sampled = true) :
    spec.Y doc h_sampled = spec.true_outcome doc.d_obs (oracle.oracle doc.content) := by
  have h_oracle := oracle.agrees_on_sampled doc h_sampled
  simp [OutcomeSpec.Y, h_oracle]

/-!
## Covariates
-/

/-- Covariate vector type for regression.
    X represents the covariates in the analysis. -/
structure CovariateSpec (Observed Missing : Type*) (d : ℕ) where
  /-- Extract covariates from observed and missing variables -/
  extract : Observed → Missing → Fin d → ℝ
  /-- Covariates that only depend on observed (for design-based sampling) -/
  extract_obs_only : Option (Observed → Fin d → ℝ) := none

/-- Covariate vector as a function Fin d → ℝ -/
def CovariateSpec.X {Obs Mis : Type*} {d : ℕ} (spec : CovariateSpec Obs Mis d)
    (d_obs : Obs) (d_mis : Mis) : Fin d → ℝ :=
  spec.extract d_obs d_mis

/-!
## Inner Product and Basic Operations
-/

/-- Inner product of two vectors.

mathlib bridge: this IS mathlib's `dotProduct` (notation `⬝ᵥ`,
`Mathlib/Data/Matrix/Mul.lean`), kept under the DSL-facing name `innerProduct`. -/
abbrev innerProduct {d : ℕ} (x : Fin d → ℝ) (β : Fin d → ℝ) : ℝ :=
  dotProduct x β

/-- `innerProduct` is definitionally mathlib's `dotProduct`. -/
theorem innerProduct_eq_dotProduct {d : ℕ} (x β : Fin d → ℝ) :
    innerProduct x β = dotProduct x β := rfl

/-- Standard logistic (sigmoid) function.

mathlib bridge: this IS mathlib's `Real.sigmoid`
(`Mathlib/Analysis/SpecialFunctions/Sigmoid.lean`), kept under the DSL-facing
name `expit`. -/
abbrev expit := Real.sigmoid

/-- `expit` is definitionally mathlib's `Real.sigmoid`. -/
theorem expit_eq_sigmoid : expit = Real.sigmoid := rfl

/-- Logit function (inverse of expit; mathlib has no `logit`, so this stays custom) -/
def logit (p : ℝ) : ℝ := Real.log (p / (1 - p))

/-!
## Helper Lemmas
-/

lemma expit_range (x : ℝ) : 0 < expit x ∧ expit x < 1 :=
  ⟨Real.sigmoid_pos x, Real.sigmoid_lt_one x⟩

lemma innerProduct_comm {d : ℕ} (x β : Fin d → ℝ) :
    innerProduct x β = innerProduct β x :=
  dotProduct_comm x β

end DSL

end
