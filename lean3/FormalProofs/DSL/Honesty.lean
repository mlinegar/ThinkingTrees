import Mathlib
import Mathlib.Probability.IdentDistrib
import Mathlib.Probability.Independence.Basic

/-!
# FormalProofs/DSL/Honesty.lean

## Sample Splitting / Honesty

This file formalizes **honest sample splitting**: a partition of documents
into training vs evaluation sets. The evaluation estimator is computed only
on the evaluation split, which is the key condition for "honesty" in causal
forests and related estimators.

We keep the formalization lightweight and design-agnostic: the split is a
pure function (document → Bool), and filtering is a list operation. This
lets TreePO treat the learned tree or summary model as fixed when applying
evaluation bounds.
-/

set_option linter.mathlibStandardSet false

open scoped Classical BigOperators NNReal ENNReal
open MeasureTheory ProbabilityTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-- Sample split into train vs eval. `isTrain = true` means training set. -/
structure SampleSplit (Doc : Type*) where
  isTrain : Doc → Bool
  train_nonempty : ∃ d, isTrain d = true
  eval_nonempty : ∃ d, isTrain d = false

/-- K-fold split (generalized honesty). -/
structure KFoldSplit (Doc : Type*) where
  K : ℕ
  fold : Doc → Fin K
  k_pos : 0 < K
  fold_nonempty : ∀ k, ∃ d, fold d = k

/-- Filter samples to the training split. -/
def filterTrain {Doc α : Type*} (split : SampleSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  samples.filter (fun s => split.isTrain (doc s))

/-- Filter samples to the evaluation split. -/
def filterEval {Doc α : Type*} (split : SampleSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  samples.filter (fun s => ! split.isTrain (doc s))

/-- Filter samples to the eval fold k (K-fold). -/
def filterEvalFold {Doc α : Type*} (split : KFoldSplit Doc) (k : Fin split.K)
    (doc : α → Doc) (samples : List α) : List α :=
  samples.filter (fun s => split.fold (doc s) = k)

/-- Filter samples to the training folds (all folds except k). -/
def filterTrainFold {Doc α : Type*} (split : KFoldSplit Doc) (k : Fin split.K)
    (doc : α → Doc) (samples : List α) : List α :=
  samples.filter (fun s => split.fold (doc s) ≠ k)

/-- A fold-specific training procedure is cross-fit honest if fold `k`'s
artifact is a function only of samples outside fold `k`. -/
def KFoldHonestTraining {Doc α β : Type*} (split : KFoldSplit Doc)
    (doc : α → Doc) (train_fn : (k : Fin split.K) → List α → β) : Prop :=
  ∀ k samples, train_fn k samples = train_fn k (filterTrainFold split k doc samples)

/-- A fold-specific evaluation procedure is cross-fit honest if fold `k`'s
evaluation statistic is a function only of samples inside fold `k`. -/
def KFoldHonestEvaluation {Doc α γ : Type*} (split : KFoldSplit Doc)
    (doc : α → Doc) (eval_fn : (k : Fin split.K) → List α → γ) : Prop :=
  ∀ k samples, eval_fn k samples = eval_fn k (filterEvalFold split k doc samples)

/-- An estimator is honest if it depends only on evaluation samples. -/
def HonestEvaluation {Doc α β : Type*} (split : SampleSplit Doc)
    (doc : α → Doc) (eval_fn : List α → β) : Prop :=
  ∀ samples, eval_fn samples = eval_fn (filterEval split doc samples)

/-- A training procedure is honest if it depends only on training samples. -/
def HonestTraining {Doc α β : Type*} (split : SampleSplit Doc)
    (doc : α → Doc) (train_fn : List α → β) : Prop :=
  ∀ samples, train_fn samples = train_fn (filterTrain split doc samples)

/-- Constructive honesty contract for train/eval statistics.
Each statistic is represented by a function of its filtered view. -/
def HonestyContract {Ω Doc α β γ : Type*} [MeasurableSpace Ω]
    (_μ : Measure Ω) [_hμ : IsProbabilityMeasure _μ]
    (split : SampleSplit Doc) (doc : α → Doc)
    (samples : Ω → List α)
    (train_stat : Ω → β) (eval_stat : Ω → γ) : Prop :=
  (∃ train_fn : List α → β, ∀ ω, train_stat ω = train_fn (filterTrain split doc (samples ω))) ∧
  (∃ eval_fn : List α → γ, ∀ ω, eval_stat ω = eval_fn (filterEval split doc (samples ω)))

/-- Backward-compatible name for the constructive honesty contract. -/
abbrev HonestyAxioms {Ω Doc α β γ : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : SampleSplit Doc) (doc : α → Doc)
    (samples : Ω → List α)
    (train_stat : Ω → β) (eval_stat : Ω → γ) : Prop :=
  HonestyContract μ split doc samples train_stat eval_stat

lemma train_only_of_honesty_contract {Ω Doc α β γ : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : SampleSplit Doc) (doc : α → Doc)
    (samples : Ω → List α)
    (train_stat : Ω → β) (eval_stat : Ω → γ)
    (h : HonestyContract μ split doc samples train_stat eval_stat) :
    ∀ ω ω',
      filterTrain split doc (samples ω) = filterTrain split doc (samples ω') →
        train_stat ω = train_stat ω' := by
  intro ω ω' h_eq
  rcases h.1 with ⟨train_fn, h_train⟩
  calc
    train_stat ω = train_fn (filterTrain split doc (samples ω)) := h_train ω
    _ = train_fn (filterTrain split doc (samples ω')) := by simp [h_eq]
    _ = train_stat ω' := (h_train ω').symm

lemma eval_only_of_honesty_contract {Ω Doc α β γ : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : SampleSplit Doc) (doc : α → Doc)
    (samples : Ω → List α)
    (train_stat : Ω → β) (eval_stat : Ω → γ)
    (h : HonestyContract μ split doc samples train_stat eval_stat) :
    ∀ ω ω',
      filterEval split doc (samples ω) = filterEval split doc (samples ω') →
        eval_stat ω = eval_stat ω' := by
  intro ω ω' h_eq
  rcases h.2 with ⟨eval_fn, h_eval⟩
  calc
    eval_stat ω = eval_fn (filterEval split doc (samples ω)) := h_eval ω
    _ = eval_fn (filterEval split doc (samples ω')) := by simp [h_eq]
    _ = eval_stat ω' := (h_eval ω').symm

/-!
## Honest Evaluation: Bound Lifting
-/

lemma honest_eval_event_eq {Ω Doc α γ : Type*} [MeasurableSpace Ω]
    (split : SampleSplit Doc) (doc : α → Doc) (samples : Ω → List α)
    (eval_fn : List α → γ) (eval_stat : Ω → γ)
    (h_def : ∀ ω, eval_stat ω = eval_fn (filterEval split doc (samples ω)))
    (P : γ → Prop) :
    {ω | P (eval_stat ω)} =
      {ω | P (eval_fn (filterEval split doc (samples ω)))} := by
  ext ω
  simp [h_def ω]

lemma honest_eval_bound {Ω Doc α γ : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : SampleSplit Doc) (doc : α → Doc) (samples : Ω → List α)
    (eval_fn : List α → γ) (eval_stat : Ω → γ)
    (h_def : ∀ ω, eval_stat ω = eval_fn (filterEval split doc (samples ω)))
    (P : γ → Prop) (δ : ℝ≥0∞) :
    μ {ω | P (eval_fn (filterEval split doc (samples ω)))} ≤ δ →
      μ {ω | P (eval_stat ω)} ≤ δ := by
  intro h
  have hset :=
    honest_eval_event_eq (split := split) (doc := doc) (samples := samples)
      (eval_fn := eval_fn) (eval_stat := eval_stat) (h_def := h_def) (P := P)
  simpa [hset] using h

/-!
## Top-Level Units And Derived Rows

TreePO pipelines distinguish the top-level statistical unit from sub-document
objects generated inside that unit.  The top-level unit is what receives
train/eval roles and what supports IID/exchangeability claims.  Leaves, nodes,
spans, and local-law rows are dependent derived objects, so their honest split
role is inherited through a parent map.
-/

/-- Paper-facing alias: the split is over top-level cases/units. -/
abbrev TopLevelSplit (Case : Type*) := SampleSplit Case

/-- The paired top-level observation used for IID/exchangeability claims:
the case itself together with its truth target. -/
def topLevelObservation {Ω Case Truth : Type*}
    (X : ℕ → Ω → Case) (truth : Case → Truth) (i : ℕ) : Ω → Case × Truth :=
  fun ω => (X i ω, truth (X i ω))

/-- Top-level IID assumption: only the paired case/truth sequence is IID.
Derived rows inside a case are intentionally not included here. -/
def TopLevelIID {Ω Case Truth : Type*}
    [MeasurableSpace Ω] [MeasurableSpace Case] [MeasurableSpace Truth]
    (μ : Measure Ω) (X : ℕ → Ω → Case) (truth : Case → Truth) : Prop :=
  iIndepFun (fun i => topLevelObservation X truth i) μ ∧
    ∀ i, IdentDistrib (topLevelObservation X truth i)
      (topLevelObservation X truth 0) μ μ

/-- Finite-dimensional exchangeability of top-level paired observations. -/
def TopLevelExchangeable {Ω Case Truth : Type*}
    [MeasurableSpace Ω] [MeasurableSpace Case] [MeasurableSpace Truth]
    (μ : Measure Ω) (X : ℕ → Ω → Case) (truth : Case → Truth) : Prop :=
  ∀ n (σ : Equiv.Perm (Fin n)),
    IdentDistrib
      (fun ω k => topLevelObservation X truth k.val ω)
      (fun ω k => topLevelObservation X truth (σ k).val ω)
      μ μ

/-- Parent map from a derived row/span/node back to its top-level unit. -/
abbrev ParentOf (Case Row : Type*) := Row → Case

/-- Two derived rows are siblings when they come from the same top-level unit. -/
def SameTopLevelUnit {Case Row : Type*} (parent : ParentOf Case Row)
    (r₁ r₂ : Row) : Prop :=
  parent r₁ = parent r₂

/-- A derived row inherits its train/eval role from its top-level unit. -/
def inheritedSplitRole {Case Row : Type*} (split : TopLevelSplit Case)
    (parent : ParentOf Case Row) (row : Row) : Bool :=
  split.isTrain (parent row)

/-- Filter derived rows to top-level training units. -/
def filterDerivedTrain {Case Row : Type*} (split : TopLevelSplit Case)
    (parent : ParentOf Case Row) (rows : List Row) : List Row :=
  filterTrain split parent rows

/-- Filter derived rows to top-level evaluation units. -/
def filterDerivedEval {Case Row : Type*} (split : TopLevelSplit Case)
    (parent : ParentOf Case Row) (rows : List Row) : List Row :=
  filterEval split parent rows

lemma filterDerivedTrain_eq_filterTrain {Case Row : Type*}
    (split : TopLevelSplit Case) (parent : ParentOf Case Row)
    (rows : List Row) :
    filterDerivedTrain split parent rows = filterTrain split parent rows := rfl

lemma filterDerivedEval_eq_filterEval {Case Row : Type*}
    (split : TopLevelSplit Case) (parent : ParentOf Case Row)
    (rows : List Row) :
    filterDerivedEval split parent rows = filterEval split parent rows := rfl

/-- Training honesty for derived rows: the training statistic may depend on
rows, but only through rows whose parent top-level units are in train. -/
def DerivedRowHonestTraining {Case Row β : Type*}
    (split : TopLevelSplit Case) (parent : ParentOf Case Row)
    (train_fn : List Row → β) : Prop :=
  HonestTraining split parent train_fn

/-- Evaluation honesty for derived rows: the estimator may depend on rows, but
only through rows whose parent top-level units are in eval. -/
def DerivedRowHonestEvaluation {Case Row γ : Type*}
    (split : TopLevelSplit Case) (parent : ParentOf Case Row)
    (eval_fn : List Row → γ) : Prop :=
  HonestEvaluation split parent eval_fn

/-- Constructive honesty contract for derived rows under a top-level split. -/
abbrev DerivedRowHonestyContract {Ω Case Row β γ : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (split : TopLevelSplit Case) (parent : ParentOf Case Row)
    (rows : Ω → List Row)
    (train_stat : Ω → β) (eval_stat : Ω → γ) : Prop :=
  HonestyContract μ split parent rows train_stat eval_stat

/-!
## Generic Chunker Objective

The chunker is formalized as an instrumental policy over top-level units.  It
chooses an admissible partition using frozen artifacts such as learned `g`,
learned `f`, oracle views, proxies, and query policies.  The objective below is
generic: concrete instantiations provide the loss, local-law mass, certificate
radius, cost, and boundary regularizer.
-/

/-- A model-aware chunker maps a top-level unit and a frozen artifact bundle to
an admissible partition candidate. -/
abbrev ChunkerPolicy (Case Partition Artifact : Type*) :=
  Case → Artifact → Partition

/-- Weighted objective components for chunker selection. -/
structure ChunkerObjectiveTerms (Case Partition Artifact : Type*) where
  downstreamLoss : Case → Partition → Artifact → ℝ
  localLawResidualMass : Case → Partition → Artifact → ℝ
  certificateRadius : Case → Partition → Artifact → ℝ
  computeOrQueryCost : Case → Partition → Artifact → ℝ
  boundaryRegularization : Case → Partition → Artifact → ℝ
  lambdaLaw : ℝ
  lambdaRadius : ℝ
  lambdaCost : ℝ
  lambdaRegularization : ℝ

namespace ChunkerObjectiveTerms

/-- Total chunker objective value for one top-level unit and frozen artifacts. -/
def value {Case Partition Artifact : Type*}
    (terms : ChunkerObjectiveTerms Case Partition Artifact)
    (x : Case) (partition : Partition) (artifact : Artifact) : ℝ :=
    terms.downstreamLoss x partition artifact
  + terms.lambdaLaw * terms.localLawResidualMass x partition artifact
  + terms.lambdaRadius * terms.certificateRadius x partition artifact
  + terms.lambdaCost * terms.computeOrQueryCost x partition artifact
  + terms.lambdaRegularization * terms.boundaryRegularization x partition artifact

end ChunkerObjectiveTerms

/-- Nonnegativity assumptions for the generic chunker objective. -/
structure ChunkerObjectiveNonnegative {Case Partition Artifact : Type*}
    (terms : ChunkerObjectiveTerms Case Partition Artifact) : Prop where
  downstream_nonneg :
    ∀ x partition artifact, 0 ≤ terms.downstreamLoss x partition artifact
  localLawResidualMass_nonneg :
    ∀ x partition artifact, 0 ≤ terms.localLawResidualMass x partition artifact
  certificateRadius_nonneg :
    ∀ x partition artifact, 0 ≤ terms.certificateRadius x partition artifact
  computeOrQueryCost_nonneg :
    ∀ x partition artifact, 0 ≤ terms.computeOrQueryCost x partition artifact
  boundaryRegularization_nonneg :
    ∀ x partition artifact, 0 ≤ terms.boundaryRegularization x partition artifact
  lambdaLaw_nonneg : 0 ≤ terms.lambdaLaw
  lambdaRadius_nonneg : 0 ≤ terms.lambdaRadius
  lambdaCost_nonneg : 0 ≤ terms.lambdaCost
  lambdaRegularization_nonneg : 0 ≤ terms.lambdaRegularization

theorem chunkerObjectiveValue_nonneg {Case Partition Artifact : Type*}
    (terms : ChunkerObjectiveTerms Case Partition Artifact)
    (h : ChunkerObjectiveNonnegative terms)
    (x : Case) (partition : Partition) (artifact : Artifact) :
    0 ≤ terms.value x partition artifact := by
  have h_law :
      0 ≤ terms.lambdaLaw * terms.localLawResidualMass x partition artifact :=
    mul_nonneg h.lambdaLaw_nonneg (h.localLawResidualMass_nonneg x partition artifact)
  have h_radius :
      0 ≤ terms.lambdaRadius * terms.certificateRadius x partition artifact :=
    mul_nonneg h.lambdaRadius_nonneg (h.certificateRadius_nonneg x partition artifact)
  have h_cost :
      0 ≤ terms.lambdaCost * terms.computeOrQueryCost x partition artifact :=
    mul_nonneg h.lambdaCost_nonneg (h.computeOrQueryCost_nonneg x partition artifact)
  have h_reg :
      0 ≤ terms.lambdaRegularization *
        terms.boundaryRegularization x partition artifact :=
    mul_nonneg h.lambdaRegularization_nonneg
      (h.boundaryRegularization_nonneg x partition artifact)
  unfold ChunkerObjectiveTerms.value
  linarith [h.downstream_nonneg x partition artifact, h_law, h_radius, h_cost, h_reg]

/-- A chunker policy is an exact objective minimizer over the admissible
partitions for every top-level unit and frozen artifact bundle. -/
def IsChunkerObjectiveMinimizer {Case Partition Artifact : Type*}
    (terms : ChunkerObjectiveTerms Case Partition Artifact)
    (admissible : Case → Partition → Prop)
    (policy : ChunkerPolicy Case Partition Artifact) : Prop :=
  ∀ x artifact,
    admissible x (policy x artifact) ∧
      ∀ partition,
        admissible x partition →
          terms.value x (policy x artifact) artifact ≤
            terms.value x partition artifact

/-!
## K-Fold Cross-Fit Aggregation

If each fold's evaluation error is bounded with high probability,
then the average evaluation error is bounded with probability at least
the sum of per-fold failure probabilities (union bound).
-/

lemma kfold_avg_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {K : ℕ} (hK : 0 < K)
    (eval : Fin K → Ω → ℝ) (mean : Fin K → ℝ) (r : Fin K → Ω → ℝ)
    (δ : Fin K → ℝ≥0∞)
    (hδ : ∀ k, μ {ω | |eval k ω - mean k| ≥ r k ω} ≤ δ k) :
    μ {ω | |(∑ k, (eval k ω - mean k)) / (K : ℝ)| ≥ (∑ k, r k ω) / (K : ℝ)} ≤
      ∑' k, δ k := by
  classical
  let s : Fin K → Set Ω := fun k => {ω | |eval k ω - mean k| ≥ r k ω}
  have h_subset :
      {ω | |(∑ k, (eval k ω - mean k)) / (K : ℝ)| ≥ (∑ k, r k ω) / (K : ℝ)} ⊆
        ⋃ k, s k := by
    intro ω hω
    by_contra hnot
    have hlt : ∀ k, |eval k ω - mean k| < r k ω := by
      intro k
      have hnotk : ω ∉ s k := by
        intro hk
        apply hnot
        exact Set.mem_iUnion.mpr ⟨k, hk⟩
      have hge : ¬ |eval k ω - mean k| ≥ r k ω := by
        simpa [s] using hnotk
      exact lt_of_not_ge hge
    have h_nonempty : (Finset.univ : Finset (Fin K)).Nonempty := by
      letI : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
      simp
    have hsum_lt :
        |∑ k, (eval k ω - mean k)| < ∑ k, r k ω := by
      have hsum_le :
          |∑ k, (eval k ω - mean k)| ≤ ∑ k, |eval k ω - mean k| := by
        simpa using
          (Finset.abs_sum_le_sum_abs (s := (Finset.univ : Finset (Fin K)))
            (f := fun k => eval k ω - mean k))
      have hsum_abs_lt :
          ∑ k, |eval k ω - mean k| < ∑ k, r k ω := by
        refine Finset.sum_lt_sum_of_nonempty h_nonempty ?_
        intro k hk
        exact hlt k
      exact lt_of_le_of_lt hsum_le hsum_abs_lt
    have hKpos : 0 < (K : ℝ) := by
      exact Nat.cast_pos.mpr hK
    have havg_lt :
        |(∑ k, (eval k ω - mean k)) / (K : ℝ)| <
          (∑ k, r k ω) / (K : ℝ) := by
      simpa [abs_div, abs_of_pos hKpos] using
        (div_lt_div_of_pos_right hsum_lt hKpos)
    have hω' :
        |(∑ k, (eval k ω - mean k)) / (K : ℝ)| ≥ (∑ k, r k ω) / (K : ℝ) := by
      simpa using hω
    exact (not_lt_of_ge hω' havg_lt)
  have h_bound : μ {ω | |(∑ k, (eval k ω - mean k)) / (K : ℝ)| ≥ (∑ k, r k ω) / (K : ℝ)} ≤
      μ (⋃ k, s k) := by
    exact measure_mono h_subset
  have h_union : μ (⋃ k, s k) ≤ ∑' k, μ (s k) := by
    exact measure_iUnion_le (s := s) (μ := μ)
  have h_sum : (∑' k, μ (s k)) ≤ ∑' k, δ k := by
    exact ENNReal.tsum_le_tsum (fun k => hδ k)
  exact h_bound.trans (h_union.trans h_sum)

/-!
## Adaptive Sampling Validity (Predictability + Exploration Floor)

This section provides a lightweight formal interface for adaptive sampling
policies used in TreePO-style pipelines:
- a predictable/adaptive score `p_adaptive`
- a fixed baseline policy `p_uniform`
- an exploration floor parameter `eps`
- a mixture rule ensuring overlap and bounded weights.
-/

/-- Adaptive policy mixed with a fixed baseline policy. -/
def adaptiveMixtureProb {Ω Doc : Type*}
    [MeasurableSpace Ω]
    (p_adaptive : Ω → Doc → ℝ)
    (p_uniform : Doc → ℝ)
    (eps : ℝ) (ω : Ω) (d : Doc) : ℝ :=
  (1 - eps) * p_adaptive ω d + eps * p_uniform d

/-- Assumption bundle for adaptive sampling with an exploration floor. -/
structure AdaptiveSamplingAssumptions {Ω Doc : Type*} [MeasurableSpace Ω] where
  p_adaptive : Ω → Doc → ℝ
  p_uniform : Doc → ℝ
  eps : ℝ
  eps_pos : 0 < eps
  eps_le_one : eps ≤ 1
  adaptive_nonneg : ∀ ω d, 0 ≤ p_adaptive ω d
  uniform_pos : ∀ d, 0 < p_uniform d
  adaptive_le_one : ∀ ω d, p_adaptive ω d ≤ 1
  uniform_le_one : ∀ d, p_uniform d ≤ 1
  predictable : ∀ d, Measurable (fun ω => p_adaptive ω d)

/-- Backward-compatible alias for adaptive sampling assumptions. -/
abbrev AdaptiveSamplingAxioms {Ω Doc : Type*} [MeasurableSpace Ω] :=
  AdaptiveSamplingAssumptions (Ω := Ω) (Doc := Doc)

namespace AdaptiveSamplingAssumptions

/-- Mixed probability used for logging/importance weighting. -/
def mixedProb {Ω Doc : Type*} [MeasurableSpace Ω]
    (a : AdaptiveSamplingAssumptions (Ω := Ω) (Doc := Doc))
    (ω : Ω) (d : Doc) : ℝ :=
  adaptiveMixtureProb a.p_adaptive a.p_uniform a.eps ω d

/-- Exploration floor guarantee:
`eps * p_uniform(d) ≤ mixedProb(ω,d)`. -/
lemma floor_lower_bound {Ω Doc : Type*} [MeasurableSpace Ω]
    (a : AdaptiveSamplingAssumptions (Ω := Ω) (Doc := Doc))
    (ω : Ω) (d : Doc) :
    a.eps * a.p_uniform d ≤ a.mixedProb ω d := by
  unfold mixedProb adaptiveMixtureProb
  have h_one_minus_nonneg : 0 ≤ 1 - a.eps := by
    linarith [a.eps_le_one]
  have h_adaptive_term_nonneg : 0 ≤ (1 - a.eps) * a.p_adaptive ω d := by
    exact mul_nonneg h_one_minus_nonneg (a.adaptive_nonneg ω d)
  linarith

/-- Mixed probability is strictly positive (overlap preserved). -/
lemma mixedProb_pos {Ω Doc : Type*} [MeasurableSpace Ω]
    (a : AdaptiveSamplingAssumptions (Ω := Ω) (Doc := Doc))
    (ω : Ω) (d : Doc) :
    0 < a.mixedProb ω d := by
  have h_floor := a.floor_lower_bound ω d
  have h_rhs_pos : 0 < a.eps * a.p_uniform d := by
    exact mul_pos a.eps_pos (a.uniform_pos d)
  exact lt_of_lt_of_le h_rhs_pos h_floor

/-- IPW weight for adaptive-mixture sampling. -/
def mixedWeight {Ω Doc : Type*} [MeasurableSpace Ω]
    (a : AdaptiveSamplingAssumptions (Ω := Ω) (Doc := Doc))
    (ω : Ω) (d : Doc) : ℝ :=
  1 / a.mixedProb ω d

/-- Exploration-floor weight bound:
`1 / mixedProb ≤ 1 / (eps * p_uniform)`. -/
lemma mixedWeight_le_inv_floor {Ω Doc : Type*} [MeasurableSpace Ω]
    (a : AdaptiveSamplingAssumptions (Ω := Ω) (Doc := Doc))
    (ω : Ω) (d : Doc) :
    a.mixedWeight ω d ≤ 1 / (a.eps * a.p_uniform d) := by
  unfold mixedWeight
  exact one_div_le_one_div_of_le
    (mul_pos a.eps_pos (a.uniform_pos d))
    (a.floor_lower_bound ω d)

end AdaptiveSamplingAssumptions

/-!
## Three-Layer Honesty (Chunker / Summarizer / Oracle)

Thinking-Trees style pipelines have three adaptive components:
1. chunker policy updates,
2. summarizer updates,
3. oracle/scorer updates.

To avoid leakage, each component gets its own train/eval split over documents.
The final pipeline evaluation is computed only on the intersection of the
three evaluation views.
-/

/-- Three independent honesty splits for chunker, summarizer, and oracle. -/
structure ThreeLayerSplit (Doc : Type*) where
  chunk : SampleSplit Doc
  summarizer : SampleSplit Doc
  oracle : SampleSplit Doc

/-- Chunker-training view. -/
def filterChunkTrain {Doc α : Type*} (splits : ThreeLayerSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  filterTrain splits.chunk doc samples

/-- Summarizer-training view. -/
def filterSummarizerTrain {Doc α : Type*} (splits : ThreeLayerSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  filterTrain splits.summarizer doc samples

/-- Oracle-training view. -/
def filterOracleTrain {Doc α : Type*} (splits : ThreeLayerSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  filterTrain splits.oracle doc samples

/-- Oracle-evaluation view. -/
def filterOracleEval {Doc α : Type*} (splits : ThreeLayerSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  filterEval splits.oracle doc samples

/-- Joint evaluation view: docs in eval subsets for all three components. -/
def filterThreeEval {Doc α : Type*} (splits : ThreeLayerSplit Doc)
    (doc : α → Doc) (samples : List α) : List α :=
  filterEval splits.oracle doc
    (filterEval splits.summarizer doc
      (filterEval splits.chunk doc samples))

/-- Component-level honesty bundle for three training procedures. -/
def ThreeLayerHonestTraining {Doc α βc βs βo : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_chunk : List α → βc)
    (train_summarizer : List α → βs)
    (train_oracle : List α → βo) : Prop :=
  HonestTraining splits.chunk doc train_chunk ∧
  HonestTraining splits.summarizer doc train_summarizer ∧
  HonestTraining splits.oracle doc train_oracle

/-- Three-layer honest evaluation: depends only on the joint eval view. -/
def ThreeLayerHonestEvaluation {Doc α γ : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (eval_fn : List α → γ) : Prop :=
  ∀ samples, eval_fn samples = eval_fn (filterThreeEval splits doc samples)

/-- Parallel-safe training means each component is honest on its own split.
This captures when chunker/summarizer/oracle training can be run independently
over disjoint filtered views (possibly in parallel execution). -/
def ParallelSafeTraining {Doc α βc βs βo : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_chunk : List α → βc)
    (train_summarizer : List α → βs)
    (train_oracle : List α → βo) : Prop :=
  ThreeLayerHonestTraining splits doc train_chunk train_summarizer train_oracle

/-- Dual-oracle honest training (teacher + proxy).
Both oracle models must be trained only on the oracle-training split. -/
def DualOracleHonestTraining {Doc α βt βp : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_teacher : List α → βt)
    (train_proxy : List α → βp) : Prop :=
  HonestTraining splits.oracle doc train_teacher ∧
  HonestTraining splits.oracle doc train_proxy

/-- Dual-oracle honest evaluation (teacher + proxy diagnostics).
Both evaluation statistics must depend only on the oracle-eval split. -/
def DualOracleHonestEvaluation {Doc α γt γp : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (eval_teacher : List α → γt)
    (eval_proxy : List α → γp) : Prop :=
  HonestEvaluation splits.oracle doc eval_teacher ∧
  HonestEvaluation splits.oracle doc eval_proxy

/-- Single-oracle honest training/evaluation contract.
The same oracle model class is used in two roles:
- `oracle_online`: training/adaptation role on oracle-train docs
- `oracle_eval`: frozen/OOF evaluation role on oracle-eval docs -/
def SingleOracleTwoViewHonesty {Doc α βo γo : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (oracle_online : List α → βo)
    (oracle_eval : List α → γo) : Prop :=
  HonestTraining splits.oracle doc oracle_online ∧
  HonestEvaluation splits.oracle doc oracle_eval

/-- Chunker-training honesty specialized to the three-layer split. -/
def ChunkerHonestTraining {Doc α βc : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_chunker : List α → βc) : Prop :=
  HonestTraining splits.chunk doc train_chunker

/-- Unified `g`/summarizer-training honesty specialized to the three-layer
split.  This is the same split component called `summarizer` in the generic
three-layer structure. -/
def UnifiedGHonestTraining {Doc α βg : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_g : List α → βg) : Prop :=
  HonestTraining splits.summarizer doc train_g

/-- Frozen-artifact evaluation: once a learned artifact bundle is fixed,
evaluation must depend only on the joint three-layer evaluation view. -/
def FrozenArtifactThreeLayerEvaluation {Doc α Artifact γ : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (artifact : Artifact) (eval_fn : Artifact → List α → γ) : Prop :=
  ThreeLayerHonestEvaluation splits doc (eval_fn artifact)

/-- Full unified learning honesty contract for chunker, learned `g`, and
oracle/readout training, plus frozen-artifact evaluation. -/
def UnifiedLearningHonesty {Doc α βc βg βo Artifact γ : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_chunker : List α → βc)
    (train_g : List α → βg)
    (train_oracle : List α → βo)
    (artifact : Artifact)
    (eval_fn : Artifact → List α → γ) : Prop :=
  ChunkerHonestTraining splits doc train_chunker ∧
  UnifiedGHonestTraining splits doc train_g ∧
  HonestTraining splits.oracle doc train_oracle ∧
  FrozenArtifactThreeLayerEvaluation splits doc artifact eval_fn

lemma dualOracleTraining_of_honest {Doc α βt βp : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_teacher : List α → βt)
    (train_proxy : List α → βp)
    (hTeacher : HonestTraining splits.oracle doc train_teacher)
    (hProxy : HonestTraining splits.oracle doc train_proxy) :
    DualOracleHonestTraining splits doc train_teacher train_proxy := by
  exact ⟨hTeacher, hProxy⟩

lemma dualOracleEvaluation_of_honest {Doc α γt γp : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (eval_teacher : List α → γt)
    (eval_proxy : List α → γp)
    (hTeacher : HonestEvaluation splits.oracle doc eval_teacher)
    (hProxy : HonestEvaluation splits.oracle doc eval_proxy) :
    DualOracleHonestEvaluation splits doc eval_teacher eval_proxy := by
  exact ⟨hTeacher, hProxy⟩

lemma singleOracleTwoView_of_honest {Doc α βo γo : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (oracle_online : List α → βo)
    (oracle_eval : List α → γo)
    (hOnline : HonestTraining splits.oracle doc oracle_online)
    (hEval : HonestEvaluation splits.oracle doc oracle_eval) :
    SingleOracleTwoViewHonesty splits doc oracle_online oracle_eval := by
  exact ⟨hOnline, hEval⟩

lemma parallelSafe_of_honest_components {Doc α βc βs βo : Type*}
    (splits : ThreeLayerSplit Doc) (doc : α → Doc)
    (train_chunk : List α → βc)
    (train_summarizer : List α → βs)
    (train_oracle : List α → βo)
    (hChunk : HonestTraining splits.chunk doc train_chunk)
    (hSum : HonestTraining splits.summarizer doc train_summarizer)
    (hOracle : HonestTraining splits.oracle doc train_oracle) :
    ParallelSafeTraining splits doc train_chunk train_summarizer train_oracle := by
  exact ⟨hChunk, hSum, hOracle⟩

lemma threeLayer_eval_event_eq {Ω Doc α γ : Type*} [MeasurableSpace Ω]
    (splits : ThreeLayerSplit Doc) (doc : α → Doc) (samples : Ω → List α)
    (eval_fn : List α → γ) (eval_stat : Ω → γ)
    (h_def : ∀ ω, eval_stat ω = eval_fn (filterThreeEval splits doc (samples ω)))
    (P : γ → Prop) :
    {ω | P (eval_stat ω)} =
      {ω | P (eval_fn (filterThreeEval splits doc (samples ω)))} := by
  ext ω
  simp [h_def ω]

lemma threeLayer_eval_bound {Ω Doc α γ : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (splits : ThreeLayerSplit Doc) (doc : α → Doc) (samples : Ω → List α)
    (eval_fn : List α → γ) (eval_stat : Ω → γ)
    (h_def : ∀ ω, eval_stat ω = eval_fn (filterThreeEval splits doc (samples ω)))
    (P : γ → Prop) (δ : ℝ≥0∞) :
    μ {ω | P (eval_fn (filterThreeEval splits doc (samples ω)))} ≤ δ →
      μ {ω | P (eval_stat ω)} ≤ δ := by
  intro h
  have hset :=
    threeLayer_eval_event_eq (splits := splits) (doc := doc) (samples := samples)
      (eval_fn := eval_fn) (eval_stat := eval_stat) (h_def := h_def) (P := P)
  simpa [hset] using h

/-!
## Worst-Case Layered Error Envelopes

These lemmas give a direct formalization of a "triple honesty" worst-case view:

- deterministic decomposition: total absolute error is bounded by the sum of
  component envelopes;
- probabilistic decomposition: failure probability of the total error event is
  bounded by the sum of component failure probabilities (union bound).
-/

/-- Deterministic three-component absolute-error envelope. -/
lemma threeLayer_abs_envelope
    (e_chunk e_sum e_oracle b_chunk b_sum b_oracle : ℝ)
    (h_chunk : |e_chunk| ≤ b_chunk)
    (h_sum : |e_sum| ≤ b_sum)
    (h_oracle : |e_oracle| ≤ b_oracle) :
    |e_chunk + e_sum + e_oracle| ≤ b_chunk + b_sum + b_oracle := by
  have h_triangle :
      |e_chunk + e_sum + e_oracle| ≤ |e_chunk| + |e_sum| + |e_oracle| := by
    calc
      |e_chunk + e_sum + e_oracle|
          = |(e_chunk + e_sum) + e_oracle| := by ring
      _ ≤ |e_chunk + e_sum| + |e_oracle| := abs_add_le _ _
      _ ≤ (|e_chunk| + |e_sum|) + |e_oracle| := by
            exact add_le_add (abs_add_le _ _) (le_refl _)
      _ = |e_chunk| + |e_sum| + |e_oracle| := by ring
  calc
    |e_chunk + e_sum + e_oracle|
        ≤ |e_chunk| + |e_sum| + |e_oracle| := h_triangle
    _ ≤ b_chunk + b_sum + b_oracle := by
          linarith

/-- Event containment for the three-component absolute-error envelope. -/
lemma threeLayer_error_event_subset
    {Ω : Type*}
    (e_chunk e_sum e_oracle : Ω → ℝ)
    (r_chunk r_sum r_oracle : Ω → ℝ) :
    {ω | |e_chunk ω + e_sum ω + e_oracle ω| ≥ r_chunk ω + r_sum ω + r_oracle ω} ⊆
      {ω | |e_chunk ω| ≥ r_chunk ω} ∪
        {ω | |e_sum ω| ≥ r_sum ω} ∪
          {ω | |e_oracle ω| ≥ r_oracle ω} := by
  intro ω hω
  by_cases h_chunk : |e_chunk ω| ≥ r_chunk ω
  · exact Or.inl (Or.inl (by simpa using h_chunk))
  · by_cases h_sum : |e_sum ω| ≥ r_sum ω
    · exact Or.inl (Or.inr (by simpa using h_sum))
    · by_cases h_oracle : |e_oracle ω| ≥ r_oracle ω
      · exact Or.inr (by simpa using h_oracle)
      · have h_chunk_lt : |e_chunk ω| < r_chunk ω := lt_of_not_ge h_chunk
        have h_sum_lt : |e_sum ω| < r_sum ω := lt_of_not_ge h_sum
        have h_oracle_lt : |e_oracle ω| < r_oracle ω := lt_of_not_ge h_oracle
        have h_abs_sum_lt :
            |e_chunk ω + e_sum ω + e_oracle ω| <
              r_chunk ω + r_sum ω + r_oracle ω := by
          have h_triangle :
              |e_chunk ω + e_sum ω + e_oracle ω| ≤
                |e_chunk ω| + |e_sum ω| + |e_oracle ω| := by
            calc
              |e_chunk ω + e_sum ω + e_oracle ω|
                  = |(e_chunk ω + e_sum ω) + e_oracle ω| := by ring
              _ ≤ |e_chunk ω + e_sum ω| + |e_oracle ω| := abs_add_le _ _
              _ ≤ (|e_chunk ω| + |e_sum ω|) + |e_oracle ω| := by
                    exact add_le_add (abs_add_le _ _) (le_refl _)
              _ = |e_chunk ω| + |e_sum ω| + |e_oracle ω| := by ring
          have h_rhs_lt :
              |e_chunk ω| + |e_sum ω| + |e_oracle ω| <
                r_chunk ω + r_sum ω + r_oracle ω := by
            have h12 : |e_chunk ω| + |e_sum ω| < r_chunk ω + r_sum ω :=
              add_lt_add h_chunk_lt h_sum_lt
            have h123 :
                (|e_chunk ω| + |e_sum ω|) + |e_oracle ω| <
                  (r_chunk ω + r_sum ω) + r_oracle ω :=
              add_lt_add h12 h_oracle_lt
            simpa [add_assoc] using h123
          exact lt_of_le_of_lt h_triangle h_rhs_lt
        exact False.elim ((not_le_of_gt h_abs_sum_lt) hω)

/-- Probabilistic three-component worst-case envelope (union bound form). -/
lemma threeLayer_error_union_bound
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (e_chunk e_sum e_oracle : Ω → ℝ)
    (r_chunk r_sum r_oracle : Ω → ℝ)
    (δ_chunk δ_sum δ_oracle : ℝ≥0∞)
    (h_chunk :
      μ {ω | |e_chunk ω| ≥ r_chunk ω} ≤ δ_chunk)
    (h_sum :
      μ {ω | |e_sum ω| ≥ r_sum ω} ≤ δ_sum)
    (h_oracle :
      μ {ω | |e_oracle ω| ≥ r_oracle ω} ≤ δ_oracle) :
    μ {ω | |e_chunk ω + e_sum ω + e_oracle ω| ≥
      r_chunk ω + r_sum ω + r_oracle ω} ≤
        δ_chunk + δ_sum + δ_oracle := by
  let E_chunk : Set Ω := {ω | |e_chunk ω| ≥ r_chunk ω}
  let E_sum : Set Ω := {ω | |e_sum ω| ≥ r_sum ω}
  let E_oracle : Set Ω := {ω | |e_oracle ω| ≥ r_oracle ω}
  let E_total : Set Ω := {ω | |e_chunk ω + e_sum ω + e_oracle ω| ≥
    r_chunk ω + r_sum ω + r_oracle ω}
  have h_subset : E_total ⊆ E_chunk ∪ E_sum ∪ E_oracle := by
    intro ω hω
    simpa [E_total, E_chunk, E_sum, E_oracle] using
      (threeLayer_error_event_subset
        (e_chunk := e_chunk) (e_sum := e_sum) (e_oracle := e_oracle)
        (r_chunk := r_chunk) (r_sum := r_sum) (r_oracle := r_oracle) hω)
  have h_measure_mono : μ E_total ≤ μ (E_chunk ∪ E_sum ∪ E_oracle) :=
    measure_mono h_subset
  have h_union :
      μ (E_chunk ∪ E_sum ∪ E_oracle) ≤ μ E_chunk + μ E_sum + μ E_oracle := by
    calc
      μ (E_chunk ∪ E_sum ∪ E_oracle)
          = μ (E_chunk ∪ (E_sum ∪ E_oracle)) := by
              simp [Set.union_assoc]
      _ ≤ μ E_chunk + μ (E_sum ∪ E_oracle) :=
            measure_union_le (μ := μ) E_chunk (E_sum ∪ E_oracle)
      _ ≤ μ E_chunk + (μ E_sum + μ E_oracle) := by
            simpa [add_assoc, add_comm, add_left_comm] using
              (add_le_add_left (measure_union_le (μ := μ) E_sum E_oracle) (μ E_chunk))
      _ = μ E_chunk + μ E_sum + μ E_oracle := by
            simp [add_assoc]
  have h_prob_sum :
      μ E_chunk + μ E_sum + μ E_oracle ≤ δ_chunk + δ_sum + δ_oracle := by
    have h12 : μ E_chunk + μ E_sum ≤ δ_chunk + δ_sum := add_le_add h_chunk h_sum
    exact add_le_add h12 h_oracle
  exact h_measure_mono.trans (h_union.trans h_prob_sum)

end DSL

end
