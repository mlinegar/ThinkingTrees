import FormalProofs.DSL.Honesty

/-!
# Document Structure, Partitions, and Run Manifests

This module records the structural commitments used by the unified learning
procedure:

* top-level units are finite ordered documents/cases;
* chunking returns admissible non-overlapping span partitions;
* nodes and local-law audit rows carry support spans and parent top-level IDs;
* run manifests log fold roles, artifact lineage, propensities, and influence
  weights for theorem-facing audit rows.

The definitions are intentionally lightweight.  They are contracts that Python
artifacts and paper tables should satisfy; they do not impose a specific text
tokenizer or tree implementation.
-/

namespace DSL

open scoped Classical BigOperators NNReal ENNReal

/-- Half-open document span `[start, stop)`. -/
structure Span where
  start : ℕ
  stop : ℕ
deriving DecidableEq, Repr

namespace Span

/-- A span is valid inside a document of length `n`. -/
def Valid (n : ℕ) (s : Span) : Prop :=
  s.start < s.stop ∧ s.stop ≤ n

/-- A position belongs to a half-open span. -/
def Contains (s : Span) (pos : ℕ) : Prop :=
  s.start ≤ pos ∧ pos < s.stop

/-- Two half-open spans do not overlap. -/
def Nonoverlapping (a b : Span) : Prop :=
  a.stop ≤ b.start ∨ b.stop ≤ a.start

theorem nonoverlapping_symm {a b : Span} :
    Nonoverlapping a b → Nonoverlapping b a := by
  intro h
  rcases h with h | h
  · exact Or.inr h
  · exact Or.inl h

theorem contains_of_valid_lt {n : ℕ} {s : Span} {pos : ℕ}
    (h_valid : Valid n s) (h_start : s.start ≤ pos) (h_stop : pos < s.stop) :
    pos < n := by
  exact lt_of_lt_of_le h_stop h_valid.2

end Span

/-- Every listed span is valid for a document of length `n`. -/
def SpansValid (n : ℕ) (spans : List Span) : Prop :=
  ∀ s ∈ spans, s.Valid n

/-- Listed spans are pairwise non-overlapping. -/
def SpansPairwiseNonoverlap (spans : List Span) : Prop :=
  spans.Pairwise Span.Nonoverlapping

/-- Listed spans cover every position in a document of length `n`. -/
def SpansCover (n : ℕ) (spans : List Span) : Prop :=
  ∀ pos, pos < n → ∃ s, s ∈ spans ∧ s.Contains pos

/-- An admissible chunk partition for a finite ordered document. -/
structure AdmissiblePartition (n : ℕ) where
  spans : List Span
  spans_nonempty : spans ≠ []
  spans_valid : SpansValid n spans
  spans_pairwise : SpansPairwiseNonoverlap spans
  spans_cover : SpansCover n spans

namespace AdmissiblePartition

/-- Every span in an admissible partition lies inside the document. -/
theorem span_valid {n : ℕ} (p : AdmissiblePartition n) {s : Span}
    (h_mem : s ∈ p.spans) : s.Valid n :=
  p.spans_valid s h_mem

/-- Every document position is covered by some partition span. -/
theorem exists_span_contains {n pos : ℕ} (p : AdmissiblePartition n)
    (h_pos : pos < n) :
    ∃ s, s ∈ p.spans ∧ s.Contains pos :=
  p.spans_cover pos h_pos

end AdmissiblePartition

/-- A top-level document/case type with a finite ordered length. -/
structure FiniteTopLevelUnit (Case : Type*) where
  length : Case → ℕ
  length_pos : ∀ x, 0 < length x

/-- A chunker output is structurally admissible for every top-level unit. -/
structure ChunkPartitionContract (Case : Type*) where
  unit : FiniteTopLevelUnit Case
  partition : (x : Case) → AdmissiblePartition (unit.length x)

/-- Supported tree nodes have a parent top-level unit and a span inside it. -/
structure SupportedNodeContract (Case Node : Type*) where
  unit : FiniteTopLevelUnit Case
  parent : Node → Case
  support : Node → Span
  support_valid : ∀ v, (support v).Valid (unit.length (parent v))

/-- Paper local-law row kind.  These are the report-facing C1/C2/C3 names. -/
inductive LocalLawKind where
  | c1_leaf
  | c2_idempotence
  | c3_merge
deriving DecidableEq, Repr

/-- The artifact lineage used to produce a reported row. -/
structure ArtifactLineage (ArtifactId : Type*) where
  chunker : ArtifactId
  g : ArtifactId
  f : ArtifactId
  oracleOnline : ArtifactId
  oracleEval : ArtifactId
  queryPolicy : ArtifactId
  proxy : Option ArtifactId

/-- Logged split roles for one top-level unit.  `true` means train and `false`
means eval, matching `SampleSplit.isTrain`. -/
structure ThreeRoleTuple where
  chunker : Bool
  g : Bool
  oracle : Bool

/-- One theorem-facing manifest row. -/
structure AuditRowLog (Case Row ArtifactId : Type*) where
  row : Row
  topLevelUnit : Case
  sourceUnit : Case
  rowId : ℕ
  nodeId : Option ℕ
  pairId : Option (ℕ × ℕ)
  foldId : ℕ
  splitSeed : ℕ
  roles : ThreeRoleTuple
  artifacts : ArtifactLineage ArtifactId
  lawKind : LocalLawKind
  support : Span
  observed : Bool
  propensity : ℝ
  effectivePropensity : ℝ
  influenceWeight : ℝ

/-- Manifest-level contract tying logged rows back to parent units and the audit
design quantities used in finite-sample certificates. -/
structure RunManifestContract (Case Row ArtifactId : Type*) where
  parent : ParentOf Case Row
  log : Row → AuditRowLog Case Row ArtifactId
  top_level_unit_logged : ∀ r, (log r).topLevelUnit = parent r
  row_logged : ∀ r, (log r).row = r
  propensity_pos : ∀ r, 0 < (log r).propensity
  effective_propensity_pos : ∀ r, 0 < (log r).effectivePropensity

namespace RunManifestContract

/-- The parent map recovered from a valid manifest agrees with the logged unit. -/
theorem parent_eq_logged {Case Row ArtifactId : Type*}
    (m : RunManifestContract Case Row ArtifactId) (r : Row) :
    m.parent r = (m.log r).topLevelUnit :=
  (m.top_level_unit_logged r).symm

/-- A manifest row has positive logged propensity. -/
theorem logged_propensity_pos {Case Row ArtifactId : Type*}
    (m : RunManifestContract Case Row ArtifactId) (r : Row) :
    0 < (m.log r).propensity :=
  m.propensity_pos r

end RunManifestContract

/-- Manifest roles are consistent with a three-layer honest split. -/
def ManifestRolesConsistent {Case Row ArtifactId : Type*}
    (splits : ThreeLayerSplit Case)
    (m : RunManifestContract Case Row ArtifactId) : Prop :=
  ∀ r,
    (m.log r).roles.chunker = splits.chunk.isTrain (m.parent r) ∧
    (m.log r).roles.g = splits.summarizer.isTrain (m.parent r) ∧
    (m.log r).roles.oracle = splits.oracle.isTrain (m.parent r)

/-- Logged row supports are valid spans in the row's parent top-level unit. -/
def ManifestSupportsValid {Case Row ArtifactId : Type*}
    (unit : FiniteTopLevelUnit Case)
    (m : RunManifestContract Case Row ArtifactId) : Prop :=
  ∀ r, (m.log r).support.Valid (unit.length (m.parent r))

/-- Audit rows inherit their train/eval role from the logged top-level unit. -/
theorem manifest_inherited_role_eq {Case Row ArtifactId : Type*}
    (split : TopLevelSplit Case)
    (m : RunManifestContract Case Row ArtifactId) (r : Row) :
    inheritedSplitRole split m.parent r =
      split.isTrain (m.log r).topLevelUnit := by
  simp [inheritedSplitRole, m.parent_eq_logged r]

end DSL
