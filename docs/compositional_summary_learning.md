# Compositional Summary Learning

This note records the package-level abstraction we want the repo to grow into.

## Problem shape

We are not solving only one problem instance such as Markov counts, LDA, or a
particular LLM summarizer. The broad object is:

1. a compositional summary operator over documents or spans;
2. a theorem-domain object we want the operator to preserve exactly or
   approximately; and
3. one or more supervision channels used to learn the operator.

The operator side is already captured by:

- [compositional_operator.py](/home/mlinegar/ThinkingTrees/src/tree/compositional_operator.py)
- [theorem_backing.py](/home/mlinegar/ThinkingTrees/src/tree/theorem_backing.py)

The new supervision-side abstraction is:

- [compositional_learning.py](/home/mlinegar/ThinkingTrees/src/tree/compositional_learning.py)
- [logged_supervision.py](/home/mlinegar/ThinkingTrees/src/core/logged_supervision.py)

`compositional_learning.py` now carries the intended problem spec, while
`logged_supervision.py` carries the realized sampled/query records.

## Two supervision channels

The package now makes explicit that labels arrive in two structurally different
ways.

### 1. Full-document labels

These supervise the final theorem-domain or downstream target directly on the
whole document.

Examples:

- a full-document score for an LLM summary
- a document-level utility or preference target
- a full-document oracle value in a simulation

This is represented by `SupervisionChannelKind.FULL_DOCUMENT` and
`full_document_supervision_channel(...)`.

### 2. Sampled substructure labels

These supervise leaves, spans, or internal nodes drawn by a sampling policy and
then labeled.

Examples:

- queried node labels for C1 / C2 / C3 local-law supervision
- sampled span labels from a task oracle
- randomly selected leaves or merges annotated during training

This is represented by `SupervisionChannelKind.SAMPLED_SUBSTRUCTURE` and
`sampled_substructure_supervision_channel(...)`.

When these labels are acquired by random sampling, the package treats
propensity logging as first-class because unbiased risk accounting typically
depends on it.

## Online oracle queries

The same channel surface now also records whether sampled labels come from:

- offline logged labels already attached to the artifact; or
- online oracle queries made under a named sampling/query policy.

That matters because the Markov and LDA simulation lanes already behave like
"query the oracle under an IPW-aware design", while the auditor does the same
for real trees. The shared schema now has room for both:

- `delivery_mode`
- `query_policy`
- `uses_online_oracle_queries`

So the repo can grow toward a single online-learning interface instead of
having one manifest shape for static labels and another for oracle-calling
workflows.

The shared helper layer now also fixes one application-neutral vocabulary for
the common protocol:

- `full_document_supervision`
- `sampled_substructure_supervision`
- `sampled_substructure_query_policy`

Auditor runs, CTreePO training runs, and Markov/LDA local-law summaries should
now read as different instantiations of that same protocol, not as separate API
families.

## Why this matters

This abstraction captures the common logic across:

- LLM summary-object learning
- mergeable-sketch/operator learning
- theorem-backed codec learning
- local-law supervision with sampled node labels
- downstream objective learning from full-document labels

The backend can vary. The supervision logic is the same:

- whole-object labels supervise the root/downstream target;
- sampled substructure labels supervise local structure;
- theorem-backedness comes from the supplied operator assumptions, not from the
  fact that a model is called "LLM" or "neural".

## Current package status

The first general API now exists as:

- `CompositionalLearningProblemSpec`
- `SupervisionChannelSpec`
- `FullDocumentLabelObservation`
- `SampledSubstructureLabelObservation`
- `FullDocumentLabelSource`
- `SampledSubstructureLabelSource`

These abstractions are now threaded into the main trainer/auditor artifacts so
concrete runs record:

1. which supervision channels were active;
2. whether sampled channels logged propensities;
3. which theorem-backing assumptions were supplied for the operator.

Realized sampled labels now also share one canonical record model:

- `SamplingMetadata`
- `LoggedLabelObservation`
- `LoggedObservationArtifact`

And the realized-observation helpers now normalize the common target slots:

- `document_level_target`
- `substructure_level_target`

Application-specific distinctions such as `sufficiency`, `c1`, `c3`, or
`node_oracle_score` now belong in logged observation context under
`supervision_signal_name`, with `application_name` and optional `law_kind`
alongside them.

Today that includes:

- CTreePO `training_result.json`
- neural-operator orchestration `summary.json`
- `AuditReport.to_dict()` / harness audit-report exports
- single-document audit `audit_report.json` and `manifest.json`
- Markov / LDA `local_law_learnability` summaries, including legacy backfills

That means the two main settings now line up on one problem-spec surface:

- LLM trainer / auditor artifacts emit `CompositionalLearningProblemSpec`
  directly.
- Markov local-law runs now emit the same spec through
  `LocalLawRunSummary.compositional_learning_problem`.
- LDA local-law runs use the same bridge, so they no longer need a separate
  manifest shape just because the backend is simulation-heavy.

The remaining follow-up is broader adoption across orchestration and comparison
manifests that ingest these run artifacts second-hand.
