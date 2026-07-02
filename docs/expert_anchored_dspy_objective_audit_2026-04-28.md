# Expert-Anchored DSPy Objective Audit

Date: 2026-04-28

## Result

The active environment smoke is an expert-root-anchored DSPy objective:

```text
per tree:
  root-level expert anchor weight = gold_standard_lambda
  total teacher-node/local-law weight = 1 - gold_standard_lambda
```

This is implemented as weighted DSPy examples, not as a single explicit loss
tensor. The expert signal is root-level Benoit expert response supervision
attached to the stored-summary traces. Raw manifesto text is not required for
this audit.

After the 2026-04-28 scale-alignment pass, the next-version default keeps
expert-root labels on the internal `1-7` scale for every dimension, including
Environment. Older raw Environment artifacts remain reproducible by explicitly
passing `EXPERT_TARGET_SCALE=raw_benoit` or checker bounds `--target-min 0
--target-max 10`.

## Implementation Plumbing

- `scripts/run_benoit_supervised_dspy_ladder.sh` sets the single-dimension
  defaults used by the next-version smoke:
  - `FULL_DOC_ANCHOR_MODE=stored_summary`
  - `FULL_DOC_ANCHOR_TARGET=expert`
  - `GOLD_STANDARD_LAMBDA=0.75`, giving gold expert-root mass `0.75` and
    teacher/local-law mass `0.25`
  - `NODE_WEIGHT_NORMALIZATION=per_tree`
  - `EXPERT_TARGET_SCALE=normalized_1_7`
- `scripts/run_alternating_ladder.py` passes those objective settings into the
  DSPy family config and resolves default expert bounds to `target_min=1`,
  `target_max=7`. Teacher node scores remain scorer-output labels with
  `scorer_output_min=1`, `scorer_output_max=7`.
- `src/ctreepo/distillation.py` builds both record families:
  - Full-doc anchor records use `target_source=expert:*`,
    `observed_target=true`, and weight `gold_standard_lambda`.
  - Teacher node/local-law records use teacher/scorer targets and
    `_node_record_weight`. With `gold_standard_lambda=0.75` and
    `node_weight_normalization=per_tree`, each teacher record weight is
    `0.25 / n_teacher_records_for_tree`.

## Smoke Evidence

Legacy raw-scale artifact already written by the earlier smoke:

```text
outputs/manifesto_fg_alternating/environment_expert_anchor_smoke_20260428_164333/ladder/dspy/leaf4096tok/iter_01_train_g/g_training_records_summary_iter_01.json
```

The summary records:

```text
full_doc_g_anchor: count 105, total weight 105.0
leaf_g:            count 105, total weight 26.25
total weight:      131.25
```

Thus the written artifact has one expert root anchor per train tree at weight
`1.0`, and one teacher-node/local-law record per train tree at weight `0.25`.
The objective metadata records `target_min=0`, `target_max=10`,
`scorer_output_min=1`, and `scorer_output_max=7`, so this specific artifact is
valid evidence for provenance/role accounting but not for the next-version
internal scale or lambda-mixture invariants.

Reusable check for that legacy artifact:

```bash
./venv/bin/python scripts/audit_expert_anchor_dspy_objective.py \
  --target-min 0 \
  --target-max 10 \
  outputs/manifesto_fg_alternating/environment_expert_anchor_smoke_20260428_164333/ladder/dspy/leaf4096tok/iter_01_train_g/g_training_records_summary_iter_01.json
```

This command is expected to fail under the strict next-version checker unless
the legacy artifact is converted to the new `gold_standard_lambda`,
`gold_anchor_weight`, and `teacher_local_law_weight` metadata.

Reusable check for next-version artifacts keeps the default `1-7` target
bounds:

```bash
./venv/bin/python scripts/audit_expert_anchor_dspy_objective.py \
  path/to/next/g_training_records_summary_iter_01.json
```

## Paper And Lean Alignment

Paper side:

- `paper/ctreepo/appendix/v7_cdx/A_optimization_projection.tex` separates the
  gold/root prediction term from local-law residuals. It also states the
  teacher-first LLM regime explicitly: local laws can be optimized in the
  learned teacher score space, while final claims about the true oracle must
  add oracle-recovery slack.
- `paper/ctreepo/sections/v7_cdx/02_mergeable_sketches.tex` states the same
  operational certification route: local-law budget measured through a learned
  query plus an oracle-recovery term.

Lean side, using existing theorem surfaces only:

- `lean3/FormalProofs/OPT/RegularizedObjective.lean` defines
  `NoCostLearnedTreeObjective`, whose value is calibration plus gold/root loss
  plus C1/C3/C2 penalties.
- `lean3/FormalProofs/OPT/DiscountedIPWObjective.lean` proves
  `fullWeightedDocumentObjective_eq_fullSupervisionTreeObjectiveFn`, packaging
  root/C1/C2/C3 supervision channels as a generic weighted objective.
- `lean3/FormalProofs/OPT/TwoStageOracleSurrogate.lean` defines
  `UniformOracleApproximation` and the two-sided oracle-recovery slack for
  transferring learned-teacher comparisons back to the true oracle.
- `lean3/FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean` defines
  `TeacherFirstLocalLawRoute` and proves that teacher-space local-law error
  plus oracle-recovery slack bounds true-oracle tree distortion.

The key invariant is not that the teacher equals the oracle. The objective is a
blend of gold root-level expert supervision and non-gold teacher/scorer
local-law supervision:

- Gold signal: the full-doc expert anchor has `observed_target=true` and weight
  `gold_standard_lambda`.
- Non-gold signal: teacher/scorer node-local-law records have total per-tree
  weight `1 - gold_standard_lambda`.

The current 4096 artifact has one `leaf_g` teacher record per tree, so it
demonstrates provenance and weight blending, but not rich non-root local-law
coverage. That teacher/local-law supervision can still affect root behavior:
it trains the `f` and `g` operators used to compose summaries and scores up to
the root. The audit checker therefore validates provenance and weights without
requiring non-root local-law roles.

## Verification Commands

```bash
source venv/bin/activate
pytest tests/ctreepo/test_distillation_labeled_nodes.py \
  tests/ctreepo/test_expert_anchor_dspy_objective_audit.py -q

cd lean3
lake env lean FormalProofs/OPT/RegularizedObjective.lean
lake env lean FormalProofs/OPT/DiscountedIPWObjective.lean
lake env lean FormalProofs/OPT/TwoStageOracleSurrogate.lean
lake env lean FormalProofs/OPT/NeuralOperatorPreferenceBridge.lean
```

No new Lean theorem names or concrete constant specializations are required for
this audit.
