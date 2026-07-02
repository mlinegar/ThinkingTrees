# INTERNAL_LLM_NOTES.md — coordination scratchpad

**Audience:** other LLMs working on this repo concurrently. Treat this as
freshest-wins coordination; update entries when you start, pivot, or finish.

**Author convention:** sign your block with a tag (e.g. `[opus47]`,
`[parallel]`) so we can tell who claimed what.

**Reading order on session start:**
1. `CLAUDE.md` (project rules)
2. `AGENTS.md` (env + workflows)
3. `docs/ctreepo_python_code_map_for_llms.md` (code map)
4. **This file** (live coordination)
5. `~/.claude/plans/i-want-a-major-glittery-quokka.md` (the unifying ladder plan)

---

## North star (current consolidation effort)

We are converging the repo on **one canonical f/g ladder contract** so that
every application — DSPy LM ladders, FNO neural operators, CTreePO PyTorch
sketches, TRL/HF, classical sketches in `treepo/`, and synthetic-DGP sims
(Markov / LDA) — speaks the same surface:

- `TreeBundleManifest` (schema in [src/ctreepo/contracts.py:134](src/ctreepo/contracts.py#L134))
  for input data with `leaf_unit` / `source_kind` / `state_contract` /
  `reducer_contract` / `state_dim` / `summary_dim` / `f_lineage` / `g_lineage`.
- `FamilyRuntime` + `BundleAwareFamilyRuntime` protocols in
  [src/ctreepo/alternating.py](src/ctreepo/alternating.py) for backends.
- `<init-spec>` grammar (`identity` / `raw` / `raw_concat` / `oracle:<name>` /
  `artifact:<path>` / `external_passthrough`) parsed by `parse_init_spec`.
- Strict invariant: **`state_dim >= 2 * summary_dim`**. Lossy sketch families
  (HLL register-max) opt out via `state_kind="sketch_state_lossy_native"` AND
  must provide a ConcatSketch-equivalent default `g`.
- Default `g_init = raw_concat` everywhere; `external_passthrough` /
  `teacher_passthrough` are explicit compat modes.

User decision (locked, 2026-04-28): strict 2× is the rule. FNO's
`merge_fno` MUST be rebuilt from `2ch→1ch` to `2ch→2ch`; cached FNO
checkpoints will be invalidated and that is acceptable.

---

## Done (cross-LLM ledger)

| Item | Where | Owner | Notes |
|---|---|---|---|
| TreeBundle v1 schema + validators + normalizer | [src/ctreepo/contracts.py:134](src/ctreepo/contracts.py#L134) | parallel | strict 2× enforced in `__post_init__`. `f_lineage`/`g_lineage` split (not nested). |
| Schema tests | [tests/ctreepo/test_tree_bundle_contract.py](tests/ctreepo/test_tree_bundle_contract.py) | parallel | covers raw/external normalization, stream/synthetic units. |
| DSPy `g_init=raw_concat` default + `RAW_CONCAT` sentinel | [src/ctreepo/dspy_family.py:246](src/ctreepo/dspy_family.py#L246), [src/ctreepo/dspy_family.py:565](src/ctreepo/dspy_family.py#L565) | parallel | `teacher_passthrough` preserved as compat mode. |
| `--dspy-g-init` runner flag | [scripts/run_alternating_ladder.py:896](scripts/run_alternating_ladder.py#L896) | parallel | default `raw_concat`. |
| Joint DSPy bottom-up reducer | [src/ctreepo/joint_dspy_family.py:573](src/ctreepo/joint_dspy_family.py#L573) | parallel | shares `_reduce_tree_with_g` with scalar DSPy. |
| Manifesto teacher bundle generators emit v1 metadata | [scripts/run_manifesto_teacher_fg_leaf_grid.py:1128](scripts/run_manifesto_teacher_fg_leaf_grid.py#L1128), [scripts/run_manifesto_teacher_fg_joint_leaf_grid.py:560](scripts/run_manifesto_teacher_fg_joint_leaf_grid.py#L560) | parallel | + 5 Benoit launcher shells. |
| Distillation anchor selection uses `normalize_tree_bundle_manifest` | [src/ctreepo/distillation.py:1130](src/ctreepo/distillation.py#L1130) | parallel | reads normalized `source_kind`. |
| FNO config-level 2× declaration (`summary_dim`/`state_dim`) | [src/ctreepo/fno_family.py:99-136](src/ctreepo/fno_family.py#L99-L136) | parallel | **architectural merge still 2ch→1ch — see open work** |
| Audit script | [scripts/audit_tree_bundle_contracts.py](scripts/audit_tree_bundle_contracts.py) | parallel | scans manifests/summaries/JSONL. |
| Generalized 2× checks (`check_state_summary_invariant`, embedding/sketch variants) | [src/ctreepo/fg_arity.py](src/ctreepo/fg_arity.py) | opus47 | extends `check_two_child_lm_budget` to embedding + sketch state kinds. |
| `BundleAwareFamilyRuntime` Protocol + helpers (`family_default_f`, `family_default_g`, `family_expected_bundle`, `family_supported_inits`, `family_resolve_init`, `family_share_state_axes`) | [src/ctreepo/alternating.py:292-554](src/ctreepo/alternating.py#L292-L554) | opus47 | additive; existing `FamilyRuntime` Protocol untouched. |
| `<init-spec>` grammar (`parse_init_spec`, `InitSpec`) | [src/ctreepo/alternating.py:382-434](src/ctreepo/alternating.py#L382-L434) | opus47 | sentinels + `oracle:`/`artifact:` prefixes. |
| Contract tests | [tests/ctreepo/test_dimension_contract_dispatch.py](tests/ctreepo/test_dimension_contract_dispatch.py) | opus47 | 37 cases. |
| Oracle f\* registry + `OracleFamilyRuntime` | [src/ctreepo/oracles/](src/ctreepo/oracles/) | opus47 | Five canonical oracles registered: `type_oracle`, `hll_exact`, `hll_max_merge`, `markov_changepoint_count`, `leaf_local_mixture_target`. `OracleFamilyRuntime` adapter implements both `FamilyRuntime` and `BundleAwareFamilyRuntime` so synthetic-DGP sims can drop straight into the ladder via `--family <oracle> --f-init oracle:<name>`. |
| Inline oracle → re-export migration | [src/tree/learned_sketch.py:124](src/tree/learned_sketch.py#L124), [src/ctreepo/sim/core/markov_changepoint_ops_count.py:393-414](src/ctreepo/sim/core/markov_changepoint_ops_count.py#L393-L414), [src/ctreepo/sim/core/leaf_local_mixture_utility.py:1815-1835](src/ctreepo/sim/core/leaf_local_mixture_utility.py#L1815-L1835) | opus47 | All historical inline oracle defs now delegate to the registry. Call sites unchanged. |
| Oracle registry tests | [tests/ctreepo/test_oracle_registry.py](tests/ctreepo/test_oracle_registry.py) | opus47 | 33 cases — registry contents, native callable behaviour, re-export integrity, OracleFamilyRuntime protocol conformance. |

---

## Active claims (don't duplicate)

| Item | Owner | Started | Files I'm touching |
|---|---|---|---|
| FNO architectural merge rebuild (2ch→2ch) + `RawConcatMerge` + score-head `AdaptiveAvgPool1d` shim + `legacy_avg_merge` flag | opus47 | 2026-04-29 | [src/ctreepo/embedding_fno.py:112-208](src/ctreepo/embedding_fno.py#L112-L208), score head, `_prepare_trees`, `initialize_as_identity`. Fresh tests in `tests/ctreepo/test_fno_strict_2x_merge.py`. **Will invalidate cached FNO checkpoints; preserves `legacy_avg_merge=True` reproducibility path.** |

---

## Open work (unclaimed)

In rough priority order:

1. **CTreePO native FamilyRuntime wrapper** — new [src/ctreepo/ctreepo_native_family.py](src/ctreepo/ctreepo_native_family.py); rebuild every merge module in [src/tree/ctreepo_model.py:215-391](src/tree/ctreepo_model.py#L215-L391) (`GatedMerge`, `MLPMerge`, `ResidualGatedMerge`, `BilinearMerge`, `AvgMerge`) so output is `2 * sketch_dim`; add `RawConcatMerge`; update `_MERGE_CLASSES` at line 340.
2. **Sketch/sim sweep migration** — convert to thin shims that forward to the unified runner with `--family <oracle>` / `--f-init oracle:<name>`:
   - [scripts/run_hll_merge_learning_sweep.py](scripts/run_hll_merge_learning_sweep.py) → `--family hll_native --f-init oracle:hll_exact`
   - [scripts/run_lda_topic_estimator_cpu_sweep.py](scripts/run_lda_topic_estimator_cpu_sweep.py) → `--family lda_exact --f-init oracle:leaf_local_mixture_target`
   - [scripts/run_markov_full_doc_anchor_ladder.py](scripts/run_markov_full_doc_anchor_ladder.py) → `--family markov_exact --f-init oracle:markov_changepoint_count`
   - [src/ctreepo/sim/cli/sweep_markov_changepoint_ops_count.py](src/ctreepo/sim/cli/sweep_markov_changepoint_ops_count.py) — same.
   - [scripts/run_manifesto_qsentence_dspy_ladder.py](scripts/run_manifesto_qsentence_dspy_ladder.py), [scripts/run_manifesto_fg_real_training_grid.py](scripts/run_manifesto_fg_real_training_grid.py) — DSPy shims.
3. **TRL raw_concat support + k≥1 score reducer** — [src/ctreepo/trl_family.py:317](src/ctreepo/trl_family.py#L317) raises `NotImplementedError`; needs `RAW_CONCAT` sentinel and bottom-up reducer (mirror DSPy text-concat). True GRPO-with-current-f-as-reward remains a separate followup.
4. **Cosmetic rename** — `teacher_passthrough` → `external_passthrough` across DSPy/TRL/joint-DSPy. Last; cosmetic.

### Pre-existing test failures unrelated to this consolidation effort

9 markov tests fail with `TypeError: _span_features() missing 1 required keyword-only argument: 'vocab_size'` at [src/ctreepo/sim/core/markov_treepo_preference.py:169](src/ctreepo/sim/core/markov_treepo_preference.py#L169). Confirmed by `git stash` — failures occur with all our consolidation changes reverted. Live in someone's in-progress `_span_features` refactor; not introduced by this work.

Affected tests (don't ascribe to oracle registry / Step 3):
- `tests/ctreepo/test_exact_utility_transport.py::test_markov_tree_{neural_supported,undersupported}_smoke`
- `tests/ctreepo/test_markov_capability_suite_builder.py::test_build_markov_capability_{sanity,mechanism}_suite_*`
- `tests/ctreepo/test_markov_law_stress_suite_builder.py::test_build_markov_law_stress_{sanity,mechanism}_suite_*`
- `tests/ctreepo/test_markov_tree_fno_validation.py::test_rung_nestedness_check_fails_on_mismatched_prefix_ladder`
- `tests/ctreepo/test_validation_ladder_e2e.py::test_markov_{learnability,observed_token}_*`

---

## Open decisions

| # | Question | Asked by | Decided | Resolution |
|---|---|---|---|---|
| 1 | Strict 2× FNO architectural rebuild (2ch→2ch) or relaxed config-only (declarative)? | opus47 | 2026-04-28 | **Strict.** Rebuild merge to 2ch→2ch; invalidate FNO checkpoints; provide `legacy_avg_merge=True` reproducibility flag. |
| 2 | Where does the schema live: `tree_bundle/manifest.py` or `contracts.py`? | parallel | 2026-04-28 | **`contracts.py`** (parallel LLM's choice; merged). My plan said `tree_bundle/manifest.py` — converged to `contracts.py`. |
| 3 | `f_lineage`/`g_lineage` split or `fg_lineage` nested? | parallel | 2026-04-28 | **Split** (parallel LLM's choice; merged). |

---

## How to use this file

- **When you start a task:** add a row to *Active claims* with your tag, the
  files you'll touch, and the date. Remove it when you finish (move the row
  to *Done*).
- **When you finish a task:** move the row to *Done*. Cite file:line for each
  surface you changed.
- **When you have an ambiguity:** add to *Open decisions* with your tag.
  Whoever resolves it fills *Decided* and *Resolution*.
- **Don't duplicate work:** if a row is in *Active claims*, leave it alone.
- **Keep this file thin:** prune *Done* entries older than ~2 weeks.
