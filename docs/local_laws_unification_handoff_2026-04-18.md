# Handoff: Unified local-law enforcement across ThinkingTrees (2026-04-18)

## Mission (what this was)

Put every training workload in the repo — sketch, Markov FNO, embedding FNO,
DSPy-GEPA LLM, TRL-GRPO LLM — through the same unified `fit()` framework
and force each one to express **strict local laws** (C1 per-leaf, C2
merge-consistency, C3 per-merge) in addition to root supervision.

User directive that drove this: "every piece needs to be able to have
local laws… otherwise we cannot be sure that the merges are actually
enforced." Without per-node supervision the merge operator can learn
arbitrary intermediate representations that happen to predict the root
— the whole point of the framework is that each node of the tree pays
for its own correctness.

## Where we ended up (current state)

**Tests: 107/107 passing** (up from 82 at session start).

### Local-law coverage

| Path | Root | C1 (leaf) | C2 (merge) | C3 (merge scalar) |
|---|---|---|---|---|
| **Markov FNO** | ✅ MSE | ✅ strict `leaf_loss` from `FNOCountSketch.forward_doc` | ✅ strict `c2_loss` | ✅ strict `c3_loss` |
| **Mergeable sketch** | ✅ scalar MSE | ✅ strict per-leaf head MSE on analytic bigram counts | ✅ strict merge-state reconstruction vs analytic sketch at every level | ✅ strict per-merge head MSE on cumulative counts |
| **Embedding FNO (Manifesto)** | ✅ normalized MSE | ✅ **strict** per-leaf RILE from Manifesto quasi-sentence codings (soft root-pull fallback when doc isn't granularly coded) | ✅ `merge(a,b) == merge(b,a)` commutativity (RILE is permutation-invariant) | ✅ **strict** per-merge RILE from Manifesto codings (soft fallback) |
| **DSPy-GEPA tree** | ✅ | ✅ per-leaf RILE MAE from `RileTreeProgram` | ✅ root-merge commutativity probe | ✅ per-internal-node RILE MAE |
| **TRL GRPO tree** | ✅ | ✅ same reward function | ✅ | ✅ |

### Single balance knob — `local_law_weight` (λ) + `*_relative_weight` (ρ)

Every objective uses the canonical Lean-facing formula:

```
loss = (1 - λ) · root + λ · [ρ_C1·C1 + ρ_C2·C2 + ρ_C3·C3] / (ρ_C1 + ρ_C2 + ρ_C3)
```

- **`local_law_weight` (λ)**: default **0.3**, range [0, 1]. Matches the
  existing `OPSCountConfig.local_law_weight` convention used in
  `scripts/run_markov_publication_bundle.py` and the Lean-aligned simulation
  tests.
- **`c1_relative_weight` / `c2_relative_weight` / `c3_relative_weight`**
  (ρ's): default **1.0 each** → uniform local-law distribution.
- Missing laws (e.g., doc without coded spans) drop out of the ρ sum
  cleanly — the compound score stays interpretable regardless of coverage,
  never NaN, and λ always represents the share the caller intended for
  active laws.

## Files to know

### New
- [src/tasks/manifesto/rile_codes.py](../src/tasks/manifesto/rile_codes.py) — canonical CMP RILE code sets + `RILECorpusIndex` loading `data/raw/manifesto_project_full/manifesto_corpus_df.csv` (2.27M rows, 2157 manifestos with granular codings) and answering `span_rile(manifesto_id, start_char, end_char)` queries.
- [parallel/unified_g_v1/src/unified_g_v1/markov/tree_task.py](../parallel/unified_g_v1/src/unified_g_v1/markov/tree_task.py) — `MarkovChangepointOracle` + `MarkovFNOModel` (wraps `FNOCountSketch`) + `MarkovChangepointObjective` + `markov_changepoint_task(...)` preset. Runs Markov through `fit()`.
- [parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree.py](../parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree.py) — `build_rile_tree_scaffold` + `rile_tree_reward` (the pure function that turns per-node predictions into the λ-weighted reward + feedback).
- [parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree_program.py](../parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree_program.py) — `RileTreeProgram` applying `RILESummarize`/`RILEMerge`/`RILEScoreSignature` at every leaf, every merge, and a commutativity probe.
- [parallel/unified_g_v1/src/unified_g_v1/training/trainers/rile_tree_feedback.py](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/rile_tree_feedback.py) — `dspy_gepa_metric_from_rollout` (→ `ScoreWithFeedback`) and `trl_grpo_rewards_from_rollouts` (→ `list[float]`).

### Modified (objective/compose side)
- [parallel/unified_g_v1/src/unified_g_v1/sketch/tree_task.py](../parallel/unified_g_v1/src/unified_g_v1/sketch/tree_task.py) — `MergeableSketchModel.forward_tree` now returns per-leaf/per-merge predictions in `forward_aux`; `mergeable_sketch_task(...)` preset takes λ/ρ.
- [parallel/unified_g_v1/src/unified_g_v1/sketch/runner.py](../parallel/unified_g_v1/src/unified_g_v1/sketch/runner.py) — refactored onto the shared `run_pytorch_training` loop (so it inherits best-checkpoint, periodic train_state snapshot, auto-resume).
- [parallel/unified_g_v1/src/unified_g_v1/training/objectives/mergeable_sketch.py](../parallel/unified_g_v1/src/unified_g_v1/training/objectives/mergeable_sketch.py) — strict C1/C2/C3 with λ/ρ.
- [parallel/unified_g_v1/src/unified_g_v1/training/oracles/mergeable_sketch.py](../parallel/unified_g_v1/src/unified_g_v1/training/oracles/mergeable_sketch.py) — emits per-leaf + per-merge analytic bigram counts + analytic merge sketches.
- [parallel/unified_g_v1/src/unified_g_v1/realdoc/embedding_fno_training.py](../parallel/unified_g_v1/src/unified_g_v1/realdoc/embedding_fno_training.py) — `EmbeddingSequenceFNOTreeModel.forward_tree` returns per-node states + swapped-merge probe; `_annotate_rile_targets_on_tree` populates per-node RILE targets from `span_rile_fn`; `EmbeddingFNOTrainingConfig` has λ/ρ; internal `TreeModelV2TrainingConfig` translates λ/ρ to its `root_weight`/`leaf_scalar_weight`/`internal_scalar_weight` slots (no C2 slot available in v2 trainer — see "known limitations").
- [parallel/unified_g_v1/src/unified_g_v1/training/oracles/manifesto_rile_embedding.py](../parallel/unified_g_v1/src/unified_g_v1/training/oracles/manifesto_rile_embedding.py) — `enforce_local_laws=True` flag wires in `RILECorpusIndex` + annotator; `TreeExample.extra` carries `leaf_rile_targets` + `internal_rile_targets`.
- [parallel/unified_g_v1/src/unified_g_v1/training/objectives/manifesto_rile_embedding.py](../parallel/unified_g_v1/src/unified_g_v1/training/objectives/manifesto_rile_embedding.py) — strict C1/C3 when per-node targets present, pure-root fallback otherwise; C2 always active via commutativity probe; λ/ρ knobs.
- [parallel/unified_g_v1/src/unified_g_v1/training/recipes.py](../parallel/unified_g_v1/src/unified_g_v1/training/recipes.py) — `manifesto_rile_embedding_fno_task(...)` preset surfaces λ/ρ.
- [parallel/unified_g_v1/src/unified_g_v1/bundles.py](../parallel/unified_g_v1/src/unified_g_v1/bundles.py) — `run_embedding_fno_train_bundle` + `run_manifesto_embedding_fno_train_bundle` take λ/ρ and forward them through the subprocess CLI.
- [parallel/unified_g_v1/scripts/run_manifesto_embedding_fno_training.py](../parallel/unified_g_v1/scripts/run_manifesto_embedding_fno_training.py) — CLI flags `--local-law-weight` + `--c{1,2,3}-relative-weight`.
- [parallel/unified_g_v1/src/unified_g_v1/training/trainers/pytorch_tree.py](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/pytorch_tree.py) — `_TreeTaskSupervisionAdapter` auto-detects 3-tuple model returns and whether an objective declares `forward_aux`, routing accordingly. This is the load-bearing interface change that lets per-forward intermediate tensors flow from model to objective.
- [parallel/unified_g_v1/src/unified_g_v1/training/tree_task.py](../parallel/unified_g_v1/src/unified_g_v1/training/tree_task.py) — `TreeObjective` protocol now declares `forward_aux: Mapping | None = None`.
- [parallel/unified_g_v1/src/unified_g_v1/training/objectives/{simple,manifesto_rile_embedding,mergeable_sketch}.py](../parallel/unified_g_v1/src/unified_g_v1/training/objectives) — all accept the new kwarg (back-compat via default None).

### New tests (+21 vs. session start)
- `tests/test_sketch_resumes.py` (2) — sketch auto-resume via pytorch_loop.
- `tests/test_markov_fit.py` (4) — Markov runs through `fit()`.
- `tests/test_sketch_local_laws.py` (2) — C1/C2/C3 MAE actually drops with training.
- `tests/test_manifesto_rile_codes.py` (4) — `rile_sign`, `span_rile`, tree annotator, strict-objective path.
- `tests/test_rile_tree_reward.py` (5) — scaffold builder + reward math.
- `tests/test_rile_tree_feedback.py` (4) — DSPy/GRPO wrappers + λ knob extremes.
- `tests/test_rile_tree_program.py` (3) — RileTreeProgram rollout visits every node.

Run: `./venv/bin/python -m pytest -q parallel/unified_g_v1/tests`

## Known limitations + what to finish next

### 1. Embedding-FNO bundle path has no C2 slot in the v2 trainer
`run_embedding_fno_training(config)` uses `TreeModelV2Trainer` (legacy),
which has `root_weight` / `leaf_scalar_weight` / `internal_scalar_weight`
slots — analogous to root/C1/C3 — but **no C2 slot**. When translating λ/ρ
I distribute λ mass across C1/C3 only; ρ_C2 is ignored in that path. The
`manifesto_rile_embedding_fno_task(...)` preset (→ `fit()` →
`ManifestoRileEmbeddingObjective`) is the full-C2 path.

**Finish**: port `run_embedding_fno_training` internals from
`TreeModelV2Trainer` to `fit()` + `ManifestoRileEmbeddingObjective`, then
delete the v2 path. Moderate refactor (~1 day); gated on no downstream
callers relying on `TreeModelV2Trainer`-specific behavior.

### 2. DSPy-GEPA tree trainer is infrastructure-only, not wired to a live LLM
[RileTreeProgram](../parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree_program.py) composes `RILESummarize`/`RILEMerge`/`RILEScoreSignature`
and produces `TreeRilePredictions`. [`dspy_gepa_metric_from_rollout`](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/rile_tree_feedback.py) converts that
into `ScoreWithFeedback`. What's NOT done: replacing the flat
`ChainOfThought(RilePredictor)` program in [`trainers/dspy_rile.py`](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/dspy_rile.py) with
`RileTreeProgram` + wiring the tree-aware metric into `dspy.teleprompt.GEPA`.
~50-line trainer edit. Requires a live vLLM endpoint to smoke-test.

**Finish**:
1. Add a new trainer `dspy_rile_tree_trainer` (next to `dspy_rile_trainer`).
2. Build a new oracle variant that yields `TreeExample`s whose
   `extra["scaffold"]` is a `RILETreeScaffold` (built via
   `build_rile_tree_scaffold` + `RILECorpusIndex`).
3. In the trainer, for each example: run `RileTreeProgram().rollout(scaffold)`
   → `TreeRilePredictions` → `dspy_gepa_metric_from_rollout(rollout)` →
   `ScoreWithFeedback`.
4. Register via `TrainerConfig(extra={"optimizer": "gepa_tree"})`.
5. Smoke: run with a small student (gemma-4-31B-NVFP4, port 8010) on a
   granularly-coded manifesto and confirm the feedback text mentions C1/C2/C3
   components.

### 3. TRL GRPO tree trainer blocked on vLLM `--enable-lora` + NVFP4
[`trl_grpo_rewards_from_rollouts`](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/rile_tree_feedback.py) is the drop-in reward function.
What's NOT done: the `grpo_tree_trainer` that (a) for each doc, runs the
base+LoRA student tree-wise via vLLM to produce `TreeRilePredictions`
rollouts, (b) hands the per-rollout rewards to `train_grpo`. Gated on
whether vLLM's `--enable-lora` flag is compatible with NVFP4-quantized
base weights (known open question from the earlier Tree-RL plan).

**Finish**:
1. Verify vLLM `--enable-lora` on NVFP4 (try `gemma-4-31B-IT-NVFP4`
   running on port 8010 with a dummy LoRA adapter). If not, fall back to
   fp16 `gemma-4-31B-it` already cached under `/mnt/data/models/`.
2. Build `grpo_tree_trainer` in
   `parallel/unified_g_v1/src/unified_g_v1/training/trainers/grpo_tree.py`:
   - For each training example, chunk + build `RILETreeScaffold`.
   - Run the student model (base + f_adapter + g_adapter) tree-wise.
   - Assemble `TreeRilePredictions`.
   - Reward function: `trl_grpo_rewards_from_rollouts(rollouts, local_law_weight=0.3)`.
   - Dispatch to `src/training/trl_training.py::train_grpo` via
     `TrainerConfig(mode="grpo", reward_funcs=..., model_name=...)`.
3. Register in [`trainers/__init__.py`](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/__init__.py) so
   `TrainerConfig(mode="grpo_tree", ...)` routes to it.
4. Smoke: 20 docs × K=2 rollouts × 10 GRPO steps, assert LoRA adapter
   state_dict changes and per-rollout rewards track the combined
   root+C1+C2+C3 signal.

### 4. Markov multi-stage training schedule
`tree_training_schedule="two_stage"` (stage1 + stage2 epochs) is used in
the legacy `run_markov_changepoint_ops_count_experiment` but the new
`fit()`-path `markov_changepoint_task(...)` only supports single-stage.
The cheap fix: add an optional `epoch: int | None = None` kwarg to
`TreeObjective.compute_loss` and thread the current epoch through
`run_pytorch_training`, then make `MarkovChangepointObjective` switch
weight configs at a configured cutoff. Not blocking any current experiment.

### 5. Adaptive audit sampling
Legacy Markov can re-sample audit nodes per epoch. The new path uses a
fixed audit set pre-computed at oracle construction. Not used by any
current experiment; would need per-epoch example-mutation hooks in the
generic loop.

### 6. Parity test vs. legacy Markov
Smoke tests exist ([test_markov_fit.py](../parallel/unified_g_v1/tests/test_markov_fit.py))
but no strict parity test between `fit()`-path and
`run_markov_changepoint_ops_count_experiment` at a fixed seed. Good
next-session task: fix seed=42, smoke config, assert
`|fit_root_mae - legacy_root_mae| / legacy_root_mae < 0.05`. Would
require threading the legacy path's `effective_data_seed` /
`effective_model_seed` through the new oracle/model.

## How to pick up from here

1. **If the user wants DSPy-GEPA tree training**: Start with the existing
   [`dspy_rile_trainer`](../parallel/unified_g_v1/src/unified_g_v1/training/trainers/dspy_rile.py) — it's a working flat baseline.
   Clone as `dspy_rile_tree_trainer` and swap in `RileTreeProgram` +
   `dspy_gepa_metric_from_rollout`. Use [`gemma-4-31B-IT-NVFP4`](http://localhost:8010) (running on port 8010, max_model_len=65536) as the student.

2. **If the user wants TRL GRPO tree training**: Verify vLLM --enable-lora
   compatibility with NVFP4 first, then build `grpo_tree_trainer`. The
   `rile_tree_feedback.trl_grpo_rewards_from_rollouts` function is the
   integration point — it takes `Sequence[TreeRilePredictions]` and
   returns per-rollout reward floats.

3. **If the user wants the bundle path to enforce C2**: port
   `run_embedding_fno_training` onto `fit()` + `ManifestoRileEmbeddingObjective`.

4. **If the user wants Markov multi-stage**: add `epoch` kwarg to
   `TreeObjective.compute_loss` + route it through `run_pytorch_training`.

5. **If the user questions parity**: write the seed=42 parity test described
   in "Known limitations #6".

## Key commits/state at handoff

- Branch: `main`
- Key new modules: `src/tasks/manifesto/rile_codes.py`,
  `parallel/unified_g_v1/src/unified_g_v1/markov/tree_task.py`,
  `parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree.py`,
  `parallel/unified_g_v1/src/unified_g_v1/realdoc/rile_tree_program.py`,
  `parallel/unified_g_v1/src/unified_g_v1/training/trainers/rile_tree_feedback.py`.
- Tests: 107/107 passing; representative commands:
  - `./venv/bin/python -m pytest -q parallel/unified_g_v1/tests`
  - `./venv/bin/python -m pytest -q parallel/unified_g_v1/tests/test_markov_fit.py parallel/unified_g_v1/tests/test_sketch_local_laws.py parallel/unified_g_v1/tests/test_manifesto_rile_codes.py parallel/unified_g_v1/tests/test_rile_tree_reward.py parallel/unified_g_v1/tests/test_rile_tree_feedback.py parallel/unified_g_v1/tests/test_rile_tree_program.py`

## Outstanding (not started)

- **FNO sweep collation** — 20 runs (4 embeddings × 5 sizes) from an earlier
  session, final numbers still need to be harvested into a comparison table.
- **Lean parity** — the new objectives should eventually map to their Lean
  `DiscountedTreeMetaObjective` counterparts; currently we match the formula
  structure but haven't proven equivalence.

## Principles to carry forward

- **Every piece must have local laws.** The `forward_aux` channel on
  `TreeObjective.compute_loss` is the load-bearing interface that made this
  work — any new model that computes per-node losses in the forward pass
  should push them through this channel.
- **Strict targets over soft fallbacks.** When the data supports per-node
  labels (Manifesto codings, Markov changepoints, analytic bigram counts),
  use them. Soft pull-to-root laws are an acceptable fallback when data is
  missing, never a default when data exists.
- **Single balance knob.** `local_law_weight` (λ) ∈ [0, 1] controls the
  root-vs-local-law balance; `*_relative_weight` (ρ) controls the
  distribution inside the local-law block. Default λ=0.3 matches the
  existing Markov publication config; default ρs=1.0 each is uniform
  distribution.
- **Honest scope.** The DSPy-GEPA and TRL-GRPO paths have the
  infrastructure (tree program, per-node rewards) built and unit-tested,
  but the trainers themselves aren't wired to live LLMs yet — that's a
  ~1-day piece of work per path plus infrastructure verification.
