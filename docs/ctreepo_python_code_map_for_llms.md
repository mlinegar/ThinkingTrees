# C-TreePO Python Code Map for LLM Handoff

Last source audit: 2026-04-23.

This document is a code-derived map of the current ThinkingTrees worktree for
LLM agents and engineers. It emphasizes the Python code that supports C-TreePO,
the end-to-end training stack, optimizer behavior, token-budget handling, and
the high-value scripts that drive the paper experiments.

This is an audit and navigation document only. It records potential issues and
inconsistencies, but does not correct them.

## Ground Rules For Future LLMs

- Treat the current source tree as authoritative. Older README/file-map text is
  useful background, but it is no longer complete.
- Do not assume a clean git tree. This audit was run against a heavily modified
  worktree with many untracked files.
- Prefer `rg`, `rg --files`, AST parsing, and direct file inspection before
  changing code.
- Keep C-TreePO and Semantic Forests distinct when reasoning about scope:
  C-TreePO is the certification/theory/simulation/training ladder; Semantic
  Forests is the systems/runtime/preference-training layer.
- When auditing token budgets, distinguish completion-token caps, sequence
  lengths, prompt lengths, chunk budgets, leaf-size budgets, and embedding input
  budgets. They are not the same knob.

Inventory from this pass:

- `src/core`: 42 Python files
- `src/tree`: 48 Python files
- `src/ctreepo`: 152 Python files
- `src/training`: 84 Python files
- `src/tasks/manifesto`: 29 Python files
- `src/preprocessing`: 8 Python files
- `src/runtime`: 11 Python files
- `src/datasets`: 5 Python files
- `treepo/src/treepo`: 40 Python files
- `scripts`: 359 Python files and 64 shell files
- `tests`: 266 Python files
- AST sweep over `src`, `scripts`, `tests`, `treepo/src`, and `treepo/tests`:
  1116 files parsed, 1 parse error.

## Reading Order

Use this order when another LLM needs to understand or safely modify the repo.

1. `AGENTS.md` for local commands, model/server assumptions, and paper scope.
2. This document for the Python map and known inconsistencies.
3. `src/ctreepo/alternating.py`, `src/ctreepo/fg_arity.py`,
   `src/ctreepo/dspy_family.py`, `src/ctreepo/trl_family.py`, and
   `src/ctreepo/fno_family.py` for the f/g ladder contract.
4. `src/training/run_pipeline.py`, `src/training/config.py`, and
   `src/training/optimization/` for the DSPy training pipeline and optimizer
   audit behavior.
5. `src/tasks/manifesto/` for the RILE/Benoit running example, scorers, teacher
   traces, and law-stress benchmark.
6. `src/tree/` and `src/training/ctreepo_trainer.py` for neural-tree and
   PyTorch/FNO training.
7. `scripts/` entry points only after identifying the workflow category below.
8. `tests/` for pinned behavior and expected compatibility surfaces.

## Repo Faces

The repo currently serves two related but different paper tracks.

| Face | Main purpose | Primary code |
| --- | --- | --- |
| C-TreePO | Fixed tree structure, local laws, IPW/certification, formal/simulation evidence, f/g alternation, manifesto and Markov examples | `src/ctreepo`, `src/tree`, `src/tasks/manifesto`, `treepo/src/treepo`, `lean3/FormalProofs`, many `scripts/run_*` and `scripts/report_*` files |
| Semantic Forests | Adaptive/windowed systems stack, batching, runtime evaluation, preference training, feedback infra, multi-server orchestration | `src/core`, `src/preprocessing`, `src/runtime`, `src/training`, `src/feedback`, `src/pipelines` |

Shared infrastructure includes tree data models, builders, strategy wrappers,
LLM clients, preprocessing, batch execution, logged supervision, optimizer
wrappers, and dataset plugins.

## Package Map

| Path | Responsibility | Notes |
| --- | --- | --- |
| `src/core/` | Core data models, LLM clients, batching, prompts, scoring, local-law data types, inference surfaces, runtime capability metadata | This is the shared systems layer. Key files include `data_models.py`, `llm_client.py`, `strategy.py`, `batch_processor.py`, `batch_orchestrator.py`, `ops_checks.py`, `prompting.py`, `scoring.py`, `engines.py`, `inference_engine.py`, `unified_runtime.py`, and supervision metadata helpers. |
| `src/preprocessing/` | Text chunking, token-budget chunking, exact leaf-size utilities, adaptive windows, window adapters, visual feedback | `chunker.py` is the general chunker. `leaf_size_utils.py` is the newer exact-token leaf-size layer for EmbeddingGemma-style token windows and no-truncation checks. |
| `src/tree/` | Tree builders, auditors, IPW, learned sketches, neural tree models, state-tree runners, packed execution, CTreePO model definitions | This is both the original OPS tree layer and the newer neural/state-tree layer. It includes classical/theory simulations and shared model code. |
| `src/ctreepo/` | C-TreePO package layer: f/g alternation, DSPy/TRL/FNO families, labeled-tree distillation, optimizer records, simulation suites, report/plot CLIs | This is the most important package for the current C-TreePO end-to-end work. |
| `src/training/` | Main training pipeline, DSPy optimizer wrappers, TRL integration, generator trainers, CTreePO trainer, TreeModel V2 trainer, supervision/preference surfaces | `run_pipeline.py` is the large canonical pipeline. `optimization/` normalizes DSPy optimizer behavior and audits. `trl_training.py` bridges to HuggingFace/TRL. |
| `src/tasks/manifesto/` | RILE/Benoit domain task, policy dimensions, scorers, rubrics, teacher traces, law-stress generator/eval/proxy helpers | This is the main real-data example for C-TreePO. |
| `src/runtime/` | Long-context runtime backbone, benchmark adapters, tracing, memory, repair, verifier abstractions | Mainly Semantic Forests/runtime evaluation support. |
| `src/datasets/` | Dataset plugin registry and plugins for JSONL, Manifesto, PDF | Thin plugin layer used by training/eval scripts. |
| `src/pipelines/` | Batched document pipeline | `batched.py` is the high-throughput document pipeline around batch orchestration and scoring. |
| `src/feedback/` | Feedback collector/server/store types | Preference and feedback infrastructure. |
| `src/diffusion/` | Diffusion/generate-first experimental stack | Used by TreePO generate-first experiments. |
| `src/experiments/` | Experiment API and experiment runner helpers | Supports structured experiment runs. |
| `src/stats/` | Sampling utilities | PPS/systematic sampling support. |
| `src/harness.py` | Public TreeAudit-style harness | C-TreePO public-facing API surface. |
| `treepo/src/treepo/` | Smaller standalone benchmark/sketch package | Contains classical sketches, HLL, LDA simulations, report helpers, and `treepo-bench`. |

## Core Package Details

`src/core/data_models.py`

- Defines `Node`, `Tree`, `AuditStatus`, `AuditResult`, plus leaf/node helpers.
- This is the common tree payload shape used by builders, auditors, and
  pipelines.

`src/core/llm_client.py`

- Defines `LLMConfig`, `LLMClient`, `LLMResponse`, `MockLLMClient`, and factory
  helpers for vLLM, SGLang, OpenAI-compatible, and engine clients.
- This is lower level than DSPy. It is used when the code wants direct
  OpenAI-compatible API calls.

`src/core/strategy.py`

- Defines strategy abstractions for tree summarization:
  `SummarizationStrategy`, `BatchedStrategy`, `DSPyStrategy`,
  `CallableStrategy`, `TournamentStrategy`, and `GatedStrategy`.
- `DSPyStrategy` is what many pipelines use to wrap DSPy leaf/merge modules
  behind a common tree-building interface.

`src/core/batch_processor.py` and `src/core/batch_orchestrator.py`

- `batch_processor.py` owns async batched OpenAI-compatible request handling,
  routing policy parsing, multi-server clients, and batched audit helpers.
- `batch_orchestrator.py` owns global pipelined tree building across documents,
  degenerate-summary fallback handling, and bulk leaf/merge scheduling.

`src/core/ops_checks.py`

- Defines Lean-aligned local-law and audit data types, including `LawKind`,
  `AuditCheckKind`, `CheckType`, `CheckConfig`, `ApproxLocalLawsBundle`,
  `LawEvaluationRecord`, and `CheckResult`.
- This is the shared vocabulary for C1/C2/C3 style checks.

`src/core/prompting.py`, `src/core/signatures.py`, `src/core/scoring.py`

- `prompting.py` contains prompt builders, summary cleaning, score parsing, and
  optional LLM fallback parsing.
- `signatures.py` contains generic DSPy signatures and modules for recursive
  summarization and oracle judging.
- `scoring.py` defines bounded oracle/scorer types and metric adapters.

`src/core/engines.py` and `src/core/inference_engine.py`

- These files define backend-neutral engine surfaces across chat, diffusion, and
  symbolic execution.
- Symbolic Markov operations are registered through the inference-engine layer.

`src/core/unified_runtime.py`

- Shared batching/runtime primitives for LLM and neural-tree execution.
- Tracks GPU batch store keys, packed views, runtime modes, telemetry, and CPU
  fallback counts.

## Preprocessing And Tokenization

`src/preprocessing/chunker.py`

- `Chunker` supports direct `max_tokens` or context-manager-based chunk sizing.
- `chunk_for_ops_token_budget` creates deterministic token-budget chunks.
- `chunk_for_ops` chooses token-budget chunking when `max_tokens` is passed,
  otherwise it falls back to character/axis/sentence/paragraph/adaptive modes.
- This is the bridge between `--max-chunk-tokens` and tree construction.

`src/preprocessing/leaf_size_utils.py`

- `char_windows_from_token_budget` converts a text into exact non-overlapping
  character windows, each corresponding to `leaf_size_tokens` tokenizer tokens
  except the final tail window.
- `assert_no_truncation` explicitly errors if an embedding/LM call would silently
  truncate a text.
- This file is important for the newer size-token f/g ladder, where
  `leaf_size_tokens` is the canonical row axis.

`src/preprocessing/adaptive_windows.py` and `src/preprocessing/window_adapters.py`

- Adaptive windowing and modality adapters for text, pages, sequences, time
  segments, and visual regions.
- These files are mostly Semantic Forests scope, but their abstractions feed
  adaptive chunking experiments.

## Tree And Model Layer

`src/tree/builder.py`

- Builds OPS trees from chunks and summarizers.
- Defines `BuildConfig`, `BuildResult`, `TreeBuilder`, `IdentitySummarizer`,
  `ConcatenatingSummarizer`, `TruncatingSummarizer`, and async/sync build
  helpers.

`src/tree/auditor.py`, `src/tree/verification.py`, `src/tree/audit_serialization.py`

- `auditor.py` performs probabilistic audit sampling and confidence accounting.
- `verification.py` checks local oracle-node behavior.
- `audit_serialization.py` writes stable audit payloads.

`src/tree/ipw.py`, `src/tree/full_tree_ipw.py`, and simulations

- `ipw.py` maps tree samples to observation-unit kinds and computes
  propensity-aware estimates.
- `full_tree_ipw.py` handles layered/full-tree IPW accounting with separate
  document-level supervision.
- Several simulation files stress IPW coverage, toy chunk problems, mergeability,
  Markov changepoints, LDA, and learned sketches.

`src/tree/ctreepo_model.py`, `src/tree/embedding_tree.py`, `src/tree/packed_execution.py`

- `ctreepo_model.py` defines the learned mergeable-sketch model family:
  leaf projectors, merge modules, readout heads, checkpoint loading, and config
  inference.
- `embedding_tree.py` builds embedding-backed trees and forwards CTreePO models
  over leaf/merge structures.
- `packed_execution.py` packs embedding trees into tensor batches for GPU
  runtime execution.

`src/tree/core_model.py`, `src/tree/tree_model_v2.py`, `src/training/tree_model_v2_trainer.py`

- These define the newer TreeModel V2 surface: encoder backends, shared tree
  neural core, score-fiber configs, scalar readouts, and supervision targets.
- Use this path for model-family-agnostic neural tree work.

`src/tree/state_tree.py`, `src/tree/state_tree_runner.py`, `src/tree/state_tree_verifiers.py`

- Generic stateful TreePO operator execution.
- These are useful for symbolic/Markov demos where a state is merged rather than
  a text summary.

`src/tree/treepo_stack.py` and `src/tree/generate_prompting.py`

- Generate-first TreePO stack builder and prompt templates for `/generate`-style
  operator surfaces.

## C-TreePO Package

`src/ctreepo/alternating.py`

- Defines the shared f/g alternation trampoline.
- Iteration naming:
  - `k=0`: `fg`, no training.
  - `k=1`: `fgf`, train f.
  - `k=2`: `fgfg`, train g.
  - `k=3`: `fgfgf`, train f again.
- The central semantic rule is that when g is trained, the scoring/reward
  function is the current student f, not the teacher. This is the intended
  alternation signal.
- The `FamilyRuntime` protocol owns `train_f`, `train_g`,
  `score_roots_with_f`, and `validate_artifact`.
- The trampoline writes per-step checkpoints with stage labels, f/g artifacts,
  split metrics, and validation metadata.

`src/ctreepo/fg_arity.py`

- Defines the canonical f/g arity budget:
  - f consumes one leaf-sized state: `leaf_size_tokens`.
  - g consumes two child states: `2 * leaf_size_tokens`.
  - g may emit a verbatim two-child concatenation: `2 * leaf_size_tokens`.
- `check_two_child_lm_budget` enforces:
  `2 * leaf_size_tokens + max_completion_tokens + prompt_template_overhead_tokens <= lm_context_window_tokens`.
- `auto_g_output_tokens` refuses g output budgets smaller than
  `2 * leaf_size_tokens`.

`src/ctreepo/dspy_family.py`

- DSPy backend family for alternating f/g.
- f and g are DSPy programs saved as artifacts.
- `DSPyFamilyConfig` carries optimizer name, budget, LM config,
  `leaf_size_tokens`, `lm_context_window_tokens`, `max_completion_tokens`,
  prompt overhead, tokenizer path, policy dimension, and optional f init path.
- It performs both config-level two-child budget checks and record-level
  no-truncation checks before DSPy optimizer/LM calls.
- Supported optimizers in this family include `gepa`, `mipro`, and bootstrap
  few-shot variants.
- Bootstrap demo stacking is capped to reduce context overflow.
- `train_f` compiles f against score-regression records using a
  `1 - abs(pred - target)` style metric.
- `train_g` compiles g while scoring candidates through the current f and the
  known node target score, which is the intended alternating behavior.

`src/ctreepo/trl_family.py`

- TRL backend family scaffold for alternating f/g.
- Artifact convention:
  - f artifact is an HF scalar-regression model directory or
    `teacher_passthrough`.
  - g artifact is an HF causal-LM SFT/GRPO directory or `teacher_passthrough`.
- It checks the same two-child LM budget as DSPy through `fg_arity.py`.
- `train_f` shells out to `scripts/distill_ctreepo_students.py` with
  `--run-f-lm-regression`.
- `train_g` currently shells out with `--run-g-sft`.
- Non-passthrough `score_roots_with_f` is not yet wired. See flagged issues.

`src/ctreepo/fno_family.py`

- FNO backend family for alternating f/g.
- f and g share one `EmbeddingCoordinateFNOTreeRegressor` state dict; f training
  updates leaf/normalization/score-head parameters, while g training updates the
  merge FNO path.
- This path uses PyTorch/FNO budgets, not LM completion-token budgets.
- `leaf_size_tokens` is still the row axis, but embedding calls are governed by
  `embedding_max_length_tokens` and `effective_embedding_dim`.

`src/ctreepo/embedding_fno.py`

- Implements the embedding-coordinate FNO model.
- Leaf path is 1-channel-to-1-channel over embedding coordinates.
- Merge path is 2-channel-to-1-channel over embedding coordinates.
- Identity init makes the leaf path a residual identity and merge path an
  average plus residual correction.

`src/ctreepo/distillation.py`

- Builds and consumes labeled-tree artifacts.
- No teacher/scorer calls happen here. Callers must provide materialized labels.
- Routes training targets:
  - `tree_operator` with `ctreepo_embedding_tree`
  - `g` with `lm_sft`
  - `f` with `embedding_ridge_proxy`
  - `f` with `lm_scalar_regression`

`src/ctreepo/opt/`

- Generic optimizer-facing primitives and record adapters.
- Important files:
  - `collect.py`: collection helpers.
  - `records.py`: optimizer record schemas.
  - `preferences.py`: preference utilities.
  - `protocols.py`: optimizer protocols.
  - `sklearn_proxy.py`, `torch_proxy.py`: proxy learners.
  - `training_adapter.py`: bridges records into training code.

`src/ctreepo/sim/`

- Simulation suite for Markov, LDA, identifiable-zero, law-stress,
  full-document anchors, full-tree IPW, CPU megasweeps, and publication bundles.
- `sim/core/` contains DGPs, metrics, theorem-feature adapters, full-doc family
  contracts, and validation utilities.
- `sim/suite/` builds named experiment grids and policy bundles.
- `sim/cli/` exposes run, sweep, plot, and report entry points used by scripts.

## Training Package

`src/training/run_pipeline.py`

- Large canonical training pipeline.
- Owns CLI parsing for dataset sizes, optimizer selection, chunk sizes,
  scorer/teacher token caps, neural-operator settings, GEPA/MIPRO budgets,
  server setup, cache behavior, and output artifacts.
- `ContextSafeLM` wraps `dspy.LM` and retries context-window failures with
  reduced `max_tokens` or `max_completion_tokens`.
- `setup_dspy` reads model context windows and generation profiles, then builds
  `ContextSafeLM` or `LoadBalancedContextSafeLM`.
- Main tree processing uses `DSPyStrategy` and settings-derived summarizer
  generation defaults.
- Comparison-module training is deliberately always GEPA, independent of the
  main `--optimizer` flag.
- Neural-operator local-law scoring resolves score/teacher port, model,
  max-token, and temperature settings separately from DSPy prompt optimization.

`src/training/config.py`

- Defines `OptimizationConfig`.
- DSPy optimizer options include `auto`, `gepa`, `bootstrap`,
  `bootstrap_random_search`, `mipro`, and `labeled_fewshot`.
- Auto thresholds:
  - up to `bootstrap_threshold`: bootstrap
  - up to `random_search_threshold`: bootstrap random search
  - up to `mipro_threshold`: MIPRO
  - above `mipro_threshold`: GEPA
- MIPRO has optional example compaction/truncation controls.
- GEPA has budget and reflection-LM controls.

`src/training/optimization/`

- Normalizes DSPy optimizer wrappers and audit behavior.
- `registry.py` registers/auto-selects optimizers.
- `bootstrap.py` handles BootstrapFewShot and BootstrapFewShotWithRandomSearch.
  If random-search teleprompter import fails, it falls back to basic bootstrap
  and records `compile_status="fallback"`.
- `mipro.py` performs optional train/val example compaction and records
  `input_mutation_flags`.
- `gepa.py` wraps GEPA and feedback-aware metrics.
- `performance.py` defines optimizer audit records and classification logic:
  `works`, `unstable_search`, `data_limited`, `implementation_fallback`,
  `objective_mismatch`, `runtime_failure`, and `forced_control`.

`src/training/trl_training.py`

- TRL/HuggingFace integration for DPO, GRPO, reward models, scalar reward
  regression, and SFT.
- `TRLSequenceConfig` contains `max_length` and `max_prompt_length`.
- DPO passes both `max_length` and `max_prompt_length` into `DPOConfig`.
- SFT passes `config.sequence.max_length` into `SFTConfig`.
- GRPO builds prompt-only records and requires online `reward_funcs`.
- GRPO token-limit propagation is a flagged inconsistency below.

`src/training/generator_trainers.py`

- Higher-level generator trainer wrappers:
  DPO, SFT, GRPO, and bootstrap finetuning.
- Converts local config into `TRLTrainingConfig` for DPO/GRPO.
- Has its own direct SFT path that uses `SFTConfig(max_seq_length=...)`, which
  differs from the generic `trl_training.py` SFT path.

`src/training/ctreepo_trainer.py`

- PyTorch trainer for embedding-tree CTreePO models.
- Handles sparse local-law supervision, optimizer/scheduler construction, train
  steps, evaluation, checkpointing, and artifact writing.

`src/training/tree_model_v2_trainer.py`

- Shared trainer surface for TreeModel V2, with scalar targets, fiber-pair
  targets, group targets, auxiliary targets, and task adapters.

`src/training/supervision/`

- Canonical supervision data surface.
- Includes dense scalar/simplex/classical learners, preference/comparative
  derivation types, reward adapters, timing contracts, optimizer metadata, and
  JSONL artifact writers.

`src/training/preference/` and `src/training/judges/`

- Preference pair types, human/oracle/GenRM/large-DSPy judges, judge capability
  adapters, and batch judge variants.

## Manifesto Task Package

`src/tasks/manifesto/pipeline.py`

- Main RILE summarization/merge/score pipeline.
- Contains `compute_output_budget`, `_call_with_budget`, manifesto summarizer
  modules, unified g, merger, scorer, and pipeline orchestration.
- Explicitly passes per-call DSPy `config={"max_tokens": ...}` in several
  summarizer/scorer paths.

`src/tasks/manifesto/dspy_signatures.py`

- DSPy signatures and modules for RILE score prediction and pairwise summary
  comparison.
- `RILEScorer` and related modules use explicit max-token caps.

`src/tasks/manifesto/dimension_scorer.py`, `dimensions.py`, `joint_scorer.py`

- Benoit-style policy dimension scoring.
- Dimension scorer wraps per-dimension DSPy prediction and keeps optimized
  predictor compatibility.
- Joint scorer shares one predictor across six policy dimensions.

`src/tasks/manifesto/teacher_trace_generator.py` and `teacher_trace_eval.py`

- Build and evaluate real-anchor teacher traces.
- Used to create labeled summary/score traces from real manifestos.

`src/tasks/manifesto/lawstress_generator.py`, `lawstress_eval.py`,
`lawstress_bootstrap_metric.py`, `lawstress_proxy.py`

- Synthetic local-law stress benchmark for information extraction and RILE-like
  scoring.
- Supports summarize-only, score/judge, bootstrap metrics, and embedding-proxy
  evaluation.

`src/tasks/manifesto/expert_benchmarks.py`, `rile_codes.py`, `corpus_metrics.py`

- Load Benoit replication archive material, Manifesto Project codings, expert
  benchmarks, and corpus-level validity metrics.

## End-To-End Workflows

### Standard Training Pipeline

Primary entry point:

- `scripts/run_training_pipeline.sh`
- `src/training/run_pipeline.py`

Flow:

1. Shell wrapper parses server/model/training flags and forwards them to
   `src.training.run_pipeline`.
2. Pipeline configures DSPy with `setup_dspy`.
3. Data/task plugin loads examples, usually manifesto RILE unless overridden.
4. Documents are chunked with char or token budgets.
5. Trees are built through strategy wrappers.
6. Preferences/supervision are collected.
7. DSPy optimizer wrappers compile scorer/summarizer/comparison modules.
8. Output directories receive final stats, optimizer diagnostics, checkpoints,
   and task artifacts.

Important behavior:

- `--optimizer` controls the main DSPy optimizer path.
- Comparison-module training is always GEPA and records itself as a control.
- `--max-chunk-tokens` takes precedence over char chunking in the OPS chunker.
- DSPy generation caps are derived from model context and settings profiles,
  then passed to `ContextSafeLM`.

### Batched Manifesto RILE Example

Primary entry point:

- `scripts/run_manifesto_batched_example.py`

Flow:

1. Load selected manifesto IDs.
2. Chunk with a fixed chunk size.
3. Run batched leaf/merge summaries.
4. Score final summaries for RILE.
5. Report chunk stats and predicted RILE.

This is the paper-friendly fixed-chunking RILE path.

### Optimized Manifesto Example

Primary entry point:

- `scripts/run_manifesto_optimized_example.sh`

Flow:

1. Optimize scorer and leaf/merge summarizers.
2. Disable adaptive/honesty paths explicitly.
3. Run selected manifesto IDs on fixed chunking.
4. Report optimized RILE outputs.

### Teacher Trace And Labeled-Tree Distillation

Primary files:

- `scripts/generate_manifesto_teacher_traces.py`
- `scripts/distill_ctreepo_students.py`
- `src/tasks/manifesto/teacher_trace_generator.py`
- `src/ctreepo/distillation.py`

Flow:

1. Real manifesto anchors are expanded or summarized by a teacher/scorer.
2. Teacher trace records are written as JSONL.
3. Labeled trees attach node-level summaries and scores.
4. Distillation exports one or more student datasets:
   - g SFT records
   - f LM scalar-regression records
   - f embedding-proxy examples
   - CTreePO embedding-tree supervision
5. TRL, embedding proxy, or PyTorch trainers consume the exported surfaces.

### Alternating f/g Ladder

Primary files:

- `src/ctreepo/alternating.py`
- `src/ctreepo/dspy_family.py`
- `src/ctreepo/trl_family.py`
- `src/ctreepo/fno_family.py`
- `scripts/run_alternating_ladder.py`
- manifesto-specific f/g grid scripts

Flow:

1. Start with `(f_init, g_init)`.
2. Evaluate `k=0` as `fg`.
3. Odd iterations train f.
4. Even iterations train g.
5. Every family returns opaque artifacts and validates them.
6. Split metrics are evaluated through `score_roots_with_f`.

Backend differences:

- DSPy implements real alternating f/g optimization with DSPy programs.
- FNO implements real alternating f/g optimization over a shared neural state.
- TRL currently implements teacher-passthrough and SFT/scalar-regression
  subprocess paths, but not non-passthrough scoring or current-f GRPO.

### TRL/HuggingFace Routes

Primary files:

- `src/training/trl_training.py`
- `src/training/generator_trainers.py`
- `scripts/distill_ctreepo_students.py`
- `scripts/train_manifesto_summary_sft.py`
- `scripts/train_manifesto_summary_grpo.py`

Routes:

- DPO: preference/comparative data to `DPOTrainer`.
- GRPO: prompt-only data plus online reward functions to `GRPOTrainer`.
- SFT: prompt/completion records to `SFTTrainer`.
- Scalar reward: records to sequence-classification/scalar regression.
- TRL family: subprocesses through distillation script for g SFT and f scalar
  regression.

### PyTorch, FNO, And Neural Operators

Primary files:

- `src/tree/ctreepo_model.py`
- `src/tree/core_model.py`
- `src/tree/tree_model_v2.py`
- `src/training/ctreepo_trainer.py`
- `src/training/tree_model_v2_trainer.py`
- `src/ctreepo/fno_family.py`
- `src/ctreepo/embedding_fno.py`
- `scripts/train_ctreepo.py`
- `scripts/train_neural_operators.py`

Flow:

1. Build embedding-backed trees or dense supervision rows.
2. Attach node-level oracle/local-law targets.
3. Train CTreePO, TreeModel V2, or FNO-style modules.
4. Evaluate root prediction, node-level local laws, and paper metrics.

These paths do not consume LM completion-token caps directly. They consume
sequence/embedding budgets, `leaf_size_tokens`, model dimensions, batch sizes,
epochs, optimizer settings, and device/runtime configuration.

### Markov Publication And Tradeoff Pipelines

Primary files:

- `scripts/run_markov_publication_bundle.py`
- `scripts/run_markov_optimization_tradeoff_pipeline.py`
- `src/ctreepo/sim/suite/`
- `src/ctreepo/sim/core/`
- `scripts/long_job.py`

Flow:

1. TOML config defines run family, publication/iteration profile, supervision
   caps, and output root.
2. `--plan-only` previews the grid.
3. Detached mode writes launcher manifests and logs.
4. Reports aggregate oracle-budget, effective-training-docs, full-doc FNO,
   parity, and local-law diagnostics.

### Runtime Evaluation

Primary files:

- `scripts/run_runtime_eval.py`
- `src/runtime/`
- `src/core/unified_runtime.py`

Flow:

1. Benchmark adapter creates problem specs.
2. Backbone produces responses.
3. Verifier checks deterministic/local criteria.
4. Trace and metrics writers record unit outputs.

## Optimizer And Backend Matrix

### DSPy Optimizers

| Optimizer | Implemented where | Main behavior | Audit caveats |
| --- | --- | --- | --- |
| `gepa` | `src/training/optimization/gepa.py`, `src/ctreepo/dspy_family.py` | Reflective prompt optimization with metric feedback and optional reflection LM | Comparison-module training always uses GEPA regardless of `--optimizer`. |
| `mipro` | `src/training/optimization/mipro.py`, `src/ctreepo/dspy_family.py` | MIPROv2 instruction/demo optimization | May compact/truncate/drop optional fields and records `input_mutation_flags`. |
| `bootstrap` | `src/training/optimization/bootstrap.py` | Basic BootstrapFewShot demos | Fewer moving parts, mostly small-data path. |
| `bootstrap_random_search` | `src/training/optimization/bootstrap.py` | Random search over bootstrap programs | Falls back to `bootstrap` if the DSPy teleprompter is unavailable. |
| `labeled_fewshot` | `src/training/optimization/bootstrap.py` | LabeledFewShot-style demo selection | Can be recorded as `noop` if unavailable. |
| `auto` | `src/training/config.py`, optimizer registry | Chooses based on dataset-size thresholds | Verify thresholds before interpreting cross-run comparisons. |

Audit fields to preserve in reports:

- `optimizer_requested`
- `optimizer_used`
- `component`
- `dataset_size`
- `dataset_regime`
- `budget_mode`
- `seed`
- `compile_status`
- `skip_reason`
- `fallback_reason`
- `exception_summary`
- `metric_before`
- `metric_after`
- `heldout_gain`
- `train_gain`
- `input_mutation_flags`
- `comparison_control_flag`

### Backend Families

| Backend | f representation | g representation | Training semantics | Budget semantics |
| --- | --- | --- | --- | --- |
| DSPy family | DSPy scorer/program | DSPy summarizer/merge program | Real alternating path: f is refined; g is optimized against current f and target score fidelity | Uses `leaf_size_tokens`, LM context window, `max_completion_tokens`, prompt overhead, and record-level token guards. |
| TRL family | HF scalar-regression model or teacher passthrough | HF causal LM SFT/GRPO model or teacher passthrough | Scaffolded: f LM regression and g SFT subprocesses exist; non-passthrough scoring and current-f GRPO are not wired | Uses f/g arity guard plus TRL sequence lengths; GRPO token-limit propagation has a gap. |
| FNO family | Leaf FNO, normalization, score head | Merge FNO | Real alternating path over one shared PyTorch model state | Uses `leaf_size_tokens`, `embedding_max_length_tokens`, effective embedding dimension, epochs/batches/LR. |
| CTreePO trainer | Learned embedding-tree model | Learned merge modules | PyTorch training over local-law/sparse supervision | Uses embedding/tree budgets, not LM completion tokens. |
| Embedding proxy | Ridge/classical or torch proxy | Usually no generative g | Score proxy or supervision baseline | Uses embedding model max length and proxy feature dimensions. |
| Raw symbolic/neural operators | State or tensor operator | Merge operator | Markov/LDA/exact or learned operator simulations | Uses DGP/config budgets, support policies, and supervision/query caps. |

## Token And Budget Vocabulary

Do not collapse these knobs:

| Knob | Meaning | Main owners |
| --- | --- | --- |
| `max_tokens` | Usually LLM completion cap in OpenAI/DSPy calls; in chunkers, maximum tokens per chunk | DSPy LM, manifesto pipeline, chunker |
| `max_completion_tokens` | Completion cap name used by some APIs/providers | `ContextSafeLM`, f/g arity checks |
| `max_prompt_length` | Prompt-token cap for TRL/DPO-style training | `TRLSequenceConfig`, DPO path, standalone GRPO script |
| `max_completion_length` | Completion-token cap for GRPO generation | Standalone manifesto GRPO script |
| `max_length` | TRL/HF sequence length for tokenization or training | `TRLSequenceConfig`, DPO/SFT/scalar reward paths |
| `max_seq_length` | TRL SFTConfig name in some TRL versions | `generator_trainers.py`, standalone SFT script |
| `max_chunk_tokens` | Maximum tokens per document chunk/tree leaf for OPS chunking | `run_pipeline.py`, `chunker.py` |
| `leaf_size_tokens` | Canonical C-TreePO size-token axis; f consumes one, g consumes two | `fg_arity.py`, DSPy/TRL/FNO families |
| `embedding_max_length_tokens` | Maximum token length of text passed to embedding model | FNO/embedding-tree paths |
| `scorer_max_tokens` / `teacher_max_tokens` | Completion cap for scorer/teacher LLM calls | training pipeline, neural-operator local-law scoring, settings |

Enforcement by path:

- DSPy pipeline:
  - `ContextSafeLM` detects `max_tokens` or `max_completion_tokens`.
  - It retries context-window failures with reduced completion caps.
  - `setup_dspy` caps max tokens by model context and generation profile.
  - `DSPyStrategy` receives summarizer-profile max tokens.
- DSPy f/g family:
  - Constructor checks two-child arity.
  - Record-level checks count actual prompt fields before optimizer calls.
  - Bootstrap demo counts are capped to reduce overflow.
- TRL:
  - `TRLSequenceConfig` has `max_length` and `max_prompt_length`.
  - DPO passes both into `DPOConfig`.
  - SFT passes `max_length` in the generic path.
  - Generator-trainer SFT uses `max_seq_length`.
  - Generic GRPO currently does not pass sequence/prompt/completion caps into
    `GRPOConfig`; standalone manifesto GRPO does.
- FNO/PyTorch:
  - No LM completion cap.
  - Uses leaf-size, embedding max length, effective embedding dimension, batch
    size, epochs, LR, and device/runtime config.
  - Oversized embedding text should error through no-truncation checks rather
    than silently truncate.

## High-Value Scripts

The scripts directory is broad. Treat scripts as workflow entry points, not as
library code. Prefer importing functionality from `src/` when implementing new
features.

| Category | Representative scripts |
| --- | --- |
| Servers and launchers | `start_dual_servers.sh`, `start_vllm.sh`, `start_sglang.sh`, `start_embedding_server.sh`, `stop_small_servers.sh`, `long_job.py`, `spawn_detached_cmd.py`, `wait_for_long_job_then_run.py` |
| Main training | `run_training_pipeline.sh`, `train_ctreepo.py`, `train_neural_operators.py`, `distill_ctreepo_students.py`, `run_optimizer_performance_audit.py` |
| Manifesto/RILE | `run_manifesto_batched_example.py`, `run_manifesto_optimized_example.sh`, `generate_manifesto_teacher_traces.py`, `generate_manifesto_lawstress.py`, `eval_manifesto_lawstress.py`, `run_manifesto_fg_real_training_grid.py`, `run_manifesto_teacher_fg_leaf_grid.py`, `train_manifesto_summary_sft.py`, `train_manifesto_summary_grpo.py` |
| Markov | `run_markov_publication_bundle.py`, `run_markov_optimization_tradeoff_pipeline.py`, `run_markov_full_doc_anchor_ladder.py`, `run_markov_full_tree_ipw_grid.py`, `run_markov_capability_suites.sh`, `report_markov_capability_map.py` |
| LDA/identifiable/law-stress | `run_segmented_lda_ctreepo_simulation.py`, `run_lda_tree_recovery_simulation.py`, `run_identifiable_zero_*`, `build_identifiable_zero_*`, `report_identifiable_zero_*`, `report_law_stress.py` |
| Reports and plots | `generate_paper_simulation_report_bundle.py`, `report_method_compare.py`, `report_optimizer_performance_audit.py`, `plot_*`, `render_*`, `aggregate_*`, `summarize_*` |
| Runtime/eval/benchmark | `run_runtime_eval.py`, `benchmark_fno_scaling.py`, `benchmark_neural_operator_comparison.py`, `eval_ctreepo_crosslang.py`, `audit_manifesto_single_doc.py` |
| Supervision tutorials | `tutorial_supervision_00_*` through `tutorial_supervision_16_*` |

## Test Map

Use tests to understand intended behavior before changing source.

| Test area | What it pins |
| --- | --- |
| `tests/ctreepo/test_unified_fg_ladder_contract.py` | f/g arity, DSPy budget guard, TRL passthrough scoring, stage summaries |
| `tests/training/test_optimizer_performance_audit.py` | optimizer audit classification, fallback/noop behavior, MIPRO compaction flags |
| `tests/training/test_gepa_feedback_wrapping.py` | GEPA feedback metric wrapping |
| `tests/training/test_trl_grpo_records.py` | GRPO record construction and reward-context preservation |
| `tests/training/test_train_neural_operators_cli.py` | neural-operator CLI wiring |
| `tests/preprocessing/test_chunker.py` and `tests/preprocessing/test_leaf_size_utils.py` | chunk/token/leaf-size behavior |
| `tests/tasks/test_manifesto_*` | manifesto f/g grids, budget coverage, teacher traces, dimension fitting, unified g runtime |
| `tests/tree/test_*` | tree builder/auditor/IPW/neural/state-tree/full-doc behavior |
| `tests/core/test_unified_runtime.py`, `tests/core/test_logged_supervision.py` | runtime telemetry and logged supervision surfaces |
| `treepo/tests/` | standalone `treepo` package sketches, suites, and reports |

Current test gap to be aware of:

- No single broad test currently proves that one top-level "max tokens" knob
  propagates consistently across DSPy, TRL, and PyTorch/FNO. The existing tests
  cover pieces of that contract.

## Flagged Issues And Inconsistencies

These are findings from static inspection. They are not fixed here.

### 1. Syntax error in FNO convergence script

- Evidence: AST parse fails on `scripts/quick_fno_tree_convergence_study.py`
  with `IndentationError` at line 118.
- Source: imports at lines 111-116 are unindented after `args =
  parser.parse_args()`, while line 118 resumes indented function body code.
- Impact: this script cannot run or import.
- Suggested follow-up: move the imports inside `main()` or move the subsequent
  body back to top-level consistently, then run the script smoke test.

### 2. Package script entrypoint points at missing `src.main`

- Evidence: `pyproject.toml:80` defines
  `thinking-trees = "src.main:main"`.
- Evidence: `src/main.py` is absent; only top-level `main.py` exists, with
  `def main()` at `main.py:161`.
- Impact: an installed `thinking-trees` console script may fail with
  `ModuleNotFoundError: No module named 'src.main'`.
- Suggested follow-up: choose either `main:main` or add `src/main.py`, then test
  package installation/console-script invocation.

### 3. Top-level `main.py` has a likely missing import

- Evidence: `main.py:205` calls `logging.getLogger(__name__)`, but the imports
  at `main.py:8-21` do not include `import logging`.
- Impact: direct CLI use may raise `NameError` after dependency imports succeed.
- Suggested follow-up: add `import logging` if this entry point remains active.

### 4. TRL f/g family is not yet full alternating TRL

- Evidence: `src/ctreepo/trl_family.py:16-19` says only the passthrough case is
  end-to-end and k>=1 raises `NotImplementedError`.
- Evidence: `train_g` uses `--run-g-sft` and explicitly says it does not yet use
  f as a GRPO reward function (`src/ctreepo/trl_family.py:250-263`).
- Evidence: non-passthrough `score_roots_with_f` raises
  `NotImplementedError` at `src/ctreepo/trl_family.py:317-321`.
- Impact: TRL rows beyond teacher passthrough are not equivalent to DSPy/FNO
  alternating semantics. They are SFT/warmstart scaffold rows, not current-f
  reward optimization rows.
- Suggested follow-up: add a `--run-g-grpo` distillation route, wrap current f
  as a reward function, and implement HF g-generate plus f-score evaluation.

### 5. Generic TRL GRPO path does not propagate sequence/prompt/completion caps

- Evidence: `TRLSequenceConfig` defines `max_length` and `max_prompt_length` at
  `src/training/trl_training.py:110-114`.
- Evidence: DPO passes `max_length` and `max_prompt_length` into `DPOConfig` at
  `src/training/trl_training.py:1384-1399`.
- Evidence: generic GRPO builds `GRPOConfig` at
  `src/training/trl_training.py:1585-1598`, but no `max_length`,
  `max_prompt_length`, or `max_completion_length` appears there.
- Contrast: standalone manifesto GRPO script conditionally passes
  `max_prompt_length` and `max_completion_length` at
  `scripts/train_manifesto_summary_grpo.py:323-326`.
- Impact: "max tokens" may behave differently between DPO/SFT and generic GRPO,
  and between generic GRPO and standalone manifesto GRPO.
- Suggested follow-up: introspect `GRPOConfig` like the standalone script and
  pass supported prompt/completion/sequence caps.

### 6. TRL SFT config name differs across paths

- Evidence: generic SFT uses `SFTConfig(max_length=...)` at
  `src/training/trl_training.py:2049-2057`.
- Evidence: generator-trainer SFT uses `SFTConfig(max_seq_length=...)` at
  `src/training/generator_trainers.py:445-453`.
- Evidence: standalone manifesto SFT introspects and passes `max_seq_length` if
  supported at `scripts/train_manifesto_summary_sft.py:154-155`.
- Impact: behavior may depend on installed TRL version. One SFT path can fail or
  ignore the intended cap while another works.
- Suggested follow-up: centralize a TRL config-builder that introspects supported
  parameters once and is reused by all SFT paths.

### 7. DSPy family docstring conflicts with implementation comments

- Evidence: `src/ctreepo/dspy_family.py:21-24` says warmstart from a prior
  compiled iterate is not wired.
- Evidence: `train_f` says the current f program is passed to
  `optimizer.compile` and should never reset at
  `src/ctreepo/dspy_family.py:564-570`.
- Evidence: `train_g` says the current g program is the `program` argument to
  `optimizer.compile` at `src/ctreepo/dspy_family.py:682-688`.
- Impact: future readers may misunderstand whether DSPy alternation warmstarts.
- Suggested follow-up: audit actual artifact load/save behavior and update the
  module docstring or implementation comments to one consistent claim.

### 8. README/file-map is stale relative to current Python surface

- Evidence: current inventory found 438 Python files under `src`, 423 top-level
  script files, and 40 Python files under standalone `treepo/src/treepo`.
- Impact: agents that rely on the README alone will miss major current modules:
  f/g families, TreeModel V2, unified runtime, optimizer audit classification,
  law-stress utilities, publication bundles, and many simulation suites.
- Suggested follow-up: either link this document from the README or refresh the
  README map after stabilizing the current codebase.

## Verification Commands Used

AST sweep:

```bash
python3 - <<'PY'
import ast
from pathlib import Path

roots = [Path("src"), Path("scripts"), Path("tests"), Path("treepo/src"), Path("treepo/tests")]
files = []
for root in roots:
    if root.exists():
        files.extend(sorted(root.rglob("*.py")))

errors = []
for path in files:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append((str(path), type(exc).__name__, getattr(exc, "lineno", None), str(exc)))

print(f"parsed_files={len(files)}")
print(f"parse_errors={len(errors)}")
for item in errors:
    print("\t".join(map(str, item)))
PY
```

Result:

```text
parsed_files=1116
parse_errors=1
scripts/quick_fno_tree_convergence_study.py    IndentationError    118    unexpected indent (<unknown>, line 118)
```

Useful targeted searches:

```bash
rg -n "optimizer|optimizer-budget|max_chunk|max_tokens|max_completion|max_prompt|max_length|leaf_size|embedding_max|teacher_max|scorer_max" \
  src/training src/ctreepo src/preprocessing src/tasks/manifesto scripts config tests

rg -n "TODO|FIXME|HACK|NotImplementedError|deprecated|fallback|noop|forced_control|input_mutation|compile_status|skip_reason|fallback_reason" \
  src scripts tests treepo
```

## Practical Guidance For Future Changes

- For documentation-only audits, update this file and avoid changing code.
- For optimizer changes, update both runtime behavior and audit fields.
- For token-budget changes, add or update tests that cover DSPy, generic TRL,
  standalone TRL scripts, and PyTorch/FNO paths separately.
- For TRL fixes, prefer a single config-builder that introspects the installed
  TRL API and records which length arguments were accepted.
- For f/g alternation changes, verify all three families against the same
  `fg_arity.py` contract and the same stage naming in `alternating.py`.
- For paper-facing runs, prefer built-in detached launchers (`--detach` or
  `scripts/long_job.py`) and keep launcher manifests with outputs.
