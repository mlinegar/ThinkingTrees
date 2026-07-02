# AGENTS.md - Quick Reference for AI Agents

## Agent Reading Order

- Start here for local setup, server commands, common workflows, and repo-wide
  working assumptions.
- Use `docs/ctreepo_python_code_map_for_llms.md` as the detailed Python code
  map for C-TreePO, Semantic Forests, optimizer behavior, token-budget handling,
  and flagged audit issues.
- Use `docs/local_law_sampling_contract.md` before touching sampled local-law
  supervision, IPW weighting, node-query rates, or root/leaf/internal sampling.
  The canonical implementation is `treepo.training.local_law` in the official
  standalone repo at `/home/mlinegar/treepo` (this venv's `treepo` editable
  install points there; the old in-repo copies are archived at `OLD_treepo/`
  and `OLD_treepo_cld/`). Experiment-specific runners should pass observed
  masks, propensities, and node weights into that master objective rather
  than implementing bespoke IPW.
- `treepo` also ships shared research surfaces TT workflows should reuse:
  `treepo.viz.write_tree_visualization_html` (standalone expandable-tree HTML
  with sampling markers, gold/prediction labels, per-node `f` readouts,
  local-law losses, audit/certificate/tradeoff panels — see
  `~/treepo/docs/visualization.md` and `~/treepo/docs/tree_and_sampling.md`),
  `treepo.sampling.sample_node_audit`/`apply_node_audit` (uniform node-audit
  designs with logged `q/N` propensities), and
  `treepo.methods.TradeoffCurve` (the named error-vs-`leaf_unit_count`
  artifact). Prefer these over ad hoc HTML dumps, bespoke node sampling, or
  one-off leaf-grid CSVs.
- Treat that code map as source-audit documentation. Do not duplicate or
  overwrite it without re-running a source inventory, AST parse sweep, and
  targeted searches over the relevant pipeline/optimizer/token-budget paths.

---

## Environment Setup

```bash
source venv/bin/activate
python3
```

---

## Server Commands

### Start Dual Servers (Most Common)

```bash
./scripts/start_dual_servers.sh
# Small model: GPUs 0,1 → Port 8000 (Nemotron-30B-NVFP4)
# Large model: GPUs 2,3 → Port 8001 (Qwen3.5-397B-A17B-NVFP4 teacher)
```

### Single Server Options

```bash
./scripts/start_dual_servers.sh --small-only  # Just port 8000
./scripts/start_dual_servers.sh --large-only  # Just port 8001
./scripts/start_vllm.sh <profile>             # Specific model profile
./scripts/download_hf_model.sh Qwen/Qwen3-Embedding-8B  # Optional pre-download
./scripts/start_embedding_server.sh           # Multilingual embedding server (port from settings)
./scripts/start_vllm.sh qwen3-embedding-8b --port 8003  # (Legacy) embedding server launch
./scripts/start_vllm.sh diffusiongemma-26b-a4b-it-nvfp4 --port 8004 --cuda-devices 0 --gpu-mem 0.85
```

### Stop Servers

```bash
./scripts/stop_small_servers.sh        # Stops 8001, 8002, 30000 (keeps 8000)
./scripts/stop_small_servers.sh --all  # Stops ALL including 8000
```

---

## Common Workflow: Training Pipeline

```bash
./scripts/start_dual_servers.sh

./scripts/run_training_pipeline.sh \
  --task manifesto_rile \
  --output-dir outputs/train_$(date +%Y%m%d_%H%M) \
  --train-samples 100 \
  --val-samples 30 \
  --test-samples 30 \
  --max-chunk-chars 8000 \
  --use-mini-trees \
  --optimizer bootstrap_random_search \
  --optimizer-budget heavy \
  --n-iterations 2
```

## Common Workflow: Batched RILE Paper Example

```bash
# Labour 1983 + comparison manifestos, batched path, 8000-char chunks
python scripts/run_manifesto_batched_example.py \
  --ids 51320_198306 51620_198306 51320_199705 \
  --chunk-size 8000 \
  --port 8000
```

## Common Workflow: Optimized Batched RILE Example

```bash
# One command: optimize scorer + leaf/merge summarizers, then run selected IDs
# This runs fixed chunking with adaptive/honesty paths explicitly disabled.
./scripts/run_manifesto_optimized_example.sh \
  --ids 51320_198306 51620_198306 \
  --chunk-size 8000 \
  --train-samples 100 \
  --val-samples 30 \
  --port 8000
```

## Common Workflow: Method Stack Compare (Fast Smoke)

```bash
python scripts/run_method_compare.py \
  --output-root outputs/method_compare_$(date +%Y%m%d_%H%M%S)

python scripts/report_method_compare.py \
  --manifest outputs/method_compare_YYYYmmdd_HHMMSS/method_compare_manifest.json
```

## Common Workflow: LawStress Benchmark (MVP)

```bash
# Generate synthetic C1/C2/C3 stress benchmark for information extraction
# (records + JSONL docs; not a STEM/math/coding task generator)
python scripts/generate_manifesto_lawstress.py \
  --output-dir outputs/lawstress_mvp \
  --teacher-base-url http://localhost:8000/v1 \
  --teacher-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4

# Stage 1: summarize only (target small model)
python scripts/eval_manifesto_lawstress.py \
  --records outputs/lawstress_mvp/lawstress_records.jsonl \
  --output-dir outputs/lawstress_eval \
  --mode summarize_only \
  --summarizer-model qwen3.5-4b

# Stage 2: score-only pass with teacher scorer (GenRM disabled)
python scripts/eval_manifesto_lawstress.py \
  --records outputs/lawstress_mvp/lawstress_records.jsonl \
  --output-dir outputs/lawstress_eval \
  --mode score_and_judge_only \
  --scorer-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --disable-genrm
```

## Common Workflow: Teacher Trace Bootstrap (Real Anchors)

```bash
# Real manifesto -> score-preserving expansion -> summary traces
./scripts/start_vllm.sh qwen3.5-397b-a17b-nvfp4 --port 8000 --cuda-devices 0,1,2,3

python scripts/generate_manifesto_teacher_traces.py \
  --output-dir outputs/teacher_trace_bootstrap \
  --train-size 120 \
  --val-size 30 \
  --test-size 30 \
  --teacher-base-url http://localhost:8000/v1 \
  --teacher-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --scorer-base-url http://localhost:8000/v1 \
  --scorer-model /mnt/data/models/nvidia/Qwen3.5-397B-A17B-NVFP4
```

## Common Workflow: Markov Publication Bundle

```bash
# Preferred current workflow: use a versioned TOML config, inspect the plan,
# then launch detached so the run survives after the launching shell exits.
python scripts/run_markov_publication_bundle.py \
  --config config/markov/publication_bundle.iteration.toml \
  --plan-only

# Full overnight publication run:
# This includes the oracle-budget / effective-training-docs frontier in the
# tradeoff report, alongside the full-doc FNO reference and parity bundle.
python scripts/run_markov_publication_bundle.py \
  --config config/markov/publication_bundle.publication.toml \
  --detach \
  --output-root outputs/markov_publication_bundle_$(date +%Y%m%d_%H%M%S) \
  --no-reuse-existing

# To start a custom config from scratch:
python scripts/run_markov_publication_bundle.py \
  --write-config-template outputs/markov_publication_bundle.custom.toml

# Check status / tail logs / stop later via the launcher manifest
python scripts/long_job.py status \
  --job-root outputs/markov_publication_bundle_YYYYmmdd_HHMMSS/launcher

tail -f outputs/markov_publication_bundle_YYYYmmdd_HHMMSS/launcher/job.log

python scripts/long_job.py stop \
  --job-root outputs/markov_publication_bundle_YYYYmmdd_HHMMSS/launcher
```

## Common Workflow: Markov Tradeoff Pipeline

```bash
python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.iteration.toml \
  --plan-only

python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.publication.toml \
  --output-root outputs/markov_tradeoff_$(date +%Y%m%d_%H%M%S)
```

## Common Workflow: Markov Contextual-Sufficiency Exact-Zero

```bash
source venv/bin/activate
python -m pip install -e ".[contextual_sbi]"

XLA_PYTHON_CLIENT_PREALLOCATE=false ctreepo sim run contextual-sbijax \
  --data-source markov \
  --load-data-bundle outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json \
  --sbijax-trainer learned_local_laws \
  --sbijax-method nasss \
  --sbijax-package-theta markov_exact_sketch \
  --sbijax-input-encoding markov_exact_sketch \
  --train-docs 1024 \
  --val-docs 256 \
  --test-docs 256 \
  --fragment-len 1 \
  --context-samples-per-doc 1 \
  --response-signature-contexts 16 \
  --response-signature-slices 8 \
  --embedding-dim 32 \
  --state-dim 25 \
  --hidden-dim 128 \
  --learning-rate 0.0003 \
  --n-iter 1000 \
  --batch-size 128 \
  --local-law-weight 1.0 \
  --local-law-leaf-weight 1.0 \
  --local-law-merge-weight 1.0 \
  --local-law-idempotence-weight 1.0 \
  --local-law-contextual-weight 1.0 \
  --seed 0 \
  --output-root outputs/optimize_to_zero_demo
```

Current status (2026-05-05): `learned_local_laws` is the exact-zero path.
Package NASS/NASSS are approximate baselines; report `theta_mae`,
`theta_first_regime_accuracy`, `theta_last_regime_accuracy`, and law eps metrics
alongside contextual MAE. See
`docs/contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`.

Post-resolution ablations (2026-05-05) are summarized in
`docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md` with the
full table artifact at `outputs/markov_contextual_ablation_grid_report_20260505.md`.
Key result: NASSS can help as a low-weight auxiliary, learned merge/decoder
variants work inside the local-law lane, and standalone `CleanUnifiedNO`
general f/g does not yet discover the exact Markov law. Relevant runners:

```bash
scripts/run_optimize_to_zero_fg_architecture_ablation.sh 0
scripts/run_optimize_to_zero_laws_hybrid_grid.sh 3
scripts/run_clean_unified_fg_contextual_ablation.sh 2
```

Data-scaling + g-ablation (2026-05-06) extended the picture: 100× more train
docs (102400 vs 1024) closes most of the count_mae gap for the flexible
learner (0.82 → 0.027 at leaf=64 / regime_one_hot / count_only). The
architectural ceiling (`regime_transition_sum`) hits 0.0005 at 102400. The
g-side is *not* the bottleneck — across 20 g-axis cells (merge_family ×
merge_loss × decoder_head, plus rep_dim × FNO-as-g), count_mae stayed within
~2× of the best result. `decoder_head=linear` is a free 19% improvement.
Engineering: training step now mini-batches merge supervision (was OOMing at
102400), eval is chunked, N²-collision diagnostic subsamples to 4096 rows.
New flags: `--merge-family {mlp,fno_rep}`, `--decoder-head {mlp,linear}`,
`--local-law-merge-loss {mse,nass_jsd,nasss_jsd}`. Full handoff:
`docs/markov_data_scaling_g_ablation_handoff_2026-05-06.md`. Runners:

```bash
# Generate the 102400-doc bundle (one-time)
./venv/bin/python scripts/prepare_markov_hazard_panel_data.py \
  --panel-ids paper_hazard_panel_v1_t128 --train-docs 102400 --seed 0 \
  --bundle-root outputs/_bundles/markov_hazard_panels_train102400 \
  --skip-prepared-cache

# Sweeps
N_ITER=200 GPUS=0,2,3 bash scripts/run_markov_fno_round5_data_scaling.sh
GPUS=0,2,3 bash scripts/run_markov_fno_round6_g_ablation.sh
GPUS=0,2,3 bash scripts/run_markov_fno_round7_repdim_fno_g.sh
```

Per-rung batch sizes for leaf-token grid sweeps (added 2026-05-02): if you
set `supervision_recovery_leaf_token_ladder = [1024, 256, 64, 16]`, you can
also set `supervision_recovery_leaf_token_batch_sizes = "16=128;64=256;..."`
to override `supervision_batch_size` per leaf-tokens rung. Keys are
`fixed_leaf_tokens`, values are batch sizes in docs. Useful when smaller
leaves want larger doc batches to keep GPU saturated.

Performance rule (2026-05-02): `FNOCountSketch.forward_doc_unified` defaults
to `collect_full_trace=False` for the training/eval hot path. Telemetry
consumers must pass `collect_full_trace=True` explicitly. Do not add per-node
`.cpu()`/`.item()` calls inside the per-doc forward path - they re-serialize
GPU work and tank throughput on long merge chains (~9x speedup measured at
recoverable_v5_t2048 leaf=16). See "Performance: forward_doc_unified
collect_full_trace" subsection in `docs/ctreepo_python_code_map_for_llms.md`.

Head capacity (2026-05-03 conservative default): `unified_g_full_local_laws_v1`
and `comparison_grid_v3` use `state_dim=128, hidden_dim=512`. A bigger model
(`state_dim=2048, hidden_dim=2048, tree_merge_hidden_dim=4096`) was tested
and failed to crack the zero-merge ~2.14 root_mae floor on
`recoverable_v5_t2048`, plus made several composition cells worse
(full100 @ leaf=256: 1.06 -> 3.72). Don't bump state_dim/hidden_dim above
128/512 without re-validating; head capacity isn't the bottleneck for that
floor. See `docs/ctreepo_python_code_map_for_llms.md` "Head Capacity"
subsection.

## Long-Running Jobs

```bash
# Official rule: for long-running jobs, prefer the built-in detached launcher.
# Do not rely on ad hoc `nohup ... &` when a script supports --detach.

# Generic detached launch
python scripts/long_job.py launch \
  --name my_long_job \
  --job-root outputs/my_long_job_launcher \
  --cwd /home/mlinegar/ThinkingTrees \
  -- ./venv/bin/python some_script.py --arg value

# Inspect / stop by launcher manifest
python scripts/long_job.py status --job-root outputs/my_long_job_launcher
python scripts/long_job.py stop --job-root outputs/my_long_job_launcher
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `start_dual_servers.sh` | Start both small (8000) and large (8001) models |
| `run_training_pipeline.sh` | Domain-agnostic training with DSPy optimization (default: manifesto_rile) |
| `run_method_compare.py` | One-command profile matrix for LLM / embedding / neural-operator / generator comparisons |
| `report_method_compare.py` | Aggregates compare manifest into JSON + Markdown summaries |
| `run_manifesto_batched_example.py` | Batched manifesto-ID runner for paper examples (chunk stats + predicted RILE) |
| `run_manifesto_optimized_example.sh` | End-to-end optimized RILE example (train DSPy modules, then evaluate IDs) |
| `generate_manifesto_teacher_traces.py` | Real-anchor teacher trace generator (score-preserving expansion + 2-hop summaries) |
| `run_markov_publication_bundle.py` | Full Markov publication bundle; prefer `--config ... --plan-only`, then `--detach` |
| `run_markov_optimization_tradeoff_pipeline.py` | Optimized Markov tradeoff grid; prefer `--config ... --plan-only` |
| `long_job.py` | Official detached launcher for long-running jobs (`launch`, `status`, `stop`) |
| `start_vllm.sh <profile>` | Start single model (reads config/settings.yaml) |
| `stop_small_servers.sh` | Gracefully stop servers |

---

## File Map

```
src/
├── core/                                  # Shared + Semantic Forests orchestration
│   ├── data_models.py                     # Node, Tree, AuditStatus/Result
│   ├── llm_client.py                      # LLMConfig, LLMClient (vLLM/SGLang/OpenAI)
│   ├── strategy.py                        # SummarizationStrategy, DSPyStrategy, CallableStrategy
│   ├── scoring.py                         # OracleScore, ScoringOracle
│   ├── ops_checks.py                      # CheckType, CheckConfig, CheckResult
│   ├── batch_processor.py                 # [SF] Async batched LLM client + request pooling
│   ├── batch_orchestrator.py              # [SF] Global pipelined tree batching across documents
│   ├── gpu_orchestrator.py                # [SF] GPU resource management
│   └── vllm_runtime.py                    # [SF] vLLM-specific runtime adapter
│
├── tree/                                  # C-TreePO core + shared
│   ├── builder.py                         # TreeBuilder, BuildConfig, BuildResult
│   ├── auditor.py                         # [CT] Auditor, AuditConfig, C1/C2/C3 checking
│   ├── ipw.py                             # [CT] TreeIPW, Horvitz-Thompson, empirical Bernstein
│   ├── ipw_simulation.py                  # [CT] IPW coverage validation
│   ├── ipw_toy_problems.py                # [CT] Worst-case audit scenarios
│   ├── mergeable_ablation.py              # [CT] k-m phase, chunk-quality sweep
│   ├── learned_sketch.py                  # [CT] Neural sketch trained from oracle queries
│   ├── learned_sketch_simulation.py       # [CT] Learned sketch vs HLL comparison
│   ├── private_sfm_comparison.py          # [SF] Privacy-utility tradeoff
│   ├── verification.py                    # OracleNodeVerifier
│   └── labeled.py                         # LabeledNode, LabeledTree
│
├── preprocessing/                         # Fixed (shared) + adaptive (SF)
│   ├── chunker.py                         # chunk_for_ops (fixed path) + AdaptiveChunkingConfig
│   ├── adaptive_windows.py                # [SF] Coarse-to-fine windowing
│   ├── window_adapters.py                 # [SF] Modality-specific adapters
│   └── visual_feedback.py                 # [SF] Content-weighted feedback
│
├── runtime/                               # [SF] Runtime evaluation
│   ├── loop.py, contracts.py, backbone.py # Core runtime
│   ├── memory.py, verifier.py, repair.py  # Budget, verification, retry
│   └── adapters/                          # RULER, LongBench, ContextBench
│
├── training/                              # Mostly [SF]
│   ├── run_pipeline.py                    # [SF] Main training entry point
│   ├── trl_training.py                    # [SF] TRL-based DPO/GRPO
│   ├── preference/                        # [SF] PreferencePair, collection, GenRM
│   └── judge_optimization.py              # Judge model optimization
│
├── preference_collection/                 # [SF] Preference collection
│   ├── types.py, collector.py             # Preference request/response + collector protocol
│   └── server.py, store.py                # Preference API and durable store
│
├── stats/                                 # Shared
│   └── sampling.py                        # PPS, systematic sampling
│
├── tasks/
│   └── manifesto/                         # RILE example (C-TreePO running example)
│       ├── data_loader.py                 # ManifestoDataset, ManifestoSample
│       ├── pipeline.py                    # ManifestoPipeline (chunk→summarize→merge→score)
│       ├── rubrics.py                     # RILE_PRESERVATION_RUBRIC
│       ├── oracle.py                      # RILE similarity scorer
│       └── constants.py                   # RILE_MIN/MAX/RANGE
│
├── harness.py                             # [CT] TreeAudit public API
└── datasets/                              # Dataset plugins (manifesto, jsonl)

scripts/
├── run_manifesto_batched_example.py       # [CT] Batched manifesto-ID running example
├── run_ipw_ci_simulation.py               # [CT] IPW CI validation
├── run_mergeable_*.py                     # [CT] Theory simulations
├── run_learned_sketch_*.py                # [CT] Sec 7.4 learned sketch
├── plot_*.py                              # Paper figures
├── run_runtime_eval.py                    # [SF] Benchmark evaluation
├── run_training_pipeline.sh               # [SF] Full training
├── start_dual_servers.sh                  # Server management
└── start_vllm.sh                          # Model launcher

lean3/FormalProofs/                        # [CT] Machine-verified proofs (92 files)

# [CT] = C-TreePO scope, [SF] = Semantic Forests scope, unmarked = shared
```

---

## Paper Scope: C-TreePO vs Semantic Forests

This codebase serves two papers from a single shared repo.

### C-TreePO (Theory + Certification)

Fixed chunking, three local laws, probabilistic audit, formal proofs, controlled simulations.

| Component | Key Files |
|-----------|-----------|
| Three local laws (C1/C2/C3) | `src/tree/auditor.py`, `src/core/ops_checks.py` |
| IPW audit theory | `src/tree/ipw.py`, `src/tree/ipw_simulation.py`, `src/tree/ipw_toy_problems.py` |
| Mergeable ablations | `src/tree/mergeable_ablation.py` |
| Learned sketch (Sec 7.4) | `src/tree/learned_sketch.py`, `src/tree/learned_sketch_simulation.py` |
| Lean proofs | `lean3/FormalProofs/` (92 files) |
| Public harness | `src/harness.py` |
| Simulations | `scripts/run_ipw_ci_simulation.py`, `scripts/run_mergeable_*.py`, `scripts/run_learned_sketch_simulation.py` |
| RILE example | `src/tasks/manifesto/` (simple fixed-chunking path) |

### Semantic Forests (Systems + Scale)

Learned/adaptive chunking, multi-tree orchestration, runtime evaluation, preference training at scale.

| Component | Key Files |
|-----------|-----------|
| Adaptive chunking | `src/preprocessing/adaptive_windows.py`, `src/preprocessing/window_adapters.py` |
| Batch orchestration | `src/core/batch_orchestrator.py`, `src/core/batch_processor.py` |
| Runtime evaluation | `src/runtime/` |
| Training pipeline | `src/training/run_pipeline.py`, `src/training/trl_training.py`, `src/training/preference/` |
| Preference collection | `src/preference_collection/` |
| SFM comparison | `src/tree/private_sfm_comparison.py` |
| Benchmarks | `scripts/run_runtime_eval.py`, `scripts/run_training_pipeline.sh` |

### Shared Infrastructure

Tree builder (`src/tree/builder.py`), data models (`src/core/data_models.py`), strategy protocol (`src/core/strategy.py`), LLM client (`src/core/llm_client.py`), fixed chunker (`src/preprocessing/chunker.py` with `strategy="axis"`), sampling utilities (`src/stats/sampling.py`), config (`config/settings.yaml`), server scripts.

---

## Theory Docs

- `docs/ctreepo_python_code_map_for_llms.md` # Python code map, optimizer/token-budget matrix, and audit findings for LLM handoff
- `docs/treepo_preference_optimization.md` # TreePO formalization map (DPO/GRPO/PPO, sampling, DSL/IPW links)
- `docs/markov_sim_status.md` # Current Markov simulation status page: JAX local-law control, regime-one-hot recovery, CleanUnifiedNO/FNO bridge state, current artifacts, and recommended next runs.
- `docs/contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md` # Markov contextual-sufficiency resolution: `learned_local_laws` hits numerical zero; NASSS plateaus on its objective, not iterations. Judge sufficiency by `theta_first/last_regime_accuracy`, not contextual MAE alone.
- `docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md` # Post-resolution ablation handoff: JAX f/g architecture grid (`--law-architecture {analytic, learned_merge, learned_decoder, fully_learned}`, `--c2-merge-target {theta, self_consistency}`), NASSS+laws hybrid, CleanUnifiedNO general f/g. Includes Lean crosswalk and cross-architecture parity matrix.
- `docs/markov_fno_local_law_bridge.md` # Bridge-experiment design: tests whether the JAX `learned_local_laws` result transfers to the PyTorch CleanUnifiedNO FNO surface. PyTorch already exposes `markov_node_witness` (↔ JAX `c2_theta`) and `markov_local_laws_fno` (↔ JAX `c2_self_consistency`); experiment runs them on matched bundle/seeds with unified metric schema. Round 1 multi-leaf bridge campaign (8h, 48/52 cells) confirmed the bridge is not solved at root_mae≈1.94 best. Round 2 Stage 1 single-leaf diagnostic confirms FNO encoder is fine (boundary BCE F1≥0.99 at doc=32–64); pooling calibration and merge composition are the next suspects.

---

## Domain Plugins

The training framework supports pluggable domains for different use cases:

| Domain | Scale | Description |
|--------|-------|-------------|
| `manifesto_rile` | -100 to +100 | Political manifesto RILE scoring (default) |
| `summarization` | 0 to 1 | Generic summarization quality evaluation |

```python
# Using tasks programmatically
from src.tasks.registry import get_task, list_tasks

task = get_task("rile")
rubric = task.rubric
predictor = task.predictor_factory()
```

---

## Key Pipeline Flags

| Flag | Options | Description |
|------|---------|-------------|
| `--domain` | manifesto_rile, summarization | Domain plugin to use (default: manifesto_rile) |
| `--start-server` | - | Auto-start vLLM (default: requires running server) |
| `--enable-genrm` | blocked | Deprecated; use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM |
| `--optimizer` | gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot | Optimization algorithm |
| `--optimizer-budget` | light, medium, heavy | Optimization intensity |
| `--n-iterations` | 1, 2+, 0 | 1=single-pass, 2+=iterative, 0=until convergence |

---

## Models Reference

**Default: Always use NVFP4 quantized versions.** All models below are NVFP4 unless noted.

| Model | Profile | GPUs | Memory | Use Case |
|-------|---------|------|--------|----------|
| Qwen3-235B-A22B-Instruct | `qwen-235b` | 4 | ~34 GiB | Best quality inference (default) |
| Nemotron-30B-A3B | `nemotron-30b-nvfp4` | 2 | ~17 GiB | Fast inference |
| Qwen3-30B-A3B-Thinking | `qwen-30b-thinking` | 4 | ~17 GiB | Reasoning tasks |
| Qwen3-Next-80B-A3B-Instruct | `qwen-80b` | 2 | ~22 GiB | Mid-size inference |
| Qwen3.5-397B-A17B | `qwen3.5-397b-a17b-nvfp4` | 4 | ~95 GiB/GPU | Large teacher/scorer |
| GLM-4.6 | `glm-4.6` | 4 | ~47 GiB | Alternative large model |
| DiffusionGemma-26B-A4B-IT | `diffusiongemma-26b-a4b-it-nvfp4` | 1 | ~88 GiB at 262K context | Diffusion LLM smoke/evaluation; see `docs/diffusiongemma_vllm.md` |

---

## Troubleshooting

### Check if servers are running

```bash
curl http://localhost:8000/v1/models  # Small model
curl http://localhost:8001/v1/models  # Large model (teacher/scorer)
curl http://localhost:8003/v1/models  # Tiny embedding model (optional)
curl http://localhost:8004/v1/models  # DiffusionGemma (optional)
```

### View server logs

```bash
tail -f logs/small_model.log
tail -f logs/large_model.log
```

### Check GPU usage

```bash
nvidia-smi
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "Connection refused" | Server not running. Start with `./scripts/start_dual_servers.sh` |
| OOM errors | Stop other servers with `./scripts/stop_small_servers.sh --all` first |
| Slow startup | Large NVFP4 models can take 2-3 min to load. Check logs for "Warmup complete" |
| Port already in use | Kill existing process with `./scripts/stop_small_servers.sh --all` |

### Resume interrupted training

```bash
./scripts/run_training_pipeline.sh --resume  # Continues from last checkpoint
```
