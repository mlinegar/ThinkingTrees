# AGENTS.md - Quick Reference for AI Agents

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
# Small model: GPUs 0,1 → Port 8000 (Nemotron-30B-FP8)
# Large model: GPUs 2,3 → Port 8001 (GenRM-NVFP4-235B)
```

### Single Server Options

```bash
./scripts/start_dual_servers.sh --small-only  # Just port 8000
./scripts/start_dual_servers.sh --large-only  # Just port 8001
./scripts/start_vllm.sh <profile>             # Specific model profile
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
  --output-dir outputs/train_$(date +%Y%m%d_%H%M) \
  --train-samples 100 \
  --val-samples 30 \
  --test-samples 30 \
  --enable-genrm \
  --genrm-port 8001 \
  --use-mini-trees \
  --optimizer bootstrap_random_search \
  --optimizer-budget heavy \
  --n-iterations 2
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `start_dual_servers.sh` | Start both small (8000) and large (8001) models |
| `run_training_pipeline.sh` | Domain-agnostic training with DSPy optimization (default: manifesto_rile) |
| `start_vllm.sh <profile>` | Start single model (reads config/settings.yaml) |
| `stop_small_servers.sh` | Gracefully stop servers |

---

## File Map

```
src/
├── core/
│   ├── data_models.py      # Node, Tree, AuditStatus/Result
│   ├── llm_client.py       # LLMConfig, LLMClient (vLLM/SGLang/OpenAI)
│   ├── signatures.py       # DSPy signatures (RecursiveSummary, OracleJudge, etc.)
│   ├── strategy.py         # SummarizationStrategy, DSPyStrategy
│   ├── batch_processor.py  # Batch processing utilities
│   └── output_parser.py    # Case-insensitive LLM output parsing
│
├── tree/
│   ├── builder.py          # TreeBuilder, BuildConfig
│   ├── labeled.py          # LabeledNode, LabeledTree, LabeledDataset
│   └── verification.py     # TreeVerifier, OracleNodeVerifier
│
├── audit/
│   ├── auditor.py          # Auditor, AuditConfig, ReviewQueue
│   └── ops_checks.py       # CheckType, CheckConfig, CheckResult
│
├── training/
│   ├── preference/         # PreferencePair, PreferenceCollector, GenRMJudge
│   ├── optimization/       # OptimizerRegistry, GEPA, MIPRO, Bootstrap
│   ├── metrics/            # Training metrics
│   └── judge_optimization.py  # JudgeOptimizer
│
├── tasks/
│   ├── base.py             # TaskPlugin, AbstractTask
│   ├── registry.py         # TaskRegistry
│   └── manifesto/          # Manifesto/RILE task (default)
│
├── manifesto/
│   ├── data_loader.py      # ManifestoSample, ManifestoDataset, splits
│   ├── ops_pipeline.py     # ManifestoOPSPipeline, PipelineConfig
│   ├── evaluation.py       # ManifestoEvaluator, metrics
│   └── training_integration.py  # TrainableManifestoPipeline
│
└── preprocessing/
    └── chunker.py          # DocumentChunker (token-based)

experiments/manifesto_rile/
├── run_training_pipeline.py    # Main training entry point
├── collect_preferences.py      # Preference data collection
└── generate_synthetic_data.py  # Synthetic data generation

scripts/
├── start_dual_servers.sh       # Start small+large models
├── run_training_pipeline.sh    # Bash wrapper for training
├── stop_small_servers.sh       # Graceful shutdown
└── start_vllm.sh               # Generic model launcher

config/
└── settings.yaml               # Model configs, generation params, all settings

data/
├── raw/manifesto_project_full/ # Raw Manifesto corpus
├── results/manifesto_rile/     # Experiment results
└── checkpoints/                # Model checkpoints
```

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
| `--enable-genrm` | - | Use GenRM for preference learning |
| `--optimizer` | gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot | Optimization algorithm |
| `--optimizer-budget` | light, medium, heavy | Optimization intensity |
| `--n-iterations` | 1, 2+, 0 | 1=single-pass, 2+=iterative, 0=until convergence |

---

## Models Reference

| Model | Profile | GPUs | Port | Use Case |
|-------|---------|------|------|----------|
| Nemotron-30B-FP8 | `nemotron-30b-fp8` | 0,1 | 8000 | Default inference |
| GenRM-NVFP4-235B | `genrm-nvfp4` | 2,3 | 8001 | Preference scoring |
| Qwen3-80B | `qwen-80b` | 2 GPUs | - | Large training target |
| Qwen3-30B-Thinking | `qwen-30b-thinking` | 4 GPUs | - | Draft model |

---

## Troubleshooting

### Check if servers are running

```bash
curl http://localhost:8000/v1/models  # Small model
curl http://localhost:8001/v1/models  # Large model (GenRM)
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
| Slow startup | GenRM can take 2-3 min to load. Check logs for "Warmup complete" |
| Port already in use | Kill existing process with `./scripts/stop_small_servers.sh --all` |

### Resume interrupted training

```bash
./scripts/run_training_pipeline.sh --resume  # Continues from last checkpoint
```
