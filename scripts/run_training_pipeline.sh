#!/bin/bash
# Training Pipeline for OPS (Oracle-Preserving Summarization)
# Runs iterative optimization: tree building + score prediction
#
# Usage:
#   ./scripts/run_training_pipeline.sh                        # Default (requires server running)
#   ./scripts/run_training_pipeline.sh --start-server         # Auto-start backend server
#   ./scripts/run_training_pipeline.sh --backend sglang --start-server
#   ./scripts/run_training_pipeline.sh --start-server --model qwen3.5-35b-a3b  # Use specific model
#   ./scripts/run_training_pipeline.sh --n-iterations 0       # Until convergence
#   ./scripts/run_training_pipeline.sh --max-metric-calls 10000  # Max out GPU usage
#   ./scripts/run_training_pipeline.sh --resume               # Resume from latest checkpoint
#   nohup ./scripts/run_training_pipeline.sh > training.log 2>&1 &
#
# Budget options: light, medium, heavy (default)
# For unlimited compute, use --max-metric-calls directly (e.g., 10000)
#
# FAST 8-HOUR CONFIG (for 4× GPU machine):
#   ./scripts/run_training_pipeline.sh \
#     --train-samples 30 --val-samples 15 --n-iterations 2 \
#     --num-threads 16 --concurrent-requests 80
#
# --start-server: Stops any running servers and starts a fresh backend instance.
# Use --backend vllm|sglang to choose runtime. Default model: nemotron-30b-nvfp4
# Available models include: nemotron-30b-nvfp4, qwen3.5-35b-a3b, qwen-80b, qwen-30b-thinking, qwen-235b, glm-4.6, olmo-32b-think
#
# --resume: Auto-finds the most recent run in the output directory and resumes
# from where it left off. Checkpoints are saved after:
#   - Phase 1: Document processing (train/val results)
#   - Phase 2: Training data creation (collector state)
#   - Each optimization round (classifier state + stats)

set -e

# ============================================================================
# Configuration (override with command line args or environment)
# ============================================================================
PORT=${PORT:-8000}
OPT_MODEL_PORT=${OPT_MODEL_PORT:-}  # Optional separate port for optimization model
BACKEND=${BACKEND:-vllm}
TASK_BACKEND=${TASK_BACKEND:-}
GENRM_BACKEND=${GENRM_BACKEND:-}
BACKEND_FALLBACK=${BACKEND_FALLBACK:-none}
SGLANG_VENV_PATH=${SGLANG_VENV_PATH:-/home/mlinegar/sglang-env}
TASK=${TASK:-manifesto_rile}        # Task plugin name (default: manifesto_rile)
DATASET=${DATASET:-}                # Dataset plugin name (default: settings.yaml datasets.default)
DATASET_PATH=${DATASET_PATH:-}      # File path for file-based datasets (e.g., jsonl)
TRAIN_SAMPLES=${TRAIN_SAMPLES:-50}
VAL_SAMPLES=${VAL_SAMPLES:-17}
TEST_SAMPLES=${TEST_SAMPLES:-17}
ROUNDS=${ROUNDS:-3}
CONCURRENT_DOCS=${CONCURRENT_DOCS:-20}
CONCURRENT_REQUESTS=${CONCURRENT_REQUESTS:-100}
MAX_CHUNK_CHARS=${MAX_CHUNK_CHARS:-4000}
MAX_CHUNK_TOKENS=${MAX_CHUNK_TOKENS:-}

# Optimizer settings
# Options: auto, gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot
# Budget options (for gepa/mipro): light, medium, heavy (or use MAX_METRIC_CALLS for direct control)
# Note: Default aligned with run_pipeline.py for consistency
OPTIMIZER=${OPTIMIZER:-gepa}
OPTIMIZER_BUDGET=${OPTIMIZER_BUDGET:-medium}
MAX_METRIC_CALLS=${MAX_METRIC_CALLS:-}  # Direct control (overrides budget)
NUM_THREADS=${NUM_THREADS:-16}  # Parallel metric evaluations (conservative default for local backend)
GEPA_LEAF_MERGE_SAMPLING_DESIGN=${GEPA_LEAF_MERGE_SAMPLING_DESIGN:-}
GEPA_IPW_ESTIMATOR=${GEPA_IPW_ESTIMATOR:-}
GEPA_IPW_MIN_PROPENSITY=${GEPA_IPW_MIN_PROPENSITY:-}
SCORER_MAX_TOKENS=${SCORER_MAX_TOKENS:-}
SCORER_TEMPERATURE=${SCORER_TEMPERATURE:-}
SCORER_STRICT_PARSE=${SCORER_STRICT_PARSE:-}
START_SERVER=${START_SERVER:-false}  # Auto-start task backend server
MODEL=${MODEL:-nemotron-30b-nvfp4}  # Model to use with --start-server
TASK_CUDA_DEVICES=${TASK_CUDA_DEVICES:-0,1}
TASK_TENSOR_PARALLEL=${TASK_TENSOR_PARALLEL:-}
GENRM_CUDA_DEVICES=${GENRM_CUDA_DEVICES:-2,3}

# Iterative optimization settings
# N_ITERATIONS: 1=single-pass oracle only, 2+=iterative (oracle→summarizer), 0=until convergence
N_ITERATIONS=${N_ITERATIONS:-1}
CONVERGENCE_THRESHOLD=${CONVERGENCE_THRESHOLD:-0.01}
CONVERGENCE_PATIENCE=${CONVERGENCE_PATIENCE:-3}
SKIP_SUMMARIZER_OPT=${SKIP_SUMMARIZER_OPT:-false}
SKIP_ORACLE_OPT=${SKIP_ORACLE_OPT:-false}

# Top-down initialization (oracle-aligned demo seeding from short docs)
USE_TOP_DOWN_INIT=${USE_TOP_DOWN_INIT:-false}
N_INIT_DEMOS=${N_INIT_DEMOS:-8}
MAX_INIT_PROMPT_TOKENS=${MAX_INIT_PROMPT_TOKENS:-${MAX_INIT_DOC_CHARS:-4000}}

# Resume from checkpoint
RESUME=${RESUME:-false}

# Dynamic GPU Allocation (vLLM sleep mode for efficient multi-model serving)
# When enabled, Python orchestrator manages servers with sleep/wake for ~6-12s transitions
# Uses all 4 GPUs in DP=2 mode for ~2x throughput during document processing
# Disable with --no-dynamic-gpu to use shell-managed single server
DYNAMIC_GPU=${DYNAMIC_GPU:-true}

# Preference-tree collection settings (modern local-law path)
START_GENRM=${START_GENRM:-false}
GENRM_PORT=${GENRM_PORT:-8001}
GENRM_MODEL=${GENRM_MODEL:-genrm-nvfp4}
GENRM_INIT_SAMPLES=${GENRM_INIT_SAMPLES:-8}
GENRM_INIT_CANDIDATES=${GENRM_INIT_CANDIDATES:-4}
PREFERENCE_INIT_SAMPLES=${PREFERENCE_INIT_SAMPLES:-}
PREFERENCE_INIT_CANDIDATES=${PREFERENCE_INIT_CANDIDATES:-}
PREFERENCE_TREE_CONCURRENCY=${PREFERENCE_TREE_CONCURRENCY:-}
PREFERENCE_SAMPLE_SEED=${PREFERENCE_SAMPLE_SEED:-}
PREFERENCE_INCREMENTAL_SAMPLING=${PREFERENCE_INCREMENTAL_SAMPLING:-}
PREFERENCE_JUDGE_BACKEND=${PREFERENCE_JUDGE_BACKEND:-}
PREFERENCE_TIE_MARGIN=${PREFERENCE_TIE_MARGIN:-}
TRAIN_COMPARISON_MODULE=${TRAIN_COMPARISON_MODULE:-false}

# Adaptive + honest chunking controls (CLI overrides settings.yaml)
ADAPTIVE_CHUNKING=${ADAPTIVE_CHUNKING:-}
ADAPTIVE_CHUNK_MIN_CHARS=${ADAPTIVE_CHUNK_MIN_CHARS:-}
ADAPTIVE_CHUNK_MAX_CHARS=${ADAPTIVE_CHUNK_MAX_CHARS:-}
ADAPTIVE_PROXY_BLEND=${ADAPTIVE_PROXY_BLEND:-}
ADAPTIVE_CROSSFIT_FOLDS=${ADAPTIVE_CROSSFIT_FOLDS:-}
ADAPTIVE_EMBEDDING_PROXY=${ADAPTIVE_EMBEDDING_PROXY:-}
ADAPTIVE_EMBEDDING_API_BASE=${ADAPTIVE_EMBEDDING_API_BASE:-}
ADAPTIVE_EMBEDDING_MODEL=${ADAPTIVE_EMBEDDING_MODEL:-}
ADAPTIVE_EMBEDDING_HEAD_METHOD=${ADAPTIVE_EMBEDDING_HEAD_METHOD:-}
ADAPTIVE_EMBEDDING_HEAD_EPOCHS=${ADAPTIVE_EMBEDDING_HEAD_EPOCHS:-}
ADAPTIVE_EMBEDDING_HEAD_LR=${ADAPTIVE_EMBEDDING_HEAD_LR:-}
ADAPTIVE_EMBEDDING_HEAD_WEIGHT_DECAY=${ADAPTIVE_EMBEDDING_HEAD_WEIGHT_DECAY:-}
ADAPTIVE_EMBEDDING_RETRAIN_ROUNDS=${ADAPTIVE_EMBEDDING_RETRAIN_ROUNDS:-}
ADAPTIVE_EMBEDDING_SCORE_KEY=${ADAPTIVE_EMBEDDING_SCORE_KEY:-}
ADAPTIVE_EMBEDDING_FULL_FINETUNE=${ADAPTIVE_EMBEDDING_FULL_FINETUNE:-}
ADAPTIVE_EMBEDDING_FINETUNE_COMMAND=${ADAPTIVE_EMBEDDING_FINETUNE_COMMAND:-}
EMBEDDING_PROXY_FAIL_ON_ERROR=${EMBEDDING_PROXY_FAIL_ON_ERROR:-}
RERUN_EMBEDDING_PROXY_ON_RESUME=${RERUN_EMBEDDING_PROXY_ON_RESUME:-}
TRAIN_NEURAL_OPERATORS=${TRAIN_NEURAL_OPERATORS:-}
NEURAL_OPERATORS_WHICH=${NEURAL_OPERATORS_WHICH:-}
NEURAL_OPERATORS_OUTPUT_DIR=${NEURAL_OPERATORS_OUTPUT_DIR:-}
NEURAL_OPERATORS_CTREEPO_ARGS=${NEURAL_OPERATORS_CTREEPO_ARGS:-}
NEURAL_OPERATORS_MERGEABLE_ARGS=${NEURAL_OPERATORS_MERGEABLE_ARGS:-}
NEURAL_OPERATORS_CTREEPO_SEARCH_SPEC=${NEURAL_OPERATORS_CTREEPO_SEARCH_SPEC:-}
NEURAL_OPERATORS_MERGEABLE_SEARCH_SPEC=${NEURAL_OPERATORS_MERGEABLE_SEARCH_SPEC:-}
NEURAL_OPERATORS_FAIL_FAST=${NEURAL_OPERATORS_FAIL_FAST:-}
NEURAL_OPERATORS_FAIL_ON_ERROR=${NEURAL_OPERATORS_FAIL_ON_ERROR:-}
RERUN_NEURAL_OPERATORS_ON_RESUME=${RERUN_NEURAL_OPERATORS_ON_RESUME:-}
NEURAL_OPERATORS_AUTO_WIRE_REPRESENTATION=${NEURAL_OPERATORS_AUTO_WIRE_REPRESENTATION:-}
HYBRID_ORACLE_SEEDED_ENSEMBLE=${HYBRID_ORACLE_SEEDED_ENSEMBLE:-}
HYBRID_SEED_LLM_MIN_WEIGHT=${HYBRID_SEED_LLM_MIN_WEIGHT:-}
HYBRID_SEED_LLM_MAX_WEIGHT=${HYBRID_SEED_LLM_MAX_WEIGHT:-}
HYBRID_OPERATOR_BOOST=${HYBRID_OPERATOR_BOOST:-}
TRAIN_GENERATOR=${TRAIN_GENERATOR:-}
GENERATOR_METHOD=${GENERATOR_METHOD:-}
GENERATOR_MODEL_OVERRIDE=${GENERATOR_MODEL_OVERRIDE:-}
GENERATOR_OUTPUT_DIR=${GENERATOR_OUTPUT_DIR:-}
OUTPUT_DIR_OVERRIDE=${OUTPUT_DIR_OVERRIDE:-}
GENERATOR_USE_LORA=${GENERATOR_USE_LORA:-}
GENERATOR_LEARNING_RATE=${GENERATOR_LEARNING_RATE:-}
GENERATOR_EPOCHS=${GENERATOR_EPOCHS:-}
GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE:-}
GENERATOR_FAIL_ON_ERROR=${GENERATOR_FAIL_ON_ERROR:-}
RERUN_GENERATOR_ON_RESUME=${RERUN_GENERATOR_ON_RESUME:-}
GENERATOR_MIN_PREFERENCES=${GENERATOR_MIN_PREFERENCES:-}
HONEST_CHUNKING=${HONEST_CHUNKING:-}
HONEST_BOUNDARY_FRACTION=${HONEST_BOUNDARY_FRACTION:-}
HONEST_SPLIT_SEED=${HONEST_SPLIT_SEED:-}
THREE_LAYER_HONESTY=${THREE_LAYER_HONESTY:-}
THREE_LAYER_SEED=${THREE_LAYER_SEED:-}
THREE_LAYER_CHUNK_TRAIN_FRACTION=${THREE_LAYER_CHUNK_TRAIN_FRACTION:-}
THREE_LAYER_SUMMARIZER_TRAIN_FRACTION=${THREE_LAYER_SUMMARIZER_TRAIN_FRACTION:-}
THREE_LAYER_ORACLE_TRAIN_FRACTION=${THREE_LAYER_ORACLE_TRAIN_FRACTION:-}

PORT_SET=false
GENRM_PORT_SET=false

# Paths (auto-detect project root from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ENV="${VLLM_ENV:-${HOME}/vllm-env}"  # Override with VLLM_ENV env var
PIPELINE_ENV="${PIPELINE_ENV:-${PROJECT_ROOT}/venv}"  # Preferred Python env for run_pipeline.py
# Output directory based on task (defaults to 'default' if no task specified)
TASK_DIR="${TASK:-default}"
OUTPUT_BASE="${PROJECT_ROOT}/data/results/${TASK_DIR}/training_pipeline"

# ============================================================================
# Help
# ============================================================================
show_help() {
    cat << 'EOF'
TRAINING PIPELINE

Usage: ./scripts/run_training_pipeline.sh [OPTIONS]

SERVER OPTIONS:
  --backend BACKEND       Backend for both task + GenRM (vllm|sglang, default: vllm)
  --task-backend BACKEND  Task backend override (vllm|sglang)
  --genrm-backend BACKEND GenRM backend override (vllm|sglang)
  --backend-fallback B    Backend fallback (none|vllm|sglang, default: none)
  --start-server          Auto-start task backend server (stops running servers)
  --no-start-server       Don't auto-start (default, requires server running)
  --model MODEL           Model profile for --start-server (default: nemotron-30b-nvfp4)
                          Available: nemotron-30b-nvfp4, nemotron-30b-fp8, qwen3.5-35b-a3b, qwen-30b-thinking, qwen-235b
                          In --dynamic-gpu mode, this is also forwarded as the
                          orchestrator task model profile override.
  --port PORT             Task server port (default: backend-specific; vllm=8000, sglang=30000)
  --opt-model-port PORT   Separate port for optimization model (optional)
  --task-cuda-devices IDS CUDA devices for task server start (default: 0,1)
  --task-tensor-parallel N  Tensor parallel override for task vLLM server start
  --sglang-venv-path PATH Path to SGLang virtual environment (default: /home/mlinegar/sglang-env)

DATA OPTIONS:
  --task NAME            Task plugin (default: manifesto_rile)
  --dataset NAME         Dataset plugin (default: settings.yaml datasets.default)
  --dataset-path PATH    Dataset path for file-based datasets (e.g., jsonl)
  --output-dir PATH      Explicit pipeline output directory
  --train-samples N       Number of training samples (default: 30)
  --val-samples N         Number of validation samples (default: 15)
  --test-samples N        Number of test samples (default: 10)
  --rounds N              Document processing rounds (default: 3)

CONCURRENCY OPTIONS:
  --concurrent-docs N     Docs to process in parallel (default: 20)
  --concurrent-requests N Concurrent LLM requests (default: 100)
  --max-chunk-chars N     Maximum chunk size in characters for tree building (default: 4000)
  --max-chunk-tokens N    Maximum chunk size in tokens for tree building; takes precedence over char chunking
  --num-threads N         Parallel metric evaluations (default: 16)

OPTIMIZER OPTIONS:
  --optimizer TYPE        Optimizer type (default: gepa)
                          Options: auto, gepa, bootstrap, bootstrap_random_search, mipro, labeled_fewshot
  --optimizer-budget BUDGET  Budget level (default: medium)
                          Options: light, medium, heavy
  --max-metric-calls N    Direct control over metric calls (overrides budget)
  --gepa-leaf-merge-sampling-design DESIGN
                          Leaf/merge GEPA sampling design:
                          two_stage_pps_bernoulli|srswor
  --gepa-ipw-estimator EST
                          Leaf/merge GEPA weighting estimator:
                          hajek|horvitz_thompson
  --gepa-ipw-min-propensity X
                          Propensity floor for IPW weighting (e.g., 1e-6)
  --scorer-max-tokens N   Override scorer completion token cap (default from task config)
  --scorer-temperature X  Override scorer temperature (default from task config)
  --scorer-strict-parse   Enable strict scorer output parsing
  --no-scorer-strict-parse
                          Disable strict scorer output parsing

ITERATIVE OPTIMIZATION:
  --n-iterations N        Number of iterations (default: 2)
                          1=single-pass oracle, 2+=iterative, 0=until convergence
  --convergence-threshold N  Threshold for early stopping (default: 0.01)
  --convergence-patience N   Rounds without improvement before stopping (default: 2)
  --skip-summarizer-opt   Skip summarizer optimization
  --skip-oracle-opt       Skip oracle optimization

TOP-DOWN INITIALIZATION:
  --use-top-down-init     Enable oracle-aligned demo seeding from short docs
  --n-init-demos N        Number of initialization demos (default: 8)
  --max-init-prompt-tokens N  Max tokens for init prompts (doc + rubric + instructions)
  --max-init-doc-chars N      Deprecated alias for --max-init-prompt-tokens

LEGACY GENRM/TOT (DEPRECATED; hard-fail):
  --start-genrm           Deprecated and blocked (use local-law bootstrap path)
  --train-comparison-module  Deprecated and blocked
  --enable-genrm          Deprecated and blocked (if forwarded via extra args)
  --optimize-judge        Deprecated and blocked (if forwarded via extra args)
  --tournament-of-tournaments
                          Deprecated and blocked (if forwarded via extra args)

ADAPTIVE / HONEST CHUNKING:
  --adaptive-chunking         Enable adaptive chunk sizing
  --no-adaptive-chunking      Disable adaptive chunk sizing
  --adaptive-chunk-min-chars N  Adaptive minimum chunk chars
  --adaptive-chunk-max-chars N  Adaptive maximum chunk chars
  --adaptive-proxy-blend X      Blend between text proxy and learned feedback [0,1]
  --adaptive-crossfit-folds N   K-fold diagnostic count for chunk-policy gap metrics
  --adaptive-embedding-proxy     Enable embedding-proxy training/query via vLLM API
  --adaptive-embedding-api-base URL   Embedding API base (default from settings)
  --adaptive-embedding-model MODEL    Embedding model id (auto-detect if omitted)
  --adaptive-embedding-head-method METHOD  Embedding head type: ridge|linear_sgd|mil_sgd
  --adaptive-embedding-head-epochs N       Epochs for trainable heads
  --adaptive-embedding-head-lr X           Learning rate for trainable heads
  --adaptive-embedding-head-weight-decay X Weight decay for trainable heads
  --adaptive-embedding-retrain-rounds N  Progressive retrain rounds for embedding head
  --adaptive-embedding-score-key KEY      Metadata key for embedding proxy scores
  --adaptive-embedding-full-finetune      Export finetune JSONL from proxy labels
  --embedding-proxy-fail-on-error         Fail pipeline on Phase 1.25 embedding-proxy runtime errors
  --no-embedding-proxy-fail-on-error      Continue pipeline even if Phase 1.25 embedding proxy fails
  --rerun-embedding-proxy-on-resume       Rerun Phase 1.25 even when resume checkpoint exists
  --no-rerun-embedding-proxy-on-resume    Reuse Phase 1.25 checkpoint on resume (default)
  --adaptive-embedding-finetune-command CMD  Optional command; placeholders:
                                             {dataset_path} {output_dir}
                                             {embedding_model} {proxy_model_artifact}
  --honest-chunking           Enable honest boundary/evaluation split
  --no-honest-chunking        Disable honest split
  --honest-boundary-fraction X  Fraction assigned to boundary split [0,1]
  --honest-split-seed N         Seed for deterministic split assignment
  --three-layer-honesty         Enable three-layer document honesty
  --no-three-layer-honesty      Disable three-layer document honesty
  --three-layer-seed N          Seed for three-layer split assignment
  --three-layer-chunk-train-fraction X       Chunker train fraction [0,1]
  --three-layer-summarizer-train-fraction X  Summarizer train fraction [0,1]
  --three-layer-oracle-train-fraction X      Oracle train fraction [0,1]

PREFERENCE-TREE COLLECTION (GENRM-FREE):
  --preference-init-samples N      Number of docs for Phase 1.5 preference trees
  --preference-init-candidates N   Candidates per tournament node
  --preference-tree-concurrency N  Concurrent tree builds for preference collection
  --preference-sample-seed N       Sampling seed for preference init docs
  --preference-incremental-sampling
  --no-preference-incremental-sampling
  --preference-judge-backend B     oracle|large_dspy
  --preference-tie-margin X        Tie margin used by oracle pairwise judge

NEURAL OPERATOR TRAINING (Phase 1.3):
  --train-neural-operators      Run scripts/train_neural_operators.py inside pipeline
  --no-train-neural-operators   Skip Phase 1.3 neural-operator training
  --neural-operators-which W    W in {both,ctreepo,mergeable_sketch}
  --neural-operators-output-dir PATH
  --neural-operators-ctreepo-args "..."
  --neural-operators-mergeable-args "..."
  --neural-operators-ctreepo-search-spec PATH
  --neural-operators-mergeable-search-spec PATH
  --neural-operators-fail-fast
  --neural-operators-fail-on-error
  --rerun-neural-operators-on-resume
  --neural-operators-auto-wire-representation
  --ctreepo-model-path PATH
  --mergeable-sketch-model-path PATH
  --hybrid-oracle-seeded-ensemble
  --hybrid-seed-llm-min-weight X
  --hybrid-seed-llm-max-weight X
  --hybrid-operator-boost X

GENERATOR TRAINING (Phase 3.25 / 3.5):
  --train-generator             Enable standalone generator fine-tuning
  --no-train-generator          Disable standalone generator fine-tuning
  --generator-method METHOD     dpo|sft|grpo|bootstrap_finetune
  --generator-model MODEL       Generator base model id/path
  --generator-output-dir PATH   Output directory for generator artifacts
  --generator-use-lora          Enable LoRA/PEFT adapters for generator training
  --no-generator-use-lora       Disable LoRA (full fine-tune path)
  --generator-learning-rate X   Generator learning rate
  --generator-epochs N          Generator training epochs
  --generator-batch-size N      Generator per-device train batch size
  --generator-min-preferences N Minimum preferences needed for generator training
  --generator-fail-on-error     Fail pipeline when generator training errors
  --no-generator-fail-on-error  Continue pipeline when generator training errors
  --rerun-generator-on-resume   Rerun generator training even if checkpoint exists
  --no-rerun-generator-on-resume
                                Reuse generator checkpoint on resume (default)

RESUME:
  --resume                Resume from latest checkpoint
  --no-resume             Don't resume (default, start fresh)

EXAMPLES:
  # Basic run (requires server already running)
  ./scripts/run_training_pipeline.sh

  # Auto-start vLLM server
  ./scripts/run_training_pipeline.sh --start-server

  # Use specific model
  ./scripts/run_training_pipeline.sh --start-server --model qwen3.5-35b-a3b

  # Fast 8-hour config for 4x GPU
  ./scripts/run_training_pipeline.sh --train-samples 30 --val-samples 15 \
    --n-iterations 2 --num-threads 16 --concurrent-requests 80

  # Run until convergence with top-down init
  ./scripts/run_training_pipeline.sh --n-iterations 0 --use-top-down-init

  # Resume from checkpoint
  ./scripts/run_training_pipeline.sh --resume

  # Local-law bootstrap (teacher scorer + proxy/GEPA)
  ./venv/bin/python scripts/run_manifesto_local_law_bootstrap_manual.py --help

  # Run in background
  nohup ./scripts/run_training_pipeline.sh > training.log 2>&1 &
EOF
    exit 0
}

# ============================================================================
# Parse command line arguments
# ============================================================================
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --port)
            PORT="$2"
            PORT_SET=true
            shift 2
            ;;
        --opt-model-port)
            OPT_MODEL_PORT="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --task-backend)
            TASK_BACKEND="$2"
            shift 2
            ;;
        --genrm-backend)
            GENRM_BACKEND="$2"
            shift 2
            ;;
        --backend-fallback)
            BACKEND_FALLBACK="$2"
            shift 2
            ;;
        --sglang-venv-path)
            SGLANG_VENV_PATH="$2"
            shift 2
            ;;
        --task-cuda-devices)
            TASK_CUDA_DEVICES="$2"
            shift 2
            ;;
        --task-tensor-parallel)
            TASK_TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --train-samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --val-samples)
            VAL_SAMPLES="$2"
            shift 2
            ;;
        --test-samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --concurrent-docs)
            CONCURRENT_DOCS="$2"
            shift 2
            ;;
        --concurrent-requests)
            CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        --max-chunk-chars)
            MAX_CHUNK_CHARS="$2"
            shift 2
            ;;
        --max-chunk-tokens)
            MAX_CHUNK_TOKENS="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --optimizer-budget)
            OPTIMIZER_BUDGET="$2"
            shift 2
            ;;
        --max-metric-calls)
            MAX_METRIC_CALLS="$2"
            shift 2
            ;;
        --num-threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --n-iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        --convergence-threshold)
            CONVERGENCE_THRESHOLD="$2"
            shift 2
            ;;
        --convergence-patience)
            CONVERGENCE_PATIENCE="$2"
            shift 2
            ;;
        --skip-summarizer-opt)
            SKIP_SUMMARIZER_OPT="true"
            shift
            ;;
        --skip-oracle-opt)
            SKIP_ORACLE_OPT="true"
            shift
            ;;
        --use-top-down-init)
            USE_TOP_DOWN_INIT="true"
            shift
            ;;
        --n-init-demos)
            N_INIT_DEMOS="$2"
            shift 2
            ;;
        --max-init-prompt-tokens)
            MAX_INIT_PROMPT_TOKENS="$2"
            shift 2
            ;;
        --max-init-doc-chars)
            MAX_INIT_PROMPT_TOKENS="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        --no-resume)
            RESUME="false"
            shift
            ;;
        --start-server)
            START_SERVER="true"
            shift
            ;;
        --no-start-server)
            START_SERVER="false"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --start-genrm)
            START_GENRM="true"
            shift
            ;;
        --no-start-genrm)
            START_GENRM="false"
            shift
            ;;
        --genrm-port)
            GENRM_PORT="$2"
            GENRM_PORT_SET=true
            shift 2
            ;;
        --genrm-model)
            GENRM_MODEL="$2"
            shift 2
            ;;
        --genrm-cuda-devices)
            GENRM_CUDA_DEVICES="$2"
            shift 2
            ;;
        --train-comparison-module)
            TRAIN_COMPARISON_MODULE="true"
            shift
            ;;
        --adaptive-chunking)
            ADAPTIVE_CHUNKING="true"
            shift
            ;;
        --no-adaptive-chunking)
            ADAPTIVE_CHUNKING="false"
            shift
            ;;
        --adaptive-chunk-min-chars)
            ADAPTIVE_CHUNK_MIN_CHARS="$2"
            shift 2
            ;;
        --adaptive-chunk-max-chars)
            ADAPTIVE_CHUNK_MAX_CHARS="$2"
            shift 2
            ;;
        --adaptive-proxy-blend)
            ADAPTIVE_PROXY_BLEND="$2"
            shift 2
            ;;
        --adaptive-crossfit-folds)
            ADAPTIVE_CROSSFIT_FOLDS="$2"
            shift 2
            ;;
        --adaptive-embedding-proxy)
            ADAPTIVE_EMBEDDING_PROXY="true"
            shift
            ;;
        --no-adaptive-embedding-proxy)
            ADAPTIVE_EMBEDDING_PROXY="false"
            shift
            ;;
        --adaptive-embedding-api-base)
            ADAPTIVE_EMBEDDING_API_BASE="$2"
            shift 2
            ;;
        --adaptive-embedding-model)
            ADAPTIVE_EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --adaptive-embedding-head-method)
            ADAPTIVE_EMBEDDING_HEAD_METHOD="$2"
            shift 2
            ;;
        --adaptive-embedding-head-epochs)
            ADAPTIVE_EMBEDDING_HEAD_EPOCHS="$2"
            shift 2
            ;;
        --adaptive-embedding-head-lr)
            ADAPTIVE_EMBEDDING_HEAD_LR="$2"
            shift 2
            ;;
        --adaptive-embedding-head-weight-decay)
            ADAPTIVE_EMBEDDING_HEAD_WEIGHT_DECAY="$2"
            shift 2
            ;;
        --adaptive-embedding-retrain-rounds)
            ADAPTIVE_EMBEDDING_RETRAIN_ROUNDS="$2"
            shift 2
            ;;
        --adaptive-embedding-score-key)
            ADAPTIVE_EMBEDDING_SCORE_KEY="$2"
            shift 2
            ;;
        --adaptive-embedding-full-finetune)
            ADAPTIVE_EMBEDDING_FULL_FINETUNE="true"
            shift
            ;;
        --no-adaptive-embedding-full-finetune)
            ADAPTIVE_EMBEDDING_FULL_FINETUNE="false"
            shift
            ;;
        --adaptive-embedding-finetune-command)
            ADAPTIVE_EMBEDDING_FINETUNE_COMMAND="$2"
            shift 2
            ;;
        --embedding-proxy-fail-on-error)
            EMBEDDING_PROXY_FAIL_ON_ERROR="true"
            shift
            ;;
        --no-embedding-proxy-fail-on-error)
            EMBEDDING_PROXY_FAIL_ON_ERROR="false"
            shift
            ;;
        --rerun-embedding-proxy-on-resume)
            RERUN_EMBEDDING_PROXY_ON_RESUME="true"
            shift
            ;;
        --no-rerun-embedding-proxy-on-resume)
            RERUN_EMBEDDING_PROXY_ON_RESUME="false"
            shift
            ;;
        --train-neural-operators)
            TRAIN_NEURAL_OPERATORS="true"
            shift
            ;;
        --no-train-neural-operators)
            TRAIN_NEURAL_OPERATORS="false"
            shift
            ;;
        --neural-operators-which)
            NEURAL_OPERATORS_WHICH="$2"
            shift 2
            ;;
        --neural-operators-output-dir)
            NEURAL_OPERATORS_OUTPUT_DIR="$2"
            shift 2
            ;;
        --neural-operators-ctreepo-args)
            NEURAL_OPERATORS_CTREEPO_ARGS="$2"
            shift 2
            ;;
        --neural-operators-mergeable-args)
            NEURAL_OPERATORS_MERGEABLE_ARGS="$2"
            shift 2
            ;;
        --neural-operators-ctreepo-search-spec)
            NEURAL_OPERATORS_CTREEPO_SEARCH_SPEC="$2"
            shift 2
            ;;
        --neural-operators-mergeable-search-spec)
            NEURAL_OPERATORS_MERGEABLE_SEARCH_SPEC="$2"
            shift 2
            ;;
        --neural-operators-fail-fast)
            NEURAL_OPERATORS_FAIL_FAST="true"
            shift
            ;;
        --no-neural-operators-fail-fast)
            NEURAL_OPERATORS_FAIL_FAST="false"
            shift
            ;;
        --neural-operators-fail-on-error)
            NEURAL_OPERATORS_FAIL_ON_ERROR="true"
            shift
            ;;
        --no-neural-operators-fail-on-error)
            NEURAL_OPERATORS_FAIL_ON_ERROR="false"
            shift
            ;;
        --rerun-neural-operators-on-resume)
            RERUN_NEURAL_OPERATORS_ON_RESUME="true"
            shift
            ;;
        --no-rerun-neural-operators-on-resume)
            RERUN_NEURAL_OPERATORS_ON_RESUME="false"
            shift
            ;;
        --neural-operators-auto-wire-representation)
            NEURAL_OPERATORS_AUTO_WIRE_REPRESENTATION="true"
            shift
            ;;
        --no-neural-operators-auto-wire-representation)
            NEURAL_OPERATORS_AUTO_WIRE_REPRESENTATION="false"
            shift
            ;;
        --hybrid-oracle-seeded-ensemble)
            HYBRID_ORACLE_SEEDED_ENSEMBLE="true"
            shift
            ;;
        --no-hybrid-oracle-seeded-ensemble)
            HYBRID_ORACLE_SEEDED_ENSEMBLE="false"
            shift
            ;;
        --hybrid-seed-llm-min-weight)
            HYBRID_SEED_LLM_MIN_WEIGHT="$2"
            shift 2
            ;;
        --hybrid-seed-llm-max-weight)
            HYBRID_SEED_LLM_MAX_WEIGHT="$2"
            shift 2
            ;;
        --hybrid-operator-boost)
            HYBRID_OPERATOR_BOOST="$2"
            shift 2
            ;;
        --train-generator)
            TRAIN_GENERATOR="true"
            shift
            ;;
        --no-train-generator)
            TRAIN_GENERATOR="false"
            shift
            ;;
        --generator-method)
            GENERATOR_METHOD="$2"
            shift 2
            ;;
        --generator-model)
            GENERATOR_MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --generator-output-dir)
            GENERATOR_OUTPUT_DIR="$2"
            shift 2
            ;;
        --generator-use-lora)
            GENERATOR_USE_LORA="true"
            shift
            ;;
        --no-generator-use-lora)
            GENERATOR_USE_LORA="false"
            shift
            ;;
        --generator-learning-rate)
            GENERATOR_LEARNING_RATE="$2"
            shift 2
            ;;
        --generator-epochs)
            GENERATOR_EPOCHS="$2"
            shift 2
            ;;
        --generator-batch-size)
            GENERATOR_BATCH_SIZE="$2"
            shift 2
            ;;
        --generator-fail-on-error)
            GENERATOR_FAIL_ON_ERROR="true"
            shift
            ;;
        --no-generator-fail-on-error)
            GENERATOR_FAIL_ON_ERROR="false"
            shift
            ;;
        --rerun-generator-on-resume)
            RERUN_GENERATOR_ON_RESUME="true"
            shift
            ;;
        --no-rerun-generator-on-resume)
            RERUN_GENERATOR_ON_RESUME="false"
            shift
            ;;
        --generator-min-preferences)
            GENERATOR_MIN_PREFERENCES="$2"
            shift 2
            ;;
        --honest-chunking)
            HONEST_CHUNKING="true"
            shift
            ;;
        --no-honest-chunking)
            HONEST_CHUNKING="false"
            shift
            ;;
        --honest-boundary-fraction)
            HONEST_BOUNDARY_FRACTION="$2"
            shift 2
            ;;
        --honest-split-seed)
            HONEST_SPLIT_SEED="$2"
            shift 2
            ;;
        --three-layer-honesty)
            THREE_LAYER_HONESTY="true"
            shift
            ;;
        --no-three-layer-honesty)
            THREE_LAYER_HONESTY="false"
            shift
            ;;
        --three-layer-seed)
            THREE_LAYER_SEED="$2"
            shift 2
            ;;
        --three-layer-chunk-train-fraction)
            THREE_LAYER_CHUNK_TRAIN_FRACTION="$2"
            shift 2
            ;;
        --three-layer-summarizer-train-fraction)
            THREE_LAYER_SUMMARIZER_TRAIN_FRACTION="$2"
            shift 2
            ;;
        --three-layer-oracle-train-fraction)
            THREE_LAYER_ORACLE_TRAIN_FRACTION="$2"
            shift 2
            ;;
        --genrm-init-samples)
            GENRM_INIT_SAMPLES="$2"
            shift 2
            ;;
        --genrm-init-candidates)
            GENRM_INIT_CANDIDATES="$2"
            shift 2
            ;;
        --preference-init-samples)
            PREFERENCE_INIT_SAMPLES="$2"
            shift 2
            ;;
        --preference-init-candidates)
            PREFERENCE_INIT_CANDIDATES="$2"
            shift 2
            ;;
        --preference-tree-concurrency)
            PREFERENCE_TREE_CONCURRENCY="$2"
            shift 2
            ;;
        --preference-sample-seed)
            PREFERENCE_SAMPLE_SEED="$2"
            shift 2
            ;;
        --preference-incremental-sampling)
            PREFERENCE_INCREMENTAL_SAMPLING="true"
            shift
            ;;
        --no-preference-incremental-sampling)
            PREFERENCE_INCREMENTAL_SAMPLING="false"
            shift
            ;;
        --preference-judge-backend)
            PREFERENCE_JUDGE_BACKEND="$2"
            shift 2
            ;;
        --preference-tie-margin)
            PREFERENCE_TIE_MARGIN="$2"
            shift 2
            ;;
        --dynamic-gpu)
            DYNAMIC_GPU="true"
            shift
            ;;
        --no-dynamic-gpu)
            DYNAMIC_GPU="false"
            shift
            ;;
        --gepa-leaf-merge-sampling-design)
            GEPA_LEAF_MERGE_SAMPLING_DESIGN="$2"
            shift 2
            ;;
        --gepa-ipw-estimator)
            GEPA_IPW_ESTIMATOR="$2"
            shift 2
            ;;
        --gepa-ipw-min-propensity)
            GEPA_IPW_MIN_PROPENSITY="$2"
            shift 2
            ;;
        --scorer-max-tokens)
            SCORER_MAX_TOKENS="$2"
            shift 2
            ;;
        --scorer-temperature)
            SCORER_TEMPERATURE="$2"
            shift 2
            ;;
        --scorer-strict-parse)
            SCORER_STRICT_PARSE="true"
            shift
            ;;
        --no-scorer-strict-parse)
            SCORER_STRICT_PARSE="false"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Normalize backend selections
BACKEND="$(printf '%s' "${BACKEND}" | tr '[:upper:]' '[:lower:]')"
TASK_BACKEND="$(printf '%s' "${TASK_BACKEND}" | tr '[:upper:]' '[:lower:]')"
GENRM_BACKEND="$(printf '%s' "${GENRM_BACKEND}" | tr '[:upper:]' '[:lower:]')"
BACKEND_FALLBACK="$(printf '%s' "${BACKEND_FALLBACK}" | tr '[:upper:]' '[:lower:]')"

if [[ "$BACKEND" != "vllm" && "$BACKEND" != "sglang" ]]; then
    echo "ERROR: --backend must be 'vllm' or 'sglang' (got '$BACKEND')"
    exit 1
fi
if [[ -z "$TASK_BACKEND" ]]; then
    TASK_BACKEND="$BACKEND"
fi
if [[ -z "$GENRM_BACKEND" ]]; then
    GENRM_BACKEND="$BACKEND"
fi
if [[ "$TASK_BACKEND" != "vllm" && "$TASK_BACKEND" != "sglang" ]]; then
    echo "ERROR: --task-backend must be 'vllm' or 'sglang' (got '$TASK_BACKEND')"
    exit 1
fi
if [[ "$GENRM_BACKEND" != "vllm" && "$GENRM_BACKEND" != "sglang" ]]; then
    echo "ERROR: --genrm-backend must be 'vllm' or 'sglang' (got '$GENRM_BACKEND')"
    exit 1
fi

if [[ "${START_GENRM}" == "true" ]] || [[ "${TRAIN_COMPARISON_MODULE}" == "true" ]]; then
    echo "ERROR: GenRM/TOT entrypoints are deprecated and blocked in this wrapper." >&2
    echo "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM." >&2
    exit 2
fi

for arg in "${EXTRA_ARGS[@]}"; do
    case "${arg}" in
        --enable-genrm|--optimize-judge|--tournament-of-tournaments)
            echo "ERROR: Deprecated flag '${arg}' is blocked." >&2
            echo "Use local-law bootstrap (teacher scorer + proxy/GEPA), no GenRM." >&2
            exit 2
            ;;
    esac
done
case "$BACKEND_FALLBACK" in
    ""|"none"|"off"|"disabled")
        BACKEND_FALLBACK="none"
        ;;
    "vllm"|"sglang")
        ;;
    *)
        echo "ERROR: --backend-fallback must be none|vllm|sglang (got '$BACKEND_FALLBACK')"
        exit 1
        ;;
esac

if [[ -n "${ADAPTIVE_EMBEDDING_HEAD_METHOD}" ]]; then
    ADAPTIVE_EMBEDDING_HEAD_METHOD="$(printf '%s' "${ADAPTIVE_EMBEDDING_HEAD_METHOD}" | tr '[:upper:]' '[:lower:]')"
    case "${ADAPTIVE_EMBEDDING_HEAD_METHOD}" in
        ridge|linear_sgd|mil_sgd)
            ;;
        *)
            echo "ERROR: --adaptive-embedding-head-method must be ridge|linear_sgd|mil_sgd (got '${ADAPTIVE_EMBEDDING_HEAD_METHOD}')"
            exit 1
            ;;
    esac
fi

if [[ -n "${PREFERENCE_JUDGE_BACKEND}" ]]; then
    PREFERENCE_JUDGE_BACKEND="$(printf '%s' "${PREFERENCE_JUDGE_BACKEND}" | tr '[:upper:]' '[:lower:]')"
    case "${PREFERENCE_JUDGE_BACKEND}" in
        oracle|large_dspy)
            ;;
        *)
            echo "ERROR: --preference-judge-backend must be oracle|large_dspy (got '${PREFERENCE_JUDGE_BACKEND}')" >&2
            exit 1
            ;;
    esac
fi

if [[ -n "${NEURAL_OPERATORS_WHICH}" ]]; then
    NEURAL_OPERATORS_WHICH="$(printf '%s' "${NEURAL_OPERATORS_WHICH}" | tr '[:upper:]' '[:lower:]')"
    case "${NEURAL_OPERATORS_WHICH}" in
        both|ctreepo|mergeable_sketch)
            ;;
        *)
            echo "ERROR: --neural-operators-which must be both|ctreepo|mergeable_sketch (got '${NEURAL_OPERATORS_WHICH}')"
            exit 1
            ;;
    esac
fi

if [[ -n "${GENERATOR_METHOD}" ]]; then
    GENERATOR_METHOD="$(printf '%s' "${GENERATOR_METHOD}" | tr '[:upper:]' '[:lower:]')"
    case "${GENERATOR_METHOD}" in
        dpo|sft|grpo|bootstrap_finetune)
            ;;
        *)
            echo "ERROR: --generator-method must be dpo|sft|grpo|bootstrap_finetune (got '${GENERATOR_METHOD}')"
            exit 1
            ;;
    esac
fi

# Backend-specific default ports from config when user did not override.
read VLLM_DEFAULT_PORT SGLANG_DEFAULT_PORT SGLANG_DEFAULT_GENRM_PORT < <(python3 -c "
import yaml
with open('${PROJECT_ROOT}/config/settings.yaml') as f:
    cfg = yaml.safe_load(f) or {}
v = cfg.get('vllm', {}) if isinstance(cfg, dict) else {}
s = cfg.get('sglang', {}) if isinstance(cfg, dict) else {}
v_port = int(v.get('port', 8000) or 8000)
s_port = int(s.get('port', 30000) or 30000)
s_genrm_port = int(s.get('genrm_port', s_port + 1) or (s_port + 1))
print(v_port, s_port, s_genrm_port)
" 2>/dev/null)
VLLM_DEFAULT_PORT=${VLLM_DEFAULT_PORT:-8000}
SGLANG_DEFAULT_PORT=${SGLANG_DEFAULT_PORT:-30000}
SGLANG_DEFAULT_GENRM_PORT=${SGLANG_DEFAULT_GENRM_PORT:-$((SGLANG_DEFAULT_PORT + 1))}
VLLM_DEFAULT_GENRM_PORT=$((VLLM_DEFAULT_PORT + 1))

if [[ "${PORT_SET}" == "false" ]]; then
    if [[ "$TASK_BACKEND" == "sglang" ]]; then
        PORT="${SGLANG_DEFAULT_PORT}"
    else
        PORT="${VLLM_DEFAULT_PORT}"
    fi
fi
if [[ "${GENRM_PORT_SET}" == "false" ]]; then
    if [[ "$GENRM_BACKEND" == "sglang" ]]; then
        GENRM_PORT="${SGLANG_DEFAULT_GENRM_PORT}"
    else
        GENRM_PORT="${VLLM_DEFAULT_GENRM_PORT}"
    fi
fi

# Dynamic GPU orchestration supports all backends.
# Non-vLLM backends use stop/start transitions on shared GPUs.
if [[ "${DYNAMIC_GPU}" == "true" ]]; then
    if [[ "$TASK_BACKEND" != "vllm" || "$GENRM_BACKEND" != "vllm" ]]; then
        echo "INFO: --dynamic-gpu with task=${TASK_BACKEND}, genrm=${GENRM_BACKEND} will use cold stop/start transitions on shared GPUs."
    fi
fi

# ============================================================================
# Setup
# ============================================================================

# Handle resume: find most recent run directory instead of creating new one
if [[ "${RESUME}" == "true" ]]; then
    # Find the most recent directory with checkpoints (handles nested dirs from old bug)
    CHECKPOINT_DIR=$(find "${OUTPUT_BASE}" -name "checkpoints" -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [[ -z "${CHECKPOINT_DIR}" ]]; then
        echo "ERROR: --resume specified but no checkpoints found in ${OUTPUT_BASE}"
        exit 1
    fi

    # Check that checkpoint has actual files
    if [[ -n "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]]; then
        # Get parent directory (the actual run dir)
        OUTPUT_DIR=$(dirname "${CHECKPOINT_DIR}")
        echo "Resuming from: ${OUTPUT_DIR}"
        echo "  Checkpoints: ${CHECKPOINT_DIR}"
        ls "${CHECKPOINT_DIR}"
    else
        echo "ERROR: Checkpoint directory is empty: ${CHECKPOINT_DIR}"
        echo "Cannot resume. Run without --resume to start fresh."
        exit 1
    fi
else
    # Create new output directory unless one was explicitly provided.
    if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
    else
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="${OUTPUT_BASE}/run_${TIMESTAMP}"
    fi
    mkdir -p "${OUTPUT_DIR}"
fi

LOG_FILE="${OUTPUT_DIR}/run.log"

# Logging function
log() {
    local ts
    local msg
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    msg="$1"
    printf '[%s] %s\n' "${ts}" "${msg}"
    printf '[%s] %s\n' "${ts}" "${msg}" >> "${LOG_FILE}"
}

# ============================================================================
# Banner
# ============================================================================
echo ""
echo "========================================================================"
echo "  TRAINING PIPELINE"
echo "========================================================================"
echo "  Started:           $(date)"
echo "  Output:            ${OUTPUT_DIR}"
echo ""
echo "  Settings:"
echo "    Task Port:       ${PORT}"
if [[ -n "${OPT_MODEL_PORT}" ]]; then
echo "    Opt Model Port:  ${OPT_MODEL_PORT}"
fi
echo "    Backends:        task=${TASK_BACKEND} genrm=${GENRM_BACKEND} fallback=${BACKEND_FALLBACK}"
if [[ "${TASK_BACKEND}" == "sglang" || "${GENRM_BACKEND}" == "sglang" || "${BACKEND_FALLBACK}" == "sglang" ]]; then
echo "    SGLang venv:     ${SGLANG_VENV_PATH}"
fi
echo "    Train Samples:   ${TRAIN_SAMPLES}"
echo "    Val Samples:     ${VAL_SAMPLES}"
echo "    Test Samples:    ${TEST_SAMPLES}"
echo "    Rounds:          ${ROUNDS}"
echo "    Concurrent Docs: ${CONCURRENT_DOCS}"
echo "    Concurrent Reqs: ${CONCURRENT_REQUESTS}"
echo "    Max Chunk Chars: ${MAX_CHUNK_CHARS}"
if [[ -n "${MAX_CHUNK_TOKENS}" ]]; then
echo "    Max Chunk Toks:  ${MAX_CHUNK_TOKENS}"
fi
echo ""
echo "  Optimizer:"
echo "    Type:            ${OPTIMIZER}"
echo "    Budget:          ${OPTIMIZER_BUDGET}"
if [[ -n "${MAX_METRIC_CALLS}" ]]; then
echo "    Max Metric Calls: ${MAX_METRIC_CALLS} (overrides budget)"
fi
if [[ -n "${GEPA_LEAF_MERGE_SAMPLING_DESIGN}" || -n "${GEPA_IPW_ESTIMATOR}" || -n "${GEPA_IPW_MIN_PROPENSITY}" ]]; then
echo "    GEPA Sampling:   ${GEPA_LEAF_MERGE_SAMPLING_DESIGN:-default}"
echo "    GEPA IPW Est:    ${GEPA_IPW_ESTIMATOR:-default}"
echo "    GEPA Min Prop:   ${GEPA_IPW_MIN_PROPENSITY:-default}"
fi
if [[ -n "${SCORER_MAX_TOKENS}" || -n "${SCORER_TEMPERATURE}" || -n "${SCORER_STRICT_PARSE}" ]]; then
echo "    Scorer MaxTok:   ${SCORER_MAX_TOKENS:-task-default}"
echo "    Scorer Temp:     ${SCORER_TEMPERATURE:-task-default}"
echo "    Scorer Strict:   ${SCORER_STRICT_PARSE:-task-default}"
fi
echo "    Threads:         ${NUM_THREADS}"
echo "    Start Server:    ${START_SERVER}"
echo "    Dynamic GPU:     ${DYNAMIC_GPU} (use --dynamic-gpu to enable sleep mode)"
if [[ "${START_SERVER}" == "true" ]]; then
echo "    Model:           ${MODEL}"
echo "    Task CUDA:       ${TASK_CUDA_DEVICES}"
if [[ -n "${TASK_TENSOR_PARALLEL}" ]]; then
echo "    Task TP:         ${TASK_TENSOR_PARALLEL}"
fi
fi
echo ""
echo "  Iterative Optimization:"
echo "    Iterations:      ${N_ITERATIONS} (0=until convergence)"
echo "    Conv Threshold:  ${CONVERGENCE_THRESHOLD}"
echo "    Conv Patience:   ${CONVERGENCE_PATIENCE}"
echo "    Skip Summarizer: ${SKIP_SUMMARIZER_OPT}"
echo "    Top-Down Init:   ${USE_TOP_DOWN_INIT} (demos: ${N_INIT_DEMOS}, max_tokens: ${MAX_INIT_PROMPT_TOKENS})"
echo ""
echo "  Legacy GenRM/TOT:"
echo "    Mode:            disabled (large-model-only migration)"
echo "    Legacy Port:     ${GENRM_PORT}"
echo ""
echo "  Adaptive/Honest Chunking:"
echo "    Adaptive:        ${ADAPTIVE_CHUNKING:-settings.yaml}"
if [[ -n "${ADAPTIVE_CHUNK_MIN_CHARS}" || -n "${ADAPTIVE_CHUNK_MAX_CHARS}" || -n "${ADAPTIVE_PROXY_BLEND}" || -n "${ADAPTIVE_CROSSFIT_FOLDS}" ]]; then
echo "    Adaptive Min:    ${ADAPTIVE_CHUNK_MIN_CHARS:-settings.yaml}"
echo "    Adaptive Max:    ${ADAPTIVE_CHUNK_MAX_CHARS:-settings.yaml}"
echo "    Proxy Blend:     ${ADAPTIVE_PROXY_BLEND:-settings.yaml}"
echo "    Crossfit Folds:  ${ADAPTIVE_CROSSFIT_FOLDS:-settings.yaml}"
fi
echo "    Honest:          ${HONEST_CHUNKING:-settings.yaml}"
if [[ -n "${HONEST_BOUNDARY_FRACTION}" || -n "${HONEST_SPLIT_SEED}" ]]; then
echo "    Boundary Frac:   ${HONEST_BOUNDARY_FRACTION:-settings.yaml}"
echo "    Split Seed:      ${HONEST_SPLIT_SEED:-settings.yaml}"
fi
echo "    Three-Layer:     ${THREE_LAYER_HONESTY:-settings.yaml}"
if [[ -n "${THREE_LAYER_SEED}" || -n "${THREE_LAYER_CHUNK_TRAIN_FRACTION}" || -n "${THREE_LAYER_SUMMARIZER_TRAIN_FRACTION}" || -n "${THREE_LAYER_ORACLE_TRAIN_FRACTION}" ]]; then
echo "    3L Seed:         ${THREE_LAYER_SEED:-settings.yaml}"
echo "    3L Chunk Train:  ${THREE_LAYER_CHUNK_TRAIN_FRACTION:-settings.yaml}"
echo "    3L Summ Train:   ${THREE_LAYER_SUMMARIZER_TRAIN_FRACTION:-settings.yaml}"
echo "    3L Oracle Train: ${THREE_LAYER_ORACLE_TRAIN_FRACTION:-settings.yaml}"
fi
if [[ -n "${ADAPTIVE_EMBEDDING_PROXY}" || -n "${ADAPTIVE_EMBEDDING_API_BASE}" || -n "${ADAPTIVE_EMBEDDING_MODEL}" || -n "${ADAPTIVE_EMBEDDING_HEAD_METHOD}" || -n "${EMBEDDING_PROXY_FAIL_ON_ERROR}" || -n "${RERUN_EMBEDDING_PROXY_ON_RESUME}" ]]; then
echo "    Embedding Proxy: ${ADAPTIVE_EMBEDDING_PROXY:-settings.yaml}"
echo "    Embed API Base:  ${ADAPTIVE_EMBEDDING_API_BASE:-settings.yaml}"
echo "    Embed Model:     ${ADAPTIVE_EMBEDDING_MODEL:-settings.yaml}"
echo "    Embed Head:      ${ADAPTIVE_EMBEDDING_HEAD_METHOD:-settings.yaml}"
echo "    Embed Fail Err:  ${EMBEDDING_PROXY_FAIL_ON_ERROR:-settings.yaml}"
echo "    Embed Rerun:     ${RERUN_EMBEDDING_PROXY_ON_RESUME:-settings.yaml}"
fi
echo ""
echo "  Neural Operators:"
echo "    Train:           ${TRAIN_NEURAL_OPERATORS:-settings.yaml}"
if [[ -n "${NEURAL_OPERATORS_WHICH}" || -n "${NEURAL_OPERATORS_OUTPUT_DIR}" || -n "${NEURAL_OPERATORS_FAIL_ON_ERROR}" || -n "${RERUN_NEURAL_OPERATORS_ON_RESUME}" ]]; then
echo "    Which:           ${NEURAL_OPERATORS_WHICH:-settings.yaml}"
echo "    Output Dir:      ${NEURAL_OPERATORS_OUTPUT_DIR:-settings.yaml}"
echo "    Fail On Error:   ${NEURAL_OPERATORS_FAIL_ON_ERROR:-settings.yaml}"
echo "    Rerun Resume:    ${RERUN_NEURAL_OPERATORS_ON_RESUME:-settings.yaml}"
fi
if [[ -n "${HYBRID_ORACLE_SEEDED_ENSEMBLE}" || -n "${HYBRID_SEED_LLM_MIN_WEIGHT}" || -n "${HYBRID_SEED_LLM_MAX_WEIGHT}" || -n "${HYBRID_OPERATOR_BOOST}" ]]; then
echo "    Hybrid Enabled:  ${HYBRID_ORACLE_SEEDED_ENSEMBLE:-settings.yaml}"
echo "    Hybrid LLM Min:  ${HYBRID_SEED_LLM_MIN_WEIGHT:-settings.yaml}"
echo "    Hybrid LLM Max:  ${HYBRID_SEED_LLM_MAX_WEIGHT:-settings.yaml}"
echo "    Hybrid Op Boost: ${HYBRID_OPERATOR_BOOST:-settings.yaml}"
fi
echo ""
echo "  Generator Training:"
echo "    Train:           ${TRAIN_GENERATOR:-settings.yaml}"
echo "    Method:          ${GENERATOR_METHOD:-settings.yaml}"
echo "    Model:           ${GENERATOR_MODEL_OVERRIDE:-settings.yaml}"
if [[ -n "${GENERATOR_USE_LORA}" || -n "${GENERATOR_LEARNING_RATE}" || -n "${GENERATOR_EPOCHS}" || -n "${GENERATOR_BATCH_SIZE}" || -n "${GENERATOR_MIN_PREFERENCES}" || -n "${GENERATOR_FAIL_ON_ERROR}" || -n "${RERUN_GENERATOR_ON_RESUME}" ]]; then
echo "    Use LoRA:        ${GENERATOR_USE_LORA:-settings.yaml}"
echo "    LR:              ${GENERATOR_LEARNING_RATE:-settings.yaml}"
echo "    Epochs:          ${GENERATOR_EPOCHS:-settings.yaml}"
echo "    Batch Size:      ${GENERATOR_BATCH_SIZE:-settings.yaml}"
echo "    Min Prefs:       ${GENERATOR_MIN_PREFERENCES:-settings.yaml}"
echo "    Fail On Error:   ${GENERATOR_FAIL_ON_ERROR:-settings.yaml}"
echo "    Rerun Resume:    ${RERUN_GENERATOR_ON_RESUME:-settings.yaml}"
fi
echo ""
echo "  Resume:"
echo "    Resume:          ${RESUME}"
echo "========================================================================"
echo ""

# ============================================================================
# Activate environment
# ============================================================================
ACTIVE_PY_ENV="${VLLM_ENV}"
if [[ -d "${PIPELINE_ENV}" ]]; then
    ACTIVE_PY_ENV="${PIPELINE_ENV}"
elif [[ ! -d "${VLLM_ENV}" ]]; then
    log "ERROR: Neither PIPELINE_ENV (${PIPELINE_ENV}) nor VLLM_ENV (${VLLM_ENV}) exists"
    exit 1
fi
log "Activating Python environment: ${ACTIVE_PY_ENV}"
source "${ACTIVE_PY_ENV}/bin/activate"
cd "${PROJECT_ROOT}"

# Add project root to Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============================================================================
# Auto-start task server (optional)
# When DYNAMIC_GPU=true, Python orchestrator manages vLLM server lifecycle.
# ============================================================================
TASK_SERVER_PID=""
ORIGINAL_MODEL=""

check_server() {
    curl -s "http://localhost:$1/v1/models" > /dev/null 2>&1
    return $?
}

if [[ "${DYNAMIC_GPU}" == "true" ]]; then
    log ""
    log "========================================================================"
    log "Dynamic GPU Allocation Enabled"
    log "========================================================================"
    log "Python orchestrator will manage backend servers for task/genrm."
    if [[ "${TASK_BACKEND}" == "vllm" && "${GENRM_BACKEND}" == "vllm" ]]; then
        log "Using vLLM sleep mode transitions where available."
    else
        log "Using cold stop/start transitions on shared GPUs for non-vLLM backends."
    fi
    log "Servers will be started/stopped dynamically between phases."
    log "This provides ~6-12 second transitions (vs 60-120s disk reload)."
    log ""
    # Skip shell-based server startup - Python handles it
elif [[ "${START_SERVER}" == "true" ]]; then
    log ""
    log "========================================================================"
    log "Starting task server backend=${TASK_BACKEND}: ${MODEL}"
    log "========================================================================"

    # Remember what model was running (if any)
    if check_server "${PORT}"; then
        ORIGINAL_MODEL=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")
        log "Original model on port ${PORT}: ${ORIGINAL_MODEL}"
    fi

    # Stop known servers before restarting.
    log "Stopping running servers on common ports..."
    "${PROJECT_ROOT}/scripts/stop_small_servers.sh" --all || true
    sleep 5

    if [[ "${TASK_BACKEND}" == "vllm" ]]; then
        TASK_VLLM_CMD=(
            "${PROJECT_ROOT}/scripts/start_vllm.sh"
            "${MODEL}"
            --port "${PORT}"
            --cuda-devices "${TASK_CUDA_DEVICES}"
        )
        if [[ -n "${TASK_TENSOR_PARALLEL}" ]]; then
            TASK_VLLM_CMD+=(--tensor-parallel "${TASK_TENSOR_PARALLEL}")
        fi
        "${TASK_VLLM_CMD[@]}" > "${OUTPUT_DIR}/task_vllm.log" 2>&1 &
    else
        "${PROJECT_ROOT}/scripts/start_sglang.sh" \
            "${MODEL}" \
            --port "${PORT}" \
            --cuda-devices "${TASK_CUDA_DEVICES}" \
            --sglang-venv-path "${SGLANG_VENV_PATH}" \
            > "${OUTPUT_DIR}/task_sglang.log" 2>&1 &
    fi

    TASK_SERVER_PID=$!
    log "Task server starting (PID: ${TASK_SERVER_PID})"

    # Wait for server to be ready (up to 600s for large models/backends)
    log "Waiting for task server on port ${PORT}..."
    for i in {1..300}; do
        if check_server "${PORT}"; then
            MODEL_INFO=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
            log "Task server ready with model: ${MODEL_INFO}"
            break
        fi
        if [[ $i -eq 300 ]]; then
            log "ERROR: Task server failed to start within 600 seconds"
            log "Check ${OUTPUT_DIR}/task_${TASK_BACKEND}.log for details"
            exit 1
        fi
        sleep 2
    done
else
    log "Checking task backend (${TASK_BACKEND}) server on port ${PORT}..."
    if check_server "${PORT}"; then
        MODEL_INFO=$(curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
        log "Task model: ${MODEL_INFO}"
    else
        log "ERROR: Task backend server not running on port ${PORT}"
        log ""
        log "Please start the task server first:"
        if [[ "${TASK_BACKEND}" == "vllm" ]]; then
            log "  ./scripts/start_vllm.sh ${MODEL} --port ${PORT}"
        else
            log "  ./scripts/start_sglang.sh ${MODEL} --port ${PORT}"
        fi
        log ""
        log "Or auto-start with:"
        log "  --start-server"
        exit 1
    fi

    # Check optimization model port if specified
    if [[ -n "${OPT_MODEL_PORT}" ]]; then
        log ""
        log "Checking optimization model on port ${OPT_MODEL_PORT}..."
        if check_server "${OPT_MODEL_PORT}"; then
            OPT_MODEL_INFO=$(curl -s "http://localhost:${OPT_MODEL_PORT}/v1/models" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
            log "Optimization model: ${OPT_MODEL_INFO}"
        else
            log "ERROR: Optimization model not running on port ${OPT_MODEL_PORT}"
            log ""
            if [[ "${TASK_BACKEND}" == "vllm" ]]; then
                log "Start an optimization model:"
                log "  ./scripts/start_vllm.sh ${MODEL} --port ${OPT_MODEL_PORT}"
            else
                log "Start an optimization model:"
                log "  ./scripts/start_sglang.sh ${MODEL} --port ${OPT_MODEL_PORT}"
            fi
            exit 1
        fi
    fi
fi

# ============================================================================
# Legacy GenRM server startup path (disabled in large-model-only mode)
# ============================================================================
GENRM_PID=""

if curl -s "http://localhost:${GENRM_PORT}/v1/models" > /dev/null 2>&1; then
    GENRM_MODEL_INFO=$(curl -s "http://localhost:${GENRM_PORT}/v1/models" | \
        python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'unknown')" 2>/dev/null || echo "unknown")
    log "INFO: Detected server on legacy GenRM port ${GENRM_PORT}: ${GENRM_MODEL_INFO} (ignored in large-model-only mode)"
fi

# ============================================================================
# Run Training Pipeline
# ============================================================================
log ""
log "========================================================================"
log "Starting Training Pipeline"
log "========================================================================"
log ""

# Build command with optional arguments
CMD=(
    python -m src.training.run_pipeline
    --port ${PORT}
    --task-backend ${TASK_BACKEND}
    --genrm-backend ${GENRM_BACKEND}
    --backend-fallback ${BACKEND_FALLBACK}
    --train-samples ${TRAIN_SAMPLES}
    --val-samples ${VAL_SAMPLES}
    --test-samples ${TEST_SAMPLES}
    --concurrent-docs ${CONCURRENT_DOCS}
    --concurrent-requests ${CONCURRENT_REQUESTS}
    --max-chunk-chars ${MAX_CHUNK_CHARS}
    --optimizer ${OPTIMIZER}
    --optimizer-budget ${OPTIMIZER_BUDGET}
    --num-threads ${NUM_THREADS}
    --n-iterations ${N_ITERATIONS}
    --convergence-threshold ${CONVERGENCE_THRESHOLD}
    --convergence-patience ${CONVERGENCE_PATIENCE}
    --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_CHUNK_TOKENS}" ]]; then
    CMD+=(--max-chunk-tokens ${MAX_CHUNK_TOKENS})
fi

if [[ "${TASK_BACKEND}" == "sglang" || "${GENRM_BACKEND}" == "sglang" || "${BACKEND_FALLBACK}" == "sglang" ]]; then
    CMD+=(--sglang-venv-path "${SGLANG_VENV_PATH}")
fi

# Add dynamic GPU allocation flag
if [[ "${DYNAMIC_GPU}" == "true" ]]; then
    CMD+=(--dynamic-gpu)
    # In dynamic mode, map --model to orchestrator task profile selection.
    if [[ -n "${MODEL}" ]]; then
        CMD+=(--dynamic-task-model-profile "${MODEL}")
    fi
else
    CMD+=(--no-dynamic-gpu)
fi

# Add optional arguments
if [[ -n "${OPT_MODEL_PORT}" ]]; then
    CMD+=(--opt-model-port ${OPT_MODEL_PORT})
fi

if [[ -n "${TASK}" ]]; then
    CMD+=(--task ${TASK})
fi

if [[ -n "${DATASET}" ]]; then
    CMD+=(--dataset ${DATASET})
fi

if [[ -n "${DATASET_PATH}" ]]; then
    CMD+=(--dataset-path "${DATASET_PATH}")
fi

if [[ -n "${MAX_METRIC_CALLS}" ]]; then
    CMD+=(--max-metric-calls ${MAX_METRIC_CALLS})
fi
if [[ -n "${GEPA_LEAF_MERGE_SAMPLING_DESIGN}" ]]; then
    CMD+=(--gepa-leaf-merge-sampling-design ${GEPA_LEAF_MERGE_SAMPLING_DESIGN})
fi
if [[ -n "${GEPA_IPW_ESTIMATOR}" ]]; then
    CMD+=(--gepa-ipw-estimator ${GEPA_IPW_ESTIMATOR})
fi
if [[ -n "${GEPA_IPW_MIN_PROPENSITY}" ]]; then
    CMD+=(--gepa-ipw-min-propensity ${GEPA_IPW_MIN_PROPENSITY})
fi
if [[ -n "${SCORER_MAX_TOKENS}" ]]; then
    CMD+=(--scorer-max-tokens ${SCORER_MAX_TOKENS})
fi
if [[ -n "${SCORER_TEMPERATURE}" ]]; then
    CMD+=(--scorer-temperature ${SCORER_TEMPERATURE})
fi
if [[ "${SCORER_STRICT_PARSE}" == "true" ]]; then
    CMD+=(--scorer-strict-parse)
elif [[ "${SCORER_STRICT_PARSE}" == "false" ]]; then
    CMD+=(--no-scorer-strict-parse)
fi

if [[ "${SKIP_SUMMARIZER_OPT}" == "true" ]]; then
    CMD+=(--skip-summarizer-opt)
fi

if [[ "${SKIP_ORACLE_OPT}" == "true" ]]; then
    CMD+=(--skip-oracle-opt)
fi

if [[ "${USE_TOP_DOWN_INIT}" == "true" ]]; then
    CMD+=(--use-top-down-init --n-init-demos ${N_INIT_DEMOS} --max-init-prompt-tokens ${MAX_INIT_PROMPT_TOKENS})
fi

if [[ "${RESUME}" == "true" ]]; then
    CMD+=(--resume)
fi

# GenRM/TOT paths are disabled; wrapper intentionally does not pass any legacy flags.

if [[ -n "${PREFERENCE_INIT_SAMPLES}" ]]; then
    CMD+=(--preference-init-samples ${PREFERENCE_INIT_SAMPLES})
elif [[ -n "${GENRM_INIT_SAMPLES}" ]]; then
    CMD+=(--genrm-init-samples ${GENRM_INIT_SAMPLES})
fi
if [[ -n "${PREFERENCE_INIT_CANDIDATES}" ]]; then
    CMD+=(--preference-init-candidates ${PREFERENCE_INIT_CANDIDATES})
elif [[ -n "${GENRM_INIT_CANDIDATES}" ]]; then
    CMD+=(--genrm-init-candidates ${GENRM_INIT_CANDIDATES})
fi
if [[ -n "${PREFERENCE_TREE_CONCURRENCY}" ]]; then
    CMD+=(--preference-tree-concurrency ${PREFERENCE_TREE_CONCURRENCY})
fi
if [[ -n "${PREFERENCE_SAMPLE_SEED}" ]]; then
    CMD+=(--preference-sample-seed ${PREFERENCE_SAMPLE_SEED})
fi
if [[ "${PREFERENCE_INCREMENTAL_SAMPLING}" == "true" ]]; then
    CMD+=(--preference-incremental-sampling)
elif [[ "${PREFERENCE_INCREMENTAL_SAMPLING}" == "false" ]]; then
    CMD+=(--no-preference-incremental-sampling)
fi
if [[ -n "${PREFERENCE_JUDGE_BACKEND}" ]]; then
    CMD+=(--preference-judge-backend ${PREFERENCE_JUDGE_BACKEND})
fi
if [[ -n "${PREFERENCE_TIE_MARGIN}" ]]; then
    CMD+=(--preference-tie-margin ${PREFERENCE_TIE_MARGIN})
fi

if [[ "${ADAPTIVE_CHUNKING}" == "true" ]]; then
    CMD+=(--adaptive-chunking)
elif [[ "${ADAPTIVE_CHUNKING}" == "false" ]]; then
    CMD+=(--no-adaptive-chunking)
fi

if [[ -n "${ADAPTIVE_CHUNK_MIN_CHARS}" ]]; then
    CMD+=(--adaptive-chunk-min-chars ${ADAPTIVE_CHUNK_MIN_CHARS})
fi

if [[ -n "${ADAPTIVE_CHUNK_MAX_CHARS}" ]]; then
    CMD+=(--adaptive-chunk-max-chars ${ADAPTIVE_CHUNK_MAX_CHARS})
fi

if [[ -n "${ADAPTIVE_PROXY_BLEND}" ]]; then
    CMD+=(--adaptive-proxy-blend ${ADAPTIVE_PROXY_BLEND})
fi
if [[ -n "${ADAPTIVE_CROSSFIT_FOLDS}" ]]; then
    CMD+=(--adaptive-crossfit-folds ${ADAPTIVE_CROSSFIT_FOLDS})
fi
if [[ "${ADAPTIVE_EMBEDDING_PROXY}" == "true" ]]; then
    CMD+=(--adaptive-embedding-proxy)
elif [[ "${ADAPTIVE_EMBEDDING_PROXY}" == "false" ]]; then
    CMD+=(--no-adaptive-embedding-proxy)
fi
if [[ -n "${ADAPTIVE_EMBEDDING_API_BASE}" ]]; then
    CMD+=(--adaptive-embedding-api-base "${ADAPTIVE_EMBEDDING_API_BASE}")
fi
if [[ -n "${ADAPTIVE_EMBEDDING_MODEL}" ]]; then
    CMD+=(--adaptive-embedding-model "${ADAPTIVE_EMBEDDING_MODEL}")
fi
if [[ -n "${ADAPTIVE_EMBEDDING_HEAD_METHOD}" ]]; then
    CMD+=(--adaptive-embedding-head-method ${ADAPTIVE_EMBEDDING_HEAD_METHOD})
fi
if [[ -n "${ADAPTIVE_EMBEDDING_HEAD_EPOCHS}" ]]; then
    CMD+=(--adaptive-embedding-head-epochs ${ADAPTIVE_EMBEDDING_HEAD_EPOCHS})
fi
if [[ -n "${ADAPTIVE_EMBEDDING_HEAD_LR}" ]]; then
    CMD+=(--adaptive-embedding-head-lr ${ADAPTIVE_EMBEDDING_HEAD_LR})
fi
if [[ -n "${ADAPTIVE_EMBEDDING_HEAD_WEIGHT_DECAY}" ]]; then
    CMD+=(--adaptive-embedding-head-weight-decay ${ADAPTIVE_EMBEDDING_HEAD_WEIGHT_DECAY})
fi
if [[ -n "${ADAPTIVE_EMBEDDING_RETRAIN_ROUNDS}" ]]; then
    CMD+=(--adaptive-embedding-retrain-rounds ${ADAPTIVE_EMBEDDING_RETRAIN_ROUNDS})
fi
if [[ -n "${ADAPTIVE_EMBEDDING_SCORE_KEY}" ]]; then
    CMD+=(--adaptive-embedding-score-key ${ADAPTIVE_EMBEDDING_SCORE_KEY})
fi
if [[ "${ADAPTIVE_EMBEDDING_FULL_FINETUNE}" == "true" ]]; then
    CMD+=(--adaptive-embedding-full-finetune)
elif [[ "${ADAPTIVE_EMBEDDING_FULL_FINETUNE}" == "false" ]]; then
    CMD+=(--no-adaptive-embedding-full-finetune)
fi
if [[ -n "${ADAPTIVE_EMBEDDING_FINETUNE_COMMAND}" ]]; then
    CMD+=(--adaptive-embedding-finetune-command "${ADAPTIVE_EMBEDDING_FINETUNE_COMMAND}")
fi
if [[ "${EMBEDDING_PROXY_FAIL_ON_ERROR}" == "true" ]]; then
    CMD+=(--embedding-proxy-fail-on-error)
elif [[ "${EMBEDDING_PROXY_FAIL_ON_ERROR}" == "false" ]]; then
    CMD+=(--no-embedding-proxy-fail-on-error)
fi
if [[ "${RERUN_EMBEDDING_PROXY_ON_RESUME}" == "true" ]]; then
    CMD+=(--rerun-embedding-proxy-on-resume)
elif [[ "${RERUN_EMBEDDING_PROXY_ON_RESUME}" == "false" ]]; then
    CMD+=(--no-rerun-embedding-proxy-on-resume)
fi
if [[ "${TRAIN_NEURAL_OPERATORS}" == "true" ]]; then
    CMD+=(--train-neural-operators)
elif [[ "${TRAIN_NEURAL_OPERATORS}" == "false" ]]; then
    CMD+=(--no-train-neural-operators)
fi
if [[ -n "${NEURAL_OPERATORS_WHICH}" ]]; then
    CMD+=(--neural-operators-which ${NEURAL_OPERATORS_WHICH})
fi
if [[ -n "${NEURAL_OPERATORS_OUTPUT_DIR}" ]]; then
    CMD+=(--neural-operators-output-dir "${NEURAL_OPERATORS_OUTPUT_DIR}")
fi
if [[ -n "${NEURAL_OPERATORS_CTREEPO_ARGS}" ]]; then
    CMD+=(--neural-operators-ctreepo-args "${NEURAL_OPERATORS_CTREEPO_ARGS}")
fi
if [[ -n "${NEURAL_OPERATORS_MERGEABLE_ARGS}" ]]; then
    CMD+=(--neural-operators-mergeable-args "${NEURAL_OPERATORS_MERGEABLE_ARGS}")
fi
if [[ -n "${NEURAL_OPERATORS_CTREEPO_SEARCH_SPEC}" ]]; then
    CMD+=(--neural-operators-ctreepo-search-spec "${NEURAL_OPERATORS_CTREEPO_SEARCH_SPEC}")
fi
if [[ -n "${NEURAL_OPERATORS_MERGEABLE_SEARCH_SPEC}" ]]; then
    CMD+=(--neural-operators-mergeable-search-spec "${NEURAL_OPERATORS_MERGEABLE_SEARCH_SPEC}")
fi
if [[ "${NEURAL_OPERATORS_FAIL_FAST}" == "true" ]]; then
    CMD+=(--neural-operators-fail-fast)
elif [[ "${NEURAL_OPERATORS_FAIL_FAST}" == "false" ]]; then
    CMD+=(--no-neural-operators-fail-fast)
fi
if [[ "${NEURAL_OPERATORS_FAIL_ON_ERROR}" == "true" ]]; then
    CMD+=(--neural-operators-fail-on-error)
elif [[ "${NEURAL_OPERATORS_FAIL_ON_ERROR}" == "false" ]]; then
    CMD+=(--no-neural-operators-fail-on-error)
fi
if [[ "${RERUN_NEURAL_OPERATORS_ON_RESUME}" == "true" ]]; then
    CMD+=(--rerun-neural-operators-on-resume)
elif [[ "${RERUN_NEURAL_OPERATORS_ON_RESUME}" == "false" ]]; then
    CMD+=(--no-rerun-neural-operators-on-resume)
fi
if [[ "${NEURAL_OPERATORS_AUTO_WIRE_REPRESENTATION}" == "true" ]]; then
    CMD+=(--neural-operators-auto-wire-representation)
elif [[ "${NEURAL_OPERATORS_AUTO_WIRE_REPRESENTATION}" == "false" ]]; then
    CMD+=(--no-neural-operators-auto-wire-representation)
fi
if [[ "${HYBRID_ORACLE_SEEDED_ENSEMBLE}" == "true" ]]; then
    CMD+=(--hybrid-oracle-seeded-ensemble)
elif [[ "${HYBRID_ORACLE_SEEDED_ENSEMBLE}" == "false" ]]; then
    CMD+=(--no-hybrid-oracle-seeded-ensemble)
fi
if [[ -n "${HYBRID_SEED_LLM_MIN_WEIGHT}" ]]; then
    CMD+=(--hybrid-seed-llm-min-weight ${HYBRID_SEED_LLM_MIN_WEIGHT})
fi
if [[ -n "${HYBRID_SEED_LLM_MAX_WEIGHT}" ]]; then
    CMD+=(--hybrid-seed-llm-max-weight ${HYBRID_SEED_LLM_MAX_WEIGHT})
fi
if [[ -n "${HYBRID_OPERATOR_BOOST}" ]]; then
    CMD+=(--hybrid-operator-boost ${HYBRID_OPERATOR_BOOST})
fi

if [[ "${HONEST_CHUNKING}" == "true" ]]; then
    CMD+=(--honest-chunking)
elif [[ "${HONEST_CHUNKING}" == "false" ]]; then
    CMD+=(--no-honest-chunking)
fi

if [[ -n "${HONEST_BOUNDARY_FRACTION}" ]]; then
    CMD+=(--honest-boundary-fraction ${HONEST_BOUNDARY_FRACTION})
fi

if [[ -n "${HONEST_SPLIT_SEED}" ]]; then
    CMD+=(--honest-split-seed ${HONEST_SPLIT_SEED})
fi

if [[ "${THREE_LAYER_HONESTY}" == "true" ]]; then
    CMD+=(--three-layer-honesty)
elif [[ "${THREE_LAYER_HONESTY}" == "false" ]]; then
    CMD+=(--no-three-layer-honesty)
fi

if [[ -n "${THREE_LAYER_SEED}" ]]; then
    CMD+=(--three-layer-seed ${THREE_LAYER_SEED})
fi

if [[ -n "${THREE_LAYER_CHUNK_TRAIN_FRACTION}" ]]; then
    CMD+=(--three-layer-chunk-train-fraction ${THREE_LAYER_CHUNK_TRAIN_FRACTION})
fi

if [[ -n "${THREE_LAYER_SUMMARIZER_TRAIN_FRACTION}" ]]; then
    CMD+=(--three-layer-summarizer-train-fraction ${THREE_LAYER_SUMMARIZER_TRAIN_FRACTION})
fi

if [[ -n "${THREE_LAYER_ORACLE_TRAIN_FRACTION}" ]]; then
    CMD+=(--three-layer-oracle-train-fraction ${THREE_LAYER_ORACLE_TRAIN_FRACTION})
fi
if [[ "${TRAIN_GENERATOR}" == "true" ]]; then
    CMD+=(--train-generator)
elif [[ "${TRAIN_GENERATOR}" == "false" ]]; then
    CMD+=(--no-train-generator)
fi
if [[ -n "${GENERATOR_METHOD}" ]]; then
    CMD+=(--generator-method ${GENERATOR_METHOD})
fi
if [[ -n "${GENERATOR_MODEL_OVERRIDE}" ]]; then
    CMD+=(--generator-model "${GENERATOR_MODEL_OVERRIDE}")
fi
if [[ -n "${GENERATOR_OUTPUT_DIR}" ]]; then
    CMD+=(--generator-output-dir "${GENERATOR_OUTPUT_DIR}")
fi
if [[ "${GENERATOR_USE_LORA}" == "true" ]]; then
    CMD+=(--generator-use-lora)
elif [[ "${GENERATOR_USE_LORA}" == "false" ]]; then
    CMD+=(--no-generator-use-lora)
fi
if [[ -n "${GENERATOR_LEARNING_RATE}" ]]; then
    CMD+=(--generator-learning-rate ${GENERATOR_LEARNING_RATE})
fi
if [[ -n "${GENERATOR_EPOCHS}" ]]; then
    CMD+=(--generator-epochs ${GENERATOR_EPOCHS})
fi
if [[ -n "${GENERATOR_BATCH_SIZE}" ]]; then
    CMD+=(--generator-batch-size ${GENERATOR_BATCH_SIZE})
fi
if [[ "${GENERATOR_FAIL_ON_ERROR}" == "true" ]]; then
    CMD+=(--generator-fail-on-error)
elif [[ "${GENERATOR_FAIL_ON_ERROR}" == "false" ]]; then
    CMD+=(--no-generator-fail-on-error)
fi
if [[ "${RERUN_GENERATOR_ON_RESUME}" == "true" ]]; then
    CMD+=(--rerun-generator-on-resume)
elif [[ "${RERUN_GENERATOR_ON_RESUME}" == "false" ]]; then
    CMD+=(--no-rerun-generator-on-resume)
fi
if [[ -n "${GENERATOR_MIN_PREFERENCES}" ]]; then
    CMD+=(--generator-min-preferences ${GENERATOR_MIN_PREFERENCES})
fi

# Add any extra args passed through
CMD+=("${EXTRA_ARGS[@]}")

# Run the command
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Cleanup: Stop task backend server if we started it
# ============================================================================
if [[ -n "${TASK_SERVER_PID}" ]]; then
    log ""
    log "Stopping task backend server we started (PID: ${TASK_SERVER_PID})..."
    kill -TERM ${TASK_SERVER_PID} 2>/dev/null || true
    sleep 2
    log "Task backend server stopped"
fi

# Stop GenRM server if we started it
if [[ -n "${GENRM_PID}" ]]; then
    log ""
    log "Stopping GenRM server we started (PID: ${GENRM_PID})..."
    kill -TERM ${GENRM_PID} 2>/dev/null || true
    sleep 2
    log "GenRM server stopped"
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "========================================================================"
log "TRAINING PIPELINE COMPLETE"
log "========================================================================"
log "Exit code: ${EXIT_CODE}"
log "Finished:  $(date)"
log "Results:   ${OUTPUT_DIR}"
log ""

# Print final stats if available
if [ -f "${OUTPUT_DIR}/final_stats.json" ]; then
    log "Final Statistics:"
    python3 -c "
import json
with open('${OUTPUT_DIR}/final_stats.json') as f:
    stats = json.load(f)

print()
if 'baseline' in stats:
    print(f\"  Baseline Train MAE: {stats['baseline']['train']['mae']:.2f}\")
    print(f\"  Baseline Val MAE:   {stats['baseline']['val']['mae']:.2f}\")

if 'test' in stats:
    print(f\"  Test Pipeline MAE:  {stats['test']['pipeline']['mae']:.2f}\")
    print(f\"  Test Classifier MAE: {stats['test']['classifier']['mae']:.2f}\")

if 'rounds' in stats:
    print()
    print('  Optimization Rounds:')
    for r in stats['rounds']:
        if 'error' not in r:
            val_mae = r.get('val_eval', {}).get('mae', 'N/A')
            if isinstance(val_mae, float):
                val_mae = f'{val_mae:.2f}'
            print(f\"    Round {r['round']}: {r['metric_before']:.3f} -> {r['metric_after']:.3f} (Val MAE: {val_mae})\")
" 2>/dev/null || true
fi

PDF_PATH="${OUTPUT_DIR}/score_report.pdf"
REPORT_LOG_PATH="${OUTPUT_DIR}/report_score_run.log"
if [[ -f "${PDF_PATH}" ]]; then
    log "Score report PDF: ${PDF_PATH}"
elif [[ -f "${PROJECT_ROOT}/scripts/report_score_run.py" ]]; then
    SPLITS=()
    for split_name in train test val; do
        if [[ -f "${OUTPUT_DIR}/${split_name}_score_report.jsonl" ]]; then
            SPLITS+=("${split_name}")
        fi
    done
    if [[ ${#SPLITS[@]} -gt 0 ]]; then
        log "Score report PDF missing; generating fallback PDF report..."
        "${ACTIVE_PY_ENV}/bin/python" \
            "${PROJECT_ROOT}/scripts/report_score_run.py" \
            --output-dir "${OUTPUT_DIR}" \
            --splits "${SPLITS[@]}" >> "${LOG_FILE}" 2>&1 || true
    fi
    if [[ -f "${PDF_PATH}" ]]; then
        log "Score report PDF: ${PDF_PATH}"
    else
        log "Score report PDF not found (checked ${PDF_PATH})"
    fi
else
    log "Score report script not found; cannot generate PDF fallback"
fi
if [[ -f "${REPORT_LOG_PATH}" ]]; then
    log "PDF generation log: ${REPORT_LOG_PATH}"
fi

log ""
log "To view full results:"
log "  cat ${OUTPUT_DIR}/final_stats.json | python -m json.tool"
log ""

exit ${EXIT_CODE}
