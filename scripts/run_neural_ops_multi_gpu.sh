#!/usr/bin/env bash
# Run parallel neural-operator training attempts across multiple GPUs.
#
# This launcher:
# 1) Ensures embedding server is running (default: starts on embedding GPU).
# 2) Fans out N attempts per training GPU in parallel.
# 3) Pins each attempt with CUDA_VISIBLE_DEVICES=<gpu> and uses --device cuda:0
#    inside the worker process, so each run targets its assigned GPU reliably.
# 4) Waits for completion and writes a simple status manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

EMBEDDING_GPU="0"
EMBEDDING_GPUS=""
EMBEDDING_PORTS=""
TRAIN_GPUS="1,2,3"
ATTEMPTS_PER_GPU=1
SEED_BASE=100

EMBEDDING_URL="http://localhost:8003/v1"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
WHICH="both"

EPOCHS=25
TRAIN_SAMPLES=400
VAL_SAMPLES=120
TEST_SAMPLES=120

START_EMBEDDING=1
MAX_WAIT_SECONDS=300
OUTPUT_ROOT=""
CTREEPO_ARGS_EXTRA=""
MERGEABLE_ARGS_EXTRA=""
DRY_RUN=0
ALLOW_CONCURRENT_EXISTING=0
PROGRESS_INTERVAL_SECONDS=15

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

usage() {
  cat <<'EOF'
Run parallel neural-operator attempts with one command.

Usage:
  ./scripts/run_neural_ops_multi_gpu.sh [OPTIONS]

Options:
  --embedding-gpu ID             GPU for embedding server (default: 0)
  --embedding-gpus IDS           Comma-separated embedding GPUs (overrides --embedding-gpu)
  --embedding-ports PORTS        Comma-separated embedding ports (default: consecutive from embedding-url port)
  --train-gpus IDS               Comma-separated training GPUs (default: 1,2,3)
  --attempts-per-gpu N           Parallel attempts per training GPU (default: 1)
  --seed-base N                  Base seed for attempts (default: 100)
  --which MODE                   both|ctreepo|mergeable_sketch (default: both)
  --epochs N                     Mergeable sketch epochs (default: 25)
  --train-samples N              Mergeable sketch train samples (default: 400)
  --val-samples N                Mergeable sketch val samples (default: 120)
  --test-samples N               Mergeable sketch test samples (default: 120)
  --embedding-url URL            Embedding endpoint (default: http://localhost:8003/v1)
  --embedding-model MODEL        Embedding model id (default: Qwen/Qwen3-Embedding-8B)
  --ctreepo-args-extra STR       Extra args appended to CTreePO args
  --mergeable-args-extra STR     Extra args appended to mergeable args
  --output-root PATH             Output root (default: outputs/neural_ops_multi_<timestamp>)
  --no-start-embedding           Require embedding server to already be running
  --max-wait-seconds N           Embedding readiness timeout (default: 300)
  --dry-run                      Print commands without running
  --allow-concurrent-existing    Allow launch even if other neural-op workers are running
  --progress-interval-seconds N  Progress heartbeat interval (default: 15)
  -h, --help                     Show this help

Example:
  ./scripts/run_neural_ops_multi_gpu.sh \
    --embedding-gpu 0 \
    --train-gpus 1,2,3 \
    --attempts-per-gpu 2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-gpu) EMBEDDING_GPU="$2"; shift 2 ;;
    --embedding-gpus) EMBEDDING_GPUS="$2"; shift 2 ;;
    --embedding-ports) EMBEDDING_PORTS="$2"; shift 2 ;;
    --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
    --attempts-per-gpu) ATTEMPTS_PER_GPU="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --which) WHICH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --train-samples) TRAIN_SAMPLES="$2"; shift 2 ;;
    --val-samples) VAL_SAMPLES="$2"; shift 2 ;;
    --test-samples) TEST_SAMPLES="$2"; shift 2 ;;
    --embedding-url) EMBEDDING_URL="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --ctreepo-args-extra) CTREEPO_ARGS_EXTRA="$2"; shift 2 ;;
    --mergeable-args-extra) MERGEABLE_ARGS_EXTRA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --no-start-embedding) START_EMBEDDING=0; shift ;;
    --max-wait-seconds) MAX_WAIT_SECONDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-concurrent-existing) ALLOW_CONCURRENT_EXISTING=1; shift ;;
    --progress-interval-seconds) PROGRESS_INTERVAL_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$PROJECT_ROOT/outputs/neural_ops_multi_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_ROOT"

STATUS_FILE="$OUTPUT_ROOT/launch_status.tsv"
echo -e "run_name\tgpu\tseed\tpid\texit_code\tembedding_url\tlog_path\toutput_dir" > "$STATUS_FILE"

if [[ -z "$EMBEDDING_GPUS" ]]; then
  EMBEDDING_GPUS="$EMBEDDING_GPU"
fi

default_embedding_port="$(echo "$EMBEDDING_URL" | sed -n 's#.*:\([0-9][0-9]*\)/v1.*#\1#p')"
if [[ -z "$default_embedding_port" ]]; then
  default_embedding_port="8003"
fi

IFS=',' read -r -a EMB_GPU_LIST_RAW <<< "$EMBEDDING_GPUS"
declare -a EMB_GPU_LIST=()
for item in "${EMB_GPU_LIST_RAW[@]}"; do
  trimmed="$(echo "$item" | xargs)"
  if [[ -n "$trimmed" ]]; then
    EMB_GPU_LIST+=("$trimmed")
  fi
done
if [[ "${#EMB_GPU_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: no embedding GPUs resolved" >&2
  exit 2
fi

declare -a EMB_PORT_LIST=()
if [[ -n "$EMBEDDING_PORTS" ]]; then
  IFS=',' read -r -a EMB_PORT_LIST_RAW <<< "$EMBEDDING_PORTS"
  for item in "${EMB_PORT_LIST_RAW[@]}"; do
    trimmed="$(echo "$item" | xargs)"
    if [[ -n "$trimmed" ]]; then
      EMB_PORT_LIST+=("$trimmed")
    fi
  done
  if [[ "${#EMB_PORT_LIST[@]}" -ne "${#EMB_GPU_LIST[@]}" ]]; then
    echo "ERROR: --embedding-ports count (${#EMB_PORT_LIST[@]}) must match --embedding-gpus count (${#EMB_GPU_LIST[@]})" >&2
    exit 2
  fi
else
  for ((i=0; i<${#EMB_GPU_LIST[@]}; i++)); do
    EMB_PORT_LIST+=("$((default_embedding_port + i))")
  done
fi

declare -a EMBEDDING_URL_LIST=()
if [[ "$START_EMBEDDING" -eq 0 && "${#EMB_GPU_LIST[@]}" -eq 1 && -z "$EMBEDDING_PORTS" ]]; then
  EMBEDDING_URL_LIST+=("$EMBEDDING_URL")
else
  for port in "${EMB_PORT_LIST[@]}"; do
    EMBEDDING_URL_LIST+=("http://localhost:${port}/v1")
  done
fi

echo "===================================================="

if [[ "$ALLOW_CONCURRENT_EXISTING" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
  if pgrep -f 'scripts/train_neural_operators.py|scripts/train_rile_embedding_sketch.py|scripts/train_ctreepo.py' >/dev/null 2>&1; then
    echo "ERROR: existing neural-operator worker processes detected." >&2
    echo "Stop them first, or rerun with --allow-concurrent-existing." >&2
    echo "Hint: pkill -f 'scripts/train_neural_operators.py|scripts/train_rile_embedding_sketch.py|scripts/train_ctreepo.py'" >&2
    exit 1
  fi
fi
echo "Neural Ops Multi-GPU Launcher"
echo "===================================================="
echo "Output root:         $OUTPUT_ROOT"
echo "Embedding GPUs:      $EMBEDDING_GPUS"
echo "Embedding ports:     $(IFS=,; echo "${EMB_PORT_LIST[*]}")"
echo "Embedding URLs:      $(IFS=,; echo "${EMBEDDING_URL_LIST[*]}")"
echo "Training GPUs:       $TRAIN_GPUS"
echo "Attempts per GPU:    $ATTEMPTS_PER_GPU"
echo "Which:               $WHICH"
echo "Embedding model:     $EMBEDDING_MODEL"
echo "Progress interval:   ${PROGRESS_INTERVAL_SECONDS}s"
echo "Python:              $PYTHON_BIN"
echo "===================================================="

if [[ "$START_EMBEDDING" -eq 1 ]]; then
  for ((i=0; i<${#EMB_GPU_LIST[@]}; i++)); do
    emb_gpu="${EMB_GPU_LIST[$i]}"
    emb_port="${EMB_PORT_LIST[$i]}"
    emb_log="$PROJECT_ROOT/logs/embedding_model_${emb_port}.log"
    START_CMD=(
      "$SCRIPT_DIR/start_embedding_server.sh"
      --port "$emb_port"
      --cuda-devices "$emb_gpu"
      --log-file "$emb_log"
      --max-wait-seconds "$MAX_WAIT_SECONDS"
    )
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] ${START_CMD[*]}"
    else
      "${START_CMD[@]}"
    fi
  done
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  for url in "${EMBEDDING_URL_LIST[@]}"; do
    ready_url="${url%/}/models"
    echo "Checking embedding server readiness: $ready_url"
    if ! curl -fsS "$ready_url" >/dev/null; then
      echo "ERROR: embedding server not reachable at $ready_url" >&2
      exit 1
    fi
  done
fi

IFS=',' read -r -a GPU_LIST <<< "$TRAIN_GPUS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: --train-gpus is empty" >&2
  exit 2
fi

declare -a PIDS=()
declare -a META=()
launch_index=0

for GPU_RAW in "${GPU_LIST[@]}"; do
  GPU="$(echo "$GPU_RAW" | xargs)"
  if [[ -z "$GPU" ]]; then
    continue
  fi
  for (( attempt=1; attempt<=ATTEMPTS_PER_GPU; attempt++ )); do
    SEED=$((SEED_BASE + launch_index))
    RUN_NAME="gpu${GPU}_run${attempt}_seed${SEED}"
    OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
    LOG_PATH="$OUT_DIR/run.log"
    mkdir -p "$OUT_DIR"

    # train_ctreepo.py requires data-selection args; default to pilot for robust smoke runs.
    CTREEPO_ARGS="--pilot --device cuda:0"
    if [[ -n "$CTREEPO_ARGS_EXTRA" ]]; then
      CTREEPO_ARGS="$CTREEPO_ARGS $CTREEPO_ARGS_EXTRA"
    fi

    MERGEABLE_ARGS="--device cuda:0 --epochs $EPOCHS --train-samples $TRAIN_SAMPLES --val-samples $VAL_SAMPLES --test-samples $TEST_SAMPLES"
    if [[ -n "$MERGEABLE_ARGS_EXTRA" ]]; then
      MERGEABLE_ARGS="$MERGEABLE_ARGS $MERGEABLE_ARGS_EXTRA"
    fi

    embedding_url_for_run="${EMBEDDING_URL_LIST[$((launch_index % ${#EMBEDDING_URL_LIST[@]}))]}"

    CMD=(
      "$PYTHON_BIN"
      "$SCRIPT_DIR/train_neural_operators.py"
      --output-dir "$OUT_DIR"
      --which "$WHICH"
      --seed "$SEED"
      --embedding-url "$embedding_url_for_run"
      --embedding-model "$EMBEDDING_MODEL"
      --ctreepo-args "$CTREEPO_ARGS"
      --mergeable-args "$MERGEABLE_ARGS"
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]} > $LOG_PATH 2>&1 &"
      echo -e "$RUN_NAME\t$GPU\t$SEED\t-\t-\t$embedding_url_for_run\t$LOG_PATH\t$OUT_DIR" >> "$STATUS_FILE"
    else
      (
        export CUDA_VISIBLE_DEVICES="$GPU"
        exec "${CMD[@]}"
      ) >"$LOG_PATH" 2>&1 &
      PID=$!
      PIDS+=("$PID")
      META+=("$RUN_NAME|$GPU|$SEED|$PID|$embedding_url_for_run|$LOG_PATH|$OUT_DIR")
      echo "Launched $RUN_NAME on GPU $GPU via $embedding_url_for_run (pid=$PID)"
    fi

    launch_index=$((launch_index + 1))
  done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete. Status: $STATUS_FILE"
  exit 0
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "ERROR: no training runs were launched" >&2
  exit 1
fi

echo "Waiting for ${#PIDS[@]} run(s) to complete..."
FAILURES=0

progress_snapshot() {
  local stage_starting=0
  local stage_ctreepo=0
  local stage_mergeable=0
  local stage_done=0
  local stage_failed=0
  local running=0
  local completed=0
  local failed=0
  local rec RUN_NAME GPU SEED PID EMB_URL LOG_PATH OUT_DIR
  local stage

  for rec in "${META[@]}"; do
    IFS='|' read -r RUN_NAME GPU SEED PID EMB_URL LOG_PATH OUT_DIR <<< "$rec"

    if [[ -n "${DONE_BY_PID[$PID]:-}" ]]; then
      completed=$((completed + 1))
      if [[ "${DONE_BY_PID[$PID]}" -ne 0 ]]; then
        failed=$((failed + 1))
      fi
      continue
    fi

    running=$((running + 1))
    if [[ ! -f "$LOG_PATH" ]]; then
      stage_starting=$((stage_starting + 1))
      continue
    fi

    if rg -q '\[mergeable_sketch\] completed successfully' "$LOG_PATH"; then
      stage_done=$((stage_done + 1))
    elif rg -q '\[(ctreepo|mergeable_sketch)\] failed' "$LOG_PATH"; then
      stage_failed=$((stage_failed + 1))
    elif rg -q '\[mergeable_sketch\] running' "$LOG_PATH"; then
      stage_mergeable=$((stage_mergeable + 1))
    elif rg -q '\[ctreepo\] running' "$LOG_PATH"; then
      stage_ctreepo=$((stage_ctreepo + 1))
    else
      stage_starting=$((stage_starting + 1))
    fi
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] progress: running=$running completed=$completed failed=$failed stages(starting=$stage_starting ctreepo=$stage_ctreepo mergeable=$stage_mergeable done=$stage_done failed=$stage_failed)"
}

declare -A DONE_BY_PID=()
TOTAL_RUNS="${#META[@]}"
COMPLETED_RUNS=0

while [[ "$COMPLETED_RUNS" -lt "$TOTAL_RUNS" ]]; do
  for rec in "${META[@]}"; do
    IFS='|' read -r RUN_NAME GPU SEED PID EMB_URL LOG_PATH OUT_DIR <<< "$rec"
    if [[ -n "${DONE_BY_PID[$PID]:-}" ]]; then
      continue
    fi
    if kill -0 "$PID" >/dev/null 2>&1; then
      continue
    fi

    EXIT_CODE=0
    if ! wait "$PID"; then
      EXIT_CODE=$?
    fi
    DONE_BY_PID[$PID]="$EXIT_CODE"
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
    if [[ "$EXIT_CODE" -ne 0 ]]; then
      FAILURES=$((FAILURES + 1))
    fi
    echo -e "$RUN_NAME\t$GPU\t$SEED\t$PID\t$EXIT_CODE\t$EMB_URL\t$LOG_PATH\t$OUT_DIR" >> "$STATUS_FILE"
    echo "Completed $RUN_NAME (gpu=$GPU, seed=$SEED, url=$EMB_URL, exit=$EXIT_CODE)"
  done

  if [[ "$COMPLETED_RUNS" -lt "$TOTAL_RUNS" ]]; then
    progress_snapshot
    sleep "$PROGRESS_INTERVAL_SECONDS"
  fi
done

echo "Status file: $STATUS_FILE"
if [[ "$FAILURES" -gt 0 ]]; then
  echo "Finished with failures: $FAILURES run(s) failed."
  exit 1
fi

echo "All runs completed successfully."
exit 0
