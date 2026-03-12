#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="outputs/lda_tree_recovery_production_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ROOT_DEFAULT}"
EXACT_CPU_CONCURRENCY=64
LEARNED_CPU_CONCURRENCY=16
GPU_TOKENS="auto"
LAUNCH=1
LAUNCH_EXACT=1
LAUNCH_LEARNED_CPU=1
LAUNCH_LEARNED_GPU=1

usage() {
  cat <<'EOF'
Usage: scripts/launch_lda_tree_recovery_production_sweeps.sh [options]

Options:
  --output-root PATH              Output root for manifests, logs, and results.
  --exact-cpu-concurrency N       xargs worker count for the exact CPU sweep. Default: 64.
  --learned-cpu-concurrency N     xargs worker count for the learned CPU shadow sweep. Default: 16.
  --gpu-tokens SPEC               GPU token spec for the learned sweep. Default: auto.
  --gpu-ids SPEC                  Alias for --gpu-tokens.
  --skip-exact-launch             Build exact CPU commands but do not launch the exact lane.
  --skip-learned-cpu-launch       Build learned CPU-shadow commands but do not launch that lane.
  --skip-learned-gpu-launch       Build learned GPU commands but do not launch that lane.
  --no-launch                     Only build manifests and runner scripts; do not start jobs.
  -h, --help                      Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --exact-cpu-concurrency)
      EXACT_CPU_CONCURRENCY="$2"
      shift 2
      ;;
    --learned-cpu-concurrency)
      LEARNED_CPU_CONCURRENCY="$2"
      shift 2
      ;;
    --gpu-tokens|--gpu-ids)
      GPU_TOKENS="$2"
      shift 2
      ;;
    --skip-exact-launch)
      LAUNCH_EXACT=0
      shift
      ;;
    --skip-learned-cpu-launch)
      LAUNCH_LEARNED_CPU=0
      shift
      ;;
    --skip-learned-gpu-launch)
      LAUNCH_LEARNED_GPU=0
      shift
      ;;
    --no-launch)
      LAUNCH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

EXACT_ROOT="${OUTPUT_ROOT}/exact_cpu"
LEARNED_GPU_ROOT="${OUTPUT_ROOT}/learned_gpu"
LEARNED_CPU_ROOT="${OUTPUT_ROOT}/learned_cpu_shadow"
LEARNED_SHARED_CACHE_ROOT="${OUTPUT_ROOT}/learned_shared_cache"
LEARNED_WORLD_CACHE_DIR="${LEARNED_SHARED_CACHE_ROOT}/world_cache"
LEARNED_PREPARED_CACHE_DIR="${LEARNED_SHARED_CACHE_ROOT}/prepared_cache"

mkdir -p "${EXACT_ROOT}" "${LEARNED_GPU_ROOT}" "${LEARNED_CPU_ROOT}" "${LEARNED_WORLD_CACHE_DIR}" "${LEARNED_PREPARED_CACHE_DIR}"

EXACT_COMMANDS="${EXACT_ROOT}/commands.txt"
LEARNED_GPU_COMMANDS="${LEARNED_GPU_ROOT}/commands.txt"
LEARNED_CPU_COMMANDS="${LEARNED_CPU_ROOT}/commands.txt"

EXACT_LOG="${EXACT_ROOT}/sweep.log"
LEARNED_CPU_LOG="${LEARNED_CPU_ROOT}/sweep.log"
LEARNED_GPU_LOG="${LEARNED_GPU_ROOT}/sweep.log"

EXACT_PID="${EXACT_ROOT}/sweep.pid"
LEARNED_CPU_PID="${LEARNED_CPU_ROOT}/sweep.pid"
LEARNED_GPU_PID="${LEARNED_GPU_ROOT}/sweep.pid"

EXACT_RUNNER="${EXACT_ROOT}/run_sweep.sh"
LEARNED_CPU_RUNNER="${LEARNED_CPU_ROOT}/run_sweep.sh"
LEARNED_GPU_RUNNER="${LEARNED_GPU_ROOT}/run_sweep.sh"

SPEC_PATH="${OUTPUT_ROOT}/sweep_spec.txt"

echo "building exact CPU commands"
venv/bin/python scripts/build_lda_tree_recovery_cmds.py \
  --out-cmds "${EXACT_COMMANDS}" \
  --output-root "${EXACT_ROOT}/results" \
  --leaf-tokens "384 192 96 48 24 16" \
  --doc-topic-concentrations "0.2 0.6 1.5" \
  --quadratic-utility-weights "0 1 2" \
  --seeds "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" \
  --test-docs 2048 \
  --skip-existing

echo "building learned GPU commands"
venv/bin/python scripts/build_lda_tree_recovery_learned_world_batch_cmds.py \
  --out-cmds "${LEARNED_GPU_COMMANDS}" \
  --output-root "${LEARNED_GPU_ROOT}/results" \
  --world-cache-dir "${LEARNED_WORLD_CACHE_DIR}" \
  --prepared-cache-dir "${LEARNED_PREPARED_CACHE_DIR}" \
  --doc-topic-concentrations "0.2 0.6 1.5" \
  --quadratic-utility-weights "0 1 2" \
  --leaf-tokens-grid "384 192 96 16" \
  --train-docs-grid "128 512 2048" \
  --state-dims "8 16 32 64 128 256 512" \
  --max-train-docs-capacity 2048 \
  --test-docs 512 \
  --full-hidden-dim 256 \
  --full-n-layers 3 \
  --n-epochs 80 \
  --batch-size 128 \
  --device auto \
  --seeds "0 1 2 3 4 5 6 7" \
  --skip-existing

echo "building learned CPU shadow commands"
venv/bin/python scripts/build_lda_tree_recovery_learned_world_batch_cmds.py \
  --out-cmds "${LEARNED_CPU_COMMANDS}" \
  --output-root "${LEARNED_CPU_ROOT}/results" \
  --world-cache-dir "${LEARNED_WORLD_CACHE_DIR}" \
  --prepared-cache-dir "${LEARNED_PREPARED_CACHE_DIR}" \
  --doc-topic-concentrations "0.6" \
  --quadratic-utility-weights "0 1 2" \
  --leaf-tokens-grid "384 96 16" \
  --train-docs-grid "128 512 2048" \
  --state-dims "8 32 128 512" \
  --max-train-docs-capacity 2048 \
  --test-docs 512 \
  --full-hidden-dim 256 \
  --full-n-layers 3 \
  --n-epochs 80 \
  --batch-size 128 \
  --device cpu \
  --seeds "0 1 2 3" \
  --skip-existing

EXACT_COUNT="$(wc -l < "${EXACT_COMMANDS}" | tr -d ' ')"
LEARNED_GPU_COUNT="$(wc -l < "${LEARNED_GPU_COMMANDS}" | tr -d ' ')"
LEARNED_CPU_COUNT="$(wc -l < "${LEARNED_CPU_COMMANDS}" | tr -d ' ')"

cat > "${EXACT_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file '${EXACT_COMMANDS}' \
  --cpu-workers ${EXACT_CPU_CONCURRENCY} \
  --gpu-tokens none \
  --log-dir '${EXACT_ROOT}/queue_logs'
EOF
chmod +x "${EXACT_RUNNER}"

cat > "${LEARNED_CPU_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file '${LEARNED_CPU_COMMANDS}' \
  --cpu-workers ${LEARNED_CPU_CONCURRENCY} \
  --gpu-tokens none \
  --log-dir '${LEARNED_CPU_ROOT}/queue_logs'
EOF
chmod +x "${LEARNED_CPU_RUNNER}"

{
  echo "generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "output_root=${OUTPUT_ROOT}"
  echo "exact_root=${EXACT_ROOT}"
  echo "learned_gpu_root=${LEARNED_GPU_ROOT}"
  echo "learned_cpu_root=${LEARNED_CPU_ROOT}"
  echo "learned_world_cache_dir=${LEARNED_WORLD_CACHE_DIR}"
  echo "learned_prepared_cache_dir=${LEARNED_PREPARED_CACHE_DIR}"
  echo "exact_cpu_concurrency=${EXACT_CPU_CONCURRENCY}"
  echo "learned_cpu_concurrency=${LEARNED_CPU_CONCURRENCY}"
  echo "gpu_tokens=${GPU_TOKENS}"
  echo "exact_commands=${EXACT_COUNT}"
  echo "learned_gpu_commands=${LEARNED_GPU_COUNT}"
  echo "learned_cpu_commands=${LEARNED_CPU_COUNT}"
  echo "exact_matrix=leaf(384,192,96,48,24,16) x dtc(0.2,0.6,1.5) x quadratic_weight(0,1,2) x seeds(32), test_docs=2048"
  echo "learned_gpu_matrix=bundled by (dtc,seed) over leaf(384,192,96,16) x quadratic_weight(0,1,2) x train(128,512,2048) x state(8,16,32,64,128,256,512), dtc(0.2,0.6,1.5), seeds(8), test_docs=512, epochs=80"
  echo "learned_cpu_shadow_matrix=bundled by (dtc,seed) over leaf(384,96,16) x quadratic_weight(0,1,2) x train(128,512,2048) x state(8,32,128,512), dtc(0.6), seeds(4), test_docs=512, epochs=80"
} > "${SPEC_PATH}"

echo "prepared lda tree recovery production sweeps"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  exact commands: ${EXACT_COUNT}"
echo "  learned gpu commands: ${LEARNED_GPU_COUNT}"
echo "  learned cpu commands: ${LEARNED_CPU_COUNT}"
echo "  spec: ${SPEC_PATH}"

if [[ "${LAUNCH}" -eq 0 ]]; then
  exit 0
fi

launch_detached() {
  local lane_name="$1"
  local runner="$2"
  local log_path="$3"
  local pid_path="$4"
  if [[ -f "${pid_path}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_path}" || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "  ${lane_name} already active: ${pid_path} -> ${old_pid}"
      return 0
    fi
  fi
  setsid "${runner}" </dev/null > "${log_path}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${pid_path}"
  echo "  launched ${lane_name}: ${runner} (pid ${pid})"
}

cat > "${LEARNED_GPU_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
venv/bin/python scripts/run_simulation_resource_queue.py \
  --cmd-file '${LEARNED_GPU_COMMANDS}' \
  --cpu-workers 1 \
  --gpu-tokens '${GPU_TOKENS}' \
  --log-dir '${LEARNED_GPU_ROOT}/queue_logs'
EOF
chmod +x "${LEARNED_GPU_RUNNER}"

if [[ "${LAUNCH_EXACT}" -eq 1 ]]; then
  launch_detached "exact_cpu" "${EXACT_RUNNER}" "${EXACT_LOG}" "${EXACT_PID}"
fi
if [[ "${LAUNCH_LEARNED_CPU}" -eq 1 ]]; then
  launch_detached "learned_cpu_shadow" "${LEARNED_CPU_RUNNER}" "${LEARNED_CPU_LOG}" "${LEARNED_CPU_PID}"
fi
if [[ "${LAUNCH_LEARNED_GPU}" -eq 1 ]]; then
  launch_detached "learned_gpu" "${LEARNED_GPU_RUNNER}" "${LEARNED_GPU_LOG}" "${LEARNED_GPU_PID}"
fi

echo "logs:"
echo "  exact: ${EXACT_LOG}"
echo "  learned_cpu: ${LEARNED_CPU_LOG}"
echo "  learned_gpu: ${LEARNED_GPU_LOG}"
