#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="outputs/simulation_intent_curated_20260305_214500/segment_lda_production_cpu_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ROOT_DEFAULT}"
CONCURRENCY=48
TEST_DOCS=512
LAUNCH=1

usage() {
  cat <<'EOF'
Usage: scripts/launch_segment_lda_production_cpu_sweep.sh [options]

Options:
  --output-root PATH   Output root for commands, logs, and simulation files.
  --concurrency N      Parallel worker count for xargs. Default: 48.
  --test-docs N        Held-out test docs per run. Default: 512.
  --no-launch          Only build the command manifest; do not start the sweep.
  -h, --help           Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --test-docs)
      TEST_DOCS="$2"
      shift 2
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
COMMANDS="${OUTPUT_ROOT}/commands.txt"
LOG_PATH="${OUTPUT_ROOT}/sweep.log"
PID_PATH="${OUTPUT_ROOT}/sweep.pid"
META_PATH="${OUTPUT_ROOT}/sweep_spec.txt"
RUNNER_PATH="${OUTPUT_ROOT}/run_sweep.sh"
: > "${COMMANDS}"

PYTHON_BIN="venv/bin/python"
SCRIPT="scripts/run_segment_lda_ops_weight_recovery_world_batch.py"

TRAIN_DOCS_CONTROL=(16 64 256 1024 4096 16384)
TRAIN_DOCS_TREE=(16 64 256 1024 4096)
CONTROL_LAMBDAS=(0 0.5 1 2)
TREE_LAMBDAS=(0 1 2)
AUDITS=(0.01 0.05 0.1 0.2 0.5 1.0)
TREE_LEAVES=(96 64 48 32 24 16)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

DOC_COUNT=0
BOUNDARY_COUNT=0
TREE_COUNT=0
BATCH_COUNT=0

join_by_space() {
  local joined=""
  for item in "$@"; do
    if [[ -n "${joined}" ]]; then
      joined+=" "
    fi
    joined+="${item}"
  done
  printf '%s' "${joined}"
}

TRAIN_DOCS_CONTROL_STR="$(join_by_space "${TRAIN_DOCS_CONTROL[@]}")"
TRAIN_DOCS_TREE_STR="$(join_by_space "${TRAIN_DOCS_TREE[@]}")"
CONTROL_LAMBDAS_STR="$(join_by_space "${CONTROL_LAMBDAS[@]}")"
TREE_LAMBDAS_STR="$(join_by_space "${TREE_LAMBDAS[@]}")"
AUDITS_STR="$(join_by_space "${AUDITS[@]}")"

append_cmd() {
  printf '%s\n' "$1" >> "${COMMANDS}"
}

emit_batch_run() {
  local family="$1"
  local process="$2"
  local leaf_tokens="$3"
  local train_docs_grid="$4"
  local lambda_grid="$5"
  local audit_grid="$6"
  local seed="$7"
  local world_train_docs_capacity="$8"
  local world_phi_extra_docs_capacity="$9"

  local cmd
  cmd="${PYTHON_BIN} ${SCRIPT}"
  cmd+=" --output-root ${OUTPUT_ROOT}"
  cmd+=" --family ${family}"
  cmd+=" --topic-process ${process}"
  cmd+=" --leaf-tokens ${leaf_tokens}"
  cmd+=" --seed ${seed}"
  cmd+=" --train-docs-grid '${train_docs_grid}'"
  cmd+=" --lambda-grid '${lambda_grid}'"
  cmd+=" --audit-fractions '${audit_grid}'"
  cmd+=" --test-docs ${TEST_DOCS}"
  cmd+=" --topic-phi-estimators true --topic-phi-docs-grid 0"
  cmd+=" --world-cache-dir ${OUTPUT_ROOT}/world_cache"
  cmd+=" --world-train-docs-capacity ${world_train_docs_capacity}"
  cmd+=" --world-test-docs-capacity ${TEST_DOCS}"
  cmd+=" --world-phi-extra-docs-capacity ${world_phi_extra_docs_capacity}"
  cmd+=" --run-all-feature-modes"
  cmd+=" --min-tokens 384 --max-tokens 384"
  append_cmd "${cmd}"
  BATCH_COUNT=$((BATCH_COUNT + 1))
}

# 1. Whole-document controls: one leaf, bag-of-words only.
for seed in "${SEEDS[@]}"; do
  emit_batch_run "whole_document_controls" "bag_of_words" 384 "${TRAIN_DOCS_CONTROL_STR}" "${CONTROL_LAMBDAS_STR}" "1.0" "${seed}" 16384 0
  DOC_COUNT=$((DOC_COUNT + ${#CONTROL_LAMBDAS[@]} * ${#TRAIN_DOCS_CONTROL[@]}))
done

# 2. One-boundary controls: exactly two leaves, both bag-of-words and segments.
for process in bag_of_words segments; do
  for seed in "${SEEDS[@]}"; do
    if [[ "${process}" == "bag_of_words" ]]; then
      emit_batch_run "one_boundary_controls" "${process}" 192 "${TRAIN_DOCS_CONTROL_STR}" "${CONTROL_LAMBDAS_STR}" "1.0" "${seed}" 16384 0
    else
      emit_batch_run "one_boundary_controls" "${process}" 192 "${TRAIN_DOCS_CONTROL_STR}" "${CONTROL_LAMBDAS_STR}" "1.0" "${seed}" 16384 0
    fi
    BOUNDARY_COUNT=$((BOUNDARY_COUNT + ${#CONTROL_LAMBDAS[@]} * ${#TRAIN_DOCS_CONTROL[@]}))
  done
done

# 3. Full-tree sweeps: deeper trees, both topic processes, multiple audit budgets.
for process in bag_of_words segments; do
  for leaf_tokens in "${TREE_LEAVES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if [[ "${process}" == "bag_of_words" ]]; then
        emit_batch_run "full_tree_sweeps" "${process}" "${leaf_tokens}" "${TRAIN_DOCS_TREE_STR}" "${TREE_LAMBDAS_STR}" "${AUDITS_STR}" "${seed}" 16384 0
      else
        emit_batch_run "full_tree_sweeps" "${process}" "${leaf_tokens}" "${TRAIN_DOCS_TREE_STR}" "${TREE_LAMBDAS_STR}" "${AUDITS_STR}" "${seed}" 4096 0
      fi
      TREE_COUNT=$((TREE_COUNT + ${#AUDITS[@]} * ${#TREE_LAMBDAS[@]} * ${#TRAIN_DOCS_TREE[@]}))
    done
  done
done

TOTAL_EVALS=$((DOC_COUNT + BOUNDARY_COUNT + TREE_COUNT))

cat > "${META_PATH}" <<EOF
generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
output_root=${OUTPUT_ROOT}
commands=${COMMANDS}
log=${LOG_PATH}
pid_file=${PID_PATH}
runner=${RUNNER_PATH}
concurrency=${CONCURRENCY}
test_docs=${TEST_DOCS}
whole_document_controls=${DOC_COUNT}
one_boundary_controls=${BOUNDARY_COUNT}
full_tree_sweeps=${TREE_COUNT}
total_batch_commands=${BATCH_COUNT}
total_evaluations=${TOTAL_EVALS}
EOF

echo "prepared sweep"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  commands: ${COMMANDS}"
echo "  total_batch_commands: ${BATCH_COUNT}"
echo "  total_evaluations: ${TOTAL_EVALS}"
echo "  whole_document_controls: ${DOC_COUNT}"
echo "  one_boundary_controls: ${BOUNDARY_COUNT}"
echo "  full_tree_sweeps: ${TREE_COUNT}"

if [[ "${LAUNCH}" -eq 0 ]]; then
  exit 0
fi

if [[ -f "${PID_PATH}" ]]; then
  OLD_PID="$(cat "${PID_PATH}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "existing sweep appears to be running with pid ${OLD_PID}; refusing to launch a second one" >&2
    exit 1
  fi
fi

cat > "${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
xargs -d '\n' -a '${COMMANDS}' -I{} -P ${CONCURRENCY} bash -lc "{}"
EOF
chmod +x "${RUNNER_PATH}"

setsid "${RUNNER_PATH}" </dev/null > "${LOG_PATH}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "${PID_PATH}"
echo "launched sweep"
echo "  pid: ${SWEEP_PID}"
echo "  log: ${LOG_PATH}"
