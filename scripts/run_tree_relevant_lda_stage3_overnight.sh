#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="outputs/tree_relevant_lda_stage3_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ROOT_DEFAULT}"
CONCURRENCY=16
TRAIN_DOCS=512
TEST_DOCS=512
LATENT_LEAF_TOKENS=64
LAUNCH=1

usage() {
  cat <<'EOF'
Usage: scripts/run_tree_relevant_lda_stage3_overnight.sh [options]

Options:
  --output-root PATH         Output root for manifests, logs, and results.
  --concurrency N            Worker count. Default: 16.
  --train-docs N             Training docs per run. Default: 512.
  --test-docs N              Held-out docs per run. Default: 512.
  --latent-leaf-tokens N     Nominal latent section size. Default: 64.
  --no-launch                Only build manifests/runner; do not start the queue.
  -h, --help                 Show this help text.
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
    --train-docs)
      TRAIN_DOCS="$2"
      shift 2
      ;;
    --test-docs)
      TEST_DOCS="$2"
      shift 2
      ;;
    --latent-leaf-tokens)
      LATENT_LEAF_TOKENS="$2"
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

RESULTS_ROOT="${OUTPUT_ROOT}/results"
COMMANDS="${OUTPUT_ROOT}/commands.txt"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
MATRIX_MD="${OUTPUT_ROOT}/stage3_matrix.md"
SPEC_PATH="${OUTPUT_ROOT}/stage3_spec.txt"
RUNNER="${OUTPUT_ROOT}/run_queue.sh"
LOG_PATH="${OUTPUT_ROOT}/stage3.log"
PID_PATH="${OUTPUT_ROOT}/stage3.pid"

echo "building tree-relevant LDA Stage 3 commands"
venv/bin/python scripts/build_tree_relevant_lda_stage3_cmds.py \
  --output-root "${RESULTS_ROOT}" \
  --cmd-file "${COMMANDS}" \
  --manifest "${MANIFEST}" \
  --matrix-md "${MATRIX_MD}" \
  --train-docs "${TRAIN_DOCS}" \
  --test-docs "${TEST_DOCS}" \
  --latent-leaf-tokens "${LATENT_LEAF_TOKENS}"

COMMAND_COUNT="$(wc -l < "${COMMANDS}" | tr -d ' ')"

cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
ionice -c3 nice -n 15 venv/bin/python scripts/run_command_queue.py \
  --cmd-file '${COMMANDS}' \
  --log-dir '${OUTPUT_ROOT}/logs' \
  --workers ${CONCURRENCY}
EOF
chmod +x "${RUNNER}"

cat > "${SPEC_PATH}" <<EOF
generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
output_root=${OUTPUT_ROOT}
results_root=${RESULTS_ROOT}
commands=${COMMANDS}
manifest=${MANIFEST}
matrix_md=${MATRIX_MD}
runner=${RUNNER}
log=${LOG_PATH}
pid=${PID_PATH}
concurrency=${CONCURRENCY}
train_docs=${TRAIN_DOCS}
test_docs=${TEST_DOCS}
latent_leaf_tokens=${LATENT_LEAF_TOKENS}
total_commands=${COMMAND_COUNT}
EOF

echo "prepared tree-relevant LDA Stage 3 queue"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  commands: ${COMMAND_COUNT}"
echo "  spec: ${SPEC_PATH}"

if [[ "${LAUNCH}" -eq 0 ]]; then
  exit 0
fi

if [[ -f "${PID_PATH}" ]]; then
  OLD_PID="$(cat "${PID_PATH}" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "existing Stage 3 queue appears to be running with pid ${OLD_PID}; refusing to relaunch" >&2
    exit 1
  fi
fi

DETACHED_PID="$(venv/bin/python scripts/spawn_detached_cmd.py \
  --pid-file "${PID_PATH}" \
  --cwd /home/mlinegar/ThinkingTrees \
  --stdin /dev/null \
  --stdout "${LOG_PATH}" \
  --stderr "${LOG_PATH}" \
  -- bash "${RUNNER}")"

echo "launched tree-relevant LDA Stage 3 queue"
echo "  pid: ${DETACHED_PID}"
echo "  log: ${LOG_PATH}"
