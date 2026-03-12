#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="outputs/tree_relevant_lda_production_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ROOT_DEFAULT}"
STAGE1_CONCURRENCY=2
STAGE2_CONCURRENCY=2
TRAIN_DOCS=512
TEST_DOCS_STAGE1=512
TEST_DOCS_STAGE2=512
LAUNCH=1

usage() {
  cat <<'EOF'
Usage: scripts/launch_tree_relevant_lda_sweeps.sh [options]

Options:
  --output-root PATH         Output root for manifests, logs, and results.
  --stage1-concurrency N     xargs worker count for Stage 1. Default: 2.
  --stage2-concurrency N     xargs worker count for Stage 2. Default: 2.
  --train-docs N             Training docs per run. Default: 512.
  --test-docs-stage1 N       Held-out docs per Stage-1 run. Default: 512.
  --test-docs-stage2 N       Held-out docs per Stage-2 run. Default: 512.
  --no-launch                Only build manifests and runner scripts; do not start jobs.
  -h, --help                 Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --stage1-concurrency)
      STAGE1_CONCURRENCY="$2"
      shift 2
      ;;
    --stage2-concurrency)
      STAGE2_CONCURRENCY="$2"
      shift 2
      ;;
    --train-docs)
      TRAIN_DOCS="$2"
      shift 2
      ;;
    --test-docs-stage1)
      TEST_DOCS_STAGE1="$2"
      shift 2
      ;;
    --test-docs-stage2)
      TEST_DOCS_STAGE2="$2"
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

STAGE1_ROOT="${OUTPUT_ROOT}/stage1"
STAGE2_ROOT="${OUTPUT_ROOT}/stage2"
mkdir -p "${STAGE1_ROOT}" "${STAGE2_ROOT}"

STAGE1_COMMANDS="${STAGE1_ROOT}/commands.txt"
STAGE2_COMMANDS="${STAGE2_ROOT}/commands.txt"

STAGE1_LOG="${STAGE1_ROOT}/sweep.log"
STAGE2_LOG="${STAGE2_ROOT}/sweep.log"

STAGE1_PID="${STAGE1_ROOT}/sweep.pid"
STAGE2_PID="${STAGE2_ROOT}/sweep.pid"

STAGE1_RUNNER="${STAGE1_ROOT}/run_sweep.sh"
STAGE2_RUNNER="${STAGE2_ROOT}/run_sweep.sh"

SPEC_PATH="${OUTPUT_ROOT}/sweep_spec.txt"

echo "building Stage-1 commands"
venv/bin/python scripts/build_lda_tree_utility_vector_cmds.py \
  --out-cmds "${STAGE1_COMMANDS}" \
  --output-root "${STAGE1_ROOT}/results" \
  --leaf-fractions "1 1/2 1/4 1/24" \
  --doc-topic-concentrations "0.2 0.6 1.5" \
  --state-dims "4 8 16 32 64 128 256 512" \
  --seeds "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  --doc-tokens 384 \
  --utility-dim 16 \
  --train-docs "${TRAIN_DOCS}" \
  --test-docs "${TEST_DOCS_STAGE1}" \
  --n-topics 8 \
  --vocab-size 512 \
  --no-run-full-doc-mlp-diag \
  --skip-existing

echo "building Stage-2 commands"
venv/bin/python scripts/build_leaf_local_mixture_utility_cmds.py \
  --out-cmds "${STAGE2_COMMANDS}" \
  --output-root "${STAGE2_ROOT}/results" \
  --leaf-fractions "1 1/2 1/4 1/24" \
  --doc-topic-concentrations "0.6" \
  --local-mixture-concentrations "64 8 1 0.25" \
  --lambda-grid "0 1 2" \
  --budget-regimes "all_leaves_labeled fixed_oracle_budget" \
  --leaf-label-budgets "2 4 8 16 24" \
  --seeds "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15" \
  --doc-tokens 384 \
  --train-docs "${TRAIN_DOCS}" \
  --test-docs "${TEST_DOCS_STAGE2}" \
  --skip-existing

STAGE1_COUNT="$(wc -l < "${STAGE1_COMMANDS}" | tr -d ' ')"
STAGE2_COUNT="$(wc -l < "${STAGE2_COMMANDS}" | tr -d ' ')"

cat > "${STAGE1_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
ionice -c3 nice -n 15 xargs -d '\n' -a '${STAGE1_COMMANDS}' -I{} -P ${STAGE1_CONCURRENCY} bash -lc "{}"
EOF
chmod +x "${STAGE1_RUNNER}"

cat > "${STAGE2_RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
cd /home/mlinegar/ThinkingTrees
ionice -c3 nice -n 15 xargs -d '\n' -a '${STAGE2_COMMANDS}' -I{} -P ${STAGE2_CONCURRENCY} bash -lc "{}"
EOF
chmod +x "${STAGE2_RUNNER}"

cat > "${SPEC_PATH}" <<EOF
generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
output_root=${OUTPUT_ROOT}
stage1_root=${STAGE1_ROOT}
stage2_root=${STAGE2_ROOT}
stage1_commands=${STAGE1_COMMANDS}
stage2_commands=${STAGE2_COMMANDS}
stage1_log=${STAGE1_LOG}
stage2_log=${STAGE2_LOG}
stage1_pid=${STAGE1_PID}
stage2_pid=${STAGE2_PID}
stage1_runner=${STAGE1_RUNNER}
stage2_runner=${STAGE2_RUNNER}
stage1_concurrency=${STAGE1_CONCURRENCY}
stage2_concurrency=${STAGE2_CONCURRENCY}
train_docs=${TRAIN_DOCS}
test_docs_stage1=${TEST_DOCS_STAGE1}
test_docs_stage2=${TEST_DOCS_STAGE2}
stage1_leaf_fractions=1 1/2 1/4 1/24
stage1_dtc=0.2 0.6 1.5
stage1_state_dims=4 8 16 32 64 128 256 512
stage1_seeds=16
stage1_total_commands=${STAGE1_COUNT}
stage1_full_doc_mlp_diag=false
stage2_leaf_fractions=1 1/2 1/4 1/24
stage2_dtc=0.6
stage2_tau=64 8 1 0.25
stage2_lambda=0 1 2
stage2_budget_regimes=all_leaves_labeled fixed_oracle_budget
stage2_leaf_label_budgets=2 4 8 16 24
stage2_seeds=16
stage2_total_commands=${STAGE2_COUNT}
EOF

echo "prepared tree-relevant LDA sweeps"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  stage1_commands: ${STAGE1_COUNT}"
echo "  stage2_commands: ${STAGE2_COUNT}"
echo "  spec: ${SPEC_PATH}"

if [[ "${LAUNCH}" -eq 0 ]]; then
  exit 0
fi

check_existing() {
  local pid_path="$1"
  if [[ -f "${pid_path}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_path}" || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "existing sweep appears to be running with pid ${old_pid}; refusing to relaunch ${pid_path}" >&2
      exit 1
    fi
  fi
}

check_existing "${STAGE1_PID}"
check_existing "${STAGE2_PID}"

setsid "${STAGE1_RUNNER}" </dev/null > "${STAGE1_LOG}" 2>&1 &
STAGE1_SWEEP_PID=$!
echo "${STAGE1_SWEEP_PID}" > "${STAGE1_PID}"

setsid "${STAGE2_RUNNER}" </dev/null > "${STAGE2_LOG}" 2>&1 &
STAGE2_SWEEP_PID=$!
echo "${STAGE2_SWEEP_PID}" > "${STAGE2_PID}"

echo "launched tree-relevant LDA sweeps"
echo "  stage1_pid: ${STAGE1_SWEEP_PID}"
echo "  stage1_log: ${STAGE1_LOG}"
echo "  stage2_pid: ${STAGE2_SWEEP_PID}"
echo "  stage2_log: ${STAGE2_LOG}"
