#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source venv/bin/activate

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/tree_neural_parallel_overnight_${STAMP}}"
LOG_ROOT="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

mapfile -t MIG_UUIDS < <(python - <<'PY'
from scripts import run_tree_neural_full_doc_mig as mig
for uuid in mig._discover_mig_uuids():
    print(uuid)
PY
)

if (( ${#MIG_UUIDS[@]} < 8 )); then
  echo "Need at least 8 MIG UUIDs; found ${#MIG_UUIDS[@]}" >&2
  exit 1
fi

TRACK_A_UUIDS=("${MIG_UUIDS[@]:0:4}")
TRACK_B_UUIDS=("${MIG_UUIDS[@]:4}")

TRACK_A_CSV="$(IFS=,; echo "${TRACK_A_UUIDS[*]}")"
TRACK_B_CSV="$(IFS=,; echo "${TRACK_B_UUIDS[*]}")"

TRACK_A_ROOT="${OUT_ROOT}/track_a_slotwise_scaling"
TRACK_B_ROOT="${OUT_ROOT}/track_b_shared_feature_search"

TRACK_A_LOG="${LOG_ROOT}/track_a_slotwise_scaling.log"
TRACK_B_LOG="${LOG_ROOT}/track_b_shared_feature_search.log"

nohup bash -lc "
  set -euo pipefail
  cd '${REPO_ROOT}'
  source venv/bin/activate
  exec python3 -u scripts/run_tree_neural_slotwise_scaling_push.py \
    --output-root '${TRACK_A_ROOT}' \
    --mig-uuids '${TRACK_A_CSV}'
" >"${TRACK_A_LOG}" 2>&1 &
TRACK_A_PID=$!

nohup bash -lc "
  set -euo pipefail
  cd '${REPO_ROOT}'
  source venv/bin/activate
  exec python3 -u scripts/run_tree_neural_learned_surface_push.py \
    --output-root '${TRACK_B_ROOT}' \
    --mig-uuids '${TRACK_B_CSV}'
" >"${TRACK_B_LOG}" 2>&1 &
TRACK_B_PID=$!

cat >"${OUT_ROOT}/parallel_overnight_env.sh" <<EOF
export TREE_NEURAL_PARALLEL_OVERNIGHT_ROOT="${OUT_ROOT}"
export TREE_NEURAL_TRACK_A_ROOT="${TRACK_A_ROOT}"
export TREE_NEURAL_TRACK_B_ROOT="${TRACK_B_ROOT}"
export TREE_NEURAL_TRACK_A_PID="${TRACK_A_PID}"
export TREE_NEURAL_TRACK_B_PID="${TRACK_B_PID}"
export TREE_NEURAL_TRACK_A_LOG="${TRACK_A_LOG}"
export TREE_NEURAL_TRACK_B_LOG="${TRACK_B_LOG}"
EOF

printf '%s\n' \
  "parallel overnight launched" \
  "root: ${OUT_ROOT}" \
  "track_a_root: ${TRACK_A_ROOT}" \
  "track_b_root: ${TRACK_B_ROOT}" \
  "track_a_pid: ${TRACK_A_PID}" \
  "track_b_pid: ${TRACK_B_PID}" \
  "track_a_log: ${TRACK_A_LOG}" \
  "track_b_log: ${TRACK_B_LOG}"
