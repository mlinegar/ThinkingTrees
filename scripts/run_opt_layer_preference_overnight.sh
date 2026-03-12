#!/usr/bin/env bash
set -euo pipefail

# Adaptive overnight opt-layer preference recovery run.
# Defaults to the "second half" of 128 CPUs: cores 64-127.
#
# Usage:
#   ./scripts/run_opt_layer_preference_overnight.sh
#
# Optional env overrides:
#   JOBS=64 CPU_SET=64-127 DURATION_HOURS=10 TARGET_PREF_ACC=0.90 \
#   MAX_ROUNDS=0 DETACH=1 EXTRA_ARGS="--initial-trials-per-setting 12" \
#   ./scripts/run_opt_layer_preference_overnight.sh

JOBS="${JOBS:-64}"
CPU_SET="${CPU_SET:-64-127}"
DURATION_HOURS="${DURATION_HOURS:-10}"
TARGET_PREF_ACC="${TARGET_PREF_ACC:-0.90}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
SEED="${SEED:-0}"
DETACH="${DETACH:-1}"
ENABLE_TORCH="${ENABLE_TORCH:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

OUT_ROOT="${OUT_ROOT:-outputs/opt_layer_preference_overnight_${STAMP}}"
LOG_PATH="${LOG_PATH:-logs/${STAMP}_opt_layer_preference_overnight.log}"
PID_PATH="${PID_PATH:-logs/${STAMP}_opt_layer_preference_overnight.pid}"
LATEST_ENV="${LATEST_ENV:-logs/opt_layer_preference_overnight_latest.env}"

mkdir -p "$(dirname "${LOG_PATH}")" "${OUT_ROOT}"

CMD=(
  taskset -c "${CPU_SET}"
  ./venv/bin/python -u scripts/run_opt_layer_preference_overnight.py
  --output-dir "${OUT_ROOT}"
  --duration-hours "${DURATION_HOURS}"
  --jobs "${JOBS}"
  --cpu-set "${CPU_SET}"
  --target-pref-accuracy "${TARGET_PREF_ACC}"
  --max-rounds "${MAX_ROUNDS}"
  --seed "${SEED}"
)

if [[ "${ENABLE_TORCH}" == "1" ]]; then
  CMD+=(--enable-torch)
else
  CMD+=(--no-enable-torch)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

if [[ "${DETACH}" == "1" ]]; then
  PID="$(
    ./venv/bin/python scripts/spawn_detached_cmd.py \
      --pid-file "${PID_PATH}" \
      --cwd "$(pwd)" \
      --stdout "${LOG_PATH}" \
      --stderr "${LOG_PATH}" \
      -- "${CMD[@]}"
  )"
  echo "started_detached pid=${PID}"
else
  echo "running_foreground log=${LOG_PATH}"
  "${CMD[@]}" | tee -a "${LOG_PATH}"
  PID="foreground"
fi

cat > "${LATEST_ENV}" <<EOF
RUN_STAMP=${STAMP}
RUN_PID=${PID}
RUN_PID_FILE=${PID_PATH}
RUN_LOG=${LOG_PATH}
RUN_OUTPUT_DIR=${OUT_ROOT}
EOF

echo "pid_file=${PID_PATH}"
echo "log=${LOG_PATH}"
echo "output_dir=${OUT_ROOT}"
echo "latest_env=${LATEST_ENV}"
echo "monitor_cmd=tail -f ${LOG_PATH}"
