#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="outputs/markov_local_law_learnability_20260307_015157"
ROOT="${MARKOV_BACKFILL_ROOT:-$ROOT_DEFAULT}"
REQUIRE_FIELD="${MARKOV_BACKFILL_REQUIRE_FIELD:-test_objective_full_labels}"
INPUT_ROOT="${ROOT}"
LOCAL_LAW_ROOT="${ROOT}/markov_changepoint_ops_count/local_law_learnability"
REPORT_OUT="${MARKOV_REPORT_OUTPUT_DIR:-${LOCAL_LAW_ROOT}/local_law_report}"
EXPECTED_RUNS="${MARKOV_EXPECTED_RUN_COUNT:-1792}"
APPEND_WAIT_PID="${MARKOV_APPEND_WAIT_PID:-}"
APPEND_WORKERS="${MARKOV_APPEND_WORKERS:-16}"
FIRST_PASS_MIN_AGE_SECONDS="${MARKOV_BACKFILL_MIN_AGE_SECONDS:-600}"
STATUS_NOTE_DEFAULT="Legacy rows missing exact held-out objective fields were replayed from saved configs/seeds on CPU; report uses exact test objective whenever present."
STATUS_NOTE="${MARKOV_STATUS_NOTE:-$STATUS_NOTE_DEFAULT}"

TOTAL_CPUS="$(nproc)"
if [[ -n "${MARKOV_BACKFILL_WORKERS:-}" ]]; then
  FIRST_PASS_WORKERS="${MARKOV_BACKFILL_WORKERS}"
else
  if [[ -n "${APPEND_WAIT_PID}" ]] && ps -p "${APPEND_WAIT_PID}" >/dev/null 2>&1; then
    FIRST_PASS_WORKERS="$(( TOTAL_CPUS - APPEND_WORKERS ))"
  else
    FIRST_PASS_WORKERS="${TOTAL_CPUS}"
  fi
fi
if (( FIRST_PASS_WORKERS < 1 )); then
  FIRST_PASS_WORKERS=1
fi
SECOND_PASS_WORKERS="${MARKOV_BACKFILL_FINAL_WORKERS:-${TOTAL_CPUS}}"
if (( SECOND_PASS_WORKERS < 1 )); then
  SECOND_PASS_WORKERS=1
fi

BACKFILL_LOG_ROOT="${ROOT}/backfill_logs"
mkdir -p "${BACKFILL_LOG_ROOT}"

source venv/bin/activate

python scripts/backfill_markov_local_law_objectives.py \
  --input-root "${INPUT_ROOT}" \
  --require-field "${REQUIRE_FIELD}" \
  --device cpu \
  --torch-threads 1 \
  --workers "${FIRST_PASS_WORKERS}" \
  --min-age-seconds "${FIRST_PASS_MIN_AGE_SECONDS}" \
  --log-dir "${BACKFILL_LOG_ROOT}/phase1_logs" \
  --manifest-path "${BACKFILL_LOG_ROOT}/phase1_manifest.json"

if [[ -n "${APPEND_WAIT_PID}" ]]; then
  while ps -p "${APPEND_WAIT_PID}" >/dev/null 2>&1; do
    sleep 60
  done
fi

python scripts/backfill_markov_local_law_objectives.py \
  --input-root "${INPUT_ROOT}" \
  --require-field "${REQUIRE_FIELD}" \
  --device cpu \
  --torch-threads 1 \
  --workers "${SECOND_PASS_WORKERS}" \
  --min-age-seconds 0 \
  --log-dir "${BACKFILL_LOG_ROOT}/phase2_logs" \
  --manifest-path "${BACKFILL_LOG_ROOT}/phase2_manifest.json"

python scripts/report_markov_local_law_learnability.py \
  --input-root "${LOCAL_LAW_ROOT}" \
  --output-dir "${REPORT_OUT}" \
  --expected-run-count "${EXPECTED_RUNS}" \
  --status-note "${STATUS_NOTE}"
