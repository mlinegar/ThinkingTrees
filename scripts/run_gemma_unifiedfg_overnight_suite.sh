#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/gemma_unifiedfg_overnight_${TIMESTAMP}}"

TASK_PORT="${TASK_PORT:-8005}"
EMBEDDING_URL="${EMBEDDING_URL:-http://localhost:8006/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-google/embeddinggemma-300m}"

SNAPSHOT_INTERVAL_SEC="${SNAPSHOT_INTERVAL_SEC:-1800}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"

COMMON_CONCURRENT_DOCS="${COMMON_CONCURRENT_DOCS:-2}"
COMMON_CONCURRENT_REQUESTS="${COMMON_CONCURRENT_REQUESTS:-8}"
COMMON_NUM_THREADS="${COMMON_NUM_THREADS:-8}"
COMMON_SCORER_MAX_TOKENS="${COMMON_SCORER_MAX_TOKENS:-64}"
PIPELINE_PHASE1_MAX_TOKENS_SUMMARY="${PIPELINE_PHASE1_MAX_TOKENS_SUMMARY:-128}"
PIPELINE_PHASE1_MAX_TOKENS_SCORE="${PIPELINE_PHASE1_MAX_TOKENS_SCORE:-64}"

OPTIMIZED_IDS="${OPTIMIZED_IDS:-51620_198306 51320_198306 51320_199705}"
OPTIMIZED_CHUNK_SIZE="${OPTIMIZED_CHUNK_SIZE:-4000}"
OPTIMIZED_CHUNK_TOKENS="${OPTIMIZED_CHUNK_TOKENS:-}"
OPTIMIZED_TRAIN_SAMPLES="${OPTIMIZED_TRAIN_SAMPLES:-24}"
OPTIMIZED_VAL_SAMPLES="${OPTIMIZED_VAL_SAMPLES:-8}"
OPTIMIZED_ITERATIONS="${OPTIMIZED_ITERATIONS:-2}"
OPTIMIZED_BUDGET="${OPTIMIZED_BUDGET:-light}"

PIPELINE_TRAIN_SAMPLES="${PIPELINE_TRAIN_SAMPLES:-36}"
PIPELINE_VAL_SAMPLES="${PIPELINE_VAL_SAMPLES:-12}"
PIPELINE_TEST_SAMPLES="${PIPELINE_TEST_SAMPLES:-12}"
PIPELINE_ITERATIONS="${PIPELINE_ITERATIONS:-2}"
PIPELINE_CHUNK_SIZE="${PIPELINE_CHUNK_SIZE:-5000}"
PIPELINE_CHUNK_TOKENS="${PIPELINE_CHUNK_TOKENS:-}"
PIPELINE_BUDGET="${PIPELINE_BUDGET:-light}"

HYBRID_HEAD_METHOD="${HYBRID_HEAD_METHOD:-linear_sgd}"
HYBRID_HEAD_EPOCHS="${HYBRID_HEAD_EPOCHS:-20}"
HYBRID_HEAD_LR="${HYBRID_HEAD_LR:-0.003}"
HYBRID_HEAD_WEIGHT_DECAY="${HYBRID_HEAD_WEIGHT_DECAY:-0.0001}"
HYBRID_RETRAIN_ROUNDS="${HYBRID_RETRAIN_ROUNDS:-1}"

EMBED_DOC_IDS="${EMBED_DOC_IDS:-51320_198306 51620_198306 51320_199705}"
EMBED_WINDOW_CHARS="${EMBED_WINDOW_CHARS:-2000 4000 6000}"
RUN_OPTIMIZED_EXAMPLE="${RUN_OPTIMIZED_EXAMPLE:-true}"
RUN_PIPELINE_FIXED="${RUN_PIPELINE_FIXED:-true}"
RUN_PIPELINE_EMBEDPROXY="${RUN_PIPELINE_EMBEDPROXY:-true}"
RUN_EMBEDDING_SWEEP="${RUN_EMBEDDING_SWEEP:-true}"

mkdir -p "${OUTPUT_ROOT}"

read -r -a OPTIMIZED_IDS_ARR <<< "${OPTIMIZED_IDS}"
read -r -a EMBED_DOC_IDS_ARR <<< "${EMBED_DOC_IDS}"
read -r -a EMBED_WINDOW_CHARS_ARR <<< "${EMBED_WINDOW_CHARS}"

STATUS_TSV="${OUTPUT_ROOT}/step_status.tsv"
SNAPSHOT_TSV="${OUTPUT_ROOT}/snapshot_status.tsv"
RUN_LOG="${OUTPUT_ROOT}/overnight.log"
SUMMARY_JSON="${OUTPUT_ROOT}/summary.json"
LIVE_PROGRESS_JSON="${OUTPUT_ROOT}/live_progress.json"
MANIFEST_JSON="${OUTPUT_ROOT}/suite_manifest.json"

printf 'step\tstatus\toutput\n' > "${STATUS_TSV}"
printf 'step\tsnapshot_at\tsnapshot_dir\n' > "${SNAPSHOT_TSV}"

failures=0

log() {
  printf '%s %s\n' "[$(date --iso-8601=seconds)]" "$*" | tee -a "${RUN_LOG}"
}

refresh_live_progress() {
  if [[ -x "${PYTHON_BIN}" ]] && [[ -f "${PROJECT_ROOT}/scripts/summarize_unifiedfg_progress.py" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_unifiedfg_progress.py" \
      --output-root "${OUTPUT_ROOT}" \
      --json-out "${LIVE_PROGRESS_JSON}" >> "${RUN_LOG}" 2>&1 || true
  fi
}

mark_status() {
  local step="$1"
  local status="$2"
  local output="$3"
  printf '%s\t%s\t%s\n' "${step}" "${status}" "${output}" >> "${STATUS_TSV}"
}

require_ready() {
  local name="$1"
  local url="$2"
  if curl -fsS "${url}" > /dev/null; then
    log "ready: ${name} ${url}"
    return 0
  fi
  log "ERROR: ${name} not ready at ${url}"
  return 1
}

snapshot_output_dir() {
  local step="$1"
  local output_dir="$2"
  if [[ ! -d "${output_dir}" ]]; then
    return 0
  fi

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local snapshot_dir="${OUTPUT_ROOT}/snapshots/${step}/${stamp}"
  mkdir -p "${snapshot_dir}"

  local dir_name=""
  for dir_name in checkpoints trained_modules preferences exports generator neural_operators; do
    if [[ -d "${output_dir}/${dir_name}" ]]; then
      rsync -a "${output_dir}/${dir_name}/" "${snapshot_dir}/${dir_name}/" >> "${RUN_LOG}" 2>&1 || true
    fi
  done

  find "${output_dir}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.log' -o -name '*.md' -o -name '*.txt' -o -name '*.pdf' \) \
    -exec cp -a {} "${snapshot_dir}/" \; >> "${RUN_LOG}" 2>&1 || true

  printf '%s\t%s\t%s\n' "${step}" "${stamp}" "${snapshot_dir}" >> "${SNAPSHOT_TSV}"
  log "snapshot: ${step} -> ${snapshot_dir}"
}

run_step() {
  local step="$1"
  local output="$2"
  shift 2

  log "start: ${step}"
  log "cmd: $*"
  refresh_live_progress
  if "$@" >> "${RUN_LOG}" 2>&1; then
    log "done: ${step}"
    mark_status "${step}" "ok" "${output}"
  else
    local exit_code=$?
    log "FAIL(${exit_code}): ${step}"
    mark_status "${step}" "failed:${exit_code}" "${output}"
    failures=$((failures + 1))
  fi
  refresh_live_progress
}

run_step_with_snapshots() {
  local step="$1"
  local output_dir="$2"
  shift 2

  mkdir -p "${output_dir}"
  log "start: ${step}"
  log "cmd: $*"
  refresh_live_progress
  "$@" >> "${RUN_LOG}" 2>&1 &
  local cmd_pid=$!
  local next_snapshot
  next_snapshot=$(( $(date +%s) + SNAPSHOT_INTERVAL_SEC ))

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    sleep "${POLL_INTERVAL_SEC}"
    refresh_live_progress
    if [[ $(date +%s) -ge ${next_snapshot} ]]; then
      snapshot_output_dir "${step}" "${output_dir}"
      next_snapshot=$(( next_snapshot + SNAPSHOT_INTERVAL_SEC ))
    fi
  done

  wait "${cmd_pid}"
  local exit_code=$?
  snapshot_output_dir "${step}" "${output_dir}"
  if [[ ${exit_code} -eq 0 ]]; then
    log "done: ${step}"
    mark_status "${step}" "ok" "${output_dir}"
  else
    log "FAIL(${exit_code}): ${step}"
    mark_status "${step}" "failed:${exit_code}" "${output_dir}"
    failures=$((failures + 1))
  fi
  refresh_live_progress
}

cat > "${MANIFEST_JSON}" <<EOF
{
  "generated_at": "$(date --iso-8601=seconds)",
  "output_root": "${OUTPUT_ROOT}",
  "task_port": ${TASK_PORT},
  "embedding_url": "${EMBEDDING_URL}",
  "embedding_model": "${EMBEDDING_MODEL}",
  "snapshot_interval_sec": ${SNAPSHOT_INTERVAL_SEC},
  "poll_interval_sec": ${POLL_INTERVAL_SEC},
  "suite": "gemma_unifiedfg_overnight"
}
EOF

require_ready "Gemma text" "http://localhost:${TASK_PORT}/v1/models" || exit 1
require_ready "EmbeddingGemma" "${EMBEDDING_URL}/models" || exit 1

log "output_root=${OUTPUT_ROOT}"
log "snapshot_interval_sec=${SNAPSHOT_INTERVAL_SEC}"
log "optimized_ids=${OPTIMIZED_IDS}"
if [[ -n "${OPTIMIZED_CHUNK_TOKENS}" ]]; then
  log "optimized_chunk_tokens=${OPTIMIZED_CHUNK_TOKENS}"
fi
if [[ -n "${PIPELINE_CHUNK_TOKENS}" ]]; then
  log "pipeline_chunk_tokens=${PIPELINE_CHUNK_TOKENS}"
fi
refresh_live_progress

OPTIMIZED_STEP_SUFFIX="chunk_${OPTIMIZED_CHUNK_SIZE}"
if [[ -n "${OPTIMIZED_CHUNK_TOKENS}" ]]; then
  OPTIMIZED_STEP_SUFFIX="token_${OPTIMIZED_CHUNK_TOKENS}"
fi
PIPELINE_STEP_SUFFIX="chunk_${PIPELINE_CHUNK_SIZE}"
if [[ -n "${PIPELINE_CHUNK_TOKENS}" ]]; then
  PIPELINE_STEP_SUFFIX="token_${PIPELINE_CHUNK_TOKENS}"
fi

if [[ "${RUN_OPTIMIZED_EXAMPLE}" == "true" ]]; then
  OPTIMIZED_CMD=(
    env
      PORT="${TASK_PORT}"
      PUBLISH_LATEST=false
      "${PROJECT_ROOT}/scripts/run_manifesto_optimized_example.sh"
        --no-start-server
        --no-dynamic-gpu
        --port "${TASK_PORT}"
        --chunk-size "${OPTIMIZED_CHUNK_SIZE}"
        --train-samples "${OPTIMIZED_TRAIN_SAMPLES}"
        --val-samples "${OPTIMIZED_VAL_SAMPLES}"
        --optimizer gepa
        --optimizer-budget "${OPTIMIZED_BUDGET}"
        --num-threads "${COMMON_NUM_THREADS}"
        --concurrent-docs "${COMMON_CONCURRENT_DOCS}"
        --concurrent-requests "${COMMON_CONCURRENT_REQUESTS}"
        --phase1-max-tokens-summary 128
        --phase1-max-tokens-score 64
        --n-iterations "${OPTIMIZED_ITERATIONS}"
        --ids "${OPTIMIZED_IDS_ARR[@]}"
        --output-dir "${OUTPUT_ROOT}/optimized_example_gemma_${OPTIMIZED_STEP_SUFFIX}"
  )
  if [[ -n "${OPTIMIZED_CHUNK_TOKENS}" ]]; then
    OPTIMIZED_CMD+=(--chunk-tokens "${OPTIMIZED_CHUNK_TOKENS}")
  fi
  run_step_with_snapshots \
    "optimized_example_gemma_${OPTIMIZED_STEP_SUFFIX}" \
    "${OUTPUT_ROOT}/optimized_example_gemma_${OPTIMIZED_STEP_SUFFIX}" \
    "${OPTIMIZED_CMD[@]}"
fi

if [[ "${RUN_PIPELINE_FIXED}" == "true" ]]; then
  PIPELINE_FIXED_CMD=(
    bash "${PROJECT_ROOT}/scripts/run_training_pipeline.sh"
        --no-start-server
        --port "${TASK_PORT}"
        --task manifesto_rile
        --train-samples "${PIPELINE_TRAIN_SAMPLES}"
        --val-samples "${PIPELINE_VAL_SAMPLES}"
        --test-samples "${PIPELINE_TEST_SAMPLES}"
        --max-chunk-chars "${PIPELINE_CHUNK_SIZE}"
        --optimizer gepa
        --optimizer-budget "${PIPELINE_BUDGET}"
        --n-iterations "${PIPELINE_ITERATIONS}"
        --num-threads "${COMMON_NUM_THREADS}"
        --concurrent-docs "${COMMON_CONCURRENT_DOCS}"
        --concurrent-requests "${COMMON_CONCURRENT_REQUESTS}"
        --scorer-max-tokens "${COMMON_SCORER_MAX_TOKENS}"
        --no-dynamic-gpu
        --no-adaptive-chunking
        --no-honest-chunking
        --no-three-layer-honesty
        --no-phase1-score-requests
        --no-phase1-run-baseline
        --phase1-max-tokens-summary "${PIPELINE_PHASE1_MAX_TOKENS_SUMMARY}"
        --phase1-max-tokens-score "${PIPELINE_PHASE1_MAX_TOKENS_SCORE}"
        --no-train-neural-operators
        --no-train-generator
        --output-dir "${OUTPUT_ROOT}/training_pipeline_gemma_fixed_${PIPELINE_STEP_SUFFIX}"
  )
  if [[ -n "${PIPELINE_CHUNK_TOKENS}" ]]; then
    PIPELINE_FIXED_CMD+=(--max-chunk-tokens "${PIPELINE_CHUNK_TOKENS}")
  fi
  run_step_with_snapshots \
    "training_pipeline_gemma_fixed_${PIPELINE_STEP_SUFFIX}" \
    "${OUTPUT_ROOT}/training_pipeline_gemma_fixed_${PIPELINE_STEP_SUFFIX}" \
    "${PIPELINE_FIXED_CMD[@]}"
fi

if [[ "${RUN_PIPELINE_EMBEDPROXY}" == "true" ]]; then
  PIPELINE_EMBED_CMD=(
    bash "${PROJECT_ROOT}/scripts/run_training_pipeline.sh"
        --no-start-server
        --port "${TASK_PORT}"
        --task manifesto_rile
        --train-samples "${PIPELINE_TRAIN_SAMPLES}"
        --val-samples "${PIPELINE_VAL_SAMPLES}"
        --test-samples "${PIPELINE_TEST_SAMPLES}"
        --max-chunk-chars "${PIPELINE_CHUNK_SIZE}"
        --optimizer gepa
        --optimizer-budget "${PIPELINE_BUDGET}"
        --n-iterations "${PIPELINE_ITERATIONS}"
        --num-threads "${COMMON_NUM_THREADS}"
        --concurrent-docs "${COMMON_CONCURRENT_DOCS}"
        --concurrent-requests "${COMMON_CONCURRENT_REQUESTS}"
        --scorer-max-tokens "${COMMON_SCORER_MAX_TOKENS}"
        --no-dynamic-gpu
        --no-adaptive-chunking
        --no-honest-chunking
        --no-three-layer-honesty
        --no-phase1-score-requests
        --no-phase1-run-baseline
        --phase1-max-tokens-summary "${PIPELINE_PHASE1_MAX_TOKENS_SUMMARY}"
        --phase1-max-tokens-score "${PIPELINE_PHASE1_MAX_TOKENS_SCORE}"
        --adaptive-embedding-proxy
        --adaptive-embedding-api-base "${EMBEDDING_URL}"
        --adaptive-embedding-model "${EMBEDDING_MODEL}"
        --adaptive-embedding-head-method "${HYBRID_HEAD_METHOD}"
        --adaptive-embedding-head-epochs "${HYBRID_HEAD_EPOCHS}"
        --adaptive-embedding-head-lr "${HYBRID_HEAD_LR}"
        --adaptive-embedding-head-weight-decay "${HYBRID_HEAD_WEIGHT_DECAY}"
        --adaptive-embedding-retrain-rounds "${HYBRID_RETRAIN_ROUNDS}"
        --no-train-neural-operators
        --no-train-generator
        --output-dir "${OUTPUT_ROOT}/training_pipeline_gemma_embedproxy_${PIPELINE_STEP_SUFFIX}"
  )
  if [[ -n "${PIPELINE_CHUNK_TOKENS}" ]]; then
    PIPELINE_EMBED_CMD+=(--max-chunk-tokens "${PIPELINE_CHUNK_TOKENS}")
  fi
  run_step_with_snapshots \
    "training_pipeline_gemma_embedproxy_${PIPELINE_STEP_SUFFIX}" \
    "${OUTPUT_ROOT}/training_pipeline_gemma_embedproxy_${PIPELINE_STEP_SUFFIX}" \
    "${PIPELINE_EMBED_CMD[@]}"
fi

if [[ "${RUN_EMBEDDING_SWEEP}" == "true" ]]; then
  for window_chars in "${EMBED_WINDOW_CHARS_ARR[@]}"; do
    run_step \
      "embedding_smoke_window_${window_chars}" \
      "${OUTPUT_ROOT}/embedding_smoke_window_${window_chars}.json" \
      "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_multilang_embedding_smoke.py" \
        --ids "${EMBED_DOC_IDS_ARR[@]}" \
        --embedding-url "${EMBEDDING_URL}" \
        --embedding-model "${EMBEDDING_MODEL}" \
        --window-chars "${window_chars}" \
        --max-windows 6 \
        --json-out "${OUTPUT_ROOT}/embedding_smoke_window_${window_chars}.json"
  done
fi

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${STATUS_TSV}" "${SNAPSHOT_TSV}" "${SUMMARY_JSON}" <<'PY' >> "${RUN_LOG}" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
status_tsv = Path(sys.argv[2])
snapshot_tsv = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

steps: list[dict[str, object]] = []
for line in status_tsv.read_text(encoding="utf-8").splitlines()[1:]:
    if not line.strip():
        continue
    step, status, output = line.split("\t", 2)
    output_path = Path(output)
    final_stats_path = output_path / "final_stats.json"
    checkpoint_progress = output_path / "checkpoints" / "progress.json"
    steps.append(
        {
            "step": step,
            "status": status,
            "output": output,
            "has_final_stats": final_stats_path.exists(),
            "has_checkpoint_progress": checkpoint_progress.exists(),
            "has_trained_modules": (output_path / "trained_modules").exists(),
        }
    )

snapshot_rows: list[dict[str, str]] = []
for line in snapshot_tsv.read_text(encoding="utf-8").splitlines()[1:]:
    if not line.strip():
        continue
    step, snapshot_at, snapshot_dir = line.split("\t", 2)
    snapshot_rows.append(
        {
            "step": step,
            "snapshot_at": snapshot_at,
            "snapshot_dir": snapshot_dir,
        }
    )

payload = {
    "output_root": str(output_root),
    "steps": steps,
    "snapshots": snapshot_rows,
}
summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"summary": str(summary_path)}, indent=2))
PY

log "summary=${SUMMARY_JSON}"
log "failures=${failures}"
refresh_live_progress
exit 0
