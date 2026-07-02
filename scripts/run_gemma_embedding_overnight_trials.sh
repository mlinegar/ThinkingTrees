#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/gemma_embedding_overnight_${TIMESTAMP}}"

TASK_PORT="${TASK_PORT:-8005}"
EMBEDDING_URL="${EMBEDDING_URL:-http://localhost:8006/v1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-google/embeddinggemma-300m}"

LLM_DOC_IDS="${LLM_DOC_IDS:-51620_198306 51320_198306}"
EMBED_DOC_IDS="${EMBED_DOC_IDS:-51320_198306 51620_198306 51320_199705}"
LLM_CHUNK_SIZES="${LLM_CHUNK_SIZES:-3000 4000 5000}"
EMBED_WINDOW_CHARS="${EMBED_WINDOW_CHARS:-2000 4000 6000}"

OPTIMIZE_EXAMPLE="${OPTIMIZE_EXAMPLE:-true}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-gepa}"
OPTIMIZER_BUDGET="${OPTIMIZER_BUDGET:-light}"
OPTIMIZER_CHUNK_SIZE="${OPTIMIZER_CHUNK_SIZE:-4000}"
OPTIMIZER_TRAIN_SAMPLES="${OPTIMIZER_TRAIN_SAMPLES:-60}"
OPTIMIZER_VAL_SAMPLES="${OPTIMIZER_VAL_SAMPLES:-18}"
OPTIMIZER_IDS="${OPTIMIZER_IDS:-51620_198306 51320_198306}"

mkdir -p "${OUTPUT_ROOT}"

read -r -a LLM_DOC_IDS_ARR <<< "${LLM_DOC_IDS}"
read -r -a EMBED_DOC_IDS_ARR <<< "${EMBED_DOC_IDS}"
read -r -a LLM_CHUNK_SIZES_ARR <<< "${LLM_CHUNK_SIZES}"
read -r -a EMBED_WINDOW_CHARS_ARR <<< "${EMBED_WINDOW_CHARS}"
read -r -a OPTIMIZER_IDS_ARR <<< "${OPTIMIZER_IDS}"

STATUS_TSV="${OUTPUT_ROOT}/step_status.tsv"
SUMMARY_JSON="${OUTPUT_ROOT}/summary.json"
RUN_LOG="${OUTPUT_ROOT}/overnight.log"

echo -e "step\tstatus\toutput" > "${STATUS_TSV}"

failures=0

log() {
  printf '%s %s\n' "[$(date --iso-8601=seconds)]" "$*" | tee -a "${RUN_LOG}"
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

run_step() {
  local step="$1"
  local output="$2"
  shift 2

  log "start: ${step}"
  log "cmd: $*"
  if "$@" >> "${RUN_LOG}" 2>&1; then
    log "done: ${step}"
    mark_status "${step}" "ok" "${output}"
  else
    local exit_code=$?
    log "FAIL(${exit_code}): ${step}"
    mark_status "${step}" "failed:${exit_code}" "${output}"
    failures=$((failures + 1))
  fi
}

require_ready "Gemma text" "http://localhost:${TASK_PORT}/v1/models" || exit 1
require_ready "EmbeddingGemma" "${EMBEDDING_URL}/models" || exit 1

log "output_root=${OUTPUT_ROOT}"
log "llm_doc_ids=${LLM_DOC_IDS}"
log "embed_doc_ids=${EMBED_DOC_IDS}"

for chunk_size in "${LLM_CHUNK_SIZES_ARR[@]}"; do
  run_step \
    "llm_baseline_chunk_${chunk_size}" \
    "${OUTPUT_ROOT}/llm_baseline_chunk_${chunk_size}.json" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_manifesto_batched_example.py" \
      --ids "${LLM_DOC_IDS_ARR[@]}" \
      --chunk-size "${chunk_size}" \
      --port "${TASK_PORT}" \
      --concurrent-docs 1 \
      --concurrent-requests 4 \
      --batch-size 4 \
      --temperature 0.0 \
      --max-tokens 128 \
      --no-baseline \
      --no-use-published-modules \
      --output "${OUTPUT_ROOT}/llm_baseline_chunk_${chunk_size}.json"
done

if [[ -f "${PROJECT_ROOT}/outputs/latest/manifesto_rile/trained_modules/unified_g_final.json" ]]; then
  for chunk_size in "${LLM_CHUNK_SIZES_ARR[@]}"; do
    run_step \
      "llm_published_chunk_${chunk_size}" \
      "${OUTPUT_ROOT}/llm_published_chunk_${chunk_size}.json" \
      "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_manifesto_batched_example.py" \
        --ids "${LLM_DOC_IDS_ARR[@]}" \
        --chunk-size "${chunk_size}" \
        --port "${TASK_PORT}" \
        --concurrent-docs 1 \
        --concurrent-requests 4 \
        --batch-size 4 \
        --temperature 0.0 \
        --max-tokens 128 \
        --no-baseline \
        --use-published-modules \
        --output "${OUTPUT_ROOT}/llm_published_chunk_${chunk_size}.json"
  done
else
  log "skip: published module sweep (published modules not found)"
fi

for window_chars in "${EMBED_WINDOW_CHARS_ARR[@]}"; do
  run_step \
    "embedding_window_${window_chars}" \
    "${OUTPUT_ROOT}/embedding_window_${window_chars}.json" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_multilang_embedding_smoke.py" \
      --ids "${EMBED_DOC_IDS_ARR[@]}" \
      --embedding-url "${EMBEDDING_URL}" \
      --embedding-model "${EMBEDDING_MODEL}" \
      --window-chars "${window_chars}" \
      --max-windows 6 \
      --json-out "${OUTPUT_ROOT}/embedding_window_${window_chars}.json"
done

if [[ "${OPTIMIZE_EXAMPLE}" == "true" ]]; then
  run_step \
    "optimized_example_${OPTIMIZER_TYPE}_chunk_${OPTIMIZER_CHUNK_SIZE}" \
    "${OUTPUT_ROOT}/optimized_example_${OPTIMIZER_TYPE}_chunk_${OPTIMIZER_CHUNK_SIZE}" \
    env \
      PORT="${TASK_PORT}" \
      PUBLISH_LATEST=false \
      "${PROJECT_ROOT}/scripts/run_manifesto_optimized_example.sh" \
        --no-start-server \
        --port "${TASK_PORT}" \
        --chunk-size "${OPTIMIZER_CHUNK_SIZE}" \
        --train-samples "${OPTIMIZER_TRAIN_SAMPLES}" \
        --val-samples "${OPTIMIZER_VAL_SAMPLES}" \
        --optimizer "${OPTIMIZER_TYPE}" \
        --optimizer-budget "${OPTIMIZER_BUDGET}" \
        --num-threads 8 \
        --phase1-max-tokens-summary 128 \
        --phase1-max-tokens-score 64 \
        --ids "${OPTIMIZER_IDS_ARR[@]}" \
        --output-dir "${OUTPUT_ROOT}/optimized_example_${OPTIMIZER_TYPE}_chunk_${OPTIMIZER_CHUNK_SIZE}"
fi

"${PYTHON_BIN}" - <<'PY' "${OUTPUT_ROOT}" "${SUMMARY_JSON}" >> "${RUN_LOG}" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

payload: dict[str, object] = {
    "output_root": str(output_root),
    "llm_runs": [],
    "embedding_runs": [],
}

for path in sorted(output_root.glob("llm_*.json")):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    results = []
    for row in data.get("results", []):
        if not isinstance(row, dict):
            continue
        results.append(
            {
                "manifesto_id": row.get("manifesto_id"),
                "expert_rile": row.get("expert_rile"),
                "predicted_rile": row.get("predicted_rile"),
                "absolute_gap_rile": row.get("absolute_gap_rile"),
                "tree_leaves": row.get("tree_leaves"),
                "error": row.get("error"),
            }
        )
    payload["llm_runs"].append(
        {
            "file": path.name,
            "chunk_analysis": data.get("chunk_analysis", []),
            "results": results,
        }
    )

for path in sorted(output_root.glob("embedding_*.json")):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    payload["embedding_runs"].append(
        {
            "file": path.name,
            "ids": data.get("ids", []),
            "pairwise": data.get("pairwise", []),
        }
    )

summary_path.write_text(json.dumps(payload, indent=2))
print(json.dumps({"summary": str(summary_path)}, indent=2))
PY

log "summary=${SUMMARY_JSON}"
log "failures=${failures}"
exit 0
