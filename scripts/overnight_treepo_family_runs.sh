#!/usr/bin/env bash
# Overnight "basic test runs" for every major model via the canonical treepo path.
#
# Two phases, fail-safe (never aborts the whole run on one failure):
#   OFFLINE (always): cross-family x leaf-size matrix + treepo.methods suite
#                     (oracle / learnable_constant / fno / lda + all family
#                      offline contracts). No servers, no GPU servers needed.
#   LIVE (opt-in via TT_RUN_LIVE_TESTS=1; self-gating): treepo integration tests
#                     (dspy-live, llm-live, fno-live, probe-unified-no-live) +
#                     the TRL family smoke. Each test skips itself if its server /
#                     data / GPU dependency is absent, so missing deps => skip,
#                     not failure.
#
# Usage:
#   scripts/overnight_treepo_family_runs.sh [OUT_DIR]
# Env:
#   TT_RUN_LIVE_TESTS=1   enable the live phase (default: off)
#   VLLM_MODEL            served model id for live LLM/dspy tests
#                         (default google/gemma-4-31B-it)
#   VLLM_HOST/VLLM_PORT   live endpoint (default localhost:8000)

set -uo pipefail

TT_ROOT="/home/mlinegar/ThinkingTrees"
TREEPO_ROOT="/home/mlinegar/treepo"
TT_PY="${TT_ROOT}/venv/bin/python"
TREEPO_PY="${TREEPO_ROOT}/.venv/bin/python"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-${TT_ROOT}/outputs/overnight_treepo_families_${STAMP}}"
mkdir -p "${OUT}"
SUMMARY="${OUT}/SUMMARY.md"

# Capture the requested live flag, then FORCE the ambient env to offline so the
# offline phase never accidentally un-gates a TT_RUN_LIVE_TESTS-gated test (e.g.
# the TRL SFT subprocess under tests/methods/). The live phase sets
# TT_RUN_LIVE_TESTS=1 inline only for its own invocations.
RUN_LIVE="${TT_RUN_LIVE_TESTS:-0}"
export TT_RUN_LIVE_TESTS=0
: "${VLLM_MODEL:=google/gemma-4-31B-it}"
: "${VLLM_HOST:=localhost}"
: "${VLLM_PORT:=8000}"

echo "# Overnight treepo family runs — ${STAMP}" > "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "Output dir: ${OUT}" >> "${SUMMARY}"
echo "Live phase: requested=${RUN_LIVE} (model=${VLLM_MODEL} @ ${VLLM_HOST}:${VLLM_PORT})" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"
echo "| section | result | log |" >> "${SUMMARY}"
echo "|---|---|---|" >> "${SUMMARY}"

run_section () {
  # run_section <name> <logfile> <command...>
  local name="$1"; shift
  local log="$1"; shift
  echo "=== [${name}] $(date -u +%H:%M:%S) ===" | tee -a "${OUT}/run.log"
  "$@" > "${log}" 2>&1
  local rc=$?
  local tail_line
  tail_line="$(grep -E 'passed|failed|error|skipped|no tests ran' "${log}" | tail -1 | sed 's/|/ /g')"
  if [ "${rc}" -eq 0 ]; then
    echo "| ${name} | ✅ rc=0 — ${tail_line} | $(basename "${log}") |" >> "${SUMMARY}"
  else
    echo "| ${name} | ⚠️ rc=${rc} — ${tail_line} | $(basename "${log}") |" >> "${SUMMARY}"
  fi
  echo "[${name}] rc=${rc} — ${tail_line}" | tee -a "${OUT}/run.log"
}

# ----------------------------------------------------------------------------- #
# OFFLINE PHASE (guaranteed)
# ----------------------------------------------------------------------------- #
run_section "offline:matrix(all families x leaf sizes)" "${OUT}/offline_matrix.log" \
  "${TT_PY}" -m pytest "${TT_ROOT}/tests/ctreepo/test_treepo_families_across_leaf_sizes.py" \
  -v -p no:cacheprovider

run_section "offline:treepo.methods suite (oracle/fno/lda/learnable + contracts)" "${OUT}/offline_treepo_methods.log" \
  bash -c "cd '${TREEPO_ROOT}' && '${TREEPO_PY}' -m pytest tests/methods --ignore=tests/methods/integration -q -p no:cacheprovider"

run_section "offline:LDA real synthetic recovery" "${OUT}/offline_lda.log" \
  bash -c "cd '${TREEPO_ROOT}' && '${TREEPO_PY}' -m pytest tests/methods/test_fit_real_lda.py -v -p no:cacheprovider"

# ----------------------------------------------------------------------------- #
# LIVE SERVER BOOTSTRAP (opt-in via TT_START_SERVERS=1)
# Starts the LLM server the dspy/llm-live tests need, then waits for health.
# Failure here is non-fatal: live tests self-skip if the endpoint never comes up.
# ----------------------------------------------------------------------------- #
: "${TT_START_SERVERS:=0}"
: "${VLLM_PROFILE:=gemma-4-31b-it}"
: "${VLLM_CUDA_DEVICES:=0,1}"
: "${VLLM_HEALTH_TIMEOUT:=1800}"   # seconds to wait for the model to load

if [ "${RUN_LIVE}" = "1" ] && [ "${TT_START_SERVERS}" = "1" ]; then
  echo "=== [live:start LLM server ${VLLM_PROFILE} :${VLLM_PORT}] $(date -u +%H:%M:%S) ===" | tee -a "${OUT}/run.log"
  if curl -s -m 4 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[live:server] already up on :${VLLM_PORT}" | tee -a "${OUT}/run.log"
  else
    # The only vLLM is the gemma/dgemma fork (precompiled, links CUDA 13). In a
    # non-interactive env start_vllm.sh's cu13 LD setup may not fire, so the
    # precompiled extension fails to find libcudart.so.13. Prepend the cu13 libs
    # explicitly (verified: `LD_LIBRARY_PATH=<cu13>/lib python -c 'import vllm._C'`
    # succeeds). Non-invasive; does not modify start_vllm.sh.
    : "${VLLM_CU13_LIB:=/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib}"
    if [ -d "${VLLM_CU13_LIB}" ]; then
      export LD_LIBRARY_PATH="${VLLM_CU13_LIB}:${LD_LIBRARY_PATH:-}"
      echo "[live:server] prepended cu13 libs to LD_LIBRARY_PATH" | tee -a "${OUT}/run.log"
    fi
    nohup bash "${TT_ROOT}/scripts/start_vllm.sh" "${VLLM_PROFILE}" \
      --port "${VLLM_PORT}" --cuda-devices "${VLLM_CUDA_DEVICES}" --gpu-mem 0.85 \
      > "${OUT}/vllm_server.log" 2>&1 &
    STARTED_SERVER=1
    echo "[live:server] launched pid=$! profile=${VLLM_PROFILE} devices=${VLLM_CUDA_DEVICES}" | tee -a "${OUT}/run.log"
    waited=0
    until curl -s -m 4 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
      sleep 15; waited=$((waited+15))
      if [ "${waited}" -ge "${VLLM_HEALTH_TIMEOUT}" ]; then
        echo "[live:server] TIMEOUT after ${waited}s — live LLM tests will self-skip" | tee -a "${OUT}/run.log"
        break
      fi
    done
    if curl -s -m 4 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "[live:server] healthy after ${waited}s" | tee -a "${OUT}/run.log"
      echo "| live:LLM server | ✅ healthy (${VLLM_PROFILE} @ :${VLLM_PORT}, ${waited}s) | vllm_server.log |" >> "${SUMMARY}"
    else
      echo "| live:LLM server | ⚠️ not healthy within ${VLLM_HEALTH_TIMEOUT}s | vllm_server.log |" >> "${SUMMARY}"
    fi
  fi
fi

# ----------------------------------------------------------------------------- #
# LIVE PHASE (opt-in; tests self-skip when their server/data/GPU is absent)
# ----------------------------------------------------------------------------- #
if [ "${RUN_LIVE}" = "1" ]; then
  run_section "live:treepo integration (dspy/llm/fno/probe-NO)" "${OUT}/live_integration.log" \
    bash -c "cd '${TREEPO_ROOT}' && TT_RUN_LIVE_TESTS=1 VLLM_MODEL='${VLLM_MODEL}' VLLM_HOST='${VLLM_HOST}' VLLM_PORT='${VLLM_PORT}' '${TREEPO_PY}' -m pytest tests/methods/integration -v -p no:cacheprovider"

  run_section "live:TRL family smoke (GPU + tiny model)" "${OUT}/live_trl.log" \
    bash -c "cd '${TREEPO_ROOT}' && TT_RUN_LIVE_TESTS=1 CUDA_VISIBLE_DEVICES='${TRL_CUDA_DEVICE:-3}' '${TREEPO_PY}' -m pytest tests/methods/test_trl_family.py -v -p no:cacheprovider"
else
  echo "| live:* | ⏭️ skipped (live not requested) | — |" >> "${SUMMARY}"
fi

# ----------------------------------------------------------------------------- #
# CLEANUP — free GPUs if this job started the server
# ----------------------------------------------------------------------------- #
if [ "${STARTED_SERVER:-0}" = "1" ] && [ "${TT_KEEP_SERVERS:-0}" != "1" ]; then
  echo "=== [cleanup] stopping LLM server we started $(date -u +%H:%M:%S) ===" | tee -a "${OUT}/run.log"
  bash "${TT_ROOT}/scripts/stop_small_servers.sh" --all > "${OUT}/server_stop.log" 2>&1 || true
  echo "| cleanup:LLM server stopped | ✅ (set TT_KEEP_SERVERS=1 to keep) | server_stop.log |" >> "${SUMMARY}"
fi

echo "" >> "${SUMMARY}"
echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${SUMMARY}"
echo "DONE — summary at ${SUMMARY}" | tee -a "${OUT}/run.log"
