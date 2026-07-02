#!/usr/bin/env bash
# OLD_: archived 2026-07-02; launches OLD_run_dgemma_fleet_{live,dspy}.py. Kept for reference; do not run.
# Bring up a 4-GPU dgemma fleet (one server per GPU, standard vLLM OpenAI API)
# and run live diffusion across leaf sizes fanned across all 4. Fail-safe.
#
#   ports 8004 8005 8006 8007  ->  GPUs 0 1 2 3
#
# Env:
#   TT_KEEP_SERVERS=1   leave the fleet up after the run (default: stop it)
#   DGEMMA_PROFILE      vllm profile (default diffusiongemma-26b-a4b-it-nvfp4)
#   DGEMMA_MODEL        served model id (default RedHatAI/diffusiongemma-26B-A4B-it-NVFP4)

set -uo pipefail

TT_ROOT="/home/mlinegar/ThinkingTrees"
TT_PY="${TT_ROOT}/venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-${TT_ROOT}/outputs/dgemma_fleet_live_${STAMP}}"
mkdir -p "${OUT}"

: "${DGEMMA_PROFILE:=diffusiongemma-26b-a4b-it-nvfp4}"
: "${DGEMMA_MODEL:=RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}"
: "${VLLM_CU13_LIB:=/home/mlinegar/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib}"
: "${FLEET_HEALTH_TIMEOUT:=2400}"

PORTS=(8004 8005 8006 8007)
GPUS=(0 1 2 3)
STARTED=()

if [ -d "${VLLM_CU13_LIB}" ]; then
  export LD_LIBRARY_PATH="${VLLM_CU13_LIB}:${LD_LIBRARY_PATH:-}"
  echo "[fleet] cu13 libs on LD_LIBRARY_PATH" | tee -a "${OUT}/run.log"
fi

# Start any port not already serving.
for i in "${!PORTS[@]}"; do
  port="${PORTS[$i]}"; gpu="${GPUS[$i]}"
  if curl -s -m4 "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
    echo "[fleet] :${port} already up" | tee -a "${OUT}/run.log"
    continue
  fi
  echo "[fleet] starting dgemma :${port} on GPU ${gpu}" | tee -a "${OUT}/run.log"
  nohup bash "${TT_ROOT}/scripts/start_vllm.sh" "${DGEMMA_PROFILE}" \
    --port "${port}" --cuda-devices "${gpu}" --gpu-mem 0.85 \
    > "${OUT}/server_${port}.log" 2>&1 &
  STARTED+=("${port}")
done

# Wait for all four healthy.
echo "[fleet] waiting for all ports healthy (<=${FLEET_HEALTH_TIMEOUT}s)" | tee -a "${OUT}/run.log"
waited=0
while :; do
  up=0
  for port in "${PORTS[@]}"; do
    curl -s -m4 "http://localhost:${port}/v1/models" >/dev/null 2>&1 && up=$((up+1))
  done
  echo "[fleet] healthy ${up}/4 (t=${waited}s)" | tee -a "${OUT}/run.log"
  [ "${up}" -ge 4 ] && break
  [ "${waited}" -ge "${FLEET_HEALTH_TIMEOUT}" ] && { echo "[fleet] TIMEOUT — proceeding with ${up}/4" | tee -a "${OUT}/run.log"; break; }
  sleep 20; waited=$((waited+20))
done

# Build the live endpoint list from whatever is actually up.
LIVE_PORTS=()
for port in "${PORTS[@]}"; do
  curl -s -m4 "http://localhost:${port}/v1/models" >/dev/null 2>&1 && LIVE_PORTS+=("${port}")
done
echo "[fleet] live ports: ${LIVE_PORTS[*]:-none}" | tee -a "${OUT}/run.log"

if [ "${#LIVE_PORTS[@]}" -ge 1 ]; then
  echo "[fleet] running live diffusion (zero-shot) across leaf sizes" | tee -a "${OUT}/run.log"
  "${TT_PY}" "${TT_ROOT}/scripts/run_dgemma_fleet_live.py" \
    --ports "${LIVE_PORTS[@]}" --model "${DGEMMA_MODEL}" \
    --leaf-sizes 2 4 8 --out "${OUT}" >> "${OUT}/run.log" 2>&1
  echo "[fleet] diffusion driver rc=$?" | tee -a "${OUT}/run.log"

  echo "[fleet] running live dspy on dgemma (standard LLM path, round-robin)" | tee -a "${OUT}/run.log"
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TREEPO_MODEL_DIR="${TREEPO_MODEL_DIR:-/mnt/data/models}" \
  "${TT_PY}" "${TT_ROOT}/scripts/run_dgemma_fleet_dspy.py" \
    --ports "${LIVE_PORTS[@]}" --model "${DGEMMA_MODEL}" \
    --out "${OUT}" >> "${OUT}/run.log" 2>&1
  echo "[fleet] dspy driver rc=$?" | tee -a "${OUT}/run.log"
else
  echo "[fleet] no servers up — skipping drivers" | tee -a "${OUT}/run.log"
fi

# Cleanup: stop servers WE started (unless asked to keep).
if [ "${TT_KEEP_SERVERS:-0}" != "1" ] && [ "${#STARTED[@]}" -gt 0 ]; then
  echo "[fleet] stopping servers we started: ${STARTED[*]}" | tee -a "${OUT}/run.log"
  bash "${TT_ROOT}/scripts/stop_small_servers.sh" --all > "${OUT}/server_stop.log" 2>&1 || true
fi

echo "DONE — outputs in ${OUT}" | tee -a "${OUT}/run.log"
