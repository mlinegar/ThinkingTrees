#!/usr/bin/env bash
# Run vLLM and SGLang gate benchmarks sequentially on the same port.
#
# This script:
# 1) starts one backend on a shared OpenAI-compatible port
# 2) runs scripts/bench_arch_gates.py (Manifesto RILE + RULER)
# 3) stops servers
# 4) repeats for the other backend
# 5) writes a compact comparison summary
#
# Example:
#   ./scripts/run_backend_parity_gate.sh \
#     --profile nemotron-30b-nvfp4 \
#     --port 8000 \
#     --run-id quick_ablation \
#     --ruler-max-units 1 \
#     --ruler-max-problems 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY_BIN="${REPO_ROOT}/venv/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

PROFILE="nemotron-30b-nvfp4"
PORT=8000
OUTPUT_ROOT="${REPO_ROOT}/outputs/backend_parity"
RUN_ID=""
RULER_PHASE="S0_smoke"
RULER_MODE="runtime_full"
RULER_MAX_UNITS=1
RULER_MAX_PROBLEMS=20
RUN_VLLM=true
RUN_SGLANG=true
CUDA_DEVICES=""
MANIFESTO_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --ruler-phase)
      RULER_PHASE="$2"; shift 2 ;;
    --ruler-mode)
      RULER_MODE="$2"; shift 2 ;;
    --ruler-max-units)
      RULER_MAX_UNITS="$2"; shift 2 ;;
    --ruler-max-problems)
      RULER_MAX_PROBLEMS="$2"; shift 2 ;;
    --manifesto-ids)
      shift
      while [[ $# -gt 0 && "${1:-}" != --* ]]; do
        MANIFESTO_IDS+=("$1")
        shift
      done
      ;;
    --cuda-devices)
      CUDA_DEVICES="$2"; shift 2 ;;
    --vllm-only)
      RUN_SGLANG=false; shift ;;
    --sglang-only)
      RUN_VLLM=false; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

timestamp="$(date +%Y%m%d_%H%M%S)"
run_suffix="${RUN_ID:-$timestamp}"
run_root="${OUTPUT_ROOT}/${run_suffix}"
mkdir -p "${run_root}"

wait_for_endpoint() {
  local port="$1"
  local label="$2"
  local max_wait="${3:-300}"
  local elapsed=0
  while (( elapsed < max_wait )); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      echo "[ok] ${label} endpoint ready on :${port}"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "[error] ${label} endpoint failed to become ready on :${port} within ${max_wait}s" >&2
  return 1
}

start_backend() {
  local backend="$1"
  local server_log="$2"

  "${REPO_ROOT}/scripts/stop_small_servers.sh" --all >/dev/null 2>&1 || true

  if [[ "$backend" == "vllm" ]]; then
    cmd=("${REPO_ROOT}/scripts/start_vllm.sh" "${PROFILE}" "--port" "${PORT}")
  else
    cmd=("${REPO_ROOT}/scripts/start_sglang.sh" "${PROFILE}" "--port" "${PORT}")
  fi

  if [[ -n "$CUDA_DEVICES" ]]; then
    cmd+=("--cuda-devices" "$CUDA_DEVICES")
  fi

  echo "[run] starting ${backend} on :${PORT}"
  "${cmd[@]}" >"${server_log}" 2>&1 &
  server_pid=$!
  echo "${server_pid}" >"${server_log}.pid"

  if ! wait_for_endpoint "${PORT}" "${backend}"; then
    if kill -0 "${server_pid}" >/dev/null 2>&1; then
      kill "${server_pid}" >/dev/null 2>&1 || true
    fi
    return 1
  fi
  return 0
}

run_gate() {
  local backend="$1"
  local gate_dir="${run_root}/${backend}"
  local gate_log="${run_root}/${backend}_gate.log"
  mkdir -p "${gate_dir}"

  cmd=(
    "${PY_BIN}" "${REPO_ROOT}/scripts/bench_arch_gates.py"
    --run-dir "${gate_dir}"
    --manifesto-port "${PORT}"
    --ruler-phase "${RULER_PHASE}"
    --ruler-mode "${RULER_MODE}"
    --ruler-max-units "${RULER_MAX_UNITS}"
    --ruler-max-problems "${RULER_MAX_PROBLEMS}"
    --ruler-backend "${backend}"
    --ruler-backend-fallback none
    --ruler-model-base-url "http://localhost:${PORT}/v1"
  )

  if [[ ${#MANIFESTO_IDS[@]} -gt 0 ]]; then
    cmd+=(--manifesto-ids "${MANIFESTO_IDS[@]}")
  fi

  echo "[run] bench_arch_gates (${backend})"
  "${cmd[@]}" >"${gate_log}" 2>&1
}

run_backend() {
  local backend="$1"
  local server_log="${run_root}/${backend}_server.log"

  if ! start_backend "${backend}" "${server_log}"; then
    echo "[fail] failed to start ${backend}. See ${server_log}" >&2
    return 1
  fi

  if ! run_gate "${backend}"; then
    echo "[fail] gate run failed for ${backend}. See ${run_root}/${backend}_gate.log" >&2
    "${REPO_ROOT}/scripts/stop_small_servers.sh" --all >/dev/null 2>&1 || true
    return 1
  fi

  "${REPO_ROOT}/scripts/stop_small_servers.sh" --all >/dev/null 2>&1 || true
  return 0
}

status_vllm="skipped"
status_sglang="skipped"

if [[ "${RUN_VLLM}" == "true" ]]; then
  if run_backend "vllm"; then
    status_vllm="ok"
  else
    status_vllm="failed"
  fi
fi

if [[ "${RUN_SGLANG}" == "true" ]]; then
  if run_backend "sglang"; then
    status_sglang="ok"
  else
    status_sglang="failed"
  fi
fi

"${REPO_ROOT}/scripts/stop_small_servers.sh" --all >/dev/null 2>&1 || true

summary_path="${run_root}/comparison_summary.json"
"${PY_BIN}" - "${run_root}" "${status_vllm}" "${status_sglang}" >"${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
status_vllm = str(sys.argv[2])
status_sglang = str(sys.argv[3])

def load_metric(backend: str):
    gate_path = run_root / backend / "gate_metrics.json"
    if not gate_path.exists():
        return None
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    m = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return {
        "manifesto_mae": m.get("manifesto_mae"),
        "ruler_primary_mean": m.get("ruler_primary_mean"),
        "gate_metrics_path": str(gate_path),
    }

vllm = load_metric("vllm")
sglang = load_metric("sglang")

summary = {
    "run_root": str(run_root),
    "status": {"vllm": status_vllm, "sglang": status_sglang},
    "vllm": vllm,
    "sglang": sglang,
}

if vllm and sglang:
    try:
        v_mae = float(vllm["manifesto_mae"])
        s_mae = float(sglang["manifesto_mae"])
        summary["delta_manifesto_mae_sglang_minus_vllm"] = s_mae - v_mae
    except Exception:
        pass
    try:
        v_r = float(vllm["ruler_primary_mean"])
        s_r = float(sglang["ruler_primary_mean"])
        summary["delta_ruler_primary_mean_sglang_minus_vllm"] = s_r - v_r
    except Exception:
        pass

print(json.dumps(summary, indent=2))
PY

echo "=========================================="
echo "Backend parity run complete"
echo "Run root: ${run_root}"
echo "Summary: ${summary_path}"
echo "Status: vllm=${status_vllm} sglang=${status_sglang}"
echo "=========================================="

if [[ "${status_vllm}" == "failed" || "${status_sglang}" == "failed" ]]; then
  exit 1
fi
