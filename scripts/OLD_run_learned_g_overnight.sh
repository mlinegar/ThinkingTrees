#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PY_BIN="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

JOBS="${JOBS:-64}"
MAX_HOURS="${MAX_HOURS:-10}"
MAX_ROUNDS="${MAX_ROUNDS:-24}"
CANDIDATES_PER_EXPERIMENT="${CANDIDATES_PER_EXPERIMENT:-8}"
SEEDS="${SEEDS:-0,1,2,3}"
PROFILE="${PROFILE:-overnight}"
OUT_ROOT="${OUT_ROOT:-}"
STOP_WHEN_RECOVERED="${STOP_WHEN_RECOVERED:-0}"
ALLOW_PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"; shift 2 ;;
    --max-hours)
      MAX_HOURS="$2"; shift 2 ;;
    --max-rounds)
      MAX_ROUNDS="$2"; shift 2 ;;
    --candidates-per-experiment)
      CANDIDATES_PER_EXPERIMENT="$2"; shift 2 ;;
    --seeds)
      SEEDS="$2"; shift 2 ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --stop-when-recovered)
      STOP_WHEN_RECOVERED=1; shift ;;
    --allow-parallel)
      ALLOW_PARALLEL=true; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--jobs N] [--max-hours H] [--max-rounds N] [--candidates-per-experiment N] [--seeds CSV] [--profile overnight|smoke] [--out-root PATH] [--stop-when-recovered] [--allow-parallel]" >&2
      exit 1 ;;
  esac
done

if [[ "$ALLOW_PARALLEL" != "true" ]]; then
  existing="$(pgrep -af "scripts/run_learned_g_overnight.py" || true)"
  if [[ -n "$existing" ]]; then
    echo "Refusing to launch duplicate learned-g overnight run." >&2
    echo "Existing run(s):" >&2
    echo "$existing" >&2
    echo "Use --allow-parallel to override." >&2
    exit 2
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="${REPO_ROOT}/outputs/learned_g_overnight_${STAMP}"
fi
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

RUN_LOG="${OUT_ROOT}/runner.log"
PID_FILE="${OUT_ROOT}/runner.pid"
STATUS_FILE="${OUT_ROOT}/overnight_status.json"
REPORT_FILE="${OUT_ROOT}/overnight_report.md"
HISTORY_FILE="${OUT_ROOT}/round_history.jsonl"

cmd=(
  "$PY_BIN" -u "${REPO_ROOT}/scripts/run_learned_g_overnight.py"
  --output-root "$OUT_ROOT"
  --jobs "$JOBS"
  --max-hours "$MAX_HOURS"
  --max-rounds "$MAX_ROUNDS"
  --candidates-per-experiment "$CANDIDATES_PER_EXPERIMENT"
  --seeds "$SEEDS"
  --profile "$PROFILE"
)
if [[ "$STOP_WHEN_RECOVERED" == "1" ]]; then
  cmd+=(--stop-when-recovered)
fi

nohup stdbuf -oL -eL "${cmd[@]}" >"$RUN_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"$PID_FILE"

echo "PID=${pid}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG=${RUN_LOG}"
echo "STATUS=${STATUS_FILE}"
echo "HISTORY=${HISTORY_FILE}"
echo "REPORT=${REPORT_FILE}"
echo ""
echo "Monitor:"
echo "  tail -f ${RUN_LOG}"
