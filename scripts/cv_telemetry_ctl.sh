#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/cv_telemetry_ctl.sh start  --cv-output-dir <dir> [--interval-sec N] [--out-jsonl <path>]
  scripts/cv_telemetry_ctl.sh stop   --cv-output-dir <dir>
  scripts/cv_telemetry_ctl.sh status --cv-output-dir <dir>
  scripts/cv_telemetry_ctl.sh summarize --cv-output-dir <dir> [--throughput-json <path>]

Notes:
  - PID file: <cv-output-dir>/telemetry/cv_telemetry.pid
  - Log file: <cv-output-dir>/telemetry/cv_telemetry.log
  - JSONL default: <cv-output-dir>/telemetry/cv_telemetry.jsonl
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ACTION="$1"
shift

CV_DIR=""
INTERVAL_SEC="30"
OUT_JSONL=""
THROUGHPUT_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cv-output-dir)
      CV_DIR="$2"
      shift 2
      ;;
    --interval-sec)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --out-jsonl)
      OUT_JSONL="$2"
      shift 2
      ;;
    --throughput-json)
      THROUGHPUT_JSON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CV_DIR" ]]; then
  echo "--cv-output-dir is required." >&2
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CV_DIR_ABS="$(cd "$ROOT_DIR" && realpath "$CV_DIR")"
TELEM_DIR="$CV_DIR_ABS/telemetry"
PID_FILE="$TELEM_DIR/cv_telemetry.pid"
LOG_FILE="$TELEM_DIR/cv_telemetry.log"
if [[ -z "$OUT_JSONL" ]]; then
  OUT_JSONL="$TELEM_DIR/cv_telemetry.jsonl"
fi

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_logger() {
  mkdir -p "$TELEM_DIR"
  if is_running; then
    echo "Telemetry logger already running (pid $(cat "$PID_FILE"))."
    exit 0
  fi

  cd "$ROOT_DIR"
  nohup ./venv/bin/python scripts/log_cv_telemetry.py \
    --cv-output-dir "$CV_DIR_ABS" \
    --interval-sec "$INTERVAL_SEC" \
    --out-jsonl "$OUT_JSONL" \
    >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  sleep 1
  if ! is_running; then
    echo "Telemetry logger failed to stay running. Check: $LOG_FILE" >&2
    exit 1
  fi

  echo "Started telemetry logger."
  echo "  pid file:  $PID_FILE"
  echo "  log file:  $LOG_FILE"
  echo "  jsonl out: $OUT_JSONL"
}

stop_logger() {
  if ! is_running; then
    echo "Telemetry logger is not running."
    if [[ -f "$PID_FILE" ]]; then
      : > "$PID_FILE"
    fi
    exit 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  : > "$PID_FILE"
  echo "Stopped telemetry logger (pid $pid)."
}

status_logger() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "Telemetry logger: running (pid $pid)"
  else
    echo "Telemetry logger: stopped"
  fi
  echo "  pid file:  $PID_FILE"
  echo "  log file:  $LOG_FILE"
  echo "  jsonl out: $OUT_JSONL"
  if [[ -f "$OUT_JSONL" ]]; then
    echo "  samples:   $(wc -l < "$OUT_JSONL")"
    echo "  last:      $(tail -n 1 "$OUT_JSONL" 2>/dev/null || true)"
  fi
}

summarize_logger() {
  cd "$ROOT_DIR"
  local cmd=(./venv/bin/python scripts/summarize_cv_telemetry.py --cv-output-dir "$CV_DIR_ABS")
  if [[ -n "$OUT_JSONL" ]]; then
    cmd+=(--telemetry-jsonl "$OUT_JSONL")
  fi
  if [[ -n "$THROUGHPUT_JSON" ]]; then
    cmd+=(--throughput-json "$THROUGHPUT_JSON")
  fi
  "${cmd[@]}"
}

case "$ACTION" in
  start)
    start_logger
    ;;
  stop)
    stop_logger
    ;;
  status)
    status_logger
    ;;
  summarize)
    summarize_logger
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage
    exit 1
    ;;
esac
