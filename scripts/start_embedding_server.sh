#!/bin/bash
# Start a dedicated multilingual embedding server (vLLM) for ThinkingTrees.
#
# Defaults:
# - Profile: settings.yaml servers.embedding_profile (fallback: qwen3-embedding-8b)
# - Port:    derived from settings.yaml servers.embedding_url (fallback: 8003)
#
# Usage:
#   ./scripts/start_embedding_server.sh
#   ./scripts/start_embedding_server.sh --cuda-devices 3
#   ./scripts/start_embedding_server.sh --port 8003 --profile qwen3-embedding-8b
#
# Notes:
# - If you change the server URL, also set EMBEDDING_URL (or update settings.yaml).
# - If you change the served model id, set EMBEDDING_MODEL (or update settings.yaml).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

PROFILE_OVERRIDE=""
PORT_OVERRIDE=""
CUDA_DEVICES=""
LOG_FILE="$PROJECT_ROOT/logs/embedding_model.log"
FOREGROUND=false
MAX_WAIT_SECONDS=300

show_help() {
    cat <<'EOF'
Start ThinkingTrees embedding server (vLLM)

Usage:
  ./scripts/start_embedding_server.sh [OPTIONS]

Options:
  --profile PROFILE         vLLM profile (default: settings servers.embedding_profile)
  --port PORT               Server port (default: settings servers.embedding_url port)
  --cuda-devices IDS        Set CUDA_VISIBLE_DEVICES (e.g. 3)
  --log-file PATH           Log file (default: logs/embedding_model.log)
  --foreground              Run in foreground (default: background + readiness wait)
  --max-wait-seconds N      Readiness wait timeout (default: 300)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --profile)
            PROFILE_OVERRIDE="$2"
            shift 2
            ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --foreground)
            FOREGROUND=true
            shift
            ;;
        --max-wait-seconds)
            MAX_WAIT_SECONDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# Detached launchers such as scripts/long_job.py run under systemd. In that
# context, keep the server in the foreground so the unit owns the actual API
# process instead of a short-lived wrapper that backgrounds it.
if [[ "$FOREGROUND" != "true" ]]; then
    if [[ -n "${INVOCATION_ID:-}" || -n "${JOURNAL_STREAM:-}" ]]; then
        FOREGROUND=true
    fi
fi

read -r DEFAULT_PROFILE DEFAULT_PORT < <(python3 - <<PY
import os
from urllib.parse import urlparse

import yaml

cfg = {}
with open("${CONFIG_FILE}") as f:
    cfg = yaml.safe_load(f) or {}
servers = cfg.get("servers", {}) if isinstance(cfg, dict) else {}

url = (
    os.environ.get("EMBEDDING_URL")
    or (servers.get("embedding_url") if isinstance(servers, dict) else None)
    or "http://localhost:8003/v1"
)
parsed = urlparse(str(url))
port = parsed.port or 8003

profile = (
    (servers.get("embedding_profile") if isinstance(servers, dict) else None)
    or "qwen3-embedding-8b"
)

print(str(profile), int(port))
PY
)

PROFILE="${PROFILE_OVERRIDE:-$DEFAULT_PROFILE}"
PORT="${PORT_OVERRIDE:-$DEFAULT_PORT}"

mkdir -p "$(dirname "$LOG_FILE")"

check_server() {
    curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1
    return $?
}

CMD=("$SCRIPT_DIR/start_vllm.sh" "$PROFILE" --port "$PORT")
if [[ -n "$CUDA_DEVICES" ]]; then
    CMD+=(--cuda-devices "$CUDA_DEVICES")
fi

echo "=========================================="
echo "ThinkingTrees Embedding Server Launcher"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Port:    $PORT"
if [[ -n "$CUDA_DEVICES" ]]; then
    echo "CUDA:    $CUDA_DEVICES"
fi
echo "Log:     $LOG_FILE"
echo "=========================================="

if [[ "$FOREGROUND" == "true" ]]; then
    exec "${CMD[@]}"
fi

if check_server; then
    echo "Embedding server already running on port $PORT."
    exit 0
fi

"${CMD[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
echo "Started embedding server (PID: $PID)"

echo "Waiting for embedding server to be ready..."
waited=0
while ! check_server; do
    sleep 5
    waited=$((waited + 5))
    if [[ "$waited" -ge "$MAX_WAIT_SECONDS" ]]; then
        echo "ERROR: embedding server did not start within $MAX_WAIT_SECONDS seconds"
        echo "  Tail log: tail -n 80 $LOG_FILE"
        exit 1
    fi
    echo "  Still waiting... (${waited}s)"
done

echo "Embedding server is ready."
echo "Test with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "Stop with:"
echo "  ./scripts/stop_small_servers.sh ${PORT}"
