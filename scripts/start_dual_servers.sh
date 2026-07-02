#!/bin/bash
# Start dual backend servers for ThinkingTrees pipeline
#
# Small model (NVFP4 Nemotron 30B) on GPUs 0,1
# Large model (Qwen3.5-397B-A17B-NVFP4 teacher) on GPUs 2,3
# Ports are backend-aware defaults from config/settings.yaml.
#
# Usage:
#   ./scripts/start_dual_servers.sh              # Start both servers
#   ./scripts/start_dual_servers.sh --small-only # Start only small model
#   ./scripts/start_dual_servers.sh --large-only # Start only large model
#   ./scripts/start_dual_servers.sh --with-embedding  # Also start embedding server
#   ./scripts/start_dual_servers.sh --backend sglang  # Start both with SGLang

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

# Parse arguments
START_SMALL=true
START_LARGE=true
START_EMBEDDING=false
BACKEND="vllm"
SGLANG_VENV_PATH=""
SMALL_GPUS="0,1"
LARGE_GPUS="2,3"
EMBEDDING_GPUS=""
EMBEDDING_PROFILE_OVERRIDE=""
EMBEDDING_PORT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --small-only)
            START_LARGE=false
            shift
            ;;
        --large-only)
            START_SMALL=false
            shift
            ;;
        --with-embedding)
            START_EMBEDDING=true
            shift
            ;;
        --backend)
            BACKEND="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2
            ;;
        --sglang-venv-path)
            SGLANG_VENV_PATH="$2"
            shift 2
            ;;
        --small-gpus)
            SMALL_GPUS="$2"
            shift 2
            ;;
        --large-gpus)
            LARGE_GPUS="$2"
            shift 2
            ;;
        --embedding-gpus)
            EMBEDDING_GPUS="$2"
            shift 2
            ;;
        --embedding-profile)
            EMBEDDING_PROFILE_OVERRIDE="$2"
            shift 2
            ;;
        --embedding-port)
            EMBEDDING_PORT_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--small-only|--large-only] [--with-embedding] [--backend vllm|sglang] [--small-gpus IDS] [--large-gpus IDS] [--embedding-gpus IDS] [--embedding-profile NAME] [--embedding-port PORT] [--sglang-venv-path PATH]"
            exit 1
            ;;
    esac
done

if [[ "$BACKEND" != "vllm" && "$BACKEND" != "sglang" ]]; then
    echo "ERROR: --backend must be 'vllm' or 'sglang' (got '$BACKEND')"
    exit 1
fi

read -r VLLM_PORT SGLANG_PORT SGLANG_GENRM_PORT EMBEDDING_PORT EMBEDDING_PROFILE < <(python3 - <<PY
import yaml
from urllib.parse import urlparse
cfg = {}
with open("${CONFIG_FILE}") as f:
    cfg = yaml.safe_load(f) or {}
v = cfg.get("vllm", {}) if isinstance(cfg, dict) else {}
s = cfg.get("sglang", {}) if isinstance(cfg, dict) else {}
servers = cfg.get("servers", {}) if isinstance(cfg, dict) else {}
v_port = int(v.get("port", 8000) or 8000)
s_port = int(s.get("port", 30000) or 30000)
s_genrm_port = int(s.get("genrm_port", s_port + 1) or (s_port + 1))
embed_url = str(servers.get("embedding_url", "http://localhost:8003/v1") or "http://localhost:8003/v1")
embed_port = int(urlparse(embed_url).port or 8003)
embed_profile = str(servers.get("embedding_profile", "qwen3-embedding-8b") or "qwen3-embedding-8b")
print(v_port, s_port, s_genrm_port, embed_port, embed_profile)
PY
)
VLLM_PORT=${VLLM_PORT:-8000}
SGLANG_PORT=${SGLANG_PORT:-30000}
SGLANG_GENRM_PORT=${SGLANG_GENRM_PORT:-$((SGLANG_PORT + 1))}
EMBEDDING_PORT=${EMBEDDING_PORT:-8003}
EMBEDDING_PROFILE=${EMBEDDING_PROFILE:-qwen3-embedding-8b}
if [[ -n "$EMBEDDING_PORT_OVERRIDE" ]]; then
    EMBEDDING_PORT="$EMBEDDING_PORT_OVERRIDE"
fi
if [[ -n "$EMBEDDING_PROFILE_OVERRIDE" ]]; then
    EMBEDDING_PROFILE="$EMBEDDING_PROFILE_OVERRIDE"
fi
if [[ -z "$EMBEDDING_GPUS" ]]; then
    EMBEDDING_GPUS="$SMALL_GPUS"
fi
VLLM_GENRM_PORT=$((VLLM_PORT + 1))

if [[ "$BACKEND" == "sglang" ]]; then
    SMALL_PORT="$SGLANG_PORT"
    LARGE_PORT="$SGLANG_GENRM_PORT"
else
    SMALL_PORT="$VLLM_PORT"
    LARGE_PORT="$VLLM_GENRM_PORT"
fi

echo "=========================================="
echo "ThinkingTrees Dual Server Launcher"
echo "=========================================="
echo "Backend: $BACKEND"
echo "Ports:   small=${SMALL_PORT} large=${LARGE_PORT}"
if [[ "$START_EMBEDDING" == "true" ]]; then
    echo "         embed=${EMBEDDING_PORT}"
fi
echo "GPUs:    small=${SMALL_GPUS} large=${LARGE_GPUS}"
if [[ "$START_EMBEDDING" == "true" ]]; then
    echo "         embed=${EMBEDDING_GPUS}"
fi
if [[ -n "$SGLANG_VENV_PATH" ]]; then
    echo "SGLang venv: $SGLANG_VENV_PATH"
fi

# Create logs directory before background redirects.
mkdir -p "$SCRIPT_DIR/../logs"

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=300  # 5 minutes
    local waited=0

    echo "Waiting for $name server (port $port) to be ready..."
    while ! curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [[ $waited -ge $max_wait ]]; then
            echo "ERROR: $name server did not start within $max_wait seconds"
            return 1
        fi
        echo "  Still waiting... ($waited seconds)"
    done
    echo "$name server is ready!"
}

# Start small model server.
if [[ "$START_SMALL" == "true" ]]; then
    echo ""
    echo "Starting Small Model (NVFP4 Nemotron 30B)..."
    echo "  GPUs: ${SMALL_GPUS} | Port: ${SMALL_PORT}"
    if [[ "$BACKEND" == "vllm" ]]; then
        # Nemotron profiles have been more stable with eager startup (avoids CUDA graph path).
        CUDA_VISIBLE_DEVICES="${SMALL_GPUS}" "$SCRIPT_DIR/start_vllm.sh" nemotron-30b-nvfp4 \
            --port "${SMALL_PORT}" --kv-cache-dtype auto --enforce-eager > "$SCRIPT_DIR/../logs/small_model.log" 2>&1 &
    else
        SMALL_CMD=("$SCRIPT_DIR/start_sglang.sh" nemotron-30b-nvfp4 --port "${SMALL_PORT}")
        if [[ -n "$SGLANG_VENV_PATH" ]]; then
            SMALL_CMD+=(--sglang-venv-path "$SGLANG_VENV_PATH")
        fi
        CUDA_VISIBLE_DEVICES="${SMALL_GPUS}" "${SMALL_CMD[@]}" > "$SCRIPT_DIR/../logs/small_model.log" 2>&1 &
    fi
    SMALL_PID=$!
    echo "  PID: $SMALL_PID"
    echo "  Log: logs/small_model.log"
fi

# Start large model server.
if [[ "$START_LARGE" == "true" ]]; then
    echo ""
    echo "Starting Large Model (Qwen3.5-397B-A17B-NVFP4 teacher)..."
    echo "  GPUs: ${LARGE_GPUS} | Port: ${LARGE_PORT}"
    if [[ "$BACKEND" == "vllm" ]]; then
        CUDA_VISIBLE_DEVICES="${LARGE_GPUS}" "$SCRIPT_DIR/start_vllm.sh" qwen3.5-397b-a17b-nvfp4 \
            --port "${LARGE_PORT}" --cuda-devices "${LARGE_GPUS}" > "$SCRIPT_DIR/../logs/large_model.log" 2>&1 &
    else
        LARGE_CMD=("$SCRIPT_DIR/start_oracle_server.sh" \
            --backend "$BACKEND" --model qwen3.5-397b-a17b-nvfp4 --port "${LARGE_PORT}" --cuda-devices "${LARGE_GPUS}")
        if [[ -n "$SGLANG_VENV_PATH" ]]; then
            LARGE_CMD+=(--sglang-venv-path "$SGLANG_VENV_PATH")
        fi
        CUDA_VISIBLE_DEVICES="${LARGE_GPUS}" "${LARGE_CMD[@]}" > "$SCRIPT_DIR/../logs/large_model.log" 2>&1 &
    fi
    LARGE_PID=$!
    echo "  PID: $LARGE_PID"
    echo "  Log: logs/large_model.log"
fi

# Start embedding model server (typically colocated with small model).
if [[ "$START_EMBEDDING" == "true" ]]; then
    echo ""
    echo "Starting Embedding Model (${EMBEDDING_PROFILE})..."
    echo "  GPUs: ${EMBEDDING_GPUS} | Port: ${EMBEDDING_PORT}"
    CUDA_VISIBLE_DEVICES="${EMBEDDING_GPUS}" "$SCRIPT_DIR/start_vllm.sh" "${EMBEDDING_PROFILE}" \
        --port "${EMBEDDING_PORT}" > "$SCRIPT_DIR/../logs/embedding_model.log" 2>&1 &
    EMBEDDING_PID=$!
    echo "  PID: $EMBEDDING_PID"
    echo "  Log: logs/embedding_model.log"
fi

echo ""
echo "=========================================="
echo "Servers starting in background..."
echo "=========================================="

# Wait for servers if running in foreground mode
if [[ "$START_SMALL" == "true" ]]; then
    wait_for_server "${SMALL_PORT}" "Small model" &
    WAIT_SMALL_PID=$!
fi

if [[ "$START_LARGE" == "true" ]]; then
    wait_for_server "${LARGE_PORT}" "Large model" &
    WAIT_LARGE_PID=$!
fi
if [[ "$START_EMBEDDING" == "true" ]]; then
    wait_for_server "${EMBEDDING_PORT}" "Embedding model" &
    WAIT_EMBEDDING_PID=$!
fi

# Wait for all server checks
if [[ "$START_SMALL" == "true" ]]; then
    wait $WAIT_SMALL_PID
fi
if [[ "$START_LARGE" == "true" ]]; then
    wait $WAIT_LARGE_PID
fi
if [[ "$START_EMBEDDING" == "true" ]]; then
    wait $WAIT_EMBEDDING_PID
fi

echo ""
echo "=========================================="
echo "All servers ready!"
echo "=========================================="
echo ""
echo "Test with:"
if [[ "$START_SMALL" == "true" ]]; then
    echo "  curl http://localhost:${SMALL_PORT}/v1/models"
fi
if [[ "$START_LARGE" == "true" ]]; then
    echo "  curl http://localhost:${LARGE_PORT}/v1/models"
fi
if [[ "$START_EMBEDDING" == "true" ]]; then
    echo "  curl http://localhost:${EMBEDDING_PORT}/v1/models"
fi
echo ""
echo "Run OPS tree test:"
echo "  python main.py --input data/raw/manifesto_corpus_benoit/texts/33220_199603.txt --port ${SMALL_PORT} -v"
echo ""
echo "To stop servers:"
if [[ "$START_SMALL" == "true" ]]; then
    echo "  kill $SMALL_PID  # Small model"
fi
if [[ "$START_LARGE" == "true" ]]; then
    echo "  kill $LARGE_PID  # Large model"
fi
if [[ "$START_EMBEDDING" == "true" ]]; then
    echo "  kill $EMBEDDING_PID  # Embedding model"
fi

# Keep script running to maintain background processes
echo ""
echo "Press Ctrl+C to stop all servers..."
wait
