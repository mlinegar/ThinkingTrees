#!/bin/bash
# Start dual vLLM servers for ThinkingTrees pipeline
#
# Small model (FP8 Nemotron 30B) on GPUs 0,1 - Port 8000
# Large model (GenRM NVFP4 235B) on GPUs 2,3 - Port 8001
#
# Usage:
#   ./scripts/start_dual_servers.sh              # Start both servers
#   ./scripts/start_dual_servers.sh --small-only # Start only small model
#   ./scripts/start_dual_servers.sh --large-only # Start only large model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
START_SMALL=true
START_LARGE=true
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--small-only|--large-only]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ThinkingTrees Dual Server Launcher"
echo "=========================================="

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

# Start small model server (GPUs 0,1)
if [[ "$START_SMALL" == "true" ]]; then
    echo ""
    echo "Starting Small Model (FP8 Nemotron 30B)..."
    echo "  GPUs: 0,1 | Port: 8000"
    CUDA_VISIBLE_DEVICES=0,1 "$SCRIPT_DIR/start_vllm.sh" nemotron-30b-fp8 \
        --port 8000 --kv-cache-dtype auto > "$SCRIPT_DIR/../logs/small_model.log" 2>&1 &
    SMALL_PID=$!
    echo "  PID: $SMALL_PID"
    echo "  Log: logs/small_model.log"
fi

# Start large model server (GPUs 2,3)
if [[ "$START_LARGE" == "true" ]]; then
    echo ""
    echo "Starting Large Model (GenRM NVFP4 235B)..."
    echo "  GPUs: 2,3 | Port: 8001"
    CUDA_VISIBLE_DEVICES=2,3 "$SCRIPT_DIR/start_vllm.sh" genrm-nvfp4 \
        --port 8001 > "$SCRIPT_DIR/../logs/large_model.log" 2>&1 &
    LARGE_PID=$!
    echo "  PID: $LARGE_PID"
    echo "  Log: logs/large_model.log"
fi

echo ""
echo "=========================================="
echo "Servers starting in background..."
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/../logs"

# Wait for servers if running in foreground mode
if [[ "$START_SMALL" == "true" ]]; then
    wait_for_server 8000 "Small model" &
    WAIT_SMALL_PID=$!
fi

if [[ "$START_LARGE" == "true" ]]; then
    wait_for_server 8001 "Large model" &
    WAIT_LARGE_PID=$!
fi

# Wait for all server checks
if [[ "$START_SMALL" == "true" ]]; then
    wait $WAIT_SMALL_PID
fi
if [[ "$START_LARGE" == "true" ]]; then
    wait $WAIT_LARGE_PID
fi

echo ""
echo "=========================================="
echo "All servers ready!"
echo "=========================================="
echo ""
echo "Test with:"
if [[ "$START_SMALL" == "true" ]]; then
    echo "  curl http://localhost:8000/v1/models"
fi
if [[ "$START_LARGE" == "true" ]]; then
    echo "  curl http://localhost:8001/v1/models"
fi
echo ""
echo "Run OPS tree test:"
echo "  python main.py --input data/raw/manifesto_project_full/texts/33220_199603.txt --port 8000 -v"
echo ""
echo "To stop servers:"
if [[ "$START_SMALL" == "true" ]]; then
    echo "  kill $SMALL_PID  # Small model"
fi
if [[ "$START_LARGE" == "true" ]]; then
    echo "  kill $LARGE_PID  # Large model"
fi

# Keep script running to maintain background processes
echo ""
echo "Press Ctrl+C to stop all servers..."
wait
