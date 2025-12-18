#!/bin/bash
# Stop vLLM/SGLang servers to free GPU memory for optimization
#
# Usage:
#   ./scripts/stop_small_servers.sh              # Stop servers on all common ports
#   ./scripts/stop_small_servers.sh 8001 8002    # Stop servers on specific ports
#   ./scripts/stop_small_servers.sh --all        # Stop ALL vLLM servers (including main)
#
# This script gracefully stops vLLM/SGLang servers to free GPU memory
# for optimization runs that need all GPUs with tensor parallelism.

set -e

# Check for --all flag
if [[ "$1" == "--all" ]]; then
    # Stop ALL vLLM servers including main server on 8000
    PORTS_TO_STOP="8000 8001 8002 30000"
    echo "Stopping ALL vLLM servers (including main server on 8000)..."
else
    # Default: stop auxiliary servers only
    PORTS_TO_STOP=${@:-"8001 8002 30000"}
fi

echo "========================================"
echo "  GPU Memory Consolidation"
echo "========================================"
echo ""

stopped_count=0

for port in $PORTS_TO_STOP; do
    # Check if server is running on this port
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "Found server on port ${port}, stopping..."

        # Find PID using the port
        PID=$(lsof -t -i:${port} 2>/dev/null || true)

        if [ -n "$PID" ]; then
            # Send SIGTERM for graceful shutdown
            kill -TERM $PID 2>/dev/null || true
            echo "  Sent SIGTERM to PID $PID"

            # Wait up to 10 seconds for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 $PID 2>/dev/null; then
                    echo "  Server stopped gracefully"
                    break
                fi
                sleep 1
            done

            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                echo "  Server still running, sending SIGKILL..."
                kill -9 $PID 2>/dev/null || true
            fi

            ((stopped_count++)) || true
        else
            echo "  Could not find PID for port ${port}"
        fi
    else
        echo "No server running on port ${port}"
    fi
done

echo ""
echo "Stopped ${stopped_count} server(s)"
echo ""

# Wait for GPU memory to be released
echo "Waiting for GPU memory release..."
sleep 5

# Show GPU memory status
echo ""
echo "Current GPU memory status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not found"
fi

echo ""
echo "Done. GPU memory consolidated for optimization."
