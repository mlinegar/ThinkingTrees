#!/bin/bash
# Standalone SGLang diffusion launcher for the fixed-binary prototype.
#
# This wraps scripts/start_sglang.sh so the diffusion path has a separate entry
# point and explicit DLLM flags without changing the existing chat/completions
# launcher contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE=""
ARGS=()

show_help() {
    cat <<'EOF'
SGLang Diffusion Launcher

Usage:
  ./scripts/start_sglang_diffusion.sh [PROFILE] [OPTIONS]

Options:
  --port PORT
  --cuda-devices IDS
  --tensor-parallel N
  --max-model-len N
  --mem-fraction-static RATIO
  --sglang-venv-path PATH
  --dllm-algorithm NAME
  --dllm-algorithm-config JSON
  -h, --help

Notes:
  - Unknown flags are forwarded to SGLang unchanged.
  - This launcher keeps the diffusion path separate from the AR chat launcher.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --*)
            ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                ARGS+=("$2")
                shift
            fi
            shift
            ;;
        *)
            if [[ -z "$PROFILE" ]]; then
                PROFILE="$1"
            else
                ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -n "$PROFILE" ]]; then
    exec "$SCRIPT_DIR/start_sglang.sh" "$PROFILE" "${ARGS[@]}"
fi

exec "$SCRIPT_DIR/start_sglang.sh" "${ARGS[@]}"
