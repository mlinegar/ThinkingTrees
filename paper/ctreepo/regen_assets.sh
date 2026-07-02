#!/usr/bin/env bash
# Master driver: regenerate every figure and table consumed by main_new.tex
# from the source-of-truth `outputs/` JSONs / CSVs, and stage them under
# paper/ctreepo/assets/<example>/{figures,tables}/.
#
# This is the *only* command a reader needs to run before recompiling the
# paper. Per-example sub-scripts live in paper/ctreepo/scripts/ and can be
# called individually for faster iteration on a single example.
#
# Usage:
#     bash paper/ctreepo/regen_assets.sh [example...]
#
# With no arguments, regenerates every example. With one or more example
# names (markov, hll, benoit), regenerates only those.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/paper/ctreepo/scripts"

EXAMPLES=("$@")
if [[ ${#EXAMPLES[@]} -eq 0 ]]; then
    EXAMPLES=(markov hll benoit)
fi

failed=0
for ex in "${EXAMPLES[@]}"; do
    script="$SCRIPTS_DIR/regen_${ex}.sh"
    if [[ ! -x "$script" ]]; then
        echo "[regen_assets] no sub-script for example=$ex (expected $script)"
        failed=1
        continue
    fi
    echo "=== [$ex] regenerating ==="
    if ! bash "$script"; then
        echo "[regen_assets] FAILED: example=$ex"
        failed=1
    fi
done

if [[ $failed -ne 0 ]]; then
    echo "[regen_assets] one or more examples failed; check output above"
    exit 1
fi
echo "[regen_assets] all examples regenerated successfully"
