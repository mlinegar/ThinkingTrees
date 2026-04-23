#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
scripts/launch_lda_tree_recovery_production_sweeps.sh was retired in the v2 simulation API.
Use the canonical suite commands instead:
  venv/bin/python -m src.ctreepo.cli sim suite lda-tree-recovery-progress build --output-root <root>
  venv/bin/python -m src.ctreepo.cli sim suite lda-tree-recovery-progress run --output-root <root> --jobs <jobs> --gpu-tokens <spec>
  venv/bin/python -m src.ctreepo.cli sim suite lda-tree-recovery-progress report --output-root <root>
EOF
exit 2
