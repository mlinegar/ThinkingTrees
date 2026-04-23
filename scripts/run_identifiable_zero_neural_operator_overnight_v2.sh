#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
scripts/run_identifiable_zero_neural_operator_overnight_v2.sh was retired in the v2 simulation API.
Use the canonical suite commands instead:
  venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-neural-operator build --output-root <root>
  venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-neural-operator run --output-root <root> --jobs <jobs> --gpu-tokens <spec>
  venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-neural-operator report --output-root <root>
EOF
exit 2
