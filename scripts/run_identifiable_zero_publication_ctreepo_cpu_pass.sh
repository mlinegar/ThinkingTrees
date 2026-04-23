#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
scripts/run_identifiable_zero_publication_ctreepo_cpu_pass.sh was retired in the v2 simulation API.
Use the canonical suite commands instead:
  venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo build --output-root <root>
  venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo run --output-root <root> --jobs <jobs> --gpu-tokens none
  venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo report --output-root <root>
EOF
exit 2
