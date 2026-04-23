#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
scripts/run_lda_law_stress_suites.sh was retired in the v2 simulation API.
Use the canonical suite commands instead:
  venv/bin/python -m src.ctreepo.cli sim suite law-stress build --groups lda_sanity_suite --output-root <root>
  venv/bin/python -m src.ctreepo.cli sim suite law-stress run --groups lda_sanity_suite --output-root <root> --jobs <jobs> --gpu-tokens <spec>
  venv/bin/python -m src.ctreepo.cli sim suite law-stress report --family lda --output-root <root>
EOF
exit 2
