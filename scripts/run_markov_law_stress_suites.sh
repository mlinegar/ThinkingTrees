#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'EOF'
scripts/run_markov_law_stress_suites.sh was retired in the v2 simulation API.
Use the canonical suite commands instead:
  venv/bin/python -m src.ctreepo.cli sim suite law-stress build --groups 'markov_sanity_suite markov_mechanism_suite' --output-root <root>
  venv/bin/python -m src.ctreepo.cli sim suite law-stress run --groups 'markov_sanity_suite markov_mechanism_suite' --output-root <root> --jobs <jobs> --gpu-tokens <spec>
  venv/bin/python -m src.ctreepo.cli sim suite law-stress report --family markov --output-root <root>
EOF
exit 2
