#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FORMAL_ROOT=""
JOBS="128"

usage() {
  cat <<'EOF'
Usage: scripts/run_formal_publication_recovery_cpu_job.sh --formal-root PATH [options]

Options:
  --formal-root PATH   Formal rerun root.
  --jobs N             CPU worker count. Default: 128.
  -h, --help           Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formal-root)
      FORMAL_ROOT="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${FORMAL_ROOT}" ]]; then
  echo "Missing --formal-root" >&2
  usage >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

echo "cpu_recovery_start formal_root=${FORMAL_ROOT} jobs=${JOBS}"
export JOBS="${JOBS}"
export GPU_TOKENS="none"
export MARKOV_DEVICE="cpu"
export OUT_ROOT="${FORMAL_ROOT}/identifiable_zero_learnability"
bash ./scripts/run_identifiable_zero_learnability_overnight.sh

echo "cpu_recovery_report_start"
venv/bin/python scripts/report_identifiable_zero_learnability.py \
  --output-root "${FORMAL_ROOT}/identifiable_zero_learnability" \
  --emit-pdf
venv/bin/python scripts/generate_paper_simulation_report_bundle.py \
  --formal-root "${FORMAL_ROOT}"
echo "cpu_recovery_done"
