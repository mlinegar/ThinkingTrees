#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FORMAL_ROOT=""
LOG_DIR=""
MIG_UUIDS=""

usage() {
  cat <<'EOF'
Usage: scripts/run_formal_publication_recovery_mig_job.sh --formal-root PATH [options]

Options:
  --formal-root PATH   Formal rerun root.
  --log-dir PATH       Queue log directory.
  --mig-uuids TEXT     Space/comma-separated MIG UUIDs. Default: auto-discover.
  -h, --help           Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formal-root)
      FORMAL_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --mig-uuids)
      MIG_UUIDS="$2"
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

discover_migs() {
  nvidia-smi -L 2>/dev/null | sed -n '/MIG /s/.*(UUID: \([^)]*\)).*/\1/p' | paste -sd' ' -
}

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${FORMAL_ROOT}/paper_reports/logs/publication_clean_recovery_queue_logs"
elif [[ "$(basename "${LOG_DIR}")" == "logs" ]]; then
  LOG_DIR="${LOG_DIR}/publication_clean_recovery_queue_logs"
fi

if [[ -z "${MIG_UUIDS}" ]]; then
  MIG_UUIDS="$(discover_migs)"
fi

if [[ -z "${MIG_UUIDS}" ]]; then
  echo "No MIG UUIDs discovered; pass --mig-uuids explicitly." >&2
  exit 2
fi

CMD_FILE="${FORMAL_ROOT}/commands/identifiable_zero_longrun_clean_gpu.txt"

echo "mig_recovery_start formal_root=${FORMAL_ROOT} log_dir=${LOG_DIR}"
venv/bin/python scripts/run_mig_command_queue.py \
  --cmd-file "${CMD_FILE}" \
  --log-dir "${LOG_DIR}" \
  --mig-uuids "${MIG_UUIDS}"

MISSING_COUNT="$(
venv/bin/python - "${CMD_FILE}" <<'PY'
from pathlib import Path
import shlex
import sys

cmd_file = Path(sys.argv[1])
missing = 0
for line in cmd_file.read_text(encoding="utf-8").splitlines():
    item = line.strip()
    if not item:
        continue
    tokens = shlex.split(item)
    out = None
    for idx, token in enumerate(tokens[:-1]):
        if token == "--json-summary":
            out = Path(tokens[idx + 1])
            break
    if out is not None and not out.exists():
        missing += 1
print(missing)
PY
)"

echo "mig_recovery_missing_count=${MISSING_COUNT}"
if [[ "${MISSING_COUNT}" != "0" ]]; then
  echo "mig_recovery_incomplete"
  exit 1
fi

echo "mig_recovery_report_start"
venv/bin/python scripts/report_identifiable_zero_suite_publication_clean.py \
  --output-root "${FORMAL_ROOT}/identifiable_zero_longrun_clean" \
  --emit-pdf
venv/bin/python scripts/generate_paper_simulation_report_bundle.py \
  --formal-root "${FORMAL_ROOT}"
echo "mig_recovery_done"
