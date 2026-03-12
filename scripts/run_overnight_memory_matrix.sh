#!/usr/bin/env bash
# Launch the manifesto memory overnight perf matrix in the background.
#
# Example:
#   ./scripts/run_overnight_memory_matrix.sh
#   ./scripts/run_overnight_memory_matrix.sh --profile overnight_memory_main

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

PY_BIN="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

MANIFEST="config/perf/manifesto_memory_overnight_matrix.yaml"
PROFILE="overnight_memory_full"
RUN_ID=""
FAIL_ON_REGRESSION=false
ALLOW_PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"; shift 2 ;;
    --profile)
      PROFILE="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --fail-on-regression)
      FAIL_ON_REGRESSION=true; shift ;;
    --allow-parallel)
      ALLOW_PARALLEL=true; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--manifest PATH] [--profile NAME] [--run-id ID] [--fail-on-regression] [--allow-parallel]" >&2
      exit 1 ;;
  esac
done

if [[ "$ALLOW_PARALLEL" != "true" ]]; then
  existing="$(pgrep -af "scripts/run_perf_harness.py.*manifesto_memory_overnight_matrix.yaml.*--profile ${PROFILE}" || true)"
  if [[ -n "$existing" ]]; then
    echo "Refusing to launch duplicate overnight matrix for profile '${PROFILE}'." >&2
    echo "Existing run(s):" >&2
    echo "$existing" >&2
    echo "If you really want parallel runs, pass --allow-parallel." >&2
    exit 2
  fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_suffix="${RUN_ID:-overnight_memory_${timestamp}}"
out_dir="${REPO_ROOT}/outputs/perf_harness/overnight_memory/${run_suffix}"
mkdir -p "$out_dir"

cmd=(
  "$PY_BIN" -u "${REPO_ROOT}/scripts/run_perf_harness.py"
  --manifest "$MANIFEST"
  --profile "$PROFILE"
  --output "${out_dir}/result.json"
)

if [[ "$FAIL_ON_REGRESSION" == "true" ]]; then
  cmd+=(--fail-on-regression)
fi

system_report_json="${out_dir}/system_comparison.json"
system_report_md="${out_dir}/system_comparison.md"

runner_script="${out_dir}/runner.sh"
cat >"${runner_script}" <<EOF
#!/usr/bin/env bash
set +e
$(printf '%q ' "${cmd[@]}")
rc=\$?
if [[ -f "${out_dir}/result.json" ]]; then
  "${PY_BIN}" -u "${REPO_ROOT}/scripts/recommend_manifesto_memory_defaults.py" \
    --artifact "${out_dir}/result.json" \
    --output "${out_dir}/recommended_defaults.json" || true
  "${PY_BIN}" -u "${REPO_ROOT}/scripts/report_manifesto_system_comparison.py" \
    --artifact "${out_dir}/result.json" \
    --recommended-defaults "${out_dir}/recommended_defaults.json" \
    --output "${system_report_json}" \
    --markdown-out "${system_report_md}" || true
fi
exit \$rc
EOF
chmod +x "${runner_script}"

nohup stdbuf -oL -eL bash "${runner_script}" >"${out_dir}/runner.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" >"${out_dir}/runner.pid"

echo "PID=${pid}"
echo "OUT=${out_dir}"
echo "LOG=${out_dir}/runner.log"
echo "ARTIFACT=${out_dir}/result.json"
echo "DEFAULTS=${out_dir}/recommended_defaults.json"
echo "SYSTEM_REPORT_JSON=${system_report_json}"
echo "SYSTEM_REPORT_MD=${system_report_md}"
echo ""
echo "Monitor:"
echo "  tail -f ${out_dir}/runner.log"
