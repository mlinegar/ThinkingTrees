#!/usr/bin/env bash
# Run several scalar Manifesto dimensions through the DSPy f/g ladder, grouped
# by leaf-size context buckets. This keeps vLLM sizing/batching efficient:
# start one server per leaf/context group, then run every requested dimension
# through the still-hot server before moving to the next group.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/manifesto_ladder_runtime.sh"

ROOT="${1:-outputs/manifesto_fg_alternating/scalar_dims_benoit_g0init_fresh_dspy_$(date +%Y%m%d_%H%M%S)}"
DIMENSIONS="${DIMENSIONS:-economic social immigration eu environment decentralization}"
SOURCE_RESULTS_ROOT="${SOURCE_RESULTS_ROOT:-outputs/overnight_benoit/full_pipeline}"
SPLIT_IDS_DIR="${SPLIT_IDS_DIR:-}"
DIMENSION_ROOT_OVERRIDES="${DIMENSION_ROOT_OVERRIDES:-}"
DIMENSION_LEAF_FILTERS="${DIMENSION_LEAF_FILTERS:-}"

PROFILE="${PROFILE:-gemma-4-31b-it-nvfp4}"
PORT="${PORT:-8010}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
KEEP_LAST_SERVER="${KEEP_LAST_SERVER:-0}"
PRESTOP_SERVER_JOB_ROOT="${PRESTOP_SERVER_JOB_ROOT:-}"
REUSE_FIRST_SERVER_JOB_ROOT="${REUSE_FIRST_SERVER_JOB_ROOT:-}"
LEAF_CONTEXT_GROUPS="${LEAF_CONTEXT_GROUPS:-$(manifesto_leaf_context_group_defaults)}"
GROUP_INDEX_OFFSET="${GROUP_INDEX_OFFSET:-0}"

WAIT_FOR_JOB_ROOTS="${WAIT_FOR_JOB_ROOTS:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-120}"
RERUN_COMPLETED="${RERUN_COMPLETED:-0}"
PLOT_LADDER_GRID="${PLOT_LADDER_GRID:-1}"
PLOT_PREDICTION_DISTS="${PLOT_PREDICTION_DISTS:-1}"
BALANCED_TEST_N="${BALANCED_TEST_N:-}"
BALANCED_VAL_N="${BALANCED_VAL_N:-${VAL_N:-30}}"
BALANCED_MIN_TEST_N="${BALANCED_MIN_TEST_N:-${BALANCED_TEST_N}}"
BALANCED_MIN_TRAIN_N="${BALANCED_MIN_TRAIN_N:-1}"

NOFILE_LIMIT="${NOFILE_LIMIT:-65535}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-g}"
INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}"
INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-0}"
STAGE_NAMING="${STAGE_NAMING:-powers}"
DSPY_OPTIMIZER="${DSPY_OPTIMIZER:-mipro}"
DSPY_BUDGET="${DSPY_BUDGET:-light}"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-128}"
DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-0}"
DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT:-batch}"
DSPY_BATCH_SIZE="${DSPY_BATCH_SIZE:-64}"
DSPY_BATCH_TIMEOUT="${DSPY_BATCH_TIMEOUT:-0.02}"
DSPY_BATCH_REQUEST_TIMEOUT="${DSPY_BATCH_REQUEST_TIMEOUT:-300}"
DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT="${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT:-}"
DSPY_BATCH_ROUTING_POLICY="${DSPY_BATCH_ROUTING_POLICY:-affinity_load_aware}"
DSPY_F_INIT_PATH="${DSPY_F_INIT_PATH:-}"
DSPY_F_INIT_MODE="${DSPY_F_INIT_MODE:-pretuned_scorer}"
DSPY_MAX_TRAIN_RECORDS="${DSPY_MAX_TRAIN_RECORDS:-}"
DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS="${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS:-3}"
DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS:-1500}"
TEACHER_TIMEOUT_SECONDS="${TEACHER_TIMEOUT_SECONDS:-600}"
SCORER_TIMEOUT_SECONDS="${SCORER_TIMEOUT_SECONDS:-600}"
EXPERT_TARGET_SCALE_OVERRIDE="${EXPERT_TARGET_SCALE:-}"
SCORING_CONTEXT_SOURCE="${SCORING_CONTEXT_SOURCE:-compact}"
ROOT_LABEL_SOURCES="${ROOT_LABEL_SOURCES:-stored_summary}"
ROOT_LABEL_TARGET="${ROOT_LABEL_TARGET:-expert}"
if [[ -n "${FULL_DOC_ANCHOR_MODE+x}" || -n "${FULL_DOC_ANCHOR_TARGET+x}" ]]; then
  echo "ERROR: FULL_DOC_ANCHOR_* is no longer public; use ROOT_LABEL_SOURCES and ROOT_LABEL_TARGET." >&2
  exit 2
fi
if [[ -n "${FULL_DOC_ANCHOR_WEIGHT+x}" ]]; then
  echo "ERROR: full-doc anchor weight is no longer supported; use LOCAL_LAW_WEIGHT." >&2
  exit 2
fi
if [[ -n "${TEACHER_NODE_LAMBDA+x}" ]]; then
  echo "ERROR: teacher-node lambda is no longer supported; use LOCAL_LAW_WEIGHT." >&2
  exit 2
fi
if [[ -n "${GOLD_STANDARD_LAMBDA+x}" ]]; then
  echo "ERROR: GOLD_STANDARD_LAMBDA is no longer an objective input; use LOCAL_LAW_WEIGHT." >&2
  exit 2
fi
LOCAL_LAW_WEIGHT="${LOCAL_LAW_WEIGHT:-}"
if [[ -z "${ROOT_LABEL_SOURCES}" ]]; then
  LOCAL_LAW_WEIGHT="${LOCAL_LAW_WEIGHT:-1.0}"
else
  LOCAL_LAW_WEIGHT="${LOCAL_LAW_WEIGHT:-0.25}"
fi
NODE_WEIGHT_NORMALIZATION="${NODE_WEIGHT_NORMALIZATION:-per_tree}"

ulimit -n "${NOFILE_LIMIT}" 2>/dev/null || true
mkdir -p "${ROOT}"

current_server_job_root=""

stop_server_job_root() {
  local job_root="$1"
  if [[ -z "${job_root}" || ! -f "${job_root}/manifest.json" ]]; then
    return
  fi
  ./venv/bin/python scripts/long_job.py stop --job-root "${job_root}" >/dev/null 2>&1 || true
}

cleanup() {
  if [[ "${KEEP_LAST_SERVER}" != "1" && -n "${current_server_job_root}" ]]; then
    echo "=== $(date -u) :: stopping server ${current_server_job_root} ==="
    stop_server_job_root "${current_server_job_root}"
  fi
}
trap cleanup EXIT

long_job_running() {
  local job_root="$1"
  if [[ -z "${job_root}" || ! -f "${job_root}/manifest.json" ]]; then
    return 1
  fi
  ./venv/bin/python scripts/long_job.py status --job-root "${job_root}" --tail-lines 0 \
    | ./venv/bin/python -c 'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("running") else 1)'
}

wait_for_job_root() {
  local job_root="$1"
  if [[ -z "${job_root}" ]]; then
    return
  fi
  while long_job_running "${job_root}"; do
    echo "=== $(date -u) :: waiting for prior job ${job_root} ==="
    sleep "${WAIT_POLL_SECONDS}"
  done
}

wait_for_server() {
  local port="$1"
  local timeout="$2"
  local deadline=$((SECONDS + timeout))
  until curl -sS --max-time 3 "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "ERROR: server on port ${port} did not become ready within ${timeout}s" >&2
      return 1
    fi
    sleep 5
  done
}

start_server_for_group() {
  local idx="$1"
  local context_len="$2"
  local gpu_mem="$3"
  local max_num_seqs="$4"
  local job_root="${ROOT}/server_ctx${context_len}_seq${max_num_seqs}_group${idx}"

  echo "=== $(date -u) :: starting ${PROFILE} on :${PORT} context=${context_len} gpu_mem=${gpu_mem} max_num_seqs=${max_num_seqs} ==="
  ./venv/bin/python scripts/long_job.py launch \
    --name "scalar_dims_gemma4_31b_8010_ctx${context_len}_seq${max_num_seqs}_group${idx}" \
    --job-root "${job_root}" \
    --cwd "$(pwd)" \
    -- ./scripts/start_vllm.sh "${PROFILE}" \
      --port "${PORT}" \
      --cuda-devices "${CUDA_DEVICES}" \
      --tensor-parallel "${TENSOR_PARALLEL}" \
      --max-model-len "${context_len}" \
      --gpu-mem "${gpu_mem}" \
      --max-num-seqs "${max_num_seqs}" \
    >/dev/null
  current_server_job_root="${job_root}"
  wait_for_server "${PORT}" "${SERVER_START_TIMEOUT_SECONDS}"
}

leaf_dir_name() {
  printf 'leaf%04dtok' "$1"
}

dimension_root_for() {
  local dimension="$1"
  local override key value
  for override in ${DIMENSION_ROOT_OVERRIDES}; do
    key="${override%%=*}"
    value="${override#*=}"
    if [[ "${key}" == "${dimension}" && "${value}" != "${override}" ]]; then
      printf '%s\n' "${value}"
      return
    fi
  done
  printf '%s\n' "${ROOT}/${dimension}"
}

dimension_leaf_filter_for() {
  local dimension="$1"
  local override key value
  for override in ${DIMENSION_LEAF_FILTERS}; do
    key="${override%%=*}"
    value="${override#*=}"
    if [[ "${key}" == "${dimension}" && "${value}" != "${override}" ]]; then
      printf '%s\n' "${value}"
      return
    fi
  done
}

expert_target_scale_for() {
  local dimension="$1"
  if [[ -n "${EXPERT_TARGET_SCALE_OVERRIDE}" ]]; then
    printf '%s\n' "${EXPERT_TARGET_SCALE_OVERRIDE}"
    return
  fi
  printf 'normalized_1_7\n'
}

filter_leaves_csv() {
  local leaves_csv="$1"
  local allowed_csv="$2"
  local raw_leaf leaf raw_allowed allowed
  local filtered=()

  if [[ -z "${allowed_csv}" ]]; then
    printf '%s\n' "${leaves_csv}"
    return
  fi

  IFS=',' read -r -a leaf_values <<< "${leaves_csv}"
  IFS=',' read -r -a allowed_values <<< "${allowed_csv}"
  for raw_leaf in "${leaf_values[@]}"; do
    leaf="$(echo "${raw_leaf}" | xargs)"
    [[ -z "${leaf}" ]] && continue
    for raw_allowed in "${allowed_values[@]}"; do
      allowed="$(echo "${raw_allowed}" | xargs)"
      if [[ "${leaf}" == "${allowed}" ]]; then
        filtered+=("${leaf}")
        break
      fi
    done
  done

  local IFS=,
  printf '%s\n' "${filtered[*]}"
}

missing_ladder_leaves_csv() {
  local dim_root="$1"
  local leaves_csv="$2"
  local min_test_docs="${3:-0}"
  local missing=()
  local raw_leaf leaf leaf_dir

  if [[ "${RERUN_COMPLETED}" == "1" ]]; then
    printf '%s\n' "${leaves_csv}"
    return
  fi

  IFS=',' read -r -a leaf_values <<< "${leaves_csv}"
  for raw_leaf in "${leaf_values[@]}"; do
    leaf="$(echo "${raw_leaf}" | xargs)"
    [[ -z "${leaf}" ]] && continue
    leaf_dir="$(leaf_dir_name "${leaf}")"
    local history_path="${dim_root}/ladder/dspy/${leaf_dir}/iteration_history.json"
    if [[ ! -f "${history_path}" ]]; then
      missing+=("${leaf}")
    elif ! ladder_history_meets_min_test "${history_path}" "${min_test_docs}"; then
      missing+=("${leaf}")
    fi
  done

  local IFS=,
  printf '%s\n' "${missing[*]}"
}

tree_bundle_kind_matches_request() {
  local bundle="$1"
  local expected_kind="$2"
  local dimension="$3"
  local split_ids_dir="${4:-}"
  if [[ -z "${expected_kind}" ]]; then
    expected_kind="raw_manifesto_token_tree"
  fi
  if [[ ! -f "${bundle}/manifest.json" ]]; then
    return 1
  fi
  ./venv/bin/python - "${bundle}/manifest.json" "${expected_kind}" "${dimension}" "${split_ids_dir}" <<'PY'
import json
import sys
from pathlib import Path
from collections.abc import Mapping
from src.ctreepo.contracts import (
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    normalize_tree_bundle_manifest,
)

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
dimension = sys.argv[3]
split_ids_dir = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
config = manifest.get("config") or {}
try:
    source_kind = str(normalize_tree_bundle_manifest(config).get("source_kind") or "").strip()
except Exception:
    source_kind = ""
if source_kind == SOURCE_KIND_RAW_INPUT:
    kind = "raw_manifesto_token_tree"
elif source_kind == SOURCE_KIND_EXTERNAL_STATE:
    kind = "external_summary_token_tree"
else:
    kind = str(config.get("tree_bundle_kind") or "").strip()
if not kind:
    legacy = str(config.get("tree_text_source") or "").strip()
    if legacy == "existing_summary":
        kind = "external_summary_token_tree"
    elif legacy == "aligned_text":
        kind = "raw_manifesto_token_tree"
actual_dimension = str(config.get("dimension") or manifest.get("dimension") or "").strip()
def split_digest(payload):
    if not isinstance(payload, Mapping):
        return ""
    stack = [payload]
    seen = set()
    while stack:
        candidate = stack.pop(0)
        if not isinstance(candidate, Mapping):
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        value = candidate.get("split_manifest_digest")
        if value:
            return str(value)
        for key in ("config", "metadata", "tree_bundle_manifest"):
            child = candidate.get(key)
            if isinstance(child, Mapping):
                stack.append(child)
    return ""

expected_digest = ""
if split_ids_dir is not None:
    summary_path = split_ids_dir / "coverage_split_summary.json"
    if not summary_path.exists():
        raise SystemExit(1)
    expected_digest = str(json.loads(summary_path.read_text(encoding="utf-8")).get("split_manifest_digest") or "")
actual_digest = split_digest(manifest)
split_ok = not expected_digest or actual_digest == expected_digest
ok = kind == expected and (not actual_dimension or actual_dimension == dimension) and split_ok
raise SystemExit(0 if ok else 1)
PY
}

jsonl_row_count() {
  local path="$1"
  ./venv/bin/python - "${path}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    print(sum(1 for line in handle if line.strip()))
PY
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${name} must be a non-negative integer, got '${value}'" >&2
    exit 2
  fi
}

ladder_history_meets_min_test() {
  local history_path="$1"
  local min_test_docs="${2:-0}"
  if [[ -z "${min_test_docs}" || "${min_test_docs}" == "0" ]]; then
    return 0
  fi
  ./venv/bin/python - "${history_path}" "${min_test_docs}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_test = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
iterations = payload.get("iterations") or []
if not iterations:
    raise SystemExit(1)
best_n = 0
for row in iterations:
    metrics = ((row.get("split_metrics") or {}).get("test") or {})
    try:
        best_n = max(best_n, int(metrics.get("n") or 0))
    except (TypeError, ValueError):
        pass
raise SystemExit(0 if best_n >= min_test else 1)
PY
}

resolve_split_plan() {
  local dimension="$1"
  local source_results="$2"
  local default_train="${TRAIN_N:-140}"
  local default_val="${VAL_N:-30}"
  local default_test="${TEST_N:-48}"
  local min_test="${MIN_TEST_DOCS:-0}"

  if [[ -z "${BALANCED_TEST_N}" ]]; then
    printf '%s\t%s\t%s\t%s\n' "${default_train}" "${default_val}" "${default_test}" "${min_test}"
    return
  fi

  require_nonnegative_int "BALANCED_TEST_N" "${BALANCED_TEST_N}"
  require_nonnegative_int "BALANCED_VAL_N" "${BALANCED_VAL_N}"
  require_nonnegative_int "BALANCED_MIN_TEST_N" "${BALANCED_MIN_TEST_N}"
  require_nonnegative_int "BALANCED_MIN_TRAIN_N" "${BALANCED_MIN_TRAIN_N}"

  local row_count train_n
  row_count="$(jsonl_row_count "${source_results}")"
  require_nonnegative_int "row count for ${source_results}" "${row_count}"
  train_n=$((row_count - BALANCED_VAL_N - BALANCED_TEST_N))
  if (( train_n < BALANCED_MIN_TRAIN_N )); then
    cat >&2 <<EOF
ERROR: balanced split for ${dimension} is infeasible.
source_results=${source_results}
rows=${row_count} requested train>=${BALANCED_MIN_TRAIN_N}, val=${BALANCED_VAL_N}, test=${BALANCED_TEST_N}
EOF
    exit 2
  fi
  printf '%s\t%s\t%s\t%s\n' "${train_n}" "${BALANCED_VAL_N}" "${BALANCED_TEST_N}" "${BALANCED_MIN_TEST_N}"
}

if [[ -n "${WAIT_FOR_JOB_ROOTS}" ]]; then
  for wait_root in ${WAIT_FOR_JOB_ROOTS}; do
    wait_for_job_root "${wait_root}"
  done
fi

if [[ -n "${PRESTOP_SERVER_JOB_ROOT}" ]]; then
  stop_server_job_root "${PRESTOP_SERVER_JOB_ROOT}"
fi

echo "=== $(date -u) :: scalar dimensions=${DIMENSIONS} root=${ROOT} ==="
echo "=== $(date -u) :: leaf/context groups=${LEAF_CONTEXT_GROUPS} ==="
echo "=== $(date -u) :: split_ids_dir=${SPLIT_IDS_DIR:-none} source_results_root=${SOURCE_RESULTS_ROOT} ==="
echo "=== $(date -u) :: objective root_label_sources=${ROOT_LABEL_SOURCES} root_label_target=${ROOT_LABEL_TARGET} local_law_weight=${LOCAL_LAW_WEIGHT} node_weight_normalization=${NODE_WEIGHT_NORMALIZATION} expert_target_scale_override=${EXPERT_TARGET_SCALE_OVERRIDE:-per-dimension-default} ==="

idx="${GROUP_INDEX_OFFSET}"
group_ordinal=0
for group in ${LEAF_CONTEXT_GROUPS}; do
  group_ordinal=$((group_ordinal + 1))
  idx=$((idx + 1))
  IFS=':' read -r leaves context_len gpu_mem max_num_seqs dspy_batch_max_concurrent <<< "${group}"
  if [[ -z "${leaves:-}" || -z "${context_len:-}" || -z "${gpu_mem:-}" || -z "${max_num_seqs:-}" ]]; then
    echo "ERROR: bad LEAF_CONTEXT_GROUPS entry '${group}'" >&2
    exit 2
  fi
  dspy_batch_max_concurrent="${dspy_batch_max_concurrent:-${max_num_seqs}}"

  if [[ "${group_ordinal}" == "1" && -n "${REUSE_FIRST_SERVER_JOB_ROOT}" ]]; then
    current_server_job_root="${REUSE_FIRST_SERVER_JOB_ROOT}"
    echo "=== $(date -u) :: reusing first group server ${current_server_job_root} leaves=${leaves} context=${context_len} ==="
    wait_for_server "${PORT}" "${SERVER_START_TIMEOUT_SECONDS}"
  elif [[ -n "${current_server_job_root}" ]]; then
    echo "=== $(date -u) :: stopping prior group server ${current_server_job_root} ==="
    stop_server_job_root "${current_server_job_root}"
    current_server_job_root=""
    start_server_for_group "${idx}" "${context_len}" "${gpu_mem}" "${max_num_seqs}"
  else
    start_server_for_group "${idx}" "${context_len}" "${gpu_mem}" "${max_num_seqs}"
  fi

  for dimension in ${DIMENSIONS}; do
    dim_root="$(dimension_root_for "${dimension}")"
    tree_bundle="${dim_root}/tree_bundle"
    source_results="${SOURCE_RESULTS_ROOT}/${dimension}/per_manifesto.jsonl"
    dimension_expert_target_scale="$(expert_target_scale_for "${dimension}")"
    filtered_leaves="$(filter_leaves_csv "${leaves}" "$(dimension_leaf_filter_for "${dimension}")")"

    if [[ -z "${filtered_leaves}" ]]; then
      echo "=== $(date -u) :: ${dimension}: skipping leaves=${leaves}; outside dimension leaf filter ==="
      continue
    fi

    if [[ ! -f "${source_results}" ]]; then
      echo "ERROR: missing source results for ${dimension}: ${source_results}" >&2
      exit 2
    fi
    IFS=$'\t' read -r dimension_train_n dimension_val_n dimension_test_n dimension_min_test_docs \
      <<< "$(resolve_split_plan "${dimension}" "${source_results}")"
    run_leaves="$(missing_ladder_leaves_csv "${dim_root}" "${filtered_leaves}" "${dimension_min_test_docs}")"
    requested_tree_bundle_kind="${TREE_BUNDLE_KIND:-}"
    if [[ -z "${requested_tree_bundle_kind}" ]]; then
      case "${SOURCE_KIND:-}" in
        raw_input) requested_tree_bundle_kind="raw_manifesto_token_tree" ;;
        external_state) requested_tree_bundle_kind="external_summary_token_tree" ;;
        "")
          case "${TREE_TEXT_SOURCE:-}" in
            existing_summary) requested_tree_bundle_kind="external_summary_token_tree" ;;
            aligned_text|"") requested_tree_bundle_kind="raw_manifesto_token_tree" ;;
            *)
              echo "ERROR: unsupported legacy TREE_TEXT_SOURCE=${TREE_TEXT_SOURCE}" >&2
              exit 2
              ;;
          esac
          ;;
        *)
          echo "ERROR: unsupported SOURCE_KIND=${SOURCE_KIND}; expected raw_input or external_state" >&2
          exit 2
          ;;
      esac
    fi
    if [[ -z "${TREE_BUNDLE_KIND:-}" ]]; then
      case "${requested_tree_bundle_kind}" in
        raw_manifesto_token_tree) requested_source_kind="raw_input" ;;
        external_summary_token_tree) requested_source_kind="external_state" ;;
        *) requested_source_kind="${SOURCE_KIND:-}" ;;
      esac
    else
      requested_source_kind="${SOURCE_KIND:-}"
    fi
    if [[ -z "${requested_source_kind}" ]]; then
      case "${TREE_TEXT_SOURCE:-}" in
        existing_summary) requested_tree_bundle_kind="external_summary_token_tree" ;;
        aligned_text|"") requested_tree_bundle_kind="raw_manifesto_token_tree" ;;
      esac
      case "${requested_tree_bundle_kind}" in
        raw_manifesto_token_tree) requested_source_kind="raw_input" ;;
        external_summary_token_tree) requested_source_kind="external_state" ;;
      esac
    fi
    if [[ -z "${run_leaves}" ]] && ! tree_bundle_kind_matches_request "${tree_bundle}" "${requested_tree_bundle_kind}" "${dimension}" "${SPLIT_IDS_DIR}"; then
      echo "=== $(date -u) :: ${dimension}: existing ladder rows found, but tree bundle ${tree_bundle} is missing/incompatible with kind=${requested_tree_bundle_kind}; rerunning leaves=${filtered_leaves} ==="
      run_leaves="${filtered_leaves}"
    fi

    if [[ -z "${run_leaves}" ]]; then
      echo "=== $(date -u) :: ${dimension}: skipping leaves=${filtered_leaves}; ladder rows already complete with test_n>=${dimension_min_test_docs} ==="
      continue
    fi

    mkdir -p "${dim_root}"
    echo "=== $(date -u) :: ${dimension}: running leaves=${run_leaves} split train=${dimension_train_n} val=${dimension_val_n} test=${dimension_test_n} min_test_docs=${dimension_min_test_docs} context=${context_len} dspy_batch_max_concurrent=${dspy_batch_max_concurrent} expert_target_scale=${dimension_expert_target_scale} ==="
    env \
      DIMENSION="${dimension}" \
      TREE_BUNDLE="${tree_bundle}" \
      LEAF_SIZE_TOKENS="${run_leaves}" \
      LM_CONTEXT_TOKENS="${context_len}" \
      DSPY_BATCH_MAX_CONCURRENT="${dspy_batch_max_concurrent}" \
      MAX_ITERATIONS="${MAX_ITERATIONS}" \
      FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE}" \
      INITIAL_F_DEGREE="${INITIAL_F_DEGREE}" \
      INITIAL_G_DEGREE="${INITIAL_G_DEGREE}" \
      STAGE_NAMING="${STAGE_NAMING}" \
      DSPY_OPTIMIZER="${DSPY_OPTIMIZER}" \
      DSPY_BUDGET="${DSPY_BUDGET}" \
      DSPY_NUM_THREADS="${DSPY_NUM_THREADS}" \
      DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS}" \
      DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT}" \
      DSPY_BATCH_SIZE="${DSPY_BATCH_SIZE}" \
      DSPY_BATCH_TIMEOUT="${DSPY_BATCH_TIMEOUT}" \
      DSPY_BATCH_REQUEST_TIMEOUT="${DSPY_BATCH_REQUEST_TIMEOUT}" \
      DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT="${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT}" \
      DSPY_BATCH_ROUTING_POLICY="${DSPY_BATCH_ROUTING_POLICY}" \
      DSPY_F_INIT_PATH="${DSPY_F_INIT_PATH}" \
      DSPY_F_INIT_MODE="${DSPY_F_INIT_MODE}" \
      DSPY_MAX_TRAIN_RECORDS="${DSPY_MAX_TRAIN_RECORDS}" \
      DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS="${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS}" \
      DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS}" \
      SOURCE_RESULTS="${source_results}" \
      SPLIT_SOURCE="${SPLIT_SOURCE:-results-order}" \
      SPLIT_IDS_DIR="${SPLIT_IDS_DIR}" \
      SOURCE_KIND="${requested_source_kind:-raw_input}" \
      TREE_BUNDLE_KIND="${TREE_BUNDLE_KIND:-}" \
      TREE_STATE_SOURCE="${TREE_STATE_SOURCE:-}" \
      EXTERNAL_STATE_PRODUCER="${EXTERNAL_STATE_PRODUCER:-}" \
      ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE="${ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE:-0}" \
      TEACHER_SUMMARY_MODE="${TEACHER_SUMMARY_MODE:-teacher}" \
      TEACHER_SUMMARY_TEMPERATURE="${TEACHER_SUMMARY_TEMPERATURE:-0.0}" \
      TEACHER_IDEMPOTENCE_MODE="${TEACHER_IDEMPOTENCE_MODE:-off}" \
      TEACHER_SCORE_INPUT="${TEACHER_SCORE_INPUT:-teacher_summary}" \
      TEACHER_MISSING_SCORE_POLICY="${TEACHER_MISSING_SCORE_POLICY:-neutral}" \
      TEACHER_TIMEOUT_SECONDS="${TEACHER_TIMEOUT_SECONDS}" \
      SCORER_TIMEOUT_SECONDS="${SCORER_TIMEOUT_SECONDS}" \
      EXPERT_TARGET_SCALE="${dimension_expert_target_scale}" \
      SCORING_CONTEXT_SOURCE="${SCORING_CONTEXT_SOURCE}" \
      ROOT_LABEL_SOURCES="${ROOT_LABEL_SOURCES}" \
      ROOT_LABEL_TARGET="${ROOT_LABEL_TARGET}" \
      LOCAL_LAW_WEIGHT="${LOCAL_LAW_WEIGHT}" \
      NODE_WEIGHT_NORMALIZATION="${NODE_WEIGHT_NORMALIZATION}" \
      SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-0}" \
      RESUMMARY_MAX_TOKENS="${RESUMMARY_MAX_TOKENS:-0}" \
      SCORE_MAX_CHARS="${SCORE_MAX_CHARS:-24000}" \
      NODE_SUMMARY_MAX_CHARS="${NODE_SUMMARY_MAX_CHARS:-32000}" \
      RESUMMARY_MAX_CHARS="${RESUMMARY_MAX_CHARS:-24000}" \
      TRAIN_N="${dimension_train_n}" \
      VAL_N="${dimension_val_n}" \
      TEST_N="${dimension_test_n}" \
      MIN_TEST_DOCS="${dimension_min_test_docs}" \
      TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-32}" \
      TEACHER_LM_CONCURRENCY="${TEACHER_LM_CONCURRENCY:-16}" \
      PLOT_LADDER_GRID=0 \
      bash scripts/run_benoit_supervised_dspy_ladder.sh "${dim_root}" \
      2>&1 | tee "${ROOT}/ladder_group${idx}_${dimension}.log"
  done
done

if [[ "${KEEP_LAST_SERVER}" != "1" && -n "${current_server_job_root}" ]]; then
  echo "=== $(date -u) :: stopping final group server ${current_server_job_root} ==="
  stop_server_job_root "${current_server_job_root}"
  current_server_job_root=""
fi

if [[ "${PLOT_LADDER_GRID}" == "1" ]]; then
  for dimension in ${DIMENSIONS}; do
    dim_root="${ROOT}/${dimension}"
    plot_dir="${dim_root}/plots"
    if [[ ! -d "${dim_root}/ladder" ]]; then
      continue
    fi
    echo "=== $(date -u) :: plotting ${dimension} ladder -> ${plot_dir} ==="
    ./venv/bin/python scripts/plot_manifesto_fg_ladder_grid.py \
      --input-root "${dim_root}" \
      --figure-title "Manifesto ${dimension} f/g ladder" \
      --figure-subtitle "Single-dimension optimization with fresh scalar ${dimension} teacher traces; grouped server/batching launcher." \
      --output-dir "${plot_dir}" \
      2>&1 | tee "${ROOT}/plot_${dimension}.log" \
      || echo "warning: ${dimension} ladder grid plotting failed" >&2
  done
fi

if [[ "${PLOT_PREDICTION_DISTS}" == "1" ]]; then
  prediction_plot_args=(
    --source-root "${SOURCE_RESULTS_ROOT}"
    --output-dir "${ROOT}/plots_prediction_distributions"
  )
  for dimension in ${DIMENSIONS}; do
    prediction_plot_args+=(--ladder-root "${dimension}=${ROOT}/${dimension}")
  done
  echo "=== $(date -u) :: plotting prediction distributions -> ${ROOT}/plots_prediction_distributions ==="
  ./venv/bin/python scripts/plot_manifesto_prediction_distributions.py \
    "${prediction_plot_args[@]}" \
    2>&1 | tee "${ROOT}/plot_prediction_distributions.log" \
    || echo "warning: prediction distribution plotting failed" >&2
fi

echo "=== $(date -u) :: done ==="
