#!/usr/bin/env bash
# Build or reuse a saved LabeledTree bundle for one Manifesto dimension, then
# run a DSPy f/g alternating ladder over that bundle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/manifesto_ladder_runtime.sh"

DIMENSION="${DIMENSION:-economic}"
ROOT="${1:-outputs/manifesto_fg_alternating/${DIMENSION}_benoit_raw_tree_bundle_dspy_medium_$(date +%Y%m%d_%H%M%S)}"
if [[ -n "${TREE_BUNDLE+x}" && -n "${TEACHER_DIR+x}" && "${TREE_BUNDLE}" != "${TEACHER_DIR}" ]]; then
  echo "ERROR: TREE_BUNDLE and legacy TEACHER_DIR disagree: TREE_BUNDLE=${TREE_BUNDLE} TEACHER_DIR=${TEACHER_DIR}" >&2
  exit 2
fi
TREE_BUNDLE="${TREE_BUNDLE:-${TEACHER_DIR:-${ROOT}/tree_bundle}}"
TEACHER_DIR="${TREE_BUNDLE}"  # legacy alias used by older helper names below
LADDER_DIR="${ROOT}/ladder"
TEACHER_LOG="${ROOT}/teacher.log"
LADDER_LOG="${ROOT}/ladder.log"
DSPY_NUM_THREADS="${DSPY_NUM_THREADS:-128}"
NOFILE_LIMIT="${NOFILE_LIMIT:-65535}"
LEAF_SIZE_TOKENS="${LEAF_SIZE_TOKENS:-256,512}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4}"
FIRST_TRAIN_SIDE="${FIRST_TRAIN_SIDE:-f}"
INITIAL_F_DEGREE="${INITIAL_F_DEGREE:-1}"
INITIAL_G_DEGREE="${INITIAL_G_DEGREE:-1}"
STAGE_NAMING="${STAGE_NAMING:-legacy}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-0}"
RESUMMARY_MAX_TOKENS="${RESUMMARY_MAX_TOKENS:-0}"
DSPY_MAX_TOKENS="${DSPY_MAX_TOKENS:-0}"
DSPY_OPTIMIZER="${DSPY_OPTIMIZER:-mipro}"
DSPY_BUDGET="${DSPY_BUDGET:-light}"
DSPY_LM_TRANSPORT="${DSPY_LM_TRANSPORT:-batch}"
DSPY_BATCH_MAX_CONCURRENT="${DSPY_BATCH_MAX_CONCURRENT:-}"
DSPY_BATCH_SIZE="${DSPY_BATCH_SIZE:-64}"
DSPY_BATCH_TIMEOUT="${DSPY_BATCH_TIMEOUT:-0.02}"
DSPY_BATCH_REQUEST_TIMEOUT="${DSPY_BATCH_REQUEST_TIMEOUT:-300}"
DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT="${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT:-}"
DSPY_BATCH_ROUTING_POLICY="${DSPY_BATCH_ROUTING_POLICY:-affinity_load_aware}"
DSPY_F_INIT_PATH="${DSPY_F_INIT_PATH:-}"
DSPY_F_INIT_MODE="${DSPY_F_INIT_MODE:-pretuned_scorer}"
DSPY_MAX_TRAIN_RECORDS="${DSPY_MAX_TRAIN_RECORDS:-}"
DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS="${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS:-3}"
LM_CONTEXT_TOKENS="${LM_CONTEXT_TOKENS:-}"
SCORE_MAX_CHARS="${SCORE_MAX_CHARS:-24000}"
NODE_SUMMARY_MAX_CHARS="${NODE_SUMMARY_MAX_CHARS:-32000}"
RESUMMARY_MAX_CHARS="${RESUMMARY_MAX_CHARS:-24000}"
DSPY_PROMPT_OVERHEAD_TOKENS="${DSPY_PROMPT_OVERHEAD_TOKENS:-1500}"
SOURCE_RESULTS="${SOURCE_RESULTS:-outputs/overnight_benoit/full_pipeline/${DIMENSION}/per_manifesto.jsonl}"
SPLIT_SOURCE="${SPLIT_SOURCE:-results-order}"
SPLIT_IDS_DIR="${SPLIT_IDS_DIR:-}"
LEGACY_TREE_TEXT_SOURCE="${TREE_TEXT_SOURCE:-}"
TREE_BUNDLE_KIND="${TREE_BUNDLE_KIND:-}"
SOURCE_KIND="${SOURCE_KIND:-}"
case "${LEGACY_TREE_TEXT_SOURCE}" in
  "")
    ;;
  aligned_text)
    if [[ -n "${TREE_BUNDLE_KIND}" && "${TREE_BUNDLE_KIND}" != "raw_manifesto_token_tree" ]]; then
      echo "ERROR: TREE_TEXT_SOURCE=aligned_text conflicts with TREE_BUNDLE_KIND=${TREE_BUNDLE_KIND}" >&2
      exit 2
    fi
    TREE_BUNDLE_KIND="raw_manifesto_token_tree"
    ;;
  existing_summary)
    if [[ -n "${TREE_BUNDLE_KIND}" && "${TREE_BUNDLE_KIND}" != "external_summary_token_tree" ]]; then
      echo "ERROR: TREE_TEXT_SOURCE=existing_summary conflicts with TREE_BUNDLE_KIND=${TREE_BUNDLE_KIND}" >&2
      exit 2
    fi
    TREE_BUNDLE_KIND="external_summary_token_tree"
    ;;
  *)
    echo "ERROR: unsupported legacy TREE_TEXT_SOURCE=${LEGACY_TREE_TEXT_SOURCE}; use TREE_BUNDLE_KIND instead." >&2
    exit 2
    ;;
esac
if [[ -z "${TREE_BUNDLE_KIND}" ]]; then
  case "${SOURCE_KIND}" in
    "") TREE_BUNDLE_KIND="raw_manifesto_token_tree" ;;
    raw_input) TREE_BUNDLE_KIND="raw_manifesto_token_tree" ;;
    external_state) TREE_BUNDLE_KIND="external_summary_token_tree" ;;
    *) echo "ERROR: unsupported SOURCE_KIND=${SOURCE_KIND}; expected raw_input or external_state" >&2; exit 2 ;;
  esac
fi
TREE_BUNDLE_KIND="${TREE_BUNDLE_KIND:-raw_manifesto_token_tree}"
case "${TREE_BUNDLE_KIND}" in
  raw_manifesto_token_tree)
    if [[ -n "${SOURCE_KIND}" && "${SOURCE_KIND}" != "raw_input" ]]; then
      echo "ERROR: TREE_BUNDLE_KIND=${TREE_BUNDLE_KIND} conflicts with SOURCE_KIND=${SOURCE_KIND}" >&2
      exit 2
    fi
    SOURCE_KIND="raw_input"
    TREE_TEXT_SOURCE="aligned_text"
    TREE_STATE_SOURCE="${TREE_STATE_SOURCE:-raw_input}"
    ;;
  external_summary_token_tree)
    if [[ -n "${SOURCE_KIND}" && "${SOURCE_KIND}" != "external_state" ]]; then
      echo "ERROR: TREE_BUNDLE_KIND=${TREE_BUNDLE_KIND} conflicts with SOURCE_KIND=${SOURCE_KIND}" >&2
      exit 2
    fi
    SOURCE_KIND="external_state"
    TREE_TEXT_SOURCE="existing_summary"
    TREE_STATE_SOURCE="${TREE_STATE_SOURCE:-external_state}"
    EXTERNAL_STATE_PRODUCER="${EXTERNAL_STATE_PRODUCER:-g_benoit}"
    ;;
  *)
    echo "ERROR: unsupported TREE_BUNDLE_KIND=${TREE_BUNDLE_KIND}" >&2
    exit 2
    ;;
esac
EXTERNAL_STATE_PRODUCER="${EXTERNAL_STATE_PRODUCER:-}"
if [[ "${TREE_BUNDLE_KIND}" == "raw_manifesto_token_tree" && -n "${EXTERNAL_STATE_PRODUCER}" ]]; then
  echo "ERROR: EXTERNAL_STATE_PRODUCER is only valid for external_summary_token_tree bundles" >&2
  exit 2
fi
ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE="${ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE:-${ALLOW_INCOMPATIBLE_TEACHER_REUSE:-0}}"
TEACHER_SUMMARY_MODE="${TEACHER_SUMMARY_MODE:-teacher}"
TEACHER_SUMMARY_TEMPERATURE="${TEACHER_SUMMARY_TEMPERATURE:-0.0}"
TEACHER_IDEMPOTENCE_MODE="${TEACHER_IDEMPOTENCE_MODE:-off}"
TEACHER_SCORE_INPUT="${TEACHER_SCORE_INPUT:-teacher_summary}"
TEACHER_MISSING_SCORE_POLICY="${TEACHER_MISSING_SCORE_POLICY:-neutral}"
TEACHER_TIMEOUT_SECONDS="${TEACHER_TIMEOUT_SECONDS:-600}"
SCORER_TIMEOUT_SECONDS="${SCORER_TIMEOUT_SECONDS:-600}"
EXPERT_TARGET_SCALE="${EXPERT_TARGET_SCALE:-}"
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
TRAIN_N="${TRAIN_N:-140}"
VAL_N="${VAL_N:-30}"
TEST_N="${TEST_N:-48}"
MIN_TEST_DOCS="${MIN_TEST_DOCS:-0}"
TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-32}"
TEACHER_LM_CONCURRENCY="${TEACHER_LM_CONCURRENCY:-16}"

LM_CONTEXT_TOKENS="$(manifesto_resolve_lm_context_tokens "${LEAF_SIZE_TOKENS}" "${LM_CONTEXT_TOKENS}")"
DSPY_BATCH_MAX_CONCURRENT="$(manifesto_resolve_dspy_batch_max_concurrent "${LM_CONTEXT_TOKENS}" "${DSPY_BATCH_MAX_CONCURRENT}")"
if [[ -z "${EXPERT_TARGET_SCALE}" ]]; then
  EXPERT_TARGET_SCALE="normalized_1_7"
fi

ulimit -n "${NOFILE_LIMIT}" 2>/dev/null || true

mkdir -p "${ROOT}"

teacher_leaf_meets_min_test() {
  local leaf_dir="$1"
  local min_test_docs="${2:-0}"
  if [[ -z "${min_test_docs}" || "${min_test_docs}" == "0" ]]; then
    return 0
  fi
  if [[ ! -f "${leaf_dir}/summary.json" ]]; then
    return 1
  fi
  ./venv/bin/python - "${leaf_dir}/summary.json" "${min_test_docs}" <<'PY'
import json
import sys
from pathlib import Path
from src.ctreepo.contracts import (
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    normalize_tree_bundle_manifest,
)

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
min_test = int(sys.argv[2])
tree_counts = summary.get("tree_counts") or {}
try:
    test_count = int(tree_counts.get("test") or 0)
except (TypeError, ValueError):
    test_count = 0
raise SystemExit(0 if test_count >= min_test else 1)
PY
}

assert_tree_bundle_compatible() {
  local bundle="$1"
  local expected_kind="$2"
  local dimension="$3"
  local leaves_csv="$4"
  local min_test_docs="${5:-0}"
  local expert_target_scale="$6"
  local external_state_producer="${7:-}"
  local allow_unsafe="${8:-0}"
  local split_ids_dir="${9:-}"
  local manifest_path="${bundle}/manifest.json"
  if [[ ! -f "${manifest_path}" ]]; then
    if [[ "${allow_unsafe}" == "1" ]]; then
      echo "WARNING: ${bundle} has no manifest.json; ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE=1 bypasses compatibility checks" >&2
      return
    fi
    echo "ERROR: TREE_BUNDLE=${bundle} is missing manifest.json; refusing to reuse without ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE=1" >&2
    exit 2
  fi
  ./venv/bin/python - \
    "${bundle}" \
    "${expected_kind}" \
    "${dimension}" \
    "${leaves_csv}" \
    "${min_test_docs}" \
    "${expert_target_scale}" \
    "${external_state_producer}" \
    "${allow_unsafe}" \
    "${split_ids_dir}" <<'PY'
import json
import sys
from pathlib import Path
from collections.abc import Mapping

from src.ctreepo.contracts import (
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    normalize_tree_bundle_manifest,
)

bundle = Path(sys.argv[1])
expected_kind = sys.argv[2]
dimension = sys.argv[3]
leaves = [int(part.strip()) for part in sys.argv[4].split(",") if part.strip()]
min_test = int(sys.argv[5] or 0)
expert_target_scale = sys.argv[6]
external_state_producer = sys.argv[7]
allow_unsafe = sys.argv[8] == "1"
split_ids_dir = Path(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] else None

manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
config = manifest.get("config") if isinstance(manifest, dict) else {}
if not isinstance(config, dict):
    config = {}

def split_digest_from_payload(payload):
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

expected_split_digest = ""
if split_ids_dir is not None:
    summary_path = split_ids_dir / "coverage_split_summary.json"
    if summary_path.exists():
        split_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_split_digest = str(split_payload.get("split_manifest_digest") or "")
    else:
        errors = [f"SPLIT_IDS_DIR={split_ids_dir} is missing coverage_split_summary.json"]
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        if not allow_unsafe:
            raise SystemExit(2)
        expected_split_digest = ""

def infer_kind(payload):
    try:
        source_kind = str(normalize_tree_bundle_manifest(payload).get("source_kind") or "").strip()
    except Exception:
        source_kind = ""
    if source_kind == SOURCE_KIND_RAW_INPUT:
        return "raw_manifesto_token_tree"
    if source_kind == SOURCE_KIND_EXTERNAL_STATE:
        return "external_summary_token_tree"
    kind = str(payload.get("tree_bundle_kind") or "").strip()
    if kind:
        return kind
    legacy = str(payload.get("tree_text_source") or "").strip()
    if legacy == "existing_summary":
        return "external_summary_token_tree"
    if legacy == "aligned_text":
        return "raw_manifesto_token_tree"
    return ""

actual_kind = infer_kind(config) or infer_kind(manifest if isinstance(manifest, dict) else {})
errors = []
warnings = []
actual_split_digest = split_digest_from_payload(manifest)
if expected_split_digest:
    if actual_split_digest and actual_split_digest != expected_split_digest:
        errors.append(
            "split_manifest_digest mismatch: "
            f"expected {expected_split_digest}, found {actual_split_digest}"
        )
    elif not actual_split_digest:
        errors.append("split_manifest_digest missing from reused tree bundle manifest")
if actual_kind and actual_kind != expected_kind:
    errors.append(
        f"tree_bundle_kind mismatch: expected {expected_kind}, found {actual_kind}"
    )
elif not actual_kind:
    errors.append("tree_bundle_kind missing and could not be inferred")

actual_dimension = str(config.get("dimension") or manifest.get("dimension") or "").strip()
if actual_dimension and actual_dimension != dimension:
    errors.append(f"dimension mismatch: expected {dimension}, found {actual_dimension}")

actual_scale = str(config.get("expert_target_scale") or "").strip()
if actual_scale and actual_scale != expert_target_scale:
    errors.append(
        f"expert_target_scale mismatch: expected {expert_target_scale}, found {actual_scale}"
    )

raw_producer = config.get("external_state_producer")
actual_producer = "" if raw_producer is None else str(raw_producer).strip()
if expected_kind == "external_summary_token_tree":
    if external_state_producer and actual_producer and actual_producer != external_state_producer:
        errors.append(
            "external_state_producer mismatch: "
            f"expected {external_state_producer}, found {actual_producer}"
        )
else:
    if actual_producer:
        errors.append(
            f"raw_manifesto_token_tree bundle unexpectedly has external_state_producer={actual_producer}"
        )

manifest_leaves = set()
for raw in config.get("leaf_size_tokens") or []:
    try:
        manifest_leaves.add(int(raw))
    except (TypeError, ValueError):
        pass
if manifest_leaves:
    missing = [leaf for leaf in leaves if leaf not in manifest_leaves]
    if missing:
        errors.append(f"requested leaves absent from manifest leaf_size_tokens: {missing}")

for leaf in leaves:
    leaf_dir = bundle / f"leaf{leaf:04d}tok"
    summary_path = leaf_dir / "summary.json"
    trees_path = leaf_dir / "labeled_trees.jsonl"
    if not trees_path.exists():
        errors.append(f"missing labeled trees for leaf_size_tokens={leaf}: {trees_path}")
        continue
    tree_rows = sum(1 for line in trees_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if tree_rows <= 0:
        errors.append(f"leaf{leaf:04d}tok has zero labeled trees: {trees_path}")
    if not summary_path.exists():
        errors.append(f"missing summary for leaf_size_tokens={leaf}: {summary_path}")
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if expected_split_digest:
        leaf_split_digest = split_digest_from_payload(summary)
        if leaf_split_digest and leaf_split_digest != expected_split_digest:
            errors.append(
                f"leaf{leaf:04d}tok split_manifest_digest mismatch: "
                f"expected {expected_split_digest}, found {leaf_split_digest}"
            )
        elif not leaf_split_digest:
            errors.append(
                f"leaf{leaf:04d}tok summary missing split_manifest_digest"
            )
    summary_kind = infer_kind(summary)
    if summary_kind and summary_kind != expected_kind:
        errors.append(
            f"leaf{leaf:04d}tok summary kind mismatch: expected {expected_kind}, found {summary_kind}"
        )
    tree_counts = summary.get("tree_counts") or {}
    try:
        total_count = int(tree_counts.get("total") or 0)
    except (TypeError, ValueError):
        total_count = 0
    if total_count <= 0:
        errors.append(f"leaf{leaf:04d}tok summary reports zero total labeled trees")
    try:
        test_count = int(tree_counts.get("test") or 0)
    except (TypeError, ValueError):
        test_count = 0
    if min_test and test_count < min_test:
        errors.append(
            f"leaf{leaf:04d}tok test_count={test_count} below MIN_TEST_DOCS={min_test}"
        )
    if expected_kind == "raw_manifesto_token_tree":
        try:
            total_trees = int(tree_counts.get("total") or 0)
        except (TypeError, ValueError):
            total_trees = 0
        try:
            node_count = int(summary.get("node_count") or 0)
        except (TypeError, ValueError):
            node_count = 0
        leaf_stats = summary.get("leaf_count_stats") or {}
        try:
            mean_leaves = float(leaf_stats.get("mean") or 0.0)
        except (TypeError, ValueError):
            mean_leaves = 0.0
        if total_trees and node_count <= total_trees:
            warnings.append(
                f"leaf{leaf:04d}tok raw bundle has node_count={node_count} <= tree_count={total_trees}"
            )
        if total_trees and mean_leaves <= 1.0:
            warnings.append(
                f"leaf{leaf:04d}tok raw bundle mean_leaves={mean_leaves:.3f}; check that this is not an external-summary surface"
            )

for warning in warnings:
    print(f"WARNING: {warning}", file=sys.stderr)
if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not allow_unsafe:
        raise SystemExit(2)
    print(
        "WARNING: ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE=1 bypasses the errors above",
        file=sys.stderr,
    )
PY
}

assert_teacher_min_test() {
  local min_test_docs="${1:-0}"
  if [[ -z "${min_test_docs}" || "${min_test_docs}" == "0" ]]; then
    return
  fi
  local raw_leaf_size leaf_size leaf_dir summary_path
  IFS=',' read -r -a assert_leaf_values <<< "${LEAF_SIZE_TOKENS}"
  for raw_leaf_size in "${assert_leaf_values[@]}"; do
    leaf_size="$(echo "${raw_leaf_size}" | xargs)"
    [[ -z "${leaf_size}" ]] && continue
    leaf_dir="${TEACHER_DIR}/$(printf 'leaf%04dtok' "${leaf_size}")"
    summary_path="${leaf_dir}/summary.json"
    if ! teacher_leaf_meets_min_test "${leaf_dir}" "${min_test_docs}"; then
      local actual_test="missing"
      if [[ -f "${summary_path}" ]]; then
        actual_test="$(./venv/bin/python - "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(((payload.get("tree_counts") or {}).get("test") or 0))
PY
)"
      fi
      echo "ERROR: ${DIMENSION} teacher traces for leaf_size_tokens=${leaf_size} have test_count=${actual_test}, below MIN_TEST_DOCS=${min_test_docs}" >&2
      exit 2
    fi
  done
}

dspy_batch_args=(
  --dspy-lm-transport "${DSPY_LM_TRANSPORT}"
  --dspy-batch-max-concurrent "${DSPY_BATCH_MAX_CONCURRENT}"
  --dspy-batch-size "${DSPY_BATCH_SIZE}"
  --dspy-batch-timeout "${DSPY_BATCH_TIMEOUT}"
  --dspy-batch-request-timeout "${DSPY_BATCH_REQUEST_TIMEOUT}"
  --dspy-batch-routing-policy "${DSPY_BATCH_ROUTING_POLICY}"
)
if [[ -n "${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT}" ]]; then
  dspy_batch_args+=(--dspy-batch-await-response-timeout "${DSPY_BATCH_AWAIT_RESPONSE_TIMEOUT}")
fi

dspy_mipro_args=()
if [[ -n "${DSPY_MIPRO_NUM_CANDIDATES:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-num-candidates "${DSPY_MIPRO_NUM_CANDIDATES}")
fi
if [[ -n "${DSPY_MIPRO_NUM_TRIALS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-num-trials "${DSPY_MIPRO_NUM_TRIALS}")
fi
if [[ -n "${DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-max-bootstrapped-demos "${DSPY_MIPRO_MAX_BOOTSTRAPPED_DEMOS}")
fi
if [[ -n "${DSPY_MIPRO_MAX_LABELED_DEMOS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-max-labeled-demos "${DSPY_MIPRO_MAX_LABELED_DEMOS}")
fi
if [[ -n "${DSPY_MIPRO_MINIBATCH_SIZE:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-minibatch-size "${DSPY_MIPRO_MINIBATCH_SIZE}")
fi
if [[ -n "${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS:-}" ]]; then
  dspy_mipro_args+=(--dspy-mipro-minibatch-full-eval-steps "${DSPY_MIPRO_MINIBATCH_FULL_EVAL_STEPS}")
fi
if [[ -n "${DSPY_MAX_TRAIN_RECORDS:-}" ]]; then
  dspy_mipro_args+=(--dspy-max-train-records "${DSPY_MAX_TRAIN_RECORDS}")
fi

dspy_f_init_args=(--dspy-f-init-mode "${DSPY_F_INIT_MODE}")
if [[ -n "${DSPY_F_INIT_PATH}" ]]; then
  dspy_f_init_args+=(--dspy-f-init-path "${DSPY_F_INIT_PATH}")
fi

anchor_args=(
  --root-label-sources "${ROOT_LABEL_SOURCES}"
  --root-label-target "${ROOT_LABEL_TARGET}"
  --node-weight-normalization "${NODE_WEIGHT_NORMALIZATION}"
)
anchor_args+=(--local-law-weight "${LOCAL_LAW_WEIGHT}")

tree_bundle_build_args=(
  --source-kind "${SOURCE_KIND}"
  --tree-bundle-kind "${TREE_BUNDLE_KIND}"
  --tree-state-source "${TREE_STATE_SOURCE}"
)
if [[ -n "${EXTERNAL_STATE_PRODUCER}" ]]; then
  tree_bundle_build_args+=(--external-state-producer "${EXTERNAL_STATE_PRODUCER}")
fi
split_alignment_args=()
if [[ -n "${SPLIT_IDS_DIR}" ]]; then
  split_alignment_args+=(--alignment-run-dir "${SPLIT_IDS_DIR}")
fi

echo "=== $(date -u) :: runtime context=${LM_CONTEXT_TOKENS} dspy_batch_max_concurrent=${DSPY_BATCH_MAX_CONCURRENT} ==="
echo "=== $(date -u) :: tree_bundle=${TREE_BUNDLE} source_kind=${SOURCE_KIND} legacy_kind=${TREE_BUNDLE_KIND} tree_state_source=${TREE_STATE_SOURCE} external_state_producer=${EXTERNAL_STATE_PRODUCER:-none} split_ids_dir=${SPLIT_IDS_DIR:-none} ==="
echo "=== $(date -u) :: objective root_label_sources=${ROOT_LABEL_SOURCES} root_label_target=${ROOT_LABEL_TARGET} local_law_weight=${LOCAL_LAW_WEIGHT} node_weight_normalization=${NODE_WEIGHT_NORMALIZATION} expert_target_scale=${EXPERT_TARGET_SCALE} ==="
echo "=== $(date -u) :: ${DIMENSION}: preflighting DSPy f/g arity budgets ==="
./venv/bin/python scripts/run_alternating_ladder.py \
  --families dspy \
  --dimension "${DIMENSION}" \
  --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --first-train-side "${FIRST_TRAIN_SIDE}" \
  --initial-f-degree "${INITIAL_F_DEGREE}" \
  --initial-g-degree "${INITIAL_G_DEGREE}" \
  --stage-naming "${STAGE_NAMING}" \
  --tree-bundle "${TREE_BUNDLE}" \
  --dspy-optimizer "${DSPY_OPTIMIZER}" \
  --dspy-budget "${DSPY_BUDGET}" \
  --dspy-num-threads "${DSPY_NUM_THREADS}" \
  --dspy-api-base http://localhost:8010/v1 \
  --dspy-model openai/nvidia/Gemma-4-31B-IT-NVFP4 \
  --dspy-api-key EMPTY \
  --dspy-max-tokens "${DSPY_MAX_TOKENS}" \
  --dspy-lm-context-tokens "${LM_CONTEXT_TOKENS}" \
  --dspy-prompt-overhead-tokens "${DSPY_PROMPT_OVERHEAD_TOKENS}" \
  "${dspy_f_init_args[@]}" \
  "${dspy_batch_args[@]}" \
  "${dspy_mipro_args[@]}" \
  "${anchor_args[@]}" \
  --preflight-only \
  --output-dir "${ROOT}/preflight/ladder_budget" \
  2>&1 | tee "${ROOT}/preflight_ladder_budget.log"

leaf_traces_exist=1
IFS=',' read -r -a leaf_size_token_values <<< "${LEAF_SIZE_TOKENS}"
for raw_leaf_size in "${leaf_size_token_values[@]}"; do
  leaf_size="$(echo "${raw_leaf_size}" | xargs)"
  if [[ -z "${leaf_size}" ]]; then
    continue
  fi
  leaf_dir="$(printf 'leaf%04dtok' "${leaf_size}")"
  if [[ ! -f "${TEACHER_DIR}/${leaf_dir}/labeled_trees.jsonl" ]]; then
    leaf_traces_exist=0
    break
  elif ! teacher_leaf_meets_min_test "${TEACHER_DIR}/${leaf_dir}" "${MIN_TEST_DOCS}"; then
    leaf_traces_exist=0
    break
  fi
done

if [[ "${SKIP_TEACHER:-0}" == "1" ]] || [[ "${leaf_traces_exist}" == "1" ]]; then
  assert_tree_bundle_compatible \
    "${TREE_BUNDLE}" \
    "${TREE_BUNDLE_KIND}" \
    "${DIMENSION}" \
    "${LEAF_SIZE_TOKENS}" \
    "${MIN_TEST_DOCS}" \
    "${EXPERT_TARGET_SCALE}" \
    "${EXTERNAL_STATE_PRODUCER}" \
    "${ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE}" \
    "${SPLIT_IDS_DIR}"
  echo "=== $(date -u) :: reusing existing tree bundle in ${TREE_BUNDLE} ==="
else
  echo "=== $(date -u) :: ${DIMENSION}: building tree bundle source=${SOURCE_RESULTS} source_kind=${SOURCE_KIND} internal_tree_text_source=${TREE_TEXT_SOURCE} summary_mode=${TEACHER_SUMMARY_MODE} leaf_size_tokens=${LEAF_SIZE_TOKENS} ==="
  ./venv/bin/python scripts/run_manifesto_teacher_fg_leaf_grid.py \
    --dimension "${DIMENSION}" \
    --source-results "${SOURCE_RESULTS}" \
    --split-source "${SPLIT_SOURCE}" \
    "${split_alignment_args[@]}" \
    "${tree_bundle_build_args[@]}" \
    --summary-mode "${TEACHER_SUMMARY_MODE}" \
    --summary-temperature "${TEACHER_SUMMARY_TEMPERATURE}" \
    --summary-max-tokens "${SUMMARY_MAX_TOKENS}" \
    --idempotence-mode "${TEACHER_IDEMPOTENCE_MODE}" \
    --score-input "${TEACHER_SCORE_INPUT}" \
    --expert-target-scale "${EXPERT_TARGET_SCALE}" \
    --scoring-context-source "${SCORING_CONTEXT_SOURCE}" \
    --missing-score-policy "${TEACHER_MISSING_SCORE_POLICY}" \
    --resummary-max-tokens "${RESUMMARY_MAX_TOKENS}" \
    --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
    --train-n "${TRAIN_N}" \
    --val-n "${VAL_N}" \
    --test-n "${TEST_N}" \
    --min-test-docs "${MIN_TEST_DOCS}" \
    --teacher-base-url http://localhost:8010/v1 \
    --teacher-model nvidia/Gemma-4-31B-IT-NVFP4 \
    --teacher-api-key EMPTY \
    --scorer-base-url http://localhost:8010/v1 \
    --scorer-model nvidia/Gemma-4-31B-IT-NVFP4 \
    --scorer-api-key EMPTY \
    --teacher-timeout-seconds "${TEACHER_TIMEOUT_SECONDS}" \
    --scorer-timeout-seconds "${SCORER_TIMEOUT_SECONDS}" \
    --num-workers "${TEACHER_NUM_WORKERS}" \
    --lm-concurrency "${TEACHER_LM_CONCURRENCY}" \
    --score-max-chars "${SCORE_MAX_CHARS}" \
    --node-summary-max-chars "${NODE_SUMMARY_MAX_CHARS}" \
    --resummary-max-chars "${RESUMMARY_MAX_CHARS}" \
    --output-dir "${TREE_BUNDLE}" \
    2>&1 | tee "${TEACHER_LOG}"
fi

assert_tree_bundle_compatible \
  "${TREE_BUNDLE}" \
  "${TREE_BUNDLE_KIND}" \
  "${DIMENSION}" \
  "${LEAF_SIZE_TOKENS}" \
  "${MIN_TEST_DOCS}" \
  "${EXPERT_TARGET_SCALE}" \
  "${EXTERNAL_STATE_PRODUCER}" \
  "${ALLOW_INCOMPATIBLE_TREE_BUNDLE_REUSE}" \
  "${SPLIT_IDS_DIR}"
manifesto_audit_tree_bundle "${TREE_BUNDLE}" "${SOURCE_KIND}" "${EXPERT_TARGET_SCALE}"
assert_teacher_min_test "${MIN_TEST_DOCS}"

tree_bundle_ladder_args=(--dspy-g-init-mode raw_concat)
if [[ "${SOURCE_KIND}" == "external_state" ]]; then
  tree_bundle_ladder_args=(
    --allow-external-state-tree-bundle
    --dspy-g-init-mode teacher_passthrough
  )
fi
if [[ "${ALLOW_LEGACY_TREE_BUNDLE:-0}" == "1" ]]; then
  tree_bundle_ladder_args+=(--allow-legacy-tree-bundle)
fi

echo "=== $(date -u) :: dspy alternating ladder max_iterations=${MAX_ITERATIONS} first_train_side=${FIRST_TRAIN_SIDE} initial_f_degree=${INITIAL_F_DEGREE} initial_g_degree=${INITIAL_G_DEGREE} stage_naming=${STAGE_NAMING} (leaf_size_tokens=${LEAF_SIZE_TOKENS}, threads=${DSPY_NUM_THREADS}, nofile=$(ulimit -n)) ==="
./venv/bin/python scripts/run_alternating_ladder.py \
  --families dspy \
  --dimension "${DIMENSION}" \
  --leaf-size-tokens "${LEAF_SIZE_TOKENS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --first-train-side "${FIRST_TRAIN_SIDE}" \
  --initial-f-degree "${INITIAL_F_DEGREE}" \
  --initial-g-degree "${INITIAL_G_DEGREE}" \
  --stage-naming "${STAGE_NAMING}" \
  --tree-bundle "${TREE_BUNDLE}" \
  --dspy-optimizer "${DSPY_OPTIMIZER}" \
  --dspy-budget "${DSPY_BUDGET}" \
  --dspy-num-threads "${DSPY_NUM_THREADS}" \
  --dspy-api-base http://localhost:8010/v1 \
  --dspy-model openai/nvidia/Gemma-4-31B-IT-NVFP4 \
  --dspy-api-key EMPTY \
  --dspy-max-tokens "${DSPY_MAX_TOKENS}" \
  --dspy-lm-context-tokens "${LM_CONTEXT_TOKENS}" \
  --dspy-prompt-overhead-tokens "${DSPY_PROMPT_OVERHEAD_TOKENS}" \
  "${dspy_f_init_args[@]}" \
  "${tree_bundle_ladder_args[@]}" \
  "${dspy_batch_args[@]}" \
  "${dspy_mipro_args[@]}" \
  "${anchor_args[@]}" \
  --fail-on-row-error \
  --output-dir "${LADDER_DIR}" \
  2>&1 | tee "${LADDER_LOG}"

echo "=== $(date -u) :: done ==="
if [[ -f "${LADDER_DIR}/grid_summary.md" ]]; then
  cat "${LADDER_DIR}/grid_summary.md"
fi

if [[ "${PLOT_LADDER_GRID:-1}" == "1" ]]; then
  PLOT_DIR="${PLOT_DIR:-${ROOT}/plots}"
  plot_args=()
  if [[ -n "${PLOT_INPUT_ROOTS:-}" ]]; then
    read -r -a plot_input_roots <<< "${PLOT_INPUT_ROOTS}"
    for plot_root in "${plot_input_roots[@]}"; do
      plot_args+=(--input-root "${plot_root}")
    done
  else
    plot_args+=(--input-root "${ROOT}")
  fi
  if [[ -n "${PLOT_STAGES:-}" ]]; then
    plot_args+=(--stages "${PLOT_STAGES}")
  fi
  plot_args+=(--figure-title "${PLOT_FIGURE_TITLE:-Manifesto ${DIMENSION} f/g ladder}")
  plot_args+=(--figure-subtitle "${PLOT_FIGURE_SUBTITLE:-Single-dimension ${DIMENSION} optimization with fresh scalar teacher traces.}")
  if [[ -n "${PLOT_EXTERNAL_PEARSON_MIN:-}" ]]; then
    plot_args+=(--external-pearson-min "${PLOT_EXTERNAL_PEARSON_MIN}")
  fi
  if [[ -n "${PLOT_EXTERNAL_PEARSON_MAX:-}" ]]; then
    plot_args+=(--external-pearson-max "${PLOT_EXTERNAL_PEARSON_MAX}")
  fi
  if [[ -n "${PLOT_STAGE_LABELS:-}" ]]; then
    read -r -a plot_stage_labels <<< "${PLOT_STAGE_LABELS}"
    for plot_stage_label in "${plot_stage_labels[@]}"; do
      plot_args+=(--stage-label "${plot_stage_label}")
    done
  fi
  echo "=== $(date -u) :: plotting ladder grid -> ${PLOT_DIR} ==="
  ./venv/bin/python scripts/plot_manifesto_fg_ladder_grid.py \
    "${plot_args[@]}" \
    --output-dir "${PLOT_DIR}" \
    || echo "warning: ladder grid plotting failed" >&2
fi

if [[ -n "${PLOT_BUNDLES:-}" ]]; then
  bundle_args=()
  read -r -a plot_bundle_names <<< "${PLOT_BUNDLES}"
  for plot_bundle_name in "${plot_bundle_names[@]}"; do
    bundle_args+=(--bundle "${plot_bundle_name}")
  done
  if [[ -n "${PLOT_BUNDLE_RAW_RUN_ROOT:-}" ]]; then
    bundle_args+=(--raw-run-root "${PLOT_BUNDLE_RAW_RUN_ROOT}")
  else
    bundle_args+=(--raw-run-root "${ROOT}")
  fi
  echo "=== $(date -u) :: rendering plot bundles ${PLOT_BUNDLES} ==="
  ./venv/bin/python scripts/render_manifesto_fg_plot_bundles.py \
    "${bundle_args[@]}" \
    || echo "warning: bundle plotting failed" >&2
fi
