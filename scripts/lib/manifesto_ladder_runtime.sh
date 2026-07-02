#!/usr/bin/env bash
# Shared runtime sizing for Manifesto ladder runs.

manifesto_leaf_context_group_defaults() {
  printf '%s\n' \
    '256,512:8192:0.90:1024:1024 1024:12000:0.90:768:768 2048:20000:0.90:512:512 4096:51200:0.92:256:256 8192:65536:0.94:128:128'
}

manifesto_context_for_leaf_size() {
  local leaf_size="$1"
  if (( leaf_size <= 512 )); then
    printf '8192\n'
  elif (( leaf_size <= 1024 )); then
    printf '12000\n'
  elif (( leaf_size <= 2048 )); then
    printf '20000\n'
  elif (( leaf_size <= 4096 )); then
    printf '51200\n'
  else
    printf '65536\n'
  fi
}

manifesto_dspy_concurrency_for_context() {
  local context_len="$1"
  if (( context_len <= 8192 )); then
    printf '1024\n'
  elif (( context_len <= 12000 )); then
    printf '768\n'
  elif (( context_len <= 20000 )); then
    printf '512\n'
  elif (( context_len <= 51200 )); then
    printf '256\n'
  else
    printf '128\n'
  fi
}

manifesto_resolve_lm_context_tokens() {
  local leaf_size_tokens="$1"
  local explicit_context="${2:-}"
  if [[ -n "${explicit_context}" ]]; then
    printf '%s\n' "${explicit_context}"
    return 0
  fi

  local resolved_context=""
  local context_len=""
  local raw_leaf_size=""
  local leaf_size=""
  local seen_leaf=0
  IFS=',' read -r -a leaf_values <<< "${leaf_size_tokens}"
  for raw_leaf_size in "${leaf_values[@]}"; do
    leaf_size="$(echo "${raw_leaf_size}" | xargs)"
    [[ -z "${leaf_size}" ]] && continue
    if ! [[ "${leaf_size}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: bad LEAF_SIZE_TOKENS entry '${leaf_size}'" >&2
      return 2
    fi
    seen_leaf=1
    context_len="$(manifesto_context_for_leaf_size "${leaf_size}")"
    if [[ -z "${resolved_context}" ]]; then
      resolved_context="${context_len}"
    elif [[ "${resolved_context}" != "${context_len}" ]]; then
      cat >&2 <<EOF
ERROR: LEAF_SIZE_TOKENS=${leaf_size_tokens} spans multiple context/concurrency buckets.
Use scripts/run_benoit_joint_ladder_context_groups.sh, or set LM_CONTEXT_TOKENS
and DSPY_BATCH_MAX_CONCURRENT explicitly if you really want one server config.
EOF
      return 2
    fi
  done

  if [[ "${seen_leaf}" == "0" ]]; then
    echo "ERROR: LEAF_SIZE_TOKENS is empty" >&2
    return 2
  fi
  printf '%s\n' "${resolved_context}"
}

manifesto_resolve_dspy_batch_max_concurrent() {
  local context_len="$1"
  local explicit_concurrency="${2:-}"
  if [[ -n "${explicit_concurrency}" ]]; then
    printf '%s\n' "${explicit_concurrency}"
    return 0
  fi
  manifesto_dspy_concurrency_for_context "${context_len}"
}

manifesto_audit_tree_bundle() {
  local bundle="$1"
  local expected_source_kind="${2:-raw_input}"
  local expected_target_scale="${3:-}"
  local allow_legacy="${ALLOW_LEGACY_TREE_BUNDLE:-0}"
  local cmd=(
    ./venv/bin/python scripts/audit_tree_bundle_contracts.py
    "${bundle}"
    --require-tree-bundle
    --expected-domain manifesto_rile
    --expected-leaf-unit text_token
  )
  if [[ -n "${expected_source_kind}" && "${expected_source_kind}" != "any" ]]; then
    cmd+=(--expected-source-kind "${expected_source_kind}")
  fi
  if [[ -n "${expected_target_scale}" ]]; then
    cmd+=(--expected-target-scale "${expected_target_scale}")
  fi
  if [[ "${allow_legacy}" == "1" ]]; then
    cmd+=(--allow-legacy)
  fi
  if [[ "${expected_source_kind}" == "external_state" || "${ALLOW_EXTERNAL_STATE_TREE_BUNDLE:-0}" == "1" ]]; then
    cmd+=(--allow-external-state)
  fi
  echo "=== $(date -u) :: auditing TreeBundle ${bundle} expected_source_kind=${expected_source_kind} ==="
  "${cmd[@]}"
}
