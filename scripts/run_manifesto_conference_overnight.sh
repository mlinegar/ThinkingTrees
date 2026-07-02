#!/usr/bin/env bash
# Conference comparison queue for manifesto qsentence and full-document axes.
#
# The script is intentionally sequential around model servers: qsentence LLM
# runs need short-context fleets, full-doc runs need one long-context worker,
# and FNO sweeps run after LLM servers are stopped to free GPUs.

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/mlinegar/ThinkingTrees}
cd "$REPO_ROOT" || exit 2

PY=${PY:-./venv/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-outputs/manifesto_conference_overnight_${STAMP}}
mkdir -p "$OUT_ROOT"/{logs,reports}

FULL_GRID=${FULL_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid}
SMOKE_GRID=${SMOKE_GRID:-outputs/manifesto_qsentence_dspy_labeled_grid_smoke}
BENOIT_GRID=${BENOIT_GRID:-$OUT_ROOT/benoit_qsentence_grid}

DGEMMA_MODEL=${DGEMMA_MODEL:-openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}
DGEMMA_TOKENIZER=${DGEMMA_TOKENIZER:-/mnt/data/models/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4}
GEMMA4_MODEL=${GEMMA4_MODEL:-openai/nvidia/Gemma-4-31B-IT-NVFP4}
EMBED_MODEL=${EMBED_MODEL:-/mnt/data/models/google/embeddinggemma-300m}

FULL_DOC_SPLIT_DIR=${FULL_DOC_SPLIT_DIR:-outputs/manifesto_full_doc_gemma4_256k_20260428_225823/coverage_split_benoit_full_docs_20260428_232048}
FULL_DOC_INITIAL_PROGRAM=${FULL_DOC_INITIAL_PROGRAM:-outputs/manifesto_full_doc_gemma4_256k_20260428_225823/global_f_default_plainlm_doc150k_maxtok32_live_20260429_051700/program/dspy_program}

RUN_LLM_QSENTENCE=${RUN_LLM_QSENTENCE:-1}
RUN_FNO_CMP=${RUN_FNO_CMP:-1}
RUN_BENOIT_QSENTENCE=${RUN_BENOIT_QSENTENCE:-1}
RUN_FULLDOC_DGEMMA=${RUN_FULLDOC_DGEMMA:-1}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT_ROOT/logs/queue.log"
}

run_step() {
  local name=$1
  shift
  local log_file="$OUT_ROOT/logs/${name}.log"
  log "START $name"
  "$@" >"$log_file" 2>&1
  local status=$?
  log "END $name status=$status log=$log_file"
  return "$status"
}

wait_for_server() {
  local url=$1
  local seconds=${2:-900}
  local deadline=$((SECONDS + seconds))
  until curl -fsS "$url/models" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      log "WARN server did not become ready: $url"
      return 1
    fi
    sleep 10
  done
  log "READY $url"
}

stop_job_root() {
  local root=$1
  if [[ -f "$root/manifest.json" ]]; then
    "$PY" scripts/long_job.py stop --job-root "$root" >/dev/null 2>&1 || true
  fi
}

stop_model_servers() {
  ./scripts/stop_small_servers.sh --all >/dev/null 2>&1 || true
  for i in 0 1 2 3; do
    stop_job_root "outputs/diffusiongemma_qsentence_worker_gpu${i}"
    stop_job_root "outputs/gemma4_qsentence_server_gpu${i}_launcher"
  done
  for root in "$OUT_ROOT"/servers/*; do
    [[ -d "$root" ]] || continue
    stop_job_root "$root"
  done
  sleep 8
}

start_dgemma_short_fleet() {
  mkdir -p "$OUT_ROOT/servers"
  log "Starting DiffusionGemma short-context fleet on ports 8004-8007"
  for i in 0 1 2 3; do
    local port=$((8004 + i))
    ./scripts/start_diffusiongemma_qsentence_worker.sh "$i" "$port" 16 0.75 8192 32768 >/dev/null
  done
  for port in 8004 8005 8006 8007; do
    wait_for_server "http://localhost:${port}/v1" 1200 || return 1
  done
}

start_gemma4_fleet() {
  mkdir -p "$OUT_ROOT/servers"
  log "Starting Gemma4 short-context fleet on ports 8010-8013"
  for i in 0 1 2 3; do
    local port=$((8010 + i))
    local root="$OUT_ROOT/servers/gemma4_${i}"
    stop_job_root "$root"
    "$PY" scripts/long_job.py launch \
      --name "conf_gemma4_${i}" \
      --job-root "$root" \
      --cwd "$REPO_ROOT" \
      --replace-existing \
      -- ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port "$port" --cuda-devices "$i" \
      >/dev/null
  done
  for port in 8010 8011 8012 8013; do
    wait_for_server "http://localhost:${port}/v1" 1500 || return 1
  done
}

start_dgemma_long_worker() {
  mkdir -p "$OUT_ROOT/servers"
  log "Starting DiffusionGemma long-context worker on port 8004"
  ./scripts/start_diffusiongemma_qsentence_worker.sh 0 8004 4 0.85 16384 262144 >/dev/null
  wait_for_server "http://localhost:8004/v1" 1800
}

queue_existing_reports() {
  run_step existing_substrate_report \
    "$PY" scripts/compare_manifesto_qsentence_substrates.py \
      dgemma_leaf1=outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
      dgemma_grid=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
      fno_leaf1=outputs/manifesto_qsentence_fno_embeddinggemma_full \
      fno_grid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
      --output-dir "$OUT_ROOT/reports/existing_substrate_latest" \
    || true

  run_step existing_perdim_report \
    "$PY" scripts/summarize_manifesto_qsentence_per_dimension.py \
      dgemma=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
      fno=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
      gemma4_smoke=outputs/manifesto_qsentence_gemma4_small \
      --output-dir "$OUT_ROOT/reports/existing_per_dimension_latest" \
    || true
}

run_llm_qsentence_axes() {
  [[ "$RUN_LLM_QSENTENCE" == "1" ]] || return 0

  stop_model_servers
  start_dgemma_short_fleet || return 1
  TT_DSPY_DROP_RESPONSE_FORMAT=1 TT_SKIP_FULL_TREE_TRACES=1 run_step dgemma_full_leafgrid \
    "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
      --fg-grid-dir "$FULL_GRID" \
      --output-dir "$OUT_ROOT/dgemma_full_leafgrid" \
      --leaf-qsentences "16,8,4,2" \
      --max-iterations 2 \
      --target-dimensions all \
      --dspy-optimizer gepa \
      --dspy-budget light \
      --dspy-max-train-records 2048 \
      --dspy-model "$DGEMMA_MODEL" \
      --dspy-api-base "http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1" \
      --dspy-num-threads 640 \
      --dspy-batch-max-concurrent 1024 \
      --dspy-batch-size 640 \
      --dspy-lm-context-tokens 32768 \
      --verbose \
    || true

  stop_model_servers
  start_gemma4_fleet || return 1
  TT_DSPY_DROP_RESPONSE_FORMAT=0 TT_SKIP_FULL_TREE_TRACES=1 run_step gemma4_full_coarse \
    "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
      --fg-grid-dir "$FULL_GRID" \
      --output-dir "$OUT_ROOT/gemma4_full_coarse" \
      --leaf-qsentences "16,8,4,2" \
      --max-iterations 2 \
      --target-dimensions all \
      --dspy-optimizer gepa \
      --dspy-budget light \
      --dspy-max-train-records 2048 \
      --dspy-model "$GEMMA4_MODEL" \
      --dspy-api-base "http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1" \
      --dspy-num-threads 384 \
      --dspy-batch-max-concurrent 768 \
      --dspy-batch-size 384 \
      --dspy-lm-context-tokens 32768 \
      --verbose \
    || true

  stop_model_servers
}

run_fno_dim_batch() {
  local root=$1
  local grid=$2
  local leaf_qsentences=$3
  shift 3
  local dims=("$@")
  local pids=()
  local names=()
  local i=0

  mkdir -p "$root"
  for dim in "${dims[@]}"; do
    local gpu=$((i % 4))
    local out="$root/$dim"
    local log_file="$OUT_ROOT/logs/fno_${root##*/}_${dim}.log"
    mkdir -p "$out"
    log "START fno ${root##*/} dim=$dim gpu=$gpu"
    (
      CUDA_VISIBLE_DEVICES="$gpu" TT_SKIP_FULL_TREE_TRACES=1 TT_EXPORT_FULL_TREE_TRACES=0 \
        "$PY" scripts/run_manifesto_qsentence_dspy_ladder.py \
          --family fno \
          --fg-grid-dir "$grid" \
          --output-dir "$out" \
          --leaf-qsentences "$leaf_qsentences" \
          --max-iterations 2 \
          --fno-target-dimension "$dim" \
          --embedding-backend local-hf \
          --embedding-model "$EMBED_MODEL" \
          --embedding-device cuda \
          --fno-device cuda \
          --fno-epochs 8 \
          --fno-batch-size 16 \
          --fno-learning-rate 0.003 \
          --fno-root-weight 1.0 \
          --fno-leaf-weight 0.25 \
          --fno-merge-weight 1.0 \
          --fno-merge-mode gated
    ) >"$log_file" 2>&1 &
    pids+=("$!")
    names+=("$dim")
    i=$((i + 1))

    if (( ${#pids[@]} == 4 )); then
      local j
      for j in "${!pids[@]}"; do
        if wait "${pids[$j]}"; then
          log "END fno ${root##*/} dim=${names[$j]} status=0 log=$OUT_ROOT/logs/fno_${root##*/}_${names[$j]}.log"
        else
          log "END fno ${root##*/} dim=${names[$j]} status=$? log=$OUT_ROOT/logs/fno_${root##*/}_${names[$j]}.log"
        fi
      done
      pids=()
      names=()
    fi
  done

  local j
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then
      log "END fno ${root##*/} dim=${names[$j]} status=0 log=$OUT_ROOT/logs/fno_${root##*/}_${names[$j]}.log"
    else
      log "END fno ${root##*/} dim=${names[$j]} status=$? log=$OUT_ROOT/logs/fno_${root##*/}_${names[$j]}.log"
    fi
  done
}

run_fno_cmp_axes() {
  [[ "$RUN_FNO_CMP" == "1" ]] || return 0
  stop_model_servers
  run_fno_dim_batch "$OUT_ROOT/fno_cmp_per_dimension" "$FULL_GRID" "16,8,4,2" \
    rile domain_4 domain_5 domain_6 domain_3 domain_7 domain_1 domain_2
}

run_benoit_qsentence_axes() {
  [[ "$RUN_BENOIT_QSENTENCE" == "1" ]] || return 0
  stop_model_servers

  run_step build_benoit_qsentence_grid \
    "$PY" scripts/build_manifesto_qsentence_benoit_grid.py \
      --output-dir "$BENOIT_GRID" \
      --leaf-qsentences "1,8,16" \
    || return 1

  run_fno_dim_batch "$OUT_ROOT/benoit_fno_qsentence" "$BENOIT_GRID" "1,8,16" \
    economic social immigration eu environment decentralization
}

run_full_doc_dgemma_axis() {
  [[ "$RUN_FULLDOC_DGEMMA" == "1" ]] || return 0
  stop_model_servers
  start_dgemma_long_worker || return 1

  TT_DSPY_DROP_RESPONSE_FORMAT=1 run_step full_doc_dgemma_global_f \
    "$PY" scripts/run_manifesto_full_doc_dspy_global_f.py \
      --split-dir "$FULL_DOC_SPLIT_DIR" \
      --output-dir "$OUT_ROOT/full_doc_dgemma_global_f" \
      --base-url http://localhost:8004/v1 \
      --model RedHatAI/diffusiongemma-26B-A4B-it-NVFP4 \
      --train-docs 50 \
      --val-docs 30 \
      --test-docs 30 \
      --optimizer mipro \
      --dspy-budget light \
      --initial-program-dir "$FULL_DOC_INITIAL_PROGRAM" \
      --mipro-num-trials 4 \
      --mipro-minibatch-size 16 \
      --mipro-minibatch-full-eval-steps 2 \
      --max-bootstrapped-demos 0 \
      --max-labeled-demos 0 \
      --mipro-skip-bootstrap \
      --no-mipro-fewshot-aware-proposer \
      --no-mipro-data-aware-proposer \
      --mipro-view-data-batch-size 0 \
      --mipro-prompt-max-tokens 2048 \
      --mipro-prompt-temperature 0.7 \
      --dspy-num-threads 2 \
      --eval-num-threads 2 \
      --no-use-batched-lm \
      --train-max-input-tokens 150000 \
      --val-max-input-tokens 150000 \
      --test-max-input-tokens 150000 \
      --tokenizer-model "$DGEMMA_TOKENIZER" \
      --token-cache-dir "$OUT_ROOT/token_cache_dgemma_150k" \
      --max-tokens 32 \
      --timeout-seconds 900 \
      --min-doc-chars 2000 \
      --disable-dspy-cache \
    || true

  stop_model_servers
}

write_final_reports() {
  run_step new_substrate_report \
    "$PY" scripts/compare_manifesto_qsentence_substrates.py \
      dgemma_existing_leaf1=outputs/manifesto_qsentence_diffusiongemma_full_leaf1 \
      dgemma_new_leafgrid="$OUT_ROOT/dgemma_full_leafgrid" \
      gemma4_new_coarse="$OUT_ROOT/gemma4_full_coarse" \
      fno_existing_leaf1=outputs/manifesto_qsentence_fno_embeddinggemma_full \
      fno_existing_leafgrid=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
      --output-dir "$OUT_ROOT/reports/new_substrate_report" \
    || true

  run_step new_perdim_report \
    "$PY" scripts/summarize_manifesto_qsentence_per_dimension.py \
      dgemma_existing=outputs/manifesto_qsentence_diffusiongemma_FULL218_leafgrid \
      dgemma_new="$OUT_ROOT/dgemma_full_leafgrid" \
      gemma4_new="$OUT_ROOT/gemma4_full_coarse" \
      fno_existing=outputs/manifesto_qsentence_fno_embeddinggemma_full_leafgrid \
      fno_new="$OUT_ROOT/fno_cmp_per_dimension" \
      --output-dir "$OUT_ROOT/reports/new_per_dimension_report" \
    || true

  {
    echo "Manifesto conference overnight queue"
    echo "started_at=$STAMP"
    echo "finished_at=$(date +%Y%m%d_%H%M%S)"
    echo
    echo "Primary reports:"
    echo "- $OUT_ROOT/reports/existing_substrate_latest/comparison.md"
    echo "- $OUT_ROOT/reports/existing_per_dimension_latest/FINAL_SUMMARY.md"
    echo "- $OUT_ROOT/reports/new_substrate_report/comparison.md"
    echo "- $OUT_ROOT/reports/new_per_dimension_report/FINAL_SUMMARY.md"
    echo
    echo "Run outputs:"
    echo "- $OUT_ROOT/dgemma_full_leafgrid"
    echo "- $OUT_ROOT/gemma4_full_coarse"
    echo "- $OUT_ROOT/fno_cmp_per_dimension"
    echo "- $OUT_ROOT/benoit_qsentence_grid"
    echo "- $OUT_ROOT/benoit_fno_qsentence"
    echo "- $OUT_ROOT/full_doc_dgemma_global_f"
  } > "$OUT_ROOT/README.txt"
  log "Wrote $OUT_ROOT/README.txt"
}

main() {
  log "Queue root: $OUT_ROOT"
  log "Run flags: LLM=$RUN_LLM_QSENTENCE FNO_CMP=$RUN_FNO_CMP BENOIT=$RUN_BENOIT_QSENTENCE FULLDOC_DGEMMA=$RUN_FULLDOC_DGEMMA"

  queue_existing_reports
  run_llm_qsentence_axes
  run_fno_cmp_axes
  run_benoit_qsentence_axes
  run_full_doc_dgemma_axis
  write_final_reports
  stop_model_servers
  log "DONE"
}

main "$@"
