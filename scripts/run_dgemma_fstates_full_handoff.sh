#!/usr/bin/env bash
# Sequential handoff: after the Gemma-4 f_states full run finishes, tear down the
# Gemma-4 TP=4 server, bring up a single-GPU DiffusionGemma worker, and run the
# IDENTICAL sampled-supervision f_states config against dgemma.
#
# Intended to be launched (detached) once the Gemma-4 run's metrics.json exists.
set -euo pipefail

REPO=/home/mlinegar/ThinkingTrees
cd "$REPO"
PY="$REPO/venv/bin/python"

DGEMMA_MODEL="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"
DGEMMA_PORT=8004
DGEMMA_GPU=0
F_ARTIFACT="outputs/manifesto_parallel_llm_qsentence_20260622_223500_gemma4safe/gemma4_fixed_leafgrid/dspy/leafq016/iter_01_train_f/f_qsentence_dspy_iter_01"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/manifesto_qsentence_sampled_supervision_fstates_full_DGEMMA_leaf16_s16_${TS}"

log() { echo "[handoff $(date +%H:%M:%S)] $*"; }

wait_server() {
  local url="$1" timeout="${2:-1800}" waited=0
  while ! curl -fsS "$url/models" >/dev/null 2>&1; do
    sleep 10; waited=$((waited+10))
    if [ "$waited" -ge "$timeout" ]; then
      log "ERROR: server $url not up after ${timeout}s"; return 1
    fi
  done
  log "server $url is UP (after ${waited}s)"
}

# 1) Tear down the Gemma-4 TP=4 server to free all GPUs for dgemma.
#    Use the long_job stop path (NOT pkill -f, which can match unrelated jobs by
#    their command-line args and kill the wrong process).
GEMMA_SERVER_JOB_ROOT="$REPO/outputs/manifesto_leaf16_gemma4_server_standalone"
log "stopping Gemma-4 server via long_job stop ($GEMMA_SERVER_JOB_ROOT)"
"$PY" "$REPO/scripts/long_job.py" stop --job-root "$GEMMA_SERVER_JOB_ROOT" 2>/dev/null || \
  log "WARN: long_job stop returned nonzero (server may be managed differently); continuing"
# wait for GPU memory to actually free
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | head -1)
  if [ "${used:-99999}" -lt 5000 ]; then log "GPU $i has <5GB used; freed"; break; fi
  sleep 5
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

# 2) Bring up the single-GPU dgemma worker (script defaults: seqs 16, mem 0.75).
log "launching dgemma worker on GPU ${DGEMMA_GPU} port ${DGEMMA_PORT}"
bash "$REPO/scripts/start_diffusiongemma_qsentence_worker.sh" "$DGEMMA_GPU" "$DGEMMA_PORT" 16 0.75
wait_server "http://localhost:${DGEMMA_PORT}/v1" 1800

# 3) Run the identical f_states config against dgemma.
#    dgemma client tuning (matches prior runs): threads/concurrent/batch = 32/32/16,
#    response_format dropped, bare-field wrap on.
log "starting dgemma f_states full run -> $OUT"
TT_DSPY_DROP_RESPONSE_FORMAT=1 \
TT_DSPY_WRAP_BARE_FIELD_OUTPUT=1 \
TT_SKIP_FULL_TREE_TRACES=1 \
"$PY" scripts/run_manifesto_qsentence_sampled_supervision.py \
  --output-dir "$OUT" \
  --f-artifact "$F_ARTIFACT" \
  --leaf-qsentences 16 --leaf-size-tokens 512 \
  --sample-leaf-count 16 --samples-per-doc 1 \
  --train-docs 0 --eval-docs 0 \
  --max-train-examples 100000 \
  --sample-state-source f_states \
  --dspy-api-base "http://localhost:${DGEMMA_PORT}/v1" \
  --dspy-model "$DGEMMA_MODEL" \
  --dspy-num-threads 32 --f-prewarm-threads 32 \
  --dspy-batch-max-concurrent 32 --dspy-batch-size 16 \
  --dspy-budget light --dspy-max-tokens 1024 \
  --dspy-gepa-val-examples 32 --dspy-reflection-minibatch-size 1 \
  --verbose

log "dgemma f_states full run complete: $OUT"
echo "$OUT"
