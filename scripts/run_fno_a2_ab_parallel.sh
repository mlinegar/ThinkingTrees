#!/usr/bin/env bash
# Lean A2 merge-consistency A/B, PARALLEL across GPUs, reusing a SHARED embed cache.
# Trains A2 (f(parent) == merge of child readings, through f) as the principled merge
# objective instead of node-MSE-vs-additive-target (which lets averaging win). Arms:
#   control  : node-MSE only (no A2)            -- the bar (averager)
#   a2state  : A2 with state-space merge + assoc penalty (A2-direct)
#   a2readout: A2 with the phi-form readout merge (A3-literal, assoc+comm by construction)
# All merge_mode=mlp, domain_4 leaf8. Reuses CACHE (no re-embed) if EMBED_CACHE set.
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${A2_ROOT:-outputs/fno_a2_ab}
mkdir -p "$OUT"
LOG="$OUT/runner.log"
EPOCHS=${FNO_EPOCHS:-8}; DIM=${FNO_DIM:-domain_4}; LEAF=8
GPUS=(${FNO_GPUS:-0 1 2 3})
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
CACHE=${EMBED_CACHE:-$OUT/embed_cache}
A2W=${A2_WEIGHT:-1.0}; ASSOCW=${ASSOC_WEIGHT:-0.5}
PY=./venv/bin/python
log() { echo "[a2-ab $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_arm() {  # <name> <gpu> <extra ladder flags...>
  local name="$1" gpu="$2"; shift 2
  local ddir="$OUT/$name"
  log "arm=$name gpu=$gpu -> $ddir"
  if [ ! -f "$ddir/fno/leafq00${LEAF}/iteration_history.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
      --family fno --embedding-backend local-hf \
      --embedding-model /mnt/data/models/google/embeddinggemma-300m \
      --embedding-device cuda --embedding-batch-size 128 \
      --embedding-cache-dir "$CACHE" \
      --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
      --max-iterations 2 --fno-epochs "$EPOCHS" --fno-batch-size 16 \
      --fno-learning-rate 3e-3 --fno-merge-mode mlp \
      --fno-target-dimension "$DIM" "$@" \
      --output-dir "$ddir" --verbose >> "$ddir.train.log" 2>&1 \
      || { log "WARN arm=$name TRAIN FAILED"; return 1; }
  fi
  local states="$ddir/g_node_states_${DIM}_leaf${LEAF}.jsonl"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/dump_fno_g_node_states.py \
    --run-dir "$ddir/fno" --leaf-qsentences "$LEAF" --fg-grid-dir "$GRID" \
    --target-dimension "$DIM" \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-cache-dir "$CACHE" \
    --fno-merge-mode mlp --out-jsonl "$states" >> "$ddir.score.log" 2>&1 \
    || { log "WARN arm=$name DUMP FAILED"; return 1; }
  $PY scripts/eval_qsentence_merge_by_level.py \
    --labeled-trees "$GRID/leafq00${LEAF}/labeled_trees.jsonl" --split test \
    --g-states-jsonl "$states" --lopsidedness-strength 4.0 \
    --out-json "$ddir/merge_by_level_${DIM}.json" >> "$ddir.score.log" 2>&1 \
    && log "arm=$name scored" || log "WARN arm=$name SCORE FAILED"
}

log "start: dim=$DIM leaf=$LEAF a2w=$A2W assocw=$ASSOCW cache=$CACHE gpus=(${GPUS[*]})"
# If the cache is cold, prewarm with control on one GPU first; else fan all 3.
if [ ! -d "$CACHE" ] || [ -z "$(ls -A "$CACHE" 2>/dev/null)" ]; then
  log "cache cold -> prewarm with control"
  run_arm control "${GPUS[0]}" --fno-g-a2-weight 0
  run_arm a2state "${GPUS[1%${#GPUS[@]}]}" --fno-g-a2-weight "$A2W" --fno-a2-mode state --fno-g-assoc-weight "$ASSOCW" &
  run_arm a2readout "${GPUS[2%${#GPUS[@]}]}" --fno-g-a2-weight "$A2W" --fno-a2-mode readout &
  wait
else
  log "cache warm -> fan all 3 arms"
  run_arm control   "${GPUS[0]}" --fno-g-a2-weight 0 &
  run_arm a2state   "${GPUS[1%${#GPUS[@]}]}" --fno-g-a2-weight "$A2W" --fno-a2-mode state --fno-g-assoc-weight "$ASSOCW" &
  run_arm a2readout "${GPUS[2%${#GPUS[@]}]}" --fno-g-a2-weight "$A2W" --fno-a2-mode readout &
  wait
fi
log "all arms complete"

log "=== SUMMARY (per-node merge wmae; bar=equal_avg 0.00406, ceiling=0) ==="
$PY - "$OUT" "$DIM" <<'PYEOF' | tee -a "$LOG"
import json, sys, glob, os
out, dim = sys.argv[1], sys.argv[2]
for name in ("control","a2state","a2readout"):
    f=os.path.join(out,name,"merge_by_level_%s.json"%dim)
    try:
        d=json.load(open(f)); p=d.get("pooled_weighted",{})
        print("  %-10s learned_g wmae=%s"%(name, "%.5f"%p.get("learned_g_wmae")))
    except Exception as e:
        print("  %-10s n/a (%s)"%(name, e))
PYEOF
