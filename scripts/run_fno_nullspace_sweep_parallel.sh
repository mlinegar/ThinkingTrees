#!/usr/bin/env bash
# FNO null-space law sweep, PARALLEL across all 4 GPUs with a SHARED disk embedding
# cache. The dominant cost (embeddinggemma over ~190 trees) is paid ONCE: a prewarm
# arm populates the cache, then every sweep point loads embeddings from disk and only
# the tiny FNO trains — so all points fan out concurrently across GPUs.
#
# Sweep: --fno-g-null-space-weight in NS_WEIGHTS (default "0 0.5 1.0 2.0 4.0"), all
# merge_mode=mlp, domain_4 leaf8. Point 0.0 == the no-law control.
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${NS_ROOT:-outputs/fno_nullspace_sweep}
mkdir -p "$OUT"
LOG="$OUT/runner.log"
EPOCHS=${FNO_EPOCHS:-8}; DIM=${FNO_DIM:-domain_4}; LEAF=8
NS_WEIGHTS=${NS_WEIGHTS:-"0 0.5 1.0 2.0 4.0"}
GPUS=(${FNO_GPUS:-0 1 2 3})
GRID=outputs/manifesto_qsentence_dspy_labeled_grid
CACHE="$OUT/embed_cache"
PY=./venv/bin/python
log() { echo "[ns-sweep $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_point() {  # <weight> <gpu>
  local w="$1" gpu="$2"
  local name="ns_${w}"
  local ddir="$OUT/$name"
  log "point w=$w gpu=$gpu -> $ddir"
  if [ ! -f "$ddir/fno/leafq00${LEAF}/iteration_history.json" ]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
      --family fno --embedding-backend local-hf \
      --embedding-model /mnt/data/models/google/embeddinggemma-300m \
      --embedding-device cuda --embedding-batch-size 128 \
      --embedding-cache-dir "$CACHE" \
      --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
      --max-iterations 2 --fno-epochs "$EPOCHS" --fno-batch-size 16 \
      --fno-learning-rate 3e-3 --fno-merge-mode mlp \
      --fno-target-dimension "$DIM" --fno-g-null-space-weight "$w" \
      --output-dir "$ddir" --verbose >> "$ddir.train.log" 2>&1 \
      || { log "WARN point w=$w TRAIN FAILED"; return 1; }
  fi
  local states="$ddir/g_node_states_${DIM}_leaf${LEAF}.jsonl"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/dump_fno_g_node_states.py \
    --run-dir "$ddir/fno" --leaf-qsentences "$LEAF" --fg-grid-dir "$GRID" \
    --target-dimension "$DIM" \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-cache-dir "$CACHE" \
    --fno-merge-mode mlp --out-jsonl "$states" >> "$ddir.score.log" 2>&1 \
    || { log "WARN point w=$w DUMP FAILED"; return 1; }
  $PY scripts/eval_qsentence_merge_by_level.py \
    --labeled-trees "$GRID/leafq00${LEAF}/labeled_trees.jsonl" --split test \
    --g-states-jsonl "$states" --lopsidedness-strength 4.0 \
    --out-json "$ddir/merge_by_level_${DIM}.json" >> "$ddir.score.log" 2>&1 \
    && log "point w=$w scored" || log "WARN point w=$w SCORE FAILED"
}

read -ra WEIGHTS <<< "$NS_WEIGHTS"
log "start: dim=$DIM leaf=$LEAF weights=(${WEIGHTS[*]}) gpus=(${GPUS[*]}) cache=$CACHE"

# Phase 1: PREWARM the shared embedding cache once (first weight, first GPU). All
# later points then hit the cache (read-only) — no redundant embedding, no write race.
log "=== prewarm (populate embedding cache) ==="
run_point "${WEIGHTS[0]}" "${GPUS[0]}"
log "=== prewarm done; fanning remaining ${#WEIGHTS[@]} points across ${#GPUS[@]} GPUs ==="

# Phase 2: fan the REMAINING points across all GPUs, round-robin, max ${#GPUS[@]} at once.
pids=()
gi=0
for ((i=1; i<${#WEIGHTS[@]}; i++)); do
  gpu=${GPUS[$((gi % ${#GPUS[@]}))]}
  run_point "${WEIGHTS[$i]}" "$gpu" &
  pids+=($!)
  gi=$((gi+1))
  # throttle to #GPUS concurrent
  if (( ${#pids[@]} >= ${#GPUS[@]} )); then
    wait "${pids[0]}"; pids=("${pids[@]:1}")
  fi
done
wait
log "all sweep points complete"

# Summarize
log "=== SUMMARY (per-node merge wmae; bar=equal_avg 0.00406, ceiling=0) ==="
$PY - "$OUT" "$DIM" <<'PYEOF' | tee -a "$LOG"
import json, sys, glob, os
out, dim = sys.argv[1], sys.argv[2]
rows=[]
for f in sorted(glob.glob(os.path.join(out,"ns_*","merge_by_level_%s.json"%dim))):
    w=os.path.basename(os.path.dirname(f)).replace("ns_","")
    try:
        d=json.load(open(f)); p=d.get("pooled_weighted",{})
        rows.append((float(w), p.get("learned_g_wmae")))
    except Exception as e:
        rows.append((float(w), None))
for w,v in sorted(rows):
    print("  null_w=%-5s learned_g wmae=%s"%(w, ("%.5f"%v) if v is not None else "n/a"))
PYEOF
