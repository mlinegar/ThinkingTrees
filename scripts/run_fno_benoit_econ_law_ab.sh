#!/usr/bin/env bash
# Through-f law A/B on the NON-ADDITIVE Benoit economic doc label.
#
# This is the decisive test of the two laws. Grid = outputs/benoit_llmseg_economic_none
# (leaves = real per-chunk LLM econ scores [Law 1: g(leaf)~gold]; ROOT = the Benoit
# EXPERT economic mean [Law 2: g(X)~X], r=1.0 vs expert across 177 docs). Unlike
# domain_4 (a count ratio = additive), the expert mean is NOT a leaf-mean rollup
# (CV R^2 0.40 from CMP codes), so Law 2 CANNOT be satisfied by averaging — a real
# non-additive merge is required.
#
# 3 arms, all merge_mode=mlp, dim=economic, leaf16:
#   control   : node-MSE only (averager bar)
#   a2state   : + A2 state-merge consistency + assoc penalty
#   a2readout : + A2 via the phi-form readout merge (A3-literal, assoc+comm)
# Score: doc reconstruction Pearson vs expert (baseline chunk-FNO 0.66-0.73) + per-node
# merge. Win = an A2 arm's doc Pearson and/or merge beats control where domain_4 couldn't.
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${ECON_ROOT:-outputs/fno_benoit_econ_law_ab}
mkdir -p "$OUT"
LOG="$OUT/runner.log"
EPOCHS=${FNO_EPOCHS:-8}; DIM=economic; LEAF=16
GPUS=(${FNO_GPUS:-0 1 2 3})
GRID=outputs/benoit_llmseg_economic_none
CACHE="$OUT/embed_cache"
A2W=${A2_WEIGHT:-1.0}; ASSOCW=${ASSOC_WEIGHT:-0.5}
PY=./venv/bin/python
log() { echo "[econ-law $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_arm() {  # <name> <gpu> <extra flags...>
  local name="$1" gpu="$2"; shift 2
  local ddir="$OUT/$name"
  log "arm=$name gpu=$gpu -> $ddir"
  if [ ! -f "$ddir/fno/leafq0${LEAF}/iteration_history.json" ] && [ ! -f "$ddir/fno/leafq${LEAF}/iteration_history.json" ]; then
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
      || { log "WARN arm=$name TRAIN FAILED (see $ddir.train.log)"; return 1; }
  fi
  log "arm=$name trained"
}

log "start: dim=$DIM leaf=$LEAF a2w=$A2W grid=$GRID cache=$CACHE gpus=(${GPUS[*]})"
# Prewarm the cache with control (one GPU), then fan the two A2 arms.
run_arm control "${GPUS[0]}" --fno-g-a2-weight 0
run_arm a2state "${GPUS[1%${#GPUS[@]}]}" --fno-g-a2-weight "$A2W" --fno-a2-mode state --fno-g-assoc-weight "$ASSOCW" &
run_arm a2readout "${GPUS[2%${#GPUS[@]}]}" --fno-g-a2-weight "$A2W" --fno-a2-mode readout &
wait
log "all arms trained"

log "=== SUMMARY (doc reconstruction = Law 2 g(X)~X; baseline chunk-FNO econ 0.66-0.73) ==="
$PY - "$OUT" "$LEAF" <<'PYEOF' | tee -a "$LOG"
import json, sys, glob, os
out, leaf = sys.argv[1], int(sys.argv[2])
for name in ("control","a2state","a2readout"):
    hit=None
    for pat in (f"leafq0{leaf}", f"leafq{leaf}", f"leafq{leaf:03d}"):
        f=os.path.join(out,name,"fno",pat,"iteration_history.json")
        if os.path.exists(f): hit=f; break
    if not hit:
        print("  %-10s n/a (no iteration_history)"%name); continue
    d=json.load(open(hit)); it=d["iterations"][-1]["split_metrics"]["test"]
    print("  %-10s doc_MAE=%.4f  doc_Pearson=%.4f"%(name, it.get("internal_f_mae",0), it.get("internal_f_pearson",0)))
PYEOF
