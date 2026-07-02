#!/usr/bin/env bash
# Seed replication of the winning corner: does the corrected merge BEAT f-only
# robustly, and is the gamma-down trend real? Lambda=0.5 fixed; gamma in {0.5,0.25,0.1}
# x seeds {42,7,123}; plus root_only (Lambda=0) per seed as the reference. Reports the
# (merge_it2 - f_only) gap -- positive = the learned merge beats reading the doc
# through f directly (the handoff's open problem).
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${SWEEP_ROOT:-outputs/fno_benoit_econ_gamma_seed_replication}
mkdir -p "$OUT"; LOG="$OUT/runner.log"
EPOCHS=${FNO_EPOCHS:-8}; DIM=economic; LEAF=16
GRID=outputs/benoit_llmseg_economic_none
CACHE=${EMBED_CACHE:-outputs/fno_benoit_econ_law_ab_20260625_065231/embed_cache}
PY=./venv/bin/python; GPUS=(0 1 2 3)
log() { echo "[gl-rep $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

ARMS=()
for seed in 42 7 123; do
  ARMS+=("root_s${seed}:0.0:1.0:${seed}")
  for gam in 0.5 0.25 0.1; do
    ARMS+=("lam50_g${gam/./}_s${seed}:0.5:${gam}:${seed}")
  done
done

run_arm() {  # name lambda gamma seed gpu
  local name="$1" lam="$2" gam="$3" seed="$4" gpu="$5"
  local ddir="$OUT/$name"
  if [ -f "$ddir/fno/leafq0${LEAF}/iteration_history.json" ]; then log "skip $name"; return 0; fi
  log "arm=$name lam=$lam gam=$gam seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 128 --embedding-cache-dir "$CACHE" \
    --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
    --max-iterations 2 --fno-epochs "$EPOCHS" --fno-batch-size 16 \
    --fno-learning-rate 3e-3 --fno-merge-mode mlp --fno-target-dimension "$DIM" \
    --fno-local-law-weight "$lam" --fno-gamma-depth "$gam" --fno-seed "$seed" \
    --output-dir "$ddir" --verbose >> "$ddir.train.log" 2>&1 \
    || { log "WARN $name FAILED"; return 1; }
  log "arm=$name done"
}

log "start: ${#ARMS[@]} arms epochs=$EPOCHS"
# prewarm cache with first arm, then fan
IFS=: read -r n l g s <<< "${ARMS[0]}"; run_arm "$n" "$l" "$g" "$s" "${GPUS[0]}"
i=0
for spec in "${ARMS[@]:1}"; do
  IFS=: read -r n l g s <<< "$spec"
  run_arm "$n" "$l" "$g" "$s" "${GPUS[$(( i % 4 ))]}" &
  i=$((i+1)); (( i % 4 == 0 )) && wait
done
wait
log "all done"

$PY - "$OUT" "$LEAF" <<'PYEOF' | tee -a "$LOG"
import json, os, sys
out, leaf = sys.argv[1], int(sys.argv[2])
def load(name):
    for pat in (f"leafq0{leaf}", f"leafq{leaf}"):
        p=os.path.join(out,name,"fno",pat,"iteration_history.json")
        if os.path.exists(p): return json.load(open(p))["iterations"]
    return None
def fin(its,k):
    t=its[-1]["split_metrics"].get("test",{}); return t.get(k)
import statistics as st
print("%-10s %-22s %-22s %-10s"%("config","merge_it2 (mean+/-sd)","f_only (mean+/-sd)","gap"))
for lam,gam,tag in [(0.0,1.0,"root_only"),(0.5,0.5,"lam50_g0.5"),(0.5,0.25,"lam50_g0.25"),(0.5,0.1,"lam50_g0.1")]:
    ms,fs=[],[]
    for seed in (42,7,123):
        name = f"root_s{seed}" if tag=="root_only" else f"lam50_g{str(gam).replace('.','')}_s{seed}"
        its=load(name)
        if not its: continue
        m=fin(its,"internal_f_pearson"); f=fin(its,"f_only_internal_pearson")
        if isinstance(m,(int,float)): ms.append(m)
        if isinstance(f,(int,float)): fs.append(f)
    def ms_(xs): return ("%.3f+/-%.3f"%(st.mean(xs), st.pstdev(xs))) if xs else "na"
    gap = ("%+.3f"%(st.mean(ms)-st.mean(fs))) if ms and fs else "na"
    print("%-10s %-22s %-22s %-10s"%(tag, ms_(ms), ms_(fs), gap))
PYEOF
