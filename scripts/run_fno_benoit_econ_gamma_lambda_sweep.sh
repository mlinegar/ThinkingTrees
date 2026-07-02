#!/usr/bin/env bash
# Honest (gamma, Lambda) sweep of the CANONICAL local-law objective on the
# NON-additive Benoit economic doc label. Replaces the ill-defined "averager
# control": the reference is ROOT-ONLY (Lambda=0, pure doc-label fit at the root --
# the setting where the merge regresses to averaging). Question: does turning on the
# distributed laws (Lambda>0, gamma>0) beat root-only on doc reconstruction, now that
# the corrected merge law f*(A.B)=f*(g(A).g(B)) ties interiors to the parent-text read?
#
# Each arm logs BOTH the merge doc-reconstruction (internal_f_*) and the f-only
# baseline (f_only_internal_* = read the whole doc through f, the standing target).
set -u
ROOT=/home/mlinegar/ThinkingTrees
cd "$ROOT"
OUT=${SWEEP_ROOT:-outputs/fno_benoit_econ_gamma_lambda_sweep}
mkdir -p "$OUT"
LOG="$OUT/runner.log"
EPOCHS=${FNO_EPOCHS:-8}; DIM=economic; LEAF=16
GRID=outputs/benoit_llmseg_economic_none
CACHE=${EMBED_CACHE:-outputs/fno_benoit_econ_law_ab_20260625_065231/embed_cache}
PY=./venv/bin/python
GPUS=(0 1 2 3)
log() { echo "[gl-sweep $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# arm spec: name:lambda:gamma
ARMS=(
  "root_only:0.0:1.0"
  "lam25_g1:0.25:1.0"
  "lam50_g1:0.50:1.0"
  "lam75_g1:0.75:1.0"
  "lam50_g05:0.50:0.5"
  "lam50_g025:0.50:0.25"
  "lam100_g1:1.0:1.0"
)

run_arm() {  # <name> <lambda> <gamma> <gpu>
  local name="$1" lam="$2" gam="$3" gpu="$4"
  local ddir="$OUT/$name"
  if [ -f "$ddir/fno/leafq0${LEAF}/iteration_history.json" ]; then log "skip $name (done)"; return 0; fi
  log "arm=$name lambda=$lam gamma=$gam gpu=$gpu -> $ddir"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m \
    --embedding-device cuda --embedding-batch-size 128 \
    --embedding-cache-dir "$CACHE" \
    --fg-grid-dir "$GRID" --leaf-qsentences "$LEAF" \
    --max-iterations 2 --fno-epochs "$EPOCHS" --fno-batch-size 16 \
    --fno-learning-rate 3e-3 --fno-merge-mode mlp --fno-target-dimension "$DIM" \
    --fno-local-law-weight "$lam" --fno-gamma-depth "$gam" \
    --output-dir "$ddir" --verbose >> "$ddir.train.log" 2>&1 \
    || { log "WARN arm=$name FAILED (see $ddir.train.log)"; return 1; }
  log "arm=$name done"
}

log "start: ${#ARMS[@]} arms, epochs=$EPOCHS, grid=$GRID, cache=$CACHE"
# Prewarm the cache with the first arm (populate embeddings once), then fan the rest.
IFS=: read -r n0 l0 g0 <<< "${ARMS[0]}"; run_arm "$n0" "$l0" "$g0" "${GPUS[0]}"
i=0
for spec in "${ARMS[@]:1}"; do
  IFS=: read -r n l g <<< "$spec"
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  run_arm "$n" "$l" "$g" "$gpu" &
  i=$((i+1))
  # keep at most 4 concurrent
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
log "all arms done"

# Summary
$PY - "$OUT" "$LEAF" <<'PYEOF' | tee -a "$LOG"
import json, os, sys
out, leaf = sys.argv[1], int(sys.argv[2])
arms = ["root_only","lam25_g1","lam50_g1","lam75_g1","lam50_g05","lam50_g025","lam100_g1"]
print("%-12s %-22s %-22s" % ("arm","merge_pearson(it1/it2)","f_only_pearson(it1/it2)"))
for name in arms:
    hit=None
    for pat in (f"leafq0{leaf}", f"leafq{leaf}", f"leafq{leaf:03d}"):
        p=os.path.join(out,name,"fno",pat,"iteration_history.json")
        if os.path.exists(p): hit=p; break
    if not hit: print("%-12s n/a"%name); continue
    its=json.load(open(hit))["iterations"]
    def g(i,k):
        t=its[i]["split_metrics"].get("test",{}); v=t.get(k); return ("%.3f"%v) if isinstance(v,(int,float)) else "na"
    mp = "%s/%s" % (g(1,"internal_f_pearson"), g(-1,"internal_f_pearson"))
    fp = "%s/%s" % (g(1,"f_only_internal_pearson"), g(-1,"f_only_internal_pearson"))
    print("%-12s %-22s %-22s" % (name, mp, fp))
PYEOF
log "summary written"
