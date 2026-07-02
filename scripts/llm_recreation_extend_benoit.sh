#!/usr/bin/env bash
# Extend the LLM-recreation A/B (g(f(llm qsents)) vs g(f(gold qsents))) to the 4
# remaining Benoit expert dims: social, immigration, environment, decentralization.
# (eu + economic already done in outputs/llm_recreation/ via llm_recreation_pipeline.sh.)
#
# Reuses the EXISTING LLM segmentation (outputs/benoit_llmseg_grid, 177 docs) and the
# EXISTING gold _none grids (built from outputs/benoit_chunk_scores_forced/* this run).
# So the only LLM work is scoring the llmseg grid leaves for the 4 dims; gold leaf
# scores already exist.
#
# STAGES:
#  llm  : gemma-score llmseg grid leaves for the 4 dims (force-score) + relabel _none
#  fno  : path-specific f+g on gold + llm grids, 4 dims, 3 seeds -> outputs/llm_recreation/
#  Splits are aligned to gold per-doc AFTER relabel (same fix as eu/econ: identical TEST docs).
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

DIMS=(social immigration environment decentralization)
API=http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1
MODEL="openai/nvidia/Gemma-4-31B-IT-NVFP4"
LLMSEG_GRID=outputs/benoit_llmseg_grid
declare -A RW=( [social]=10 [immigration]=10 [environment]=10 [decentralization]=10 )
# (eu used rw10, econ rw3. The 4 here are leaf-sparse/weak like eu -> rw10 default;
#  matches the supmix finding that weak dims want higher root weight.)

require_fleet() {
  local up=0
  for p in 8010 8011 8012 8013; do
    curl -s -m 3 http://localhost:$p/v1/models 2>/dev/null | grep -qi gemma && up=$((up+1))
  done
  [ "$up" -ge 1 ] || { LOG "FATAL: no gemma fleet on 8010-8013."; exit 2; }
  LOG "gemma fleet replicas up: $up/4"
}

align_splits() {  # rewrite llm grid per-doc split to match gold (identical TEST docs)
  local dim=$1
  $PY - "$dim" <<'PYEOF'
import json, sys, os, shutil
dim=sys.argv[1]
gold=f"outputs/benoit_chunkgrid_forced_{dim}_none/leafq016/labeled_trees.jsonl"
llm =f"outputs/benoit_llmseg_{dim}_none/leafq016/labeled_trees.jsonl"
sp={json.loads(l)["doc_id"]:(json.loads(l).get("metadata") or {}).get("split") for l in open(gold)}
out=[]; cnt={}
for l in open(llm):
    t=json.loads(l); s=sp.get(t["doc_id"]); t.setdefault("metadata",{})["split"]=s
    cnt[s]=cnt.get(s,0)+1; out.append(json.dumps(t))
open(llm,"w").write("\n".join(out)+"\n")
gs=f"outputs/benoit_chunkgrid_forced_{dim}_none/split_ids.json"
ls=f"outputs/benoit_llmseg_{dim}_none/split_ids.json"
if os.path.exists(gs): shutil.copy(gs, ls)
print(f"  {dim} llm splits aligned -> {cnt}")
PYEOF
}

stage_llm() {
  require_fleet
  for dim in "${DIMS[@]}"; do
    local sc=outputs/benoit_llmseg_scores/leafq016_${dim}.json
    if [ -f "$sc" ] && [ "$(stat -c%s "$sc")" -gt 100 ]; then
      LOG "score skip ($dim): $sc exists & non-empty"
    else
      LOG "Stage llm: gemma-scores llmseg leaves for $dim (force-score)"
      $PY scripts/score_benoit_chunks.py \
        --grid-dir "$LLMSEG_GRID" --leaf 16 --dimensions "$dim" \
        --model "$MODEL" --api-base "$API" --force-score \
        --output outputs/benoit_llmseg_scores > logs/llmseg_score_${dim}.log 2>&1
    fi
    LOG "relabel llmseg ($dim, merge-supervision none)"
    $PY scripts/relabel_benoit_grid_with_chunks.py \
      --src-grid "$LLMSEG_GRID" --leaf 16 --dim "$dim" \
      --chunk-scores "$sc" \
      --expert-targets outputs/benoit_qsentence_targets/expert_means_raw.json \
      --merge-supervision none \
      --output-dir outputs/benoit_llmseg_${dim}_none > logs/llmseg_relabel_${dim}.log 2>&1
    align_splits "$dim"
  done
  LOG "Stage llm done. Tear down fleet, then: bash scripts/llm_recreation_extend_benoit.sh fno"
}

ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
EMB=/mnt/data/models/google/embeddinggemma-300m
SEEDS=(101 202 303)

run_path() {  # tag grid dim seed gpu
  local tag=$1 grid=$2 dim=$3 sd=$4 g=$5
  local out=outputs/llm_recreation/${dim}/${tag}/seed_${sd}
  [ -f "$out/fno/leafq016/prediction_records/iter_02_post_eval.jsonl" ] && { LOG "  skip $out"; return; }
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
    $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf --embedding-model "$EMB" \
    --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
    --fg-grid-dir "$grid" --leaf-qsentences 16 --max-iterations 2 \
    --fno-target-dimension "$dim" --eval-split test \
    "${ARCH[@]}" --fno-root-weight "${RW[$dim]}" \
    --fno-seed "$sd" --output-dir "$out" > "${out}.log" 2>&1 &
}

stage_fno() {
  LOG "FNO: path-specific f+g, 4 dims, gold + llm, 3 seeds."
  GPU=0
  for dim in "${DIMS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      run_path gold "outputs/benoit_chunkgrid_forced_${dim}_none" "$dim" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
      run_path llm  "outputs/benoit_llmseg_${dim}_none" "$dim" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
    done
  done
  wait
  LOG "LLM_RECREATION_EXTEND_BENOIT_FNO_COMPLETE"
}

case "${1:-llm}" in
  llm) stage_llm ;;
  fno) stage_fno ;;
  *) echo "usage: $0 {llm|fno}"; exit 1 ;;
esac
