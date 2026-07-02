#!/usr/bin/env bash
# Overnight chain: after the dgemma LLM arm finishes, expand the Benoit chunk-FNO
# recipe from economic to all 6 dims.
#
#   Phase A: wait for dgemma LLM-arm (FULL218) to finish, then stop the dgemma fleet.
#   Phase B: bring up gemma4 fleet (8010-8013), score leaf+merge spans for the 5
#            missing dims (social/immigration/eu/environment/decentralization).
#   Phase C: relabel a chunkgrid per dim (llm_span merges).
#   Phase D: stop gemma4 fleet; per-dim FNO ladder with the HPO-winning econ recipe,
#            3 seeds each, on the 4 GPUs.
#
# Self-driving; launched under long_job.py so it survives session close.
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

DIMS=(social immigration eu environment decentralization)
SRC_GRID=outputs/benoit_qsentence_grid_full
SCORE_DIR=outputs/benoit_chunk_scores_forced
EXPERT=outputs/benoit_qsentence_targets/expert_means_raw.json
LEAF=16
# HPO-winning econ recipe (outputs/hpo_econ_chunkfno/best.json)
FNO_ARGS=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.886
  --fno-root-weight 3.076)
SEEDS=(101 202 303)

# ---------- Phase A: wait for LLM arm, free dgemma GPUs ----------
LOG "Phase A: waiting for dgemma FULL218 LLM arm to finish..."
while systemctl --user is-active --quiet codex-long-job-*full218.service 2>/dev/null; do sleep 120; done
LOG "LLM arm finished. Stopping dgemma fleet (8004-8007)."
for i in 0 1 2 3; do
  $PY scripts/long_job.py stop --job-root outputs/diffusiongemma_qsentence_worker_gpu$i >/dev/null 2>&1 || true
done
sleep 20

# ---------- Phase B: gemma4 fleet + merge-span scoring (5 dims) ----------
LOG "Phase B: launching gemma4 fleet (8010-8013)."
for i in 0 1 2 3; do
  port=$((8010 + i))
  $PY scripts/long_job.py launch \
    --name "gemma4_score_gpu${i}" \
    --description "gemma4 scorer GPU ${i} port ${port}" \
    --job-root outputs/gemma4_score_gpu${i} --cwd "$PWD" --replace-existing \
    -- bash -lc "cd $PWD && ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port ${port} --cuda-devices ${i}" >/dev/null 2>&1
done
LOG "Waiting for gemma4 servers..."
for n in $(seq 1 40); do
  up=0; for p in 8010 8011 8012 8013; do curl -s -m 2 http://localhost:$p/v1/models 2>/dev/null | grep -qi gemma && up=$((up+1)); done
  LOG "  gemma4 $up/4 up"; [ "$up" -eq 4 ] && break; sleep 30
done

API=http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1
for dim in "${DIMS[@]}"; do
  LOG "Phase B: scoring LEAVES for $dim"
  $PY scripts/score_benoit_chunks.py --grid-dir "$SRC_GRID" --leaf $LEAF \
    --dimensions "$dim" --api-base "$API" --force-score --node-levels leaves \
    --output "$SCORE_DIR" > outputs/score_${dim}_leaves.log 2>&1
  LOG "Phase B: scoring MERGES for $dim"
  $PY scripts/score_benoit_chunks.py --grid-dir "$SRC_GRID" --leaf $LEAF \
    --dimensions "$dim" --api-base "$API" --force-score --node-levels merges \
    --output "$SCORE_DIR" > outputs/score_${dim}_merges.log 2>&1
done

# ---------- Phase C: relabel chunkgrids ----------
for dim in "${DIMS[@]}"; do
  LOG "Phase C: relabel chunkgrid for $dim (llm_span)"
  $PY scripts/relabel_benoit_grid_with_chunks.py \
    --src-grid "$SRC_GRID" --leaf $LEAF --dim "$dim" \
    --chunk-scores "$SCORE_DIR/leafq016_${dim}.json" \
    --merge-scores "$SCORE_DIR/leafq016_${dim}_merges.json" \
    --expert-targets "$EXPERT" --merge-supervision llm_span \
    --output-dir outputs/benoit_chunkgrid_forced_${dim}_llmspan \
    > outputs/relabel_${dim}.log 2>&1
done

# ---------- Phase D: stop gemma4, per-dim FNO ladders (3 seeds each) ----------
LOG "Phase D: stopping gemma4 fleet, starting FNO ladders."
for i in 0 1 2 3; do
  $PY scripts/long_job.py stop --job-root outputs/gemma4_score_gpu$i >/dev/null 2>&1 || true
done
sleep 20

for dim in "${DIMS[@]}"; do
  GRID=outputs/benoit_chunkgrid_forced_${dim}_llmspan
  [ -d "$GRID" ] || { LOG "  SKIP $dim (no grid — scoring/relabel failed)"; continue; }
  for idx in 0 1 2; do
    s=${SEEDS[$idx]}; g=$idx   # 3 seeds across GPUs 0,1,2 in parallel; GPU3 idle spare
    out=outputs/benoit_6dim_fno/${dim}/seed_${s}
    mkdir -p "$(dirname "$out")"
    CUDA_VISIBLE_DEVICES=$g TT_EXPORT_FULL_TREE_TRACES=0 \
      $PY scripts/run_manifesto_qsentence_dspy_ladder.py \
      --family fno --embedding-backend local-hf \
      --embedding-model /mnt/data/models/google/embeddinggemma-300m \
      --embedding-device cuda --embedding-batch-size 64 --fno-device cuda \
      --fg-grid-dir "$GRID" --leaf-qsentences 16 --max-iterations 2 \
      --fno-target-dimension "$dim" --eval-split test "${FNO_ARGS[@]}" \
      --fno-seed "$s" --output-dir "$out" > "${out}.log" 2>&1 &
  done
  wait
  LOG "Phase D: $dim done (3 seeds)"
done

LOG "CHAIN_COMPLETE — all 5 dims scored + FNO'd (econ already done)."
