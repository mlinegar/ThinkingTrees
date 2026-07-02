#!/usr/bin/env bash
# LLM-recreation experiment: can the LLM recreate the doc dimension from its OWN
# q-sentences (no gold human segmentation), with g aligned via gold-as-teacher?
#
# DESIGN (user calls):
#  - segmentation: LLM splits docs at ~GOLD granularity (structurally comparable A/B)
#  - g alignment: GOLD path supervises the LLM path (gold composed-root = teacher target)
#  - f/g: PATH-SPECIFIC (each path trains its own f+g; llm path must stand alone at deploy,
#    gold only as the training target) -> the deployable pipeline
#  - dims: eu (hard) + economic (control)
#
# PRECONDITION: gemma-4-31B fleet up on 8010-8013 (the user starts it via ! command):
#   for i in 0 1 2 3; do ./scripts/start_vllm.sh gemma-4-31b-it-nvfp4 --port $((8010+i)) \
#     --cuda-devices $i > logs/gemma_fleet_$i.log 2>&1 & done
#
# STAGES:
#  1. generate_llm_qsentences.py  -> outputs/benoit_llmseg/manifesto_corpus_llmseg.csv
#  2. build_manifesto_qsentence_benoit_grid.py --corpus-csv <llmseg>  -> llmseg grid
#     then score_benoit_chunks.py (force-score) on the llmseg grid (eu, econ)
#     relabel_benoit_grid_with_chunks.py merge-supervision none -> llmseg _none grids
#  3. (gold path already built+scored this session: benoit_chunkgrid_forced_{eu,econ}_none)
#     run the ladder on the GOLD grid, capture per-doc composed-root predictions
#     -> these become the TEACHER targets for the llm path
#  4. relabel llmseg grid root targets with gold-path predictions; run the ladder on the
#     LLM grid; compare llm-path root vs gold-path root vs expert.
#
# This script runs stages 1-2 (LLM-dependent) + 3-4 (FNO). The FNO stages need the GPUs the
# fleet occupies, so the fleet must be torn down between stage 2 and stage 3 (the runner
# pauses and waits for the operator, OR pass --auto-teardown to stop the fleet jobs).
set -uo pipefail
cd /home/mlinegar/ThinkingTrees
PY=./venv/bin/python
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

DIMS=(eu economic)
API=http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1
MODEL="openai/nvidia/Gemma-4-31B-IT-NVFP4"
LLMSEG_CSV=outputs/benoit_llmseg/manifesto_corpus_llmseg.csv
LLMSEG_GRID=outputs/benoit_llmseg_grid

require_fleet() {
  local up=0
  for p in 8010 8011 8012 8013; do
    curl -s -m 3 http://localhost:$p/v1/models 2>/dev/null | grep -qi gemma && up=$((up+1))
  done
  [ "$up" -ge 1 ] || { LOG "FATAL: no gemma fleet on 8010-8013. Start it first."; exit 2; }
  LOG "gemma fleet replicas up: $up/4"
}

# ---------- Stage 1: LLM segmentation ----------
stage1() {
  require_fleet
  if [ -f "$LLMSEG_CSV" ]; then LOG "stage1 skip: $LLMSEG_CSV exists"; return; fi
  LOG "Stage 1: gemma segments docs into quasi-sentences (~gold granularity)"
  $PY scripts/generate_llm_qsentences.py \
    --model "$MODEL" --api-base "$API" --output-csv "$LLMSEG_CSV" \
    > logs/llmseg_generate.log 2>&1
  LOG "  -> $LLMSEG_CSV ($(wc -l < "$LLMSEG_CSV") rows)"
}

# ---------- Stage 2: build + score the llmseg grid ----------
stage2() {
  require_fleet
  if [ ! -d "$LLMSEG_GRID/leafq016" ]; then
    LOG "Stage 2a: build llmseg q-sentence grid from the LLM corpus CSV"
    $PY scripts/build_manifesto_qsentence_benoit_grid.py \
      --corpus-csv "$LLMSEG_CSV" --leaf-qsentences 16 \
      --output-dir "$LLMSEG_GRID" > logs/llmseg_build.log 2>&1
  else LOG "stage2a skip: $LLMSEG_GRID exists"; fi
  for dim in "${DIMS[@]}"; do
    local sc=outputs/benoit_llmseg_scores/leafq016_${dim}.json
    if [ -f "$sc" ]; then LOG "stage2b skip ($dim): $sc"; continue; fi
    LOG "Stage 2b: gemma-scores llmseg leaves for $dim (force-score)"
    $PY scripts/score_benoit_chunks.py \
      --grid-dir "$LLMSEG_GRID" --leaf 16 --dimensions "$dim" \
      --model "$MODEL" --api-base "$API" --force-score \
      --output outputs/benoit_llmseg_scores > logs/llmseg_score_${dim}.log 2>&1
    LOG "Stage 2c: relabel llmseg grid ($dim, merge-supervision none)"
    $PY scripts/relabel_benoit_grid_with_chunks.py \
      --src-grid "$LLMSEG_GRID" --leaf 16 --dim "$dim" \
      --chunk-scores "$sc" \
      --expert-targets outputs/benoit_qsentence_targets/expert_means_raw.json \
      --merge-supervision none \
      --output-dir outputs/benoit_llmseg_${dim}_none > logs/llmseg_relabel_${dim}.log 2>&1
  done
  LOG "Stage 2 done. Tear down the gemma fleet, then run: bash scripts/llm_recreation_pipeline.sh fno"
}

# ---------- Stages 3-4: FNO (needs the GPUs the fleet held) ----------
ARCH=(--fno-n-modes 384 --fno-hidden-channels 32 --fno-n-layers 2
  --fno-head-hidden-dim 256 --fno-epochs 12 --fno-learning-rate 0.00237
  --fno-weight-decay 0.0000185 --fno-leaf-weight 1.569 --fno-merge-weight 0.0)
declare -A RW=( [eu]=10 [economic]=3 )
EMB=/mnt/data/models/google/embeddinggemma-300m
SEEDS=(101 202 303)

run_path() {  # path_tag, grid, dim, seed, gpu  -> trains f+g, emits per-doc root preds
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
  LOG "Stages 3-4: PATH-SPECIFIC f+g. gold path (teacher) + llm path, eu+econ, 3 seeds."
  GPU=0
  for dim in "${DIMS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      # gold path = teacher (already-built gold _none grid)
      run_path gold "outputs/benoit_chunkgrid_forced_${dim}_none" "$dim" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
      # llm path = deployable (its own segmentation+scores). Root target stays the
      # EXPERT mean here; the gold-as-teacher comparison is computed at scoring time
      # (llm-path root vs gold-path root vs expert) so we keep paths fully independent.
      run_path llm "outputs/benoit_llmseg_${dim}_none" "$dim" "$sd" "$GPU"
      GPU=$(( (GPU+1) % 4 )); [ "$GPU" -eq 0 ] && wait
    done
  done
  wait
  LOG "LLM_RECREATION_FNO_COMPLETE"
}

case "${1:-llm}" in
  llm) stage1; stage2 ;;
  fno) stage_fno ;;
  all) stage1; stage2; stage_fno ;;
  *) echo "usage: $0 {llm|fno|all}"; exit 1 ;;
esac
