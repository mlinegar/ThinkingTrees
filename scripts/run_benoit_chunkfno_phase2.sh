#!/usr/bin/env bash
# Phase 2: clear the LLM servers (free all 4 GPUs), relabel every (variant, dim)
# grid with the LLM chunk scores, and train all 12 chunk-FNO ladders in parallel
# round-robin across the 4 GPUs (GPU embedding; ~97GB free per GPU once servers
# are down). Run AFTER Phase 1 (chunk scoring) completes.
set -u
cd /home/mlinegar/ThinkingTrees
DIMS="economic social immigration eu environment decentralization"

echo "[phase2 $(date +%H:%M:%S)] clearing LLM servers (free all GPUs)"
pkill -f "vllm serve.*diffusiongemma" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 25

echo "[phase2 $(date +%H:%M:%S)] relabeling grids"
for v in na forced; do
  for d in $DIMS; do
    f="outputs/benoit_chunk_scores_${v}/leafq016_${d}.json"
    [ -f "$f" ] || { echo "  skip $v/$d (no scores)"; continue; }
    ./venv/bin/python scripts/relabel_benoit_grid_with_chunks.py \
      --chunk-scores "$f" --dim "$d" \
      --output-dir "outputs/benoit_chunkgrid_${v}_${d}" >/dev/null 2>&1 \
      && echo "  relabeled $v/$d" || echo "  WARN relabel failed $v/$d"
  done
done

echo "[phase2 $(date +%H:%M:%S)] launching 12 chunk-FNO in parallel across 4 GPUs"
gpu=0
pids=()
for v in na forced; do
  for d in $DIMS; do
    [ -d "outputs/benoit_chunkgrid_${v}_${d}/leafq016" ] || continue
    out="outputs/benoit_chunkfno_${v}_${d}"
    rm -rf "$out"
    CUDA_VISIBLE_DEVICES=$gpu ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
      --family fno --embedding-backend local-hf \
      --embedding-model /mnt/data/models/google/embeddinggemma-300m \
      --embedding-device cuda --embedding-batch-size 64 \
      --fg-grid-dir "outputs/benoit_chunkgrid_${v}_${d}" \
      --leaf-qsentences "16" --max-iterations 2 --fno-epochs 8 \
      --fno-batch-size 16 --fno-learning-rate 3e-3 --fno-leaf-weight 1.0 --fno-merge-weight 0.2 --fno-root-weight 20.0 --fno-target-dimension "$d" \
      --output-dir "$out" > "${out}.log" 2>&1 &
    pids+=($!)
    echo "  launched $v/$d on GPU$gpu (pid $!)"
    gpu=$(( (gpu + 1) % 4 ))
    sleep 3
  done
done
echo "[phase2 $(date +%H:%M:%S)] waiting on ${#pids[@]} FNO jobs"
wait
echo "[phase2 $(date +%H:%M:%S)] all chunk-FNO done"
