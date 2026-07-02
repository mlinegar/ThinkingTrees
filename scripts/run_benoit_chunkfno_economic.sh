#!/usr/bin/env bash
set -u
cd /home/mlinegar/ThinkingTrees
for v in na forced; do
  echo "[chunkfno $(date +%H:%M:%S)] === $v economic starting ==="
  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python scripts/run_manifesto_qsentence_dspy_ladder.py \
    --family fno --embedding-backend local-hf \
    --embedding-model /mnt/data/models/google/embeddinggemma-300m --embedding-device cuda --embedding-batch-size 32 \
    --fg-grid-dir outputs/benoit_chunkgrid_${v}_economic \
    --leaf-qsentences "16" --max-iterations 2 --fno-epochs 8 --fno-batch-size 16 --fno-learning-rate 3e-3 \
    --fno-target-dimension economic \
    --output-dir outputs/benoit_chunkfno_${v}_economic >> outputs/benoit_chunkfno_${v}_economic.log 2>&1 \
    && echo "[chunkfno $(date +%H:%M:%S)] === $v done ===" || echo "[chunkfno] $v FAILED"
done
echo "[chunkfno $(date +%H:%M:%S)] both done"
